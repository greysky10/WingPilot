#! python3.12
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
from botocore.config import Config


TICKER_PATTERN = re.compile(r"^O:(?P<root>[A-Z]+)(?P<expiry>\d{6})(?P<type>[CP])(?P<strike_raw>\d{8})$")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a strategy-focused SPX options daily-history dataset from Massive flat files. "
            "This is intended for historical-chain backtests where a REST per-contract crawl is too slow."
        )
    )
    parser.add_argument("--start", default="2025-04-10", help="Inclusive trade-date start, for example 2025-04-10.")
    parser.add_argument("--end", default="2026-04-09", help="Inclusive trade-date end, for example 2026-04-09.")
    parser.add_argument(
        "--bars-csv",
        default=r".\corridor_outputs\fit_search\SPX_5_mins_bars_20250408_20260409.csv",
        help="Underlying SPX intraday bars CSV used to derive trade dates and strike windows.",
    )
    parser.add_argument(
        "--output-dir",
        default=r".\data\massive_spx_strategy_history_longdte_10_35_flatfiles",
        help="Directory for per-day parts and the final CSV dataset.",
    )
    parser.add_argument("--symbol", default="SPX", help="Underlying symbol label written into the output dataset.")
    parser.add_argument(
        "--option-roots",
        default="SPX,SPXW",
        help="Comma-separated OCC roots to keep, for example SPX,SPXW.",
    )
    parser.add_argument(
        "--contract-types",
        default="call",
        help="Comma-separated contract types to keep. Supported values: call, put.",
    )
    parser.add_argument("--dte-min", type=int, default=10, help="Minimum calendar DTE to retain.")
    parser.add_argument("--dte-max", type=int, default=35, help="Maximum calendar DTE to retain.")
    parser.add_argument(
        "--max-width-points",
        type=float,
        default=15.0,
        help="Maximum butterfly width expected in downstream searches. Used only to expand strike windows.",
    )
    parser.add_argument(
        "--strike-buffer-points",
        type=float,
        default=25.0,
        help="Extra points added above and below the strike window around the underlying close.",
    )
    parser.add_argument(
        "--center-rounding",
        type=float,
        default=5.0,
        help="Strike rounding increment used when expanding strike windows.",
    )
    parser.add_argument("--bucket", default=os.getenv("MASSIVE_S3_BUCKET", "flatfiles"), help="Massive flat-files bucket name.")
    parser.add_argument(
        "--s3-endpoint",
        default=os.getenv("MASSIVE_S3_ENDPOINT", "https://files.massive.com"),
        help="Massive flat-files S3 endpoint.",
    )
    parser.add_argument(
        "--access-key-id",
        default=os.getenv("MASSIVE_ACCESS_KEY_ID", ""),
        help="Massive flat-files access key id. Falls back to MASSIVE_ACCESS_KEY_ID.",
    )
    parser.add_argument(
        "--secret-access-key",
        default=os.getenv("MASSIVE_SECRET_ACCESS_KEY", ""),
        help="Massive flat-files secret access key. Falls back to MASSIVE_SECRET_ACCESS_KEY.",
    )
    return parser.parse_args(argv)


def _round_down(value: float, increment: float) -> float:
    return increment * int(value / increment)


def _round_up(value: float, increment: float) -> float:
    quotient = int(value / increment)
    if abs((quotient * increment) - value) < 1e-9:
        return quotient * increment
    return (quotient + 1) * increment


def _normalize_contract_types(raw: str) -> set[str]:
    values = {item.strip().lower() for item in str(raw).split(",") if item.strip()}
    allowed = {"call", "put"}
    invalid = sorted(values - allowed)
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(f"Unsupported contract types: {joined}")
    return values or {"call"}


def _load_daily_underlying(bars_csv: Path, start: str, end: str, symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(bars_csv)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame.loc[frame["symbol"] == str(symbol).upper()].copy()
    if frame.empty:
        raise ValueError(f"No rows found for symbol {symbol} in {bars_csv}")
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    frame = frame[(frame["timestamp"] >= start_ts) & (frame["timestamp"] < end_ts)].copy()
    if frame.empty:
        raise ValueError(f"No rows found for {symbol} between {start} and {end} in {bars_csv}")
    frame["trade_date"] = frame["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    daily = (
        frame.groupby("trade_date", as_index=False)
        .agg(close=("close", "last"))
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    return daily


def _session_client(access_key_id: str, secret_access_key: str, endpoint_url: str):
    if not access_key_id or not secret_access_key:
        raise ValueError(
            "Massive flat-files credentials are required. Provide --access-key-id/--secret-access-key "
            "or set MASSIVE_ACCESS_KEY_ID and MASSIVE_SECRET_ACCESS_KEY."
        )
    session = boto3.session.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )
    return session.client("s3", endpoint_url=endpoint_url, config=Config(signature_version="s3v4"))


def _part_path(parts_dir: Path, trade_date: str) -> Path:
    return parts_dir / f"{trade_date}.csv"


def _flatfile_key(trade_date: str) -> str:
    year, month, _day = trade_date.split("-", maxsplit=2)
    return f"us_options_opra/day_aggs_v1/{year}/{month}/{trade_date}.csv.gz"


def _build_filtered_frame(raw: pd.DataFrame, trade_date: str, symbol: str, option_roots: set[str], contract_types: set[str], strike_lo: float, strike_hi: float, dte_min: int, dte_max: int) -> pd.DataFrame:
    tickers = raw["ticker"].astype(str)
    parsed = tickers.str.extract(TICKER_PATTERN)
    working = raw.join(parsed)
    working = working[working["root"].isin(option_roots)].copy()
    if working.empty:
        return pd.DataFrame()

    working["expiry"] = pd.to_datetime(working["expiry"], format="%y%m%d", errors="coerce")
    working["strike"] = pd.to_numeric(working["strike_raw"], errors="coerce") / 1000.0
    working["dte"] = (working["expiry"] - pd.Timestamp(trade_date)).dt.days
    working["type"] = working["type"].map({"C": "call", "P": "put"})
    working = working[
        working["expiry"].notna()
        & working["strike"].notna()
        & working["type"].isin(contract_types)
        & working["dte"].between(int(dte_min), int(dte_max))
        & working["strike"].between(float(strike_lo), float(strike_hi))
    ].copy()
    if working.empty:
        return pd.DataFrame()

    working["date"] = trade_date
    working["option_ticker"] = working["ticker"].astype(str)
    working["underlying"] = str(symbol).upper()
    working["expiry"] = working["expiry"].dt.strftime("%Y-%m-%d")
    columns = [
        "date",
        "option_ticker",
        "underlying",
        "expiry",
        "strike",
        "type",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "transactions",
    ]
    for name in columns:
        if name not in working.columns:
            working[name] = pd.NA
    return working[columns].sort_values(["expiry", "strike", "type", "option_ticker"]).reset_index(drop=True)


def build_dataset(args: argparse.Namespace) -> Path:
    bars_csv = Path(args.bars_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    daily = _load_daily_underlying(bars_csv, str(args.start), str(args.end), str(args.symbol))
    option_roots = {item.strip().upper() for item in str(args.option_roots).split(",") if item.strip()}
    contract_types = _normalize_contract_types(str(args.contract_types))
    rounding = max(0.5, float(args.center_rounding))
    reach = max(0.0, float(args.max_width_points)) + max(0.0, float(args.strike_buffer_points))
    s3 = _session_client(str(args.access_key_id), str(args.secret_access_key), str(args.s3_endpoint))

    checkpoint_path = output_dir / "checkpoint.json"
    final_path = output_dir / "spx_options_daily_history.csv"
    completed_dates: list[str] = []

    for row in daily.itertuples(index=False):
        trade_date = str(row.trade_date)
        close = float(row.close)
        part_path = _part_path(parts_dir, trade_date)
        strike_lo = _round_down(close - reach, rounding)
        strike_hi = _round_up(close + reach, rounding)
        if part_path.exists():
            completed_dates.append(trade_date)
            continue

        key = _flatfile_key(trade_date)
        print(
            f"Filtering {trade_date} | close={close:.2f} | dte={int(args.dte_min)}..{int(args.dte_max)} | "
            f"strikes={strike_lo:.2f}..{strike_hi:.2f}"
        )
        obj = s3.get_object(Bucket=str(args.bucket), Key=key)
        raw = pd.read_csv(io.BytesIO(gzip.decompress(obj["Body"].read())))
        filtered = _build_filtered_frame(
            raw=raw,
            trade_date=trade_date,
            symbol=str(args.symbol),
            option_roots=option_roots,
            contract_types=contract_types,
            strike_lo=strike_lo,
            strike_hi=strike_hi,
            dte_min=int(args.dte_min),
            dte_max=int(args.dte_max),
        )
        filtered.to_csv(part_path, index=False)
        completed_dates.append(trade_date)
        checkpoint = {
            "bars_csv": str(bars_csv),
            "output_dir": str(output_dir),
            "final_dataset_path": str(final_path),
            "completed_dates": completed_dates,
            "last_trade_date": trade_date,
            "last_row_count": int(len(filtered)),
        }
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        print(f"Wrote {part_path.name} with {len(filtered)} rows.")

    frames: list[pd.DataFrame] = []
    for part_path in sorted(parts_dir.glob("*.csv")):
        if part_path.stat().st_size == 0:
            continue
        frame = pd.read_csv(part_path)
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No filtered rows were written to {parts_dir}")

    final = pd.concat(frames, ignore_index=True)
    final = final.sort_values(["date", "expiry", "strike", "type", "option_ticker"]).reset_index(drop=True)
    final.to_csv(final_path, index=False)
    print(f"Saved {len(final)} rows to {final_path}")
    return final_path


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    build_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
