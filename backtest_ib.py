#! python3.12
from __future__ import annotations

import argparse
import time
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from backtest import BacktestConfig, print_summary, replay_frame
from live_ib import build_stock_contract, connect_ib, get_missing_dependencies
from strategy import load_local_env, parse_bool, pd, require_strategy_dependencies


@dataclass
class IBBacktestConfig(BacktestConfig):
    ib_host: str = "127.0.0.1"
    ib_port: int = 4001
    ib_client_id: int = 7
    ib_timeout: float = 10.0
    ib_market_data_type: int = 3
    ib_exchange: str = "SMART"
    ib_currency: str = "USD"
    ib_duration: str = "5 D"
    ib_bar_size: str = "1 min"
    ib_what_to_show: str = "TRADES"
    ib_use_rth: bool = True
    ib_chunk_duration: str = ""
    ib_request_pause_ms: int = 250


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch IB historical bars in memory and print backtest win rate.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbol list. Defaults to SYMBOLS from .env.")
    parser.add_argument("--duration", default=None, help="IB duration string, for example '5 D' or '30 D'.")
    parser.add_argument("--bar-size", default=None, help="IB bar size, for example '1 min'.")
    parser.add_argument("--what-to-show", default=None, help="IB whatToShow value, for example TRADES.")
    parser.add_argument("--use-rth", default=None, help="true/false. Defaults to IB_USE_RTH from .env.")
    parser.add_argument("--client-id", type=int, default=None, help="IB client id override. Default is 7.")
    parser.add_argument("--chunk-duration", default=None, help="Optional per-request chunk duration, for example '10 D'.")
    parser.add_argument("--request-pause-ms", type=int, default=250, help="Pause between chunked IB history requests.")
    parser.add_argument("--signals-out", default="", help="Optional CSV path for emitted signals.")
    parser.add_argument("--show-signals", action="store_true", help="Print each emitted signal during replay.")
    parser.add_argument("--check", action="store_true", help="Validate IB connectivity and contract qualification only.")
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> IBBacktestConfig:
    load_local_env()

    symbols_text = args.symbols or os.getenv("SYMBOLS", "SPY")
    symbols = [x.strip().upper() for x in symbols_text.split(",") if x.strip()]
    if not symbols:
        raise RuntimeError("No symbols configured. Use --symbols or set SYMBOLS in .env.")

    return IBBacktestConfig(
        symbols=symbols,
        signal_csv_path=args.signals_out,
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
        print_json=parse_bool(os.getenv("PRINT_JSON"), False),
        quiet=not args.show_signals,
        ib_host=os.getenv("IB_HOST", "127.0.0.1"),
        ib_port=int(os.getenv("IB_PORT", "4001")),
        ib_client_id=args.client_id if args.client_id is not None else 7,
        ib_timeout=float(os.getenv("IB_TIMEOUT", "10")),
        ib_market_data_type=int(os.getenv("IB_MARKET_DATA_TYPE", "3")),
        ib_exchange=os.getenv("IB_EXCHANGE", "SMART"),
        ib_currency=os.getenv("IB_CURRENCY", "USD"),
        ib_duration=args.duration or os.getenv("IB_DURATION", "5 D"),
        ib_bar_size=args.bar_size or os.getenv("IB_BAR_SIZE", "1 min"),
        ib_what_to_show=args.what_to_show or os.getenv("IB_WHAT_TO_SHOW", "TRADES"),
        ib_use_rth=parse_bool(args.use_rth if args.use_rth is not None else os.getenv("IB_USE_RTH"), True),
        ib_chunk_duration=args.chunk_duration or os.getenv("IB_CHUNK_DURATION", ""),
        ib_request_pause_ms=max(0, args.request_pause_ms),
    )


def duration_to_timedelta(duration_text: str):
    number_text, unit_text = duration_text.strip().split()
    number = int(number_text)
    unit = unit_text.upper()
    if unit.startswith("S"):
        return pd.Timedelta(seconds=number)
    if unit.startswith("D"):
        return pd.Timedelta(days=number)
    if unit.startswith("W"):
        return pd.Timedelta(weeks=number)
    if unit.startswith("M"):
        return pd.Timedelta(days=30 * number)
    if unit.startswith("Y"):
        return pd.Timedelta(days=365 * number)
    raise ValueError(f"Unsupported duration string: {duration_text}")


def default_chunk_duration(bar_size: str) -> str:
    lower = bar_size.strip().lower()
    if lower == "1 min":
        return "10 D"
    if lower in {"2 mins", "3 mins", "5 mins"}:
        return "30 D"
    if lower in {"10 mins", "15 mins", "20 mins", "30 mins"}:
        return "90 D"
    return "180 D"


def fetch_ib_frame(cfg: IBBacktestConfig):
    if pd is None:
        raise RuntimeError("pandas is not installed.")

    missing = get_missing_dependencies()
    if missing:
        raise RuntimeError("Missing packages: " + ", ".join(missing))

    rows = []
    overall_duration = duration_to_timedelta(cfg.ib_duration)
    chunk_duration_text = cfg.ib_chunk_duration.strip() or default_chunk_duration(cfg.ib_bar_size)
    chunk_duration = duration_to_timedelta(chunk_duration_text)
    end_ts = pd.Timestamp.now(tz="UTC")
    start_ts = end_ts - overall_duration
    ib = connect_ib(cfg, readonly=True)
    try:
        for symbol in cfg.symbols:
            contract = build_stock_contract(symbol, cfg)
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                raise RuntimeError(f"Unable to qualify IB contract for {symbol}.")

            symbol_rows = []
            request_end = end_ts
            request_count = 0

            while request_end > start_ts:
                end_text = request_end.strftime("%Y%m%d-%H:%M:%S")
                bars = ib.reqHistoricalData(
                    qualified[0],
                    endDateTime=end_text,
                    durationStr=chunk_duration_text,
                    barSizeSetting=cfg.ib_bar_size,
                    whatToShow=cfg.ib_what_to_show,
                    useRTH=cfg.ib_use_rth,
                    formatDate=2,
                    keepUpToDate=False,
                )
                request_count += 1

                if not bars:
                    break

                oldest_bar_ts = None
                added = 0
                for bar in bars:
                    bar_ts = pd.Timestamp(bar.date)
                    bar_ts = bar_ts.tz_convert("UTC") if bar_ts.tzinfo else bar_ts.tz_localize("UTC")
                    if bar_ts < start_ts or bar_ts > end_ts:
                        continue
                    symbol_rows.append(
                        {
                            "timestamp": bar_ts,
                            "symbol": symbol,
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": float(bar.volume),
                        }
                    )
                    oldest_bar_ts = bar_ts if oldest_bar_ts is None else min(oldest_bar_ts, bar_ts)
                    added += 1

                print(
                    f"Fetched chunk {request_count} for {symbol} | duration={chunk_duration_text} | "
                    f"bars={len(bars)} | kept={added}"
                )

                if oldest_bar_ts is None or oldest_bar_ts <= start_ts:
                    break

                request_end = oldest_bar_ts - pd.Timedelta(seconds=1)
                if cfg.ib_request_pause_ms > 0:
                    time.sleep(cfg.ib_request_pause_ms / 1000.0)

            if not symbol_rows:
                raise RuntimeError(f"IB returned no bars for {symbol}.")

            rows.extend(symbol_rows)
            print(f"Fetched {len(symbol_rows)} total bars for {symbol}")
    finally:
        if ib.isConnected():
            ib.disconnect()

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    frame = frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return frame


def run_startup_check(cfg: IBBacktestConfig) -> int:
    print(f"Interpreter: {sys.executable}")

    missing = get_missing_dependencies()
    if missing:
        print("Dependency check: FAIL")
        print(f"Missing packages: {', '.join(missing)}")
        return 1
    print("Dependency check: OK")

    try:
        frame = fetch_ib_frame(cfg)
    except Exception as exc:
        print("IB fetch: FAIL")
        print(str(exc))
        return 1

    print("IB fetch: OK")
    print(f"Rows: {len(frame)}")
    print(f"Symbols: {', '.join(sorted(frame['symbol'].astype(str).unique()))}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    load_local_env()
    require_strategy_dependencies()

    try:
        cfg = load_config(args)
        if args.check:
            return run_startup_check(cfg)

        frame = fetch_ib_frame(cfg)
        summary = replay_frame(frame, cfg, quiet=cfg.quiet)
        print_summary(summary)
        return 0
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
