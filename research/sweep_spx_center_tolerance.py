#! python3.12
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corridor.backtest.engine import CorridorBacktestEngine
from corridor.config import CorridorConfig
from corridor.data.ib_contracts import default_center_rounding_for_symbol
from corridor.data.ib_loader import IBHistoricalRequest, fetch_intraday_bars
from corridor.models import CenterMethod
from corridor.report.summary import save_backtest_outputs
from strategy import load_local_env


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an SPX corridor sweep focused on center_tolerance and write a comparison CSV."
    )
    parser.add_argument("--symbol", default="SPX", help="Underlying symbol.")
    parser.add_argument("--start", default="2025-03-30", help="UTC start date.")
    parser.add_argument("--end", default="2026-03-30", help="UTC end date.")
    parser.add_argument("--timeframe", default="5 mins", help="Intraday timeframe label.")
    parser.add_argument("--client-id", type=int, default=170, help="IB client id for the one-time history fetch.")
    parser.add_argument("--bars-csv", default="", help="Optional cached intraday bars CSV to avoid refetching from IB.")
    parser.add_argument(
        "--output-root",
        default=r".\corridor_outputs\spx_grid_center_tol",
        help="Root directory for per-run outputs and the summary CSV.",
    )
    return parser.parse_args(argv)


def build_base_config(args: argparse.Namespace) -> CorridorConfig:
    symbol = args.symbol.upper()
    return CorridorConfig(
        symbol=symbol,
        timeframe=args.timeframe,
        center_method=CenterMethod.VWAP,
        center_rounding=default_center_rounding_for_symbol(symbol),
        payoff_mode="simplified",
        ib_host=os.getenv("IB_HOST", "127.0.0.1"),
        ib_port=int(os.getenv("IB_PORT", "4001")),
        ib_client_id=args.client_id,
        starting_capital=100000.0,
        contracts_per_layer=1,
        option_multiplier=100,
        # Fixed from the prior churn-friendly SPX sweep.
        coverage_band_width=40.0,
        recenter_threshold=7.0,
        drift_persistence_bars=6,
        rebuild_cooldown_minutes=45,
    )


def load_frame(cfg: CorridorConfig, start: str, end: str, bars_csv: str, cache_path: Path) -> pd.DataFrame:
    if bars_csv:
        frame = pd.read_csv(bars_csv)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.sort_values("timestamp").reset_index(drop=True)
    if cache_path.exists():
        frame = pd.read_csv(cache_path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.sort_values("timestamp").reset_index(drop=True)

    request = IBHistoricalRequest(
        symbol=cfg.symbol,
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        bar_size=cfg.timeframe,
        host=cfg.ib_host,
        port=cfg.ib_port,
        client_id=cfg.ib_client_id,
        exchange=cfg.ib_exchange,
        currency=cfg.ib_currency,
        what_to_show=cfg.ib_what_to_show,
        use_rth=cfg.ib_use_rth,
        chunk_duration=cfg.ib_chunk_duration,
    )
    frame = fetch_intraday_bars(request)
    frame.to_csv(cache_path, index=False)
    return frame


def parameter_grid(base: CorridorConfig):
    tolerance_values = [round(base.center_tolerance * factor, 2) for factor in (1.5, 2.0, 3.0, 4.0, 5.0, 6.0)]
    width_values = [round(base.butterfly_width * factor, 2) for factor in (1.0, 1.5, 2.0, 2.5, 3.0)]
    for tolerance in tolerance_values:
        for width in width_values:
            yield {
                "center_tolerance": tolerance,
                "butterfly_width": width,
                "coverage_band_width": max(40.0, round(width * 2.0, 2)),
                "recenter_threshold": float(base.recenter_threshold),
                "drift_persistence_bars": int(base.drift_persistence_bars),
                "rebuild_cooldown_minutes": int(base.rebuild_cooldown_minutes),
            }


def config_name(params: dict[str, float | int]) -> str:
    def fmt(value: float | int) -> str:
        if isinstance(value, int) or float(value).is_integer():
            return str(int(value))
        return str(value).replace(".", "p")

    return (
        f"tol{fmt(params['center_tolerance'])}"
        f"_w{fmt(params['butterfly_width'])}"
        f"_rt{fmt(params['recenter_threshold'])}"
        f"_dp{fmt(params['drift_persistence_bars'])}"
        f"_cd{fmt(params['rebuild_cooldown_minutes'])}"
    )


def is_viable(row: dict[str, object]) -> bool:
    occupancy = float(row.get("corridor_occupancy_rate") or 0.0)
    rebuilds = float(row.get("average_rebuilds_per_day") or 0.0)
    profit_factor = float(row.get("profit_factor_by_closed_layer") or 0.0)
    return occupancy > 0.40 and rebuilds < 3.0 and profit_factor > 1.0


def run_sweep(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "spx_center_tolerance_summary.csv"
    cache_path = output_root / f"{args.symbol.upper()}_{args.timeframe.replace(' ', '_')}_bars.csv"

    base_cfg = build_base_config(args)
    frame = load_frame(base_cfg, args.start, args.end, args.bars_csv, cache_path)

    rows: list[dict[str, object]] = []
    for params in parameter_grid(base_cfg):
        name = config_name(params)
        run_dir = output_root / name
        summary_file = run_dir / "summary.json"

        if summary_file.exists():
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            print(f"[center_tol] reuse {name}")
        else:
            cfg = build_base_config(args)
            cfg.output_dir = run_dir
            cfg.center_tolerance = float(params["center_tolerance"])
            cfg.butterfly_width = float(params["butterfly_width"])
            cfg.coverage_band_width = float(params["coverage_band_width"])
            result = CorridorBacktestEngine(cfg).run(frame)
            save_backtest_outputs(run_dir, result)
            (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
            summary = result.summary

        row = {
            "config_name": name,
            "center_tolerance": float(params["center_tolerance"]),
            "butterfly_width": float(params["butterfly_width"]),
            "recenter_threshold": float(params["recenter_threshold"]),
            "drift_persistence_bars": int(params["drift_persistence_bars"]),
            "rebuild_cooldown_minutes": int(params["rebuild_cooldown_minutes"]),
            "corridor_occupancy_rate": summary["corridor_occupancy_rate"],
            "average_rebuilds_per_day": summary["average_rebuilds_per_day"],
            "closed_layers": summary["closed_layers"],
            "win_rate_by_closed_layer": summary["win_rate_by_closed_layer"],
            "profit_factor_by_closed_layer": summary["profit_factor_by_closed_layer"],
            "profit_factor_by_day": summary["profit_factor_by_day"],
            "return_on_capital": summary["return_on_capital"],
            "best_day_pnl_dollars": summary["best_day_pnl_dollars"],
            "worst_day_pnl_dollars": summary["worst_day_pnl_dollars"],
            "max_gross_deployment_dollars": summary["max_gross_deployment_dollars"],
        }
        row["is_viable"] = is_viable(row)
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(
            f"[center_tol] {name} | occ={row['corridor_occupancy_rate']:.2%} | "
            f"rebuilds/day={row['average_rebuilds_per_day']:.2f} | "
            f"pf={_fmt_number(row['profit_factor_by_closed_layer'])} | "
            f"roc={_fmt_pct(row['return_on_capital'])}"
        )

    summary_frame = pd.DataFrame(rows)
    summary_frame = summary_frame.sort_values(
        by=[
            "corridor_occupancy_rate",
            "average_rebuilds_per_day",
            "profit_factor_by_closed_layer",
            "return_on_capital",
        ],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    summary_frame.to_csv(summary_path, index=False)
    return summary_frame, summary_path


def _fmt_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def _fmt_number(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def main(argv: Optional[list[str]] = None) -> int:
    load_local_env()
    args = parse_args(argv)
    summary_frame, summary_path = run_sweep(args)
    viable_count = int(summary_frame["is_viable"].sum()) if not summary_frame.empty else 0
    print(f"[center_tol] wrote {summary_path}")
    print(f"[center_tol] runs={len(summary_frame)} | viable={viable_count}")
    if not summary_frame.empty:
        top = summary_frame.iloc[0]
        print(
            "[center_tol] top | "
            f"{top['config_name']} | occ={float(top['corridor_occupancy_rate']):.2%} | "
            f"rebuilds/day={float(top['average_rebuilds_per_day']):.2f} | "
            f"pf={_fmt_number(top['profit_factor_by_closed_layer'])} | "
            f"roc={_fmt_pct(top['return_on_capital'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
