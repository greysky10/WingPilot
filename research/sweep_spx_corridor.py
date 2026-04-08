#! python3.12
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

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
        description=(
            "Run a coarse SPX corridor parameter sweep and save per-run artifacts plus a comparison CSV."
        )
    )
    parser.add_argument("--symbol", default="SPX", help="Underlying symbol for the sweep.")
    parser.add_argument("--start", default="2025-03-30", help="UTC start date for the backtest window.")
    parser.add_argument("--end", default="2026-03-30", help="UTC end date for the backtest window.")
    parser.add_argument("--timeframe", default="5 mins", help="Intraday timeframe label.")
    parser.add_argument("--client-id", type=int, default=160, help="IB client id for the one-time history fetch.")
    parser.add_argument(
        "--output-root",
        default=r".\corridor_outputs\spx_grid",
        help="Root directory for the sweep outputs.",
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
    )


def load_frame(cfg: CorridorConfig, start: str, end: str) -> pd.DataFrame:
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
    return fetch_intraday_bars(request)


def parameter_grid(base: CorridorConfig) -> Iterable[dict[str, float | int]]:
    current_width = float(base.butterfly_width)
    current_coverage = float(base.coverage_band_width)
    current_recenter = float(base.recenter_threshold)
    current_drift = int(base.drift_persistence_bars)
    current_cooldown = int(base.rebuild_cooldown_minutes)

    width_values = [current_width * factor for factor in (2.0, 3.0, 4.0)]
    coverage_values = [current_coverage * factor for factor in (2.0, 3.0, 4.0)]
    recenter_values = [round(current_recenter * factor, 2) for factor in (1.5, 2.0)]
    drift_values = [current_drift * factor for factor in (2, 3)]
    cooldown_values = [current_cooldown, current_cooldown * 3]

    for width, coverage, recenter, drift_bars, cooldown in itertools.product(
        width_values,
        coverage_values,
        recenter_values,
        drift_values,
        cooldown_values,
    ):
        # Keep the corridor geometry coherent: total coverage should span at least
        # the primary butterfly wings on both sides.
        if coverage < 2.0 * width:
            continue
        yield {
            "butterfly_width": width,
            "coverage_band_width": coverage,
            "recenter_threshold": recenter,
            "drift_persistence_bars": int(drift_bars),
            "rebuild_cooldown_minutes": int(cooldown),
        }


def config_name(params: dict[str, float | int]) -> str:
    def fmt(value: float | int) -> str:
        if isinstance(value, int) or float(value).is_integer():
            return str(int(value))
        return str(value).replace(".", "p")

    return (
        f"w{fmt(params['butterfly_width'])}"
        f"_cb{fmt(params['coverage_band_width'])}"
        f"_rt{fmt(params['recenter_threshold'])}"
        f"_dp{fmt(params['drift_persistence_bars'])}"
        f"_cd{fmt(params['rebuild_cooldown_minutes'])}"
    )


def structural_score(row: dict[str, object]) -> float:
    occupancy = float(row.get("corridor_occupancy_rate") or 0.0)
    rebuilds = float(row.get("average_rebuilds_per_day") or 0.0)
    profit_factor = float(row.get("profit_factor_by_closed_layer") or 0.0)

    occupancy_score = min(occupancy / 0.50, 1.0)
    rebuild_score = min(3.0 / rebuilds, 1.0) if rebuilds > 0 else 1.0
    profit_factor_score = min(profit_factor / 1.0, 1.0)
    return round((occupancy_score * 0.4) + (rebuild_score * 0.3) + (profit_factor_score * 0.3), 6)


def is_viable(row: dict[str, object]) -> bool:
    occupancy = float(row.get("corridor_occupancy_rate") or 0.0)
    rebuilds = float(row.get("average_rebuilds_per_day") or 0.0)
    profit_factor = float(row.get("profit_factor_by_closed_layer") or 0.0)
    return occupancy > 0.50 and rebuilds < 3.0 and profit_factor > 1.0


def run_sweep(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "spx_grid_summary.csv"

    base_cfg = build_base_config(args)
    frame = load_frame(base_cfg, args.start, args.end)

    rows: list[dict[str, object]] = []
    for params in parameter_grid(base_cfg):
        name = config_name(params)
        run_dir = output_root / name
        summary_file = run_dir / "summary.json"
        if summary_file.exists():
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            cfg = build_base_config(args)
            cfg.output_dir = run_dir
            cfg.butterfly_width = float(params["butterfly_width"])
            cfg.coverage_band_width = float(params["coverage_band_width"])
            cfg.recenter_threshold = float(params["recenter_threshold"])
            cfg.drift_persistence_bars = int(params["drift_persistence_bars"])
            cfg.rebuild_cooldown_minutes = int(params["rebuild_cooldown_minutes"])
            print(f"[grid] reuse {name}")
        else:
            cfg = build_base_config(args)
            cfg.output_dir = run_dir
            cfg.butterfly_width = float(params["butterfly_width"])
            cfg.coverage_band_width = float(params["coverage_band_width"])
            cfg.recenter_threshold = float(params["recenter_threshold"])
            cfg.drift_persistence_bars = int(params["drift_persistence_bars"])
            cfg.rebuild_cooldown_minutes = int(params["rebuild_cooldown_minutes"])

            result = CorridorBacktestEngine(cfg).run(frame)
            save_backtest_outputs(run_dir, result)
            (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
            summary = result.summary

        row = {
            "config_name": name,
            "butterfly_width": float(params["butterfly_width"]),
            "coverage_band": float(params["coverage_band_width"]),
            "recenter_threshold": float(params["recenter_threshold"]),
            "drift_persistence_bars": int(params["drift_persistence_bars"]),
            "rebuild_cooldown": int(params["rebuild_cooldown_minutes"]),
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
        row["structural_score"] = structural_score(row)
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(
            f"[grid] {name} | occ={row['corridor_occupancy_rate']:.2%} | "
            f"rebuilds/day={row['average_rebuilds_per_day']:.2f} | "
            f"pf={_fmt_number(row['profit_factor_by_closed_layer'])} | "
            f"roc={_fmt_pct(row['return_on_capital'])}"
        )

    summary_frame = pd.DataFrame(rows)
    summary_frame = summary_frame.sort_values(
        by=[
            "is_viable",
            "structural_score",
            "profit_factor_by_closed_layer",
            "corridor_occupancy_rate",
            "average_rebuilds_per_day",
            "return_on_capital",
        ],
        ascending=[False, False, False, False, True, False],
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
    print(f"[grid] wrote {summary_path}")
    print(f"[grid] runs={len(summary_frame)} | viable={viable_count}")
    if not summary_frame.empty:
        top = summary_frame.iloc[0]
        print(
            "[grid] top | "
            f"{top['config_name']} | occ={float(top['corridor_occupancy_rate']):.2%} | "
            f"rebuilds/day={float(top['average_rebuilds_per_day']):.2f} | "
            f"pf={_fmt_number(top['profit_factor_by_closed_layer'])} | "
            f"roc={_fmt_pct(top['return_on_capital'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
