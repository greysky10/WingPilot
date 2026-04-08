#! python3.12
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corridor.data.historical_loader import HistoricalLoadConfig, load_intraday_bars
from weekly_corridor.backtest import WeeklyBacktestEngine
from weekly_corridor.config import WeeklyCorridorConfig
from weekly_corridor.report import save_backtest_outputs


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a weekly SPX corridor parameter sweep on cached bars.")
    parser.add_argument(
        "--bars-csv",
        default=".\\corridor_outputs\\spx_grid_center_tol\\SPX_5_mins_bars.csv",
        help="Normalized intraday bars CSV used for the weekly sweep.",
    )
    parser.add_argument("--symbol", default="SPX", help="Ticker symbol.")
    parser.add_argument("--start", help="Optional UTC start date.")
    parser.add_argument("--end", help="Optional UTC end date.")
    parser.add_argument(
        "--output-root",
        default=".\\weekly_corridor_outputs\\spx_grid",
        help="Directory where per-run outputs and the summary CSV are written.",
    )
    return parser.parse_args(argv)


def load_frame(args: argparse.Namespace) -> pd.DataFrame:
    start = pd.Timestamp(args.start) if args.start else None
    end = pd.Timestamp(args.end) if args.end else None
    return load_intraday_bars(
        HistoricalLoadConfig(
            csv_path=Path(args.bars_csv),
            symbol=args.symbol.upper(),
            start=start,
            end=end,
        )
    )


def sweep_configs(base: WeeklyCorridorConfig) -> Iterable[tuple[str, WeeklyCorridorConfig]]:
    regime_profiles = {
        "strict": {"weekly_range_width_threshold_pct": 0.055, "weekly_trend_slope_threshold_pct": 0.012, "weekly_momentum_threshold_pct": 0.008},
        "medium": {"weekly_range_width_threshold_pct": 0.070, "weekly_trend_slope_threshold_pct": 0.016, "weekly_momentum_threshold_pct": 0.012},
        "loose": {"weekly_range_width_threshold_pct": 0.085, "weekly_trend_slope_threshold_pct": 0.020, "weekly_momentum_threshold_pct": 0.014},
    }
    width_spacing_pairs = [
        (50.0, 50.0),
        (60.0, 50.0),
        (75.0, 50.0),
        (75.0, 75.0),
        (90.0, 60.0),
        (100.0, 75.0),
    ]
    tolerances = [50.0, 75.0, 100.0, 125.0]

    for profile_name, regime_kwargs in regime_profiles.items():
        for width, spacing in width_spacing_pairs:
            for tolerance in tolerances:
                coverage = max(200.0, (spacing + width) * 2.0)
                name = f"{profile_name}_w{int(width)}_s{int(spacing)}_tol{int(tolerance)}"
                yield name, replace(
                    base,
                    butterfly_width=width,
                    center_spacing=spacing,
                    weekly_center_tolerance=tolerance,
                    target_total_coverage=coverage,
                    **regime_kwargs,
                )


def structural_rank_key(row: pd.Series) -> tuple:
    pf = row["profit_factor_by_closed_layer"]
    roc = row["return_on_capital"]
    return (
        -float(row["weekly_occupancy_rate"]),
        int(row["weeks_aborted"]),
        -float(pf) if pd.notna(pf) else float("inf"),
        -float(roc) if pd.notna(roc) else float("inf"),
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    frame = load_frame(args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base = WeeklyCorridorConfig(symbol=args.symbol.upper(), output_dir=output_root)
    rows: list[dict[str, object]] = []

    for config_name, cfg in sweep_configs(base):
        cfg.output_dir = output_root / config_name
        result = WeeklyBacktestEngine(cfg).run(frame)
        save_backtest_outputs(cfg.output_dir, result)
        rows.append(
            {
                "config_name": config_name,
                "butterfly_width": cfg.butterfly_width,
                "center_spacing": cfg.center_spacing,
                "weekly_center_tolerance": cfg.weekly_center_tolerance,
                "target_total_coverage": cfg.target_total_coverage,
                "weekly_range_width_threshold_pct": cfg.weekly_range_width_threshold_pct,
                "weekly_trend_slope_threshold_pct": cfg.weekly_trend_slope_threshold_pct,
                "weekly_momentum_threshold_pct": cfg.weekly_momentum_threshold_pct,
                "net_modeled_pnl": result.summary["net_modeled_pnl"],
                "net_dollar_pnl": result.summary["net_dollar_pnl"],
                "return_on_capital": result.summary["return_on_capital"],
                "max_gross_deployment_dollars": result.summary["max_gross_deployment_dollars"],
                "worst_day_pnl_dollars": result.summary["worst_day_pnl_dollars"],
                "worst_week_pnl_dollars": result.summary["worst_week_pnl_dollars"],
                "profit_factor_by_closed_layer": result.summary["profit_factor_by_closed_layer"],
                "profit_factor_by_week": result.summary["profit_factor_by_week"],
                "weekly_occupancy_rate": result.summary["weekly_occupancy_rate"],
                "avg_adjustments_per_week": result.summary["avg_adjustments_per_week"],
                "weeks_traded": result.summary["weeks_traded"],
                "weeks_aborted": result.summary["weeks_aborted"],
                "max_active_butterflies": result.summary["max_active_butterflies"],
                "closed_layers": result.summary["closed_layers"],
                "winning_layers": result.summary["winning_layers"],
                "losing_layers": result.summary["losing_layers"],
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(by=["weekly_occupancy_rate", "weeks_aborted", "profit_factor_by_closed_layer", "return_on_capital"], ascending=[False, True, False, False])
    summary_path = output_root / "weekly_spx_grid_summary.csv"
    summary.to_csv(summary_path, index=False)

    top = summary.head(5)
    print(f"Completed {len(summary)} weekly SPX configurations.")
    print(f"Summary CSV: {summary_path}")
    print("Top 5 by structural fit:")
    for row in top.itertuples(index=False):
        print(
            f"- {row.config_name} | occupancy={row.weekly_occupancy_rate:.2%} | "
            f"weeks_aborted={row.weeks_aborted} | pf={_fmt(row.profit_factor_by_closed_layer)} | "
            f"roc={_fmt(row.return_on_capital)}"
        )
    return 0


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
