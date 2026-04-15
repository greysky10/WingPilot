#! python3.12
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.param_search_historical_chain_axis3 import (
    config_name,
    init_worker,
    materialize_run,
    run_one,
    sort_frame,
    write_rank_outputs,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search longer-DTE historical-chain corridor variants while preserving the existing "
            "entry/exit framework. Intended for swing-style SPX call butterfly testing."
        )
    )
    parser.add_argument(
        "--bars-csv",
        default=r".\corridor_outputs\fit_search\SPX_5_mins_bars_20250408_20260409.csv",
        help="Underlying SPX intraday bars CSV.",
    )
    parser.add_argument(
        "--historical-chain-path",
        default=r".\data\massive_spx_strategy_history_longdte_10_35_flatfiles\spx_options_daily_history.csv",
        help="Longer-DTE historical-chain CSV produced by run_massive_spx_backfill.py.",
    )
    parser.add_argument("--start", default="2025-04-10", help="Inclusive UTC start date.")
    parser.add_argument("--end", default="2026-04-09", help="Inclusive UTC end date.")
    parser.add_argument(
        "--output-dir",
        default=r".\corridor_outputs\fit_search\historical_chain_longdte_search",
        help="Directory for summaries and per-run configs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Parallel worker count.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap for smoke-testing the grid. Set 0 to run the full search.",
    )
    parser.add_argument(
        "--branch",
        default="both",
        choices=["both", "intraday", "daily"],
        help="Limit the search to a subset of tasks.",
    )
    parser.add_argument(
        "--min-active-layers",
        type=int,
        default=10,
        help="Minimum closed layers used by the ranking score to treat a run as active.",
    )
    return parser.parse_args(argv)


def intraday_longdte_tasks() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dte_profiles = [
        {"profile_dte": "dte_10_14_12", "dte_min": 10, "dte_max": 14, "default_dte": 12},
        {"profile_dte": "dte_14_21_18", "dte_min": 14, "dte_max": 21, "default_dte": 18},
        {"profile_dte": "dte_21_35_28", "dte_min": 21, "dte_max": 35, "default_dte": 28},
    ]
    width_profiles = [
        {
            "profile_geometry": "w5_tight",
            "butterfly_width": 5.0,
            "coverage_band_width": 10.0,
            "center_tolerance": 5.0,
            "recenter_threshold": 10.0,
        },
        {
            "profile_geometry": "w10_base",
            "butterfly_width": 10.0,
            "coverage_band_width": 20.0,
            "center_tolerance": 10.0,
            "recenter_threshold": 20.0,
        },
    ]
    confidence_values = [0.55, 0.70]
    momentum_values = [0.0015, 0.0025]
    entry_ends = ["10:00", "10:30"]
    hold_sessions = [2, 5]
    close_dte_values = [3, 5]
    stop_values = [0.25, 0.50]
    take_values = [0.50, 1.00]

    for dte in dte_profiles:
        for width in width_profiles:
            for confidence in confidence_values:
                for momentum in momentum_values:
                    for entry_end in entry_ends:
                        for max_hold in hold_sessions:
                            for close_dte in close_dte_values:
                                for stop in stop_values:
                                    for take in take_values:
                                        rows.append(
                                            {
                                                "bar_mode": "intraday_5m",
                                                "profile_branch": "intraday_longdte",
                                                **dte,
                                                "profile_geometry": width["profile_geometry"],
                                                "butterfly_width": width["butterfly_width"],
                                                "coverage_band_width": width["coverage_band_width"],
                                                "center_tolerance": width["center_tolerance"],
                                                "recenter_threshold": width["recenter_threshold"],
                                                "center_lookback": 36,
                                                "regime_lookback": 48,
                                                "range_width_threshold_pct": 0.012,
                                                "trend_slope_threshold_pct": 0.0015,
                                                "breakout_buffer_pct": 0.0025,
                                                "primary_entry_min_center_confidence": confidence,
                                                "primary_entry_max_momentum_pct": momentum,
                                                "primary_entry_end": entry_end,
                                                "drift_persistence_bars": 8,
                                                "rebuild_cooldown_minutes": 120,
                                                "hold_overnight": True,
                                                "max_hold_sessions": max_hold,
                                                "close_when_dte_lte": close_dte,
                                                "primary_stop_loss_pct": stop,
                                                "primary_take_profit_pct": take,
                                                "stop_label": f"stop_{str(stop).replace('.', 'p')}",
                                                "take_label": f"take_{str(take).replace('.', 'p')}",
                                            }
                                        )
    return rows


def daily_longdte_tasks() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dte_profiles = [
        {"profile_dte": "dte_10_14_12", "dte_min": 10, "dte_max": 14, "default_dte": 12},
        {"profile_dte": "dte_14_21_18", "dte_min": 14, "dte_max": 21, "default_dte": 18},
        {"profile_dte": "dte_21_35_28", "dte_min": 21, "dte_max": 35, "default_dte": 28},
    ]
    width_profiles = [
        {
            "profile_geometry": "w5_daily_base",
            "butterfly_width": 5.0,
            "coverage_band_width": 25.0,
            "center_tolerance": 15.0,
            "recenter_threshold": 25.0,
        },
        {
            "profile_geometry": "w10_daily_mid",
            "butterfly_width": 10.0,
            "coverage_band_width": 30.0,
            "center_tolerance": 20.0,
            "recenter_threshold": 30.0,
        },
        {
            "profile_geometry": "w15_daily_wide",
            "butterfly_width": 15.0,
            "coverage_band_width": 40.0,
            "center_tolerance": 25.0,
            "recenter_threshold": 35.0,
        },
    ]
    momentum_values = [0.013, 0.0175, 0.0225]
    hold_sessions = [3, 5, 7]
    close_dte_values = [3, 5]
    stop_values = [0.50, 1.00]
    take_values = [0.25, 0.50, 1.00]

    for dte in dte_profiles:
        for width in width_profiles:
            for momentum in momentum_values:
                for max_hold in hold_sessions:
                    for close_dte in close_dte_values:
                        for stop in stop_values:
                            for take in take_values:
                                rows.append(
                                    {
                                        "bar_mode": "daily_close",
                                        "profile_branch": "daily_longdte",
                                        **dte,
                                        "profile_geometry": width["profile_geometry"],
                                        "butterfly_width": width["butterfly_width"],
                                        "coverage_band_width": width["coverage_band_width"],
                                        "center_tolerance": width["center_tolerance"],
                                        "recenter_threshold": width["recenter_threshold"],
                                        "center_lookback": 5,
                                        "regime_lookback": 12,
                                        "range_width_threshold_pct": 0.03,
                                        "trend_slope_threshold_pct": 0.004,
                                        "breakout_buffer_pct": 0.005,
                                        "primary_entry_min_center_confidence": 0.0,
                                        "primary_entry_max_momentum_pct": momentum,
                                        "primary_entry_end": "15:30",
                                        "drift_persistence_bars": 1,
                                        "rebuild_cooldown_minutes": 0,
                                        "hold_overnight": True,
                                        "max_hold_sessions": max_hold,
                                        "close_when_dte_lte": close_dte,
                                        "primary_stop_loss_pct": stop,
                                        "primary_take_profit_pct": take,
                                        "stop_label": f"stop_{str(stop).replace('.', 'p')}",
                                        "take_label": f"take_{str(take).replace('.', 'p')}",
                                    }
                                )
    return rows


def parameter_grid() -> list[dict[str, object]]:
    return intraday_longdte_tasks() + daily_longdte_tasks()


def filter_parameter_grid(rows: list[dict[str, object]], branch: str) -> list[dict[str, object]]:
    if branch == "intraday":
        return [row for row in rows if str(row["bar_mode"]) == "intraday_5m"]
    if branch == "daily":
        return [row for row in rows if str(row["bar_mode"]) == "daily_close"]
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    bars_csv = Path(args.bars_csv).resolve()
    historical_chain_path = Path(args.historical_chain_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs").mkdir(parents=True, exist_ok=True)

    if not bars_csv.exists():
        raise FileNotFoundError(f"Bars CSV not found: {bars_csv}")
    if not historical_chain_path.exists():
        raise FileNotFoundError(f"Historical-chain dataset not found: {historical_chain_path}")

    params_list = filter_parameter_grid(parameter_grid(), str(args.branch))
    if int(args.max_cases) > 0:
        params_list = params_list[: int(args.max_cases)]
    tasks = [(config_name(params), params, str(output_dir)) for params in params_list]
    print(f"Running {len(tasks)} long-DTE cases with workers={max(1, int(args.workers))}.")
    rows: list[dict[str, object]] = []

    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=init_worker,
        initargs=(
            str(bars_csv),
            str(historical_chain_path),
            str(args.start),
            str(args.end),
            int(args.min_active_layers),
        ),
    ) as pool:
        future_map = {pool.submit(run_one, task): task[0] for task in tasks}
        for idx, future in enumerate(as_completed(future_map), start=1):
            row = future.result()
            rows.append(row)
            current = pd.DataFrame(rows)
            write_rank_outputs(output_dir, current)
            print(
                f"[{idx}/{len(tasks)}] {row['config_name']} | "
                f"mode={row['bar_mode']} | "
                f"pnl={float(row['net_dollar_pnl']):.2f} | "
                f"roc={float(row['return_on_capital']):.2%} | "
                f"pf_day={float(row['profit_factor_by_day']):.3f} | "
                f"layers={int(row['closed_layers'])}"
            )

    final = sort_frame(pd.DataFrame(rows))
    write_rank_outputs(output_dir, final)
    active = final.loc[final["is_active_fit"] == True].copy()  # noqa: E712
    material_targets: list[tuple[str, dict[str, object]]] = []
    if not active.empty:
        material_targets.append(("best_score_active", active.iloc[0].to_dict()))
        material_targets.append(
            ("best_net_active", active.sort_values(["net_dollar_pnl", "closed_layers"], ascending=[False, False]).iloc[0].to_dict())
        )
        for mode in ["daily_close", "intraday_5m"]:
            subset = active[active["bar_mode"] == mode]
            if subset.empty:
                continue
            material_targets.append((f"best_{mode}_score", subset.iloc[0].to_dict()))
            material_targets.append(
                (
                    f"best_{mode}_net",
                    subset.sort_values(["net_dollar_pnl", "closed_layers"], ascending=[False, False]).iloc[0].to_dict(),
                )
            )

    seen: set[str] = set()
    for label, row in material_targets:
        config_name_value = str(row["config_name"])
        if config_name_value in seen:
            continue
        seen.add(config_name_value)
        params = next(params for params in params_list if config_name(params) == config_name_value)
        materialize_run(
            bars_csv=bars_csv,
            historical_chain_path=historical_chain_path,
            start=str(args.start),
            end=str(args.end),
            params=params,
            destination=output_dir / label,
        )
    print(f"Done. Summary written to {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
