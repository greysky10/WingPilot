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
    init_worker,
    materialize_run,
    run_one,
    sort_frame,
    write_rank_outputs,
)


FOMC_EVENT_DATES = (
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10",
    "2026-01-28",
    "2026-03-18",
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search daily entry stacking on top of the long-DTE historical-chain branch, "
            "adding a new entry batch on later sessions when the primary entry filter passes."
        )
    )
    parser.add_argument(
        "--bars-csv",
        default=r".\corridor_outputs\fit_search\SPX_5_mins_bars_20250408_20260409.csv",
        help="Underlying SPX intraday bars CSV.",
    )
    parser.add_argument(
        "--historical-chain-path",
        default=r".\data\massive_spx_strategy_history_longdte_10_35_both_flatfiles\spx_options_daily_history.csv",
        help="Historical-chain CSV produced by the flat-files backfill.",
    )
    parser.add_argument("--start", default="2025-04-10", help="Inclusive UTC start date.")
    parser.add_argument("--end", default="2026-04-09", help="Inclusive UTC end date.")
    parser.add_argument(
        "--output-dir",
        default=r".\corridor_outputs\fit_search\historical_chain_daily_stack_search",
        help="Directory for summaries and materialized best runs.",
    )
    parser.add_argument("--workers", type=int, default=12, help="Parallel worker count.")
    parser.add_argument("--max-cases", type=int, default=0, help="Optional cap for smoke-testing the grid.")
    parser.add_argument("--min-active-layers", type=int, default=20, help="Minimum closed layers treated as active.")
    return parser.parse_args(argv)


def parameter_grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stack_profiles = [
        {"profile_dte": "single_28", "dte_min": 21, "dte_max": 35, "default_dte": 28, "layer_dte_targets": (28,), "max_layers_values": (1, 2, 3)},
        {"profile_dte": "ladder_21_28", "dte_min": 21, "dte_max": 28, "default_dte": 21, "layer_dte_targets": (21, 28), "max_layers_values": (2, 4, 6)},
        {"profile_dte": "ladder_28_35", "dte_min": 28, "dte_max": 35, "default_dte": 28, "layer_dte_targets": (28, 35), "max_layers_values": (2, 4, 6)},
        {"profile_dte": "ladder_21_28_35", "dte_min": 21, "dte_max": 35, "default_dte": 21, "layer_dte_targets": (21, 28, 35), "max_layers_values": (3, 6)},
    ]
    entry_ends = ["09:45", "09:50"]
    confidence_values = [0.55, 0.60]
    momentum_values = [0.0015, 0.0020]
    hold_sessions = [3, 5]
    close_dte_values = [2, 3]
    exit_scopes = ["all", "individual"]

    for profile in stack_profiles:
        ladder = tuple(int(value) for value in profile["layer_dte_targets"])
        for max_layers in profile["max_layers_values"]:
            allow_daily = max_layers > len(ladder)
            for exit_scope in exit_scopes:
                for entry_end in entry_ends:
                    for confidence in confidence_values:
                        for momentum in momentum_values:
                            for max_hold in hold_sessions:
                                for close_dte in close_dte_values:
                                    rows.append(
                                        {
                                            "bar_mode": "intraday_5m",
                                            "profile_branch": "daily_stack",
                                            "profile_dte": str(profile["profile_dte"]),
                                            "dte_min": int(profile["dte_min"]),
                                            "dte_max": int(profile["dte_max"]),
                                            "default_dte": int(profile["default_dte"]),
                                            "layer_dte_targets": ladder,
                                            "layer_exit_scope": str(exit_scope),
                                            "allow_daily_entry_additions": bool(allow_daily),
                                            "max_active_butterfly_layers": int(max_layers),
                                            "profile_geometry": "w10_longdte",
                                            "butterfly_width": 10.0,
                                            "coverage_band_width": 20.0,
                                            "center_tolerance": 10.0,
                                            "recenter_threshold": 20.0,
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
                                            "primary_stop_loss_pct": 0.25,
                                            "primary_take_profit_pct": 1.0,
                                            "option_right_preference": "call",
                                            "skip_event_days": True,
                                            "event_dates": FOMC_EVENT_DATES,
                                            "skip_gap_days": True,
                                            "max_entry_gap_pct": 0.010,
                                            "max_acceptable_option_spread": 0.20,
                                            "per_contract_slippage": 0.075,
                                            "stress_profile": "none",
                                            "stop_label": "stop_0p25",
                                            "take_label": "take_1p0",
                                        }
                                    )
    return rows


def config_name(params: dict[str, object]) -> str:
    ladder = "-".join(str(int(value)) for value in params["layer_dte_targets"])
    return "_".join(
        [
            str(params["bar_mode"]),
            str(params["profile_branch"]),
            str(params["profile_dte"]),
            f"ladder_{ladder}",
            f"maxlayers_{int(params['max_active_butterfly_layers'])}",
            f"daily_{int(bool(params['allow_daily_entry_additions']))}",
            f"exit_{params['layer_exit_scope']}",
            f"end_{str(params['primary_entry_end']).replace(':', '')}",
            f"conf_{str(params['primary_entry_min_center_confidence']).replace('.', 'p')}",
            f"mom_{str(params['primary_entry_max_momentum_pct']).replace('.', 'p')}",
            f"hold_{int(params['max_hold_sessions'])}",
            f"dteclose_{int(params['close_when_dte_lte'])}",
        ]
    )


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

    params_list = parameter_grid()
    if int(args.max_cases) > 0:
        params_list = params_list[: int(args.max_cases)]
    tasks = [(config_name(params), params, str(output_dir)) for params in params_list]
    print(f"Running {len(tasks)} daily-stack cases with workers={max(1, int(args.workers))}.")
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
        for label, subset in {
            "best_daily_additions": active[active["allow_daily_entry_additions"] == True],  # noqa: E712
            "best_no_daily_additions": active[active["allow_daily_entry_additions"] == False],  # noqa: E712
        }.items():
            if subset.empty:
                continue
            material_targets.append((label, subset.iloc[0].to_dict()))

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
