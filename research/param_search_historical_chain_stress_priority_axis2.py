#! python3.12
from __future__ import annotations

import argparse
import json
import os
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


FILTER_EVENT_DATES = (
    "2025-05-07",
    "2025-05-27",
    "2025-06-18",
    "2025-07-01",
    "2025-07-02",
    "2025-07-03",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-11-24",
    "2025-11-25",
    "2025-11-26",
    "2025-12-10",
    "2025-12-22",
    "2025-12-23",
    "2025-12-24",
    "2025-12-26",
    "2025-12-29",
    "2025-12-30",
    "2025-12-31",
    "2026-01-28",
    "2026-03-18",
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a second stress-priority historical-chain search around the best "
            "single-28DTE stress-positive branch, varying entry end, momentum, "
            "close_when_dte_lte, and max_layers."
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
        default=r".\corridor_outputs\fit_search\historical_chain_stress_priority_axis2_search",
        help="Directory for summaries and materialized best runs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(12, (os.cpu_count() or 4) - 1)),
        help="Parallel worker count.",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Optional cap for smoke-testing the grid.")
    parser.add_argument(
        "--min-active-layers",
        type=int,
        default=8,
        help="Minimum closed layers treated as active for ranking.",
    )
    return parser.parse_args(argv)


def parameter_grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry_end in ("09:45", "09:50"):
        for momentum in (0.0008, 0.0010, 0.0012):
            for close_dte in (1, 2, 3):
                for max_layers in (3, 4):
                    rows.append(
                        {
                            "bar_mode": "intraday_5m",
                            "profile_branch": "stress_priority_axis2",
                            "profile_dte": "single_28",
                            "profile_geometry": "w10_single_28",
                            "dte_min": 21,
                            "dte_max": 35,
                            "default_dte": 28,
                            "layer_dte_targets": (28,),
                            "layer_exit_scope": "all",
                            "allow_daily_entry_additions": True,
                            "max_active_butterfly_layers": max_layers,
                            "butterfly_width": 10.0,
                            "coverage_band_width": 20.0,
                            "center_tolerance": 10.0,
                            "recenter_threshold": 20.0,
                            "center_lookback": 36,
                            "regime_lookback": 48,
                            "range_width_threshold_pct": 0.012,
                            "trend_slope_threshold_pct": 0.0015,
                            "breakout_buffer_pct": 0.0025,
                            "primary_entry_end": entry_end,
                            "primary_entry_min_center_confidence": 0.60,
                            "primary_entry_max_momentum_pct": momentum,
                            "skip_entry_weekdays": ("mon", "thu", "fri"),
                            "weekday_profile": "tue_wed_only",
                            "drift_persistence_bars": 8,
                            "rebuild_cooldown_minutes": 120,
                            "hold_overnight": True,
                            "max_hold_sessions": 2,
                            "close_when_dte_lte": close_dte,
                            "primary_stop_loss_pct": 0.25,
                            "primary_take_profit_pct": 0.10,
                            "block_same_day_reentry_after_take_profit": True,
                            "option_right_preference": "call",
                            "skip_event_days": True,
                            "event_dates": FILTER_EVENT_DATES,
                            "skip_gap_days": True,
                            "max_entry_gap_pct": 0.010,
                            "max_acceptable_option_spread": 0.20,
                            "near_spread_dte_max": 14,
                            "near_max_acceptable_option_spread": 0.10,
                            "mid_max_acceptable_option_spread": 0.18,
                            "far_spread_dte_min": 28,
                            "far_max_acceptable_option_spread": 0.28,
                            "per_contract_slippage": 0.10,
                            "stress_profile": "conservative",
                        }
                    )
    return rows


def config_name(params: dict[str, object]) -> str:
    return "_".join(
        [
            str(params["profile_branch"]),
            f"layers_{int(params['max_active_butterfly_layers'])}",
            f"end_{str(params['primary_entry_end']).replace(':', '')}",
            f"mom_{str(params['primary_entry_max_momentum_pct']).replace('.', 'p')}",
            f"dteclose_{int(params['close_when_dte_lte'])}",
        ]
    )


def base_match_params(params: dict[str, object]) -> dict[str, object]:
    matched = dict(params)
    matched["stress_profile"] = "none"
    matched["per_contract_slippage"] = 0.075
    return matched


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
    print(f"Running {len(tasks)} stress-priority axis2 cases with workers={max(1, int(args.workers))}.")
    rows: list[dict[str, object]] = []

    with ProcessPoolExecutor(
        max_workers=max(1, min(int(args.workers), len(tasks))),
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
            current = sort_frame(pd.DataFrame(rows))
            write_rank_outputs(output_dir, current)
            print(
                f"[{idx}/{len(tasks)}] {row['config_name']} | "
                f"pnl={float(row['net_dollar_pnl']):.2f} | "
                f"dd={float(row['max_drawdown_dollars']):.2f} | "
                f"pf_day={float(row['profit_factor_by_day']):.3f} | "
                f"layers={int(row['closed_layers'])}"
            )

    final = sort_frame(pd.DataFrame(rows))
    write_rank_outputs(output_dir, final)
    positive = final.loc[final["net_dollar_pnl"] > 0].copy()
    positive.to_csv(output_dir / "summary_positive.csv", index=False)

    active = final.loc[final["is_active_fit"] == True].copy()  # noqa: E712
    chosen = active.iloc[0].to_dict() if not active.empty else final.iloc[0].to_dict()
    chosen_name = str(chosen["config_name"])
    chosen_params = next(params for params in params_list if config_name(params) == chosen_name)

    materialize_run(
        bars_csv=bars_csv,
        historical_chain_path=historical_chain_path,
        start=str(args.start),
        end=str(args.end),
        params=chosen_params,
        destination=output_dir / "best_stress_active",
    )

    matched_base = base_match_params(chosen_params)
    materialize_run(
        bars_csv=bars_csv,
        historical_chain_path=historical_chain_path,
        start=str(args.start),
        end=str(args.end),
        params=matched_base,
        destination=output_dir / "best_stress_active_base_match",
    )

    (output_dir / "best_stress_active" / "rank_row.json").write_text(
        json.dumps(chosen, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Done. Summary written to {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
