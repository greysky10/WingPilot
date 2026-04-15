#! python3.12
from __future__ import annotations

import argparse
import json
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
            "Search a broader weekday-filtered refinement grid around the current best "
            "daily ladder historical-chain branch."
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
        default=r".\corridor_outputs\fit_search\historical_chain_weekday_filtered_refine_search",
        help="Directory for summaries and materialized best runs.",
    )
    parser.add_argument("--workers", type=int, default=12, help="Parallel worker count.")
    parser.add_argument("--max-cases", type=int, default=0, help="Optional cap for smoke-testing the grid.")
    parser.add_argument("--min-active-layers", type=int, default=20, help="Minimum closed layers treated as active.")
    parser.add_argument("--stress-top-k", type=int, default=24, help="How many top base configs to re-test with conservative stress.")
    return parser.parse_args(argv)


def parameter_grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    weekday_profiles = [
        {"weekday_profile": "none", "skip_entry_weekdays": ()},
        {"weekday_profile": "fri", "skip_entry_weekdays": ("fri",)},
        {"weekday_profile": "mon_fri", "skip_entry_weekdays": ("mon", "fri")},
        {"weekday_profile": "tue_wed_only", "skip_entry_weekdays": ("mon", "thu", "fri")},
    ]
    spread_profiles = [
        {
            "spread_profile": "tight",
            "near_spread_dte_max": 14,
            "near_max_acceptable_option_spread": 0.10,
            "mid_max_acceptable_option_spread": 0.18,
            "far_spread_dte_min": 28,
            "far_max_acceptable_option_spread": 0.28,
        },
        {
            "spread_profile": "base",
            "near_spread_dte_max": 14,
            "near_max_acceptable_option_spread": 0.12,
            "mid_max_acceptable_option_spread": 0.20,
            "far_spread_dte_min": 28,
            "far_max_acceptable_option_spread": 0.30,
        },
    ]
    for max_layers in (4, 6):
        for entry_end in ("09:45", "09:50", "10:00"):
            for confidence in (0.55, 0.60):
                for momentum in (0.0010, 0.0015, 0.0020):
                    for max_hold in (3, 4):
                        for close_dte in (2, 3):
                            for spread in spread_profiles:
                                for weekdays in weekday_profiles:
                                    rows.append(
                                        {
                                            "bar_mode": "intraday_5m",
                                            "profile_branch": "weekday_filtered_refine",
                                            "profile_dte": "ladder_28_35",
                                            "profile_geometry": "w10_longdte",
                                            "dte_min": 21,
                                            "dte_max": 35,
                                            "default_dte": 28,
                                            "layer_dte_targets": (28, 35),
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
                                            "primary_entry_min_center_confidence": confidence,
                                            "primary_entry_max_momentum_pct": momentum,
                                            "skip_entry_weekdays": tuple(str(value) for value in weekdays["skip_entry_weekdays"]),
                                            "weekday_profile": str(weekdays["weekday_profile"]),
                                            "drift_persistence_bars": 8,
                                            "rebuild_cooldown_minutes": 120,
                                            "hold_overnight": True,
                                            "max_hold_sessions": max_hold,
                                            "close_when_dte_lte": close_dte,
                                            "primary_stop_loss_pct": 0.25,
                                            "primary_take_profit_pct": 1.0,
                                            "option_right_preference": "call",
                                            "skip_event_days": True,
                                            "event_dates": FILTER_EVENT_DATES,
                                            "skip_gap_days": True,
                                            "max_entry_gap_pct": 0.010,
                                            "max_acceptable_option_spread": 0.20,
                                            "near_spread_dte_max": int(spread["near_spread_dte_max"]),
                                            "near_max_acceptable_option_spread": float(spread["near_max_acceptable_option_spread"]),
                                            "mid_max_acceptable_option_spread": float(spread["mid_max_acceptable_option_spread"]),
                                            "far_spread_dte_min": int(spread["far_spread_dte_min"]),
                                            "far_max_acceptable_option_spread": float(spread["far_max_acceptable_option_spread"]),
                                            "spread_profile": str(spread["spread_profile"]),
                                            "per_contract_slippage": 0.075,
                                            "stress_profile": "none",
                                        }
                                    )
    return rows


def config_name(params: dict[str, object]) -> str:
    ladder = "-".join(str(int(value)) for value in params["layer_dte_targets"])
    weekdays = "-".join(str(value) for value in params.get("skip_entry_weekdays", ())) or "none"
    return "_".join(
        [
            str(params["bar_mode"]),
            str(params["profile_branch"]),
            str(params["profile_dte"]),
            f"ladder_{ladder}",
            f"maxlayers_{int(params['max_active_butterfly_layers'])}",
            f"spread_{params['spread_profile']}",
            f"wdays_{weekdays}",
            f"end_{str(params['primary_entry_end']).replace(':', '')}",
            f"conf_{str(params['primary_entry_min_center_confidence']).replace('.', 'p')}",
            f"mom_{str(params['primary_entry_max_momentum_pct']).replace('.', 'p')}",
            f"hold_{int(params['max_hold_sessions'])}",
            f"dteclose_{int(params['close_when_dte_lte'])}",
        ]
    )


def stress_params(params: dict[str, object]) -> dict[str, object]:
    mutated = dict(params)
    mutated["stress_profile"] = "conservative"
    mutated["per_contract_slippage"] = max(0.10, float(mutated.get("per_contract_slippage", 0.0) or 0.0))
    return mutated


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
    print(f"Running {len(tasks)} weekday-filtered refinement cases with workers={max(1, int(args.workers))}.")
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
            "best_weekday_none": active[active["weekday_profile"] == "none"],
            "best_weekday_fri": active[active["weekday_profile"] == "fri"],
            "best_weekday_mon_fri": active[active["weekday_profile"] == "mon_fri"],
            "best_weekday_tue_wed_only": active[active["weekday_profile"] == "tue_wed_only"],
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

    stress_rows: list[dict[str, object]] = []
    stress_candidates: list[dict[str, object]] = []
    if not active.empty and int(args.stress_top_k) > 0:
        top_score = active.head(int(args.stress_top_k))
        top_net = active.sort_values(["net_dollar_pnl", "closed_layers"], ascending=[False, False]).head(int(args.stress_top_k))
        seen_names: set[str] = set()
        for _, row in pd.concat([top_score, top_net], ignore_index=True).iterrows():
            row_dict = row.to_dict()
            config_name_value = str(row_dict["config_name"])
            if config_name_value in seen_names:
                continue
            seen_names.add(config_name_value)
            base_params = next(params for params in params_list if config_name(params) == config_name_value)
            stress_candidates.append(stress_params(base_params))

    if stress_candidates:
        stress_output_dir = output_dir / "stress_retests"
        stress_output_dir.mkdir(parents=True, exist_ok=True)
        stress_tasks = [
            (config_name(params) + "_stress", params, str(stress_output_dir))
            for params in stress_candidates
        ]
        print(f"Running {len(stress_tasks)} conservative stress re-tests.")
        with ProcessPoolExecutor(
            max_workers=max(1, min(int(args.workers), len(stress_tasks))),
            initializer=init_worker,
            initargs=(
                str(bars_csv),
                str(historical_chain_path),
                str(args.start),
                str(args.end),
                int(args.min_active_layers),
            ),
        ) as pool:
            future_map = {pool.submit(run_one, task): task[0] for task in stress_tasks}
            for idx, future in enumerate(as_completed(future_map), start=1):
                row = future.result()
                stress_rows.append(row)
                print(
                    f"[stress {idx}/{len(stress_tasks)}] {row['config_name']} | "
                    f"pnl={float(row['net_dollar_pnl']):.2f} | "
                    f"dd={float(row['max_drawdown_dollars']):.2f} | "
                    f"pf_day={float(row['profit_factor_by_day']):.3f}"
                )

        stress_frame = sort_frame(pd.DataFrame(stress_rows))
        stress_summary = output_dir / "stress_top_summary.csv"
        stress_frame.to_csv(stress_summary, index=False)
        if not stress_frame.empty:
            best_stress = stress_frame.iloc[0].to_dict()
            best_stress_name = str(best_stress["config_name"]).removesuffix("_stress")
            params = next(params for params in stress_candidates if config_name(params) == best_stress_name)
            materialize_run(
                bars_csv=bars_csv,
                historical_chain_path=historical_chain_path,
                start=str(args.start),
                end=str(args.end),
                params=params,
                destination=output_dir / "best_stress_active",
            )
            (output_dir / "best_stress_active" / "rank_row.json").write_text(
                json.dumps(best_stress, indent=2, default=str),
                encoding="utf-8",
            )

    print(f"Done. Summary written to {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
