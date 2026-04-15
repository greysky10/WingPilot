#! python3.12
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corridor.backtest.engine import CorridorBacktestEngine
from corridor.config import CorridorConfig
from corridor.data.ib_contracts import default_center_rounding_for_symbol
from corridor.models import CenterMethod


FRAME: pd.DataFrame | None = None
BARS_CSV: Path | None = None
HISTORICAL_CHAIN_PATH: Path | None = None
WINDOW_START: pd.Timestamp | None = None
WINDOW_END: pd.Timestamp | None = None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable SPX historical-chain parameter sweep over the finished Massive dataset and "
            "rank configurations by risk-adjusted performance."
        )
    )
    parser.add_argument(
        "--bars-csv",
        default=r".\corridor_outputs\fit_search\SPX_5_mins_bars_20250408_20260409.csv",
        help="Underlying SPX intraday bars CSV.",
    )
    parser.add_argument(
        "--historical-chain-path",
        default=r".\data\massive_spx_strategy_history\spx_options_daily_history.csv",
        help="Historical-chain CSV produced by run_massive_spx_backfill.py.",
    )
    parser.add_argument("--start", default="2025-04-10", help="Inclusive UTC start date.")
    parser.add_argument("--end", default="2026-04-09", help="Inclusive UTC end date.")
    parser.add_argument(
        "--output-dir",
        default=r".\corridor_outputs\fit_search\historical_chain_param_search",
        help="Directory for summaries and per-run configs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(6, (os.cpu_count() or 4) - 1)),
        help="Parallel worker count.",
    )
    return parser.parse_args(argv)


def init_worker(
    bars_csv: str,
    historical_chain_path: str,
    start: str,
    end: str,
) -> None:
    global FRAME, BARS_CSV, HISTORICAL_CHAIN_PATH, WINDOW_START, WINDOW_END
    BARS_CSV = Path(bars_csv).resolve()
    HISTORICAL_CHAIN_PATH = Path(historical_chain_path).resolve()
    WINDOW_START = pd.Timestamp(start, tz="UTC")
    WINDOW_END = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)

    frame = pd.read_csv(BARS_CSV)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame[(frame["symbol"] == "SPX") & (frame["timestamp"] >= WINDOW_START) & (frame["timestamp"] < WINDOW_END)]
    FRAME = frame.sort_values("timestamp").reset_index(drop=True)


def base_config(historical_chain_path: Path) -> CorridorConfig:
    return CorridorConfig(
        symbol="SPX",
        timeframe="5 mins",
        center_method=CenterMethod.VWAP,
        center_rounding=default_center_rounding_for_symbol("SPX"),
        payoff_mode="historical_chain",
        historical_chain_path=str(historical_chain_path),
        starting_capital=100000.0,
        contracts_per_layer=1,
        option_multiplier=100,
        butterfly_width=10.0,
        wing_mode="symmetric",
        broken_wing_extra_width=0.0,
        coverage_band_width=20.0,
        center_tolerance=5.0,
        recenter_threshold=10.0,
        drift_persistence_bars=8,
        rebuild_cooldown_minutes=120,
        max_active_butterfly_layers=1,
        primary_entry_end="12:00",
        primary_entry_min_center_confidence=0.0,
        primary_entry_max_momentum_pct=1.0,
        primary_entry_max_volume_ratio=999.0,
        primary_stop_loss_pct=0.0,
        primary_take_profit_pct=0.0,
        hold_overnight=True,
        max_hold_sessions=1,
        close_when_dte_lte=2,
        dte_min=4,
        dte_max=10,
        default_dte=7,
    )


def dte_profiles() -> list[dict[str, object]]:
    return [
        {"profile_dte": "dte_4_6_5", "dte_min": 4, "dte_max": 6, "default_dte": 5},
        {"profile_dte": "dte_4_8_6", "dte_min": 4, "dte_max": 8, "default_dte": 6},
        {"profile_dte": "dte_4_10_7", "dte_min": 4, "dte_max": 10, "default_dte": 7},
        {"profile_dte": "dte_6_10_8", "dte_min": 6, "dte_max": 10, "default_dte": 8},
    ]


def hold_profiles() -> list[dict[str, object]]:
    return [
        {
            "profile_hold": "intraday_tight",
            "hold_overnight": False,
            "max_hold_sessions": 0,
            "close_when_dte_lte": 0,
            "primary_entry_end": "11:30",
            "drift_persistence_bars": 8,
            "rebuild_cooldown_minutes": 120,
        },
        {
            "profile_hold": "overnight_1",
            "hold_overnight": True,
            "max_hold_sessions": 1,
            "close_when_dte_lte": 2,
            "primary_entry_end": "12:00",
            "drift_persistence_bars": 8,
            "rebuild_cooldown_minutes": 120,
        },
        {
            "profile_hold": "overnight_2",
            "hold_overnight": True,
            "max_hold_sessions": 2,
            "close_when_dte_lte": 2,
            "primary_entry_end": "12:00",
            "drift_persistence_bars": 12,
            "rebuild_cooldown_minutes": 180,
        },
    ]


def tolerance_profiles() -> list[dict[str, object]]:
    return [
        {"profile_tol": "tol5", "center_tolerance": 5.0, "recenter_threshold": 10.0},
        {"profile_tol": "tol10", "center_tolerance": 10.0, "recenter_threshold": 20.0},
    ]


def layer_profiles() -> list[dict[str, object]]:
    return [
        {"profile_layers": "layers1", "max_active_butterfly_layers": 1},
        {"profile_layers": "layers2", "max_active_butterfly_layers": 2},
    ]


def parameter_grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dte in dte_profiles():
        for hold in hold_profiles():
            for tol in tolerance_profiles():
                for layers in layer_profiles():
                    params = {
                        **dte,
                        **hold,
                        **tol,
                        **layers,
                        "butterfly_width": 10.0,
                        "wing_mode": "symmetric",
                        "broken_wing_extra_width": 0.0,
                        "coverage_band_width": 20.0,
                        "primary_stop_loss_pct": 0.0,
                        "primary_take_profit_pct": 0.0,
                    }
                    rows.append(params)
    return rows


def config_name(params: dict[str, object]) -> str:
    return (
        f"{params['profile_dte']}_"
        f"{params['profile_hold']}_"
        f"{params['profile_tol']}_"
        f"{params['profile_layers']}"
    )


def run_one(task: tuple[str, dict[str, object], str]) -> dict[str, object]:
    global FRAME, HISTORICAL_CHAIN_PATH
    if FRAME is None or HISTORICAL_CHAIN_PATH is None:
        raise RuntimeError("Worker was not initialized.")

    name, params, output_root = task
    run_dir = Path(output_root) / "runs" / name
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    frame = FRAME
    if frame.empty:
        raise RuntimeError("Historical bars frame is empty.")

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        cfg = base_config(HISTORICAL_CHAIN_PATH)
        cfg.output_dir = run_dir
        cfg.dte_min = int(params["dte_min"])
        cfg.dte_max = int(params["dte_max"])
        cfg.default_dte = int(params["default_dte"])
        cfg.hold_overnight = bool(params["hold_overnight"])
        cfg.max_hold_sessions = int(params["max_hold_sessions"])
        cfg.close_when_dte_lte = int(params["close_when_dte_lte"])
        cfg.primary_entry_end = str(params["primary_entry_end"])
        cfg.drift_persistence_bars = int(params["drift_persistence_bars"])
        cfg.rebuild_cooldown_minutes = int(params["rebuild_cooldown_minutes"])
        cfg.center_tolerance = float(params["center_tolerance"])
        cfg.recenter_threshold = float(params["recenter_threshold"])
        cfg.max_active_butterfly_layers = int(params["max_active_butterfly_layers"])
        cfg.butterfly_width = float(params["butterfly_width"])
        cfg.wing_mode = str(params["wing_mode"])
        cfg.broken_wing_extra_width = float(params["broken_wing_extra_width"])
        cfg.coverage_band_width = float(params["coverage_band_width"])
        cfg.primary_stop_loss_pct = float(params["primary_stop_loss_pct"])
        cfg.primary_take_profit_pct = float(params["primary_take_profit_pct"])

        result = CorridorBacktestEngine(cfg).run(frame)
        summary = result.summary
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        config_path.write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    return summarize_row(name, params, summary)


def summarize_row(
    name: str,
    params: dict[str, object],
    summary: dict[str, object],
) -> dict[str, object]:
    multiplier = float(summary.get("option_multiplier") or 100.0)
    contracts = float(summary.get("contracts_per_layer") or 1.0)
    max_drawdown_points = float(summary.get("max_drawdown") or 0.0)
    max_drawdown_dollars = max_drawdown_points * multiplier * contracts
    net_dollar_pnl = float(summary.get("net_dollar_pnl") or 0.0)
    occupancy = float(summary.get("corridor_occupancy_rate") or 0.0)
    rebuilds = float(summary.get("average_rebuilds_per_day") or 0.0)
    win_rate = float(summary.get("win_rate_by_closed_layer") or 0.0)
    score = net_dollar_pnl + (max_drawdown_dollars * 0.35) + (occupancy * 5000.0) - (rebuilds * 250.0) + (win_rate * 2500.0)

    return {
        "config_name": name,
        "profile_dte": params["profile_dte"],
        "profile_hold": params["profile_hold"],
        "profile_tol": params["profile_tol"],
        "profile_layers": params["profile_layers"],
        "dte_min": int(params["dte_min"]),
        "dte_max": int(params["dte_max"]),
        "default_dte": int(params["default_dte"]),
        "hold_overnight": bool(params["hold_overnight"]),
        "max_hold_sessions": int(params["max_hold_sessions"]),
        "close_when_dte_lte": int(params["close_when_dte_lte"]),
        "primary_entry_end": str(params["primary_entry_end"]),
        "drift_persistence_bars": int(params["drift_persistence_bars"]),
        "rebuild_cooldown_minutes": int(params["rebuild_cooldown_minutes"]),
        "center_tolerance": float(params["center_tolerance"]),
        "recenter_threshold": float(params["recenter_threshold"]),
        "max_active_butterfly_layers": int(params["max_active_butterfly_layers"]),
        "butterfly_width": float(params["butterfly_width"]),
        "wing_mode": str(params["wing_mode"]),
        "net_dollar_pnl": net_dollar_pnl,
        "return_on_capital": float(summary.get("return_on_capital") or 0.0),
        "net_modeled_pnl": float(summary.get("net_modeled_pnl") or 0.0),
        "closed_layers": int(summary.get("closed_layers") or 0),
        "winning_layers": int(summary.get("winning_layers") or 0),
        "losing_layers": int(summary.get("losing_layers") or 0),
        "win_rate_by_closed_layer": win_rate,
        "profit_factor_by_closed_layer": float(summary.get("profit_factor_by_closed_layer") or 0.0),
        "profit_factor_by_day": float(summary.get("profit_factor_by_day") or 0.0),
        "corridor_occupancy_rate": occupancy,
        "average_rebuilds_per_day": rebuilds,
        "max_drawdown_points": max_drawdown_points,
        "max_drawdown_dollars": max_drawdown_dollars,
        "max_gross_deployment_dollars": float(summary.get("max_gross_deployment_dollars") or 0.0),
        "max_modeled_capital_at_risk": float(summary.get("max_modeled_capital_at_risk") or 0.0),
        "score": score,
    }


def sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=[
            "score",
            "net_dollar_pnl",
            "return_on_capital",
            "win_rate_by_closed_layer",
            "corridor_occupancy_rate",
            "average_rebuilds_per_day",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    bars_csv = Path(args.bars_csv).resolve()
    historical_chain_path = Path(args.historical_chain_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not bars_csv.exists():
        raise FileNotFoundError(f"Bars CSV not found: {bars_csv}")
    if not historical_chain_path.exists():
        raise FileNotFoundError(f"Historical-chain dataset not found: {historical_chain_path}")

    params_list = parameter_grid()
    tasks = [(config_name(params), params, str(output_dir)) for params in params_list]
    rows: list[dict[str, object]] = []
    summary_csv = output_dir / "summary.csv"
    top10_json = output_dir / "top10.json"
    workers = max(1, int(args.workers))

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(str(bars_csv), str(historical_chain_path), str(args.start), str(args.end)),
    ) as pool:
        future_map = {pool.submit(run_one, task): task[0] for task in tasks}
        for idx, future in enumerate(as_completed(future_map), start=1):
            row = future.result()
            rows.append(row)
            current = sort_frame(pd.DataFrame(rows))
            current.to_csv(summary_csv, index=False)
            current.head(10).to_json(top10_json, orient="records", indent=2)
            print(
                f"[{idx}/{len(tasks)}] {row['config_name']} | "
                f"pnl={float(row['net_dollar_pnl']):.2f} | "
                f"roc={float(row['return_on_capital']):.2%} | "
                f"dd=${float(row['max_drawdown_dollars']):.2f} | "
                f"layers={int(row['closed_layers'])}"
            )

    final = sort_frame(pd.DataFrame(rows))
    final.to_csv(summary_csv, index=False)
    final.head(10).to_json(top10_json, orient="records", indent=2)
    best = final.iloc[0].to_dict()
    print("BEST")
    print(json.dumps(best, default=str))
    print(f"runs={len(final)} summary={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
