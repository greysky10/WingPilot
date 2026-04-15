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
from corridor.report.summary import save_backtest_outputs


FRAME_BY_MODE: dict[str, pd.DataFrame] = {}
HISTORICAL_CHAIN_PATH: Path | None = None
MIN_ACTIVE_LAYERS: int = 10


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a broader SPX historical-chain axis set around width, entry filters, and a daily-close "
            "state-machine variant aligned to daily option marks."
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
        default=r".\corridor_outputs\fit_search\historical_chain_axis2_search",
        help="Directory for summaries and per-run configs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(12, (os.cpu_count() or 4) - 1)),
        help="Parallel worker count.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap for smoke-testing the grid. Set 0 to run the full search.",
    )
    parser.add_argument(
        "--min-active-layers",
        type=int,
        default=10,
        help="Minimum closed layers for a run to rank as an active strategy fit.",
    )
    return parser.parse_args(argv)


def load_intraday_frame(bars_csv: Path, start: str, end: str) -> pd.DataFrame:
    window_start = pd.Timestamp(start, tz="UTC")
    window_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    frame = pd.read_csv(bars_csv)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame[
        (frame["symbol"] == "SPX")
        & (frame["timestamp"] >= window_start)
        & (frame["timestamp"] < window_end)
    ].copy()
    return frame.sort_values("timestamp").reset_index(drop=True)


def build_daily_close_frame(frame: pd.DataFrame) -> pd.DataFrame:
    local_ts = frame["timestamp"].dt.tz_convert("America/New_York")
    daily = (
        frame.assign(local_date=local_ts.dt.strftime("%Y-%m-%d"))
        .groupby(["symbol", "local_date"], as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
    )
    daily["timestamp"] = (
        pd.to_datetime(daily["local_date"] + " 15:30")
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    daily = daily[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
    return daily.sort_values("timestamp").reset_index(drop=True)


def init_worker(
    bars_csv: str,
    historical_chain_path: str,
    start: str,
    end: str,
    min_active_layers: int,
) -> None:
    global FRAME_BY_MODE, HISTORICAL_CHAIN_PATH, MIN_ACTIVE_LAYERS
    bars_path = Path(bars_csv).resolve()
    intraday = load_intraday_frame(bars_path, start, end)
    daily = build_daily_close_frame(intraday)
    FRAME_BY_MODE = {
        "intraday_5m": intraday,
        "daily_close": daily,
    }
    HISTORICAL_CHAIN_PATH = Path(historical_chain_path).resolve()
    MIN_ACTIVE_LAYERS = int(min_active_layers)


def base_config(historical_chain_path: Path, bar_mode: str) -> CorridorConfig:
    cfg = CorridorConfig(
        symbol="SPX",
        center_method=CenterMethod.VWAP,
        center_rounding=default_center_rounding_for_symbol("SPX"),
        payoff_mode="historical_chain",
        historical_chain_path=str(historical_chain_path),
        starting_capital=100000.0,
        contracts_per_layer=1,
        option_multiplier=100,
        max_acceptable_option_spread=0.25,
        per_contract_slippage=0.05,
        slippage=0.05,
        butterfly_width=10.0,
        wing_mode="symmetric",
        broken_wing_extra_width=0.0,
        coverage_band_width=20.0,
        center_tolerance=10.0,
        center_tolerance_atr_multiplier=0.0,
        atr_lookback=14,
        recenter_threshold=20.0,
        drift_persistence_bars=8,
        rebuild_cooldown_minutes=120,
        max_active_butterfly_layers=1,
        primary_entry_end="11:30",
        primary_entry_min_center_confidence=0.0,
        primary_entry_max_momentum_pct=1.0,
        primary_entry_max_volume_ratio=999.0,
        primary_stop_loss_pct=0.0,
        primary_take_profit_pct=0.0,
        hold_overnight=False,
        max_hold_sessions=0,
        close_when_dte_lte=0,
        dte_min=4,
        dte_max=6,
        default_dte=5,
    )
    if bar_mode == "intraday_5m":
        cfg.timeframe = "5 mins"
        cfg.center_lookback = 36
        cfg.regime_lookback = 48
        cfg.range_width_threshold_pct = 0.012
        cfg.trend_slope_threshold_pct = 0.0015
        cfg.breakout_buffer_pct = 0.0025
    elif bar_mode == "daily_close":
        cfg.timeframe = "1 day"
        cfg.center_lookback = 8
        cfg.regime_lookback = 16
        cfg.range_width_threshold_pct = 0.03
        cfg.trend_slope_threshold_pct = 0.0045
        cfg.breakout_buffer_pct = 0.006
        cfg.primary_entry_end = "15:30"
        cfg.hold_overnight = True
        cfg.max_hold_sessions = 2
        cfg.close_when_dte_lte = 1
        cfg.drift_persistence_bars = 1
        cfg.rebuild_cooldown_minutes = 0
        cfg.valid_trading_end = "15:30"
    else:
        raise ValueError(f"Unsupported bar_mode: {bar_mode}")
    return cfg


def intraday_geometry_profiles() -> list[dict[str, object]]:
    return [
        {
            "profile_geometry": "w5_tight",
            "butterfly_width": 5.0,
            "coverage_band_width": 10.0,
            "center_tolerance": 5.0,
            "recenter_threshold": 10.0,
        },
        {
            "profile_geometry": "w5_loose",
            "butterfly_width": 5.0,
            "coverage_band_width": 15.0,
            "center_tolerance": 7.5,
            "recenter_threshold": 15.0,
        },
        {
            "profile_geometry": "w10_tight",
            "butterfly_width": 10.0,
            "coverage_band_width": 20.0,
            "center_tolerance": 10.0,
            "recenter_threshold": 20.0,
        },
        {
            "profile_geometry": "w10_loose",
            "butterfly_width": 10.0,
            "coverage_band_width": 30.0,
            "center_tolerance": 15.0,
            "recenter_threshold": 30.0,
        },
    ]


def intraday_filter_profiles() -> list[dict[str, object]]:
    return [
        {
            "profile_filter": "loose",
            "primary_entry_min_center_confidence": 0.0,
            "primary_entry_max_momentum_pct": 1.0,
        },
        {
            "profile_filter": "conf25_mom45bp",
            "primary_entry_min_center_confidence": 0.25,
            "primary_entry_max_momentum_pct": 0.0045,
        },
        {
            "profile_filter": "conf45_mom30bp",
            "primary_entry_min_center_confidence": 0.45,
            "primary_entry_max_momentum_pct": 0.0030,
        },
        {
            "profile_filter": "conf65_mom17bp",
            "primary_entry_min_center_confidence": 0.65,
            "primary_entry_max_momentum_pct": 0.0017,
        },
    ]


def intraday_entry_end_profiles() -> list[dict[str, object]]:
    return [
        {"profile_entry": "end_1030", "primary_entry_end": "10:30"},
        {"profile_entry": "end_1100", "primary_entry_end": "11:00"},
        {"profile_entry": "end_1130", "primary_entry_end": "11:30"},
        {"profile_entry": "end_1200", "primary_entry_end": "12:00"},
    ]


def daily_geometry_profiles() -> list[dict[str, object]]:
    return [
        {
            "profile_geometry": "w5_daily_base",
            "butterfly_width": 5.0,
            "coverage_band_width": 25.0,
            "center_tolerance": 15.0,
            "recenter_threshold": 25.0,
        },
        {
            "profile_geometry": "w5_daily_wide",
            "butterfly_width": 5.0,
            "coverage_band_width": 35.0,
            "center_tolerance": 20.0,
            "recenter_threshold": 35.0,
        },
        {
            "profile_geometry": "w10_daily_base",
            "butterfly_width": 10.0,
            "coverage_band_width": 30.0,
            "center_tolerance": 20.0,
            "recenter_threshold": 30.0,
        },
        {
            "profile_geometry": "w10_daily_wide",
            "butterfly_width": 10.0,
            "coverage_band_width": 40.0,
            "center_tolerance": 25.0,
            "recenter_threshold": 40.0,
        },
    ]


def daily_filter_profiles() -> list[dict[str, object]]:
    return [
        {"profile_filter": "mom_off", "primary_entry_min_center_confidence": 0.0, "primary_entry_max_momentum_pct": 1.0},
        {"profile_filter": "mom_250bp", "primary_entry_min_center_confidence": 0.0, "primary_entry_max_momentum_pct": 0.025},
        {"profile_filter": "mom_175bp", "primary_entry_min_center_confidence": 0.0, "primary_entry_max_momentum_pct": 0.0175},
        {"profile_filter": "mom_110bp", "primary_entry_min_center_confidence": 0.0, "primary_entry_max_momentum_pct": 0.011},
    ]


def intraday_dte_profiles() -> list[dict[str, object]]:
    return [
        {"profile_dte": "dte_4_6_5", "dte_min": 4, "dte_max": 6, "default_dte": 5},
        {"profile_dte": "dte_4_8_6", "dte_min": 4, "dte_max": 8, "default_dte": 6},
    ]


def daily_dte_profiles() -> list[dict[str, object]]:
    return [
        {"profile_dte": "dte_4_6_5", "dte_min": 4, "dte_max": 6, "default_dte": 5},
        {"profile_dte": "dte_6_10_8", "dte_min": 6, "dte_max": 10, "default_dte": 8},
    ]


def daily_execution_profiles() -> list[dict[str, object]]:
    return [
        {
            "profile_execution": "daily_fast_1d",
            "center_lookback": 5,
            "regime_lookback": 10,
            "range_width_threshold_pct": 0.03,
            "trend_slope_threshold_pct": 0.004,
            "breakout_buffer_pct": 0.005,
            "drift_persistence_bars": 1,
            "rebuild_cooldown_minutes": 0,
            "hold_overnight": True,
            "max_hold_sessions": 1,
            "close_when_dte_lte": 1,
        },
        {
            "profile_execution": "daily_fast_2d",
            "center_lookback": 5,
            "regime_lookback": 12,
            "range_width_threshold_pct": 0.03,
            "trend_slope_threshold_pct": 0.004,
            "breakout_buffer_pct": 0.005,
            "drift_persistence_bars": 1,
            "rebuild_cooldown_minutes": 0,
            "hold_overnight": True,
            "max_hold_sessions": 2,
            "close_when_dte_lte": 1,
        },
        {
            "profile_execution": "daily_strict_1d",
            "center_lookback": 8,
            "regime_lookback": 16,
            "range_width_threshold_pct": 0.025,
            "trend_slope_threshold_pct": 0.0035,
            "breakout_buffer_pct": 0.0045,
            "drift_persistence_bars": 1,
            "rebuild_cooldown_minutes": 0,
            "hold_overnight": True,
            "max_hold_sessions": 1,
            "close_when_dte_lte": 1,
        },
        {
            "profile_execution": "daily_strict_2d",
            "center_lookback": 8,
            "regime_lookback": 16,
            "range_width_threshold_pct": 0.025,
            "trend_slope_threshold_pct": 0.0035,
            "breakout_buffer_pct": 0.0045,
            "drift_persistence_bars": 2,
            "rebuild_cooldown_minutes": 0,
            "hold_overnight": True,
            "max_hold_sessions": 2,
            "close_when_dte_lte": 1,
        },
    ]


def parameter_grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dte in intraday_dte_profiles():
        for geometry in intraday_geometry_profiles():
            for filters in intraday_filter_profiles():
                for entry in intraday_entry_end_profiles():
                    rows.append(
                        {
                            "bar_mode": "intraday_5m",
                            "profile_execution": "intraday_fixed_tol",
                            **dte,
                            **geometry,
                            **filters,
                            **entry,
                            "center_lookback": 36,
                            "regime_lookback": 48,
                            "range_width_threshold_pct": 0.012,
                            "trend_slope_threshold_pct": 0.0015,
                            "breakout_buffer_pct": 0.0025,
                            "center_tolerance_atr_multiplier": 0.0,
                            "drift_persistence_bars": 8,
                            "rebuild_cooldown_minutes": 120,
                            "hold_overnight": False,
                            "max_hold_sessions": 0,
                            "close_when_dte_lte": 0,
                        }
                    )

    for dte in daily_dte_profiles():
        for execution in daily_execution_profiles():
            for geometry in daily_geometry_profiles():
                for filters in daily_filter_profiles():
                    rows.append(
                        {
                            "bar_mode": "daily_close",
                            **dte,
                            **execution,
                            **geometry,
                            **filters,
                            "primary_entry_end": "15:30",
                            "center_tolerance_atr_multiplier": 0.0,
                        }
                    )
    return rows


def config_name(params: dict[str, object]) -> str:
    parts = [
        str(params["bar_mode"]),
        str(params["profile_execution"]),
        str(params["profile_dte"]),
        str(params["profile_geometry"]),
        str(params["profile_filter"]),
    ]
    if params["bar_mode"] == "intraday_5m":
        parts.append(str(params["profile_entry"]))
    return "_".join(parts)


def apply_params(cfg: CorridorConfig, params: dict[str, object], output_dir: Path) -> CorridorConfig:
    cfg.output_dir = output_dir
    cfg.center_lookback = int(params["center_lookback"])
    cfg.regime_lookback = int(params["regime_lookback"])
    cfg.range_width_threshold_pct = float(params["range_width_threshold_pct"])
    cfg.trend_slope_threshold_pct = float(params["trend_slope_threshold_pct"])
    cfg.breakout_buffer_pct = float(params["breakout_buffer_pct"])
    cfg.butterfly_width = float(params["butterfly_width"])
    cfg.coverage_band_width = float(params["coverage_band_width"])
    cfg.center_tolerance = float(params["center_tolerance"])
    cfg.center_tolerance_atr_multiplier = float(params["center_tolerance_atr_multiplier"])
    cfg.recenter_threshold = float(params["recenter_threshold"])
    cfg.primary_entry_end = str(params["primary_entry_end"])
    cfg.primary_entry_min_center_confidence = float(params["primary_entry_min_center_confidence"])
    cfg.primary_entry_max_momentum_pct = float(params["primary_entry_max_momentum_pct"])
    cfg.primary_entry_max_volume_ratio = 999.0
    cfg.drift_persistence_bars = int(params["drift_persistence_bars"])
    cfg.rebuild_cooldown_minutes = int(params["rebuild_cooldown_minutes"])
    cfg.hold_overnight = bool(params["hold_overnight"])
    cfg.max_hold_sessions = int(params["max_hold_sessions"])
    cfg.close_when_dte_lte = int(params["close_when_dte_lte"])
    cfg.dte_min = int(params["dte_min"])
    cfg.dte_max = int(params["dte_max"])
    cfg.default_dte = int(params["default_dte"])
    return cfg


def frame_for_mode(bar_mode: str) -> pd.DataFrame:
    frame = FRAME_BY_MODE.get(bar_mode)
    if frame is None:
        raise RuntimeError(f"Frame not loaded for bar_mode={bar_mode}")
    if frame.empty:
        raise RuntimeError(f"Frame is empty for bar_mode={bar_mode}")
    return frame


def run_one(task: tuple[str, dict[str, object], str]) -> dict[str, object]:
    global HISTORICAL_CHAIN_PATH
    if HISTORICAL_CHAIN_PATH is None:
        raise RuntimeError("Worker was not initialized.")

    name, params, output_root = task
    run_dir = Path(output_root) / "runs" / name
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    frame = frame_for_mode(str(params["bar_mode"]))

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        cfg = base_config(HISTORICAL_CHAIN_PATH, str(params["bar_mode"]))
        cfg = apply_params(cfg, params, run_dir)
        result = CorridorBacktestEngine(cfg).run(frame)
        summary = result.summary
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        config_path.write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    return summarize_row(name, params, summary)


def summarize_row(name: str, params: dict[str, object], summary: dict[str, object]) -> dict[str, object]:
    multiplier = float(summary.get("option_multiplier") or 100.0)
    contracts = float(summary.get("contracts_per_layer") or 1.0)
    max_drawdown_points = float(summary.get("max_drawdown") or 0.0)
    max_drawdown_dollars = max_drawdown_points * multiplier * contracts
    net_dollar_pnl = float(summary.get("net_dollar_pnl") or 0.0)
    rebuilds = float(summary.get("average_rebuilds_per_day") or 0.0)
    win_rate = float(summary.get("win_rate_by_closed_layer") or 0.0)
    pf_day = float(summary.get("profit_factor_by_day") or 0.0)
    pf_layer = float(summary.get("profit_factor_by_closed_layer") or 0.0)
    closed_layers = int(summary.get("closed_layers") or 0)
    activity_shortfall = max(0, int(MIN_ACTIVE_LAYERS) - closed_layers)
    activity_penalty = float(activity_shortfall * 1500.0)
    score = (
        net_dollar_pnl
        + (max_drawdown_dollars * 0.20)
        + (pf_day * 4000.0)
        + (pf_layer * 2500.0)
        + (win_rate * 1000.0)
        - (rebuilds * 200.0)
        - activity_penalty
    )
    is_active = closed_layers >= int(MIN_ACTIVE_LAYERS)
    return {
        "config_name": name,
        "bar_mode": str(params["bar_mode"]),
        "profile_execution": str(params["profile_execution"]),
        "profile_dte": str(params["profile_dte"]),
        "profile_geometry": str(params["profile_geometry"]),
        "profile_filter": str(params["profile_filter"]),
        "profile_entry": str(params.get("profile_entry", "")),
        "dte_min": int(params["dte_min"]),
        "dte_max": int(params["dte_max"]),
        "default_dte": int(params["default_dte"]),
        "butterfly_width": float(params["butterfly_width"]),
        "coverage_band_width": float(params["coverage_band_width"]),
        "center_tolerance": float(params["center_tolerance"]),
        "center_tolerance_atr_multiplier": float(params["center_tolerance_atr_multiplier"]),
        "recenter_threshold": float(params["recenter_threshold"]),
        "primary_entry_end": str(params["primary_entry_end"]),
        "primary_entry_min_center_confidence": float(params["primary_entry_min_center_confidence"]),
        "primary_entry_max_momentum_pct": float(params["primary_entry_max_momentum_pct"]),
        "center_lookback": int(params["center_lookback"]),
        "regime_lookback": int(params["regime_lookback"]),
        "range_width_threshold_pct": float(params["range_width_threshold_pct"]),
        "trend_slope_threshold_pct": float(params["trend_slope_threshold_pct"]),
        "breakout_buffer_pct": float(params["breakout_buffer_pct"]),
        "drift_persistence_bars": int(params["drift_persistence_bars"]),
        "rebuild_cooldown_minutes": int(params["rebuild_cooldown_minutes"]),
        "hold_overnight": bool(params["hold_overnight"]),
        "max_hold_sessions": int(params["max_hold_sessions"]),
        "close_when_dte_lte": int(params["close_when_dte_lte"]),
        "net_dollar_pnl": net_dollar_pnl,
        "return_on_capital": float(summary.get("return_on_capital") or 0.0),
        "net_modeled_pnl": float(summary.get("net_modeled_pnl") or 0.0),
        "closed_layers": closed_layers,
        "winning_layers": int(summary.get("winning_layers") or 0),
        "losing_layers": int(summary.get("losing_layers") or 0),
        "win_rate_by_closed_layer": win_rate,
        "profit_factor_by_closed_layer": pf_layer,
        "profit_factor_by_day": pf_day,
        "corridor_occupancy_rate": float(summary.get("corridor_occupancy_rate") or 0.0),
        "average_rebuilds_per_day": rebuilds,
        "max_drawdown_points": max_drawdown_points,
        "max_drawdown_dollars": max_drawdown_dollars,
        "max_gross_deployment_dollars": float(summary.get("max_gross_deployment_dollars") or 0.0),
        "is_active_fit": bool(is_active),
        "activity_penalty": activity_penalty,
        "score": score,
    }


def sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(
        by=[
            "is_active_fit",
            "score",
            "net_dollar_pnl",
            "return_on_capital",
            "profit_factor_by_day",
            "profit_factor_by_closed_layer",
            "max_drawdown_dollars",
            "average_rebuilds_per_day",
        ],
        ascending=[False, False, False, False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def write_rank_outputs(output_dir: Path, frame: pd.DataFrame) -> None:
    summary_csv = output_dir / "summary.csv"
    top10_json = output_dir / "top10.json"
    best_by_mode_json = output_dir / "best_by_mode.json"
    active_csv = output_dir / "summary_active.csv"
    best_overall_json = output_dir / "best_overall.json"

    ordered = sort_frame(frame)
    ordered.to_csv(summary_csv, index=False)
    ordered.head(10).to_json(top10_json, orient="records", indent=2)

    active = ordered.loc[ordered["is_active_fit"] == True].copy()  # noqa: E712
    active.to_csv(active_csv, index=False)
    if not ordered.empty:
        best_overall_json.write_text(json.dumps(ordered.iloc[0].to_dict(), indent=2, default=str), encoding="utf-8")
    by_mode: dict[str, dict[str, object]] = {}
    for mode in ["intraday_5m", "daily_close"]:
        subset = active[active["bar_mode"] == mode]
        if subset.empty:
            subset = ordered[ordered["bar_mode"] == mode]
        if not subset.empty:
            by_mode[mode] = subset.iloc[0].to_dict()
    best_by_mode_json.write_text(json.dumps(by_mode, indent=2, default=str), encoding="utf-8")


def materialize_run(
    bars_csv: Path,
    historical_chain_path: Path,
    start: str,
    end: str,
    params: dict[str, object],
    destination: Path,
) -> None:
    intraday = load_intraday_frame(bars_csv, start, end)
    daily = build_daily_close_frame(intraday)
    frames = {
        "intraday_5m": intraday,
        "daily_close": daily,
    }
    frame = frames[str(params["bar_mode"])]
    cfg = base_config(historical_chain_path, str(params["bar_mode"]))
    cfg = apply_params(cfg, params, destination)
    result = CorridorBacktestEngine(cfg).run(frame)
    destination.mkdir(parents=True, exist_ok=True)
    save_backtest_outputs(destination, result)
    (destination / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
    (destination / "rank_row.json").write_text(
        json.dumps(summarize_row(destination.name, params, result.summary), indent=2, default=str),
        encoding="utf-8",
    )


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
    if int(args.max_cases) > 0:
        params_list = params_list[: int(args.max_cases)]
    tasks = [(config_name(params), params, str(output_dir)) for params in params_list]
    print(f"Running {len(tasks)} cases with workers={max(1, int(args.workers))}.")
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
                f"dd=${float(row['max_drawdown_dollars']):.2f} | "
                f"layers={int(row['closed_layers'])}"
            )

    final = sort_frame(pd.DataFrame(rows))
    write_rank_outputs(output_dir, final)
    print("MATERIALIZING BEST RUNS")
    if not final.empty:
        best_overall = final.iloc[0].to_dict()
        materialize_run(
            bars_csv=bars_csv,
            historical_chain_path=historical_chain_path,
            start=str(args.start),
            end=str(args.end),
            params=next(params for params in params_list if config_name(params) == best_overall["config_name"]),
            destination=output_dir / "best_overall",
        )
    for mode in ["intraday_5m", "daily_close"]:
        subset = final.loc[(final["bar_mode"] == mode) & (final["is_active_fit"] == True)].copy()  # noqa: E712
        if subset.empty:
            subset = final.loc[final["bar_mode"] == mode].copy()
        if subset.empty:
            continue
        best_row = subset.iloc[0].to_dict()
        materialize_run(
            bars_csv=bars_csv,
            historical_chain_path=historical_chain_path,
            start=str(args.start),
            end=str(args.end),
            params=next(params for params in params_list if config_name(params) == best_row["config_name"]),
            destination=output_dir / f"best_{mode}",
        )
    print(f"Done. Summary written to {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
