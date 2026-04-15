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
MIN_ACTIVE_LAYERS: int = 15


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a targeted historical-chain search around the best daily-close and intraday branches "
            "from axis2, with more emphasis on exit rules and entry-filter refinement."
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
        default=r".\corridor_outputs\fit_search\historical_chain_axis3_search",
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
        "--branch",
        default="both",
        choices=["both", "intraday", "daily"],
        help="Limit the search to a subset of tasks.",
    )
    parser.add_argument(
        "--min-active-layers",
        type=int,
        default=15,
        help="Minimum closed layers used by the ranking score to treat a run as active.",
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
        butterfly_width=5.0,
        wing_mode="symmetric",
        broken_wing_extra_width=0.0,
        option_right_preference="call",
        coverage_band_width=25.0,
        center_tolerance=15.0,
        center_tolerance_atr_multiplier=0.0,
        atr_lookback=14,
        recenter_threshold=25.0,
        drift_persistence_bars=1,
        rebuild_cooldown_minutes=0,
        max_active_butterfly_layers=1,
        primary_entry_end="15:30",
        primary_entry_min_center_confidence=0.0,
        primary_entry_max_momentum_pct=0.0175,
        primary_entry_max_volume_ratio=999.0,
        skip_event_days=False,
        skip_gap_days=False,
        max_entry_gap_pct=0.0,
        primary_stop_loss_pct=0.0,
        primary_take_profit_pct=0.0,
        hold_overnight=True,
        max_hold_sessions=2,
        close_when_dte_lte=1,
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
        cfg.center_tolerance = 10.0
        cfg.coverage_band_width = 20.0
        cfg.recenter_threshold = 20.0
        cfg.drift_persistence_bars = 8
        cfg.rebuild_cooldown_minutes = 120
        cfg.hold_overnight = False
        cfg.max_hold_sessions = 0
        cfg.close_when_dte_lte = 0
        cfg.primary_entry_end = "10:30"
    elif bar_mode == "daily_close":
        cfg.timeframe = "1 day"
        cfg.center_lookback = 5
        cfg.regime_lookback = 12
        cfg.range_width_threshold_pct = 0.03
        cfg.trend_slope_threshold_pct = 0.004
        cfg.breakout_buffer_pct = 0.005
        cfg.valid_trading_end = "15:30"
    else:
        raise ValueError(f"Unsupported bar_mode: {bar_mode}")
    return cfg


def intraday_refine_tasks() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dte_profiles = [
        {"profile_dte": "dte_4_6_5", "dte_min": 4, "dte_max": 6, "default_dte": 5},
        {"profile_dte": "dte_4_8_6", "dte_min": 4, "dte_max": 8, "default_dte": 6},
    ]
    width_profiles = [
        {
            "profile_geometry": "w10_tight",
            "butterfly_width": 10.0,
            "coverage_band_width": 20.0,
            "center_tolerance": 10.0,
            "recenter_threshold": 20.0,
            "confidence_values": [0.60, 0.65, 0.70],
            "momentum_values": [0.0015, 0.0017, 0.0020],
            "entry_ends": ["09:50", "10:00", "10:15", "10:30"],
        },
        {
            "profile_geometry": "w5_tight",
            "butterfly_width": 5.0,
            "coverage_band_width": 10.0,
            "center_tolerance": 5.0,
            "recenter_threshold": 10.0,
            "confidence_values": [0.35, 0.45, 0.55],
            "momentum_values": [0.0025, 0.0030, 0.0035],
            "entry_ends": ["10:00", "10:30", "11:00"],
        },
    ]
    stop_values = [0.0, 0.25, 0.50]
    take_values = [0.0, 0.25, 0.50]
    for dte in dte_profiles:
        for width in width_profiles:
            for confidence in width["confidence_values"]:
                for momentum in width["momentum_values"]:
                    for entry_end in width["entry_ends"]:
                        for stop in stop_values:
                            for take in take_values:
                                rows.append(
                                    {
                                        "bar_mode": "intraday_5m",
                                        "profile_branch": "intraday_refine",
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
                                        "hold_overnight": False,
                                        "max_hold_sessions": 0,
                                        "close_when_dte_lte": 0,
                                        "primary_stop_loss_pct": stop,
                                        "primary_take_profit_pct": take,
                                        "stop_label": f"stop_{str(stop).replace('.', 'p')}",
                                        "take_label": f"take_{str(take).replace('.', 'p')}",
                                    }
                                )
    return rows


def daily_exit_tasks() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dte_profiles = [
        {"profile_dte": "dte_4_6_5", "dte_min": 4, "dte_max": 6, "default_dte": 5},
        {"profile_dte": "dte_4_7_6", "dte_min": 4, "dte_max": 7, "default_dte": 6},
    ]
    momentum_values = [0.013, 0.0175, 0.0225]
    hold_sessions = [1, 2, 3]
    close_dte_values = [0, 1]
    stop_values = [0.0, 0.50, 1.00, 2.00]
    take_values = [0.0, 0.25, 0.50, 1.00]
    for dte in dte_profiles:
        for momentum in momentum_values:
            for max_hold in hold_sessions:
                for close_dte in close_dte_values:
                    for stop in stop_values:
                        for take in take_values:
                            rows.append(
                                {
                                    "bar_mode": "daily_close",
                                    "profile_branch": "daily_exit",
                                    **dte,
                                    "profile_geometry": "w5_daily_base",
                                    "butterfly_width": 5.0,
                                    "coverage_band_width": 25.0,
                                    "center_tolerance": 15.0,
                                    "recenter_threshold": 25.0,
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
    return intraday_refine_tasks() + daily_exit_tasks()


def filter_parameter_grid(rows: list[dict[str, object]], branch: str) -> list[dict[str, object]]:
    if branch == "intraday":
        return [row for row in rows if str(row["bar_mode"]) == "intraday_5m"]
    if branch == "daily":
        return [row for row in rows if str(row["bar_mode"]) == "daily_close"]
    return rows


def config_name(params: dict[str, object]) -> str:
    parts = [
        str(params["bar_mode"]),
        str(params["profile_branch"]),
        str(params["profile_dte"]),
        str(params["profile_geometry"]),
        f"conf_{str(params['primary_entry_min_center_confidence']).replace('.', 'p')}",
        f"mom_{str(params['primary_entry_max_momentum_pct']).replace('.', 'p')}",
        f"end_{str(params['primary_entry_end']).replace(':', '')}",
        f"hold_{int(params['max_hold_sessions'])}",
        f"dteclose_{int(params['close_when_dte_lte'])}",
        str(params["stop_label"]),
        str(params["take_label"]),
    ]
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
    if "body_strike_offset_points" in params:
        cfg.body_strike_offset_points = float(params["body_strike_offset_points"])
    cfg.recenter_threshold = float(params["recenter_threshold"])
    cfg.primary_entry_min_center_confidence = float(params["primary_entry_min_center_confidence"])
    cfg.primary_entry_max_momentum_pct = float(params["primary_entry_max_momentum_pct"])
    cfg.primary_entry_end = str(params["primary_entry_end"])
    if "skip_entry_weekdays" in params:
        cfg.skip_entry_weekdays = tuple(str(value) for value in params["skip_entry_weekdays"])  # type: ignore[arg-type]
    cfg.drift_persistence_bars = int(params["drift_persistence_bars"])
    cfg.rebuild_cooldown_minutes = int(params["rebuild_cooldown_minutes"])
    cfg.hold_overnight = bool(params["hold_overnight"])
    cfg.max_hold_sessions = int(params["max_hold_sessions"])
    cfg.close_when_dte_lte = int(params["close_when_dte_lte"])
    cfg.dte_min = int(params["dte_min"])
    cfg.dte_max = int(params["dte_max"])
    cfg.default_dte = int(params["default_dte"])
    if "max_active_butterfly_layers" in params:
        cfg.max_active_butterfly_layers = int(params["max_active_butterfly_layers"])
    if "layer_dte_targets" in params:
        cfg.layer_dte_targets = tuple(int(value) for value in params["layer_dte_targets"])  # type: ignore[arg-type]
        cfg.max_active_butterfly_layers = max(cfg.max_active_butterfly_layers, len(cfg.layer_dte_targets))
    if "layer_exit_scope" in params:
        cfg.layer_exit_scope = str(params["layer_exit_scope"])
    if "allow_daily_entry_additions" in params:
        cfg.allow_daily_entry_additions = bool(params["allow_daily_entry_additions"])
    cfg.primary_stop_loss_pct = float(params["primary_stop_loss_pct"])
    cfg.primary_take_profit_pct = float(params["primary_take_profit_pct"])
    if "block_same_day_reentry_after_take_profit" in params:
        cfg.block_same_day_reentry_after_take_profit = bool(params["block_same_day_reentry_after_take_profit"])
    if "option_right_preference" in params:
        cfg.option_right_preference = str(params["option_right_preference"])
    if "skip_event_days" in params:
        cfg.skip_event_days = bool(params["skip_event_days"])
    if "event_dates" in params:
        cfg.event_dates = tuple(str(value) for value in params["event_dates"])  # type: ignore[arg-type]
    if "skip_gap_days" in params:
        cfg.skip_gap_days = bool(params["skip_gap_days"])
    if "max_entry_gap_pct" in params:
        cfg.max_entry_gap_pct = float(params["max_entry_gap_pct"])
    if "max_acceptable_option_spread" in params:
        cfg.max_acceptable_option_spread = float(params["max_acceptable_option_spread"])
    if "near_spread_dte_max" in params:
        cfg.near_spread_dte_max = int(params["near_spread_dte_max"])
    if "near_max_acceptable_option_spread" in params:
        cfg.near_max_acceptable_option_spread = float(params["near_max_acceptable_option_spread"])
    if "mid_max_acceptable_option_spread" in params:
        cfg.mid_max_acceptable_option_spread = float(params["mid_max_acceptable_option_spread"])
    if "far_spread_dte_min" in params:
        cfg.far_spread_dte_min = int(params["far_spread_dte_min"])
    if "far_max_acceptable_option_spread" in params:
        cfg.far_max_acceptable_option_spread = float(params["far_max_acceptable_option_spread"])
    if "per_contract_slippage" in params:
        cfg.per_contract_slippage = float(params["per_contract_slippage"])
        cfg.slippage = cfg.per_contract_slippage
    if "stress_profile" in params:
        cfg.stress_profile = str(params["stress_profile"])
        _apply_stress_profile(cfg)
    return cfg


def _apply_stress_profile(cfg: CorridorConfig) -> None:
    if cfg.stress_profile == "conservative":
        cfg.stress_entry_debit_multiplier = 1.2
        cfg.stress_peak_value_multiplier = 0.7
        cfg.stress_residual_floor_multiplier = 0.5
        cfg.stress_slippage_multiplier = 2.0
        cfg.stress_close_value_haircut_pct = 0.15
        return
    cfg.stress_entry_debit_multiplier = 1.0
    cfg.stress_peak_value_multiplier = 1.0
    cfg.stress_residual_floor_multiplier = 1.0
    cfg.stress_slippage_multiplier = 1.0
    cfg.stress_close_value_haircut_pct = 0.0


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
    avg_closed = float(summary.get("average_closed_layer_pnl") or 0.0)
    execution_filtered_entries = int(summary.get("execution_filtered_entries") or 0)
    activity_shortfall = max(0, int(MIN_ACTIVE_LAYERS) - closed_layers)
    activity_penalty = float(activity_shortfall * 250.0)
    score = (
        net_dollar_pnl
        + (pf_day * 5000.0)
        + (pf_layer * 3500.0)
        + (win_rate * 1500.0)
        + (avg_closed * 250.0)
        - (abs(max_drawdown_dollars) * 0.15)
        - (rebuilds * 200.0)
        - (execution_filtered_entries * 25.0)
        - activity_penalty
    )
    is_active = closed_layers >= int(MIN_ACTIVE_LAYERS)
    return {
        "config_name": name,
        "bar_mode": str(params["bar_mode"]),
        "profile_branch": str(params["profile_branch"]),
        "profile_dte": str(params["profile_dte"]),
        "profile_geometry": str(params["profile_geometry"]),
        "spread_profile": str(params.get("spread_profile", "")),
        "weekday_profile": str(params.get("weekday_profile", "")),
        "dte_min": int(params["dte_min"]),
        "dte_max": int(params["dte_max"]),
        "default_dte": int(params["default_dte"]),
        "layer_dte_targets": ",".join(str(int(value)) for value in params.get("layer_dte_targets", ())),
        "layer_exit_scope": str(params.get("layer_exit_scope", "all")),
        "allow_daily_entry_additions": bool(params.get("allow_daily_entry_additions", False)),
        "max_active_butterfly_layers": int(params.get("max_active_butterfly_layers", 1)),
        "butterfly_width": float(params["butterfly_width"]),
        "coverage_band_width": float(params["coverage_band_width"]),
        "center_tolerance": float(params["center_tolerance"]),
        "body_strike_offset_points": float(params.get("body_strike_offset_points", 0.0)),
        "recenter_threshold": float(params["recenter_threshold"]),
        "primary_entry_end": str(params["primary_entry_end"]),
        "skip_entry_weekdays": ",".join(str(value) for value in params.get("skip_entry_weekdays", ())),
        "primary_entry_min_center_confidence": float(params["primary_entry_min_center_confidence"]),
        "primary_entry_max_momentum_pct": float(params["primary_entry_max_momentum_pct"]),
        "option_right_preference": str(params.get("option_right_preference", "call")),
        "event_dates": tuple(str(value) for value in params.get("event_dates", ())),
        "center_lookback": int(params["center_lookback"]),
        "regime_lookback": int(params["regime_lookback"]),
        "range_width_threshold_pct": float(params["range_width_threshold_pct"]),
        "trend_slope_threshold_pct": float(params["trend_slope_threshold_pct"]),
        "breakout_buffer_pct": float(params["breakout_buffer_pct"]),
        "drift_persistence_bars": int(params["drift_persistence_bars"]),
        "rebuild_cooldown_minutes": int(params["rebuild_cooldown_minutes"]),
        "hold_overnight": bool(params["hold_overnight"]),
        "skip_event_days": bool(params.get("skip_event_days", False)),
        "skip_gap_days": bool(params.get("skip_gap_days", False)),
        "max_entry_gap_pct": float(params.get("max_entry_gap_pct", 0.0)),
        "max_hold_sessions": int(params["max_hold_sessions"]),
        "close_when_dte_lte": int(params["close_when_dte_lte"]),
        "primary_stop_loss_pct": float(params["primary_stop_loss_pct"]),
        "primary_take_profit_pct": float(params["primary_take_profit_pct"]),
        "block_same_day_reentry_after_take_profit": bool(params.get("block_same_day_reentry_after_take_profit", False)),
        "max_acceptable_option_spread": float(params.get("max_acceptable_option_spread", 0.25)),
        "near_spread_dte_max": int(params.get("near_spread_dte_max", 0)),
        "near_max_acceptable_option_spread": float(params.get("near_max_acceptable_option_spread", 0.0)),
        "mid_max_acceptable_option_spread": float(params.get("mid_max_acceptable_option_spread", 0.0)),
        "far_spread_dte_min": int(params.get("far_spread_dte_min", 0)),
        "far_max_acceptable_option_spread": float(params.get("far_max_acceptable_option_spread", 0.0)),
        "per_contract_slippage": float(params.get("per_contract_slippage", 0.05)),
        "stress_profile": str(params.get("stress_profile", "none")),
        "net_dollar_pnl": net_dollar_pnl,
        "return_on_capital": float(summary.get("return_on_capital") or 0.0),
        "net_modeled_pnl": float(summary.get("net_modeled_pnl") or 0.0),
        "closed_layers": closed_layers,
        "winning_layers": int(summary.get("winning_layers") or 0),
        "losing_layers": int(summary.get("losing_layers") or 0),
        "win_rate_by_closed_layer": win_rate,
        "average_closed_layer_pnl": avg_closed,
        "profit_factor_by_closed_layer": pf_layer,
        "profit_factor_by_day": pf_day,
        "corridor_occupancy_rate": float(summary.get("corridor_occupancy_rate") or 0.0),
        "average_rebuilds_per_day": rebuilds,
        "execution_filtered_entries": execution_filtered_entries,
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
            "profit_factor_by_day",
            "profit_factor_by_closed_layer",
            "average_closed_layer_pnl",
            "max_drawdown_dollars",
        ],
        ascending=[False, False, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def write_rank_outputs(output_dir: Path, frame: pd.DataFrame) -> None:
    summary_csv = output_dir / "summary.csv"
    top10_json = output_dir / "top10.json"
    active_csv = output_dir / "summary_active.csv"
    best_by_mode_json = output_dir / "best_by_mode.json"
    best_by_net_json = output_dir / "best_by_net_active.json"
    ordered = sort_frame(frame)
    ordered.to_csv(summary_csv, index=False)
    ordered.head(10).to_json(top10_json, orient="records", indent=2)
    active = ordered.loc[ordered["is_active_fit"] == True].copy()  # noqa: E712
    active.to_csv(active_csv, index=False)
    by_mode: dict[str, dict[str, object]] = {}
    by_net: dict[str, dict[str, object]] = {}
    for mode in ["daily_close", "intraday_5m"]:
        subset = active[active["bar_mode"] == mode]
        if not subset.empty:
            by_mode[mode] = subset.iloc[0].to_dict()
            by_net[mode] = subset.sort_values(["net_dollar_pnl", "closed_layers"], ascending=[False, False]).iloc[0].to_dict()
    best_by_mode_json.write_text(json.dumps(by_mode, indent=2, default=str), encoding="utf-8")
    best_by_net_json.write_text(json.dumps(by_net, indent=2, default=str), encoding="utf-8")


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
    frame = daily if str(params["bar_mode"]) == "daily_close" else intraday
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
    (output_dir / "runs").mkdir(parents=True, exist_ok=True)

    if not bars_csv.exists():
        raise FileNotFoundError(f"Bars CSV not found: {bars_csv}")
    if not historical_chain_path.exists():
        raise FileNotFoundError(f"Historical-chain dataset not found: {historical_chain_path}")

    params_list = filter_parameter_grid(parameter_grid(), str(args.branch))
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
