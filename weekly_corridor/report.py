from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import WeeklyCorridorConfig
from .models import ActionRecord, BacktestArtifacts, EquityPoint, TransitionRecord, WeeklyActionType


def transitions_to_frame(records: list[TransitionRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": record.timestamp.isoformat(),
                "symbol": record.symbol,
                "week_key": record.week_key,
                "from_state": record.from_state.value,
                "to_state": record.to_state.value,
                "reason": record.reason,
                "regime": record.regime.value,
                "price": record.price,
                "center_price": record.center_price,
                "active_butterflies": record.active_butterflies,
                "adjustments_this_week": record.adjustments_this_week,
            }
            for record in records
        ]
    )


def actions_to_frame(records: list[ActionRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": record.timestamp.isoformat(),
                "symbol": record.symbol,
                "week_key": record.week_key,
                "action": record.action.value,
                "state": record.state.value,
                "price": record.price,
                "center_price": record.center_price,
                "layer_id": record.layer_id,
                "detail": record.detail,
                **record.metadata,
            }
            for record in records
        ]
    )


def equity_to_frame(records: list[EquityPoint]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": record.timestamp.isoformat(),
                "symbol": record.symbol,
                "week_key": record.week_key,
                "price": record.price,
                "regime": record.regime.value,
                "state": record.state.value,
                "bar_pnl": record.bar_pnl,
                "realized_pnl": record.realized_pnl,
                "unrealized_pnl": record.unrealized_pnl,
                "gross_realized_pnl": record.gross_realized_pnl,
                "gross_unrealized_pnl": record.gross_unrealized_pnl,
                "gross_total_equity": record.gross_total_equity,
                "total_equity": record.total_equity,
                "gross_deployment": record.gross_deployment,
                "weekly_occupancy": record.weekly_occupancy,
                "active_butterflies": record.active_butterflies,
            }
            for record in records
        ]
    )


def compute_summary(
    config: WeeklyCorridorConfig,
    actions: list[ActionRecord],
    equity_curve: list[EquityPoint],
) -> dict[str, Any]:
    summary = {
        "symbol": config.symbol,
        "decision_timeframe": config.decision_timeframe,
        "net_modeled_pnl": 0.0,
        "net_dollar_pnl": 0.0,
        "return_on_capital": None,
        "max_gross_deployment_dollars": 0.0,
        "worst_day_pnl_dollars": 0.0,
        "worst_week_pnl_dollars": 0.0,
        "profit_factor_by_closed_layer": None,
        "profit_factor_by_week": None,
        "weekly_occupancy_rate": 0.0,
        "avg_adjustments_per_week": 0.0,
        "weeks_traded": 0,
        "weeks_aborted": 0,
        "max_active_butterflies": 0,
        "closed_layers": 0,
        "winning_layers": 0,
        "losing_layers": 0,
        "metric_definitions": _metric_definitions(),
    }
    if not equity_curve:
        return summary

    equity = equity_to_frame(equity_curve)
    equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
    equity["date"] = equity["timestamp"].dt.tz_convert("America/New_York").dt.date
    equity["week_key"] = equity["week_key"].astype(str)

    actions_frame = actions_to_frame(actions)
    if not actions_frame.empty:
        actions_frame["timestamp"] = pd.to_datetime(actions_frame["timestamp"], utc=True)
        actions_frame["week_key"] = actions_frame["week_key"].astype(str)

    net_modeled_pnl = float(equity["total_equity"].iloc[-1])
    net_dollar_pnl = net_modeled_pnl * config.option_multiplier * config.contracts_per_layer
    return_on_capital = net_dollar_pnl / config.starting_capital if config.starting_capital else None
    max_gross_deployment = float(equity["gross_deployment"].max()) if not equity.empty else 0.0

    day_pnl = equity.groupby("date")["bar_pnl"].sum()
    week_pnl = equity.groupby("week_key")["bar_pnl"].sum()
    worst_day_pnl_dollars = float(day_pnl.min()) * config.option_multiplier * config.contracts_per_layer if not day_pnl.empty else 0.0
    worst_week_pnl_dollars = float(week_pnl.min()) * config.option_multiplier * config.contracts_per_layer if not week_pnl.empty else 0.0

    closed = pd.DataFrame()
    if not actions_frame.empty and "realized_pnl" in actions_frame.columns:
        closed = actions_frame.loc[actions_frame["realized_pnl"].notna()].copy()
        if not closed.empty:
            closed["realized_pnl"] = closed["realized_pnl"].astype(float)

    winners = closed.loc[closed["realized_pnl"] > 0.0, "realized_pnl"] if not closed.empty else pd.Series(dtype=float)
    losers = closed.loc[closed["realized_pnl"] < 0.0, "realized_pnl"] if not closed.empty else pd.Series(dtype=float)
    gross_winners = float(winners.sum()) if len(winners) else 0.0
    gross_losers = float(abs(losers.sum())) if len(losers) else 0.0
    profit_factor_by_closed_layer = gross_winners / gross_losers if gross_losers > 0 else None

    winning_weeks = week_pnl[week_pnl > 0.0]
    losing_weeks = week_pnl[week_pnl < 0.0]
    gross_positive_weeks = float(winning_weeks.sum()) if len(winning_weeks) else 0.0
    gross_negative_weeks = float(abs(losing_weeks.sum())) if len(losing_weeks) else 0.0
    profit_factor_by_week = gross_positive_weeks / gross_negative_weeks if gross_negative_weeks > 0 else None

    adjustment_count = 0
    weeks_traded = 0
    weeks_aborted = 0
    if not actions_frame.empty:
        adjustment_count = int((actions_frame["action"] == WeeklyActionType.ADD_ADJUSTMENT.value).sum())
        weeks_traded = int(actions_frame.loc[actions_frame["action"] == WeeklyActionType.DEPLOY_INITIAL.value, "week_key"].nunique())
        weeks_aborted = int(actions_frame.loc[actions_frame["action"] == WeeklyActionType.ABORT.value, "week_key"].nunique())

    summary.update(
        {
            "net_modeled_pnl": round(net_modeled_pnl, 4),
            "net_dollar_pnl": round(net_dollar_pnl, 4),
            "return_on_capital": round(float(return_on_capital), 6) if return_on_capital is not None else None,
            "max_gross_deployment_dollars": round(max_gross_deployment * config.option_multiplier * config.contracts_per_layer, 4),
            "worst_day_pnl_dollars": round(worst_day_pnl_dollars, 4),
            "worst_week_pnl_dollars": round(worst_week_pnl_dollars, 4),
            "profit_factor_by_closed_layer": round(float(profit_factor_by_closed_layer), 6)
            if profit_factor_by_closed_layer is not None
            else None,
            "profit_factor_by_week": round(float(profit_factor_by_week), 6) if profit_factor_by_week is not None else None,
            "weekly_occupancy_rate": round(float(equity["weekly_occupancy"].mean()), 4),
            "avg_adjustments_per_week": round(float(adjustment_count / weeks_traded), 4) if weeks_traded else 0.0,
            "weeks_traded": weeks_traded,
            "weeks_aborted": weeks_aborted,
            "max_active_butterflies": int(equity["active_butterflies"].max()),
            "closed_layers": int(len(closed)),
            "winning_layers": int(len(winners)),
            "losing_layers": int(len(losers)),
        }
    )
    return summary


def save_backtest_outputs(output_dir: Path, result: Any) -> BacktestArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    transitions_path = output_dir / "transitions.csv"
    actions_path = output_dir / "actions.csv"
    closed_layers_path = output_dir / "closed_layers.csv"
    summary_path = output_dir / "summary.json"
    equity_curve_path = output_dir / "equity_curve.csv"

    transitions_to_frame(result.transitions).to_csv(transitions_path, index=False)
    actions_frame = actions_to_frame(result.actions)
    actions_frame.to_csv(actions_path, index=False)
    closed_frame = actions_frame.loc[actions_frame["realized_pnl"].notna()] if "realized_pnl" in actions_frame.columns else pd.DataFrame()
    closed_frame.to_csv(closed_layers_path, index=False)
    equity_to_frame(result.equity_curve).to_csv(equity_curve_path, index=False)
    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    return BacktestArtifacts(
        transitions_path=transitions_path,
        actions_path=actions_path,
        closed_layers_path=closed_layers_path,
        summary_path=summary_path,
        equity_curve_path=equity_curve_path,
    )


def _metric_definitions() -> dict[str, str]:
    return {
        "net_modeled_pnl": "Final modeled weekly corridor equity in model points after friction.",
        "net_dollar_pnl": "net_modeled_pnl * option_multiplier * contracts_per_layer.",
        "return_on_capital": "net_dollar_pnl / starting_capital.",
        "max_gross_deployment_dollars": "Peak active layer entry cost sum converted to dollars.",
        "worst_day_pnl_dollars": "Minimum grouped daily bar_pnl converted to dollars.",
        "worst_week_pnl_dollars": "Minimum grouped week bar_pnl converted to dollars.",
        "profit_factor_by_closed_layer": "Gross winner realized_pnl divided by absolute gross loser realized_pnl.",
        "profit_factor_by_week": "Gross positive weekly modeled pnl divided by absolute gross negative weekly modeled pnl.",
        "weekly_occupancy_rate": "Share of bars where price stayed inside the active butterfly corridor span.",
        "avg_adjustments_per_week": "ADD_ADJUSTMENT action count / weeks_traded.",
        "weeks_traded": "Distinct week_key count with DEPLOY_INITIAL actions.",
        "weeks_aborted": "Distinct week_key count with ABORT actions.",
        "max_active_butterflies": "Maximum concurrent active weekly butterflies observed in the equity curve.",
    }
