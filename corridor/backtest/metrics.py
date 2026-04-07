from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from corridor.backtest.trades import actions_to_frame, equity_to_frame
from corridor.config import CorridorConfig
from corridor.models import ActionRecord, ActionType, EquityPoint


def compute_metrics(config: CorridorConfig, actions: list[ActionRecord], equity_curve: list[EquityPoint]) -> dict[str, Any]:
    """Compute modeled-unit and capital-normalized summary metrics.

    Notes:
    - ``total_return`` is preserved for backward compatibility, but it is only a
      modeled-unit alias for the final net ``total_equity`` value.
    - No percentage return is inferred unless the code explicitly divides by a
      capital or risk denominator.
    """

    definitions = _metric_definitions()
    base_summary = _base_summary(config, definitions)
    if not equity_curve:
        return base_summary

    equity = equity_to_frame(equity_curve)
    equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
    equity["date"] = equity["timestamp"].dt.tz_convert("America/New_York").dt.date
    equity["hour"] = equity["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    equity["cummax"] = equity["total_equity"].cummax()
    equity["drawdown"] = equity["total_equity"] - equity["cummax"]

    actions_frame = actions_to_frame(actions)
    closed_layer_stats = _closed_layer_stats(actions_frame)
    filtered_entry_stats = _filtered_entry_stats(actions_frame)
    paper_spread_penalty_stats = _paper_spread_penalty_stats(actions_frame, config)
    rebuild_stats = _rebuild_stats(actions_frame, equity)
    risk_stats = _modeled_risk_stats(config, actions_frame, equity)

    model_points = float(equity["total_equity"].iloc[-1])
    gross_modeled_pnl = float(equity["gross_total_equity"].iloc[-1])
    net_modeled_pnl = model_points
    dollar_pnl_per_1_lot = net_modeled_pnl * config.option_multiplier
    net_dollar_pnl = dollar_pnl_per_1_lot * config.contracts_per_layer
    gross_dollar_pnl = gross_modeled_pnl * config.option_multiplier * config.contracts_per_layer
    friction_adjustment_dollars = gross_dollar_pnl - net_dollar_pnl
    max_modeled_state_capital_at_risk = risk_stats["max_modeled_state_capital_at_risk"]
    max_modeled_execution_capital_at_risk = risk_stats["max_modeled_execution_capital_at_risk"]
    max_modeled_close_friction_reserve = risk_stats["max_modeled_close_friction_reserve"]
    max_modeled_capital_at_risk = risk_stats["max_modeled_capital_at_risk"]
    max_gross_deployment_dollars = (
        max_modeled_execution_capital_at_risk * config.option_multiplier * config.contracts_per_layer
    )
    max_modeled_state_capital_at_risk_dollars = (
        max_modeled_state_capital_at_risk * config.option_multiplier * config.contracts_per_layer
    )
    max_modeled_execution_capital_at_risk_dollars = (
        max_modeled_execution_capital_at_risk * config.option_multiplier * config.contracts_per_layer
    )
    max_modeled_close_friction_reserve_dollars = (
        max_modeled_close_friction_reserve * config.option_multiplier * config.contracts_per_layer
    )
    max_modeled_capital_at_risk_dollars = max_modeled_capital_at_risk * config.option_multiplier * config.contracts_per_layer
    return_on_capital = net_dollar_pnl / config.starting_capital if config.starting_capital else None
    return_on_max_risk = (
        net_dollar_pnl / max_modeled_capital_at_risk_dollars if max_modeled_capital_at_risk_dollars > 0 else None
    )

    day_regime = (
        equity.groupby("date")["regime"]
        .agg(lambda values: Counter(values).most_common(1)[0][0] if len(values) else "NEUTRAL")
        .to_dict()
    )
    day_pnl = equity.groupby("date")["bar_pnl"].sum()

    trend_day_losses = [float(day_pnl[date]) for date, regime in day_regime.items() if regime in {"TREND_UP", "TREND_DOWN"}]
    range_day_pnls = [float(day_pnl[date]) for date, regime in day_regime.items() if regime == "RANGE"]
    worst_day_pnl = float(day_pnl.min()) if not day_pnl.empty else 0.0
    best_day_pnl = float(day_pnl.max()) if not day_pnl.empty else 0.0
    worst_day_pnl_dollars = worst_day_pnl * config.option_multiplier * config.contracts_per_layer
    best_day_pnl_dollars = best_day_pnl * config.option_multiplier * config.contracts_per_layer
    winning_days = day_pnl[day_pnl > 0.0]
    losing_days = day_pnl[day_pnl < 0.0]
    gross_winning_days = float(winning_days.sum()) if len(winning_days) else 0.0
    gross_losing_days = float(abs(losing_days.sum())) if len(losing_days) else 0.0
    profit_factor_by_day = gross_winning_days / gross_losing_days if gross_losing_days > 0 else None
    gross_winners_dollars = closed_layer_stats["gross_winners"] * config.option_multiplier * config.contracts_per_layer
    gross_losers_dollars = closed_layer_stats["gross_losers"] * config.option_multiplier * config.contracts_per_layer

    base_summary.update(
        {
            # Backward-compatible alias. This remains modeled-unit output only.
            "total_return": round(net_modeled_pnl, 4),
            "total_return_units": "modeled_points",
            "stress_profile": config.stress_profile,
            "stress_entry_debit_multiplier": round(float(config.stress_entry_debit_multiplier), 4),
            "stress_peak_value_multiplier": round(float(config.stress_peak_value_multiplier), 4),
            "stress_residual_floor_multiplier": round(float(config.stress_residual_floor_multiplier), 4),
            "stress_slippage_multiplier": round(float(config.stress_slippage_multiplier), 4),
            "stress_close_value_haircut_pct": round(float(config.stress_close_value_haircut_pct), 4),
            "model_points": round(model_points, 4),
            "gross_modeled_pnl": round(gross_modeled_pnl, 4),
            "net_modeled_pnl": round(net_modeled_pnl, 4),
            "starting_capital": round(float(config.starting_capital), 4),
            "contracts_per_layer": int(config.contracts_per_layer),
            "option_multiplier": int(config.option_multiplier),
            "per_contract_slippage": round(float(config.per_contract_slippage), 4),
            "dollar_pnl_per_1_lot": round(dollar_pnl_per_1_lot, 4),
            "gross_dollar_pnl": round(gross_dollar_pnl, 4),
            "net_dollar_pnl": round(net_dollar_pnl, 4),
            "gross_profit": round(gross_dollar_pnl, 4),
            "net_slippage_adjusted_profit": round(net_dollar_pnl, 4),
            "friction_adjustment_dollars": round(friction_adjustment_dollars, 4),
            "max_gross_deployment_dollars": round(max_gross_deployment_dollars, 4),
            "max_modeled_state_capital_at_risk": round(max_modeled_state_capital_at_risk, 4),
            "max_modeled_state_capital_at_risk_dollars": round(max_modeled_state_capital_at_risk_dollars, 4),
            "max_modeled_execution_capital_at_risk": round(max_modeled_execution_capital_at_risk, 4),
            "max_modeled_execution_capital_at_risk_dollars": round(max_modeled_execution_capital_at_risk_dollars, 4),
            "max_modeled_close_friction_reserve": round(max_modeled_close_friction_reserve, 4),
            "max_modeled_close_friction_reserve_dollars": round(max_modeled_close_friction_reserve_dollars, 4),
            "max_modeled_capital_at_risk": round(max_modeled_capital_at_risk, 4),
            "max_modeled_capital_at_risk_dollars": round(max_modeled_capital_at_risk_dollars, 4),
            "max_modeled_capital_at_risk_assumption": risk_stats["assumption"],
            "return_on_capital": round(float(return_on_capital), 6) if return_on_capital is not None else None,
            "return_on_max_risk": round(float(return_on_max_risk), 6) if return_on_max_risk is not None else None,
            "max_drawdown": round(float(equity["drawdown"].min()), 4),
            "average_rebuilds_per_day": round(float(rebuild_stats["average_rebuilds_per_day"]), 4),
            "pnl_by_regime": {
                key: round(float(value), 4) for key, value in equity.groupby("regime")["bar_pnl"].sum().to_dict().items()
            },
            "pnl_by_time_of_day": {
                key: round(float(value), 4) for key, value in equity.groupby("hour")["bar_pnl"].sum().to_dict().items()
            },
            "worst_trend_day_loss": round(float(min(trend_day_losses)) if trend_day_losses else 0.0, 4),
            "worst_day_pnl": round(worst_day_pnl, 4),
            "worst_day_pnl_dollars": round(worst_day_pnl_dollars, 4),
            "best_day_pnl": round(best_day_pnl, 4),
            "best_day_pnl_dollars": round(best_day_pnl_dollars, 4),
            "profit_factor_by_day": round(float(profit_factor_by_day), 6) if profit_factor_by_day is not None else None,
            "average_range_day_pnl": round(float(sum(range_day_pnls) / len(range_day_pnls)) if range_day_pnls else 0.0, 4),
            "cost_drag_from_rebuilding": round(float(rebuild_stats["cost_drag_from_rebuilding"]), 4),
            "corridor_occupancy_rate": round(float(equity["corridor_occupancy"].mean()), 4),
            "closed_layers": int(closed_layer_stats["closed_layers"]),
            "winning_layers": int(closed_layer_stats["winning_layers"]),
            "losing_layers": int(closed_layer_stats["losing_layers"]),
            "flat_layers": int(closed_layer_stats["flat_layers"]),
            "gross_winners": round(float(closed_layer_stats["gross_winners"]), 4),
            "gross_losers": round(float(closed_layer_stats["gross_losers"]), 4),
            "gross_winners_dollars": round(gross_winners_dollars, 4),
            "gross_losers_dollars": round(gross_losers_dollars, 4),
            "paper_spread_gate_enabled": bool(config.paper_spread_gate_enabled),
            "paper_spread_gate_mode": config.paper_spread_gate_mode,
            "paper_spread_gate_source": config.paper_spread_gate_source,
            "paper_spread_gate_spread_ratio": round(float(config.paper_spread_gate_spread_ratio), 4),
            "paper_spread_gate_total_spread": round(float(config.paper_spread_gate_total_spread), 4),
            "paper_spread_gate_sample_count": int(config.paper_spread_gate_sample_count),
            "paper_spread_gate_rejection_count": int(config.paper_spread_gate_rejection_count),
            "paper_spread_penalty_per_side": round(float(paper_spread_penalty_stats["per_side"]), 4),
            "paper_spread_penalty_round_trip": round(float(paper_spread_penalty_stats["round_trip"]), 4),
            "paper_spread_penalty_dollars": round(float(paper_spread_penalty_stats["dollars"]), 4),
            "execution_filtered_entries": int(filtered_entry_stats["filtered_entries"]),
            "execution_filtered_primary_entries": int(filtered_entry_stats["filtered_primary_entries"]),
            "execution_filtered_supplementals": int(filtered_entry_stats["filtered_supplementals"]),
            "win_rate_by_closed_layer": round(float(closed_layer_stats["win_rate_by_closed_layer"]), 6)
            if closed_layer_stats["win_rate_by_closed_layer"] is not None
            else None,
            "average_closed_layer_pnl": round(float(closed_layer_stats["average_closed_layer_pnl"]), 4)
            if closed_layer_stats["average_closed_layer_pnl"] is not None
            else None,
            "average_winner_pnl": round(float(closed_layer_stats["average_winner_pnl"]), 4)
            if closed_layer_stats["average_winner_pnl"] is not None
            else None,
            "average_loser_pnl": round(float(closed_layer_stats["average_loser_pnl"]), 4)
            if closed_layer_stats["average_loser_pnl"] is not None
            else None,
            "profit_factor_by_closed_layer": round(float(closed_layer_stats["profit_factor_by_closed_layer"]), 6)
            if closed_layer_stats["profit_factor_by_closed_layer"] is not None
            else None,
        }
    )
    return base_summary


def _rebuild_stats(actions_frame: pd.DataFrame, equity: pd.DataFrame) -> dict[str, float]:
    if actions_frame.empty:
        return {"average_rebuilds_per_day": 0.0, "cost_drag_from_rebuilding": 0.0}

    actions_frame = actions_frame.copy()
    actions_frame["timestamp"] = pd.to_datetime(actions_frame["timestamp"], utc=True)
    actions_frame["date"] = actions_frame["timestamp"].dt.tz_convert("America/New_York").dt.date
    rebuild_count = int((actions_frame["action"] == ActionType.REBUILD_REQUESTED.value).sum())
    active_days = max(1, equity["date"].nunique())
    rebuilt = actions_frame.loc[actions_frame["action"] == ActionType.REBUILT.value].copy()
    if rebuilt.empty:
        cost_drag_from_rebuilding = 0.0
    else:
        entry_cost_sum = float(rebuilt["entry_cost"].fillna(0.0).sum()) if "entry_cost" in rebuilt.columns else 0.0
        friction_column = "entry_friction_cost" if "entry_friction_cost" in rebuilt.columns else "friction_cost"
        friction_sum = float(rebuilt[friction_column].fillna(0.0).sum()) if friction_column in rebuilt.columns else 0.0
        cost_drag_from_rebuilding = entry_cost_sum + friction_sum
    return {
        "average_rebuilds_per_day": rebuild_count / active_days,
        "cost_drag_from_rebuilding": cost_drag_from_rebuilding,
    }


def _closed_layer_stats(actions_frame: pd.DataFrame) -> dict[str, float | int | None]:
    if actions_frame.empty or "realized_pnl" not in actions_frame.columns:
        return {
            "closed_layers": 0,
            "winning_layers": 0,
            "losing_layers": 0,
            "flat_layers": 0,
            "gross_winners": 0.0,
            "gross_losers": 0.0,
            "win_rate_by_closed_layer": None,
            "average_closed_layer_pnl": None,
            "average_winner_pnl": None,
            "average_loser_pnl": None,
            "profit_factor_by_closed_layer": None,
        }

    closed = actions_frame.loc[actions_frame["realized_pnl"].notna()].copy()
    if closed.empty:
        return {
            "closed_layers": 0,
            "winning_layers": 0,
            "losing_layers": 0,
            "flat_layers": 0,
            "gross_winners": 0.0,
            "gross_losers": 0.0,
            "win_rate_by_closed_layer": None,
            "average_closed_layer_pnl": None,
            "average_winner_pnl": None,
            "average_loser_pnl": None,
            "profit_factor_by_closed_layer": None,
        }

    closed["realized_pnl"] = closed["realized_pnl"].astype(float)
    winners = closed.loc[closed["realized_pnl"] > 0.0, "realized_pnl"]
    losers = closed.loc[closed["realized_pnl"] < 0.0, "realized_pnl"]
    flats = closed.loc[closed["realized_pnl"] == 0.0, "realized_pnl"]
    gross_winners = float(winners.sum()) if len(winners) else 0.0
    gross_losers = float(abs(losers.sum())) if len(losers) else 0.0
    profit_factor = gross_winners / gross_losers if gross_losers > 0 else (None if gross_winners == 0 else None)

    return {
        "closed_layers": int(len(closed)),
        "winning_layers": int(len(winners)),
        "losing_layers": int(len(losers)),
        "flat_layers": int(len(flats)),
        "gross_winners": gross_winners,
        "gross_losers": gross_losers,
        "win_rate_by_closed_layer": float(len(winners) / len(closed)) if len(closed) else None,
        "average_closed_layer_pnl": float(closed["realized_pnl"].mean()) if len(closed) else None,
        "average_winner_pnl": float(winners.mean()) if len(winners) else None,
        "average_loser_pnl": float(losers.mean()) if len(losers) else None,
        "profit_factor_by_closed_layer": profit_factor,
    }


def _filtered_entry_stats(actions_frame: pd.DataFrame) -> dict[str, int]:
    if actions_frame.empty or "action" not in actions_frame.columns:
        return {
            "filtered_entries": 0,
            "filtered_primary_entries": 0,
            "filtered_supplementals": 0,
        }

    filtered = actions_frame.loc[actions_frame["action"] == ActionType.ENTRY_FILTERED.value].copy()
    if filtered.empty:
        return {
            "filtered_entries": 0,
            "filtered_primary_entries": 0,
            "filtered_supplementals": 0,
        }

    if "kind" in filtered.columns:
        kinds = filtered["kind"].fillna("").astype(str)
    else:
        kinds = pd.Series(dtype=str)
    return {
        "filtered_entries": int(len(filtered)),
        "filtered_primary_entries": int((kinds == "PRIMARY").sum()),
        "filtered_supplementals": int((kinds == "SUPPLEMENTAL").sum()),
    }


def _paper_spread_penalty_stats(actions_frame: pd.DataFrame, config: CorridorConfig) -> dict[str, float]:
    round_trip = 0.0
    per_side = 0.0
    if config.paper_spread_gate_enabled and config.paper_spread_gate_mode == "tax":
        round_trip = max(0.0, float(config.paper_spread_gate_total_spread) - float(config.max_acceptable_option_spread))
        per_side = round_trip / 2.0
    if actions_frame.empty:
        return {"per_side": per_side, "round_trip": round_trip, "dollars": 0.0}

    dollars = 0.0
    for column in ["paper_spread_entry_penalty", "paper_spread_close_penalty"]:
        if column not in actions_frame.columns:
            continue
        dollars += float(actions_frame[column].fillna(0.0).sum()) * config.option_multiplier * config.contracts_per_layer
    return {"per_side": per_side, "round_trip": round_trip, "dollars": dollars}


def _modeled_risk_stats(config: CorridorConfig, actions_frame: pd.DataFrame, equity: pd.DataFrame) -> dict[str, float | str]:
    stable_peak = float(equity["modeled_capital_at_risk"].max()) if not equity.empty else 0.0
    execution_peak = stable_peak
    execution_peak_layers = 0

    if not equity.empty and "active_layers" in equity.columns:
        stable_peak_rows = equity.loc[equity["modeled_capital_at_risk"] == stable_peak]
        if not stable_peak_rows.empty:
            execution_peak_layers = int(stable_peak_rows["active_layers"].max())

    if actions_frame.empty:
        close_friction = _modeled_close_friction_per_layer(config)
        close_reserve = execution_peak_layers * close_friction
        return {
            "max_modeled_state_capital_at_risk": stable_peak,
            "max_modeled_execution_capital_at_risk": execution_peak,
            "max_modeled_close_friction_reserve": close_reserve,
            "max_modeled_capital_at_risk": execution_peak + close_reserve,
            "assumption": "stable_state_entry_cost_plus_close_friction_reserve",
        }

    ordered = actions_frame.copy().reset_index(drop=True)
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
    ordered["__seq"] = range(len(ordered))
    ordered = ordered.sort_values(["timestamp", "__seq"])

    active_open_costs: dict[int, float] = {}
    for row in ordered.itertuples(index=False):
        if _action_opens_layer(row):
            layer_id = getattr(row, "layer_id", None)
            entry_cost = getattr(row, "entry_cost", None)
            if pd.isna(layer_id) or pd.isna(entry_cost):
                continue
            active_open_costs[int(layer_id)] = float(entry_cost)
            execution_peak = max(execution_peak, sum(active_open_costs.values()))
            execution_peak_layers = max(execution_peak_layers, len(active_open_costs))
        elif _action_closes_layer(row):
            layer_id = getattr(row, "layer_id", None)
            if pd.isna(layer_id):
                continue
            active_open_costs.pop(int(layer_id), None)

    active_open_costs.clear()
    conservative_peak = stable_peak
    conservative_peak_layers = execution_peak_layers
    grouped = ordered.groupby("timestamp", sort=True)
    for _, bucket in grouped:
        bucket_opens: list[tuple[int, float]] = []
        bucket_closes: list[int] = []
        for row in bucket.itertuples(index=False):
            if _action_opens_layer(row):
                layer_id = getattr(row, "layer_id", None)
                entry_cost = getattr(row, "entry_cost", None)
                if pd.isna(layer_id) or pd.isna(entry_cost):
                    continue
                bucket_opens.append((int(layer_id), float(entry_cost)))
            elif _action_closes_layer(row):
                layer_id = getattr(row, "layer_id", None)
                if pd.isna(layer_id):
                    continue
                bucket_closes.append(int(layer_id))

        assumed_peak = sum(active_open_costs.values()) + sum(cost for _, cost in bucket_opens)
        assumed_layers = len(active_open_costs) + len(bucket_opens)
        if assumed_peak > conservative_peak:
            conservative_peak = assumed_peak
            conservative_peak_layers = assumed_layers

        for layer_id in bucket_closes:
            active_open_costs.pop(layer_id, None)
        for layer_id, entry_cost in bucket_opens:
            active_open_costs[layer_id] = entry_cost

    close_friction = _modeled_close_friction_per_layer(config)
    close_reserve = conservative_peak_layers * close_friction
    return {
        "max_modeled_state_capital_at_risk": stable_peak,
        "max_modeled_execution_capital_at_risk": conservative_peak,
        "max_modeled_close_friction_reserve": close_reserve,
        "max_modeled_capital_at_risk": conservative_peak + close_reserve,
        "assumption": "conservative_open_before_close_within_same_timestamp_plus_close_friction_reserve",
    }


def _action_opens_layer(row: Any) -> bool:
    action = getattr(row, "action", "")
    detail = getattr(row, "detail", "") or ""
    return action in {ActionType.ENTER_PRIMARY.value, ActionType.ADD_SUPPLEMENTAL.value} or (
        action == ActionType.REBUILT.value and isinstance(detail, str) and detail.startswith("Established")
    )


def _action_closes_layer(row: Any) -> bool:
    action = getattr(row, "action", "")
    detail = getattr(row, "detail", "") or ""
    return action in {
        ActionType.SESSION_FLUSH.value,
        ActionType.ABORTED.value,
        ActionType.STOP_LOSS.value,
        ActionType.TAKE_PROFIT.value,
    } or (
        action == ActionType.REBUILT.value and isinstance(detail, str) and detail.startswith("Removed")
    )


def _modeled_close_friction_per_layer(config: CorridorConfig) -> float:
    slippage_contracts = 4.0
    if config.wing_mode in {"broken_upper", "broken_lower", "adaptive"} and float(config.broken_wing_extra_width) > 0:
        slippage_contracts = 5.0
    slippage_cost = float(config.per_contract_slippage) * slippage_contracts * float(config.stress_slippage_multiplier)
    commission_cost = (float(config.commission_per_contract) * 4.0) / float(config.option_multiplier)
    return slippage_cost + commission_cost


def _base_summary(config: CorridorConfig, definitions: dict[str, str]) -> dict[str, Any]:
    return {
        "symbol": config.symbol,
        "timeframe": config.timeframe,
        "wing_mode": config.wing_mode,
        "broken_wing_extra_width": round(float(config.broken_wing_extra_width), 4),
        "total_return": 0.0,
        "total_return_units": "modeled_points",
        "stress_profile": config.stress_profile,
        "stress_entry_debit_multiplier": round(float(config.stress_entry_debit_multiplier), 4),
        "stress_peak_value_multiplier": round(float(config.stress_peak_value_multiplier), 4),
        "stress_residual_floor_multiplier": round(float(config.stress_residual_floor_multiplier), 4),
        "stress_slippage_multiplier": round(float(config.stress_slippage_multiplier), 4),
        "stress_close_value_haircut_pct": round(float(config.stress_close_value_haircut_pct), 4),
        "model_points": 0.0,
        "gross_modeled_pnl": 0.0,
        "net_modeled_pnl": 0.0,
        "starting_capital": round(float(config.starting_capital), 4),
        "contracts_per_layer": int(config.contracts_per_layer),
        "option_multiplier": int(config.option_multiplier),
        "per_contract_slippage": round(float(config.per_contract_slippage), 4),
        "dollar_pnl_per_1_lot": 0.0,
        "gross_dollar_pnl": 0.0,
        "net_dollar_pnl": 0.0,
        "gross_profit": 0.0,
        "net_slippage_adjusted_profit": 0.0,
        "friction_adjustment_dollars": 0.0,
        "max_gross_deployment_dollars": 0.0,
        "max_modeled_state_capital_at_risk": 0.0,
        "max_modeled_state_capital_at_risk_dollars": 0.0,
        "max_modeled_execution_capital_at_risk": 0.0,
        "max_modeled_execution_capital_at_risk_dollars": 0.0,
        "max_modeled_close_friction_reserve": 0.0,
        "max_modeled_close_friction_reserve_dollars": 0.0,
        "max_modeled_capital_at_risk": 0.0,
        "max_modeled_capital_at_risk_dollars": 0.0,
        "max_modeled_capital_at_risk_assumption": "conservative_open_before_close_within_same_timestamp_plus_close_friction_reserve",
        "return_on_capital": None,
        "return_on_max_risk": None,
        "max_drawdown": 0.0,
        "average_rebuilds_per_day": 0.0,
        "pnl_by_regime": {},
        "pnl_by_time_of_day": {},
        "worst_trend_day_loss": 0.0,
        "worst_day_pnl": 0.0,
        "worst_day_pnl_dollars": 0.0,
        "best_day_pnl": 0.0,
        "best_day_pnl_dollars": 0.0,
        "profit_factor_by_day": None,
        "average_range_day_pnl": 0.0,
        "cost_drag_from_rebuilding": 0.0,
        "corridor_occupancy_rate": 0.0,
        "closed_layers": 0,
        "winning_layers": 0,
        "losing_layers": 0,
        "flat_layers": 0,
        "gross_winners": 0.0,
        "gross_losers": 0.0,
        "gross_winners_dollars": 0.0,
        "gross_losers_dollars": 0.0,
        "paper_spread_gate_enabled": bool(config.paper_spread_gate_enabled),
        "paper_spread_gate_mode": config.paper_spread_gate_mode,
        "paper_spread_gate_source": config.paper_spread_gate_source,
        "paper_spread_gate_spread_ratio": round(float(config.paper_spread_gate_spread_ratio), 4),
        "paper_spread_gate_total_spread": round(float(config.paper_spread_gate_total_spread), 4),
        "paper_spread_gate_sample_count": int(config.paper_spread_gate_sample_count),
        "paper_spread_gate_rejection_count": int(config.paper_spread_gate_rejection_count),
        "paper_spread_penalty_per_side": 0.0,
        "paper_spread_penalty_round_trip": 0.0,
        "paper_spread_penalty_dollars": 0.0,
        "execution_filtered_entries": 0,
        "execution_filtered_primary_entries": 0,
        "execution_filtered_supplementals": 0,
        "win_rate_by_closed_layer": None,
        "average_closed_layer_pnl": None,
        "average_winner_pnl": None,
        "average_loser_pnl": None,
        "profit_factor_by_closed_layer": None,
        "metric_definitions": definitions,
    }


def _metric_definitions() -> dict[str, str]:
    return {
        "wing_mode": "Butterfly geometry mode used by the backtest or runner: symmetric, broken_upper, broken_lower, or adaptive.",
        "broken_wing_extra_width": "Extra width added to the broken side when wing_mode is asymmetric.",
        "total_return": "Backward-compatible alias for net_modeled_pnl; equals final total_equity from the modeled equity curve and is not normalized by capital.",
        "stress_profile": "Named simplified-pricer stress profile used for the run.",
        "stress_entry_debit_multiplier": "Multiplier applied to simplified entry debit before entry friction.",
        "stress_peak_value_multiplier": "Multiplier applied to the simplified peak-value assumption used by mark_to_model.",
        "stress_residual_floor_multiplier": "Multiplier applied to the simplified residual floor used by mark_to_model.",
        "stress_slippage_multiplier": "Multiplier applied to modeled slippage inside friction_per_layer.",
        "stress_close_value_haircut_pct": "Additional percentage haircut applied to close_value after modeled friction.",
        "model_points": "Final net modeled equity in model points; computed as equity_curve.total_equity[-1].",
        "gross_modeled_pnl": "Final modeled equity before modeled friction; computed as equity_curve.gross_total_equity[-1].",
        "net_modeled_pnl": "Final modeled equity after modeled friction; computed as equity_curve.total_equity[-1].",
        "per_contract_slippage": "Modeled per-contract slippage in option points applied to each open/close action.",
        "dollar_pnl_per_1_lot": "net_modeled_pnl * option_multiplier.",
        "gross_dollar_pnl": "gross_modeled_pnl * option_multiplier * contracts_per_layer.",
        "net_dollar_pnl": "net_modeled_pnl * option_multiplier * contracts_per_layer.",
        "gross_profit": "Alias of gross_dollar_pnl for quick gross-vs-net comparison in dollars.",
        "net_slippage_adjusted_profit": "Alias of net_dollar_pnl after modeled commission and slippage adjustments in dollars.",
        "friction_adjustment_dollars": "gross_dollar_pnl - net_dollar_pnl; total modeled commission/slippage drag in dollars.",
        "max_gross_deployment_dollars": "Peak gross modeled deployment dollars at the conservative execution-time peak before adding close-side friction reserve.",
        "max_modeled_state_capital_at_risk": "Peak stable-state modeled entry cost from equity_curve.modeled_capital_at_risk; excludes intra-timestamp overlap.",
        "max_modeled_execution_capital_at_risk": "Conservative execution-time peak modeled entry cost assuming opens can consume capital before same-timestamp closes release it.",
        "max_modeled_close_friction_reserve": "Additional modeled reserve for one close-side friction charge per open layer at the conservative execution peak.",
        "max_modeled_capital_at_risk": "Conservative modeled risk proxy = max_modeled_execution_capital_at_risk + max_modeled_close_friction_reserve.",
        "return_on_capital": "net_dollar_pnl / starting_capital.",
        "return_on_max_risk": "net_dollar_pnl / max_modeled_capital_at_risk_dollars using the conservative modeled risk proxy denominator.",
        "worst_day_pnl_dollars": "Minimum daily net modeled PnL in dollars; computed from grouped day_pnl * option_multiplier * contracts_per_layer.",
        "best_day_pnl_dollars": "Maximum daily net modeled PnL in dollars; computed from grouped day_pnl * option_multiplier * contracts_per_layer.",
        "profit_factor_by_day": "Gross positive daily modeled PnL divided by absolute gross negative daily modeled PnL.",
        "closed_layers": "Count of close actions with realized_pnl metadata.",
        "gross_winners_dollars": "Sum of positive closed-layer realized_pnl values converted to dollars.",
        "gross_losers_dollars": "Absolute sum of negative closed-layer realized_pnl values converted to dollars.",
        "paper_spread_gate_enabled": "Whether a paper-calibrated spread execution gate was applied to modeled entries in the backtest.",
        "paper_spread_gate_mode": "How paper diagnostics are applied in the backtest: tax adds extra friction, hard_reject skips entries.",
        "paper_spread_gate_source": "Source JSON used to calibrate the paper spread execution gate.",
        "paper_spread_gate_spread_ratio": "Median spread_ratio from sampled paper candidates rejected for spread_too_wide.",
        "paper_spread_gate_total_spread": "Median total_spread from sampled paper candidates rejected for spread_too_wide.",
        "paper_spread_gate_sample_count": "Count of sampled spread_too_wide paper candidates used for the execution gate calibration.",
        "paper_spread_gate_rejection_count": "Reported spread_too_wide rejection count from the paper diagnostics source.",
        "paper_spread_penalty_per_side": "Additional modeled per-side spread penalty in option points when paper diagnostics mode is tax.",
        "paper_spread_penalty_round_trip": "Additional modeled round-trip spread penalty in option points when paper diagnostics mode is tax.",
        "paper_spread_penalty_dollars": "Total paper-calibrated spread penalty dollars applied across entry and close actions in the run.",
        "execution_filtered_entries": "Count of modeled entries skipped by the paper-calibrated spread gate.",
        "execution_filtered_primary_entries": "Count of primary-layer modeled entries skipped by the paper-calibrated spread gate.",
        "execution_filtered_supplementals": "Count of supplemental modeled entries skipped by the paper-calibrated spread gate.",
        "win_rate_by_closed_layer": "winning_layers / closed_layers using realized_pnl > 0 as a winner.",
        "average_closed_layer_pnl": "Mean realized_pnl across all closed layers.",
        "average_winner_pnl": "Mean realized_pnl across closed layers with realized_pnl > 0.",
        "average_loser_pnl": "Mean realized_pnl across closed layers with realized_pnl < 0.",
        "profit_factor_by_closed_layer": "Gross winners divided by absolute gross losers using closed-layer realized_pnl values.",
    }
