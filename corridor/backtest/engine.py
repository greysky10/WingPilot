from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from corridor.backtest.metrics import compute_metrics
from corridor.config import CorridorConfig
from corridor.models import ActionRecord, ActionType, ActiveButterfly, CorridorState, EquityPoint, Regime
from corridor.options.butterfly_pricer import SimplifiedButterflyPricer
from corridor.options.synthetic_chain import SyntheticChainButterflyPricer
from corridor.strategy.center_estimator import CenterEstimator
from corridor.strategy.corridor_state_machine import CorridorStateMachine
from corridor.strategy.regime import RangeRegimeDetector


@dataclass(slots=True)
class BacktestResult:
    transitions: list
    actions: list[ActionRecord]
    equity_curve: list[EquityPoint]
    summary: dict


class CorridorBacktestEngine:
    """Run the corridor logic bar-by-bar and optionally price simplified butterflies."""

    def __init__(self, config: CorridorConfig) -> None:
        self.config = config
        self.detector = RangeRegimeDetector(config)
        self.center_estimator = CenterEstimator(config)
        self.state_machine = CorridorStateMachine(config)
        if config.payoff_mode == "simplified":
            self.pricer = SimplifiedButterflyPricer(config)
        elif config.payoff_mode == "synthetic_chain":
            self.pricer = SyntheticChainButterflyPricer.from_config(config)
        else:
            self.pricer = None

    def run(self, frame: pd.DataFrame) -> BacktestResult:
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        transitions = []
        actions: list[ActionRecord] = []
        equity_curve: list[EquityPoint] = []
        prev_total_equity = 0.0
        gross_realized_pnl = 0.0

        for idx, row in frame.iterrows():
            history = frame.iloc[: idx + 1]
            timestamp = pd.Timestamp(row["timestamp"])
            price = float(row["close"])
            symbol = str(row["symbol"]).upper()
            regime = self.detector.evaluate(history)
            center = self.center_estimator.estimate(history)

            prior_layers = {layer.layer_id: layer for layer in self.state_machine.context.active_layers}
            protective_exit = self._protective_exit_signal(timestamp, price, history)
            if protective_exit is not None:
                action_type, detail, extra_metadata = protective_exit
                step = self.state_machine.flatten_positions(
                    symbol,
                    timestamp,
                    price,
                    action_type,
                    detail,
                    regime,
                    extra_metadata=extra_metadata,
                )
            else:
                step = self.state_machine.process_bar(symbol, timestamp, price, regime, center)
            transitions.extend(step.transitions)

            current_layers = {layer.layer_id: layer for layer in self.state_machine.context.active_layers}
            opened_ids = sorted(set(current_layers) - set(prior_layers))
            closed_ids = sorted(set(prior_layers) - set(current_layers))

            if (
                self.pricer is not None
                and opened_ids
                and self.config.paper_spread_gate_enabled
                and self.config.paper_spread_gate_mode == "hard_reject"
            ):
                opened_ids = self._apply_paper_spread_gate(
                    symbol,
                    timestamp,
                    price,
                    step.actions,
                    step.transitions,
                    current_layers,
                    opened_ids,
                )
                current_layers = {layer.layer_id: layer for layer in self.state_machine.context.active_layers}
                closed_ids = sorted(set(prior_layers) - set(current_layers))

            if self.pricer is not None and opened_ids and self.config.payoff_mode == "synthetic_chain":
                opened_ids = self._apply_synthetic_chain_gate(
                    symbol,
                    timestamp,
                    price,
                    step.actions,
                    step.transitions,
                    current_layers,
                    opened_ids,
                )
                current_layers = {layer.layer_id: layer for layer in self.state_machine.context.active_layers}
                closed_ids = sorted(set(prior_layers) - set(current_layers))

            if self.pricer is not None:
                for layer_id in opened_ids:
                    layer = current_layers[layer_id]
                    paper_spread_entry_penalty = self._paper_spread_tax_per_side()
                    layer.entry_debit = self.pricer.entry_debit(layer)
                    layer.entry_friction_cost = self.pricer.friction_per_layer(layer) + paper_spread_entry_penalty
                    layer.entry_cost = layer.entry_debit + layer.entry_friction_cost
                    layer.metadata["modeled_max_loss"] = self.pricer.modeled_max_loss(layer)
                    layer.metadata["entry_slippage_cost"] = self.pricer.slippage_cost_per_layer(layer)
                    layer.metadata["entry_commission_cost"] = self.pricer.commission_cost_per_layer()
                    layer.metadata["paper_spread_entry_penalty"] = paper_spread_entry_penalty
                    layer.metadata["paper_spread_close_penalty"] = self._paper_spread_tax_per_side()
                    layer.metadata["paper_spread_penalty_round_trip"] = (
                        layer.metadata["paper_spread_entry_penalty"] + layer.metadata["paper_spread_close_penalty"]
                    )
                    layer.last_mark = layer.entry_cost
                    self._enrich_action(
                        step.actions,
                        layer.layer_id,
                        entry_cost=round(layer.entry_cost, 4),
                        entry_debit=round(layer.entry_debit, 4),
                        friction_cost=round(layer.entry_friction_cost, 4),
                        entry_friction_cost=round(layer.entry_friction_cost, 4),
                        entry_slippage_cost=round(float(layer.metadata["entry_slippage_cost"]), 4),
                        entry_commission_cost=round(float(layer.metadata["entry_commission_cost"]), 4),
                        paper_spread_entry_penalty=round(float(layer.metadata["paper_spread_entry_penalty"]), 4),
                        paper_spread_close_penalty=round(float(layer.metadata["paper_spread_close_penalty"]), 4),
                        paper_spread_penalty_round_trip=round(float(layer.metadata["paper_spread_penalty_round_trip"]), 4),
                        modeled_max_loss=round(float(layer.metadata["modeled_max_loss"]), 4),
                        contracts_per_layer=self.config.contracts_per_layer,
                        option_multiplier=self.config.option_multiplier,
                    )

                for layer_id in closed_ids:
                    layer = prior_layers[layer_id]
                    gross_close_value = self.pricer.mark_to_model(layer, price, timestamp)
                    paper_spread_close_penalty = float(layer.metadata.get("paper_spread_close_penalty", self._paper_spread_tax_per_side()))
                    close_friction = self.pricer.friction_per_layer(layer) + paper_spread_close_penalty
                    close_value = self.pricer.close_value(layer, price, timestamp) - paper_spread_close_penalty
                    layer.exit_value = close_value
                    layer.close_friction_cost = close_friction
                    close_slippage_cost = self.pricer.slippage_cost_per_layer(layer)
                    close_commission_cost = self.pricer.commission_cost_per_layer()
                    gross_realized = gross_close_value - layer.entry_debit
                    realized = close_value - layer.entry_cost
                    gross_realized_pnl += gross_realized
                    self.state_machine.context.realized_pnl += realized
                    self._enrich_action(
                        step.actions,
                        layer.layer_id,
                        close_value=round(close_value, 4),
                        gross_close_value=round(gross_close_value, 4),
                        realized_pnl=round(realized, 4),
                        gross_realized_pnl=round(gross_realized, 4),
                        entry_cost=round(layer.entry_cost, 4),
                        entry_debit=round(layer.entry_debit, 4),
                        entry_friction_cost=round(layer.entry_friction_cost, 4),
                        close_friction_cost=round(close_friction, 4),
                        close_slippage_cost=round(close_slippage_cost, 4),
                        close_commission_cost=round(close_commission_cost, 4),
                        paper_spread_entry_penalty=round(float(layer.metadata.get("paper_spread_entry_penalty", 0.0)), 4),
                        paper_spread_close_penalty=round(paper_spread_close_penalty, 4),
                        paper_spread_penalty_round_trip=round(
                            float(layer.metadata.get("paper_spread_entry_penalty", 0.0)) + paper_spread_close_penalty,
                            4,
                        ),
                        stress_profile=self.config.stress_profile,
                        contracts_per_layer=self.config.contracts_per_layer,
                        option_multiplier=self.config.option_multiplier,
                    )

            actions.extend(step.actions)

            unrealized = 0.0
            gross_unrealized = 0.0
            modeled_capital_at_risk = 0.0
            if self.pricer is not None:
                for layer in self.state_machine.context.active_layers:
                    layer.last_mark = self.pricer.mark_to_model(layer, price, timestamp)
                    unrealized += layer.last_mark - layer.entry_cost
                    gross_unrealized += layer.last_mark - layer.entry_debit
                    modeled_capital_at_risk += float(layer.metadata.get("modeled_max_loss", layer.entry_cost))

            gross_total_equity = gross_realized_pnl + gross_unrealized
            total_equity = self.state_machine.context.realized_pnl + unrealized
            bar_pnl = total_equity - prev_total_equity
            prev_total_equity = total_equity
            occupancy = center is not None and center.tolerance_low <= price <= center.tolerance_high

            equity_curve.append(
                EquityPoint(
                    timestamp=timestamp,
                    symbol=symbol,
                    price=price,
                    regime=regime.regime if regime is not None else Regime.NEUTRAL,
                    state=self.state_machine.context.state,
                    bar_pnl=bar_pnl,
                    realized_pnl=self.state_machine.context.realized_pnl,
                    unrealized_pnl=unrealized,
                    gross_realized_pnl=gross_realized_pnl,
                    gross_unrealized_pnl=gross_unrealized,
                    gross_total_equity=gross_total_equity,
                    total_equity=total_equity,
                    modeled_capital_at_risk=modeled_capital_at_risk,
                    corridor_occupancy=occupancy,
                    active_layers=len(self.state_machine.context.active_layers),
                )
            )

        summary = compute_metrics(self.config, actions, equity_curve)
        return BacktestResult(transitions=transitions, actions=actions, equity_curve=equity_curve, summary=summary)

    @staticmethod
    def _enrich_action(records: list[ActionRecord], layer_id: int, **metadata) -> None:
        for record in records:
            if record.layer_id == layer_id:
                record.metadata.update(metadata)

    def _protective_exit_signal(
        self,
        timestamp: pd.Timestamp,
        price: float,
        history: pd.DataFrame,
    ) -> tuple[ActionType, str, dict[str, float | str]] | None:
        if self.pricer is None:
            return None
        if not self.state_machine.context.active_layers:
            return None

        primary = self._primary_layer(self.state_machine.context.active_layers)
        if primary is None or primary.entry_cost <= 0:
            return None

        close_value = self.pricer.close_value(primary, price, timestamp)
        return_pct = (close_value - primary.entry_cost) / primary.entry_cost

        if self.config.primary_stop_loss_pct > 0 and return_pct <= -self.config.primary_stop_loss_pct:
            return (
                ActionType.STOP_LOSS,
                "Primary stop-loss reached.",
                {"primary_return_pct": round(return_pct, 4)},
            )
        if self.config.primary_take_profit_pct > 0 and return_pct >= self.config.primary_take_profit_pct:
            return (
                ActionType.TAKE_PROFIT,
                "Primary take-profit reached.",
                {"primary_return_pct": round(return_pct, 4)},
            )
        if self.config.close_when_dte_lte > 0:
            remaining_dte = self._remaining_dte_calendar_days(primary, timestamp)
            if remaining_dte <= self.config.close_when_dte_lte:
                return (
                    ActionType.MAX_HOLD,
                    f"Max-hold DTE threshold reached (remaining_dte={remaining_dte}).",
                    {"remaining_dte": int(remaining_dte)},
                )
        if self.config.max_hold_sessions > 0:
            sessions_held = self._held_session_count(primary, timestamp, history)
            if sessions_held >= self.config.max_hold_sessions:
                return (
                    ActionType.MAX_HOLD,
                    f"Max-hold session threshold reached (sessions_held={sessions_held}).",
                    {"sessions_held": int(sessions_held)},
                )
        return None

    @staticmethod
    def _remaining_dte_calendar_days(layer: ActiveButterfly, timestamp: pd.Timestamp) -> int:
        opened_local = pd.Timestamp(layer.created_at)
        if opened_local.tzinfo is None:
            opened_local = opened_local.tz_localize("UTC")
        opened_local = opened_local.tz_convert("America/New_York")
        current_local = pd.Timestamp(timestamp)
        if current_local.tzinfo is None:
            current_local = current_local.tz_localize("UTC")
        current_local = current_local.tz_convert("America/New_York")
        elapsed = max(0, int((current_local.date() - opened_local.date()).days))
        return int(layer.dte) - elapsed

    @staticmethod
    def _held_session_count(layer: ActiveButterfly, timestamp: pd.Timestamp, history: pd.DataFrame) -> int:
        opened_local = pd.Timestamp(layer.created_at)
        if opened_local.tzinfo is None:
            opened_local = opened_local.tz_localize("UTC")
        opened_local_date = opened_local.tz_convert("America/New_York").date()
        current_local = pd.Timestamp(timestamp)
        if current_local.tzinfo is None:
            current_local = current_local.tz_localize("UTC")
        current_local_date = current_local.tz_convert("America/New_York").date()
        if current_local_date < opened_local_date:
            return 0
        local_dates = pd.to_datetime(history["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date
        mask = (local_dates >= opened_local_date) & (local_dates <= current_local_date)
        unique_sessions = int(local_dates[mask].nunique())
        return unique_sessions if unique_sessions > 0 else 1

    def _apply_paper_spread_gate(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        price: float,
        actions: list[ActionRecord],
        transitions: list,
        current_layers: dict[int, ActiveButterfly],
        opened_ids: list[int],
    ) -> list[int]:
        kept_ids: list[int] = []
        for layer_id in opened_ids:
            layer = current_layers.get(layer_id)
            if layer is None:
                continue
            entry_debit = self.pricer.entry_debit(layer)
            estimated_total_spread = max(
                float(self.config.paper_spread_gate_total_spread),
                float(entry_debit) * float(self.config.paper_spread_gate_spread_ratio),
            )
            if estimated_total_spread <= float(self.config.max_acceptable_option_spread):
                kept_ids.append(layer_id)
                continue

            source_action = self._pop_open_action(actions, layer_id)
            self.state_machine.context.active_layers = [
                active for active in self.state_machine.context.active_layers if active.layer_id != layer_id
            ]
            self._rollback_filtered_entry_state(source_action, transitions)
            metadata = self._layer_metadata(layer)
            metadata.update(
                {
                    "source_action": source_action.action.value if source_action is not None else "UNKNOWN",
                    "entry_debit_estimate": round(entry_debit, 4),
                    "estimated_total_spread": round(estimated_total_spread, 4),
                    "estimated_spread_ratio": round(estimated_total_spread / max(float(entry_debit), 0.01), 4),
                    "max_acceptable_option_spread": round(float(self.config.max_acceptable_option_spread), 4),
                    "paper_spread_gate_source": self.config.paper_spread_gate_source,
                    "paper_spread_gate_sample_count": int(self.config.paper_spread_gate_sample_count),
                    "paper_spread_gate_rejection_count": int(self.config.paper_spread_gate_rejection_count),
                }
            )
            actions.append(
                ActionRecord(
                    timestamp=timestamp,
                    symbol=symbol,
                    action=ActionType.ENTRY_FILTERED,
                    state=self.state_machine.context.state,
                    price=price,
                    center_price=self.state_machine.context.current_center,
                    layer_id=layer.layer_id,
                    detail="Paper-calibrated spread gate rejected the modeled entry.",
                    metadata=metadata,
                )
            )
        return kept_ids

    def _apply_synthetic_chain_gate(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        price: float,
        actions: list[ActionRecord],
        transitions: list,
        current_layers: dict[int, ActiveButterfly],
        opened_ids: list[int],
    ) -> list[int]:
        if not isinstance(self.pricer, SyntheticChainButterflyPricer):
            return opened_ids

        kept_ids: list[int] = []
        for layer_id in opened_ids:
            layer = current_layers.get(layer_id)
            if layer is None:
                continue
            estimated_spread = self.pricer.estimated_total_spread(layer)
            if estimated_spread <= float(self.config.max_acceptable_option_spread):
                kept_ids.append(layer_id)
                continue

            source_action = self._pop_open_action(actions, layer_id)
            self.state_machine.context.active_layers = [
                active for active in self.state_machine.context.active_layers if active.layer_id != layer_id
            ]
            self._rollback_filtered_entry_state(source_action, transitions)
            metadata = self._layer_metadata(layer)
            metadata.update(
                {
                    "source_action": source_action.action.value if source_action is not None else "UNKNOWN",
                    "entry_debit_estimate": round(self.pricer.entry_debit(layer), 4),
                    "estimated_total_spread": round(estimated_spread, 4),
                    "estimated_spread_ratio": round(self.pricer.estimated_spread_ratio(layer), 4),
                    "max_acceptable_option_spread": round(float(self.config.max_acceptable_option_spread), 4),
                    "synthetic_chain_state_path": self.config.synthetic_chain_state_path,
                    "synthetic_chain_report_path": self.config.synthetic_chain_report_path,
                }
            )
            actions.append(
                ActionRecord(
                    timestamp=timestamp,
                    symbol=symbol,
                    action=ActionType.ENTRY_FILTERED,
                    state=self.state_machine.context.state,
                    price=price,
                    center_price=self.state_machine.context.current_center,
                    layer_id=layer.layer_id,
                    detail="Synthetic chain spread gate rejected the modeled entry.",
                    metadata=metadata,
                )
            )
        return kept_ids

    def _rollback_filtered_entry_state(self, source_action: ActionRecord | None, transitions: list) -> None:
        if source_action is None:
            return
        if source_action.action not in {ActionType.ENTER_PRIMARY, ActionType.REBUILT}:
            return
        if self.state_machine.context.active_layers:
            return
        if transitions:
            last_transition = transitions[-1]
            if getattr(last_transition, "to_state", None) == self.state_machine.context.state:
                transitions.pop()
        self.state_machine.context.state = CorridorState.IDLE
        self.state_machine.context.current_center = None
        self.state_machine.context.drift_count = 0

    def _paper_spread_tax_per_side(self) -> float:
        if not self.config.paper_spread_gate_enabled or self.config.paper_spread_gate_mode != "tax":
            return 0.0
        round_trip_penalty = max(
            0.0,
            float(self.config.paper_spread_gate_total_spread) - float(self.config.max_acceptable_option_spread),
        )
        return round_trip_penalty / 2.0

    @staticmethod
    def _pop_open_action(actions: list[ActionRecord], layer_id: int) -> ActionRecord | None:
        for index in range(len(actions) - 1, -1, -1):
            record = actions[index]
            if record.layer_id != layer_id:
                continue
            if record.action not in {ActionType.ENTER_PRIMARY, ActionType.ADD_SUPPLEMENTAL, ActionType.REBUILT}:
                continue
            actions.pop(index)
            return record
        return None

    @staticmethod
    def _primary_layer(layers: list[ActiveButterfly]) -> ActiveButterfly | None:
        for layer in layers:
            if layer.kind.value == "PRIMARY":
                return layer
        return layers[0] if layers else None

    @staticmethod
    def _layer_metadata(layer: ActiveButterfly) -> dict[str, float | str]:
        return {
            "kind": layer.kind.value,
            "center_price": round(layer.center_price, 4),
            "lower_strike": round(layer.lower_strike, 4),
            "body_strike": round(layer.body_strike, 4),
            "upper_strike": round(layer.upper_strike, 4),
            "width": round(layer.width, 4),
            "lower_width": round(layer.lower_width, 4),
            "upper_width": round(layer.upper_width, 4),
            "wing_mode": str(layer.metadata.get("wing_mode", "symmetric")),
            "dte": layer.dte,
        }
