from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from corridor.backtest.metrics import compute_metrics
from corridor.config import CorridorConfig
from corridor.models import ActionRecord, ActiveButterfly, EquityPoint, Regime
from corridor.options.butterfly_pricer import SimplifiedButterflyPricer
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
        self.pricer = SimplifiedButterflyPricer(config) if config.payoff_mode == "simplified" else None

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
            step = self.state_machine.process_bar(symbol, timestamp, price, regime, center)
            transitions.extend(step.transitions)

            current_layers = {layer.layer_id: layer for layer in self.state_machine.context.active_layers}
            opened_ids = sorted(set(current_layers) - set(prior_layers))
            closed_ids = sorted(set(prior_layers) - set(current_layers))

            if self.pricer is not None:
                for layer_id in opened_ids:
                    layer = current_layers[layer_id]
                    layer.entry_debit = self.pricer.entry_debit(layer)
                    layer.entry_friction_cost = self.pricer.friction_per_layer()
                    layer.entry_cost = layer.entry_debit + layer.entry_friction_cost
                    layer.last_mark = layer.entry_cost
                    self._enrich_action(
                        step.actions,
                        layer.layer_id,
                        entry_cost=round(layer.entry_cost, 4),
                        entry_debit=round(layer.entry_debit, 4),
                        friction_cost=round(layer.entry_friction_cost, 4),
                        entry_friction_cost=round(layer.entry_friction_cost, 4),
                        contracts_per_layer=self.config.contracts_per_layer,
                        option_multiplier=self.config.option_multiplier,
                    )

                for layer_id in closed_ids:
                    layer = prior_layers[layer_id]
                    gross_close_value = self.pricer.mark_to_model(layer, price, timestamp)
                    close_friction = self.pricer.friction_per_layer()
                    close_value = max(0.0, gross_close_value - close_friction)
                    layer.exit_value = close_value
                    layer.close_friction_cost = close_friction
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
                    modeled_capital_at_risk += layer.entry_cost

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
