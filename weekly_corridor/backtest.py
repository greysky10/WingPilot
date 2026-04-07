from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .center_estimator import WeeklyCenterEstimator
from .config import WeeklyCorridorConfig
from .models import ActionRecord, EquityPoint, WeeklyBacktestResult
from .report import compute_summary
from .state_machine import WeeklyCorridorStateMachine
from .strategy import WeeklyButterflyPricer, WeeklyRegimeClassifier, corridor_bounds, prepare_weekly_frame, trading_week_key


class WeeklyBacktestEngine:
    """Run the separate weekly SPX corridor strategy on resampled decision bars."""

    def __init__(self, config: WeeklyCorridorConfig) -> None:
        self.config = config
        self.detector = WeeklyRegimeClassifier(config)
        self.center_estimator = WeeklyCenterEstimator(config)
        self.state_machine = WeeklyCorridorStateMachine(config)
        self.pricer = WeeklyButterflyPricer(config)

    def run(self, frame: pd.DataFrame) -> WeeklyBacktestResult:
        decision_frame = prepare_weekly_frame(frame, self.config.decision_timeframe)
        transitions = []
        actions: list[ActionRecord] = []
        equity_curve: list[EquityPoint] = []
        prev_total_equity = 0.0
        gross_realized_pnl = 0.0

        for idx, row in decision_frame.iterrows():
            history = decision_frame.iloc[: idx + 1]
            timestamp = pd.Timestamp(row["timestamp"])
            price = float(row["close"])
            symbol = str(row["symbol"]).upper()
            regime = self.detector.evaluate(history)
            center = self.center_estimator.estimate(history)

            prior_layers = {layer.layer_id: layer for layer in self.state_machine.context.active_butterflies}
            step = self.state_machine.process_bar(symbol, timestamp, price, regime, center)
            transitions.extend(step.transitions)

            current_layers = {layer.layer_id: layer for layer in self.state_machine.context.active_butterflies}
            opened_ids = sorted(set(current_layers) - set(prior_layers))
            closed_ids = sorted(set(prior_layers) - set(current_layers))

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
                    entry_friction_cost=round(layer.entry_friction_cost, 4),
                    gross_deployment=round(layer.entry_cost, 4),
                    week_key=layer.week_key,
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
                    close_friction_cost=round(close_friction, 4),
                    week_key=layer.week_key,
                )

            actions.extend(step.actions)

            unrealized = 0.0
            gross_unrealized = 0.0
            gross_deployment = 0.0
            for layer in self.state_machine.context.active_butterflies:
                layer.last_mark = self.pricer.mark_to_model(layer, price, timestamp)
                unrealized += layer.last_mark - layer.entry_cost
                gross_unrealized += layer.last_mark - layer.entry_debit
                gross_deployment += layer.entry_cost

            gross_total_equity = gross_realized_pnl + gross_unrealized
            total_equity = self.state_machine.context.realized_pnl + unrealized
            bar_pnl = total_equity - prev_total_equity
            prev_total_equity = total_equity
            lower_bound, upper_bound = corridor_bounds(self.state_machine.context.active_butterflies)
            occupancy = lower_bound is not None and lower_bound <= price <= upper_bound

            equity_curve.append(
                EquityPoint(
                    timestamp=timestamp,
                    symbol=symbol,
                    week_key=trading_week_key(timestamp),
                    price=price,
                    regime=regime.regime,
                    state=self.state_machine.context.state,
                    bar_pnl=bar_pnl,
                    realized_pnl=self.state_machine.context.realized_pnl,
                    unrealized_pnl=unrealized,
                    gross_realized_pnl=gross_realized_pnl,
                    gross_unrealized_pnl=gross_unrealized,
                    gross_total_equity=gross_total_equity,
                    total_equity=total_equity,
                    gross_deployment=gross_deployment,
                    weekly_occupancy=bool(occupancy),
                    active_butterflies=len(self.state_machine.context.active_butterflies),
                )
            )

        summary = compute_summary(self.config, actions, equity_curve)
        return WeeklyBacktestResult(transitions=transitions, actions=actions, equity_curve=equity_curve, summary=summary)

    @staticmethod
    def _enrich_action(records: list[ActionRecord], layer_id: int, **metadata) -> None:
        for record in records:
            if record.layer_id == layer_id:
                record.metadata.update(metadata)
