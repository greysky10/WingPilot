from __future__ import annotations

import unittest

import pandas as pd

from corridor.backtest.engine import CorridorBacktestEngine
from corridor.config import CorridorConfig
from corridor.models import (
    ActiveButterfly,
    ActionType,
    CenterEstimate,
    CenterMethod,
    CorridorState,
    LayerKind,
    Regime,
    RegimeSnapshot,
)


class CorridorEngineProtectiveExitTests(unittest.TestCase):
    def test_primary_take_profit_signal_fires(self) -> None:
        cfg = CorridorConfig(
            symbol="SPX",
            payoff_mode="simplified",
            butterfly_width=30.0,
            primary_take_profit_pct=0.10,
        )
        engine = CorridorBacktestEngine(cfg)
        created_at = pd.Timestamp("2026-03-31 14:00:00", tz="UTC")
        layer = ActiveButterfly(
            layer_id=1,
            kind=LayerKind.PRIMARY,
            center_price=6400.0,
            width=30.0,
            lower_width=30.0,
            upper_width=30.0,
            lower_strike=6370.0,
            body_strike=6400.0,
            upper_strike=6430.0,
            created_at=created_at,
            dte=7,
        )
        layer.entry_debit = engine.pricer.entry_debit(layer)
        layer.entry_friction_cost = engine.pricer.friction_per_layer()
        layer.entry_cost = engine.pricer.entry_cost(layer)
        engine.state_machine.context.active_layers = [layer]
        history = pd.DataFrame(
            [
                {
                    "timestamp": created_at + pd.Timedelta(days=1),
                    "symbol": "SPX",
                    "open": 6400.0,
                    "high": 6400.0,
                    "low": 6400.0,
                    "close": 6400.0,
                    "volume": 1.0,
                }
            ]
        )

        signal = engine._protective_exit_signal(created_at + pd.Timedelta(days=1), 6400.0, history)
        self.assertIsNotNone(signal)
        self.assertEqual(signal[0], ActionType.TAKE_PROFIT)

    def test_primary_stop_loss_signal_fires(self) -> None:
        cfg = CorridorConfig(
            symbol="SPX",
            payoff_mode="simplified",
            butterfly_width=30.0,
            primary_stop_loss_pct=0.20,
        )
        engine = CorridorBacktestEngine(cfg)
        created_at = pd.Timestamp("2026-03-31 14:00:00", tz="UTC")
        layer = ActiveButterfly(
            layer_id=1,
            kind=LayerKind.PRIMARY,
            center_price=6400.0,
            width=30.0,
            lower_width=30.0,
            upper_width=30.0,
            lower_strike=6370.0,
            body_strike=6400.0,
            upper_strike=6430.0,
            created_at=created_at,
            dte=7,
        )
        layer.entry_debit = engine.pricer.entry_debit(layer)
        layer.entry_friction_cost = engine.pricer.friction_per_layer()
        layer.entry_cost = engine.pricer.entry_cost(layer)
        engine.state_machine.context.active_layers = [layer]
        history = pd.DataFrame(
            [
                {
                    "timestamp": created_at + pd.Timedelta(days=1),
                    "symbol": "SPX",
                    "open": 6460.0,
                    "high": 6460.0,
                    "low": 6460.0,
                    "close": 6460.0,
                    "volume": 1.0,
                }
            ]
        )

        signal = engine._protective_exit_signal(created_at + pd.Timedelta(days=1), 6460.0, history)
        self.assertIsNotNone(signal)
        self.assertEqual(signal[0], ActionType.STOP_LOSS)

    def test_individual_exit_scope_only_flags_expiring_layer(self) -> None:
        cfg = CorridorConfig(
            symbol="SPX",
            payoff_mode="simplified",
            butterfly_width=30.0,
            layer_exit_scope="individual",
            close_when_dte_lte=3,
        )
        engine = CorridorBacktestEngine(cfg)
        created_at = pd.Timestamp("2026-03-31 14:00:00", tz="UTC")
        short_layer = ActiveButterfly(
            layer_id=1,
            kind=LayerKind.PRIMARY,
            center_price=6400.0,
            width=30.0,
            lower_width=30.0,
            upper_width=30.0,
            lower_strike=6370.0,
            body_strike=6400.0,
            upper_strike=6430.0,
            created_at=created_at,
            dte=3,
        )
        long_layer = ActiveButterfly(
            layer_id=2,
            kind=LayerKind.SUPPLEMENTAL,
            center_price=6400.0,
            width=30.0,
            lower_width=30.0,
            upper_width=30.0,
            lower_strike=6370.0,
            body_strike=6400.0,
            upper_strike=6430.0,
            created_at=created_at,
            dte=10,
        )
        for layer in (short_layer, long_layer):
            layer.entry_debit = engine.pricer.entry_debit(layer)
            layer.entry_friction_cost = engine.pricer.friction_per_layer()
            layer.entry_cost = engine.pricer.entry_cost(layer)
        engine.state_machine.context.active_layers = [short_layer, long_layer]
        history = pd.DataFrame(
            [
                {
                    "timestamp": created_at,
                    "symbol": "SPX",
                    "open": 6400.0,
                    "high": 6400.0,
                    "low": 6400.0,
                    "close": 6400.0,
                    "volume": 1.0,
                }
            ]
        )

        signals = engine._protective_exit_signals(created_at, 6400.0, history)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0][0], 1)
        self.assertEqual(signals[0][1], ActionType.MAX_HOLD)
        self.assertEqual(signals[0][3]["remaining_dte"], 3)

    def test_take_profit_on_new_session_blocks_same_day_reentry(self) -> None:
        cfg = CorridorConfig(
            symbol="SPX",
            payoff_mode="simplified",
            butterfly_width=30.0,
            center_lookback=2,
            regime_lookback=2,
            hold_overnight=True,
            primary_take_profit_pct=0.10,
            block_same_day_reentry_after_take_profit=True,
            primary_entry_end="15:30",
        )
        engine = CorridorBacktestEngine(cfg)
        previous_session = pd.Timestamp("2026-03-30 19:55:00", tz="UTC")
        take_profit_bar = pd.Timestamp("2026-03-31 13:30:00", tz="UTC")
        next_bar = pd.Timestamp("2026-03-31 13:50:00", tz="UTC")
        center = CenterEstimate(
            timestamp=take_profit_bar,
            center_price=6400.0,
            lower_band=6370.0,
            upper_band=6430.0,
            tolerance_low=6390.0,
            tolerance_high=6410.0,
            method=CenterMethod.VWAP,
            confidence=0.9,
        )

        class _Detector:
            @staticmethod
            def evaluate(history):
                return RegimeSnapshot(
                    pd.Timestamp(history.iloc[-1]["timestamp"]),
                    Regime.RANGE,
                    0.01,
                    0.0,
                    0.0,
                    1.0,
                    False,
                    False,
                )

        class _Estimator:
            @staticmethod
            def estimate(history):
                return center

        engine.detector = _Detector()
        engine.center_estimator = _Estimator()

        layer = ActiveButterfly(
            layer_id=1,
            kind=LayerKind.PRIMARY,
            center_price=6400.0,
            width=30.0,
            lower_width=30.0,
            upper_width=30.0,
            lower_strike=6370.0,
            body_strike=6400.0,
            upper_strike=6430.0,
            created_at=previous_session,
            dte=7,
        )
        layer.entry_debit = engine.pricer.entry_debit(layer)
        layer.entry_friction_cost = engine.pricer.friction_per_layer()
        layer.entry_cost = engine.pricer.entry_cost(layer)
        engine.state_machine.context.state = CorridorState.ACTIVE_CENTERED
        engine.state_machine.context.active_layers = [layer]
        engine.state_machine.context.session_date = "2026-03-30"
        engine.state_machine.context.last_processed_close = 6400.0

        frame = pd.DataFrame(
            [
                {
                    "timestamp": take_profit_bar,
                    "symbol": "SPX",
                    "open": 6400.0,
                    "high": 6400.0,
                    "low": 6400.0,
                    "close": 6400.0,
                    "volume": 1.0,
                },
                {
                    "timestamp": next_bar,
                    "symbol": "SPX",
                    "open": 6400.0,
                    "high": 6400.0,
                    "low": 6400.0,
                    "close": 6400.0,
                    "volume": 1.0,
                },
            ]
        )

        result = engine.run(frame)
        enter_actions = [action for action in result.actions if action.action == ActionType.ENTER_PRIMARY]
        take_profit_actions = [action for action in result.actions if action.action == ActionType.TAKE_PROFIT]

        self.assertEqual(len(take_profit_actions), 1)
        self.assertEqual(len(enter_actions), 0)
        self.assertEqual(engine.state_machine.context.last_take_profit_session_date, "2026-03-31")


if __name__ == "__main__":
    unittest.main()
