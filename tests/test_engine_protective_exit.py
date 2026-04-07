from __future__ import annotations

import unittest

import pandas as pd

from corridor.backtest.engine import CorridorBacktestEngine
from corridor.config import CorridorConfig
from corridor.models import ActiveButterfly, ActionType, LayerKind


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

        signal = engine._protective_exit_signal(created_at + pd.Timedelta(days=1), 6400.0)
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

        signal = engine._protective_exit_signal(created_at + pd.Timedelta(days=1), 6460.0)
        self.assertIsNotNone(signal)
        self.assertEqual(signal[0], ActionType.STOP_LOSS)


if __name__ == "__main__":
    unittest.main()
