from __future__ import annotations

import unittest

import pandas as pd

from corridor.backtest.engine import CorridorBacktestEngine
from corridor.config import CorridorConfig
from corridor.models import ActionRecord, ActionType, ActiveButterfly, CorridorState, LayerKind, Regime, TransitionRecord


class BacktestExecutionGateTests(unittest.TestCase):
    def test_paper_spread_gate_filters_primary_entry(self) -> None:
        cfg = CorridorConfig(
            symbol="SPX",
            payoff_mode="simplified",
            butterfly_width=50.0,
            max_acceptable_option_spread=0.40,
            paper_spread_gate_enabled=True,
            paper_spread_gate_mode="hard_reject",
            paper_spread_gate_source="paper_test_summary.json",
            paper_spread_gate_spread_ratio=0.32,
            paper_spread_gate_total_spread=1.5,
            paper_spread_gate_sample_count=8,
            paper_spread_gate_rejection_count=11,
        )
        engine = CorridorBacktestEngine(cfg)
        ts = pd.Timestamp("2026-04-06 16:20:00", tz="UTC")
        layer = ActiveButterfly(
            layer_id=1,
            kind=LayerKind.PRIMARY,
            center_price=6600.0,
            width=50.0,
            lower_width=50.0,
            upper_width=50.0,
            lower_strike=6550.0,
            body_strike=6600.0,
            upper_strike=6650.0,
            created_at=ts,
            dte=7,
        )
        layer.metadata["wing_mode"] = "symmetric"
        engine.state_machine.context.state = CorridorState.ACTIVE_CENTERED
        engine.state_machine.context.current_center = 6600.0
        engine.state_machine.context.active_layers = [layer]

        actions = [
            ActionRecord(
                timestamp=ts,
                symbol="SPX",
                action=ActionType.ENTER_PRIMARY,
                state=CorridorState.ACTIVE_CENTERED,
                price=6602.0,
                center_price=6600.0,
                layer_id=1,
                detail="Opened the primary butterfly corridor layer.",
                metadata={"kind": "PRIMARY"},
            )
        ]
        transitions = [
            TransitionRecord(
                timestamp=ts,
                symbol="SPX",
                from_state=CorridorState.IDLE,
                to_state=CorridorState.ACTIVE_CENTERED,
                reason="Entered range corridor.",
                regime=Regime.RANGE,
                price=6602.0,
                center_price=6600.0,
                drift_count=0,
                layer_count=1,
            )
        ]

        kept = engine._apply_paper_spread_gate(
            "SPX",
            ts,
            6602.0,
            actions,
            transitions,
            {1: layer},
            [1],
        )

        self.assertEqual(kept, [])
        self.assertEqual(engine.state_machine.context.active_layers, [])
        self.assertEqual(engine.state_machine.context.state, CorridorState.IDLE)
        self.assertEqual(len(transitions), 0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, ActionType.ENTRY_FILTERED)
        self.assertEqual(actions[0].metadata["source_action"], ActionType.ENTER_PRIMARY.value)


if __name__ == "__main__":
    unittest.main()
