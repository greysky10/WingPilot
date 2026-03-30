from __future__ import annotations

import unittest

import pandas as pd

from corridor.config import CorridorConfig
from corridor.models import CenterEstimate, CenterMethod, Regime, RegimeSnapshot
from corridor.strategy.recenter_rules import RecenterRuleEngine


class RecenterRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = CorridorConfig(
            center_tolerance=2.0,
            recenter_threshold=4.0,
            drift_persistence_bars=3,
            rebuild_cooldown_minutes=30,
            abort_volume_threshold=1.6,
            abort_momentum_threshold=0.01,
        )
        self.engine = RecenterRuleEngine(self.cfg)
        self.timestamp = pd.Timestamp("2025-01-02 15:00:00", tz="UTC")
        self.center = CenterEstimate(
            timestamp=self.timestamp,
            center_price=100.0,
            lower_band=94.0,
            upper_band=106.0,
            tolerance_low=98.0,
            tolerance_high=102.0,
            method=CenterMethod.VWAP,
            confidence=0.9,
        )

    def test_drift_persistence_triggers_rebuild(self) -> None:
        snapshot = RegimeSnapshot(self.timestamp, Regime.RANGE, 0.01, 0.0, 0.001, 1.0, False, False)
        assessment = self.engine.evaluate(self.timestamp, 105.5, self.center, snapshot, 2, None)
        self.assertTrue(assessment.should_rebuild)
        self.assertEqual(assessment.next_drift_count, 3)

    def test_rebuild_cooldown_blocks_rebuild(self) -> None:
        snapshot = RegimeSnapshot(self.timestamp, Regime.RANGE, 0.01, 0.0, 0.001, 1.0, False, False)
        assessment = self.engine.evaluate(
            self.timestamp,
            105.5,
            self.center,
            snapshot,
            2,
            self.timestamp - pd.Timedelta(minutes=10),
        )
        self.assertFalse(assessment.should_rebuild)
        self.assertFalse(assessment.cooldown_ready)

    def test_abort_trigger_fires_on_trend_transition(self) -> None:
        snapshot = RegimeSnapshot(self.timestamp, Regime.TREND_UP, 0.03, 0.02, 0.015, 2.0, True, False)
        assessment = self.engine.evaluate(self.timestamp, 107.0, self.center, snapshot, 1, None)
        self.assertTrue(assessment.should_abort)
        self.assertIn("TREND_UP", assessment.abort_reason)


if __name__ == "__main__":
    unittest.main()
