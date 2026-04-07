from __future__ import annotations

import unittest

import pandas as pd

from corridor.config import CorridorConfig
from corridor.models import CenterEstimate, CenterMethod, CorridorState, Regime, RegimeSnapshot
from corridor.strategy.corridor_state_machine import CorridorStateMachine


class CorridorStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = CorridorConfig(
            center_tolerance=2.0,
            recenter_threshold=4.0,
            drift_persistence_bars=2,
            rebuild_cooldown_minutes=0,
            max_active_butterfly_layers=1,
        )
        self.machine = CorridorStateMachine(self.cfg)
        self.base_ts = pd.Timestamp("2025-01-02 15:00:00", tz="UTC")
        self.center = CenterEstimate(
            timestamp=self.base_ts,
            center_price=100.0,
            lower_band=94.0,
            upper_band=106.0,
            tolerance_low=98.0,
            tolerance_high=102.0,
            method=CenterMethod.VWAP,
            confidence=0.9,
        )

    def test_idle_to_active_to_drifting_to_rebuild(self) -> None:
        range_snapshot = RegimeSnapshot(self.base_ts, Regime.RANGE, 0.01, 0.0, 0.0, 1.0, False, False)
        step1 = self.machine.process_bar("SPY", self.base_ts, 100.0, range_snapshot, self.center)
        self.assertEqual(self.machine.context.state, CorridorState.ACTIVE_CENTERED)
        self.assertEqual(len(step1.transitions), 1)

        step2 = self.machine.process_bar("SPY", self.base_ts + pd.Timedelta(minutes=5), 104.5, range_snapshot, self.center)
        self.assertEqual(self.machine.context.state, CorridorState.DRIFTING)

        step3 = self.machine.process_bar("SPY", self.base_ts + pd.Timedelta(minutes=10), 105.0, range_snapshot, self.center)
        self.assertEqual(self.machine.context.state, CorridorState.REBUILD)

        new_center = CenterEstimate(
            timestamp=self.base_ts + pd.Timedelta(minutes=15),
            center_price=105.0,
            lower_band=99.0,
            upper_band=111.0,
            tolerance_low=103.0,
            tolerance_high=107.0,
            method=CenterMethod.VWAP,
            confidence=0.8,
        )
        step4 = self.machine.process_bar("SPY", self.base_ts + pd.Timedelta(minutes=15), 105.0, range_snapshot, new_center)
        self.assertEqual(self.machine.context.state, CorridorState.ACTIVE_CENTERED)
        self.assertEqual(self.machine.context.current_center, 105.0)

    def test_any_state_aborts_on_trend(self) -> None:
        range_snapshot = RegimeSnapshot(self.base_ts, Regime.RANGE, 0.01, 0.0, 0.0, 1.0, False, False)
        self.machine.process_bar("SPY", self.base_ts, 100.0, range_snapshot, self.center)
        trend_snapshot = RegimeSnapshot(
            self.base_ts + pd.Timedelta(minutes=5),
            Regime.TREND_UP,
            0.03,
            0.01,
            0.02,
            2.0,
            True,
            False,
        )
        self.machine.process_bar("SPY", self.base_ts + pd.Timedelta(minutes=5), 107.0, trend_snapshot, self.center)
        self.assertEqual(self.machine.context.state, CorridorState.ABORT)

    def test_idle_entry_respects_stricter_primary_filters(self) -> None:
        cfg = CorridorConfig(
            center_tolerance=2.0,
            recenter_threshold=4.0,
            drift_persistence_bars=2,
            rebuild_cooldown_minutes=0,
            max_active_butterfly_layers=1,
            primary_entry_min_center_confidence=0.95,
            primary_entry_max_momentum_pct=0.0005,
            primary_entry_max_volume_ratio=1.1,
            primary_entry_end="10:00",
        )
        machine = CorridorStateMachine(cfg)
        center = CenterEstimate(
            timestamp=self.base_ts,
            center_price=100.0,
            lower_band=94.0,
            upper_band=106.0,
            tolerance_low=98.0,
            tolerance_high=102.0,
            method=CenterMethod.VWAP,
            confidence=0.90,
        )
        range_snapshot = RegimeSnapshot(self.base_ts, Regime.RANGE, 0.01, 0.0, 0.001, 1.2, False, False)
        step = machine.process_bar("SPY", self.base_ts, 100.0, range_snapshot, center)
        self.assertEqual(machine.context.state, CorridorState.IDLE)
        self.assertEqual(len(step.actions), 0)

    def test_idle_entry_respects_event_day_block(self) -> None:
        cfg = CorridorConfig(
            center_tolerance=2.0,
            recenter_threshold=4.0,
            drift_persistence_bars=2,
            rebuild_cooldown_minutes=0,
            max_active_butterfly_layers=1,
            skip_event_days=True,
            event_dates=("2025-01-02",),
        )
        machine = CorridorStateMachine(cfg)
        range_snapshot = RegimeSnapshot(self.base_ts, Regime.RANGE, 0.01, 0.0, 0.0, 1.0, False, False)
        step = machine.process_bar("SPY", self.base_ts, 100.0, range_snapshot, self.center)
        self.assertEqual(machine.context.state, CorridorState.IDLE)
        self.assertEqual(len(step.actions), 0)

    def test_broken_upper_entry_creates_asymmetric_strikes(self) -> None:
        cfg = CorridorConfig(
            butterfly_width=30.0,
            wing_mode="broken_upper",
            broken_wing_extra_width=20.0,
            max_active_butterfly_layers=1,
        )
        machine = CorridorStateMachine(cfg)
        range_snapshot = RegimeSnapshot(self.base_ts, Regime.RANGE, 0.01, 0.0, 0.0, 1.0, False, False)
        machine.process_bar("SPY", self.base_ts, 100.0, range_snapshot, self.center)
        layer = machine.context.active_layers[0]
        self.assertEqual(layer.lower_width, 30.0)
        self.assertEqual(layer.upper_width, 50.0)
        self.assertEqual(layer.lower_strike, 70.0)
        self.assertEqual(layer.upper_strike, 150.0)


if __name__ == "__main__":
    unittest.main()
