from __future__ import annotations

import unittest

import pandas as pd

from corridor.models import CenterMethod
from weekly_corridor.center_estimator import WeeklyCenterEstimator
from weekly_corridor.config import WeeklyCorridorConfig
from weekly_corridor.models import WeeklyCenterEstimate, WeeklyRegime, WeeklyRegimeSnapshot, WeeklyState
from weekly_corridor.state_machine import WeeklyCorridorStateMachine


class WeeklyCenterEstimatorTests(unittest.TestCase):
    def test_weekly_center_generation_rounds_to_spx_increment(self) -> None:
        cfg = WeeklyCorridorConfig(center_method=CenterMethod.VWAP, center_lookback_bars=8, center_rounding=5.0)
        estimator = WeeklyCenterEstimator(cfg)
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-06 15:00", periods=8, freq="30min", tz="UTC"),
                "open": [6000, 6001, 6002, 6003, 6004, 6005, 6006, 6007],
                "high": [6002, 6003, 6004, 6005, 6006, 6007, 6008, 6009],
                "low": [5998, 5999, 6000, 6001, 6002, 6003, 6004, 6005],
                "close": [6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008],
                "volume": [100] * 8,
            }
        )
        center = estimator.estimate(frame)
        self.assertIsNotNone(center)
        self.assertEqual(center.center_price, 6005.0)
        self.assertEqual(center.tolerance_low, 5955.0)
        self.assertEqual(center.tolerance_high, 6055.0)


class WeeklyStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = WeeklyCorridorConfig(
            butterfly_width=50.0,
            center_spacing=50.0,
            weekly_center_tolerance=50.0,
            max_active_butterflies=4,
            max_adjustments_per_week=1,
            default_dte=12,
            min_remaining_dte=5,
            min_hold_trading_days=4,
            max_hold_trading_days=7,
            valid_trading_start="10:00",
            valid_trading_end="15:30",
            forced_exit_time="15:00",
        )
        self.machine = WeeklyCorridorStateMachine(self.cfg)
        self.center = WeeklyCenterEstimate(
            timestamp=pd.Timestamp("2025-01-06 15:00:00", tz="UTC"),
            center_price=6000.0,
            lower_coverage=5900.0,
            upper_coverage=6100.0,
            tolerance_low=5950.0,
            tolerance_high=6050.0,
            method=CenterMethod.VWAP,
            confidence=0.8,
        )

    def _range_snapshot(self, timestamp: pd.Timestamp) -> WeeklyRegimeSnapshot:
        return WeeklyRegimeSnapshot(
            timestamp=timestamp,
            regime=WeeklyRegime.RANGE,
            width_pct=0.02,
            slope_pct=0.0,
            momentum_pct=0.0,
            breakout_up=False,
            breakout_down=False,
        )

    def test_initial_deployment_creates_three_butterflies(self) -> None:
        timestamp = pd.Timestamp("2025-01-06 15:00:00", tz="UTC")
        step = self.machine.process_bar("SPX", timestamp, 6000.0, self._range_snapshot(timestamp), self.center)
        self.assertEqual(self.machine.context.state, WeeklyState.ACTIVE)
        self.assertEqual(len(self.machine.context.active_butterflies), 3)
        self.assertEqual([layer.body_strike for layer in self.machine.context.active_butterflies], [5950.0, 6000.0, 6050.0])
        self.assertEqual(len(step.actions), 3)

    def test_only_one_adjustment_is_allowed_per_week(self) -> None:
        monday = pd.Timestamp("2025-01-06 15:00:00", tz="UTC")
        self.machine.process_bar("SPX", monday, 6000.0, self._range_snapshot(monday), self.center)
        tuesday = pd.Timestamp("2025-01-07 16:00:00", tz="UTC")
        step1 = self.machine.process_bar("SPX", tuesday, 6065.0, self._range_snapshot(tuesday), self.center)
        self.assertEqual(self.machine.context.state, WeeklyState.ADJUSTED)
        self.assertEqual(len(self.machine.context.active_butterflies), 4)
        self.assertEqual(len(step1.actions), 1)
        wednesday = pd.Timestamp("2025-01-08 16:00:00", tz="UTC")
        step2 = self.machine.process_bar("SPX", wednesday, 6085.0, self._range_snapshot(wednesday), self.center)
        self.assertEqual(self.machine.context.adjustments_this_week, 1)
        self.assertEqual(len(self.machine.context.active_butterflies), 4)
        self.assertEqual(len(step2.actions), 0)

    def test_trend_week_aborts_open_butterflies(self) -> None:
        monday = pd.Timestamp("2025-01-06 15:00:00", tz="UTC")
        self.machine.process_bar("SPX", monday, 6000.0, self._range_snapshot(monday), self.center)
        trend = WeeklyRegimeSnapshot(
            timestamp=pd.Timestamp("2025-01-07 16:00:00", tz="UTC"),
            regime=WeeklyRegime.TREND_UP,
            width_pct=0.08,
            slope_pct=0.03,
            momentum_pct=0.02,
            breakout_up=True,
            breakout_down=False,
        )
        step = self.machine.process_bar("SPX", trend.timestamp, 6120.0, trend, self.center)
        self.assertEqual(self.machine.context.state, WeeklyState.ABORTED)
        self.assertEqual(len(self.machine.context.active_butterflies), 0)
        self.assertEqual(len(step.actions), 3)

    def test_forced_weekly_exit_closes_the_corridor(self) -> None:
        monday = pd.Timestamp("2025-01-06 15:00:00", tz="UTC")
        self.machine.process_bar("SPX", monday, 6000.0, self._range_snapshot(monday), self.center)
        friday = pd.Timestamp("2025-01-10 20:15:00", tz="UTC")
        step = self.machine.process_bar("SPX", friday, 6005.0, self._range_snapshot(friday), self.center)
        self.assertEqual(self.machine.context.state, WeeklyState.EXITED)
        self.assertEqual(len(self.machine.context.active_butterflies), 0)
        self.assertEqual(len(step.actions), 3)


if __name__ == "__main__":
    unittest.main()
