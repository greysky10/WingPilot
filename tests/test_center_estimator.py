from __future__ import annotations

import unittest

import pandas as pd

from corridor.config import CorridorConfig
from corridor.models import CenterMethod
from corridor.strategy.center_estimator import CenterEstimator, round_center


class CenterEstimatorTests(unittest.TestCase):
    def test_round_center_respects_increment(self) -> None:
        self.assertEqual(round_center(598.62, 1.0), 599.0)
        self.assertEqual(round_center(598.24, 0.5), 598.0)
        self.assertEqual(round_center(598.26, 0.5), 598.5)

    def test_vwap_center_estimation(self) -> None:
        cfg = CorridorConfig(center_method=CenterMethod.VWAP, center_lookback=4, center_rounding=1.0)
        estimator = CenterEstimator(cfg)
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-02 15:00", periods=4, freq="5min", tz="UTC"),
                "open": [100, 101, 102, 103],
                "high": [101, 102, 103, 104],
                "low": [99, 100, 101, 102],
                "close": [100.0, 101.0, 102.0, 103.0],
                "volume": [100, 200, 300, 400],
            }
        )
        center = estimator.estimate(frame)
        self.assertIsNotNone(center)
        self.assertEqual(center.center_price, 102.0)

    def test_dynamic_tolerance_uses_atr_floor_and_multiplier(self) -> None:
        cfg = CorridorConfig(
            center_method=CenterMethod.VWAP,
            center_lookback=4,
            atr_lookback=4,
            center_tolerance=2.5,
            center_tolerance_atr_multiplier=2.0,
            center_rounding=1.0,
        )
        estimator = CenterEstimator(cfg)
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-02 15:00", periods=4, freq="5min", tz="UTC"),
                "open": [100.0, 104.0, 101.0, 105.0],
                "high": [106.0, 108.0, 107.0, 109.0],
                "low": [98.0, 100.0, 99.0, 101.0],
                "close": [104.0, 101.0, 105.0, 102.0],
                "volume": [100, 200, 300, 400],
            }
        )

        center = estimator.estimate(frame)

        self.assertIsNotNone(center)
        self.assertGreater(center.actual_tolerance, cfg.center_tolerance)
        self.assertAlmostEqual(center.actual_tolerance, center.diagnostics["actual_tolerance"], places=6)
        self.assertGreater(center.diagnostics["atr"], 0.0)


if __name__ == "__main__":
    unittest.main()
