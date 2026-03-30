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


if __name__ == "__main__":
    unittest.main()
