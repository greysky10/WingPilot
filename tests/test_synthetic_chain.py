from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from corridor.config import CorridorConfig
from corridor.models import ActiveButterfly, LayerKind
from corridor.options.synthetic_chain import SyntheticChainButterflyPricer, load_synthetic_chain_calibration


class SyntheticChainCalibrationTests(unittest.TestCase):
    def test_synthetic_chain_calibration_and_pricer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state_path = root / "paper_state.json"
            report_path = root / "paper_daily_report.json"

            state_payload = {
                "timestamp": "2026-04-09T19:50:00+00:00",
                "symbol": "SPX",
                "price": 6829.77,
                "candidates": [
                    {
                        "expiry": "20260414",
                        "right": "CALL",
                        "wing_mode": "broken_upper",
                        "lower_width": 100.0,
                        "upper_width": 120.0,
                        "lower_strike": 6725.0,
                        "body_strike": 6825.0,
                        "upper_strike": 6945.0,
                        "net_debit": 34.6,
                        "total_spread": 2.2,
                        "spread_ratio": 0.0636,
                        "body_distance": 0.0,
                    },
                    {
                        "expiry": "20260415",
                        "right": "CALL",
                        "wing_mode": "broken_upper",
                        "lower_width": 100.0,
                        "upper_width": 120.0,
                        "lower_strike": 6725.0,
                        "body_strike": 6825.0,
                        "upper_strike": 6945.0,
                        "net_debit": 29.45,
                        "total_spread": 2.3,
                        "spread_ratio": 0.0781,
                        "body_distance": 0.0,
                    },
                ],
            }
            report_payload = {
                "candidate_diagnostics": {
                    "attempted_structures": 20,
                    "rejection_counts": {"spread_too_wide": 4},
                    "sample_rejections": [
                        {
                            "reason": "spread_too_wide",
                            "body_strike": 6815.0,
                            "lower_strike": 6715.0,
                            "upper_strike": 6935.0,
                            "total_spread": 14.7,
                        }
                    ],
                }
            }
            state_path.write_text(json.dumps(state_payload), encoding="utf-8")
            report_path.write_text(json.dumps(report_payload), encoding="utf-8")

            cfg = CorridorConfig(
                symbol="SPX",
                payoff_mode="synthetic_chain",
                butterfly_width=100.0,
                wing_mode="broken_upper",
                broken_wing_extra_width=20.0,
                synthetic_chain_state_path=str(state_path),
                synthetic_chain_report_path=str(report_path),
            )

            calibration = load_synthetic_chain_calibration(cfg)
            self.assertEqual(calibration.symbol, "SPX")
            self.assertGreater(len(calibration.anchors), 0)
            self.assertGreaterEqual(calibration.rejection_spread_multiplier, 1.0)

            pricer = SyntheticChainButterflyPricer.from_config(cfg)
            created_at = pd.Timestamp("2026-04-09 18:25:00", tz="UTC")
            layer = ActiveButterfly(
                layer_id=1,
                kind=LayerKind.PRIMARY,
                center_price=6825.0,
                width=100.0,
                lower_width=100.0,
                upper_width=120.0,
                lower_strike=6725.0,
                body_strike=6825.0,
                upper_strike=6945.0,
                created_at=created_at,
                dte=5,
            )
            layer.entry_debit = pricer.entry_debit(layer)
            layer.entry_friction_cost = pricer.friction_per_layer(layer)
            layer.entry_cost = pricer.entry_cost(layer)

            self.assertGreater(layer.entry_debit, 0.0)
            self.assertGreater(pricer.estimated_total_spread(layer), 0.0)
            self.assertGreater(pricer.mark_to_model(layer, 6825.0, created_at + pd.Timedelta(days=1)), 0.0)
            self.assertLess(pricer.close_value(layer, 7000.0, created_at + pd.Timedelta(days=5)), layer.entry_cost)


if __name__ == "__main__":
    unittest.main()
