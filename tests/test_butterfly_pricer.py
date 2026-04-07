from __future__ import annotations

import unittest

import pandas as pd

from corridor.config import CorridorConfig
from corridor.models import ActiveButterfly, LayerKind
from corridor.options.butterfly_pricer import SimplifiedButterflyPricer


class SimplifiedButterflyPricerStressTests(unittest.TestCase):
    def test_conservative_stress_profile_is_harsher_than_base(self) -> None:
        base_cfg = CorridorConfig(symbol="SPX", butterfly_width=30.0)
        stress_cfg = CorridorConfig(
            symbol="SPX",
            butterfly_width=30.0,
            per_contract_slippage=0.05,
            stress_profile="conservative",
            stress_entry_debit_multiplier=1.2,
            stress_peak_value_multiplier=0.7,
            stress_residual_floor_multiplier=0.5,
            stress_slippage_multiplier=2.0,
            stress_close_value_haircut_pct=0.15,
        )

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
        timestamp = created_at + pd.Timedelta(days=1)

        base_pricer = SimplifiedButterflyPricer(base_cfg)
        stress_pricer = SimplifiedButterflyPricer(stress_cfg)

        self.assertGreater(stress_pricer.entry_debit(layer), base_pricer.entry_debit(layer))
        self.assertGreater(stress_pricer.friction_per_layer(layer), base_pricer.friction_per_layer(layer))
        self.assertLess(
            stress_pricer.close_value(layer, spot=6400.0, timestamp=timestamp),
            base_pricer.close_value(layer, spot=6400.0, timestamp=timestamp),
        )

    def test_broken_wing_increases_modeled_max_loss(self) -> None:
        cfg = CorridorConfig(symbol="SPX", butterfly_width=30.0, wing_mode="broken_upper", broken_wing_extra_width=20.0)
        pricer = SimplifiedButterflyPricer(cfg)
        layer = ActiveButterfly(
            layer_id=1,
            kind=LayerKind.PRIMARY,
            center_price=6400.0,
            width=30.0,
            lower_width=30.0,
            upper_width=50.0,
            lower_strike=6370.0,
            body_strike=6400.0,
            upper_strike=6450.0,
            created_at=pd.Timestamp("2026-03-31 14:00:00", tz="UTC"),
            dte=7,
        )

        layer.entry_cost = pricer.entry_cost(layer)
        self.assertGreater(pricer.modeled_max_loss(layer), layer.entry_cost)
        symmetric_layer = ActiveButterfly(
            layer_id=2,
            kind=LayerKind.PRIMARY,
            center_price=6400.0,
            width=30.0,
            lower_width=30.0,
            upper_width=30.0,
            lower_strike=6370.0,
            body_strike=6400.0,
            upper_strike=6430.0,
            created_at=pd.Timestamp("2026-03-31 14:00:00", tz="UTC"),
            dte=7,
        )
        self.assertGreater(pricer.slippage_cost_per_layer(layer), pricer.slippage_cost_per_layer(symmetric_layer))


if __name__ == "__main__":
    unittest.main()
