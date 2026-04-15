from __future__ import annotations

import unittest

from corridor.config import CorridorConfig
from corridor.options.butterfly_selector import select_butterflies, select_butterflies_with_diagnostics
from corridor.options.chain_loader import OptionQuote


class ButterflySelectorTests(unittest.TestCase):
    def test_strike_selection_returns_centered_candidate(self) -> None:
        cfg = CorridorConfig(butterfly_width=5.0, center_rounding=1.0, max_acceptable_option_spread=0.5)
        quotes = [
            OptionQuote("SPY", "2025-01-17", 595.0, "CALL", 4.8, 4.9, 4.85, 0.2),
            OptionQuote("SPY", "2025-01-17", 600.0, "CALL", 2.4, 2.5, 2.45, 0.2),
            OptionQuote("SPY", "2025-01-17", 605.0, "CALL", 0.8, 0.9, 0.85, 0.2),
        ]
        candidates = select_butterflies(quotes, center_price=600.0, width=5.0, config=cfg)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.body_strike, 600.0)
        self.assertAlmostEqual(candidate.net_debit, 0.8, places=6)

    def test_selector_prefers_better_spread_ratio_for_nearby_body(self) -> None:
        cfg = CorridorConfig(
            butterfly_width=5.0,
            center_rounding=0.5,
            max_acceptable_option_spread=0.6,
            candidate_body_search_steps=1,
        )
        quotes = [
            OptionQuote("SPY", "2025-01-17", 594.0, "CALL", 5.1, 5.2, 5.15, 0.2),
            OptionQuote("SPY", "2025-01-17", 599.0, "CALL", 2.5, 2.66, 2.58, 0.2),
            OptionQuote("SPY", "2025-01-17", 604.0, "CALL", 0.9, 1.0, 0.95, 0.2),
            OptionQuote("SPY", "2025-01-17", 595.0, "CALL", 4.8, 4.9, 4.85, 0.2),
            OptionQuote("SPY", "2025-01-17", 600.0, "CALL", 2.4, 2.5, 2.45, 0.2),
            OptionQuote("SPY", "2025-01-17", 605.0, "CALL", 0.8, 0.9, 0.85, 0.2),
        ]
        candidates = select_butterflies(quotes, center_price=599.5, width=5.0, config=cfg)
        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0].body_strike, 600.0)
        self.assertLess(candidates[0].spread_ratio, candidates[1].spread_ratio)

    def test_selector_returns_diagnostics_for_rejected_candidates(self) -> None:
        cfg = CorridorConfig(
            butterfly_width=5.0,
            center_rounding=1.0,
            max_acceptable_option_spread=0.05,
            candidate_body_search_steps=0,
            wing_mode="adaptive",
            broken_wing_extra_width=2.0,
        )
        quotes = [
            OptionQuote("SPY", "2025-01-17", 595.0, "CALL", 4.8, 4.9, 4.85, 0.2),
            OptionQuote("SPY", "2025-01-17", 600.0, "CALL", 2.4, 2.5, 2.45, 0.2),
            OptionQuote("SPY", "2025-01-17", 605.0, "CALL", 0.8, 0.9, 0.85, 0.2),
        ]

        candidates, diagnostics = select_butterflies_with_diagnostics(quotes, center_price=600.0, width=5.0, config=cfg)

        self.assertEqual(candidates, [])
        self.assertEqual(diagnostics.rejection_counts["spread_too_wide"], 1)
        self.assertGreaterEqual(diagnostics.rejection_counts["missing_legs"], 2)
        self.assertTrue(diagnostics.sample_rejections)

    def test_selector_applies_dte_tiered_spread_caps(self) -> None:
        cfg = CorridorConfig(
            butterfly_width=5.0,
            center_rounding=1.0,
            max_acceptable_option_spread=0.20,
            near_spread_dte_max=10,
            near_max_acceptable_option_spread=0.10,
            far_spread_dte_min=20,
            far_max_acceptable_option_spread=0.25,
        )
        quotes = [
            OptionQuote("SPY", "20260306", 595.0, "CALL", 4.75, 4.80, 4.775, 0.2),
            OptionQuote("SPY", "20260306", 600.0, "CALL", 2.30, 2.35, 2.325, 0.2),
            OptionQuote("SPY", "20260306", 605.0, "CALL", 0.75, 0.80, 0.775, 0.2),
            OptionQuote("SPY", "20260328", 595.0, "CALL", 5.10, 5.15, 5.125, 0.2),
            OptionQuote("SPY", "20260328", 600.0, "CALL", 2.55, 2.60, 2.575, 0.2),
            OptionQuote("SPY", "20260328", 605.0, "CALL", 0.95, 1.00, 0.975, 0.2),
        ]

        candidates = select_butterflies(
            quotes,
            center_price=600.0,
            width=5.0,
            config=cfg,
            reference_date="2026-03-01",
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].expiry, "20260328")
        self.assertEqual(candidates[0].calendar_dte, 27)

    def test_selector_can_build_put_butterfly(self) -> None:
        cfg = CorridorConfig(
            butterfly_width=5.0,
            center_rounding=1.0,
            max_acceptable_option_spread=0.5,
            option_right_preference="put",
        )
        quotes = [
            OptionQuote("SPY", "2025-01-17", 595.0, "PUT", 0.8, 0.9, 0.85, 0.2),
            OptionQuote("SPY", "2025-01-17", 600.0, "PUT", 2.4, 2.5, 2.45, 0.2),
            OptionQuote("SPY", "2025-01-17", 605.0, "PUT", 4.8, 4.9, 4.85, 0.2),
        ]

        candidates = select_butterflies(quotes, center_price=600.0, width=5.0, config=cfg)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].right, "PUT")
        self.assertEqual(candidates[0].body_strike, 600.0)

    def test_selector_auto_can_choose_put_when_calls_missing(self) -> None:
        cfg = CorridorConfig(
            butterfly_width=5.0,
            center_rounding=1.0,
            max_acceptable_option_spread=0.5,
            option_right_preference="auto",
        )
        quotes = [
            OptionQuote("SPY", "2025-01-17", 595.0, "PUT", 0.8, 0.9, 0.85, 0.2),
            OptionQuote("SPY", "2025-01-17", 600.0, "PUT", 2.4, 2.5, 2.45, 0.2),
            OptionQuote("SPY", "2025-01-17", 605.0, "PUT", 4.8, 4.9, 4.85, 0.2),
        ]

        candidates = select_butterflies(quotes, center_price=600.0, width=5.0, config=cfg)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].right, "PUT")


if __name__ == "__main__":
    unittest.main()
