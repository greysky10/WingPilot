from __future__ import annotations

import unittest

from corridor.config import CorridorConfig
from corridor.options.butterfly_selector import select_butterflies
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


if __name__ == "__main__":
    unittest.main()
