from __future__ import annotations

import unittest

from corridor.data.ib_contracts import default_center_rounding_for_symbol, get_ib_symbol_spec


class IBContractsTest(unittest.TestCase):
    def test_spx_uses_index_contract_defaults(self) -> None:
        spec = get_ib_symbol_spec("SPX")
        self.assertEqual(spec.underlying_sec_type, "IND")
        self.assertEqual(spec.underlying_exchange, "CBOE")
        self.assertEqual(spec.option_exchange, "SMART")
        self.assertEqual(spec.preferred_trading_classes, ("SPXW", "SPX"))
        self.assertEqual(default_center_rounding_for_symbol("SPX"), 5.0)

    def test_spy_uses_stock_contract_defaults(self) -> None:
        spec = get_ib_symbol_spec("SPY")
        self.assertEqual(spec.underlying_sec_type, "STK")
        self.assertEqual(spec.underlying_exchange, "SMART")
        self.assertEqual(spec.option_exchange, "SMART")
        self.assertEqual(spec.preferred_trading_classes, ("SPY",))
        self.assertEqual(default_center_rounding_for_symbol("SPY"), 1.0)


if __name__ == "__main__":
    unittest.main()
