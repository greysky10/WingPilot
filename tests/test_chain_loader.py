from __future__ import annotations

import unittest
from types import SimpleNamespace

from corridor.options.chain_loader import _qualified_contracts_only


class ChainLoaderTests(unittest.TestCase):
    def test_qualified_contracts_only_filters_missing_conids(self) -> None:
        contracts = [
            SimpleNamespace(conId=123),
            SimpleNamespace(conId=0),
            SimpleNamespace(conId=None),
            SimpleNamespace(conId=456),
        ]

        filtered = _qualified_contracts_only(contracts)

        self.assertEqual([contract.conId for contract in filtered], [123, 456])


if __name__ == "__main__":
    unittest.main()
