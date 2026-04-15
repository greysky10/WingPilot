from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from corridor.data.massive_options import (
    BackfillCheckpoint,
    MassiveBackfillConfig,
    MassiveClientConfig,
    MassiveRESTClient,
    StrategyUniverseConfig,
    _prepare_contracts_frame,
    _resolve_contract_history_window,
    assemble_final_dataset,
    normalize_option_bars,
    write_dataframe,
)


class _PaginationStubClient(MassiveRESTClient):
    def __init__(self, payloads: dict[str, dict]) -> None:
        self.payloads = payloads
        self.seen_urls: list[str] = []
        super().__init__(MassiveClientConfig(api_key="test-key"))

    def _build_url(self, path_or_url: str, params=None) -> str:  # type: ignore[override]
        return path_or_url

    def request_json(self, path_or_url: str, params=None) -> dict:  # type: ignore[override]
        self.seen_urls.append(path_or_url)
        return self.payloads[path_or_url]


class _RetryStubClient(MassiveRESTClient):
    def __init__(self) -> None:
        super().__init__(MassiveClientConfig(api_key="test-key", max_retries=3, retry_backoff_seconds=0.0))
        self.calls = 0

    def _build_url(self, path_or_url: str, params=None) -> str:  # type: ignore[override]
        return str(path_or_url)

    def _request_once(self, url: str) -> dict:  # type: ignore[override]
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("read timed out")
        return {"status": "OK", "results": []}


class MassiveOptionsTests(unittest.TestCase):
    def _write_strategy_bars(self, path: Path) -> None:
        pd.DataFrame(
            [
                {
                    "timestamp": "2025-04-10T13:30:00+00:00",
                    "symbol": "SPX",
                    "open": 5005.0,
                    "high": 5010.0,
                    "low": 5000.0,
                    "close": 5008.0,
                    "volume": 1000.0,
                },
                {
                    "timestamp": "2025-04-10T19:55:00+00:00",
                    "symbol": "SPX",
                    "open": 5008.0,
                    "high": 5010.0,
                    "low": 5004.0,
                    "close": 5006.0,
                    "volume": 900.0,
                },
                {
                    "timestamp": "2025-04-11T13:30:00+00:00",
                    "symbol": "SPX",
                    "open": 5010.0,
                    "high": 5015.0,
                    "low": 5005.0,
                    "close": 5012.0,
                    "volume": 1100.0,
                },
                {
                    "timestamp": "2025-04-11T19:55:00+00:00",
                    "symbol": "SPX",
                    "open": 5012.0,
                    "high": 5015.0,
                    "low": 5008.0,
                    "close": 5011.0,
                    "volume": 850.0,
                },
            ]
        ).to_csv(path, index=False)

    def test_iter_paginated_follows_next_url(self) -> None:
        client = _PaginationStubClient(
            {
                "page-1": {
                    "results": [{"ticker": "ONE"}, {"ticker": "TWO"}],
                    "next_url": "page-2",
                    "status": "OK",
                },
                "page-2": {
                    "results": [{"ticker": "THREE"}],
                    "status": "OK",
                },
            }
        )

        results = list(client.iter_paginated("page-1"))

        self.assertEqual([row["ticker"] for row in results], ["ONE", "TWO", "THREE"])
        self.assertEqual(client.seen_urls, ["page-1", "page-2"])

    def test_request_json_retries_timeout_error(self) -> None:
        client = _RetryStubClient()

        payload = client.request_json("timeout-page")

        self.assertEqual(payload["status"], "OK")
        self.assertEqual(client.calls, 2)

    def test_prepare_contracts_frame_falls_back_to_second_underlying(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            cfg = MassiveBackfillConfig(
                start_date=date(2025, 4, 10),
                end_date=date(2026, 4, 10),
                output_dir=output_dir,
                contract_underlyings=("I:SPX", "SPX"),
                output_format="csv",
                resume=False,
            )
            checkpoint = BackfillCheckpoint(
                contract_underlyings=["I:SPX", "SPX"],
                start_date=cfg.start_date.isoformat(),
                end_date=cfg.end_date.isoformat(),
                output_format="csv",
            )
            contracts_path = output_dir / "contracts.csv"
            spx_frame = pd.DataFrame(
                [
                    {
                        "ticker": "O:SPXTEST",
                        "underlying_ticker": "SPX",
                        "contract_type": "call",
                        "expiration_date": "2026-04-17",
                        "strike_price": 5000.0,
                    }
                ]
            )

            with patch("corridor.data.massive_options.fetch_contracts_for_underlying") as fetch_mock:
                fetch_mock.side_effect = [pd.DataFrame(), spx_frame]
                frame, selected = _prepare_contracts_frame(
                    client=object(),  # type: ignore[arg-type]
                    cfg=cfg,
                    contracts_path=contracts_path,
                    checkpoint=checkpoint,
                    output_format="csv",
                )

            self.assertEqual(selected, "SPX")
            self.assertEqual(frame["ticker"].tolist(), ["O:SPXTEST"])
            self.assertTrue(contracts_path.exists())

    def test_prepare_contracts_frame_strategy_only_uses_narrow_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            bars_path = output_dir / "spx_bars.csv"
            self._write_strategy_bars(bars_path)
            strategy_cfg = StrategyUniverseConfig(
                bars_csv=bars_path,
                symbol="SPX",
                contract_types=("call",),
                dte_min=4,
                dte_max=6,
                center_rounding=5.0,
                butterfly_width=10.0,
                strike_buffer_points=0.0,
                slice_days=2,
            )
            cfg = MassiveBackfillConfig(
                start_date=date(2025, 4, 10),
                end_date=date(2025, 4, 11),
                output_dir=output_dir,
                contract_underlyings=("SPX",),
                output_format="csv",
                resume=False,
                strategy_universe=strategy_cfg,
            )
            checkpoint = BackfillCheckpoint(
                contract_underlyings=["SPX"],
                start_date=cfg.start_date.isoformat(),
                end_date=cfg.end_date.isoformat(),
                output_format="csv",
            )
            contracts_path = output_dir / "contracts.csv"
            fetched = pd.DataFrame(
                [
                    {
                        "ticker": "O:SPXNEAR",
                        "underlying_ticker": "SPX",
                        "contract_type": "call",
                        "expiration_date": "2025-04-16",
                        "strike_price": 5020.0,
                    },
                    {
                        "ticker": "O:SPXFAR",
                        "underlying_ticker": "SPX",
                        "contract_type": "call",
                        "expiration_date": "2025-04-16",
                        "strike_price": 5200.0,
                    },
                ]
            )

            with patch("corridor.data.massive_options.fetch_contracts_for_underlying") as fetch_mock:
                fetch_mock.return_value = fetched
                frame, selected = _prepare_contracts_frame(
                    client=object(),  # type: ignore[arg-type]
                    cfg=cfg,
                    contracts_path=contracts_path,
                    checkpoint=checkpoint,
                    output_format="csv",
                )

            self.assertEqual(selected, "SPX")
            self.assertEqual(frame["ticker"].tolist(), ["O:SPXNEAR"])
            kwargs = fetch_mock.call_args.kwargs
            self.assertEqual(kwargs["contract_type"], "call")
            self.assertEqual(kwargs["start_date"], date(2025, 4, 14))
            self.assertEqual(kwargs["end_date"], date(2025, 4, 17))
            self.assertEqual(kwargs["strike_price_gte"], 4990.0)
            self.assertEqual(kwargs["strike_price_lte"], 5025.0)
            self.assertTrue(contracts_path.exists())

    def test_strategy_only_history_window_uses_dte_max_not_full_backfill_range(self) -> None:
        cfg = MassiveBackfillConfig(
            start_date=date(2025, 4, 10),
            end_date=date(2025, 5, 1),
            output_dir=Path("."),
            contract_underlyings=("SPX",),
            output_format="csv",
            resume=False,
            strategy_universe=StrategyUniverseConfig(
                bars_csv=Path("unused.csv"),
                symbol="SPX",
                contract_types=("call",),
                dte_min=4,
                dte_max=6,
            ),
        )

        window = _resolve_contract_history_window({"expiration_date": "2025-04-17"}, cfg)

        self.assertEqual(window, (date(2025, 4, 11), date(2025, 4, 17)))

    def test_normalize_option_bars_builds_daily_chain_rows(self) -> None:
        contract = {
            "ticker": "O:SPXTEST",
            "underlying_ticker": "SPX",
            "contract_type": "put",
            "expiration_date": "2026-04-17",
            "strike_price": 4950.0,
        }
        bars = pd.DataFrame(
            [
                {
                    "t": 1767389400000,
                    "o": 12.5,
                    "h": 14.0,
                    "l": 11.75,
                    "c": 13.25,
                    "v": 1234,
                    "n": 56,
                    "vw": 13.1,
                }
            ]
        )

        normalized = normalize_option_bars(contract, bars)

        self.assertEqual(normalized["date"].tolist(), ["2026-01-02"])
        self.assertEqual(normalized["option_ticker"].tolist(), ["O:SPXTEST"])
        self.assertEqual(normalized["underlying"].tolist(), ["SPX"])
        self.assertEqual(normalized["expiry"].tolist(), ["2026-04-17"])
        self.assertEqual(normalized["type"].tolist(), ["put"])
        self.assertAlmostEqual(float(normalized["close"].iloc[0]), 13.25)
        self.assertEqual(int(normalized["volume"].iloc[0]), 1234)

    def test_assemble_final_dataset_dedupes_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            parts_dir = Path(tmp_dir)
            part_one = pd.DataFrame(
                [
                    {
                        "date": "2026-01-02",
                        "option_ticker": "O:ONE",
                        "underlying": "SPX",
                        "expiry": "2026-01-16",
                        "strike": 5000.0,
                        "type": "call",
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.5,
                        "close": 10.5,
                        "volume": 100,
                        "transactions": 5,
                        "vwap": 10.2,
                    }
                ]
            )
            part_two = pd.DataFrame(
                [
                    {
                        "date": "2026-01-02",
                        "option_ticker": "O:ONE",
                        "underlying": "SPX",
                        "expiry": "2026-01-16",
                        "strike": 5000.0,
                        "type": "call",
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.5,
                        "close": 10.5,
                        "volume": 100,
                        "transactions": 5,
                        "vwap": 10.2,
                    },
                    {
                        "date": "2026-01-03",
                        "option_ticker": "O:TWO",
                        "underlying": "SPX",
                        "expiry": "2026-01-16",
                        "strike": 5050.0,
                        "type": "put",
                        "open": 9.0,
                        "high": 10.0,
                        "low": 8.5,
                        "close": 9.25,
                        "volume": 80,
                        "transactions": 4,
                        "vwap": 9.1,
                    },
                ]
            )
            write_dataframe(part_one, parts_dir / "bars_part_00000.csv", "csv")
            write_dataframe(part_two, parts_dir / "bars_part_00001.csv", "csv")

            merged = assemble_final_dataset(parts_dir, "csv")

            self.assertEqual(merged["option_ticker"].tolist(), ["O:ONE", "O:TWO"])
            self.assertEqual(merged["date"].tolist(), ["2026-01-02", "2026-01-03"])


if __name__ == "__main__":
    unittest.main()
