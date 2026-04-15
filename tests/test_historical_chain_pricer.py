from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from corridor.backtest.engine import CorridorBacktestEngine
from corridor.config import CorridorConfig
from corridor.models import ActionRecord, ActionType, ActiveButterfly, CorridorState, LayerKind
from corridor.options.historical_chain import HistoricalChainButterflyPricer


def _sample_chain_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "option_ticker": "O:SPXW260109C04990000",
                "underlying": "SPX",
                "expiry": "2026-01-09",
                "strike": 4990.0,
                "type": "call",
                "open": 18.0,
                "high": 18.5,
                "low": 17.5,
                "close": 18.0,
                "volume": 100,
            },
            {
                "date": "2026-01-02",
                "option_ticker": "O:SPXW260109C05000000",
                "underlying": "SPX",
                "expiry": "2026-01-09",
                "strike": 5000.0,
                "type": "call",
                "open": 12.0,
                "high": 12.5,
                "low": 11.5,
                "close": 12.0,
                "volume": 100,
            },
            {
                "date": "2026-01-02",
                "option_ticker": "O:SPXW260109C05010000",
                "underlying": "SPX",
                "expiry": "2026-01-09",
                "strike": 5010.0,
                "type": "call",
                "open": 8.0,
                "high": 8.5,
                "low": 7.5,
                "close": 8.0,
                "volume": 100,
            },
            {
                "date": "2026-01-03",
                "option_ticker": "O:SPXW260109C04990000",
                "underlying": "SPX",
                "expiry": "2026-01-09",
                "strike": 4990.0,
                "type": "call",
                "open": 20.0,
                "high": 20.5,
                "low": 19.5,
                "close": 20.0,
                "volume": 100,
            },
            {
                "date": "2026-01-03",
                "option_ticker": "O:SPXW260109C05000000",
                "underlying": "SPX",
                "expiry": "2026-01-09",
                "strike": 5000.0,
                "type": "call",
                "open": 11.0,
                "high": 11.5,
                "low": 10.5,
                "close": 11.0,
                "volume": 100,
            },
            {
                "date": "2026-01-03",
                "option_ticker": "O:SPXW260109C05010000",
                "underlying": "SPX",
                "expiry": "2026-01-09",
                "strike": 5010.0,
                "type": "call",
                "open": 6.0,
                "high": 6.5,
                "low": 5.5,
                "close": 6.0,
                "volume": 100,
            },
            {
                "date": "2026-01-02",
                "option_ticker": "O:SPXW260109P04990000",
                "underlying": "SPX",
                "expiry": "2026-01-09",
                "strike": 4990.0,
                "type": "put",
                "open": 8.0,
                "high": 8.5,
                "low": 7.5,
                "close": 8.0,
                "volume": 100,
            },
            {
                "date": "2026-01-02",
                "option_ticker": "O:SPXW260109P05000000",
                "underlying": "SPX",
                "expiry": "2026-01-09",
                "strike": 5000.0,
                "type": "put",
                "open": 12.0,
                "high": 12.5,
                "low": 11.5,
                "close": 12.0,
                "volume": 100,
            },
            {
                "date": "2026-01-02",
                "option_ticker": "O:SPXW260109P05010000",
                "underlying": "SPX",
                "expiry": "2026-01-09",
                "strike": 5010.0,
                "type": "put",
                "open": 18.0,
                "high": 18.5,
                "low": 17.5,
                "close": 18.0,
                "volume": 100,
            },
            {
                "date": "2026-01-03",
                "option_ticker": "O:SPXW260109P04990000",
                "underlying": "SPX",
                "expiry": "2026-01-09",
                "strike": 4990.0,
                "type": "put",
                "open": 6.0,
                "high": 6.5,
                "low": 5.5,
                "close": 6.0,
                "volume": 100,
            },
            {
                "date": "2026-01-03",
                "option_ticker": "O:SPXW260109P05000000",
                "underlying": "SPX",
                "expiry": "2026-01-09",
                "strike": 5000.0,
                "type": "put",
                "open": 11.0,
                "high": 11.5,
                "low": 10.5,
                "close": 11.0,
                "volume": 100,
            },
            {
                "date": "2026-01-03",
                "option_ticker": "O:SPXW260109P05010000",
                "underlying": "SPX",
                "expiry": "2026-01-09",
                "strike": 5010.0,
                "type": "put",
                "open": 20.0,
                "high": 20.5,
                "low": 19.5,
                "close": 20.0,
                "volume": 100,
            },
        ]
    )


def _build_layer(center_price: float = 5000.0) -> ActiveButterfly:
    return ActiveButterfly(
        layer_id=1,
        kind=LayerKind.PRIMARY,
        center_price=center_price,
        width=10.0,
        lower_width=10.0,
        upper_width=10.0,
        lower_strike=center_price - 10.0,
        body_strike=center_price,
        upper_strike=center_price + 10.0,
        created_at=pd.Timestamp("2026-01-02T15:00:00Z"),
        dte=7,
    )


class HistoricalChainPricerTests(unittest.TestCase):
    def _write_dataset(self, tmp_dir: str) -> Path:
        path = Path(tmp_dir) / "historical_chain.csv"
        _sample_chain_frame().to_csv(path, index=False)
        return path

    def _config(self, dataset_path: Path) -> CorridorConfig:
        return CorridorConfig(
            symbol="SPX",
            center_rounding=5.0,
            butterfly_width=10.0,
            dte_min=4,
            dte_max=10,
            payoff_mode="historical_chain",
            historical_chain_path=str(dataset_path),
        )

    def test_historical_chain_pricer_attaches_candidate_and_marks_combo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = self._write_dataset(tmp_dir)
            pricer = HistoricalChainButterflyPricer.from_config(self._config(dataset_path))
            layer = _build_layer()

            selection = pricer.attach_candidate(layer, "SPX", pd.Timestamp("2026-01-02T15:00:00Z"))

            self.assertIsNotNone(selection)
            self.assertEqual(layer.metadata["historical_chain_lower_ticker"], "O:SPXW260109C04990000")
            self.assertEqual(layer.metadata["historical_chain_body_ticker"], "O:SPXW260109C05000000")
            self.assertEqual(layer.metadata["historical_chain_upper_ticker"], "O:SPXW260109C05010000")
            self.assertAlmostEqual(pricer.entry_debit(layer), 2.0)
            self.assertAlmostEqual(pricer.mark_to_model(layer, 5005.0, pd.Timestamp("2026-01-03T15:00:00Z")), 4.0)

    def test_engine_historical_chain_selection_enriches_open_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = self._write_dataset(tmp_dir)
            engine = CorridorBacktestEngine(self._config(dataset_path))
            layer = _build_layer()
            engine.state_machine.context.active_layers = [layer]
            engine.state_machine.context.state = CorridorState.ACTIVE_CENTERED
            engine.state_machine.context.current_center = 5000.0
            actions = [
                ActionRecord(
                    timestamp=pd.Timestamp("2026-01-02T15:00:00Z"),
                    symbol="SPX",
                    action=ActionType.ENTER_PRIMARY,
                    state=CorridorState.ACTIVE_CENTERED,
                    price=5000.0,
                    center_price=5000.0,
                    layer_id=1,
                    detail="open",
                    metadata={},
                )
            ]

            kept = engine._apply_historical_chain_selection(
                symbol="SPX",
                timestamp=pd.Timestamp("2026-01-02T15:00:00Z"),
                price=5000.0,
                actions=actions,
                transitions=[],
                current_layers={1: layer},
                opened_ids=[1],
            )

            self.assertEqual(kept, [1])
            self.assertEqual(actions[0].metadata["historical_chain_expiry"], "2026-01-09")
            self.assertEqual(actions[0].metadata["historical_chain_lower_ticker"], "O:SPXW260109C04990000")

    def test_engine_historical_chain_selection_filters_unmatched_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = self._write_dataset(tmp_dir)
            engine = CorridorBacktestEngine(self._config(dataset_path))
            layer = _build_layer(center_price=5100.0)
            engine.state_machine.context.active_layers = [layer]
            engine.state_machine.context.state = CorridorState.ACTIVE_CENTERED
            engine.state_machine.context.current_center = 5100.0
            actions = [
                ActionRecord(
                    timestamp=pd.Timestamp("2026-01-02T15:00:00Z"),
                    symbol="SPX",
                    action=ActionType.ENTER_PRIMARY,
                    state=CorridorState.ACTIVE_CENTERED,
                    price=5100.0,
                    center_price=5100.0,
                    layer_id=1,
                    detail="open",
                    metadata={},
                )
            ]

            kept = engine._apply_historical_chain_selection(
                symbol="SPX",
                timestamp=pd.Timestamp("2026-01-02T15:00:00Z"),
                price=5100.0,
                actions=actions,
                transitions=[],
                current_layers={1: layer},
                opened_ids=[1],
            )

            self.assertEqual(kept, [])
            self.assertEqual(len(engine.state_machine.context.active_layers), 0)
            self.assertEqual(engine.state_machine.context.state, CorridorState.IDLE)
            self.assertEqual(actions[-1].action, ActionType.ENTRY_FILTERED)

    def test_historical_chain_pricer_can_attach_put_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = self._write_dataset(tmp_dir)
            cfg = self._config(dataset_path)
            cfg.option_right_preference = "put"
            pricer = HistoricalChainButterflyPricer.from_config(cfg)
            layer = _build_layer()

            selection = pricer.attach_candidate(layer, "SPX", pd.Timestamp("2026-01-02T15:00:00Z"))

            self.assertIsNotNone(selection)
            self.assertEqual(layer.metadata["historical_chain_right"], "PUT")
            self.assertEqual(layer.metadata["historical_chain_lower_ticker"], "O:SPXW260109P04990000")
            self.assertEqual(layer.metadata["historical_chain_body_ticker"], "O:SPXW260109P05000000")
            self.assertEqual(layer.metadata["historical_chain_upper_ticker"], "O:SPXW260109P05010000")
            self.assertAlmostEqual(pricer.entry_debit(layer), 2.0)
            self.assertAlmostEqual(pricer.mark_to_model(layer, 4995.0, pd.Timestamp("2026-01-03T15:00:00Z")), 4.0)


if __name__ == "__main__":
    unittest.main()
