from __future__ import annotations

import unittest

import pandas as pd

from corridor.backtest.metrics import compute_metrics
from corridor.config import CorridorConfig
from corridor.models import ActionRecord, ActionType, CorridorState, EquityPoint, Regime


class CorridorMetricsTests(unittest.TestCase):
    def test_summary_separates_modeled_and_capital_normalized_fields(self) -> None:
        cfg = CorridorConfig(
            symbol="SPY",
            timeframe="5 mins",
            starting_capital=100000.0,
            contracts_per_layer=2,
            option_multiplier=100,
        )
        ts1 = pd.Timestamp("2025-01-02 15:00:00", tz="UTC")
        ts2 = ts1 + pd.Timedelta(minutes=5)
        ts3 = ts1 + pd.Timedelta(days=1)
        ts4 = ts3 + pd.Timedelta(minutes=5)
        ts5 = ts4 + pd.Timedelta(minutes=5)

        actions = [
            ActionRecord(
                timestamp=ts1,
                symbol="SPY",
                action=ActionType.REBUILD_REQUESTED,
                state=CorridorState.DRIFTING,
                price=100.0,
                center_price=100.0,
                layer_id=None,
                detail="request rebuild",
            ),
            ActionRecord(
                timestamp=ts2,
                symbol="SPY",
                action=ActionType.REBUILT,
                state=CorridorState.ACTIVE_CENTERED,
                price=100.5,
                center_price=100.0,
                layer_id=1,
                detail="Established a fresh primary butterfly corridor layer.",
                metadata={"entry_cost": 1.156, "friction_cost": 0.056, "entry_friction_cost": 0.056},
            ),
            ActionRecord(
                timestamp=ts3,
                symbol="SPY",
                action=ActionType.SESSION_FLUSH,
                state=CorridorState.IDLE,
                price=101.0,
                center_price=100.0,
                layer_id=1,
                detail="Session window closed.",
                metadata={"realized_pnl": 0.8},
            ),
            ActionRecord(
                timestamp=ts5,
                symbol="SPY",
                action=ActionType.ABORTED,
                state=CorridorState.ABORT,
                price=99.0,
                center_price=100.0,
                layer_id=2,
                detail="Regime turned TREND_DOWN.",
                metadata={"realized_pnl": -0.3},
            ),
        ]

        equity_curve = [
            EquityPoint(
                timestamp=ts1,
                symbol="SPY",
                price=100.0,
                regime=Regime.RANGE,
                state=CorridorState.ACTIVE_CENTERED,
                bar_pnl=0.5,
                realized_pnl=0.0,
                unrealized_pnl=0.5,
                gross_realized_pnl=0.0,
                gross_unrealized_pnl=0.6,
                gross_total_equity=0.6,
                total_equity=0.5,
                modeled_capital_at_risk=1.156,
                corridor_occupancy=True,
                active_layers=1,
            ),
            EquityPoint(
                timestamp=ts2,
                symbol="SPY",
                price=100.5,
                regime=Regime.RANGE,
                state=CorridorState.ACTIVE_CENTERED,
                bar_pnl=0.5,
                realized_pnl=0.0,
                unrealized_pnl=1.0,
                gross_realized_pnl=0.0,
                gross_unrealized_pnl=1.2,
                gross_total_equity=1.2,
                total_equity=1.0,
                modeled_capital_at_risk=1.156,
                corridor_occupancy=False,
                active_layers=1,
            ),
            EquityPoint(
                timestamp=ts3,
                symbol="SPY",
                price=101.0,
                regime=Regime.RANGE,
                state=CorridorState.IDLE,
                bar_pnl=-0.2,
                realized_pnl=0.8,
                unrealized_pnl=0.0,
                gross_realized_pnl=1.0,
                gross_unrealized_pnl=0.0,
                gross_total_equity=1.0,
                total_equity=0.8,
                modeled_capital_at_risk=0.0,
                corridor_occupancy=True,
                active_layers=0,
            ),
            EquityPoint(
                timestamp=ts4,
                symbol="SPY",
                price=99.5,
                regime=Regime.TREND_DOWN,
                state=CorridorState.ABORT,
                bar_pnl=-0.4,
                realized_pnl=0.4,
                unrealized_pnl=0.0,
                gross_realized_pnl=0.7,
                gross_unrealized_pnl=0.0,
                gross_total_equity=0.7,
                total_equity=0.4,
                modeled_capital_at_risk=0.5,
                corridor_occupancy=False,
                active_layers=0,
            ),
            EquityPoint(
                timestamp=ts5,
                symbol="SPY",
                price=99.0,
                regime=Regime.TREND_DOWN,
                state=CorridorState.ABORT,
                bar_pnl=1.6,
                realized_pnl=2.0,
                unrealized_pnl=0.0,
                gross_realized_pnl=2.4,
                gross_unrealized_pnl=0.0,
                gross_total_equity=2.4,
                total_equity=2.0,
                modeled_capital_at_risk=0.0,
                corridor_occupancy=False,
                active_layers=0,
            ),
        ]

        summary = compute_metrics(cfg, actions, equity_curve)

        self.assertEqual(summary["total_return"], 2.0)
        self.assertEqual(summary["total_return_units"], "modeled_points")
        self.assertEqual(summary["model_points"], 2.0)
        self.assertEqual(summary["gross_modeled_pnl"], 2.4)
        self.assertEqual(summary["net_modeled_pnl"], 2.0)
        self.assertEqual(summary["dollar_pnl_per_1_lot"], 200.0)
        self.assertEqual(summary["net_dollar_pnl"], 400.0)
        self.assertEqual(summary["gross_dollar_pnl"], 480.0)
        self.assertEqual(summary["max_gross_deployment_dollars"], 231.2)
        self.assertEqual(summary["return_on_capital"], 0.004)
        self.assertEqual(summary["max_modeled_state_capital_at_risk"], 1.156)
        self.assertEqual(summary["max_modeled_execution_capital_at_risk"], 1.156)
        self.assertEqual(summary["max_modeled_close_friction_reserve"], 0.056)
        self.assertEqual(summary["max_modeled_capital_at_risk"], 1.212)
        self.assertAlmostEqual(summary["return_on_max_risk"], 400.0 / 242.4, places=6)
        self.assertEqual(summary["worst_day_pnl"], 1.0)
        self.assertEqual(summary["worst_day_pnl_dollars"], 200.0)
        self.assertEqual(summary["best_day_pnl"], 1.0)
        self.assertEqual(summary["best_day_pnl_dollars"], 200.0)
        self.assertEqual(summary["profit_factor_by_day"], None)
        self.assertEqual(summary["closed_layers"], 2)
        self.assertEqual(summary["winning_layers"], 1)
        self.assertEqual(summary["losing_layers"], 1)
        self.assertEqual(summary["flat_layers"], 0)
        self.assertEqual(summary["gross_winners"], 0.8)
        self.assertEqual(summary["gross_losers"], 0.3)
        self.assertEqual(summary["gross_winners_dollars"], 160.0)
        self.assertEqual(summary["gross_losers_dollars"], 60.0)
        self.assertEqual(summary["win_rate_by_closed_layer"], 0.5)
        self.assertEqual(summary["profit_factor_by_closed_layer"], 2.666667)
        self.assertEqual(summary["average_closed_layer_pnl"], 0.25)
        self.assertEqual(summary["average_winner_pnl"], 0.8)
        self.assertEqual(summary["average_loser_pnl"], -0.3)
        self.assertEqual(summary["average_rebuilds_per_day"], 0.5)
        self.assertEqual(summary["cost_drag_from_rebuilding"], 1.212)
        self.assertEqual(summary["corridor_occupancy_rate"], 0.4)
        self.assertIn("return_on_capital", summary["metric_definitions"])
        self.assertIn("max_modeled_execution_capital_at_risk", summary["metric_definitions"])

    def test_day_level_profit_factor_and_best_worst_day_fields(self) -> None:
        cfg = CorridorConfig(
            symbol="SPY",
            timeframe="5 mins",
            starting_capital=100000.0,
            contracts_per_layer=1,
            option_multiplier=100,
        )
        ts1 = pd.Timestamp("2025-01-02 15:00:00", tz="UTC")
        ts2 = ts1 + pd.Timedelta(minutes=5)
        ts3 = ts1 + pd.Timedelta(days=1)

        actions: list[ActionRecord] = []
        equity_curve = [
            EquityPoint(
                timestamp=ts1,
                symbol="SPY",
                price=100.0,
                regime=Regime.RANGE,
                state=CorridorState.ACTIVE_CENTERED,
                bar_pnl=1.2,
                realized_pnl=1.2,
                unrealized_pnl=0.0,
                gross_realized_pnl=1.3,
                gross_unrealized_pnl=0.0,
                gross_total_equity=1.3,
                total_equity=1.2,
                modeled_capital_at_risk=1.156,
                corridor_occupancy=True,
                active_layers=1,
            ),
            EquityPoint(
                timestamp=ts2,
                symbol="SPY",
                price=100.2,
                regime=Regime.RANGE,
                state=CorridorState.IDLE,
                bar_pnl=-0.2,
                realized_pnl=1.0,
                unrealized_pnl=0.0,
                gross_realized_pnl=1.1,
                gross_unrealized_pnl=0.0,
                gross_total_equity=1.1,
                total_equity=1.0,
                modeled_capital_at_risk=0.0,
                corridor_occupancy=True,
                active_layers=0,
            ),
            EquityPoint(
                timestamp=ts3,
                symbol="SPY",
                price=99.0,
                regime=Regime.TREND_DOWN,
                state=CorridorState.ABORT,
                bar_pnl=-0.4,
                realized_pnl=0.6,
                unrealized_pnl=0.0,
                gross_realized_pnl=0.7,
                gross_unrealized_pnl=0.0,
                gross_total_equity=0.7,
                total_equity=0.6,
                modeled_capital_at_risk=0.0,
                corridor_occupancy=False,
                active_layers=0,
            ),
        ]

        summary = compute_metrics(cfg, actions, equity_curve)

        self.assertEqual(summary["best_day_pnl"], 1.0)
        self.assertEqual(summary["best_day_pnl_dollars"], 100.0)
        self.assertEqual(summary["worst_day_pnl"], -0.4)
        self.assertEqual(summary["worst_day_pnl_dollars"], -40.0)
        self.assertEqual(summary["profit_factor_by_day"], 2.5)

    def test_conservative_execution_risk_counts_same_timestamp_overlap(self) -> None:
        cfg = CorridorConfig(
            symbol="SPY",
            timeframe="5 mins",
            starting_capital=100000.0,
            contracts_per_layer=1,
            option_multiplier=100,
        )
        ts1 = pd.Timestamp("2025-01-02 15:00:00", tz="UTC")
        ts2 = ts1 + pd.Timedelta(minutes=5)

        actions = [
            ActionRecord(
                timestamp=ts1,
                symbol="SPY",
                action=ActionType.ENTER_PRIMARY,
                state=CorridorState.ACTIVE_CENTERED,
                price=100.0,
                center_price=100.0,
                layer_id=1,
                detail="Opened the primary butterfly corridor layer.",
                metadata={"entry_cost": 1.156, "friction_cost": 0.056, "entry_friction_cost": 0.056},
            ),
            ActionRecord(
                timestamp=ts2,
                symbol="SPY",
                action=ActionType.REBUILT,
                state=CorridorState.ACTIVE_CENTERED,
                price=101.0,
                center_price=101.0,
                layer_id=2,
                detail="Established a fresh primary butterfly corridor layer.",
                metadata={"entry_cost": 1.156, "friction_cost": 0.056, "entry_friction_cost": 0.056},
            ),
            ActionRecord(
                timestamp=ts2,
                symbol="SPY",
                action=ActionType.REBUILT,
                state=CorridorState.REBUILD,
                price=101.0,
                center_price=100.0,
                layer_id=1,
                detail="Removed prior layers for rebuild.",
                metadata={"entry_cost": 1.156, "realized_pnl": -0.3},
            ),
        ]

        equity_curve = [
            EquityPoint(
                timestamp=ts1,
                symbol="SPY",
                price=100.0,
                regime=Regime.RANGE,
                state=CorridorState.ACTIVE_CENTERED,
                bar_pnl=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                gross_realized_pnl=0.0,
                gross_unrealized_pnl=0.0,
                gross_total_equity=0.0,
                total_equity=0.0,
                modeled_capital_at_risk=1.156,
                corridor_occupancy=True,
                active_layers=1,
            ),
            EquityPoint(
                timestamp=ts2,
                symbol="SPY",
                price=101.0,
                regime=Regime.RANGE,
                state=CorridorState.ACTIVE_CENTERED,
                bar_pnl=0.0,
                realized_pnl=-0.3,
                unrealized_pnl=0.0,
                gross_realized_pnl=-0.2,
                gross_unrealized_pnl=0.0,
                gross_total_equity=-0.2,
                total_equity=-0.3,
                modeled_capital_at_risk=1.156,
                corridor_occupancy=True,
                active_layers=1,
            ),
        ]

        summary = compute_metrics(cfg, actions, equity_curve)

        self.assertEqual(summary["max_modeled_state_capital_at_risk"], 1.156)
        self.assertEqual(summary["max_modeled_execution_capital_at_risk"], 2.312)
        self.assertEqual(summary["max_gross_deployment_dollars"], 231.2)
        self.assertEqual(summary["max_modeled_close_friction_reserve"], 0.112)
        self.assertEqual(summary["max_modeled_capital_at_risk"], 2.424)
        self.assertEqual(
            summary["max_modeled_capital_at_risk_assumption"],
            "conservative_open_before_close_within_same_timestamp_plus_close_friction_reserve",
        )


if __name__ == "__main__":
    unittest.main()
