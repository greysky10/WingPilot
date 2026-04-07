from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from corridor.config import CorridorConfig
from corridor.execution.paper import (
    ManagedPosition,
    PaperCorridorRunner,
    PaperRunnerConfig,
    build_paper_test_summary,
    managed_position_to_payload,
)
from corridor.models import ActionRecord, ActionType, CorridorState, LayerKind, Regime
from corridor.options.butterfly_selector import ButterflyCandidate


class PaperCorridorRunnerTests(unittest.TestCase):
    def make_runner(self, *, paper_execution: bool = True, start_flat: bool = True) -> PaperCorridorRunner:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        cfg = CorridorConfig(symbol="SPY")
        runner_cfg = PaperRunnerConfig(
            symbol="SPY",
            paper_execution=paper_execution,
            start_flat=start_flat,
            output_dir=Path(tmpdir.name),
        )
        return PaperCorridorRunner(cfg, runner_cfg)

    @staticmethod
    def make_candidate(body: float = 635.0) -> ButterflyCandidate:
        return ButterflyCandidate(
            symbol="SPY",
            expiry="20260401",
            lower_strike=body - 5.0,
            body_strike=body,
            upper_strike=body + 5.0,
            lower_width=5.0,
            upper_width=5.0,
            net_debit=0.75,
            total_spread=0.10,
            max_risk=75.0,
            max_reward=425.0,
            right="CALL",
        )

    def test_close_rejection_keeps_position_and_halts_execution(self) -> None:
        runner = self.make_runner()
        candidate = self.make_candidate()
        runner.positions[2] = ManagedPosition(
            layer_id=2,
            candidate=candidate,
            quantity=1,
            opened_at=pd.Timestamp("2026-03-30 17:15:00", tz="UTC"),
            open_limit=0.81,
            open_status="Filled",
            source_action=ActionType.ADD_SUPPLEMENTAL.value,
        )
        trade = SimpleNamespace(
            orderStatus=SimpleNamespace(status="Cancelled"),
            order=SimpleNamespace(orderId=88),
            log=[SimpleNamespace(message="Error 201, reqId 88: Order rejected - reason:Riskless combination orders are not allowed.")],
            advancedError="",
        )
        order_logs: list[dict[str, object]] = []
        runner._place_combo_order = lambda _candidate, _side, _limit: trade  # type: ignore[method-assign]
        runner._log_order = lambda record: order_logs.append(record)  # type: ignore[method-assign]

        action = ActionRecord(
            timestamp=pd.Timestamp("2026-03-30 17:40:00", tz="UTC"),
            symbol="SPY",
            action=ActionType.REBUILT,
            state=CorridorState.REBUILD,
            price=633.51,
            center_price=635.0,
            layer_id=2,
            detail="Removed prior layers for rebuild.",
        )
        runner._close_position(action)

        self.assertIn(2, runner.positions)
        self.assertIsNotNone(runner.execution_halted_reason)
        self.assertIn("Riskless combination orders are not allowed", runner.execution_halted_reason or "")
        self.assertEqual(runner.positions[2].close_status, "Cancelled")
        self.assertTrue(order_logs)
        self.assertEqual(order_logs[-1]["status"], "Cancelled")

    def test_startup_guard_rejects_existing_option_positions_when_starting_flat(self) -> None:
        runner = self.make_runner(start_flat=True)
        runner._account_option_position_counts = lambda: {("20260401", 630.0, "C"): 1}  # type: ignore[method-assign]

        with self.assertRaises(RuntimeError):
            runner._guard_startup_account_state()

    def test_position_is_flat_in_account_after_other_layers_only_remain(self) -> None:
        runner = self.make_runner()
        candidate_a = self.make_candidate(635.0)
        candidate_b = self.make_candidate(637.0)
        runner.positions[2] = ManagedPosition(
            layer_id=2,
            candidate=candidate_a,
            quantity=1,
            opened_at=pd.Timestamp("2026-03-30 17:15:00", tz="UTC"),
            open_limit=0.81,
            open_status="Filled",
            source_action=ActionType.ADD_SUPPLEMENTAL.value,
            close_order_id=88,
        )
        runner.positions[3] = ManagedPosition(
            layer_id=3,
            candidate=candidate_b,
            quantity=1,
            opened_at=pd.Timestamp("2026-03-30 17:40:00", tz="UTC"),
            open_limit=0.77,
            open_status="Filled",
            source_action=ActionType.REBUILT.value,
        )

        actual_counts = runner._candidate_leg_counts(candidate_b, 1)
        self.assertTrue(runner._position_is_flat_in_account(runner.positions[2], actual_counts))

    def test_poll_once_keeps_seeded_state_when_history_refresh_fails(self) -> None:
        runner = self.make_runner()
        base_ts = pd.Timestamp("2026-03-30 13:30:00", tz="UTC")
        runner.history = pd.DataFrame(
            [
                {
                    "timestamp": base_ts + pd.Timedelta(minutes=5 * index),
                    "symbol": "SPY",
                    "open": 630.0,
                    "high": 631.0,
                    "low": 629.0,
                    "close": 630.5,
                    "volume": 1000.0,
                }
                for index in range(runner.required_warmup_bars)
            ]
        )
        runner.fetch_recent_history = lambda: (_ for _ in ()).throw(RuntimeError("hmdb outage"))  # type: ignore[method-assign]
        runner._refresh_positions_from_account = lambda: None  # type: ignore[method-assign]
        runner._write_state_snapshot = lambda: None  # type: ignore[method-assign]

        runner.poll_once()

        self.assertFalse(runner.warmup_mode)
        self.assertEqual(runner.history_refresh_error, "hmdb outage")
        self.assertEqual(len(runner.history), runner.required_warmup_bars)

    def test_restore_recovery_state_loads_adopted_positions(self) -> None:
        runner = self.make_runner(start_flat=False)
        adopted = ManagedPosition(
            layer_id=7,
            candidate=self.make_candidate(635.0),
            quantity=1,
            opened_at=pd.Timestamp("2026-03-30 17:00:00", tz="UTC"),
            open_limit=0.0,
            open_status="AdoptedFromIB",
            source_action="ADOPTED",
            layer_kind=LayerKind.PRIMARY.value,
        )
        runner.logger.write_recovery(
            {
                "symbol": "SPY",
                "saved_at": pd.Timestamp("2026-03-30 17:10:00", tz="UTC").isoformat(),
                "state": "ACTIVE_CENTERED",
                "current_center": 635.0,
                "next_layer_id": 8,
                "positions": [managed_position_to_payload(adopted)],
            }
        )

        runner._restore_recovery_state()

        self.assertIn(7, runner.positions)
        self.assertEqual(runner.machine.context.state, CorridorState.ACTIVE_CENTERED)
        self.assertEqual(runner.machine.context.current_center, 635.0)
        self.assertEqual(runner.machine.context.next_layer_id, 8)
        self.assertEqual(len(runner.machine.context.active_layers), 1)
        self.assertEqual(runner.machine.context.active_layers[0].body_strike, 635.0)

    def test_state_snapshot_reports_history_seeded_mode(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner.history = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2026-03-31 14:30:00", tz="UTC"),
                    "symbol": "SPY",
                    "open": 630.0,
                    "high": 631.0,
                    "low": 629.0,
                    "close": 630.5,
                    "volume": 1000.0,
                }
            ]
        )
        runner.history_seed_status = "History seed successful. Loaded 1 completed bars from IB historical data."

        payload = runner._build_state_snapshot()

        self.assertEqual(payload["startup_mode"], "history_seeded")
        self.assertTrue(payload["history_seeded"])
        self.assertIn("History seed successful", payload["history_seed_status"])

    def test_state_snapshot_reports_warmup_only_mode(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner._ensure_underlying_ticker = lambda: None  # type: ignore[method-assign]
        runner._activate_warmup_mode("hmdb unavailable")

        payload = runner._build_state_snapshot()

        self.assertEqual(payload["startup_mode"], "warmup_only")
        self.assertFalse(payload["history_seeded"])
        self.assertIn("warmup-only", payload["history_seed_status"])
        self.assertEqual(payload["warmup_reason"], "hmdb unavailable")

    def test_write_state_snapshot_emits_daily_report(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner.history = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2026-03-31 14:30:00", tz="UTC"),
                    "symbol": "SPY",
                    "open": 630.0,
                    "high": 631.0,
                    "low": 629.0,
                    "close": 630.5,
                    "volume": 1000.0,
                }
            ]
        )
        runner.history_seed_status = "History seed successful."

        runner._write_state_snapshot()

        self.assertTrue(runner.logger.paths["daily_report_json"].exists())
        self.assertTrue(runner.logger.paths["daily_report_csv"].exists())
        self.assertTrue(runner.logger.paths["test_summary_json"].exists())
        self.assertTrue(runner.logger.paths["test_summary_csv"].exists())
        self.assertTrue(runner.logger.paths["test_summary_txt"].exists())

    def test_daily_report_aggregates_fill_quality_metrics(self) -> None:
        runner = self.make_runner(paper_execution=False)
        now = pd.Timestamp.utcnow().floor("min")
        runner.history = pd.DataFrame(
            [
                {
                    "timestamp": now,
                    "symbol": "SPY",
                    "open": 630.0,
                    "high": 631.0,
                    "low": 629.0,
                    "close": 630.5,
                    "volume": 1000.0,
                }
            ]
        )
        runner.history_seed_status = "History seed successful."
        runner.logger.write_order(
            {
                "timestamp": now.isoformat(),
                "layer_id": 1,
                "symbol": "SPY",
                "side": "OPEN",
                "mode": "paper",
                "status": "Filled",
                "order_id": 10,
                "quantity": 1,
                "expiry": "20260401",
                "right": "CALL",
                "lower_strike": 630.0,
                "body_strike": 635.0,
                "upper_strike": 640.0,
                "limit_price": 0.82,
                "fill_price": 0.80,
                "quote_reference": 0.75,
                "limit_vs_quote": 0.07,
                "fill_edge_vs_quote": -0.05,
                "fill_edge_vs_limit": 0.02,
                "net_debit": 0.75,
                "total_spread": 0.10,
                "spread_ratio": 0.1333,
                "reason": "Opened the primary butterfly corridor layer.",
            }
        )
        runner.logger.write_order(
            {
                "timestamp": now.isoformat(),
                "layer_id": 1,
                "symbol": "SPY",
                "side": "CLOSE",
                "mode": "paper",
                "status": "Filled",
                "order_id": 11,
                "quantity": 1,
                "expiry": "20260401",
                "right": "CALL",
                "lower_strike": 630.0,
                "body_strike": 635.0,
                "upper_strike": 640.0,
                "limit_price": 0.70,
                "fill_price": 0.72,
                "quote_reference": 0.75,
                "limit_vs_quote": 0.05,
                "fill_edge_vs_quote": -0.03,
                "fill_edge_vs_limit": 0.02,
                "net_debit": 0.75,
                "total_spread": 0.10,
                "spread_ratio": 0.1333,
                "reason": "Primary take-profit reached.",
            }
        )

        runner._write_state_snapshot()

        report = json.loads(runner.logger.paths["daily_report_json"].read_text(encoding="utf-8"))
        self.assertEqual(report["filled_open_orders_today"], 1)
        self.assertEqual(report["filled_close_orders_today"], 1)
        self.assertAlmostEqual(report["avg_open_fill_edge_vs_quote"], -0.05, places=6)
        self.assertAlmostEqual(report["avg_open_fill_edge_vs_limit"], 0.02, places=6)
        self.assertAlmostEqual(report["avg_close_fill_edge_vs_quote"], -0.03, places=6)
        self.assertAlmostEqual(report["avg_close_fill_edge_vs_limit"], 0.02, places=6)
        self.assertAlmostEqual(report["avg_filled_spread_ratio"], 0.1333, places=6)

    def test_candidate_execution_issue_rejects_wide_spread_ratio(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner.runner_cfg.max_spread_pct_of_debit = 0.40
        bad_candidate = ButterflyCandidate(
            symbol="SPY",
            expiry="20260401",
            lower_strike=630.0,
            body_strike=635.0,
            upper_strike=640.0,
            lower_width=5.0,
            upper_width=5.0,
            net_debit=0.50,
            total_spread=0.30,
            max_risk=50.0,
            max_reward=450.0,
            right="CALL",
        )

        issue = runner._candidate_execution_issue(bad_candidate)

        self.assertIsNotNone(issue)
        self.assertIn("spread/debit ratio", issue or "")

    def test_open_position_does_not_track_unfilled_combo(self) -> None:
        runner = self.make_runner(paper_execution=True)
        candidate = self.make_candidate()
        runner._select_candidate = lambda _target_body: candidate  # type: ignore[method-assign]
        runner._log_order = lambda _record: None  # type: ignore[method-assign]
        trade = SimpleNamespace(
            orderStatus=SimpleNamespace(status="Submitted"),
            order=SimpleNamespace(orderId=42),
            log=[SimpleNamespace(message="still working")],
            advancedError="",
        )
        runner._place_combo_order = lambda _candidate, _side, _limit: trade  # type: ignore[method-assign]

        action = ActionRecord(
            timestamp=pd.Timestamp("2026-03-30 17:15:00", tz="UTC"),
            symbol="SPY",
            action=ActionType.ENTER_PRIMARY,
            state=CorridorState.ACTIVE_CENTERED,
            price=635.0,
            center_price=635.0,
            layer_id=5,
            detail="Opened the primary butterfly corridor layer.",
            metadata={"body_strike": 635.0, "kind": LayerKind.PRIMARY.value},
        )
        center = SimpleNamespace(center_price=635.0)
        regime = SimpleNamespace(regime=Regime.RANGE)

        runner._open_position(action, center, regime)

        self.assertNotIn(5, runner.positions)

    def test_protective_exit_signal_triggers_take_profit(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner.cfg.primary_take_profit_pct = 0.20
        runner.positions[1] = ManagedPosition(
            layer_id=1,
            candidate=self.make_candidate(635.0),
            quantity=1,
            opened_at=pd.Timestamp("2026-03-30 17:15:00", tz="UTC"),
            open_limit=0.80,
            open_status="Filled",
            source_action=ActionType.ENTER_PRIMARY.value,
            open_fill_price=0.80,
            layer_kind=LayerKind.PRIMARY.value,
        )
        runner._refresh_candidate_quote = lambda _candidate: ButterflyCandidate(  # type: ignore[method-assign]
            symbol="SPY",
            expiry="20260401",
            lower_strike=630.0,
            body_strike=635.0,
            upper_strike=640.0,
            lower_width=5.0,
            upper_width=5.0,
            net_debit=1.10,
            total_spread=0.10,
            max_risk=110.0,
            max_reward=390.0,
            right="CALL",
        )

        signal = runner._protective_exit_signal(pd.Timestamp("2026-03-30 18:00:00", tz="UTC"), 636.0)

        self.assertIsNotNone(signal)
        self.assertEqual(signal[0], ActionType.TAKE_PROFIT)

    def test_protective_exit_signal_triggers_stop_loss(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner.cfg.primary_stop_loss_pct = 0.25
        runner.positions[1] = ManagedPosition(
            layer_id=1,
            candidate=self.make_candidate(635.0),
            quantity=1,
            opened_at=pd.Timestamp("2026-03-30 17:15:00", tz="UTC"),
            open_limit=0.80,
            open_status="Filled",
            source_action=ActionType.ENTER_PRIMARY.value,
            open_fill_price=0.80,
            layer_kind=LayerKind.PRIMARY.value,
        )
        runner._refresh_candidate_quote = lambda _candidate: ButterflyCandidate(  # type: ignore[method-assign]
            symbol="SPY",
            expiry="20260401",
            lower_strike=630.0,
            body_strike=635.0,
            upper_strike=640.0,
            lower_width=5.0,
            upper_width=5.0,
            net_debit=0.50,
            total_spread=0.10,
            max_risk=50.0,
            max_reward=450.0,
            right="CALL",
        )

        signal = runner._protective_exit_signal(pd.Timestamp("2026-03-30 18:00:00", tz="UTC"), 633.0)

        self.assertIsNotNone(signal)
        self.assertEqual(signal[0], ActionType.STOP_LOSS)

    def test_adaptive_mode_prefers_symmetric_when_execution_safe(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner.cfg.wing_mode = "adaptive"
        runner.cfg.broken_wing_extra_width = 20.0
        symmetric = self.make_candidate(635.0)
        broken = ButterflyCandidate(
            symbol="SPY",
            expiry="20260401",
            lower_strike=630.0,
            body_strike=635.0,
            upper_strike=660.0,
            lower_width=5.0,
            upper_width=25.0,
            net_debit=0.60,
            total_spread=0.08,
            max_risk=260.0,
            max_reward=440.0,
            right="CALL",
            wing_mode="broken_upper",
            spread_ratio=0.1333,
        )
        runner._load_candidates = lambda _target_body: [broken, symmetric]  # type: ignore[method-assign]

        chosen = runner._select_candidate(635.0)

        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.wing_mode, "symmetric")

    def test_adaptive_mode_falls_back_to_broken_wing_when_symmetric_is_poor(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner.cfg.wing_mode = "adaptive"
        runner.cfg.broken_wing_extra_width = 20.0
        runner.runner_cfg.max_spread_pct_of_debit = 0.40
        symmetric = ButterflyCandidate(
            symbol="SPY",
            expiry="20260401",
            lower_strike=630.0,
            body_strike=635.0,
            upper_strike=640.0,
            lower_width=5.0,
            upper_width=5.0,
            net_debit=0.50,
            total_spread=0.30,
            max_risk=50.0,
            max_reward=450.0,
            right="CALL",
            wing_mode="symmetric",
            spread_ratio=0.60,
        )
        broken = ButterflyCandidate(
            symbol="SPY",
            expiry="20260401",
            lower_strike=630.0,
            body_strike=635.0,
            upper_strike=660.0,
            lower_width=5.0,
            upper_width=25.0,
            net_debit=0.70,
            total_spread=0.10,
            max_risk=270.0,
            max_reward=430.0,
            right="CALL",
            wing_mode="broken_upper",
            spread_ratio=0.1429,
        )
        runner._load_candidates = lambda _target_body: [symmetric, broken]  # type: ignore[method-assign]

        chosen = runner._select_candidate(635.0)

        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.wing_mode, "broken_upper")
        self.assertEqual(runner.wing_stats["broken_upper"], 1)
        self.assertEqual(runner.wing_stats["guard_fails"], 1)

    def test_runner_restores_persistent_wing_stats_from_state_snapshot(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner.logger.write_state(
            {
                "wing_stats": {
                    "symmetric": 3,
                    "broken_upper": 1,
                    "broken_lower": 2,
                    "guard_fails": 4,
                }
            }
        )

        restored = PaperCorridorRunner(runner.cfg, runner.runner_cfg)

        self.assertEqual(
            restored.wing_stats,
            {"symmetric": 3, "broken_upper": 1, "broken_lower": 2, "guard_fails": 4},
        )

    def test_daily_report_includes_adaptive_fallback_fields(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner.history = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2026-03-31 14:30:00", tz="UTC"),
                    "symbol": "SPY",
                    "open": 630.0,
                    "high": 631.0,
                    "low": 629.0,
                    "close": 630.5,
                    "volume": 1000.0,
                }
            ]
        )
        runner.wing_stats.update({"symmetric": 4, "broken_upper": 1, "broken_lower": 1, "guard_fails": 2})

        runner._write_state_snapshot()

        report = json.loads(runner.logger.paths["daily_report_json"].read_text(encoding="utf-8"))
        self.assertAlmostEqual(report["adaptive_fallback_rate"], 0.3333, places=4)
        self.assertEqual(report["fallback_type_distribution"]["counts"]["broken_upper"], 1)
        self.assertEqual(report["fallback_type_distribution"]["counts"]["broken_lower"], 1)

    def test_paper_test_summary_marks_execution_halt_as_fail(self) -> None:
        runner = self.make_runner(paper_execution=False)
        state_payload = {
            "symbol": "SPY",
            "execution_halted_reason": "close rejected",
        }
        daily_report = {
            "report_timestamp": pd.Timestamp("2026-03-31 18:00:00", tz="UTC").isoformat(),
            "report_date": "2026-03-31",
            "symbol": "SPY",
            "execution_mode": "paper",
            "startup_mode": "history_seeded",
            "history_seeded": True,
            "model_ready": True,
            "warmup_mode": False,
            "execution_halted_reason": "close rejected",
            "latest_state": "ACTIVE_CENTERED",
            "latest_regime": "RANGE",
            "filled_orders_today": 2,
            "open_positions_count": 1,
            "blocked_or_skipped_orders_today": 0,
            "adaptive_fallback_rate": 0.10,
            "fallback_type_distribution": {"counts": {"broken_upper": 1, "broken_lower": 0}},
        }

        summary = build_paper_test_summary(state_payload, daily_report)

        self.assertEqual(summary["overall_status"], "FAIL")
        self.assertEqual(summary["checks"]["runner"]["status"], "FAIL")

    def test_paper_test_summary_surfaces_top_candidate_rejection(self) -> None:
        summary = build_paper_test_summary(
            {"symbol": "SPY"},
            {
                "report_timestamp": pd.Timestamp("2026-03-31 18:00:00", tz="UTC").isoformat(),
                "report_date": "2026-03-31",
                "symbol": "SPY",
                "execution_mode": "paper",
                "startup_mode": "history_seeded",
                "history_seeded": True,
                "model_ready": True,
                "warmup_mode": False,
                "execution_halted_reason": None,
                "latest_state": "ACTIVE_CENTERED",
                "latest_regime": "RANGE",
                "filled_orders_today": 0,
                "open_positions_count": 0,
                "blocked_or_skipped_orders_today": 2,
                "adaptive_fallback_rate": 0.0,
                "fallback_type_distribution": {"counts": {"broken_upper": 0, "broken_lower": 0}},
                "candidate_diagnostics": {"rejection_counts": {"missing_legs": 12, "spread_too_wide": 3}},
            },
        )

        self.assertEqual(summary["checks"]["diagnostics"]["status"], "WARN")
        self.assertIn("missing_legs=12", summary["checks"]["diagnostics"]["message"])

    def test_combo_chase_aborts_when_center_drifts_after_timeout(self) -> None:
        runner = self.make_runner(paper_execution=True)
        candidate = self.make_candidate(635.0)
        trade = SimpleNamespace(
            orderStatus=SimpleNamespace(status="Submitted", filled=0.0, avgFillPrice=0.0),
            order=SimpleNamespace(orderId=77),
            fills=[],
            log=[SimpleNamespace(message="still working")],
            advancedError="",
        )
        runner.ib.cancelOrder = lambda _order: None  # type: ignore[method-assign]
        runner.ib.sleep = lambda _seconds: None  # type: ignore[method-assign]
        runner._place_combo_order_once = lambda _candidate, _side, _limit: trade  # type: ignore[method-assign]
        runner._chase_should_abort_from_drift = lambda _candidate: True  # type: ignore[method-assign]

        chased = runner._place_combo_order_with_chase(candidate, "BUY", 0.82)

        self.assertIs(chased, trade)
        self.assertEqual(chased.fillAudit["abort_reason"], "fill_timeout_abort_center_drift")
        self.assertEqual(chased.fillAudit["steps"][0]["limit_price"], 0.82)

    def test_poll_once_logs_timestamp_when_no_new_completed_bars(self) -> None:
        runner = self.make_runner(paper_execution=False)
        ts = pd.Timestamp("2026-03-31 14:30:00", tz="UTC")
        frame = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "symbol": "SPY",
                    "open": 630.0,
                    "high": 631.0,
                    "low": 629.0,
                    "close": 630.5,
                    "volume": 1000.0,
                }
            ]
        )
        runner.history = frame.copy()
        runner.last_processed_ts = ts
        runner.fetch_recent_history = lambda: frame.copy()  # type: ignore[method-assign]
        runner._refresh_positions_from_account = lambda: None  # type: ignore[method-assign]
        runner._write_state_snapshot = lambda: None  # type: ignore[method-assign]

        with patch("builtins.print") as mocked_print:
            runner.poll_once()

        mocked_print.assert_any_call(unittest.mock.ANY)
        first_arg = mocked_print.call_args_list[0].args[0]
        self.assertIn("No new completed bars.", first_arg)
        self.assertIn("ts=", first_arg)


if __name__ == "__main__":
    unittest.main()
