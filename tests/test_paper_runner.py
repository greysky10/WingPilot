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
from corridor.options.chain_loader import OptionQuote


class PaperCorridorRunnerTests(unittest.TestCase):
    def make_runner(
        self,
        *,
        paper_execution: bool = True,
        start_flat: bool = True,
        **runner_overrides,
    ) -> PaperCorridorRunner:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        cfg = CorridorConfig(symbol="SPY")
        runner_cfg = PaperRunnerConfig(
            symbol="SPY",
            paper_execution=paper_execution,
            start_flat=start_flat,
            output_dir=Path(tmpdir.name),
            **runner_overrides,
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
        runner._place_combo_order = lambda *_args, **_kwargs: trade  # type: ignore[method-assign]
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

    def test_restore_recovery_state_restores_last_primary_entry_session_date(self) -> None:
        runner = self.make_runner(start_flat=False)
        runner.logger.write_recovery(
            {
                "symbol": "SPY",
                "saved_at": pd.Timestamp("2026-03-30 17:10:00", tz="UTC").isoformat(),
                "state": "ACTIVE_CENTERED",
                "current_center": 635.0,
                "next_layer_id": 8,
                "last_primary_entry_session_date": "2026-03-30",
                "positions": [],
            }
        )

        runner._restore_recovery_state()

        self.assertEqual(runner.machine.context.last_primary_entry_session_date, "2026-03-30")

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

    @patch("corridor.execution.paper.send_discord_text_alert")
    def test_write_state_snapshot_can_send_discord_summary(self, discord_mock) -> None:
        runner = self.make_runner(paper_execution=False, discord_summary=True)
        runner.discord_webhook_url = "https://discord.test/webhook"
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

        discord_mock.assert_called_once()
        sent_message = discord_mock.call_args.args[1]
        self.assertIn("Discord Paper Summary", sent_message)
        self.assertIn("Paper Test Summary", sent_message)

    @patch("corridor.execution.paper.send_discord_text_alert")
    def test_identical_discord_summary_is_throttled(self, discord_mock) -> None:
        runner = self.make_runner(
            paper_execution=False,
            discord_summary=True,
            discord_summary_min_interval_minutes=30,
        )
        runner.discord_webhook_url = "https://discord.test/webhook"
        daily_report = {
            "report_date": "2026-04-15",
            "candidate_count": 1,
            "candidate_status": "Loaded 1 candidate butterflies for the current center.",
            "paper_smoke_mode": False,
        }
        summary_payload = {
            "overall_status": "PASS",
            "filled_orders_today": 0,
            "open_positions_count": 0,
            "latest_state": "IDLE",
            "latest_regime": "RANGE",
        }
        summary_text = "Paper Test Summary | PASS | 2026-04-15 | SPY\n"

        runner._maybe_send_discord_test_summary(daily_report, summary_payload, summary_text)
        runner._maybe_send_discord_test_summary(daily_report, summary_payload, summary_text)

        discord_mock.assert_called_once()

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
        runner.cfg.max_acceptable_option_spread = 1.0
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
        runner._select_candidate = lambda _target_body, **_kwargs: candidate  # type: ignore[method-assign]
        runner._log_order = lambda _record: None  # type: ignore[method-assign]
        trade = SimpleNamespace(
            orderStatus=SimpleNamespace(status="Submitted"),
            order=SimpleNamespace(orderId=42),
            log=[SimpleNamespace(message="still working")],
            advancedError="",
        )
        runner._place_combo_order = lambda *_args, **_kwargs: trade  # type: ignore[method-assign]

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

    def test_smoke_mode_can_open_without_range_regime(self) -> None:
        runner = self.make_runner(paper_execution=False, paper_smoke_mode=True)
        runner.history = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2026-04-15 14:00:00", tz="UTC") + pd.Timedelta(minutes=5 * index),
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
        runner.machine.context.session_date = "2026-04-15"
        runner._select_candidate = lambda _target_body, **_kwargs: self.make_candidate()  # type: ignore[method-assign]
        row = pd.Series(
            {
                "timestamp": pd.Timestamp("2026-04-15 14:45:00", tz="UTC"),
                "symbol": "SPY",
                "open": 630.0,
                "high": 631.0,
                "low": 629.0,
                "close": 630.5,
                "volume": 1000.0,
            }
        )

        runner._process_smoke_mode_bar(
            row,
            regime=SimpleNamespace(regime=Regime.TREND_UP),
            center=SimpleNamespace(center_price=635.0),
            allow_orders=True,
            emit_logs=False,
        )

        self.assertEqual(len(runner.positions), 1)
        opened = next(iter(runner.positions.values()))
        self.assertEqual(opened.source_action, ActionType.SMOKE_ENTRY.value)

    def test_smoke_mode_forces_timed_close(self) -> None:
        runner = self.make_runner(
            paper_execution=True,
            paper_smoke_mode=True,
            smoke_force_close_minutes=45,
        )
        candidate = self.make_candidate()
        opened_at = pd.Timestamp("2026-04-15 13:45:00", tz="UTC")
        runner.positions[2] = ManagedPosition(
            layer_id=2,
            candidate=candidate,
            quantity=1,
            opened_at=opened_at,
            open_limit=0.81,
            open_status="Filled",
            source_action=ActionType.SMOKE_ENTRY.value,
        )
        trade = SimpleNamespace(
            orderStatus=SimpleNamespace(status="Filled", avgFillPrice=0.70),
            order=SimpleNamespace(orderId=88),
            fillAudit={"steps": [{"step": 1, "status": "Filled"}]},
            log=[],
            advancedError="",
        )
        runner._place_combo_order = lambda *_args, **_kwargs: trade  # type: ignore[method-assign]
        runner.machine.context.current_center = 635.0

        row = pd.Series(
            {
                "timestamp": pd.Timestamp("2026-04-15 14:35:00", tz="UTC"),
                "symbol": "SPY",
                "open": 630.0,
                "high": 631.0,
                "low": 629.0,
                "close": 630.5,
                "volume": 1000.0,
            }
        )

        runner._process_smoke_mode_bar(
            row,
            regime=SimpleNamespace(regime=Regime.RANGE),
            center=SimpleNamespace(center_price=635.0),
            allow_orders=True,
            emit_logs=False,
        )

        self.assertFalse(runner.positions)

    def test_open_position_sends_discord_alert_once_for_filled_paper_open(self) -> None:
        runner = self.make_runner(paper_execution=True)
        runner.discord_webhook_url = "https://discord.test/webhook"
        candidate = self.make_candidate()
        runner._select_candidate = lambda _target_body, **_kwargs: candidate  # type: ignore[method-assign]
        runner._log_order = lambda _record: None  # type: ignore[method-assign]
        trade = SimpleNamespace(
            orderStatus=SimpleNamespace(status="Filled", avgFillPrice=0.82),
            order=SimpleNamespace(orderId=42),
            fillAudit={"steps": [{"step": 1, "status": "Filled"}]},
            log=[],
            advancedError="",
        )
        runner._place_combo_order = lambda *_args, **_kwargs: trade  # type: ignore[method-assign]
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

        with patch("corridor.execution.paper.send_discord_json_alert", return_value=True) as send_mock:
            runner._open_position(action, center, regime)

        self.assertIn(5, runner.positions)
        send_mock.assert_called_once()
        webhook_url, payload = send_mock.call_args.args
        self.assertEqual(webhook_url, "https://discord.test/webhook")
        self.assertEqual(payload["symbol"], "SPY")
        self.assertEqual(payload["expiry"], "20260401")
        self.assertEqual(payload["lower_strike"], 630.0)
        self.assertEqual(payload["body_strike"], 635.0)
        self.assertEqual(payload["upper_strike"], 640.0)
        self.assertEqual(payload["wing_mode"], "symmetric")
        self.assertEqual(payload["net_debit"], 0.75)
        self.assertEqual(payload["max_risk"], 75.0)
        self.assertEqual(payload["max_reward"], 425.0)
        self.assertEqual(payload["timestamp"], "2026-03-30T17:15:00+00:00")
        self.assertEqual(payload["status"], "Filled")
        self.assertEqual(payload["layer_id"], 5)
        self.assertEqual(payload["quantity"], 1)
        self.assertEqual(payload["order_id"], 42)
        self.assertEqual(payload["limit_price"], 0.78)
        self.assertEqual(payload["fill_price"], 0.82)
        self.assertEqual(payload["mode"], "paper")
        self.assertEqual(payload["fill_audit"], {"steps": [{"step": 1, "status": "Filled"}]})

    def test_open_position_does_not_send_discord_alert_for_dry_run_open(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner.discord_webhook_url = "https://discord.test/webhook"
        candidate = self.make_candidate()
        runner._select_candidate = lambda _target_body, **_kwargs: candidate  # type: ignore[method-assign]
        runner._log_order = lambda _record: None  # type: ignore[method-assign]
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

        with patch("corridor.execution.paper.send_discord_json_alert", return_value=True) as send_mock:
            runner._open_position(action, center, regime)

        self.assertIn(5, runner.positions)
        send_mock.assert_not_called()

    def test_open_position_does_not_send_discord_alert_for_rejected_or_unfilled_opens(self) -> None:
        scenarios = [
            (
                "unfilled",
                SimpleNamespace(
                    orderStatus=SimpleNamespace(status="Submitted"),
                    order=SimpleNamespace(orderId=42),
                    log=[SimpleNamespace(message="still working")],
                    advancedError="",
                ),
            ),
            (
                "rejected",
                SimpleNamespace(
                    orderStatus=SimpleNamespace(status="Cancelled"),
                    order=SimpleNamespace(orderId=43),
                    log=[SimpleNamespace(message="Order rejected")],
                    advancedError="",
                ),
            ),
        ]

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

        for label, trade in scenarios:
            with self.subTest(label=label):
                runner = self.make_runner(paper_execution=True)
                runner.discord_webhook_url = "https://discord.test/webhook"
                candidate = self.make_candidate()
                runner._select_candidate = lambda _target_body, **_kwargs: candidate  # type: ignore[method-assign]
                runner._log_order = lambda _record: None  # type: ignore[method-assign]
                runner._place_combo_order = lambda *_args, trade=trade, **_kwargs: trade  # type: ignore[method-assign]

                with patch("corridor.execution.paper.send_discord_json_alert", return_value=True) as send_mock:
                    runner._open_position(action, center, regime)

                send_mock.assert_not_called()

    def test_discord_helper_failure_does_not_break_open_position(self) -> None:
        runner = self.make_runner(paper_execution=True)
        runner.discord_webhook_url = "https://discord.test/webhook"
        candidate = self.make_candidate()
        runner._select_candidate = lambda _target_body, **_kwargs: candidate  # type: ignore[method-assign]
        runner._log_order = lambda _record: None  # type: ignore[method-assign]
        trade = SimpleNamespace(
            orderStatus=SimpleNamespace(status="Filled", avgFillPrice=0.82),
            order=SimpleNamespace(orderId=42),
            fillAudit={"steps": [{"step": 1, "status": "Filled"}]},
            log=[],
            advancedError="",
        )
        runner._place_combo_order = lambda *_args, **_kwargs: trade  # type: ignore[method-assign]
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

        with patch("corridor.notifications.discord.urllib.request.urlopen", side_effect=OSError("network down")):
            runner._open_position(action, center, regime)

        self.assertIn(5, runner.positions)
        self.assertEqual(runner.positions[5].open_status, "Filled")

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
        runner._load_candidates = lambda _target_body, **_kwargs: [broken, symmetric]  # type: ignore[method-assign]

        chosen = runner._select_candidate(635.0)

        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.wing_mode, "symmetric")

    def test_adaptive_mode_falls_back_to_broken_wing_when_symmetric_is_poor(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner.cfg.wing_mode = "adaptive"
        runner.cfg.broken_wing_extra_width = 20.0
        runner.cfg.max_acceptable_option_spread = 1.0
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
        runner._load_candidates = lambda _target_body, **_kwargs: [symmetric, broken]  # type: ignore[method-assign]

        chosen = runner._select_candidate(635.0)

        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.wing_mode, "broken_upper")
        self.assertEqual(runner.wing_stats["broken_upper"], 1)
        self.assertEqual(runner.wing_stats["guard_fails"], 1)

    def test_select_candidate_prefers_target_dte_closest_match(self) -> None:
        runner = self.make_runner(paper_execution=False)
        near = ButterflyCandidate(
            symbol="SPY",
            expiry="20260401",
            lower_strike=630.0,
            body_strike=635.0,
            upper_strike=640.0,
            lower_width=5.0,
            upper_width=5.0,
            net_debit=0.70,
            total_spread=0.10,
            max_risk=70.0,
            max_reward=430.0,
            right="CALL",
            calendar_dte=21,
        )
        far = ButterflyCandidate(
            symbol="SPY",
            expiry="20260408",
            lower_strike=630.0,
            body_strike=635.0,
            upper_strike=640.0,
            lower_width=5.0,
            upper_width=5.0,
            net_debit=0.70,
            total_spread=0.10,
            max_risk=70.0,
            max_reward=430.0,
            right="CALL",
            calendar_dte=28,
        )
        runner._load_candidates = lambda _target_body, **_kwargs: [near, far]  # type: ignore[method-assign]

        chosen = runner._select_candidate(635.0, target_dte=28, reference_ts=pd.Timestamp("2026-03-11 14:30:00", tz="UTC"))

        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.expiry, "20260408")

    def test_candidate_execution_issue_uses_dte_tiered_spread_cap(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner.cfg.max_acceptable_option_spread = 0.20
        runner.cfg.near_spread_dte_max = 10
        runner.cfg.near_max_acceptable_option_spread = 0.10
        runner.cfg.far_spread_dte_min = 20
        runner.cfg.far_max_acceptable_option_spread = 0.30

        near = self.make_candidate()
        near.total_spread = 0.16
        near.calendar_dte = 7
        far = self.make_candidate()
        far.total_spread = 0.16
        far.calendar_dte = 28

        self.assertIn("max_option_spread", runner._candidate_execution_issue(near) or "")
        self.assertIsNone(runner._candidate_execution_issue(far))

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

    def test_build_discord_position_detail_lines_include_leg_prices_and_pnl(self) -> None:
        runner = self.make_runner(paper_execution=True)
        runner.positions[5] = ManagedPosition(
            layer_id=5,
            candidate=self.make_candidate(),
            quantity=1,
            opened_at=pd.Timestamp("2026-03-30 17:15:00", tz="UTC"),
            open_limit=0.82,
            open_status="Filled",
            source_action=ActionType.ENTER_PRIMARY.value,
            open_fill_price=0.82,
            entry_leg_prices={"lower": 1.15, "body": 0.40, "upper": 0.47},
            layer_kind=LayerKind.PRIMARY.value,
        )
        quote_map = {
            630.0: OptionQuote(
                symbol="SPY",
                expiry="20260401",
                strike=630.0,
                right="CALL",
                bid=1.20,
                ask=1.30,
                last=0.0,
                implied_vol=None,
            ),
            635.0: OptionQuote(
                symbol="SPY",
                expiry="20260401",
                strike=635.0,
                right="CALL",
                bid=0.30,
                ask=0.40,
                last=0.0,
                implied_vol=None,
            ),
            640.0: OptionQuote(
                symbol="SPY",
                expiry="20260401",
                strike=640.0,
                right="CALL",
                bid=0.55,
                ask=0.65,
                last=0.0,
                implied_vol=None,
            ),
        }
        runner._load_structure_quote_map = lambda _candidate: quote_map  # type: ignore[method-assign]
        runner._account_option_average_costs = lambda: {  # type: ignore[method-assign]
            ("20260401", 630.0, "C"): (115.0, 100.0),
            ("20260401", 635.0, "C"): (40.0, 100.0),
            ("20260401", 640.0, "C"): (47.0, 100.0),
        }

        lines = runner._build_discord_position_detail_lines()

        self.assertEqual(
            lines[0],
            "Position 5 | sleeve=main | SPY 20260401 CALL | qty=1 | strikes=630.0/635.0/640.0 | combo_entry=0.82 | combo_now=1.15 | combo_pnl=$+33.00",
        )
        self.assertEqual(lines[1], "+1x 630.0 | entry=1.15 | current=1.25 | pnl=$+10.00")
        self.assertEqual(lines[2], "-2x 635.0 | entry=0.40 | current=0.35 | pnl=$+10.00")
        self.assertEqual(lines[3], "+1x 640.0 | entry=0.47 | current=0.60 | pnl=$+13.00")

    def test_poll_once_sends_discord_message_when_no_new_completed_bars(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner.discord_webhook_url = "https://discord.test/webhook"
        runner._intraday_pnl_log_suffix = lambda: (  # type: ignore[method-assign]
            " | today_est_pnl=$+815.00"
            " | realized=$+0.00"
            " | unrealized=$+815.00"
            " | open_positions=1"
        )
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

        with (
            patch("builtins.print"),
            patch("corridor.execution.paper.send_discord_text_alert", return_value=True) as send_mock,
        ):
            runner.poll_once()

        send_mock.assert_called_once()
        webhook_url, message = send_mock.call_args.args
        self.assertEqual(webhook_url, "https://discord.test/webhook")
        self.assertIn("No new completed bars.", message)
        self.assertIn("ts=", message)
        self.assertIn("today_est_pnl=$+815.00", message)
        self.assertIn("open_positions=1", message)

    def test_poll_once_appends_position_details_to_discord_message(self) -> None:
        runner = self.make_runner(paper_execution=False)
        runner.discord_webhook_url = "https://discord.test/webhook"
        runner._intraday_pnl_log_suffix = lambda: " | today_est_pnl=$+10.00 | realized=$+0.00 | unrealized=$+10.00 | open_positions=1"  # type: ignore[method-assign]
        runner._build_discord_position_detail_lines = lambda: [  # type: ignore[method-assign]
            "Position 5 | SPY 20260401 CALL | qty=1 | strikes=630.0/635.0/640.0 | combo_entry=0.82 | combo_now=0.92 | combo_pnl=$+10.00",
            "+1x 630.0 | entry=1.15 | current=1.20 | pnl=$+5.00",
        ]
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

        with (
            patch("builtins.print"),
            patch("corridor.execution.paper.send_discord_text_alert", return_value=True) as send_mock,
        ):
            runner.poll_once()

        message = send_mock.call_args.args[1]
        self.assertIn("No new completed bars.", message)
        self.assertIn("Position 5 | SPY 20260401 CALL", message)
        self.assertIn("+1x 630.0 | entry=1.15 | current=1.20 | pnl=$+5.00", message)


if __name__ == "__main__":
    unittest.main()
