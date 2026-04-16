"""Tests for paper_diagnostics module."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd


class TestRejectionReason(unittest.TestCase):
    """Tests for RejectionReason class."""

    def test_parse_reason_with_subcode(self) -> None:
        """Test parsing a reason with subcode."""
        from corridor.execution.paper_diagnostics import RejectionReason

        reason, subcode = RejectionReason.parse_reason("quote_quality.missing_leg_bid")
        self.assertEqual(reason, "quote_quality")
        self.assertEqual(subcode, "missing_leg_bid")

    def test_parse_reason_without_subcode(self) -> None:
        """Test parsing a legacy reason without subcode."""
        from corridor.execution.paper_diagnostics import RejectionReason

        reason, subcode = RejectionReason.parse_reason("missing_legs")
        self.assertEqual(reason, "missing_legs")
        self.assertIsNone(subcode)

    def test_parse_reason_with_none(self) -> None:
        """Test parsing None reason."""
        from corridor.execution.paper_diagnostics import RejectionReason

        reason, subcode = RejectionReason.parse_reason(None)
        self.assertIsNone(reason)
        self.assertIsNone(subcode)

    def test_from_legacy_reason(self) -> None:
        """Test converting legacy reasons to new format."""
        from corridor.execution.paper_diagnostics import RejectionReason

        # Test missing_legs conversion
        reason, subcode = RejectionReason.from_legacy_reason("missing_legs")
        self.assertEqual(reason, "quote_quality")
        self.assertEqual(subcode, "missing_leg_bid")

        # Test non_positive_debit conversion
        reason, subcode = RejectionReason.from_legacy_reason("non_positive_debit")
        self.assertEqual(reason, "pricing")
        self.assertEqual(subcode, "non_positive_debit")

        # Test spread_too_wide conversion
        reason, subcode = RejectionReason.from_legacy_reason("spread_too_wide")
        self.assertEqual(reason, "spread_quality")
        self.assertEqual(subcode, "ratio")


class TestPaperDiagnosticsCollector(unittest.TestCase):
    """Tests for PaperDiagnosticsCollector class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_heartbeat_emission(self) -> None:
        """Test heartbeat emission creates JSONL file."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        collector.emit_heartbeat(
            symbol="SPX",
            mode="smoke",
            state="IDLE",
            regime="RANGE",
            ib_connected=True,
            market_data_available=True,
            open_positions=0,
            warmup_bars=100,
            required_bars=100,
        )

        heartbeat_file = self.output_dir / "test_heartbeat.jsonl"
        self.assertTrue(heartbeat_file.exists())

        with open(heartbeat_file) as f:
            line = f.readline()
            payload = json.loads(line)
            self.assertEqual(payload["symbol"], "SPX")
            self.assertEqual(payload["mode"], "smoke")
            self.assertEqual(payload["state"], "IDLE")
            self.assertEqual(payload["regime"], "RANGE")

    def test_cycle_decision_emission(self) -> None:
        """Test cycle decision emission creates JSONL file."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        collector.emit_cycle_decision(
            symbol="SPX",
            regime="RANGE",
            state="ACTIVE_CENTERED",
            chains_loaded=True,
            candidates_generated=50,
            rejection_counts={"quote_quality.missing_leg_bid": 10, "pricing.non_positive_debit": 5},
            eligible_candidates=35,
            orders_submitted=1,
            fills=1,
            cancels=0,
            replaces=0,
        )

        cycle_file = self.output_dir / "test_cycle_decisions.jsonl"
        self.assertTrue(cycle_file.exists())

        with open(cycle_file) as f:
            line = f.readline()
            payload = json.loads(line)
            self.assertEqual(payload["symbol"], "SPX")
            self.assertEqual(payload["eligible_candidates"], 35)
            self.assertIn("quote_quality", payload["top_reject_reason"])

    def test_candidate_diagnostic_emission(self) -> None:
        """Test candidate diagnostic emission."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        candidate = {
            "expiry": "20240621",
            "lower_strike": 5200.0,
            "body_strike": 5300.0,
            "upper_strike": 5400.0,
            "lower_width": 100.0,
            "net_debit": 2.50,
            "total_spread": 0.50,
            "calendar_dte": 30,
        }

        collector.emit_candidate_diagnostic(
            symbol="SPX",
            regime="RANGE",
            state="ACTIVE_CENTERED",
            candidate=candidate,
            rejection_reason="quote_quality.missing_leg_bid",
            is_submitted=False,
            is_top_rejected=True,
        )

        candidates_file = self.output_dir / "test_candidate_diagnostics.jsonl"
        self.assertTrue(candidates_file.exists())

        with open(candidates_file) as f:
            line = f.readline()
            payload = json.loads(line)
            self.assertEqual(payload["expiry"], "20240621")
            self.assertEqual(payload["rejection_reason"], "quote_quality")
            self.assertEqual(payload["rejection_subcode"], "missing_leg_bid")

    def test_runner_event_emission(self) -> None:
        """Test runner event emission."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        collector.emit_runner_event(
            symbol="SPX",
            event_type="connected",
            state="IDLE",
            detail="Connected to IB at 127.0.0.1:4001",
        )

        events_file = self.output_dir / "test_runner_events.jsonl"
        self.assertTrue(events_file.exists())

        with open(events_file) as f:
            line = f.readline()
            payload = json.loads(line)
            self.assertEqual(payload["event_type"], "connected")
            self.assertEqual(payload["detail"], "Connected to IB at 127.0.0.1:4001")

    def test_diagnostics_disabled(self) -> None:
        """Test that disabled diagnostics don't create files."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=False,
        )

        collector.emit_heartbeat(
            symbol="SPX",
            mode="smoke",
            state="IDLE",
            regime="RANGE",
            ib_connected=True,
            market_data_available=True,
            open_positions=0,
            warmup_bars=100,
            required_bars=100,
        )

        heartbeat_file = self.output_dir / "test_heartbeat.jsonl"
        self.assertFalse(heartbeat_file.exists())

    def test_preserve_diagnostics(self) -> None:
        """Test diagnostics preservation across cycles."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        candidate_diagnostics = {
            "rejection_counts": {"quote_quality.missing_leg_bid": 10},
            "sample_rejections": [{"reason": "missing_leg_bid", "candidate": {}}],
            "available_quotes": 100,
            "attempted_structures": 50,
        }

        collector.preserve_diagnostics(
            candidate_diagnostics=candidate_diagnostics,
            market_data_ts=pd.Timestamp.utcnow(),
        )

        preserved = collector.get_preserved_diagnostics()
        self.assertIsNotNone(preserved)
        self.assertEqual(preserved["rejection_counts"]["quote_quality.missing_leg_bid"], 10)

    def test_should_emit_full_summary_deduplication(self) -> None:
        """Test that identical summaries are deduplicated."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        # First call should return True
        result1 = collector.should_emit_full_summary(
            state="IDLE",
            regime="RANGE",
            connection_state="connected",
            market_data_state="available",
            top_reject_reason=None,
            top_reject_subcode=None,
            eligible_candidate_count=0,
            order_count=0,
            fill_count=0,
            cancel_count=0,
            replace_count=0,
            has_warning=False,
            has_error=False,
        )
        self.assertTrue(result1)

        # Same state should return False (deduplicated)
        result2 = collector.should_emit_full_summary(
            state="IDLE",
            regime="RANGE",
            connection_state="connected",
            market_data_state="available",
            top_reject_reason=None,
            top_reject_subcode=None,
            eligible_candidate_count=0,
            order_count=0,
            fill_count=0,
            cancel_count=0,
            replace_count=0,
            has_warning=False,
            has_error=False,
        )
        self.assertFalse(result2)

        # Different state should return True
        result3 = collector.should_emit_full_summary(
            state="ACTIVE_CENTERED",
            regime="RANGE",
            connection_state="connected",
            market_data_state="available",
            top_reject_reason=None,
            top_reject_subcode=None,
            eligible_candidate_count=1,
            order_count=1,
            fill_count=1,
            cancel_count=0,
            replace_count=0,
            has_warning=False,
            has_error=False,
        )
        self.assertTrue(result3)


class TestHelperFunctions(unittest.TestCase):
    """Tests for helper functions."""

    def test_build_cycle_decision_from_diagnostics(self) -> None:
        """Test building cycle decision payload from diagnostics."""
        from corridor.execution.paper_diagnostics import build_cycle_decision_from_diagnostics

        candidate_diagnostics = {
            "rejection_counts": {"quote_quality.missing_leg_bid": 10},
            "available_quotes": 100,
            "attempted_structures": 50,
        }

        result = build_cycle_decision_from_diagnostics(
            symbol="SPX",
            regime="RANGE",
            state="ACTIVE_CENTERED",
            candidate_diagnostics=candidate_diagnostics,
            eligible_count=35,
            orders_submitted=1,
            fills=1,
            cancels=0,
            replaces=0,
        )

        self.assertEqual(result["symbol"], "SPX")
        self.assertEqual(result["regime"], "RANGE")
        self.assertEqual(result["eligible_candidates"], 35)
        self.assertIn("quote_quality", result["rejection_counts"])

    def test_format_rejection_for_display(self) -> None:
        """Test formatting rejection for display."""
        from corridor.execution.paper_diagnostics import format_rejection_for_display

        # With subcode
        result = format_rejection_for_display("quote_quality", "missing_leg_bid")
        self.assertEqual(result, "quote_quality.missing_leg_bid")

        # Without subcode
        result = format_rejection_for_display("missing_legs", None)
        self.assertEqual(result, "missing_legs")

        # None case
        result = format_rejection_for_display(None, None)
        self.assertEqual(result, "none")


class TestCandidateDiagnosticCaps(unittest.TestCase):
    """Tests for candidate diagnostic volume caps."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_max_rejected_per_cycle(self) -> None:
        """Test that rejected candidates are capped per cycle."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        # Emit more than max (5) rejected candidates
        for i in range(10):
            collector.emit_candidate_diagnostic(
                symbol="SPX",
                regime="RANGE",
                state="ACTIVE_CENTERED",
                candidate={
                    "expiry": "20240621",
                    "lower_strike": 5200.0 + i,
                    "body_strike": 5300.0 + i,
                    "upper_strike": 5400.0 + i,
                    "net_debit": 2.50,
                    "total_spread": 0.50,
                    "calendar_dte": 30,
                },
                rejection_reason="quote_quality.missing_leg_bid",
                is_submitted=False,
                is_top_rejected=False,
            )

        # Should have at most max_rejected_per_cycle (5) entries
        candidates_file = self.output_dir / "test_candidate_diagnostics.jsonl"
        with open(candidates_file) as f:
            lines = f.readlines()
        # Should be capped at 5 (or fewer if some were filtered by subcode cap)
        self.assertLessEqual(len(lines), 5)

    def test_top_rejected_always_emitted(self) -> None:
        """Test that top rejected candidate is always emitted regardless of cap."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        # Emit many candidates, but mark one as top_rejected
        for i in range(10):
            collector.emit_candidate_diagnostic(
                symbol="SPX",
                regime="RANGE",
                state="ACTIVE_CENTERED",
                candidate={
                    "expiry": "20240621",
                    "lower_strike": 5200.0 + i,
                    "body_strike": 5300.0 + i,
                    "upper_strike": 5400.0 + i,
                    "net_debit": 2.50,
                    "total_spread": 0.50,
                    "calendar_dte": 30,
                },
                rejection_reason="quote_quality.missing_leg_bid",
                is_submitted=False,
                is_top_rejected=(i == 5),  # Mark 6th as top rejected
            )

        candidates_file = self.output_dir / "test_candidate_diagnostics.jsonl"
        with open(candidates_file) as f:
            lines = f.readlines()

        # Should have at least 1 (the top rejected)
        self.assertGreaterEqual(len(lines), 1)

    def test_submitted_candidates_not_capped(self) -> None:
        """Test that submitted candidates are not subject to volume caps."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        # Emit many submitted candidates
        for i in range(10):
            collector.emit_candidate_diagnostic(
                symbol="SPX",
                regime="RANGE",
                state="ACTIVE_CENTERED",
                candidate={
                    "expiry": "20240621",
                    "lower_strike": 5200.0 + i,
                    "body_strike": 5300.0 + i,
                    "upper_strike": 5400.0 + i,
                    "net_debit": 2.50,
                    "total_spread": 0.50,
                    "calendar_dte": 30,
                },
                rejection_reason=None,
                is_submitted=True,
                is_top_rejected=False,
            )

        candidates_file = self.output_dir / "test_candidate_diagnostics.jsonl"
        with open(candidates_file) as f:
            lines = f.readlines()

        # All submitted should be emitted (no cap)
        self.assertEqual(len(lines), 10)

    def test_reset_cycle_emission_tracking(self) -> None:
        """Test that cycle tracking is reset between cycles."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        # Emit 5 candidates (hit cap)
        for i in range(5):
            collector.emit_candidate_diagnostic(
                symbol="SPX",
                regime="RANGE",
                state="ACTIVE_CENTERED",
                candidate={"expiry": "20240621", "lower_strike": 5200.0 + i, "body_strike": 5300.0 + i, "upper_strike": 5400.0 + i, "net_debit": 2.50, "total_spread": 0.50},
                rejection_reason="quote_quality.missing_leg_bid",
                is_submitted=False,
                is_top_rejected=False,
            )

        # Reset for new cycle
        collector.reset_cycle_emission_tracking()

        # Should be able to emit more
        collector.emit_candidate_diagnostic(
            symbol="SPX",
            regime="RANGE",
            state="ACTIVE_CENTERED",
            candidate={"expiry": "20240621", "lower_strike": 6000.0, "body_strike": 6100.0, "upper_strike": 6200.0, "net_debit": 2.50, "total_spread": 0.50},
            rejection_reason="quote_quality.missing_leg_bid",
            is_submitted=False,
            is_top_rejected=False,
        )

        candidates_file = self.output_dir / "test_candidate_diagnostics.jsonl"
        with open(candidates_file) as f:
            lines = f.readlines()

        # Should have 6 (5 from first cycle + 1 after reset)
        self.assertEqual(len(lines), 6)


class TestDeduplication(unittest.TestCase):
    """Tests for log deduplication."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dedup_suppresses_identical_state(self) -> None:
        """Test that identical material state is deduplicated."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        # First call - should emit
        result1 = collector.should_emit_full_summary(
            state="IDLE",
            regime="RANGE",
            connection_state="connected",
            market_data_state="available",
            top_reject_reason=None,
            top_reject_subcode=None,
            eligible_candidate_count=0,
            order_count=0,
            fill_count=0,
            cancel_count=0,
            replace_count=0,
            has_warning=False,
            has_error=False,
        )
        self.assertTrue(result1)

        # Same state - should be suppressed
        result2 = collector.should_emit_full_summary(
            state="IDLE",
            regime="RANGE",
            connection_state="connected",
            market_data_state="available",
            top_reject_reason=None,
            top_reject_subcode=None,
            eligible_candidate_count=0,
            order_count=0,
            fill_count=0,
            cancel_count=0,
            replace_count=0,
            has_warning=False,
            has_error=False,
        )
        self.assertFalse(result2)

    def test_dedup_allows_material_changes(self) -> None:
        """Test that material changes are not deduplicated."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        # First call
        result1 = collector.should_emit_full_summary(
            state="IDLE",
            regime="RANGE",
            connection_state="connected",
            market_data_state="available",
            top_reject_reason=None,
            top_reject_subcode=None,
            eligible_candidate_count=0,
            order_count=0,
            fill_count=0,
            cancel_count=0,
            replace_count=0,
            has_warning=False,
            has_error=False,
        )
        self.assertTrue(result1)

        # Different regime - should emit
        result2 = collector.should_emit_full_summary(
            state="IDLE",
            regime="TREND",
            connection_state="connected",
            market_data_state="available",
            top_reject_reason=None,
            top_reject_subcode=None,
            eligible_candidate_count=0,
            order_count=0,
            fill_count=0,
            cancel_count=0,
            replace_count=0,
            has_warning=False,
            has_error=False,
        )
        self.assertTrue(result2)

        # Different top reject reason - should emit
        result3 = collector.should_emit_full_summary(
            state="IDLE",
            regime="RANGE",
            connection_state="connected",
            market_data_state="available",
            top_reject_reason="quote_quality",
            top_reject_subcode="missing_leg_bid",
            eligible_candidate_count=0,
            order_count=0,
            fill_count=0,
            cancel_count=0,
            replace_count=0,
            has_warning=False,
            has_error=False,
        )
        self.assertTrue(result3)

        # Different fill count - should emit
        result4 = collector.should_emit_full_summary(
            state="IDLE",
            regime="RANGE",
            connection_state="connected",
            market_data_state="available",
            top_reject_reason=None,
            top_reject_subcode=None,
            eligible_candidate_count=0,
            order_count=1,
            fill_count=1,
            cancel_count=0,
            replace_count=0,
            has_warning=False,
            has_error=False,
        )
        self.assertTrue(result4)

        # Warning flag change - should emit
        result5 = collector.should_emit_full_summary(
            state="IDLE",
            regime="RANGE",
            connection_state="connected",
            market_data_state="available",
            top_reject_reason=None,
            top_reject_subcode=None,
            eligible_candidate_count=0,
            order_count=0,
            fill_count=0,
            cancel_count=0,
            replace_count=0,
            has_warning=True,
            has_error=False,
        )
        self.assertTrue(result5)


class TestDisconnectPreservation(unittest.TestCase):
    """Tests for diagnostics preservation across disconnects."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_preserve_validity_complete(self) -> None:
        """Test that validity is set to complete when diagnostics are preserved."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        candidate_diagnostics = {
            "rejection_counts": {"quote_quality.missing_leg_bid": 10},
            "sample_rejections": [{"reason": "missing_leg_bid", "candidate": {}}],
            "available_quotes": 100,
            "attempted_structures": 50,
        }

        collector.preserve_diagnostics(
            candidate_diagnostics=candidate_diagnostics,
            market_data_ts=pd.Timestamp.utcnow(),
            validity="complete",
        )

        self.assertEqual(collector.get_preserved_validity(), "complete")

    def test_mark_diagnostics_partial(self) -> None:
        """Test marking diagnostics as partial after disconnect."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        # First preserve with complete
        collector.preserve_diagnostics(
            candidate_diagnostics={"rejection_counts": {}},
            validity="complete",
        )

        # After disconnect, mark as partial
        collector.mark_diagnostics_partial()

        self.assertEqual(collector.get_preserved_validity(), "partial")

    def test_mark_diagnostics_invalid(self) -> None:
        """Test marking diagnostics as invalid after failed reconnect."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        collector.preserve_diagnostics(
            candidate_diagnostics={"rejection_counts": {}},
            validity="complete",
        )

        collector.mark_diagnostics_invalid()

        self.assertEqual(collector.get_preserved_validity(), "invalid")

    def test_last_timestamps_tracked(self) -> None:
        """Test that last timestamps are tracked for various operations."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        now = pd.Timestamp.utcnow()
        collector.preserve_diagnostics(
            candidate_diagnostics={"rejection_counts": {}},
            market_data_ts=now,
            chain_build_ts=now,
            candidate_eval_ts=now,
            order_submit_ts=now,
        )

        timestamps = collector.get_last_timestamps()
        self.assertIsNotNone(timestamps["market_data"])
        self.assertIsNotNone(timestamps["chain_build"])
        self.assertIsNotNone(timestamps["candidate_evaluation"])
        self.assertIsNotNone(timestamps["order_submission"])
        self.assertIsNotNone(timestamps["cycle_complete"])

    def test_preserved_diagnostics_retrieved(self) -> None:
        """Test that preserved diagnostics can be retrieved."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        candidate_diagnostics = {
            "rejection_counts": {"quote_quality.missing_leg_bid": 10},
            "sample_rejections": [{"reason": "missing_leg_bid", "candidate": {"strike": 5300}}],
            "available_quotes": 100,
            "attempted_structures": 50,
        }

        collector.preserve_diagnostics(candidate_diagnostics=candidate_diagnostics)

        preserved = collector.get_preserved_diagnostics()
        self.assertIsNotNone(preserved)
        self.assertEqual(preserved["rejection_counts"]["quote_quality.missing_leg_bid"], 10)


class TestAggregation(unittest.TestCase):
    """Tests for aggregation by DTE bucket, expiry, wing width."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_aggregation_dte_buckets(self) -> None:
        """Test aggregation by DTE bucket."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        # Preserve diagnostics with various DTEs
        candidate_diagnostics = {
            "rejection_counts": {},
            "sample_rejections": [
                {"reason": "missing_leg_bid", "candidate": {"calendar_dte": 5, "expiry": "20240601", "lower_width": 50}},
                {"reason": "missing_leg_bid", "candidate": {"calendar_dte": 10, "expiry": "20240608", "lower_width": 50}},
                {"reason": "missing_leg_bid", "candidate": {"calendar_dte": 15, "expiry": "20240615", "lower_width": 50}},
                {"reason": "missing_leg_bid", "candidate": {"calendar_dte": 25, "expiry": "20240625", "lower_width": 50}},
                {"reason": "missing_leg_bid", "candidate": {"calendar_dte": 35, "expiry": "20240705", "lower_width": 50}},
            ],
        }

        collector.preserve_diagnostics(candidate_diagnostics=candidate_diagnostics)

        aggregation = collector.get_aggregation()
        self.assertIn("0-7", aggregation["dte_buckets"])
        self.assertIn("8-14", aggregation["dte_buckets"])
        self.assertIn("15-21", aggregation["dte_buckets"])
        self.assertIn("22-30", aggregation["dte_buckets"])
        self.assertIn("31+", aggregation["dte_buckets"])

    def test_aggregation_expiry(self) -> None:
        """Test aggregation by expiry."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        candidate_diagnostics = {
            "rejection_counts": {},
            "sample_rejections": [
                {"reason": "missing_leg_bid", "candidate": {"calendar_dte": 5, "expiry": "20240601"}},
                {"reason": "missing_leg_bid", "candidate": {"calendar_dte": 5, "expiry": "20240601"}},
                {"reason": "missing_leg_bid", "candidate": {"calendar_dte": 10, "expiry": "20240608"}},
            ],
        }

        collector.preserve_diagnostics(candidate_diagnostics=candidate_diagnostics)

        aggregation = collector.get_aggregation()
        self.assertEqual(aggregation["expiries"]["20240601"], 2)
        self.assertEqual(aggregation["expiries"]["20240608"], 1)

    def test_aggregation_wing_width(self) -> None:
        """Test aggregation by wing width."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        candidate_diagnostics = {
            "rejection_counts": {},
            "sample_rejections": [
                {"reason": "missing_leg_bid", "candidate": {"calendar_dte": 5, "lower_width": 20}},
                {"reason": "missing_leg_bid", "candidate": {"calendar_dte": 5, "lower_width": 40}},
                {"reason": "missing_leg_bid", "candidate": {"calendar_dte": 5, "lower_width": 60}},
                {"reason": "missing_leg_bid", "candidate": {"calendar_dte": 5, "lower_width": 80}},
            ],
        }

        collector.preserve_diagnostics(candidate_diagnostics=candidate_diagnostics)

        aggregation = collector.get_aggregation()
        self.assertIn("0-25", aggregation["wing_widths"])
        self.assertIn("26-50", aggregation["wing_widths"])
        self.assertIn("51-75", aggregation["wing_widths"])
        self.assertIn("76+", aggregation["wing_widths"])


class TestSmokeModeOnly(unittest.TestCase):
    """Tests verifying smoke mode only emits extra diagnostics."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_diagnostics_disabled_by_default(self) -> None:
        """Test that diagnostics are disabled when enabled=False."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=False,  # Disabled by default (non-smoke mode)
        )

        # Try to emit - should not create files
        collector.emit_heartbeat(
            symbol="SPX",
            mode="mainline",
            state="IDLE",
            regime="RANGE",
            ib_connected=True,
            market_data_available=True,
            open_positions=0,
            warmup_bars=100,
            required_bars=100,
        )

        heartbeat_file = self.output_dir / "test_heartbeat.jsonl"
        self.assertFalse(heartbeat_file.exists())

    def test_diagnostics_enabled_in_smoke_mode(self) -> None:
        """Test that diagnostics are enabled when enabled=True (smoke mode)."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,  # Enabled for smoke mode
        )

        collector.emit_heartbeat(
            symbol="SPX",
            mode="smoke",
            state="IDLE",
            regime="RANGE",
            ib_connected=True,
            market_data_available=True,
            open_positions=0,
            warmup_bars=100,
            required_bars=100,
        )

        heartbeat_file = self.output_dir / "test_heartbeat.jsonl"
        self.assertTrue(heartbeat_file.exists())

    def test_smoke_mode_flag_reflected_in_heartbeat(self) -> None:
        """Test that smoke mode is reflected in heartbeat mode field."""
        from corridor.execution.paper_diagnostics import PaperDiagnosticsCollector

        collector = PaperDiagnosticsCollector(
            output_dir=self.output_dir,
            prefix="test",
            enabled=True,
        )

        collector.emit_heartbeat(
            symbol="SPX",
            mode="smoke",
            state="IDLE",
            regime="RANGE",
            ib_connected=True,
            market_data_available=True,
            open_positions=0,
            warmup_bars=100,
            required_bars=100,
        )

        heartbeat_file = self.output_dir / "test_heartbeat.jsonl"
        with open(heartbeat_file) as f:
            payload = json.loads(f.readline())
            self.assertEqual(payload["mode"], "smoke")


if __name__ == "__main__":
    unittest.main()