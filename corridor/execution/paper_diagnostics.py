#! python3.12
"""
Paper runner diagnostics module for structured logging and artifact generation.

This module provides:
- Heartbeat logging: compact periodic status updates
- Cycle decision logging: decision funnel per evaluation cycle
- Candidate diagnostic logging: detailed candidate-level diagnostics
- Structured artifacts: JSONL/JSON output for smoke mode analysis
- Log deduplication: suppress repeated full summaries
- Diagnostics preservation: maintain diagnostics across disconnects
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# =============================================================================
# Rejection Reason Hierarchy
# =============================================================================


class RejectionReason:
    """Structured rejection reasons with subcodes for detailed diagnostics."""

    # Quote quality issues
    QUOTE_QUALITY = "quote_quality"
    MISSING_LEG_BID = "quote_quality.missing_leg_bid"
    MISSING_LEG_ASK = "quote_quality.missing_leg_ask"
    STALE_QUOTE = "quote_quality.stale_quote"
    CROSSED_MARKET = "quote_quality.crossed_market"

    # Non-positive debit issues
    NON_POSITIVE_DEBIT = "non_positive_debit"
    ZERO_MID = "non_positive_debit.zero_mid"
    NEGATIVE_MID = "non_positive_debit.negative_mid"
    BAD_COMBO_MATH = "non_positive_debit.bad_combo_math"

    # Spread issues
    SPREAD_TOO_WIDE = "spread_too_wide"
    SPREAD_ABSOLUTE = "spread_too_wide.absolute"
    SPREAD_RATIO = "spread_too_wide.ratio"
    SPREAD_QUOTE_GAP_DISTORTED = "spread_too_wide.quote_gap_distorted"

    # Runner/connectivity issues
    RUNNER = "runner"
    SOCKET_DISCONNECT = "runner.socket_disconnect"
    MARKET_DATA_UNAVAILABLE = "runner.market_data_unavailable"

    # Legacy reasons (kept for backward compatibility)
    MISSING_LEGS = "missing_legs"
    SPREAD_TOO_WIDE_LEGACY = "spread_too_wide"

    @staticmethod
    def parse_reason(reason: str | None) -> tuple[str, Optional[str]]:
        """Parse a reason string into (top_level, subcode) tuple."""
        if not reason:
            return "unknown", None
        if "." in reason:
            parts = reason.split(".", 1)
            return parts[0], parts[1]
        return reason, None

    @staticmethod
    def from_legacy_reason(reason: str | None) -> str:
        """Convert legacy reasons to new hierarchy."""
        if not reason:
            return "unknown"
        if reason == "missing_legs":
            return RejectionReason.QUOTE_QUALITY
        if reason == "non_positive_debit":
            return RejectionReason.NON_POSITIVE_DEBIT
        if reason == "spread_too_wide":
            return RejectionReason.SPREAD_TOO_WIDE
        return reason


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(slots=True)
class HeartbeatPayload:
    """Compact heartbeat log payload."""
    timestamp: str
    mode: str
    symbol: str
    state: str
    regime: str
    ib_connection: str
    market_data: str
    open_positions: int
    warmup_bars: int
    required_bars: int


@dataclass(slots=True)
class CycleDecisionPayload:
    """Cycle decision funnel payload."""
    timestamp: str
    symbol: str
    regime: str
    state: str
    chains_loaded: bool
    candidates_generated: int
    rejected_quote_quality: int
    rejected_non_positive_debit: int
    rejected_spread: int
    rejected_other: int
    eligible_candidates: int
    orders_submitted: int
    fills: int
    cancels: int
    replaces: int
    top_reject_reason: Optional[str] = None
    top_reject_subcode: Optional[str] = None


@dataclass(slots=True)
class CandidateDiagnosticPayload:
    """Detailed candidate diagnostic payload."""
    timestamp: str
    symbol: str
    regime: str
    state: str
    expiry: str
    dte: Optional[int]
    lower_strike: float
    body_strike: float
    upper_strike: float
    wing_width: float
    target_debit: float
    computed_mid_debit: float
    bid_debit: Optional[float]
    ask_debit: Optional[float]
    absolute_spread: float
    spread_ratio: Optional[float]
    quote_complete: bool
    rejection_reason: Optional[str] = None
    rejection_subcode: Optional[str] = None
    is_submitted: bool = False
    is_top_rejected: bool = False


@dataclass(slots=True)
class RunnerEventPayload:
    """Runner connectivity event payload."""
    timestamp: str
    symbol: str
    event_type: str
    connection_state: str
    market_data_state: str
    previous_diagnostics_available: bool
    previous_diagnostics_valid: str  # "complete", "partial", "invalid", "none"
    cycle_complete: bool
    message: Optional[str] = None


# =============================================================================
# Diagnostics Collector
# =============================================================================


class PaperDiagnosticsCollector:
    """
    Collects and manages paper runner diagnostics across cycles.
    
    Provides:
    - Heartbeat generation
    - Cycle decision funnel tracking
    - Candidate diagnostics with subcode hierarchy
    - Log deduplication
    - Diagnostics preservation across disconnects
    """

    def __init__(self, output_dir: Path, prefix: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.output_dir = output_dir
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Artifact paths
        self._heartbeat_path = self.output_dir / f"{prefix}_heartbeat.jsonl"
        self._cycle_path = self.output_dir / f"{prefix}_cycle_decisions.jsonl"
        self._candidates_path = self.output_dir / f"{prefix}_candidate_diagnostics.jsonl"
        self._runner_events_path = self.output_dir / f"{prefix}_runner_events.jsonl"
        self._summary_path = self.output_dir / f"{prefix}_diagnostics_summary.json"

        # State tracking for deduplication
        self._last_heartbeat: Optional[HeartbeatPayload] = None
        self._last_summary_state: Optional[dict[str, Any]] = None
        self._suppression_keys: dict[str, str] = {}

        # Preserved diagnostics across disconnects
        self._preserved_candidate_diagnostics: Optional[dict[str, Any]] = None
        self._preserved_rejection_counts: dict[str, int] = {}
        self._preserved_sample_rejections: list[dict[str, Any]] = []
        self._preserved_validity: str = "none"  # "complete", "partial", "invalid", "none"
        self._last_successful_cycle_ts: Optional[pd.Timestamp] = None
        self._last_market_data_ts: Optional[pd.Timestamp] = None
        self._last_chain_build_ts: Optional[pd.Timestamp] = None
        self._last_candidate_eval_ts: Optional[pd.Timestamp] = None
        self._last_order_submit_ts: Optional[pd.Timestamp] = None

        # Cycle tracking
        self._cycle_count: int = 0

        # Volume caps for candidate diagnostics
        self._max_rejected_per_cycle: int = 5  # Cap rejected candidates per cycle
        self._max_per_subcode: int = 2  # Cap samples per subcode
        self._emitted_this_cycle: set[str] = set()  # Track emitted for dedup within cycle

    def _now_iso(self) -> str:
        return pd.Timestamp.utcnow().isoformat()

    def _now_ny(self) -> str:
        return pd.Timestamp.utcnow().tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M:%S %Z")

    # -------------------------------------------------------------------------
    # Heartbeat Logging
    # -------------------------------------------------------------------------

    def emit_heartbeat(
        self,
        symbol: str,
        mode: str,
        state: str,
        regime: str,
        ib_connected: bool,
        market_data_available: bool,
        open_positions: int,
        warmup_bars: int,
        required_bars: int,
    ) -> None:
        """Emit a compact heartbeat log entry."""
        if not self.enabled:
            return

        payload = HeartbeatPayload(
            timestamp=self._now_iso(),
            mode=mode,
            symbol=symbol,
            state=state,
            regime=regime,
            ib_connection="connected" if ib_connected else "disconnected",
            market_data="available" if market_data_available else "unavailable",
            open_positions=open_positions,
            warmup_bars=warmup_bars,
            required_bars=required_bars,
        )

        self._last_heartbeat = payload
        self._write_jsonl(self._heartbeat_path, payload)

    # -------------------------------------------------------------------------
    # Cycle Decision Logging
    # -------------------------------------------------------------------------

    def emit_cycle_decision(
        self,
        symbol: str,
        regime: str,
        state: str,
        chains_loaded: bool,
        candidates_generated: int,
        rejection_counts: dict[str, int],
        eligible_candidates: int,
        orders_submitted: int,
        fills: int,
        cancels: int,
        replaces: int,
    ) -> None:
        """Emit a cycle decision funnel log entry."""
        if not self.enabled:
            return

        self._cycle_count += 1

        # Parse rejection counts into top-level and subcode
        top_reject_reason: Optional[str] = None
        top_reject_subcode: Optional[str] = None

        if rejection_counts:
            # Find top rejection by count, then by name
            sorted_rejections = sorted(
                rejection_counts.items(),
                key=lambda item: (-int(item[1]), item[0])
            )
            if sorted_rejections:
                top_reason, _ = sorted_rejections[0]
                top_reject_reason, top_reject_subcode = RejectionReason.parse_reason(top_reason)

        # Aggregate rejection categories
        rejected_quote_quality = 0
        rejected_non_positive_debit = 0
        rejected_spread = 0
        rejected_other = 0

        for reason, count in rejection_counts.items():
            top_level, _ = RejectionReason.parse_reason(reason)
            if top_level == "quote_quality":
                rejected_quote_quality += count
            elif top_level == "non_positive_debit":
                rejected_non_positive_debit += count
            elif top_level == "spread_too_wide":
                rejected_spread += count
            else:
                rejected_other += count

        payload = CycleDecisionPayload(
            timestamp=self._now_iso(),
            symbol=symbol,
            regime=regime,
            state=state,
            chains_loaded=chains_loaded,
            candidates_generated=candidates_generated,
            rejected_quote_quality=rejected_quote_quality,
            rejected_non_positive_debit=rejected_non_positive_debit,
            rejected_spread=rejected_spread,
            rejected_other=rejected_other,
            eligible_candidates=eligible_candidates,
            orders_submitted=orders_submitted,
            fills=fills,
            cancels=cancels,
            replaces=replaces,
            top_reject_reason=top_reject_reason,
            top_reject_subcode=top_reject_subcode,
        )

        self._write_jsonl(self._cycle_path, payload)

    # -------------------------------------------------------------------------
    # Candidate Diagnostic Logging
    # -------------------------------------------------------------------------

    def emit_candidate_diagnostic(
        self,
        symbol: str,
        regime: str,
        state: str,
        candidate: dict[str, Any],
        rejection_reason: Optional[str] = None,
        is_submitted: bool = False,
        is_top_rejected: bool = False,
    ) -> None:
        """Emit a detailed candidate diagnostic log entry with volume caps."""
        if not self.enabled:
            return

        # Parse rejection into reason + subcode
        reason, subcode = RejectionReason.parse_reason(rejection_reason)

        # Volume caps: limit rejected candidates per cycle
        if not is_submitted and not is_top_rejected:
            # Check if we've hit the cap for this subcode
            subcode_key = subcode or reason or "unknown"
            if subcode_key in self._emitted_this_cycle:
                return
            if len(self._emitted_this_cycle) >= self._max_rejected_per_cycle:
                return
            self._emitted_this_cycle.add(subcode_key)

        # Also cap per-subcode
        if subcode:
            per_subcode_key = f"subcode:{subcode}"
            subcode_count = sum(1 for k in self._emitted_this_cycle if k.startswith("subcode:"))
            if subcode_count >= self._max_per_subcode and per_subcode_key not in self._emitted_this_cycle:
                # Allow if this is the top rejected
                if not is_top_rejected:
                    return

        payload = CandidateDiagnosticPayload(
            timestamp=self._now_iso(),
            symbol=symbol,
            regime=regime,
            state=state,
            expiry=str(candidate.get("expiry", "")),
            dte=candidate.get("calendar_dte"),
            lower_strike=float(candidate.get("lower_strike", 0)),
            body_strike=float(candidate.get("body_strike", 0)),
            upper_strike=float(candidate.get("upper_strike", 0)),
            wing_width=float(candidate.get("lower_width", 0)),
            target_debit=float(candidate.get("net_debit", 0)),
            computed_mid_debit=float(candidate.get("net_debit", 0)),
            bid_debit=candidate.get("bid_debit"),
            ask_debit=candidate.get("ask_debit"),
            absolute_spread=float(candidate.get("total_spread", 0)),
            spread_ratio=candidate.get("spread_ratio"),
            quote_complete=self._check_quote_completeness(candidate),
            rejection_reason=reason,
            rejection_subcode=subcode,
            is_submitted=is_submitted,
            is_top_rejected=is_top_rejected,
        )

        self._write_jsonl(self._candidates_path, payload)

    def reset_cycle_emission_tracking(self) -> None:
        """Reset emission tracking at start of new cycle."""
        self._emitted_this_cycle.clear()

    def _check_quote_completeness(self, candidate: dict[str, Any]) -> bool:
        """Check if candidate has complete quote data."""
        required_fields = ["lower_strike", "body_strike", "upper_strike", "net_debit", "total_spread"]
        return all(candidate.get(field) is not None for field in required_fields)

    # -------------------------------------------------------------------------
    # Runner Event Logging
    # -------------------------------------------------------------------------

    def emit_runner_event(
        self,
        symbol: str,
        event_type: str,
        connection_state: str,
        market_data_state: str,
        cycle_complete: bool,
        message: Optional[str] = None,
    ) -> None:
        """Emit a runner connectivity event."""
        if not self.enabled:
            return

        payload = RunnerEventPayload(
            timestamp=self._now_iso(),
            symbol=symbol,
            event_type=event_type,
            connection_state=connection_state,
            market_data_state=market_data_state,
            previous_diagnostics_available=self._preserved_candidate_diagnostics is not None,
            previous_diagnostics_valid=self._preserved_validity,
            cycle_complete=cycle_complete,
            message=message,
        )

        self._write_jsonl(self._runner_events_path, payload)

    def emit_socket_disconnect(self, symbol: str, message: Optional[str] = None) -> None:
        """Emit a socket disconnect event while preserving diagnostics."""
        self.emit_runner_event(
            symbol=symbol,
            event_type="socket_disconnect",
            connection_state="disconnected",
            market_data_state="unknown",
            cycle_complete=False,
            message=message,
        )

    def emit_socket_reconnect(self, symbol: str, message: Optional[str] = None) -> None:
        """Emit a socket reconnect event."""
        self.emit_runner_event(
            symbol=symbol,
            event_type="socket_reconnect",
            connection_state="connected",
            market_data_state="available",
            cycle_complete=True,
            message=message,
        )

    # -------------------------------------------------------------------------
    # Diagnostics Preservation
    # -------------------------------------------------------------------------

    def preserve_diagnostics(
        self,
        candidate_diagnostics: Optional[dict[str, Any]],
        market_data_ts: Optional[pd.Timestamp] = None,
        chain_build_ts: Optional[pd.Timestamp] = None,
        candidate_eval_ts: Optional[pd.Timestamp] = None,
        order_submit_ts: Optional[pd.Timestamp] = None,
        validity: str = "complete",
    ) -> None:
        """Preserve candidate diagnostics across cycles/disconnects."""
        if candidate_diagnostics:
            self._preserved_candidate_diagnostics = candidate_diagnostics
            self._preserved_rejection_counts = candidate_diagnostics.get("rejection_counts", {})
            self._preserved_sample_rejections = candidate_diagnostics.get("sample_rejections", [])
            self._preserved_validity = validity
            self._last_successful_cycle_ts = pd.Timestamp.utcnow()

        if market_data_ts:
            self._last_market_data_ts = market_data_ts
        if chain_build_ts:
            self._last_chain_build_ts = chain_build_ts
        if candidate_eval_ts:
            self._last_candidate_eval_ts = candidate_eval_ts
        if order_submit_ts:
            self._last_order_submit_ts = order_submit_ts

    def mark_diagnostics_partial(self) -> None:
        """Mark preserved diagnostics as partial after disconnect."""
        if self._preserved_candidate_diagnostics is not None:
            self._preserved_validity = "partial"

    def mark_diagnostics_invalid(self) -> None:
        """Mark preserved diagnostics as invalid after failed reconnect."""
        self._preserved_validity = "invalid"

    def get_preserved_diagnostics(self) -> Optional[dict[str, Any]]:
        """Retrieve preserved diagnostics."""
        return self._preserved_candidate_diagnostics

    def get_preserved_validity(self) -> str:
        """Get the validity state of preserved diagnostics."""
        return self._preserved_validity

    def get_last_timestamps(self) -> dict[str, Optional[str]]:
        """Get last successful timestamps for various operations."""
        return {
            "market_data": self._last_market_data_ts.isoformat() if self._last_market_data_ts else None,
            "chain_build": self._last_chain_build_ts.isoformat() if self._last_chain_build_ts else None,
            "candidate_evaluation": self._last_candidate_eval_ts.isoformat() if self._last_candidate_eval_ts else None,
            "order_submission": self._last_order_submit_ts.isoformat() if self._last_order_submit_ts else None,
            "cycle_complete": self._last_successful_cycle_ts.isoformat() if self._last_successful_cycle_ts else None,
        }

    # -------------------------------------------------------------------------
    # Log Deduplication
    # -------------------------------------------------------------------------

    def should_emit_full_summary(
        self,
        state: str,
        regime: str,
        connection_state: str,
        market_data_state: str,
        top_reject_reason: Optional[str],
        top_reject_subcode: Optional[str],
        eligible_candidate_count: int,
        order_count: int,
        fill_count: int,
        cancel_count: int,
        replace_count: int,
        has_warning: bool,
        has_error: bool,
    ) -> bool:
        """Determine if a full summary should be emitted based on changes."""
        if not self.enabled:
            return False

        current_key = "|".join([
            state,
            regime,
            connection_state,
            market_data_state,
            top_reject_reason or "none",
            top_reject_subcode or "none",
            str(eligible_candidate_count),
            str(order_count),
            str(fill_count),
            str(cancel_count),
            str(replace_count),
            "warn" if has_warning else "ok",
            "err" if has_error else "ok",
        ])

        if self._last_summary_state == current_key:
            return False

        self._last_summary_state = current_key
        return True

    # -------------------------------------------------------------------------
    # Structured Artifacts
    # -------------------------------------------------------------------------

    def write_cycle_summary(
        self,
        symbol: str,
        date: str,
        cycle_count: int,
        rejection_counts: dict[str, int],
        sample_rejections: list[dict[str, Any]],
        runner_events: list[dict[str, Any]],
    ) -> None:
        """Write a cycle summary artifact."""
        if not self.enabled:
            return

        summary = {
            "report_date": date,
            "symbol": symbol,
            "total_cycles": cycle_count,
            "rejection_counts": rejection_counts,
            "rejection_subcodes": self._aggregate_subcodes(rejection_counts),
            "sample_rejections": sample_rejections[:10],  # Keep top 10
            "runner_events": runner_events[-20:],  # Keep last 20 events
            "last_timestamps": self.get_last_timestamps(),
            "generated_at": self._now_iso(),
        }

        self._summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _aggregate_subcodes(self, rejection_counts: dict[str, int]) -> dict[str, int]:
        """Aggregate rejection counts by subcode."""
        subcode_counts: dict[str, int] = {}
        for reason, count in rejection_counts.items():
            top_level, subcode = RejectionReason.parse_reason(reason)
            if subcode:
                key = f"{top_level}.{subcode}"
                subcode_counts[key] = subcode_counts.get(key, 0) + count
            else:
                subcode_counts[top_level] = subcode_counts.get(top_level, 0) + count
        return subcode_counts

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    def _write_jsonl(self, path: Path, payload: Any) -> None:
        """Write a JSONL entry to a file."""
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def get_diagnostics_summary(self) -> dict[str, Any]:
        """Get a summary of current diagnostics state."""
        return {
            "enabled": self.enabled,
            "cycle_count": self._cycle_count,
            "preserved_diagnostics_available": self._preserved_candidate_diagnostics is not None,
            "preserved_diagnostics_validity": self._preserved_validity,
            "preserved_rejection_count": sum(self._preserved_rejection_counts.values()),
            "last_timestamps": self.get_last_timestamps(),
            "aggregation": self.get_aggregation(),
        }

    # -------------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------------

    def get_aggregation(self) -> dict[str, Any]:
        """Get aggregated diagnostics by DTE bucket, expiry, wing width."""
        if not self._preserved_sample_rejections:
            return {"dte_buckets": {}, "expiries": {}, "wing_widths": {}}

        dte_buckets: dict[str, int] = {}
        expiries: dict[str, int] = {}
        wing_widths: dict[str, int] = {}

        for rejection in self._preserved_sample_rejections:
            candidate = rejection.get("candidate", {})

            # DTE bucket aggregation
            dte = candidate.get("calendar_dte")
            if dte is not None:
                if dte <= 7:
                    bucket = "0-7"
                elif dte <= 14:
                    bucket = "8-14"
                elif dte <= 21:
                    bucket = "15-21"
                elif dte <= 30:
                    bucket = "22-30"
                else:
                    bucket = "31+"
                dte_buckets[bucket] = dte_buckets.get(bucket, 0) + 1

            # Expiry aggregation
            expiry = candidate.get("expiry")
            if expiry:
                expiries[expiry] = expiries.get(expiry, 0) + 1

            # Wing width aggregation
            wing_width = candidate.get("lower_width") or candidate.get("upper_width")
            if wing_width is not None:
                width = float(wing_width)
                if width <= 25:
                    bucket = "0-25"
                elif width <= 50:
                    bucket = "26-50"
                elif width <= 75:
                    bucket = "51-75"
                else:
                    bucket = "76+"
                wing_widths[bucket] = wing_widths.get(bucket, 0) + 1

        return {
            "dte_buckets": dte_buckets,
            "expiries": expiries,
            "wing_widths": wing_widths,
        }


# =============================================================================
# Helper Functions
# =============================================================================


def build_cycle_decision_from_diagnostics(
    symbol: str,
    regime: str,
    state: str,
    candidate_diagnostics: Optional[dict[str, Any]],
    eligible_count: int,
    orders_submitted: int,
    fills: int,
    cancels: int,
    replaces: int,
) -> dict[str, Any]:
    """Build a cycle decision payload from candidate diagnostics."""
    rejection_counts = candidate_diagnostics.get("rejection_counts", {}) if candidate_diagnostics else {}
    chains_loaded = candidate_diagnostics.get("available_quotes", 0) > 0 if candidate_diagnostics else False
    candidates_generated = candidate_diagnostics.get("attempted_structures", 0) if candidate_diagnostics else 0

    return {
        "symbol": symbol,
        "regime": regime,
        "state": state,
        "chains_loaded": chains_loaded,
        "candidates_generated": candidates_generated,
        "rejection_counts": rejection_counts,
        "eligible_candidates": eligible_count,
        "orders_submitted": orders_submitted,
        "fills": fills,
        "cancels": cancels,
        "replaces": replaces,
    }


def format_rejection_for_display(reason: str | None, subcode: Optional[str]) -> str:
    """Format rejection reason for human-readable display."""
    if not reason:
        return "unknown"
    if subcode:
        return f"{reason} ({subcode})"
    return reason


def diagnose_no_fill_session(
    rejection_counts: dict[str, int],
    sample_rejections: list[dict[str, Any]],
    runner_events: list[dict[str, Any]],
    last_timestamps: dict[str, Optional[str]],
) -> dict[str, Any]:
    """
    Diagnose why a session had no fills.
    
    Returns a structured diagnosis with:
    - Primary failure category
    - Likely root cause
    - Recommended action
    """
    total_rejections = sum(rejection_counts.values())

    if not rejection_counts:
        return {
            "diagnosis": "no_candidates",
            "primary_category": "strategy",
            "root_cause": "No candidate butterflies were generated for the current center.",
            "recommended_action": "Check regime filter and center estimation.",
        }

    # Find top rejection
    top_reason = max(rejection_counts.items(), key=lambda item: (int(item[1]), item[0]))[0]
    top_level, subcode = RejectionReason.parse_reason(top_reason)
    top_count = rejection_counts[top_reason]

    # Check for runner/connectivity issues
    disconnect_events = [e for e in runner_events if e.get("event_type") == "socket_disconnect"]
    if disconnect_events:
        return {
            "diagnosis": "connectivity",
            "primary_category": "connectivity",
            "root_cause": f"IB socket disconnected {len(disconnect_events)} time(s) during the session.",
            "recommended_action": "Check IB Gateway/TWS connectivity and network stability.",
        }

    # Analyze by category
    if top_level == "quote_quality":
        return {
            "diagnosis": "quote_quality",
            "primary_category": "quote_quality",
            "root_cause": f"Top rejection: {format_rejection_for_display(top_level, subcode)} ({top_count} rejections)",
            "recommended_action": "Check option chain data quality. May need to wait for better liquidity or use different expiry.",
        }

    if top_level == "non_positive_debit":
        return {
            "diagnosis": "non_positive_debit",
            "primary_category": "quote_quality",
            "root_cause": f"Top rejection: {format_rejection_for_display(top_level, subcode)} ({top_count} rejections)",
            "recommended_action": "Check option quote sanity. Prices may be stale or market is unusual.",
        }

    if top_level == "spread_too_wide":
        if subcode == "ratio":
            return {
                "diagnosis": "spread_ratio",
                "primary_category": "spread_filter",
                "root_cause": f"Top rejection: spread_too_wide.ratio ({top_count} rejections)",
                "recommended_action": "Spread/debit ratio exceeds configured threshold. Consider widening spread tolerance or waiting for tighter markets.",
            }
        return {
            "diagnosis": "spread_absolute",
            "primary_category": "spread_filter",
            "root_cause": f"Top rejection: spread_too_wide.absolute ({top_count} rejections)",
            "recommended_action": "Absolute spread exceeds configured threshold. Consider widening spread tolerance.",
        }

    return {
        "diagnosis": "other",
        "primary_category": "unknown",
        "root_cause": f"Top rejection: {top_reason} ({top_count} rejections)",
        "recommended_action": "Review rejection samples for details.",
    }