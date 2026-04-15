from __future__ import annotations

from collections import Counter
import csv
import json
import math
import os
import socket
import time
from configparser import ConfigParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from corridor.config import CorridorConfig
from corridor.data.ib_contracts import build_option_contract, build_underlying_contract
from corridor.models import ActiveButterfly, ActionRecord, ActionType, CorridorState, LayerKind, Regime
from corridor.notifications.discord import send_discord_json_alert, send_discord_text_alert
from corridor.options.butterfly_selector import ButterflyCandidate, select_butterflies_with_diagnostics
from corridor.options.chain_loader import IBOptionChainLoader, OptionQuote
from corridor.options.combo_builder import ComboLegSpec, build_butterfly_combo
from corridor.strategy.center_estimator import CenterEstimator
from corridor.strategy.corridor_state_machine import CorridorStateMachine
from corridor.strategy.regime import RangeRegimeDetector


try:
    from ib_insync import IB, LimitOrder, util
except ImportError:  # pragma: no cover - optional dependency
    IB = None
    LimitOrder = None
    util = None


@dataclass(slots=True)
class PaperRunnerConfig:
    symbol: str = "SPX"
    mode: str = "delayed"
    host: str = "127.0.0.1"
    port: int = 4001
    client_id: int = 71
    chain_client_id_offset: int = 100
    quantity: int = 1
    poll_seconds: int = 30
    history_days: int = 5
    start_flat: bool = True
    paper_execution: bool = False
    once: bool = False
    check_only: bool = False
    output_dir: Path = Path("corridor_outputs") / "paper_runner"
    order_tif: str = "DAY"
    log_prefix: str = "paper"
    max_spread_pct_of_debit: float = 0.40
    combo_fill_wait_seconds: float = 1.0
    combo_max_chase_steps: int = 3
    combo_chase_fraction_of_spread: float = 0.20
    combo_max_total_debit_ratio: float = 1.15


@dataclass(slots=True)
class ManagedPosition:
    layer_id: int
    candidate: ButterflyCandidate
    quantity: int
    opened_at: pd.Timestamp
    open_limit: float
    open_status: str
    source_action: str
    open_fill_price: Optional[float] = None
    entry_leg_prices: Optional[dict[str, float]] = None
    layer_kind: str = LayerKind.PRIMARY.value
    order_id: Optional[int] = None
    close_order_id: Optional[int] = None
    close_requested_at: Optional[pd.Timestamp] = None
    closed_at: Optional[pd.Timestamp] = None
    close_limit: Optional[float] = None
    close_fill_price: Optional[float] = None
    close_status: str = ""
    close_failure_reason: str = ""


@dataclass(slots=True)
class InProgressBar:
    bucket_start: pd.Timestamp
    last_sample_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def managed_position_to_payload(position: ManagedPosition) -> dict[str, Any]:
    return {
        "layer_id": position.layer_id,
        "quantity": position.quantity,
        "opened_at": position.opened_at.isoformat(),
        "open_limit": position.open_limit,
        "open_fill_price": position.open_fill_price,
        "entry_leg_prices": (
            {str(key): float(value) for key, value in position.entry_leg_prices.items()}
            if position.entry_leg_prices
            else None
        ),
        "open_status": position.open_status,
        "source_action": position.source_action,
        "layer_kind": position.layer_kind,
        "order_id": position.order_id,
        "close_order_id": position.close_order_id,
        "close_requested_at": (
            position.close_requested_at.isoformat() if position.close_requested_at is not None else None
        ),
        "closed_at": position.closed_at.isoformat() if position.closed_at is not None else None,
        "close_limit": position.close_limit,
        "close_fill_price": position.close_fill_price,
        "close_status": position.close_status,
        "close_failure_reason": position.close_failure_reason,
        "candidate": {
            "symbol": position.candidate.symbol,
            "expiry": position.candidate.expiry,
            "lower_strike": position.candidate.lower_strike,
            "body_strike": position.candidate.body_strike,
            "upper_strike": position.candidate.upper_strike,
            "lower_width": position.candidate.lower_width,
            "upper_width": position.candidate.upper_width,
            "trading_class": position.candidate.trading_class,
            "net_debit": position.candidate.net_debit,
            "total_spread": position.candidate.total_spread,
            "max_risk": position.candidate.max_risk,
            "max_reward": position.candidate.max_reward,
            "right": position.candidate.right,
            "wing_mode": position.candidate.wing_mode,
        },
    }


def managed_position_from_payload(payload: dict[str, Any]) -> ManagedPosition:
    candidate_payload = payload["candidate"]
    candidate = ButterflyCandidate(
        symbol=str(candidate_payload["symbol"]),
        expiry=str(candidate_payload["expiry"]),
        lower_strike=float(candidate_payload["lower_strike"]),
        body_strike=float(candidate_payload["body_strike"]),
        upper_strike=float(candidate_payload["upper_strike"]),
        lower_width=float(candidate_payload.get("lower_width", float(candidate_payload["body_strike"]) - float(candidate_payload["lower_strike"]))),
        upper_width=float(candidate_payload.get("upper_width", float(candidate_payload["upper_strike"]) - float(candidate_payload["body_strike"]))),
        trading_class=(
            str(candidate_payload.get("trading_class", "") or "") or None
        ),
        net_debit=float(candidate_payload["net_debit"]),
        total_spread=float(candidate_payload["total_spread"]),
        max_risk=float(candidate_payload["max_risk"]),
        max_reward=float(candidate_payload["max_reward"]),
        right=str(candidate_payload["right"]),
        wing_mode=str(candidate_payload.get("wing_mode", "symmetric") or "symmetric"),
    )
    return ManagedPosition(
        layer_id=int(payload["layer_id"]),
        candidate=candidate,
        quantity=int(payload["quantity"]),
        opened_at=_ensure_utc_timestamp(pd.Timestamp(payload["opened_at"])),
        open_limit=float(payload.get("open_limit", 0.0) or 0.0),
        open_fill_price=_coerce_optional_float(payload.get("open_fill_price")),
        entry_leg_prices=(
            {
                str(key): float(value)
                for key, value in dict(payload.get("entry_leg_prices") or {}).items()
            }
            or None
        ),
        open_status=str(payload.get("open_status", "")),
        source_action=str(payload.get("source_action", "")),
        layer_kind=str(payload.get("layer_kind", LayerKind.PRIMARY.value)),
        order_id=_coerce_optional_int(payload.get("order_id")),
        close_order_id=_coerce_optional_int(payload.get("close_order_id")),
        close_requested_at=_coerce_optional_timestamp(payload.get("close_requested_at")),
        closed_at=_coerce_optional_timestamp(payload.get("closed_at")),
        close_limit=_coerce_optional_float(payload.get("close_limit")),
        close_fill_price=_coerce_optional_float(payload.get("close_fill_price")),
        close_status=str(payload.get("close_status", "")),
        close_failure_reason=str(payload.get("close_failure_reason", "")),
    )


def _coerce_optional_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value in (None, ""):
        return None
    return _ensure_utc_timestamp(pd.Timestamp(value))


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _require_ib() -> None:
    if IB is None or LimitOrder is None or util is None:
        raise RuntimeError("ib_insync is required for run_paper_corridor.py")


def _ensure_utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _log_timestamp_now() -> str:
    return _ensure_utc_timestamp(pd.Timestamp.utcnow()).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M:%S %Z")


def _timeframe_to_delta(label: str) -> pd.Timedelta:
    value, unit = label.strip().split(maxsplit=1)
    amount = int(value)
    unit = unit.lower()
    if unit.startswith("min"):
        return pd.Timedelta(minutes=amount)
    if unit.startswith("hour"):
        return pd.Timedelta(hours=amount)
    if unit.startswith("day"):
        return pd.Timedelta(days=amount)
    raise ValueError(f"Unsupported timeframe: {label}")


def _format_ib_history_error(errors: list[tuple[int, str]], context: str) -> str:
    if not errors:
        return f"IB returned no historical bars for {context}."

    code, message = errors[-1]
    cleaned = message.strip()
    if code == 162 and "different IP address" in cleaned:
        return (
            f"IB historical bars unavailable for {context}: {cleaned} "
            "Close the other trading TWS/Gateway session or reconnect from the same IP."
        )
    return f"IB historical bars unavailable for {context}: [{code}] {cleaned}"


def _status_priority(status: str) -> int:
    mapping = {"PASS": 0, "WARN": 1, "FAIL": 2}
    return mapping.get(str(status).upper(), 1)


def _points_to_dollars(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value) * 100.0, 2)


def _format_optional_dollar(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"${value:.2f}"


def _format_optional_price(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _format_signed_dollar(value: float) -> str:
    return f"${value:+.2f}"


def _make_check(status: str, message: str) -> dict[str, str]:
    return {"status": status, "message": message}


def build_paper_test_summary(state_payload: dict[str, Any], daily_report_payload: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, dict[str, str]] = {}

    history_seeded = bool(daily_report_payload.get("history_seeded"))
    model_ready = bool(daily_report_payload.get("model_ready"))
    warmup_mode = bool(daily_report_payload.get("warmup_mode"))
    execution_halted_reason = str(daily_report_payload.get("execution_halted_reason") or "").strip()
    candidate_error = str(daily_report_payload.get("candidate_error") or "").strip()
    filled_orders = int(daily_report_payload.get("filled_orders_today") or 0)
    blocked_or_skipped = int(daily_report_payload.get("blocked_or_skipped_orders_today") or 0)
    skipped_orders = int(daily_report_payload.get("skipped_orders_today") or 0)
    execution_failure_orders = int(daily_report_payload.get("execution_failure_orders_today") or 0)
    open_positions_count = int(daily_report_payload.get("open_positions_count") or 0)
    candidate_diagnostics = daily_report_payload.get("candidate_diagnostics") or {}
    rejection_counts = candidate_diagnostics.get("rejection_counts") if isinstance(candidate_diagnostics, dict) else {}
    top_rejection = None
    if isinstance(rejection_counts, dict) and rejection_counts:
        top_rejection = max(rejection_counts.items(), key=lambda item: (int(item[1]), item[0]))

    if history_seeded and model_ready and not warmup_mode:
        checks["startup"] = _make_check("PASS", "History seeded and model ready.")
    elif model_ready:
        checks["startup"] = _make_check("WARN", "Model ready, but warmup/history state is not ideal.")
    else:
        checks["startup"] = _make_check("WARN", "Runner is still warming up or waiting for enough bars.")

    if execution_halted_reason:
        checks["runner"] = _make_check("FAIL", execution_halted_reason)
    elif candidate_error:
        checks["runner"] = _make_check("WARN", candidate_error)
    else:
        checks["runner"] = _make_check("PASS", "Runner is operating without a recorded execution halt.")

    open_fill_edge = daily_report_payload.get("avg_open_fill_edge_vs_quote")
    close_fill_edge = daily_report_payload.get("avg_close_fill_edge_vs_quote")
    open_fill_edge = float(open_fill_edge) if open_fill_edge not in (None, "") else None
    close_fill_edge = float(close_fill_edge) if close_fill_edge not in (None, "") else None
    open_fill_edge_dollars = _points_to_dollars(open_fill_edge)
    close_fill_edge_dollars = _points_to_dollars(close_fill_edge)

    if open_fill_edge is None and close_fill_edge is None:
        checks["fills"] = _make_check("WARN", "No filled orders yet today; fill quality is not evaluated.")
    else:
        fill_status = "PASS"
        if open_fill_edge is not None and open_fill_edge < -0.60:
            fill_status = "FAIL"
        elif close_fill_edge is not None and close_fill_edge < -0.80:
            fill_status = "FAIL"
        elif open_fill_edge is not None and open_fill_edge < -0.30:
            fill_status = "WARN"
        elif close_fill_edge is not None and close_fill_edge < -0.40:
            fill_status = "WARN"
        checks["fills"] = _make_check(
            fill_status,
            (
                "Average fill edge vs quote "
                f"(open={_format_optional_dollar(open_fill_edge_dollars)}, "
                f"close={_format_optional_dollar(close_fill_edge_dollars)})."
            ),
        )

    spread_ratio = daily_report_payload.get("avg_filled_spread_ratio")
    spread_ratio = float(spread_ratio) if spread_ratio not in (None, "") else None
    if spread_ratio is None:
        checks["spread"] = _make_check("WARN", "No filled orders yet; spread quality is not evaluated.")
    elif spread_ratio <= 0.25:
        checks["spread"] = _make_check("PASS", f"Average filled spread ratio is {spread_ratio:.4f}.")
    elif spread_ratio <= 0.40:
        checks["spread"] = _make_check("WARN", f"Average filled spread ratio is elevated at {spread_ratio:.4f}.")
    else:
        checks["spread"] = _make_check("FAIL", f"Average filled spread ratio is too wide at {spread_ratio:.4f}.")

    configured_wing_mode = str(
        daily_report_payload.get("configured_wing_mode")
        or state_payload.get("configured_wing_mode")
        or ""
    ).strip()
    fallback_rate = float(daily_report_payload.get("adaptive_fallback_rate") or 0.0)
    fallback_dist = daily_report_payload.get("fallback_type_distribution") or {}
    fallback_counts = fallback_dist.get("counts") if isinstance(fallback_dist, dict) else {}
    broken_upper = int((fallback_counts or {}).get("broken_upper", 0))
    broken_lower = int((fallback_counts or {}).get("broken_lower", 0))
    if configured_wing_mode and configured_wing_mode != "adaptive":
        checks["adaptive"] = _make_check(
            "PASS",
            f"Adaptive fallback not applicable because configured wing_mode={configured_wing_mode}.",
        )
    else:
        if fallback_rate <= 0.35:
            adaptive_status = "PASS"
        elif fallback_rate <= 0.60:
            adaptive_status = "WARN"
        else:
            adaptive_status = "FAIL"
        checks["adaptive"] = _make_check(
            adaptive_status,
            (
                f"Adaptive fallback rate {fallback_rate:.2%} "
                f"(broken_upper={broken_upper}, broken_lower={broken_lower})."
            ),
        )

    if execution_failure_orders == 0 and blocked_or_skipped == 0:
        checks["orders"] = _make_check(
            "PASS",
            f"Filled orders today={filled_orders}, blocked/skipped={blocked_or_skipped}, open_positions={open_positions_count}.",
        )
    elif execution_failure_orders == 0:
        checks["orders"] = _make_check(
            "WARN",
            (
                f"No real execution failures yet; skipped candidate attempts={skipped_orders}, "
                f"filled={filled_orders}, open_positions={open_positions_count}."
            ),
        )
    elif execution_failure_orders <= 2:
        checks["orders"] = _make_check(
            "WARN",
            (
                f"Execution failures today={execution_failure_orders}; skipped candidate attempts={skipped_orders}, "
                f"filled={filled_orders}, open_positions={open_positions_count}."
            ),
        )
    else:
        checks["orders"] = _make_check(
            "FAIL",
            (
                f"Too many real execution failures today={execution_failure_orders}; "
                f"skipped candidate attempts={skipped_orders}, filled={filled_orders}, "
                f"open_positions={open_positions_count}."
            ),
        )

    if top_rejection is None:
        checks["diagnostics"] = _make_check("WARN", "No candidate rejection diagnostics available yet.")
    else:
        reason, count = top_rejection
        diagnostic_status = "WARN" if count > 0 else "PASS"
        checks["diagnostics"] = _make_check(
            diagnostic_status,
            f"Top candidate rejection: {reason}={count}.",
        )

    overall_status = "PASS"
    for payload in checks.values():
        if _status_priority(payload["status"]) > _status_priority(overall_status):
            overall_status = payload["status"]

    suggested_action = {
        "PASS": "Paper test looks healthy. Keep running and review the fill audit after the session.",
        "WARN": "Paper test is usable, but review the warning fields before considering live capital.",
        "FAIL": "Do not promote this setup. Fix the failing execution or quality checks first.",
    }[overall_status]

    return {
        "report_timestamp": daily_report_payload.get("report_timestamp"),
        "report_date": daily_report_payload.get("report_date"),
        "symbol": daily_report_payload.get("symbol") or state_payload.get("symbol"),
        "overall_status": overall_status,
        "headline": {
            "PASS": "Paper day is within pass thresholds.",
            "WARN": "Paper day is mixed; some checks need review.",
            "FAIL": "Paper day failed one or more hard checks.",
        }[overall_status],
        "suggested_action": suggested_action,
        "latest_state": daily_report_payload.get("latest_state"),
        "latest_regime": daily_report_payload.get("latest_regime"),
        "execution_mode": daily_report_payload.get("execution_mode"),
        "startup_mode": daily_report_payload.get("startup_mode"),
        "filled_orders_today": filled_orders,
        "open_positions_count": open_positions_count,
        "avg_open_fill_edge_vs_quote_dollars": open_fill_edge_dollars,
        "avg_close_fill_edge_vs_quote_dollars": close_fill_edge_dollars,
        "avg_filled_spread_ratio": spread_ratio,
        "adaptive_fallback_rate": fallback_rate,
        "candidate_diagnostics": candidate_diagnostics,
        "checks": checks,
        "pass_count": sum(1 for payload in checks.values() if payload["status"] == "PASS"),
        "warn_count": sum(1 for payload in checks.values() if payload["status"] == "WARN"),
        "fail_count": sum(1 for payload in checks.values() if payload["status"] == "FAIL"),
    }


def format_paper_test_summary(summary_payload: dict[str, Any]) -> str:
    checks = summary_payload.get("checks", {})
    lines = [
        (
            f"Paper Test Summary | {summary_payload.get('overall_status', 'WARN')} | "
            f"{summary_payload.get('report_date', 'n/a')} | {summary_payload.get('symbol', 'n/a')}"
        ),
        (
            f"Headline: {summary_payload.get('headline', '')} "
            f"State={summary_payload.get('latest_state', 'n/a')} "
            f"Regime={summary_payload.get('latest_regime', 'n/a')} "
            f"Startup={summary_payload.get('startup_mode', 'n/a')}"
        ),
    ]
    for key in ["startup", "runner", "fills", "spread", "adaptive", "orders", "diagnostics"]:
        payload = checks.get(key)
        if not isinstance(payload, dict):
            continue
        lines.append(f"{key.capitalize()}: {payload.get('status', 'WARN')} | {payload.get('message', '')}")
    lines.append(f"Action: {summary_payload.get('suggested_action', '')}")
    return "\n".join(lines) + "\n"


class CsvEventLogger:
    """Append runtime transitions, actions, and orders to CSV files."""

    def __init__(self, output_dir: Path, prefix: str) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "transitions": self.output_dir / f"{self.prefix}_transitions.csv",
            "actions": self.output_dir / f"{self.prefix}_actions.csv",
            "orders": self.output_dir / f"{self.prefix}_orders.csv",
            "state": self.output_dir / f"{self.prefix}_state.json",
            "recovery": self.output_dir / f"{self.prefix}_recovery.json",
            "daily_report_json": self.output_dir / f"{self.prefix}_daily_report.json",
            "daily_report_csv": self.output_dir / f"{self.prefix}_daily_report.csv",
            "test_summary_json": self.output_dir / f"{self.prefix}_test_summary.json",
            "test_summary_csv": self.output_dir / f"{self.prefix}_test_summary.csv",
            "test_summary_txt": self.output_dir / f"{self.prefix}_test_summary.txt",
        }

    def write_transition(self, record: dict[str, Any]) -> None:
        self._append(self.paths["transitions"], record)

    def write_action(self, record: dict[str, Any]) -> None:
        self._append(self.paths["actions"], record)

    def write_order(self, record: dict[str, Any]) -> None:
        self._append(self.paths["orders"], record)

    def write_state(self, payload: dict[str, Any]) -> None:
        self.paths["state"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_recovery(self, payload: dict[str, Any]) -> None:
        self.paths["recovery"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_daily_report(self, payload: dict[str, Any]) -> None:
        self.paths["daily_report_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self.paths["daily_report_csv"].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload.keys()))
            writer.writeheader()
            writer.writerow(payload)

    def write_test_summary(self, payload: dict[str, Any], summary_text: str) -> None:
        self.paths["test_summary_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self.paths["test_summary_csv"].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload.keys()))
            writer.writeheader()
            writer.writerow(payload)
        self.paths["test_summary_txt"].write_text(summary_text, encoding="utf-8")

    def read_state(self) -> Optional[dict[str, Any]]:
        path = self.paths["state"]
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def read_recovery(self) -> Optional[dict[str, Any]]:
        path = self.paths["recovery"]
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _append(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if needs_header:
                writer.writeheader()
            writer.writerow(row)


class PaperCorridorRunner:
    """Run the corridor logic on live IBKR bars and optionally place paper combo orders."""

    def __init__(self, corridor_cfg: CorridorConfig, runner_cfg: PaperRunnerConfig) -> None:
        _require_ib()
        self.cfg = corridor_cfg
        self.runner_cfg = runner_cfg
        self.ib = IB()
        self.detector = RangeRegimeDetector(corridor_cfg)
        self.estimator = CenterEstimator(corridor_cfg)
        self.machine = CorridorStateMachine(corridor_cfg)
        self.logger = CsvEventLogger(runner_cfg.output_dir, runner_cfg.log_prefix)
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
        self.history = pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        self.underlying_contract = None
        self.last_processed_ts: Optional[pd.Timestamp] = None
        self.positions: dict[int, ManagedPosition] = {}
        self.bar_delta = _timeframe_to_delta(self.cfg.timeframe)
        self.market_data_type = 3 if self.runner_cfg.mode == "delayed" else 1
        self.market_ticker = None
        self.partial_bar: Optional[InProgressBar] = None
        self.last_underlying_volume_total: Optional[float] = None
        self.warmup_mode = False
        self.warmup_reason: Optional[str] = None
        self.warmup_complete_announced = False
        self.execution_halted_reason: Optional[str] = None
        self.history_refresh_error: Optional[str] = None
        self.warmup_quote_error: Optional[str] = None
        self.last_outage_notice: Optional[str] = None
        self.history_seed_status: str = "History seed has not run yet."
        self.latest_candidate_diagnostics: dict[str, Any] | None = None
        self.wing_stats: dict[str, int] = {
            "symmetric": 0,
            "broken_upper": 0,
            "broken_lower": 0,
            "guard_fails": 0,
        }
        self._restore_persistent_stats()

    def _restore_persistent_stats(self) -> None:
        payload = self.logger.read_state()
        if not payload:
            return
        restored = payload.get("wing_stats")
        if not isinstance(restored, dict):
            return
        for key in self.wing_stats:
            value = restored.get(key)
            if isinstance(value, (int, float)):
                self.wing_stats[key] = int(value)

    def _adaptive_stats_payload(self) -> tuple[float, dict[str, Any]]:
        symmetric = int(self.wing_stats.get("symmetric", 0))
        broken_upper = int(self.wing_stats.get("broken_upper", 0))
        broken_lower = int(self.wing_stats.get("broken_lower", 0))
        fallback_total = broken_upper + broken_lower
        total_selected = symmetric + fallback_total
        fallback_rate = (fallback_total / total_selected) if total_selected > 0 else 0.0
        distribution = {
            "counts": {
                "broken_upper": broken_upper,
                "broken_lower": broken_lower,
            },
            "rates": {
                "broken_upper": round(broken_upper / fallback_total, 4) if fallback_total > 0 else 0.0,
                "broken_lower": round(broken_lower / fallback_total, 4) if fallback_total > 0 else 0.0,
            },
            "guard_fails": int(self.wing_stats.get("guard_fails", 0)),
            "total_selected": total_selected,
        }
        return round(fallback_rate, 4), distribution

    def run(self) -> int:
        self.connect()
        try:
            self._seed_or_fallback_to_warmup()
            self._restore_recovery_state()
            self._guard_startup_account_state()
            if self.runner_cfg.check_only:
                if self.warmup_mode:
                    self._poll_warmup_once(allow_orders=False)
                self.print_snapshot(label="startup-check", persist=True)
                return 0

            if self.runner_cfg.once:
                self.poll_once()
                self.print_snapshot(label="single-pass", persist=True)
                return 0

            print(
                f"Paper corridor runner started | symbol={self.cfg.symbol} | mode={self.runner_cfg.mode} | "
                f"execution={'paper' if self.runner_cfg.paper_execution else 'dry-run'} | timeframe={self.cfg.timeframe}"
            )
            while True:
                self.poll_once()
                time.sleep(max(5, self.runner_cfg.poll_seconds))
        except KeyboardInterrupt:
            print("Paper corridor runner stopped.")
            return 0
        finally:
            self.disconnect()

    def connect(self) -> None:
        port = self._resolve_ib_port(self.runner_cfg.host, self.runner_cfg.port)
        if port != self.runner_cfg.port:
            print(f"Requested IB port {self.runner_cfg.port} was not reachable; using {port} instead.")
            self.runner_cfg.port = port
        self.ib.connect(self.runner_cfg.host, self.runner_cfg.port, clientId=self.runner_cfg.client_id, timeout=10)
        self.ib.reqMarketDataType(self.market_data_type)
        self.underlying_contract = build_underlying_contract(self.cfg.symbol, self.cfg.ib_exchange, self.cfg.ib_currency)
        self.ib.qualifyContracts(self.underlying_contract)

    def disconnect(self) -> None:
        if self.market_ticker is not None and self.underlying_contract is not None:
            try:
                self.ib.cancelMktData(self.underlying_contract)
            except Exception:
                pass
        if self.ib.isConnected():
            self.ib.disconnect()

    def _restore_recovery_state(self) -> None:
        if self.runner_cfg.start_flat:
            return
        payload = self.logger.read_recovery()
        if not payload:
            return
        if str(payload.get("symbol", "")).upper() != self.cfg.symbol.upper():
            return

        recovered = [
            managed_position_from_payload(item)
            for item in payload.get("positions", [])
        ]
        self.positions = {position.layer_id: position for position in recovered}
        self.machine.context.active_layers = [
            self._managed_position_to_active_layer(position)
            for position in sorted(self.positions.values(), key=lambda item: item.layer_id)
        ]
        if self.positions:
            self.machine.context.state = CorridorState.ACTIVE_CENTERED
            self.machine.context.current_center = float(
                payload.get("current_center")
                or self._primary_center_from_positions(self.positions.values())
            )
            self.machine.context.next_layer_id = max(self.positions) + 1
        else:
            state_name = str(payload.get("state", CorridorState.IDLE.value))
            try:
                self.machine.context.state = CorridorState(state_name)
            except ValueError:
                self.machine.context.state = CorridorState.IDLE
            self.machine.context.current_center = _coerce_optional_float(payload.get("current_center"))
            self.machine.context.next_layer_id = int(payload.get("next_layer_id", 1) or 1)
        restored_last_entry = str(payload.get("last_primary_entry_session_date") or "").strip()
        if restored_last_entry:
            self.machine.context.last_primary_entry_session_date = restored_last_entry
        restored_last_take_profit = str(payload.get("last_take_profit_session_date") or "").strip()
        if restored_last_take_profit:
            self.machine.context.last_take_profit_session_date = restored_last_take_profit
        elif self.positions:
            primary = self._primary_position()
            if primary is not None:
                opened_local = _ensure_utc_timestamp(primary.opened_at).tz_convert("America/New_York").date().isoformat()
                self.machine.context.last_primary_entry_session_date = opened_local
        print(
            f"Loaded {len(self.positions)} recovered paper position(s) from "
            f"{self.logger.paths['recovery'].name}."
        )

    def _seed_or_fallback_to_warmup(self) -> None:
        try:
            frame = self.fetch_recent_history()
            self.seed_from_history(frame)
            self.warmup_mode = False
            self.warmup_reason = None
            self.history_refresh_error = None
            self.history_seed_status = (
                f"History seed successful. Loaded {len(self.history)} completed bars from IB historical data."
            )
            print(
                f"Startup mode | history-seeded | history_bars={len(self.history)} | "
                f"required_bars={self.required_warmup_bars} | model_ready={len(self.history) >= self.required_warmup_bars}"
            )
        except RuntimeError as exc:
            message = str(exc).strip() or exc.__class__.__name__
            self._activate_warmup_mode(message)

    def fetch_recent_history(self) -> pd.DataFrame:
        duration = f"{max(1, self.runner_cfg.history_days)} D"
        errors: list[tuple[int, str]] = []

        def capture_error(_req_id: int, error_code: int, error_string: str, contract) -> None:
            if contract is None or getattr(contract, "symbol", None) == self.cfg.symbol:
                errors.append((error_code, error_string))

        self.ib.errorEvent += capture_error
        try:
            bars = self.ib.reqHistoricalData(
                self.underlying_contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=self.cfg.timeframe,
                whatToShow=self.cfg.ib_what_to_show,
                useRTH=self.cfg.ib_use_rth,
                formatDate=1,
            )
        finally:
            self.ib.errorEvent -= capture_error

        if not bars:
            raise RuntimeError(_format_ib_history_error(errors, f"{self.cfg.symbol} ({duration}, {self.cfg.timeframe})"))

        frame = util.df(bars)
        if frame is None or frame.empty:
            raise RuntimeError(_format_ib_history_error(errors, f"{self.cfg.symbol} ({duration}, {self.cfg.timeframe})"))
        frame["timestamp"] = pd.to_datetime(frame["date"], utc=True)
        frame["symbol"] = self.cfg.symbol
        frame = frame[["timestamp", "symbol", "open", "high", "low", "close", "volume"]].copy()
        frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp", "symbol"])
        return frame.reset_index(drop=True)

    def seed_from_history(self, frame: pd.DataFrame) -> None:
        eligible = self._completed_bars(frame)
        if eligible.empty:
            raise RuntimeError("No completed bars available to seed the paper runner.")
        self.history = eligible.copy().reset_index(drop=True)
        self.last_processed_ts = pd.Timestamp(self.history["timestamp"].iloc[-1])

        if self.runner_cfg.start_flat:
            self.machine = CorridorStateMachine(self.cfg)
            self.positions.clear()
            print(f"Seeded {len(self.history)} completed bars and reset to flat start.")
            return

        for _, row in self.history.iterrows():
            self._process_bar(row, allow_orders=False, emit_logs=False)
        print(f"Seeded {len(self.history)} completed bars and synced live state from history.")

    def poll_once(self) -> None:
        if self.warmup_mode:
            self._poll_warmup_once(allow_orders=True)
            return

        try:
            fresh = self.fetch_recent_history()
        except RuntimeError as exc:
            message = str(exc).strip() or exc.__class__.__name__
            if len(self.history) >= self.required_warmup_bars:
                self.history_refresh_error = message
                self._print_outage_notice(
                    "Historical refresh unavailable; preserving current seeded state and retrying next poll. "
                    f"Reason: {message}"
                )
                self._refresh_positions_from_account()
                self._write_state_snapshot()
                return
            self._activate_warmup_mode(message)
            self._poll_warmup_once(allow_orders=True)
            return
        self.history_refresh_error = None
        self.last_outage_notice = None

        eligible = self._completed_bars(fresh)
        if self.last_processed_ts is not None:
            eligible = eligible[eligible["timestamp"] > self.last_processed_ts]
        if eligible.empty:
            self._refresh_positions_from_account()
            self._retry_pending_session_closes()
            message = (
                f"No new completed bars. | ts={_log_timestamp_now()}"
                f"{self._intraday_pnl_log_suffix()}"
            )
            print(message)
            self._maybe_send_discord_log_alert(message)
            self._write_state_snapshot()
            return

        for _, row in eligible.iterrows():
            self.history = (
                pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)
                .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
                .sort_values("timestamp")
                .tail(max(300, self.cfg.regime_lookback * 4))
                .reset_index(drop=True)
            )
            self._process_bar(row, allow_orders=True)
            self.last_processed_ts = pd.Timestamp(row["timestamp"])

        self._refresh_positions_from_account()
        self._write_state_snapshot()

    def _activate_warmup_mode(self, reason: str) -> None:
        self.warmup_mode = True
        self.warmup_reason = reason
        self.warmup_quote_error = None
        self.history = self.history.copy()
        self.partial_bar = None
        self.last_underlying_volume_total = None
        self.warmup_complete_announced = len(self.history) >= self.required_warmup_bars
        self.history_seed_status = (
            f"History seed unavailable. Running in warmup-only mode from live market data. Reason: {reason}"
        )
        self._ensure_underlying_ticker()
        print(f"History seed unavailable; falling back to warmup mode. Reason: {reason}")
        print(
            f"Warmup mode active | completed_bars={len(self.history)} | "
            f"required_bars={self.required_warmup_bars} | timeframe={self.cfg.timeframe}"
        )
        print(
            f"Startup mode | warmup-only | history_bars={len(self.history)} | "
            f"required_bars={self.required_warmup_bars} | model_ready={len(self.history) >= self.required_warmup_bars}"
        )

    @property
    def required_warmup_bars(self) -> int:
        return max(self.cfg.center_lookback, self.cfg.regime_lookback)

    def _intraday_pnl_log_suffix(self) -> str:
        realized_dollars = self._today_realized_pnl_dollars()
        unrealized_dollars = self._open_unrealized_pnl_dollars()
        total_dollars = realized_dollars + unrealized_dollars
        return (
            f" | today_est_pnl={_format_signed_dollar(total_dollars)}"
            f" | realized={_format_signed_dollar(realized_dollars)}"
            f" | unrealized={_format_signed_dollar(unrealized_dollars)}"
            f" | open_positions={len(self.positions)}"
        )

    def _today_realized_pnl_dollars(self) -> float:
        local_date_iso = _ensure_utc_timestamp(pd.Timestamp.utcnow()).tz_convert("America/New_York").date().isoformat()
        orders_today = self._rows_for_local_date(self.logger.paths["orders"], local_date_iso)
        entry_queues: dict[int, list[tuple[float, int]]] = {}
        realized_dollars = 0.0
        for row in orders_today:
            if row.get("status") != "Filled":
                continue
            try:
                layer_id = int(row.get("layer_id") or 0)
                quantity = int(row.get("quantity") or 0)
                fill_price = float(row.get("fill_price") or 0.0)
            except (TypeError, ValueError):
                continue
            if layer_id <= 0 or quantity <= 0 or fill_price <= 0:
                continue
            side = str(row.get("side") or "").strip().upper()
            if side == "OPEN":
                entry_queues.setdefault(layer_id, []).append((fill_price, quantity))
                continue
            if side != "CLOSE":
                continue
            queue = entry_queues.get(layer_id)
            if not queue:
                continue
            entry_fill_price, entry_quantity = queue.pop(0)
            matched_quantity = min(quantity, entry_quantity)
            realized_dollars += (fill_price - entry_fill_price) * 100.0 * matched_quantity
            remaining_quantity = entry_quantity - matched_quantity
            if remaining_quantity > 0:
                queue.insert(0, (entry_fill_price, remaining_quantity))
        return round(realized_dollars, 2)

    def _open_unrealized_pnl_dollars(self) -> float:
        unrealized_dollars = 0.0
        for position in self.positions.values():
            entry_basis = self._position_entry_basis(position)
            if entry_basis <= 0:
                continue
            live_candidate = self._refresh_candidate_quote(position.candidate)
            if live_candidate is None:
                continue
            position.candidate = live_candidate
            close_value = self._combo_limit_price(live_candidate, side="SELL")
            unrealized_dollars += (close_value - entry_basis) * 100.0 * float(position.quantity)
        return round(unrealized_dollars, 2)

    def _retry_pending_session_closes(self) -> None:
        if self.cfg.hold_overnight:
            return
        if not self.positions or self.execution_halted_reason:
            return
        now_utc = _ensure_utc_timestamp(pd.Timestamp.utcnow())
        local_now = now_utc.tz_convert("America/New_York")
        if local_now.time() <= self.machine.end_time:
            return
        retry_delay_seconds = max(
            int(self.runner_cfg.poll_seconds),
            int(math.ceil(float(self.runner_cfg.combo_fill_wait_seconds) * max(1, int(self.runner_cfg.combo_max_chase_steps)))) + 5,
        )
        for position in list(self.positions.values()):
            last_attempt = position.close_requested_at
            if last_attempt is not None:
                elapsed = (now_utc - _ensure_utc_timestamp(last_attempt)).total_seconds()
                if elapsed < retry_delay_seconds:
                    continue
            action = ActionRecord(
                timestamp=now_utc,
                symbol=self.cfg.symbol,
                action=ActionType.SESSION_FLUSH,
                state=CorridorState.IDLE,
                price=float(self._latest_underlying_price() or 0.0),
                center_price=float(position.candidate.body_strike),
                layer_id=position.layer_id,
                detail="Retrying session flush close for an unfilled paper position.",
                metadata={"retry_close": True},
            )
            print(
                f"Action | {action.action.value} | layer={action.layer_id} | "
                f"price={action.price:.2f} | detail={action.detail}"
            )
            self._close_position(action)

    def _ensure_underlying_ticker(self) -> None:
        if self.market_ticker is not None:
            return
        self.market_ticker = self.ib.reqMktData(self.underlying_contract, "", False, False)
        self.ib.sleep(1.0)

    @staticmethod
    def _clean_number(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric) or math.isinf(numeric) or numeric < 0:
            return None
        return numeric

    def _extract_underlying_sample(self) -> Optional[tuple[pd.Timestamp, float, float]]:
        self._ensure_underlying_ticker()
        self.ib.sleep(0.25)

        ticker = self.market_ticker
        sample_time = getattr(ticker, "time", None)
        if sample_time is None:
            timestamp = _ensure_utc_timestamp(pd.Timestamp.utcnow())
        else:
            timestamp = _ensure_utc_timestamp(pd.Timestamp(sample_time))

        last = self._clean_number(getattr(ticker, "last", None))
        bid = self._clean_number(getattr(ticker, "bid", None))
        ask = self._clean_number(getattr(ticker, "ask", None))
        close = self._clean_number(getattr(ticker, "close", None))

        price: Optional[float] = None
        if last is not None and last > 0:
            price = last
        elif bid is not None and ask is not None and bid > 0 and ask > 0:
            price = (bid + ask) / 2.0
        elif close is not None and close > 0:
            price = close
        if price is None:
            self.warmup_quote_error = "Underlying bid/ask/last/close are unavailable from IB market data."
            return None
        self.warmup_quote_error = None

        volume_total = self._clean_number(getattr(ticker, "volume", None))
        last_size = self._clean_number(getattr(ticker, "lastSize", None)) or 0.0
        if volume_total is not None:
            if self.last_underlying_volume_total is None:
                volume_delta = 0.0
            else:
                volume_delta = max(0.0, volume_total - self.last_underlying_volume_total)
            self.last_underlying_volume_total = volume_total
        else:
            volume_delta = last_size

        return timestamp, float(price), float(volume_delta)

    def _bucket_start(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        ts = _ensure_utc_timestamp(timestamp)
        bucket_ns = self.bar_delta.value
        floored = (ts.value // bucket_ns) * bucket_ns
        return pd.Timestamp(floored, tz="UTC")

    def _start_partial_bar(self, bucket_start: pd.Timestamp, sample_time: pd.Timestamp, price: float, volume: float) -> None:
        self.partial_bar = InProgressBar(
            bucket_start=bucket_start,
            last_sample_time=sample_time,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=max(0.0, volume),
        )

    def _append_completed_bar(self, bar: InProgressBar, allow_orders: bool) -> None:
        row = {
            "timestamp": bar.bucket_start,
            "symbol": self.cfg.symbol,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        }
        self.history = (
            pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)
            .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
            .sort_values("timestamp")
            .tail(max(300, self.required_warmup_bars * 4))
            .reset_index(drop=True)
        )
        self.last_processed_ts = pd.Timestamp(row["timestamp"])
        self._process_bar(pd.Series(row), allow_orders=allow_orders)

        if not self.warmup_complete_announced and len(self.history) >= self.required_warmup_bars:
            self.warmup_complete_announced = True
            print(
                f"Warmup complete | completed_bars={len(self.history)} | "
                f"required_bars={self.required_warmup_bars} | model context is now ready."
            )

    def _poll_warmup_once(self, allow_orders: bool) -> None:
        sample = self._extract_underlying_sample()
        if sample is None:
            self._print_outage_notice(
                "Warmup | underlying quote unavailable."
            )
            self._refresh_positions_from_account()
            self._write_state_snapshot()
            return

        sample_time, price, volume_delta = sample
        bucket_start = self._bucket_start(sample_time)

        if self.partial_bar is None:
            self._start_partial_bar(bucket_start, sample_time, price, volume_delta)
            print(
                f"Warmup | started current bar bucket={bucket_start.isoformat()} | "
                f"completed_bars={len(self.history)}/{self.required_warmup_bars}"
            )
            self._refresh_positions_from_account()
            self._write_state_snapshot()
            return

        if bucket_start == self.partial_bar.bucket_start:
            self.partial_bar.last_sample_time = sample_time
            self.partial_bar.high = max(self.partial_bar.high, price)
            self.partial_bar.low = min(self.partial_bar.low, price)
            self.partial_bar.close = price
            self.partial_bar.volume += max(0.0, volume_delta)
            print(
                f"Warmup | collecting current bar bucket={bucket_start.isoformat()} | "
                f"completed_bars={len(self.history)}/{self.required_warmup_bars}"
            )
            self._refresh_positions_from_account()
            self._write_state_snapshot()
            return

        completed = self.partial_bar
        self._append_completed_bar(completed, allow_orders=allow_orders)
        self._start_partial_bar(bucket_start, sample_time, price, volume_delta)
        print(
            f"Warmup | completed_bar={completed.bucket_start.isoformat()} | "
            f"history_bars={len(self.history)}/{self.required_warmup_bars}"
        )
        self._refresh_positions_from_account()
        self._write_state_snapshot()

    def print_snapshot(self, label: str, persist: bool = False) -> None:
        payload = self._build_state_snapshot()
        if persist:
            self.logger.write_state(payload)
        latest_ts = payload["timestamp"]
        candidates = payload.get("candidates", [])
        price = payload.get("price")
        price_label = f"{price:.2f}" if isinstance(price, (int, float)) else "n/a"
        print(
            f"{label} | ts={latest_ts} | price={price_label} | "
            f"regime={payload['regime']} | center={payload['center']} | actual_tolerance={payload.get('actual_tolerance')} | "
            f"state={payload['state']} | open_positions={len(self.positions)} | "
            f"candidates={len(candidates)} | warmup_mode={payload['warmup_mode']} | "
            f"history_bars={payload['history_bars']}/{payload['warmup_required_bars']} | "
            f"startup_mode={payload['startup_mode']}"
        )
        print(
            "History | "
            f"history_seeded={payload['history_seeded']} | "
            f"status={payload['history_seed_status']}"
        )
        if payload.get("execution_halted_reason"):
            print(f"Execution Halted | {payload['execution_halted_reason']}")
        if payload.get("candidate_status"):
            print(f"Candidates | {payload['candidate_status']}")
        if payload.get("candidate_error"):
            print(f"Candidate Error | {payload['candidate_error']}")
        if payload.get("candidate_diagnostics"):
            print(f"Candidate Diagnostics | {json.dumps(payload['candidate_diagnostics'], sort_keys=True)}")
        for idx, candidate in enumerate(candidates[:3], start=1):
            print(
                f"Candidate {idx} | exp={candidate['expiry']} | "
                f"{candidate['lower_strike']}/{candidate['body_strike']}/{candidate['upper_strike']} {candidate['right']} | "
                f"wing={candidate.get('wing_mode', 'symmetric')} | "
                f"debit={candidate['net_debit']:.2f} | spread={candidate['total_spread']:.2f} | "
                f"spread_ratio={candidate['spread_ratio']:.2f} | rr={candidate['reward_to_risk']:.2f} | "
                f"max_reward={candidate['max_reward']:.2f}"
            )

    def _process_bar(self, row: pd.Series, allow_orders: bool, emit_logs: bool = True) -> None:
        timestamp = pd.Timestamp(row["timestamp"])
        price = float(row["close"])
        regime = self.detector.evaluate(self.history)
        center = self.estimator.estimate(self.history)
        bar_open = float(row["open"]) if "open" in row and pd.notna(row["open"]) else None
        self.machine.sync_session_context(timestamp, price, bar_open)
        protective_exit = self._protective_exit_signal(timestamp, price)
        if protective_exit is not None:
            action_type, detail, extra_metadata = protective_exit
            step = self.machine.flatten_positions(
                self.cfg.symbol,
                timestamp,
                price,
                action_type,
                detail,
                regime,
                extra_metadata=extra_metadata,
            )
        else:
            step = self.machine.process_bar(self.cfg.symbol, timestamp, price, regime, center, bar_open=bar_open)

        for transition in step.transitions:
            if emit_logs:
                self.logger.write_transition(
                    {
                        "timestamp": transition.timestamp.isoformat(),
                        "symbol": transition.symbol,
                        "from_state": transition.from_state.value,
                        "to_state": transition.to_state.value,
                        "reason": transition.reason,
                        "regime": transition.regime.value,
                        "price": round(transition.price, 4),
                        "center_price": transition.center_price,
                        "drift_count": transition.drift_count,
                        "layer_count": transition.layer_count,
                    }
                )
                print(
                    f"Transition | {transition.from_state.value} -> {transition.to_state.value} | "
                    f"price={transition.price:.2f} | reason={transition.reason}"
                )

        for action in step.actions:
            if emit_logs:
                action_row = {
                    "timestamp": action.timestamp.isoformat(),
                    "symbol": action.symbol,
                    "action": action.action.value,
                    "state": action.state.value,
                    "price": round(action.price, 4),
                    "center_price": action.center_price,
                    "layer_id": action.layer_id,
                    "detail": action.detail,
                    **action.metadata,
                }
                self.logger.write_action(action_row)
                print(
                    f"Action | {action.action.value} | layer={action.layer_id} | "
                    f"price={action.price:.2f} | detail={action.detail}"
                )
            if allow_orders:
                self._handle_action(action, center, regime)

    def _handle_action(self, action: ActionRecord, center, regime) -> None:
        if action.action in {ActionType.DRIFT_STARTED, ActionType.DRIFT_RESOLVED, ActionType.REBUILD_REQUESTED}:
            return

        if action.action in {
            ActionType.SESSION_FLUSH,
            ActionType.ABORTED,
            ActionType.STOP_LOSS,
            ActionType.TAKE_PROFIT,
            ActionType.MAX_HOLD,
        }:
            if action.layer_id is not None:
                self._close_position(action)
            return

        if action.action == ActionType.REBUILT:
            if action.detail.startswith("Established"):
                self._open_position(action, center, regime)
            else:
                if action.layer_id is not None:
                    self._close_position(action)
            return

        if action.action in {ActionType.ENTER_PRIMARY, ActionType.ADD_SUPPLEMENTAL}:
            self._open_position(action, center, regime)

    def _protective_exit_signal(
        self,
        timestamp: pd.Timestamp,
        price: float,
    ) -> tuple[ActionType, str, dict[str, float | str]] | None:
        if not self.positions:
            return None

        primary = self._primary_position()
        if primary is None:
            return None

        if self.cfg.primary_stop_loss_pct > 0 or self.cfg.primary_take_profit_pct > 0:
            return_pct = self._position_return_pct(primary)
            if return_pct is not None:
                if self.cfg.primary_stop_loss_pct > 0 and return_pct <= -self.cfg.primary_stop_loss_pct:
                    return (
                        ActionType.STOP_LOSS,
                        "Primary stop-loss reached.",
                        {"primary_return_pct": round(return_pct, 4)},
                    )
                if self.cfg.primary_take_profit_pct > 0 and return_pct >= self.cfg.primary_take_profit_pct:
                    return (
                        ActionType.TAKE_PROFIT,
                        "Primary take-profit reached.",
                        {"primary_return_pct": round(return_pct, 4)},
                    )
        if self.cfg.close_when_dte_lte > 0:
            remaining_dte = self._remaining_dte_calendar_days(primary, timestamp)
            if remaining_dte is not None and remaining_dte <= self.cfg.close_when_dte_lte:
                return (
                    ActionType.MAX_HOLD,
                    f"Max-hold DTE threshold reached (remaining_dte={remaining_dte}).",
                    {"remaining_dte": int(remaining_dte)},
                )
        if self.cfg.max_hold_sessions > 0:
            sessions_held = self._held_session_count(primary, timestamp)
            if sessions_held >= self.cfg.max_hold_sessions:
                return (
                    ActionType.MAX_HOLD,
                    f"Max-hold session threshold reached (sessions_held={sessions_held}).",
                    {"sessions_held": int(sessions_held)},
                )
        return None

    def _remaining_dte_calendar_days(self, position: ManagedPosition, timestamp: pd.Timestamp) -> Optional[int]:
        expiry = pd.to_datetime(str(position.candidate.expiry), format="%Y%m%d", errors="coerce")
        if pd.isna(expiry):
            return None
        local_ts = _ensure_utc_timestamp(pd.Timestamp(timestamp)).tz_convert("America/New_York")
        return int((expiry.date() - local_ts.date()).days)

    def _held_session_count(self, position: ManagedPosition, timestamp: pd.Timestamp) -> int:
        opened_local = _ensure_utc_timestamp(position.opened_at).tz_convert("America/New_York").date()
        current_local = _ensure_utc_timestamp(pd.Timestamp(timestamp)).tz_convert("America/New_York").date()
        if current_local < opened_local:
            return 0
        if self.history.empty:
            return 1
        local_dates = pd.to_datetime(self.history["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date
        mask = (local_dates >= opened_local) & (local_dates <= current_local)
        unique_sessions = int(local_dates[mask].nunique())
        return unique_sessions if unique_sessions > 0 else 1

    def _position_return_pct(self, position: ManagedPosition) -> Optional[float]:
        live_candidate = self._refresh_candidate_quote(position.candidate)
        if live_candidate is None:
            return None
        position.candidate = live_candidate

        entry_basis = self._position_entry_basis(position)
        if entry_basis <= 0:
            return None

        close_value = self._combo_limit_price(live_candidate, side="SELL")
        return (close_value - entry_basis) / entry_basis

    @staticmethod
    def _position_entry_basis(position: ManagedPosition) -> float:
        if position.open_fill_price is not None and position.open_fill_price > 0:
            return float(position.open_fill_price)
        if position.open_limit > 0:
            return float(position.open_limit)
        if position.candidate.net_debit > 0:
            return float(position.candidate.net_debit)
        return 0.0

    def _primary_position(self) -> Optional[ManagedPosition]:
        for position in self.positions.values():
            if position.layer_kind == LayerKind.PRIMARY.value:
                return position
        return next(iter(self.positions.values()), None)

    def _open_position(self, action: ActionRecord, center, regime) -> None:
        if action.layer_id is None or action.layer_id in self.positions:
            return
        if self.execution_halted_reason:
            self._rollback_unfilled_open(action.layer_id)
            self._log_order(
                {
                    "timestamp": action.timestamp.isoformat(),
                    "layer_id": action.layer_id,
                    "symbol": self.cfg.symbol,
                    "side": "OPEN",
                    "mode": "paper" if self.runner_cfg.paper_execution else "dry-run",
                    "status": "blocked",
                    "reason": self.execution_halted_reason,
                }
            )
            return
        if center is None or regime is None or regime.regime != Regime.RANGE:
            return

        target_body = float(action.metadata.get("body_strike") or action.metadata.get("center_price") or action.center_price or 0.0)
        target_dte = int(action.metadata.get("configured_target_dte") or self.cfg.default_dte)
        candidate = self._select_candidate(target_body, target_dte=target_dte, reference_ts=action.timestamp)
        if candidate is None:
            self._rollback_unfilled_open(action.layer_id)
            self._log_order(
                {
                    "timestamp": action.timestamp.isoformat(),
                    "layer_id": action.layer_id,
                    "symbol": self.cfg.symbol,
                    "side": "OPEN",
                    "mode": "paper" if self.runner_cfg.paper_execution else "dry-run",
                    "status": "skipped",
                    "reason": "No candidate butterfly matched the corridor center.",
                    "target_dte": target_dte,
                }
            )
            return
        candidate_issue = self._candidate_execution_issue(candidate)
        if candidate_issue is not None:
            self._rollback_unfilled_open(action.layer_id)
            self._log_order(
                {
                    "timestamp": action.timestamp.isoformat(),
                    "layer_id": action.layer_id,
                    "symbol": self.cfg.symbol,
                    "side": "OPEN",
                    "mode": "paper" if self.runner_cfg.paper_execution else "dry-run",
                    "status": "skipped",
                    "reason": candidate_issue,
                    "target_dte": target_dte,
                    "calendar_dte": candidate.calendar_dte,
                    "expiry": candidate.expiry,
                    "right": candidate.right,
                    "lower_strike": candidate.lower_strike,
                    "body_strike": candidate.body_strike,
                    "upper_strike": candidate.upper_strike,
                    **self._order_pricing_fields("OPEN", candidate, None, None),
                    "net_debit": round(candidate.net_debit, 4),
                    "total_spread": round(candidate.total_spread, 4),
                }
            )
            return

        limit_price = self._combo_limit_price(candidate, side="BUY")
        status = "dry_run"
        order_id = None
        open_fill_price = None
        failure_reason = ""
        if self.runner_cfg.paper_execution:
            trade = self._place_combo_order(candidate, "BUY", limit_price)
            status = trade.orderStatus.status or "submitted"
            order_id = getattr(trade.order, "orderId", None)
            fill_audit = getattr(trade, "fillAudit", None)
            if self._trade_was_rejected(trade):
                failure_reason = self._describe_trade_failure(trade)
                if fill_audit and fill_audit.get("abort_reason"):
                    failure_reason = str(fill_audit["abort_reason"])
                self._rollback_unfilled_open(action.layer_id)
                self._log_order(
                    {
                        "timestamp": action.timestamp.isoformat(),
                        "layer_id": action.layer_id,
                        "symbol": self.cfg.symbol,
                        "side": "OPEN",
                        "mode": "paper",
                        "status": status,
                        "order_id": order_id,
                        "quantity": self.runner_cfg.quantity,
                        "target_dte": target_dte,
                        "calendar_dte": candidate.calendar_dte,
                        "expiry": candidate.expiry,
                        "trading_class": candidate.trading_class,
                        "right": candidate.right,
                        "lower_strike": candidate.lower_strike,
                        "body_strike": candidate.body_strike,
                        "upper_strike": candidate.upper_strike,
                        "limit_price": round(limit_price, 2),
                        "fill_price": self._trade_fill_price(trade),
                        "fill_audit": json.dumps(fill_audit, sort_keys=True) if fill_audit else None,
                        **self._order_pricing_fields("OPEN", candidate, limit_price, self._trade_fill_price(trade)),
                        "net_debit": round(candidate.net_debit, 4),
                        "total_spread": round(candidate.total_spread, 4),
                        "reason": failure_reason,
                    }
                )
                if not self._is_benign_trade_abort(trade, fill_audit, failure_reason):
                    self._halt_execution(
                        f"Paper order rejected while opening layer {action.layer_id}: {failure_reason}"
                    )
                return
            if not self._trade_is_filled(trade):
                failure_reason = self._describe_trade_failure(trade)
                if fill_audit and fill_audit.get("abort_reason"):
                    failure_reason = str(fill_audit["abort_reason"])
                self._rollback_unfilled_open(action.layer_id)
                self._log_order(
                    {
                        "timestamp": action.timestamp.isoformat(),
                        "layer_id": action.layer_id,
                        "symbol": self.cfg.symbol,
                        "side": "OPEN",
                        "mode": "paper",
                        "status": status,
                        "order_id": order_id,
                        "quantity": self.runner_cfg.quantity,
                        "target_dte": target_dte,
                        "calendar_dte": candidate.calendar_dte,
                        "expiry": candidate.expiry,
                        "trading_class": candidate.trading_class,
                        "right": candidate.right,
                        "lower_strike": candidate.lower_strike,
                        "body_strike": candidate.body_strike,
                        "upper_strike": candidate.upper_strike,
                        "limit_price": round(limit_price, 2),
                        "fill_price": self._trade_fill_price(trade),
                        "fill_audit": json.dumps(fill_audit, sort_keys=True) if fill_audit else None,
                        **self._order_pricing_fields("OPEN", candidate, limit_price, self._trade_fill_price(trade)),
                        "net_debit": round(candidate.net_debit, 4),
                        "total_spread": round(candidate.total_spread, 4),
                        "reason": failure_reason or "Combo order did not fill within the chase window.",
                    }
                )
                return
            open_fill_price = self._trade_fill_price(trade)
        else:
            fill_audit = None

        entry_leg_prices = self._capture_entry_leg_prices(candidate)
        self.positions[action.layer_id] = ManagedPosition(
            layer_id=action.layer_id,
            candidate=candidate,
            quantity=self.runner_cfg.quantity,
            opened_at=action.timestamp,
            open_limit=limit_price,
            open_status=status,
            source_action=action.action.value,
            open_fill_price=open_fill_price,
            entry_leg_prices=entry_leg_prices,
            layer_kind=str(action.metadata.get("kind", LayerKind.PRIMARY.value)),
            order_id=order_id,
        )
        self._log_order(
            {
                "timestamp": action.timestamp.isoformat(),
                "layer_id": action.layer_id,
                "symbol": self.cfg.symbol,
                "side": "OPEN",
                "mode": "paper" if self.runner_cfg.paper_execution else "dry-run",
                "status": status,
                "order_id": order_id,
                "quantity": self.runner_cfg.quantity,
                "target_dte": target_dte,
                "calendar_dte": candidate.calendar_dte,
                "expiry": candidate.expiry,
                "right": candidate.right,
                "lower_strike": candidate.lower_strike,
                "body_strike": candidate.body_strike,
                "upper_strike": candidate.upper_strike,
                "limit_price": round(limit_price, 2),
                "fill_price": open_fill_price,
                "fill_audit": json.dumps(fill_audit, sort_keys=True) if fill_audit else None,
                **self._order_pricing_fields("OPEN", candidate, limit_price, open_fill_price),
                "net_debit": round(candidate.net_debit, 4),
                "total_spread": round(candidate.total_spread, 4),
                "reason": action.detail,
            }
        )
        self._maybe_send_open_discord_alert(
            action,
            status=status,
            order_id=order_id,
            limit_price=limit_price,
            open_fill_price=open_fill_price,
            fill_audit=fill_audit,
        )

    def _close_position(self, action: ActionRecord) -> None:
        position = self.positions.get(action.layer_id or -1)
        if position is None:
            self._log_order(
                {
                    "timestamp": action.timestamp.isoformat(),
                    "layer_id": action.layer_id,
                    "symbol": self.cfg.symbol,
                    "side": "CLOSE",
                    "mode": "paper" if self.runner_cfg.paper_execution else "dry-run",
                    "status": "skipped",
                    "reason": "No tracked position for the closing action.",
                }
            )
            return
        if self.execution_halted_reason:
            self._log_order(
                {
                    "timestamp": action.timestamp.isoformat(),
                    "layer_id": action.layer_id,
                    "symbol": self.cfg.symbol,
                    "side": "CLOSE",
                    "mode": "paper" if self.runner_cfg.paper_execution else "dry-run",
                    "status": "blocked",
                    "reason": self.execution_halted_reason,
                }
            )
            return

        live_candidate = self._refresh_candidate_quote(position.candidate) or position.candidate
        candidate_issue = self._candidate_execution_issue(live_candidate)
        if candidate_issue is not None:
            self._log_order(
                {
                    "timestamp": action.timestamp.isoformat(),
                    "layer_id": action.layer_id,
                    "symbol": self.cfg.symbol,
                    "side": "CLOSE",
                    "mode": "paper" if self.runner_cfg.paper_execution else "dry-run",
                    "status": "blocked",
                    "reason": candidate_issue,
                    "expiry": live_candidate.expiry,
                    "right": live_candidate.right,
                    "lower_strike": live_candidate.lower_strike,
                    "body_strike": live_candidate.body_strike,
                    "upper_strike": live_candidate.upper_strike,
                    **self._order_pricing_fields("CLOSE", live_candidate, None, None),
                    "net_debit": round(live_candidate.net_debit, 4),
                    "total_spread": round(live_candidate.total_spread, 4),
                }
            )
            self._halt_execution(f"Live close quote is not execution-safe for layer {action.layer_id}: {candidate_issue}")
            return

        limit_price = self._combo_limit_price(live_candidate, side="SELL")
        status = "dry_run"
        order_id = None
        close_fill_price = None
        failure_reason = ""
        if self.runner_cfg.paper_execution:
            trade = self._place_combo_order(live_candidate, "SELL", limit_price)
            status = trade.orderStatus.status or "submitted"
            order_id = getattr(trade.order, "orderId", None)
            fill_audit = getattr(trade, "fillAudit", None)
            if self._trade_was_rejected(trade):
                failure_reason = self._describe_trade_failure(trade)
            elif not self._trade_is_filled(trade):
                failure_reason = self._describe_trade_failure(trade) or "Combo close did not fill within the chase window."
            else:
                close_fill_price = self._trade_fill_price(trade)
            if failure_reason and fill_audit and fill_audit.get("abort_reason"):
                failure_reason = str(fill_audit["abort_reason"])
        else:
            fill_audit = None

        position.close_requested_at = action.timestamp
        position.close_limit = limit_price
        position.close_fill_price = close_fill_price
        position.close_status = status
        position.close_order_id = order_id
        position.close_failure_reason = failure_reason
        position.candidate = live_candidate
        self._log_order(
            {
                "timestamp": action.timestamp.isoformat(),
                "layer_id": action.layer_id,
                "symbol": self.cfg.symbol,
                "side": "CLOSE",
                "mode": "paper" if self.runner_cfg.paper_execution else "dry-run",
                "status": status,
                "order_id": order_id,
                "quantity": position.quantity,
                "expiry": live_candidate.expiry,
                "right": live_candidate.right,
                "lower_strike": live_candidate.lower_strike,
                "body_strike": live_candidate.body_strike,
                "upper_strike": live_candidate.upper_strike,
                "limit_price": round(limit_price, 2),
                "fill_price": close_fill_price,
                "fill_audit": json.dumps(fill_audit, sort_keys=True) if fill_audit else None,
                **self._order_pricing_fields("CLOSE", live_candidate, limit_price, close_fill_price),
                "net_debit": round(live_candidate.net_debit, 4),
                "total_spread": round(live_candidate.total_spread, 4),
                "reason": failure_reason or action.detail,
            }
        )
        if failure_reason:
            if not self._is_benign_trade_abort(trade, fill_audit, failure_reason):
                self._halt_execution(f"Paper close failed for layer {action.layer_id}: {failure_reason}")
            return
        if status.lower() == "filled":
            position.closed_at = action.timestamp
            del self.positions[position.layer_id]

    def _maybe_send_open_discord_alert(
        self,
        action: ActionRecord,
        *,
        status: str,
        order_id: Optional[int],
        limit_price: float,
        open_fill_price: Optional[float],
        fill_audit: Optional[dict[str, Any]],
    ) -> None:
        if not self.runner_cfg.paper_execution:
            return
        if str(status).lower() != "filled":
            return
        if open_fill_price is None:
            return
        if action.layer_id is None or action.layer_id not in self.positions:
            return
        if not self.discord_webhook_url:
            return

        position = self.positions[action.layer_id]
        candidate = position.candidate
        payload: dict[str, Any] = {
            "symbol": candidate.symbol,
            "expiry": candidate.expiry,
            "lower_strike": candidate.lower_strike,
            "body_strike": candidate.body_strike,
            "upper_strike": candidate.upper_strike,
            "wing_mode": candidate.wing_mode,
            "net_debit": round(float(candidate.net_debit), 4),
            "max_risk": round(float(candidate.max_risk), 2),
            "max_reward": round(float(candidate.max_reward), 2),
            "timestamp": position.opened_at.isoformat(),
            "status": status,
            "layer_id": position.layer_id,
            "quantity": position.quantity,
            "order_id": order_id,
            "limit_price": round(float(limit_price), 2),
            "fill_price": round(float(open_fill_price), 4),
            "mode": "paper",
        }
        if fill_audit is not None:
            payload["fill_audit"] = fill_audit

        send_discord_json_alert(self.discord_webhook_url, payload)

    def _select_candidate(
        self,
        target_body: float,
        *,
        target_dte: Optional[int] = None,
        reference_ts: Optional[pd.Timestamp] = None,
    ) -> Optional[ButterflyCandidate]:
        candidates = self._load_candidates(target_body, target_dte=target_dte, reference_ts=reference_ts)
        if not candidates:
            return None
        if self.cfg.wing_mode != "adaptive":
            chosen = self._best_candidate(candidates, target_body, target_dte=target_dte)
            self._record_selected_wing(chosen)
            return chosen

        symmetric = [candidate for candidate in candidates if candidate.wing_mode == "symmetric"]
        broken = [candidate for candidate in candidates if candidate.wing_mode != "symmetric"]
        best_symmetric = self._best_candidate(symmetric, target_body, target_dte=target_dte)
        symmetric_issue = self._candidate_execution_issue(best_symmetric) if best_symmetric is not None else None
        if best_symmetric is not None and symmetric_issue is None:
            self._record_selected_wing(best_symmetric)
            return best_symmetric
        if best_symmetric is not None and symmetric_issue is not None and "spread/debit ratio" in symmetric_issue:
            self.wing_stats["guard_fails"] += 1
        safe_broken = [candidate for candidate in broken if self._candidate_execution_issue(candidate) is None]
        if safe_broken:
            chosen = self._best_candidate(safe_broken, target_body, target_dte=target_dte)
            self._record_selected_wing(chosen)
            return chosen
        chosen = best_symmetric or self._best_candidate(candidates, target_body, target_dte=target_dte)
        self._record_selected_wing(chosen)
        return chosen

    def _record_selected_wing(self, candidate: Optional[ButterflyCandidate]) -> None:
        if candidate is None:
            return
        wing_mode = str(candidate.wing_mode or "symmetric")
        if wing_mode not in self.wing_stats:
            return
        self.wing_stats[wing_mode] += 1

    @staticmethod
    def _best_candidate(
        candidates: list[ButterflyCandidate],
        target_body: float,
        *,
        target_dte: Optional[int] = None,
    ) -> Optional[ButterflyCandidate]:
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda candidate: (
                abs((candidate.calendar_dte if candidate.calendar_dte is not None else target_dte or 0) - int(target_dte or 0)),
                abs(candidate.body_strike - target_body),
                0 if candidate.wing_mode == "symmetric" else 1,
                candidate.spread_ratio,
                candidate.total_spread,
                candidate.net_debit,
            ),
        )

    def _load_candidates(
        self,
        target_body: float,
        *,
        target_dte: Optional[int] = None,
        reference_ts: Optional[pd.Timestamp] = None,
    ) -> list[ButterflyCandidate]:
        candidates, diagnostics = self._load_candidates_with_diagnostics(
            target_body,
            target_dte=target_dte,
            reference_ts=reference_ts,
        )
        self.latest_candidate_diagnostics = diagnostics
        return candidates

    def _load_candidates_with_diagnostics(
        self,
        target_body: float,
        *,
        target_dte: Optional[int] = None,
        reference_ts: Optional[pd.Timestamp] = None,
    ) -> tuple[list[ButterflyCandidate], dict[str, Any]]:
        loader = IBOptionChainLoader(
            self.runner_cfg.host,
            self.runner_cfg.port,
            self.runner_cfg.client_id + self.runner_cfg.chain_client_id_offset,
            self.cfg.ib_exchange,
            self.cfg.ib_currency,
        )
        quotes = loader.load_candidates(
            self.cfg.symbol,
            target_body,
            self.cfg.butterfly_width,
            self.cfg.broken_wing_extra_width,
            self.cfg.wing_mode,
            self.cfg.dte_min,
            self.cfg.dte_max,
            body_search_steps=self.cfg.candidate_body_search_steps,
            center_rounding=self.cfg.center_rounding,
            market_data_type=self.market_data_type,
        )
        reference_date = None
        if reference_ts is not None:
            reference_date = _ensure_utc_timestamp(pd.Timestamp(reference_ts)).tz_convert("America/New_York").date()
        candidates, diagnostics = select_butterflies_with_diagnostics(
            quotes,
            target_body,
            self.cfg.butterfly_width,
            self.cfg,
            reference_date=reference_date,
        )
        diagnostic_payload = {
            "target_body": round(float(target_body), 4),
            "target_dte": int(target_dte) if target_dte is not None else None,
            "available_quotes": int(diagnostics.available_quotes),
            "expiries_considered": int(diagnostics.expiries_considered),
            "call_bodies_considered": int(diagnostics.call_bodies_considered),
            "attempted_structures": int(diagnostics.attempted_structures),
            "rejection_counts": dict(sorted(diagnostics.rejection_counts.items())),
            "sample_rejections": diagnostics.sample_rejections,
        }
        return candidates, diagnostic_payload

    def _combo_limit_price(self, candidate: ButterflyCandidate, side: str) -> float:
        spread_cap = self._max_spread_cap_for_candidate(candidate)
        spread_buffer = min(max(0.01, candidate.total_spread * 0.25), max(0.02, spread_cap))
        if side == "BUY":
            return max(0.01, round(candidate.net_debit + spread_buffer, 2))
        return max(0.01, round(max(0.01, candidate.net_debit - spread_buffer), 2))

    def _place_combo_order(self, candidate: ButterflyCandidate, side: str, limit_price: float):
        return self._place_combo_order_with_chase(candidate, side, limit_price)

    def _place_combo_order_with_chase(self, candidate: ButterflyCandidate, side: str, limit_price: float):
        current_limit = limit_price
        last_trade = None
        initial_midpoint = round(float(candidate.net_debit), 4)
        max_total_debit_limit = None
        if side == "BUY":
            max_total_debit_limit = round(initial_midpoint * self.runner_cfg.combo_max_total_debit_ratio, 2)
        fill_audit: dict[str, Any] = {
            "initial_quote_midpoint": initial_midpoint,
            "max_total_debit_limit": max_total_debit_limit,
            "steps": [],
            "final_limit_offset_from_initial_mid": None,
            "final_fill_offset_from_initial_mid": None,
            "abort_reason": None,
        }
        for step in range(max(1, self.runner_cfg.combo_max_chase_steps)):
            trade = self._place_combo_order_once(candidate, side, current_limit)
            last_trade = trade
            status = str(getattr(trade.orderStatus, "status", "") or "")
            fill_price = self._trade_fill_price(trade)
            fill_audit["steps"].append(
                {
                    "step": step + 1,
                    "limit_price": round(current_limit, 2),
                    "status": status,
                    "fill_price": round(fill_price, 4) if fill_price is not None else None,
                    "filled_any": self._trade_has_any_fill(trade),
                    "limit_offset_from_initial_mid": round(current_limit - initial_midpoint, 4),
                    "fill_offset_from_initial_mid": round(fill_price - initial_midpoint, 4) if fill_price is not None else None,
                }
            )
            if self._trade_was_rejected(trade) or self._trade_is_filled(trade):
                fill_audit["final_limit_offset_from_initial_mid"] = round(current_limit - initial_midpoint, 4)
                fill_audit["final_fill_offset_from_initial_mid"] = (
                    round(fill_price - initial_midpoint, 4) if fill_price is not None else None
                )
                setattr(trade, "fillAudit", fill_audit)
                return trade
            if status not in {"Submitted", "PreSubmitted", "PendingSubmit", "ApiPending"}:
                fill_audit["final_limit_offset_from_initial_mid"] = round(current_limit - initial_midpoint, 4)
                fill_audit["final_fill_offset_from_initial_mid"] = (
                    round(fill_price - initial_midpoint, 4) if fill_price is not None else None
                )
                setattr(trade, "fillAudit", fill_audit)
                return trade
            if not self._trade_has_any_fill(trade) and self._chase_should_abort_from_drift(candidate):
                fill_audit["abort_reason"] = "fill_timeout_abort_center_drift"
                try:
                    self.ib.cancelOrder(trade.order)
                    self.ib.sleep(0.5)
                except Exception:
                    pass
                fill_audit["final_limit_offset_from_initial_mid"] = round(current_limit - initial_midpoint, 4)
                fill_audit["final_fill_offset_from_initial_mid"] = (
                    round(fill_price - initial_midpoint, 4) if fill_price is not None else None
                )
                setattr(trade, "fillAudit", fill_audit)
                return trade
            try:
                self.ib.cancelOrder(trade.order)
                self.ib.sleep(0.5)
            except Exception:
                fill_audit["final_limit_offset_from_initial_mid"] = round(current_limit - initial_midpoint, 4)
                fill_audit["final_fill_offset_from_initial_mid"] = (
                    round(fill_price - initial_midpoint, 4) if fill_price is not None else None
                )
                setattr(trade, "fillAudit", fill_audit)
                return trade
            if step == self.runner_cfg.combo_max_chase_steps - 1:
                fill_audit["abort_reason"] = fill_audit["abort_reason"] or "chase_window_exhausted"
                fill_audit["final_limit_offset_from_initial_mid"] = round(current_limit - initial_midpoint, 4)
                fill_audit["final_fill_offset_from_initial_mid"] = (
                    round(fill_price - initial_midpoint, 4) if fill_price is not None else None
                )
                setattr(trade, "fillAudit", fill_audit)
                return trade
            next_limit = self._next_combo_limit(candidate, side, current_limit, max_total_debit_limit)
            if side == "BUY" and max_total_debit_limit is not None and next_limit >= max_total_debit_limit and current_limit >= max_total_debit_limit:
                fill_audit["abort_reason"] = "max_total_debit_limit_reached"
                fill_audit["final_limit_offset_from_initial_mid"] = round(current_limit - initial_midpoint, 4)
                fill_audit["final_fill_offset_from_initial_mid"] = (
                    round(fill_price - initial_midpoint, 4) if fill_price is not None else None
                )
                setattr(trade, "fillAudit", fill_audit)
                return trade
            current_limit = next_limit
        if last_trade is not None:
            setattr(last_trade, "fillAudit", fill_audit)
        return last_trade

    def _place_combo_order_once(self, candidate: ButterflyCandidate, side: str, limit_price: float):
        right = "C" if candidate.right == "CALL" else "P"
        lower = build_option_contract(
            symbol=self.cfg.symbol,
            expiry=candidate.expiry,
            strike=candidate.lower_strike,
            right=right,
            exchange=self.cfg.ib_exchange,
            currency=self.cfg.ib_currency,
            trading_class=candidate.trading_class,
        )
        body = build_option_contract(
            symbol=self.cfg.symbol,
            expiry=candidate.expiry,
            strike=candidate.body_strike,
            right=right,
            exchange=self.cfg.ib_exchange,
            currency=self.cfg.ib_currency,
            trading_class=candidate.trading_class,
        )
        upper = build_option_contract(
            symbol=self.cfg.symbol,
            expiry=candidate.expiry,
            strike=candidate.upper_strike,
            right=right,
            exchange=self.cfg.ib_exchange,
            currency=self.cfg.ib_currency,
            trading_class=candidate.trading_class,
        )
        qualified = self.ib.qualifyContracts(lower, body, upper)
        if len(qualified) != 3:
            raise RuntimeError("Unable to qualify option legs for combo order.")

        # Keep the BAG leg vector fixed for both open and close orders. IB applies the
        # parent order action to the combo, so reversing leg actions on a SELL can
        # point back at the original long butterfly instead of flattening it.
        leg_specs = self._long_butterfly_leg_specs(qualified)
        combo = build_butterfly_combo(self.cfg.symbol, self.cfg.ib_currency, self.cfg.ib_exchange, leg_specs)
        order = LimitOrder(side, self.runner_cfg.quantity, limit_price, tif=self.runner_cfg.order_tif)
        order.orderRef = f"corridor:{self.cfg.symbol}:{side}:{candidate.expiry}:{candidate.body_strike}"
        trade = self.ib.placeOrder(combo, order)
        self.ib.sleep(max(0.2, float(self.runner_cfg.combo_fill_wait_seconds)))
        return trade

    def _next_combo_limit(
        self,
        candidate: ButterflyCandidate,
        side: str,
        current_limit: float,
        max_total_debit_limit: Optional[float],
    ) -> float:
        chase_step = max(0.01, round(candidate.total_spread * self.runner_cfg.combo_chase_fraction_of_spread, 2))
        # Let the chase cap expand with the configured aggressiveness instead of
        # always stopping at half the quoted spread.
        max_buffer_fraction = self._combo_chase_cap_fraction()
        max_buffer = max(0.02, round(candidate.total_spread * max_buffer_fraction, 2))
        if side == "BUY":
            cap = round(candidate.net_debit + max_buffer, 2)
            if max_total_debit_limit is not None:
                cap = min(cap, max_total_debit_limit)
            return round(min(cap, current_limit + chase_step), 2)
        floor = max(0.01, round(candidate.net_debit - max_buffer, 2))
        return round(max(floor, current_limit - chase_step), 2)

    def _combo_chase_cap_fraction(self) -> float:
        configured_fraction = max(0.0, float(self.runner_cfg.combo_chase_fraction_of_spread))
        configured_steps = max(1, int(self.runner_cfg.combo_max_chase_steps))
        requested_fraction = configured_fraction * configured_steps
        return min(1.0, max(0.5, requested_fraction))

    def _chase_should_abort_from_drift(self, candidate: ButterflyCandidate) -> bool:
        latest_price = self._latest_underlying_price()
        if latest_price is None:
            return False
        center = self.estimator.estimate(self.history) if not self.history.empty else None
        center_price = float(center.center_price) if center is not None else float(candidate.body_strike)
        tolerance = float(center.actual_tolerance) if center is not None else float(self.cfg.center_tolerance)
        return abs(float(latest_price) - center_price) > tolerance

    def _candidate_execution_issue(self, candidate: ButterflyCandidate) -> Optional[str]:
        if candidate.net_debit <= 0:
            return "Candidate net debit is non-positive."
        if candidate.total_spread <= 0:
            return "Candidate total spread is non-positive."
        allowed_spread = self._max_spread_cap_for_candidate(candidate)
        if candidate.total_spread > allowed_spread:
            return (
                f"Candidate total spread {candidate.total_spread:.2f} exceeds "
                f"max_option_spread={allowed_spread:.2f} for dte={candidate.calendar_dte if candidate.calendar_dte is not None else 'unknown'}."
            )
        spread_ratio = candidate.total_spread / max(candidate.net_debit, 0.01)
        if spread_ratio > self.runner_cfg.max_spread_pct_of_debit:
            return (
                f"Candidate spread/debit ratio {spread_ratio:.2f} exceeds "
                f"max_spread_pct_of_debit={self.runner_cfg.max_spread_pct_of_debit:.2f}."
            )
        return None

    def _max_spread_cap_for_candidate(self, candidate: ButterflyCandidate) -> float:
        return float(self.cfg.max_acceptable_option_spread_for_dte(candidate.calendar_dte))

    @staticmethod
    def _order_pricing_fields(
        side: str,
        candidate: ButterflyCandidate,
        limit_price: Optional[float],
        fill_price: Optional[float],
    ) -> dict[str, Any]:
        quote_reference = round(float(candidate.net_debit), 4)
        spread_ratio = round(float(candidate.total_spread) / max(float(candidate.net_debit), 0.01), 4)
        payload: dict[str, Any] = {
            "quote_reference": quote_reference,
            "spread_ratio": spread_ratio,
        }
        if limit_price is not None:
            payload["limit_vs_quote"] = round(
                (float(limit_price) - float(candidate.net_debit))
                if side == "OPEN"
                else (float(candidate.net_debit) - float(limit_price)),
                4,
            )
        else:
            payload["limit_vs_quote"] = None
        if fill_price is not None:
            payload["fill_edge_vs_quote"] = round(
                (float(candidate.net_debit) - float(fill_price))
                if side == "OPEN"
                else (float(fill_price) - float(candidate.net_debit)),
                4,
            )
            payload["fill_edge_vs_limit"] = (
                round(
                    (float(limit_price) - float(fill_price))
                    if side == "OPEN"
                    else (float(fill_price) - float(limit_price)),
                    4,
                )
                if limit_price is not None
                else None
            )
        else:
            payload["fill_edge_vs_quote"] = None
            payload["fill_edge_vs_limit"] = None
        return payload

    def _load_structure_quote_map(self, candidate: ButterflyCandidate) -> Optional[dict[float, OptionQuote]]:
        loader = IBOptionChainLoader(
            self.runner_cfg.host,
            self.runner_cfg.port,
            self.runner_cfg.client_id + self.runner_cfg.chain_client_id_offset,
            self.cfg.ib_exchange,
            self.cfg.ib_currency,
        )
        try:
            quotes = loader.load_structure_quotes(
                symbol=self.cfg.symbol,
                expiry=candidate.expiry,
                lower_strike=candidate.lower_strike,
                body_strike=candidate.body_strike,
                upper_strike=candidate.upper_strike,
                right=candidate.right,
                market_data_type=self.market_data_type,
                trading_class=candidate.trading_class,
            )
        except Exception:
            return None
        if len(quotes) != 3:
            return None
        by_strike = {float(quote.strike): quote for quote in quotes}
        lower = by_strike.get(float(candidate.lower_strike))
        body = by_strike.get(float(candidate.body_strike))
        upper = by_strike.get(float(candidate.upper_strike))
        if lower is None or body is None or upper is None:
            return None
        return by_strike

    @staticmethod
    def _candidate_from_structure_quote_map(
        candidate: ButterflyCandidate,
        by_strike: dict[float, OptionQuote],
    ) -> Optional[ButterflyCandidate]:
        lower = by_strike.get(float(candidate.lower_strike))
        body = by_strike.get(float(candidate.body_strike))
        upper = by_strike.get(float(candidate.upper_strike))
        if lower is None or body is None or upper is None:
            return None
        net_debit = lower.mid - (2.0 * body.mid) + upper.mid
        total_spread = lower.spread + (2.0 * body.spread) + upper.spread
        if net_debit <= 0:
            return None
        lower_width = float(candidate.body_strike) - float(candidate.lower_strike)
        upper_width = float(candidate.upper_strike) - float(candidate.body_strike)
        reward_width = min(lower_width, upper_width)
        extra_tail_risk = abs(upper_width - lower_width)
        return ButterflyCandidate(
            symbol=candidate.symbol,
            expiry=candidate.expiry,
            lower_strike=candidate.lower_strike,
            body_strike=candidate.body_strike,
            upper_strike=candidate.upper_strike,
            lower_width=lower_width,
            upper_width=upper_width,
            net_debit=net_debit,
            total_spread=total_spread,
            max_risk=max(0.0, (net_debit + extra_tail_risk) * 100.0),
            max_reward=max(0.0, (reward_width - net_debit) * 100.0),
            right=candidate.right,
            trading_class=candidate.trading_class,
            wing_mode=candidate.wing_mode,
            spread_ratio=total_spread / max(net_debit, 0.01),
            reward_to_risk=max(0.0, (reward_width - net_debit) * 100.0) / max(max(0.0, (net_debit + extra_tail_risk) * 100.0), 0.01),
            body_distance=candidate.body_distance,
        )

    def _refresh_candidate_quote(self, candidate: ButterflyCandidate) -> Optional[ButterflyCandidate]:
        quote_map = self._load_structure_quote_map(candidate)
        if quote_map is None:
            return None
        return self._candidate_from_structure_quote_map(candidate, quote_map)

    def _capture_entry_leg_prices(self, candidate: ButterflyCandidate) -> Optional[dict[str, float]]:
        if not self.ib.isConnected():
            return None
        quote_map = self._load_structure_quote_map(candidate)
        if quote_map is None:
            return None
        lower = quote_map.get(float(candidate.lower_strike))
        body = quote_map.get(float(candidate.body_strike))
        upper = quote_map.get(float(candidate.upper_strike))
        if lower is None or body is None or upper is None:
            return None
        if lower.mid <= 0 or body.mid <= 0 or upper.mid <= 0:
            return None
        return {
            "lower": round(float(lower.mid), 4),
            "body": round(float(body.mid), 4),
            "upper": round(float(upper.mid), 4),
        }

    def _guard_startup_account_state(self) -> None:
        if not self.runner_cfg.paper_execution or self.runner_cfg.check_only:
            return
        actual = self._account_option_position_counts()
        if not actual:
            return
        if self.runner_cfg.start_flat and not self.positions:
            raise RuntimeError(
                f"Existing {self.cfg.symbol} option positions are already open in the connected paper account. "
                "Flatten them before starting flat paper execution."
            )
        differences = self._diff_account_vs_tracked_positions(actual)
        if differences:
            raise RuntimeError(
                "Connected paper account positions do not match the runner's tracked corridor state. "
                f"Differences: {self._format_position_differences(differences)}"
            )

    def _refresh_positions_from_account(self) -> None:
        if not self.runner_cfg.paper_execution or not self.ib.isConnected():
            return
        try:
            actual = self._account_option_position_counts()
        except Exception as exc:
            print(f"Account Sync Warning | {exc}")
            return

        for layer_id, position in list(self.positions.items()):
            if position.close_order_id is None:
                continue
            if self._position_is_flat_in_account(position, actual):
                position.closed_at = _ensure_utc_timestamp(pd.Timestamp.utcnow())
                position.close_status = position.close_status or "FlatConfirmed"
                del self.positions[layer_id]

        differences = self._diff_account_vs_tracked_positions(actual)
        if differences:
            self._halt_execution(
                "Paper account option legs diverged from the runner state. "
                f"Differences: {self._format_position_differences(differences)}"
            )

    def _managed_position_to_active_layer(self, position: ManagedPosition) -> ActiveButterfly:
        try:
            kind = LayerKind(position.layer_kind)
        except ValueError:
            kind = LayerKind.PRIMARY
        lower_width = float(position.candidate.body_strike) - float(position.candidate.lower_strike)
        upper_width = float(position.candidate.upper_strike) - float(position.candidate.body_strike)
        width = min(lower_width, upper_width)
        expiry = pd.Timestamp(str(position.candidate.expiry))
        dte = max(1, int((expiry.date() - position.opened_at.date()).days))
        return ActiveButterfly(
            layer_id=position.layer_id,
            kind=kind,
            center_price=float(position.candidate.body_strike),
            width=width,
            lower_width=lower_width,
            upper_width=upper_width,
            lower_strike=float(position.candidate.lower_strike),
            body_strike=float(position.candidate.body_strike),
            upper_strike=float(position.candidate.upper_strike),
            created_at=position.opened_at,
            dte=dte,
            entry_debit=float(position.candidate.net_debit),
            entry_cost=float(position.candidate.net_debit),
            metadata={"adopted": True, "wing_mode": position.candidate.wing_mode},
        )

    @staticmethod
    def _primary_center_from_positions(positions: Any) -> float:
        ordered = sorted(
            positions,
            key=lambda item: (
                0 if item.layer_kind == LayerKind.PRIMARY.value else 1,
                item.layer_id,
            ),
        )
        return float(ordered[0].candidate.body_strike) if ordered else 0.0

    def _build_recovery_payload(self) -> dict[str, Any]:
        if self.positions:
            current_center = self._primary_center_from_positions(self.positions.values())
        else:
            current_center = self.machine.context.current_center
        return {
            "symbol": self.cfg.symbol,
            "saved_at": _ensure_utc_timestamp(pd.Timestamp.utcnow()).isoformat(),
            "state": self.machine.context.state.value,
            "current_center": current_center,
            "next_layer_id": self.machine.context.next_layer_id,
            "last_primary_entry_session_date": self.machine.context.last_primary_entry_session_date,
            "last_take_profit_session_date": self.machine.context.last_take_profit_session_date,
            "positions": [
                managed_position_to_payload(position)
                for position in sorted(self.positions.values(), key=lambda item: item.layer_id)
            ],
        }

    def _account_option_position_counts(self) -> Counter[tuple[str, float, str]]:
        counts: Counter[tuple[str, float, str]] = Counter()
        for position in self.ib.positions():
            contract = getattr(position, "contract", None)
            if contract is None:
                continue
            if getattr(contract, "symbol", "").upper() != self.cfg.symbol.upper():
                continue
            if getattr(contract, "secType", "") != "OPT":
                continue
            quantity = int(round(float(getattr(position, "position", 0.0) or 0.0)))
            if quantity == 0:
                continue
            counts[self._option_leg_key(contract)] += quantity
        return counts

    def _tracked_option_position_counts(
        self,
        exclude_layer_id: Optional[int] = None,
    ) -> Counter[tuple[str, float, str]]:
        counts: Counter[tuple[str, float, str]] = Counter()
        for layer_id, position in self.positions.items():
            if exclude_layer_id is not None and layer_id == exclude_layer_id:
                continue
            counts.update(self._candidate_leg_counts(position.candidate, position.quantity))
        return counts

    def _diff_account_vs_tracked_positions(
        self,
        actual: Optional[Counter[tuple[str, float, str]]] = None,
    ) -> dict[tuple[str, float, str], dict[str, int]]:
        actual_counts = actual if actual is not None else self._account_option_position_counts()
        tracked_counts = self._tracked_option_position_counts()
        differences: dict[tuple[str, float, str], dict[str, int]] = {}
        for key in sorted(set(actual_counts) | set(tracked_counts)):
            account_qty = int(actual_counts.get(key, 0))
            tracked_qty = int(tracked_counts.get(key, 0))
            if account_qty != tracked_qty:
                differences[key] = {"account": account_qty, "tracked": tracked_qty}
        return differences

    def _position_is_flat_in_account(
        self,
        position: ManagedPosition,
        actual_counts: Counter[tuple[str, float, str]],
    ) -> bool:
        other_counts = self._tracked_option_position_counts(exclude_layer_id=position.layer_id)
        for key in self._candidate_leg_counts(position.candidate, position.quantity):
            residual = int(actual_counts.get(key, 0)) - int(other_counts.get(key, 0))
            if residual != 0:
                return False
        return True

    @staticmethod
    def _option_leg_key(contract) -> tuple[str, float, str]:
        expiry = str(getattr(contract, "lastTradeDateOrContractMonth", ""))
        strike = float(getattr(contract, "strike", 0.0))
        right = str(getattr(contract, "right", "")).upper()
        return expiry, strike, right

    @staticmethod
    def _candidate_leg_counts(candidate: ButterflyCandidate, quantity: int) -> Counter[tuple[str, float, str]]:
        right = "C" if candidate.right == "CALL" else "P"
        counts: Counter[tuple[str, float, str]] = Counter()
        counts[(candidate.expiry, float(candidate.lower_strike), right)] += quantity
        counts[(candidate.expiry, float(candidate.body_strike), right)] -= 2 * quantity
        counts[(candidate.expiry, float(candidate.upper_strike), right)] += quantity
        return counts

    @staticmethod
    def _format_position_differences(
        differences: dict[tuple[str, float, str], dict[str, int]]
    ) -> str:
        parts: list[str] = []
        for (expiry, strike, right), values in sorted(differences.items()):
            parts.append(
                f"{expiry} {right} {strike:.1f} account={values['account']} tracked={values['tracked']}"
            )
        return "; ".join(parts)

    @staticmethod
    def _trade_was_rejected(trade) -> bool:
        status = str(getattr(trade.orderStatus, "status", "") or "").strip()
        return status in {"Cancelled", "ApiCancelled", "Inactive"}

    @staticmethod
    def _trade_is_filled(trade) -> bool:
        status = str(getattr(trade.orderStatus, "status", "") or "").strip()
        return status == "Filled"

    @staticmethod
    def _trade_has_any_fill(trade) -> bool:
        try:
            filled = float(getattr(trade.orderStatus, "filled", 0.0) or 0.0)
        except (TypeError, ValueError):
            filled = 0.0
        return filled > 0.0 or bool(getattr(trade, "fills", None))

    @staticmethod
    def _trade_fill_price(trade) -> Optional[float]:
        avg_fill = float(getattr(trade.orderStatus, "avgFillPrice", 0.0) or 0.0)
        if avg_fill > 0:
            return avg_fill
        fills = getattr(trade, "fills", None) or []
        if fills:
            execution = getattr(fills[-1], "execution", None)
            price = float(getattr(execution, "price", 0.0) or 0.0)
            if price > 0:
                return price
        return None

    @staticmethod
    def _describe_trade_failure(trade) -> str:
        for entry in reversed(getattr(trade, "log", []) or []):
            message = str(getattr(entry, "message", "") or "").strip()
            if message:
                return message
        advanced_error = str(getattr(trade, "advancedError", "") or "").strip()
        if advanced_error:
            return advanced_error
        status = str(getattr(trade.orderStatus, "status", "") or "").strip() or "unknown"
        return f"Order ended in status={status}."

    @staticmethod
    def _is_benign_trade_abort(trade, fill_audit: Optional[dict[str, Any]], failure_reason: str) -> bool:
        abort_reason = str((fill_audit or {}).get("abort_reason") or "").strip()
        if abort_reason in {
            "chase_window_exhausted",
            "fill_timeout_abort_center_drift",
            "max_total_debit_limit_reached",
        }:
            return True
        status = str(getattr(trade.orderStatus, "status", "") or "").strip()
        if status != "Cancelled":
            return False
        normalized_failure_reason = str(failure_reason or "").strip()
        if normalized_failure_reason and normalized_failure_reason != "Order ended in status=Cancelled.":
            if "needs to be cancelled is not found" not in normalized_failure_reason:
                return False
        for entry in reversed(getattr(trade, "log", []) or []):
            message = str(getattr(entry, "message", "") or "").strip()
            error_code = getattr(entry, "errorCode", 0) or 0
            try:
                error_code = int(error_code)
            except (TypeError, ValueError):
                return False
            if error_code not in {0, 10147}:
                return False
            if message and error_code != 10147:
                return False
            if message and error_code == 10147 and "needs to be cancelled is not found" not in message:
                    return False
        advanced_error = str(getattr(trade, "advancedError", "") or "").strip()
        return advanced_error == ""

    def _halt_execution(self, reason: str) -> None:
        if self.execution_halted_reason:
            return
        self.execution_halted_reason = reason
        print(f"Execution Halted | {reason}")

    def _print_outage_notice(self, message: str) -> None:
        if message == self.last_outage_notice:
            return
        self.last_outage_notice = message
        print(message)

    @staticmethod
    def _position_leg_definitions(position: ManagedPosition) -> list[dict[str, Any]]:
        return [
            {
                "name": "lower",
                "strike": float(position.candidate.lower_strike),
                "coefficient": 1,
                "signed_quantity": int(position.quantity),
            },
            {
                "name": "body",
                "strike": float(position.candidate.body_strike),
                "coefficient": -2,
                "signed_quantity": -2 * int(position.quantity),
            },
            {
                "name": "upper",
                "strike": float(position.candidate.upper_strike),
                "coefficient": 1,
                "signed_quantity": int(position.quantity),
            },
        ]

    @staticmethod
    def _normalize_option_average_cost(
        avg_cost: float,
        multiplier: float,
        reference_price: Optional[float],
    ) -> Optional[float]:
        raw = abs(float(avg_cost or 0.0))
        if raw <= 0:
            return None
        candidates = [raw]
        if multiplier > 1:
            candidates.append(raw / multiplier)
        valid = [value for value in candidates if value > 0]
        if not valid:
            return None
        if reference_price is not None and reference_price > 0:
            return min(valid, key=lambda value: abs(value - reference_price))
        if multiplier > 1 and raw >= multiplier:
            return raw / multiplier
        return raw

    def _account_option_average_costs(self) -> dict[tuple[str, float, str], tuple[float, float]]:
        if not self.runner_cfg.paper_execution or not self.ib.isConnected():
            return {}
        costs: dict[tuple[str, float, str], tuple[float, float]] = {}
        try:
            positions = self.ib.positions()
        except Exception:
            return costs
        for account_position in positions:
            contract = getattr(account_position, "contract", None)
            if contract is None:
                continue
            if getattr(contract, "symbol", "").upper() != self.cfg.symbol.upper():
                continue
            if getattr(contract, "secType", "") != "OPT":
                continue
            quantity = int(round(float(getattr(account_position, "position", 0.0) or 0.0)))
            if quantity == 0:
                continue
            avg_cost = float(getattr(account_position, "avgCost", 0.0) or 0.0)
            multiplier = float(getattr(contract, "multiplier", 100.0) or 100.0)
            costs[self._option_leg_key(contract)] = (avg_cost, multiplier)
        return costs

    def _build_discord_position_detail_lines(self) -> list[str]:
        if not self.positions:
            return []
        account_avg_costs = self._account_option_average_costs()
        ordered_positions = sorted(
            self.positions.values(),
            key=lambda item: (
                0 if item.layer_kind == LayerKind.PRIMARY.value else 1,
                item.layer_id,
            ),
        )
        lines: list[str] = []
        for index, position in enumerate(ordered_positions):
            lines.extend(self._build_discord_position_lines(position, account_avg_costs))
            if index < len(ordered_positions) - 1:
                lines.append("")
        return lines

    def _build_discord_position_lines(
        self,
        position: ManagedPosition,
        account_avg_costs: dict[tuple[str, float, str], tuple[float, float]],
    ) -> list[str]:
        quote_map = self._load_structure_quote_map(position.candidate)
        if quote_map is not None:
            live_candidate = self._candidate_from_structure_quote_map(position.candidate, quote_map)
            if live_candidate is not None:
                position.candidate = live_candidate

        right_code = "C" if position.candidate.right == "CALL" else "P"
        combo_current = None
        if quote_map is not None:
            lower = quote_map.get(float(position.candidate.lower_strike))
            body = quote_map.get(float(position.candidate.body_strike))
            upper = quote_map.get(float(position.candidate.upper_strike))
            if lower is not None and body is not None and upper is not None:
                combo_current = round(lower.mid - (2.0 * body.mid) + upper.mid, 4)

        combo_entry = None
        combo_pnl_dollars = None
        leg_lines: list[str] = []
        derived_combo_entry = 0.0
        derived_entry_complete = True
        for leg in self._position_leg_definitions(position):
            strike = float(leg["strike"])
            current_price = None
            if quote_map is not None:
                quote = quote_map.get(strike)
                if quote is not None and quote.mid > 0:
                    current_price = round(float(quote.mid), 4)
            key = (position.candidate.expiry, strike, right_code)
            entry_price = None
            account_cost = account_avg_costs.get(key)
            if account_cost is not None:
                entry_price = self._normalize_option_average_cost(
                    account_cost[0],
                    account_cost[1],
                    current_price,
                )
            if entry_price is None and position.entry_leg_prices:
                entry_price = _coerce_optional_float(position.entry_leg_prices.get(str(leg["name"])))
            if entry_price is not None:
                entry_price = round(float(entry_price), 4)
                derived_combo_entry += float(leg["coefficient"]) * entry_price
            else:
                derived_entry_complete = False
            pnl_dollars = None
            if entry_price is not None and current_price is not None:
                pnl_dollars = round(
                    (float(current_price) - float(entry_price))
                    * 100.0
                    * float(leg["signed_quantity"]),
                    2,
                )
            signed_quantity = int(leg["signed_quantity"])
            pnl_label = "n/a" if pnl_dollars is None else _format_signed_dollar(pnl_dollars)
            leg_lines.append(
                f"{signed_quantity:+d}x {strike:.1f} | entry={_format_optional_price(entry_price)}"
                f" | current={_format_optional_price(current_price)}"
                f" | pnl={pnl_label}"
            )

        if derived_entry_complete:
            combo_entry = round(derived_combo_entry, 4)
        else:
            fallback_entry = self._position_entry_basis(position)
            if fallback_entry > 0:
                combo_entry = round(float(fallback_entry), 4)
        if combo_entry is not None and combo_current is not None:
            combo_pnl_dollars = round(
                (float(combo_current) - float(combo_entry)) * 100.0 * float(position.quantity),
                2,
            )
        combo_pnl_label = "n/a" if combo_pnl_dollars is None else _format_signed_dollar(combo_pnl_dollars)
        header = (
            f"Position {position.layer_id} | {position.candidate.symbol} {position.candidate.expiry}"
            f" {position.candidate.right} | qty={position.quantity}"
            f" | strikes={position.candidate.lower_strike:.1f}/{position.candidate.body_strike:.1f}/{position.candidate.upper_strike:.1f}"
            f" | combo_entry={_format_optional_price(combo_entry)}"
            f" | combo_now={_format_optional_price(combo_current)}"
            f" | combo_pnl={combo_pnl_label}"
        )
        return [header, *leg_lines]

    def _maybe_send_discord_log_alert(self, message: str) -> None:
        if not self.discord_webhook_url:
            return
        detail_lines = self._build_discord_position_detail_lines()
        payload = message if not detail_lines else message + "\n" + "\n".join(detail_lines)
        send_discord_text_alert(self.discord_webhook_url, payload)

    def _long_butterfly_leg_specs(self, qualified_contracts) -> list[ComboLegSpec]:
        return [
            ComboLegSpec(con_id=qualified_contracts[0].conId, ratio=1, action="BUY", exchange=self.cfg.ib_exchange),
            ComboLegSpec(con_id=qualified_contracts[1].conId, ratio=2, action="SELL", exchange=self.cfg.ib_exchange),
            ComboLegSpec(con_id=qualified_contracts[2].conId, ratio=1, action="BUY", exchange=self.cfg.ib_exchange),
        ]

    def _latest_underlying_price(self) -> Optional[float]:
        self._ensure_underlying_ticker()
        self.ib.sleep(0.1)
        ticker = self.market_ticker
        last = self._clean_number(getattr(ticker, "last", None))
        bid = self._clean_number(getattr(ticker, "bid", None))
        ask = self._clean_number(getattr(ticker, "ask", None))
        close = self._clean_number(getattr(ticker, "close", None))
        if last is not None and last > 0:
            return float(last)
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return float((bid + ask) / 2.0)
        if close is not None and close > 0:
            return float(close)
        return None

    def _rollback_unfilled_open(self, layer_id: Optional[int]) -> None:
        if layer_id is None:
            return
        ctx = self.machine.context
        ctx.active_layers = [layer for layer in ctx.active_layers if layer.layer_id != layer_id]
        ctx.drift_count = 0
        if ctx.active_layers:
            ctx.state = CorridorState.ACTIVE_CENTERED
            ctx.current_center = float(ctx.active_layers[0].body_strike)
        else:
            ctx.state = CorridorState.IDLE
            ctx.current_center = None

    def _completed_bars(self, frame: pd.DataFrame) -> pd.DataFrame:
        now_utc = _ensure_utc_timestamp(pd.Timestamp.utcnow())
        out = frame.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out[out["timestamp"] + self.bar_delta <= now_utc]
        return out.sort_values("timestamp").reset_index(drop=True)

    def _write_state_snapshot(self) -> None:
        payload = self._build_state_snapshot()
        self.logger.write_state(payload)
        self.logger.write_recovery(self._build_recovery_payload())
        daily_report = self._build_daily_report(payload)
        self.logger.write_daily_report(daily_report)
        test_summary = build_paper_test_summary(payload, daily_report)
        self.logger.write_test_summary(test_summary, format_paper_test_summary(test_summary))

    def _build_daily_report(self, state_payload: dict[str, Any]) -> dict[str, Any]:
        now_utc = _ensure_utc_timestamp(pd.Timestamp.utcnow())
        report_date = now_utc.tz_convert("America/New_York").date().isoformat()
        adaptive_fallback_rate, fallback_type_distribution = self._adaptive_stats_payload()
        transitions_today = self._rows_for_local_date(self.logger.paths["transitions"], report_date)
        actions_today = self._rows_for_local_date(self.logger.paths["actions"], report_date)
        orders_today = self._rows_for_local_date(self.logger.paths["orders"], report_date)

        action_counts = Counter(row.get("action", "") for row in actions_today if row.get("action"))
        order_status_counts = Counter(row.get("status", "") for row in orders_today if row.get("status"))
        order_side_counts = Counter(row.get("side", "") for row in orders_today if row.get("side"))
        order_reason_counts = Counter(row.get("reason", "") for row in orders_today if row.get("reason"))
        skipped_orders_today = sum(1 for row in orders_today if row.get("status") == "skipped")
        execution_failure_orders_today = sum(
            1 for row in orders_today if row.get("status") in {"blocked", "Cancelled", "ApiCancelled", "Inactive"}
        )

        open_orders_submitted = sum(
            1 for row in orders_today if row.get("side") == "OPEN" and row.get("status") in {"Filled", "Submitted", "PreSubmitted"}
        )
        close_orders_submitted = sum(
            1 for row in orders_today if row.get("side") == "CLOSE" and row.get("status") in {"Filled", "Submitted", "PreSubmitted"}
        )
        filled_open_orders = [row for row in orders_today if row.get("side") == "OPEN" and row.get("status") == "Filled"]
        filled_close_orders = [row for row in orders_today if row.get("side") == "CLOSE" and row.get("status") == "Filled"]

        return {
            "report_timestamp": now_utc.isoformat(),
            "report_date": report_date,
            "symbol": self.cfg.symbol,
            "configured_wing_mode": state_payload.get("configured_wing_mode"),
            "execution_mode": state_payload.get("execution_mode"),
            "startup_mode": state_payload.get("startup_mode"),
            "history_seeded": state_payload.get("history_seeded"),
            "history_seed_status": state_payload.get("history_seed_status"),
            "model_ready": state_payload.get("model_ready"),
            "warmup_mode": state_payload.get("warmup_mode"),
            "execution_halted_reason": state_payload.get("execution_halted_reason"),
            "latest_timestamp": state_payload.get("timestamp"),
            "latest_price": state_payload.get("price"),
            "latest_regime": state_payload.get("regime"),
            "latest_center": state_payload.get("center"),
            "latest_actual_tolerance": state_payload.get("actual_tolerance"),
            "latest_state": state_payload.get("state"),
            "history_bars": state_payload.get("history_bars"),
            "warmup_remaining_bars": state_payload.get("warmup_remaining_bars"),
            "candidate_count": len(state_payload.get("candidates", [])),
            "candidate_status": state_payload.get("candidate_status"),
            "candidate_error": state_payload.get("candidate_error"),
            "candidate_diagnostics": state_payload.get("candidate_diagnostics"),
            "wing_stats": state_payload.get("wing_stats"),
            "adaptive_fallback_rate": adaptive_fallback_rate,
            "fallback_type_distribution": fallback_type_distribution,
            "open_positions_count": len(state_payload.get("open_positions", [])),
            "open_position_bodies": ",".join(
                str(item.get("body_strike")) for item in state_payload.get("open_positions", []) if item.get("body_strike") is not None
            ),
            "transitions_today_total": len(transitions_today),
            "actions_today_total": len(actions_today),
            "orders_today_total": len(orders_today),
            "actions_today_by_type": dict(sorted(action_counts.items())),
            "orders_today_by_status": dict(sorted(order_status_counts.items())),
            "orders_today_by_side": dict(sorted(order_side_counts.items())),
            "top_order_reasons_today": dict(order_reason_counts.most_common(5)),
            "open_orders_submitted_today": open_orders_submitted,
            "close_orders_submitted_today": close_orders_submitted,
            "filled_orders_today": sum(1 for row in orders_today if row.get("status") == "Filled"),
            "filled_open_orders_today": len(filled_open_orders),
            "filled_close_orders_today": len(filled_close_orders),
            "skipped_orders_today": skipped_orders_today,
            "execution_failure_orders_today": execution_failure_orders_today,
            "blocked_or_skipped_orders_today": sum(
                1 for row in orders_today if row.get("status") in {"blocked", "skipped"}
            ),
            "avg_open_fill_edge_vs_quote": self._average_csv_float(filled_open_orders, "fill_edge_vs_quote"),
            "avg_open_fill_edge_vs_limit": self._average_csv_float(filled_open_orders, "fill_edge_vs_limit"),
            "avg_open_limit_vs_quote": self._average_csv_float(filled_open_orders, "limit_vs_quote"),
            "avg_close_fill_edge_vs_quote": self._average_csv_float(filled_close_orders, "fill_edge_vs_quote"),
            "avg_close_fill_edge_vs_limit": self._average_csv_float(filled_close_orders, "fill_edge_vs_limit"),
            "avg_close_limit_vs_quote": self._average_csv_float(filled_close_orders, "limit_vs_quote"),
            "avg_filled_spread_ratio": self._average_csv_float(
                [row for row in orders_today if row.get("status") == "Filled"],
                "spread_ratio",
            ),
            "worst_open_fill_edge_vs_quote": self._min_csv_float(filled_open_orders, "fill_edge_vs_quote"),
            "worst_close_fill_edge_vs_quote": self._min_csv_float(filled_close_orders, "fill_edge_vs_quote"),
            "best_open_fill_edge_vs_quote": self._max_csv_float(filled_open_orders, "fill_edge_vs_quote"),
            "best_close_fill_edge_vs_quote": self._max_csv_float(filled_close_orders, "fill_edge_vs_quote"),
        }

    @staticmethod
    def _rows_for_local_date(path: Path, local_date_iso: str) -> list[dict[str, str]]:
        if not path.exists():
            return []
        rows: list[dict[str, str]] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = row.get("timestamp")
                if not value:
                    continue
                ts = _ensure_utc_timestamp(pd.Timestamp(value))
                row_date = ts.tz_convert("America/New_York").date().isoformat()
                if row_date == local_date_iso:
                    rows.append(row)
        return rows

    @staticmethod
    def _csv_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _average_csv_float(cls, rows: list[dict[str, str]], key: str) -> Optional[float]:
        values = [value for row in rows if (value := cls._csv_float(row.get(key))) is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    @classmethod
    def _min_csv_float(cls, rows: list[dict[str, str]], key: str) -> Optional[float]:
        values = [value for row in rows if (value := cls._csv_float(row.get(key))) is not None]
        if not values:
            return None
        return round(min(values), 4)

    @classmethod
    def _max_csv_float(cls, rows: list[dict[str, str]], key: str) -> Optional[float]:
        values = [value for row in rows if (value := cls._csv_float(row.get(key))) is not None]
        if not values:
            return None
        return round(max(values), 4)

    def _build_state_snapshot(self) -> dict[str, Any]:
        regime = self.detector.evaluate(self.history)
        center = self.estimator.estimate(self.history)
        latest = self.history.iloc[-1] if not self.history.empty else None
        latest_timestamp = latest["timestamp"].isoformat() if latest is not None else None
        latest_price = round(float(latest["close"]), 4) if latest is not None else None
        if latest is None and self.partial_bar is not None:
            latest_timestamp = self.partial_bar.last_sample_time.isoformat()
            latest_price = round(float(self.partial_bar.close), 4)
        candidates: list[dict[str, Any]] = []
        candidate_status = "Candidate selection skipped."
        candidate_error: Optional[str] = None
        candidate_diagnostics: Optional[dict[str, Any]] = None
        history_bars = len(self.history)
        warmup_remaining = max(0, self.required_warmup_bars - history_bars)
        model_ready = history_bars >= self.required_warmup_bars
        if self.warmup_mode and not model_ready:
            candidate_status = (
                f"Warming up from live market data only. "
                f"Completed bars: {history_bars}/{self.required_warmup_bars}."
            )
        elif regime is not None and regime.regime == Regime.RANGE and center is not None:
            try:
                self.latest_candidate_diagnostics = None
                loaded = self._load_candidates(center.center_price)
                candidate_diagnostics = self.latest_candidate_diagnostics
                candidates = [
                    {
                        "expiry": candidate.expiry,
                        "right": candidate.right,
                        "wing_mode": candidate.wing_mode,
                        "lower_width": round(candidate.lower_width, 4),
                        "upper_width": round(candidate.upper_width, 4),
                        "lower_strike": candidate.lower_strike,
                        "body_strike": candidate.body_strike,
                        "upper_strike": candidate.upper_strike,
                        "net_debit": round(candidate.net_debit, 4),
                        "total_spread": round(candidate.total_spread, 4),
                        "spread_ratio": round(candidate.spread_ratio, 4),
                        "reward_to_risk": round(candidate.reward_to_risk, 4),
                        "body_distance": round(candidate.body_distance, 4),
                        "max_risk": round(candidate.max_risk, 2),
                        "max_reward": round(candidate.max_reward, 2),
                    }
                    for candidate in loaded[:5]
                ]
                if candidates:
                    candidate_status = f"Loaded {len(candidates)} candidate butterflies for the current center."
                else:
                    candidate_status = "No qualifying butterflies matched the current center and spread filters."
                    if candidate_diagnostics is not None:
                        rejection_counts = candidate_diagnostics.get("rejection_counts", {})
                        if rejection_counts:
                            ordered = ", ".join(
                                f"{name}={count}"
                                for name, count in sorted(
                                    rejection_counts.items(),
                                    key=lambda item: (-int(item[1]), item[0]),
                                )
                            )
                            candidate_status += f" Rejections: {ordered}."
            except Exception as exc:
                candidate_error = str(exc).strip() or exc.__class__.__name__
                candidate_status = "Option-chain lookup failed; see candidate error."
                candidate_diagnostics = self.latest_candidate_diagnostics
        elif regime is not None and regime.regime != Regime.RANGE:
            candidate_status = f"Skipped candidate selection because regime={regime.regime.value}."
        elif center is None:
            candidate_status = "Skipped candidate selection because no valid center is available."

        payload = {
            "timestamp": latest_timestamp,
            "symbol": self.cfg.symbol,
            "price": latest_price,
            "regime": regime.regime.value if regime is not None else None,
            "center": center.center_price if center is not None else None,
            "actual_tolerance": round(center.actual_tolerance, 4) if center is not None else None,
            "state": self.machine.context.state.value,
            "startup_mode": "warmup_only" if self.warmup_mode else ("history_seeded" if history_bars > 0 else "empty"),
            "history_seeded": bool(not self.warmup_mode and history_bars > 0),
            "history_seed_status": self.history_seed_status,
            "warmup_mode": self.warmup_mode,
            "warmup_reason": self.warmup_reason,
            "history_refresh_error": self.history_refresh_error,
            "warmup_quote_error": self.warmup_quote_error,
            "history_bars": history_bars,
            "warmup_required_bars": self.required_warmup_bars,
            "warmup_remaining_bars": warmup_remaining,
            "model_ready": model_ready,
            "partial_bar": (
                {
                    "bucket_start": self.partial_bar.bucket_start.isoformat(),
                    "last_sample_time": self.partial_bar.last_sample_time.isoformat(),
                    "open": round(self.partial_bar.open, 4),
                    "high": round(self.partial_bar.high, 4),
                    "low": round(self.partial_bar.low, 4),
                    "close": round(self.partial_bar.close, 4),
                    "volume": round(self.partial_bar.volume, 4),
                }
                if self.partial_bar is not None
                else None
            ),
            "open_positions": [
                {
                    "layer_id": position.layer_id,
                    "expiry": position.candidate.expiry,
                    "trading_class": position.candidate.trading_class,
                    "right": position.candidate.right,
                    "lower_strike": position.candidate.lower_strike,
                    "body_strike": position.candidate.body_strike,
                    "upper_strike": position.candidate.upper_strike,
                    "layer_kind": position.layer_kind,
                    "quantity": position.quantity,
                    "open_limit": position.open_limit,
                    "open_fill_price": position.open_fill_price,
                    "open_status": position.open_status,
                    "source_action": position.source_action,
                    "close_requested_at": (
                        position.close_requested_at.isoformat() if position.close_requested_at is not None else None
                    ),
                    "close_limit": position.close_limit,
                    "close_fill_price": position.close_fill_price,
                    "close_status": position.close_status,
                    "close_failure_reason": position.close_failure_reason or None,
                }
                for position in sorted(self.positions.values(), key=lambda item: item.layer_id)
            ],
            "execution_mode": "paper" if self.runner_cfg.paper_execution else "dry-run",
            "execution_halted_reason": self.execution_halted_reason,
            "configured_wing_mode": self.cfg.wing_mode,
            "wing_stats": dict(self.wing_stats),
            "candidates": candidates,
            "candidate_status": candidate_status,
            "candidate_error": candidate_error,
            "candidate_diagnostics": candidate_diagnostics,
        }
        adaptive_fallback_rate, fallback_type_distribution = self._adaptive_stats_payload()
        payload["adaptive_fallback_rate"] = adaptive_fallback_rate
        payload["fallback_type_distribution"] = fallback_type_distribution
        return payload

    def _log_order(self, record: dict[str, Any]) -> None:
        self.logger.write_order(record)

    @staticmethod
    def _port_is_listening(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    def _resolve_ib_port(self, host: str, requested_port: int) -> int:
        local_hosts = {"127.0.0.1", "localhost", "::1"}
        if host not in local_hosts:
            return requested_port
        if self._port_is_listening(host, requested_port):
            return requested_port

        configured_port = self._detect_local_gateway_port()
        candidates: list[int] = []
        if configured_port is not None:
            candidates.append(configured_port)
        candidates.extend([4002, 4001, 4000, 7497, 7496])

        for candidate in candidates:
            if candidate == requested_port:
                continue
            if self._port_is_listening(host, candidate):
                return candidate
        return requested_port

    @staticmethod
    def _detect_local_gateway_port() -> Optional[int]:
        config_candidates = [
            Path("C:/Jts/ibgateway/1045/jts.ini"),
            Path.home() / "Jts" / "jts.ini",
        ]
        config_candidates.extend(Path("C:/Jts/ibgateway").glob("*/jts.ini"))
        config_candidates.extend((Path.home() / "Jts" / "ibgateway").glob("*/jts.ini"))

        seen: set[Path] = set()
        for path in config_candidates:
            resolved = path.resolve()
            if resolved in seen or not resolved.exists():
                continue
            seen.add(resolved)

            parser = ConfigParser()
            try:
                parser.read(resolved, encoding="utf-8")
                if parser.has_option("IBGateway", "LocalServerPort"):
                    value = parser.get("IBGateway", "LocalServerPort").strip()
                    return int(value)
            except (OSError, ValueError):
                continue
        return None
