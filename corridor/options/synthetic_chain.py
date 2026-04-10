from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from corridor.config import CorridorConfig
from corridor.models import ActiveButterfly


@dataclass(slots=True)
class SyntheticChainAnchor:
    remaining_dte: float
    reward_width: float
    extra_tail_width: float
    net_debit: float
    total_spread: float
    spread_ratio: float
    body_distance: float


@dataclass(slots=True)
class SyntheticChainCalibration:
    symbol: str
    reference_timestamp: pd.Timestamp
    reference_spot: float
    anchors: list[SyntheticChainAnchor]
    rejection_spread_multiplier: float
    source_state_path: str
    source_report_path: str


def _default_state_path(symbol: str) -> Path:
    return Path("corridor_outputs") / "paper_runner" / symbol.upper() / "paper_state.json"


def _default_report_path(symbol: str) -> Path:
    return Path("corridor_outputs") / "paper_runner" / symbol.upper() / "paper_daily_report.json"


def _ensure_utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _calendar_dte(expiry: str, reference_ts: pd.Timestamp) -> Optional[int]:
    expiry_ts = pd.to_datetime(str(expiry), format="%Y%m%d", errors="coerce")
    if pd.isna(expiry_ts):
        return None
    local_ref = _ensure_utc_timestamp(reference_ts).tz_convert("America/New_York")
    return int((expiry_ts.date() - local_ref.date()).days)


def _median(values: list[float], default: float) -> float:
    cleaned = sorted(float(value) for value in values if value is not None)
    if not cleaned:
        return default
    mid = len(cleaned) // 2
    if len(cleaned) % 2:
        return cleaned[mid]
    return (cleaned[mid - 1] + cleaned[mid]) / 2.0


def load_synthetic_chain_calibration(
    config: CorridorConfig,
    state_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> SyntheticChainCalibration:
    resolved_state_path = Path(
        config.synthetic_chain_state_path or state_path or _default_state_path(config.symbol)
    )
    resolved_report_path = Path(
        config.synthetic_chain_report_path or report_path or _default_report_path(config.symbol)
    )
    if not resolved_state_path.exists():
        raise FileNotFoundError(
            f"Synthetic chain calibration requires a paper state snapshot: {resolved_state_path}"
        )
    if not resolved_report_path.exists():
        raise FileNotFoundError(
            f"Synthetic chain calibration requires a paper daily report: {resolved_report_path}"
        )
    config.synthetic_chain_state_path = str(resolved_state_path)
    config.synthetic_chain_report_path = str(resolved_report_path)

    state_payload = json.loads(resolved_state_path.read_text(encoding="utf-8"))
    report_payload = json.loads(resolved_report_path.read_text(encoding="utf-8"))
    reference_ts = _ensure_utc_timestamp(state_payload.get("timestamp") or report_payload.get("latest_timestamp"))
    reference_spot = float(
        state_payload.get("price")
        or report_payload.get("latest_price")
        or 0.0
    )

    anchors: list[SyntheticChainAnchor] = []
    for candidate in state_payload.get("candidates", []):
        try:
            reward_width = min(float(candidate["lower_width"]), float(candidate["upper_width"]))
            extra_tail_width = abs(float(candidate["upper_width"]) - float(candidate["lower_width"]))
            remaining_dte = _calendar_dte(str(candidate["expiry"]), reference_ts)
            if remaining_dte is None:
                continue
            anchors.append(
                SyntheticChainAnchor(
                    remaining_dte=float(remaining_dte),
                    reward_width=max(0.01, reward_width),
                    extra_tail_width=max(0.0, extra_tail_width),
                    net_debit=max(0.01, float(candidate["net_debit"])),
                    total_spread=max(0.01, float(candidate["total_spread"])),
                    spread_ratio=max(0.0, float(candidate.get("spread_ratio") or 0.0)),
                    body_distance=abs(float(candidate.get("body_distance") or 0.0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    if not anchors:
        raise ValueError(
            f"Synthetic chain calibration found no valid anchors in {resolved_state_path}"
        )

    diagnostics = report_payload.get("candidate_diagnostics") or state_payload.get("candidate_diagnostics") or {}
    rejection_counts = diagnostics.get("rejection_counts", {}) if isinstance(diagnostics, dict) else {}
    spread_rejections = float(rejection_counts.get("spread_too_wide", 0) or 0.0)
    attempted = float(diagnostics.get("attempted_structures", 0) or 0.0)
    rejection_rate = (spread_rejections / attempted) if attempted > 0 else 0.0

    accepted_spread_per_width = [
        anchor.total_spread / max(anchor.reward_width, 0.01)
        for anchor in anchors
    ]
    rejected_spread_per_width: list[float] = []
    for sample in diagnostics.get("sample_rejections", []) if isinstance(diagnostics, dict) else []:
        if str(sample.get("reason", "")) != "spread_too_wide":
            continue
        try:
            lower = float(sample["lower_strike"])
            upper = float(sample["upper_strike"])
            body = float(sample["body_strike"])
            reward_width = max(0.01, min(body - lower, upper - body))
            rejected_spread_per_width.append(float(sample["total_spread"]) / reward_width)
        except (KeyError, TypeError, ValueError):
            continue

    accepted_median = _median(accepted_spread_per_width, default=0.05)
    rejected_median = _median(rejected_spread_per_width, default=accepted_median)
    rejection_spread_multiplier = 1.0
    if accepted_median > 0:
        widened = rejected_median / accepted_median
        rejection_spread_multiplier = 1.0 + min(0.75, rejection_rate) * max(0.0, widened - 1.0)

    return SyntheticChainCalibration(
        symbol=str(state_payload.get("symbol") or config.symbol).upper(),
        reference_timestamp=reference_ts,
        reference_spot=reference_spot,
        anchors=anchors,
        rejection_spread_multiplier=max(1.0, rejection_spread_multiplier),
        source_state_path=str(resolved_state_path),
        source_report_path=str(resolved_report_path),
    )


class SyntheticChainButterflyPricer:
    """Approximate butterfly pricing from a live-paper chain snapshot and spread diagnostics."""

    def __init__(self, config: CorridorConfig, calibration: SyntheticChainCalibration) -> None:
        self.config = config
        self.calibration = calibration

    @classmethod
    def from_config(cls, config: CorridorConfig) -> SyntheticChainButterflyPricer:
        return cls(config, load_synthetic_chain_calibration(config))

    def entry_debit(self, layer: ActiveButterfly) -> float:
        reward_width = self._reward_width(layer)
        debit_per_width = self._weighted_anchor_value(
            target_remaining_dte=float(layer.dte),
            reward_width=reward_width,
            extra_tail_width=self._extra_tail_width(layer),
            extractor=lambda anchor: anchor.net_debit / max(anchor.reward_width, 0.01),
        )
        return max(0.01, reward_width * debit_per_width * self.config.stress_entry_debit_multiplier)

    def estimated_total_spread(self, layer: ActiveButterfly, remaining_dte: Optional[float] = None) -> float:
        reward_width = self._reward_width(layer)
        spread_per_width = self._weighted_anchor_value(
            target_remaining_dte=float(layer.dte if remaining_dte is None else remaining_dte),
            reward_width=reward_width,
            extra_tail_width=self._extra_tail_width(layer),
            extractor=lambda anchor: anchor.total_spread / max(anchor.reward_width, 0.01),
        )
        spread = reward_width * spread_per_width * self.calibration.rejection_spread_multiplier
        return max(0.01, spread)

    def estimated_spread_ratio(self, layer: ActiveButterfly, remaining_dte: Optional[float] = None) -> float:
        debit = self.entry_debit(layer)
        spread = self.estimated_total_spread(layer, remaining_dte=remaining_dte)
        return spread / max(debit, 0.01)

    def entry_cost(self, layer: ActiveButterfly) -> float:
        return self.entry_debit(layer) + self.friction_per_layer(layer)

    def mark_to_model(self, layer: ActiveButterfly, spot: float, timestamp: pd.Timestamp) -> float:
        reward_width = self._reward_width(layer)
        elapsed_days = max(0.0, (_ensure_utc_timestamp(timestamp) - _ensure_utc_timestamp(layer.created_at)).total_seconds() / 86400.0)
        time_progress = min(1.0, elapsed_days / max(float(layer.dte), 1.0))
        remaining_dte = max(0.0, float(layer.dte) - elapsed_days)

        side_width = max(self._side_width(layer, spot), 0.01)
        drift_units = abs(float(spot) - float(layer.center_price)) / side_width
        proximity = max(0.0, 1.0 - min(drift_units, 1.5) / 1.5)

        entry_cost = layer.entry_cost if layer.entry_cost > 0 else self.entry_cost(layer)
        residual_floor = max(
            self.estimated_total_spread(layer, remaining_dte=remaining_dte) * 0.5,
            reward_width * 0.02,
        )
        current_surface = entry_cost * proximity + residual_floor * (1.0 - proximity)
        terminal_value = self.terminal_combo_value(layer, spot)
        mark = current_surface * (1.0 - time_progress) + terminal_value * time_progress
        return float(mark)

    def close_value(self, layer: ActiveButterfly, spot: float, timestamp: pd.Timestamp) -> float:
        elapsed_days = max(0.0, (_ensure_utc_timestamp(timestamp) - _ensure_utc_timestamp(layer.created_at)).total_seconds() / 86400.0)
        remaining_dte = max(0.0, float(layer.dte) - elapsed_days)
        mark = self.mark_to_model(layer, spot, timestamp)
        close_before_haircut = mark - self.friction_per_layer(layer, remaining_dte=remaining_dte)
        if close_before_haircut >= 0:
            return max(0.0, close_before_haircut * (1.0 - self.config.stress_close_value_haircut_pct))
        return close_before_haircut * (1.0 + self.config.stress_close_value_haircut_pct)

    def friction_per_layer(self, layer: ActiveButterfly | None = None, remaining_dte: Optional[float] = None) -> float:
        return self.slippage_cost_per_layer(layer, remaining_dte=remaining_dte) + self.commission_cost_per_layer()

    def commission_cost_per_layer(self) -> float:
        return (self.config.commission_per_contract * 4.0) / float(self.config.option_multiplier)

    def slippage_cost_per_layer(self, layer: ActiveButterfly | None = None, remaining_dte: Optional[float] = None) -> float:
        if layer is None:
            reward_width = max(0.01, min(float(self.config.butterfly_width), float(self.config.butterfly_width) + max(0.0, float(self.config.broken_wing_extra_width))))
            extra_tail = max(0.0, float(self.config.broken_wing_extra_width))
            estimated_spread = self._weighted_anchor_value(
                target_remaining_dte=float(self.config.default_dte),
                reward_width=reward_width,
                extra_tail_width=extra_tail,
                extractor=lambda anchor: anchor.total_spread,
            )
        else:
            estimated_spread = self.estimated_total_spread(layer, remaining_dte=remaining_dte)
        return min(max(0.01, estimated_spread * 0.25), max(0.02, float(self.config.max_acceptable_option_spread)))

    def modeled_max_loss(self, layer: ActiveButterfly) -> float:
        return self.entry_cost(layer) + self._extra_tail_width(layer)

    @staticmethod
    def terminal_combo_value(layer: ActiveButterfly, spot: float) -> float:
        spot = float(spot)
        lower = float(layer.lower_strike)
        body = float(layer.body_strike)
        upper = float(layer.upper_strike)
        return (
            max(spot - lower, 0.0)
            - (2.0 * max(spot - body, 0.0))
            + max(spot - upper, 0.0)
        )

    @staticmethod
    def _reward_width(layer: ActiveButterfly) -> float:
        return max(0.01, min(float(layer.lower_width), float(layer.upper_width)))

    @staticmethod
    def _extra_tail_width(layer: ActiveButterfly) -> float:
        return abs(float(layer.upper_width) - float(layer.lower_width))

    @staticmethod
    def _side_width(layer: ActiveButterfly, spot: float) -> float:
        return float(layer.upper_width) if float(spot) >= float(layer.center_price) else float(layer.lower_width)

    def _weighted_anchor_value(
        self,
        target_remaining_dte: float,
        reward_width: float,
        extra_tail_width: float,
        extractor,
    ) -> float:
        scored: list[tuple[float, float]] = []
        for anchor in self.calibration.anchors:
            width_gap = abs(anchor.reward_width - reward_width) / max(reward_width, 1.0)
            tail_gap = abs(anchor.extra_tail_width - extra_tail_width) / max(reward_width, 1.0)
            dte_gap = abs(anchor.remaining_dte - target_remaining_dte)
            body_penalty = anchor.body_distance / max(reward_width, 1.0)
            distance = (width_gap * 3.0) + (tail_gap * 4.0) + (dte_gap * 0.35) + body_penalty
            scored.append((1.0 / max(distance, 0.05), float(extractor(anchor))))
        if not scored:
            return 0.01
        total_weight = sum(weight for weight, _ in scored)
        return sum(weight * value for weight, value in scored) / max(total_weight, 1e-9)
