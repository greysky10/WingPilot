from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from corridor.config import CorridorConfig
from corridor.models import CenterEstimate, Regime, RegimeSnapshot


@dataclass(slots=True)
class DriftAssessment:
    outside_tolerance: bool
    outside_coverage: bool
    drift_distance: float
    next_drift_count: int
    cooldown_ready: bool
    should_rebuild: bool
    should_abort: bool
    abort_reason: str


class RecenterRuleEngine:
    """Evaluate drift persistence, rebuild cooldown, and abort conditions."""

    def __init__(self, config: CorridorConfig) -> None:
        self.config = config

    def evaluate(
        self,
        timestamp: pd.Timestamp,
        price: float,
        center: Optional[CenterEstimate],
        regime: Optional[RegimeSnapshot],
        current_drift_count: int,
        last_rebuild_at: Optional[pd.Timestamp],
    ) -> DriftAssessment:
        if center is None:
            return DriftAssessment(False, False, 0.0, 0, True, False, False, "")

        drift_distance = abs(price - center.center_price)
        outside_tolerance = drift_distance > self.config.center_tolerance
        outside_coverage = price < center.lower_band or price > center.upper_band
        next_drift_count = current_drift_count + 1 if outside_tolerance else 0

        cooldown_ready = True
        if last_rebuild_at is not None:
            cooldown_ready = (timestamp - last_rebuild_at) >= pd.Timedelta(minutes=self.config.rebuild_cooldown_minutes)

        should_abort = False
        abort_reason = ""
        if regime is not None:
            if regime.regime in {Regime.TREND_UP, Regime.TREND_DOWN}:
                should_abort = True
                abort_reason = f"Regime turned {regime.regime.value}."
            elif regime.volume_ratio >= self.config.abort_volume_threshold and outside_coverage:
                should_abort = True
                abort_reason = "Volume expansion while price broke out of the corridor."
            elif abs(regime.momentum_pct) >= self.config.abort_momentum_threshold and outside_tolerance:
                should_abort = True
                abort_reason = "Momentum expansion exceeded the abort threshold."

        should_rebuild = (
            outside_tolerance
            and drift_distance >= self.config.recenter_threshold
            and next_drift_count >= self.config.drift_persistence_bars
            and cooldown_ready
            and not should_abort
        )

        return DriftAssessment(
            outside_tolerance=outside_tolerance,
            outside_coverage=outside_coverage,
            drift_distance=drift_distance,
            next_drift_count=next_drift_count,
            cooldown_ready=cooldown_ready,
            should_rebuild=should_rebuild,
            should_abort=should_abort,
            abort_reason=abort_reason,
        )
