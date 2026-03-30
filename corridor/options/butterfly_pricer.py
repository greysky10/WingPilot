from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from corridor.config import CorridorConfig
from corridor.models import ActiveButterfly


@dataclass(slots=True)
class SimplifiedButterflyPricer:
    """Approximate butterfly pricing from center, width, DTE, and drift."""

    config: CorridorConfig

    def entry_debit(self, layer: ActiveButterfly) -> float:
        return layer.width * self.config.simplified_entry_debit_pct_of_width

    def entry_cost(self, layer: ActiveButterfly) -> float:
        return self.entry_debit(layer) + self.friction_per_layer()

    def mark_to_model(self, layer: ActiveButterfly, spot: float, timestamp: pd.Timestamp) -> float:
        width = max(layer.width, 0.01)
        drift_units = abs(spot - layer.center_price) / width
        proximity = max(0.0, 1.0 - min(drift_units, 1.5) / 1.5)

        elapsed_days = max(0.0, (timestamp - layer.created_at).total_seconds() / 86400.0)
        time_progress = min(1.0, elapsed_days / max(layer.dte, 1))
        peak_value = layer.width * self.config.simplified_peak_value_pct_of_width
        residual_floor = layer.width * self.config.simplified_residual_floor_pct

        inside_value = layer.entry_cost + (peak_value - layer.entry_cost) * proximity * (0.25 + 0.75 * time_progress)
        outside_decay = residual_floor * max(0.0, 1.0 - time_progress * 0.7) * max(0.0, 1.25 - drift_units)

        if drift_units <= 1.0:
            return max(residual_floor, inside_value)
        return max(0.0, outside_decay)

    def close_value(self, layer: ActiveButterfly, spot: float, timestamp: pd.Timestamp) -> float:
        mark = self.mark_to_model(layer, spot, timestamp)
        friction = self.friction_per_layer()
        return max(0.0, mark - friction)

    def friction_per_layer(self) -> float:
        """Return modeled round-turn entry/exit friction per layer in model points."""

        per_contract_cost = (self.config.commission_per_contract * 4.0) / float(self.config.option_multiplier)
        return self.config.slippage + per_contract_cost
