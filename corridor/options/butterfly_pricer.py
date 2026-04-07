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
        reward_width = self._reward_width(layer)
        debit = reward_width * self.config.simplified_entry_debit_pct_of_width
        asymmetry = abs(layer.upper_width - layer.lower_width)
        if asymmetry > 0:
            debit = max(reward_width * 0.05, debit - (asymmetry * 0.08))
        return debit * self.config.stress_entry_debit_multiplier

    def entry_cost(self, layer: ActiveButterfly) -> float:
        return self.entry_debit(layer) + self.friction_per_layer(layer)

    def mark_to_model(self, layer: ActiveButterfly, spot: float, timestamp: pd.Timestamp) -> float:
        side_width = max(self._side_width(layer, spot), 0.01)
        reward_width = max(self._reward_width(layer), 0.01)
        drift_units = abs(spot - layer.center_price) / side_width
        proximity = max(0.0, 1.0 - min(drift_units, 1.5) / 1.5)

        elapsed_days = max(0.0, (timestamp - layer.created_at).total_seconds() / 86400.0)
        time_progress = min(1.0, elapsed_days / max(layer.dte, 1))
        peak_value = (
            reward_width
            * self.config.simplified_peak_value_pct_of_width
            * self.config.stress_peak_value_multiplier
        )
        residual_floor = (
            reward_width
            * self.config.simplified_residual_floor_pct
            * self.config.stress_residual_floor_multiplier
        )

        inside_value = layer.entry_cost + (peak_value - layer.entry_cost) * proximity * (0.25 + 0.75 * time_progress)
        outside_decay = residual_floor * max(0.0, 1.0 - time_progress * 0.7) * max(0.0, 1.25 - drift_units)
        terminal_tail_value = self._terminal_tail_value(layer, spot) * time_progress

        if drift_units <= 1.0:
            return max(terminal_tail_value, max(residual_floor, inside_value))

        if terminal_tail_value >= 0:
            return max(0.0, outside_decay)

        broken_progress = min(
            1.0,
            max(0.0, abs(spot - layer.center_price) - reward_width) / max(abs(layer.upper_width - layer.lower_width), reward_width),
        )
        return outside_decay * (1.0 - broken_progress) + terminal_tail_value * broken_progress

    def close_value(self, layer: ActiveButterfly, spot: float, timestamp: pd.Timestamp) -> float:
        mark = self.mark_to_model(layer, spot, timestamp)
        friction = self.friction_per_layer(layer)
        close_before_haircut = mark - friction
        if close_before_haircut >= 0:
            return max(0.0, close_before_haircut * (1.0 - self.config.stress_close_value_haircut_pct))
        return close_before_haircut * (1.0 + self.config.stress_close_value_haircut_pct)

    def friction_per_layer(self, layer: ActiveButterfly | None = None) -> float:
        """Return modeled one-side execution friction per layer in model points."""

        return self.slippage_cost_per_layer(layer) + self.commission_cost_per_layer()

    def commission_cost_per_layer(self) -> float:
        return (self.config.commission_per_contract * 4.0) / float(self.config.option_multiplier)

    def slippage_cost_per_layer(self, layer: ActiveButterfly | None = None) -> float:
        contract_equivalents = self.slippage_contract_equivalents(layer)
        return (
            self.config.per_contract_slippage
            * contract_equivalents
            * self.config.stress_slippage_multiplier
        )

    def slippage_contract_equivalents(self, layer: ActiveButterfly | None = None) -> float:
        lower_width, upper_width = self._effective_widths(layer)
        if abs(lower_width - upper_width) > 0:
            return 5.0
        return 4.0

    def modeled_max_loss(self, layer: ActiveButterfly) -> float:
        broken_side_extra = abs(layer.upper_width - layer.lower_width)
        return layer.entry_cost + broken_side_extra

    @staticmethod
    def _reward_width(layer: ActiveButterfly) -> float:
        return max(0.01, min(layer.lower_width, layer.upper_width))

    def _effective_widths(self, layer: ActiveButterfly | None) -> tuple[float, float]:
        if layer is not None:
            return float(layer.lower_width), float(layer.upper_width)
        width = max(0.01, float(self.config.butterfly_width))
        extra = max(0.0, float(self.config.broken_wing_extra_width))
        if self.config.wing_mode == "broken_upper":
            return width, width + extra
        if self.config.wing_mode == "broken_lower":
            return width + extra, width
        return width, width

    @staticmethod
    def _side_width(layer: ActiveButterfly, spot: float) -> float:
        return layer.upper_width if spot >= layer.center_price else layer.lower_width

    @staticmethod
    def _terminal_tail_value(layer: ActiveButterfly, spot: float) -> float:
        if spot >= layer.center_price and layer.upper_width > layer.lower_width:
            return -(layer.upper_width - layer.lower_width)
        if spot < layer.center_price and layer.lower_width > layer.upper_width:
            return -(layer.lower_width - layer.upper_width)
        return 0.0
