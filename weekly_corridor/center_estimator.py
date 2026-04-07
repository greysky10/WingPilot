from __future__ import annotations

import pandas as pd

from corridor.models import CenterMethod

from .config import WeeklyCorridorConfig
from .models import WeeklyCenterEstimate


def round_center(price: float, increment: float) -> float:
    if increment <= 0:
        return float(price)
    return round(price / increment) * increment


class WeeklyCenterEstimator:
    """Estimate a weekly center from multi-day intraday bars."""

    def __init__(self, config: WeeklyCorridorConfig) -> None:
        self.config = config

    def estimate(self, frame: pd.DataFrame) -> WeeklyCenterEstimate | None:
        if frame.empty:
            return None

        sample = frame.tail(self.config.center_lookback_bars).copy()
        if len(sample) < max(8, self.config.center_lookback_bars // 3):
            return None

        method = self.config.center_method
        if method == CenterMethod.VWAP:
            typical = (sample["high"] + sample["low"] + sample["close"]) / 3.0
            raw_center = float((typical * sample["volume"]).sum() / max(sample["volume"].sum(), 1.0))
        elif method == CenterMethod.MEAN_MID:
            raw_center = float(((sample["high"] + sample["low"]) / 2.0).mean())
        else:
            raw_center = float(sample["close"].median())

        center_price = float(round_center(raw_center, self.config.center_rounding))
        span = self.config.weekly_span()
        realized_std = float(sample["close"].std(ddof=0)) if len(sample) > 1 else 0.0
        confidence = max(0.0, min(1.0, 1.0 - realized_std / max(span, 1.0)))
        return WeeklyCenterEstimate(
            timestamp=pd.Timestamp(sample["timestamp"].iloc[-1]),
            center_price=center_price,
            lower_coverage=center_price - span,
            upper_coverage=center_price + span,
            tolerance_low=center_price - self.config.weekly_center_tolerance,
            tolerance_high=center_price + self.config.weekly_center_tolerance,
            method=method,
            confidence=confidence,
            diagnostics={
                "raw_center": raw_center,
                "rounded_center": center_price,
                "realized_std": realized_std,
                "weekly_span": span,
                "lookback_bars": float(len(sample)),
                "tolerance_width": self.config.weekly_center_tolerance * 2.0,
            },
        )
