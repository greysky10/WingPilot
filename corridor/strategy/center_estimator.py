from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from corridor.config import CorridorConfig
from corridor.models import CenterEstimate, CenterMethod


def round_center(price: float, increment: float) -> float:
    """Round a center price to the configured strike increment."""

    if increment <= 0:
        return round(price, 6)
    return round(round(price / increment) * increment, 6)


@dataclass(slots=True)
class CenterEstimator:
    """Estimate the corridor center from recent bars."""

    config: CorridorConfig

    def estimate(self, history: pd.DataFrame) -> CenterEstimate | None:
        if len(history) < self.config.center_lookback:
            return None

        window = history.tail(self.config.center_lookback).copy()
        typical = (window["high"] + window["low"] + window["close"]) / 3.0
        timestamp = pd.Timestamp(window["timestamp"].iloc[-1])

        if self.config.center_method == CenterMethod.MEAN_MID:
            raw_center = float(((window["high"] + window["low"]) / 2.0).mean())
        elif self.config.center_method == CenterMethod.VWAP:
            weights = window["volume"].replace(0, np.nan).fillna(1.0)
            raw_center = float((window["close"] * weights).sum() / weights.sum())
        else:
            step = max(self.config.center_rounding / 2.0, 0.5)
            bins = np.arange(window["low"].min(), window["high"].max() + step, step)
            if len(bins) < 2:
                return None
            bucket_ids = np.digitize(typical, bins, right=False) - 1
            volume_by_bucket: dict[int, float] = {}
            for bucket_id, volume in zip(bucket_ids, window["volume"], strict=False):
                volume_by_bucket[bucket_id] = volume_by_bucket.get(bucket_id, 0.0) + float(volume)
            best_bucket = max(volume_by_bucket, key=volume_by_bucket.get)
            raw_center = float(bins[max(0, min(best_bucket, len(bins) - 1))])

        center = round_center(raw_center, self.config.center_rounding)
        band_half = self.config.coverage_band_width / 2.0
        tol_half = self.config.center_tolerance
        dispersion = float(typical.std(ddof=0)) if len(typical) > 1 else 0.0
        confidence = 0.0 if band_half <= 0 else max(0.0, 1.0 - min(1.0, dispersion / max(1e-6, band_half)))

        return CenterEstimate(
            timestamp=timestamp,
            center_price=center,
            lower_band=center - band_half,
            upper_band=center + band_half,
            tolerance_low=center - tol_half,
            tolerance_high=center + tol_half,
            method=self.config.center_method,
            confidence=confidence,
            diagnostics={
                "raw_center": raw_center,
                "dispersion": dispersion,
                "typical_last": float(typical.iloc[-1]),
            },
        )
