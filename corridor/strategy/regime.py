from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from corridor.config import CorridorConfig
from corridor.models import Regime, RegimeSnapshot


@dataclass(slots=True)
class RangeRegimeDetector:
    """Detect range and trend transitions from intraday bars."""

    config: CorridorConfig

    def evaluate(self, history: pd.DataFrame) -> RegimeSnapshot | None:
        if len(history) < max(8, self.config.regime_lookback):
            return None

        window = history.tail(self.config.regime_lookback).copy()
        close = window["close"]
        high = window["high"]
        low = window["low"]
        volume = window["volume"]

        fast_span = max(4, self.config.regime_lookback // 4)
        slow_span = max(8, self.config.regime_lookback // 2)
        ema_fast = close.ewm(span=fast_span, adjust=False).mean()
        ema_slow = close.ewm(span=slow_span, adjust=False).mean()

        px = float(close.iloc[-1])
        range_width_pct = float((high.max() - low.min()) / px) if px else 0.0
        trend_slope_pct = float((ema_fast.iloc[-1] - ema_slow.iloc[-1]) / px) if px else 0.0
        lookback_bar = close.iloc[-min(len(close), 6)]
        momentum_pct = float((px - lookback_bar) / lookback_bar) if lookback_bar else 0.0
        avg_volume = float(volume.iloc[:-1].mean()) if len(volume) > 1 else float(volume.iloc[-1])
        volume_ratio = float(volume.iloc[-1] / avg_volume) if avg_volume else 1.0

        prior_high = float(high.iloc[:-1].max()) if len(high) > 1 else float(high.iloc[-1])
        prior_low = float(low.iloc[:-1].min()) if len(low) > 1 else float(low.iloc[-1])
        breakout_up = px >= prior_high * (1.0 + self.config.breakout_buffer_pct)
        breakout_down = px <= prior_low * (1.0 - self.config.breakout_buffer_pct)

        if breakout_up or (trend_slope_pct >= self.config.trend_slope_threshold_pct and momentum_pct > 0):
            regime = Regime.TREND_UP
        elif breakout_down or (trend_slope_pct <= -self.config.trend_slope_threshold_pct and momentum_pct < 0):
            regime = Regime.TREND_DOWN
        elif range_width_pct <= self.config.range_width_threshold_pct and abs(trend_slope_pct) < self.config.trend_slope_threshold_pct:
            regime = Regime.RANGE
        else:
            regime = Regime.NEUTRAL

        return RegimeSnapshot(
            timestamp=pd.Timestamp(window["timestamp"].iloc[-1]),
            regime=regime,
            range_width_pct=range_width_pct,
            trend_slope_pct=trend_slope_pct,
            momentum_pct=momentum_pct,
            volume_ratio=volume_ratio,
            breakout_up=breakout_up,
            breakout_down=breakout_down,
        )
