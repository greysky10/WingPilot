from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .config import WeeklyCorridorConfig
from .models import WeeklyButterfly, WeeklyLayerKind, WeeklyRegime, WeeklyRegimeSnapshot


def parse_timeframe_rule(label: str) -> str:
    normalized = label.strip().lower().replace("minutes", "min").replace("minute", "min").replace("mins", "min")
    normalized = normalized.replace(" ", "")
    mapping = {
        "30min": "30min",
        "60min": "60min",
        "1h": "60min",
        "60m": "60min",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported weekly decision timeframe: {label}")
    return mapping[normalized]


def trading_week_key(timestamp: pd.Timestamp) -> str:
    local = pd.Timestamp(timestamp).tz_convert("America/New_York")
    monday = (local - pd.Timedelta(days=local.weekday())).normalize()
    return monday.strftime("%Y-%m-%d")


def local_trading_date(timestamp: pd.Timestamp) -> date:
    return pd.Timestamp(timestamp).tz_convert("America/New_York").date()


def bday_count(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> int:
    return len(pd.bdate_range(start=local_trading_date(start_ts), end=local_trading_date(end_ts)))


def is_event_week(timestamp: pd.Timestamp, event_dates: tuple[str, ...]) -> bool:
    if not event_dates:
        return False
    local = pd.Timestamp(timestamp).tz_convert("America/New_York")
    monday = (local - pd.Timedelta(days=local.weekday())).normalize().date()
    friday = monday + pd.Timedelta(days=4)
    normalized: set[date] = set()
    for value in event_dates:
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is None:
            normalized.add(parsed.tz_localize("America/New_York").date())
        else:
            normalized.add(parsed.tz_convert("America/New_York").date())
    return any(monday <= event_date <= friday for event_date in normalized)


def prepare_weekly_frame(frame: pd.DataFrame, decision_timeframe: str) -> pd.DataFrame:
    """Resample intraday bars to the weekly strategy decision timeframe."""

    raw = frame.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    rule = parse_timeframe_rule(decision_timeframe)

    pieces: list[pd.DataFrame] = []
    for symbol, symbol_frame in raw.groupby("symbol", sort=False):
        local = symbol_frame.copy()
        local["timestamp_ny"] = local["timestamp"].dt.tz_convert("America/New_York")
        local["session_date"] = local["timestamp_ny"].dt.date
        for _, session in local.groupby("session_date", sort=True):
            indexed = session.set_index("timestamp_ny")
            resampled = (
                indexed.resample(rule, label="right", closed="right", origin="start_day", offset="30min")
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna(subset=["open", "high", "low", "close"])
            )
            if resampled.empty:
                continue
            resampled = resampled.reset_index()
            resampled["timestamp"] = resampled["timestamp_ny"].dt.tz_convert("UTC")
            resampled["symbol"] = symbol
            pieces.append(resampled[["timestamp", "symbol", "open", "high", "low", "close", "volume"]])

    if not pieces:
        raise ValueError("Weekly decision frame is empty after resampling.")

    return pd.concat(pieces, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def round_strike(price: float, increment: float) -> float:
    if increment <= 0:
        return float(price)
    return round(price / increment) * increment


def build_initial_butterflies(
    config: WeeklyCorridorConfig,
    center_price: float,
    timestamp: pd.Timestamp,
    week_key: str,
    starting_layer_id: int,
) -> list[WeeklyButterfly]:
    centers = [center_price - config.center_spacing, center_price, center_price + config.center_spacing]
    butterflies: list[WeeklyButterfly] = []
    for offset, candidate_center in enumerate(centers[: max(0, min(3, config.max_active_butterflies))]):
        body = round_strike(candidate_center, config.center_rounding)
        butterflies.append(
            WeeklyButterfly(
                layer_id=starting_layer_id + offset,
                kind=WeeklyLayerKind.INITIAL,
                week_key=week_key,
                center_price=body,
                width=config.butterfly_width,
                lower_strike=body - config.butterfly_width,
                body_strike=body,
                upper_strike=body + config.butterfly_width,
                created_at=timestamp,
                dte=config.default_dte,
            )
        )
    return butterflies


def build_adjustment_butterfly(
    config: WeeklyCorridorConfig,
    center_price: float,
    timestamp: pd.Timestamp,
    week_key: str,
    layer_id: int,
) -> WeeklyButterfly:
    body = round_strike(center_price, config.center_rounding)
    return WeeklyButterfly(
        layer_id=layer_id,
        kind=WeeklyLayerKind.ADJUSTMENT,
        week_key=week_key,
        center_price=body,
        width=config.butterfly_width,
        lower_strike=body - config.butterfly_width,
        body_strike=body,
        upper_strike=body + config.butterfly_width,
        created_at=timestamp,
        dte=config.default_dte,
    )


def corridor_bounds(layers: list[WeeklyButterfly]) -> tuple[float | None, float | None]:
    if not layers:
        return None, None
    return min(layer.lower_strike for layer in layers), max(layer.upper_strike for layer in layers)


@dataclass(slots=True)
class WeeklyRegimeClassifier:
    config: WeeklyCorridorConfig

    def evaluate(self, history: pd.DataFrame) -> WeeklyRegimeSnapshot:
        timestamp = pd.Timestamp(history["timestamp"].iloc[-1])
        sample = history.tail(self.config.regime_lookback_bars).copy()
        if len(sample) < max(12, self.config.regime_lookback_bars // 3):
            return WeeklyRegimeSnapshot(
                timestamp=timestamp,
                regime=WeeklyRegime.NEUTRAL,
                width_pct=0.0,
                slope_pct=0.0,
                momentum_pct=0.0,
                breakout_up=False,
                breakout_down=False,
                event_blocked=False,
            )

        event_blocked = self.config.skip_event_weeks and is_event_week(timestamp, self.config.event_dates)
        avg_close = max(float(sample["close"].mean()), 1.0)
        width_pct = float((sample["high"].max() - sample["low"].min()) / avg_close)
        slope_pct = float((sample["close"].iloc[-1] - sample["close"].iloc[0]) / avg_close)
        baseline = float(sample["close"].tail(min(6, len(sample))).mean())
        momentum_pct = float((sample["close"].iloc[-1] - baseline) / avg_close)

        if len(sample) > 1:
            prior_high = float(sample["high"].iloc[:-1].max())
            prior_low = float(sample["low"].iloc[:-1].min())
        else:
            prior_high = float(sample["high"].iloc[-1])
            prior_low = float(sample["low"].iloc[-1])
        close = float(sample["close"].iloc[-1])
        breakout_up = close > prior_high * (1.0 + self.config.breakout_buffer_pct)
        breakout_down = close < prior_low * (1.0 - self.config.breakout_buffer_pct)

        if event_blocked:
            regime = WeeklyRegime.EVENT_BLOCKED
        elif breakout_up or slope_pct >= self.config.weekly_trend_slope_threshold_pct or momentum_pct >= self.config.weekly_momentum_threshold_pct:
            regime = WeeklyRegime.TREND_UP
        elif breakout_down or slope_pct <= -self.config.weekly_trend_slope_threshold_pct or momentum_pct <= -self.config.weekly_momentum_threshold_pct:
            regime = WeeklyRegime.TREND_DOWN
        elif (
            width_pct <= self.config.weekly_range_width_threshold_pct
            and abs(slope_pct) <= self.config.weekly_trend_slope_threshold_pct
            and abs(momentum_pct) <= self.config.weekly_momentum_threshold_pct
        ):
            regime = WeeklyRegime.RANGE
        else:
            regime = WeeklyRegime.NEUTRAL

        return WeeklyRegimeSnapshot(
            timestamp=timestamp,
            regime=regime,
            width_pct=width_pct,
            slope_pct=slope_pct,
            momentum_pct=momentum_pct,
            breakout_up=breakout_up,
            breakout_down=breakout_down,
            event_blocked=event_blocked,
            diagnostics={
                "avg_close": avg_close,
                "prior_high": prior_high,
                "prior_low": prior_low,
            },
        )


@dataclass(slots=True)
class WeeklyButterflyPricer:
    config: WeeklyCorridorConfig

    def entry_debit(self, layer: WeeklyButterfly) -> float:
        return layer.width * self.config.simplified_entry_debit_pct_of_width

    def friction_per_layer(self) -> float:
        per_contract_cost = (self.config.commission_per_contract * 4.0) / float(self.config.option_multiplier)
        return self.config.slippage + per_contract_cost

    def mark_to_model(self, layer: WeeklyButterfly, spot: float, timestamp: pd.Timestamp) -> float:
        width = max(layer.width, 0.01)
        drift_units = abs(spot - layer.center_price) / width
        proximity = max(0.0, 1.0 - min(drift_units, 1.8) / 1.8)
        elapsed_days = max(0.0, (timestamp - layer.created_at).total_seconds() / 86400.0)
        time_progress = min(1.0, elapsed_days / max(layer.dte, 1))
        peak_value = layer.width * self.config.simplified_peak_value_pct_of_width
        entry_cost = layer.entry_cost if layer.entry_cost else self.entry_debit(layer) + self.friction_per_layer()
        residual_floor = layer.width * self.config.simplified_residual_floor_pct

        inside_value = entry_cost + (peak_value - entry_cost) * proximity * (0.2 + 0.8 * time_progress)
        outside_decay = residual_floor * max(0.0, 1.0 - time_progress * 0.55) * max(0.0, 1.4 - drift_units)
        if drift_units <= 1.0:
            return max(residual_floor, inside_value)
        return max(0.0, outside_decay)
