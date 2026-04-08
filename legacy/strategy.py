from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    pd = None  # type: ignore[assignment]
    PANDAS_IMPORT_ERROR = exc
else:
    PANDAS_IMPORT_ERROR = None

try:
    import pytz
except ModuleNotFoundError as exc:
    pytz = None  # type: ignore[assignment]
    PYTZ_IMPORT_ERROR = exc
else:
    PYTZ_IMPORT_ERROR = None


TORONTO_TZ = pytz.timezone("America/Toronto") if pytz is not None else None
UTC = dt.timezone.utc
MARKET_OPEN = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)
STRATEGY_DEPENDENCY_HINT = "py -3.12 -m pip install pandas pytz"


def load_local_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_missing_strategy_dependencies() -> List[str]:
    missing: List[str] = []
    if PANDAS_IMPORT_ERROR is not None:
        missing.append("pandas")
    if PYTZ_IMPORT_ERROR is not None:
        missing.append("pytz")
    return missing


def require_strategy_dependencies() -> None:
    missing = get_missing_strategy_dependencies()
    if not missing:
        return
    raise SystemExit(
        "Missing required packages: "
        + ", ".join(missing)
        + ". Install them with `"
        + STRATEGY_DEPENDENCY_HINT
        + "`."
    )


def require_toronto_tz():
    if TORONTO_TZ is None:
        raise RuntimeError("pytz is required to resolve America/Toronto market hours.")
    return TORONTO_TZ


def coerce_utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert(UTC) if ts.tzinfo else ts.tz_localize(UTC)


class Regime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    NO_TRADE = "NO_TRADE"


class SignalType(str, Enum):
    TREND_LONG = "TREND_LONG"
    TREND_SHORT = "TREND_SHORT"
    RANGE_BUTTERFLY_ZONE = "RANGE_BUTTERFLY_ZONE"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class StrategyConfig:
    symbols: List[str] = field(default_factory=lambda: ["SPY"])
    max_bars_kept: int = 600

    # Trend detection
    ema_fast: int = 5
    ema_mid: int = 10
    ema_slow: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_period: int = 14
    stoch_smooth_k: int = 3
    stoch_smooth_d: int = 3
    trend_fast_timeframe: str = "15min"
    trend_slow_timeframe: str = "30min"

    # Signal thresholds
    min_volume_ratio: float = 1.05
    min_trend_spread_pct: float = 0.0008
    max_range_spread_pct: float = 0.0006
    pullback_tolerance_pct: float = 0.0015
    stop_below_recent_low_pct: float = 0.0008
    stop_above_recent_high_pct: float = 0.0008
    min_rr: float = 1.5

    # Range center / butterfly heuristic
    range_center_lookback: int = 30
    range_center_band_pct: float = 0.0012

    # Risk / notification
    cooldown_minutes: int = 30
    time_stop_minutes: int = 30
    time_stop_min_progress_pct: float = 0.0
    max_same_direction_entries_per_session: int = 1
    opening_range_minutes: int = 30
    require_opening_range_breakout: bool = True
    max_range_signals_per_session: int = 1
    signal_csv_path: str = "signals.csv"
    discord_webhook_url: str = ""
    print_json: bool = False

    # Option planning
    option_directional_expiry_days: int = 3
    option_range_expiry_days: int = 1
    option_strike_step: float = 1.0
    option_range_wing_width: float = 10.0


@dataclass
class PositionState:
    side: Optional[str] = None
    entry_price: Optional[float] = None
    entry_time: Optional[pd.Timestamp] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    last_signal_time: Optional[pd.Timestamp] = None
    session_date: Optional[dt.date] = None
    long_entries_today: int = 0
    short_entries_today: int = 0
    range_signals_today: int = 0


@dataclass
class EmittedSignal:
    symbol: str
    signal: SignalType
    regime: Regime
    price: float
    timestamp: pd.Timestamp
    reason: str
    extras: Dict[str, float | str]


@dataclass
class OptionPlan:
    structure: str
    expiry: str
    legs: List[str]
    notes: str = ""

    def to_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "structure": self.structure,
            "expiry": self.expiry,
            "legs": self.legs,
        }
        if self.notes:
            payload["notes"] = self.notes
        return payload

    def summary(self) -> str:
        summary = f"{self.structure} | exp={self.expiry} | {' / '.join(self.legs)}"
        if self.notes:
            summary += f" | notes={self.notes}"
        return summary


def _advance_trading_days(start_date: dt.date, sessions_ahead: int) -> dt.date:
    target = start_date
    remaining = max(0, sessions_ahead)
    while remaining > 0:
        target += dt.timedelta(days=1)
        if target.weekday() < 5:
            remaining -= 1
    while target.weekday() >= 5:
        target += dt.timedelta(days=1)
    return target


def _round_strike(value: float, step: float, mode: str) -> float:
    if step <= 0:
        step = 1.0
    units = value / step
    if mode == "up":
        return round(math.ceil(units) * step, 6)
    if mode == "down":
        return round(math.floor(units) * step, 6)
    return round(round(units) * step, 6)


def _format_strike(value: float) -> str:
    rounded = round(value, 6)
    if float(rounded).is_integer():
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def build_option_plan(signal: EmittedSignal, cfg: Optional[StrategyConfig] = None) -> Optional[OptionPlan]:
    cfg = cfg or StrategyConfig()
    step = cfg.option_strike_step if cfg.option_strike_step > 0 else 1.0
    local_date = signal.timestamp.tz_convert(require_toronto_tz()).date()

    if signal.signal == SignalType.TREND_LONG:
        expiry = _advance_trading_days(local_date, cfg.option_directional_expiry_days).isoformat()
        long_strike = _round_strike(signal.price, step, "down")
        target_price = float(signal.extras.get("target_price", signal.price + max(step * 2.0, signal.price * 0.003)))
        short_strike = _round_strike(target_price, step, "up")
        if short_strike <= long_strike:
            short_strike = long_strike + step
        stop_price = signal.extras.get("stop_price")
        notes = []
        if stop_price is not None:
            notes.append(f"invalidate_below={float(stop_price):.2f}")
        notes.append(f"target~{target_price:.2f}")
        return OptionPlan(
            structure="call_debit_spread",
            expiry=expiry,
            legs=[
                f"BUY 1 {signal.symbol} {expiry} {_format_strike(long_strike)} CALL",
                f"SELL 1 {signal.symbol} {expiry} {_format_strike(short_strike)} CALL",
            ],
            notes=", ".join(notes),
        )

    if signal.signal == SignalType.TREND_SHORT:
        expiry = _advance_trading_days(local_date, cfg.option_directional_expiry_days).isoformat()
        long_strike = _round_strike(signal.price, step, "up")
        target_price = float(signal.extras.get("target_price", signal.price - max(step * 2.0, signal.price * 0.003)))
        short_strike = _round_strike(target_price, step, "down")
        if short_strike >= long_strike:
            short_strike = long_strike - step
        stop_price = signal.extras.get("stop_price")
        notes = []
        if stop_price is not None:
            notes.append(f"invalidate_above={float(stop_price):.2f}")
        notes.append(f"target~{target_price:.2f}")
        return OptionPlan(
            structure="put_debit_spread",
            expiry=expiry,
            legs=[
                f"BUY 1 {signal.symbol} {expiry} {_format_strike(long_strike)} PUT",
                f"SELL 1 {signal.symbol} {expiry} {_format_strike(short_strike)} PUT",
            ],
            notes=", ".join(notes),
        )

    if signal.signal == SignalType.RANGE_BUTTERFLY_ZONE:
        expiry = _advance_trading_days(local_date, cfg.option_range_expiry_days).isoformat()
        center_price = float(signal.extras.get("range_center", signal.price))
        center_strike = _round_strike(center_price, step, "nearest")
        wing = cfg.option_range_wing_width if cfg.option_range_wing_width > 0 else step
        wing = max(step, wing)
        lower_strike = center_strike - wing
        upper_strike = center_strike + wing
        width_pct = signal.extras.get("range_width_pct")
        note = f"range_center~{center_price:.2f}"
        if width_pct is not None:
            note += f", width_pct={float(width_pct):.4f}"
        return OptionPlan(
            structure="call_butterfly",
            expiry=expiry,
            legs=[
                f"BUY 1 {signal.symbol} {expiry} {_format_strike(lower_strike)} CALL",
                f"SELL 2 {signal.symbol} {expiry} {_format_strike(center_strike)} CALL",
                f"BUY 1 {signal.symbol} {expiry} {_format_strike(upper_strike)} CALL",
            ],
            notes=note,
        )

    if signal.signal in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT):
        side = "bullish" if signal.signal == SignalType.EXIT_LONG else "bearish"
        return OptionPlan(
            structure="close_directional_spread",
            expiry="",
            legs=[f"CLOSE existing {side} exposure for {signal.symbol}"],
            notes="regime no longer supports the prior directional trade",
        )

    return None


class AlertSink:
    def __init__(
        self,
        discord_webhook_url: str = "",
        print_json: bool = False,
        csv_path: str = "signals.csv",
        console_output: bool = True,
        strategy_cfg: Optional[StrategyConfig] = None,
    ) -> None:
        self.discord_webhook_url = discord_webhook_url.strip()
        self.print_json = print_json
        self.csv_path = Path(csv_path).expanduser() if csv_path.strip() else None
        self.console_output = console_output
        self.strategy_cfg = strategy_cfg

    def send(self, s: EmittedSignal) -> None:
        local_dt = s.timestamp.tz_convert(require_toronto_tz())
        local_ts = local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        option_plan = build_option_plan(s, self.strategy_cfg)
        payload = {
            "symbol": s.symbol,
            "signal": s.signal.value,
            "regime": s.regime.value,
            "price": round(s.price, 4),
            "timestamp": local_ts,
            "reason": s.reason,
            "extras": s.extras,
        }
        if option_plan is not None:
            payload["option_plan"] = option_plan.to_payload()

        if self.console_output:
            if self.print_json:
                print(json.dumps(payload, ensure_ascii=False))
            else:
                try:
                    print(
                        f"[{local_ts}] {s.symbol} | {s.signal.value} | {s.regime.value} | "
                        f"price={s.price:.2f} | {s.reason} | extras={s.extras}"
                        + (f" | option={option_plan.summary()}" if option_plan is not None else "")
                    )
                except OSError:
                    pass

        if self.csv_path:
            self._append_csv(s, local_ts, option_plan)

        if self.discord_webhook_url:
            try:
                body = json.dumps(
                    {"content": "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"}
                ).encode("utf-8")
                request = urllib.request.Request(
                    self.discord_webhook_url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=5):
                    pass
            except Exception as exc:
                print(f"Discord alert failed: {exc}", file=sys.stderr)

    def _append_csv(self, s: EmittedSignal, local_ts: str, option_plan: Optional[OptionPlan]) -> None:
        assert self.csv_path is not None

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0

        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(
                    [
                        "timestamp_utc",
                        "timestamp_local",
                        "symbol",
                        "signal",
                        "regime",
                        "price",
                        "reason",
                        "extras_json",
                        "option_structure",
                        "option_expiry",
                        "option_legs",
                        "option_notes",
                    ]
                )
            writer.writerow(
                [
                    s.timestamp.tz_convert(UTC).isoformat(),
                    local_ts,
                    s.symbol,
                    s.signal.value,
                    s.regime.value,
                    round(s.price, 4),
                    s.reason,
                    json.dumps(s.extras, ensure_ascii=False, sort_keys=True),
                    option_plan.structure if option_plan is not None else "",
                    option_plan.expiry if option_plan is not None else "",
                    " ; ".join(option_plan.legs) if option_plan is not None else "",
                    option_plan.notes if option_plan is not None else "",
                ]
            )


class IndicatorEngine:
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
        ema_fast = IndicatorEngine.ema(close, fast)
        ema_slow = IndicatorEngine.ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})

    @staticmethod
    def stochastic_kdj(df: pd.DataFrame, period: int, smooth_k: int, smooth_d: int) -> pd.DataFrame:
        low_n = df["low"].rolling(period).min()
        high_n = df["high"].rolling(period).max()
        rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, math.nan) * 100.0
        k = rsv.rolling(smooth_k).mean()
        d = k.rolling(smooth_d).mean()
        j = 3 * k - 2 * d
        return pd.DataFrame({"k": k, "d": d, "j": j})

    @staticmethod
    def enrich(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
        out = df.copy()
        out["ema_fast"] = IndicatorEngine.ema(out["close"], cfg.ema_fast)
        out["ema_mid"] = IndicatorEngine.ema(out["close"], cfg.ema_mid)
        out["ema_slow"] = IndicatorEngine.ema(out["close"], cfg.ema_slow)

        macd_df = IndicatorEngine.macd(out["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        out = pd.concat([out, macd_df], axis=1)

        kdj_df = IndicatorEngine.stochastic_kdj(out, cfg.stoch_period, cfg.stoch_smooth_k, cfg.stoch_smooth_d)
        out = pd.concat([out, kdj_df], axis=1)

        out["vol_ma20"] = out["volume"].rolling(20).mean()
        out["volume_ratio"] = out["volume"] / out["vol_ma20"]
        out["trend_spread_pct"] = (out["ema_fast"] - out["ema_slow"]).abs() / out["close"]
        out["hl_range_pct"] = (out["high"] - out["low"]) / out["close"]
        out["mid_price"] = (out["high"] + out["low"] + out["close"]) / 3
        return out


class SignalEngine:
    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self.positions: Dict[str, PositionState] = defaultdict(PositionState)

    def _session_date(self, ts: pd.Timestamp) -> dt.date:
        return ts.tz_convert(require_toronto_tz()).date()

    def _minutes_from_open(self, ts: pd.Timestamp) -> int:
        local = ts.tz_convert(require_toronto_tz())
        open_dt = pd.Timestamp.combine(local.date(), MARKET_OPEN).tz_localize(require_toronto_tz())
        return int((local - open_dt).total_seconds() // 60)

    def _roll_session_state(self, pos: PositionState, ts: pd.Timestamp) -> None:
        session_date = self._session_date(ts)
        if pos.session_date == session_date:
            return
        pos.session_date = session_date
        pos.long_entries_today = 0
        pos.short_entries_today = 0
        pos.range_signals_today = 0

    def classify_regime(self, df_3m: pd.DataFrame, df_10m: pd.DataFrame) -> Regime:
        if len(df_3m) < 40 or len(df_10m) < 20:
            return Regime.NO_TRADE

        a = df_3m.iloc[-1]
        b = df_10m.iloc[-1]

        bull_alignment = a["ema_fast"] > a["ema_mid"] > a["ema_slow"] and b["ema_fast"] > b["ema_mid"] > b["ema_slow"]
        bear_alignment = a["ema_fast"] < a["ema_mid"] < a["ema_slow"] and b["ema_fast"] < b["ema_mid"] < b["ema_slow"]

        bull_momentum = a["macd_hist"] > 0 and a["k"] > a["d"] and a["j"] > a["d"]
        bear_momentum = a["macd_hist"] < 0 and a["k"] < a["d"] and a["j"] < a["d"]

        strong_spread = float(a["trend_spread_pct"]) >= self.cfg.min_trend_spread_pct
        weak_spread = float(a["trend_spread_pct"]) <= self.cfg.max_range_spread_pct

        if bull_alignment and bull_momentum and strong_spread:
            return Regime.TREND_UP
        if bear_alignment and bear_momentum and strong_spread:
            return Regime.TREND_DOWN
        if weak_spread:
            return Regime.RANGE
        return Regime.NO_TRADE

    def _cooldown_active(self, symbol: str, ts: pd.Timestamp) -> bool:
        last_ts = self.positions[symbol].last_signal_time
        if last_ts is None:
            return False
        return (ts - last_ts) < pd.Timedelta(minutes=self.cfg.cooldown_minutes)

    def _recent_swing_low(self, df: pd.DataFrame, lookback: int = 6) -> float:
        return float(df["low"].tail(lookback).min())

    def _recent_swing_high(self, df: pd.DataFrame, lookback: int = 6) -> float:
        return float(df["high"].tail(lookback).max())

    def _calc_range_center(self, df: pd.DataFrame) -> float:
        lookback = min(self.cfg.range_center_lookback, len(df))
        sub = df.tail(lookback)
        return float(sub["mid_price"].mean())

    def evaluate(self, symbol: str, df_1m: pd.DataFrame) -> Optional[EmittedSignal]:
        if len(df_1m) < 120:
            return None

        df_3m_raw = df_1m.resample(self.cfg.trend_fast_timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        df_10m_raw = df_1m.resample(self.cfg.trend_slow_timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        df_3m = IndicatorEngine.enrich(df_3m_raw, self.cfg)
        df_10m = IndicatorEngine.enrich(df_10m_raw, self.cfg)
        regime = self.classify_regime(df_3m, df_10m)

        last_1m = df_1m.iloc[-1]
        last_3m = df_3m.iloc[-1]
        ts = df_1m.index[-1]
        px = float(last_1m["close"])
        pos = self.positions[symbol]
        self._roll_session_state(pos, ts)
        minutes_from_open = self._minutes_from_open(ts)
        opening_range_ready = minutes_from_open >= self.cfg.opening_range_minutes
        session_df = df_1m[df_1m.index.tz_convert(require_toronto_tz()).date == self._session_date(ts)]
        local = ts.tz_convert(require_toronto_tz())
        opening_range_end = pd.Timestamp.combine(local.date(), MARKET_OPEN).tz_localize(require_toronto_tz()) + pd.Timedelta(
            minutes=self.cfg.opening_range_minutes
        )
        opening_range = session_df[session_df.index.tz_convert(require_toronto_tz()) < opening_range_end]
        opening_range_high = float(opening_range["high"].max()) if not opening_range.empty else math.nan
        opening_range_low = float(opening_range["low"].min()) if not opening_range.empty else math.nan

        if pos.side == "LONG":
            if pos.stop_price is not None and px <= pos.stop_price:
                pos.side = None
                pos.last_signal_time = ts
                return EmittedSignal(
                    symbol=symbol,
                    signal=SignalType.EXIT_LONG,
                    regime=regime,
                    price=px,
                    timestamp=ts,
                    reason="Long stop hit or trend failed.",
                    extras={"stop_price": round(pos.stop_price, 4)},
                )
            if pos.entry_time is not None and pos.entry_price is not None:
                held_minutes = (ts - pos.entry_time) / pd.Timedelta(minutes=1)
                progress_pct = (px - pos.entry_price) / pos.entry_price
                if held_minutes >= self.cfg.time_stop_minutes and progress_pct <= self.cfg.time_stop_min_progress_pct:
                    pos.side = None
                    pos.last_signal_time = ts
                    return EmittedSignal(
                        symbol=symbol,
                        signal=SignalType.EXIT_LONG,
                        regime=regime,
                        price=px,
                        timestamp=ts,
                        reason=f"Long exited on {self.cfg.time_stop_minutes}-minute time stop after insufficient progress.",
                        extras={"held_minutes": round(float(held_minutes), 1)},
                    )
            if px < float(last_3m["ema_fast"]) or regime == Regime.TREND_DOWN:
                pos.side = None
                pos.last_signal_time = ts
                return EmittedSignal(
                    symbol=symbol,
                    signal=SignalType.EXIT_LONG,
                    regime=regime,
                    price=px,
                    timestamp=ts,
                    reason="Long exited because price lost the 3-minute 5MA or the regime flipped bearish.",
                    extras={"ema_fast": round(float(last_3m["ema_fast"]), 4)},
                )

        if pos.side == "SHORT":
            if pos.stop_price is not None and px >= pos.stop_price:
                pos.side = None
                pos.last_signal_time = ts
                return EmittedSignal(
                    symbol=symbol,
                    signal=SignalType.EXIT_SHORT,
                    regime=regime,
                    price=px,
                    timestamp=ts,
                    reason="Short stop hit or trend failed.",
                    extras={"stop_price": round(pos.stop_price, 4)},
                )
            if pos.entry_time is not None and pos.entry_price is not None:
                held_minutes = (ts - pos.entry_time) / pd.Timedelta(minutes=1)
                progress_pct = (pos.entry_price - px) / pos.entry_price
                if held_minutes >= self.cfg.time_stop_minutes and progress_pct <= self.cfg.time_stop_min_progress_pct:
                    pos.side = None
                    pos.last_signal_time = ts
                    return EmittedSignal(
                        symbol=symbol,
                        signal=SignalType.EXIT_SHORT,
                        regime=regime,
                        price=px,
                        timestamp=ts,
                        reason=f"Short exited on {self.cfg.time_stop_minutes}-minute time stop after insufficient progress.",
                        extras={"held_minutes": round(float(held_minutes), 1)},
                    )
            if px > float(last_3m["ema_fast"]) or regime == Regime.TREND_UP:
                pos.side = None
                pos.last_signal_time = ts
                return EmittedSignal(
                    symbol=symbol,
                    signal=SignalType.EXIT_SHORT,
                    regime=regime,
                    price=px,
                    timestamp=ts,
                    reason="Short exited because price reclaimed the 3-minute 5MA or the regime flipped bullish.",
                    extras={"ema_fast": round(float(last_3m["ema_fast"]), 4)},
                )

        if self._cooldown_active(symbol, ts):
            return None

        if regime == Regime.TREND_UP and pos.side is None:
            if self.cfg.require_opening_range_breakout and (not opening_range_ready or px <= opening_range_high):
                return None
            if pos.long_entries_today >= self.cfg.max_same_direction_entries_per_session:
                return None
            price_near_fast = abs(px - float(last_3m["ema_fast"])) / px <= self.cfg.pullback_tolerance_pct
            momentum_ok = float(last_3m["macd_hist"]) > 0 and float(last_3m["volume_ratio"]) >= self.cfg.min_volume_ratio
            recent_low = self._recent_swing_low(df_3m)
            stop_price = recent_low * (1.0 - self.cfg.stop_below_recent_low_pct)
            risk = px - stop_price
            target = px + max(risk * self.cfg.min_rr, px * 0.003)

            if price_near_fast and momentum_ok and risk > 0:
                pos.side = "LONG"
                pos.entry_price = px
                pos.entry_time = ts
                pos.stop_price = stop_price
                pos.target_price = target
                pos.last_signal_time = ts
                pos.long_entries_today += 1
                return EmittedSignal(
                    symbol=symbol,
                    signal=SignalType.TREND_LONG,
                    regime=regime,
                    price=px,
                    timestamp=ts,
                    reason="Trend-up pullback to fast EMA with positive momentum/volume confirmation.",
                    extras={
                        "ema_fast": round(float(last_3m["ema_fast"]), 4),
                        "stop_price": round(stop_price, 4),
                        "target_price": round(target, 4),
                        "volume_ratio": round(float(last_3m["volume_ratio"]), 3),
                        "opening_range_high": round(opening_range_high, 4),
                    },
                )

        if regime == Regime.TREND_DOWN and pos.side is None:
            if self.cfg.require_opening_range_breakout and (not opening_range_ready or px >= opening_range_low):
                return None
            if pos.short_entries_today >= self.cfg.max_same_direction_entries_per_session:
                return None
            price_near_fast = abs(px - float(last_3m["ema_fast"])) / px <= self.cfg.pullback_tolerance_pct
            momentum_ok = float(last_3m["macd_hist"]) < 0 and float(last_3m["volume_ratio"]) >= self.cfg.min_volume_ratio
            recent_high = self._recent_swing_high(df_3m)
            stop_price = recent_high * (1.0 + self.cfg.stop_above_recent_high_pct)
            risk = stop_price - px
            target = px - max(risk * self.cfg.min_rr, px * 0.003)

            if price_near_fast and momentum_ok and risk > 0:
                pos.side = "SHORT"
                pos.entry_price = px
                pos.entry_time = ts
                pos.stop_price = stop_price
                pos.target_price = target
                pos.last_signal_time = ts
                pos.short_entries_today += 1
                return EmittedSignal(
                    symbol=symbol,
                    signal=SignalType.TREND_SHORT,
                    regime=regime,
                    price=px,
                    timestamp=ts,
                    reason="Trend-down pullback to fast EMA with negative momentum/volume confirmation.",
                    extras={
                        "ema_fast": round(float(last_3m["ema_fast"]), 4),
                        "stop_price": round(stop_price, 4),
                        "target_price": round(target, 4),
                        "volume_ratio": round(float(last_3m["volume_ratio"]), 3),
                        "opening_range_low": round(opening_range_low, 4),
                    },
                )

        if regime == Regime.RANGE:
            if not opening_range_ready or pos.range_signals_today >= self.cfg.max_range_signals_per_session:
                return None
            center = self._calc_range_center(df_3m)
            distance = abs(px - center) / px
            last20 = df_3m.tail(20)
            hi = float(last20["high"].max())
            lo = float(last20["low"].min())
            regime_width_pct = (hi - lo) / px
            if distance <= self.cfg.range_center_band_pct and regime_width_pct <= 0.01:
                self.positions[symbol].last_signal_time = ts
                self.positions[symbol].range_signals_today += 1
                return EmittedSignal(
                    symbol=symbol,
                    signal=SignalType.RANGE_BUTTERFLY_ZONE,
                    regime=regime,
                    price=px,
                    timestamp=ts,
                    reason="Price is near estimated range center; suitable for neutral structures like butterfly/iron fly review.",
                    extras={
                        "range_center": round(center, 4),
                        "range_width_pct": round(regime_width_pct, 4),
                        "opening_range_high": round(opening_range_high, 4),
                        "opening_range_low": round(opening_range_low, 4),
                    },
                )

        return None


class LiveBarStore:
    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self.frames: Dict[str, pd.DataFrame] = {}

    def upsert_bar(self, symbol: str, ts: pd.Timestamp, open_: float, high: float, low: float, close: float, volume: float) -> pd.DataFrame:
        frame = self.frames.get(symbol)
        idx = pd.DatetimeIndex([ts])
        row = pd.DataFrame(
            {
                "open": [open_],
                "high": [high],
                "low": [low],
                "close": [close],
                "volume": [volume],
            },
            index=idx,
        )
        row.index = row.index.tz_convert(UTC) if row.index.tz is not None else row.index.tz_localize(UTC)

        if frame is None:
            frame = row
        else:
            frame = pd.concat([frame, row])
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()

        if len(frame) > self.cfg.max_bars_kept:
            frame = frame.tail(self.cfg.max_bars_kept)

        self.frames[symbol] = frame
        return frame


class TradingSessionFilter:
    @staticmethod
    def is_regular_hours(ts: pd.Timestamp) -> bool:
        local = ts.tz_convert(require_toronto_tz())
        return MARKET_OPEN <= local.time() <= MARKET_CLOSE and local.weekday() < 5


class StrategyPipeline:
    def __init__(self, cfg: StrategyConfig, alerts: Optional[AlertSink] = None) -> None:
        self.cfg = cfg
        self.alerts = alerts or AlertSink(
            cfg.discord_webhook_url,
            cfg.print_json,
            cfg.signal_csv_path,
            strategy_cfg=cfg,
        )
        self.store = LiveBarStore(cfg)
        self.engine = SignalEngine(cfg)

    def store_bar(self, symbol: str, ts, open_: float, high: float, low: float, close: float, volume: float) -> Optional[pd.DataFrame]:
        bar_ts = coerce_utc_timestamp(ts)
        if not TradingSessionFilter.is_regular_hours(bar_ts):
            return None

        return self.store.upsert_bar(
            symbol=symbol,
            ts=bar_ts,
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )

    def process_bar(
        self,
        symbol: str,
        ts,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        emit_signals: bool = True,
    ) -> Optional[EmittedSignal]:
        df = self.store_bar(symbol, ts, open_, high, low, close, volume)
        if df is None or not emit_signals:
            return None

        emitted = self.engine.evaluate(symbol, df)
        if emitted:
            self.alerts.send(emitted)
        return emitted


def suggest_option_structure(signal: EmittedSignal) -> str:
    plan = build_option_plan(signal)
    if plan is None:
        return "No option structure suggestion."
    return plan.summary()
