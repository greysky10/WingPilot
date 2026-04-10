from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(slots=True)
class HistoricalLoadConfig:
    csv_path: Path
    symbol: Optional[str] = None
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"


def _ensure_utc(ts: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(ts, utc=True, errors="raise")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize("UTC")
    return parsed.dt.tz_convert("UTC")


def _ensure_boundary_utc(value: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def load_intraday_bars(cfg: HistoricalLoadConfig) -> pd.DataFrame:
    """Load normalized intraday OHLCV bars from CSV."""

    path = Path(cfg.csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Historical bars CSV not found: {path}")

    frame = pd.read_csv(path)
    required = {
        cfg.timestamp_col,
        cfg.open_col,
        cfg.high_col,
        cfg.low_col,
        cfg.close_col,
        cfg.volume_col,
    }
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required historical columns: {', '.join(sorted(missing))}")

    if cfg.symbol_col not in frame.columns:
        if not cfg.symbol:
            raise ValueError("CSV has no symbol column and no symbol override was provided.")
        frame[cfg.symbol_col] = cfg.symbol.upper()

    frame = frame.rename(
        columns={
            cfg.timestamp_col: "timestamp",
            cfg.symbol_col: "symbol",
            cfg.open_col: "open",
            cfg.high_col: "high",
            cfg.low_col: "low",
            cfg.close_col: "close",
            cfg.volume_col: "volume",
        }
    )
    frame["timestamp"] = _ensure_utc(frame["timestamp"])
    frame["symbol"] = frame["symbol"].astype(str).str.upper()

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).sort_values("timestamp")

    start = _ensure_boundary_utc(cfg.start)
    end = _ensure_boundary_utc(cfg.end)
    if cfg.symbol:
        frame = frame[frame["symbol"] == cfg.symbol.upper()]
    if start is not None:
        frame = frame[frame["timestamp"] >= start]
    if end is not None:
        frame = frame[frame["timestamp"] <= end]

    if frame.empty:
        raise ValueError("Historical bars frame is empty after filtering.")

    return frame.reset_index(drop=True)
