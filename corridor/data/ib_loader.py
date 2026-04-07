from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import pandas as pd

from corridor.data.ib_contracts import build_underlying_contract


try:
    from ib_insync import IB, util
except ImportError:  # pragma: no cover - exercised only when ib_insync is missing
    IB = None
    util = None


@dataclass(slots=True)
class IBHistoricalRequest:
    symbol: str
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    bar_size: str = "5 mins"
    host: str = "127.0.0.1"
    port: int = 4001
    client_id: int = 41
    exchange: str = "SMART"
    currency: str = "USD"
    what_to_show: str = "TRADES"
    use_rth: bool = True
    chunk_duration: str = "30 D"


def _require_ib() -> None:
    if IB is None or util is None:
        raise RuntimeError("ib_insync is required for IB data loading. Install it or use --bars-csv.")


def _duration_to_timedelta(duration: str) -> timedelta:
    value, unit = duration.strip().split(maxsplit=1)
    amount = int(value)
    unit = unit.upper()
    if unit.startswith("D"):
        return timedelta(days=amount)
    if unit.startswith("W"):
        return timedelta(weeks=amount)
    if unit.startswith("M"):
        return timedelta(days=30 * amount)
    if unit.startswith("Y"):
        return timedelta(days=365 * amount)
    raise ValueError(f"Unsupported IB duration string: {duration}")


def _ensure_utc_timestamp(value: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _format_ib_history_error(errors: list[tuple[int, str]], context: str) -> str:
    if not errors:
        return f"IB returned no historical bars for {context}."

    code, message = errors[-1]
    cleaned = message.strip()
    if code == 162 and "different IP address" in cleaned:
        return (
            f"IB historical bars unavailable for {context}: {cleaned} "
            "Close the other trading TWS/Gateway session or reconnect from the same IP."
        )
    return f"IB historical bars unavailable for {context}: [{code}] {cleaned}"


def fetch_intraday_bars(req: IBHistoricalRequest) -> pd.DataFrame:
    """Fetch IBKR historical bars in backwards chunks and return a normalized frame."""

    _require_ib()

    now_utc = _ensure_utc_timestamp(pd.Timestamp.utcnow())
    end = _ensure_utc_timestamp(req.end) if req.end is not None else now_utc
    start = _ensure_utc_timestamp(req.start) if req.start is not None else end - pd.Timedelta(days=30)
    chunk_span = _duration_to_timedelta(req.chunk_duration)
    contract = build_underlying_contract(req.symbol.upper(), req.exchange, req.currency)

    ib = IB()
    ib.connect(req.host, req.port, clientId=req.client_id, timeout=10)
    try:
        ib.qualifyContracts(contract)
        frames: list[pd.DataFrame] = []
        cursor = end
        request_counter = 0

        while cursor > start:
            request_counter += 1
            errors: list[tuple[int, str]] = []

            def capture_error(_req_id: int, error_code: int, error_string: str, error_contract) -> None:
                if error_contract is None or getattr(error_contract, "symbol", None) == req.symbol.upper():
                    errors.append((error_code, error_string))

            ib.errorEvent += capture_error
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=cursor.to_pydatetime(),
                    durationStr=req.chunk_duration,
                    barSizeSetting=req.bar_size,
                    whatToShow=req.what_to_show,
                    useRTH=req.use_rth,
                    formatDate=1,
                )
            finally:
                ib.errorEvent -= capture_error

            if not bars:
                if not frames:
                    raise RuntimeError(
                        _format_ib_history_error(
                            errors,
                            f"{req.symbol.upper()} ({req.chunk_duration}, {req.bar_size})",
                        )
                    )
                break

            frame = util.df(bars)
            if frame is None or frame.empty:
                if not frames:
                    raise RuntimeError(
                        _format_ib_history_error(
                            errors,
                            f"{req.symbol.upper()} ({req.chunk_duration}, {req.bar_size})",
                        )
                    )
                break

            frame["timestamp"] = pd.to_datetime(frame["date"], utc=True)
            frame["symbol"] = req.symbol.upper()
            frame = frame.rename(
                columns={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]

            kept = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].copy()
            frames.append(kept)
            earliest = frame["timestamp"].min()
            if pd.isna(earliest) or earliest >= cursor:
                break
            cursor = earliest - pd.Timedelta(minutes=1)
            if cursor <= start:
                break
            if request_counter > math.ceil((end - start) / pd.Timedelta(chunk_span)):
                break

        if not frames:
            raise RuntimeError("IB returned no historical bars.")

        merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp", "symbol"]).sort_values("timestamp")
        return merged.reset_index(drop=True)
    finally:
        ib.disconnect()
