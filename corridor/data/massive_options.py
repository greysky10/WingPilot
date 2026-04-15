from __future__ import annotations

import json
import math
import random
import socket
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib import error, parse, request
from zoneinfo import ZoneInfo

import pandas as pd

from corridor.data.historical_loader import HistoricalLoadConfig, load_intraday_bars


MASSIVE_API_BASE_URL = "https://api.massive.com"
MASSIVE_CONTRACTS_PATH = "/v3/reference/options/contracts"
MASSIVE_DAILY_BARS_PATH = "/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
DEFAULT_CONTRACT_UNDERLYINGS = ("I:SPX", "SPX")
OUTPUT_FORMATS = {"csv", "parquet"}
VALID_CONTRACT_TYPES = {"call", "put"}
VALID_WING_MODES = {"symmetric", "broken_upper", "broken_lower", "adaptive"}
NEW_YORK_TZ = ZoneInfo("America/New_York")
NORMALIZED_DATASET_COLUMNS = [
    "date",
    "option_ticker",
    "underlying",
    "expiry",
    "strike",
    "type",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "transactions",
    "vwap",
]


class MassiveAPIError(RuntimeError):
    """Raised when Massive returns an API-level or HTTP-level error."""


class MassiveAuthorizationError(MassiveAPIError):
    """Raised when the current Massive key is not entitled for a request."""


@dataclass(slots=True)
class MassiveClientConfig:
    api_key: str
    base_url: str = MASSIVE_API_BASE_URL
    timeout_seconds: float = 30.0
    max_retries: int = 5
    max_rate_limit_retries: int = 100
    retry_backoff_seconds: float = 1.0
    min_request_interval_seconds: float = 0.0
    rate_limit_sleep_seconds: float = 70.0
    user_agent: str = "DaySpyMassiveSPXBackfill/1.0"


@dataclass(slots=True)
class StrategyUniverseConfig:
    bars_csv: Path
    symbol: str = "SPX"
    contract_types: tuple[str, ...] = ("call",)
    dte_min: int = 4
    dte_max: int = 10
    center_rounding: float = 5.0
    butterfly_width: float = 10.0
    wing_mode: str = "symmetric"
    broken_wing_extra_width: float = 0.0
    strike_buffer_points: float = 10.0
    slice_days: int = 7

    def __post_init__(self) -> None:
        self.bars_csv = Path(self.bars_csv)
        self.symbol = str(self.symbol).upper().strip()
        normalized_types = tuple(
            item.strip().lower() for item in self.contract_types if item and str(item).strip()
        )
        if not normalized_types:
            raise ValueError("Strategy universe requires at least one contract type.")
        invalid_types = sorted({item for item in normalized_types if item not in VALID_CONTRACT_TYPES})
        if invalid_types:
            raise ValueError(
                f"Unsupported contract types for strategy universe: {', '.join(invalid_types)}. "
                f"Use one of {sorted(VALID_CONTRACT_TYPES)}."
            )
        self.contract_types = normalized_types
        self.dte_min = max(0, int(self.dte_min))
        self.dte_max = max(self.dte_min, int(self.dte_max))
        self.center_rounding = max(0.5, float(self.center_rounding))
        self.butterfly_width = max(0.5, float(self.butterfly_width))
        self.wing_mode = str(self.wing_mode or "symmetric").lower()
        if self.wing_mode not in VALID_WING_MODES:
            raise ValueError(
                f"Unsupported wing_mode {self.wing_mode!r}. Use one of {sorted(VALID_WING_MODES)}."
            )
        self.broken_wing_extra_width = max(0.0, float(self.broken_wing_extra_width))
        self.strike_buffer_points = max(0.0, float(self.strike_buffer_points))
        self.slice_days = max(1, int(self.slice_days))


@dataclass(slots=True)
class StrategyQueryWindow:
    trade_start: date
    trade_end: date
    expiration_start: date
    expiration_end: date
    strike_price_gte: float
    strike_price_lte: float


@dataclass(slots=True)
class MassiveBackfillConfig:
    start_date: date
    end_date: date
    output_dir: Path
    contract_underlyings: tuple[str, ...] = DEFAULT_CONTRACT_UNDERLYINGS
    output_format: str = "parquet"
    contract_page_size: int = 1000
    bars_limit: int = 50000
    batch_size: int = 100
    expiration_buffer_days: int = 365
    contract_limit: Optional[int] = None
    resume: bool = True
    strategy_universe: Optional[StrategyUniverseConfig] = None

    def __post_init__(self) -> None:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date.")
        self.output_dir = Path(self.output_dir)
        self.contract_underlyings = tuple(
            item.strip() for item in self.contract_underlyings if item and item.strip()
        )
        if not self.contract_underlyings:
            raise ValueError("At least one contract underlying must be provided.")
        normalized_format = self.output_format.strip().lower()
        if normalized_format not in OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported output_format {self.output_format!r}. Use one of {sorted(OUTPUT_FORMATS)}."
            )
        self.output_format = normalized_format
        self.contract_page_size = max(1, min(1000, int(self.contract_page_size)))
        self.bars_limit = max(1, min(50000, int(self.bars_limit)))
        self.batch_size = max(1, int(self.batch_size))
        self.expiration_buffer_days = max(0, int(self.expiration_buffer_days))
        if self.contract_limit is not None:
            self.contract_limit = max(1, int(self.contract_limit))


@dataclass(slots=True)
class BackfillCheckpoint:
    contract_underlyings: list[str]
    selected_contract_underlying: str = ""
    start_date: str = ""
    end_date: str = ""
    output_format: str = "parquet"
    contracts_path: str = ""
    final_dataset_path: str = ""
    batch_index: int = 0
    completed_tickers: list[str] = field(default_factory=list)
    completed_parts: list[str] = field(default_factory=list)
    last_updated_utc: str = ""
    strategy_signature: str = ""


@dataclass(slots=True)
class MassiveBackfillResult:
    selected_contract_underlying: str
    contracts_path: Path
    final_dataset_path: Path
    part_count: int
    contract_count: int
    row_count: int
    output_format: str


class MassiveRESTClient:
    def __init__(self, cfg: MassiveClientConfig) -> None:
        if not cfg.api_key.strip():
            raise ValueError("Massive API key cannot be empty.")
        self.cfg = cfg
        self._last_request_monotonic: Optional[float] = None

    def iter_paginated(
        self,
        path: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Iterable[dict[str, Any]]:
        next_url: Optional[str] = self._build_url(path, params)
        while next_url:
            payload = self.request_json(next_url)
            results = payload.get("results", [])
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        yield item
            next_url = payload.get("next_url")

    def request_json(
        self,
        path_or_url: str,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        url = self._build_url(path_or_url, params)
        attempts = 0
        rate_limit_attempts = 0
        last_error: Optional[Exception] = None
        while attempts < self.cfg.max_retries:
            try:
                self._throttle_if_needed()
                return self._request_once(url)
            except MassiveAuthorizationError:
                raise
            except MassiveAPIError as exc:
                last_error = exc
                if self._is_rate_limit_message(str(exc)):
                    rate_limit_attempts += 1
                    if rate_limit_attempts > self.cfg.max_rate_limit_retries:
                        raise
                    print(
                        f"Massive rate limit hit for {url}. "
                        f"Sleeping {self.cfg.rate_limit_sleep_seconds:.0f}s before retry {rate_limit_attempts}."
                    )
                    self._record_request_time()
                    time.sleep(self.cfg.rate_limit_sleep_seconds + random.uniform(0.0, 1.0))
                    continue
                attempts += 1
                if attempts >= self.cfg.max_retries:
                    raise
            except (error.URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                attempts += 1
                if attempts >= self.cfg.max_retries:
                    break
            self._record_request_time()
            sleep_seconds = self.cfg.retry_backoff_seconds * (2 ** (attempts - 1))
            time.sleep(sleep_seconds + random.uniform(0.0, 0.25))
        if last_error is not None:
            raise MassiveAPIError(f"Massive request failed after {attempts} attempts: {last_error}") from last_error
        raise MassiveAPIError("Massive request failed before any attempt completed.")

    def _request_once(self, url: str) -> dict[str, Any]:
        req = request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Accept": "application/json",
                "User-Agent": self.cfg.user_agent,
            },
        )
        try:
            with request.urlopen(req, timeout=self.cfg.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = self._extract_http_error_message(body) or f"HTTP {exc.code}"
            if exc.code in {401, 403}:
                raise MassiveAuthorizationError(message) from exc
            if exc.code in {408, 425, 429, 500, 502, 503, 504}:
                raise MassiveAPIError(message) from exc
            raise MassiveAPIError(message) from exc

        status = str(payload.get("status", "")).upper()
        if status in {"NOT_AUTHORIZED", "UNAUTHORIZED"}:
            raise MassiveAuthorizationError(
                str(payload.get("message") or payload.get("error") or "Massive authorization failed.")
            )
        if status and status not in {"OK", "DELAYED"}:
            raise MassiveAPIError(
                str(payload.get("message") or payload.get("error") or f"Massive returned status {status}.")
            )
        self._record_request_time()
        return payload

    def _build_url(self, path_or_url: str, params: Optional[dict[str, Any]] = None) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            if params:
                parsed = parse.urlparse(path_or_url)
                existing = dict(parse.parse_qsl(parsed.query, keep_blank_values=True))
                existing.update(
                    {
                        key: self._coerce_query_value(value)
                        for key, value in params.items()
                        if value is not None
                    }
                )
                query = parse.urlencode(existing)
                return parse.urlunparse(parsed._replace(query=query))
            return path_or_url

        base = self.cfg.base_url.rstrip("/")
        url = f"{base}/{path_or_url.lstrip('/')}"
        if not params:
            return url
        query = parse.urlencode(
            {key: self._coerce_query_value(value) for key, value in params.items() if value is not None},
            doseq=True,
        )
        return f"{url}?{query}"

    @staticmethod
    def _coerce_query_value(value: Any) -> Any:
        if isinstance(value, bool):
            return str(value).lower()
        return value

    @staticmethod
    def _extract_http_error_message(body: str) -> str:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return body.strip()
        if isinstance(payload, dict):
            message = payload.get("message") or payload.get("error")
            if message:
                return str(message)
        return body.strip()

    def _throttle_if_needed(self) -> None:
        interval = max(0.0, float(self.cfg.min_request_interval_seconds))
        if interval <= 0 or self._last_request_monotonic is None:
            return
        elapsed = time.monotonic() - self._last_request_monotonic
        if elapsed < interval:
            time.sleep(interval - elapsed)

    def _record_request_time(self) -> None:
        self._last_request_monotonic = time.monotonic()

    @staticmethod
    def _is_rate_limit_message(message: str) -> bool:
        lowered = str(message).lower()
        return "maximum requests per minute" in lowered or "rate limit" in lowered or "too many requests" in lowered


def backfill_massive_spx_options(
    client: MassiveRESTClient,
    cfg: MassiveBackfillConfig,
) -> MassiveBackfillResult:
    mode_label = "strategy-only" if cfg.strategy_universe is not None else "full-chain"
    print(
        f"Starting Massive SPX options backfill for {cfg.start_date.isoformat()} through {cfg.end_date.isoformat()} "
        f"into {cfg.output_dir} ({mode_label})."
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = cfg.output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.output_dir / "checkpoint.json"

    output_format = resolve_output_format(cfg.output_format)
    if output_format != cfg.output_format:
        print("Parquet engine not available. Falling back to CSV output.")

    checkpoint = _load_checkpoint(checkpoint_path) if cfg.resume else None
    if checkpoint is not None:
        _validate_checkpoint(checkpoint, cfg, output_format)
    else:
        checkpoint = BackfillCheckpoint(
            contract_underlyings=list(cfg.contract_underlyings),
            start_date=cfg.start_date.isoformat(),
            end_date=cfg.end_date.isoformat(),
            output_format=output_format,
            strategy_signature=_strategy_signature(cfg),
        )

    contracts_path = _resolve_contracts_path(cfg.output_dir, output_format)
    final_dataset_path = _resolve_final_dataset_path(cfg.output_dir, output_format)
    checkpoint.contracts_path = str(contracts_path)
    checkpoint.final_dataset_path = str(final_dataset_path)
    checkpoint.strategy_signature = _strategy_signature(cfg)

    contracts_frame, selected_underlying = _prepare_contracts_frame(
        client,
        cfg,
        contracts_path,
        checkpoint,
        output_format,
    )
    checkpoint.selected_contract_underlying = selected_underlying
    print(f"Using contract discovery underlying {selected_underlying}. Contracts loaded: {len(contracts_frame)}")

    completed_tickers = set(checkpoint.completed_tickers)
    pending_frame = contracts_frame[~contracts_frame["ticker"].isin(completed_tickers)].copy()
    print(f"Contracts already completed: {len(completed_tickers)}. Remaining: {len(pending_frame)}")
    batch_frames: list[pd.DataFrame] = []
    completed_batch_tickers: list[str] = []

    for _, contract_row in pending_frame.iterrows():
        ticker = str(contract_row["ticker"])
        history_window = _resolve_contract_history_window(contract_row, cfg)
        if history_window is None:
            print(f"Skipping {ticker} because it has no remaining strategy-relevant trade dates.")
            batch_frames.append(pd.DataFrame(columns=NORMALIZED_DATASET_COLUMNS))
            completed_batch_tickers.append(ticker)
        else:
            window_start, window_end = history_window
            bars_frame = fetch_option_daily_bars(
                client=client,
                option_ticker=ticker,
                start_date=window_start,
                end_date=window_end,
                limit=cfg.bars_limit,
            )
            normalized = normalize_option_bars(contract_row, bars_frame)
            batch_frames.append(normalized)
            completed_batch_tickers.append(ticker)

        if len(completed_batch_tickers) >= cfg.batch_size:
            _flush_batch(parts_dir, batch_frames, completed_batch_tickers, checkpoint, checkpoint_path, output_format)
            batch_frames = []
            completed_batch_tickers = []

    if completed_batch_tickers:
        _flush_batch(parts_dir, batch_frames, completed_batch_tickers, checkpoint, checkpoint_path, output_format)

    combined = assemble_final_dataset(parts_dir, output_format)
    write_dataframe(combined, final_dataset_path, output_format)

    checkpoint.final_dataset_path = str(final_dataset_path)
    checkpoint.last_updated_utc = pd.Timestamp.utcnow().isoformat()
    _save_checkpoint(checkpoint_path, checkpoint)

    return MassiveBackfillResult(
        selected_contract_underlying=selected_underlying,
        contracts_path=contracts_path,
        final_dataset_path=final_dataset_path,
        part_count=len(list(parts_dir.glob(f"*.{output_format}"))),
        contract_count=len(contracts_frame),
        row_count=len(combined),
        output_format=output_format,
    )


def fetch_contracts_for_underlying(
    client: MassiveRESTClient,
    underlying: str,
    start_date: date,
    end_date: date,
    page_size: int = 1000,
    expiration_buffer_days: int = 365,
    contract_type: Optional[str] = None,
    strike_price_gte: Optional[float] = None,
    strike_price_lte: Optional[float] = None,
) -> pd.DataFrame:
    expiration_end = end_date + timedelta(days=expiration_buffer_days)
    params: dict[str, Any] = {
        "underlying_ticker": underlying,
        "expired": True,
        "limit": page_size,
        "order": "asc",
        "sort": "expiration_date",
        "expiration_date.gte": start_date.isoformat(),
        "expiration_date.lte": expiration_end.isoformat(),
    }
    if contract_type is not None:
        params["contract_type"] = str(contract_type).lower()
    if strike_price_gte is not None:
        params["strike_price.gte"] = float(strike_price_gte)
    if strike_price_lte is not None:
        params["strike_price.lte"] = float(strike_price_lte)

    rows = list(client.iter_paginated(MASSIVE_CONTRACTS_PATH, params))
    if not rows:
        return pd.DataFrame(
            columns=["ticker", "underlying_ticker", "contract_type", "expiration_date", "strike_price"]
        )

    frame = pd.DataFrame.from_records(rows)
    required = ["ticker", "underlying_ticker", "contract_type", "expiration_date", "strike_price"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise MassiveAPIError(f"Massive contract response is missing required fields: {', '.join(sorted(missing))}")

    frame = frame.copy()
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"], errors="coerce").dt.date.astype("string")
    frame["strike_price"] = pd.to_numeric(frame["strike_price"], errors="coerce")
    frame["contract_type"] = frame["contract_type"].astype(str).str.lower()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["underlying_ticker"] = frame["underlying_ticker"].astype(str)

    frame = frame.dropna(
        subset=["ticker", "underlying_ticker", "contract_type", "expiration_date", "strike_price"]
    )
    frame = frame.drop_duplicates(subset=["ticker"]).sort_values(
        ["expiration_date", "strike_price", "contract_type", "ticker"]
    )
    return frame.reset_index(drop=True)


def fetch_option_daily_bars(
    client: MassiveRESTClient,
    option_ticker: str,
    start_date: date,
    end_date: date,
    limit: int = 50000,
) -> pd.DataFrame:
    path = MASSIVE_DAILY_BARS_PATH.format(
        ticker=parse.quote(option_ticker, safe=":"),
        start=start_date.isoformat(),
        end=end_date.isoformat(),
    )
    payload = client.request_json(path, {"sort": "asc", "limit": limit})
    results = payload.get("results", [])
    if not results:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v", "n", "vw", "t"])
    return pd.DataFrame.from_records(results)


def normalize_option_bars(
    contract_row: pd.Series | dict[str, Any],
    bars_frame: pd.DataFrame,
) -> pd.DataFrame:
    if bars_frame.empty:
        return pd.DataFrame(columns=NORMALIZED_DATASET_COLUMNS)

    contract = dict(contract_row)
    frame = bars_frame.copy()
    frame["t"] = pd.to_numeric(frame["t"], errors="coerce")
    frame = frame.dropna(subset=["t"])
    timestamps = pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert(NEW_YORK_TZ)

    normalized = pd.DataFrame(
        {
            "date": timestamps.dt.strftime("%Y-%m-%d"),
            "option_ticker": str(contract["ticker"]),
            "underlying": str(contract["underlying_ticker"]),
            "expiry": str(contract["expiration_date"]),
            "strike": pd.to_numeric(contract["strike_price"]),
            "type": str(contract["contract_type"]).lower(),
            "open": pd.to_numeric(frame.get("o"), errors="coerce"),
            "high": pd.to_numeric(frame.get("h"), errors="coerce"),
            "low": pd.to_numeric(frame.get("l"), errors="coerce"),
            "close": pd.to_numeric(frame.get("c"), errors="coerce"),
            "volume": pd.to_numeric(frame.get("v"), errors="coerce").fillna(0),
            "transactions": pd.to_numeric(frame.get("n"), errors="coerce"),
            "vwap": pd.to_numeric(frame.get("vw"), errors="coerce"),
        }
    )
    normalized = normalized.dropna(
        subset=[
            "date",
            "option_ticker",
            "underlying",
            "expiry",
            "strike",
            "type",
            "open",
            "high",
            "low",
            "close",
        ]
    )
    normalized = normalized.drop_duplicates(subset=["date", "option_ticker"]).sort_values(
        ["date", "expiry", "strike", "type", "option_ticker"]
    )
    return normalized.reset_index(drop=True)


def assemble_final_dataset(parts_dir: Path, output_format: str) -> pd.DataFrame:
    part_paths = sorted(parts_dir.glob(f"*.{output_format}"))
    if not part_paths:
        return pd.DataFrame(columns=NORMALIZED_DATASET_COLUMNS)
    frames = [read_dataframe(path, output_format) for path in part_paths]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=NORMALIZED_DATASET_COLUMNS)
    if merged.empty:
        return pd.DataFrame(columns=NORMALIZED_DATASET_COLUMNS)
    merged = merged.drop_duplicates(subset=["date", "option_ticker"]).sort_values(
        ["date", "expiry", "strike", "type", "option_ticker"]
    )
    return merged.reset_index(drop=True)


def resolve_output_format(requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized != "parquet":
        return normalized
    try:
        import pyarrow  # noqa: F401

        return "parquet"
    except ModuleNotFoundError:
        try:
            import fastparquet  # type: ignore  # noqa: F401

            return "parquet"
        except ModuleNotFoundError:
            return "csv"


def write_dataframe(frame: pd.DataFrame, path: Path, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "parquet":
        frame.to_parquet(path, index=False)
        return
    frame.to_csv(path, index=False)


def read_dataframe(path: Path, output_format: str) -> pd.DataFrame:
    if output_format == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _prepare_contracts_frame(
    client: MassiveRESTClient,
    cfg: MassiveBackfillConfig,
    contracts_path: Path,
    checkpoint: BackfillCheckpoint,
    output_format: str,
) -> tuple[pd.DataFrame, str]:
    if cfg.resume and contracts_path.exists():
        stored = read_dataframe(contracts_path, output_format)
        selected = checkpoint.selected_contract_underlying or str(stored["underlying_ticker"].iloc[0])
        print(f"Resuming from cached contracts file {contracts_path}.")
        return stored.reset_index(drop=True), selected

    if cfg.strategy_universe is not None:
        return _prepare_strategy_contracts_frame(client, cfg, contracts_path, checkpoint, output_format)

    for underlying in cfg.contract_underlyings:
        print(f"Fetching contracts for underlying {underlying}...")
        frame = fetch_contracts_for_underlying(
            client=client,
            underlying=underlying,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            page_size=cfg.contract_page_size,
            expiration_buffer_days=cfg.expiration_buffer_days,
        )
        if frame.empty:
            print(f"No contracts returned for underlying {underlying}.")
            continue
        if cfg.contract_limit is not None:
            frame = frame.head(cfg.contract_limit).copy()
        write_dataframe(frame, contracts_path, output_format)
        print(f"Saved {len(frame)} contracts to {contracts_path}.")
        checkpoint.selected_contract_underlying = underlying
        checkpoint.contracts_path = str(contracts_path)
        checkpoint.last_updated_utc = pd.Timestamp.utcnow().isoformat()
        return frame.reset_index(drop=True), underlying

    joined = ", ".join(cfg.contract_underlyings)
    raise MassiveAPIError(
        f"Massive returned no option contracts for the requested underlyings: {joined}. "
        "Check the symbol mapping and your plan entitlements."
    )


def _prepare_strategy_contracts_frame(
    client: MassiveRESTClient,
    cfg: MassiveBackfillConfig,
    contracts_path: Path,
    checkpoint: BackfillCheckpoint,
    output_format: str,
) -> tuple[pd.DataFrame, str]:
    strategy_cfg = cfg.strategy_universe
    if strategy_cfg is None:
        raise ValueError("Strategy universe config is required for strategy-only contract preparation.")

    daily_underlying = _load_strategy_underlying_daily_frame(strategy_cfg, cfg.start_date, cfg.end_date)
    query_windows = _build_strategy_query_windows(daily_underlying, strategy_cfg)
    print(
        f"Strategy-only universe loaded {len(daily_underlying)} trade dates from {strategy_cfg.bars_csv}. "
        f"Query windows: {len(query_windows)}."
    )

    for underlying in cfg.contract_underlyings:
        print(f"Fetching strategy-only contracts for underlying {underlying}...")
        window_frames: list[pd.DataFrame] = []
        for index, window in enumerate(query_windows, start=1):
            print(
                f"  Window {index}/{len(query_windows)} | "
                f"trade_dates={window.trade_start.isoformat()}..{window.trade_end.isoformat()} | "
                f"expiries={window.expiration_start.isoformat()}..{window.expiration_end.isoformat()} | "
                f"strikes={window.strike_price_gte:.2f}..{window.strike_price_lte:.2f}"
            )
            for contract_type in strategy_cfg.contract_types:
                frame = fetch_contracts_for_underlying(
                    client=client,
                    underlying=underlying,
                    start_date=window.expiration_start,
                    end_date=window.expiration_end,
                    page_size=cfg.contract_page_size,
                    expiration_buffer_days=0,
                    contract_type=contract_type,
                    strike_price_gte=window.strike_price_gte,
                    strike_price_lte=window.strike_price_lte,
                )
                if not frame.empty:
                    window_frames.append(frame)

        if not window_frames:
            print(f"No strategy-only contracts returned for underlying {underlying}.")
            continue

        frame = pd.concat(window_frames, ignore_index=True)
        frame = frame.drop_duplicates(subset=["ticker"]).sort_values(
            ["expiration_date", "strike_price", "contract_type", "ticker"]
        )
        filtered = _filter_contracts_for_strategy(frame, daily_underlying, strategy_cfg)
        if filtered.empty:
            print(f"Strategy-only post-filter removed every contract for underlying {underlying}.")
            continue
        if cfg.contract_limit is not None:
            filtered = filtered.head(cfg.contract_limit).copy()
        write_dataframe(filtered, contracts_path, output_format)
        print(f"Saved {len(filtered)} strategy-only contracts to {contracts_path}.")
        checkpoint.selected_contract_underlying = underlying
        checkpoint.contracts_path = str(contracts_path)
        checkpoint.last_updated_utc = pd.Timestamp.utcnow().isoformat()
        return filtered.reset_index(drop=True), underlying

    joined = ", ".join(cfg.contract_underlyings)
    raise MassiveAPIError(
        f"Massive returned no strategy-only option contracts for the requested underlyings: {joined}. "
        "Reduce slice size or widen the strike buffer if the filter is too narrow."
    )


def _load_strategy_underlying_daily_frame(
    strategy_cfg: StrategyUniverseConfig,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if not strategy_cfg.bars_csv.exists():
        raise FileNotFoundError(
            f"Strategy-only contract discovery requires an underlying bars CSV. Not found: {strategy_cfg.bars_csv}"
        )

    frame = load_intraday_bars(
        HistoricalLoadConfig(
            csv_path=strategy_cfg.bars_csv,
            symbol=strategy_cfg.symbol,
            start=_local_day_boundary_utc(start_date),
            end=_local_day_boundary_utc(end_date + timedelta(days=1)),
        )
    )
    localized = frame["timestamp"].dt.tz_convert(NEW_YORK_TZ)
    daily = (
        frame.assign(trade_date=localized.dt.date)
        .groupby("trade_date", as_index=False)
        .agg(
            session_open=("open", "first"),
            session_high=("high", "max"),
            session_low=("low", "min"),
            session_close=("close", "last"),
        )
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    if daily.empty:
        raise ValueError("Strategy-only contract discovery found no underlying trade dates after loading bars.")

    reach = _strategy_reach_points(strategy_cfg)
    round_to = max(0.5, float(strategy_cfg.center_rounding))
    daily["query_strike_min"] = daily["session_low"].map(
        lambda value: _round_down(float(value) - reach - strategy_cfg.strike_buffer_points, round_to)
    )
    daily["query_strike_max"] = daily["session_high"].map(
        lambda value: _round_up(float(value) + reach + strategy_cfg.strike_buffer_points, round_to)
    )
    return daily


def _build_strategy_query_windows(
    daily_underlying: pd.DataFrame,
    strategy_cfg: StrategyUniverseConfig,
) -> list[StrategyQueryWindow]:
    if daily_underlying.empty:
        return []
    ordered = daily_underlying.sort_values("trade_date").reset_index(drop=True)
    windows: list[StrategyQueryWindow] = []
    for start_index in range(0, len(ordered), strategy_cfg.slice_days):
        slice_frame = ordered.iloc[start_index : start_index + strategy_cfg.slice_days].copy()
        trade_start = slice_frame["trade_date"].iloc[0]
        trade_end = slice_frame["trade_date"].iloc[-1]
        windows.append(
            StrategyQueryWindow(
                trade_start=trade_start,
                trade_end=trade_end,
                expiration_start=trade_start + timedelta(days=strategy_cfg.dte_min),
                expiration_end=trade_end + timedelta(days=strategy_cfg.dte_max),
                strike_price_gte=float(slice_frame["query_strike_min"].min()),
                strike_price_lte=float(slice_frame["query_strike_max"].max()),
            )
        )
    return windows


def _filter_contracts_for_strategy(
    frame: pd.DataFrame,
    daily_underlying: pd.DataFrame,
    strategy_cfg: StrategyUniverseConfig,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    contract_types = set(strategy_cfg.contract_types)
    kept_frames: list[pd.DataFrame] = []
    for expiry_value, expiry_group in frame.groupby("expiration_date", sort=False):
        expiry_ts = pd.to_datetime(expiry_value, errors="coerce")
        if pd.isna(expiry_ts):
            continue
        earliest_trade = expiry_ts.date() - timedelta(days=strategy_cfg.dte_max)
        latest_trade = expiry_ts.date() - timedelta(days=strategy_cfg.dte_min)
        relevant = daily_underlying[
            (daily_underlying["trade_date"] >= earliest_trade)
            & (daily_underlying["trade_date"] <= latest_trade)
        ]
        if relevant.empty:
            continue
        strike_min = float(relevant["query_strike_min"].min())
        strike_max = float(relevant["query_strike_max"].max())
        filtered = expiry_group.copy()
        filtered["strike_price"] = pd.to_numeric(filtered["strike_price"], errors="coerce")
        filtered["contract_type"] = filtered["contract_type"].astype(str).str.lower()
        filtered = filtered[
            filtered["contract_type"].isin(contract_types)
            & filtered["strike_price"].notna()
            & (filtered["strike_price"] >= strike_min)
            & (filtered["strike_price"] <= strike_max)
        ]
        if not filtered.empty:
            kept_frames.append(filtered)

    if not kept_frames:
        return frame.head(0).copy()
    merged = pd.concat(kept_frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["ticker"]).sort_values(
        ["expiration_date", "strike_price", "contract_type", "ticker"]
    )
    return merged.reset_index(drop=True)


def _resolve_contract_history_window(
    contract_row: pd.Series | dict[str, Any],
    cfg: MassiveBackfillConfig,
) -> Optional[tuple[date, date]]:
    if cfg.strategy_universe is None:
        return cfg.start_date, cfg.end_date

    expiry_ts = pd.to_datetime(dict(contract_row).get("expiration_date"), errors="coerce")
    if pd.isna(expiry_ts):
        return cfg.start_date, cfg.end_date

    window_start = max(
        cfg.start_date,
        expiry_ts.date() - timedelta(days=int(cfg.strategy_universe.dte_max)),
    )
    window_end = min(cfg.end_date, expiry_ts.date())
    if window_end < window_start:
        return None
    return window_start, window_end


def _strategy_reach_points(strategy_cfg: StrategyUniverseConfig) -> float:
    return float(strategy_cfg.butterfly_width) + max(0.0, float(strategy_cfg.broken_wing_extra_width))


def _local_day_boundary_utc(value: date) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(NEW_YORK_TZ).tz_convert("UTC")


def _round_down(value: float, increment: float) -> float:
    return round(math.floor(float(value) / increment) * increment, 6)


def _round_up(value: float, increment: float) -> float:
    return round(math.ceil(float(value) / increment) * increment, 6)


def _flush_batch(
    parts_dir: Path,
    batch_frames: list[pd.DataFrame],
    completed_batch_tickers: list[str],
    checkpoint: BackfillCheckpoint,
    checkpoint_path: Path,
    output_format: str,
) -> None:
    combined = pd.concat(batch_frames, ignore_index=True) if batch_frames else pd.DataFrame(columns=NORMALIZED_DATASET_COLUMNS)
    part_name = f"bars_part_{checkpoint.batch_index:05d}.{output_format}"
    part_path = parts_dir / part_name
    write_dataframe(combined, part_path, output_format)
    print(
        f"Wrote {part_name} with {len(completed_batch_tickers)} contracts and {len(combined)} rows. "
        f"Completed {len(checkpoint.completed_tickers) + len(completed_batch_tickers)} total tickers."
    )

    checkpoint.batch_index += 1
    checkpoint.completed_tickers = sorted(set(checkpoint.completed_tickers).union(completed_batch_tickers))
    checkpoint.completed_parts = sorted(set(checkpoint.completed_parts).union({part_name}))
    checkpoint.last_updated_utc = pd.Timestamp.utcnow().isoformat()
    _save_checkpoint(checkpoint_path, checkpoint)


def _resolve_contracts_path(output_dir: Path, output_format: str) -> Path:
    return output_dir / f"contracts.{output_format}"


def _resolve_final_dataset_path(output_dir: Path, output_format: str) -> Path:
    return output_dir / f"spx_options_daily_history.{output_format}"


def _load_checkpoint(path: Path) -> Optional[BackfillCheckpoint]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BackfillCheckpoint(**payload)


def _save_checkpoint(path: Path, checkpoint: BackfillCheckpoint) -> None:
    path.write_text(json.dumps(asdict(checkpoint), indent=2), encoding="utf-8")


def _strategy_signature(cfg: MassiveBackfillConfig) -> str:
    if cfg.strategy_universe is None:
        return ""
    strategy = cfg.strategy_universe
    payload = {
        "bars_csv": str(strategy.bars_csv.resolve()) if strategy.bars_csv.exists() else str(strategy.bars_csv),
        "symbol": strategy.symbol,
        "contract_types": list(strategy.contract_types),
        "dte_min": int(strategy.dte_min),
        "dte_max": int(strategy.dte_max),
        "center_rounding": float(strategy.center_rounding),
        "butterfly_width": float(strategy.butterfly_width),
        "wing_mode": str(strategy.wing_mode),
        "broken_wing_extra_width": float(strategy.broken_wing_extra_width),
        "strike_buffer_points": float(strategy.strike_buffer_points),
        "slice_days": int(strategy.slice_days),
    }
    return json.dumps(payload, sort_keys=True)


def _validate_checkpoint(
    checkpoint: BackfillCheckpoint,
    cfg: MassiveBackfillConfig,
    output_format: str,
) -> None:
    if checkpoint.start_date != cfg.start_date.isoformat() or checkpoint.end_date != cfg.end_date.isoformat():
        raise ValueError("Existing checkpoint does not match the requested date window.")
    if tuple(checkpoint.contract_underlyings) != tuple(cfg.contract_underlyings):
        raise ValueError("Existing checkpoint does not match the requested contract underlyings.")
    if checkpoint.output_format != output_format:
        raise ValueError("Existing checkpoint uses a different output format.")
    if checkpoint.strategy_signature != _strategy_signature(cfg):
        raise ValueError("Existing checkpoint does not match the requested strategy-only filter configuration.")
