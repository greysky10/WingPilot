from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from corridor.config import CorridorConfig
from corridor.models import ActiveButterfly
from corridor.options.butterfly_selector import ButterflyCandidate, select_butterflies_with_diagnostics
from corridor.options.chain_loader import OptionQuote


DEFAULT_HISTORICAL_CHAIN_BASENAME = "spx_options_daily_history"
DEFAULT_HISTORICAL_CHAIN_DIR = Path("data") / "massive_spx_history"


@dataclass(slots=True)
class HistoricalChainSelection:
    trade_date: str
    expiry: str
    lower_option_ticker: str
    body_option_ticker: str
    upper_option_ticker: str
    candidate: ButterflyCandidate


def _ensure_local_date_str(timestamp: pd.Timestamp | str) -> str:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/New_York").strftime("%Y-%m-%d")


def _calendar_dte(expiry: str, trade_date: str) -> Optional[int]:
    expiry_date = pd.to_datetime(str(expiry), errors="coerce")
    trade_date_value = pd.to_datetime(str(trade_date), errors="coerce")
    if pd.isna(expiry_date) or pd.isna(trade_date_value):
        return None
    return int((expiry_date.date() - trade_date_value.date()).days)


def _infer_default_historical_chain_path() -> Optional[Path]:
    for suffix in ["parquet", "csv"]:
        candidate = DEFAULT_HISTORICAL_CHAIN_DIR / f"{DEFAULT_HISTORICAL_CHAIN_BASENAME}.{suffix}"
        if candidate.exists():
            return candidate
    return None


class HistoricalOptionChainStore:
    def __init__(self, frame: pd.DataFrame, source_path: Path, price_field: str = "close") -> None:
        required = {"date", "option_ticker", "underlying", "expiry", "strike", "type", price_field}
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise ValueError(f"Historical chain dataset is missing required columns: {', '.join(sorted(missing))}")

        cleaned = frame.copy()
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        cleaned["expiry"] = pd.to_datetime(cleaned["expiry"], errors="coerce").dt.strftime("%Y-%m-%d")
        cleaned["strike"] = pd.to_numeric(cleaned["strike"], errors="coerce")
        cleaned["type"] = cleaned["type"].astype(str).str.lower()
        cleaned["option_ticker"] = cleaned["option_ticker"].astype(str)
        cleaned["underlying"] = cleaned["underlying"].astype(str).str.upper()
        cleaned[price_field] = pd.to_numeric(cleaned[price_field], errors="coerce")
        cleaned = cleaned.dropna(subset=["date", "expiry", "strike", "type", "option_ticker", price_field])
        cleaned = cleaned.sort_values(["date", "expiry", "strike", "type", "option_ticker"]).reset_index(drop=True)

        self.frame = cleaned
        self.source_path = Path(source_path)
        self.price_field = price_field
        self._trade_date_cache: dict[str, pd.DataFrame] = {}
        self._ticker_history_cache: dict[str, pd.DataFrame] = {}

    @classmethod
    def from_path(cls, path: Path, price_field: str = "close") -> HistoricalOptionChainStore:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Historical chain dataset not found: {resolved}")
        if resolved.suffix.lower() == ".parquet":
            frame = pd.read_parquet(resolved)
        else:
            frame = pd.read_csv(resolved)
        return cls(frame=frame, source_path=resolved, price_field=price_field)

    def select_candidate(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        center_price: float,
        width: float,
        target_dte: int,
        config: CorridorConfig,
    ) -> Optional[HistoricalChainSelection]:
        trade_date = _ensure_local_date_str(timestamp)
        trade_frame = self._trade_date_frame(trade_date)
        if trade_frame.empty:
            return None

        symbol_frame = trade_frame[trade_frame["underlying"] == str(symbol).upper()].copy()
        if symbol_frame.empty:
            return None

        dte_series = pd.to_datetime(symbol_frame["expiry"], errors="coerce").dt.date
        trade_date_value = pd.Timestamp(trade_date).date()
        dte_values = [(value - trade_date_value).days if pd.notna(value) else None for value in dte_series]
        symbol_frame = symbol_frame.assign(calendar_dte=dte_values)
        symbol_frame = symbol_frame[
            symbol_frame["calendar_dte"].notna()
            & (symbol_frame["calendar_dte"] >= int(config.dte_min))
            & (symbol_frame["calendar_dte"] <= int(config.dte_max))
        ].copy()
        if symbol_frame.empty:
            return None

        quotes = [
            OptionQuote(
                symbol=str(row["underlying"]),
                expiry=str(row["expiry"]),
                strike=float(row["strike"]),
                right="CALL" if str(row["type"]).lower() == "call" else "PUT",
                bid=float(row[self.price_field]),
                ask=float(row[self.price_field]),
                last=float(row[self.price_field]),
                implied_vol=None,
            )
            for _, row in symbol_frame.iterrows()
            if float(row[self.price_field]) > 0
        ]
        if not quotes:
            return None

        candidates, _diagnostics = select_butterflies_with_diagnostics(
            quotes,
            center_price,
            width,
            config,
            reference_date=trade_date,
        )
        if not candidates:
            return None

        ordered = sorted(
            candidates,
            key=lambda item: (
                abs((_calendar_dte(item.expiry, trade_date) or target_dte) - int(target_dte)),
                item.body_distance,
                0 if item.wing_mode == str(config.wing_mode) else 1,
                item.total_spread,
                item.net_debit,
                item.expiry,
            ),
        )
        selected = ordered[0]
        option_type = str(selected.right).lower()
        lower_ticker = self._lookup_option_ticker(symbol_frame, selected.expiry, selected.lower_strike, option_type)
        body_ticker = self._lookup_option_ticker(symbol_frame, selected.expiry, selected.body_strike, option_type)
        upper_ticker = self._lookup_option_ticker(symbol_frame, selected.expiry, selected.upper_strike, option_type)
        if not (lower_ticker and body_ticker and upper_ticker):
            return None

        return HistoricalChainSelection(
            trade_date=trade_date,
            expiry=str(selected.expiry),
            lower_option_ticker=lower_ticker,
            body_option_ticker=body_ticker,
            upper_option_ticker=upper_ticker,
            candidate=selected,
        )

    def combo_mark(
        self,
        layer: ActiveButterfly,
        timestamp: pd.Timestamp,
        spot: float,
    ) -> float:
        trade_date = _ensure_local_date_str(timestamp)
        expiry = str(layer.metadata.get("historical_chain_expiry", ""))
        if not expiry:
            raise ValueError("Historical chain layer has no stored expiry metadata.")

        lower = self._price_for_leg(str(layer.metadata.get("historical_chain_lower_ticker", "")), trade_date)
        body = self._price_for_leg(str(layer.metadata.get("historical_chain_body_ticker", "")), trade_date)
        upper = self._price_for_leg(str(layer.metadata.get("historical_chain_upper_ticker", "")), trade_date)
        if lower is not None and body is not None and upper is not None:
            return float(lower - (2.0 * body) + upper)

        if pd.Timestamp(trade_date).date() > pd.Timestamp(expiry).date():
            return self.terminal_combo_value(layer, float(spot))

        latest_available = self._latest_combo_before(layer, trade_date)
        if latest_available is not None:
            return latest_available

        return self.terminal_combo_value(layer, float(spot))

    def _trade_date_frame(self, trade_date: str) -> pd.DataFrame:
        cached = self._trade_date_cache.get(trade_date)
        if cached is not None:
            return cached
        frame = self.frame[self.frame["date"] == trade_date].copy()
        self._trade_date_cache[trade_date] = frame
        return frame

    def _ticker_history(self, ticker: str) -> pd.DataFrame:
        cached = self._ticker_history_cache.get(ticker)
        if cached is not None:
            return cached
        history = self.frame[self.frame["option_ticker"] == ticker].sort_values("date").copy()
        self._ticker_history_cache[ticker] = history
        return history

    def _price_for_leg(self, ticker: str, trade_date: str) -> Optional[float]:
        if not ticker:
            return None
        history = self._ticker_history(ticker)
        if history.empty:
            return None
        exact = history[history["date"] == trade_date]
        if not exact.empty:
            return float(exact[self.price_field].iloc[-1])
        prior = history[history["date"] < trade_date]
        if prior.empty:
            return None
        return float(prior[self.price_field].iloc[-1])

    def _latest_combo_before(self, layer: ActiveButterfly, trade_date: str) -> Optional[float]:
        lower_ticker = str(layer.metadata.get("historical_chain_lower_ticker", ""))
        body_ticker = str(layer.metadata.get("historical_chain_body_ticker", ""))
        upper_ticker = str(layer.metadata.get("historical_chain_upper_ticker", ""))
        if not (lower_ticker and body_ticker and upper_ticker):
            return None

        lower_history = self._ticker_history(lower_ticker)
        body_history = self._ticker_history(body_ticker)
        upper_history = self._ticker_history(upper_ticker)
        if lower_history.empty or body_history.empty or upper_history.empty:
            return None

        lower_prior = lower_history[lower_history["date"] < trade_date]
        body_prior = body_history[body_history["date"] < trade_date]
        upper_prior = upper_history[upper_history["date"] < trade_date]
        if lower_prior.empty or body_prior.empty or upper_prior.empty:
            return None
        latest_common_date = min(
            str(lower_prior["date"].iloc[-1]),
            str(body_prior["date"].iloc[-1]),
            str(upper_prior["date"].iloc[-1]),
        )
        lower = self._price_for_leg(lower_ticker, latest_common_date)
        body = self._price_for_leg(body_ticker, latest_common_date)
        upper = self._price_for_leg(upper_ticker, latest_common_date)
        if lower is None or body is None or upper is None:
            return None
        return float(lower - (2.0 * body) + upper)

    @staticmethod
    def _lookup_option_ticker(
        trade_frame: pd.DataFrame,
        expiry: str,
        strike: float,
        option_type: str,
    ) -> Optional[str]:
        matches = trade_frame[
            (trade_frame["expiry"] == str(expiry))
            & (trade_frame["type"] == str(option_type).lower())
            & ((trade_frame["strike"] - float(strike)).abs() < 1e-9)
        ]
        if matches.empty:
            return None
        return str(matches["option_ticker"].iloc[0])

    @staticmethod
    def terminal_combo_value(layer: ActiveButterfly, spot: float) -> float:
        lower = float(layer.lower_strike)
        body = float(layer.body_strike)
        upper = float(layer.upper_strike)
        option_right = str(layer.metadata.get("historical_chain_right", "CALL")).upper()
        if option_right == "PUT":
            return (
                max(lower - float(spot), 0.0)
                - (2.0 * max(body - float(spot), 0.0))
                + max(upper - float(spot), 0.0)
            )
        return (
            max(float(spot) - lower, 0.0)
            - (2.0 * max(float(spot) - body, 0.0))
            + max(float(spot) - upper, 0.0)
        )


class HistoricalChainButterflyPricer:
    def __init__(self, config: CorridorConfig, store: HistoricalOptionChainStore) -> None:
        self.config = config
        self.store = store

    @classmethod
    def from_config(cls, config: CorridorConfig) -> HistoricalChainButterflyPricer:
        resolved_path: Optional[Path]
        if config.historical_chain_path:
            resolved_path = Path(config.historical_chain_path)
        else:
            resolved_path = _infer_default_historical_chain_path()
        if resolved_path is None:
            raise FileNotFoundError(
                "Historical chain payoff mode requires --historical-chain-path or a dataset at "
                f"{DEFAULT_HISTORICAL_CHAIN_DIR / (DEFAULT_HISTORICAL_CHAIN_BASENAME + '.parquet')} "
                "or .csv."
            )
        config.historical_chain_path = str(resolved_path)
        return cls(config, HistoricalOptionChainStore.from_path(resolved_path, price_field=config.historical_chain_price_field))

    def attach_candidate(self, layer: ActiveButterfly, symbol: str, timestamp: pd.Timestamp) -> Optional[HistoricalChainSelection]:
        selection = self.store.select_candidate(
            symbol=symbol,
            timestamp=timestamp,
            center_price=float(layer.center_price),
            width=float(layer.width),
            target_dte=int(layer.dte),
            config=self.config,
        )
        if selection is None:
            return None

        candidate = selection.candidate
        selected_dte = _calendar_dte(selection.expiry, selection.trade_date)
        layer.center_price = float(candidate.body_strike)
        layer.lower_strike = float(candidate.lower_strike)
        layer.body_strike = float(candidate.body_strike)
        layer.upper_strike = float(candidate.upper_strike)
        layer.lower_width = float(candidate.lower_width)
        layer.upper_width = float(candidate.upper_width)
        layer.width = min(float(candidate.lower_width), float(candidate.upper_width))
        if selected_dte is not None:
            layer.dte = int(selected_dte)
        layer.metadata.update(
            {
                "wing_mode": str(candidate.wing_mode),
                "historical_chain_path": str(self.store.source_path),
                "historical_chain_price_field": str(self.store.price_field),
                "historical_chain_trade_date": selection.trade_date,
                "historical_chain_expiry": str(selection.expiry),
                "historical_chain_right": str(candidate.right),
                "historical_chain_lower_ticker": str(selection.lower_option_ticker),
                "historical_chain_body_ticker": str(selection.body_option_ticker),
                "historical_chain_upper_ticker": str(selection.upper_option_ticker),
                "historical_chain_selected_net_debit": float(candidate.net_debit),
                "historical_chain_selected_total_spread": float(candidate.total_spread),
                "historical_chain_selected_body_distance": float(candidate.body_distance),
                "historical_chain_selected_dte": int(selected_dte) if selected_dte is not None else int(layer.dte),
            }
        )
        return selection

    def entry_debit(self, layer: ActiveButterfly) -> float:
        return self.store.combo_mark(layer, pd.Timestamp(layer.created_at), float(layer.center_price))

    def entry_cost(self, layer: ActiveButterfly) -> float:
        return self.entry_debit(layer) + self.friction_per_layer(layer)

    def mark_to_model(self, layer: ActiveButterfly, spot: float, timestamp: pd.Timestamp) -> float:
        return self.store.combo_mark(layer, timestamp, spot)

    def close_value(self, layer: ActiveButterfly, spot: float, timestamp: pd.Timestamp) -> float:
        mark = self.mark_to_model(layer, spot, timestamp)
        friction = self.friction_per_layer(layer)
        close_before_haircut = mark - friction
        if close_before_haircut >= 0:
            return max(0.0, close_before_haircut * (1.0 - self.config.stress_close_value_haircut_pct))
        return close_before_haircut * (1.0 + self.config.stress_close_value_haircut_pct)

    def friction_per_layer(self, layer: ActiveButterfly | None = None) -> float:
        return self.slippage_cost_per_layer(layer) + self.commission_cost_per_layer()

    def commission_cost_per_layer(self) -> float:
        return (self.config.commission_per_contract * 4.0) / float(self.config.option_multiplier)

    def slippage_cost_per_layer(self, layer: ActiveButterfly | None = None) -> float:
        contract_equivalents = self.slippage_contract_equivalents(layer)
        return self.config.per_contract_slippage * contract_equivalents * self.config.stress_slippage_multiplier

    @staticmethod
    def slippage_contract_equivalents(layer: ActiveButterfly | None = None) -> float:
        if layer is not None and abs(float(layer.upper_width) - float(layer.lower_width)) > 0:
            return 5.0
        return 4.0

    def modeled_max_loss(self, layer: ActiveButterfly) -> float:
        return layer.entry_cost + abs(float(layer.upper_width) - float(layer.lower_width))
