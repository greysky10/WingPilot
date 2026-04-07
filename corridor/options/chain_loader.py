from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from corridor.data.ib_contracts import build_option_contract, build_underlying_contract, chain_sort_key


try:
    from ib_insync import IB
except ImportError:  # pragma: no cover - optional dependency
    IB = None


@dataclass(slots=True)
class OptionQuote:
    symbol: str
    expiry: str
    strike: float
    right: str
    bid: float
    ask: float
    last: float
    implied_vol: float | None
    trading_class: str | None = None

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        if self.last > 0:
            return self.last
        return 0.0

    @property
    def spread(self) -> float:
        if self.bid <= 0 or self.ask <= 0:
            return 999.0
        return self.ask - self.bid


class IBOptionChainLoader:
    """Optional IBKR options chain loader using ib_insync."""

    def __init__(self, host: str, port: int, client_id: int, exchange: str = "SMART", currency: str = "USD") -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.exchange = exchange
        self.currency = currency

    def load_candidates(
        self,
        symbol: str,
        center_price: float,
        width: float,
        extra_width: float,
        wing_mode: str,
        dte_min: int,
        dte_max: int,
        body_search_steps: int = 2,
        center_rounding: float = 1.0,
        market_data_type: int = 3,
    ) -> list[OptionQuote]:
        if IB is None:
            raise RuntimeError("ib_insync is required for option-chain loading.")

        ib = IB()
        ib.connect(self.host, self.port, clientId=self.client_id, timeout=10)
        try:
            underlying = build_underlying_contract(symbol.upper(), self.exchange, self.currency)
            ib.qualifyContracts(underlying)
            ib.reqMarketDataType(market_data_type)

            chains = ib.reqSecDefOptParams(underlying.symbol, "", underlying.secType, underlying.conId)
            if not chains:
                return []
            ordered_chains = sorted(chains, key=lambda item: chain_sort_key(symbol.upper(), self.exchange, item))

            today = pd.Timestamp.utcnow().date()
            chain = None
            expiries: list[str] = []
            available_strikes: list[float] = []
            for item in ordered_chains:
                item_expiries = [
                    expiry
                    for expiry in sorted(item.expirations)
                    if dte_min <= (pd.Timestamp(expiry).date() - today).days <= dte_max
                ]
                item_strikes = sorted(float(strike) for strike in item.strikes)
                if item_expiries and item_strikes:
                    chain = item
                    expiries = item_expiries
                    available_strikes = item_strikes
                    break
            if chain is None:
                return []

            body = _nearest_strike(available_strikes, center_price)
            body_candidates = {
                _nearest_strike(
                    available_strikes,
                    body + (offset * max(center_rounding, 1.0)),
                )
                for offset in range(-max(0, body_search_steps), max(0, body_search_steps) + 1)
            }
            strikes = sorted(
                {
                    strike
                    for body_candidate in body_candidates
                    for strike in _candidate_strikes(
                        available_strikes,
                        body_candidate,
                        width,
                        extra_width,
                        wing_mode,
                    )
                }
            )
            contracts = [
                build_option_contract(
                    symbol=symbol.upper(),
                    expiry=expiry,
                    strike=strike,
                    right="C",
                    exchange=self.exchange,
                    currency=self.currency,
                    trading_class=str(getattr(chain, "tradingClass", "") or "") or None,
                )
                for expiry in expiries
                for strike in strikes
            ]
            if not contracts:
                return []

            contracts = _quietly_qualify_contracts(ib, contracts)
            if not contracts:
                return []
            market_data_errors: list[int] = []

            def capture_error(_req_id: int, error_code: int, _error_string: str, _contract) -> None:
                if error_code in {354, 10091}:
                    market_data_errors.append(error_code)

            ib.errorEvent += capture_error
            wrapper_logger = logging.getLogger("ib_insync.wrapper")
            previous_level = wrapper_logger.level
            wrapper_logger.setLevel(logging.CRITICAL)
            try:
                tickers = ib.reqTickers(*contracts)
            finally:
                wrapper_logger.setLevel(previous_level)
                ib.errorEvent -= capture_error
            quotes: list[OptionQuote] = []
            for contract, ticker in zip(contracts, tickers, strict=False):
                model_iv = None
                if getattr(ticker, "modelGreeks", None) is not None:
                    model_iv = getattr(ticker.modelGreeks, "impliedVol", None)
                quotes.append(
                    OptionQuote(
                        symbol=contract.symbol,
                        expiry=contract.lastTradeDateOrContractMonth,
                        strike=float(contract.strike),
                        right="CALL" if contract.right == "C" else "PUT",
                        trading_class=str(getattr(contract, "tradingClass", "") or "") or None,
                        bid=float(ticker.bid or 0.0),
                        ask=float(ticker.ask or 0.0),
                        last=float(ticker.last or 0.0),
                        implied_vol=float(model_iv) if model_iv is not None else None,
                    )
                )
            if market_data_errors and not any(quote.mid > 0 for quote in quotes):
                raise RuntimeError(
                    "IB option quotes are unavailable for API on this account. "
                    "Enable delayed/live US options market data permissions in IBKR."
                )
            return quotes
        finally:
            ib.disconnect()

    def load_structure_quotes(
        self,
        symbol: str,
        expiry: str,
        lower_strike: float,
        body_strike: float,
        upper_strike: float,
        right: str,
        market_data_type: int = 3,
        trading_class: Optional[str] = None,
    ) -> list[OptionQuote]:
        if IB is None:
            raise RuntimeError("ib_insync is required for option-chain loading.")

        ib = IB()
        ib.connect(self.host, self.port, clientId=self.client_id, timeout=10)
        try:
            underlying = build_underlying_contract(symbol.upper(), self.exchange, self.currency)
            ib.qualifyContracts(underlying)
            ib.reqMarketDataType(market_data_type)

            right_code = "C" if str(right).upper() in {"CALL", "C"} else "P"
            contracts = [
                build_option_contract(
                    symbol=symbol.upper(),
                    expiry=expiry,
                    strike=strike,
                    right=right_code,
                    exchange=self.exchange,
                    currency=self.currency,
                    trading_class=trading_class,
                )
                for strike in [lower_strike, body_strike, upper_strike]
            ]
            contracts = _quietly_qualify_contracts(ib, contracts)
            if not contracts:
                return []
            market_data_errors: list[int] = []

            def capture_error(_req_id: int, error_code: int, _error_string: str, _contract) -> None:
                if error_code in {354, 10091}:
                    market_data_errors.append(error_code)

            ib.errorEvent += capture_error
            try:
                tickers = ib.reqTickers(*contracts)
            finally:
                ib.errorEvent -= capture_error

            quotes: list[OptionQuote] = []
            for contract, ticker in zip(contracts, tickers, strict=False):
                model_iv = None
                if getattr(ticker, "modelGreeks", None) is not None:
                    model_iv = getattr(ticker.modelGreeks, "impliedVol", None)
                quotes.append(
                    OptionQuote(
                        symbol=contract.symbol,
                        expiry=contract.lastTradeDateOrContractMonth,
                        strike=float(contract.strike),
                        right="CALL" if contract.right == "C" else "PUT",
                        trading_class=str(getattr(contract, "tradingClass", "") or "") or None,
                        bid=float(ticker.bid or 0.0),
                        ask=float(ticker.ask or 0.0),
                        last=float(ticker.last or 0.0),
                        implied_vol=float(model_iv) if model_iv is not None else None,
                    )
                )
            if market_data_errors and not any(quote.mid > 0 for quote in quotes):
                raise RuntimeError(
                    "IB option quotes are unavailable for API on this account. "
                    "Enable delayed/live US options market data permissions in IBKR."
                )
            return quotes
        finally:
            ib.disconnect()


def _nearest_strike(strikes: Iterable[float], target: float) -> float:
    return min(strikes, key=lambda strike: (abs(strike - target), strike))


def _quietly_qualify_contracts(ib: object, contracts: Iterable[object]) -> list[object]:
    contracts = list(contracts)
    if not contracts:
        return []
    wrapper_logger = logging.getLogger("ib_insync.wrapper")
    ib_logger = logging.getLogger("ib_insync.ib")
    previous_wrapper_level = wrapper_logger.level
    previous_ib_level = ib_logger.level
    wrapper_logger.setLevel(logging.CRITICAL)
    ib_logger.setLevel(logging.CRITICAL)
    try:
        qualified = ib.qualifyContracts(*contracts)
    finally:
        wrapper_logger.setLevel(previous_wrapper_level)
        ib_logger.setLevel(previous_ib_level)
    return _qualified_contracts_only(qualified)


def _qualified_contracts_only(contracts: Iterable[object]) -> list[object]:
    qualified: list[object] = []
    for contract in contracts:
        con_id = getattr(contract, "conId", None)
        if con_id in (None, 0):
            continue
        qualified.append(contract)
    return qualified


def _candidate_strikes(
    available_strikes: Iterable[float],
    body_candidate: float,
    width: float,
    extra_width: float,
    wing_mode: str,
) -> set[float]:
    strikes = {
        _nearest_strike(available_strikes, body_candidate - width),
        _nearest_strike(available_strikes, body_candidate),
        _nearest_strike(available_strikes, body_candidate + width),
    }
    if extra_width <= 0:
        return strikes
    if wing_mode in {"broken_upper", "adaptive"}:
        strikes.add(_nearest_strike(available_strikes, body_candidate + width + extra_width))
    if wing_mode in {"broken_lower", "adaptive"}:
        strikes.add(_nearest_strike(available_strikes, body_candidate - width - extra_width))
    return strikes
