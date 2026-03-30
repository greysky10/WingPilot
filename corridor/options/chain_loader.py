from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


try:
    from ib_insync import IB, Option, Stock
except ImportError:  # pragma: no cover - optional dependency
    IB = None
    Option = None
    Stock = None


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
        dte_min: int,
        dte_max: int,
        market_data_type: int = 3,
    ) -> list[OptionQuote]:
        if IB is None or Option is None or Stock is None:
            raise RuntimeError("ib_insync is required for option-chain loading.")

        ib = IB()
        ib.connect(self.host, self.port, clientId=self.client_id, timeout=10)
        try:
            underlying = Stock(symbol.upper(), self.exchange, self.currency)
            ib.qualifyContracts(underlying)
            ib.reqMarketDataType(market_data_type)

            chains = ib.reqSecDefOptParams(underlying.symbol, "", underlying.secType, underlying.conId)
            if not chains:
                return []
            chain = chains[0]

            today = pd.Timestamp.utcnow().date()
            expiries = [
                expiry
                for expiry in sorted(chain.expirations)
                if dte_min <= (pd.Timestamp(expiry).date() - today).days <= dte_max
            ]
            if not expiries:
                return []

            available_strikes = sorted(float(strike) for strike in chain.strikes)
            if not available_strikes:
                return []

            body = _nearest_strike(available_strikes, center_price)
            strikes = sorted(
                {
                    _nearest_strike(available_strikes, body - width),
                    _nearest_strike(available_strikes, body),
                    _nearest_strike(available_strikes, body + width),
                }
            )
            contracts = [
                Option(
                    symbol=symbol.upper(),
                    lastTradeDateOrContractMonth=expiry,
                    strike=strike,
                    right="C",
                    exchange=self.exchange,
                    currency=self.currency,
                )
                for expiry in expiries
                for strike in strikes
            ]
            if not contracts:
                return []

            ib.qualifyContracts(*contracts)
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
