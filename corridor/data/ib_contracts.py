from __future__ import annotations

from dataclasses import dataclass


try:
    from ib_insync import Index, Option, Stock
except ImportError:  # pragma: no cover - optional dependency
    Index = None
    Option = None
    Stock = None


@dataclass(frozen=True, slots=True)
class IBSymbolSpec:
    symbol: str
    underlying_sec_type: str
    underlying_exchange: str
    option_exchange: str
    currency: str
    center_rounding: float
    preferred_trading_classes: tuple[str, ...]


def get_ib_symbol_spec(symbol: str, exchange: str = "SMART", currency: str = "USD") -> IBSymbolSpec:
    normalized = symbol.upper()
    if normalized == "SPX":
        return IBSymbolSpec(
            symbol=normalized,
            underlying_sec_type="IND",
            underlying_exchange="CBOE",
            option_exchange=exchange or "SMART",
            currency=currency,
            center_rounding=5.0,
            preferred_trading_classes=("SPXW", "SPX"),
        )
    return IBSymbolSpec(
        symbol=normalized,
        underlying_sec_type="STK",
        underlying_exchange=exchange,
        option_exchange=exchange,
        currency=currency,
        center_rounding=1.0,
        preferred_trading_classes=(normalized,),
    )


def default_center_rounding_for_symbol(symbol: str) -> float:
    return get_ib_symbol_spec(symbol).center_rounding


def build_underlying_contract(symbol: str, exchange: str = "SMART", currency: str = "USD"):
    spec = get_ib_symbol_spec(symbol, exchange=exchange, currency=currency)
    if spec.underlying_sec_type == "IND":
        if Index is None:
            raise RuntimeError("ib_insync is required to build index contracts.")
        return Index(spec.symbol, spec.underlying_exchange, spec.currency)
    if Stock is None:
        raise RuntimeError("ib_insync is required to build stock contracts.")
    return Stock(spec.symbol, spec.underlying_exchange, spec.currency)


def build_option_contract(
    symbol: str,
    expiry: str,
    strike: float,
    right: str,
    exchange: str = "SMART",
    currency: str = "USD",
    trading_class: str | None = None,
):
    spec = get_ib_symbol_spec(symbol, exchange=exchange, currency=currency)
    if Option is None:
        raise RuntimeError("ib_insync is required to build option contracts.")
    kwargs = {
        "symbol": spec.symbol,
        "lastTradeDateOrContractMonth": expiry,
        "strike": strike,
        "right": right,
        "exchange": spec.option_exchange,
        "currency": spec.currency,
    }
    if trading_class:
        kwargs["tradingClass"] = trading_class
    return Option(**kwargs)


def chain_sort_key(symbol: str, option_exchange: str, chain) -> tuple[int, int, str, str]:
    spec = get_ib_symbol_spec(symbol, exchange=option_exchange)
    exchange_priority = 0 if getattr(chain, "exchange", "") == spec.option_exchange else 1
    trading_class = str(getattr(chain, "tradingClass", "") or "")
    try:
        class_priority = spec.preferred_trading_classes.index(trading_class)
    except ValueError:
        class_priority = len(spec.preferred_trading_classes)
    return (
        exchange_priority,
        class_priority,
        trading_class,
        str(getattr(chain, "exchange", "") or ""),
    )
