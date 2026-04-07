#! python3.12
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from corridor.data.ib_contracts import build_option_contract, build_underlying_contract, chain_sort_key

try:
    from ib_insync import IB
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    IB = None  # type: ignore[assignment]
    IB_IMPORT_ERROR = exc
else:
    IB_IMPORT_ERROR = None

from strategy import load_local_env


IB_DEPENDENCY_HINT = "py -3.12 -m pip install ib_insync pandas pytz"


@dataclass(slots=True)
class QuoteProbeResult:
    market_data_type: int
    label: str
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    close: Optional[float]
    has_two_sided_quote: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether IBKR API can retrieve option quotes and classify them as LIVE, DELAYED, or NOT_AVAILABLE."
    )
    parser.add_argument("--symbol", default="SPX", help="Underlying symbol to probe.")
    parser.add_argument("--host", default=os.getenv("IB_HOST", "localhost"), help="IB host.")
    parser.add_argument("--port", type=int, default=int(os.getenv("IB_PORT", "4002")), help="IB API port.")
    parser.add_argument("--client-id", type=int, default=90, help="IB client id for the probe.")
    parser.add_argument("--exchange", default=os.getenv("IB_EXCHANGE", "SMART"), help="Exchange for stock and option contracts.")
    parser.add_argument("--currency", default=os.getenv("IB_CURRENCY", "USD"), help="Contract currency.")
    parser.add_argument("--right", choices=["C", "P"], default="C", help="Option side to probe.")
    parser.add_argument("--expiry", default="", help="Exact expiry YYYYMMDD. If omitted, choose nearest available expiry.")
    parser.add_argument("--strike", type=float, default=0.0, help="Exact strike. If omitted, choose nearest ATM strike.")
    parser.add_argument("--dte-min", type=int, default=1, help="Minimum calendar days to expiry when auto-selecting.")
    parser.add_argument("--wait-seconds", type=float, default=5.0, help="How long to wait for market data after requesting.")
    return parser.parse_args()


def require_ib() -> None:
    if IB_IMPORT_ERROR is None:
        return
    raise SystemExit(
        "Missing required package: ib_insync. Install it with `"
        + IB_DEPENDENCY_HINT
        + "`."
    )


def clean_number(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    if numeric < 0:
        return None
    return numeric


def choose_reference_price(ib: IB, contract) -> float:
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="5 D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    if not bars:
        raise RuntimeError("IB returned no underlying history, so ATM strike could not be chosen.")
    return float(bars[-1].close)


def choose_option_contract(
    ib: IB,
    symbol: str,
    exchange: str,
    currency: str,
    expiry_arg: str,
    strike_arg: float,
    right: str,
    dte_min: int,
) -> tuple[object, float]:
    underlying_contract = build_underlying_contract(symbol, exchange, currency)
    qualified = ib.qualifyContracts(underlying_contract)
    if not qualified:
        raise RuntimeError(f"Unable to qualify underlying contract for {symbol}.")
    underlying = qualified[0]
    reference_price = choose_reference_price(ib, underlying)

    chains = ib.reqSecDefOptParams(underlying.symbol, "", underlying.secType, underlying.conId)
    if not chains:
        raise RuntimeError(f"IB returned no option chain definitions for {symbol}.")
    preferred = sorted(chains, key=lambda item: chain_sort_key(symbol, exchange, item))[0]

    expiry = expiry_arg.strip()
    if not expiry:
        min_expiry = (date.today() + timedelta(days=max(0, dte_min))).strftime("%Y%m%d")
        valid_expiries = sorted(item for item in preferred.expirations if item >= min_expiry)
        if not valid_expiries:
            raise RuntimeError(f"No option expiry found for {symbol} at or beyond {min_expiry}.")
        expiry = valid_expiries[0]

    strike = float(strike_arg)
    if strike <= 0:
        valid_strikes = sorted(float(item) for item in preferred.strikes if item > 0)
        if not valid_strikes:
            raise RuntimeError(f"No strikes returned in option chain for {symbol}.")
        strike = min(valid_strikes, key=lambda item: abs(item - reference_price))

    option_contract = build_option_contract(
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        right=right,
        exchange=exchange,
        currency=currency,
        trading_class=str(getattr(preferred, "tradingClass", "") or "") or None,
    )
    qualified_option = ib.qualifyContracts(option_contract)
    if not qualified_option:
        raise RuntimeError(f"Unable to qualify option contract for {symbol} {expiry} {strike} {right}.")
    return qualified_option[0], reference_price


def probe_quote(ib: IB, contract, market_data_type: int, wait_seconds: float) -> QuoteProbeResult:
    label = "LIVE" if market_data_type == 1 else "DELAYED"
    ib.reqMarketDataType(market_data_type)
    ticker = ib.reqMktData(contract, "", False, False)
    try:
        ib.sleep(wait_seconds)
        bid = clean_number(ticker.bid)
        ask = clean_number(ticker.ask)
        last = clean_number(ticker.last)
        close = clean_number(ticker.close)
        return QuoteProbeResult(
            market_data_type=market_data_type,
            label=label,
            bid=bid,
            ask=ask,
            last=last,
            close=close,
            has_two_sided_quote=bid is not None and ask is not None and bid > 0 and ask > 0,
        )
    finally:
        ib.cancelMktData(contract)


def main() -> int:
    load_local_env()
    args = parse_args()
    require_ib()

    ib = IB()
    errors: list[str] = []

    def on_error(req_id: int, error_code: int, error_string: str, contract) -> None:
        symbol = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "")
        errors.append(f"{error_code}: {error_string} {symbol}".strip())

    ib.errorEvent += on_error
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)
        option_contract, reference_price = choose_option_contract(
            ib=ib,
            symbol=args.symbol.upper(),
            exchange=args.exchange,
            currency=args.currency,
            expiry_arg=args.expiry,
            strike_arg=args.strike,
            right=args.right,
            dte_min=args.dte_min,
        )

        live_probe = probe_quote(ib, option_contract, 1, args.wait_seconds)
        delayed_probe = probe_quote(ib, option_contract, 3, args.wait_seconds)

        if live_probe.has_two_sided_quote:
            status = "LIVE"
            selected_probe = live_probe
        elif delayed_probe.has_two_sided_quote:
            status = "DELAYED"
            selected_probe = delayed_probe
        else:
            status = "NOT_AVAILABLE"
            selected_probe = delayed_probe

        print(f"Status: {status}")
        print(
            "Selected Contract: "
            f"{option_contract.localSymbol} | expiry={option_contract.lastTradeDateOrContractMonth} | "
            f"strike={float(option_contract.strike):.2f} | right={option_contract.right}"
        )
        print(f"Reference Underlying Close: {reference_price:.2f}")
        print(
            "Live Probe: "
            f"bid={live_probe.bid} ask={live_probe.ask} last={live_probe.last} close={live_probe.close}"
        )
        print(
            "Delayed Probe: "
            f"bid={delayed_probe.bid} ask={delayed_probe.ask} last={delayed_probe.last} close={delayed_probe.close}"
        )
        print(
            "Chosen Quote Set: "
            f"{selected_probe.label} | bid={selected_probe.bid} ask={selected_probe.ask} "
            f"last={selected_probe.last} close={selected_probe.close}"
        )
        if errors:
            print("IB Errors:")
            for item in errors:
                print(f"- {item}")
        return 0 if status != "NOT_AVAILABLE" else 2
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        print(f"Status: ERROR")
        print(f"Error: {message}")
        if errors:
            print("IB Errors:")
            for item in errors:
                print(f"- {item}")
        return 1
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
