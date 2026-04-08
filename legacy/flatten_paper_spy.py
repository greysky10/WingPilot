#! python3.12
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

try:
    from ib_insync import IB, LimitOrder, MarketOrder
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    IB = None  # type: ignore[assignment]
    LimitOrder = None  # type: ignore[assignment]
    MarketOrder = None  # type: ignore[assignment]
    IB_IMPORT_ERROR = exc
else:
    IB_IMPORT_ERROR = None

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legacy.strategy import load_local_env


IB_DEPENDENCY_HINT = "py -3.12 -m pip install ib_insync pandas pytz"


@dataclass(slots=True)
class CloseIntent:
    local_symbol: str
    expiry: str
    strike: float
    right: str
    position: int
    action: str
    quantity: int
    order_type: str
    limit_price: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten open option positions in the connected IB paper account."
    )
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol to flatten.")
    parser.add_argument("--host", default=os.getenv("IB_HOST", "localhost"), help="IB host.")
    parser.add_argument("--port", type=int, default=int(os.getenv("IB_PORT", "4002")), help="IB API port.")
    parser.add_argument("--client-id", type=int, default=130, help="IB client id for the flatten script.")
    parser.add_argument(
        "--order-type",
        choices=["MKT", "LMT"],
        default="MKT",
        help="Close with market orders or marketable limit orders.",
    )
    parser.add_argument(
        "--mode",
        choices=["delayed", "live"],
        default="delayed",
        help="Market-data mode used only when building limit orders.",
    )
    parser.add_argument("--tif", default="DAY", help="Time in force for close orders.")
    parser.add_argument(
        "--limit-buffer",
        type=float,
        default=0.05,
        help="Extra price aggressiveness for limit closes. BUY uses ask+buffer, SELL uses bid-buffer.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=1.0,
        help="How long to wait after each submitted order before printing status.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually send close orders. Without this flag the script is dry-run only.",
    )
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
    if math.isnan(numeric) or math.isinf(numeric) or numeric <= 0:
        return None
    return numeric


def find_option_positions(ib: IB, symbol: str) -> list:
    matches = []
    for position in ib.positions():
        contract = getattr(position, "contract", None)
        if contract is None:
            continue
        if getattr(contract, "symbol", "").upper() != symbol.upper():
            continue
        if getattr(contract, "secType", "") != "OPT":
            continue
        quantity = int(round(float(getattr(position, "position", 0.0) or 0.0)))
        if quantity == 0:
            continue
        matches.append(position)
    return sorted(
        matches,
        key=lambda item: (
            getattr(item.contract, "lastTradeDateOrContractMonth", ""),
            float(getattr(item.contract, "strike", 0.0)),
            getattr(item.contract, "right", ""),
        ),
    )


def build_close_intent(position, order_type: str) -> CloseIntent:
    contract = position.contract
    quantity = int(round(float(position.position or 0.0)))
    action = "SELL" if quantity > 0 else "BUY"
    return CloseIntent(
        local_symbol=getattr(contract, "localSymbol", ""),
        expiry=str(getattr(contract, "lastTradeDateOrContractMonth", "")),
        strike=float(getattr(contract, "strike", 0.0)),
        right=str(getattr(contract, "right", "")),
        position=quantity,
        action=action,
        quantity=abs(quantity),
        order_type=order_type,
        limit_price=None,
    )


def compute_limit_price(ib: IB, contract, action: str, market_data_type: int, buffer: float) -> float:
    ib.reqMarketDataType(market_data_type)
    ticker = ib.reqMktData(contract, "", False, False)
    try:
        ib.sleep(1.5)
        bid = clean_number(getattr(ticker, "bid", None))
        ask = clean_number(getattr(ticker, "ask", None))
        last = clean_number(getattr(ticker, "last", None))
        close = clean_number(getattr(ticker, "close", None))
    finally:
        ib.cancelMktData(contract)

    reference = ask if action == "BUY" else bid
    if reference is None:
        reference = last or close
    if reference is None:
        raise RuntimeError(
            f"No usable quote available to build a marketable limit order for {getattr(contract, 'localSymbol', contract.symbol)}."
        )

    if action == "BUY":
        return round(max(0.01, reference + max(0.01, buffer)), 2)
    return round(max(0.01, reference - max(0.01, buffer)), 2)


def submit_close_order(
    ib: IB,
    position,
    intent: CloseIntent,
    tif: str,
    market_data_type: int,
    limit_buffer: float,
    wait_seconds: float,
):
    contract = position.contract
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise RuntimeError(f"Unable to qualify option contract for {intent.local_symbol}.")
    contract = qualified[0]

    if intent.order_type == "MKT":
        order = MarketOrder(intent.action, intent.quantity, tif=tif)
    else:
        intent.limit_price = compute_limit_price(ib, contract, intent.action, market_data_type, limit_buffer)
        order = LimitOrder(intent.action, intent.quantity, intent.limit_price, tif=tif)

    order.orderRef = f"flatten:{contract.symbol}:{intent.expiry}:{intent.strike}:{intent.right}"
    trade = ib.placeOrder(contract, order)
    ib.sleep(max(0.5, wait_seconds))
    return trade


def main() -> int:
    load_local_env()
    args = parse_args()
    require_ib()

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)
        positions = find_option_positions(ib, args.symbol)
        if not positions:
            print(f"No open {args.symbol.upper()} option positions found in the connected IB account.")
            return 0

        print(
            f"Found {len(positions)} open {args.symbol.upper()} option position(s) in the connected IB account."
        )
        intents = [build_close_intent(position, args.order_type) for position in positions]
        for intent in intents:
            side = "long" if intent.position > 0 else "short"
            print(
                f"- {intent.local_symbol} | pos={intent.position:+d} ({side}) | "
                f"close_action={intent.action} {intent.quantity} | order_type={intent.order_type}"
            )

        if not args.submit:
            print("Dry run only. Re-run with `--submit` to send the closing orders.")
            return 0

        market_data_type = 3 if args.mode == "delayed" else 1
        for position, intent in zip(positions, intents, strict=True):
            trade = submit_close_order(
                ib=ib,
                position=position,
                intent=intent,
                tif=args.tif,
                market_data_type=market_data_type,
                limit_buffer=args.limit_buffer,
                wait_seconds=args.wait_seconds,
            )
            status = str(getattr(trade.orderStatus, "status", "") or "").strip() or "submitted"
            order_id = getattr(trade.order, "orderId", None)
            suffix = f" | limit={intent.limit_price:.2f}" if intent.limit_price is not None else ""
            print(
                f"Submitted | order_id={order_id} | {intent.local_symbol} | "
                f"{intent.action} {intent.quantity} {intent.order_type} | status={status}{suffix}"
            )
            for entry in getattr(trade, "log", []) or []:
                message = str(getattr(entry, "message", "") or "").strip()
                if message:
                    print(f"  log: {message}")
        return 0
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        print(f"Flatten failed: {message}")
        return 1
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
