from __future__ import annotations

from dataclasses import dataclass


try:
    from ib_insync import Bag, ComboLeg, Contract
except ImportError:  # pragma: no cover - optional dependency
    Bag = None
    ComboLeg = None
    Contract = None


@dataclass(slots=True)
class ComboLegSpec:
    con_id: int
    ratio: int
    action: str
    exchange: str = "SMART"


def build_butterfly_combo(symbol: str, currency: str, exchange: str, legs: list[ComboLegSpec]):
    """Build a BAG/combination contract scaffold for future paper trading."""

    if Bag is None or ComboLeg is None or Contract is None:
        raise RuntimeError("ib_insync is required for combo contract building.")

    combo = Contract()
    combo.symbol = symbol
    combo.secType = "BAG"
    combo.currency = currency
    combo.exchange = exchange
    combo.comboLegs = [
        ComboLeg(conId=leg.con_id, ratio=leg.ratio, action=leg.action, exchange=leg.exchange)
        for leg in legs
    ]
    return combo
