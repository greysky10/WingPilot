from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from corridor.config import CorridorConfig
from corridor.options.chain_loader import OptionQuote


@dataclass(slots=True)
class ButterflyCandidate:
    symbol: str
    expiry: str
    lower_strike: float
    body_strike: float
    upper_strike: float
    net_debit: float
    total_spread: float
    max_risk: float
    max_reward: float
    right: str


def select_butterflies(
    quotes: Iterable[OptionQuote],
    center_price: float,
    width: float,
    config: CorridorConfig,
) -> list[ButterflyCandidate]:
    """Select candidate butterflies around the current center."""

    by_key: dict[tuple[str, float, str], OptionQuote] = {}
    for quote in quotes:
        by_key[(quote.expiry, quote.strike, quote.right)] = quote

    candidates: list[ButterflyCandidate] = []
    body = round(round(center_price / config.center_rounding) * config.center_rounding, 6)
    lower = body - width
    upper = body + width

    for expiry in sorted({quote.expiry for quote in quotes}):
        call_lower = by_key.get((expiry, lower, "CALL"))
        call_body = by_key.get((expiry, body, "CALL"))
        call_upper = by_key.get((expiry, upper, "CALL"))
        if call_lower and call_body and call_upper:
            spread = call_lower.spread + 2.0 * call_body.spread + call_upper.spread
            debit = call_lower.mid - 2.0 * call_body.mid + call_upper.mid
            if debit > 0 and spread <= config.max_acceptable_option_spread:
                candidates.append(
                    ButterflyCandidate(
                        symbol=call_body.symbol,
                        expiry=expiry,
                        lower_strike=lower,
                        body_strike=body,
                        upper_strike=upper,
                        net_debit=debit,
                        total_spread=spread,
                        max_risk=debit * 100.0,
                        max_reward=max(0.0, (width - debit) * 100.0),
                        right="CALL",
                    )
                )
    return sorted(candidates, key=lambda item: (item.total_spread, item.net_debit))
