from __future__ import annotations

from datetime import UTC, date, datetime
from dataclasses import dataclass
from typing import Iterable, Optional

from corridor.config import CorridorConfig
from corridor.options.chain_loader import OptionQuote


@dataclass(slots=True)
class ButterflyCandidate:
    symbol: str
    expiry: str
    lower_strike: float
    body_strike: float
    upper_strike: float
    lower_width: float
    upper_width: float
    net_debit: float
    total_spread: float
    max_risk: float
    max_reward: float
    right: str
    trading_class: str | None = None
    wing_mode: str = "symmetric"
    spread_ratio: float = 0.0
    reward_to_risk: float = 0.0
    body_distance: float = 0.0
    calendar_dte: int | None = None


@dataclass(slots=True)
class ButterflySelectionDiagnostics:
    available_quotes: int
    expiries_considered: int
    call_bodies_considered: int
    attempted_structures: int
    rejection_counts: dict[str, int]
    sample_rejections: list[dict[str, float | str]]


def select_butterflies(
    quotes: Iterable[OptionQuote],
    center_price: float,
    width: float,
    config: CorridorConfig,
    reference_date: Optional[date | str] = None,
) -> list[ButterflyCandidate]:
    candidates, _diagnostics = select_butterflies_with_diagnostics(
        quotes,
        center_price,
        width,
        config,
        reference_date=reference_date,
    )
    return candidates


def select_butterflies_with_diagnostics(
    quotes: Iterable[OptionQuote],
    center_price: float,
    width: float,
    config: CorridorConfig,
    reference_date: Optional[date | str] = None,
) -> tuple[list[ButterflyCandidate], ButterflySelectionDiagnostics]:
    """Select candidate butterflies around the current center."""

    by_key: dict[tuple[str, float, str], OptionQuote] = {}
    for quote in quotes:
        by_key[(quote.expiry, quote.strike, quote.right)] = quote

    candidates: list[ButterflyCandidate] = []
    rejection_counts = {
        "missing_legs": 0,
        "non_positive_debit": 0,
        "spread_too_wide": 0,
    }
    sample_rejections: list[dict[str, float | str]] = []
    target_body = round(round(center_price / config.center_rounding) * config.center_rounding, 6)
    max_body_distance = max(config.center_rounding, config.center_rounding * max(0, config.candidate_body_search_steps))
    expiries_considered = 0
    call_bodies_considered = 0
    attempted_structures = 0
    allowed_rights = _allowed_option_rights(config)

    for expiry in sorted({quote.expiry for quote in quotes}):
        expiry_had_quotes = False
        for option_right in sorted(allowed_rights):
            right_quotes = {
                quote.strike: quote
                for quote in quotes
                if quote.expiry == expiry and quote.right == option_right
            }
            if not right_quotes:
                continue
            expiry_had_quotes = True
            for body in sorted(right_quotes):
                if abs(body - target_body) > max_body_distance:
                    continue
                call_bodies_considered += 1
                allowed_modes = _allowed_wing_modes(config)
                if "symmetric" in allowed_modes:
                    attempted_structures += 1
                    candidate, reason = _build_candidate(
                        right_quotes,
                        option_right,
                        body,
                        width,
                        width,
                        "symmetric",
                        target_body,
                        reference_date=reference_date,
                    )
                    if candidate is not None and candidate.total_spread <= config.max_acceptable_option_spread_for_dte(candidate.calendar_dte):
                        candidates.append(candidate)
                    elif candidate is None:
                        _record_rejection(rejection_counts, sample_rejections, expiry, body, "symmetric", reason, None)
                    else:
                        _record_rejection(rejection_counts, sample_rejections, expiry, body, "symmetric", "spread_too_wide", candidate)
                extra_width = max(0.0, float(config.broken_wing_extra_width))
                if extra_width > 0.0 and "broken_upper" in allowed_modes:
                    attempted_structures += 1
                    candidate, reason = _build_candidate(
                        right_quotes,
                        option_right,
                        body,
                        width,
                        width + extra_width,
                        "broken_upper",
                        target_body,
                        reference_date=reference_date,
                    )
                    if candidate is not None and candidate.total_spread <= config.max_acceptable_option_spread_for_dte(candidate.calendar_dte):
                        candidates.append(candidate)
                    elif candidate is None:
                        _record_rejection(rejection_counts, sample_rejections, expiry, body, "broken_upper", reason, None)
                    else:
                        _record_rejection(rejection_counts, sample_rejections, expiry, body, "broken_upper", "spread_too_wide", candidate)
                if extra_width > 0.0 and "broken_lower" in allowed_modes:
                    attempted_structures += 1
                    candidate, reason = _build_candidate(
                        right_quotes,
                        option_right,
                        body,
                        width + extra_width,
                        width,
                        "broken_lower",
                        target_body,
                        reference_date=reference_date,
                    )
                    if candidate is not None and candidate.total_spread <= config.max_acceptable_option_spread_for_dte(candidate.calendar_dte):
                        candidates.append(candidate)
                    elif candidate is None:
                        _record_rejection(rejection_counts, sample_rejections, expiry, body, "broken_lower", reason, None)
                    else:
                        _record_rejection(rejection_counts, sample_rejections, expiry, body, "broken_lower", "spread_too_wide", candidate)
        if expiry_had_quotes:
            expiries_considered += 1
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            0 if item.wing_mode == "symmetric" else 1,
            item.body_distance,
            item.spread_ratio,
            item.total_spread,
            -item.reward_to_risk,
            item.net_debit,
        ),
    )
    diagnostics = ButterflySelectionDiagnostics(
        available_quotes=len(by_key),
        expiries_considered=expiries_considered,
        call_bodies_considered=call_bodies_considered,
        attempted_structures=attempted_structures,
        rejection_counts=rejection_counts,
        sample_rejections=sample_rejections[:8],
    )
    return sorted_candidates, diagnostics


def _allowed_wing_modes(config: CorridorConfig) -> set[str]:
    if config.wing_mode == "adaptive":
        return {"symmetric", "broken_upper", "broken_lower"}
    return {str(config.wing_mode)}


def _build_candidate(
    right_quotes: dict[float, OptionQuote],
    option_right: str,
    body: float,
    lower_width: float,
    upper_width: float,
    wing_mode: str,
    target_body: float,
    reference_date: Optional[date | str] = None,
) -> tuple[ButterflyCandidate | None, str | None]:
    lower = round(body - lower_width, 6)
    upper = round(body + upper_width, 6)
    lower_quote = right_quotes.get(lower)
    body_quote = right_quotes.get(body)
    upper_quote = right_quotes.get(upper)
    if not (lower_quote and body_quote and upper_quote):
        return None, "missing_legs"
    spread = lower_quote.spread + 2.0 * body_quote.spread + upper_quote.spread
    debit = lower_quote.mid - 2.0 * body_quote.mid + upper_quote.mid
    if debit <= 0:
        return None, "non_positive_debit"
    reward_width = min(lower_width, upper_width)
    extra_tail_risk = abs(upper_width - lower_width)
    max_risk = max(0.0, (debit + extra_tail_risk) * 100.0)
    max_reward = max(0.0, (reward_width - debit) * 100.0)
    spread_ratio = spread / max(debit, 0.01)
    reward_to_risk = max_reward / max(max_risk, 0.01)
    calendar_dte = _candidate_calendar_dte(body_quote.expiry, reference_date)
    return ButterflyCandidate(
        symbol=body_quote.symbol,
        expiry=body_quote.expiry if hasattr(body_quote, "expiry") else body_quote.symbol,
        lower_strike=lower,
        body_strike=body,
        upper_strike=upper,
        lower_width=lower_width,
        upper_width=upper_width,
        trading_class=body_quote.trading_class,
        net_debit=debit,
        total_spread=spread,
        max_risk=max_risk,
        max_reward=max_reward,
        right=str(option_right).upper(),
        wing_mode=wing_mode,
        spread_ratio=spread_ratio,
        reward_to_risk=reward_to_risk,
        body_distance=abs(body - target_body),
        calendar_dte=calendar_dte,
    ), None


def _allowed_option_rights(config: CorridorConfig) -> set[str]:
    preference = str(config.option_right_preference).strip().lower()
    if preference == "put":
        return {"PUT"}
    if preference == "auto":
        return {"CALL", "PUT"}
    return {"CALL"}


def _record_rejection(
    rejection_counts: dict[str, int],
    sample_rejections: list[dict[str, float | str]],
    expiry: str,
    body: float,
    wing_mode: str,
    reason: str | None,
    candidate: ButterflyCandidate | None,
) -> None:
    key = str(reason or "unknown")
    rejection_counts[key] = rejection_counts.get(key, 0) + 1
    if len(sample_rejections) >= 8:
        return
    sample: dict[str, float | str] = {
        "expiry": expiry,
        "body_strike": body,
        "wing_mode": wing_mode,
        "reason": key,
    }
    if candidate is not None:
        sample["net_debit"] = round(candidate.net_debit, 4)
        sample["total_spread"] = round(candidate.total_spread, 4)
        sample["spread_ratio"] = round(candidate.spread_ratio, 4)
        sample["lower_strike"] = round(candidate.lower_strike, 4)
        sample["upper_strike"] = round(candidate.upper_strike, 4)
        if candidate.calendar_dte is not None:
            sample["calendar_dte"] = int(candidate.calendar_dte)
    sample_rejections.append(sample)


def _candidate_calendar_dte(expiry: str, reference_date: Optional[date | str]) -> int | None:
    try:
        expiry_date = _parse_expiry_date(expiry)
    except ValueError:
        return None
    base_date = _coerce_reference_date(reference_date)
    return int((expiry_date - base_date).days)


def _coerce_reference_date(reference_date: Optional[date | str]) -> date:
    if reference_date is None:
        return datetime.now(UTC).date()
    if isinstance(reference_date, date):
        return reference_date
    raw = str(reference_date).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return datetime.fromisoformat(raw).date()


def _parse_expiry_date(expiry: str) -> date:
    raw = str(expiry).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return datetime.fromisoformat(raw).date()
