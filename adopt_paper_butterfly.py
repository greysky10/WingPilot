#! python3.12
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from ib_insync import IB
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    IB = None  # type: ignore[assignment]
    IB_IMPORT_ERROR = exc
else:
    IB_IMPORT_ERROR = None

import pandas as pd

from corridor.execution.paper import CsvEventLogger, ManagedPosition, managed_position_to_payload
from corridor.models import LayerKind
from corridor.options.butterfly_selector import ButterflyCandidate
from strategy import load_local_env


IB_DEPENDENCY_HINT = "py -3.12 -m pip install ib_insync pandas pytz"


@dataclass(slots=True)
class AccountLeg:
    expiry: str
    strike: float
    right: str
    trading_class: str | None
    quantity: int
    local_symbol: str


@dataclass(slots=True)
class AdoptedStructure:
    expiry: str
    right: str
    trading_class: str | None
    lower_strike: float
    body_strike: float
    upper_strike: float
    quantity: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adopt existing butterfly positions from the connected IB paper account into the corridor recovery state."
    )
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol to adopt.")
    parser.add_argument("--host", default=os.getenv("IB_HOST", "localhost"), help="IB host.")
    parser.add_argument("--port", type=int, default=int(os.getenv("IB_PORT", "4002")), help="IB API port.")
    parser.add_argument("--client-id", type=int, default=140, help="IB client id for the adoption scan.")
    parser.add_argument("--output-dir", default="", help="Optional runner output directory. Defaults to corridor_outputs/paper_runner/<symbol>.")
    parser.add_argument("--prefix", default="paper", help="Runner file prefix used for state/recovery files.")
    parser.add_argument("--layer-id-start", type=int, default=1, help="First layer id to assign in the recovery file.")
    parser.add_argument("--opened-at", default="", help="Optional ISO timestamp to use as the adopted open time.")
    parser.add_argument("--write", action="store_true", help="Write the recovery file. Without this flag the script only previews the adoption.")
    return parser.parse_args()


def require_ib() -> None:
    if IB_IMPORT_ERROR is None:
        return
    raise SystemExit(
        "Missing required package: ib_insync. Install it with `"
        + IB_DEPENDENCY_HINT
        + "`."
    )


def find_option_legs(ib: IB, symbol: str) -> list[AccountLeg]:
    legs: list[AccountLeg] = []
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
        legs.append(
            AccountLeg(
                expiry=str(getattr(contract, "lastTradeDateOrContractMonth", "")),
                strike=float(getattr(contract, "strike", 0.0)),
                right=str(getattr(contract, "right", "")).upper(),
                trading_class=str(getattr(contract, "tradingClass", "") or "") or None,
                quantity=quantity,
                local_symbol=str(getattr(contract, "localSymbol", "")),
            )
        )
    return sorted(legs, key=lambda item: (item.expiry, item.right, item.strike))


def infer_long_butterflies(legs: list[AccountLeg]) -> tuple[list[AdoptedStructure], list[AccountLeg]]:
    structures: list[AdoptedStructure] = []
    leftovers: list[AccountLeg] = []
    grouped: dict[tuple[str, str, str | None], Counter[float]] = defaultdict(Counter)
    for leg in legs:
        grouped[(leg.expiry, leg.right, leg.trading_class)][leg.strike] += leg.quantity

    for (expiry, right, trading_class), counts in grouped.items():
        while True:
            candidates: list[tuple[float, float, float, float, int]] = []
            for body_strike, body_qty in counts.items():
                if body_qty >= 0 or abs(body_qty) < 2:
                    continue
                for lower_strike, lower_qty in counts.items():
                    if lower_strike >= body_strike or lower_qty <= 0:
                        continue
                    width = round(body_strike - lower_strike, 6)
                    upper_strike = round(body_strike + width, 6)
                    upper_qty = counts.get(upper_strike, 0)
                    if upper_qty <= 0:
                        continue
                    lots = min(lower_qty, upper_qty, abs(body_qty) // 2)
                    if lots <= 0:
                        continue
                    candidates.append((width, body_strike, lower_strike, upper_strike, lots))

            if not candidates:
                break

            width, body_strike, lower_strike, upper_strike, lots = min(
                candidates,
                key=lambda item: (item[0], abs(item[1]), item[2], item[3]),
            )
            structures.append(
                AdoptedStructure(
                    expiry=expiry,
                    right=right,
                    trading_class=trading_class,
                    lower_strike=lower_strike,
                    body_strike=body_strike,
                    upper_strike=upper_strike,
                    quantity=lots,
                )
            )
            counts[lower_strike] -= lots
            counts[body_strike] += 2 * lots
            counts[upper_strike] -= lots

        for strike, quantity in sorted(counts.items()):
            if quantity:
                leftovers.append(
                    AccountLeg(
                        expiry=expiry,
                        strike=strike,
                        right=right,
                        trading_class=trading_class,
                        quantity=int(quantity),
                        local_symbol="",
                    )
                )
    return structures, leftovers


def assign_layer_kinds(structures: list[AdoptedStructure]) -> list[str]:
    if not structures:
        return []
    mean_body = sum(item.body_strike for item in structures) / len(structures)
    primary_index = min(
        range(len(structures)),
        key=lambda idx: (abs(structures[idx].body_strike - mean_body), idx),
    )
    kinds = [LayerKind.SUPPLEMENTAL.value] * len(structures)
    kinds[primary_index] = LayerKind.PRIMARY.value
    return kinds


def build_recovery_payload(
    symbol: str,
    structures: list[AdoptedStructure],
    layer_id_start: int,
    opened_at: pd.Timestamp,
) -> dict:
    ordered = sorted(structures, key=lambda item: (item.expiry, item.body_strike, item.lower_strike))
    kinds = assign_layer_kinds(ordered)
    managed_positions: list[ManagedPosition] = []
    for offset, (structure, layer_kind) in enumerate(zip(ordered, kinds, strict=True)):
        width = float(structure.upper_strike - structure.body_strike)
        candidate = ButterflyCandidate(
            symbol=symbol,
            expiry=structure.expiry,
            lower_strike=float(structure.lower_strike),
            body_strike=float(structure.body_strike),
            upper_strike=float(structure.upper_strike),
            trading_class=structure.trading_class,
            net_debit=0.0,
            total_spread=0.0,
            max_risk=0.0,
            max_reward=max(0.0, width * 100.0),
            right="CALL" if structure.right == "C" else "PUT",
        )
        managed_positions.append(
            ManagedPosition(
                layer_id=layer_id_start + offset,
                candidate=candidate,
                quantity=int(structure.quantity),
                opened_at=opened_at,
                open_limit=0.0,
                open_status="AdoptedFromIB",
                source_action="ADOPTED",
                layer_kind=layer_kind,
            )
        )

    current_center = next(
        (position.candidate.body_strike for position in managed_positions if position.layer_kind == LayerKind.PRIMARY.value),
        managed_positions[0].candidate.body_strike if managed_positions else None,
    )
    return {
        "symbol": symbol,
        "saved_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "state": "ACTIVE_CENTERED" if managed_positions else "IDLE",
        "current_center": current_center,
        "next_layer_id": layer_id_start + len(managed_positions),
        "positions": [managed_position_to_payload(position) for position in managed_positions],
    }


def main() -> int:
    load_local_env()
    args = parse_args()
    require_ib()

    symbol = args.symbol.upper()
    output_dir = Path(args.output_dir) if args.output_dir else Path("corridor_outputs") / "paper_runner" / symbol
    if args.opened_at:
        parsed = pd.Timestamp(args.opened_at)
        opened_at = parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
    else:
        opened_at = pd.Timestamp.now(tz="UTC")

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)
        legs = find_option_legs(ib, symbol)
        if not legs:
            print(f"No open {symbol} option legs found in the connected IB account.")
            return 1

        print(f"Found {len(legs)} open {symbol} option leg(s):")
        for leg in legs:
            trading_class = f" {leg.trading_class}" if leg.trading_class else ""
            print(f"- {leg.expiry}{trading_class} {leg.right} {leg.strike:.1f} | qty={leg.quantity:+d} | {leg.local_symbol}")

        structures, leftovers = infer_long_butterflies(legs)
        if leftovers:
            print("Adoption failed: not all option legs can be explained as clean long butterflies.")
            for leg in leftovers:
                print(f"- leftover {leg.expiry} {leg.right} {leg.strike:.1f} | qty={leg.quantity:+d}")
            return 1
        if not structures:
            print("Adoption failed: no clean long butterfly structures were inferred from the live account positions.")
            return 1

        payload = build_recovery_payload(symbol, structures, args.layer_id_start, opened_at)
        print(f"Inferred {len(structures)} butterfly structure(s):")
        for index, position in enumerate(payload["positions"], start=1):
            candidate = position["candidate"]
            print(
                f"- layer={position['layer_id']} | kind={position['layer_kind']} | "
                f"{candidate['expiry']} {candidate.get('trading_class') or ''} {candidate['right']} "
                f"{candidate['lower_strike']:.1f}/{candidate['body_strike']:.1f}/{candidate['upper_strike']:.1f} "
                f"| qty={position['quantity']}"
            )

        logger = CsvEventLogger(output_dir, args.prefix)
        if not args.write:
            print(f"Dry run only. Re-run with `--write` to save {logger.paths['recovery']}.")
            return 0

        logger.write_recovery(payload)
        print(f"Wrote recovery state to {logger.paths['recovery']}.")
        print(
            "Restart the runner with `--sync-on-start` so it restores the adopted positions and reconciles them against IB."
        )
        return 0
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        print(f"Adoption failed: {message}")
        return 1
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
