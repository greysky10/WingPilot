from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from corridor.models import ActionRecord, EquityPoint, TransitionRecord


def transitions_to_frame(records: list[TransitionRecord]) -> pd.DataFrame:
    rows = [
        {
            "timestamp": record.timestamp.isoformat(),
            "symbol": record.symbol,
            "from_state": record.from_state.value,
            "to_state": record.to_state.value,
            "reason": record.reason,
            "regime": record.regime.value,
            "price": record.price,
            "center_price": record.center_price,
            "drift_count": record.drift_count,
            "layer_count": record.layer_count,
        }
        for record in records
    ]
    return pd.DataFrame(rows)


def actions_to_frame(records: list[ActionRecord]) -> pd.DataFrame:
    rows = [
        {
            "timestamp": record.timestamp.isoformat(),
            "symbol": record.symbol,
            "action": record.action.value,
            "state": record.state.value,
            "price": record.price,
            "center_price": record.center_price,
            "layer_id": record.layer_id,
            "detail": record.detail,
            **record.metadata,
        }
        for record in records
    ]
    return pd.DataFrame(rows)


def equity_to_frame(records: list[EquityPoint]) -> pd.DataFrame:
    rows = [
        {
            "timestamp": record.timestamp.isoformat(),
            "symbol": record.symbol,
            "price": record.price,
            "regime": record.regime.value,
            "state": record.state.value,
            "bar_pnl": record.bar_pnl,
            "realized_pnl": record.realized_pnl,
            "unrealized_pnl": record.unrealized_pnl,
            "gross_realized_pnl": record.gross_realized_pnl,
            "gross_unrealized_pnl": record.gross_unrealized_pnl,
            "gross_total_equity": record.gross_total_equity,
            "total_equity": record.total_equity,
            "modeled_capital_at_risk": record.modeled_capital_at_risk,
            "corridor_occupancy": record.corridor_occupancy,
            "active_layers": record.active_layers,
        }
        for record in records
    ]
    return pd.DataFrame(rows)
