from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class Regime(str, Enum):
    RANGE = "RANGE"
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    NEUTRAL = "NEUTRAL"


class CenterMethod(str, Enum):
    MEAN_MID = "mean_mid"
    VWAP = "vwap"
    ROLLING_POC = "rolling_poc"


class CorridorState(str, Enum):
    IDLE = "IDLE"
    ACTIVE_CENTERED = "ACTIVE_CENTERED"
    DRIFTING = "DRIFTING"
    REBUILD = "REBUILD"
    ABORT = "ABORT"


class ActionType(str, Enum):
    ENTER_PRIMARY = "ENTER_PRIMARY"
    ADD_SUPPLEMENTAL = "ADD_SUPPLEMENTAL"
    ENTRY_FILTERED = "ENTRY_FILTERED"
    DRIFT_STARTED = "DRIFT_STARTED"
    DRIFT_RESOLVED = "DRIFT_RESOLVED"
    REBUILD_REQUESTED = "REBUILD_REQUESTED"
    REBUILT = "REBUILT"
    ABORTED = "ABORTED"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    MAX_HOLD = "MAX_HOLD"
    SESSION_FLUSH = "SESSION_FLUSH"
    LIVE_PREP = "LIVE_PREP"


class LayerKind(str, Enum):
    PRIMARY = "PRIMARY"
    SUPPLEMENTAL = "SUPPLEMENTAL"


@dataclass(slots=True)
class CenterEstimate:
    timestamp: pd.Timestamp
    center_price: float
    lower_band: float
    upper_band: float
    tolerance_low: float
    tolerance_high: float
    method: CenterMethod
    confidence: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def actual_tolerance(self) -> float:
        return max(0.0, (float(self.tolerance_high) - float(self.tolerance_low)) / 2.0)


@dataclass(slots=True)
class RegimeSnapshot:
    timestamp: pd.Timestamp
    regime: Regime
    range_width_pct: float
    trend_slope_pct: float
    momentum_pct: float
    volume_ratio: float
    breakout_up: bool
    breakout_down: bool

    @property
    def is_trend(self) -> bool:
        return self.regime in {Regime.TREND_UP, Regime.TREND_DOWN}


@dataclass(slots=True)
class ActiveButterfly:
    layer_id: int
    kind: LayerKind
    center_price: float
    width: float
    lower_width: float
    upper_width: float
    lower_strike: float
    body_strike: float
    upper_strike: float
    created_at: pd.Timestamp
    dte: int
    entry_debit: float = 0.0
    entry_friction_cost: float = 0.0
    entry_cost: float = 0.0
    last_mark: float = 0.0
    closed_at: Optional[pd.Timestamp] = None
    close_friction_cost: float = 0.0
    exit_value: float = 0.0
    exit_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CorridorContext:
    state: CorridorState = CorridorState.IDLE
    current_center: Optional[float] = None
    drift_count: int = 0
    last_rebuild_at: Optional[pd.Timestamp] = None
    last_abort_at: Optional[pd.Timestamp] = None
    next_layer_id: int = 1
    active_layers: list[ActiveButterfly] = field(default_factory=list)
    realized_pnl: float = 0.0
    last_state_change_at: Optional[pd.Timestamp] = None
    session_date: Optional[str] = None
    previous_session_close: Optional[float] = None
    current_session_gap_pct: Optional[float] = None
    current_session_gap_filtered: bool = False
    last_processed_close: Optional[float] = None
    last_primary_entry_session_date: Optional[str] = None
    last_take_profit_session_date: Optional[str] = None


@dataclass(slots=True)
class TransitionRecord:
    timestamp: pd.Timestamp
    symbol: str
    from_state: CorridorState
    to_state: CorridorState
    reason: str
    regime: Regime
    price: float
    center_price: Optional[float]
    drift_count: int
    layer_count: int


@dataclass(slots=True)
class ActionRecord:
    timestamp: pd.Timestamp
    symbol: str
    action: ActionType
    state: CorridorState
    price: float
    center_price: Optional[float]
    layer_id: Optional[int]
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EquityPoint:
    timestamp: pd.Timestamp
    symbol: str
    price: float
    regime: Regime
    state: CorridorState
    bar_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    gross_realized_pnl: float
    gross_unrealized_pnl: float
    gross_total_equity: float
    total_equity: float
    modeled_capital_at_risk: float
    corridor_occupancy: bool
    active_layers: int


@dataclass(slots=True)
class BacktestArtifacts:
    transitions_path: Path
    actions_path: Path
    summary_path: Path
    equity_curve_path: Path
