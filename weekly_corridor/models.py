from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from corridor.models import CenterMethod


class WeeklyRegime(str, Enum):
    RANGE = "RANGE"
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    NEUTRAL = "NEUTRAL"
    EVENT_BLOCKED = "EVENT_BLOCKED"


class WeeklyState(str, Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    ADJUSTED = "ADJUSTED"
    ABORTED = "ABORTED"
    EXITED = "EXITED"


class WeeklyActionType(str, Enum):
    DEPLOY_INITIAL = "DEPLOY_INITIAL"
    ADD_ADJUSTMENT = "ADD_ADJUSTMENT"
    ABORT = "ABORT"
    EXIT_HOLD = "EXIT_HOLD"
    EXIT_DTE = "EXIT_DTE"
    EXIT_WEEK = "EXIT_WEEK"
    SKIP_EVENT_WEEK = "SKIP_EVENT_WEEK"


class WeeklyLayerKind(str, Enum):
    INITIAL = "INITIAL"
    ADJUSTMENT = "ADJUSTMENT"


@dataclass(slots=True)
class WeeklyCenterEstimate:
    timestamp: pd.Timestamp
    center_price: float
    lower_coverage: float
    upper_coverage: float
    tolerance_low: float
    tolerance_high: float
    method: CenterMethod
    confidence: float
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class WeeklyRegimeSnapshot:
    timestamp: pd.Timestamp
    regime: WeeklyRegime
    width_pct: float
    slope_pct: float
    momentum_pct: float
    breakout_up: bool
    breakout_down: bool
    event_blocked: bool = False
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def is_trend(self) -> bool:
        return self.regime in {WeeklyRegime.TREND_UP, WeeklyRegime.TREND_DOWN}

    @property
    def is_range(self) -> bool:
        return self.regime == WeeklyRegime.RANGE


@dataclass(slots=True)
class WeeklyButterfly:
    layer_id: int
    kind: WeeklyLayerKind
    week_key: str
    center_price: float
    width: float
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
class WeeklyContext:
    state: WeeklyState = WeeklyState.IDLE
    current_week_key: Optional[str] = None
    current_center: Optional[float] = None
    current_tolerance_low: Optional[float] = None
    current_tolerance_high: Optional[float] = None
    active_butterflies: list[WeeklyButterfly] = field(default_factory=list)
    next_layer_id: int = 1
    deployed_at: Optional[pd.Timestamp] = None
    adjustments_this_week: int = 0
    realized_pnl: float = 0.0
    last_state_change_at: Optional[pd.Timestamp] = None
    event_week_skipped: bool = False


@dataclass(slots=True)
class TransitionRecord:
    timestamp: pd.Timestamp
    symbol: str
    week_key: str
    from_state: WeeklyState
    to_state: WeeklyState
    reason: str
    regime: WeeklyRegime
    price: float
    center_price: Optional[float]
    active_butterflies: int
    adjustments_this_week: int


@dataclass(slots=True)
class ActionRecord:
    timestamp: pd.Timestamp
    symbol: str
    week_key: str
    action: WeeklyActionType
    state: WeeklyState
    price: float
    center_price: Optional[float]
    layer_id: Optional[int]
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EquityPoint:
    timestamp: pd.Timestamp
    symbol: str
    week_key: str
    price: float
    regime: WeeklyRegime
    state: WeeklyState
    bar_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    gross_realized_pnl: float
    gross_unrealized_pnl: float
    gross_total_equity: float
    total_equity: float
    gross_deployment: float
    weekly_occupancy: bool
    active_butterflies: int


@dataclass(slots=True)
class WeeklyBacktestResult:
    transitions: list[TransitionRecord]
    actions: list[ActionRecord]
    equity_curve: list[EquityPoint]
    summary: dict[str, Any]


@dataclass(slots=True)
class BacktestArtifacts:
    transitions_path: Path
    actions_path: Path
    closed_layers_path: Path
    summary_path: Path
    equity_curve_path: Path
