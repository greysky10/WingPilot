from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

from .config import WeeklyCorridorConfig
from .models import (
    ActionRecord,
    TransitionRecord,
    WeeklyActionType,
    WeeklyCenterEstimate,
    WeeklyContext,
    WeeklyRegimeSnapshot,
    WeeklyState,
)
from .strategy import bday_count, build_adjustment_butterfly, build_initial_butterflies, trading_week_key


@dataclass(slots=True)
class WeeklyStepResult:
    transitions: list[TransitionRecord]
    actions: list[ActionRecord]


class WeeklyCorridorStateMachine:
    """Weekly state machine with capped adjustments and explicit weekly exits."""

    def __init__(self, config: WeeklyCorridorConfig) -> None:
        self.config = config
        self.context = WeeklyContext()

    def process_bar(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        price: float,
        regime: WeeklyRegimeSnapshot,
        center: WeeklyCenterEstimate | None,
    ) -> WeeklyStepResult:
        transitions: list[TransitionRecord] = []
        actions: list[ActionRecord] = []
        week_key = trading_week_key(timestamp)
        local_ts = pd.Timestamp(timestamp).tz_convert("America/New_York")

        if self.context.current_week_key is None:
            self.context.current_week_key = week_key

        if self.context.active_butterflies and self.context.current_week_key != week_key:
            self._transition(
                transitions,
                timestamp,
                symbol,
                WeeklyState.EXITED,
                "Rolled into a new week; exiting the prior weekly corridor.",
                regime,
                price,
            )
            self._close_all(actions, timestamp, symbol, week_key, price, "Week rollover exit.", WeeklyActionType.EXIT_WEEK)

        if not self.context.active_butterflies and self.context.current_week_key != week_key:
            self._reset_for_new_week(week_key)

        if self.context.state == WeeklyState.IDLE:
            if regime.event_blocked and self._within_entry_window(local_ts):
                if not self.context.event_week_skipped:
                    actions.append(
                        ActionRecord(
                            timestamp=timestamp,
                            symbol=symbol,
                            week_key=week_key,
                            action=WeeklyActionType.SKIP_EVENT_WEEK,
                            state=self.context.state,
                            price=price,
                            center_price=center.center_price if center else None,
                            layer_id=None,
                            detail="Skipped deployment because this week contains a flagged event date.",
                        )
                    )
                    self.context.event_week_skipped = True
                return WeeklyStepResult(transitions, actions)

            if center is not None and regime.is_range and self._within_entry_window(local_ts):
                butterflies = build_initial_butterflies(
                    self.config,
                    center.center_price,
                    timestamp,
                    week_key,
                    self.context.next_layer_id,
                )
                if butterflies:
                    self.context.active_butterflies.extend(butterflies)
                    self.context.next_layer_id += len(butterflies)
                    self.context.current_center = center.center_price
                    self.context.current_tolerance_low = center.tolerance_low
                    self.context.current_tolerance_high = center.tolerance_high
                    self.context.deployed_at = timestamp
                    self.context.current_week_key = week_key
                    self.context.adjustments_this_week = 0
                    self.context.event_week_skipped = False
                    self._transition(
                        transitions,
                        timestamp,
                        symbol,
                        WeeklyState.ACTIVE,
                        "Deployed the initial 3-butterfly weekly corridor.",
                        regime,
                        price,
                    )
                    for layer in butterflies:
                        actions.append(
                            ActionRecord(
                                timestamp=timestamp,
                                symbol=symbol,
                                week_key=week_key,
                                action=WeeklyActionType.DEPLOY_INITIAL,
                                state=self.context.state,
                                price=price,
                                center_price=self.context.current_center,
                                layer_id=layer.layer_id,
                                detail="Opened an initial weekly butterfly layer.",
                                metadata={
                                    "kind": layer.kind.value,
                                    "body_strike": layer.body_strike,
                                    "lower_strike": layer.lower_strike,
                                    "upper_strike": layer.upper_strike,
                                    "width": layer.width,
                                    "dte": layer.dte,
                                },
                            )
                        )
                return WeeklyStepResult(transitions, actions)

        if self.context.state in {WeeklyState.ACTIVE, WeeklyState.ADJUSTED} and self.context.active_butterflies:
            if regime.is_trend:
                self._transition(
                    transitions,
                    timestamp,
                    symbol,
                    WeeklyState.ABORTED,
                    "Trend or breakout expansion invalidated the weekly corridor.",
                    regime,
                    price,
                )
                self._close_all(actions, timestamp, symbol, week_key, price, "Aborted on trend-dominant week.", WeeklyActionType.ABORT)
                return WeeklyStepResult(transitions, actions)

            exit_action = self._exit_action(local_ts, timestamp)
            if exit_action is not None:
                exit_reason = {
                    WeeklyActionType.EXIT_WEEK: "Forced weekly exit window reached.",
                    WeeklyActionType.EXIT_DTE: "Remaining DTE fell below the minimum buffer.",
                    WeeklyActionType.EXIT_HOLD: "Holding window reached its maximum trading-day length.",
                }[exit_action]
                self._transition(
                    transitions,
                    timestamp,
                    symbol,
                    WeeklyState.EXITED,
                    exit_reason,
                    regime,
                    price,
                )
                self._close_all(actions, timestamp, symbol, week_key, price, exit_reason, exit_action)
                return WeeklyStepResult(transitions, actions)

            if self._should_adjust(price) and self.context.adjustments_this_week < self.config.max_adjustments_per_week:
                if len(self.context.active_butterflies) < self.config.max_active_butterflies:
                    layer = build_adjustment_butterfly(
                        self.config,
                        price,
                        timestamp,
                        week_key,
                        self.context.next_layer_id,
                    )
                    existing_bodies = {round(active.body_strike, 4) for active in self.context.active_butterflies}
                    if round(layer.body_strike, 4) not in existing_bodies:
                        self.context.active_butterflies.append(layer)
                        self.context.next_layer_id += 1
                        self.context.adjustments_this_week += 1
                        self._transition(
                            transitions,
                            timestamp,
                            symbol,
                            WeeklyState.ADJUSTED,
                            "Added a single weekly adjustment butterfly after sustained drift.",
                            regime,
                            price,
                        )
                        actions.append(
                            ActionRecord(
                                timestamp=timestamp,
                                symbol=symbol,
                                week_key=week_key,
                                action=WeeklyActionType.ADD_ADJUSTMENT,
                                state=self.context.state,
                                price=price,
                                center_price=self.context.current_center,
                                layer_id=layer.layer_id,
                                detail="Added one supplemental weekly butterfly after price moved outside tolerance.",
                                metadata={
                                    "kind": layer.kind.value,
                                    "body_strike": layer.body_strike,
                                    "lower_strike": layer.lower_strike,
                                    "upper_strike": layer.upper_strike,
                                    "width": layer.width,
                                    "adjustments_this_week": self.context.adjustments_this_week,
                                    "dte": layer.dte,
                                },
                            )
                        )

        return WeeklyStepResult(transitions, actions)

    def _reset_for_new_week(self, week_key: str) -> None:
        self.context.state = WeeklyState.IDLE
        self.context.current_week_key = week_key
        self.context.current_center = None
        self.context.current_tolerance_low = None
        self.context.current_tolerance_high = None
        self.context.active_butterflies = []
        self.context.deployed_at = None
        self.context.adjustments_this_week = 0
        self.context.last_state_change_at = None
        self.context.event_week_skipped = False

    def _within_entry_window(self, local_ts: pd.Timestamp) -> bool:
        if not (self.config.entry_start_weekday <= local_ts.weekday() <= self.config.entry_end_weekday):
            return False
        start = _parse_clock(self.config.valid_trading_start)
        end = _parse_clock(self.config.valid_trading_end)
        current = local_ts.timetz().replace(tzinfo=None)
        return start <= current <= end

    def _should_adjust(self, price: float) -> bool:
        if self.context.current_center is None:
            return False
        return abs(price - self.context.current_center) > self.config.weekly_center_tolerance

    def _remaining_dte(self, timestamp: pd.Timestamp) -> int:
        if self.context.deployed_at is None:
            return self.config.default_dte
        elapsed_days = max(0, (pd.Timestamp(timestamp) - self.context.deployed_at).days)
        return max(0, self.config.default_dte - elapsed_days)

    def _exit_action(self, local_ts: pd.Timestamp, timestamp: pd.Timestamp) -> WeeklyActionType | None:
        if self.context.deployed_at is None:
            return None
        if self._remaining_dte(timestamp) <= self.config.min_remaining_dte:
            return WeeklyActionType.EXIT_DTE
        if bday_count(self.context.deployed_at, timestamp) >= self.config.max_hold_trading_days:
            return WeeklyActionType.EXIT_HOLD
        forced_time = _parse_clock(self.config.forced_exit_time)
        current = local_ts.timetz().replace(tzinfo=None)
        if local_ts.weekday() >= self.config.forced_exit_weekday and current >= forced_time:
            return WeeklyActionType.EXIT_WEEK
        return None

    def _close_all(
        self,
        actions: list[ActionRecord],
        timestamp: pd.Timestamp,
        symbol: str,
        week_key: str,
        price: float,
        detail: str,
        action_type: WeeklyActionType,
    ) -> None:
        for layer in list(self.context.active_butterflies):
            actions.append(
                ActionRecord(
                    timestamp=timestamp,
                    symbol=symbol,
                    week_key=week_key,
                    action=action_type,
                    state=self.context.state,
                    price=price,
                    center_price=self.context.current_center,
                    layer_id=layer.layer_id,
                    detail=detail,
                    metadata={
                        "kind": layer.kind.value,
                        "body_strike": layer.body_strike,
                        "lower_strike": layer.lower_strike,
                        "upper_strike": layer.upper_strike,
                        "width": layer.width,
                    },
                )
            )
        self.context.active_butterflies = []

    def _transition(
        self,
        transitions: list[TransitionRecord],
        timestamp: pd.Timestamp,
        symbol: str,
        to_state: WeeklyState,
        reason: str,
        regime: WeeklyRegimeSnapshot,
        price: float,
    ) -> None:
        from_state = self.context.state
        self.context.state = to_state
        self.context.last_state_change_at = timestamp
        transitions.append(
            TransitionRecord(
                timestamp=timestamp,
                symbol=symbol,
                week_key=trading_week_key(timestamp),
                from_state=from_state,
                to_state=to_state,
                reason=reason,
                regime=regime.regime,
                price=price,
                center_price=self.context.current_center,
                active_butterflies=len(self.context.active_butterflies),
                adjustments_this_week=self.context.adjustments_this_week,
            )
        )


def _parse_clock(value: str) -> time:
    return pd.Timestamp(f"2000-01-01 {value}").time()
