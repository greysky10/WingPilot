from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Optional

import pandas as pd

from corridor.config import CorridorConfig
from corridor.models import (
    ActionRecord,
    ActionType,
    ActiveButterfly,
    CenterEstimate,
    CorridorContext,
    CorridorState,
    LayerKind,
    Regime,
    RegimeSnapshot,
    TransitionRecord,
)
from corridor.strategy.recenter_rules import DriftAssessment, RecenterRuleEngine


@dataclass(slots=True)
class CorridorStepResult:
    transitions: list[TransitionRecord]
    actions: list[ActionRecord]


def _parse_time(value: str) -> time:
    hour, minute = value.split(":", maxsplit=1)
    return time(hour=int(hour), minute=int(minute))


class CorridorStateMachine:
    """Finite-state machine for the dynamic butterfly corridor."""

    def __init__(self, config: CorridorConfig) -> None:
        self.config = config
        self.context = CorridorContext()
        self.rules = RecenterRuleEngine(config)
        self.start_time = _parse_time(config.valid_trading_start)
        self.end_time = _parse_time(config.valid_trading_end)
        self.primary_entry_end_time = _parse_time(config.primary_entry_end)

    def process_bar(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        price: float,
        regime: Optional[RegimeSnapshot],
        center: Optional[CenterEstimate],
    ) -> CorridorStepResult:
        transitions: list[TransitionRecord] = []
        actions: list[ActionRecord] = []
        ctx = self.context

        if timestamp.tzinfo is not None:
            local = timestamp.tz_convert("America/New_York")
        else:
            local = timestamp.tz_localize("UTC").tz_convert("America/New_York")
        in_window = self.start_time <= local.time() <= self.end_time

        if not in_window:
            if self.config.hold_overnight and ctx.active_layers:
                if ctx.state != CorridorState.ACTIVE_CENTERED:
                    transitions.append(
                        self._transition(
                            symbol,
                            timestamp,
                            CorridorState.ACTIVE_CENTERED,
                            "Holding active positions overnight outside the trading window.",
                            regime,
                            price,
                        )
                    )
                ctx.state = CorridorState.ACTIVE_CENTERED
                ctx.drift_count = 0
                if center is not None:
                    ctx.current_center = center.center_price
                return CorridorStepResult(transitions=transitions, actions=actions)

            if ctx.state in {CorridorState.ACTIVE_CENTERED, CorridorState.DRIFTING, CorridorState.REBUILD}:
                transitions.append(self._transition(symbol, timestamp, CorridorState.IDLE, "Outside valid trading window.", regime, price))
                actions.extend(self._close_all_layers(symbol, timestamp, price, ActionType.SESSION_FLUSH, "Session window closed."))
                ctx.state = CorridorState.IDLE
                ctx.drift_count = 0
                ctx.current_center = None
                return CorridorStepResult(transitions=transitions, actions=actions)

        assessment = self.rules.evaluate(timestamp, price, center, regime, ctx.drift_count, ctx.last_rebuild_at)
        if assessment.should_abort and ctx.state != CorridorState.ABORT:
            transitions.append(self._transition(symbol, timestamp, CorridorState.ABORT, assessment.abort_reason, regime, price))
            actions.extend(self._close_all_layers(symbol, timestamp, price, ActionType.ABORTED, assessment.abort_reason))
            ctx.state = CorridorState.ABORT
            ctx.current_center = center.center_price if center else None
            ctx.last_abort_at = timestamp
            ctx.drift_count = 0
            return CorridorStepResult(transitions=transitions, actions=actions)

        if ctx.state == CorridorState.ABORT and in_window and regime is not None and regime.regime == Regime.RANGE and center is not None:
            transitions.append(self._transition(symbol, timestamp, CorridorState.IDLE, "Range conditions returned after abort.", regime, price))
            ctx.state = CorridorState.IDLE

        if ctx.state == CorridorState.IDLE:
            if in_window and self._primary_entry_allowed(local, regime, center):
                layer = self._open_layer(timestamp, center.center_price, LayerKind.PRIMARY, self.config.default_dte)
                ctx.active_layers = [layer]
                ctx.current_center = center.center_price
                ctx.drift_count = 0
                transitions.append(self._transition(symbol, timestamp, CorridorState.ACTIVE_CENTERED, "Entered range corridor.", regime, price))
                actions.append(
                    ActionRecord(
                        timestamp=timestamp,
                        symbol=symbol,
                        action=ActionType.ENTER_PRIMARY,
                        state=CorridorState.ACTIVE_CENTERED,
                        price=price,
                        center_price=center.center_price,
                        layer_id=layer.layer_id,
                        detail="Opened the primary butterfly corridor layer.",
                        metadata=self._layer_metadata(layer),
                    )
                )
                ctx.state = CorridorState.ACTIVE_CENTERED
                return CorridorStepResult(transitions=transitions, actions=actions)
            return CorridorStepResult(transitions=transitions, actions=actions)

        if ctx.state == CorridorState.ACTIVE_CENTERED:
            ctx.current_center = center.center_price if center else ctx.current_center
            if center is not None and assessment.outside_tolerance:
                ctx.drift_count = assessment.next_drift_count
                transitions.append(self._transition(symbol, timestamp, CorridorState.DRIFTING, "Price moved outside the tolerance band.", regime, price))
                actions.append(
                    ActionRecord(
                        timestamp=timestamp,
                        symbol=symbol,
                        action=ActionType.DRIFT_STARTED,
                        state=CorridorState.DRIFTING,
                        price=price,
                        center_price=center.center_price,
                        layer_id=None,
                        detail="Corridor drift started.",
                        metadata={"drift_distance": round(assessment.drift_distance, 4)},
                    )
                )
                ctx.state = CorridorState.DRIFTING
                return CorridorStepResult(transitions=transitions, actions=actions)

            if center is not None and self._should_add_supplemental_layer(price, center) and len(ctx.active_layers) < self.config.max_active_butterfly_layers:
                layer = self._open_layer(timestamp, (price + center.center_price) / 2.0, LayerKind.SUPPLEMENTAL, self.config.default_dte)
                ctx.active_layers.append(layer)
                actions.append(
                    ActionRecord(
                        timestamp=timestamp,
                        symbol=symbol,
                        action=ActionType.ADD_SUPPLEMENTAL,
                        state=CorridorState.ACTIVE_CENTERED,
                        price=price,
                        center_price=center.center_price,
                        layer_id=layer.layer_id,
                        detail="Added a supplemental butterfly layer near the edge of the corridor.",
                        metadata=self._layer_metadata(layer),
                    )
                )
            return CorridorStepResult(transitions=transitions, actions=actions)

        if ctx.state == CorridorState.DRIFTING:
            ctx.current_center = center.center_price if center else ctx.current_center
            ctx.drift_count = assessment.next_drift_count
            if center is not None and not assessment.outside_tolerance:
                transitions.append(self._transition(symbol, timestamp, CorridorState.ACTIVE_CENTERED, "Price returned to the center band.", regime, price))
                actions.append(
                    ActionRecord(
                        timestamp=timestamp,
                        symbol=symbol,
                        action=ActionType.DRIFT_RESOLVED,
                        state=CorridorState.ACTIVE_CENTERED,
                        price=price,
                        center_price=center.center_price,
                        layer_id=None,
                        detail="Drift resolved without rebuilding.",
                    )
                )
                ctx.state = CorridorState.ACTIVE_CENTERED
                ctx.drift_count = 0
                return CorridorStepResult(transitions=transitions, actions=actions)

            if center is not None and assessment.should_rebuild:
                transitions.append(self._transition(symbol, timestamp, CorridorState.REBUILD, "Drift persisted and rebuild cooldown passed.", regime, price))
                actions.append(
                    ActionRecord(
                        timestamp=timestamp,
                        symbol=symbol,
                        action=ActionType.REBUILD_REQUESTED,
                        state=CorridorState.REBUILD,
                        price=price,
                        center_price=center.center_price,
                        layer_id=None,
                        detail="Requested corridor rebuild after sustained drift.",
                        metadata={"drift_count": ctx.drift_count, "drift_distance": round(assessment.drift_distance, 4)},
                    )
                )
                ctx.state = CorridorState.REBUILD
                return CorridorStepResult(transitions=transitions, actions=actions)

            return CorridorStepResult(transitions=transitions, actions=actions)

        if ctx.state == CorridorState.REBUILD:
            if center is None:
                return CorridorStepResult(transitions=transitions, actions=actions)
            actions.extend(self._close_all_layers(symbol, timestamp, price, ActionType.REBUILT, "Removed prior layers for rebuild."))
            layer = self._open_layer(timestamp, center.center_price, LayerKind.PRIMARY, self.config.default_dte)
            ctx.active_layers = [layer]
            ctx.current_center = center.center_price
            ctx.last_rebuild_at = timestamp
            ctx.drift_count = 0
            transitions.append(self._transition(symbol, timestamp, CorridorState.ACTIVE_CENTERED, "Rebuilt corridor around the new center.", regime, price))
            actions.append(
                ActionRecord(
                    timestamp=timestamp,
                    symbol=symbol,
                    action=ActionType.REBUILT,
                    state=CorridorState.ACTIVE_CENTERED,
                    price=price,
                    center_price=center.center_price,
                    layer_id=layer.layer_id,
                    detail="Established a fresh primary butterfly corridor layer.",
                    metadata=self._layer_metadata(layer),
                )
            )
            ctx.state = CorridorState.ACTIVE_CENTERED
            return CorridorStepResult(transitions=transitions, actions=actions)

        return CorridorStepResult(transitions=transitions, actions=actions)

    def _transition(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        to_state: CorridorState,
        reason: str,
        regime: Optional[RegimeSnapshot],
        price: float,
    ) -> TransitionRecord:
        ctx = self.context
        record = TransitionRecord(
            timestamp=timestamp,
            symbol=symbol,
            from_state=ctx.state,
            to_state=to_state,
            reason=reason,
            regime=regime.regime if regime is not None else Regime.NEUTRAL,
            price=price,
            center_price=ctx.current_center,
            drift_count=ctx.drift_count,
            layer_count=len(ctx.active_layers),
        )
        ctx.last_state_change_at = timestamp
        return record

    def _open_layer(self, timestamp: pd.Timestamp, center_price: float, kind: LayerKind, dte: int) -> ActiveButterfly:
        ctx = self.context
        rounded_center = round(center_price / self.config.center_rounding) * self.config.center_rounding
        width = self.config.butterfly_width
        extra_width = max(0.0, float(self.config.broken_wing_extra_width))
        lower_width = width
        upper_width = width
        if self.config.wing_mode == "broken_upper":
            upper_width = width + extra_width
        elif self.config.wing_mode == "broken_lower":
            lower_width = width + extra_width
        layer = ActiveButterfly(
            layer_id=ctx.next_layer_id,
            kind=kind,
            center_price=rounded_center,
            width=width,
            lower_width=lower_width,
            upper_width=upper_width,
            lower_strike=rounded_center - lower_width,
            body_strike=rounded_center,
            upper_strike=rounded_center + upper_width,
            created_at=timestamp,
            dte=dte,
        )
        layer.metadata["wing_mode"] = self.config.wing_mode
        ctx.next_layer_id += 1
        return layer

    def _close_all_layers(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        price: float,
        action_type: ActionType,
        detail: str,
        extra_metadata: Optional[dict[str, float | str]] = None,
    ) -> list[ActionRecord]:
        ctx = self.context
        actions: list[ActionRecord] = []
        for layer in ctx.active_layers:
            layer.closed_at = timestamp
            layer.exit_reason = detail
            metadata = self._layer_metadata(layer)
            if extra_metadata:
                metadata.update(extra_metadata)
            actions.append(
                ActionRecord(
                    timestamp=timestamp,
                    symbol=symbol,
                    action=action_type,
                    state=ctx.state,
                    price=price,
                    center_price=ctx.current_center,
                    layer_id=layer.layer_id,
                    detail=detail,
                    metadata=metadata,
                )
            )
        ctx.active_layers = []
        return actions

    def flatten_positions(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        price: float,
        action_type: ActionType,
        detail: str,
        regime: Optional[RegimeSnapshot],
        extra_metadata: Optional[dict[str, float | str]] = None,
    ) -> CorridorStepResult:
        ctx = self.context
        if not ctx.active_layers:
            return CorridorStepResult(transitions=[], actions=[])

        transitions: list[TransitionRecord] = []
        if ctx.state != CorridorState.IDLE:
            transitions.append(self._transition(symbol, timestamp, CorridorState.IDLE, detail, regime, price))

        actions = self._close_all_layers(symbol, timestamp, price, action_type, detail, extra_metadata=extra_metadata)
        ctx.state = CorridorState.IDLE
        ctx.drift_count = 0
        ctx.current_center = None
        return CorridorStepResult(transitions=transitions, actions=actions)

    def _should_add_supplemental_layer(self, price: float, center: CenterEstimate) -> bool:
        if self.config.max_active_butterfly_layers <= 1:
            return False
        if not self.context.active_layers:
            return False
        edge_distance = abs(price - center.center_price)
        return center.actual_tolerance * 0.6 < edge_distance <= self.config.coverage_band_width / 2.0

    def _primary_entry_allowed(
        self,
        local_ts: pd.Timestamp,
        regime: Optional[RegimeSnapshot],
        center: Optional[CenterEstimate],
    ) -> bool:
        if regime is None or center is None:
            return False
        if regime.regime != Regime.RANGE:
            return False
        if self._is_event_day(local_ts):
            return False
        local_time = local_ts.timetz().replace(tzinfo=None)
        if local_time > self.primary_entry_end_time:
            return False
        if center.confidence < self.config.primary_entry_min_center_confidence:
            return False
        if abs(regime.momentum_pct) > self.config.primary_entry_max_momentum_pct:
            return False
        if regime.volume_ratio > self.config.primary_entry_max_volume_ratio:
            return False
        if regime.breakout_up or regime.breakout_down:
            return False
        return True

    def _is_event_day(self, local_ts: pd.Timestamp) -> bool:
        if not self.config.skip_event_days or not self.config.event_dates:
            return False
        current_date = local_ts.date()
        for value in self.config.event_dates:
            parsed = pd.Timestamp(value)
            if parsed.tzinfo is None:
                event_date = parsed.tz_localize("America/New_York").date()
            else:
                event_date = parsed.tz_convert("America/New_York").date()
            if event_date == current_date:
                return True
        return False

    @staticmethod
    def _layer_metadata(layer: ActiveButterfly) -> dict[str, float | str]:
        return {
            "kind": layer.kind.value,
            "center_price": round(layer.center_price, 4),
            "lower_strike": round(layer.lower_strike, 4),
            "body_strike": round(layer.body_strike, 4),
            "upper_strike": round(layer.upper_strike, 4),
            "width": round(layer.width, 4),
            "lower_width": round(layer.lower_width, 4),
            "upper_width": round(layer.upper_width, 4),
            "wing_mode": str(layer.metadata.get("wing_mode", "symmetric")),
            "dte": layer.dte,
        }
