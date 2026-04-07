from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from corridor.models import CenterMethod


@dataclass(slots=True)
class WeeklyCorridorConfig:
    """Configuration for the separate weekly SPX butterfly corridor path."""

    symbol: str = "SPX"
    decision_timeframe: str = "30 mins"
    center_lookback_bars: int = 65
    regime_lookback_bars: int = 65
    center_method: CenterMethod = CenterMethod.VWAP
    center_rounding: float = 5.0
    weekly_range_width_threshold_pct: float = 0.055
    weekly_trend_slope_threshold_pct: float = 0.012
    weekly_momentum_threshold_pct: float = 0.008
    breakout_buffer_pct: float = 0.004
    butterfly_width: float = 50.0
    center_spacing: float = 50.0
    weekly_center_tolerance: float = 50.0
    target_total_coverage: float = 200.0
    max_active_butterflies: int = 4
    max_adjustments_per_week: int = 1
    entry_start_weekday: int = 0
    entry_end_weekday: int = 1
    valid_trading_start: str = "10:00"
    valid_trading_end: str = "15:30"
    dte_min: int = 10
    dte_max: int = 14
    default_dte: int = 12
    min_remaining_dte: int = 5
    min_hold_trading_days: int = 4
    max_hold_trading_days: int = 7
    forced_exit_weekday: int = 4
    forced_exit_time: str = "15:00"
    skip_event_weeks: bool = True
    event_dates: tuple[str, ...] = field(default_factory=tuple)
    starting_capital: float = 100000.0
    contracts_per_layer: int = 1
    option_multiplier: int = 100
    slippage: float = 0.03
    commission_per_contract: float = 0.65
    payoff_mode: str = "simplified"
    simplified_entry_debit_pct_of_width: float = 0.22
    simplified_peak_value_pct_of_width: float = 0.85
    simplified_residual_floor_pct: float = 0.05
    ib_host: str = "127.0.0.1"
    ib_port: int = 4001
    ib_client_id: int = 180
    ib_exchange: str = "SMART"
    ib_currency: str = "USD"
    ib_use_rth: bool = True
    ib_what_to_show: str = "TRADES"
    ib_chunk_duration: str = "30 D"
    output_dir: Path = field(default_factory=lambda: Path("weekly_corridor_outputs"))

    def weekly_span(self) -> float:
        return max(self.target_total_coverage / 2.0, self.center_spacing + self.butterfly_width)
