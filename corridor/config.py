from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import CenterMethod


@dataclass(slots=True)
class CorridorConfig:
    """Configuration for the corridor backtest and live-prep workflows."""

    symbol: str = "SPY"
    timeframe: str = "5 mins"
    center_lookback: int = 36
    center_method: CenterMethod = CenterMethod.VWAP
    center_rounding: float = 1.0
    regime_lookback: int = 48
    range_width_threshold_pct: float = 0.012
    trend_slope_threshold_pct: float = 0.0015
    breakout_buffer_pct: float = 0.0025
    butterfly_width: float = 5.0
    coverage_band_width: float = 10.0
    center_tolerance: float = 2.5
    recenter_threshold: float = 3.5
    drift_persistence_bars: int = 2
    rebuild_cooldown_minutes: int = 15
    max_active_butterfly_layers: int = 2
    valid_trading_start: str = "09:45"
    valid_trading_end: str = "15:30"
    abort_volume_threshold: float = 2.2
    abort_momentum_threshold: float = 0.01
    dte_min: int = 2
    dte_max: int = 7
    default_dte: int = 5
    max_acceptable_option_spread: float = 0.15
    slippage: float = 0.03
    commission_per_contract: float = 0.65
    starting_capital: float = 100000.0
    contracts_per_layer: int = 1
    option_multiplier: int = 100
    payoff_mode: str = "simplified"
    simplified_entry_debit_pct_of_width: float = 0.22
    simplified_peak_value_pct_of_width: float = 0.85
    simplified_residual_floor_pct: float = 0.05
    occupancy_credit_per_bar: float = 0.0
    drift_penalty_per_bar: float = 0.0
    ib_host: str = "127.0.0.1"
    ib_port: int = 4001
    ib_client_id: int = 41
    ib_exchange: str = "SMART"
    ib_currency: str = "USD"
    ib_use_rth: bool = True
    ib_what_to_show: str = "TRADES"
    ib_chunk_duration: str = "30 D"
    output_dir: Path = field(default_factory=lambda: Path("corridor_outputs"))

    def time_window_label(self) -> str:
        return f"{self.valid_trading_start}-{self.valid_trading_end}"
