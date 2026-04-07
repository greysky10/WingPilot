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
    butterfly_width: float = 10.0
    wing_mode: str = "symmetric"
    broken_wing_extra_width: float = 0.0
    coverage_band_width: float = 20.0
    center_tolerance: float = 2.5
    center_tolerance_atr_multiplier: float = 1.0
    atr_lookback: int = 14
    recenter_threshold: float = 3.5
    drift_persistence_bars: int = 2
    rebuild_cooldown_minutes: int = 15
    max_active_butterfly_layers: int = 3
    valid_trading_start: str = "09:45"
    valid_trading_end: str = "15:30"
    primary_entry_end: str = "15:30"
    primary_entry_min_center_confidence: float = 0.0
    primary_entry_max_momentum_pct: float = 1.0
    primary_entry_max_volume_ratio: float = 999.0
    skip_event_days: bool = False
    event_dates: tuple[str, ...] = field(default_factory=tuple)
    abort_volume_threshold: float = 2.2
    abort_momentum_threshold: float = 0.01
    primary_stop_loss_pct: float = 0.0
    primary_take_profit_pct: float = 0.0
    dte_min: int = 4
    dte_max: int = 10
    default_dte: int = 7
    max_acceptable_option_spread: float = 0.25
    per_contract_slippage: float = 0.05
    # Deprecated alias kept for backward compatibility with older configs/tests.
    slippage: float = 0.03
    commission_per_contract: float = 0.65
    starting_capital: float = 100000.0
    contracts_per_layer: int = 1
    option_multiplier: int = 100
    payoff_mode: str = "simplified"
    simplified_entry_debit_pct_of_width: float = 0.22
    simplified_peak_value_pct_of_width: float = 0.85
    simplified_residual_floor_pct: float = 0.05
    stress_profile: str = "none"
    stress_entry_debit_multiplier: float = 1.0
    stress_peak_value_multiplier: float = 1.0
    stress_residual_floor_multiplier: float = 1.0
    stress_slippage_multiplier: float = 1.0
    stress_close_value_haircut_pct: float = 0.0
    candidate_body_search_steps: int = 2
    occupancy_credit_per_bar: float = 0.0
    drift_penalty_per_bar: float = 0.0
    paper_spread_gate_enabled: bool = False
    paper_spread_gate_mode: str = "none"
    paper_spread_gate_source: str = ""
    paper_spread_gate_spread_ratio: float = 0.0
    paper_spread_gate_total_spread: float = 0.0
    paper_spread_gate_sample_count: int = 0
    paper_spread_gate_rejection_count: int = 0
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
