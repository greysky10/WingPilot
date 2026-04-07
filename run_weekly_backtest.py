#! python3.12
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from corridor.data.historical_loader import HistoricalLoadConfig, load_intraday_bars
from corridor.data.ib_contracts import default_center_rounding_for_symbol
from corridor.data.ib_loader import IBHistoricalRequest, fetch_intraday_bars
from corridor.models import CenterMethod
from strategy import load_local_env
from weekly_corridor.backtest import WeeklyBacktestEngine
from weekly_corridor.config import WeeklyCorridorConfig
from weekly_corridor.report import save_backtest_outputs


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the separate weekly SPX butterfly corridor backtest on 30-minute or 60-minute decision bars."
    )
    parser.add_argument("--symbol", default="SPX", help="Ticker symbol.")
    parser.add_argument("--start", help="UTC start date, for example 2025-01-01.")
    parser.add_argument("--end", help="UTC end date, for example 2025-12-31.")
    parser.add_argument("--bars-csv", help="Optional intraday bars CSV. When omitted, IBKR history is used.")
    parser.add_argument("--decision-timeframe", default="30 mins", help="Decision timeframe, for example '30 mins' or '60 mins'.")
    parser.add_argument(
        "--center-method",
        default=CenterMethod.VWAP.value,
        choices=[item.value for item in CenterMethod],
        help="Weekly center estimation method.",
    )
    parser.add_argument("--center-lookback-bars", type=int, default=65, help="Lookback bars for weekly center estimation.")
    parser.add_argument("--regime-lookback-bars", type=int, default=65, help="Lookback bars for weekly range/trend classification.")
    parser.add_argument("--butterfly-width", type=float, default=50.0, help="Wing width for each butterfly.")
    parser.add_argument("--center-spacing", type=float, default=50.0, help="Spacing between the three initial butterfly centers.")
    parser.add_argument("--weekly-center-tolerance", type=float, default=50.0, help="Material drift distance from the weekly center before adjustment is allowed.")
    parser.add_argument("--target-total-coverage", type=float, default=200.0, help="Target initial total corridor coverage in points.")
    parser.add_argument("--max-active-butterflies", type=int, default=4, help="Maximum concurrent weekly butterflies.")
    parser.add_argument("--max-adjustments-per-week", type=int, default=1, help="Maximum weekly adjustments.")
    parser.add_argument("--weekly-range-width-threshold-pct", type=float, default=0.055, help="Maximum multi-day width pct still considered range-like.")
    parser.add_argument("--weekly-trend-slope-threshold-pct", type=float, default=0.012, help="Slope threshold that classifies a week as trending.")
    parser.add_argument("--weekly-momentum-threshold-pct", type=float, default=0.008, help="Momentum expansion threshold that classifies a week as trending.")
    parser.add_argument("--breakout-buffer-pct", type=float, default=0.004, help="Breakout buffer beyond prior multi-day highs/lows.")
    parser.add_argument("--dte-min", type=int, default=10, help="Minimum starting DTE assumption.")
    parser.add_argument("--dte-max", type=int, default=14, help="Maximum starting DTE assumption.")
    parser.add_argument("--default-dte", type=int, default=12, help="Default DTE assigned to the simplified weekly butterflies.")
    parser.add_argument("--min-remaining-dte", type=int, default=5, help="Force exit once remaining DTE falls to or below this buffer.")
    parser.add_argument("--min-hold-trading-days", type=int, default=4, help="Minimum intended holding window in trading days.")
    parser.add_argument("--max-hold-trading-days", type=int, default=7, help="Maximum intended holding window in trading days.")
    parser.add_argument("--event-date", action="append", default=[], help="Optional YYYY-MM-DD dates used to skip major event weeks.")
    parser.add_argument("--client-id", type=int, default=180, help="IB client id when fetching from IBKR.")
    parser.add_argument("--starting-capital", type=float, default=100000.0, help="Capital base used for normalized return reporting.")
    parser.add_argument("--contracts-per-layer", type=int, default=1, help="Contract count assumption per weekly butterfly.")
    parser.add_argument("--option-multiplier", type=int, default=100, help="Option multiplier for dollar conversion.")
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> WeeklyCorridorConfig:
    cfg = WeeklyCorridorConfig(
        symbol=args.symbol.upper(),
        decision_timeframe=args.decision_timeframe,
        center_method=CenterMethod(args.center_method),
        center_lookback_bars=max(10, int(args.center_lookback_bars)),
        regime_lookback_bars=max(10, int(args.regime_lookback_bars)),
        center_rounding=default_center_rounding_for_symbol(args.symbol.upper()),
        butterfly_width=max(5.0, float(args.butterfly_width)),
        center_spacing=max(5.0, float(args.center_spacing)),
        weekly_center_tolerance=max(5.0, float(args.weekly_center_tolerance)),
        target_total_coverage=max(50.0, float(args.target_total_coverage)),
        max_active_butterflies=max(3, int(args.max_active_butterflies)),
        max_adjustments_per_week=max(0, int(args.max_adjustments_per_week)),
        weekly_range_width_threshold_pct=max(0.001, float(args.weekly_range_width_threshold_pct)),
        weekly_trend_slope_threshold_pct=max(0.0001, float(args.weekly_trend_slope_threshold_pct)),
        weekly_momentum_threshold_pct=max(0.0001, float(args.weekly_momentum_threshold_pct)),
        breakout_buffer_pct=max(0.0, float(args.breakout_buffer_pct)),
        dte_min=max(1, int(args.dte_min)),
        dte_max=max(1, int(args.dte_max)),
        default_dte=max(1, int(args.default_dte)),
        min_remaining_dte=max(0, int(args.min_remaining_dte)),
        min_hold_trading_days=max(1, int(args.min_hold_trading_days)),
        max_hold_trading_days=max(1, int(args.max_hold_trading_days)),
        event_dates=tuple(args.event_date),
        ib_host=os.getenv("IB_HOST", "127.0.0.1"),
        ib_port=int(os.getenv("IB_PORT", "4001")),
        ib_client_id=args.client_id,
        starting_capital=max(0.0, float(args.starting_capital)),
        contracts_per_layer=max(1, int(args.contracts_per_layer)),
        option_multiplier=max(1, int(args.option_multiplier)),
    )
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    else:
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        cfg.output_dir = Path("weekly_corridor_outputs") / f"{cfg.symbol}_{stamp}"
    return cfg


def load_frame(cfg: WeeklyCorridorConfig, args: argparse.Namespace) -> pd.DataFrame:
    start = pd.Timestamp(args.start) if args.start else None
    end = pd.Timestamp(args.end) if args.end else None
    if args.bars_csv:
        return load_intraday_bars(
            HistoricalLoadConfig(
                csv_path=Path(args.bars_csv),
                symbol=cfg.symbol,
                start=start,
                end=end,
            )
        )

    request = IBHistoricalRequest(
        symbol=cfg.symbol,
        start=start,
        end=end,
        bar_size="5 mins",
        host=cfg.ib_host,
        port=cfg.ib_port,
        client_id=cfg.ib_client_id,
        exchange=cfg.ib_exchange,
        currency=cfg.ib_currency,
        what_to_show=cfg.ib_what_to_show,
        use_rth=cfg.ib_use_rth,
        chunk_duration=cfg.ib_chunk_duration,
    )
    return fetch_intraday_bars(request)


def main(argv: Optional[list[str]] = None) -> int:
    load_local_env()
    args = parse_args(argv)
    cfg = build_config(args)
    frame = load_frame(cfg, args)
    result = WeeklyBacktestEngine(cfg).run(frame)
    artifacts = save_backtest_outputs(cfg.output_dir, result)

    print(f"Weekly backtest complete for {cfg.symbol} | decision_timeframe={cfg.decision_timeframe} | rows={len(frame)}")
    print(f"Transitions: {artifacts.transitions_path}")
    print(f"Actions: {artifacts.actions_path}")
    print(f"Closed layers: {artifacts.closed_layers_path}")
    print(f"Summary: {artifacts.summary_path}")
    print(f"Equity: {artifacts.equity_curve_path}")
    print(
        "Summary | "
        f"net_modeled_pnl={result.summary['net_modeled_pnl']:.4f} | "
        f"net_dollar_pnl={result.summary['net_dollar_pnl']:.2f} | "
        f"return_on_capital={_format_ratio(result.summary['return_on_capital'])} | "
        f"weekly_occupancy={result.summary['weekly_occupancy_rate']:.2%}"
    )
    print(
        "Risk | "
        f"max_gross_deployment_dollars={result.summary['max_gross_deployment_dollars']:.2f} | "
        f"worst_day_pnl_dollars={result.summary['worst_day_pnl_dollars']:.2f} | "
        f"worst_week_pnl_dollars={result.summary['worst_week_pnl_dollars']:.2f}"
    )
    print(
        "Quality | "
        f"weeks_traded={result.summary['weeks_traded']} | "
        f"weeks_aborted={result.summary['weeks_aborted']} | "
        f"avg_adjustments_per_week={result.summary['avg_adjustments_per_week']:.2f} | "
        f"profit_factor_by_closed_layer={_format_number(result.summary['profit_factor_by_closed_layer'])} | "
        f"profit_factor_by_week={_format_number(result.summary['profit_factor_by_week'])} | "
        f"max_active_butterflies={result.summary['max_active_butterflies']}"
    )
    return 0


def _format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _format_number(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
