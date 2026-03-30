#! python3.12
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from corridor.config import CorridorConfig
from corridor.data.historical_loader import HistoricalLoadConfig, load_intraday_bars
from corridor.data.ib_loader import IBHistoricalRequest, fetch_intraday_bars
from corridor.models import CenterMethod
from corridor.backtest.engine import CorridorBacktestEngine
from corridor.report.plots import save_equity_plot
from corridor.report.summary import save_backtest_outputs


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a dynamic corridor backtest on intraday bars. Modeled PnL units are reported separately from capital-normalized returns."
    )
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol.")
    parser.add_argument("--start", help="UTC start date, for example 2025-01-01.")
    parser.add_argument("--end", help="UTC end date, for example 2025-12-31.")
    parser.add_argument("--bars-csv", help="Optional intraday bars CSV. When omitted, IBKR history is used.")
    parser.add_argument("--timeframe", default="5 mins", help="Intraday timeframe label.")
    parser.add_argument("--center-lookback", type=int, default=36, help="Lookback bars for center estimation.")
    parser.add_argument(
        "--center-method",
        default=CenterMethod.VWAP.value,
        choices=[item.value for item in CenterMethod],
        help="Center estimation method.",
    )
    parser.add_argument(
        "--payoff-mode",
        default="underlying_only",
        choices=["underlying_only", "simplified"],
        help="Use underlying-only decision logging or the simplified butterfly payoff model.",
    )
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    parser.add_argument("--client-id", type=int, default=41, help="IB client id when fetching from IBKR.")
    parser.add_argument("--starting-capital", type=float, default=100000.0, help="Capital base used for normalized return reporting.")
    parser.add_argument("--contracts-per-layer", type=int, default=1, help="Contract count assumption per butterfly layer.")
    parser.add_argument("--option-multiplier", type=int, default=100, help="Option contract multiplier for dollar conversion.")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> CorridorConfig:
    cfg = CorridorConfig(
        symbol=args.symbol.upper(),
        timeframe=args.timeframe,
        center_lookback=max(10, args.center_lookback),
        center_method=CenterMethod(args.center_method),
        payoff_mode="simplified" if args.payoff_mode == "simplified" else "underlying_only",
        ib_client_id=args.client_id,
        starting_capital=max(0.0, float(args.starting_capital)),
        contracts_per_layer=max(1, int(args.contracts_per_layer)),
        option_multiplier=max(1, int(args.option_multiplier)),
    )
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    else:
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        cfg.output_dir = Path("corridor_outputs") / f"{cfg.symbol}_{stamp}"
    return cfg


def load_frame(cfg: CorridorConfig, args: argparse.Namespace) -> pd.DataFrame:
    start = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end = pd.Timestamp(args.end, tz="UTC") if args.end else None
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
        bar_size=cfg.timeframe,
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
    args = parse_args(argv)
    cfg = build_config(args)
    frame = load_frame(cfg, args)
    result = CorridorBacktestEngine(cfg).run(frame)
    artifacts = save_backtest_outputs(cfg.output_dir, result)

    try:
        from corridor.backtest.trades import equity_to_frame

        save_equity_plot(equity_to_frame(result.equity_curve), cfg.output_dir / "equity_curve.png")
    except Exception:
        pass

    print(f"Backtest complete for {cfg.symbol} | payoff_mode={cfg.payoff_mode} | rows={len(frame)}")
    print(f"Transitions: {artifacts.transitions_path}")
    print(f"Actions: {artifacts.actions_path}")
    print(f"Summary: {artifacts.summary_path}")
    print(f"Equity: {artifacts.equity_curve_path}")
    print(
        "Modeled | "
        f"total_return(modeled_units)={result.summary['total_return']:.4f} | "
        f"model_points={result.summary['model_points']:.4f} | "
        f"gross_modeled_pnl={result.summary['gross_modeled_pnl']:.4f} | "
        f"net_modeled_pnl={result.summary['net_modeled_pnl']:.4f}"
    )
    print(
        "Capital-Normalized | "
        f"starting_capital={result.summary['starting_capital']:.2f} | "
        f"contracts_per_layer={result.summary['contracts_per_layer']} | "
        f"option_multiplier={result.summary['option_multiplier']} | "
        f"net_dollar_pnl={result.summary['net_dollar_pnl']:.2f} | "
        f"max_gross_deployment_dollars={result.summary['max_gross_deployment_dollars']:.2f} | "
        f"max_modeled_capital_at_risk={result.summary['max_modeled_capital_at_risk']:.4f} | "
        f"return_on_capital={_format_ratio(result.summary['return_on_capital'])} | "
        f"return_on_max_risk={_format_ratio(result.summary['return_on_max_risk'])}"
    )
    print(
        "Closed Layers | "
        f"closed_layers={result.summary['closed_layers']} | "
        f"wins={result.summary['winning_layers']} | "
        f"losses={result.summary['losing_layers']} | "
        f"gross_winners_dollars={result.summary['gross_winners_dollars']:.2f} | "
        f"gross_losers_dollars={result.summary['gross_losers_dollars']:.2f} | "
        f"win_rate={_format_ratio(result.summary['win_rate_by_closed_layer'])} | "
        f"profit_factor={_format_number(result.summary['profit_factor_by_closed_layer'])} | "
        f"avg_closed_layer_pnl={_format_number(result.summary['average_closed_layer_pnl'])}"
    )
    print(
        "Context | "
        f"max_drawdown={result.summary['max_drawdown']:.4f} | "
        f"best_day_pnl_dollars={result.summary['best_day_pnl_dollars']:.2f} | "
        f"worst_day_pnl_dollars={result.summary['worst_day_pnl_dollars']:.2f} | "
        f"profit_factor_by_day={_format_number(result.summary['profit_factor_by_day'])} | "
        f"occupancy={result.summary['corridor_occupancy_rate']:.2%} | "
        f"avg_rebuilds_per_day={result.summary['average_rebuilds_per_day']:.2f}"
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
