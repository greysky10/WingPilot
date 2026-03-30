#! python3.12
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from backtest_ib import fetch_ib_frame, get_missing_dependencies
from backtest_options import (
    OptionBacktestConfig,
    print_summary,
    simulate_directional_option_trades,
    summarize_option_trades,
    write_trades_csv,
    replay_signals,
)
from strategy import load_local_env, parse_bool, require_strategy_dependencies


@dataclass
class IBOptionBacktestConfig(OptionBacktestConfig):
    ib_host: str = "127.0.0.1"
    ib_port: int = 4001
    ib_client_id: int = 17
    ib_timeout: float = 10.0
    ib_market_data_type: int = 3
    ib_exchange: str = "SMART"
    ib_currency: str = "USD"
    ib_duration: str = "1 Y"
    ib_bar_size: str = "1 min"
    ib_what_to_show: str = "TRADES"
    ib_use_rth: bool = True
    ib_chunk_duration: str = ""
    ib_request_pause_ms: int = 250


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch IB historical bars in chunks and backtest DaySpy option strategies."
    )
    parser.add_argument("--symbols", default="SPY", help="Comma-separated symbol list. Defaults to SPY.")
    parser.add_argument("--duration", default="1 Y", help="IB duration string, for example '1 Y'.")
    parser.add_argument("--bar-size", default="1 min", help="IB bar size, for example '1 min'.")
    parser.add_argument("--what-to-show", default=None, help="IB whatToShow value, for example TRADES.")
    parser.add_argument("--use-rth", default=None, help="true/false. Defaults to IB_USE_RTH from .env.")
    parser.add_argument("--client-id", type=int, default=17, help="IB client id override.")
    parser.add_argument("--chunk-duration", default=None, help="Optional per-request chunk duration, for example '10 D'.")
    parser.add_argument("--request-pause-ms", type=int, default=250, help="Pause between chunked IB history requests.")
    parser.add_argument("--signals-out", default="", help="Optional CSV path for emitted signals.")
    parser.add_argument("--trades-out", default="option_backtest_ib_trades.csv", help="Where closed option trades should be written.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-signal terminal output and print only the summary.")
    parser.add_argument("--vol-lookback", type=int, default=120, help="Lookback bars for realized-vol estimate.")
    parser.add_argument("--risk-free-rate", type=float, default=0.0, help="Annual risk-free rate for Black-Scholes.")
    parser.add_argument("--contracts", type=int, default=10, help="Number of spreads to trade per signal. Defaults to 10.")
    parser.add_argument(
        "--invert-directional-spreads",
        action="store_true",
        help="Enter the opposite directional debit spread on each TREND_LONG/TREND_SHORT signal, but keep the original entry/exit timing.",
    )
    parser.add_argument("--check", action="store_true", help="Validate dependencies and IB connectivity only.")
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> IBOptionBacktestConfig:
    load_local_env()
    symbols = [x.strip().upper() for x in (args.symbols or "SPY").split(",") if x.strip()]
    if not symbols:
        raise RuntimeError("No symbols configured. Use --symbols.")

    return IBOptionBacktestConfig(
        symbols=symbols,
        signal_csv_path=args.signals_out,
        trades_csv_path=args.trades_out,
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
        print_json=parse_bool(os.getenv("PRINT_JSON"), False),
        quiet=args.quiet,
        vol_lookback_bars=max(20, args.vol_lookback),
        risk_free_rate=args.risk_free_rate,
        invert_directional_spreads=args.invert_directional_spreads,
        contracts_per_trade=max(1, args.contracts),
        ib_host=os.getenv("IB_HOST", "127.0.0.1"),
        ib_port=int(os.getenv("IB_PORT", "4001")),
        ib_client_id=args.client_id,
        ib_timeout=float(os.getenv("IB_TIMEOUT", "10")),
        ib_market_data_type=int(os.getenv("IB_MARKET_DATA_TYPE", "3")),
        ib_exchange=os.getenv("IB_EXCHANGE", "SMART"),
        ib_currency=os.getenv("IB_CURRENCY", "USD"),
        ib_duration=args.duration,
        ib_bar_size=args.bar_size,
        ib_what_to_show=args.what_to_show or os.getenv("IB_WHAT_TO_SHOW", "TRADES"),
        ib_use_rth=parse_bool(args.use_rth if args.use_rth is not None else os.getenv("IB_USE_RTH"), True),
        ib_chunk_duration=args.chunk_duration or os.getenv("IB_CHUNK_DURATION", ""),
        ib_request_pause_ms=max(0, args.request_pause_ms),
    )


def run_startup_check(cfg: IBOptionBacktestConfig) -> int:
    print(f"Interpreter: {sys.executable}")
    missing = get_missing_dependencies()
    if missing:
        print("Dependency check: FAIL")
        print(f"Missing packages: {', '.join(missing)}")
        return 1
    print("Dependency check: OK")

    try:
        frame = fetch_ib_frame(cfg)
    except Exception as exc:
        print("IB fetch: FAIL")
        print(str(exc))
        return 1

    print("IB fetch: OK")
    print(f"Rows: {len(frame)}")
    print(f"Symbols: {', '.join(sorted(frame['symbol'].astype(str).unique()))}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    load_local_env()
    require_strategy_dependencies()

    try:
        cfg = load_config(args)
        if args.check:
            return run_startup_check(cfg)

        frame = fetch_ib_frame(cfg)
        signals = replay_signals(frame, cfg)
        trades, open_positions = simulate_directional_option_trades(frame, signals, cfg)
        write_trades_csv(cfg.trades_csv_path, trades)
        summary = summarize_option_trades(signals, trades, open_positions)
        print_summary(summary)
        print("Assumption | theoretical Black-Scholes pricing from underlying bars; historical option IV/fills are not included.")
        return 0
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
