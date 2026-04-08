#! python3.12
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legacy.live_ib import build_stock_contract, connect_ib, get_missing_dependencies
from legacy.strategy import load_local_env, parse_bool, require_strategy_dependencies


@dataclass
class ExportIBConfig:
    symbols: List[str]
    output_path: str
    ib_host: str = "127.0.0.1"
    ib_port: int = 4001
    ib_client_id: int = 3
    ib_timeout: float = 10.0
    ib_market_data_type: int = 3
    ib_exchange: str = "SMART"
    ib_currency: str = "USD"
    ib_duration: str = "5 D"
    ib_bar_size: str = "1 min"
    ib_what_to_show: str = "TRADES"
    ib_use_rth: bool = True


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export historical IB bars to a CSV for backtest.py.")
    parser.add_argument("output_csv", help="Path to write the exported CSV bars file.")
    parser.add_argument("--symbols", help="Comma-separated symbol list. Defaults to SYMBOLS from .env.")
    parser.add_argument("--duration", default=None, help="IB duration string, for example '5 D' or '30 D'.")
    parser.add_argument("--bar-size", default=None, help="IB bar size, for example '1 min'.")
    parser.add_argument("--what-to-show", default=None, help="IB whatToShow value, for example TRADES.")
    parser.add_argument("--use-rth", default=None, help="true/false. Defaults to IB_USE_RTH from .env.")
    parser.add_argument("--client-id", type=int, default=None, help="IB client id override.")
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> ExportIBConfig:
    load_local_env()
    symbols_text = args.symbols or os.getenv("SYMBOLS", "SPY")
    symbols = [x.strip().upper() for x in symbols_text.split(",") if x.strip()]
    if not symbols:
        raise RuntimeError("No symbols provided. Use --symbols or set SYMBOLS in .env.")

    return ExportIBConfig(
        symbols=symbols,
        output_path=args.output_csv,
        ib_host=os.getenv("IB_HOST", "127.0.0.1"),
        ib_port=int(os.getenv("IB_PORT", "4001")),
        ib_client_id=args.client_id if args.client_id is not None else int(os.getenv("IB_CLIENT_ID", "2")),
        ib_timeout=float(os.getenv("IB_TIMEOUT", "10")),
        ib_market_data_type=int(os.getenv("IB_MARKET_DATA_TYPE", "3")),
        ib_exchange=os.getenv("IB_EXCHANGE", "SMART"),
        ib_currency=os.getenv("IB_CURRENCY", "USD"),
        ib_duration=args.duration or os.getenv("IB_DURATION", "5 D"),
        ib_bar_size=args.bar_size or os.getenv("IB_BAR_SIZE", "1 min"),
        ib_what_to_show=args.what_to_show or os.getenv("IB_WHAT_TO_SHOW", "TRADES"),
        ib_use_rth=parse_bool(args.use_rth if args.use_rth is not None else os.getenv("IB_USE_RTH"), True),
    )


def export_bars(cfg: ExportIBConfig) -> int:
    missing = get_missing_dependencies()
    if missing:
        raise RuntimeError("Missing packages: " + ", ".join(missing))

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    ib = connect_ib(cfg, readonly=True)
    try:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "symbol", "open", "high", "low", "close", "volume"])

            for symbol in cfg.symbols:
                contract = build_stock_contract(symbol, cfg)
                qualified = ib.qualifyContracts(contract)
                if not qualified:
                    raise RuntimeError(f"Unable to qualify IB contract for {symbol}.")

                bars = ib.reqHistoricalData(
                    qualified[0],
                    endDateTime="",
                    durationStr=cfg.ib_duration,
                    barSizeSetting=cfg.ib_bar_size,
                    whatToShow=cfg.ib_what_to_show,
                    useRTH=cfg.ib_use_rth,
                    formatDate=2,
                    keepUpToDate=False,
                )

                symbol_rows = 0
                for bar in bars:
                    writer.writerow(
                        [
                            bar.date.isoformat(),
                            symbol,
                            float(bar.open),
                            float(bar.high),
                            float(bar.low),
                            float(bar.close),
                            float(bar.volume),
                        ]
                    )
                    symbol_rows += 1
                    row_count += 1

                print(f"Exported {symbol_rows} bars for {symbol}")
    finally:
        if ib.isConnected():
            ib.disconnect()

    print(f"Wrote {row_count} rows to {output_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    require_strategy_dependencies()
    try:
        cfg = load_config(args)
        return export_bars(cfg)
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
