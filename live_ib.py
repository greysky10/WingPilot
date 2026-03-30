#! python3.12
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from ib_insync import IB, Stock
except ModuleNotFoundError as exc:
    IB = None  # type: ignore[assignment]
    Stock = None  # type: ignore[assignment]
    IB_IMPORT_ERROR = exc
else:
    IB_IMPORT_ERROR = None

from strategy import (
    AlertSink,
    StrategyConfig,
    StrategyPipeline,
    get_missing_strategy_dependencies,
    load_local_env,
    parse_bool,
    require_strategy_dependencies,
)


IB_DEPENDENCY_HINT = "py -3.12 -m pip install ib_insync pandas pytz"


@dataclass
class LiveIBConfig(StrategyConfig):
    ib_host: str = "127.0.0.1"
    ib_port: int = 4001
    ib_client_id: int = 2
    ib_timeout: float = 10.0
    ib_market_data_type: int = 3
    ib_exchange: str = "SMART"
    ib_currency: str = "USD"
    ib_duration: str = "3 D"
    ib_bar_size: str = "1 min"
    ib_what_to_show: str = "TRADES"
    ib_use_rth: bool = True


def get_missing_dependencies() -> List[str]:
    missing = get_missing_strategy_dependencies()
    if IB_IMPORT_ERROR is not None:
        missing.append("ib_insync")
    return missing


def require_runtime_dependencies() -> None:
    missing = get_missing_dependencies()
    if not missing:
        return
    raise SystemExit(
        "Missing required packages: "
        + ", ".join(missing)
        + ". Install them with `"
        + IB_DEPENDENCY_HINT
        + "`."
    )


def connect_ib(cfg: LiveIBConfig, readonly: bool = True):
    if IB is None:
        raise RuntimeError("ib_insync is not installed.")

    ib = IB()
    try:
        ib.connect(
            cfg.ib_host,
            cfg.ib_port,
            clientId=cfg.ib_client_id,
            timeout=cfg.ib_timeout,
            readonly=readonly,
        )
    except TimeoutError as exc:
        raise RuntimeError(
            f"IB connection timed out for {cfg.ib_host}:{cfg.ib_port} with clientId={cfg.ib_client_id}. "
            "Check that TWS/IB Gateway is running and that IB_CLIENT_ID is not already in use."
        ) from exc
    except Exception as exc:
        message = str(exc).strip()
        if "client id is already in use" in message.lower():
            raise RuntimeError(
                f"IB clientId {cfg.ib_client_id} is already in use. Stop the other IB API session or set IB_CLIENT_ID to a different value."
            ) from exc
        raise

    if not ib.isConnected():
        raise RuntimeError(
            f"Unable to connect to IB at {cfg.ib_host}:{cfg.ib_port} with clientId={cfg.ib_client_id}."
        )

    ib.reqMarketDataType(cfg.ib_market_data_type)
    return ib


def build_stock_contract(symbol: str, cfg: LiveIBConfig):
    if Stock is None:
        raise RuntimeError("ib_insync is not installed.")
    return Stock(symbol, cfg.ib_exchange, cfg.ib_currency)


class IBStrategyApp:
    def __init__(self, cfg: LiveIBConfig) -> None:
        self.cfg = cfg
        self.pipeline = StrategyPipeline(
            cfg,
            alerts=AlertSink(
                cfg.discord_webhook_url,
                cfg.print_json,
                cfg.signal_csv_path,
                strategy_cfg=cfg,
            ),
        )
        self.ib = connect_ib(cfg)
        self.contracts: Dict[str, object] = {}
        self.live_bars: Dict[str, object] = {}
        self.bar_handlers: Dict[str, object] = {}

    def _process_completed_bar(self, symbol: str, bar, emit_signals: bool) -> None:
        if emit_signals:
            self.pipeline.process_bar(
                symbol=symbol,
                ts=bar.date,
                open_=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
                emit_signals=True,
            )
            return

        self.pipeline.store_bar(
            symbol=symbol,
            ts=bar.date,
            open_=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
        )

    def _seed_symbol(self, symbol: str, bars) -> None:
        if len(bars) < 2:
            raise RuntimeError(
                f"IB returned insufficient 1-minute history for {symbol}. Increase IB_DURATION or verify market data access."
            )

        seeded = 0
        for bar in list(bars)[:-1]:
            before_len = len(self.pipeline.store.frames.get(symbol, []))
            self._process_completed_bar(symbol, bar, emit_signals=False)
            after_len = len(self.pipeline.store.frames.get(symbol, []))
            if after_len > before_len:
                seeded += 1
        print(f"Seeded {symbol} with {seeded} completed 1-minute bars")

    def _make_bar_handler(self, symbol: str):
        def on_bar_update(bars, has_new_bar: bool) -> None:
            if not has_new_bar or len(bars) < 2:
                return
            self._process_completed_bar(symbol, bars[-2], emit_signals=True)

        return on_bar_update

    def _subscribe_symbol(self, symbol: str) -> None:
        contract = build_stock_contract(symbol, self.cfg)
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(f"Unable to qualify IB contract for {symbol}.")

        live_bars = self.ib.reqHistoricalData(
            qualified[0],
            endDateTime="",
            durationStr=self.cfg.ib_duration,
            barSizeSetting=self.cfg.ib_bar_size,
            whatToShow=self.cfg.ib_what_to_show,
            useRTH=self.cfg.ib_use_rth,
            formatDate=2,
            keepUpToDate=True,
        )

        self.contracts[symbol] = qualified[0]
        self.live_bars[symbol] = live_bars
        self._seed_symbol(symbol, live_bars)

        handler = self._make_bar_handler(symbol)
        live_bars.updateEvent += handler
        self.bar_handlers[symbol] = handler

        print(
            f"Subscribed {symbol} via IB | exchange={self.cfg.ib_exchange} | "
            f"market_data_type={self.cfg.ib_market_data_type}"
        )

    def stop(self) -> None:
        for live_bars in self.live_bars.values():
            try:
                self.ib.cancelHistoricalData(live_bars)
            except Exception:
                pass
        if self.ib.isConnected():
            self.ib.disconnect()

    def run(self) -> None:
        try:
            for symbol in self.cfg.symbols:
                self._subscribe_symbol(symbol)

            print(
                f"Starting live signal engine for: {', '.join(self.cfg.symbols)} | "
                f"provider=IB | market_data_type={self.cfg.ib_market_data_type}"
            )
            self.ib.run()
        except KeyboardInterrupt:
            print("keyboard interrupt, bye")
        finally:
            self.stop()


def load_config() -> LiveIBConfig:
    load_local_env()
    symbols = [x.strip().upper() for x in os.getenv("SYMBOLS", "SPY").split(",") if x.strip()]
    return LiveIBConfig(
        symbols=symbols,
        ib_host=os.getenv("IB_HOST", "127.0.0.1"),
        ib_port=int(os.getenv("IB_PORT", "4001")),
        ib_client_id=int(os.getenv("IB_CLIENT_ID", "2")),
        ib_timeout=float(os.getenv("IB_TIMEOUT", "10")),
        ib_market_data_type=int(os.getenv("IB_MARKET_DATA_TYPE", "3")),
        ib_exchange=os.getenv("IB_EXCHANGE", "SMART"),
        ib_currency=os.getenv("IB_CURRENCY", "USD"),
        ib_duration=os.getenv("IB_DURATION", "3 D"),
        ib_bar_size=os.getenv("IB_BAR_SIZE", "1 min"),
        ib_what_to_show=os.getenv("IB_WHAT_TO_SHOW", "TRADES"),
        ib_use_rth=parse_bool(os.getenv("IB_USE_RTH"), True),
        signal_csv_path=os.getenv("SIGNAL_CSV_PATH", "signals.csv"),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
        print_json=parse_bool(os.getenv("PRINT_JSON"), False),
    )


def run_startup_check() -> int:
    load_local_env()

    print(f"Interpreter: {sys.executable}")

    missing = get_missing_dependencies()
    if missing:
        print("Dependency check: FAIL")
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install them with: {IB_DEPENDENCY_HINT}")
        return 1

    print("Dependency check: OK")

    cfg = load_config()
    try:
        ib = connect_ib(cfg, readonly=True)
    except Exception as exc:
        print("IB connection: FAIL")
        print(str(exc))
        return 1

    try:
        for symbol in cfg.symbols:
            contract = build_stock_contract(symbol, cfg)
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                raise RuntimeError(f"Unable to qualify IB contract for {symbol}.")
        print("IB connection: OK")
        print("Contract check: OK")
    except Exception as exc:
        print("Contract check: FAIL")
        print(str(exc))
        return 1
    finally:
        if ib.isConnected():
            ib.disconnect()

    print("Startup check passed.")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DaySpy live IB signal engine.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate dependencies, connect to IB, and qualify the configured symbols.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.check:
        return run_startup_check()

    load_local_env()
    require_strategy_dependencies()
    require_runtime_dependencies()

    try:
        cfg = load_config()
        app = IBStrategyApp(cfg)
        app.run()
        return 0
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
