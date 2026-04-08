#! python3.12
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legacy.strategy import (
    AlertSink,
    EmittedSignal,
    SignalType,
    StrategyConfig,
    StrategyPipeline,
    get_missing_strategy_dependencies,
    load_local_env,
    parse_bool,
    pd,
    require_strategy_dependencies,
)


@dataclass
class BacktestConfig(StrategyConfig):
    bars_csv_path: str = ""
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"
    quiet: bool = False


@dataclass
class ClosedTrade:
    symbol: str
    side: str
    entry_time: object
    exit_time: object
    entry_price: float
    exit_price: float

    @property
    def pnl_points(self) -> float:
        if self.side == "LONG":
            return self.exit_price - self.entry_price
        return self.entry_price - self.exit_price

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return self.pnl_points / self.entry_price


@dataclass
class BacktestSummary:
    total_bars: int
    total_signals: int
    directional_entries: int
    range_signals: int
    closed_trades: int
    wins: int
    losses: int
    flats: int
    open_positions: int
    net_points: float
    avg_points: float
    avg_pct: float
    unique_symbols: int

    @property
    def win_rate(self) -> float:
        if self.closed_trades == 0:
            return 0.0
        return self.wins / self.closed_trades


class CollectingSink:
    def __init__(
        self,
        cfg: StrategyConfig,
        csv_path: str,
        discord_webhook_url: str = "",
        print_json: bool = False,
        console_output: bool = True,
    ) -> None:
        self.signals: List[EmittedSignal] = []
        self.base_sink = AlertSink(
            discord_webhook_url=discord_webhook_url,
            print_json=print_json,
            csv_path=csv_path,
            console_output=console_output,
            strategy_cfg=cfg,
        )

    def send(self, signal: EmittedSignal) -> None:
        self.signals.append(signal)
        self.base_sink.send(signal)


def load_bars_frame(cfg: BacktestConfig):
    if pd is None:
        raise RuntimeError("pandas is not installed.")

    bars_path = Path(cfg.bars_csv_path)
    if not bars_path.is_file():
        raise RuntimeError(f"Bars CSV not found: {bars_path}")

    frame = pd.read_csv(bars_path)
    required = [cfg.timestamp_col, cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col]
    missing_cols = [name for name in required if name not in frame.columns]
    if missing_cols:
        raise RuntimeError(
            "Bars CSV is missing required columns: " + ", ".join(missing_cols)
        )

    if cfg.volume_col not in frame.columns:
        frame[cfg.volume_col] = 0.0

    if cfg.symbol_col not in frame.columns:
        if len(cfg.symbols) != 1:
            raise RuntimeError(
                "Bars CSV has no symbol column. Pass --symbol or configure exactly one symbol."
            )
        frame[cfg.symbol_col] = cfg.symbols[0]

    frame[cfg.timestamp_col] = pd.to_datetime(frame[cfg.timestamp_col], utc=True)
    frame = frame.sort_values([cfg.timestamp_col, cfg.symbol_col]).reset_index(drop=True)
    return frame


def run_startup_check(cfg: BacktestConfig) -> int:
    print(f"Interpreter: {sys.executable}")

    missing = get_missing_strategy_dependencies()
    if missing:
        print("Dependency check: FAIL")
        print(f"Missing packages: {', '.join(missing)}")
        return 1

    print("Dependency check: OK")

    try:
        frame = load_bars_frame(cfg)
    except Exception as exc:
        print("CSV check: FAIL")
        print(str(exc))
        return 1

    print("CSV check: OK")
    print(f"Rows: {len(frame)}")
    print(f"Symbols: {', '.join(sorted(frame[cfg.symbol_col].astype(str).unique()))}")
    return 0


def summarize_signals(
    signals: List[EmittedSignal],
    total_bars: int,
    unique_symbols: int,
) -> BacktestSummary:
    open_positions: Dict[str, EmittedSignal] = {}
    closed: List[ClosedTrade] = []
    directional_entries = 0
    range_signals = 0

    for signal in signals:
        if signal.signal == SignalType.RANGE_BUTTERFLY_ZONE:
            range_signals += 1
            continue

        if signal.signal == SignalType.TREND_LONG:
            directional_entries += 1
            open_positions[signal.symbol] = signal
            continue

        if signal.signal == SignalType.TREND_SHORT:
            directional_entries += 1
            open_positions[signal.symbol] = signal
            continue

        if signal.signal not in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT):
            continue

        entry = open_positions.get(signal.symbol)
        if entry is None:
            continue

        if entry.signal == SignalType.TREND_LONG and signal.signal == SignalType.EXIT_LONG:
            closed.append(
                ClosedTrade(
                    symbol=signal.symbol,
                    side="LONG",
                    entry_time=entry.timestamp,
                    exit_time=signal.timestamp,
                    entry_price=entry.price,
                    exit_price=signal.price,
                )
            )
            del open_positions[signal.symbol]
            continue

        if entry.signal == SignalType.TREND_SHORT and signal.signal == SignalType.EXIT_SHORT:
            closed.append(
                ClosedTrade(
                    symbol=signal.symbol,
                    side="SHORT",
                    entry_time=entry.timestamp,
                    exit_time=signal.timestamp,
                    entry_price=entry.price,
                    exit_price=signal.price,
                )
            )
            del open_positions[signal.symbol]

    wins = sum(1 for trade in closed if trade.pnl_points > 0)
    losses = sum(1 for trade in closed if trade.pnl_points < 0)
    flats = sum(1 for trade in closed if trade.pnl_points == 0)
    net_points = sum(trade.pnl_points for trade in closed)
    avg_points = net_points / len(closed) if closed else 0.0
    avg_pct = sum(trade.pnl_pct for trade in closed) / len(closed) if closed else 0.0

    return BacktestSummary(
        total_bars=total_bars,
        total_signals=len(signals),
        directional_entries=directional_entries,
        range_signals=range_signals,
        closed_trades=len(closed),
        wins=wins,
        losses=losses,
        flats=flats,
        open_positions=len(open_positions),
        net_points=net_points,
        avg_points=avg_points,
        avg_pct=avg_pct,
        unique_symbols=unique_symbols,
    )


def print_summary(summary: BacktestSummary) -> None:
    print(
        f"Summary | bars={summary.total_bars} | symbols={summary.unique_symbols} | "
        f"signals={summary.total_signals} | directional_entries={summary.directional_entries} | "
        f"range_signals={summary.range_signals}"
    )
    print(
        f"Trades | closed={summary.closed_trades} | wins={summary.wins} | losses={summary.losses} | "
        f"flats={summary.flats} | open={summary.open_positions} | "
        f"win_rate={summary.win_rate * 100:.2f}%"
    )
    print(
        f"PnL | net_points={summary.net_points:.2f} | avg_points={summary.avg_points:.4f} | "
        f"avg_pct={summary.avg_pct * 100:.3f}%"
    )


def replay_frame(frame, cfg: BacktestConfig, quiet: bool = False) -> BacktestSummary:
    sink = CollectingSink(
        cfg=cfg,
        csv_path=cfg.signal_csv_path,
        discord_webhook_url=cfg.discord_webhook_url,
        print_json=cfg.print_json,
        console_output=not quiet,
    )
    pipeline = StrategyPipeline(cfg, alerts=sink)

    signal_count = 0
    for row in frame.itertuples(index=False):
        emitted = pipeline.process_bar(
            symbol=str(getattr(row, cfg.symbol_col)).upper(),
            ts=getattr(row, cfg.timestamp_col),
            open_=float(getattr(row, cfg.open_col)),
            high=float(getattr(row, cfg.high_col)),
            low=float(getattr(row, cfg.low_col)),
            close=float(getattr(row, cfg.close_col)),
            volume=float(getattr(row, cfg.volume_col)),
            emit_signals=True,
        )
        if emitted:
            signal_count += 1

    summary = summarize_signals(
        sink.signals,
        total_bars=len(frame),
        unique_symbols=int(frame[cfg.symbol_col].nunique()),
    )
    if signal_count != summary.total_signals:
        raise RuntimeError("Signal count mismatch while building backtest summary.")
    return summary


def run_backtest(cfg: BacktestConfig) -> int:
    frame = load_bars_frame(cfg)
    summary = replay_frame(frame, cfg, quiet=cfg.quiet)
    print_summary(summary)
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay historical bars through the DaySpy strategy.")
    parser.add_argument("bars_csv", help="Path to a CSV file containing 1-minute OHLCV bars.")
    parser.add_argument("--check", action="store_true", help="Validate dependencies and CSV layout only.")
    parser.add_argument("--symbol", help="Use this symbol when the CSV has no symbol column.")
    parser.add_argument("--signals-out", default="backtest_signals.csv", help="Where emitted signals should be written.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-signal terminal output and print only the summary.")
    parser.add_argument("--timestamp-col", default="timestamp", help="CSV column containing the bar timestamp.")
    parser.add_argument("--symbol-col", default="symbol", help="CSV column containing the ticker symbol.")
    parser.add_argument("--open-col", default="open", help="CSV column containing the open price.")
    parser.add_argument("--high-col", default="high", help="CSV column containing the high price.")
    parser.add_argument("--low-col", default="low", help="CSV column containing the low price.")
    parser.add_argument("--close-col", default="close", help="CSV column containing the close price.")
    parser.add_argument("--volume-col", default="volume", help="CSV column containing the volume.")
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> BacktestConfig:
    load_local_env()

    env_symbols = [x.strip().upper() for x in os.getenv("SYMBOLS", "").split(",") if x.strip()]
    symbols = [args.symbol.strip().upper()] if args.symbol else env_symbols or ["SPY"]

    return BacktestConfig(
        symbols=symbols,
        signal_csv_path=args.signals_out,
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
        print_json=parse_bool(os.getenv("PRINT_JSON"), False),
        bars_csv_path=args.bars_csv,
        timestamp_col=args.timestamp_col,
        symbol_col=args.symbol_col,
        open_col=args.open_col,
        high_col=args.high_col,
        low_col=args.low_col,
        close_col=args.close_col,
        volume_col=args.volume_col,
        quiet=args.quiet,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    load_local_env()
    require_strategy_dependencies()

    try:
        cfg = load_config(args)
        if args.check:
            return run_startup_check(cfg)
        return run_backtest(cfg)
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
