#! python3.12
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legacy.backtest import BacktestConfig, CollectingSink, load_bars_frame
from legacy.strategy import (
    EmittedSignal,
    SignalType,
    StrategyPipeline,
    build_option_plan,
    get_missing_strategy_dependencies,
    load_local_env,
    parse_bool,
    pd,
    require_strategy_dependencies,
    require_toronto_tz,
)


LEG_RE = re.compile(
    r"^(BUY|SELL)\s+(\d+)\s+([A-Z]+)\s+(\d{4}-\d{2}-\d{2})\s+([0-9]+(?:\.[0-9]+)?)\s+(CALL|PUT)$"
)


@dataclass
class OptionBacktestConfig(BacktestConfig):
    trades_csv_path: str = "option_backtest_trades.csv"
    vol_lookback_bars: int = 120
    risk_free_rate: float = 0.0
    invert_directional_spreads: bool = False
    contracts_per_trade: int = 10


@dataclass
class ParsedLeg:
    side: str
    quantity: int
    symbol: str
    expiry: str
    strike: float
    right: str


@dataclass
class OpenOptionTrade:
    entry_signal: EmittedSignal
    structure: str
    legs: List[ParsedLeg]
    expiry_ts: object
    entry_value: float
    sigma: float


@dataclass
class ClosedOptionTrade:
    symbol: str
    structure: str
    entry_time: object
    exit_time: object
    expiry_date: str
    entry_underlying: float
    exit_underlying: float
    entry_value: float
    exit_value: float
    sigma: float
    exit_reason: str
    legs_text: str

    @property
    def pnl_points(self) -> float:
        return self.exit_value - self.entry_value

    @property
    def pnl_dollars(self) -> float:
        return self.pnl_points * 100.0

    @property
    def return_pct(self) -> float:
        if self.entry_value <= 0:
            return 0.0
        return self.pnl_points / self.entry_value


@dataclass
class OptionBacktestSummary:
    total_signals: int
    total_option_entries: int
    closed_trades: int
    wins: int
    losses: int
    flats: int
    open_positions: int
    net_points: float
    net_dollars: float
    avg_points: float
    avg_return_pct: float
    structures: Counter
    symbols: Counter

    @property
    def win_rate(self) -> float:
        if self.closed_trades == 0:
            return 0.0
        return self.wins / self.closed_trades


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest DaySpy directional option spreads using theoretical pricing from underlying bars."
    )
    parser.add_argument("bars_csv", help="Path to a CSV file containing 1-minute OHLCV bars.")
    parser.add_argument("--check", action="store_true", help="Validate dependencies and CSV layout only.")
    parser.add_argument("--symbol", help="Use this symbol when the CSV has no symbol column.")
    parser.add_argument("--signals-out", default="", help="Optional CSV path for emitted signals.")
    parser.add_argument("--trades-out", default="option_backtest_trades.csv", help="Where closed option trades should be written.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-signal terminal output and print only the summary.")
    parser.add_argument("--vol-lookback", type=int, default=120, help="Lookback bars for realized-vol estimate.")
    parser.add_argument("--risk-free-rate", type=float, default=0.0, help="Annual risk-free rate for Black-Scholes.")
    parser.add_argument("--contracts", type=int, default=10, help="Number of spreads to trade per signal. Defaults to 10.")
    parser.add_argument(
        "--invert-directional-spreads",
        action="store_true",
        help="Enter the opposite directional debit spread on each TREND_LONG/TREND_SHORT signal, but keep the original entry/exit timing.",
    )
    parser.add_argument("--timestamp-col", default="timestamp", help="CSV column containing the bar timestamp.")
    parser.add_argument("--symbol-col", default="symbol", help="CSV column containing the ticker symbol.")
    parser.add_argument("--open-col", default="open", help="CSV column containing the open price.")
    parser.add_argument("--high-col", default="high", help="CSV column containing the high price.")
    parser.add_argument("--low-col", default="low", help="CSV column containing the low price.")
    parser.add_argument("--close-col", default="close", help="CSV column containing the close price.")
    parser.add_argument("--volume-col", default="volume", help="CSV column containing the volume.")
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> OptionBacktestConfig:
    load_local_env()

    env_symbols = [x.strip().upper() for x in os.getenv("SYMBOLS", "").split(",") if x.strip()]
    symbols = [args.symbol.strip().upper()] if args.symbol else env_symbols or ["SPY"]

    return OptionBacktestConfig(
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
        trades_csv_path=args.trades_out,
        vol_lookback_bars=max(20, args.vol_lookback),
        risk_free_rate=args.risk_free_rate,
        invert_directional_spreads=args.invert_directional_spreads,
        contracts_per_trade=max(1, args.contracts),
    )


def run_startup_check(cfg: OptionBacktestConfig) -> int:
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


def replay_signals(frame, cfg: OptionBacktestConfig) -> List[EmittedSignal]:
    sink = CollectingSink(
        cfg=cfg,
        csv_path=cfg.signal_csv_path,
        discord_webhook_url=cfg.discord_webhook_url,
        print_json=cfg.print_json,
        console_output=not cfg.quiet,
    )
    pipeline = StrategyPipeline(cfg, alerts=sink)

    for row in frame.itertuples(index=False):
        pipeline.process_bar(
            symbol=str(getattr(row, cfg.symbol_col)).upper(),
            ts=getattr(row, cfg.timestamp_col),
            open_=float(getattr(row, cfg.open_col)),
            high=float(getattr(row, cfg.high_col)),
            low=float(getattr(row, cfg.low_col)),
            close=float(getattr(row, cfg.close_col)),
            volume=float(getattr(row, cfg.volume_col)),
            emit_signals=True,
        )

    return sink.signals


def parse_leg(text: str) -> ParsedLeg:
    match = LEG_RE.match(text.strip())
    if not match:
        raise ValueError(f"Unable to parse option leg: {text}")
    side, quantity, symbol, expiry, strike, right = match.groups()
    return ParsedLeg(
        side=side,
        quantity=int(quantity),
        symbol=symbol,
        expiry=expiry,
        strike=float(strike),
        right=right,
    )


def scale_legs(legs: List[ParsedLeg], multiplier: int) -> List[ParsedLeg]:
    size = max(1, int(multiplier))
    if size == 1:
        return legs
    return [
        ParsedLeg(
            side=leg.side,
            quantity=leg.quantity * size,
            symbol=leg.symbol,
            expiry=leg.expiry,
            strike=leg.strike,
            right=leg.right,
        )
        for leg in legs
    ]


def expiry_close_utc(expiry_text: str):
    local_dt = require_toronto_tz().localize(pd.Timestamp(f"{expiry_text} 16:00:00").to_pydatetime())
    return pd.Timestamp(local_dt).tz_convert("UTC")


def norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def black_scholes_price(spot: float, strike: float, years: float, rate: float, sigma: float, right: str) -> float:
    intrinsic = max(spot - strike, 0.0) if right == "CALL" else max(strike - spot, 0.0)
    if years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return intrinsic

    sigma = max(sigma, 1e-6)
    sqrt_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if right == "CALL":
        return spot * norm_cdf(d1) - strike * math.exp(-rate * years) * norm_cdf(d2)
    return strike * math.exp(-rate * years) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def years_to_expiry(now_ts, expiry_ts) -> float:
    seconds = max(0.0, (expiry_ts - now_ts).total_seconds())
    return seconds / (365.0 * 24.0 * 60.0 * 60.0)


def price_plan(spot: float, now_ts, expiry_ts, sigma: float, rate: float, legs: List[ParsedLeg]) -> float:
    total = 0.0
    years = years_to_expiry(now_ts, expiry_ts)
    for leg in legs:
        value = black_scholes_price(spot, leg.strike, years, rate, sigma, leg.right)
        signed_qty = leg.quantity if leg.side == "BUY" else -leg.quantity
        total += signed_qty * value
    return total


def estimate_realized_vol(symbol_frame, asof_ts, lookback_bars: int) -> float:
    history = symbol_frame.loc[:asof_ts, "close"].tail(lookback_bars + 1)
    returns = history.pct_change().dropna()
    if len(returns) < 20:
        return 0.25
    sigma = float(returns.std(ddof=0) * math.sqrt(252.0 * 390.0))
    return min(max(sigma, 0.10), 1.50)


def lookup_spot(symbol_frame, ts) -> Optional[float]:
    history = symbol_frame.loc[:ts, "close"]
    if history.empty:
        return None
    return float(history.iloc[-1])


def write_trades_csv(path: str, trades: List[ClosedOptionTrade]) -> None:
    if not path.strip():
        return

    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "symbol": trade.symbol,
            "structure": trade.structure,
            "entry_time_utc": trade.entry_time.isoformat(),
            "exit_time_utc": trade.exit_time.isoformat(),
            "expiry_date": trade.expiry_date,
            "entry_underlying": round(trade.entry_underlying, 4),
            "exit_underlying": round(trade.exit_underlying, 4),
            "entry_value": round(trade.entry_value, 4),
            "exit_value": round(trade.exit_value, 4),
            "pnl_points": round(trade.pnl_points, 4),
            "pnl_dollars": round(trade.pnl_dollars, 2),
            "return_pct": round(trade.return_pct * 100.0, 4),
            "sigma": round(trade.sigma, 4),
            "exit_reason": trade.exit_reason,
            "legs": trade.legs_text,
        }
        for trade in trades
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def close_trade(
    trade: OpenOptionTrade,
    exit_ts,
    exit_spot: float,
    exit_reason: str,
    rate: float,
) -> ClosedOptionTrade:
    exit_value = price_plan(exit_spot, exit_ts, trade.expiry_ts, trade.sigma, rate, trade.legs)
    return ClosedOptionTrade(
        symbol=trade.entry_signal.symbol,
        structure=trade.structure,
        entry_time=trade.entry_signal.timestamp,
        exit_time=exit_ts,
        expiry_date=str(trade.expiry_ts.date()),
        entry_underlying=trade.entry_signal.price,
        exit_underlying=exit_spot,
        entry_value=trade.entry_value,
        exit_value=exit_value,
        sigma=trade.sigma,
        exit_reason=exit_reason,
        legs_text=" ; ".join(
            f"{leg.side} {leg.quantity} {leg.symbol} {leg.expiry} {int(leg.strike) if leg.strike.is_integer() else leg.strike} {leg.right}"
            for leg in trade.legs
        ),
    )


def build_backtest_option_plan(signal: EmittedSignal, cfg: OptionBacktestConfig):
    if not cfg.invert_directional_spreads or signal.signal not in (SignalType.TREND_LONG, SignalType.TREND_SHORT):
        return build_option_plan(signal, cfg)

    stop_price = signal.extras.get("stop_price")
    if stop_price is None:
        risk = max(cfg.option_strike_step, signal.price * 0.003)
    else:
        risk = abs(signal.price - float(stop_price))
    target_move = max(risk * cfg.min_rr, signal.price * 0.003)

    if signal.signal == SignalType.TREND_LONG:
        mirrored_signal = EmittedSignal(
            symbol=signal.symbol,
            signal=SignalType.TREND_SHORT,
            regime=signal.regime,
            price=signal.price,
            timestamp=signal.timestamp,
            reason=signal.reason,
            extras={
                "stop_price": round(signal.price + risk, 4),
                "target_price": round(signal.price - target_move, 4),
            },
        )
    else:
        mirrored_signal = EmittedSignal(
            symbol=signal.symbol,
            signal=SignalType.TREND_LONG,
            regime=signal.regime,
            price=signal.price,
            timestamp=signal.timestamp,
            reason=signal.reason,
            extras={
                "stop_price": round(max(0.01, signal.price - risk), 4),
                "target_price": round(signal.price + target_move, 4),
            },
        )

    option_plan = build_option_plan(mirrored_signal, cfg)
    if option_plan is None:
        return None

    note = f"inverted_from={signal.signal.value}"
    option_plan.notes = f"{option_plan.notes}, {note}" if option_plan.notes else note
    return option_plan


def summarize_option_trades(
    signals: List[EmittedSignal],
    trades: List[ClosedOptionTrade],
    open_positions: Dict[str, OpenOptionTrade],
) -> OptionBacktestSummary:
    wins = sum(1 for trade in trades if trade.pnl_points > 0)
    losses = sum(1 for trade in trades if trade.pnl_points < 0)
    flats = sum(1 for trade in trades if trade.pnl_points == 0)
    net_points = sum(trade.pnl_points for trade in trades)
    net_dollars = sum(trade.pnl_dollars for trade in trades)
    avg_points = net_points / len(trades) if trades else 0.0
    avg_return_pct = sum(trade.return_pct for trade in trades) / len(trades) if trades else 0.0
    structures = Counter(trade.structure for trade in trades)
    symbols = Counter(trade.symbol for trade in trades)

    return OptionBacktestSummary(
        total_signals=len(signals),
        total_option_entries=len(trades) + len(open_positions),
        closed_trades=len(trades),
        wins=wins,
        losses=losses,
        flats=flats,
        open_positions=len(open_positions),
        net_points=net_points,
        net_dollars=net_dollars,
        avg_points=avg_points,
        avg_return_pct=avg_return_pct,
        structures=structures,
        symbols=symbols,
    )


def print_summary(summary: OptionBacktestSummary) -> None:
    print(
        f"Option Summary | signals={summary.total_signals} | option_entries={summary.total_option_entries} | "
        f"closed={summary.closed_trades} | open={summary.open_positions}"
    )
    print(
        f"Option Trades | wins={summary.wins} | losses={summary.losses} | flats={summary.flats} | "
        f"win_rate={summary.win_rate * 100.0:.2f}%"
    )
    print(
        f"Option PnL | net_points={summary.net_points:.4f} | net_dollars={summary.net_dollars:.2f} | "
        f"avg_points={summary.avg_points:.4f} | avg_return={summary.avg_return_pct * 100.0:.2f}%"
    )
    if summary.structures:
        by_structure = ", ".join(f"{name}={count}" for name, count in sorted(summary.structures.items()))
        print(f"By Structure | {by_structure}")
    if summary.symbols:
        by_symbol = ", ".join(f"{name}={count}" for name, count in sorted(summary.symbols.items()))
        print(f"By Symbol | {by_symbol}")


def simulate_directional_option_trades(frame, signals: List[EmittedSignal], cfg: OptionBacktestConfig):
    indexed: Dict[str, object] = {}
    for symbol in sorted(frame[cfg.symbol_col].astype(str).str.upper().unique()):
        symbol_frame = frame[frame[cfg.symbol_col].astype(str).str.upper() == symbol].copy()
        symbol_frame = symbol_frame.sort_values(cfg.timestamp_col).set_index(cfg.timestamp_col)
        indexed[symbol] = symbol_frame

    open_positions: Dict[str, OpenOptionTrade] = {}
    closed: List[ClosedOptionTrade] = []

    for signal in signals:
        symbol = signal.symbol
        symbol_frame = indexed.get(symbol)
        if symbol_frame is None:
            continue

        # Expire any open trade before processing the next signal for that symbol.
        current = open_positions.get(symbol)
        if current is not None and signal.timestamp >= current.expiry_ts:
            exit_spot = lookup_spot(symbol_frame, current.expiry_ts)
            if exit_spot is not None:
                closed.append(close_trade(current, current.expiry_ts, exit_spot, "expiry", cfg.risk_free_rate))
            del open_positions[symbol]
            current = None

        if signal.signal in (SignalType.TREND_LONG, SignalType.TREND_SHORT):
            if current is not None:
                continue

            option_plan = build_backtest_option_plan(signal, cfg)
            if option_plan is None or option_plan.structure not in {"call_debit_spread", "put_debit_spread"}:
                continue

            legs = scale_legs([parse_leg(text) for text in option_plan.legs], cfg.contracts_per_trade)
            expiry_ts = expiry_close_utc(option_plan.expiry)
            sigma = estimate_realized_vol(symbol_frame, signal.timestamp, cfg.vol_lookback_bars)
            entry_value = price_plan(signal.price, signal.timestamp, expiry_ts, sigma, cfg.risk_free_rate, legs)
            if entry_value <= 0:
                continue

            open_positions[symbol] = OpenOptionTrade(
                entry_signal=signal,
                structure=option_plan.structure,
                legs=legs,
                expiry_ts=expiry_ts,
                entry_value=entry_value,
                sigma=sigma,
            )
            continue

        current = open_positions.get(symbol)
        if current is None:
            continue

        if current.entry_signal.signal == SignalType.TREND_LONG and signal.signal == SignalType.EXIT_LONG:
            closed.append(close_trade(current, signal.timestamp, signal.price, "signal_exit", cfg.risk_free_rate))
            del open_positions[symbol]
            continue

        if current.entry_signal.signal == SignalType.TREND_SHORT and signal.signal == SignalType.EXIT_SHORT:
            closed.append(close_trade(current, signal.timestamp, signal.price, "signal_exit", cfg.risk_free_rate))
            del open_positions[symbol]

    last_ts = frame[cfg.timestamp_col].max()
    for symbol, current in list(open_positions.items()):
        symbol_frame = indexed.get(symbol)
        if symbol_frame is None:
            continue
        if current.expiry_ts <= last_ts:
            exit_ts = current.expiry_ts
            exit_reason = "expiry"
        else:
            exit_ts = last_ts
            exit_reason = "dataset_end"
        exit_spot = lookup_spot(symbol_frame, exit_ts)
        if exit_spot is not None:
            closed.append(close_trade(current, exit_ts, exit_spot, exit_reason, cfg.risk_free_rate))
        del open_positions[symbol]

    return closed, open_positions


def run_backtest(cfg: OptionBacktestConfig) -> int:
    frame = load_bars_frame(cfg)
    signals = replay_signals(frame, cfg)
    trades, open_positions = simulate_directional_option_trades(frame, signals, cfg)
    write_trades_csv(cfg.trades_csv_path, trades)
    summary = summarize_option_trades(signals, trades, open_positions)
    print_summary(summary)
    print("Assumption | theoretical Black-Scholes pricing from underlying bars; historical option IV/fills are not included.")
    return 0


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
