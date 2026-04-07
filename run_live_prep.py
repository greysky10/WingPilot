#! python3.12
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from corridor.config import CorridorConfig
from corridor.data.ib_contracts import default_center_rounding_for_symbol
from corridor.data.historical_loader import HistoricalLoadConfig, load_intraday_bars
from corridor.data.ib_loader import IBHistoricalRequest, fetch_intraday_bars
from corridor.models import CenterMethod
from corridor.options.butterfly_selector import select_butterflies
from corridor.options.chain_loader import IBOptionChainLoader
from corridor.strategy.center_estimator import CenterEstimator
from corridor.strategy.regime import RangeRegimeDetector
from strategy import load_local_env


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a live corridor snapshot and candidate butterflies.")
    parser.add_argument("--symbol", default="SPX", help="Ticker symbol.")
    parser.add_argument("--mode", default="delayed", choices=["delayed", "live"], help="IB market-data mode.")
    parser.add_argument("--bars-csv", help="Optional bars CSV instead of IBKR.")
    parser.add_argument("--center-method", default=CenterMethod.VWAP.value, choices=[item.value for item in CenterMethod])
    parser.add_argument("--client-id", type=int, default=51, help="IB client id.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args(argv)


def load_recent_frame(cfg: CorridorConfig, args: argparse.Namespace) -> pd.DataFrame:
    if args.bars_csv:
        return load_intraday_bars(HistoricalLoadConfig(csv_path=Path(args.bars_csv), symbol=cfg.symbol))

    now_utc = pd.Timestamp.utcnow()
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    else:
        now_utc = now_utc.tz_convert("UTC")

    request = IBHistoricalRequest(
        symbol=cfg.symbol,
        start=now_utc - pd.Timedelta(days=5),
        end=now_utc,
        bar_size=cfg.timeframe,
        host=cfg.ib_host,
        port=cfg.ib_port,
        client_id=cfg.ib_client_id,
        exchange=cfg.ib_exchange,
        currency=cfg.ib_currency,
        what_to_show=cfg.ib_what_to_show,
        use_rth=cfg.ib_use_rth,
        chunk_duration="5 D",
    )
    return fetch_intraday_bars(request)


def main(argv: Optional[list[str]] = None) -> int:
    load_local_env()
    args = parse_args(argv)
    cfg = CorridorConfig(
        symbol=args.symbol.upper(),
        center_method=CenterMethod(args.center_method),
        center_rounding=default_center_rounding_for_symbol(args.symbol.upper()),
        ib_host=os.getenv("IB_HOST", "127.0.0.1"),
        ib_port=int(os.getenv("IB_PORT", "4001")),
        ib_client_id=args.client_id,
        payoff_mode="underlying_only",
    )
    frame = load_recent_frame(cfg, args)
    detector = RangeRegimeDetector(cfg)
    estimator = CenterEstimator(cfg)
    regime = detector.evaluate(frame)
    center = estimator.estimate(frame)
    latest = frame.iloc[-1]

    payload = {
        "symbol": cfg.symbol,
        "timestamp": pd.Timestamp(latest["timestamp"]).isoformat(),
        "price": float(latest["close"]),
        "mode": args.mode,
        "regime": regime.regime.value if regime is not None else None,
        "center": center.center_price if center is not None else None,
        "center_band": {
            "lower": center.lower_band if center is not None else None,
            "upper": center.upper_band if center is not None else None,
        },
        "candidates": [],
    }

    if regime is not None and regime.regime.value == "RANGE" and center is not None and not args.bars_csv:
        loader = IBOptionChainLoader(cfg.ib_host, cfg.ib_port, cfg.ib_client_id + 1, cfg.ib_exchange, cfg.ib_currency)
        quotes = loader.load_candidates(
            cfg.symbol,
            center.center_price,
            cfg.butterfly_width,
            cfg.dte_min,
            cfg.dte_max,
            market_data_type=3 if args.mode == "delayed" else 1,
        )
        payload["candidates"] = [
            {
                "expiry": candidate.expiry,
                "lower_strike": candidate.lower_strike,
                "body_strike": candidate.body_strike,
                "upper_strike": candidate.upper_strike,
                "net_debit": candidate.net_debit,
                "total_spread": candidate.total_spread,
                "max_risk": candidate.max_risk,
                "max_reward": candidate.max_reward,
                "right": candidate.right,
            }
            for candidate in select_butterflies(quotes, center.center_price, cfg.butterfly_width, cfg)
        ]

    output_path = Path(args.output) if args.output else Path("corridor_outputs") / f"{cfg.symbol}_live_prep.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Live prep saved to {output_path}")
    print(f"Regime={payload['regime']} | center={payload['center']} | candidates={len(payload['candidates'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
