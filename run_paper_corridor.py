#! python3.12
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from corridor.config import CorridorConfig
from corridor.models import CenterMethod
from corridor.execution.paper import PaperCorridorRunner, PaperRunnerConfig
from strategy import load_local_env


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the corridor strategy in dry-run or IB paper-execution mode.")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol.")
    parser.add_argument("--mode", default="delayed", choices=["delayed", "live"], help="IB market-data mode.")
    parser.add_argument("--host", default=os.getenv("IB_HOST", "127.0.0.1"), help="IB host.")
    parser.add_argument("--port", type=int, default=int(os.getenv("IB_PORT", "4001")), help="IB API port.")
    parser.add_argument("--client-id", type=int, default=71, help="IB client id.")
    parser.add_argument("--quantity", type=int, default=1, help="Number of butterfly combos per signal.")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval for new completed bars.")
    parser.add_argument("--history-days", type=int, default=5, help="Historical days to seed before live polling.")
    parser.add_argument("--center-method", default=CenterMethod.VWAP.value, choices=[item.value for item in CenterMethod])
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    parser.add_argument("--paper-execution", action="store_true", help="Submit combo orders to the connected IB paper account.")
    parser.add_argument("--sync-on-start", action="store_true", help="Keep the seeded corridor state instead of resetting flat.")
    parser.add_argument("--once", action="store_true", help="Run a single polling pass and exit.")
    parser.add_argument("--check", action="store_true", help="Validate connectivity, compute the current snapshot, and exit.")
    return parser.parse_args(argv)


def build_configs(args: argparse.Namespace) -> tuple[CorridorConfig, PaperRunnerConfig]:
    corridor_cfg = CorridorConfig(
        symbol=args.symbol.upper(),
        center_method=CenterMethod(args.center_method),
        payoff_mode="underlying_only",
        ib_client_id=args.client_id,
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path("corridor_outputs") / "paper_runner" / corridor_cfg.symbol
    runner_cfg = PaperRunnerConfig(
        symbol=corridor_cfg.symbol,
        mode=args.mode,
        host=args.host,
        port=int(args.port),
        client_id=args.client_id,
        quantity=max(1, args.quantity),
        poll_seconds=max(5, args.poll_seconds),
        history_days=max(1, args.history_days),
        start_flat=not args.sync_on_start,
        paper_execution=args.paper_execution,
        once=args.once,
        check_only=args.check,
        output_dir=output_dir,
    )
    return corridor_cfg, runner_cfg


def main(argv: Optional[list[str]] = None) -> int:
    load_local_env()
    args = parse_args(argv)
    corridor_cfg, runner_cfg = build_configs(args)
    runner = PaperCorridorRunner(corridor_cfg, runner_cfg)
    try:
        return runner.run()
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        print(message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
