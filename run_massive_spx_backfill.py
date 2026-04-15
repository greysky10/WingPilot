#! python3.12
from __future__ import annotations

import argparse
import os
from datetime import date
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from corridor.data.massive_options import (
    MassiveAPIError,
    MassiveAuthorizationError,
    MassiveBackfillConfig,
    MassiveClientConfig,
    MassiveRESTClient,
    StrategyUniverseConfig,
    backfill_massive_spx_options,
)
from strategy import load_local_env


DEFAULT_OUTPUT_DIR = Path("data") / "massive_spx_history"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill 1 year of SPX options daily history from Massive by enumerating contracts first and then "
            "requesting daily bars for each option ticker. Use --strategy-only to build a strategy-relevant subset "
            "from local SPX bars instead of the full chain universe."
        )
    )
    parser.add_argument("--start", help="Inclusive start date in YYYY-MM-DD format. Defaults to 365 days before --end.")
    parser.add_argument("--end", help="Inclusive end date in YYYY-MM-DD format. Defaults to yesterday in America/Toronto.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for contracts, checkpoint, part files, and the final dataset.",
    )
    parser.add_argument(
        "--format",
        default="parquet",
        choices=["parquet", "csv"],
        help="Preferred output format. Falls back to CSV when parquet support is unavailable.",
    )
    parser.add_argument(
        "--contract-underlyings",
        default="I:SPX,SPX",
        help="Comma-separated contract-discovery underlyings. The runner tries them in order until Massive returns contracts.",
    )
    parser.add_argument(
        "--expiration-buffer-days",
        type=int,
        default=365,
        help="How far past --end to search option expirations so longer-dated contracts active during the window are included.",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Contracts to process before writing a checkpointed part file.")
    parser.add_argument("--contract-page-size", type=int, default=1000, help="Per-page size for the All Contracts endpoint.")
    parser.add_argument("--bars-limit", type=int, default=50000, help="Limit passed to Massive custom bars requests.")
    parser.add_argument("--contract-limit", type=int, help="Optional cap on the number of contracts for smoke tests.")
    parser.add_argument(
        "--strategy-only",
        action="store_true",
        help="Use local underlying bars to build a strategy-only option universe instead of the full chain.",
    )
    parser.add_argument(
        "--bars-csv",
        default="",
        help="Underlying SPX intraday bars CSV used to narrow strikes and expiries in --strategy-only mode.",
    )
    parser.add_argument(
        "--bars-symbol",
        default="SPX",
        help="Underlying symbol expected in --bars-csv. Only used in --strategy-only mode.",
    )
    parser.add_argument("--dte-min", type=int, default=4, help="Minimum calendar DTE for strategy-only contract discovery.")
    parser.add_argument("--dte-max", type=int, default=10, help="Maximum calendar DTE for strategy-only contract discovery.")
    parser.add_argument(
        "--center-rounding",
        type=float,
        default=5.0,
        help="Strike rounding increment used to expand strategy-only strike windows.",
    )
    parser.add_argument(
        "--butterfly-width",
        type=float,
        default=10.0,
        help="Base butterfly width in strike points used for strategy-only strike windows.",
    )
    parser.add_argument(
        "--wing-mode",
        default="symmetric",
        choices=["symmetric", "broken_upper", "broken_lower", "adaptive"],
        help="Butterfly geometry used for strategy-only strike windows.",
    )
    parser.add_argument(
        "--broken-wing-extra-width",
        type=float,
        default=0.0,
        help="Extra broken-wing width used when building strategy-only strike windows.",
    )
    parser.add_argument(
        "--strategy-strike-buffer-points",
        type=float,
        default=10.0,
        help="Extra points added above and below the strategy-only strike window.",
    )
    parser.add_argument(
        "--strategy-slice-days",
        type=int,
        default=7,
        help="Trade-date slice size for strategy-only contract discovery.",
    )
    parser.add_argument(
        "--strategy-contract-types",
        default="call",
        help="Comma-separated contract types for strategy-only discovery. Defaults to call because the current backtest uses call butterflies.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="HTTP timeout per Massive request.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retry count for retryable Massive requests.")
    parser.add_argument("--max-rate-limit-retries", type=int, default=100, help="How many rate-limit sleeps to tolerate before failing.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0, help="Base exponential backoff in seconds.")
    parser.add_argument("--min-request-interval-seconds", type=float, default=1.0, help="Minimum delay between Massive requests.")
    parser.add_argument("--rate-limit-sleep-seconds", type=float, default=70.0, help="Sleep time after Massive reports a per-minute rate limit.")
    parser.add_argument("--api-key", default="", help="Massive API key. Prefer MASSIVE_API_KEY in the environment instead.")
    parser.add_argument("--api-key-env", default="MASSIVE_API_KEY", help="Environment variable that stores the Massive API key.")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume logic.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    load_local_env()
    args = parse_args(argv)

    end_date = _parse_date(args.end) if args.end else _default_end_date()
    start_date = _parse_date(args.start) if args.start else end_date - timedelta(days=365)
    strategy_universe = _build_strategy_universe(args) if args.strategy_only else None
    api_key = str(args.api_key or os.getenv(args.api_key_env, "")).strip()
    if not api_key:
        raise SystemExit(
            f"Massive API key not found. Set {args.api_key_env} in the environment or pass --api-key explicitly."
        )

    client = MassiveRESTClient(
        MassiveClientConfig(
            api_key=api_key,
            timeout_seconds=float(args.timeout_seconds),
            max_retries=max(1, int(args.max_retries)),
            max_rate_limit_retries=max(1, int(args.max_rate_limit_retries)),
            retry_backoff_seconds=max(0.1, float(args.retry_backoff_seconds)),
            min_request_interval_seconds=max(0.0, float(args.min_request_interval_seconds)),
            rate_limit_sleep_seconds=max(1.0, float(args.rate_limit_sleep_seconds)),
        )
    )
    cfg = MassiveBackfillConfig(
        start_date=start_date,
        end_date=end_date,
        output_dir=Path(args.output_dir),
        contract_underlyings=tuple(item.strip() for item in str(args.contract_underlyings).split(",") if item.strip()),
        output_format=args.format,
        contract_page_size=args.contract_page_size,
        bars_limit=args.bars_limit,
        batch_size=args.batch_size,
        expiration_buffer_days=args.expiration_buffer_days,
        contract_limit=args.contract_limit,
        resume=not args.no_resume,
        strategy_universe=strategy_universe,
    )

    try:
        result = backfill_massive_spx_options(client, cfg)
    except MassiveAuthorizationError as exc:
        raise SystemExit(f"Massive authorization/entitlement error: {exc}") from exc
    except MassiveAPIError as exc:
        raise SystemExit(f"Massive API error: {exc}") from exc

    print(f"Massive SPX options backfill complete for {start_date.isoformat()} through {end_date.isoformat()}.")
    print(f"Contract discovery underlying used: {result.selected_contract_underlying}")
    print(f"Contracts saved to {result.contracts_path}")
    print(f"Final dataset saved to {result.final_dataset_path}")
    print(f"Contracts processed: {result.contract_count}")
    print(f"Output rows: {result.row_count}")
    print(f"Part files written: {result.part_count}")
    print(f"Output format: {result.output_format}")
    return 0


def _default_end_date() -> date:
    today = pd.Timestamp.now(tz="America/Toronto").date()
    return today - timedelta(days=1)


def _parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def _build_strategy_universe(args: argparse.Namespace) -> StrategyUniverseConfig:
    bars_csv = str(args.bars_csv or "").strip()
    if not bars_csv:
        raise SystemExit("--strategy-only requires --bars-csv so the puller can narrow strikes from local SPX bars.")
    return StrategyUniverseConfig(
        bars_csv=Path(bars_csv),
        symbol=str(args.bars_symbol or "SPX").upper(),
        contract_types=tuple(
            item.strip().lower()
            for item in str(args.strategy_contract_types or "").split(",")
            if item.strip()
        ),
        dte_min=max(0, int(args.dte_min)),
        dte_max=max(0, int(args.dte_max)),
        center_rounding=max(0.5, float(args.center_rounding)),
        butterfly_width=max(0.5, float(args.butterfly_width)),
        wing_mode=str(args.wing_mode),
        broken_wing_extra_width=max(0.0, float(args.broken_wing_extra_width)),
        strike_buffer_points=max(0.0, float(args.strategy_strike_buffer_points)),
        slice_days=max(1, int(args.strategy_slice_days)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
