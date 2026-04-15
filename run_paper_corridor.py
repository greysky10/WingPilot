#! python3.12
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from corridor.config import CorridorConfig
from corridor.data.ib_contracts import default_center_rounding_for_symbol
from corridor.models import CenterMethod
from corridor.execution.paper import PaperCorridorRunner, PaperRunnerConfig
from strategy import load_local_env


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the corridor strategy in dry-run or IB paper-execution mode.")
    parser.add_argument("--symbol", default="SPX", help="Ticker symbol.")
    parser.add_argument("--mode", default="delayed", choices=["delayed", "live"], help="IB market-data mode.")
    parser.add_argument("--host", default=os.getenv("IB_HOST", "127.0.0.1"), help="IB host.")
    parser.add_argument("--port", type=int, default=int(os.getenv("IB_PORT", "4001")), help="IB API port.")
    parser.add_argument("--client-id", type=int, default=71, help="IB client id.")
    parser.add_argument("--quantity", type=int, default=1, help="Number of butterfly combos per signal.")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval for new completed bars.")
    parser.add_argument("--history-days", type=int, default=5, help="Historical days to seed before live polling.")
    parser.add_argument("--center-method", default=CenterMethod.VWAP.value, choices=[item.value for item in CenterMethod])
    parser.add_argument("--body-strike-offset-points", type=float, default=0.0, help="Offset the target butterfly body strike away from the rounded center by this many strike points.")
    parser.add_argument("--butterfly-width", type=float, default=10.0, help="Butterfly wing width in strike points.")
    parser.add_argument(
        "--wing-mode",
        default="symmetric",
        choices=["symmetric", "broken_upper", "broken_lower", "adaptive"],
        help="Butterfly geometry mode. Adaptive keeps symmetric as default and only falls back to broken-wing candidates when the symmetric candidate is execution-poor.",
    )
    parser.add_argument(
        "--option-right",
        default="call",
        choices=["call", "put", "auto"],
        help="Option side preference for candidate selection.",
    )
    parser.add_argument(
        "--broken-wing-extra-width",
        type=float,
        default=0.0,
        help="Extra strike width added to the broken side when wing-mode is asymmetric or adaptive.",
    )
    parser.add_argument("--coverage-band-width", type=float, default=20.0, help="Total corridor coverage width in strike points.")
    parser.add_argument("--center-tolerance", type=float, default=2.5, help="Minimum half-width of the center tolerance band.")
    parser.add_argument("--center-tolerance-atr-multiplier", type=float, default=1.0, help="Dynamic tolerance multiplier: actual_tolerance = max(center_tolerance, ATR * multiplier).")
    parser.add_argument("--atr-lookback", type=int, default=14, help="ATR lookback bars used for dynamic center tolerance.")
    parser.add_argument("--recenter-threshold", type=float, default=3.5, help="Distance from center required before drift counts as rebuild-worthy.")
    parser.add_argument("--drift-persistence-bars", type=int, default=2, help="Completed bars required outside tolerance before a rebuild is requested.")
    parser.add_argument("--rebuild-cooldown-minutes", type=int, default=15, help="Minimum minutes between rebuilds.")
    parser.add_argument("--max-layers", type=int, default=3, help="Maximum number of simultaneously active butterfly layers.")
    parser.add_argument("--candidate-body-search-steps", type=int, default=2, help="How many nearby rounded body strikes to search on each side of the target center.")
    parser.add_argument("--dte-min", type=int, default=4, help="Minimum calendar days to expiry for candidate butterflies.")
    parser.add_argument("--dte-max", type=int, default=10, help="Maximum calendar days to expiry for candidate butterflies.")
    parser.add_argument("--default-dte", type=int, default=7, help="Metadata DTE assigned to newly opened corridor layers.")
    parser.add_argument(
        "--layer-dte-targets",
        default="",
        help="Optional comma-separated DTE ladder opened together on entry, for example 21,28,35. Empty keeps single-DTE behavior.",
    )
    parser.add_argument(
        "--layer-exit-scope",
        default="all",
        choices=["all", "individual"],
        help="Close all active layers from the primary signal, or close each layer independently.",
    )
    parser.add_argument(
        "--allow-daily-entry-additions",
        action="store_true",
        help="When positions are already open overnight, allow one new entry batch on later sessions if the primary entry filter passes and layer capacity remains.",
    )
    parser.add_argument("--max-option-spread", type=float, default=0.25, help="Maximum aggregate bid/ask spread allowed for a butterfly candidate.")
    parser.add_argument("--near-spread-dte-max", type=int, default=0, help="If set, expiries at or below this DTE use --near-max-option-spread.")
    parser.add_argument("--near-max-option-spread", type=float, default=0.0, help="Absolute spread cap for nearer-DTE entries. Set 0 to reuse --max-option-spread.")
    parser.add_argument("--mid-max-option-spread", type=float, default=0.0, help="Absolute spread cap for middle-DTE entries between the near and far buckets. Set 0 to reuse --max-option-spread.")
    parser.add_argument("--far-spread-dte-min", type=int, default=0, help="If set, expiries at or above this DTE use --far-max-option-spread.")
    parser.add_argument("--far-max-option-spread", type=float, default=0.0, help="Absolute spread cap for farther-DTE entries. Set 0 to reuse --max-option-spread.")
    parser.add_argument("--primary-entry-end", default="15:30", help="Latest New York time allowed for a new primary entry.")
    parser.add_argument("--primary-entry-min-center-confidence", type=float, default=0.0, help="Minimum center confidence required to open a primary layer.")
    parser.add_argument("--primary-entry-max-momentum-pct", type=float, default=1.0, help="Maximum absolute momentum_pct allowed for a new primary layer.")
    parser.add_argument("--primary-entry-max-volume-ratio", type=float, default=999.0, help="Maximum volume_ratio allowed for a new primary layer.")
    parser.add_argument("--skip-entry-weekdays", default="", help="Comma-separated New York weekdays to block for new entries, for example mon,fri.")
    parser.add_argument("--primary-stop-loss-pct", type=float, default=0.0, help="Close all active butterflies if the primary layer falls below this return threshold.")
    parser.add_argument("--primary-take-profit-pct", type=float, default=0.0, help="Close all active butterflies if the primary layer rises above this return threshold.")
    parser.add_argument("--block-same-day-reentry-after-take-profit", action="store_true", help="After a take-profit closes exposure, block fresh entries for the rest of that New York session.")
    parser.add_argument("--hold-overnight", action="store_true", help="Do not force-flatten active positions at the session end; resume them on the next sync-on-start run.")
    parser.add_argument("--max-hold-sessions", type=int, default=0, help="Close the primary position after this many trading sessions. Set 0 to disable.")
    parser.add_argument("--close-when-dte-lte", type=int, default=0, help="Close the primary position when its remaining calendar DTE is less than or equal to this threshold. Set 0 to disable.")
    parser.add_argument("--skip-event-days", action="store_true", help="Block new primary entries on configured event dates.")
    parser.add_argument("--event-dates", default="", help="Comma-separated New York dates to block, for example 2026-04-10,2026-05-06.")
    parser.add_argument("--skip-gap-days", action="store_true", help="Block new primary entries when the session gap exceeds --max-entry-gap-pct.")
    parser.add_argument("--max-entry-gap-pct", type=float, default=0.0, help="Maximum absolute session gap percentage allowed for a new primary entry. Set 0 to disable.")
    parser.add_argument("--max-spread-pct-of-debit", type=float, default=0.40, help="Maximum allowed total_spread / net_debit ratio for paper execution.")
    parser.add_argument("--combo-fill-wait-seconds", type=float, default=1.0, help="Seconds to wait for each combo limit attempt before cancelling/chasing.")
    parser.add_argument("--combo-chase-steps", type=int, default=3, help="Maximum number of combo limit attempts before giving up.")
    parser.add_argument("--combo-chase-spread-fraction", type=float, default=0.20, help="Fraction of combo spread used for each chase step.")
    parser.add_argument("--combo-max-total-debit-ratio", type=float, default=1.15, help="Absolute maximum BUY debit as a multiple of the initial combo midpoint during chasing.")
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    parser.add_argument("--paper-execution", action="store_true", help="Submit combo orders to the connected IB paper account.")
    parser.add_argument(
        "--paper-smoke-mode",
        action="store_true",
        help="Run a higher-frequency paper execution test mode that relaxes signal gating and forces an intraday close after a fixed hold time.",
    )
    parser.add_argument(
        "--smoke-max-entries-per-day",
        type=int,
        default=1,
        help="Maximum number of paper smoke-mode entries allowed per New York session.",
    )
    parser.add_argument(
        "--smoke-quantity",
        type=int,
        default=1,
        help="Number of butterfly combos per smoke-mode entry.",
    )
    parser.add_argument(
        "--smoke-entry-end",
        default="11:00",
        help="Latest New York time allowed for a new smoke-mode entry.",
    )
    parser.add_argument(
        "--smoke-force-close-minutes",
        type=int,
        default=45,
        help="Minutes to hold a smoke-mode paper position before forcing an exit on the next completed bar.",
    )
    parser.add_argument(
        "--sync-on-start",
        action="store_true",
        help="Restore the persisted paper-runner recovery state and reconcile it against the live IB account instead of resetting flat.",
    )
    parser.add_argument("--once", action="store_true", help="Run a single polling pass and exit.")
    parser.add_argument("--check", action="store_true", help="Validate connectivity, compute the current snapshot, and exit.")
    parser.add_argument(
        "--discord-summary",
        action="store_true",
        help="Send the generated paper test summary text to DISCORD_WEBHOOK_URL with change-aware throttling.",
    )
    parser.add_argument(
        "--discord-summary-min-interval-minutes",
        type=int,
        default=30,
        help="Minimum minutes between repeated Discord summary posts when the summary signature has not changed.",
    )
    return parser.parse_args(argv)


def build_configs(args: argparse.Namespace) -> tuple[CorridorConfig, PaperRunnerConfig]:
    parsed_layer_dtes = tuple(
        max(1, int(item.strip()))
        for item in str(args.layer_dte_targets or "").split(",")
        if item.strip()
    )
    max_layers = max(1, int(args.max_layers), len(parsed_layer_dtes))
    corridor_cfg = CorridorConfig(
        symbol=args.symbol.upper(),
        center_method=CenterMethod(args.center_method),
        center_rounding=default_center_rounding_for_symbol(args.symbol.upper()),
        body_strike_offset_points=float(args.body_strike_offset_points),
        payoff_mode="underlying_only",
        ib_client_id=args.client_id,
        butterfly_width=max(1.0, float(args.butterfly_width)),
        wing_mode=str(args.wing_mode),
        broken_wing_extra_width=max(0.0, float(args.broken_wing_extra_width)),
        option_right_preference=str(args.option_right),
        coverage_band_width=max(2.0, float(args.coverage_band_width)),
        center_tolerance=max(0.5, float(args.center_tolerance)),
        center_tolerance_atr_multiplier=max(0.0, float(args.center_tolerance_atr_multiplier)),
        atr_lookback=max(2, int(args.atr_lookback)),
        recenter_threshold=max(0.5, float(args.recenter_threshold)),
        drift_persistence_bars=max(1, int(args.drift_persistence_bars)),
        rebuild_cooldown_minutes=max(0, int(args.rebuild_cooldown_minutes)),
        max_active_butterfly_layers=max_layers,
        candidate_body_search_steps=max(0, int(args.candidate_body_search_steps)),
        dte_min=max(1, int(args.dte_min)),
        dte_max=max(max(1, int(args.dte_min)), int(args.dte_max)),
        default_dte=max(1, int(args.default_dte)),
        layer_dte_targets=parsed_layer_dtes,
        layer_exit_scope=str(args.layer_exit_scope),
        allow_daily_entry_additions=bool(args.allow_daily_entry_additions),
        max_acceptable_option_spread=max(0.01, float(args.max_option_spread)),
        near_spread_dte_max=max(0, int(args.near_spread_dte_max)),
        near_max_acceptable_option_spread=max(0.0, float(args.near_max_option_spread)),
        mid_max_acceptable_option_spread=max(0.0, float(args.mid_max_option_spread)),
        far_spread_dte_min=max(0, int(args.far_spread_dte_min)),
        far_max_acceptable_option_spread=max(0.0, float(args.far_max_option_spread)),
        primary_entry_end=str(args.primary_entry_end),
        primary_entry_min_center_confidence=max(0.0, min(1.0, float(args.primary_entry_min_center_confidence))),
        primary_entry_max_momentum_pct=max(0.0, float(args.primary_entry_max_momentum_pct)),
        primary_entry_max_volume_ratio=max(0.0, float(args.primary_entry_max_volume_ratio)),
        skip_entry_weekdays=tuple(
            item.strip().lower()
            for item in str(args.skip_entry_weekdays or "").split(",")
            if item.strip()
        ),
        primary_stop_loss_pct=max(0.0, float(args.primary_stop_loss_pct)),
        primary_take_profit_pct=max(0.0, float(args.primary_take_profit_pct)),
        block_same_day_reentry_after_take_profit=bool(args.block_same_day_reentry_after_take_profit),
        hold_overnight=bool(args.hold_overnight),
        max_hold_sessions=max(0, int(args.max_hold_sessions)),
        close_when_dte_lte=max(0, int(args.close_when_dte_lte)),
        skip_event_days=bool(args.skip_event_days),
        event_dates=tuple(
            item.strip()
            for item in str(args.event_dates or "").split(",")
            if item.strip()
        ),
        skip_gap_days=bool(args.skip_gap_days),
        max_entry_gap_pct=max(0.0, float(args.max_entry_gap_pct)),
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
        paper_smoke_mode=bool(args.paper_smoke_mode),
        smoke_max_entries_per_day=max(1, int(args.smoke_max_entries_per_day)),
        smoke_quantity=max(1, int(args.smoke_quantity)),
        smoke_entry_end=str(args.smoke_entry_end),
        smoke_force_close_minutes=max(1, int(args.smoke_force_close_minutes)),
        discord_summary=bool(args.discord_summary),
        discord_summary_min_interval_minutes=max(1, int(args.discord_summary_min_interval_minutes)),
        output_dir=output_dir,
        max_spread_pct_of_debit=max(0.05, float(args.max_spread_pct_of_debit)),
        combo_fill_wait_seconds=max(0.2, float(args.combo_fill_wait_seconds)),
        combo_max_chase_steps=max(1, int(args.combo_chase_steps)),
        combo_chase_fraction_of_spread=max(0.01, float(args.combo_chase_spread_fraction)),
        combo_max_total_debit_ratio=max(1.0, float(args.combo_max_total_debit_ratio)),
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
