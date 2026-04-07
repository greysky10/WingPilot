#! python3.12
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
from typing import Optional

import pandas as pd

from corridor.config import CorridorConfig
from corridor.data.ib_contracts import default_center_rounding_for_symbol
from corridor.data.historical_loader import HistoricalLoadConfig, load_intraday_bars
from corridor.data.ib_loader import IBHistoricalRequest, fetch_intraday_bars
from corridor.models import CenterMethod
from corridor.backtest.engine import CorridorBacktestEngine
from corridor.report.plots import save_equity_plot
from corridor.report.summary import save_backtest_outputs
from strategy import load_local_env


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a dynamic corridor backtest on intraday bars. Modeled PnL units are reported separately from capital-normalized returns."
    )
    parser.add_argument("--symbol", default="SPX", help="Ticker symbol.")
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
    parser.add_argument("--per-contract-slippage", type=float, default=0.05, help="Modeled per-contract slippage in option points for each open/close action.")
    parser.add_argument("--max-option-spread", type=float, default=0.25, help="Absolute combo-spread cap in option points, aligned with the paper runner filter.")
    parser.add_argument("--butterfly-width", type=float, default=10.0, help="Butterfly wing width in strike points.")
    parser.add_argument(
        "--wing-mode",
        default="symmetric",
        choices=["symmetric", "broken_upper", "broken_lower"],
        help="Strike geometry mode for the modeled butterfly.",
    )
    parser.add_argument(
        "--broken-wing-extra-width",
        type=float,
        default=0.0,
        help="Extra width added to the broken side when wing-mode is asymmetric.",
    )
    parser.add_argument("--coverage-band-width", type=float, default=20.0, help="Total corridor coverage width in strike points.")
    parser.add_argument("--center-tolerance", type=float, default=2.5, help="Minimum half-width of the center occupancy / drift tolerance band.")
    parser.add_argument("--center-tolerance-atr-multiplier", type=float, default=1.0, help="Dynamic tolerance multiplier: actual_tolerance = max(center_tolerance, ATR * multiplier).")
    parser.add_argument("--atr-lookback", type=int, default=14, help="ATR lookback bars used for dynamic center tolerance.")
    parser.add_argument("--recenter-threshold", type=float, default=3.5, help="Minimum drift distance before a rebuild is allowed.")
    parser.add_argument("--drift-persistence-bars", type=int, default=2, help="Bars required outside tolerance before rebuilding.")
    parser.add_argument("--rebuild-cooldown-minutes", type=int, default=15, help="Cooldown after a rebuild before another rebuild is allowed.")
    parser.add_argument("--max-layers", type=int, default=3, help="Maximum total active butterflies, including the primary layer.")
    parser.add_argument("--primary-entry-end", default="15:30", help="Latest New York time allowed for a new primary entry.")
    parser.add_argument("--primary-entry-min-center-confidence", type=float, default=0.0, help="Minimum center confidence required for a new primary entry.")
    parser.add_argument("--primary-entry-max-momentum-pct", type=float, default=1.0, help="Maximum absolute momentum_pct allowed for a new primary entry.")
    parser.add_argument("--primary-entry-max-volume-ratio", type=float, default=999.0, help="Maximum volume_ratio allowed for a new primary entry.")
    parser.add_argument("--skip-event-days", action="store_true", help="Block new primary entries on configured event dates.")
    parser.add_argument("--event-dates", default="", help="Comma-separated New York dates to block, for example 2026-04-10,2026-05-06.")
    parser.add_argument("--primary-stop-loss-pct", type=float, default=0.0, help="Protective stop-loss threshold for the primary layer, measured as a fraction of entry cost.")
    parser.add_argument("--primary-take-profit-pct", type=float, default=0.0, help="Take-profit threshold for the primary layer, measured as a fraction of entry cost.")
    parser.add_argument(
        "--stress-profile",
        default="none",
        choices=["none", "conservative"],
        help="Optional stress profile for the simplified butterfly pricer.",
    )
    parser.add_argument(
        "--paper-diagnostics-json",
        default="",
        help="Optional paper runner JSON (paper_state.json, paper_daily_report.json, or paper_test_summary.json) used to calibrate a spread-based execution gate in backtests.",
    )
    parser.add_argument(
        "--paper-diagnostics-mode",
        default="tax",
        choices=["tax", "hard_reject"],
        help="How to apply paper diagnostics in the backtest: add spread tax or hard-reject entries.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> CorridorConfig:
    cfg = CorridorConfig(
        symbol=args.symbol.upper(),
        timeframe=args.timeframe,
        center_lookback=max(10, args.center_lookback),
        center_method=CenterMethod(args.center_method),
        center_rounding=default_center_rounding_for_symbol(args.symbol.upper()),
        butterfly_width=max(1.0, float(args.butterfly_width)),
        wing_mode=str(args.wing_mode),
        broken_wing_extra_width=max(0.0, float(args.broken_wing_extra_width)),
        coverage_band_width=max(2.0, float(args.coverage_band_width)),
        center_tolerance=max(0.5, float(args.center_tolerance)),
        center_tolerance_atr_multiplier=max(0.0, float(args.center_tolerance_atr_multiplier)),
        atr_lookback=max(2, int(args.atr_lookback)),
        recenter_threshold=max(0.5, float(args.recenter_threshold)),
        drift_persistence_bars=max(1, int(args.drift_persistence_bars)),
        rebuild_cooldown_minutes=max(0, int(args.rebuild_cooldown_minutes)),
        max_active_butterfly_layers=max(1, int(args.max_layers)),
        primary_entry_end=args.primary_entry_end,
        primary_entry_min_center_confidence=max(0.0, min(1.0, float(args.primary_entry_min_center_confidence))),
        primary_entry_max_momentum_pct=max(0.0, float(args.primary_entry_max_momentum_pct)),
        primary_entry_max_volume_ratio=max(0.0, float(args.primary_entry_max_volume_ratio)),
        skip_event_days=bool(args.skip_event_days),
        event_dates=tuple(
            item.strip()
            for item in str(args.event_dates or "").split(",")
            if item.strip()
        ),
        primary_stop_loss_pct=max(0.0, float(args.primary_stop_loss_pct)),
        primary_take_profit_pct=max(0.0, float(args.primary_take_profit_pct)),
        payoff_mode="simplified" if args.payoff_mode == "simplified" else "underlying_only",
        ib_host=os.getenv("IB_HOST", "127.0.0.1"),
        ib_port=int(os.getenv("IB_PORT", "4001")),
        ib_client_id=args.client_id,
        starting_capital=max(0.0, float(args.starting_capital)),
        contracts_per_layer=max(1, int(args.contracts_per_layer)),
        option_multiplier=max(1, int(args.option_multiplier)),
        max_acceptable_option_spread=max(0.0, float(args.max_option_spread)),
        per_contract_slippage=max(0.0, float(args.per_contract_slippage)),
        slippage=max(0.0, float(args.per_contract_slippage)),
        stress_profile=args.stress_profile,
    )
    _apply_stress_profile(cfg, args.stress_profile)
    if args.paper_diagnostics_json:
        _apply_paper_spread_gate(cfg, Path(args.paper_diagnostics_json), str(args.paper_diagnostics_mode))
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    else:
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        cfg.output_dir = Path("corridor_outputs") / f"{cfg.symbol}_{stamp}"
    return cfg


def _apply_stress_profile(cfg: CorridorConfig, stress_profile: str) -> None:
    if stress_profile == "conservative":
        cfg.stress_entry_debit_multiplier = 1.2
        cfg.stress_peak_value_multiplier = 0.7
        cfg.stress_residual_floor_multiplier = 0.5
        cfg.stress_slippage_multiplier = 2.0
        cfg.stress_close_value_haircut_pct = 0.15
        return

    cfg.stress_entry_debit_multiplier = 1.0
    cfg.stress_peak_value_multiplier = 1.0
    cfg.stress_residual_floor_multiplier = 1.0
    cfg.stress_slippage_multiplier = 1.0
    cfg.stress_close_value_haircut_pct = 0.0


def _apply_paper_spread_gate(cfg: CorridorConfig, diagnostics_path: Path, mode: str) -> None:
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    candidate_diagnostics = _extract_candidate_diagnostics(payload)
    if not candidate_diagnostics and diagnostics_path.name.endswith("_test_summary.json"):
        parent = diagnostics_path.parent
        for fallback_name in ["paper_daily_report.json", "paper_state.json"]:
            fallback_path = parent / fallback_name
            if not fallback_path.exists():
                continue
            fallback_payload = json.loads(fallback_path.read_text(encoding="utf-8"))
            candidate_diagnostics = _extract_candidate_diagnostics(fallback_payload)
            if candidate_diagnostics:
                diagnostics_path = fallback_path
                break
    rejection_counts = candidate_diagnostics.get("rejection_counts", {}) if isinstance(candidate_diagnostics, dict) else {}
    samples = candidate_diagnostics.get("sample_rejections", []) if isinstance(candidate_diagnostics, dict) else []
    spread_samples = [
        sample
        for sample in samples
        if str(sample.get("reason", "")) == "spread_too_wide"
        and sample.get("spread_ratio") not in (None, "")
        and sample.get("total_spread") not in (None, "")
    ]
    if not spread_samples:
        return

    cfg.paper_spread_gate_enabled = True
    cfg.paper_spread_gate_mode = str(mode)
    cfg.paper_spread_gate_source = str(diagnostics_path)
    cfg.paper_spread_gate_spread_ratio = float(median(float(sample["spread_ratio"]) for sample in spread_samples))
    cfg.paper_spread_gate_total_spread = float(median(float(sample["total_spread"]) for sample in spread_samples))
    cfg.paper_spread_gate_sample_count = len(spread_samples)
    cfg.paper_spread_gate_rejection_count = int(rejection_counts.get("spread_too_wide", len(spread_samples)))


def _extract_candidate_diagnostics(payload: dict) -> dict:
    if isinstance(payload.get("candidate_diagnostics"), dict):
        return payload["candidate_diagnostics"]
    if isinstance(payload.get("rejection_counts"), dict):
        return payload
    return {}


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
    load_local_env()
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
        "Stress | "
        f"profile={cfg.stress_profile} | "
        f"entry_mult={cfg.stress_entry_debit_multiplier:.2f} | "
        f"peak_mult={cfg.stress_peak_value_multiplier:.2f} | "
        f"residual_mult={cfg.stress_residual_floor_multiplier:.2f} | "
        f"slippage_mult={cfg.stress_slippage_multiplier:.2f} | "
        f"close_haircut={cfg.stress_close_value_haircut_pct:.0%}"
    )
    print(
        "Controls | "
        f"wing_mode={cfg.wing_mode} | "
        f"broken_extra={cfg.broken_wing_extra_width:.2f} | "
        f"max_option_spread={cfg.max_acceptable_option_spread:.2f} | "
        f"max_layers={cfg.max_active_butterfly_layers} | "
        f"primary_entry_end={cfg.primary_entry_end} | "
        f"atr_mult={cfg.center_tolerance_atr_multiplier:.2f} | "
        f"atr_lookback={cfg.atr_lookback} | "
        f"min_center_conf={cfg.primary_entry_min_center_confidence:.2f} | "
        f"max_entry_momentum={cfg.primary_entry_max_momentum_pct:.4f} | "
        f"max_entry_volume={cfg.primary_entry_max_volume_ratio:.2f} | "
        f"primary_stop={cfg.primary_stop_loss_pct:.0%} | "
        f"primary_take_profit={cfg.primary_take_profit_pct:.0%}"
    )
    if cfg.paper_spread_gate_enabled:
        print(
            "Execution Gate | "
            f"mode={cfg.paper_spread_gate_mode} | "
            f"source={cfg.paper_spread_gate_source} | "
            f"median_spread_ratio={cfg.paper_spread_gate_spread_ratio:.4f} | "
            f"median_total_spread={cfg.paper_spread_gate_total_spread:.4f} | "
            f"samples={cfg.paper_spread_gate_sample_count} | "
            f"spread_rejections={cfg.paper_spread_gate_rejection_count}"
        )
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
        f"per_contract_slippage={result.summary['per_contract_slippage']:.4f} | "
        f"net_dollar_pnl={result.summary['net_dollar_pnl']:.2f} | "
        f"gross_profit={result.summary['gross_profit']:.2f} | "
        f"net_slippage_adjusted_profit={result.summary['net_slippage_adjusted_profit']:.2f} | "
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
