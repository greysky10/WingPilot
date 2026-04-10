from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from corridor.backtest.engine import CorridorBacktestEngine
from corridor.config import CorridorConfig
from corridor.data.historical_loader import HistoricalLoadConfig, load_intraday_bars
from corridor.data.ib_contracts import default_center_rounding_for_symbol
from corridor.models import CenterMethod


_FRAME: Optional[pd.DataFrame] = None


def _coerce_utc_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _rounded_broken_extra(width: float) -> float:
    return max(5.0, round((width * 0.2) / 5.0) * 5.0)


def _load_frame(frame_path: str, symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    return load_intraday_bars(
        HistoricalLoadConfig(
            csv_path=Path(frame_path),
            symbol=symbol.upper(),
            start=_coerce_utc_timestamp(start),
            end=_coerce_utc_timestamp(end),
        )
    )


def _init_worker(frame_path: str, symbol: str, start: Optional[str], end: Optional[str]) -> None:
    global _FRAME
    _FRAME = _load_frame(frame_path, symbol, start, end)


def _base_config(symbol: str, output_dir: Path) -> CorridorConfig:
    return CorridorConfig(
        symbol=symbol.upper(),
        timeframe="5 mins",
        center_method=CenterMethod.VWAP,
        center_rounding=default_center_rounding_for_symbol(symbol.upper()),
        butterfly_width=100.0,
        wing_mode="broken_upper",
        broken_wing_extra_width=20.0,
        coverage_band_width=200.0,
        center_tolerance=15.0,
        center_tolerance_atr_multiplier=1.0,
        atr_lookback=14,
        recenter_threshold=16.0,
        drift_persistence_bars=8,
        rebuild_cooldown_minutes=60,
        max_active_butterfly_layers=1,
        primary_entry_end="15:00",
        primary_entry_min_center_confidence=0.60,
        primary_entry_max_momentum_pct=0.0010,
        primary_entry_max_volume_ratio=1.15,
        primary_stop_loss_pct=0.25,
        primary_take_profit_pct=0.20,
        hold_overnight=True,
        max_hold_sessions=3,
        close_when_dte_lte=1,
        default_dte=7,
        max_acceptable_option_spread=1.35,
        per_contract_slippage=0.05,
        payoff_mode="synthetic_chain",
        synthetic_chain_state_path="corridor_outputs/paper_runner/SPX/paper_state.json",
        synthetic_chain_report_path="corridor_outputs/paper_runner/SPX/paper_daily_report.json",
        output_dir=output_dir,
    )


def _stage1_cases(symbol: str) -> list[dict[str, Any]]:
    widths = [25.0, 50.0, 75.0, 100.0]
    dtes = [3, 5, 7]
    wing_modes = ["symmetric", "broken_upper", "broken_lower"]
    cases: list[dict[str, Any]] = []
    case_id = 1
    for wing_mode in wing_modes:
        for width in widths:
            for default_dte in dtes:
                extra_width = 0.0 if wing_mode == "symmetric" else _rounded_broken_extra(width)
                cases.append(
                    {
                        "case_id": f"s1_{case_id:03d}",
                        "stage": "stage1",
                        "symbol": symbol.upper(),
                        "wing_mode": wing_mode,
                        "butterfly_width": width,
                        "broken_wing_extra_width": extra_width,
                        "default_dte": default_dte,
                        "max_option_spread": 5.0,
                    }
                )
                case_id += 1
    return cases


def _stage2_cases(symbol: str, stage1_rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    ranked = sorted(
        stage1_rows,
        key=lambda row: (
            float(row.get("net_dollar_pnl", float("-inf"))),
            float(row.get("gross_dollar_pnl", float("-inf"))),
            -int(row.get("execution_filtered_entries", 0)),
        ),
        reverse=True,
    )
    selected = ranked[:top_n]
    spread_caps = [1.35, 1.5, 2.0, 3.0, 5.0]
    cases: list[dict[str, Any]] = []
    case_id = 1
    for row in selected:
        for spread_cap in spread_caps:
            cases.append(
                {
                    "case_id": f"s2_{case_id:03d}",
                    "stage": "stage2",
                    "symbol": symbol.upper(),
                    "stage1_source_case_id": row["case_id"],
                    "wing_mode": row["wing_mode"],
                    "butterfly_width": float(row["butterfly_width"]),
                    "broken_wing_extra_width": float(row["broken_wing_extra_width"]),
                    "default_dte": int(row["default_dte"]),
                    "max_option_spread": float(spread_cap),
                }
            )
            case_id += 1
    return cases


def _run_case(case: dict[str, Any], output_dir: str) -> dict[str, Any]:
    global _FRAME
    if _FRAME is None:
        raise RuntimeError("Worker frame was not initialized.")

    cfg = _base_config(case["symbol"], Path(output_dir))
    cfg.wing_mode = str(case["wing_mode"])
    cfg.butterfly_width = float(case["butterfly_width"])
    cfg.broken_wing_extra_width = float(case["broken_wing_extra_width"])
    cfg.default_dte = int(case["default_dte"])
    cfg.max_acceptable_option_spread = float(case["max_option_spread"])

    result = CorridorBacktestEngine(cfg).run(_FRAME)
    summary = result.summary
    return {
        **case,
        "net_dollar_pnl": float(summary.get("net_dollar_pnl", 0.0) or 0.0),
        "gross_dollar_pnl": float(summary.get("gross_dollar_pnl", 0.0) or 0.0),
        "friction_adjustment_dollars": float(summary.get("friction_adjustment_dollars", 0.0) or 0.0),
        "closed_layers": int(summary.get("closed_layers", 0) or 0),
        "winning_layers": int(summary.get("winning_layers", 0) or 0),
        "losing_layers": int(summary.get("losing_layers", 0) or 0),
        "execution_filtered_entries": int(summary.get("execution_filtered_entries", 0) or 0),
        "win_rate_by_closed_layer": summary.get("win_rate_by_closed_layer"),
        "profit_factor_by_closed_layer": summary.get("profit_factor_by_closed_layer"),
        "return_on_capital": summary.get("return_on_capital"),
        "max_drawdown": float(summary.get("max_drawdown", 0.0) or 0.0),
        "max_gross_deployment_dollars": float(summary.get("max_gross_deployment_dollars", 0.0) or 0.0),
        "max_modeled_capital_at_risk_dollars": float(summary.get("max_modeled_capital_at_risk_dollars", 0.0) or 0.0),
        "synthetic_chain_state_path": str(summary.get("synthetic_chain_state_path", "")),
        "synthetic_chain_report_path": str(summary.get("synthetic_chain_report_path", "")),
    }


def _run_cases(
    cases: list[dict[str, Any]],
    frame_path: Path,
    symbol: str,
    start: Optional[str],
    end: Optional[str],
    output_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    if not cases:
        return []

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(frame_path), symbol, start, end),
    ) as executor:
        future_map = {
            executor.submit(_run_case, case, str(output_dir)): case
            for case in cases
        }
        for future in as_completed(future_map):
            case = future_map[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # pragma: no cover - defensive path for long searches
                rows.append(
                    {
                        **case,
                        "error": str(exc),
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep synthetic-chain backtests across a coarse parameter grid.")
    parser.add_argument("--symbol", default="SPX")
    parser.add_argument("--bars-csv", required=True)
    parser.add_argument("--start", default="2025-04-09")
    parser.add_argument("--end", default="2026-04-09")
    parser.add_argument("--output-dir", default="corridor_outputs/fit_search/synthetic_chain_sweep")
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--top-n", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_cases = _stage1_cases(args.symbol)
    stage1_rows = _run_cases(
        stage1_cases,
        frame_path=Path(args.bars_csv),
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        output_dir=output_dir,
        workers=max(1, int(args.workers)),
    )
    stage1_frame = pd.DataFrame(stage1_rows).sort_values(
        by=["net_dollar_pnl", "gross_dollar_pnl", "closed_layers"],
        ascending=[False, False, False],
    )
    stage1_path = output_dir / "stage1.csv"
    stage1_frame.to_csv(stage1_path, index=False)

    valid_stage1 = [row for row in stage1_rows if "error" not in row]
    stage2_cases = _stage2_cases(args.symbol, valid_stage1, top_n=max(1, int(args.top_n)))
    stage2_rows = _run_cases(
        stage2_cases,
        frame_path=Path(args.bars_csv),
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        output_dir=output_dir,
        workers=max(1, int(args.workers)),
    )
    stage2_frame = pd.DataFrame(stage2_rows).sort_values(
        by=["net_dollar_pnl", "gross_dollar_pnl", "closed_layers"],
        ascending=[False, False, False],
    )
    stage2_path = output_dir / "stage2.csv"
    stage2_frame.to_csv(stage2_path, index=False)

    combined = pd.concat([stage1_frame, stage2_frame], ignore_index=True, sort=False)
    combined = combined.sort_values(
        by=["net_dollar_pnl", "gross_dollar_pnl", "closed_layers"],
        ascending=[False, False, False],
    )
    combined_path = output_dir / "combined.csv"
    combined.to_csv(combined_path, index=False)

    top10_path = output_dir / "top10.json"
    top10_path.write_text(
        json.dumps(combined.head(10).to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    summary = {
        "stage1_cases": len(stage1_cases),
        "stage2_cases": len(stage2_cases),
        "workers": int(args.workers),
        "stage1_path": str(stage1_path),
        "stage2_path": str(stage2_path),
        "combined_path": str(combined_path),
        "top10_path": str(top10_path),
        "best_case": combined.head(1).to_dict(orient="records"),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Stage 1 complete: {stage1_path}")
    print(f"Stage 2 complete: {stage2_path}")
    print(f"Combined results: {combined_path}")
    print(f"Top 10: {top10_path}")
    if not combined.empty:
        best = combined.iloc[0]
        print(
            "Best | "
            f"stage={best.get('stage')} | "
            f"wing={best.get('wing_mode')} | "
            f"width={best.get('butterfly_width')} | "
            f"extra={best.get('broken_wing_extra_width')} | "
            f"dte={best.get('default_dte')} | "
            f"spread_cap={best.get('max_option_spread')} | "
            f"net_dollar_pnl={best.get('net_dollar_pnl')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
