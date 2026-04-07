#! python3.12
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from corridor.backtest.engine import CorridorBacktestEngine
from corridor.config import CorridorConfig
from corridor.data.ib_contracts import default_center_rounding_for_symbol
from corridor.models import CenterMethod
from corridor.report.summary import save_backtest_outputs
from strategy import load_local_env


FRAME: pd.DataFrame | None = None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare symmetric vs broken-wing daily SPX corridor variants on cached bars "
            "using the conservative stressed simplified pricer."
        )
    )
    parser.add_argument(
        "--bars-csv",
        default=r".\corridor_outputs\spx_grid_center_tol\SPX_5_mins_bars.csv",
        help="Cached normalized SPX bars CSV.",
    )
    parser.add_argument(
        "--output-root",
        default=r".\corridor_outputs\spx_broken_wing_compare",
        help="Root directory for broken-wing comparison outputs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(6, (os.cpu_count() or 4) - 1)),
        help="Parallel worker count.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap for smoke testing. 0 means full grid.",
    )
    return parser.parse_args(argv)


def base_config() -> CorridorConfig:
    return CorridorConfig(
        symbol="SPX",
        timeframe="5 mins",
        center_method=CenterMethod.VWAP,
        center_rounding=default_center_rounding_for_symbol("SPX"),
        payoff_mode="simplified",
        starting_capital=100000.0,
        contracts_per_layer=1,
        option_multiplier=100,
        stress_profile="conservative",
        stress_entry_debit_multiplier=1.2,
        stress_peak_value_multiplier=0.7,
        stress_residual_floor_multiplier=0.5,
        stress_slippage_multiplier=2.0,
        stress_close_value_haircut_pct=0.15,
        butterfly_width=80.0,
        wing_mode="symmetric",
        broken_wing_extra_width=0.0,
        coverage_band_width=160.0,
        center_tolerance=12.5,
        recenter_threshold=16.0,
        drift_persistence_bars=8,
        rebuild_cooldown_minutes=60,
        max_active_butterfly_layers=2,
        primary_entry_end="13:30",
        primary_entry_min_center_confidence=0.60,
        primary_entry_max_momentum_pct=0.0010,
        primary_entry_max_volume_ratio=1.15,
        primary_stop_loss_pct=0.25,
        primary_take_profit_pct=0.20,
    )


def compare_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    geometries = [
        ("symmetric", 0.0),
        ("broken_upper", 20.0),
        ("broken_upper", 40.0),
        ("broken_lower", 20.0),
        ("broken_lower", 40.0),
    ]
    for width in (60.0, 80.0, 100.0):
        for tolerance in (10.0, 12.5, 15.0):
            for recenter in (14.0, 16.0, 18.0):
                for drift in (8, 10):
                    for cooldown in (60, 90):
                        for wing_mode, broken_extra in geometries:
                            rows.append(
                                {
                                    "wing_mode": wing_mode,
                                    "broken_wing_extra_width": broken_extra,
                                    "butterfly_width": width,
                                    "coverage_band_width": width * 2.0,
                                    "center_tolerance": tolerance,
                                    "recenter_threshold": recenter,
                                    "drift_persistence_bars": drift,
                                    "rebuild_cooldown_minutes": cooldown,
                                    "max_active_butterfly_layers": 2,
                                    "primary_stop_loss_pct": 0.25,
                                    "primary_take_profit_pct": 0.20,
                                }
                            )
    return rows


def config_name(params: dict[str, Any]) -> str:
    def fmt(value: float | int) -> str:
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return str(value).replace(".", "p")

    return (
        f"{params['wing_mode']}"
        f"_bw{fmt(params['broken_wing_extra_width'])}"
        f"_w{fmt(params['butterfly_width'])}"
        f"_tol{fmt(params['center_tolerance'])}"
        f"_rt{fmt(params['recenter_threshold'])}"
        f"_dp{fmt(params['drift_persistence_bars'])}"
        f"_cd{fmt(params['rebuild_cooldown_minutes'])}"
    )


def _init_worker(csv_path: str) -> None:
    global FRAME
    frame = pd.read_csv(csv_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    FRAME = frame.sort_values("timestamp").reset_index(drop=True)


def _summary_row(name: str, params: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    multiplier = float(summary.get("option_multiplier") or 100)
    contracts = float(summary.get("contracts_per_layer") or 1)
    max_drawdown_dollars = float(summary.get("max_drawdown") or 0.0) * multiplier * contracts
    return {
        "config_name": name,
        "wing_mode": params["wing_mode"],
        "broken_wing_extra_width": params["broken_wing_extra_width"],
        "butterfly_width": params["butterfly_width"],
        "coverage_band_width": params["coverage_band_width"],
        "center_tolerance": params["center_tolerance"],
        "recenter_threshold": params["recenter_threshold"],
        "drift_persistence_bars": params["drift_persistence_bars"],
        "rebuild_cooldown_minutes": params["rebuild_cooldown_minutes"],
        "net_dollar_pnl": summary["net_dollar_pnl"],
        "return_on_capital": summary["return_on_capital"],
        "max_drawdown_dollars": max_drawdown_dollars,
        "corridor_occupancy_rate": summary["corridor_occupancy_rate"],
        "average_rebuilds_per_day": summary["average_rebuilds_per_day"],
        "closed_layers": summary["closed_layers"],
        "win_rate_by_closed_layer": summary["win_rate_by_closed_layer"],
        "profit_factor_by_closed_layer": summary["profit_factor_by_closed_layer"],
        "profit_factor_by_day": summary["profit_factor_by_day"],
        "best_day_pnl_dollars": summary["best_day_pnl_dollars"],
        "worst_day_pnl_dollars": summary["worst_day_pnl_dollars"],
        "max_gross_deployment_dollars": summary["max_gross_deployment_dollars"],
        "max_modeled_capital_at_risk_dollars": summary["max_modeled_capital_at_risk_dollars"],
    }


def _run_one(task: tuple[str, dict[str, Any], str]) -> dict[str, Any]:
    global FRAME
    name, params, output_root = task
    if FRAME is None:
        raise RuntimeError("Worker frame is not initialized.")

    run_dir = Path(output_root) / name
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return _summary_row(name, params, summary)

    cfg = base_config()
    cfg.output_dir = run_dir
    cfg.wing_mode = str(params["wing_mode"])
    cfg.broken_wing_extra_width = float(params["broken_wing_extra_width"])
    cfg.butterfly_width = float(params["butterfly_width"])
    cfg.coverage_band_width = float(params["coverage_band_width"])
    cfg.center_tolerance = float(params["center_tolerance"])
    cfg.recenter_threshold = float(params["recenter_threshold"])
    cfg.drift_persistence_bars = int(params["drift_persistence_bars"])
    cfg.rebuild_cooldown_minutes = int(params["rebuild_cooldown_minutes"])
    cfg.max_active_butterfly_layers = int(params["max_active_butterfly_layers"])
    cfg.primary_stop_loss_pct = float(params["primary_stop_loss_pct"])
    cfg.primary_take_profit_pct = float(params["primary_take_profit_pct"])

    result = CorridorBacktestEngine(cfg).run(FRAME)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_backtest_outputs(run_dir, result)
    config_path.write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
    return _summary_row(name, params, result.summary)


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        by=[
            "net_dollar_pnl",
            "profit_factor_by_closed_layer",
            "return_on_capital",
            "max_drawdown_dollars",
            "corridor_occupancy_rate",
            "average_rebuilds_per_day",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)


def _fmt_optional(value: Any, style: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if style == "pct":
        return f"{float(value):.2%}"
    return f"{float(value):.4f}"


def run_tasks(
    tasks: list[tuple[str, dict[str, Any], str]],
    summary_csv: Path,
    workers: int,
    bars_csv: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_init_worker, initargs=(bars_csv,)) as pool:
        future_map = {pool.submit(_run_one, task): task[0] for task in tasks}
        for future in as_completed(future_map):
            row = future.result()
            rows.append(row)
            frame = _sort_frame(pd.DataFrame(rows))
            frame.to_csv(summary_csv, index=False)
            print(
                f"[broken] {row['config_name']} | pnl={float(row['net_dollar_pnl']):.2f} | "
                f"pf={_fmt_optional(row['profit_factor_by_closed_layer'], 'num')} | "
                f"roc={_fmt_optional(row['return_on_capital'], 'pct')} | "
                f"dd={float(row['max_drawdown_dollars']):.2f}"
            )
    frame = _sort_frame(pd.DataFrame(rows))
    frame.to_csv(summary_csv, index=False)
    return frame


def save_top_detailed(top_rows: pd.DataFrame, bars_csv: str, output_root: Path, top_n: int = 8) -> None:
    frame = pd.read_csv(bars_csv)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    for _, row in top_rows.head(top_n).iterrows():
        detail_dir = output_root / "detailed" / str(row["config_name"])
        if (detail_dir / "summary.json").exists():
            continue
        cfg = base_config()
        cfg.output_dir = detail_dir
        cfg.wing_mode = str(row["wing_mode"])
        cfg.broken_wing_extra_width = float(row["broken_wing_extra_width"])
        cfg.butterfly_width = float(row["butterfly_width"])
        cfg.coverage_band_width = float(row["coverage_band_width"])
        cfg.center_tolerance = float(row["center_tolerance"])
        cfg.recenter_threshold = float(row["recenter_threshold"])
        cfg.drift_persistence_bars = int(row["drift_persistence_bars"])
        cfg.rebuild_cooldown_minutes = int(row["rebuild_cooldown_minutes"])
        result = CorridorBacktestEngine(cfg).run(frame)
        save_backtest_outputs(detail_dir, result)
        (detail_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    load_local_env()
    args = parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    bars_csv = str(Path(args.bars_csv).resolve())
    if not Path(bars_csv).exists():
        raise FileNotFoundError(f"Bars CSV not found: {bars_csv}")

    params = compare_grid()
    if args.limit > 0:
        params = params[: args.limit]
    tasks = [(config_name(row), row, str(output_root / "runs")) for row in params]
    summary_csv = output_root / "spx_broken_wing_compare_summary.csv"
    frame = run_tasks(tasks, summary_csv, args.workers, bars_csv)
    grouped = (
        frame.groupby(["wing_mode", "broken_wing_extra_width"], dropna=False)[
            [
                "net_dollar_pnl",
                "profit_factor_by_closed_layer",
                "return_on_capital",
                "max_drawdown_dollars",
                "corridor_occupancy_rate",
                "average_rebuilds_per_day",
            ]
        ]
        .mean()
        .reset_index()
    )
    grouped = grouped.sort_values(
        by=["net_dollar_pnl", "profit_factor_by_closed_layer", "return_on_capital"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    grouped_csv = output_root / "spx_broken_wing_geometry_summary.csv"
    grouped.to_csv(grouped_csv, index=False)

    save_top_detailed(frame, bars_csv, output_root, top_n=8)

    if not frame.empty:
        top = frame.iloc[0]
        print(
            "[broken] best | "
            f"{top['config_name']} | pnl={float(top['net_dollar_pnl']):.2f} | "
            f"pf={_fmt_optional(top['profit_factor_by_closed_layer'], 'num')} | "
            f"roc={_fmt_optional(top['return_on_capital'], 'pct')} | "
            f"dd={float(top['max_drawdown_dollars']):.2f}"
        )
    print(f"[broken] wrote {summary_csv}")
    print(f"[broken] wrote {grouped_csv}")
    print(f"[broken] runs={len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
