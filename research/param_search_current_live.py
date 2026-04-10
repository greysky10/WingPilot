#! python3.12
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corridor.backtest.engine import CorridorBacktestEngine
from corridor.config import CorridorConfig
from corridor.data.ib_contracts import default_center_rounding_for_symbol
from corridor.models import CenterMethod


BARS_CSV = Path(r"corridor_outputs\fit_search\SPX_5_mins_bars_20250408_20260409.csv").resolve()
OUT_ROOT = Path(r"corridor_outputs\fit_search\param_search_current_live").resolve()
RUNS_DIR = OUT_ROOT / "runs"
FRAME: pd.DataFrame | None = None


def init_worker(csv_path: str) -> None:
    global FRAME
    frame = pd.read_csv(csv_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    FRAME = frame.sort_values("timestamp").reset_index(drop=True)


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


def structural_grid() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    geometries = [
        ("symmetric", 0.0),
        ("broken_upper", 20.0),
        ("broken_lower", 20.0),
    ]
    for wing_mode, broken_extra in geometries:
        for width in (60.0, 80.0, 100.0):
            for tolerance in (10.0, 12.5, 15.0):
                for recenter in (14.0, 16.0, 18.0):
                    rows.append(
                        {
                            "stage": "structural",
                            "wing_mode": wing_mode,
                            "broken_wing_extra_width": broken_extra,
                            "butterfly_width": width,
                            "coverage_band_width": width * 2.0,
                            "center_tolerance": tolerance,
                            "recenter_threshold": recenter,
                            "drift_persistence_bars": 8,
                            "rebuild_cooldown_minutes": 60,
                            "max_active_butterfly_layers": 2,
                            "primary_stop_loss_pct": 0.25,
                            "primary_take_profit_pct": 0.20,
                        }
                    )
    return rows


def refine_grid(top_rows: pd.DataFrame) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    seen: set[tuple[tuple[str, object], ...]] = set()
    for _, row in top_rows.iterrows():
        base = {
            "wing_mode": str(row["wing_mode"]),
            "broken_wing_extra_width": float(row["broken_wing_extra_width"]),
            "butterfly_width": float(row["butterfly_width"]),
            "coverage_band_width": float(row["coverage_band_width"]),
            "center_tolerance": float(row["center_tolerance"]),
            "recenter_threshold": float(row["recenter_threshold"]),
        }
        for drift in (8, 10):
            for cooldown in (60, 90):
                for max_layers in (1, 2):
                    for stop in (0.20, 0.25):
                        for take in (0.15, 0.20):
                            params = {
                                "stage": "refine",
                                **base,
                                "drift_persistence_bars": drift,
                                "rebuild_cooldown_minutes": cooldown,
                                "max_active_butterfly_layers": max_layers,
                                "primary_stop_loss_pct": stop,
                                "primary_take_profit_pct": take,
                            }
                            key = tuple(sorted(params.items()))
                            if key in seen:
                                continue
                            seen.add(key)
                            rows.append(params)
    return rows


def fmt(value: float | int | str) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(value).replace(".", "p")


def config_name(params: dict[str, float | int | str]) -> str:
    return (
        f"{params['stage']}_"
        f"{params['wing_mode']}_bw{fmt(params['broken_wing_extra_width'])}"
        f"_w{fmt(params['butterfly_width'])}"
        f"_tol{fmt(params['center_tolerance'])}"
        f"_rt{fmt(params['recenter_threshold'])}"
        f"_dp{fmt(params['drift_persistence_bars'])}"
        f"_cd{fmt(params['rebuild_cooldown_minutes'])}"
        f"_ml{fmt(params['max_active_butterfly_layers'])}"
        f"_sl{fmt(params['primary_stop_loss_pct'])}"
        f"_tp{fmt(params['primary_take_profit_pct'])}"
    )


def summary_row(name: str, params: dict[str, float | int | str], summary: dict[str, object]) -> dict[str, object]:
    multiplier = float(summary.get("option_multiplier") or 100)
    contracts = float(summary.get("contracts_per_layer") or 1)
    max_dd_dollars = float(summary.get("max_drawdown") or 0.0) * multiplier * contracts
    net_pnl = float(summary.get("net_dollar_pnl") or 0.0)
    occupancy = float(summary.get("corridor_occupancy_rate") or 0.0)
    return {
        "config_name": name,
        "stage": params["stage"],
        "wing_mode": params["wing_mode"],
        "broken_wing_extra_width": float(params["broken_wing_extra_width"]),
        "butterfly_width": float(params["butterfly_width"]),
        "coverage_band_width": float(params["coverage_band_width"]),
        "center_tolerance": float(params["center_tolerance"]),
        "recenter_threshold": float(params["recenter_threshold"]),
        "drift_persistence_bars": int(params["drift_persistence_bars"]),
        "rebuild_cooldown_minutes": int(params["rebuild_cooldown_minutes"]),
        "max_layers": int(params["max_active_butterfly_layers"]),
        "primary_stop_loss_pct": float(params["primary_stop_loss_pct"]),
        "primary_take_profit_pct": float(params["primary_take_profit_pct"]),
        "net_dollar_pnl": net_pnl,
        "return_on_capital": float(summary.get("return_on_capital") or 0.0),
        "max_drawdown_points": float(summary.get("max_drawdown") or 0.0),
        "max_drawdown_dollars": max_dd_dollars,
        "corridor_occupancy_rate": occupancy,
        "average_rebuilds_per_day": float(summary.get("average_rebuilds_per_day") or 0.0),
        "closed_layers": int(summary.get("closed_layers") or 0),
        "win_rate_by_closed_layer": summary.get("win_rate_by_closed_layer"),
        "profit_factor_by_closed_layer": summary.get("profit_factor_by_closed_layer"),
        "profit_factor_by_day": summary.get("profit_factor_by_day"),
        "best_day_pnl_dollars": float(summary.get("best_day_pnl_dollars") or 0.0),
        "worst_day_pnl_dollars": float(summary.get("worst_day_pnl_dollars") or 0.0),
        "max_gross_deployment_dollars": float(summary.get("max_gross_deployment_dollars") or 0.0),
        "score": net_pnl + (max_dd_dollars * 25.0) + (occupancy * 10000.0),
    }


def run_one(task: tuple[str, dict[str, float | int | str]]) -> dict[str, object]:
    global FRAME
    if FRAME is None:
        raise RuntimeError("Worker frame is not initialized.")

    name, params = task
    run_dir = RUNS_DIR / name
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return summary_row(name, params, summary)

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
    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
    return summary_row(name, params, result.summary)


def sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=[
            "score",
            "net_dollar_pnl",
            "return_on_capital",
            "corridor_occupancy_rate",
            "average_rebuilds_per_day",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def run_tasks(tasks: list[tuple[str, dict[str, float | int | str]]], csv_path: Path, workers: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(str(BARS_CSV),)) as pool:
        future_map = {pool.submit(run_one, task): task[0] for task in tasks}
        for idx, future in enumerate(as_completed(future_map), start=1):
            row = future.result()
            rows.append(row)
            df = sort_frame(pd.DataFrame(rows))
            df.to_csv(csv_path, index=False)
            print(
                f"[{idx}/{len(tasks)}] {row['config_name']} | "
                f"pnl={float(row['net_dollar_pnl']):.2f} | "
                f"score={float(row['score']):.2f} | "
                f"occ={float(row['corridor_occupancy_rate']):.2%}"
            )
    df = sort_frame(pd.DataFrame(rows))
    df.to_csv(csv_path, index=False)
    return df


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not BARS_CSV.exists():
        raise FileNotFoundError(f"Bars CSV not found: {BARS_CSV}")

    workers = max(1, min(6, (os.cpu_count() or 4) - 1))

    structural = structural_grid()
    structural_tasks = [(config_name(params), params) for params in structural]
    structural_df = run_tasks(structural_tasks, OUT_ROOT / "structural_summary.csv", workers)

    top_structural = structural_df.head(2)
    refine = refine_grid(top_structural)
    refine_tasks = [(config_name(params), params) for params in refine]
    refine_df = run_tasks(refine_tasks, OUT_ROOT / "refine_summary.csv", workers)

    combined = pd.concat([structural_df, refine_df], ignore_index=True)
    combined = sort_frame(combined.drop_duplicates(subset=["config_name"], keep="last"))
    combined.to_csv(OUT_ROOT / "combined_summary.csv", index=False)
    combined.head(10).to_json(OUT_ROOT / "top10.json", orient="records", indent=2)

    best = combined.iloc[0]
    print("BEST")
    print(best.to_json())
    print(f"runs={len(combined)} structural={len(structural_df)} refine={len(refine_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
