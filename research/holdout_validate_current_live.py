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
OUT_ROOT = Path(r"corridor_outputs\fit_search\holdout_validate_current_live").resolve()
RUNS_DIR = OUT_ROOT / "runs"
FRAME: pd.DataFrame | None = None

SPLITS = {
    "train_2025_04_08_to_2026_01_07": (
        pd.Timestamp("2025-04-08T00:00:00Z"),
        pd.Timestamp("2026-01-08T00:00:00Z"),
    ),
    "test_2026_01_08_to_2026_04_08": (
        pd.Timestamp("2026-01-08T00:00:00Z"),
        pd.Timestamp("2026-04-09T00:00:00Z"),
    ),
}


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


def structural_grid() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    geometries = [
        ("symmetric", 0.0),
        ("broken_upper", 20.0),
        ("broken_lower", 20.0),
    ]
    for wing_mode, broken_extra in geometries:
        for width in (60.0, 80.0, 100.0):
            for tolerance in (10.0, 12.5, 15.0):
                rows.append(
                    {
                        "wing_mode": wing_mode,
                        "broken_wing_extra_width": broken_extra,
                        "butterfly_width": width,
                        "coverage_band_width": width * 2.0,
                        "center_tolerance": tolerance,
                        "recenter_threshold": 16.0,
                        "drift_persistence_bars": 8,
                        "rebuild_cooldown_minutes": 60,
                        "max_active_butterfly_layers": 2,
                        "primary_stop_loss_pct": 0.25,
                        "primary_take_profit_pct": 0.20,
                    }
                )
    return rows


def fmt(value: float | int | str) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(value).replace(".", "p")


def config_name(params: dict[str, float | str]) -> str:
    return (
        f"{params['wing_mode']}_"
        f"bw{fmt(params['broken_wing_extra_width'])}"
        f"_w{fmt(params['butterfly_width'])}"
        f"_tol{fmt(params['center_tolerance'])}"
    )


def summary_row(
    split_name: str,
    params: dict[str, float | str],
    summary: dict[str, object],
    frame: pd.DataFrame,
) -> dict[str, object]:
    multiplier = float(summary.get("option_multiplier") or 100)
    contracts = float(summary.get("contracts_per_layer") or 1)
    max_dd_dollars = float(summary.get("max_drawdown") or 0.0) * multiplier * contracts
    session_days = int(frame["timestamp"].dt.date.nunique()) if not frame.empty else 0
    net_pnl = float(summary.get("net_dollar_pnl") or 0.0)
    return {
        "split": split_name,
        "config_name": config_name(params),
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
        "bars": int(len(frame)),
        "session_days": session_days,
        "net_dollar_pnl": net_pnl,
        "pnl_per_day": (net_pnl / session_days) if session_days else 0.0,
        "return_on_capital": float(summary.get("return_on_capital") or 0.0),
        "max_drawdown_dollars": max_dd_dollars,
        "corridor_occupancy_rate": float(summary.get("corridor_occupancy_rate") or 0.0),
        "closed_layers": int(summary.get("closed_layers") or 0),
        "win_rate_by_closed_layer": summary.get("win_rate_by_closed_layer"),
        "score": net_pnl + (max_dd_dollars * 25.0),
    }


def run_one(task: tuple[str, dict[str, float | str]]) -> dict[str, object]:
    global FRAME
    if FRAME is None:
        raise RuntimeError("Worker frame is not initialized.")

    split_name, params = task
    split_start, split_end = SPLITS[split_name]
    frame = FRAME[(FRAME["timestamp"] >= split_start) & (FRAME["timestamp"] < split_end)].copy()
    if frame.empty:
        raise RuntimeError(f"Split {split_name} produced no rows.")

    run_dir = RUNS_DIR / split_name / config_name(params)
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return summary_row(split_name, params, summary, frame)

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

    result = CorridorBacktestEngine(cfg).run(frame)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
    return summary_row(split_name, params, result.summary, frame)


def sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["split", "score", "net_dollar_pnl", "pnl_per_day", "corridor_occupancy_rate"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)


def build_holdout_comparison(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["split"] == "train_2025_04_08_to_2026_01_07"].copy()
    test = df[df["split"] == "test_2026_01_08_to_2026_04_08"].copy()

    train["train_rank"] = train["score"].rank(method="dense", ascending=False).astype(int)
    test["test_rank"] = test["score"].rank(method="dense", ascending=False).astype(int)

    merged = train.merge(
        test,
        on=[
            "config_name",
            "wing_mode",
            "broken_wing_extra_width",
            "butterfly_width",
            "coverage_band_width",
            "center_tolerance",
            "recenter_threshold",
            "drift_persistence_bars",
            "rebuild_cooldown_minutes",
            "max_layers",
            "primary_stop_loss_pct",
            "primary_take_profit_pct",
        ],
        suffixes=("_train", "_test"),
    )
    merged["rank_delta"] = merged["test_rank"] - merged["train_rank"]
    merged["pnl_per_day_delta"] = merged["pnl_per_day_test"] - merged["pnl_per_day_train"]
    merged["score_delta"] = merged["score_test"] - merged["score_train"]
    return merged.sort_values(
        by=["test_rank", "train_rank", "score_test", "net_dollar_pnl_test"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not BARS_CSV.exists():
        raise FileNotFoundError(f"Bars CSV not found: {BARS_CSV}")

    tasks = [(split_name, params) for split_name in SPLITS for params in structural_grid()]
    workers = max(1, min(6, (os.cpu_count() or 4) - 1))
    rows: list[dict[str, object]] = []

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(str(BARS_CSV),)) as pool:
        future_map = {pool.submit(run_one, task): task for task in tasks}
        for idx, future in enumerate(as_completed(future_map), start=1):
            split_name, params = future_map[future]
            row = future.result()
            rows.append(row)
            df = sort_frame(pd.DataFrame(rows))
            df.to_csv(OUT_ROOT / "holdout_summary.csv", index=False)
            print(
                f"[{idx}/{len(tasks)}] {split_name} | {config_name(params)} | "
                f"pnl={float(row['net_dollar_pnl']):.2f} | "
                f"pnl/day={float(row['pnl_per_day']):.2f}"
            )

    df = sort_frame(pd.DataFrame(rows))
    df.to_csv(OUT_ROOT / "holdout_summary.csv", index=False)

    comparison = build_holdout_comparison(df)
    comparison.to_csv(OUT_ROOT / "holdout_comparison.csv", index=False)
    top10 = comparison.head(10).to_dict(orient="records")
    (OUT_ROOT / "top10_holdout.json").write_text(json.dumps(top10, indent=2), encoding="utf-8")

    print("BEST_TEST")
    print(json.dumps(top10[0], default=str))
    print(f"rows={len(df)} comparisons={len(comparison)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
