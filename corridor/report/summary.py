from __future__ import annotations

import json
from pathlib import Path

from corridor.backtest.engine import BacktestResult
from corridor.backtest.trades import actions_to_frame, equity_to_frame, transitions_to_frame
from corridor.models import BacktestArtifacts


def save_backtest_outputs(output_dir: Path, result: BacktestResult) -> BacktestArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)

    transitions_path = output_dir / "transitions.csv"
    actions_path = output_dir / "actions.csv"
    summary_path = output_dir / "summary.json"
    equity_curve_path = output_dir / "equity_curve.csv"

    transitions_to_frame(result.transitions).to_csv(transitions_path, index=False)
    actions_to_frame(result.actions).to_csv(actions_path, index=False)
    equity_to_frame(result.equity_curve).to_csv(equity_curve_path, index=False)
    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")

    return BacktestArtifacts(
        transitions_path=transitions_path,
        actions_path=actions_path,
        summary_path=summary_path,
        equity_curve_path=equity_curve_path,
    )
