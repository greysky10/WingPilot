from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_equity_plot(equity_frame: pd.DataFrame, output_path: Path) -> bool:
    """Save an equity curve plot when matplotlib is available."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        return False

    if equity_frame.empty:
        return False

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(pd.to_datetime(equity_frame["timestamp"]), equity_frame["total_equity"], color="#0f766e", linewidth=1.6)
    axis.set_title("Corridor Equity Curve")
    axis.set_xlabel("Time")
    axis.set_ylabel("PnL")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True
