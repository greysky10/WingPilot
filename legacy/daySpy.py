#! python3.12
"""
Compatibility entrypoint for the DaySpy live IB runner.

Primary modules:
- strategy.py: shared signal logic and bar-processing pipeline
- live_ib.py: IB live runner
- backtest.py: historical replay runner
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legacy.live_ib import main


if __name__ == "__main__":
    raise SystemExit(main())
