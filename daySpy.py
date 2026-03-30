#! python3.12
"""
Compatibility entrypoint for the DaySpy live IB runner.

Primary modules:
- strategy.py: shared signal logic and bar-processing pipeline
- live_ib.py: IB live runner
- backtest.py: historical replay runner
"""

from live_ib import main


if __name__ == "__main__":
    raise SystemExit(main())
