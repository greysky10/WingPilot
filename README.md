# DaySpy

DaySpy is a Python-based options trading research project focused on SPX butterfly corridor strategies.

The current codebase has two active strategy paths:

- `daily intraday corridor` in `corridor/`
- `weekly corridor` in `weekly_corridor/`

The daily SPX path is the main workflow today. It supports backtesting, live-prep snapshots, and Interactive Brokers paper execution. The weekly path is still research-grade and backtest-only.

## Current Status

- Main focus: daily SPX corridor butterfly strategy
- Supported workflows: local backtests, IB-backed live-prep snapshots, IB paper execution
- Current code of record: `corridor/`, `weekly_corridor/`, `run_backtest.py`, `run_paper_corridor.py`, `run_weekly_backtest.py`
- Archived experiments now live under `legacy/` and parameter sweeps live under `research/`

Important limitations:

- Daily and weekly backtests use a simplified butterfly pricer, not historical option-chain replay
- Paper execution logs are useful for workflow validation, not proof of live profitability
- The weekly strategy is not ready for paper/live deployment

## What The Project Does

The daily intraday corridor engine is built around a bar-by-bar strategy loop:

1. Classify the market as range-like or trend-dominant
2. Estimate an intraday center using recent bars
3. Select butterfly candidates around that center
4. Open, hold, rebuild, or exit positions through a state machine
5. Enforce execution-quality filters and protective exits
6. Write reports for transitions, orders, fills, and paper-day diagnostics

The system currently includes:

- range/trend regime detection
- center estimation and dynamic tolerance logic
- state-machine based entry, drift, rebuild, abort, and flatten flows
- option-chain loading and butterfly candidate selection
- Interactive Brokers paper combo-order execution
- daily paper-runner recovery and audit reporting
- local backtest and smoke-test tooling

## Repository Layout

```text
DaySpy/
|- corridor/                # Daily intraday corridor engine
|  |- backtest/             # Backtest engine and metrics
|  |- data/                 # IB data loading and contract helpers
|  |- execution/            # Paper runner
|  |- options/              # Butterfly pricing, combo building, chain selection
|  |- report/               # CSV/JSON summaries and plots
|  \- strategy/             # Regime detection, center estimation, state machine
|- weekly_corridor/         # Separate weekly corridor research path
|- legacy/                  # Archived earlier scripts and utilities
|- research/                # Parameter sweeps and research runners
|- data/
|  \- samples/              # Local sample inputs such as my_bars.csv
|- artifacts/
|  |- legacy/               # Archived CSV outputs from older runners
|  \- tmp/                  # Scratch snapshots and temporary exports
|- tests/                   # Automated test suite
|- run_backtest.py          # Daily backtest entry point
|- run_paper_corridor.py    # Daily paper runner entry point
|- run_live_prep.py         # Snapshot current regime/center/candidates
|- run_weekly_backtest.py   # Weekly backtest entry point
|- check_option_data.py     # IB quote probe for options availability
|- paper_test_summary.py    # Summarize paper-runner health for the day
|- smoke_test.py            # Local smoke workflow
|- strategy.py              # Compatibility wrapper for shared legacy helpers
\- README.md
```

## Requirements

- Windows or any environment where the scripts can run with Python 3.12
- Python `3.12`
- Core packages: `pandas`, `numpy`, `pytz`
- IB workflows: `ib_insync`
- Optional plotting: `matplotlib`
- Interactive Brokers TWS or Gateway for any live-prep, quote-check, or paper-execution flow

There is no pinned dependency file yet. A practical setup is:

```powershell
py -3.12 -m pip install pandas numpy pytz ib_insync matplotlib
```

## Environment Setup

Copy `.env.example` to `.env` and adjust values if needed:

```env
IB_HOST=localhost
IB_PORT=4002
IB_CLIENT_ID=2
IB_TIMEOUT=10
IB_EXCHANGE=SMART
IB_CURRENCY=USD
```

Typical IB paper defaults in this project are:

- host: `localhost`
- port: `4002`
- account session: IB paper

## Quick Start

### 1. Run the automated tests

```powershell
py -m unittest discover -s tests -v
```

### 2. Run the local smoke workflow

This runs unit tests plus a small backtest/live-prep sequence against a local CSV:

```powershell
py .\smoke_test.py --symbol SPY --bars-csv .\data\samples\my_bars.csv
```

### 3. Run the daily corridor backtest

Using a local bar file:

```powershell
py .\run_backtest.py --symbol SPX --bars-csv .\corridor_outputs\spx_grid_center_tol\SPX_5_mins_bars.csv --payoff-mode simplified --output-dir .\corridor_outputs\manual_run
```

When `--bars-csv` is omitted, the backtest can fetch historical bars from IBKR instead.

### 4. Prepare a live snapshot

This computes the latest regime, center, and candidate structures:

```powershell
py .\run_live_prep.py --symbol SPX --mode delayed --output .\corridor_outputs\live_prep\snapshot.json
```

### 5. Confirm option quotes are available

Before paper execution, verify that SPX option quotes are actually available through IB:

```powershell
py .\check_option_data.py --symbol SPX --host localhost --port 4002 --client-id 190 --dte-min 4
```

Healthy output should show `Status: LIVE` or `Status: DELAYED` with real bid/ask values.

### 6. Run a paper-runner health check

This validates startup state and computes a current snapshot without sending orders:

```powershell
py .\run_paper_corridor.py --symbol SPX --mode delayed --port 4002 --client-id 200 --check
```

### 7. Run the daily SPX paper corridor

Example paper-execution command:

```powershell
py .\run_paper_corridor.py --symbol SPX --mode delayed --port 4002 --client-id 200 --paper-execution --center-method vwap --butterfly-width 80 --coverage-band-width 160 --center-tolerance 12.5 --recenter-threshold 16 --drift-persistence-bars 8 --rebuild-cooldown-minutes 60 --max-layers 2 --candidate-body-search-steps 2 --dte-min 4 --dte-max 10 --max-option-spread 0.25 --primary-entry-end 13:30 --primary-entry-min-center-confidence 0.60 --primary-entry-max-momentum-pct 0.0010 --primary-entry-max-volume-ratio 1.15 --primary-stop-loss-pct 0.25 --primary-take-profit-pct 0.20 --max-spread-pct-of-debit 0.40 --combo-fill-wait-seconds 1.0 --combo-chase-steps 3 --combo-chase-spread-fraction 0.20
```

### 8. Summarize the current paper day

```powershell
py .\paper_test_summary.py --symbol SPX --write
```

### 9. Run the weekly research backtest

```powershell
py .\run_weekly_backtest.py --symbol SPX --bars-csv .\corridor_outputs\spx_grid_center_tol\SPX_5_mins_bars.csv
```

## Outputs

### Daily backtest outputs

The daily backtest writes artifacts under `corridor_outputs/...`, typically including:

- `summary.json`
- `actions.csv`
- `transitions.csv`
- `equity_curve.csv`

### Weekly backtest outputs

The weekly backtest writes artifacts under `weekly_corridor_outputs/...`, typically including:

- `summary.json`
- `actions.csv`
- `transitions.csv`
- `closed_layers.csv`
- `equity_curve.csv`

### Paper-runner outputs

The paper runner writes under `corridor_outputs/paper_runner/<SYMBOL>/`, including:

- `paper_state.json`
- `paper_orders.csv`
- `paper_transitions.csv`
- `paper_daily_report.json`
- `paper_test_summary.json`
- `paper_test_summary.csv`
- `paper_test_summary.txt`

These files are the main source of truth for paper-day diagnostics, fill quality, startup mode, and recovery state.

## Testing Coverage

The test suite currently focuses on the parts of the system that are easy to regress and expensive to debug manually:

- corridor state transitions
- rebuild and abort rules
- center estimation
- butterfly pricing and strike selection
- option-chain filtering
- backtest execution gates
- IB contract helpers
- paper-runner recovery, reporting, and protective exits
- weekly corridor behavior

Run the full suite with:

```powershell
py -m unittest discover -s tests -v
```

## Known Limitations

- The backtests do not replay full historical option chains
- Reported returns depend on explicit normalization assumptions such as starting capital, contract count, and option multiplier
- IB historical data or quote access can fail when TWS/Gateway sessions are misconfigured or connected from another IP
- The paper runner is designed for supervised paper usage, not unattended production trading
- The weekly corridor path is exploratory and should be treated as research only

## Notes

- If you want the shortest path through the project, start with `run_backtest.py`, `run_live_prep.py`, and `run_paper_corridor.py`
- If you are reviewing architecture, focus on `corridor/strategy/`, `corridor/options/`, `corridor/execution/`, and `corridor/backtest/`
- If you are reviewing research status, treat `corridor/` as the active path and `weekly_corridor/` as a separate experimental branch
- Older standalone runners and utilities were moved into `legacy/` to keep the project root focused on the active workflow
- Parameter sweeps were moved into `research/`, and loose sample/temp files were grouped under `data/` and `artifacts/`

## Disclaimer

This repository is a personal research and paper-trading project. Nothing here should be interpreted as financial advice or as evidence of a validated live trading edge.
