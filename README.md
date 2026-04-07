# DaySpy

DaySpy currently has two corridor strategy paths:

- `daily intraday corridor` in [corridor](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor)
- `weekly corridor` in [weekly_corridor](c:\Users\Alan\OneDrive\Documents\DaySpy\weekly_corridor)

The daily SPX path is the only one set up for paper-trading workflow. The weekly path is backtest-only and still research-grade.

## Current Status

- Main focus: `daily SPX corridor butterfly`
- Paper trading: supported through [run_paper_corridor.py](c:\Users\Alan\OneDrive\Documents\DaySpy\run_paper_corridor.py)
- Backtesting: supported through [run_backtest.py](c:\Users\Alan\OneDrive\Documents\DaySpy\run_backtest.py)
- Weekly strategy: separate path, backtest only, not recommended for live/paper deployment yet

Important:

- The backtests still use a **simplified butterfly pricer**, not real historical option-chain replay.
- `total_return` is a modeled-unit alias, not a percent return.
- Use stressed backtests and paper execution logs as filters, not proof of live profitability.

## Strategy Summary

### Daily Intraday Corridor

Core files:

- [corridor\strategy\corridor_state_machine.py](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor\strategy\corridor_state_machine.py)
- [corridor\strategy\regime.py](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor\strategy\regime.py)
- [corridor\strategy\center_estimator.py](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor\strategy\center_estimator.py)
- [corridor\options\butterfly_selector.py](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor\options\butterfly_selector.py)
- [corridor\execution\paper.py](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor\execution\paper.py)

Idea:

- trade only when the market looks `RANGE`
- estimate an intraday center
- deploy a butterfly around that center
- tolerate some drift
- rebuild only after sustained drift
- abort if the market becomes trend-dominant

Main indicators used:

- `range_width_pct`
- `trend_slope_pct`
- `momentum_pct`
- `volume_ratio`
- breakout checks above/below recent range
- center confidence from the center estimator

Current stricter daily SPX controls:

- stricter primary entry
- fewer rebuilds
- max `2` active butterflies total
- primary stop-loss / take-profit
- quote-quality filter for paper execution
- combo-only chase logic, no manual leg fallback

### Daily Strike Logic

Default symmetric structure:

- `body = rounded center`
- `lower = body - butterfly_width`
- `upper = body + butterfly_width`

Adaptive paper mode:

- default behavior remains `symmetric`
- `adaptive` mode first tries the symmetric candidate
- if the symmetric candidate is execution-poor, it can fall back to a broken-wing candidate

Current adaptive trigger:

- the best symmetric candidate fails the execution-quality guard
- practically, this is mainly `spread_ratio > max_spread_pct_of_debit`

This is a **paper/live candidate selection feature**, not a true historical adaptive backtest.

### Daily Entry, Exit, and Cut Logic

Primary entry:

- only when regime is `RANGE`
- only inside the valid intraday window
- blocked if center confidence is too low
- blocked if momentum is too high
- blocked if volume ratio is too high
- blocked on configured event days if enabled

Supplemental:

- added only near the corridor edge
- total active butterflies are capped by `max_layers`

Rebuild:

- only after price stays outside tolerance long enough
- rebuild also respects cooldown

Abort:

- trend up/down regime
- breakout behavior
- momentum expansion

Protective exits:

- `primary_stop_loss_pct`
- `primary_take_profit_pct`

Session exit:

- positions are flushed outside the valid trading window

### Weekly Corridor

Core files:

- [weekly_corridor\config.py](c:\Users\Alan\OneDrive\Documents\DaySpy\weekly_corridor\config.py)
- [weekly_corridor\strategy.py](c:\Users\Alan\OneDrive\Documents\DaySpy\weekly_corridor\strategy.py)
- [weekly_corridor\state_machine.py](c:\Users\Alan\OneDrive\Documents\DaySpy\weekly_corridor\state_machine.py)
- [run_weekly_backtest.py](c:\Users\Alan\OneDrive\Documents\DaySpy\run_weekly_backtest.py)

Weekly is separate from the intraday corridor:

- longer decision horizon
- weekly center
- initial 3-butterfly corridor
- one weekly adjustment max by default
- backtest only

It is currently **not** the recommended live focus.

## Setup

### 1. IB Paper Connection

Expected defaults:

- host: `localhost`
- port: `4002`
- account: `paper`

Environment can be loaded from [`.env`](c:\Users\Alan\OneDrive\Documents\DaySpy\.env):

```env
IB_HOST=localhost
IB_PORT=4002
```

### 2. Check Option Quotes

Before running the paper runner, confirm SPX option quotes are available:

```powershell
py .\check_option_data.py --symbol SPX --host localhost --port 4002 --client-id 190 --dte-min 4
```

Good result:

- `Status: DELAYED` or `Status: LIVE`
- real `bid` and `ask` values, not all `None`

## Backtests

### Daily Backtest

Runner:

- [run_backtest.py](c:\Users\Alan\OneDrive\Documents\DaySpy\run_backtest.py)

Example daily SPX stressed backtest:

```powershell
py .\run_backtest.py --symbol SPX --bars-csv .\corridor_outputs\spx_grid_center_tol\SPX_5_mins_bars.csv --payoff-mode simplified --stress-profile conservative --butterfly-width 80 --wing-mode symmetric --broken-wing-extra-width 0 --coverage-band-width 160 --center-tolerance 12.5 --recenter-threshold 16 --drift-persistence-bars 8 --rebuild-cooldown-minutes 60 --max-layers 2 --primary-entry-end 13:30 --primary-entry-min-center-confidence 0.60 --primary-entry-max-momentum-pct 0.0010 --primary-entry-max-volume-ratio 1.15 --primary-stop-loss-pct 0.25 --primary-take-profit-pct 0.20 --output-dir .\corridor_outputs\spx_daily_manual_run
```

Outputs:

- `summary.json`
- `actions.csv`
- `transitions.csv`
- `equity_curve.csv`

### Broken-Wing Comparison Research

Research scripts:

- [sweep_spx_broken_wing_compare.py](c:\Users\Alan\OneDrive\Documents\DaySpy\sweep_spx_broken_wing_compare.py)
- [sweep_spx_daily_stressed.py](c:\Users\Alan\OneDrive\Documents\DaySpy\sweep_spx_daily_stressed.py)

The broken-wing comparison is **backtest-only research** right now. It should not be treated as validated live edge.

## Daily SPX Paper Trading

### Recommended Startup Sequence

1. Check option quotes:

```powershell
py .\check_option_data.py --symbol SPX --host localhost --port 4002 --client-id 190 --dte-min 4
```

2. Snapshot check:

```powershell
py .\run_paper_corridor.py --symbol SPX --mode delayed --port 4002 --client-id 200 --check
```

Healthy startup should show:

- `Startup mode | history-seeded`
- `model_ready=True`
- `warmup_mode=False`

### Recommended Daily SPX Paper Command

Symmetric default:

```powershell
py .\run_paper_corridor.py --symbol SPX --mode delayed --port 4002 --client-id 200 --paper-execution --center-method vwap --butterfly-width 80 --coverage-band-width 160 --center-tolerance 12.5 --recenter-threshold 16 --drift-persistence-bars 8 --rebuild-cooldown-minutes 60 --max-layers 2 --candidate-body-search-steps 2 --dte-min 4 --dte-max 10 --max-option-spread 0.25 --primary-entry-end 13:30 --primary-entry-min-center-confidence 0.60 --primary-entry-max-momentum-pct 0.0010 --primary-entry-max-volume-ratio 1.15 --primary-stop-loss-pct 0.25 --primary-take-profit-pct 0.20 --max-spread-pct-of-debit 0.40 --combo-fill-wait-seconds 1.0 --combo-chase-steps 3 --combo-chase-spread-fraction 0.20
```

Adaptive fallback mode:

```powershell
py .\run_paper_corridor.py --symbol SPX --mode delayed --port 4002 --client-id 200 --paper-execution --wing-mode adaptive --broken-wing-extra-width 20 --center-method vwap --butterfly-width 80 --coverage-band-width 160 --center-tolerance 12.5 --recenter-threshold 16 --drift-persistence-bars 8 --rebuild-cooldown-minutes 60 --max-layers 2 --candidate-body-search-steps 2 --dte-min 4 --dte-max 10 --max-option-spread 0.25 --primary-entry-end 13:30 --primary-entry-min-center-confidence 0.60 --primary-entry-max-momentum-pct 0.0010 --primary-entry-max-volume-ratio 1.15 --primary-stop-loss-pct 0.25 --primary-take-profit-pct 0.20 --max-spread-pct-of-debit 0.40 --combo-fill-wait-seconds 1.0 --combo-chase-steps 3 --combo-chase-spread-fraction 0.20
```

Meaning of adaptive:

- use symmetric by default
- only fall back to broken-wing when the symmetric candidate is execution-poor

### Common Runtime Behavior

Normal:

- `No new completed bars.`
- `IDLE -> ABORT` on trend days
- no order if there is no valid `RANGE` setup

Not ideal but still recoverable:

- `Startup mode | warmup-only`

This means history seed failed and the runner is warming up from live data.

## Files To Watch During Paper Trading

State:

- [paper_state.json](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor_outputs\paper_runner\SPX\paper_state.json)

Transitions:

- [paper_transitions.csv](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor_outputs\paper_runner\SPX\paper_transitions.csv)

Orders:

- [paper_orders.csv](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor_outputs\paper_runner\SPY\paper_orders.csv)

Recovery:

- [paper_recovery.json](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor_outputs\paper_runner\SPX\paper_recovery.json)

Runner logs:

- [paper_runner_stdout.log](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor_outputs\paper_runner\SPY\paper_runner_stdout.log)
- [paper_runner_stderr.log](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor_outputs\paper_runner\SPY\paper_runner_stderr.log)

Daily report:

- [paper_daily_report.json](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor_outputs\paper_runner\SPX\paper_daily_report.json)
- [paper_daily_report.csv](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor_outputs\paper_runner\SPX\paper_daily_report.csv)
- [paper_test_summary.json](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor_outputs\paper_runner\SPX\paper_test_summary.json)
- [paper_test_summary.csv](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor_outputs\paper_runner\SPX\paper_test_summary.csv)
- [paper_test_summary.txt](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor_outputs\paper_runner\SPX\paper_test_summary.txt)

Note:

- older `SPY` paper logs may still exist from earlier testing
- current daily focus should be `SPX`

## Daily Report And Fill Audit

The paper runner now tracks quote-vs-limit-vs-fill quality.

Important fields:

- `quote_reference`
- `spread_ratio`
- `limit_vs_quote`
- `fill_edge_vs_quote`
- `fill_edge_vs_limit`

Daily aggregates:

- `avg_open_fill_edge_vs_quote`
- `avg_close_fill_edge_vs_quote`
- `avg_open_fill_edge_vs_limit`
- `avg_close_fill_edge_vs_limit`
- `avg_filled_spread_ratio`
- `best_*`
- `worst_*`

How to use them:

- negative `avg_open_fill_edge_vs_quote` means real entries are worse than target quote
- negative `avg_close_fill_edge_vs_quote` means real exits are worse than target quote
- high `avg_filled_spread_ratio` means your quote quality is poor and you should skip more trades

Quick daily check:

```powershell
py .\paper_test_summary.py --symbol SPX
```

This prints a short `PASS / WARN / FAIL` summary for the current paper day and rewrites the same judgment into `paper_test_summary.*` when the runner updates its state.

Levers to tighten:

- `--max-spread-pct-of-debit`
- `--combo-chase-steps`
- `--combo-chase-spread-fraction`

## Recovery After Interruption

IB account is the source of truth. The runner can restore local state, but it should never guess through a mismatch.

### Best Recovery Method

If you think there may be an open SPX butterfly after interruption:

1. Preview adoption:

```powershell
py .\adopt_paper_butterfly.py --symbol SPX --host localhost --port 4002 --client-id 201
```

2. If the open butterfly is clean and expected, write recovery:

```powershell
py .\adopt_paper_butterfly.py --symbol SPX --host localhost --port 4002 --client-id 201 --write
```

3. Restart with sync:

```powershell
py .\run_paper_corridor.py --symbol SPX --mode delayed --port 4002 --client-id 200 --paper-execution --sync-on-start --wing-mode adaptive --broken-wing-extra-width 20 --center-method vwap --butterfly-width 80 --coverage-band-width 160 --center-tolerance 12.5 --recenter-threshold 16 --drift-persistence-bars 8 --rebuild-cooldown-minutes 60 --max-layers 2 --candidate-body-search-steps 2 --dte-min 4 --dte-max 10 --max-option-spread 0.25 --primary-entry-end 13:30 --primary-entry-min-center-confidence 0.60 --primary-entry-max-momentum-pct 0.0010 --primary-entry-max-volume-ratio 1.15 --primary-stop-loss-pct 0.25 --primary-take-profit-pct 0.20 --max-spread-pct-of-debit 0.40 --combo-fill-wait-seconds 1.0 --combo-chase-steps 3 --combo-chase-spread-fraction 0.20
```

### Flat Restart

If you know the paper account is flat, restart normally without sync.

### Forced Flatten

Dry run:

```powershell
py .\flatten_paper_spy.py --symbol SPX --host localhost --port 4002 --client-id 202
```

Submit flatten:

```powershell
py .\flatten_paper_spy.py --symbol SPX --host localhost --port 4002 --client-id 202 --submit
```

Even though the script name says `spy`, it works with `--symbol SPX`.

Use flatten if:

- recovery is stale
- account state is messy
- manual intervention created ambiguity
- sync-on-start refuses to continue

## Common Problems

### `Trading TWS session is connected from a different IP address`

This is an IB historical data / session issue, not necessarily a second local process.

What to do:

- close Client Portal / mobile sessions
- avoid VPN or network switching
- restart IB Gateway
- use warmup fallback if needed

### `No market data during competing live session`

IB is refusing the quote stream. Treat the account connection as not healthy for unattended paper execution until it clears.

### Existing positions are already open

This means the runner is protecting you from starting flat on top of real paper positions.

Resolve with:

- adoption plus `--sync-on-start`
- or flatten first

### Combo quote is too wide

If the runner skips entries due to quote quality:

- lower trade frequency is normal
- tighten or relax `--max-spread-pct-of-debit` carefully
- do not bypass combo-only protection by legging manually in code

## Reporting Notes

Backtest reporting details live in:

- [corridor\README.md](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor\README.md)

Weekly path notes live in:

- [weekly_corridor\README.md](c:\Users\Alan\OneDrive\Documents\DaySpy\weekly_corridor\README.md)

Current reporting rules:

- `total_return` is only a backward-compatible modeled-unit alias
- `return_on_capital` is explicit capital normalization
- `max_modeled_capital_at_risk` is a modeled conservative proxy, not true market tail risk

## Recommended Operating Discipline

For the first few paper days:

1. verify option quotes first
2. run `--check`
3. run the paper command
4. do not keep changing parameters intraday
5. review the daily report after the session

What you want to learn from paper:

- are entries only happening on intended range days
- are combo fills close to target quote assumptions
- are exits materially worse than expected
- how often adaptive mode actually uses broken wings

What paper does **not** prove:

- true live profitability
- real historical chain economics
- realistic live slippage in stress

## Tests

Main validation command:

```powershell
py -m unittest discover -s .\tests -p "test_*py" -v
```

Syntax check:

```powershell
py -m py_compile .\run_backtest.py .\run_paper_corridor.py .\corridor\execution\paper.py
```

## Short Version

If you only need the minimum:

1. check quotes
2. run `--check`
3. run daily SPX paper with either:
   - symmetric default
   - or `adaptive` fallback mode
4. after interruption:
   - adopt if position is open
   - restart with `--sync-on-start`
   - flatten if state is messy
