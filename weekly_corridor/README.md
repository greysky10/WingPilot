# Weekly Corridor

This package is a separate strategy path from the intraday corridor system.

## How It Differs From The Intraday Corridor

- It works on `30-minute` or `60-minute` decision bars, not the intraday `5-minute` corridor loop.
- It uses a **weekly center** estimated from multi-day bars, not an intraday center that can be rebuilt repeatedly inside one session.
- It deploys an initial **3-butterfly weekly corridor** around that weekly center.
- It holds structures for roughly `4-7` trading days with `10-14` DTE assumptions.
- It allows at most **one weekly adjustment** by default instead of repeated intraday layering/rebuilds.
- It aborts on clearly trend-dominant or breakout weeks, and it forces an end-of-week / low-DTE exit.

Run the weekly backtest with:

```powershell
py .\run_weekly_backtest.py --symbol SPX --bars-csv .\corridor_outputs\spx_grid_center_tol\SPX_5_mins_bars.csv
```
