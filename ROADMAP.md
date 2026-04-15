# DaySpy Paper-to-Live Roadmap

Start date: 2026-04-15

## Objective

Move from paper testing to small live deployment without relying on low-frequency signal accumulation alone.

The plan has two parallel tracks:

- `q3 compromise` mainline: validate strategy quality.
- `paper smoke mode`: validate execution quality quickly.

`paper smoke mode` is planned next. It is not implemented yet.

## Current Mainline

- Strategy family: long DTE SPX call butterflies
- Current practical paper version: `q3 compromise`
- Typical frequency: about 12 trades per year
- Purpose: strategy quality, not fast execution sampling

## Phases

| Phase | Time Span | Goal | What To Run | Exit Criteria |
|---|---:|---|---|---|
| 1. Execution validation | 1-2 weeks | Confirm IB paper combo execution is usable | `paper smoke mode` plus `q3 compromise` | Fill quality is acceptable and chase behavior is stable |
| 2. Mainline paper accumulation | 3-6 weeks | Validate the long DTE mainline in real paper conditions | `q3 compromise` | Enough full open/hold/close cycles with no obvious execution mismatch |
| 3. Tiny live pilot | Week 4-8 | Validate live vs paper with minimal real risk | smallest live size only | Live fills and behavior stay within tolerance |
| 4. Gradual scaling | Month 2-3 | Decide whether to move from `q3` to `q4` or add compounding | live pilot plus ongoing paper | Execution quality holds after scaling |

## Decision Gates

### Before tiny live

- Combo orders fill reliably enough
- Spread gate is not blocking most valid entries
- Chase count is not consistently excessive
- Paper fills are not obviously overstating achievable price
- `q3 compromise` still behaves reasonably in live paper conditions

### Before scaling up

- Tiny live results look similar to paper
- Drawdowns stay within plan
- No persistent fill degradation after size increase
- Position management works cleanly across open, hold, and close

## Time Estimates

### Fast path

- Add `paper smoke mode`
- Run it alongside `q3 compromise`
- Earliest tiny live start: 2-4 weeks

### More prudent path

- Validate execution first
- Build a meaningful set of mainline paper samples
- Tiny live start: 4-8 weeks

### Slow path

- No `paper smoke mode`
- Rely only on the current low-frequency mainline
- Likely timeline: 1-2 months or longer before live

## Why Two Tracks Are Needed

The current long DTE mainline is too low frequency to validate execution quickly.

Without a separate execution-testing mode, waiting for enough paper samples is too slow. The strategy track and the execution track should not be treated as the same problem.

## Immediate Next Steps

1. Keep running `q3 compromise` on paper.
2. Implement `paper smoke mode`.
3. Track daily execution metrics:
   - fill rate
   - combo chase count
   - spread gate rejections
   - fill quality vs expected debit
   - order lifecycle issues
4. Do not move to `q4` until `q3` fills look acceptable in real paper conditions.

## Current Recommendation

- Short term: run `q3 compromise`
- Next build task: implement `paper smoke mode`
- Scaling target: do not jump directly to larger size until execution quality is confirmed
