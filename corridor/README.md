# Corridor Reporting Notes

The corridor backtest reports two different kinds of metrics:

- Modeled metrics: option-like model points produced by the simplified butterfly pricer.
- Capital-normalized metrics: dollar and return figures derived from explicit assumptions.

## Paper Runner Diagnostics

When running in smoke mode (`--smoke`), the paper runner emits structured diagnostic artifacts to help debug candidate selection and order execution.

### Diagnostic Artifacts

| File | Description |
|------|-------------|
| `{prefix}_heartbeat.jsonl` | Compact heartbeat each poll cycle |
| `{prefix}_cycle_decisions.jsonl` | Cycle decision funnel |
| `{prefix}_candidate_diagnostics.jsonl` | Per-candidate diagnostics |
| `{prefix}_runner_events.jsonl` | Connectivity events |
| `{prefix}_diagnostics_summary.json` | Aggregated summary |

### Field Definitions

#### heartbeat.jsonl
| Field | Type | Description |
|-------|------|-------------|
| timestamp | ISO8601 | UTC timestamp |
| mode | string | "smoke" or "mainline" |
| symbol | string | Underlying symbol (e.g., "SPX") |
| state | string | Runner state (IDLE, ACTIVE_CENTERED, etc.) |
| regime | string | Market regime (RANGE, TREND, etc.) |
| ib_connection | string | "connected" or "disconnected" |
| market_data | string | "available" or "unavailable" |
| open_positions | int | Current open position count |
| warmup_bars | int | History bars collected |
| required_bars | int | Bars required for model readiness |

**Guaranteed**: All fields always present.

#### cycle_decisions.jsonl
| Field | Type | Description |
|-------|------|-------------|
| timestamp | ISO8601 | UTC timestamp |
| symbol | string | Underlying symbol |
| regime | string | Market regime |
| state | string | Runner state |
| chains_loaded | bool | Whether option chains loaded |
| candidates_generated | int | Total candidate structures attempted |
| rejected_quote_quality | int | Count rejected due to quote issues |
| rejected_non_positive_debit | int | Count rejected due to pricing |
| rejected_spread | int | Count rejected due to spread |
| rejected_other | int | Count rejected for other reasons |
| eligible_candidates | int | Candidates passing all filters |
| orders_submitted | int | Orders submitted this cycle |
| fills | int | Fills received this cycle |
| cancels | int | Cancels this cycle |
| replaces | int | Replace orders this cycle |
| top_reject_reason | string? | Primary rejection category |
| top_reject_subcode | string? | Primary rejection subcode |

**Guaranteed**: All numeric fields always present. top_reject_reason/subcode may be null.

#### candidate_diagnostics.jsonl
| Field | Type | Description |
|-------|------|-------------|
| timestamp | ISO8601 | UTC timestamp |
| symbol | string | Underlying symbol |
| regime | string | Market regime |
| state | string | Runner state |
| expiry | string | Option expiry (YYYYMMDD) |
| dte | int? | Days to expiration |
| lower_strike | float | Lower wing strike |
| body_strike | float | Body strike |
| upper_strike | float | Upper wing strike |
| wing_width | float | Wing width in points |
| target_debit | float | Target debit from candidate |
| computed_mid_debit | float | Computed mid from quotes |
| bid_debit | float? | Bid-side debit |
| ask_debit | float? | Ask-side debit |
| absolute_spread | float | Total spread in points |
| spread_ratio | float? | Spread/debit ratio |
| quote_complete | bool | All leg quotes present |
| rejection_reason | string? | Top-level rejection category |
| rejection_subcode | string? | Rejection subcode |
| is_submitted | bool | Candidate was submitted |
| is_top_rejected | bool | Top rejection this cycle |

**Guaranteed**: strike fields, expiry always present. Quote fields may be null.

#### runner_events.jsonl
| Field | Type | Description |
|-------|------|-------------|
| timestamp | ISO8601 | UTC timestamp |
| symbol | string | Underlying symbol |
| event_type | string | connected, disconnected, socket_disconnect, socket_reconnect |
| connection_state | string | Current IB connection state |
| market_data_state | string | Current market data state |
| previous_diagnostics_available | bool | Diagnostics preserved from prior cycle |
| previous_diagnostics_validity | string | "complete", "partial", "invalid", "none" |
| cycle_complete | bool | Whether cycle completed |
| message | string? | Optional detail message |

**Guaranteed**: All fields except message always present.

### Sampling and Cap Rules

- **Heartbeat**: Emitted every poll cycle (no cap)
- **Cycle Decision**: Emitted every evaluation cycle (no cap)
- **Candidate Diagnostics**:
  - Max 5 rejected candidates per cycle
  - Max 2 per subcode within a cycle
  - Top rejected candidate always emitted
  - Submitted candidates not capped
  - Cycle tracking resets at start of each cycle

### Deduplication

Full summaries are deduplicated based on material state:
- state, regime, connection_state, market_data_state
- top_reject_reason, top_reject_subcode
- eligible_candidate_count, order_count, fill_count, cancel_count, replace_count
- has_warning, has_error

Identical material state suppresses repeated full summaries.

### Disconnect Preservation

When IB disconnects:
1. Last valid diagnostics are preserved with validity="complete"
2. After disconnect, validity marked as "partial"
3. After failed reconnect, validity marked as "invalid"
4. Timestamps tracked: market_data, chain_build, candidate_evaluation, order_submission, cycle_complete

### Reading No-Fill Sessions

1. Check `cycle_decisions.jsonl` for `eligible_candidates=0` or low values
2. Look at `rejected_quote_quality`, `rejected_non_positive_debit`, `rejected_spread` for primary failure
3. Cross-reference `top_reject_reason` and `top_reject_subcode`
4. Review `candidate_diagnostics.jsonl` for specific strike/quote issues
5. Check `runner_events.jsonl` for disconnects during session

### Rejection Reason Hierarchy

| Top Level | Subcodes | Description |
|-----------|----------|-------------|
| `quote_quality` | `missing_leg_bid`, `missing_leg_ask`, `incomplete_spread`, `stale_quote` | Option quote issues |
| `non_positive_debit` | `zero_mid`, `negative_mid`, `bad_combo_math` | Candidate pricing problems |
| `spread_too_wide` | `absolute`, `ratio`, `quote_gap_distorted` | Spread ratio or width issues |
| `runner` | `socket_disconnect`, `market_data_unavailable` | Runner/connectivity issues |

### Running Smoke Mode

```bash
python run_paper_corridor.py --smoke --output-dir corridor_outputs/smoke_test
```

The diagnostics are enabled automatically when `--smoke` is passed. Artifacts are written to the configured output directory.
