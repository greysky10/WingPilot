# Corridor Reporting Notes

The corridor backtest reports two different kinds of metrics:

- Modeled metrics: option-like model points produced by the simplified butterfly pricer.
- Capital-normalized metrics: dollar and return figures derived from explicit assumptions.

Important:

- `total_return` is kept only for backward compatibility.
- `total_return` is **not** a percent return.
- `total_return` is the same modeled-unit value as `net_modeled_pnl` and `model_points`.

Current formulas:

- `model_points = equity_curve.total_equity[-1]`
- `gross_modeled_pnl = equity_curve.gross_total_equity[-1]`
- `net_modeled_pnl = equity_curve.total_equity[-1]`
- `dollar_pnl_per_1_lot = net_modeled_pnl * option_multiplier`
- `net_dollar_pnl = net_modeled_pnl * option_multiplier * contracts_per_layer`
- `max_gross_deployment_dollars = max_modeled_execution_capital_at_risk * option_multiplier * contracts_per_layer`
- `max_modeled_state_capital_at_risk = max(equity_curve.modeled_capital_at_risk)`
- `max_modeled_execution_capital_at_risk = conservative same-timestamp open-before-close peak based on action replay`
- `max_modeled_close_friction_reserve = close-side friction reserve at the conservative execution peak`
- `max_modeled_capital_at_risk = max_modeled_execution_capital_at_risk + max_modeled_close_friction_reserve`
- `worst_day_pnl_dollars = min(daily net modeled pnl) * option_multiplier * contracts_per_layer`
- `best_day_pnl_dollars = max(daily net modeled pnl) * option_multiplier * contracts_per_layer`
- `profit_factor_by_day = gross positive daily modeled pnl / abs(gross negative daily modeled pnl)`
- `gross_winners_dollars = sum(positive closed-layer realized pnl) * option_multiplier * contracts_per_layer`
- `gross_losers_dollars = abs(sum(negative closed-layer realized pnl)) * option_multiplier * contracts_per_layer`
- `profit_factor_by_closed_layer = gross winners / abs(gross losers)` using closed-layer realized PnL
- `return_on_capital = net_dollar_pnl / starting_capital`
- `return_on_max_risk = net_dollar_pnl / max_modeled_capital_at_risk_dollars`

The default normalization assumptions live in [config.py](c:\Users\Alan\OneDrive\Documents\DaySpy\corridor\config.py):

- `starting_capital = 100000.0`
- `contracts_per_layer = 1`
- `option_multiplier = 100`
