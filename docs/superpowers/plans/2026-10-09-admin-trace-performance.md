# Admin trace performance

Approved scope: a lightweight equity chart with decision/execution navigation, live refresh, and explicit missing/partial states.

1. Data: admin-only trace performance endpoint resolves only the exact trace.run_id. Return bounded chart points from existing equity storage; missing results allow the UI to use recorded decision snapshots. Never infer a run by agent name or date.
2. UI: SVG chart above timeline, market-time axis, sample return/drawdown, execution links joined by decision_id. Distinguish snapshot performance from final backtest metrics; do not invent equity at execution times.
3. Validation: test access control, missing run, invalid values, sampling, decision linkage, failure/empty states. Refresh performance while running with stale-response protection. Keep previous data on refresh failure. Fetch main and inspect PR state before publishing.
