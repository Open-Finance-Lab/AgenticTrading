# Admin trace performance

Approved scope: a lightweight equity chart with decision/execution navigation, live refresh, and explicit missing/partial states.

1. Data: admin-only trace performance endpoint resolves only the exact trace.run_id. Return bounded chart points from existing equity storage; missing results allow the UI to use recorded decision snapshots. Never infer a run by agent name or date.
2. UI: SVG chart above timeline, market-time axis, sample return/drawdown, execution links joined by decision_id. Distinguish snapshot performance from final backtest metrics; do not invent equity at execution times.
3. Validation: test access control, missing run, invalid values, sampling, decision linkage, failure/empty states. Refresh performance while running with stale-response protection. Keep previous data on refresh failure. Fetch main and inspect PR state before publishing.

## Follow-up: readable timeline and export (2026-10-10)

Approved: collapsed event summaries, decision/step grouping, raw JSON behind a second disclosure, failed groups open by default, and preserving disclosure state on polling. Group only through explicit IDs; never infer relationships from adjacent timestamps. Trade navigation opens the matching decision group.

Admin backtest details use an admin-gated exact-run projection across owners, rather than redirecting to the session-filtered trading interface. Display only stored fields and explicit missing-result states.

Exports: admin-only JSON and Markdown, schema version and snapshot timestamp, fixed highest event sequence, all pages through that sequence, sensitive-field redaction, artifact references only. Build the output in a spooled temporary file before returning headers. Test >100 events, concurrent append, permission rejection, readable/escaped Markdown, and cross-owner exact-run lookup.
