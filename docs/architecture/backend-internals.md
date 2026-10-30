# Backend internals

Living developer detail behind rules that `CLAUDE.md` states in one line. It is not dated: update the matching section in the same PR as the code. Dated design records live in `docs/superpowers/`. GitHub-only (Read the Docs builds `docs/source/` only). Per-variable configuration: `docs/architecture/configuration.md`. Paths are repo-relative; a section that says it is under `dashboard/backend/` uses the short form. Symbols are named, not numbered, because line numbers rot.

## LLM billing internals

Expands the `CLAUDE.md` rule "LLM backtest billing, the live path" (`POST /backtest/run` with `decision_source='llm'` needs `billing_mode`; per-call reserve and settle; fail closed). Paths are under `dashboard/backend/`.

### Launch preflight and the claim

`_enforce_platform_credits_balance` (`api/routers/backtests.py`) refuses a `platform_credits` launch before any child spawns: 402 when the balance cannot cover the run, 403 for a restricted account.

- **Estimate.** `_estimated_platform_credits` returns `_PlatformCreditsEstimate(calls, needed_micro)`, per call shape: sum of `count x expected cost`, with the shape that has the largest reservation ceiling counted once at that ceiling instead of its expected cost (the run can only be short by one in-flight reservation). Prompt size is `_PREFLIGHT_PROMPT_FIXED_BYTES` (3000) plus `_PREFLIGHT_PROMPT_BYTES_PER_SYMBOL` (311) per symbol plus the caller's prompt text; output tokens are the model's median capped at the first-attempt ceiling. Inputs are typical, not worst case: it is a refusal, so over-estimating is the unsafe direction (calibration is in the comment above the constants). The balance is first reduced by what the owner's other running platform-credits runs still expect to spend (`_owner_platform_credit_slots`, `_credits_committed_micro`).
- **Race.** `_CreditsClaim` is re-checked under the ledger lock in `_try_acquire_backtest_slot`, so two launches that both passed the read cannot share one balance.
- **Fail-open on purpose.** An unpriceable run or a failed balance read lets the run through; the per-call reservation in `infrastructure/llm/execution/service.py` stays the hard gate. BYOK never reads the balance.

### Empty balance mid-run

`_execute_platform` (`infrastructure/llm/execution/service.py`) maps the store's `InsufficientCreditsError` to `INSUFFICIENT_CREDITS` (analytics `credits_insufficient`), not the catch-all `BILLING_FAILED`; the category, the "Not enough ATL Credits..." message and `run_failed_line` live in `infrastructure/llm/execution/errors.py`. The child prints `ERROR: llm.run_failed category=<value>` (`run_failed_line`; fixed tokens, never upstream text); the parent (`_child_llm_failure`) shows the safe message and records the category instead of a traceback tail and `internal_error`. Non-owners see only `_LLM_FAILURE_PUBLIC_ERROR`.

### Welcome grant

`DEFAULT_SIGNUP_CREDIT_CAMPAIGN = "default_signup_credits_v2"` (`domain/credits/service.py`), 8 Credits. Holders of a superseded campaign (v1, 1.5 Credits) do not get v2, because the boot backfill (`backfill_default_signup_credits`, called from `app.py`) replays the campaign over every account. Change an amount under a new campaign key, never in place: the amount is part of the idempotency digest.

### Parent-side run lifecycle

- **`finalize_run` again in the parent.** `run_backtest_background` repeats `finalize_run` in its `finally` (lane passed as the `BillingMode` enum, not a string), because the child's cleanup does not run when it is killed (timeout, cancel route); otherwise a reservation is stranded. **Any new launch path must thread `billing_mode` into `run_backtest_background`**, or a platform-credits timeout reports as free.
- **Failover.** The route applies `ATL_PLATFORM_PROVIDER_ORDER` and signs the ordered tuple into the handoff (`provider_ids`); `_execute_with_platform_failover` (`infrastructure/llm/execution/service.py`) tries exactly that tuple and never re-derives or widens it. It moves to the next candidate only on `_PLATFORM_FAILOVER_CATEGORIES` (credential missing/invalid, provider unavailable/timeout, quota exhausted); any other category (`INSUFFICIENT_CREDITS`, `BILLING_FAILED`, ...) raises at once. When every candidate fails, the error raised is the first timeout/unavailable any candidate ended on (`first_call_specific`), else the last candidate's error: a quota-dead OpenRouter fallback must not relabel a CommonStack timeout as `provider_quota_exhausted`. Repeats and failover share one `attempt_index` counter.
- **`ERROR: llm.` lines.** The child prints `ERROR: llm.platform_quota_exhausted provider=<id> fallback=<next|none>` once per provider per process, flushed (CommonStack has no balance endpoint, so a failed call is the only way to see it drain; BYOK never prints it), and `ERROR: llm.provider_attempt_failed ...` per failed timeout/unavailable attempt (BYOK too; other failure categories print nothing here). `_drain_stream` echoes each such line to the parent's stdout live, through `_redact_credentials`, because the timeout path never dumps the capture and a long run's middle is elided. `_without_relayed_lines` drops them from the end-of-run dump so each reaches the log once; the capture keeps them for the failure summary.
- **Timeout is an outcome.** `/backtest/status` answers `timed_out` with a `timeout` block: `limit_seconds` for everyone, plus `spent_micro`, `model_calls`, `billing_mode` for the owner only. Analytics stays `backtest_failed` with `error_category="run_timeout"`. `_status_snapshot_for` strips the owner fields unless `_slot_owned_by` holds. `_slot_visible_to` is deliberately weaker (for a built-in agent it admits the `session_id` the running branch publishes to every visitor), so never reuse it for the whole payload: the outcome is public on purpose, account-level facts are not.
- **Spend lookup cannot block.** `credits_service.sum_run_llm_spend` runs on a daemon thread joined for `TIMEOUT_SPEND_LOOKUP_SECONDS` (10): `except Exception` answers a query that raises, not one that never returns, and inline it stranded a concurrency slot behind a dead child. Design: `docs/superpowers/specs/2026-09-14-backtest-timeout-outcome-design.md`.

### Ledger tables and who is billed

Reservations go to `credit_llm_reservations` and settled usage to `credit_llm_usage_entries` (`domain/credits/repository_postgres.py`); never `credit_ledger_entries`, whose `entry_type` CHECK admits only purchase / refund / admin_grant_assign / admin_grant_reclaim. One reservation is one provider attempt (unique on user, run, `call_index`, `attempt_index`). The handoff is built by `create_execution_handoff` (`infrastructure/llm/execution/handoff.py`); BYOK's `finalize_run` returns `[]` and never touches the ledger. Design: `docs/superpowers/plans/2026-08-24-unified-llm-execution-layer.md`.

Research runs reserve `RESEARCH_RUN_ESTIMATE_USD_CENTS` (default 100) and settle at agent-reported usage, else the estimate. Other paths billed to the operator's keys rather than the user's Credits: leaderboard entries through their `dashboard/config/leaderboard.json` `integration`, Discord `/ask` (`domain/chat/service.py`), Trading Algo chat and backtest (`/api/algo/*`; the backtest is hourly Claude decisions, timed out at 60 minutes in `domain/backtesting/algo_service.py`), the AI Hedge Fund bridge (`OPENROUTER_API_KEY`) and the switched-off Robinhood path.

### Pinned sampling policy

Policy lives on `CatalogModel.sampling` (`domain/model_providers/execution_catalog.py`); the field is required, no default, because a defaulted temperature 0 on an OpenAI reasoning model 400s every call. Claude Haiku 4.5 / Sonnet 4.6: temperature 0. GPT-5.5: reasoning effort `low`, no temperature. Gemini 3.1 Pro: `PROVIDER_DEFAULT` (sends nothing; Google advises against low temperatures on Gemini 3, revisit once measured). DeepSeek V4 Pro / Qwen3.7 Plus: `PINNED_NO_THINKING` (temperature 0, `reasoning_effort: "none"`), which `infrastructure/llm/execution/adapters/openai.py` (not the legacy `infrastructure/llm/adapters/openai.py`) sends as `thinking: {type: "disabled"}` to CommonStack only (`THINKING_TOGGLE_PROVIDERS`, keyed on provider id; an admin-registered OpenAI-shaped server keeps the `reasoning` shape).

- **Applied in one place.** `AnthropicCompatibleExecutionClient` (`infrastructure/llm/execution/client.py`) resolves the policy from the signed handoff's `model_id` (`sampling_policy_for`) and overrides what the caller passed, so pipeline steps, the truncation-recovery retry, failover and post-trade review all carry it. Do not thread a copy through argv, the engine or `pipeline_runner`: it can disagree with the signed model id. A run without a handoff pins nothing and records `provider_default`.
- **Recording.** Adapters report the controls actually sent (`AdapterResponse.sampling_wire`, e.g. `temperature=0.0;thinking=disabled`). `sampling_record()` feeds `agent_runs.metadata.llm_sampling`: `policy` (`pinned_v1` | `provider_default`; `unresolved` for a model outside the catalog; `leaderboard_entry` derived at read time for leaderboard rows), `temperature`, `reasoning_effort`, `model`, `wire` (provider id to controls, per lane that answered).
- **Surfacing.** `GET /api/backtest/runs*` returns `RunMetadata`, an allow-list. `_sanitized_llm_sampling` keeps the four scalar keys and a `wire` map capped at 8, and returns `None` (not `{}`, truthy in the browser) when nothing survives; a new key must be added to `_LLM_SAMPLING_FIELDS` or it never reaches the browser. `formatBacktestSampling` (`dashboard/frontend/app.js`) reads Pinned / Provider default / Entry config / Not recorded; `policy` decides the prefix, never the presence of values.

**Comparing two runs.** `dashboard/scripts/diff_backtest_runs.py <run_a> <run_b>` prints the first divergent bar, the divergent count and the final-equity gap, plus each run's recorded sampling. Pipeline-runtime runs write no `backtest_decisions`, so `basis` is `equity` and the decision fields are `None`, not `0`. Pinned temperature does not make providers deterministic (batching and mixture-of-experts routing vary the draw, and each bar's prompt embeds the previous answer), so the useful signal is a later and smaller divergence, not zero.

**Tests.** `tests/test_backtests_router.py` (preflight, claim recheck, timeout outcome, `billing_mode` threading), `tests/test_backtest_cancel.py` (relay and redaction), `tests/infrastructure/llm/test_platform_credits_env_fallback.py` (failover), `tests/domain/credits/test_promotion_grants.py` (welcome grant), `tests/test_execution_catalog_sampling.py`, `tests/infrastructure/llm/test_execution_client_sampling.py`, `tests/test_backtest_sampling_row.py` (sampling).

## Backtest progress phases and the two-clock contract

Expands the `CLAUDE.md` rule: read durations from `duration_seconds`/`imports_seconds`/`preflight_seconds` (monotonic), never difference `started_at`/`ended_at` (wall clock). Paths are under `dashboard/backend/domain/backtesting/` unless given in full.

### Phases

`PROGRESS_PHASES` (`engine.py`) = `starting`, `loading_bars`, `indicators`, `first_decision`, `running`, `saving`. `publish_phase` sets `loading_bars`, `indicators`, `first_decision` and `saving`; `starting` is opened by `_init_progress_phases` from the parent's `--launched-at` (a bare CLI run without it never opens one), and `running` by `_publish_live_progress` through `_set_progress_phase`. `/backtest/status` maps four to card sentences (`PROGRESS_PHASE_MESSAGES` in `api/routers/backtests.py`); `starting` and `running` are deliberately absent. Per-phase timings are in `phases[]` of the run's progress file, which the parent unlinks when the run ends (`run_backtest_background`, `finally`); afterwards use the child's `⏱  phase ...` stdout lines in the service log, which survive.

### The two-clock contract

- **Instants are wall clock.** `started_at`, `ended_at`, `child_entered_at`, `imports_done_at` exist to correlate with log lines and nothing else.
- **Durations are monotonic.** Every finished `phases[]` entry except `starting` carries `duration_seconds` from `time.monotonic` (`steady_clock` in `engine.py`); `starting` also carries `imports_seconds` and `preflight_seconds`. Never re-derive a duration by differencing `started_at`/`ended_at`: an NTP step distorts it and a backward step makes it negative.
- **The one exception is `starting`.** Its total (opened at the parent's `--launched-at`, closed in the child) and its `spawn+interpreter` split (`child_entered_at - started_at`) cross the parent-to-child boundary, and `monotonic()` has a per-process epoch, so these two stay wall clock. `_init_progress_phases` leaves the steady mark `None` for `starting`, and `_set_progress_phase` reads `None` as "use the wall clock".
- **In-child splits use the steady pair.** `imports+stores` and `preflight` both start and end inside the child. `dashboard/scripts/backtest_hourly_agent.py` reads `CHILD_ENTERED_STEADY` at the top of the script and `IMPORTS_DONE_STEADY` once imports finish, and hands them to the engine in `startup_clock` (`child_entered_steady`, `imports_done_steady`) next to the wall stamps and `schema_init_seconds`. `imports+stores` closes in the script but `preflight` closes in the engine, so the engine computes both from the raw marks, which keeps the two splits on one clock by construction. The raw marks are not published (meaningless across processes).
- **`schema_init_seconds`** rides on `starting`; `0.0` in a worker is evidence the flag fired, not a gap.
- **Silent fallback, so guarded.** A caller passing only the three wall stamps silently gets wall-clock splits, so `tests/test_backtest_launch_phases.py` AST-checks that the script still hands over the steady pair, both are `time.monotonic()` reads, and they bracket the imports. The startup clock is seeded only when `--launched-at` opens `starting`; a bare CLI run must not leak those keys onto `loading_bars`.

### `fetch_seconds`

`loading_bars` carries `fetch_seconds` via `record_phase_metric`, which attaches to the open phase, is dropped when none is open, and is cleared on every transition. On exit the phase prints `fetch <n>s | aggregate+verify <n>s`; with a warm bar cache the residual is the aggregation cost, measured. Tests must bound `fetch_seconds` by `duration_seconds`, never by `ended_at - started_at` (cross-clock).

**Tests.** `tests/backtesting/test_engine_progress_phases.py` (the cited contract: wall-step cannot distort in-child durations or splits, `starting` stays wall clock, startup keys do not leak, `fetch_seconds` split line), `tests/test_backtest_launch_phases.py` (steady-pair hand-over), `tests/test_backtest_progress_card.py` (card sentences).

## Robinhood live switch (`execution/`)

Expands the `CLAUDE.md` `execution/` rule: the real-money path is switched off by one constant; its routes stay mounted and imported, so do not delete or comment them out. Off-state behaviour and the restore checklist live only here (tests and comments that cite "CLAUDE.md, `execution/`" mean this section). Paths are under `dashboard/backend/` unless given in full.

**Switch.** `ROBINHOOD_LIVE_ENABLED = False` in `domain/backtesting/constants.py`, read at call time through `robinhood_live_enabled()`. Real capital is out of scope (`docs/superpowers/specs/2026-10-14-atl-direction.md`); off since 2026-10-22. One flag, not commented-out code, so the suite keeps exercising what a restore turns back on. Separate from `ROBINHOOD_EXECUTE` (`execution/robinhood_live_service.py`), which gates sending orders once the path is on and defaults off. The router `api/routers/robinhood_live.py` serves `/api/v1/robinhood` and calls that module, which holds the risk gate (`_risk_gate_orders`), the pre-trade review gate (`_review_blocks_order`) and the JSONL audit log under `dashboard/storage/audit/robinhood/`.

### Off-state behaviour

Pinned by `tests/test_robinhood_live_disabled.py`; the editor half runs under `node`, and its `*_when_switched_on` cases flip the flag.

- `require_robinhood_live` (`api/auth.py`, imported by `api/routers/robinhood_live.py`) answers 404, before authentication, on `POST /api/auth/robinhood/start`, `POST /api/auth/robinhood/complete` and `POST /api/v1/robinhood/agents/{id}/live-run`.
- `GET /api/auth/robinhood/callback` redirects to `/app?robinhood=disabled` without exchanging the code, so a link in flight at deploy obtains no tokens.
- `PATCH /api/v1/agents/{id}` refuses `live_trading_enabled: true` with 409 (`false` is still accepted; see `api/routers/agents.py`). It is the column's only writer, so a stale tab or an API client cannot re-arm an agent an operator cleared.
- **Two routes stay live on purpose.** `GET /api/v1/robinhood/status` reports the switch as `enabled` and skips the portfolio read; it is the agent editor's only copy of the switch (`robinhoodLiveEnabled` in `dashboard/frontend/js/agent-editor.js`; no JS mirror to flip, unlike `PAPER_TRADING_ENABLED`); signed-out, a failed check or a missing field all read as off. `DELETE /api/v1/robinhood/disconnect` lets an account linked before the switch delete its stored tokens (encrypted, in `broker_connections`, the durable store).
- **Editor.** The panel ships hidden; a linked account sees it with Disconnect alone; nobody else sees anything. Off, save omits `live_trading_enabled` from the PATCH (read as "leave alone"), so a save neither arms nor disarms.

### Restore checklist

In order:

1. Clear `external_agents.live_trading_enabled` first (`domain/agents/repository.py`): agents armed before the switch keep the flag, and a restore would bring them back armed.
2. Flip the constant.
3. Put back the two real-money sentences cut from the frontend: the second sentence of `agentCapitalRealMoneyNote` in `dashboard/frontend/app.html` (now "Every test here uses simulated money."; the full text was "Every test here uses simulated money. Real money is involved only if you explicitly connect a brokerage account and turn on live trading.", cut in `f3856487`), and the Community helper's "Live trading requires an explicitly connected brokerage account." (it followed "Performance is based on simulated tests and may vary between runs." in the Community view). Lengthen the pin `_CANONICAL_NO_REAL_MONEY_SENTENCE` in `tests/test_frontend_shelves.py` to match; the cut sentences have no pin of their own. The Community `shortened` string in `test_community_page_carries_the_no_real_money_sentence_once` is a substring of the restored helper, so it passes either way; lengthen it too if you want the restore pinned.
4. Rewrite the "switched off" text in `docs/source/lab/live_trading.rst` (admonition at the top), `docs/source/lab/operating_modes.rst` ("Live trading"), `docs/source/lab/architecture.rst` (the `/api/v1/robinhood/*` row), `.env.example` (Robinhood block header) and `docs/local-run.md` (the `ROBINHOOD_EXECUTE` bullet and the "Live trading" bullet), plus the agent-facing statements: `CLAUDE.md` (project blurb and the `execution/` bullet), `docs/architecture/configuration.md` (the Robinhood section), and the "switched off" comments in `api/auth.py`, `api/routers/robinhood_live.py` and `domain/backtesting/constants.py`. User-facing `docs/source/` and `README` edits go through the owner.
5. Turn the off-state cases of `tests/test_robinhood_live_disabled.py` into on-state ones.

After a restore, a signed-out visitor sees no panel (the editor learns the switch only from a signed-in status check); the old "Sign in to connect" prompt does not come back.

**Docs that deliberately still describe the path**, each saying it is switched off: `docs/source/lab/live_trading.rst`, its toctree entry in `docs/source/index.rst`, the see-also in `docs/source/lab/overview.rst`, the "Live trading" entry in `docs/source/lab/operating_modes.rst` (the live-board plan's rename text points at it), and the `/api/v1/robinhood/*` row of `docs/source/lab/architecture.rst`. The landing page, `docs/source/lab/key_features.rst`, the lead of `docs/source/lab/index.rst` and getting started leave it out; `README.md` mentions Robinhood only in its directory tree.

**Tests.** `tests/test_robinhood_live_disabled.py` (all of the above), `tests/test_frontend_shelves.py` (the no-real-money pins), `tests/test_app_composition.py` (the Robinhood route tuples).

## Persistence: twin parity and analytics tables

Expands the `CLAUDE.md` rule: every dual-backend store is a SQLite / `*_postgres.py` twin pair registered in `_TWINS`, and the value-analytics stores are built only through `build_value_analytics_store()`. Local runs use the SQLite twin, so a dialect mix-up fails on Postgres production alone. Paths are under `dashboard/backend/`.

### The four parity axes

`tests/test_store_twin_parity.py` is static (no live Postgres needed):

1. Every SQLite method exists on the Postgres twin with a matching signature.
2. Both declare the same columns, and the Postgres twin repeats every SQLite lazy migration (`CREATE TABLE IF NOT EXISTS` is a no-op on a deployed table, so a column added only to the `CREATE` never reaches production).
3. Each `CREATE INDEX` sits below the `ADD COLUMN` it depends on (an index created before its column killed a production boot).
4. `_DUPLICATED_BODIES` declares, per pair, which method bodies are identical across the twins (compared by `ast.unparse`). `test_duplicated_bodies_match_their_declaration` fails both ways: a declared body that stops matching (a fix applied to one copy only, which every other axis passes) and a body that becomes identical without being declared.

Consequences: adding a method to both halves of a pair can turn the build red by itself. Share the implementation (mixin or helper) or name the method with a reason in `_DUPLICATED_BODIES`; never delete the name to silence it. A new `*_postgres.py` must be added to both `_TWINS` and `_DUPLICATED_BODIES` here (`test_every_postgres_twin_module_is_registered`, `test_every_twin_declares_its_duplicated_bodies`) and to the separate list also named `_TWINS` in `tests/test_backtest_worker_schema_skip.py` (`test_the_guarded_list_accounts_for_every_twin`; see the DDL guard below).

### Value-analytics pair

`ValueAnalyticsStore` (`domain/analytics/value_repository.py`) and `PostgresValueAnalyticsStore` (`domain/analytics/value_repository_postgres.py`) are split by dialect. `build_value_analytics_store()` dispatches on whether the resolved `analytics_base` has `database_url`. Both constructors raise on a base of the wrong dialect instead of failing at first query.

### SQLite WAL and run metadata

`enable_wal` (`database.py`) is the one WAL helper, shared by `BacktestDatabase` (`database.py`), `RunStore` (`domain/runs/repository.py`) and the trace store; schemas are lazy and self-migrating. `agent_runs.metadata` (JSON) records the effective `LLM_MAX_OUTPUT_TOKENS` per run as `llm_max_output_tokens` (`domain/backtesting/engine.py`).

### Backtest-worker DDL guard (`ATL_BACKTEST_WORKER`)

Twelve of the thirteen `_TWINS` pairs in `tests/test_store_twin_parity.py` own a schema and route constructor DDL through `db_url.init_schema_unless_worker`; `PostgresValueAnalyticsStore` composes other stores and has no `_init_schema` (`_NO_OWN_DDL_TWINS`). The inline research store (`domain/agents/research_store.py`) uses the same helper and is in neither registry. All twelve are guarded, not only the ones a child constructs today, because which stores a backtest child imports is an accident of the import graph, reached by three routes: (1) the script imports `db` and the credits and model-provider services, whose modules build their stores at import (`model_providers/repository.py` also pulls in the agent and broker stores); (2) `domain/analytics/instrumentation.py` -> `analytics/service.py` builds `analytics_service` at import, which resolves the credits, provider, agent and run stores (`run_store` is a SQLite singleton, not a twin); (3) `trace_store` arrives at run time via `HourlyBacktester._create_market_data_provider` whenever the run has a `live_run_id`. One new module-level import in the engine's reach could add an unguarded payer with nothing going red, hence the AST pins in `tests/test_backtest_worker_schema_skip.py`: `test_no_postgres_twin_runs_ddl_outside_the_guard` (no non-test `*_postgres.py` calls any `self._...schema...()` from `__init__`), `test_a_twin_that_owns_a_schema_method_routes_it_through_the_guard`, and `test_the_guarded_list_accounts_for_every_twin` (guarded plus exempt equals `_TWINS`). A new Postgres twin must call `init_schema_unless_worker` from `__init__` and be added to the `_TWINS` list in that test file; `schema_init_seconds` on the `starting` phase is covered under "Backtest progress phases and the two-clock contract".

### Analytics tables and retention

- `user_activity` is current state and is never swept.
- `analytics_events` (raw events) keep 180 days (`RAW_EVENT_RETENTION_DAYS`, `domain/analytics/retention.py`), deleted in bounded batches by the retention service.
- `user_daily_facts` and `lifecycle_transitions` use the same horizon via `expire_facts_for_date` and are rolled up into `analytics_daily_rollups` before they expire; the facts and the transition log expire together.
- The anonymous `analytics_daily_rollups` are kept indefinitely (nothing deletes them; a rollup pass only replaces one day's rows).
- Admin access rows keep 365 days (`ADMIN_ACCESS_RETENTION_DAYS`).
- The five-state snapshot tables and `domain/analytics/maintenance.py` are deleted. A read path that wants "state" reads yesterday's fact or computes it live, never a snapshot (`test_deleted_five_state_module_is_not_importable` in `tests/test_architecture_boundaries.py`).

**Tests.** `tests/test_store_twin_parity.py` (axes, registration, fix guidance in its failure message), `tests/test_architecture_boundaries.py` (deleted snapshot module).

## Capital Allocation scope

Expands the CLAUDE.md rule that `PortfolioService` sums agent sleeves over an `AgentScope` (`api/routers/portfolio.py::_scope`). Paths are under `dashboard/backend/`.

- `user_portfolios.cash_available` stays account-scoped: it is keyed by user, so persisting the scoped figure would make two browsers of one account rewrite it forever. `_split_allocations` (`domain/portfolios/service.py`) returns `(owned, visible)`: `owned` is persisted, `visible` returned, the gap reported as `portfolio.unclaimed_allocated` (`tests/test_portfolio_scope_sync.py::test_persisted_cash_cache_stays_account_scoped`). A new drift field ships with its UI reader.
- Starter provisioning (`domain/agents/service.py::provision_starter_agents`) stays account-scoped on purpose: the claim has not run yet, and scoping it would let a returning guest's large sleeve starve the three starter agents.
- Agent grid page size is MEASURED off the CSS column ladder (never add `grid-template-rows` or mirror the ladder in JS; see the warning in `dashboard/frontend/styles.css`); the grid note counts legend-listed agents only (`tests/test_agent_grid_pagination.py`, `tests/test_allocation_panel_grid_note.py`).

## Repeat LLM runs: median and sample band

Expands the `CLAUDE.md` rule that a leaderboard LLM entry publishes one run that the model genuinely drove. Paths are under `dashboard/backend/domain/leaderboard/` (`service.py`) unless given in full.

**Why.** One run is one draw: repeats of a pinned model on identical inputs have spread widely in measured return (run data from the #602/#604 work, not checkable from code). So an entry with at least 2 comparable repeats publishes their median run, with curve, metrics and rank all taken from that one row (`_median_run`; an even count has two middle runs, and the tie breaks on the draw, not the return, because the pool is fixed before any run produced a return).

**Storage and deploy.** Repeats are stored under `LEADERBOARD_SAMPLE_MODE = "leaderboard_sample"`, a separate mode so every lookup keyed on `LEADERBOARD_MODE` stays blind to them. Deploy with `dashboard/scripts/deploy_leaderboard_model.py --entry <id> --samples N [--force]`.

**Payload.** The entry carries a `samples` block (`count`, `min_return`, `max_return`, `returns`) and `sample_band` (`_sample_band`: an index-parallel `lower`/`upper` envelope over the pooled runs' chart curves, plus `runs`). The chart shades the band behind the median and the table prints "Median of N runs · lo to hi".

**One pool.** `_vetted_pool` forms the pool once, so the median, the label and the band always stand on the same runs.

- The pool's seed is the largest group of recorded seeds (`_seed_refused`, tolerance `_SEED_MATCH_TOLERANCE = 0.01`, issue #365); its days are the span most seeded rows cover.
- A row with another recorded seed, other trading days, or no stored equity is dropped with a once-per-process WARNING (`_pool_member_fault`, `_warn_on_ignored_samples`).
- Samples with no recorded config, or of different configs, never pool.
- A repeat pool yields to a primary that matches the current config better; the primary joins the pool only if it passes the same fault check.
- The same seed guard omits any run recorded at a different seed than the board's base (an entry seeded at $10k does not appear on a $100k board).

**Tests.** `tests/test_leaderboard_samples.py` (pooling, median, tie-break, config mismatch), `tests/test_leaderboard_band_frontend.py` (chart band and label), `tests/test_leaderboard_curve_integrity.py` (curves).

## LLM reply parsing: JSON-or-nothing

Expands the `CLAUDE.md` rule that the portfolio manager enforces order limits in code after parsing, and the H6 guard that `llm_decisions` counts only steps the model genuinely drove. Paths are under `dashboard/backend/`.

**Parsing.** `parse_llm_response` (`infrastructure/llm/backtest_harness.py`) strips markdown fences, slices from the first `{` to the last `}`, repairs with `fix_json_formatting` and bracket balancing, and returns `None` otherwise. A truncated or unclosed-envelope reply first gets one recovery request (`RECOVERY_MAX_OUTPUT_TOKENS`, defined in `infrastructure/llm/pipeline_runner.py`; `truncation_reason` decides it was truncated) in `make_trading_decision_with_llm` before it counts as `None`. Exactly one, never a loop.

**What a `None` or unusable reply costs** (`domain/backtesting/portfolio_manager.py::make_trading_decision_with_llm`, `_absorb_strict_llm_failure`) depends on the run type:

- **Billed run** (execution client, `fail_closed=True`, `strict_llm=True`): the `except LLMDecisionError` arm re-raises, so the run aborts on the first one. The one failure that spends a strike instead is a transient provider outage (`is_transient_provider_failure`, `infrastructure/llm/execution/errors.py`), and that strike holds the step (no actions); it never trades rule-based.
- **Strict run without a handoff** (`strict_llm` with no execution client: an iFinD A-share `llm` run from the CLI, set in `HourlyBacktester.__init__` in `domain/backtesting/engine.py`): a strike trades that step rule-based, except a transient provider outage (`is_transient_provider_failure`), whose strike holds the step as on a billed run (`_absorb_strict_llm_failure(..., hold=True)`). Budget = `int(total_steps * STRICT_LLM_MAX_FALLBACK_RATIO)` (0.02), with 1 strike from `STRICT_LLM_MIN_STEPS_FOR_ONE_STRIKE` (25) steps; past it the run aborts. The budget must stay inside the leaderboard H6 coverage floor (`MIN_LLM_DECISION_COVERAGE`, 0.95, `domain/leaderboard/service.py`), or a run could spend its whole budget and still be refused at publish (`test_strict_llm_budget_clears_h6`). Strikes do not increment `llm_decisions`.
- **Non-strict run**: `None` means no actions that step; a valid empty `actions` list means a rule-based fallback.
- `strict_llm` is forced False for non-pipeline runtimes (`engine.py`).

**Tests.** `tests/llm/test_backtest_harness.py` (parse shapes, one recovery attempt), `tests/backtesting/test_portfolio_manager_move.py` (strict paths and `test_strict_llm_budget_*`), `tests/domain/leaderboard/test_deploy_guard.py` (H6).

## Unified run lifecycle

Expands the `CLAUDE.md` rule that everything passed to `register_reaper_sweep` runs inline on the one reaper tick. v1 (`/backtest`, `/api/v1`) and v2 (`/api/v2`) runs share one `protocol_runs` ledger. Paths are under `dashboard/backend/`.

- **One cap.** Active-run caps (`MAX_ACTIVE_RUNS_PER_AGENT`, 5; `MAX_ACTIVE_RUNS_GLOBAL`, 100) are checked under the single `_create_lock` in `domain/runs/service.py`, which `api/v2/runs.py` imports, so one create per surface cannot race past the combined cap.
- **One reaper.** One thread ticks every `RUN_REAPER_INTERVAL_SECONDS` (60; `REAPER_INTERVAL_SECONDS` in code). It drains abandoned runs and heartbeats live ones. Other surfaces add per-pass work through `register_reaper_sweep()`; `app.py` registers `reap_v2_runs` and `sweep_terminal_sessions`. There is no per-sweep interval, so every sweep shares the 60 s tick. The analytics daily job is not a sweep, on purpose (`domain/analytics/daily_job.py`).
- **Multi-worker recovery.** `protocol_runs` has `owner_instance` and `heartbeat_at` columns (`domain/runs/repository.py`). A run with no heartbeat for `RUN_HEARTBEAT_STALE_SECONDS` (300; 0 disables) is failed by any instance's reaper (`fail_stale_runs`), and a legacy row with a null heartbeat is judged by `updated_at`, then `created_at` (`COALESCE(heartbeat_at, updated_at, created_at)`), so it is not stale on that ground alone. `RUN_RECOVERY_ON_STARTUP` (default on) fails all non-terminal rows at boot and must be `0` for multi-worker or rolling deploys, or a booting instance fails its neighbours' live runs.
- **Tombstones.** A terminal v2 run is replaced by a DB-backed `ArchivedBacktestBackend` (`execution/backtest_backend.py`). Step and idempotency state survives restarts, and v2 `cancel`/`status` report the true terminal status. 

**Tests.** `tests/test_run_lifecycle_unification.py`, `tests/test_global_run_cap.py`.

## Indicator warm-up pad

Expands the `CLAUDE.md` rule that a backtest's first bar must see warm indicators without the pad ever trading. Owner: `dashboard/backend/domain/backtesting/indicator_warmup.py` (its module docstring has the rationale). Paths are under `dashboard/backend/` unless given in full.

- **Fetch start.** `warmup_fetch_start(start)` (`infrastructure/market_data/provider.py`) = start minus `INDICATOR_WARMUP_CALENDAR_DAYS` (30 calendar days: enough for sma50 on A-share's 4 bars/day, and it also clears a week-long exchange closure). Used by `HourlyBacktester.load_data` (`domain/backtesting/engine.py`), the shared protocol and `/api/v2` dataset (`_build_dataset` in `domain/backtesting/market_data_store.py`) and the boot bar-cache warm (`infrastructure/market_data/bar_cache_warm.py`, same function so cache keys match).
- **Split before anything trades.** `split_at_start` cuts every frame at `start_date` before aggregation and before anything trades, values or gates. Pad bars reach the indicators only, through `warm_indicators`. The A-share market-rule and corporate-action check sees only the traded window.
- **Depth floor.** iFinD's per-symbol depth floor counts from `depth_start=start_date`, so pad bars cannot clear it. Engine only, and `IFIND_ASHARE` only (`_build_dataset` is Alpaca-only and never passes it). It is offered only to a `fetch_bars` that declares the keyword (`_accepts_keyword(..., via_var_keyword=False)`), never through a `**kwargs` wrapper. The adapters count the same way (`infrastructure/market_data/ifind_adapter.py`, `ifind_ashare.py`).
- **A pad can be worse than none; both cases are recorded.**
  1. *Price-scale break.* Prices are unadjusted on both feeds, so a split or 除权除息 inside the pad poisons the window's sma50 and bands. `trim_at_unadjusted_gaps` cuts each symbol's pad at its last close-to-close break (more than 21% for CN, `CORPORATE_ACTION_GAP` in `infrastructure/market_data/ifind_market_rules.py`; more than 30% for US, `_DEFAULT_GAP_THRESHOLD` in `indicator_warmup.py`; a break on `start_date` drops the pad).
  2. *Thin pad* (suspension, recent listing). `warmup_evidence` writes `agent_runs.metadata.indicator_warmup`: `fetch_start`, `bars_for_warm_start` (49, `PAD_BARS_FOR_WARM_FIRST_BAR` = `LONGEST_LOOKBACK_BARS` 50 minus 1), `min_pad_bars`, `short_symbols` ({symbol: pad bars it got}) and `unadjusted_gap_trims`. The requested start alone reads identically whether history arrived or not. `MarketDataset.indicator_warmup` carries the same block.
- **Window end.** The bar window ends at `settled_exclusive_end` (`infrastructure/market_data/provider.py`): the inclusive `end_date` plus one day, never past the start of the market's current day. When it stops at today, `window_provenance` records `open_session_excluded`. The bound is resolved once per run (`HourlyBacktester.provider_end_date`) and handed to the baseline worker via `BaselineJob.provider_end`, so the index baseline covers the same window. Malformed dates raise (`parse_ymd`). Rows since 2026-09-28 (commit `9bff1a94`) record `end_date_inclusive` and `provider_end_date` in `agent_runs.metadata`; older rows ended a day short of the same `end_date` label.
- **Agent rows only.** Baseline rows record nothing (no indicators).
- **Not padded.** The index-baseline refetch; `vnpy_simulation` (it scripts prices by bar index from the fetch start and seeds on the requested window, so a pad would move the scripted path); and hosted runtimes (`ai_hedge_fund` loads its own lookback). `_wants_indicator_warmup` is True only for the pipeline runtime on a non-vnpy source.

**Tests.** `tests/backtesting/test_engine_indicator_warmup.py`, `tests/test_market_data_store_warmup.py`, and the depth-floor cases in `tests/infrastructure/market_data/test_ifind_adapter.py` and `test_ifind_ashare.py`.

## Live board refresh

Expands the `CLAUDE.md` rule that the Live GET is read-only and only `refresh_live_leaderboard` writes. Paths are under `dashboard/backend/` unless given in full.

- `.github/workflows/live-leaderboard.yml` only POSTs `/api/v1/leaderboard/live/refresh` and exits on the 202 (runs last 6-8 s), so a green run proves only that the work was enqueued. GitHub starts the `30 22 * * 1-5` cron hours late (01:28-02:01 UTC observed in late October 2026), so check after that before calling a night missed.
- The work runs on a daemon thread in the Render process (`domain/leaderboard/live.py::_run_live_refresh_background`). A redeploy mid-run kills it without an error. Failures are only log lines (`ERROR: live.model_deploy_failed entry=...` from `domain/leaderboard/roster_failures.py`, `Live leaderboard background refresh failed`), because `leaderboard_live_refresh.json` sits on the ephemeral disk.
- Diagnose from `live_status` on `GET /api/v1/leaderboard?period=live`: `snapshot_stale` (newest stored freeze is behind the clock's), `frozen_through` vs `printed_through`, `catching_up`, `models_pending`.
- The POST answers 503 when the month has no completed cash session yet (`live_freeze_config` returns `None`). An LLM entry under 95% decision coverage is an H6 refusal for that segment (`tests/domain/leaderboard/test_deploy_guard.py`).

## Legacy Season-0 preview (dead code)

Expands the `CLAUDE.md` leaderboard rule that there are two boards (Competition and Live) and no season engine yet. The two-week "Season 0" preview was superseded on 2026-10-14: seasons are now calendar months, and Season 1 waits for the new engine. **Do not revive it.** Delete code and pins together. Paths are under `dashboard/backend/domain/leaderboard/` (`service.py`) unless given in full.

**Dead on every served route.** `get_leaderboard` dispatches `period=live` to `live.get_live_leaderboard` through `_PERIOD_BOARDS` (`register_period_board`, called at the bottom of `live.py`), and raises `RuntimeError("Live leaderboard module is not loaded")` if it is unregistered, instead of building the contest-window preview. The router also calls `get_live_leaderboard` directly (`api/routers/leaderboard.py`). The served Live payload carries `live_status` and no `season` key (`test_only_the_live_board_carries_live_status`).

**Period coercion (live behaviour, keep).** `VALID_PERIODS` is `("contest", "daily", "live")` (`service.py`). `_normalize_period` strips and lowercases the value, and an unrecognised one is coerced to `contest` rather than answered with a 4xx, so a typo in `?period=` returns the Competition board with HTTP 200. The router sends `live` to `get_live_leaderboard` before this runs.

**Reached only by tests:** `resolve_leaderboard_config('live')`, `build_season_payload`, `season_window`, `_season_trading_days`, `_preview_season_dates`, `PREVIEW_SEASON_NUMBER`, the `season` block in `dashboard/config/leaderboard.json`, and every `isLiveBoard()` branch in `dashboard/frontend/js/leaderboard.js` (a comment there marks them unreachable). `app.js` only calls `loadLeaderboardData('contest')`; the Live tab renders through `dashboard/frontend/js/live-leaderboard.js`. `live.py` still hardcodes `phase_label` "Season 0".

**Pinned by** `tests/test_leaderboard_season.py` and `tests/test_frontend_live_trading_board.py` (a legacy file name; the board is the Live Leaderboard).

**If season chrome is ever rebuilt, keep these four invariants** (each is documented where it lives):

1. The "preview" banner keys off evidence of an advance (`seasonHasAdvanced` in `dashboard/frontend/js/leaderboard.js`: `season.last_advanced_date` or `trading_days_elapsed > 0`), never `period === 'live'`. Knowing the word "live" says nothing about whether a session ran.
2. Season 0 is falsy. Read the number through `displayedSeasonNumber()` with `Number.isFinite`, and print it through `displayedSeasonLabel()`; the payload's `season.number` is the single owner.
3. An elapsed or unparseable season window yields `(None, None)`, never the contest window (`season_window` and its helpers in `service.py`).
4. Season length is untrusted config on a public GET: clamp once to `1..MAX_SEASON_TRADING_DAYS` (260) and publish the clamped value.
