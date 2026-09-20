# A Backtest You Can Watch Start and Run Twice — Design

**Follows:** [#474](https://github.com/Open-Finance-Lab/AgenticTrading/issues/474) (items 1, 2, 5 shipped or on `feat/474-backtest-timeout-outcome`), and the 2026-09-20 architecture diagnosis recorded in the session memory `backtest-latency-nondeterminism-diagnosis`.
**Scope:** `runtime_type="pipeline"` + `decision_source="llm"`, launched from `POST /backtest/run`. The hosted `ai_hedge_fund` runtime and the CLI-only paths are touched only where the same code runs.
**Status:** approved in brainstorming 2026-09-20. One decision changed after the code was read and is marked **⚠ changed**.

## The three symptoms, and which this closes

A user launching the onboarding backtest today gets, in order:

1. **A dark start.** The card says "Starting backtest…" and nothing else until the first hourly bar has finished its whole pipeline. Behind that sentence the child cold-imports pandas and three SDKs, four module-level store singletons each open a fresh connection to Neon and run their schema DDL in `__init__` (`database_postgres.py:59`, `credits/repository_postgres.py:665`, `analytics/repository_postgres.py:195`, `model_providers/repository_postgres.py:154`), Alpaca bars are fetched and aggregated with a Python `iterrows` loop (`domain/backtesting/bar_aggregation.py`), and then bar 1 runs every pipeline step. The only progress write in the engine is `_publish_live_progress` (`engine.py:588`), called only inside the bar loop (`:1621`). Nothing before that is visible, and nothing before that is measured.
2. **A long run.** Wall-clock is bars × decision steps × one blocking model call. That floor is architectural and out of scope here.
3. **Results that do not reproduce.** The pipeline request builder `_create_pipeline_response` (`infrastructure/llm/pipeline_runner.py:505`) sends `model`, `max_tokens`, `system` and `messages`, and nothing else. No temperature, no reasoning effort. The worker client already accepts and validates both (`execution/client.py:102-122`) and every adapter forwards temperature (`adapters/anthropic.py:85`, `gemini.py:60`, `openai.py:113`). The engine passes neither on either branch (`engine.py:1483`). Each bar's prompt embeds the previous bar's sampled action, so a 49-bar run is 49 dependent samples with no re-anchoring.

This design closes **1** (the start is visible, and cheaper where the cost is waste) and **3** (sampling is pinned, recorded, and shown). It does not close **2**; see *Out of scope*.

## What the user sees afterwards

- Within one poll tick of launch the card names a phase with an elapsed clock: *Loading 7 days of market data…*, *Calculating indicators…*, *Waiting on the first model decision…*, then the existing *Backtest running… step k/N*. The progress bar goes determinate as soon as the bar count is known, before the first model call.
- The results panel carries a **Sampling** row: *Pinned · temperature 0*, or *Pinned · reasoning effort low (this model ignores temperature)*, or *Not recorded* for runs written before this shipped.
- The same configuration run three times produces curves that diverge later and by less. The number is measured and recorded in the plan, not promised on the card. **The card never says "deterministic".** Providers are not deterministic at temperature 0, mixture-of-experts models least of all; what is pinned is the request, and what is shown is exactly that.

## Track A — the start is visible, then cheaper

### Progress contract

The child already owns the progress file (`--progress-file`, rewritten whole on every step; the reader in `_read_progress_file` tolerates a torn write by returning `None`). Track A adds a second writer beside `_publish_live_progress`, `_publish_phase(name)`, and three fields to the payload:

| field | type | meaning |
|---|---|---|
| `phase` | `"starting" \| "loading_bars" \| "indicators" \| "first_decision" \| "running" \| "saving"` | the phase in progress |
| `phase_started_at` | float, epoch seconds | when the current phase began |
| `phases` | list of `{name, started_at, ended_at}` | every finished phase, in order |

`step` and `total_steps` keep their meaning. A phase payload carries `step: 0` and, once `load_data` has built the decision bars, the real `total_steps`. The frontend's `deriveRunningProgress` already requires `step > 0` before it anchors an ETA (`app.js:7853`), and `updateLiveBacktestChart` / `updateLiveTradingLog` return early on a payload with no curve or no orders (`app.js:8776`, `:9149`), so pre-loop payloads are safe for every existing reader. `_publish_live_progress` sets `phase: "running"` and appends the finished pre-loop phases so the history survives the first step.

Phases are marked by the engine, not the script: `load_data` marks `loading_bars`, `calculate_indicators` marks `indicators`, `run_agent_backtest` marks `first_decision` before bar 1 and the first `_publish_live_progress` flips it to `running`; `save` marks `saving`. The script's `main()` prints the same phases to stdout with elapsed seconds, so a CLI run measures itself the same way.

### What the reader says

`/backtest/status` (`api/routers/backtests.py:3458-3476`) currently derives its sentence from `step/total_steps` and otherwise falls back to *Backtest is running… (multi-step agent pipeline; may take several minutes)*. It gains one phase → sentence table, and the card renders the same sentence with its existing elapsed clock. The message is built server-side for the same reason the staleness age is: both ends of the phase clock are read in one process.

### Where the start is waste

Two costs are pure waste and are removed; the rest is measured first.

**Schema DDL in the child.** The parent runs every store's DDL at import, so by the time it spawns a child the schema exists. The parent sets `ATL_BACKTEST_WORKER=1` in the child's environment (the `env` copy at `backtests.py:1540`), and each Postgres twin skips `_init_schema()` when it sees it, printing one line (`<store>: schema init skipped (backtest worker)`). The SQLite twins are unchanged: their DDL is local and fast, and tests construct them against fresh temp files. **The variable names the process role, not the DDL**, so a human never sets it globally, an older parent that does not set it gets today's behaviour, and future child-only behaviour (turning the analytics snapshot projection off, Track C) has a home.

**Aggregation.** `aggregate_bars` walks every 5-minute source bar with `iterrows`. It is vectorised (pandas `resample` on the session-aligned index) **only if** the measured `indicators`-plus-`loading_bars` phase exceeds two seconds locally with aggregation dominating. The plan carries the gate; a change the measurement does not justify is not made.

Everything else in the start (SDK imports, the Alpaca fetch, Neon connection setup) is measured by the phase payload and left alone in this release.

### Measurement

The phase payload *is* the instrument. Locally, one pipeline+LLM run with the OpenRouter and Alpaca keys in `dashboard/.env` records every pre-loop phase. The one cost that cannot be measured locally, Neon DDL and connection setup, is measured by the first prod run after Track A deploys, from the same payload.

## Track B — sampling pinned, recorded, shown

### ⚠ changed: the policy is a per-model table, not a flag

The brainstorm said "models flagged `reasoning` in the catalog". The flag is on **`ProviderCapabilities.reasoning`** (`domain/model_providers/models.py:24`), a per-provider capability, and the execution catalog (`domain/model_providers/execution_catalog.py:27`) is six `CatalogModel(catalog_id, label, vendor)` entries with no per-model capability at all. A provider-level flag cannot say whether *this* model rejects temperature. The policy therefore lives on the catalog entry:

| catalog model | temperature | reasoning_effort | why |
|---|---|---|---|
| `anthropic/claude-haiku-4-5`, `anthropic/claude-sonnet-4-6` | 0 | — | the Anthropic adapter never enables extended thinking, so temperature is accepted |
| `openai/gpt-5.5` | — | `low` | OpenAI reasoning models reject a non-default temperature |
| `google/gemini-3.1-pro-preview` | 0 | — | temperature accepted; the Gemini adapter has no thinking control |
| `deepseek/deepseek-v4-pro`, `qwen/qwen3.7-plus` | 0 | `low` | OpenRouter forwards both; the model ignores temperature while thinking and the effort bound is what tames the ceiling |

`CatalogModel` gains a `sampling: SamplingPolicy` field (`temperature: float | None`, `reasoning_effort: str | None`) with the table above as the values. A model not in the catalog cannot be launched from the dashboard (`preflight_execution_model`, `backtests.py:3216/3235`), so there is no default row. The CLI path (`--use-llm` without a handoff, real Anthropic SDK) defaults to temperature 0 and no reasoning effort.

### The wire

The parent resolves the route it already preflights and reads the policy off it (`ExecutionModelRoute` gains the same `sampling` field). The values ride argv: `--llm-temperature 0` and `--llm-reasoning-effort low`, omitted when `None`. They are not secrets and do not belong in the signed handoff envelope. From there:

```
backtest_hourly_agent.main() → HourlyBacktester(llm_temperature=…, llm_reasoning_effort=…)
  → PortfolioManager.make_trading_decision_with_llm(temperature=…, reasoning_effort=…)
    → run_pipeline_decision(…, sampling=…)            # pipeline branch, :462
    → _request_trading_decision(…, temperature=…)      # single-prompt branch, already threaded
      → _create_pipeline_response(…)                   # passes each value only when not None
        → client.messages.create(…, temperature=…, reasoning_effort=…)
```

*Only when not None* is load-bearing: the legacy Anthropic SDK client on the CLI path rejects an unknown `reasoning_effort` kwarg. `_retry_with_recovery_budget` carries the same sampling, because it is the same request at a higher ceiling. The single-prompt branch receives the same values, so one policy governs both branches.

### What the adapters do with it

The OpenAI adapter already sends `reasoning_effort` for `openrouter` and `openai_compatible` providers as `extra_body.reasoning` (`adapters/openai.py:115-131`). It gains the native `openai` case, passing `reasoning_effort` as the top-level Chat Completions parameter, gated on the policy so a non-reasoning OpenAI model is never sent it. The Anthropic and Gemini adapters keep ignoring `reasoning_effort`; no catalog row asks them for it. Metadata records what was **requested**; the adapter's filtering rule is documented in code and pinned by tests, not re-recorded per call.

### Persistence and display

The engine writes to `agent_runs.metadata` beside `llm_max_output_tokens` (`engine.py:1052`):

```json
"llm_sampling": {"temperature": 0.0, "reasoning_effort": null, "policy": "pinned_v1", "catalog_id": "anthropic/claude-sonnet-4-6"}
```

`renderBacktestRunConfig` (`app.js:9646`) adds the Sampling row from this block. Absence renders *Not recorded*, following `test_backtest_run_provenance.py`'s rule that an unrecorded field is unknown, not an accusation.

### Proof

A dev script, `dashboard/scripts/diff_backtest_runs.py <run_id_a> <run_id_b>`, reads `backtest_decisions` for both runs and prints the first divergent bar, the count of divergent bars, and the final-equity gap. Verification runs one configuration three times before Track B and three times after, locally, and the plan's final section records the table. The expected outcome is *later and smaller* divergence, not zero.

## Failure behaviour

- **Progress write fails.** Logged, never fatal, exactly as `_publish_live_progress` behaves today (`engine.py:641`).
- **Child sees `ATL_BACKTEST_WORKER` but the schema is missing.** Impossible from a parent that imported the stores, and a stale image gets a loud psycopg error on the first query, which fails the run visibly. Accepted.
- **A provider rejects the pinned parameter.** The call fails as a provider error, the run fails visibly with the provider's message, and `llm_sampling` in metadata shows which rule caused it. A test walks every catalog row against every adapter type it can route to and asserts the pair is one the adapter is known to send.
- **`reasoning_effort` requested of an adapter that ignores it.** Nothing is sent; metadata says *requested*, the card says *Pinned · reasoning effort low*, and the adapter rule is where the truth lives. No catalog row currently triggers this.

## Testing

**Track A.** Engine: a fake loader drives `load_data` → `calculate_indicators` → the first bar, and the progress file is asserted at each phase (`phase`, `phases`, `total_steps` known before the first decision). Router: `test_backtests_router.py` asserts the sentence for each phase and the unchanged sentence for the legacy payload. Frontend: `test_backtest_progress_card.py`'s node harness renders each phase sentence and the determinate bar at step 0. Stores: one test per Postgres twin asserts `_init_schema` is skipped with the variable set and called without it, and `test_backtest_db_postgres.py`'s `@pg_only` case boots a twin under the flag against a live schema.

**Track B.** `test_pipeline_runner.py` asserts both attempts of a step carry the sampling, and that `None` values are absent from the kwargs. `test_execution_client.py` already pins validation. A router test asserts argv per catalog row. A catalog test pins the policy table. An adapter test asserts the native OpenAI case sends `reasoning_effort` only when asked. An engine test asserts `llm_sampling` in metadata. A node-harness test renders all three Sampling row states.

## Sequencing

0. Rebase `feat/474-backtest-timeout-outcome` onto main, re-pin the five cache-buster files, open its PR, land it.
1. **PR A** — Track A, branched from main after 0.
2. **PR B** — Track B, branched from main after A lands. Both tracks touch the status route, the card and run metadata; sequential PRs keep each diff reviewable and a regression in one from blocking the other.

## Out of scope, deliberately

- **Track C:** a per-bar deadline, dropping `PROVIDER_TIMEOUT` from backtest failover, and moving the analytics snapshot rebuild (`_emit_model_usage`, `execution/service.py:172` → a synchronous projection because the singleton is built with `project_snapshots=True`, `domain/analytics/service.py:244`) off the hot path. `ATL_BACKTEST_WORKER` is where that switch will live.
- **Track D:** re-run and compare. It changes what a result means.
- The decision-cadence door (`frequency.py` has one legal value) and the result-contract door (single curve vs N-run band). Both wait for the Track B measurement.
- Changing the modal's default model (`app.js:239`, DeepSeek V4 Pro). If the measurement shows the default route still truncates at reasoning-low, that is a separate product decision.
- Caching bars in `market_data_store`.

## Documentation

- `CLAUDE.md`: an environment bullet for `ATL_BACKTEST_WORKER` (child-only, set by the parent, never by an operator) and a paragraph under the LLM backtest billing bullet for the sampling policy.
- User-facing docs: `docs/source/lab/operating_modes.rst` and `key_features.rst` describe the backtest flow and should be checked for a progress or reproducibility claim after each PR ships. Not edited in this work; surfaced as a follow-up.
