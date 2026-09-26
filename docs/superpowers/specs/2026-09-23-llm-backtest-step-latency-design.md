# LLM Backtest Step Latency (#522) — Program and S1 Design

**Status:** approved in brainstorming 2026-09-23. This document splits #522 into sub-projects and fully
designs the first one (S1). S2–S5 are scoped here so the order is on record; each one gets its own
spec before any code.

**Amended 2026-09-26**, after a check against work done since.
- Draft PR #535 (`feat/commonstack-first`, 2026-09-25) already ships §5.3's Haiku allowlist
  migration, generalised as `commonstack-allowlist-v1`. S1b continues on that branch.
- The provider order is now an operator knob, `ATL_PLATFORM_PROVIDER_ORDER`, rather than a
  hard-coded tuple (§5.1). The billing hint copy is neutral for the same reason (§5.4).

**Amended again 2026-09-26**, after #535's review fix `ab8919b6`. §5.1, §5.3, §5.4 and §7 now
describe what #535 ships:
- an explicit `provider_id` is honoured only when it is in the configured order;
- only platform-only lanes move in `list_execution_options`;
- the worker's legacy `("openrouter",)` expansion is deleted, not left alone;
- each allowlist backfill names only the ids it introduced and skips an emptied allowlist;
- the billing hint names no provider.

**Issues:** #522 (the preflight is ~5× optimistic), #523 (the OpenRouter platform key is out of quota), #524 (a killed run loses
every bar). **Builds on:** `2026-09-20-backtest-speed-and-trust-design.md`. Its *Track C* named the
analytics rebuild this design removes, and its *Track B* is S3 below.

## 1. The problem, measured

A pipeline + LLM backtest runs in a subprocess with a fixed 3600s budget. #522's run
(`agent_20260922_202119_fc7bd724`, DeepSeek V4 Pro, 70 bars requested) was killed with **38 bars
done**, not the 47 its `model_calls` suggested. Its 47 calls were 38 first attempts plus 9 recovery retries. Per
bar: median 58s, mean 92s. The preflight (`PIPELINE_SECONDS_PER_LLM_CALL`, default 15) admitted it.

Measured from prod, read-only. `credit_llm_reservations` stamps every attempt. `analytics_events`
has microsecond `received_at` and token counts on `model_usage_recorded`:

| component of the median ~62s call | cost | root cause |
|---|---|---|
| synchronous analytics snapshot rebuild | ~31s: 6 events × ~5.1s | `instrumentation._recalculate_snapshot` falls back to `states.recalculate_user_snapshots` when no recalculator is registered. Only `app.py`'s startup registers the no-op, and the backtest child never runs that hook |
| model generation | ~25s | output-bound: ~2.2k input tokens, median ~1.4k output tokens |
| dead OpenRouter attempt + per-attempt setup | ~3–4s | OpenRouter is tried first and fails `quota_exhausted` on every call (#523) |

Four more facts come from the same data:

- **The rebuild is systemic, and #485 made it worse in the worker.** It appears across all 35
  platform-credit runs since 09-12 (2,392 inter-event gaps). The per-event cost went from ~2.8s to
  ~5.2s on **2026-09-17**, the day #485 merged. #485 fixed the web process and roughly doubled the
  cost in the one process it didn't touch.
- **Calls ≠ bars.** DeepSeek's reasoning used up the 2,000-token ceiling on 24% of bars (8 empty
  replies, 1 truncated). Each of those bars paid for a second full call at 4,096. That's **1.24 calls per bar**.
- **The SDK regenerates silently.** `adapters/openai.py` builds `OpenAI()` with the default
  `max_retries=2` on a non-streaming client with a 60s read timeout. A generation longer than 60s is thrown away and
  regenerated from scratch.
  - Only DeepSeek V4 Pro exceeds 55s: 0 of 659 Sonnet/GPT-5.5 calls do.
  - Nothing lands in 57–66s or 100–125s.
  - Tail calls only reach a normal token rate once one or two ~60s timeouts are subtracted.
  - This cost #522's run ~540s, 15% of its budget.
- **Commonstack already carries all the traffic.** Since 09-10: 1,194 calls settled on Commonstack and 2 on OpenRouter. All 1,338 OpenRouter attempts
  released `provider_quota_exhausted`.

## 2. Goals and non-goals

**Goals.**
- A default 70-bar pipeline run finishes inside the existing 3600s budget.
- ATL Credits spend Commonstack's remaining balance before OpenRouter's.
- The moment Commonstack runs dry is visible to the operator.

**Non-goals.**
- Raising the 3600s budget.
- Parallelism inside a run: the decision chain depends on the path taken, and no mature harness parallelizes it.
- Changing decision cadence.
- Batch APIs: their latency SLA is hours.
- Per-analyst fan-out: it multiplies calls.
- Silent neutral fallbacks: they contradict fail-visible.

## 3. Sub-projects, in order

| # | Sub-project | Gate | Expected effect |
|---|---|---|---|
| **S1a** | Unregistered analytics recalculator becomes inert | this spec | −~31s per call |
| **S1b** | ATL Credits route through Commonstack first; quota-exhaustion ERROR; Haiku 4.5 on Commonstack | this spec | −~3s per call; spends Commonstack credit first |
| S2 | Own the retry/timeout layer (no SDK regeneration on timeout) + per-run provider cooldown | own spec | −~15% of run budget on DeepSeek |
| S3 | Execute Track B of `2026-09-20-backtest-speed-and-trust-design.md` after a ~10-call probe; decide the first-attempt ceiling from the probe | own plan | fewer recovery calls |
| S4 | Per-bar timing in run metadata, then recalibrate the preflight **per bar** (with the calls-per-bar multiplier) | own spec | admission matches reality |
| S5 | Persist per bar, keyed on bar index, and resume by replay (#524) | own spec | a killed run keeps its bars |

**Order rationale.**
- S1 removes pure waste without changing any request shape.
- S2 comes before S3 so the Track B probe measures the model, not SDK regeneration.
- S4 is last among the latency items because the preflight must be calibrated against the harness that remains once the waste is gone.
- S5 is independent and can go in parallel once S1 lands.

**What S1 alone buys, on paper.**

| | total | per bar | 70 bars |
|---|---|---|---|
| #522's loop | ~3,500s for 38 bars | ~92s | — |
| S1 saves | ~34s × 47 calls ≈ 1,600s | | |
| after S1 | ~1,900s | ~50s | **≈3,500s: at the edge** |
| after S2 (−~540s of regeneration) | | ~36s | **≈2,500s** |

S1 is necessary, not sufficient. The goal needs S2 as well.

**Deferred, deliberately.** Decision cadence, result contract (single curve vs an N-run band), and
OpenRouter host routing (`provider.sort`, `:nitro`). The first two wait for the Track B measurement.
The third is moot while OpenRouter is not the primary lane.

**Related, separate workstream.** The 2026-09-25 default-trading-instruction work (central DB
`knowledge/atl-default-trading-prompt.md`) found two things S3 and S4 must account for:
- A rule-based, holdings-only instruction roughly halved seconds per bar and output tokens.
- CommonStack returns `response_invalid` on about 19% of calls.

That work changes the prompt, not the harness, so it does not touch S1.

## 4. S1a — the unregistered recalculator is inert

**Change.** In `domain/analytics/instrumentation.py`, `_recalculate_snapshot` returns without doing
anything when `_snapshot_recalculator is None`. The lazy import of `states.recalculate_user_snapshots`
is deleted. `register_snapshot_recalculator` is unchanged, and a registered callback still runs for
every snapshot-relevant event.

**Why flip the default instead of registering the no-op in the worker.** The 2026-09-20 spec
expected the switch to be keyed on `ATL_BACKTEST_WORKER`. That would fix the backtest child and
leave the same trap in every other process that never runs `app.py`'s startup: CLI scripts, a
future worker service. #485 was correct for the process it changed and made the one it didn't
see worse. A default that is expensive unless something disables it is inherited by every new entry
point. The web process has had synchronous projection off since #485, so no reader depends on it.
`user_analytics_snapshots` is refreshed only by the reaper's repair sweep, throttled to a 24-hour
staleness window, and a worker-emitted event is picked up the same way a web-emitted one already is.
The daily job writes `user_daily_facts`, not snapshots. So a worker-emitted event (`credits_*`,
`model_usage_recorded`, `backtest_*`) now reaches the snapshot up to about a day late, where before
it landed at once. That is accepted: the admin read paths move to `user_daily_facts` in PR B, which
also deletes the repair sweep.

**Unchanged.**
- `disable_synchronous_projection()` and its call at `app.py:334-343` stay. The call now states the
  intent explicitly instead of carrying the fix. The function's docstring and the `app.py` comment
  are updated to say so.
- `SNAPSHOT_RELEVANT_EVENTS` is unchanged.
- The `project_snapshots` guard in `_emit` is unchanged.

**Tests** (`tests/domain/analytics/test_instrumentation.py`).
- New: with `_snapshot_recalculator` set to `None` and a non-projecting service, emitting a
  snapshot-relevant event never calls `states.recalculate_user_snapshots`. The test patches it to
  record calls. This is the backtest child's exact state.
- `test_disable_synchronous_projection_makes_the_fallback_a_no_op` still passes. Its name and
  docstring are updated, since the fallback it neutralises no longer exists.
- `test_stored_event_recalculates_snapshot_best_effort` and `test_analytics_integration.py:235`
  register their own callbacks and are unaffected.

## 5. S1b — Commonstack first for ATL Credits

### 5.1 Order

- A new `platform_provider_order()` in `domain/model_providers/service.py` reads
  `ATL_PLATFORM_PROVIDER_ORDER`, a comma-separated list of provider ids. It defaults to
  `commonstack,openrouter`.
  - It is read per call, not at import, so a bad value can never kill boot.
  - Tokens are stripped and lower-cased, and duplicates are dropped.
  - An empty or unset value gives the default silently.
  - If any token fails `validate_provider_id`, the whole value is rejected. It logs one `WARNING`
    per distinct bad value and falls back to the default. A typo'd order must not half-apply.
  - It takes an optional `known_provider_ids` (the registry). With it, a well-formed but unknown id
    (`commonstak`) is rejected the same way, instead of silently dropping the lane it misspells.
  - A provider left out of the list is never a candidate. That is deliberate: it is the operator's
    way to take a drained or broken lane out without a deploy. An explicit `provider_id` in the
    request body is honoured first **only when it is in the configured order**: an API caller naming
    a pulled lane must not reopen it. If nothing is left, the route already answers 422
    (`backtests.py`, `if not provider_ids`), so the run fails visibly.
  - Changing it on Render takes effect at the next restart. Render restarts the service on any
    env change.
- `ModelProviderService.resolve_platform_execution_candidates` iterates
  `(preferred, *configured)`, where `configured = platform_provider_order(registry)`, and keeps an id
  only if it is in `configured`.
- `list_execution_options` moves **only platform-only lanes** (today just Commonstack): they go last,
  in the configured order. Every BYOK-capable provider, OpenRouter included, keeps repository order,
  so the env var cannot change the BYOK default `app.js` takes from `providers[0]`. Platform routing
  order is decided by the route, not by this list.
- BYOK is unaffected: Commonstack is not BYOK-enabled, and a BYOK run carries a single provider.
- Platform-credit runs send no `provider_id` (`app.js:10839-10841` sets it only for BYOK), so this
  tuple alone decides the order. The failover loop in `_execute_with_platform_failover` is unchanged:
  on an eligible failure (`_PLATFORM_FAILOVER_CATEGORIES`) it moves on to OpenRouter.

**Why the order matters beyond the ~3s.** Today Commonstack is spent first only because OpenRouter
is broken. The day someone tops up the OpenRouter key (the fix #523 asks for), all traffic moves to
OpenRouter and Commonstack's balance sits unused. The flip makes the spending order a decision
instead of an accident.

**Deleted:** the legacy `("openrouter",)` expansion in `_execute_with_platform_failover`
(`execution/service.py`). It widened a lone OpenRouter candidate to OpenRouter → Commonstack:
hard-coded OpenRouter-first, and blind to an order the route had rejected. The worker now treats the
route's candidate tuple as authoritative. A lone candidate is what the route decided, and it is not
widened.

### 5.2 Quota exhaustion is loud, once

In `_execute_with_platform_failover`, the first time a platform provider raises
`PROVIDER_QUOTA_EXHAUSTED` in a process, it prints:

```
ERROR: llm.platform_quota_exhausted provider=commonstack fallback=openrouter
```

- `fallback=` names the next candidate, or `none`.
- Printing is once per provider per process: a module-level set guarded by a lock. A backtest child
  is one run, so a drained Commonstack produces one line per run instead of one per call.
- The line is printed by the child and reaches the Render service log.
- BYOK never prints it. A user's own key running dry is the user's concern, and the call already
  fails with that message.

**Why an ERROR line and not a balance check.** Commonstack exposes no balance endpoint: `/v1/credits`,
`/v1/balance`, `/v1/billing/balance`, `/v1/account` and `/v1/user/balance` all return 404 as of
2026-09-23. Exhaustion can only be observed as a failed call.

### 5.3 Claude Haiku 4.5 on Commonstack

Haiku 4.5 is the only catalog model outside `COMMONSTACK_MODEL_ALLOWLIST`, so with OpenRouter dead
it has no working ATL Credits route. Three platform runs since 09-12 died after a single OpenRouter
attempt with no Commonstack candidate. Haiku is the only model that produces that shape (inferred:
those runs wrote no `agent_runs` row). Commonstack lists `anthropic/claude-haiku-4-5` at $1 / $5 per
million tokens. That equals what `pricing.price_for_model` already returns through the `claude-haiku-4`
entry, so no pricing-table change is needed.

- `COMMONSTACK_MODEL_ALLOWLIST` gains `anthropic/claude-haiku-4-5`, which covers fresh databases.
- **Existing databases need a migration.** The seed is `INSERT … ON CONFLICT (provider_id) DO NOTHING`
  (`repository_postgres.py:218`), and the admin UI passes `capabilities` through unedited, so
  changing the constant never reaches prod's row.
  - **Shipped on draft PR #535** as the `COMMONSTACK_ALLOWLIST_BACKFILLS` registry in
    `repository_common.py`, one `(migration_id, model_ids)` entry per addition. Its first entry is
    `("commonstack-allowlist-v1", ("anthropic/claude-haiku-4-5",))`. Both twins run it in
    `_init_schema`, after the seed. It follows the `openrouter-platform-key-v1` pattern
    (`model_provider_migrations`).
  - The pure helper `commonstack_allowlist_backfill(capabilities_json, model_ids)` appends only the
    ids **that migration introduced**. It must never append "whatever the allowlist now holds",
    because that would re-add every model an admin had removed from the live row. It keeps
    admin-added ids and their order, and bumps `updated_at`.
  - The next allowlist addition **appends a new entry** (`-v2`, …) naming only its new ids. A
    shipped entry is never edited.
  - It skips an **empty** stored allowlist. An empty allowlist routes nothing, so it is how an admin
    turns the lane off, and adding one model would turn it back on at boot.
  - It records itself even when nothing changed, so an admin's later removal survives a reboot.
  - It leaves an unreadable row untouched rather than writing default capabilities over it.
  - `test_backtest_worker_schema_skip.py`'s reason string for `repository_postgres.py` already
    names the backfill. The detection does not change, because the file already mutates data
    through the seed. A child can skip it for the same reason as the seed: the parent applies it
    at boot before spawning anything.
- **Before merge:** one probe call to Haiku through Commonstack. Commonstack's Anthropic lane once
  returned a canned greeting (`domain/chat/service.py:77`). Sonnet 4.6 has since served 12 pipeline
  runs through it cleanly. A canned greeting would fail JSON validation, which is not a failover
  category, so it could not leak into a curve.

### 5.4 Copy and developer docs

- The backtest modal's billing hint (`app.js`, `'ATL Credits automatically use OpenRouter first…'`)
  becomes *"ATL Credits cover the model calls. ATL picks an available provider automatically."* The
  order is now an env var, and a lane can be pulled entirely, so copy that names a provider or
  promises a fallback could go stale without a code change. `test_byok_backtest_frontend.py` pins the string and is updated with it.
- The `app.js?v=` cache-buster is bumped in `app.html`, and in every test that pins it. Find those
  with a grep, not a remembered count.
- `2026-09-01-platform-provider-auto-routing-design.md` gets a dated amendment line under its summary,
  pointing here.

## 6. Failure behaviour

- **Commonstack drains.**
  - The first quota failure prints the ERROR line, and the call fails over to OpenRouter.
  - While #523 stands, OpenRouter fails too, and the run fails visibly with the quota message.
  - Topping up OpenRouter before Commonstack drains keeps runs working.
- **Commonstack answers exhaustion in an unrecognised shape.** `adapters/base.py:204` maps HTTP 402
  and structured or plain-text balance messages to `PROVIDER_QUOTA_EXHAUSTED`. Commonstack's real
  out-of-credit response has not been observed.
  - A shape that maps to another failover category (`CREDENTIAL_INVALID`, `PROVIDER_UNAVAILABLE`)
    still fails over, but prints no ERROR line.
  - A shape that maps to a non-failover category fails the call.
  - Either way the run's error category names it. The first real exhaustion is the test; if the line
    did not fire, extend the mapping then.
- **Haiku on Commonstack returns garbage.** The call fails JSON validation (`RESPONSE_INVALID`),
  takes the existing recovery retry, and falls back per the strict gate. It never publishes as an
  LLM curve.

## 7. Testing

- **S1a:** see §4.
- **S1b, order:**
  - `platform_provider_order()`: unset gives `("commonstack", "openrouter")`. `" OpenRouter , commonstack,openrouter "`
    gives `("openrouter", "commonstack")`. A value with an invalid token gives the default and
    prints one `WARNING`, once per distinct value.
  - `test_execution_catalog.py::test_platform_candidates_prefer_openrouter_and_support_commonstack_only`
    is renamed (`…_prefer_commonstack_and_follow_the_env_order`) and inverted. With both keys,
    `("commonstack", "openrouter")`, and `preferred_provider_id="openrouter"` gives
    `("openrouter", "commonstack")`. Commonstack-only still gives `("commonstack",)`. OpenRouter-only
    gives `("openrouter",)`. With `ATL_PLATFORM_PROVIDER_ORDER=openrouter,commonstack`,
    `("openrouter", "commonstack")`. With `ATL_PLATFORM_PROVIDER_ORDER=openrouter`, Commonstack is
    excluded even with its key set, and `preferred_provider_id="commonstack"` gives
    `("openrouter",)`: the pulled lane stays pulled.
  - Haiku with both keys resolves to `("commonstack", "openrouter")`. That needs #535's allowlist.
  - `test_service.py::test_execution_options_keep_openrouter_ahead_of_commonstack` is renamed
    (`…_follow_platform_order_and_keep_byok_order`). With a BYOK provider whose name sorts after
    OpenRouter added, the order is `anthropic, gemini, openai, openrouter, xai, commonstack` under the
    default and under every env value tried: only the platform-only lane moves.
- **S1b, quota signal:** two consecutive calls that each fail `PROVIDER_QUOTA_EXHAUSTED` on
  Commonstack print exactly one line, naming the fallback. A quota failure with no next candidate
  prints `fallback=none`. BYOK prints nothing. The per-process set is reset in the fixture.
- **S1b, migration:** covered by #535's tests in both twins. Postgres runs in CI via `TEST_POSTGRES_URL`.
- **Pricing:** `price_for_model("anthropic/claude-haiku-4-5") == (1.0, 5.0)`.
- The whole backend suite passes (`pytest dashboard/backend/tests/`), with `node` on `PATH` so the
  frontend-harness tests run instead of skipping.

## 8. Rollout and verification

- S1a and S1b are separate PRs, shipped in that order, so one prod run can attribute each saving.
- The spec and the plan ride with S1a. S1b is draft PR #535, completed on its own branch.
- **After S1a deploys:** in the next platform run, consecutive `analytics_events` for one call
  (`credits_settled` → `credits_refunded`) are milliseconds apart, not ~5s.
- **After S1b deploys:** every `credit_llm_reservations` row with `attempt_index = 0` has
  `provider_id = 'commonstack'`, and OpenRouter rows appear only after a Commonstack failure.
- **Neither PR touches `PIPELINE_SECONDS_PER_LLM_CALL`.** Recalibrating it is S4, against the
  harness that remains.

## 9. Documentation

- **`CLAUDE.md`:**
  - Under the admin-layer section's note on the PR 0 burners, one sentence saying the instrumentation
    fallback is now inert by default, and why: the child never runs `app.py` startup.
  - Under the LLM backtest billing bullet, the Commonstack-first order and the quota ERROR line.
  - A new env bullet for `ATL_PLATFORM_PROVIDER_ORDER`, and `tests/conftest.py` strips it.
- **User-facing docs to update after S1–S2 ship** (not edited in this work):
  - `app.html:1305` promises a backtest takes "several minutes"; a real pipeline run takes ~60–90 minutes.
  - `strategy.html:170` may make a similar claim.
  - `docs/source/lab/operating_modes.rst` and `key_features.rst`, per the 2026-09-20 spec.
