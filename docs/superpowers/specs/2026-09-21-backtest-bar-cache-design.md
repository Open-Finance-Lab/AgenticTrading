# Backtest Bar Cache — Design

> **Status: DESIGN ONLY — nothing under `dashboard/` implements this.** Verified
> 2026-09-21: no `bar_cache` module exists, and `AlpacaDataLoader.fetch_bars`
> calls the Alpaca SDK directly with no cache in between. Written against
> `main` at `1167ae97` (PR #501, Track A).

**Goal:** a repeated `(symbol, window, timeframe, feed)` bar request is served from
disk instead of from Alpaca, across process boundaries, without ever serving a
truncated or wrong-tape window.

## 1. Why — the measured case, and the part of it that is *not* measured

PR #501 instrumented the pre-first-bar window of a dashboard backtest. Of the ~21s
before bar 1, **~86% is the `loading_bars` phase** — 18.15s rule-based, 18.67s on an
LLM run — against 2.6s for imports and store construction and 0.00s for the child's
schema DDL locally. `loading_bars` is exactly the body of
`HourlyBacktester.load_data` (`domain/backtesting/engine.py`, `publish_phase("loading_bars")`
through the next phase marker in `calculate_indicators`).

Three facts qualify that number, and the design turns on them.

**A. 18.15s is not the whole market-data cost of a run.** A default Mag7 run also
fetches **DJIA_30** for the index baseline (`engine.py`, the `fetch_bars(DJIA_30, …)`
call inside the index-baseline block) and aggregates those thirty symbols too.
`index_baseline_enabled=True` on both US profiles, so this is the ordinary path, not
an edge case. It runs **after** `publish_phase("saving")`, so it is outside the phase
Track A measured entirely. Users experience it as the gap between the last bar and
the result appearing.

**B. The fetch/aggregate split inside `loading_bars` is unknown and not obtainable
from what shipped.** `loading_bars` is one opaque phase name. The 4.13–4.58s
aggregation figure in the Track A plan is a *synthetic* benchmark whose dataset shape
is not the measured run's, so it cannot be subtracted. Aggregation does run inside
the phase — `aggregate_bars_by_symbol` is called from `load_data`, and `intraday_mode`
is true for the default US profile (5m source → 60m decisions) — but how much of the
18s it owns is not established. **This design deliberately does not depend on that
number, and §9 explains how shipping it produces the number.**

**C. The ceiling depends on which runtime real users run, and that is unmeasured.**
A rule-based run's `first_decision` is 0.32s, so market data is most of its wall
clock. An LLM run is 49 bars × ~14s per model call ≈ 11–12 minutes, where 18s is
~2.5%. The honest framing is therefore **time-to-first-bar, not total runtime**:
Track A made the opening wait legible, this makes it short. Settling the ceiling
needs a read of `agent_runs.runtime_type` / `decision_source` in the prod
`ATL-runs-main` database; it is not a blocker for this work and is filed as a
follow-up (§12).

## 2. Non-goals

Named so they do not drift in:

- **Caching the aggregated output** (fetch → aggregate → quality as one unit). That
  is a separate, larger change; §9 is the gate that decides whether it is worth
  doing at all.
- **Unifying `market_data_store` with `engine.load_data`.** The two are near
  line-for-line duplicates of one pipeline and have already drifted — the store
  hardcodes `market="US", timezone="US/Eastern"` where the engine passes
  `self.profile.market` / `self.profile.timezone`, and the store's key has no market
  dimension at all. Fixing that touches the Agent-Environment Protocol path and
  `/api/v2`, both shipped. Filed as a follow-up (§12), not done here.
- **`_find_cached_run`'s missing feed dimension** in `domain/leaderboard/service.py`
  — the same hazard class this design guards against, already shipped on a different
  cache. Filed, not fixed here.
- **The iFinD A-share path.** Excluded from v1; see §5.
- **Durability.** This cache is ephemeral by design; see §6.

## 3. Where it sits

A new module `dashboard/backend/infrastructure/market_data/bar_cache.py`, applied
*inside* `AlpacaDataLoader.fetch_bars`.

The seam is chosen for reach. There are five real `fetch_bars` call sites and all
five inherit the cache with no changes at the call site:

| call site | what it fetches |
|---|---|
| `domain/backtesting/engine.py` — `load_data` | the agent universe (the measured 18.15s) |
| `domain/backtesting/engine.py` — index baseline | `DJIA_30`, after `saving` (fact A above) |
| `domain/backtesting/market_data_store.py` — `_build_dataset` | the in-process protocol/v2 path |
| `domain/leaderboard/baselines.py` | the contest window |
| `infrastructure/market_data/alpaca_bars.py` itself | the >100-symbol batch recursion |

### Per-symbol, not per-request

One cache entry per symbol, not per request. The wrapper splits the requested symbol
list into hits and misses, fetches **only the misses** from Alpaca, and merges.

**Ordering matters and must be explicit.** The cache is resolved at the *top* of
`fetch_bars`, **before** the existing `len(symbols) > 100` batch-recursion branch,
and only the missing symbols are passed down to the existing logic. Placing it after
the recursion would run the cache twice per request (once in the outer call, once per
100-symbol chunk) and would let the chunking, rather than the cache, decide what is
fetched. With the cache first, the recursion sees a shorter list and behaves exactly
as it does today.

This is what makes fact A cheap: after a Mag7 run, the DJIA_30 baseline fetch finds
five of its thirty symbols (`AAPL`, `AMZN`, `GOOGL`, `MSFT`, `NVDA`) already on disk
and requests twenty-five. Per-request keying would miss that entirely, because the
symbol lists differ.

### Two things a cache hit must restore

A hit that skips these is a silent behaviour change, not a speed-up:

1. **`self.last_fetch`.** `market_data_store._build_dataset` reads it to verify the
   source timeframe with `evidence="fetch"`. A hit that leaves it stale silently
   downgrades that verification to the weaker `evidence="configured"` path. The
   wrapper stores the `last_fetch` dict alongside the entry and restores it.
2. **The `.attrs` stamps** — `FRAME_ATTR_FEED` (`alpaca_feed`),
   `FRAME_ATTR_SIP_FALLBACK` (`alpaca_sip_fallback`), `FRAME_ATTR_END_CLAMPED`
   (`alpaca_end_clamped`). `feed_provenance()` reads these back and the engine
   persists the result into `agent_runs.metadata`. **Verified empirically on
   pandas 2.3.2: `DataFrame.attrs` survives a `to_parquet`/`read_parquet`
   round-trip**, along with the index name and its `datetime64[ns, UTC]` dtype. No
   sidecar is needed for the stamps. The `last_fetch` dict is request-level rather
   than frame-level and is stored as a small JSON sidecar per entry.

On a mixed hit/miss request, cached frames and freshly fetched frames both carry
their own stamps, so `feed_provenance()` correctly reports `mixed` if they differ —
which, given the feed is in the key (§4), can only happen on the clamp/fallback axis.

## 4. The cache key

```
(symbol, start, end, source_timeframe, resolved_feed, SCHEMA_VERSION)
```

- **`source_timeframe`** is a *mutable instance attribute* set by
  `configure_source_timeframe`, not an argument to `fetch_bars`. It must be read at
  call time. It materially changes the bars returned for an identical
  `(symbol, start, end)`.
- **`resolved_feed`** comes from `_resolve_data_feed()`, which re-reads
  `ALPACA_DATA_FEED` per call rather than caching it at import. Any of the four
  supported values (`iex`, `sip`, `delayed_sip`, `otc`) is a distinct key.
  CLAUDE.md is explicit that curves priced off different feeds are not comparable;
  omitting the feed is the exact defect `market_data_store`'s own key has today.
- **`SCHEMA_VERSION`** is a module constant bumped whenever the stored layout
  changes, so a format change invalidates the whole cache at once instead of
  producing unreadable entries.
- **Symbols are keyed individually**, which sidesteps the order-sensitivity bug in
  `market_data_store._dataset_key` (`tuple(symbols)`, so the same *set* in a
  different order is a miss).

## 5. What must never be written — the safety core

The cache is only correct because of what it refuses to store. **An entry is written
only when every one of these is true:**

| refuse to cache when | why |
|---|---|
| the frame carries `alpaca_end_clamped=True` | A SIP request reaching into the last `ALPACA_SIP_DELAY_MINUTES` (default 15) has its `end` capped to now−15m. The **same requested window returns a shorter frame depending on when you ask.** Storing it under the full window's key makes that truncation permanent for the life of the instance. |
| the frame carries `alpaca_sip_fallback=True` | The IEX-on-refusal retry re-requests with the **original unclamped `end`** and never sets `end_clamped`, so the result looks pristine while being a different tape at ~2.5% of volume. |
| the window has not **settled** — its `end` is less than 24 hours in the past | The two flags above only fire on the SIP-on-Basic path. `_effective_end` returns `(end, False)` for IEX, and `clamp_end_for_sip` returns `end` unchanged under `ALPACA_ALLOW_RECENT_SIP=1` or a zero delay — so a request whose end is still ahead of the clock comes back **partial with `end_clamped=False`**. `baselines.py` passes `end_date+1` and `backtests.py` has no future-date check, so that shape reaches the loader. Alpaca's `end` is exclusive and filters on each bar's opening timestamp, so a date-only `end` of `D` covers bars through `D-1`'s session; a day's margin covers the longest supported source bar plus every feed delay without the cache learning timeframe arithmetic. The price is that a window ending yesterday is served cold until tomorrow — a shape none of the default windows takes. |
| the client is unconfigured (`if not self.client: return {}`) | Otherwise "Alpaca is not configured" is cached as "this symbol has no data." |
| the symbol is absent from the response | Same reason: a missing symbol is not a negative fact worth persisting. |
| the data source is not Alpaca | See below. |

**The clamp and fallback flags are read off the frames, not off `last_fetch`.**
`_record_fetch` runs once per request, and for a >100-symbol call that is once per
100-symbol chunk — so `last_fetch` describes only the *last* chunk. With chunk one on
IEX fallback and chunk two on SIP it reads `sip_fallback_to_iex=False`, and a refusal
keyed on it would store the IEX frames under the SIP key for the TTL. The `.attrs`
stamps are per frame and cover every chunk; the wrapper folds them with `any()`,
whole-batch, because the rule is whole-batch.

**A failed miss-fetch still fails the request.** Before the cache a call returned what
Alpaca had or `{}`, and `engine.load_data` raises on `{}`. With five Dow names on disk
and Alpaca down for the other twenty-five, returning the hits alone would run a
five-symbol "Dow" with an index baseline priced off five names and the frequency
verification quietly downgraded to `evidence="configured"`. The wrapper returns `{}`
when the uncached fetch returned nothing *and* left `last_fetch` unset — the loader's
failure signature — and otherwise merges, which for a request that merely had no bars
for a symbol is exactly the old answer.

**A TTL on top**, default **7 days**, bounds exposure to a vendor revising bars for
an already-closed date. Nothing in the market-data layer handles retroactive
corrections and it could not be established whether Alpaca issues them, so the TTL
is a hedge against an unknown rather than a known — it is deliberately cheap
insurance, not a claim that revisions happen.

**The iFinD A-share path is excluded from v1**, explicitly and with a stated reason:
its hourly bars are requested unadjusted (`CPS=no`) and its corporate-action gap
check is computed from separately fetched unadjusted daily closes, so a cached
A-share window needs its own correctness argument about 除权除息 handling. It is not
the onboarding flow and does not need to ride along. The exclusion is enforced by
the cache living inside `AlpacaDataLoader`, not by a runtime check.

## 6. Storage, atomicity, eviction

**Location:** a new `BAR_CACHE_DIR = DATA_DIR / "bar_cache"` constant in
`dashboard/backend/paths.py`, on the container's ephemeral filesystem. It sits under
`dashboard/storage/data`, which `.gitignore:225` already ignores wholesale, so
entries can never be staged by accident.

**Ephemeral is sufficient, and that is the point.** Render's live service has
**no persistent disk** (`disk: null`; the `disks:` block in `render.yaml` is
documentation). This cache does not need to survive a redeploy — it only needs to
outlive a `Popen`, which is precisely what the existing in-process
`market_data_store` cannot do. A backtest child is a fresh interpreter, so a
module-level `OrderedDict` is empty on every run; a file on the instance's disk is
shared by every child on that instance.

⚠ Do **not** write into `dashboard/storage/data/cache/`. That directory holds **nine
git-tracked** per-symbol CSVs (`AAPL_2024-01-01_2024-01-31_1d.csv` and siblings),
orphaned from any current code path — gitignored going forward by
`.gitignore:222-223`, but gitignore does not untrack what is already tracked. Reusing
it would interleave live cache entries with tracked files and put a binary-ish diff in
front of every reviewer. `bar_cache/` is a fresh, wholly ignored sibling.

- **Atomic writes:** write to a temp file in the same directory, then `os.replace`.
  Up to `MAX_ACTIVE_DASHBOARD_BACKTESTS` (default 5) children run concurrently on one
  instance and will race the same key. `os.replace` is atomic on POSIX, so a reader
  sees either the old entry or the complete new one, never a partial parquet.
- **An entry is two files, and a half-entry is left to its writer.** The sidecar
  lands first, the parquet second (the commit point). A reader that finds one
  without the other has a miss — but it must not *clear* it on sight: a concurrent
  child is legitimately between its two `os.replace` calls, and unlinking its sidecar
  destroys that write. Under contention every child would pay the fetch and the key
  could stay cold indefinitely. Only a half-entry older than a **one-hour grace**
  (`_STRAY_GRACE_SECONDS`, shared with the temp-file sweep) is cleared, by the reader
  that trips over it or by the eviction pass.
- **Eviction:** a total size cap, **default 256 MB**, with LRU eviction by mtime, so
  arbitrary user windows cannot grow the cache without bound. The default is
  deliberately generous relative to the data: one symbol over a 7-weekday window at
  5m bars is roughly 550 rows across six columns, which is tens of kilobytes of
  parquet — so 256 MB holds thousands of symbol-windows. It is a runaway bound, not
  a working-set estimate. Two consequences of that framing:
  - **The pass never evicts the batch that triggered it.** Filesystem mtimes are
    coarse, so entries written within one tick tie, and a pure LRU order could pick
    the symbols just fetched as the victims — a paid fetch that never becomes a hit.
    The just-written paths are protected; among the rest, ties break on name so the
    victim is deterministic.
  - **The full directory scan is O(entries) and does not run on every write.** The
    writing process keeps a running estimate (bytes at its last scan plus what it has
    written since) and rescans only when that could have crossed the cap, or when the
    last scan is older than **60 seconds** (`_SWEEP_INTERVAL_SECONDS`) — the time
    bound exists because other processes' writes are invisible to the estimate. The
    cap is therefore enforced within a minute of being crossed, not on the byte,
    which is what a runaway bound needs and a quota would not tolerate.
- **Read failures are misses.** A corrupt, truncated or unreadable entry is deleted
  and treated as a miss, never raised. A cache must not be able to fail a backtest.

**RAM is unaffected.** Entries are read per request and not retained, so this does
not change what `MAX_ACTIVE_DASHBOARD_BACKTESTS=5` is sized against — and nothing has
ever measured one child's resident set (issue #475).

## 7. Configuration and observability

- **`ATL_BAR_CACHE`** — enabled by default when unset; on for `1`/`true`/`yes`/`on`,
  **off for anything else** — a recognised `0`/`false`/`no`/`off` silently, junk with
  a `WARNING`. The vocabulary is `allow_recent_sip`'s, in the same package, and so is
  the "anything else is off" rule; the warning is the only addition. Junk reads as
  off rather than as the default because the only reason to set a kill switch is to
  turn it off, and a typo'd kill switch that stayed on would defeat the switch at
  exactly the moment someone reached for it. Default-on is deliberate: an opt-in
  cache that is off in prod delivers nothing, and the §5 exclusion rules make it fail
  safe. The blast radius is one deploy, because the store is ephemeral.
- A **startup log line** naming the choice, matching the existing convention
  (`run history backend: postgres (…)` / `… sqlite (ephemeral on Render)`):
  `bar cache: enabled (<dir>, cap <N>MB)` or `bar cache: disabled`.
- Junk or out-of-range values for the size cap and TTL **log and fall back** rather
  than raising at import. This module is imported at app boot, and an unparseable
  value read with a bare `int()` at module scope has killed boot in this repo before.

## 8. Warm-on-boot

A background thread at startup pre-fetches the three default windows:

- `dashboard/config/defaults.json` — Mag7, 2026-05-04 → 2026-05-12 (the onboarding modal)
- `DJIA_30` over that **same** window — fact A in §1: every default run also fetches
  the full Dow for the index baseline, on the ordinary path, after
  `publish_phase("saving")`, and `engine.py`'s index-baseline block passes
  `self.start_date`/`self.end_date` verbatim, so it is the same key. Five of the
  thirty names are already hits from the first window; twenty-five are requested.
  Without this the default run's *visible* wait ends warm and its uncounted tail
  still runs cold.
- the `POST /backtest/run` endpoint default — `DJIA_30`, 2026-05-01 → 2026-05-07 (a
  bare request resolves to the `djia_30` profile)

Without it, a cold instance charges the first visitor full price, which defeats the
stated audience (prod users on the live dashboard).

It is a startup hook on the **parent web process** (`app.py`), not on the backtest
child — the child never runs `app.py`, so there is nothing to suppress there.

**Cost, named rather than discovered later:** three batched Alpaca calls per deploy
(this said two until 2026-09-21, before the index-baseline window was counted), and
merging to `main` auto-deploys prod via the CI hook. Negligible quota, but it is a
new recurring outbound call. The window list is the one owner of the count — tests
assert `len(warm_windows())`, never a literal. It runs on a background thread so it cannot delay
boot or fail the health check, and a failure is logged and swallowed — a cold cache
is the status quo, not an outage.

⚠ **`tests/conftest.py` must disable the warm step**, the same way it strips
`RENDER`, `IFIND_*` and the other environment that changes behaviour under test.
Without that, importing the app in the suite would attempt live Alpaca calls — which
is both a network dependency in an offline suite and a real spend. This is a
required part of the change, not a nicety.

## 9. What shipping this measures

With the cache in place, **the residual `loading_bars` time on a warm key is the
aggregation half** — directly, with no synthetic benchmark and no subtraction. That
is the number §1-B says cannot be obtained today, and it is what decides whether
caching the *aggregated* output is worth a second change or whether the fetch was the
whole story.

To make it readable from a log rather than re-derived by hand, add one `⏱` line to
the existing phase instrumentation splitting `loading_bars` into fetch and
post-fetch, following the precedent `starting` already sets with its four
sub-numbers (spawn, imports+stores, schema DDL, preflight).

## 10. Testing

All offline; no test makes a live network call.

- Hit, miss and expiry against a fake loader returning canned bars.
- **Mutation-tested both directions** on the §5 rules: a response stamped
  `end_clamped=True` is **not** written, a response stamped
  `sip_fallback_to_iex=True` is **not** written, and a window whose `end` has not
  settled is **not** written — including through the real `fetch_bars` on `iex`,
  where no flag fires. Each test must be shown to fail when the guard is removed — a
  guard never seen to fail is a comment.
- The flags are derived from the frames: a >100-symbol request whose first chunk
  fell back to IEX and whose last did not stores nothing, although `last_fetch` says
  no fallback happened.
- A failed miss-fetch returns `{}` even with hits on disk; a successful fetch that
  merely had no bars for a symbol still returns the hits.
- A hit restores `last_fetch` and all three `.attrs` stamps.
- A mixed hit/miss request fetches **only** the missing symbols and returns the full
  set.
- Concurrent writers: two writers racing one key leave a readable entry; a fresh
  half-entry is a miss but is left for its writer; a stale one is cleared.
- Eviction: LRU order is asserted with mtimes set *after* all writes (coarse mtimes
  tie otherwise); a write never evicts its own batch; the full scan is skipped when
  the estimate cannot have crossed the cap and re-run once the interval elapses.
- A corrupt entry is treated as a miss and removed, and the fetch still succeeds.
- The unconfigured-client path caches nothing.
- Key sensitivity: changing `ALPACA_DATA_FEED` or `source_timeframe` misses.
- Phase metrics: a metric recorded with no phase open lands nowhere, and a startup
  clock passed without a launch time does not surface on `loading_bars`.

## 11. Risks

- **The sequencing bet.** If aggregation turns out to own most of the 18s, this
  change alone underdelivers and needs a second. Accepted deliberately: it is cheap,
  its reach is five call sites, and its measurement (§9) is what makes the second
  change's cost/benefit real instead of assumed. Track A already made the opposite
  mistake once — it optimised imports and child DDL against an inferred premise, and
  measurement refuted the premise.
- **Scale-out.** An ephemeral per-container cache degrades linearly if prod ever runs
  more than one instance (N cold caches, N copies). Prod is `numInstances: 1` today
  and nothing shared exists to avoid this — no Redis, no S3, and all three Postgres
  databases are scoped elsewhere by explicit design. This is a property of the
  storage choice, not of the seam, and it would apply equally to any of the
  alternatives considered.
- **Default-on changes behaviour for every user on the first deploy.** Mitigated by
  the fail-safe exclusion rules, the kill switch, and the ephemeral store.

## 12. Follow-ups to file when this lands

1. `market_data_store` hardcodes `market="US", timezone="US/Eastern"` in its
   aggregation call and has no market dimension in its key — a latent A-share defect
   on the protocol/v2 path.
2. `_find_cached_run` (`domain/leaderboard/service.py`) keys a persisted
   `agent_runs` row without the feed, the same hazard CLAUDE.md warns about.
3. `market_data_store._dataset_key` uses order-sensitive `tuple(symbols)`.
4. `baseline_generator._fetch_bars_for_symbol` is dead code — referenced only by
   `tests/test_baseline_generator_offline.py`.
5. Read `agent_runs.runtime_type` / `decision_source` in prod to settle the
   LLM-vs-rule-based mix, which is the ceiling on every latency change of this kind
   (§1-C).
6. Stale git-tracked CSVs under `dashboard/storage/data/cache/` with no reader.

## 13. Line anchors

Every file reference in this document is **advisory**. `main` moves, and eleven
anchors written by Track A's own branch pointed at wrong lines on the tree that
shipped them. Grep for the quoted symbol or signature; never jump to a number.
