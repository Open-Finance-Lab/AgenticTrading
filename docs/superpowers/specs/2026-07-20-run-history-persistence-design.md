# Durable run history (`AGENT_RUNS_DATABASE_URL` Postgres backend)

**Date:** 2026-07-20 · **Amended:** 2026-07-29 (three passes — see *Amendments* at the end)
**Issue:** #140 (Agent run history still evaporates on every deploy)
**Status:** Approved design — phase 1 of the run-history migration (the "cold half")
**Predecessor:** `2026-07-15-agent-strategy-persistence-design.md` (Decision 1 there deferred
run history to "a later phase"; this spec is that phase's first slice)

> **Read the Amendments section before implementing.** Between 2026-07-20 and 2026-07-29 the
> SQLite side moved under this spec's feet — most importantly `insert_equity_points`, whose
> semantics inverted in `b988f24`. The affected passages below are corrected in place; the
> Amendments log records what changed and why, so a later reader can tell a correction from
> the original design intent.
>
> **Two amendment passes, both 2026-07-29.** Pass 1 was a drift sweep (what changed in the
> code). Pass 2 followed an architecture review and is the more important one: it caught two
> places where the design as written *fails the repo's own parity guard*, three follow-up
> issues that were already fixed on `main` (filing them would have created dead issues), and
> the T2/T4 scale work that changed both the connection recipe and finalize's failure profile.
> The recurring cause is the same each time: passages that restate code facts (line numbers,
> method lists, connection recipes) decay; the *decisions* have not moved once.
>
> **Pass 3** is a rename, applied before any code shipped: the env var is
> `AGENT_RUNS_DATABASE_URL`, never the bare `RUNS_DATABASE_URL` this spec originally proposed
> (rationale in Decision 3). It is already set in Render — see Rollout step 2.

## Problem

Prod runs on Render's free tier with no persistent disk, so `DATABASE_PATH` resets to the
git-committed seed `backtest.db` on every deploy — and merging to `main` auto-deploys. Since
PR #134, agents/versions/strategies survive deploys (`CONTENT_DATABASE_URL`) and accounts
survive (`USERS_DATABASE_URL`), but every backtest run, equity curve, trade log, and decision
log still dies with the deploy. Concretely:

- The daily leaderboard's LLM entries (`agent_runs` rows) evaporate, which is why issue #145
  (the unscheduled refresh job) is blocked: an off-instance cron would write to a database
  prod never reads.
- A registered agent survives a deploy while its entire run history does not.
- Run-detail pages (equity, trades, decision logs) silently lose everything created since the
  last deploy.

Issue #140's standing decision (2026-07-18) was "accept + document the split"; the fleet
wiring plan (2026-07-20) promoted resolving it to the literal first step (P0.1/D1) because it
gates #145 and the paper-trading track. This spec executes that promotion.

## Decisions (settled with Felix, 2026-07-20)

1. **Scope: the cold half only.** Five tables move to Postgres — `agent_runs`,
   `equity_timeseries`, `trades`, `backtest_decisions`, `run_manifest`. All are written at
   run finalize (batched) or once per run. The hot per-step tables — `idempotency_keys`,
   `protocol_runs`, `protocol_steps` — **stay in SQLite**: they are written 1–2× per step
   synchronously inside the agent's HTTP request, where a Neon round-trip (and free-tier
   cold-start, multi-second wake) would sit inside the decision deadline (60s today, via
   `EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS`; this spec originally said 30s, so the headroom
   is larger than argued, not smaller). Per-step
   operational state is ephemeral today and remains ephemeral; that is unchanged behavior,
   not a regression. Moving it durably is a follow-up with its own latency design.

2. **Topology: a dedicated Neon project** (`ATL-runs-main`, provisioned 2026-07-20, Postgres
   18.4 — matching CI's `postgres:18` pin from issue #138). This is the formal D1 sign-off.
   Rationale: its own free-tier storage/compute allotment isolates the largest, hottest
   tables' growth from the auth-critical users/content database, and the consumer sweep found
   **no code path that JOINs run tables with content/users tables** — every cross-reference is
   already a two-query Python join (and already cross-database in prod since #134), so
   same-database JOIN capability buys nothing today. Connection string lives only in Render's
   env config (and CI secrets if ever needed) — never in the repo.

3. **Env var: `AGENT_RUNS_DATABASE_URL`.** Scoped name per the established convention (states
   what it backs; never `DATABASE_URL`, which ambient Heroku-convention values could poison).
   The `AGENT_` prefix is deliberate and was chosen over a bare `RUNS_DATABASE_URL`
   (renamed 2026-07-29, before any code shipped): "runs" alone is one of the most overloaded
   words in a CI-adjacent repo — workflow runs, test runs, backtest runs — whereas the tables
   this actually backs are `agent_runs` and its children, so the var now names its primary
   table. It also groups visually with the agent-facing surface it serves. No fallback chain
   to/from `CONTENT_DATABASE_URL` or `USERS_DATABASE_URL` — unset means ephemeral SQLite, and
   the startup line makes the choice visible.

4. **R2/object-storage offload: deferred, seam reserved.** The only sizable blob in scope is
   `backtest_decisions.actions_submitted` (~880 KB per 252-step protocol run; zero such rows
   exist in prod today). The dedicated 0.5 GB Neon project holds hundreds of blob-inline runs;
   daily leaderboard runs add only ~25 KB each (equity points, no decisions). We add a
   nullable `actions_trace_ref TEXT` column **at ship time** — reserving the offload seam
   while it is free, instead of walking into the forgotten-`ALTER` → `UndefinedColumn` trap
   later. No R2 client, credentials, or read-path indirection in this change.

   **The column goes in *both* twins, not Postgres alone** (corrected 2026-07-29, pass 2 —
   the original text put it on the Postgres side only, which does not survive CI).
   `test_postgres_twin_schema_columns_match_sqlite` compares column sets with **bidirectional**
   set-equality (`test_store_twin_parity.py:478–497`); a Postgres-only column lands in that
   test's `postgres-only=[…]` drift report and reddens the suite. So: SQLite gets
   `actions_trace_ref TEXT` in `CREATE TABLE` **and** an idempotent lazy
   `ALTER TABLE backtest_decisions ADD COLUMN`, Postgres gets the same pair. Narrowing the
   assertion instead (the test's docstring permits it: "if a divergence is ever deliberate,
   narrow this assertion explicitly") was rejected — the carve-out costs the same edit and
   permanently weakens the guard for every *future* column, to buy an unused nullable column's
   absence from a table that holds zero rows. One caution: the new SQLite lazy `ALTER` fires
   against whatever `DATABASE_PATH` points at, which in local dev is the committed seed
   `backtest.db` — do not commit that mutation (the #244 integrity guard compares *content*
   and will wave a schema-only ALTER straight through).

5. **Non-goals, explicit.** (a) Multi-worker execution safety: the per-agent active-run cap
   and idempotent-replay guarantees are enforced by in-process `threading.Lock`s, and the DB
   rows are replay caches — moving tables to Postgres makes *history* durable but does not
   make horizontal scaling of live runs safe. Single-worker assumption stands (v2 spec §12).
   (b) Fixing adjacent pre-existing defects found during the design sweep — filed as
   follow-up issues instead, so this PR stays one concern. *(Corrected 2026-07-29, pass 2:
   three of the four defects originally named here were fixed on `main` between 2026-07-20
   and 2026-07-24 — the `/paper/start-session` insert, the `list_agents_with_stats` N+1, and
   the `metadata` gap. The only one still live is the same defect class in a different file,
   `dashboard/scripts/backtest.py:397`. See "Out of scope" below; the stale three must **not**
   be filed at merge.)*

## Architecture

### Backend selection

`database.py` keeps `BacktestDatabase` (SQLite) untouched as the default. A factory cloned
from `users.py::_build_user_store()` replaces the bare singleton assignment:

```python
def _build_backtest_db():
    database_url = os.getenv("AGENT_RUNS_DATABASE_URL")   # this var only, deliberately.
    if database_url:
        from dashboard.backend.database_postgres import PostgresBacktestDatabase
        print(f"run history backend: postgres ({describe_database_url(database_url)})")
        return PostgresBacktestDatabase(database_url)
    print("run history backend: sqlite (ephemeral on Render)")
    return BacktestDatabase()

db = _build_backtest_db()
```

`print()`, not `logger.info()` — backend loggers emit nothing under the deployed uvicorn
config. Fail-loud: `PostgresBacktestDatabase.__init__` validates via `require_postgres_url()`
(never echoes the input), runs DDL eagerly, and an unreachable Postgres fails app startup
rather than silently falling back to SQLite. Both shared helpers come from `db_url.py`.

**In scope: delete the raw-SQLite startup debug blocks in `app.py:116–153`** (added
2026-07-29, pass 2). Two near-duplicate blocks there open `sqlite3.connect(str(DB_PATH))`
directly and print `agent_runs` counts, bypassing the `db` singleton — the one seam leak in
the codebase. Post-migration they would print counts from the *ephemeral SQLite file* on
every boot, immediately beside the `run history backend: postgres (…)` line that is this
design's only misconfiguration tripwire. Two contradictory numbers next to each other is
worse than no number: it teaches the reader to distrust the tripwire. They are duplicated
dead weight regardless, so the fix is deletion; if a boot-time row count is wanted, route it
through `db` so it reports whichever backend is actually live.

### The delegation twin (why not split the class)

Fourteen backend modules import the `db` singleton (thirteen by `from … import db`, one via
`db_module.db` — which is also why a grep-only recount reads thirteen); `BacktestDatabase`'s public surface mixes
cold-half methods with two hot-half methods (`get_idempotency`, `put_idempotency`). Splitting
the class would touch every import site. Instead, `PostgresBacktestDatabase` implements the
cold-half surface against Neon and **delegates the idempotency methods to an embedded plain
`BacktestDatabase`** (same `DATABASE_PATH` SQLite file, same WAL setup), so the hot path
never gains a network round-trip and no call site changes:

```python
class PostgresBacktestDatabase:
    def __init__(self, database_url):
        self.database_url = require_postgres_url(database_url)
        self._sqlite = BacktestDatabase()      # hot half: idempotency_keys stays local
        self._init_schema()

    # Both take step_index -- signature corrected 2026-07-29; the original sample
    # omitted it from the getter, which test_store_twin_parity would have failed.
    def get_idempotency(self, run_id, step_index, idem_key):
        return self._sqlite.get_idempotency(run_id, step_index, idem_key)
    def put_idempotency(self, run_id, step_index, idem_key, ack):
        return self._sqlite.put_idempotency(run_id, step_index, idem_key, ack)
```

Methods spanning both halves operate on both backends: `clear_all()` truncates the five
Postgres tables *and* delegates to the embedded store; `delete_run()` deletes the Postgres
row (children go via FK cascade) and clears any local idempotency rows for that run.

**`run_manifest` is a behavior change here, not a port** (identified 2026-07-29, pass 2).
Today's SQLite `clear_all()` and `delete_run()` touch **four** tables and never
`run_manifest` (`database.py:970–994`), so manifest rows are orphaned by both operations —
and `run_manifest` has no FK to `agent_runs`, so there is nothing for a Postgres cascade to
sweep either. **Decision: fix both sides** — add `run_manifest` to `clear_all` and
`delete_run` in *both* `BacktestDatabase` and the twin, as an explicit, tested part of this
PR. Rationale: the `@pg_only` mirror suite asserts *observable* parity, so a twin that is
silently more correct than its original is a test failure waiting to happen or, worse, a
divergence nobody notices; and the fix is two lines per side against a table holding zero
rows in prod. The alternative (port the four-table behavior verbatim, file the orphan as a
follow-up) is the smaller diff and is the fallback if Felix prefers strict one-concern
discipline — but then say so in the twin's docstring, because "the twin leaks manifest rows
on purpose" is not inferable from the code.

`domain/runs/repository.py` (`RunStore` — `protocol_runs`/`protocol_steps`) is untouched.

### Method surface to port (the contract, enumerated)

Cold-half methods of `BacktestDatabase`, ported 1:1 with identical signatures and return
shapes (dict rows via `psycopg.rows.dict_row`, mirroring `sqlite3.Row` usage):

- Writers: `insert_run`, `update_run_baselines`, `insert_equity_points` (+ the single-point
  variant `insert_equity_point`), `insert_trades`, `insert_decisions`, `insert_run_manifest`,
  `delete_run`, `clear_all`.

**Signatures are copied from the live class, not from this list.** `test_store_twin_parity.py`
asserts matching signature *triples* across all six twin pairs, so a parameter that post-dates
this spec fails that guard rather than shipping — `insert_equity_points(self, run_id, points,
replace=True)` is exactly such a parameter (added `b988f24`, 2026-07-23). Read each signature
out of `database.py` at implementation time.
- Readers: `get_run`, `get_run_with_session`, `get_all_runs`, `get_runs_by_session`,
  `get_runs_by_sessions`, `get_runs_by_mode`, `get_equity_curve`, **`get_equity_curves`**
  (plural, `database.py:753` — added to this list 2026-07-29, pass 2; it was covered only by
  the "remaining accessors" hedge below, but it is load-bearing for shipped UI: its prod
  caller `domain/agents/service.py:185` feeds every My Agents equity sparkline, so omitting
  it breaks that page rather than some unused path; and it is the one method the twin must
  *not* port line-for-line — see the N+1 bullet under Performance trade-offs), `get_trades`,
  `get_decisions`, `get_run_manifest`, and any remaining cold-half accessors enumerated
  during implementation (the implementing plan lists the exact set from the class).

Dialect-neutral pure helpers (row shapers, ID formats) are imported from `database.py`, not
reimplemented — same reuse rule as the six existing twins.

## Postgres schema

DDL targets the **post-migration** SQLite shape (i.e. what `_migrate_schema` produces),
translated:

- `agent_runs`: as today incl. `session_id DEFAULT 'legacy-demo-session'`, `llm_model
  DEFAULT 'rule-based'`, `baseline_djia_run_id`, `baseline_buyhold_run_id`, token/cost
  columns, `metadata TEXT` (JSON; NULL-tolerant — the protocol path never populates it).
  `created_at`/`updated_at` stay `TEXT`, populated app-side in SQLite's
  `CURRENT_TIMESTAMP` format (`YYYY-MM-DD HH:MM:SS`, UTC) — the twin-precedent shape
  (`users_postgres.py` stores TEXT via `_utcnow_iso()`), keeping read shapes identical.
- `equity_timeseries`, `trades`, `backtest_decisions`: as today, with **real, enforced FKs**
  `REFERENCES agent_runs(run_id) ON DELETE CASCADE`. SQLite declares these FKs but never
  enforces them (no `PRAGMA foreign_keys=ON` anywhere); Postgres enforces by default, and
  CASCADE both simplifies `delete_run` and prevents dangling children.
- `backtest_decisions` additionally gains `actions_trace_ref TEXT` (nullable, unused — the
  reserved R2 seam per Decision 4). **In both twins**: SQLite's `CREATE TABLE` plus a lazy
  `ALTER TABLE … ADD COLUMN`, Postgres's `CREATE TABLE` plus its `ADD COLUMN IF NOT EXISTS`
  counterpart. A Postgres-only column fails the bidirectional column-parity check — see
  Decision 4 for why we widen the column rather than narrow the assertion.
- `run_manifest`: as today.
- Indexes ported 1:1 (`idx_agent_runs_session`, `idx_agent_runs_session_mode`,
  `idx_run_timestamp`, `idx_trades_run`, `idx_decisions_run`).
- **Plus `UNIQUE(run_id, timestamp)` on `equity_timeseries`** — a table *constraint*, not one
  of the five named indexes, which is why the list above missed it. It is load-bearing twice
  over: `ON CONFLICT (run_id, timestamp)` requires a matching unique constraint or Postgres
  raises, and it is the natural key that makes a rerun replace rather than duplicate. SQLite
  reaches the same state two ways — inline in `CREATE TABLE` for new installs, and via the
  `_ensure_equity_timeseries_uniqueness` / `_has_equity_timeseries_unique_index` lazy
  migration for legacy tables (added by the equity-uniqueness work, `b988f24`). The twin needs
  the constraint in its DDL *and* an idempotent `ALTER TABLE … ADD CONSTRAINT`-equivalent
  migration.

  **Nothing static enforces that second half — do not expect the guard to catch it**
  (corrected 2026-07-29, pass 2; pass 1 claimed `test_store_twin_parity.py` "asserts every
  SQLite lazy migration has a Postgres counterpart"). The lazy-migration parity check
  regex-matches `ALTER TABLE … ADD COLUMN` strings only (`test_store_twin_parity.py:232–236`,
  asserted at `:518–536`), and it is one-directional besides. `_ensure_equity_timeseries_uniqueness`
  adds a unique *index*, contains no `ADD COLUMN`, and is therefore **invisible** to that
  check. The only thing that would catch an omitted Postgres counterpart is the `@pg_only`
  behavioral upsert test — which is precisely why that test is mandatory rather than nice to
  have. Stakes are low in practice: the Neon project is empty, so the inline `CREATE TABLE`
  constraint suffices for every row that will ever exist there, and the migration is
  belt-and-suspenders for a database created before it. But an unenforced requirement must be
  labelled unenforced, or the next reader deletes it on the guard's authority.
- Every twin carries the institutional-memory comment block: **"ADDING A COLUMN LATER? It
  must go in an `ALTER TABLE … ADD COLUMN IF NOT EXISTS` below, not just the CREATE"** — with
  the cross-reference to `repository_postgres.py`'s explanation of why nothing else catches
  the omission.

### Dialect-sensitive idioms (verified against the code)

1. **`INSERT OR REPLACE` → `INSERT … ON CONFLICT … DO UPDATE`.** Four call sites, at
   `database.py` 516 (`insert_run`, conflict key `run_id`), 567 (`insert_equity_point`,
   single-point variant, key `(run_id, timestamp)`), 607 (`insert_equity_points`, batch, same
   key), 956 (`insert_run_manifest`, key `run_id`). This is load-bearing for the daily
   leaderboard: `force_refresh` re-runs a deterministic `run_id`
   (`lb_<strategy>_<start>_<end>`) and relies on overwrite idempotency.

   **`insert_equity_points` is the exception — it must keep its delete.** (Corrected
   2026-07-29; the original text here said "true upsert avoids the delete entirely", which
   `b988f24` made wrong three days after this spec was written.) The method now does
   `DELETE FROM equity_timeseries WHERE run_id = ?` **then** an `executemany` insert, both
   inside one transaction, gated by a `replace: bool = True` parameter, with an empty-`points`
   list short-circuiting to a no-op rather than a wipe. The delete is not incidental: a rerun
   can legitimately produce a *different* set of timestamps (fewer bars, a partial run, a
   changed symbol list), and the `(run_id, timestamp)` key only collapses timestamps that
   *repeat*. Without the delete, leftovers of the previous, longer curve stay spliced into the
   new one — silently, and precisely in the force-refresh case. Port the delete-then-insert
   shape verbatim; a pure upsert here is a data-corruption bug, not a simplification.

   The original FK rationale was also backwards and is retracted: `equity_timeseries` is the
   *child* of `agent_runs`, so deleting its rows cannot violate a foreign key in either
   direction. The delete is safe; it is the *behavior* that requires it.
2. **Both timestamps diverge on upsert, in opposite directions — and both are deliberate.**
   (Rewritten 2026-07-29, pass 2; the original text named only `created_at` and misdescribed
   `updated_at`.) `insert_run`'s INSERT column list is 18 columns and omits **both**
   timestamps (`database.py:515–521`), which are `TIMESTAMP DEFAULT CURRENT_TIMESTAMP`
   (`:87–88`). So SQLite's REPLACE delete+insert resets `created_at` *and* `updated_at` on
   every `force_refresh` — `updated_at` is not maintained separately, which is what the
   original "`updated_at` exists for that" line got wrong.

   - **`created_at` is preserved** (divergence A): the `DO UPDATE SET` clause simply doesn't
     touch it. SQLite's reset is accidental and nothing reads the column as "last refreshed".
   - **`updated_at` must be explicitly `SET` to now** (divergence B): if `DO UPDATE SET` also
     omits it, Postgres keeps the value from the *original* insert while SQLite refreshes it
     — a silent divergence in the opposite direction, landing exactly on `force_refresh`, the
     one case where "when was this row last rewritten" is the question being asked. Omitting
     it is the natural copy-paste outcome of fixing divergence A, so state it explicitly in
     the upsert.

   Mirror tests name **both** as exceptions, not parity bugs: assert `created_at` unchanged
   and `updated_at` advanced across a re-insert of the same `run_id`.
3. **`trades` / `backtest_decisions` are append-only plain INSERTs** (autoincrement ids;
   written once per run at finalize) — ported as plain INSERTs; Postgres `id` columns use
   `BIGINT GENERATED BY DEFAULT AS IDENTITY`.
4. **Batch writes keep the one-connection-one-commit-per-method shape** using psycopg3
   `executemany` (pipelined) — explicitly not connection-per-row, which would be the plausible
   copy-paste regression from the low-write twins' shape. Under the shared pool (idiom 6) that
   regression is worse than it used to be: per-row would mean a pool *checkout* per row, so
   2,102 equity rows would serialize behind `max_size=5` and can trip the 10 s pool timeout —
   a failure that looks like Neon being slow rather than like a loop written wrong.
5. **No cross-method transaction is introduced.** Finalize's four writes are independent
   commits today (run row, equity, trades, decisions), with `status='completed'` flipped
   before the baseline block and baseline failures swallowed-but-printed. Postgres keeps
   exactly those semantics — changing atomicity is out of scope and would alter observable
   failure behavior.

   **The partial-failure story, stated** (added 2026-07-29, pass 2 — it was implicit, and
   "independent commits" is only half an answer once the commits cross a network). Two
   distinct paths now:

   - *In-request (the four writes).* A mid-finalize failure leaves the run row committed with
     some children missing, and the agent's final decision-submit gets a 500 after its run
     was already marked complete. That is unchanged **in kind** from SQLite — but SQLite's
     failure modes were disk-full and locked-DB, i.e. rare; Neon adds cold-start timeouts and
     transient network faults, i.e. routine. Accepted as-is for phase 1 (no cross-method
     transaction), because the alternative — one transaction spanning four methods — changes
     the class contract for every caller, not just the twin.
   - *Post-response (baselines).* T2 moved baseline generation off the request into a
     background worker thread (`external_run_service.py:606–614`), so only the four writes are
     synchronous. This strengthens the latency case, but it moves baseline write failures
     *after* the agent has its 200, into a thread whose exceptions are swallowed-but-printed
     and which nobody watches. Per the repo's "fail-closed is not fail-visible" doctrine, the
     worker's handler must print enough to separate **"Neon unreachable"** from **"nothing to
     write"** — those two produce an identical silent no-op today, and the observable symptom
     of the first is merely a leaderboard row without baselines.
6. **Connections come from the shared pool, not `psycopg.connect()`** (corrected 2026-07-29,
   pass 2 — the original "per-call `psycopg.connect()` … same as the four existing twins"
   describes the pre-T4 codebase). There are **six** twins now and all of them go through
   `db_pool.get_pool(database_url)` (psycopg_pool, `max_size=5`, `max_idle=300 s`,
   `POOL_TIMEOUT_SECONDS=10.0`). Consequences for the new twin:
   - Take connections as `with pool.connection() as conn:` — still context-managed, still
     commit/rollback on exit.
   - **Do not pass `row_factory=dict_row` per call**: the pool sets `dict_row` at
     construction, so the rows arrive as dicts already.
   - The 10 s pool timeout doubles as the ceiling on a Neon free-tier cold-start wait, which
     is the concrete number behind the latency paragraph below.
   - Still use the pooled (`-pooler`) URL — the client-side pool and Neon's server-side pooler
     are complementary, not alternatives.

## Performance trade-offs (accepted, with eyes open)

- **Write load fits the pooled-connection recipe** because the hot half stayed local: a run's
  finalize is ~10–14 pool checkouts total, of which only **4 are in-request** (run row,
  equity, trades, decisions) — T2 moved the two baseline write sequences and the
  baseline-pointer update onto a background worker thread. A full daily leaderboard refresh is
  ~2 per strategy. Nothing is per-step. (Checkout counts updated 2026-07-29, pass 2: pooled
  reuse, not fresh connects.)
- **The one latency-sensitive moment:** the four in-request writes execute inside the final
  decision-submit request, so a Neon cold-start can add seconds to that last ack — bounded by
  the pool's 10 s timeout. Accepted, with more headroom than originally argued: the decision
  deadline is **60 s**, and T2 already took the baseline work out of that window. The daily
  refresh job (the main recurring writer) is a batch context where latency is irrelevant.
- **Reads move to the network**: `GET /runs` and `/api/v2/leaderboard` are full-table scans
  on a public route; fine at current scale (tens of rows), noted as the place pagination
  lands if the table ever grows hot (issue #203). *(Corrected 2026-07-29, pass 2: the
  `list_agents_with_stats` N+1 cited here was fixed on `main` by `fe71a8a`, 2026-07-24 — the
  **run rows** now batch through `get_runs_by_sessions`, `domain/agents/service.py:390`.)*
- **One N+1 survives, inside a reader:** `get_equity_curves(run_ids)` (`database.py:753–758`)
  is a Python `for` loop calling `get_equity_curve` once per id, and
  `domain/agents/service.py:185` hands it one run id **per agent**. Free on a local SQLite
  file; over Neon it is one network round-trip per agent on every My Agents page load. The
  twin must therefore implement `get_equity_curves` as a **single** `WHERE run_id = ANY(%s)`
  query grouped in Python — same signature, same return shape, so nothing about parity or the
  contract changes; the loop is an implementation detail of the SQLite side, not a promise.
  (Found 2026-07-29 while writing the implementation plan, after pass 2 had already asserted
  the read-side N+1 was gone. It was gone from the *run* fetch only.)
- **Neon CU-burn**: the cold half's request profile (batchy, bursty, idle most of the day) is
  the friendly case for scale-to-zero. The DBOS-spike CU measurement from the tech-stack
  blueprint is unaffected by this change.

## Migration, seeding & rollout

Order matters; each step is verifiable before the next:

1. **Provision** — done 2026-07-20: Neon project `ATL-runs-main` (Postgres 18.4, empty).
   Credential verified working; lives in Render env config only.
   *Re-verified 2026-07-29:* endpoint `ep-orange-sound-…-pooler…/neondb`, server
   PostgreSQL **18.4**, **zero public tables** (no `_init_schema` has ever run against it), so
   there is no half-migrated state to reconcile. Distinct from the users/content endpoint
   (`ep-cool-wave-…-pooler`), confirming Decision 2's dedicated-project topology. (Endpoint
   IDs are truncated deliberately: this repo is **public**, and the full ID is the resolvable
   hostname of a production database. The distinctness claim rests on the differing slugs, so
   nothing is lost. Read the live values from the Render env config, never from this doc.)
   Confirmed also that `_postgres_testing.require_local_postgres_url` **rejects this host**,
   so the `@pg_only` destructive fixtures cannot wipe it — the #136/PR #144 localhost
   allowlist covers the new database with no change. The corollary is that the Postgres tier
   cannot be exercised against this URL locally: CI's `postgres:18-alpine` remains the only
   place the tier actually runs.
2. ~~**Set `AGENT_RUNS_DATABASE_URL` in the Render dashboard *before* merging**~~ — **done
   2026-07-29**, ahead of the implementation, on the web service only (the Discord-bot worker
   has neither of the sibling URLs and talks HTTP). Pooled (`-pooler`) `ATL-runs-main` string,
   round-trip verified; the write did not trigger a deploy, so it takes effect with the merge
   deploy. The discipline still applies in reverse: **re-verify it is present immediately
   before merging** — unset silently selects ephemeral SQLite, and the startup line is the only
   tripwire.
3. **Merge → auto-deploy → verify** the log line: `run history backend: postgres
   (ep-…-pooler…/neondb)`. Wrong/typo'd URL shows up here (host/db named, credentials never).
4. **Backfill** — one-time idempotent script `dashboard/scripts/backfill_runs_to_postgres.py`:
   reads a SQLite file (default: the committed seed DB), upserts `agent_runs` (17 rows) and
   `equity_timeseries` (**2,102** rows — re-counted 2026-07-29; the 2,585 originally recorded
   here pre-dates the seed scrub) plus any rows in the other three tables (`trades`,
   `backtest_decisions`, `run_manifest` — all still zero) into the new database. Treat these
   as expected-order-of-magnitude, not assertions: re-count at implementation time and have
   the script report what it moved. Idempotent by construction (same upserts as the twin),
   safe to re-run. Until it runs, prod's `/runs` listing is empty — run it immediately after
   the first green deploy. The 3 `defaults.json` demo run IDs ride along (they're vestigial
   to the frontend but remain in the public listing).
5. **#145 unlock (the payoff):** an off-instance scheduler (GitHub Actions cron per fleet
   plan D4) can now run `refresh_daily_leaderboard.py` with `AGENT_RUNS_DATABASE_URL` pointed
   at the same Neon project, and prod serves what it writes. Wiring that cron is issue #145's
   own PR, not this one.

Local dev and the test suite are untouched: `AGENT_RUNS_DATABASE_URL` unset → SQLite exactly
as today; `tests/conftest.py` strips the new var at import time alongside the other two.

## Testing

Same three-tier structure as the #134/#136/#137 net:

1. **Ordinary CI (SQLite)** — existing suite must stay green untouched; plus `capsys` tests
   for both startup lines (never `caplog`), and a factory test that `AGENT_RUNS_DATABASE_URL`
   selection + stripping behaves like its two siblings.
2. **`@pg_only` tier (live Postgres, runs in CI via the existing `postgres:18-alpine`
   service + `TEST_POSTGRES_URL`)** — mirror parity suite running the same behavioral
   assertions against `PostgresBacktestDatabase`: round-trip every ported method; the three
   upsert paths (incl. leaderboard-style re-insert of an existing `run_id` with equity rows
   present — the FK-sensitive case); **both** timestamp divergences (`created_at` preserved,
   `updated_at` advanced — see dialect idiom 2); FK
   cascade on `delete_run`, plus the explicit `run_manifest` deletion that no FK covers;
   `executemany` batch inserts; delegation (idempotency calls land
   in SQLite, never Postgres). **The new destructive fixture MUST call
   `require_local_postgres_url()` before any `DELETE FROM`** — the standing 5th-fixture
   convention from #136 — and joins `test_postgres_url_guard.py`'s parametrized fixture list
   so ordinary CI proves the guard fires. *(Caveat added 2026-07-29, pass 2: that list is
   hand-maintained and already misses `pg_portfolio_store` and the brokers fixture, so
   "joins the list" is a convention nobody enforces — adding the new fixture is a manual step
   that will not be caught if skipped. Worth an explicit checklist line in the implementation
   plan; making the list self-populating is out of scope here.)*
3. **Coverage-net wiring** — the new module follows the `TEST_POSTGRES_URL`-at-import
   `pg_only` marker pattern, which fails open (skips) when unset; `test_ci_postgres_wired`
   already makes that loud in CI, and the new tests ride the same net.

Backfill script gets its own test: seed-fixture SQLite → in-test Postgres → row counts +
spot-checked parity + re-run idempotency.

## Config & docs surface

- `.env.example`: add `AGENT_RUNS_DATABASE_URL` with a one-line comment.
- `render.yaml`: add `AGENT_RUNS_DATABASE_URL` (`sync: false`) beside `USERS_DATABASE_URL` /
  `CONTENT_DATABASE_URL`.
- `CLAUDE.md`: extend the env-var bullet list; fix the now-ambiguous "Persisted stores
  (protocol runs, strategies) live in this DB" line; update the "Prod deploy reality" bullet
  (run history no longer evaporates once this ships).
- `2026-07-15-agent-strategy-persistence-design.md`: no edit (historical record), but this
  spec is its named phase-2 successor.

## User-facing docs made true/stale by this change (follow-ups, not edited mid-session)

*(List re-swept 2026-07-29, pass 2 — line numbers corrected and the hosted `docs/source/`
pages added; they were missing entirely from the original.)*

- `README.md:85` (not :81) and `docs/architecture/dashboard-target-structure.md:213` —
  file-structure diagrams say `storage/ # backtest.db` without distinguishing what now lives
  in Postgres; the latter is even annotated "(unchanged)". Needs a note once this ships.
- `docs/source/lab/architecture.rst:4,16` — describes the data flow as SQLite-only; goes
  stale the moment this deploys. Ships to ReadTheDocs ~2 min after merge with no CI build
  step to catch breakage, so edit deliberately.
- `docs/source/lab/{accounts.rst:9–10, getting_started.rst:57, key_features.rst:10,
  operating_modes.rst:13–14}` — each implies run/leaderboard history persists. This change
  makes those claims **true**; they are listed so someone confirms rather than assumes.
- `docs/source/lab/live_trading.rst:217` — the "links do not survive a redeploy" caveat
  becomes partly obsolete once run rows are durable.
- The durability list in the 2026-07-15 spec (lines 418–430) is **5 documents / 7 pointers**,
  not "7 docs" as written here originally — those docs promised agent durability (now true);
  run-history durability claims should be re-checked once this lands.

## Out of scope (follow-up issues, filed at merge time)

**Re-verified at source 2026-07-29, pass 2. Three of the five original entries were fixed on
`main` between 2026-07-20 and 2026-07-24 — filing them at merge would have created three dead
issues.** Struck below rather than deleted, so a reader of the original list can see they were
checked rather than forgotten:

1. **Hot half durability** (`idempotency_keys`, `protocol_runs`, `protocol_steps`) — needs
   its own latency design (write-behind/batching vs per-step round-trips vs coarser
   snapshots). #140 stays open as the tracker until decided otherwise. **Still live.**
2. **`insert_run` missing `session_id` in `dashboard/scripts/backtest.py:397`** — the same
   defect class as the `/paper/start-session` bug below, in the sibling that was missed when
   that one was fixed. The CLI's save path calls `db.insert_run(...)` without the required
   `session_id`, raising an unhandled `TypeError` on **every** run; `backtest_custom_algo.py:323`
   passes it correctly, which is why the pattern looked fixed. **Still live — this is the
   entry to file.** *(Replaces the original item 2:* ~~broken `/paper/start-session` run
   insert, `api/routers/paper_trading.py:273`~~ *— fixed on `main`; the call at
   `paper_trading.py:278` now passes `session_id=run_id` with an explanatory comment.)*
3. ~~**`list_agents_with_stats` N+1**~~ — **fixed** on `main` by `fe71a8a` (2026-07-24); it
   batches through `get_runs_by_sessions` (`domain/agents/service.py:390`, docstring says "no
   N+1"). Do not file.
4. ~~**`agent_runs.metadata` inconsistency**~~ — **fixed** on `main` by `17bf0121`
   (2026-07-24); `external_run_service.py:583–586` now passes
   `metadata={"decision_timeout_seconds": …, "timeout_holds": …}`. Do not file.
5. **R2 `actions_trace_ref` offload** — when protocol-run volume materializes. **Still live**
   (the column ships now; only the offload is deferred).

Net: **two** issues to file at merge (items 1 and 2), not five.

## Amendments — pass 1 (2026-07-29, drift sweep)

The design was approved 2026-07-20 but no implementation plan was written and no code was
ever produced — `AGENT_RUNS_DATABASE_URL` appears nowhere in the repo, and `database.py:998`
is still the bare `db = BacktestDatabase()`. In the intervening nine days the SQLite side
moved.
Each item below was verified at source before the correction was made in place:

| # | Drift | Resolution |
|---|---|---|
| 1 | `insert_equity_points` inverted its semantics in `b988f24` (2026-07-23) — it now deletes the run's rows and re-inserts, behind a new `replace: bool = True` parameter, with an empty-list no-op guard. The spec said a true upsert "avoids the delete entirely". | Dialect idiom #1 rewritten. The delete is mandatory: a rerun can yield a *different* timestamp set, and a pure upsert leaves the previous curve's leftovers spliced into the new one on force-refresh. The FK rationale was backwards (child rows cannot violate the FK) and is retracted. |
| 2 | The `replace=` parameter is absent from the ported-method list. | Method-surface section now says signatures are read out of the live class, and names `test_store_twin_parity.py`'s signature-triple check as the guard that fails otherwise. |
| 3 | `UNIQUE(run_id, timestamp)` on `equity_timeseries` is a table constraint, so the "indexes ported 1:1" list omitted it — yet `ON CONFLICT (run_id, timestamp)` raises without it. | Added to the DDL section, together with the requirement for a Postgres counterpart to the `_ensure_equity_timeseries_uniqueness` lazy migration. *(Pass-2 correction: the same edit asserted that `test_store_twin_parity.py` enforces that counterpart. It does not — see pass 2, item ②. The requirement stands; the enforcement claim was wrong.)* |
| 4 | Delegation sample showed `get_idempotency(run_id, idem_key)`; the real method takes `step_index` too. | Sample corrected. |
| 5 | Backfill cited 2,585 `equity_timeseries` rows; the scrubbed seed holds 2,102 (`agent_runs` unchanged at 17). Decision deadline is 60s, not the 30s the latency argument cites. | Counts updated and marked re-count-at-implementation; deadline corrected — the larger value strengthens Decision 1 rather than weakening it. |

**Unchanged and re-confirmed at source.** The cold/hot split is real: every write to the five
cold tables is batched at finalize (`insert_equity_points(run_id, whole_curve)`,
`insert_trades`, `insert_decisions`), and the only per-step writers are `db.put_idempotency`
(v2) and `run_store.save_step`/`finalize_step`. This **refutes issue #140's own stated
caveat** — "`agent_runs` is the hottest/largest table — every backtest step writes decision
rows" — which is the estimate that made this work look expensive enough to defer twice. It is
wrong: no per-step write touches `agent_runs`, `equity_timeseries`, `trades`, or
`backtest_decisions`. Correct #140 when this ships rather than leaving the caveat to deter a
future reader.

Scope re-confirmed with Felix 2026-07-29: **cold half only**. #145 (leaderboard cron), #203
(paginate `GET /runs`) and #244 (seed-DB guard) stay separate, as the spec's one-concern
argument intends.

## Amendments — pass 2 (2026-07-29, architecture review)

Pass 1 corrected what the code had changed. Pass 2 followed a Q0–Q7 architecture review of
the amended document and found what pass 1 could not see from a drift diff: two places where
the design **fails a guard it cites as an ally**, three follow-ups that were already fixed,
and the T2/T4 scale work whose effects are spread across four sections. Every item was
re-verified at source before being written in. **No decision changed** — the cold/hot split,
the dedicated Neon project, the scoped env var, and the delegation twin all survived review
untouched, which is itself the finding worth keeping: the decisions have never drifted; only
the code-restating passages have, twice now.

| # | Finding | Resolution |
|---|---------|------------|
| ① | Decision 4 put `actions_trace_ref` on the Postgres twin only, but `test_postgres_twin_schema_columns_match_sqlite` (`test_store_twin_parity.py:478–497`) does **bidirectional** set-equality — a Postgres-only column reddens CI. | Column now specified for **both** twins (CREATE + lazy ADD COLUMN on each side). Narrowing the assertion was considered and rejected — same edit cost, permanent guard weakening. Decision 4 + DDL section. |
| ② | Pass 1 claimed the guard "asserts every SQLite lazy migration has a Postgres counterpart". It matches `ALTER TABLE … ADD COLUMN` strings only (`:232–236`, `:518–536`) and is one-directional; `_ensure_equity_timeseries_uniqueness` adds an *index* and is invisible to it. | DDL section now labels the requirement **unenforced**, names the `@pg_only` upsert test as the only real catch, and notes the low stakes (empty Neon project ⇒ inline DDL suffices). |
| ③ | Three of five "Out of scope, filed at merge" items were fixed on `main` after this spec was written: `/paper/start-session` `session_id` (now `paper_trading.py:278`), `list_agents_with_stats` N+1 (`fe71a8a`, `service.py:390`), `metadata` gap (`17bf0121`, `external_run_service.py:583–586`). Filing them would create three dead issues. | Out-of-scope list struck-through with evidence; Decision 5(b) and the read-side perf bullet corrected. **Two** issues to file at merge, not five. |
| ④ | A live instance of the *same* defect class was found where the fixed one used to be: `dashboard/scripts/backtest.py:397` calls `insert_run` without the required `session_id` — unhandled `TypeError` on every CLI save (`backtest_custom_algo.py:323` is correct, which is why the pattern read as fixed). | Takes the vacated slot as out-of-scope item 2. |
| ⑤ | Dialect idiom 2 said "`updated_at` exists for that". It doesn't: `insert_run`'s 18-column list (`database.py:515–521`) omits **both** timestamps (`:87–88`), so REPLACE resets both. A `DO UPDATE` that only protects `created_at` leaves `updated_at` frozen — divergence in the opposite direction, exactly on `force_refresh`. | Idiom 2 rewritten as two named divergences (A: preserve `created_at`; B: explicitly `SET updated_at`). Mirror tests assert both. |
| ⑥ | Idiom 6's "per-call `psycopg.connect()` … same as the four existing twins" pre-dates T4: there are **six** twins and all use `db_pool.get_pool()` (`max_size=5`, `max_idle=300 s`, 10 s timeout, `dict_row` at construction). | Idiom 6 rewritten to the pool recipe (incl. *don't* pass `row_factory` per call); idiom 4 and the write-load bullet updated; "four existing twins" → six throughout. |
| ⑦ | T2 moved baseline generation to a background worker (`external_run_service.py:606–614`), so only 4 writes are in-request. This strengthens the latency case but relocates baseline failures to a post-response thread that swallows-and-prints. The spec had no partial-failure story for either path. | Idiom 5 gains an explicit two-path failure story; the worker's handler must distinguish "Neon unreachable" from "nothing to write" per the fail-closed-is-not-fail-visible doctrine. Latency bullet corrected to the 60 s deadline. |
| ⑧ | `get_equity_curves` (plural, `database.py:753`) was missing from the reader list, covered only by the "remaining accessors" hedge — yet `domain/agents/service.py:185` uses it for every My Agents sparkline. Reading it during plan-writing showed it is a **Python loop over `get_equity_curve`**, i.e. one query per agent — the read-side N+1 that pass 2 had just declared fixed (the `fe71a8a` fix covered the run rows, not the curves). | Named explicitly in the method surface; the twin must implement it as a single `WHERE run_id = ANY(%s)` query. Perf section gains its own bullet. |
| ⑨ | `clear_all`/`delete_run` touch four tables today and never `run_manifest` (`database.py:970–994`), which has no FK for a cascade to follow — so the spec's "truncates the five Postgres tables" was a silent behavior change, not a port. | Made an explicit decision: fix **both** sides (two lines each) with a mirror test, with the port-verbatim fallback and its docstring requirement written down. |
| ⑩ | `app.py:116–153` holds two duplicate startup-debug blocks that read `agent_runs` counts through a raw `sqlite3.connect(DB_PATH)`, bypassing the singleton — post-migration they print stale ephemeral-SQLite numbers beside the `run history backend: postgres` tripwire, the design's only misconfiguration signal. | Removal (or routing through `db`) added to scope, in the Backend-selection section. |
| minor | "Thirteen backend modules import `db`" — it is fourteen (one imports `db_module.db`, so a grep for the `from … import db` form undercounts). `test_postgres_url_guard.py`'s parametrized fixture list is hand-maintained and already omits `pg_portfolio_store` and the brokers fixture, so "joins the list" is an unenforced convention. User-facing docs list had `README.md:81` (actually `:85`), omitted the five hosted `docs/source/lab/*.rst` pages, and called the 2026-07-15 durability list "7 docs" (it is 5 docs / 7 pointers). | All corrected in place; the fixture-list gap is flagged as a manual checklist item for the implementation plan. |

**Review verdict (Q0–Q7):** PASS on principle, scalability, customizability, cost and trust
boundaries; CONCERN on failure-handling (⑦, now written down), observability (⑩, now in
scope) and the reality check (①–③, now fixed in-doc). **No BLOCKERs — the walking skeleton is
specable; build it.** The seam is genuinely narrow: zero call-site changes, the twin registry
auto-detects `database_postgres.py`, and delegation is invisible to the signature guard.
