# Timed-Out Backtest Outcome Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pipeline LLM backtest killed by the 3600-second parent timeout reports its own terminal outcome — naming the limit, disclosing what it cost, and telling the user which two levers to move — instead of a truncated `TimeoutExpired` argv fragment nobody sees.

**Architecture:** A new `except subprocess.TimeoutExpired` arm in `run_backtest_background` finalizes the slot with `timed_out=True` plus a facts-only `timeout_detail` dict (limit, lane, spend, call count). The slot carries it through `_slot_snapshot` to a new `/backtest/status` branch between `cancelled` and `error`. The client raises its poll ceiling above the server budget so the verdict always arrives before the poller quits, and renders the three copy lines itself. Analytics keeps `backtest_failed` and adds `error_category="run_timeout"`.

**Tech Stack:** FastAPI + SQLite/Postgres twin repositories (`dashboard.backend`), vanilla JS frontend with no build step (`dashboard/frontend/app.js`), pytest, `node -e` harnesses over lifted JS source, optional `discord.py`.

**Spec:** `docs/superpowers/specs/2026-09-14-backtest-timeout-outcome-design.md`

## Global Constraints

- **Scope is `runtime_type="pipeline"` + `decision_source="llm"`.** Do not touch the hosted `ai_hedge_fund` path or `_enforce_ai_hedge_fund_window`.
- **Do not change** `PIPELINE_SUBPROCESS_TIMEOUT_SECONDS` (3600), `SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS` (600), or `MAX_SUBPROCESS_TIMEOUT_SECONDS` (14400). The server budget does not move.
- **No refunds.** This change discloses spend; it does not reverse it.
- **`error` stays `None` on the timed-out path.** Routing a timeout through `error` is caught by the existing `elif slot.get("error")` branch and nothing on screen changes.
- **Never run a credits query while holding `_backtest_slots_lock`.** `_finalize_slot_locked`'s docstring states the rule: the analytics emit stays outside the lock because "it can reach a store, and this lock is taken by every status poll and every launch." A store read is the same hazard.
- **The server owns facts; the client owns words.** `/backtest/status` returns numbers and enum values only. Every user-visible sentence is composed in `app.js`, so `tests/test_app_copy_register.py` can see it.
- **Never commit mutations to `dashboard/storage/data/backtest.db`.** Check `git status` before every commit and unstage it if it appears.
- **Exact copy strings**, used verbatim:
  - `'Backtest stopped at the time limit.'` — status payload `message` fallback
  - `'Backtest stopped at the time limit'` — progress panel title (no trailing period)
  - `'Stopped at the {N}-minute limit.'` — first copy line, `N` derived, never a literal
  - `'Model calls completed before the stop cost {amount} Credits.'` — second line, omitted entirely on BYOK
  - `'Shorten the date range, or use fewer pipeline steps, then run it again.'` — third line
  - `'Lost contact with this backtest. It may still be running — check the Backtest tab later.'` — the client's poll-ceiling message (em dash, no number)
- **Exact constant values:** `BACKTEST_BUDGET_SECONDS = 3600`, `BACKTEST_POLL_MAX_SECONDS = 4200`, `error_category` string `"run_timeout"`, CSS class `is-timed-out`, status key `timed_out`.
- **Run tests from the repo root:** `pytest dashboard/backend/tests/... -v`. `node` must be on `PATH` for the frontend harnesses — they *skip* without it, so a green run proves nothing about JS coverage.

---

## File Structure

| File | Responsibility in this change |
|---|---|
| `dashboard/backend/domain/credits/repository.py` | `CreditsStore.sum_run_llm_spend` + `idx_credit_llm_usage_user_run` (SQLite) |
| `dashboard/backend/domain/credits/repository_postgres.py` | the same two, on the Postgres twin |
| `dashboard/backend/domain/credits/service.py` | one-line passthrough on `CreditsService` |
| `dashboard/backend/domain/analytics/models.py` | `"run_timeout"` in `ALLOWED_ERROR_CATEGORIES` |
| `dashboard/backend/domain/analytics/repository.py` | SQLite CHECK text **and** the migration sentinel |
| `dashboard/backend/domain/analytics/repository_postgres.py` | base DDL CHECK **and** the `ADD CONSTRAINT` in `_init_schema` |
| `dashboard/backend/api/routers/backtests.py` | slot field, finalize kwargs, snapshot, status branch, the `except` arm, the `billing_mode` parameter |
| `dashboard/frontend/app.js` | two constants, `formatBacktestTimeoutMessage`, `renderBacktestTimeoutPanel`, `showBacktestRunProgress` flag, poll-ceiling copy |
| `dashboard/frontend/styles.css` | `.backtest-run-progress.is-timed-out` |
| `dashboard/backend/integrations/discord_bot.py` | `timed_out` + `cancelled` branches in `watch_and_deliver_backtest` |
| `CLAUDE.md` | correct the LLM-billing bullet's table name |

**Note on `error_category`:** the spec's §6 names three edit sites. There are **five** — it missed `repository_postgres.py:61` (base DDL CHECK) and `repository_postgres.py:230-242` (the `ADD CONSTRAINT` re-added on every `_init_schema`). Task 2 covers all five.

**Note on store parity:** the spec's §3 point 4 says the two credits stores "share no base class… this is why the twin tests are not optional." True of the runtime, but `tests/test_store_twin_parity.py::test_postgres_twin_exposes_every_sqlite_method` and `::test_postgres_twin_signatures_match_sqlite` already enforce it statically — adding the method to SQLite alone fails CI on its own. The behavioural twin tests in Task 1 are still required (parity of *existence* is not parity of *semantics*), but the safety net is stronger than the spec assumed.

---

## Task 1: Credits — `sum_run_llm_spend` on both stores

**Files:**
- Modify: `dashboard/backend/domain/credits/repository.py` (add index beside `idx_credit_llm_usage_user_id`; add method near `release_run_llm_reservations`, ~`:1619`)
- Modify: `dashboard/backend/domain/credits/repository_postgres.py` (add index to `CREDITS_POSTGRES_DDL` after `idx_credit_llm_usage_user_id`; add method near `release_run_llm_reservations`, ~`:1369`)
- Modify: `dashboard/backend/domain/credits/service.py` (passthrough after `release_run_llm_reservations`, ~`:414`)
- Test: `dashboard/backend/tests/domain/credits/test_repository.py`
- Test: `dashboard/backend/tests/domain/credits/test_repository_postgres.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `CreditsStore.sum_run_llm_spend(user_id: int, run_id: str) -> tuple[int, int]`, `PostgresCreditsStore.sum_run_llm_spend(user_id: int, run_id: str) -> tuple[int, int]`, and `CreditsService.sum_run_llm_spend(user_id: int, run_id: str) -> tuple[int, int]`. Returns `(micro_credits_spent_positive, distinct_call_count)`. Task 4 calls the **service** method via the module-level `credits_service` singleton.

- [ ] **Step 1: Write the failing SQLite test**

Append to `dashboard/backend/tests/domain/credits/test_repository.py`. The module already has `_store`, `_pending_order`, `_pay_order` and `_settle_activity_call` helpers — use them, do not write new ones.

```python
def test_sum_run_llm_spend_totals_settled_calls_for_one_run(tmp_path):
    """Positive micro-Credits and a distinct call count, scoped to one run."""
    store = _store(tmp_path)
    _pending_order(store, amount_usd_cents=1000, credits_micro=10_000_000)
    _pay_order(store, amount_usd_cents=1000)

    _settle_activity_call(store, run_id="run-a", call_index=0, actual_micro=25_000)
    _settle_activity_call(store, run_id="run-a", call_index=1, actual_micro=17_318)
    _settle_activity_call(store, run_id="run-b", call_index=0, actual_micro=900_000)

    assert store.sum_run_llm_spend(1, "run-a") == (42_318, 2)
    # A run with no settled calls is (0, 0), not an error -- the caller uses
    # this to say "nothing had settled yet", which is a real outcome.
    assert store.sum_run_llm_spend(1, "run-never-ran") == (0, 0)


def test_sum_run_llm_spend_is_scoped_to_the_owning_account(tmp_path):
    """A guessed run id must not report another account's spend."""
    store = _store(tmp_path)
    _pending_order(store, amount_usd_cents=1000, credits_micro=10_000_000)
    _pay_order(store, amount_usd_cents=1000)
    _settle_activity_call(store, run_id="run-a", call_index=0, actual_micro=25_000)

    assert store.sum_run_llm_spend(1, "run-a") == (25_000, 1)
    assert store.sum_run_llm_spend(2, "run-a") == (0, 0)


def test_sum_run_llm_spend_includes_recovery_entries(tmp_path):
    """A recovery row is the FIRST debit of money settle could not charge.

    `_settle` records `outstanding_micro = actual_micro - debit_micro` -- the
    part of a call's real cost that could not be debited for want of funds --
    and the recovery entry debits exactly that remainder later. Excluding it
    (which `_llm_reservation_result_in_transaction` and the activity feed both
    do, for their own different questions) would under-report a charge to the
    person being charged.

    Written against the table directly rather than by driving an overage
    through the webhook: this test is about the aggregate's filter, and a
    fixture that has to arrange a shortfall to assert a SQL predicate stops
    being readable as a statement about that predicate.
    """
    store = _store(tmp_path)
    _pending_order(store, amount_usd_cents=1000, credits_micro=10_000_000)
    _pay_order(store, amount_usd_cents=1000)
    settled = _settle_activity_call(
        store, run_id="run-a", call_index=0, actual_micro=25_000
    )

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            INSERT INTO credit_llm_usage_entries (
                user_id, reservation_id, run_id, call_index, bucket,
                amount_micro, operation_key, evidence_json, created_at
            ) VALUES (?, ?, ?, ?, 'purchased', ?, ?, '{"t":"test"}', ?)
            """,
            (
                1,
                settled["reservation_id"],
                "run-a",
                0,
                -4_000,
                f"{settled['reservation_id']}:recovery:test:purchased",
                "2026-09-14T00:00:00+00:00",
            ),
        )

    # 25_000 settled + 4_000 recovered. Still ONE call: the recovery pays off
    # the same call_index, so counting it again would inflate the count.
    assert store.sum_run_llm_spend(1, "run-a") == (29_000, 1)


def test_sum_run_llm_spend_rejects_blank_identifiers(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.sum_run_llm_spend(1, "   ")
    with pytest.raises(ValueError):
        store.sum_run_llm_spend(0, "run-a")


def test_credit_llm_usage_entries_are_indexed_by_user_and_run(tmp_path):
    """Without this index the new aggregate table-scans a table that grows
    with every LLM call ever made."""
    store = _store(tmp_path)
    with sqlite3.connect(store.db_path) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
    assert "idx_credit_llm_usage_user_run" in names
```

If `CreditsStore` does not expose `db_path` as a public attribute, use the same `tmp_path / "credits.db"` literal `_store` uses instead of `store.db_path` — check `_store`'s body first and match whichever it is.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/domain/credits/test_repository.py -k sum_run_llm_spend -v`
Expected: FAIL with `AttributeError: 'CreditsStore' object has no attribute 'sum_run_llm_spend'`

Run: `pytest dashboard/backend/tests/domain/credits/test_repository.py -k indexed_by_user_and_run -v`
Expected: FAIL — `assert 'idx_credit_llm_usage_user_run' in names`

- [ ] **Step 3: Add the SQLite index**

In `dashboard/backend/domain/credits/repository.py`, directly after the existing `idx_credit_llm_usage_user_id` block (~`:556-560`):

```python
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_credit_llm_usage_user_run
            ON credit_llm_usage_entries(user_id, run_id)
            """
        )
```

`CREATE INDEX IF NOT EXISTS` is idempotent and runs on every schema init, so existing databases pick it up with no migration step. It also serves the evidence sub-query at `repository.py:2311-2316`, which already filters on exactly this pair.

- [ ] **Step 4: Add the SQLite method**

In `dashboard/backend/domain/credits/repository.py`, immediately after `release_run_llm_reservations` (which ends ~`:1648`):

```python
    def sum_run_llm_spend(self, user_id: int, run_id: str) -> tuple[int, int]:
        """Settled LLM spend for one run: ``(micro_credits, distinct_calls)``.

        The spend comes back POSITIVE. ``credit_llm_usage_entries.amount_micro``
        is stored negative (its CHECK enforces ``< 0``) and every other reader in
        this module negates on the way out; see
        ``_llm_reservation_result_in_transaction``.

        ``:recovery:`` entries are deliberately INCLUDED, which is the opposite
        of what that method and the activity feed do. They are answering "what
        did this one reservation settle for"; this answers "what was the account
        charged". ``_settle`` records ``outstanding_micro = actual_micro -
        debit_micro`` -- the part of a call's real cost that could not be debited
        for want of funds -- and a recovery row is the FIRST debit of that
        remainder, not a second debit of money already counted. Filtering it
        would under-report a charge to the person being charged, which is the
        one direction a disclosure must never round.

        Scoped by ``user_id`` as well as ``run_id`` so a guessed run id cannot
        report someone else's spend.
        """
        _positive_integer(user_id, "user_id")
        run_id = _required_text(run_id, "run_id", max_length=128)
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(-amount_micro), 0) AS spent_micro,
                    COUNT(DISTINCT call_index) AS call_count
                FROM credit_llm_usage_entries
                WHERE user_id = ? AND run_id = ?
                """,
                (user_id, run_id),
            ).fetchone()
        return (int(row["spent_micro"] or 0), int(row["call_count"] or 0))
```

`_positive_integer` and `_required_text` are already imported from `repository_common` at the top of this module — do not re-import them.

- [ ] **Step 5: Run the SQLite tests to verify they pass**

Run: `pytest dashboard/backend/tests/domain/credits/test_repository.py -v`
Expected: PASS (whole file — the index change touches shared schema init)

- [ ] **Step 6: Write the failing Postgres twin test**

Append to `dashboard/backend/tests/domain/credits/test_repository_postgres.py`. That module already defines `pg_only` and imports `require_local_postgres_url`; find the existing `@pg_only` fixture that provisions a store and a paid account and reuse it rather than writing a new one — match the name and call shape of whatever the neighbouring `@pg_only` LLM-reservation tests use.

```python
def test_postgres_schema_indexes_llm_usage_by_user_and_run():
    """Static DDL guard -- runs without a live Postgres.

    The twin's index set is what makes the new aggregate cheap on the engine
    that actually holds production volume, and CI's Postgres container is empty
    on every run, so no live test would notice its absence.
    """
    assert "idx_credit_llm_usage_user_run" in pg_module.CREDITS_POSTGRES_DDL
    assert (
        "ON credit_llm_usage_entries(user_id, run_id)"
        in pg_module.CREDITS_POSTGRES_DDL
    )


@pg_only
def test_postgres_sum_run_llm_spend_matches_the_sqlite_twin(pg_store_with_credits):
    """Same question, same answer, on the engine with no shared base class.

    Includes a ``:recovery:`` row for the same reason the SQLite case does: the
    filter direction is the one thing about this method that is easy to get
    backwards, and the twins have no ABC to keep them honest about it.
    """
    store, user_id = pg_store_with_credits

    _settle_call_pg(store, user_id, run_id="run-a", call_index=0, actual_micro=25_000)
    _settle_call_pg(store, user_id, run_id="run-a", call_index=1, actual_micro=17_318)
    _settle_call_pg(store, user_id, run_id="run-b", call_index=0, actual_micro=900_000)

    assert store.sum_run_llm_spend(user_id, "run-a") == (42_318, 2)
    assert store.sum_run_llm_spend(user_id, "run-never-ran") == (0, 0)
    assert store.sum_run_llm_spend(user_id + 9999, "run-a") == (0, 0)

    with store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT reservation_id FROM credit_llm_reservations WHERE run_id = %s ORDER BY call_index LIMIT 1", ("run-a",))
            reservation_id = cur.fetchone()["reservation_id"]
            cur.execute(
                """
                INSERT INTO credit_llm_usage_entries (
                    user_id, reservation_id, run_id, call_index, bucket,
                    amount_micro, operation_key, evidence_json, created_at
                ) VALUES (%s, %s, %s, %s, 'purchased', %s, %s, '{"t":"test"}', %s)
                """,
                (
                    user_id,
                    reservation_id,
                    "run-a",
                    0,
                    -4_000,
                    f"{reservation_id}:recovery:test:purchased",
                    "2026-09-14T00:00:00+00:00",
                ),
            )

    assert store.sum_run_llm_spend(user_id, "run-a") == (46_318, 2)
```

Write `_settle_call_pg` as a module-level helper mirroring `_settle_activity_call` from the SQLite module (reserve, then settle), using the Postgres store's identical `reserve_llm_credits` / `settle_llm_credits` signatures. If the module already has an equivalent helper, use that instead.

- [ ] **Step 7: Run the Postgres tests to verify the static one fails**

Run: `pytest dashboard/backend/tests/domain/credits/test_repository_postgres.py -k llm_usage_by_user_and_run -v`
Expected: FAIL — `assert 'idx_credit_llm_usage_user_run' in ...`

The `@pg_only` case skips without `TEST_POSTGRES_URL`. That is expected and is not "passing" — say so in the report.

- [ ] **Step 8: Add the Postgres index and method**

In `dashboard/backend/domain/credits/repository_postgres.py`, inside `CREDITS_POSTGRES_DDL`, directly after the existing `idx_credit_llm_usage_user_id` statement (~`:329-330`):

```sql
CREATE INDEX IF NOT EXISTS idx_credit_llm_usage_user_run
ON credit_llm_usage_entries(user_id, run_id);
```

It belongs in the base DDL, **not** in `CREDITS_POSTGRES_GRANT_MIGRATION_DDL`: both columns are original columns of `credit_llm_usage_entries`, not ones an `ALTER TABLE … ADD COLUMN` introduces, so the #432 ordering rule the module comment describes does not apply. `test_postgres_twin_indexes_a_migrated_column_only_after_adding_it` enforces that rule and will confirm it.

Then, immediately after `release_run_llm_reservations` (~`:1377`):

```python
    def sum_run_llm_spend(self, user_id: int, run_id: str) -> tuple[int, int]:
        """Settled LLM spend for one run: ``(micro_credits, distinct_calls)``.

        Twin of ``CreditsStore.sum_run_llm_spend``; see that docstring for why
        ``:recovery:`` entries are included rather than filtered.
        """
        _positive_integer(user_id, "user_id")
        run_id = _required_text(run_id, "run_id", max_length=128)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COALESCE(SUM(-amount_micro), 0) AS spent_micro,
                        COUNT(DISTINCT call_index) AS call_count
                    FROM credit_llm_usage_entries
                    WHERE user_id = %s AND run_id = %s
                    """,
                    (user_id, run_id),
                )
                row = cur.fetchone()
        return (int(row["spent_micro"] or 0), int(row["call_count"] or 0))
```

The pool sets `row_factory=dict_row` (`dashboard/backend/db_pool.py:49`), so `row["spent_micro"]` is correct here.

- [ ] **Step 9: Add the service passthrough**

In `dashboard/backend/domain/credits/service.py`, immediately after `release_run_llm_reservations` (~`:414`):

```python
    def sum_run_llm_spend(self, user_id: int, run_id: str) -> tuple[int, int]:
        """Settled LLM spend for one run: ``(micro_credits, distinct_calls)``.

        A read, so unlike its neighbours it emits no analytics buckets: it is
        reporting what was already recorded, not recording anything.
        """
        return self.store.sum_run_llm_spend(user_id, run_id)
```

- [ ] **Step 10: Run the credits and parity suites**

Run: `pytest dashboard/backend/tests/domain/credits/ dashboard/backend/tests/test_store_twin_parity.py -v`
Expected: PASS. `test_postgres_twin_exposes_every_sqlite_method` and `test_postgres_twin_signatures_match_sqlite` both cover the new method — if either fails, the two signatures disagree.

- [ ] **Step 11: Commit**

```bash
git add dashboard/backend/domain/credits/repository.py \
        dashboard/backend/domain/credits/repository_postgres.py \
        dashboard/backend/domain/credits/service.py \
        dashboard/backend/tests/domain/credits/test_repository.py \
        dashboard/backend/tests/domain/credits/test_repository_postgres.py
git commit -m "feat: sum settled LLM spend for one run"
```

---

## Task 2: Analytics — the `run_timeout` error category

**Files:**
- Modify: `dashboard/backend/domain/analytics/models.py:59-68` (`ALLOWED_ERROR_CATEGORIES`)
- Modify: `dashboard/backend/domain/analytics/repository.py:60-68` (SQLite DDL CHECK) and `:325` (the migration sentinel)
- Modify: `dashboard/backend/domain/analytics/repository_postgres.py:58-65` (base DDL CHECK) and `:230-242` (the `ADD CONSTRAINT` in `_init_schema`)
- Test: `dashboard/backend/tests/domain/analytics/test_models.py`
- Test: `dashboard/backend/tests/domain/analytics/test_repository_contract.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `"run_timeout"` accepted as an `error_category` by `AnalyticsStore`, `PostgresAnalyticsStore`, and `analytics_instrumentation.emit_run_event`. Task 3 emits it.

**Why not a new event name.** `backtest_failed` stays. A cancel earned `backtest_cancelled` because a cancel is *not* a failure — the user changed their mind. A timeout **is** a failure: the user asked for a backtest, got nothing, and was charged. A new event name would need nine registrations, three of which (`metrics.py::_TERMINAL_FAILURE`, `states.py`'s consecutive-failure alert, `backfill.py::_TERMINAL_RUN_EVENTS`) would silently stop counting timeouts toward the metrics that exist to notice runs failing. `error_category` is the field that exists to carry the reason.

**Not `provider_timeout`.** That means the upstream model provider timed out — a different fact with a different remedy.

- [ ] **Step 1: Write the failing tests**

In `dashboard/backend/tests/domain/analytics/test_models.py`, extend the existing assertion at `:88` (which reads `{...} <= ALLOWED_ERROR_CATEGORIES`) by adding `"run_timeout"` to that set literal, and append:

```python
def test_run_timeout_is_its_own_error_category():
    """A budget exhaustion must be distinguishable from a crash.

    Before this, `_finalize_slot_locked` hardcoded `internal_error` for every
    non-None error, so the product's own analytics could not tell "we ran out of
    the hour we gave ourselves" from "something threw".
    """
    assert "run_timeout" in ALLOWED_ERROR_CATEGORIES
    assert "internal_error" in ALLOWED_ERROR_CATEGORIES
```

In `dashboard/backend/tests/domain/analytics/test_repository_contract.py`, append:

```python
def test_sqlite_accepts_run_timeout_category(sqlite_contract):
    store, _admin_id, user_id = sqlite_contract
    event = event_record(
        user_id,
        event_id="10000000-0000-4000-8000-000000000007",
        event_name="backtest_failed",
        event_group="run",
        event_source="server",
        source_event_id="run:timeout-contract",
        session_id=None,
        page_view=None,
        error_category="run_timeout",
    )
    assert store.append_event(event).event.error_category == "run_timeout"


def test_sqlite_migrates_a_database_stuck_on_the_previous_category_set(tmp_path):
    """The sentinel trap, pinned.

    `_migrate_error_category_constraint` skips the table rebuild when the live
    CHECK already names the LAST category added. Leave that sentinel on
    `provider_quota_exhausted` while adding `run_timeout` and every EXISTING
    SQLite database keeps its old CHECK -- rejecting the new category at write
    time -- while a fresh database accepts it. Green tests, broken prod. The
    only way to see it is to build a database at the previous generation, which
    is what this does.
    """
    db_path = tmp_path / "analytics-pre-run-timeout.db"
    users = UserStore(db_path=db_path)
    user = users.create_user(
        "pre-run-timeout@example.test",
        "Pre Run Timeout User",
        "SecurePass1!",
    )
    previous_ddl = ANALYTICS_SQLITE_DDL.replace(
        "'model_not_allowed', 'internal_error', 'run_timeout'",
        "'model_not_allowed', 'internal_error'",
    )
    assert previous_ddl != ANALYTICS_SQLITE_DDL, (
        "the replace found no anchor -- update it to match the new DDL text, "
        "or this test silently asserts nothing"
    )
    with sqlite3.connect(db_path) as conn:
        conn.executescript(previous_ddl)

    store = AnalyticsStore(db_path=db_path)
    event = event_record(
        int(user["id"]),
        event_id="10000000-0000-4000-8000-000000000008",
        event_name="backtest_failed",
        event_group="run",
        event_source="server",
        source_event_id="run:timeout-migration",
        session_id=None,
        page_view=None,
        error_category="run_timeout",
    )
    assert store.append_event(event).created is True
```

The `assert previous_ddl != ANALYTICS_SQLITE_DDL` line is load-bearing: `str.replace` on a missing anchor returns the string unchanged, which would make this test build a *current* database and assert nothing at all. The two existing legacy-migration tests in this file have the same hazard and no such guard — do not copy their omission.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/domain/analytics/ -k run_timeout -v`
Expected: FAIL — `assert 'run_timeout' in ALLOWED_ERROR_CATEGORIES`, and the contract cases fail on the CHECK constraint.

- [ ] **Step 3: Add the category in all five places**

`dashboard/backend/domain/analytics/models.py:59-68`:

```python
ALLOWED_ERROR_CATEGORIES = {
    "credential_invalid",
    "credential_missing",
    "provider_timeout",
    "provider_unavailable",
    "provider_quota_exhausted",
    "credits_unavailable",
    "model_not_allowed",
    "internal_error",
    # The parent's own wall-clock budget ran out, not the provider's. A run
    # that hit this returned nothing AND was billed for what settled first, so
    # it stays a failure -- `backtest_failed`, outcome `failed` -- and this
    # field carries the reason. `provider_timeout` is a different fact with a
    # different remedy.
    "run_timeout",
}
```

`dashboard/backend/domain/analytics/repository.py:61-68` (inside `ANALYTICS_SQLITE_DDL`):

```sql
    error_category TEXT CHECK (
        error_category IS NULL OR error_category IN (
            'credential_invalid', 'credential_missing', 'provider_timeout',
            'provider_unavailable', 'provider_quota_exhausted',
            'credits_unavailable',
            'model_not_allowed', 'internal_error', 'run_timeout'
        )
    ),
```

Keep `'provider_unavailable', 'provider_quota_exhausted',\n            'credits_unavailable',` on its own lines exactly as it is now — `test_sqlite_migrates_legacy_error_category_constraint` and `test_sqlite_defers_error_category_migration_until_users_table_exists` both `.replace()` on that exact substring, and re-wrapping it breaks them silently (their `.replace()` would no-op and they would stop testing the migration).

`dashboard/backend/domain/analytics/repository.py:325` — the sentinel:

```python
        # Keyed on the LAST category added, not on any stable marker: a database
        # whose CHECK already names it is current, and anything older needs the
        # table rebuilt. BUMP THIS EVERY TIME A CATEGORY IS ADDED. Leave it
        # behind and existing databases keep the previous CHECK -- rejecting the
        # new value at write time while a fresh database accepts it, so the
        # whole suite stays green and only prod breaks.
        if "run_timeout" in table_sql:
            return
```

`dashboard/backend/domain/analytics/repository_postgres.py:58-65` (inside `ANALYTICS_POSTGRES_DDL`) — the same CHECK, same added value:

```sql
    error_category TEXT CHECK (
        error_category IS NULL OR error_category IN (
            'credential_invalid', 'credential_missing', 'provider_timeout',
            'provider_unavailable', 'provider_quota_exhausted',
            'credits_unavailable',
            'model_not_allowed', 'internal_error', 'run_timeout'
        )
    ),
```

`dashboard/backend/domain/analytics/repository_postgres.py:230-242` — the `ADD CONSTRAINT` inside `_init_schema`:

```python
                cur.execute(
                    """
                    ALTER TABLE analytics_events
                    ADD CONSTRAINT analytics_events_error_category_check
                    CHECK (
                        error_category IS NULL OR error_category IN (
                            'credential_invalid', 'credential_missing',
                            'provider_timeout', 'provider_unavailable',
                            'provider_quota_exhausted', 'credits_unavailable',
                            'model_not_allowed', 'internal_error',
                            'run_timeout'
                        )
                    )
                    """
                )
```

The Postgres twin needs no sentinel: `_init_schema` already does an unconditional `DROP CONSTRAINT IF EXISTS` immediately before this `ADD CONSTRAINT`, so it self-heals on every boot. That asymmetry is why the SQLite sentinel is the easy one to miss.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/domain/analytics/ -v`
Expected: PASS, including the two pre-existing legacy-migration cases.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/models.py \
        dashboard/backend/domain/analytics/repository.py \
        dashboard/backend/domain/analytics/repository_postgres.py \
        dashboard/backend/tests/domain/analytics/test_models.py \
        dashboard/backend/tests/domain/analytics/test_repository_contract.py
git commit -m "feat: add the run_timeout analytics error category"
```

---

## Task 3: The fourth terminal slot outcome and its status branch

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py` — `_slot_snapshot` (`:672-698`), `_try_acquire_backtest_slot` slot literal (`:763-781`), `_finalize_slot` (`:854-877`), `_finalize_slot_locked` (`:879-965`), `get_backtest_status` (`:3276`)
- Test: `dashboard/backend/tests/test_backtests_router.py`

**Interfaces:**
- Consumes: `"run_timeout"` from Task 2.
- Produces:
  - `_finalize_slot(live_run_id, *, error, runs_count, cancelled=False, timed_out=False, timeout_detail=None)` — Task 4 calls this.
  - `_finalize_slot_locked(...)` with the same two new keywords.
  - Slot keys `"timed_out": bool` and `"timeout_detail": Optional[Dict[str, Any]]`, both carried by `_slot_snapshot`.
  - `GET /backtest/status` answers `{"running": False, "timed_out": True, "elapsed_seconds": int, "live_run_id": str|None, "message": str, "timeout": {...}}` — the `timeout` key present only when the slot has a detail dict. Task 6 reads it.
  - `timeout_detail` shape: `{"limit_seconds": int, "billing_mode": str, "spent_micro": int|None, "model_calls": int|None}`.

**Design note — why `timeout` is a nested dict.** A conditionally-attached sub-object on a *terminal* branch is a new shape here; the only precedent, `progress`, is running-only. The alternative is four more flat keys on a route with no Pydantic response model to document them. Deliberate extension, not oversight.

**Design note — `model_calls` is payload-only.** No copy renders it (Task 6's three lines name the amount, not the count). It is in the payload because it is free from the same aggregate and it is what distinguishes "nothing had settled yet" from "the cost rounded to a very small number" for anyone reading the payload directly. Do not delete it as dead, and do not add copy for it.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_backtests_router.py`:

```python
def test_finalize_slot_records_a_timeout_as_its_own_outcome():
    """Four outcomes, not three: succeeded, failed, cancelled, timed out.

    A timeout is not a cancel (the user did nothing) and not a crash (nothing
    threw -- the product ran out of the budget it set itself, and billed for
    what settled first). Asserted as separate fields rather than one tuple
    equality so a failure names which field moved.
    """
    run_id = "agent_timeout_outcome"
    session_id = str(uuid.uuid4())
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=None
        )
        is None
    )

    detail = {
        "limit_seconds": 3600,
        "billing_mode": "platform_credits",
        "spent_micro": 42_318,
        "model_calls": 2,
    }
    bt._finalize_slot(
        run_id, error=None, runs_count=0, timed_out=True, timeout_detail=detail
    )

    slot = bt._recent_slots[run_id]
    assert slot["timed_out"] is True
    assert slot["cancelled"] is False
    assert slot["error"] is None
    assert slot["running"] is False
    assert slot["timeout_detail"] == detail


def test_status_reports_timed_out_as_neither_an_error_nor_a_success(client):
    """Both frontend surfaces branch on these keys.

    A timeout routed through `error` paints the red "Backtest did not start"
    panel for something the user did not cause; one routed through `success`
    would claim results that do not exist. Neither key may appear.
    """
    session_id = str(uuid.uuid4())
    headers = {"X-Session-Id": session_id}
    run_id = "agent_timeout_status"
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=None
        )
        is None
    )
    bt._finalize_slot(
        run_id,
        error=None,
        runs_count=0,
        timed_out=True,
        timeout_detail={
            "limit_seconds": 3600,
            "billing_mode": "platform_credits",
            "spent_micro": 42_318,
            "model_calls": 2,
        },
    )

    body = client.get(
        f"/backtest/status?live_run_id={run_id}", headers=headers
    ).json()

    assert body["running"] is False
    assert body["timed_out"] is True
    assert "error" not in body
    assert "success" not in body
    assert body["message"] == "Backtest stopped at the time limit."
    assert body["timeout"] == {
        "limit_seconds": 3600,
        "billing_mode": "platform_credits",
        "spent_micro": 42_318,
        "model_calls": 2,
    }


def test_status_omits_the_timeout_block_when_no_detail_was_recorded(client):
    """The branch must still answer for a slot finalized without a detail dict
    -- the shape is conditional, so absence is a real case, not an error."""
    session_id = str(uuid.uuid4())
    headers = {"X-Session-Id": session_id}
    run_id = "agent_timeout_no_detail"
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=None
        )
        is None
    )
    bt._finalize_slot(run_id, error=None, runs_count=0, timed_out=True)

    body = client.get(
        f"/backtest/status?live_run_id={run_id}", headers=headers
    ).json()

    assert body["timed_out"] is True
    assert "timeout" not in body


def test_timed_out_run_emits_backtest_failed_with_the_timeout_reason(monkeypatch):
    """A timeout IS a failure -- the user got nothing and was charged -- so it
    keeps `backtest_failed` and the success-rate KPI stays honest. What was
    missing is the reason, and `error_category` is the field that carries it."""
    emitted = []
    monkeypatch.setattr(
        bt.analytics_instrumentation,
        "emit_run_event",
        lambda **kwargs: emitted.append(kwargs),
    )

    run_id = "agent_timeout_analytics"
    session_id = str(uuid.uuid4())
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=4242
        )
        is None
    )
    bt._finalize_slot(run_id, error=None, runs_count=0, timed_out=True)

    assert emitted == [
        {
            "event_name": "backtest_failed",
            "user_id": 4242,
            "run_id": run_id,
            "error_category": "run_timeout",
        }
    ]
```

If `_try_acquire_backtest_slot` does not accept `user_id=4242` without a real account, set the slot's owner the way the neighbouring cancel tests do — read `dashboard/backend/tests/test_backtest_cancel.py`'s `_acquire` helper and match it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_backtests_router.py -k timed_out -v`
Expected: FAIL with `TypeError: _finalize_slot() got an unexpected keyword argument 'timed_out'`

- [ ] **Step 3: Add `timed_out` to the slot literal**

In `_try_acquire_backtest_slot` (`backtests.py:763-781`), beside `"cancelled": False`:

```python
            "cancel_requested": False,
            "cancelled": False,
            # The fourth terminal outcome. Seeded here rather than relied on as
            # a missing key, so `_slot_snapshot` and the status route can read
            # it with `.get()` and get False rather than None on every run that
            # did not time out.
            "timed_out": False,
            "timeout_detail": None,
```

- [ ] **Step 4: Carry both fields through `_slot_snapshot`**

In `_slot_snapshot` (`:684-697`), beside the `cancelled` line:

```python
        "cancelled": bool(slot.get("cancelled")),
        "timed_out": bool(slot.get("timed_out")),
        # The facts the status route reports back: budget, lane, spend, call
        # count. Composed once by the worker's timeout arm -- the only scope
        # holding all four at once -- and never recomputed on the read path.
        "timeout_detail": slot.get("timeout_detail"),
```

Also extend the `_slot_snapshot` docstring's existing sentence about `cancelled` — it explains that the field is listed because `/backtest/status` branches on it and a snapshot that dropped it would report a cancelled run as "no backtest has been run yet". Say the same of `timed_out`, in one clause.

- [ ] **Step 5: Add the two keywords to both finalize functions**

`_finalize_slot` (`:854-877`):

```python
def _finalize_slot(
    live_run_id: str,
    *,
    error: Optional[str],
    runs_count: int,
    cancelled: bool = False,
    timed_out: bool = False,
    timeout_detail: Optional[Dict[str, Any]] = None,
) -> None:
    """Move a slot to its terminal state.

    Four outcomes, not two. ``cancelled`` is deliberately NOT routed through
    ``error``: a cancel is the owner's own deliberate action, and reporting it
    back to them as a failure is the same class of lie as reporting a run the
    model never drove as a clean success (issue #169, the change this one
    stacks on). The status route branches on it, and the analytics event below
    is ``backtest_cancelled`` rather than ``backtest_failed`` for the same
    reason -- a dashboard that counts cancels as failures measures the product
    as broken every time a user changes their mind.

    ``timed_out`` is the fourth, and it is neither of the other two. Not a
    cancel: the user did nothing, and telling them they stopped their own run
    is a different lie in the same family. Not a crash: nothing threw -- the
    product ran out of the wall-clock budget it set itself, and billed them for
    the model calls that settled before it did. So ``error`` stays None here
    (routing it through ``error`` lands in the existing error branch and
    nothing on screen changes), the user-facing outcome is its own status
    branch, and the analytics event below stays ``backtest_failed`` with
    ``error_category="run_timeout"`` -- because unlike a cancel, a timeout IS a
    failure, and moving it out of that event would quietly lift timeouts out of
    the success-rate KPI.

    ``timeout_detail`` is facts only -- budget, lane, spend, call count -- never
    a sentence. The client composes the copy so the copy register can see it.
    """
    with _backtest_slots_lock:
        event = _finalize_slot_locked(
            live_run_id,
            error=error,
            runs_count=runs_count,
            cancelled=cancelled,
            timed_out=timed_out,
            timeout_detail=timeout_detail,
        )
    _emit_slot_run_event(event)
```

`_finalize_slot_locked` (`:879-887`) takes the same two keywords with the same defaults. Leave its existing docstring alone except to note that the emit-outside-the-lock rule now also covers the credits read its caller must do (one sentence).

- [ ] **Step 6: Write the fields and branch the analytics return**

In `_finalize_slot_locked`, beside `slot["cancelled"] = bool(cancelled)` (`:935`):

```python
    slot["cancelled"] = bool(cancelled)
    slot["timed_out"] = bool(timed_out)
    slot["timeout_detail"] = timeout_detail
```

And in the analytics return block (`:951-964`), insert the timeout arm **between** the cancel arm and the succeeded/failed computation:

```python
    if cancelled:
        return {
            "event_name": "backtest_cancelled",
            "user_id": int(user_id),
            "run_id": live_run_id,
            "error_category": None,
        }
    if timed_out:
        # Still `backtest_failed`: the user asked for a backtest, got nothing,
        # and was billed. A separate event name would need nine registrations
        # and would drop timeouts out of `metrics.py`'s `_TERMINAL_FAILURE`,
        # `states.py`'s consecutive-failure alert and `backfill.py`'s terminal
        # map -- the three things that exist to notice runs failing. What was
        # missing is the reason, not the outcome.
        return {
            "event_name": "backtest_failed",
            "user_id": int(user_id),
            "run_id": live_run_id,
            "error_category": "run_timeout",
        }
    succeeded = error is None and runs_count > 0
```

- [ ] **Step 7: Add the status route branch**

In `get_backtest_status`, **after** the `elif slot.get("cancelled"):` block and **before** `elif slot.get("error"):`:

```python
    elif slot.get("timed_out"):
        # Between cancelled and error, and never routed through either. The
        # shape mirrors the cancel branch -- no `error` key, no `success` key --
        # because nothing failed and nothing completed. What is new is
        # `timeout`, a facts-only sub-object the client turns into a sentence:
        # the amount has to be formatted by the same helper the Credits page
        # uses, and composing the sentence here would put the number's
        # formatting and the user's copy under two different owners.
        payload = {
            "running": False,
            "timed_out": True,
            "elapsed_seconds": int(slot.get("elapsed_seconds") or 0),
            "live_run_id": slot.get("live_run_id"),
            # Fallback for any client that does not know this branch, matching
            # "Backtest cancelled." above.
            "message": "Backtest stopped at the time limit.",
        }
        timeout_detail = slot.get("timeout_detail")
        if timeout_detail:
            payload["timeout"] = timeout_detail
        return payload
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/test_backtest_cancel.py -v`
Expected: PASS. The cancel suite must stay green — the new branch sits after the cancel branch precisely so a cancel can never fall into it.

- [ ] **Step 9: Mutation-check the branch order**

Temporarily move the new `elif slot.get("timed_out"):` block **above** the `elif slot.get("cancelled"):` block and re-run `pytest dashboard/backend/tests/test_backtest_cancel.py -v`. If it still passes, the ordering is not actually pinned — add an assertion to `test_finalize_slot_records_a_timeout_as_its_own_outcome` that a slot finalized with `cancelled=True` reports `cancelled`, not `timed_out`. Restore the original order before committing.

- [ ] **Step 10: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py \
        dashboard/backend/tests/test_backtests_router.py
git commit -m "feat: give a timed-out backtest its own terminal outcome"
```

---

## Task 4: Detect the timeout and record what it cost

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py` — `run_backtest_background` signature (`:1381-1401`), the new `except` arm (before `:1636`), the `finally`'s `finalize_run` call (`:1652-1655`), the thread kwargs in `run_backtest_endpoint` (`:3178-3196`)
- Test: `dashboard/backend/tests/test_backtests_router.py` (extend `test_pipeline_timeout_finalizes_execution_and_slot_once` at `:917`)

**Interfaces:**
- Consumes: `credits_service.sum_run_llm_spend(user_id, run_id)` (Task 1), `_finalize_slot(..., timed_out=True, timeout_detail=...)` (Task 3).
- Produces: `run_backtest_background(..., billing_mode: Optional[str] = None)` — one new keyword-defaulted parameter at the end of the signature.

**⚠ The lane is not in scope, so it must be threaded.** `billing_mode` exists only in `run_backtest_endpoint` (`:2825`), where it is folded into the opaque, signed `execution_handoff_payload` before the thread is started. The background function sees the payload string, never the enum, and cannot decode it (the handoff TTL is 300s against a 3600s timeout). Hence the new parameter.

**⚠ The account id comes from the slot, not a second parameter.** `run_backtest_background` has no `user_id` — `session_id` is the browser session. `_slot_analytics_user_id(live_run_id)` (`:847`) already exists and already does the lock-held read of `_active_slots`/`_recent_slots`; reuse it. The slot is the single owner of *whose run this is*, and a second parameter could drift from it.

**⚠ `resolved_live_run_id = None` after finalizing is mandatory.** Without it the `finally`'s `if resolved_live_run_id: _finalize_slot(..., error=None, runs_count=0)` runs a second time and overwrites the timeout with a fake zero-run success — silently erasing the outcome this whole change exists to report. The `_BacktestCancelled` and generic arms both do this and say why.

- [ ] **Step 1: Extend the existing timeout test**

`test_pipeline_timeout_finalizes_execution_and_slot_once` (`:917`) already drives this exact path via `FakeChild(timeout_waits=1)`. Its `spy_finalize_slot` will raise `TypeError` once Task 3's keywords exist, so it must be updated either way. Replace the spy and extend the assertions:

```python
    def spy_finalize_slot(
        live_run_id,
        *,
        error,
        runs_count,
        cancelled=False,
        timed_out=False,
        timeout_detail=None,
    ):
        finalized_slots.append((live_run_id, runs_count))
        return real_finalize_slot(
            live_run_id,
            error=error,
            runs_count=runs_count,
            cancelled=cancelled,
            timed_out=timed_out,
            timeout_detail=timeout_detail,
        )
```

and append to the existing assertions at the end of that test:

```python
    # The outcome, not just the cleanup. Before this the TimeoutExpired fell
    # into the generic `except Exception` arm and the user received a tail
    # slice of the child's argv -- `_sanitize_backtest_error` truncates from
    # the END, which is right for a stack trace and wrong for a TimeoutExpired
    # whose informative clause trails a long command line.
    assert bt._recent_slots[run_id]["timed_out"] is True
    assert bt._recent_slots[run_id]["error"] is None
    # Exactly once. A second _finalize_slot from the `finally` would overwrite
    # the timeout with a zero-run completion.
    assert finalized_slots == [(run_id, 0)]
```

Then add the two lane cases:

```python
def _drive_pipeline_timeout(monkeypatch, *, run_id, billing_mode, user_id=None):
    """Run the real worker to a parent timeout and return the finalized slot.

    Shared by the two lane cases below rather than copied: the setup is eight
    monkeypatches and a FakeChild, and a copy that drifts in one of them stops
    testing the branch it names.
    """
    session_id = str(uuid.uuid4())
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=user_id
        )
        is None
    )

    child = FakeChild(stdout="run header line\n", timeout_waits=1)

    class FakeExecutionService:
        def __init__(self, **_kwargs):
            pass

        def finalize_run(self, execution_run_id, *, billing_mode=None):
            return []

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: child)
    monkeypatch.setattr(bt, "LLMExecutionService", FakeExecutionService)
    monkeypatch.setattr(bt, "get_model_provider_service", lambda: object())
    monkeypatch.setattr(bt, "run_backtest_background", _REAL_RUN_BACKTEST_BACKGROUND)
    monkeypatch.setattr(
        bt.credits_service,
        "sum_run_llm_spend",
        lambda uid, rid: (42_318, 2),
    )

    bt.run_backtest_background(
        start_date="2026-01-01",
        end_date="2026-01-02",
        session_id=session_id,
        live_run_id=run_id,
        decision_source="rule_based",
        execution_handoff_payload="opaque-test-handoff",
        billing_mode=billing_mode,
    )
    return bt._recent_slots[run_id]


def test_platform_credits_timeout_discloses_what_the_run_cost(monkeypatch):
    """The whole point: the user waited an hour, got nothing, and was billed.

    `finalize_run` in the `finally` only RELEASES open holds -- it never settles
    -- so every settled row was already written by the child as it went, and the
    sum read here is final.
    """
    slot = _drive_pipeline_timeout(
        monkeypatch,
        run_id="agent_timeout_platform",
        billing_mode="platform_credits",
        user_id=4242,
    )

    assert slot["timed_out"] is True
    assert slot["timeout_detail"] == {
        "limit_seconds": 3600,
        "billing_mode": "platform_credits",
        "spent_micro": 42_318,
        "model_calls": 2,
    }


def test_byok_timeout_reports_no_spend_rather_than_zero(monkeypatch):
    """None, not 0. BYOK never touches the ATL ledger, so a zero here would be
    indistinguishable from "platform lane, nothing settled yet" -- and the card
    would tell a BYOK user their run cost 0.000000 Credits, which is a claim
    about a ledger that has no row for them at all."""
    slot = _drive_pipeline_timeout(
        monkeypatch,
        run_id="agent_timeout_byok",
        billing_mode="byok",
        user_id=4242,
    )

    assert slot["timeout_detail"] == {
        "limit_seconds": 3600,
        "billing_mode": "byok",
        "spent_micro": None,
        "model_calls": None,
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_backtests_router.py -k "pipeline_timeout or timeout_discloses or timeout_reports_no_spend" -v`
Expected: FAIL with `TypeError: run_backtest_background() got an unexpected keyword argument 'billing_mode'`

- [ ] **Step 3: Add the parameter to `run_backtest_background`**

At the end of the signature (`backtests.py:1400`), after `universe_selection`:

```python
    universe_selection: Optional[Dict[str, Any]] = None,
    # The billing lane, threaded from the route because it is NOT otherwise
    # reachable here: it is folded into the opaque signed
    # `execution_handoff_payload` before this function is called, and that
    # envelope's TTL (300s) is far shorter than this run's budget (3600s), so
    # decoding it at timeout time would fail even if we had the key. Used only
    # to decide whether a timeout has a Credits cost worth reporting.
    billing_mode: Optional[str] = None,
):
```

- [ ] **Step 4: Add the `except subprocess.TimeoutExpired` arm**

Insert **between** `except _BacktestCancelled:` and `except Exception as e:` (i.e. immediately before `backtests.py:1636`):

```python
    except subprocess.TimeoutExpired:
        # Ahead of the generic arm, which would send this through
        # `_sanitize_backtest_error` -- `_redact_credentials(...)[-max_chars:]`,
        # a TAIL truncation. That is right for a stack trace and wrong for a
        # TimeoutExpired, whose informative clause trails a long argv: the user
        # waits up to an hour and receives a fragment of a command line.
        print(
            f"⏱️  Backtest hit the {subprocess_timeout}s limit: "
            f"{resolved_live_run_id}",
            flush=True,
        )
        if resolved_live_run_id:
            # Read the owner under the ledger lock, then run the credits query
            # OUTSIDE it. `_finalize_slot_locked`'s docstring gives the rule for
            # the analytics emit -- "it can reach a store, and this lock is
            # taken by every status poll and every launch" -- and a credits
            # aggregate is the same hazard, over a bigger table.
            timeout_user_id = _slot_analytics_user_id(resolved_live_run_id)
            spent_micro = None
            model_calls = None
            if billing_mode == "platform_credits" and timeout_user_id is not None:
                # Complete at this moment: the `finally`'s `finalize_run` only
                # RELEASES open reservations (see
                # `release_run_llm_reservations`), it never settles, so every
                # settled row was written by the child as it went. The hold for
                # the call that was interrupted is released, not charged --
                # excluding it is the right answer, not a rounding error.
                try:
                    spent_micro, model_calls = credits_service.sum_run_llm_spend(
                        timeout_user_id, resolved_live_run_id
                    )
                except Exception as exc:  # a read, and never worth losing the outcome over
                    print(
                        f"⚠️ timeout spend lookup failed for "
                        f"{resolved_live_run_id}: {exc}",
                        flush=True,
                    )
            _finalize_slot(
                resolved_live_run_id,
                error=None,
                runs_count=0,
                timed_out=True,
                timeout_detail={
                    # The budget THIS run was given, read from the local bound
                    # at the call site above -- never re-derived from
                    # `_backtest_subprocess_timeout` and never hardcoded to
                    # 3600, so the card cannot report a budget the run did not
                    # actually have.
                    "limit_seconds": int(subprocess_timeout),
                    "billing_mode": billing_mode or "byok",
                    "spent_micro": spent_micro,
                    "model_calls": model_calls,
                },
            )
            # MANDATORY, exactly as in the two arms above: without it the
            # `finally`'s `if resolved_live_run_id:` finalizes a second time
            # with `error=None, runs_count=0` and overwrites the timeout with a
            # fake zero-run success.
            resolved_live_run_id = None
```

`subprocess_timeout` is the local assigned at `:1553-1555`, immediately before `_run_backtest_subprocess`. A `TimeoutExpired` can only be raised by that subprocess's own wait, so the name is always bound when this arm runs.

**Do not** add a `str(e)` or read `e.output`/`e.stderr` here. The child's captured output on a timeout is a known gap (see the spec's "What this does not close"); keeping it out of the user-facing outcome is deliberate.

- [ ] **Step 5: Pass the lane from the route**

In `run_backtest_endpoint`'s thread kwargs (`:3180-3196`), add one entry beside `execution_handoff_payload`:

```python
            "execution_handoff_payload": execution_handoff_payload,
            "billing_mode": billing_mode.value if billing_mode is not None else None,
```

`.value`, not the enum: the worker compares against the string `"platform_credits"`, and a thread argument that carries an enum through a module the tests monkeypatch is one more thing that can be the wrong type at the one moment it matters.

- [ ] **Step 6: Pass the lane to `finalize_run` in the `finally`**

At `:1652-1655`:

```python
                LLMExecutionService(
                    providers=get_model_provider_service(),
                    credits=credits_service,
                ).finalize_run(execution_run_id, billing_mode=billing_mode)
```

`finalize_run`'s own docstring says "Worker callers should pass their known lane; retaining the in-process set keeps the one-argument form safe too" — but that in-process set (`_platform_runs`) belongs to the *child's* service instance and the parent constructs a fresh one, so it is always empty here and a BYOK run reaches `release_run_llm_reservations` on this path today. It finds nothing (BYOK never reserves), so nothing is broken; the documented contract simply is not holding, and the lane we now have makes it hold. **This step is independently droppable** — if a reviewer objects to bundling it, revert this one step and everything else still works.

Check `finalize_run`'s signature at `dashboard/backend/infrastructure/llm/execution/service.py:200` before editing: it is `finalize_run(run_id, *, billing_mode=None)` and `billing_mode` there is the `BillingMode` enum, not a string. Convert at the call site to match whatever that signature declares — if it wants the enum, pass the enum from the route instead of `.value` for this call only, or re-derive it. Do not change `finalize_run`'s own signature.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/test_backtest_cancel.py dashboard/backend/tests/test_credit_metering.py dashboard/backend/tests/test_market_data_features.py -v`
Expected: PASS. The last two exercise `run_backtest_background`'s signature through spies; a positional-argument mistake in Step 3 shows up there.

- [ ] **Step 8: Verify a cancel still beats a timeout**

No code needed — confirm by reading. The cancel route finalizes the slot under the ledger lock as it accepts, so by the time this arm calls `_finalize_slot` the run is already in `_recent_slots` with `cancelled=True`, and `_finalize_slot_locked`'s `if finished is not None and finished.get("cancelled"): return None` guard leaves it alone. Note this in the task report; add no test (`test_backtest_cancel.py` already pins that guard).

- [ ] **Step 9: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py \
        dashboard/backend/tests/test_backtests_router.py
git commit -m "feat: detect a pipeline backtest timeout and record its cost"
```

---

## Task 5: Split the poll ceiling from the progress-bar denominator

**Files:**
- Modify: `dashboard/frontend/app.js:20` (add a second constant), `:8039` (the `maxSeconds` default), `:8640` (the ceiling message)
- Test: `dashboard/backend/tests/test_backtest_progress_card.py` (`test_frontend_backtest_observation_window_is_sixty_minutes` at `:581`, the `_run_panel` harness at `:498`, `_resolve_entry` at `:592`, `_list_entries` at `:618`, and the card harness at `:287`)
- Test: `dashboard/backend/tests/test_running_backtest_store.py:102` (harness prelude)
- Test: `dashboard/backend/tests/test_app_copy_register.py:199-206`
- Test: `dashboard/backend/tests/test_ifind_ashare_frontend.py` (new source-shape case, following `test_the_ifind_prefill_window_is_runnable_under_the_server_cap` at `:554`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `BACKTEST_BUDGET_SECONDS = 3600` and `BACKTEST_POLL_MAX_SECONDS = 4200` as top-level `const`s in `app.js`. Task 6 uses neither directly, but its harness must load `BACKTEST_BUDGET_SECONDS` wherever it loads `updateBacktestRunProgress`.

**The defect this closes (#474 item 5).** `BACKTEST_POLL_MAX_SECONDS` is currently `3600`, byte-identical to the server's `PIPELINE_SUBPROCESS_TIMEOUT_SECONDS`. At the ceiling the poller stops, clears the live run, and paints its own guess — at the same instant the server begins finalizing. Whatever the server concludes is written after the client stopped looking.

**Why two constants and not one bumped value.** The constant has **five** consumers and they do not all want the same number:

| Site | What it is | Which constant |
|---|---|---|
| `ensureBacktestPolling` `:8336` | poll ceiling | `POLL_MAX` |
| `pollBacktestStatus` `:10288` | poll ceiling | `POLL_MAX` |
| `listRunningBacktests` `:5924` | orphan-entry staleness GC | `POLL_MAX` |
| `getAgentBacktestRunning` `:6001` | orphan-entry staleness GC | `POLL_MAX` |
| `updateBacktestRunProgress` `maxSeconds` default `:8039` | **progress-bar denominator** | **`BUDGET`** |

The last row answers "how far through its budget is this run?", so it must track the server budget. Raise the single constant and the bar silently starts under-reporting progress toward the limit it is drawing. The two GC sites correctly take the poll ceiling — an entry must not be reaped before the server has had its chance to answer.

- [ ] **Step 1: Write the failing source-shape test**

Append to `dashboard/backend/tests/test_ifind_ashare_frontend.py` (the module that already owns the client-vs-server constant guards):

```python
def test_the_client_outlives_the_server_budget_it_draws(js):
    """Two constants, two different jobs -- and the reason they must not
    reconverge (issue #474 item 5).

    `BACKTEST_BUDGET_SECONDS` is the progress bar's denominator, so it has to BE
    the server's budget. `BACKTEST_POLL_MAX_SECONDS` is how long the client
    watches, so it has to be LONGER -- otherwise the poller gives up at the same
    instant the server starts finalizing, and the verdict is written after the
    client stopped looking. They were one constant holding one value, which made
    that failure invisible.

    Asserted against the imported server constants rather than copies, so the
    client cannot drift from the budget it is drawing.
    """
    import re as _re

    from dashboard.backend.api.routers.backtests import (
        PIPELINE_SUBPROCESS_TIMEOUT_SECONDS,
        SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS,
    )

    budget = _re.search(r"const BACKTEST_BUDGET_SECONDS = (\d+);", js)
    ceiling = _re.search(r"const BACKTEST_POLL_MAX_SECONDS = (\d+);", js)
    assert budget and ceiling, "both backtest window constants must exist in app.js"

    assert int(budget.group(1)) == PIPELINE_SUBPROCESS_TIMEOUT_SECONDS
    assert int(ceiling.group(1)) > int(budget.group(1)), (
        "the client must keep polling past the server's budget, or it can never "
        "receive the server's own verdict"
    )
    assert (
        int(ceiling.group(1))
        == PIPELINE_SUBPROCESS_TIMEOUT_SECONDS + SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS
    )
```

Check the `js` fixture's definition in that module first — if it yields the app.js text, use it as written; if it yields something else, read `APP_JS` from `dashboard.backend.tests._frontend_source` instead.

Update `dashboard/backend/tests/test_app_copy_register.py::test_backtest_hint_uses_strategy_and_limit`:

```python
def test_backtest_hint_uses_strategy_and_limit():
    assert "Multi-step strategies can take several minutes (limit: 60 minutes)." in _HTML
    assert "Multi-step strategies can take several minutes (limit: 10 minutes)." not in _HTML
    assert "Chat with Claude; backtests use Alpaca historical data and multi-step runs can take several minutes." in _HTML
    assert "about 3–10 minutes" not in _HTML
    # The poll ceiling now sits ABOVE the server budget, so reaching it no
    # longer means the run timed out -- the server's own `timed_out` verdict
    # arrives ten minutes earlier. Reaching it means no terminal answer ever
    # arrived: a crash, a redeploy, a dropped connection. The replacement
    # carries no number, which removes the drift risk permanently.
    assert (
        "Lost contact with this backtest. It may still be running — check the "
        "Backtest tab later."
    ) in _JS
    assert "Timed out after 60 minutes." not in _JS
    assert "Timed out after 10 minutes." not in _JS
    assert "Multi-step agent pipelines" not in _HTML
```

Replace `test_frontend_backtest_observation_window_is_sixty_minutes` in `dashboard/backend/tests/test_backtest_progress_card.py:581-589` — it pins `BACKTEST_POLL_MAX_SECONDS == 3600` and is now wrong:

```python
def test_frontend_progress_bar_is_drawn_against_the_server_budget():
    """The denominator, executed rather than read.

    `updateBacktestRunProgress`'s `maxSeconds` default answers "how far through
    its budget is this run?", so it must be the server's number and not the
    client's watching window. The two were one constant; separating them is
    what stops raising the poll ceiling from silently shrinking every bar.
    """
    assert (
        _node(
            js_const("BACKTEST_BUDGET_SECONDS")
            + "console.log(JSON.stringify(BACKTEST_BUDGET_SECONDS));"
        )
        == 3600
    )


def test_frontend_keeps_polling_past_the_server_budget():
    assert (
        _node(
            js_const("BACKTEST_POLL_MAX_SECONDS")
            + "console.log(JSON.stringify(BACKTEST_POLL_MAX_SECONDS));"
        )
        == 4200
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_ifind_ashare_frontend.py -k outlives_the_server_budget dashboard/backend/tests/test_app_copy_register.py -k backtest_hint -v`
Expected: FAIL — `BACKTEST_BUDGET_SECONDS` is not in app.js; the copy register fails on the missing "Lost contact" string.

- [ ] **Step 3: Add the second constant**

Replace `dashboard/frontend/app.js:20`:

```js
// Two numbers, deliberately not one. The budget is the SERVER's -- it mirrors
// PIPELINE_SUBPROCESS_TIMEOUT_SECONDS and is what the progress bar is drawn
// against, so the bar answers "how far through its budget is this run?". The
// poll ceiling is how long this page keeps WATCHING, and it has to be longer:
// they were the same value, so the poller gave up at the same instant the
// server began finalizing and the server's own verdict was written after the
// client stopped looking (issue #474 item 5). The margin is the repo's own
// SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS. Pinned to both server constants by
// test_ifind_ashare_frontend.py.
const BACKTEST_BUDGET_SECONDS = 3600;   // mirrors PIPELINE_SUBPROCESS_TIMEOUT_SECONDS
const BACKTEST_POLL_MAX_SECONDS = 4200; // budget + SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS
```

- [ ] **Step 4: Point the denominator at the budget**

`dashboard/frontend/app.js:8039`:

```js
    maxSeconds = BACKTEST_BUDGET_SECONDS,
```

Leave the other four call sites on `BACKTEST_POLL_MAX_SECONDS` unchanged.

- [ ] **Step 5: Rewrite the ceiling message**

`dashboard/frontend/app.js:8637-8642` — inside the `if (attempts >= maxAttempts)` block:

```js
                if (isViewingLiveBacktest(liveBacktestRunId)) {
                    showBacktestRunProgress(true, { isError: true });
                    updateBacktestRunProgress({
                        elapsedSeconds: maxAttempts,
                        // Not "timed out": the ceiling now sits above the
                        // server's budget, so a real timeout arrives as a
                        // `timed_out` status ten minutes before this. Getting
                        // here means no terminal answer ever came -- a crash, a
                        // redeploy, a dropped connection. No number in it, so
                        // it cannot drift from a constant again.
                        message: 'Lost contact with this backtest. It may still be running — check the Backtest tab later.',
                    });
                }
```

Note the em dash (—), matching the copy-register assertion exactly.

**`app.html:1288` stays exactly as it is.** "Multi-step strategies can take several minutes (limit: 60 minutes)." describes the **server** budget, which does not move.

- [ ] **Step 6: Update every node harness prelude that loads the old constant**

`js_const("BACKTEST_POLL_MAX_SECONDS")` still resolves, so harnesses that only need the ceiling are fine. But any harness that executes `updateBacktestRunProgress` now needs `BACKTEST_BUDGET_SECONDS` too, or node raises `ReferenceError`. Add `js_const("BACKTEST_BUDGET_SECONDS"),` beside the existing `js_const("BACKTEST_POLL_MAX_SECONDS"),` line in:

- `dashboard/backend/tests/test_backtest_progress_card.py:287` (the card harness)
- `dashboard/backend/tests/test_backtest_progress_card.py:508` (`_run_panel`)
- `dashboard/backend/tests/test_backtest_progress_card.py:603` (`_resolve_entry`)
- `dashboard/backend/tests/test_backtest_progress_card.py:622` (`_list_entries`)
- `dashboard/backend/tests/test_backtest_progress_card.py:810`
- `dashboard/backend/tests/test_running_backtest_store.py:102`

Adding it to all six is cheaper than reasoning about which need it, and an unused `const` in a node prelude costs nothing. `test_run_panel_falls_back_to_the_elapsed_guess` (`:577`) asserts `width == "2%"  # 60 / 3600` — that stays correct, because the denominator is still 3600; update its comment to say `60 / BACKTEST_BUDGET_SECONDS`.

- [ ] **Step 7: Run the full frontend test set**

Run: `pytest dashboard/backend/tests/test_backtest_progress_card.py dashboard/backend/tests/test_running_backtest_store.py dashboard/backend/tests/test_app_copy_register.py dashboard/backend/tests/test_ifind_ashare_frontend.py -v`
Expected: PASS, with **no skips** on the node cases. If they skip, `node` is not on `PATH` and this step has verified nothing — install node and re-run before reporting.

- [ ] **Step 8: Commit**

```bash
git add dashboard/frontend/app.js \
        dashboard/backend/tests/test_backtest_progress_card.py \
        dashboard/backend/tests/test_running_backtest_store.py \
        dashboard/backend/tests/test_app_copy_register.py \
        dashboard/backend/tests/test_ifind_ashare_frontend.py
git commit -m "fix: keep polling a backtest past the server's own budget"
```

---

## Task 6: The timed-out card

**Files:**
- Modify: `dashboard/frontend/app.js` — new `formatBacktestTimeoutMessage`, new `renderBacktestTimeoutPanel`, `showBacktestRunProgress` (`:7992-8024`), the `finishedFocused` dispatch (`:8554-8584`)
- Modify: `dashboard/frontend/styles.css` (after the `.is-cancelled` rules, ~`:6636`)
- Test: `dashboard/backend/tests/test_backtest_progress_card.py`

**Interfaces:**
- Consumes: the `timed_out` / `timeout` status payload from Task 3; `BACKTEST_BUDGET_SECONDS` from Task 5.
- Produces: `formatBacktestTimeoutMessage(timeout) -> string` and `renderBacktestTimeoutPanel(status, displayElapsed) -> void` as top-level functions in `app.js`; `showBacktestRunProgress(show, { isError, isFinished, isCancelled, isTimedOut })`.

**Why a named helper rather than an inline branch.** The cancel and error branches are inline in `ensureBacktestPolling`'s `setInterval` callback, which `_frontend_source.fn_body` cannot slice — so neither is covered by a node test today. The spec requires node coverage of the timeout copy (including the BYOK omission and the branch ordering), and the only way to get it is a top-level function. `settleFinishedBacktestPanel` is the existing precedent for factoring a terminal-branch renderer out of that callback.

**Copy joins with a space, not newlines.** `updateBacktestRunProgress` writes `messageEl.textContent`, so a `\n` would render as a space anyway without a `white-space` rule. One paragraph, three sentences.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_backtest_progress_card.py`:

```python
def _timeout_message(timeout_js: str) -> str:
    """Run the real formatBacktestTimeoutMessage against the real formatter.

    `credit-format.js` is loaded rather than stubbed: the amount's precision is
    the whole reason the card and the Credits page agree, and a stub that
    rounded differently would let them diverge with every case still green.
    """
    format_js = (
        FRONTEND / "js" / "credit-format.js"
    ).read_text(encoding="utf-8")
    script = "\n".join(
        [
            "const window = globalThis;",
            format_js,
            fn_body("function formatBacktestTimeoutMessage("),
            f"console.log(JSON.stringify(formatBacktestTimeoutMessage({timeout_js})));",
        ]
    )
    return _node(script)


def test_timeout_card_names_the_limit_the_cost_and_both_levers():
    """Three facts, in the order a user needs them: what happened, what it cost,
    what to change. The third line names BOTH levers, matching the 422 that
    refuses an over-long pipeline window -- a user who meets both refusals
    should hear one story."""
    message = _timeout_message(
        "{limit_seconds: 3600, billing_mode: 'platform_credits',"
        " spent_micro: 42318, model_calls: 2}"
    )
    assert message == (
        "Stopped at the 60-minute limit. "
        "Model calls completed before the stop cost 0.042318 Credits. "
        "Shorten the date range, or use fewer pipeline steps, then run it again."
    )


def test_timeout_card_derives_the_minutes_from_the_budget_it_was_given():
    """Derived, never a literal. `app.js`'s old ceiling message hardcoded
    "60 minutes" and would have gone on saying it after the number moved."""
    message = _timeout_message(
        "{limit_seconds: 7200, billing_mode: 'byok',"
        " spent_micro: null, model_calls: null}"
    )
    assert message.startswith("Stopped at the 120-minute limit.")


def test_timeout_card_omits_the_cost_line_entirely_on_byok():
    """Not "0.000000 Credits" -- BYOK never touches the ATL ledger, so any
    amount at all is a claim about a row that does not exist."""
    message = _timeout_message(
        "{limit_seconds: 3600, billing_mode: 'byok',"
        " spent_micro: null, model_calls: null}"
    )
    assert "Credits" not in message
    assert message == (
        "Stopped at the 60-minute limit. "
        "Shorten the date range, or use fewer pipeline steps, then run it again."
    )


def test_timeout_card_survives_a_payload_with_no_timeout_block():
    """The status route attaches `timeout` conditionally, so absence is a real
    case. The card must still say what happened rather than throwing inside the
    poll callback, which would silently stop every other run's polling too."""
    message = _timeout_message("undefined")
    assert message == (
        "Stopped at the time limit. "
        "Shorten the date range, or use fewer pipeline steps, then run it again."
    )


def test_timed_out_panel_is_its_own_state_not_an_error_shade():
    """`is-timed-out`, not `is-error`. The user did not do anything wrong, and
    the panel must not read as though they did -- the same argument the
    `is-cancelled` comment already makes for the third state."""
    body = fn_body("function showBacktestRunProgress(")
    assert "isTimedOut" in body
    assert "'is-timed-out'" in body
    assert "Backtest stopped at the time limit" in body
    blocks = css_blocks(".backtest-run-progress.is-timed-out")
    assert blocks, ".backtest-run-progress.is-timed-out has no styles.css rule"


def test_poll_dispatch_takes_the_timeout_branch_before_the_error_branch():
    """A payload with `timed_out` must not reach `status.error`.

    The server sends no `error` key for a timeout, so the error branch would
    paint the red "Backtest did not start" panel with an undefined message --
    for a run that started, ran for an hour, and was billed.

    Sliced to the `finishedFocused` block FIRST. `status.cancelled` also appears
    earlier in `ensureBacktestPolling`, in the toast that announces a cancel on
    an unfocused card, and a scan over the whole function would anchor on that
    one instead -- passing even if the dispatch's cancel branch were deleted
    outright.
    """
    body = fn_body("function ensureBacktestPolling(")
    dispatch = body[body.index("if (finishedFocused) {"):]
    cancelled_at = dispatch.index("status.cancelled")
    timeout_at = dispatch.index("status.timed_out")
    error_at = dispatch.index("status.error")
    assert cancelled_at < timeout_at < error_at, (
        "the finishedFocused dispatch must test cancelled, then timed_out, then "
        "error -- the server sends no `error` key for either of the first two"
    )
```

Add `FRONTEND` and `css_blocks` to this module's imports:

```python
from dashboard.backend.tests._frontend_source import (
    FRONTEND,
    css_blocks,
    fn_body,
    js_const,
)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_backtest_progress_card.py -k timeout -v`
Expected: FAIL — `ValueError: substring not found` from `fn_body("function formatBacktestTimeoutMessage(")`

- [ ] **Step 3: Add the message formatter**

In `dashboard/frontend/app.js`, immediately before `function showBacktestRunProgress(` (~`:7992`):

```js
/**
 * The three sentences a timed-out backtest owes its user.
 *
 * Composed HERE, not on the server, for the same reason the season badge's
 * number has exactly one owner: the amount has to be formatted by the same
 * helper the Credits page uses (`CreditFormat.formatCreditsMicro`, exact to six
 * decimal places), and a sentence built server-side would put that formatting
 * and this copy under two owners that drift. The server sends facts; this turns
 * them into words, where `test_app_copy_register.py` can see them.
 *
 * `timeout` may be undefined -- `/backtest/status` attaches the block only when
 * the worker recorded one -- so every field is read defensively. Throwing here
 * would throw inside the poll callback and stop polling for every other run on
 * the page.
 */
function formatBacktestTimeoutMessage(timeout) {
    const limitSeconds = Number(timeout && timeout.limit_seconds);
    const lines = [
        // Derived, never a literal: the old ceiling message hardcoded
        // "60 minutes" and would have gone on saying it after the budget moved.
        Number.isFinite(limitSeconds) && limitSeconds > 0
            ? `Stopped at the ${Math.round(limitSeconds / 60)}-minute limit.`
            : 'Stopped at the time limit.',
    ];
    const spentMicro = timeout ? timeout.spent_micro : null;
    if (spentMicro !== null && spentMicro !== undefined) {
        // Omitted entirely on BYOK rather than rendered as zero: BYOK never
        // touches the ATL ledger, so "0.000000 Credits" is a claim about a row
        // that does not exist.
        lines.push(
            `Model calls completed before the stop cost ${window.CreditFormat.formatCreditsMicro(spentMicro)} Credits.`,
        );
    }
    // Both levers, matching the 422 that refuses an over-long pipeline window.
    // Naming only the window is a dead end for a user whose real problem is a
    // wide pipeline.
    lines.push('Shorten the date range, or use fewer pipeline steps, then run it again.');
    return lines.join(' ');
}
```

- [ ] **Step 4: Add the fourth state to `showBacktestRunProgress`**

```js
function showBacktestRunProgress(
    show,
    { isError = false, isFinished = false, isCancelled = false, isTimedOut = false } = {},
) {
```

and inside it, beside the `is-cancelled` toggle:

```js
    panel.classList.toggle('is-cancelled', !!isCancelled);
    // A fourth state, and not a shade of the first either. A run the product
    // stopped because it ran out of the budget it set itself is not the user's
    // error -- amber, like a warning, not `is-error`'s red.
    panel.classList.toggle('is-timed-out', !!isTimedOut);
```

Extend `terminal` and the title chain:

```js
    const terminal = !!isError || !!isCancelled || !!isTimedOut || !!isFinished;
    if (title) {
        if (isError) title.textContent = 'Backtest did not start';
        else if (isCancelled) title.textContent = 'Backtest cancelled';
        else if (isTimedOut) title.textContent = 'Backtest stopped at the time limit';
        else if (isFinished) title.textContent = 'Backtest complete';
        else title.textContent = 'Backtest in progress';
    }
```

`elapsed` stays visible (`elapsed.hidden = !!isError` is unchanged): on a timed-out run the elapsed clock is the duration, which is exactly what the user wants to see.

- [ ] **Step 5: Add the panel renderer**

Immediately after `formatBacktestTimeoutMessage`:

```js
/**
 * Paint the Backtest panel for a run the server stopped at its time limit.
 *
 * Factored out of the poll callback rather than inlined beside the cancel and
 * error branches so `_frontend_source.fn_body` can lift it: neither of those
 * two is reachable by a node harness today, and this state's copy -- the
 * derived minutes, the BYOK omission, the exact amount -- is precisely what
 * needs executing rather than grepping.
 *
 * Run config repainted FIRST and with a null run, exactly as the cancel branch
 * does: the previous paint came from the running branch and still says
 * "Running", and a run stopped mid-flight has no coverage verdict for the Model
 * coverage cell to read.
 */
function renderBacktestTimeoutPanel(status, displayElapsed, launchRunId) {
    renderBacktestRunConfig(null, {
        launchConfig: getBacktestLaunchConfig(launchRunId),
        statusLabel: 'Stopped at limit',
    });
    showBacktestRunProgress(true, { isTimedOut: true });
    updateBacktestRunProgress({
        elapsedSeconds: displayElapsed,
        message: formatBacktestTimeoutMessage(status && status.timeout),
    });
}
```

- [ ] **Step 6: Dispatch to it, between cancelled and error**

In `ensureBacktestPolling`'s `finishedFocused` block, insert between the `if (status.cancelled) { … }` block (ending `:8576`) and `else if (status.error)`:

```js
                } else if (status.timed_out) {
                    // Between cancelled and error, and never through either.
                    // The server sends no `error` key for a timeout, so the
                    // error branch below would paint the red "Backtest did not
                    // start" panel with an undefined message -- for a run that
                    // started, ran for an hour, and was billed.
                    renderBacktestTimeoutPanel(
                        status,
                        displayElapsed,
                        liveId || finishedId,
                    );
                } else if (status.error) {
```

- [ ] **Step 7: Add the CSS**

In `dashboard/frontend/styles.css`, after the `.backtest-run-progress.is-cancelled .backtest-run-elapsed` rule (~`:6636`):

```css
/*
 * A FOURTH terminal state, and not a shade of `is-error` -- the argument the
 * `is-cancelled` comment above makes applies here too, for a different reason.
 * A cancel is neutral because the user chose it. A timeout is amber because
 * nobody chose it: the product ran out of the budget it set itself, and billed
 * the user for the model calls that settled first. Red would say they broke
 * something; slate would say nothing happened. Neither is true.
 */
.backtest-run-progress.is-timed-out {
    border-color: rgba(251, 191, 36, 0.45);
    background: rgba(251, 191, 36, 0.08);
}

.backtest-run-progress.is-timed-out .backtest-run-elapsed {
    color: #fbbf24;
}
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtest_progress_card.py -v`
Expected: PASS with **no skips**.

- [ ] **Step 9: Mutation-check the branch order**

Temporarily swap the `else if (status.timed_out)` and `else if (status.error)` blocks in `app.js` and re-run `pytest dashboard/backend/tests/test_backtest_progress_card.py -k takes_the_timeout_branch -v`. Expected: FAIL. If it passes, the assertion is vacuous — fix it before restoring the order. Restore the correct order and re-run before committing.

- [ ] **Step 10: Bump the app.js cache-buster**

`dashboard/frontend/app.html:2701` — `app.js?v=131` becomes `app.js?v=132`. Tasks 5 and 6 both changed `app.js`; one bump covers both.

- [ ] **Step 11: Commit**

```bash
git add dashboard/frontend/app.js dashboard/frontend/app.html \
        dashboard/frontend/styles.css \
        dashboard/backend/tests/test_backtest_progress_card.py
git commit -m "feat: explain a timed-out backtest on the card"
```

---

## Task 7: Teach the Discord watcher the two states it falls through

**Files:**
- Modify: `dashboard/backend/integrations/discord_bot.py:542-568` (`watch_and_deliver_backtest`'s poll loop)
- Test: `dashboard/backend/tests/integrations/test_discord_watcher.py`

**Interfaces:**
- Consumes: the `timed_out` / `timeout` status payload from Task 3.
- Produces: nothing other tasks depend on.

**The bug.** The loop recognises exactly three shapes — `running` (continue), `error` (break, report failure), `success or runs_count` (break, report success). A `timed_out` payload matches none, so the body falls through with neither `continue` nor `break`, the loop runs out its `_MAX_POLLS = 360` × `_POLL_INTERVAL_SEC = 5` budget, and the `for…else` delivers *"Backtest is still running after 30 minutes."* — thirty minutes of silence followed by a wrong answer, for an outcome the server already knew.

**`cancelled` has this bug today.** Same shape, same two lines. This change is what makes it visible, and shipping the fix for one while leaving the other is shipping a bug we just read. **Independently droppable** — if a reviewer wants the `cancelled` half out, remove that one branch and its one test; nothing else depends on it.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/integrations/test_discord_watcher.py`, using the module's existing `job_store` fixture, `_PostRecorder` and `_install_common`:

```python
def test_watcher_reports_a_timed_out_run_instead_of_waiting_out_the_budget(
    job_store, monkeypatch
):
    """The loop knew only running/error/success, so a `timed_out` payload fell
    through with neither continue nor break -- 360 polls later the for/else
    delivered "still running after 30 minutes" for an outcome the server had
    already reported on the second poll."""
    live_run_id = "agent_20260914_timeout01"
    job = job_store.create_job(
        discord_user_id="42",
        channel_id=99,
        session_id="sess-timeout",
        label="t1me0ut",
        live_run_id=live_run_id,
    )
    poster = _PostRecorder()
    _install_common(monkeypatch, poster)

    polls = {"n": 0}

    async def fake_api_get(path: str, *, headers=None, timeout: int = 30):
        assert path == "/backtest/status"
        polls["n"] += 1
        if polls["n"] == 1:
            return {"running": True}
        return {
            "running": False,
            "timed_out": True,
            "elapsed_seconds": 3600,
            "live_run_id": live_run_id,
            "message": "Backtest stopped at the time limit.",
            "timeout": {
                "limit_seconds": 3600,
                "billing_mode": "platform_credits",
                "spent_micro": 42_318,
                "model_calls": 2,
            },
        }

    monkeypatch.setattr(bot, "api_get", fake_api_get)

    asyncio.run(bot.watch_and_deliver_backtest(job.job_id))

    # Broke out on the second poll, not after 360 of them.
    assert polls["n"] == 2
    assert len(poster.calls) == 1
    content = poster.calls[0]["content"]
    assert "60-minute limit" in content
    assert "still running after 30 minutes" not in content


def test_watcher_reports_a_cancelled_run_instead_of_waiting_out_the_budget(
    job_store, monkeypatch
):
    """The same gap, which this surface has had since the cancel route shipped:
    a user who cancels a Discord-launched backtest gets the identical
    thirty-minute non-answer."""
    live_run_id = "agent_20260914_cancel01"
    job = job_store.create_job(
        discord_user_id="42",
        channel_id=99,
        session_id="sess-cancel",
        label="cance11ed",
        live_run_id=live_run_id,
    )
    poster = _PostRecorder()
    _install_common(monkeypatch, poster)

    polls = {"n": 0}

    async def fake_api_get(path: str, *, headers=None, timeout: int = 30):
        assert path == "/backtest/status"
        polls["n"] += 1
        return {
            "running": False,
            "cancelled": True,
            "elapsed_seconds": 120,
            "live_run_id": live_run_id,
            "message": "Backtest cancelled.",
        }

    monkeypatch.setattr(bot, "api_get", fake_api_get)

    asyncio.run(bot.watch_and_deliver_backtest(job.job_id))

    assert polls["n"] == 1
    assert len(poster.calls) == 1
    content = poster.calls[0]["content"]
    assert "cancelled" in content.lower()
    assert "still running after 30 minutes" not in content
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/integrations/test_discord_watcher.py -k "timed_out or cancelled" -v`
Expected: FAIL — `assert polls["n"] == 2` sees 360 (the loop ran its whole budget), and the posted content is the "still running after 30 minutes" text.

These cases `importorskip('discord')` via the module header. If they skip, install the optional dependency (`pip install -r requirements-discord.txt`) — a skip here has verified nothing.

- [ ] **Step 3: Add the two branches**

In `watch_and_deliver_backtest`'s poll loop, **after** the `if status.get("running"): … continue` block and **before** `if status.get("error"):`:

```python
            if status.get("timed_out"):
                # The server stopped this run at its own wall-clock budget and
                # said so. Without this branch the payload matched none of the
                # three shapes below, fell through with neither continue nor
                # break, and the for/else delivered "still running after 30
                # minutes" -- half an hour after the answer was available.
                detail = status.get("timeout") or {}
                limit_seconds = detail.get("limit_seconds")
                minutes = (
                    round(int(limit_seconds) / 60)
                    if isinstance(limit_seconds, int) and limit_seconds > 0
                    else None
                )
                parts = [
                    f"Stopped at the {minutes}-minute limit."
                    if minutes
                    else "Stopped at the time limit."
                ]
                spent_micro = detail.get("spent_micro")
                if isinstance(spent_micro, int):
                    parts.append(
                        f"Model calls completed before the stop cost "
                        f"{spent_micro / 1_000_000:.6f} Credits."
                    )
                parts.append(
                    "Shorten the date range, or use fewer pipeline steps, "
                    "then run it again."
                )
                terminal_error = " ".join(parts)
                break

            if status.get("cancelled"):
                # The same gap, which this surface has had since the cancel
                # route shipped. Fixed here because this change is what made it
                # visible, and leaving it would be shipping a bug we just read.
                terminal_error = "Backtest cancelled."
                break
```

Both use `terminal_error` and therefore the existing `**Backtest failed** · …` delivery path. That is a knowingly loose fit — neither is a failure — but widening the delivery block into a three-way outcome is a larger change than this task, and the alternative today is thirty minutes of silence followed by a wrong answer. Note it in the task report as a follow-up worth filing, and do not widen it here.

The Credits amount is formatted with plain Python division rather than a shared helper: Discord has no `CreditFormat`, and `:.6f` reproduces its six-decimal contract exactly for every value this path can produce.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/integrations/ -v`
Expected: PASS with no skips.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/integrations/discord_bot.py \
        dashboard/backend/tests/integrations/test_discord_watcher.py
git commit -m "fix: relay stopped and cancelled backtests to Discord"
```

---

## Task 8: Correct the billing documentation

**Files:**
- Modify: `CLAUDE.md` (the "LLM backtest billing" bullet)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

**The error.** The bullet says `_execute_platform` "reserves a ceiling against `credit_ledger_entries`". It reserves against `credit_llm_reservations` and settles into `credit_llm_usage_entries`. `credit_ledger_entries` cannot hold either row — its `entry_type` CHECK (`domain/credits/repository.py:75`) admits only `purchase` / `refund` / `admin_grant_assign` / `admin_grant_reclaim`. This PR's work goes through exactly that code, so the correction belongs here.

- [ ] **Step 1: Fix the sentence**

In `CLAUDE.md`, in the bullet beginning "**LLM backtest billing — the live path**", replace:

> `_execute_platform` reserves a ceiling against `credit_ledger_entries`, `_settle` debits the provider-reported tokens at a pricing snapshot, and `_release_after_failure` hands the hold back.

with:

> `_execute_platform` reserves a ceiling as a `credit_llm_reservations` row, `_settle` debits the provider-reported tokens into `credit_llm_usage_entries` at a pricing snapshot, and `_release_after_failure` hands the hold back. Not `credit_ledger_entries` — that table's `entry_type` CHECK admits only `purchase` / `refund` / `admin_grant_assign` / `admin_grant_reclaim`, so it can hold neither row. (This bullet named the wrong table until 2026-09-14.)

- [ ] **Step 2: Add the timeout outcome to the same bullet**

Append one sentence to that bullet, after the existing sentence about the parent repeating `finalize_run` in its `finally`:

> A run killed by that timeout now finalizes as its own terminal outcome rather than a generic exception: `/backtest/status` answers `timed_out` with a `timeout` block naming the budget, the lane, and the settled spend (`credits_service.sum_run_llm_spend`), and the analytics event stays `backtest_failed` with `error_category="run_timeout"`. Design: `docs/superpowers/specs/2026-09-14-backtest-timeout-outcome-design.md`.

- [ ] **Step 3: Verify no test pins the old sentence**

Run: `grep -rn "reserves a ceiling against" dashboard/ docs/ 2>/dev/null`
Expected: no hits outside `CLAUDE.md` itself (which you have just changed).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: name the right table in the LLM billing bullet"
```

---

## Final verification

- [ ] **Step 1: Run the full backend suite**

Run: `pytest dashboard/backend/tests/ -q`
Expected: PASS. Note the skip count and check it against a `git stash` baseline — a *new* skip means a node or discord case silently stopped running.

- [ ] **Step 2: Confirm the seed database was not committed**

Run: `git status --short dashboard/storage/data/` and `git log --stat -8 -- dashboard/storage/data/`
Expected: empty. If `backtest.db` appears in either, `git restore` it and amend the commit that carried it — importing a store module runs `CREATE TABLE IF NOT EXISTS` against `DATABASE_PATH`, so running the app locally mutates that file.

- [ ] **Step 3: Confirm no credentials in the diff**

Run: `git diff main...HEAD | grep -nEi "sk_live|sk_test|postgres://|IFIND_(REFRESH|ACCESS)_TOKEN=|ANTHROPIC_API_KEY="`
Expected: no hits.

---

## Deferred — ask before acting

These came out of the spec's **Found, not fixed**. Filing an issue on a shared repo implicitly assigns work to someone else, so **ask the user first** and do not file unprompted.

1. **The Credits activity feed under-reports a run that needed overage recovery.** Its per-run rows filter `operation_key NOT LIKE '%:recovery:%'` (`domain/credits/repository.py:2259`, `:2314`), so a run whose shortfall was recovered later shows less than the user actually paid. This plan's aggregate deliberately does not copy that filter, which means the timeout card and the activity feed can disagree for such a run — the card being the correct one.
2. **The Discord watcher delivers a timeout and a cancel through the "Backtest failed" block** (Task 7, Step 3). Neither is a failure. Widening that delivery path into a three-way outcome is its own change.
3. **The child's captured output is discarded on a timeout.** The re-raised `TimeoutExpired` carries `output`/`stderr`, and keeping the tail of the child's stderr in the *server log* would help diagnose why a run overran. Not needed for the user-facing outcome.

## What this plan does not close

- **`MAX_SUBPROCESS_TIMEOUT_SECONDS = 14400`.** A hosted AI Hedge Fund run can be granted a four-hour budget, far above the 4200-second poll ceiling, so those cards still self-clear early — now with honest "lost contact" copy rather than a wrong timeout claim. Hosted is outside #474 item 2's declared scope.
- **The outcome is not persisted.** Like a cancel, a timed-out run writes no `agent_runs` row; the record lives only in `_recent_slots`, capped at 50 entries, and dies with the process. After a redeploy the explanation is gone.
- **The legacy mirror cannot carry it.** `_mirror_slot_to_legacy` writes neither `cancelled` nor `elapsed_seconds` onto the module-level `backtest_status` dict, so a caller that falls through to that fallback will not see `timed_out` — exactly as it does not see `cancelled` today.
