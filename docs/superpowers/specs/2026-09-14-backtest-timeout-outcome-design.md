# A Timed-Out Backtest Explains Itself — Design

**Issue:** [#474](https://github.com/Open-Finance-Lab/AgenticTrading/issues/474) item 2, with item 5 folded in.
**Scope:** `runtime_type="pipeline"` + `decision_source="llm"`. Hosted `ai_hedge_fund` is out of scope, as #474 declares.
**Status:** approved in brainstorming 2026-09-14; three decisions below changed after the code was read, and are marked **⚠ changed**.

## The defect

A pipeline LLM backtest that exhausts the fixed 3600-second parent budget is killed
(SIGTERM → grace → SIGKILL) and `_run_backtest_subprocess` re-raises
`subprocess.TimeoutExpired` (`api/routers/backtests.py:2341-2360`). Nothing catches it
specifically. It lands in the generic arm at `:1636`:

```python
except Exception as e:
    summary = _sanitize_backtest_error(e, 500, extra_secret=financial_datasets_api_key)
```

`_sanitize_backtest_error` (`:2399`) is `_redact_credentials(error, extra_secret)[-max_chars:]`
— it truncates from the **tail**, which is right for a stack trace and wrong for
`TimeoutExpired`, whose informative clause trails a long argv. The user waits up to an hour
and receives a fragment of a command line.

Three things make it worse than an ordinary bad error message:

1. **It is billed.** LLM spend settles per call as the run proceeds. The parent's `finally`
   (`:1646-1702`) repeats `finalize_run`, which *releases open reservations* — it does not
   unwind anything already settled. The user is charged for a run that returned nothing and
   explained nothing.
2. **The card has already gone.** `BACKTEST_POLL_MAX_SECONDS` (`app.js:20`) is `3600`,
   byte-identical to `PIPELINE_SUBPROCESS_TIMEOUT_SECONDS` (`backtests.py:1710`). At the
   ceiling the poller stops, clears the live run and paints its own guess —
   *"Timed out after 60 minutes. The backtest may still be running in the background."*
   (`app.js:8640`) — at the same instant the server begins finalizing. Whatever the server
   concludes is written after the client stopped looking. This is #474 item 5.
3. **Telemetry cannot see it either.** `_finalize_slot_locked` (`:958-964`) emits
   `backtest_failed` with `error_category` hardcoded to `"internal_error"` for any non-`None`
   error, so a budget exhaustion is indistinguishable from a crash in the product's own
   analytics.

## Goals

1. A timed-out pipeline run reports a **distinct terminal outcome**, not a failure.
2. The card says what happened, **what it cost**, and **which two levers to move** — the same
   two levers item 1's 422 names, so a user who meets both refusals hears one story.
3. The client stops guessing: the server's verdict arrives **before** the poller gives up.
4. A timeout is distinguishable from a crash in analytics.

## Non-goals

- Making the backtest faster, or changing LLM call count, cadence or retry policy.
- Changing `PIPELINE_SUBPROCESS_TIMEOUT_SECONDS` (3600), `MAX_SUBPROCESS_TIMEOUT_SECONDS`
  (14400) or `SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS` (600). The server budget does not move.
- Refunding the spend. Disclosing it is this change; a refund policy is not.
- Touching the hosted `ai_hedge_fund` path or `_enforce_ai_hedge_fund_window`.
- Persisting the outcome. See **What this does not close**.

---

## 1. Detection

A new arm **before** the generic one in `run_backtest_background` (`backtests.py:1381`; note
the real name — there is no `_run_backtest_in_background`):

```python
except subprocess.TimeoutExpired:
    ...
except Exception as e:      # unchanged, still last
```

`_run_backtest_subprocess` keeps raising exactly as it does. The SIGTERM → grace → SIGKILL →
reader-join contract its comments protect (`:2341-2360`) is untouched, and no function is
converted to return a flag. Nothing else in the module catches `TimeoutExpired` from the main
wait: the only other catch (`:1106`) is the short grace-period wait inside
`_kill_backtest_process_after_grace`, a different concern.

**Mandatory ordering detail.** The existing generic arm calls `_finalize_slot(...)` at `:1644`
and then sets `resolved_live_run_id = None` at `:1645`, *specifically* so the `finally`'s
`if resolved_live_run_id: _finalize_slot(..., error=None, runs_count=0)` (`:1661-1662`) does
not run a second time and overwrite the failure with a fake zero-run success. The new arm must
do the same. Omitting that line silently erases the outcome this whole design exists to
report.

## 2. A fourth terminal outcome

`_finalize_slot` (`:854-877`) and `_finalize_slot_locked` (`:879-965`) gain
`timed_out: bool = False`, exactly parallel to `cancelled`. `_finalize_slot`'s docstring
already argues the case — *"Three outcomes, not two … a dashboard that counts cancels as
failures measures the product as broken every time a user changes their mind"* — and this
makes it four. The new docstring paragraph must say why a timeout is **not** a cancel and
**not** a crash: the user did nothing wrong, the product ran out of the budget it set itself,
and it took their money doing so.

Five touch points, all of which already exist for `cancelled`:

| Site | Change |
|---|---|
| `_try_acquire_backtest_slot` slot literal (`:758-781`) | add `"timed_out": False` beside `"cancelled": False` |
| `_finalize_slot_locked` (`:935`) | `slot["timed_out"] = bool(timed_out)` beside the `cancelled` write, plus `slot["timeout_detail"] = timeout_detail` |
| `_slot_snapshot` (`:672-698`, `cancelled` at `:687`) | carry `timed_out` and `timeout_detail` through, same line |
| `get_backtest_status` (`:3276`) | a new branch **after** `cancelled` (`:3339`) and **before** `error` (`:3356`) |
| `_finalize_slot_locked` analytics return (`:951-964`) | see §6 |

Both functions take one further keyword, `timeout_detail: Optional[Dict[str, Any]] = None`.
Its `limit_seconds` is the local `subprocess_timeout` assigned at `:1553-1555` — bound before
the subprocess runs, and a `TimeoutExpired` can only be raised by that subprocess's wait, so it
is always bound when the new arm reads it. Read the budget from that local, never by
re-deriving it from `_backtest_subprocess_timeout` or by hardcoding 3600: the card must report
the budget this run was actually given.

The `except` arm is the only place that can build it — it is the only scope holding the lane,
the budget and the run id at once — so it computes the dict and hands it to `_finalize_slot`,
which stores it on the slot. The status route reads it back and never computes anything.

`error` stays `None` on this path, as it does for every `cancelled=True` call site. Routing a
timeout through `error` would be caught by the existing `elif slot.get("error")` branch and
nothing on screen would change.

### The status payload

The `cancelled` branch (`:3344-3355`) is a deliberately minimal five-key dict — no `error`
key at all, no `success` key. The timeout branch mirrors that shape and adds one conditional
sub-object:

```python
elif slot.get("timed_out"):
    payload = {
        "running": False,
        "timed_out": True,
        "elapsed_seconds": int(slot.get("elapsed_seconds") or 0),
        "live_run_id": slot.get("live_run_id"),
        "message": "Backtest stopped at the time limit.",
    }
    if timeout_detail:
        payload["timeout"] = timeout_detail
    return payload
```

`timeout` carries facts, never sentences:

```
{
  "limit_seconds":  int,          # the budget this run was given
  "billing_mode":   "platform_credits" | "byok",
  "spent_micro":    int | None,   # None ⇔ not the platform lane
  "model_calls":    int | None,   # distinct settled calls, same condition
}
```

A conditionally-attached dict on the payload is a new shape for a *terminal* branch — the only
precedent, `progress` (`:3336-3337`), is running-only. That is a deliberate extension, not an
oversight: the alternative is four more flat keys on a route that has no response model to
document them.

**The server owns facts; the client owns words.** `formatCreditsMicro` lives in
`js/credit-format.js` and the copy register (`tests/test_app_copy_register.py`) asserts against
`app.js`/`app.html` source. Composing the sentence server-side would put the number's
formatting and the user's copy under two different owners, which is the failure CLAUDE.md
records for the season badge. The `message` string above is a fallback for any client that
does not know the new branch, matching `"Backtest cancelled."` at `:3354`.

## 3. The billed amount

LLM spend does **not** live in `credit_ledger_entries` — that table's `entry_type` CHECK
admits only `purchase` / `refund` / `admin_grant_assign` / `admin_grant_reclaim`
(`domain/credits/repository.py:75`). The rows are `credit_llm_usage_entries`
(`repository.py:180`, Postgres twin `repository_postgres.py:315`): one per settled call,
keyed by `run_id`, with `amount_micro INTEGER NOT NULL CHECK (amount_micro < 0)`.

New method, hand-written on **both** stores plus a service passthrough:

```
CreditsStore.sum_run_llm_spend(user_id: int, run_id: str) -> tuple[int, int]
        # (micro_credits_spent_positive, distinct_call_count)
```

Four things this must get right, each of which is a real trap:

1. **Include `:recovery:` rows — do not filter them out.** This inverts what the brainstorm
   assumed, and the economics settle it. `_settle` computes
   `outstanding_micro = actual_micro - debit_micro` (`repository.py:1299`): the part of a
   call's real cost that could **not** be debited at settle time for want of funds. The
   recovery entry written at `:1504` debits exactly that remainder when funds later arrive. It
   is therefore the **first** charge of money never charged before, not a duplicate of money
   already counted. Settlement debits plus recovery debits for a reservation never exceed the
   call's true `actual_micro`.

   Filtering them would under-report what the user was actually charged — the worse lie for a
   sentence whose whole job is to disclose a charge. The existing exclusions at `:1100`,
   `:2259` and `:2314` are the reservation-scoped settlement view and the Credits activity
   feed; they are **not** a precedent to copy here (see **Found, not fixed**). Other aggregates
   — `domain/analytics/value_repository.py:787-800`'s `consumed_micro`, and `backfill.py`'s
   `_credit_candidates` — deliberately include recovery rows for this same reason.

   In practice a recovery row almost never exists at timeout time: recovery runs from Stripe
   webhook fulfilment (`repository.py:2071`) and grant assignment (`:3192`), not from the run.
   The rule matters anyway, because getting it wrong is invisible until the one case where it
   is not.
2. **Scope by `user_id` as well as `run_id`.** The status route authorizes the slot by owner
   already; scoping the query too costs nothing and means a guessed run id cannot leak spend.
3. **Add the missing index.** `credit_llm_usage_entries` has **no** index on `run_id` in either
   engine — only `idx_credit_llm_usage_user_id (user_id, id DESC)` (`repository.py:558`,
   `repository_postgres.py:329`). A new aggregate would table-scale a table that grows with
   every LLM call ever made. Add `idx_credit_llm_usage_user_run (user_id, run_id)` to both,
   via the existing `CREATE INDEX IF NOT EXISTS` self-migration. It also serves the evidence
   sub-query at `repository.py:2311-2316`, which already filters on exactly that pair.
4. **`CreditsStore` and `PostgresCreditsStore` share no base class.** There is no ABC or
   Protocol to add the method to — parity is convention plus a warning comment. This is why
   §7's twin tests are not optional.

Return sign: the stored amounts are negative; the method returns a **positive** spend, matching
`_llm_reservation_result_in_transaction`'s `-int(entry["amount_micro"])` convention
(`repository.py:1109-1114`).

### Why reading it in the `except` arm is complete

`finalize_run` runs later, in the `finally` (`:1652-1655`). It only calls
`release_run_llm_reservations` (`infrastructure/llm/execution/service.py:201-224`) — it
releases *open holds*, it never settles. Every settled row was written by the child as it went,
so the sum is already final at the moment the timeout is caught. The reservation for the call
that was interrupted is released, not charged; excluding it is the right answer, not a rounding
error.

### ⚠ changed — the lane is not in scope, so it must be threaded

The brainstorm assumed `billing_mode` was available at the `except` arm. It is not. It exists
only in `run_backtest_endpoint` (`:2797`), where it is folded into the opaque, signed
`execution_handoff_payload` before `run_backtest_background` is ever called. The background
function sees the payload string and `execution_run_id` (`:1438`), never the enum.

So `run_backtest_background` gains **one** new parameter, `billing_mode: Optional[str] = None`,
passed from the route. The lane then gates the query — BYOK skips it entirely rather than
reporting the zero it would return, keeping *"0 because BYOK"* and *"0 because nothing settled
yet"* distinguishable. `credits_service` is already imported at module scope (`:100`) and
already used in the `finally`, so the new call site adds no dependency.

**The account id comes from the slot, not from a second parameter.** `run_backtest_background`
has no `user_id` in its signature (`:1381-1401`) — `session_id` is the browser session, not the
account. Rather than thread a second argument that could drift from the slot's own idea of the
owner, the `except` arm reads it back through a small lock-held accessor over `_active_slots`,
the same source `_finalize_slot_locked` already uses for the analytics event (`:948`). The slot
is the single owner of *whose run this is*, and keeping one owner is what stops the spend query
being scoped to a different account than the one the slot authorized.

⚠ **The lookup must not move inside the ledger lock.** `_finalize_slot_locked`'s docstring
(`:897-899`) says the analytics emit stays outside the lock deliberately, "it can reach a store,
and this lock is taken by every status poll and every launch". A credits query is a store read
with the same hazard, and a larger one. Read the id under the lock; run the query outside it;
then finalize.

**Adjacent one-line fix, in scope because the lane is now there.** The `finally` calls
`.finalize_run(execution_run_id)` with no lane (`:1655`). `finalize_run`'s own docstring says
*"Worker callers should pass their known lane; retaining the in-process set keeps the
one-argument form safe too"* — but that in-process set (`_platform_runs`) belongs to the
*child's* service instance, and the parent constructs a fresh one, so the set is always empty
there. A BYOK run therefore reaches `release_run_llm_reservations` on the parent path today. It
finds nothing (BYOK never reserves), so nothing is wrong today — but the documented contract is
not actually holding, and passing the lane we now have makes it hold. Flagged separately so it
can be dropped without affecting anything else.

## 4. Copy

Rendered by a new branch in the poll loop's `finishedFocused` block, placed **between**
`if (status.cancelled)` (`app.js:8554`) and `else if (status.error)` (`:8577`).

```
Stopped at the 60-minute limit.
Model calls completed before the stop cost 0.042318 Credits.
Shorten the date range, or use fewer pipeline steps, then run it again.
```

- The minute figure is `Math.round(timeout.limit_seconds / 60)`, derived rather than a
  literal, so the card cannot drift from the budget the way `app.js:8640` did. For the pipeline
  path `limit_seconds` is always 3600, so it always reads "60-minute"; the derivation exists so
  that stays true without anyone remembering to update copy.
- The amount renders through `window.CreditFormat.formatCreditsMicro` — already loaded on
  `app.html:2704`, exact to six decimal places by deliberate design ("Exact ATL Credit
  formatting"). Exactness beats prettiness here precisely so the card and the Credits activity
  page report in the same unit at the same precision — with one documented divergence, the
  recovered-overage case in **Found, not fixed**.
- The second line is **omitted entirely** on BYOK (`spent_micro === null`), not rendered as
  zero.
- The third line names both levers, matching item 1's 422. A message naming only the window is
  a dead end for a user whose real problem is a wide pipeline.

`formatBacktestError` (`app.js:7452`) is **not** touched. It early-returns the raw string for
every non-A-share source, and this state no longer travels as an error anyway.

`showBacktestRunProgress` (`app.js:7992-8024`) takes a fourth flag, `isTimedOut`, alongside
`isError` / `isCancelled` / `isFinished`. It gets its own class — `is-timed-out`, styled as a
warning rather than `is-error`'s red — for the reason the existing comment gives for
`is-cancelled` being "a third state, not a shade of the second": the user did not do anything
wrong. It joins the `terminal` disjunction so the progress track, the hint and the Cancel
button all hide. Title: `'Backtest stopped at the time limit'`.

## 5. Poll margin (#474 item 5's checkbox)

One constant is doing two jobs, and item 5 is the moment to separate them. `app.js` gains a
second:

```js
const BACKTEST_BUDGET_SECONDS = 3600;   // mirrors PIPELINE_SUBPROCESS_TIMEOUT_SECONDS
const BACKTEST_POLL_MAX_SECONDS = 4200; // budget + SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS
```

4200 is the pipeline budget plus the 600-second `SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS` the repo
already uses as its margin unit.

`BACKTEST_POLL_MAX_SECONDS` has **five** consumers, not three, and they do not all want the
same number:

| Site | What it is | Which constant |
|---|---|---|
| `ensureBacktestPolling` (`:8336`, `setInterval(…, 1000)` at `:8665`) | poll ceiling | `POLL_MAX` |
| `pollBacktestStatus` (`:10288`, 1000 ms sleeps) | poll ceiling | `POLL_MAX` |
| `listRunningBacktests` (`:5924`) | orphan-entry staleness GC | `POLL_MAX` |
| `getAgentBacktestRunning` (`:6001`) | orphan-entry staleness GC | `POLL_MAX` |
| `updateBacktestRunProgress` `maxSeconds` default (`:8039`, used `:8066-8069`) | **progress-bar denominator** | **`BUDGET`** |

The last row is why a bare constant bump would have been wrong. That value is the denominator
of the progress bar's fill percentage — it answers *"how far through its budget is this run?"*
— so it must track the **server budget**, not the client's watching window. Today the two are
equal and the distinction is invisible; raise the one constant and the bar silently starts
under-reporting progress toward the limit it is drawing. The two GC sites correctly take the
poll ceiling: an entry must not be reaped before the server has had its chance to answer.

This lifts a constraint item 1's plan set deliberately (*"Do not touch `app.js`'s
`BACKTEST_POLL_MAX_SECONDS` … Raising it is #474 item 5's territory, deliberately deferred"*).
This is that territory.

### ⚠ changed — the client's existing timeout copy must change with it

`app.js:8640` currently paints *"Timed out after 60 minutes. The backtest may still be running
in the background."* at the ceiling, pinned by `test_app_copy_register.py:204`. Once the
ceiling clears the server budget, reaching it no longer means the run timed out — the server's
`timed_out` verdict arrives ten minutes earlier. Reaching 4200 s now means **no terminal answer
ever arrived**: a crash, a redeploy, a dropped connection. The copy becomes

```
Lost contact with this backtest. It may still be running — check the Backtest tab later.
```

No number in it, which also removes the drift risk permanently. `test_app_copy_register.py`'s
`test_backtest_hint_uses_strategy_and_limit` gains the new string as a positive assertion and
the old one as a negative, following the pattern already there for the retired 10-minute copy.

`app.html:1288` — *"Multi-step strategies can take several minutes (limit: 60 minutes)."* —
describes the **server** budget, which does not move. It stays exactly as it is.

## 6. ⚠ changed — analytics carries the reason, not a new event name

The brainstorm said this would emit `backtest_timed_out` instead of `backtest_failed`. Reading
the analytics layer changed the answer.

A new event name would have to be registered in **nine** places, not the six the brainstorm
guessed: `models.py:36-38` and its `EVENT_GROUP_BY_NAME` at `:87-89`, `instrumentation.py:38-40`
and its outcome map at `:222-224`, `lifecycle.py:42-44`, `query_service.py:41`, plus three the
brainstorm missed entirely — `metrics.py`'s `_TERMINAL_FAILURE` (`:16`, which drives the
published backtest success-rate KPI), `states.py`'s terminal-run-event set and its
`newest_three` failed-run check (which drives the three-consecutive-failures operational
alert), and `backfill.py`'s `_TERMINAL_RUN_EVENTS` mapping. Miss any one of the last three and
a timed-out run silently stops counting toward the metrics that exist to notice runs failing.

More to the point, it would be *less* truthful. A cancel earned its own event name because a
cancel is **not a failure** — the user changed their mind, and counting it as one measures the
product as broken. A timeout **is** a failure: the user asked for a backtest, got nothing, and
was charged. Keeping it inside `backtest_failed` keeps the success-rate metric honest. What is
missing is not a new outcome but the *reason*, and `error_category` is the field that exists to
carry it.

So: `_finalize_slot_locked` returns `event_name="backtest_failed"` with
`error_category="run_timeout"` on the timed-out path. `outcome` stays `"failed"`, so
`ALLOWED_OUTCOMES` and its two CHECK constraints are untouched. The admin analytics panel
surfaces `error_category` already, so timeouts become visible with no new label plumbing.

⚠ **Three edits, not two, and the third is a trap.** Adding the category needs
`ALLOWED_ERROR_CATEGORIES` (`models.py:59-68`), the CHECK text in `ANALYTICS_SQLITE_DDL`
(`analytics/repository.py:60-68`) — **and** the sentinel inside
`_migrate_error_category_constraint` (`repository.py:310-324`), which reads

```python
if "provider_quota_exhausted" in table_sql:
    return
```

and skips the table rebuild when it matches. That sentinel is keyed on the *last* category
added. Leave it and every existing SQLite database keeps its old CHECK, rejecting
`run_timeout` at write time while a fresh database accepts it — green tests, broken prod. Bump
it to `"run_timeout"`. The Postgres twin needs no sentinel: it does an unconditional
`DROP CONSTRAINT IF EXISTS` / `ADD CONSTRAINT` (`repository_postgres.py:208, 232`) and
self-heals.

It is deliberately **not** the existing `provider_timeout`: that means the upstream model
provider timed out, which is a different fact with a different remedy.

The user-facing outcome stays fully distinct — `timed_out` on the status payload, its own card
state, its own copy. Only the analytics *event name* is shared.

## 7. The Discord watcher must learn the new state

`/backtest/status` has a second consumer the brainstorm did not account for:
`integrations/discord_bot.py::watch_and_deliver_backtest` (`:542-568`). Its loop recognises
exactly three shapes — `running` (continue), `error` (break, report failure), and
`success or runs_count` (break, report success). A `timed_out` payload matches none of them, so
the body falls through with neither `continue` nor `break`, the loop runs out its
`_MAX_POLLS = 360` × `_POLL_INTERVAL_SEC = 5` budget, and the `for…else` delivers

> Backtest is still running after 30 minutes. Check the dashboard later, or ask an admin to
> inspect the API worker.

— thirty minutes of silence followed by a wrong answer, for an outcome the server already knew.

It gets a `timed_out` branch that breaks and relays the real explanation.

**The same branch fixes `cancelled`, which has this bug today.** The watcher has no `cancelled`
handling either, so a user who cancels a Discord-launched backtest gets the identical
thirty-minute non-answer. It is the same shape, the same two lines, and this change is what
makes the gap visible — leaving the second half of it in place while fixing the first would be
shipping a bug we just read. Called out separately so it can be dropped on request.

`discord.py` is an optional dependency (`requirements-discord.txt`); its tests
`importorskip('discord')`, so the new cases follow that convention.

## 8. Testing

**Router** (`tests/test_backtests_router.py`; `test_pipeline_timeout_finalizes_execution_and_slot_once`
at `:917` already drives this path and is the case to extend, via
`tests/_fake_child.py::FakeChild(timeout_waits=…)`):
- `TimeoutExpired` → the slot's terminal state is `(timed_out, cancelled, error, running) ==
  (True, False, None, False)`, asserted as separate fields the way
  `test_backtest_cancel.py:69-102` does.
- `GET /backtest/status` returns the `timed_out` branch, with no `error` key and no `success`
  key.
- `timeout.spent_micro` and `model_calls` are present on `platform_credits` and **`None` on
  `byok`** — the case that pins the lane gating rather than the zero.
- The `finally` does not re-finalize (the `resolved_live_run_id = None` guard), asserted by the
  slot still reporting `timed_out` after the thread completes.
- The analytics event is `backtest_failed` with `error_category="run_timeout"`.

**Credits repositories** — `sum_run_llm_spend` on SQLite (`tests/domain/credits/test_repository.py`)
and on the Postgres twin (`test_repository_postgres.py`, `@pg_only`, guarded by
`_postgres_testing.require_local_postgres_url`). Both must include a case with a `:recovery:`
entry present, asserting it is **excluded** — the twins share no base class, so this is the
only thing keeping them in step.

**Frontend under `node`** (`_frontend_source.fn_body`, per `test_backtest_progress_card.py`):
- the timeout branch paints its three lines, with the amount formatted by `CreditFormat`;
- the BYOK case omits the cost line entirely;
- `status.error` is **not** taken for a `timed_out` payload — mutation-checked by reordering
  the branches and confirming the test fails.

**Discord watcher** (`tests/test_discord_watcher.py`, which today covers only running / error /
success): a `timed_out` payload breaks the loop and relays the timeout message rather than
running out the poll budget. A `cancelled` payload, likewise.

**Source-shape** (the technique at
`test_ifind_ashare_frontend.py::test_the_ifind_prefill_window_is_runnable_under_the_server_cap`),
two assertions rather than one, because there are now two constants:
- `BACKTEST_BUDGET_SECONDS` **equals** the imported `PIPELINE_SUBPROCESS_TIMEOUT_SECONDS` — the
  progress bar is drawing the server's budget, so it must be the server's number;
- `BACKTEST_POLL_MAX_SECONDS` is **strictly greater** than `BACKTEST_BUDGET_SECONDS` — the
  client must outlive the server's verdict.

Together these are what stop the two from silently reconverging, which is the whole of #474
item 5.

**Copy register** — `test_app_copy_register.py` updated for the new ceiling string, positive and
negative.

## What this does not close

- **`MAX_SUBPROCESS_TIMEOUT_SECONDS = 14400`.** A hosted AI Hedge Fund run can be granted a
  four-hour budget, and 4200 s is far below it, so those cards still self-clear early — now
  with honest "lost contact" copy rather than a wrong timeout claim. Hosted is out of #474 item
  2's declared scope and belongs to item 5 proper.
- **The outcome is not persisted.** A cancelled run writes no `agent_runs` row, and a timed-out
  run will not either: the record lives only in `_recent_slots`, capped at 50 entries
  (`backtests.py:941-943`), and dies with the process. After a redeploy the explanation is
  gone. Giving terminal outcomes durable storage is a larger change than this one.
- **The legacy mirror cannot carry it.** `_mirror_slot_to_legacy` (`:701-710`) writes neither
  `cancelled` nor `elapsed_seconds` onto the module-level `backtest_status` dict, so a caller
  that falls through to that fallback (no slot resolved) will not see `timed_out`, exactly as
  it does not see `cancelled` today.
- **The child's captured output is still discarded.** The re-raised `TimeoutExpired` carries
  `output`/`stderr` as attributes, but `_sanitize_backtest_error` only does `str(e)`. Keeping
  the tail of the child's stderr in the *server log* on a timeout would be useful; it is not
  needed for the user-facing outcome and is left alone here.
- **No refund.** Disclosing the spend is not deciding what to do about it.

## Found, not fixed

- **`CLAUDE.md`'s LLM-billing bullet is wrong about the table.** It says `_execute_platform`
  "reserves a ceiling against `credit_ledger_entries`". It reserves against
  `credit_llm_reservations` and settles into `credit_llm_usage_entries`;
  `credit_ledger_entries` cannot hold either row — its `entry_type` CHECK forbids them. This
  PR's doc commit corrects that line, since the work goes through exactly that code.
- **The Credits activity feed under-reports a run that needed overage recovery.** Its per-run
  rows filter `operation_key NOT LIKE '%:recovery:%'` (`domain/credits/repository.py:2259`,
  `:2314`), so a run whose shortfall was recovered later shows less than the user actually
  paid. This spec's aggregate deliberately does **not** copy that filter (§3), which means the
  timeout card and the activity feed can disagree for such a run — the card being the correct
  one. Worth an issue against the feed; out of scope here, and not a reason to propagate the
  omission.
- **Vocabulary collision worth knowing.** `timed_out` already exists as a *step* status on the
  Agent-Environment Protocol surface (`docs/api/agent-environment-protocol-v1.md`), where it
  means a decision deadline was missed. That is a different surface with a different field;
  the protocol and SDK docs are not made stale by this change. Naming the dashboard outcome the
  same thing is consistent vocabulary, not a shared field.
