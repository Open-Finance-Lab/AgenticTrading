# Live-Board Fixes P3: Attempt Ledger — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Price every billable Live Leaderboard model attempt into a durable monthly ledger, stop live model deploys at `LEADERBOARD_MONTHLY_BUDGET_USD` (default $50/month), and show each model's last nightly outcome and a budget-halt line on the public board.

**Architecture:** A new leaf module, `domain/leaderboard/live_budget.py`, keeps one `agent_runs` row per (month, entry) in its own session and mode, and owns the cap, the outcome classifier and a `LiveAttempt` that `live.deploy_live_model_increment` opens only once a deploy has billable work. The deploy checks the cap before any model build or bar fetch, upserts the ledger at every session close and prices whatever ended the attempt; `refresh_live_leaderboard` reports halts apart from failures, and the public GET gains only a boolean and a whitelisted `{on, outcome, kind}` per model, which `js/live-leaderboard.js` renders.

**Tech Stack:** Python 3.13 / FastAPI backend (`agent_runs` through the existing `db.insert_run` / `get_run` / `get_runs_by_session` on both the SQLite and Postgres twins), pytest from the repo root, vanilla-JS dashboard tested under `node` against a fake DOM.

**Spec:** `docs/superpowers/specs/2026-10-14-atl-direction.md` §3 item 5 (no silent failures), §6 (Live Leaderboard fixes) and §7 (Budget and keys). Decisions, Global Constraints, merge order and Review Focus live in the index plan `docs/superpowers/plans/2026-10-14-live-board-fixes.md`; every task here inherits its Global Constraints.

**Merge order:** P1 → P2 → P3 → P4 → P5 → P6 → P7 PR A → P7 PR B (owner-gated). Deadlines: P3 before 2026-11-01; P4 by 2026-10-31 (else bump both live_from values); P1 merged and deployed by 2026-11-20. One executor per contiguous range; phases are not safe to run in parallel (several append to the same test files and bump the same cache-busters).

## Phase 3: Attempt ledger: $50/month hard stop and visible nightly failures

**Why.** Spec §7 requires "a $50/month hard stop enforced in code before Season 1"; `LEADERBOARD_MONTHLY_BUDGET_USD` was decided on 08-15 and never built, and the live board's calls go straight to `COMMONSTACK_API_KEY`, outside the Credits layer. The checkpoint rows cannot be the spend record: a session H6 refuses to store, a run that dies before its next session close, and every row a forced or config replay deletes were billed and leave no row. Spec §6 also asks for failures to be visible; today a failed nightly model exists only in a log line and an ephemeral state file. This phase prices every billable attempt into a durable ledger, enforces the cap, and publishes a coarse per-model "last nightly run" status (Decision 5 in the index plan).

**Depends on.** No code from another phase. Merge order runs it after P1 (`2026-10-14-live-board-fixes-p1-early-close.md`) and P2 (`2026-10-14-live-board-fixes-p2-memory-robustness.md`, which carries the roster-loop change, Decision 4 in the index plan), and before P4 (`2026-10-14-live-board-fixes-p4-roster.md`, Decision 6 in the index plan), P6 (`2026-10-14-live-board-fixes-p6-copy.md`, Decision 8 in the index plan) and P7 (`2026-10-14-live-board-fixes-p7-records.md`, Decision 9 in the index plan); all are in `docs/superpowers/plans/`. P1 and P2's roster-loop half (#668) are on main, and P2's unmerged half edits only `portfolio_manager.py` and two trade-memory test files, none of this phase's. The anchors were written to avoid the lines P4 and P6 rewrite, the halt clause is inserted between the deploy call and the generic `except Exception` that #668 put there, and the roster-level tests derive the roster from `live.live_llm_entries(...)` instead of naming models. Where an anchor could have moved, the step says where the same code goes.

> **Deviation from Decision 5 (index plan).** Decision 5 in the index plan puts the in-run check at each session close "before the H6 branch": upsert the ledger, then stop if this attempt's average session cost would cross the cap. The upsert stays exactly there: `attempt.record_session()` is the first statement of `on_session_close`, ahead of the gap check and H6, because a session H6 refuses to store was billed. The stop moved: `attempt.stop_if_next_session_would_cross()` runs **after** the session's checkpoint decision, and only while another session of the segment remains. So the session just paid for is stored instead of being thrown away and bought again on the next replay, and a finished night is never relabelled as a halt. The price is an overshoot of at most about one session's cost per entry per month above the cap: the pre-check admits an attempt while the month is still under the cap and nothing projects that attempt's first session, and once the month is over the cap every later pre-check halts the entry before any bar fetch. That is acceptable for a hard stop whose job is to catch runaway spend: one session costs cents against a $50 cap (the whole live board ran about $0.35/month when measured on 2026-10-14, spec §7), and the alternative discards work that was already billed.

**Branch / PR title.** `feat/live-attempt-ledger` / `feat(live): monthly budget hard stop and attempt ledger`. Before Task 3.1, cut the branch from the remote main:

```bash
git status --short                                       # must print nothing: switch -c carries uncommitted edits along
gh pr list --head feat/live-attempt-ledger --state all   # expect no rows; if one exists, stop and read it first
git fetch origin main
git switch -c feat/live-attempt-ledger --no-track origin/main
```

This does not switch to or pull the local `main`, so it also works inside a pre-made worktree while `main` is checked out elsewhere; commit on this branch, not on the worktree's own branch. Parallel sessions share one checkout: run `git branch --show-current` before every commit below; it must print `feat/live-attempt-ledger`. Commit commands below omit any `Co-Authored-By` trailer; an agent executing this plan adds its own.

Every `find` block below was checked to occur exactly once on main@011d6411, which has P1 and P2's Tasks 2.1–2.2 (#668), and every quoted pass/fail count was measured by applying this plan's text, task by task, to a copy of main@011d6411. main@653180aa changes none of the files this phase edits. Line hints (`≈:N`) are approximate and were written against older trees; the `find` blocks are what locate an edit. If `origin/main` has moved on, the anchors still apply (they are verbatim blocks, not line numbers); treat a small drift in a neighbour-suite total as main having grown. The named failing tests and their messages are the real check.

**Since #668 (2026-10-30).** P2's Tasks 2.1–2.2 merged as #668, with review fixes the anchors above never saw. Two things changed in the roster loop. The per-entry failure bookkeeping moved into `domain/leaderboard/roster_failures.py` (`RosterFailures`), which stops the roster once two entries in a row fail with the same error. And the success path (`roster.traded(row)`, `successes.append(row)`) now runs after the `try` statement, not inside it. Task 3.3's Edits 7–9 are written against that loop: the halt clause sits ahead of the generic handler, the success path moves into the `try` statement's `else` (Edit 8), and the budget read after the loop is guarded (Edit 9). P2's unmerged half (Tasks 2.3–2.4) adds no test to the files this phase counts.

**Behaviour this phase pins (read before Task 3.3).**
- A deploy that is already current costs nothing, writes nothing and is never halted. `deploy_live_model_increment` has two ways to be current and both now return before the attempt is opened: the existing `latest.end_date == freeze_end` return, and the `live_increment_bounds(...) is None` return, which on main sits after `get_strategy` and which Task 3.3 moves up into the head (a dry run that left it in place built the model, could be halted by the pre-check and wrote an `ok` ledger row for an entry with nothing to trade). The move has one side effect, which is deliberate: a config change on an entry that is current this way is picked up by its next increment instead of replaying the month at once. That is already how the first cached return behaves, and the second is reachable only when `freeze_end` is not a session day, which `live_clock` never produces.
- Pre-check: before `lb_service.get_strategy` and before any bar fetch, `month spent >= cap` raises `LiveBudgetExceeded`. `0` is a real cap that blocks every billable deploy.
- At every session close, the ledger is upserted **first** (before the gap check and before H6). The in-run stop runs **after** the session's checkpoint decision, and only while another session of the segment remains: the session just paid for is stored, and the last session of a segment is never stopped by its own projection. So the overshoot is at most about one session per entry per month.
- In the roster loop, a halt is neither a failure nor a success. It goes to `budget_halted`, never to `model_failures` or `model_results`, and it neither counts toward nor resets `RosterFailures`' shared-fault stop, whether it came from the pre-check or between sessions. The budget read after the loop cannot stop the prune and the state save.
- Every attempt ends in `LiveAttempt.finish(exc)`, which writes `metadata.last_attempt = {on: <freeze_end>, outcome, kind}`. A per-session upsert keeps the previous `last_attempt`, so a process killed mid-run leaves the previous night's outcome, dated, rather than a guess.
- `kind` is matched on the exception's class names (the exact builtin names first, then the parts `Timeout` and `Connect`, which client-library classes such as `psycopg_pool.PoolTimeout` and `httpx.ConnectError` carry) and never carries message text. The public GET gains `{on, outcome, kind}` per model, through a whitelist, and a boolean `live_status.budget_halted`. It adds no new dollar figure; the per-entry `est_cost_usd` predates this plan (`_entry_from_strategy` already publishes each entry's checkpoint-row `est_cost_usd`, `input_tokens`, `output_tokens` and `llm_calls`; that stays out of scope). The cap and the month-to-date spend stay in the refresh result and the log.
- Ledger rows on public routes, verified at source on main@17c552c5: the board, `_find_cached_run`, `GET /api/backtest/runs` (the caller's session, mode `backtest` only) and `/api/v2` (run ids starting `run_`) never return them. `GET /runs?mode=leaderboard_spend` (public, no session) does list them, through the `RunMetadata` allow-list: run id, the entry's name and id (`agent_name`, `llm_model`), dates, `created_at`, `initial_equity` 0, the cumulative `llm_calls` and derived decision-provenance labels, with no `est_cost_usd`, no token counts and no `metadata` (so no `last_attempt`). `GET /api/backtest/{run_id}` returns one only to a caller who sends `X-Session-Id: leaderboard-live-spend`, as an empty curve with null return metrics; `/runs/{run_id}` rejects that header because it is not a UUID. None of them exposes dollars or tokens, so this phase adds no exclusion; the last test of Task 3.1 pins all three.

**Interfaces this phase produces.**
- New module `dashboard/backend/domain/leaderboard/live_budget.py`:
  - Constants: `LIVE_SPEND_SESSION_ID = "leaderboard-live-spend"`, `LIVE_SPEND_MODE = "leaderboard_spend"`, `LEDGER_NAME = "live_spend"`, `BUDGET_ENV = "LEADERBOARD_MONTHLY_BUDGET_USD"`, `DEFAULT_MONTHLY_BUDGET_USD = 50.0`, `MAX_MONTHLY_BUDGET_USD = 10_000.0`, `OUTCOME_OK = "ok"`, `OUTCOME_FAILED = "failed"`, `OUTCOME_BUDGET_HALTED = "budget_halted"`, `OUTCOMES`, `KIND_BUDGET = "budget"`, `KIND_MODEL_COVERAGE = "model_coverage"`, `KIND_MARKET_DATA_GAP = "market_data_gap"`, `KIND_TIMEOUT = "timeout"`, `KIND_NETWORK = "network"`, `KIND_INTERRUPTED = "interrupted"`, `KIND_ERROR = "error"`, `KINDS`.
  - `class LiveBudgetExceeded(RuntimeError)`.
  - `monthly_budget_usd() -> float`; `spend_run_id(month_start: str, entry_id: str) -> str` (→ `"live_spend_202608_deepseek_v4_pro"`); `month_ledger_rows(month_start: str) -> Dict[str, Dict]`; `budget_status(month_start: str, *, checkpoint_rows=None) -> {"month": "YYYY-MM", "cap_usd": float, "spent_usd": float, "remaining_usd": float, "halted": bool}`; `classify_attempt(exc: Optional[BaseException]) -> Tuple[str, Optional[str]]`; `clean_last_attempt(value) -> Optional[{"on", "outcome", "kind"}]`; `last_attempts(month_start: str) -> Dict[str, {"on", "outcome", "kind"}]`.
  - `class LiveAttempt(entry, *, month_start: str, freeze_end: str, checkpoint_rows: Mapping[str, Mapping])` with attributes `strategy_impl`, `cap_usd`, `sessions_closed` and methods `attempt_usd()`, `entry_spent_usd()`, `month_spent_usd()`, `check_can_start()`, `record_session()`, `stop_if_next_session_would_cross()`, `finish(exc: Optional[BaseException])`.
- `dashboard/backend/domain/leaderboard/live.py`:
  - New private `_trade_live_segment(entry, freeze_cfg, attempt, *, resume_from, prior_curve, latest, force_refresh, allow_fallback, bars_memo) -> Dict` (the billable half of `deploy_live_model_increment`; the public signature is unchanged). It never returns a cached row: both already-current returns live in `deploy_live_model_increment`'s head, ahead of the `LiveAttempt`.
  - `refresh_live_leaderboard(deploy_models=True)` result gains `budget_halted: List[str]` and `budget: Optional[<budget_status dict>]` (`None` when the read fails); `models_deployed = not roster.failures and not halted`.
  - Log lines: `ERROR: live.budget_exhausted month=YYYY-MM cap_usd=<.2f> spent_usd=<.2f>|unread skipped=<id,id>` (once per refresh, flushed), `ERROR: live.budget_status_failed error=<ExceptionType>` (the budget read after the roster loop failed; the refresh still prunes and saves) and `ERROR: live.spend_ledger_write_failed entry=<id> error=<ExceptionType>`.
  - `get_live_leaderboard()` payload: `live_status["budget_halted"]: bool`; every entry gains `last_attempt: Optional[{"on": "YYYY-MM-DD", "outcome": "ok"|"failed"|"budget_halted", "kind": Optional[str]}]` (always `None` for baselines and indices).
- `LLMAgentStrategy.input_tokens` / `.output_tokens` are current after every decision step (the H6 counters still move only at a session close).
- `dashboard/frontend/js/live-leaderboard.js`: `LIVE_ATTEMPT_KIND_COPY`, `liveLastAttemptCopy(entry) -> string`, `liveLastAttemptBadge(entry) -> 'Paused'|'Last run failed'|''`, `liveBudgetHaltLine(status) -> string`.
- Test helpers: `_ScriptedAgent(..., step_tokens=(10, 10))` and `_clear_live_and_spend_rows()` in `dashboard/backend/tests/test_live_leaderboard.py`; `_insert_ledger`, `_ledger`, `_attempt`, `_impl`, `_DiesBeforeTheClose`, `_no_billable_work` in `dashboard/backend/tests/test_live_budget.py`.

### Task 3.1: The `live_budget` module (cap, ledger row, outcome classes)

**Files:**
- Create: `dashboard/backend/domain/leaderboard/live_budget.py`
- Create: `dashboard/backend/tests/test_live_budget.py`
- Modify: `dashboard/backend/tests/conftest.py`

**Interfaces:**
- Consumes: `db.insert_run` / `db.get_run` / `db.get_runs_by_session` (both twins; `insert_run` is an upsert on SQLite and Postgres, so no new DB method and no twin-parity change), `token_cost.estimate_cost_usd`, `token_cost.PRICING_SOURCE_VERSION`.
- Produces: everything listed for `live_budget.py` above. Nothing in `live.py` calls it yet.

Consumer audit done for this task (every non-test reader of `agent_runs` on main@17c552c5): `GET /runs` (`api/routers/backtests.py` `get_runs`, public; it defaults to `mode == 'backtest'`, and `?mode=leaderboard_spend` lists ledger rows through `RunMetadata`, which has no `est_cost_usd`, token or `metadata` field), `GET /api/backtest/runs` and the other session-scoped routes (filter the caller's session and mode `backtest`/`baseline`; `GET /api/backtest/{run_id}` answers only a caller who sends the ledger's session id, with an empty curve), `api/v2/leaderboard.build_leaderboard` (run ids starting `run_` only), `db.get_runs_by_mode` callers (`backtest`, `paper_baseline`), `db.aggregate_operator_cost_for_day` (rows with an owner only; ledger rows have none), `domain/agents/service.py` and `domain/analytics/backfill.py` (agent/user sessions), `lb_service._cached_run_index` / `_find_cached_run` (mode `leaderboard`), and `tests/test_seed_database_integrity.py` (reads the committed seed DB read-only; the suite never writes it). None counts a ledger row; the last test below pins the ones that could. `dashboard/scripts/backfill_runs_to_postgres.py` copies every row, ledger included, which is the wanted behaviour.

Classifier audit done for this task (main@17c552c5): no provider timeout or connection error escapes a live deploy, because `LLMAgentStrategy` calls `PortfolioManager.make_trading_decision_with_llm` with `strict_llm` off, on the legacy provider client from `make_llm_client` (not the execution layer, so no typed error carrying a category exists on this path), and that method's `except Exception` turns any SDK error into a rule-based step that later surfaces as an H6 refusal (`model_coverage`), while an Alpaca request error comes back as an empty fetch and then `RuntimeError("No market data returned …")` (`error`); what can still escape under a client-library name is a store timeout such as `psycopg_pool.PoolTimeout` from a Neon checkout in a checkpoint or ledger write, none of whose MRO names is `TimeoutError` or `ConnectionError` (nor are those of `httpx.ReadTimeout`, `httpx.ConnectError`, `anthropic.APITimeoutError` or `anthropic.APIConnectionError`), so `classify_attempt` matches the name parts `Timeout` and then `Connect` after the exact names, and the board's copy for `timeout` reads "a request timed out" rather than blaming the model provider.

- [ ] **Step 1: Write the failing test.** Create `dashboard/backend/tests/test_live_budget.py` with exactly:

```python
"""The Live Leaderboard's monthly spend ledger and $50/month hard stop.

Every billable live attempt is priced into one ``agent_runs`` row per
(month, entry) in the ``leaderboard-live-spend`` session, whatever ended it:
success, a crash, an H6 refusal, a forced replay or the budget itself.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import psycopg_pool
import pytest
from fastapi.testclient import TestClient

from dashboard.backend.api.v2.leaderboard import build_leaderboard
from dashboard.backend.app import app
from dashboard.backend.database import db
from dashboard.backend.domain.leaderboard import live_budget
import dashboard.backend.domain.leaderboard.live as live
import dashboard.backend.domain.leaderboard.service as lb_service

_AUG = "2026-08-01"
_FREEZE = "2026-08-27"
_DEEPSEEK_MODEL = "deepseek/deepseek-v4-pro"  # $0.435 per 1M input tokens
_PROVIDER_REQUEST = httpx.Request("POST", "https://api.commonstack.ai/v1/messages")


def _clear_live_rows():
    for session in (live.LIVE_SESSION_ID, live_budget.LIVE_SPEND_SESSION_ID):
        for run in db.get_runs_by_session(session) or []:
            db.delete_run(run["run_id"])


@pytest.fixture(autouse=True)
def _isolated_ledger(monkeypatch, tmp_path):
    """The test DB is session-scoped: start and end every test with no board
    rows and no ledger rows, the cap at its default, and no warning memo."""
    monkeypatch.setattr(live, "_LIVE_REFRESH_STATE_PATH", tmp_path / "live_state.json")
    monkeypatch.delenv(live_budget.BUDGET_ENV, raising=False)
    monkeypatch.setattr(live_budget, "_warned_values", set())
    _clear_live_rows()
    yield
    _clear_live_rows()


def _impl(input_tokens=0, output_tokens=0, llm_calls=0, model_id=_DEEPSEEK_MODEL):
    return SimpleNamespace(
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        llm_calls=llm_calls,
    )


def _entry(entry_id="deepseek_v4_pro"):
    return {"id": entry_id, "name": f"{entry_id} name", "model_id": _DEEPSEEK_MODEL}


def _attempt(entry_id="deepseek_v4_pro", checkpoint_rows=None, freeze_end=_FREEZE, month=_AUG):
    return live_budget.LiveAttempt(
        _entry(entry_id),
        month_start=month,
        freeze_end=freeze_end,
        checkpoint_rows=checkpoint_rows or {},
    )


def _insert_ledger(entry_id, usd, *, month=_AUG, last_attempt=None):
    metadata = {"ledger": live_budget.LEDGER_NAME}
    if last_attempt is not None:
        metadata["last_attempt"] = last_attempt
    db.insert_run(
        run_id=live_budget.spend_run_id(month, entry_id),
        session_id=live_budget.LIVE_SPEND_SESSION_ID,
        agent_name=entry_id,
        mode=live_budget.LIVE_SPEND_MODE,
        start_date=month,
        end_date=month,
        initial_equity=0.0,
        llm_model=entry_id,
        est_cost_usd=usd,
        metadata=metadata,
    )


def _ledger(entry_id="deepseek_v4_pro", month=_AUG):
    return db.get_run(live_budget.spend_run_id(month, entry_id))


# --- the cap ---------------------------------------------------------------


def test_budget_defaults_to_fifty_dollars_when_unset(capsys):
    assert live_budget.monthly_budget_usd() == 50.0
    assert capsys.readouterr().out == ""


def test_a_blank_budget_is_the_default_without_a_warning(monkeypatch, capsys):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "   ")
    assert live_budget.monthly_budget_usd() == 50.0
    assert capsys.readouterr().out == ""


def test_the_budget_is_read_on_every_call_not_at_import(monkeypatch):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "12.5")
    assert live_budget.monthly_budget_usd() == 12.5
    monkeypatch.setenv(live_budget.BUDGET_ENV, " 7 ")
    assert live_budget.monthly_budget_usd() == 7.0


@pytest.mark.parametrize("raw", ["abc", "nan", "inf", "-inf", "-1", "10000.01", "1e9"])
def test_a_junk_budget_warns_once_and_falls_back(monkeypatch, capsys, raw):
    monkeypatch.setenv(live_budget.BUDGET_ENV, raw)
    assert live_budget.monthly_budget_usd() == 50.0
    assert live_budget.monthly_budget_usd() == 50.0
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert lines == [
        f"WARNING: LEADERBOARD_MONTHLY_BUDGET_USD={raw!r} is not a dollar amount "
        "between 0 and 10000; using the default 50.00"
    ]


@pytest.mark.parametrize("raw", ["0", "0.0", "-0"])
def test_zero_is_a_cap_that_blocks_everything_not_off(monkeypatch, raw):
    monkeypatch.setenv(live_budget.BUDGET_ENV, raw)
    assert live_budget.monthly_budget_usd() == 0.0
    with pytest.raises(live_budget.LiveBudgetExceeded, match=r"\$0.00 of its \$0.00"):
        _attempt().check_can_start()


def test_the_conftest_strips_the_budget_for_the_suite():
    conftest = (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in conftest.splitlines() if not line.lstrip().startswith("#")
    )
    assert 'os.environ.pop("LEADERBOARD_MONTHLY_BUDGET_USD", None)' in code


# --- outcome classes ---------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (None, ("ok", None)),
        (live_budget.LiveBudgetExceeded("spent"), ("budget_halted", "budget")),
        (lb_service.LeaderboardFallbackError("h6"), ("failed", "model_coverage")),
        (live.LiveSessionGapError("gap"), ("failed", "market_data_gap")),
        (TimeoutError("slow"), ("failed", "timeout")),
        (ConnectionResetError("reset"), ("failed", "network")),
        # Client-library classes that subclass neither builtin, matched on the
        # name parts "Timeout" and then "Connect". APITimeoutError subclasses
        # APIConnectionError, so it also pins that order.
        (anthropic.APITimeoutError(request=_PROVIDER_REQUEST), ("failed", "timeout")),
        (anthropic.APIConnectionError(request=_PROVIDER_REQUEST), ("failed", "network")),
        (httpx.ReadTimeout("read timed out"), ("failed", "timeout")),
        (httpx.ConnectError("connection refused"), ("failed", "network")),
        (psycopg_pool.PoolTimeout("no connection in 30s"), ("failed", "timeout")),
        (RuntimeError("worker killed"), ("failed", "error")),
        (ValueError("bad config"), ("failed", "error")),
        (KeyboardInterrupt(), ("failed", "interrupted")),
    ],
)
def test_an_attempt_is_classified_without_its_message(exc, expected):
    assert live_budget.classify_attempt(exc) == expected


def test_last_attempt_is_whitelisted_to_three_fields():
    clean = live_budget.clean_last_attempt
    assert clean({
        "on": "2026-08-27", "outcome": "failed", "kind": "network",
        "error": "sk-live-SECRET", "spent_usd": 49.0,
    }) == {"on": "2026-08-27", "outcome": "failed", "kind": "network"}
    assert clean({"on": "2026-08-27", "outcome": "ok", "kind": "network"})["kind"] is None
    assert clean({"on": "2026-08-27", "outcome": "failed", "kind": "boom"})["kind"] == "error"
    assert clean({"on": "2026-08-27", "outcome": "exploded"}) is None
    assert clean({"on": "Aug 27", "outcome": "ok"}) is None
    assert clean(["2026-08-27", "ok"]) is None
    assert clean(None) is None


# --- the ledger row ------------------------------------------------------------


def test_spend_run_id_is_one_row_per_month_and_entry():
    assert live_budget.spend_run_id(_AUG, "deepseek_v4_pro") == "live_spend_202608_deepseek_v4_pro"


def test_attempts_accumulate_into_one_row_and_carry_no_error_text():
    first = _attempt()
    first.strategy_impl = _impl(input_tokens=1_000_000, llm_calls=7)
    first.record_session()
    row = _ledger()
    assert row["session_id"] == "leaderboard-live-spend"
    assert row["mode"] == "leaderboard_spend"
    assert row["start_date"] == _AUG and row["end_date"] == _FREEZE
    assert row["est_cost_usd"] == pytest.approx(0.435)
    assert row["input_tokens"] == 1_000_000 and row["llm_calls"] == 7
    # A session close is not the end of the attempt: no outcome yet.
    assert "last_attempt" not in row["metadata"]
    first.finish(None)
    assert _ledger()["metadata"]["last_attempt"] == {
        "on": _FREEZE, "outcome": "ok", "kind": None,
    }

    second = _attempt()
    second.strategy_impl = _impl(input_tokens=1_000_000, llm_calls=3)
    second.finish(RuntimeError("provider said sk-live-SECRET"))
    row = _ledger()
    assert row["est_cost_usd"] == pytest.approx(0.87)
    assert row["input_tokens"] == 2_000_000 and row["llm_calls"] == 10
    assert row["metadata"]["attempts"] == 2
    assert row["metadata"]["price_source"] == lb_service.token_cost.PRICING_SOURCE_VERSION
    assert row["metadata"]["last_attempt"] == {
        "on": _FREEZE, "outcome": "failed", "kind": "error",
    }
    assert "SECRET" not in json.dumps(row, default=str)


def test_a_session_close_keeps_the_previous_outcome_until_the_attempt_ends():
    _insert_ledger("deepseek_v4_pro", 1.0, last_attempt={
        "on": "2026-08-26", "outcome": "ok", "kind": None,
    })
    attempt = _attempt()
    attempt.strategy_impl = _impl(input_tokens=1_000_000)
    attempt.record_session()
    assert _ledger()["metadata"]["last_attempt"]["on"] == "2026-08-26"
    assert _ledger()["est_cost_usd"] == pytest.approx(1.435)


def test_the_first_attempt_seeds_from_a_larger_checkpoint():
    checkpoint = {
        "deepseek_v4_pro": {
            "est_cost_usd": 3.0, "input_tokens": 5, "output_tokens": 6, "llm_calls": 7,
        }
    }
    assert live_budget.budget_status(_AUG, checkpoint_rows=checkpoint)["spent_usd"] == 3.0
    attempt = _attempt(checkpoint_rows=checkpoint)
    attempt.strategy_impl = _impl(input_tokens=1_000_000, llm_calls=7)
    attempt.finish(None)
    row = _ledger()
    assert row["est_cost_usd"] == pytest.approx(3.435)
    assert row["input_tokens"] == 1_000_005
    assert row["output_tokens"] == 6
    assert row["llm_calls"] == 14


def test_a_larger_ledger_wins_over_the_checkpoint():
    _insert_ledger("deepseek_v4_pro", 9.0)
    checkpoint = {"deepseek_v4_pro": {"est_cost_usd": 3.0}}
    assert live_budget.budget_status(_AUG, checkpoint_rows=checkpoint)["spent_usd"] == 9.0


def test_month_spend_sums_every_entry_of_that_month_only():
    _insert_ledger("deepseek_v4_pro", 1.0)
    _insert_ledger("dropped_model", 2.0)
    _insert_ledger("deepseek_v4_pro", 60.0, month="2026-07-01")
    checkpoint = {"qwen3_7_plus": {"est_cost_usd": 0.5}, "spy_index": {"est_cost_usd": 0}}
    august = live_budget.budget_status(_AUG, checkpoint_rows=checkpoint)
    assert august == {
        "month": "2026-08",
        "cap_usd": 50.0,
        "spent_usd": 3.5,
        "remaining_usd": 46.5,
        "halted": False,
    }
    july = live_budget.budget_status("2026-07-01")
    assert july["spent_usd"] == 60.0 and july["halted"] is True
    attempt = _attempt(checkpoint_rows=checkpoint)
    assert attempt.month_spent_usd() == pytest.approx(3.5)


@pytest.mark.parametrize("figure", [None, "n/a", float("nan"), float("inf"), -3.0])
def test_an_unreadable_spend_figure_counts_as_nothing(figure):
    """Rows written before the ledger can carry NULL or junk costs; they must
    neither crash the pre-check nor turn the month's total into nan."""
    status = live_budget.budget_status(
        _AUG, checkpoint_rows={"deepseek_v4_pro": {"est_cost_usd": figure}}
    )
    assert status["spent_usd"] == 0.0 and status["halted"] is False


def test_the_pre_check_and_the_in_run_stop(monkeypatch):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "1.0")
    _insert_ledger("qwen3_7_plus", 0.6)
    attempt = _attempt()
    attempt.check_can_start()  # 0.60 < 1.00
    attempt.strategy_impl = _impl(input_tokens=400_000)  # $0.174 per session
    attempt.record_session()
    attempt.stop_if_next_session_would_cross()  # 0.774 + 0.174 = 0.948 <= 1.00
    attempt.strategy_impl.input_tokens = 800_000
    attempt.record_session()
    with pytest.raises(live_budget.LiveBudgetExceeded, match="would cross it"):
        attempt.stop_if_next_session_would_cross()  # 0.948 + 0.174 > 1.00
    assert _ledger()["est_cost_usd"] == pytest.approx(0.348)

    _insert_ledger("qwen3_7_plus", 1.0)
    with pytest.raises(live_budget.LiveBudgetExceeded, match="not started"):
        _attempt().check_can_start()


def test_last_attempts_reads_this_month_only():
    _insert_ledger("deepseek_v4_pro", 1.0, last_attempt={
        "on": "2026-08-27", "outcome": "budget_halted", "kind": "budget",
    })
    _insert_ledger("qwen3_7_plus", 1.0, month="2026-07-01", last_attempt={
        "on": "2026-07-31", "outcome": "failed", "kind": "network",
    })
    _insert_ledger("claude_haiku_4_5", 1.0)  # no attempt finished yet
    assert live_budget.last_attempts(_AUG) == {
        "deepseek_v4_pro": {"on": "2026-08-27", "outcome": "budget_halted", "kind": "budget"},
    }


# --- nobody else reads the ledger ------------------------------------------------


def test_ledger_rows_are_invisible_to_every_other_reader_and_survive_clear():
    today = datetime.now(timezone.utc).date()
    operator_cost_before = db.aggregate_operator_cost_for_day(today)
    attempt = _attempt()
    attempt.strategy_impl = _impl(input_tokens=1_000_000)
    attempt.finish(None)
    run_id = live_budget.spend_run_id(_AUG, "deepseek_v4_pro")

    # The admin daily job's operator-cost roll-up skips rows with no owner.
    assert db.aggregate_operator_cost_for_day(today) == operator_cost_before

    assert live.latest_live_month_runs(_AUG, _FREEZE) == {}
    assert live._entry_month_runs("deepseek_v4_pro", _AUG, _FREEZE) == []
    for session in (live.LIVE_SESSION_ID, live_budget.LIVE_SPEND_SESSION_ID):
        assert lb_service._find_cached_run("deepseek_v4_pro", _AUG, _FREEZE, session) is None
    assert run_id not in {row["run_id"] for row in build_leaderboard(db.get_all_runs())}

    # The public run listing defaults to mode=backtest. Asked for the ledger's
    # mode by name it does list the row, through the RunMetadata allow-list:
    # no dollar figure, no token counts and no metadata (so no last_attempt).
    client = TestClient(app)
    headers = {"X-Session-Id": "ledger-test"}
    listed = client.get("/runs", headers=headers)
    assert listed.status_code == 200, listed.text
    assert run_id not in {row["run_id"] for row in listed.json()}
    by_mode = client.get("/runs?mode=leaderboard_spend", headers=headers).json()
    record = next(row for row in by_mode if row["run_id"] == run_id)
    assert not {"est_cost_usd", "input_tokens", "output_tokens", "metadata"} & set(record)
    # The per-run routes: one answers only a caller who names the ledger's
    # session, with an empty curve and no dollar figure; the other rejects that
    # session id outright, since it is not a UUID.
    spend_session = {"X-Session-Id": live_budget.LIVE_SPEND_SESSION_ID}
    single = client.get(f"/api/backtest/{run_id}", headers=spend_session)
    assert single.status_code == 200, single.text
    assert single.json()["data"] == []
    assert "0.435" not in single.text and "input_tokens" not in single.text
    assert client.get(f"/runs/{run_id}", headers=spend_session).status_code == 400

    # The money was spent: clearing the board's rows must not forget it.
    live.clear_live_session_runs()
    assert db.get_run(run_id)["est_cost_usd"] == pytest.approx(0.435)
```

- [ ] **Step 2: Run it and confirm it fails.**

```bash
pytest dashboard/backend/tests/test_live_budget.py -q -p no:cacheprovider
```

Expected: a collection error, `ImportError: cannot import name 'live_budget' from 'dashboard.backend.domain.leaderboard'`, and `1 error`.

- [ ] **Step 3: Implement.** Create `dashboard/backend/domain/leaderboard/live_budget.py` with exactly:

```python
"""Monthly spend ledger and hard stop for the Live Leaderboard's model deploys.

The live board's model calls go straight to the operator's CommonStack key and
bypass the Credits layer, so nothing else bounds them. Checkpoint rows cannot
serve as the record either: a session H6 refuses to store, a run that dies
before its next session close, and every row a forced or config replay deletes
were all billed and leave no row behind. So every billable attempt is priced
into one ``agent_runs`` row per (month, entry), cumulative across successes,
failures, H6 refusals and replays, and the cap is checked against that.

The row is invisible to every other reader by construction: its session
(``leaderboard-live-spend``) is not the board's, its mode
(``leaderboard_spend``) is not ``leaderboard`` or ``backtest``, and its run id
does not start with ``run_``. ``clear_live_session_runs`` deletes only the
board's session, so the ledger survives a ``--clear``: that money was spent.

Spend is an estimate: provider-reported tokens times the
``infrastructure/llm/pricing.py`` snapshot, because CommonStack has no usage
endpoint. A month-to-date figure per entry is ``max(ledger, latest checkpoint
row)``, so the first deploy after this module shipped starts from what the
checkpoints already record instead of from zero.

A leaf: imports the runs database and the price table, never ``live`` or
``service`` (``live`` imports this module).
"""

from __future__ import annotations

import json
import math
import os
from datetime import date
from typing import Any, Dict, Mapping, Optional, Tuple

from dashboard.backend.database import db
from dashboard.backend.infrastructure.llm import token_cost

LIVE_SPEND_SESSION_ID = "leaderboard-live-spend"
LIVE_SPEND_MODE = "leaderboard_spend"
LEDGER_NAME = "live_spend"

BUDGET_ENV = "LEADERBOARD_MONTHLY_BUDGET_USD"
DEFAULT_MONTHLY_BUDGET_USD = 50.0
MAX_MONTHLY_BUDGET_USD = 10_000.0

OUTCOME_OK = "ok"
OUTCOME_FAILED = "failed"
OUTCOME_BUDGET_HALTED = "budget_halted"
OUTCOMES = frozenset({OUTCOME_OK, OUTCOME_FAILED, OUTCOME_BUDGET_HALTED})

KIND_BUDGET = "budget"
KIND_MODEL_COVERAGE = "model_coverage"
KIND_MARKET_DATA_GAP = "market_data_gap"
KIND_TIMEOUT = "timeout"
KIND_NETWORK = "network"
KIND_INTERRUPTED = "interrupted"
KIND_ERROR = "error"
KINDS = frozenset({
    KIND_BUDGET,
    KIND_MODEL_COVERAGE,
    KIND_MARKET_DATA_GAP,
    KIND_TIMEOUT,
    KIND_NETWORK,
    KIND_INTERRUPTED,
    KIND_ERROR,
})

# Matched by class *name* along the exception's MRO, so this leaf never imports
# the modules that define them (live.py, service.py) or any client library.
# Exact names first, in order: TimeoutError is an OSError, so it is tested
# before anything broader.
_KIND_BY_TYPE_NAME: Tuple[Tuple[str, str], ...] = (
    ("LeaderboardFallbackError", KIND_MODEL_COVERAGE),
    ("LiveSessionGapError", KIND_MARKET_DATA_GAP),
    ("TimeoutError", KIND_TIMEOUT),
    ("ConnectionError", KIND_NETWORK),
)
# Then name parts, for client-library classes that subclass neither builtin:
# psycopg_pool.PoolTimeout, httpx.ReadTimeout and anthropic.APITimeoutError
# carry "Timeout"; httpx.ConnectError and anthropic.APIConnectionError carry
# "Connect". "Timeout" goes first because APITimeoutError subclasses
# APIConnectionError. On the live path a provider SDK error never gets here
# (PortfolioManager turns it into a rule-based step, which H6 then reports as
# model_coverage); a store timeout from a checkpoint or ledger write does.
_KIND_BY_TYPE_NAME_PART: Tuple[Tuple[str, str], ...] = (
    ("Timeout", KIND_TIMEOUT),
    ("Connect", KIND_NETWORK),
)

# Junk values already reported, so a refresh that reads the cap once per entry
# prints the warning once rather than once per read.
_warned_values: set = set()


class LiveBudgetExceeded(RuntimeError):
    """The month's live model budget is spent; the entry is not traded further.

    A ``RuntimeError`` so a caller that predates it still treats it as a
    failed deploy, but ``refresh_live_leaderboard`` catches it first and
    reports it as a halt, not a failure.
    """


def monthly_budget_usd() -> float:
    """The live board's monthly model budget in USD, read on every call.

    Unset or blank is the default, silently. Junk, ``nan``/``inf``, negative
    or above ``MAX_MONTHLY_BUDGET_USD`` warns once and falls back to the
    default: this runs on every live deploy and must never raise. ``0`` is a
    real cap that blocks every billable live deploy, not "off": a hard stop
    that a typo could turn into "unlimited" would fail in the wrong direction.
    To lift the cap, set a large number.
    """
    raw = os.getenv(BUDGET_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MONTHLY_BUDGET_USD
    try:
        value: Optional[float] = float(raw.strip())
    except ValueError:
        value = None
    if (
        value is None
        or not math.isfinite(value)
        or value < 0
        or value > MAX_MONTHLY_BUDGET_USD
    ):
        if raw not in _warned_values:
            _warned_values.add(raw)
            print(
                f"WARNING: {BUDGET_ENV}={raw!r} is not a dollar amount between 0 "
                f"and {MAX_MONTHLY_BUDGET_USD:.0f}; using the default "
                f"{DEFAULT_MONTHLY_BUDGET_USD:.2f}",
                flush=True,
            )
        return DEFAULT_MONTHLY_BUDGET_USD
    return 0.0 if value == 0 else value


def spend_run_id(month_start: str, entry_id: str) -> str:
    """The ledger row's id: one per (month, entry), so writes are upserts."""
    return f"live_spend_{month_start[:4]}{month_start[5:7]}_{entry_id}"


def _metadata(run: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    meta = (run or {}).get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except ValueError:
            return {}
    return meta if isinstance(meta, dict) else {}


def _usd(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def _count(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _spend(run: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    run = run or {}
    return {
        "usd": _usd(run.get("est_cost_usd")),
        "input_tokens": _count(run.get("input_tokens")),
        "output_tokens": _count(run.get("output_tokens")),
        "llm_calls": _count(run.get("llm_calls")),
    }


def month_ledger_rows(month_start: str) -> Dict[str, Dict[str, Any]]:
    """This month's ledger row per entry id."""
    rows: Dict[str, Dict[str, Any]] = {}
    for run in db.get_runs_by_session(LIVE_SPEND_SESSION_ID) or []:
        if run.get("mode") != LIVE_SPEND_MODE or run.get("start_date") != month_start:
            continue
        entry_id = run.get("llm_model")
        if entry_id:
            rows[str(entry_id)] = run
    return rows


def _merged_spend(
    ledger: Mapping[str, Mapping[str, Any]],
    checkpoint_rows: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Month-to-date spend per entry: whichever of ledger and checkpoint is larger.

    In steady state the ledger is the larger (it also holds failed and replayed
    attempts). The checkpoint wins only before an entry's first priced attempt,
    which is what makes the first deploy after rollout start from real spend.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for entry_id in set(ledger) | set(checkpoint_rows):
        ledger_spend = _spend(ledger.get(entry_id))
        checkpoint_spend = _spend(checkpoint_rows.get(entry_id))
        merged[entry_id] = (
            checkpoint_spend
            if checkpoint_spend["usd"] > ledger_spend["usd"]
            else ledger_spend
        )
    return merged


def budget_status(
    month_start: str,
    *,
    checkpoint_rows: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Cap, month-to-date spend and whether the cap is reached. Operator-facing:
    the refresh result and log carry it, the public GET never does."""
    spend = _merged_spend(month_ledger_rows(month_start), checkpoint_rows or {})
    spent = round(sum(item["usd"] for item in spend.values()), 6)
    cap = monthly_budget_usd()
    return {
        "month": month_start[:7],
        "cap_usd": cap,
        "spent_usd": spent,
        "remaining_usd": round(max(cap - spent, 0.0), 6),
        "halted": spent >= cap,
    }


def classify_attempt(exc: Optional[BaseException]) -> Tuple[str, Optional[str]]:
    """``(outcome, kind)`` for how an attempt ended. The kind is a coarse
    class and never the exception's message, which can carry provider text."""
    if exc is None:
        return OUTCOME_OK, None
    if isinstance(exc, LiveBudgetExceeded):
        return OUTCOME_BUDGET_HALTED, KIND_BUDGET
    if not isinstance(exc, Exception):
        return OUTCOME_FAILED, KIND_INTERRUPTED
    names = [cls.__name__ for cls in type(exc).__mro__]
    for name, kind in _KIND_BY_TYPE_NAME:
        if name in names:
            return OUTCOME_FAILED, kind
    for part, kind in _KIND_BY_TYPE_NAME_PART:
        if any(part in name for name in names):
            return OUTCOME_FAILED, kind
    return OUTCOME_FAILED, KIND_ERROR


def clean_last_attempt(value: Any) -> Optional[Dict[str, Any]]:
    """The public ``{on, outcome, kind}`` triple, or None for anything else.

    Whitelisted rather than passed through, so a key a later writer adds (an
    error message, a dollar figure) cannot reach the public payload.
    """
    if not isinstance(value, dict):
        return None
    on = value.get("on")
    outcome = value.get("outcome")
    if not isinstance(on, str) or outcome not in OUTCOMES:
        return None
    try:
        date.fromisoformat(on)
    except ValueError:
        return None
    kind = value.get("kind")
    if outcome == OUTCOME_OK:
        kind = None
    elif kind not in KINDS:
        kind = KIND_ERROR
    return {"on": on, "outcome": outcome, "kind": kind}


def last_attempts(month_start: str) -> Dict[str, Dict[str, Any]]:
    """Each entry's last finished attempt this month, as ``clean_last_attempt``."""
    out: Dict[str, Dict[str, Any]] = {}
    for entry_id, run in month_ledger_rows(month_start).items():
        last = clean_last_attempt(_metadata(run).get("last_attempt"))
        if last is not None:
            out[entry_id] = last
    return out


class LiveAttempt:
    """One billable deploy of one live entry, priced into the month's ledger.

    Built only once a deploy has work to do, so an entry that is already
    current costs nothing and writes nothing. ``check_can_start`` is the
    pre-check (before any bar fetch or model build), ``record_session`` the
    upsert at every session close, ``stop_if_next_session_would_cross`` the
    in-run stop, and ``finish`` the final pricing, whatever ended the attempt.
    """

    def __init__(
        self,
        entry: Mapping[str, Any],
        *,
        month_start: str,
        freeze_end: str,
        checkpoint_rows: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.entry_id = str(entry["id"])
        self.agent_name = str(entry.get("name") or self.entry_id)
        self.default_model_id = entry.get("model_id")
        self.month_start = month_start
        self.freeze_end = freeze_end
        self.cap_usd = monthly_budget_usd()
        self.strategy_impl: Any = None
        self.sessions_closed = 0
        ledger = month_ledger_rows(month_start)
        spend = _merged_spend(ledger, checkpoint_rows)
        self._base = spend.get(self.entry_id) or _spend(None)
        self._others_usd = sum(
            item["usd"] for entry_id, item in spend.items() if entry_id != self.entry_id
        )
        meta = _metadata(ledger.get(self.entry_id))
        self._prior_attempts = _count(meta.get("attempts"))
        self._prior_last = clean_last_attempt(meta.get("last_attempt"))

    def attempt_usd(self) -> float:
        """What this attempt has billed so far, from the strategy's own tokens."""
        impl = self.strategy_impl
        if impl is None:
            return 0.0
        model_id = getattr(impl, "model_id", None) or self.default_model_id
        return token_cost.estimate_cost_usd(
            model_id,
            _count(getattr(impl, "input_tokens", 0)),
            _count(getattr(impl, "output_tokens", 0)),
        )

    def entry_spent_usd(self) -> float:
        return round(self._base["usd"] + self.attempt_usd(), 6)

    def month_spent_usd(self) -> float:
        return round(self._others_usd + self.entry_spent_usd(), 6)

    def _exceeded(self, spent: float, why: str) -> LiveBudgetExceeded:
        return LiveBudgetExceeded(
            f"Live '{self.entry_id}': month {self.month_start[:7]} has spent "
            f"${spent:.2f} of its ${self.cap_usd:.2f} model budget ({BUDGET_ENV}); {why}"
        )

    def check_can_start(self) -> None:
        spent = self.month_spent_usd()
        if spent >= self.cap_usd:
            raise self._exceeded(spent, "not started")

    def record_session(self) -> None:
        """Upsert the ledger with everything billed so far. Called at every
        session close before H6 is consulted, so a session H6 refuses to store
        is priced too, and a process killed later loses at most one session."""
        self.sessions_closed += 1
        self._write(last_attempt=self._prior_last, attempts=self._prior_attempts)

    def stop_if_next_session_would_cross(self) -> None:
        """Raise when this attempt's own average session cost would take the
        month past the cap. Only called while a session is still to come."""
        if self.sessions_closed <= 0:
            return
        spent = self.month_spent_usd()
        per_session = self.attempt_usd() / self.sessions_closed
        if spent + per_session > self.cap_usd:
            raise self._exceeded(
                spent,
                f"stopped before a session averaging ${per_session:.4f} would cross it",
            )

    def finish(self, exc: Optional[BaseException]) -> None:
        """Final upsert, with how the attempt ended as ``last_attempt``."""
        outcome, kind = classify_attempt(exc)
        self._write(
            last_attempt={"on": self.freeze_end, "outcome": outcome, "kind": kind},
            attempts=self._prior_attempts + 1,
        )

    def _write(self, *, last_attempt: Optional[Dict[str, Any]], attempts: int) -> None:
        impl = self.strategy_impl
        metadata: Dict[str, Any] = {
            "ledger": LEDGER_NAME,
            "price_source": token_cost.PRICING_SOURCE_VERSION,
            "attempts": attempts,
        }
        if last_attempt is not None:
            metadata["last_attempt"] = last_attempt
        db.insert_run(
            run_id=spend_run_id(self.month_start, self.entry_id),
            session_id=LIVE_SPEND_SESSION_ID,
            agent_name=self.agent_name,
            mode=LIVE_SPEND_MODE,
            start_date=self.month_start,
            end_date=self.freeze_end,
            initial_equity=0.0,
            llm_model=self.entry_id,
            llm_calls=self._base["llm_calls"] + _count(getattr(impl, "llm_calls", 0)),
            input_tokens=self._base["input_tokens"]
            + _count(getattr(impl, "input_tokens", 0)),
            output_tokens=self._base["output_tokens"]
            + _count(getattr(impl, "output_tokens", 0)),
            est_cost_usd=self.entry_spent_usd(),
            metadata=metadata,
        )
```

**Strip the env var for the suite.** In `dashboard/backend/tests/conftest.py`, find:

```python
os.environ.pop("LEADERBOARD_DAILY_AUTO_DEPLOY", None)
os.environ.pop("LEADERBOARD_DAILY_REFRESH_SECRET", None)
```

Replace with:

```python
os.environ.pop("LEADERBOARD_DAILY_AUTO_DEPLOY", None)
os.environ.pop("LEADERBOARD_DAILY_REFRESH_SECRET", None)
# The Live board's monthly model budget. A developer with a small or zero cap
# exported would see every live deploy test halt on the budget instead of
# trading; the budget tests set it per case via monkeypatch.
os.environ.pop("LEADERBOARD_MONTHLY_BUDGET_USD", None)
```

- [ ] **Step 4: Run it and confirm it passes.**

```bash
pytest dashboard/backend/tests/test_live_budget.py -q -p no:cacheprovider
```

Expected: `43 passed`. Then the neighbours (nothing calls the module yet, so they must be unchanged; `test_architecture_boundaries.py` shells out to `git ls-files`, so run it from the repo checkout):

```bash
pytest dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/test_env_isolation.py dashboard/backend/tests/test_architecture_boundaries.py -q -p no:cacheprovider
```

Expected: green; measured `136 passed`.

- [ ] **Step 5: Commit.**

```bash
git add dashboard/backend/domain/leaderboard/live_budget.py dashboard/backend/tests/test_live_budget.py dashboard/backend/tests/conftest.py
git commit -m "feat(live): add the monthly spend ledger module"
```

### Task 3.2: `LLMAgentStrategy` publishes billed tokens after every step

**Files:**
- Modify: `dashboard/backend/domain/leaderboard/strategies/llm_agent.py`
- Test: `dashboard/backend/tests/test_live_budget.py`

**Interfaces:**
- Consumes: `PortfolioManager.input_tokens` / `.output_tokens` (already counted per call by `_record_llm_usage`).
- Produces: `LLMAgentStrategy.input_tokens` / `.output_tokens` current after each decision call returns, so `LiveAttempt.finish` can price a run that dies between session closes. `llm_calls`, `llm_decisions`, `decision_steps` and the snapshot still move only in `_record_progress` (H6 untouched). At a session close the mirrored values equal what `_record_progress` sets, so checkpoint rows are unchanged.

The test goes in `test_live_budget.py`, not `test_live_leaderboard.py`, because P2 (`2026-10-14-live-board-fixes-p2-memory-robustness.md`; Decision 3 in the index plan) adds tests next to `test_llm_agent_fires_the_hook_per_session_from_one_book`; keeping this one apart avoids a merge conflict.

- [ ] **Step 1: Write the failing test.** Append to the end of `dashboard/backend/tests/test_live_budget.py` (two blank lines before the section comment):

```python
# --- tokens a crash leaves behind ---------------------------------------------


def _token_counting_manager(*, raise_in_decision_on=None, raise_in_execute_on=None):
    """A PortfolioManager stand-in that bills 100 input / 50 output tokens per
    model call and can die inside the call or just after it."""

    class Manager:
        def __init__(self, **kwargs):
            self.equity_history = []
            self.trades = []
            self.llm_calls = self.llm_decisions = 0
            self.input_tokens = self.output_tokens = 0

        def restore_state(self, snapshot):
            pass

        def snapshot_state(self):
            return {"cash": 1.0, "positions": {}}

        def get_portfolio_state(self, *args):
            return {}

        def make_trading_decision_with_llm(self, state, client, **kwargs):
            if self.llm_calls + 1 == raise_in_decision_on:
                raise RuntimeError("provider down")
            self.llm_calls += 1
            self.llm_decisions += 1
            self.input_tokens += 100
            self.output_tokens += 50
            return {"actions": []}

        def execute_actions(self, *args):
            if self.llm_calls == raise_in_execute_on:
                raise RuntimeError("broker rejected")

        def update_equity(self, market_data, price_cache, ts):
            self.equity_history.append({"timestamp": ts, "equity": 100.0})

        def get_equity_curve(self):
            return self.equity_history

    return Manager


@pytest.mark.parametrize(
    ("manager_kwargs", "tokens"),
    [
        # Dies inside the 4th call, before it is billed: three calls priced.
        ({"raise_in_decision_on": 4}, (300, 150)),
        # Dies after the 2nd call returned and was billed, before its fills.
        ({"raise_in_execute_on": 2}, (200, 100)),
    ],
)
def test_llm_agent_publishes_tokens_billed_before_a_crash(monkeypatch, manager_kwargs, tokens):
    """The manager is local to the loop, so tokens it counted were lost with it
    when a run died before its next session close. The Live spend ledger
    prices a dead attempt from the strategy's own counters."""
    import pandas as pd

    from dashboard.backend.domain.leaderboard.strategies import llm_agent

    monkeypatch.setattr(
        llm_agent, "PortfolioManager", _token_counting_manager(**manager_kwargs)
    )
    stamps = [pd.Timestamp(f"2026-08-03T{h}:00:00Z") for h in range(13, 20)]
    strategy = llm_agent.LLMAgentStrategy({"strategy": "llm_agent", "model_id": "m"})
    with pytest.raises(RuntimeError):
        strategy._run_decision_loop(
            client=object(), timestamps=stamps, symbols=["AAPL"], data={},
            price_cache={}, initial_capital=100.0, model_id="m",
            on_session_close=lambda day, curve: None,
        )
    assert (strategy.input_tokens, strategy.output_tokens) == tokens
    # The H6 counters still move only at a session close, which never came.
    assert strategy.llm_calls == 0 and strategy.decision_steps == 0
```

- [ ] **Step 2: Run it and confirm it fails.**

```bash
pytest dashboard/backend/tests/test_live_budget.py -q -p no:cacheprovider -k tokens_billed_before_a_crash
```

Expected: `2 failed`, with `assert (0, 0) == (300, 150)` and `assert (0, 0) == (200, 100)`.

- [ ] **Step 3: Implement.**

**Edit 1 — document the per-step counters (in `LLMAgentStrategy.__init__`, ≈:157 on main@17c552c5).** In `dashboard/backend/domain/leaderboard/strategies/llm_agent.py`, find:

```python
        self.planned_decision_steps = 0
        self.input_tokens = 0
        self.output_tokens = 0
```

Replace with:

```python
        self.planned_decision_steps = 0
        # Billed tokens are current after every decision step, not only at a
        # session close like the counters above: the Live board's spend ledger
        # prices a run that dies mid-session from these two.
        self.input_tokens = 0
        self.output_tokens = 0
```

**Edit 2 — mirror the tokens right after the decision call (in `_run_decision_loop`, ≈:246).** In `dashboard/backend/domain/leaderboard/strategies/llm_agent.py`, find:

```python
            else:
                decision = manager.make_trading_decision(state)

            manager.execute_actions(decision.get("actions", []), market_data, ts)
```

Replace with:

```python
            else:
                decision = manager.make_trading_decision(state)
            # Published per step because the manager is local to this loop: a
            # run that dies before its next session close would otherwise take
            # the tokens it already billed with it. At a close these equal what
            # _record_progress sets, so a checkpoint reads the same numbers.
            self.input_tokens = manager.input_tokens
            self.output_tokens = manager.output_tokens

            manager.execute_actions(decision.get("actions", []), market_data, ts)
```

Placed before `execute_actions`, not after `update_equity`: a call that returned and was billed must be priced even when its fills raise (the second test case).

- [ ] **Step 4: Run it and confirm it passes.**

```bash
pytest dashboard/backend/tests/test_live_budget.py -q -p no:cacheprovider
pytest dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/test_leaderboard_samples.py dashboard/backend/tests/domain/leaderboard -q -p no:cacheprovider
```

Expected: `45 passed` for the first command; the second stays green (its fake managers all define `input_tokens`/`output_tokens`; measured `239 passed`).

- [ ] **Step 5: Commit.**

```bash
git add dashboard/backend/domain/leaderboard/strategies/llm_agent.py dashboard/backend/tests/test_live_budget.py
git commit -m "fix(live): publish billed tokens after every decision step"
```

### Task 3.3: Enforce the cap in the live deploy and report halts in the refresh

**Files:**
- Modify: `dashboard/backend/domain/leaderboard/live.py`
- Modify: `dashboard/backend/tests/test_live_leaderboard.py` (fixture cleanup and `_ScriptedAgent.step_tokens` only)
- Test: `dashboard/backend/tests/test_live_budget.py`

**Interfaces:**
- Consumes: `live_budget.LiveAttempt`, `live_budget.LiveBudgetExceeded`, `live_budget.budget_status` (Task 3.1); per-step tokens (Task 3.2); `latest_live_month_runs`, `live_increment_bounds`, `_checkpoint_passes_h6`, `_write_live_checkpoint` (unchanged).
- Produces: `_trade_live_segment(...)`; both already-current returns in `deploy_live_model_increment`'s head, ahead of the `LiveAttempt`; refresh result keys `budget_halted` and `budget`; the `ERROR: live.budget_exhausted …`, `ERROR: live.budget_status_failed …` and `ERROR: live.spend_ledger_write_failed …` lines.

- [ ] **Step 1: Write the failing tests.**

**Edit 1 — the shared fixture also clears the ledger (every deploy now writes a ledger row, and the test DB is session-scoped).** In `dashboard/backend/tests/test_live_leaderboard.py`, find:

```python
    monkeypatch.setattr(live, "_LIVE_REFRESH_STATE_PATH", tmp_path / "live_state.json")
    for run in db.get_runs_by_session("leaderboard-live") or []:
        db.delete_run(run["run_id"])
    yield
    for run in db.get_runs_by_session("leaderboard-live") or []:
        db.delete_run(run["run_id"])
```

Replace with:

```python
    monkeypatch.setattr(live, "_LIVE_REFRESH_STATE_PATH", tmp_path / "live_state.json")
    _clear_live_and_spend_rows()
    yield
    _clear_live_and_spend_rows()


def _clear_live_and_spend_rows():
    # Every live deploy also prices itself into the spend ledger's session.
    for session in ("leaderboard-live", "leaderboard-live-spend"):
        for run in db.get_runs_by_session(session) or []:
            db.delete_run(run["run_id"])
```

**Edit 2 — `_ScriptedAgent` takes a per-session token count (default keeps today's 10/10).** In `dashboard/backend/tests/test_live_leaderboard.py`, find:

```python
    def __init__(self, *, steps=7, misses=None, crash_on=(), no_bars=(), runs=None):
        self.steps = steps
```

Replace with:

```python
    def __init__(self, *, steps=7, misses=None, crash_on=(), no_bars=(), runs=None,
                 step_tokens=(10, 10)):
        self.steps = steps
        # (input, output) tokens billed per traded session.
        self.step_tokens = step_tokens
```

**Edit 3 — use it.** In `dashboard/backend/tests/test_live_leaderboard.py`, find:

```python
            self.input_tokens += 10
            self.output_tokens += 10
```

Replace with:

```python
            self.input_tokens += self.step_tokens[0]
            self.output_tokens += self.step_tokens[1]
```

**Edit 4 — import the live-board test helpers.** In `dashboard/backend/tests/test_live_budget.py`, find:

```python
import dashboard.backend.domain.leaderboard.service as lb_service
```

Replace with:

```python
import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.tests.test_live_leaderboard import (
    SEED,
    _ET,
    _ScriptedAgent,
    _deepseek,
    _insert_checkpoint,
    _insert_live_run,
    _live_aug27,
    _month_rows,
    _one_bar_frame,
    _pt,
    _use_agent,
)
```

**Edit 5 — the tests.** Append to the end of `dashboard/backend/tests/test_live_budget.py` (two blank lines before the section comment):

```python
# --- enforcement in the live deploy ---------------------------------------------


def _no_billable_work(*_args, **_kwargs):
    raise AssertionError("a budget halt reached the model or the market data")


class _DiesBeforeTheClose(_ScriptedAgent):
    """Bills a session, then dies before that session's close hook fires."""

    def __init__(self, die_on, error, **kwargs):
        super().__init__(**kwargs)
        self.die_on = die_on
        self.error = error

    def run(self, bars, start, end, capital, starting_snapshot=None, on_session_close=None):
        def hook(day, curve):
            if day.isoformat() == self.die_on:
                raise self.error
            on_session_close(day, curve)

        return super().run(
            bars, start, end, capital,
            starting_snapshot=starting_snapshot, on_session_close=hook,
        )


def test_a_zero_budget_blocks_before_any_bar_fetch_or_model_build(monkeypatch):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0")
    monkeypatch.setattr(lb_service, "get_strategy", _no_billable_work)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", _no_billable_work)
    freeze_cfg = _live_aug27()
    with pytest.raises(live_budget.LiveBudgetExceeded, match="not started"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert _month_rows("deepseek_v4_pro") == []
    ledger = _ledger()
    assert ledger["est_cost_usd"] == 0
    assert ledger["metadata"]["last_attempt"] == {
        "on": _FREEZE, "outcome": "budget_halted", "kind": "budget",
    }


def test_an_entry_already_current_is_not_halted_and_writes_nothing(monkeypatch):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0")
    _insert_checkpoint(_FREEZE, "08-27")
    freeze_cfg = _live_aug27()
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert row["cached"] is True
    assert _ledger() is None


def test_a_snapshot_that_already_covers_a_non_session_freeze_is_not_halted(monkeypatch):
    """The second way to be current: the newest checkpoint is the last session
    before a freeze_end that is not itself a session. live_clock never freezes
    on such a day, but a deploy handed one must not build the model, fetch
    bars, write a ledger row or halt either."""
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0")
    monkeypatch.setattr(lb_service, "get_strategy", _no_billable_work)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", _no_billable_work)
    _insert_checkpoint("2026-08-28", "08-28")
    freeze_cfg = {**_live_aug27(), "end_date": "2026-08-29"}  # a Saturday
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert row["cached"] is True and row["segment"] is None
    assert row["run_id"] == "lb_deepseek_v4_pro_20260801_20260828"
    assert _ledger() is None


def test_a_catch_up_stops_before_the_session_that_would_cross_the_cap(monkeypatch):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0.8")
    _insert_checkpoint("2026-08-24", "08-24")
    agent = _use_agent(monkeypatch, _ScriptedAgent(step_tokens=(1_000_000, 0)))
    freeze_cfg = _live_aug27()
    with pytest.raises(live_budget.LiveBudgetExceeded, match="would cross it"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    # $0.435 spent; a second $0.435 session would reach $0.87 > $0.80.
    assert agent.traded == ["2026-08-25"]
    # The session that was paid for is stored, not thrown away.
    assert [r["end_date"] for r in _month_rows("deepseek_v4_pro")] == ["2026-08-25"]
    ledger = _ledger()
    assert ledger["est_cost_usd"] == pytest.approx(0.435)
    assert ledger["metadata"]["last_attempt"]["outcome"] == "budget_halted"


def test_the_last_session_of_a_segment_is_never_stopped_by_its_projection(monkeypatch):
    """Nothing is left to stop after it, so raising would only relabel a
    finished night as a halt. The next night's pre-check decides."""
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0.5")
    _insert_checkpoint("2026-08-26", "08-26")
    _use_agent(monkeypatch, _ScriptedAgent(step_tokens=(1_000_000, 0)))
    freeze_cfg = _live_aug27()
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert row["window"]["end_date"] == _FREEZE
    assert _ledger()["metadata"]["last_attempt"]["outcome"] == "ok"


def test_a_run_that_dies_mid_session_is_priced_in_full(monkeypatch):
    _insert_checkpoint("2026-08-24", "08-24")
    _use_agent(monkeypatch, _DiesBeforeTheClose(
        "2026-08-26",
        ConnectionResetError("reset by peer, key sk-live-SECRET"),
        step_tokens=(1_000_000, 0),
    ))
    freeze_cfg = _live_aug27()
    with pytest.raises(ConnectionResetError):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert [r["end_date"] for r in _month_rows("deepseek_v4_pro")] == ["2026-08-25"]
    ledger = _ledger()
    # 08-25 was priced at its close; 08-26's tokens only by the final pricing.
    assert ledger["est_cost_usd"] == pytest.approx(0.87)
    assert ledger["metadata"]["last_attempt"] == {
        "on": _FREEZE, "outcome": "failed", "kind": "network",
    }
    assert "SECRET" not in json.dumps(ledger, default=str)


def test_a_ledger_that_cannot_be_written_never_hides_the_deploy_error(monkeypatch, capsys):
    _insert_checkpoint("2026-08-24", "08-24")
    real_insert = db.insert_run

    def ledger_down_at_the_end(**row):
        if row["run_id"].startswith("live_spend_") and "last_attempt" in row["metadata"]:
            raise OSError("ledger disk full")
        return real_insert(**row)

    monkeypatch.setattr(db, "insert_run", ledger_down_at_the_end)
    _use_agent(monkeypatch, _ScriptedAgent(
        crash_on={"2026-08-26"}, step_tokens=(1_000_000, 0),
    ))
    freeze_cfg = _live_aug27()
    with pytest.raises(RuntimeError, match="worker killed"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert (
        "ERROR: live.spend_ledger_write_failed entry=deepseek_v4_pro error=OSError"
        in capsys.readouterr().out
    )
    # The session-close upsert for 08-25 landed before the final one failed.
    assert _ledger()["est_cost_usd"] == pytest.approx(0.435)


def test_a_segment_h6_refuses_to_store_is_still_billed(monkeypatch):
    _use_agent(monkeypatch, _ScriptedAgent(
        misses={"2026-08-03": 7}, step_tokens=(1_000_000, 0),
    ))
    freeze_cfg = _live_aug27()
    with pytest.raises(lb_service.LeaderboardFallbackError):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert _month_rows("deepseek_v4_pro") == []
    ledger = _ledger()
    assert ledger["est_cost_usd"] == pytest.approx(0.435)
    assert ledger["metadata"]["last_attempt"]["kind"] == "model_coverage"


def test_a_forced_replay_adds_to_the_month_instead_of_resetting_it(monkeypatch):
    freeze_cfg = _live_aug27()
    _use_agent(monkeypatch, _ScriptedAgent(step_tokens=(1_000_000, 0)))
    live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    _use_agent(monkeypatch, _ScriptedAgent(step_tokens=(1_000_000, 0)))
    row = live.deploy_live_model_increment(
        _deepseek(freeze_cfg), freeze_cfg, force_refresh=True
    )
    month = 19 * 0.435  # 19 sessions, Aug 3 to Aug 27
    # The checkpoint chain starts over with the replay; the money does not.
    assert db.get_run(row["run_id"])["est_cost_usd"] == pytest.approx(month)
    assert _ledger()["est_cost_usd"] == pytest.approx(2 * month)
    assert _ledger()["metadata"]["attempts"] == 2


def test_the_first_deploy_after_rollout_counts_what_checkpoints_already_spent(monkeypatch):
    # A row written before the ledger existed records $49.50 of the $50 cap.
    _insert_live_run(
        "lb_deepseek_v4_pro_20260801_20260824", "deepseek_v4_pro", "2026-08-24",
        [_pt("2026-08-24T19:00:00+00:00", SEED)],
        metadata={live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}, "day": "08-24"}},
        est_cost_usd=49.5,
    )
    agent = _use_agent(monkeypatch, _ScriptedAgent(step_tokens=(1_000_000, 0)))
    freeze_cfg = _live_aug27()
    with pytest.raises(live_budget.LiveBudgetExceeded):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert agent.traded == ["2026-08-25"]
    assert _ledger()["est_cost_usd"] == pytest.approx(49.935)


def test_a_new_month_starts_with_nothing_spent(monkeypatch):
    _insert_ledger("deepseek_v4_pro", 60.0, month="2026-07-01")
    _use_agent(monkeypatch, _ScriptedAgent())
    freeze_cfg = _live_aug27()
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert row["cached"] is False
    assert _ledger(month="2026-07-01")["est_cost_usd"] == 60.0
    assert _ledger()["est_cost_usd"] < 1


def test_a_spent_month_halts_the_roster_with_one_log_line(monkeypatch, capsys):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0")
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "get_strategy", _no_billable_work)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", _no_billable_work)
    as_of = datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    roster = [entry["id"] for entry in live.live_llm_entries(_live_aug27())]

    result = live.refresh_live_leaderboard(deploy_models=True, as_of=as_of)
    assert result["models_deployed"] is False
    assert result["model_failures"] == []
    assert result["budget_halted"] == roster
    assert result["budget"] == {
        "month": "2026-08",
        "cap_usd": 0.0,
        "spent_usd": 0.0,
        "remaining_usd": 0.0,
        "halted": True,
    }
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if "live.budget_exhausted" in line] == [
        "ERROR: live.budget_exhausted month=2026-08 cap_usd=0.00 spent_usd=0.00 "
        f"skipped={','.join(roster)}"
    ]
    assert "live.model_deploy_failed" not in out
    # Saved, and not mistaken for a finished night: the next cron tries again.
    assert live._LIVE_REFRESH_STATE_PATH.exists()
    assert live.refresh_live_leaderboard(deploy_models=True, as_of=as_of)["skipped"] is False


def test_halts_and_failures_are_reported_apart(monkeypatch, capsys):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0.8")
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())
    freeze_cfg = _live_aug27()
    deepseek = _deepseek(freeze_cfg)
    twin = {**deepseek, "id": "deepseek_twin", "name": "DeepSeek twin"}
    monkeypatch.setattr(live, "live_llm_entries", lambda config=None: [deepseek, twin])
    dies = _DiesBeforeTheClose(
        "2026-08-03", RuntimeError("provider reset"), step_tokens=(2_000_000, 0),
    )
    monkeypatch.setattr(
        lb_service,
        "get_strategy",
        lambda entry: dies if entry["id"] == "deepseek_v4_pro" else _no_billable_work(),
    )

    result = live.refresh_live_leaderboard(
        deploy_models=True, as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    )
    # DeepSeek billed $0.87 and died; that spend leaves the twin nothing.
    assert [f["entry_id"] for f in result["model_failures"]] == ["deepseek_v4_pro"]
    assert result["budget_halted"] == ["deepseek_twin"]
    assert result["models_deployed"] is False
    assert result["budget"]["spent_usd"] == pytest.approx(0.87)
    out = capsys.readouterr().out
    assert "ERROR: live.model_deploy_failed entry=deepseek_v4_pro" in out
    assert (
        "ERROR: live.budget_exhausted month=2026-08 cap_usd=0.80 spent_usd=0.87 "
        "skipped=deepseek_twin"
    ) in out


def test_a_halt_after_a_success_reports_each_entry_once(monkeypatch):
    """Since #668 the success path follows the try statement, so it must run
    only in the statement's else clause: a halt that fell through to it would
    append the entry before it to model_results a second time."""
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0.4")
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())
    freeze_cfg = _live_aug27()
    deepseek = _deepseek(freeze_cfg)
    twin = {**deepseek, "id": "deepseek_twin", "name": "DeepSeek twin"}
    monkeypatch.setattr(live, "live_llm_entries", lambda config=None: [deepseek, twin])
    # One session left to trade, so DeepSeek's own projection never stops it.
    _insert_live_run(
        "lb_deepseek_v4_pro_20260801_20260826", "deepseek_v4_pro", "2026-08-26",
        [_pt("2026-08-26T19:00:00+00:00", SEED)],
        metadata={live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}, "day": "08-26"}},
    )
    monkeypatch.setattr(
        lb_service,
        "get_strategy",
        lambda entry: _ScriptedAgent(step_tokens=(1_000_000, 0))
        if entry["id"] == "deepseek_v4_pro" else _no_billable_work(),
    )

    result = live.refresh_live_leaderboard(
        deploy_models=True, as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    )
    # DeepSeek's one session ($0.435) spends the $0.40 cap; the twin's
    # pre-check halts it before it is built.
    assert [row["entry_id"] for row in result["model_results"]] == ["deepseek_v4_pro"]
    assert result["budget_halted"] == ["deepseek_twin"]
    assert result["model_failures"] == []


def test_a_halt_between_two_shared_failures_does_not_reset_the_stop(monkeypatch, capsys):
    """A halt is kept out of RosterFailures, even one that closed sessions
    first: it stops before the segment's last checkpoint and its end-of-run
    checks, so it has not shown that the shared path works. Two failures with
    the same error either side of it still stop the roster. A wrong stop leaves
    the rest unbilled until the next night; a wrong reset bills another entry
    against the same fault."""
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    entries = [{"id": f"entry_{n}", "name": f"Entry {n}"} for n in range(4)]
    monkeypatch.setattr(live, "live_llm_entries", lambda config=None: entries)
    called = []

    def deploy(entry, freeze_cfg, **kwargs):
        called.append(entry["id"])
        if entry["id"] == "entry_1":
            raise live_budget.LiveBudgetExceeded("stopped between sessions")
        raise OSError("disk full")

    monkeypatch.setattr(live, "deploy_live_model_increment", deploy)

    result = live.refresh_live_leaderboard(
        deploy_models=True, as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    )
    assert result["budget_halted"] == ["entry_1"]
    assert called == ["entry_0", "entry_1", "entry_2"]
    assert [f["entry_id"] for f in result["model_failures"]] == [
        "entry_0",
        "entry_2",
        "entry_3",
    ]
    assert (
        "ERROR: live.model_not_attempted entry=entry_3 after=entry_0,entry_2 "
        "error=OSError"
    ) in capsys.readouterr().out


def test_a_budget_read_that_fails_still_prunes_and_saves(monkeypatch, capsys):
    """The budget read runs after every model has traded and been billed. If it
    raised out of the refresh, the prune and the state save would be skipped,
    which is what #668 stopped the roster loop from doing."""
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())
    monkeypatch.setattr(lb_service, "get_strategy", lambda entry: _ScriptedAgent())

    def unreadable(*args, **kwargs):
        raise OSError("pool timeout")

    monkeypatch.setattr(live_budget, "budget_status", unreadable)

    result = live.refresh_live_leaderboard(
        deploy_models=True, as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    )
    assert result["budget"] is None
    assert result["models_deployed"] is True
    assert "pruned_runs" in result
    assert live._live_refresh_state()["budget"] is None
    assert "ERROR: live.budget_status_failed error=OSError" in capsys.readouterr().out


def test_a_clean_night_reports_the_budget_and_no_halt(monkeypatch, capsys):
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())
    monkeypatch.setattr(lb_service, "get_strategy", lambda entry: _ScriptedAgent())
    roster = [entry["id"] for entry in live.live_llm_entries(_live_aug27())]
    result = live.refresh_live_leaderboard(
        deploy_models=True, as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    )
    assert result["models_deployed"] is True
    assert result["budget_halted"] == []
    assert result["budget"]["halted"] is False
    assert result["budget"]["cap_usd"] == 50.0
    assert set(live_budget.month_ledger_rows(_AUG)) == set(roster)
    assert "live.budget_exhausted" not in capsys.readouterr().out
```

Numbers used above: DeepSeek is $0.435 per 1M input tokens and `_ScriptedAgent` bills `step_tokens` once per traded session, so `step_tokens=(1_000_000, 0)` is $0.435 a session; Aug 3–27 2026 is 19 sessions.

- [ ] **Step 2: Run them and confirm they fail.**

```bash
pytest dashboard/backend/tests/test_live_budget.py -q -p no:cacheprovider
```

Expected: `16 failed, 46 passed`. The failures are the sixteen new tests other than `test_an_entry_already_current_is_not_halted_and_writes_nothing` (a guard that already holds). Example messages:
- `AssertionError: a budget halt reached the model or the market data`: the zero-budget test, and `test_a_snapshot_that_already_covers_a_non_session_freeze_is_not_halted`, which on main reaches `get_strategy` before its own cached return.
- `TypeError: 'NoneType' object is not subscriptable`: no ledger row was written (five tests, including `test_a_new_month_starts_with_nothing_spent`).
- `Failed: DID NOT RAISE <class 'dashboard.backend.domain.leaderboard.live_budget.LiveBudgetExceeded'>`: the catch-up stop and first-deploy seeding.
- `AssertionError: assert 'ERROR: live.spend_ledger_write_failed entry=deepseek_v4_pro error=OSError' in '…'`: the ledger-write test.
- `KeyError: 'budget_halted'`: the clean-night refresh, the halt-after-a-success refresh and `test_a_halt_between_two_shared_failures_does_not_reset_the_stop` (on main the halt is recorded as a failure with its own error, which resets the streak, so the fourth entry is deployed).
- `KeyError: 'budget'`: `test_a_budget_read_that_fails_still_prunes_and_saves`.
- `test_halts_and_failures_are_reported_apart` fails on `assert ['deepseek_v4...eepseek_twin'] == ['deepseek_v4_pro']`: on main the twin's `_no_billable_work` `AssertionError` is recorded as a second failure.
- The whole-refresh halt test fails on `assert result["model_failures"] == []`, with `_no_billable_work`'s `AssertionError` text in the diff. Since #668 the roster loop catches any `Exception` per entry, so that error is recorded as a failure instead of escaping the refresh.

`pytest dashboard/backend/tests/test_live_leaderboard.py -q -p no:cacheprovider` must still pass in full after Step 1's Edits 1–3, which touch only its fixture and `_ScriptedAgent`. Measured: `77 passed`.

- [ ] **Step 3: Implement.** Nine edits to `dashboard/backend/domain/leaderboard/live.py`, in order.

**Edit 1 — `deploy_live_model_increment` no longer needs `initial_capital` in its head (it moves to `_trade_live_segment`; leaving it is an unused variable).** In `dashboard/backend/domain/leaderboard/live.py`, find:

```python
    entry_id = entry["id"]
    month_start = freeze_cfg["start_date"]
    freeze_end = freeze_cfg["end_date"]
    initial_capital = float(freeze_cfg.get("initial_capital", INITIAL_CAPITAL))

    rows = _entry_month_runs(entry_id, month_start, freeze_end)
```

Replace with:

```python
    entry_id = entry["id"]
    month_start = freeze_cfg["start_date"]
    freeze_end = freeze_cfg["end_date"]

    rows = _entry_month_runs(entry_id, month_start, freeze_end)
```

**Edit 2 — import the module (top of file, ≈:30).** In `dashboard/backend/domain/leaderboard/live.py`, find:

```python
from dashboard.backend.database import db
from dashboard.backend.domain.leaderboard.baselines import INITIAL_CAPITAL, calc_metrics
```

Replace with:

```python
from dashboard.backend.database import db
from dashboard.backend.domain.leaderboard import live_budget
from dashboard.backend.domain.leaderboard.baselines import INITIAL_CAPITAL, calc_metrics
```

**Edit 3 — return the second already-current case early, then split `deploy_live_model_increment` at the first billable line (≈:822). Everything after the inserted text is the rest of the old body, which becomes `_trade_live_segment`'s body; Edit 4 removes the one block of it this edit makes unreachable. The body uses only `resume_from`, `prior_curve`, `latest`, `force_refresh`, `allow_fallback`, `bars_memo` and the four names re-derived at its top.** In `dashboard/backend/domain/leaderboard/live.py`, find:

```python
            "llm_calls": latest.get("llm_calls"),
        }

    strategy_impl = lb_service.get_strategy(entry)
    config_changed: List[str] = []
```

Replace with:

```python
            "llm_calls": latest.get("llm_calls"),
        }

    # The other way to be current: the stored snapshot already covers
    # freeze_end by the session calendar (only a freeze_end that is not itself
    # a session gets past the check above). Returned here, before an attempt is
    # opened, for the same reason: nothing would trade, so nothing is priced,
    # written or halted. A config change on an entry this current is picked up
    # by its next increment, exactly as for the cached return above.
    if (
        resume_from is not None
        and not force_refresh
        and live_increment_bounds(
            str(resume_from.get("end_date") or ""),
            month_start=month_start,
            freeze_end=freeze_end,
        )
        is None
    ):
        return {
            "entry_id": entry_id,
            "run_id": resume_from.get("run_id"),
            "cached": True,
            "increment": False,
            "model": entry.get("model"),
            "window": {"start_date": month_start, "end_date": freeze_end},
            "segment": None,
        }

    # Everything past here can bill, so the attempt is priced into the month's
    # spend ledger whatever ends it: at every session close, and once more
    # below on success, failure or a budget halt. The pre-check runs before
    # the model is built or a bar is fetched.
    attempt = live_budget.LiveAttempt(
        entry,
        month_start=month_start,
        freeze_end=freeze_end,
        checkpoint_rows=latest_live_month_runs(month_start, freeze_end),
    )
    try:
        attempt.check_can_start()
        row = _trade_live_segment(
            entry,
            freeze_cfg,
            attempt,
            resume_from=resume_from,
            prior_curve=prior_curve,
            latest=latest,
            force_refresh=force_refresh,
            allow_fallback=allow_fallback,
            bars_memo=bars_memo,
        )
    except BaseException as exc:
        try:
            attempt.finish(exc)
        except Exception as ledger_exc:
            # The deploy's own error is the one worth raising; a ledger that
            # could not be written is reported, not allowed to replace it.
            print(
                f"ERROR: live.spend_ledger_write_failed entry={entry_id} "
                f"error={type(ledger_exc).__name__}",
                flush=True,
            )
        raise
    attempt.finish(None)
    return row


def _trade_live_segment(
    entry: Dict[str, Any],
    freeze_cfg: Dict[str, Any],
    attempt: live_budget.LiveAttempt,
    *,
    resume_from: Optional[Dict[str, Any]],
    prior_curve: List[Dict[str, Any]],
    latest: Optional[Dict[str, Any]],
    force_refresh: bool,
    allow_fallback: bool,
    bars_memo: Optional[Dict[Tuple[Any, ...], Any]],
) -> Dict[str, Any]:
    """The billable half of ``deploy_live_model_increment``.

    Builds the model, fetches bars and trades the segment as one run that
    checkpoints each session as it closes. ``attempt`` hears about every
    session close before H6 is consulted, because a session H6 refuses to
    store was billed all the same.
    """
    entry_id = entry["id"]
    month_start = freeze_cfg["start_date"]
    freeze_end = freeze_cfg["end_date"]
    initial_capital = float(freeze_cfg.get("initial_capital", INITIAL_CAPITAL))

    strategy_impl = lb_service.get_strategy(entry)
    attempt.strategy_impl = strategy_impl
    config_changed: List[str] = []
```

If the cached-return dict above this point has changed shape, anchor on `    strategy_impl = lb_service.get_strategy(entry)\n    config_changed: List[str] = []` alone: insert the same text (from `# The other way to be current` through `initial_capital = ...` and the blank line after it) above the `get_strategy` line, and add `    attempt.strategy_impl = strategy_impl` directly below it.

**Edit 4 — drop the body's cached return, now unreachable (inside what is now `_trade_live_segment`, ≈:863 on main@17c552c5).** The body reaches this branch only with the same `resume_from` the head just tested (a forced or config replay sets it to `None` first), so `bounds` is never `None` here. Left in place it would be dead code that, if ever reached, wrote an `ok` ledger row for a deploy that traded nothing. The find block starts at `if bounds is None:`, which occurs once (the head's new check reads `live_increment_bounds(...) is None`). In `dashboard/backend/domain/leaderboard/live.py`, find:

```python
        if bounds is None:
            return {
                "entry_id": entry_id,
                "run_id": resume_from.get("run_id"),
                "cached": True,
                "increment": False,
                "model": entry.get("model"),
                "window": {"start_date": month_start, "end_date": freeze_end},
                "segment": None,
            }
        segment_start, segment_end = bounds
```

Replace with:

```python
        # Never None here: deploy_live_model_increment returns the cached row
        # before it opens a priced attempt when the snapshot already covers
        # freeze_end, and this branch sees the same resume point.
        segment_start, segment_end = bounds
```

**Edit 5 — price every session close before the gap and H6 checks, and stop between sessions (the nested `on_session_close`, ≈:917).** In `dashboard/backend/domain/leaderboard/live.py`, find:

```python
    def on_session_close(day: date, segment_curve: List[Dict[str, Any]]) -> None:
        missing: List[date] = []
        while cursor["next"] < len(expected) and expected[cursor["next"]] < day:
            missing.append(expected[cursor["next"]])
            cursor["next"] += 1
        if missing:
            raise _session_gap(entry_id, missing, segment_start, segment_end)
        if cursor["next"] < len(expected) and expected[cursor["next"]] == day:
            cursor["next"] += 1
        if not _checkpoint_passes_h6(
            entry_id, entry, strategy_impl, allow_fallback=allow_fallback
        ):
            print(
                f"  live {entry_id}: {day.isoformat()} not checkpointed; the "
                f"segment's model coverage is below the H6 threshold so far"
            )
            return
        written.append(
            _write_live_checkpoint(
                entry,
                freeze_cfg,
                strategy_impl,
                day_iso=day.isoformat(),
                curve=_stitch_equity_curves(prior_curve, segment_curve),
                base=resume_from,
                lineage=lineage,
                provenance=provenance,
                supersede_later=replace_later,
            )
        )
```

Replace with:

```python
    def on_session_close(day: date, segment_curve: List[Dict[str, Any]]) -> None:
        # Priced first: a gap or an H6 refusal below can drop this session,
        # but nothing un-bills it.
        attempt.record_session()
        missing: List[date] = []
        while cursor["next"] < len(expected) and expected[cursor["next"]] < day:
            missing.append(expected[cursor["next"]])
            cursor["next"] += 1
        if missing:
            raise _session_gap(entry_id, missing, segment_start, segment_end)
        if cursor["next"] < len(expected) and expected[cursor["next"]] == day:
            cursor["next"] += 1
        if not _checkpoint_passes_h6(
            entry_id, entry, strategy_impl, allow_fallback=allow_fallback
        ):
            print(
                f"  live {entry_id}: {day.isoformat()} not checkpointed; the "
                f"segment's model coverage is below the H6 threshold so far"
            )
        else:
            written.append(
                _write_live_checkpoint(
                    entry,
                    freeze_cfg,
                    strategy_impl,
                    day_iso=day.isoformat(),
                    curve=_stitch_equity_curves(prior_curve, segment_curve),
                    base=resume_from,
                    lineage=lineage,
                    provenance=provenance,
                    supersede_later=replace_later,
                )
            )
        # Stop before a session this attempt's own average says would take the
        # month past its cap. The session just traded is already stored, so a
        # stop never throws away work that was paid for; after the segment's
        # last session there is nothing left to stop.
        if cursor["next"] < len(expected):
            attempt.stop_if_next_session_would_cross()
```

**Edit 6 — `refresh_live_leaderboard`: a halted list (≈:1099 on main@011d6411).** In `dashboard/backend/domain/leaderboard/live.py`, find:

```python
        bars_memo: Dict[Tuple[Any, ...], Any] = {}
        for entry in live_llm_entries(freeze_cfg):
```

Replace with:

```python
        bars_memo: Dict[Tuple[Any, ...], Any] = {}
        halted: List[str] = []
        for entry in live_llm_entries(freeze_cfg):
```

**Edit 7 — catch the halt ahead of the generic handler (≈:1111 on main@011d6411).** In `dashboard/backend/domain/leaderboard/live.py`, find:

```python
                    bars_memo=bars_memo,
                )
            except Exception as exc:
```

Replace with:

```python
                    bars_memo=bars_memo,
                )
            except live_budget.LiveBudgetExceeded:
                # Not a failure: the month's model budget is spent, or the next
                # session would overspend it. The entry keeps the checkpoints it
                # stored and is tried again next night. Once the month has
                # reached the cap, the pre-check halts it there before any bar
                # fetch; until then it trades again and is stopped when its next
                # session would cross the cap. Kept out of ``roster``: a halt
                # stops before the segment's last checkpoint and its end-of-run
                # checks, so it neither extends nor resets the shared-fault
                # streak.
                halted.append(entry_id)
            except Exception as exc:
```

This puts the new clause directly ahead of the generic `except Exception`. `LiveBudgetExceeded` is a `RuntimeError`, so the order is what keeps a halt out of `model_failures`. The clause has no `continue`, because Edit 8 moves the success path into the `try` statement's `else`.

A halt is not passed to `RosterFailures`, so it neither extends nor resets the shared-fault streak. That holds for both kinds of halt:
- A pre-check halt runs none of the shared path. Once one entry halts there, every later entry halts there too, at zero cost.
- A halt between sessions (`stop_if_next_session_would_cross`) has closed sessions. But it stops before the segment's last checkpoint and its end-of-run checks (the session-gap check, the segment's H6 check and the stored-through check), so it has not shown that the shared path works, which is what `roster.traded` assumes.

The cost runs one way. If two failures with the same error sit on either side of a halt, the roster stops, and the entries after it go unbilled until the next night. Resetting the streak at the halt would instead bill another entry against the same fault. `test_a_halt_between_two_shared_failures_does_not_reset_the_stop` pins this.

**Edit 8 — run the success path only when the deploy returned (≈:1126 on main@011d6411).** In `dashboard/backend/domain/leaderboard/live.py`, find:

```python
                roster.failed(entry_id, exc)
                continue
            roster.traded(row)
            successes.append(row)
```

Replace with:

```python
                roster.failed(entry_id, exc)
            else:
                roster.traded(row)
                successes.append(row)
```

Since #668 the success path (`roster.traded(row)`, `successes.append(row)`) runs after the `try` statement, guarded only by a `continue` in each `except` clause. In the `else` clause, no handler can reach it, so a clause added later cannot fall through into it. Without this edit, the halt clause would fall through. After a success, that appends the previous entry's row a second time. Before any entry has traded, it raises `UnboundLocalError` out of the loop and skips the prune and the state save. `test_a_halt_after_a_success_reports_each_entry_once` pins the first case. The second is pinned by `test_a_spent_month_halts_the_roster_with_one_log_line` (the first entry halts) and `test_halts_and_failures_are_reported_apart` (a failure, then a halt).

**Edit 9 — report the budget once per refresh, without letting its read stop the refresh (≈:1130 on main@011d6411).** In `dashboard/backend/domain/leaderboard/live.py`, find:

```python
        result["models_deployed"] = not roster.failures
        result["model_results"] = successes
        result["model_failures"] = roster.failures
```

Replace with:

```python
        try:
            budget: Optional[Dict[str, Any]] = live_budget.budget_status(
                freeze_cfg["start_date"],
                checkpoint_rows=latest_live_month_runs(
                    freeze_cfg["start_date"], freeze_cfg["end_date"]
                ),
            )
        except Exception as exc:
            # Two database reads after every model has traded and been billed.
            # A failed read must not skip the prune and the state save below.
            print(
                f"ERROR: live.budget_status_failed error={type(exc).__name__}",
                flush=True,
            )
            budget = None
        if halted:
            # One line for the whole roster: the state file is on an ephemeral
            # disk, so the log is the record an operator actually reads.
            spent = f"{budget['spent_usd']:.2f}" if budget else "unread"
            print(
                f"ERROR: live.budget_exhausted month={freeze_cfg['start_date'][:7]} "
                f"cap_usd={live_budget.monthly_budget_usd():.2f} spent_usd={spent} "
                f"skipped={','.join(halted)}",
                flush=True,
            )
        result["models_deployed"] = not roster.failures and not halted
        result["model_results"] = successes
        result["model_failures"] = roster.failures
        result["budget_halted"] = halted
        result["budget"] = budget
```

The budget read is guarded because it runs after the roster loop. Unguarded, a database error there (a Neon blip, a `PoolTimeout`) would raise out of `refresh_live_leaderboard` after every model was billed and checkpointed, skipping the prune and the state save and losing `model_results`. #668 removed exactly that failure from the loop above it. `test_a_budget_read_that_fails_still_prunes_and_saves` pins it. The month and the cap in the `budget_exhausted` line need no database (`monthly_budget_usd()` reads the environment), so only the spent figure can read `unread`.

No unmerged phase that merges before this one changes these lines.

- [ ] **Step 4: Run them and confirm they pass.**

```bash
pytest dashboard/backend/tests/test_live_budget.py -q -p no:cacheprovider
pytest dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/test_leaderboard_season.py dashboard/backend/tests/test_live_trade_stats.py dashboard/backend/tests/test_architecture_boundaries.py -q -p no:cacheprovider
```

Expected: `62 passed`, then the neighbours green (measured `156 passed` for the four files together).

- [ ] **Step 5: Commit.**

```bash
git add dashboard/backend/domain/leaderboard/live.py dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/test_live_budget.py
git commit -m "feat(live): enforce the monthly budget on live deploys"
```

### Task 3.4: The public GET shows `budget_halted` and each model's `last_attempt`

**Files:**
- Modify: `dashboard/backend/domain/leaderboard/live.py`
- Test: `dashboard/backend/tests/test_live_budget.py`

**Interfaces:**
- Consumes: `live_budget.last_attempts`, `live_budget.OUTCOME_BUDGET_HALTED`.
- Produces: `live_status["budget_halted"]: bool` and per-entry `last_attempt` (whitelisted `{on, outcome, kind}`, `None` for non-model rows). It adds no new dollar figure and no error text; the per-entry `est_cost_usd` predates this plan (read off the checkpoint row by `_entry_from_strategy`, out of scope here). One extra `get_runs_by_session` read per GET, on the same database the GET already reads.

- [ ] **Step 1: Write the failing tests.** Append to the end of `dashboard/backend/tests/test_live_budget.py` (two blank lines before the section comment):

```python
# --- what the public GET shows -----------------------------------------------------


def _get_aug27():
    return live.get_live_leaderboard(as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET))


@pytest.fixture
def _read_only(monkeypatch):
    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("the live GET path reached the compute path")

    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", _must_not_run)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", _must_not_run)


def test_get_live_gains_only_a_halt_flag_and_last_attempts(monkeypatch, _read_only):
    """The ledger's whole public footprint is ``live_status.budget_halted`` and
    each model's ``last_attempt``. The per-entry ``est_cost_usd`` the GET
    already published from the checkpoint row predates the ledger and is left
    as it was; the cap and the month-to-date spend never reach the payload."""
    monkeypatch.setenv(live_budget.BUDGET_ENV, "48.7654")
    _insert_live_run(
        "lb_deepseek_v4_pro_20260801_20260827", "deepseek_v4_pro", _FREEZE,
        [_pt(f"{_FREEZE}T19:00:00+00:00", SEED)],
        metadata={live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}, "day": "08-27"}},
        est_cost_usd=12.3456,
    )
    before = _get_aug27()
    _insert_ledger("deepseek_v4_pro", 48.7654, last_attempt={
        "on": _FREEZE, "outcome": "budget_halted", "kind": "budget",
        "error": "sk-live-SECRET", "spent_usd": 48.7654,
    })
    after = _get_aug27()

    # live_status gains one boolean, and the ledger moves nothing else in it.
    assert before["live_status"]["budget_halted"] is False
    assert after["live_status"]["budget_halted"] is True
    assert set(after["live_status"]) == set(before["live_status"])
    assert {
        key for key, value in after["live_status"].items()
        if value != before["live_status"][key]
    } == {"budget_halted"}

    # Each entry gains last_attempt, exactly {on, outcome, kind}; nothing else moves.
    before_by_id = {entry["entry_id"]: entry for entry in before["entries"]}
    assert {entry["entry_id"] for entry in after["entries"]} == set(before_by_id)
    for entry in after["entries"]:
        old = before_by_id[entry["entry_id"]]
        assert set(entry) == set(old)
        assert {key for key in entry if entry[key] != old[key]} <= {"last_attempt"}
    deepseek = next(e for e in after["entries"] if e["entry_id"] == "deepseek_v4_pro")
    assert deepseek["last_attempt"] == {
        "on": _FREEZE, "outcome": "budget_halted", "kind": "budget",
    }
    assert all(e["last_attempt"] is None for e in after["entries"] if not e["is_model"])

    # The pre-existing per-entry figure is the checkpoint row's, untouched...
    assert deepseek["est_cost_usd"] == pytest.approx(12.3456)
    # ...and neither the cap nor the month-to-date spend reaches the payload.
    text = json.dumps(after)
    assert "48.7654" not in text
    assert "SECRET" not in text
    for key in ("cap_usd", "spent_usd", "remaining_usd"):
        assert key not in text


def test_get_live_reports_a_failed_night_without_halting_the_board(_read_only):
    _insert_ledger("deepseek_v4_pro", 0.5, last_attempt={
        "on": _FREEZE, "outcome": "failed", "kind": "market_data_gap",
    })
    payload = _get_aug27()
    assert payload["live_status"]["budget_halted"] is False
    deepseek = next(e for e in payload["entries"] if e["entry_id"] == "deepseek_v4_pro")
    assert deepseek["last_attempt"]["outcome"] == "failed"
    assert deepseek["last_attempt"]["kind"] == "market_data_gap"


def test_get_live_ignores_last_months_ledger_and_an_empty_one(_read_only):
    payload = _get_aug27()
    assert payload["live_status"]["budget_halted"] is False
    assert all(entry["last_attempt"] is None for entry in payload["entries"])
    _insert_ledger("deepseek_v4_pro", 60.0, month="2026-07-01", last_attempt={
        "on": "2026-07-31", "outcome": "budget_halted", "kind": "budget",
    })
    assert _get_aug27()["live_status"]["budget_halted"] is False
```

- [ ] **Step 2: Run them and confirm they fail.**

```bash
pytest dashboard/backend/tests/test_live_budget.py -q -p no:cacheprovider
```

Expected: `3 failed, 62 passed`, each with `KeyError: 'budget_halted'`.

- [ ] **Step 3: Implement.** Three edits to `get_live_leaderboard` in `dashboard/backend/domain/leaderboard/live.py`.

**Edit 1 — read the month's attempts once (≈:1412).** In `dashboard/backend/domain/leaderboard/live.py`, find:

```python
    runs_by_entry = (
        latest_live_month_runs(freeze_start, freeze_end) if freeze_cfg else {}
    )
```

Replace with:

```python
    runs_by_entry = (
        latest_live_month_runs(freeze_start, freeze_end) if freeze_cfg else {}
    )
    # Each model's last billable attempt this month, from the spend ledger:
    # its date, outcome and a coarse kind only. It adds no dollar figure and no
    # error message: this GET is public and the owner pays for the calls.
    attempts = live_budget.last_attempts(freeze_start) if freeze_cfg else {}
```

**Edit 2 — attach them per entry (inside the `for strategy in strategies:` loop, ≈:1466).** In `dashboard/backend/domain/leaderboard/live.py`, find:

```python
        (ranked if entry["status"] == "frozen" else pending).append(entry)
```

Replace with:

```python
        entry["last_attempt"] = (
            attempts.get(entry["entry_id"]) if entry.get("is_model") else None
        )
        (ranked if entry["status"] == "frozen" else pending).append(entry)
```

If P6 (`2026-10-14-live-board-fixes-p6-copy.md`, #586) has changed this line, put the `entry["last_attempt"] = …` statement immediately after the `entry = _entry_from_strategy(...)` call in the same loop.

**Edit 3 — the board-level flag (the `live_status` dict, ≈:1509).** In `dashboard/backend/domain/leaderboard/live.py`, find:

```python
        "has_prints": bool(ranked),
        "freeze_start": freeze_start,
        "freeze_end": freeze_end,
    }
```

Replace with:

```python
        "has_prints": bool(ranked),
        "freeze_start": freeze_start,
        "freeze_end": freeze_end,
        # Some model's last attempt stopped at the monthly budget. A flag
        # only: how much was spent stays in the refresh result and the log.
        "budget_halted": any(
            (entry.get("last_attempt") or {}).get("outcome")
            == live_budget.OUTCOME_BUDGET_HALTED
            for entry in entries
        ),
    }
```

`entries` is the ranked-plus-pending list built just above `live_status`.

- [ ] **Step 4: Run them and confirm they pass.**

```bash
pytest dashboard/backend/tests/test_live_budget.py -q -p no:cacheprovider
pytest dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/test_leaderboard_season.py -q -p no:cacheprovider
```

Expected: `65 passed`, then the neighbours green (measured `98 passed`).

- [ ] **Step 5: Commit.**

```bash
git add dashboard/backend/domain/leaderboard/live.py dashboard/backend/tests/test_live_budget.py
git commit -m "feat(live): publish budget halts and last attempts on the live board"
```

### Task 3.5: The live board renders the halt line and "Last nightly run"

**Files:**
- Modify: `dashboard/frontend/js/live-leaderboard.js`
- Modify: `dashboard/frontend/app.html` (cache-buster only)
- Create: `dashboard/backend/tests/test_frontend_live_budget.py`

**Interfaces:**
- Consumes: `live_status.budget_halted`, `entry.last_attempt` (Task 3.4); existing `liveAxisTick`, `escapeHtml`, `#livePhaseBanner`, `#liveLeaderboardTableBody`, `#liveSelectedDetail`.
- Produces: `LIVE_ATTEMPT_KIND_COPY`, `liveLastAttemptCopy(entry)`, `liveLastAttemptBadge(entry)`, `liveBudgetHaltLine(status)`; a second `team-badge` span (`Paused` / `Last run failed`) in the table's name cell; a "Last nightly run" `team-detail-row` in the detail panel; the halt sentence appended to `#livePhaseBanner`. Reuses existing CSS classes only: `styles.css` and `app.js` are not touched.

The tests live in a new file rather than `test_frontend_live_leaderboard.py`, which P6 (`2026-10-14-live-board-fixes-p6-copy.md`; Decision 8 in the index plan) rewrites. They run the whole of `live-leaderboard.js` under node against a fake DOM, so they test behaviour, not source text.

**Once P6 lands.** P6 Task 6.5 makes the banner read `payload.entries` instead of `live_status.models_cached`, so from then on `_banner_after` below must send one printed model row; P6's Edits A and B add it (`_PRINTED_MODEL`). Merge order puts P6 after this phase, so normally there is nothing to do here, and the `_HEALTHY` comment line and the first two lines of `_banner_after` must stay exactly as written below for those anchors to match. If P6 has already merged when you execute this task, apply P6 Task 6.5's Edits A and B to `test_frontend_live_budget.py` right after Step 1.

- [ ] **Step 1: Write the failing tests.** Create `dashboard/backend/tests/test_frontend_live_budget.py` with exactly:

```python
"""The Live Leaderboard's budget-halt banner line and "Last nightly run" status.

Runs the whole of js/live-leaderboard.js under node against a fake DOM: the
file is top-level declarations plus two ``window.`` assignments, so it loads
without a browser. The ledger adds only ``live_status.budget_halted`` and
per-entry ``last_attempt = {on, outcome, kind}`` to the payload; neither
carries a dollar figure or error text.
"""

import json
import re
import shutil
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import run_node

_LIVE_JS = (
    Path(__file__).resolve().parents[2] / "frontend" / "js" / "live-leaderboard.js"
).read_text(encoding="utf-8")

_needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

# escapeHtml lives in app.js; this stub escapes the same characters.
_DOM_PRELUDE = """
globalThis.window = {};
const elements = {};
globalThis.document = {
  getElementById(id) {
    if (!elements[id]) {
      elements[id] = { textContent: '', innerHTML: '', hidden: true, classList: { toggle() {} } };
    }
    return elements[id];
  },
  querySelector() { return null; },
  querySelectorAll() { return []; },
  addEventListener() {},
};
globalThis.escapeHtml = (value) => String(value)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
"""

_HALT_LINE = (
    "Model runs are paused: the monthly model budget is used up. "
    "Baselines keep updating."
)

# A board with prints, current, and models cached: the existing banner is hidden.
_HEALTHY = {
    "has_prints": True,
    "snapshot_stale": False,
    "models_cached": 1,
    "frozen_through": "2026-08-27",
}


def _run_live_js(tail: str):
    return run_node(_DOM_PRELUDE + _LIVE_JS + "\n" + tail)


def _attempt(outcome, kind=None, on="2026-08-27"):
    return {
        "entry_id": "deepseek_v4_pro",
        "is_model": True,
        "last_attempt": {"on": on, "outcome": outcome, "kind": kind},
    }


def _banner_after(status):
    payload = {"window": {"label": "2026-08-01 — 2026-08-31"}, "entries": [], "live_status": status}
    return _run_live_js(
        f"updateLiveHeader({json.dumps(payload)});"
        "const b = document.getElementById('livePhaseBanner');"
        "process.stdout.write(JSON.stringify({hidden: b.hidden, text: b.textContent}));"
    )


@_needs_node
def test_last_nightly_run_copy_names_the_day_outcome_and_reason():
    entries = [
        _attempt("ok"),
        _attempt("failed", "market_data_gap"),
        _attempt("budget_halted", "budget"),
        _attempt("failed", "timeout"),
        _attempt("failed", "not-a-kind"),
        {"entry_id": "spy_index", "last_attempt": None},
        {"entry_id": "old_payload"},
    ]
    out = _run_live_js(
        f"const entries = {json.dumps(entries)};"
        "process.stdout.write(JSON.stringify({"
        "copy: entries.map(liveLastAttemptCopy), badge: entries.map(liveLastAttemptBadge)}));"
    )
    assert out["copy"] == [
        "Aug 27 · completed",
        "Aug 27 · failed: market data was missing for a session",
        "Aug 27 · paused: monthly model budget used up",
        "Aug 27 · failed: a request timed out",
        "Aug 27 · failed: an internal error",
        "",
        "",
    ]
    assert out["badge"] == [
        "", "Last run failed", "Paused", "Last run failed", "Last run failed", "", "",
    ]


@_needs_node
def test_a_budget_halt_shows_a_banner_line_on_a_healthy_board():
    assert _banner_after({**_HEALTHY, "budget_halted": True}) == {
        "hidden": False,
        "text": _HALT_LINE,
    }


@_needs_node
def test_the_budget_line_joins_an_existing_banner_message():
    banner = _banner_after({
        **_HEALTHY,
        "snapshot_stale": True,
        "snapshot_end": "2026-08-26",
        "budget_halted": True,
    })
    assert banner["hidden"] is False
    assert banner["text"].endswith(" " + _HALT_LINE)
    # The stale-snapshot message is kept, not replaced.
    assert banner["text"][: -len(_HALT_LINE)].strip()


@pytest.mark.parametrize("flag", [False, None, "true", 1])
@_needs_node
def test_no_budget_line_without_a_real_true(flag):
    status = dict(_HEALTHY)
    if flag is not None:
        status["budget_halted"] = flag
    assert _banner_after(status) == {"hidden": True, "text": ""}


@_needs_node
def test_the_table_and_detail_show_a_failed_night():
    entry = {
        **_attempt("failed", "network"),
        "model": "DeepSeek V4 Pro",
        "team_badge": "Model",
        "status": "frozen",
        "portfolio_value": 100500,
        "cumulative_return": 0.005,
        "snapshot_end": "2026-08-26",
    }
    out = _run_live_js(
        f"livePayload = {{entries: [{json.dumps(entry)}]}};"
        "populateLiveTable();"
        f"renderLiveDetail({json.dumps(entry)});"
        "process.stdout.write(JSON.stringify({"
        "row: document.getElementById('liveLeaderboardTableBody').innerHTML,"
        "detail: document.getElementById('liveSelectedDetail').innerHTML}));"
    )
    assert '<span class="team-badge">Last run failed</span>' in out["row"]
    assert re.search(
        r'<span class="team-detail-label">Last nightly run</span>'
        r'<span class="team-detail-value">Aug 27 · failed: a connection failed</span>',
        out["detail"],
    )


@_needs_node
def test_a_model_with_no_attempt_this_month_shows_no_run_row_or_badge():
    entry = {
        "entry_id": "deepseek_v4_pro",
        "model": "DeepSeek V4 Pro",
        "is_model": True,
        "status": "pending",
        "last_attempt": None,
    }
    out = _run_live_js(
        f"livePayload = {{entries: [{json.dumps(entry)}]}};"
        "populateLiveTable();"
        f"renderLiveDetail({json.dumps(entry)});"
        "process.stdout.write(JSON.stringify({"
        "row: document.getElementById('liveLeaderboardTableBody').innerHTML,"
        "detail: document.getElementById('liveSelectedDetail').innerHTML}));"
    )
    assert "Last nightly run" not in out["detail"]
    assert "Last run failed" not in out["row"] and "Paused" not in out["row"]


def test_the_board_never_renders_spend_figures():
    for key in ("est_cost_usd", "spent_usd", "cap_usd", "remaining_usd"):
        assert key not in _LIVE_JS
```

- [ ] **Step 2: Run them and confirm they fail.**

```bash
pytest dashboard/backend/tests/test_frontend_live_budget.py -q -p no:cacheprovider
```

Expected: `4 failed, 6 passed`. The failures: `test_last_nightly_run_copy_names_the_day_outcome_and_reason` (node exits with `ReferenceError: liveLastAttemptCopy is not defined`), `test_a_budget_halt_shows_a_banner_line_on_a_healthy_board` (`{'hidden': True, 'text': ''}` where a visible banner was expected), `test_the_budget_line_joins_an_existing_banner_message` (`assert False`: the banner holds only the stale-snapshot sentence, `'Showing the snapshot through 2026-08-26. The session through 2026-08-27 appears after the next nightly refresh.'`, which does not end with the halt line) and `test_the_table_and_detail_show_a_failed_night` (no `Last run failed` badge). The six that pass are guards for what must stay hidden.

- [ ] **Step 3: Implement.** Six edits to `dashboard/frontend/js/live-leaderboard.js`, then the cache-buster.

**Edit 1 — the helpers, directly above `isLiveBoardVisible` (≈:121).** In `dashboard/frontend/js/live-leaderboard.js`, find:

```js
function isLiveBoardVisible() {
```

Replace with:

```js
// A model's last nightly attempt, as the payload's `last_attempt` gives it:
// {on, outcome, kind} and nothing else, so no dollar figure and no error message;
// `kind` is a coarse class and unknown ones read as an internal error.
const LIVE_ATTEMPT_KIND_COPY = {
  budget: 'monthly model budget used up',
  model_coverage: 'the model did not drive enough of its decisions',
  market_data_gap: 'market data was missing for a session',
  // Not "the model provider": on the live path a provider timeout becomes a
  // rule-based step (reported as model_coverage); what arrives here is a
  // store or other client timeout.
  timeout: 'a request timed out',
  network: 'a connection failed',
  interrupted: 'the run was interrupted',
  error: 'an internal error',
};

function liveLastAttemptCopy(entry) {
  const attempt = entry && entry.last_attempt;
  if (!attempt || !attempt.on) return '';
  const day = liveAxisTick(attempt.on) || String(attempt.on);
  if (attempt.outcome === 'ok') return `${day} · completed`;
  const why = LIVE_ATTEMPT_KIND_COPY[attempt.kind] || LIVE_ATTEMPT_KIND_COPY.error;
  if (attempt.outcome === 'budget_halted') return `${day} · paused: ${why}`;
  if (attempt.outcome === 'failed') return `${day} · failed: ${why}`;
  return '';
}

function liveLastAttemptBadge(entry) {
  const outcome = entry && entry.last_attempt && entry.last_attempt.outcome;
  if (outcome === 'budget_halted') return 'Paused';
  if (outcome === 'failed') return 'Last run failed';
  return '';
}

// Only a real `true`: a missing or malformed flag shows nothing rather than
// telling visitors the models stopped. No promise of when models resume: a cap
// of 0 halts the new month too, and a cap raised mid-month resumes them the
// next night.
function liveBudgetHaltLine(status) {
  if (!status || status.budget_halted !== true) return '';
  return 'Model runs are paused: the monthly model budget is used up. Baselines keep updating.';
}

function isLiveBoardVisible() {
```

**Edit 2 — `updateLiveHeader`: append the halt line after the existing banner branches, above the zone caption (≈:219).** In `dashboard/frontend/js/live-leaderboard.js`, find:

```js
  const zone = document.getElementById('liveZoneCaption');
```

Replace with:

```js
  const haltLine = liveBudgetHaltLine(status);
  if (banner && haltLine) {
    banner.hidden = false;
    banner.textContent = banner.textContent ? `${banner.textContent} ${haltLine}` : haltLine;
  }

  const zone = document.getElementById('liveZoneCaption');
```

This runs after the existing `if (banner) { … }` block whatever that block becomes (P6 edits its messages), so a halt never replaces the stale-snapshot or no-prints message; it joins it.

**Edit 3 — `populateLiveTable`: compute the badge (≈:312).** In `dashboard/frontend/js/live-leaderboard.js`, find:

```js
    const dash = (html) => (printed ? html : '—');
```

Replace with:

```js
    const dash = (html) => (printed ? html : '—');
    const attemptBadge = liveLastAttemptBadge(entry);
```

**Edit 4 — render it after the type badge.** In `dashboard/frontend/js/live-leaderboard.js`, find:

```js
            <span class="team-badge">${escapeHtml(liveFormatBadge(entry.team_badge))}</span>
```

Replace with:

```js
            <span class="team-badge">${escapeHtml(liveFormatBadge(entry.team_badge))}</span>
            ${attemptBadge ? `<span class="team-badge">${escapeHtml(attemptBadge)}</span>` : ''}
```

**Edit 5 — `renderLiveDetail`: compute the copy (≈:347).** In `dashboard/frontend/js/live-leaderboard.js`, find:

```js
  const hasPrints = !liveEntryIsPending(entry);
```

Replace with:

```js
  const hasPrints = !liveEntryIsPending(entry);
  const lastRun = liveLastAttemptCopy(entry);
```

**Edit 6 — add the detail row at the end of the panel, after whatever the last row is (P6 may reword the `Through` row).** In `dashboard/frontend/js/live-leaderboard.js`, find:

```js
  `;
}

function livePickerGroups(entries) {
```

Replace with:

```js
    ${lastRun ? `<div class="team-detail-row"><span class="team-detail-label">Last nightly run</span><span class="team-detail-value">${escapeHtml(lastRun)}</span></div>` : ''}
  `;
}

function livePickerGroups(entries) {
```

**Edit 7 — cache-buster.** In `dashboard/frontend/app.html`, find the one `<script src="js/live-leaderboard.js?v=N" defer></script>` tag and raise `N` by one from whatever main has when you execute this (on main@011d6411 it is `?v=5`, so it becomes `?v=6`). No test pins the exact number (`test_live_script_is_cache_busted` only requires one). Do not touch `js/leaderboard.js?v=`, `styles.css?v=` or `app.js?v=`.

- [ ] **Step 4: Run them and confirm they pass.**

```bash
node --check dashboard/frontend/js/live-leaderboard.js
pytest dashboard/backend/tests/test_frontend_live_budget.py -q -p no:cacheprovider
pytest dashboard/backend/tests/test_frontend_live_leaderboard.py dashboard/backend/tests/test_frontend_live_trading_board.py dashboard/backend/tests/test_frontend_fast_boot.py -q -p no:cacheprovider
```

Expected: `node --check` prints nothing and exits 0; `10 passed`; neighbours green (measured `56 passed`). The node tests are skipped (not failed) on a machine without `node`; this WSL has node v24, so a skip here means something is wrong.

- [ ] **Step 5: Commit.**

```bash
git add dashboard/frontend/js/live-leaderboard.js dashboard/frontend/app.html dashboard/backend/tests/test_frontend_live_budget.py
git commit -m "feat(live): render budget halts and the last nightly run"
```

### Task 3.6: Document `LEADERBOARD_MONTHLY_BUDGET_USD` in the repo CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (repo root; developer doc, part of this PR)

**Interfaces:**
- Consumes: the behaviour built in Tasks 3.1–3.5.
- Produces: one env bullet, placed directly after the `LEADERBOARD_DAILY_AUTO_DEPLOY` bullet. P7 (`2026-10-14-live-board-fixes-p7-records.md`; Decision 9 in the index plan) rewrites the neighbouring live-board passages; it must edit this bullet, not add a second one.

- [ ] **Step 1: Write the failing test.** No code test: this is prose. The check is a grep that must find nothing before the edit:

```bash
grep -c "LEADERBOARD_MONTHLY_BUDGET_USD" CLAUDE.md
```

- [ ] **Step 2: Run it and confirm it fails.** Expected output: `0` (grep exits 1).

- [ ] **Step 3: Implement.** In `CLAUDE.md`, find this text (the start of the `NEWS_FEED_FROZEN` bullet, which follows the `LEADERBOARD_DAILY_AUTO_DEPLOY` bullet on main@011d6411, ≈:90):

```markdown
- `NEWS_FEED_FROZEN` (optional, **unset = frozen** while FinSearch's feed timers are paused, since 2026-10-11): read per request by
```

Replace it with the new bullet, a newline, and the same text:

```markdown
- `LEADERBOARD_MONTHLY_BUDGET_USD` (optional, default **50**, range 0–10000; **`0` blocks every billable live deploy — it is not "off"**): the Live Leaderboard's monthly model-spend hard stop (`domain/leaderboard/live_budget.py`, read per call by `monthly_budget_usd()`, never at import). Blank is the default, silently; junk, `nan`/`inf`, negative or over 10000 prints one `WARNING` per distinct value and uses 50 — it never raises, since every live deploy reads it. To lift the cap set a large number: a typo must not turn a hard stop into "unlimited". **Why a ledger rather than the checkpoint rows:** the live calls go through `COMMONSTACK_API_KEY` and bypass the Credits layer, and checkpoint rows undercount — a session H6 refuses to store, a run that dies before its next session close, and every row a forced or config replay deletes were all billed and leave nothing behind. So every billable attempt in `deploy_live_model_increment` is priced into one `agent_runs` row per (month, entry): session `leaderboard-live-spend`, mode `leaderboard_spend`, run id `live_spend_<YYYYMM>_<entry_id>`, cumulative `est_cost_usd`/tokens/`llm_calls` across successes, failures, H6 refusals and replays, plus `metadata.last_attempt = {on: <freeze_end>, outcome: ok|failed|budget_halted, kind}` (`kind` is a coarse class, never the error text). Cost is provider-reported tokens × the `infrastructure/llm/pricing.py` snapshot — an estimate, since CommonStack has no usage endpoint — and `LLMAgentStrategy` publishes its tokens after every decision step so a run that dies mid-session is still priced. Month-to-date per entry is `max(ledger, latest checkpoint row)`, so the first deploy after rollout starts from what the checkpoints already recorded. Two checks: before the model is built or a bar fetched (`spent >= cap` → `LiveBudgetExceeded`, a `RuntimeError`), and at each session close, where the ledger is upserted **before** H6 and the run stops when this attempt's average session cost would cross the cap with a session still to come — after storing the session it just paid for. The last session of a segment is never stopped by its own projection, so the overshoot is about one session per entry per month, not zero. `refresh_live_leaderboard` catches the halt ahead of the generic handler: halted entries go to `result["budget_halted"]`, not `model_failures`; `result["budget"]` carries cap/spent/remaining (operator-facing: the state file and the log, never the GET); exactly one flushed `ERROR: live.budget_exhausted month=YYYY-MM cap_usd=… spent_usd=… skipped=a,b` line is printed; a budget read that fails after the roster prints `ERROR: live.budget_status_failed error=<type>`, leaves `result["budget"]` `None` and `spent_usd=unread`, and does not stop the prune and the state save; and `models_deployed` stays false so the next cron retries, halting again at zero cost until the month rolls or the cap is raised. The public GET gains only `live_status.budget_halted` and per-entry `last_attempt`, which `js/live-leaderboard.js` renders as a banner line and a "Last nightly run" row; it adds no new dollar figure (the per-entry `est_cost_usd` it already published from checkpoint rows predates the ledger). The ledger is invisible to the board, `_find_cached_run`, `/api/backtest/runs`, v2 and the admin operator-cost roll-up (no owner); the public `/runs?mode=leaderboard_spend` does list its rows, but only through the `RunMetadata` allow-list, which carries the call count and no dollar, token or `metadata` field. It **survives `clear_live_session_runs` / `--clear` on purpose**: that money was spent. It is durable only because `AGENT_RUNS_DATABASE_URL` is set in prod; on ephemeral SQLite a redeploy zeroes it. It covers the live board only — contest/daily `deploy_model_run` is not budgeted. `tests/conftest.py` strips it.
- `NEWS_FEED_FROZEN` (optional, **unset = frozen** while FinSearch's feed timers are paused, since 2026-10-11): read per request by
```

If P7 has already moved `LEADERBOARD_DAILY_AUTO_DEPLOY`, put the bullet directly after it wherever it now is; the anchor above is only the line that follows it today.

- [ ] **Step 4: Run it and confirm it passes.**

```bash
grep -c "LEADERBOARD_MONTHLY_BUDGET_USD" CLAUDE.md
grep -n "live.budget_exhausted\|leaderboard-live-spend" CLAUDE.md
```

Expected: `1`, then one line containing both strings. Re-read the bullet against `live_budget.py` and the `live.py` edits: every name it cites (`monthly_budget_usd`, `LiveBudgetExceeded`, `budget_halted`, `last_attempt`, `clear_live_session_runs`) must exist.

- [ ] **Step 5: Commit.**

```bash
git add CLAUDE.md
git commit -m "docs: document LEADERBOARD_MONTHLY_BUDGET_USD"
```

### Task 3.7: Open the PR

**Files:** none changed.

**Interfaces:**
- Consumes: the six commits above (Tasks 3.1–3.6) on `feat/live-attempt-ledger`.
- Produces: one PR against `main`.

- [ ] **Step 1: Run the full suite.**

```bash
pytest dashboard/backend/tests -q -p no:cacheprovider
```

Expected: green apart from two files that fail locally for environmental reasons, and only locally (CI installs the pinned requirements): `test_report_pdf.py`, because this environment has no `reportlab`, and five cases of `test_robinhood_mcp_transport.py`, because its installed `mcp` (1.28.1) is older than the `mcp==1.30.0` pin. Both fail the same way on main without this phase. To skip the first, deselect it:

```bash
pytest dashboard/backend/tests -q -p no:cacheprovider --deselect dashboard/backend/tests/test_report_pdf.py
```

Do not add `-x` (one failure would hide the rest) or `--timeout` (`pytest-timeout` is not installed here; pytest rejects the flag). It takes about five to seven minutes. Measured on a copy of main@011d6411 with Tasks 3.1–3.6 applied: `5 failed, 7096 passed, 185 skipped, 9 deselected, 25 xfailed`, the five being the `test_robinhood_mcp_transport.py` cases above. On a later main, add the tests it gained. Any other failure is a regression in this branch.

- [ ] **Step 2: Confirm the branch and that no PR already exists for it.**

```bash
git branch --show-current
gh pr list --head feat/live-attempt-ledger --state all
```

Expected: `feat/live-attempt-ledger` (the branch cut before Task 3.1), and no rows. If the branch is wrong, a sibling session switched the shared checkout, or the commits landed on a pre-made worktree's own branch: stop, and move the six commits onto `feat/live-attempt-ledger` before pushing.

- [ ] **Step 3: Push and open the PR.**

```bash
git push -u origin feat/live-attempt-ledger
gh pr create --base main --head feat/live-attempt-ledger \
  --title "feat(live): monthly budget hard stop and attempt ledger" \
  --body-file - <<'EOF'
**Do not merge from 22:15 UTC on a weekday until that night's live refresh has finished** (the check is in `docs/superpowers/plans/2026-10-14-live-board-fixes.md`, Merging and records). GitHub starts the 22:30 UTC cron as late as 03:05 UTC, and the refresh keeps running on Render after the workflow run ends. The merge auto-deploys prod, and the restart kills a running refresh, so the sessions it had not checkpointed are billed again on the next night.

Prices every billable Live Leaderboard attempt into a durable ledger and stops model deploys at `LEADERBOARD_MONTHLY_BUDGET_USD` (default $50/month; spec §7).

- New `domain/leaderboard/live_budget.py`: one `agent_runs` row per (month, entry) in session `leaderboard-live-spend`, cumulative across successes, crashes, H6 refusals and forced replays; seeded from existing checkpoint rows on first use. No new DB methods.
- `deploy_live_model_increment`: both already-current returns happen before the attempt (never priced or halted); pre-check before any model build or bar fetch; ledger upsert at every session close before H6; stop before a session that would cross the cap; a crash is priced. `LLMAgentStrategy` now publishes tokens per step.
- `refresh_live_leaderboard`: halts are not failures and do not touch the shared-fault stop; one `ERROR: live.budget_exhausted …` line; `models_deployed` stays false while halted. The success path moves into the `try` statement's `else`, and a failed budget read logs `ERROR: live.budget_status_failed` without skipping the prune and the state save.
- Public GET adds only `live_status.budget_halted` and per-model `last_attempt {on, outcome, kind}`: no new dollar figure (the per-entry `est_cost_usd` predates this PR) and no error text. The live tab shows a banner line and a "Last nightly run" row.
- Test plan: `pytest dashboard/backend/tests/test_live_budget.py dashboard/backend/tests/test_frontend_live_budget.py dashboard/backend/tests/test_live_leaderboard.py -q`; after deploy, the next nightly log has no `live.budget_exhausted` line and `GET /api/v1/leaderboard?period=live` shows `budget_halted: false` with DeepSeek's `last_attempt.outcome` `ok`.
EOF
```

Add the PR-body attribution line your setup requires, if any. The PR is not gated: the default cap ($50) is far above today's live spend (about $0.35/month), so nothing halts on deploy.

- [ ] **Step 4: Immediately before merging, re-check the cache-buster.** This PR bumps one tag, `js/live-leaderboard.js?v=` (Task 3.5 Edit 7).

```bash
git fetch origin main
git show origin/main:dashboard/frontend/app.html | grep -o 'js/live-leaderboard.js?v=[0-9]*'
git show HEAD:dashboard/frontend/app.html | grep -o 'js/live-leaderboard.js?v=[0-9]*'
```

Expected: `origin/main` holds exactly this branch's value minus one. If main has moved (another PR bumped the same tag), rebase onto `origin/main`, bump again from main's value, and push before merging. If the rebase ever leaves this PR bumping `js/leaderboard.js?v=` or `home-page.js?v=` as well, also update their exact pins in `dashboard/backend/tests/test_frontend_fast_boot.py` (≈:200-201) in the same commit. Identical one-line bumps from two branches merge with no conflict, so this check is the only thing that stops the second PR's JS shipping under a version browsers already cached. Then merge, outside the window named in the PR body's last paragraph.

#### Phase 3 review-focus candidates

1. **Old checkpoint rows with NULL or junk `est_cost_usd`** (rows written before cost tracking, or a bad migration). Added: `test_an_unreadable_spend_figure_counts_as_nothing` in Task 3.1 (`None`, text, `nan`, `inf`, negative all count as $0, never crash the pre-check or poison the total).
2. **The ledger write itself fails** (Neon outage, disk full) while a deploy is failing for another reason. Added: `test_a_ledger_that_cannot_be_written_never_hides_the_deploy_error` in Task 3.3 (the original error still propagates, one `ERROR: live.spend_ledger_write_failed` line is printed, and the per-session upsert that did land is kept). On the success path a failed final write makes the deploy report a failure, by design; since #668 widened the per-entry catch to `Exception`, a raw driver error there (e.g. `sqlite3.OperationalError`) is recorded as that entry's failure, with its traceback, exactly as a failed checkpoint write is. If two entries in a row fail that way, `RosterFailures` lists the rest as not attempted.
3. **Render kills the process mid-attempt** (a deploy during the nightly refresh): no `finish` runs. Covered by `test_a_session_close_keeps_the_previous_outcome_until_the_attempt_ends` (the ledger holds spend through the last session close and the previous, dated `last_attempt`), so the board shows the last real outcome rather than a guess.
4. **Two refreshes in different processes at once** (an operator running `scripts/refresh_live_leaderboard.py --force` with prod credentials while the cron's background thread runs). Not tested: both attempts read the same base and the later upsert wins, losing at most one attempt's spend. The in-process `_live_refresh_running` lock already prevents this inside one web process; a cross-process lock needs a DB-level claim and is a follow-up issue, not this phase.
5. **A roster model with no price row** (a new slug on a key switch) is priced at the $1/$5 fallback, which over-estimates and so halts early: the safe direction for a hard stop. P4 (`2026-10-14-live-board-fixes-p4-roster.md`; Decision 6 in the index plan) adds the "every live model id has a listed price" test, so it is not duplicated here.
6. **The second already-current return** (`live_increment_bounds(...) is None`, reached when `freeze_end` is not a session day). The dry run of the first draft left it inside `_trade_live_segment`, after the pre-check and `get_strategy`: under a spent budget it halted an entry with nothing to trade, and otherwise it built the model and wrote an `ok` ledger row. Fixed in Task 3.3 Edits 3–4 (the check moves into the head, ahead of the attempt) and pinned by `test_a_snapshot_that_already_covers_a_non_session_freeze_is_not_halted`. Review the one semantic change: a config change on an entry that is current this way now waits for its next increment, as it already did for the `latest.end_date == freeze_end` return. `live_clock` never produces such a `freeze_end`, so prod behaviour is unchanged.
7. **Ledger rows leaking between test modules.** The test DB is session-scoped and every live deploy now writes a `leaderboard-live-spend` row. Covered: `test_live_leaderboard.py`'s `_isolated_live_session`, which P2's and P4's new live test modules import, now clears both sessions (Task 3.3 Step 1 Edit 1), and `test_live_budget.py` clears both in its own autouse fixture. A future module that deploys live models without either fixture would leave rows that only the GET's `last_attempt` reads.
8. **A halt next to the roster bookkeeping #668 added.** Since #668 the success path runs after the `try` statement, so Task 3.3 Edit 8 moves it into the statement's `else`, where no `except` clause can fall through into it. Covered: `test_a_halt_after_a_success_reports_each_entry_once` (success, then halt), `test_a_spent_month_halts_the_roster_with_one_log_line` (the first entry halts) and `test_halts_and_failures_are_reported_apart` (failure, then halt). A halt is deliberately kept out of `RosterFailures`, even one between sessions: it neither counts toward nor resets the two-in-a-row shared-fault stop (Edit 7's note gives the cost argument). Pinned by `test_a_halt_between_two_shared_failures_does_not_reset_the_stop`.
9. **The budget read after the roster fails** (a Neon blip or `PoolTimeout` once every model has been billed). Task 3.3 Edit 9 guards it, so the refresh still prunes and saves state and keeps `model_results`. Added: `test_a_budget_read_that_fails_still_prunes_and_saves`.
