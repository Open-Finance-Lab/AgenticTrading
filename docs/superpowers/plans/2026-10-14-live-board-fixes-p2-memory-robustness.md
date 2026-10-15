# Live-Board Fixes P2: Roster-Loop Robustness and Nightly Trade Memory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the nightly Live refresh trading the rest of the roster when one model raises, pin what a failed night stores, and carry each model's 24h trade memory across the nightly snapshot with a marker of where that memory began.

**Architecture:** `refresh_live_leaderboard` widens its per-entry catch to `Exception`. `PortfolioManager.snapshot_state` adds the 24h trade tail and a `trade_memory`/`trade_memory_since` marker to the snapshot that `live._write_live_checkpoint` already stores on every `agent_runs` checkpoint; `restore_state` loads the tail into a read-only `restored_trades` that the prompt reads before the manager's own trades. `llm_agent.py`, `live_trade_stats.py`, fees, slippage and prompt wording stay untouched.

**Tech Stack:** Python (3.13 in the Dockerfile) / FastAPI backend, domain layer only (`domain/leaderboard/live.py`, `domain/backtesting/portfolio_manager.py`), pandas, pytest from the repo root (`tests/conftest.py` points `DATABASE_PATH` at a temp SQLite file).

**Spec:** `docs/superpowers/specs/2026-10-14-atl-direction.md` §6 ("Fixes to the board while it runs on the legacy engine": trade memory across nights; a failed model replays from the 1st). Decisions, Global Constraints, merge order and Review Focus live in the index plan `docs/superpowers/plans/2026-10-14-live-board-fixes.md`; every task here inherits its Global Constraints.

**Merge order:** P1 → P2 → P3 → P4 → P5 → P6 → P7 PR A → P7 PR B (owner-gated). Deadlines: P3 before 2026-11-01; P4 by 2026-10-31 (else bump both live_from values); P1 merged and deployed by 2026-11-20. One executor per contiguous range; phases are not safe to run in parallel (several append to the same test files and bump the same cache-busters).

## Phase 2: Roster-loop robustness and nightly trade memory

**Why.** Spec §6 lists two legacy-engine fixes this phase owns: "Trade memory across nights" and "A failed model replays from the 1st every night". Both were checked against the code on 2026-10-14 (Decisions 3 and 4 in the index plan):

- **Trade memory.** Every nightly increment is a fresh process with a fresh `PortfolioManager`, and `snapshot_state` does not carry `trades`. So the first decisions of every night see an empty `recent_trades` block, and the "do not reverse a trade from the last day" rule (`domain/agents/defaults.py:80`, `validator.py:612`) cannot fire.
- **Replay from the 1st.** The claim is wrong as written. Since #582, every session that clears H6 is checkpointed one by one. The real gap is in the roster loop: it catches only three exception types, so a `ConnectionError` from one model stops every later model and also skips the prune and the state save.
- **Marking the switch.** Carrying the memory changes the prompt's inputs mid-month, so the month this merges in joins nights that opened with no memory of the night before to nights that opened with it. Each checkpoint deletes the entry's earlier rows, so the newest row is the only record left; its snapshot therefore says both that it carries memory (`trade_memory`) and from which bar the memory began (`trade_memory_since`).

**Depends on:** nothing in code. Following the merge order, branch from `main` after P1 (`2026-10-14-live-board-fixes-p1-early-close.md`) has merged. This phase shares no code hunks with P1. Both phases edit `tests/test_live_leaderboard.py`, but at different anchors. P3 (`2026-10-14-live-board-fixes-p3-attempt-ledger.md`, Decision 5 in the index plan) adds its `LiveBudgetExceeded` handler above the `except Exception` introduced here, so merge this phase before that one.

Pass counts below are measured on main@17c552c5 with only this phase applied, and, where it differs, on main@17c552c5 with P1 applied first. P1 takes `test_live_leaderboard.py` from 57 tests to 70. If `main` has moved further, a neighbour-suite total that is off by the tests `main` gained is not a failure; the named failing tests and their messages are the real check.

**Branch / PR title:** branch `fix/live-memory-robustness`, PR title `fix(live): roster error isolation and nightly trade memory`.

Before cutting the branch, check that the name is unused: `gh pr list --head fix/live-memory-robustness --state all` must print nothing (it printed nothing on 2026-10-14). Then cut it from the remote main:

```bash
git fetch origin main
git switch -c fix/live-memory-robustness --no-track origin/main
```

This does not switch to or pull the local `main`, so it also works inside a worktree while `main` is checked out elsewhere (where `git switch main` fails with "already used by worktree"). `--no-track` keeps the new branch from tracking `origin/main`.

Parallel sessions share one checkout. Run `git branch --show-current` before every commit below; it must print `fix/live-memory-robustness`.

Commit commands below omit any `Co-Authored-By` trailer. An agent executing this plan adds its own.

**Interfaces this phase produces**

- `dashboard.backend.domain.backtesting.portfolio_manager.TRADE_MEMORY_WINDOW: datetime.timedelta = timedelta(hours=24)`. This is the `recent_trades` look-back. It replaces the literal `timedelta(hours=24)` in `make_trading_decision_with_llm`.
- `dashboard.backend.domain.backtesting.portfolio_manager.SNAPSHOT_TRADE_MEMORY_LIMIT: int = 200`. It caps the trades a snapshot writes and a restore accepts. It is a guard against runaway growth, not a tuning knob.
- `PortfolioManager.restored_trades: List[Dict[str, Any]]` (new instance attribute, `[]` after `__init__`). It holds trade memory restored from a snapshot. Each item is `{"symbol": str, "side": str, "shares": float, "price": float, "timestamp": pandas.Timestamp (always tz-aware)}`. Nothing outside `restore_state` writes to it, and it never enters `self.trades`.
- `PortfolioManager.snapshot_state()` gains the key `"recent_trades"`. Its value is a `List[{"symbol": str, "side": str, "shares": float, "price": float, "timestamp": str}]`:
  - `timestamp` is ISO-8601 with an offset, e.g. `"2026-08-26T18:00:00+00:00"`.
  - The list is ordered oldest first.
  - It holds every restored or own trade newer than (newest trade − `TRADE_MEMORY_WINDOW`), keeping at most the newest `SNAPSHOT_TRADE_MEMORY_LIMIT`.
  - It is strict JSON: no NaN and no numpy types.
- `PortfolioManager.restore_state(snapshot)` reads `snapshot["recent_trades"]` into `restored_trades`:
  - If the key is absent or the value is not a list, the result is `[]`.
  - Unusable rows are skipped.
  - Naive timestamps are read as UTC.
  - The result is capped at `SNAPSHOT_TRADE_MEMORY_LIMIT`.
  - The existing early return on an empty snapshot is unchanged.
- `dashboard.backend.domain.backtesting.portfolio_manager.TRADE_MEMORY_VERSION: int = 1`, and `PortfolioManager.trade_memory_since: Optional[pandas.Timestamp]` (new instance attribute, `None` after `__init__`).
- `PortfolioManager.snapshot_state()` also gains `"trade_memory": TRADE_MEMORY_VERSION` (always) and `"trade_memory_since": str | None`: the first bar from which this chain of snapshots has carried trade memory, ISO-8601 with an offset. It is the restored `trade_memory_since` when there is one, else the timestamp of this manager's first equity point (its first decision bar), else `None` before the manager has decided anything. Rule for a reader: a session dated after that stamp opened with the previous session's trades in its prompt; the session the stamp falls in did not, unless it was the month's first and had nothing to remember.
- `PortfolioManager.restore_state(snapshot)` sets `trade_memory_since` from `snapshot["trade_memory_since"]` only when `snapshot.get("trade_memory") == TRADE_MEMORY_VERSION` and the stamp is readable (naive read as UTC); otherwise `None`, so the mark restarts at this manager's first bar. A snapshot written before this phase has neither key and restores exactly as before.
- Where the marker lands: `LLMAgentStrategy._record_progress` sets `last_portfolio_snapshot = manager.snapshot_state()` at every session close, and `live._write_live_checkpoint` stores that dict as `metadata["live_portfolio_snapshot"]` (`live.LIVE_SNAPSHOT_KEY`) in its final `db.insert_run`. So the marker reaches every checkpoint row with no change to `live.py` or `llm_agent.py`. The `live_increment` lineage was not used: it would need a `live.py` edit and describes one segment only, while `_drop_superseded_live_rows` deletes every earlier row of the month as each checkpoint lands.
- The prompt loop in `make_trading_decision_with_llm` reads `chain(self.restored_trades, self.trades)`. The window test (`> timestamp - TRADE_MEMORY_WINDOW`) and the five emitted fields are byte-identical to before.
- Private helpers in `portfolio_manager.py` (later phases may reuse them, but must not rename them without updating the tests below):
  - `_memory_timestamp(value) -> Optional[pd.Timestamp]`
  - `_memory_trade(trade) -> Optional[Dict[str, Any]]`
  - `PortfolioManager._trade_memory_tail() -> List[Dict[str, Any]]`
  - `PortfolioManager._trade_memory_since() -> Optional[str]`
- `dashboard.backend.domain.leaderboard.live.refresh_live_leaderboard`: the per-entry handler becomes `except Exception as exc:`. A later phase that needs a type-specific outcome (for example `LiveBudgetExceeded`) adds its `except` clause **above** this one.
  - The ERROR line is unchanged: `ERROR: live.model_deploy_failed entry=<id> error=<ExceptionTypeName>: <message>`. It already named the type on main.
  - The shape of `model_failures` items is unchanged: `{"entry_id": str, "error": str}`.
- Test helpers other phases may import:
  - `_DeadModelAgent(_ScriptedAgent)` in `dashboard/backend/tests/test_live_leaderboard.py`.
  - `_ReplyByClock`, `_OneSymbolAgent(LLMAgentStrategy)`, `_capture`, `_night`, `_buy`, `_shown`, `_two_nights` in `dashboard/backend/tests/test_live_trade_memory.py`.

Untouched on purpose:
- `domain/leaderboard/strategies/llm_agent.py`: `_record_progress` still copies `manager.trades` only.
- `domain/leaderboard/live_trade_stats.py`.
- `domain/backtesting/external_run_service.py`: it has its own `recent_trades` loop over its own session's trades.
- All fees, slippage and prompt wording.

---

### Task 2.1: The roster loop survives any `Exception` from one model

**Files:**
- Modify: `dashboard/backend/domain/leaderboard/live.py` (`refresh_live_leaderboard`)
- Test: `dashboard/backend/tests/test_live_leaderboard.py`

**Interfaces:**
- Consumes these existing names:
  - from `live`: `refresh_live_leaderboard`, `live_freeze_config`, `live_llm_entries`, `_live_refresh_state`;
  - from `lb_service`: `ensure_leaderboard_runs`, `fetch_hourly_bars`, `get_strategy`;
  - test helpers: `_deepseek`, `_ScriptedAgent`, `_one_bar_frame`, `_insert_live_run`, `_pt`, `SEED`, `_ET`, `db`.
- Produces: the per-entry handler is now `except Exception as exc:`, and `refresh_live_leaderboard`'s docstring states the contract.

- [ ] **Step 1: Write the failing test**

In `dashboard/backend/tests/test_live_leaderboard.py`, find the decorator of `test_reference_start_clamps_to_the_prior_month` (≈:1265 on main@17c552c5):

```python
@pytest.mark.parametrize(
    ("start", "expected"),
```

Replace with:

```python
def test_an_unexpected_error_from_one_model_does_not_stop_the_roster(
    monkeypatch, capsys,
):
    """The per-entry catch named three types. Anything else (a dropped database
    connection, a socket error from a provider SDK) escaped it: every later
    entry went untraded, and the prune and the state save never ran."""
    as_of = datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    freeze_cfg = live.live_freeze_config(as_of)
    deepseek = _deepseek(freeze_cfg)
    second = {**deepseek, "id": "second_model", "name": "Second Model"}
    monkeypatch.setattr(live, "live_llm_entries", lambda *args, **kwargs: [deepseek, second])
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())
    healthy = _ScriptedAgent()

    def get_strategy(entry):
        if entry["id"] == "deepseek_v4_pro":
            raise ConnectionError("server closed the connection unexpectedly")
        return healthy

    monkeypatch.setattr(lb_service, "get_strategy", get_strategy)
    # A superseded row the end-of-refresh prune must still remove.
    for end in ("2026-08-25", "2026-08-26"):
        _insert_live_run(
            f"lb_spy_index_20260801_{end.replace('-', '')}", "spy_index", end,
            [_pt(f"{end}T19:00:00+00:00", SEED)],
        )

    result = live.refresh_live_leaderboard(deploy_models=True, as_of=as_of)

    assert result["models_deployed"] is False
    assert result["model_failures"] == [
        {
            "entry_id": "deepseek_v4_pro",
            "error": "server closed the connection unexpectedly",
        }
    ]
    assert [row["entry_id"] for row in result["model_results"]] == ["second_model"]
    assert healthy.traded[-1] == "2026-08-27"
    assert (
        "ERROR: live.model_deploy_failed entry=deepseek_v4_pro "
        "error=ConnectionError: server closed the connection unexpectedly"
    ) in capsys.readouterr().out
    assert result["pruned_runs"] == 1
    assert db.get_run("lb_spy_index_20260801_20260825") is None
    saved = live._live_refresh_state()
    assert saved["window_key"] == result["window_key"]
    assert saved["model_failures"] == result["model_failures"]


def test_a_base_exception_still_stops_the_refresh(monkeypatch):
    """Exception, not BaseException: a shutdown must not be logged as one
    model's failure while the loop carries on billing the rest."""
    as_of = datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())

    def get_strategy(entry):
        raise SystemExit("worker shutting down")

    monkeypatch.setattr(lb_service, "get_strategy", get_strategy)
    with pytest.raises(SystemExit):
        live.refresh_live_leaderboard(deploy_models=True, as_of=as_of)
    assert live._live_refresh_state() == {}


@pytest.mark.parametrize(
    ("start", "expected"),
```

The second test passes on main. It pins that `BaseException` still propagates, so nobody "fixes" the catch all the way up to `BaseException`. `live.live_llm_entries` is monkeypatched with an argument-agnostic lambda, so the first test does not depend on the roster: Decision 6 in the index plan (P4, `2026-10-14-live-board-fixes-p4-roster.md`) drops Nemotron and gates new entries by `live_from`.

- [ ] **Step 2: Run it and confirm it fails**

```bash
pytest "dashboard/backend/tests/test_live_leaderboard.py::test_an_unexpected_error_from_one_model_does_not_stop_the_roster" "dashboard/backend/tests/test_live_leaderboard.py::test_a_base_exception_still_stops_the_refresh" -q -p no:cacheprovider
```

Expected: `1 failed, 1 passed`. The failure is the exception itself escaping `refresh_live_leaderboard`:

```
E           ConnectionError: server closed the connection unexpectedly
FAILED dashboard/backend/tests/test_live_leaderboard.py::test_an_unexpected_error_from_one_model_does_not_stop_the_roster
```

- [ ] **Step 3: Implement**

In `dashboard/backend/domain/leaderboard/live.py`, function `refresh_live_leaderboard` (≈:1078), find:

```python
            except (lb_service.LeaderboardFallbackError, ValueError, RuntimeError) as exc:
                # The background refresh keeps this list only in a state file on
                # an ephemeral disk; the log line is the part an operator sees.
```

Replace with:

```python
            except Exception as exc:
                # Any Exception, not a list of expected types: one model's
                # dropped database connection or SDK socket error must not leave
                # the rest of the roster untraded and skip the prune and state
                # save below. BaseException (a shutdown) still propagates. The
                # type name in the line is what tells an operator a programming
                # error from a provider outage, since both now land here.
                # The background refresh keeps this list only in a state file on
                # an ephemeral disk; the log line is the part an operator sees.
```

In the same function's docstring (≈:1014), find:

```python
    session. When ``deploy_models`` is True, each LLM entry continues from the
    latest snapshot and only trades sessions not yet stored (typically one
    cash day). Public GET never calls this.
    """
```

Replace with:

```python
    session. When ``deploy_models`` is True, each LLM entry continues from the
    latest snapshot and only trades sessions not yet stored (typically one
    cash day). An entry that raises is logged and listed in
    ``model_failures``; the rest of the roster still trades, and the prune and
    the state save still run. Public GET never calls this.
    """
```

Leave the `print(...)` and the `failures.append(...)` below the handler as they are. On main the line already prints `error={type(exc).__name__}: {exc}`.

- [ ] **Step 4: Run it and confirm it passes**

```bash
pytest "dashboard/backend/tests/test_live_leaderboard.py::test_an_unexpected_error_from_one_model_does_not_stop_the_roster" "dashboard/backend/tests/test_live_leaderboard.py::test_a_base_exception_still_stops_the_refresh" -q -p no:cacheprovider
pytest dashboard/backend/tests/test_live_leaderboard.py -q -p no:cacheprovider
```

Expected: `2 passed`, then the whole file green with `59 passed`: the file has 57 tests on main@17c552c5, and this task adds 2. With P1 merged first, expect `72 passed` (P1 leaves the file at 70).

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/leaderboard/live.py dashboard/backend/tests/test_live_leaderboard.py
git commit -m "fix(live): keep the roster going when one model raises" -m "The per-entry catch named three types, so a ConnectionError or a database error from one model skipped every later model, the prune and the state save. BaseException still propagates."
```

---

### Task 2.2: Characterization tests for what a failed night stores and where it resumes

**Files:**
- Test: `dashboard/backend/tests/test_live_leaderboard.py`

This task changes no production code. The tests pass on main. Their job is to record the true H6 boundary, so that a later "checkpoint on failure" change cannot quietly publish a prefix the model did not drive.

**Interfaces:**
- Consumes these existing test helpers: `_ScriptedAgent`, `_use_agent`, `_live_aug27`, `_deepseek`, `_insert_checkpoint`, `_month_rows`, `_pt`, `SEED`; and `live.deploy_live_model_increment`, `live._trading_days_inclusive`, `lb_service.LeaderboardFallbackError`.
- Produces `_DeadModelAgent(_ScriptedAgent)`. It is a total-fallback stand-in: every session is offered and traded, but `llm_calls` and `llm_decisions` stay at 0.

- [ ] **Step 1: Write the test**

In `dashboard/backend/tests/test_live_leaderboard.py`, find (≈:1219 on main@17c552c5):

```python
def test_a_session_with_no_bars_stops_the_run_loudly(monkeypatch, capsys):
```

Replace with:

```python
# What a failed night leaves behind, pinned as it is (2026-10-14). The rule
# these encode: a session is checkpointed only once the segment traded so far
# clears H6, so billed sessions H6 has not yet accepted are lost when the run
# dies, and the next night re-bills them. Storing that prefix anyway would
# publish a curve the model did not drive; do not "fix" these by doing so.


class _DeadModelAgent(_ScriptedAgent):
    """Every call fails before generating anything (a gateway 404): each step
    falls back to rule-based, so sessions are offered and traded while
    ``llm_calls`` and ``llm_decisions`` never move."""

    def run(self, bars, start, end, capital, starting_snapshot=None, on_session_close=None):
        self.runs.append((start, end, (starting_snapshot or {}).get("day")))
        days = live._trading_days_inclusive(date.fromisoformat(start), date.fromisoformat(end))
        self.planned_decision_steps = self.steps * len(days)
        curve = []
        for day in days:
            iso = day.isoformat()
            self.traded.append(iso)
            self.decision_steps += self.steps
            curve.append(_pt(f"{iso}T19:00:00+00:00", SEED))
            self.last_portfolio_snapshot = {"cash": SEED, "positions": {}, "day": iso[5:]}
            on_session_close(day, list(curve))
        return curve


def test_a_dead_model_stores_nothing_and_restarts_at_the_1st_every_night(monkeypatch):
    freeze_cfg = _live_aug27()
    for _night in range(2):
        runs = []
        agent = _use_agent(monkeypatch, _DeadModelAgent(runs=runs))
        with pytest.raises(lb_service.LeaderboardFallbackError, match="llm_calls=0"):
            live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
        # Aug 1 is a Saturday: the first session is the 3rd, and it is the
        # only one traded before the guard stops the run.
        assert runs == [("2026-08-01", "2026-08-27", None)]
        assert agent.traded == ["2026-08-03"]
        assert agent.llm_calls == 0
        assert _month_rows("deepseek_v4_pro") == []


def test_an_early_miss_then_a_crash_loses_the_unchecked_prefix(monkeypatch):
    freeze_cfg = _live_aug27()
    agent = _use_agent(
        monkeypatch,
        _ScriptedAgent(misses={"2026-08-03": 1}, crash_on={"2026-08-05"}),
    )
    with pytest.raises(RuntimeError, match="worker killed"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    # 6/7 and 13/14 are under 95%, so neither close was stored, yet both
    # sessions were billed.
    assert agent.traded == ["2026-08-03", "2026-08-04"]
    assert agent.llm_calls == 14
    assert _month_rows("deepseek_v4_pro") == []

    runs = []
    _use_agent(monkeypatch, _ScriptedAgent(runs=runs))
    live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert runs == [("2026-08-01", "2026-08-27", None)]


def test_a_single_miss_on_a_nightly_increment_keeps_the_prior_checkpoint(monkeypatch):
    freeze_cfg = _live_aug27()
    _insert_checkpoint("2026-08-26", "08-26")
    agent = _use_agent(monkeypatch, _ScriptedAgent(misses={"2026-08-27": 1}))
    with pytest.raises(lb_service.LeaderboardFallbackError, match="6/7"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert agent.llm_calls == 7
    assert [r["end_date"] for r in _month_rows("deepseek_v4_pro")] == ["2026-08-26"]

    # The retry resumes from that checkpoint, not from the 1st.
    runs = []
    _use_agent(monkeypatch, _ScriptedAgent(runs=runs))
    live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert runs == [("2026-08-27", "2026-08-27", "08-26")]


def test_a_session_with_no_bars_stops_the_run_loudly(monkeypatch, capsys):
```

- [ ] **Step 2: Run it, confirm it passes on main, then confirm it bites**

```bash
pytest dashboard/backend/tests/test_live_leaderboard.py -k "dead_model_stores_nothing or early_miss_then_a_crash or single_miss_on_a_nightly" -q -p no:cacheprovider
```

Expected: `3 passed`. These are characterization tests.

Now prove that the H6 boundary test would catch someone storing the refused prefix. In `dashboard/backend/domain/leaderboard/live.py`, function `_checkpoint_passes_h6`, temporarily replace:

```python
            lb_service.MIN_LLM_DECISION_COVERAGE * planned
        ):
            return False
```

with:

```python
            lb_service.MIN_LLM_DECISION_COVERAGE * planned
        ):
            return True
```

Re-run the command above. Expected: `1 failed, 2 passed`, with these lines among the output. The `created_at` time (`21:54:46` here) is the wall clock of the run, so it varies.

```
>       assert _month_rows("deepseek_v4_pro") == []
E       AssertionError: assert [{'agent_name...:54:46', ...}] == []
E         Left contains one more item: {'agent_name': 'DeepSeek V4 Pro', 'baseline_buyhold_run_id': None, 'baseline_djia_run_id': None, 'created_at': '2026-10-14 21:54:46', ...}
FAILED dashboard/backend/tests/test_live_leaderboard.py::test_an_early_miss_then_a_crash_loses_the_unchecked_prefix
```

Restore the file and confirm that only the test file is modified:

```bash
git checkout -- dashboard/backend/domain/leaderboard/live.py
git status --short    # expect exactly: M dashboard/backend/tests/test_live_leaderboard.py
```

- [ ] **Step 3: Implement**

No production change. The behaviour pinned here is the intended behaviour (Decision 4 in the index plan: never checkpoint an H6-refused prefix).

- [ ] **Step 4: Run the file**

```bash
pytest dashboard/backend/tests/test_live_leaderboard.py -q -p no:cacheprovider
```

Expected: all green with `62 passed`: 57 on main@17c552c5, plus 2 from Task 2.1, plus 3 from this task. With P1 merged first: `75 passed`.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/tests/test_live_leaderboard.py
git commit -m "test(live): pin what a failed night stores and where it resumes" -m "A dead model stores nothing and restarts at the 1st; an early miss then a crash loses the prefix H6 never accepted; a single nightly miss keeps the prior checkpoint. Sessions that clear H6 are checkpointed one by one (#582)."
```

---

### Task 2.3: `PortfolioManager` carries the 24h trade memory in its snapshot

**Files:**
- Modify: `dashboard/backend/domain/backtesting/portfolio_manager.py`
- Create: `dashboard/backend/tests/backtesting/test_trade_memory.py`

**Interfaces:**
- Consumes: `PortfolioManager.snapshot_state` / `restore_state` / `make_trading_decision_with_llm` / `update_equity` and its `equity_history` list, and the module-level `create_prompt` (which the test monkeypatches, as `tests/backtesting/test_portfolio_manager_move.py::test_liquidity_market_cap_and_industry_reach_llm_snapshot` does).
- Produces: `TRADE_MEMORY_WINDOW`, `SNAPSHOT_TRADE_MEMORY_LIMIT`, `TRADE_MEMORY_VERSION`, `_memory_timestamp`, `_memory_trade`, `PortfolioManager.restored_trades`, `PortfolioManager.trade_memory_since`, `PortfolioManager._trade_memory_tail`, `PortfolioManager._trade_memory_since`, and the `"recent_trades"`, `"trade_memory"` and `"trade_memory_since"` snapshot keys. The exact shapes are in the phase header.

The memory must never go into `self.trades`. `LLMAgentStrategy._record_progress` sets `_num_trades = len(manager.trades)` and `trades = list(manager.trades)`, and `live._write_live_checkpoint` adds `num_trades()` to the carried count and folds `strategy_impl.trades` into `live_trade_stats` on every checkpoint. A restored trade in `self.trades` would therefore be counted again on every later night. Task 2.4 tests this end to end.

The marker lives in the snapshot because the snapshot is the one thing that reaches every checkpoint row unchanged: `_record_progress` stores `manager.snapshot_state()` as `last_portfolio_snapshot`, and `_write_live_checkpoint` writes that dict to `metadata["live_portfolio_snapshot"]`. The stamp has to chain through `restore_state` because each checkpoint deletes the month's earlier rows (`_drop_superseded_live_rows`), so a bare `trade_memory: 1` on the surviving row could not say which session the memory began in.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/backtesting/test_trade_memory.py`:

```python
"""24h trade memory carried across a ``PortfolioManager`` snapshot.

The Live board runs each night in a fresh process with a fresh manager. The
book always crossed that boundary through ``snapshot_state``/``restore_state``;
the trades the prompt's ``recent_trades`` block reads did not, so the first
decisions of every night saw no trades at all and the "do not reverse a trade
from the last day" rule could not fire. These pin the carried memory: what is
kept, how it reads back, that it never joins ``trades``, which the Live board
counts and folds into the month's trade statistics, and the marker that says
from which bar a chain of snapshots has carried it.
"""

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from dashboard.backend.domain.backtesting import portfolio_manager as pm_mod
from dashboard.backend.domain.backtesting.portfolio_manager import (
    SNAPSHOT_TRADE_MEMORY_LIMIT,
    TRADE_MEMORY_VERSION,
    TRADE_MEMORY_WINDOW,
    PortfolioManager,
)


def _trade(ts, symbol="AAPL", side="BUY", shares=10, price=100.0, **extra):
    return {
        "timestamp": pd.Timestamp(ts),
        "symbol": symbol,
        "side": side,
        "shares": shares,
        "price": price,
        **extra,
    }


def _memory(ts, symbol="AAPL", side="BUY", shares=10.0, price=100.0):
    return {
        "symbol": symbol,
        "side": side,
        "shares": shares,
        "price": price,
        "timestamp": ts,
    }


class _Usage:
    input_tokens = 1
    output_tokens = 1


class _Reply:
    content = [type("Block", (), {"text": json.dumps({"actions": []})})()]
    usage = _Usage()


class _Client:
    class messages:
        @staticmethod
        def create(**_kwargs):
            return _Reply()


def _prompt_memory(manager, monkeypatch, decision_ts):
    """The ``recent_trades`` block one decision at ``decision_ts`` is shown."""
    seen = []

    def capture(snapshot, **_kwargs):
        seen.append(snapshot)
        return "prompt"

    monkeypatch.setattr(pm_mod, "create_prompt", capture)
    state = {
        "timestamp": pd.Timestamp(decision_ts),
        "cash": manager.cash,
        "positions": [],
        "positions_value": 0,
        "total_equity": manager.cash,
        "market_signals": {},
    }
    manager.make_trading_decision_with_llm(state, _Client())
    assert len(seen) == 1
    return seen[0]["recent_trades"]


def test_window_and_cap_constants():
    assert TRADE_MEMORY_WINDOW == timedelta(hours=24)
    assert SNAPSHOT_TRADE_MEMORY_LIMIT == 200
    assert TRADE_MEMORY_VERSION == 1


def test_snapshot_keeps_the_24h_tail_as_strict_json():
    manager = PortfolioManager(initial_capital=10_000)
    manager.trades = [
        _trade("2026-08-25T14:00:00Z", side="BUY"),
        _trade(
            "2026-08-26T14:00:00Z",
            shares=np.int64(10),
            price=np.float64(100.125),
            reason="[LLM] momentum (confidence: 90%)",
            cost=1001.25,
        ),
        _trade("2026-08-26T18:00:00Z", symbol="MSFT", side="SELL", shares=5.0, price=200.5),
    ]
    snap = manager.snapshot_state()
    # 08-25 14:00 is more than 24h before the newest trade; reason/cost are
    # not prompt fields and are not stored.
    assert snap["recent_trades"] == [
        _memory("2026-08-26T14:00:00+00:00", price=100.125),
        _memory("2026-08-26T18:00:00+00:00", symbol="MSFT", side="SELL", shares=5.0, price=200.5),
    ]
    assert json.loads(json.dumps(snap, allow_nan=False)) == snap


def test_snapshot_without_trades_has_empty_memory():
    assert PortfolioManager().snapshot_state()["recent_trades"] == []


def test_snapshot_memory_is_capped_to_the_newest_trades():
    manager = PortfolioManager(initial_capital=10_000)
    start = pd.Timestamp("2026-08-26T14:00:00Z")
    manager.trades = [
        _trade(start + pd.Timedelta(minutes=i), shares=i + 1) for i in range(250)
    ]
    memory = manager.snapshot_state()["recent_trades"]
    assert len(memory) == SNAPSHOT_TRADE_MEMORY_LIMIT
    assert memory[0]["shares"] == 51.0
    assert memory[-1]["shares"] == 250.0


def test_snapshot_skips_trade_rows_it_cannot_describe():
    manager = PortfolioManager(initial_capital=10_000)
    manager.trades = [
        _trade("2026-08-26T14:00:00Z"),
        {"symbol": "AAPL", "side": "BUY", "shares": 1, "price": 1.0},
        _trade("2026-08-26T15:00:00Z", price=float("nan")),
        _trade("2026-08-26T16:00:00Z", shares=None),
        {**_trade("2026-08-26T17:00:00Z"), "timestamp": "not-a-date"},
    ]
    assert manager.snapshot_state()["recent_trades"] == [
        _memory("2026-08-26T14:00:00+00:00"),
    ]


def test_restore_keeps_memory_out_of_the_trade_list():
    night1 = PortfolioManager(initial_capital=10_000)
    night1.trades = [
        _trade("2026-08-26T14:00:00Z"),
        _trade("2026-08-26T18:00:00Z", side="SELL", shares=4),
    ]
    stored = json.loads(json.dumps(night1.snapshot_state()))

    night2 = PortfolioManager(initial_capital=10_000)
    night2.restore_state(stored)
    assert night2.trades == []
    assert [t["timestamp"] for t in night2.restored_trades] == [
        pd.Timestamp("2026-08-26T14:00:00Z"),
        pd.Timestamp("2026-08-26T18:00:00Z"),
    ]
    assert all(t["timestamp"].tzinfo is not None for t in night2.restored_trades)
    assert [(t["side"], t["shares"]) for t in night2.restored_trades] == [
        ("BUY", 10.0),
        ("SELL", 4.0),
    ]


def test_an_old_snapshot_restores_no_memory():
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({"cash": 100.0, "positions": {"AAPL": 1}})
    assert manager.restored_trades == []
    assert manager.cash == 100.0


def test_restore_skips_corrupt_memory_rows():
    valid = _memory("2026-08-26T18:00:00+00:00")
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({
        "cash": 100.0,
        "positions": {},
        "recent_trades": [
            {k: v for k, v in valid.items() if k != "timestamp"},
            {**valid, "timestamp": "not-a-date"},
            {**valid, "timestamp": None},
            {**valid, "timestamp": 1756231200},
            {k: v for k, v in valid.items() if k != "symbol"},
            {**valid, "side": ""},
            {**valid, "price": "NaN"},
            {**valid, "shares": "ten"},
            "garbage",
            None,
            valid,
        ],
    })
    assert manager.restored_trades == [
        {**valid, "timestamp": pd.Timestamp("2026-08-26T18:00:00Z")},
    ]


@pytest.mark.parametrize("memory", ["not a list", {"AAPL": 1}, 7])
def test_restore_ignores_memory_that_is_not_a_list(memory):
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({"cash": 100.0, "positions": {}, "recent_trades": memory})
    assert manager.restored_trades == []


def test_restore_reads_a_naive_timestamp_as_utc():
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({
        "cash": 100.0,
        "recent_trades": [_memory("2026-08-26T18:00:00")],
    })
    assert manager.restored_trades[0]["timestamp"] == pd.Timestamp("2026-08-26T18:00:00Z")


def test_restore_caps_an_oversized_memory():
    start = pd.Timestamp("2026-08-26T14:00:00Z")
    rows = [
        _memory((start + pd.Timedelta(minutes=i)).isoformat(), shares=float(i + 1))
        for i in range(250)
    ]
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({"cash": 100.0, "recent_trades": rows})
    assert len(manager.restored_trades) == SNAPSHOT_TRADE_MEMORY_LIMIT
    assert manager.restored_trades[0]["shares"] == 51.0


def test_prompt_shows_restored_trades_inside_24h_before_its_own(monkeypatch):
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({
        "cash": 10_000.0,
        "recent_trades": [
            _memory("2026-08-26T14:00:00+00:00", price=100.126),
            _memory("2026-08-26T18:00:00+00:00", price=100.126),
        ],
    })
    manager.trades = [_trade("2026-08-27T14:00:00Z", symbol="MSFT", side="SELL", shares=5)]

    memory = _prompt_memory(manager, monkeypatch, "2026-08-27T15:00:00Z")
    assert memory == [
        _memory("2026-08-26T18:00:00+00:00", price=100.13),
        {
            "symbol": "MSFT",
            "side": "SELL",
            "shares": 5,
            "price": 100.0,
            "timestamp": "2026-08-27T14:00:00+00:00",
        },
    ]
    assert manager.trades == [
        _trade("2026-08-27T14:00:00Z", symbol="MSFT", side="SELL", shares=5)
    ]


def test_a_weekend_empties_monday_mornings_memory(monkeypatch):
    friday = PortfolioManager(initial_capital=10_000)
    friday.trades = [_trade("2026-08-28T19:00:00Z")]
    monday = PortfolioManager(initial_capital=10_000)
    monday.restore_state(json.loads(json.dumps(friday.snapshot_state())))
    assert len(monday.restored_trades) == 1
    assert _prompt_memory(monday, monkeypatch, "2026-08-31T14:00:00Z") == []


def test_memory_chains_across_nights():
    night1 = PortfolioManager(initial_capital=10_000)
    night1.trades = [
        _trade("2026-08-26T14:00:00Z"),
        _trade("2026-08-26T18:00:00Z"),
    ]
    night2 = PortfolioManager(initial_capital=10_000)
    night2.restore_state(json.loads(json.dumps(night1.snapshot_state())))
    night2.trades = [_trade("2026-08-27T15:00:00Z", side="SELL", shares=20)]
    # Anchored on the new newest trade: 08-26 14:00 is now over 24h old.
    assert [
        (row["timestamp"], row["side"])
        for row in night2.snapshot_state()["recent_trades"]
    ] == [
        ("2026-08-26T18:00:00+00:00", "BUY"),
        ("2026-08-27T15:00:00+00:00", "SELL"),
    ]


def _decide_at(manager, ts):
    """Record one bar's equity point, as the decision loop does after each bar."""
    manager.update_equity({}, None, pd.Timestamp(ts))


def test_snapshot_marks_the_memory_from_the_managers_first_bar():
    manager = PortfolioManager(initial_capital=10_000)
    snap = manager.snapshot_state()
    assert (snap["trade_memory"], snap["trade_memory_since"]) == (1, None)
    _decide_at(manager, "2026-08-26T14:00:00Z")
    _decide_at(manager, "2026-08-26T19:00:00Z")
    snap = manager.snapshot_state()
    assert (snap["trade_memory"], snap["trade_memory_since"]) == (
        1,
        "2026-08-26T14:00:00+00:00",
    )
    assert json.loads(json.dumps(snap, allow_nan=False)) == snap


def test_restore_carries_where_the_memory_started():
    night1 = PortfolioManager(initial_capital=10_000)
    _decide_at(night1, "2026-08-26T14:00:00Z")
    night2 = PortfolioManager(initial_capital=10_000)
    night2.restore_state(json.loads(json.dumps(night1.snapshot_state())))
    assert night2.trade_memory_since == pd.Timestamp("2026-08-26T14:00:00Z")
    _decide_at(night2, "2026-08-27T14:00:00Z")
    assert night2.snapshot_state()["trade_memory_since"] == "2026-08-26T14:00:00+00:00"


@pytest.mark.parametrize(
    "marker",
    [
        {},
        {"trade_memory_since": "2026-08-03T14:00:00+00:00"},
        {"trade_memory": TRADE_MEMORY_VERSION, "trade_memory_since": "not-a-date"},
        {"trade_memory": TRADE_MEMORY_VERSION, "trade_memory_since": None},
    ],
    ids=["pre-marker", "stamp-without-marker", "unreadable-stamp", "no-stamp"],
)
def test_a_snapshot_without_a_readable_mark_restarts_it_here(marker):
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({"cash": 100.0, "positions": {}, **marker})
    assert manager.trade_memory_since is None
    _decide_at(manager, "2026-08-27T14:00:00Z")
    assert manager.snapshot_state()["trade_memory_since"] == "2026-08-27T14:00:00+00:00"
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
pytest dashboard/backend/tests/backtesting/test_trade_memory.py -q -p no:cacheprovider
```

Expected: a collection error:

```
E   ImportError: cannot import name 'SNAPSHOT_TRADE_MEMORY_LIMIT' from 'dashboard.backend.domain.backtesting.portfolio_manager'
```

- [ ] **Step 3: Implement**

Make all seven edits below in `dashboard/backend/domain/backtesting/portfolio_manager.py`. Each find block occurs exactly once on main@17c552c5.

(1) Imports (≈:28):

```python
import json
import math
from datetime import date, datetime, timedelta
from types import MappingProxyType
```

Replace with:

```python
import json
import math
from datetime import date, datetime, timedelta
from itertools import chain
from types import MappingProxyType
```

(2) Constants and helpers, right after `SNAPSHOT_SHORTLIST_SIZE` (≈:98):

```python
# Above that -- and always on the single-prompt path -- the snapshot keeps this
# many top-trend names plus current holdings. A pipeline prompt says so
# (``universe_note``); the single-prompt one lists every VALID SYMBOL instead.
SNAPSHOT_SHORTLIST_SIZE = 12
```

Replace with:

```python
# Above that -- and always on the single-prompt path -- the snapshot keeps this
# many top-trend names plus current holdings. A pipeline prompt says so
# (``universe_note``); the single-prompt one lists every VALID SYMBOL instead.
SNAPSHOT_SHORTLIST_SIZE = 12

# How far back a decision's ``recent_trades`` block looks. The Live board
# carries this memory across its nightly increments through ``snapshot_state``,
# so it is also how much trade history a snapshot keeps.
TRADE_MEMORY_WINDOW = timedelta(hours=24)

# Runaway bound on the trades a snapshot carries (and a restore accepts). Six
# decisions a session over a 30-name universe is at most 180 trades inside the
# window, so the window, not this cap, normally decides what is kept.
SNAPSHOT_TRADE_MEMORY_LIMIT = 200

# Marks a snapshot that carries ``recent_trades``; ``trade_memory_since`` beside
# it says from which bar the chain of snapshots has carried it (see
# ``PortfolioManager._trade_memory_since``). The Live month the memory shipped
# in joins nights that opened with no memory of the night before to nights
# that opened with it, and these two keys are the record of where.
TRADE_MEMORY_VERSION = 1


def _memory_timestamp(value: Any) -> Optional[pd.Timestamp]:
    """``value`` as a tz-aware Timestamp, or None when it is not a readable one.

    A naive stamp is read as UTC, the zone the board's bars are stamped in, so
    a restored trade can always be compared with a tz-aware decision clock
    (comparing naive with aware raises, and the decision path would swallow
    that into a rule-based fallback).
    """
    if not isinstance(value, (str, datetime)):
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def _memory_trade(trade: Any) -> Optional[Dict[str, Any]]:
    """The prompt's five fields of one trade, or None for a row it cannot use.

    Snapshotting and restoring both pass through here, so a corrupt row is
    dropped on either side instead of failing the session it was read in.
    """
    if not isinstance(trade, Mapping):
        return None
    symbol, side = trade.get("symbol"), trade.get("side")
    if not (isinstance(symbol, str) and symbol and isinstance(side, str) and side):
        return None
    ts = _memory_timestamp(trade.get("timestamp"))
    if ts is None:
        return None
    try:
        shares = float(trade.get("shares"))
        price = float(trade.get("price"))
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(shares) and math.isfinite(price)):
        return None
    return {
        "symbol": symbol,
        "side": side,
        "shares": shares,
        "price": price,
        "timestamp": ts,
    }
```

(3) `PortfolioManager.__init__` (≈:147):

```python
        self.entry_prices = {}  # {symbol: entry_price}
        self.trades = []
        self.order_events = []
```

Replace with:

```python
        self.entry_prices = {}  # {symbol: entry_price}
        self.trades = []
        # Trades from earlier sessions, restored from a snapshot only so the
        # prompt's 24h ``recent_trades`` block can see them. Never merged into
        # ``trades``: LLMAgentStrategy copies ``trades`` into its trade count and
        # the Live board folds that copy into the month's trade stats on every
        # checkpoint (live.py ``_write_live_checkpoint``), so a restored trade
        # there would be counted again on every later night.
        self.restored_trades: List[Dict[str, Any]] = []
        # The bar this chain of snapshots began carrying trade memory from,
        # restored from a snapshot that records it. None means the chain
        # begins with this manager (``_trade_memory_since``).
        self.trade_memory_since: Optional[pd.Timestamp] = None
        self.order_events = []
```

(4) `snapshot_state` docstring (≈:194):

```python
    def snapshot_state(self) -> Dict[str, Any]:
        """JSON-safe book for resuming a later session (Live daily increment)."""
```

Replace with:

```python
    def snapshot_state(self) -> Dict[str, Any]:
        """JSON-safe book for resuming a later session (Live daily increment).

        Carries cash, positions, T+1 lots and ``recent_trades``: the trade
        memory the next session's prompts read (see ``_trade_memory_tail``),
        marked by ``trade_memory`` and ``trade_memory_since``.
        """
```

(5) The end of `snapshot_state`, plus the new `_trade_memory_tail` and `_trade_memory_since` methods and the `restore_state` docstring (≈:225):

```python
            "t_plus_one_enabled": bool(self.t_plus_one_enabled),
        }

    def restore_state(self, snapshot: Optional[Dict[str, Any]]) -> None:
        """Load cash/positions/T+1 lots from ``snapshot_state`` (no-op if empty)."""
```

Replace with:

```python
            "t_plus_one_enabled": bool(self.t_plus_one_enabled),
            "recent_trades": self._trade_memory_tail(),
            "trade_memory": TRADE_MEMORY_VERSION,
            "trade_memory_since": self._trade_memory_since(),
        }

    def _trade_memory_tail(self) -> List[Dict[str, Any]]:
        """Restored and own trades a later prompt can still be shown, JSON-safe.

        Every later decision is at or after the newest trade, so a trade a full
        ``TRADE_MEMORY_WINDOW`` older than that one can never pass a later
        decision's cutoff: the tail anchored on the newest trade is a superset
        of what any later ``recent_trades`` block reads. Oldest first, at most
        ``SNAPSHOT_TRADE_MEMORY_LIMIT`` (the newest), timestamps as ISO text.
        """
        rows = [
            row
            for row in map(_memory_trade, chain(self.restored_trades, self.trades))
            if row is not None
        ]
        if not rows:
            return []
        rows.sort(key=lambda row: row["timestamp"])
        cutoff = rows[-1]["timestamp"] - TRADE_MEMORY_WINDOW
        tail = [row for row in rows if row["timestamp"] > cutoff]
        return [
            {**row, "timestamp": row["timestamp"].isoformat()}
            for row in tail[-SNAPSHOT_TRADE_MEMORY_LIMIT:]
        ]

    def _trade_memory_since(self) -> Optional[str]:
        """The bar from which this chain of snapshots has carried trade memory,
        as ISO text, or None before this manager has decided anything.

        Restored from a snapshot that records it; otherwise this manager's
        first bar, because a manager that restored no memory is where the
        memory starts. A session dated after the stamp opened with the previous
        session's trades in its prompt; the session it falls in did not, unless
        it was the month's first and had nothing to remember. The Live board
        deletes every superseded checkpoint, so its newest row is the only one
        left to say whether its month joined the two, and a bare version flag
        could not say where.
        """
        since = self.trade_memory_since
        if since is None and self.equity_history:
            since = _memory_timestamp(self.equity_history[0].get("timestamp"))
        return since.isoformat() if since is not None else None

    def restore_state(self, snapshot: Optional[Dict[str, Any]]) -> None:
        """Load cash/positions/T+1 lots and the 24h trade memory from
        ``snapshot_state`` (no-op if empty).

        The memory goes to ``restored_trades``, never ``trades``. A snapshot
        written before it existed restores none; a corrupt memory row is
        skipped rather than allowed to fail the session. ``trade_memory_since``
        is kept only from a snapshot marked ``trade_memory``.
        """
```

(6) The end of `restore_state` (≈:268). Keep the `if not snapshot: return` at its top untouched:

```python
                restored.append({"quantity": quantity, "buy_date": buy})
            if restored:
                lots_out[str(symbol)] = restored
        self.frozen_lots = lots_out
```

Replace with:

```python
                restored.append({"quantity": quantity, "buy_date": buy})
            if restored:
                lots_out[str(symbol)] = restored
        self.frozen_lots = lots_out
        memory = snapshot.get("recent_trades")
        restored_trades = [
            row
            for row in map(_memory_trade, memory if isinstance(memory, list) else [])
            if row is not None
        ]
        self.restored_trades = restored_trades[-SNAPSHOT_TRADE_MEMORY_LIMIT:]
        # Unmarked (written before the memory existed) or unreadable: the mark
        # restarts at this manager's first bar, which can only record the
        # switch later than it happened, never earlier.
        self.trade_memory_since = (
            _memory_timestamp(snapshot.get("trade_memory_since"))
            if snapshot.get("trade_memory") == TRADE_MEMORY_VERSION
            else None
        )
```

(7) The prompt loop in `make_trading_decision_with_llm` (≈:444). Only these lines change; the `recent_trades.append({...})` body below stays byte-identical:

```python
            recent_trades = []
            cutoff_time = timestamp - timedelta(hours=24)
            for trade in self.trades:
                if trade["timestamp"] > cutoff_time:
```

Replace with:

```python
            recent_trades = []
            cutoff_time = timestamp - TRADE_MEMORY_WINDOW
            # Restored memory first: it is older than anything this manager
            # has traded itself, so the block stays in time order.
            for trade in chain(self.restored_trades, self.trades):
                if trade["timestamp"] > cutoff_time:
```

`timedelta` stays imported, because `TRADE_MEMORY_WINDOW` uses it. Do not touch `llm_agent.py`, `live.py` or `live_trade_stats.py`.

- [ ] **Step 4: Run it and confirm it passes**

```bash
pytest dashboard/backend/tests/backtesting/test_trade_memory.py -q -p no:cacheprovider
pytest dashboard/backend/tests/backtesting/test_portfolio_manager_move.py dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/test_live_trade_stats.py dashboard/backend/tests/domain/leaderboard/test_llm_agent_instruction.py dashboard/backend/tests/domain/backtesting/test_snapshot_universe.py dashboard/backend/tests/test_architecture_boundaries.py -q -p no:cacheprovider
```

Expected:
- `22 passed`: 17 test functions, two of them parametrized (three and four ways).
- The neighbour run is all green with `199 passed`. These six files hold 194 tests on main@17c552c5. Tasks 2.1 and 2.2 add 5 to `test_live_leaderboard.py`, and Task 2.3 adds none to these files. With P1 merged first: `212 passed`.
- The existing snapshot tests keep passing unmodified. `test_portfolio_snapshot_roundtrip_keeps_cash_and_lots` asserts per field. The minimal `{"cash", "positions"}` snapshots elsewhere in `test_live_leaderboard.py` are the old-snapshot cases. The `Manager`/`FakeManager` fakes define their own `snapshot_state`/`restore_state`.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/backtesting/portfolio_manager.py dashboard/backend/tests/backtesting/test_trade_memory.py
git commit -m "fix(live): carry the 24h trade memory across nightly snapshots" -m "snapshot_state omitted trades, so each night's first prompts saw no recent_trades. The memory is restored into a separate read-only restored_trades, never trades, which the Live board counts and folds into the month's stats. Each snapshot is marked trade_memory=1 with trade_memory_since, the bar the memory began, carried through restore so the newest checkpoint still shows where a month switched."
```

---

### Task 2.4: Live-board tests for two nights of trade memory

**Files:**
- Create: `dashboard/backend/tests/test_live_trade_memory.py`

The test runs two nights through `live.deploy_live_model_increment`, using the real `LLMAgentStrategy._run_decision_loop`, the real `PortfolioManager` and the real checkpoint writes into `agent_runs`. Only the bar fetch and the model are replaced. `_OneSymbolAgent` overrides `run()` to skip the bar download and the indicator build, which makes the test independent of Phase 1's session-calendar changes.

The helpers and the autouse `_isolated_live_session` fixture are imported from `test_live_leaderboard.py`, following the repo precedent in `test_credits_api_review_fixes.py` and `test_leaderboard_samples.py`.

**Interfaces:**
- Consumes:
  - Task 2.3's `"recent_trades"`, `"trade_memory"` and `"trade_memory_since"` snapshot keys, `restored_trades` and the prompt loop;
  - `live.deploy_live_model_increment`, `live.live_freeze_config`, `live._snapshot_from_run`, `live.LIVE_SNAPSHOT_KEY`, `live._trading_days_inclusive`;
  - `live_trade_stats.TRADE_STATS_KEY`, `llm_agent.LLMAgentStrategy`;
  - `pm_mod.create_prompt` (monkeypatched);
  - test helpers `_ET`, `_deepseek`, `_insert_checkpoint`, `_isolated_live_session`, `_month_rows`, `_one_bar_frame`.
- Produces the test helpers `_ReplyByClock`, `_OneSymbolAgent`, `_capture`, `_night`, `_buy`, `_shown`, `_two_nights`.

- [ ] **Step 1: Write the test**

Create `dashboard/backend/tests/test_live_trade_memory.py`:

```python
"""Live board: the 24h trade memory survives the nightly process boundary.

Two nights through ``deploy_live_model_increment`` with the real
``LLMAgentStrategy`` decision loop and the real ``PortfolioManager``; only the
bar fetch and the model are replaced. Each night builds a fresh strategy and a
fresh manager, as the nightly job does, so whatever night 2 knows about night 1
arrived through the checkpoint row in ``agent_runs``.
"""

import json
from datetime import date, datetime

import pandas as pd

from dashboard.backend.database import db
from dashboard.backend.domain.backtesting import portfolio_manager as pm_mod
import dashboard.backend.domain.leaderboard.live as live
from dashboard.backend.domain.leaderboard.live_trade_stats import TRADE_STATS_KEY
import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.domain.leaderboard.strategies import llm_agent
from dashboard.backend.tests.test_live_leaderboard import (  # noqa: F401 - fixture
    _ET,
    _deepseek,
    _insert_checkpoint,
    _isolated_live_session,
    _month_rows,
    _one_bar_frame,
)

# Open stamps of the six decision bars, 10:00-15:00 ET in August (UTC-4).
_DECISION_HOURS_UTC = (14, 15, 16, 17, 18, 19)


class _ReplyByClock:
    """A model that answers each decision by the prompt's timestamp.

    ``_capture`` makes the prompt text the snapshot's timestamp, so the
    script can say what to do at which bar; every other bar holds.
    """

    def __init__(self, actions_at):
        actions_by_stamp = actions_at

        class _Messages:
            @staticmethod
            def create(**kwargs):
                stamp = kwargs["messages"][0]["content"]
                actions = actions_by_stamp.get(
                    stamp, [{"symbol": "AAPL", "action": "hold", "confidence": 0.9}]
                )
                block = type("Block", (), {"text": json.dumps({"actions": actions})})()
                usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 5})()
                return type("Reply", (), {"content": [block], "usage": usage})()

        self.messages = _Messages()


class _OneSymbolAgent(llm_agent.LLMAgentStrategy):
    """The real decision loop over one flat-priced symbol and a scripted model.

    Only ``run`` is replaced, to skip the bar download and indicator build;
    the loop, the manager, the snapshot and the session-close hook are real.
    """

    def __init__(self, entry, actions_at):
        super().__init__(entry)
        self._client = _ReplyByClock(actions_at)

    def required_symbols(self):
        return ["AAPL"]

    def run(self, bars, start_date, end_date, initial_capital,
            starting_snapshot=None, on_session_close=None):
        days = live._trading_days_inclusive(
            date.fromisoformat(start_date), date.fromisoformat(end_date)
        )
        stamps = [
            pd.Timestamp(f"{day.isoformat()}T{hour}:00:00Z")
            for day in days
            for hour in _DECISION_HOURS_UTC
        ]
        frame = pd.DataFrame(
            {"close": [100.0] * len(stamps)}, index=pd.DatetimeIndex(stamps)
        )
        self.used_llm = True
        self.planned_decision_steps = len(stamps)
        manager = self._run_decision_loop(
            client=self._client,
            timestamps=stamps,
            symbols=["AAPL"],
            data={"AAPL": frame},
            price_cache={},
            initial_capital=initial_capital,
            model_id=self.model_id,
            starting_snapshot=starting_snapshot,
            on_session_close=on_session_close,
        )
        self._record_progress(manager, decision_steps=len(stamps))
        return manager.get_equity_curve()


def _capture(monkeypatch):
    """Record every prompt snapshot by its decision timestamp."""
    prompts = {}

    def capture(snapshot, **_kwargs):
        prompts[snapshot["timestamp"]] = snapshot
        return snapshot["timestamp"]

    monkeypatch.setattr(pm_mod, "create_prompt", capture)
    return prompts


def _night(monkeypatch, day, actions_at):
    """One nightly increment as of 17:39 ET on Aug ``day``, in a fresh strategy."""
    freeze_cfg = live.live_freeze_config(datetime(2026, 8, day, 17, 39, tzinfo=_ET))
    entry = _deepseek(freeze_cfg)
    agent = _OneSymbolAgent(entry, actions_at)
    monkeypatch.setattr(lb_service, "get_strategy", lambda _entry: agent)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())
    row = live.deploy_live_model_increment(entry, freeze_cfg)
    return agent, row


def _buy(shares):
    return [{"symbol": "AAPL", "action": "buy", "confidence": 0.9, "position_size": shares}]


def _shown(snapshot):
    return [(t["side"], t["shares"], t["timestamp"]) for t in snapshot["recent_trades"]]


def _two_nights(monkeypatch):
    """Night 1 (Aug 26) buys 10 at 10:00 and 10 at 14:00 ET; night 2 (Aug 27)
    sells all 20 at 11:00 ET. Each resumes from a checkpoint and trades one
    session. Returns the prompts, both nights, and night 1's stored snapshot
    (read before night 2 supersedes its row)."""
    # A checkpoint from before the memory existed: night 1 resumes from it.
    _insert_checkpoint("2026-08-25", "08-25")
    prompts = _capture(monkeypatch)
    night1 = _night(monkeypatch, 26, {
        "2026-08-26T14:00:00+00:00": _buy(10),
        "2026-08-26T18:00:00+00:00": _buy(10),
    })
    stored1 = live._snapshot_from_run(db.get_run(night1[1]["run_id"]))
    night2 = _night(monkeypatch, 27, {
        "2026-08-27T15:00:00+00:00": [
            {"symbol": "AAPL", "action": "sell", "confidence": 0.9},
        ],
    })
    assert night1[1]["segment"] == {"start_date": "2026-08-26", "end_date": "2026-08-26"}
    assert night2[1]["segment"] == {"start_date": "2026-08-27", "end_date": "2026-08-27"}
    return prompts, night1, night2, stored1


def test_night_two_prompts_see_night_ones_trades_for_24_hours(monkeypatch):
    prompts, _night1, _night2, _stored1 = _two_nights(monkeypatch)
    # 10:00 ET: night 1's 14:00 ET buy is 20h old and shown; its 10:00 ET buy
    # is exactly 24h old and is not (the window is strict).
    assert _shown(prompts["2026-08-27T14:00:00+00:00"]) == [
        ("BUY", 10.0, "2026-08-26T18:00:00+00:00"),
    ]
    # After night 2's own sell, the restored buy still comes first.
    assert _shown(prompts["2026-08-27T16:00:00+00:00"]) == [
        ("BUY", 10.0, "2026-08-26T18:00:00+00:00"),
        ("SELL", 20.0, "2026-08-27T15:00:00+00:00"),
    ]
    # 14:00 ET: night 1's last buy is now 24h old as well.
    assert _shown(prompts["2026-08-27T18:00:00+00:00"]) == [
        ("SELL", 20.0, "2026-08-27T15:00:00+00:00"),
    ]


def test_restored_trades_are_never_counted_as_new_trades(monkeypatch):
    _prompts, (night1, _row1), (night2, row2), _stored1 = _two_nights(monkeypatch)
    assert night1.num_trades() == 2
    # Night 2 traded once; the two buys it remembers are not its trades.
    assert night2.num_trades() == 1
    assert [t["side"] for t in night2.trades] == ["SELL"]
    stored = db.get_run(row2["run_id"])
    assert stored["num_trades"] == 3
    stats = stored["metadata"][TRADE_STATS_KEY]
    assert (stats["buys"], stats["sells"]) == (2, 1)
    assert [r["run_id"] for r in _month_rows("deepseek_v4_pro")] == [row2["run_id"]]


def test_each_checkpoint_round_trips_the_memory_through_agent_runs(monkeypatch):
    _prompts, _night1, (_night2, row2), stored1 = _two_nights(monkeypatch)
    assert stored1["recent_trades"] == [
        {
            "symbol": "AAPL",
            "side": "BUY",
            "shares": 10.0,
            "price": 100.0,
            "timestamp": "2026-08-26T14:00:00+00:00",
        },
        {
            "symbol": "AAPL",
            "side": "BUY",
            "shares": 10.0,
            "price": 100.0,
            "timestamp": "2026-08-26T18:00:00+00:00",
        },
    ]
    # Night 2's row carries the tail anchored on its own sell: the 10:00 ET
    # buy from night 1 is more than 24h older than that and is dropped.
    stored2 = live._snapshot_from_run(db.get_run(row2["run_id"]))
    assert [(t["side"], t["timestamp"]) for t in stored2["recent_trades"]] == [
        ("BUY", "2026-08-26T18:00:00+00:00"),
        ("SELL", "2026-08-27T15:00:00+00:00"),
    ]


def test_the_checkpoint_row_records_where_the_memory_started(monkeypatch):
    _prompts, (night1, _row1), (_night2, row2), stored1 = _two_nights(monkeypatch)
    # Night 1 resumed from a checkpoint written before the marker existed, so
    # the memory starts at its first decision bar, 10:00 ET on Aug 26.
    first_bar = "2026-08-26T14:00:00+00:00"
    snap1 = night1.last_portfolio_snapshot
    assert (snap1["trade_memory"], snap1["trade_memory_since"]) == (1, first_bar)
    assert (stored1["trade_memory"], stored1["trade_memory_since"]) == (1, first_bar)
    # Night 1's row is superseded and deleted. The one row left still says the
    # month joined a night without memory (Aug 26) to nights with it.
    assert [r["run_id"] for r in _month_rows("deepseek_v4_pro")] == [row2["run_id"]]
    stored2 = db.get_run(row2["run_id"])["metadata"][live.LIVE_SNAPSHOT_KEY]
    assert (stored2["trade_memory"], stored2["trade_memory_since"]) == (1, first_bar)
```

- [ ] **Step 2: Run it, then confirm each test fails without the behaviour it guards**

```bash
pytest dashboard/backend/tests/test_live_trade_memory.py -q -p no:cacheprovider
```

Expected: `4 passed`, because Task 2.3 already implemented the behaviour. Before Task 2.3, the first, third and fourth tests fail. Their first failing assertions are `assert [] == [('BUY', 10.0, ...)]`, `KeyError: 'recent_trades'` and `KeyError: 'trade_memory'`. The fourth test is the one that reads the marker back from the stored `agent_runs` row: `stored1` comes from `live._snapshot_from_run(db.get_run(...))` and `stored2` straight from the row's `metadata["live_portfolio_snapshot"]`.

Apply each mutation below to `dashboard/backend/domain/backtesting/portfolio_manager.py` on its own. After each one, re-run the command, check the expected failure, then restore with `git checkout -- dashboard/backend/domain/backtesting/portfolio_manager.py`.

(a) Leak the memory into `trades` (the double count). Find:

```python
        self.restored_trades = restored_trades[-SNAPSHOT_TRADE_MEMORY_LIMIT:]
```

Replace with:

```python
        self.restored_trades = restored_trades[-SNAPSHOT_TRADE_MEMORY_LIMIT:]
        self.trades = list(self.restored_trades)
```

Expected: `3 failed, 1 passed`. The counting test stops at `assert night2.num_trades() == 1` with `assert 3 == 1`.

(b) Drop restored memory from the prompt. Find:

```python
            for trade in chain(self.restored_trades, self.trades):
```

Replace with:

```python
            for trade in self.trades:
```

Expected: `1 failed, 3 passed`. `test_night_two_prompts_see_night_ones_trades_for_24_hours` fails with `assert [] == [('BUY', 10.0, '2026-08-26T18:00:00+00:00')]`.

(c) Stop writing the memory. Find:

```python
            "recent_trades": self._trade_memory_tail(),
```

Replace with:

```python
            "recent_trades": [],
```

Expected: `2 failed, 2 passed`. The prompt test fails, and so does `test_each_checkpoint_round_trips_the_memory_through_agent_runs` at `assert stored1["recent_trades"] == [...]` (`assert [] == [{...}]`).

(d) Stop carrying the marker's stamp through a restore. Find:

```python
            if snapshot.get("trade_memory") == TRADE_MEMORY_VERSION
```

Replace with:

```python
            if False
```

Expected: `1 failed, 3 passed`. `test_the_checkpoint_row_records_where_the_memory_started` fails on night 2's row with `At index 1 diff: '2026-08-27T14:00:00+00:00' != '2026-08-26T14:00:00+00:00'`: the stamp restarts at night 2's own first bar, which would hide the night without memory.

After the last restore, `git status --short` must show only `?? dashboard/backend/tests/test_live_trade_memory.py`.

- [ ] **Step 3: Implement**

No production change. The behaviour comes from Task 2.3.

- [ ] **Step 4: Run it and the neighbours**

```bash
pytest dashboard/backend/tests/test_live_trade_memory.py dashboard/backend/tests/test_live_leaderboard.py dashboard/backend/tests/backtesting/test_trade_memory.py dashboard/backend/tests/test_live_trade_stats.py -q -p no:cacheprovider
```

Expected: all green with `91 passed`: 4 + 62 + 22 + 3. With P1 merged first: `104 passed` (4 + 75 + 22 + 3).

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/tests/test_live_trade_memory.py
git commit -m "test(live): two nights of trade memory through the checkpoint row" -m "Night 2's prompts show night 1's trades for exactly 24h, restored trades are never counted in num_trades or the month's trade stats, each checkpoint round-trips the memory through agent_runs, and the surviving row still records the bar the memory began."
```

---

### Task 2.5: Open the PR

**Files:** none changed.

- [ ] **Step 1: Run the full suite**

```bash
pytest dashboard/backend/tests -q -p no:cacheprovider
```

Expected: green, apart from `test_report_pdf.py` when run locally. The run takes about 8 minutes.
- Locally, every failure must be in `dashboard/backend/tests/test_report_pdf.py`. Those are environmental (no `reportlab`), and CI installs it. Judge by file, not count: 7 such failures were measured on main@17c552c5 with this phase applied.
- Do not add `-x`: the first environmental failure would stop the run. Do not add `--timeout`: `pytest-timeout` is not installed, so pytest rejects the flag.
- Without `node` on `PATH`, the frontend-behaviour tests skip rather than fail. This phase touches no JS.

- [ ] **Step 2: Check the diff is exactly this phase**

```bash
git fetch origin main
git diff --stat origin/main...HEAD
```

Expected: exactly these five files.
- `dashboard/backend/domain/leaderboard/live.py`
- `dashboard/backend/domain/backtesting/portfolio_manager.py`
- `dashboard/backend/tests/test_live_leaderboard.py`
- `dashboard/backend/tests/backtesting/test_trade_memory.py`
- `dashboard/backend/tests/test_live_trade_memory.py`

If `main` moved, run `git rebase origin/main` and repeat Step 1.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin fix/live-memory-robustness
gh pr create --base main --head fix/live-memory-robustness \
  --title "fix(live): roster error isolation and nightly trade memory" \
  --body-file - <<'EOF'
- Roster loop: the per-entry catch in `refresh_live_leaderboard` is now `except Exception`. It was three types, so a `ConnectionError` or a DB error from one model skipped every later model, the prune and the state save. `BaseException` still propagates. The ERROR line already names the type.
- Characterization tests pin how a failed night behaves:
  - A dead model stores nothing and restarts at the 1st.
  - An early miss followed by a crash loses the prefix H6 never accepted, and that prefix is re-billed.
  - A single nightly miss keeps the prior checkpoint.

  Sessions that clear H6 have been checkpointed one by one since #582, so spec §6's "never checkpoints" is wrong as written. Storing an H6-refused prefix stays forbidden.
- Trade memory: `snapshot_state` now writes `recent_trades`, which holds the last 24h of trades (prompt fields only, at most 200, strict JSON). `restore_state` loads them into a new read-only `PortfolioManager.restored_trades`, which the prompt reads before the manager's own trades. Restored trades never enter `trades`, so `num_trades` and the month's trade stats are not double counted. This is tested end to end through `agent_runs`. No change to fees, slippage, prompt wording, `llm_agent.py` or `live_trade_stats.py`.
- The month this merges in mixes sessions with and without trade memory; the first checkpoint carrying trade_memory=1 marks the switch. Each checkpoint deletes the month's earlier rows, so the snapshot also carries `trade_memory_since` through every restore: the first bar of the night that started the memory. A session dated after it opened with the previous night's trades.
- Prod behaviour change: the first night after deploy restores a pre-change snapshot and has no memory. From the second night on, each night's first prompts see the previous session's trades. Stored rows are not rewritten and nothing replays. A full replay and a nightly increment now give the model the same memory.
- Do not merge between 22:15 and 23:30 UTC on a weekday. The merge auto-deploys prod, and the Render restart kills the 22:30 UTC live refresh thread mid-run, so that night's session is billed again on the next night.

Test plan: `pytest dashboard/backend/tests -q -p no:cacheprovider`
EOF
```

Add the PR-body attribution line your harness requires, if any.

- [ ] **Step 4: After merge**

This step is for the owner or operator, who has prod access.
- After the first weekday 22:30 UTC refresh that follows the deploy, the service log should show no `live.model_deploy_failed` line for a healthy model.
- After that first refresh, the newest `leaderboard-live` row for `deepseek_v4_pro` should have `metadata.live_portfolio_snapshot.trade_memory` equal to `1` and `trade_memory_since` on the first session that refresh traded (its first decision bar). Note the date: it is the month's switch.
- After the second such refresh, that model's newest row should have `metadata.live_portfolio_snapshot.recent_trades` as a list (it may be empty if the model did not trade that session), and `trade_memory_since` should be unchanged from the first refresh. A later date means the stamp is not being carried through the restore.

File anything unexpected as an issue, not as a PR comment.

#### Phase 2 review-focus candidates

1. **A naive decision clock meeting restored memory.** `trade["timestamp"] > cutoff_time` raises `TypeError` if the decision timestamp is naive while a restored stamp is aware. The decision path swallows that into a rule-based fallback on every step, and H6 then refuses the night loudly. No test added, because no caller can reach it:
   - Only `LLMAgentStrategy` restores memory.
   - Its decision stamps are Alpaca's tz-aware bars.
   - Restored stamps are always made tz-aware (`test_restore_reads_a_naive_timestamp_as_utc`).

   Pinning the fallback would pin the wrong behaviour.
2. **An exception with an empty message, e.g. `ConnectionError()`.** The log line still reads `error=ConnectionError: `, which is the part an operator sees. The ephemeral state file's `error` is `""`. No test added: the attempt ledger (Decision 5 in the index plan, P3) records a durable `kind` for exactly this, and pinning an empty string buys nothing.
3. **A failure after the roster loop.** If `prune_superseded_live_runs` or `_save_live_refresh_state` raises (database or disk down at the end of the run), the refresh still aborts. `_run_live_refresh_background` logs "background refresh failed". Not tested: this is outside Decision 4 in the index plan, which is about the per-entry catch, and the models' checkpoints are already written by then.
4. **The first night after deploy.** It restores a pre-change snapshot with no `recent_trades`. This is covered:
   - `test_an_old_snapshot_restores_no_memory` (Task 2.3) covers it at the manager level.
   - Night 1 in `_two_nights` (Task 2.4) resumes from a pre-change `_insert_checkpoint` row through the real live path, and `test_the_checkpoint_row_records_where_the_memory_started` pins the marker that night writes.
5. **Prompt text parity between a nightly increment and a full replay.** A restored trade renders `"shares": 10.0`, while a trade the manager made itself renders the executor's `10`. Prices (rounded to 2dp) and ISO timestamps are identical. This is accepted as cosmetic and not tested. Positions restored from a snapshot already render as floats on main, so the two paths were never byte-identical.
6. **`test_a_base_exception_still_stops_the_refresh` uses the real roster.** Unlike its sibling, it does not monkeypatch `live.live_llm_entries`, so it needs at least one live LLM entry for August 2026 whose `get_strategy` call is reached. DeepSeek has no `live_from` and stays on the roster under Decision 6 in the index plan (P4), and the budget pre-check of Decision 5 in the index plan (P3) passes on an empty ledger, so it holds after those phases. If a later change breaks that, the test fails loudly with `DID NOT RAISE <class 'SystemExit'>` rather than passing silently. Verified passing on main@17c552c5 with this phase applied.
7. **`trade_memory_since` errs late, never early.** A marked snapshot whose stamp is missing or unreadable restarts the mark at the restoring manager's first bar (`test_a_snapshot_without_a_readable_mark_restarts_it_here`). That can only under-report the sessions that had memory. The other direction, a stamp earlier than the truth, would need a snapshot carrying the marker that was written before this phase, which cannot exist.
