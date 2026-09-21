# Backtest Visible Start (Track A) Implementation Plan

> **Status: IMPLEMENTED — shipped as PR #501 (`feat/backtest-visible-start`).**
> The unticked boxes below are an artefact of how the plan was executed, not
> work outstanding; they were never maintained during the run. Read the tree,
> not this file, for what the branch actually does. Kept for its rationale and
> its measurements, which the code does not restate.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dashboard backtest's card names what the child is doing from the first poll tick *after the child reaches `load_data`*. The window before that — parent setup, `Popen`, interpreter start, pandas, and 7 of the repo's 12 module-level store singletons — is **measured** as a retroactive `starting` phase rather than narrated, because no writer exists that early (spec, *Why `starting` is measured and not narrated*). The child also stops repeating the parent's schema DDL, and every pre-loop phase is measured by the same payload that drives the card.

**Architecture:** The engine gains a second progress-file writer, `publish_phase`, beside `_publish_live_progress`; each phase write carries the phase in progress plus the finished phases with timestamps. The parent passes its launch time so the child can account for the gap before it could write anything. `/backtest/status` turns the phase into a sentence; the Backtest panel prints that sentence and the My Agents card prints a short label. The eleven Postgres store twins that own a schema skip `_init_schema()` — through one helper in `db_url.py`, not eleven copies of it — when the parent marks the process as a backtest worker, and that helper times the DDL it does run, so `starting` can say how much of itself was schema.

**Tech Stack:** Python 3 / FastAPI backend (`dashboard.backend` package), vanilla JS frontend lifted into `node -e` by `dashboard/backend/tests/_frontend_source.py`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-20-backtest-speed-and-trust-design.md` (Track A sections).

## Global Constraints

- Run everything from the repo root. Tests: `pytest dashboard/backend/tests/<file> -v`. Node must be on `PATH` for the `test_*card*.py` / `*_frontend.py` modules (they skip, not fail, without it).
- Phase names are exactly `starting`, `loading_bars`, `indicators`, `first_decision`, `running`, `saving`. Nothing else is accepted. `starting` is the one name that is never a *live* `phase` value: the first write to the progress file is `publish_phase("loading_bars")` inside `load_data`, and everything before it runs at module import, before `HourlyBacktester` exists. It exists only so the launch gap has a name inside `phases[]`. Neither `PROGRESS_PHASE_MESSAGES` (Task 3) nor a non-empty `BACKTEST_PHASE_LABELS` entry (Task 4) may claim otherwise — an unreachable sentence reads like a promise the card keeps. The phase *name* stays one name; its **record** is split (Task 1), because one undifferentiated number cannot say which part of the launch an optimisation moved.
- The worker flag is `ATL_BACKTEST_WORKER=1`, set by the parent in the child's environment and nowhere else.
- Every change to `dashboard/frontend/app.js` bumps the `app.js?v=N` pin in **five** files: `dashboard/frontend/app.html` and the four tests found by `grep -rln "app.js?v=" dashboard/backend/tests/*.py`. Grep, never recall the number. `styles.css` is not touched by this plan.
- Commit messages: `type: summary` (feat / fix / test / docs), ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Branch: `feat/backtest-visible-start`, cut from `origin/main` **after** Task 0's PR has merged. Never push to a branch whose PR has merged.

---

### Task 0: Land the #474 branch first

Both tracks edit the status route, the progress card and run metadata. `feat/474-backtest-timeout-outcome` already edits the first two. It merges first so this plan's diff is reviewable on its own.

**Files:** none new. Branch `feat/474-backtest-timeout-outcome`.

- [ ] **Step 1: Confirm no PR exists yet**

Run: `gh pr list --head feat/474-backtest-timeout-outcome --state all`
Expected: empty. If a merged PR is listed, stop: cut a fresh branch for anything further.

- [ ] **Step 2: Rebase onto main**

```bash
git fetch origin main
git switch feat/474-backtest-timeout-outcome
git rebase origin/main
```

Expected: conflicts only in the five cache-buster pin files (`app.html` plus four tests). Resolve each by taking `origin/main`'s numbers **plus one** for both `styles.css?v=` and `app.js?v=` (the branch changes both files). Re-derive the numbers:

```bash
git show origin/main:dashboard/frontend/app.html | grep -o 'styles.css?v=[0-9]*\|app.js?v=[0-9]*'
```

Then `grep -rn "styles.css?v=\|app.js?v=" dashboard/frontend/app.html dashboard/backend/tests/*.py` must show one number per asset everywhere.

- [ ] **Step 3: Run the branch's targeted tests**

Run: `pytest dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/test_backtest_progress_card.py dashboard/backend/tests/test_backtest_cancel.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_backtest_comparison_frontend.py dashboard/backend/tests/test_analytics_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py -q`
Expected: all pass.

- [ ] **Step 4: Push and open the PR**

```bash
git push --force-with-lease origin feat/474-backtest-timeout-outcome
gh pr create --title "feat: explain a timed-out backtest" --body "$(cat <<'EOF'
Closes #474 items 2 and 5. Spec: docs/superpowers/specs/2026-09-14-backtest-timeout-outcome-design.md.

- /backtest/status answers `timed_out` with the budget, lane and settled spend
- the card outlives the 3600s budget and renders the timed-out panel
- Discord watcher outlives the budget too

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: After it merges, cut this plan's branch**

```bash
git fetch origin main
git switch -c feat/backtest-visible-start origin/main
```

---

### Task 1: The engine publishes phases

**Files:**
- Modify: `dashboard/backend/domain/backtesting/engine.py` (the import block at `:17-23`; constructor kwargs at `:180-205` and attribute block at `:236-238`; `_publish_live_progress` at `:588-643`; `load_data` at `:645`; `calculate_indicators` at `:876`; `run_agent_backtest` after `total_steps = len(all_timestamps)` at `:1423` and before the agent `db.insert_run(` at `:1748`)
- Test: `dashboard/backend/tests/backtesting/test_engine_progress_phases.py` (new)

**Interfaces:**
- Produces: `HourlyBacktester(..., launched_at: Optional[float] = None, startup_clock: Optional[Dict[str, float]] = None)`; `HourlyBacktester.publish_phase(name: str, *, total_steps: Optional[int] = None) -> None`; `HourlyBacktester._init_progress_phases(launched_at=None, startup_clock=None) -> None`; module constant `PROGRESS_PHASES: tuple[str, ...]`. Progress-file payload gains `phase: str | None`, `phase_started_at: float | None`, `phases: list[{name, started_at, ended_at}]`. A phase write **before** the loop carries `step: 0`, `total_steps: int` (0 until known) and `equity_curve: []`; a phase write **after** it carries the last live payload forward unchanged except for the phase fields — see Step 4 for why the skeleton is destructive there. The `starting` entry alone also carries `child_entered_at`, `imports_done_at` and `schema_init_seconds` when the script supplies them. Every transition additionally prints one `⏱  phase …` line to stdout.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/backtesting/test_engine_progress_phases.py`:

```python
"""The child publishes its pre-loop phases, so the card is never dark.

Before this, `_publish_live_progress` was the only writer of the progress
file and it ran only inside the bar loop. Everything before bar 1 -- imports,
schema DDL, the bar fetch, aggregation, indicators, then bar 1's full pipeline
-- was invisible to the card and unmeasured. Each phase write here carries the
phase in progress and the finished ones with timestamps, so the payload that
drives the card is also the measurement.
"""
import ast
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import pytz

from dashboard.backend.domain.backtesting import engine as engine_module
from dashboard.backend.domain.backtesting.currency import CurrencyContext
from dashboard.backend.domain.backtesting.engine import (
    PROGRESS_PHASES,
    HourlyBacktester,
)


def _bare(tmp_path, launched_at=None):
    backtester = object.__new__(HourlyBacktester)
    backtester.progress_file = str(tmp_path / "progress.json")
    backtester.live_run_id = "agent_phases"
    # Needed by the cases that call `_publish_live_progress`, not by
    # `publish_phase`. The live writer's payload calls `_serialize_trades`
    # (engine.py:626 -> the def at :412), which resolves the currency context
    # at `:414` -- its second statement, and above the loop over trades
    # rather than inside it, so an empty trades list does not spare it. On a
    # __new__'d instance `_require_currency_context` (`:822`) raises
    # MarketDataUnavailableError. Verified by running it. Without this line
    # four cases below ERROR rather than run, one of them
    # (`test_a_terminal_phase_keeps_the_loops_last_numbers`) the only
    # guard on the `saving`-blanks-the-card regression this whole task exists
    # for -- so the headline fix would ship unpinned while the file looked
    # covered. `tests/backtesting/test_ifind_ashare_engine.py:258` sets the
    # same line for the same reason; this is the established pattern, not a
    # workaround.
    backtester.currency_context = CurrencyContext.identity("USD", "US/Eastern")
    backtester._init_progress_phases(launched_at)
    return backtester


def _payload(tmp_path):
    return json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))


def _manager():
    return SimpleNamespace(
        get_equity_curve=lambda: [],
        trades=[],
        rejected_orders=[],
        order_events=[],
    )


def _manager_with_curve():
    """A manager whose payload is worth losing -- the point of the terminal
    phase test below is that `saving` must not throw this away."""
    return SimpleNamespace(
        get_equity_curve=lambda: [
            {"timestamp": "2026-04-01T14:00:00", "equity": 1000.0, "cash": 0.0,
             "positions_value": 1000.0},
            {"timestamp": "2026-04-01T15:00:00", "equity": 1100.0, "cash": 0.0,
             "positions_value": 1100.0},
        ],
        trades=[],
        rejected_orders=[],
        order_events=[],
    )


def test_first_phase_closes_the_launch_gap(tmp_path):
    launched_at = time.time() - 5
    backtester = _bare(tmp_path, launched_at=launched_at)

    backtester.publish_phase("loading_bars")

    payload = _payload(tmp_path)
    assert payload["run_id"] == "agent_phases"
    assert payload["phase"] == "loading_bars"
    assert payload["step"] == 0
    assert payload["total_steps"] == 0
    assert payload["equity_curve"] == []
    assert [p["name"] for p in payload["phases"]] == ["starting"]
    assert payload["phases"][0]["started_at"] == launched_at
    assert payload["phases"][0]["ended_at"] >= launched_at + 5
    assert payload["phase_started_at"] == payload["phases"][0]["ended_at"]


def test_phases_accumulate_in_order_and_carry_the_bar_count(tmp_path):
    backtester = _bare(tmp_path)

    backtester.publish_phase("loading_bars")
    backtester.publish_phase("indicators")
    backtester.publish_phase("first_decision", total_steps=49)

    payload = _payload(tmp_path)
    assert payload["phase"] == "first_decision"
    assert payload["total_steps"] == 49
    assert [p["name"] for p in payload["phases"]] == ["loading_bars", "indicators"]
    for finished in payload["phases"]:
        assert finished["ended_at"] >= finished["started_at"]


def test_republishing_the_same_phase_does_not_close_it(tmp_path):
    backtester = _bare(tmp_path)
    backtester.publish_phase("loading_bars")
    started = _payload(tmp_path)["phase_started_at"]

    backtester.publish_phase("loading_bars")

    payload = _payload(tmp_path)
    assert payload["phases"] == []
    assert payload["phase_started_at"] == started


def test_live_progress_flips_to_running_and_keeps_the_history(tmp_path):
    backtester = _bare(tmp_path)
    backtester.publish_phase("first_decision", total_steps=49)

    backtester._publish_live_progress(1, 49, _manager())

    payload = _payload(tmp_path)
    assert payload["step"] == 1
    assert payload["total_steps"] == 49
    assert payload["phase"] == "running"
    assert [p["name"] for p in payload["phases"]] == ["first_decision"]

    backtester._publish_live_progress(2, 49, _manager())
    assert [p["name"] for p in _payload(tmp_path)["phases"]] == ["first_decision"]


def test_saving_closes_running(tmp_path):
    backtester = _bare(tmp_path)
    backtester._publish_live_progress(49, 49, _manager())

    backtester.publish_phase("saving")

    payload = _payload(tmp_path)
    assert payload["phase"] == "saving"
    assert payload["phases"][-1]["name"] == "running"
    assert payload["total_steps"] == 49


def test_unknown_phase_is_refused(tmp_path):
    backtester = _bare(tmp_path)
    with pytest.raises(ValueError):
        backtester.publish_phase("warming_up")
    assert PROGRESS_PHASES == (
        "starting",
        "loading_bars",
        "indicators",
        "first_decision",
        "running",
        "saving",
    )


@pytest.mark.parametrize("launched_at", [0, 0.0], ids=["zero-int", "zero-float"])
def test_launch_time_zero_is_a_time_not_an_absence(tmp_path, launched_at):
    """`--launched-at 0` parses to 0.0 through argparse's type=float, and 0.0
    is falsy. A truthiness gate dropped the entire `starting` phase for it --
    the one window this plan can only measure, never narrate -- with no error
    and no log line. The gate is `is not None`, and deliberately nothing more:
    an absurd launch time now yields an absurdly long `starting` row in
    phases[], which is visible and therefore fixable. Range-checking it would
    restore the silent drop under a different name."""
    backtester = _bare(tmp_path, launched_at=launched_at)

    backtester.publish_phase("loading_bars")

    payload = _payload(tmp_path)
    assert [p["name"] for p in payload["phases"]] == ["starting"]
    assert payload["phases"][0]["started_at"] == 0.0


def test_legacy_new_construction_still_publishes_live_progress(tmp_path):
    """Callers that build the engine with __new__ (tests, legacy tools) never
    ran _init_progress_phases; the live writer must not care."""
    backtester = object.__new__(HourlyBacktester)
    backtester.progress_file = str(tmp_path / "progress.json")
    backtester.live_run_id = "agent_legacy"
    # Set here too rather than routed through `_bare`: this case's whole point
    # is that nothing called `_init_progress_phases`, so it cannot use the
    # helper. See the comment there for why `_publish_live_progress` raises
    # without it.
    backtester.currency_context = CurrencyContext.identity("USD", "US/Eastern")

    backtester._publish_live_progress(3, 10, _manager())

    payload = _payload(tmp_path)
    assert payload["step"] == 3
    assert payload["phase"] == "running"
    assert payload["phases"] == []


def test_no_progress_file_means_no_write_and_no_error(tmp_path):
    backtester = object.__new__(HourlyBacktester)
    backtester.progress_file = None
    backtester.live_run_id = None
    backtester._init_progress_phases(None)

    backtester.publish_phase("loading_bars")

    assert not (tmp_path / "progress.json").exists()


def test_a_missing_progress_file_attribute_is_not_an_error(tmp_path):
    """Absent is not None, and only this case covers absent.

    The case above sets `progress_file = None`, which a bare
    `if not self.progress_file` handles fine. Step 6 makes `publish_phase` the
    first statement of `load_data`, and `load_data` is documented as usable on
    an instance built with `__new__` (engine.py:647-648) -- where the attribute
    was never assigned at all, because `__init__` assigns it and the class does
    not carry it. That caller exists today:
    test_market_data_errors.py::test_engine_load_data_empty_raises_not_exits
    sets data_loader, start_date, end_date, data_source and symbols, and
    nothing else. Without the getattr in `publish_phase` it fails with an
    AttributeError that names neither this task nor the
    MarketDataUnavailableError it is actually asserting.
    """
    backtester = object.__new__(HourlyBacktester)

    backtester.publish_phase("loading_bars")

    # The phase clock still ran: it is state, not a side effect of writing.
    assert backtester._progress_phase == "loading_bars"
    assert list(tmp_path.iterdir()) == []


def test_write_failure_is_reported_not_raised(tmp_path, capsys):
    backtester = _bare(tmp_path)
    backtester.progress_file = str(tmp_path / "missing" / "progress.json")

    backtester.publish_phase("loading_bars")

    assert "Could not write live progress" in capsys.readouterr().out


def test_a_terminal_phase_keeps_the_loops_last_numbers(tmp_path):
    """`saving` is published AFTER the loop, and a payload of zeros is a lie
    there.

    The pre-loop skeleton (`step: 0`, `equity_curve: []`) is honest before bar
    1 and destructive after the last one: the Backtest panel computes its bar
    straight off this file (`stepPct`, app.js:8600 -- the 1s poller, one of
    two such sites; Task 4 gives both one helper) and snaps 99% -> 0%, and
    the My Agents fold *replaces* its stored entry (app.js:8609), so the
    sparkline, the equity label and `49/49` all blank at the finish line of
    every run -- for the whole baseline/persistence tail, with nothing red
    anywhere. Asserting `phase == "saving"` cannot see any of that; asserting
    what the payload still carries can.
    """
    backtester = _bare(tmp_path)
    backtester._publish_live_progress(49, 49, _manager_with_curve())

    backtester.publish_phase("saving")

    payload = _payload(tmp_path)
    assert payload["phase"] == "saving"
    assert payload["step"] == 49
    assert payload["total_steps"] == 49
    assert [point["equity"] for point in payload["equity_curve"]] == [1000.0, 1100.0]
    # Keys the skeleton does not have at all: their presence is the proof the
    # last payload was carried rather than rebuilt.
    assert "trades" in payload
    assert "order_events_count" in payload


def test_a_pre_loop_phase_still_publishes_the_skeleton(tmp_path):
    """The other half of the rule above: with nothing published yet there is
    nothing to carry, and inventing a step here would give the card an ETA
    anchor before the loop exists."""
    backtester = _bare(tmp_path)

    backtester.publish_phase("loading_bars")

    payload = _payload(tmp_path)
    assert payload["step"] == 0
    assert payload["equity_curve"] == []
    assert "trades" not in payload


def test_a_file_less_engine_still_closes_first_decision():
    """The phase clock is state, not a side effect of writing.

    `_set_progress_phase` runs *above* `_publish_live_progress`'s
    `if not self.progress_file: return` (engine.py:590). Below that return an
    engine with no progress file -- every CLI run, the external-run session,
    the algo service -- never leaves `first_decision`, and the next transition
    closes a `first_decision` spanning the entire bar loop: the one number this
    phase exists to produce, "how long until the first model decision?",
    reported as hours instead of seconds, on the path where the stdout line is
    the only reader there is.
    """
    backtester = object.__new__(HourlyBacktester)
    backtester.progress_file = None
    backtester.live_run_id = None
    backtester._init_progress_phases(None)
    backtester.publish_phase("first_decision", total_steps=2)

    backtester._publish_live_progress(1, 2, _manager())
    backtester._publish_live_progress(2, 2, _manager())
    backtester.publish_phase("saving")

    assert [p["name"] for p in backtester._progress_phases] == [
        "first_decision",
        "running",
    ]


def test_the_starting_record_splits_what_the_child_could_not_write(tmp_path):
    """One `starting` phase, four numbers -- and MISSING is not zero.

    An undifferentiated `starting` cannot say whether the start is dominated by
    process spawn, by pandas and the SDK imports, or by store construction and
    DDL, so it cannot say whether Task 5 helped. The phase *name* stays one
    name; the *record* carries the boundaries. A mark nobody passed is absent
    rather than 0.0, because an unmeasured phase and a free one are different
    facts. `schema_init_seconds` is the deliberate exception: present and 0.0
    in a worker, which is the evidence the flag fired.
    """
    launched_at = time.time() - 6
    backtester = object.__new__(HourlyBacktester)
    backtester.progress_file = str(tmp_path / "progress.json")
    backtester.live_run_id = "agent_split"
    backtester._init_progress_phases(
        launched_at,
        startup_clock={
            "child_entered_at": launched_at + 0.4,
            "imports_done_at": launched_at + 3.1,
            "schema_init_seconds": 0.0,
        },
    )

    backtester.publish_phase("loading_bars")
    payload = _payload(tmp_path)
    starting = payload["phases"][0]
    assert starting["name"] == "starting"
    assert starting["started_at"] == launched_at
    assert starting["child_entered_at"] == launched_at + 0.4
    assert starting["imports_done_at"] == launched_at + 3.1
    assert starting["schema_init_seconds"] == 0.0
    # Only `starting` carries them: it alone describes an interval the child
    # could not write to. Anywhere else they would be three constants repeated.
    backtester.publish_phase("indicators")
    later = _payload(tmp_path)["phases"][1]
    assert later["name"] == "loading_bars"
    assert set(later) == {"name", "started_at", "ended_at"}

    # An in-process engine passes nothing, so nothing is claimed.
    bare = _bare(tmp_path, launched_at=time.time() - 2)
    bare.publish_phase("loading_bars")
    assert set(_payload(tmp_path)["phases"][0]) == {"name", "started_at", "ended_at"}


def test_every_phase_is_published_where_the_work_happens():
    """The four marking sites of Step 6, pinned -- and nothing else here can.

    Every unit case above builds the engine with `object.__new__` and calls
    `publish_phase` by hand, so not one of them can see whether the *engine*
    ever calls it. Delete all four marks in Step 6 and every one of them stays
    green while the card goes dark again, which is the whole defect this task
    exists to fix.

    Position, not just presence. `load_data`'s mark has to be its first
    statement or the phase opens *after* the fetch it is naming, and a run
    that spends ninety seconds inside `fetch_bars` is exactly the run the card
    is dark for. That is the half the drive test below cannot check: it sees
    that `loading_bars` was published, not that it was published before the
    fetch.

    AST, not `"publish_phase" in source`: a substring test passes on a call
    sitting in a comment, in a docstring, or in a different method entirely.

    This guard and `test_a_fake_loader_drives_every_phase_in_order` below are
    a pair, and the spec's Testing section asks for both halves: this one
    reads the source and can pin *where* a mark sits but never that it runs;
    that one runs the real methods over a fake tape and can pin the order, the
    bar count and the carry but never the position. Neither replaces the
    other. Do not delete this one on the grounds that the drive test covers
    the marks.
    """
    tree = ast.parse(Path(engine_module.__file__).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HourlyBacktester"
    )
    methods = {
        node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)
    }

    def published(method):
        """(phase name, top-level statement index) per self.publish_phase call."""
        found = []
        for index, statement in enumerate(method.body):
            for node in ast.walk(statement):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "publish_phase"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    found.append((node.args[0].value, index))
        return found

    # "First statement" means first after the docstring, which is itself an
    # AST statement. Both methods have one, and Step 6 puts the mark below it.
    def first_body_index(method):
        return 1 if ast.get_docstring(method) is not None else 0

    for name, phase in (
        ("load_data", "loading_bars"),
        ("calculate_indicators", "indicators"),
    ):
        marks = published(methods[name])
        assert marks == [(phase, first_body_index(methods[name]))], (
            f"{name} must publish {phase!r} as its first statement after the "
            f"docstring, and nowhere else; found {marks}"
        )

    # `run_agent_backtest` marks two points inside its body rather than at the
    # top, so order is what can be pinned: `first_decision` only once the bar
    # count exists (there is nothing to report before it), and `saving` before
    # the agent's own row is written, which is the write the phase names.
    run = methods["run_agent_backtest"]
    marks = published(run)
    assert [phase for phase, _ in marks] == ["first_decision", "saving"], marks
    (_, first_decision_at), (_, saving_at) = marks
    total_steps_at = min(
        index
        for index, statement in enumerate(run.body)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "total_steps"
            for target in statement.targets
        )
    )
    insert_run_at = min(
        index
        for index, statement in enumerate(run.body)
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "insert_run"
    )
    assert total_steps_at < first_decision_at < saving_at < insert_run_at, (
        total_steps_at,
        first_decision_at,
        saving_at,
        insert_run_at,
    )


def test_a_fake_loader_drives_every_phase_in_order(tmp_path, monkeypatch):
    """The spec's drive test: the real methods, over a fake tape, in order.

    Spec, Testing/Track A: "a fake loader drives `load_data` ->
    `calculate_indicators` -> the first bar, and the progress file is asserted
    at each phase (`phase`, `phases`, `total_steps` known before the first
    decision)". This is that case; the AST guard above is its other half, for
    the one thing this cannot see (see that docstring).

    Four claims live only here:

    1. The marks are *reached*. The guard proves they are in the source at the
       right statement index; a mark inside a method nothing calls satisfies
       it just as well.
    2. The names arrive in the documented order at runtime, with `running`
       interleaved by `_publish_live_progress` rather than by `publish_phase`
       -- so the order is a property of two writers agreeing, which no
       single-writer unit case above exercises.
    3. The bar count published with `first_decision` is the real one. Source
       order only says `total_steps = ...` precedes the call; it says nothing
       about the value reaching the file, and `0/N` on the card is that value.
       Pinned against the number of decisions the loop actually asks for.
    4. `saving` carries the loop's last payload in a *real* run.
       `test_a_terminal_phase_keeps_the_loops_last_numbers` pins the same rule
       against a hand-built manager, which cannot catch a carry broken by
       something the real loop does to the payload between the last step and
       the terminal mark.

    The fakes follow `tests/backtesting/test_engine_move.py:42-116`, the
    established loader/db pair for driving the real engine with no network and
    no real DB, and the 70-bar tape is the one its own
    `test_run_agent_backtest_smoke` (`:230`) already runs end to end -- 70
    bars in, 70 decision bars out, one decision each. They are copied rather
    than imported: that module imports
    `dashboard.scripts.backtest_hourly_agent` at module scope, and a case that
    goes red when an unrelated file renames a fixture points at the wrong
    place.
    """

    class _Loader:
        def __init__(self, bars):
            self.bars = bars

        def fetch_bars(self, symbols, start_date, end_date):
            return {symbol: self.bars[symbol] for symbol in symbols}

    class _DB:
        def insert_run(self, **kwargs):
            pass

        def insert_equity_points(self, run_id, points):
            pass

        def insert_trades(self, run_id, trades):
            pass

        def insert_decisions(self, run_id, decisions):
            pass

    eastern = pytz.timezone("US/Eastern")
    stamps = []
    day = datetime(2026, 3, 2)  # a Monday
    while len(stamps) < 70:
        if day.weekday() < 5:
            stamps.extend(
                eastern.localize(datetime(day.year, day.month, day.day, hour, 0))
                for hour in range(10, 16)
            )
        day += timedelta(days=1)
    stamps = stamps[:70]
    prices = [100.0 + ((index % 7) - 3) * 0.5 + index * 0.1 for index in range(70)]
    frame = pd.DataFrame(
        {
            "open": prices,
            "high": [price + 1 for price in prices],
            "low": [price - 1 for price in prices],
            "close": prices,
            "volume": [1000] * 70,
        },
        index=pd.DatetimeIndex(stamps),
    )
    monkeypatch.setattr(
        engine_module,
        "create_market_data_provider",
        lambda data_source="alpaca", universe=None: _Loader(
            {"AAPL": frame, "MSFT": frame.copy()}
        ),
    )
    monkeypatch.setattr(engine_module, "db", _DB())

    decisions_asked = []
    at_first_decision = []
    real_decision = engine_module.PortfolioManager.make_trading_decision

    def snapshotting_decision(self, state):
        # The loop asks for the decision (`engine.py:1499`) before it publishes
        # the step (`:1621`), so this hook is the only point from which the
        # `first_decision` payload -- the one the card renders as `0/N` -- can
        # be read while it is still what the file says. It delegates rather
        # than scripting a reply: the subject is the phase machinery, and a
        # scripted decision would quietly change which branches of the loop run.
        decisions_asked.append(state["timestamp"])
        if not at_first_decision:
            at_first_decision.append(_payload(tmp_path))
        return real_decision(self, state)

    monkeypatch.setattr(
        engine_module.PortfolioManager,
        "make_trading_decision",
        snapshotting_decision,
    )

    launched_at = time.time() - 4
    backtester = HourlyBacktester(
        "2026-03-01",
        "2026-04-01",
        "phase-drive",
        use_llm=False,
        symbols=["AAPL", "MSFT"],
        live_run_id="agent_drive",
        progress_file=str(tmp_path / "progress.json"),
        launched_at=launched_at,
    )
    # Construction alone writes nothing: `_init_progress_phases` only opens
    # `starting` in memory, so a run that dies before `load_data` leaves no
    # half-built payload on disk for the poller to read.
    assert not (tmp_path / "progress.json").exists()

    backtester.load_data()
    after_load = _payload(tmp_path)
    assert after_load["run_id"] == "agent_drive"
    assert after_load["phase"] == "loading_bars"
    assert [p["name"] for p in after_load["phases"]] == ["starting"]
    assert after_load["phases"][0]["started_at"] == launched_at
    assert after_load["step"] == 0
    assert after_load["total_steps"] == 0
    assert after_load["equity_curve"] == []

    backtester.calculate_indicators()
    after_indicators = _payload(tmp_path)
    assert after_indicators["phase"] == "indicators"
    assert [p["name"] for p in after_indicators["phases"]] == [
        "starting",
        "loading_bars",
    ]

    run_id, equity_curve = backtester.run_agent_backtest()
    assert run_id and equity_curve

    assert at_first_decision, "the loop never asked for a decision"
    opening = at_first_decision[0]
    assert opening["phase"] == "first_decision"
    assert opening["step"] == 0
    assert [p["name"] for p in opening["phases"]] == [
        "starting",
        "loading_bars",
        "indicators",
    ]
    # Claim 3: the N the card shows before step 1 is the count of steps the
    # loop goes on to take, not a placeholder that happens to be non-zero.
    assert opening["total_steps"] == len(decisions_asked) > 0

    final = _payload(tmp_path)
    assert final["phase"] == "saving"
    assert [p["name"] for p in final["phases"]] == [
        "starting",
        "loading_bars",
        "indicators",
        "first_decision",
        "running",
    ]
    # Claim 4. `step`/`total_steps` survive the terminal mark, and the two keys
    # the pre-loop skeleton does not have at all are the proof the last live
    # payload was carried rather than rebuilt.
    assert final["step"] == final["total_steps"] == opening["total_steps"]
    assert final["equity_curve"]
    assert "trades" in final
    assert "order_events_count" in final
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/backtesting/test_engine_progress_phases.py -v`
Expected: FAIL with `ImportError: cannot import name 'PROGRESS_PHASES'` — a collection error, so every case in the module goes red at once. Three of them go green later than the rest: `test_every_phase_is_published_where_the_work_happens` reads the shipped `engine.py` and `test_a_fake_loader_drives_every_phase_in_order` runs it, so both stay red until **Step 6** adds the four marks; `test_a_missing_progress_file_attribute_is_not_an_error` until Step 4's `getattr`. Steps 3-5 are not expected to clear the file.

The two Step 6 cases fail differently on the way, and the difference is the point: after Steps 3-5 the guard fails on its own assertion (`marks == []`), while the drive test fails on `_payload`'s `FileNotFoundError` — no writer ran, so there is no progress file at all. A drive test that goes green before Step 6 means the fake tape is not reaching `load_data`; check the loader patch before touching the engine.

- [ ] **Step 3: Add the constant, the import and the constructor kwarg**

In `dashboard/backend/domain/backtesting/engine.py`:

Add the wall clock to the `from` block, directly below `from math import ceil` (`:22`), which keeps that block sorted (bisect, datetime, math, time, typing):

```python
# `from time import ...`, not `import time`: line 21's
# `from datetime import date, datetime, time` binds the bare name `time` to
# `datetime.time`, which this module *calls as a constructor* for the A-share
# session bounds (`time(9, 30) <= local_time <= time(11, 30)`, :1209-1210). A
# plain `import time` above it is rebound by that later line, so `time.time()`
# raises AttributeError; below it, those four session bounds silently become
# calls on the time module instead. Green on every US run either way, red only
# on an A-share one.
from time import time as wall_clock
```

Every wall-clock read this task adds is `wall_clock()` — one site, `_set_progress_phase`'s `now = …` in Step 4.

Do not reach for a test to tell you which placement is right. The four `time(...)` calls live in `_market_hours_only` (`:1193`), reached from `run_agent_backtest` at `:1360` and `:1381`, and only on the `profile.market == "CN"` branch — so a bare `import time` placed *below* `:21` type-checks, imports, and passes every US-profile run in the suite, failing only on an A-share backtest. The alias is the placement that cannot be wrong either way.

Next to the other `LIVE_PROGRESS_*` module constants (`grep -n LIVE_PROGRESS_REJECTED_ORDER_LIMIT dashboard/backend/domain/backtesting/engine.py`), add:

```python
#: Every phase the child publishes to the progress file, in the order a run
#: passes through them. `running` is set by `_publish_live_progress`; the rest
#: by `publish_phase`. The status route and the card key on these names.
PROGRESS_PHASES = (
    "starting",
    "loading_bars",
    "indicators",
    "first_decision",
    "running",
    "saving",
)
```

In `HourlyBacktester.__init__` (`:180`), append two keywords after `source_timeframe: Optional[str] = None,`:

```python
        launched_at: Optional[float] = None,
        startup_clock: Optional[Dict[str, float]] = None,
```

(`Dict` is already imported — `:23` is `from typing import Any, Dict, List, Optional, Tuple`.)

`startup_clock` is a plain dict handed in by the caller rather than something the engine reads for itself, and that is deliberate. The three marks it carries (`child_entered_at`, `imports_done_at`, `schema_init_seconds`) describe an interval that ended before this object existed, and only the script can see them. Having the engine read `db_url.schema_init_seconds()` itself would look equivalent and is not: that counter is process-global, so an in-process engine — the external-run session, the algo service, the suite — would publish the *parent's* boot DDL as this run's schema cost. A wrong number, published as confidently as a right one.

Directly after `self.progress_file = (progress_file or "").strip() or None` (`:238`), add:

```python
        self._init_progress_phases(launched_at, startup_clock)
```

- [ ] **Step 4: Add the phase machinery above `_publish_live_progress`**

Insert immediately before `def _publish_live_progress(` (`:588`):

```python
    # -- Progress phases -----------------------------------------------------
    #
    # `_publish_live_progress` is the only writer once the bar loop starts;
    # before it, nothing wrote anything, and the card sat on its launch
    # sentence for the whole start (imports, six stores' DDL, the bar fetch,
    # aggregation, indicators, then bar 1's full pipeline). Each phase write
    # carries the phase in progress plus the finished ones with timestamps, so
    # the same payload that drives the card is the measurement of the start.
    # Every accessor tolerates an instance built with __new__ (tests, legacy
    # tools) that never ran _init_progress_phases -- and `publish_phase`
    # additionally tolerates `progress_file` being *absent* rather than None,
    # because Step 6 makes it the first statement of `load_data`, which is
    # documented as usable on exactly such an instance.

    def _init_progress_phases(
        self,
        launched_at: Optional[float] = None,
        startup_clock: Optional[Dict[str, float]] = None,
    ) -> None:
        """Open the implicit `starting` phase at the parent's launch time.

        The parent alone can see the gap between spawning the child and the
        child's first write (imports, store startup); passing its clock in is
        how that gap becomes a measured phase instead of a missing one. A
        caller that passes nothing (the CLI without --launched-at, tests,
        `__new__` instances) opens no phase at all, so the first
        `publish_phase` records no `starting` entry rather than inventing one
        from the child's own clock -- which would measure zero and read as a
        start that cost nothing.

        `starting` is deliberately ONE phase name -- the card has nothing
        useful to say about spawn vs imports vs stores, and a name the status
        route must translate is a name the frontend must learn. But one
        undifferentiated number cannot justify an optimisation either: it lumps
        process spawn, pandas and three SDK imports, seven store constructions
        (six of them Postgres twins, and only those six run DDL) and that DDL
        into a single figure that moves for reasons nobody can attribute. So
        the *record* is split even though the *phase* is not.
        With the script's two stamps and its reading of db_url's accumulator,
        the `starting` entry yields four numbers from one run:

            spawn + interpreter    = child_entered_at - started_at
            imports incl. stores   = imports_done_at - child_entered_at
            of which schema DDL    = schema_init_seconds   (0.0 in a worker)
            preflight remainder    = ended_at - imports_done_at

        Undo the split and the Final verification table can say that `starting`
        got shorter but not which of those four moved -- which is the same as
        not knowing whether removing the DDL did anything.

        A mark nobody passed is *absent*, not zero: an in-process engine gets a
        plain three-key record. `schema_init_seconds` is the one exception --
        present and 0.0 in a worker, because that zero is the evidence the flag
        fired, not a gap.
        """
        # `is not None`, not truthiness: 0.0 is a launch time, and argparse's
        # `type=float` hands it over intact. A falsy gate dropped the whole
        # `starting` phase for `--launched-at 0` -- deleting the one number
        # this plan exists to produce, with nothing in the file to say so.
        # Deliberately NOT range-checked: `starting` is never a live phase, so
        # an absurd clock lands only in `phases[]`, where it reads as an
        # absurdly long row -- visible, and therefore fixable. A bounds check
        # would turn that back into a silent absence, which is the one outcome
        # this repo never accepts (see IFIND_ALLOW_CORPORATE_ACTION_GAPS in
        # CLAUDE.md: a labelled wrong number, never a silent one). The other
        # malformed forms cannot reach here at all -- argparse's `type=float`
        # rejects an empty string and a BSD `date +%s.%N`'s trailing "N" at the
        # CLI boundary, loudly.
        self._progress_phase: Optional[str] = (
            "starting" if launched_at is not None else None
        )
        self._progress_phase_started_at: Optional[float] = (
            float(launched_at) if launched_at is not None else None
        )
        extra: Dict = {}
        for key in ("child_entered_at", "imports_done_at", "schema_init_seconds"):
            value = (startup_clock or {}).get(key)
            if value is not None:
                extra[key] = float(value)
        self._progress_phase_extra: Dict = extra
        self._progress_phases: List[Dict] = []
        self._progress_total_steps: int = 0

    def _set_progress_phase(
        self, name: str, *, total_steps: Optional[int] = None
    ) -> bool:
        """Close the current phase and open ``name``. True if the phase moved."""
        if name not in PROGRESS_PHASES:
            raise ValueError(f"unknown progress phase: {name!r}")
        if not hasattr(self, "_progress_phases"):
            self._init_progress_phases()
        if total_steps is not None:
            self._progress_total_steps = int(total_steps)
        if self._progress_phase == name:
            return False
        now = wall_clock()
        if self._progress_phase is not None:
            finished = {
                "name": self._progress_phase,
                "started_at": self._progress_phase_started_at,
                "ended_at": now,
            }
            if self._progress_phase == "starting":
                finished.update(getattr(self, "_progress_phase_extra", {}))
            self._progress_phases.append(finished)
            # stdout as well as the file, and from here rather than from
            # main(). The parent unlinks the progress file the moment the run
            # ends (`backtests.py:1810`, inside run_backtest_background's
            # `finally` at `:1771`), so `phases[]` can only be read by racing a
            # live poll; the child's stdout is captured head+tail
            # (SUBPROCESS_LOG_HEAD_CHARS / _TAIL_CHARS, 32k each, `:2299-2300`)
            # and dumped into the parent's log under
            # `=== BACKTEST SCRIPT OUTPUT ===` (`:1628`), where it keeps. It is
            # also the ONLY phase record a CLI run, the external-run session or
            # the algo service has -- none of them writes a progress file.
            # main() could not do this job: it cannot see `first_decision` open
            # and close inside run_agent_backtest, and a second elapsed clock in
            # the script is the two-owners pattern this repo already documents.
            # `is None`, not `or`: `started_at` is 0.0 for the very launch
            # clock this plan exists to measure, and `0.0 or now` would print
            # that phase as having cost nothing -- the same falsy-zero trap
            # _init_progress_phases's gate is about, one line further on.
            opened = finished["started_at"]
            elapsed = finished["ended_at"] - (now if opened is None else opened)
            print(f"⏱  phase {name} (after {finished['name']} {elapsed:.2f}s)", flush=True)
            if finished["name"] == "starting" and {
                "child_entered_at",
                "imports_done_at",
            } <= finished.keys():
                # Printed on this transition, not saved for a summary at the
                # end: `saving` fires before the agent's insert_run with two
                # baseline runs still to print, so a closing table can land in
                # the truncated middle of the parent's bounded capture. These
                # lines are at the head.
                print(
                    f"     spawn+interpreter "
                    f"{finished['child_entered_at'] - finished['started_at']:.2f}s"
                    f" | imports+stores "
                    f"{finished['imports_done_at'] - finished['child_entered_at']:.2f}s"
                    f" (schema DDL {finished.get('schema_init_seconds', 0.0):.2f}s)"
                    f" | preflight "
                    f"{finished['ended_at'] - finished['imports_done_at']:.2f}s",
                    flush=True,
                )
        else:
            print(f"⏱  phase {name}", flush=True)
        self._progress_phase = name
        self._progress_phase_started_at = now
        return True

    def _progress_phase_fields(self) -> Dict:
        if not hasattr(self, "_progress_phases"):
            self._init_progress_phases()
        return {
            "phase": self._progress_phase,
            "phase_started_at": self._progress_phase_started_at,
            "phases": list(self._progress_phases),
        }

    def publish_phase(self, name: str, *, total_steps: Optional[int] = None) -> None:
        """Record a phase transition and, with a progress file, publish it.

        **Before** the loop there is nothing to carry, so the payload is the
        skeleton `step: 0` / `equity_curve: []`, and every existing reader of
        the file (chart, trading log, ETA anchor) keeps its early return while
        only the message and the bar count change.

        **After** the loop there is, and writing that skeleton over it was a
        real regression rather than a cosmetic one. `saving` fires at the end
        of a 49-bar run: a payload of `step: 0, equity_curve: []` snaps the
        Backtest panel's bar from 99% to 0 -- `stepPct` is computed straight
        off this file's `step`/`total_steps` (`app.js:8600`, the 1s poller;
        `attachToLiveBacktest` at `:8432` holds a byte-identical second copy,
        which Task 4 replaces with one shared helper) and never passes through
        the fold, so no frontend guard can reach it -- and the My
        Agents fold *replaces* its stored entry (`app.js:8609`), blanking the
        sparkline, the equity label and `49/49` for the whole
        baseline/persistence tail, with nothing red anywhere. So a phase write
        carries the last published payload forward and changes only the phase
        fields and the bar count: the file never says less than it last said.

        The carry is what makes the fix hold at the source. Task 4 Step 6 adds
        a second, independent guard in the browser for any writer that still
        publishes a bare phase tick after real progress; neither is a substitute
        for the other, because the panel's bar bypasses the fold and the fold
        outlives this engine's payload shape.
        """
        self._set_progress_phase(name, total_steps=total_steps)
        # `getattr`, not `self.progress_file`. Step 6 makes this method the
        # first statement of `load_data`, and `load_data` is documented as
        # usable on an instance built with `__new__` (its own comment,
        # engine.py:647-648) -- so this read is now the first attribute such a
        # caller touches. `progress_file` is assigned in `__init__` and is not
        # a class attribute, so on that instance it is *absent*, and a bare
        # read raises AttributeError rather than returning None. Verified by
        # running it. The one such caller in the suite is
        # test_market_data_errors.py::test_engine_load_data_empty_raises_not_exits,
        # which sets five attributes and not this one; it would report an
        # AttributeError naming neither this task nor its own subject, in
        # place of the MarketDataUnavailableError it asserts.
        if not getattr(self, "progress_file", None):
            return
        last = getattr(self, "_progress_last_payload", None) or {}
        payload = dict(last)
        payload.update(
            {
                "run_id": self.live_run_id,
                # `last.get(...)`, not `self._progress_...`: the step and the
                # curve belong to the loop, and re-deriving them here would be
                # a second owner for numbers _publish_live_progress already
                # published. Absent (pre-loop) they default to the skeleton.
                "step": int(last.get("step") or 0),
                "total_steps": self._progress_total_steps,
                "equity_curve": last.get("equity_curve") or [],
                **self._progress_phase_fields(),
            }
        )
        self._write_progress_payload(payload)

    def _write_progress_payload(self, payload: Dict) -> None:
        from pathlib import Path

        # Remembered before the write, not after it: this is the payload this
        # process intends the file to hold, and a phase published after a failed
        # write must still carry the loop's numbers rather than silently fall
        # back to the skeleton. One reference to data the manager already holds.
        self._progress_last_payload: Dict = payload
        try:
            Path(self.progress_file).write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            print(f"   ⚠️  Could not write live progress: {exc}")
```

- [ ] **Step 5: Make `_publish_live_progress` set `running` and share the writer**

In `_publish_live_progress` (`:588`), make the phase transition the **first statement of the method, above** the existing early return. Verified at source: `:588-591` is the `def`, the docstring, then `if not self.progress_file: return`.

```python
    def _publish_live_progress(self, step: int, total_steps: int, manager) -> None:
        """Write incremental equity curve snapshots for live dashboard charting."""
        # Above the early return, not below it. The phase clock is state, not a
        # side effect of writing, and the progress file is only one of its
        # readers. Below the return an engine with no progress file -- every
        # CLI run, the external-run session, the algo service -- never leaves
        # `first_decision`, so `publish_phase("saving")` closes a
        # `first_decision` spanning the entire bar loop: the one number this
        # phase exists to produce, published as hours instead of seconds, in
        # the row of the table that is supposed to justify the whole track.
        # `publish_phase` is already the right way round; this is the only
        # asymmetry. `_set_progress_phase`'s hasattr guard self-initialises, so
        # a `__new__`-built instance is unaffected.
        self._set_progress_phase("running", total_steps=total_steps)
        if not self.progress_file:
            return
```

In its `payload = {` literal, add the phase fields as the last entry, after `"order_events_count": len(unfilled_order_events),`:

```python
            **self._progress_phase_fields(),
```

Replace the trailing write

```python
        try:
            Path(self.progress_file).write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            print(f"   ⚠️  Could not write live progress: {exc}")
```

with:

```python
        self._write_progress_payload(payload)
```

and delete the now-unused `from pathlib import Path` at the top of that method.

- [ ] **Step 6: Mark the phases where the work happens**

`load_data` (`:645`): make `self.publish_phase("loading_bars")` the first statement of the body, before the `symbols = getattr(self, "symbols", ())` line.

`calculate_indicators` (`:876`): make `self.publish_phase("indicators")` the first statement, before the `print("\n📈 Calculating technical indicators...")`.

`run_agent_backtest`: directly after `total_steps = len(all_timestamps)` (`:1423`), add:

```python
        self.publish_phase("first_decision", total_steps=total_steps)
```

Still in `run_agent_backtest`, immediately before the `db.insert_run(` call that writes the agent's own row (the first of three `db.insert_run(` calls in the file, `:1748`, re-derived 2026-09-20; the two later ones at `:1859` and `:1941` write baselines — there is no `save` method to hang this on), add:

```python
        self.publish_phase("saving")
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/backtesting/test_engine_progress_phases.py dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py dashboard/backend/tests/test_market_data_errors.py dashboard/backend/tests/test_agent_runs_metadata.py -v`
Expected: all pass.

`test_ifind_ashare_engine.py` is in the list because it drives `_publish_live_progress` on a `__new__`-built instance (`:286`, built at `:255`). `test_market_data_errors.py` is in it because of **Step 6**, not Step 4: `test_engine_load_data_empty_raises_not_exits` (`:46-58`) calls `load_data` on a `__new__` instance that sets `data_loader`, `start_date`, `end_date`, `data_source` and `symbols` — and not `progress_file` — so it is the one existing case in the suite that a `publish_phase` at the top of `load_data` can break, and it breaks with an `AttributeError` that names neither this task nor the error the test is asserting. Re-derived 2026-09-21: it is the *only* such caller. Every `load_data()` call in `test_ifind_ashare_engine.py` is on a fully constructed engine, and no other module calls `load_data` or `calculate_indicators` on a `__new__` instance.

- [ ] **Step 8: Commit**

```bash
git add dashboard/backend/domain/backtesting/engine.py dashboard/backend/tests/backtesting/test_engine_progress_phases.py
git commit -m "feat: publish pre-loop phases to the backtest progress file

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The parent passes its launch time; the child accepts it

**Files:**
- Modify: `dashboard/scripts/backtest_hourly_agent.py` (module preamble above `import sys` at `:19`; a second stamp after the last module-level import, `get_model_provider_service` at `:178`, above the `# Main` banner at `:181-183`; argparse block around `:223-224`; the `HourlyBacktester(` call at `:443`)
- Modify: `dashboard/backend/api/routers/backtests.py` — two sites inside `run_backtest_background` (`:1434`): the `progress_file = str(` → `_update_slot(` block at `:1499-1509` (one launch clock, stamped once) and the `--run-id` / `--progress-file` argv line at `:1594`
- Test: `dashboard/backend/tests/test_backtest_launch_phases.py` (new)

**Interfaces:**
- Consumes: `HourlyBacktester(launched_at=..., startup_clock=...)` from Task 1.
- Produces: child argv `--launched-at <epoch seconds, 3 decimals>`, always present for dashboard launches, carrying the *same* clock the run slot's `started_at` carries — one `time.time()` reading spent twice; module constants `CHILD_ENTERED_AT` / `IMPORTS_DONE_AT` stamped either side of the script's import block; `HourlyBacktester(startup_clock={...})`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_backtest_launch_phases.py`:

```python
"""The parent tells the child when it was launched.

The child's first progress write can only account for the gap before it
(imports, store startup) if it knows when the parent spawned it. That clock
rides argv beside --run-id, so a run's `starting` phase is measured rather
than missing.
"""
import os
import subprocess
import sys
import time

import dashboard.backend.api.routers.backtests as backtests
from dashboard.backend.tests._fake_child import FakeChild

REAL_RUN_BACKTEST_BACKGROUND = backtests.run_backtest_background


def _launch(monkeypatch):
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        return FakeChild()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(backtests.db, "get_runs_by_mode", lambda mode: [])
    before = time.time()
    REAL_RUN_BACKTEST_BACKGROUND(
        "2026-04-01",
        "2026-04-08",
        "session-id",
        decision_source="rule_based",
    )
    captured["before"] = before
    captured["after"] = time.time()
    return captured


def test_child_argv_carries_the_launch_time(monkeypatch):
    captured = _launch(monkeypatch)
    command = captured["command"]
    launched_at = float(command[command.index("--launched-at") + 1])
    assert captured["before"] - 1 <= launched_at <= captured["after"] + 1


def test_script_accepts_launched_at(tmp_path):
    result = subprocess.run(
        [sys.executable, "dashboard/scripts/backtest_hourly_agent.py", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "DATABASE_PATH": str(tmp_path / "backtest.db")},
    )
    assert result.returncode == 0, result.stderr
    assert "--launched-at" in result.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_backtest_launch_phases.py -v`
Expected: FAIL — `ValueError: '--launched-at' is not in list` and `assert '--launched-at' in ...`.

- [ ] **Step 3: Add the script argument and pass it to the engine**

First, stamp both ends of the import block — `starting` is one phase, but it must not be one number. At the very top of `dashboard/scripts/backtest_hourly_agent.py`, directly below the module docstring and **above** `import sys` (`:19`), add:

```python
import time

#: Wall clock at the first executable statement of this process. With the
#: parent's `--launched-at` this brackets the cost the child cannot otherwise
#: see -- fork/exec, interpreter boot, site-packages. Taken before the imports
#: below because *they* are the next thing to account for, and one mark cannot
#: tell the two apart. Wall clock, not perf_counter: it has to be comparable
#: with a timestamp the parent took in another process. `time` is a builtin C
#: module, so this stamp costs nothing it measures.
CHILD_ENTERED_AT = time.time()
```

Directly after the last module-level import, `from dashboard.backend.domain.model_providers.service import get_model_provider_service` (`:178`), and above the `# Main` banner (`:181-183`), add:

```python
#: Wall clock once every import above has run -- pandas, three SDKs, and the
#: seven store singletons those imports construct as a side effect (six of them
#: Postgres-capable). The gap to CHILD_ENTERED_AT is the number Task 5 moves a
#: part of; without it, `starting` cannot say which part.
IMPORTS_DONE_AT = time.time()
```

Then the argument itself. In `dashboard/scripts/backtest_hourly_agent.py`, directly after the `--progress-file` `add_argument` (`:224`), add:

```python
    parser.add_argument(
        "--launched-at",
        type=float,
        default=None,
        help=(
            "Epoch seconds at which the parent launched this process. The child "
            "records the gap before its first progress write (imports, store "
            "startup) as its 'starting' phase."
        ),
    )
```

In the `backtester = HourlyBacktester(` call (`:443`), add after `execution_client=execution_client,` (`:461`):

```python
        launched_at=args.launched_at,
        # Read here rather than inside the engine on purpose. These marks
        # describe an interval that ended before the engine existed, and the
        # third one Task 5 adds (`db_url.schema_init_seconds()`) is
        # process-global: an engine that read it for itself would report the
        # PARENT's boot DDL as this run's schema cost on every in-process path
        # (the external-run session, the algo service, the suite). Handed in,
        # an in-process engine passes nothing and the key is simply absent.
        startup_clock={
            "child_entered_at": CHILD_ENTERED_AT,
            "imports_done_at": IMPORTS_DONE_AT,
        },
```

`schema_init_seconds` is deliberately **not** added here. `db_url.schema_init_seconds()` does not exist until Task 5 Step 3, and a plan whose intermediate commit crashes the CLI on its first real run is a plan nobody can bisect. Task 5 Step 3 adds the third key once the accumulator exists; until then the engine's "a mark nobody passed is absent, not zero" rule covers the gap, which is the same rule that covers every in-process engine forever.

- [ ] **Step 4: Stamp one launch clock in the parent and spend it twice**

First, hoist the clock. In `run_backtest_background`, replace (`:1499-1509`)

```python
        progress_file = str(
            Path(tempfile.gettempdir()) / f"backtest_progress_{resolved_live_run_id}.json"
        )
        _update_slot(
            resolved_live_run_id,
            running=True,
            error=None,
            started_at=time.time(),
```

with:

```python
        progress_file = str(
            Path(tempfile.gettempdir()) / f"backtest_progress_{resolved_live_run_id}.json"
        )
        # One reading, spent twice: the slot's `started_at` (what the card's
        # elapsed counter counts from) and the child's `--launched-at` below.
        # Stamped HERE rather than at the argv line so `starting` covers the
        # whole window the card cannot narrate -- this function's own setup,
        # Popen, interpreter start, and the child's module imports. Two
        # time.time() calls would leave everything between here and :1594
        # (venv probe, env copy, temp-file writes, argv build) inside the
        # card's elapsed clock but outside the one phase duration this plan
        # exists to produce, which is the direction that under-reports.
        launched_at = time.time()
        _update_slot(
            resolved_live_run_id,
            running=True,
            error=None,
            started_at=launched_at,
```

Then stamp it in argv. In `run_backtest_background` (`dashboard/backend/api/routers/backtests.py:1594`), replace

```python
        cmd += ["--run-id", resolved_live_run_id, "--progress-file", progress_file]
```

with:

```python
        cmd += [
            "--run-id", resolved_live_run_id,
            "--progress-file", progress_file,
            # The child cannot see the gap before its own first write; this is
            # how that gap becomes its measured `starting` phase.
            "--launched-at", f"{launched_at:.3f}",
        ]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtest_launch_phases.py dashboard/backend/tests/test_market_data_features.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add dashboard/scripts/backtest_hourly_agent.py dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtest_launch_phases.py
git commit -m "feat: hand the child its launch time for the starting phase

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: The status route names the phase

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py` (new helper beside `_read_progress_file` at `:1381`; the running branch of `get_backtest_status` at `:3459-3465`)
- Test: `dashboard/backend/tests/test_backtest_progress_status.py` (append), `dashboard/backend/tests/test_backtests_router.py` (append after `test_backtest_status_includes_live_progress` at `:1443` — the post-Task-0 number; that file is `:1325` on `origin/main`)

**Interfaces:**
- Consumes: progress payload fields `phase`, `step`, `total_steps` from Task 1.
- Produces: `PROGRESS_PHASE_MESSAGES: dict[str, str]` (four entries — `starting` and `running` are deliberately absent) and `_progress_message(progress: Optional[Dict[str, Any]]) -> str` in `backtests.py`. `/backtest/status` `message` becomes the phase sentence while `step == 0`, and also for `saving`, which is the one phase published after the loop and therefore the one that arrives with a step count of its own.

- [ ] **Step 1: Write the failing helper tests**

Append to `dashboard/backend/tests/test_backtest_progress_status.py`:

```python
import pytest


_DEFAULT = "Backtest is running… (multi-step agent pipeline; may take several minutes)"


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        ("loading_bars", "Loading market data…"),
        ("indicators", "Calculating indicators…"),
        ("first_decision", "Waiting on the first model decision…"),
        ("saving", "Saving results…"),
    ],
)
def test_phase_sentence(phase, expected):
    assert backtests._progress_message({"step": 0, "total_steps": 0, "phase": phase}) == expected


def test_first_decision_names_the_queued_bars_once_known():
    assert (
        backtests._progress_message({"step": 0, "total_steps": 49, "phase": "first_decision"})
        == "Waiting on the first model decision… (49 decision bars queued)"
    )


def test_a_real_step_wins_over_the_phase():
    assert (
        backtests._progress_message({"step": 5, "total_steps": 100, "phase": "running"})
        == "Backtest running… step 5/100 (5%)"
    )


def test_legacy_payload_without_a_phase_keeps_the_generic_sentence():
    assert backtests._progress_message({"step": 0, "total_steps": 49}) == _DEFAULT
    assert backtests._progress_message(None) == _DEFAULT


def test_unknown_phase_names_nothing_it_cannot_explain():
    assert backtests._progress_message({"step": 0, "total_steps": 0, "phase": "warp"}) == _DEFAULT


def test_saving_wins_over_the_step_count_it_arrives_with():
    """`saving` is the one phase published AFTER the loop, and Task 1 makes its
    payload carry the loop's final numbers rather than zeros. A step-first rule
    would therefore freeze the panel on "step 49/49 (99%)" for the whole
    baseline/persistence tail and leave the `saving` sentence unreachable -- a
    dead entry in the table, which is exactly what the `starting` guard below
    exists to prevent."""
    assert (
        backtests._progress_message({"step": 49, "total_steps": 49, "phase": "saving"})
        == "Saving results…"
    )


def test_starting_has_no_sentence_because_nothing_publishes_it():
    """`starting` names the launch gap inside `phases[]`; it is never a live
    `phase`. The first write is publish_phase("loading_bars") at the top of
    load_data, and everything before it -- parent setup, Popen, interpreter
    start, pandas, the module-level store singletons -- runs before
    HourlyBacktester exists. A sentence here would be unreachable code that
    reads like a promise the card keeps. Add one only in the same commit as a
    writer that runs before the imports do, and only if it has something to
    say that "Starting backtest…" does not."""
    from dashboard.backend.domain.backtesting.engine import PROGRESS_PHASES

    assert "starting" in PROGRESS_PHASES
    assert "starting" not in backtests.PROGRESS_PHASE_MESSAGES
    assert (
        backtests._progress_message({"step": 0, "total_steps": 0, "phase": "starting"})
        == _DEFAULT
    )


def test_every_sentence_names_a_phase_the_engine_knows():
    """A subset check, not a count. The table holds 4 entries against
    PROGRESS_PHASES' 6 -- `running` and `starting` are both absent, for
    different reasons -- so an equality or a len() here would go red on a
    correct addition and teach the next author to delete the guard."""
    from dashboard.backend.domain.backtesting.engine import PROGRESS_PHASES

    assert set(backtests.PROGRESS_PHASE_MESSAGES) <= set(PROGRESS_PHASES)
```

Append to `dashboard/backend/tests/test_backtests_router.py`, directly after `test_backtest_status_includes_live_progress`:

```python
def test_backtest_status_names_the_pre_loop_phase(tmp_path):
    progress_file = tmp_path / "progress.json"
    progress_file.write_text(json.dumps({
        "run_id": "agent_phase",
        "step": 0,
        "total_steps": 49,
        "equity_curve": [],
        "phase": "first_decision",
        "phase_started_at": time.time(),
        "phases": [{"name": "loading_bars", "started_at": 1.0, "ended_at": 2.0}],
    }), encoding="utf-8")
    bt.backtest_status.update({
        "running": True,
        "error": None,
        "started_at": time.time(),
        "progress_file": str(progress_file),
        "live_run_id": "agent_phase",
    })
    resp = TestClient(app).get("/backtest/status", headers=_sess())
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "Waiting on the first model decision… (49 decision bars queued)"
    assert body["progress"]["phase"] == "first_decision"
    assert body["progress"]["phases"][0]["name"] == "loading_bars"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_backtest_progress_status.py dashboard/backend/tests/test_backtests_router.py::test_backtest_status_names_the_pre_loop_phase -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_progress_message'` and the router test failing on the message.

- [ ] **Step 3: Add the helper**

In `dashboard/backend/api/routers/backtests.py`, directly above `def _read_progress_file(` (`:1381`), add:

```python
#: Sentence per pre-loop phase (engine.py PROGRESS_PHASES). Built here rather
#: than in the browser for the same reason the staleness age is: both ends of
#: the phase clock are read in this process. Four entries against
#: PROGRESS_PHASES' six; two names are absent on purpose. `running` -- a step
#: count owns that sentence. `starting` -- nothing can publish it: the child's
#: first write is publish_phase("loading_bars") inside load_data, and the whole
#: launch (parent setup, Popen, interpreter, pandas, the module-level store
#: singletons) happens before HourlyBacktester exists. `starting` is the
#: retroactive name of that gap in `phases[]`, and the card keeps its existing
#: "Starting backtest…" plus the startup-staleness notice while it lasts.
#: Adding a sentence here does not make the card say it.
PROGRESS_PHASE_MESSAGES = {
    "loading_bars": "Loading market data…",
    "indicators": "Calculating indicators…",
    "first_decision": "Waiting on the first model decision…",
    "saving": "Saving results…",
}

_PROGRESS_DEFAULT_MESSAGE = (
    "Backtest is running… (multi-step agent pipeline; may take several minutes)"
)


def _progress_message(progress: Optional[Dict[str, Any]]) -> str:
    """The one sentence the card prints for a running backtest.

    A real step wins: once the loop has published, the phase is `running` and
    the count is the news. Before that, the phase names what the child is
    doing. A payload with neither (an older child, or a phase this build does
    not know) gets the generic sentence rather than a guess.
    """
    if not progress:
        return _PROGRESS_DEFAULT_MESSAGE
    step = int(progress.get("step") or 0)
    total = int(progress.get("total_steps") or 0)
    phase = str(progress.get("phase") or "")
    # `saving` is the one phase that outranks the step count, because it is the
    # one published AFTER the loop -- and since Task 1 a terminal phase write
    # carries the loop's final numbers forward instead of zeroing them. Leave
    # the count first and the panel freezes on "step 49/49 (99%)" for the whole
    # baseline/persistence tail while the run is demonstrably doing something
    # else, and the "Saving results…" entry below becomes unreachable: a
    # sentence in a table that nothing can ever print, which is the same defect
    # the `starting` guards in the tests exist to prevent.
    if phase != "saving" and step > 0 and total > 0:
        pct = min(99, round(100 * step / total))
        return f"Backtest running… step {step}/{total} ({pct}%)"
    message = PROGRESS_PHASE_MESSAGES.get(phase)
    if message is None:
        return _PROGRESS_DEFAULT_MESSAGE
    if total > 0 and phase == "first_decision":
        return f"{message} ({total} decision bars queued)"
    return message
```

- [ ] **Step 4: Use it in the running branch**

In `get_backtest_status`, replace (`:3459-3465` — the `message = …` line through the `f"Backtest running… step …"` assignment; `:3466` is `payload = {` and stays)

```python
        message = "Backtest is running… (multi-step agent pipeline; may take several minutes)"
        if progress:
            step = int(progress.get("step") or 0)
            total = int(progress.get("total_steps") or 0)
            if total > 0:
                pct = min(99, round(100 * step / total))
                message = f"Backtest running… step {step}/{total} ({pct}%)"
```

with:

```python
        message = _progress_message(progress)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtest_progress_status.py dashboard/backend/tests/test_backtests_router.py -v`
Expected: all pass. `test_backtest_status_includes_live_progress` still sees `step 5/100`.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtest_progress_status.py dashboard/backend/tests/test_backtests_router.py
git commit -m "feat: name the pre-loop phase in /backtest/status

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The cards show the phase

The Backtest panel's 1 s **poller** already prints `status.message`, so Task 3 reaches that one surface with no change. Two things around it do need one, and an earlier draft of this preamble said neither did.

**There are two readers of the raw `/backtest/status` payload, not one.** `attachToLiveBacktest` (`app.js:8382`) is handed `statusProgress` — the same raw payload, assigned at `:11571` and passed at `:11592`, on the dropdown / deep-link / reload-mid-run path — and builds its **own** message, ignoring `status.message` entirely (`:8436-8438`). Left alone it greets the new pre-loop payload with *Backtest running… step 0/49 (0%)*.

**And both of those sites gate the bar on `total > 0` alone, never `step > 0`** (`:8432`, `:8600`). `100 * 0 / 49` is a finite `0`, and `updateBacktestRunProgress` treats any finite `stepPct` as authoritative over its elapsed-based creep (`:8228-8231`), so it writes `width: 0%`. Today this is invisible: no progress file exists before the first step, `status.progress` is absent, `Number(undefined)` is `NaN` and the creep survives. After Task 1 the file exists from `loading_bars` — harmless while `total_steps` is still 0, and then the moment `first_decision` publishes the bar count the panel's bar **snaps back from the creep to a flat 0%** and sits there until the first real step. An empty track with nothing moving in it is exactly the *reads as broken* outcome the indeterminate-bar rule exists to prevent, reintroduced on the surface this preamble had written off as needing no change. Step 4 gives both sites one shared owner for that rule; Step 2 pins it at both ends.

Beyond that, this task covers the My Agents card (one short line), the two progress-store functions both cards derive from, and the staleness notice they share — which until now hardcoded a cause (*long model steps*) that is false for every phase this task makes visible.

**Files:**
- Modify: `dashboard/frontend/app.js` — add `BACKTEST_PHASE_LABELS`, `formatBacktestPhase` and `backtestStepPercent` above the `/**` that opens `formatStartupStaleness`'s doc comment (`:7888`, **not** the `function` line at `:7901` — Step 4 says why); edit `formatProgressStaleness` (`:7881`), `resolveRunningNotice` (`:7909`), `advanceBacktestProgress` (`:7932`), `deriveRunningProgress` (`stepLabel` at `:8058`, `detail` at `:8063`), and the **two raw-payload bar sites** — `attachToLiveBacktest`'s inline `stepPct` and message (`:8432`, `:8436-8438`, inside the function at `:8382`) and the poller's inline `stepPct` (`:8600`), both in Step 4. ⚠ Those anchors are `origin/main`'s, re-derived 2026-09-20; **Task 0's rebase merges app.js from both sides**, so not one of them survives it exactly. Grep for the `function <name>(` line, never jump to the number. The two `stepPct` sites are found with `grep -n "const stepPct = Number.isFinite" dashboard/frontend/app.js` — exactly two hits, and they are byte-identical to each other, which is how both came to carry the same defect.
- Not modified: `updateBacktestRunProgress` (`:8198`) itself. Its `Number.isFinite(stepPct)` branch (`:8228-8231`) is correct — a finite percentage *should* outrank the elapsed guess — and the bug is that its two callers hand it a finite `0`. Fixing it in the consumer would make a genuine 0% unrepresentable and leave the wrong number still being computed twice.
- Modify: the five `app.js?v=` pins
- Test — **every** node harness that lifts one of the four *liftable* functions this task changes (`formatProgressStaleness`, `resolveRunningNotice`, `advanceBacktestProgress`, `deriveRunningProgress` — `attachToLiveBacktest` and the poller change too but neither is liftable, which is why Step 2 pins them by source shape instead), because a harness missing `BACKTEST_PHASE_LABELS` or `formatBacktestPhase` throws `ReferenceError` and `assert result.returncode == 0` fires. Enumerated with `grep -rn "function resolveRunningNotice(\|function advanceBacktestProgress(\|function deriveRunningProgress(\|function formatProgressStaleness(" dashboard/backend/tests/` — re-run it rather than trusting this list:
  - `dashboard/backend/tests/test_backtest_progress_format.py` — `_HELPERS` (`:31`) and its one builder `_eval` (`:41`)
  - `dashboard/backend/tests/test_agent_card_live_chart.py` — `_SPARK_HELPERS` (`:46`), `_fold` (`:81`), `_render_raw` (`:151`); append tests
  - `dashboard/backend/tests/test_backtest_progress_card.py` — `_PROGRESS_HELPERS` (`:50`) and the three builders that spread it, `_render` (`:75`), `_patch` (`:271`) and `_run_panel` (`:507`), plus `_advance` (`:443`) which lifts `advanceBacktestProgress` on its own; append tests. (Numbers from the post-Task-0 tree, where this file differs from `origin/main`.)
  - Not touched, and for three different reasons — check the reason, not the file name, before adding a fifth builder. `test_running_backtest_store.py` *stubs* `deriveRunningProgress` (`:431`) rather than lifting it. `test_backtest_cancel_frontend.py` and `test_onboarding_checklist.py` do run node harnesses, but they lift only functions this task leaves alone (`renderAgentRunningActions` at `:45`; `agentRunCount` and `deriveOnboardingChecklist` at `:70-71`) — **not** because nothing in those modules executes. And the remaining `fn_body(...)` uses across all three, plus several in `test_backtest_progress_card.py` itself, are source-text guards that read the slice as a string. Two of those do read `advanceBacktestProgress` and the poller after Step 4 and Step 6 rewrite them (`test_backtest_progress_card.py:763-767` asserts `progress_age_seconds` is still read and `progress_updated_at` still is not; `:743-753` asserts the per-run progress assignment still precedes `refreshRunningAgentCards()`), which both edits preserve — that is why that module is in Step 9's run list rather than merely unedited.

**Interfaces:**
- Consumes: progress payload `phase`, `phase_started_at`, `progress_age_seconds`, `step`, `total_steps`.
- Produces: `const BACKTEST_PHASE_LABELS`; `function formatBacktestPhase(phase) -> string`; `function backtestStepPercent(progress) -> number | null` (the raw-payload bar percentage, `null` until `step > 0`), now the single owner of a rule the two call sites each held a copy of; `formatProgressStaleness(secondsSinceUpdate, phase)` gains a phase-chosen cause clause; `advanceBacktestProgress` now returns `{step: 0, totalSteps, phase, phaseStartedAt, equityCurve: [], openingEquity: null, ageSeconds, ageAt}` for a phase tick (no `firstStep`/`firstStepAt` keys), carries `phase` on the ordinary path when the payload names one, and carries a late phase tick onto the stored entry instead of replacing it; `deriveRunningProgress` fills `stepLabel` with `0/N` and `detail` with the phase label before the first step.
- `determinate` is untouched and stays false at step 0 (`app.js:8005`): it gates `pct` and the ETA, both meaningless there, and the renderer paints a determinate bar by setting an explicit width (`app.js:6299-6303`), so a determinate 0% bar would be an empty track with the indeterminate sweep switched off.

- [ ] **Step 1: Write the failing fold tests**

In `dashboard/backend/tests/test_agent_card_live_chart.py`, change `_fold` (`:81`) to lift the new helper. `js_const` matches `^const NAME = [^;]+;`, and `[^;]` spans newlines, so the multi-line labels literal lifts whole — as long as no value in it ever contains a `;`, which would truncate the declaration into invalid JS and fail loudly under node rather than quietly.

```python
def _fold(progress_js: str, previous_js: str = "null") -> object:
    script = "\n".join(
        [
            js_const("LIVE_SPARK_MAX_POINTS"),
            js_const("BACKTEST_PHASE_LABELS"),
            fn_body("function formatBacktestPhase("),
            fn_body("function advanceBacktestProgress("),
            f"console.log(JSON.stringify(advanceBacktestProgress("
            f"{previous_js}, {progress_js}, Date.now())));",
        ]
    )
    return _node(script)
```

Append after `test_fold_still_refuses_a_tick_with_no_step`:

```python
def test_fold_accepts_a_phase_tick_without_a_step():
    """A pre-loop phase folds so the card can name it, but without an ETA
    anchor: firstStep/firstStepAt stay absent so the first real step still
    sets the rate and the launch-biased ETA cannot come back."""
    folded = _fold(
        "{step: 0, total_steps: 49, phase: 'first_decision',"
        " phase_started_at: 1700000000, progress_age_seconds: 4}"
    )
    assert folded["step"] == 0
    assert folded["totalSteps"] == 49
    assert folded["phase"] == "first_decision"
    assert folded["phaseStartedAt"] == 1700000000
    assert folded["ageSeconds"] == 4
    assert folded["equityCurve"] == []
    assert "firstStep" not in folded
    assert "firstStepAt" not in folded


def test_fold_ignores_a_phase_it_cannot_name():
    assert _fold("{step: 0, total_steps: 49, phase: 'warp'}") is None


def test_first_real_step_anchors_after_a_phase_tick():
    previous = (
        "{step: 0, totalSteps: 49, phase: 'first_decision',"
        " ageSeconds: 4, ageAt: Date.now()}"
    )
    folded = _fold("{step: 1, total_steps: 49}", previous_js=previous)
    assert folded["firstStep"] == 1
    assert "phase" not in folded


def test_fold_refuses_a_starting_tick():
    """`starting` carries an empty label, so it folds like any phase the card
    cannot name. Nothing publishes it today; this pins that adding a publisher
    without adding a label is a clean no-op rather than a blank phase line."""
    assert _fold("{step: 0, total_steps: 0, phase: 'starting'}") is None


def test_fold_refuses_an_inherited_object_key_as_a_phase():
    """`BACKTEST_PHASE_LABELS[phase]` on a plain object literal answers for
    every key on Object.prototype: `constructor` returns a *function*, which is
    truthy, so the guard above would accept the tick and the card's detail line
    would interpolate a function body. `warp` cannot catch that -- it is a
    miss, and the prototype keys are hits. Hence hasOwnProperty in the helper.
    """
    for key in ("constructor", "toString", "valueOf", "hasOwnProperty"):
        assert _fold(f"{{step: 0, total_steps: 49, phase: '{key}'}}") is None


def test_fold_carries_a_phase_the_payload_names():
    """The ordinary path keeps the phase too, so the card can name `saving`
    -- which now arrives with the loop's real step count (Task 1), not with
    zeros. A payload that names no phase still leaves no key: the anchor test
    above pins that, and it is also the honest render ("this tick said nothing
    about the phase" rather than "the phase is null")."""
    folded = _fold("{step: 49, total_steps: 49, phase: 'saving'}")
    assert folded["step"] == 49
    assert folded["phase"] == "saving"


def test_a_late_phase_tick_does_not_blank_the_card():
    """The second lock on the Task 1 regression, and it is not redundant.

    `saving` arrives AFTER the loop has published, and this fold *replaces*
    the stored entry (app.js:7932 onwards). A bare phase payload carries step 0
    and an empty curve, so accepting it wholesale wipes the sparkline, the
    equity label and the determinate bar at the finish line of every run --
    with nothing red anywhere, and after the run spent its whole length earning
    them. Task 1 stops the engine emitting that payload; this stops the browser
    trusting one if any writer still does. Neither covers the other: the
    Backtest panel's own bar is computed from the raw payload (`stepPct`,
    app.js:8600 and again at :8432) and never passes through here at all.
    """
    previous = (
        "{step: 49, totalSteps: 49, equityCurve: [1000, 1100], openingEquity: 1000,"
        " firstStep: 1, firstStepAt: Date.now() - 60000,"
        " ageSeconds: 1, ageAt: Date.now()}"
    )
    folded = _fold(
        "{step: 0, total_steps: 49, phase: 'saving', progress_age_seconds: 0}",
        previous_js=previous,
    )
    assert folded["step"] == 49
    assert folded["equityCurve"] == [1000, 1100]
    assert folded["openingEquity"] == 1000
    assert folded["firstStep"] == 1
    assert folded["phase"] == "saving"
```

- [ ] **Step 2: Write the failing card tests, and repair every node harness**

Four of the shipped functions this task changes are liftable, and **seven** node-harness builders across three test modules lift at least one of them — `_eval`, `_fold`, `_render_raw`, `_render`, `_patch`, `_run_panel`, `_advance`, which is exactly the set the **Files** block above enumerates. (This sentence said "five" until 2026-09-21, against an enumeration that already listed seven; an implementer who stopped at the headline left two harnesses throwing `ReferenceError`. `test_agent_card_live_chart.py`'s `_render` (`:140`) is *not* an eighth — it delegates to `_render_raw`, which is why `_render_raw` is the only builder there that spreads `_SPARK_HELPERS`.) (`attachToLiveBacktest` and the 1 s poller change as well; neither can be lifted, so the last case in this step pins them by source shape.) A builder that gains `formatBacktestPhase` without `BACKTEST_PHASE_LABELS` — or keeps a helper list that no longer matches the source — throws `ReferenceError`, node exits non-zero, and `assert result.returncode == 0` fires. Repair **six** of the seven in this step — `_fold` is the seventh and Step 1 already did it; the run list in Step 9 names every file.

`dashboard/backend/tests/test_backtest_progress_card.py` (post-Task-0 numbers):

- add **`APP_JS`** to the module's `from dashboard.backend.tests._frontend_source import (…)` block (`:33-39` post-Task-0) — one name, and **do not rewrite the list**. The last source-shape case below reads the raw file, which is the only import this module genuinely gains. An earlier draft of this bullet said to replace the whole statement with a flat `import APP_JS, css_blocks, fn_body, js_const, strip_comments`, which is right only against `origin/main`, where it is one line of three names. Task 0 rewrites it into the parenthesized block above, which already carries `FRONTEND` (used at `:882`) and `strip_comments` (used at `:970`) — so the flat replacement silently deletes `FRONTEND` and NameErrors a case this task never touches, while re-adding a name Task 0 already added. Read the block first, then add the one missing name.
- add `"function formatBacktestPhase(",` and `"function backtestStepPercent(",` as the first two entries of `_PROGRESS_HELPERS` (`:50`). That tuple is shared by three builders on purpose — its own comment says why ("a helper missing from one list is a ReferenceError, but a helper *stubbed* in one list quietly tests the stub") — so these two lines reach `_render`, `_patch` and `_run_panel`. `backtestStepPercent` is unused by two of the three; listing it there anyway is the same trade the tuple already makes, and it is what lets `_run_panel` compute its `stepPct` the way the shipped call sites do instead of hand-writing the number the test is trying to pin.
- add `js_const("BACKTEST_PHASE_LABELS"),` directly after the existing `js_const("BACKTEST_STALE_SECONDS"),` in each of those three builders: `_render` (`:75`, const at `:78`), `_patch` (`:271`, const at `:296`) and `_run_panel` (`:507`, const at `:518`). The constant is not in the helper tuple because the tuple holds function signatures for `fn_body`; consts are listed per builder.
- in `_advance` (`:443`), which lifts `advanceBacktestProgress` alone, add both `js_const("BACKTEST_PHASE_LABELS"),` and `fn_body("function formatBacktestPhase("),` after the existing `js_const("LIVE_SPARK_MAX_POINTS"),`.

`dashboard/backend/tests/test_agent_card_live_chart.py`:

- add `"function formatBacktestPhase(",` to `_SPARK_HELPERS` (`:46`) — it lifts `resolveRunningNotice` and `deriveRunningProgress`, both of which now call it.
- add `js_const("BACKTEST_PHASE_LABELS"),` after `js_const("BACKTEST_STALE_SECONDS"),` in `_render_raw` (`:151`, const at `:154`), the only builder that spreads `_SPARK_HELPERS`. (`_fold` is already done in Step 1.)

`dashboard/backend/tests/test_backtest_progress_format.py`:

- add `"function formatBacktestPhase(",` to `_HELPERS` (`:31`) — it lifts `resolveRunningNotice`.
- add `js_const("BACKTEST_PHASE_LABELS"),` after `js_const("BACKTEST_STALE_SECONDS"),` in `_eval` (`:41`, const at `:44`), its only builder.

Then append to `test_backtest_progress_card.py`:

```python
def test_card_names_the_phase_before_the_first_step():
    html = _render(
        "{step: 0, totalSteps: 49, phase: 'first_decision',"
        " ageSeconds: 2, ageAt: Date.now(), elapsedSeconds: 40}"
    )
    assert "Waiting on first decision" in html
    assert "0/49" in html
    assert "Still starting up" not in html
    # The `0/N` label must not drag the bar along with it. `determinate` is
    # gated on `step > 0` (app.js:8005) and this task does not touch that, so
    # the assertion holds structurally -- but the case that pins indeterminacy
    # today, test_card_falls_back_to_indeterminate_before_the_first_step,
    # predates this work and renders a payload with no phase and no bar count.
    # The shape this task *introduces* -- phase set, total known, step 0 -- is
    # unpinned without this line, and it is the shape the design's user-facing
    # promise leans on. A determinate bar here is an empty track with the
    # indeterminate sweep switched off (app.js:6299-6303): it reads as broken
    # rather than as starting.
    assert "is-determinate" not in html


def test_card_keeps_the_startup_notice_when_nothing_was_published():
    html = _render("{elapsedSeconds: 1000}")
    assert "Still starting up" in html


def test_a_long_phase_reads_as_no_progress_not_as_starting():
    html = _render(
        "{step: 0, totalSteps: 0, phase: 'loading_bars',"
        " ageSeconds: 1000, ageAt: Date.now(), elapsedSeconds: 1010}"
    )
    assert "Loading market data" in html
    assert "No progress for" in html
    assert "Still starting up" not in html
    # ...and does not blame a model that has not been called yet. This is the
    # whole reason the notice takes the phase: routing a pre-model phase into
    # "long model steps can do this" is the same unsupported claim the spec's
    # copy rule bans ("The card never says 'deterministic'"), pointed the other
    # way. A wedged Alpaca/iFinD fetch is the case that produces it.
    assert "long model steps" not in html
    assert "a wide bar window can do this" in html


def test_the_starting_phase_has_no_label_so_the_card_keeps_the_startup_notice():
    """The empty label is what keeps the honest sentence. Were `starting` ever
    published with a label, the fold would accept it and resolveRunningNotice
    would route off formatStartupStaleness ("Still starting up — no steps
    reported after Nm", true during imports) onto formatProgressStaleness ("No
    progress for Nm", false in every clause it could pick before the child has
    even reached load_data)."""
    html = _render("{elapsedSeconds: 1000, phase: 'starting'}")
    assert "Still starting up" in html
    assert "No progress for" not in html


def test_the_card_names_saving_instead_of_a_frozen_percentage():
    """`saving` now arrives carrying the loop's final numbers (Task 1), so the
    card is determinate and `99%` would win the detail line -- a number that
    has stopped moving, beside a run that is demonstrably still working. The
    phase label outranks the percentage for exactly the phases that have one;
    `running`'s label is empty, so a mid-loop tick still reads `35%`."""
    html = _render(
        "{step: 49, totalSteps: 49, phase: 'saving',"
        " ageSeconds: 1, ageAt: Date.now(), elapsedSeconds: 600}"
    )
    assert "49/49" in html
    assert "Saving results" in html


def _step_percent(payload_js: str) -> object:
    return _node(
        fn_body("function backtestStepPercent(")
        + f"console.log(JSON.stringify(backtestStepPercent({payload_js})));"
    )


def test_step_percent_is_null_until_a_real_step():
    """`0/49` is a label, not a percentage.

    Until Task 1 no progress file existed before the first step, so the two
    raw-payload sites saw no `progress` at all, `Number(undefined)` was NaN,
    and the panel bar kept its elapsed-based creep. Now `first_decision`
    publishes `{step: 0, total_steps: 49}` -- and a `total > 0` test alone
    returns a finite 0, which updateBacktestRunProgress takes as authoritative
    over the creep (app.js:8228-8231) and writes as `width: 0%`. An empty
    track with the sweep switched off is the *reads as broken* state the whole
    task exists to avoid.
    """
    assert _step_percent("{step: 0, total_steps: 49}") is None
    assert _step_percent("{step: 0, total_steps: 0}") is None
    assert _step_percent("{}") is None
    assert _step_percent("null") is None
    # ...and a real step is unaffected: this helper replaced two identical
    # inline copies, so the ordinary path must be byte-for-byte what it was.
    assert _step_percent("{step: 84, total_steps: 240}") == 35.0


def test_run_panel_keeps_the_elapsed_bar_at_step_zero():
    """The consumer end of the same rule, driven through the real
    updateBacktestRunProgress rather than asserted on the helper's return
    value -- because the defect was never in the arithmetic, it was in what
    the consumer does with a finite 0."""
    panel = _run_panel(
        "{elapsedSeconds: 60, message: 'Waiting on the first model decision…',"
        " stepPct: backtestStepPercent({step: 0, total_steps: 49})}"
    )
    assert panel["width"] == "2%"  # 60 / 3600, the elapsed fallback
    assert "step 0/49" not in panel["message"]


def test_both_raw_payload_surfaces_share_the_step_percent_rule():
    """A source-shape guard, because neither call site is liftable:
    attachToLiveBacktest is DOM-bound and the poller lives inside an async
    tick. Two byte-identical inline copies are how both acquired the
    `total > 0` defect in the first place, so what is worth pinning is that
    there is one owner left -- not that each copy was fixed."""
    source = strip_comments(APP_JS)
    # Exactly the two inline copies, and deliberately NOT the bare
    # `Number.isFinite(step) && Number.isFinite(total) && total > 0`: that
    # substring also sits inside deriveRunningProgress's `determinate` line
    # (app.js:8005), which is correct code this task does not touch, so a
    # guard written that way fails on the shipped tree for the wrong reason.
    assert "const stepPct = Number.isFinite(" not in source
    # definition + attachToLiveBacktest + the 1s poller
    assert source.count("backtestStepPercent(") == 3
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_agent_card_live_chart.py dashboard/backend/tests/test_backtest_progress_card.py dashboard/backend/tests/test_backtest_progress_format.py -v`
Expected: FAIL — `js_const("BACKTEST_PHASE_LABELS")` raises `AssertionError: BACKTEST_PHASE_LABELS not found in app.js` in every harness, and `fn_body("function formatBacktestPhase(")` and `fn_body("function backtestStepPercent(")` raise for the same reason. `test_both_raw_payload_surfaces_share_the_step_percent_rule` fails on its own terms rather than on a missing lift: `const stepPct = Number.isFinite(` is still there, twice. All three modules must be in this run: two of them are only touched by Step 2's harness repair, and a repair nobody watched fail is a repair nobody has tested.

- [ ] **Step 4: Add the labels and the two helpers, and give the panel's bar one owner**

In `dashboard/frontend/app.js`, directly above the `/**` that opens `formatStartupStaleness`'s doc comment (`:7888`), add:

⚠ Same rule as the **Files** block above, restated here because this is the only anchor in this task that is an *insertion point* rather than an edit site: a wrong edit site fails to match and stops you, a wrong insertion point accepts the text. **Grep for the `function formatStartupStaleness(` line; never jump to the number** — Task 0's rebase merges `app.js` from both sides, so it moves — then walk *up* past its doc comment to the `/**` that opens it, and insert above **that** line.

Three candidates here, one right, and the two wrong ones fail differently. Re-derived 2026-09-21: `:7888-7900` is `formatStartupStaleness`'s doc comment and `:7901` is its `function` line.

- **`:7893` — inside the comment.** The new block's `/**` opens *within* that comment and the **outer** one closes on the new block's `*/`, leaving `:7894-7900` as bare tokens: a syntax error in the shipped `app.js`. An earlier draft of this line said `:7893` while the **Files** block above said `:7901`, and the two halves of one task disagreeing about one insertion point is how the wrong one gets used.
- **`:7901` — between the comment and the function it documents.** Valid JavaScript, and still wrong: `/** Startup-phase counterpart of formatProgressStaleness … */` ends up sitting on top of `const BACKTEST_PHASE_LABELS`, describing a constant it has nothing to do with, while the function it was written for is left bare. Both this step and the **Files** block said `:7901` until 2026-09-21 — which is to say the warning below spent its whole length on the loud failure and then aimed the reader at the quiet one.
- **`:7888` — above the comment's own `/**`.** Correct.

And **nothing in this repo would catch either mistake.** `js_const` is `re.search(r"^const NAME = [^;]+;", APP_JS, re.MULTILINE)` and `fn_body` slices from a signature by brace matching (`tests/_frontend_source.py:134-172`): both read the file as *text*, lift one fragment, and run only that fragment under node. A comment block broken seven lines above — or correctly formed and attached to the wrong declaration — is invisible to all of them, no test parses `app.js` whole, and the first reader is the browser. That is why the anchor is grepped, not counted.

```js
/**
 * Short labels for the pre-loop phases the child publishes (engine.py
 * PROGRESS_PHASES). The Backtest panel prints the server's own sentence; the
 * agent card has one short line, so it gets these. The phase *names* are the
 * contract between the two; each surface owns only its wording. `running` is
 * empty on purpose: a step count owns that line. `starting` is empty for a
 * different reason -- nothing publishes it. The child's first write happens
 * inside load_data, after the imports that dominate the launch, so `starting`
 * exists only as the retroactive gap name in `phases[]`. Listed rather than
 * deleted so this table still enumerates every phase name the engine knows;
 * empty so advanceBacktestProgress keeps refusing such a tick and the card
 * keeps the startup-staleness notice, which is the true sentence there.
 */
const BACKTEST_PHASE_LABELS = {
    starting: '',
    loading_bars: 'Loading market data',
    indicators: 'Calculating indicators',
    first_decision: 'Waiting on first decision',
    running: '',
    saving: 'Saving results',
};

function formatBacktestPhase(phase) {
    // hasOwnProperty, not `LABELS[phase] || ''`. A plain object literal answers
    // for every key on Object.prototype, and `LABELS['constructor']` is a
    // *function* -- truthy, so advanceBacktestProgress's phase guard would
    // accept the tick and deriveRunningProgress would interpolate a function
    // body into the card's detail line. The phase arrives from a JSON file
    // written by a subprocess, so "no caller would pass that" is not an
    // argument available here.
    if (typeof phase !== 'string') return '';
    return Object.prototype.hasOwnProperty.call(BACKTEST_PHASE_LABELS, phase)
        ? BACKTEST_PHASE_LABELS[phase]
        : '';
}

/**
 * The Backtest panel's bar percentage, from the RAW /backtest/status payload.
 *
 * One owner because there are two callers -- attachToLiveBacktest and the 1s
 * poller -- and they were byte-identical inline copies, which is how both came
 * to carry the same defect: they gated on `total > 0` and never on `step > 0`.
 * Since Task 1 the progress file exists before the first step, so
 * `first_decision` publishes `{step: 0, total_steps: N}`; the old test returned
 * a finite 0, updateBacktestRunProgress takes any finite percentage as
 * authoritative over its elapsed-based creep, and the bar snapped to a flat 0%
 * and stopped moving until the first real step. An empty track with the
 * indeterminate sweep switched off reads as broken, which is the same judgement
 * deriveRunningProgress already encodes for the card (`determinate` requires
 * `step > 0`) -- this is that rule, on the surface that had its own copy.
 *
 * Deliberately NOT shared with deriveRunningProgress: that one reads the
 * *folded* entry (`totalSteps`, camelCase) and also owns `pct`, the ETA and
 * `determinate`. Same rule, two payload shapes; merging them would mean one
 * function that has to know which shape it was handed.
 */
function backtestStepPercent(progress) {
    const step = Number(progress?.step);
    const total = Number(progress?.total_steps);
    return Number.isFinite(step) && step > 0 && Number.isFinite(total) && total > 0
        ? (100 * step / total)
        : null;
}
```

Then replace the two inline copies. Find them with `grep -n "const stepPct = Number.isFinite" dashboard/frontend/app.js` — exactly two hits.

In the **1 s poller** (`:8600`), three lines become one:

```js
                    const stepPct = backtestStepPercent(status.progress);
```

In **`attachToLiveBacktest`** (`:8432`), the same replacement, plus the message fallback it needs because this call site ignores `status.message` and writes its own:

```js
        const stepPct = backtestStepPercent(progress);
        updateBacktestRunProgress({
            message: stepPct != null
                ? `Backtest running… step ${progress.step}/${progress.total_steps} (${Math.round(stepPct)}%)`
                // The phase, not the bare generic, and not "step 0/49": this
                // is the dropdown / deep-link / reload-mid-run path, so it is
                // the first thing a returning user sees, and with the poller
                // a second away it must not contradict what that writes next.
                // `formatBacktestPhase` returns '' for `running` and
                // `starting`, so the generic still covers every case that has
                // no phase to name.
                : (formatBacktestPhase(progress.phase) || 'Backtest is running…'),
            stepPct,
        });
```

Note what is **not** passed here: no `elapsedSeconds` and no `progress`. Without the former `updateBacktestRunProgress` leaves the bar's width untouched rather than writing an elapsed guess — correct, since the poller owns the clock and arrives within the second. And its `progress` parameter expects the *folded* camelCase entry, not this raw payload; handing it the raw one would read `totalSteps`/`ageSeconds` off an object that has neither and quietly disable the staleness notice it exists to drive.

The `const step` / `const total` locals above each site go away with the copies; delete them rather than leaving them unused.

- [ ] **Step 5: Make the notice phase-aware, and stop it blaming the model**

Routing a pre-loop phase into the existing staleness notice is only half of it. `formatProgressStaleness` hardcodes *"long model steps can do this"* — true once the model is in the loop and flatly false before it, and the phase this task makes nameable earliest is `loading_bars`, where a wedged Alpaca or iFinD fetch would now print exactly that. The spec's own copy rule ("The card never says 'deterministic'") is the same rule pointed the other way: the card does not assert a cause it cannot support. So the notice takes the phase and picks its clause.

Replace `formatProgressStaleness` (`:7881`) with:

```js
function formatProgressStaleness(secondsSinceUpdate, phase) {
    const gap = Number(secondsSinceUpdate);
    if (!Number.isFinite(gap) || gap < BACKTEST_STALE_SECONDS) return null;
    const minutes = Math.floor(gap / 60);
    // The cause clause names what is actually running. Declared inside the
    // function rather than beside BACKTEST_PHASE_LABELS on purpose: it has
    // exactly one reader, and fn_body carries it into every node harness for
    // free -- a second top-level const would have to be hand-added to five
    // script builders, which is the ReferenceError this task already pays off
    // once. hasOwnProperty for the same reason formatBacktestPhase uses it:
    // `causes['constructor']` is a truthy function on a plain literal, and
    // this string goes straight into the sentence.
    const causes = {
        loading_bars: 'a wide bar window can do this',
        indicators: 'a wide bar window can do this',
        saving: 'writing results can do this',
    };
    const named =
        typeof phase === 'string'
        && Object.prototype.hasOwnProperty.call(causes, phase);
    const cause = named ? causes[phase] : 'long model steps can do this';
    return `No progress for ${minutes}m — ${cause}.`;
}
```

The default is unchanged, so every caller that passes no phase — the whole existing suite, and any payload from a child that predates this plan — gets today's sentence byte-for-byte. `first_decision` and `running` deliberately keep it: those are the phases where a model call really is what the run is waiting on.

Then replace `resolveRunningNotice` (`:7909`) with:

```js
function resolveRunningNotice(running) {
    const step = Number(running.step);
    // A child that publishes phases rewrites the file at every transition, so
    // its mtime ages exactly like a step's: a long phase reads as "No
    // progress for Nm", which is the true statement. Only a child that has
    // published nothing at all falls back to the elapsed-based notice.
    if ((!Number.isFinite(step) || step <= 0) && !formatBacktestPhase(running.phase)) {
        return formatStartupStaleness(running.elapsedSeconds);
    }
    const age = resolveProgressAgeSeconds(running);
    return age === null ? null : formatProgressStaleness(age, running.phase);
}
```

- [ ] **Step 6: Fold a phase tick**

In `advanceBacktestProgress` (`:7932`), replace the opening

```js
    const step = Number(progress?.step);
    const total = Number(progress?.total_steps);
    if (!Number.isFinite(step) || step <= 0) return null;
```

with:

```js
    const step = Number(progress?.step);
    const total = Number(progress?.total_steps);
    if (!Number.isFinite(step) || step <= 0) {
        // A pre-loop phase folds so the card can name it, but without an ETA
        // anchor: firstStep/firstStepAt stay absent so the first real step
        // still sets the rate, and the launch-biased ETA this branch exists
        // to prevent cannot return. A payload with no nameable phase is the
        // pre-confirmation entry this function has always refused.
        if (!formatBacktestPhase(progress?.phase)) return null;
        // A late phase tick -- `saving`, arriving after the loop has
        // published -- must not be folded as a fresh start. This function
        // *replaces* the stored entry, and a bare phase payload carries step 0
        // and an empty curve, so accepting it wholesale blanks the sparkline,
        // the equity label and the determinate bar at the finish line of every
        // run. Carry the phase onto what is already there instead. The age is
        // refreshed because the file really was just rewritten -- keeping the
        // old one would start the staleness notice against a fresh write.
        //
        // Task 1 already stops the engine emitting such a payload (a terminal
        // phase write carries the loop's numbers forward). This is the second
        // lock, for any other writer, and the two are not interchangeable: the
        // Backtest panel's own bar is computed from the raw payload
        // (`stepPct`, :8600 and :8432) and never reaches this function.
        const previousStep = previous ? Number(previous.step) : NaN;
        if (Number.isFinite(previousStep) && previousStep > 0) {
            const lateAge = Number(progress?.progress_age_seconds);
            return {
                ...previous,
                phase: String(progress.phase),
                ageSeconds: Number.isFinite(lateAge) ? lateAge : null,
                ageAt: now,
            };
        }
        const phaseAge = Number(progress?.progress_age_seconds);
        const phaseStartedAt = Number(progress?.phase_started_at);
        return {
            step: 0,
            totalSteps: Number.isFinite(total) ? total : 0,
            phase: String(progress.phase),
            phaseStartedAt: Number.isFinite(phaseStartedAt) ? phaseStartedAt : null,
            equityCurve: [],
            openingEquity: null,
            ageSeconds: Number.isFinite(phaseAge) ? phaseAge : null,
            ageAt: now,
        };
    }
```

Then, in the same function's final `return {` literal (`:7960` onwards), add one entry directly after `totalSteps: total,`:

```js
        // Carried on the ordinary path too, so the card can name `saving` --
        // which since Task 1 arrives with the loop's real step count rather
        // than zeros, and therefore never takes the step-0 branch above.
        // Spread conditionally rather than `phase: … ?? null`: a payload that
        // names no phase must leave no key at all, which is what
        // test_first_real_step_anchors_after_a_phase_tick pins, and which is
        // also the honest render -- "this tick said nothing about the phase"
        // is not "the phase is null".
        ...(typeof progress?.phase === 'string' ? { phase: progress.phase } : null),
```

Leave the rest of the function as it is: with a phase record as `previous`, `Number(previous.firstStep)` is `NaN`, so the first real step anchors itself exactly as before.

- [ ] **Step 7: Derive the label**

In `deriveRunningProgress`, replace

```js
        stepLabel: determinate ? `${step}/${total}` : '',
```

with

```js
        stepLabel: determinate ? `${step}/${total}` : (total > 0 ? `0/${total}` : ''),
```

and replace

```js
        detail: [determinate ? `${pct}%` : null, eta].filter(Boolean).join(' · '),
```

with

```js
        detail: [
            // The label first, the percentage as its fallback -- not the other
            // way round. `running`'s label is empty, so a mid-loop tick still
            // reads `35%` exactly as today, and the pre-loop phases are
            // indeterminate anyway. The case this ordering exists for is
            // `saving`: since Task 1 it arrives carrying the loop's final
            // numbers, so `determinate` is true and a percentage that has
            // stopped moving would win the line for the whole
            // baseline/persistence tail, beside a run that is plainly still
            // working.
            formatBacktestPhase(running.phase) || (determinate ? `${pct}%` : null),
            eta,
        ].filter(Boolean).join(' · '),
```

- [ ] **Step 8: Bump the cache-buster in five files**

```bash
grep -rn "app.js?v=" dashboard/frontend/app.html dashboard/backend/tests/*.py
```

Take the number N shown (identical everywhere) and replace `app.js?v=N` with `app.js?v=N+1` in every file listed. Re-run the grep: exactly one number, five files.

- [ ] **Step 9: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_agent_card_live_chart.py dashboard/backend/tests/test_backtest_progress_card.py dashboard/backend/tests/test_backtest_progress_format.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_backtest_comparison_frontend.py dashboard/backend/tests/test_analytics_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_running_backtest_store.py dashboard/backend/tests/test_backtest_cancel_frontend.py dashboard/backend/tests/test_onboarding_checklist.py -v`
Expected: all pass. The list is every module that lifts or inspects one of those four liftable functions, plus the four cache-buster pins. `test_backtest_progress_format.py` is the one the earlier draft of this plan omitted: its `_HELPERS` lifts `resolveRunningNotice`, and five of its cases pass step-less or step-0 objects, so a missing `formatBacktestPhase` is a non-zero node exit and a hard failure there — not a skip. `test_running_backtest_store.py`, `test_backtest_cancel_frontend.py` and `test_onboarding_checklist.py` need no edit (stub or source-text only) and are here to prove it.

If any of them is skipped rather than passing, `node` is not on `PATH` and this whole layer of coverage is silently absent — install it before reading the run as green.

- [ ] **Step 10: Commit**

```bash
git add dashboard/frontend/app.js dashboard/frontend/app.html dashboard/backend/tests/
git commit -m "feat: name the backtest phase on the agent card before step one

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: The child skips the parent's schema DDL

**Eleven twins, not the six a child imports today.** A child importing `dashboard.scripts.backtest_hourly_agent` constructs six Postgres twins — `PostgresBacktestDatabase` (55 statements), `PostgresModelProviderStore` (18), `PostgresAnalyticsStore` (16), `PostgresAgentStore` (14), `PostgresCreditsStore` (3), `BrokerConnectionStorePostgres` (1): **107 statements over six `_init_schema` calls** before a bar is read. A warm schema does not shrink that — 72 of the 107 are `ALTER TABLE … ADD COLUMN IF NOT EXISTS` and the other five `ALTER`s are constraint drops and re-adds, all of which round-trip whether or not the column or constraint is already there.

That six is a property of today's import graph, not of the design, and the graph reaches it by **two independent routes**. Both were traced by execution rather than by reading, on 2026-09-20; an earlier draft of this paragraph credited the whole set to the analytics chain, which the probe below disproves in one line.

**Route one — the child's own module-level imports, which build five of the six.** `backtest_hourly_agent.py:40` imports `db` outright, the one store the backtest deliberately wants. `:177` imports `domain/credits/service.py`, which reaches `credits.repository` (and so `credits_store`) at its own `:27`; `:178` imports `domain/model_providers/service.py`, which reaches `.repository` (and so `model_provider_store`) at its `:38`, and `model_providers/repository.py` imports `database`, `agents/repository.py` and `brokers/repository.py` at its lines 12, 13 and 14. Importing nothing but those two repository modules — `python3 -c "import dashboard.backend.domain.model_providers.repository, dashboard.backend.domain.credits.repository"`, with `DATABASE_PATH` pointed at a scratch file — already builds `db`, `agent_store`, `broker_store`, `model_provider_store` and `credits_store`, with `domain/analytics/service.py` never imported at all.

**Route two — the analytics chain, which adds the sixth.** Both of those service modules also import `domain/analytics/instrumentation.py` (`credits/service.py:42`, `model_providers/service.py:19`), whose `:15` imports `analytics/service.py`, which imports `analytics_store` at its own `:17` and then runs `analytics_service = _build_analytics_service()` at module level (`:253`). That builds a value store (`value_store=build_value_analytics_store(analytics_store)`, `:243` → `value_repository.py:1067-1068`, which sees a `database_url` on the resolved base and imports `PostgresValueAnalyticsStore`), and **constructing** it resolves `credits_store`, `model_provider_store`, `agent_store` and `run_store` as *constructor-time* defaults inside `__init__` (`value_repository_postgres.py:79-100`, mirrored at `value_repository.py:342-363`). By then route one has already built the first three, so all this route actually owes is `analytics_store` — plus `run_store`, which is a seventh module-level singleton (`domain/runs/repository.py:469`) rather than a seventh *twin*: `domain/runs/` has no `repository_postgres.py`, so protocol runs are SQLite either way and this task never guards them.

Neither route is legible from the child's own import block, which names `db` and two service modules and nothing else; the remaining five twins are built two to four modules further in, and `run_store` — the seventh singleton, which has no twin — only as a constructor's default argument. That is precisely why "which twins a child builds" is not a fact anyone can maintain by reading. One new module-level import in the engine's reach adds a payer back with nothing to catch it, and the CLAUDE.md sentence goes stale again — which is exactly how the "each Postgres twin" claim this task is fixing came to be written. So guard all eleven that own a schema: the parent constructs every one at import, so skipping is safe for each, and the five beyond the child's current graph cost one line apiece. The invariant that is then worth pinning is not "these six" but "no Postgres twin runs DDL outside the guard" — a `rglob` source-shape assertion that cannot rot, in place of a prose warning and a probe somebody has to remember to run.

Re-derive the child's set before trusting the six, rather than taking the two routes above on this document's word — the paragraph they replaced was confidently wrong:

```bash
PYTHONPATH=. python3 - <<'PY'
import importlib, os, sys, tempfile
os.environ["DATABASE_PATH"] = os.path.join(tempfile.mkdtemp(), "probe.db")
importlib.import_module("dashboard.scripts.backtest_hourly_agent")
for mod, attr in [
    ("dashboard.backend.database", "db"),
    ("dashboard.backend.domain.agents.repository", "agent_store"),
    ("dashboard.backend.domain.analytics.repository", "analytics_store"),
    ("dashboard.backend.domain.brokers.repository", "broker_store"),
    ("dashboard.backend.domain.credits.repository", "credits_store"),
    ("dashboard.backend.domain.model_providers.repository", "model_provider_store"),
    ("dashboard.backend.domain.runs.repository", "run_store"),
    ("dashboard.backend.users", "user_store"),
    ("dashboard.backend.domain.portfolios.repository", "portfolio_store"),
    ("dashboard.backend.domain.strategies.repository", "strategy_store"),
    ("dashboard.backend.domain.agents.version_repository", "agent_version_store"),
    ("dashboard.backend.domain.agents.credential_store", "agent_credential_store"),
]:
    print(("BUILT " if mod in sys.modules else "absent"), attr)
PY
```

Measured 2026-09-20: **seven** BUILT — `db`, `agent_store`, `analytics_store`, `broker_store`, `credits_store`, `model_provider_store`, `run_store`; the other five absent. Six of the seven have a Postgres twin. `run_store` does not: `domain/runs/repository.py` has no `*_postgres.py` sibling at all, protocol runs are SQLite by design (see the `DATABASE_PATH` bullet in CLAUDE.md), and its DDL is a local file, not a Neon round trip.

**And the helper times the DDL it does run, which is where the pre-change number comes from.** Prod is the only place this cost exists — locally every store resolves to its SQLite twin, which this task never guards — and the deploy that ships this task is the one that removes it, so no child on that deploy can report the "before". The parent can: it runs the same DDL against the same Neon databases at boot and is never a worker. That is why the helper prints in both directions, and why the ran-it direction prints its cost. It is also the positive signal in the skip direction: without a printed line, "the flag reached the child" and "this build has no guard" are the same silence.

**Files:**
- Modify: `dashboard/backend/db_url.py` (new constant + three functions; add `import os`, `import time` and `from collections.abc import Callable` — verified: the module imports only `annotations` and `urlsplit` today, so none of the three collides)
- Modify, one line each, the eleven Postgres twins that call `self._init_schema()` from `__init__`: `dashboard/backend/database_postgres.py:59`, `dashboard/backend/users_postgres.py:80`, `dashboard/backend/domain/agents/repository_postgres.py:35`, `dashboard/backend/domain/agents/version_repository_postgres.py:30`, `dashboard/backend/domain/agents/credential_store_postgres.py:18`, `dashboard/backend/domain/analytics/repository_postgres.py:195`, `dashboard/backend/domain/brokers/repository_postgres.py:70`, `dashboard/backend/domain/credits/repository_postgres.py:665`, `dashboard/backend/domain/model_providers/repository_postgres.py:154`, `dashboard/backend/domain/portfolios/repository_postgres.py:25`, `dashboard/backend/domain/strategies/repository_postgres.py:33`. The twelfth pair in `tests/test_store_twin_parity.py::_TWINS`, `PostgresValueAnalyticsStore`, is **not** modified: it has no `_init_schema` — it composes the other stores, which that file's `_NO_OWN_DDL_TWINS` already records with a reason. (The credits line is `:665` on the post-Task-0 tree, where that file differs from `origin/main`'s `:662`; every other number here is identical on both.)
- Modify: `dashboard/scripts/backtest_hourly_agent.py` (one import beside `IMPORTS_DONE_AT`, one key in the `startup_clock` Task 2 added)
- Modify: `dashboard/backend/api/routers/backtests.py` — `run_backtest_background` (`:1434`), the `env = os.environ.copy()` line at `:1540`
- Modify: `dashboard/backend/tests/conftest.py:92` (strip list — re-derived 2026-09-20 on both trees; an earlier draft of this plan said `:88`, which is `EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS`)
- Modify: `CLAUDE.md` (Environment & credentials bullets)
- Test: `dashboard/backend/tests/test_backtest_worker_schema_skip.py` (new); `dashboard/backend/tests/test_backtest_launch_phases.py` (append)

**What the wider scope cannot break.** All eleven constructors are structurally identical — `self.database_url = require_postgres_url(database_url)` then `self._init_schema()`, with only `PostgresBacktestDatabase` carrying its `self._sqlite = BacktestDatabase()` line between — and all eleven already import from `dashboard.backend.db_url`, so each diff is one replaced line plus one extended import. `test_store_twin_parity`'s fourth axis **does** read `__init__`: `_method_code` collects every `FunctionDef` in the class body with no `_`-prefix filter, so do *not* justify this by claiming the axis skips it. The reason it is safe is narrower and was checked: no pair's `__init__` is byte-identical today (`_DUPLICATED_BODIES` declares none), and this change edits only the Postgres side, making the two copies more different — so neither the `diverged` nor the `undeclared` assertion can fire. `_DIALECT_BRANCH_PATTERN` matches `is_postgres` and `hasattr(…, "database_url")`; `init_schema_unless_worker` matches neither, so no `_DIALECT_BRANCH_ALLOWLIST` entry is needed.

**Interfaces:**
- Produces: `db_url.BACKTEST_WORKER_ENV = "ATL_BACKTEST_WORKER"`; `db_url.schema_init_skipped() -> bool`; `db_url.schema_init_seconds() -> float`; `db_url.init_schema_unless_worker(label: str, init_schema: Callable[[], None]) -> None`. Child env carries `ATL_BACKTEST_WORKER=1`, and the child's `startup_clock` gains `schema_init_seconds`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_backtest_worker_schema_skip.py`:

```python
"""A dashboard backtest child must not repeat the parent's schema DDL.

The parent runs every store's `_init_schema` at import, so the schema exists
before it spawns anything. A child that ran it again paid a pooled Neon
checkout plus a batch of DDL round trips per store before reading a single
bar -- pure start-up waste, and invisible because nothing before the bar loop
was measured. The flag names the process role, not the DDL, so no operator
sets it globally and later child-only behaviour has a home.

Eleven twins, not the six a child imports today: which stores a child
constructs is an accident of the import graph, so the invariant worth pinning
is "no Postgres twin runs DDL outside the guard", not "these six".
"""
import importlib
from pathlib import Path

import pytest

from dashboard.backend import db_url

_URL = "postgresql://atl:not-a-secret@db.example.invalid:5432/atl_test"
_BACKEND = Path(__file__).resolve().parents[1]

# (module, class, the label that store's factory already prints in its
# "<label> backend: postgres (...)" boot line). Imported inside the test
# bodies, not here: the registry is plain strings, so an import error fails
# one case instead of erroring collection and aborting the session -- the
# convention test_store_twin_parity.py documents for the same reason.
_TWINS = [
    ("dashboard.backend.database_postgres", "PostgresBacktestDatabase", "run history"),
    ("dashboard.backend.users_postgres", "PostgresUserStore", "user_store"),
    ("dashboard.backend.domain.agents.repository_postgres", "PostgresAgentStore", "agent_store"),
    ("dashboard.backend.domain.agents.version_repository_postgres", "PostgresAgentVersionStore", "agent_version_store"),
    ("dashboard.backend.domain.agents.credential_store_postgres", "PostgresAgentCredentialStore", "agent_credential_store"),
    ("dashboard.backend.domain.analytics.repository_postgres", "PostgresAnalyticsStore", "analytics_store"),
    ("dashboard.backend.domain.brokers.repository_postgres", "BrokerConnectionStorePostgres", "broker_connections"),
    ("dashboard.backend.domain.credits.repository_postgres", "PostgresCreditsStore", "credits_store"),
    ("dashboard.backend.domain.model_providers.repository_postgres", "PostgresModelProviderStore", "model_provider_store"),
    ("dashboard.backend.domain.portfolios.repository_postgres", "PostgresPortfolioStore", "portfolio_store"),
    ("dashboard.backend.domain.strategies.repository_postgres", "PostgresStrategyStore", "strategy_store"),
]
_IDS = [name for _m, name, _l in _TWINS]


def test_only_the_literal_one_arms_the_flag(monkeypatch):
    monkeypatch.delenv(db_url.BACKTEST_WORKER_ENV, raising=False)
    assert db_url.schema_init_skipped() is False
    monkeypatch.setenv(db_url.BACKTEST_WORKER_ENV, "1")
    assert db_url.schema_init_skipped() is True
    monkeypatch.setenv(db_url.BACKTEST_WORKER_ENV, "true")
    assert db_url.schema_init_skipped() is False


@pytest.mark.parametrize(("module", "name", "label"), _TWINS, ids=_IDS)
def test_a_worker_skips_schema_init(monkeypatch, capsys, module, name, label):
    calls = []
    cls = getattr(importlib.import_module(module), name)
    monkeypatch.setattr(cls, "_init_schema", lambda self: calls.append("ddl"))
    monkeypatch.setenv(db_url.BACKTEST_WORKER_ENV, "1")

    cls(_URL)

    assert calls == []
    # The label is the token that store's factory already prints in its
    # "<label> backend: postgres (...)" line, so one grep finds both halves
    # of a store's start-up story.
    assert f"{label} backend: schema init skipped (backtest worker)" in capsys.readouterr().out


@pytest.mark.parametrize(("module", "name", "label"), _TWINS, ids=_IDS)
def test_the_parent_still_runs_schema_init(monkeypatch, capsys, module, name, label):
    calls = []
    cls = getattr(importlib.import_module(module), name)
    monkeypatch.setattr(cls, "_init_schema", lambda self: calls.append("ddl"))
    monkeypatch.delenv(db_url.BACKTEST_WORKER_ENV, raising=False)

    cls(_URL)

    assert calls == ["ddl"]
    # Timed, not merely run. This line in the parent's Render boot log is the
    # only pre-change number the deploy that removes the child's copy can still
    # produce -- the parent runs the same DDL against the same databases and is
    # never a worker.
    out = capsys.readouterr().out
    assert f"{label} backend: schema init " in out and "skipped" not in out


def test_a_worker_accumulates_no_schema_time(monkeypatch):
    """The accumulator is the measurement, so its zero has to be a real zero.

    `starting.schema_init_seconds` is what tells the Final verification table
    how much of the start was DDL. If a skipped init still charged time, the
    prod run's 0.0 would stop being evidence that the flag fired. Read as a
    delta, not an absolute: the counter is process-global and never resets.
    """
    monkeypatch.setenv(db_url.BACKTEST_WORKER_ENV, "1")
    before = db_url.schema_init_seconds()
    for module, name, _label in _TWINS:
        cls = getattr(importlib.import_module(module), name)
        monkeypatch.setattr(cls, "_init_schema", lambda self: None)
        cls(_URL)
    assert db_url.schema_init_seconds() == before


def test_no_postgres_twin_runs_ddl_outside_the_guard():
    """A twelfth twin must not be able to arrive unguarded.

    Which twins a backtest child constructs is an accident of the import
    graph -- six of the eleven on the graph as it stands -- so the invariant
    worth pinning is not "the six" but "no Postgres twin calls _init_schema
    from __init__ directly". Eleven non-test modules matched before this
    change and none may after. The twelfth file under this glob,
    value_repository_postgres.py, has no _init_schema at all.
    """
    offenders = sorted(
        str(path.relative_to(_BACKEND))
        for path in _BACKEND.rglob("*_postgres.py")
        if "tests" not in path.parts
        and "self._init_schema()" in path.read_text(encoding="utf-8")
    )
    assert offenders == [], (
        "these Postgres twins run schema DDL outside "
        "db_url.init_schema_unless_worker, so a backtest child repeats it: "
        f"{offenders}"
    )


def test_the_guarded_list_accounts_for_every_twin():
    """Guarded + deliberately-exempt must equal the parity registry.

    The sentence this plan writes into CLAUDE.md claims a scope. A claim
    about "every twin" that is not checked against the list of twins is how
    the four-versus-twelve error got written down in the first place.

    The exempt half is read from test_store_twin_parity's own
    _NO_OWN_DDL_TWINS rather than restated here: that registry already names
    PostgresValueAnalyticsStore with the reason (it composes the other
    stores), and a second copy of the exemption is a second owner that can
    disagree with the first. Imported in the body, not at module scope, so a
    rename there fails this one case instead of aborting collection.
    """
    from dashboard.backend.tests.test_store_twin_parity import (
        _NO_OWN_DDL_TWINS,
        _TWIN_IDS,
    )

    assert set(_IDS) | set(_NO_OWN_DDL_TWINS) == set(_TWIN_IDS)
    assert not (set(_IDS) & set(_NO_OWN_DDL_TWINS)), (
        "a twin cannot be both guarded and exempt from owning DDL"
    )
```

Append to `dashboard/backend/tests/test_backtest_launch_phases.py`:

```python
def test_child_env_is_marked_as_a_backtest_worker(monkeypatch):
    monkeypatch.delenv("ATL_BACKTEST_WORKER", raising=False)
    captured = _launch(monkeypatch)
    assert captured["env"]["ATL_BACKTEST_WORKER"] == "1"
    # The parent's own environment is untouched: the flag is for the child.
    assert "ATL_BACKTEST_WORKER" not in os.environ
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_backtest_worker_schema_skip.py dashboard/backend/tests/test_backtest_launch_phases.py -v`
Expected: FAIL with `AttributeError: module 'dashboard.backend.db_url' has no attribute 'BACKTEST_WORKER_ENV'`, a `KeyError: 'ATL_BACKTEST_WORKER'`, and `test_no_postgres_twin_runs_ddl_outside_the_guard` listing all **eleven** twin modules as offenders.

- [ ] **Step 3: Add the flag reader, the accumulator and the one guard**

In `dashboard/backend/db_url.py`, add `import os`, `import time` and `from collections.abc import Callable` above the existing `from urllib.parse import urlsplit`, and append at the end of the module:

```python
#: Set by `api/routers/backtests.py` in a dashboard backtest child's
#: environment and nowhere else.
BACKTEST_WORKER_ENV = "ATL_BACKTEST_WORKER"


def schema_init_skipped() -> bool:
    """True inside a dashboard backtest child, which must not repeat DDL.

    The parent ran every store's ``_init_schema`` at import, so the schema
    exists by the time it spawns anything; a child that ran it again paid a
    fresh pooled connection and a batch of DDL round trips per store to Neon
    before reading a single bar. Named for the process role rather than the
    DDL so no operator sets it globally, and so later child-only behaviour
    (turning the analytics snapshot projection off, for one) has a home.
    Only the literal ``1`` arms it.
    """
    return os.environ.get(BACKTEST_WORKER_ENV, "").strip() == "1"


#: Seconds this process has spent inside store ``_init_schema()`` calls.
#: Process-global and never reset: it is read once, by the backtest child's
#: script, just before it builds the engine. Do not read it from a request
#: path -- in a long-lived parent it is the whole boot's DDL, not this call's.
_schema_init_seconds = 0.0


def schema_init_seconds() -> float:
    """Total DDL time this process has paid, for the ``starting`` record.

    It is what makes that phase attributable: ``starting`` is one interval
    covering process spawn, pandas and three SDK imports, and the seven store
    singletons those imports construct as a side effect (six of them Postgres
    twins, which are the six that run DDL) -- and a single number that moves
    for any of those reasons cannot say whether removing the DDL helped. In a
    worker it is
    ``0.0`` -- present and zero, which is the evidence the flag fired, not a
    missing field.
    """
    return _schema_init_seconds


def init_schema_unless_worker(label: str, init_schema: Callable[[], None]) -> None:
    """Run ``init_schema`` unless this process is a dashboard backtest child.

    Defined once rather than cloned into each twin's ``__init__`` -- the same
    call this module's header makes for describe_database_url, for a sharper
    version of the same reason. Eleven hand-copied
    ``if schema_init_skipped(): print(...) else: self._init_schema()`` blocks
    are eleven chances for one to invert its condition or lose its log line,
    and the two failures are asymmetric. An inverted guard fails in the
    *parent*: it stops running a ``CREATE TABLE IF NOT EXISTS`` that would
    have been free, and the first symptom is an UndefinedTable or
    UndefinedColumn on the deployed database, from a store nobody edited. A
    lost log line makes "the child skipped DDL" and "this twin was never
    constructed" the same empty stdout. One condition, one line, one place to
    read them.

    **Both directions print, and the ran-it direction prints its cost.** A skip
    line alone would leave "the flag reached the child" and "this build has no
    guard at all" producing the same silence in a log. And the timing is the
    only pre-change number the deploy that removes the cost can still produce:
    the parent runs this same DDL against the same databases at boot and is
    never a worker, so the parent's boot log *is* the baseline the child's skip
    is measured against. Locally every store resolves to its SQLite twin and
    never reaches here at all, which is why no local before/after of the flag
    means anything.

    ``label`` is the token that store's factory already prints in its
    ``<label> backend: postgres (...)`` boot line, so ``grep 'backend:'`` reads
    as one story.
    """
    global _schema_init_seconds
    if schema_init_skipped():
        print(f"{label} backend: schema init skipped (backtest worker)", flush=True)
        return
    started = time.perf_counter()
    init_schema()
    elapsed = time.perf_counter() - started
    _schema_init_seconds += elapsed
    print(f"{label} backend: schema init {elapsed:.2f}s", flush=True)
```

Then close the loop Task 2 left open. In `dashboard/scripts/backtest_hourly_agent.py`, add `from dashboard.backend import db_url` directly above the `IMPORTS_DONE_AT` stamp (`:179`-ish, after the last existing import), and add the third key to the `startup_clock` dict in the `HourlyBacktester(` call:

```python
            "schema_init_seconds": db_url.schema_init_seconds(),
```

It lands here rather than in Task 2 because the function is born here, and a plan whose intermediate commit crashes the CLI on its first real run is a plan nobody can bisect.

- [ ] **Step 4: Guard the eleven twins that own a schema**

Each of the eleven runs `self._init_schema()` as the last statement of its
`__init__`. Replace that one line with a call to the shared guard, and extend
the `from dashboard.backend.db_url import require_postgres_url` line each of
them already has to
`from dashboard.backend.db_url import init_schema_unless_worker, require_postgres_url`.
Nothing else in any constructor moves. The bound method is resolved at call
time, so a test that monkeypatches `_init_schema` on the class still sees the
substitute.

The worked example, `dashboard/backend/database_postgres.py` — the only one of
the eleven with a line between the URL check and the DDL call, and it stays
where it is:

```python
    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        self._sqlite = BacktestDatabase()   # hot half: idempotency_keys stays local
        init_schema_unless_worker("run history", self._init_schema)
```

and the same single line, with its own label, in the other ten:

| file | `self._init_schema()` at | label |
|---|---|---|
| `users_postgres.py` | 80 | `user_store` |
| `domain/agents/repository_postgres.py` | 35 | `agent_store` |
| `domain/agents/version_repository_postgres.py` | 30 | `agent_version_store` |
| `domain/agents/credential_store_postgres.py` | 18 | `agent_credential_store` |
| `domain/analytics/repository_postgres.py` | 195 | `analytics_store` |
| `domain/brokers/repository_postgres.py` | 70 | `broker_connections` |
| `domain/credits/repository_postgres.py` | 665 | `credits_store` |
| `domain/model_providers/repository_postgres.py` | 154 | `model_provider_store` |
| `domain/portfolios/repository_postgres.py` | 25 | `portfolio_store` |
| `domain/strategies/repository_postgres.py` | 33 | `strategy_store` |

Two names in that table are worth reading twice. The brokers class is
`BrokerConnectionStorePostgres`, **not** `PostgresBrokerConnectionStore` — it is
the one twin named the other way round, and the test registry in Step 1 has to
match it or the case errors at import. And every label but one is copied
verbatim from that store's existing factory boot line, so the skip line and the
backend line sort together under one `grep 'backend:'`; the exception is
`model_provider_store`, whose `_build_model_provider_store()` prints **no** boot
line at all — it is the one store of the eleven with none, so this is the first
use of that token. Do not invent a different one to make it look symmetrical; if
that factory ever gains a boot line it must use this same token.

The SQLite twins are untouched: their DDL is local and fast, and the suite
constructs them against fresh temp files that genuinely need it.

Two residuals this does **not** remove, both of which stay inside the
`starting` interval and are why `imports+stores` will not fall by the DDL
saving alone: `PostgresBacktestDatabase.__init__` builds an embedded
`BacktestDatabase()` *before* the guard, so the child still runs the local
SQLite schema check and the `trades` migration; and the pooled connection
itself is opened lazily (`db_pool.get_pool`, one cached pool per URL,
`min_size=0`), so a guarded child that never queries during import opens no
pool at all — which is the point, but it means the handshake the parent pays is
in neither figure.

- [ ] **Step 5: Mark the child in the parent**

In `run_backtest_background` (`dashboard/backend/api/routers/backtests.py:1434`), directly after `env = os.environ.copy()` (`:1540`), add:

```python
        # The child must not repeat this process's schema DDL (db_url.
        # schema_init_skipped). Set on the copy only: the parent is not a
        # worker, and a value that leaked into os.environ would make the next
        # uvicorn reload skip DDL it actually needs.
        env[BACKTEST_WORKER_ENV] = "1"
```

and add `BACKTEST_WORKER_ENV` to the module's existing `from dashboard.backend.db_url import ...` line (create the import beside the other `dashboard.backend` imports if the module has none: `from dashboard.backend.db_url import BACKTEST_WORKER_ENV`).

- [ ] **Step 6: Strip it in conftest and document it**

In `dashboard/backend/tests/conftest.py`, after `os.environ.pop("MAX_LEGACY_ACTIVE_PER_SESSION", None)` (`:92` — re-derived 2026-09-20 on both trees; an earlier draft of this plan said `:88`, which is the `EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS` line), add:

```python
# A shell that exported the backtest-worker flag would make every Postgres
# twin the suite constructs skip DDL and fail on its first query.
os.environ.pop("ATL_BACKTEST_WORKER", None)
```

In `CLAUDE.md`, under **Environment & credentials**, add this bullet directly after the `MAX_ACTIVE_DASHBOARD_BACKTESTS` bullet:

```markdown
- `ATL_BACKTEST_WORKER` (**never set by an operator**): `run_backtest_background` sets it to `1` in a dashboard backtest child's environment and nowhere else. Every Postgres store twin that owns a schema — **eleven of the twelve pairs in `tests/test_store_twin_parity.py::_TWINS`**; the twelfth, `PostgresValueAnalyticsStore`, composes the other stores and has no `_init_schema`, which that file's `_NO_OWN_DDL_TWINS` already records — routes its constructor's DDL through `db_url.init_schema_unless_worker`, which skips it on the literal `1` and prints `<store> backend: schema init skipped (backtest worker)`, and otherwise runs it, times it into `db_url.schema_init_seconds()` and prints `<store> backend: schema init <n>s`. The parent ran that DDL at import, and the child was paying a pooled Neon checkout plus a batch of round trips per store before reading a bar: on today's import graph six twins construct in a child — `PostgresBacktestDatabase` (55 statements), `PostgresModelProviderStore` (18), `PostgresAnalyticsStore` (16), `PostgresAgentStore` (14), `PostgresCreditsStore` (3), `BrokerConnectionStorePostgres` (1), 107 statements over six checkouts. A warm schema does not shrink that: 72 of the 107 are `ADD COLUMN IF NOT EXISTS` and the other five `ALTER`s are constraint drops and re-adds, so they round-trip whether or not the column is there. **Both log directions are deliberate:** the timed line is the parent's own boot cost, which is the only pre-change baseline the deploy that removes the child's copy can still produce, and a silent skip would look exactly like a build with no guard. ⚠ **All eleven are guarded, not just those six, on purpose.** Which stores a child imports is an accident of the import graph, and the accident arrives by **two independent routes**. The script imports `db` deliberately (`backtest_hourly_agent.py:40`) and then reaches `credits_store` and `model_provider_store` through the two service imports at `:177-178` (`credits/service.py:27`, `model_providers/service.py:38`), with `model_providers/repository.py:13-14` pulling in `agents/repository.py` and `brokers/repository.py` — five of the six twins, with `domain/analytics/service.py` never imported. Only the sixth, `analytics_store`, needs the analytics chain those same service modules also pull in (`domain/analytics/instrumentation.py:15` → `analytics/service.py:17`, whose module-level `analytics_service = _build_analytics_service()` at `:253` then constructs a value store that resolves `credits_store`, `model_provider_store`, `agent_store` and `run_store` as constructor-time defaults — the first three already built by route one, `run_store` (like `analytics_store`) built here for the first time). `run_store` does not move the count: it is a seventh module-level singleton (`domain/runs/repository.py:469`), not a seventh twin — `domain/runs/` has no `repository_postgres.py`, so protocol runs are SQLite either way and nothing here guards them. Neither route is legible from the child's own import block, which names `db` and two service modules and nothing else, so one new module-level import in the engine's reach would otherwise re-add an unguarded payer with nothing going red. `test_backtest_worker_schema_skip.py` pins the invariant that survives that: no non-test `*_postgres.py` calls `self._init_schema()` outside the guard, and guarded ∪ exempt equals `_TWINS`. One shared helper rather than eleven copies, for the reason `db_url.py`'s header gives about its scrubber — an inverted copy fails in the *parent*, as an `UndefinedTable` on prod from a store nobody edited, and a dropped log line makes "skipped" and "never constructed" the same empty stdout. Named for the process role, not the DDL, so nobody exports it globally and later child-only behaviour has a home. The child also publishes pre-loop **phases** to its progress file (`engine.py:PROGRESS_PHASES`: `starting`, `loading_bars`, `indicators`, `first_decision`, `running`, `saving`), which `/backtest/status` turns into the card's sentence and which double as the start-up measurement — read `phases[]` off a run's progress file for the per-phase timings, or the child's `⏱  phase …` stdout lines out of the service log, which is the copy that survives (the parent unlinks the progress file when the run ends). The `starting` entry additionally carries `child_entered_at`, `imports_done_at` and `schema_init_seconds`, splitting that one interval into spawn, imports-including-stores, of-which-DDL and preflight: one phase name, four numbers, because a single figure covering all four cannot say which of them an optimisation moved. `tests/conftest.py` strips the flag.
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtest_worker_schema_skip.py dashboard/backend/tests/test_backtest_launch_phases.py dashboard/backend/tests/test_store_twin_parity.py dashboard/backend/tests/test_postgres_url_guard.py dashboard/backend/tests/test_backtest_db_postgres.py dashboard/backend/tests/test_users_postgres.py dashboard/backend/tests/test_agent_store_postgres.py dashboard/backend/tests/test_model_provider_store_postgres.py dashboard/backend/tests/test_portfolio_store_postgres.py dashboard/backend/tests/test_strategy_store_postgres.py dashboard/backend/tests/domain/analytics/test_repository_postgres.py dashboard/backend/tests/domain/credits/test_repository_postgres.py dashboard/backend/tests/test_market_data_features.py -v`
Expected: all pass (`@pg_only` cases skip without `TEST_POSTGRES_URL`).

`test_backtest_worker_schema_skip.py` reports **26** cases: eleven × two parametrized, plus `test_only_the_literal_one_arms_the_flag`, `test_a_worker_accumulates_no_schema_time`, `test_no_postgres_twin_runs_ddl_outside_the_guard` and `test_the_guarded_list_accounts_for_every_twin`. That 26 is `2 × len(_TWINS) + 4` against **this file's own** eleven-entry `_TWINS` (Step 1), not the twelve-pair registry in `test_store_twin_parity.py`, and it moves the moment a twelfth schema-owning twin is added — recount from the file rather than asserting it back.

`test_store_twin_parity.py` is in the list because eleven twins' `__init__` bodies just changed. Its fourth axis, `_DUPLICATED_BODIES`, compares each Postgres twin against its **SQLite** counterpart — whose `__init__` takes a path and is untouched — and no twin currently declares `__init__` as duplicated, so nothing should move. Confirm that rather than reason about it twice.

`test_postgres_url_guard.py` is there because `db_url.py` itself grows three members in this task. And there are no `test_*_postgres.py` modules for the broker, agent-version or agent-credential twins — verified by `ls dashboard/backend/tests/**/*postgres*` — so the parity file plus this task's own parametrized cases are what cover those three.

- [ ] **Step 8: Commit**

```bash
git add dashboard/backend/db_url.py dashboard/backend/database_postgres.py dashboard/backend/users_postgres.py dashboard/backend/domain/agents/repository_postgres.py dashboard/backend/domain/agents/version_repository_postgres.py dashboard/backend/domain/agents/credential_store_postgres.py dashboard/backend/domain/analytics/repository_postgres.py dashboard/backend/domain/brokers/repository_postgres.py dashboard/backend/domain/credits/repository_postgres.py dashboard/backend/domain/model_providers/repository_postgres.py dashboard/backend/domain/portfolios/repository_postgres.py dashboard/backend/domain/strategies/repository_postgres.py dashboard/scripts/backtest_hourly_agent.py dashboard/backend/api/routers/backtests.py dashboard/backend/tests/conftest.py dashboard/backend/tests/test_backtest_worker_schema_skip.py dashboard/backend/tests/test_backtest_launch_phases.py CLAUDE.md
git commit -m "feat: skip schema DDL inside the backtest worker

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Measure the start, then decide on aggregation

The phase payload is the instrument. This task records the numbers the spec asks for and applies the one gated optimisation.

**Files:**
- Create (scratch, not committed): `<scratchpad>/measure_start.sh`, `<scratchpad>/measure_llm_start.py` (the shell script's LLM arm — the launcher refuses an LLM run, Step 1), `<scratchpad>/bench_aggregation.py`, `<scratchpad>/count_child_ddl.py` (Step 8). `measure_llm_start.py` must sit **beside** `measure_start.sh`, which resolves it as `$SCRIPT_DIR/measure_llm_start.py`; the other two are invoked directly, from the repo root, with `PYTHONPATH=.`.
- Modify (only if the gate fires): `dashboard/backend/domain/backtesting/bar_aggregation.py:120-143` (re-derived 2026-09-20; an earlier draft said `:112-136`)
- Test (only if the gate fires): `dashboard/backend/tests/backtesting/test_bar_aggregation.py` (existing; must stay green)
- Modify: this plan's **Final verification** tables (three of them now: the phase split, the per-store DDL, and the import-time statement count from Step 8)

- [ ] **Step 1: Run one rule-based and one LLM backtest locally with a progress file**

Write `<scratchpad>/measure_start.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git rev-parse --show-toplevel)"
set -a; source dashboard/.env; set +a
OUT="${1:?scratch dir}"; MODE="${2:?rule|llm}"
mkdir -p "$OUT"
# Not optional. Unset, `db` resolves to the committed seed backtest.db and both
# arms write their runs into a tracked file -- the same trap Track B's rerun.py
# documents, and the reason Task 7 Step 2 checks the seed DB at all.
export DATABASE_PATH="$OUT/measure.db"
LAUNCHED="$(date +%s.%N)"
if [ "$MODE" = "llm" ]; then
  # The engine in-process, deliberately NOT the launcher this same script uses
  # for the rule-based arm. Why, below.
  PYTHONPATH=. python3 "$SCRIPT_DIR/measure_llm_start.py" \
    "$LAUNCHED" "$OUT/progress_llm.json" \
    | tee "$OUT/stdout_llm.log"
else
  python3 dashboard/scripts/backtest_hourly_agent.py \
    --start 2026-09-08 --end 2026-09-16 --session-id measure-rule \
    --no-llm --run-id measure_rule \
    --progress-file "$OUT/progress_rule.json" \
    --launched-at "$LAUNCHED" \
    | tee "$OUT/stdout_rule.log"
fi
python3 - "$OUT/progress_$MODE.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
seen = {ph["name"]: ph for ph in p["phases"]}


def line(label, value):
    """MISSING, never blank and never 0.00s.

    An unmeasured mark and a free one are different facts (spec, *Failure
    behaviour*), and every way of rendering them the same is some kind of
    default: a blank cell in the table below, or `.get(key, 0.0)` here.
    """
    print(f"{label:16s}" + (f"{value:8.2f}s" if value is not None else "    MISSING"))


def gap(record, later, earlier):
    end, begin = record.get(later), record.get(earlier)
    return None if end is None or begin is None else end - begin


for name in ("starting", "loading_bars", "indicators", "first_decision", "running"):
    ph = seen.get(name)
    if ph is None:
        line(name, None)
        continue
    line(name, ph["ended_at"] - ph["started_at"])
    if name != "starting":
        continue
    # The same rule one level down, and this is where the first draft of this
    # reader broke it twice. `_init_progress_phases` writes each mark only
    # when the caller passed it, so any of the three can be absent on its own
    # -- a child built before Task 5 Step 3, for one, has no
    # `schema_init_seconds` at all. Gating the whole split on
    # `"imports_done_at" in ph` printed *no line*, so an unsplit `starting`
    # and one whose split was simply not read came out identical; and
    # `.get("schema_init_seconds", 0.0)` printed `0.00`, which is the exact
    # value this plan defines elsewhere as the evidence the worker flag fired.
    # A MISSING costs one grep. A fabricated 0.00 gets copied into a table and
    # believed.
    line("  spawn+interp", gap(ph, "child_entered_at", "started_at"))
    line("  imports+stores", gap(ph, "imports_done_at", "child_entered_at"))
    line("    schema DDL", ph.get("schema_init_seconds"))
    line("  preflight", gap(ph, "ended_at", "imports_done_at"))
print(f'{p["phase"]:15s} (in progress)')
PY
```

**The LLM arm cannot go through the launcher, and an earlier draft of this step said it could.** `backtest_hourly_agent.py --use-llm` exits **2** with `error: explicit LLM execution requires a signed execution handoff` — Track B's Task 8 Step 2 records this, verified by running it, and it reproduces from source here: `--use-llm` leaves `requested_decision_source` `None` (`:300-305`), so `resolve_decision_source` returns the default Alpaca profile's `default_decision_source`, which is `llm` (`infrastructure/market_data/profiles.py:206`); `--runtime-type` defaults to `pipeline` (`:207-208`); and with no `--execution-handoff-stdin` the `elif` at `:324-328` refuses exactly that pair. Minting a real handoff locally needs an account id, a registered provider row and a stored credential. So the whole LLM column — the `starting` split *and* the `first_decision` number this track exists to produce — would have been unobtainable as written.

That refusal is a rule about **who pays for the call**, not about whether the engine can make one. The engine has no equivalent: with `execution_client=None` it builds `make_llm_client()` and takes the single-prompt branch (`engine.py:374-375`), the same path the leaderboard's `llm_agent` runs on. So drive the engine directly, the way Track B's `rerun.py` does — plus the three things `rerun.py` has no reason to pass: `progress_file`, `launched_at` and `startup_clock`.

Write `<scratchpad>/measure_llm_start.py` beside the shell script:

```python
#!/usr/bin/env python3
"""One LLM backtest driven in-process, instrumented like the real child.

    PYTHONPATH=. python3 measure_llm_start.py <launched_at> <progress_file>

Called only by measure_start.sh, which supplies both and exports
DATABASE_PATH. Not `backtest_hourly_agent.py --use-llm`: see the step above.
"""
import time

# First executable statement, mirroring the CHILD_ENTERED_AT stamp Task 2
# Step 3 puts above `import sys` in backtest_hourly_agent.py (`:19`).
# Everything the real child pays at import has to sit BELOW this line, or the
# two columns of the table stop meaning the same thing.
CHILD_ENTERED_AT = time.time()

import sys

# Imported for its cost, not for a symbol. This is the module a real child
# loads, and it is what constructs the seven store singletons; importing the
# engine alone would measure a smaller graph and quietly under-report the LLM
# column's `imports+stores` against the rule-based one. Safe to import: `main()`
# is behind `if __name__ == "__main__":` (:573), so nothing runs and no SIGTERM
# handler is installed (:574).
import dashboard.scripts.backtest_hourly_agent  # noqa: F401
from dashboard.backend import db_url
from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.infrastructure.market_data.profiles import (
    LLM_DECISION_SOURCE,
)

IMPORTS_DONE_AT = time.time()


def main(argv):
    launched_at, progress_file = float(argv[1]), argv[2]
    backtester = HourlyBacktester(
        "2026-09-08",
        "2026-09-16",
        "measure-llm",
        use_llm=True,
        decision_source=LLM_DECISION_SOURCE,
        # None on purpose: engine.py:374-375 then builds make_llm_client()
        # itself and takes the single-prompt branch. A handoff client is the
        # one thing this script cannot mint.
        execution_client=None,
        live_run_id="measure_llm",
        # The three the shell arm gets from argv. Without progress_file the
        # engine still advances the phase clock but writes nothing, and this
        # step has nothing to read.
        progress_file=progress_file,
        launched_at=launched_at,
        startup_clock={
            "child_entered_at": CHILD_ENTERED_AT,
            "imports_done_at": IMPORTS_DONE_AT,
            "schema_init_seconds": db_url.schema_init_seconds(),
        },
    )
    # Fail loudly here, exactly as Track B's rerun.py does, and for a sharper
    # reason: this script's whole output is the `first_decision` number, "how
    # long until the first model decision?". With no execution_client and the
    # default ALPACA source `self.strict_llm` is False (engine.py:311-317), so
    # BOTH LLM-unavailable paths degrade instead of raising -- no SDK rewrites
    # decision_source to rule_based (:363), and a make_llm_client() that
    # returns None prints a warning, sets use_llm = False and rewrites it
    # again (:375-386). The degraded run then completes, writes a full
    # phases[] and exits 0. A rule-based run DOES produce a first_decision row
    # (see below), so the two are byte-indistinguishable in the artifact --
    # and the number recorded in the `local, LLM` column would be the cost of
    # a decision no model made.
    if not backtester.use_llm or backtester.decision_source != LLM_DECISION_SOURCE:
        raise SystemExit(
            "measure_llm_start.py: no LLM client resolved -- this run would be "
            "rule-based, and its first_decision number would be recorded as a "
            "model's. Set COMMONSTACK_API_KEY / OPENROUTER_API_KEY / "
            "ANTHROPIC_API_KEY."
        )
    # Printed by the script, not read off a banner: the engine's
    # `✅ LLM initialized (model=...)` line (:388) is emitted only on the
    # branch that succeeded, so its absence is the signal -- and an absence is
    # the one thing an operator scrolling stdout_llm.log does not notice. The
    # client class separates CommonStack from native Anthropic, which the
    # model slug alone does not.
    print(f"client={type(backtester.llm_client).__name__} model={backtester.model}")
    backtester.load_data()
    backtester.calculate_indicators()
    run_id, _curve = backtester.run_agent_backtest()
    print(run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

Run: `bash <scratchpad>/measure_start.sh <scratchpad>/measure rule` then `... llm`. Use the interpreter the test suite runs under (`python3` here; `uv run python3 ...` if the environment needs it). The LLM run spends real tokens through whatever `make_llm_client()` resolves — CommonStack, or native Anthropic when only `ANTHROPIC_API_KEY` is set. Record the slug `run_agent_backtest` prints, not "the default".

Two caveats on the LLM column that the rule-based column does not carry, both a consequence of driving the engine rather than the launcher. `spawn + interpreter` is real — the shell stamps `LAUNCHED` before `python3` starts, exactly as the parent does. But the launcher's own argparse, profile resolution and handoff preflight happen between `IMPORTS_DONE_AT` and the engine in a real child and are simply absent here, so the LLM row's `preflight remainder` is a **floor**, not the child's. And the driver's `schema_init_seconds` is read the same way the script reads it, so it is `0.00` here for the same reason it is `0.00` in the rule-based arm: SQLite, not a saving (below).

Two things this step is *not*. It is **not a DDL measurement**: with no `*_DATABASE_URL` set every store here resolves to its **SQLite** twin, `init_schema_unless_worker` is never reached, no `backend: schema init` line appears, and `schema DDL` prints `0.00s` because nothing ran — which is why the Final verification tables' local DDL column reads `n/a (sqlite)` and never `0.00`. (It prints `MISSING` instead only if the *script* never passed the key, which on a post-Task-5 build means something is wrong with the driver, not with the DDL. The reader keeps those two outcomes apart on purpose.) And it is **not a second instrument**: `main()` prints no phase timings and nothing in this plan adds any; its three numbered banners (`1️⃣`/`2️⃣`/`3️⃣`, `backtest_hourly_agent.py:473-492`) carry no times and are not phase names. The progress file and the child's `⏱  phase` stdout lines are the same clock, which is why both arms write one — `--progress-file` on the rule-based side, the `progress_file=` kwarg on the LLM side — even though no dashboard is watching.

A rule-based run **does** produce a `first_decision` row: `run_agent_backtest` publishes it at `engine.py:1423` regardless of `decision_source`. It will be short; record the number rather than `n/a`. That is also exactly why the driver refuses to start without a client: a degraded LLM arm writes the same row from the same code path, so nothing in the artifact or in `stdout_llm.log` would contradict it. **If the guard fires, leave the whole `local, LLM` column blank and say why** — do not fall back to running it anyway and labelling the result "LLM".

Record every phase duration from both runs — including the four-way `starting` split — in the **Final verification** tables below. `MISSING` is a result; a blank cell is not.

- [ ] **Step 2: Time the aggregation on its own**

Write `<scratchpad>/bench_aggregation.py`:

```python
"""Is aggregate_bars_by_symbol worth vectorising? Gate: > 2.0s on the onboarding shape.

The gate decides whether Step 4 is worth attempting. It is not the threshold
the result is judged against -- Step 4 removes the row walk only, which is
~11% of this function; see Step 5.
"""
import time
import numpy as np
import pandas as pd
import pytz
from dashboard.backend.domain.backtesting.bar_aggregation import aggregate_bars_by_symbol

eastern = pytz.timezone("US/Eastern")
days = pd.bdate_range("2026-09-08", "2026-09-16")
stamps = []
for day in days:
    start = eastern.localize(pd.Timestamp(day.date()) + pd.Timedelta(hours=9, minutes=30))
    stamps.extend(start + pd.Timedelta(minutes=5 * i) for i in range(78))
index = pd.DatetimeIndex(stamps).tz_convert("UTC")
rng = np.random.default_rng(0)
def frame():
    close = 100 + rng.standard_normal(len(index)).cumsum()
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
        "volume": rng.integers(100, 1000, len(index)), "vwap": close,
    }, index=index)
bars = {f"S{i:02d}": frame() for i in range(30)}
t0 = time.perf_counter()
aggregate_bars_by_symbol(bars, source_timeframe="5m", decision_timeframe="60m")
print(f"aggregate_bars_by_symbol: {time.perf_counter() - t0:.2f}s for 30 symbols x {len(index)} bars")
```

Run: `PYTHONPATH=. python3 <scratchpad>/bench_aggregation.py` from the repo root. Record the number. `PYTHONPATH=.` is load-bearing and an earlier draft of this line omitted it: `python3 <path>/script.py` puts the *script's* directory on `sys.path[0]` and never the cwd, so `from dashboard.backend...` raises `ModuleNotFoundError: No module named 'dashboard'` (verified 2026-09-20). `backtest_hourly_agent.py` gets away without it only because it carries the `if not __package__: ensure_repo_root()` bootstrap (`:34-37`); a scratch script has none.

- [ ] **Step 3: Apply the gate**

If the printed time is **at most 2.0s**, skip Steps 4-6, note "aggregation gate: not fired (<time>)" in the Final verification table, and go to Step 7.

If it is **above 2.0s**, continue — and read the gate for what it is. It decides whether the change is worth *attempting*; it is not the acceptance criterion for the change. It times the whole function, while Step 4 removes only the row walk, which is about a tenth of it (Step 5 carries the profile). An "after" reading still above 2.0s is the expected outcome here, not a failed step.

- [ ] **Step 4: Replace the per-row Series walk with an index walk plus groupby**

In `dashboard/backend/domain/backtesting/bar_aggregation.py`, replace the block from `buckets: dict[pd.Timestamp, list[pd.Series]] = {}` through `bucket_ends[bucket_start] = bucket_end` (`:120-137`) and `records: list[dict] = []` plus the loop head `for bucket_start in sorted(buckets):` / `group = pd.DataFrame(buckets[bucket_start])` / `group = group.sort_index()` / `bucket_end = bucket_ends[bucket_start]` (`:139-143`) with:

```python
    # Walk the index, not the rows: iterrows builds a Series per source bar,
    # and the onboarding shape has ~16k of them (7 weekdays x 78 five-minute
    # bars x 30 symbols). That is the cheapest thing here to remove, not the
    # expensive one, and the comment first drafted for this block claimed the
    # opposite. Measured 2026-09-21 under cProfile: iterrows is ~0.8s of this
    # function's ~9.2s cumulative, while the per-bucket work carries the rest
    # (_weighted_vwap ~1.9s, group.apply(pd.to_numeric) ~1.3s, plus a
    # DataFrame and a pd.date_range built for each of 1,470 buckets). End to
    # end this buys ~11%. It also lands in `loading_bars`, not `indicators`:
    # aggregate_bars_by_symbol is called from load_data (engine.py:723). The
    # per-bucket logic below is byte-for-byte what it was; only how rows find
    # their bucket changed.
    keep: list[bool] = []
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for timestamp in local.index:
        session = _session_for_timestamp(timestamp, windows)
        if session is None:
            keep.append(False)
            continue
        session_start, session_end = session
        elapsed_minutes = int((timestamp - session_start).total_seconds() // 60)
        offset_minutes = (elapsed_minutes // decision_minutes) * decision_minutes
        bucket_start = session_start + pd.Timedelta(minutes=offset_minutes)
        bucket_end = min(
            bucket_start + pd.Timedelta(minutes=decision_minutes), session_end
        )
        # A source bar can only belong to a decision bucket that has not ended.
        if bucket_start >= bucket_end:
            keep.append(False)
            continue
        keep.append(True)
        starts.append(bucket_start)
        ends.append(bucket_end)

    kept = local.loc[keep].copy()
    kept["_bucket_start"] = starts
    kept["_bucket_end"] = ends

    records: list[dict] = []
    for bucket_start, bucket in kept.groupby("_bucket_start", sort=True):
        bucket_end = bucket["_bucket_end"].iloc[0]
        group = bucket.drop(columns=["_bucket_start", "_bucket_end"]).sort_index()
```

The `records: list[dict] = []` line at `:139` **precedes** the old loop head rather than following it, and is inside the replaced range above, so it is declared exactly once (in the new block). Everything from `expected = int(` (`:144`) onward is unchanged. Re-derive both ranges with `grep -n "buckets: dict\|bucket_ends\[bucket_start\] = bucket_end\|records: list\[dict\] = \[\]\|bucket_end = bucket_ends\[bucket_start\]" dashboard/backend/domain/backtesting/bar_aggregation.py` before editing — they were `:112-131`/`:133-137` in an earlier draft of this plan and were wrong.

- [ ] **Step 5: Run the aggregation tests and the benchmark**

Run: `pytest dashboard/backend/tests/backtesting/test_bar_aggregation.py dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py -v`
Expected: all pass, unchanged assertions.

Run: `PYTHONPATH=. python3 <scratchpad>/bench_aggregation.py` (same reason as Step 2 — without it the `dashboard.*` import fails)
Expected: **lower than the Step 2 reading, and probably still above 2.0s.** Record before *and* after in the table, then stop. Do not keep cutting until the number clears the gate.

This line said "below 2.0s" until 2026-09-21, which is a target the Step 4 change cannot reach, and an implementer chasing it would either widen the diff well past this task or write a working step up as a failure. Measured on this checkout by applying Step 4 byte-for-byte to a scratch copy of the module: **4.53s → 4.06s**, and **4.72s → 4.04s** on a repeat with the two orders swapped — about 11%. The rewrite itself is sound; that was checked separately. Its output is `assert_frame_equal`-identical to the current implementation on a full US session, a CN session with the lunch break, a gappy/duplicate/off-grid frame, and a frame whose rows all fall outside the session. It simply is not where the time is.

cProfile on the unmodified function says where: `iterrows` is ~0.8s of 9.2s cumulative, against `_weighted_vwap` ~1.9s, the per-bucket `group.apply(pd.to_numeric)` ~1.3s, and a `DataFrame` plus a `pd.date_range`/`.difference()` built for each of the 1,470 buckets this shape produces.

So an above-gate "after" number is a **result to write into the table in words**: *the row walk was not the cost.* What remains is per-bucket work, and vectorising that is a different change with a different risk profile — it touches the quality counters and the VWAP weighting, which the gappy / duplicate / off-grid cases exist to protect. Out of scope for a measure-first task: if the number still matters once the `starting` split has landed and been read, file it rather than growing this step.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/backtesting/bar_aggregation.py
git commit -m "perf: walk the bar index instead of iterrows when aggregating

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 7: Record the measurement in this plan and commit**

Fill the **local** columns of the Final verification tables with the two runs and the gate outcome. The DDL table's local column is `n/a (sqlite)` for every store — write that, not `0.00`, and not a blank. The prod columns come from Task 7 Step 5. Then:

```bash
git add docs/superpowers/plans/2026-09-20-backtest-visible-start.md
git commit -m "docs: record the measured backtest start phases

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 8: Count the DDL the child no longer runs, and record it**

The phase clock measures wall time; this counts what was removed. It needs no
Neon: stub the pool and let the import graph speak.

Write `<scratchpad>/count_child_ddl.py`:

```python
"""How many schema statements does a backtest child run at import, and from which twins?

    PYTHONPATH=. python3 count_child_ddl.py before|after

The argument is not cosmetic: it decides whether the probe is a backtest child
or not. `after` sets ATL_BACKTEST_WORKER=1, which is what a real child gets
from run_backtest_background's env copy and what the guard added in Task 5
reads; `before` pops it, so an inherited value in the shell cannot quietly
arm a run that is supposed to be unguarded.
"""
import contextlib, importlib, os, sys, tempfile, traceback
MODE = sys.argv[1] if len(sys.argv) > 1 else "before"
assert MODE in ("before", "after"), "usage: count_child_ddl.py before|after"
os.environ["DATABASE_PATH"] = os.path.join(tempfile.mkdtemp(), "probe.db")
URL = "postgresql://u:p@db.example.invalid:5432/atl"
for var in ("USERS_DATABASE_URL", "CONTENT_DATABASE_URL", "AGENT_RUNS_DATABASE_URL"):
    os.environ[var] = URL
# Set before any store is built. The guard reads the environment per
# construction, so a later assignment would also work -- but the probe is
# simulating the child, and a child has this set from its first instruction.
if MODE == "after":
    os.environ["ATL_BACKTEST_WORKER"] = "1"
else:
    os.environ.pop("ATL_BACKTEST_WORKER", None)
import dashboard.backend.db_pool as db_pool
per = {}
def _who():
    for f in reversed(traceback.extract_stack()):
        if f.name == "_init_schema":
            return f.filename
    return "OUTSIDE__init_schema"
class Cur:
    def execute(self, sql, *a, **k): per.setdefault(_who(), []).append(sql); return self
    def fetchone(self): return None
    def fetchall(self): return []
    def __enter__(self): return self
    def __exit__(self, *a): return False
class Conn:
    def cursor(self, *a, **k): return Cur()
    def execute(self, sql, *a, **k): per.setdefault(_who(), []).append(sql); return Cur()
    def commit(self): pass
    def rollback(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
class Pool:
    @contextlib.contextmanager
    def connection(self): yield Conn()
db_pool.get_pool = lambda url: Pool()
sys.argv = ["backtest_hourly_agent.py"]
importlib.import_module("dashboard.scripts.backtest_hourly_agent")
for path, stmts in sorted(per.items(), key=lambda kv: -len(kv[1])):
    print(f"{os.path.relpath(path):70s} {len(stmts):4d}")
print(f"TOTAL {sum(len(v) for v in per.values())} statements over {len(per)} _init_schema calls")
# The accumulator the `starting` record carries, printed from the same process
# that just did (or skipped) the DDL, so the 0.0 a prod worker will publish is
# visible here rather than only inferable. getattr: on origin/main the function
# does not exist yet, and a NameError would bury the count above it.
seconds = getattr(importlib.import_module("dashboard.backend.db_url"),
                  "schema_init_seconds", None)
print(f"MODE {MODE}  schema_init_seconds "
      + (f"{seconds():.4f}" if seconds else "n/a (pre-Task-5 build)"))
```

Run it **twice, with different arguments, and that asymmetry is the point** —
the probe is simulating the child, not the parent. On `origin/main`:
`PYTHONPATH=. python3 <scratchpad>/count_child_ddl.py before`. On this branch:
`PYTHONPATH=. python3 <scratchpad>/count_child_ddl.py after`.

`origin/main` is run **without** the flag because it has no guard to arm —
setting it there changes nothing and would only suggest it had. This branch is
run **with** it, because that is the one input that makes the guard fire;
`ATL_BACKTEST_WORKER` is set by `run_backtest_background` in the child's env
copy and by nothing else, so a probe that imports the child in-process without
setting it is not a child. An earlier draft of this step omitted it and asked
for the "after" number anyway: run as written, every twin took the ran-it path
and the branch printed **107** again — read either as "the guard did nothing"
or as a bug hunt through a guard that works.

Expected `before` (re-measured 2026-09-20 by running exactly this probe): six
twins, **107** statements over six `_init_schema` calls — 55 / 18 / 16 / 14 /
3 / 1 for run history, model providers, analytics, agents, credits, brokers —
and `schema_init_seconds n/a (pre-Task-5 build)`. Expected `after`: **zero**
statements over **zero** `_init_schema` calls, `schema_init_seconds 0.0000`,
and six `… backend: schema init skipped (backtest worker)` lines above the
table. Those six lines are what fills the `twins constructed` cell of the
after column: the counter keys on frames named `_init_schema`, so a twin that
was constructed and skipped its DDL is invisible to it — a fact the `0`
statement count cannot distinguish from a twin that was never built at all.
Count the skip lines, do not infer them. Record all four rows below.

Two things worth knowing before reading the number. A warm prod schema does
**not** make these cheaper: 72 of the 107 are `ALTER TABLE … ADD COLUMN IF NOT
EXISTS`, and the five remaining `ALTER`s are two constraint drops, one
constraint add and two `ALTER COLUMN … DROP NOT NULL` — all of which round-trip
whether or not the column, constraint or nullability is already as asked;
the rest are 10 `CREATE TABLE`, 8 `CREATE INDEX`, 1 `CREATE UNIQUE INDEX`, 6
`INSERT`, 2 `DO`, 2 `UPDATE`, 1 `SELECT`. And the per-twin figures are
**indicative, not pinned**: they move whenever any twin's `_init_schema` gains
a statement, and nothing fails when they do. What *is* pinned is the shape —
`test_no_postgres_twin_runs_ddl_outside_the_guard` asserts no twin runs DDL
outside the guard, which is the claim the CLAUDE.md bullet makes. Re-run the
script rather than quoting this table if a later change needs the number.

`DATABASE_PATH` is redirected to a temp file above so the committed seed DB is
untouched; confirm rather than assume — `git status --short
dashboard/storage/data/backtest.db` must be empty, and
`git checkout -- dashboard/storage/data/backtest.db` if it is not.

```bash
git add docs/superpowers/plans/2026-09-20-backtest-visible-start.md
git commit -m "docs: record the schema DDL the backtest child no longer runs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Full verification and PR

- [ ] **Step 1: Full suite**

Run: `pytest dashboard/backend/tests/ -q`
Expected: green (skips only for `node`-less or `@pg_only` cases if those apply on this machine).

- [ ] **Step 2: Seed DB untouched**

Run: `git status --short dashboard/storage/data/backtest.db`
Expected: empty. If the file changed, `git checkout -- dashboard/storage/data/backtest.db`.

- [ ] **Step 3: Cache-buster consistency**

Run: `grep -rn "app.js?v=\|styles.css?v=" dashboard/frontend/app.html dashboard/backend/tests/*.py`
Expected: one number per asset across all files.

- [ ] **Step 4: Open the PR**

```bash
git push -u origin feat/backtest-visible-start
gh pr create --title "feat: show the backtest phase from the first poll" --body "$(cat <<'EOF'
Track A of docs/superpowers/specs/2026-09-20-backtest-speed-and-trust-design.md.

- child publishes `starting → loading_bars → indicators → first_decision → running → saving` to the progress file, with timestamps
- `/backtest/status` names the phase; the My Agents card shows a short label and `0/N` once bars are counted
- child skips Postgres schema DDL under `ATL_BACKTEST_WORKER=1` (parent-set only) — all eleven twins that own a schema, through one shared `db_url.init_schema_unless_worker`, not a copied guard per store
- `init_schema_unless_worker` prints and times both directions, so the parent's own boot log is the pre-change DDL baseline
- measured start phases, with `starting` split into spawn / imports+stores / of-which-DDL / preflight, in the plan's final tables

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: After deploy, read both halves out of the service log**

Prod is the only place Neon DDL and connection setup exist at all — locally every store resolves to its SQLite twin, which this plan never guards, so a local before/after of `ATL_BACKTEST_WORKER` measures exactly zero and would fill the table with a saving that did not happen.

Do **not** race a live poll for the phases: `/backtest/status` exposes `phases[]` only while the run is in flight, and the parent unlinks the progress file the moment it ends (`backtests.py:1810`, inside `run_backtest_background`'s `finally` at `:1771`). The child prints each transition and the `starting` split to stdout, and the parent dumps that — bounded head+tail, `SUBPROCESS_LOG_HEAD_CHARS`/`_TAIL_CHARS`, 32k each (`:2299-2300`) — into the service log under `=== BACKTEST SCRIPT OUTPUT ===` (`:1628`). These lines are at the head, so truncation cannot eat them.

Two readings, both from the deploy that ships this PR:

1. **Pre-change — the parent's own boot.** Open the Render deploy log for `srv-d7lbmpjbc2fs73bcr6t0` and `grep 'backend: schema init'`. The parent is not a worker, so it runs every store's DDL against the same Neon databases the child used to: eleven timed lines, six of which name the twins a child constructs, and those six are the round trips the child no longer pays. Note which line is preceded by `🏊 pg pool created for …` — that store's figure carries the cold handshake for its URL, and the later stores on the same URL do not.
2. **After — the child.** Launch one onboarding backtest, then grep the same log for the child's `schema init skipped` lines and its `⏱  phase` / `spawn+interpreter …` lines. Expect **six** skip lines (the six twins its import graph builds, not eleven) and `schema DDL 0.00s` in the split. That zero is the check that the flag actually fired in production — it is not a saving. Five skips and a sixth *timed* line is the flag failing to arrive at one twin, which is exactly the case the timed direction exists to make visible.

The parent's timings are a proxy for the child's and are labelled one: same statements, same databases, same Render-to-Neon path, but the parent boots on a warmed filesystem, so the two are comparable in shape and not to the second. It is also a single sample with no repeat, so the table records one run and says so.

Fill both tables below, then commit the update **on a fresh branch** — never on this one after it merges.

---

## Final verification

Phase durations in seconds, from `phases[]` in each run's progress file (`measure_start.sh` prints them) or from the child's `⏱  phase` stdout lines. **A blank cell is not a result** — write `MISSING` if the phase did not appear. Do not clamp a negative duration to zero.

`starting` is one phase name and four numbers. The first four rows are its split; do not read an import cost or a Neon cost out of the total alone.

| measurement | local, rule-based | local, LLM | prod, first run after deploy |
|---|---|---|---|
| spawn + interpreter (`child_entered_at − started_at`) | 0.03 | 0.01 | |
| imports incl. stores (`imports_done_at − child_entered_at`) | 2.59 | 2.45 | |
| — of which schema DDL (`schema_init_seconds`) | n/a (sqlite) | n/a (sqlite) | 0.00 (worker — see below) |
| preflight remainder (`ended_at − imports_done_at`) | 0.08 | 0.09 (**floor** — see below) | |
| **starting** (parent's launch stamp → the child's first `publish_phase`, total) | **2.70** | **2.55** | |
| loading_bars | **18.15** | **18.67** | |
| indicators | 0.35 | 0.34 | |
| first_decision | 0.32 | **13.80** | |
| aggregation gate (Task 6 Step 3) | **fired** — 4.58s → 4.13s | n/a | n/a |

`first_decision` has a real number in the rule-based column: `run_agent_backtest` publishes the phase at `engine.py:1423` regardless of `decision_source`. It is short there, which is itself the point of comparison.

The **aggregation gate** row takes two numbers when it fires, not one, and the second is very likely still above the gate — measured 2026-09-21, Step 4 moves this function from ~4.5s to ~4.1s. Write both and write the conclusion beside them in a sentence: the row walk was ~0.8s of a 9.2s profile, so removing it was worth doing and was never going to be the fix. See Task 6 Step 5 for the profile and for what is deliberately left alone.

The **local, LLM** column comes from an in-process driver rather than from `backtest_hourly_agent.py`, because the launcher exits 2 on an LLM run without a signed handoff (Task 6 Step 1). Two consequences belong beside the numbers rather than being rediscovered from them. Its `spawn + interpreter` is real — the shell stamps the launch clock before `python3` starts, exactly as the parent does — and its `imports incl. stores` is comparable, because the driver imports `dashboard.scripts.backtest_hourly_agent` for its cost. But its `preflight remainder` is a **floor**: the launcher's argparse, profile resolution and handoff preflight sit between `imports_done_at` and the engine in a real child and are absent here. Label that cell, do not quietly compare it across columns.

`starting` is the headline number of this plan, not a footnote: it is the entire window the card cannot narrate, and this is the first time anyone measures it. Reference point taken **2026-09-20** on WSL2 with local SQLite: importing `dashboard.scripts.backtest_hourly_agent` alone cost **2.43 / 2.46 / 2.61 s** over three runs and constructed **7 of the repo's 12** module-level store singletons — no parent setup, no `Popen`, no argparse, no Neon. Bare interpreter spawn was **~14 ms** (median of 5), so the imports are essentially the whole gap. Treat ~2.5 s as a floor, not an estimate; if prod's number is large, the answer is a follow-up that defers those imports out of the child's module scope, not a sentence.

Phase sentences: `starting` has none, on either surface, and two tests say so (`test_backtest_progress_status.py`, `test_backtest_progress_card.py`).

Schema DDL per store, from the `<store> backend: schema init …` lines. The **parent boot** column is the pre-change number — the parent runs this same DDL against the same databases and is never a worker. Six of the eleven are twins a child constructs; the other five are guarded so a future import cannot re-add an unguarded payer, and their prod-child cell is `not constructed`, which is not the same fact as `skipped`.

| store | local | prod parent boot (pre-change) | prod child (after) |
|---|---|---|---|
| run history | n/a (sqlite) | | skipped |
| credits_store | n/a (sqlite) | | skipped |
| analytics_store | n/a (sqlite) | | skipped |
| model_provider_store | n/a (sqlite) | | skipped |
| agent_store | n/a (sqlite) | | skipped |
| broker_connections | n/a (sqlite) | | skipped |
| user_store | n/a (sqlite) | | not constructed |
| agent_version_store | n/a (sqlite) | | not constructed |
| agent_credential_store | n/a (sqlite) | | not constructed |
| portfolio_store | n/a (sqlite) | | not constructed |
| strategy_store | n/a (sqlite) | | not constructed |

| child schema DDL at import (Task 6 Step 8) | before — `origin/main`, probe arg `before` | after — this branch, probe arg `after` |
|---|---|---|
| twins constructed | 6 of 11 guarded | **6** (counted from the `schema init skipped` lines, not inferred from the statement count) |
| statements | 107 | **0** |
| `_init_schema` calls | 6 | **0** |
| `db_url.schema_init_seconds()` | n/a (pre-Task-5 build) | **0.0000** |

The two probe arguments are not symmetric and must not be made so: `before` pops `ATL_BACKTEST_WORKER`, `after` sets it to `1`. `origin/main` has no guard to arm, so the flag there would change nothing while implying it had; this branch has nothing *but* the flag to distinguish a child from a parent. And the `twins constructed` cell of the after column is read off the six skip lines, not off the statement count — the counter keys on frames named `_init_schema`, so a twin that skipped its DDL and a twin that was never built produce the same `0`.

How to read these, because they are not all the same kind of number:

- **Local is `n/a (sqlite)`, never `0.00`.** With no `*_DATABASE_URL` set every store resolves to its SQLite twin, which this plan never guards, so `init_schema_unless_worker` is never reached. A `0.00` here would be a fabricated saving, indistinguishable from a measurement a month from now.
- **The prod child's zero is a check, not a saving.** It says the flag fired. The saving is the parent-boot column.
- **The parent's figures are a proxy.** Same statements, same databases, same Render-to-Neon path; different process warmth. Comparable in shape, not to the second. One sample, no variance estimate.
- **Connection setup is in none of the DDL figures** — it sits inside `imports incl. stores`, and a guarded child that never queries during import opens no pool at all. If that row dwarfs the DDL row, the dark start is imports, not schema, and the next optimisation is not more DDL work. Write that conclusion down if the numbers say it; measuring first was what earned the right to reach it.

### What the local numbers actually say (measured 2026-09-21)

They do not say what this plan assumed when it was written, and the honest reading
is worth more than the tidy one.

**The dark window is ~21 s, and `loading_bars` is ~86% of it** — 18.15 s of 21.20 s
rule-based (85.6%) and 18.67 s of 21.56 s on the LLM arm (86.6%), against 2.59 s / 2.45 s for the whole
of imports-and-stores. Both arms agree to within half a second on every pre-loop
phase, which is what makes the split trustworthy: they share no process, no code path
into the engine, and no launcher.

Three consequences follow, and the last one is the one to act on.

- **The DDL guard's local saving is zero, and that is not a disappointment — it is the
  measurement working.** Every store here resolves to its SQLite twin, so
  `init_schema_unless_worker` is never reached and `schema DDL` is `n/a (sqlite)`, never
  `0.00`. The guard's payoff is the **107 statements over 6 `_init_schema` calls** the
  Step 8 probe counted, which on Render are Neon round trips rather than local no-ops.
  Counting statements instead of trusting this wall clock is precisely why Step 8 exists;
  a local before/after would have reported a saving of nothing and been believed.
- **The aggregation gate fired and the rewrite worked, and it still was not the fix.**
  4.58 s → 4.13 s. Four independent measurement sets (implementer, controller ×2, reviewer)
  put the before arm at **4.57–4.95 s** and the after arm at **4.07–4.32 s** — a consistent
  ~10–14% improvement, with the before arm visibly the noisier of the two, so read the
  single-pair figure as a midpoint rather than a precise delta. ~10% of one function that is itself a
  fraction of `loading_bars`. The row walk was ~0.8 s of a ~9.2 s profile, so removing it
  was worth doing and was never going to be the fix; `_weighted_vwap` and the per-bucket
  work carry the rest and are deliberately untouched.
- **The next optimisation is bar loading, not schema and not aggregation.** `loading_bars`
  is where the user's dark window lives, and nothing in Track A touches it. That is a
  correct outcome for a plan whose job was to make the window *visible* rather than
  short — but it means "the start is slow" now has a measured owner, and it is not the
  one this plan set out to blame. File it against bar fetching/caching; do not reach for
  more DDL work.

`first_decision` is the other number that moved: **0.32 s rule-based against 13.80 s on
the LLM arm** (deepseek/deepseek-v4-pro through the CommonStack gateway). That gap is the
entire argument for naming the phase — it is 13.8 s in which a card showing `0%` and an
elapsed timer is indistinguishable from a hung run.


Full suite: `pytest dashboard/backend/tests/ -q` → **5103 passed, 165 skipped, 0 failed** (335.4s, 2026-09-21). Two earlier full runs on this branch — the controller's at `060cc93a` and the final reviewer's independent one at `22e717bf` — both read **5102 / 165 / 0**; the +1 is the cross-language phase-label guard added for the final review's one Important finding. `origin/main` measured 5046 on the same command before this branch, so the branch adds 57 cases net.
Seed DB: clean (`git status --short dashboard/storage/data/backtest.db` empty, after Task 6 Steps 1 and 8 as well as at the end). Both of those steps redirect `DATABASE_PATH` at a scratch file precisely because they would otherwise write real runs into a tracked one — check anyway; the redirect is the fix, the check is the proof.
Cache-busters: one number per asset, **five** files (`dashboard/frontend/app.html` plus `test_frontend_fast_boot.py`, `test_backtest_comparison_frontend.py`, `test_analytics_frontend.py`, `test_admin_analytics_frontend.py`) — re-derived 2026-09-20 with `grep -rln "app.js?v=" dashboard/frontend/app.html dashboard/backend/tests/*.py`, unchanged by this plan. The count is per-document and moves when a test file starts or stops loading `app.js`; grep it, never carry it forward.
Store twins guarded: **eleven** of the twelve pairs in `test_store_twin_parity.py::_TWINS` (`PostgresValueAnalyticsStore` has no `_init_schema`; its exemption is that file's `_NO_OWN_DDL_TWINS`), of which **six** are the twins the child's import graph actually constructs. `test_backtest_worker_schema_skip.py` → **26** cases (`2 × 11 + 4` — the eleven entries of *that file's own* `_TWINS` list, parametrized twice, plus four named cases). Resolve the symbol to the twelve-pair parity registry named one clause earlier and the arithmetic gives 28, which is the wrong answer to a different question: the two lists are deliberately different lengths, because the twelfth pair owns no DDL. All pass. Re-derive both numbers with the probe in Task 5's preamble before trusting either.

## Out of scope (from the spec)

Per-bar deadlines, dropping `PROVIDER_TIMEOUT` from backtest failover, moving the analytics snapshot rebuild off the hot path, bar caching, the cadence and result-contract doors. Track B (pinned sampling) is its own plan: `docs/superpowers/plans/2026-09-20-backtest-pinned-sampling.md`.

## User-facing docs to check after this ships

`docs/source/lab/operating_modes.rst` and `docs/source/lab/key_features.rst` describe the backtest flow; neither currently claims anything about progress. Re-read once the phase card is live and file a docs follow-up if a screenshot or sentence is now stale.
