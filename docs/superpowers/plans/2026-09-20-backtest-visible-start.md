# Backtest Visible Start (Track A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dashboard backtest's card names what the child is doing from the first poll tick, the child stops repeating the parent's schema DDL, and the pre-loop phases are measured by the same payload that drives the card.

**Architecture:** The engine gains a second progress-file writer, `publish_phase`, beside `_publish_live_progress`; each phase write carries the phase in progress plus the finished phases with timestamps. The parent passes its launch time so the child can account for the gap before it could write anything. `/backtest/status` turns the phase into a sentence; the Backtest panel prints that sentence and the My Agents card prints a short label. The four Postgres store twins skip `_init_schema()` when the parent marks the process as a backtest worker.

**Tech Stack:** Python 3 / FastAPI backend (`dashboard.backend` package), vanilla JS frontend lifted into `node -e` by `dashboard/backend/tests/_frontend_source.py`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-20-backtest-speed-and-trust-design.md` (Track A sections).

## Global Constraints

- Run everything from the repo root. Tests: `pytest dashboard/backend/tests/<file> -v`. Node must be on `PATH` for the `test_*card*.py` / `*_frontend.py` modules (they skip, not fail, without it).
- Phase names are exactly `starting`, `loading_bars`, `indicators`, `first_decision`, `running`, `saving`. Nothing else is accepted.
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
- Modify: `dashboard/backend/domain/backtesting/engine.py` (imports at `:18`; constructor kwargs at `:180-205` and attribute block at `:236-238`; `_publish_live_progress` at `:588-643`; `load_data` at `:645`; `calculate_indicators` at `:876`; `run_agent_backtest` after `total_steps = len(all_timestamps)` at `:1423` and before the agent `db.insert_run(` at `:1748`)
- Test: `dashboard/backend/tests/backtesting/test_engine_progress_phases.py` (new)

**Interfaces:**
- Produces: `HourlyBacktester(..., launched_at: Optional[float] = None)`; `HourlyBacktester.publish_phase(name: str, *, total_steps: Optional[int] = None) -> None`; `HourlyBacktester._init_progress_phases(launched_at: Optional[float] = None) -> None`; module constant `PROGRESS_PHASES: tuple[str, ...]`. Progress-file payload gains `phase: str | None`, `phase_started_at: float | None`, `phases: list[{name, started_at, ended_at}]`; a phase write carries `step: 0`, `total_steps: int` (0 until known) and `equity_curve: []`.

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
import json
import time
from types import SimpleNamespace

import pytest

from dashboard.backend.domain.backtesting.engine import (
    PROGRESS_PHASES,
    HourlyBacktester,
)


def _bare(tmp_path, launched_at=None):
    backtester = object.__new__(HourlyBacktester)
    backtester.progress_file = str(tmp_path / "progress.json")
    backtester.live_run_id = "agent_phases"
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


def test_legacy_new_construction_still_publishes_live_progress(tmp_path):
    """Callers that build the engine with __new__ (tests, legacy tools) never
    ran _init_progress_phases; the live writer must not care."""
    backtester = object.__new__(HourlyBacktester)
    backtester.progress_file = str(tmp_path / "progress.json")
    backtester.live_run_id = "agent_legacy"

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


def test_write_failure_is_reported_not_raised(tmp_path, capsys):
    backtester = _bare(tmp_path)
    backtester.progress_file = str(tmp_path / "missing" / "progress.json")

    backtester.publish_phase("loading_bars")

    assert "Could not write live progress" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/backtesting/test_engine_progress_phases.py -v`
Expected: FAIL with `ImportError: cannot import name 'PROGRESS_PHASES'`.

- [ ] **Step 3: Add the constant, the import and the constructor kwarg**

In `dashboard/backend/domain/backtesting/engine.py`:

Add `import time` directly below `import json` (`:18`).

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

In `HourlyBacktester.__init__` (`:180`), append one keyword after `source_timeframe: Optional[str] = None,`:

```python
        launched_at: Optional[float] = None,
```

Directly after `self.progress_file = (progress_file or "").strip() or None` (`:238`), add:

```python
        self._init_progress_phases(launched_at)
```

- [ ] **Step 4: Add the phase machinery above `_publish_live_progress`**

Insert immediately before `def _publish_live_progress(` (`:588`):

```python
    # -- Progress phases -----------------------------------------------------
    #
    # `_publish_live_progress` is the only writer once the bar loop starts;
    # before it, nothing wrote anything, and the card sat on its launch
    # sentence for the whole start (imports, four stores' DDL, the bar fetch,
    # aggregation, indicators, then bar 1's full pipeline). Each phase write
    # carries the phase in progress plus the finished ones with timestamps, so
    # the same payload that drives the card is the measurement of the start.
    # Every accessor tolerates an instance built with __new__ (tests, legacy
    # tools) that never ran _init_progress_phases.

    def _init_progress_phases(self, launched_at: Optional[float] = None) -> None:
        """Open the implicit `starting` phase at the parent's launch time.

        The parent alone can see the gap between spawning the child and the
        child's first write (imports, store startup); passing its clock in is
        how that gap becomes a measured phase instead of a missing one.
        """
        self._progress_phase: Optional[str] = "starting" if launched_at else None
        self._progress_phase_started_at: Optional[float] = (
            float(launched_at) if launched_at else None
        )
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
        now = time.time()
        if self._progress_phase is not None:
            self._progress_phases.append(
                {
                    "name": self._progress_phase,
                    "started_at": self._progress_phase_started_at,
                    "ended_at": now,
                }
            )
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

        A phase payload carries `step: 0`, the bar count once it is known,
        and an empty curve, so every existing reader of the file (chart,
        trading log, ETA anchor) keeps its early return and only the message
        and the bar change.
        """
        self._set_progress_phase(name, total_steps=total_steps)
        if not self.progress_file:
            return
        self._write_progress_payload(
            {
                "run_id": self.live_run_id,
                "step": 0,
                "total_steps": self._progress_total_steps,
                "equity_curve": [],
                **self._progress_phase_fields(),
            }
        )

    def _write_progress_payload(self, payload: Dict) -> None:
        from pathlib import Path

        try:
            Path(self.progress_file).write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            print(f"   ⚠️  Could not write live progress: {exc}")
```

- [ ] **Step 5: Make `_publish_live_progress` set `running` and share the writer**

In `_publish_live_progress` (`:588`), directly after the existing early return

```python
        if not self.progress_file:
            return
```

add:

```python
        self._set_progress_phase("running", total_steps=total_steps)
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

Still in `run_agent_backtest`, immediately before the `db.insert_run(` call that writes the agent's own row (the first of three `db.insert_run(` calls in the file, `:1748` at the time of writing; the two later ones write baselines), add:

```python
        self.publish_phase("saving")
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/backtesting/test_engine_progress_phases.py dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py dashboard/backend/tests/test_agent_runs_metadata.py -v`
Expected: all pass (the ifind module drives `_publish_live_progress` on a `__new__` instance at `:286`; it must still pass).

- [ ] **Step 8: Commit**

```bash
git add dashboard/backend/domain/backtesting/engine.py dashboard/backend/tests/backtesting/test_engine_progress_phases.py
git commit -m "feat: publish pre-loop phases to the backtest progress file

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The parent passes its launch time; the child accepts it

**Files:**
- Modify: `dashboard/scripts/backtest_hourly_agent.py` (argparse block around `:223-224`; the `HourlyBacktester(` call at `:443`)
- Modify: `dashboard/backend/api/routers/backtests.py:1594` (the `--run-id` / `--progress-file` argv line inside `run_backtest_background`)
- Test: `dashboard/backend/tests/test_backtest_launch_phases.py` (new)

**Interfaces:**
- Consumes: `HourlyBacktester(launched_at=...)` from Task 1.
- Produces: child argv `--launched-at <epoch seconds, 3 decimals>`, always present for dashboard launches.

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

In `dashboard/scripts/backtest_hourly_agent.py`, directly after the `--progress-file` `add_argument` (`:224`), add:

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

In the `backtester = HourlyBacktester(` call (`:443`), add after `execution_client=execution_client,`:

```python
        launched_at=args.launched_at,
```

- [ ] **Step 4: Stamp the launch time in the parent's argv**

In `run_backtest_background` (`dashboard/backend/api/routers/backtests.py:1594`), replace

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
            "--launched-at", f"{time.time():.3f}",
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
- Modify: `dashboard/backend/api/routers/backtests.py` (new helper beside `_read_progress_file` at `:1381`; the running branch of `get_backtest_status` at `:3459-3466`)
- Test: `dashboard/backend/tests/test_backtest_progress_status.py` (append), `dashboard/backend/tests/test_backtests_router.py` (append after `test_backtest_status_includes_live_progress` at `:1325`)

**Interfaces:**
- Consumes: progress payload fields `phase`, `step`, `total_steps` from Task 1.
- Produces: `PROGRESS_PHASE_MESSAGES: dict[str, str]` and `_progress_message(progress: Optional[Dict[str, Any]]) -> str` in `backtests.py`. `/backtest/status` `message` becomes the phase sentence while `step == 0`.

- [ ] **Step 1: Write the failing helper tests**

Append to `dashboard/backend/tests/test_backtest_progress_status.py`:

```python
import pytest


_DEFAULT = "Backtest is running… (multi-step agent pipeline; may take several minutes)"


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        ("starting", "Starting backtest…"),
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
#: the phase clock are read in this process. `running` is absent on purpose --
#: a step count owns that sentence.
PROGRESS_PHASE_MESSAGES = {
    "starting": "Starting backtest…",
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
    if step > 0 and total > 0:
        pct = min(99, round(100 * step / total))
        return f"Backtest running… step {step}/{total} ({pct}%)"
    message = PROGRESS_PHASE_MESSAGES.get(str(progress.get("phase") or ""))
    if message is None:
        return _PROGRESS_DEFAULT_MESSAGE
    if total > 0 and progress.get("phase") == "first_decision":
        return f"{message} ({total} decision bars queued)"
    return message
```

- [ ] **Step 4: Use it in the running branch**

In `get_backtest_status`, replace (`:3459-3466`)

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

The Backtest panel already prints `status.message`, so Task 3 reaches it with no change. This task covers the My Agents card (one short line) and the two progress-store functions that both cards derive from.

**Files:**
- Modify: `dashboard/frontend/app.js` — add `BACKTEST_PHASE_LABELS` + `formatBacktestPhase` above `function formatStartupStaleness(` (`:7893`); edit `resolveRunningNotice` (`:7907`), `advanceBacktestProgress` (`:7929`), `deriveRunningProgress` (`stepLabel`/`detail` near the end of the function, `:8010-8016`)
- Modify: the five `app.js?v=` pins
- Test: `dashboard/backend/tests/test_agent_card_live_chart.py` (edit `_fold` at `:81`, append tests), `dashboard/backend/tests/test_backtest_progress_card.py` (edit `_PROGRESS_HELPERS` at `:40`, `_render` at `:68`, append tests)

**Interfaces:**
- Consumes: progress payload `phase`, `phase_started_at`, `progress_age_seconds`, `step`, `total_steps`.
- Produces: `const BACKTEST_PHASE_LABELS`; `function formatBacktestPhase(phase) -> string`; `advanceBacktestProgress` now returns `{step: 0, totalSteps, phase, phaseStartedAt, equityCurve: [], openingEquity: null, ageSeconds, ageAt}` for a phase tick (no `firstStep`/`firstStepAt` keys); `deriveRunningProgress` fills `stepLabel` with `0/N` and `detail` with the phase label before the first step.

- [ ] **Step 1: Write the failing fold tests**

In `dashboard/backend/tests/test_agent_card_live_chart.py`, change `_fold` (`:81`) to lift the new helper:

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
```

- [ ] **Step 2: Write the failing card tests**

In `dashboard/backend/tests/test_backtest_progress_card.py`, add `"function formatBacktestPhase(",` as the first entry of `_PROGRESS_HELPERS` (`:40`), and in `_render` (`:68`) add `js_const("BACKTEST_PHASE_LABELS"),` directly after `js_const("BACKTEST_STALE_SECONDS"),`. Append:

```python
def test_card_names_the_phase_before_the_first_step():
    html = _render(
        "{step: 0, totalSteps: 49, phase: 'first_decision',"
        " ageSeconds: 2, ageAt: Date.now(), elapsedSeconds: 40}"
    )
    assert "Waiting on first decision" in html
    assert "0/49" in html
    assert "Still starting up" not in html


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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_agent_card_live_chart.py dashboard/backend/tests/test_backtest_progress_card.py -v`
Expected: FAIL — `js_const("BACKTEST_PHASE_LABELS")` raises because the constant does not exist.

- [ ] **Step 4: Add the labels and the helper**

In `dashboard/frontend/app.js`, directly above `function formatStartupStaleness(` (`:7893`), add:

```js
/**
 * Short labels for the pre-loop phases the child publishes (engine.py
 * PROGRESS_PHASES). The Backtest panel prints the server's own sentence; the
 * agent card has one short line, so it gets these. The phase *names* are the
 * contract between the two; each surface owns only its wording. `running` is
 * empty on purpose: a step count owns that line.
 */
const BACKTEST_PHASE_LABELS = {
    starting: 'Starting',
    loading_bars: 'Loading market data',
    indicators: 'Calculating indicators',
    first_decision: 'Waiting on first decision',
    running: '',
    saving: 'Saving results',
};

function formatBacktestPhase(phase) {
    return typeof phase === 'string' ? (BACKTEST_PHASE_LABELS[phase] || '') : '';
}
```

- [ ] **Step 5: Make the notice phase-aware**

Replace `resolveRunningNotice` (`:7907`) with:

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
    return age === null ? null : formatProgressStaleness(age);
}
```

- [ ] **Step 6: Fold a phase tick**

In `advanceBacktestProgress` (`:7929`), replace the opening

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
            determinate ? `${pct}%` : formatBacktestPhase(running.phase),
            eta,
        ].filter(Boolean).join(' · '),
```

- [ ] **Step 8: Bump the cache-buster in five files**

```bash
grep -rn "app.js?v=" dashboard/frontend/app.html dashboard/backend/tests/*.py
```

Take the number N shown (identical everywhere) and replace `app.js?v=N` with `app.js?v=N+1` in every file listed. Re-run the grep: exactly one number, five files.

- [ ] **Step 9: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_agent_card_live_chart.py dashboard/backend/tests/test_backtest_progress_card.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_backtest_comparison_frontend.py dashboard/backend/tests/test_analytics_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_running_backtest_store.py -v`
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add dashboard/frontend/app.js dashboard/frontend/app.html dashboard/backend/tests/
git commit -m "feat: name the backtest phase on the agent card before step one

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: The child skips the parent's schema DDL

**Files:**
- Modify: `dashboard/backend/db_url.py` (new constant + function; add `import os`)
- Modify: `dashboard/backend/database_postgres.py:56-59`, `dashboard/backend/domain/credits/repository_postgres.py:663-665`, `dashboard/backend/domain/analytics/repository_postgres.py:193-195`, `dashboard/backend/domain/model_providers/repository_postgres.py:152-154` (each twin's `__init__`)
- Modify: `dashboard/backend/api/routers/backtests.py:1540` (the `env = os.environ.copy()` block)
- Modify: `dashboard/backend/tests/conftest.py:88` (strip list)
- Modify: `CLAUDE.md` (Environment & credentials bullets)
- Test: `dashboard/backend/tests/test_backtest_worker_schema_skip.py` (new); `dashboard/backend/tests/test_backtest_launch_phases.py` (append)

**Interfaces:**
- Produces: `db_url.BACKTEST_WORKER_ENV = "ATL_BACKTEST_WORKER"`; `db_url.schema_init_skipped() -> bool`. Child env carries `ATL_BACKTEST_WORKER=1`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_backtest_worker_schema_skip.py`:

```python
"""A dashboard backtest child must not repeat the parent's schema DDL.

The parent runs every store's `_init_schema` at import, so the schema exists
before it spawns anything. A child that ran it again paid a fresh pooled
connection plus a batch of DDL round trips per store to Neon before reading
a single bar -- pure start-up waste, and invisible because nothing before the
bar loop was measured. The flag names the process role, not the DDL, so no
operator sets it globally and later child-only behaviour has a home.
"""
import pytest

from dashboard.backend import database_postgres, db_url
from dashboard.backend.domain.analytics import repository_postgres as analytics_pg
from dashboard.backend.domain.credits import repository_postgres as credits_pg
from dashboard.backend.domain.model_providers import (
    repository_postgres as providers_pg,
)

_URL = "postgresql://atl:not-a-secret@db.example.invalid:5432/atl_test"

_TWINS = [
    (database_postgres, "PostgresBacktestDatabase"),
    (credits_pg, "PostgresCreditsStore"),
    (analytics_pg, "PostgresAnalyticsStore"),
    (providers_pg, "PostgresModelProviderStore"),
]


def test_only_the_literal_one_arms_the_flag(monkeypatch):
    monkeypatch.delenv(db_url.BACKTEST_WORKER_ENV, raising=False)
    assert db_url.schema_init_skipped() is False
    monkeypatch.setenv(db_url.BACKTEST_WORKER_ENV, "1")
    assert db_url.schema_init_skipped() is True
    monkeypatch.setenv(db_url.BACKTEST_WORKER_ENV, "true")
    assert db_url.schema_init_skipped() is False


@pytest.mark.parametrize(("module", "name"), _TWINS, ids=[n for _m, n in _TWINS])
def test_a_worker_skips_schema_init(monkeypatch, capsys, module, name):
    calls = []
    cls = getattr(module, name)
    monkeypatch.setattr(cls, "_init_schema", lambda self: calls.append("ddl"))
    monkeypatch.setenv(db_url.BACKTEST_WORKER_ENV, "1")

    cls(_URL)

    assert calls == []
    assert "schema init skipped" in capsys.readouterr().out


@pytest.mark.parametrize(("module", "name"), _TWINS, ids=[n for _m, n in _TWINS])
def test_the_parent_still_runs_schema_init(monkeypatch, module, name):
    calls = []
    cls = getattr(module, name)
    monkeypatch.setattr(cls, "_init_schema", lambda self: calls.append("ddl"))
    monkeypatch.delenv(db_url.BACKTEST_WORKER_ENV, raising=False)

    cls(_URL)

    assert calls == ["ddl"]
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
Expected: FAIL with `AttributeError: module 'dashboard.backend.db_url' has no attribute 'BACKTEST_WORKER_ENV'` and a `KeyError: 'ATL_BACKTEST_WORKER'`.

- [ ] **Step 3: Add the flag reader**

In `dashboard/backend/db_url.py`, add `import os` to the imports, and append at the end of the module:

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
```

- [ ] **Step 4: Guard the four twins**

In each twin's `__init__`, replace the bare `self._init_schema()` call with the guarded form and import the reader beside the existing `require_postgres_url` import.

`dashboard/backend/database_postgres.py` — change the import to `from dashboard.backend.db_url import require_postgres_url, schema_init_skipped` and the constructor to:

```python
    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        self._sqlite = BacktestDatabase()   # hot half: idempotency_keys stays local
        if schema_init_skipped():
            print("run history backend: schema init skipped (backtest worker)", flush=True)
        else:
            self._init_schema()
```

`dashboard/backend/domain/credits/repository_postgres.py`:

```python
    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        if schema_init_skipped():
            print("credits_store backend: schema init skipped (backtest worker)", flush=True)
        else:
            self._init_schema()
```

`dashboard/backend/domain/analytics/repository_postgres.py`:

```python
    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        if schema_init_skipped():
            print("analytics_store backend: schema init skipped (backtest worker)", flush=True)
        else:
            self._init_schema()
```

`dashboard/backend/domain/model_providers/repository_postgres.py`:

```python
    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        if schema_init_skipped():
            print("model_provider_store backend: schema init skipped (backtest worker)", flush=True)
        else:
            self._init_schema()
```

In each of the three domain twins, extend the existing `from dashboard.backend.db_url import require_postgres_url` line to `from dashboard.backend.db_url import require_postgres_url, schema_init_skipped`.

- [ ] **Step 5: Mark the child in the parent**

In `run_backtest_background` (`dashboard/backend/api/routers/backtests.py:1540`), directly after `env = os.environ.copy()`, add:

```python
        # The child must not repeat this process's schema DDL (db_url.
        # schema_init_skipped). Set on the copy only: the parent is not a
        # worker, and a value that leaked into os.environ would make the next
        # uvicorn reload skip DDL it actually needs.
        env[BACKTEST_WORKER_ENV] = "1"
```

and add `BACKTEST_WORKER_ENV` to the module's existing `from dashboard.backend.db_url import ...` line (create the import beside the other `dashboard.backend` imports if the module has none: `from dashboard.backend.db_url import BACKTEST_WORKER_ENV`).

- [ ] **Step 6: Strip it in conftest and document it**

In `dashboard/backend/tests/conftest.py`, after `os.environ.pop("MAX_LEGACY_ACTIVE_PER_SESSION", None)` (`:88`), add:

```python
# A shell that exported the backtest-worker flag would make every Postgres
# twin the suite constructs skip DDL and fail on its first query.
os.environ.pop("ATL_BACKTEST_WORKER", None)
```

In `CLAUDE.md`, under **Environment & credentials**, add this bullet directly after the `MAX_ACTIVE_DASHBOARD_BACKTESTS` bullet:

```markdown
- `ATL_BACKTEST_WORKER` (**never set by an operator**): `run_backtest_background` sets it to `1` in a dashboard backtest child's environment and nowhere else. Each Postgres store twin skips its `_init_schema()` when it sees the literal `1` (`db_url.schema_init_skipped`), because the parent ran that DDL at import and the child was paying a fresh Neon connection plus a batch of DDL round trips per store before reading a bar. Named for the process role, not the DDL, so nobody exports it globally and later child-only behaviour has a home. The child also publishes pre-loop **phases** to its progress file (`engine.py:PROGRESS_PHASES`: `starting`, `loading_bars`, `indicators`, `first_decision`, `running`, `saving`), which `/backtest/status` turns into the card's sentence and which double as the start-up measurement — read `phases[]` off a run's progress file for the per-phase timings. `tests/conftest.py` strips the flag.
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtest_worker_schema_skip.py dashboard/backend/tests/test_backtest_launch_phases.py dashboard/backend/tests/test_backtest_db_postgres.py dashboard/backend/tests/test_market_data_features.py -v`
Expected: all pass (`@pg_only` cases skip without `TEST_POSTGRES_URL`).

- [ ] **Step 8: Commit**

```bash
git add dashboard/backend/db_url.py dashboard/backend/database_postgres.py dashboard/backend/domain/credits/repository_postgres.py dashboard/backend/domain/analytics/repository_postgres.py dashboard/backend/domain/model_providers/repository_postgres.py dashboard/backend/api/routers/backtests.py dashboard/backend/tests/conftest.py dashboard/backend/tests/test_backtest_worker_schema_skip.py dashboard/backend/tests/test_backtest_launch_phases.py CLAUDE.md
git commit -m "feat: skip schema DDL inside the backtest worker

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Measure the start, then decide on aggregation

The phase payload is the instrument. This task records the numbers the spec asks for and applies the one gated optimisation.

**Files:**
- Create (scratch, not committed): `<scratchpad>/measure_start.sh`, `<scratchpad>/bench_aggregation.py`
- Modify (only if the gate fires): `dashboard/backend/domain/backtesting/bar_aggregation.py:112-136`
- Test (only if the gate fires): `dashboard/backend/tests/backtesting/test_bar_aggregation.py` (existing; must stay green)
- Modify: this plan's **Final verification** table

- [ ] **Step 1: Run one rule-based and one LLM backtest locally with a progress file**

Write `<scratchpad>/measure_start.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
set -a; source dashboard/.env; set +a
OUT="${1:?scratch dir}"; MODE="${2:?rule|llm}"
mkdir -p "$OUT"
LAUNCHED="$(date +%s.%N)"
if [ "$MODE" = "llm" ]; then FLAG=--use-llm; else FLAG=--no-llm; fi
python3 dashboard/scripts/backtest_hourly_agent.py \
  --start 2026-09-08 --end 2026-09-16 --session-id "measure-$MODE" \
  $FLAG --run-id "measure_$MODE" \
  --progress-file "$OUT/progress_$MODE.json" \
  --launched-at "$LAUNCHED" \
  | tee "$OUT/stdout_$MODE.log"
python3 - "$OUT/progress_$MODE.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
for ph in p["phases"]:
    print(f'{ph["name"]:15s} {ph["ended_at"] - ph["started_at"]:8.2f}s')
print(f'{p["phase"]:15s} (in progress)')
PY
```

Run: `bash <scratchpad>/measure_start.sh <scratchpad>/measure rule` then `... llm`. Use the interpreter the test suite runs under (`python3` here; `uv run python3 ...` if the environment needs it). The LLM run spends real tokens through the CLI's Anthropic client; it is the only way to time `first_decision`.

Record every phase duration from both runs in the **Final verification** table below.

- [ ] **Step 2: Time the aggregation on its own**

Write `<scratchpad>/bench_aggregation.py`:

```python
"""Is aggregate_bars_by_symbol worth vectorising? Gate: > 2.0s on the onboarding shape."""
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

Run: `python3 <scratchpad>/bench_aggregation.py` from the repo root. Record the number.

- [ ] **Step 3: Apply the gate**

If the printed time is **at most 2.0s**, skip Steps 4-6, note "aggregation gate: not fired (<time>)" in the Final verification table, and go to Step 7.

If it is **above 2.0s**, continue.

- [ ] **Step 4: Replace the per-row Series walk with an index walk plus groupby**

In `dashboard/backend/domain/backtesting/bar_aggregation.py`, replace the block from `buckets: dict[pd.Timestamp, list[pd.Series]] = {}` through `bucket_ends[bucket_start] = bucket_end` (`:112-131`) and the loop head `for bucket_start in sorted(buckets):` / `group = pd.DataFrame(buckets[bucket_start])` / `group = group.sort_index()` / `bucket_end = bucket_ends[bucket_start]` (`:133-137`) with:

```python
    # Walk the index, not the rows: iterrows builds a Series per source bar,
    # and 16k of them (7 days x 78 five-minute bars x 30 symbols) is what made
    # the child's `indicators` phase cost seconds. The per-bucket logic below
    # is byte-for-byte what it was; only how rows find their bucket changed.
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

Delete the original `records: list[dict] = []` line that followed the old loop head, so it is declared once (as above). Everything from `expected = int(` onward is unchanged.

- [ ] **Step 5: Run the aggregation tests and the benchmark**

Run: `pytest dashboard/backend/tests/backtesting/test_bar_aggregation.py dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py -v`
Expected: all pass, unchanged assertions.

Run: `python3 <scratchpad>/bench_aggregation.py`
Expected: below 2.0s. Record before/after in the table.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/backtesting/bar_aggregation.py
git commit -m "perf: walk the bar index instead of iterrows when aggregating

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 7: Record the measurement in this plan and commit**

Fill the **Final verification** table below with the two local runs (and the gate outcome), then:

```bash
git add docs/superpowers/plans/2026-09-20-backtest-visible-start.md
git commit -m "docs: record the measured backtest start phases

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
- child skips Postgres schema DDL under `ATL_BACKTEST_WORKER=1` (parent-set only)
- measured start phases in the plan's final table

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: After deploy, read the first prod run's phases**

Prod is the only place Neon DDL and connection setup are measured. After the merge deploys, launch one onboarding backtest and read its progress file's `phases[]` through `/backtest/status` while it runs (the `progress` block carries them). Add the row to the table and commit the update on a fresh branch (never on this one after it merges).

---

## Final verification

| phase | local, rule-based | local, LLM | prod, first run after deploy |
|---|---|---|---|
| starting | | | |
| loading_bars | | | |
| indicators | | | |
| first_decision | n/a | | |
| aggregation gate | fired / not fired (<time>) | | |

Full suite: `pytest dashboard/backend/tests/ -q` → _result_.
Seed DB: clean.
Cache-busters: one number per asset, five files.

## Out of scope (from the spec)

Per-bar deadlines, dropping `PROVIDER_TIMEOUT` from backtest failover, moving the analytics snapshot rebuild off the hot path, bar caching, the cadence and result-contract doors. Track B (pinned sampling) is its own plan: `docs/superpowers/plans/2026-09-20-backtest-pinned-sampling.md`.

## User-facing docs to check after this ships

`docs/source/lab/operating_modes.rst` and `docs/source/lab/key_features.rst` describe the backtest flow; neither currently claims anything about progress. Re-read once the phase card is live and file a docs follow-up if a screenshot or sentence is now stale.
