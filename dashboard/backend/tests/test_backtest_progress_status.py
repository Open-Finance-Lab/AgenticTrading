"""Progress-file freshness, so the UI can tell 'working' from 'stuck'.

The status payload already carried step/total_steps; what it could not answer
was whether those numbers were current. A run whose subprocess wedges keeps
reporting its last step forever, which reads identically to steady progress.
"""

import json
import re
import os
import time

from dashboard.backend.api.routers import backtests


def _seed(tmp_path, monkeypatch, name="backtest_progress_test.json"):
    progress_file = tmp_path / name
    progress_file.write_text(json.dumps({"step": 7, "total_steps": 240}), encoding="utf-8")
    monkeypatch.setitem(backtests.backtest_status, "progress_file", str(progress_file))
    return progress_file


def test_progress_carries_the_file_mtime(tmp_path, monkeypatch):
    progress_file = _seed(tmp_path, monkeypatch)

    payload = backtests._read_backtest_progress()

    assert payload["step"] == 7
    assert payload["total_steps"] == 240
    assert payload["progress_updated_at"] == progress_file.stat().st_mtime
    assert payload["progress_updated_at"] <= time.time() + 1


def test_progress_carries_a_server_computed_age(tmp_path, monkeypatch):
    """The age, not just the timestamp, is what the browser reads.

    Differencing the mtime against the client clock makes any machine more than
    the staleness threshold out of step indistinguishable from a wedged run: a
    fast clock pins a permanent "No progress for 47m" onto a healthy backtest, a
    slow one suppresses the warning forever. Both ends of this subtraction are
    read in one process, so it carries no skew.
    """
    _seed(tmp_path, monkeypatch)

    payload = backtests._read_backtest_progress()

    assert 0 <= payload["progress_age_seconds"] < 30


def test_age_reports_the_real_gap_for_a_wedged_run(tmp_path, monkeypatch):
    """The case the field exists for: a subprocess that stopped writing keeps
    reporting its last step forever, which reads identically to progress."""
    progress_file = _seed(tmp_path, monkeypatch, "stale.json")
    five_minutes_ago = time.time() - 300
    os.utime(progress_file, (five_minutes_ago, five_minutes_ago))

    payload = backtests._read_backtest_progress()

    assert 295 < payload["progress_age_seconds"] < 310


def test_age_is_never_negative(tmp_path, monkeypatch):
    """A clock stepping backwards between the write and this read would
    otherwise report "-3s", which reads as a bug rather than as freshness."""
    progress_file = _seed(tmp_path, monkeypatch, "future.json")
    later = time.time() + 600
    os.utime(progress_file, (later, later))

    assert backtests._read_backtest_progress()["progress_age_seconds"] == 0.0


def test_missing_progress_file_still_returns_none(tmp_path, monkeypatch):
    """Unchanged behaviour: the status payload omits `progress` entirely rather
    than shipping a half-populated object."""
    monkeypatch.setitem(
        backtests.backtest_status, "progress_file", str(tmp_path / "nope.json")
    )
    assert backtests._read_backtest_progress() is None


def test_malformed_progress_file_still_returns_none(tmp_path, monkeypatch):
    progress_file = tmp_path / "broken.json"
    progress_file.write_text("{not json", encoding="utf-8")
    monkeypatch.setitem(backtests.backtest_status, "progress_file", str(progress_file))
    assert backtests._read_backtest_progress() is None


def test_non_dict_progress_file_still_returns_none(tmp_path, monkeypatch):
    progress_file = tmp_path / "list.json"
    progress_file.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setitem(backtests.backtest_status, "progress_file", str(progress_file))
    assert backtests._read_backtest_progress() is None


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


def test_the_card_labels_every_phase_the_engine_publishes():
    """The third declaration of the phase vocabulary, pinned like the other two.

    `PROGRESS_PHASES` is pinned as an exact tuple in
    `tests/backtesting/test_engine_progress_phases.py`, and the sibling case
    above pins `PROGRESS_PHASE_MESSAGES` as a subset of it. `app.js`'s
    `BACKTEST_PHASE_LABELS` is the third declaration of that same vocabulary,
    in a third language, with no shared source -- and nothing compared it to
    the other two. The only other JS test touching that table pins prototype
    safety (`hasOwnProperty`), not membership.

    What the gap costs: add a pre-loop phase to `PROGRESS_PHASES` and forget
    app.js, and `formatBacktestPhase` returns `''` -> `advanceBacktestProgress`
    returns `null` -> the poller overwrites the stored entry with that null,
    and the card falls back to "Starting backtest..." for the whole of the new
    phase. Nothing throws, nothing logs, the suite stays green.

    Lives in THIS module rather than beside the card's other frontend cases,
    and the placement is load-bearing: `test_backtest_progress_card.py`
    carries a module-level `skipif(shutil.which("node") is None)` that would
    apply here too and silently take this guard with it on a node-less
    machine -- reproducing, in the guard itself, the exact failure shape it
    exists to catch. This reads app.js as *text*, so it needs no `node` and
    belongs with the other vocabulary checks.
    """
    from dashboard.backend.tests._frontend_source import js_const
    from dashboard.backend.domain.backtesting.engine import PROGRESS_PHASES

    declaration = js_const("BACKTEST_PHASE_LABELS")
    labelled = set(re.findall(r"^\s*(\w+):", declaration, re.MULTILINE))

    missing = set(PROGRESS_PHASES) - labelled
    assert not missing, (
        f"app.js BACKTEST_PHASE_LABELS has no entry for {sorted(missing)}. "
        "Add one -- '' if the phase should render no label, as `starting` and "
        "`running` deliberately do. An absent key is not the same as an empty "
        "one: absent blanks the card back to 'Starting backtest...' with nothing "
        "red anywhere."
    )
    extra = labelled - set(PROGRESS_PHASES)
    assert not extra, (
        f"app.js BACKTEST_PHASE_LABELS labels {sorted(extra)}, which the engine "
        "never publishes. Either the phase was removed from PROGRESS_PHASES and "
        "this entry is dead, or it is a typo that will never match."
    )
