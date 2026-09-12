"""A finished backtest must say what actually produced its numbers (#169).

A dashboard backtest could fall back to rule-based logic on every single step
and still answer ``/backtest/status`` with "Backtest completed successfully".
Two layers had to be closed, and the tests split the same way:

* the run row was already honest (``llm_model`` stays ``"rule-based"`` when no
  call ever succeeded) and nothing read it at the HTTP boundary;
* ``llm_decisions`` -- steps the model actually drove -- was never persisted at
  all, so the *worst* case was invisible: every step made a billed call, every
  response was unusable, and the row reads ``llm_calls = 30``,
  ``llm_model = "claude-..."``, which looks perfectly clean.

That second case is the one this file exists for. A coverage check keyed on
``llm_calls`` waves it through, and dashboard backtests are subprocesses, so
the in-process H6 guard that would have caught it never sees the run.
"""

import copy
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.database import db
import dashboard.backend.api.routers.backtests as bt
from dashboard.backend.domain.backtesting import provenance
from dashboard.backend.domain.backtesting.provenance import (
    DECISION_PROVENANCE_LLM,
    DECISION_PROVENANCE_PARTIAL,
    DECISION_PROVENANCE_RULE_BASED,
    DECISION_PROVENANCE_UNKNOWN,
    classify_decision_provenance,
    run_decision_provenance,
)
from dashboard.backend.domain.leaderboard import service as leaderboard_service
from dashboard.backend.tests._frontend_source import APP_JS, fn_body, strip_comments


# ===========================================================================
# The verdict helper
# ===========================================================================

def test_every_step_billed_and_none_usable_is_rule_based():
    """The case the two counters exist for.

    ``llm_calls == decision_steps`` with ``llm_decisions == 0``: every step
    reached the model, every response was billed, and every one of them was
    unusable, so every step traded rule-based. Coverage keyed on ``llm_calls``
    reports this run as 100% model-driven -- which is exactly how a curve the
    model never drove gets published under its name.
    """
    assert classify_decision_provenance(
        llm_model="claude-haiku-4-5",
        llm_calls=30,
        llm_decisions=0,
        decision_steps=30,
    ) == DECISION_PROVENANCE_RULE_BASED


def test_a_clean_llm_run_classifies_as_llm():
    assert classify_decision_provenance(
        llm_model="claude-haiku-4-5",
        llm_calls=30,
        llm_decisions=30,
        decision_steps=30,
    ) == DECISION_PROVENANCE_LLM


def test_a_run_just_over_the_threshold_is_still_llm():
    """A genuine run absorbs a transient blip; that is what the margin is for."""
    assert classify_decision_provenance(
        llm_model="claude-haiku-4-5",
        llm_calls=161,
        llm_decisions=159,
        decision_steps=161,
    ) == DECISION_PROVENANCE_LLM


def test_a_mostly_rule_based_run_classifies_as_partial():
    assert classify_decision_provenance(
        llm_model="claude-haiku-4-5",
        llm_calls=30,
        llm_decisions=10,
        decision_steps=30,
    ) == DECISION_PROVENANCE_PARTIAL


def test_the_honest_rule_based_label_is_taken_at_its_word():
    """Layer A: the engine already persists this label; nothing read it."""
    assert classify_decision_provenance(
        llm_model="rule-based", llm_calls=0, llm_decisions=0, decision_steps=30
    ) == DECISION_PROVENANCE_RULE_BASED


def test_a_model_name_beside_zero_calls_is_rule_based():
    """The shape H6 refuses to publish: a rule-based curve wearing a name."""
    assert classify_decision_provenance(
        llm_model="claude-haiku-4-5",
        llm_calls=0,
        llm_decisions=0,
        decision_steps=30,
    ) == DECISION_PROVENANCE_RULE_BASED


def test_an_unrecorded_counter_is_unknown_not_an_accusation():
    """A row written before the column existed reads back as 0 like any other.

    Reporting those as rule-based would be the same lie in the other direction
    -- and every historical run is one of them.
    """
    assert classify_decision_provenance(
        llm_model="claude-haiku-4-5",
        llm_calls=30,
        llm_decisions=None,
        decision_steps=None,
    ) == DECISION_PROVENANCE_UNKNOWN


def test_a_missing_denominator_still_catches_the_total_fallback():
    """Zero usable decisions is rule-based however many steps there were."""
    assert classify_decision_provenance(
        llm_model="claude-haiku-4-5",
        llm_calls=30,
        llm_decisions=0,
        decision_steps=None,
    ) == DECISION_PROVENANCE_RULE_BASED


def test_the_threshold_has_exactly_one_owner():
    """The dashboard and the leaderboard must not be able to disagree about
    "the model drove it" -- restating 0.95 here is how they start to."""
    assert (
        provenance.MIN_LLM_DECISION_COVERAGE
        is leaderboard_service.MIN_LLM_DECISION_COVERAGE
    )
    source = (
        provenance.__file__
    )
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "0.95" not in text


def test_the_verdict_matches_the_h6_guard_on_the_same_numbers():
    """Same numbers, same answer: whatever the leaderboard refuses to publish
    as a fallback must not read as model-driven on the dashboard."""

    class _LlmEntry:
        used_llm = True

    for decisions, expected_refusal in ((0, True), (10, True), (29, False)):
        refused = False
        try:
            leaderboard_service._reject_if_llm_fallback(
                "entry",
                _LlmEntry(),
                30,
                llm_decisions=decisions,
                decision_steps=30,
            )
        except leaderboard_service.LeaderboardFallbackError:
            refused = True
        assert refused is expected_refusal
        verdict = classify_decision_provenance(
            llm_model="claude-haiku-4-5",
            llm_calls=30,
            llm_decisions=decisions,
            decision_steps=30,
        )
        assert (verdict != DECISION_PROVENANCE_LLM) is expected_refusal


# ===========================================================================
# Reading one persisted row
# ===========================================================================

def _row(**overrides):
    row = {
        "run_id": "agent_1",
        "llm_model": "claude-haiku-4-5",
        "llm_calls": 30,
        "llm_decisions": 30,
        "metadata": {"decision_source": "llm", "decision_steps": 30},
    }
    row.update(overrides)
    return row


def test_a_row_reports_what_was_asked_for_and_what_happened():
    block = run_decision_provenance(
        _row(llm_decisions=0, metadata={"decision_source": "llm", "decision_steps": 30})
    )

    assert block["decision_source"] == "llm"          # what the caller asked for
    assert block["decision_provenance"] == DECISION_PROVENANCE_RULE_BASED
    assert block["decision_fallback"] is True         # the two disagreeing
    assert block["llm_calls"] == 30
    assert block["llm_decisions"] == 0
    assert block["decision_steps"] == 30
    assert "fell back" in block["decision_note"]


def test_an_intentional_rule_based_run_is_labelled_but_not_a_fallback():
    block = run_decision_provenance(
        _row(
            llm_model="rule-based",
            llm_calls=0,
            llm_decisions=0,
            metadata={"decision_source": "rule_based", "decision_steps": 30},
        )
    )

    assert block["decision_provenance"] == DECISION_PROVENANCE_RULE_BASED
    assert block["decision_fallback"] is False
    assert "fell back" not in block["decision_note"]


def test_a_clean_run_gets_no_note():
    block = run_decision_provenance(_row())

    assert block["decision_provenance"] == DECISION_PROVENANCE_LLM
    assert block["decision_note"] is None
    assert block["decision_fallback"] is False


def test_a_row_without_the_witness_reports_unknown():
    """``metadata.decision_steps`` is what says this row records llm_decisions.

    The column was added with DEFAULT 0, so a pre-migration row comes back with
    llm_decisions == 0 exactly like a total fallback does. Without the witness
    every run that predates this feature would be accused of one.
    """
    block = run_decision_provenance(
        _row(llm_decisions=0, metadata={"data_source": "alpaca"})
    )

    assert block["decision_provenance"] == DECISION_PROVENANCE_UNKNOWN
    assert block["llm_decisions"] is None
    assert block["decision_note"] is None


def test_no_row_yields_no_block():
    assert run_decision_provenance(None) is None


# ===========================================================================
# The column round-trips
# ===========================================================================

def test_llm_decisions_round_trips_through_insert_run():
    """Distinct values on purpose: a write that dropped the new argument and
    echoed llm_calls into it would pass with the two set equal."""
    run_id = f"prov_{uuid.uuid4().hex[:8]}"
    db.insert_run(
        run_id=run_id,
        session_id="provenance-session",
        agent_name="Agent",
        mode="backtest",
        start_date="2026-03-01",
        end_date="2026-04-01",
        initial_equity=100000.0,
        final_equity=101000.0,
        total_return=0.01,
        num_trades=3,
        llm_model="claude-haiku-4-5",
        llm_calls=30,
        llm_decisions=7,
        metadata={"decision_source": "llm", "decision_steps": 30},
    )

    stored = db.get_run(run_id)
    assert stored["llm_calls"] == 30
    assert stored["llm_decisions"] == 7
    assert stored["metadata"]["decision_steps"] == 30
    assert (
        run_decision_provenance(stored)["decision_provenance"]
        == DECISION_PROVENANCE_PARTIAL
    )


def test_the_postgres_backfill_carries_llm_decisions():
    """The SQLite -> Postgres copy must not drop the counter.

    0 is also what a row with no counter reads as, so a backfill that omitted
    it would republish every migrated LLM run as a total fallback.
    """
    from dashboard.scripts import backfill_runs_to_postgres as backfill

    captured: dict = {}

    class _Target:
        def insert_run(self, **kwargs):
            captured.update(kwargs)

    backfill._insert_one_run(_Target(), {
        "run_id": "r",
        "session_id": "s",
        "agent_name": "Agent",
        "mode": "backtest",
        "start_date": "2026-03-01",
        "end_date": "2026-04-01",
        "initial_equity": 100000.0,
        "llm_calls": 30,
        "llm_decisions": 29,
    })

    assert captured["llm_calls"] == 30
    assert captured["llm_decisions"] == 29


def test_insert_run_defaults_llm_decisions_for_callers_that_have_none():
    """Baselines and paper trading call insert_run without it and must not
    start failing; their rows are rule-based by construction anyway."""
    run_id = f"prov_{uuid.uuid4().hex[:8]}"
    db.insert_run(
        run_id=run_id,
        session_id="provenance-session",
        agent_name="buy-and-hold",
        mode="backtest",
        start_date="2026-03-01",
        end_date="2026-04-01",
        initial_equity=100000.0,
    )

    assert db.get_run(run_id)["llm_decisions"] == 0


# ===========================================================================
# GET /backtest/status
# ===========================================================================

@pytest.fixture
def status_mirror():
    """Save/restore the legacy ``backtest_status`` mirror this route falls back
    to when no slot is registered (the shape the existing router tests use)."""
    original = copy.deepcopy(bt.backtest_status)
    yield bt.backtest_status
    bt.backtest_status.clear()
    bt.backtest_status.update(original)


def _finished_status(status_mirror, session_id, run_id, **run_kwargs):
    db.insert_run(
        run_id=run_id,
        session_id=session_id,
        agent_name="Agent",
        mode="backtest",
        start_date="2026-03-01",
        end_date="2026-04-01",
        initial_equity=100000.0,
        final_equity=101000.0,
        total_return=0.01,
        **run_kwargs,
    )
    status_mirror.update({
        "running": False,
        "error": None,
        "runs_count": 1,
        "started_at": None,
        "progress_file": None,
        "live_run_id": run_id,
    })
    resp = TestClient(app).get(
        "/backtest/status", headers={"X-Session-Id": session_id}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_status_cannot_report_a_silent_fallback_as_a_clean_success(status_mirror):
    """The bug, end to end: every step billed, none usable, reported clean."""
    session_id = str(uuid.uuid4())
    body = _finished_status(
        status_mirror,
        session_id,
        f"agent_{uuid.uuid4().hex[:8]}",
        llm_model="claude-haiku-4-5",
        llm_calls=30,
        llm_decisions=0,
        metadata={"decision_source": "llm", "decision_steps": 30},
    )

    assert body["success"] is True
    assert body["decision_provenance"] == DECISION_PROVENANCE_RULE_BASED
    assert body["decision_fallback"] is True
    assert body["llm_calls"] == 30
    assert body["llm_decisions"] == 0
    assert body["decision_steps"] == 30
    assert body["message"] != "Backtest completed successfully"
    assert "fell back" in body["message"]


def test_status_keeps_every_field_it_already_had(status_mirror):
    """Additive only: app.js and the legacy surface both poll this route."""
    session_id = str(uuid.uuid4())
    run_id = f"agent_{uuid.uuid4().hex[:8]}"
    body = _finished_status(
        status_mirror,
        session_id,
        run_id,
        llm_model="claude-haiku-4-5",
        llm_calls=30,
        llm_decisions=30,
        metadata={"decision_source": "llm", "decision_steps": 30},
    )

    assert body["running"] is False
    assert body["success"] is True
    assert body["runs_count"] == 1
    assert body["session_id"] == session_id
    assert body["live_run_id"] == run_id
    assert body["message"] == "Backtest completed successfully"
    assert body["decision_provenance"] == DECISION_PROVENANCE_LLM
    assert body["decision_note"] is None


def test_status_reports_a_partial_fallback(status_mirror):
    session_id = str(uuid.uuid4())
    body = _finished_status(
        status_mirror,
        session_id,
        f"agent_{uuid.uuid4().hex[:8]}",
        llm_model="claude-haiku-4-5",
        llm_calls=30,
        llm_decisions=10,
        metadata={"decision_source": "llm", "decision_steps": 30},
    )

    assert body["decision_provenance"] == DECISION_PROVENANCE_PARTIAL
    assert body["decision_fallback"] is True
    assert "10 of 30" in body["message"]


def test_status_does_not_accuse_a_run_that_predates_the_counter(status_mirror):
    session_id = str(uuid.uuid4())
    body = _finished_status(
        status_mirror,
        session_id,
        f"agent_{uuid.uuid4().hex[:8]}",
        llm_model="claude-haiku-4-5",
        llm_calls=30,
        metadata={"data_source": "alpaca"},
    )

    assert body["decision_provenance"] == DECISION_PROVENANCE_UNKNOWN
    assert body["message"] == "Backtest completed successfully"


def test_status_never_answers_with_another_runs_provenance(status_mirror):
    """A baseline row is inserted *after* the agent run and carries no witness.

    ``get_runs_by_session`` is newest-first, so a fallback that simply took the
    first row would report the DJIA baseline's counters as the agent's.
    """
    session_id = str(uuid.uuid4())
    agent_run = f"agent_{uuid.uuid4().hex[:8]}"
    body = _finished_status(
        status_mirror,
        session_id,
        agent_run,
        llm_model="claude-haiku-4-5",
        llm_calls=30,
        llm_decisions=0,
        metadata={"decision_source": "llm", "decision_steps": 30},
    )
    assert body["decision_provenance"] == DECISION_PROVENANCE_RULE_BASED

    db.insert_run(
        run_id=f"djia_index_{uuid.uuid4().hex[:8]}",
        session_id=session_id,
        agent_name="DJIA",
        mode="backtest",
        start_date="2026-03-01",
        end_date="2026-04-01",
        initial_equity=100000.0,
        metadata={"data_source": "alpaca"},
    )
    again = TestClient(app).get(
        "/backtest/status", headers={"X-Session-Id": session_id}
    )
    assert again.json()["decision_provenance"] == DECISION_PROVENANCE_RULE_BASED


# ===========================================================================
# /app renders it
# ===========================================================================

def test_the_success_branch_renders_the_verdict():
    """A finished run's only "what produced this" surface is this panel."""
    body = strip_comments(fn_body("function ensureBacktestPolling"))
    assert "formatDecisionProvenance(status)" in body


def test_a_fallback_run_does_not_auto_hide_the_panel():
    """2.5s is not long enough to read a sentence nobody expected, and the
    numbers loadData() has just painted stay on screen afterwards."""
    body = strip_comments(fn_body("function ensureBacktestPolling"))
    assert "if (!backtestFellBackFromTheModel(status)) {" in body
    # The auto-hide must be *inside* that guard, not beside it.
    guarded = body[body.index("backtestFellBackFromTheModel(status)"):]
    assert guarded.index("showBacktestRunProgress(false), 2500") < guarded.index("} else {")


def test_the_browser_does_not_compose_its_own_provenance_copy():
    """One message, one owner. A template here would drift from the server's
    wording the first time either side is edited."""
    body = strip_comments(fn_body("function formatDecisionProvenance"))
    assert "status?.decision_note" in body
    # The wording lives on the server; a copy of it here is a second owner.
    shipped = strip_comments(APP_JS)
    for phrase in ("fell back to rule-based logic", "no model decisions"):
        assert phrase not in shipped
