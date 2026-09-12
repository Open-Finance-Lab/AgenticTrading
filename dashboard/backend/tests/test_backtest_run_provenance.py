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
import shutil
import subprocess
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
from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    APP_JS,
    fn_body,
    js_const,
    strip_comments,
)


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


def test_a_usage_blind_provider_is_not_mistaken_for_a_fallback():
    """``llm_calls`` only ticks for a response whose usage could be read
    (``PortfolioManager._record_llm_usage``), so a provider that reports none
    yields 0 calls on a run the model drove every step of. Reading that as
    rule-based is the same lie in the other direction."""
    assert classify_decision_provenance(
        llm_model="claude-haiku-4-5",
        llm_calls=0,
        llm_decisions=30,
        decision_steps=30,
    ) == DECISION_PROVENANCE_LLM


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


def test_a_silently_downgraded_run_is_a_fallback_not_a_choice():
    """The run this whole feature is named for, and the one shape that got it
    backwards.

    Both LLM-availability downgrades in the engine rewrite `decision_source` to
    rule_based, so the row a silent fallback writes carries the *outcome* under
    the name that is supposed to carry the *request*. Read that way, the worst
    case in the system -- asked for a model, never called one -- reported as an
    intentional rule-based run, with no badge and a note telling the user this
    is what they ordered. The request is recorded under its own key so the two
    can disagree, which is the entire point of the block.
    """
    block = run_decision_provenance(
        _row(
            llm_model="rule-based",
            llm_calls=0,
            llm_decisions=0,
            metadata={
                "decision_source": "rule_based",        # what it did
                "requested_decision_source": "llm",     # what it was asked
                "decision_steps": 30,
            },
        )
    )

    assert block["decision_source"] == "llm"
    assert block["decision_provenance"] == DECISION_PROVENANCE_RULE_BASED
    assert block["decision_fallback"] is True
    assert "fell back" in block["decision_note"]
    # And the ratio is no longer suppressed as "nothing to report about a
    # rule-based run": zero of thirty is the finding.
    assert block["decision_badge"] == "0 of 30 steps model-driven"


def test_an_intentional_rule_based_run_records_the_same_source_twice():
    """The other half of the same key. A caller that ordered rule-based logic
    got what it ordered, and the *only* thing separating that row from the
    fallback above is these two fields agreeing."""
    block = run_decision_provenance(
        _row(
            llm_model="rule-based",
            llm_calls=0,
            llm_decisions=0,
            metadata={
                "decision_source": "rule_based",
                "requested_decision_source": "rule_based",
                "decision_steps": 30,
            },
        )
    )

    assert block["decision_source"] == "rule_based"
    assert block["decision_fallback"] is False
    assert block["decision_badge"] is None
    assert "fell back" not in block["decision_note"]


def test_a_row_written_before_the_request_key_reads_the_source_it_has():
    """Rows already in the database carry only `decision_source`, and for a run
    that was never downgraded it *is* the request. Withholding the verdict from
    them would trade one wrong answer for no answer on every historical row;
    the rows this cannot speak for are the downgraded ones, which are also the
    rows that never recorded the question."""
    block = run_decision_provenance(
        _row(llm_decisions=0, metadata={"decision_source": "llm", "decision_steps": 30})
    )

    assert block["decision_source"] == "llm"
    assert block["decision_fallback"] is True


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
# The N-of-M badge, and why its bar is not the H6 bar
# ===========================================================================

def test_a_publishable_run_still_badges_below_one_hundred_percent():
    """The acceptance criterion, and the whole reason the badge is driven by
    the counts rather than by the verdict.

    159/161 clears MIN_LLM_DECISION_COVERAGE, so the leaderboard would publish
    it and the verdict stays ``llm``. The user still paid for two steps the
    model did not answer, and "may this publish" is not the same question as
    "should the user be told something degraded".
    """
    block = run_decision_provenance(
        _row(
            llm_calls=161,
            llm_decisions=159,
            metadata={"decision_source": "llm", "decision_steps": 161},
        )
    )

    assert block["decision_provenance"] == DECISION_PROVENANCE_LLM
    assert block["decision_fallback"] is False
    assert block["decision_badge"] == "159 of 161 steps model-driven"
    assert block["decision_note"] is not None


def test_full_coverage_shows_no_badge():
    assert run_decision_provenance(_row())["decision_badge"] is None


def test_the_badge_fires_on_a_single_held_step():
    """One step short of 100% is still short of 100%. The bar is equality, so
    there is no second threshold here to drift away from the first."""
    block = run_decision_provenance(
        _row(
            llm_calls=30,
            llm_decisions=29,
            metadata={"decision_source": "llm", "decision_steps": 30},
        )
    )
    assert block["decision_badge"] == "29 of 30 steps model-driven"


def test_a_total_fallback_badges_zero_of_n():
    block = run_decision_provenance(
        _row(llm_decisions=0, metadata={"decision_source": "llm", "decision_steps": 30})
    )
    assert block["decision_badge"] == "0 of 30 steps model-driven"


def test_an_intentional_rule_based_run_shows_no_coverage_ratio():
    """There is no model coverage to report a ratio about -- the caller asked
    for none. The note already labels the run."""
    block = run_decision_provenance(
        _row(
            llm_model="rule-based",
            llm_calls=0,
            llm_decisions=0,
            metadata={"decision_source": "rule_based", "decision_steps": 30},
        )
    )
    assert block["decision_badge"] is None
    assert block["decision_note"] == "Rule-based strategy — no model decisions."


def test_a_row_without_the_witness_shows_no_badge():
    """A ratio invented out of a defaulted 0 would read as a finding."""
    block = run_decision_provenance(
        _row(llm_decisions=0, metadata={"data_source": "alpaca"})
    )
    assert block["decision_badge"] is None


def test_the_badge_bar_and_the_publish_bar_are_different_numbers():
    """Pins the asymmetry itself, not one example of it.

    Sweeping the whole coverage range: every run short of 100% badges, while
    only runs under MIN_LLM_DECISION_COVERAGE lose the ``llm`` verdict. A
    change that collapsed the two would break this and nothing else.
    """
    steps = 100
    badged = set()
    verdicts = {}
    for decisions in range(steps + 1):
        block = run_decision_provenance(
            _row(
                llm_calls=steps,
                llm_decisions=decisions,
                metadata={"decision_source": "llm", "decision_steps": steps},
            )
        )
        if block["decision_badge"]:
            badged.add(decisions)
        verdicts[decisions] = block["decision_provenance"]

    assert badged == set(range(steps))            # everything below 100%
    assert steps not in badged                    # and nothing at it
    publishable = {n for n, v in verdicts.items() if v == DECISION_PROVENANCE_LLM}
    assert min(publishable) == 95                 # the H6 bar, not 100
    assert publishable < badged | {steps}         # strictly the looser test


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


def test_status_carries_the_badge_for_a_publishable_but_degraded_run(status_mirror):
    """Coverage over the H6 bar and under 100%: no fallback warning, but the
    user is still told two steps were not model-driven."""
    session_id = str(uuid.uuid4())
    body = _finished_status(
        status_mirror,
        session_id,
        f"agent_{uuid.uuid4().hex[:8]}",
        llm_model="claude-haiku-4-5",
        llm_calls=161,
        llm_decisions=159,
        metadata={"decision_source": "llm", "decision_steps": 161},
    )

    assert body["decision_provenance"] == DECISION_PROVENANCE_LLM
    assert body["decision_fallback"] is False
    assert body["decision_badge"] == "159 of 161 steps model-driven"
    assert body["llm_decisions"] == 159 and body["decision_steps"] == 161


# ===========================================================================
# The badge survives the poll: it rides the run record too
# ===========================================================================

def test_the_run_record_carries_the_verdict_and_the_counts():
    """``/backtest/status`` is transient -- the panel it feeds is gone seconds
    after the run ends and never comes back on a reload. The results view reads
    the run record, so the badge has to live there or it is not really on the
    result at all."""
    session_id = str(uuid.uuid4())
    run_id = f"agent_{uuid.uuid4().hex[:8]}"
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
        llm_model="claude-haiku-4-5",
        llm_calls=30,
        llm_decisions=10,
        metadata={"decision_source": "llm", "decision_steps": 30},
    )

    resp = TestClient(app).get(
        "/api/backtest/runs", headers={"X-Session-Id": session_id}
    )
    assert resp.status_code == 200, resp.text
    record = next(r for r in resp.json() if r["run_id"] == run_id)

    assert record["decision_source"] == "llm"        # asked for
    assert record["decision_provenance"] == DECISION_PROVENANCE_PARTIAL  # got
    assert record["decision_badge"] == "10 of 30 steps model-driven"
    assert record["decision_fallback"] is True
    assert record["llm_calls"] == 30
    assert record["llm_decisions"] == 10
    assert record["decision_steps"] == 30


def test_the_run_record_keeps_requested_and_observed_apart():
    """`decision_source` on this model already meant "what was requested"
    before this feature. The provenance block is applied after the metadata
    copy and must not overwrite it with the observed verdict."""
    session_id = str(uuid.uuid4())
    run_id = f"agent_{uuid.uuid4().hex[:8]}"
    db.insert_run(
        run_id=run_id,
        session_id=session_id,
        agent_name="Agent",
        mode="backtest",
        start_date="2026-03-01",
        end_date="2026-04-01",
        initial_equity=100000.0,
        llm_model="claude-haiku-4-5",
        llm_calls=30,
        llm_decisions=0,
        metadata={"decision_source": "llm", "decision_steps": 30},
    )

    resp = TestClient(app).get(
        "/api/backtest/runs", headers={"X-Session-Id": session_id}
    )
    record = next(r for r in resp.json() if r["run_id"] == run_id)

    assert record["decision_source"] == "llm"
    assert record["decision_provenance"] == DECISION_PROVENANCE_RULE_BASED


# ===========================================================================
# /app renders it
# ===========================================================================

def test_the_success_branch_renders_the_verdict():
    """A finished run's only "what produced this" surface is this panel."""
    body = strip_comments(fn_body("function ensureBacktestPolling"))
    assert "formatDecisionProvenance(status)" in body


def _settle(status):
    """Run the completion branch's panel decision under node, as it is actually
    reached: after loadData() has already hidden the panel.

    Executed rather than grepped because the first cut of this guard asserted
    the *shape* of the branch -- a `if (!backtestFellBackFromTheModel(status))`
    wrapping the dismissal timer -- and that shape was satisfied by code which
    could not work. `await loadData()` runs first and hides this panel on its
    way past, so withholding the 2.5s timeout withheld a hide from an
    already-hidden panel, and the fallback sentence the branch existed to keep
    on screen was never on screen. A source-shape assertion cannot see that;
    only running the thing against a panel loadData() has just hidden can.

    Returns the panel state a user would be looking at once the dismissal
    timer, if one was scheduled, has fired.
    """
    harness = """
        let panelVisible = false;   // loadData() has just hidden it
        let panelMessage = 'Completed in 1:01.';
        let panelFinished = false;
        const timers = [];
        function showBacktestRunProgress(show, { isFinished = false } = {}) {
            panelVisible = !!show;
            panelFinished = !!isFinished;
        }
        function updateBacktestRunProgress({ message }) {
            if (message) panelMessage = message;
        }
        globalThis.setTimeout = (fn) => { timers.push(fn); };
    """
    script = "\n".join(
        [
            js_const("BACKTEST_COMPLETION_DISMISS_MS"),
            fn_body("function backtestFellBackFromTheModel"),
            fn_body("function settleFinishedBacktestPanel"),
            harness,
            # Parenthesised: an implicit concatenation between two list items
            # is indistinguishable from a missing comma, which here would
            # silently drop a statement from the script instead of failing.
            (
                f"settleFinishedBacktestPanel({json.dumps(status)}, 61, "
                f"{json.dumps(_COMPLETION_MESSAGE)});"
            ),
            "const scheduled = timers.length;",
            "timers.forEach((fn) => fn());",   # the dismissal timeout elapses
            (
                "console.log(JSON.stringify("
                "{ panelVisible, panelMessage, panelFinished, scheduled }));"
            ),
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


_COMPLETION_MESSAGE = (
    "Completed in 1:01. No step used the model \u2014 every decision fell back "
    "to rule-based logic."
)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_fallback_run_still_has_its_panel_up_after_loaddata_hid_it():
    """The load-bearing case: loadData() has already hidden the panel, so the
    branch has to put it back rather than decline to take it down."""
    settled = _settle({"success": True, "decision_fallback": True})

    assert settled["panelVisible"] is True
    assert settled["panelMessage"] == _COMPLETION_MESSAGE
    assert settled["scheduled"] == 0   # and nothing is queued to take it away
    # A panel that outlives its run must stop describing one in flight: the
    # markup's defaults are "Backtest in progress", a progress track and a
    # hint about the 60-minute limit, all of which were only ever true while
    # something was still happening.
    assert settled["panelFinished"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_clean_run_dismisses_itself():
    """The other half. A run the model drove says only that it finished, which
    the results loadData() has just painted say better."""
    settled = _settle({"success": True, "decision_fallback": False})

    assert settled["panelVisible"] is False
    assert settled["scheduled"] == 1


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_run_that_predates_the_provenance_fields_dismisses_itself():
    """A status payload from a backend that does not send `decision_fallback`
    is not a fallback. Treating "no answer" as "yes" would pin the panel open
    on every run during a rolling deploy."""
    settled = _settle({"success": True})

    assert settled["panelVisible"] is False
    assert settled["scheduled"] == 1


def test_a_finished_panel_stops_advertising_a_run_in_flight():
    """`isFinished` has to reach the three elements the markup defaults for a
    running backtest -- the title, the progress track and the wait hint. The
    elapsed clock is deliberately not one of them: on a finished run it is the
    duration."""
    body = strip_comments(fn_body("function showBacktestRunProgress"))
    assert "'Backtest complete'" in body
    # The three in-flight elements hide off one `terminal` flag, which a
    # cancelled run (issue #273) joined after this guard was written. Pinning
    # the flag's definition plus its two readers keeps the contract this test
    # exists for -- isFinished still reaches the track and the hint -- without
    # pinning one expression's spelling, which the third state had to rewrite.
    assert "const terminal = !!isError || !!isCancelled || !!isFinished;" in body
    assert "track.hidden = terminal" in body
    assert "hint.hidden = terminal" in body
    assert "elapsed.hidden = !!isError" in body


def test_the_panel_is_settled_after_loaddata_not_before():
    """The ordering is the whole bug. Settling first would re-show a panel that
    loadData() then hides, which is the same failure facing the other way."""
    body = strip_comments(fn_body("function ensureBacktestPolling"))
    assert body.index("await loadData();") < body.index("settleFinishedBacktestPanel(")


def test_the_results_view_renders_the_coverage_badge():
    """The persistent half: a cell on the run's own config panel, so the
    N-of-M indicator is still there after the completion panel is gone."""
    assert 'id="backtestConfigDecisionCoverageRow"' in APP_HTML
    assert 'id="backtestConfigDecisionCoverage"' in APP_HTML
    body = strip_comments(fn_body("function renderBacktestRunConfig"))
    assert "run?.decision_badge" in body
    assert "coverageRow.hidden = !coverageBadge" in body


def test_the_browser_owns_no_coverage_threshold():
    """The badge's bar is equality with the step count and the verdict's bar is
    the server's constant. Neither is a number the frontend may hold: a copy of
    0.95 here is how the dashboard and the leaderboard start disagreeing about
    whether the model drove a run."""
    body = strip_comments(fn_body("function renderBacktestRunConfig"))
    assert "0.95" not in body
    assert "MIN_LLM_DECISION_COVERAGE" not in strip_comments(APP_JS)


def test_the_browser_does_not_compose_its_own_provenance_copy():
    """One message, one owner. A template here would drift from the server's
    wording the first time either side is edited."""
    body = strip_comments(fn_body("function formatDecisionProvenance"))
    assert "status?.decision_note" in body
    # The wording lives on the server; a copy of it here is a second owner.
    shipped = strip_comments(APP_JS)
    for phrase in ("fell back to rule-based logic", "no model decisions"):
        assert phrase not in shipped
