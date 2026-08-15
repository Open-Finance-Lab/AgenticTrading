"""Concurrent dashboard backtest slots — the parts a green suite missed.

The multi-slot runner replaced a process-wide single-flight flag with a ledger
of per-owner slots. Every case here failed on the branch that introduced it,
while its own tests passed: the defects all live in the seams between the new
ledger and the legacy ``backtest_status`` mirror it did not remove, and nothing
exercised a *second* concurrent run or a *second* caller.

The through-line: under single flight, "the run" was unambiguous, so a helper
that answered with the most recent run was always right. Every fallback written
under that assumption became a wrong answer the moment two runs could coexist,
and every one of them returns HTTP 200.
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
import dashboard.backend.api.routers.backtests as bt


def _sess() -> dict:
    return {"X-Session-Id": str(uuid.uuid4())}


@pytest.fixture(autouse=True)
def _clean_slots():
    bt._reset_slots_for_tests()
    yield
    bt._reset_slots_for_tests()


# ===========================================================================
# GET /backtest/status?live_run_id= — ownership
# ===========================================================================

def test_status_by_run_id_does_not_serve_another_callers_run():
    """The lookup is an authorization boundary, not a convenience.

    A session id in this codebase is an access grant (see ``_owner_context``:
    with no X-Browser-Id it doubles as the ownership credential), so answering
    a stranger's poll with the run's ``session_id`` hands over the credential
    itself — not merely the fact that a backtest is running.
    """
    victim = _sess()
    attacker = _sess()

    assert bt._try_acquire_backtest_slot(
        live_run_id="victim-run",
        session_id=victim["X-Session-Id"],
        user_id=None,
    ) is None

    client = TestClient(app)
    resp = client.get(
        "/backtest/status", params={"live_run_id": "victim-run"}, headers=attacker
    )
    assert resp.status_code == 404, resp.text
    assert victim["X-Session-Id"] not in resp.text

    # The owner still gets their own run.
    mine = client.get(
        "/backtest/status", params={"live_run_id": "victim-run"}, headers=victim
    )
    assert mine.status_code == 200
    assert mine.json()["running"] is True


def test_unknown_run_id_is_404_not_a_sibling_run():
    """An id that resolves to nothing must not fall through to the session scan.

    The caller passed an id precisely to disambiguate between their own
    concurrent runs. Answering with whichever run the scan finds first turns
    "how is run B doing?" into run A's progress, silently, with a 200.
    """
    owner = _sess()
    assert bt._try_acquire_backtest_slot(
        live_run_id="run-a", session_id=owner["X-Session-Id"], user_id=None
    ) is None

    resp = TestClient(app).get(
        "/backtest/status", params={"live_run_id": "run-b-never-existed"}, headers=owner
    )
    assert resp.status_code == 404, resp.text
    assert "run-a" not in resp.text


def test_unknown_and_forbidden_ids_are_indistinguishable():
    """Same status, same body — or the route becomes a run-id oracle."""
    victim = _sess()
    attacker = _sess()
    bt._try_acquire_backtest_slot(
        live_run_id="real-run", session_id=victim["X-Session-Id"], user_id=None
    )

    client = TestClient(app)
    forbidden = client.get(
        "/backtest/status", params={"live_run_id": "real-run"}, headers=attacker
    )
    unknown = client.get(
        "/backtest/status", params={"live_run_id": "no-such-run"}, headers=attacker
    )
    assert forbidden.status_code == unknown.status_code == 404
    assert forbidden.json() == unknown.json()


# ===========================================================================
# Progress files
# ===========================================================================

def test_a_run_with_no_progress_yet_does_not_report_a_siblings(tmp_path):
    """``progress_file or <the global mirror>`` was a sibling-run leak.

    Between accept and the subprocess's first write, a slot's progress_file is
    None. Falling back to the mirror made a backtest started one second ago
    open at whatever percentage another run had reached.
    """
    # Two visitors running the same built-in agent: one shared results session,
    # two independent slot budgets. The concurrency this PR exists to allow.
    owner = _sess()
    other_progress = tmp_path / "other.json"
    other_progress.write_text(
        json.dumps({"step": 200, "total_steps": 240}), encoding="utf-8"
    )

    # An older run that has been writing progress for a while.
    assert bt._try_acquire_backtest_slot(
        live_run_id="older",
        session_id=owner["X-Session-Id"],
        owner_session="visitor-1",
        user_id=None,
    ) is None
    bt._update_slot("older", progress_file=str(other_progress))

    # A run accepted a moment ago: no progress file yet.
    assert bt._try_acquire_backtest_slot(
        live_run_id="fresh",
        session_id=owner["X-Session-Id"],
        owner_session="visitor-2",
        user_id=None,
    ) is None

    resp = TestClient(app).get(
        "/backtest/status", params={"live_run_id": "fresh"}, headers=owner
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["running"] is True
    assert "progress" not in body, body
    assert "200/240" not in body["message"]


# ===========================================================================
# The server-wide ceiling
# ===========================================================================

@pytest.mark.parametrize("raw", ["", "   ", "twenty", "5.5", "-1", None])
def test_bad_ceiling_values_fall_back_instead_of_killing_boot(monkeypatch, raw):
    """A typo in a Render field must not be an import-time crash.

    The bare ``int(os.getenv(...))`` raised ValueError at module import, which
    on this deployment means the app does not boot at all — and a negative
    value parsed fine and refused every backtest forever.
    """
    if raw is None:
        monkeypatch.delenv("MAX_ACTIVE_DASHBOARD_BACKTESTS", raising=False)
    else:
        monkeypatch.setenv("MAX_ACTIVE_DASHBOARD_BACKTESTS", raw)
    assert (
        bt._max_active_dashboard_backtests()
        == bt._DEFAULT_MAX_ACTIVE_DASHBOARD_BACKTESTS
    )


def test_a_suspended_account_is_told_it_is_suspended(monkeypatch):
    """Quota 0 is the admin console's suspension value, not "one at a time".

    ``max_for_owner <= 1`` folded the two together and told a suspended user to
    "wait for it to complete" — waiting on a run they do not have, forever.
    """
    import dashboard.backend.users as users_module

    class _Suspended:
        @staticmethod
        def get_entitlements(_user_id):
            return {"max_concurrent_backtests": 0}

    monkeypatch.setattr(users_module, "user_store", _Suspended())

    refusal = bt._try_acquire_backtest_slot(
        live_run_id="suspended-run", session_id=str(uuid.uuid4()), user_id=7
    )
    assert refusal is not None
    assert "disabled" in refusal
    assert "already running" not in refusal
    # And nothing was registered for an account that may not run.
    assert bt.count_active_dashboard_backtests() == 0


def test_client_side_guard_refuses_a_zero_limit(monkeypatch):
    """The browser's predictive guard must not read 0 as "limit unknown".

    ``Number(0)`` is falsy and ``limit <= 0`` meant "no information, defer to
    the server" — which waved through exactly the account that must never
    launch. Asserted on the shipped source because /app has no build step and
    no JS test runner (the convention in ``_frontend_source``).
    """
    from dashboard.backend.tests._frontend_source import fn_body

    body = fn_body("function backtestConcurrencyRefusal(")
    assert "limit === 0" in body, body
    assert "limit <= 0" not in body, body
    assert "disabled for this account" in body, body


def test_ceiling_accepts_real_values_including_zero(monkeypatch):
    monkeypatch.setenv("MAX_ACTIVE_DASHBOARD_BACKTESTS", "12")
    assert bt._max_active_dashboard_backtests() == 12
    # 0 is "drain the runner", a legitimate operator setting rather than a typo.
    monkeypatch.setenv("MAX_ACTIVE_DASHBOARD_BACKTESTS", "0")
    assert bt._max_active_dashboard_backtests() == 0


def test_default_ceiling_lets_one_account_reach_its_own_quota():
    """A global cap below the per-account default makes the entitlement a lie.

    Read from the constants rather than pinned to a literal: whichever one
    moves, this is the relationship that has to survive.
    """
    import dashboard.backend.users as users_module

    assert (
        bt._DEFAULT_MAX_ACTIVE_DASHBOARD_BACKTESTS
        >= users_module.DEFAULT_MAX_CONCURRENT_BACKTESTS
    )


# ===========================================================================
# Who the slot is billed to
# ===========================================================================

def test_two_visitors_to_one_builtin_agent_do_not_share_a_slot():
    """Built-in agent runs file under the *agent's* session, not the caller's.

    Keying the cap on that session put every anonymous visitor into one bucket
    with a cap of 1, so one person's backtest refused everyone else's — on the
    public surface this platform's landing page points at.
    """
    builtin_session = str(uuid.uuid4())
    visitor_a = str(uuid.uuid4())
    visitor_b = str(uuid.uuid4())

    assert bt._try_acquire_backtest_slot(
        live_run_id="run-a",
        session_id=builtin_session,
        owner_session=visitor_a,
        user_id=None,
    ) is None
    assert bt._try_acquire_backtest_slot(
        live_run_id="run-b",
        session_id=builtin_session,
        owner_session=visitor_b,
        user_id=None,
    ) is None

    # ...and one visitor still cannot start two.
    refused = bt._try_acquire_backtest_slot(
        live_run_id="run-a2",
        session_id=builtin_session,
        owner_session=visitor_a,
        user_id=None,
    )
    assert refused and "already running" in refused


def test_a_builtin_run_is_visible_to_the_visitor_who_started_it():
    """Ownership must accept either identity, or the poll 404s its own run."""
    builtin_session = str(uuid.uuid4())
    visitor = {"X-Session-Id": str(uuid.uuid4())}

    bt._try_acquire_backtest_slot(
        live_run_id="builtin-run",
        session_id=builtin_session,
        owner_session=visitor["X-Session-Id"],
        user_id=None,
    )
    resp = TestClient(app).get(
        "/backtest/status", params={"live_run_id": "builtin-run"}, headers=visitor
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["running"] is True


# ===========================================================================
# The legacy mirror
# ===========================================================================

def test_the_mirror_never_advertises_one_run_beside_anothers_fields():
    """``_update_slot``'s slot-less branch wrote fields but not the id.

    It copied running/started_at/progress_file into the mirror while leaving
    live_run_id pointing at whatever ran last, publishing a pair that never
    coexisted — and both the status route and the concurrency count read it.
    """
    bt._update_slot("unslotted-run", running=True, progress_file="/tmp/x.json")
    assert bt.backtest_status["live_run_id"] == "unslotted-run"
    assert bt.backtest_status["progress_file"] == "/tmp/x.json"


def test_an_unslotted_write_does_not_hijack_a_registered_runs_mirror():
    owner = _sess()
    bt._try_acquire_backtest_slot(
        live_run_id="registered", session_id=owner["X-Session-Id"], user_id=None
    )
    bt._update_slot("stranger", running=True, progress_file="/tmp/stranger.json")
    assert bt.backtest_status["live_run_id"] == "registered"
    assert bt.backtest_status["progress_file"] != "/tmp/stranger.json"


def test_active_count_reads_the_ledger_not_the_mirror():
    """Admin stats reported 1 while five runs were live."""
    owner = _sess()
    for index in range(3):
        assert bt._try_acquire_backtest_slot(
            live_run_id=f"r{index}",
            session_id=f"{owner['X-Session-Id']}-{index}",
            user_id=None,
        ) is None
    assert bt.count_active_dashboard_backtests() == 3


def test_releasing_a_slot_frees_capacity_and_leaves_no_ghost():
    """A slot taken for a run that never started must vanish, not 'complete'.

    Finalising it instead parks a runs_count:0 entry in ``_recent_slots``,
    which the status route renders as "Backtest completed but no runs found
    for this session" — a failure message for a request that was refused.
    """
    owner = _sess()
    bt._try_acquire_backtest_slot(
        live_run_id="never-started", session_id=owner["X-Session-Id"], user_id=None
    )
    assert bt.count_active_dashboard_backtests() == 1

    bt._release_slot("never-started")
    assert bt.count_active_dashboard_backtests() == 0
    assert bt.backtest_status["running"] is False

    resp = TestClient(app).get(
        "/backtest/status", params={"live_run_id": "never-started"}, headers=owner
    )
    assert resp.status_code == 404, resp.text


# ===========================================================================
# Adapted-pipeline write-back
# ===========================================================================

def _run_row_with_adaptation(final_pipeline):
    return {
        "metadata": json.dumps(
            {
                "prompt_adaptations": [{"day": 1}],
                "final_pipeline": final_pipeline,
            }
        )
    }


def test_write_back_is_skipped_when_the_user_edited_the_agent_mid_run(monkeypatch):
    """Configure stays open during a run — so the user's own edit is at stake.

    Last-writer-wins here silently discards work the user explicitly saved,
    and the adapted pipeline is still recoverable from the run's metadata, so
    declining to write is the only asymmetric-cost-aware choice.
    """
    started_from = [{"name": "step-1"}]
    edited_since = [{"name": "step-1-EDITED-BY-USER"}]
    adapted = [{"name": "step-1-adapted"}]

    monkeypatch.setattr(
        bt.db, "get_run", lambda _run_id: _run_row_with_adaptation(adapted)
    )
    monkeypatch.setattr(
        bt.agent_service, "get_agent", lambda _agent_id: {"pipeline": edited_since}
    )
    writes = []
    monkeypatch.setattr(
        bt.agent_service,
        "update_agent",
        lambda agent_id, **kw: writes.append((agent_id, kw)),
    )

    bt._maybe_writeback_adapted_pipeline("agent-1", "run-1", started_from)
    assert writes == []


def test_write_back_still_happens_when_nothing_changed(monkeypatch):
    started_from = [{"name": "step-1"}]
    adapted = [{"name": "step-1-adapted"}]

    monkeypatch.setattr(
        bt.db, "get_run", lambda _run_id: _run_row_with_adaptation(adapted)
    )
    monkeypatch.setattr(
        bt.agent_service, "get_agent", lambda _agent_id: {"pipeline": started_from}
    )
    writes = []
    monkeypatch.setattr(
        bt.agent_service,
        "update_agent",
        lambda agent_id, **kw: writes.append((agent_id, kw)),
    )

    bt._maybe_writeback_adapted_pipeline("agent-1", "run-1", started_from)
    assert writes == [("agent-1", {"pipeline": adapted})]


def test_write_back_compares_through_json_not_object_identity(monkeypatch):
    """Stores hand pipelines back as a JSON string; a raw != would always differ."""
    started_from = [{"name": "step-1"}]
    adapted = [{"name": "step-1-adapted"}]

    monkeypatch.setattr(
        bt.db, "get_run", lambda _run_id: _run_row_with_adaptation(adapted)
    )
    monkeypatch.setattr(
        bt.agent_service,
        "get_agent",
        lambda _agent_id: {"pipeline": json.dumps(started_from)},
    )
    writes = []
    monkeypatch.setattr(
        bt.agent_service,
        "update_agent",
        lambda agent_id, **kw: writes.append((agent_id, kw)),
    )

    bt._maybe_writeback_adapted_pipeline("agent-1", "run-1", started_from)
    assert writes == [("agent-1", {"pipeline": adapted})]
