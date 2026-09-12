"""Cancelling a running dashboard backtest, and bounding what the parent keeps.

Two issues, one change, because they meet at the same line. ``run_backtest_
background`` used ``subprocess.run(capture_output=True, timeout=...)``, which

* returns no handle, so between launch and return there was nothing for a
  cancel route to act on (#273), and
* accumulates the child's entire stdout and stderr in parent memory for the
  life of the run (#308).

``Popen`` + drained pipes fixes both, and moves four things the stdlib used to
own into this repo: the wall-clock budget, the stream drain, the
SIGTERM/SIGKILL escalation, and the handle's lifetime on the slot. Each of
those is a case here.

The through-line for the cancel half: a cancel is the *user's own deliberate
action*, so every path that reports it must keep it distinct from a failure.
``_finalize_slot`` gained a third terminal state rather than a second flavour of
``error`` for that reason, and the assertions below check the distinction
survives at each boundary it crosses — slot, status route, analytics event.
"""

import subprocess
import uuid

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
import dashboard.backend.api.routers.backtests as bt
from dashboard.backend.tests._fake_child import FakeChild

_REAL_RUN_BACKTEST_BACKGROUND = bt.run_backtest_background


def _sess() -> dict:
    return {"X-Session-Id": str(uuid.uuid4())}


@pytest.fixture(autouse=True)
def _clean_slots():
    bt._reset_slots_for_tests()
    yield
    bt._reset_slots_for_tests()


@pytest.fixture
def client():
    return TestClient(app)


def _acquire(run_id: str, session_id: str) -> None:
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=None
        )
        is None
    )


# ===========================================================================
# POST /backtest/cancel
# ===========================================================================

def test_cancel_terminates_the_child_frees_the_quota_and_is_not_an_error():
    """The three things a cancel has to do, in one case.

    Freeing the quota is the half that is easy to leave to the worker thread:
    it would get there eventually, and the user would meanwhile be told they
    already have a backtest running. The route releases the slot as it accepts,
    so the very next launch is allowed.
    """
    owner = _sess()
    run_id = "agent_cancel_basic"
    _acquire(run_id, owner["X-Session-Id"])
    child = FakeChild()
    assert bt._attach_backtest_process(run_id, child) is True

    client = TestClient(app)
    resp = client.post("/backtest/cancel", json={"live_run_id": run_id}, headers=owner)

    assert resp.status_code == 200, resp.text
    assert resp.json()["cancelled"] is True
    # SIGTERM is delivered inline, so the caller's request has taken effect
    # before the response is written.
    assert child.terminated == 1
    # Quota freed immediately: the ledger no longer counts this run.
    assert run_id not in bt._active_slots
    assert bt._count_active_for_owner(f"session:{owner['X-Session-Id']}") == 0
    # ...and the outcome is cancelled, not failed.
    slot = bt._recent_slots[run_id]
    assert (slot["cancelled"], slot["error"], slot["running"]) == (True, None, False)


def test_status_reports_cancelled_as_its_own_state_not_as_a_failure():
    """`cancelled` must not arrive dressed as `error` or as `success`.

    Both frontend surfaces branch on these keys. A cancel routed through
    ``error`` paints the red "Backtest did not start" panel for an action the
    user took on purpose; one routed through ``success`` would claim results
    that do not exist.
    """
    owner = _sess()
    run_id = "agent_cancel_status"
    _acquire(run_id, owner["X-Session-Id"])
    bt._attach_backtest_process(run_id, FakeChild())

    client = TestClient(app)
    client.post("/backtest/cancel", json={"live_run_id": run_id}, headers=owner)

    resp = client.get(
        "/backtest/status", params={"live_run_id": run_id}, headers=owner
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["cancelled"] is True
    assert body["running"] is False
    assert body["message"] == "Backtest cancelled."
    assert "error" not in body
    assert "success" not in body


def test_cancelled_status_carries_no_decision_verdict():
    """Cancel and the provenance fields this PR stacks on must not mix.

    #458 made the completed branch report what actually drove a run --
    ``decision_note``, ``decision_badge``, the coverage counters. A cancelled
    run has no such verdict: it was stopped mid-flight, so there is nothing to
    be honest *about*, and emitting a coverage judgement for it would be a
    statement about steps that never ran. The two are sibling branches of the
    same if-chain, and this is what keeps them siblings.
    """
    owner = _sess()
    run_id = "agent_cancel_no_verdict"
    _acquire(run_id, owner["X-Session-Id"])
    bt._attach_backtest_process(run_id, FakeChild())

    client = TestClient(app)
    client.post("/backtest/cancel", json={"live_run_id": run_id}, headers=owner)
    body = client.get(
        "/backtest/status", params={"live_run_id": run_id}, headers=owner
    ).json()

    for field in (
        "decision_provenance",
        "decision_source",
        "decision_fallback",
        "decision_note",
        "decision_badge",
        "decision_steps",
        "llm_model",
        "llm_calls",
        "llm_decisions",
        "success",
    ):
        assert field not in body, field


def test_cancel_of_an_unknown_run_and_of_someone_elses_are_identically_404():
    """The refusal must not be usable as an oracle.

    A session id in this codebase is an access grant, not a label (see
    ``_owner_context``), so a response that distinguishes "no such run" from
    "not yours" lets a caller enumerate run ids and then learn whose they are.
    ``_resolve_status_slot``'s docstring states this for the status route; the
    cancel route reuses ``_slot_visible_to`` rather than restating the rule, and
    this case pins that the two answers stay byte-identical.
    """
    victim = _sess()
    attacker = _sess()
    run_id = "agent_cancel_victim"
    _acquire(run_id, victim["X-Session-Id"])
    bt._attach_backtest_process(run_id, FakeChild())

    client = TestClient(app)
    unknown = client.post(
        "/backtest/cancel", json={"live_run_id": "agent_no_such_run"}, headers=attacker
    )
    foreign = client.post(
        "/backtest/cancel", json={"live_run_id": run_id}, headers=attacker
    )

    assert unknown.status_code == foreign.status_code == 404
    assert unknown.json() == foreign.json()
    assert victim["X-Session-Id"] not in foreign.text
    # ...and the victim's run is untouched by the attempt.
    assert bt._active_slots[run_id]["running"] is True
    assert bt._active_slots[run_id]["cancel_requested"] is False


def test_cancel_that_races_a_completion_does_not_invent_a_cancel():
    """Issue #273 asks for this by name.

    The status route already carried a documented completion-detection race, and
    the tempting shape here — mark it cancelled, tell the user it worked — turns
    a finished run into a fabricated outcome. The honest answer is that there
    was nothing left to stop.
    """
    owner = _sess()
    run_id = "agent_cancel_late"
    _acquire(run_id, owner["X-Session-Id"])
    bt._finalize_slot(run_id, error=None, runs_count=3)

    client = TestClient(app)
    resp = client.post("/backtest/cancel", json={"live_run_id": run_id}, headers=owner)

    assert resp.status_code == 200, resp.text
    assert resp.json()["cancelled"] is False
    assert "already finished" in resp.json()["message"]
    # The completed run keeps its own verdict.
    assert bt._recent_slots[run_id]["cancelled"] is False
    assert bt._recent_slots[run_id]["runs_count"] == 3


def test_cancel_before_the_child_launches_kills_it_at_launch(monkeypatch):
    """The slot is taken in the request handler; the child starts a thread later.

    A cancel in that window has no handle to signal. Answering it with a shrug
    would leave a subprocess running against a quota its owner has already been
    told is free — the exact leak the slot ledger exists to prevent — so the
    refusal is carried forward to ``_attach_backtest_process`` and the child is
    killed the moment it exists.
    """
    owner = _sess()
    run_id = "agent_cancel_prelaunch"
    _acquire(run_id, owner["X-Session-Id"])

    client = TestClient(app)
    resp = client.post("/backtest/cancel", json={"live_run_id": run_id}, headers=owner)
    assert resp.status_code == 200 and resp.json()["cancelled"] is True

    child = FakeChild()
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: child)

    with pytest.raises(bt._BacktestCancelled):
        bt._run_backtest_subprocess(
            ["python", "-c", "pass"],
            cwd=".",
            env={},
            stdin_payload="",
            timeout=60,
            live_run_id=run_id,
        )

    assert bt._attach_backtest_process(run_id, child) is False
    assert child.terminated == 1


def test_a_late_worker_finalize_cannot_overwrite_a_cancel():
    """The route finalizes instead of waiting for the worker thread.

    That is what frees the quota immediately, and it leaves a window of
    microseconds in which the worker's own finalize still arrives. The cancel
    has to win: the alternative is a cancelled run reported as a zero-run
    completion, which the status route renders as "No backtest has been run
    yet". Worse, the branch a late finalize falls into writes the legacy
    ``backtest_status`` mirror unconditionally, so it would also stamp this
    run's outcome onto whichever run that mirror currently describes.
    """
    owner = _sess()
    run_id = "agent_cancel_late_finalize"
    _acquire(run_id, owner["X-Session-Id"])
    bt._attach_backtest_process(run_id, FakeChild())
    bt._cancel_backtest_slot(
        live_run_id=run_id, session_id=owner["X-Session-Id"], user_id=None
    )
    other_run = "agent_unrelated_mirror_owner"
    _acquire(other_run, owner["X-Session-Id"])

    # The worker, a beat late, finalizing what it believes was a failure.
    bt._finalize_slot(run_id, error="Backtest failed with return code -15", runs_count=0)

    slot = bt._recent_slots[run_id]
    assert (slot["cancelled"], slot["error"]) == (True, None)
    # The unrelated run the mirror describes is untouched.
    assert bt.backtest_status["live_run_id"] == other_run
    assert bt.backtest_status["error"] is None
    assert bt.backtest_status["running"] is True


def test_sigterm_escalates_to_sigkill_only_when_the_child_ignores_it():
    """Terminate, grace, kill — and no kill for a child that goes quietly."""
    stubborn = FakeChild(timeout_waits=1)
    bt._kill_backtest_process_after_grace(stubborn, grace_seconds=0)
    assert stubborn.killed == 1

    obedient = FakeChild()
    bt._kill_backtest_process_after_grace(obedient, grace_seconds=0)
    assert obedient.killed == 0


def test_cancel_emits_a_cancelled_analytics_event_not_a_failure(monkeypatch):
    """A dashboard that counts cancels as failures measures the product as
    broken every time a user changes their mind."""
    events = []
    monkeypatch.setattr(
        bt.analytics_instrumentation,
        "emit_run_event",
        lambda **kwargs: events.append(kwargs),
    )
    run_id = "agent_cancel_analytics"
    session_id = str(uuid.uuid4())
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=4242
        )
        is None
    )

    bt._finalize_slot(run_id, error=None, runs_count=0, cancelled=True)

    assert [e["event_name"] for e in events] == ["backtest_cancelled"]
    assert events[0]["error_category"] is None


# ===========================================================================
# The worker under Popen
# ===========================================================================

def test_worker_reports_a_cancelled_child_as_cancelled_not_as_return_code(monkeypatch):
    """A child killed by our own SIGTERM exits non-zero.

    Reading that return code the ordinary way reports the user's cancel back to
    them as "Backtest failed with return code -15". The worker therefore checks
    the cancel flag *before* the return-code branch — and after the log dump, so
    a cancelled run still leaves its output behind.
    """
    owner = _sess()
    run_id = "agent_cancel_worker"
    session_id = owner["X-Session-Id"]
    _acquire(run_id, session_id)

    finalized = []
    real_finalize = bt._finalize_slot

    def spy_finalize(live_run_id, *, error, runs_count, cancelled=False):
        finalized.append((live_run_id, error, runs_count, cancelled))
        return real_finalize(
            live_run_id, error=error, runs_count=runs_count, cancelled=cancelled
        )

    class CancellingChild(FakeChild):
        """Cancels itself from inside the parent's wait — the real ordering."""

        def wait(self, timeout=None):
            bt._cancel_backtest_slot(
                live_run_id=run_id, session_id=session_id, user_id=None
            )
            return super().wait(timeout)

    child = CancellingChild(returncode=-15, stdout="universe: DJIA30\n")
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: child)
    monkeypatch.setattr(bt, "_finalize_slot", spy_finalize)
    monkeypatch.setattr(bt, "run_backtest_background", _REAL_RUN_BACKTEST_BACKGROUND)

    bt.run_backtest_background(
        start_date="2026-01-01",
        end_date="2026-01-02",
        session_id=session_id,
        live_run_id=run_id,
        decision_source="rule_based",
    )

    # Finalized exactly once, by the cancel, as cancelled. A second call from
    # the worker's `finally` would overwrite that with a zero-run completion,
    # which the status route reports as "No backtest has been run yet".
    assert finalized == [(run_id, None, 0, True)]
    slot = bt._recent_slots[run_id]
    assert (slot["cancelled"], slot["error"]) == (True, None)
    assert run_id not in bt._active_slots

    resp = TestClient(app).get(
        "/backtest/status", params={"live_run_id": run_id}, headers=owner
    )
    assert resp.json()["cancelled"] is True
    assert "error" not in resp.json()


def test_worker_still_reports_a_genuine_failure_as_an_error(monkeypatch):
    """The cancel branch must not swallow ordinary non-zero exits."""
    owner = _sess()
    run_id = "agent_plain_failure"
    _acquire(run_id, owner["X-Session-Id"])

    child = FakeChild(returncode=2, stderr="boom\n")
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: child)
    monkeypatch.setattr(bt, "run_backtest_background", _REAL_RUN_BACKTEST_BACKGROUND)

    bt.run_backtest_background(
        start_date="2026-01-01",
        end_date="2026-01-02",
        session_id=owner["X-Session-Id"],
        live_run_id=run_id,
        decision_source="rule_based",
    )

    slot = bt._recent_slots[run_id]
    assert slot["cancelled"] is False
    assert "return code 2" in slot["error"]


def test_handoff_payload_still_reaches_the_child_over_stdin(monkeypatch):
    """The signed worker handoff never travels as an argv or an env var.

    ``subprocess.run(input=...)`` did the write; under ``Popen`` it is this
    module's job, and losing it would push a credential-bearing payload back
    onto the command line — or simply fail every LLM run.
    """
    run_id = "agent_stdin_payload"
    session_id = str(uuid.uuid4())
    _acquire(run_id, session_id)
    child = FakeChild()
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: child)

    bt._run_backtest_subprocess(
        ["python", "-c", "pass"],
        cwd=".",
        env={},
        stdin_payload="opaque-signed-handoff",
        timeout=60,
        live_run_id=run_id,
    )

    assert child.stdin.value == "opaque-signed-handoff"
    assert child.stdin.closed is True


# ===========================================================================
# Bounded parent retention (#308)
# ===========================================================================

def test_retention_keeps_the_head_and_the_tail_and_marks_the_gap():
    """Head AND tail, because they answer different questions.

    The head says what the run *is* (universe, decision source, FX bootstrap);
    the tail says what went wrong. A plain ring buffer loses the first and a
    prefix buffer loses the second, and print() is the only log channel the
    deployed config has.
    """
    capture = bt._BoundedStreamCapture(head_chars=40, tail_chars=40)
    capture.feed("HEAD universe: DJIA30\n")
    for index in range(500):
        capture.feed(f"filler line {index}\n")
    capture.feed("TAIL traceback: boom\n")

    text = capture.text()

    assert "HEAD universe: DJIA30" in text
    assert "TAIL traceback: boom" in text
    assert "filler line 200" not in text
    # The gap is announced. A dump that silently omits its middle is the same
    # unmarked lie as a fallback that cannot be told from a success.
    assert "characters of backtest output dropped" in text
    assert capture.dropped_chars > 0


def test_retention_holds_against_a_child_that_floods_its_pipe(monkeypatch):
    """The bound is on the PARENT's resident set, which is what Render kills."""
    monkeypatch.setattr(bt, "SUBPROCESS_LOG_HEAD_CHARS", 200)
    monkeypatch.setattr(bt, "SUBPROCESS_LOG_TAIL_CHARS", 200)
    flood = "".join(f"noisy backtest line {index}\n" for index in range(20_000))
    child = FakeChild(stdout="FIRST LINE\n" + flood + "LAST LINE\n")
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: child)

    result = bt._run_backtest_subprocess(
        ["python", "-c", "pass"],
        cwd=".",
        env={},
        stdin_payload="",
        timeout=60,
        live_run_id=None,
    )

    assert len(flood) > 400_000
    # Bounded by head + tail + the marker, not by what the child chose to write.
    assert len(result.stdout) < 1_000
    assert result.stdout.startswith("FIRST LINE")
    assert result.stdout.rstrip().endswith("LAST LINE")


def test_redaction_still_applies_to_everything_retained(monkeypatch, capsys):
    """Truncation and redaction stayed separate concerns, in that order.

    Bounding the buffer must not open a hole in the credential scrub: the head
    and the tail are both printed, so both are redacted. The secret is planted
    at both ends precisely so a pass cannot come from the middle being dropped.
    """
    monkeypatch.setattr(bt, "SUBPROCESS_LOG_HEAD_CHARS", 300)
    monkeypatch.setattr(bt, "SUBPROCESS_LOG_TAIL_CHARS", 300)
    secret = "financial-datasets-plaintext-canary"
    run_id = "agent_redaction_bounded"
    session_id = str(uuid.uuid4())
    _acquire(run_id, session_id)

    noise = "".join(f"line {index}\n" for index in range(5_000))
    child = FakeChild(
        returncode=0,
        stdout=f"HEAD key={secret}\n{noise}TAIL key={secret}\n",
    )
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: child)
    monkeypatch.setattr(bt, "run_backtest_background", _REAL_RUN_BACKTEST_BACKGROUND)

    bt.run_backtest_background(
        start_date="2026-01-01",
        end_date="2026-01-02",
        session_id=session_id,
        live_run_id=run_id,
        decision_source="rule_based",
        financial_datasets_api_key=secret,
    )

    printed = capsys.readouterr().out
    assert secret not in printed
    assert printed.count("[REDACTED]") >= 2
    assert "characters of backtest output dropped" in printed


# ===========================================================================
# AI Hedge Fund pre-flight window bound (#308)
# ===========================================================================

def test_ai_hedge_fund_window_bound_refuses_a_range_it_cannot_survive(monkeypatch):
    """A refusal a user cannot act on is the same dead end as no exit at all.

    422 rather than 503, and the message carries both numbers: this one is the
    caller's to fix by shortening the range.
    """
    monkeypatch.setattr(bt, "MAX_AI_HEDGE_FUND_TRADING_DAYS", 10)

    with pytest.raises(bt.HTTPException) as excinfo:
        bt._enforce_ai_hedge_fund_window("2026-01-01", "2026-03-31")

    assert excinfo.value.status_code == 422
    assert "10 trading days" in excinfo.value.detail
    assert "Shorten the date range" in excinfo.value.detail
    # A window inside the bound is untouched.
    assert bt._enforce_ai_hedge_fund_window("2026-01-01", "2026-01-09") is None


def test_ai_hedge_fund_window_bound_of_zero_turns_the_runtime_off(monkeypatch):
    """0 disables, the same meaning MAX_ACTIVE_DASHBOARD_BACKTESTS gives it —
    the operator exit for a host that cannot run this at all."""
    monkeypatch.setattr(bt, "MAX_AI_HEDGE_FUND_TRADING_DAYS", 0)

    with pytest.raises(bt.HTTPException) as excinfo:
        bt._enforce_ai_hedge_fund_window("2026-01-01", "2026-01-02")

    # 503, not 422: nothing the caller can change about their request fixes it.
    assert excinfo.value.status_code == 503


@pytest.mark.parametrize("raw", ["ten", "", "   ", "-1", "9999", "1.5"])
def test_ai_hedge_fund_window_bound_falls_back_rather_than_raising(monkeypatch, raw):
    """CLAUDE.md records that a bare int() at module scope in this very module
    once killed app boot. A mistyped Render field must not do it again."""
    monkeypatch.setenv("MAX_AI_HEDGE_FUND_TRADING_DAYS", raw)

    assert (
        bt._max_ai_hedge_fund_trading_days()
        == bt._DEFAULT_MAX_AI_HEDGE_FUND_TRADING_DAYS
    )


def test_ai_hedge_fund_window_bound_honours_a_valid_override(monkeypatch):
    monkeypatch.setenv("MAX_AI_HEDGE_FUND_TRADING_DAYS", "25")
    assert bt._max_ai_hedge_fund_trading_days() == 25
    monkeypatch.setenv("MAX_AI_HEDGE_FUND_TRADING_DAYS", "0")
    assert bt._max_ai_hedge_fund_trading_days() == 0


def test_backtest_run_refuses_an_over_long_hosted_window(client, monkeypatch):
    """Route level: the refusal lands before any credential is touched.

    With no OPENROUTER_API_KEY configured this agent's short-window runs answer
    503 (`test_ai_hedge_fund_requires_openrouter_not_direct_openai`). A long
    window answering 422 instead is the proof that the pre-flight runs first —
    a window this deployment refuses is refused whether or not the agent is
    fully configured.
    """
    monkeypatch.setattr(bt, "MAX_AI_HEDGE_FUND_TRADING_DAYS", 10)
    headers = {"X-Session-Id": str(uuid.uuid4())}
    agent = client.post(
        "/api/v1/agents/marketplace/ai-hedge-fund/clone",
        json={},
        headers=headers,
    ).json()["agent"]

    response = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-01-01",
            "end_date": "2026-03-31",
            "agent_id": agent["agent_id"],
        },
        headers=headers,
    )

    assert response.status_code == 422, response.text
    assert "10 trading days" in response.text
