"""Adversarial-review fixes for the /api/v2 merge (reconciliation hardening).

Each test pins a finding confirmed by the 5-lens review of merge a93d39d:
v2 code written against the pre-hardening flat backend inherited failure
modes this PR had already fixed on the v1 surface.
"""

import time
import uuid

import pytest
from fastapi.testclient import TestClient

import dashboard.backend.api.v2.runs as runs_mod
from dashboard.backend.api.v2.errors import ApiError
from dashboard.backend.api.v2.rate_limit import TokenBucketLimiter
from dashboard.backend.app import app
from dashboard.backend.execution.backtest_backend import BacktestBackend, ext
from dashboard.backend.tests._v2_fakes import FakeBackend

client = TestClient(app)


def _agent(name):
    r = client.post("/api/v2/agents", json={"name": name}).json()
    return r["api_key"], r["session_id"], r["agent_id"]


# -- SystemExit guard on the background loader ------------------------------

def test_background_load_systemexit_marks_run_failed(monkeypatch):
    """AlpacaDataLoader sys.exit()s when creds are absent; a daemon thread
    silently swallows an uncaught SystemExit, so `except Exception` alone
    leaves the run stuck in "loading" forever (the B0 failure mode, fixed on
    v1 in this PR but not ported to the v2 backend by the merge)."""

    class _Boom:
        def __init__(self):
            raise SystemExit(1)

    monkeypatch.setattr(ext, "AlpacaDataLoader", _Boom)
    backend = BacktestBackend(
        run_id="run_se_guard", session_id="sess_se", agent_name="a",
        model_name="m", start_date="2026-04-15", end_date="2026-04-16",
    )
    backend.start_background_load()
    deadline = time.time() + 5
    while time.time() < deadline and backend.session.status == "loading":
        time.sleep(0.02)
    assert backend.session.status == "failed"
    assert backend.session.error


# -- apply_decisions maps engine rejections to typed errors ------------------

class _RejectingSession:
    step_index = 3
    status = "waiting_decision"

    def __init__(self, result):
        self._result = result

    def submit_decisions(self, payload):
        return self._result


def _backend_with(session):
    backend = BacktestBackend.__new__(BacktestBackend)
    backend.run_id = "run_reject"
    backend.session = session
    return backend


def test_apply_decisions_late_decision_raises_step_already_closed():
    """A deadline-expired submission must surface as the spec'd
    step_already_closed error, not as an accepted-looking ack attributed to
    decision_source="external_agent"."""
    backend = _backend_with(_RejectingSession({
        "accepted": False, "error": "step_already_closed",
        "outcome": "timeout_hold", "next_step": 4, "status": "waiting_decision",
    }))
    with pytest.raises(ApiError) as ei:
        backend.apply_decisions([])
    assert ei.value.code == "step_already_closed"
    assert ei.value.status == 409
    assert ei.value.retryable is True
    assert ei.value.details["outcome"] == "timeout_hold"


def test_apply_decisions_completed_run_raises_invalid_status():
    backend = _backend_with(_RejectingSession({
        "accepted": False, "error": "backtest_already_completed", "run_id": "r",
    }))
    with pytest.raises(ApiError) as ei:
        backend.apply_decisions([])
    assert ei.value.code == "invalid_status"
    assert ei.value.status == 409


def test_apply_decisions_not_waiting_raises_invalid_status_with_state():
    backend = _backend_with(_RejectingSession({
        "accepted": False, "error": "invalid_status:loading",
    }))
    with pytest.raises(ApiError) as ei:
        backend.apply_decisions([])
    assert ei.value.code == "invalid_status"
    assert "loading" in ei.value.message


# -- v2 run lookup: no existence oracle --------------------------------------

def test_v2_run_lookup_is_not_an_existence_oracle():
    """404 bodies for "doesn't exist" and "not yours" must be identical in
    shape — a message-text difference lets any registered key enumerate
    which run ids exist."""
    _, sid_a, _ = _agent("oracle-owner")
    key_b, _, _ = _agent("oracle-prober")
    run_id = f"run_oracle_{uuid.uuid4().hex[:6]}"
    runs_mod.register_run(run_id, FakeBackend(run_id=run_id, session_id=sid_a), sid_a)

    denied = client.get(f"/api/v2/runs/{run_id}", headers={"X-API-Key": key_b})
    missing = client.get(
        f"/api/v2/runs/run_missing_{uuid.uuid4().hex[:6]}", headers={"X-API-Key": key_b}
    )
    assert denied.status_code == 404 and missing.status_code == 404
    d, m = denied.json()["error"], missing.json()["error"]
    assert d["code"] == m["code"] == "run_not_found"
    # Same template both ways: only the caller's own probed id may differ.
    assert d["message"] == f"Run {run_id} not found"
    assert m["message"].endswith("not found")


# -- per-agent active-run cap -------------------------------------------------

class _StubBackend:
    loop = "lockstep"
    news_sentiment_source = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start_background_load(self):
        pass

    def is_active(self):
        return True

    def status(self):
        return {"status": "waiting_decision"}


def test_create_run_enforces_per_agent_active_cap(monkeypatch):
    """v1 got MAX_ACTIVE_RUNS_PER_AGENT in H4; the same agent identity must
    not be able to sidestep it by creating runs through v2 instead."""
    key, _, _ = _agent("capped-agent")
    monkeypatch.setattr(runs_mod, "BacktestBackend", _StubBackend)
    monkeypatch.setattr(runs_mod, "MAX_ACTIVE_RUNS_PER_AGENT", 2)

    body = {"start_date": "2026-04-15", "end_date": "2026-04-16"}
    headers = {"X-API-Key": key}
    assert client.post("/api/v2/runs", json=body, headers=headers).status_code == 200
    assert client.post("/api/v2/runs", json=body, headers=headers).status_code == 200
    r3 = client.post("/api/v2/runs", json=body, headers=headers)
    assert r3.status_code == 429
    err = r3.json()["error"]
    assert err["code"] == "too_many_active_runs"
    assert err["details"]["limit"] == 2


def test_cap_ignores_other_agents_and_terminal_runs(monkeypatch):
    key, _, _ = _agent("capped-agent-2")

    class _DoneBackend(_StubBackend):
        def is_active(self):
            return False

        def status(self):
            return {"status": "completed"}

    monkeypatch.setattr(runs_mod, "MAX_ACTIVE_RUNS_PER_AGENT", 1)
    monkeypatch.setattr(runs_mod, "BacktestBackend", _DoneBackend)
    body = {"start_date": "2026-04-15", "end_date": "2026-04-16"}
    headers = {"X-API-Key": key}
    # A terminal run does not count against the cap.
    assert client.post("/api/v2/runs", json=body, headers=headers).status_code == 200
    assert client.post("/api/v2/runs", json=body, headers=headers).status_code == 200


def test_create_run_enforces_per_account_entitlement_cap(monkeypatch):
    """The account-level ``max_concurrent_backtests`` entitlement binds on v2
    exactly as on v1 (test_protocol_api covers that copy) — both surfaces call
    the same check under the shared create lock, and this pins the v2 wiring
    against the two call sites drifting apart."""
    from dashboard.backend import users as users_module
    from dashboard.backend.domain.agents.repository import agent_store

    user = users_module.user_store.create_user(
        email=f"v2cap-{uuid.uuid4().hex[:8]}@example.com",
        display_name="V2 Cap",
        password="SecurePass1!",
    )
    # v2 registration never binds an owner, so claim ownership at the store —
    # the same state /api/v1/agents produces for a signed-in create.
    agent = agent_store.create_agent(name="v2-acct-capped", owner_user_id=user["id"])
    monkeypatch.setattr(runs_mod, "BacktestBackend", _StubBackend)

    body = {"start_date": "2026-04-15", "end_date": "2026-04-16"}
    headers = {"X-API-Key": agent["api_key"]}
    # Set the quota rather than leaning on the default: the default is every
    # pre-existing account's live limit on the deploy that ships this plane, so
    # a test pinned to it makes lowering it look free.
    users_module.user_store.set_entitlements(user["id"], max_concurrent_backtests=1)
    assert client.post("/api/v2/runs", json=body, headers=headers).status_code == 200
    second = client.post("/api/v2/runs", json=body, headers=headers)
    assert second.status_code == 429
    err = second.json()["error"]
    assert err["code"] == "too_many_active_runs_for_account"
    assert err["details"]["limit"] == 1


# -- registration flood control ----------------------------------------------

def test_v2_registration_is_rate_limited(monkeypatch):
    """POST /api/v2/agents is unauthenticated by design; without a per-client
    budget it is an unbounded DB-write + rate-limit-bucket flood vector."""
    import dashboard.backend.api.v2.agents as agents_mod
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter

    monkeypatch.setattr(
        agents_mod, "_register_rate_limiter",
        FixedWindowRateLimiter(max_events=3, window_seconds=3600),
    )
    headers = {"X-Browser-Id": "reg-flood-client"}
    codes = [
        client.post("/api/v2/agents", json={"name": f"flood-{i}"}, headers=headers).status_code
        for i in range(5)
    ]
    assert codes[:3] == [200, 200, 200]
    assert set(codes[3:]) == {429}


# -- token-bucket registry is bounded ------------------------------------------

def test_token_bucket_registry_is_bounded():
    lim = TokenBucketLimiter(per_minute=60, max_buckets=100)
    for i in range(250):
        lim.check(f"agent_{i}")
    assert len(lim._buckets) <= 100


# -- rejected submits replay as the same error, never as fresh execution -------

class _ClosingBackend:
    """First submit rejects like a deadline race: the engine auto-holds the
    step and ADVANCES before raising."""
    loop = "lockstep"

    def __init__(self):
        self.calls = 0
        self._step = 3

    def current_step_index(self):
        return self._step

    def apply_decisions(self, actions):
        self.calls += 1
        self._step = 4
        raise ApiError(
            "step_already_closed", "closed", status=409, retryable=True,
            details={"outcome": "timeout_hold", "next_step": 4},
        )


def test_rejected_submit_replays_same_error_not_next_step():
    """A step-consuming rejection must be idempotency-cached: the engine
    advanced, so a same-key retry that misses the cache would pass the step
    re-check and execute the stale actions against the NEXT step's prices."""
    _, sid, _ = _agent("replay-agent")
    run_id = f"run_replay_{uuid.uuid4().hex[:6]}"
    backend = _ClosingBackend()
    runs_mod.register_run(run_id, backend, sid)
    key = f"idem_{uuid.uuid4().hex[:6]}"

    with pytest.raises(ApiError) as first:
        runs_mod._submit_for(run_id, sid, key, [])
    assert first.value.code == "step_already_closed"

    with pytest.raises(ApiError) as retry:
        runs_mod._submit_for(run_id, sid, key, [])
    assert retry.value.code == "step_already_closed"
    assert retry.value.status == 409
    assert backend.calls == 1, "retry must replay the cached rejection, not re-execute"


def test_non_consuming_rejection_is_not_cached():
    """invalid_status (run loading/closed) consumes nothing server-side; the
    same key must be retryable once the run becomes ready."""
    _, sid, _ = _agent("retry-agent")
    run_id = f"run_retry_{uuid.uuid4().hex[:6]}"

    class _LoadingThenReady:
        loop = "lockstep"

        def __init__(self):
            self.calls = 0

        def current_step_index(self):
            return 0

        def apply_decisions(self, actions):
            self.calls += 1
            if self.calls == 1:
                raise ApiError("invalid_status", "Run is not awaiting a decision (status: loading)",
                               status=409, retryable=True)
            return {"accepted": True, "executed": [], "rejected": [],
                    "decision_source": "external_agent", "next_step": 1,
                    "status": "waiting_decision", "run_id": run_id, "metrics": None}

    backend = _LoadingThenReady()
    runs_mod.register_run(run_id, backend, sid)
    key = f"idem_{uuid.uuid4().hex[:6]}"
    with pytest.raises(ApiError):
        runs_mod._submit_for(run_id, sid, key, [])
    ack = runs_mod._submit_for(run_id, sid, key, [])
    assert ack["accepted"] is True
    assert backend.calls == 2


# -- cap counting must be passive (no engine side effects under the lock) ------

def test_active_run_count_is_side_effect_free():
    """Counting active runs happens under the global create lock; on a LIVE
    backend it must stay passive — only the is_active() peek, never status()
    or advance(), which can cascade into _maybe_apply_timeout/_finalize
    (seconds of baselines). (Counting itself now reads the protocol_runs
    ledger shared with v1; the registry walk only reconciles finished
    backends, so a live one must be left completely untouched.)"""

    class _TrackingBackend:
        def __init__(self):
            self.status_called = False
            self.advance_called = False

        def is_active(self):
            return True

        def status(self):
            self.status_called = True
            return {"status": "waiting_decision"}

        def advance(self):
            self.advance_called = True

    from dashboard.backend.domain.runs.repository import run_store

    backend = _TrackingBackend()
    agent_id = f"agent_passive_{uuid.uuid4().hex[:6]}"
    run_id = f"run_passive_{uuid.uuid4().hex[:6]}"
    # The count reads the shared protocol_runs ledger (one row per v2 run).
    run_store.create_run(
        run_id=run_id, agent_id=agent_id, agent_version_id=None,
        session_id="sid_passive", environment_id=None,
        environment_type="backtest", config={}, backtest_id=None,
        status="running",
    )
    runs_mod.register_run(run_id, backend, "sid_passive", agent_id)
    assert runs_mod._active_run_count(agent_id) == 1
    assert backend.status_called is False, (
        "cap counting must use the passive is_active() peek, not status()"
    )
    assert backend.advance_called is False, (
        "cap counting must never drive the run forward"
    )


# -- 429s carry the rate-limit headers ------------------------------------------

def test_enforce_429_carries_rate_limit_headers(monkeypatch):
    """The 429 itself must tell the client its budget — api_error_handler
    builds a fresh JSONResponse, so headers set via the injected Response
    object before the raise are otherwise silently dropped."""
    from dashboard.backend.api.v2 import rate_limit as rl

    monkeypatch.setattr(rl, "limiter", rl.TokenBucketLimiter(per_minute=1, burst=1))
    monkeypatch.setattr(runs_mod, "BacktestBackend", _StubBackend)
    key, _, _ = _agent("hdr-agent")
    body = {"start_date": "2026-04-15", "end_date": "2026-04-16"}
    headers = {"X-API-Key": key}
    assert client.post("/api/v2/runs", json=body, headers=headers).status_code == 200
    r = client.post("/api/v2/runs", json=body, headers=headers)
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "rate_limited"
    for name in ("x-ratelimit-limit", "x-ratelimit-remaining", "retry-after"):
        assert name in r.headers, f"{name} missing on the 429"


# -- the engine literals _raise_rejection matches must not drift ----------------

def test_engine_rejection_literals_still_present():
    """_raise_rejection string-matches the engine's error literals; if the
    engine rewords them, every rejection would fall through to a generic 422.
    Source-guard both sides of the stringly-typed contract."""
    from pathlib import Path

    from dashboard.backend.domain.backtesting import external_run_service as ext

    src = Path(ext.__file__).read_text(encoding="utf-8")
    for literal in ('"backtest_already_completed"', '"step_already_closed"',
                    'f"invalid_status:'):
        assert literal in src, literal


# -- CORS exposes the v2 rate-limit headers ------------------------------------

def test_rate_limit_headers_exposed_to_cors_clients():
    """The spec promises X-RateLimit-*/Retry-After to clients; without
    Access-Control-Expose-Headers browsers strip them from fetch() results."""
    key, _, _ = _agent("cors-agent")
    r = client.get(
        "/api/v2/agents/me",
        headers={"X-API-Key": key, "Origin": "https://agentic-trading-lab.vercel.app"},
    )
    exposed = r.headers.get("access-control-expose-headers", "").lower()
    for header in ("x-ratelimit-limit", "x-ratelimit-remaining",
                   "x-ratelimit-reset", "retry-after"):
        assert header in exposed, f"{header} missing from {exposed!r}"
