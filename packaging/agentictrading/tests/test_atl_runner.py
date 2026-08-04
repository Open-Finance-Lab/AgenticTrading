"""Unit tests for ``AgentRunner`` driving the full step loop over a fake backend."""

from __future__ import annotations

import json

import pytest

from agentictrading import (
    AgentRunner,
    ATLClient,
    ATLRunFailedError,
    ATLTimeoutError,
    RunResult,
)
from agentictrading.runner import wait_for_poll

API_KEY = "ag_secret_should_never_leak"


def _client() -> ATLClient:
    return ATLClient("http://test.local", API_KEY, timeout=5)


_AWAITING = {
    "status": "awaiting_decision",
    "run_id": "run_1",
    "step_id": "step_1",
    "sequence": 0,
    "observation": {"market": {"features": {}}, "portfolio": {"cash": 100, "equity": 100, "positions": []}},
    "constraints": {},
}
_LOADING = {"status": "loading", "run_id": "run_1", "message": "loading"}
_COMPLETED = {"status": "completed", "run_id": "run_1", "result_run_id": "ext_1"}


class RunnerBackend:
    """Routes the handful of endpoints AgentRunner touches."""

    def __init__(self, step_sequence, result=None):
        self.step_sequence = list(step_sequence)
        self.result = result or {"run_id": "run_1", "status": "completed",
                                 "metrics": {"total_return": 0.05}}
        self.submitted = []

    def __call__(self, req):
        url = req.full_url
        method = req.get_method()
        if method == "POST" and url.endswith("/api/v1/runs"):
            return (200, {"run_id": "run_1", "status": "created"})
        if url.endswith("/steps/next"):
            return (200, self.step_sequence.pop(0))
        if url.endswith("/decision"):
            self.submitted.append(json.loads(req.data.decode()))
            return (200, {"run_id": "run_1", "step_id": "step_1", "accepted": True,
                          "validation": {"passed": True, "rejections": []}, "fills": [],
                          "portfolio_after": {}, "run_status": "running"})
        if url.endswith("/result"):
            return (200, self.result)
        raise AssertionError(f"unexpected url {url}")


def _run(runner, **kw):
    kw.setdefault("poll_interval", 0.001)  # small positive: >0 is now required
    return runner.run_backtest(
        "agv_1",
        environment_id="us-equity-hourly-v1",
        start_date="2026-04-15",
        end_date="2026-04-16",
        **kw,
    )


def test_runner_loading_then_decision_then_completed(fake_http):
    class HoldAgent:
        def __init__(self):
            self.decide_calls = 0
            self.exec_results = []

        def decide(self, observation):
            self.decide_calls += 1
            return {"orders": [], "rationale": "hold"}

        def on_execution_result(self, result):
            self.exec_results.append(result)

    backend = RunnerBackend([_LOADING, _AWAITING, _COMPLETED])
    fake_http(backend)
    agent = HoldAgent()
    result = _run(AgentRunner(_client(), agent))

    assert isinstance(result, RunResult)
    assert result.metrics["total_return"] == 0.05
    assert agent.decide_calls == 1
    assert len(agent.exec_results) == 1  # hook fired once
    assert len(backend.submitted) == 1


def test_runner_completed_immediately(fake_http):
    class Agent:
        def decide(self, observation):
            raise AssertionError("decide should not be called")

    backend = RunnerBackend([_COMPLETED])
    fake_http(backend)
    result = _run(AgentRunner(_client(), Agent()))
    assert result.metrics["total_return"] == 0.05
    assert backend.submitted == []


def test_runner_failed_state_raises(fake_http):
    class Agent:
        def decide(self, observation):
            return {"orders": []}

    backend = RunnerBackend([{"status": "failed", "run_id": "run_1", "message": "engine crashed"}])
    fake_http(backend)
    with pytest.raises(ATLRunFailedError) as ei:
        _run(AgentRunner(_client(), Agent()))
    assert "engine crashed" in str(ei.value)


def test_runner_hook_failure_surfaces_and_no_resubmit(fake_http):
    class BoomAgent:
        def __init__(self):
            self.decide_calls = 0

        def decide(self, observation):
            self.decide_calls += 1
            return {"orders": []}

        def on_execution_result(self, result):
            raise RuntimeError("hook fail")

    backend = RunnerBackend([_AWAITING, _COMPLETED])
    fake_http(backend)
    agent = BoomAgent()
    with pytest.raises(RuntimeError, match="hook fail"):
        _run(AgentRunner(_client(), agent))
    assert agent.decide_calls == 1
    assert len(backend.submitted) == 1  # accepted decision never resubmitted


def test_runner_optional_hook_absent_ok(fake_http):
    class MinimalAgent:
        def decide(self, observation):
            return {"orders": []}

    backend = RunnerBackend([_AWAITING, _COMPLETED])
    fake_http(backend)
    result = _run(AgentRunner(_client(), MinimalAgent()))
    assert result.metrics["total_return"] == 0.05


def test_runner_on_run_completed_hook(fake_http):
    class Agent:
        def __init__(self):
            self.completed = []

        def decide(self, observation):
            return {"orders": []}

        def on_run_completed(self, result):
            self.completed.append(result)

    backend = RunnerBackend([_AWAITING, _COMPLETED])
    fake_http(backend)
    agent = Agent()
    _run(AgentRunner(_client(), agent))
    assert len(agent.completed) == 1


def test_runner_max_wait_timeout(fake_http):
    class Agent:
        def decide(self, observation):
            return {"orders": []}

    backend = RunnerBackend([_LOADING, _LOADING, _LOADING])
    fake_http(backend)
    with pytest.raises(ATLTimeoutError):
        _run(AgentRunner(_client(), Agent()), max_wait_seconds=0)


class _RecordingClient:
    """Captures each sleep instead of performing it."""

    def __init__(self) -> None:
        self.sleeps = []

    def wait(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def test_wait_for_poll_spends_the_idle_budget_and_then_times_out():
    """The cumulative idle time must advance on every poll.

    ``AgentRunner`` and the TradingAgents replay runner share this helper, so a
    regression that stopped accumulating would disable the timeout on *both*
    surfaces and leave a stuck run polling forever. ``max_wait_seconds=0``
    (covered above) raises on the first call and never reaches this path.
    """
    client = _RecordingClient()
    waited = 0.0
    for _ in range(3):
        waited = wait_for_poll(client, 2.0, waited, 5.0, "loading")

    # The final poll is clamped so the run never sleeps past its budget.
    assert client.sleeps == [2.0, 2.0, 1.0]
    assert waited == 5.0

    with pytest.raises(ATLTimeoutError):
        wait_for_poll(client, 2.0, waited, 5.0, "loading")
    assert client.sleeps == [2.0, 2.0, 1.0]  # the timeout must not sleep again


def test_wait_for_poll_without_a_budget_never_times_out():
    client = _RecordingClient()
    waited = 0.0
    for _ in range(4):
        waited = wait_for_poll(client, 1.5, waited, None, "loading")
    assert client.sleeps == [1.5, 1.5, 1.5, 1.5]
    assert waited == 6.0


def test_runner_unexpected_status_raises(fake_http):
    class Agent:
        def decide(self, observation):
            return {"orders": []}

    from agentictrading import ATLAPIError

    backend = RunnerBackend([{"status": "weird", "run_id": "run_1"}])
    fake_http(backend)
    with pytest.raises(ATLAPIError):
        _run(AgentRunner(_client(), Agent()))


def test_runner_requires_decide():
    class NotAnAgent:
        pass

    with pytest.raises(TypeError):
        AgentRunner(_client(), NotAnAgent())


# ----------------------------------------------------------------------
# H5 — SDK runner survives the per-step decision deadline
# ----------------------------------------------------------------------


class _HoldAgent:
    def __init__(self):
        self.exec_results = []

    def decide(self, observation):
        return {"orders": [], "rationale": "hold"}

    def on_execution_result(self, result):
        self.exec_results.append(result)


def _err(code, message="err"):
    return {"detail": {"protocol_version": "1.0", "error": {"code": code, "message": message}}}


class ConflictOnDecideBackend:
    """Returns a 409 (with a configurable code) on the FIRST decision submit."""

    def __init__(self, step_sequence, code):
        self.step_sequence = list(step_sequence)
        self.code = code
        self.submitted = 0

    def __call__(self, req):
        url = req.full_url
        method = req.get_method()
        if method == "POST" and url.endswith("/api/v1/runs"):
            return (200, {"run_id": "run_1", "status": "created"})
        if url.endswith("/steps/next"):
            return (200, self.step_sequence.pop(0))
        if url.endswith("/decision"):
            self.submitted += 1
            return (409, _err(self.code))
        if url.endswith("/result"):
            return (200, {"run_id": "run_1", "status": "completed", "metrics": {"total_return": 0.0}})
        raise AssertionError(f"unexpected url {url}")


def test_runner_deadline_exceeded_advances_not_aborts(fake_http):
    """A slow decision (409 decision_deadline_exceeded) must NOT abort the run —
    the step auto-held server-side, so the runner advances and completes."""
    backend = ConflictOnDecideBackend([_AWAITING, _COMPLETED], code="decision_deadline_exceeded")
    fake_http(backend)
    agent = _HoldAgent()
    result = _run(AgentRunner(_client(), agent))

    assert isinstance(result, RunResult)
    assert result.metrics["total_return"] == 0.0
    assert backend.submitted == 1
    assert agent.exec_results == []  # no execution result for the auto-held step


def test_runner_step_already_finalized_advances(fake_http):
    backend = ConflictOnDecideBackend([_AWAITING, _COMPLETED], code="step_already_finalized")
    fake_http(backend)
    result = _run(AgentRunner(_client(), _HoldAgent()))
    assert isinstance(result, RunResult)


def test_runner_other_conflict_still_raises_with_run_id(fake_http):
    """A conflict that is NOT an auto-hold (e.g. run_completed) still raises, and
    the run id is attached to the error."""
    from agentictrading import ATLConflictError

    backend = ConflictOnDecideBackend([_AWAITING, _COMPLETED], code="run_completed")
    fake_http(backend)
    with pytest.raises(ATLConflictError) as ei:
        _run(AgentRunner(_client(), _HoldAgent()))
    assert ei.value.code == "run_completed"
    assert ei.value.run_id == "run_1"


class MaxStepsBackend:
    """Always awaiting; /result 409s (run not finalized), /metrics works."""

    def __init__(self):
        self.result_calls = 0

    def __call__(self, req):
        url = req.full_url
        method = req.get_method()
        if method == "POST" and url.endswith("/api/v1/runs"):
            return (200, {"run_id": "run_1", "status": "created"})
        if url.endswith("/steps/next"):
            return (200, _AWAITING)
        if url.endswith("/decision"):
            return (200, {"run_id": "run_1", "step_id": "step_1", "accepted": True,
                          "validation": {"passed": True, "rejections": []}, "fills": [],
                          "portfolio_after": {}, "run_status": "running"})
        if url.endswith("/metrics"):
            return (200, {"run_id": "run_1", "status": "running", "metrics": {"total_return": 0.01}})
        if url.endswith("/result"):
            self.result_calls += 1
            return (409, _err("run_not_completed", "not done"))
        raise AssertionError(f"unexpected url {url}")


def test_runner_max_steps_returns_metrics_not_409(fake_http):
    """Stopping via max_steps returns metrics-so-far, not a /result 409."""
    backend = MaxStepsBackend()
    fake_http(backend)
    result = _run(AgentRunner(_client(), _HoldAgent()), max_steps=1)

    assert isinstance(result, RunResult)
    assert result.status == "running"
    assert result.metrics["total_return"] == 0.01
    assert backend.result_calls == 0  # never asked the backend for a final result


def test_runner_rejects_nonpositive_poll_interval():
    runner = AgentRunner(_client(), _HoldAgent())
    for bad in (0, -1, -0.5):
        with pytest.raises(ValueError):
            runner.run_backtest(
                "agv_1",
                environment_id="us-equity-hourly-v1",
                start_date="2026-04-15",
                end_date="2026-04-16",
                poll_interval=bad,
            )


def test_runner_attaches_run_id_to_backend_errors(fake_http):
    from agentictrading import ATLAPIError

    backend = RunnerBackend([{"status": "weird", "run_id": "run_1"}])
    fake_http(backend)
    with pytest.raises(ATLAPIError) as ei:
        _run(AgentRunner(_client(), _HoldAgent()))
    assert ei.value.run_id == "run_1"
