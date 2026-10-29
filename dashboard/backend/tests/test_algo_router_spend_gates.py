"""/api/algo/chat and /execute spend the platform Anthropic key: gate and bound them."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

import dashboard.backend.domain.backtesting.algo_service as algo_service
from dashboard.backend.api.auth import get_current_user
from dashboard.backend.app import app

BLOCKS = {
    "info_retrieval": "watch earnings",
    "signal_transfer": "score sentiment",
    "trading_algorithm": "buy on positive",
    "stop_loss_take_profit": "5% stop",
}
HEADERS = {"X-Session-Id": "algo-spend-gate-session"}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def signed_in(client):
    app.dependency_overrides[get_current_user] = lambda: {"id": 1, "email": "t@example.com"}
    yield client
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def no_spawn(monkeypatch):
    """Fail loudly if a request ever reaches the subprocess launcher."""
    monkeypatch.setattr(
        algo_service,
        "execute_algo",
        lambda *a, **k: pytest.fail("execute_algo must not be reached"),
    )
    import dashboard.backend.api.routers.algo as algo_router

    monkeypatch.setattr(
        algo_router, "execute_algo", lambda *a, **k: pytest.fail("execute_algo must not be reached")
    )
    monkeypatch.setattr(
        algo_router, "process_chat", lambda *a, **k: pytest.fail("process_chat must not be reached")
    )


@pytest.mark.parametrize("path, body", [
    ("/api/algo/chat", {"message": "hi"}),
    ("/api/algo/execute", {"blocks": BLOCKS}),
])
def test_anonymous_caller_is_refused(client, no_spawn, path, body):
    assert client.post(path, json=body, headers=HEADERS).status_code == 401


def test_oversized_chat_message_is_refused(signed_in, no_spawn):
    resp = signed_in.post("/api/algo/chat", json={"message": "a" * 4001}, headers=HEADERS)
    assert resp.status_code == 422


def test_oversized_block_is_refused(signed_in, no_spawn):
    blocks = dict(BLOCKS, trading_algorithm="x" * 5001)
    resp = signed_in.post("/api/algo/execute", json={"blocks": blocks}, headers=HEADERS)
    assert resp.status_code == 422


@pytest.mark.parametrize("start, end", [
    ("2020-01-01", "2026-06-01"),   # years, not days
    ("2025-03-10", "2025-03-01"),   # inverted
    ("2025-13-01", "2025-13-05"),   # not a date
    ("2025-03-01", None),           # half a window
])
def test_bad_window_is_refused_before_spawn(signed_in, no_spawn, start, end):
    body = {"blocks": BLOCKS, "start_date": start, "end_date": end}
    assert signed_in.post("/api/algo/execute", json=body, headers=HEADERS).status_code == 422


def test_second_run_is_refused_whatever_the_session(monkeypatch):
    """The concurrency check was per X-Session-Id, which the caller chooses."""
    monkeypatch.setattr(algo_service, "algo_status", dict(algo_service.algo_status, running=True,
                                                          session_id="someone-else"))
    with pytest.raises(RuntimeError, match="already running"):
        algo_service.execute_algo(BLOCKS, "a-fresh-session")


def test_concurrent_executes_start_one_run(monkeypatch, tmp_path):
    """The route is a sync def, so two POSTs run on two threadpool threads.

    The ``running`` check and its claim sat ~35 lines apart with no lock, so
    both requests could pass the check and each start a subprocess on the
    platform key. ``_default_backtest_dates`` runs between the two; holding
    the first caller there until the second arrives opens that window on
    every run instead of by luck.
    """
    monkeypatch.setattr(algo_service, "algo_status", dict(algo_service.algo_status, running=False))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(algo_service, "_has_alpaca_credentials", lambda: True)
    monkeypatch.setattr(algo_service, "_ensure_data_dir", lambda: None)
    monkeypatch.setattr(algo_service, "CONFIG_DIR", tmp_path)
    launched = []
    monkeypatch.setattr(algo_service, "_run_backtest_background", lambda *args: launched.append(args))

    arrivals = []
    second_arrived = threading.Event()

    def held_dates():
        arrivals.append(None)
        if len(arrivals) == 2:
            second_arrived.set()
        second_arrived.wait(timeout=0.5)
        return "2025-03-03", "2025-03-07"

    monkeypatch.setattr(algo_service, "_default_backtest_dates", held_dates)

    outcomes = []

    def post():
        try:
            algo_service.execute_algo(BLOCKS, "a-fresh-session")
            outcomes.append("started")
        except RuntimeError as exc:
            outcomes.append(str(exc))

    callers = [threading.Thread(target=post) for _ in range(2)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=5)

    deadline = time.monotonic() + 2
    while len(launched) < outcomes.count("started") and time.monotonic() < deadline:
        time.sleep(0.01)
    assert outcomes.count("started") == 1, outcomes
    assert len(launched) == 1
    assert any("already running" in outcome for outcome in outcomes), outcomes
