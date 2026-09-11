"""$0 is a legal amount of simulated backtest capital.

The Configure screen's Allocated Capital card holds two fields, and until now
they disagreed about zero: ``cash_allocation`` (the paper-trading sleeve)
accepted ``$0`` at every layer, while ``backtest_allocation`` (simulated money)
carried a ``$1`` floor at *seven* independent places -- the ``min`` attribute,
the editor's own validator, two ``Field(ge=...)`` bounds, ``/backtest/run``'s
explicit 422, the protocol run service's 400, and ``resolve_initial_capital``,
which silently rewrote 0 to the $1,000 default rather than refusing it. Users
read that split as a bug in the card, which is what it looks like: two boxes
side by side, one takes 0 and the other will not say why it won't.

**Why the floor was load-bearing, and what replaced it.** Return is computed as
``(final_eq - initial_capital) / initial_capital`` at five sites (four in
``engine.py``, one in ``external_run_service.py``), none of them guarded. The
``$1`` floor was the only thing standing between a user's typed 0 and a
``ZeroDivisionError`` partway through a run. Lifting the floor therefore had to
come with ``fractional_return()``, which owns that arithmetic in one place --
otherwise the 422 would simply have been traded for a 500.

A $0 run is a real, if degenerate, backtest: no cash means no fills, so every
step holds, the curve is flat at $0, and the honest return is 0.00%.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.domain.backtesting.constants import (
    INITIAL_CAPITAL,
    MIN_BACKTEST_INITIAL_CAPITAL,
    fractional_return,
    resolve_initial_capital,
)


def _headers():
    session = str(uuid.uuid4())
    return {"X-Session-Id": session, "X-Browser-Id": session}


# --------------------------------------------------------------------------
# The constant and the two functions that read it
# --------------------------------------------------------------------------


def test_the_backtest_capital_floor_is_zero():
    assert MIN_BACKTEST_INITIAL_CAPITAL == 0


def test_zero_survives_resolution_instead_of_becoming_the_default():
    """The one that made this a *silent* bug rather than a loud one.

    ``if value <= 0: return INITIAL_CAPITAL`` meant a 0 that got past the API
    came back as $1,000 -- the field would have accepted the number and the run
    would have used a different one, with nothing anywhere saying so.
    """
    assert resolve_initial_capital(0) == 0.0
    assert resolve_initial_capital("0") == 0.0
    assert resolve_initial_capital(0.0) == 0.0


@pytest.mark.parametrize("absent", [None, "", "abc", object()])
def test_absent_or_unparseable_capital_still_falls_back_to_the_default(absent):
    """Zero is a value; absent is not. Keeping them apart is the whole fix."""
    assert resolve_initial_capital(absent) == float(INITIAL_CAPITAL)


def test_negative_capital_is_still_treated_as_invalid():
    """Nothing asked for negative money; it stays a fallback, not a balance."""
    assert resolve_initial_capital(-1) == float(INITIAL_CAPITAL)
    assert resolve_initial_capital(-3_000) == float(INITIAL_CAPITAL)


def test_fractional_return_reports_zero_rather_than_dividing_by_zero():
    assert fractional_return(0.0, 0.0) == 0.0
    # A $0 run cannot produce equity, but a caller must not explode if it does.
    assert fractional_return(5.0, 0.0) == 0.0


def test_fractional_return_is_unchanged_for_every_funded_run():
    assert fractional_return(1_100.0, 1_000.0) == pytest.approx(0.1)
    assert fractional_return(900.0, 1_000.0) == pytest.approx(-0.1)
    assert fractional_return(1_000.0, 1_000.0) == 0.0


# --------------------------------------------------------------------------
# The agents API: storing $0 on the agent
# --------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    import dashboard.backend.domain.agents.repository as agent_store_module
    import dashboard.backend.api.routers.agents as agents_api
    import dashboard.backend.database as db_module

    db_path = tmp_path / "test.db"
    test_agents = agent_store_module.AgentStore(db_path=db_path)
    test_db = db_module.BacktestDatabase(db_path=db_path)
    monkeypatch.setattr(agent_store_module, "agent_store", test_agents)
    monkeypatch.setattr(agents_api.agent_service, "agents", test_agents)
    monkeypatch.setattr(agents_api.agent_service, "db", test_db)
    monkeypatch.setattr(db_module, "db", test_db)
    return TestClient(app)


def test_create_accepts_zero_backtest_allocation(client):
    resp = client.post(
        "/api/v1/agents",
        json={"name": "broke", "backtest_allocation": 0},
        headers=_headers(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["agent"]["backtest_allocation"] == 0


def test_patch_accepts_zero_backtest_allocation(client):
    headers = _headers()
    created = client.post(
        "/api/v1/agents",
        json={"name": "funded", "backtest_allocation": 2_000},
        headers=headers,
    ).json()["agent"]

    resp = client.patch(
        f"/api/v1/agents/{created['agent_id']}",
        json={"backtest_allocation": 0},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["agent"]["backtest_allocation"] == 0


def test_zero_reads_back_as_zero_and_not_as_unset(client):
    """``0`` and ``None`` are different states and must survive as different.

    A NULL column means "never configured" and falls back to the default; a
    stored 0 means the owner chose 0. An ``or``-chain anywhere on this path
    collapses them, and the collapse is invisible -- the run simply uses
    $1,000.
    """
    headers = _headers()
    zero = client.post(
        "/api/v1/agents",
        json={"name": "zero", "backtest_allocation": 0},
        headers=headers,
    ).json()["agent"]
    unset = client.post(
        "/api/v1/agents", json={"name": "unset"}, headers=headers
    ).json()["agent"]

    assert zero["backtest_allocation"] == 0
    assert unset["backtest_allocation"] is None

    reread = client.get(
        f"/api/v1/agents/{zero['agent_id']}", headers=headers
    ).json()
    agent = reread.get("agent", reread)
    assert agent["backtest_allocation"] == 0


def test_negative_backtest_allocation_is_still_rejected(client):
    resp = client.post(
        "/api/v1/agents",
        json={"name": "negative", "backtest_allocation": -1},
        headers=_headers(),
    )
    assert resp.status_code == 422


# --------------------------------------------------------------------------
# The chart routes: a stored 0 must not be read as a missing value
# --------------------------------------------------------------------------


def test_a_stored_zero_scales_the_baselines_to_zero_not_to_a_thousand():
    """``initial_equity or first_equity or 1_000`` reads zero as missing.

    That chain was correct right up until zero became reachable. With a $0 run
    it scaled DJIA and buy-and-hold to $1,000 and drew them a thousand times
    above an agent curve sitting flat on the axis -- a chart that says the
    agent was wiped out when in fact it was never funded.
    """
    from dashboard.backend.api.routers.backtests import _run_initial_capital

    assert _run_initial_capital({"initial_equity": 0}, 0) == 0.0
    assert _run_initial_capital({"initial_equity": 0.0}, 500.0) == 0.0


def test_a_missing_stored_capital_still_falls_back():
    from dashboard.backend.api.routers.backtests import _run_initial_capital

    assert _run_initial_capital({}, 2_500.0) == 2_500.0
    assert _run_initial_capital({"initial_equity": None}, 2_500.0) == 2_500.0
    assert _run_initial_capital({}, None) == float(INITIAL_CAPITAL)


def test_a_stored_nan_does_not_poison_every_scaled_point():
    """NaN is a float and is not None, so it clears both of the other guards."""
    from dashboard.backend.api.routers.backtests import _run_initial_capital

    assert _run_initial_capital({"initial_equity": float("nan")}, 900.0) == 900.0
    assert _run_initial_capital(
        {"initial_equity": float("nan")}, float("nan")
    ) == float(INITIAL_CAPITAL)


# --------------------------------------------------------------------------
# The engine arithmetic the floor used to be standing in front of
# --------------------------------------------------------------------------


def test_the_engine_computes_a_zero_capital_return_without_raising():
    """Guards the actual call sites, not just the helper.

    ``fractional_return`` passing its own unit tests proves nothing about
    whether ``engine.py`` still divides directly -- this asserts the shipped
    expression, so re-inlining the division fails here rather than in prod.
    """
    import inspect

    from dashboard.backend.domain.backtesting import (
        engine as engine_module,
        external_run_service as ext_module,
    )

    for module in (engine_module, ext_module):
        source = inspect.getsource(module)
        assert "/ self.initial_capital" not in source, (
            f"{module.__name__} divides by initial_capital directly; a $0 run "
            "raises ZeroDivisionError. Use fractional_return()."
        )
        assert "fractional_return(" in source


# --------------------------------------------------------------------------
# The two route gates that carried their own hand-written "> 0"
# --------------------------------------------------------------------------


def test_backtest_run_rejects_negative_capital_and_names_the_floor():
    """The refusal has to move with the constant, not restate a literal.

    ``/backtest/run`` validated ``initial_capital <= 0`` inline, independently
    of ``MIN_BACKTEST_INITIAL_CAPITAL``. A floor duplicated as a literal is a
    floor that stops moving when the constant does -- which is how one field
    ended up with seven of them disagreeing.
    """
    resp = TestClient(app).post(
        "/backtest/run", json={"initial_capital": -1}, headers=_headers()
    )
    assert resp.status_code == 422
    assert "negative" in resp.json()["detail"]
    assert f"{MIN_BACKTEST_INITIAL_CAPITAL:g}" in resp.json()["detail"]


def test_neither_route_gate_hardcodes_a_positive_floor_any_more():
    import inspect

    from dashboard.backend.api.routers import backtests as backtests_module
    from dashboard.backend.domain.runs import service as runs_service

    for module, field in (
        (backtests_module, "initial_capital"),
        (runs_service, "requested"),
    ):
        source = inspect.getsource(module)
        assert f"{field} <= 0" not in source, (
            f"{module.__name__} still refuses $0 with its own literal floor."
        )
        assert "MIN_BACKTEST_INITIAL_CAPITAL" in source
