"""Portfolio allocate / reclaim / delete-returns (#175)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.tests.auth_cookies_helpers import _cookie_session_token

# These are imported as *modules*, not `from ... import <name>`, because the
# fixture below monkeypatches attributes on them and the tests must read those
# attributes at call time to see the patched value. Importing the classes
# directly as well would bind two names to one module (CodeQL
# py/import-and-import-from) and make it easy to grab a stale, unpatched store.
import dashboard.backend.domain.backtesting.constants as backtesting_constants
import dashboard.backend.domain.agents.repository as agent_repo
import dashboard.backend.domain.agents.service as agent_service_module
import dashboard.backend.domain.portfolios.repository as portfolio_repo
import dashboard.backend.domain.portfolios.service as portfolio_service_module
import dashboard.backend.users as users_module
from dashboard.backend.app import app
from dashboard.backend.domain.agents.defaults import STARTER_AGENTS

# Plain values read off the module (it is imported whole for the
# PAPER_TRADING_ENABLED patch below; a second `from` import of it is CodeQL
# py/import-and-import-from). None of these three is ever patched.
DEFAULT_AGENT_CASH_ALLOCATION = backtesting_constants.DEFAULT_AGENT_CASH_ALLOCATION
DEFAULT_PORTFOLIO_EQUITY = backtesting_constants.DEFAULT_PORTFOLIO_EQUITY
MAX_AGENT_CASH_ALLOCATION = backtesting_constants.MAX_AGENT_CASH_ALLOCATION

# Signup mints one starter per STARTER_AGENTS entry, each funded at
# DEFAULT_AGENT_CASH_ALLOCATION. Cash checks after _signup() subtract this,
# or they describe an empty-account world that no longer exists.
STARTER_ALLOCATED = len(STARTER_AGENTS) * float(DEFAULT_AGENT_CASH_ALLOCATION)


def _cash_after_signup() -> float:
    return float(DEFAULT_PORTFOLIO_EQUITY) - STARTER_ALLOCATED


@pytest.fixture(autouse=True)
def paper_trading_on(monkeypatch):
    """These tests pin the ledger that paper trading draws on, so they run with
    it switched on: while it is off, nothing picks a non-zero sleeve on the
    caller's behalf (``new_agent_cash_allocation``) and every default below
    would be $0."""
    monkeypatch.setattr(backtesting_constants, "PAPER_TRADING_ENABLED", True)


@pytest.fixture
def env(monkeypatch):
    """Auth + portfolio + agents share one content DB (ledger ↔ sleeve)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        user_store = users_module.UserStore(db_path=root / "users.db")
        content_db = root / "content.db"
        portfolio_store = portfolio_repo.PortfolioStore(db_path=content_db)
        agent_store = agent_repo.AgentStore(db_path=content_db)

        monkeypatch.setattr(users_module, "user_store", user_store)
        monkeypatch.setattr(portfolio_repo, "portfolio_store", portfolio_store)
        monkeypatch.setattr(portfolio_service_module, "portfolio_store", portfolio_store)
        monkeypatch.setattr(agent_repo, "agent_store", agent_store)
        monkeypatch.setattr(agent_service_module.agent_service, "agents", agent_store)
        yield TestClient(app), agent_store, portfolio_store


@pytest.fixture
def client(env) -> TestClient:
    return env[0]


def _signup(client: TestClient, email: str = "alloc@example.com") -> tuple[str, dict]:
    resp = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "display_name": "Alloc User",
            "password": "securepass1",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "token" not in data
    return _cookie_session_token(client), data["user"]


def _auth(token: str, browser: str = "browser-alloc-1") -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "X-Browser-Id": browser,
        "X-Session-Id": browser,
    }


def test_create_agent_debits_portfolio(client):
    token, _ = _signup(client)
    headers = _auth(token)

    before = client.get("/api/v1/portfolio", headers=headers).json()["portfolio"]
    assert before["cash_available"] == _cash_after_signup()
    assert before["allocated"] == STARTER_ALLOCATED

    created = client.post(
        "/api/v1/agents",
        headers=headers,
        json={
            "name": "Funded",
            "model_name": "local-model",
            "agent_type": "builtin",
            "cash_allocation": 2500,
        },
    )
    assert created.status_code == 200, created.text
    agent = created.json()["agent"]
    assert agent["cash_allocation"] == 2500

    after = client.get("/api/v1/portfolio", headers=headers).json()["portfolio"]
    assert after["cash_available"] == _cash_after_signup() - 2500
    assert after["allocated"] == STARTER_ALLOCATED + 2500


def test_allocate_and_reclaim_endpoints(client):
    token, _ = _signup(client, email="xfer@example.com")
    headers = _auth(token)

    created = client.post(
        "/api/v1/agents",
        headers=headers,
        json={
            "name": "Xfer",
            "model_name": "local-model",
            "agent_type": "builtin",
            "cash_allocation": 0,
        },
    )
    assert created.status_code == 200, created.text
    agent_id = created.json()["agent"]["agent_id"]

    alloc = client.post(
        "/api/v1/portfolio/allocate",
        headers=headers,
        json={"agent_id": agent_id, "amount": 1500},
    )
    assert alloc.status_code == 200, alloc.text
    assert alloc.json()["portfolio"]["cash_available"] == _cash_after_signup() - 1500
    assert alloc.json()["agent"]["cash_allocation"] == 1500

    reclaim = client.post(
        "/api/v1/portfolio/reclaim",
        headers=headers,
        json={"agent_id": agent_id, "amount": 500},
    )
    assert reclaim.status_code == 200, reclaim.text
    assert reclaim.json()["portfolio"]["cash_available"] == _cash_after_signup() - 1000
    assert reclaim.json()["agent"]["cash_allocation"] == 1000


def test_allocate_blocked_when_insufficient_cash(client):
    token, _ = _signup(client, email="poor@example.com")
    headers = _auth(token)

    # Starters already hold $3k. Two more maxed sleeves leave $1k, so a $3k
    # allocate is within the per-agent max but over remaining cash.
    for i in range(2):
        filled = client.post(
            "/api/v1/agents",
            headers=headers,
            json={
                "name": f"Drain{i}",
                "model_name": "local-model",
                "cash_allocation": float(MAX_AGENT_CASH_ALLOCATION),
            },
        )
        assert filled.status_code == 200, filled.text

    created = client.post(
        "/api/v1/agents",
        headers=headers,
        json={"name": "A", "model_name": "local-model", "cash_allocation": 0},
    )
    agent_id = created.json()["agent"]["agent_id"]

    # Remaining unallocated is $1,000; request is within the per-agent max but
    # still more than cash available → business-rule 400 (not pydantic 422).
    too_much = client.post(
        "/api/v1/portfolio/allocate",
        headers=headers,
        json={"agent_id": agent_id, "amount": float(MAX_AGENT_CASH_ALLOCATION)},
    )
    assert too_much.status_code == 400


def test_delete_agent_returns_funds(client):
    token, _ = _signup(client, email="del@example.com")
    headers = _auth(token)
    created = client.post(
        "/api/v1/agents",
        headers=headers,
        json={
            "name": "Temp",
            "model_name": "local-model",
            "agent_type": "builtin",
            "cash_allocation": float(DEFAULT_AGENT_CASH_ALLOCATION),
        },
    )
    assert created.status_code == 200, created.text
    agent_id = created.json()["agent"]["agent_id"]

    mid = client.get("/api/v1/portfolio", headers=headers).json()["portfolio"]
    assert mid["cash_available"] == _cash_after_signup() - float(
        DEFAULT_AGENT_CASH_ALLOCATION
    )

    deleted = client.delete(f"/api/v1/agents/{agent_id}", headers=headers)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["reclaimed"] == float(DEFAULT_AGENT_CASH_ALLOCATION)

    after = client.get("/api/v1/portfolio", headers=headers).json()["portfolio"]
    assert after["cash_available"] == _cash_after_signup()
    assert after["allocated"] == STARTER_ALLOCATED


def test_patch_cash_allocation_moves_ledger(client):
    token, _ = _signup(client, email="patch@example.com")
    headers = _auth(token)
    created = client.post(
        "/api/v1/agents",
        headers=headers,
        json={"name": "P", "model_name": "local-model", "cash_allocation": 1000},
    )
    agent_id = created.json()["agent"]["agent_id"]

    up = client.patch(
        f"/api/v1/agents/{agent_id}",
        headers=headers,
        json={"cash_allocation": 3000},
    )
    assert up.status_code == 200, up.text
    assert up.json()["agent"]["cash_allocation"] == 3000
    pf = client.get("/api/v1/portfolio", headers=headers).json()["portfolio"]
    assert pf["cash_available"] == _cash_after_signup() - 3000

    down = client.patch(
        f"/api/v1/agents/{agent_id}",
        headers=headers,
        json={"cash_allocation": 500},
    )
    assert down.status_code == 200, down.text
    pf2 = client.get("/api/v1/portfolio", headers=headers).json()["portfolio"]
    assert pf2["cash_available"] == _cash_after_signup() - 500


# ---------------------------------------------------------------------------
# Ledger reconciliation: agent sleeves are the source of truth for "allocated".
#
# Agents predating #175 -- and guest agents later claimed by a signed-in user
# via ``reclaim_on_session_match`` -- hold a cash_allocation that never debited
# any ledger. Without reconciliation their owner's portfolio bootstraps at the
# full $10k, so the sleeve is counted twice and every later transfer is wrong.
# ---------------------------------------------------------------------------


def _preexisting_agent(agent_store, user_id: int, sleeve: float, name: str = "Legacy"):
    """An agent written straight to the store, exactly as pre-#175 code did."""
    return agent_store.create_agent(
        name=name,
        model_name="local-model",
        owner_user_id=user_id,
        owner_browser_session="browser-alloc-1",
        agent_type="builtin",
        cash_allocation=sleeve,
    )


def test_portfolio_reports_preexisting_sleeves_as_allocated(env):
    client, agent_store, _ = env
    token, user = _signup(client, email="legacy-get@example.com")
    _preexisting_agent(agent_store, user["id"], 1000.0)

    pf = client.get("/api/v1/portfolio", headers=_auth(token)).json()["portfolio"]

    assert pf["allocated"] == STARTER_ALLOCATED + 1000.0
    assert pf["cash_available"] == _cash_after_signup() - 1000.0


def test_deleting_a_preexisting_agent_succeeds_and_restores_cash(env):
    client, agent_store, _ = env
    token, user = _signup(client, email="legacy-del@example.com")
    headers = _auth(token)
    agent_id = _preexisting_agent(agent_store, user["id"], 1000.0)["agent_id"]

    deleted = client.delete(f"/api/v1/agents/{agent_id}", headers=headers)

    assert deleted.status_code == 200, deleted.text
    assert agent_store.get_agent(agent_id) is None
    after = client.get("/api/v1/portfolio", headers=headers).json()["portfolio"]
    assert after["cash_available"] == _cash_after_signup()
    assert after["allocated"] == STARTER_ALLOCATED


def test_preexisting_agent_allocation_can_be_lowered(env):
    client, agent_store, _ = env
    token, user = _signup(client, email="legacy-patch@example.com")
    headers = _auth(token)
    agent_id = _preexisting_agent(agent_store, user["id"], 1000.0)["agent_id"]

    resp = client.patch(
        f"/api/v1/agents/{agent_id}", headers=headers, json={"cash_allocation": 400}
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["agent"]["cash_allocation"] == 400
    pf = client.get("/api/v1/portfolio", headers=headers).json()["portfolio"]
    assert pf["cash_available"] == _cash_after_signup() - 400


def test_preexisting_sleeve_cannot_be_spent_twice(env):
    """Cash already in a legacy sleeve is not also available to allocate."""
    client, agent_store, _ = env
    token, user = _signup(client, email="legacy-double@example.com")
    headers = _auth(token)
    # Written straight to the store, as pre-#175 code (and the former $1M cap)
    # allowed — legacy sleeves may exceed today's per-agent max.
    _preexisting_agent(agent_store, user["id"], 8000.0)
    target = _preexisting_agent(agent_store, user["id"], 0.0, name="Target")

    # $2k remains. A cap-sized request passes validation but must fail the
    # cash check — unless the legacy sleeve was double-counted as available.
    resp = client.post(
        "/api/v1/portfolio/allocate",
        headers=headers,
        json={"agent_id": target["agent_id"], "amount": float(MAX_AGENT_CASH_ALLOCATION)},
    )

    assert resp.status_code == 400, resp.text


def test_creating_an_agent_cannot_overspend_preexisting_sleeves(env):
    client, agent_store, _ = env
    token, user = _signup(client, email="legacy-create@example.com")
    headers = _auth(token)
    _preexisting_agent(agent_store, user["id"], 9500.0)

    resp = client.post(
        "/api/v1/agents",
        headers=headers,
        json={"name": "TooBig", "model_name": "local-model", "cash_allocation": 1000},
    )

    assert resp.status_code == 400, resp.text


# ---------------------------------------------------------------------------
# A failed agent write must not move money.
# ---------------------------------------------------------------------------


def test_patch_failure_leaves_the_ledger_untouched(env):
    client, agent_store, _ = env
    token, _ = _signup(client, email="patchfail@example.com")
    headers = _auth(token)
    created = client.post(
        "/api/v1/agents",
        headers=headers,
        json={"name": "PF", "model_name": "local-model", "cash_allocation": 1000},
    )
    agent_id = created.json()["agent"]["agent_id"]

    def boom(*args, **kwargs):
        raise RuntimeError("simulated failure writing the non-cash fields")

    # Restore by hand rather than monkeypatch.undo(), which would also revert
    # the fixture's temp-store patches and send the assertions below at the
    # real databases.
    real_update = agent_service_module.agent_service.update_agent
    agent_service_module.agent_service.update_agent = boom
    try:
        with pytest.raises(RuntimeError):
            client.patch(
                f"/api/v1/agents/{agent_id}",
                headers=headers,
                json={"cash_allocation": 3000, "name": "Renamed"},
            )
    finally:
        agent_service_module.agent_service.update_agent = real_update

    assert agent_store.get_agent(agent_id)["cash_allocation"] == 1000
    pf = client.get("/api/v1/portfolio", headers=headers).json()["portfolio"]
    assert pf["cash_available"] == _cash_after_signup() - 1000


# ---------------------------------------------------------------------------
# Transfer endpoint auth / ownership / bounds.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["allocate", "reclaim"])
def test_transfer_endpoints_require_authentication(client, route):
    resp = client.post(
        f"/api/v1/portfolio/{route}", json={"agent_id": "agent_x", "amount": 100}
    )
    assert resp.status_code == 401


@pytest.mark.parametrize("route", ["allocate", "reclaim"])
def test_transfer_endpoints_reject_another_users_agent(client, route):
    owner_token, _ = _signup(client, email="owner@example.com")
    created = client.post(
        "/api/v1/agents",
        headers=_auth(owner_token, browser="browser-owner"),
        json={"name": "Owned", "model_name": "local-model", "cash_allocation": 1000},
    )
    agent_id = created.json()["agent"]["agent_id"]
    intruder_token, _ = _signup(client, email="intruder@example.com")

    resp = client.post(
        f"/api/v1/portfolio/{route}",
        headers=_auth(intruder_token, browser="browser-intruder"),
        json={"agent_id": agent_id, "amount": 100},
    )

    assert resp.status_code == 403, resp.text


def test_transfer_endpoints_404_on_unknown_agent(client):
    token, _ = _signup(client, email="ghost@example.com")
    resp = client.post(
        "/api/v1/portfolio/allocate",
        headers=_auth(token),
        json={"agent_id": "agent_does_not_exist", "amount": 100},
    )
    assert resp.status_code == 404


def test_reclaiming_more_than_the_sleeve_is_rejected(client):
    token, _ = _signup(client, email="overreclaim@example.com")
    headers = _auth(token)
    created = client.post(
        "/api/v1/agents",
        headers=headers,
        json={"name": "R", "model_name": "local-model", "cash_allocation": 500},
    )
    agent_id = created.json()["agent"]["agent_id"]

    resp = client.post(
        "/api/v1/portfolio/reclaim",
        headers=headers,
        json={"agent_id": agent_id, "amount": 501},
    )

    assert resp.status_code == 400, resp.text
    assert client.get("/api/v1/portfolio", headers=headers).json()["portfolio"][
        "cash_available"
    ] == _cash_after_signup() - 500


def test_concurrent_allocations_cannot_overspend(env):
    """Two simultaneous transfers must not both pass the cash check."""
    import threading

    client, agent_store, _ = env
    token, user = _signup(client, email="race@example.com")
    headers = _auth(token)
    # Starters hold $3k. Drain one more max sleeve so $4k remains: either
    # cap-sized request below fits on its own, but the two together overspend.
    drained = client.post(
        "/api/v1/agents",
        headers=headers,
        json={
            "name": "Drain",
            "model_name": "local-model",
            "cash_allocation": float(MAX_AGENT_CASH_ALLOCATION),
        },
    )
    assert drained.status_code == 200, drained.text
    agents = [
        client.post(
            "/api/v1/agents",
            headers=headers,
            json={"name": f"R{i}", "model_name": "local-model", "cash_allocation": 0},
        ).json()["agent"]["agent_id"]
        for i in range(2)
    ]

    barrier = threading.Barrier(2, timeout=10)
    results = []

    def transfer(agent_id):
        barrier.wait()
        results.append(
            client.post(
                "/api/v1/portfolio/allocate",
                headers=headers,
                json={"agent_id": agent_id, "amount": float(MAX_AGENT_CASH_ALLOCATION)},
            ).status_code
        )

    threads = [threading.Thread(target=transfer, args=(a,)) for a in agents]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == [200, 400], results
    total_sleeves = sum(
        float(a["cash_allocation"] or 0)
        for a in agent_store.list_agents(owner_user_id=user["id"])
    )
    assert total_sleeves == STARTER_ALLOCATED + 2 * float(MAX_AGENT_CASH_ALLOCATION)


def test_service_serialises_concurrent_allocations_for_one_user(env, monkeypatch):
    """The HTTP test above rides FastAPI's single event loop, which hides the
    check-then-write window entirely. Drive the service from real threads so
    the window is actually exercised.

    The sleeve write is slowed deliberately. Without it the whole
    check-and-write sequence fits inside one GIL switch interval (5ms by
    default), so the threads run to completion one after another and the test
    passes whether or not anything serialises them -- i.e. it would not catch
    the removal of the lock it exists to protect.
    """
    import threading
    import time

    # Read through the module so this picks up the fixture's patched store.
    real_store = agent_repo.agent_store
    InsufficientCashError = portfolio_service_module.InsufficientCashError
    portfolio_service = portfolio_service_module.portfolio_service

    client, agent_store, _ = env
    _, user = _signup(client, email="svc-race@example.com")
    uid = user["id"]
    # Starters hold $3k. Drain one more max sleeve so $4k remains: either
    # cap-sized allocation below fits on its own, but the two together overspend.
    _preexisting_agent(
        agent_store, uid, float(MAX_AGENT_CASH_ALLOCATION), name="Drain"
    )
    agents = [
        _preexisting_agent(agent_store, uid, 0.0, name=f"Race{i}") for i in range(2)
    ]

    inner_update = real_store.update_agent

    def slow_update(*args, **kwargs):
        # Stands in for a slow content DB (prod's is a remote Neon instance,
        # not a local file), widening the window to something a thread
        # scheduler will actually interleave.
        time.sleep(0.2)
        return inner_update(*args, **kwargs)

    monkeypatch.setattr(real_store, "update_agent", slow_update)

    barrier = threading.Barrier(2, timeout=10)
    rejected = []

    def allocate(agent):
        barrier.wait()
        try:
            portfolio_service.allocate_to_agent(
                owner_user_id=uid, agent=agent, amount=float(MAX_AGENT_CASH_ALLOCATION)
            )
        except InsufficientCashError:
            rejected.append(agent["agent_id"])

    threads = [threading.Thread(target=allocate, args=(a,)) for a in agents]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = sum(
        float(a["cash_allocation"] or 0)
        for a in agent_store.list_agents(owner_user_id=uid)
    )
    assert len(rejected) == 1, f"expected one rejection, got {rejected}"
    expected = STARTER_ALLOCATED + 2 * float(MAX_AGENT_CASH_ALLOCATION)
    assert total == expected, f"over-allocated: {total} against a 10000 account"


# ---------------------------------------------------------------------------
# Paper trading switched off (the shipped state). The dashboard hides My
# Portfolio then, so a sleeve the server picks by default is a reservation
# nobody can see or release -- and enough of them refuse a later create with
# "Insufficient unallocated cash". Every path that chooses a sleeve for the
# caller (signup starters, a create that omits one, marketplace clone,
# duplicate) reserves $0; an explicit value is still honoured.
# ---------------------------------------------------------------------------


def _switch_paper_trading_off(monkeypatch):
    # Runs in the test body, after the autouse paper_trading_on fixture.
    monkeypatch.setattr(backtesting_constants, "PAPER_TRADING_ENABLED", False)


def test_paper_trading_ships_switched_off():
    # Read the source, not the module attribute: the autouse fixture above has
    # already switched the flag on for this test.
    source = Path(backtesting_constants.__file__).read_text(encoding="utf-8")
    assert "\nPAPER_TRADING_ENABLED = False\n" in source


def test_signup_starters_reserve_nothing(client, monkeypatch):
    _switch_paper_trading_off(monkeypatch)
    token, _ = _signup(client, "paper-off-signup@example.com")
    headers = _auth(token)

    agents = client.get("/api/v1/agents", headers=headers).json()["agents"]
    assert agents, "signup should still provision the starter agents"
    assert all(float(agent["cash_allocation"] or 0) == 0 for agent in agents)

    portfolio = client.get("/api/v1/portfolio", headers=headers).json()["portfolio"]
    assert portfolio["allocated"] == 0
    assert portfolio["cash_available"] == float(DEFAULT_PORTFOLIO_EQUITY)


def test_create_without_a_sleeve_reserves_nothing(client, monkeypatch):
    _switch_paper_trading_off(monkeypatch)
    token, _ = _signup(client, "paper-off-create@example.com")
    headers = _auth(token)

    created = client.post(
        "/api/v1/agents",
        headers=headers,
        json={"name": "No sleeve", "model_name": "local-model", "agent_type": "builtin"},
    )
    assert created.status_code == 200, created.text
    assert created.json()["agent"]["cash_allocation"] == 0


def test_an_explicit_sleeve_is_still_honoured(client, monkeypatch):
    _switch_paper_trading_off(monkeypatch)
    token, _ = _signup(client, "paper-off-explicit@example.com")
    headers = _auth(token)

    created = client.post(
        "/api/v1/agents",
        headers=headers,
        json={
            "name": "Explicit",
            "model_name": "local-model",
            "agent_type": "builtin",
            "cash_allocation": 1500,
        },
    )
    assert created.status_code == 200, created.text
    assert created.json()["agent"]["cash_allocation"] == 1500
    portfolio = client.get("/api/v1/portfolio", headers=headers).json()["portfolio"]
    assert portfolio["allocated"] == 1500


def test_clone_and_duplicate_reserve_nothing_and_never_run_out(client, monkeypatch):
    """Twelve copies would have needed $12,000 of a $10,000 ledger at $1,000
    each -- the "Insufficient unallocated cash" refusal this change removes."""
    _switch_paper_trading_off(monkeypatch)
    token, _ = _signup(client, "paper-off-copies@example.com")
    headers = _auth(token)

    cloned = client.post(
        "/api/v1/agents/marketplace/claude-haiku-4-5/clone", headers=headers, json={}
    )
    assert cloned.status_code == 200, cloned.text
    source = cloned.json()["agent"]
    assert source["cash_allocation"] == 0

    for _ in range(12):
        copy = client.post(
            f"/api/v1/agents/{source['agent_id']}/duplicate",
            headers=headers,
            json={"model_name": "deepseek/deepseek-v4-pro"},
        )
        assert copy.status_code == 200, copy.text
        assert copy.json()["agent"]["cash_allocation"] == 0

    portfolio = client.get("/api/v1/portfolio", headers=headers).json()["portfolio"]
    assert portfolio["allocated"] == 0
