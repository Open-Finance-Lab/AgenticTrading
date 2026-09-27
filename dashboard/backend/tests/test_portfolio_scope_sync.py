"""The Capital Allocation panel must agree with the My Agents list.

``GET /api/v1/agents`` answers with the owner's agents **union** the unclaimed
guest agents of this browser (and the active trading-session agent). The
portfolio panel is drawn from that same list -- one legend row and one pie
slice per agent carrying a sleeve -- while ``allocated``/``cash_available``
used to be summed over the owner's agents *alone*.

The two therefore disagreed exactly when a guest agent had not been claimed
yet: a real dollar figure rendered in the legend that the "Allocated to
Agents" total did not include, with the difference offered back to the user as
unallocated cash they could spend twice.

These tests pin the union scope on the portfolio side; the one thing that must
NOT follow it (the persisted ``user_portfolios.cash_available`` cache, keyed by
account and so unable to depend on which browser asked); and the scope's own
limit -- it is built from caller-supplied headers, so it makes the panel's two
figures *agree* rather than bounding total allocation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.tests.auth_cookies_helpers import _cookie_session_token

import dashboard.backend.domain.backtesting.constants as backtesting_constants
import dashboard.backend.domain.agents.repository as agent_repo
import dashboard.backend.domain.agents.service as agent_service_module
import dashboard.backend.domain.portfolios.repository as portfolio_repo
import dashboard.backend.domain.portfolios.service as portfolio_service_module
import dashboard.backend.users as users_module
from dashboard.backend.app import app
from dashboard.backend.csrf import csrf_cookie_name

BROWSER = "browser-scope-sync"


@pytest.fixture(autouse=True)
def paper_trading_on(monkeypatch):
    """These tests pin the ledger that paper trading draws on, so they run with
    it switched on: while it is off, nothing picks a non-zero sleeve on the
    caller's behalf (``new_agent_cash_allocation``) and every default below
    would be $0."""
    monkeypatch.setattr(backtesting_constants, "PAPER_TRADING_ENABLED", True)


@pytest.fixture
def env(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        content_db = root / "content.db"
        user_store = users_module.UserStore(db_path=root / "users.db")
        portfolio_store = portfolio_repo.PortfolioStore(db_path=content_db)
        agent_store = agent_repo.AgentStore(db_path=content_db)

        monkeypatch.setattr(users_module, "user_store", user_store)
        monkeypatch.setattr(portfolio_repo, "portfolio_store", portfolio_store)
        monkeypatch.setattr(portfolio_service_module, "portfolio_store", portfolio_store)
        monkeypatch.setattr(agent_repo, "agent_store", agent_store)
        # agent_service binds its store at construction (default argument), so
        # patching the module attribute alone leaves the API on the real DB.
        monkeypatch.setattr(agent_service_module.agent_service, "agents", agent_store)
        yield TestClient(app), user_store, portfolio_store, agent_store


def _signup(client: TestClient, email: str = "scope@example.com") -> dict:
    resp = client.post(
        "/api/auth/signup",
        json={"email": email, "display_name": "Scope", "password": "securepass1"},
    )
    assert resp.status_code == 200, resp.text
    assert _cookie_session_token(client)
    return resp.json()


def _headers(client: TestClient, browser: str = BROWSER) -> dict:
    return {
        "X-Browser-Id": browser,
        "Origin": "http://testserver",
        "X-CSRF-Token": client.cookies.get(csrf_cookie_name()) or "",
    }


def _listed(client: TestClient, browser: str = BROWSER) -> list:
    resp = client.get("/api/v1/agents", headers=_headers(client, browser))
    assert resp.status_code == 200, resp.text
    return resp.json()["agents"]


def _portfolio(client: TestClient, browser: str = BROWSER) -> dict:
    resp = client.get("/api/v1/portfolio", headers=_headers(client, browser))
    assert resp.status_code == 200, resp.text
    return resp.json()["portfolio"]


def _legend_sum(agents) -> float:
    """What the pie/legend renders: one row per listed agent carrying a sleeve."""
    return sum(float(a.get("cash_allocation") or 0) for a in agents)


def test_unclaimed_browser_agent_counts_toward_allocated(env):
    """The legend's rows and the "Allocated to Agents" total must add up the same."""
    client, user_store, _portfolio_store, agent_store = env
    agent_store.create_agent(
        name="Guest sleeve",
        owner_user_id=None,
        owner_browser_session=BROWSER,
        agent_type="builtin",
        cash_allocation=2500.0,
    )
    _signup(client)

    listed = _listed(client)
    portfolio = _portfolio(client)

    assert any(a.get("owner_user_id") is None for a in listed), (
        "fixture no longer exercises the unclaimed-agent case"
    )
    assert portfolio["allocated"] == pytest.approx(_legend_sum(listed))


def test_unclaimed_sleeve_is_not_offered_back_as_spendable_cash(env):
    """cash_available follows the same scope: visibly committed money is not free."""
    client, _user_store, _portfolio_store, agent_store = env
    agent_store.create_agent(
        name="Guest sleeve",
        owner_user_id=None,
        owner_browser_session=BROWSER,
        agent_type="builtin",
        cash_allocation=2500.0,
    )
    _signup(client)

    portfolio = _portfolio(client)
    assert portfolio["cash_available"] == pytest.approx(
        max(portfolio["equity"] - portfolio["allocated"], 0.0)
    )


def test_unclaimed_total_is_reported_separately(env):
    """The drift is named in the payload rather than only folded into a total.

    An unclaimed sleeve riding a signed-in account's panel is a claim that has
    not happened yet, not a steady state. Reporting it keeps "nothing to claim"
    distinguishable from "the claim silently failed" -- the fail-closed-is-not-
    fail-visible rule this repo has been bitten by before.
    """
    client, _user_store, _portfolio_store, agent_store = env
    agent_store.create_agent(
        name="Guest sleeve",
        owner_user_id=None,
        owner_browser_session=BROWSER,
        agent_type="builtin",
        cash_allocation=2500.0,
    )
    _signup(client)

    portfolio = _portfolio(client)
    assert portfolio["unclaimed_allocated"] == pytest.approx(2500.0)


def test_account_with_no_guest_agents_reports_zero_unclaimed(env):
    client, _user_store, _portfolio_store, _agent_store = env
    _signup(client)
    portfolio = _portfolio(client)
    assert portfolio["unclaimed_allocated"] == pytest.approx(0.0)
    assert portfolio["allocated"] > 0  # starter agents provisioned at signup


def test_persisted_cash_cache_stays_account_scoped(env):
    """The stored row must not flap between browsers of the same account.

    ``user_portfolios`` is keyed by user, so a cache that followed the visible
    scope would be rewritten on every read from a browser holding a different
    set of guest agents -- two browsers writing over each other forever.
    """
    client, user_store, portfolio_store, agent_store = env
    agent_store.create_agent(
        name="Guest sleeve",
        owner_user_id=None,
        owner_browser_session=BROWSER,
        agent_type="builtin",
        cash_allocation=2500.0,
    )
    _signup(client)
    user = user_store.get_user_by_email("scope@example.com")

    scoped = _portfolio(client)
    stored_after_scoped = portfolio_store.get_or_create(user["id"])["cash_available"]

    # A second browser sees none of the guest agents.
    plain = _portfolio(client, browser="browser-other")
    stored_after_plain = portfolio_store.get_or_create(user["id"])["cash_available"]

    assert stored_after_scoped == pytest.approx(stored_after_plain)
    assert stored_after_plain == pytest.approx(
        max(plain["equity"] - plain["allocated"], 0.0)
    )
    assert scoped["allocated"] > plain["allocated"]


def test_allocation_from_the_browser_cannot_spend_an_unclaimed_sleeve_twice(env):
    """The validation path uses the same scope the panel displays.

    Scoped to the *browser*, deliberately: the scope comes from caller-supplied
    headers, so this pins that a browser seeing the sleeve in its own legend is
    validated against it -- not that no client anywhere can allocate past it. A
    caller sending no browser identity still gets the account-only figure; see
    ``api/routers/portfolio._scope`` for why that is the right answer there and
    what catches the over-allocation instead.
    """
    client, user_store, _portfolio_store, agent_store = env
    agent_store.create_agent(
        name="Guest sleeve",
        owner_user_id=None,
        owner_browser_session=BROWSER,
        agent_type="builtin",
        cash_allocation=9000.0,
    )
    _signup(client)
    user = user_store.get_user_by_email("scope@example.com")
    target = agent_store.create_agent(
        name="Target",
        owner_user_id=user["id"],
        owner_browser_session=BROWSER,
        agent_type="builtin",
        cash_allocation=0.0,
    )

    portfolio = _portfolio(client)
    # Starter agents (3 x 1000) plus the 9000 guest sleeve already exceed equity.
    assert portfolio["cash_available"] == pytest.approx(0.0)

    resp = client.post(
        "/api/v1/portfolio/allocate",
        json={"agent_id": target["agent_id"], "amount": 500.0},
        headers=_headers(client),
    )
    assert resp.status_code == 400, resp.text
    assert "insufficient" in resp.json()["detail"].lower()


def test_a_header_less_caller_is_validated_against_the_account_alone(env):
    """The documented limit of the scope, pinned rather than only described.

    ``AgentScope`` is built from caller-supplied headers, so a client that
    sends none -- the SDK, curl, a cron -- is validated against the owner's
    sleeves alone, exactly as before the scope existed. That is the right
    answer for such a caller (an unclaimed guest agent belongs to no account
    yet), and it means the scope makes the panel's two figures *agree*; it is
    not a ceiling on total allocation. The over-allocation it lets through
    surfaces at claim time through ``_reconcile``, which derives
    ``cash_available`` and so reports the state instead of hiding it.

    If this ever starts returning 400, the bound became real -- update
    ``api/routers/portfolio._scope`` rather than deleting the case.
    """
    client, user_store, _portfolio_store, agent_store = env
    agent_store.create_agent(
        name="Guest sleeve",
        owner_user_id=None,
        owner_browser_session=BROWSER,
        agent_type="builtin",
        cash_allocation=9000.0,
    )
    _signup(client)
    user = user_store.get_user_by_email("scope@example.com")
    target = agent_store.create_agent(
        name="Target",
        owner_user_id=user["id"],
        owner_browser_session=BROWSER,
        agent_type="builtin",
        cash_allocation=0.0,
    )

    # No X-Browser-Id and no X-Session-Id: nothing to scope by.
    bare = {
        "Origin": "http://testserver",
        "X-CSRF-Token": client.cookies.get(csrf_cookie_name()) or "",
    }
    resp = client.get("/api/v1/portfolio", headers=bare)
    assert resp.status_code == 200, resp.text
    portfolio = resp.json()["portfolio"]
    # Starter agents only (3 x 1000) — the 9000 guest sleeve is invisible here.
    assert portfolio["allocated"] == pytest.approx(3000.0)
    assert portfolio["cash_available"] == pytest.approx(7000.0)
    assert portfolio["unclaimed_allocated"] == pytest.approx(0.0)

    resp = client.post(
        "/api/v1/portfolio/allocate",
        json={"agent_id": target["agent_id"], "amount": 500.0},
        headers=bare,
    )
    assert resp.status_code == 200, resp.text
