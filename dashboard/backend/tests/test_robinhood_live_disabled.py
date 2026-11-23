"""The Robinhood real-money path is switched off by one flag, not deleted.

Real capital is out of scope (docs/superpowers/specs/2026-10-14-atl-direction.md),
so ``ROBINHOOD_LIVE_ENABLED`` (``domain/backtesting/constants.py``) is False.
Every route stays mounted and every handler stays imported, so this suite
exercises the code a restore would turn back on rather than trusting that an
uncomment would still boot.

What "off" means, pinned here by behaviour rather than by spelling:

* the routes that start a link, finish one or place a live run answer 404;
* the OAuth callback returns the browser to /app with ``robinhood=disabled``
  and never exchanges its code, so a link in flight at deploy obtains nothing;
* ``status`` stays up and reports the switch as ``enabled`` -- the agent
  editor's only copy of it -- without calling Robinhood;
* ``disconnect`` stays up, because it is how an account linked before the
  switch went off deletes its stored real-money tokens;
* ``PATCH /api/v1/agents/{id}`` refuses ``live_trading_enabled: true``, so an
  operator who clears the column is not undone by a stale tab or an API client;
* the editor shows the panel only to a linked account, with Disconnect alone,
  and its save leaves ``live_trading_enabled`` out of the PATCH.

The ``*_when_switched_on`` cases flip the flag and show the same routes reach
their handlers, which is the claim that a restore is one line. The rest of the
restore checklist is in CLAUDE.md (`execution/`).
"""

from __future__ import annotations

import json
import shutil
import uuid
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

import dashboard.backend.api.auth as auth_module
import dashboard.backend.api.routers.robinhood_live as robinhood_router
import dashboard.backend.database as db_module
import dashboard.backend.domain.agents.repository as agent_store_module
import dashboard.backend.domain.backtesting.constants as constants
from dashboard.backend.app import app
from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    APP_JS,
    FRONTEND,
    fn_body,
    run_node,
)

_EDITOR_JS = (FRONTEND / "js" / "agent-editor.js").read_text(encoding="utf-8")
_USER = {"id": 7, "email": "linked@example.com"}

_needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def signed_in():
    app.dependency_overrides[auth_module.get_current_user] = lambda: dict(_USER)
    yield
    app.dependency_overrides.pop(auth_module.get_current_user, None)


@pytest.fixture
def switched_on(monkeypatch):
    monkeypatch.setattr(constants, "ROBINHOOD_LIVE_ENABLED", True)


def test_the_switch_is_off():
    assert constants.ROBINHOOD_LIVE_ENABLED is False
    assert constants.robinhood_live_enabled() is False


def test_every_robinhood_route_stays_mounted():
    mounted = {
        (method, route.path)
        for route in app.routes
        if "robinhood" in getattr(route, "path", "")
        for method in getattr(route, "methods", ())
    }
    assert mounted == {
        ("POST", "/api/auth/robinhood/start"),
        ("GET", "/api/auth/robinhood/callback"),
        ("POST", "/api/auth/robinhood/complete"),
        ("GET", "/api/v1/robinhood/status"),
        ("DELETE", "/api/v1/robinhood/disconnect"),
        ("POST", "/api/v1/robinhood/agents/{agent_id}/live-run"),
    }


# --- routes, switched off -------------------------------------------------


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("post", "/api/auth/robinhood/start", {"agent_id": "a1"}),
        ("post", "/api/auth/robinhood/complete", {"link_code": "abcdefgh12"}),
        ("post", "/api/v1/robinhood/agents/a1/live-run", {"dry_run": True}),
    ],
)
def test_link_and_live_run_routes_answer_404(client, signed_in, monkeypatch, method, path, body):
    monkeypatch.setattr(
        auth_module.robinhood_oauth,
        "register_client",
        lambda: pytest.fail("a switched-off start must not reach Robinhood"),
    )
    monkeypatch.setattr(
        auth_module.broker_store,
        "upsert_tokens",
        lambda *a, **k: pytest.fail("a switched-off complete must not store tokens"),
    )

    async def _no_live_run(**_kwargs):
        pytest.fail("a switched-off live-run must not reach the live service")

    monkeypatch.setattr(robinhood_router.live_service, "run_live_for_agent", _no_live_run)

    resp = getattr(client, method)(path, json=body)

    assert resp.status_code == 404
    assert resp.json()["detail"] == auth_module.ROBINHOOD_LIVE_SWITCHED_OFF_DETAIL


def test_the_switch_answers_before_authentication(client):
    """Off, the gated routes look unmounted to everyone, signed in or not."""
    resp = client.post("/api/v1/robinhood/agents/a1/live-run", json={"dry_run": True})
    assert resp.status_code == 404


def test_callback_returns_to_the_app_without_exchanging_the_code(client, monkeypatch):
    monkeypatch.setattr(
        auth_module.robinhood_oauth,
        "exchange_code_for_tokens",
        lambda **_kwargs: pytest.fail("a switched-off callback must not exchange its code"),
    )

    resp = client.get(
        "/api/auth/robinhood/callback",
        params={"code": "upstream-code", "state": "anything"},
        follow_redirects=False,
    )

    assert resp.status_code == 302
    location = urlparse(resp.headers["location"])
    assert location.path.endswith("/app")
    assert parse_qs(location.query) == {"robinhood": ["disabled"]}


def test_status_reports_the_switch_without_calling_robinhood(client, signed_in, monkeypatch):
    seen = {}

    async def _status(user_id, *, include_portfolio=False):
        seen.update(user_id=user_id, include_portfolio=include_portfolio)
        return {"broker": "robinhood", "connected": True}

    monkeypatch.setattr(robinhood_router.live_service, "get_connection_status", _status)

    resp = client.get("/api/v1/robinhood/status", params={"portfolio": "true"})

    assert resp.status_code == 200
    assert resp.json() == {"broker": "robinhood", "connected": True, "enabled": False}
    # A portfolio read goes to Robinhood's MCP server; off, only the local link is read.
    assert seen == {"user_id": _USER["id"], "include_portfolio": False}


@pytest.mark.parametrize("revoked", [True, False])
def test_disconnect_still_deletes_the_stored_tokens(client, signed_in, monkeypatch, revoked):
    deleted = []

    async def _revoke(user_id):
        assert user_id == _USER["id"]
        return revoked

    monkeypatch.setattr(robinhood_router.live_service, "revoke_connection", _revoke)
    monkeypatch.setattr(robinhood_router.broker_store, "delete", deleted.append)

    resp = client.delete("/api/v1/robinhood/disconnect")

    assert resp.status_code == 200
    assert deleted == [_USER["id"]]
    # revoke_connection reports failure as False rather than raising, so the
    # response must carry its answer, not "it returned".
    assert resp.json() == {"status": "ok", "connected": False, "revoked_upstream": revoked}


# --- routes, switched on (the restore is one line) -------------------------


def test_routes_reach_their_handlers_when_switched_on(client, signed_in, switched_on, monkeypatch):
    monkeypatch.setattr(auth_module.robinhood_oauth, "oauth_configured", lambda: False)
    monkeypatch.setattr(auth_module.pending_links, "pop", lambda _code: None)
    monkeypatch.setattr(robinhood_router.agent_service, "get_agent", lambda _agent_id: None)

    start = client.post("/api/auth/robinhood/start", json={})
    complete = client.post("/api/auth/robinhood/complete", json={"link_code": "abcdefgh12"})
    live_run = client.post("/api/v1/robinhood/agents/a1/live-run", json={"dry_run": True})

    assert (start.status_code, start.json()["detail"]) == (503, "Robinhood OAuth is not configured")
    assert (complete.status_code, complete.json()["detail"]) == (400, "Link expired - please connect again.")
    assert (live_run.status_code, live_run.json()["detail"]) == (404, "Agent not found")


def test_callback_exchanges_the_code_when_switched_on(client, switched_on, monkeypatch):
    monkeypatch.setattr(
        auth_module.robinhood_oauth,
        "parse_oauth_state",
        lambda _state: (_ for _ in ()).throw(ValueError("state_expired")),
    )

    resp = client.get(
        "/api/auth/robinhood/callback",
        params={"code": "upstream-code", "state": "expired"},
        follow_redirects=False,
    )

    assert parse_qs(urlparse(resp.headers["location"]).query) == {
        "robinhood": ["error"],
        "reason": ["state_expired"],
    }


# --- PATCH /api/v1/agents: the arming switch -------------------------------


@pytest.fixture
def agents_client(tmp_path, monkeypatch):
    import dashboard.backend.api.routers.agents as agents_api

    db_path = tmp_path / "test.db"
    test_agents = agent_store_module.AgentStore(db_path=db_path)
    test_db = db_module.BacktestDatabase(db_path=db_path)
    monkeypatch.setattr(agent_store_module, "agent_store", test_agents)
    monkeypatch.setattr(agents_api.agent_service, "agents", test_agents)
    monkeypatch.setattr(agents_api.agent_service, "db", test_db)
    monkeypatch.setattr(db_module, "db", test_db)
    return TestClient(app)


def _guest_agent(client):
    headers = {"X-Session-Id": str(uuid.uuid4())}
    created = client.post(
        "/api/v1/agents", json={"name": "alpha", "agent_type": "builtin"}, headers=headers
    )
    assert created.status_code == 200, created.text
    return created.json()["agent"]["agent_id"], headers


def test_patch_refuses_to_arm_an_agent(agents_client):
    agent_id, headers = _guest_agent(agents_client)

    resp = agents_client.patch(
        f"/api/v1/agents/{agent_id}", json={"live_trading_enabled": True}, headers=headers
    )

    assert resp.status_code == 409
    assert "switched off" in resp.json()["detail"]
    stored = agents_client.get(f"/api/v1/agents/{agent_id}", headers=headers).json()["agent"]
    assert stored["live_trading_enabled"] is False


def test_patch_still_disarms_and_still_saves_without_the_field(agents_client):
    agent_id, headers = _guest_agent(agents_client)

    disarm = agents_client.patch(
        f"/api/v1/agents/{agent_id}", json={"live_trading_enabled": False}, headers=headers
    )
    rename = agents_client.patch(
        f"/api/v1/agents/{agent_id}", json={"name": "beta"}, headers=headers
    )

    assert disarm.status_code == 200, disarm.text
    assert rename.status_code == 200, rename.text
    assert rename.json()["agent"]["name"] == "beta"


def test_patch_arms_an_agent_when_switched_on(agents_client, switched_on):
    agent_id, headers = _guest_agent(agents_client)

    resp = agents_client.patch(
        f"/api/v1/agents/{agent_id}", json={"live_trading_enabled": True}, headers=headers
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["agent"]["live_trading_enabled"] is True


# --- the agent editor -------------------------------------------------------


def _panel_open_tag() -> str:
    start = APP_HTML.index('id="agentEditorBrokerPanel"')
    return APP_HTML[APP_HTML.rindex("<", 0, start) : APP_HTML.index(">", start) + 1]


def test_the_panel_ships_hidden_and_its_routes_are_still_called():
    assert " hidden>" in _panel_open_tag()
    assert "refreshRobinhoodStatus();" in fn_body("function open(agent)", _EDITOR_JS)
    assert "handleRobinhoodOAuthReturn();" in fn_body("function initAuthUI(", APP_JS)
    assert "let robinhoodLiveEnabled = false;" in _EDITOR_JS


_PANEL_IDS = (
    "agentEditorBrokerPanel",
    "agentEditorRobinhoodIntro",
    "agentEditorLiveTradingToggle",
    "agentEditorConnectRobinhoodBtn",
    "agentEditorDisconnectRobinhoodBtn",
    "agentEditorRunLiveBtn",
)


def _run_editor(scenarios: dict) -> dict:
    lifted = "\n".join(
        fn_body(signature, _EDITOR_JS)
        for signature in (
            "function formatUsd(",
            "function setBrokerMessage(",
            "function setRunLiveProminence(",
            "function setBrokerPanelVisible(",
            "function setLiveControlsVisible(",
            "function renderRobinhoodSwitchedOff(",
            "async function refreshRobinhoodStatus(",
            "async function patchAgent(",
        )
    )
    script = f"""
let els = {{}};
function el(id) {{
  if (!els[id]) {{
    els[id] = {{ id, hidden: false, textContent: '', className: '', disabled: false,
                 checked: false, title: '', classList: {{ toggle() {{}} }} }};
  }}
  return els[id];
}}
const document = {{ getElementById: el }};
const API_BASE = '';
let robinhoodLiveEnabled = false;
let currentAgent = {{ agent_id: 'a1', live_trading_enabled: true }};
let signedIn = true;
let statusAnswer = null;
let patched = null;
function isEditorSignedIn() {{ return signedIn; }}
function authHeaders() {{ return {{}}; }}
function serializePipeline(p) {{ return p; }}
async function fetchWithTimeout() {{
  if (statusAnswer instanceof Error) throw statusAnswer;
  return {{ ok: true, json: async () => statusAnswer }};
}}
const window = {{ API: {{ patch: async (_url, payload) => {{ patched = payload; return {{ agent: {{}} }}; }} }} }};
console.warn = () => {{}};
{lifted}
const scenarios = {json.dumps(scenarios)};
(async () => {{
  const out = {{}};
  for (const [name, s] of Object.entries(scenarios)) {{
    els = {{}};
    el('agentEditorBrokerPanel').hidden = true;
    el('agentEditorDisconnectRobinhoodBtn').hidden = true;
    signedIn = s.signedIn;
    statusAnswer = s.error ? new Error('offline') : s.answer;
    await refreshRobinhoodStatus();
    const visible = {{}};
    for (const id of {json.dumps(_PANEL_IDS)}) visible[id] = !el(id).hidden;
    await patchAgent(currentAgent, 'n', '', null, 0, 1000, '',
                     robinhoodLiveEnabled ? true : undefined, null, null);
    out[name] = {{ visible, enabled: robinhoodLiveEnabled,
                   sendsLiveFlag: Object.prototype.hasOwnProperty.call(patched, 'live_trading_enabled') }};
  }}
  process.stdout.write(JSON.stringify(out));
}})();
"""
    return run_node(script)


@_needs_node
def test_editor_shows_a_linked_account_only_its_disconnect_button():
    result = _run_editor(
        {
            "linked": {"signedIn": True, "answer": {"connected": True, "enabled": False}},
            "unlinked": {"signedIn": True, "answer": {"connected": False, "enabled": False}},
            "pre_switch_server": {"signedIn": True, "answer": {"connected": True}},
            "status_failed": {"signedIn": True, "error": True},
            "signed_out": {"signedIn": False},
        }
    )

    linked = result["linked"]["visible"]
    assert linked["agentEditorBrokerPanel"] and linked["agentEditorDisconnectRobinhoodBtn"]
    assert not any(
        linked[i]
        for i in (
            "agentEditorRobinhoodIntro",
            "agentEditorLiveTradingToggle",
            "agentEditorConnectRobinhoodBtn",
            "agentEditorRunLiveBtn",
        )
    )
    # A server with no `enabled` field predates the switch; silence is not "on".
    assert result["pre_switch_server"]["visible"] == linked
    for name in ("unlinked", "status_failed", "signed_out"):
        assert result[name]["visible"]["agentEditorBrokerPanel"] is False, name
    for name, outcome in result.items():
        assert outcome["enabled"] is False, name
        assert outcome["sendsLiveFlag"] is False, name


@_needs_node
def test_editor_offers_the_whole_panel_when_switched_on():
    result = _run_editor(
        {"on": {"signedIn": True, "answer": {"connected": False, "enabled": True}}}
    )["on"]

    assert result["enabled"] is True
    assert result["sendsLiveFlag"] is True
    visible = result["visible"]
    assert all(
        visible[i]
        for i in (
            "agentEditorBrokerPanel",
            "agentEditorRobinhoodIntro",
            "agentEditorLiveTradingToggle",
            "agentEditorConnectRobinhoodBtn",
            "agentEditorRunLiveBtn",
        )
    )
    assert visible["agentEditorDisconnectRobinhoodBtn"] is False


def test_save_passes_the_live_flag_only_while_the_switch_is_on():
    save = fn_body("async function save(", _EDITOR_JS)
    assert "robinhoodLiveEnabled ? state.live_trading_enabled : undefined" in save
    for guarded in ("async function connectRobinhood(", "async function runLive("):
        body = fn_body(guarded, _EDITOR_JS)
        assert body.index("if (!robinhoodLiveEnabled)") < body.index("fetch("), guarded
