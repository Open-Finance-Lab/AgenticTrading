"""Tests for registered external agents API."""

import uuid

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app


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


def test_create_and_list_agents(client):
    browser_session = str(uuid.uuid4())
    headers = {"X-Session-Id": browser_session}

    created = client.post(
        "/api/v1/agents",
        json={"name": "my-strategy", "model_name": "rsi-demo"},
        headers=headers,
    )
    assert created.status_code == 200
    body = created.json()
    assert body["api_key"].startswith("ag_")
    assert body["session_id"]
    assert body["agent"]["name"] == "my-strategy"

    listed = client.get("/api/v1/agents", headers=headers)
    assert listed.status_code == 200
    agents = listed.json()["agents"]
    assert len(agents) == 1
    assert agents[0]["agent_id"] == body["agent"]["agent_id"]


def test_create_builtin_agent_and_public_listing(client):
    browser_session = str(uuid.uuid4())
    headers = {"X-Session-Id": browser_session}

    created = client.post(
        "/api/v1/agents",
        json={
            "name": "Momentum Alpha",
            "model_name": "anthropic/claude-haiku-4-5",
            "agent_type": "builtin",
            "description": "Trend-following hosted agent",
        },
        headers=headers,
    )
    assert created.status_code == 200
    agent = created.json()["agent"]
    assert agent["agent_type"] == "builtin"
    assert agent["description"] == "Trend-following hosted agent"

    # The public builtin listing requires no auth/session and exposes the agent.
    listing = client.get("/api/v1/agents/builtin")
    assert listing.status_code == 200
    builtin = listing.json()["agents"]
    assert any(a["agent_id"] == agent["agent_id"] for a in builtin)
    entry = next(a for a in builtin if a["agent_id"] == agent["agent_id"])
    assert entry["model_name"] == "anthropic/claude-haiku-4-5"
    assert "api_key" not in entry and "owner_user_id" not in entry


def test_builtin_agent_card_counts_website_runs(client):
    """Built-in agents surface all session runs, not only ext_ ones."""
    browser_session = str(uuid.uuid4())
    headers = {"X-Session-Id": browser_session}

    created = client.post(
        "/api/v1/agents",
        json={"name": "WebBot", "agent_type": "builtin"},
        headers=headers,
    ).json()["agent"]

    import dashboard.backend.database as db_module

    db_module.db.insert_run(
        run_id="run_website_1",  # NOT an ext_ run — produced by /backtest/run.
        session_id=created["session_id"],
        agent_name="WebBot",
        mode="backtest",
        start_date="2026-04-15",
        end_date="2026-04-16",
        initial_equity=100000,
        final_equity=102000,
        total_return=0.02,
        sharpe_ratio=0.8,
        max_drawdown=-0.01,
        num_trades=4,
        llm_model="anthropic/claude-haiku-4-5",
    )

    # Matching real frontend usage: X-Session-Id is scoped to the active agent's
    # trading session, while X-Browser-Id carries the stable owner credential the
    # dashboard sends on every request (owner_browser_session), which is what
    # authorizes access — the session_id alone is not an ownership credential.
    active_headers = {
        "X-Session-Id": created["session_id"],
        "X-Browser-Id": browser_session,
    }
    fetched = client.get(
        f"/api/v1/agents/{created['agent_id']}", headers=active_headers
    )
    assert fetched.status_code == 200
    enriched = fetched.json()["agent"]
    assert enriched["run_count"] == 1
    assert enriched["latest_run"]["run_id"] == "run_website_1"


def test_resolve_api_key(client):
    browser_session = str(uuid.uuid4())
    headers = {"X-Session-Id": browser_session}
    created = client.post(
        "/api/v1/agents",
        json={"name": "resolver-test"},
        headers=headers,
    ).json()

    resolved = client.get(
        "/api/v1/agents/resolve",
        headers={"X-API-Key": created["api_key"]},
    )
    assert resolved.status_code == 200
    data = resolved.json()
    assert data["session_id"] == created["session_id"]
    assert data["name"] == "resolver-test"


def test_resolve_invalid_api_key(client):
    resp = client.get("/api/v1/agents/resolve", headers={"X-API-Key": "ag_invalid"})
    assert resp.status_code == 401


def test_import_session_from_backtest_runs(client):
    browser_session = str(uuid.uuid4())
    headers = {"X-Session-Id": browser_session}

    import dashboard.backend.database as db_module

    db_module.db.insert_run(
        run_id="ext_test_import",
        session_id=browser_session,
        agent_name="my-strategy",
        mode="backtest",
        start_date="2026-04-15",
        end_date="2026-04-16",
        initial_equity=100000,
        final_equity=101000,
        total_return=0.01,
        sharpe_ratio=0.5,
        max_drawdown=-0.02,
        num_trades=3,
        llm_model="rsi-demo",
    )

    imported = client.post("/api/v1/agents/import-session", json={}, headers=headers)
    assert imported.status_code == 200
    body = imported.json()
    assert body["agent"]["name"] == "my-strategy"
    assert body["agent"]["session_id"] == browser_session

    listed = client.get("/api/v1/agents", headers=headers)
    assert len(listed.json()["agents"]) == 1


def test_claim_account_links_browser_agents(client):
    browser_session = str(uuid.uuid4())
    anon_headers = {"X-Session-Id": browser_session, "X-Browser-Id": browser_session}

    created = client.post(
        "/api/v1/agents",
        json={"name": "pre-login-agent"},
        headers=anon_headers,
    )
    assert created.status_code == 200
    agent_id = created.json()["agent"]["agent_id"]

    signup = client.post(
        "/api/auth/signup",
        json={
            "email": "claim-test@example.com",
            "display_name": "Claim Test",
            "password": "securepass123",
        },
    )
    assert signup.status_code == 200
    token = signup.json()["token"]
    auth_headers = {
        **anon_headers,
        "Authorization": f"Bearer {token}",
    }

    claimed = client.post("/api/v1/agents/claim-account", headers=auth_headers)
    assert claimed.status_code == 200
    body = claimed.json()
    assert body["claimed"] >= 1
    assert any(a["agent_id"] == agent_id for a in body["agents"])

    listed = client.get("/api/v1/agents", headers={"Authorization": f"Bearer {token}"})
    assert listed.status_code == 200
    assert len(listed.json()["agents"]) == 1
    assert listed.json()["agents"][0]["agent_id"] == agent_id


def test_signed_in_list_includes_unclaimed_browser_foundation_agent(client):
    """Regression: logout shows guest starter; signup must still see it pre-claim."""
    browser_session = str(uuid.uuid4())
    anon_headers = {"X-Session-Id": browser_session, "X-Browser-Id": browser_session}

    created = client.post(
        "/api/v1/agents",
        json={
            "name": "My Foundation Agent",
            "model_name": "deepseek/deepseek-v4-pro",
            "agent_type": "builtin",
        },
        headers=anon_headers,
    )
    assert created.status_code == 200
    agent_id = created.json()["agent"]["agent_id"]

    signup = client.post(
        "/api/auth/signup",
        json={
            "email": "foundation-list@example.com",
            "display_name": "Foundation List",
            "password": "securepass123",
        },
    )
    assert signup.status_code == 200
    token = signup.json()["token"]
    auth_headers = {
        **anon_headers,
        "Authorization": f"Bearer {token}",
    }

    # Before claim-account: signed-in listing must still surface the guest agent.
    listed = client.get("/api/v1/agents", headers=auth_headers)
    assert listed.status_code == 200
    ids = [a["agent_id"] for a in listed.json()["agents"]]
    assert agent_id in ids
    assert any(a.get("name") == "My Foundation Agent" for a in listed.json()["agents"])


def test_rotate_api_key(client):
    browser_session = str(uuid.uuid4())
    headers = {"X-Session-Id": browser_session, "X-Browser-Id": browser_session}

    created = client.post(
        "/api/v1/agents",
        json={"name": "rotate-me"},
        headers=headers,
    ).json()
    agent_id = created["agent"]["agent_id"]
    old_key = created["api_key"]

    rotated = client.post(f"/api/v1/agents/{agent_id}/rotate-api-key", headers=headers)
    assert rotated.status_code == 200
    new_key = rotated.json()["api_key"]
    assert new_key.startswith("ag_")
    assert new_key != old_key

    assert client.get("/api/v1/agents/resolve", headers={"X-API-Key": old_key}).status_code == 401
    resolved = client.get("/api/v1/agents/resolve", headers={"X-API-Key": new_key})
    assert resolved.status_code == 200
    assert resolved.json()["agent_id"] == agent_id


def test_builtin_listing_does_not_leak_session_id(client):
    """The public /builtin listing must not expose the ownership-sensitive
    session_id (regression for the unauthenticated-takeover vulnerability)."""
    owner = str(uuid.uuid4())
    created = client.post(
        "/api/v1/agents",
        json={"name": "victim-bot", "agent_type": "builtin"},
        headers={"X-Session-Id": owner},
    ).json()
    agent_id = created["agent"]["agent_id"]

    listing = client.get("/api/v1/agents/builtin")
    assert listing.status_code == 200
    entry = next(a for a in listing.json()["agents"] if a["agent_id"] == agent_id)
    assert "session_id" not in entry, "public builtin listing leaks session_id"


def test_leaked_session_id_cannot_take_over_agent(client):
    """Even if an attacker learns an agent's session_id, replaying it as
    X-Session-Id must not grant ownership on state-changing routes."""
    owner = str(uuid.uuid4())
    created = client.post(
        "/api/v1/agents",
        json={"name": "victim-bot", "agent_type": "builtin"},
        headers={"X-Session-Id": owner},
    ).json()
    agent_id = created["agent"]["agent_id"]
    leaked_session_id = created["session_id"]

    attacker = {"X-Session-Id": leaked_session_id}
    delete_response = client.delete(f"/api/v1/agents/{agent_id}", headers=attacker)
    assert delete_response.status_code == 403
    assert (
        client.post(f"/api/v1/agents/{agent_id}/rotate-api-key", headers=attacker).status_code
        == 403
    )

    # The legitimate owner (real browser session) still manages the agent.
    owner_headers = {"X-Session-Id": owner}
    rotated = client.post(f"/api/v1/agents/{agent_id}/rotate-api-key", headers=owner_headers)
    assert rotated.status_code == 200

    # The agent's own API key is a valid credential for state-changing routes.
    new_key = rotated.json()["api_key"]
    deleted = client.delete(f"/api/v1/agents/{agent_id}", headers={"X-API-Key": new_key})
    assert deleted.status_code == 200


def test_patch_agent_name_and_pipeline(client):
    browser_session = str(uuid.uuid4())
    headers = {"X-Session-Id": browser_session, "X-Browser-Id": browser_session}

    created = client.post(
        "/api/v1/agents",
        json={
            "name": "Test_in",
            "model_name": "anthropic/claude-haiku-4-5",
            "agent_type": "builtin",
        },
        headers=headers,
    )
    assert created.status_code == 200
    agent_id = created.json()["agent"]["agent_id"]

    patched = client.patch(
        f"/api/v1/agents/{agent_id}",
        json={"name": "Renamed Agent"},
        headers=headers,
    )
    assert patched.status_code == 200
    assert patched.json()["agent"]["name"] == "Renamed Agent"

    listed = client.get("/api/v1/agents", headers=headers)
    assert listed.json()["agents"][0]["name"] == "Renamed Agent"


def test_patch_agent_legacy_session_owner(client):
    """Dashboard may reclaim agents when X-Session-Id matches the agent session."""
    browser_session = str(uuid.uuid4())
    create_headers = {"X-Session-Id": browser_session, "X-Browser-Id": browser_session}
    created = client.post(
        "/api/v1/agents",
        json={"name": "Legacy Owner", "agent_type": "builtin"},
        headers=create_headers,
    )
    assert created.status_code == 200
    agent = created.json()["agent"]
    agent_id = agent["agent_id"]
    session_id = agent["session_id"]

    modern_headers = {
        "X-Session-Id": session_id,
        "X-Browser-Id": str(uuid.uuid4()),
    }
    patched = client.patch(
        f"/api/v1/agents/{agent_id}",
        json={"name": "Renamed Legacy"},
        headers=modern_headers,
    )
    assert patched.status_code == 200
    assert patched.json()["agent"]["name"] == "Renamed Legacy"


def test_builtin_listing_batches_run_stats_queries(client, monkeypatch):
    """LOW #9 — the public, unauthenticated /agents/builtin listing must not
    issue one runs-by-session query per agent (N+1): the stats lookup happens
    in a single batched query no matter how many builtin agents exist."""
    for i in range(3):
        resp = client.post(
            "/api/v1/agents",
            json={"name": f"builtin-{i}", "agent_type": "builtin"},
            headers={"X-Session-Id": str(uuid.uuid4())},
        )
        assert resp.status_code == 200, resp.text

    import dashboard.backend.api.routers.agents as agents_api

    svc_db = agents_api.agent_service.db
    calls = {"per_session": 0, "batch": 0}
    orig_single = svc_db.get_runs_by_session
    orig_batch = svc_db.get_runs_by_sessions  # must exist — AttributeError = RED

    def counting_single(session_id):
        calls["per_session"] += 1
        return orig_single(session_id)

    def counting_batch(session_ids):
        calls["batch"] += 1
        return orig_batch(session_ids)

    monkeypatch.setattr(svc_db, "get_runs_by_session", counting_single)
    monkeypatch.setattr(svc_db, "get_runs_by_sessions", counting_batch)

    listing = client.get("/api/v1/agents/builtin")
    assert listing.status_code == 200
    assert len(listing.json()["agents"]) == 3
    assert calls["per_session"] == 0, "listing still queries per agent (N+1)"
    assert calls["batch"] == 1, "listing must fetch all stats in one query"


def test_owner_listing_batches_run_stats_queries(client, monkeypatch):
    """Owned /agents listing must batch run-stats the same way as /agents/builtin."""
    browser_session = str(uuid.uuid4())
    headers = {"X-Session-Id": browser_session, "X-Browser-Id": browser_session}
    for i in range(3):
        resp = client.post(
            "/api/v1/agents",
            json={"name": f"owned-{i}", "agent_type": "external"},
            headers=headers,
        )
        assert resp.status_code == 200, resp.text

    import dashboard.backend.api.routers.agents as agents_api

    svc_db = agents_api.agent_service.db
    calls = {"per_session": 0, "batch": 0}
    orig_single = svc_db.get_runs_by_session
    orig_batch = svc_db.get_runs_by_sessions

    def counting_single(session_id):
        calls["per_session"] += 1
        return orig_single(session_id)

    def counting_batch(session_ids):
        calls["batch"] += 1
        return orig_batch(session_ids)

    monkeypatch.setattr(svc_db, "get_runs_by_session", counting_single)
    monkeypatch.setattr(svc_db, "get_runs_by_sessions", counting_batch)

    listing = client.get("/api/v1/agents", headers=headers)
    assert listing.status_code == 200
    assert len(listing.json()["agents"]) == 3
    assert calls["per_session"] == 0, "owner listing still queries per agent (N+1)"
    assert calls["batch"] == 1, "owner listing must fetch all stats in one query"


def test_cash_allocation_cap_is_three_thousand(client):
    """Per-agent sleeve max is $3,000."""
    from dashboard.backend.domain.backtesting.constants import (
        MAX_AGENT_CASH_ALLOCATION,
        MAX_BACKTEST_INITIAL_CAPITAL,
        resolve_initial_capital,
    )

    assert MAX_AGENT_CASH_ALLOCATION == 3_000
    assert MAX_BACKTEST_INITIAL_CAPITAL == 10_000
    # Clamp behavior follows the backtest capital constant (not the sleeve max).
    assert resolve_initial_capital(3_000) == 3_000.0
    assert resolve_initial_capital(10_000) == 10_000.0
    assert resolve_initial_capital(50_000) == 10_000.0
    assert resolve_initial_capital(None) == 1_000.0

    browser_session = str(uuid.uuid4())
    headers = {"X-Session-Id": browser_session, "X-Browser-Id": browser_session}

    ok = client.post(
        "/api/v1/agents",
        json={
            "name": "Capped",
            "agent_type": "builtin",
            "cash_allocation": 3_000,
        },
        headers=headers,
    )
    assert ok.status_code == 200
    assert ok.json()["agent"]["cash_allocation"] == 3_000

    too_big = client.post(
        "/api/v1/agents",
        json={"name": "Too big", "agent_type": "builtin", "cash_allocation": 3_001},
        headers=headers,
    )
    assert too_big.status_code == 422


def test_patch_agent_model_name(client):
    """Demo 1: the Configure screen can change the model after creation."""
    browser_session = str(uuid.uuid4())
    headers = {"X-Session-Id": browser_session, "X-Browser-Id": browser_session}

    created = client.post(
        "/api/v1/agents",
        json={
            "name": "Model Swapper",
            "model_name": "anthropic/claude-haiku-4-5",
            "agent_type": "builtin",
        },
        headers=headers,
    )
    assert created.status_code == 200
    agent_id = created.json()["agent"]["agent_id"]

    patched = client.patch(
        f"/api/v1/agents/{agent_id}",
        json={"model_name": "deepseek/deepseek-v4-pro"},
        headers=headers,
    )
    assert patched.status_code == 200
    assert patched.json()["agent"]["model_name"] == "deepseek/deepseek-v4-pro"

    # Absent field leaves the model untouched.
    renamed = client.patch(
        f"/api/v1/agents/{agent_id}",
        json={"name": "Still Swapped"},
        headers=headers,
    )
    assert renamed.status_code == 200
    assert renamed.json()["agent"]["model_name"] == "deepseek/deepseek-v4-pro"

    # Empty string is rejected by validation.
    empty = client.patch(
        f"/api/v1/agents/{agent_id}",
        json={"model_name": ""},
        headers=headers,
    )
    assert empty.status_code == 422

    # Whitespace-only is also rejected: min_length=1 counts the raw length, so
    # "   " would otherwise pass and then strip to "" in a NOT NULL column.
    blank_model = client.patch(
        f"/api/v1/agents/{agent_id}",
        json={"model_name": "   "},
        headers=headers,
    )
    assert blank_model.status_code == 422

    # The same blank guard covers name (identical strip-to-empty hazard).
    blank_name = client.patch(
        f"/api/v1/agents/{agent_id}",
        json={"name": "   "},
        headers=headers,
    )
    assert blank_name.status_code == 422

    # The model was never mutated by the rejected requests.
    unchanged = client.get(f"/api/v1/agents/{agent_id}", headers=headers)
    assert unchanged.json()["agent"]["model_name"] == "deepseek/deepseek-v4-pro"

    # model_name alone is a valid update (not "No fields to update").
    only_model = client.patch(
        f"/api/v1/agents/{agent_id}",
        json={"model_name": "openai/gpt-5.5"},
        headers=headers,
    )
    assert only_model.status_code == 200
    assert only_model.json()["agent"]["model_name"] == "openai/gpt-5.5"


def test_marketplace_listing_and_clone(client):
    import dashboard.backend.domain.agents.marketplace as marketplace_mod

    marketplace_mod.reload_marketplace_catalog()
    listing = client.get("/api/v1/agents/marketplace")
    assert listing.status_code == 200
    templates = listing.json()["templates"]
    assert templates
    assert any(t["template_id"] == "balanced-starter" for t in templates)

    browser_session = str(uuid.uuid4())
    headers = {"X-Session-Id": browser_session}
    cloned = client.post(
        "/api/v1/agents/marketplace/balanced-starter/clone",
        json={},
        headers=headers,
    )
    assert cloned.status_code == 200
    agent = cloned.json()["agent"]
    assert agent["name"] == "Balanced Starter"
    assert agent["agent_type"] == "builtin"
    assert agent.get("pipeline")
    assert agent["pipeline"][0]["presetKey"] == "simple_instruction"

    listed = client.get("/api/v1/agents", headers=headers)
    assert listed.status_code == 200
    assert any(a["agent_id"] == agent["agent_id"] for a in listed.json()["agents"])


def test_marketplace_clone_unknown_template(client):
    browser_session = str(uuid.uuid4())
    headers = {"X-Session-Id": browser_session}
    resp = client.post(
        "/api/v1/agents/marketplace/does-not-exist/clone",
        json={},
        headers=headers,
    )
    assert resp.status_code == 404
