"""POST /api/v1/agents/{agent_id}/duplicate -- the "Run on another model" hook.

Copies an existing built-in agent onto a different model, server-side so the
pipeline copy and the ownership check are one path. It deliberately does NOT
start a backtest: auto-firing would spend LLM credits on a click the user did
not frame as "run".
"""

import uuid

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.domain.agents.credential_store import AgentCredentialStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    import dashboard.backend.domain.agents.repository as agent_store_module
    import dashboard.backend.api.routers.agents as agents_api
    import dashboard.backend.domain.agents.service as agent_service_module
    import dashboard.backend.database as db_module
    import dashboard.backend.domain.brokers.repository as broker_repository

    db_path = tmp_path / "test.db"
    test_agents = agent_store_module.AgentStore(db_path=db_path)
    test_credentials = AgentCredentialStore(db_path=db_path)
    test_db = db_module.BacktestDatabase(db_path=db_path)
    monkeypatch.setenv(
        broker_repository._KEY_ENV_VAR, Fernet.generate_key().decode()
    )
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)
    monkeypatch.setattr(agent_store_module, "agent_store", test_agents)
    monkeypatch.setattr(agents_api.agent_service, "agents", test_agents)
    monkeypatch.setattr(agents_api.agent_service, "db", test_db)
    monkeypatch.setattr(agents_api, "agent_credential_store", test_credentials)
    # The service retires runtime-owned credentials on a runtime switch, so it
    # needs the same store the router writes through.
    monkeypatch.setattr(
        agent_service_module, "agent_credential_store", test_credentials
    )
    monkeypatch.setattr(db_module, "db", test_db)
    agents_api._credential_rate_limiter.reset()
    return TestClient(app)


def _create_builtin(client, headers, *, model="anthropic/claude-haiku-4-5"):
    created = client.post(
        "/api/v1/agents",
        json={
            "name": "Source Agent",
            "model_name": model,
            "agent_type": "builtin",
            "description": "The original.",
        },
        headers=headers,
    )
    assert created.status_code == 200, created.text
    return created.json()["agent"]


def test_duplicate_copies_the_agent_onto_a_new_model(client):
    headers = {"X-Session-Id": str(uuid.uuid4())}
    source = _create_builtin(client, headers)

    response = client.post(
        f"/api/v1/agents/{source['agent_id']}/duplicate",
        json={"model_name": "deepseek/deepseek-v4-pro", "name": "Source Agent (DeepSeek)"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    copy = response.json()["agent"]
    assert copy["agent_id"] != source["agent_id"]
    assert copy["model_name"] == "deepseek/deepseek-v4-pro"
    assert copy["name"] == "Source Agent (DeepSeek)"
    assert copy["description"] == "The original."
    # A duplicate never auto-launches a backtest -- the user lands on the copy
    # with Run primed, not mid-run.
    assert copy["run_count"] == 0 and copy["latest_run"] is None


def test_duplicate_copies_the_pipeline(client):
    headers = {"X-Session-Id": str(uuid.uuid4())}
    source = _create_builtin(client, headers)
    pipeline = [
        {
            "id": "sub_custom",
            "presetKey": "simple_instruction",
            "label": "Trading instruction",
            "prompt": "Buy only what you would hold for a week.",
            "outputFormat": "JSON: { \"orders\": [] }",
        }
    ]
    patched = client.patch(
        f"/api/v1/agents/{source['agent_id']}",
        json={"pipeline": pipeline},
        headers=headers,
    )
    assert patched.status_code == 200, patched.text

    response = client.post(
        f"/api/v1/agents/{source['agent_id']}/duplicate",
        json={"model_name": "qwen/qwen3.7-plus"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    copy_id = response.json()["agent"]["agent_id"]
    fetched = client.get(f"/api/v1/agents/{copy_id}", headers=headers).json()["agent"]
    assert [step["prompt"] for step in fetched["pipeline"]] == [
        "Buy only what you would hold for a week."
    ]


def test_duplicate_copies_backtest_allocation_but_not_cash_allocation(client):
    """The one thing a user does with a duplicate is compare its equity curve
    against the source's -- different starting capital makes those curves
    incomparable. ``backtest_allocation`` is simulated capital with no ledger
    coupling, so it is safe to copy. ``cash_allocation`` IS a real ledger debit
    (the route reserves only DEFAULT_AGENT_CASH_ALLOCATION for it) and must
    stay un-copied."""
    headers = {"X-Session-Id": str(uuid.uuid4())}
    created = client.post(
        "/api/v1/agents",
        json={
            "name": "Source Agent",
            "model_name": "anthropic/claude-haiku-4-5",
            "agent_type": "builtin",
            "cash_allocation": 2500,
            "backtest_allocation": 2000,
        },
        headers=headers,
    )
    assert created.status_code == 200, created.text
    source = created.json()["agent"]
    assert source["backtest_allocation"] == 2000

    response = client.post(
        f"/api/v1/agents/{source['agent_id']}/duplicate",
        json={"model_name": "deepseek/deepseek-v4-pro"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    copy = response.json()["agent"]
    assert copy["backtest_allocation"] == 2000
    assert copy["cash_allocation"] == 1000


def test_duplicate_defaults_the_name(client):
    headers = {"X-Session-Id": str(uuid.uuid4())}
    source = _create_builtin(client, headers)
    response = client.post(
        f"/api/v1/agents/{source['agent_id']}/duplicate",
        json={"model_name": "openai/gpt-5.5"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    assert response.json()["agent"]["name"] == "Source Agent copy"


def test_duplicate_rejects_another_owners_agent(client):
    owner = {"X-Session-Id": str(uuid.uuid4())}
    attacker = {"X-Session-Id": str(uuid.uuid4())}
    source = _create_builtin(client, owner)
    response = client.post(
        f"/api/v1/agents/{source['agent_id']}/duplicate",
        json={"model_name": "openai/gpt-5.5"},
        headers=attacker,
    )
    assert response.status_code in (403, 404), response.text


def test_duplicate_rejects_an_unknown_agent(client):
    response = client.post(
        f"/api/v1/agents/{uuid.uuid4()}/duplicate",
        json={"model_name": "openai/gpt-5.5"},
        headers={"X-Session-Id": str(uuid.uuid4())},
    )
    assert response.status_code == 404, response.text


def test_duplicate_rejects_an_external_agent(client):
    """External agents mint an API key on create -- not a surface this hook opens."""
    headers = {"X-Session-Id": str(uuid.uuid4())}
    created = client.post(
        "/api/v1/agents",
        json={"name": "Connected", "model_name": "local-model", "agent_type": "external"},
        headers=headers,
    )
    assert created.status_code == 200, created.text
    response = client.post(
        f"/api/v1/agents/{created.json()['agent']['agent_id']}/duplicate",
        json={"model_name": "openai/gpt-5.5"},
        headers=headers,
    )
    assert response.status_code == 400, response.text


@pytest.mark.parametrize("body", [{}, {"model_name": ""}, {"model_name": "   "}])
def test_duplicate_requires_a_model_name(client, body):
    headers = {"X-Session-Id": str(uuid.uuid4())}
    source = _create_builtin(client, headers)
    response = client.post(
        f"/api/v1/agents/{source['agent_id']}/duplicate", json=body, headers=headers
    )
    assert response.status_code == 422, response.text


def test_duplicate_does_not_leak_the_plaintext_api_key(client):
    """repository.create_agent() mints a plaintext api_key on EVERY agent it
    creates, built-in included, and attaches it directly on the dict it
    returns. duplicate_agent only sheds that key on the has_own_pipeline
    branch, because that branch reassigns ``agent`` from update_agent()'s
    return value (which never carries the key). A built-in agent with no
    pipeline of its own -- e.g. one on the ai_hedge_fund runtime -- takes the
    other branch, and the raw dict (key attached) used to flow straight
    through agent_with_stats's shallow copy into the response."""
    headers = {"X-Session-Id": str(uuid.uuid4())}
    source = client.post(
        "/api/v1/agents",
        json={
            "name": "Hedge Fund Source",
            "model_name": "anthropic/claude-haiku-4-5",
            "agent_type": "builtin",
            "runtime_type": "ai_hedge_fund",
        },
        headers=headers,
    )
    assert source.status_code == 200, source.text

    response = client.post(
        f"/api/v1/agents/{source.json()['agent']['agent_id']}/duplicate",
        json={"model_name": "openai/gpt-5.5"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    assert "api_key" not in response.json()["agent"]


def test_create_agent_still_returns_the_one_time_api_key(client):
    """Guard for the fix above: agent_with_stats now strips api_key from the
    enriched copy it returns, but POST /api/v1/agents's top-level api_key
    field pops the key off the *original*, un-enriched dict -- a separate
    object -- so the one-time reveal must be unaffected."""
    headers = {"X-Session-Id": str(uuid.uuid4())}
    created = client.post(
        "/api/v1/agents",
        json={"name": "Keyed", "model_name": "local-model", "agent_type": "external"},
        headers=headers,
    )
    assert created.status_code == 200, created.text
    assert created.json()["api_key"].startswith("ag_")


def test_marketplace_clone_does_not_leak_the_plaintext_api_key(client):
    """Pre-existing leak, not introduced by this PR: clone_marketplace_template
    has the identical has_own_pipeline branch as duplicate_agent, and the
    ``ai-hedge-fund`` template carries no pipeline of its own, so cloning it
    takes the leaking branch today. Closed by the same agent_with_stats fix."""
    headers = {"X-Session-Id": str(uuid.uuid4())}
    cloned = client.post(
        "/api/v1/agents/marketplace/ai-hedge-fund/clone",
        json={},
        headers=headers,
    )
    assert cloned.status_code == 200, cloned.text
    assert "api_key" not in cloned.json()["agent"]
