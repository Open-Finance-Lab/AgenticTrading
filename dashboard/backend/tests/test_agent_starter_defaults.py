"""Starter-instruction seeding and the simple-only Configure screen.

Two things are guarded here.

**Seeding.** The starter instruction used to be applied by a follow-up ``PATCH``
issued from ``app.js`` after the agent was created, wrapped in a
``catch { console.warn }``. ``PATCH`` was missing from the CORS ``allow_methods``
(fixed in #245), so in production that preflight 400'd and the seed silently did
nothing -- every default agent shipped with an empty pipeline and quietly fell
back to the generic hourly prompt at backtest time. Seeding now happens inside
``AgentService.create_agent``, in the same call that creates the row.

**Contract sync.** The preset key and output format are duplicated in
``dashboard/frontend/app.js`` because the browser has to recognise a
server-seeded pipeline as the editable simple kind. If the two copies drift,
``isSimplePipeline()`` stops matching and every default agent renders the
"saving replaces your custom pipeline" warning it should never show.
"""

import re
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
import dashboard.backend.domain.agents.repository as agent_store_module
import dashboard.backend.domain.agents.marketplace as marketplace_module
from dashboard.backend.domain.agents.defaults import (
    DEFAULT_STARTER_INSTRUCTION,
    SIMPLE_INSTRUCTION_OUTPUT_FORMAT,
    SIMPLE_INSTRUCTION_PRESET_KEY,
    STARTER_AGENT_DESCRIPTION,
    STARTER_AGENT_MODEL,
    STARTER_AGENT_NAME,
    STARTER_AGENTS,
)
from dashboard.backend.tests._frontend_source import js_string_const

AgentStore = agent_store_module.AgentStore

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_EDITOR_JS = (_FRONTEND / "js" / "agent-editor.js").read_text(encoding="utf-8")


@pytest.fixture
def client(tmp_path, monkeypatch):
    import dashboard.backend.api.routers.agents as agents_api
    import dashboard.backend.database as db_module

    db_path = tmp_path / "test.db"
    test_agents = AgentStore(db_path=db_path)
    test_db = db_module.BacktestDatabase(db_path=db_path)
    monkeypatch.setattr(agent_store_module, "agent_store", test_agents)
    monkeypatch.setattr(agents_api.agent_service, "agents", test_agents)
    monkeypatch.setattr(agents_api.agent_service, "db", test_db)
    monkeypatch.setattr(db_module, "db", test_db)
    return TestClient(app)


def _create(client, **overrides):
    """Create an agent; returns (body, owner headers) so callers can re-read it."""
    payload = {"name": "seeded", "model_name": "deepseek/deepseek-v4-pro"}
    payload.update(overrides)
    headers = {"X-Session-Id": str(uuid.uuid4())}
    response = client.post("/api/v1/agents", json=payload, headers=headers)
    assert response.status_code == 200, response.text
    return response.json(), headers


# --------------------------------------------------------------------------
# Server-side seeding
# --------------------------------------------------------------------------


def test_provision_starter_agents_creates_the_three_model_cards(client):
    from dashboard.backend.domain.agents.service import agent_service

    created = agent_service.provision_starter_agents(owner_user_id=42)
    assert {a["model_name"] for a in created} == {s["model_name"] for s in STARTER_AGENTS}
    by_model = {a["model_name"]: a for a in created}
    for spec in STARTER_AGENTS:
        agent = by_model[spec["model_name"]]
        assert agent["name"] == spec["name"]
        assert agent["agent_type"] == "builtin"
        assert agent["description"] == spec["description"]
        assert agent["pipeline"]


def test_provision_starter_agents_is_idempotent_on_model_name(client):
    from dashboard.backend.domain.agents.service import agent_service

    first = agent_service.provision_starter_agents(owner_user_id=42)
    second = agent_service.provision_starter_agents(owner_user_id=42)
    assert len(first) == len(STARTER_AGENTS)
    assert second == []


def test_provision_renames_a_starter_whose_title_drifted_from_its_model(client):
    """A Claude card stored under the DeepSeek title must be rewritten.

    The grid title used to read `agent.name` while the subtitle read the
    model, so a drifted row showed "DeepSeek V4 Pro" over "Claude Sonnet 4.6".
    """
    from dashboard.backend.domain.agents.service import agent_service

    drifted = agent_service.create_agent(
        name="DeepSeek V4 Pro",
        model_name="anthropic/claude-sonnet-4-6",
        owner_user_id=42,
        owner_browser_session=None,
        agent_type="builtin",
    )
    agent_service.provision_starter_agents(owner_user_id=42)
    listed = agent_service.agents.list_agents(owner_user_id=42)
    claude = next(
        a for a in listed if a["model_name"] == "anthropic/claude-sonnet-4-6"
    )
    assert claude["agent_id"] == drifted["agent_id"]
    assert claude["name"] == "Claude Sonnet 4.6"


def test_provision_starter_agent_fail_open_does_not_raise(client, monkeypatch):
    from dashboard.backend.domain.agents.service import agent_service

    def boom(**_kwargs):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(agent_service, "create_agent", boom)
    assert agent_service.provision_starter_agent(owner_user_id=1) is None
    assert agent_service.provision_starter_agents(owner_user_id=1) == []


def test_new_builtin_agent_is_seeded_with_the_starter_instruction(client):
    agent = _create(client, agent_type="builtin")[0]["agent"]

    pipeline = agent.get("pipeline")
    assert pipeline, "a built-in agent must be usable without opening Configure"
    assert len(pipeline) == 1
    assert pipeline[0]["presetKey"] == SIMPLE_INSTRUCTION_PRESET_KEY
    assert pipeline[0]["prompt"] == DEFAULT_STARTER_INSTRUCTION
    assert pipeline[0]["outputFormat"] == SIMPLE_INSTRUCTION_OUTPUT_FORMAT


def test_seeded_pipeline_survives_a_reread(client):
    """The seed is persisted, not just decorated onto the create response."""
    created, headers = _create(client, agent_type="builtin")
    created = created["agent"]

    fetched = client.get(f"/api/v1/agents/{created['agent_id']}", headers=headers)
    assert fetched.status_code == 200
    pipeline = fetched.json()["agent"]["pipeline"]
    assert pipeline[0]["prompt"] == DEFAULT_STARTER_INSTRUCTION


def test_external_agents_are_not_seeded(client):
    """External agents are driven by the protocol API and never use the editor."""
    agent = _create(client, agent_type="external")[0]["agent"]
    assert not agent.get("pipeline")


def test_create_still_returns_the_one_time_api_key(client):
    """Seeding must not clobber the create response.

    ``create_agent``'s dict carries the plaintext ``api_key`` that the route pops
    and shows exactly once; it is never recoverable afterwards. ``update_agent``
    re-reads the row, which stores only the hash -- so seeding by reassigning the
    agent dict would drop the key and 500 the request.
    """
    body, _ = _create(client, agent_type="builtin")
    assert body["api_key"].startswith("ag_")
    assert body["session_id"]


def test_marketplace_clone_keeps_its_own_pipeline(client):
    """A template with its own pipeline must not be overwritten by the seed."""
    cloned = client.post(
        "/api/v1/agents/marketplace/pipeline-analyst/clone",
        json={},
        headers={"X-Session-Id": str(uuid.uuid4())},
    )
    assert cloned.status_code == 200, cloned.text
    pipeline = cloned.json()["agent"]["pipeline"]
    assert len(pipeline) == 3
    assert [step["presetKey"] for step in pipeline] == [
        "info_gather",
        "info_to_signal",
        "signal_to_execution",
    ]


def test_marketplace_clone_without_a_pipeline_gets_the_starter_seed(client, monkeypatch):
    """A template that ships no pipeline of its own must still land usable.

    Every real template in marketplace.json currently carries a pipeline, so this
    branch of clone_marketplace_template (seed_default_pipeline=True) is otherwise
    untested. Fake the catalog lookup rather than editing the real config.
    """
    fake_template = {
        "template_id": "no-pipeline-template",
        "name": "No Pipeline Template",
        "model_name": "local-model",
        "description": "A template with no pipeline of its own.",
    }
    monkeypatch.setattr(
        marketplace_module,
        "get_marketplace_template",
        lambda template_id: fake_template if template_id == "no-pipeline-template" else None,
    )
    cloned = client.post(
        "/api/v1/agents/marketplace/no-pipeline-template/clone",
        json={},
        headers={"X-Session-Id": str(uuid.uuid4())},
    )
    assert cloned.status_code == 200, cloned.text
    pipeline = cloned.json()["agent"]["pipeline"]
    assert len(pipeline) == 1
    assert pipeline[0]["presetKey"] == SIMPLE_INSTRUCTION_PRESET_KEY
    assert pipeline[0]["prompt"] == DEFAULT_STARTER_INSTRUCTION


# --------------------------------------------------------------------------
# JS <-> Python contract sync
# --------------------------------------------------------------------------


#: Shared with the node-executing frontend guards -- see _frontend_source.py.
#: Kept under the local name so the assertions below read unchanged.
_js_const = js_string_const


def test_the_js_constants_are_still_matched():
    """Guard the guard: a rename must fail loudly rather than pass vacuously."""
    for name in (
        "SIMPLE_INSTRUCTION_PRESET_KEY",
        "SIMPLE_INSTRUCTION_OUTPUT_FORMAT",
        "DEFAULT_STARTER_INSTRUCTION",
    ):
        assert _js_const(name)


def test_preset_key_matches_the_frontend():
    assert _js_const("SIMPLE_INSTRUCTION_PRESET_KEY") == SIMPLE_INSTRUCTION_PRESET_KEY


def test_output_format_matches_the_frontend():
    assert (
        _js_const("SIMPLE_INSTRUCTION_OUTPUT_FORMAT") == SIMPLE_INSTRUCTION_OUTPUT_FORMAT
    )


def test_starter_instruction_matches_the_frontend():
    assert _js_const("DEFAULT_STARTER_INSTRUCTION") == DEFAULT_STARTER_INSTRUCTION


def test_starter_agent_identity_matches_the_frontend():
    assert _js_const("DEFAULT_STARTER_AGENT_NAME") == STARTER_AGENT_NAME
    assert _js_const("DEFAULT_FOUNDATION_MODEL") == STARTER_AGENT_MODEL
    assert _js_const("DEFAULT_STARTER_AGENT_DESCRIPTION") == STARTER_AGENT_DESCRIPTION


def test_starter_agents_roster_matches_the_frontend():
    from dashboard.backend.tests._frontend_source import js_const

    decl = js_const("STARTER_AGENTS")
    assert len(STARTER_AGENTS) == 3
    for spec in STARTER_AGENTS:
        assert spec["name"] in decl
        assert spec["model_name"] in decl
        assert spec["description"] in decl


def test_the_editor_can_read_the_starter_instruction():
    """The editor shows the default in its disclosure, so app.js must publish it.

    (This used to back a *backfill* that injected the text into the textarea;
    that was removed when an empty instruction became a supported state --
    see tests/test_my_agents_instruction_ui.py.)
    """
    assert "window.DEFAULT_STARTER_INSTRUCTION = DEFAULT_STARTER_INSTRUCTION" in _APP_JS
    assert "window.DEFAULT_STARTER_INSTRUCTION" in _EDITOR_JS


def test_app_js_no_longer_seeds_the_pipeline_over_the_network():
    """The fail-open seed PATCH is the bug; it must not come back."""
    assert "Starter instruction seed failed" not in _APP_JS


# --------------------------------------------------------------------------
# Advanced mode is gone
# --------------------------------------------------------------------------


ADVANCED_MARKUP = [
    "agentEditorModeSimple",
    "agentEditorModeAdvanced",
    "agentEditorAdvancedPanel",
    "agentEditorPipeline",
    "agentEditorAddSelect",
    "agentEditorAddBtn",
    "agentEditorResetBtn",
    "agentEditorStepCount",
    "agentEditorCustomName",
]


@pytest.mark.parametrize("marker", ADVANCED_MARKUP)
def test_configure_screen_has_no_advanced_mode_markup(marker):
    assert marker not in _APP_HTML


@pytest.mark.parametrize("symbol", ["setEditorMode", "editorMode", "SUB_AGENT_PRESETS"])
def test_agent_editor_has_no_mode_switching_code(symbol):
    assert symbol not in _EDITOR_JS


def test_the_simple_panel_is_always_visible():
    """It is the only panel now, so it must not ship with a `hidden` attribute."""
    match = re.search(r'<div id="agentEditorSimplePanel"[^>]*>', _APP_HTML)
    assert match, "the simple panel is missing from app.html"
    assert "hidden" not in match.group(0)
