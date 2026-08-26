"""Phase 3A2 — agent lifecycle service tests.

Exercise the extracted ``AgentService`` workflows directly against isolated
temporary SQLite databases (never the live DB). The service is constructed with
per-test repository + database instances so no global state leaks between tests.
"""

import ast
from pathlib import Path

import pytest

from dashboard.backend.database import BacktestDatabase
from dashboard.backend.domain.agents import service as service_module
from dashboard.backend.domain.agents import repository, version_repository
from dashboard.backend.domain.agents.repository import AgentStore
from dashboard.backend.domain.agents.version_repository import AgentVersionStore
from dashboard.backend.domain.agents.service import (
    AgentAccessDeniedError,
    AgentNotFoundError,
    AgentService,
    InvalidVersionFieldError,
    NoExternalRunsError,
    agent_service,
    sample_equity_sparkline,
)


@pytest.fixture
def svc(tmp_path):
    db_path = tmp_path / "service.db"
    return AgentService(
        agents=AgentStore(db_path=db_path),
        versions=AgentVersionStore(db_path=db_path),
        database=BacktestDatabase(db_path=db_path),
    )


def _insert_ext_run(database, *, run_id, session_id, agent_name="strat", llm_model="m1"):
    database.insert_run(
        run_id=run_id,
        session_id=session_id,
        agent_name=agent_name,
        mode="backtest",
        start_date="2026-04-15",
        end_date="2026-04-16",
        initial_equity=100000,
        final_equity=101000,
        total_return=0.01,
        sharpe_ratio=0.5,
        max_drawdown=-0.02,
        num_trades=3,
        llm_model=llm_model,
    )


# ---------------------------------------------------------------------------
# Wiring / import boundary
# ---------------------------------------------------------------------------

def test_singleton_wired_to_canonical_repositories():
    assert agent_service.agents is repository.agent_store
    assert agent_service.versions is version_repository.agent_version_store


def test_service_does_not_import_api_scripts_or_fastapi():
    tree = ast.parse(Path(service_module.__file__).read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    for m in mods:
        assert not m.startswith("dashboard.backend.api"), m
        assert not m.startswith("dashboard.scripts"), m
        assert m != "fastapi" and not m.startswith("fastapi."), m
        assert m != "backtest_hourly_agent", m


# ---------------------------------------------------------------------------
# Create / register
# ---------------------------------------------------------------------------

def test_create_agent_returns_full_schema(svc):
    agent = svc.create_agent(
        name="my-strategy",
        model_name="rsi-demo",
        owner_user_id=None,
        owner_browser_session="b1",
    )
    assert agent["name"] == "my-strategy"
    assert agent["model_name"] == "rsi-demo"
    assert agent["agent_id"].startswith("agent_")
    assert agent["api_key"].startswith("ag_")


def test_owned_agent_lifecycle_emits_after_repository_success(svc, monkeypatch):
    events = []
    monkeypatch.setattr(
        service_module.analytics_instrumentation,
        "emit_agent_event",
        lambda **kwargs: events.append(kwargs),
    )

    agent = svc.create_agent(
        name="owned",
        model_name="model",
        owner_user_id=7,
        owner_browser_session="browser",
    )
    svc.update_agent(agent["agent_id"], name="updated")
    assert svc.delete_agent(agent["agent_id"]) is True
    assert svc.delete_agent(agent["agent_id"]) is False

    assert [event["event_name"] for event in events] == [
        "agent_created",
        "agent_updated",
        "agent_deleted",
    ]
    assert all(event["user_id"] == 7 for event in events)


def test_guest_agent_lifecycle_is_not_attributed_to_user(svc, monkeypatch):
    events = []
    monkeypatch.setattr(
        service_module.analytics_instrumentation,
        "emit_agent_event",
        lambda **kwargs: events.append(kwargs),
    )

    agent = svc.create_agent(
        name="guest",
        model_name="model",
        owner_user_id=None,
        owner_browser_session="browser",
    )
    svc.update_agent(agent["agent_id"], name="updated")
    svc.delete_agent(agent["agent_id"])

    assert events == []


def test_runtime_type_update_resets_runtime_specific_config(svc):
    agent = svc.create_agent(
        name="hosted",
        model_name="gpt-4.1",
        owner_user_id=None,
        owner_browser_session="b1",
        runtime_type="ai_hedge_fund",
        runtime_config={"analysts": ["technical_analyst"]},
    )

    updated = svc.update_agent(agent["agent_id"], runtime_type="pipeline")

    assert updated["runtime_type"] == "pipeline"
    assert updated["runtime_config"] == {}


def test_create_agent_with_stats_schema(svc):
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=None, owner_browser_session="b1"
    )
    enriched = svc.agent_with_stats(agent)
    assert enriched["run_count"] == 0
    assert enriched["latest_run"] is None
    assert enriched["runs"] == []
    assert enriched["total_llm_calls"] == 0
    assert enriched["total_input_tokens"] == 0
    assert enriched["total_output_tokens"] == 0
    assert enriched["total_est_cost_usd"] == 0


# ---------------------------------------------------------------------------
# Run statistics
# ---------------------------------------------------------------------------

def test_agent_with_stats_counts_only_external_runs(svc):
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=None, owner_browser_session="b1"
    )
    session = agent["session_id"]
    _insert_ext_run(svc.db, run_id="ext_1", session_id=session)
    svc.db.insert_run(
        run_id="plain_1",
        session_id=session,
        agent_name="x",
        mode="backtest",
        start_date="2026-04-15",
        end_date="2026-04-16",
        initial_equity=100000,
        final_equity=100000,
        total_return=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        num_trades=0,
        llm_model="m",
    )
    enriched = svc.agent_with_stats(agent)
    assert enriched["run_count"] == 1
    assert enriched["latest_run"] is not None
    assert len(enriched["runs"]) == 1


def test_sample_equity_sparkline_keeps_ends_when_downsampling():
    curve = [{"equity": float(i)} for i in range(100)]
    spark = sample_equity_sparkline(curve, max_points=10)
    assert len(spark) == 10
    assert spark[0] == 0.0
    assert spark[-1] == 99.0


def test_list_agents_with_stats_attaches_equity_sparkline(svc):
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=None, owner_browser_session="b1"
    )
    session = agent["session_id"]
    _insert_ext_run(svc.db, run_id="ext_1", session_id=session)
    svc.db.insert_equity_points(
        "ext_1",
        [
            {
                "timestamp": f"2026-04-15T{10 + i:02d}:00:00",
                "equity": 100000 + i * 250,
                "cash": 50000,
                "positions_value": 50000 + i * 250,
                "daily_return": 0.0,
            }
            for i in range(8)
        ],
    )
    listed = svc.list_agents_with_stats(
        owner_user_id=None,
        owner_browser_session="b1",
        trading_session_id=None,
    )
    assert len(listed) == 1
    spark = listed[0]["equity_sparkline"]
    assert spark[0] == 100000.0
    assert spark[-1] == 100000 + 7 * 250
    assert listed[0]["latest_run"]["equity_sparkline"] == spark


def test_list_external_runs(svc):
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=None, owner_browser_session="b1"
    )
    session = agent["session_id"]
    _insert_ext_run(svc.db, run_id="ext_1", session_id=session)
    runs = svc.list_external_runs(session)
    assert [r["run_id"] for r in runs] == ["ext_1"]
    assert svc.list_external_runs("no-such-session") == []


# ---------------------------------------------------------------------------
# Ownership / access
# ---------------------------------------------------------------------------

def test_require_access_by_user(svc):
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=1, owner_browser_session=None
    )
    got = svc.require_access(agent["agent_id"], user_id=1)
    assert got["agent_id"] == agent["agent_id"]


def test_require_access_cross_user_rejected(svc):
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=1, owner_browser_session=None
    )
    with pytest.raises(AgentAccessDeniedError):
        svc.require_access(agent["agent_id"], user_id=2)


def test_require_access_missing_agent(svc):
    with pytest.raises(AgentNotFoundError):
        svc.require_access("agent_does_not_exist", user_id=1)


def test_require_access_rejects_bare_session_id(svc):
    """A matching trading session_id is NOT an ownership credential (regression
    guard for the unauthenticated-takeover bug). Ownership requires a real
    credential — owner_user_id / owner_browser_session — or the agent API key."""
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=99, owner_browser_session=None
    )
    # Knowing only the agent's (discoverable) session_id must NOT grant access.
    with pytest.raises(AgentAccessDeniedError):
        svc.require_access(agent["agent_id"], user_id=None, browser_session=None)
    # The real owner (user_id) is still recognized.
    got = svc.require_access(agent["agent_id"], user_id=99)
    assert got["agent_id"] == agent["agent_id"]


def test_require_access_reclaim_does_not_steal_bound_agent(svc):
    """Matching trading session + browser must not reassign another account."""
    agent = svc.create_agent(
        name="A",
        model_name="m",
        owner_user_id=1,
        owner_browser_session="b-owner",
    )
    with pytest.raises(AgentAccessDeniedError):
        svc.require_access(
            agent["agent_id"],
            user_id=2,
            browser_session="b-thief",
            trading_session=agent["session_id"],
            reclaim_on_session_match=True,
        )
    assert svc.get_agent(agent["agent_id"])["owner_user_id"] == 1


def test_session_match_grant_never_binds_an_account(svc):
    """A session_id match grants access but must not claim the agent.

    ``owns_agent`` now refuses browser-only access to a bound row, so an
    account binding made off a bare session match is irreversible: the real
    owner's browser credential stops working with no way back. Access itself
    stays open (test_patch_agent_legacy_session_owner pins that contract) --
    only the ownership write is withheld.
    """
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=None, owner_browser_session="b-victim"
    )
    got, owned = svc.resolve_access(
        agent["agent_id"],
        user_id=2,
        browser_session="b-thief",
        trading_session=agent["session_id"],
        reclaim_on_session_match=True,
    )
    assert got["agent_id"] == agent["agent_id"]
    assert owned is False
    assert svc.get_agent(agent["agent_id"])["owner_user_id"] is None
    # Still reversible: the original browser wins it back on its own session id.
    _, owned_back = svc.resolve_access(
        agent["agent_id"],
        browser_session="b-victim",
        trading_session=agent["session_id"],
        reclaim_on_session_match=True,
    )
    assert owned_back is False
    assert svc.agents.owns_agent(agent, owner_browser_session="b-victim") is True


def test_resolve_access_reports_real_credentials_as_owned(svc):
    """The flag must be True for the credentials that legitimately claim."""
    guest = svc.create_agent(
        name="G", model_name="m", owner_user_id=None, owner_browser_session="b1"
    )
    assert svc.resolve_access(guest["agent_id"], browser_session="b1")[1] is True

    bound = svc.create_agent(
        name="B", model_name="m", owner_user_id=7, owner_browser_session="b2"
    )
    assert svc.resolve_access(bound["agent_id"], user_id=7)[1] is True


def test_import_session_rejects_another_accounts_agent(svc):
    """import-session must not rename, re-own, or hand back a bound agent."""
    _insert_ext_run(svc.db, run_id="ext_1", session_id="sess-a", agent_name="orig")
    owned = svc.agents.register_or_get_agent(
        session_id="sess-a", name="orig", owner_user_id=1
    )
    with pytest.raises(AgentAccessDeniedError):
        svc.import_session(
            session_id="sess-a", user_id=2, name="stolen", model_name="m"
        )
    row = svc.get_agent(owned["agent_id"])
    assert row["owner_user_id"] == 1
    assert row["name"] == "orig"


def test_activate_does_not_overwrite_other_owner(svc):
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=1, owner_browser_session="b1"
    )
    svc.activate_agent(agent["agent_id"], user_id=2, browser_session="b1")
    assert svc.get_agent(agent["agent_id"])["owner_user_id"] == 1


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------

def test_claim_account_agents(svc):
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=None, owner_browser_session="b1"
    )
    claimed, agents = svc.claim_account_agents(browser_session="b1", user_id=10)
    assert claimed >= 1
    assert any(a["agent_id"] == agent["agent_id"] for a in agents)
    assert svc.get_agent(agent["agent_id"])["owner_user_id"] == 10
    # Returned agents are stats-enriched.
    assert "run_count" in agents[0]


# ---------------------------------------------------------------------------
# API key rotation / delete
# ---------------------------------------------------------------------------

def test_rotate_api_key(svc):
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=None, owner_browser_session="b1"
    )
    old_key = agent["api_key"]
    new_key = svc.rotate_api_key(agent["agent_id"])
    assert new_key.startswith("ag_")
    assert new_key != old_key
    assert svc.resolve_api_key(old_key) is None
    assert svc.resolve_api_key(new_key)["agent_id"] == agent["agent_id"]
    assert svc.rotate_api_key("missing") is None


def test_delete_agent(svc):
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=None, owner_browser_session="b1"
    )
    assert svc.delete_agent(agent["agent_id"]) is True
    assert svc.get_agent(agent["agent_id"]) is None
    assert svc.delete_agent(agent["agent_id"]) is False


def test_activate_agent_claims_ownership(svc):
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=None, owner_browser_session="b1"
    )
    svc.activate_agent(agent["agent_id"], user_id=55, browser_session="b1")
    assert svc.get_agent(agent["agent_id"])["owner_user_id"] == 55


# ---------------------------------------------------------------------------
# Import session
# ---------------------------------------------------------------------------

def test_import_session_without_runs_raises(svc):
    with pytest.raises(NoExternalRunsError):
        svc.import_session(
            session_id="empty-session", user_id=None, name=None, model_name=None
        )


def test_import_session_creates_then_idempotent(svc):
    session = "sess-import"
    _insert_ext_run(
        svc.db, run_id="ext_1", session_id=session,
        agent_name="from-run", llm_model="model-from-run",
    )
    agent, imported = svc.import_session(
        session_id=session, user_id=None, name=None, model_name=None
    )
    assert imported is True
    assert agent["session_id"] == session
    assert agent["name"] == "from-run"
    assert agent["model_name"] == "model-from-run"

    agent2, imported2 = svc.import_session(
        session_id=session, user_id=None, name=None, model_name=None
    )
    assert imported2 is False
    assert agent2["agent_id"] == agent["agent_id"]


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

def _create_version(svc, agent_id, **overrides):
    params = dict(
        agent_id=agent_id,
        version="0.1.0",
        execution_mode="external",
        architecture=None,
        model_backbones=[],
        decision_frequency="1h",
        code_commit=None,
        prompt_hash=None,
        config_hash=None,
        prompt=None,
        config=None,
        verification_level="self_reported",
    )
    params.update(overrides)
    return svc.create_version(**params)


def test_create_and_get_version(svc):
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=None, owner_browser_session="b1"
    )
    version = _create_version(svc, agent["agent_id"])
    assert version["agent_version_id"].startswith("agv_")
    assert version["agent_id"] == agent["agent_id"]
    fetched = svc.get_version(version["agent_version_id"])
    assert fetched["agent_version_id"] == version["agent_version_id"]


def test_get_version_missing(svc):
    assert svc.get_version("agv_missing") is None


def test_create_version_invalid_execution_mode(svc):
    with pytest.raises(InvalidVersionFieldError) as exc:
        _create_version(svc, "agent_x", execution_mode="bogus")
    assert str(exc.value) == "Invalid execution_mode: bogus"


def test_create_version_invalid_verification_level(svc):
    with pytest.raises(InvalidVersionFieldError) as exc:
        _create_version(svc, "agent_x", verification_level="bogus")
    assert str(exc.value) == "Invalid verification_level: bogus"


def test_list_versions_newest_first(svc):
    agent = svc.create_agent(
        name="A", model_name="m", owner_user_id=None, owner_browser_session="b1"
    )
    v1 = _create_version(svc, agent["agent_id"], version="0.1.0")
    v2 = _create_version(svc, agent["agent_id"], version="0.2.0")
    versions = svc.list_versions(agent["agent_id"])
    ids = [v["agent_version_id"] for v in versions]
    assert set(ids) == {v1["agent_version_id"], v2["agent_version_id"]}
    assert svc.list_versions("agent_with_no_versions") == []
