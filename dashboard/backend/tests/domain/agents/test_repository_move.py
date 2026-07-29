"""Phase 3A1 — agent repository move + characterization.

Verifies identity/re-export, the domain->api/scripts import boundary, and that
``AgentStore`` behaves exactly as before. All tests use an isolated temporary
SQLite database (never the live DB).
"""

import ast
from pathlib import Path

import pytest

from dashboard.backend.domain.agents import repository
from dashboard.backend.domain.agents.repository import AgentStore


@pytest.fixture
def store(tmp_path):
    return AgentStore(db_path=tmp_path / "agents.db")


# ---------------------------------------------------------------------------
# Canonical identity
# ---------------------------------------------------------------------------

def test_canonical_module_identity():
    assert repository.AgentStore.__module__ == (
        "dashboard.backend.domain.agents.repository"
    )


def test_singleton_uses_test_database():
    # conftest points DATABASE_PATH at a temp DB; the singleton must live there,
    # never the live backtest.db.
    from dashboard.backend.database import DB_PATH

    assert Path(repository.agent_store.db_path) == Path(DB_PATH)
    assert "storage/data/backtest.db" not in str(repository.agent_store.db_path)


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------

def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module)
    return mods


def test_domain_modules_do_not_import_api_or_scripts():
    for mod in (repository.__file__, ):
        mods = _imported_modules(Path(mod))
        for m in mods:
            assert not m.startswith("dashboard.backend.api"), m
            assert not m.startswith("dashboard.scripts"), m
            assert m != "fastapi" and not m.startswith("fastapi."), m


# ---------------------------------------------------------------------------
# AgentStore characterization
# ---------------------------------------------------------------------------

def test_create_agent_schema(store):
    agent = store.create_agent(name="My Agent", model_name="gpt-x", owner_user_id=7)
    assert set(agent.keys()) == {
        "agent_id", "name", "session_id", "model_name", "agent_type",
        "description", "pipeline", "cash_allocation", "backtest_allocation",
        "api_key_prefix", "owner_user_id", "scopes",
        "created_at", "last_used_at", "api_key", "live_trading_enabled",
    }
    assert agent["name"] == "My Agent"
    assert agent["model_name"] == "gpt-x"
    assert agent["owner_user_id"] == 7
    assert agent["agent_id"].startswith("agent_")
    assert agent["api_key"].startswith("ag_")
    assert agent["api_key_prefix"] == agent["api_key"][:12]
    # api_key_hash must never leak in the public dict.
    assert "api_key_hash" not in agent


def test_get_agent_and_missing(store):
    created = store.create_agent(name="A")
    fetched = store.get_agent(created["agent_id"])
    assert fetched["agent_id"] == created["agent_id"]
    assert "api_key" not in fetched  # only create returns the raw key
    assert store.get_agent("nope") is None


def test_get_agent_by_session(store):
    created = store.create_agent(name="A", session_id="sess-xyz")
    fetched = store.get_agent_by_session("sess-xyz")
    assert fetched["agent_id"] == created["agent_id"]
    assert store.get_agent_by_session("missing") is None


def test_resolve_api_key(store):
    created = store.create_agent(name="A")
    resolved = store.resolve_api_key(created["api_key"])
    assert resolved["agent_id"] == created["agent_id"]
    assert store.resolve_api_key("ag_wrong") is None
    assert store.resolve_api_key("") is None
    assert store.resolve_api_key("   ") is None


def test_register_or_get_agent_idempotent(store):
    first = store.register_or_get_agent(session_id="s1", name="A", model_name="m1")
    again = store.register_or_get_agent(session_id="s1", name="A2", model_name="m2")
    assert first["agent_id"] == again["agent_id"]
    assert again["name"] == "A2"
    assert again["model_name"] == "m2"


def test_list_agents_owner_filters(store):
    a_user = store.create_agent(name="U", owner_user_id=1)
    b_browser = store.create_agent(name="B", owner_browser_session="browser-1")
    store.create_agent(name="C", owner_user_id=2)

    by_user = store.list_agents(owner_user_id=1)
    assert [a["agent_id"] for a in by_user] == [a_user["agent_id"]]

    by_browser = store.list_agents(owner_browser_session="browser-1")
    assert [a["agent_id"] for a in by_browser] == [b_browser["agent_id"]]

    by_session = store.list_agents(trading_session_id=a_user["session_id"])
    assert [a["agent_id"] for a in by_session] == [a_user["agent_id"]]

    assert store.list_agents() == []


def test_list_agents_signed_in_includes_unclaimed_browser_agents(store):
    """Signed-in listing must not hide guest-provisioned agents awaiting claim."""
    owned = store.create_agent(name="Owned", owner_user_id=7, owner_browser_session="b1")
    unclaimed = store.create_agent(name="Guest", owner_browser_session="b1")
    other_user = store.create_agent(
        name="Other", owner_user_id=99, owner_browser_session="b1"
    )
    other_browser = store.create_agent(name="Elsewhere", owner_browser_session="b2")

    listed = store.list_agents(owner_user_id=7, owner_browser_session="b1")
    ids = {a["agent_id"] for a in listed}
    assert owned["agent_id"] in ids
    assert unclaimed["agent_id"] in ids
    assert other_user["agent_id"] not in ids
    assert other_browser["agent_id"] not in ids


def test_list_agents_merges_owned_and_unclaimed_by_recency(store, monkeypatch):
    """The owned-rows and unclaimed-browser-rows groups must merge into one
    global ORDER BY, not just be independently sorted within each group.

    Without the merge, a browser agent created after every owned agent would
    still sort last, because list_agents appends the unclaimed-browser group
    only after the owned group is fully built.
    """
    import itertools

    day = itertools.count(1)
    monkeypatch.setattr(
        repository, "_utcnow_iso", lambda: f"2020-01-{next(day):02d}T00:00:00+00:00"
    )

    older_owned = store.create_agent(name="Older Owned", owner_user_id=7, owner_browser_session="b1")
    newer_unclaimed = store.create_agent(name="Newer Guest", owner_browser_session="b1")

    listed = store.list_agents(owner_user_id=7, owner_browser_session="b1")
    assert [a["agent_id"] for a in listed] == [
        newer_unclaimed["agent_id"],
        older_owned["agent_id"],
    ]


def test_rotate_api_key(store):
    created = store.create_agent(name="A")
    new_key = store.rotate_api_key(created["agent_id"])
    assert new_key.startswith("ag_")
    assert new_key != created["api_key"]
    assert store.resolve_api_key(new_key)["agent_id"] == created["agent_id"]
    assert store.resolve_api_key(created["api_key"]) is None  # old key invalidated
    assert store.rotate_api_key("missing") is None


def test_claim_agent_and_browser_claim(store):
    created = store.create_agent(name="A", owner_browser_session="b1")
    store.claim_agent(created["agent_id"], owner_user_id=42)
    assert store.get_agent(created["agent_id"])["owner_user_id"] == 42

    other = store.create_agent(name="B", owner_browser_session="b2")
    count = store.claim_browser_agents_to_user("b2", 99)
    assert count == 1
    assert store.get_agent(other["agent_id"])["owner_user_id"] == 99
    assert store.claim_browser_agents_to_user("", 1) == 0


def test_delete_agent(store):
    created = store.create_agent(name="A")
    assert store.delete_agent(created["agent_id"]) is True
    assert store.get_agent(created["agent_id"]) is None
    assert store.delete_agent(created["agent_id"]) is False


def test_owns_agent(store):
    created = store.create_agent(name="A", owner_user_id=5, owner_browser_session="bz")
    stored = store.get_agent(created["agent_id"])
    assert store.owns_agent(stored, owner_user_id=5) is True
    assert store.owns_agent(stored, owner_user_id=6) is False
    # session_id is NOT an ownership credential — it is discoverable, so matching
    # it must never grant ownership (regression guard for the takeover bug).
    assert store.owns_agent(stored, owner_browser_session=stored["session_id"]) is False
    # The real owner_browser_session IS recognized — owns_agent reads it straight
    # from the row (the public agent dict omits it as a private credential).
    assert store.owns_agent(stored, owner_browser_session="bz") is True
