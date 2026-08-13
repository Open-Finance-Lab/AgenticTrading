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
        "runtime_type", "runtime_config",
        "description", "pipeline", "cash_allocation", "backtest_allocation",
        "api_key_prefix", "owner_user_id", "scopes",
        "created_at", "last_used_at", "api_key", "live_trading_enabled",
        "category",
    }
    assert agent["name"] == "My Agent"
    assert agent["model_name"] == "gpt-x"
    assert agent["owner_user_id"] == 7
    assert agent["agent_id"].startswith("agent_")
    assert agent["api_key"].startswith("ag_")
    assert agent["api_key_prefix"] == agent["api_key"][:12]
    # api_key_hash must never leak in the public dict.
    assert "api_key_hash" not in agent
    assert agent["runtime_type"] == "pipeline"
    assert agent["runtime_config"] == {}


def test_legacy_agent_schema_migrates_runtime_defaults(tmp_path):
    import sqlite3

    db_path = tmp_path / "legacy-agents.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE external_agents (
            agent_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            session_id TEXT NOT NULL UNIQUE,
            api_key_hash TEXT NOT NULL UNIQUE,
            api_key_prefix TEXT NOT NULL,
            model_name TEXT NOT NULL DEFAULT 'local-model',
            owner_user_id INTEGER,
            owner_browser_session TEXT,
            created_at TEXT,
            last_used_at TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO external_agents (
            agent_id, name, session_id, api_key_hash, api_key_prefix
        ) VALUES ('agent_legacy', 'Legacy', 'legacy-session', 'hash', 'ag_legacy')
        """
    )
    conn.commit()
    conn.close()

    migrated = AgentStore(db_path=db_path).get_agent("agent_legacy")

    assert migrated["runtime_type"] == "pipeline"
    assert migrated["runtime_config"] == {}


def test_agent_row_carries_category(store):
    agent = store.create_agent(name="Cat Test", agent_type="builtin", category="us_stocks")
    assert agent["category"] == "us_stocks"


def test_category_defaults_to_none_and_survives_update_clear(store):
    agent = store.create_agent(name="Cat Default")
    assert agent["category"] is None
    store.update_agent(agent["agent_id"], category="cn_ashares")
    assert store.get_agent(agent["agent_id"])["category"] == "cn_ashares"
    store.update_agent(agent["agent_id"], category=None)
    assert store.get_agent(agent["agent_id"])["category"] is None


def test_category_column_added_to_preexisting_table(tmp_path):
    import sqlite3

    db_path = tmp_path / "legacy-category-agents.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE external_agents (
            agent_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            session_id TEXT NOT NULL UNIQUE,
            api_key_hash TEXT NOT NULL UNIQUE,
            api_key_prefix TEXT NOT NULL,
            model_name TEXT NOT NULL DEFAULT 'local-model',
            owner_user_id INTEGER,
            owner_browser_session TEXT,
            created_at TEXT,
            last_used_at TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO external_agents (
            agent_id, name, session_id, api_key_hash, api_key_prefix
        ) VALUES ('agent_legacy_cat', 'Legacy', 'legacy-cat-session', 'hash', 'ag_legacy')
        """
    )
    conn.commit()
    conn.close()

    migrated = AgentStore(db_path=db_path).get_agent("agent_legacy_cat")

    assert migrated["category"] is None


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

    # trading_session_id alone is not enough — must also match ownership.
    assert store.list_agents(trading_session_id=a_user["session_id"]) == []
    by_session = store.list_agents(
        owner_user_id=1, trading_session_id=a_user["session_id"]
    )
    assert [a["agent_id"] for a in by_session] == [a_user["agent_id"]]

    assert store.list_agents() == []


def test_anonymous_list_hides_account_bound_browser_agents(store):
    """Logout must not re-surface agents the previous signed-in user created."""
    bound = store.create_agent(
        name="Bound", owner_user_id=1, owner_browser_session="b1"
    )
    guest = store.create_agent(name="Guest", owner_browser_session="b1")
    listed = store.list_agents(owner_browser_session="b1")
    ids = {a["agent_id"] for a in listed}
    assert guest["agent_id"] in ids
    assert bound["agent_id"] not in ids


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


def test_register_or_get_agent_does_not_steal_another_account(store):
    """The fourth writer of owner_user_id.

    register_or_get_agent is reachable from POST /agents/import-session, whose
    session_id is read straight off a caller-supplied header. A bare
    ``COALESCE(?, owner_user_id)`` takes the first non-null value, so any
    signed-in caller could re-own (and rename) another account's agent.
    """
    created = store.create_agent(
        name="Owned", owner_user_id=1, owner_browser_session="b-owner"
    )
    store.register_or_get_agent(
        session_id=created["session_id"],
        name="Renamed by thief",
        model_name="thief-model",
        owner_user_id=2,
        owner_browser_session="b-thief",
    )
    row = store.get_agent(created["agent_id"])
    assert row["owner_user_id"] == 1
    assert row["name"] == "Owned"

    # The real owner still gets the idempotent name/model refresh.
    store.register_or_get_agent(
        session_id=created["session_id"],
        name="Renamed by owner",
        model_name="m",
        owner_user_id=1,
    )
    assert store.get_agent(created["agent_id"])["name"] == "Renamed by owner"


def test_list_agents_session_fold_keeps_import_session_rows(store):
    """Regression: the anonymous session fold must not require a browser match.

    import-session rows are stamped owner_browser_session = session_id, so a
    browser that later sends its own X-Browser-Id matches neither the browser
    fold nor a browser-equality session fold -- the agent vanishes from
    My Agents even though its owner is looking right at it.
    """
    imported = store.register_or_get_agent(
        session_id="sess-import", name="I", owner_browser_session="sess-import"
    )
    listed = store.list_agents(
        owner_browser_session="browser-abc", trading_session_id="sess-import"
    )
    assert [a["agent_id"] for a in listed] == [imported["agent_id"]]

    # Still no leak: another browser's guest agent stays hidden even when its
    # session id is known, and account-bound rows never appear anonymously.
    other = store.create_agent(name="O", owner_browser_session="b-other")
    assert store.list_agents(
        owner_browser_session="browser-abc", trading_session_id=other["session_id"]
    ) == []
    bound = store.create_agent(name="Bound", owner_user_id=9, owner_browser_session="sess-b")
    assert store.list_agents(
        owner_browser_session="browser-abc", trading_session_id=bound["session_id"]
    ) == []


def test_claim_agent_does_not_steal_another_account(store):
    created = store.create_agent(
        name="A", owner_user_id=1, owner_browser_session="b1"
    )
    store.claim_agent(
        created["agent_id"], owner_user_id=2, owner_browser_session="b1"
    )
    assert store.get_agent(created["agent_id"])["owner_user_id"] == 1


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
    # Once bound to an account, browser id alone must not grant access — that is
    # how logout / a later login on the same machine took over another user's agents.
    assert store.owns_agent(stored, owner_browser_session="bz") is False

    guest = store.create_agent(name="G", owner_browser_session="guest-b")
    assert store.owns_agent(guest, owner_browser_session="guest-b") is True
    assert store.owns_agent(guest, owner_user_id=5) is False
