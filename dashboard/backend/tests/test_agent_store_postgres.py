"""PostgresAgentStore / PostgresAgentVersionStore tests.

Two tiers, mirroring test_users_postgres.py:
1. Dispatch-logic tests (no live Postgres needed) - verify the module
   factories pick the right store class based on CONTENT_DATABASE_URL.
2. Behavioral tests against a real Postgres - skipped unless
   TEST_POSTGRES_URL is set. Point it at a throwaway database, e.g.:
     docker run --rm -e POSTGRES_PASSWORD=test -e POSTGRES_DB=atl_test \
       -p 5433:5432 postgres:18-alpine
     export TEST_POSTGRES_URL=postgresql://postgres:test@localhost:5433/atl_test

Do NOT copy the raw-SQL fixture pattern from test_v2_http_runs.py /
test_v2_auth.py (SQLite-only `?` placeholders); use public store methods.
"""

import os

import pytest

from dashboard.backend.tests._postgres_testing import require_local_postgres_url

TEST_POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")

pg_only = pytest.mark.skipif(
    not TEST_POSTGRES_URL,
    reason="TEST_POSTGRES_URL not set; skipping live-Postgres tests",
)


# --- dispatch tests (agent store) -------------------------------------------

def test_build_agent_store_defaults_to_sqlite(monkeypatch, capsys):
    import dashboard.backend.domain.agents.repository as repo_module

    monkeypatch.delenv("CONTENT_DATABASE_URL", raising=False)
    store = repo_module._build_agent_store()
    assert isinstance(store, repo_module.AgentStore)
    assert "agent_store backend: sqlite (ephemeral on Render)" in capsys.readouterr().out


def test_build_agent_store_picks_postgres_when_url_set(monkeypatch, capsys):
    import dashboard.backend.domain.agents.repository as repo_module
    import dashboard.backend.domain.agents.repository_postgres as repo_pg_module

    created = {}

    class FakePostgresAgentStore:
        def __init__(self, database_url):
            created["database_url"] = database_url

    monkeypatch.setattr(repo_pg_module, "PostgresAgentStore", FakePostgresAgentStore)
    monkeypatch.setenv("CONTENT_DATABASE_URL", "postgresql://fake/db")

    store = repo_module._build_agent_store()

    assert isinstance(store, FakePostgresAgentStore)
    assert created["database_url"] == "postgresql://fake/db"
    # capsys (the factory print()s) and the target is named -- see Task 3.
    assert "agent_store backend: postgres (fake/db)" in capsys.readouterr().out


def test_build_agent_store_ignores_users_database_url(monkeypatch, capsys):
    """The mirror of test_users_postgres.py's ignores_content_database_url.

    The two vars are scoped per store and neither falls back to the other (spec,
    Decision 2). That was guarded in one direction only -- nothing stopped a
    future "convenience" fallback from quietly binding agents to the *accounts*
    database, which is exactly the kind of one-line change that reads like an
    improvement and keeps the suite green.
    """
    import dashboard.backend.domain.agents.repository as repo_module

    monkeypatch.delenv("CONTENT_DATABASE_URL", raising=False)
    monkeypatch.setenv("USERS_DATABASE_URL", "postgresql://fake/users")

    store = repo_module._build_agent_store()

    assert isinstance(store, repo_module.AgentStore)
    assert "agent_store backend: sqlite (ephemeral on Render)" in capsys.readouterr().out


def test_build_agent_store_never_prints_the_credentials(monkeypatch, capsys):
    import dashboard.backend.domain.agents.repository as repo_module
    import dashboard.backend.domain.agents.repository_postgres as repo_pg_module

    class FakePostgresAgentStore:
        def __init__(self, database_url):
            pass

    monkeypatch.setattr(repo_pg_module, "PostgresAgentStore", FakePostgresAgentStore)
    monkeypatch.setenv("CONTENT_DATABASE_URL", "postgresql://admin:sup3r-s3cret@host/db")

    repo_module._build_agent_store()

    out = capsys.readouterr().out
    assert "sup3r-s3cret" not in out
    assert "agent_store backend: postgres (host/db)" in out


def test_unreachable_postgres_agent_store_raises_instead_of_falling_back():
    """Fail loud: a set-but-unreachable URL must not silently degrade to SQLite.

    The only tier that runs PostgresAgentStore.__init__ (and therefore its
    _init_schema DDL path) without a live server -- the dispatch tests above
    monkeypatch the class away. A closed port refuses instantly; connect_timeout
    stops a DROP-style firewall from hanging the suite.
    """
    import psycopg

    import dashboard.backend.domain.agents.repository_postgres as repo_pg

    with pytest.raises(psycopg.OperationalError):
        repo_pg.PostgresAgentStore("postgresql://u:p@127.0.0.1:1/nope?connect_timeout=2")


def test_malformed_url_is_rejected_before_psycopg_can_echo_it():
    """A typo'd CONTENT_DATABASE_URL must not put the password in the log.

    psycopg parses anything not starting with postgresql:// as a keyword DSN and
    quotes the whole input back ('missing "=" after "<the entire URL>"'). This
    runs at import time with no try/except, so that message is the boot failure
    and it lands in Render's log. require_postgres_url must therefore be wired
    into __init__ -- testing the helper alone would not catch it being dropped
    from the constructor.
    """
    import dashboard.backend.domain.agents.repository_postgres as repo_pg

    with pytest.raises(ValueError) as excinfo:
        repo_pg.PostgresAgentStore('"postgresql://u:sup3r-s3cret@ep-x.neon.tech/atl"')
    assert "sup3r-s3cret" not in str(excinfo.value)


# --- live-Postgres behavioral tests (agent store) ---------------------------

@pytest.fixture
def pg_agent_store():
    require_local_postgres_url(TEST_POSTGRES_URL)
    import dashboard.backend.domain.agents.repository_postgres as repo_pg

    store = repo_pg.PostgresAgentStore(TEST_POSTGRES_URL)
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM external_agents")
    yield store


@pg_only
def test_agent_key_lifecycle_postgres(pg_agent_store):
    created = pg_agent_store.create_agent(
        name="PG Agent", owner_browser_session="bs_1", description="hello"
    )
    assert created["agent_id"].startswith("agent_")
    assert created["api_key"].startswith("ag_")
    assert created["api_key_prefix"] == created["api_key"][:12]

    resolved = pg_agent_store.resolve_api_key(created["api_key"])
    assert resolved is not None
    assert resolved["agent_id"] == created["agent_id"]
    assert resolved["last_used_at"] is not None

    new_key = pg_agent_store.rotate_api_key(created["agent_id"])
    assert new_key is not None and new_key != created["api_key"]
    assert pg_agent_store.resolve_api_key(created["api_key"]) is None
    assert pg_agent_store.resolve_api_key(new_key)["agent_id"] == created["agent_id"]

    assert pg_agent_store.rotate_api_key("agent_missing") is None
    assert pg_agent_store.resolve_api_key("") is None


@pg_only
def test_browser_claim_and_ownership_postgres(pg_agent_store):
    created = pg_agent_store.create_agent(name="Claimable", owner_browser_session="bs_2")
    assert pg_agent_store.owns_agent(created, owner_browser_session="bs_2") is True
    assert pg_agent_store.owns_agent(created, owner_user_id=42) is False

    claimed = pg_agent_store.claim_browser_agents_to_user("bs_2", user_id=42)
    assert claimed == 1
    assert pg_agent_store.owns_agent(created, owner_user_id=42) is True

    listed = pg_agent_store.list_agents(owner_user_id=42)
    assert [a["agent_id"] for a in listed] == [created["agent_id"]]
    assert listed[0]["owner_user_id"] == 42


@pg_only
def test_list_agents_signed_in_includes_unclaimed_browser_agents_postgres(
    pg_agent_store,
):
    """Postgres mirror of the SQLite regression test in test_repository_move.py.

    #235 added an unclaimed-browser-session fallback to list_agents so a
    signed-in user still sees a guest-provisioned agent whose claim hasn't
    landed yet. The SQLite twin had a test for the exclusion boundaries
    (other user's claimed agent, other browser's agent); this store's twin
    -- the one CONTENT_DATABASE_URL actually points prod at -- had none.
    """
    owned = pg_agent_store.create_agent(
        name="Owned", owner_user_id=7, owner_browser_session="b1"
    )
    unclaimed = pg_agent_store.create_agent(name="Guest", owner_browser_session="b1")
    other_user = pg_agent_store.create_agent(
        name="Other", owner_user_id=99, owner_browser_session="b1"
    )
    other_browser = pg_agent_store.create_agent(name="Elsewhere", owner_browser_session="b2")

    listed = pg_agent_store.list_agents(owner_user_id=7, owner_browser_session="b1")
    ids = {a["agent_id"] for a in listed}
    assert owned["agent_id"] in ids
    assert unclaimed["agent_id"] in ids
    assert other_user["agent_id"] not in ids
    assert other_browser["agent_id"] not in ids


@pg_only
def test_list_agents_merges_owned_and_unclaimed_by_recency_postgres(
    pg_agent_store, monkeypatch
):
    """The owned-rows and unclaimed-browser-rows groups must merge into one
    global ORDER BY, not just be independently sorted within each group.

    Without the merge, a browser agent created after every owned agent would
    still sort last, because list_agents appends the unclaimed-browser group
    only after the owned group is fully built.
    """
    import itertools

    import dashboard.backend.domain.agents.repository_postgres as repo_pg

    day = itertools.count(1)
    monkeypatch.setattr(
        repo_pg, "_utcnow_iso", lambda: f"2020-01-{next(day):02d}T00:00:00+00:00"
    )

    older_owned = pg_agent_store.create_agent(
        name="Older Owned", owner_user_id=7, owner_browser_session="b1"
    )
    newer_unclaimed = pg_agent_store.create_agent(
        name="Newer Guest", owner_browser_session="b1"
    )

    listed = pg_agent_store.list_agents(owner_user_id=7, owner_browser_session="b1")
    assert [a["agent_id"] for a in listed] == [
        newer_unclaimed["agent_id"],
        older_owned["agent_id"],
    ]


@pg_only
def test_register_or_get_agent_is_idempotent_postgres(pg_agent_store):
    first = pg_agent_store.register_or_get_agent(session_id="sess-1", name="A")
    again = pg_agent_store.register_or_get_agent(session_id="sess-1", name="A renamed")
    assert again["agent_id"] == first["agent_id"]
    assert again["name"] == "A renamed"
    assert pg_agent_store.get_agent_by_session("sess-1")["agent_id"] == first["agent_id"]


@pg_only
def test_update_agent_partial_updates_postgres(pg_agent_store):
    created = pg_agent_store.create_agent(name="Updatable")

    updated = pg_agent_store.update_agent(
        created["agent_id"], name="Renamed", pipeline=[{"presetKey": "news"}]
    )
    assert updated["name"] == "Renamed"
    assert updated["pipeline"] == [{"presetKey": "news"}]

    # Omitted kwargs (the _UNSET sentinel) must leave stored fields untouched.
    updated2 = pg_agent_store.update_agent(created["agent_id"], description="desc only")
    assert updated2["description"] == "desc only"
    assert updated2["pipeline"] == [{"presetKey": "news"}]

    # Explicit None clears the pipeline.
    updated3 = pg_agent_store.update_agent(created["agent_id"], pipeline=None)
    assert updated3["pipeline"] is None

    # No kwargs at all returns the current record unchanged.
    same = pg_agent_store.update_agent(created["agent_id"])
    assert same["name"] == "Renamed"

    assert pg_agent_store.update_agent("agent_missing", name="X") is None


@pg_only
def test_update_agent_live_trading_enabled_postgres(pg_agent_store):
    """#227 regression: live_trading_enabled reached only the SQLite twin.

    The missing kwarg TypeError'd EVERY update_agent call on prod (the service
    always passes it, as _UNSET), so any Configure save — rename-only included
    — 500'd. Round-trip the flag against a real Postgres: default False, on,
    off, and untouched by an unrelated update.
    """
    created = pg_agent_store.create_agent(name="Live Toggle PG")
    assert created["live_trading_enabled"] is False

    enabled = pg_agent_store.update_agent(created["agent_id"], live_trading_enabled=True)
    assert enabled["live_trading_enabled"] is True

    # Omitting the kwarg (service sends _UNSET) leaves the stored flag alone.
    renamed = pg_agent_store.update_agent(created["agent_id"], name="Renamed Live")
    assert renamed["live_trading_enabled"] is True

    disabled = pg_agent_store.update_agent(created["agent_id"], live_trading_enabled=False)
    assert disabled["live_trading_enabled"] is False


@pg_only
def test_update_agent_category_postgres(pg_agent_store):
    """Same shape as the #227 regression above, for the shelf slug.

    Postgres is what prod runs, and the twin's ``update_agent`` builds its SET
    clause independently of the SQLite one -- a category branch that exists only
    on the SQLite side would pass every default-tier test and then drop every
    shelf change in production. Round-trip: default NULL, set, untouched by an
    unrelated update, cleared.
    """
    created = pg_agent_store.create_agent(name="Shelved PG", category="us_stocks")
    assert created["category"] == "us_stocks"

    moved = pg_agent_store.update_agent(created["agent_id"], category="cn_ashares")
    assert moved["category"] == "cn_ashares"

    # Omitting the kwarg (service sends _UNSET) leaves the stored shelf alone.
    renamed = pg_agent_store.update_agent(created["agent_id"], name="Renamed Shelved")
    assert renamed["category"] == "cn_ashares"

    cleared = pg_agent_store.update_agent(created["agent_id"], category=None)
    assert cleared["category"] is None

    unshelved = pg_agent_store.create_agent(name="Unshelved PG")
    assert unshelved["category"] is None


@pg_only
def test_update_agent_model_name_postgres(pg_agent_store):
    created = pg_agent_store.create_agent(
        name="Model Swap PG",
        model_name="anthropic/claude-haiku-4-5",
        owner_user_id=None,
        owner_browser_session="pg-model-swap",
    )
    updated = pg_agent_store.update_agent(
        created["agent_id"], model_name="deepseek/deepseek-v4-pro"
    )
    assert updated["model_name"] == "deepseek/deepseek-v4-pro"

    # Omitting model_name leaves it unchanged.
    renamed = pg_agent_store.update_agent(created["agent_id"], name="Renamed")
    assert renamed["model_name"] == "deepseek/deepseek-v4-pro"


@pg_only
def test_cash_allocation_round_trips_as_a_float_postgres(pg_agent_store):
    """Pin the feature's only non-TEXT column against a real Postgres.

    Every other @pg_only test leaves cash_allocation NULL, yet service.py
    defaults it to float(DEFAULT_AGENT_CASH_ALLOCATION) when a caller omits it
    -- so every real agent registration writes a float on a column nothing here
    exercised. Declare it TEXT (the type the other 14 columns have) and the whole
    tier still passes; the first prod registration is what finds out.
    """
    created = pg_agent_store.create_agent(name="Funded", cash_allocation=25000.5)
    assert created["cash_allocation"] == 25000.5
    assert isinstance(created["cash_allocation"], float)

    assert pg_agent_store.get_agent(created["agent_id"])["cash_allocation"] == 25000.5

    updated = pg_agent_store.update_agent(created["agent_id"], cash_allocation=100.25)
    assert updated["cash_allocation"] == 100.25

    # _UNSET vs None: omitting it leaves the stored value, passing None clears it.
    assert pg_agent_store.update_agent(created["agent_id"], name="Renamed")[
        "cash_allocation"
    ] == 100.25
    assert pg_agent_store.update_agent(created["agent_id"], cash_allocation=None)[
        "cash_allocation"
    ] is None


@pg_only
def test_builtin_listing_and_delete_postgres(pg_agent_store):
    builtin = pg_agent_store.create_agent(name="Builtin", agent_type="builtin")
    external = pg_agent_store.create_agent(name="External")

    builtin_ids = [a["agent_id"] for a in pg_agent_store.list_builtin_agents()]
    assert builtin["agent_id"] in builtin_ids
    assert external["agent_id"] not in builtin_ids

    assert pg_agent_store.delete_agent(builtin["agent_id"]) is True
    assert pg_agent_store.delete_agent(builtin["agent_id"]) is False
    assert pg_agent_store.get_agent(builtin["agent_id"]) is None


@pg_only
def test_agent_schema_lazily_migrates_an_old_table_postgres(pg_agent_store):
    """#135: the twin must ADD COLUMN IF NOT EXISTS for post-ship columns.

    Simulate a deployment created before the seven lazy-migration columns
    (agent_type, description, pipeline_config, cash_allocation,
    backtest_allocation, scopes, live_trading_enabled, runtime_type,
    runtime_config) existed:
    a table with only the original base schema, already holding a real agent row
    that predates those columns. Re-running _init_schema() -- what every redeploy
    does -- must bring it up to the current shape. CREATE TABLE IF NOT EXISTS
    no-ops on the existing table, so without an ALTER path the columns never
    appear and the first real create_agent (which names them) raises
    UndefinedColumn: exactly the silent prod-500 this issue describes.

    The pre-existing row -- not the empty table -- is the risk surface worth
    asserting on. `scopes` is an authorization input, so a legacy row must emerge
    from the migration carrying the column DEFAULT; a NULL scopes would be an
    authz hole. That backfill is precisely what ADD COLUMN ... NOT NULL DEFAULT
    does for existing rows, and it is what this asserts.

    The legacy shape below is SYNTHETIC, not historical: the Postgres twin shipped
    in #134 with all five columns already folded into its CREATE TABLE, so no Neon
    deployment has ever had this table. What this pins is the ALTER statements
    themselves -- drop any one of them, or its NOT NULL DEFAULT, and this goes red.
    The forward risk #135 actually names -- a *sixth* column added to CREATE TABLE
    only, never reaching an existing deployment -- cannot be covered by a test
    written today; the ADDING A COLUMN LATER? comment in repository_postgres.py is
    what guards that, and it is the thing to keep alive.
    """
    import dashboard.backend.domain.agents.repository as repo_module
    import dashboard.backend.domain.agents.repository_postgres as repo_pg

    with pg_agent_store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS external_agents CASCADE")
            cur.execute(
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
            # A real agent registered against the pre-migration schema.
            cur.execute(
                """
                INSERT INTO external_agents (
                    agent_id, name, session_id, api_key_hash, api_key_prefix
                ) VALUES ('agent_legacy', 'Legacy', 'sess_legacy',
                          'hash_legacy', 'ag_legacy')
                """
            )

    # A redeploy re-runs _init_schema() against the existing "old" table.
    repo_pg.PostgresAgentStore(TEST_POSTGRES_URL)

    with pg_agent_store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'external_agents'"
            )
            columns = {r["column_name"] for r in cur.fetchall()}
            # Read the raw columns straight from the legacy row: _public_agent
            # masks a NULL/empty scopes back to DEFAULT_SCOPES, so only the stored
            # value proves the ADD COLUMN carried its NOT NULL DEFAULT for the
            # pre-existing row rather than leaving it NULL.
            cur.execute(
                "SELECT agent_type, scopes FROM external_agents "
                "WHERE agent_id = 'agent_legacy'"
            )
            legacy = cur.fetchone()
    assert {
        "agent_type",
        "description",
        "pipeline_config",
        "cash_allocation",
        "backtest_allocation",
        "scopes",
        "live_trading_enabled",
        "runtime_type",
        "runtime_config",
        "category",
    } <= columns
    # The pre-existing row was backfilled with the column defaults, not NULL.
    assert legacy["agent_type"] == "external"
    assert legacy["scopes"] == repo_module.DEFAULT_SCOPES

    migrated = pg_agent_store.get_agent("agent_legacy")
    assert migrated["runtime_type"] == "pipeline"
    assert migrated["runtime_config"] == {}

    # The store is actually usable after the migration: a real registration
    # writes every migrated column, and the scopes column default applies.
    created = pg_agent_store.create_agent(
        name="Post-migration", cash_allocation=123.0, backtest_allocation=123.0
    )
    assert created["cash_allocation"] == 123.0
    assert created["backtest_allocation"] == 123.0


@pg_only
def test_claim_and_reclaim_agent_postgres(pg_agent_store):
    """#137 gap 1: claim_agent / reclaim_agent had zero Postgres coverage.

    claim_agent COALESCEs owner_browser_session (a None arg keeps the stored
    one); reclaim_agent overwrites it unconditionally. Swapping those two
    semantics is the mutation this pins.
    """
    created = pg_agent_store.create_agent(
        name="Claimable2", owner_browser_session="bs_old"
    )

    pg_agent_store.claim_agent(created["agent_id"], owner_user_id=7)
    assert pg_agent_store.owns_agent(created, owner_user_id=7) is True
    # Bound to a user: browser id alone must not grant ownership.
    assert pg_agent_store.owns_agent(created, owner_browser_session="bs_old") is False

    pg_agent_store.reclaim_agent(
        created["agent_id"], owner_user_id=7, owner_browser_session="bs_new"
    )
    # reclaim refreshed the browser stamp for the same user; browser-only
    # access stays denied while the agent remains account-bound.
    assert pg_agent_store.owns_agent(created, owner_browser_session="bs_new") is False
    assert pg_agent_store.owns_agent(created, owner_browser_session="bs_old") is False
    assert pg_agent_store.owns_agent(created, owner_user_id=7) is True


@pg_only
def test_claim_reclaim_accept_null_owner_user_id_postgres(pg_agent_store):
    """A logged-out caller passes owner_user_id=None through both writers.

    psycopg sends None with an unspecified type OID, so a bare
    ``%s IS NOT NULL`` in the no-steal guard cannot be type-inferred and
    Postgres aborts the statement with 42P08 ("could not determine data type of
    parameter"). Every anonymous activate/restore takes this path, so the
    uncast form 500s the common case in prod while the SQLite twin -- whose
    parameters are untyped -- stays green. Only this tier can see it.
    """
    created = pg_agent_store.create_agent(name="Anon", owner_browser_session="bs_old")

    # No user id anywhere: the guard must still parse and still bind nothing.
    pg_agent_store.reclaim_agent(created["agent_id"], owner_browser_session="bs_new")
    assert pg_agent_store.owns_agent(created, owner_browser_session="bs_new") is True
    assert pg_agent_store.get_agent(created["agent_id"])["owner_user_id"] is None

    pg_agent_store.claim_agent(created["agent_id"], owner_browser_session="bs_third")
    assert pg_agent_store.owns_agent(created, owner_browser_session="bs_third") is True
    assert pg_agent_store.get_agent(created["agent_id"])["owner_user_id"] is None


@pg_only
def test_register_or_get_agent_does_not_steal_account_postgres(pg_agent_store):
    """import-session must not re-own an agent bound to a different account."""
    created = pg_agent_store.create_agent(
        name="Owned", owner_user_id=1, owner_browser_session="bs_owner"
    )
    pg_agent_store.register_or_get_agent(
        session_id=created["session_id"],
        name="Renamed",
        model_name="thief-model",
        owner_user_id=2,
        owner_browser_session="bs_thief",
    )
    row = pg_agent_store.get_agent(created["agent_id"])
    assert row["owner_user_id"] == 1
    assert row["name"] == "Owned"

    # The real owner still gets the idempotent update.
    pg_agent_store.register_or_get_agent(
        session_id=created["session_id"],
        name="Renamed by owner",
        model_name="m",
        owner_user_id=1,
    )
    assert pg_agent_store.get_agent(created["agent_id"])["name"] == "Renamed by owner"


@pg_only
def test_create_agent_stores_default_scopes_verbatim_postgres(pg_agent_store):
    """#137 gap 2: scopes is an authorization surface with no @pg_only assertion.

    create_agent omits scopes from its INSERT, so the column DEFAULT fills it.
    Read the raw column, not the public dict: _public_agent does
    ``data.get("scopes") or DEFAULT_SCOPES``, so an empty/dropped default is
    masked back to DEFAULT_SCOPES in the public view -- only the stored value
    proves the column default itself is right.
    """
    import dashboard.backend.domain.agents.repository as repo_module

    created = pg_agent_store.create_agent(name="Scoped")  # no scopes passed
    with pg_agent_store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT scopes FROM external_agents WHERE agent_id = %s",
                (created["agent_id"],),
            )
            raw = cur.fetchone()["scopes"]
    assert raw == repo_module.DEFAULT_SCOPES


@pg_only
def test_resolve_api_key_touches_last_used_at_postgres(pg_agent_store, monkeypatch):
    """#137 gap 3: touch-on-use was only asserted `is not None`.

    create_agent already sets last_used_at non-None, so a dropped touch-on-use
    UPDATE in resolve_api_key would go uncaught. Force a distinct, strictly-later
    timestamp so the assertion can't hide behind 1-second timestamp resolution
    (create and resolve otherwise land in the same second).
    """
    import dashboard.backend.domain.agents.repository_postgres as repo_pg

    created = pg_agent_store.create_agent(name="Toucher")
    before = pg_agent_store.get_agent(created["agent_id"])["last_used_at"]

    monkeypatch.setattr(repo_pg, "_utcnow_iso", lambda: "2099-01-01T00:00:00+00:00")
    pg_agent_store.resolve_api_key(created["api_key"])

    after = pg_agent_store.get_agent(created["agent_id"])["last_used_at"]
    assert after == "2099-01-01T00:00:00+00:00"
    assert after != before


@pg_only
def test_listings_order_newest_first_postgres(pg_agent_store, monkeypatch):
    """#137 gap 4: nothing exercised ORDER BY on list_agents / list_builtin_agents.

    Both queries are ``ORDER BY created_at DESC``; an ASC/DESC swap in either was
    invisible. Feed monotonically increasing created_at values -- 1s resolution
    would otherwise make same-second ties non-deterministic.
    """
    import itertools

    import dashboard.backend.domain.agents.repository_postgres as repo_pg

    # Unbounded monotonic clock: robust if create_agent ever calls _utcnow_iso a
    # different number of times (a fixed-length iterator would raise an opaque
    # StopIteration instead of a clean assertion failure).
    day = itertools.count(1)
    monkeypatch.setattr(
        repo_pg, "_utcnow_iso", lambda: f"2020-01-{next(day):02d}T00:00:00+00:00"
    )

    a = pg_agent_store.create_agent(name="A", owner_user_id=99)
    b = pg_agent_store.create_agent(name="B", owner_user_id=99)
    c = pg_agent_store.create_agent(name="C", owner_user_id=99)
    assert [x["agent_id"] for x in pg_agent_store.list_agents(owner_user_id=99)] == [
        c["agent_id"],
        b["agent_id"],
        a["agent_id"],
    ]

    d = pg_agent_store.create_agent(name="D", agent_type="builtin")
    e = pg_agent_store.create_agent(name="E", agent_type="builtin")
    f = pg_agent_store.create_agent(name="F", agent_type="builtin")
    assert [x["agent_id"] for x in pg_agent_store.list_builtin_agents()] == [
        f["agent_id"],
        e["agent_id"],
        d["agent_id"],
    ]


# --- dispatch tests (agent version store) ------------------------------------

def test_build_agent_version_store_defaults_to_sqlite(monkeypatch, capsys):
    import dashboard.backend.domain.agents.version_repository as vrepo_module

    monkeypatch.delenv("CONTENT_DATABASE_URL", raising=False)
    store = vrepo_module._build_agent_version_store()
    assert isinstance(store, vrepo_module.AgentVersionStore)
    assert (
        "agent_version_store backend: sqlite (ephemeral on Render)"
        in capsys.readouterr().out
    )


def test_build_agent_version_store_picks_postgres_when_url_set(monkeypatch, capsys):
    import dashboard.backend.domain.agents.version_repository as vrepo_module
    import dashboard.backend.domain.agents.version_repository_postgres as vrepo_pg_module

    created = {}

    class FakePostgresAgentVersionStore:
        def __init__(self, database_url):
            created["database_url"] = database_url

    monkeypatch.setattr(
        vrepo_pg_module, "PostgresAgentVersionStore", FakePostgresAgentVersionStore
    )
    monkeypatch.setenv("CONTENT_DATABASE_URL", "postgresql://fake/db")

    store = vrepo_module._build_agent_version_store()

    assert isinstance(store, FakePostgresAgentVersionStore)
    assert created["database_url"] == "postgresql://fake/db"
    assert "agent_version_store backend: postgres (fake/db)" in capsys.readouterr().out


def test_build_agent_version_store_ignores_users_database_url(monkeypatch, capsys):
    """See the agent-store twin of this test above."""
    import dashboard.backend.domain.agents.version_repository as vrepo_module

    monkeypatch.delenv("CONTENT_DATABASE_URL", raising=False)
    monkeypatch.setenv("USERS_DATABASE_URL", "postgresql://fake/users")

    store = vrepo_module._build_agent_version_store()

    assert isinstance(store, vrepo_module.AgentVersionStore)
    assert (
        "agent_version_store backend: sqlite (ephemeral on Render)"
        in capsys.readouterr().out
    )


def test_unreachable_postgres_version_store_raises_instead_of_falling_back():
    """Fail loud — see the agent-store twin of this test above."""
    import psycopg

    import dashboard.backend.domain.agents.version_repository_postgres as vrepo_pg

    with pytest.raises(psycopg.OperationalError):
        vrepo_pg.PostgresAgentVersionStore("postgresql://u:p@127.0.0.1:1/nope?connect_timeout=2")


def test_malformed_url_is_rejected_before_psycopg_can_echo_it_version_store():
    """See the agent-store twin of this test above."""
    import dashboard.backend.domain.agents.version_repository_postgres as vrepo_pg

    with pytest.raises(ValueError) as excinfo:
        vrepo_pg.PostgresAgentVersionStore('"postgresql://u:sup3r-s3cret@ep-x.neon.tech/atl"')
    assert "sup3r-s3cret" not in str(excinfo.value)


# --- live-Postgres behavioral tests (agent version store) --------------------

@pytest.fixture
def pg_version_store():
    require_local_postgres_url(TEST_POSTGRES_URL)
    import dashboard.backend.domain.agents.version_repository_postgres as vrepo_pg

    store = vrepo_pg.PostgresAgentVersionStore(TEST_POSTGRES_URL)
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM agent_versions")
    yield store


@pg_only
def test_version_create_get_list_postgres(pg_version_store):
    v1 = pg_version_store.create_version(
        agent_id="agent_x",
        version="1.0",
        model_backbones=["claude-sonnet-5"],
        prompt="You are a trader.",
        config={"risk": "low"},
    )
    assert v1["agent_version_id"].startswith("agv_")
    assert v1["model_backbones"] == ["claude-sonnet-5"]
    # Hashes derived from raw prompt/config when not passed explicitly.
    assert v1["prompt_hash"] and len(v1["prompt_hash"]) == 16
    assert v1["config_hash"] and len(v1["config_hash"]) == 16

    v2 = pg_version_store.create_version(agent_id="agent_x", version="1.1")
    listed = pg_version_store.list_versions("agent_x")
    # Both creates can land in the same second (1s timestamp resolution), and
    # the tiebreak is agent_version_id DESC (random hex) — so assert membership
    # and count, not a specific order.
    assert {v["agent_version_id"] for v in listed} == {
        v1["agent_version_id"],
        v2["agent_version_id"],
    }
    assert len(listed) == 2

    fetched = pg_version_store.get_version(v1["agent_version_id"])
    assert fetched == v1
    assert pg_version_store.get_version("agv_missing") is None
    assert pg_version_store.list_versions("agent_other") == []
