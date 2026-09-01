"""Dispatch and live-Postgres tests for the Credits store twin."""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from dashboard.backend import db_pool
from dashboard.backend.domain.credits import repository as repo_module
from dashboard.backend.domain.credits import repository_postgres as pg_module
from dashboard.backend.domain.credits.repository_common import (
    GrantPoolInsufficientError,
    GrantReclaimExceedsAvailableError,
    IdempotencyConflictError,
    LLMReservationConflictError,
)
from dashboard.backend.tests._postgres_testing import require_local_postgres_url
from dashboard.backend.tests.domain.credits.test_grant_repository_contract import (
    assert_four_operations_are_signed_and_paired,
    assert_grant_mutations_leave_purchased_balance_unchanged,
)


TEST_POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")
MONTH_START_ISO = "2026-08-01T00:00:00+00:00"

pg_only = pytest.mark.skipif(
    not TEST_POSTGRES_URL,
    reason="TEST_POSTGRES_URL not set; skipping live-Postgres tests",
)


def test_build_credits_store_defaults_to_sqlite(monkeypatch, capsys):
    monkeypatch.delenv("USERS_DATABASE_URL", raising=False)
    store = repo_module._build_credits_store()

    assert isinstance(store, repo_module.CreditsStore)
    assert (
        "credits_store backend: sqlite (ephemeral on Render)" in capsys.readouterr().out
    )


def test_postgres_schema_declares_the_welcome_promotion_ledger():
    ddl = pg_module.CREDITS_POSTGRES_DDL

    assert "CREATE TABLE IF NOT EXISTS credit_promotion_grants" in ddl
    assert "campaign_key TEXT NOT NULL" in ddl
    assert "operation_id TEXT NOT NULL UNIQUE" in ddl
    assert "idempotency_key TEXT NOT NULL UNIQUE" in ddl
    assert "UNIQUE (campaign_key, user_id)" in ddl


def test_postgres_schema_allows_settlement_overage_and_migrates_legacy_check():
    assert "settled_micro <= reserved_micro" not in pg_module.CREDITS_POSTGRES_DDL
    assert (
        "pg_get_constraintdef(oid)"
        in pg_module.CREDITS_POSTGRES_GRANT_MIGRATION_DDL
    )
    assert "legacy_constraint" in pg_module.CREDITS_POSTGRES_GRANT_MIGRATION_DDL
    assert (
        "DROP CONSTRAINT IF EXISTS credit_llm_reservations_settled_micro_check"
        in pg_module.CREDITS_POSTGRES_GRANT_MIGRATION_DDL
    )
    assert (
        "credit_llm_reservations_settled_micro_nonnegative_check"
        in pg_module.CREDITS_POSTGRES_GRANT_MIGRATION_DDL
    )


def test_postgres_schema_tracks_provider_attempt_identity():
    assert "provider_id TEXT" in pg_module.CREDITS_POSTGRES_DDL
    assert "attempt_index INTEGER NOT NULL DEFAULT 0" in pg_module.CREDITS_POSTGRES_DDL
    assert (
        "credit_llm_reservations_logical_attempt_key"
        in pg_module.CREDITS_POSTGRES_DDL
    )
    assert "ARRAY['user_id', 'run_id', 'call_index']::name[]" in (
        pg_module.CREDITS_POSTGRES_GRANT_MIGRATION_DDL
    )


def test_postgres_provider_attempt_index_is_created_after_its_column():
    base_ddl = pg_module.CREDITS_POSTGRES_DDL
    migration_ddl = pg_module.CREDITS_POSTGRES_GRANT_MIGRATION_DDL

    assert "idx_credit_llm_reservations_run_status" not in base_ddl
    add_column = migration_ddl.index(
        "ADD COLUMN IF NOT EXISTS attempt_index INTEGER NOT NULL DEFAULT 0"
    )
    create_index = migration_ddl.index("idx_credit_llm_reservations_run_status")
    assert add_column < create_index


def test_build_credits_store_picks_postgres_from_users_url(monkeypatch, capsys):
    created = {}

    class FakePostgresCreditsStore:
        def __init__(self, database_url):
            created["database_url"] = database_url

    monkeypatch.setattr(pg_module, "PostgresCreditsStore", FakePostgresCreditsStore)
    monkeypatch.setenv("USERS_DATABASE_URL", "postgresql://fake/accounts")

    store = repo_module._build_credits_store()

    assert isinstance(store, FakePostgresCreditsStore)
    assert created["database_url"] == "postgresql://fake/accounts"
    assert "credits_store backend: postgres (fake/accounts)" in capsys.readouterr().out


def test_build_credits_store_ignores_other_database_urls(monkeypatch, capsys):
    monkeypatch.delenv("USERS_DATABASE_URL", raising=False)
    monkeypatch.setenv("CONTENT_DATABASE_URL", "postgresql://fake/content")
    monkeypatch.setenv("AGENT_RUNS_DATABASE_URL", "postgresql://fake/runs")

    store = repo_module._build_credits_store()

    assert isinstance(store, repo_module.CreditsStore)
    assert (
        "credits_store backend: sqlite (ephemeral on Render)" in capsys.readouterr().out
    )


def test_build_credits_store_never_prints_credentials(monkeypatch, capsys):
    class FakePostgresCreditsStore:
        def __init__(self, database_url):
            pass

    monkeypatch.setattr(pg_module, "PostgresCreditsStore", FakePostgresCreditsStore)
    monkeypatch.setenv(
        "USERS_DATABASE_URL",
        "postgresql://admin:sup3r-s3cret@host/accounts",
    )

    repo_module._build_credits_store()

    output = capsys.readouterr().out
    assert "sup3r-s3cret" not in output
    assert "credits_store backend: postgres (host/accounts)" in output


def test_malformed_url_is_rejected_without_echoing_credentials():
    with pytest.raises(ValueError) as excinfo:
        pg_module.PostgresCreditsStore(
            '"postgresql://u:sup3r-s3cret@ep-x.neon.tech/atl"'
        )
    assert "sup3r-s3cret" not in str(excinfo.value)


def test_unreachable_postgres_raises_instead_of_falling_back():
    with pytest.raises(psycopg.OperationalError):
        pg_module.PostgresCreditsStore(
            "postgresql://u:p@127.0.0.1:1/nope?connect_timeout=1"
        )


def _schema_url(database_url: str, schema: str) -> str:
    parts = urlsplit(database_url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


LEGACY_CREDITS_POSTGRES_DDL = """
CREATE TABLE credit_accounts (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'restricted')),
    created_at TEXT NOT NULL
);

CREATE TABLE credit_payment_orders (
    sequence BIGSERIAL PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    client_request_id TEXT NOT NULL,
    stripe_mode TEXT NOT NULL DEFAULT 'test',
    currency TEXT NOT NULL DEFAULT 'usd',
    amount_usd_cents BIGINT NOT NULL,
    credits_micro BIGINT NOT NULL,
    status TEXT NOT NULL,
    stripe_checkout_session_id TEXT UNIQUE,
    stripe_payment_intent_id TEXT UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    paid_at TEXT,
    UNIQUE (user_id, client_request_id),
    FOREIGN KEY (user_id) REFERENCES credit_accounts(user_id) ON DELETE CASCADE
);

CREATE TABLE credit_refund_requests (
    sequence BIGSERIAL PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    payment_order_id TEXT NOT NULL
        REFERENCES credit_payment_orders(id) ON DELETE RESTRICT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    requested_by_user_id INTEGER REFERENCES users(id) ON DELETE RESTRICT,
    amount_usd_cents BIGINT NOT NULL,
    credits_micro BIGINT NOT NULL,
    status TEXT NOT NULL,
    stripe_refund_id TEXT UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    succeeded_at TEXT
);

CREATE TABLE stripe_webhook_events (
    stripe_event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    livemode BOOLEAN NOT NULL,
    object_id TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    outcome TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE credit_llm_reservations (
    reservation_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    run_id TEXT NOT NULL,
    call_index INTEGER NOT NULL,
    reserved_micro BIGINT NOT NULL,
    reserved_grant_micro BIGINT NOT NULL,
    reserved_purchased_micro BIGINT NOT NULL,
    settled_micro BIGINT NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',
    operation_key TEXT NOT NULL UNIQUE,
    request_digest TEXT NOT NULL,
    evidence_json TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (reserved_micro = reserved_grant_micro + reserved_purchased_micro),
    UNIQUE (user_id, run_id, call_index)
);

CREATE TABLE credit_ledger_entries (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    entry_type TEXT NOT NULL CHECK (entry_type IN ('purchase', 'refund')),
    amount_micro BIGINT NOT NULL CHECK (amount_micro <> 0),
    payment_order_id TEXT NOT NULL
        REFERENCES credit_payment_orders(id) ON DELETE RESTRICT,
    refund_request_id TEXT
        REFERENCES credit_refund_requests(id) ON DELETE RESTRICT,
    stripe_event_id TEXT NOT NULL
        REFERENCES stripe_webhook_events(stripe_event_id) ON DELETE RESTRICT,
    operation_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
"""


@pytest.fixture
def pg_credits_store():
    base_url = require_local_postgres_url(TEST_POSTGRES_URL)
    schema = f"credits_{uuid.uuid4().hex}"
    with psycopg.connect(base_url) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

    scoped_url = _schema_url(base_url, schema)
    try:
        with psycopg.connect(scoped_url) as conn:
            conn.execute(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO users (
                        id, email, display_name, password_hash, role, created_at
                    )
                    VALUES (%s, %s, %s, 'unused', %s, '2026-08-13T00:00:00+00:00')
                    """,
                    [
                        (1, "buyer@example.com", "Buyer", "user"),
                        (2, "admin@example.com", "Admin", "admin"),
                        (3, "other@example.com", "Other", "user"),
                    ],
                )

        yield pg_module.PostgresCreditsStore(scoped_url)
    finally:
        db_pool._reset_for_tests()
        with psycopg.connect(base_url) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


@pytest.fixture
def pg_legacy_credits_url():
    base_url = require_local_postgres_url(TEST_POSTGRES_URL)
    schema = f"credits_legacy_{uuid.uuid4().hex}"
    with psycopg.connect(base_url) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

    scoped_url = _schema_url(base_url, schema)
    try:
        with psycopg.connect(scoped_url) as conn:
            conn.execute(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO users (
                        id, email, display_name, password_hash, role, created_at
                    ) VALUES (%s, %s, %s, 'unused', %s, '2026-08-01T00:00:00+00:00')
                    """,
                    [
                        (1, "legacy@example.com", "Legacy", "user"),
                        (2, "admin@example.com", "Admin", "admin"),
                    ],
                )
            conn.execute(LEGACY_CREDITS_POSTGRES_DDL)
            conn.execute(
                """
                INSERT INTO credit_accounts (user_id, status, created_at)
                VALUES (1, 'active', '2026-08-01T00:00:00+00:00')
                """
            )
            conn.execute(
                """
                INSERT INTO credit_payment_orders (
                    id, user_id, client_request_id, stripe_mode, currency,
                    amount_usd_cents, credits_micro, status,
                    stripe_checkout_session_id, stripe_payment_intent_id,
                    created_at, updated_at, paid_at
                ) VALUES (
                    'ord_legacy', 1, 'legacy-request', 'test', 'usd',
                    1000, 10000000, 'partially_refunded',
                    'cs_legacy', 'pi_legacy',
                    '2026-08-01T00:00:00+00:00',
                    '2026-08-01T00:00:00+00:00',
                    '2026-08-01T00:00:00+00:00'
                )
                """
            )
            conn.execute(
                """
                INSERT INTO credit_refund_requests (
                    id, payment_order_id, user_id, requested_by_user_id,
                    amount_usd_cents, credits_micro, status, stripe_refund_id,
                    created_at, updated_at, succeeded_at
                ) VALUES (
                    'rfnd_legacy', 'ord_legacy', 1, 2,
                    400, 4000000, 'succeeded', 're_legacy',
                    '2026-08-02T00:00:00+00:00',
                    '2026-08-02T00:00:00+00:00',
                    '2026-08-02T00:00:00+00:00'
                )
                """
            )
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO stripe_webhook_events (
                        stripe_event_id, event_type, livemode, object_id,
                        payload_sha256, outcome, created_at
                    ) VALUES (%s, %s, FALSE, %s, %s, 'processed', %s)
                    """,
                    [
                        (
                            "evt_legacy_purchase",
                            "checkout.session.completed",
                            "cs_legacy",
                            "a" * 64,
                            "2026-08-01T00:00:00+00:00",
                        ),
                        (
                            "evt_legacy_refund",
                            "refund.updated",
                            "re_legacy",
                            "b" * 64,
                            "2026-08-02T00:00:00+00:00",
                        ),
                    ],
                )
                cur.executemany(
                    """
                    INSERT INTO credit_ledger_entries (
                        user_id, entry_type, amount_micro, payment_order_id,
                        refund_request_id, stripe_event_id, operation_key, created_at
                    ) VALUES (%s, %s, %s, 'ord_legacy', %s, %s, %s, %s)
                    """,
                    [
                        (
                            1,
                            "purchase",
                            10_000_000,
                            None,
                            "evt_legacy_purchase",
                            "stripe:legacy-purchase",
                            "2026-08-01T00:00:00+00:00",
                        ),
                        (
                            1,
                            "refund",
                            -4_000_000,
                            "rfnd_legacy",
                            "evt_legacy_refund",
                            "stripe:legacy-refund",
                            "2026-08-02T00:00:00+00:00",
                        ),
                    ],
                )

        yield scoped_url
    finally:
        db_pool._reset_for_tests()
        with psycopg.connect(base_url) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


def _pending_order(
    store,
    *,
    order_id: str = "ord_10",
    client_request_id: str = "11111111-1111-4111-8111-111111111111",
    cents: int = 1000,
):
    store.create_or_get_order(
        order_id=order_id,
        user_id=1,
        client_request_id=client_request_id,
        amount_usd_cents=cents,
        credits_micro=cents * 10_000,
    )
    return store.attach_checkout_session(
        order_id, checkout_session_id=f"cs_test_{order_id}"
    )


def _pay_order(
    store,
    *,
    order_id: str = "ord_10",
    event_id: str = "evt_paid",
    cents: int = 1000,
):
    return store.settle_paid_checkout(
        event_id=event_id,
        event_type="checkout.session.completed",
        livemode=False,
        object_id=f"cs_test_{order_id}",
        payload_sha256=event_id.ljust(64, "a"),
        order_id=order_id,
        checkout_session_id=f"cs_test_{order_id}",
        payment_intent_id=f"pi_test_{order_id}",
        currency="usd",
        amount_usd_cents=cents,
    )


def _grant_args(operation: str, *, amount_micro: int) -> dict[str, object]:
    return {
        "pool_id": "default",
        "amount_micro": amount_micro,
        "operation_id": f"pg_grant_{operation}",
        "idempotency_key": f"pg_request_{operation}",
        "request_digest": f"pg_digest_{operation}",
        "actor_user_id": 2,
        "source": "postgres_contract",
        "reason": f"Postgres contract operation {operation}.",
    }


@pg_only
def test_postgres_llm_overage_uses_unreserved_balance(pg_credits_store):
    store = pg_credits_store
    _pending_order(store, cents=110)
    _pay_order(store, cents=110)
    reservation = store.reserve_llm_credits(
        reservation_id="pg-partial-overage",
        user_id=1,
        run_id="pg-partial-overage-run",
        call_index=0,
        provider_id="openrouter",
        attempt_index=0,
        reserved_micro=1_000_000,
        operation_key="pg-partial-overage-operation",
        request_digest="p" * 64,
    )

    settled = store.settle_llm_credits(
        reservation["reservation_id"],
        actual_micro=1_250_000,
        evidence={"provider_id": "openrouter", "model_id": "qwen/qwen3"},
    )

    assert settled["settled_micro"] == 1_100_000
    assert settled["outstanding_micro"] == 150_000
    assert settled["purchased_debited_micro"] == 1_100_000
    assert store.get_account_billing_state(1)["account_status"] == "restricted"


@pg_only
def test_postgres_migration_drops_legacy_unnamed_settlement_ceiling(pg_credits_store):
    store = pg_credits_store
    with psycopg.connect(store.database_url) as conn:
        conn.execute(
            "ALTER TABLE credit_llm_reservations "
            "ADD CHECK (settled_micro <= reserved_micro)"
        )

    pg_module.PostgresCreditsStore(store.database_url)

    with psycopg.connect(store.database_url, row_factory=dict_row) as conn:
        constraints = conn.execute(
            """
            SELECT pg_get_constraintdef(oid) AS definition
            FROM pg_constraint
            WHERE conrelid = 'credit_llm_reservations'::regclass
              AND contype = 'c'
            """
        ).fetchall()

    assert not any(
        "settled_micro <= reserved_micro" in str(row["definition"])
        for row in constraints
    )

    with psycopg.connect(store.database_url) as conn:
        conn.execute(
            """
            INSERT INTO credit_llm_reservations (
                reservation_id, user_id, run_id, call_index, reserved_micro,
                reserved_grant_micro, reserved_purchased_micro, settled_micro,
                actual_micro, outstanding_micro, outstanding_recovered_micro,
                status, operation_key, request_digest, created_at, updated_at
            ) VALUES (
                'pg-migrated-overage-reservation', 1, 'pg-migrated-overage-run', 0,
                1000000, 1000000, 0, 0, 0, 0, 0, 'open',
                'pg-migrated-overage-operation', repeat('m', 64),
                '2026-09-01T00:00:00+00:00', '2026-09-01T00:00:00+00:00'
            )
            """
        )
        conn.execute(
            """
            UPDATE credit_llm_reservations
            SET settled_micro = 1100000,
                actual_micro = 1100000,
                status = 'settled'
            WHERE reservation_id = 'pg-migrated-overage-reservation'
            """
        )
        settled = conn.execute(
            """
            SELECT settled_micro, actual_micro, outstanding_micro
            FROM credit_llm_reservations
            WHERE reservation_id = 'pg-migrated-overage-reservation'
            """
        ).fetchone()

    assert int(settled[0]) == 1_100_000
    assert int(settled[1]) == 1_100_000
    assert int(settled[2]) == 0


@pg_only
def test_postgres_llm_overage_uses_grant_before_restricting(pg_credits_store):
    store = pg_credits_store
    store.fund_grant_pool(**_grant_args("covered_fund", amount_micro=2_000_000))
    store.assign_grant(
        user_id=1,
        **_grant_args("covered_assign", amount_micro=2_000_000),
    )
    reservation = store.reserve_llm_credits(
        reservation_id="pg-covered-overage",
        user_id=1,
        run_id="pg-covered-overage-run",
        call_index=0,
        provider_id="openrouter",
        attempt_index=0,
        reserved_micro=1_000_000,
        operation_key="pg-covered-overage-operation",
        request_digest="q" * 64,
    )

    settled = store.settle_llm_credits(
        reservation["reservation_id"],
        actual_micro=1_250_000,
        evidence={"provider_id": "openrouter", "model_id": "qwen/qwen3"},
    )

    assert settled["settled_micro"] == 1_250_000
    assert settled["outstanding_micro"] == 0
    assert settled["grant_debited_micro"] == 1_250_000
    assert store.get_account_billing_state(1)["account_status"] == "active"


@pg_only
def test_postgres_provider_attempts_share_logical_call_after_release(pg_credits_store):
    store = pg_credits_store
    _pending_order(store, cents=100)
    _pay_order(store, cents=100)
    primary = store.reserve_llm_credits(
        reservation_id="pg-attempt-primary",
        user_id=1,
        run_id="pg-attempt-run",
        call_index=4,
        attempt_index=0,
        provider_id="openrouter",
        reserved_micro=100_000,
        operation_key="pg-attempt-primary-operation",
        request_digest="a" * 64,
    )
    store.release_llm_credits(primary["reservation_id"], reason="provider_quota_exhausted")
    fallback = store.reserve_llm_credits(
        reservation_id="pg-attempt-fallback",
        user_id=1,
        run_id="pg-attempt-run",
        call_index=4,
        attempt_index=1,
        provider_id="commonstack",
        reserved_micro=100_000,
        operation_key="pg-attempt-fallback-operation",
        request_digest="c" * 64,
    )
    assert fallback["attempt_index"] == 1
    assert fallback["provider_id"] == "commonstack"

    with pytest.raises(LLMReservationConflictError):
        store.reserve_llm_credits(
            reservation_id="pg-attempt-fallback",
            user_id=1,
            run_id="pg-attempt-run",
            call_index=4,
            attempt_index=1,
            provider_id="openrouter",
            reserved_micro=100_000,
            operation_key="pg-attempt-fallback-operation",
            request_digest="c" * 64,
        )


@pg_only
def test_postgres_runs_shared_four_operation_contract(pg_credits_store):
    assert_four_operations_are_signed_and_paired(pg_credits_store)


@pg_only
def test_postgres_runs_shared_purchased_isolation_contract(pg_credits_store):
    assert_grant_mutations_leave_purchased_balance_unchanged(pg_credits_store)


@pg_only
def test_postgres_rejects_overdraft_and_restricted_assignment(pg_credits_store):
    store = pg_credits_store
    store.fund_grant_pool(**_grant_args("guard_fund", amount_micro=2_000_000))
    store.assign_grant(
        user_id=1,
        **_grant_args("guard_assign", amount_micro=1_000_000),
    )

    with pytest.raises(GrantPoolInsufficientError):
        store.reduce_grant_pool(
            **_grant_args("guard_overdraft", amount_micro=1_000_001)
        )
    store.restrict_account(1)
    with pytest.raises(repo_module.CreditAccountRestrictedStoreError):
        store.assign_grant(
            user_id=1,
            **_grant_args("guard_restricted", amount_micro=1_000_000),
        )

    reclaimed = store.reclaim_grant(
        user_id=1,
        **_grant_args("guard_reclaim", amount_micro=1_000_000),
    )
    assert reclaimed["user_balance"]["grant_available_micro"] == 0
    assert (
        store.get_grant_pool_summary("default", MONTH_START_ISO)["pool_available_micro"]
        == 2_000_000
    )


@pg_only
def test_postgres_batch_projection_and_activity_pagination(pg_credits_store):
    store = pg_credits_store
    store.fund_grant_pool(**_grant_args("batch_fund", amount_micro=10_000_000))
    store.assign_grant(
        user_id=1,
        **_grant_args("batch_assign_one", amount_micro=3_000_000),
    )
    store.assign_grant(
        user_id=3,
        **_grant_args("batch_assign_three", amount_micro=2_000_000),
    )

    projections = store.get_balance_projections([1, 3, 2, 1])
    assert list(projections) == [1, 3, 2]
    assert projections[1]["grant_available_micro"] == 3_000_000
    assert projections[3]["grant_available_micro"] == 2_000_000
    assert projections[2]["total_available_micro"] == 0

    first = store.list_grant_pool_activity("default", limit=2)
    second = store.list_grant_pool_activity(
        "default", limit=2, cursor=first["next_cursor"]
    )
    assert len(first["items"]) == 2
    assert len(second["items"]) == 1
    assert {item["id"] for item in first["items"]}.isdisjoint(
        item["id"] for item in second["items"]
    )


@pg_only
def test_postgres_activity_aggregates_calls_before_pagination(pg_credits_store):
    store = pg_credits_store
    _pending_order(store)
    _pay_order(store)
    for call_index, amount in enumerate((137, 1_147)):
        reservation_id = f"pg-activity:{call_index}"
        reservation = store.reserve_llm_credits(
            reservation_id=reservation_id,
            user_id=1,
            run_id="run-pg-activity",
            call_index=call_index,
            provider_id="openrouter",
            attempt_index=0,
            reserved_micro=amount,
            operation_key=f"reserve:{reservation_id}",
            request_digest=str(call_index).ljust(64, "b"),
        )
        store.settle_llm_credits(
            reservation["reservation_id"],
            actual_micro=amount,
            evidence={
                "billing_source": "platform_credits",
                "pricing_snapshot": {
                    "provider_id": "openrouter",
                    "model_id": "anthropic/claude-haiku-4-5",
                },
            },
        )

    first = store.list_ledger_entries(1, limit=1)
    second = store.list_ledger_entries(1, limit=1, cursor=first["next_cursor"])
    usage = next(
        item
        for item in [*first["items"], *second["items"]]
        if item["entry_type"] == "backtest_usage"
    )

    assert usage["amount_micro"] == -1_284
    assert usage["model_call_count"] == 2
    assert usage["provider_id"] == "openrouter"
    assert usage["model_id"] == "anthropic/claude-haiku-4-5"
    assert first["items"][0]["id"] != second["items"][0]["id"]


@pg_only
def test_postgres_grant_pair_rolls_back_on_second_insert_failure(
    pg_credits_store, monkeypatch
):
    store = pg_credits_store
    store.fund_grant_pool(**_grant_args("rollback_fund", amount_micro=3_000_000))

    def fail_pool_insert(*args, **kwargs):
        raise RuntimeError("injected pool insert failure")

    monkeypatch.setattr(
        store,
        "_insert_grant_pool_entry_in_transaction",
        fail_pool_insert,
    )
    with pytest.raises(RuntimeError, match="injected"):
        store.assign_grant(
            user_id=1,
            **_grant_args("rollback_assign", amount_micro=1_000_000),
        )

    assert store.get_balance_projection(1)["grant_committed_micro"] == 0
    assert (
        store.get_grant_pool_summary("default", MONTH_START_ISO)["pool_available_micro"]
        == 3_000_000
    )
    assert len(store.list_grant_pool_activity("default")["items"]) == 1


@pg_only
def test_postgres_idempotent_grant_replay_returns_original_result(pg_credits_store):
    store = pg_credits_store
    store.fund_grant_pool(**_grant_args("replay_fund", amount_micro=10_000_000))
    command = _grant_args("replay_assign", amount_micro=2_000_000)

    first = store.assign_grant(user_id=1, **command)
    store.assign_grant(
        user_id=1,
        **_grant_args("replay_later_assign", amount_micro=1_000_000),
    )
    with psycopg.connect(store.database_url) as conn:
        conn.execute(
            """
            UPDATE credit_grant_pools
            SET name = 'Renamed Pool', status = 'disabled'
            WHERE pool_id = 'default'
            """
        )
    replayed = store.assign_grant(user_id=1, **command)

    assert replayed == first
    current = store.get_grant_pool_summary("default", MONTH_START_ISO)
    assert current["pool_name"] == "Renamed Pool"
    assert current["pool_status"] == "disabled"
    assert store.get_balance_projection(1)["grant_committed_micro"] == 3_000_000
    with pytest.raises(IdempotencyConflictError):
        store.assign_grant(
            user_id=1,
            **{**command, "source": "different_source"},
        )


@pg_only
def test_postgres_concurrent_assign_cannot_overdraw_pool(pg_credits_store):
    store = pg_credits_store
    store.fund_grant_pool(**_grant_args("assign_race_fund", amount_micro=1_000_000))

    def assign(user_id: int) -> str:
        try:
            store.assign_grant(
                user_id=user_id,
                **_grant_args(f"assign_race_{user_id}", amount_micro=1_000_000),
            )
            return "assigned"
        except GrantPoolInsufficientError:
            return "insufficient"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(assign, (1, 3)))

    assert sorted(outcomes) == ["assigned", "insufficient"]
    summary = store.get_grant_pool_summary("default", MONTH_START_ISO)
    assert summary["pool_available_micro"] == 0
    assert summary["allocated_to_users_micro"] == 1_000_000


@pg_only
def test_postgres_concurrent_reclaim_cannot_overdraw_user_grant(pg_credits_store):
    store = pg_credits_store
    store.fund_grant_pool(**_grant_args("reclaim_race_fund", amount_micro=2_000_000))
    store.assign_grant(
        user_id=1,
        **_grant_args("reclaim_race_seed", amount_micro=1_000_000),
    )

    def reclaim(number: int) -> str:
        try:
            store.reclaim_grant(
                user_id=1,
                **_grant_args(f"reclaim_race_{number}", amount_micro=1_000_000),
            )
            return "reclaimed"
        except GrantReclaimExceedsAvailableError:
            return "insufficient"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(reclaim, (1, 2)))

    assert sorted(outcomes) == ["insufficient", "reclaimed"]
    assert store.get_balance_projection(1)["grant_available_micro"] == 0
    assert (
        store.get_grant_pool_summary("default", MONTH_START_ISO)["pool_available_micro"]
        == 2_000_000
    )


@pg_only
def test_postgres_grant_pool_summary_is_consistent_during_concurrent_mutations(
    pg_credits_store, monkeypatch
):
    store = pg_credits_store
    store.fund_grant_pool(**_grant_args("snapshot_fund", amount_micro=10_000_000))
    summary_executed = Event()
    assignment_committed = Event()
    original_get_connection = store._get_connection

    class PausingCursor:
        def __init__(self, cursor):
            self._cursor = cursor

        def __enter__(self):
            self._cursor.__enter__()
            return self

        def __exit__(self, *args):
            return self._cursor.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self._cursor, name)

        def execute(self, query, params=None):
            result = self._cursor.execute(query, params)
            if (
                isinstance(query, str)
                and "assigned_this_month_micro" in query
                and not summary_executed.is_set()
            ):
                summary_executed.set()
                assert assignment_committed.wait(timeout=5)
            return result

    class PausingConnection:
        def __init__(self, connection):
            self._connection = connection

        def __getattr__(self, name):
            return getattr(self._connection, name)

        def cursor(self, *args, **kwargs):
            return PausingCursor(self._connection.cursor(*args, **kwargs))

    @contextmanager
    def paused_connection():
        with original_get_connection() as connection:
            yield PausingConnection(connection)

    monkeypatch.setattr(store, "_get_connection", paused_connection)

    with ThreadPoolExecutor(max_workers=1) as executor:
        summary_future = executor.submit(
            store.get_grant_pool_summary, "default", MONTH_START_ISO
        )
        assert summary_executed.wait(timeout=5)
        try:
            store.assign_grant(
                user_id=1,
                **_grant_args("snapshot_assign", amount_micro=3_000_000),
            )
        finally:
            assignment_committed.set()
        concurrent_summary = summary_future.result(timeout=5)

    assert concurrent_summary["pool_available_micro"] == 10_000_000
    assert concurrent_summary["allocated_to_users_micro"] == 0
    assert concurrent_summary["assigned_this_month_micro"] == 0

    fresh = store.get_grant_pool_summary("default", MONTH_START_ISO)
    assert fresh["pool_available_micro"] == 7_000_000
    assert fresh["allocated_to_users_micro"] == 3_000_000
    assert fresh["assigned_this_month_micro"] == 3_000_000


@pg_only
def test_postgres_credit_ledger_rejects_blank_operation_key(pg_credits_store):
    store = pg_credits_store
    _pending_order(store)
    _pay_order(store)

    with psycopg.connect(store.database_url) as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.transaction():
                conn.execute("UPDATE credit_ledger_entries SET operation_key = '   '")


@pg_only
def test_postgres_migration_preserves_legacy_stripe_ledger(pg_legacy_credits_url):
    pg_module.PostgresCreditsStore(pg_legacy_credits_url)

    with psycopg.connect(pg_legacy_credits_url, row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT * FROM credit_ledger_entries ORDER BY id"
        ).fetchall()
        pool = conn.execute(
            "SELECT * FROM credit_grant_pools WHERE pool_id = 'default'"
        ).fetchone()
        pool_entries = conn.execute(
            "SELECT COUNT(*) AS count FROM credit_grant_pool_ledger_entries"
        ).fetchone()
        constraints = {
            row["conname"]
            for row in conn.execute(
                """
                SELECT conname
                FROM pg_constraint
                WHERE conrelid = 'credit_ledger_entries'::regclass
                """
            ).fetchall()
        }

    assert [row["id"] for row in rows] == [1, 2]
    assert [row["amount_micro"] for row in rows] == [10_000_000, -4_000_000]
    assert [row["bucket"] for row in rows] == ["purchased", "purchased"]
    assert [row["operation_id"] for row in rows] == [
        "stripe:legacy-purchase",
        "stripe:legacy-refund",
    ]
    assert [row["idempotency_key"] for row in rows] == [
        "stripe:legacy-purchase",
        "stripe:legacy-refund",
    ]
    assert [row["refund_request_id"] for row in rows] == [None, "rfnd_legacy"]
    assert [row["stripe_event_id"] for row in rows] == [
        "evt_legacy_purchase",
        "evt_legacy_refund",
    ]
    assert all(row["request_digest"] is None for row in rows)
    assert all(row["actor_user_id"] is None for row in rows)
    assert all(row["source"] == "stripe" for row in rows)
    assert all(row["reference_type"] is None for row in rows)
    assert all(row["reference_id"] is None for row in rows)
    assert pool["name"] == "Platform Research Grants"
    assert pool["status"] == "active"
    assert pool_entries["count"] == 0
    assert {
        "credit_ledger_entries_bucket_check",
        "credit_ledger_entries_operation_key_check",
        "credit_ledger_entries_operation_id_check",
        "credit_ledger_entries_idempotency_key_check",
        "credit_ledger_entries_request_digest_check",
        "credit_ledger_entries_source_check",
        "credit_ledger_entries_reason_check",
        "credit_ledger_entries_shape",
        "credit_ledger_entries_actor_user_id_fkey",
    } <= constraints

    with psycopg.connect(pg_legacy_credits_url) as conn:
        conn.execute(
            """
            UPDATE credit_grant_pools
            SET name = 'Renamed Pool', status = 'disabled'
            WHERE pool_id = 'default'
            """
        )
    db_pool._reset_for_tests()
    pg_module.PostgresCreditsStore(pg_legacy_credits_url)
    with psycopg.connect(pg_legacy_credits_url, row_factory=dict_row) as conn:
        reopened_pool = conn.execute(
            "SELECT * FROM credit_grant_pools WHERE pool_id = 'default'"
        ).fetchone()
        reopened_pool_entries = conn.execute(
            "SELECT COUNT(*) AS count FROM credit_grant_pool_ledger_entries"
        ).fetchone()

    assert reopened_pool["name"] == "Renamed Pool"
    assert reopened_pool["status"] == "disabled"
    assert reopened_pool_entries["count"] == 0


@pg_only
def test_postgres_migration_adds_attempt_index_before_dependent_index(
    pg_legacy_credits_url,
):
    with psycopg.connect(pg_legacy_credits_url) as conn:
        conn.execute(
            """
            INSERT INTO credit_llm_reservations (
                reservation_id, user_id, run_id, call_index,
                reserved_micro, reserved_grant_micro, reserved_purchased_micro,
                operation_key, request_digest, created_at, updated_at
            ) VALUES (
                'legacy-reservation', 1, 'legacy-run', 0,
                100, 100, 0, 'legacy-operation', 'legacy-digest',
                '2026-08-01T00:00:00+00:00', '2026-08-01T00:00:00+00:00'
            )
            """
        )

    db_pool._reset_for_tests()
    pg_module.PostgresCreditsStore(pg_legacy_credits_url)

    with psycopg.connect(pg_legacy_credits_url, row_factory=dict_row) as conn:
        columns = {
            row["column_name"]
            for row in conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'credit_llm_reservations'
                """
            )
        }
        reservation = conn.execute(
            """
            SELECT attempt_index, provider_id
            FROM credit_llm_reservations
            WHERE reservation_id = 'legacy-reservation'
            """
        ).fetchone()
        index_names = {
            row["indexname"]
            for row in conn.execute(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = current_schema()
                  AND tablename = 'credit_llm_reservations'
                """
            )
        }

    assert {"attempt_index", "provider_id"} <= columns
    assert reservation == {"attempt_index": 0, "provider_id": None}
    assert "idx_credit_llm_reservations_run_status" in index_names

    db_pool._reset_for_tests()
    pg_module.PostgresCreditsStore(pg_legacy_credits_url)


def _remove_postgres_grant_pool_snapshots(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        conn.execute(
            """
            UPDATE credit_grant_pools
            SET name = 'Legacy Research Pool', status = 'disabled'
            WHERE pool_id = 'default'
            """
        )
        conn.execute(
            """
            ALTER TABLE credit_grant_pool_ledger_entries
            DROP COLUMN pool_name_snapshot,
            DROP COLUMN pool_status_snapshot
            """
        )


@pg_only
def test_postgres_migrates_pre_snapshot_grant_pool_ledger(pg_credits_store):
    store = pg_credits_store
    fund_command = _grant_args("upgrade_fund", amount_micro=10_000_000)
    assign_command = _grant_args("upgrade_assign", amount_micro=3_000_000)
    funded = store.fund_grant_pool(**fund_command)
    assigned = store.assign_grant(user_id=1, **assign_command)

    _remove_postgres_grant_pool_snapshots(store.database_url)
    db_pool._reset_for_tests()
    upgraded = pg_module.PostgresCreditsStore(store.database_url)
    with psycopg.connect(upgraded.database_url, row_factory=dict_row) as conn:
        columns = {
            row["column_name"]: row["is_nullable"]
            for row in conn.execute(
                """
                SELECT column_name, is_nullable
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'credit_grant_pool_ledger_entries'
                """
            ).fetchall()
        }
        rows = conn.execute(
            """
            SELECT * FROM credit_grant_pool_ledger_entries ORDER BY id
            """
        ).fetchall()
        constraints = {
            row["conname"]
            for row in conn.execute(
                """
                SELECT conname
                FROM pg_constraint
                WHERE conrelid = 'credit_grant_pool_ledger_entries'::regclass
                """
            ).fetchall()
        }

    assert columns["pool_name_snapshot"] == "NO"
    assert columns["pool_status_snapshot"] == "NO"
    assert [row["id"] for row in rows] == [
        funded["entry"]["id"],
        assigned["entry"]["id"],
    ]
    assert [row["amount_micro"] for row in rows] == [10_000_000, -3_000_000]
    assert rows[1]["user_ledger_entry_id"] == assigned["user_entry"]["id"]
    assert all(row["pool_name_snapshot"] == "Legacy Research Pool" for row in rows)
    assert all(row["pool_status_snapshot"] == "disabled" for row in rows)
    assert {
        "credit_grant_pool_ledger_entries_pool_name_snapshot_check",
        "credit_grant_pool_ledger_entries_pool_status_snapshot_check",
    } <= constraints

    with psycopg.connect(upgraded.database_url) as conn:
        conn.execute(
            """
            UPDATE credit_grant_pools
            SET name = 'Current Pool', status = 'active'
            WHERE pool_id = 'default'
            """
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE credit_grant_pool_ledger_entries
                    SET pool_name_snapshot = ' '
                    WHERE id = %s
                    """,
                    (funded["entry"]["id"],),
                )

    replayed = upgraded.assign_grant(user_id=1, **assign_command)
    assert replayed["entry"]["id"] == assigned["entry"]["id"]
    assert replayed["pool"]["name"] == "Legacy Research Pool"
    assert replayed["pool"]["status"] == "disabled"
    assert replayed["pool"]["balance_micro"] == 7_000_000
    assert replayed["user_balance"]["grant_available_micro"] == 3_000_000


@pg_only
def test_failed_postgres_pool_snapshot_migration_restores_old_table(
    pg_credits_store, monkeypatch
):
    store = pg_credits_store
    funded = store.fund_grant_pool(
        **_grant_args("rollback_fund", amount_micro=4_000_000)
    )
    _remove_postgres_grant_pool_snapshots(store.database_url)
    db_pool._reset_for_tests()

    original_migration = pg_module.CREDITS_POSTGRES_GRANT_MIGRATION_DDL
    monkeypatch.setattr(
        pg_module,
        "CREDITS_POSTGRES_GRANT_MIGRATION_DDL",
        f"{original_migration}\nSELECT 1 / 0;",
    )

    with pytest.raises(psycopg.errors.DivisionByZero):
        pg_module.PostgresCreditsStore(store.database_url)
    db_pool._reset_for_tests()

    with psycopg.connect(store.database_url, row_factory=dict_row) as conn:
        columns = {
            row["column_name"]
            for row in conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'credit_grant_pool_ledger_entries'
                """
            ).fetchall()
        }
        rows = conn.execute(
            """
            SELECT id, amount_micro, operation_id
            FROM credit_grant_pool_ledger_entries
            ORDER BY id
            """
        ).fetchall()
        pool = conn.execute(
            """
            SELECT name, status FROM credit_grant_pools WHERE pool_id = 'default'
            """
        ).fetchone()

    assert "pool_name_snapshot" not in columns
    assert "pool_status_snapshot" not in columns
    assert rows == [
        {
            "id": funded["entry"]["id"],
            "amount_micro": 4_000_000,
            "operation_id": "pg_grant_rollback_fund",
        }
    ]
    assert pool == {"name": "Legacy Research Pool", "status": "disabled"}


@pg_only
def test_postgres_failed_grant_migration_rolls_back_every_schema_change(
    pg_legacy_credits_url, monkeypatch
):
    original_migration = pg_module.CREDITS_POSTGRES_GRANT_MIGRATION_DDL
    monkeypatch.setattr(
        pg_module,
        "CREDITS_POSTGRES_GRANT_MIGRATION_DDL",
        f"{original_migration}\nSELECT 1 / 0;",
    )

    with pytest.raises(psycopg.errors.DivisionByZero):
        pg_module.PostgresCreditsStore(pg_legacy_credits_url)
    db_pool._reset_for_tests()

    with psycopg.connect(pg_legacy_credits_url, row_factory=dict_row) as conn:
        columns = {
            row["column_name"]
            for row in conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'credit_ledger_entries'
                """
            ).fetchall()
        }
        rows = conn.execute(
            """
            SELECT id, operation_key, amount_micro
            FROM credit_ledger_entries
            ORDER BY id
            """
        ).fetchall()
        grant_pool_table = conn.execute(
            "SELECT to_regclass('credit_grant_pools') AS table_name"
        ).fetchone()

    assert columns == {
        "id",
        "user_id",
        "entry_type",
        "amount_micro",
        "payment_order_id",
        "refund_request_id",
        "stripe_event_id",
        "operation_key",
        "created_at",
    }
    assert [(row["id"], row["operation_key"], row["amount_micro"]) for row in rows] == [
        (1, "stripe:legacy-purchase", 10_000_000),
        (2, "stripe:legacy-refund", -4_000_000),
    ]
    assert grant_pool_table["table_name"] is None


@pg_only
def test_purchase_and_duplicate_webhooks_post_once(pg_credits_store):
    store = pg_credits_store
    _pending_order(store)

    first = _pay_order(store)
    duplicate_event = _pay_order(store)
    second_event = _pay_order(store, event_id="evt_paid_retry")

    assert first == {
        "outcome": "processed",
        "balance_micro": 10_000_000,
        "recovered_micro": 0,
        "outstanding_micro": 0,
        "account_status": "active",
        "restriction_reason": None,
    }
    assert duplicate_event == {
        "outcome": "duplicate",
        "balance_micro": 10_000_000,
    }
    assert second_event == {
        "outcome": "duplicate",
        "balance_micro": 10_000_000,
    }
    assert store.get_balance_micro(1) == 10_000_000
    assert len(store.list_ledger_entries(1)["items"]) == 1


@pg_only
def test_live_or_tampered_payment_never_posts_credits(pg_credits_store):
    store = pg_credits_store
    _pending_order(store)

    result = store.settle_paid_checkout(
        event_id="evt_live",
        event_type="checkout.session.completed",
        livemode=True,
        object_id="cs_test_ord_10",
        payload_sha256="b" * 64,
        order_id="ord_10",
        checkout_session_id="cs_test_ord_10",
        payment_intent_id="pi_live_wrong",
        currency="usd",
        amount_usd_cents=1000,
    )

    assert result["outcome"] == "rejected"
    assert store.get_balance_micro(1) == 0


@pg_only
def test_expired_checkout_projects_terminal_state_without_credit(pg_credits_store):
    store = pg_credits_store
    _pending_order(store)

    result = store.settle_unpaid_checkout(
        event_id="evt_expired",
        event_type="checkout.session.expired",
        livemode=False,
        object_id="cs_test_ord_10",
        payload_sha256="e" * 64,
        order_id="ord_10",
        checkout_session_id="cs_test_ord_10",
        terminal_status="expired",
    )

    assert result == {"outcome": "processed", "status": "expired"}
    assert store.get_order_for_user("ord_10", 1)["status"] == "expired"
    assert store.get_balance_micro(1) == 0


@pg_only
def test_partial_then_full_refund_projects_balance_and_order(pg_credits_store):
    store = pg_credits_store
    _pending_order(store)
    _pay_order(store)

    for number, cents in ((1, 400), (2, 600)):
        refund_id = f"refund_{number}"
        stripe_refund_id = f"re_test_{number}"
        store.reserve_refund(
            refund_id=refund_id,
            payment_order_id="ord_10",
            user_id=1,
            requested_by_user_id=2,
            amount_usd_cents=cents,
            credits_micro=cents * 10_000,
        )
        store.attach_stripe_refund(refund_id, stripe_refund_id=stripe_refund_id)
        settled = store.settle_succeeded_refund(
            event_id=f"evt_refund_{number}",
            event_type="refund.updated",
            livemode=False,
            object_id=stripe_refund_id,
            payload_sha256=str(number) * 64,
            refund_id=refund_id,
            stripe_refund_id=stripe_refund_id,
            payment_intent_id="pi_test_ord_10",
            currency="usd",
            amount_usd_cents=cents,
        )
        assert settled["outcome"] == "processed"

    assert store.get_balance_micro(1) == 0
    assert store.get_order_for_user("ord_10", 1)["status"] == "refunded"
    assert store.get_order_for_user("ord_10", 3) is None


@pg_only
def test_concurrent_refund_reservations_cannot_over_refund(pg_credits_store):
    store = pg_credits_store
    _pending_order(store)
    _pay_order(store)

    def reserve(number: int):
        try:
            store.reserve_refund(
                refund_id=f"refund_race_{number}",
                payment_order_id="ord_10",
                user_id=1,
                requested_by_user_id=2,
                amount_usd_cents=700,
                credits_micro=7_000_000,
            )
            return "reserved"
        except repo_module.RefundNotAllowedError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(reserve, (1, 2)))

    assert sorted(outcomes) == ["rejected", "reserved"]
    order = store.list_orders_for_admin()["items"][0]
    assert order["refundable_usd_cents"] == 300


@pg_only
def test_failed_refund_releases_the_purchase_lot(pg_credits_store):
    store = pg_credits_store
    _pending_order(store)
    _pay_order(store)
    store.reserve_refund(
        refund_id="refund_failed",
        payment_order_id="ord_10",
        user_id=1,
        requested_by_user_id=2,
        amount_usd_cents=1000,
        credits_micro=10_000_000,
    )
    store.attach_stripe_refund("refund_failed", stripe_refund_id="re_test_failed")

    failed = store.fail_refund(
        event_id="evt_refund_failed",
        event_type="refund.failed",
        livemode=False,
        object_id="re_test_failed",
        payload_sha256="f" * 64,
        refund_id="refund_failed",
        stripe_refund_id="re_test_failed",
    )
    replacement = store.reserve_refund(
        refund_id="refund_replacement",
        payment_order_id="ord_10",
        user_id=1,
        requested_by_user_id=2,
        amount_usd_cents=1000,
        credits_micro=10_000_000,
    )

    assert failed["outcome"] == "processed"
    assert replacement["status"] == "pending"
