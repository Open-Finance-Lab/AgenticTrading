"""Dispatch and live-Postgres tests for the Credits store twin."""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql

from dashboard.backend.domain.credits.repository import RefundNotAllowedError
from dashboard.backend.tests._postgres_testing import require_local_postgres_url


TEST_POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")

pg_only = pytest.mark.skipif(
    not TEST_POSTGRES_URL,
    reason="TEST_POSTGRES_URL not set; skipping live-Postgres tests",
)


def test_build_credits_store_defaults_to_sqlite(monkeypatch, capsys):
    import dashboard.backend.domain.credits.repository as repo_module

    monkeypatch.delenv("USERS_DATABASE_URL", raising=False)
    store = repo_module._build_credits_store()

    assert isinstance(store, repo_module.CreditsStore)
    assert (
        "credits_store backend: sqlite (ephemeral on Render)" in capsys.readouterr().out
    )


def test_build_credits_store_picks_postgres_from_users_url(monkeypatch, capsys):
    import dashboard.backend.domain.credits.repository as repo_module
    import dashboard.backend.domain.credits.repository_postgres as pg_module

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
    import dashboard.backend.domain.credits.repository as repo_module

    monkeypatch.delenv("USERS_DATABASE_URL", raising=False)
    monkeypatch.setenv("CONTENT_DATABASE_URL", "postgresql://fake/content")
    monkeypatch.setenv("AGENT_RUNS_DATABASE_URL", "postgresql://fake/runs")

    store = repo_module._build_credits_store()

    assert isinstance(store, repo_module.CreditsStore)
    assert (
        "credits_store backend: sqlite (ephemeral on Render)" in capsys.readouterr().out
    )


def test_build_credits_store_never_prints_credentials(monkeypatch, capsys):
    import dashboard.backend.domain.credits.repository as repo_module
    import dashboard.backend.domain.credits.repository_postgres as pg_module

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
    import dashboard.backend.domain.credits.repository_postgres as pg_module

    with pytest.raises(ValueError) as excinfo:
        pg_module.PostgresCreditsStore(
            '"postgresql://u:sup3r-s3cret@ep-x.neon.tech/atl"'
        )
    assert "sup3r-s3cret" not in str(excinfo.value)


def test_unreachable_postgres_raises_instead_of_falling_back():
    import dashboard.backend.domain.credits.repository_postgres as pg_module

    with pytest.raises(psycopg.OperationalError):
        pg_module.PostgresCreditsStore(
            "postgresql://u:p@127.0.0.1:1/nope?connect_timeout=1"
        )


def _schema_url(database_url: str, schema: str) -> str:
    parts = urlsplit(database_url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


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
            conn.executemany(
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

        from dashboard.backend.domain.credits.repository_postgres import (
            PostgresCreditsStore,
        )

        yield PostgresCreditsStore(scoped_url)
    finally:
        from dashboard.backend import db_pool

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
    order = store.create_or_get_order(
        order_id=order_id,
        user_id=1,
        client_request_id=client_request_id,
        amount_usd_cents=cents,
        credits_micro=cents * 10_000,
    )
    return store.attach_checkout_session(
        order_id, checkout_session_id=f"cs_test_{order_id}"
    )


def _pay_order(store, *, order_id: str = "ord_10", event_id: str = "evt_paid"):
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
        amount_usd_cents=1000,
    )


@pg_only
def test_purchase_and_duplicate_webhooks_post_once(pg_credits_store):
    store = pg_credits_store
    _pending_order(store)

    first = _pay_order(store)
    duplicate_event = _pay_order(store)
    second_event = _pay_order(store, event_id="evt_paid_retry")

    assert first == {"outcome": "processed", "balance_micro": 10_000_000}
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
        except RefundNotAllowedError:
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
