"""Welcome Credits are auditable, spendable, and exactly-once per user."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.credits.repository_common import (
    CreditAccountRestrictedStoreError,
)
from dashboard.backend.domain.credits.service import (
    DEFAULT_SIGNUP_CREDIT_CAMPAIGN,
    DEFAULT_SIGNUP_CREDITS_MICRO,
    CreditsService,
)


def _store(tmp_path, *, user_count: int = 2) -> CreditsStore:
    path = tmp_path / "promotion.db"
    with sqlite3.connect(path) as conn:
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
            ) VALUES (?, ?, ?, 'unused', 'user', '2026-08-28T00:00:00+00:00')
            """,
            [
                (user_id, f"user{user_id}@example.com", f"User {user_id}")
                for user_id in range(1, user_count + 1)
            ],
        )
    return CreditsStore(path)


def test_welcome_grant_is_exactly_once_and_visible_in_activity(tmp_path):
    store = _store(tmp_path)
    service = CreditsService(store=store)

    assert service.grant_default_signup_credits(1) is True
    assert service.grant_default_signup_credits(1) is False

    balance = service.get_balance(1)
    assert balance.grant_available_micro == DEFAULT_SIGNUP_CREDITS_MICRO
    assert balance.display_grant_credits == "1.500000"
    activity = service.list_ledger(1, limit=10, cursor=None)["items"]
    assert len(activity) == 1
    assert activity[0]["entry_type"] == "system_promotion_grant"
    assert activity[0]["reference_type"] == "promotion"
    assert activity[0]["reference_id"] == DEFAULT_SIGNUP_CREDIT_CAMPAIGN
    assert activity[0]["actor_user_id"] is None


def test_promotion_activity_cursor_can_page_into_older_entries(tmp_path):
    store = _store(tmp_path, user_count=1)
    service = CreditsService(store=store)
    service.grant_default_signup_credits(1)
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute(
            """
            INSERT INTO credit_ledger_entries (
                user_id, bucket, entry_type, amount_micro,
                operation_key, operation_id, idempotency_key, request_digest,
                actor_user_id, source, reason, reference_type, reference_id,
                created_at
            ) VALUES (
                1, 'grant', 'admin_grant_assign', 250000,
                'older-grant-operation:user', 'older-grant-operation',
                'older-grant-idempotency:user', 'older-grant-digest',
                1, 'test', 'Older grant.', 'grant_pool', 'test',
                '2026-08-27T00:00:00+00:00'
            )
            """
        )

    first_page = service.list_ledger(1, limit=1, cursor=None)
    assert first_page["items"][0]["entry_type"] == "system_promotion_grant"
    assert first_page["next_cursor"]

    second_page = service.list_ledger(1, limit=1, cursor=first_page["next_cursor"])
    assert second_page["items"][0]["entry_type"] == "admin_grant_assign"


def test_concurrent_welcome_grants_create_only_one_credit_lot(tmp_path):
    store = _store(tmp_path)

    def grant() -> bool:
        return CreditsService(
            store=CreditsStore(store.db_path)
        ).grant_default_signup_credits(1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: grant(), range(2)))

    assert sorted(outcomes) == [False, True]
    assert store.get_balance_micro(1) == DEFAULT_SIGNUP_CREDITS_MICRO


def test_backfill_is_rerunnable_and_covers_every_user(tmp_path):
    store = _store(tmp_path, user_count=3)
    service = CreditsService(store=store)

    assert service.backfill_default_signup_credits() == {
        "total": 3,
        "granted": 3,
        "existing": 0,
        "failed": 0,
    }
    assert service.backfill_default_signup_credits() == {
        "total": 3,
        "granted": 0,
        "existing": 3,
        "failed": 0,
    }
    assert service.get_balance_projections([1, 2, 3])[3].grant_available_micro == (
        DEFAULT_SIGNUP_CREDITS_MICRO
    )


def test_welcome_grant_funds_platform_credit_reservations(tmp_path):
    store = _store(tmp_path)
    service = CreditsService(store=store)
    service.grant_default_signup_credits(1)

    reservation = service.reserve_llm_credits(
        user_id=1,
        run_id="welcome-credit-run",
        call_index=0,
        provider_id="openrouter",
        attempt_index=0,
        amount_micro=1_000_000,
    )

    assert reservation.status == "open"
    assert service.get_balance(1).grant_available_micro == 500_000


def test_restricted_account_receives_campaign_but_cannot_spend_it(tmp_path):
    store = _store(tmp_path)
    service = CreditsService(store=store)
    store.restrict_account(1)

    assert service.grant_default_signup_credits(1) is True
    assert store.get_balance_micro(1) == DEFAULT_SIGNUP_CREDITS_MICRO
    with pytest.raises(CreditAccountRestrictedStoreError):
        service.reserve_llm_credits(
            user_id=1,
            run_id="restricted-run",
            call_index=0,
            provider_id="openrouter",
            attempt_index=0,
            amount_micro=1,
        )
