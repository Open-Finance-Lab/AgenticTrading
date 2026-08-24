"""Service orchestration contract for administrator Grant Credits."""

from __future__ import annotations

from uuid import UUID

from dashboard.backend.domain.credits.models import (
    AssignGrantRequest,
    BalanceProjection,
    FundGrantPoolRequest,
    GrantPoolSummary,
)
from dashboard.backend.domain.credits.repository_common import _canonical_digest
from dashboard.backend.domain.credits.service import CreditsService


CLIENT_REQUEST_ID = UUID("11111111-1111-4111-8111-111111111111")


def _projection() -> dict[str, int | str]:
    return {
        "grant_committed_micro": 2_000_000,
        "purchased_committed_micro": 3_000_000,
        "grant_available_micro": 2_000_000,
        "purchased_available_micro": 3_000_000,
        "total_available_micro": 5_000_000,
        "display_grant_credits": "2.000000",
        "display_purchased_credits": "3.000000",
        "display_total_credits": "5.000000",
    }


def _summary() -> dict[str, int | str]:
    return {
        "pool_id": "default",
        "pool_name": "Platform Research Grants",
        "pool_status": "active",
        "pool_available_micro": 8_000_000,
        "allocated_to_users_micro": 2_000_000,
        "assigned_this_month_micro": 2_000_000,
        "reclaimed_this_month_micro": 0,
        "month_start_iso": "2026-08-01T00:00:00+00:00",
    }


class RecordingStore:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def ensure_account(self, user_id: int):
        self.calls.append(("ensure_account", {"user_id": user_id}))
        return {"status": "active"}

    def get_balance_projection(self, user_id: int):
        self.calls.append(("get_balance_projection", {"user_id": user_id}))
        return _projection()

    def assign_grant(self, **kwargs):
        self.calls.append(("assign_grant", kwargs))
        return {
            "entry": {
                "id": 101,
                "pool_id": "default",
                "operation_id": kwargs["operation_id"],
                "entry_type": "assign",
                "amount_micro": -kwargs["amount_micro"],
                "actor_user_id": kwargs["actor_user_id"],
                "source": kwargs["source"],
                "reason": kwargs["reason"],
                "created_at": "2026-08-23T00:00:00+00:00",
                "user_id": 9,
                "user_ledger_entry_id": 202,
            },
            "user_entry": {"id": 202},
            "pool": {
                "pool_id": "default",
                "name": "Platform Research Grants",
                "status": "active",
                "balance_micro": 8_000_000,
            },
            "user_balance": _projection(),
        }

    def fund_grant_pool(self, **kwargs):
        self.calls.append(("fund_grant_pool", kwargs))
        return {
            "entry": {
                "id": 301,
                "pool_id": "default",
                "operation_id": kwargs["operation_id"],
                "entry_type": "fund",
                "amount_micro": kwargs["amount_micro"],
                "actor_user_id": kwargs["actor_user_id"],
                "source": kwargs["source"],
                "reason": kwargs["reason"],
                "created_at": "2026-08-23T00:00:00+00:00",
                "user_id": None,
                "user_ledger_entry_id": None,
            },
            "user_entry": None,
            "pool": None,
            "user_balance": None,
        }

    def get_grant_pool_summary(self, pool_id: str, month_start_iso: str):
        self.calls.append(
            (
                "get_grant_pool_summary",
                {"pool_id": pool_id, "month_start_iso": month_start_iso},
            )
        )
        return _summary()


def test_get_balance_uses_bucket_projection_and_keeps_legacy_aliases():
    store = RecordingStore()
    result = CreditsService(store=store).get_balance(9)

    assert result.balance_micro == 5_000_000
    assert result.display_credits == "5.000000"
    assert result.grant_available_micro == 2_000_000
    assert result.purchased_available_micro == 3_000_000
    assert result.spending_enabled is False


def test_assign_grant_binds_actor_target_and_audit_text_to_digest():
    store = RecordingStore()
    request = AssignGrantRequest(
        client_request_id=CLIENT_REQUEST_ID,
        amount_micro=2_000_000,
        source="research_budget",
        reason="Approved pilot.",
    )

    result = CreditsService(store=store).assign_grant(
        admin_id=7, user_id=9, request=request
    )
    name, call = next(item for item in store.calls if item[0] == "assign_grant")

    expected_parts = {
        "operation": "assign",
        "actor_user_id": 7,
        "pool_id": "default",
        "user_id": 9,
        "amount_micro": 2_000_000,
        "source": "research_budget",
        "reason": "Approved pilot.",
    }
    assert name == "assign_grant"
    assert call["operation_id"].startswith("grant_assign_")
    assert call["idempotency_key"] == f"admin-grant:{CLIENT_REQUEST_ID}"
    assert call["request_digest"] == _canonical_digest(expected_parts)
    assert result.operation_type == "assign_grant"
    assert result.target_user_id == 9
    assert result.pool_ledger_entry_id == 101
    assert result.user_ledger_entry_id == 202
    assert isinstance(result.pool, GrantPoolSummary)
    assert isinstance(result.user_balance, BalanceProjection)


def test_fund_grant_pool_uses_default_pool_and_exact_amount():
    store = RecordingStore()
    request = FundGrantPoolRequest(
        client_request_id=CLIENT_REQUEST_ID,
        amount_micro=4_000_000,
        source="operations_budget",
        reason="Fund the pool.",
    )

    CreditsService(store=store).fund_grant_pool(admin_id=7, request=request)
    name, call = store.calls[-1]
    assert name == "fund_grant_pool"
    assert call["pool_id"] == "default"
    assert call["amount_micro"] == 4_000_000
