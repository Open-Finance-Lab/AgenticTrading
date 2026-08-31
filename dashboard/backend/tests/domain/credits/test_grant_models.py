"""Contract tests for administrator-funded Grant Credits."""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from dashboard.backend.domain.credits.models import (
    AssignGrantRequest,
    BalanceProjection,
    BalanceResult,
    FundGrantPoolRequest,
    GrantMutationResult,
    GrantPoolSummary,
    ReclaimGrantRequest,
    ReduceGrantPoolRequest,
)
from dashboard.backend.domain.credits.repository_common import (
    CreditAccountRestrictedStoreError,
    CreditsStoreError,
    GrantPoolInsufficientError,
    GrantReclaimExceedsAvailableError,
    IdempotencyConflictError,
    _canonical_digest,
    _nonzero_integer,
    _required_text,
)


CLIENT_REQUEST_ID = UUID("11111111-1111-4111-8111-111111111111")


def _command_payload() -> dict[str, object]:
    return {
        "client_request_id": CLIENT_REQUEST_ID,
        "amount_micro": 1_000_000,
        "source": "operations_budget",
        "reason": "Research allocation.",
    }


@pytest.mark.parametrize("amount", [True, 1.0, "1000000", 0, -1])
def test_grant_commands_require_positive_strict_micro_credit_integer(amount):
    with pytest.raises(ValidationError):
        FundGrantPoolRequest(**{**_command_payload(), "amount_micro": amount})


@pytest.mark.parametrize(
    ("field", "value"),
    [("source", ""), ("source", " x"), ("reason", " ")],
)
def test_grant_commands_require_trimmed_audit_text(field, value):
    payload = _command_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        AssignGrantRequest(**payload)


@pytest.mark.parametrize(
    "request_type",
    [
        FundGrantPoolRequest,
        ReduceGrantPoolRequest,
        AssignGrantRequest,
        ReclaimGrantRequest,
    ],
)
def test_grant_commands_reject_extra_keys(request_type):
    with pytest.raises(ValidationError, match="extra"):
        request_type(**_command_payload(), unexpected=True)


@pytest.mark.parametrize("pool_id", ["", " ", " default", "default ", 7, "x" * 121])
def test_user_grant_commands_require_a_trimmed_bounded_pool_id(pool_id):
    with pytest.raises(ValidationError):
        AssignGrantRequest(**_command_payload(), pool_id=pool_id)


@pytest.mark.parametrize("request_type", [AssignGrantRequest, ReclaimGrantRequest])
def test_user_grant_commands_default_to_the_default_pool(request_type):
    assert request_type(**_command_payload()).pool_id == "default"


def test_balance_projection_is_frozen_and_serializes_exactly():
    projection = BalanceProjection(
        grant_committed_micro=4_000_000,
        purchased_committed_micro=2_500_000,
        grant_available_micro=4_000_000,
        purchased_available_micro=2_500_000,
        total_available_micro=6_500_000,
        display_grant_credits="4.000000",
        display_purchased_credits="2.500000",
        display_total_credits="6.500000",
    )

    assert projection.model_dump() == {
        "grant_committed_micro": 4_000_000,
        "purchased_committed_micro": 2_500_000,
        "grant_available_micro": 4_000_000,
        "purchased_available_micro": 2_500_000,
        "total_available_micro": 6_500_000,
        "display_grant_credits": "4.000000",
        "display_purchased_credits": "2.500000",
        "display_total_credits": "6.500000",
    }
    with pytest.raises(ValidationError, match="frozen"):
        projection.grant_available_micro = 0


def test_balance_result_exposes_total_aliases_and_separate_credit_buckets():
    balance = BalanceResult(
        balance_micro=6_500_000,
        display_credits="6.500000",
        grant_committed_micro=4_000_000,
        purchased_committed_micro=2_500_000,
        grant_available_micro=4_000_000,
        purchased_available_micro=2_500_000,
        total_available_micro=6_500_000,
        display_grant_credits="4.000000",
        display_purchased_credits="2.500000",
        display_total_credits="6.500000",
        account_status="active",
        billing_available=True,
    )

    assert balance.model_dump() == {
        "balance_micro": 6_500_000,
        "display_credits": "6.500000",
        "grant_committed_micro": 4_000_000,
        "purchased_committed_micro": 2_500_000,
        "grant_available_micro": 4_000_000,
        "purchased_available_micro": 2_500_000,
        "total_available_micro": 6_500_000,
        "display_grant_credits": "4.000000",
        "display_purchased_credits": "2.500000",
        "display_total_credits": "6.500000",
        "spending_enabled": False,
        "account_status": "active",
        "billing_available": True,
        "restriction_reason": None,
        "outstanding_credits_micro": 0,
    }
    assert balance.balance_micro == balance.total_available_micro
    assert balance.display_credits == balance.display_total_credits
    with pytest.raises(ValidationError, match="frozen"):
        balance.balance_micro = 0


def test_legacy_balance_construction_projects_existing_balance_as_purchased():
    balance = BalanceResult(
        balance_micro=2_500_000,
        display_credits="2.500000",
        account_status="active",
        billing_available=True,
    )

    assert balance.grant_committed_micro == 0
    assert balance.purchased_committed_micro == 2_500_000
    assert balance.grant_available_micro == 0
    assert balance.purchased_available_micro == 2_500_000
    assert balance.total_available_micro == balance.balance_micro
    assert balance.display_total_credits == balance.display_credits


def test_balance_result_rejects_partial_new_projection_payloads():
    with pytest.raises(ValidationError, match="all together"):
        BalanceResult(
            balance_micro=2_500_000,
            display_credits="2.500000",
            purchased_available_micro=2_500_000,
            account_status="active",
            billing_available=True,
        )


def test_balance_result_rejects_enabled_spending():
    with pytest.raises(ValidationError):
        BalanceResult(
            balance_micro=0,
            display_credits="0.000000",
            spending_enabled=True,
            account_status="active",
            billing_available=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("balance_micro", 1), ("display_credits", "0.000001")],
)
def test_balance_result_rejects_values_that_conflict_with_total_aliases(field, value):
    payload = {
        "balance_micro": 0,
        "display_credits": "0.000000",
        "grant_committed_micro": 0,
        "purchased_committed_micro": 0,
        "grant_available_micro": 0,
        "purchased_available_micro": 0,
        "total_available_micro": 0,
        "display_grant_credits": "0.000000",
        "display_purchased_credits": "0.000000",
        "display_total_credits": "0.000000",
        "account_status": "active",
        "billing_available": True,
    }
    payload[field] = value

    with pytest.raises(ValidationError, match="alias"):
        BalanceResult(**payload)


def _projection_payload() -> dict[str, object]:
    return {
        "grant_committed_micro": 4_000_000,
        "purchased_committed_micro": 2_500_000,
        "grant_available_micro": 4_000_000,
        "purchased_available_micro": 2_500_000,
        "total_available_micro": 6_500_000,
        "display_grant_credits": "4.000000",
        "display_purchased_credits": "2.500000",
        "display_total_credits": "6.500000",
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"total_available_micro": 6_000_000}, "sum"),
        ({"display_grant_credits": "4.000001"}, "display_grant_credits"),
        ({"display_purchased_credits": "2.500001"}, "display_purchased_credits"),
        ({"display_total_credits": "6.500001"}, "display_total_credits"),
    ],
)
def test_balance_projection_rejects_inconsistent_accounting(updates, message):
    with pytest.raises(ValidationError, match=message):
        BalanceProjection(**{**_projection_payload(), **updates})


def test_balance_projection_allows_open_reservations():
    projection = BalanceProjection(
        **{
            **_projection_payload(),
            "grant_committed_micro": 5_000_000,
            "purchased_committed_micro": 3_000_000,
        }
    )

    assert projection.grant_committed_micro - projection.grant_available_micro == 1_000_000
    assert (
        projection.purchased_committed_micro
        - projection.purchased_available_micro
        == 500_000
    )


def test_balance_result_rejects_inconsistent_full_projection():
    with pytest.raises(ValidationError, match="sum"):
        BalanceResult(
            balance_micro=6_500_000,
            display_credits="6.500000",
            **{
                **_projection_payload(),
                "purchased_committed_micro": 2_000_000,
                "purchased_available_micro": 2_000_000,
            },
            account_status="active",
            billing_available=True,
        )


def _pool_summary() -> GrantPoolSummary:
    return GrantPoolSummary(
        pool_id="default",
        pool_name="Default Grant Pool",
        pool_status="active",
        pool_available_micro=90_000_000,
        allocated_to_users_micro=10_000_000,
        assigned_this_month_micro=12_000_000,
        reclaimed_this_month_micro=2_000_000,
        display_pool_available_credits="90.000000",
        display_allocated_to_users_credits="10.000000",
        display_assigned_this_month_credits="12.000000",
        display_reclaimed_this_month_credits="2.000000",
        month_start_iso="2026-08-01T00:00:00+00:00",
    )


def test_grant_pool_summary_has_exact_metrics_and_is_frozen():
    summary = _pool_summary()

    assert summary.model_dump() == {
        "pool_id": "default",
        "pool_name": "Default Grant Pool",
        "pool_status": "active",
        "pool_available_micro": 90_000_000,
        "allocated_to_users_micro": 10_000_000,
        "assigned_this_month_micro": 12_000_000,
        "reclaimed_this_month_micro": 2_000_000,
        "display_pool_available_credits": "90.000000",
        "display_allocated_to_users_credits": "10.000000",
        "display_assigned_this_month_credits": "12.000000",
        "display_reclaimed_this_month_credits": "2.000000",
        "month_start_iso": "2026-08-01T00:00:00+00:00",
    }
    with pytest.raises(ValidationError, match="frozen"):
        summary.pool_status = "disabled"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("display_pool_available_credits", "90.000001"),
        ("display_allocated_to_users_credits", "10.000001"),
        ("display_assigned_this_month_credits", "12.000001"),
        ("display_reclaimed_this_month_credits", "2.000001"),
    ],
)
def test_grant_pool_summary_rejects_inconsistent_display_values(field, value):
    payload = _pool_summary().model_dump()
    payload[field] = value

    with pytest.raises(ValidationError, match=field):
        GrantPoolSummary(**payload)


def test_grant_mutation_result_has_typed_evidence_and_is_frozen():
    projection = BalanceProjection(
        grant_committed_micro=5_000_000,
        purchased_committed_micro=3_000_000,
        grant_available_micro=5_000_000,
        purchased_available_micro=3_000_000,
        total_available_micro=8_000_000,
        display_grant_credits="5.000000",
        display_purchased_credits="3.000000",
        display_total_credits="8.000000",
    )
    result = GrantMutationResult(
        operation_id="grant:11111111-1111-4111-8111-111111111111",
        operation_type="assign_grant",
        actor_user_id=7,
        target_user_id=42,
        amount_micro=5_000_000,
        source="operations_budget",
        reason="Research allocation.",
        created_at="2026-08-22T10:00:00+00:00",
        pool=_pool_summary(),
        user_balance=projection,
        pool_ledger_entry_id=101,
        user_ledger_entry_id=202,
    )

    assert result.model_dump() == {
        "operation_id": "grant:11111111-1111-4111-8111-111111111111",
        "operation_type": "assign_grant",
        "actor_user_id": 7,
        "target_user_id": 42,
        "amount_micro": 5_000_000,
        "source": "operations_budget",
        "reason": "Research allocation.",
        "created_at": "2026-08-22T10:00:00+00:00",
        "pool": _pool_summary().model_dump(),
        "user_balance": projection.model_dump(),
        "pool_ledger_entry_id": 101,
        "user_ledger_entry_id": 202,
        "recovery": None,
    }
    with pytest.raises(ValidationError, match="frozen"):
        result.amount_micro = 1


def test_grant_mutation_result_allows_optional_evidence_to_be_absent():
    result = GrantMutationResult(
        operation_id="grant:11111111-1111-4111-8111-111111111111",
        operation_type="fund_grant_pool",
        actor_user_id=7,
        target_user_id=None,
        amount_micro=5_000_000,
        source="operations_budget",
        reason="Fund the pool.",
        created_at="2026-08-22T10:00:00+00:00",
    )

    assert result.target_user_id is None
    assert result.pool is None
    assert result.user_balance is None
    assert result.pool_ledger_entry_id is None


@pytest.mark.parametrize(
    ("model_type", "payload"),
    [
        (BalanceProjection, {**_projection_payload(), "unexpected": "stale"}),
        (
            BalanceResult,
            {
                "balance_micro": 0,
                "display_credits": "0.000000",
                "account_status": "active",
                "billing_available": True,
                "unexpected": "stale",
            },
        ),
        (GrantPoolSummary, {**_pool_summary().model_dump(), "unexpected": "stale"}),
        (
            GrantMutationResult,
            {
                "operation_id": "grant:1",
                "operation_type": "fund_grant_pool",
                "actor_user_id": 7,
                "target_user_id": None,
                "amount_micro": 1_000_000,
                "source": "operations_budget",
                "reason": "Fund the pool.",
                "created_at": "2026-08-22T10:00:00+00:00",
                "pool_summary": None,
            },
        ),
    ],
)
def test_financial_result_models_reject_unknown_fields(model_type, payload):
    with pytest.raises(ValidationError, match="extra"):
        model_type(**payload)


@pytest.mark.parametrize(
    ("model_type", "payload"),
    [
        (
            BalanceProjection,
            {
                **_projection_payload(),
                "grant_committed_micro": True,
                "grant_available_micro": True,
                "total_available_micro": 2_500_001,
                "display_grant_credits": "0.000001",
                "display_total_credits": "2.500001",
            },
        ),
        (
            GrantPoolSummary,
            {
                **_pool_summary().model_dump(),
                "pool_available_micro": 90_000_000.0,
            },
        ),
        (
            GrantMutationResult,
            {
                "operation_id": "grant:1",
                "operation_type": "fund_grant_pool",
                "actor_user_id": 7,
                "target_user_id": None,
                "amount_micro": True,
                "source": "operations_budget",
                "reason": "Fund the pool.",
                "created_at": "2026-08-22T10:00:00+00:00",
            },
        ),
    ],
)
def test_financial_result_models_require_strict_integers(model_type, payload):
    with pytest.raises(ValidationError):
        model_type(**payload)


@pytest.mark.parametrize("value", [None, 1, True, "", " ", " x", "x "])
def test_required_text_rejects_non_text_blank_and_untrimmed_values(value):
    with pytest.raises(ValueError):
        _required_text(value, "source")


def test_required_text_enforces_optional_maximum_length():
    assert _required_text("exact", "source", max_length=5) == "exact"
    with pytest.raises(ValueError, match="at most 5"):
        _required_text("longer", "source", max_length=5)


@pytest.mark.parametrize("value", [True, False, 0, 1.0, "1", None])
def test_nonzero_integer_rejects_bool_non_integer_and_zero(value):
    with pytest.raises(ValueError):
        _nonzero_integer(value, "amount_micro")


@pytest.mark.parametrize("value", [-5, 7])
def test_nonzero_integer_preserves_sign(value):
    assert _nonzero_integer(value, "amount_micro") == value


def test_canonical_digest_is_order_independent_and_value_sensitive():
    first = _canonical_digest({"operation": "assign", "amount_micro": 1_000_000})
    reordered = _canonical_digest({"amount_micro": 1_000_000, "operation": "assign"})
    changed = _canonical_digest({"operation": "assign", "amount_micro": 2_000_000})
    changed_field = _canonical_digest(
        {"operation_type": "assign", "amount_micro": 1_000_000}
    )

    assert first == reordered
    assert changed != first
    assert changed_field != first
    assert len(first) == 64


def test_canonical_digest_rejects_non_json_values():
    with pytest.raises(TypeError):
        _canonical_digest({"client_request_id": CLIENT_REQUEST_ID})


@pytest.mark.parametrize(
    "error_type",
    [
        IdempotencyConflictError,
        GrantPoolInsufficientError,
        GrantReclaimExceedsAvailableError,
        CreditAccountRestrictedStoreError,
    ],
)
def test_grant_store_errors_share_the_credits_store_hierarchy(error_type):
    error = error_type("expected failure")

    assert isinstance(error, CreditsStoreError)
    assert str(error) == "expected failure"
