"""Typed inputs and outputs for the ATL Credits billing boundary."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)


CreditPackageId = Literal["usd_5", "usd_10", "usd_20", "usd_50"]

FIXED_PACKAGES_USD_CENTS: dict[str, int] = {
    "usd_5": 500,
    "usd_10": 1000,
    "usd_20": 2000,
    "usd_50": 5000,
}
MIN_CUSTOM_USD_CENTS = 500
MAX_CUSTOM_USD_CENTS = 20_000
MICRO_CREDITS_PER_USD_CENT = 10_000


def credits_micro_for_cents(amount_usd_cents: int) -> int:
    if (
        isinstance(amount_usd_cents, bool)
        or not isinstance(amount_usd_cents, int)
        or amount_usd_cents <= 0
    ):
        raise ValueError("amount_usd_cents must be a positive integer")
    return amount_usd_cents * MICRO_CREDITS_PER_USD_CENT


def format_credits(credits_micro: int) -> str:
    sign = "-" if credits_micro < 0 else ""
    absolute = abs(int(credits_micro))
    whole, fraction = divmod(absolute, 1_000_000)
    return f"{sign}{whole}.{fraction:06d}"


class CheckoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_request_id: UUID
    package_id: CreditPackageId | None = None
    custom_amount_usd_cents: StrictInt | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> "CheckoutRequest":
        selected = self.package_id is not None
        custom = self.custom_amount_usd_cents is not None
        if selected == custom:
            raise ValueError("select exactly one fixed package or one custom amount")
        if custom and not (
            MIN_CUSTOM_USD_CENTS <= self.custom_amount_usd_cents <= MAX_CUSTOM_USD_CENTS
        ):
            raise ValueError("custom amount must be from 500 through 20,000 cents")
        return self

    @property
    def amount_usd_cents(self) -> int:
        if self.package_id is not None:
            return FIXED_PACKAGES_USD_CENTS[self.package_id]
        return int(self.custom_amount_usd_cents)


class AdminRefundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_request_id: UUID
    payment_order_id: str
    amount_usd_cents: StrictInt

    @model_validator(mode="after")
    def validate_refund(self) -> "AdminRefundRequest":
        if not self.payment_order_id.strip():
            raise ValueError("payment_order_id is required")
        if self.amount_usd_cents <= 0:
            raise ValueError("amount_usd_cents must be a positive integer")
        return self


class _GrantCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_request_id: UUID
    amount_micro: StrictInt = Field(gt=0)
    source: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("source", "reason")
    @classmethod
    def validate_trimmed_audit_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("audit text must be trimmed")
        return value


class FundGrantPoolRequest(_GrantCommand):
    pass


class ReduceGrantPoolRequest(_GrantCommand):
    pass


class _UserGrantCommand(_GrantCommand):
    pool_id: str = Field(default="default", min_length=1, max_length=120)

    @field_validator("pool_id")
    @classmethod
    def validate_pool_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("pool_id must be trimmed")
        return value


class AssignGrantRequest(_UserGrantCommand):
    pass


class ReclaimGrantRequest(_UserGrantCommand):
    pass


_BALANCE_PROJECTION_FIELDS = frozenset(
    {
        "grant_committed_micro",
        "purchased_committed_micro",
        "grant_available_micro",
        "purchased_available_micro",
        "total_available_micro",
        "display_grant_credits",
        "display_purchased_credits",
        "display_total_credits",
    }
)


def _validate_balance_projection(
    *,
    grant_committed_micro: int,
    purchased_committed_micro: int,
    grant_available_micro: int,
    purchased_available_micro: int,
    total_available_micro: int,
    display_grant_credits: str,
    display_purchased_credits: str,
    display_total_credits: str,
) -> None:
    if (
        grant_committed_micro - grant_available_micro < 0
        or purchased_committed_micro - purchased_available_micro < 0
    ):
        raise ValueError("available balances cannot exceed committed balances")
    if total_available_micro != grant_available_micro + purchased_available_micro:
        raise ValueError(
            "total_available_micro must equal the sum of available balances"
        )
    expected_displays = {
        "display_grant_credits": format_credits(grant_available_micro),
        "display_purchased_credits": format_credits(purchased_available_micro),
        "display_total_credits": format_credits(total_available_micro),
    }
    actual_displays = {
        "display_grant_credits": display_grant_credits,
        "display_purchased_credits": display_purchased_credits,
        "display_total_credits": display_total_credits,
    }
    for field_name, expected in expected_displays.items():
        if actual_displays[field_name] != expected:
            raise ValueError(f"{field_name} must match its authoritative amount")


class BalanceProjection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    grant_committed_micro: StrictInt
    purchased_committed_micro: StrictInt
    grant_available_micro: StrictInt
    purchased_available_micro: StrictInt
    total_available_micro: StrictInt
    display_grant_credits: str
    display_purchased_credits: str
    display_total_credits: str

    @model_validator(mode="after")
    def validate_accounting(self) -> "BalanceProjection":
        _validate_balance_projection(**self.model_dump())
        return self


class BalanceResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    balance_micro: StrictInt
    display_credits: str
    grant_committed_micro: StrictInt
    purchased_committed_micro: StrictInt
    grant_available_micro: StrictInt
    purchased_available_micro: StrictInt
    total_available_micro: StrictInt
    display_grant_credits: str
    display_purchased_credits: str
    display_total_credits: str
    spending_enabled: Literal[False] = False
    account_status: str
    billing_available: bool

    @model_validator(mode="before")
    @classmethod
    def populate_legacy_purchase_projection(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        projected = dict(value)
        supplied_projection_fields = _BALANCE_PROJECTION_FIELDS.intersection(projected)
        if (
            supplied_projection_fields
            and supplied_projection_fields != _BALANCE_PROJECTION_FIELDS
        ):
            raise ValueError("balance projection fields must be supplied all together")
        if supplied_projection_fields:
            return projected
        balance_micro = projected.get("balance_micro")
        display_credits = projected.get("display_credits")
        projected.update(
            grant_committed_micro=0,
            purchased_committed_micro=balance_micro,
            grant_available_micro=0,
            purchased_available_micro=balance_micro,
            total_available_micro=balance_micro,
            display_grant_credits=format_credits(0),
            display_purchased_credits=display_credits,
            display_total_credits=display_credits,
        )
        return projected

    @model_validator(mode="after")
    def validate_total_aliases(self) -> "BalanceResult":
        _validate_balance_projection(
            grant_committed_micro=self.grant_committed_micro,
            purchased_committed_micro=self.purchased_committed_micro,
            grant_available_micro=self.grant_available_micro,
            purchased_available_micro=self.purchased_available_micro,
            total_available_micro=self.total_available_micro,
            display_grant_credits=self.display_grant_credits,
            display_purchased_credits=self.display_purchased_credits,
            display_total_credits=self.display_total_credits,
        )
        if self.balance_micro != self.total_available_micro:
            raise ValueError("balance_micro must alias total_available_micro")
        if self.display_credits != self.display_total_credits:
            raise ValueError("display_credits must alias display_total_credits")
        return self


class GrantPoolSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pool_id: str
    pool_name: str
    pool_status: str
    pool_available_micro: StrictInt
    allocated_to_users_micro: StrictInt
    assigned_this_month_micro: StrictInt
    reclaimed_this_month_micro: StrictInt
    display_pool_available_credits: str
    display_allocated_to_users_credits: str
    display_assigned_this_month_credits: str
    display_reclaimed_this_month_credits: str
    month_start_iso: str

    @model_validator(mode="after")
    def validate_display_values(self) -> "GrantPoolSummary":
        display_amounts = {
            "display_pool_available_credits": self.pool_available_micro,
            "display_allocated_to_users_credits": self.allocated_to_users_micro,
            "display_assigned_this_month_credits": self.assigned_this_month_micro,
            "display_reclaimed_this_month_credits": self.reclaimed_this_month_micro,
        }
        for field_name, amount_micro in display_amounts.items():
            if getattr(self, field_name) != format_credits(amount_micro):
                raise ValueError(f"{field_name} must match its authoritative amount")
        return self


class GrantMutationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    operation_id: str
    operation_type: str
    actor_user_id: StrictInt
    target_user_id: StrictInt | None
    amount_micro: StrictInt
    source: str
    reason: str
    created_at: str
    pool: GrantPoolSummary | None = None
    user_balance: BalanceProjection | None = None
    pool_ledger_entry_id: StrictInt | None = None
    user_ledger_entry_id: StrictInt | None = None


class CheckoutResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_id: str
    checkout_session_id: str
    checkout_url: str
    amount_usd_cents: int
    credits_micro: int
    order_status: str


class RefundCreationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    refund_id: str
    stripe_refund_id: str
    payment_order_id: str
    amount_usd_cents: int
    credits_micro: int
    refund_status: str


class WebhookResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: str
    event_type: str
    reason: str | None = None
    balance_micro: int | None = None
    account_restricted: bool = False


class LLMReservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reservation_id: str
    user_id: StrictInt
    run_id: str
    call_index: StrictInt
    reserved_micro: StrictInt
    settled_micro: StrictInt
    status: Literal["open", "settled", "released"]
    created_at: str
    updated_at: str
    failure_reason: str | None = None


class LLMSettlementResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reservation_id: str
    user_id: StrictInt
    run_id: str
    reserved_micro: StrictInt
    settled_micro: StrictInt
    released_micro: StrictInt
    status: Literal["open", "settled", "released"]
    grant_debited_micro: StrictInt = 0
    purchased_debited_micro: StrictInt = 0
    ledger_entry_ids: tuple[StrictInt, ...] = ()
