"""Typed inputs and outputs for the ATL Credits billing boundary."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StrictInt, model_validator


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


class BalanceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    balance_micro: int
    display_credits: str
    account_status: str
    billing_available: bool


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
