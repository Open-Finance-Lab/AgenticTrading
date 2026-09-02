"""Provider-neutral request, usage, pricing, and billing evidence models."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dashboard.backend.domain.model_providers.repository_common import (
    validate_provider_id,
)

from dashboard.backend.infrastructure.llm.pricing import (
    PRICING_SOURCE_VERSION,
    price_for_model,
)


class BillingMode(StrEnum):
    PLATFORM_CREDITS = "platform_credits"
    BYOK = "byok"


class LLMMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=120_000)

    @field_validator("content")
    @classmethod
    def trim_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content must not be blank")
        return value


class UsagePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_output_tokens: int = Field(gt=0, le=32_768)
    require_provider_usage: bool = True


class LLMExecutionRequest(BaseModel):
    """A secret-free request accepted by the execution service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    run_id: str = Field(min_length=1, max_length=128)
    call_index: int = Field(ge=0)
    billing_mode: BillingMode
    provider_id: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9_]+$")
    provider_ids: tuple[str, ...] = Field(default=(), max_length=8)
    model_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/\-]{0,63}$",
    )
    system_message: str | None = Field(default=None, max_length=120_000)
    messages: tuple[LLMMessage, ...] = Field(min_length=1, max_length=128)
    usage_policy: UsagePolicy
    temperature: float | None = Field(default=None, ge=0, le=2)
    reasoning_effort: str | None = Field(default=None, max_length=32)

    @field_validator("run_id", "provider_id", "model_id")
    @classmethod
    def trim_identifiers(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("identifier must not be blank")
        return value

    @model_validator(mode="before")
    @classmethod
    def populate_provider_candidates(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "provider_ids" not in payload or payload["provider_ids"] is None:
            payload["provider_ids"] = (payload.get("provider_id"),)
        return payload

    @field_validator("provider_ids")
    @classmethod
    def validate_provider_candidates(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        try:
            cleaned = tuple(validate_provider_id(value) for value in values)
        except Exception as exc:
            raise ValueError("provider_ids contains an invalid provider id") from exc
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("provider_ids must be ordered and unique")
        return cleaned

    @model_validator(mode="after")
    def validate_provider_candidate_head(self) -> "LLMExecutionRequest":
        candidates = self.provider_ids or (self.provider_id,)
        if candidates[0] != self.provider_id:
            raise ValueError("provider_ids must start with provider_id")
        object.__setattr__(self, "provider_ids", candidates)
        return self

    @field_validator("system_message")
    @classmethod
    def trim_system_message(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("reasoning_effort")
    @classmethod
    def trim_reasoning_effort(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower()
        return value or None


class LLMUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    usage_available: bool = True

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class PricingSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=2, max_length=64)
    model_id: str = Field(min_length=1, max_length=64)
    input_usd_per_million_tokens: float = Field(ge=0)
    output_usd_per_million_tokens: float = Field(ge=0)
    currency: Literal["USD"] = "USD"
    source_version: str = Field(min_length=1, max_length=64)

    @classmethod
    def from_model(
        cls,
        model_id: str,
        provider_id: str = "unknown",
        *,
        source_version: str = PRICING_SOURCE_VERSION,
    ) -> "PricingSnapshot":
        input_price, output_price = price_for_model(model_id)
        return cls(
            provider_id=provider_id,
            model_id=model_id,
            input_usd_per_million_tokens=input_price,
            output_usd_per_million_tokens=output_price,
            source_version=source_version,
        )


class BillingEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    billing_source: BillingMode
    usage_authority: Literal[
        "provider_usage_pricing_snapshot",
        "provider_reported_cost",
        "unavailable",
        "not_billable_by_atl",
    ]
    provider_cost_usd: float | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    pricing_snapshot: PricingSnapshot | None = None
    provider_cost_credits_micro: int = Field(default=0, ge=0)
    debited_credits_micro: int = Field(default=0, ge=0)
    outstanding_credits_micro: int = Field(default=0, ge=0)


class LLMExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=120_000)
    provider_id: str = Field(min_length=2, max_length=64)
    requested_provider_id: str | None = Field(
        default=None,
        min_length=2,
        max_length=64,
    )
    model_id: str = Field(min_length=1, max_length=64)
    credential_id: str | None = Field(default=None, max_length=128)
    credential_key_last_four: str | None = Field(default=None, min_length=4, max_length=4)
    usage: LLMUsage
    billing: BillingEvidence
    # Normalised provider stop reason (``"max_tokens"`` when the reply was cut
    # at the output ceiling); ``None`` when the provider reported none.
    finish_reason: str | None = Field(default=None, max_length=32)


class LLMRunEvidence(BaseModel):
    """Secret-free aggregate of every completed model call in one run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    billing_mode: BillingMode
    provider_id: str = Field(min_length=2, max_length=64)
    requested_provider_id: str | None = Field(
        default=None,
        min_length=2,
        max_length=64,
    )
    provider_ids: tuple[str, ...] = ()
    provider_mixed: bool = False
    model_id: str = Field(min_length=1, max_length=64)
    credential_id: str | None = Field(default=None, max_length=128)
    credential_key_last_four: str | None = Field(
        default=None,
        min_length=4,
        max_length=4,
    )
    call_count: int = Field(ge=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    usage_available: bool
    provider_cost_usd: float | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    pricing_snapshot: PricingSnapshot | None = None
    debited_credits_micro: int = Field(ge=0)
    outstanding_credits_micro: int = Field(ge=0)
    outcome: Literal["byok", "settled", "settled_overage", "unavailable"]

    @model_validator(mode="before")
    @classmethod
    def populate_provider_attribution(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        provider_id = payload.get("provider_id")
        if "requested_provider_id" not in payload:
            payload["requested_provider_id"] = provider_id
        if "provider_ids" not in payload:
            payload["provider_ids"] = (
                () if provider_id == "mixed" else (provider_id,)
            )
        if "provider_mixed" not in payload:
            payload["provider_mixed"] = provider_id == "mixed"
        return payload

    @field_validator("provider_id", "requested_provider_id")
    @classmethod
    def validate_provider_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return validate_provider_id(value)
        except Exception as exc:
            raise ValueError("invalid provider id") from exc

    @field_validator("provider_ids")
    @classmethod
    def validate_provider_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        try:
            cleaned = tuple(validate_provider_id(value) for value in values)
        except Exception as exc:
            raise ValueError("provider_ids contains an invalid provider id") from exc
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("provider_ids must be ordered and unique")
        return cleaned

    @model_validator(mode="after")
    def validate_provider_mix(self) -> "LLMRunEvidence":
        if self.provider_mixed:
            if self.provider_id != "mixed" or len(self.provider_ids) < 2:
                raise ValueError("mixed provider evidence requires multiple providers")
            if (
                self.credential_id is not None
                or self.credential_key_last_four is not None
                or self.pricing_snapshot is not None
            ):
                raise ValueError(
                    "mixed provider evidence cannot claim one credential or price"
                )
        elif self.provider_ids != (self.provider_id,):
            raise ValueError("uniform provider evidence requires one matching provider")
        return self

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


__all__ = [
    "BillingEvidence",
    "BillingMode",
    "LLMExecutionRequest",
    "LLMExecutionResult",
    "LLMRunEvidence",
    "LLMMessage",
    "LLMUsage",
    "PricingSnapshot",
    "UsagePolicy",
]
