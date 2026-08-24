"""Strict public and internal types for model providers and credentials."""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

AdapterType = Literal[
    "openrouter", "openai", "anthropic", "gemini", "openai_compatible"
]
CredentialStatus = Literal[
    "verified", "invalid", "verification_unavailable", "revoked"
]


class ProviderCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_discovery: bool = True
    system_messages: bool = True
    reasoning: bool = False
    cached_token_usage: bool = False
    reasoning_token_usage: bool = False
    reported_monetary_cost: bool = False
    supported_parameters: tuple[str, ...] = ()
    model_allowlist: tuple[str, ...] = ()

    @field_validator("model_allowlist")
    @classmethod
    def validate_model_allowlist(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        cleaned: list[str] = []
        for value in values:
            item = str(value).strip()
            if (
                len(item) > 64
                or not re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{0,63}$", item)
            ):
                raise ValueError("model_allowlist contains an invalid model id")
            if item not in cleaned:
                cleaned.append(item)
        return tuple(cleaned)


class ProviderRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9_]+$")
    display_name: str = Field(min_length=1, max_length=100)
    adapter_type: AdapterType
    approved_base_url: str
    capabilities: ProviderCapabilities = Field(default_factory=ProviderCapabilities)
    byok_enabled: bool = True
    platform_enabled: bool = False
    status: Literal["enabled", "disabled"] = "enabled"
    created_at: str | None = None
    updated_at: str | None = None


class ExecutionModelOption(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=100)


class ExecutionProviderOption(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=2, max_length=64)
    display_name: str = Field(min_length=1, max_length=100)
    adapter_type: AdapterType
    byok_available: bool
    platform_credits_available: bool
    models: tuple[ExecutionModelOption, ...] = ()


class AdminProviderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=100)
    adapter_type: AdapterType
    approved_base_url: str = Field(min_length=1, max_length=500)
    capabilities: ProviderCapabilities = Field(default_factory=ProviderCapabilities)
    byok_enabled: bool = True
    platform_enabled: bool = False
    status: Literal["enabled", "disabled"] = "enabled"
    source: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=500)
    idempotency_key: str = Field(min_length=8, max_length=200)

    @field_validator("display_name", "source", "reason", "idempotency_key")
    @classmethod
    def trim_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class AdminPlatformCredentialRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr = Field(min_length=8, max_length=4096)
    source: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=500)
    idempotency_key: str = Field(min_length=8, max_length=200)

    @field_validator("api_key", mode="before")
    @classmethod
    def trim_api_key(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("value must not be blank")
        return value

    @field_validator("source", "reason", "idempotency_key")
    @classmethod
    def trim_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class AdminPlatformCredentialPublic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str
    key_last_four: str = Field(min_length=4, max_length=4)
    status: CredentialStatus
    updated_at: str
    last_verified_at: str | None = None


class AdminProviderActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=500)
    idempotency_key: str = Field(min_length=8, max_length=200)

    @field_validator("source", "reason", "idempotency_key")
    @classmethod
    def trim_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class UserCredentialPublic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    credential_id: UUID
    provider_id: str
    label: str
    key_last_four: str = Field(min_length=4, max_length=4)
    status: CredentialStatus
    # Safe, provider-controlled wording only; never include upstream response bodies.
    verification_message: str = Field(default="", max_length=240)
    is_default: bool = False
    created_at: str
    updated_at: str
    last_verified_at: str | None = None


class UserCredentialCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9_]+$")
    label: str = Field(min_length=1, max_length=80)
    api_key: SecretStr = Field(min_length=8, max_length=4096)
    set_default: bool = False

    @field_validator("label")
    @classmethod
    def trim_label(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("api_key", mode="before")
    @classmethod
    def trim_api_key(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("value must not be blank")
        return value


class CredentialValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: CredentialStatus
    message: str = Field(max_length=240)
    models: list[str] = Field(default_factory=list, max_length=200)
