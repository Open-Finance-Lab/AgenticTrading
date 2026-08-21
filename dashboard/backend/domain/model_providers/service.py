"""User credential lifecycle orchestration for approved model providers."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx

from dashboard.backend.infrastructure.llm.adapters import get_adapter

from .models import (
    AdminPlatformCredentialPublic,
    AdminPlatformCredentialRequest,
    AdminProviderRequest,
    CredentialValidation,
    ProviderRecord,
    UserCredentialCreate,
    UserCredentialPublic,
)
from .repository import ModelProviderStore, model_provider_store
from .repository_common import (
    CredentialConflictError,
    ProviderNotFoundError,
    validate_adapter_type,
    validate_approved_origin,
    validate_provider_id,
)


class CredentialAdapter(Protocol):
    def validate(
        self,
        base_url: str,
        secret: str,
        *,
        client: httpx.Client | None = None,
    ) -> CredentialValidation: ...


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ModelProviderService:
    """Apply provider allowlisting, verification, and credential lifecycle rules."""

    def __init__(
        self,
        *,
        store: ModelProviderStore,
        adapter_resolver: Callable[[str], CredentialAdapter] = get_adapter,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.store = store
        self.adapter_resolver = adapter_resolver
        self.http_client = http_client

    def list_providers(self) -> list[ProviderRecord]:
        return [
            ProviderRecord.model_validate(provider)
            for provider in self.store.list_enabled_providers(mode="byok")
        ]

    def list_admin_providers(self) -> list[ProviderRecord]:
        return [
            ProviderRecord.model_validate(provider)
            for provider in self.store.list_all_providers()
        ]

    def get_platform_credential(
        self, provider_id: str
    ) -> AdminPlatformCredentialPublic | None:
        credential = self.store.get_platform_credential_public(provider_id)
        return (
            AdminPlatformCredentialPublic.model_validate(credential)
            if credential
            else None
        )

    def upsert_provider(
        self,
        admin_user_id: int,
        provider_id: str,
        request: AdminProviderRequest,
    ) -> ProviderRecord:
        provider_id = validate_provider_id(provider_id)
        validate_adapter_type(request.adapter_type)
        approved_base_url = validate_approved_origin(request.approved_base_url)
        if self._admin_operation_replayed(
            admin_user_id,
            "upsert_provider",
            provider_id,
            request.idempotency_key,
        ):
            provider = self.store.get_provider(provider_id)
            if not provider:
                raise ProviderNotFoundError("provider not found")
            return ProviderRecord.model_validate(provider)
        provider = self.store.upsert_provider(
            provider_id=provider_id,
            display_name=request.display_name,
            adapter_type=request.adapter_type,
            approved_base_url=approved_base_url,
            capabilities=request.capabilities,
            byok_enabled=request.byok_enabled,
            platform_enabled=request.platform_enabled,
            status=request.status,
        )
        self.store.record_admin_operation(
            actor_user_id=admin_user_id,
            operation="upsert_provider",
            provider_id=provider_id,
            source=request.source,
            reason=request.reason,
            idempotency_key=request.idempotency_key,
        )
        return ProviderRecord.model_validate(provider)

    def set_platform_credential(
        self,
        admin_user_id: int,
        provider_id: str,
        request: AdminPlatformCredentialRequest,
    ) -> AdminPlatformCredentialPublic:
        provider = self.store.get_provider(provider_id)
        if not provider:
            raise ProviderNotFoundError("provider not found")
        if self._admin_operation_replayed(
            admin_user_id,
            "set_platform_credential",
            provider_id,
            request.idempotency_key,
        ):
            credential = self.store.get_platform_credential_public(provider_id)
            if not credential:
                raise ProviderNotFoundError("platform credential not found")
            return AdminPlatformCredentialPublic.model_validate(credential)
        secret = request.api_key.get_secret_value()
        self.store.upsert_platform_credential(
            provider_id=provider_id,
            secret=secret,
            status="verification_unavailable",
        )
        result = self._verify_platform_credential(provider_id, secret)
        self.store.record_admin_operation(
            actor_user_id=admin_user_id,
            operation="set_platform_credential",
            provider_id=provider_id,
            source=request.source,
            reason=request.reason,
            idempotency_key=request.idempotency_key,
        )
        return AdminPlatformCredentialPublic.model_validate(result)

    def reverify_platform_credential(
        self,
        admin_user_id: int,
        provider_id: str,
        *,
        source: str,
        reason: str,
        idempotency_key: str,
    ) -> AdminPlatformCredentialPublic:
        provider = self.store.get_provider(provider_id)
        if not provider:
            raise ProviderNotFoundError("provider not found")
        if self._admin_operation_replayed(
            admin_user_id,
            "reverify_platform_credential",
            provider_id,
            idempotency_key,
        ):
            credential = self.store.get_platform_credential_public(provider_id)
            if not credential:
                raise ProviderNotFoundError("platform credential not found")
            return AdminPlatformCredentialPublic.model_validate(credential)
        secret = self.store.get_platform_credential_secret_any_status(provider_id)
        if not secret:
            raise ProviderNotFoundError("platform credential not found")
        result = self._verify_platform_credential(provider_id, secret)
        self.store.record_admin_operation(
            actor_user_id=admin_user_id,
            operation="reverify_platform_credential",
            provider_id=provider_id,
            source=source,
            reason=reason,
            idempotency_key=idempotency_key,
        )
        return AdminPlatformCredentialPublic.model_validate(result)

    def revoke_platform_credential(
        self,
        admin_user_id: int,
        provider_id: str,
        *,
        source: str,
        reason: str,
        idempotency_key: str,
    ) -> bool:
        if self._admin_operation_replayed(
            admin_user_id,
            "revoke_platform_credential",
            provider_id,
            idempotency_key,
        ):
            return True
        deleted = self.store.delete_platform_credential(provider_id)
        if not deleted:
            raise ProviderNotFoundError("platform credential not found")
        self.store.record_admin_operation(
            actor_user_id=admin_user_id,
            operation="revoke_platform_credential",
            provider_id=provider_id,
            source=source,
            reason=reason,
            idempotency_key=idempotency_key,
        )
        return True

    def resolve_platform_secret(self, provider_id: str) -> str | None:
        provider = self.store.get_provider(provider_id)
        if not provider or provider["status"] != "enabled" or not provider["platform_enabled"]:
            return None
        return self.store.get_platform_credential_secret(provider_id)

    def _admin_operation_replayed(
        self,
        admin_user_id: int,
        operation: str,
        provider_id: str,
        idempotency_key: str,
    ) -> bool:
        existing = self.store.get_admin_operation(idempotency_key)
        if not existing:
            return False
        if (
            int(existing["actor_user_id"]) != int(admin_user_id)
            or existing["operation"] != operation
            or existing["provider_id"] != provider_id
        ):
            raise CredentialConflictError("idempotency key already used")
        return True

    def _verify_platform_credential(self, provider_id: str, secret: str) -> dict[str, Any]:
        provider = self.store.get_provider(provider_id)
        adapter = self.adapter_resolver(provider["adapter_type"])
        try:
            validation = adapter.validate(
                provider["approved_base_url"], secret, client=self.http_client
            )
        except Exception:
            validation = CredentialValidation(
                status="verification_unavailable",
                message="Provider verification was unavailable.",
            )
        return self.store.set_platform_credential_status(
            provider_id,
            status=validation.status,
            last_verified_at=_utcnow_iso() if validation.status == "verified" else None,
        )

    def create_credential(
        self,
        user_id: int,
        request: UserCredentialCreate,
    ) -> UserCredentialPublic:
        provider = self._get_byok_provider(request.provider_id)
        secret = request.api_key.get_secret_value()
        created = self.store.create_user_credential(
            user_id=user_id,
            provider_id=provider["provider_id"],
            label=request.label,
            secret=secret,
        )
        return self._verify_and_update(
            user_id=user_id,
            credential=created,
            provider=provider,
            secret=secret,
            set_default=request.set_default,
        )

    def list_credentials(
        self,
        user_id: int,
        provider_id: str | None = None,
    ) -> list[UserCredentialPublic]:
        return [
            UserCredentialPublic.model_validate(credential)
            for credential in self.store.list_user_credentials(user_id, provider_id)
        ]

    def reverify_credential(
        self,
        user_id: int,
        credential_id: str,
    ) -> UserCredentialPublic:
        secret = self.store.get_user_credential_secret(user_id, credential_id)
        credential = self.store.get_user_credential_public(user_id, credential_id)
        if not credential:
            raise CredentialConflictError("credential is not active")
        if credential["status"] == "revoked":
            raise CredentialConflictError("revoked credentials cannot be verified")
        provider = self._get_byok_provider(credential["provider_id"])
        return self._verify_and_update(
            user_id=user_id,
            credential=credential,
            provider=provider,
            secret=secret,
            set_default=False,
        )

    def set_default_credential(
        self,
        user_id: int,
        credential_id: str,
    ) -> UserCredentialPublic:
        return UserCredentialPublic.model_validate(
            self.store.set_default_user_credential(user_id, credential_id)
        )

    def revoke_credential(
        self,
        user_id: int,
        credential_id: str,
    ) -> UserCredentialPublic:
        return UserCredentialPublic.model_validate(
            self.store.revoke_user_credential(user_id, credential_id)
        )

    def _get_byok_provider(self, provider_id: str) -> dict[str, Any]:
        provider = self.store.get_provider(provider_id)
        if (
            not provider
            or provider["status"] != "enabled"
            or not provider["byok_enabled"]
        ):
            raise ProviderNotFoundError("provider is not available for BYOK")
        return provider

    def _verify_and_update(
        self,
        *,
        user_id: int,
        credential: dict[str, Any],
        provider: dict[str, Any],
        secret: str,
        set_default: bool,
    ) -> UserCredentialPublic:
        adapter = self.adapter_resolver(provider["adapter_type"])
        try:
            validation = adapter.validate(
                provider["approved_base_url"],
                secret,
                client=self.http_client,
            )
        except Exception:
            # Adapter failures are deliberately collapsed without including the
            # exception, request headers, URL query, or credential in output.
            validation = CredentialValidation(
                status="verification_unavailable",
                message="Provider verification was unavailable.",
            )
        updated = self.store.set_user_credential_status(
            user_id,
            credential["credential_id"],
            status=validation.status,
            last_verified_at=(
                _utcnow_iso() if validation.status == "verified" else None
            ),
        )
        if set_default and validation.status == "verified":
            updated = self.store.set_default_user_credential(
                user_id,
                credential["credential_id"],
            )
        return UserCredentialPublic.model_validate(updated)


model_provider_service = ModelProviderService(store=model_provider_store)
