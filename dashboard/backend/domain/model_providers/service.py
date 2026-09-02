"""User credential lifecycle orchestration for approved model providers."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from dataclasses import dataclass
import json
import os
from typing import Any, Protocol

import httpx

from dashboard.backend.infrastructure.llm.adapters import get_adapter
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
)
from dashboard.backend.domain.analytics import instrumentation as analytics_instrumentation

from .execution_catalog import (
    ExecutionModelRoute,
    list_execution_model_routes,
    resolve_execution_model_route,
    UnsupportedExecutionModel,
)
from .models import (
    AdminPlatformCredentialPublic,
    AdminPlatformCredentialRequest,
    AdminProviderRequest,
    CredentialValidation,
    ExecutionModelOption,
    ExecutionProviderOption,
    ProviderRecord,
    UserCredentialCreate,
    UserCredentialPublic,
)
from .repository import (
    ModelProviderStore,
    ensure_credential_encryption_ready,
    model_provider_store,
)
from .repository_common import (
    CredentialConflictError,
    ProviderNotFoundError,
    canonical_request_digest,
    secret_fingerprint,
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


@dataclass(frozen=True)
class ResolvedCredential:
    """Transient credential material used only while constructing a provider client."""

    credential_id: str | None
    provider_id: str
    key_last_four: str
    secret: str


class CredentialResolutionError(LLMExecutionError):
    """A safe failure raised before a worker starts a model call."""

    def __init__(
        self,
        category: ExecutionErrorCategory,
        message: str | None = None,
    ) -> None:
        super().__init__(category, message)


_ENVIRONMENT_PLATFORM_KEY_NAMES = {
    "openrouter": "OPENROUTER_API_KEY",
    "commonstack": "COMMONSTACK_API_KEY",
}


def _environment_platform_secret(provider_id: str) -> str | None:
    """Return an explicitly mapped deployment secret without persisting it."""

    variable_name = _ENVIRONMENT_PLATFORM_KEY_NAMES.get(provider_id)
    if not variable_name:
        return None
    secret = os.getenv(variable_name, "").strip()
    return secret or None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


_SAFE_VERIFICATION_MESSAGES = frozenset(
    {
        "API key verified.",
        "The provider rejected this API key.",
        "The provider rejected this API key or request.",
        "Provider address is not allowed.",
        "Provider address could not be resolved.",
        "Provider verification timed out or was unavailable.",
        "Provider verification was unavailable.",
        "The provider returned an unexpected redirect.",
        "The provider is temporarily unavailable.",
        "Provider returned an unexpected response.",
        "Provider returned an invalid verification response.",
        "Provider returned an invalid model list.",
    }
)


def _safe_verification_message(validation: CredentialValidation) -> str:
    """Persist only fixed, non-sensitive adapter wording."""

    if validation.message in _SAFE_VERIFICATION_MESSAGES:
        return validation.message
    return {
        "verified": "API key verified.",
        "invalid": "The provider rejected this API key.",
        "verification_unavailable": "Provider verification was unavailable.",
        "revoked": "Credential revoked.",
    }.get(validation.status, "Provider verification was unavailable.")


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

    def list_execution_options(
        self,
        user_id: int,
    ) -> list[ExecutionProviderOption]:
        default_counts: dict[str, int] = {}
        for credential in self.store.list_user_credentials(int(user_id)):
            if credential["status"] != "verified" or not credential["is_default"]:
                continue
            provider_id = str(credential["provider_id"])
            default_counts[provider_id] = default_counts.get(provider_id, 0) + 1

        options: list[ExecutionProviderOption] = []
        for raw_provider in self.store.list_all_providers():
            provider = ProviderRecord.model_validate(raw_provider)
            if provider.status != "enabled":
                continue
            platform = self.store.get_platform_credential_public(
                provider.provider_id
            )
            platform_credential_available = bool(
                platform and platform["status"] == "verified"
            )
            environment_available = bool(
                _environment_platform_secret(provider.provider_id)
            )
            routes = list_execution_model_routes(provider)
            options.append(
                ExecutionProviderOption(
                    provider_id=provider.provider_id,
                    display_name=provider.display_name,
                    adapter_type=provider.adapter_type,
                    byok_available=(
                        provider.byok_enabled
                        and default_counts.get(provider.provider_id, 0) == 1
                    ),
                    platform_credits_available=(
                        provider.platform_enabled
                        and (
                            platform_credential_available
                            or environment_available
                        )
                    ),
                    models=tuple(
                        ExecutionModelOption(
                            model_id=route.catalog_id,
                            label=route.label,
                        )
                        for route in routes
                    ),
                )
            )
        # Preserve the repository's existing order while keeping OpenRouter as
        # the preferred platform lane before the CommonStack fallback.
        options.sort(key=lambda option: option.provider_id == "commonstack")
        return options

    def resolve_platform_execution_candidates(
        self,
        catalog_model_id: str,
        preferred_provider_id: str | None = None,
    ) -> tuple[str, ...]:
        """Return compatible, credentialed platform providers in preference order."""

        providers = {
            str(raw["provider_id"]): ProviderRecord.model_validate(raw)
            for raw in self.store.list_all_providers()
            if raw.get("provider_id")
        }
        preferred = (
            validate_provider_id(preferred_provider_id)
            if preferred_provider_id and preferred_provider_id.strip()
            else None
        )
        ordered_ids: list[str] = []
        for provider_id in (preferred, "openrouter", "commonstack"):
            if provider_id and provider_id not in ordered_ids:
                ordered_ids.append(provider_id)
        candidates: list[str] = []
        for provider_id in ordered_ids:
            provider = providers.get(provider_id)
            if not provider or provider.status != "enabled" or not provider.platform_enabled:
                continue
            try:
                self.preflight_execution_model(provider_id, catalog_model_id)
                self.preflight_platform_credential(provider_id)
            except (
                ProviderNotFoundError,
                CredentialResolutionError,
                UnsupportedExecutionModel,
            ):
                continue
            candidates.append(provider_id)
        return tuple(candidates)

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
        request_digest = canonical_request_digest(
            {
                "operation": "upsert_provider",
                "actor_user_id": int(admin_user_id),
                "provider_id": provider_id,
                "display_name": request.display_name,
                "adapter_type": request.adapter_type,
                "approved_base_url": approved_base_url,
                "capabilities": request.capabilities.model_dump(mode="json"),
                "byok_enabled": request.byok_enabled,
                "platform_enabled": request.platform_enabled,
                "status": request.status,
                "source": request.source,
                "reason": request.reason,
            }
        )
        replay = self._admin_operation_replayed(
            admin_user_id,
            "upsert_provider",
            provider_id,
            request.idempotency_key,
            request_digest,
        )
        if replay:
            provider = self._replay_snapshot(replay) or self.store.get_provider(provider_id)
            if not provider:
                raise ProviderNotFoundError("provider not found")
            return ProviderRecord.model_validate(provider)
        audit = {
            "actor_user_id": admin_user_id,
            "operation": "upsert_provider",
            "provider_id": provider_id,
            "source": request.source,
            "reason": request.reason,
            "idempotency_key": request.idempotency_key,
            "request_digest": request_digest,
        }
        atomic = getattr(self.store, "upsert_provider_with_audit", None)
        if atomic:
            provider = atomic(
                provider_id=provider_id,
                display_name=request.display_name,
                adapter_type=request.adapter_type,
                approved_base_url=approved_base_url,
                capabilities=request.capabilities,
                byok_enabled=request.byok_enabled,
                platform_enabled=request.platform_enabled,
                status=request.status,
                audit=audit,
            )
        else:
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
                **audit,
                result_json=json.dumps(
                    ProviderRecord.model_validate(provider).model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
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
        secret = request.api_key.get_secret_value()
        fingerprint = secret_fingerprint(secret)
        request_digest = canonical_request_digest(
            {
                "operation": "set_platform_credential",
                "actor_user_id": int(admin_user_id),
                "provider_id": provider_id,
                "secret_fingerprint": fingerprint,
                "source": request.source,
                "reason": request.reason,
            }
        )
        replay = self._admin_operation_replayed(
            admin_user_id,
            "set_platform_credential",
            provider_id,
            request.idempotency_key,
            request_digest,
        )
        if replay:
            credential = self._replay_snapshot(replay) or self.store.get_platform_credential_public(provider_id)
            if not credential:
                raise ProviderNotFoundError("platform credential not found")
            return AdminPlatformCredentialPublic.model_validate(credential)
        validation = self._validate_credential(provider, secret)
        audit = {
            "actor_user_id": admin_user_id,
            "operation": "set_platform_credential",
            "provider_id": provider_id,
            "source": request.source,
            "reason": request.reason,
            "idempotency_key": request.idempotency_key,
            "request_digest": request_digest,
            "secret_fingerprint": fingerprint,
        }
        atomic = getattr(self.store, "upsert_platform_credential_with_audit", None)
        if atomic:
            result = atomic(
                provider_id=provider_id,
                secret=secret,
                status=validation.status,
                last_verified_at=_utcnow_iso() if validation.status == "verified" else None,
                audit=audit,
            )
        else:
            result = self.store.upsert_platform_credential(
                provider_id=provider_id,
                secret=secret,
                status=validation.status,
                last_verified_at=_utcnow_iso() if validation.status == "verified" else None,
            )
            self.store.record_admin_operation(
                **audit,
                result_json=json.dumps(result, sort_keys=True, separators=(",", ":")),
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
        secret = self.store.get_platform_credential_secret_any_status(provider_id)
        if not secret:
            raise ProviderNotFoundError("platform credential not found")
        fingerprint = secret_fingerprint(secret)
        request_digest = canonical_request_digest(
            {
                "operation": "reverify_platform_credential",
                "actor_user_id": int(admin_user_id),
                "provider_id": provider_id,
                "secret_fingerprint": fingerprint,
                "source": source,
                "reason": reason,
            }
        )
        replay = self._admin_operation_replayed(
            admin_user_id,
            "reverify_platform_credential",
            provider_id,
            idempotency_key,
            request_digest,
        )
        if replay:
            credential = self._replay_snapshot(replay) or self.store.get_platform_credential_public(provider_id)
            if not credential:
                raise ProviderNotFoundError("platform credential not found")
            return AdminPlatformCredentialPublic.model_validate(credential)
        validation = self._validate_credential(provider, secret)
        audit = {
            "actor_user_id": admin_user_id,
            "operation": "reverify_platform_credential",
            "provider_id": provider_id,
            "source": source,
            "reason": reason,
            "idempotency_key": idempotency_key,
            "request_digest": request_digest,
            "secret_fingerprint": fingerprint,
        }
        atomic = getattr(self.store, "set_platform_credential_status_with_audit", None)
        if atomic:
            result = atomic(
                provider_id=provider_id,
                status=validation.status,
                last_verified_at=_utcnow_iso() if validation.status == "verified" else None,
                audit=audit,
            )
        else:
            result = self.store.set_platform_credential_status(
                provider_id,
                status=validation.status,
                last_verified_at=_utcnow_iso() if validation.status == "verified" else None,
            )
            self.store.record_admin_operation(
                **audit,
                result_json=json.dumps(result, sort_keys=True, separators=(",", ":")),
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
        request_digest = canonical_request_digest(
            {
                "operation": "revoke_platform_credential",
                "actor_user_id": int(admin_user_id),
                "provider_id": provider_id,
                "source": source,
                "reason": reason,
            }
        )
        replay = self._admin_operation_replayed(
            admin_user_id,
            "revoke_platform_credential",
            provider_id,
            idempotency_key,
            request_digest,
        )
        if replay:
            return True
        audit = {
            "actor_user_id": admin_user_id,
            "operation": "revoke_platform_credential",
            "provider_id": provider_id,
            "source": source,
            "reason": reason,
            "idempotency_key": idempotency_key,
            "request_digest": request_digest,
        }
        atomic = getattr(self.store, "delete_platform_credential_with_audit", None)
        if atomic:
            atomic(audit=audit)
        else:
            deleted = self.store.delete_platform_credential(provider_id)
            if not deleted:
                raise ProviderNotFoundError("platform credential not found")
            self.store.record_admin_operation(
                **audit,
                result_json=json.dumps({"revoked": True}),
            )
        return True

    def resolve_user_default_credential(
        self, user_id: int, provider_id: str
    ) -> ResolvedCredential:
        """Resolve the caller's one verified default BYOK credential."""

        provider = self.store.get_provider(provider_id)
        if (
            not provider
            or provider["status"] != "enabled"
            or not provider["byok_enabled"]
        ):
            raise CredentialResolutionError(
                ExecutionErrorCategory.CREDENTIAL_MISSING
            )
        credential = self.store.get_verified_default_user_credential(
            int(user_id), provider_id
        )
        if not credential:
            raise CredentialResolutionError(ExecutionErrorCategory.CREDENTIAL_MISSING)
        return ResolvedCredential(
            credential_id=str(credential["credential_id"]),
            provider_id=str(credential["provider_id"]),
            key_last_four=str(credential["key_last_four"])[-4:],
            secret=str(credential["secret"]),
        )

    def preflight_execution_model(
        self,
        provider_id: str,
        catalog_model_id: str,
    ) -> ExecutionModelRoute:
        provider = self.store.get_provider(provider_id)
        if not provider or provider["status"] != "enabled":
            raise ProviderNotFoundError("provider not found")
        return resolve_execution_model_route(
            ProviderRecord.model_validate(provider),
            catalog_model_id,
        )

    def resolve_platform_credential(self, provider_id: str) -> ResolvedCredential:
        """Resolve the enabled provider's one verified platform credential."""

        provider = self.store.get_provider(provider_id)
        if (
            not provider
            or provider["status"] != "enabled"
            or not provider["platform_enabled"]
        ):
            raise CredentialResolutionError(
                ExecutionErrorCategory.CREDENTIAL_MISSING
            )
        credential = self.store.get_verified_platform_credential(provider_id)
        if credential:
            return ResolvedCredential(
                credential_id=None,
                provider_id=str(credential["provider_id"]),
                key_last_four=str(credential["key_last_four"])[-4:],
                secret=str(credential["secret"]),
            )
        secret = _environment_platform_secret(provider_id)
        if secret:
            return ResolvedCredential(
                credential_id=None,
                provider_id=provider_id,
                key_last_four=secret[-4:],
                secret=secret,
            )
        raise CredentialResolutionError(ExecutionErrorCategory.CREDENTIAL_MISSING)

    def preflight_user_default_credential(
        self, user_id: int, provider_id: str
    ) -> None:
        """Check a BYOK lane without decrypting its credential.

        The API performs this before it starts a worker. The worker is the
        only process that later calls ``resolve_user_default_credential`` and
        holds a transient plaintext key.
        """

        provider = self.store.get_provider(provider_id)
        if (
            not provider
            or provider["status"] != "enabled"
            or not provider["byok_enabled"]
        ):
            raise CredentialResolutionError(
                ExecutionErrorCategory.CREDENTIAL_MISSING
            )
        matching = [
            credential
            for credential in self.store.list_user_credentials(
                int(user_id), provider_id
            )
            if credential["status"] == "verified" and credential["is_default"]
        ]
        if len(matching) != 1:
            raise CredentialResolutionError(
                ExecutionErrorCategory.CREDENTIAL_MISSING
            )

    def preflight_platform_credential(self, provider_id: str) -> None:
        """Check a Platform Credits lane without decrypting its credential."""

        provider = self.store.get_provider(provider_id)
        if (
            not provider
            or provider["status"] != "enabled"
            or not provider["platform_enabled"]
        ):
            raise CredentialResolutionError(
                ExecutionErrorCategory.CREDENTIAL_MISSING
            )
        credential = self.store.get_platform_credential_public(provider_id)
        if credential and credential["status"] == "verified":
            return
        if _environment_platform_secret(provider_id):
            return
        if not credential or credential["status"] != "verified":
            raise CredentialResolutionError(
                ExecutionErrorCategory.CREDENTIAL_MISSING
            )

    def resolve_platform_secret(self, provider_id: str) -> str | None:
        """Legacy nullable wrapper retained for discovery-only callers."""

        try:
            return self.resolve_platform_credential(provider_id).secret
        except CredentialResolutionError:
            return None

    def _admin_operation_replayed(
        self,
        admin_user_id: int,
        operation: str,
        provider_id: str,
        idempotency_key: str,
        request_digest: str,
    ) -> dict[str, Any] | None:
        existing = self.store.get_admin_operation(idempotency_key)
        if not existing:
            return None
        if (
            int(existing["actor_user_id"]) != int(admin_user_id)
            or existing["operation"] != operation
            or existing["provider_id"] != provider_id
        ):
            raise CredentialConflictError("idempotency key already used")
        if "request_digest" in existing and existing.get("request_digest") != request_digest:
            raise CredentialConflictError("idempotency key already used for different input")
        return existing

    def _validate_credential(
        self, provider: dict[str, Any], secret: str
    ) -> CredentialValidation:
        adapter = self.adapter_resolver(provider["adapter_type"])
        try:
            return adapter.validate(
                provider["approved_base_url"], secret, client=self.http_client
            )
        except Exception:
            return CredentialValidation(
                status="verification_unavailable",
                message="Provider verification was unavailable.",
            )

    @staticmethod
    def _replay_snapshot(existing: dict[str, Any]) -> dict[str, Any] | None:
        raw = existing.get("result_json")
        if not raw:
            return None
        try:
            snapshot = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return snapshot if isinstance(snapshot, dict) else None

    def create_credential(
        self,
        user_id: int,
        request: UserCredentialCreate,
    ) -> UserCredentialPublic:
        provider = self._get_byok_provider(request.provider_id)
        secret = request.api_key.get_secret_value()
        ensure_credential_encryption_ready()
        validation = self._validate_credential(provider, secret)
        created = self.store.create_user_credential(
            user_id=user_id,
            provider_id=provider["provider_id"],
            label=request.label,
            secret=secret,
            status=validation.status,
            verification_message=_safe_verification_message(validation),
            set_default=request.set_default,
            last_verified_at=(
                _utcnow_iso() if validation.status == "verified" else None
            ),
        )
        result = UserCredentialPublic.model_validate(created)
        analytics_instrumentation.emit_credential_event(
            event_name="credential_saved",
            user_id=user_id,
            credential_id=str(result.credential_id),
            provider_id=result.provider_id,
        )
        if result.status == "verified":
            analytics_instrumentation.emit_credential_event(
                event_name="credential_verified",
                user_id=user_id,
                credential_id=str(result.credential_id),
                provider_id=result.provider_id,
            )
            if result.is_default:
                analytics_instrumentation.emit_credential_event(
                    event_name="credential_defaulted",
                    user_id=user_id,
                    credential_id=str(result.credential_id),
                    provider_id=result.provider_id,
                    version=result.updated_at,
                )
        elif result.status == "invalid":
            analytics_instrumentation.emit_safe_error_event(
                user_id=user_id,
                source_record_type="credential",
                source_record_id=str(result.credential_id),
                error_category="credential_invalid",
                version=result.updated_at,
            )
        elif result.status == "verification_unavailable":
            analytics_instrumentation.emit_safe_error_event(
                user_id=user_id,
                source_record_type="credential",
                source_record_id=str(result.credential_id),
                error_category="provider_unavailable",
                version=result.updated_at,
            )
        return result

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
        result = self._verify_and_update(
            user_id=user_id,
            credential=credential,
            provider=provider,
            secret=secret,
            set_default=False,
        )
        if result.status == "verified":
            analytics_instrumentation.emit_credential_event(
                event_name="credential_reverified",
                user_id=user_id,
                credential_id=str(result.credential_id),
                provider_id=result.provider_id,
                version=result.updated_at,
            )
        else:
            analytics_instrumentation.emit_safe_error_event(
                user_id=user_id,
                source_record_type="credential",
                source_record_id=str(result.credential_id),
                error_category=(
                    "credential_invalid"
                    if result.status == "invalid"
                    else "provider_unavailable"
                ),
                version=result.updated_at,
            )
        return result

    def set_default_credential(
        self,
        user_id: int,
        credential_id: str,
    ) -> UserCredentialPublic:
        result = UserCredentialPublic.model_validate(
            self.store.set_default_user_credential(user_id, credential_id)
        )
        analytics_instrumentation.emit_credential_event(
            event_name="credential_defaulted",
            user_id=user_id,
            credential_id=str(result.credential_id),
            provider_id=result.provider_id,
            version=result.updated_at,
        )
        return result

    def revoke_credential(
        self,
        user_id: int,
        credential_id: str,
    ) -> UserCredentialPublic:
        result = UserCredentialPublic.model_validate(
            self.store.revoke_user_credential(user_id, credential_id)
        )
        analytics_instrumentation.emit_credential_event(
            event_name="credential_revoked",
            user_id=user_id,
            credential_id=str(result.credential_id),
            provider_id=result.provider_id,
        )
        return result

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
        validation = self._validate_credential(provider, secret)
        updated = self.store.set_user_credential_status(
            user_id,
            credential["credential_id"],
            status=validation.status,
            verification_message=_safe_verification_message(validation),
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


def get_model_provider_service() -> ModelProviderService:
    """Return the application service used by credential API dependencies."""

    return model_provider_service
