"""Admin provider registry and platform credential service tests."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from dashboard.backend.domain.brokers import repository as broker_repository
from dashboard.backend.domain.model_providers.models import (
    AdminPlatformCredentialRequest,
    AdminProviderRequest,
    CredentialValidation,
    ProviderCapabilities,
)
from dashboard.backend.domain.model_providers.repository_common import (
    CredentialConflictError,
    InvalidProviderOriginError,
    ProviderNotFoundError,
)
from dashboard.backend.domain.model_providers.service import ModelProviderService


class FakeAdapter:
    def __init__(self, *results: CredentialValidation):
        self.results = list(results)
        self.calls: list[tuple[str, str]] = []

    def validate(self, base_url: str, secret: str, *, client=None) -> CredentialValidation:
        self.calls.append((base_url, secret))
        return self.results.pop(0)


class FakeAdminStore:
    def __init__(self):
        self.providers = {
            "openrouter": {
                "provider_id": "openrouter",
                "display_name": "OpenRouter",
                "adapter_type": "openrouter",
                "approved_base_url": "https://openrouter.ai/api/v1",
                "capabilities": ProviderCapabilities().model_dump(),
                "byok_enabled": True,
                "platform_enabled": False,
                "status": "enabled",
                "created_at": "2026-08-19T00:00:00+00:00",
                "updated_at": "2026-08-19T00:00:00+00:00",
            }
        }
        self.platform: dict[str, dict] = {}
        self.audit: list[dict] = []

    def get_provider(self, provider_id):
        return self.providers.get(provider_id)

    def list_all_providers(self):
        return list(self.providers.values())

    def upsert_provider(self, **values):
        current = self.providers.get(values["provider_id"], {})
        current.update(values)
        self.providers[values["provider_id"]] = current
        return current

    def upsert_platform_credential(self, **values):
        public = {
            "provider_id": values["provider_id"],
            "key_last_four": values.get("key_last_four") or values["secret"][-4:],
            "status": values.get("status", "verification_unavailable"),
            "updated_at": "2026-08-19T00:01:00+00:00",
            "last_verified_at": values.get("last_verified_at"),
        }
        self.platform[values["provider_id"]] = {**public, "secret": values["secret"]}
        return public

    def get_platform_credential_public(self, provider_id):
        value = self.platform.get(provider_id)
        return {key: value[key] for key in value if key != "secret"} if value else None

    def get_platform_credential_secret(self, provider_id):
        value = self.platform.get(provider_id)
        return value["secret"] if value and value["status"] == "verified" else None

    def get_platform_credential_secret_any_status(self, provider_id):
        value = self.platform.get(provider_id)
        return value["secret"] if value else None

    def delete_platform_credential(self, provider_id):
        return self.platform.pop(provider_id, None) is not None

    def set_platform_credential_status(self, provider_id, *, status, last_verified_at=None):
        value = self.platform[provider_id]
        value["status"] = status
        if last_verified_at is not None:
            value["last_verified_at"] = last_verified_at
        return {key: value[key] for key in value if key != "secret"}

    def record_admin_operation(self, **values):
        self.audit.append(values)

    def get_admin_operation(self, idempotency_key):
        return next(
            (
                operation
                for operation in self.audit
                if operation["idempotency_key"] == idempotency_key
            ),
            None,
        )


def _verified() -> CredentialValidation:
    return CredentialValidation(status="verified", message="API key verified.")


def _unavailable() -> CredentialValidation:
    return CredentialValidation(
        status="verification_unavailable",
        message="Provider verification was unavailable.",
    )


def _provider_request(**overrides) -> AdminProviderRequest:
    values = {
        "display_name": "Approved Compatible",
        "adapter_type": "openai_compatible",
        "approved_base_url": "https://models.example.com/v1",
        "capabilities": ProviderCapabilities(model_discovery=True),
        "byok_enabled": True,
        "platform_enabled": True,
        "status": "enabled",
        "source": "admin-console",
        "reason": "Approved after provider review.",
        "idempotency_key": "provider-approval-001",
    }
    values.update(overrides)
    return AdminProviderRequest(**values)


def _credential_request(**overrides) -> AdminPlatformCredentialRequest:
    values = {
        "api_key": "sk-platform-fake-abcd",
        "source": "admin-console",
        "reason": "Configure platform model access.",
        "idempotency_key": "platform-key-001",
    }
    values.update(overrides)
    return AdminPlatformCredentialRequest(**values)


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch):
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)


def _service(store, adapter):
    return ModelProviderService(
        store=store,
        adapter_resolver=lambda _adapter_type: adapter,
    )


def test_admin_can_approve_provider_and_records_audit_metadata():
    store = FakeAdminStore()
    service = _service(store, FakeAdapter())

    provider = service.upsert_provider(11, "approved_compatible", _provider_request())

    assert provider.provider_id == "approved_compatible"
    assert provider.platform_enabled is True
    assert store.audit[-1]["actor_user_id"] == 11
    assert store.audit[-1]["source"] == "admin-console"
    assert store.audit[-1]["reason"] == "Approved after provider review."
    assert store.audit[-1]["idempotency_key"] == "provider-approval-001"


def test_admin_provider_request_rejects_private_and_non_https_origins():
    service = _service(FakeAdminStore(), FakeAdapter())
    for base_url in (
        "http://models.example.com/v1",
        "https://127.0.0.1/v1",
        "https://models.example.com/v1?token=secret",
    ):
        with pytest.raises(InvalidProviderOriginError):
            service.upsert_provider(
                11,
                "approved_compatible",
                _provider_request(approved_base_url=base_url),
            )


def test_platform_credential_is_verified_and_never_returns_plaintext():
    store = FakeAdminStore()
    adapter = FakeAdapter(_verified())
    service = _service(store, adapter)

    public = service.set_platform_credential(11, "openrouter", _credential_request())

    assert public.key_last_four == "abcd"
    assert public.status == "verified"
    assert "sk-platform-fake-abcd" not in public.model_dump_json()
    assert adapter.calls == [("https://openrouter.ai/api/v1", "sk-platform-fake-abcd")]
    assert store.audit[-1]["actor_user_id"] == 11


def test_transient_platform_verification_stays_unavailable():
    store = FakeAdminStore()
    service = _service(store, FakeAdapter(_unavailable()))

    public = service.set_platform_credential(11, "openrouter", _credential_request())

    assert public.status == "verification_unavailable"
    assert public.last_verified_at is None
    assert service.resolve_platform_secret("openrouter") is None


def test_transient_reverification_preserves_last_success_timestamp():
    store = FakeAdminStore()
    service = _service(store, FakeAdapter(_verified(), _unavailable()))
    created = service.set_platform_credential(
        11,
        "openrouter",
        _credential_request(idempotency_key="platform-create-history-001"),
    )

    unavailable = service.reverify_platform_credential(
        11,
        "openrouter",
        source="admin-console",
        reason="Provider was temporarily unavailable.",
        idempotency_key="platform-reverify-history-001",
    )

    assert created.last_verified_at is not None
    assert unavailable.status == "verification_unavailable"
    assert unavailable.last_verified_at == created.last_verified_at
    assert service.resolve_platform_secret("openrouter") is None


def test_reverify_platform_credential_requires_existing_key():
    store = FakeAdminStore()
    service = _service(store, FakeAdapter())

    with pytest.raises(ProviderNotFoundError):
        service.reverify_platform_credential(
            11,
            "openrouter",
            source="admin-console",
            reason="Retry verification.",
            idempotency_key="platform-reverify-001",
        )


def test_disable_provider_preserves_registry_but_blocks_byok_and_platform_use():
    store = FakeAdminStore()
    service = _service(store, FakeAdapter())

    service.upsert_provider(
        11,
        "openrouter",
        _provider_request(
            display_name="OpenRouter",
            adapter_type="openrouter",
            approved_base_url="https://openrouter.ai/api/v1",
            platform_enabled=False,
            status="disabled",
        ),
    )

    assert store.get_provider("openrouter")["status"] == "disabled"
    assert service.list_admin_providers()[0].status == "disabled"


def test_platform_credential_retry_is_idempotent_and_does_not_verify_twice():
    store = FakeAdminStore()
    adapter = FakeAdapter(_verified())
    service = _service(store, adapter)
    request = _credential_request()

    first = service.set_platform_credential(11, "openrouter", request)
    replayed = service.set_platform_credential(11, "openrouter", request)

    assert replayed == first
    assert len(adapter.calls) == 1
    assert len(store.audit) == 1


def test_idempotency_key_cannot_be_reused_for_another_operation():
    store = FakeAdminStore()
    service = _service(store, FakeAdapter())
    service.upsert_provider(11, "approved_compatible", _provider_request())

    with pytest.raises(CredentialConflictError, match="idempotency key"):
        service.revoke_platform_credential(
            11,
            "approved_compatible",
            source="admin-console",
            reason="Remove platform access.",
            idempotency_key="provider-approval-001",
        )
