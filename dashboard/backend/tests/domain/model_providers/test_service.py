"""Credential lifecycle service tests with fake, non-billable adapters."""

from __future__ import annotations

import sqlite3

import pytest
from cryptography.fernet import Fernet

from dashboard.backend.domain.brokers import repository as broker_repository
from dashboard.backend.domain.model_providers.models import (
    CredentialValidation,
    UserCredentialCreate,
)
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.model_providers.repository_common import (
    CredentialConflictError,
    CredentialOwnershipError,
)
from dashboard.backend.domain.model_providers.service import ModelProviderService


class FakeAdapter:
    def __init__(self, *results: CredentialValidation):
        self.results = list(results)
        self.calls: list[tuple[str, str]] = []

    def validate(self, base_url: str, secret: str, *, client=None) -> CredentialValidation:
        self.calls.append((base_url, secret))
        return self.results.pop(0)


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch):
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)


def _service(tmp_path, adapter: FakeAdapter) -> tuple[ModelProviderService, ModelProviderStore]:
    store = ModelProviderStore(tmp_path / "model-provider-service.db")
    return (
        ModelProviderService(
            store=store,
            adapter_resolver=lambda _adapter_type: adapter,
        ),
        store,
    )


def _request(*, label: str = "Research", set_default: bool = False) -> UserCredentialCreate:
    return UserCredentialCreate(
        provider_id="openrouter",
        label=label,
        api_key="sk-or-fake-service-abcd",
        set_default=set_default,
    )


def _validation(status: str) -> CredentialValidation:
    return CredentialValidation(
        status=status,
        message={
            "verified": "API key verified.",
            "invalid": "The provider rejected this API key.",
            "verification_unavailable": "Provider verification was unavailable.",
        }[status],
        models=["fake/model"] if status == "verified" else [],
    )


def test_create_encrypts_verifies_and_returns_only_public_metadata(tmp_path):
    adapter = FakeAdapter(_validation("verified"))
    service, store = _service(tmp_path, adapter)

    created = service.create_credential(7, _request(set_default=True))

    assert created.status == "verified"
    assert created.is_default is True
    assert created.key_last_four == "abcd"
    assert created.last_verified_at is not None
    assert adapter.calls == [
        ("https://openrouter.ai/api/v1", "sk-or-fake-service-abcd")
    ]
    serialized = created.model_dump_json()
    assert "sk-or-fake-service-abcd" not in serialized
    assert "api_key" not in serialized
    with sqlite3.connect(store.db_path) as conn:
        stored = conn.execute(
            "SELECT api_key_enc FROM user_model_credentials WHERE credential_id = ?",
            (str(created.credential_id),),
        ).fetchone()[0]
    assert stored != "sk-or-fake-service-abcd"
    assert "sk-or-fake-service-abcd" not in stored


@pytest.mark.parametrize("status", ["invalid", "verification_unavailable"])
def test_create_preserves_non_verified_outcome_without_default(tmp_path, status):
    service, _store = _service(tmp_path, FakeAdapter(_validation(status)))

    created = service.create_credential(7, _request(set_default=True))

    assert created.status == status
    assert created.is_default is False
    assert created.last_verified_at is None


def test_create_requires_configured_encryption_key(tmp_path, monkeypatch):
    service, store = _service(tmp_path, FakeAdapter(_validation("verified")))
    monkeypatch.delenv("BROKER_TOKEN_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)

    with pytest.raises(RuntimeError, match="BROKER_TOKEN_ENCRYPTION_KEY is not set"):
        service.create_credential(7, _request())

    assert store.list_user_credentials(7) == []


def test_reverify_updates_status_and_can_then_become_default(tmp_path):
    adapter = FakeAdapter(
        _validation("verification_unavailable"),
        _validation("verified"),
    )
    service, _store = _service(tmp_path, adapter)
    created = service.create_credential(7, _request())

    verified = service.reverify_credential(7, str(created.credential_id))
    defaulted = service.set_default_credential(7, str(created.credential_id))

    assert verified.status == "verified"
    assert verified.last_verified_at is not None
    assert defaulted.is_default is True
    assert len(adapter.calls) == 2


def test_only_one_verified_default_exists_per_user_and_provider(tmp_path):
    adapter = FakeAdapter(_validation("verified"), _validation("verified"))
    service, _store = _service(tmp_path, adapter)
    first = service.create_credential(7, _request(label="Research", set_default=True))
    second = service.create_credential(7, _request(label="Personal", set_default=True))

    listed = service.list_credentials(7)

    assert {item.label for item in listed} == {"Research", "Personal"}
    assert [item.credential_id for item in listed if item.is_default] == [
        second.credential_id
    ]
    assert first.credential_id != second.credential_id


def test_unverified_credential_cannot_be_default(tmp_path):
    service, _store = _service(tmp_path, FakeAdapter(_validation("invalid")))
    created = service.create_credential(7, _request())

    with pytest.raises(CredentialConflictError, match="only verified"):
        service.set_default_credential(7, str(created.credential_id))


@pytest.mark.parametrize("operation", ["reverify", "default", "revoke"])
def test_credential_mutations_enforce_user_ownership(tmp_path, operation):
    adapter = FakeAdapter(_validation("verified"))
    service, _store = _service(tmp_path, adapter)
    created = service.create_credential(7, _request())

    with pytest.raises(CredentialOwnershipError):
        if operation == "reverify":
            service.reverify_credential(8, str(created.credential_id))
        elif operation == "default":
            service.set_default_credential(8, str(created.credential_id))
        else:
            service.revoke_credential(8, str(created.credential_id))


def test_revoke_removes_credential_from_active_list(tmp_path):
    service, _store = _service(tmp_path, FakeAdapter(_validation("verified")))
    created = service.create_credential(7, _request(set_default=True))

    revoked = service.revoke_credential(7, str(created.credential_id))

    assert revoked.status == "revoked"
    assert revoked.is_default is False
    assert service.list_credentials(7) == []


def test_list_providers_returns_only_enabled_byok_records(tmp_path):
    service, _store = _service(tmp_path, FakeAdapter())

    providers = service.list_providers()

    assert {provider.provider_id for provider in providers} == {
        "anthropic",
        "gemini",
        "openai",
        "openrouter",
    }
    assert all(provider.byok_enabled for provider in providers)
