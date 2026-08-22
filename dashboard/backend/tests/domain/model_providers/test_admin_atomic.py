"""Admin mutation digest, fingerprint, replay, and transaction contracts."""

from __future__ import annotations

import json
import sqlite3

import pytest
from cryptography.fernet import Fernet

from dashboard.backend.domain.brokers import repository as broker_repository
from dashboard.backend.domain.model_providers.models import (
    AdminPlatformCredentialRequest,
    AdminProviderRequest,
    CredentialValidation,
    ProviderCapabilities,
)
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.model_providers.repository_common import (
    CredentialConflictError,
)
from dashboard.backend.domain.model_providers.service import ModelProviderService


class FakeAdapter:
    def __init__(self, *results: CredentialValidation):
        self.results = list(results)
        self.calls: list[str] = []

    def validate(self, base_url: str, secret: str, *, client=None):
        self.calls.append(secret)
        return self.results.pop(0)


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch):
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)


def _provider_request(**overrides) -> AdminProviderRequest:
    values = {
        "display_name": "Atomic Compatible",
        "adapter_type": "openai_compatible",
        "approved_base_url": "https://models.example.com/v1",
        "capabilities": ProviderCapabilities(model_discovery=True),
        "byok_enabled": True,
        "platform_enabled": True,
        "status": "enabled",
        "source": "admin-console",
        "reason": "Approve the provider for testing.",
        "idempotency_key": "atomic-provider-001",
    }
    values.update(overrides)
    return AdminProviderRequest(**values)


def _credential_request(**overrides) -> AdminPlatformCredentialRequest:
    values = {
        "api_key": "sk-fake-platform-atomic-abcd",
        "source": "admin-console",
        "reason": "Configure the platform credential.",
        "idempotency_key": "atomic-credential-001",
    }
    values.update(overrides)
    return AdminPlatformCredentialRequest(**values)


def _service(tmp_path, adapter):
    store = ModelProviderStore(tmp_path / "admin-atomic.db")
    return ModelProviderService(
        store=store,
        adapter_resolver=lambda _adapter_type: adapter,
    ), store


def test_admin_audit_binds_digest_and_hmac_without_secret(tmp_path):
    adapter = FakeAdapter(
        CredentialValidation(status="verified", message="fake verification")
    )
    service, store = _service(tmp_path, adapter)

    created = service.set_platform_credential(11, "openrouter", _credential_request())
    operation = store.get_admin_operation("atomic-credential-001")

    assert created.status == "verified"
    assert operation["request_digest"]
    assert len(operation["request_digest"]) == 64
    assert operation["secret_fingerprint"]
    assert len(operation["secret_fingerprint"]) == 64
    assert "sk-fake-platform-atomic-abcd" not in json.dumps(operation)
    assert json.loads(operation["result_json"])["key_last_four"] == "abcd"

    replayed = service.set_platform_credential(11, "openrouter", _credential_request())
    assert replayed == created
    assert len(adapter.calls) == 1


def test_idempotency_key_with_different_input_is_conflict(tmp_path):
    service, _store = _service(tmp_path, FakeAdapter())
    service.upsert_provider(11, "atomic_compatible", _provider_request())

    with pytest.raises(CredentialConflictError, match="different input"):
        service.upsert_provider(
            11,
            "atomic_compatible",
            _provider_request(display_name="Changed but reused key"),
        )


def test_replay_returns_original_safe_snapshot(tmp_path):
    service, store = _service(tmp_path, FakeAdapter())
    request = _provider_request()
    created = service.upsert_provider(11, "atomic_compatible", request)
    store.upsert_provider(
        provider_id="atomic_compatible",
        display_name="Mutated outside original operation",
        adapter_type="openai_compatible",
        approved_base_url="https://models.example.com/v1",
        capabilities=ProviderCapabilities(model_discovery=True),
        byok_enabled=True,
        platform_enabled=True,
        status="enabled",
    )

    replayed = service.upsert_provider(11, "atomic_compatible", request)

    assert replayed.display_name == created.display_name == "Atomic Compatible"


@pytest.mark.parametrize("operation", ["provider", "platform"])
def test_mutation_rolls_back_when_audit_insert_fails(tmp_path, monkeypatch, operation):
    adapter = FakeAdapter(
        CredentialValidation(status="verified", message="fake verification")
    )
    service, store = _service(tmp_path, adapter)

    def fail_audit(*_args, **_kwargs):
        raise RuntimeError("audit insert failed")

    monkeypatch.setattr(store, "_insert_admin_operation_in_transaction", fail_audit)

    with pytest.raises(RuntimeError, match="audit insert failed"):
        if operation == "provider":
            service.upsert_provider(11, "atomic_compatible", _provider_request())
        else:
            service.set_platform_credential(11, "openrouter", _credential_request())

    assert store.get_admin_operation("atomic-provider-001") is None
    assert store.get_admin_operation("atomic-credential-001") is None
    if operation == "provider":
        assert store.get_provider("atomic_compatible") is None
    else:
        assert store.get_platform_credential_public("openrouter") is None


def test_legacy_admin_operation_table_gains_digest_and_snapshot_columns(tmp_path):
    database_path = tmp_path / "legacy-admin-audit.db"
    with sqlite3.connect(database_path) as conn:
        conn.execute(
            """
            CREATE TABLE model_provider_admin_operations (
                operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_user_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                source TEXT NOT NULL,
                reason TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )

    store = ModelProviderStore(database_path)
    with store._get_connection() as conn:
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(model_provider_admin_operations)"
            ).fetchall()
        }

    assert {"request_digest", "secret_fingerprint", "result_json"} <= columns


def test_admin_secret_storage_fails_closed_without_encryption_key(tmp_path, monkeypatch):
    service, store = _service(
        tmp_path,
        FakeAdapter(CredentialValidation(status="verified", message="fake verification")),
    )
    monkeypatch.delenv("BROKER_TOKEN_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)

    with pytest.raises(RuntimeError, match="BROKER_TOKEN_ENCRYPTION_KEY"):
        service.set_platform_credential(11, "openrouter", _credential_request())

    assert store.get_platform_credential_public("openrouter") is None
    assert store.get_admin_operation("atomic-credential-001") is None
