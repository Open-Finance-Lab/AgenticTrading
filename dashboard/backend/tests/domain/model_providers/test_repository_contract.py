"""Backend-neutral lifecycle and migration contracts for the credential stores."""

from __future__ import annotations

import sqlite3

import pytest
from cryptography.fernet import Fernet

from dashboard.backend.domain.brokers import repository as broker_repository
from dashboard.backend.domain.brokers.repository import _encrypt
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.model_providers.execution_catalog import (
    UnsupportedExecutionModel,
    list_execution_model_routes,
    resolve_execution_model_route,
)
from dashboard.backend.domain.model_providers.models import ProviderRecord
from dashboard.backend.domain.model_providers.repository_common import CredentialConflictError


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch):
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)


@pytest.fixture
def store(tmp_path):
    return ModelProviderStore(tmp_path / "provider-contract.db")


@pytest.fixture
def store_factory(tmp_path):
    database_path = tmp_path / "provider-reopen.db"
    return lambda: ModelProviderStore(database_path)


def test_revoke_crypto_shreds_secret_and_allows_label_reuse(store):
    created = store.create_user_credential(
        user_id=7,
        provider_id="openai",
        label="Research",
        secret="sk-fake-research-abcd",
        status="verified",
        set_default=True,
    )
    revoked = store.revoke_user_credential(7, created["credential_id"])
    assert revoked["status"] == "revoked"
    assert revoked["key_last_four"] == "abcd"
    with pytest.raises(CredentialConflictError):
        store.get_user_credential_secret(7, created["credential_id"])

    raw = store._get_connection().execute(
        "SELECT api_key_enc FROM user_model_credentials WHERE credential_id = ?",
        (created["credential_id"],),
    ).fetchone()
    assert raw["api_key_enc"] is None

    replacement = store.create_user_credential(
        user_id=7,
        provider_id="openai",
        label="Research",
        secret="sk-fake-replacement-wxyz",
    )
    assert replacement["credential_id"] != created["credential_id"]


def test_seed_initialization_does_not_overwrite_admin_configuration(store_factory):
    store = store_factory()
    provider = store.get_provider("openai")
    store.upsert_provider(
        provider_id="openai",
        display_name="Approved OpenAI",
        adapter_type="openai",
        approved_base_url=provider["approved_base_url"],
        capabilities=provider["capabilities"],
        byok_enabled=False,
        platform_enabled=False,
        status="disabled",
    )
    reopened = store_factory()
    assert reopened.get_provider("openai")["display_name"] == "Approved OpenAI"
    assert reopened.get_provider("openai")["status"] == "disabled"


def test_seeded_openrouter_is_platform_enabled(store):
    assert store.get_provider("openrouter")["platform_enabled"] is True


def test_seeded_commonstack_is_platform_only_with_verified_model_allowlist(store):
    provider = store.get_provider("commonstack")

    assert provider["adapter_type"] == "openai_compatible"
    assert provider["approved_base_url"] == "https://api.commonstack.ai/v1"
    assert provider["byok_enabled"] is False
    assert provider["platform_enabled"] is True
    assert provider["capabilities"].model_allowlist == (
        "openai/gpt-5.5",
        "google/gemini-3.1-pro-preview",
        "anthropic/claude-sonnet-4-6",
        "deepseek/deepseek-v4-pro",
        "qwen/qwen3.7-plus",
    )


def test_commonstack_routes_only_the_verified_catalog_models(store):
    provider = ProviderRecord.model_validate(store.get_provider("commonstack"))

    assert [route.catalog_id for route in list_execution_model_routes(provider)] == [
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-5.5",
        "google/gemini-3.1-pro-preview",
        "deepseek/deepseek-v4-pro",
        "qwen/qwen3.7-plus",
    ]
    with pytest.raises(UnsupportedExecutionModel):
        resolve_execution_model_route(provider, "anthropic/claude-haiku-4-5")


def test_legacy_openrouter_platform_flag_migrates_when_environment_key_exists(
    store_factory, monkeypatch
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    store = store_factory()
    connection = store._get_connection()
    connection.execute(
        "UPDATE provider_registry SET platform_enabled = 0 WHERE provider_id = 'openrouter'"
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-migration-test-abcd")
    reopened = store_factory()

    assert reopened.get_provider("openrouter")["platform_enabled"] is True
    marker = reopened._get_connection().execute(
        "SELECT migration_id FROM model_provider_migrations"
    ).fetchone()
    assert marker[0] == "openrouter-platform-key-v1"


def test_legacy_migration_respects_admin_platform_disable(
    store_factory, monkeypatch
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    store = store_factory()
    provider = store.get_provider("openrouter")
    store.upsert_provider_with_audit(
        provider_id="openrouter",
        display_name=provider["display_name"],
        adapter_type=provider["adapter_type"],
        approved_base_url=provider["approved_base_url"],
        capabilities=provider["capabilities"],
        byok_enabled=True,
        platform_enabled=False,
        status="enabled",
        audit={
            "actor_user_id": 11,
            "operation": "upsert_provider",
            "provider_id": "openrouter",
            "source": "admin-console",
            "reason": "Disable platform access for maintenance.",
            "idempotency_key": "openrouter-disable-migration-test",
        },
    )

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-migration-test-abcd")
    reopened = store_factory()

    assert reopened.get_provider("openrouter")["platform_enabled"] is False
    assert (
        reopened._get_connection()
        .execute("SELECT COUNT(*) FROM model_provider_migrations")
        .fetchone()[0]
        == 0
    )


def test_legacy_migration_does_not_override_existing_platform_credential(
    store_factory, monkeypatch
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    store = store_factory()
    provider = store.get_provider("openrouter")
    store.upsert_provider(
        provider_id="openrouter",
        display_name=provider["display_name"],
        adapter_type=provider["adapter_type"],
        approved_base_url=provider["approved_base_url"],
        capabilities=provider["capabilities"],
        byok_enabled=True,
        platform_enabled=False,
        status="enabled",
    )
    store.upsert_platform_credential(
        provider_id="openrouter",
        secret="sk-or-existing-platform-abcd",
        status="verified",
    )

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-migration-test-wxyz")
    reopened = store_factory()

    assert reopened.get_provider("openrouter")["platform_enabled"] is False


def test_platform_revoke_crypto_shreds_secret(store):
    store.upsert_platform_credential(
        provider_id="openai",
        secret="sk-fake-platform-abcd",
        status="verified",
    )
    assert store.delete_platform_credential("openai") is True
    public = store.get_platform_credential_public("openai")
    assert public["status"] == "revoked"
    assert store.get_platform_credential_secret_any_status("openai") is None
    raw = store._get_connection().execute(
        "SELECT api_key_enc FROM platform_model_credentials WHERE provider_id = 'openai'"
    ).fetchone()
    assert raw["api_key_enc"] is None


def test_legacy_sqlite_schema_is_migrated_without_losing_active_rows(tmp_path):
    database_path = tmp_path / "legacy-provider.db"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE provider_registry (
            provider_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            adapter_type TEXT NOT NULL,
            approved_base_url TEXT NOT NULL,
            capabilities_json TEXT NOT NULL,
            byok_enabled INTEGER NOT NULL,
            platform_enabled INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE user_model_credentials (
            credential_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            provider_id TEXT NOT NULL,
            label TEXT NOT NULL,
            api_key_enc TEXT NOT NULL,
            key_last_four TEXT NOT NULL,
            status TEXT NOT NULL,
            is_default INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_verified_at TEXT,
            revoked_at TEXT,
            UNIQUE(user_id, provider_id, label)
        );
        CREATE TABLE platform_model_credentials (
            provider_id TEXT PRIMARY KEY,
            api_key_enc TEXT NOT NULL,
            key_last_four TEXT NOT NULL,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_verified_at TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO provider_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "openai",
            "OpenAI",
            "openai",
            "https://api.openai.com/v1",
            '{"model_discovery": true}',
            1,
            0,
            "enabled",
            "2026-08-22T00:00:00+00:00",
            "2026-08-22T00:00:00+00:00",
        ),
    )
    connection.execute(
        "INSERT INTO user_model_credentials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "active-id",
            7,
            "openai",
            "Research",
            _encrypt("sk-legacy-active-abcd"),
            "abcd",
            "verified",
            1,
            "2026-08-22T00:00:00+00:00",
            "2026-08-22T00:00:00+00:00",
            "2026-08-22T00:00:00+00:00",
            None,
        ),
    )
    connection.execute(
        "INSERT INTO user_model_credentials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "revoked-id",
            7,
            "openai",
            "Old",
            _encrypt("sk-legacy-revoked-wxyz"),
            "wxyz",
            "revoked",
            0,
            "2026-08-22T00:00:00+00:00",
            "2026-08-22T00:00:00+00:00",
            None,
            "2026-08-22T00:00:00+00:00",
        ),
    )
    connection.commit()
    connection.close()

    migrated = ModelProviderStore(database_path)
    assert migrated.get_user_credential_secret(7, "active-id") == "sk-legacy-active-abcd"
    with pytest.raises(CredentialConflictError):
        migrated.get_user_credential_secret(7, "revoked-id")
    assert migrated.create_user_credential(
        user_id=7,
        provider_id="openai",
        label="Old",
        secret="sk-reused-label-1234",
    )["label"] == "Old"
