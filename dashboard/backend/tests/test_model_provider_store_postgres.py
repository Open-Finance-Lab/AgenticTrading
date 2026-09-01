"""PostgreSQL contract tests for the user model credential vault."""

from __future__ import annotations

import os

import psycopg
import pytest
from cryptography.fernet import Fernet

from dashboard.backend.domain.brokers import repository as broker_repository
from dashboard.backend.domain.model_providers.models import (
    ProviderCapabilities,
    ProviderRecord,
)
from dashboard.backend.domain.model_providers.execution_catalog import (
    UnsupportedExecutionModel,
    list_execution_model_routes,
    resolve_execution_model_route,
)
from dashboard.backend.domain.model_providers.repository_common import (
    CredentialConflictError,
    CredentialOwnershipError,
)
from dashboard.backend.domain.model_providers.repository_postgres import (
    PostgresModelProviderStore,
)
from dashboard.backend.tests._postgres_testing import require_local_postgres_url


TEST_POSTGRES_URL = require_local_postgres_url(os.getenv("TEST_POSTGRES_URL"))
pg_only = pytest.mark.skipif(
    not TEST_POSTGRES_URL,
    reason="TEST_POSTGRES_URL is not configured",
)


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch):
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)


@pytest.fixture
def postgres_store():
    require_local_postgres_url(TEST_POSTGRES_URL)
    with psycopg.connect(TEST_POSTGRES_URL, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS model_provider_admin_operations CASCADE")
            cur.execute("DROP TABLE IF EXISTS platform_model_credentials CASCADE")
            cur.execute("DROP TABLE IF EXISTS user_model_credentials CASCADE")
            cur.execute("DROP TABLE IF EXISTS provider_registry CASCADE")
    store = PostgresModelProviderStore(TEST_POSTGRES_URL)
    yield store
    with psycopg.connect(TEST_POSTGRES_URL, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS model_provider_admin_operations CASCADE")
            cur.execute("DROP TABLE IF EXISTS platform_model_credentials CASCADE")
            cur.execute("DROP TABLE IF EXISTS user_model_credentials CASCADE")
            cur.execute("DROP TABLE IF EXISTS provider_registry CASCADE")


def test_postgres_store_rejects_non_postgres_url_before_connecting():
    with pytest.raises(ValueError, match="postgresql://"):
        PostgresModelProviderStore("sqlite:///tmp/not-postgres.db")


@pg_only
def test_postgres_seeded_commonstack_is_platform_only_with_allowlisted_models(
    postgres_store,
):
    provider = postgres_store.get_provider("commonstack")

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

    provider_record = ProviderRecord.model_validate(provider)
    assert [
        route.catalog_id for route in list_execution_model_routes(provider_record)
    ] == [
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-5.5",
        "google/gemini-3.1-pro-preview",
        "deepseek/deepseek-v4-pro",
        "qwen/qwen3.7-plus",
    ]
    with pytest.raises(UnsupportedExecutionModel):
        resolve_execution_model_route(provider_record, "anthropic/claude-haiku-4-5")


@pg_only
def test_postgres_user_credential_lifecycle(postgres_store):
    first = postgres_store.create_user_credential(
        user_id=7,
        provider_id="openrouter",
        label="Research",
        secret="sk-or-fake-postgres-one",
        status="verified",
        set_default=True,
        last_verified_at="2026-08-19T00:00:00+00:00",
    )
    second = postgres_store.create_user_credential(
        user_id=7,
        provider_id="openrouter",
        label="Personal",
        secret="sk-or-fake-postgres-two",
        status="verified",
        set_default=True,
        last_verified_at="2026-08-19T00:01:00+00:00",
    )

    listed = postgres_store.list_user_credentials(7)

    assert {item["label"] for item in listed} == {"Research", "Personal"}
    assert [item["credential_id"] for item in listed if item["is_default"]] == [
        second["credential_id"]
    ]
    assert postgres_store.get_user_credential_secret(
        7, first["credential_id"]
    ) == "sk-or-fake-postgres-one"
    assert "sk-or-fake-postgres-one" not in repr(first)

    with pytest.raises(CredentialOwnershipError):
        postgres_store.get_user_credential_secret(8, first["credential_id"])

    revoked = postgres_store.revoke_user_credential(7, second["credential_id"])
    assert revoked["status"] == "revoked"
    assert revoked["is_default"] is False
    assert {item["credential_id"] for item in postgres_store.list_user_credentials(7)} == {
        first["credential_id"]
    }


@pg_only
def test_postgres_duplicate_label_and_verified_default_rules(postgres_store):
    created = postgres_store.create_user_credential(
        user_id=7,
        provider_id="openai",
        label="Primary",
        secret="sk-fake-postgres-primary",
        status="invalid",
    )

    with pytest.raises(CredentialConflictError, match="only verified"):
        postgres_store.set_default_user_credential(7, created["credential_id"])
    with pytest.raises(CredentialConflictError, match="label"):
        postgres_store.create_user_credential(
            user_id=7,
            provider_id="openai",
            label="Primary",
            secret="sk-fake-postgres-duplicate",
        )


@pg_only
def test_postgres_approved_compatible_provider_and_platform_secret(postgres_store):
    provider = postgres_store.upsert_provider(
        provider_id="approved_compatible",
        display_name="Approved Compatible",
        adapter_type="openai_compatible",
        approved_base_url="https://models.example.com/v1",
        capabilities=ProviderCapabilities(model_discovery=True),
        byok_enabled=True,
        platform_enabled=True,
    )
    public = postgres_store.upsert_platform_credential(
        provider_id=provider["provider_id"],
        secret="sk-fake-platform-postgres",
        status="verified",
        last_verified_at="2026-08-19T00:02:00+00:00",
    )

    assert public["key_last_four"] == "gres"
    assert "sk-fake-platform-postgres" not in repr(public)
    assert postgres_store.get_platform_credential_secret(
        "approved_compatible"
    ) == "sk-fake-platform-postgres"
    assert postgres_store.delete_platform_credential("approved_compatible") is True
    assert postgres_store.get_platform_credential_secret("approved_compatible") is None
