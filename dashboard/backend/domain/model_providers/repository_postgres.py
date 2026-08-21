"""PostgreSQL provider registry and encrypted user credential vault."""

from __future__ import annotations

import uuid
from typing import Any

import psycopg

from dashboard.backend.db_url import require_postgres_url
from dashboard.backend.domain.agents.repository import _utcnow_iso
from dashboard.backend.domain.brokers.repository import _decrypt, _encrypt

from .models import ProviderCapabilities
from .repository_common import (
    CredentialConflictError,
    CredentialNotFoundError,
    CredentialOwnershipError,
    ProviderNotFoundError,
    SEEDED_PROVIDERS,
    deserialize_capabilities,
    serialize_capabilities,
    validate_adapter_type,
    validate_approved_origin,
)


MODEL_PROVIDERS_POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS provider_registry (
    provider_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    adapter_type TEXT NOT NULL,
    approved_base_url TEXT NOT NULL,
    capabilities_json TEXT NOT NULL,
    byok_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    platform_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    status TEXT NOT NULL DEFAULT 'enabled',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (adapter_type IN ('openrouter', 'openai', 'anthropic', 'gemini', 'openai_compatible')),
    CHECK (status IN ('enabled', 'disabled'))
);

CREATE TABLE IF NOT EXISTS user_model_credentials (
    credential_id TEXT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    provider_id TEXT NOT NULL,
    label TEXT NOT NULL,
    api_key_enc TEXT NOT NULL,
    key_last_four TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'verification_unavailable',
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_verified_at TEXT,
    revoked_at TEXT,
    UNIQUE(user_id, provider_id, label),
    FOREIGN KEY(provider_id) REFERENCES provider_registry(provider_id) ON DELETE RESTRICT,
    CHECK (status IN ('verified', 'invalid', 'verification_unavailable', 'revoked'))
);

CREATE INDEX IF NOT EXISTS idx_user_model_credentials_owner
ON user_model_credentials(user_id, provider_id, updated_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS uq_user_model_credentials_default
ON user_model_credentials(user_id, provider_id)
WHERE is_default = TRUE AND status = 'verified';

CREATE TABLE IF NOT EXISTS platform_model_credentials (
    provider_id TEXT PRIMARY KEY,
    api_key_enc TEXT NOT NULL,
    key_last_four TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'verification_unavailable',
    updated_at TEXT NOT NULL,
    last_verified_at TEXT,
    FOREIGN KEY(provider_id) REFERENCES provider_registry(provider_id) ON DELETE RESTRICT,
    CHECK (status IN ('verified', 'invalid', 'verification_unavailable', 'revoked'))
);

CREATE TABLE IF NOT EXISTS model_provider_admin_operations (
    operation_id BIGSERIAL PRIMARY KEY,
    actor_user_id BIGINT NOT NULL,
    operation TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    source TEXT NOT NULL,
    reason TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
"""


def _public_provider(row: dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    return {
        "provider_id": data["provider_id"],
        "display_name": data["display_name"],
        "adapter_type": data["adapter_type"],
        "approved_base_url": data["approved_base_url"],
        "capabilities": deserialize_capabilities(data.get("capabilities_json")),
        "byok_enabled": bool(data["byok_enabled"]),
        "platform_enabled": bool(data["platform_enabled"]),
        "status": data["status"],
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
    }


def _public_credential(row: dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    return {
        "credential_id": data["credential_id"],
        "provider_id": data["provider_id"],
        "label": data["label"],
        "key_last_four": data["key_last_four"],
        "status": data["status"],
        "is_default": bool(data["is_default"]),
        "created_at": data["created_at"],
        "updated_at": data["updated_at"],
        "last_verified_at": data["last_verified_at"],
    }


class PostgresModelProviderStore:
    """PostgreSQL twin of ``ModelProviderStore``."""

    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        self._init_schema()

    def _get_connection(self):
        from dashboard.backend.db_pool import get_pool

        return get_pool(self.database_url).connection()

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(MODEL_PROVIDERS_POSTGRES_DDL)
                now = _utcnow_iso()
                for item in SEEDED_PROVIDERS:
                    cur.execute(
                        """
                        INSERT INTO provider_registry (
                            provider_id, display_name, adapter_type, approved_base_url,
                            capabilities_json, byok_enabled, platform_enabled,
                            status, created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, TRUE, FALSE, 'enabled', %s, %s)
                        ON CONFLICT (provider_id) DO UPDATE SET
                            display_name = EXCLUDED.display_name,
                            adapter_type = EXCLUDED.adapter_type,
                            approved_base_url = EXCLUDED.approved_base_url,
                            capabilities_json = EXCLUDED.capabilities_json,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (
                            item["provider_id"],
                            item["display_name"],
                            item["adapter_type"],
                            item["approved_base_url"],
                            serialize_capabilities(item["capabilities"]),
                            now,
                            now,
                        ),
                    )

    def list_enabled_providers(self, *, mode: str = "byok") -> list[dict[str, Any]]:
        if mode not in {"byok", "platform"}:
            raise ValueError("unsupported provider mode")
        column = "byok_enabled" if mode == "byok" else "platform_enabled"
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM provider_registry WHERE status = 'enabled' AND {column} = TRUE ORDER BY display_name"
                )
                rows = cur.fetchall()
        return [_public_provider(row) for row in rows]

    def list_all_providers(self) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM provider_registry ORDER BY display_name")
                rows = cur.fetchall()
        return [_public_provider(row) for row in rows]

    def record_admin_operation(self, **values: Any) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_provider_admin_operations (
                        actor_user_id, operation, provider_id, source, reason,
                        idempotency_key, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (idempotency_key) DO NOTHING
                    """,
                    (
                        int(values["actor_user_id"]),
                        str(values["operation"]),
                        str(values["provider_id"]),
                        str(values["source"]),
                        str(values["reason"]),
                        str(values["idempotency_key"]),
                        _utcnow_iso(),
                    ),
                )

    def get_admin_operation(self, idempotency_key: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM model_provider_admin_operations WHERE idempotency_key = %s",
                    (str(idempotency_key),),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def get_provider(self, provider_id: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM provider_registry WHERE provider_id = %s",
                    (provider_id,),
                )
                row = cur.fetchone()
        return _public_provider(row) if row else None

    def upsert_provider(
        self,
        *,
        provider_id: str,
        display_name: str,
        adapter_type: str,
        approved_base_url: str,
        capabilities: ProviderCapabilities | dict[str, Any],
        byok_enabled: bool,
        platform_enabled: bool,
        status: str = "enabled",
    ) -> dict[str, Any]:
        validate_adapter_type(adapter_type)
        approved_base_url = validate_approved_origin(approved_base_url)
        if status not in {"enabled", "disabled"}:
            raise ValueError("invalid provider status")
        now = _utcnow_iso()
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO provider_registry (
                        provider_id, display_name, adapter_type, approved_base_url,
                        capabilities_json, byok_enabled, platform_enabled,
                        status, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (provider_id) DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        adapter_type = EXCLUDED.adapter_type,
                        approved_base_url = EXCLUDED.approved_base_url,
                        capabilities_json = EXCLUDED.capabilities_json,
                        byok_enabled = EXCLUDED.byok_enabled,
                        platform_enabled = EXCLUDED.platform_enabled,
                        status = EXCLUDED.status,
                        updated_at = EXCLUDED.updated_at
                    RETURNING *
                    """,
                    (
                        provider_id,
                        display_name.strip(),
                        adapter_type,
                        approved_base_url,
                        serialize_capabilities(capabilities),
                        bool(byok_enabled),
                        bool(platform_enabled),
                        status,
                        now,
                        now,
                    ),
                )
                row = cur.fetchone()
        return _public_provider(row)

    def create_user_credential(
        self,
        *,
        user_id: int,
        credential_id: str | None = None,
        provider_id: str,
        label: str,
        secret: str,
        key_last_four: str | None = None,
        status: str = "verification_unavailable",
        set_default: bool = False,
        last_verified_at: str | None = None,
    ) -> dict[str, Any]:
        provider = self.get_provider(provider_id)
        if not provider or provider["status"] != "enabled" or not provider["byok_enabled"]:
            raise ProviderNotFoundError("provider is not available for BYOK")
        if status not in {"verified", "invalid", "verification_unavailable", "revoked"}:
            raise ValueError("invalid credential status")
        credential_id = credential_id or str(uuid.uuid4())
        now = _utcnow_iso()
        encrypted = _encrypt(secret)
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    if set_default and status == "verified":
                        cur.execute(
                            "UPDATE user_model_credentials SET is_default = FALSE WHERE user_id = %s AND provider_id = %s",
                            (int(user_id), provider_id),
                        )
                    cur.execute(
                        """
                        INSERT INTO user_model_credentials (
                            credential_id, user_id, provider_id, label, api_key_enc,
                            key_last_four, status, is_default, created_at,
                            updated_at, last_verified_at, revoked_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING *
                        """,
                        (
                            str(credential_id),
                            int(user_id),
                            provider_id,
                            label.strip(),
                            encrypted,
                            (key_last_four or secret[-4:])[-4:],
                            status,
                            bool(set_default and status == "verified"),
                            now,
                            now,
                            last_verified_at,
                            now if status == "revoked" else None,
                        ),
                    )
                    row = cur.fetchone()
        except psycopg.IntegrityError as exc:
            raise CredentialConflictError("credential label already exists") from exc
        return _public_credential(row)

    def list_user_credentials(self, user_id: int, provider_id: str | None = None) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if provider_id:
                    cur.execute(
                        "SELECT * FROM user_model_credentials WHERE user_id = %s AND provider_id = %s AND status <> 'revoked' ORDER BY provider_id, is_default DESC, label",
                        (int(user_id), provider_id),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM user_model_credentials WHERE user_id = %s AND status <> 'revoked' ORDER BY provider_id, is_default DESC, label",
                        (int(user_id),),
                    )
                rows = cur.fetchall()
        return [_public_credential(row) for row in rows]

    def get_user_credential_public(self, user_id: int, credential_id: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM user_model_credentials WHERE credential_id = %s AND user_id = %s",
                    (str(credential_id), int(user_id)),
                )
                row = cur.fetchone()
        return _public_credential(row) if row else None

    def get_user_credential_secret(self, user_id: int, credential_id: str) -> str:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id, api_key_enc, status FROM user_model_credentials WHERE credential_id = %s",
                    (str(credential_id),),
                )
                row = cur.fetchone()
        if not row:
            raise CredentialNotFoundError("credential not found")
        if int(row["user_id"]) != int(user_id):
            raise CredentialOwnershipError("credential does not belong to this user")
        if row["status"] == "revoked":
            raise CredentialConflictError("revoked credentials cannot be read")
        return _decrypt(row["api_key_enc"])

    def set_user_credential_status(
        self,
        user_id: int,
        credential_id: str,
        *,
        status: str,
        last_verified_at: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"verified", "invalid", "verification_unavailable", "revoked"}:
            raise ValueError("invalid credential status")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM user_model_credentials WHERE credential_id = %s FOR UPDATE",
                    (str(credential_id),),
                )
                row = cur.fetchone()
                if not row:
                    raise CredentialNotFoundError("credential not found")
                if int(row["user_id"]) != int(user_id):
                    raise CredentialOwnershipError("credential does not belong to this user")
                now = _utcnow_iso()
                cur.execute(
                    """
                    UPDATE user_model_credentials
                    SET status = %s,
                        is_default = CASE WHEN %s <> 'verified' THEN FALSE ELSE is_default END,
                        updated_at = %s,
                        last_verified_at = COALESCE(%s, last_verified_at),
                        revoked_at = CASE WHEN %s = 'revoked' THEN %s ELSE revoked_at END
                    WHERE credential_id = %s
                    RETURNING *
                    """,
                    (status, status, now, last_verified_at, status, now, str(credential_id)),
                )
                result = cur.fetchone()
        return _public_credential(result)

    def set_default_user_credential(self, user_id: int, credential_id: str) -> dict[str, Any]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM user_model_credentials WHERE credential_id = %s FOR UPDATE",
                    (str(credential_id),),
                )
                row = cur.fetchone()
                if not row:
                    raise CredentialNotFoundError("credential not found")
                if int(row["user_id"]) != int(user_id):
                    raise CredentialOwnershipError("credential does not belong to this user")
                if row["status"] != "verified":
                    raise CredentialConflictError("only verified credentials can be default")
                cur.execute(
                    "UPDATE user_model_credentials SET is_default = FALSE WHERE user_id = %s AND provider_id = %s",
                    (int(user_id), row["provider_id"]),
                )
                cur.execute(
                    "UPDATE user_model_credentials SET is_default = TRUE, updated_at = %s WHERE credential_id = %s RETURNING *",
                    (_utcnow_iso(), str(credential_id)),
                )
                result = cur.fetchone()
        return _public_credential(result)

    def revoke_user_credential(self, user_id: int, credential_id: str) -> dict[str, Any]:
        return self.set_user_credential_status(user_id, credential_id, status="revoked")

    def upsert_platform_credential(
        self,
        *,
        provider_id: str,
        secret: str,
        key_last_four: str | None = None,
        status: str = "verification_unavailable",
        last_verified_at: str | None = None,
    ) -> dict[str, Any]:
        if not self.get_provider(provider_id):
            raise ProviderNotFoundError("provider not found")
        if status not in {"verified", "invalid", "verification_unavailable", "revoked"}:
            raise ValueError("invalid credential status")
        now = _utcnow_iso()
        encrypted = _encrypt(secret)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO platform_model_credentials (
                        provider_id, api_key_enc, key_last_four, status,
                        updated_at, last_verified_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (provider_id) DO UPDATE SET
                        api_key_enc = EXCLUDED.api_key_enc,
                        key_last_four = EXCLUDED.key_last_four,
                        status = EXCLUDED.status,
                        updated_at = EXCLUDED.updated_at,
                        last_verified_at = EXCLUDED.last_verified_at
                    RETURNING provider_id, key_last_four, status, updated_at, last_verified_at
                    """,
                    (
                        provider_id,
                        encrypted,
                        (key_last_four or secret[-4:])[-4:],
                        status,
                        now,
                        last_verified_at,
                    ),
                )
                row = cur.fetchone()
        return dict(row)

    def get_platform_credential_secret(self, provider_id: str) -> str | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT api_key_enc, status FROM platform_model_credentials WHERE provider_id = %s",
                    (provider_id,),
                )
                row = cur.fetchone()
        if not row or row["status"] != "verified":
            return None
        return _decrypt(row["api_key_enc"])

    def get_platform_credential_secret_any_status(self, provider_id: str) -> str | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT api_key_enc FROM platform_model_credentials WHERE provider_id = %s",
                    (provider_id,),
                )
                row = cur.fetchone()
        return _decrypt(row["api_key_enc"]) if row else None

    def set_platform_credential_status(
        self,
        provider_id: str,
        *,
        status: str,
        last_verified_at: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"verified", "invalid", "verification_unavailable", "revoked"}:
            raise ValueError("invalid credential status")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE platform_model_credentials SET status = %s, updated_at = %s, last_verified_at = COALESCE(%s, last_verified_at) WHERE provider_id = %s RETURNING provider_id, key_last_four, status, updated_at, last_verified_at",
                    (status, _utcnow_iso(), last_verified_at, provider_id),
                )
                row = cur.fetchone()
        if not row:
            raise ProviderNotFoundError("platform credential not found")
        return dict(row)

    def get_platform_credential_public(self, provider_id: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT provider_id, key_last_four, status, updated_at, last_verified_at FROM platform_model_credentials WHERE provider_id = %s",
                    (provider_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def delete_platform_credential(self, provider_id: str) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM platform_model_credentials WHERE provider_id = %s",
                    (provider_id,),
                )
                deleted = cur.rowcount > 0
        return deleted
