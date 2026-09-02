"""PostgreSQL provider registry and encrypted user credential vault."""

from __future__ import annotations

import uuid
import json
import os
from typing import Any

import psycopg

from dashboard.backend.db_url import require_postgres_url
from dashboard.backend.domain.agents.repository import _utcnow_iso
from dashboard.backend.domain.brokers.repository import _decrypt, _encrypt

from .models import ProviderCapabilities, ProviderRecord
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
    api_key_enc TEXT,
    key_last_four TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'verification_unavailable',
    verification_message TEXT NOT NULL DEFAULT '',
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_verified_at TEXT,
    revoked_at TEXT,
    FOREIGN KEY(provider_id) REFERENCES provider_registry(provider_id) ON DELETE RESTRICT,
    CHECK (status IN ('verified', 'invalid', 'verification_unavailable', 'revoked')),
    CHECK ((status = 'revoked' AND api_key_enc IS NULL) OR (status <> 'revoked' AND api_key_enc IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS idx_user_model_credentials_owner
ON user_model_credentials(user_id, provider_id, updated_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_user_model_credentials_active_label
ON user_model_credentials(user_id, provider_id, label)
WHERE status <> 'revoked';

CREATE UNIQUE INDEX IF NOT EXISTS uq_user_model_credentials_default
ON user_model_credentials(user_id, provider_id)
WHERE is_default = TRUE AND status = 'verified';

CREATE TABLE IF NOT EXISTS platform_model_credentials (
    provider_id TEXT PRIMARY KEY,
    api_key_enc TEXT,
    key_last_four TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'verification_unavailable',
    updated_at TEXT NOT NULL,
    last_verified_at TEXT,
    FOREIGN KEY(provider_id) REFERENCES provider_registry(provider_id) ON DELETE RESTRICT,
    CHECK (status IN ('verified', 'invalid', 'verification_unavailable', 'revoked')),
    CHECK ((status = 'revoked' AND api_key_enc IS NULL) OR (status <> 'revoked' AND api_key_enc IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS model_provider_admin_operations (
    operation_id BIGSERIAL PRIMARY KEY,
    actor_user_id BIGINT NOT NULL,
    operation TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    source TEXT NOT NULL,
    reason TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    request_digest TEXT NOT NULL DEFAULT '',
    secret_fingerprint TEXT,
    result_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_provider_migrations (
    migration_id TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
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
    verification_message = str(data.get("verification_message") or "").strip()
    if not verification_message:
        verification_message = {
            "verified": "API key verified.",
            "invalid": "The provider rejected this API key.",
            "verification_unavailable": "Provider verification was unavailable.",
            "revoked": "Credential revoked.",
        }.get(data["status"], "")
    return {
        "credential_id": data["credential_id"],
        "provider_id": data["provider_id"],
        "label": data["label"],
        "key_last_four": data["key_last_four"],
        "status": data["status"],
        "verification_message": verification_message,
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
                cur.execute("ALTER TABLE user_model_credentials ALTER COLUMN api_key_enc DROP NOT NULL")
                cur.execute(
                    "ALTER TABLE user_model_credentials ADD COLUMN IF NOT EXISTS verification_message TEXT NOT NULL DEFAULT ''"
                )
                cur.execute("ALTER TABLE platform_model_credentials ALTER COLUMN api_key_enc DROP NOT NULL")
                cur.execute(
                    "ALTER TABLE model_provider_admin_operations ADD COLUMN IF NOT EXISTS request_digest TEXT NOT NULL DEFAULT ''"
                )
                cur.execute(
                    "ALTER TABLE model_provider_admin_operations ADD COLUMN IF NOT EXISTS secret_fingerprint TEXT"
                )
                cur.execute(
                    "ALTER TABLE model_provider_admin_operations ADD COLUMN IF NOT EXISTS result_json TEXT"
                )
                cur.execute("UPDATE user_model_credentials SET api_key_enc = NULL WHERE status = 'revoked'")
                cur.execute("UPDATE platform_model_credentials SET api_key_enc = NULL WHERE status = 'revoked'")
                cur.execute(
                    "ALTER TABLE user_model_credentials DROP CONSTRAINT IF EXISTS user_model_credentials_user_id_provider_id_label_key"
                )
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        ALTER TABLE user_model_credentials
                        ADD CONSTRAINT user_model_credentials_ciphertext_state
                        CHECK ((status = 'revoked' AND api_key_enc IS NULL) OR (status <> 'revoked' AND api_key_enc IS NOT NULL));
                    EXCEPTION WHEN duplicate_object THEN NULL;
                    END $$
                    """
                )
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        ALTER TABLE platform_model_credentials
                        ADD CONSTRAINT platform_model_credentials_ciphertext_state
                        CHECK ((status = 'revoked' AND api_key_enc IS NULL) OR (status <> 'revoked' AND api_key_enc IS NOT NULL));
                    EXCEPTION WHEN duplicate_object THEN NULL;
                    END $$
                    """
                )
                now = _utcnow_iso()
                for item in SEEDED_PROVIDERS:
                    cur.execute(
                        """
                        INSERT INTO provider_registry (
                            provider_id, display_name, adapter_type, approved_base_url,
                            capabilities_json, byok_enabled, platform_enabled,
                            status, created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'enabled', %s, %s)
                        ON CONFLICT (provider_id) DO NOTHING
                        """,
                        (
                            item["provider_id"],
                            item["display_name"],
                            item["adapter_type"],
                            item["approved_base_url"],
                            serialize_capabilities(item["capabilities"]),
                            bool(item.get("byok_enabled", True)),
                            bool(item.get("platform_enabled", False)),
                            now,
                            now,
                        ),
                    )
                self._migrate_legacy_openrouter_platform_flag(cur)

    @staticmethod
    def _migrate_legacy_openrouter_platform_flag(cur) -> None:
        """Enable the legacy Render-backed OpenRouter lane once, safely."""

        migration_id = "openrouter-platform-key-v1"
        cur.execute(
            "SELECT 1 FROM model_provider_migrations WHERE migration_id = %s",
            (migration_id,),
        )
        if cur.fetchone():
            return
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            return
        cur.execute(
            "SELECT status, platform_enabled FROM provider_registry WHERE provider_id = 'openrouter'"
        )
        provider = cur.fetchone()
        if not provider or provider["status"] != "enabled" or provider["platform_enabled"]:
            return
        cur.execute(
            "SELECT 1 FROM platform_model_credentials WHERE provider_id = 'openrouter'"
        )
        if cur.fetchone():
            return
        cur.execute(
            """
            SELECT operation, result_json
            FROM model_provider_admin_operations
            WHERE provider_id = 'openrouter' AND operation = 'upsert_provider'
            ORDER BY operation_id DESC
            LIMIT 1
            """
        )
        latest_operation = cur.fetchone()
        if latest_operation:
            try:
                snapshot = json.loads(latest_operation["result_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                snapshot = {}
            if snapshot.get("platform_enabled") is False:
                return
        now = _utcnow_iso()
        cur.execute(
            "UPDATE provider_registry SET platform_enabled = TRUE, updated_at = %s WHERE provider_id = 'openrouter'",
            (now,),
        )
        cur.execute(
            "INSERT INTO model_provider_migrations (migration_id, applied_at) VALUES (%s, %s)",
            (migration_id, now),
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
                        idempotency_key, request_digest, secret_fingerprint,
                        result_json, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (idempotency_key) DO NOTHING
                    """,
                    (
                        int(values["actor_user_id"]),
                        str(values["operation"]),
                        str(values["provider_id"]),
                        str(values["source"]),
                        str(values["reason"]),
                        str(values["idempotency_key"]),
                        str(values.get("request_digest", "")),
                        values.get("secret_fingerprint"),
                        values.get("result_json"),
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

    @staticmethod
    def _insert_admin_operation_in_transaction(
        cur, audit: dict[str, Any], result_json: str
    ) -> None:
        cur.execute(
            """
            INSERT INTO model_provider_admin_operations (
                actor_user_id, operation, provider_id, source, reason,
                idempotency_key, request_digest, secret_fingerprint,
                result_json, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING operation_id
            """,
            (
                int(audit["actor_user_id"]),
                str(audit["operation"]),
                str(audit["provider_id"]),
                str(audit["source"]),
                str(audit["reason"]),
                str(audit["idempotency_key"]),
                str(audit.get("request_digest", "")),
                audit.get("secret_fingerprint"),
                result_json,
                _utcnow_iso(),
            ),
        )
        if cur.fetchone() is None:
            raise CredentialConflictError("idempotency key already used")

    @staticmethod
    def _provider_snapshot(row: dict[str, Any]) -> str:
        return json.dumps(
            ProviderRecord.model_validate(_public_provider(row)).model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )

    def upsert_provider_with_audit(self, *, audit: dict[str, Any], **values: Any) -> dict[str, Any]:
        validate_adapter_type(values["adapter_type"])
        approved_base_url = validate_approved_origin(values["approved_base_url"])
        if values.get("status", "enabled") not in {"enabled", "disabled"}:
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
                        values["provider_id"],
                        str(values["display_name"]).strip(),
                        values["adapter_type"],
                        approved_base_url,
                        serialize_capabilities(values["capabilities"]),
                        bool(values["byok_enabled"]),
                        bool(values["platform_enabled"]),
                        values.get("status", "enabled"),
                        now,
                        now,
                    ),
                )
                row = cur.fetchone()
                self._insert_admin_operation_in_transaction(
                    cur, audit, self._provider_snapshot(row)
                )
        return _public_provider(row)

    def upsert_platform_credential_with_audit(
        self,
        *,
        provider_id: str,
        secret: str,
        status: str,
        last_verified_at: str | None,
        audit: dict[str, Any],
    ) -> dict[str, Any]:
        if status not in {"verified", "invalid", "verification_unavailable", "revoked"}:
            raise ValueError("invalid credential status")
        encrypted = None if status == "revoked" else _encrypt(secret)
        now = _utcnow_iso()
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT provider_id FROM provider_registry WHERE provider_id = %s",
                    (provider_id,),
                )
                if cur.fetchone() is None:
                    raise ProviderNotFoundError("provider not found")
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
                        secret[-4:],
                        status,
                        now,
                        last_verified_at,
                    ),
                )
                row = dict(cur.fetchone())
                self._insert_admin_operation_in_transaction(
                    cur,
                    audit,
                    json.dumps(row, sort_keys=True, separators=(",", ":")),
                )
        return row

    def set_platform_credential_status_with_audit(
        self,
        provider_id: str,
        *,
        status: str,
        last_verified_at: str | None,
        audit: dict[str, Any],
    ) -> dict[str, Any]:
        if status not in {"verified", "invalid", "verification_unavailable", "revoked"}:
            raise ValueError("invalid credential status")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE platform_model_credentials
                    SET api_key_enc = CASE WHEN %s = 'revoked' THEN NULL ELSE api_key_enc END,
                        status = %s,
                        updated_at = %s,
                        last_verified_at = COALESCE(%s, last_verified_at)
                    WHERE provider_id = %s
                    RETURNING provider_id, key_last_four, status, updated_at, last_verified_at
                    """,
                    (status, status, _utcnow_iso(), last_verified_at, provider_id),
                )
                row = cur.fetchone()
                if row is None:
                    raise ProviderNotFoundError("platform credential not found")
                public = dict(row)
                self._insert_admin_operation_in_transaction(
                    cur,
                    audit,
                    json.dumps(public, sort_keys=True, separators=(",", ":")),
                )
        return public

    def delete_platform_credential_with_audit(
        self, *, audit: dict[str, Any]
    ) -> bool:
        provider_id = str(audit["provider_id"])
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE platform_model_credentials
                    SET api_key_enc = NULL, status = 'revoked', updated_at = %s
                    WHERE provider_id = %s
                    RETURNING provider_id, key_last_four, status, updated_at, last_verified_at
                    """,
                    (_utcnow_iso(), provider_id),
                )
                row = cur.fetchone()
                if row is None:
                    raise ProviderNotFoundError("platform credential not found")
                self._insert_admin_operation_in_transaction(
                    cur,
                    audit,
                    json.dumps(dict(row), sort_keys=True, separators=(",", ":")),
                )
        return True

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
        verification_message: str | None = None,
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
                            key_last_four, status, verification_message, is_default,
                            created_at, updated_at, last_verified_at, revoked_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                            (verification_message or "").strip(),
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
        if row["status"] == "revoked" or not row["api_key_enc"]:
            raise CredentialConflictError("revoked credentials cannot be read")
        return _decrypt(row["api_key_enc"])

    def get_verified_default_user_credential(
        self, user_id: int, provider_id: str
    ) -> dict[str, Any] | None:
        """Return one verified default credential with a transient decrypted secret."""

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT credential_id, provider_id, key_last_four, status, api_key_enc
                    FROM user_model_credentials
                    WHERE user_id = %s
                      AND provider_id = %s
                      AND status = 'verified'
                      AND is_default = TRUE
                    """,
                    (int(user_id), str(provider_id)),
                )
                row = cur.fetchone()
        if not row or not row["api_key_enc"]:
            return None
        return {
            "credential_id": row["credential_id"],
            "provider_id": row["provider_id"],
            "key_last_four": row["key_last_four"],
            "status": row["status"],
            "secret": _decrypt(row["api_key_enc"]),
        }

    def set_user_credential_status(
        self,
        user_id: int,
        credential_id: str,
        *,
        status: str,
        last_verified_at: str | None = None,
        verification_message: str | None = None,
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
                    SET api_key_enc = CASE WHEN %s = 'revoked' THEN NULL ELSE api_key_enc END,
                        status = %s,
                        verification_message = COALESCE(%s, CASE WHEN %s = 'revoked' THEN 'Credential revoked.' ELSE verification_message END),
                        is_default = CASE WHEN %s <> 'verified' THEN FALSE ELSE is_default END,
                        updated_at = %s,
                        last_verified_at = COALESCE(%s, last_verified_at),
                        revoked_at = CASE WHEN %s = 'revoked' THEN %s ELSE revoked_at END
                    WHERE credential_id = %s
                    RETURNING *
                    """,
                    (status, status, verification_message, status, status, now, last_verified_at, status, now, str(credential_id)),
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
        encrypted = None if status == "revoked" else _encrypt(secret)
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
        if not row or row["status"] != "verified" or not row["api_key_enc"]:
            return None
        return _decrypt(row["api_key_enc"])

    def get_verified_platform_credential(
        self, provider_id: str
    ) -> dict[str, Any] | None:
        """Return the verified platform credential with a transient secret."""

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT provider_id, key_last_four, status, api_key_enc
                    FROM platform_model_credentials
                    WHERE provider_id = %s AND status = 'verified'
                    """,
                    (str(provider_id),),
                )
                row = cur.fetchone()
        if not row or not row["api_key_enc"]:
            return None
        return {
            "provider_id": row["provider_id"],
            "key_last_four": row["key_last_four"],
            "status": row["status"],
            "secret": _decrypt(row["api_key_enc"]),
        }

    def get_platform_credential_secret_any_status(self, provider_id: str) -> str | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT api_key_enc, status FROM platform_model_credentials WHERE provider_id = %s",
                    (provider_id,),
                )
                row = cur.fetchone()
        return _decrypt(row["api_key_enc"]) if row and row["status"] != "revoked" and row["api_key_enc"] else None

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
                    "UPDATE platform_model_credentials SET api_key_enc = CASE WHEN %s = 'revoked' THEN NULL ELSE api_key_enc END, status = %s, updated_at = %s, last_verified_at = COALESCE(%s, last_verified_at) WHERE provider_id = %s RETURNING provider_id, key_last_four, status, updated_at, last_verified_at",
                    (status, status, _utcnow_iso(), last_verified_at, provider_id),
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
                    "UPDATE platform_model_credentials SET api_key_enc = NULL, status = 'revoked', updated_at = %s WHERE provider_id = %s",
                    (_utcnow_iso(), provider_id),
                )
                deleted = cur.rowcount > 0
        return deleted
