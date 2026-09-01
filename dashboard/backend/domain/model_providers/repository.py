"""SQLite model provider registry and encrypted user credential vault."""

from __future__ import annotations

import os
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from dashboard.backend.database import DB_PATH
from dashboard.backend.domain.agents.repository import _utcnow_iso
from dashboard.backend.domain.brokers.repository import _decrypt, _encrypt, _get_fernet

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
from .models import ProviderRecord


def ensure_credential_encryption_ready() -> None:
    """Fail before verification when encrypted credential storage is unavailable."""

    _get_fernet()


class ModelProviderStore:
    """Persist provider metadata and encrypted credentials in one local DB."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.Error:
            pass
        return conn

    def _init_schema(self) -> None:
        conn = self._get_connection()
        self._migrate_legacy_credential_schema(conn)
        self._migrate_legacy_platform_schema(conn)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS provider_registry (
                provider_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                adapter_type TEXT NOT NULL,
                approved_base_url TEXT NOT NULL,
                capabilities_json TEXT NOT NULL,
                byok_enabled INTEGER NOT NULL DEFAULT 1,
                platform_enabled INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'enabled',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (adapter_type IN ('openrouter', 'openai', 'anthropic', 'gemini', 'openai_compatible')),
                CHECK (status IN ('enabled', 'disabled'))
            );
            CREATE TABLE IF NOT EXISTS user_model_credentials (
                credential_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                provider_id TEXT NOT NULL,
                label TEXT NOT NULL,
                api_key_enc TEXT,
                key_last_four TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'verification_unavailable',
                verification_message TEXT NOT NULL DEFAULT '',
                is_default INTEGER NOT NULL DEFAULT 0,
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
                WHERE is_default = 1 AND status = 'verified';
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
                operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_user_id INTEGER NOT NULL,
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
        )
        self._ensure_user_credential_columns(conn)
        self._ensure_admin_operation_columns(conn)
        now = _utcnow_iso()
        for item in SEEDED_PROVIDERS:
            conn.execute(
                """
                INSERT INTO provider_registry (
                    provider_id, display_name, adapter_type, approved_base_url,
                    capabilities_json, byok_enabled, platform_enabled, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'enabled', ?, ?)
                ON CONFLICT(provider_id) DO NOTHING
                """,
                (
                    item["provider_id"],
                    item["display_name"],
                    item["adapter_type"],
                    item["approved_base_url"],
                    serialize_capabilities(item["capabilities"]),
                    int(bool(item.get("byok_enabled", True))),
                    int(bool(item.get("platform_enabled", False))),
                    now,
                    now,
                ),
            )
        self._migrate_legacy_openrouter_platform_flag(conn)
        conn.commit()
        conn.close()

    @staticmethod
    def _migrate_legacy_openrouter_platform_flag(conn: sqlite3.Connection) -> None:
        """Enable the legacy Render-backed OpenRouter lane once, safely."""

        migration_id = "openrouter-platform-key-v1"
        if conn.execute(
            "SELECT 1 FROM model_provider_migrations WHERE migration_id = ?",
            (migration_id,),
        ).fetchone():
            return
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            return
        provider = conn.execute(
            "SELECT status, platform_enabled FROM provider_registry WHERE provider_id = 'openrouter'"
        ).fetchone()
        if not provider or provider["status"] != "enabled" or provider["platform_enabled"]:
            return
        if conn.execute(
            "SELECT 1 FROM platform_model_credentials WHERE provider_id = 'openrouter'"
        ).fetchone():
            return
        latest_operation = conn.execute(
            """
            SELECT operation, result_json
            FROM model_provider_admin_operations
            WHERE provider_id = 'openrouter' AND operation = 'upsert_provider'
            ORDER BY operation_id DESC
            LIMIT 1
            """
        ).fetchone()
        if latest_operation:
            try:
                snapshot = json.loads(latest_operation["result_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                snapshot = {}
            if snapshot.get("platform_enabled") is False:
                return
        conn.execute(
            "UPDATE provider_registry SET platform_enabled = 1, updated_at = ? WHERE provider_id = 'openrouter'",
            (_utcnow_iso(),),
        )
        conn.execute(
            "INSERT INTO model_provider_migrations (migration_id, applied_at) VALUES (?, ?)",
            (migration_id, _utcnow_iso()),
        )

    @staticmethod
    def _ensure_user_credential_columns(conn: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(user_model_credentials)"
            ).fetchall()
        }
        if "verification_message" not in columns:
            conn.execute(
                "ALTER TABLE user_model_credentials ADD COLUMN verification_message TEXT NOT NULL DEFAULT ''"
            )

    @staticmethod
    def _ensure_admin_operation_columns(conn: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(model_provider_admin_operations)"
            ).fetchall()
        }
        if "request_digest" not in columns:
            conn.execute(
                "ALTER TABLE model_provider_admin_operations ADD COLUMN request_digest TEXT NOT NULL DEFAULT ''"
            )
        if "secret_fingerprint" not in columns:
            conn.execute(
                "ALTER TABLE model_provider_admin_operations ADD COLUMN secret_fingerprint TEXT"
            )
        if "result_json" not in columns:
            conn.execute(
                "ALTER TABLE model_provider_admin_operations ADD COLUMN result_json TEXT"
            )

    @staticmethod
    def _migrate_legacy_platform_schema(conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'platform_model_credentials'"
        ).fetchone()
        if not row:
            return
        columns = {
            item[1]: {"notnull": bool(item[3])}
            for item in conn.execute("PRAGMA table_info(platform_model_credentials)").fetchall()
        }
        if not columns.get("api_key_enc", {}).get("notnull"):
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("ALTER TABLE platform_model_credentials RENAME TO platform_model_credentials_legacy")
            conn.execute(
                """
                CREATE TABLE platform_model_credentials (
                    provider_id TEXT PRIMARY KEY,
                    api_key_enc TEXT,
                    key_last_four TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'verification_unavailable',
                    updated_at TEXT NOT NULL,
                    last_verified_at TEXT,
                    FOREIGN KEY(provider_id) REFERENCES provider_registry(provider_id) ON DELETE RESTRICT,
                    CHECK (status IN ('verified', 'invalid', 'verification_unavailable', 'revoked')),
                    CHECK ((status = 'revoked' AND api_key_enc IS NULL) OR (status <> 'revoked' AND api_key_enc IS NOT NULL))
                )
                """
            )
            conn.execute(
                """
                INSERT INTO platform_model_credentials (
                    provider_id, api_key_enc, key_last_four, status, updated_at, last_verified_at
                )
                SELECT provider_id,
                       CASE WHEN status = 'revoked' THEN NULL ELSE api_key_enc END,
                       key_last_four, status, updated_at, last_verified_at
                FROM platform_model_credentials_legacy
                """
            )
            conn.execute("DROP TABLE platform_model_credentials_legacy")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _migrate_legacy_credential_schema(conn: sqlite3.Connection) -> None:
        """Rebuild the pre-tombstone SQLite table without losing active rows."""
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'user_model_credentials'"
        ).fetchone()
        if not row:
            return
        columns = {
            item[1]: {"notnull": bool(item[3])}
            for item in conn.execute("PRAGMA table_info(user_model_credentials)").fetchall()
        }
        legacy_sql = (row[0] or "").upper()
        needs_rebuild = bool(
            columns.get("api_key_enc", {}).get("notnull")
            or "UNIQUE(USER_ID, PROVIDER_ID, LABEL)" in legacy_sql
        )
        if not needs_rebuild:
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("ALTER TABLE user_model_credentials RENAME TO user_model_credentials_legacy")
            conn.execute(
                """
                CREATE TABLE user_model_credentials (
                    credential_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    provider_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    api_key_enc TEXT,
                    key_last_four TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'verification_unavailable',
                    verification_message TEXT NOT NULL DEFAULT '',
                    is_default INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_verified_at TEXT,
                    revoked_at TEXT,
                    FOREIGN KEY(provider_id) REFERENCES provider_registry(provider_id) ON DELETE RESTRICT,
                    CHECK (status IN ('verified', 'invalid', 'verification_unavailable', 'revoked')),
                    CHECK ((status = 'revoked' AND api_key_enc IS NULL) OR (status <> 'revoked' AND api_key_enc IS NOT NULL))
                )
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX idx_user_model_credentials_active_label ON user_model_credentials(user_id, provider_id, label) WHERE status <> 'revoked'"
            )
            conn.execute(
                "CREATE UNIQUE INDEX uq_user_model_credentials_default ON user_model_credentials(user_id, provider_id) WHERE is_default = 1 AND status = 'verified'"
            )
            conn.execute(
                "CREATE INDEX idx_user_model_credentials_owner ON user_model_credentials(user_id, provider_id, updated_at DESC)"
            )
            conn.execute(
                """
                INSERT INTO user_model_credentials (
                    credential_id, user_id, provider_id, label, api_key_enc,
                    key_last_four, status, verification_message, is_default, created_at, updated_at,
                    last_verified_at, revoked_at
                )
                SELECT credential_id, user_id, provider_id, label,
                       CASE WHEN status = 'revoked' THEN NULL ELSE api_key_enc END,
                       key_last_four, status, '', is_default, created_at, updated_at,
                       last_verified_at, revoked_at
                FROM user_model_credentials_legacy
                """
            )
            conn.execute("DROP TABLE user_model_credentials_legacy")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _public_provider(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
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

    @staticmethod
    def _public_credential(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
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

    def list_enabled_providers(self, *, mode: str = "byok") -> list[dict[str, Any]]:
        column = "byok_enabled" if mode == "byok" else "platform_enabled"
        if mode not in {"byok", "platform"}:
            raise ValueError("unsupported provider mode")
        conn = self._get_connection()
        rows = conn.execute(
            f"SELECT * FROM provider_registry WHERE status = 'enabled' AND {column} = 1 ORDER BY display_name"
        ).fetchall()
        conn.close()
        return [self._public_provider(row) for row in rows]

    def list_all_providers(self) -> list[dict[str, Any]]:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM provider_registry ORDER BY display_name"
        ).fetchall()
        conn.close()
        return [self._public_provider(row) for row in rows]

    def record_admin_operation(self, **values: Any) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO model_provider_admin_operations (
                    actor_user_id, operation, provider_id, source, reason,
                    idempotency_key, request_digest, secret_fingerprint,
                    result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING
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
            conn.commit()
        finally:
            conn.close()

    def get_admin_operation(self, idempotency_key: str) -> dict[str, Any] | None:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM model_provider_admin_operations WHERE idempotency_key = ?",
            (str(idempotency_key),),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def _insert_admin_operation_in_transaction(
        conn: sqlite3.Connection,
        audit: dict[str, Any],
        result_json: str,
    ) -> None:
        cursor = conn.execute(
            """
            INSERT INTO model_provider_admin_operations (
                actor_user_id, operation, provider_id, source, reason,
                idempotency_key, request_digest, secret_fingerprint,
                result_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(idempotency_key) DO NOTHING
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
        if cursor.rowcount == 0:
            raise CredentialConflictError("idempotency key already used")

    @staticmethod
    def _provider_snapshot(row: sqlite3.Row) -> str:
        return json.dumps(
            ProviderRecord.model_validate(ModelProviderStore._public_provider(row)).model_dump(
                mode="json"
            ),
            sort_keys=True,
            separators=(",", ":"),
        )

    def upsert_provider_with_audit(self, *, audit: dict[str, Any], **values: Any) -> dict[str, Any]:
        validate_adapter_type(values["adapter_type"])
        approved_base_url = validate_approved_origin(values["approved_base_url"])
        if values.get("status", "enabled") not in {"enabled", "disabled"}:
            raise ValueError("invalid provider status")
        now = _utcnow_iso()
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO provider_registry (
                    provider_id, display_name, adapter_type, approved_base_url,
                    capabilities_json, byok_enabled, platform_enabled, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    adapter_type = excluded.adapter_type,
                    approved_base_url = excluded.approved_base_url,
                    capabilities_json = excluded.capabilities_json,
                    byok_enabled = excluded.byok_enabled,
                    platform_enabled = excluded.platform_enabled,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    values["provider_id"],
                    str(values["display_name"]).strip(),
                    values["adapter_type"],
                    approved_base_url,
                    serialize_capabilities(values["capabilities"]),
                    int(values["byok_enabled"]),
                    int(values["platform_enabled"]),
                    values.get("status", "enabled"),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM provider_registry WHERE provider_id = ?",
                (values["provider_id"],),
            ).fetchone()
            self._insert_admin_operation_in_transaction(
                conn, audit, self._provider_snapshot(row)
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return self._public_provider(row)

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
        encrypted_secret = None if status == "revoked" else _encrypt(secret)
        now = _utcnow_iso()
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            provider = conn.execute(
                "SELECT provider_id FROM provider_registry WHERE provider_id = ?",
                (provider_id,),
            ).fetchone()
            if not provider:
                raise ProviderNotFoundError("provider not found")
            conn.execute(
                """
                INSERT INTO platform_model_credentials (
                    provider_id, api_key_enc, key_last_four, status, updated_at, last_verified_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider_id) DO UPDATE SET
                    api_key_enc = excluded.api_key_enc,
                    key_last_four = excluded.key_last_four,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    last_verified_at = excluded.last_verified_at
                """,
                (provider_id, encrypted_secret, secret[-4:], status, now, last_verified_at),
            )
            row = conn.execute(
                "SELECT provider_id, key_last_four, status, updated_at, last_verified_at FROM platform_model_credentials WHERE provider_id = ?",
                (provider_id,),
            ).fetchone()
            public = dict(row)
            self._insert_admin_operation_in_transaction(
                conn,
                audit,
                json.dumps(public, sort_keys=True, separators=(",", ":")),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return public

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
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE platform_model_credentials SET api_key_enc = CASE WHEN ? = 'revoked' THEN NULL ELSE api_key_enc END, status = ?, updated_at = ?, last_verified_at = COALESCE(?, last_verified_at) WHERE provider_id = ?",
                (status, status, _utcnow_iso(), last_verified_at, provider_id),
            )
            row = conn.execute(
                "SELECT provider_id, key_last_four, status, updated_at, last_verified_at FROM platform_model_credentials WHERE provider_id = ?",
                (provider_id,),
            ).fetchone()
            if not row:
                raise ProviderNotFoundError("platform credential not found")
            public = dict(row)
            self._insert_admin_operation_in_transaction(
                conn,
                audit,
                json.dumps(public, sort_keys=True, separators=(",", ":")),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return public

    def delete_platform_credential_with_audit(
        self, *, audit: dict[str, Any]
    ) -> bool:
        provider_id = str(audit["provider_id"])
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE platform_model_credentials SET api_key_enc = NULL, status = 'revoked', updated_at = ? WHERE provider_id = ?",
                (_utcnow_iso(), provider_id),
            )
            if cur.rowcount == 0:
                raise ProviderNotFoundError("platform credential not found")
            row = conn.execute(
                "SELECT provider_id, key_last_four, status, updated_at, last_verified_at FROM platform_model_credentials WHERE provider_id = ?",
                (provider_id,),
            ).fetchone()
            self._insert_admin_operation_in_transaction(
                conn,
                audit,
                json.dumps(dict(row), sort_keys=True, separators=(",", ":")),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return True

    def get_provider(self, provider_id: str) -> dict[str, Any] | None:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM provider_registry WHERE provider_id = ?", (provider_id,)
        ).fetchone()
        conn.close()
        return self._public_provider(row) if row else None

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
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO provider_registry (
                provider_id, display_name, adapter_type, approved_base_url,
                capabilities_json, byok_enabled, platform_enabled, status,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider_id) DO UPDATE SET
                display_name = excluded.display_name,
                adapter_type = excluded.adapter_type,
                approved_base_url = excluded.approved_base_url,
                capabilities_json = excluded.capabilities_json,
                byok_enabled = excluded.byok_enabled,
                platform_enabled = excluded.platform_enabled,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                provider_id,
                display_name.strip(),
                adapter_type,
                approved_base_url,
                serialize_capabilities(capabilities),
                int(byok_enabled),
                int(platform_enabled),
                status,
                now,
                now,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM provider_registry WHERE provider_id = ?", (provider_id,)
        ).fetchone()
        conn.close()
        return self._public_provider(row)

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
        last_four = key_last_four or secret[-4:]
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if set_default and status == "verified":
                conn.execute(
                    "UPDATE user_model_credentials SET is_default = 0 WHERE user_id = ? AND provider_id = ?",
                    (int(user_id), provider_id),
                )
            conn.execute(
                """
                INSERT INTO user_model_credentials (
                    credential_id, user_id, provider_id, label, api_key_enc,
                    key_last_four, status, verification_message, is_default,
                    created_at, updated_at, last_verified_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(credential_id),
                    int(user_id),
                    provider_id,
                    label.strip(),
                    _encrypt(secret),
                    last_four[-4:],
                    status,
                    (verification_message or "").strip(),
                    int(bool(set_default and status == "verified")),
                    now,
                    now,
                    last_verified_at,
                    now if status == "revoked" else None,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM user_model_credentials WHERE credential_id = ?",
                (str(credential_id),),
            ).fetchone()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise CredentialConflictError("credential label already exists") from exc
        finally:
            conn.close()
        return self._public_credential(row)

    def list_user_credentials(self, user_id: int, provider_id: str | None = None) -> list[dict[str, Any]]:
        conn = self._get_connection()
        if provider_id:
            rows = conn.execute(
                "SELECT * FROM user_model_credentials WHERE user_id = ? AND provider_id = ? AND status <> 'revoked' ORDER BY provider_id, is_default DESC, label",
                (int(user_id), provider_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM user_model_credentials WHERE user_id = ? AND status <> 'revoked' ORDER BY provider_id, is_default DESC, label",
                (int(user_id),),
            ).fetchall()
        conn.close()
        return [self._public_credential(row) for row in rows]

    def get_user_credential_public(self, user_id: int, credential_id: str) -> dict[str, Any] | None:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM user_model_credentials WHERE credential_id = ? AND user_id = ?",
            (str(credential_id), int(user_id)),
        ).fetchone()
        conn.close()
        return self._public_credential(row) if row else None

    def get_user_credential_secret(self, user_id: int, credential_id: str) -> str:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM user_model_credentials WHERE credential_id = ?",
            (str(credential_id),),
        ).fetchone()
        conn.close()
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

        conn = self._get_connection()
        row = conn.execute(
            """
            SELECT credential_id, provider_id, key_last_four, status, api_key_enc
            FROM user_model_credentials
            WHERE user_id = ?
              AND provider_id = ?
              AND status = 'verified'
              AND is_default = 1
            """,
            (int(user_id), str(provider_id)),
        ).fetchone()
        conn.close()
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
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM user_model_credentials WHERE credential_id = ?",
                (str(credential_id),),
            ).fetchone()
            if not row:
                raise CredentialNotFoundError("credential not found")
            if int(row["user_id"]) != int(user_id):
                raise CredentialOwnershipError("credential does not belong to this user")
            now = _utcnow_iso()
            conn.execute(
                "UPDATE user_model_credentials SET api_key_enc = CASE WHEN ? = 'revoked' THEN NULL ELSE api_key_enc END, status = ?, verification_message = COALESCE(?, CASE WHEN ? = 'revoked' THEN 'Credential revoked.' ELSE verification_message END), is_default = CASE WHEN ? <> 'verified' THEN 0 ELSE is_default END, updated_at = ?, last_verified_at = COALESCE(?, last_verified_at), revoked_at = CASE WHEN ? = 'revoked' THEN ? ELSE revoked_at END WHERE credential_id = ?",
                (status, status, verification_message, status, status, now, last_verified_at, status, now, str(credential_id)),
            )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM user_model_credentials WHERE credential_id = ?", (str(credential_id),)
            ).fetchone()
        finally:
            conn.close()
        return self._public_credential(result)

    def set_default_user_credential(self, user_id: int, credential_id: str) -> dict[str, Any]:
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM user_model_credentials WHERE credential_id = ?", (str(credential_id),)
            ).fetchone()
            if not row:
                raise CredentialNotFoundError("credential not found")
            if int(row["user_id"]) != int(user_id):
                raise CredentialOwnershipError("credential does not belong to this user")
            if row["status"] != "verified":
                raise CredentialConflictError("only verified credentials can be default")
            conn.execute(
                "UPDATE user_model_credentials SET is_default = 0 WHERE user_id = ? AND provider_id = ?",
                (int(user_id), row["provider_id"]),
            )
            conn.execute(
                "UPDATE user_model_credentials SET is_default = 1, updated_at = ? WHERE credential_id = ?",
                (_utcnow_iso(), str(credential_id)),
            )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM user_model_credentials WHERE credential_id = ?", (str(credential_id),)
            ).fetchone()
        finally:
            conn.close()
        return self._public_credential(result)

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
        provider = self.get_provider(provider_id)
        if not provider:
            raise ProviderNotFoundError("provider not found")
        if status not in {"verified", "invalid", "verification_unavailable", "revoked"}:
            raise ValueError("invalid credential status")
        now = _utcnow_iso()
        conn = self._get_connection()
        encrypted_secret = None if status == "revoked" else _encrypt(secret)
        conn.execute(
            """
            INSERT INTO platform_model_credentials (
                provider_id, api_key_enc, key_last_four, status, updated_at, last_verified_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider_id) DO UPDATE SET
                api_key_enc = excluded.api_key_enc,
                key_last_four = excluded.key_last_four,
                status = excluded.status,
                updated_at = excluded.updated_at,
                last_verified_at = excluded.last_verified_at
            """,
            (provider_id, encrypted_secret, (key_last_four or secret[-4:])[-4:], status, now, last_verified_at),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM platform_model_credentials WHERE provider_id = ?", (provider_id,)
        ).fetchone()
        conn.close()
        return {
            "provider_id": row["provider_id"],
            "key_last_four": row["key_last_four"],
            "status": row["status"],
            "updated_at": row["updated_at"],
            "last_verified_at": row["last_verified_at"],
        }

    def get_platform_credential_secret(self, provider_id: str) -> str | None:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT api_key_enc, status FROM platform_model_credentials WHERE provider_id = ?",
            (provider_id,),
        ).fetchone()
        conn.close()
        if not row or row["status"] != "verified" or not row["api_key_enc"]:
            return None
        return _decrypt(row["api_key_enc"])

    def get_verified_platform_credential(
        self, provider_id: str
    ) -> dict[str, Any] | None:
        """Return the verified platform credential with a transient secret."""

        conn = self._get_connection()
        row = conn.execute(
            """
            SELECT provider_id, key_last_four, status, api_key_enc
            FROM platform_model_credentials
            WHERE provider_id = ? AND status = 'verified'
            """,
            (str(provider_id),),
        ).fetchone()
        conn.close()
        if not row or not row["api_key_enc"]:
            return None
        return {
            "provider_id": row["provider_id"],
            "key_last_four": row["key_last_four"],
            "status": row["status"],
            "secret": _decrypt(row["api_key_enc"]),
        }

    def get_platform_credential_secret_any_status(self, provider_id: str) -> str | None:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT api_key_enc, status FROM platform_model_credentials WHERE provider_id = ?",
            (provider_id,),
        ).fetchone()
        conn.close()
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
        conn = self._get_connection()
        conn.execute(
            "UPDATE platform_model_credentials SET api_key_enc = CASE WHEN ? = 'revoked' THEN NULL ELSE api_key_enc END, status = ?, updated_at = ?, last_verified_at = COALESCE(?, last_verified_at) WHERE provider_id = ?",
            (status, status, _utcnow_iso(), last_verified_at, provider_id),
        )
        conn.commit()
        row = conn.execute(
            "SELECT provider_id, key_last_four, status, updated_at, last_verified_at FROM platform_model_credentials WHERE provider_id = ?",
            (provider_id,),
        ).fetchone()
        conn.close()
        if not row:
            raise ProviderNotFoundError("platform credential not found")
        return dict(row)

    def get_platform_credential_public(self, provider_id: str) -> dict[str, Any] | None:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT provider_id, key_last_four, status, updated_at, last_verified_at FROM platform_model_credentials WHERE provider_id = ?",
            (provider_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def delete_platform_credential(self, provider_id: str) -> bool:
        conn = self._get_connection()
        cur = conn.execute(
            "UPDATE platform_model_credentials SET api_key_enc = NULL, status = 'revoked', updated_at = ? WHERE provider_id = ?",
            (_utcnow_iso(), provider_id),
        )
        conn.commit()
        conn.close()
        return cur.rowcount > 0


def _build_model_provider_store() -> ModelProviderStore:
    # The Postgres twin is selected when the accounts database is configured;
    # SQLite remains the local and test default.
    database_url = (os.getenv("USERS_DATABASE_URL") or "").strip()
    if database_url:
        from .repository_postgres import PostgresModelProviderStore

        return PostgresModelProviderStore(database_url)
    return ModelProviderStore()


model_provider_store = _build_model_provider_store()
