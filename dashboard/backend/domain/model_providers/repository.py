"""SQLite model provider registry and encrypted user credential vault."""

from __future__ import annotations

import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from dashboard.backend.database import DB_PATH
from dashboard.backend.domain.agents.repository import _utcnow_iso
from dashboard.backend.domain.brokers.repository import _decrypt, _encrypt

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
                api_key_enc TEXT NOT NULL,
                key_last_four TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'verification_unavailable',
                is_default INTEGER NOT NULL DEFAULT 0,
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
                operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_user_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                source TEXT NOT NULL,
                reason TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            """
        )
        now = _utcnow_iso()
        for item in SEEDED_PROVIDERS:
            conn.execute(
                """
                INSERT INTO provider_registry (
                    provider_id, display_name, adapter_type, approved_base_url,
                    capabilities_json, byok_enabled, platform_enabled, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, 0, 'enabled', ?, ?)
                ON CONFLICT(provider_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    adapter_type = excluded.adapter_type,
                    approved_base_url = excluded.approved_base_url,
                    capabilities_json = excluded.capabilities_json,
                    updated_at = excluded.updated_at
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
        conn.commit()
        conn.close()

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
                    idempotency_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING
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
                    key_last_four, status, is_default, created_at, updated_at,
                    last_verified_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(credential_id),
                    int(user_id),
                    provider_id,
                    label.strip(),
                    _encrypt(secret),
                    last_four[-4:],
                    status,
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
                "UPDATE user_model_credentials SET status = ?, is_default = CASE WHEN ? <> 'verified' THEN 0 ELSE is_default END, updated_at = ?, last_verified_at = COALESCE(?, last_verified_at), revoked_at = CASE WHEN ? = 'revoked' THEN ? ELSE revoked_at END WHERE credential_id = ?",
                (status, status, now, last_verified_at, status, now, str(credential_id)),
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
            (provider_id, _encrypt(secret), (key_last_four or secret[-4:])[-4:], status, now, last_verified_at),
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
        if not row or row["status"] != "verified":
            return None
        return _decrypt(row["api_key_enc"])

    def get_platform_credential_secret_any_status(self, provider_id: str) -> str | None:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT api_key_enc FROM platform_model_credentials WHERE provider_id = ?",
            (provider_id,),
        ).fetchone()
        conn.close()
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
        conn = self._get_connection()
        conn.execute(
            "UPDATE platform_model_credentials SET status = ?, updated_at = ?, last_verified_at = COALESCE(?, last_verified_at) WHERE provider_id = ?",
            (status, _utcnow_iso(), last_verified_at, provider_id),
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
        cur = conn.execute("DELETE FROM platform_model_credentials WHERE provider_id = ?", (provider_id,))
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
