"""
Postgres-backed UserStore implementation.

Selected instead of the default SQLite UserStore when USERS_DATABASE_URL is
set (see users.py's _build_user_store). Exists because the SQLite UserStore
shares DB_PATH with backtest data, and the deployed backend runs on a
disk-less Render free-tier host where that file resets on every deploy --
silently deleting every account (see CLAUDE.md gotchas).
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import psycopg

from dashboard.backend.db_url import require_postgres_url
from dashboard.backend.users import _utcnow, _utcnow_iso, hash_password, public_user, verify_password

SESSION_TTL_DAYS = 7


class PostgresUserStore:
    """Minimal user + auth session persistence, backed by Postgres."""

    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        self._init_schema()

    def _get_connection(self):
        # Pooled checkout: same context-manager transaction semantics as
        # psycopg.connect (commit on clean exit), returned to the pool on close.
        from dashboard.backend.db_pool import get_pool
        return get_pool(self.database_url).connection()

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        email TEXT NOT NULL UNIQUE,
                        display_name TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL DEFAULT 'user',
                        created_at TEXT NOT NULL,
                        discord_user_id TEXT,
                        avatar TEXT
                    )
                    """
                )
                # Lazy migration for existing deployments created before Discord linking.
                cur.execute(
                    """
                    ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS discord_user_id TEXT
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS avatar TEXT
                    """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_users_discord_user_id
                    ON users(discord_user_id)
                    WHERE discord_user_id IS NOT NULL
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS auth_sessions (
                        token TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id
                    ON auth_sessions(user_id)
                    """
                )

    def create_user(self, email: str, display_name: str, password: str) -> Dict[str, Any]:
        normalized_email = email.strip().lower()
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO users (email, display_name, password_hash, role, created_at)
                        VALUES (%s, %s, %s, 'user', %s)
                        RETURNING *
                        """,
                        (normalized_email, display_name.strip(), hash_password(password), _utcnow_iso()),
                    )
                    row = cur.fetchone()
        except psycopg.errors.UniqueViolation as exc:
            raise ValueError("email_already_registered") from exc
        return public_user(row)

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM users WHERE email = %s",
                    (email.strip().lower(),),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def authenticate(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        user = self.get_user_by_email(email)
        if not user:
            return None
        if not verify_password(password, user["password_hash"]):
            return None
        return user

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        now = _utcnow()
        created_at = now.replace(microsecond=0).isoformat()
        expires_at = (now + timedelta(days=SESSION_TTL_DAYS)).replace(microsecond=0).isoformat()
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO auth_sessions (token, user_id, created_at, expires_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (token, user_id, created_at, expires_at),
                )
        return token

    def get_user_for_token(self, token: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT users.*
                    FROM auth_sessions
                    JOIN users ON users.id = auth_sessions.user_id
                    WHERE auth_sessions.token = %s
                    """,
                    (token,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                cur.execute(
                    "SELECT expires_at FROM auth_sessions WHERE token = %s",
                    (token,),
                )
                session_row = cur.fetchone()

        if not session_row:
            return None

        expires_at = datetime.fromisoformat(session_row["expires_at"])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at < _utcnow():
            self.delete_session(token)
            return None

        return dict(row)

    def delete_session(self, token: str) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM auth_sessions WHERE token = %s", (token,))

    def update_password(self, user_id: int, new_password: str) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET password_hash = %s WHERE id = %s",
                    (hash_password(new_password), user_id),
                )

    def delete_other_sessions(self, user_id: int, keep_token: Optional[str]) -> None:
        """Revoke every session for the user except keep_token (None = all)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if keep_token:
                    cur.execute(
                        "DELETE FROM auth_sessions WHERE user_id = %s AND token != %s",
                        (user_id, keep_token),
                    )
                else:
                    cur.execute(
                        "DELETE FROM auth_sessions WHERE user_id = %s", (user_id,)
                    )

    def set_avatar(self, user_id: int, avatar: Optional[str]) -> Dict[str, Any]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET avatar = %s WHERE id = %s RETURNING *",
                    (avatar, user_id),
                )
                row = cur.fetchone()
        if not row:
            raise ValueError("user_not_found")
        return public_user(row)

    def get_user_by_discord_id(self, discord_user_id: str) -> Optional[Dict[str, Any]]:
        discord_id = str(discord_user_id).strip()
        if not discord_id:
            return None
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM users WHERE discord_user_id = %s",
                    (discord_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def link_discord_user(self, user_id: int, discord_user_id: str) -> Dict[str, Any]:
        """Attach a Discord snowflake to a website user.

        Raises ValueError('discord_already_linked') if another account owns it.
        """
        discord_id = str(discord_user_id).strip()
        if not discord_id:
            raise ValueError("invalid_discord_user_id")

        existing = self.get_user_by_discord_id(discord_id)
        if existing and int(existing["id"]) != int(user_id):
            raise ValueError("discord_already_linked")

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE users SET discord_user_id = %s WHERE id = %s RETURNING *",
                        (discord_id, user_id),
                    )
                    row = cur.fetchone()
        except psycopg.errors.UniqueViolation as exc:
            raise ValueError("discord_already_linked") from exc

        if not row:
            raise ValueError("user_not_found")
        return public_user(row)

    def unlink_discord_user(self, user_id: int) -> Dict[str, Any]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET discord_user_id = NULL WHERE id = %s RETURNING *",
                    (user_id,),
                )
                row = cur.fetchone()
        if not row:
            raise ValueError("user_not_found")
        return public_user(row)
