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
from typing import Any, Dict, List, Optional

import psycopg

from dashboard.backend.db_url import require_postgres_url
from dashboard.backend.users import (
    EMAIL_CHANGE_TTL_MINUTES,
    _utcnow,
    _utcnow_iso,
    hash_password,
    is_expired,
    public_user,
    verify_password,
)

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
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS email_change_requests (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        new_email TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        code_hash TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        used_at TEXT,
                        cancelled_at TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_email_change_requests_user_id
                    ON email_change_requests(user_id)
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

    def update_display_name(self, user_id: int, display_name: str) -> Dict[str, Any]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET display_name = %s WHERE id = %s RETURNING *",
                    (display_name.strip(), user_id),
                )
                row = cur.fetchone()
        if not row:
            raise ValueError("user_not_found")
        return public_user(row)

    def _email_change_expiry(self) -> str:
        return (
            (_utcnow() + timedelta(minutes=EMAIL_CHANGE_TTL_MINUTES))
            .replace(microsecond=0)
            .isoformat()
        )

    def create_email_change_request(
        self, user_id: int, new_email: str, code_hash: str
    ) -> Dict[str, Any]:
        """Supersede any in-flight request with a fresh stage-'old' one.

        Supersede, not DELETE -- see the SQLite twin: the log has to survive for
        the daily and 7-day limits to have anything to read.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE email_change_requests SET cancelled_at = %s
                    WHERE user_id = %s AND used_at IS NULL AND cancelled_at IS NULL
                    """,
                    (_utcnow_iso(), user_id),
                )
                cur.execute(
                    """
                    INSERT INTO email_change_requests
                        (user_id, new_email, stage, code_hash, created_at, expires_at)
                    VALUES (%s, %s, 'old', %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        user_id,
                        new_email.strip().lower(),
                        code_hash,
                        _utcnow_iso(),
                        self._email_change_expiry(),
                    ),
                )
                row = cur.fetchone()
        return dict(row)

    def get_active_email_change(self, user_id: int) -> Optional[Dict[str, Any]]:
        """The user's in-flight request, or None if absent, used, cancelled, or expired."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM email_change_requests
                    WHERE user_id = %s AND used_at IS NULL AND cancelled_at IS NULL
                    ORDER BY id DESC LIMIT 1
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
        if not row or is_expired(row["expires_at"]):
            return None
        return dict(row)

    def advance_email_change(self, request_id: int, code_hash: str) -> Dict[str, Any]:
        """Move a verified stage-'old' request to stage 'new' with a fresh code."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE email_change_requests
                    SET stage = 'new', code_hash = %s, attempts = 0, expires_at = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (code_hash, self._email_change_expiry(), request_id),
                )
                row = cur.fetchone()
        if not row:
            raise ValueError("email_change_request_not_found")
        return dict(row)

    def record_email_change_attempt(self, request_id: int) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE email_change_requests
                    SET attempts = attempts + 1
                    WHERE id = %s
                    RETURNING attempts
                    """,
                    (request_id,),
                )
                row = cur.fetchone()
        if not row:
            raise ValueError("email_change_request_not_found")
        return int(row["attempts"])

    def mark_email_change_used(self, request_id: int) -> None:
        """Retire a completed request without deleting it (see the SQLite twin)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE email_change_requests SET used_at = %s WHERE id = %s",
                    (_utcnow_iso(), request_id),
                )

    def cancel_email_change(self, user_id: int) -> None:
        """Deactivate without deleting, scoped to still-active rows.

        See the SQLite twin's cancel_email_change for both halves of the why.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE email_change_requests SET cancelled_at = %s
                    WHERE user_id = %s AND used_at IS NULL AND cancelled_at IS NULL
                    """,
                    (_utcnow_iso(), user_id),
                )

    def last_email_change_request_at(self, user_id: int) -> Optional[str]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT created_at FROM email_change_requests
                    WHERE user_id = %s ORDER BY id DESC LIMIT 1
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
        return str(row["created_at"]) if row else None

    def last_email_change_completed_at(self, user_id: int) -> Optional[str]:
        """When this user's email last actually changed (see the SQLite twin)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT used_at FROM email_change_requests
                    WHERE user_id = %s AND used_at IS NOT NULL
                    ORDER BY used_at DESC LIMIT 1
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
        return str(row["used_at"]) if row else None

    def email_change_request_times_since(self, user_id: int, since: str) -> List[str]:
        """created_at of every request at or after `since`, oldest first.

        created_at is TEXT here, exactly as in the SQLite twin, so `>=` is the
        same lexicographic comparison over the same fixed-width ISO-8601 form.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT created_at FROM email_change_requests
                    WHERE user_id = %s AND created_at >= %s
                    ORDER BY created_at ASC
                    """,
                    (user_id, since),
                )
                rows = cur.fetchall()
        return [str(row["created_at"]) for row in rows]

    def update_email(self, user_id: int, new_email: str) -> Dict[str, Any]:
        # .lower() is mandatory, not stylistic: this twin's users.email UNIQUE is
        # case-SENSITIVE (the SQLite twin's is COLLATE NOCASE), so skipping it
        # would let two casings of one address coexist in prod while being
        # rejected locally -- twin drift no SQLite-only test run can see.
        normalized_email = new_email.strip().lower()
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE users SET email = %s WHERE id = %s RETURNING *",
                        (normalized_email, user_id),
                    )
                    row = cur.fetchone()
        except psycopg.errors.UniqueViolation as exc:
            raise ValueError("email_already_registered") from exc
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
