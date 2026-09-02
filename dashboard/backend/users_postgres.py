"""
Postgres-backed UserStore implementation.

Selected instead of the default SQLite UserStore when USERS_DATABASE_URL is
set (see users.py's _build_user_store). Exists because the SQLite UserStore
shares DB_PATH with backtest data, and the deployed backend runs on a
disk-less Render free-tier host where that file resets on every deploy --
silently deleting every account (see CLAUDE.md gotchas).
"""

from datetime import timedelta
from typing import Any, Dict, List, Optional

import psycopg

from dashboard.backend.db_url import require_postgres_url
from dashboard.backend.session_tokens import (
    absolute_expiry,
    hash_session_token,
    idle_deadline,
    new_session_token,
    should_touch_last_seen,
)
from dashboard.backend.users import (
    ADMIN_ROLE_LOCK_KEY,
    CREDITS_REFUND_POSTGRES,
    CREDITS_SEED_POSTGRES,
    CREDITS_SPEND_POSTGRES,
    EMAIL_CHANGE_TTL_MINUTES,
    ENTITLEMENTS_UPSERT_POSTGRES,
    PASSWORD_RESET_MAX_ATTEMPTS,
    PASSWORD_RESET_TTL_MINUTES,
    USER_COUNTS_SQL,
    VALID_ROLES,
    _expiry_iso,
    _utcnow,
    _utcnow_iso,
    admin_user_rows_to_payloads,
    admin_user_search_pattern,
    credits_refund_params,
    credits_seed_applies,
    credits_seed_params,
    credits_spend_params,
    entitlements_upsert_params,
    format_stored_timestamp,
    hash_password,
    is_expired,
    parse_stored_timestamp,
    public_entitlements,
    public_user,
    public_user_with_entitlements,
    validate_entitlement_patch,
    verify_password_for_account,
)

# Mirrors users.AUTH_SESSIONS_DDL in Postgres dialect. Declared here rather than
# imported so test_store_twin_parity can read both column lists out of their own
# module's source; kept a plain (non-f) string for the same reason.
AUTH_SESSIONS_DDL = """
    CREATE TABLE IF NOT EXISTS auth_sessions (
        token_hash TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        revoked_at TEXT,
        user_agent TEXT,
        ip_prefix TEXT
    )
"""


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
                cur.execute(AUTH_SESSIONS_DDL)
                cur.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = 'auth_sessions'
                      AND column_name = 'token'
                    """
                )
                if cur.fetchone():
                    # Legacy plaintext-token table: cannot migrate without raw
                    # tokens. Drop and recreate; users must sign in again.
                    #
                    # current_schema() is not optional: information_schema.columns
                    # spans every schema the role can see, so an unqualified
                    # match can report a table this connection will never touch
                    # -- and the action taken on that report is a DROP.
                    cur.execute("DROP TABLE auth_sessions")
                    # Same DDL as the fresh install: the table is gone, so
                    # IF NOT EXISTS creates it. One literal, one schema.
                    cur.execute(AUTH_SESSIONS_DDL)
                    # print(), not logger: see users._build_user_store(). This
                    # drops every live login on a durable database.
                    print(
                        "auth_sessions: migrated from plaintext tokens to "
                        "hashed tokens; all existing sessions were dropped and "
                        "users must sign in again"
                    )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id
                    ON auth_sessions(user_id)
                    """
                )
                # Supports purge_expired_sessions(); without it the sweep is a
                # full scan of a table whose whole purpose here is to grow.
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires_at
                    ON auth_sessions(expires_at)
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
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS password_reset_requests (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
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
                    CREATE INDEX IF NOT EXISTS idx_password_reset_requests_user_id
                    ON password_reset_requests(user_id)
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_entitlements (
                        user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                        max_concurrent_backtests INTEGER NOT NULL DEFAULT 1,
                        credits INTEGER NOT NULL DEFAULT 0,
                        updated_at TEXT,
                        updated_by_admin_id INTEGER
                    )
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
        # One bcrypt compare on both branches -- see verify_password_for_account.
        user = self.get_user_by_email(email)
        if not verify_password_for_account(
            password, user["password_hash"] if user else None
        ):
            return None
        return user

    def create_session(
        self,
        user_id: int,
        *,
        user_agent: Optional[str] = None,
        ip_prefix: Optional[str] = None,
    ) -> str:
        raw_token = new_session_token()
        token_hash = hash_session_token(raw_token)
        now = _utcnow().replace(microsecond=0)
        created_at = format_stored_timestamp(now)
        expires_at = format_stored_timestamp(absolute_expiry(now))
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO auth_sessions (
                        token_hash, user_id, created_at, last_seen_at, expires_at,
                        user_agent, ip_prefix
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        token_hash,
                        user_id,
                        created_at,
                        created_at,
                        expires_at,
                        user_agent or None,
                        ip_prefix or None,
                    ),
                )
                cur.execute(
                    "DELETE FROM auth_sessions WHERE expires_at < %s",
                    (created_at,),
                )
        return raw_token

    def purge_expired_sessions(self) -> int:
        """Delete session rows past their absolute expiry. Returns the count.

        Twin of UserStore.purge_expired_sessions -- see there for why a soft
        revocation without a sweep leaves rows that nothing ever removes.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM auth_sessions WHERE expires_at < %s",
                    (format_stored_timestamp(_utcnow()),),
                )
                return max(cur.rowcount, 0)

    def get_user_for_token(self, token: str) -> Optional[Dict[str, Any]]:
        if not token or not str(token).strip():
            return None
        token_hash = hash_session_token(token.strip())
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        users.*,
                        auth_sessions.created_at AS session_created_at,
                        auth_sessions.last_seen_at AS session_last_seen_at,
                        auth_sessions.expires_at AS session_expires_at,
                        auth_sessions.revoked_at AS session_revoked_at,
                        user_entitlements.max_concurrent_backtests
                            AS ent_max_concurrent_backtests,
                        user_entitlements.credits AS ent_credits,
                        user_entitlements.updated_at AS ent_updated_at,
                        user_entitlements.updated_by_admin_id
                            AS ent_updated_by_admin_id
                    FROM auth_sessions
                    JOIN users ON users.id = auth_sessions.user_id
                    LEFT JOIN user_entitlements
                        ON user_entitlements.user_id = users.id
                    WHERE auth_sessions.token_hash = %s
                    """,
                    (token_hash,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                data = dict(row)
                revoked_at = data.pop("session_revoked_at")
                created_at = parse_stored_timestamp(data.pop("session_created_at"))
                last_seen_raw = data.pop("session_last_seen_at")
                last_seen_at = parse_stored_timestamp(
                    last_seen_raw or format_stored_timestamp(created_at)
                )
                expires_at = parse_stored_timestamp(data.pop("session_expires_at"))
                now = _utcnow()

                if revoked_at:
                    return None
                if expires_at < now:
                    cur.execute(
                        "DELETE FROM auth_sessions WHERE token_hash = %s",
                        (token_hash,),
                    )
                    return None
                if idle_deadline(last_seen_at) < now:
                    cur.execute(
                        "DELETE FROM auth_sessions WHERE token_hash = %s",
                        (token_hash,),
                    )
                    return None

                if should_touch_last_seen(last_seen_at, now):
                    cur.execute(
                        """
                        UPDATE auth_sessions
                        SET last_seen_at = %s
                        WHERE token_hash = %s
                        """,
                        (
                            format_stored_timestamp(now.replace(microsecond=0)),
                            token_hash,
                        ),
                    )
                return data

    def delete_session(self, token: str) -> None:
        if not token or not str(token).strip():
            return
        token_hash = hash_session_token(token.strip())
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE auth_sessions
                    SET revoked_at = %s
                    WHERE token_hash = %s AND revoked_at IS NULL
                    """,
                    (
                        format_stored_timestamp(_utcnow().replace(microsecond=0)),
                        token_hash,
                    ),
                )

    def update_password(self, user_id: int, new_password: str) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET password_hash = %s WHERE id = %s",
                    (hash_password(new_password), user_id),
                )

    def delete_other_sessions(self, user_id: int, keep_token: Optional[str]) -> None:
        """Revoke every session for the user except keep_token (None = all)."""
        now = format_stored_timestamp(_utcnow().replace(microsecond=0))
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if keep_token:
                    keep_hash = hash_session_token(keep_token.strip())
                    cur.execute(
                        """
                        UPDATE auth_sessions
                        SET revoked_at = %s
                        WHERE user_id = %s AND token_hash != %s AND revoked_at IS NULL
                        """,
                        (now, user_id, keep_hash),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE auth_sessions
                        SET revoked_at = %s
                        WHERE user_id = %s AND revoked_at IS NULL
                        """,
                        (now, user_id),
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
        return _expiry_iso(EMAIL_CHANGE_TTL_MINUTES)

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

    def _password_reset_expiry(self) -> str:
        return _expiry_iso(PASSWORD_RESET_TTL_MINUTES)

    def create_password_reset_request(
        self, user_id: int, code_hash: str
    ) -> Dict[str, Any]:
        """Supersede any in-flight reset request with a fresh one.

        Cancel + insert in one transaction (the connection context manager),
        for the same racing-creates reason as the SQLite twin.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE password_reset_requests SET cancelled_at = %s
                    WHERE user_id = %s AND used_at IS NULL AND cancelled_at IS NULL
                    """,
                    (_utcnow_iso(), user_id),
                )
                cur.execute(
                    """
                    INSERT INTO password_reset_requests
                        (user_id, code_hash, created_at, expires_at)
                    VALUES (%s, %s, %s, %s)
                    RETURNING *
                    """,
                    (user_id, code_hash, _utcnow_iso(), self._password_reset_expiry()),
                )
                row = cur.fetchone()
        return dict(row)

    def get_active_password_reset(self, user_id: int) -> Optional[Dict[str, Any]]:
        """The user's in-flight reset, or None if absent, used, cancelled, or expired."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM password_reset_requests
                    WHERE user_id = %s AND used_at IS NULL AND cancelled_at IS NULL
                    ORDER BY id DESC LIMIT 1
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
        if not row or is_expired(row["expires_at"]):
            return None
        return dict(row)

    def record_password_reset_attempt(self, request_id: int) -> int:
        """Count one wrong code; cancel at the cap (see the SQLite twin).

        Same single conditional UPDATE, so the 5-attempt guarantee holds under
        concurrency here too.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE password_reset_requests
                    SET attempts = attempts + 1,
                        cancelled_at = CASE
                            WHEN attempts + 1 >= %s THEN %s ELSE NULL
                        END
                    WHERE id = %s AND used_at IS NULL AND cancelled_at IS NULL
                        AND attempts < %s
                    RETURNING attempts
                    """,
                    (
                        PASSWORD_RESET_MAX_ATTEMPTS,
                        _utcnow_iso(),
                        request_id,
                        PASSWORD_RESET_MAX_ATTEMPTS,
                    ),
                )
                row = cur.fetchone()
        if not row:
            return PASSWORD_RESET_MAX_ATTEMPTS
        return int(row["attempts"])

    def mark_password_reset_used(self, request_id: int) -> bool:
        """Consume the code: the CAS only one concurrent redeem can win."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE password_reset_requests SET used_at = %s
                    WHERE id = %s AND used_at IS NULL AND cancelled_at IS NULL
                    """,
                    (_utcnow_iso(), request_id),
                )
                return cur.rowcount == 1

    def cancel_password_reset(self, user_id: int) -> None:
        """Deactivate without deleting, scoped to still-active rows."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE password_reset_requests SET cancelled_at = %s
                    WHERE user_id = %s AND used_at IS NULL AND cancelled_at IS NULL
                    """,
                    (_utcnow_iso(), user_id),
                )

    def last_password_reset_request_at(self, user_id: int) -> Optional[str]:
        """Status-blind cooldown read (see the SQLite twin)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT created_at FROM password_reset_requests
                    WHERE user_id = %s ORDER BY id DESC LIMIT 1
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
        return str(row["created_at"]) if row else None

    def password_reset_request_times_since(self, user_id: int, since: str) -> List[str]:
        """created_at of every request at or after `since`, oldest first."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT created_at FROM password_reset_requests
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

    def get_entitlements(self, user_id: int) -> Dict[str, Any]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM user_entitlements WHERE user_id = %s",
                    (int(user_id),),
                )
                row = cur.fetchone()
        return public_entitlements(row, int(user_id))

    def try_spend_credits(self, user_id: int, amount: int = 1) -> Optional[int]:
        """Debit ``amount`` credits; ``None`` means the balance was too low.

        Twin of the SQLite ``try_spend_credits`` — same contract, same three
        statements, one transaction. The seed selects from ``users``, so the
        enforced FK is never reached and a missing account simply seeds
        nothing and is refused.
        """
        amount = int(amount)
        if amount <= 0:
            raise ValueError("invalid_credit_amount")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if credits_seed_applies(amount):
                    cur.execute(CREDITS_SEED_POSTGRES, credits_seed_params(user_id))
                cur.execute(
                    CREDITS_SPEND_POSTGRES, credits_spend_params(user_id, amount)
                )
                if cur.rowcount != 1:
                    return None
                cur.execute(
                    "SELECT credits FROM user_entitlements WHERE user_id = %s",
                    (int(user_id),),
                )
                row = cur.fetchone()
        return int(dict(row)["credits"]) if row else None

    def refund_credits(self, user_id: int, amount: int = 1) -> Optional[int]:
        """Return ``amount`` credits, clamped to ``MAX_CREDITS_CAP``.

        ``None`` when the account has no entitlements row — nothing was ever
        debited from it, so there is nothing to give back.
        """
        amount = int(amount)
        if amount <= 0:
            raise ValueError("invalid_credit_amount")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    CREDITS_REFUND_POSTGRES, credits_refund_params(user_id, amount)
                )
                if cur.rowcount != 1:
                    return None
                cur.execute(
                    "SELECT credits FROM user_entitlements WHERE user_id = %s",
                    (int(user_id),),
                )
                row = cur.fetchone()
        return int(dict(row)["credits"]) if row else None

    def set_entitlements(
        self,
        user_id: int,
        *,
        max_concurrent_backtests: Optional[int] = None,
        credits: Optional[int] = None,
        updated_by_admin_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        if max_concurrent_backtests is None and credits is None:
            return self.get_entitlements(user_id)

        next_max, next_credits = validate_entitlement_patch(
            max_concurrent_backtests, credits
        )
        # Single statement on Neon: RETURNING replaces the read-back query, and
        # the enforced FK replaces the pre-check SELECT — atomically, where the
        # old check-then-write could pass for a user deleted in between. The
        # FK violation maps to the same user_not_found the SQLite twin raises,
        # instead of escaping as a 500 the console cannot explain.
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        ENTITLEMENTS_UPSERT_POSTGRES + "\nRETURNING *",
                        entitlements_upsert_params(
                            user_id, next_max, next_credits, updated_by_admin_id
                        ),
                    )
                    row = cur.fetchone()
        except psycopg.errors.ForeignKeyViolation as exc:
            raise ValueError("user_not_found") from exc
        return public_entitlements(dict(row) if row else None, int(user_id))

    def apply_admin_patch(
        self,
        user_id: int,
        *,
        role: Optional[str] = None,
        max_concurrent_backtests: Optional[int] = None,
        credits: Optional[int] = None,
        updated_by_admin_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Role and entitlements for one user, in a single transaction."""
        normalized_role: Optional[str] = None
        if role is not None:
            normalized_role = (role or "").strip().lower()
            if normalized_role not in VALID_ROLES:
                raise ValueError("invalid_role")
        next_max, next_credits = validate_entitlement_patch(
            max_concurrent_backtests, credits
        )
        touches_entitlements = next_max is not None or next_credits is not None

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(%s)", (ADMIN_ROLE_LOCK_KEY,)
                    )
                    cur.execute("SELECT * FROM users WHERE id = %s", (int(user_id),))
                    user_row = cur.fetchone()
                    if not user_row:
                        raise ValueError("user_not_found")
                    if normalized_role is not None:
                        current_role = dict(user_row)["role"]
                        if normalized_role != "admin" and current_role == "admin":
                            cur.execute(
                                "SELECT COUNT(*) AS n FROM users WHERE role = 'admin'"
                            )
                            if int(cur.fetchone()["n"] or 0) <= 1:
                                raise ValueError("last_admin")
                        cur.execute(
                            "UPDATE users SET role = %s WHERE id = %s RETURNING *",
                            (normalized_role, int(user_id)),
                        )
                        user_row = cur.fetchone()
                    if touches_entitlements:
                        cur.execute(
                            ENTITLEMENTS_UPSERT_POSTGRES + "\nRETURNING *",
                            entitlements_upsert_params(
                                user_id, next_max, next_credits, updated_by_admin_id
                            ),
                        )
                        ent_row = cur.fetchone()
                    else:
                        cur.execute(
                            "SELECT * FROM user_entitlements WHERE user_id = %s",
                            (int(user_id),),
                        )
                        ent_row = cur.fetchone()
        except psycopg.errors.ForeignKeyViolation as exc:
            # Same latent race set_entitlements maps: the enforced FK is the
            # real existence check, and its violation is "user not found", not
            # an unexplained 500. (SQLite never enforces this FK, so only this
            # twin can take the branch.)
            raise ValueError("user_not_found") from exc
        # Built from the rows this transaction read and wrote — not a pair of
        # fresh post-commit lookups that cost two more Neon round-trips and can
        # observe a concurrent admin's later write.
        return public_user_with_entitlements(
            user_row,
            public_entitlements(dict(ent_row) if ent_row else None, int(user_id)),
        )

    def promote_first_admin(self, user_id: int) -> Dict[str, Any]:
        """Promote ``user_id`` to admin iff no admin row exists yet."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (ADMIN_ROLE_LOCK_KEY,))
                cur.execute("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'")
                n = int(cur.fetchone()["n"] or 0)
                if n > 0:
                    raise ValueError("admin_exists")
                cur.execute(
                    "SELECT * FROM users WHERE id = %s",
                    (int(user_id),),
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError("user_not_found")
                cur.execute(
                    "UPDATE users SET role = 'admin' WHERE id = %s RETURNING *",
                    (int(user_id),),
                )
                row = cur.fetchone()
        if not row:
            raise ValueError("user_not_found")
        return public_user(row)

    def count_users_and_admins(self) -> Dict[str, int]:
        """Total accounts and how many are admins, in one query (see the twin)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(USER_COUNTS_SQL)
                row = cur.fetchone()
        data = dict(row) if row else {}
        return {
            "users": int(data.get("n_users") or 0),
            "admins": int(data.get("n_admins") or 0),
        }

    def count_users(self, query: str | None = None) -> int:
        pattern = admin_user_search_pattern(query)
        where_sql = ""
        params: tuple[str, ...] = ()
        if pattern is not None:
            where_sql = (
                " WHERE lower(email) LIKE lower(%s) ESCAPE '\\'"
                " OR lower(display_name) LIKE lower(%s) ESCAPE '\\'"
            )
            params = (pattern, pattern)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) AS n FROM users{where_sql}", params)
                row = cur.fetchone()
        return int(row["n"] if row else 0)

    def list_users_admin(
        self, *, limit: int = 100, offset: int = 0, query: str | None = None
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        pattern = admin_user_search_pattern(query)
        where_sql = ""
        params: tuple[object, ...] = (limit, offset)
        if pattern is not None:
            where_sql = (
                " WHERE lower(users.email) LIKE lower(%s) ESCAPE '\\'"
                " OR lower(users.display_name) LIKE lower(%s) ESCAPE '\\'"
            )
            params = (pattern, pattern, limit, offset)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT users.*,
                           user_entitlements.max_concurrent_backtests,
                           user_entitlements.credits,
                           user_entitlements.updated_at AS entitlements_updated_at,
                           user_entitlements.updated_by_admin_id
                    FROM users
                    LEFT JOIN user_entitlements ON user_entitlements.user_id = users.id
                    {where_sql}
                    ORDER BY users.id ASC
                    LIMIT %s OFFSET %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        return admin_user_rows_to_payloads(rows)

    def get_user_admin(self, user_id: int) -> Optional[Dict[str, Any]]:
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        return public_user_with_entitlements(user, self.get_entitlements(user_id))
