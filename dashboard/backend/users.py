"""
User accounts and auth session storage (SQLite, same database file as backtests).

Identity is ``users.id``, never ``users.email``. Email is a *mutable contact
attribute* -- it is the login handle and nothing else, and get_user_by_email()
exists only to resolve it back to an id at authenticate() time. Anything that
grants or withholds something (sessions, agents, portfolios, and any future
entitlement or billing record) must key on the id, or a user could shed or
inherit state by editing their address. ``email_change_requests`` is kept
append-only so that history stays auditable for exactly that reason.
"""

import base64
import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import bcrypt

from dashboard.backend.database import DB_PATH
from dashboard.backend.db_url import describe_database_url
from dashboard.backend.session_tokens import (
    absolute_expiry,
    hash_session_token,
    idle_deadline,
    new_session_token,
    should_touch_last_seen,
)

SESSION_TTL_DAYS = 7  # legacy name; runtime policy lives in session_tokens.py
BCRYPT_ROUNDS = 12
LEGACY_PBKDF2_ITERATIONS = 100_000
BCRYPT_MAX_BYTES = 72
EMAIL_CHANGE_TTL_MINUTES = 15
EMAIL_CHANGE_MAX_ATTEMPTS = 5
# Three windows, deliberately distinct -- see api/auth.py::request_email_change.
#   COOLDOWN   throttles one request against the next.
#   PER_DAY    bounds the shared Brevo quota one account can consume.
#   MIN_DAYS   is the product policy: email is a contact attribute, not a thing
#              you churn. It is keyed on a *completed* change, not a request, so
#              a mistyped address does not cost the user a week.
EMAIL_CHANGE_COOLDOWN_SECONDS = 60
EMAIL_CHANGE_MAX_REQUESTS_PER_DAY = 3
EMAIL_CHANGE_MIN_INTERVAL_DAYS = 7


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    # Delegates so the write format and the format callers build bounds in
    # (format_stored_timestamp, below) cannot drift apart.
    return format_stored_timestamp(_utcnow())


def parse_stored_timestamp(value: str) -> datetime:
    """Read a timestamp written by either twin.

    Both stores write _utcnow_iso() (offset-aware ISO-8601), but rows predating
    that convention -- or written by SQLite's CURRENT_TIMESTAMP default -- come
    back naive. Treat naive as UTC, which is what every writer here means.
    """
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def format_stored_timestamp(value: datetime) -> str:
    """Render a datetime the way every writer in this module stores one.

    The inverse of parse_stored_timestamp, and public for the same reason:
    callers that build a comparison bound (a rolling-window start, say) must
    produce the exact form the columns hold, because those comparisons run as
    string comparisons in SQL.
    """
    return value.replace(microsecond=0).isoformat()


def is_expired(expires_at: str) -> bool:
    return parse_stored_timestamp(expires_at) < _utcnow()


def _bcrypt_secret(password: str) -> bytes:
    """
    Return the bytes to feed bcrypt, without ever silently dropping any of them.

    bcrypt hashes at most the first 72 bytes and ignores the rest with no error, so
    two passwords sharing a 72-byte prefix verify against the same hash. NIST 800-63B
    5.1.1.2 forbids truncating a subscriber's secret, and password_policy.MAX_LENGTH
    accepts 128 characters, so anything past the cap is folded into a fixed-size
    digest first -- then every byte the user typed affects the hash.

    base64 of the digest, not the raw digest: raw SHA-256 output can contain NUL
    bytes, which C bcrypt implementations treat as end-of-string -- that would
    reintroduce truncation at the first NUL. The base64 form is 44 ASCII bytes,
    comfortably inside the cap.

    CodeQL flags the SHA-256 below as py/weak-sensitive-data-hashing. It is a false
    positive: this digest is never stored or compared as a credential, it is only a
    length-reduction step whose sole consumer is bcrypt, which supplies the salt and
    the work factor. The digest is also deliberately conditional -- passwords at or
    under the cap reach bcrypt untouched. That keeps the common path a single bcrypt
    call, and it keeps "password shucking" (cracking a leaked unsalted SHA-256 of
    the same secret, then confirming it with one bcrypt call) off the table for every
    password short enough to plausibly appear in such a corpus.
    """
    raw = password.encode("utf-8")
    if len(raw) <= BCRYPT_MAX_BYTES:
        return raw
    return base64.b64encode(hashlib.sha256(raw).digest())


def hash_password(password: str) -> str:
    hashed = bcrypt.hashpw(
        _bcrypt_secret(password),
        bcrypt.gensalt(rounds=BCRYPT_ROUNDS),
    )
    return hashed.decode("utf-8")


def _verify_legacy_pbkdf2(password: str, password_hash: str) -> bool:
    """Verify passwords hashed before the bcrypt migration."""
    try:
        salt, expected = password_hash.split("$", 1)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        LEGACY_PBKDF2_ITERATIONS,
    )
    return secrets.compare_digest(digest.hex(), expected)


def verify_password(password: str, password_hash: str) -> bool:
    if password_hash.startswith(("$2a$", "$2b$", "$2y$")):
        encoded = password_hash.encode("utf-8")
        try:
            if bcrypt.checkpw(_bcrypt_secret(password), encoded):
                return True
            # Accounts created before the pre-hash above stored bcrypt(raw), where
            # bcrypt itself dropped everything past byte 72. Only over-cap passwords
            # can hash differently under the two schemes, so this second, more
            # expensive check runs for those alone -- never on the common path.
            if len(password.encode("utf-8")) > BCRYPT_MAX_BYTES:
                return bcrypt.checkpw(password.encode("utf-8"), encoded)
            return False
        except ValueError:
            return False
    return _verify_legacy_pbkdf2(password, password_hash)


def public_user(row: sqlite3.Row | Dict[str, Any]) -> Dict[str, Any]:
    data = dict(row)
    discord_user_id = data.get("discord_user_id")
    return {
        "id": data["id"],
        "email": data["email"],
        "display_name": data["display_name"],
        "role": data["role"],
        "created_at": data["created_at"],
        "avatar": data.get("avatar"),
        "discord_linked": bool(discord_user_id),
        "discord_user_id": str(discord_user_id) if discord_user_id else None,
    }


class UserStore:
    """Minimal user + auth session persistence."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                last_seen_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                revoked_at TIMESTAMP,
                user_agent TEXT,
                ip_prefix TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        # Pre-hash plaintext-token schema: wipe and rebuild. Existing sessions
        # cannot be re-hashed without the raw token; users must sign in again.
        cursor.execute("PRAGMA table_info(auth_sessions)")
        session_columns = {row[1] for row in cursor.fetchall()}
        if session_columns and "token_hash" not in session_columns:
            cursor.execute("DROP TABLE auth_sessions")
            cursor.execute(
                """
                CREATE TABLE auth_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_seen_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    revoked_at TIMESTAMP,
                    user_agent TEXT,
                    ip_prefix TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id
            ON auth_sessions(user_id)
            """
        )
        # Lazy migration: Discord OAuth link column (nullable unique).
        cursor.execute("PRAGMA table_info(users)")
        columns = {row[1] for row in cursor.fetchall()}
        if "discord_user_id" not in columns:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN discord_user_id TEXT"
            )
        if "avatar" not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN avatar TEXT")
        cursor.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_discord_user_id
            ON users(discord_user_id)
            WHERE discord_user_id IS NOT NULL
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS email_change_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                new_email TEXT NOT NULL,
                stage TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used_at TIMESTAMP,
                cancelled_at TIMESTAMP,
                -- Declared but not enforced: SQLite disables FK checks per
                -- connection unless PRAGMA foreign_keys = ON is issued, and
                -- _get_connection() never issues it (turning it on would change
                -- deletion semantics for every table in this store, well beyond
                -- this task). Deleting a user therefore leaves this row orphaned
                -- rather than cascaded away -- tolerable because users.id is
                -- AUTOINCREMENT and ids are never reused, so an orphaned row can
                -- never be misattributed to a different user. The Postgres twin
                -- declares the same constraint and does enforce it.
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_email_change_requests_user_id
            ON email_change_requests(user_id)
            """
        )
        conn.commit()
        conn.close()

    def create_user(self, email: str, display_name: str, password: str) -> Dict[str, Any]:
        normalized_email = email.strip().lower()
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO users (email, display_name, password_hash, role)
                VALUES (?, ?, ?, 'user')
                """,
                (normalized_email, display_name.strip(), hash_password(password)),
            )
            conn.commit()
            user_id = cursor.lastrowid
        except sqlite3.IntegrityError as exc:
            conn.close()
            raise ValueError("email_already_registered") from exc

        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        return public_user(row)

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE",
            (email.strip().lower(),),
        )
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def authenticate(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        user = self.get_user_by_email(email)
        if not user:
            return None
        if not verify_password(password, user["password_hash"]):
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
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO auth_sessions (
                token_hash, user_id, created_at, last_seen_at, expires_at,
                user_agent, ip_prefix
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                token_hash,
                user_id,
                created_at,
                created_at,
                expires_at,
                (user_agent or None),
                (ip_prefix or None),
            ),
        )
        conn.commit()
        conn.close()
        return raw_token

    def get_user_for_token(self, token: str) -> Optional[Dict[str, Any]]:
        if not token or not str(token).strip():
            return None
        token_hash = hash_session_token(token.strip())
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                users.*,
                auth_sessions.created_at AS session_created_at,
                auth_sessions.last_seen_at AS session_last_seen_at,
                auth_sessions.expires_at AS session_expires_at,
                auth_sessions.revoked_at AS session_revoked_at
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token_hash = ?
            """,
            (token_hash,),
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        data = dict(row)
        revoked_at = data.pop("session_revoked_at", None)
        created_at = parse_stored_timestamp(data.pop("session_created_at"))
        last_seen_at = parse_stored_timestamp(
            data.pop("session_last_seen_at") or format_stored_timestamp(created_at)
        )
        expires_at = parse_stored_timestamp(data.pop("session_expires_at"))
        now = _utcnow()

        if revoked_at:
            conn.close()
            return None
        if expires_at < now:
            cursor.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,)
            )
            conn.commit()
            conn.close()
            return None
        if idle_deadline(last_seen_at) < now:
            cursor.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,)
            )
            conn.commit()
            conn.close()
            return None

        if should_touch_last_seen(last_seen_at, now):
            cursor.execute(
                """
                UPDATE auth_sessions
                SET last_seen_at = ?
                WHERE token_hash = ?
                """,
                (format_stored_timestamp(now.replace(microsecond=0)), token_hash),
            )
            conn.commit()
        conn.close()
        return data

    def delete_session(self, token: str) -> None:
        if not token or not str(token).strip():
            return
        token_hash = hash_session_token(token.strip())
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE auth_sessions
            SET revoked_at = ?
            WHERE token_hash = ? AND revoked_at IS NULL
            """,
            (format_stored_timestamp(_utcnow().replace(microsecond=0)), token_hash),
        )
        conn.commit()
        conn.close()

    def update_password(self, user_id: int, new_password: str) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(new_password), user_id),
        )
        conn.commit()
        conn.close()

    def delete_other_sessions(self, user_id: int, keep_token: Optional[str]) -> None:
        """Revoke every session for the user except keep_token (None = all)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = format_stored_timestamp(_utcnow().replace(microsecond=0))
        if keep_token:
            keep_hash = hash_session_token(keep_token.strip())
            cursor.execute(
                """
                UPDATE auth_sessions
                SET revoked_at = ?
                WHERE user_id = ? AND token_hash != ? AND revoked_at IS NULL
                """,
                (now, user_id, keep_hash),
            )
        else:
            cursor.execute(
                """
                UPDATE auth_sessions
                SET revoked_at = ?
                WHERE user_id = ? AND revoked_at IS NULL
                """,
                (now, user_id),
            )
        conn.commit()
        conn.close()

    def set_avatar(self, user_id: int, avatar: Optional[str]) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET avatar = ? WHERE id = ?", (avatar, user_id))
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise ValueError("user_not_found")
        return public_user(row)

    def update_display_name(self, user_id: int, display_name: str) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET display_name = ? WHERE id = ?",
            (display_name.strip(), user_id),
        )
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
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
        """Supersede any in-flight request for this user with a fresh stage-'old' one.

        Supersede, not DELETE: this table is an append-only log. Deleting would
        erase the used_at that EMAIL_CHANGE_MIN_INTERVAL_DAYS reads and the
        created_at rows EMAIL_CHANGE_MAX_REQUESTS_PER_DAY counts, so the very
        act of making another request would clear both limits.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE email_change_requests SET cancelled_at = ?
            WHERE user_id = ? AND used_at IS NULL AND cancelled_at IS NULL
            """,
            (_utcnow_iso(), user_id),
        )
        cursor.execute(
            """
            INSERT INTO email_change_requests
                (user_id, new_email, stage, code_hash, created_at, expires_at)
            VALUES (?, ?, 'old', ?, ?, ?)
            """,
            (
                user_id,
                new_email.strip().lower(),
                code_hash,
                _utcnow_iso(),
                self._email_change_expiry(),
            ),
        )
        conn.commit()
        request_id = cursor.lastrowid
        cursor.execute(
            "SELECT * FROM email_change_requests WHERE id = ?", (request_id,)
        )
        row = cursor.fetchone()
        conn.close()
        return dict(row)

    def get_active_email_change(self, user_id: int) -> Optional[Dict[str, Any]]:
        """The user's in-flight request, or None if absent, used, cancelled, or expired."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM email_change_requests
            WHERE user_id = ? AND used_at IS NULL AND cancelled_at IS NULL
            ORDER BY id DESC LIMIT 1
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row or is_expired(row["expires_at"]):
            return None
        return dict(row)

    def advance_email_change(self, request_id: int, code_hash: str) -> Dict[str, Any]:
        """Move a verified stage-'old' request to stage 'new' with a fresh code."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE email_change_requests
            SET stage = 'new', code_hash = ?, attempts = 0, expires_at = ?
            WHERE id = ?
            """,
            (code_hash, self._email_change_expiry(), request_id),
        )
        conn.commit()
        cursor.execute(
            "SELECT * FROM email_change_requests WHERE id = ?", (request_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise ValueError("email_change_request_not_found")
        return dict(row)

    def record_email_change_attempt(self, request_id: int) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE email_change_requests SET attempts = attempts + 1 WHERE id = ?",
            (request_id,),
        )
        conn.commit()
        cursor.execute(
            "SELECT attempts FROM email_change_requests WHERE id = ?", (request_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise ValueError("email_change_request_not_found")
        return int(row["attempts"])

    def mark_email_change_used(self, request_id: int) -> None:
        """Retire a completed request without deleting it.

        used_at makes get_active_email_change skip the row while
        last_email_change_request_at still sees it, so the cooldown applies to a
        change that just succeeded as well as one still in flight.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE email_change_requests SET used_at = ? WHERE id = ?",
            (_utcnow_iso(), request_id),
        )
        conn.commit()
        conn.close()

    def cancel_email_change(self, user_id: int) -> None:
        """Deactivate the user's request without deleting it.

        Mirrors mark_email_change_used: cancelled_at makes get_active_email_change
        skip the row while last_email_change_request_at still sees it. Deleting
        instead would let an authenticated caller who knows the password loop
        request/cancel/request with the 60-second cooldown never enforced --
        mail-bombing the account and burning the shared Brevo quota.

        Scoped to rows that are still active. Stamping cancelled_at over an
        already-used row would claim a change that actually completed had been
        cancelled, which is wrong in a log that is now kept for audit.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE email_change_requests SET cancelled_at = ?
            WHERE user_id = ? AND used_at IS NULL AND cancelled_at IS NULL
            """,
            (_utcnow_iso(), user_id),
        )
        conn.commit()
        conn.close()

    def last_email_change_request_at(self, user_id: int) -> Optional[str]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT created_at FROM email_change_requests
            WHERE user_id = ? ORDER BY id DESC LIMIT 1
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()
        return str(row["created_at"]) if row else None

    def last_email_change_completed_at(self, user_id: int) -> Optional[str]:
        """When this user's email last actually changed, or None if it never has.

        Ordered by used_at rather than id: a request created earlier can be
        completed later, so row order is not completion order.

        Set by mark_email_change_used in a separate transaction from the
        update_email that precedes it. A crash between the two leaves the email
        changed with the clock unstarted -- accepted, because this is a churn
        policy rather than a security boundary, and the 24-hour and 60-second
        limits both still apply. Making it atomic would mean a column on
        ``users``, i.e. an ALTER on the live accounts table, for a window of a
        few milliseconds.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT used_at FROM email_change_requests
            WHERE user_id = ? AND used_at IS NOT NULL
            ORDER BY used_at DESC LIMIT 1
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()
        return str(row["used_at"]) if row else None

    def email_change_request_times_since(self, user_id: int, since: str) -> List[str]:
        """created_at of every request made at or after `since`, oldest first.

        Returns the timestamps rather than a bare count so the caller can say
        *when* the rolling window frees up -- the answer is the oldest entry
        plus the window, which a COUNT(*) cannot supply.

        String comparison, not date arithmetic: both twins write
        _utcnow_iso(), a fixed-width offset-aware ISO-8601 form that sorts
        lexicographically. parse_stored_timestamp tolerates naive legacy rows on
        read, but none can exist here -- this table has only ever been written
        by the code above it.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT created_at FROM email_change_requests
            WHERE user_id = ? AND created_at >= ?
            ORDER BY created_at ASC
            """,
            (user_id, since),
        )
        rows = cursor.fetchall()
        conn.close()
        return [str(row["created_at"]) for row in rows]

    def update_email(self, user_id: int, new_email: str) -> Dict[str, Any]:
        normalized_email = new_email.strip().lower()
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE users SET email = ? WHERE id = ?",
                (normalized_email, user_id),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.close()
            raise ValueError("email_already_registered") from exc
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise ValueError("user_not_found")
        return public_user(row)

    def get_user_by_discord_id(self, discord_user_id: str) -> Optional[Dict[str, Any]]:
        discord_id = str(discord_user_id).strip()
        if not discord_id:
            return None
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE discord_user_id = ?",
            (discord_id,),
        )
        row = cursor.fetchone()
        conn.close()
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

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE users SET discord_user_id = ? WHERE id = ?",
                (discord_id, user_id),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.close()
            raise ValueError("discord_already_linked") from exc

        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise ValueError("user_not_found")
        return public_user(row)

    def unlink_discord_user(self, user_id: int) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET discord_user_id = NULL WHERE id = ?",
            (user_id,),
        )
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise ValueError("user_not_found")
        return public_user(row)


def _build_user_store():
    # USERS_DATABASE_URL only, deliberately: CONTENT_DATABASE_URL is scoped to
    # agents/versions/strategies and must not select the account database
    # (spec, Decision 2). Do not "simplify" this into a fallback chain.
    database_url = os.getenv("USERS_DATABASE_URL")
    if database_url:
        from dashboard.backend.users_postgres import PostgresUserStore

        # print(), not logger.info(): dashboard.backend.* loggers sit at WARNING
        # in every real deployment (nothing here configures logging; uvicorn's
        # LOGGING_CONFIG has no 'root' key), so an info() line would be invisible
        # exactly where it matters. Name the target too -- "postgres" alone reads
        # the same whether this is the intended Neon DB or a typo'd/staging URL.
        print(f"user_store backend: postgres ({describe_database_url(database_url)})")
        return PostgresUserStore(database_url)
    print("user_store backend: sqlite (ephemeral on Render)")
    return UserStore()


user_store = _build_user_store()
