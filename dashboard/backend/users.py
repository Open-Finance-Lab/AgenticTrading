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
    require_session_hash_secret,
    should_touch_last_seen,
)

# One literal, executed both on a fresh install and after the legacy-schema
# DROP, so the two paths cannot produce different tables. Kept as a plain
# (non-f) string: test_store_twin_parity reads column names out of the source
# text, and an interpolation would collapse to a placeholder there.
AUTH_SESSIONS_DDL = """
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
# Password-reset flow (#187). Same code machinery as email-change, but the
# requester is anonymous, so the cooldown and daily cap are the durable
# backstop behind api/auth.py's in-process limiters (which reset on redeploy).
PASSWORD_RESET_TTL_MINUTES = 15
PASSWORD_RESET_MAX_ATTEMPTS = 5
PASSWORD_RESET_COOLDOWN_SECONDS = 300
PASSWORD_RESET_MAX_REQUESTS_PER_DAY = 5


def _expiry_iso(minutes: int) -> str:
    """Shared expiry stamp for the emailed-code request writers (email change
    and password reset); microseconds dropped so the stored form stays
    fixed-width for the string-comparison window reads."""
    return (_utcnow() + timedelta(minutes=minutes)).replace(microsecond=0).isoformat()


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


def _best_effort_write(conn, cursor, sql: str, params: tuple) -> bool:
    """Run a housekeeping write that must never fail its caller.

    get_user_for_token runs on every authenticated request and was read-only
    until sessions were hashed; it now touches last_seen_at and reaps rows it
    finds dead. None of that is what the caller asked for, so losing it to a
    lost race for the write lock is free -- while letting sqlite3's
    OperationalError escape turns a valid session into a 500, and an expired
    one into a 500 instead of a 401.
    """
    try:
        cursor.execute(sql, params)
        conn.commit()
        return True
    except sqlite3.OperationalError:
        conn.rollback()
        return False


_DUMMY_PASSWORD_HASH: Optional[str] = None


def _dummy_password_hash() -> str:
    """A bcrypt hash of a value nobody can present, built on first use.

    Lazily, because generating it costs a full bcrypt round (~190 ms at
    ``BCRYPT_ROUNDS=12``) and every CLI script and test collection imports this
    module. The cost of that is one skewed sample: the first unknown-email
    login after a restart pays the hash *and* the compare. An attacker would
    have to already be probing at that exact moment to see it.
    """
    global _DUMMY_PASSWORD_HASH
    if _DUMMY_PASSWORD_HASH is None:
        _DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))
    return _DUMMY_PASSWORD_HASH


def verify_password_for_account(password: str, password_hash: Optional[str]) -> bool:
    """``verify_password`` that costs the same whether or not the account exists.

    Returning early when the email matches no row makes a miss ~3000x faster
    than a wrong password (measured: 0.06 ms vs 182 ms), so response *time*
    answers "is this address registered?" no matter how carefully the response
    *body* is made uniform. Hashing against a throwaway hash on that path puts
    both branches through the same single bcrypt compare.

    Callers must still discard the result for a missing account -- this only
    equalises the clock, it never authenticates one.
    """
    if password_hash is None:
        verify_password(password, _dummy_password_hash())
        return False
    return verify_password(password, password_hash)


VALID_ROLES = frozenset({"user", "admin"})
# Hard caps for admin PATCH — keep a runaway slider from scheduling dozens of
# LLM subprocesses on one Render box.
MAX_CONCURRENT_BACKTESTS_CAP = 20
MAX_CREDITS_CAP = 1_000_000
# Postgres advisory-lock key for serializing admin-role mutations (bootstrap
# one-shot + last-admin demotion). SQLite uses BEGIN IMMEDIATE instead.
ADMIN_ROLE_LOCK_KEY = 0xA71AD01
# Built-in fallback for the default quota below. Mirrors
# ``domain.runs.service.MAX_ACTIVE_RUNS_PER_AGENT``; not imported from there
# because domain/runs imports this module's store, and users.py must stay
# import-light for callers that never create runs.
_BUILTIN_DEFAULT_CONCURRENT_BACKTESTS = 5
# Built-in fallback for the default credit grant below. See
# ``_default_entitlement`` for why neither default is 0.
_BUILTIN_DEFAULT_CREDITS = 100


def _default_entitlement(env_var: str, builtin: int, cap: int) -> int:
    """Read an entitlement default from ``env_var``, bounded by ``0..cap``.

    Shared by both defaults because they answer the same question with the
    same failure modes. Out-of-range or unparseable values fall back to the
    built-in rather than raising at import time — a typo in a Render var must
    not take the process down — and say so on the operator's only real channel
    (print, not logging: logger output is invisible under deployed uvicorn).
    """
    raw = os.getenv(env_var)
    if raw is None or not raw.strip():
        return builtin
    try:
        value = int(raw.strip())
    except ValueError:
        print(f"users: {env_var} is not an integer; using the built-in default")
        return builtin
    if value < 0 or value > cap:
        print(
            f"users: {env_var} is out of range (0..{cap}); "
            "using the built-in default"
        )
        return builtin
    return value


# Slots an account gets before an admin has ever touched its row.
#
# Deliberately **not** 1. Nothing seeds ``user_entitlements`` at signup and
# nothing backfills it on deploy, so this constant is the live limit for every
# account that exists today — the day this ships, it silently becomes
# everyone's quota. At 1 that is a regression rather than a control: before the
# entitlement plane, one account's concurrency was bounded only by
# ``MAX_ACTIVE_RUNS_PER_AGENT`` (5) per agent it owned, so every multi-agent
# user would have woken up throttled by an admin console none of them can open.
# Matching that per-agent number keeps a single-agent account exactly where it
# was and leaves the *control* intact: an admin lowers a specific account, they
# do not have to raise everyone first.
#
# Env-overridable like every other run-capacity knob
# (``MAX_ACTIVE_RUNS_PER_AGENT``/``MAX_ACTIVE_RUNS_GLOBAL``), so an operator can
# tighten the site-wide floor without a deploy.
DEFAULT_MAX_CONCURRENT_BACKTESTS = _default_entitlement(
    "DEFAULT_MAX_CONCURRENT_BACKTESTS",
    _BUILTIN_DEFAULT_CONCURRENT_BACKTESTS,
    MAX_CONCURRENT_BACKTESTS_CAP,
)

# LLM backtests an account may run before an admin has ever touched its row.
#
# Inert unless ``CREDITS_METERING_ENABLED`` is armed (see
# ``domain/entitlements/credits.py``) — until then nothing reads a balance and
# this is only what ``/auth/me`` reports. It matters at the moment metering is
# switched on, which is exactly why it is **not** 0: nothing backfills
# ``user_entitlements``, so a zero default would turn one env var into a
# site-wide lockout, discovered through support tickets rather than a metric.
# 100 is a real bound (an abusive signup cannot burn unbounded operator LLM
# budget) that no ordinary user reaches — the per-client limiter already caps
# ``/backtest/run`` at 10/hour, so 100 credits is at least ten hours of
# continuous, deliberate use. An operator who wants stricter lowers
# ``DEFAULT_CREDITS``; one who wants a specific account stopped sets that row
# to 0, which reads as "suspended" exactly as it does for the quota above.
DEFAULT_CREDITS = _default_entitlement(
    "DEFAULT_CREDITS", _BUILTIN_DEFAULT_CREDITS, MAX_CREDITS_CAP
)


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


def default_entitlements(user_id: int) -> Dict[str, Any]:
    return {
        "user_id": int(user_id),
        "max_concurrent_backtests": DEFAULT_MAX_CONCURRENT_BACKTESTS,
        "credits": DEFAULT_CREDITS,
        "updated_at": None,
        "updated_by_admin_id": None,
    }


def public_entitlements(row: sqlite3.Row | Dict[str, Any] | None, user_id: int) -> Dict[str, Any]:
    # NULL columns mean "no entitlements row" (a LEFT JOIN miss), not zero: fall
    # back to the defaults per field rather than crashing on ``int(None)``.
    if not row:
        return default_entitlements(user_id)
    data = dict(row)
    max_concurrent = data.get("max_concurrent_backtests")
    credits = data.get("credits")
    return {
        "user_id": int(data.get("user_id") or user_id),
        "max_concurrent_backtests": (
            int(max_concurrent)
            if max_concurrent is not None
            else DEFAULT_MAX_CONCURRENT_BACKTESTS
        ),
        "credits": int(credits) if credits is not None else DEFAULT_CREDITS,
        "updated_at": data.get("updated_at"),
        "updated_by_admin_id": (
            int(data["updated_by_admin_id"])
            if data.get("updated_by_admin_id") is not None
            else None
        ),
    }


# Column aliases the session join adds so ``/api/auth/me`` can report
# entitlements without a second round-trip. Prefixed to avoid colliding with
# ``users.*`` (both tables have an ``updated_at``-shaped column).
SESSION_ENTITLEMENT_PREFIX = "ent_"


def entitlements_from_session_row(
    row: sqlite3.Row | Dict[str, Any] | None, user_id: int
) -> Optional[Dict[str, Any]]:
    """Public entitlements built from a ``get_user_for_token`` row.

    Returns ``None`` when the row carries no ``ent_*`` columns at all — that
    means the caller's store did not join ``user_entitlements`` and must query
    it separately. A column that is *present but NULL* is different: it is a
    LEFT JOIN miss, i.e. a user with no entitlements row yet, so defaults apply.
    """
    data = dict(row or {})
    if f"{SESSION_ENTITLEMENT_PREFIX}max_concurrent_backtests" not in data:
        return None
    return public_entitlements(
        {
            "user_id": user_id,
            "max_concurrent_backtests": data.get("ent_max_concurrent_backtests"),
            "credits": data.get("ent_credits"),
            "updated_at": data.get("ent_updated_at"),
            "updated_by_admin_id": data.get("ent_updated_by_admin_id"),
        },
        user_id,
    )


def public_user_with_entitlements(
    row: sqlite3.Row | Dict[str, Any],
    entitlements: Dict[str, Any],
) -> Dict[str, Any]:
    """Admin projection: the public user plus entitlements, minus ``avatar``.

    Avatars are data: URIs bounded at 200_000 chars each on write
    (``api/auth.py``). The admin console renders emails, names, roles and two
    numbers — never the image — so carrying them would make a 100-row page tens
    of megabytes of response body on a free-tier box for nothing. Callers that
    merge this into a stored user (``saveAdminUserRole``) spread it over the
    existing object, so an absent key leaves the cached avatar intact.
    """
    payload = public_user(row)
    payload.pop("avatar", None)
    payload["entitlements"] = entitlements
    return payload


def validate_entitlement_patch(
    max_concurrent_backtests: Optional[int], credits: Optional[int]
) -> tuple[Optional[int], Optional[int]]:
    """Range-check the provided fields; ``None`` stays ``None`` (unchanged).

    Each field is bounded independently, so validating only what was supplied
    is equivalent to validating the merged row — and it means the store never
    has to read the current values to decide whether a patch is legal.
    """
    next_max: Optional[int] = None
    if max_concurrent_backtests is not None:
        next_max = int(max_concurrent_backtests)
        # Floor is 0, not 1. A floor equal to the default meant the console
        # could meter an account but never stop one: an admin watching an
        # abusive signup burn LLM budget could only lower it to the value a
        # fresh account already has. 0 reads as "suspended" and costs nothing
        # to implement — check_owner_active_run_cap refuses at active >= limit,
        # so a zero budget refuses the first run.
        if next_max < 0 or next_max > MAX_CONCURRENT_BACKTESTS_CAP:
            raise ValueError("invalid_max_concurrent_backtests")
    next_credits: Optional[int] = None
    if credits is not None:
        next_credits = int(credits)
        if next_credits < 0 or next_credits > MAX_CREDITS_CAP:
            raise ValueError("invalid_credits")
    return next_max, next_credits


# Upsert that leaves an omitted field alone instead of rewriting it with the
# value this request happened to read a moment ago. The old read-modify-write
# spanned two connections, so two admins patching different fields at once lost
# one of the two edits. ``COALESCE(<param>, <existing column>)`` pushes the
# merge into the statement, where the row is locked. Note the DO UPDATE clause
# repeats the raw parameters rather than reading ``excluded.*``: the VALUES row
# has already had defaults substituted, so ``excluded`` cannot tell "not
# supplied" from "supplied as the default".
_ENTITLEMENTS_UPSERT_TEMPLATE = """
INSERT INTO user_entitlements (
    user_id, max_concurrent_backtests, credits, updated_at, updated_by_admin_id
)
VALUES ({p}, COALESCE({max_cast}, {p}), COALESCE({credits_cast}, {p}), {p}, {p})
ON CONFLICT (user_id) DO UPDATE SET
    max_concurrent_backtests = COALESCE(
        {max_cast}, user_entitlements.max_concurrent_backtests
    ),
    credits = COALESCE({credits_cast}, user_entitlements.credits),
    updated_at = EXCLUDED.updated_at,
    updated_by_admin_id = EXCLUDED.updated_by_admin_id
"""

ENTITLEMENTS_UPSERT_SQLITE = _ENTITLEMENTS_UPSERT_TEMPLATE.format(
    p="?", max_cast="?", credits_cast="?"
)
# psycopg sends a bare ``None`` as an untyped NULL (OID 0), which Postgres
# refuses to resolve inside COALESCE ("could not determine data type"). The
# explicit ``::integer`` casts are load-bearing, and no SQLite test can catch
# their absence.
ENTITLEMENTS_UPSERT_POSTGRES = _ENTITLEMENTS_UPSERT_TEMPLATE.format(
    p="%s", max_cast="%s::integer", credits_cast="%s::integer"
)


def entitlements_upsert_params(
    user_id: int,
    next_max: Optional[int],
    next_credits: Optional[int],
    updated_by_admin_id: Optional[int],
) -> tuple:
    """Positional parameters for ``ENTITLEMENTS_UPSERT_*`` (same order in both)."""
    return (
        int(user_id),
        next_max,
        DEFAULT_MAX_CONCURRENT_BACKTESTS,
        next_credits,
        DEFAULT_CREDITS,
        format_stored_timestamp(_utcnow().replace(microsecond=0)),
        int(updated_by_admin_id) if updated_by_admin_id is not None else None,
        next_max,
        next_credits,
    )


# --- Credit metering -------------------------------------------------------
#
# Three statements rather than one upsert, because a spend is not an edit:
#
# 1. SEED materialises the defaults for an account that has never been touched.
#    ``DO NOTHING`` on conflict, so it can never overwrite a balance — and it
#    is skipped entirely when the default cannot cover the spend, so a refusal
#    never freezes today's ``DEFAULT_CREDITS`` into a row that a later, more
#    generous value would then fail to reach.
# 2. SPEND is the whole guard: a single conditional UPDATE whose ``credits >=``
#    predicate and decrement evaluate under the row lock the UPDATE already
#    takes. ``rowcount`` distinguishes spent from refused. A read-then-write
#    would need an explicit transaction to say the same thing, and would still
#    be wrong on the branch where no row exists to lock.
# 3. REFUND is unconditional but clamped, so a double refund cannot mint a
#    balance past the cap an admin PATCH is held to.
#
# None of the three touches ``updated_at`` / ``updated_by_admin_id``: those
# columns are admin-edit provenance, and stamping them on a metered spend would
# make the console report every backtest as an entitlement change.
# ``SELECT ... FROM users WHERE id`` rather than ``VALUES``: it inserts nothing
# for an account that does not exist, which is what keeps the two twins
# behaving alike. Postgres enforces the FK and would raise; SQLite does not
# enforce it here (see the DDL note on email_change_requests) and would happily
# strand a ghost entitlements row. Sourcing the row from ``users`` makes both
# dialects agree without an exception handler on one side only.
# ``{i}`` is an integer-typed parameter. Every one of them carries an explicit
# ``::integer`` on the Postgres side, for the same reason the entitlements
# upsert does: psycopg sends a parameter with no type OID and Postgres resolves
# parameter types during parse analysis, where several of these sit in a
# position that supplies no context to resolve from. The seed is the sharp
# case -- ``INSERT ... SELECT $1`` analyses the SELECT *without* the insert
# target's column types, so an uncast parameter is a hard 42P08 ("could not
# determine data type of parameter") on every call. LEAST() is the other:
# it is polymorphic, so its second argument has nothing to infer from either.
# No SQLite test can catch any of this -- the same statements run clean there.
_CREDITS_SEED_TEMPLATE = """
INSERT INTO user_entitlements (
    user_id, max_concurrent_backtests, credits, updated_at, updated_by_admin_id
)
SELECT {i}, {i}, {i}, NULL, NULL FROM users WHERE id = {i}
ON CONFLICT (user_id) DO NOTHING
"""

_CREDITS_SPEND_TEMPLATE = """
UPDATE user_entitlements
   SET credits = credits - {i}
 WHERE user_id = {i} AND credits >= {i}
"""

# SQLite spells the two-argument minimum MIN(); Postgres spells it LEAST() and
# reserves MIN() for the aggregate.
_CREDITS_REFUND_TEMPLATE = """
UPDATE user_entitlements
   SET credits = {least}(credits + {i}, {i})
 WHERE user_id = {i}
"""

CREDITS_SEED_SQLITE = _CREDITS_SEED_TEMPLATE.format(i="?")
CREDITS_SEED_POSTGRES = _CREDITS_SEED_TEMPLATE.format(i="%s::integer")
CREDITS_SPEND_SQLITE = _CREDITS_SPEND_TEMPLATE.format(i="?")
CREDITS_SPEND_POSTGRES = _CREDITS_SPEND_TEMPLATE.format(i="%s::integer")
CREDITS_REFUND_SQLITE = _CREDITS_REFUND_TEMPLATE.format(i="?", least="MIN")
CREDITS_REFUND_POSTGRES = _CREDITS_REFUND_TEMPLATE.format(i="%s::integer", least="LEAST")


def credits_seed_params(user_id: int) -> tuple:
    """Positional parameters for ``CREDITS_SEED_*`` (same order in both).

    Reads the module defaults at call time, not import time, so a test (or an
    operator restart with a new ``DEFAULT_CREDITS``) sees the current value.
    """
    return (
        int(user_id),
        DEFAULT_MAX_CONCURRENT_BACKTESTS,
        DEFAULT_CREDITS,
        int(user_id),
    )


def credits_seed_applies(amount: int) -> bool:
    """Whether the default grant could cover a spend of ``amount``.

    A function rather than a constant so both twins read the *current*
    ``DEFAULT_CREDITS`` — ``users_postgres`` imports names once at module load,
    and importing the int would pin it to whatever the value was at import.
    """
    return DEFAULT_CREDITS >= int(amount)


def credits_spend_params(user_id: int, amount: int) -> tuple:
    """Positional parameters for ``CREDITS_SPEND_*`` (same order in both)."""
    return (int(amount), int(user_id), int(amount))


def credits_refund_params(user_id: int, amount: int) -> tuple:
    """Positional parameters for ``CREDITS_REFUND_*`` (same order in both)."""
    return (int(amount), MAX_CREDITS_CAP, int(user_id))


# Parameter-free, so one literal serves both dialects. SUM(CASE) rather than
# COUNT(*) FILTER only to keep it that way (SQLite grew FILTER late).
USER_COUNTS_SQL = """
    SELECT COUNT(*) AS n_users,
           COALESCE(SUM(CASE WHEN role = 'admin' THEN 1 ELSE 0 END), 0) AS n_admins
    FROM users
"""


def admin_user_rows_to_payloads(
    rows: List[sqlite3.Row | Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Project ``list_users_admin`` join rows into the admin payload shape.

    Shared by both twins so the projection cannot drift between them — the
    parity guard compares signatures and DDL, never bodies, so a fork of this
    loop would diverge invisibly. Distinguishes "no entitlements row" (every
    joined column NULL → defaults) from a row that exists with stored values.
    """
    results: List[Dict[str, Any]] = []
    for row in rows:
        data = dict(row)
        has_entitlement_row = (
            data.get("max_concurrent_backtests") is not None
            or data.get("credits") is not None
            or data.get("entitlements_updated_at") is not None
        )
        entitlements = (
            public_entitlements(
                {
                    "user_id": data["id"],
                    "max_concurrent_backtests": data.get("max_concurrent_backtests"),
                    "credits": data.get("credits"),
                    "updated_at": data.get("entitlements_updated_at"),
                    "updated_by_admin_id": data.get("updated_by_admin_id"),
                },
                int(data["id"]),
            )
            if has_entitlement_row
            else default_entitlements(int(data["id"]))
        )
        results.append(public_user_with_entitlements(data, entitlements))
    return results


def admin_user_search_pattern(query: str | None) -> str | None:
    """Return an escaped case-insensitive LIKE pattern for Admin search."""
    text = str(query or "").strip()
    if not text:
        return None
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


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
        cursor.execute(AUTH_SESSIONS_DDL)
        # Pre-hash plaintext-token schema: wipe and rebuild. Existing sessions
        # cannot be re-hashed without the raw token; users must sign in again.
        cursor.execute("PRAGMA table_info(auth_sessions)")
        session_columns = {row[1] for row in cursor.fetchall()}
        if session_columns and "token_hash" not in session_columns:
            cursor.execute("DROP TABLE auth_sessions")
            # Same DDL: the table is gone, so IF NOT EXISTS creates it. Reusing
            # the one literal is what keeps the migrated schema from drifting
            # away from the fresh-install schema.
            cursor.execute(AUTH_SESSIONS_DDL)
            # print(), not logger: see _build_user_store() below. Dropping every
            # live login is not something to do silently -- otherwise the only
            # symptom is a wave of users being signed out with nothing in the
            # deploy log tying it to the release.
            print(
                "auth_sessions: migrated from plaintext tokens to hashed "
                "tokens; all existing sessions were dropped and users must "
                "sign in again"
            )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id
            ON auth_sessions(user_id)
            """
        )
        # Supports purge_expired_sessions(); without it the sweep is a full scan
        # of a table whose whole purpose here is to keep growing.
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires_at
            ON auth_sessions(expires_at)
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
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS password_reset_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                code_hash TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used_at TIMESTAMP,
                cancelled_at TIMESTAMP,
                -- Declared but not enforced on SQLite; the Postgres twin does
                -- enforce it. Same trade-off, same reasons, as the
                -- email_change_requests FK note above.
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_password_reset_requests_user_id
            ON password_reset_requests(user_id)
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_entitlements (
                user_id INTEGER PRIMARY KEY,
                max_concurrent_backtests INTEGER NOT NULL DEFAULT 1,
                credits INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT,
                updated_by_admin_id INTEGER,
                -- Declared but not enforced on SQLite; the Postgres twin does
                -- enforce it (set_entitlements maps its violation to
                -- user_not_found). Same trade-off, same reasons, as the
                -- email_change_requests FK note above.
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
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
        # Commit the login first, then reap. Sweeping inside the same
        # transaction would let a failed housekeeping DELETE take the INSERT
        # down with it -- losing the session the user just asked for.
        _best_effort_write(
            conn,
            cursor,
            "DELETE FROM auth_sessions WHERE expires_at < ?",
            (format_stored_timestamp(now),),
        )
        conn.close()
        return raw_token

    def purge_expired_sessions(self) -> int:
        """Delete session rows past their absolute expiry. Returns the count.

        Revocation is a soft UPDATE and get_user_for_token returns at the
        revoked check before any cleanup, so without this a revoked row lives
        forever; expired rows are only deleted when someone re-presents the
        dead token, which nobody does. Keying the sweep on expires_at collects
        both, and bounds the table at "sessions created in the last
        SESSION_TTL_DAYS" -- the same bound it had before hashing, when logout
        was a DELETE.

        Run from create_session (logins are rare next to reads, and it is the
        moment the table grows) rather than from the read path, which must stay
        free of writes it does not need.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM auth_sessions WHERE expires_at < ?",
            (format_stored_timestamp(_utcnow()),),
        )
        removed = cursor.rowcount
        conn.commit()
        conn.close()
        return max(removed, 0)

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
                auth_sessions.revoked_at AS session_revoked_at,
                user_entitlements.max_concurrent_backtests
                    AS ent_max_concurrent_backtests,
                user_entitlements.credits AS ent_credits,
                user_entitlements.updated_at AS ent_updated_at,
                user_entitlements.updated_by_admin_id AS ent_updated_by_admin_id
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            LEFT JOIN user_entitlements ON user_entitlements.user_id = users.id
            WHERE auth_sessions.token_hash = ?
            """,
            (token_hash,),
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        data = dict(row)
        revoked_at = data.pop("session_revoked_at")
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
            _best_effort_write(
                conn,
                cursor,
                "DELETE FROM auth_sessions WHERE token_hash = ?",
                (token_hash,),
            )
            conn.close()
            return None
        if idle_deadline(last_seen_at) < now:
            _best_effort_write(
                conn,
                cursor,
                "DELETE FROM auth_sessions WHERE token_hash = ?",
                (token_hash,),
            )
            conn.close()
            return None

        if should_touch_last_seen(last_seen_at, now):
            _best_effort_write(
                conn,
                cursor,
                "UPDATE auth_sessions SET last_seen_at = ? WHERE token_hash = ?",
                (format_stored_timestamp(now.replace(microsecond=0)), token_hash),
            )
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
        return _expiry_iso(EMAIL_CHANGE_TTL_MINUTES)

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

    def _password_reset_expiry(self) -> str:
        return _expiry_iso(PASSWORD_RESET_TTL_MINUTES)

    def create_password_reset_request(
        self, user_id: int, code_hash: str
    ) -> Dict[str, Any]:
        """Supersede any in-flight reset request with a fresh one.

        Cancel + insert ride one transaction on one connection, so two racing
        creates cannot leave the earlier-delivered code alive: last write wins
        cleanly, and the loser's code fails as "no active row". Supersede, not
        DELETE -- the append-only log is what the cooldown and daily cap read.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE password_reset_requests SET cancelled_at = ?
            WHERE user_id = ? AND used_at IS NULL AND cancelled_at IS NULL
            """,
            (_utcnow_iso(), user_id),
        )
        cursor.execute(
            """
            INSERT INTO password_reset_requests
                (user_id, code_hash, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, code_hash, _utcnow_iso(), self._password_reset_expiry()),
        )
        conn.commit()
        request_id = cursor.lastrowid
        cursor.execute(
            "SELECT * FROM password_reset_requests WHERE id = ?", (request_id,)
        )
        row = cursor.fetchone()
        conn.close()
        return dict(row)

    def get_active_password_reset(self, user_id: int) -> Optional[Dict[str, Any]]:
        """The user's in-flight reset, or None if absent, used, cancelled, or expired.

        Expiry lives here, not in the route, mirroring get_active_email_change.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM password_reset_requests
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

    def record_password_reset_attempt(self, request_id: int) -> int:
        """Count one wrong code; cancel the request when the cap is reached.

        One conditional UPDATE, not read-then-write: the 5-attempt cap is
        stated as a guarantee, so it must hold under a concurrent burst. The
        ``attempts < cap`` predicate and the cancelling CASE evaluate under the
        row lock the UPDATE takes, so no interleaving can push attempts past
        the cap or leave a capped row active. Returns the attempt count after
        this call; a row that is already inactive (capped, used, or cancelled
        by a concurrent caller) reports the cap.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE password_reset_requests
            SET attempts = attempts + 1,
                cancelled_at = CASE
                    WHEN attempts + 1 >= ? THEN ? ELSE NULL
                END
            WHERE id = ? AND used_at IS NULL AND cancelled_at IS NULL
                AND attempts < ?
            """,
            (
                PASSWORD_RESET_MAX_ATTEMPTS,
                _utcnow_iso(),
                request_id,
                PASSWORD_RESET_MAX_ATTEMPTS,
            ),
        )
        counted = cursor.rowcount == 1
        conn.commit()
        if not counted:
            conn.close()
            return PASSWORD_RESET_MAX_ATTEMPTS
        cursor.execute(
            "SELECT attempts FROM password_reset_requests WHERE id = ?",
            (request_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise ValueError("password_reset_request_not_found")
        return int(row["attempts"])

    def mark_password_reset_used(self, request_id: int) -> bool:
        """Consume the code: a compare-and-swap only one caller can win.

        Acting on rowcount is what makes redemption single-use under
        concurrency -- the loser of two simultaneous redeems sees False and
        reports the generic failure instead of also resetting the password.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE password_reset_requests SET used_at = ?
            WHERE id = ? AND used_at IS NULL AND cancelled_at IS NULL
            """,
            (_utcnow_iso(), request_id),
        )
        won = cursor.rowcount == 1
        conn.commit()
        conn.close()
        return won

    def cancel_password_reset(self, user_id: int) -> None:
        """Deactivate without deleting, scoped to still-active rows.

        Same shape as cancel_email_change: the row must survive for the
        status-blind cooldown reads, and an already-used row keeps its used_at.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE password_reset_requests SET cancelled_at = ?
            WHERE user_id = ? AND used_at IS NULL AND cancelled_at IS NULL
            """,
            (_utcnow_iso(), user_id),
        )
        conn.commit()
        conn.close()

    def last_password_reset_request_at(self, user_id: int) -> Optional[str]:
        """Status-blind: cancelled/used rows still gate the cooldown, so
        cancelling can never be used to mint codes faster."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT created_at FROM password_reset_requests
            WHERE user_id = ? ORDER BY id DESC LIMIT 1
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()
        return str(row["created_at"]) if row else None

    def password_reset_request_times_since(self, user_id: int, since: str) -> List[str]:
        """created_at of every request at or after `since`, oldest first.

        String comparison over the fixed-width ISO-8601 form both twins write;
        see email_change_request_times_since for why that is sound.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT created_at FROM password_reset_requests
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

    def get_entitlements(self, user_id: int) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM user_entitlements WHERE user_id = ?",
            (int(user_id),),
        )
        row = cursor.fetchone()
        conn.close()
        return public_entitlements(row, int(user_id))

    def try_spend_credits(self, user_id: int, amount: int = 1) -> Optional[int]:
        """Debit ``amount`` credits; ``None`` means the balance was too low.

        Returns the balance remaining after a successful debit. ``None`` is a
        refusal, not a fault — an account at zero is the metering working — so
        callers answer it with 402 rather than 500.
        """
        amount = int(amount)
        if amount <= 0:
            raise ValueError("invalid_credit_amount")
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if credits_seed_applies(amount):
                cursor.execute(CREDITS_SEED_SQLITE, credits_seed_params(user_id))
            cursor.execute(CREDITS_SPEND_SQLITE, credits_spend_params(user_id, amount))
            if cursor.rowcount != 1:
                # Refused. Note this can never coincide with a fresh seed: the
                # seed only runs when the default covers the spend, so either
                # it just created a row that satisfies the predicate, or the
                # row already existed and the seed was a no-op. A refusal
                # therefore never leaves today's default frozen into a new row.
                conn.rollback()
                return None
            cursor.execute(
                "SELECT credits FROM user_entitlements WHERE user_id = ?",
                (int(user_id),),
            )
            row = cursor.fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return int(dict(row)["credits"]) if row else None

    def refund_credits(self, user_id: int, amount: int = 1) -> Optional[int]:
        """Return ``amount`` credits, clamped to ``MAX_CREDITS_CAP``.

        ``None`` when the account has no entitlements row — nothing was ever
        debited from it, so there is nothing to give back.
        """
        amount = int(amount)
        if amount <= 0:
            raise ValueError("invalid_credit_amount")
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                CREDITS_REFUND_SQLITE, credits_refund_params(user_id, amount)
            )
            if cursor.rowcount != 1:
                conn.rollback()
                return None
            cursor.execute(
                "SELECT credits FROM user_entitlements WHERE user_id = ?",
                (int(user_id),),
            )
            row = cursor.fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
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
        # One connection, one transaction: the user-existence check rides
        # inside it (SQLite never enforces the FK here — see the DDL note on
        # email_change_requests — so a check on a separate connection could
        # pass for a row deleted before the write lands, leaving a ghost
        # entitlements row), and the read-back does too, instead of a third
        # round-trip after commit.
        conn = self._get_connection()
        conn.isolation_level = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM users WHERE id = ?", (int(user_id),))
            if not cursor.fetchone():
                raise ValueError("user_not_found")
            cursor.execute(ENTITLEMENTS_UPSERT_SQLITE, entitlements_upsert_params(
                user_id, next_max, next_credits, updated_by_admin_id
            ))
            cursor.execute(
                "SELECT * FROM user_entitlements WHERE user_id = ?",
                (int(user_id),),
            )
            row = cursor.fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
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
        """Role and entitlements for one user, in a single transaction.

        ``PATCH /api/admin/users/{id}`` advertises one atomic change. Doing it
        as a role write then ``set_entitlements`` meant a failure on the
        second leg left the role change committed behind a 500 response, so the
        console showed the old row while the database held the new one.
        """
        normalized_role: Optional[str] = None
        if role is not None:
            normalized_role = (role or "").strip().lower()
            if normalized_role not in VALID_ROLES:
                raise ValueError("invalid_role")
        next_max, next_credits = validate_entitlement_patch(
            max_concurrent_backtests, credits
        )
        touches_entitlements = next_max is not None or next_credits is not None

        conn = self._get_connection()
        conn.isolation_level = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (int(user_id),))
            row = cursor.fetchone()
            if not row:
                raise ValueError("user_not_found")
            if normalized_role is not None:
                current_role = dict(row)["role"]
                if normalized_role != "admin" and current_role == "admin":
                    cursor.execute("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'")
                    if int(cursor.fetchone()["n"] or 0) <= 1:
                        raise ValueError("last_admin")
                cursor.execute(
                    "UPDATE users SET role = ? WHERE id = ?",
                    (normalized_role, int(user_id)),
                )
            if touches_entitlements:
                cursor.execute(ENTITLEMENTS_UPSERT_SQLITE, entitlements_upsert_params(
                    user_id, next_max, next_credits, updated_by_admin_id
                ))
            # Read back inside the transaction: what this returns is exactly
            # what this patch committed, not whatever a concurrent admin wrote
            # between commit and a fresh pair of lookups.
            cursor.execute("SELECT * FROM users WHERE id = ?", (int(user_id),))
            user_row = cursor.fetchone()
            cursor.execute(
                "SELECT * FROM user_entitlements WHERE user_id = ?",
                (int(user_id),),
            )
            ent_row = cursor.fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        if not user_row:
            return None
        return public_user_with_entitlements(
            user_row,
            public_entitlements(dict(ent_row) if ent_row else None, int(user_id)),
        )

    def promote_first_admin(self, user_id: int) -> Dict[str, Any]:
        """Promote ``user_id`` to admin iff no admin row exists yet."""
        conn = self._get_connection()
        conn.isolation_level = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'")
            n = int(cursor.fetchone()["n"] or 0)
            if n > 0:
                raise ValueError("admin_exists")
            cursor.execute("SELECT * FROM users WHERE id = ?", (int(user_id),))
            row = cursor.fetchone()
            if not row:
                raise ValueError("user_not_found")
            cursor.execute(
                "UPDATE users SET role = 'admin' WHERE id = ?",
                (int(user_id),),
            )
            conn.commit()
            cursor.execute("SELECT * FROM users WHERE id = ?", (int(user_id),))
            row = cursor.fetchone()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        if not row:
            raise ValueError("user_not_found")
        return public_user(row)

    def count_users_and_admins(self) -> Dict[str, int]:
        """Total accounts and how many are admins, in one query.

        The admin stats header wants both; separate COUNTs were two
        round-trips to answer one question. SUM(CASE) rather than COUNT(*)
        FILTER so the literal is identical in both twins' dialects.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(USER_COUNTS_SQL)
        row = cursor.fetchone()
        conn.close()
        data = dict(row) if row else {}
        return {
            "users": int(data.get("n_users") or 0),
            "admins": int(data.get("n_admins") or 0),
        }

    def count_users(self, query: str | None = None) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        pattern = admin_user_search_pattern(query)
        where_sql = ""
        params: tuple[str, ...] = ()
        if pattern is not None:
            where_sql = (
                " WHERE lower(email) LIKE lower(?) ESCAPE '\\'"
                " OR lower(display_name) LIKE lower(?) ESCAPE '\\'"
            )
            params = (pattern, pattern)
        cursor.execute(f"SELECT COUNT(*) AS n FROM users{where_sql}", params)
        row = cursor.fetchone()
        conn.close()
        return int(row["n"] if row else 0)

    def list_users_admin(
        self, *, limit: int = 100, offset: int = 0, query: str | None = None
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        conn = self._get_connection()
        cursor = conn.cursor()
        pattern = admin_user_search_pattern(query)
        where_sql = ""
        params: tuple[object, ...] = (limit, offset)
        if pattern is not None:
            where_sql = (
                " WHERE lower(users.email) LIKE lower(?) ESCAPE '\\'"
                " OR lower(users.display_name) LIKE lower(?) ESCAPE '\\'"
            )
            params = (pattern, pattern, limit, offset)
        cursor.execute(
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
            LIMIT ? OFFSET ?
            """,
            params,
        )
        rows = cursor.fetchall()
        conn.close()
        return admin_user_rows_to_payloads(rows)

    def get_user_admin(self, user_id: int) -> Optional[Dict[str, Any]]:
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        return public_user_with_entitlements(user, self.get_entitlements(user_id))


def _build_user_store():
    # Resolve the session HMAC key before anything can serve a request. It is
    # only read on the session paths, so without this a prod deploy missing
    # SESSION_HASH_SECRET boots clean, answers /health 200, and then 500s every
    # login and every authenticated request. Failing the boot instead leaves
    # the previous Render version live.
    require_session_hash_secret()

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
