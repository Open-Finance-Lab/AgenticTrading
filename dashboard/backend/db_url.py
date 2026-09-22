"""Describe a database URL for logging, without leaking its credentials.

Used by every store factory (users, agents, agent versions, strategies) to log
*which* Postgres it bound to rather than merely that it bound to "postgres". A
bare backend name cannot distinguish the intended Neon database from staging, or
from a URL with a typo'd host -- they produce byte-identical startup logs, which
is the failure shape CLAUDE.md's "Fail-closed is not fail-visible" section exists
to warn about. (The scoped CONTENT_/USERS_ names rule out an *accidental*
collision with another tool's env var; they do nothing about a wrong value
deliberately set.)

Defined once rather than cloned into each store module (this feature's pattern
everywhere else) because it is credential-scrubbing code: four hand-copied
scrubbers is four chances for one of them to leak a password into a log.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from urllib.parse import urlsplit

# psycopg reads a connection string as a URL only when it starts with one of
# these; anything else is parsed as a keyword DSN. See require_postgres_url.
_POSTGRES_URL_SCHEMES = ("postgresql://", "postgres://")


def require_postgres_url(database_url: str) -> str:
    """Return ``database_url``, or raise if psycopg would read it as a keyword DSN.

    psycopg treats a connection string as a URL only when it starts with
    ``postgresql://`` or ``postgres://``. Anything else is parsed as a keyword
    DSN (``host=... password=...``) and the resulting ProgrammingError quotes
    the *entire* input back::

        missing "=" after ""postgresql://u:hunter2@ep-x.neon.tech/atl"" in
        connection info string

    The store factories construct their twin at import time with no try/except
    -- fail-loud is deliberate -- so that message *is* the boot failure, and it
    carries the live password into the deploy log. Every malformed shape
    observed leaks (a value pasted with wrapping quotes, an uppercase scheme,
    ``postgre://``, a single slash, a leading space, no scheme at all) and every
    well-formed one does not, so the scheme check is the exact boundary rather
    than a heuristic.

    Still fail-loud: a bad value raises here instead of reaching psycopg. The
    message quotes no part of the input, for the same reason
    describe_database_url echoes nothing it could not parse.
    """
    if not isinstance(database_url, str) or not database_url.startswith(
        _POSTGRES_URL_SCHEMES
    ):
        raise ValueError(
            "database URL must start with 'postgresql://' or 'postgres://'. "
            "Refusing to hand it to psycopg, which parses a non-URL as a "
            "keyword DSN and quotes the whole value -- password included -- "
            "into the error it raises. Check the env var for wrapping quotes "
            "or a typo'd scheme."
        )
    return database_url


def describe_database_url(database_url: str) -> str:
    """Return ``host[:port]/dbname`` for ``database_url``, never its credentials.

    Returns ``"?/?"`` for anything not parseable as a URL. That constant is
    deliberate: psycopg also accepts keyword/DSN strings (``host=... dbname=...
    password=...``), and urlsplit puts the *entire* such string -- password
    included -- in ``.path``. Echoing any part of unparseable input back into a
    log is exactly the leak this helper exists to prevent, so it echoes nothing.
    Prod uses a URL, so the degraded case costs only log detail.

    Never raises: this runs inside a factory at import time, and a log helper
    that explodes on an odd URL would take the whole app down with it.
    """
    try:
        parts = urlsplit(database_url)
        host = parts.hostname
        port = "" if parts.port is None else f":{parts.port}"
    except ValueError:
        # urlsplit, or .port on a non-integer port, rejected the input.
        return "?/?"
    if not host:
        return "?/?"
    dbname = parts.path.lstrip("/") or "?"
    # urlsplit strips the brackets from an IPv6 literal, and IPv6 addresses
    # are colon-delimited themselves, so an unbracketed "::1:5432" cannot be
    # read as host-vs-port. Restore them: an ambiguous line is not visibility.
    if ":" in host:
        host = f"[{host}]"
    return f"{host}{port}/{dbname}"


#: Set by `api/routers/backtests.py` in a dashboard backtest child's
#: environment and nowhere else.
BACKTEST_WORKER_ENV = "ATL_BACKTEST_WORKER"


def schema_init_skipped() -> bool:
    """True inside a dashboard backtest child, which must not repeat DDL.

    The parent ran every store's ``_init_schema`` at import, so the schema
    exists by the time it spawns anything; a child that ran it again paid a
    fresh pooled connection and a batch of DDL round trips per store to Neon
    before reading a single bar. Named for the process role rather than the
    DDL so no operator sets it globally, and so later child-only behaviour
    (turning the analytics snapshot projection off, for one) has a home.
    Only the literal ``1`` arms it.
    """
    return os.environ.get(BACKTEST_WORKER_ENV, "").strip() == "1"


#: Seconds this process has spent inside store ``_init_schema()`` calls.
#: Process-global and never reset: it is read once, by the backtest child's
#: script, just before it builds the engine. Do not read it from a request
#: path -- in a long-lived parent it is the whole boot's DDL, not this call's.
_schema_init_seconds = 0.0


def schema_init_seconds() -> float:
    """Total DDL time this process has paid, for the ``starting`` record.

    It is what makes that phase attributable: ``starting`` is one interval
    covering process spawn, pandas and three SDK imports, and the seven store
    singletons those imports construct as a side effect (six of them Postgres
    twins, which are the six that run DDL) -- and a single number that moves
    for any of those reasons cannot say whether removing the DDL helped. In a
    worker it is
    ``0.0`` -- present and zero, which is the evidence the flag fired, not a
    missing field.
    """
    return _schema_init_seconds


def init_schema_unless_worker(label: str, init_schema: Callable[[], None]) -> None:
    """Run ``init_schema`` unless this process is a dashboard backtest child.

    Defined once rather than cloned into each twin's ``__init__`` -- the same
    call this module's header makes for describe_database_url, for a sharper
    version of the same reason. Eleven hand-copied
    ``if schema_init_skipped(): print(...) else: self._init_schema()`` blocks
    are eleven chances for one to invert its condition or lose its log line,
    and the two failures are asymmetric. An inverted guard fails in the
    *parent*: it stops running a ``CREATE TABLE IF NOT EXISTS`` that would
    have been free, and the first symptom is an UndefinedTable or
    UndefinedColumn on the deployed database, from a store nobody edited. A
    lost log line makes "the child skipped DDL" and "this twin was never
    constructed" the same empty stdout. One condition, one line, one place to
    read them.

    **Both directions print, and the ran-it direction prints its cost.** A skip
    line alone would leave "the flag reached the child" and "this build has no
    guard at all" producing the same silence in a log. And the timing is the
    only pre-change number the deploy that removes the cost can still produce:
    the parent runs this same DDL against the same databases at boot and is
    never a worker, so the parent's boot log *is* the baseline the child's skip
    is measured against. Locally every store resolves to its SQLite twin and
    never reaches here at all, which is why no local before/after of the flag
    means anything.

    ``label`` is the token that store's factory already prints in its
    ``<label> backend: postgres (...)`` boot line, so ``grep 'backend:'`` reads
    as one story.

    **What is skipped is wider than the name.** ``_init_schema`` is DDL in
    eight of the eleven twins, but three of them also run data statements
    inside it -- the credits twin seeds the default grant pool and backfills
    two ledger columns, the model-providers twin scrubs ``api_key_enc`` on
    revoked credentials and seeds the provider registry, and the user twin
    repairs ``users.user_group`` values outside the allowed set. This call
    skips those in a child too. They are safe to skip **because the parent
    applied them at boot before spawning anything**, which is a property of
    how dashboard backtests are launched rather than of this function: a
    worker that ever ran without a parent ahead of it would not have them.
    ``tests/test_backtest_worker_schema_skip.py`` keeps that set declared, so
    a fourth one cannot arrive unnoticed -- read it before adding a migration
    to any guarded ``_init_schema``.
    """
    global _schema_init_seconds
    if schema_init_skipped():
        print(f"{label} backend: schema init skipped (backtest worker)", flush=True)
        return
    started = time.perf_counter()
    init_schema()
    elapsed = time.perf_counter() - started
    _schema_init_seconds += elapsed
    print(f"{label} backend: schema init {elapsed:.2f}s", flush=True)
