"""Shared psycopg3 connection pools, one per database URL (T4).

Replaces the fresh psycopg.connect() (a full TLS handshake to Neon) every
store call used to pay. Small and short-lived by design: max_size 5 fits the
single-worker deployment, and max_idle 300s closes idle sockets before Neon's
scale-to-zero suspend can hand back a dead one. row_factory is configured at
pool construction because every twin relies on dict-style row access.
"""

from __future__ import annotations

import threading
from typing import Dict

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from dashboard.backend.db_url import describe_database_url

# Max seconds a caller waits for a pooled connection before .connection()
# raises PoolTimeout (a psycopg.OperationalError subclass). Bounds request
# latency on a down/cold DB: psycopg.connect() failed fast, but a pool retries
# creation in the background, so an unbounded wait would inherit the 30s pool
# default. 10s tolerates a Neon scale-to-zero resume while failing loud on a
# genuine outage. A module constant (not env) so tests can monkeypatch it low
# and not pay the full wait on the fail-loud "unreachable URL" cases.
POOL_TIMEOUT_SECONDS = 10.0

_pools: Dict[str, ConnectionPool] = {}
_lock = threading.Lock()


def get_pool(database_url: str) -> ConnectionPool:
    """One cached pool per URL; construction is lazy and logged."""
    with _lock:
        pool = _pools.get(database_url)
        if pool is None:
            pool = ConnectionPool(
                database_url,
                min_size=0,
                max_size=5,
                max_idle=300,
                timeout=POOL_TIMEOUT_SECONDS,
                kwargs={"row_factory": dict_row},
                open=True,
            )
            _pools[database_url] = pool
            print(f"🏊 pg pool created for {describe_database_url(database_url)}")
    return pool


def close_all_pools() -> None:
    """Close every cached pool and forget it.

    Long-running processes never need this -- the pools are the point. It
    exists for the two places that *end*: the test teardown below, and
    short-lived CLI scripts. Without it a script that touched Postgres exits
    through psycopg_pool's own teardown, which emits
    ``couldn't stop thread 'pool-1-worker-N' within 5.0 seconds`` once per
    worker *after* the script's own success message -- so a clean run reads
    like a partial failure.

    Pools are dropped from the cache, not just closed, because a closed pool
    left in ``_pools`` would be handed to the next ``get_pool()`` caller and
    raise on use.
    """
    with _lock:
        for pool in _pools.values():
            try:
                pool.close()
            except Exception:
                pass  # best-effort teardown: a pool that won't close must not break test reset
        _pools.clear()


def _reset_for_tests() -> None:
    close_all_pools()
