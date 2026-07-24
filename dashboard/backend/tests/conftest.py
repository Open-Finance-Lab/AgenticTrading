"""Pytest configuration for the dashboard backend test suite.

Phase 0.5 — Isolate the test database.

The backend resolves its SQLite path at *import time*::

    # database.py
    DB_PATH = Path(os.getenv("DATABASE_PATH", str(DEFAULT_DB_PATH)))
    ...
    db = BacktestDatabase()  # built once, against DB_PATH

and every store/repository reads ``DB_PATH`` / ``DEFAULT_DB_PATH`` from that same
module. So pointing ``DATABASE_PATH`` at a fresh temporary file *before any
backend module is imported* isolates the entire data layer in one place.

This module is imported by pytest before the test modules in this directory, so
setting the environment variable here (at import time, not in a fixture)
guarantees it is in effect before ``app`` / ``database`` are first imported.

Guarantees:
* The live database ``dashboard/storage/data/backtest.db`` is never read,
  written, copied, reset, or deleted by the test run.
* An ambient ``USERS_DATABASE_URL`` or ``CONTENT_DATABASE_URL`` in the developer's
  shell can never make the test run reach for a real Postgres store: both are
  unset here for the same import-time reason ``DATABASE_PATH`` is pinned above.
* Schema creation and migrations run automatically when ``BacktestDatabase`` is
  constructed against the temporary path.
* Production behavior is unchanged: this only affects the pytest process, which
  does not run in production.
"""

import atexit
import os
import shutil
import tempfile

# Create an isolated, empty temporary database BEFORE backend modules import.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="atl_test_db_")
_TEST_DB_PATH = os.path.join(_TEST_DB_DIR, "test_backtest.db")

# Only set it for the test process; never touch the real DATABASE_PATH/live file.
os.environ["DATABASE_PATH"] = _TEST_DB_PATH

# Never let a stray USERS_DATABASE_URL from the developer's shell make
# dashboard.backend.users._build_user_store() reach for Postgres at import
# time; tests must always fall back to the plain SQLite UserStore.
os.environ.pop("USERS_DATABASE_URL", None)

# Same guarantee for CONTENT_DATABASE_URL: it selects Postgres backends for the
# agent / agent-version / strategy stores, so a value inherited from the
# developer's environment (a sourced prod .env, a deploy shell) would point the
# whole test suite at a real database. Strip it before any backend module is
# imported.
os.environ.pop("CONTENT_DATABASE_URL", None)

# T1+ scale knobs are read once at import (like MAX_ACTIVE_RUNS_PER_AGENT); a
# stray shell value would silently skew the whole run. Same rationale as the
# DB-URL strips above. Later tiers append their vars here.
os.environ.pop("MARKET_DATA_CACHE_MAX_ENTRIES", None)
os.environ.pop("BASELINE_QUEUE_MAX", None)
os.environ.pop("EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS", None)
os.environ.pop("MAX_ACTIVE_RUNS_GLOBAL", None)
os.environ.pop("AGENT_AUTH_CACHE_TTL_SECONDS", None)


@atexit.register
def _cleanup_test_db_dir() -> None:
    """Remove the temporary database directory when the test process exits."""
    shutil.rmtree(_TEST_DB_DIR, ignore_errors=True)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_shared_scale_state():
    """The market-data store is a module-level cache shared across the test
    process; without a per-test reset, one test's synthetic bars would be
    served to every later test with the same (symbols, dates) key."""
    from dashboard.backend.domain.backtesting import baseline_worker, market_data_store
    from dashboard.backend.domain.agents import auth_cache
    market_data_store._reset_for_tests()
    baseline_worker._reset_for_tests()
    auth_cache._reset_for_tests()
    yield
    # Best-effort drain so a job enqueued in this test doesn't leak into the
    # next. Note pytest tears fixtures down LIFO, so a test's own monkeypatches
    # may already be reverted here — in practice patched baseline fakes return
    # instantly, so the queue is empty long before teardown. Any T2 test that
    # gates or slows the worker must call baseline_worker.wait_idle() itself
    # before returning (every test in this plan does).
    baseline_worker.wait_idle(timeout=5)
    baseline_worker._reset_for_tests()
    market_data_store._reset_for_tests()
    auth_cache._reset_for_tests()
