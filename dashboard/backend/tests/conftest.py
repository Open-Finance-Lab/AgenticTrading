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

# A developer's deployment pseudonymization secret must never affect tests.
os.environ.pop("ANALYTICS_PSEUDONYMIZATION_KEY", None)

# Hermetic HMAC key for session-token digests (see session_tokens.py).
os.environ["SESSION_HASH_SECRET"] = "test-session-hash-secret"

# Cookie CSRF is on by default in production; the broad TestClient suite does
# not yet attach X-CSRF-Token on every mutating call. Dedicated CSRF tests
# re-enable enforcement via monkeypatch (see test_csrf.py).
os.environ["ATL_CSRF"] = "0"

# The session cookie's name and Secure flag key off these (auth_cookies.py).
# Inherited from a shell (a sourced prod .env, a Render-like deploy env) they
# flip every test onto __Host-atl_session + Secure, which TestClient's plain
# http://testserver jar refuses — surfacing as misleading "missing session
# cookie" failures across the auth-dependent modules.
for _cookie_var in ("ATL_COOKIE_SECURE", "ATL_ENV", "RENDER"):
    os.environ.pop(_cookie_var, None)

# Same guarantee for CONTENT_DATABASE_URL: it selects Postgres backends for the
# agent / agent-version / strategy stores, so a value inherited from the
# developer's environment (a sourced prod .env, a deploy shell) would point the
# whole test suite at a real database. Strip it before any backend module is
# imported.
os.environ.pop("CONTENT_DATABASE_URL", None)

# Same guarantee for AGENT_RUNS_DATABASE_URL: it selects the Postgres backend for
# backtest run history (agent_runs, equity_timeseries, trades,
# backtest_decisions, run_manifest). A value inherited from the developer's
# environment would point the whole suite at the live runs database -- whose
# @pg_only-style destructive helpers would then run against prod. Strip it
# before any backend module is imported.
os.environ.pop("AGENT_RUNS_DATABASE_URL", None)

# T1+ scale knobs are read once at import (like MAX_ACTIVE_RUNS_PER_AGENT); a
# stray shell value would silently skew the whole run. Same rationale as the
# DB-URL strips above. Later tiers append their vars here.
os.environ.pop("MARKET_DATA_CACHE_MAX_ENTRIES", None)
os.environ.pop("BASELINE_QUEUE_MAX", None)
os.environ.pop("EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS", None)
os.environ.pop("MAX_ACTIVE_RUNS_GLOBAL", None)
os.environ.pop("MAX_ACTIVE_DASHBOARD_BACKTESTS", None)
os.environ.pop("AGENT_AUTH_CACHE_TTL_SECONDS", None)
os.environ.pop("MAX_LEGACY_ACTIVE_PER_SESSION", None)
os.environ.pop("MAX_LEGACY_ACTIVE_GLOBAL", None)
os.environ.pop("ALPACA_HTTP_TIMEOUT_SECONDS", None)
os.environ.pop("ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS", None)
os.environ.pop("LEGACY_SESSION_RETENTION_SECONDS", None)
# Read once at import into users.DEFAULT_MAX_CONCURRENT_BACKTESTS, which every
# entitlement default and every per-account cap test resolves through.
os.environ.pop("DEFAULT_MAX_CONCURRENT_BACKTESTS", None)
os.environ.pop("DEFAULT_CREDITS", None)
# Credit metering is strict opt-in and off by default (see
# domain/entitlements/credits.py). A developer with it exported would otherwise
# have every LLM-backtest test 402 on an empty balance -- and, worse, a suite
# that passed for them would be asserting the metered path everywhere while CI
# asserted the unmetered one. Tests that exercise metering set it via
# monkeypatch.
os.environ.pop("CREDITS_METERING_ENABLED", None)

# Mail credentials: a developer with a real BREVO_API_KEY exported would
# otherwise have the suite send live email, and would see the
# unconfigured-provider tests fail for a reason that has nothing to do with
# their change. Individual tests set these back via monkeypatch.
os.environ.pop("BREVO_API_KEY", None)
os.environ.pop("ACCOUNT_EMAIL_FROM", None)
os.environ.pop("ACCOUNT_EMAIL_FROM_NAME", None)

# Billing tests must never inherit a developer's Stripe credentials or enable
# Test Mode accidentally. Individual tests set explicit fake values.
os.environ.pop("ATL_STRIPE_TEST_BILLING_ENABLED", None)
os.environ.pop("STRIPE_SECRET_KEY", None)
os.environ.pop("STRIPE_WEBHOOK_SECRET", None)

# Daily Leaderboard knobs. LEADERBOARD_DAILY_AUTO_DEPLOY is the flag that lets a
# public GET of ?period=daily kick off deploy_model_run for every competition
# entry -- real, billable LLM calls. A developer with it exported would have the
# suite spend their credits. LEADERBOARD_DAILY_REFRESH_SECRET is stripped so the
# secret-gate tests see a known-unset baseline rather than the shell's value.
os.environ.pop("LEADERBOARD_DAILY_AUTO_DEPLOY", None)
os.environ.pop("LEADERBOARD_DAILY_REFRESH_SECRET", None)
# Same known-unset baseline for the first-admin bootstrap secret: a developer
# with it exported would otherwise make the unconfigured-refusal tests fail,
# and could accidentally promote the suite's throwaway accounts.
os.environ.pop("ADMIN_BOOTSTRAP_SECRET", None)
# The iFinD A-share credentials, for the same reason plus a sharper one: a
# developer with IFIND_REFRESH_TOKEN exported (from `dashboard/.env`, which
# reaches os.environ as soon as any sibling test imports `app`) makes every
# static-token client in the suite open with an unexpected token exchange,
# silently consuming the first queued response of each fake session. That
# presents as ~27 unrelated assertion failures rather than as a config leak.
os.environ.pop("IFIND_REFRESH_TOKEN", None)
os.environ.pop("IFIND_ACCESS_TOKEN", None)
os.environ.pop("IFIND_BASE_URL", None)


@atexit.register
def _cleanup_test_db_dir() -> None:
    """Remove the temporary database directory when the test process exits."""
    shutil.rmtree(_TEST_DB_DIR, ignore_errors=True)


from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import requests  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "daily_refresh_background: test drives the real Daily Leaderboard "
        "background scheduler; opts out of the autouse no-op guard.",
    )


@pytest.fixture(autouse=True)
def _clear_ifind_access_token_cache():
    """Keep the process-wide iFinD token cache from leaking between tests.

    The cache is deliberately module-level (one exchange per process, per the
    design), which means without this every test that exchanges a token hands
    it to the next one -- and a fake session that expected to be asked for a
    token would silently never be called.
    """
    from dashboard.backend.infrastructure.market_data import ifind_client

    ifind_client._ACCESS_TOKEN_CACHE.clear()
    yield
    ifind_client._ACCESS_TOKEN_CACHE.clear()


@pytest.fixture(autouse=True)
def _no_background_daily_refresh(request, monkeypatch, tmp_path):
    """Keep the Daily Leaderboard's background worker out of the test process.

    ``get_leaderboard(period="daily")`` calls
    ``maybe_schedule_daily_leaderboard_refresh()`` on every request, so *any*
    test that hits ``GET /api/v1/leaderboard?period=daily`` would otherwise
    spawn a real thread into ``ensure_leaderboard_runs`` (live Alpaca fetch) and
    -- with LEADERBOARD_DAILY_AUTO_DEPLOY on -- ``deploy_model_run`` for every
    competition entry. That thread outlives the test, races the temp DB, and
    writes state into the repo working tree.

    Also redirects the refresh state file at all times, including for the tests
    that opt out, so nothing lands in dashboard/storage/data/.
    """
    from dashboard.backend.domain.leaderboard import service as lb_service

    monkeypatch.setattr(
        lb_service,
        "_DAILY_REFRESH_STATE_PATH",
        tmp_path / "leaderboard_daily_refresh.json",
    )
    if request.node.get_closest_marker("daily_refresh_background"):
        return
    monkeypatch.setattr(
        lb_service, "maybe_schedule_daily_leaderboard_refresh", lambda **_: False
    )


@pytest.fixture(autouse=True)
def _reset_shared_scale_state(monkeypatch):
    """The market-data store is a module-level cache shared across the test
    process; without a per-test reset, one test's synthetic bars would be
    served to every later test with the same (symbols, dates) key. The auth
    cache and pg pools are the same kind of process-global state; reset all
    three, and cap the pool-connect wait so the fail-loud "unreachable URL"
    tests raise in ~1s instead of blocking the whole prod-default timeout."""
    from dashboard.backend.domain.backtesting import baseline_worker, market_data_store
    from dashboard.backend.domain.agents import auth_cache
    from dashboard.backend import db_pool
    from dashboard.backend.api import auth as auth_api
    from dashboard.backend.api.routers import credits as credits_router
    from dashboard.backend.api.routers import leaderboard as leaderboard_router
    from dashboard.backend.api.routers import backtests as backtests_router

    monkeypatch.setattr(db_pool, "POOL_TIMEOUT_SECONDS", 1.0)
    market_data_store._reset_for_tests()
    baseline_worker._reset_for_tests()
    auth_cache._reset_for_tests()
    db_pool._reset_for_tests()
    # Login/signup limiters are process-global; without a reset, the shared
    # TestClient peer (testclient) burns the signup IP budget across the suite.
    auth_api._LOGIN_IP_LIMITER.reset()
    auth_api._LOGIN_EMAIL_LIMITER.reset()
    auth_api._SIGNUP_IP_LIMITER.reset()
    auth_api._SIGNUP_EMAIL_LIMITER.reset()
    auth_api._FORGOT_IP_LIMITER.reset()
    auth_api._FORGOT_EMAIL_LIMITER.reset()
    auth_api._FORGOT_GLOBAL_LIMITER.reset()
    auth_api._RESET_IP_LIMITER.reset()
    auth_api._RESET_EMAIL_LIMITER.reset()
    # Same reason: every TestClient request shares one client key, so the daily
    # refresh budget would otherwise be consumed cumulatively across the suite
    # and later tests would start seeing 429s.
    leaderboard_router._daily_refresh_rate_limiter.reset()
    credits_router._CHECKOUT_LIMITER.reset()
    credits_router._ORDER_POLL_LIMITER.reset()
    credits_router._ADMIN_REFUND_LIMITER.reset()
    credits_router._WEBHOOK_LIMITER.reset()
    # Dashboard backtest slots are process-global. Tests mock the worker so
    # _finalize_slot never runs; without a reset the 20-slot process cap
    # refuses later /backtest/run calls with success:false.
    backtests_router._reset_slots_for_tests()
    backtests_router._backtest_rate_limiter.reset()
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
    db_pool._reset_for_tests()
    backtests_router._reset_slots_for_tests()
    backtests_router._backtest_rate_limiter.reset()


@pytest.fixture(autouse=True)
def _no_live_yahoo(monkeypatch):
    """No test may reach query1.finance.yahoo.com.

    Restores a failure signal the outage guard would otherwise erase. Now that
    ``market_index_baselines_with_status`` swallows ``requests.RequestException``
    — and ``ConnectionError`` is exactly what a socket-blocked CI box raises —
    a test that forgets to stub the fetch no longer fails. It *passes*, silently,
    with empty baselines. That is the signal issue #320 and PRs #331/#334 relied
    on to notice tests depending on live Yahoo.

    Narrow on purpose: blocking sockets wholesale would break the @pg_only tier,
    which needs a real connection to TEST_POSTGRES_URL.

    Rebinds the *name* ``requests`` inside the ``_yahoo`` module rather than
    setting an attribute on the requests module itself — the latter is one
    shared object, so it would swap HTTP for the whole process.
    """
    from dashboard.backend.domain.leaderboard.strategies import _yahoo

    def _blocked(*_args, **_kwargs):
        raise AssertionError(
            "test reached live Yahoo. Stub "
            "'dashboard.backend.equity_plot.fetch_index_hourly' (or _yahoo.requests) "
            "-- an unstubbed fetch now degrades silently instead of failing."
        )

    monkeypatch.setattr(
        _yahoo,
        "requests",
        SimpleNamespace(get=_blocked, RequestException=requests.RequestException),
    )
