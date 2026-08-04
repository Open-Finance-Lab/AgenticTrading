"""The @pg_only destructive-fixture guard must refuse non-local databases.

These are pure-function tests -- no live Postgres -- so they run in ordinary
CI, unlike the @pg_only tier they protect. See issue #136.
"""

import importlib

import pytest

from dashboard.backend.tests._postgres_testing import require_local_postgres_url

# Re-import the destructive fixtures so we can prove each one refuses a remote
# URL. Requesting them here (rather than in their own @pg_only tests) exercises
# the guard in ordinary CI: it must raise *before* any connection is attempted.
from dashboard.backend.tests.test_agent_store_postgres import (  # noqa: F401
    pg_agent_store,
    pg_version_store,
)
from dashboard.backend.tests.test_backtest_db_postgres import (  # noqa: F401
    pg_backtest_db,
)
from dashboard.backend.tests.test_strategy_store_postgres import (  # noqa: F401
    pg_strategy_store,
)
from dashboard.backend.tests.test_users_postgres import (  # noqa: F401
    temp_postgres_store,
)


# Sample hosts here are placeholders, never a real endpoint id. The guard only
# asks "is this host local?", so `ep-x` satisfies every case a real one would --
# and the rest of the suite already standardises on it (test_db_url.py,
# test_agent_store_postgres.py, ...). This repo is public: naming a live Neon
# endpoint in a fixture publishes the exact database an attacker would target,
# and unlike a password it cannot be rotated away afterwards.


def test_rejects_remote_neon_host():
    remote = "postgresql://user:pw@ep-x-pooler.neon.tech/neondb"
    with pytest.raises(RuntimeError, match="non-local"):
        require_local_postgres_url(remote)


def test_rejects_remote_host_even_with_localhost_in_credentials():
    # Naive substring matching on "localhost" would wrongly allow this; the
    # real host is neon.tech. urlsplit() extracts the host, ignoring userinfo.
    sneaky = "postgresql://localhost:localhost@ep-x.neon.tech/db"
    with pytest.raises(RuntimeError):
        require_local_postgres_url(sneaky)


def test_allows_localhost():
    url = "postgresql://postgres:test@localhost:5433/atl_test"
    assert require_local_postgres_url(url) == url


def test_allows_127_0_0_1():
    url = "postgresql://postgres:test@127.0.0.1:5432/atl_test"
    assert require_local_postgres_url(url) == url


def test_unset_url_passes_through():
    # Unset TEST_POSTGRES_URL drives the skipif that skips the whole tier; the
    # guard only cares about set-but-remote, so None/"" must not raise.
    assert require_local_postgres_url(None) is None
    assert require_local_postgres_url("") == ""


# .invalid never resolves, so RED (an unguarded fixture) fails fast on connect
# rather than hanging; GREEN raises before the fixture ever touches the network.
_REMOTE_URL = "postgresql://user:pw@remote.invalid:5432/db"


@pytest.mark.parametrize(
    "fixture_name, origin_module",
    [
        ("pg_agent_store", "dashboard.backend.tests.test_agent_store_postgres"),
        ("pg_version_store", "dashboard.backend.tests.test_agent_store_postgres"),
        ("pg_strategy_store", "dashboard.backend.tests.test_strategy_store_postgres"),
        ("temp_postgres_store", "dashboard.backend.tests.test_users_postgres"),
        ("pg_backtest_db", "dashboard.backend.tests.test_backtest_db_postgres"),
    ],
)
def test_destructive_fixture_refuses_remote_url(
    fixture_name, origin_module, monkeypatch, request
):
    mod = importlib.import_module(origin_module)
    monkeypatch.setattr(mod, "TEST_POSTGRES_URL", _REMOTE_URL)
    with pytest.raises(RuntimeError, match="non-local"):
        request.getfixturevalue(fixture_name)
