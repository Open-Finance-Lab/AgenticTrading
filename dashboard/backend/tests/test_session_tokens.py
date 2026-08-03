"""Session-token hashing and lifetime policy (``session_tokens.py``).

The store-level behaviour these functions drive lives in ``test_auth.py``; this
file covers the policy module itself, where the failure modes are all about
*configuration* rather than SQL.
"""

import pytest

from dashboard.backend import session_tokens
from dashboard.backend.session_tokens import (
    _DEV_FALLBACK_SECRET,
    hash_session_token,
    new_session_token,
    require_session_hash_secret,
    session_hash_secret,
    session_idle_hours,
    session_ttl_days,
    should_touch_last_seen,
)


@pytest.fixture(autouse=True)
def _forget_reported_problems():
    """The env-problem announcements dedupe; each test needs a clean slate."""
    session_tokens._reported_env_problems.clear()
    yield
    session_tokens._reported_env_problems.clear()


@pytest.fixture
def unconfigured(monkeypatch):
    """No secret, and nothing marking the process as production."""
    monkeypatch.delenv("SESSION_HASH_SECRET", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("ATL_ENV", raising=False)


def test_production_without_a_secret_refuses_to_hash(monkeypatch, unconfigured):
    monkeypatch.setenv("RENDER", "true")
    with pytest.raises(RuntimeError, match="SESSION_HASH_SECRET"):
        session_hash_secret()


def test_production_without_a_secret_fails_the_startup_check(monkeypatch, unconfigured):
    monkeypatch.setenv("ATL_ENV", "production")
    with pytest.raises(RuntimeError, match="SESSION_HASH_SECRET"):
        require_session_hash_secret()


def test_the_dev_fallback_announces_itself_on_stdout(unconfigured, capsys):
    """print(), not logger.warning().

    ``dashboard.backend.*`` loggers sit at WARNING with no handler in every real
    deployment, so a logger call here emits nothing -- the same trap
    ``_build_user_store`` documents. A fallback HMAC key that swaps itself in
    silently is exactly the kind of thing that has to reach the deploy log.
    """
    require_session_hash_secret()
    assert "SESSION_HASH_SECRET" in capsys.readouterr().out
    assert session_hash_secret() == _DEV_FALLBACK_SECRET


def test_hashing_stays_quiet_on_the_per_request_path(unconfigured, capsys):
    """The announcement is a startup event, not a per-request one.

    get_user_for_token() hashes on every authenticated call; a print() in that
    path would bury the deploy log under one line per request.
    """
    require_session_hash_secret()
    capsys.readouterr()
    hash_session_token("abc")
    hash_session_token("def")
    assert capsys.readouterr().out == ""


def test_a_garbage_lifetime_is_reported_not_swallowed(monkeypatch, capsys):
    """A typo'd TTL silently reverting to the default is an auth-policy change.

    ``SESSION_TTL_DAYS=7d`` looks like it worked. Falling back is the right
    behaviour -- doing it without a word is not.
    """
    monkeypatch.setenv("SESSION_TTL_DAYS", "7d")
    assert session_ttl_days() == 7
    assert "SESSION_TTL_DAYS" in capsys.readouterr().out


def test_an_out_of_range_lifetime_is_reported_not_swallowed(monkeypatch, capsys):
    monkeypatch.setenv("SESSION_IDLE_HOURS", "0")
    assert session_idle_hours() == 24
    assert "SESSION_IDLE_HOURS" in capsys.readouterr().out


def test_a_valid_lifetime_override_is_silent(monkeypatch, capsys):
    monkeypatch.setenv("SESSION_TTL_DAYS", "30")
    assert session_ttl_days() == 30
    assert capsys.readouterr().out == ""


def test_the_digest_never_contains_the_raw_token():
    token = new_session_token()
    assert token not in hash_session_token(token)


def test_the_throttle_measures_age_from_the_stored_timestamp():
    """Guards the property that keeps an active user signed in.

    ``last_seen_at`` only advances when the throttle fires, so the age compared
    here has to be measured against the *stored* value (which keeps growing)
    rather than the previous request. Measure it per-request and a user who
    polls just inside the throttle window never refreshes and gets logged out
    at the idle deadline despite never being idle.
    """
    from datetime import datetime, timedelta, timezone

    stored = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
    assert not should_touch_last_seen(stored, stored + timedelta(seconds=540))
    assert should_touch_last_seen(stored, stored + timedelta(seconds=1080))
