"""Robinhood OAuth state, pending-link handoff, and broker token-store tests.

Everything here is offline: no HTTP, no Robinhood, no Postgres. The three
subjects are the security-critical plumbing around live-brokerage credentials:

* :mod:`dashboard.backend.infrastructure.brokers.robinhood_oauth` -- the signed
  ``state`` blob. It carries the PKCE verifier, so a forgeable or replayable
  state is an account-takeover primitive, not a nuisance.
* :mod:`dashboard.backend.infrastructure.brokers.pending_links` -- the
  single-use slot that stops the unauthenticated OAuth callback from binding one
  person's brokerage tokens to another person's account.
* :mod:`dashboard.backend.domain.brokers.repository` -- encryption at rest and
  backend selection. The old suite only round-tripped ``upsert -> get``, which
  passes just as happily if ``_encrypt`` degrades to the identity function, so
  the encryption test here reads the raw SQLite column instead.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import sqlite3
import time

import pytest
from cryptography.fernet import Fernet

from dashboard.backend.db_url import describe_database_url
from dashboard.backend.domain.brokers import repository
from dashboard.backend.domain.brokers.repository import BrokerConnectionStore
from dashboard.backend.infrastructure.brokers import pending_links, robinhood_oauth

_STATE_SECRET = "unit-test-robinhood-state-secret"


# ---------------------------------------------------------------------------
# OAuth state (mint / parse / tamper / expiry)
# ---------------------------------------------------------------------------


@pytest.fixture
def state_secret(monkeypatch):
    """Pin the HMAC key so signing is deterministic across the whole test.

    Without this the module falls back to a random per-process key, which is
    the correct production behaviour but makes "same secret in, same secret
    out" an accident of ordering rather than a property under test.
    """
    monkeypatch.setenv("ROBINHOOD_OAUTH_STATE_SECRET", _STATE_SECRET)
    monkeypatch.setattr(robinhood_oauth, "_ephemeral_state_key", None)
    return _STATE_SECRET


def _mint(**overrides) -> str:
    kwargs = {
        "agent_id": "agent_test",
        "code_verifier": "verifier-value",
        "client_id": "client_test",
    }
    kwargs.update(overrides)
    user_id = kwargs.pop("user_id", 42)
    return robinhood_oauth.mint_oauth_state(user_id, **kwargs)


def test_pkce_pair_is_random_and_sha256_linked():
    verifier, challenge = robinhood_oauth.generate_pkce_pair()
    assert verifier
    assert challenge
    assert "=" not in challenge  # base64url, unpadded, per RFC 7636
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("utf-8")).digest())
        .decode("utf-8")
        .rstrip("=")
    )
    assert challenge == expected
    # A fixed verifier would make PKCE decorative.
    assert robinhood_oauth.generate_pkce_pair()[0] != verifier


def test_oauth_state_roundtrip(state_secret):
    verifier, _challenge = robinhood_oauth.generate_pkce_pair()
    state = _mint(code_verifier=verifier)
    payload = robinhood_oauth.parse_oauth_state(state)
    assert payload["uid"] == 42
    assert payload["aid"] == "agent_test"
    assert payload["cv"] == verifier
    assert payload["cid"] == "client_test"


def test_oauth_state_signature_tamper_is_rejected(state_secret):
    state = _mint()
    raw, sig = state.rsplit(".", 1)
    # Flip one hex digit of the HMAC.
    flipped = ("0" if sig[-1] != "0" else "1") + sig[:-1]
    with pytest.raises(ValueError) as exc:
        robinhood_oauth.parse_oauth_state(f"{raw}.{flipped}")
    assert str(exc.value) == "invalid_state"


def test_oauth_state_payload_tamper_is_rejected(state_secret):
    """Re-encoding the payload without re-signing must not parse.

    This is the forgery an attacker actually wants: swap ``uid`` for the
    victim's, keep the signature, and have the callback bind tokens to the
    wrong account.
    """
    state = _mint()
    raw, sig = state.rsplit(".", 1)
    padded = raw + "=" * (-len(raw) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
    payload["uid"] = 99  # someone else's account
    forged_raw = (
        base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    assert forged_raw != raw
    with pytest.raises(ValueError) as exc:
        robinhood_oauth.parse_oauth_state(f"{forged_raw}.{sig}")
    assert str(exc.value) == "invalid_state"


def test_oauth_state_without_signature_is_rejected(state_secret):
    raw = _mint().rsplit(".", 1)[0]
    with pytest.raises(ValueError) as exc:
        robinhood_oauth.parse_oauth_state(raw)
    assert str(exc.value) == "invalid_state"


def test_oauth_state_signed_with_a_different_secret_is_rejected(state_secret):
    """An attacker who guesses the *format* but not the key gets nowhere."""
    payload = {
        "uid": 42,
        "aid": None,
        "cv": "verifier-value",
        "cid": "client_test",
        "exp": int(time.time()) + 900,
        "n": "deadbeefdeadbeef",
    }
    raw = (
        base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    sig = hmac.new(b"not-the-real-secret", raw.encode(), hashlib.sha256).hexdigest()
    with pytest.raises(ValueError) as exc:
        robinhood_oauth.parse_oauth_state(f"{raw}.{sig}")
    assert str(exc.value) == "invalid_state"


def test_expired_oauth_state_is_rejected(state_secret, monkeypatch):
    monkeypatch.setattr(robinhood_oauth, "STATE_TTL_SECONDS", -60)
    state = _mint()
    with pytest.raises(ValueError) as exc:
        robinhood_oauth.parse_oauth_state(state)
    assert str(exc.value) == "state_expired"


def test_correctly_signed_state_missing_pkce_verifier_is_rejected(state_secret):
    """A validly signed state still has to carry ``cv``/``cid``.

    Dropping either would let the callback proceed with an empty verifier,
    which defeats PKCE while looking authentic.
    """
    payload = {
        "uid": 42,
        "cid": "client_test",
        "exp": int(time.time()) + 900,
        "n": "0011223344556677",
    }
    raw = (
        base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    sig = hmac.new(
        robinhood_oauth._state_signing_key(), raw.encode(), hashlib.sha256
    ).hexdigest()
    with pytest.raises(ValueError) as exc:
        robinhood_oauth.parse_oauth_state(f"{raw}.{sig}")
    assert str(exc.value) == "invalid_state"


def test_state_secret_is_ephemeral_reports_configuration(monkeypatch):
    monkeypatch.delenv("ROBINHOOD_OAUTH_STATE_SECRET", raising=False)
    monkeypatch.delenv("DISCORD_CLIENT_SECRET", raising=False)
    assert robinhood_oauth.state_secret_is_ephemeral() is True
    monkeypatch.setenv("ROBINHOOD_OAUTH_STATE_SECRET", _STATE_SECRET)
    assert robinhood_oauth.state_secret_is_ephemeral() is False


def test_ephemeral_state_key_is_random_not_a_committed_constant(monkeypatch):
    """The fallback key must be generated, never a literal in this repository."""
    monkeypatch.delenv("ROBINHOOD_OAUTH_STATE_SECRET", raising=False)
    monkeypatch.delenv("DISCORD_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(robinhood_oauth, "_ephemeral_state_key", None)
    first = robinhood_oauth._state_signing_key()
    assert len(first) >= 32
    monkeypatch.setattr(robinhood_oauth, "_ephemeral_state_key", None)
    second = robinhood_oauth._state_signing_key()
    assert second != first


# ---------------------------------------------------------------------------
# Pending link store (the account-linking CSRF fix)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_pending_links():
    pending_links.clear()
    yield
    pending_links.clear()


def _put(user_id: int = 5, **overrides) -> str:
    kwargs = {
        "user_id": user_id,
        "agent_id": "agent_x",
        "tokens": {"access_token": "tok", "refresh_token": "ref"},
        "client_id": "cid",
    }
    kwargs.update(overrides)
    return pending_links.put(**kwargs)


def test_pending_link_pop_is_single_use():
    code = _put(user_id=5)
    record = pending_links.pop(code)
    assert record is not None
    assert record["user_id"] == 5
    assert record["agent_id"] == "agent_x"
    assert record["tokens"]["access_token"] == "tok"
    # A redeemed code is spent: replaying the callback cannot re-bind tokens.
    # pop() mutates, so it is called outside the assert -- under `python -O`
    # asserts are stripped and an inlined pop would silently not happen.
    replayed = pending_links.pop(code)
    assert replayed is None


def test_pending_link_unknown_or_empty_code_returns_none():
    unknown = pending_links.pop("never-issued")
    empty = pending_links.pop("")
    assert unknown is None
    assert empty is None


def test_expired_pending_link_returns_none(monkeypatch):
    monkeypatch.setattr(pending_links, "PENDING_TTL_SECONDS", -1)
    code = _put()
    expired = pending_links.pop(code)
    assert expired is None


def test_expired_records_are_purged_by_later_writes(monkeypatch):
    monkeypatch.setattr(pending_links, "PENDING_TTL_SECONDS", -1)
    stale = _put(user_id=1)
    monkeypatch.setattr(pending_links, "PENDING_TTL_SECONDS", 600)
    fresh = _put(user_id=2)
    survivor = pending_links.pop(fresh)
    assert stale not in pending_links._pending
    assert survivor is not None


def test_pending_link_store_evicts_instead_of_growing_unbounded(monkeypatch):
    """A replayed callback must not be able to grow the store without bound."""
    monkeypatch.setattr(pending_links, "MAX_PENDING_LINKS", 3)
    codes = [_put(user_id=i) for i in range(12)]
    stored = len(pending_links._pending)
    # Oldest evicted first; newest survives.
    oldest = pending_links.pop(codes[0])
    newest = pending_links.pop(codes[-1])
    assert stored <= 3
    assert oldest is None
    assert newest is not None


def test_pending_link_copies_the_token_dict():
    """The stored record must not alias the caller's dict."""
    tokens = {"access_token": "tok"}
    code = _put(tokens=tokens)
    tokens["access_token"] = "mutated-after-put"
    record = pending_links.pop(code)
    assert record["tokens"]["access_token"] == "tok"


def test_pending_link_codes_are_unguessable_and_unique():
    codes = {_put() for _ in range(25)}
    assert len(codes) == 25
    assert all(len(code) >= 32 for code in codes)


# ---------------------------------------------------------------------------
# Broker token store (encryption at rest + backend selection)
# ---------------------------------------------------------------------------

_ACCESS = "rh-access-token-PLAINTEXT-CANARY"
_REFRESH = "rh-refresh-token-PLAINTEXT-CANARY"


@pytest.fixture
def broker_db(tmp_path, monkeypatch):
    """A throwaway SQLite broker store with a real, per-test Fernet key.

    ``_fernet_instance`` is a module global cached on first use, so it is reset
    here (and restored by monkeypatch afterwards) -- otherwise a key set by one
    test would keep encrypting for every later test in the process.
    """
    monkeypatch.setenv(repository._KEY_ENV_VAR, Fernet.generate_key().decode())
    monkeypatch.setattr(repository, "_fernet_instance", None)
    db_file = tmp_path / "brokers.db"
    return BrokerConnectionStore(db_file), db_file


def _raw_row(db_file, user_id: int = 7):
    conn = sqlite3.connect(str(db_file))
    try:
        return conn.execute(
            "SELECT access_token_enc, refresh_token_enc FROM broker_connections "
            "WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    finally:
        conn.close()


def test_tokens_are_ciphertext_on_disk(broker_db):
    """Read the raw column, not the accessor.

    A round-trip assertion (``upsert`` then ``get_tokens``) still passes if
    ``_encrypt`` becomes the identity function, which is exactly the regression
    that would park live refresh tokens in the database as plaintext.
    """
    store, db_file = broker_db
    store.upsert_tokens(7, access_token=_ACCESS, refresh_token=_REFRESH, client_id="cid")

    row = _raw_row(db_file)
    assert row is not None
    raw_access, raw_refresh = row
    assert _ACCESS not in raw_access
    assert _REFRESH not in raw_refresh
    assert raw_access != _ACCESS
    assert raw_refresh != _REFRESH
    # Fernet tokens are urlsafe-base64 of a 0x80 version byte: "gAAAAA...".
    assert raw_access.startswith("gAAAAA")
    assert raw_refresh.startswith("gAAAAA")

    tokens = store.get_tokens(7)
    assert tokens["access_token"] == _ACCESS
    assert tokens["refresh_token"] == _REFRESH


def test_public_view_never_exposes_tokens(broker_db):
    store, _db_file = broker_db
    public = store.upsert_tokens(
        7, access_token=_ACCESS, refresh_token=_REFRESH, client_id="cid"
    )
    assert public["connected"] is True
    assert public["client_id"] == "cid"
    serialized = json.dumps(public, default=str)
    assert _ACCESS not in serialized
    assert _REFRESH not in serialized
    assert "access_token" not in public
    assert "refresh_token" not in public


def test_delete_removes_the_connection(broker_db):
    store, _db_file = broker_db
    store.upsert_tokens(7, access_token=_ACCESS, refresh_token=_REFRESH, client_id="cid")
    # Plain statement, never `assert store.delete(7) is True`: an assert-wrapped
    # side effect vanishes under `python -O` (open CodeQL alert on the old file).
    store.delete(7)
    assert store.get_public(7) is None
    assert store.get_tokens(7) is None


def test_upsert_preserves_client_id_on_relink(broker_db):
    store, _db_file = broker_db
    store.upsert_tokens(7, access_token=_ACCESS, refresh_token=_REFRESH, client_id="cid")
    store.upsert_tokens(7, access_token="rotated", refresh_token=None)
    tokens = store.get_tokens(7)
    assert tokens["access_token"] == "rotated"
    assert tokens["client_id"] == "cid"


def test_update_tokens_rewrites_ciphertext(broker_db):
    store, db_file = broker_db
    store.upsert_tokens(7, access_token=_ACCESS, refresh_token=_REFRESH, client_id="cid")
    before = _raw_row(db_file)[0]
    store.update_tokens(7, access_token="rotated-access", token_expires_at="2099-01-01T00:00:00+00:00")
    after = _raw_row(db_file)[0]
    assert after != before
    assert "rotated-access" not in after
    tokens = store.get_tokens(7)
    assert tokens["access_token"] == "rotated-access"
    # COALESCE keeps the previous refresh token when a rotation omits it.
    assert tokens["refresh_token"] == _REFRESH


def test_get_fernet_refuses_to_run_without_a_key(monkeypatch):
    """Fails closed. The old fallback derived a key from a literal published in
    this public repository, which is plaintext storage wearing a costume."""
    monkeypatch.setattr(repository, "_fernet_instance", None)
    monkeypatch.delenv(repository._KEY_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError) as exc:
        repository._get_fernet()
    message = str(exc.value)
    assert repository._KEY_ENV_VAR in message
    assert "Fernet.generate_key" in message


def test_get_fernet_rejects_a_malformed_key_without_echoing_it(monkeypatch):
    bad_key = "not-a-valid-fernet-key-CANARY"
    monkeypatch.setattr(repository, "_fernet_instance", None)
    monkeypatch.setenv(repository._KEY_ENV_VAR, bad_key)
    with pytest.raises(RuntimeError) as exc:
        repository._get_fernet()
    message = str(exc.value)
    assert repository._KEY_ENV_VAR in message
    # The message reaches logs and, via a 500, potentially a response body.
    assert bad_key not in message


def test_encrypt_requires_a_key(monkeypatch):
    monkeypatch.setattr(repository, "_fernet_instance", None)
    monkeypatch.delenv(repository._KEY_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError):
        repository._encrypt("secret")


@pytest.mark.parametrize("env_var", ["USERS_DATABASE_URL", "CONTENT_DATABASE_URL"])
def test_build_user_store_returns_the_postgres_twin(monkeypatch, capsys, env_var):
    """Pure wiring test: the factory must *return* the twin it announces.

    The previous factory printed ``postgres`` and then returned the SQLite
    store, so a deployment believed its brokerage tokens were durable while
    they were being written to a file that resets on every deploy.
    """
    from dashboard.backend.domain.brokers import repository_postgres

    constructed = {}

    def _fake_init(self, database_url):
        constructed["url"] = database_url
        self.database_url = database_url

    monkeypatch.setattr(
        repository_postgres.BrokerConnectionStorePostgres, "__init__", _fake_init
    )
    monkeypatch.delenv("USERS_DATABASE_URL", raising=False)
    monkeypatch.delenv("CONTENT_DATABASE_URL", raising=False)
    url = "postgresql://atl_user:hunter2@db.example.com:5432/atl"
    monkeypatch.setenv(env_var, url)

    store = repository._build_user_store()

    assert isinstance(store, repository_postgres.BrokerConnectionStorePostgres)
    assert not isinstance(store, BrokerConnectionStore)
    assert constructed["url"] == url

    out = capsys.readouterr().out.strip()
    # Assert the whole line, not substrings of a URL: it pins the exact contract
    # (name the host/db so a typo'd or staging target is visible) and cannot pass
    # on an accidental partial match.
    expected_target = describe_database_url(url)
    assert out == f"broker_connections backend: postgres ({expected_target})"
    # ...and the credentials never appear.
    assert "hunter2" not in out
    assert "atl_user" not in out


def test_build_user_store_falls_back_to_sqlite(monkeypatch, capsys):
    monkeypatch.delenv("USERS_DATABASE_URL", raising=False)
    monkeypatch.delenv("CONTENT_DATABASE_URL", raising=False)
    store = repository._build_user_store()
    assert isinstance(store, BrokerConnectionStore)
    assert "broker_connections backend: sqlite" in capsys.readouterr().out
