"""Robinhood OAuth state + broker store tests."""

from dashboard.backend.api import robinhood_oauth
from dashboard.backend.domain.brokers.repository import BrokerConnectionStore


def test_robinhood_oauth_state_roundtrip():
    verifier, challenge = robinhood_oauth.generate_pkce_pair()
    assert verifier
    assert challenge
    state = robinhood_oauth.mint_oauth_state(
        42,
        agent_id="agent_test",
        code_verifier=verifier,
        client_id="client_test",
    )
    payload = robinhood_oauth.parse_oauth_state(state)
    assert payload["uid"] == 42
    assert payload["aid"] == "agent_test"
    assert payload["cv"] == verifier
    assert payload["cid"] == "client_test"


def test_broker_store_encrypts_tokens(tmp_path):
    store = BrokerConnectionStore(tmp_path / "brokers.db")
    public = store.upsert_tokens(
        7,
        access_token="access-secret",
        refresh_token="refresh-secret",
        client_id="cid",
    )
    assert public["connected"] is True
    tokens = store.get_tokens(7)
    assert tokens["access_token"] == "access-secret"
    assert tokens["refresh_token"] == "refresh-secret"
    assert store.delete(7) is True
    assert store.get_public(7) is None
