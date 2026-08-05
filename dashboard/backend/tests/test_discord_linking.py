"""Discord account linking + bot-owned agents API."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
import dashboard.backend.users as users_module
from dashboard.backend.api import discord_oauth
from dashboard.backend.tests.auth_cookies_helpers import _cookie_session_token


@pytest.fixture
def temp_user_store(tmp_path, monkeypatch):
    store = users_module.UserStore(db_path=tmp_path / "discord_auth.db")

    # Both api/auth.py and the discord router resolve users_module.user_store at
    # call time (issue #185), so this single patch redirects every route below.
    monkeypatch.setattr(users_module, "user_store", store)
    return store


@pytest.fixture
def client(temp_user_store, monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_API_SECRET", "test-bot-secret")
    monkeypatch.setenv("DISCORD_CLIENT_ID", "client-id")
    monkeypatch.setenv("DISCORD_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv(
        "DISCORD_REDIRECT_URI",
        "http://testserver/api/auth/discord/callback",
    )
    monkeypatch.setenv("DISCORD_GUILD_CHANNEL_URL", "https://discord.gg/test-channel")
    return TestClient(app)


def _signup(client, email="alice@example.com", name="Alice"):
    resp = client.post(
        "/api/auth/signup",
        json={"email": email, "display_name": name, "password": "securepass1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "token" not in body
    body["token"] = _cookie_session_token(client)
    return body


def test_public_user_includes_discord_link_fields(client):
    data = _signup(client)
    user = data["user"]
    assert user["discord_linked"] is False
    assert user["discord_user_id"] is None

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {data['token']}"})
    assert me.status_code == 200
    assert me.json()["user"]["discord_linked"] is False


def test_link_discord_user_and_conflict(temp_user_store):
    a = temp_user_store.create_user("a@example.com", "A", "securepass1")
    b = temp_user_store.create_user("b@example.com", "B", "securepass1")

    linked = temp_user_store.link_discord_user(a["id"], "111")
    assert linked["discord_linked"] is True
    assert linked["discord_user_id"] == "111"

    # Idempotent re-link for same user
    again = temp_user_store.link_discord_user(a["id"], "111")
    assert again["discord_user_id"] == "111"

    with pytest.raises(ValueError, match="discord_already_linked"):
        temp_user_store.link_discord_user(b["id"], "111")


def test_discord_start_requires_auth(client):
    assert client.post("/api/auth/discord/start").status_code == 401


def test_discord_start_returns_authorize_url(client):
    data = _signup(client)
    resp = client.post(
        "/api/auth/discord/start",
        headers={"Authorization": f"Bearer {data['token']}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["already_linked"] is False
    assert "client_id=client-id" in body["authorize_url"]
    assert "scope=identify" in body["authorize_url"]
    assert body["discord_url"] == "https://discord.gg/test-channel"


def test_discord_start_already_linked(client, temp_user_store):
    data = _signup(client)
    temp_user_store.link_discord_user(data["user"]["id"], "999888")
    resp = client.post(
        "/api/auth/discord/start",
        headers={"Authorization": f"Bearer {data['token']}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["already_linked"] is True
    assert body["authorize_url"] is None
    assert body["user"]["discord_linked"] is True


def test_discord_oauth_callback_links_user(client, temp_user_store, monkeypatch):
    data = _signup(client)
    user_id = data["user"]["id"]
    state = discord_oauth.mint_oauth_state(user_id)

    monkeypatch.setattr(
        discord_oauth,
        "exchange_code_for_access_token",
        lambda code: "access-token",
    )
    monkeypatch.setattr(
        discord_oauth,
        "fetch_discord_user",
        lambda token: {"id": "discord-42", "username": "demo"},
    )

    resp = client.get(
        "/api/auth/discord/callback",
        params={"code": "abc", "state": state},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert "discord=linked" in resp.headers["location"]

    linked = temp_user_store.get_user_by_discord_id("discord-42")
    assert linked is not None
    assert linked["id"] == user_id


def test_discord_agents_requires_bot_secret(client):
    resp = client.get(
        "/api/v1/discord/agents",
        headers={"X-Discord-User-Id": "123"},
    )
    assert resp.status_code == 401


def test_discord_agents_not_linked(client):
    resp = client.get(
        "/api/v1/discord/agents",
        headers={
            "X-Discord-Bot-Secret": "test-bot-secret",
            "X-Discord-User-Id": "no-such-user",
        },
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["code"] == "discord_not_linked"


def test_discord_agents_returns_only_owner_agents(client, temp_user_store):
    alice = _signup(client, email="owner@example.com", name="Owner")
    bob = _signup(client, email="other@example.com", name="Other")

    temp_user_store.link_discord_user(alice["user"]["id"], "discord-alice")
    temp_user_store.link_discord_user(bob["user"]["id"], "discord-bob")

    browser = str(uuid.uuid4())
    created = client.post(
        "/api/v1/agents",
        json={
            "name": "Alice Bot",
            "agent_type": "builtin",
            "model_name": "anthropic/claude-haiku-4-5",
        },
        headers={
            "Authorization": f"Bearer {alice['token']}",
            "X-Session-Id": browser,
            "X-Browser-Id": browser,
        },
    )
    assert created.status_code == 200
    alice_agent_id = created.json()["agent"]["agent_id"]

    bob_browser = str(uuid.uuid4())
    client.post(
        "/api/v1/agents",
        json={"name": "Bob Bot", "agent_type": "builtin"},
        headers={
            "Authorization": f"Bearer {bob['token']}",
            "X-Session-Id": bob_browser,
            "X-Browser-Id": bob_browser,
        },
    )

    listing = client.get(
        "/api/v1/discord/agents",
        headers={
            "X-Discord-Bot-Secret": "test-bot-secret",
            "X-Discord-User-Id": "discord-alice",
        },
    )
    assert listing.status_code == 200
    body = listing.json()
    assert body["user"]["id"] == alice["user"]["id"]
    ids = {a["agent_id"] for a in body["agents"]}
    assert alice_agent_id in ids
    assert all(a["name"] != "Bob Bot" for a in body["agents"])


def test_oauth_state_roundtrip():
    state = discord_oauth.mint_oauth_state(7)
    assert discord_oauth.parse_oauth_state(state) == 7
    with pytest.raises(ValueError):
        discord_oauth.parse_oauth_state(state + "tampered")
