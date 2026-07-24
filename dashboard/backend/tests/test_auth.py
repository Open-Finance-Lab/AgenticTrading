"""
Auth API tests using a temporary SQLite database.
"""

import base64
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.users import UserStore


@pytest.fixture
def temp_user_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = UserStore(db_path=Path(tmpdir) / "auth_test.db")
        yield store


@pytest.fixture
def client(temp_user_store, monkeypatch):
    from dashboard.backend import users

    monkeypatch.setattr(users, "user_store", temp_user_store)
    return TestClient(app)


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_signup_login_me_logout_flow(client):
    signup = client.post(
        "/api/auth/signup",
        json={
            "email": "alice@example.com",
            "display_name": "Alice",
            "password": "securepass1",
        },
    )
    assert signup.status_code == 200
    signup_data = signup.json()
    assert signup_data["user"]["email"] == "alice@example.com"
    assert signup_data["user"]["display_name"] == "Alice"
    assert signup_data["user"]["role"] == "user"
    assert "password_hash" not in signup_data["user"]
    assert signup_data["token"]

    duplicate = client.post(
        "/api/auth/signup",
        json={
            "email": "alice@example.com",
            "display_name": "Alice 2",
            "password": "securepass1",
        },
    )
    assert duplicate.status_code == 409

    login = client.post(
        "/api/auth/login",
        json={"email": "alice@example.com", "password": "securepass1"},
    )
    assert login.status_code == 200
    token = login.json()["token"]

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "alice@example.com"

    logout = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout.status_code == 200

    me_after = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_after.status_code == 401


def test_me_requires_auth(client):
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_login_invalid_password(client):
    client.post(
        "/api/auth/signup",
        json={
            "email": "bob@example.com",
            "display_name": "Bob",
            "password": "securepass1",
        },
    )
    response = client.post(
        "/api/auth/login",
        json={"email": "bob@example.com", "password": "wrong-password"},
    )
    assert response.status_code == 401


def test_signup_rejects_common_password(client):
    response = client.post(
        "/api/auth/signup",
        json={
            "email": "carol@example.com",
            "display_name": "Carol",
            "password": "password1",
        },
    )
    assert response.status_code == 400
    assert "too common" in response.json()["detail"]


def test_signup_rejects_short_password_with_readable_error(client):
    response = client.post(
        "/api/auth/signup",
        json={
            "email": "carol@example.com",
            "display_name": "Carol",
            "password": "short",
        },
    )
    assert response.status_code == 400
    assert "at least 8" in response.json()["detail"]


def test_signup_rejects_password_containing_email_name(client):
    response = client.post(
        "/api/auth/signup",
        json={
            "email": "carolyn@example.com",
            "display_name": "Carol",
            "password": "carolyn-trades-2026",
        },
    )
    assert response.status_code == 400
    assert "email" in response.json()["detail"]


def _signup_and_token(client, email="dave@example.com", password="orig-sturdy-pw-1"):
    response = client.post(
        "/api/auth/signup",
        json={"email": email, "display_name": "Dave", "password": password},
    )
    assert response.status_code == 200
    return response.json()["token"]


def test_change_password_happy_path(client):
    token = _signup_and_token(client)
    response = client.post(
        "/api/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_password": "new-sturdy-pw-2"},
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Old password no longer works; new one does.
    old_login = client.post(
        "/api/auth/login",
        json={"email": "dave@example.com", "password": "orig-sturdy-pw-1"},
    )
    assert old_login.status_code == 401
    new_login = client.post(
        "/api/auth/login",
        json={"email": "dave@example.com", "password": "new-sturdy-pw-2"},
    )
    assert new_login.status_code == 200


def test_change_password_requires_auth(client):
    response = client.post(
        "/api/auth/change-password",
        json={"current_password": "x-not-relevant", "new_password": "new-sturdy-pw-2"},
    )
    assert response.status_code == 401


def test_change_password_wrong_current(client):
    token = _signup_and_token(client, email="erin@example.com")
    response = client.post(
        "/api/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "wrong-guess-1", "new_password": "new-sturdy-pw-2"},
    )
    assert response.status_code == 400
    assert "Current password is incorrect" in response.json()["detail"]


def test_change_password_rejects_weak_new_password(client):
    token = _signup_and_token(client, email="frank@example.com")
    response = client.post(
        "/api/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_password": "password1"},
    )
    assert response.status_code == 400
    assert "too common" in response.json()["detail"]
    # And the old password still works (nothing was changed).
    login = client.post(
        "/api/auth/login",
        json={"email": "frank@example.com", "password": "orig-sturdy-pw-1"},
    )
    assert login.status_code == 200


def test_change_password_invalidates_other_sessions_keeps_current(client):
    token_a = _signup_and_token(client, email="gina@example.com")
    token_b = client.post(
        "/api/auth/login",
        json={"email": "gina@example.com", "password": "orig-sturdy-pw-1"},
    ).json()["token"]

    response = client.post(
        "/api/auth/change-password",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"current_password": "orig-sturdy-pw-1", "new_password": "new-sturdy-pw-2"},
    )
    assert response.status_code == 200

    # The session that changed the password survives; the other is revoked.
    me_a = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token_a}"})
    assert me_a.status_code == 200
    me_b = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token_b}"})
    assert me_b.status_code == 401


def test_change_password_revocation_failure_still_succeeds(client, monkeypatch, capsys):
    # The password write and the other-session revocation are two separate
    # transactions. If revocation raises, the (already-durable) password change
    # must still report success rather than a misleading 500. Patch at the CLASS
    # level so it fails regardless of which UserStore instance the route resolves
    # (the `client` fixture only reassigns users_module.user_store; api/auth.py may
    # still hold the original singleton binding). `UserStore` is already imported.
    token = _signup_and_token(client, email="quinn@example.com")

    def _boom(*args, **kwargs):
        raise RuntimeError("session store unavailable")

    monkeypatch.setattr(UserStore, "delete_other_sessions", _boom)

    response = client.post(
        "/api/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_password": "new-sturdy-pw-2"},
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # The new password is live despite the revocation failure (change was durable).
    new_login = client.post(
        "/api/auth/login",
        json={"email": "quinn@example.com", "password": "new-sturdy-pw-2"},
    )
    assert new_login.status_code == 200

    # The failure is surfaced via print() (logger output is invisible in prod), not
    # swallowed silently. Assert on capsys, never caplog.
    assert "revocation failed" in capsys.readouterr().out


# JPEG magic bytes + padding. The server validates magic + base64 + size,
# not full image decode (no image library), so this is a sufficient payload.
_TINY_JPEG = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 32).decode("ascii")


def _avatar_uri(payload_b64=_TINY_JPEG, mime="image/jpeg"):
    return f"data:{mime};base64,{payload_b64}"


def test_avatar_put_and_delete_flow(client):
    token = _signup_and_token(client, email="hana@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    put = client.put("/api/auth/avatar", headers=headers, json={"avatar": _avatar_uri()})
    assert put.status_code == 200
    assert put.json()["user"]["avatar"] == _avatar_uri()

    me = client.get("/api/auth/me", headers=headers)
    assert me.json()["user"]["avatar"] == _avatar_uri()

    delete = client.delete("/api/auth/avatar", headers=headers)
    assert delete.status_code == 200
    assert delete.json()["user"]["avatar"] is None


def test_avatar_replace_overwrites_previous(client):
    token = _signup_and_token(client, email="nina@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    first = _avatar_uri()
    # A different but equally valid JPEG payload, so an UPDATE that silently kept the
    # old value (or wrote nothing) fails here instead of passing on an identical URI.
    second = _avatar_uri(
        payload_b64=base64.b64encode(b"\xff\xd8\xff" + b"\x11" * 48).decode("ascii")
    )
    assert first != second

    put_first = client.put("/api/auth/avatar", headers=headers, json={"avatar": first})
    assert put_first.status_code == 200

    put_second = client.put("/api/auth/avatar", headers=headers, json={"avatar": second})
    assert put_second.status_code == 200
    assert put_second.json()["user"]["avatar"] == second

    # Durable, not merely echoed back by the write response.
    me = client.get("/api/auth/me", headers=headers)
    assert me.json()["user"]["avatar"] == second


def test_avatar_requires_auth(client):
    put = client.put("/api/auth/avatar", json={"avatar": _avatar_uri()})
    assert put.status_code == 401
    delete = client.delete("/api/auth/avatar")
    assert delete.status_code == 401


def test_avatar_rejects_unsupported_mime(client):
    token = _signup_and_token(client, email="iris@example.com")
    response = client.put(
        "/api/auth/avatar",
        headers={"Authorization": f"Bearer {token}"},
        json={"avatar": _avatar_uri(mime="image/svg+xml")},
    )
    assert response.status_code == 400


def test_avatar_rejects_magic_number_mismatch(client):
    token = _signup_and_token(client, email="jack@example.com")
    # Declared PNG, actual bytes JPEG.
    response = client.put(
        "/api/auth/avatar",
        headers={"Authorization": f"Bearer {token}"},
        json={"avatar": _avatar_uri(mime="image/png")},
    )
    assert response.status_code == 400
    assert "match" in response.json()["detail"]


def test_avatar_rejects_invalid_base64(client):
    token = _signup_and_token(client, email="kate@example.com")
    response = client.put(
        "/api/auth/avatar",
        headers={"Authorization": f"Bearer {token}"},
        json={"avatar": "data:image/jpeg;base64,!!!not-base64!!!"},
    )
    assert response.status_code == 400


def test_avatar_rejects_oversize(client):
    token = _signup_and_token(client, email="liam@example.com")
    # Valid JPEG magic, padded past 100 KB.
    big = base64.b64encode(
        b"\xff\xd8\xff" + b"\x00" * (101 * 1024)
    ).decode("ascii")
    response = client.put(
        "/api/auth/avatar",
        headers={"Authorization": f"Bearer {token}"},
        json={"avatar": _avatar_uri(payload_b64=big)},
    )
    assert response.status_code == 400
    assert "100 KB" in response.json()["detail"]


def test_signup_response_includes_avatar_field(client):
    response = client.post(
        "/api/auth/signup",
        json={"email": "mia@example.com", "display_name": "Mia", "password": "sturdy-enough-9"},
    )
    assert response.status_code == 200
    assert response.json()["user"]["avatar"] is None
