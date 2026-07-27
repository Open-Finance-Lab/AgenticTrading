"""
Auth API tests using a temporary SQLite database.
"""

import base64
import tempfile
from datetime import timedelta
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
    # level so it fails for any UserStore instance, including the fixture's.
    # `UserStore` is already imported.
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


def test_auth_routes_resolve_the_store_at_call_time(temp_user_store, monkeypatch):
    """Issue #185: api/auth.py must not bind the user_store singleton at import.

    Patching only dashboard.backend.users must be enough to redirect every auth
    route. When auth.py holds its own import-time binding, this signup lands in
    the process-wide store and the temp store below stays empty -- silently, with
    the test still green, which is exactly how #185 survived this long.
    """
    from dashboard.backend import users as users_module

    monkeypatch.setattr(users_module, "user_store", temp_user_store)
    client = TestClient(app)

    response = client.post(
        "/api/auth/signup",
        json={
            "email": "callsite@example.com",
            "display_name": "Callsite",
            "password": "securepass1",
        },
    )
    assert response.status_code == 200
    assert temp_user_store.get_user_by_email("callsite@example.com") is not None


def test_update_display_name_happy_path(client):
    token = _signup_and_token(client, email="name@example.com")

    response = client.put(
        "/api/auth/display-name",
        headers={"Authorization": f"Bearer {token}"},
        json={"display_name": "New Name"},
    )

    assert response.status_code == 200
    assert response.json()["user"]["display_name"] == "New Name"
    # And it is durable, not just echoed back.
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["user"]["display_name"] == "New Name"


def test_update_display_name_strips_whitespace(client):
    token = _signup_and_token(client, email="trim@example.com")

    response = client.put(
        "/api/auth/display-name",
        headers={"Authorization": f"Bearer {token}"},
        json={"display_name": "  Trimmed  "},
    )

    assert response.status_code == 200
    assert response.json()["user"]["display_name"] == "Trimmed"


def test_update_display_name_rejects_whitespace_only(client):
    # Field(min_length=1) passes on "   " because pydantic measures the raw
    # string. Storing it would repeat issue #167 on a second surface.
    token = _signup_and_token(client, email="blank@example.com")

    response = client.put(
        "/api/auth/display-name",
        headers={"Authorization": f"Bearer {token}"},
        json={"display_name": "     "},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_update_display_name_requires_auth(client):
    response = client.put("/api/auth/display-name", json={"display_name": "Nope"})
    assert response.status_code == 401


def test_update_display_name_rejects_overlong_value(client):
    token = _signup_and_token(client, email="long@example.com")

    response = client.put(
        "/api/auth/display-name",
        headers={"Authorization": f"Bearer {token}"},
        json={"display_name": "x" * 101},
    )

    assert response.status_code == 422


class _Outbox(list):
    """Captured messages, plus a switch to make sending start failing.

    Subclasses list rather than pairing a bare list with a flag: a plain list
    rejects attribute assignment (`outbox.ok = False` raises AttributeError),
    and the tests read better asserting on the outbox directly.
    """

    ok = True

    def fail_sends(self):
        self.ok = False

    def resume_sends(self):
        self.ok = True


@pytest.fixture
def sent_emails(monkeypatch):
    """Capture outbound mail instead of sending it; control success/failure.

    Patches the attribute on the sender module, which is exactly how
    api/auth.py reaches it (`from ...email import sender as email_sender`,
    then `email_sender.send_email(...)`) -- one place to patch, and patching
    it works.
    """
    from dashboard.backend.infrastructure.email import sender as email_sender

    outbox = _Outbox()

    async def _fake_send(to, subject, text_body):
        outbox.append({"to": to, "subject": subject, "body": text_body})
        return outbox.ok

    monkeypatch.setattr(email_sender, "send_email", _fake_send)
    return outbox


def _code_from(email_body):
    """Pull the 6-character code out of a captured message body."""
    import re

    from dashboard.backend.verification_codes import CODE_ALPHABET

    match = re.search(rf"code is: ([{CODE_ALPHABET}]{{6}})", email_body)
    assert match, f"no code found in: {email_body!r}"
    return match.group(1)


def test_email_change_request_mails_the_original_address(client, sent_emails):
    # Not "orig@example.com": its local part "orig" is a substring of the
    # default signup password ("orig-sturdy-pw-1"), which password_policy's
    # email-name blocklist check would then reject at signup.
    token = _signup_and_token(client, email="before@example.com")

    response = client.post(
        "/api/auth/email-change",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"},
    )

    assert response.status_code == 200
    assert response.json() == {"stage": "old", "new_email": "fresh@example.com"}
    assert len(sent_emails) == 1
    # The authorizing code goes to the address the user already controls.
    assert sent_emails[0]["to"] == "before@example.com"
    assert "fresh@example.com" in sent_emails[0]["body"]


def test_email_change_request_rejects_a_wrong_password(client, sent_emails):
    token = _signup_and_token(client, email="wrongpw@example.com")

    response = client.post(
        "/api/auth/email-change",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "not-the-password", "new_email": "fresh@example.com"},
    )

    assert response.status_code == 400
    assert "Current password is incorrect" in response.json()["detail"]
    assert sent_emails == []


def test_email_change_request_rejects_the_current_address(client, sent_emails):
    token = _signup_and_token(client, email="same@example.com")

    response = client.post(
        "/api/auth/email-change",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_email": "SAME@example.com"},
    )

    assert response.status_code == 400
    assert sent_emails == []


def test_email_change_request_rejects_a_registered_address(client, sent_emails):
    _signup_and_token(client, email="taken@example.com")
    token = _signup_and_token(client, email="mover@example.com")

    response = client.post(
        "/api/auth/email-change",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_email": "taken@example.com"},
    )

    assert response.status_code == 409
    assert sent_emails == []


def test_email_change_request_is_cooldown_limited(client, sent_emails):
    token = _signup_and_token(client, email="fast@example.com")
    body = {"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"}
    headers = {"Authorization": f"Bearer {token}"}

    assert client.post("/api/auth/email-change", headers=headers, json=body).status_code == 200
    second = client.post("/api/auth/email-change", headers=headers, json=body)

    assert second.status_code == 429
    assert second.headers["Retry-After"] == "60"
    assert len(sent_emails) == 1


def test_email_change_cooldown_survives_cancel_and_resend(client, sent_emails):
    # The bug this guards against: DELETE needs only a session, not the
    # password, so without a fix a caller who knows the password could loop
    # request -> cancel -> request with the cooldown never enforced --
    # mail-bombing the account and burning the shared Brevo daily quota.
    token = _signup_and_token(client, email="bounce@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    body = {"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"}

    assert client.post("/api/auth/email-change", headers=headers, json=body).status_code == 200
    assert client.delete("/api/auth/email-change", headers=headers).status_code == 200

    second = client.post("/api/auth/email-change", headers=headers, json=body)

    assert second.status_code == 429
    assert len(sent_emails) == 1


def test_email_change_request_checks_password_before_cooldown(client, sent_emails):
    # A mistyped password must not burn the one-per-minute allowance.
    token = _signup_and_token(client, email="order@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    client.post(
        "/api/auth/email-change",
        headers=headers,
        json={"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"},
    )
    response = client.post(
        "/api/auth/email-change",
        headers=headers,
        json={"current_password": "wrong", "new_email": "other@example.com"},
    )

    assert response.status_code == 400  # not 429


def test_email_change_request_503s_when_mail_fails_and_persists_nothing(
    client, sent_emails
):
    # Send before persist: a failed send must not burn the cooldown for a code
    # that does not exist.
    token = _signup_and_token(client, email="nomail@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    sent_emails.fail_sends()

    response = client.post(
        "/api/auth/email-change",
        headers=headers,
        json={"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"},
    )

    assert response.status_code == 503
    assert client.get("/api/auth/email-change", headers=headers).json()["pending"] is False


def test_email_change_request_503s_when_the_provider_is_unconfigured(client, capsys):
    # No sent_emails fixture here: exercise the real sender with no credentials.
    token = _signup_and_token(client, email="unconfigured@example.com")

    response = client.post(
        "/api/auth/email-change",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"},
    )

    assert response.status_code == 503
    # Fail-VISIBLE: an operator can tell "not configured" from "provider down".
    # capsys, not caplog -- logger output is invisible in the deployment.
    assert "ERROR" in capsys.readouterr().out


def test_email_change_status_reports_the_pending_request(client, sent_emails):
    token = _signup_and_token(client, email="status@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    assert client.get("/api/auth/email-change", headers=headers).json() == {
        "pending": False,
        "stage": None,
        "new_email": None,
        "expires_at": None,
    }

    client.post(
        "/api/auth/email-change",
        headers=headers,
        json={"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"},
    )
    pending = client.get("/api/auth/email-change", headers=headers).json()

    assert pending["pending"] is True
    assert pending["stage"] == "old"
    assert pending["new_email"] == "fresh@example.com"
    assert pending["expires_at"]


def test_email_change_cancel_clears_the_request(client, sent_emails):
    token = _signup_and_token(client, email="cancel@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    client.post(
        "/api/auth/email-change",
        headers=headers,
        json={"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"},
    )

    assert client.delete("/api/auth/email-change", headers=headers).status_code == 200
    assert client.get("/api/auth/email-change", headers=headers).json()["pending"] is False


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/api/auth/email-change"),
        ("get", "/api/auth/email-change"),
        ("delete", "/api/auth/email-change"),
        ("post", "/api/auth/email-change/verify"),
    ],
)
def test_email_change_routes_require_auth(client, method, path):
    # GET/DELETE on this httpx/starlette pairing reject a `json=` kwarg outright
    # (TypeError, not a response) -- only POST carries a body here.
    kwargs = {"json": {}} if method == "post" else {}
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 401


def _start_email_change(client, token, new_email="fresh@example.com"):
    response = client.post(
        "/api/auth/email-change",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_email": new_email},
    )
    assert response.status_code == 200, response.text
    return response


def test_email_change_full_two_stage_happy_path(client, sent_emails):
    token = _signup_and_token(client, email="two@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    _start_email_change(client, token)

    first_code = _code_from(sent_emails[0]["body"])
    stage_two = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": first_code}
    )

    assert stage_two.status_code == 200
    assert stage_two.json() == {"stage": "new", "new_email": "fresh@example.com"}
    # The second code goes to the NEW address -- that is the reachability proof.
    assert len(sent_emails) == 2
    assert sent_emails[1]["to"] == "fresh@example.com"

    second_code = _code_from(sent_emails[1]["body"])
    done = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": second_code}
    )

    assert done.status_code == 200
    assert done.json()["status"] == "ok"
    assert done.json()["user"]["email"] == "fresh@example.com"
    # Durable, and the old address no longer signs in.
    assert client.post(
        "/api/auth/login",
        json={"email": "fresh@example.com", "password": "orig-sturdy-pw-1"},
    ).status_code == 200
    assert client.post(
        "/api/auth/login",
        json={"email": "two@example.com", "password": "orig-sturdy-pw-1"},
    ).status_code == 401


def test_email_change_verify_accepts_a_lowercase_code(client, sent_emails):
    token = _signup_and_token(client, email="lower@example.com")
    _start_email_change(client, token)

    code = _code_from(sent_emails[0]["body"]).lower()
    response = client.post(
        "/api/auth/email-change/verify",
        headers={"Authorization": f"Bearer {token}"},
        json={"code": code},
    )

    assert response.status_code == 200


def test_email_change_verify_rejects_a_wrong_code(client, sent_emails):
    token = _signup_and_token(client, email="badcode@example.com")
    _start_email_change(client, token)

    response = client.post(
        "/api/auth/email-change/verify",
        headers={"Authorization": f"Bearer {token}"},
        json={"code": "ZZZZZZ"},
    )

    assert response.status_code == 400
    assert "not correct" in response.json()["detail"]


def test_email_change_verify_gives_up_after_five_attempts(client, sent_emails):
    token = _signup_and_token(client, email="attempts@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    _start_email_change(client, token)
    real_code = _code_from(sent_emails[0]["body"])
    wrong = "ZZZZZZ" if real_code != "ZZZZZZ" else "YYYYYY"

    for _ in range(4):
        assert client.post(
            "/api/auth/email-change/verify", headers=headers, json={"code": wrong}
        ).status_code == 400

    fifth = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": wrong}
    )
    assert fifth.status_code == 400
    assert "start the email change again" in fifth.json()["detail"].lower()

    # The request is gone -- even the correct code is dead now.
    assert client.get("/api/auth/email-change", headers=headers).json()["pending"] is False
    assert client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": real_code}
    ).status_code == 400


def test_email_change_verify_rejects_an_expired_request(client, sent_emails):
    from dashboard.backend.users import _utcnow

    token = _signup_and_token(client, email="expired@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    _start_email_change(client, token)
    code = _code_from(sent_emails[0]["body"])

    # The `client` fixture patched users_module.user_store to the temp store,
    # so this reaches exactly the database the route just wrote to.
    from dashboard.backend import users as users_module

    stale = (_utcnow() - timedelta(minutes=1)).replace(microsecond=0).isoformat()
    conn = users_module.user_store._get_connection()
    conn.execute("UPDATE email_change_requests SET expires_at = ?", (stale,))
    conn.commit()
    conn.close()

    response = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": code}
    )
    assert response.status_code == 400
    assert "no email change" in response.json()["detail"].lower()


def test_email_change_verify_without_a_request_400s(client):
    token = _signup_and_token(client, email="norequest@example.com")

    response = client.post(
        "/api/auth/email-change/verify",
        headers={"Authorization": f"Bearer {token}"},
        json={"code": "ABC234"},
    )

    assert response.status_code == 400


def test_email_change_stage_two_mail_failure_leaves_stage_old_intact(
    client, sent_emails
):
    # Send before persist, the case that matters most: if stage 'new' were
    # written first and the send then failed, the user would be waiting on a
    # code that never went out while the code they DO hold stopped working.
    token = _signup_and_token(client, email="stuck@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    _start_email_change(client, token)
    code = _code_from(sent_emails[0]["body"])
    sent_emails.fail_sends()

    response = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": code}
    )

    assert response.status_code == 503
    # Still stage 'old' -- nothing was persisted ahead of the failed send.
    assert client.get("/api/auth/email-change", headers=headers).json()["stage"] == "old"

    # And this is the point of the ordering: the code the user already holds is
    # still valid, so once mail recovers they simply resubmit it. No dead end.
    sent_emails.resume_sends()
    retry = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": code}
    )
    assert retry.status_code == 200
    assert retry.json()["stage"] == "new"


def test_email_change_commit_conflicts_when_the_address_was_taken_meanwhile(
    client, sent_emails
):
    # TOCTOU backstop: the request-time 409 cannot cover a signup that lands
    # between the two stages.
    token = _signup_and_token(client, email="race@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    _start_email_change(client, token, new_email="contested@example.com")

    first_code = _code_from(sent_emails[0]["body"])
    client.post("/api/auth/email-change/verify", headers=headers, json={"code": first_code})
    second_code = _code_from(sent_emails[1]["body"])

    _signup_and_token(client, email="contested@example.com")

    response = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": second_code}
    )
    assert response.status_code == 409


def test_email_change_commit_revokes_other_sessions_but_keeps_the_caller(
    client, sent_emails
):
    token_a = _signup_and_token(client, email="sessions@example.com")
    token_b = client.post(
        "/api/auth/login",
        json={"email": "sessions@example.com", "password": "orig-sturdy-pw-1"},
    ).json()["token"]
    headers = {"Authorization": f"Bearer {token_a}"}
    _start_email_change(client, token_a)

    client.post(
        "/api/auth/email-change/verify",
        headers=headers,
        json={"code": _code_from(sent_emails[0]["body"])},
    )
    client.post(
        "/api/auth/email-change/verify",
        headers=headers,
        json={"code": _code_from(sent_emails[1]["body"])},
    )

    assert client.get("/api/auth/me", headers=headers).status_code == 200
    assert client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {token_b}"}
    ).status_code == 401


def test_changing_the_password_cancels_a_pending_email_change(client, sent_emails):
    # D7: a user who suspects compromise changes their password; an attacker's
    # in-flight email change must die with it.
    token = _signup_and_token(client, email="d7@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    _start_email_change(client, token)
    assert client.get("/api/auth/email-change", headers=headers).json()["pending"] is True

    assert client.post(
        "/api/auth/change-password",
        headers=headers,
        json={
            "current_password": "orig-sturdy-pw-1",
            "new_password": "new-sturdy-pw-2",
        },
    ).status_code == 200

    assert client.get("/api/auth/email-change", headers=headers).json()["pending"] is False
