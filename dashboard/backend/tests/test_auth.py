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
from dashboard.backend.domain.agents.defaults import STARTER_AGENTS
from dashboard.backend.users import UserStore

def _session_token(client) -> str:
    """Raw session token from the HttpOnly cookie set by login/signup."""
    from dashboard.backend.auth_cookies import session_cookie_name
    token = client.cookies.get(session_cookie_name())
    assert token, f"missing session cookie {session_cookie_name()!r} in {dict(client.cookies)}"
    return token



@pytest.fixture
def temp_user_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = UserStore(db_path=Path(tmpdir) / "auth_test.db")
        yield store


@pytest.fixture
def client(temp_user_store, monkeypatch):
    from dashboard.backend import users
    from dashboard.backend.api import auth
    from dashboard.backend.domain.credits.repository import CreditsStore
    from dashboard.backend.domain.credits.service import CreditsService

    monkeypatch.setattr(users, "user_store", temp_user_store)
    monkeypatch.setattr(
        auth,
        "credits_service",
        CreditsService(store=CreditsStore(temp_user_store.db_path)),
    )
    # The process-global auth limiters are reset by conftest's autouse
    # _reset_shared_scale_state, which pytest runs before this fixture.
    return TestClient(app)


def test_signup_and_login_keep_one_welcome_credit_grant(client):
    from dashboard.backend.api import auth

    signup = client.post(
        "/api/auth/signup",
        json={
            "email": "welcome@example.com",
            "display_name": "Welcome",
            "password": "securepass1",
        },
    )
    assert signup.status_code == 200
    user_id = signup.json()["user"]["id"]
    assert auth.credits_service.get_balance(user_id).display_credits == "1.500000"

    login = client.post(
        "/api/auth/login",
        json={"email": "welcome@example.com", "password": "securepass1"},
    )
    assert login.status_code == 200
    activity = auth.credits_service.list_ledger(user_id, limit=10, cursor=None)
    assert len(activity["items"]) == 1


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
    assert "token" not in signup_data
    assert _session_token(client)  # signup set the session cookie

    listed = client.get("/api/v1/agents")
    assert listed.status_code == 200
    listed_by_model = {
        a.get("model_name"): a for a in listed.json()["agents"]
    }
    for spec in STARTER_AGENTS:
        starter = listed_by_model.get(spec["model_name"])
        assert starter, f"signup must provision {spec['name']}"
        assert starter["name"] == spec["name"]
        assert starter["agent_type"] == "builtin"
        assert starter["pipeline"], "starter must be usable without Configure"

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
    assert "token" not in login.json()
    token = _session_token(client)

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "alice@example.com"

    logout = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout.status_code == 200

    me_after = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_after.status_code == 401


def test_signup_and_login_emit_authoritative_account_events(client, monkeypatch):
    from dashboard.backend.domain.analytics import instrumentation

    events = []
    monkeypatch.setattr(
        instrumentation,
        "emit_account_event",
        lambda **kwargs: events.append(kwargs),
    )

    signup = client.post(
        "/api/auth/signup",
        json={
            "email": "analytics@example.com",
            "display_name": "Analytics",
            "password": "securepass1",
        },
    )
    login = client.post(
        "/api/auth/login",
        json={"email": "analytics@example.com", "password": "securepass1"},
    )

    assert signup.status_code == 200
    assert login.status_code == 200
    assert [event["event_name"] for event in events] == [
        "account_signed_up",
        "authenticated_session_started",
    ]
    assert all(event["user_id"] == signup.json()["user"]["id"] for event in events)
    assert "password" not in repr(events)


def test_me_requires_auth(client):
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_me_accepts_session_cookie_without_bearer(client):
    signup = client.post(
        "/api/auth/signup",
        json={
            "email": "cookie@example.com",
            "display_name": "Cookie",
            "password": "securepass1",
        },
    )
    assert signup.status_code == 200
    assert "token" not in signup.json()
    from dashboard.backend.auth_cookies import session_cookie_name

    assert session_cookie_name() in client.cookies
    me = client.get("/api/auth/me")  # cookie jar only — no Authorization
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "cookie@example.com"


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
    assert response.json()["detail"] == "Invalid email or password."


def test_login_unknown_email_uses_same_generic_error(client):
    """Do not reveal whether an address is registered."""
    known = client.post(
        "/api/auth/signup",
        json={
            "email": "known@example.com",
            "display_name": "Known",
            "password": "securepass1",
        },
    )
    assert known.status_code == 200
    wrong_password = client.post(
        "/api/auth/login",
        json={"email": "known@example.com", "password": "not-the-password"},
    )
    unknown = client.post(
        "/api/auth/login",
        json={"email": "nobody@example.com", "password": "securepass1"},
    )
    assert wrong_password.status_code == 401
    assert unknown.status_code == 401
    assert wrong_password.json()["detail"] == unknown.json()["detail"] == (
        "Invalid email or password."
    )


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
    assert "token" not in response.json()
    return _session_token(client)


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
    client.post(
        "/api/auth/login",
        json={"email": "gina@example.com", "password": "orig-sturdy-pw-1"},
    )
    token_b = _session_token(client)

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


def _running_on_the_event_loop() -> bool:
    """True when the calling thread is the one running the event loop.

    A worker thread has no running loop, so ``get_running_loop`` raises there.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def test_password_verify_never_runs_on_the_event_loop(client, monkeypatch, sent_emails):
    """bcrypt must execute in a worker thread on every route that verifies one.

    bcrypt is deliberately slow (~190 ms of CPU). Run on the event loop it
    stalls every concurrent request for that long, which is what #292 fixed
    across 58 handlers and #297 tracked for the two auth.py routes below.

    This asserts the *property* -- no running loop on the thread doing the hash
    -- rather than the mechanism, so it stays green for whichever offload a
    handler uses (a plain ``def`` dispatched to the threadpool, asyncio.to_thread
    or run_in_threadpool) and goes red only when the hash is back on the loop.
    An assertion that a specific dispatcher was called has the failure modes
    reversed: it breaks on a correct refactor, and passes while the handler
    still blocks on everything the offload did not cover.
    """
    # `from ... import auth as auth_api`, matching this file's other five sites:
    # the plain `import dashboard.backend.api.auth` form collides with the
    # `from dashboard.backend.api.auth import _humanize_wait` further down and
    # trips py/import-and-import-from. Either form yields the module object the
    # setattr seam below needs.
    from dashboard.backend.api import auth as auth_api

    ran_on_loop = []
    real_verify = auth_api.verify_password

    def recording_verify(password, password_hash):
        ran_on_loop.append(_running_on_the_event_loop())
        return real_verify(password, password_hash)

    # Patched on the auth module -- the name both handlers actually call --
    # rather than on asyncio or users. Patching `asyncio.to_thread` would rebind
    # it for the whole interpreter, so unrelated callers would land in the
    # recorder and the assertion would be over global traffic, not these routes.
    monkeypatch.setattr(auth_api, "verify_password", recording_verify)

    token = _signup_and_token(client, email="helen@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    changed = client.post(
        "/api/auth/change-password",
        headers=headers,
        json={"current_password": "orig-sturdy-pw-1", "new_password": "new-sturdy-pw-2"},
    )
    assert changed.status_code == 200

    requested = client.post(
        "/api/auth/email-change",
        headers=headers,
        json={
            "current_password": "new-sturdy-pw-2",
            "new_email": "helen-next@example.com",
        },
    )
    assert requested.status_code == 200, requested.text

    assert len(ran_on_loop) == 2, (
        f"expected one bcrypt verify per route, saw {len(ran_on_loop)} -- the "
        "test no longer exercises what it claims to"
    )
    assert ran_on_loop == [False, False], (
        "a bcrypt verify ran on the event loop thread, stalling every "
        f"concurrent request for its duration (per-call on-loop: {ran_on_loop})"
    )


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


def _code_after(email_body, label):
    """Pull the 6-character code following `label` out of a captured body."""
    import re

    from dashboard.backend.verification_codes import CODE_ALPHABET

    match = re.search(rf"{label} ([{CODE_ALPHABET}]{{6}})", email_body)
    assert match, f"no code found in: {email_body!r}"
    return match.group(1)


def _code_from(email_body):
    return _code_after(email_body, "code is:")


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

    first = client.post("/api/auth/email-change", headers=headers, json=body)
    second = client.post("/api/auth/email-change", headers=headers, json=body)

    assert first.status_code == 200
    assert second.status_code == 429
    # A range, not == "60": the header carries the wait that is actually left,
    # so it drops below the window's width as the test itself takes time.
    assert 0 < int(second.headers["Retry-After"]) <= 60
    assert len(sent_emails) == 1


def test_email_change_cooldown_survives_cancel_and_resend(client, sent_emails):
    # The bug this guards against: DELETE needs only a session, not the
    # password, so without a fix a caller who knows the password could loop
    # request -> cancel -> request with the cooldown never enforced --
    # mail-bombing the account and burning the shared Brevo daily quota.
    token = _signup_and_token(client, email="bounce@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    body = {"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"}

    first = client.post("/api/auth/email-change", headers=headers, json=body)
    cancel = client.delete("/api/auth/email-change", headers=headers)

    assert first.status_code == 200
    assert cancel.status_code == 200

    second = client.post("/api/auth/email-change", headers=headers, json=body)

    assert second.status_code == 429
    assert len(sent_emails) == 1


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (1, "1 minute"),
        (59, "1 minute"),
        (61, "2 minutes"),          # rounds UP: "1 minute" would 429 again
        (3601, "2 hours"),
        (86_400, "1 day"),
        (86_401, "2 days"),
        (5 * 86_400 + 1, "6 days"),
    ],
)
def test_humanize_wait_always_rounds_up_to_a_whole_unit(seconds, expected):
    """Rounding down would invite a retry that is refused again.

    Exact boundaries included: 86_400 must read "1 day", not "2 days" -- an
    off-by-one in the ceiling would tell every capped user to wait a day longer
    than they have to.
    """
    from dashboard.backend.api.auth import _humanize_wait

    assert _humanize_wait(seconds) == expected


def _backdate_rows(store, table, user_id, **columns):
    """Age this user's request rows so a window-based limit can be exercised.

    The windows are a day and a week wide, so the alternative is a frozen clock.
    """
    assignments = ", ".join(f"{name} = ?" for name in columns)
    conn = store._get_connection()
    conn.execute(
        f"UPDATE {table} SET {assignments} WHERE user_id = ?",  # noqa: S608
        (*columns.values(), user_id),
    )
    conn.commit()
    conn.close()


def _backdate_email_change_rows(store, user_id, **columns):
    _backdate_rows(store, "email_change_requests", user_id, **columns)


def _backdate_latest_email_change_row(store, user_id, **columns):
    """Age only the NEWEST request row, leaving earlier rows' timestamps alone.

    The all-rows variant above cannot give rows distinct ages: every call
    rewrites the earlier rows' timestamps too.
    """
    assignments = ", ".join(f"{name} = ?" for name in columns)
    conn = store._get_connection()
    conn.execute(
        f"UPDATE email_change_requests SET {assignments} "  # noqa: S608
        "WHERE id = (SELECT MAX(id) FROM email_change_requests WHERE user_id = ?)",
        (*columns.values(), user_id),
    )
    conn.commit()
    conn.close()


def _stored_time(**delta):
    from dashboard.backend.users import _utcnow, format_stored_timestamp

    return format_stored_timestamp(_utcnow() - timedelta(**delta))


def _complete_email_change(client, token, sent_emails, new_email):
    """Drive both verification stages through to a committed change."""
    headers = {"Authorization": f"Bearer {token}"}
    before = len(sent_emails)
    started = client.post(
        "/api/auth/email-change",
        headers=headers,
        json={"current_password": "orig-sturdy-pw-1", "new_email": new_email},
    )
    assert started.status_code == 200, started.text
    stage_two = client.post(
        "/api/auth/email-change/verify",
        headers=headers,
        json={"code": _code_from(sent_emails[before]["body"])},
    )
    assert stage_two.status_code == 200, stage_two.text
    done = client.post(
        "/api/auth/email-change/verify",
        headers=headers,
        json={"code": _code_from(sent_emails[before + 1]["body"])},
    )
    assert done.status_code == 200, done.text
    return done


def test_email_change_is_blocked_for_a_week_after_one_completes(
    client, sent_emails, temp_user_store
):
    token = _signup_and_token(client, email="churn@example.com")
    _complete_email_change(client, token, sent_emails, "second@example.com")
    # Clear the 60s and daily limits so the 7-day one is what is under test.
    user = temp_user_store.get_user_by_email("second@example.com")
    _backdate_email_change_rows(
        temp_user_store, user["id"], created_at=_stored_time(days=2)
    )

    blocked = client.post(
        "/api/auth/email-change",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_email": "third@example.com"},
    )

    assert blocked.status_code == 429
    assert "once every 7 days" in blocked.json()["detail"]
    # Reports the wait that is left (~5 more days), not the window's full width.
    remaining = int(blocked.headers["Retry-After"])
    assert 4 * 86400 < remaining <= 7 * 86400
    assert len(sent_emails) == 2  # nothing new went out


def test_email_change_is_allowed_again_once_the_week_has_passed(
    client, sent_emails, temp_user_store
):
    token = _signup_and_token(client, email="patient@example.com")
    _complete_email_change(client, token, sent_emails, "second@example.com")
    user = temp_user_store.get_user_by_email("second@example.com")
    _backdate_email_change_rows(
        temp_user_store,
        user["id"],
        created_at=_stored_time(days=8),
        used_at=_stored_time(days=8),
    )

    allowed = client.post(
        "/api/auth/email-change",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_email": "third@example.com"},
    )

    assert allowed.status_code == 200
    assert len(sent_emails) == 3


def test_email_change_requests_are_capped_per_day(client, sent_emails, temp_user_store):
    """The 60s cooldown bounds a cycle; this bounds the shared provider quota.

    Without it one account can loop request -> cancel -> request all day, and
    each completed cycle also mails an address the requester chose -- so the
    cap is what stops a single account draining the platform's daily send
    allowance, half of it aimed at a third party from our sending domain.
    """
    from dashboard.backend.users import EMAIL_CHANGE_MAX_REQUESTS_PER_DAY

    token = _signup_and_token(client, email="flood@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    user = temp_user_store.get_user_by_email("flood@example.com")
    body = {"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"}

    for attempt in range(EMAIL_CHANGE_MAX_REQUESTS_PER_DAY):
        response = client.post("/api/auth/email-change", headers=headers, json=body)
        assert response.status_code == 200, f"request {attempt} -> {response.text}"
        # Step past the 60s cooldown, and give each row a DISTINCT age (3h,
        # 2h, 1h): only the newest row is touched, so earlier backdates
        # survive and the window's oldest entry -- which is what Retry-After
        # is measured from -- is distinguishable from its newest.
        _backdate_latest_email_change_row(
            temp_user_store,
            user["id"],
            created_at=_stored_time(hours=EMAIL_CHANGE_MAX_REQUESTS_PER_DAY - attempt),
        )

    capped = client.post("/api/auth/email-change", headers=headers, json=body)

    assert capped.status_code == 429
    assert "Too many email-change requests today" in capped.json()["detail"]
    assert len(sent_emails) == EMAIL_CHANGE_MAX_REQUESTS_PER_DAY
    # The window frees when the OLDEST of those requests ages out: 24h minus
    # the oldest row's 3h age, less the moment the test took to get here.
    # Measured from the newest row instead, this would read ~23h and fail.
    oldest_age = EMAIL_CHANGE_MAX_REQUESTS_PER_DAY * 3600
    retry_after = int(capped.headers["Retry-After"])
    assert 86400 - oldest_age - 60 < retry_after <= 86400 - oldest_age


def test_email_change_daily_cap_counts_cancelled_requests(
    client, sent_emails, temp_user_store
):
    """Cancelling does not un-send the message the request already triggered."""
    from dashboard.backend.users import EMAIL_CHANGE_MAX_REQUESTS_PER_DAY

    token = _signup_and_token(client, email="cancels@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    user = temp_user_store.get_user_by_email("cancels@example.com")
    body = {"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"}

    for _ in range(EMAIL_CHANGE_MAX_REQUESTS_PER_DAY):
        # Requests hoisted out of the asserts: -O strips assert statements, and
        # a stripped assert here would drop the HTTP call the loop exists for.
        requested = client.post("/api/auth/email-change", headers=headers, json=body)
        assert requested.status_code == 200, requested.text
        cancelled = client.delete("/api/auth/email-change", headers=headers)
        assert cancelled.status_code == 200, cancelled.text
        _backdate_email_change_rows(
            temp_user_store, user["id"], created_at=_stored_time(minutes=2)
        )

    capped = client.post("/api/auth/email-change", headers=headers, json=body)

    assert capped.status_code == 429
    assert "Too many email-change requests today" in capped.json()["detail"]


def test_email_change_daily_cap_window_rolls(client, sent_emails, temp_user_store):
    from dashboard.backend.users import EMAIL_CHANGE_MAX_REQUESTS_PER_DAY

    token = _signup_and_token(client, email="rolls@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    user = temp_user_store.get_user_by_email("rolls@example.com")
    body = {"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"}

    for _ in range(EMAIL_CHANGE_MAX_REQUESTS_PER_DAY):
        # Hoisted out of the assert -- see the sibling test above.
        requested = client.post("/api/auth/email-change", headers=headers, json=body)
        assert requested.status_code == 200, requested.text
        _backdate_email_change_rows(
            temp_user_store, user["id"], created_at=_stored_time(minutes=2)
        )
    capped = client.post("/api/auth/email-change", headers=headers, json=body)
    assert capped.status_code == 429, capped.text

    # Age every request out of the 24h window; the allowance comes back.
    _backdate_email_change_rows(
        temp_user_store, user["id"], created_at=_stored_time(days=1, minutes=1)
    )

    recovered = client.post("/api/auth/email-change", headers=headers, json=body)
    assert recovered.status_code == 200, recovered.text


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

    cancel = client.delete("/api/auth/email-change", headers=headers)
    assert cancel.status_code == 200
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
    fresh_login = client.post(
        "/api/auth/login",
        json={"email": "fresh@example.com", "password": "orig-sturdy-pw-1"},
    )
    assert fresh_login.status_code == 200
    stale_login = client.post(
        "/api/auth/login",
        json={"email": "two@example.com", "password": "orig-sturdy-pw-1"},
    )
    assert stale_login.status_code == 401


def test_stage_two_mail_reads_correctly_for_a_recipient_with_no_account(
    client, sent_emails
):
    """The new address may belong to someone else entirely.

    Its owner has no Agentic Trading Lab account and no password there, so
    copy reused from the stage-one mail -- "your ... account", "change your
    password" -- is wrong for them, and instructions that cannot apply to the
    reader are exactly what phishing looks like.
    """
    token = _signup_and_token(client, email="holder@example.com")
    _start_email_change(client, token)
    advanced = client.post(
        "/api/auth/email-change/verify",
        headers={"Authorization": f"Bearer {token}"},
        json={"code": _code_from(sent_emails[0]["body"])},
    )
    assert advanced.status_code == 200

    owner_body, recipient_body = (mail["body"] for mail in sent_emails)
    # The account owner is told how to react to a hijack attempt...
    assert "your Agentic Trading Lab account" in owner_body
    assert "change your password" in owner_body
    # ...but the stage-two recipient is a bystander until they opt in.
    assert "their Agentic Trading Lab account" in recipient_body
    assert "your Agentic Trading Lab account" not in recipient_body
    assert "change your password" not in recipient_body


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
        attempt = client.post(
            "/api/auth/email-change/verify", headers=headers, json={"code": wrong}
        )
        assert attempt.status_code == 400

    fifth = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": wrong}
    )
    assert fifth.status_code == 400
    assert "start the email change again" in fifth.json()["detail"].lower()

    # The request is gone -- even the correct code is dead now.
    assert client.get("/api/auth/email-change", headers=headers).json()["pending"] is False
    after_reset = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": real_code}
    )
    assert after_reset.status_code == 400


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
    client.post(
        "/api/auth/login",
        json={"email": "sessions@example.com", "password": "orig-sturdy-pw-1"},
    )
    token_b = _session_token(client)
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

    password_change = client.post(
        "/api/auth/change-password",
        headers=headers,
        json={
            "current_password": "orig-sturdy-pw-1",
            "new_password": "new-sturdy-pw-2",
        },
    )
    assert password_change.status_code == 200

    assert client.get("/api/auth/email-change", headers=headers).json()["pending"] is False


def test_session_stores_hash_not_raw_token(client, temp_user_store):
    from dashboard.backend.session_tokens import hash_session_token

    signup = client.post(
        "/api/auth/signup",
        json={
            "email": "hashme@example.com",
            "display_name": "Hash",
            "password": "securepass1",
        },
    )
    assert signup.status_code == 200
    assert "token" not in signup.json()
    token = _session_token(client)
    conn = temp_user_store._get_connection()
    conn.row_factory = None
    rows = list(conn.execute("SELECT * FROM auth_sessions"))
    conn.close()
    assert len(rows) == 1
    assert hash_session_token(token) in rows[0]
    # Every column, not just token_hash: the point of the change is that the
    # raw bearer token is nowhere in the row a database leak would expose.
    assert token not in str(rows[0])


def test_revoked_session_is_rejected(client):
    signup = client.post(
        "/api/auth/signup",
        json={
            "email": "revoke@example.com",
            "display_name": "Revoke",
            "password": "securepass1",
        },
    )
    assert "token" not in signup.json()
    token = _session_token(client)
    assert client.post(
        "/api/auth/logout", headers={"Authorization": f"Bearer {token}"}
    ).status_code == 200
    assert (
        client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code
        == 401
    )


def test_idle_session_is_rejected(client, temp_user_store, monkeypatch):
    monkeypatch.setenv("SESSION_IDLE_HOURS", "1")
    signup = client.post(
        "/api/auth/signup",
        json={
            "email": "idle@example.com",
            "display_name": "Idle",
            "password": "securepass1",
        },
    )
    assert "token" not in signup.json()
    token = _session_token(client)
    from datetime import timedelta

    from dashboard.backend.session_tokens import hash_session_token
    from dashboard.backend.users import _utcnow, format_stored_timestamp

    stale = format_stored_timestamp(_utcnow() - timedelta(hours=2))
    conn = temp_user_store._get_connection()
    conn.execute(
        "UPDATE auth_sessions SET last_seen_at = ?, created_at = ? WHERE token_hash = ?",
        (stale, stale, hash_session_token(token)),
    )
    conn.commit()
    conn.close()
    assert (
        client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code
        == 401
    )


# ---------------------------------------------------------------------------
# The plaintext-token -> token_hash migration
#
# This is the one destructive statement in the hashed-session change: it DROPs a
# table that on prod holds every live login. It runs against durable Postgres
# (USERS_DATABASE_URL) as well as SQLite, so "it worked when I tried it" is not
# coverage. The Postgres twin has its own copy in test_users_postgres.py.
# ---------------------------------------------------------------------------

_LEGACY_SESSIONS_DDL = """
    CREATE TABLE auth_sessions (
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
"""


@pytest.fixture
def legacy_session_db(tmp_path):
    """A users database still on the pre-hash schema, with one live session."""
    import sqlite3

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(_LEGACY_SESSIONS_DDL)
    conn.execute(
        "INSERT INTO users (email, display_name, password_hash) VALUES (?, ?, ?)",
        ("legacy@example.com", "Legacy", "unused-hash"),
    )
    conn.execute(
        "INSERT INTO auth_sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
        ("legacy-plaintext-token", 1, "2099-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()
    return db_path


def test_the_legacy_session_migration_keeps_accounts_and_drops_logins(legacy_session_db):
    """Sessions cannot be re-hashed without the raw token, so they go -- users stay.

    The failure this guards is a DROP that takes the accounts with it. Nothing
    would surface that until someone tried to sign in to an account that no
    longer exists.
    """
    import sqlite3

    UserStore(db_path=legacy_session_db)

    conn = sqlite3.connect(legacy_session_db)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(auth_sessions)")}
        assert "token_hash" in columns and "token" not in columns
        assert conn.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
    finally:
        conn.close()


def test_the_legacy_session_migration_restores_the_user_id_index(legacy_session_db):
    """DROP TABLE takes the table's indexes with it.

    The recreate has to come before the CREATE INDEX IF NOT EXISTS, or the index
    is silently gone and every session lookup by user_id degrades to a scan --
    with nothing failing to show it.
    """
    import sqlite3

    UserStore(db_path=legacy_session_db)

    conn = sqlite3.connect(legacy_session_db)
    try:
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
    finally:
        conn.close()
    assert "idx_auth_sessions_user_id" in indexes


def test_a_surviving_legacy_token_cannot_authenticate_after_migration(legacy_session_db):
    store = UserStore(db_path=legacy_session_db)
    assert store.get_user_for_token("legacy-plaintext-token") is None


def test_the_legacy_session_migration_announces_itself(legacy_session_db, capsys):
    """Dropping every live login in prod must not happen in silence.

    The only symptom otherwise is a wave of users being signed out with nothing
    in the deploy log to connect it to the release.
    """
    UserStore(db_path=legacy_session_db)
    assert "auth_sessions" in capsys.readouterr().out


def test_a_migrated_store_does_not_re_announce_on_the_next_boot(legacy_session_db, capsys):
    """The migration is one-shot; a restart must not keep claiming it ran."""
    UserStore(db_path=legacy_session_db)
    capsys.readouterr()
    UserStore(db_path=legacy_session_db)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Reclaiming dead session rows
#
# Revocation is a soft UPDATE of revoked_at, and get_user_for_token returns at
# the revoked check before reaching any cleanup -- so without a sweep a revoked
# row is immortal. Expired rows fare no better: they are only deleted when
# someone re-presents the dead token, which nobody does. Before hashing, logout
# DELETEd the row and the table trimmed itself.
# ---------------------------------------------------------------------------


def _session_count(store) -> int:
    conn = store._get_connection()
    try:
        return conn.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0]
    finally:
        conn.close()


def _backdate_expiry(store, token_hash: str) -> None:
    from dashboard.backend.users import _utcnow, format_stored_timestamp

    conn = store._get_connection()
    conn.execute(
        "UPDATE auth_sessions SET expires_at = ? WHERE token_hash = ?",
        (format_stored_timestamp(_utcnow() - timedelta(days=1)), token_hash),
    )
    conn.commit()
    conn.close()


def test_creating_a_session_reclaims_rows_that_have_already_expired(temp_user_store):
    from dashboard.backend.session_tokens import hash_session_token

    user = temp_user_store.create_user("sweep@example.com", "Sweep", "securepass1")
    dead = temp_user_store.create_session(user["id"])
    _backdate_expiry(temp_user_store, hash_session_token(dead))

    temp_user_store.create_session(user["id"])

    assert _session_count(temp_user_store) == 1


def test_a_revoked_session_row_is_reclaimed_once_it_expires(temp_user_store):
    """Soft revocation is not a licence to keep the row forever.

    The absolute TTL bounds it: a revoked row is collected by the same sweep
    within SESSION_TTL_DAYS at the latest, which is the pre-hash bound.
    """
    from dashboard.backend.session_tokens import hash_session_token

    user = temp_user_store.create_user("revoked@example.com", "Rev", "securepass1")
    token = temp_user_store.create_session(user["id"])
    temp_user_store.delete_session(token)
    _backdate_expiry(temp_user_store, hash_session_token(token))

    temp_user_store.create_session(user["id"])

    assert _session_count(temp_user_store) == 1


def test_the_sweep_never_touches_a_live_session(temp_user_store):
    """The negative half, and the one that matters.

    A sweep with a wrong comparison signs everybody out at their next login and
    looks exactly like a working sweep until someone complains.
    """
    user = temp_user_store.create_user("live@example.com", "Live", "securepass1")
    keep = temp_user_store.create_session(user["id"])

    temp_user_store.create_session(user["id"])

    assert _session_count(temp_user_store) == 2
    assert temp_user_store.get_user_for_token(keep) is not None


def test_a_write_lock_cannot_invalidate_a_good_session(temp_user_store, monkeypatch):
    """The last_seen_at touch is an optimisation; it must never fail the request.

    get_user_for_token was read-only until sessions were hashed. It now writes,
    so it can lose a race for the write lock -- and sqlite3 raising
    OperationalError out of here reaches get_current_user as a 500 on a session
    that is perfectly valid. Losing one throttled timestamp update is free;
    signing the user out over it is not.
    """
    import sqlite3

    from dashboard.backend.session_tokens import hash_session_token
    from dashboard.backend.users import UserStore, _utcnow, format_stored_timestamp

    monkeypatch.setenv("SESSION_LAST_SEEN_THROTTLE_SECONDS", "1")
    user = temp_user_store.create_user("locked@example.com", "Locked", "securepass1")
    token = temp_user_store.create_session(user["id"])

    stale = format_stored_timestamp(_utcnow() - timedelta(hours=1))
    conn = temp_user_store._get_connection()
    conn.execute(
        "UPDATE auth_sessions SET last_seen_at = ? WHERE token_hash = ?",
        (stale, hash_session_token(token)),
    )
    conn.commit()
    conn.close()

    # Don't sit out the default 5s busy timeout just to observe the failure.
    original_get_connection = UserStore._get_connection

    def impatient(self):
        opened = original_get_connection(self)
        opened.execute("PRAGMA busy_timeout = 0")
        return opened

    monkeypatch.setattr(UserStore, "_get_connection", impatient)

    blocker = sqlite3.connect(str(temp_user_store.db_path), timeout=0)
    blocker.execute("BEGIN IMMEDIATE")  # RESERVED: reads still pass, writes do not
    try:
        assert temp_user_store.get_user_for_token(token) is not None
    finally:
        blocker.rollback()
        blocker.close()


def test_a_revoked_but_unexpired_session_row_survives_the_sweep(temp_user_store):
    """Revoked-and-still-inside-its-TTL is the window where revoked_at is read."""
    user = temp_user_store.create_user("soft@example.com", "Soft", "securepass1")
    token = temp_user_store.create_session(user["id"])
    temp_user_store.delete_session(token)

    temp_user_store.create_session(user["id"])

    assert _session_count(temp_user_store) == 2
    assert temp_user_store.get_user_for_token(token) is None


# --- Login / signup rate limits -------------------------------------------------


@pytest.fixture
def auth_rate_limit_clock(monkeypatch):
    """Deterministic clock + tiny limits for auth rate-limit tests."""
    from dashboard.backend.api import auth as auth_api
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter

    now = [1000.0]

    def clock() -> float:
        return now[0]

    monkeypatch.setattr(
        auth_api,
        "_LOGIN_IP_LIMITER",
        FixedWindowRateLimiter(max_events=5, window_seconds=60, clock=clock),
    )
    monkeypatch.setattr(
        auth_api,
        "_LOGIN_EMAIL_LIMITER",
        FixedWindowRateLimiter(max_events=3, window_seconds=60, clock=clock),
    )
    monkeypatch.setattr(
        auth_api,
        "_SIGNUP_IP_LIMITER",
        FixedWindowRateLimiter(max_events=2, window_seconds=60, clock=clock),
    )
    monkeypatch.setattr(
        auth_api,
        "_SIGNUP_EMAIL_LIMITER",
        FixedWindowRateLimiter(max_events=2, window_seconds=60, clock=clock),
    )
    return now


def test_login_ip_rate_limit_returns_429(client, auth_rate_limit_clock, monkeypatch):
    from dashboard.backend.api import auth as auth_api
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter

    now = auth_rate_limit_clock
    monkeypatch.setattr(
        auth_api,
        "_LOGIN_EMAIL_LIMITER",
        FixedWindowRateLimiter(max_events=100, window_seconds=60, clock=lambda: now[0]),
    )
    monkeypatch.setattr(
        auth_api,
        "_SIGNUP_IP_LIMITER",
        FixedWindowRateLimiter(max_events=100, window_seconds=60, clock=lambda: now[0]),
    )
    monkeypatch.setattr(
        auth_api,
        "_SIGNUP_EMAIL_LIMITER",
        FixedWindowRateLimiter(max_events=100, window_seconds=60, clock=lambda: now[0]),
    )
    assert (
        client.post(
            "/api/auth/signup",
            json={
                "email": "rate-ip@example.com",
                "display_name": "Rate",
                "password": "securepass1",
            },
        ).status_code
        == 200
    )
    for _ in range(5):
        resp = client.post(
            "/api/auth/login",
            json={"email": "rate-ip@example.com", "password": "wrong"},
        )
        assert resp.status_code == 401, resp.text

    blocked = client.post(
        "/api/auth/login",
        json={"email": "rate-ip@example.com", "password": "wrong"},
    )
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers
    # Bounded at both ends: >= 1 alone would pass for a header telling the
    # client to wait longer than the window it is actually waiting on.
    assert 1 <= int(blocked.headers["Retry-After"]) <= 60


def test_login_email_rate_limit_after_failures(client, auth_rate_limit_clock, monkeypatch):
    from dashboard.backend.api import auth as auth_api
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter

    now = auth_rate_limit_clock
    # Generous IP budget so only the per-email failure counter trips.
    monkeypatch.setattr(
        auth_api,
        "_LOGIN_IP_LIMITER",
        FixedWindowRateLimiter(max_events=100, window_seconds=60, clock=lambda: now[0]),
    )

    client.post(
        "/api/auth/signup",
        json={
            "email": "rate-email@example.com",
            "display_name": "Rate",
            "password": "securepass1",
        },
    )
    for _ in range(3):
        resp = client.post(
            "/api/auth/login",
            json={"email": "rate-email@example.com", "password": "wrong"},
        )
        assert resp.status_code == 401, resp.text

    blocked = client.post(
        "/api/auth/login",
        json={"email": "rate-email@example.com", "password": "wrong"},
    )
    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["Retry-After"]) <= 60

    # Correct password still works: the email counter only meters *failures*
    # so an attacker cannot lock the account out of a legitimate login.
    ok = client.post(
        "/api/auth/login",
        json={"email": "rate-email@example.com", "password": "securepass1"},
    )
    assert ok.status_code == 200, ok.text

    now[0] += 61
    # After the window, wrong passwords are accepted into the failure budget again.
    again = client.post(
        "/api/auth/login",
        json={"email": "rate-email@example.com", "password": "wrong"},
    )
    assert again.status_code == 401


def test_signup_ip_rate_limit_returns_429(client, auth_rate_limit_clock):
    assert (
        client.post(
            "/api/auth/signup",
            json={
                "email": "su1@example.com",
                "display_name": "One",
                "password": "securepass1",
            },
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/auth/signup",
            json={
                "email": "su2@example.com",
                "display_name": "Two",
                "password": "securepass1",
            },
        ).status_code
        == 200
    )
    blocked = client.post(
        "/api/auth/signup",
        json={
            "email": "su3@example.com",
            "display_name": "Three",
            "password": "securepass1",
        },
    )
    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["Retry-After"]) <= 60


def test_successful_logins_do_not_consume_the_ip_budget(client, auth_rate_limit_clock):
    """The per-IP budget meters failures only.

    Without this, ``allow()`` runs before ``authenticate()`` and charges every
    caller including the ones typing the right password -- so the budget is
    really a cap on *how many people may sign in*, which is a self-inflicted
    outage rather than a control. It bites hardest exactly where the key is
    coarsest: one office, one classroom, or (before client_ip() read the
    forwarded header) every visitor to the site at once.
    """
    signup = client.post(
        "/api/auth/signup",
        json={
            "email": "busy@example.com",
            "display_name": "Busy",
            "password": "securepass1",
        },
    )
    assert signup.status_code == 200

    # The fixture's IP budget is 5; sign in more times than that.
    for i in range(8):
        ok = client.post(
            "/api/auth/login",
            json={"email": "busy@example.com", "password": "securepass1"},
        )
        assert ok.status_code == 200, f"login {i + 1} refused: {ok.text}"

    # And the budget is genuinely untouched, not merely not-yet-exhausted: a
    # wrong password is still answered 401 rather than 429.
    assert (
        client.post(
            "/api/auth/login",
            json={"email": "busy@example.com", "password": "wrong"},
        ).status_code
        == 401
    )


def test_login_ip_budget_is_per_forwarded_address(
    client, auth_rate_limit_clock, monkeypatch
):
    """Two clients behind the same proxy get their own budgets.

    ``request.client.host`` is the proxy for every visitor on a PaaS router, so
    keying on it alone puts the whole site in one bucket. Nothing here is
    spoof-proof -- the point is that honest clients stop colliding.
    """
    from dashboard.backend.api import auth as auth_api
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter

    now = auth_rate_limit_clock
    # Generous per-email budget so only the per-IP one can trip.
    monkeypatch.setattr(
        auth_api,
        "_LOGIN_EMAIL_LIMITER",
        FixedWindowRateLimiter(max_events=100, window_seconds=60, clock=lambda: now[0]),
    )

    for email in ("fwd-a@example.com", "fwd-b@example.com"):
        assert (
            client.post(
                "/api/auth/signup",
                json={
                    "email": email,
                    "display_name": "Fwd",
                    "password": "securepass1",
                },
            ).status_code
            == 200
        )

    # Exhaust the 5-failure IP budget for one address...
    for _ in range(5):
        assert (
            client.post(
                "/api/auth/login",
                json={"email": "fwd-a@example.com", "password": "wrong"},
                headers={"x-forwarded-for": "203.0.113.7"},
            ).status_code
            == 401
        )
    assert (
        client.post(
            "/api/auth/login",
            json={"email": "fwd-a@example.com", "password": "wrong"},
            headers={"x-forwarded-for": "203.0.113.7"},
        ).status_code
        == 429
    )

    # ...a different forwarded address is unaffected.
    assert (
        client.post(
            "/api/auth/login",
            json={"email": "fwd-b@example.com", "password": "wrong"},
            headers={"x-forwarded-for": "198.51.100.4"},
        ).status_code
        == 401
    )


def test_login_unknown_email_still_costs_one_password_compare(client, monkeypatch):
    """Constant-time miss path: the generic 401 copy is not enough on its own.

    Returning before bcrypt made an unknown address answer ~3000x faster than a
    wrong password (0.06 ms vs 182 ms measured), so the *timing* kept answering
    the question the *body* had stopped answering. Asserted behaviourally --
    that a compare happens, at the real cost factor -- because a wall-clock
    assertion would be flaky under CI load.
    """
    from dashboard.backend import users as users_mod

    seen: list[str] = []
    real_verify = users_mod.verify_password

    def spy(password: str, password_hash: str) -> bool:
        seen.append(password_hash)
        return real_verify(password, password_hash)

    monkeypatch.setattr(users_mod, "verify_password", spy)

    resp = client.post(
        "/api/auth/login",
        json={"email": "no-such-account@example.com", "password": "whatever1"},
    )
    assert resp.status_code == 401
    assert len(seen) == 1, "an unknown email must still pay one bcrypt compare"
    # Same cost factor as a stored password, or the two paths diverge again.
    assert seen[0].startswith("$2b$")
    assert seen[0].split("$")[2] == str(users_mod.BCRYPT_ROUNDS)


def test_weak_password_signup_does_not_consume_the_signup_budget(
    client, auth_rate_limit_clock
):
    """Policy first, budget second: iterating on a rejected password creates
    nothing, so it must not spend the allowance for the account being created."""
    for _ in range(4):
        rejected = client.post(
            "/api/auth/signup",
            json={"email": "weak@example.com", "display_name": "W", "password": "short"},
        )
        assert rejected.status_code == 400

    # The fixture's signup IP budget is 2 and is still fully available.
    for email in ("real1@example.com", "real2@example.com"):
        assert (
            client.post(
                "/api/auth/signup",
                json={
                    "email": email,
                    "display_name": "Real",
                    "password": "securepass1",
                },
            ).status_code
            == 200
        )


def test_email_with_interior_control_characters_is_rejected(client, capsys):
    """The address never gets to hold a newline in the first place.

    ``_normalize_email`` used to ``strip()`` only the *ends*, so an interior
    newline validated and was stored verbatim — and everything that later
    renders an address as plain text inherited it. The log line below is one
    sink; the admin console is the worse one, where the role-change prompt is
    built as ``Promote {email} to admin?\\n\\nThey will see Admin…`` and a
    native confirm() dialog has no markup to escape. Reject at the validator,
    which covers every sink at once rather than one guard per renderer.
    """
    forged = "auth.login_failed domain=attacker.test"
    for address in (
        f"victim@example.com\n{forged}",
        "victim@example.com\r\nX: 1",
        "vic tim@example.com",
        "victim@exam\x00ple.com",
    ):
        resp = client.post(
            "/api/auth/login", json={"email": address, "password": "whatever1"}
        )
        assert resp.status_code == 422, f"{address!r} -> {resp.status_code}"
        signup = client.post(
            "/api/auth/signup",
            json={
                "email": address,
                "display_name": "V",
                "password": "SecurePass1!",
            },
        )
        assert signup.status_code == 422, f"{address!r} -> {signup.status_code}"

    out = capsys.readouterr().out
    assert forged not in out
    assert "attacker.test" not in out


def test_email_domain_truncates_injected_text(capsys):
    """Second guard, kept: the log line must hold even if a bad address exists.

    ``_normalize_email`` is the fix; this is the belt to its braces, for rows
    that predate the rule or arrive through some future path that skips it.
    CodeQL reported the original as py/log-injection while these were
    ``logger`` calls and goes quiet at a ``print`` sink it does not model — the
    alert going away is not what makes it safe, this is.
    """
    from dashboard.backend.api.auth import _email_domain

    forged = "example.com\nauth.login_failed domain=attacker.test"
    assert _email_domain(f"victim@{forged}") == "example.com"
    assert _email_domain("victim@" + "a" * 200).endswith("a")
    assert len(_email_domain("victim@" + "a" * 200)) == 64
    assert _email_domain("no-at-sign") == "no-at-sign"


def test_env_int_disables_on_zero_and_reports_bad_overrides(monkeypatch, capsys):
    """0 disables (the MAX_ACTIVE_RUNS_GLOBAL convention) and junk is loud.

    capsys, not caplog: logger output is invisible under the deployed uvicorn
    config, so these warnings go to stdout.
    """
    from dashboard.backend.api import auth as auth_api

    monkeypatch.setenv("AUTH_FAKE_MAX", "0")
    assert auth_api._env_int("AUTH_FAKE_MAX", 30) == 0

    monkeypatch.setenv("AUTH_FAKE_MAX", "thirty")
    assert auth_api._env_int("AUTH_FAKE_MAX", 30) == 30
    assert "not an integer" in capsys.readouterr().out

    monkeypatch.setenv("AUTH_FAKE_MAX", "-1")
    assert auth_api._env_int("AUTH_FAKE_MAX", 30) == 30
    assert "below the minimum" in capsys.readouterr().out

    # A window of 0 is not a setting anyone means, so counts and windows differ.
    monkeypatch.setenv("AUTH_FAKE_WINDOW", "0")
    assert auth_api._env_int("AUTH_FAKE_WINDOW", 900, minimum=1) == 900

    monkeypatch.delenv("AUTH_FAKE_MAX")
    assert auth_api._env_int("AUTH_FAKE_MAX", 30) == 30


# ---------------------------------------------------------------------------
# Session client context (auth_sessions.user_agent / ip_prefix)
# ---------------------------------------------------------------------------


def _only_session(store):
    conn = store._get_connection()
    try:
        conn.row_factory = None
        return conn.execute(
            "SELECT user_agent, ip_prefix FROM auth_sessions"
        ).fetchall()
    finally:
        conn.close()


@pytest.mark.parametrize("route", ["signup", "login"])
def test_a_session_records_the_client_it_was_issued_to(client, temp_user_store, route):
    """The columns exist to answer "what are my signed-in devices?".

    Storing NULL in both makes the schema describe a feature that isn't there,
    and the gap only surfaces once someone builds the session list on top.
    """
    payload = {
        "email": "context@example.com",
        "display_name": "Context",
        "password": "securepass1",
    }
    headers = {"User-Agent": "AtlTest/1.0", "X-Forwarded-For": "203.0.113.42"}
    signup = client.post("/api/auth/signup", json=payload, headers=headers)
    assert signup.status_code == 200
    if route == "login":
        client.post(
            "/api/auth/login",
            json={"email": payload["email"], "password": payload["password"]},
            headers=headers,
        )

    rows = _only_session(temp_user_store)
    assert rows, "no session row was written"
    assert all(row[0] == "AtlTest/1.0" for row in rows)
    assert all(row[1] == "203.0.113.0/24" for row in rows)


def test_the_stored_address_is_a_network_not_the_client(client, temp_user_store):
    """ip_prefix, not ip. Storing the exact address turns a session table into
    a location log for every signed-in user, which is not what it is for."""
    client.post(
        "/api/auth/signup",
        json={
            "email": "coarse@example.com",
            "display_name": "Coarse",
            "password": "securepass1",
        },
        headers={"X-Forwarded-For": "198.51.100.77"},
    )
    stored = _only_session(temp_user_store)[0][1]
    assert "198.51.100.77" not in stored
    assert stored == "198.51.100.0/24"


# ---------------------------------------------------------------------------
# Cookie mechanics that only show up on the raw Set-Cookie header. TestClient's
# jar silently drops Secure cookies over http://testserver, so jar-based
# assertions cannot see the production (__Host-) branch at all — these tests
# read response.headers directly.
# ---------------------------------------------------------------------------


def test_prod_logout_deletes_the_host_cookie_with_secure(client, monkeypatch):
    """RFC 6265bis: a UA rejects any __Host-* Set-Cookie without Secure.

    delete_cookie defaults secure=False, so an unflagged deletion leaves the
    prod cookie alive in the browser after logout (revoked server-side, but
    still sent on every request). Guard the exact failure: the deletion for
    the __Host- name must itself carry Secure.
    """
    monkeypatch.setenv("ATL_COOKIE_SECURE", "true")
    response = client.post("/api/auth/logout")
    assert response.status_code == 200
    host_deletions = [
        value
        for value in response.headers.get_list("set-cookie")
        if value.startswith('__Host-atl_session="";') or value.startswith("__Host-atl_session=;")
    ]
    assert host_deletions, response.headers.get_list("set-cookie")
    for header in host_deletions:
        assert "Secure" in header
        assert "HttpOnly" in header


def test_prod_login_sets_secure_httponly_cookie_header(client, monkeypatch):
    monkeypatch.setenv("ATL_COOKIE_SECURE", "true")
    client.post(
        "/api/auth/signup",
        json={
            "email": "prodcookie@example.com",
            "display_name": "Prod",
            "password": "securepass1",
        },
    )
    set_cookie = [
        value
        for value in client.post(
            "/api/auth/login",
            json={"email": "prodcookie@example.com", "password": "securepass1"},
        ).headers.get_list("set-cookie")
        if value.startswith("__Host-atl_session=")
    ]
    assert set_cookie, "login did not set the __Host- session cookie"
    header = set_cookie[0]
    assert "Secure" in header and "HttpOnly" in header and "Path=/" in header
    assert "Domain" not in header  # __Host- forbids a Domain attribute


def test_me_upgrades_a_legacy_bearer_session_to_a_cookie(client):
    """Migration bridge: pre-cookie sessions live only in localStorage.

    app.js sends that token once as Bearer on the boot /me probe; the response
    must carry Set-Cookie so the session survives the HttpOnly migration
    instead of force-logging the user out on deploy.
    """
    client.post(
        "/api/auth/signup",
        json={
            "email": "bridge@example.com",
            "display_name": "Bridge",
            "password": "securepass1",
        },
    )
    token = _session_token(client)
    client.cookies.clear()  # simulate a browser that never got the cookie
    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    set_cookie = response.headers.get_list("set-cookie")
    assert any("atl_session=" in value for value in set_cookie), set_cookie

    # And once the cookie is present, /me must NOT keep re-setting it.
    again = client.get("/api/auth/me")
    assert again.status_code == 200
    assert not again.headers.get_list("set-cookie")


def test_set_session_cookie_rejects_malformed_tokens():
    """The /me bridge feeds set_session_cookie a client-supplied Bearer value;
    anything outside the token_urlsafe alphabet must never reach Set-Cookie."""
    from fastapi import Response

    from dashboard.backend.auth_cookies import set_session_cookie

    response = Response()
    set_session_cookie(response, "evil;\r\nSet-Cookie: hijack=1")
    assert "set-cookie" not in response.headers

    response = Response()
    set_session_cookie(response, "x" * 43)  # well-formed shape still works
    assert "set-cookie" in response.headers


# --- Password reset (#187) ---------------------------------------------------


def _reset_code_from(email_body):
    return _code_after(email_body, "reset code:")


@pytest.fixture
def reset_outbox(sent_emails, monkeypatch):
    """sent_emails plus the Brevo config forgot-password checks up front.

    The route 503s on email_configured() before any account work, and that
    reads the env at call time -- the send itself stays the patched fake.
    """
    monkeypatch.setenv("BREVO_API_KEY", "test-key")
    monkeypatch.setenv("ACCOUNT_EMAIL_FROM", "noreply@example.com")
    return sent_emails


def _widen_forgot_limiters(monkeypatch):
    """Push the in-process forgot limiters out of the way so the DB-backed
    cooldown / daily cap is what a test exercises."""
    from dashboard.backend.api import auth as auth_api
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter

    for name in ("_FORGOT_IP_LIMITER", "_FORGOT_EMAIL_LIMITER", "_FORGOT_GLOBAL_LIMITER"):
        monkeypatch.setattr(
            auth_api, name, FixedWindowRateLimiter(max_events=1000, window_seconds=60)
        )


def _request_reset(client, email):
    response = client.post("/api/auth/forgot-password", json={"email": email})
    assert response.status_code == 200, response.text
    return response


def _reset_password(client, email, code, new_password):
    return client.post(
        "/api/auth/reset-password",
        json={"email": email, "code": code, "new_password": new_password},
    )


def _reset_row_count(store) -> int:
    conn = store._get_connection()
    row = conn.execute("SELECT COUNT(*) AS n FROM password_reset_requests").fetchone()
    conn.close()
    return int(row["n"])


def _backdate_password_reset_rows(store, user_id, **columns):
    _backdate_rows(store, "password_reset_requests", user_id, **columns)


def test_forgot_password_mails_a_code_to_a_known_account(client, reset_outbox):
    _signup_and_token(client, email="resetme@example.com")

    response = _request_reset(client, "resetme@example.com")

    assert response.json() == {"status": "ok"}
    assert len(reset_outbox) == 1
    assert reset_outbox[0]["to"] == "resetme@example.com"
    assert "reset code:" in reset_outbox[0]["body"]
    assert "spam" not in reset_outbox[0]["subject"].lower()


def test_forgot_password_is_enumeration_blind(client, reset_outbox, temp_user_store):
    _signup_and_token(client, email="known-reset@example.com")

    known = client.post(
        "/api/auth/forgot-password", json={"email": "known-reset@example.com"}
    )
    unknown = client.post(
        "/api/auth/forgot-password", json={"email": "nobody-reset@example.com"}
    )

    # Byte-identical: neither status nor body says whether the account exists.
    assert known.status_code == unknown.status_code == 200
    assert known.content == unknown.content
    # The unknown address got no mail and wrote no row.
    assert len(reset_outbox) == 1
    assert _reset_row_count(temp_user_store) == 1


def test_forgot_password_503s_when_brevo_is_unconfigured(client, monkeypatch, capsys):
    monkeypatch.delenv("BREVO_API_KEY", raising=False)
    monkeypatch.delenv("ACCOUNT_EMAIL_FROM", raising=False)
    _signup_and_token(client, email="unconfigured@example.com")

    known = client.post(
        "/api/auth/forgot-password", json={"email": "unconfigured@example.com"}
    )
    unknown = client.post(
        "/api/auth/forgot-password", json={"email": "ghost-unconfigured@example.com"}
    )

    # Fail-visible, and identically for known and unknown addresses: the 503
    # is config-shaped information, not account-shaped.
    assert known.status_code == unknown.status_code == 503
    assert known.content == unknown.content
    assert "BREVO_API_KEY" in capsys.readouterr().out


def test_forgot_password_send_failure_persists_nothing(
    client, reset_outbox, temp_user_store, capsys
):
    _signup_and_token(client, email="flaky-reset@example.com")
    reset_outbox.fail_sends()

    _request_reset(client, "flaky-reset@example.com")

    # Send-before-persist: the failed send burned no cooldown.
    assert _reset_row_count(temp_user_store) == 0
    assert "send failed" in capsys.readouterr().out

    reset_outbox.resume_sends()
    _request_reset(client, "flaky-reset@example.com")
    assert _reset_row_count(temp_user_store) == 1


def test_forgot_password_cooldown_blocks_a_second_send(
    client, reset_outbox, temp_user_store, capsys, monkeypatch
):
    _widen_forgot_limiters(monkeypatch)
    _signup_and_token(client, email="cool-reset@example.com")

    _request_reset(client, "cool-reset@example.com")
    _request_reset(client, "cool-reset@example.com")  # still 200: silent skip

    assert len(reset_outbox) == 1
    assert "reason=cooldown" in capsys.readouterr().out

    user = temp_user_store.get_user_by_email("cool-reset@example.com")
    _backdate_password_reset_rows(
        temp_user_store, user["id"], created_at=_stored_time(minutes=6)
    )
    _request_reset(client, "cool-reset@example.com")
    assert len(reset_outbox) == 2


def test_forgot_password_cooldown_is_status_blind(
    client, reset_outbox, temp_user_store, capsys, monkeypatch
):
    # A cancelled row still gates the cooldown, so cancelling can never be
    # used to mint codes faster.
    _widen_forgot_limiters(monkeypatch)
    _signup_and_token(client, email="cancelled-reset@example.com")
    _request_reset(client, "cancelled-reset@example.com")
    user = temp_user_store.get_user_by_email("cancelled-reset@example.com")
    temp_user_store.cancel_password_reset(user["id"])

    _request_reset(client, "cancelled-reset@example.com")

    assert len(reset_outbox) == 1
    assert "reason=cooldown" in capsys.readouterr().out


def test_forgot_password_requests_are_capped_per_day(
    client, reset_outbox, temp_user_store, capsys, monkeypatch
):
    from dashboard.backend.users import PASSWORD_RESET_MAX_REQUESTS_PER_DAY

    _widen_forgot_limiters(monkeypatch)
    _signup_and_token(client, email="capped-reset@example.com")
    user = temp_user_store.get_user_by_email("capped-reset@example.com")

    for _ in range(PASSWORD_RESET_MAX_REQUESTS_PER_DAY):
        _request_reset(client, "capped-reset@example.com")
        # Step past the 5-minute cooldown; every row stays inside the 24h window.
        _backdate_password_reset_rows(
            temp_user_store, user["id"], created_at=_stored_time(minutes=10)
        )
    assert len(reset_outbox) == PASSWORD_RESET_MAX_REQUESTS_PER_DAY

    _request_reset(client, "capped-reset@example.com")

    assert len(reset_outbox) == PASSWORD_RESET_MAX_REQUESTS_PER_DAY
    assert "reason=daily_cap" in capsys.readouterr().out


def test_forgot_password_global_send_budget(client, reset_outbox, monkeypatch, capsys):
    from dashboard.backend.api import auth as auth_api
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter

    monkeypatch.setattr(
        auth_api,
        "_FORGOT_GLOBAL_LIMITER",
        FixedWindowRateLimiter(max_events=1, window_seconds=3600),
    )
    _signup_and_token(client, email="global-a@example.com")
    _signup_and_token(client, email="global-b@example.com")

    _request_reset(client, "global-a@example.com")
    _request_reset(client, "global-b@example.com")  # over budget: 200, no mail

    assert len(reset_outbox) == 1
    assert "reason=global_cap" in capsys.readouterr().out


def test_password_reset_happy_path(client, reset_outbox):
    _signup_and_token(client, email="happy-reset@example.com")
    _request_reset(client, "happy-reset@example.com")
    code = _reset_code_from(reset_outbox[0]["body"])

    done = _reset_password(client, "happy-reset@example.com", code, "fresh-sturdy-pw-3")

    assert done.status_code == 200, done.text
    assert done.json() == {"status": "ok"}
    old = client.post(
        "/api/auth/login",
        json={"email": "happy-reset@example.com", "password": "orig-sturdy-pw-1"},
    )
    assert old.status_code == 401
    new = client.post(
        "/api/auth/login",
        json={"email": "happy-reset@example.com", "password": "fresh-sturdy-pw-3"},
    )
    assert new.status_code == 200


def test_reset_password_failures_are_uniform(client, reset_outbox):
    _signup_and_token(client, email="uniform-reset@example.com")

    # A real account with no active request vs an unknown account: one 400.
    known = _reset_password(
        client, "uniform-reset@example.com", "ABC234", "fresh-sturdy-pw-3"
    )
    unknown = _reset_password(
        client, "ghost-uniform@example.com", "ABC234", "fresh-sturdy-pw-3"
    )

    assert known.status_code == unknown.status_code == 400
    assert known.content == unknown.content
    assert known.json()["detail"] == "Invalid or expired code."


def test_reset_password_gives_up_after_five_wrong_codes(client, reset_outbox):
    from dashboard.backend.users import PASSWORD_RESET_MAX_ATTEMPTS

    _signup_and_token(client, email="attempts-reset@example.com")
    _request_reset(client, "attempts-reset@example.com")
    real = _reset_code_from(reset_outbox[0]["body"])
    wrong = "ZZZZZZ" if real != "ZZZZZZ" else "YYYYYY"

    for _ in range(PASSWORD_RESET_MAX_ATTEMPTS):
        attempt = _reset_password(
            client, "attempts-reset@example.com", wrong, "fresh-sturdy-pw-3"
        )
        assert attempt.status_code == 400

    # The request cancelled itself at the cap: even the correct code is dead.
    after = _reset_password(
        client, "attempts-reset@example.com", real, "fresh-sturdy-pw-3"
    )
    assert after.status_code == 400
    assert after.json()["detail"] == "Invalid or expired code."


def test_reset_password_rejects_an_expired_code(client, reset_outbox, temp_user_store):
    _signup_and_token(client, email="expired-reset@example.com")
    _request_reset(client, "expired-reset@example.com")
    code = _reset_code_from(reset_outbox[0]["body"])
    user = temp_user_store.get_user_by_email("expired-reset@example.com")
    _backdate_password_reset_rows(
        temp_user_store, user["id"], expires_at=_stored_time(minutes=1)
    )

    response = _reset_password(
        client, "expired-reset@example.com", code, "fresh-sturdy-pw-3"
    )

    # The store folds expiry into "no active row"; same generic 400.
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or expired code."


def test_reset_password_code_is_single_use(client, reset_outbox):
    _signup_and_token(client, email="once-reset@example.com")
    _request_reset(client, "once-reset@example.com")
    code = _reset_code_from(reset_outbox[0]["body"])

    first = _reset_password(client, "once-reset@example.com", code, "fresh-sturdy-pw-3")
    second = _reset_password(client, "once-reset@example.com", code, "other-sturdy-pw-4")

    assert first.status_code == 200
    assert second.status_code == 400


def test_reset_password_code_is_case_and_whitespace_insensitive(client, reset_outbox):
    _signup_and_token(client, email="lower-reset@example.com")
    _request_reset(client, "lower-reset@example.com")
    code = _reset_code_from(reset_outbox[0]["body"])

    response = _reset_password(
        client, "lower-reset@example.com", f" {code.lower()} ", "fresh-sturdy-pw-3"
    )

    assert response.status_code == 200


def test_reset_password_weak_new_password_keeps_the_code_alive(client, reset_outbox):
    _signup_and_token(client, email="weakpw-reset@example.com")
    _request_reset(client, "weakpw-reset@example.com")
    code = _reset_code_from(reset_outbox[0]["body"])

    weak = _reset_password(client, "weakpw-reset@example.com", code, "password1")

    assert weak.status_code == 400
    assert "too common" in weak.json()["detail"]

    # The request row is untouched: the same still-valid code works with a
    # better password.
    retry = _reset_password(
        client, "weakpw-reset@example.com", code, "fresh-sturdy-pw-3"
    )
    assert retry.status_code == 200


def test_reset_revokes_all_sessions_and_cancels_email_change(client, reset_outbox):
    token = _signup_and_token(client, email="hijacked@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    _start_email_change(client, token)  # mail #0
    _request_reset(client, "hijacked@example.com")  # mail #1
    code = _reset_code_from(reset_outbox[1]["body"])

    done = _reset_password(client, "hijacked@example.com", code, "fresh-sturdy-pw-3")
    assert done.status_code == 200

    # ALL sessions die -- the caller proved an inbox, not a login.
    assert client.get("/api/auth/me", headers=headers).status_code == 401

    # ...and the pending email change died with them (D7).
    login = client.post(
        "/api/auth/login",
        json={"email": "hijacked@example.com", "password": "fresh-sturdy-pw-3"},
    )
    assert login.status_code == 200
    fresh = _session_token(client)
    status = client.get(
        "/api/auth/email-change", headers={"Authorization": f"Bearer {fresh}"}
    )
    assert status.json()["pending"] is False


def test_reset_revocation_failure_still_succeeds(
    client, reset_outbox, monkeypatch, capsys
):
    _signup_and_token(client, email="besteffort-reset@example.com")
    _request_reset(client, "besteffort-reset@example.com")
    code = _reset_code_from(reset_outbox[0]["body"])

    def _boom(*args, **kwargs):
        raise RuntimeError("session store unavailable")

    monkeypatch.setattr(UserStore, "delete_other_sessions", _boom)

    done = _reset_password(
        client, "besteffort-reset@example.com", code, "fresh-sturdy-pw-3"
    )

    # The durable write already landed; revocation failure is a WARNING, not a 500.
    assert done.status_code == 200
    new_login = client.post(
        "/api/auth/login",
        json={"email": "besteffort-reset@example.com", "password": "fresh-sturdy-pw-3"},
    )
    assert new_login.status_code == 200
    assert "revocation failed" in capsys.readouterr().out


def test_change_password_cancels_a_pending_reset_end_to_end(client, reset_outbox):
    token = _signup_and_token(client, email="crossflow-reset@example.com")
    _request_reset(client, "crossflow-reset@example.com")
    code = _reset_code_from(reset_outbox[0]["body"])

    changed = client.post(
        "/api/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_password": "new-sturdy-pw-2"},
    )
    assert changed.status_code == 200

    # Not merely that a cancel call fired: the code actually fails redemption.
    stale = _reset_password(
        client, "crossflow-reset@example.com", code, "fresh-sturdy-pw-3"
    )
    assert stale.status_code == 400


def test_email_change_commit_cancels_a_pending_reset_end_to_end(client, reset_outbox):
    token = _signup_and_token(client, email="movedaway@example.com")
    _request_reset(client, "movedaway@example.com")
    reset_code = _reset_code_from(reset_outbox[0]["body"])

    _complete_email_change(client, token, reset_outbox, "newhome@example.com")

    # The code was mailed to the OLD address; it must not survive the change,
    # under either the old or the new login handle.
    stale_old = _reset_password(
        client, "movedaway@example.com", reset_code, "fresh-sturdy-pw-3"
    )
    stale_new = _reset_password(
        client, "newhome@example.com", reset_code, "fresh-sturdy-pw-3"
    )
    assert stale_old.status_code == 400
    assert stale_new.status_code == 400


def test_forgot_password_ip_rate_limit_returns_429(client, monkeypatch):
    from dashboard.backend.api import auth as auth_api
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter

    monkeypatch.setenv("BREVO_API_KEY", "test-key")
    monkeypatch.setenv("ACCOUNT_EMAIL_FROM", "noreply@example.com")
    monkeypatch.setattr(
        auth_api,
        "_FORGOT_IP_LIMITER",
        FixedWindowRateLimiter(max_events=2, window_seconds=60),
    )

    for _ in range(2):
        ok = client.post(
            "/api/auth/forgot-password", json={"email": "rl-forgot@example.com"}
        )
        assert ok.status_code == 200
    blocked = client.post(
        "/api/auth/forgot-password", json={"email": "rl-forgot@example.com"}
    )

    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["Retry-After"]) <= 60


def test_reset_password_ip_rate_limit_returns_429(client, monkeypatch):
    from dashboard.backend.api import auth as auth_api
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter

    monkeypatch.setattr(
        auth_api,
        "_RESET_IP_LIMITER",
        FixedWindowRateLimiter(max_events=1, window_seconds=60),
    )

    first = _reset_password(client, "rl-reset@example.com", "ZZZZZZ", "fresh-sturdy-pw-3")
    assert first.status_code == 400  # failure charged the budget
    blocked = _reset_password(client, "rl-reset@example.com", "ZZZZZZ", "fresh-sturdy-pw-3")

    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["Retry-After"]) <= 60


def test_reset_password_per_email_limit_holds_across_client_keys(client, monkeypatch):
    # Rotating X-Browser-Id buys a fresh per-client budget, so the per-email
    # failure budget is the control that actually bounds code guessing; a dead
    # per-email budget cannot pass this.
    from dashboard.backend.api import auth as auth_api
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter

    monkeypatch.setattr(
        auth_api,
        "_RESET_EMAIL_LIMITER",
        FixedWindowRateLimiter(max_events=2, window_seconds=60),
    )
    _signup_and_token(client, email="hammered@example.com")

    for i in range(2):
        response = client.post(
            "/api/auth/reset-password",
            json={
                "email": "hammered@example.com",
                "code": "ZZZZZZ",
                "new_password": "fresh-sturdy-pw-3",
            },
            headers={"X-Browser-Id": f"attacker-{i}"},
        )
        assert response.status_code == 400
    blocked = client.post(
        "/api/auth/reset-password",
        json={
            "email": "hammered@example.com",
            "code": "ZZZZZZ",
            "new_password": "fresh-sturdy-pw-3",
        },
        headers={"X-Browser-Id": "attacker-fresh"},
    )

    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["Retry-After"]) <= 60
