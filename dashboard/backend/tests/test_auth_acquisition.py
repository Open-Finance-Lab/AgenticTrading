"""Signup acquisition capture is verified, private, and best effort."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import dashboard.backend.users as users_module
from dashboard.backend.app import app
from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.credits.service import CreditsService
from dashboard.backend.users import UserStore


SIGNING_KEY = "synthetic-signup-attribution-key"


class RecordingAttributionStore:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.records = []

    def record_initial_attribution(self, attribution):
        if self.fail:
            raise RuntimeError("synthetic storage failure")
        self.records.append(attribution)
        return attribution


def _signed_invite(*, source: str, cohort: str) -> str:
    issued_at = datetime.now(timezone.utc)
    payload = {
        "source": source,
        "cohort": cohort,
        "iat": int(issued_at.timestamp()),
        "exp": int((issued_at + timedelta(days=7)).timestamp()),
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    signature = hmac.new(
        SIGNING_KEY.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
    ).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"{encoded}.{encoded_signature}"


@pytest.fixture
def signup_api(tmp_path, monkeypatch):
    from dashboard.backend.api import auth

    db_path = tmp_path / "signup-acquisition.db"
    users = UserStore(db_path=db_path)
    attributions = RecordingAttributionStore()
    monkeypatch.setattr(users_module, "user_store", users)
    monkeypatch.setattr(
        auth,
        "credits_service",
        CreditsService(store=CreditsStore(db_path)),
    )
    monkeypatch.setattr(auth, "analytics_store", attributions)
    monkeypatch.setattr(
        auth,
        "agent_service",
        SimpleNamespace(provision_starter_agents=lambda **_kwargs: []),
    )
    monkeypatch.setattr(
        auth.analytics_instrumentation, "emit_account_event", lambda **_kwargs: None
    )
    monkeypatch.setenv("ACQUISITION_INVITE_SIGNING_KEY", SIGNING_KEY)
    with TestClient(app) as client:
        yield client, attributions


def _signup(client: TestClient, email: str, invite_token: str | None = None):
    payload = {
        "email": email,
        "display_name": "Acquisition User",
        "password": "SecurePass1!",
    }
    if invite_token is not None:
        payload["invite_token"] = invite_token
    return client.post("/api/auth/signup", json=payload)


def test_valid_signed_invite_is_saved_without_persisting_token(signup_api):
    client, store = signup_api
    token = _signed_invite(source="student", cohort="fall-course")

    response = _signup(client, "valid-invite@example.test", token)

    assert response.status_code == 200, response.text
    assert len(store.records) == 1
    attribution = store.records[0]
    assert attribution.user_id == response.json()["user"]["id"]
    assert attribution.source == "student"
    assert attribution.cohort == "fall-course"
    assert attribution.method == "invite"
    assert token not in repr(store.records)
    assert "invite_token" not in response.text


@pytest.mark.parametrize("token", [None, "tampered.token"])
def test_missing_or_tampered_invite_is_recorded_as_unknown(signup_api, token):
    client, store = signup_api

    response = _signup(client, f"unknown-{len(store.records)}@example.test", token)

    assert response.status_code == 200, response.text
    assert store.records[-1].source == "unknown"
    assert store.records[-1].cohort is None
    assert store.records[-1].method == "unknown"


def test_attribution_storage_failure_does_not_fail_signup(signup_api, monkeypatch):
    client, _store = signup_api
    from dashboard.backend.api import auth

    monkeypatch.setattr(auth, "analytics_store", RecordingAttributionStore(fail=True))
    response = _signup(
        client,
        "storage-failure@example.test",
        _signed_invite(source="community", cohort="launch-2026"),
    )

    assert response.status_code == 200, response.text
