"""Admin-only provider registry and platform credential API contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

import dashboard.backend.users as users_module
from dashboard.backend.app import app
from dashboard.backend.domain.brokers import repository as broker_repository
from dashboard.backend.domain.model_providers.models import CredentialValidation
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.model_providers.service import (
    ModelProviderService,
    get_model_provider_service,
)


class FakeAdapter:
    def __init__(self):
        self.results: list[CredentialValidation] = []
        self.calls: list[tuple[str, str]] = []

    def queue(self, status: str) -> None:
        self.results.append(
            CredentialValidation(status=status, message="Safe fake-provider result.")
        )

    def validate(self, base_url: str, secret: str, *, client=None):
        self.calls.append((base_url, secret))
        if not self.results:
            raise AssertionError("test did not queue a validation result")
        return self.results.pop(0)


@pytest.fixture
def admin_provider_api(tmp_path, monkeypatch):
    from dashboard.backend.api.routers import admin_model_providers as admin_router

    database_path = tmp_path / "admin-model-providers.db"
    users = users_module.UserStore(database_path)
    store = ModelProviderStore(database_path)
    adapter = FakeAdapter()
    service = ModelProviderService(
        store=store,
        adapter_resolver=lambda _adapter_type: adapter,
    )
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)
    monkeypatch.setattr(users_module, "user_store", users)
    app.dependency_overrides[get_model_provider_service] = lambda: service
    admin_router.reset_admin_model_provider_limiter()
    yield SimpleNamespace(
        client=TestClient(app),
        users=users,
        store=store,
        adapter=adapter,
        router=admin_router,
    )
    admin_router.reset_admin_model_provider_limiter()
    app.dependency_overrides.pop(get_model_provider_service, None)


def _signup(client: TestClient, email: str) -> dict:
    response = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "display_name": email.split("@", 1)[0],
            "password": "SecurePass1!",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["user"]


def _provider_payload(**overrides) -> dict:
    payload = {
        "display_name": "Approved Compatible",
        "adapter_type": "openai_compatible",
        "approved_base_url": "https://models.example.com/v1",
        "capabilities": {"model_discovery": True},
        "byok_enabled": True,
        "platform_enabled": True,
        "status": "enabled",
        "source": "admin-console",
        "reason": "Approved after provider review.",
        "idempotency_key": "provider-approval-api-001",
    }
    payload.update(overrides)
    return payload


def _credential_payload(**overrides) -> dict:
    payload = {
        "api_key": "sk-platform-fake-never-return-abcd",
        "source": "admin-console",
        "reason": "Configure ATL-funded model access.",
        "idempotency_key": "platform-credential-api-001",
    }
    payload.update(overrides)
    return payload


def _action_payload(**overrides) -> dict:
    payload = {
        "source": "admin-console",
        "reason": "Rotate or recheck the platform credential.",
        "idempotency_key": "platform-action-api-001",
    }
    payload.update(overrides)
    return payload


def _promote(api, user_id: int) -> None:
    api.users.apply_admin_patch(user_id, role="admin")


def test_every_admin_provider_route_refuses_regular_users(admin_provider_api):
    _signup(admin_provider_api.client, "regular-provider-user@example.com")
    calls = (
        lambda: admin_provider_api.client.get("/api/admin/model-providers"),
        lambda: admin_provider_api.client.put(
            "/api/admin/model-providers/approved_compatible",
            json=_provider_payload(),
        ),
        lambda: admin_provider_api.client.put(
            "/api/admin/model-providers/openrouter/platform-credential",
            json=_credential_payload(),
        ),
        lambda: admin_provider_api.client.post(
            "/api/admin/model-providers/openrouter/platform-credential/verify",
            json=_action_payload(),
        ),
        lambda: admin_provider_api.client.request(
            "DELETE",
            "/api/admin/model-providers/openrouter/platform-credential",
            json=_action_payload(),
        ),
    )

    assert [call().status_code for call in calls] == [403, 403, 403, 403, 403]


def test_admin_approves_compatible_provider_and_list_exposes_safe_state(
    admin_provider_api,
):
    admin = _signup(admin_provider_api.client, "provider-admin@example.com")
    _promote(admin_provider_api, admin["id"])

    created = admin_provider_api.client.put(
        "/api/admin/model-providers/approved_compatible",
        json=_provider_payload(),
    )
    listed = admin_provider_api.client.get("/api/admin/model-providers")

    assert created.status_code == 200, created.text
    assert created.json()["provider"]["provider_id"] == "approved_compatible"
    assert created.json()["provider"]["approved_base_url"] == (
        "https://models.example.com/v1"
    )
    listed_item = next(
        item
        for item in listed.json()["providers"]
        if item["provider_id"] == "approved_compatible"
    )
    assert listed_item["platform_credential"] is None


@pytest.mark.parametrize(
    "base_url",
    [
        "http://models.example.com/v1",
        "https://127.0.0.1/v1",
        "https://models.example.com/v1?key=secret",
        "https://user:pass@models.example.com/v1",
    ],
)
def test_admin_provider_api_rejects_unapproved_origins(
    admin_provider_api, base_url
):
    admin = _signup(admin_provider_api.client, f"origin-{abs(hash(base_url))}@example.com")
    _promote(admin_provider_api, admin["id"])

    response = admin_provider_api.client.put(
        "/api/admin/model-providers/approved_compatible",
        json=_provider_payload(approved_base_url=base_url),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid provider configuration."}
    assert "secret" not in response.text


def test_admin_provider_api_rejects_invalid_provider_id(admin_provider_api):
    admin = _signup(admin_provider_api.client, "provider-id-admin@example.com")
    _promote(admin_provider_api, admin["id"])

    response = admin_provider_api.client.put(
        "/api/admin/model-providers/Invalid-ID",
        json=_provider_payload(),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid provider configuration."}


def test_platform_credential_lifecycle_never_returns_plaintext(
    admin_provider_api, caplog
):
    admin = _signup(admin_provider_api.client, "credential-admin@example.com")
    _promote(admin_provider_api, admin["id"])
    secret = "sk-platform-fake-never-return-abcd"
    admin_provider_api.adapter.queue("verification_unavailable")
    admin_provider_api.adapter.queue("verified")

    created = admin_provider_api.client.put(
        "/api/admin/model-providers/openrouter/platform-credential",
        json=_credential_payload(api_key=secret),
    )
    verified = admin_provider_api.client.post(
        "/api/admin/model-providers/openrouter/platform-credential/verify",
        json=_action_payload(idempotency_key="platform-verify-api-001"),
    )
    listed = admin_provider_api.client.get("/api/admin/model-providers")

    combined = created.text + verified.text + listed.text + caplog.text
    assert created.status_code == 200, created.text
    assert created.json()["platform_credential"]["status"] == (
        "verification_unavailable"
    )
    assert verified.json()["platform_credential"]["status"] == "verified"
    assert verified.json()["platform_credential"]["key_last_four"] == "abcd"
    assert secret not in combined
    assert "api_key" not in created.text + verified.text + listed.text

    revoked = admin_provider_api.client.request(
        "DELETE",
        "/api/admin/model-providers/openrouter/platform-credential",
        json=_action_payload(idempotency_key="platform-revoke-api-001"),
    )
    assert revoked.status_code == 200
    assert revoked.json() == {"revoked": True}


def test_platform_credential_api_retry_does_not_verify_twice(admin_provider_api):
    admin = _signup(admin_provider_api.client, "idempotent-admin@example.com")
    _promote(admin_provider_api, admin["id"])
    admin_provider_api.adapter.queue("verified")
    payload = _credential_payload(idempotency_key="platform-idempotent-api-001")

    first = admin_provider_api.client.put(
        "/api/admin/model-providers/openrouter/platform-credential",
        json=payload,
    )
    replayed = admin_provider_api.client.put(
        "/api/admin/model-providers/openrouter/platform-credential",
        json=payload,
    )

    assert first.status_code == 200
    assert replayed.status_code == 200
    assert replayed.json() == first.json()
    assert len(admin_provider_api.adapter.calls) == 1


def test_invalid_platform_credential_payload_does_not_echo_secret(
    admin_provider_api,
):
    admin = _signup(admin_provider_api.client, "invalid-credential-admin@example.com")
    _promote(admin_provider_api, admin["id"])
    secret = "sk-platform-fake-invalid-payload-abcd"

    response = admin_provider_api.client.put(
        "/api/admin/model-providers/openrouter/platform-credential",
        json={**_credential_payload(api_key=secret), "status": "verified"},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid platform credential request."}
    assert secret not in response.text


def test_missing_encryption_key_fails_closed(admin_provider_api, monkeypatch):
    admin = _signup(admin_provider_api.client, "encryption-admin@example.com")
    _promote(admin_provider_api, admin["id"])
    monkeypatch.delenv("BROKER_TOKEN_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)

    response = admin_provider_api.client.put(
        "/api/admin/model-providers/openrouter/platform-credential",
        json=_credential_payload(),
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Credential encryption is unavailable."}


def test_admin_secret_canary_is_not_reflected_by_path_or_validation_errors(
    admin_provider_api,
):
    admin = _signup(admin_provider_api.client, "admin-secret-canary@example.com")
    _promote(admin_provider_api, admin["id"])
    secret = "sk-admin-url-canary-never-echo-abcd"

    invalid_path = admin_provider_api.client.put(
        f"/api/admin/model-providers/{secret}",
        json=_provider_payload(),
    )
    invalid_provider_body = admin_provider_api.client.put(
        "/api/admin/model-providers/openrouter",
        json=_provider_payload(display_name={"secret": secret}),
    )
    invalid_action_body = admin_provider_api.client.post(
        "/api/admin/model-providers/openrouter/platform-credential/verify",
        json=_action_payload(reason={"secret": secret}),
    )

    assert invalid_path.status_code == 422
    assert invalid_provider_body.status_code == 422
    assert invalid_action_body.status_code == 422
    assert secret not in invalid_path.text
    assert secret not in invalid_provider_body.text
    assert secret not in invalid_action_body.text
