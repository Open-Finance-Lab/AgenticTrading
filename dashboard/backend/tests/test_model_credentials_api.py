"""Authenticated API contract for user-owned model provider credentials."""

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
from dashboard.backend.tests.auth_cookies_helpers import _cookie_session_token


class FakeAdapter:
    def __init__(self):
        self.results: list[CredentialValidation] = []
        self.calls: list[tuple[str, str]] = []

    def queue(self, status: str) -> None:
        self.results.append(
            CredentialValidation(
                status=status,
                message="Safe fake-provider result.",
                models=["fake/model"] if status == "verified" else [],
            )
        )

    def validate(self, base_url: str, secret: str, *, client=None) -> CredentialValidation:
        self.calls.append((base_url, secret))
        if not self.results:
            raise AssertionError("test did not queue a validation result")
        return self.results.pop(0)


@pytest.fixture
def credential_api(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    database_path = tmp_path / "model-credentials-api.db"
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
    import dashboard.backend.api.routers.model_credentials as credentials_router
    credentials_router._CREDENTIAL_MUTATION_LIMITER.reset()
    yield SimpleNamespace(
        client=TestClient(app),
        users=users,
        store=store,
        adapter=adapter,
        router=credentials_router,
    )
    app.dependency_overrides.pop(get_model_provider_service, None)


def _signup(client: TestClient, email: str) -> str:
    response = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "display_name": email.split("@", 1)[0],
            "password": "securepass1",
        },
    )
    assert response.status_code == 200, response.text
    return _cookie_session_token(client)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _create(
    api,
    token: str,
    *,
    label: str = "Research",
    secret: str = "sk-or-fake-api-abcd",
    set_default: bool = False,
):
    return api.client.post(
        "/api/credits/api-keys",
        headers=_auth(token),
        json={
            "provider_id": "openrouter",
            "label": label,
            "api_key": secret,
            "set_default": set_default,
        },
    )


def test_routes_require_authentication(credential_api):
    for method, path in (
        ("get", "/api/credits/model-providers"),
        ("get", "/api/credits/execution-options"),
        ("get", "/api/credits/api-keys"),
        ("post", "/api/credits/api-keys"),
        ("post", "/api/credits/api-keys/00000000-0000-4000-8000-000000000000/verify"),
        ("post", "/api/credits/api-keys/00000000-0000-4000-8000-000000000000/default"),
        ("delete", "/api/credits/api-keys/00000000-0000-4000-8000-000000000000"),
    ):
        response = credential_api.client.request(method, path, json={})
        assert response.status_code == 401, (method, path, response.text)


def test_provider_list_contains_approved_byok_providers(credential_api):
    token = _signup(credential_api.client, "providers-api@example.com")

    response = credential_api.client.get(
        "/api/credits/model-providers", headers=_auth(token)
    )

    assert response.status_code == 200
    assert {item["provider_id"] for item in response.json()["providers"]} == {
        "anthropic",
        "gemini",
        "openai",
        "openrouter",
    }


def test_execution_options_return_safe_openrouter_models(credential_api):
    token = _signup(credential_api.client, "execution-options-api@example.com")
    secret = "sk-or-fake-execution-options-abcd"
    credential_api.adapter.queue("verified")
    created = _create(
        credential_api,
        token,
        secret=secret,
        set_default=True,
    )

    response = credential_api.client.get(
        "/api/credits/execution-options",
        headers=_auth(token),
    )

    assert created.status_code == 201
    assert response.status_code == 200
    provider = next(
        item
        for item in response.json()["providers"]
        if item["provider_id"] == "openrouter"
    )
    assert provider == {
        "provider_id": "openrouter",
        "display_name": "OpenRouter",
        "adapter_type": "openrouter",
        "byok_available": True,
        "platform_credits_available": False,
        "models": [
            {
                "model_id": "anthropic/claude-haiku-4-5",
                "label": "Claude Haiku 4.5",
            },
            {
                "model_id": "anthropic/claude-sonnet-4-6",
                "label": "Claude Sonnet 4.6",
            },
            {"model_id": "openai/gpt-5.5", "label": "GPT-5.5"},
            {
                "model_id": "google/gemini-3.1-pro-preview",
                "label": "Gemini 3.1 Pro Preview",
            },
            {
                "model_id": "deepseek/deepseek-v4-pro",
                "label": "DeepSeek V4 Pro",
            },
            {"model_id": "qwen/qwen3.7-plus", "label": "Qwen3.7 Plus"},
        ],
    }
    payload = response.text
    assert secret not in payload
    for forbidden in (
        "api_key",
        "credential_id",
        "encrypted",
        "fingerprint",
        "key_last_four",
        "proxy_url",
        "upstream_body",
    ):
        assert forbidden not in payload


def test_execution_options_expose_env_backed_platform_credits_without_secret(
    credential_api, monkeypatch
):
    token = _signup(credential_api.client, "env-platform-options-api@example.com")
    provider = credential_api.store.get_provider("openrouter")
    assert provider is not None
    credential_api.store.upsert_provider(
        provider_id="openrouter",
        display_name=provider["display_name"],
        adapter_type=provider["adapter_type"],
        approved_base_url=provider["approved_base_url"],
        capabilities=provider["capabilities"],
        byok_enabled=provider["byok_enabled"],
        platform_enabled=True,
        status=provider["status"],
    )
    sentinel = "sk-or-api-env-options-abcd"
    monkeypatch.setenv("OPENROUTER_API_KEY", sentinel)

    response = credential_api.client.get(
        "/api/credits/execution-options",
        headers=_auth(token),
    )

    assert response.status_code == 200
    item = next(
        item
        for item in response.json()["providers"]
        if item["provider_id"] == "openrouter"
    )
    assert item["platform_credits_available"] is True
    assert sentinel not in response.text
    assert "api_key" not in response.text.lower()


def test_create_and_list_never_return_or_log_full_key(credential_api, caplog):
    token = _signup(credential_api.client, "safe-output-api@example.com")
    secret = "sk-or-fake-never-return-abcd"
    credential_api.adapter.queue("verified")

    created = _create(credential_api, token, secret=secret, set_default=True)
    listed = credential_api.client.get(
        "/api/credits/api-keys", headers=_auth(token)
    )

    assert created.status_code == 201
    assert listed.status_code == 200
    payload_text = created.text + listed.text
    assert secret not in payload_text
    assert "api_key" not in payload_text
    assert created.json()["credential"]["key_last_four"] == "abcd"
    assert created.json()["credential"]["status"] == "verified"
    assert created.json()["credential"]["is_default"] is True
    assert secret not in caplog.text


def test_create_distinguishes_invalid_and_unavailable(credential_api):
    token = _signup(credential_api.client, "states-api@example.com")
    credential_api.adapter.queue("invalid")
    credential_api.adapter.queue("verification_unavailable")

    invalid = _create(credential_api, token, label="Invalid")
    unavailable = _create(credential_api, token, label="Unavailable")

    assert invalid.status_code == 201
    assert invalid.json()["credential"]["status"] == "invalid"
    assert invalid.json()["credential"]["verification_message"] == (
        "The provider rejected this API key."
    )
    assert unavailable.status_code == 201
    assert unavailable.json()["credential"]["status"] == "verification_unavailable"
    assert unavailable.json()["credential"]["verification_message"] == (
        "Provider verification was unavailable."
    )


def test_multiple_named_keys_and_one_default_per_provider(credential_api):
    token = _signup(credential_api.client, "defaults-api@example.com")
    credential_api.adapter.queue("verified")
    credential_api.adapter.queue("verified")

    first = _create(credential_api, token, label="Research", set_default=True)
    second = _create(credential_api, token, label="Personal", set_default=True)
    listed = credential_api.client.get(
        "/api/credits/api-keys", headers=_auth(token)
    ).json()["items"]

    assert first.status_code == 201
    assert second.status_code == 201
    assert {item["label"] for item in listed} == {"Research", "Personal"}
    assert [item["credential_id"] for item in listed if item["is_default"]] == [
        second.json()["credential"]["credential_id"]
    ]


def test_duplicate_label_returns_conflict_without_secret(credential_api):
    token = _signup(credential_api.client, "duplicate-api@example.com")
    credential_api.adapter.queue("verified")
    credential_api.adapter.queue("verified")
    secret = "sk-or-fake-duplicate-wxyz"
    assert _create(credential_api, token).status_code == 201

    duplicate = _create(credential_api, token, secret=secret)

    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "An API key with this name already exists."
    assert secret not in duplicate.text


def test_verify_default_and_revoke_enforce_ownership(credential_api):
    owner = _signup(credential_api.client, "owner-api@example.com")
    credential_api.adapter.queue("verification_unavailable")
    created = _create(credential_api, owner).json()["credential"]
    other = _signup(credential_api.client, "other-api@example.com")

    for method, suffix in (
        ("post", "verify"),
        ("post", "default"),
        ("delete", ""),
    ):
        path = f"/api/credits/api-keys/{created['credential_id']}"
        if suffix:
            path += f"/{suffix}"
        response = credential_api.client.request(method, path, headers=_auth(other))
        assert response.status_code == 404, (method, path, response.text)


def test_reverify_default_and_revoke_lifecycle(credential_api):
    token = _signup(credential_api.client, "lifecycle-api@example.com")
    credential_api.adapter.queue("verification_unavailable")
    credential_api.adapter.queue("verified")
    created = _create(credential_api, token).json()["credential"]
    base = f"/api/credits/api-keys/{created['credential_id']}"

    verified = credential_api.client.post(f"{base}/verify", headers=_auth(token))
    defaulted = credential_api.client.post(f"{base}/default", headers=_auth(token))
    revoked = credential_api.client.delete(base, headers=_auth(token))
    listed = credential_api.client.get(
        "/api/credits/api-keys", headers=_auth(token)
    )

    assert verified.json()["credential"]["status"] == "verified"
    assert defaulted.json()["credential"]["is_default"] is True
    assert revoked.json()["credential"]["status"] == "revoked"
    assert listed.json()["items"] == []


def test_invalid_create_payload_does_not_echo_key(credential_api):
    token = _signup(credential_api.client, "invalid-input-api@example.com")
    secret = "leaky"

    response = _create(credential_api, token, secret=secret)

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid API key request."}
    assert secret not in response.text


def test_missing_encryption_configuration_fails_closed(credential_api, monkeypatch):
    token = _signup(credential_api.client, "encryption-api@example.com")
    secret = "sk-or-fake-no-encryption-abcd"
    credential_api.adapter.queue("verified")
    monkeypatch.delenv("BROKER_TOKEN_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)

    response = _create(credential_api, token, secret=secret)

    assert response.status_code == 503
    assert response.json() == {"detail": "Credential encryption is unavailable."}
    assert secret not in response.text
    assert credential_api.store.list_user_credentials(1) == []


def test_secret_canary_is_not_reflected_by_identifier_or_filter_validation(credential_api):
    token = _signup(credential_api.client, "secret-canary-api@example.com")
    secret = "sk-fake-url-canary-never-echo-abcd"

    invalid_path = credential_api.client.post(
        f"/api/credits/api-keys/{secret}/verify",
        headers=_auth(token),
    )
    invalid_filter = credential_api.client.get(
        "/api/credits/api-keys",
        headers=_auth(token),
        params={"provider_id": secret},
    )

    assert invalid_path.status_code == 422
    assert invalid_filter.status_code == 422
    assert secret not in invalid_path.text
    assert secret not in invalid_filter.text