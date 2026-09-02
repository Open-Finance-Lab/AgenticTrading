"""Credential lifecycle service tests with fake, non-billable adapters."""

from __future__ import annotations

import sqlite3

import pytest
from cryptography.fernet import Fernet

from dashboard.backend.domain.brokers import repository as broker_repository
from dashboard.backend.domain.model_providers.models import (
    CredentialValidation,
    ProviderCapabilities,
    UserCredentialCreate,
)
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.model_providers.repository_common import (
    CredentialConflictError,
    CredentialOwnershipError,
)
from dashboard.backend.domain.model_providers.service import (
    CredentialResolutionError,
    ModelProviderService,
)
from dashboard.backend.domain.model_providers import service as service_module
from dashboard.backend.infrastructure.llm.execution.errors import ExecutionErrorCategory


class FakeAdapter:
    def __init__(self, *results: CredentialValidation):
        self.results = list(results)
        self.calls: list[tuple[str, str]] = []

    def validate(self, base_url: str, secret: str, *, client=None) -> CredentialValidation:
        self.calls.append((base_url, secret))
        return self.results.pop(0)


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch):
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)


def _service(tmp_path, adapter: FakeAdapter) -> tuple[ModelProviderService, ModelProviderStore]:
    store = ModelProviderStore(tmp_path / "model-provider-service.db")
    return (
        ModelProviderService(
            store=store,
            adapter_resolver=lambda _adapter_type: adapter,
        ),
        store,
    )


def _request(*, label: str = "Research", set_default: bool = False) -> UserCredentialCreate:
    return UserCredentialCreate(
        provider_id="openrouter",
        label=label,
        api_key="sk-or-fake-service-abcd",
        set_default=set_default,
    )


def _validation(status: str) -> CredentialValidation:
    return CredentialValidation(
        status=status,
        message={
            "verified": "API key verified.",
            "invalid": "The provider rejected this API key.",
            "verification_unavailable": "Provider verification was unavailable.",
        }[status],
        models=["fake/model"] if status == "verified" else [],
    )


def test_create_encrypts_verifies_and_returns_only_public_metadata(tmp_path):
    adapter = FakeAdapter(_validation("verified"))
    service, store = _service(tmp_path, adapter)

    created = service.create_credential(7, _request(set_default=True))

    assert created.status == "verified"
    assert created.is_default is True
    assert created.key_last_four == "abcd"
    assert created.last_verified_at is not None
    assert adapter.calls == [
        ("https://openrouter.ai/api/v1", "sk-or-fake-service-abcd")
    ]
    serialized = created.model_dump_json()
    assert "sk-or-fake-service-abcd" not in serialized
    assert "api_key" not in serialized
    with sqlite3.connect(store.db_path) as conn:
        stored = conn.execute(
            "SELECT api_key_enc FROM user_model_credentials WHERE credential_id = ?",
            (str(created.credential_id),),
        ).fetchone()[0]
    assert stored != "sk-or-fake-service-abcd"
    assert "sk-or-fake-service-abcd" not in stored


def test_credential_lifecycle_emits_only_post_commit_safe_metadata(
    tmp_path,
    monkeypatch,
):
    adapter = FakeAdapter(_validation("verified"), _validation("verified"))
    service, _store = _service(tmp_path, adapter)
    events = []
    errors = []
    monkeypatch.setattr(
        service_module.analytics_instrumentation,
        "emit_credential_event",
        lambda **kwargs: events.append(kwargs),
    )
    monkeypatch.setattr(
        service_module.analytics_instrumentation,
        "emit_safe_error_event",
        lambda **kwargs: errors.append(kwargs),
    )

    created = service.create_credential(7, _request(set_default=True))
    service.reverify_credential(7, str(created.credential_id))
    service.set_default_credential(7, str(created.credential_id))
    service.revoke_credential(7, str(created.credential_id))

    assert [event["event_name"] for event in events] == [
        "credential_saved",
        "credential_verified",
        "credential_defaulted",
        "credential_reverified",
        "credential_defaulted",
        "credential_revoked",
    ]
    assert errors == []
    assert "sk-or-fake-service-abcd" not in repr(events)


def test_invalid_credential_emits_safe_category_not_adapter_detail(
    tmp_path,
    monkeypatch,
):
    service, _store = _service(tmp_path, FakeAdapter(_validation("invalid")))
    events = []
    errors = []
    monkeypatch.setattr(
        service_module.analytics_instrumentation,
        "emit_credential_event",
        lambda **kwargs: events.append(kwargs),
    )
    monkeypatch.setattr(
        service_module.analytics_instrumentation,
        "emit_safe_error_event",
        lambda **kwargs: errors.append(kwargs),
    )

    service.create_credential(7, _request())

    assert [event["event_name"] for event in events] == ["credential_saved"]
    assert errors[0]["error_category"] == "credential_invalid"
    assert "api_key" not in repr(errors)


@pytest.mark.parametrize("status", ["invalid", "verification_unavailable"])
def test_create_preserves_non_verified_outcome_without_default(tmp_path, status):
    service, _store = _service(tmp_path, FakeAdapter(_validation(status)))

    created = service.create_credential(7, _request(set_default=True))

    assert created.status == status
    assert created.is_default is False
    assert created.last_verified_at is None


def test_create_requires_configured_encryption_key(tmp_path, monkeypatch):
    adapter = FakeAdapter(_validation("verified"))
    service, store = _service(tmp_path, adapter)
    monkeypatch.delenv("BROKER_TOKEN_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)

    with pytest.raises(RuntimeError, match="BROKER_TOKEN_ENCRYPTION_KEY is not set"):
        service.create_credential(7, _request())

    assert adapter.calls == []
    assert store.list_user_credentials(7) == []


def test_create_rejects_invalid_encryption_key_before_verification(
    tmp_path, monkeypatch
):
    adapter = FakeAdapter(_validation("verified"))
    service, store = _service(tmp_path, adapter)
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", "not-a-fernet-key")
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)

    with pytest.raises(RuntimeError, match="is set but is not a valid Fernet key"):
        service.create_credential(7, _request())

    assert adapter.calls == []
    assert store.list_user_credentials(7) == []


def test_create_persists_final_state_without_follow_up_mutations(tmp_path, monkeypatch):
    adapter = FakeAdapter(_validation("verified"))
    service, store = _service(tmp_path, adapter)
    original_create = store.create_user_credential
    create_calls = []

    def capture_create(**kwargs):
        create_calls.append(kwargs)
        return original_create(**kwargs)

    def unexpected_mutation(*_args, **_kwargs):
        raise AssertionError("credential creation must not perform a second write")

    monkeypatch.setattr(store, "create_user_credential", capture_create)
    monkeypatch.setattr(store, "set_user_credential_status", unexpected_mutation)
    monkeypatch.setattr(store, "set_default_user_credential", unexpected_mutation)

    created = service.create_credential(7, _request(set_default=True))

    assert created.status == "verified"
    assert created.is_default is True
    assert len(create_calls) == 1
    assert create_calls[0]["status"] == "verified"
    assert create_calls[0]["set_default"] is True
    assert create_calls[0]["last_verified_at"] is not None


def test_verification_message_is_fixed_and_never_persists_adapter_details(tmp_path):
    secret = "sk-or-fake-service-abcd"
    adapter = FakeAdapter(
        CredentialValidation(
            status="invalid",
            message=f"upstream body leaked {secret}",
        )
    )
    service, store = _service(tmp_path, adapter)

    created = service.create_credential(7, _request())

    assert created.verification_message == "The provider rejected this API key."
    assert secret not in created.model_dump_json()
    with sqlite3.connect(store.db_path) as conn:
        stored = conn.execute(
            "SELECT verification_message FROM user_model_credentials WHERE credential_id = ?",
            (str(created.credential_id),),
        ).fetchone()[0]
    assert stored == "The provider rejected this API key."
    assert secret not in stored


def test_reverify_updates_status_and_can_then_become_default(tmp_path):
    adapter = FakeAdapter(
        _validation("verification_unavailable"),
        _validation("verified"),
    )
    service, _store = _service(tmp_path, adapter)
    created = service.create_credential(7, _request())

    verified = service.reverify_credential(7, str(created.credential_id))
    defaulted = service.set_default_credential(7, str(created.credential_id))

    assert verified.status == "verified"
    assert verified.last_verified_at is not None
    assert defaulted.is_default is True
    assert len(adapter.calls) == 2


def test_only_one_verified_default_exists_per_user_and_provider(tmp_path):
    adapter = FakeAdapter(_validation("verified"), _validation("verified"))
    service, _store = _service(tmp_path, adapter)
    first = service.create_credential(7, _request(label="Research", set_default=True))
    second = service.create_credential(7, _request(label="Personal", set_default=True))

    listed = service.list_credentials(7)

    assert {item.label for item in listed} == {"Research", "Personal"}
    assert [item.credential_id for item in listed if item.is_default] == [
        second.credential_id
    ]
    assert first.credential_id != second.credential_id


def test_unverified_credential_cannot_be_default(tmp_path):
    service, _store = _service(tmp_path, FakeAdapter(_validation("invalid")))
    created = service.create_credential(7, _request())

    with pytest.raises(CredentialConflictError, match="only verified"):
        service.set_default_credential(7, str(created.credential_id))


@pytest.mark.parametrize("operation", ["reverify", "default", "revoke"])
def test_credential_mutations_enforce_user_ownership(tmp_path, operation):
    adapter = FakeAdapter(_validation("verified"))
    service, _store = _service(tmp_path, adapter)
    created = service.create_credential(7, _request())

    with pytest.raises(CredentialOwnershipError):
        if operation == "reverify":
            service.reverify_credential(8, str(created.credential_id))
        elif operation == "default":
            service.set_default_credential(8, str(created.credential_id))
        else:
            service.revoke_credential(8, str(created.credential_id))


def test_revoke_removes_credential_from_active_list(tmp_path):
    service, _store = _service(tmp_path, FakeAdapter(_validation("verified")))
    created = service.create_credential(7, _request(set_default=True))

    revoked = service.revoke_credential(7, str(created.credential_id))

    assert revoked.status == "revoked"
    assert revoked.is_default is False
    assert service.list_credentials(7) == []


def test_list_providers_returns_only_enabled_byok_records(tmp_path):
    service, _store = _service(tmp_path, FakeAdapter())

    providers = service.list_providers()

    assert {provider.provider_id for provider in providers} == {
        "anthropic",
        "gemini",
        "openai",
        "openrouter",
    }
    assert all(provider.byok_enabled for provider in providers)


def _set_provider_modes(
    store: ModelProviderStore,
    provider_id: str,
    *,
    byok_enabled: bool,
    platform_enabled: bool,
) -> None:
    provider = store.get_provider(provider_id)
    assert provider is not None
    store.upsert_provider(
        provider_id=provider_id,
        display_name=provider["display_name"],
        adapter_type=provider["adapter_type"],
        approved_base_url=provider["approved_base_url"],
        capabilities=provider["capabilities"],
        byok_enabled=byok_enabled,
        platform_enabled=platform_enabled,
        status=provider["status"],
    )


def test_execution_options_expose_models_only_for_safe_availability_flags(tmp_path):
    adapter = FakeAdapter(
        _validation("invalid"),
        _validation("verified"),
    )
    service, store = _service(tmp_path, adapter)
    service.create_credential(7, _request(label="Rejected", set_default=True))
    verified = service.create_credential(7, _request(label="Verified"))
    _set_provider_modes(
        store,
        "openrouter",
        byok_enabled=True,
        platform_enabled=True,
    )
    store.upsert_platform_credential(
        provider_id="openrouter",
        secret="sk-or-fake-platform-wxyz",
        status="verified",
    )

    before_default = {
        option.provider_id: option
        for option in service.list_execution_options(7)
    }["openrouter"]
    service.set_default_credential(7, str(verified.credential_id))
    after_default = {
        option.provider_id: option
        for option in service.list_execution_options(7)
    }["openrouter"]

    assert before_default.byok_available is False
    assert after_default.model_dump(mode="json") == {
        "provider_id": "openrouter",
        "display_name": "OpenRouter",
        "adapter_type": "openrouter",
        "byok_available": True,
        "platform_credits_available": True,
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


def test_platform_resolver_uses_openrouter_environment_key_when_store_is_empty(
    tmp_path, monkeypatch
):
    service, store = _service(tmp_path, FakeAdapter())
    _set_provider_modes(store, "openrouter", byok_enabled=True, platform_enabled=True)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-test-abcd")

    resolved = service.resolve_platform_credential("openrouter")

    assert resolved.credential_id is None
    assert resolved.key_last_four == "abcd"
    assert resolved.secret == "sk-or-env-test-abcd"


def test_commonstack_platform_credential_uses_explicit_environment_mapping(
    tmp_path, monkeypatch
):
    service, _store = _service(tmp_path, FakeAdapter())
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-environment-abcd")

    resolved = service.resolve_platform_credential("commonstack")

    assert resolved.provider_id == "commonstack"
    assert resolved.key_last_four == "abcd"
    assert resolved.secret == "cs-fake-environment-abcd"


def test_verified_stored_platform_credential_precedes_environment_key(
    tmp_path, monkeypatch
):
    service, store = _service(tmp_path, FakeAdapter())
    _set_provider_modes(store, "openrouter", byok_enabled=True, platform_enabled=True)
    store.upsert_platform_credential(
        provider_id="openrouter",
        secret="sk-or-stored-test-wxyz",
        status="verified",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-test-abcd")

    resolved = service.resolve_platform_credential("openrouter")

    assert resolved.key_last_four == "wxyz"
    assert resolved.secret == "sk-or-stored-test-wxyz"


def test_verified_stored_commonstack_credential_precedes_environment_key(
    tmp_path, monkeypatch
):
    service, store = _service(tmp_path, FakeAdapter())
    store.upsert_platform_credential(
        provider_id="commonstack",
        secret="cs-fake-stored-test-wxyz",
        status="verified",
    )
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-environment-abcd")

    resolved = service.resolve_platform_credential("commonstack")

    assert resolved.key_last_four == "wxyz"
    assert resolved.secret == "cs-fake-stored-test-wxyz"


def test_execution_options_keep_openrouter_ahead_of_commonstack(
    tmp_path, monkeypatch
):
    service, _store = _service(tmp_path, FakeAdapter())
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-fake-options-abcd")
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-options-wxyz")

    provider_ids = [
        option.provider_id
        for option in service.list_execution_options(7)
        if option.platform_credits_available
    ]

    assert provider_ids.index("openrouter") < provider_ids.index("commonstack")
    assert "cs-fake-options-wxyz" not in repr(service.list_execution_options(7))


def test_environment_fallback_is_provider_specific_and_fails_closed(
    tmp_path, monkeypatch
):
    service, store = _service(tmp_path, FakeAdapter())
    _set_provider_modes(store, "openai", byok_enabled=True, platform_enabled=True)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-test-abcd")

    with pytest.raises(CredentialResolutionError) as exc_info:
        service.resolve_platform_credential("openai")

    assert exc_info.value.category is ExecutionErrorCategory.CREDENTIAL_MISSING


def test_environment_fallback_requires_platform_enabled(tmp_path, monkeypatch):
    service, store = _service(tmp_path, FakeAdapter())
    _set_provider_modes(store, "openrouter", byok_enabled=True, platform_enabled=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-test-abcd")

    with pytest.raises(CredentialResolutionError) as exc_info:
        service.preflight_platform_credential("openrouter")

    assert exc_info.value.category is ExecutionErrorCategory.CREDENTIAL_MISSING


def test_execution_options_expose_environment_backed_platform_credits(
    tmp_path, monkeypatch
):
    service, store = _service(tmp_path, FakeAdapter())
    _set_provider_modes(store, "openrouter", byok_enabled=True, platform_enabled=True)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-options-abcd")

    option = next(
        item
        for item in service.list_execution_options(7)
        if item.provider_id == "openrouter"
    )

    assert option.platform_credits_available is True
    assert "sk-or-env-options-abcd" not in option.model_dump_json()


def test_execution_options_hide_platform_credits_without_any_platform_key(
    tmp_path, monkeypatch
):
    service, store = _service(tmp_path, FakeAdapter())
    _set_provider_modes(store, "openrouter", byok_enabled=True, platform_enabled=True)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    option = next(
        item
        for item in service.list_execution_options(7)
        if item.provider_id == "openrouter"
    )

    assert option.platform_credits_available is False


def test_execution_options_require_both_platform_flag_and_verified_status(tmp_path):
    service, store = _service(tmp_path, FakeAdapter())
    _set_provider_modes(
        store,
        "openai",
        byok_enabled=True,
        platform_enabled=False,
    )
    store.upsert_platform_credential(
        provider_id="openai",
        secret="sk-fake-platform-openai-abcd",
        status="verified",
    )
    _set_provider_modes(
        store,
        "anthropic",
        byok_enabled=True,
        platform_enabled=True,
    )
    store.upsert_platform_credential(
        provider_id="anthropic",
        secret="sk-fake-platform-anthropic-efgh",
        status="invalid",
    )
    _set_provider_modes(
        store,
        "gemini",
        byok_enabled=True,
        platform_enabled=True,
    )
    store.upsert_platform_credential(
        provider_id="gemini",
        secret="sk-fake-platform-gemini-ijkl",
        status="verified",
    )

    options = {
        option.provider_id: option
        for option in service.list_execution_options(7)
    }

    assert options["openai"].platform_credits_available is False
    assert options["anthropic"].platform_credits_available is False
    assert options["gemini"].platform_credits_available is True


def test_execution_options_do_not_decrypt_secrets_and_empty_allowlist_stays_empty(
    tmp_path,
    monkeypatch,
):
    service, store = _service(tmp_path, FakeAdapter())
    store.upsert_provider(
        provider_id="approved_compatible",
        display_name="Approved Compatible",
        adapter_type="openai_compatible",
        approved_base_url="https://provider.example/v1",
        capabilities=ProviderCapabilities(model_allowlist=()),
        byok_enabled=True,
        platform_enabled=False,
        status="enabled",
    )

    def unexpected_secret_access(*_args, **_kwargs):
        raise AssertionError("execution options must not decrypt credentials")

    monkeypatch.setattr(
        store,
        "get_verified_default_user_credential",
        unexpected_secret_access,
    )
    monkeypatch.setattr(
        store,
        "get_verified_platform_credential",
        unexpected_secret_access,
    )

    option = {
        item.provider_id: item
        for item in service.list_execution_options(7)
    }["approved_compatible"]

    assert option.models == ()
    assert option.byok_available is False
    assert option.platform_credits_available is False
