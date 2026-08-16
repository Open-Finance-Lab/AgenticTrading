"""Stripe Test Mode billing configuration contracts."""

from __future__ import annotations

import pytest

from dashboard.backend.domain.credits.config import (
    BillingConfigurationError,
    BillingUnavailableError,
    load_billing_config,
)


def _valid_environment() -> dict[str, str]:
    return {
        "ATL_STRIPE_TEST_BILLING_ENABLED": "1",
        "STRIPE_SECRET_KEY": "sk_test_example_not_a_real_key",
        "STRIPE_WEBHOOK_SECRET": "whsec_example_not_a_real_secret",
        "PUBLIC_APP_URL": "http://127.0.0.1:8000",
    }


def test_missing_configuration_keeps_billing_unavailable_without_crashing():
    config = load_billing_config({})

    assert config.enabled is False
    assert config.ready is False
    with pytest.raises(BillingUnavailableError, match="not configured"):
        config.require_ready()


def test_valid_test_mode_configuration_is_ready_and_normalizes_base_url():
    environment = _valid_environment()
    environment["PUBLIC_APP_URL"] = "https://atl.example/app-root/"

    config = load_billing_config(environment)

    assert config.enabled is True
    assert config.ready is True
    assert config.public_app_url == "https://atl.example/app-root"
    assert config.require_ready() is config


def test_explicit_enable_flag_is_required_even_when_test_secrets_exist():
    environment = _valid_environment()
    environment.pop("ATL_STRIPE_TEST_BILLING_ENABLED")

    config = load_billing_config(environment)

    assert config.enabled is False
    assert config.ready is False


@pytest.mark.parametrize(
    "missing",
    ["STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET", "PUBLIC_APP_URL"],
)
def test_enabled_billing_with_a_missing_value_is_unavailable(missing):
    environment = _valid_environment()
    environment.pop(missing)

    config = load_billing_config(environment)

    assert config.enabled is True
    assert config.ready is False
    with pytest.raises(BillingUnavailableError, match="not configured"):
        config.require_ready()


def test_live_secret_key_is_rejected_without_echoing_it():
    environment = _valid_environment()
    secret = "sk_live_do_not_echo_this_value"
    environment["STRIPE_SECRET_KEY"] = secret

    with pytest.raises(BillingConfigurationError) as caught:
        load_billing_config(environment)

    assert secret not in str(caught.value)
    assert "Live Mode" in str(caught.value)


@pytest.mark.parametrize(
    "value",
    [
        "ftp://atl.example",
        "https://user:password@atl.example",
        "https://atl.example/app#fragment",
        "https://",
        "not-a-url",
    ],
)
def test_unsafe_public_app_url_is_rejected(value):
    environment = _valid_environment()
    environment["PUBLIC_APP_URL"] = value

    with pytest.raises(BillingConfigurationError, match="PUBLIC_APP_URL"):
        load_billing_config(environment)


def test_unknown_enable_flag_is_a_configuration_error():
    environment = _valid_environment()
    environment["ATL_STRIPE_TEST_BILLING_ENABLED"] = "sometimes"

    with pytest.raises(BillingConfigurationError, match="enable flag"):
        load_billing_config(environment)


def test_config_repr_redacts_both_secrets():
    config = load_billing_config(_valid_environment())
    rendered = repr(config)

    assert config.secret_key not in rendered
    assert config.webhook_secret not in rendered
