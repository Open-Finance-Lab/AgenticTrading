"""Fail-closed Stripe Test Mode configuration for ATL Credits."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit


class BillingConfigurationError(ValueError):
    """Billing configuration is unsafe or internally inconsistent."""


class BillingUnavailableError(RuntimeError):
    """A billing operation was requested while Test Mode is unavailable."""


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})


def _enabled(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise BillingConfigurationError(
        "ATL Stripe Test Mode enable flag must be true or false"
    )


def _public_app_url(value: str) -> str:
    raw = value.strip()
    parsed = urlsplit(raw)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.query
    ):
        raise BillingConfigurationError(
            "PUBLIC_APP_URL must be an HTTP(S) base URL without credentials, "
            "a query, or a fragment"
        )
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


@dataclass(frozen=True, repr=False)
class BillingConfig:
    """Resolved configuration that never exposes secrets through ``repr``."""

    enabled: bool
    secret_key: str = field(repr=False)
    webhook_secret: str = field(repr=False)
    public_app_url: str
    missing: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.enabled and not self.missing

    def require_ready(self) -> "BillingConfig":
        if not self.ready:
            raise BillingUnavailableError("Stripe Test Mode billing is not configured")
        return self

    def __repr__(self) -> str:
        return (
            "BillingConfig("
            f"enabled={self.enabled!r}, ready={self.ready!r}, "
            f"public_app_url={self.public_app_url!r}, missing={self.missing!r}, "
            "secret_key=<redacted>, webhook_secret=<redacted>)"
        )


def load_billing_config(
    environment: Mapping[str, str] | None = None,
) -> BillingConfig:
    """Resolve billing configuration without making module import a hard gate.

    Disabled or incomplete Test Mode configuration produces a non-ready object
    so unrelated ATL features continue to start. Unsafe configuration fails
    immediately when billing has been explicitly enabled.
    """

    source = os.environ if environment is None else environment
    enabled = _enabled(source.get("ATL_STRIPE_TEST_BILLING_ENABLED"))
    secret_key = (source.get("STRIPE_SECRET_KEY") or "").strip()
    webhook_secret = (source.get("STRIPE_WEBHOOK_SECRET") or "").strip()
    public_app_url_raw = (source.get("PUBLIC_APP_URL") or "").strip()

    if not enabled:
        return BillingConfig(
            enabled=False,
            secret_key="",
            webhook_secret="",
            public_app_url="",
            missing=(),
        )

    if secret_key.startswith("sk_live_"):
        raise BillingConfigurationError(
            "Stripe Live Mode is not supported by this ATL billing release"
        )
    if secret_key and not secret_key.startswith("sk_test_"):
        raise BillingConfigurationError(
            "STRIPE_SECRET_KEY must be a Stripe Test Mode key"
        )
    if webhook_secret and not webhook_secret.startswith("whsec_"):
        raise BillingConfigurationError(
            "STRIPE_WEBHOOK_SECRET must be a Stripe endpoint secret"
        )

    missing = tuple(
        name
        for name, value in (
            ("STRIPE_SECRET_KEY", secret_key),
            ("STRIPE_WEBHOOK_SECRET", webhook_secret),
            ("PUBLIC_APP_URL", public_app_url_raw),
        )
        if not value
    )
    public_app_url = (
        _public_app_url(public_app_url_raw) if public_app_url_raw else ""
    )
    return BillingConfig(
        enabled=True,
        secret_key=secret_key,
        webhook_secret=webhook_secret,
        public_app_url=public_app_url,
        missing=missing,
    )
