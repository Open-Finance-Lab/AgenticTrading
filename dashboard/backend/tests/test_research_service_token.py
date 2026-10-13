"""RESEARCH_SERVICE_TOKEN must never silently default to "dev-token" in production."""

import pytest
from fastapi import HTTPException

import dashboard.backend.api.routers.research as research


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("RESEARCH_SERVICE_TOKEN", "RENDER", "ATL_ENV"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("env", [{"RENDER": "true"}, {"ATL_ENV": "production"}, {"ATL_ENV": "prod"}])
def test_production_without_a_token_refuses(monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(HTTPException) as caught:
        research._service_headers()
    assert caught.value.status_code == 503
    assert "dev-token" not in str(caught.value.detail)


def test_production_uses_the_configured_token(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RESEARCH_SERVICE_TOKEN", "  real-token  ")
    assert research._service_headers() == {"X-Service-Token": "real-token"}


def test_blank_token_counts_as_unset_in_production(monkeypatch):
    monkeypatch.setenv("ATL_ENV", "production")
    monkeypatch.setenv("RESEARCH_SERVICE_TOKEN", "   ")
    with pytest.raises(HTTPException):
        research._service_headers()


def test_local_runs_keep_the_dev_default():
    assert research._service_headers() == {"X-Service-Token": "dev-token"}
