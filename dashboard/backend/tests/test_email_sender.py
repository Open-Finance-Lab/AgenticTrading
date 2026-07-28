"""Brevo transactional-mail adapter.

Drives the coroutine with asyncio.run() rather than adding a pytest-asyncio
dependency, and monkeypatches httpx.AsyncClient inside the sender module so no
socket is ever opened.
"""

import asyncio

import pytest

from dashboard.backend.infrastructure.email import sender


class _FakeResponse:
    def __init__(self, status_code=201, text="{}"):
        self.status_code = status_code
        self.text = text


def _install_fake_client(monkeypatch, response):
    """Replace httpx.AsyncClient with a recorder; return the captured calls."""
    calls = []

    class _FakeAsyncClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, url, headers=None, json=None):
            calls.append(
                {"url": url, "headers": headers, "json": json, "timeout": self.timeout}
            )
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(sender.httpx, "AsyncClient", _FakeAsyncClient)
    return calls


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "test-key")
    monkeypatch.setenv("ACCOUNT_EMAIL_FROM", "noreply@example.com")
    monkeypatch.delenv("ACCOUNT_EMAIL_FROM_NAME", raising=False)


def test_email_configured_requires_both_vars(monkeypatch):
    monkeypatch.delenv("BREVO_API_KEY", raising=False)
    monkeypatch.delenv("ACCOUNT_EMAIL_FROM", raising=False)
    assert sender.email_configured() is False
    monkeypatch.setenv("BREVO_API_KEY", "k")
    assert sender.email_configured() is False
    monkeypatch.setenv("ACCOUNT_EMAIL_FROM", "a@b.com")
    assert sender.email_configured() is True


def test_send_email_unconfigured_returns_false_and_prints_error(monkeypatch, capsys):
    monkeypatch.delenv("BREVO_API_KEY", raising=False)
    monkeypatch.delenv("ACCOUNT_EMAIL_FROM", raising=False)

    assert asyncio.run(sender.send_email("u@example.com", "Subj", "Body")) is False

    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "BREVO_API_KEY" in out


def test_send_email_posts_the_brevo_payload(monkeypatch, configured):
    calls = _install_fake_client(monkeypatch, _FakeResponse(201))

    assert asyncio.run(sender.send_email("u@example.com", "Subj", "Body")) is True

    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://api.brevo.com/v3/smtp/email"
    assert call["headers"]["api-key"] == "test-key"
    assert call["timeout"] == sender.SEND_TIMEOUT_SECONDS
    assert call["json"]["sender"] == {
        "name": sender.DEFAULT_FROM_NAME,
        "email": "noreply@example.com",
    }
    assert call["json"]["to"] == [{"email": "u@example.com"}]
    assert call["json"]["subject"] == "Subj"
    assert call["json"]["textContent"] == "Body"


def test_send_email_uses_configured_from_name(monkeypatch, configured):
    monkeypatch.setenv("ACCOUNT_EMAIL_FROM_NAME", "Custom Name")
    calls = _install_fake_client(monkeypatch, _FakeResponse(201))

    assert asyncio.run(sender.send_email("u@example.com", "S", "B")) is True
    assert calls[0]["json"]["sender"]["name"] == "Custom Name"


def test_send_email_non_2xx_returns_false_and_prints_error(monkeypatch, configured, capsys):
    _install_fake_client(monkeypatch, _FakeResponse(401, '{"message":"bad key"}'))

    assert asyncio.run(sender.send_email("u@example.com", "S", "B")) is False

    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "401" in out


def test_send_email_strips_newlines_from_the_provider_body(monkeypatch, configured, capsys):
    # A provider body is attacker-influencable (it can echo the submitted
    # address), so it must never inject a forged second log line.
    _install_fake_client(monkeypatch, _FakeResponse(400, "line one\nERROR: forged"))

    assert asyncio.run(sender.send_email("u@example.com", "S", "B")) is False

    out = capsys.readouterr().out.strip()
    assert len(out.splitlines()) == 1


def test_send_email_transport_failure_returns_false(monkeypatch, configured, capsys):
    _install_fake_client(monkeypatch, RuntimeError("connection reset"))

    assert asyncio.run(sender.send_email("u@example.com", "S", "B")) is False

    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "connection reset" in out
