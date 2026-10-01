"""The chat and algo Anthropic clients obey the same no-replay policy as the adapters (#558).

The SDK defaults are ``max_retries=2`` and a 600s read timeout; each silent replay
regenerates and bills a whole completion, outside any ledger. These build the REAL
SDK clients (no request is sent) and pin the policy on the constructed objects,
then pin the two constructor call sites in the AST so a later edit cannot drop it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import dashboard.backend.domain.backtesting.algo_service as algo_service
import dashboard.backend.domain.chat.service as chat_service
from dashboard.backend.infrastructure.llm.execution.adapters import base as base_module


_BACKEND = Path(chat_service.__file__).resolve().parents[2]
_CLIENT_SOURCES = {
    "domain/chat/service.py": "AsyncAnthropic",
    "domain/backtesting/algo_service.py": "Anthropic",
}


@pytest.fixture(autouse=True)
def _fresh_chat_client(monkeypatch):
    monkeypatch.setattr(chat_service, "_claude_client", None, raising=False)


@pytest.mark.parametrize("commonstack", [True, False])
def test_chat_client_does_not_retry_and_uses_provider_timeout(monkeypatch, commonstack):
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-chat-policy-test")
    if commonstack:
        monkeypatch.setenv("COMMONSTACK_API_KEY", "sk-fake-commonstack-test")

    client = chat_service.get_claude_client()

    assert client.max_retries == 0
    assert client.timeout == base_module.provider_http_timeout()


def test_algo_client_does_not_retry_and_uses_provider_timeout(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-algo-policy-test")

    client = algo_service._get_anthropic_client()

    assert client is not None
    assert client.max_retries == 0
    assert client.timeout == base_module.provider_http_timeout()


@pytest.mark.parametrize("relpath,ctor", sorted(_CLIENT_SOURCES.items()))
def test_client_constructors_pass_max_retries_and_timeout(relpath, ctor):
    """Source guard: every ``ctor(...)`` call pins retries to SDK_MAX_RETRIES and sets a timeout."""

    tree = ast.parse((_BACKEND / relpath).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == ctor
    ]
    assert calls, f"{relpath}: no {ctor}(...) call found"
    for node in calls:
        keywords = {kw.arg: kw.value for kw in node.keywords}
        retries = keywords.get("max_retries")
        assert isinstance(retries, ast.Name) and retries.id == "SDK_MAX_RETRIES", relpath
        timeout = keywords.get("timeout")
        assert (
            isinstance(timeout, ast.Call)
            and isinstance(timeout.func, ast.Name)
            and timeout.func.id == "provider_http_timeout"
        ), relpath
