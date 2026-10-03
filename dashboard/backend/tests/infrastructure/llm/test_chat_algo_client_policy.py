"""Every LLM SDK client in the dashboard obeys the no-replay policy, and owns its retries (#558).

The SDK defaults are ``max_retries=2`` and a 600s read timeout; each silent replay
regenerates and bills a whole completion, outside any ledger. ``max_retries=0``
alone is only half the policy: it also drops the free retries of failures that
generated nothing, so every caller outside the billed execution layer must run
its calls through ``http_policy.(a)call_with_retries``.
"""

from __future__ import annotations

import ast
import asyncio
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import openai
import pytest

import dashboard.backend.domain.backtesting.algo_service as algo_service
import dashboard.backend.domain.chat.service as chat_service
import dashboard.backend.infrastructure.llm.http_policy as http_policy
from dashboard.backend.infrastructure.llm.execution.adapters.base import (
    map_provider_error,
)


_BACKEND = Path(chat_service.__file__).resolve().parents[2]
_DASHBOARD = _BACKEND.parent
_REPO = _DASHBOARD.parent
_SDK_CLIENTS = frozenset({"Anthropic", "AsyncAnthropic", "OpenAI", "AsyncOpenAI"})
_RETRY_RUNNERS = frozenset({"call_with_retries", "acall_with_retries"})
# Constructors that legitimately take neither kwarg literally: the adapters'
# default ``client_factory`` forwards ``**kwargs`` built by their own call
# sites, which ``test_sdk_adapter_sources_pass_max_retries_and_timeout`` pins,
# and ``LLMExecutionService`` is their retry owner.
_KWARGS_FORWARDING_FACTORIES = frozenset(
    {
        "backend/infrastructure/llm/execution/adapters/anthropic.py",
        "backend/infrastructure/llm/execution/adapters/openai.py",
    }
)


@pytest.fixture(autouse=True)
def _fresh_chat_client(monkeypatch):
    monkeypatch.setattr(chat_service, "_claude_client", None, raising=False)


# --- The constructed clients -------------------------------------------------


@pytest.mark.parametrize("commonstack", [True, False])
def test_chat_client_does_not_retry_and_reads_the_timeout_setting(monkeypatch, commonstack):
    # A value no default produces, so this fails if the client is built from
    # a hard-coded timeout rather than LLM_PROVIDER_READ_TIMEOUT_SECONDS.
    monkeypatch.setattr(http_policy, "PROVIDER_READ_TIMEOUT_SECONDS", 77)
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-chat-policy-test")
    if commonstack:
        monkeypatch.setenv("COMMONSTACK_API_KEY", "sk-fake-commonstack-test")

    client = chat_service.get_claude_client()

    assert client.max_retries == 0
    assert client.timeout.read == 77
    assert client.timeout.connect == 8.0


def test_algo_client_does_not_retry_and_reads_the_timeout_setting(monkeypatch):
    monkeypatch.setattr(http_policy, "PROVIDER_READ_TIMEOUT_SECONDS", 77)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-algo-policy-test")

    client = algo_service._get_anthropic_client()

    assert client is not None
    assert client.max_retries == 0
    assert client.timeout.read == 77
    assert client.timeout.connect == 8.0


# --- Source guards over every production module -----------------------------


def _production_sources() -> list[Path]:
    return sorted(
        path
        for path in _DASHBOARD.rglob("*.py")
        if "tests" not in path.relative_to(_DASHBOARD).parts
        and "__pycache__" not in path.parts
        and "node_modules" not in path.parts
    )


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _sdk_constructor_calls() -> list[tuple[str, ast.Call, ast.Module]]:
    found = []
    for path in _production_sources():
        source = path.read_text(encoding="utf-8")
        if not any(name in source for name in _SDK_CLIENTS):
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called_name(node) in _SDK_CLIENTS:
                found.append((path.relative_to(_DASHBOARD).as_posix(), node, tree))
    return found


def _keyword_names(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_every_sdk_client_constructor_pins_retries_and_timeout():
    """A third module building a client with SDK defaults fails here, not in prod."""

    calls = _sdk_constructor_calls()
    relpaths = {relpath for relpath, _node, _tree in calls}
    # The scan must actually see the sites it exists for.
    assert {
        "backend/domain/chat/service.py",
        "backend/domain/backtesting/algo_service.py",
        "backend/llm_integration_example.py",
        # The legacy harness's CommonStack thinking-off client. It first
        # shipped behind `(openai_cls or _openai_cls())(...)`, a call this
        # scan cannot name, with SDK defaults.
        "backend/infrastructure/llm/providers/commonstack.py",
    } <= relpaths
    for relpath, node, _tree in calls:
        where = f"{relpath}:{node.lineno}"
        if relpath in _KWARGS_FORWARDING_FACTORIES:
            assert [kw.arg for kw in node.keywords] == [None], where
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        assert _keyword_names(keywords.get("max_retries")) == "SDK_MAX_RETRIES", where
        timeout = keywords.get("timeout")
        assert (
            isinstance(timeout, ast.Call)
            and _called_name(timeout) == "provider_http_timeout"
        ), where


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def test_every_messages_create_outside_the_adapters_goes_through_a_retry_runner():
    """``max_retries=0`` with no retry owner fails a stale keep-alive or a 529 outright.

    ``chat.completions.create`` counts too: it is the same SDK call on the
    OpenAI surface, and the CommonStack thinking-off client makes it.
    """

    trees = {relpath: tree for relpath, _node, tree in _sdk_constructor_calls()}
    checked = 0
    for relpath, tree in sorted(trees.items()):
        if relpath in _KWARGS_FORWARDING_FACTORIES:
            continue
        parents = _parents(tree)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "create"
                and _keyword_names(node.func.value) in {"messages", "completions"}
            ):
                continue
            checked += 1
            lam = parents.get(node)
            runner = parents.get(lam)
            assert (
                isinstance(lam, ast.Lambda)
                and isinstance(runner, ast.Call)
                and _called_name(runner) in _RETRY_RUNNERS
                and runner.args
                and runner.args[0] is lam
            ), f"{relpath}:{node.lineno}: messages.create outside (a)call_with_retries"
    assert checked >= 3


# --- Import weight ------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "dashboard.backend.domain.chat.service",
        "dashboard.backend.domain.backtesting.algo_service",
        # Shares the adapter's wire body and response reading through leaves;
        # importing the adapter instead would load this layer per backtest.
        "dashboard.backend.infrastructure.llm.providers.commonstack",
    ],
)
def test_policy_callers_do_not_load_the_execution_layer(module, tmp_path):
    code = textwrap.dedent(
        f"""
        import sys
        import {module}
        loaded = sorted(m for m in sys.modules if m.startswith(
            "dashboard.backend.infrastructure.llm.execution"))
        assert not loaded, loaded
        print("leaf-only")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        env={**os.environ, "DATABASE_PATH": str(tmp_path / "x.db")},
    )
    assert result.returncode == 0, result.stderr
    assert "leaf-only" in result.stdout


# --- Classifying an SDK error -------------------------------------------------

_REQUEST = httpx.Request("POST", "https://api.example.test/v1/messages")


def _chained(error: BaseException, cause: BaseException) -> BaseException:
    error.__cause__ = cause
    return error


def _status_error(sdk, status: int, headers: dict[str, str] | None = None, body=None):
    response = httpx.Response(
        status, headers=headers or {}, json=body or {"error": {"type": "x"}}, request=_REQUEST
    )
    return sdk.APIStatusError("upstream", response=response, body=body)


_SDK_ERRORS = {
    "anthropic-connect-refused": _chained(
        anthropic.APIConnectionError(request=_REQUEST),
        httpx.ConnectError("refused", request=_REQUEST),
    ),
    "anthropic-stale-keepalive": _chained(
        anthropic.APIConnectionError(request=_REQUEST),
        httpx.RemoteProtocolError("Server disconnected", request=_REQUEST),
    ),
    "anthropic-read-timeout": _chained(
        anthropic.APITimeoutError(request=_REQUEST),
        httpx.ReadTimeout("read", request=_REQUEST),
    ),
    "anthropic-connect-timeout": _chained(
        anthropic.APITimeoutError(request=_REQUEST),
        httpx.ConnectTimeout("connect", request=_REQUEST),
    ),
    "anthropic-timeout-no-cause": anthropic.APITimeoutError(request=_REQUEST),
    "anthropic-529": _status_error(anthropic, 529),
    "anthropic-429-retry-after": _status_error(anthropic, 429, {"retry-after": "7"}),
    "anthropic-429-quota": _status_error(
        anthropic, 429, body={"error": {"type": "insufficient_quota"}}
    ),
    "anthropic-500-should-not-retry": _status_error(anthropic, 500, {"x-should-retry": "false"}),
    "anthropic-400": _status_error(anthropic, 400),
    "anthropic-401": _status_error(anthropic, 401),
    "anthropic-402": _status_error(anthropic, 402),
    "openai-connect-refused": _chained(
        openai.APIConnectionError(request=_REQUEST),
        httpx.ConnectError("refused", request=_REQUEST),
    ),
    "openai-503": _status_error(openai, 503),
    "plain-runtime-error": RuntimeError("not an SDK error"),
}


@pytest.mark.parametrize("name", sorted(_SDK_ERRORS))
def test_sdk_error_hint_agrees_with_the_adapter_mapper(name):
    """One classification, two entry points: drift here re-opens #558 on one side."""

    exc = _SDK_ERRORS[name]
    hint, retry_after = http_policy.sdk_error_retry_hint(exc)
    mapped = map_provider_error(exc)
    assert hint is mapped.retry_hint, name
    assert retry_after == mapped.retry_after_seconds, name


@pytest.mark.parametrize(
    "name,hint",
    [
        ("anthropic-connect-refused", http_policy.RetryHint.PRE_SEND),
        ("anthropic-connect-timeout", http_policy.RetryHint.PRE_SEND),
        ("anthropic-stale-keepalive", http_policy.RetryHint.REJECTED),
        ("anthropic-529", http_policy.RetryHint.REJECTED),
        ("anthropic-read-timeout", http_policy.RetryHint.NONE),
        ("anthropic-timeout-no-cause", http_policy.RetryHint.NONE),
        ("anthropic-401", http_policy.RetryHint.NONE),
    ],
)
def test_sdk_error_hint_values(name, hint):
    assert http_policy.sdk_error_retry_hint(_SDK_ERRORS[name])[0] is hint


# --- The retry runners --------------------------------------------------------


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _scripted(steps, clock=None, per_call_seconds=0.0):
    calls = []

    def call():
        calls.append(1)
        if clock is not None:
            clock.now += per_call_seconds
        step = steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    return call, calls


def test_runner_repeats_pre_send_failures_with_backoff_then_gives_up():
    sleeps = []
    err = _SDK_ERRORS["anthropic-connect-refused"]
    call, calls = _scripted([err, err, err, "unreached"])

    with pytest.raises(anthropic.APIConnectionError):
        http_policy.call_with_retries(call, label="t", sleep=sleeps.append)

    assert len(calls) == 1 + http_policy.MAX_SAME_PROVIDER_RETRIES
    assert sleeps == [4.0, 12.0]


def test_runner_recovers_from_a_stale_keepalive():
    sleeps = []
    call, calls = _scripted([_SDK_ERRORS["anthropic-stale-keepalive"], "ok"])

    assert http_policy.call_with_retries(call, label="t", sleep=sleeps.append) == "ok"
    assert len(calls) == 2
    assert sleeps == [4.0]


def test_runner_never_repeats_a_read_timeout():
    sleeps = []
    call, calls = _scripted([_SDK_ERRORS["anthropic-read-timeout"], "unreached"])

    with pytest.raises(anthropic.APITimeoutError):
        http_policy.call_with_retries(call, label="t", sleep=sleeps.append)

    assert len(calls) == 1
    assert sleeps == []


def test_runner_does_not_repeat_a_slow_rejection():
    # A 5xx arriving after the generation (CommonStack, 2026-09-27) is not free.
    clock = _Clock()
    sleeps = []
    call, calls = _scripted([_SDK_ERRORS["anthropic-529"], "unreached"], clock, 24.0)

    with pytest.raises(anthropic.APIStatusError):
        http_policy.call_with_retries(call, label="t", sleep=sleeps.append, clock=clock)

    assert len(calls) == 1


def test_runner_honours_retry_after_up_to_the_cap():
    sleeps = []
    call, _calls = _scripted([_SDK_ERRORS["anthropic-429-retry-after"], "ok"])
    assert http_policy.call_with_retries(call, label="t", sleep=sleeps.append) == "ok"
    assert sleeps == [7.0]

    too_long = _status_error(anthropic, 429, {"retry-after": "61"})
    call, calls = _scripted([too_long, "unreached"])
    with pytest.raises(anthropic.APIStatusError):
        http_policy.call_with_retries(call, label="t", sleep=sleeps.append)
    assert len(calls) == 1


def test_async_runner_matches_the_sync_one():
    sleeps = []
    steps = [_SDK_ERRORS["anthropic-connect-refused"], "ok"]

    async def call():
        step = steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    async def sleep(seconds):
        sleeps.append(seconds)

    assert asyncio.run(http_policy.acall_with_retries(call, label="t", sleep=sleep)) == "ok"
    assert sleeps == [4.0]


def test_runner_log_line_carries_no_exception_text(capsys):
    secret = "sk-ant-should-never-print"
    err = _chained(
        anthropic.APIConnectionError(message=secret, request=_REQUEST),
        httpx.ConnectError(secret, request=_REQUEST),
    )
    call, _calls = _scripted([err, "ok"])

    http_policy.call_with_retries(call, label="algo chat", sleep=lambda _s: None)

    out = capsys.readouterr().out
    assert "algo chat: APIConnectionError" in out
    assert secret not in out


# --- The strategy-chat panel --------------------------------------------------


def test_algo_chat_recovers_from_a_transient_failure_instead_of_the_rule_fallback(monkeypatch):
    """/api/algo/chat: a refused connection must not drop the user's edit (#592 review)."""

    reply = (
        '{"reply": "Switched to momentum.", "blocks": {"trading_algorithm": "Momentum"},'
        ' "updated_blocks": ["trading_algorithm"]}'
    )
    steps = [
        _SDK_ERRORS["anthropic-connect-refused"],
        SimpleNamespace(content=[SimpleNamespace(text=reply)]),
    ]

    def create(**_kwargs):
        step = steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    sleeps = []
    monkeypatch.setattr(
        algo_service,
        "_get_anthropic_client",
        lambda: SimpleNamespace(messages=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(algo_service, "_retry_sleep", sleeps.append)

    result = algo_service.process_chat("use momentum instead")

    assert result["reply"] == "Switched to momentum."
    assert result["blocks"]["trading_algorithm"] == "Momentum"
    assert sleeps == [4.0]
