"""Escalating the output ceiling on the first retry instead of the fifth.

An empty reply is what a reply looks like when reasoning ran out the output
ceiling before any text was emitted. Re-asking at the ceiling that just failed
asks for the same failure, and the single-prompt loop does that four times
before raising it. ``pipeline_runner`` escalates on its first retry and then
stops; these tests pin the two paths to the same policy -- including the
"and then stops" half, since repeating an escalated request is the same
request again, and escalating all four retries would make the tail case
*more* expensive rather than less.

The first request is deliberately untouched, so a decision that succeeds
first time issues exactly the request it does today. That is what keeps this a
cost change rather than a behaviour change.
"""

import json
import sqlite3
from datetime import datetime

import httpx
import pytest
from openai import _base_client as openai_base_client

from dashboard.backend.domain.backtesting import portfolio_manager as pm
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.infrastructure.llm import pipeline_runner
from dashboard.backend.infrastructure.llm.backtest_harness import (
    DEFAULT_MAX_OUTPUT_TOKENS as HARNESS_DEFAULT_MAX_OUTPUT_TOKENS,
)
from dashboard.backend.infrastructure.llm.execution import (
    service as execution_service_module,
)
from dashboard.backend.infrastructure.llm.execution.adapters import (
    base as adapters_base,
)
from dashboard.backend.infrastructure.llm.execution.adapters.openai import (
    OpenAICompatibleAdapter,
)
from dashboard.backend.infrastructure.llm.execution.client import (
    AnthropicCompatibleExecutionClient,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
    is_empty_reply,
)
from dashboard.backend.infrastructure.llm.execution.handoff import ExecutionHandoff
from dashboard.backend.infrastructure.llm.execution.models import BillingMode
from dashboard.backend.infrastructure.llm.pipeline_runner import (
    RECOVERY_MAX_OUTPUT_TOKENS,
    escalate_ceiling_on_retry,
)
from dashboard.backend.tests._execution_fixtures import (
    USER_ID,
    build_execution_service,
)

DECISION_JSON = '{"actions": []}'


class _Usage:
    input_tokens = 100
    output_tokens = 2000


class _Resp:
    """`text=None` is the thinking-only reply the retry loop exists for."""

    def __init__(self, text=None):
        self._text = text
        self.usage = _Usage()
        self.content = []


def _state():
    return {
        "timestamp": datetime(2026, 4, 1, 15, 30),
        "cash": 1000.0,
        "positions": [],
        "positions_value": 0.0,
        "total_equity": 1000.0,
        "market_signals": {
            "AAPL": {"price": 250.0, "rsi": 50.0, "macd": 0.0,
                     "macd_signal": 0.0, "sma_20": 250.0, "sma_50": 250.0,
                     "bb_upper": 260.0, "bb_lower": 240.0, "volume": 1000},
        },
    }


@pytest.fixture
def spy(monkeypatch):
    """Record the max_tokens of every request the loop issues."""
    calls = []
    plan = {"texts": []}

    _SENTINEL = object()

    def fake_request(client, *, prompt, model=None, max_tokens=_SENTINEL,
                     temperature=None, market_context=None):
        # Records the sentinel when the kwarg was not passed at all, so the
        # "unescalated calls are identical" claim is tested on call shape and
        # not just on the resolved value.
        calls.append(None if max_tokens is _SENTINEL else max_tokens)
        i = len(calls) - 1
        texts = plan["texts"]
        entry = texts[i] if i < len(texts) else DECISION_JSON
        # An exception in the plan is raised from the request itself, as the
        # execution layer does for an empty reply (#600).
        if isinstance(entry, Exception):
            raise entry
        return _Resp(entry)

    def fake_extract(response):
        if response._text is None:
            raise AttributeError(
                "No text content block in LLM response "
                "(content types: ['thinking'])")
        return response._text

    monkeypatch.setattr(pm, "_request_trading_decision", fake_request)
    monkeypatch.setattr(pm, "_extract_response_text", fake_extract)
    monkeypatch.setattr(pm, "_extract_token_usage", lambda r: (100, 2000))
    return calls, plan


def _set_flag(monkeypatch, on):
    if on:
        monkeypatch.setenv("LLM_ESCALATE_CEILING_ON_RETRY", "1")
    else:
        monkeypatch.delenv("LLM_ESCALATE_CEILING_ON_RETRY", raising=False)


def _run(manager, monkeypatch, on):
    _set_flag(monkeypatch, on)
    return manager.make_trading_decision_with_llm(
        _state(), llm_client=object(), model="m")


def _manager():
    return PortfolioManager(initial_capital=1000.0, allowed_symbols=["AAPL"])


# ------------------------------------------------------------------- flag ---

def test_the_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("LLM_ESCALATE_CEILING_ON_RETRY", raising=False)
    assert escalate_ceiling_on_retry() is False


@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("", False), ("nope", False),
])
def test_the_flag_parses_the_usual_spellings(monkeypatch, raw, expected):
    monkeypatch.setenv("LLM_ESCALATE_CEILING_ON_RETRY", raw)
    assert escalate_ceiling_on_retry() is expected


def test_the_flag_is_inert_when_the_ceiling_cannot_actually_rise(monkeypatch):
    """``RECOVERY_MAX_OUTPUT_TOKENS`` is ``max(configured, 4096)``.

    An operator running ``LLM_MAX_OUTPUT_TOKENS=8000`` therefore has a recovery
    request identical to the one that just failed, and honouring the flag would
    buy no escalation while costing three attempts against an intermittent
    empty reply. So it reads as off.
    """
    monkeypatch.setenv("LLM_ESCALATE_CEILING_ON_RETRY", "1")
    monkeypatch.setattr(
        pipeline_runner, "DEFAULT_MAX_OUTPUT_TOKENS", RECOVERY_MAX_OUTPUT_TOKENS
    )
    assert escalate_ceiling_on_retry() is False


def test_the_flag_is_on_when_the_ceiling_does_rise(monkeypatch):
    monkeypatch.setenv("LLM_ESCALATE_CEILING_ON_RETRY", "1")
    monkeypatch.setattr(
        pipeline_runner,
        "DEFAULT_MAX_OUTPUT_TOKENS",
        RECOVERY_MAX_OUTPUT_TOKENS - 1,
    )
    assert escalate_ceiling_on_retry() is True


# --------------------------------------------------- the first request ------

def test_a_first_time_success_is_one_unchanged_request(spy, monkeypatch):
    calls, plan = spy
    plan["texts"] = [DECISION_JSON]
    _run(_manager(), monkeypatch, on=True)
    assert calls == [None], "the flag must not touch a request that succeeds"


def test_the_first_request_is_never_escalated(spy, monkeypatch):
    calls, plan = spy
    plan["texts"] = [None, DECISION_JSON]
    _run(_manager(), monkeypatch, on=True)
    assert calls[0] is None


# ------------------------------------------------------- off vs on ----------

def test_off_retries_at_the_ceiling_that_just_failed(spy, monkeypatch):
    """Current behaviour: four attempts at the default, then the rescue."""
    calls, plan = spy
    plan["texts"] = [None, None, None, None, DECISION_JSON]
    _run(_manager(), monkeypatch, on=False)
    assert calls == [None, None, None, None, RECOVERY_MAX_OUTPUT_TOKENS]


def test_on_escalates_from_the_first_retry(spy, monkeypatch):
    calls, plan = spy
    plan["texts"] = [None, None, None, None, DECISION_JSON]
    _run(_manager(), monkeypatch, on=True)
    assert calls == [None, RECOVERY_MAX_OUTPUT_TOKENS]


def test_on_stops_after_the_one_escalated_retry(spy, monkeypatch):
    """The escalated retry REPLACES the same-ceiling ones; it is not applied to
    each of them.

    ``pipeline_runner`` retries once and stops, on the grounds that a reply
    already obtained at the recovery ceiling has nothing different left to ask
    for. Escalating all four retries instead would bill four recovery-ceiling
    calls for a model that never returns text -- strictly more expensive than
    the four cheap calls plus one rescue it replaced.
    """
    calls, plan = spy
    plan["texts"] = [None] * 12  # a model that never returns a text block
    _run(_manager(), monkeypatch, on=True)
    assert calls == [None, RECOVERY_MAX_OUTPUT_TOKENS]
    assert calls.count(RECOVERY_MAX_OUTPUT_TOKENS) == 1


def test_the_tail_case_is_cheaper_on_than_off(spy, monkeypatch):
    """Stated as tokens, because "fewer calls" is not by itself a saving."""
    calls, plan = spy
    plan["texts"] = [None] * 12
    _run(_manager(), monkeypatch, on=False)
    off = list(calls)
    calls.clear()
    _run(_manager(), monkeypatch, on=True)
    on = list(calls)

    def ceilings(seq):
        # ``None`` is "kwarg not passed", which resolves to the configured
        # default -- strictly below the recovery ceiling whenever the flag is
        # honoured at all. Counting it as zero would flatter the off case,
        # which is the one being argued against.
        return [
            c if c is not None else pipeline_runner.DEFAULT_MAX_OUTPUT_TOKENS
            for c in seq
        ]

    assert len(on) < len(off)
    assert sum(ceilings(on)) < sum(ceilings(off))


def test_on_reaches_the_working_ceiling_three_calls_sooner(spy, monkeypatch):
    """The saving, stated as a number: the ceiling that eventually works is
    reached on call 2 rather than call 5."""
    calls_off, plan = spy
    plan["texts"] = [None, None, None, None, DECISION_JSON]
    _run(_manager(), monkeypatch, on=False)
    first_off = calls_off.index(RECOVERY_MAX_OUTPUT_TOKENS)
    calls_off.clear()
    _run(_manager(), monkeypatch, on=True)
    first_on = calls_off.index(RECOVERY_MAX_OUTPUT_TOKENS)
    assert first_off == 4 and first_on == 1


def test_a_retry_that_succeeds_stops_the_loop(spy, monkeypatch):
    calls, plan = spy
    plan["texts"] = [None, DECISION_JSON]
    _run(_manager(), monkeypatch, on=True)
    assert calls == [None, RECOVERY_MAX_OUTPUT_TOKENS]


# ------------------------------------------- interaction with recovery ------

def test_a_reply_won_at_the_recovery_ceiling_does_not_retry_again(
        spy, monkeypatch):
    """`recovery_spent` must be set when the escalated retry is what produced
    the text; otherwise the post-parse truncation retry re-issues an identical
    request, which upstream's own comment says has nothing left to ask for."""
    calls, plan = spy
    # Unparseable text won at the escalated ceiling.
    plan["texts"] = [None, "not json at all"]
    _run(_manager(), monkeypatch, on=True)
    assert calls == [None, RECOVERY_MAX_OUTPUT_TOKENS], (
        "a third request would be the same request again")


def test_with_the_flag_off_the_truncation_retry_still_runs(spy, monkeypatch):
    calls, plan = spy
    plan["texts"] = ["not json at all"]
    _run(_manager(), monkeypatch, on=False)
    assert RECOVERY_MAX_OUTPUT_TOKENS in calls[1:], (
        "the existing post-parse recovery must be untouched when off")


def test_billing_counts_every_attempt_either_way(spy, monkeypatch):
    calls, plan = spy
    plan["texts"] = [None, DECISION_JSON]
    m = _manager()
    _run(m, monkeypatch, on=True)
    assert m.llm_calls == len(calls) == 2

    off = _manager()
    calls.clear()
    plan["texts"] = [None, None, DECISION_JSON]
    _run(off, monkeypatch, on=False)
    assert off.llm_calls == len(calls) == 3


# ------------------------------ execution-layer empty reply (#600) ----------
# The stand-in above raises the "No text content" AttributeError, which only
# the legacy client produces. The execution layer rejects an empty reply
# itself, before any message exists, as RESPONSE_INVALID marked
# ``empty_reply``, and settles what it was billed (#556) -- so a same-ceiling
# retry would be another billed generation likely to spend the same ceiling.
# It goes straight to the rescue call, as pipeline_runner's steps do,
# whatever the flag says; the rest of RESPONSE_INVALID is never retried.


def _empty_reply():
    return LLMExecutionError.for_empty_reply()


def _refused_request():
    """RESPONSE_INVALID with no marker: a contract break, not an empty reply."""
    return LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID)


class _HostedClient:
    """The one attribute of the execution client this loop reads."""

    fail_closed = True


@pytest.mark.parametrize("on", [False, True])
def test_an_execution_layer_empty_reply_goes_straight_to_the_rescue(
        spy, monkeypatch, capsys, on):
    calls, plan = spy
    plan["texts"] = [_empty_reply(), DECISION_JSON]
    _run(_manager(), monkeypatch, on=on)
    out = capsys.readouterr().out
    assert calls == [None, RECOVERY_MAX_OUTPUT_TOKENS]
    assert "going straight to the rescue call" in out
    assert "LLM decision error" not in out


@pytest.mark.parametrize("on", [False, True])
def test_an_empty_rescue_ends_a_hosted_run(spy, monkeypatch, on):
    """The terminal branch: the rescue came back empty too. A hosted run
    raises the marked error rather than continuing with no reply, and stops
    after the one rescue -- a third request would be the same request."""
    calls, plan = spy
    plan["texts"] = [_empty_reply()] * 5
    _set_flag(monkeypatch, on)
    with pytest.raises(LLMExecutionError) as raised:
        _manager().make_trading_decision_with_llm(
            _state(), llm_client=_HostedClient(), model="m", strict_llm=True)
    assert is_empty_reply(raised.value)
    assert calls == [None, RECOVERY_MAX_OUTPUT_TOKENS]


def test_a_contract_break_is_not_retried(spy, monkeypatch, capsys):
    """A request the client refuses before sending, or a settled result that
    fails validation, fails the same way every time -- and the second has
    already been billed. Neither is retried, nor logged as an empty reply."""
    calls, plan = spy
    plan["texts"] = [_refused_request()]
    _run(_manager(), monkeypatch, on=False)
    out = capsys.readouterr().out
    assert calls == [None]
    assert "Empty model response" not in out
    assert "LLM decision error" in out


def test_other_execution_errors_are_not_retried(spy, monkeypatch, capsys):
    calls, plan = spy
    plan["texts"] = [LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED)]
    _run(_manager(), monkeypatch, on=False)
    assert calls == [None], "billing, credential and quota failures stay fatal"
    assert "LLM decision error" in capsys.readouterr().out


def test_an_empty_truncation_retry_keeps_the_first_reply(
        spy, monkeypatch, capsys):
    calls, plan = spy
    plan["texts"] = ["not json at all", _empty_reply()]
    _run(_manager(), monkeypatch, on=False)
    out = capsys.readouterr().out
    assert calls == [None, RECOVERY_MAX_OUTPUT_TOKENS]
    assert "keeping the first attempt" in out
    assert "LLM decision error" not in out


def test_a_refused_truncation_retry_is_raised(spy, monkeypatch, capsys):
    calls, plan = spy
    plan["texts"] = ["not json at all", _refused_request()]
    _run(_manager(), monkeypatch, on=False)
    out = capsys.readouterr().out
    assert calls == [None, RECOVERY_MAX_OUTPUT_TOKENS]
    assert "keeping the first attempt" not in out
    assert "LLM decision error" in out


DEEPSEEK = "deepseek/deepseek-v4-pro"


def _deepseek_completion(content, *, completion_tokens, finish="stop"):
    """A chat completion as CommonStack answers it, ``reasoning_content`` and all."""
    return {
        "id": "chatcmpl-600",
        "object": "chat.completion",
        "created": 0,
        "model": DEEPSEEK,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": "RSI is neutral; weighing MACD against",
                },
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": completion_tokens,
            "total_tokens": 40 + completion_tokens,
        },
    }


def test_a_real_empty_reply_is_rescued_and_both_calls_are_billed(
        tmp_path, monkeypatch):
    """#600 end to end, on the lane prod failed on: DeepSeek V4 Pro through
    CommonStack on ATL Credits, in a strict, fail-closed run. Everything is
    real but the socket -- the OpenAI SDK, the CommonStack adapter that
    rejects the empty answer, the service that settles the 2000 tokens it
    was billed for (#556), the client and this loop. The model, not a hold,
    drove the step."""
    monkeypatch.delenv("LLM_ESCALATE_CEILING_ON_RETRY", raising=False)
    monkeypatch.delenv("BROKER_CREDENTIAL_VERIFICATION_PROXY", raising=False)
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-empty-reply-abcd")
    answers = [
        _deepseek_completion(None, completion_tokens=2000, finish="length"),
        _deepseek_completion(DECISION_JSON, completion_tokens=20),
    ]
    sent = []

    def commonstack(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=answers.pop(0), request=request)

    monkeypatch.setattr(
        adapters_base, "build_pinned_transport",
        lambda *_args, **_kwargs: httpx.MockTransport(commonstack),
    )
    monkeypatch.setattr(openai_base_client.time, "sleep", lambda _seconds: None)
    errors = []
    monkeypatch.setattr(
        execution_service_module.analytics_instrumentation,
        "emit_safe_error_event",
        lambda **kwargs: errors.append(kwargs),
    )
    service, credits_store = build_execution_service(
        tmp_path,
        monkeypatch,
        None,
        adapters={"commonstack": OpenAICompatibleAdapter()},
        # A real trading prompt's reservation at the harness's default
        # $1000/Mtok would exceed the seeded balance.
        usd_per_million_tokens=1.0,
    )
    client = AnthropicCompatibleExecutionClient(
        execution_service=service,
        handoff=ExecutionHandoff(
            user_id=USER_ID,
            run_id="run-600",
            billing_mode=BillingMode.PLATFORM_CREDITS,
            provider_id="commonstack",
            provider_ids=("commonstack",),
            model_id=DEEPSEEK,
            prompt_digest="a" * 64,
            nonce="n" * 16,
            issued_at=1,
            expires_at=2,
        ),
    )
    manager = _manager()

    decision = manager.make_trading_decision_with_llm(
        _state(), llm_client=client, model=DEEPSEEK, strict_llm=True)

    assert decision == {"actions": []}
    assert manager.llm_decisions == 1, "a hold would return the same dict"
    assert manager.llm_calls == 1, "only the reply that came back is a call"
    assert [body["max_tokens"] for body in sent] == [
        HARNESS_DEFAULT_MAX_OUTPUT_TOKENS,
        RECOVERY_MAX_OUTPUT_TOKENS,
    ]
    # The catalog's pin for this lane, as prod sends it: thinking off.
    assert [body.get("thinking") for body in sent] == [{"type": "disabled"}] * 2
    with sqlite3.connect(credits_store.db_path) as conn:
        rows = conn.execute(
            "SELECT call_index, status, settled_micro "
            "FROM credit_llm_reservations WHERE run_id = ? ORDER BY call_index",
            ("run-600",),
        ).fetchall()
    assert [(index, status) for index, status, _ in rows] == [
        (0, "settled"),
        (1, "settled"),
    ]
    # The empty reply is billed upstream for the reasoning that spent its
    # ceiling, so it is debited too: 2000 output tokens against the rescue's 20.
    assert rows[0][2] > rows[1][2] > 0
    assert [error["error_category"] for error in errors] == ["empty_response"]
