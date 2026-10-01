"""Tests for sub-agent pipeline backtest execution."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
)
from dashboard.backend.infrastructure.llm.pipeline_runner import (
    DEFAULT_CEILING_SNAPSHOT_SYMBOLS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    RECOVERY_MAX_OUTPUT_TOKENS,
    _TRUNCATION_MIN_CHARS,
    _looks_like_truncated_json,
    apply_prompt_patches,
    is_last_bar_of_trading_day,
    pipeline_output_to_decision,
    recombine_pipeline,
    run_pipeline_decision,
    split_pipeline,
    trading_day_key,
    _build_step_prompt,
)


class _PipelineResponse:
    def __init__(
        self, text, input_tokens=7, output_tokens=3, stop_reason=None, content=None
    ):
        if content is None:
            content = [SimpleNamespace(type="text", text=text)]
        self.content = content
        self.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if stop_reason is not None:
            self.stop_reason = stop_reason


class _ThinkingOnlyResponse(_PipelineResponse):
    """A reasoning model that spent the whole reply on a thinking block."""

    def __init__(self, input_tokens=5, output_tokens=7):
        super().__init__(
            "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            content=[SimpleNamespace(type="thinking", thinking="...")],
        )


def _truncated_json(pad: int = 560) -> str:
    """A reply cut off inside a string value, long enough for the structural scan."""
    return '{"orders": [{"symbol": "AAPL", "side": "buy", "reason": "' + ("x" * pad)


class _SequencedMessages:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _PipelineClient:
    def __init__(self, outcomes):
        self.messages = _SequencedMessages(outcomes)


_PIPELINE = [
    {
        "id": "decision",
        "label": "Decision",
        "prompt": "Choose the action.",
        "outputFormat": '{"orders": []}',
    }
]


def test_pipeline_output_to_decision_orders():
    parsed = {
        "actions": [],
        "orders": [
            {"symbol": "AAPL", "side": "buy", "qty": 10, "reason": "momentum"},
            {"symbol": "MSFT", "side": "hold", "qty": 0},
        ]
    }
    decision = pipeline_output_to_decision(parsed)
    assert decision is not None
    assert len(decision["actions"]) == 2
    assert decision["actions"][0]["action"] == "buy"
    assert decision["actions"][0]["position_size"] == 10


def test_pipeline_output_to_decision_actions_passthrough():
    parsed = {
        "actions": [
            {
                "action": "sell",
                "symbol": "JPM",
                "confidence": 0.9,
                "reasoning": "overbought",
                "position_size": 5,
            }
        ]
    }
    decision = pipeline_output_to_decision(parsed)
    assert decision == parsed


def test_pipeline_output_to_decision_risk_actions():
    decision = pipeline_output_to_decision(
        {
            "risk_actions": [
                {
                    "symbol": "AAPL",
                    "action": "stop_loss",
                    "size_pct": 0.5,
                    "reason": "risk limit",
                }
            ]
        }
    )

    assert decision is not None
    assert decision["actions"] == [
        {
            "action": "sell",
            "symbol": "AAPL",
            "confidence": 0.8,
            "reasoning": "risk limit",
            "position_size": 50,
        }
    ]


@pytest.mark.parametrize(
    "parsed",
    [{"actions": []}, {"orders": []}, {"risk_actions": []}],
)
def test_pipeline_output_to_decision_empty_envelope_is_hold(parsed):
    assert pipeline_output_to_decision(parsed) == {"actions": []}


@pytest.mark.parametrize(
    "parsed",
    [
        None,
        {},
        {"orders": None},
        {"orders": "not-a-list"},
        {"orders": ["not-an-order"]},
        {"actions": [], "orders": ["not-an-order"]},
        {"orders": [], "risk_actions": ["not-a-risk-action"]},
    ],
)
def test_pipeline_output_to_decision_rejects_invalid_payload(parsed):
    assert pipeline_output_to_decision(parsed) is None


def test_build_step_prompt_includes_upstream_outputs():
    prompt = _build_step_prompt(
        step_index=1,
        step={
            "label": "Information to Signal",
            "prompt": "Generate signals.",
            "outputFormat": '{"signals": []}',
        },
        market_snapshot={"timestamp": "2026-01-01T10:00:00"},
        prior_outputs=[{"step": 1, "label": "Gather", "output": {"facts": []}}],
        is_last=True,
    )
    assert "UPSTREAM PIPELINE OUTPUTS" in prompt
    assert "EXECUTION RULES" in prompt
    assert "MARKET SNAPSHOT" not in prompt


def _snapshot_of(n):
    return {"top_signals": {f"S{i:02d}": {"price": 1.0} for i in range(n)}}


def test_build_step_prompt_serializes_the_snapshot_compactly():
    prompt = _build_step_prompt(
        step_index=0, step=_PIPELINE[0], market_snapshot=_snapshot_of(2),
        prior_outputs=[], is_last=True,
    )
    assert '{"top_signals":{"S00":{"price":1.0},"S01":{"price":1.0}}}' in prompt


@pytest.mark.parametrize(
    "n, expected",
    [
        (DEFAULT_CEILING_SNAPSHOT_SYMBOLS, DEFAULT_MAX_OUTPUT_TOKENS),
        (DEFAULT_CEILING_SNAPSHOT_SYMBOLS + 1, RECOVERY_MAX_OUTPUT_TOKENS),
        (30, RECOVERY_MAX_OUTPUT_TOKENS),
    ],
)
def test_a_wide_snapshot_starts_at_the_recovery_ceiling(n, expected):
    """A 30-order reply must not be paid for twice: truncated, then retried."""
    client = _PipelineClient([_PipelineResponse('{"orders": []}')])
    run_pipeline_decision(
        client, pipeline=_PIPELINE, market_snapshot=_snapshot_of(n)
    )
    assert client.messages.calls[0]["max_tokens"] == expected


def test_a_wide_snapshot_does_not_repeat_a_truncated_recovery_sized_reply():
    """Already at the recovery ceiling: a retry would be the same request."""
    client = _PipelineClient(
        [_PipelineResponse(_truncated_json(), output_tokens=RECOVERY_MAX_OUTPUT_TOKENS)]
    )
    decision, _usage, calls, _steps = run_pipeline_decision(
        client, pipeline=_PIPELINE, market_snapshot=_snapshot_of(30)
    )
    assert decision is None
    assert calls == 1
    assert len(client.messages.calls) == 1


def test_a_wide_snapshot_still_retries_an_empty_reply():
    client = _PipelineClient(
        [
            LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID),
            _PipelineResponse('{"orders": []}'),
        ]
    )
    decision, _usage, _calls, _steps = run_pipeline_decision(
        client, pipeline=_PIPELINE, market_snapshot=_snapshot_of(30)
    )
    assert decision == {"actions": []}
    assert [c["max_tokens"] for c in client.messages.calls] == [
        RECOVERY_MAX_OUTPUT_TOKENS, RECOVERY_MAX_OUTPUT_TOKENS,
    ]


def test_split_pipeline_strips_post_trade():
    pipeline = [
        {"id": "a", "presetKey": "info_gather", "prompt": "gather"},
        {"id": "b", "presetKey": "post_trade_analysis", "prompt": "review"},
        {"id": "c", "presetKey": "info_to_signal", "prompt": "signal"},
    ]
    decision, post = split_pipeline(pipeline)
    assert [s["id"] for s in decision] == ["a", "c"]
    assert [s["id"] for s in post] == ["b"]
    assert [s["id"] for s in recombine_pipeline(decision, post)] == ["a", "c", "b"]


def test_apply_prompt_patches_by_id_and_skips_post_trade():
    decision = [
        {"id": "s1", "presetKey": "info_gather", "prompt": "old gather"},
        {"id": "s2", "presetKey": "info_to_signal", "prompt": "old signal"},
    ]
    patched, applied = apply_prompt_patches(
        decision,
        [
            {
                "step_id": "s1",
                "new_prompt": "new gather",
                "change_rationale": "missed news filter",
            },
            {
                "presetKey": "post_trade_analysis",
                "new_prompt": "should not apply",
            },
            {
                "presetKey": "info_to_signal",
                "new_prompt": "new signal",
            },
            {
                "step_id": "missing",
                "new_prompt": "",
            },
        ],
    )
    assert patched[0]["prompt"] == "new gather"
    assert patched[1]["prompt"] == "new signal"
    assert len(applied) == 2
    assert decision[0]["prompt"] == "old gather"  # deepcopy, original untouched


def test_run_pipeline_decision_retries_response_invalid_with_reasoning_preserved():
    client = _PipelineClient(
        [
            LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID),
            _PipelineResponse('{"orders": []}', input_tokens=11, output_tokens=4),
        ]
    )

    decision, usage, calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
        model="qwen/qwen3.7-plus",
    )

    assert decision == {"actions": []}
    assert usage == (11, 4)
    assert calls == 1
    assert len(client.messages.calls) == 2
    assert "reasoning_effort" not in client.messages.calls[0]
    assert "reasoning_effort" not in client.messages.calls[1]
    assert client.messages.calls[0]["max_tokens"] < RECOVERY_MAX_OUTPUT_TOKENS
    assert client.messages.calls[1]["max_tokens"] == RECOVERY_MAX_OUTPUT_TOKENS
    assert (
        client.messages.calls[0]["messages"]
        == client.messages.calls[1]["messages"]
    )


def test_run_pipeline_decision_reraises_after_one_failed_retry():
    client = _PipelineClient(
        [
            LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID),
            LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID),
        ]
    )

    with pytest.raises(LLMExecutionError) as error:
        run_pipeline_decision(
            client,
            pipeline=_PIPELINE,
            market_snapshot={"top_signals": {}},
        )

    assert error.value.category is ExecutionErrorCategory.RESPONSE_INVALID
    assert len(client.messages.calls) == 2


def test_run_pipeline_decision_retries_truncated_json_once():
    client = _PipelineClient(
        [
            _PipelineResponse(_truncated_json(), input_tokens=13, output_tokens=2000),
            _PipelineResponse('{"orders": []}', input_tokens=11, output_tokens=4),
        ]
    )

    decision, usage, calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
        model="google/gemini-3.1-pro-preview",
    )

    assert decision == {"actions": []}
    assert usage == (24, 2004)
    assert calls == 2
    assert len(client.messages.calls) == 2
    assert "reasoning_effort" not in client.messages.calls[0]
    assert "reasoning_effort" not in client.messages.calls[1]
    assert client.messages.calls[1]["max_tokens"] == RECOVERY_MAX_OUTPUT_TOKENS
    assert client.messages.calls[0]["messages"] == client.messages.calls[1]["messages"]


def test_run_pipeline_decision_retries_short_reasoning_truncation_once():
    """A reasoning-heavy response can be truncated below the old 512-char floor."""
    truncated = (
        '{"orders": [{"symbol": "BA", "side": "hold", "qty": 2, '
        '"order_type": "market", "limit_price": null, "reason": "Experi'
    )
    assert len(truncated) >= 64
    client = _PipelineClient(
        [
            _PipelineResponse(truncated, input_tokens=13, output_tokens=2000),
            _PipelineResponse('{"orders": []}', input_tokens=11, output_tokens=4),
        ]
    )

    decision, usage, calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
        model="qwen/qwen3.7-plus",
    )

    assert decision == {"actions": []}
    assert usage == (24, 2004)
    assert calls == 2
    assert len(client.messages.calls) == 2
    assert client.messages.calls[1]["max_tokens"] == RECOVERY_MAX_OUTPUT_TOKENS


def test_run_pipeline_decision_retries_short_truncation_below_the_ceiling():
    # Same ~110-char reply as the test above, but with usage well under the
    # ceiling so only the structural scan can justify the retry: this is what
    # pins #422's 64-char floor (a 512-char floor would ship green otherwise).
    truncated = (
        '{"orders": [{"symbol": "BA", "side": "hold", "qty": 2, '
        '"order_type": "market", "limit_price": null, "reason": "Experi'
    )
    assert _TRUNCATION_MIN_CHARS <= len(truncated) < 512
    client = _PipelineClient(
        [
            _PipelineResponse(truncated, input_tokens=13, output_tokens=900),
            _PipelineResponse('{"orders": []}', input_tokens=11, output_tokens=4),
        ]
    )

    decision, usage, calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
    )

    assert decision == {"actions": []}
    assert usage == (24, 904)
    assert calls == 2
    assert client.messages.calls[1]["max_tokens"] == RECOVERY_MAX_OUTPUT_TOKENS


def test_run_pipeline_decision_does_not_retry_short_malformed_json():
    client = _PipelineClient([_PipelineResponse("{\"orders\": [", output_tokens=12)])

    decision, usage, calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
    )

    assert decision is None
    assert usage == (7, 12)
    assert calls == 1
    assert len(client.messages.calls) == 1


def test_run_pipeline_decision_preserves_response_invalid_after_retry():
    class _LegacyMessages:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            raise LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID)

    client = SimpleNamespace(messages=_LegacyMessages())

    with pytest.raises(LLMExecutionError) as error:
        run_pipeline_decision(
            client,
            pipeline=_PIPELINE,
            market_snapshot={"top_signals": {}},
        )

    assert error.value.category is ExecutionErrorCategory.RESPONSE_INVALID
    assert len(client.messages.calls) == 2


def test_run_pipeline_decision_propagates_unrelated_retry_type_error():
    class _BrokenMessages:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["max_tokens"] != RECOVERY_MAX_OUTPUT_TOKENS:
                raise LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID)
            raise TypeError("client serialization failed")

    client = SimpleNamespace(messages=_BrokenMessages())

    with pytest.raises(TypeError, match="client serialization failed"):
        run_pipeline_decision(
            client,
            pipeline=_PIPELINE,
            market_snapshot={"top_signals": {}},
        )

    assert len(client.messages.calls) == 2


@pytest.mark.parametrize(
    "category",
    [
        ExecutionErrorCategory.CREDENTIAL_MISSING,
        ExecutionErrorCategory.CREDENTIAL_INVALID,
        ExecutionErrorCategory.PROVIDER_UNAVAILABLE,
        ExecutionErrorCategory.PROVIDER_TIMEOUT,
        ExecutionErrorCategory.BILLING_FAILED,
        ExecutionErrorCategory.USAGE_UNAVAILABLE,
        ExecutionErrorCategory.WORKER_FAILED,
    ],
)
def test_run_pipeline_decision_does_not_retry_other_execution_errors(category):
    client = _PipelineClient([LLMExecutionError(category)])

    with pytest.raises(LLMExecutionError) as error:
        run_pipeline_decision(
            client,
            pipeline=_PIPELINE,
            market_snapshot={"top_signals": {}},
        )

    assert error.value.category is category
    assert len(client.messages.calls) == 1


def test_run_pipeline_decision_does_not_retry_invalid_business_json():
    client = _PipelineClient([_PipelineResponse("not-json")])

    decision, usage, calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
    )

    assert decision is None
    assert usage == (7, 3)
    assert calls == 1
    assert len(client.messages.calls) == 1


def test_trading_day_boundary_helpers():
    day1 = [
        datetime(2024, 1, 2, 10, 0),
        datetime(2024, 1, 2, 11, 0),
        datetime(2024, 1, 2, 15, 0),
    ]
    day2 = day1 + [datetime(2024, 1, 3, 10, 0)]
    assert trading_day_key(day1[0]) == "2024-01-02"
    assert is_last_bar_of_trading_day(day1, 0) is False
    assert is_last_bar_of_trading_day(day1, 2) is True
    assert is_last_bar_of_trading_day(day2, 2) is True
    assert is_last_bar_of_trading_day(day2, 3) is True


def test_run_pipeline_decision_retries_short_reply_when_provider_reports_ceiling():
    # A provider-reported stop reason is exact: it fires below the structural
    # scan's length floor, and regardless of script (CJK output reaches the
    # ceiling at a fraction of the characters).
    client = _PipelineClient(
        [
            _PipelineResponse(
                '{"orders": [{"symbol": "600519.SH", "reason": "茅台强势',
                input_tokens=9,
                output_tokens=40,
                stop_reason="max_tokens",
            ),
            _PipelineResponse('{"orders": []}', input_tokens=11, output_tokens=4),
        ]
    )

    decision, usage, calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
    )

    assert decision == {"actions": []}
    assert usage == (20, 44)
    assert calls == 2
    assert client.messages.calls[1]["max_tokens"] == RECOVERY_MAX_OUTPUT_TOKENS


def test_run_pipeline_decision_retries_when_output_tokens_reach_the_ceiling():
    # No stop reason on the response (the execution client's compatibility
    # shape), but usage says the reply used every output token it was allowed.
    client = _PipelineClient(
        [
            _PipelineResponse(
                '{"orders": [',
                input_tokens=9,
                output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            ),
            _PipelineResponse('{"orders": []}', input_tokens=11, output_tokens=4),
        ]
    )

    decision, _usage, calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
    )

    assert decision == {"actions": []}
    assert calls == 2
    assert len(client.messages.calls) == 2
    assert client.messages.calls[1]["max_tokens"] == RECOVERY_MAX_OUTPUT_TOKENS


@pytest.mark.parametrize(
    "prefix",
    [
        "Here is my trading decision:\n",
        "```JSON\n",
        "\ufeff",
    ],
)
def test_run_pipeline_decision_retries_truncated_json_behind_a_preamble(prefix):
    # The structural scan tolerates what parse_llm_response tolerates — prose
    # before the object, code fences of any case, a BOM — so a truncation the
    # parser would have seen is not hidden from the retry by its preamble.
    client = _PipelineClient(
        [
            _PipelineResponse(prefix + _truncated_json(), output_tokens=900),
            _PipelineResponse('{"orders": []}', input_tokens=11, output_tokens=4),
        ]
    )

    decision, _usage, calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
    )

    assert decision == {"actions": []}
    assert calls == 2


def test_looks_like_truncated_json_scope():
    assert _looks_like_truncated_json(_truncated_json()) is True
    assert _looks_like_truncated_json("```json\n" + _truncated_json() + "\n```") is True
    # Complete objects, short fragments, mismatched delimiters (a malformed
    # reply, not a cut-off one), arrays and objects that never named a
    # decision key are not retried.
    assert _looks_like_truncated_json('{"orders": [], "note": "' + "x" * 600 + '"}') is False
    assert _looks_like_truncated_json('{"orders": [') is False
    assert _looks_like_truncated_json('{"orders": [1, 2}, "b": "' + "x" * 600) is False
    assert _looks_like_truncated_json("[" + "x" * 600) is False
    assert _looks_like_truncated_json('{"summary": "' + "x" * 600) is False
    # The length floor is exact.
    prefix = '{"orders": [{"symbol": "AAPL", "reason": "'
    at_floor = prefix + "x" * (_TRUNCATION_MIN_CHARS - len(prefix))
    assert _looks_like_truncated_json(at_floor) is True
    assert _looks_like_truncated_json(at_floor[:-1]) is False
    # A brace inside a prose preamble is not the object: this reply is
    # complete, and a retry would only buy a second billed call.
    complete = '{"orders": [{"symbol": "AAPL", "side": "buy", "reason": "' + "x" * 600 + '"}]}'
    assert _looks_like_truncated_json("Analysis {see below}\n" + complete) is False
    assert _looks_like_truncated_json("Analysis {see below\n" + complete) is False


def test_run_pipeline_decision_truncation_retry_degrades_when_retry_is_invalid():
    # The first reply was real and billed; a second reply the provider rejects
    # as invalid must not turn that into an exception that loses its usage.
    client = _PipelineClient(
        [
            _PipelineResponse(_truncated_json(), input_tokens=13, output_tokens=2000),
            LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID),
        ]
    )

    decision, usage, calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
    )

    assert decision is None
    assert usage == (13, 2000)
    assert calls == 1
    assert len(client.messages.calls) == 2


def test_run_pipeline_decision_truncation_retry_degrades_when_retry_has_no_text():
    client = _PipelineClient(
        [
            _PipelineResponse(_truncated_json(), input_tokens=13, output_tokens=2000),
            _ThinkingOnlyResponse(input_tokens=5, output_tokens=7),
        ]
    )

    decision, usage, calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
    )

    assert decision is None
    assert usage == (18, 2007)
    assert calls == 2


@pytest.mark.parametrize(
    "category",
    [
        ExecutionErrorCategory.CREDENTIAL_INVALID,
        ExecutionErrorCategory.PROVIDER_UNAVAILABLE,
        ExecutionErrorCategory.BILLING_FAILED,
    ],
)
def test_run_pipeline_decision_truncation_retry_propagates_infrastructure_errors(
    category,
):
    client = _PipelineClient(
        [
            _PipelineResponse(_truncated_json(), output_tokens=2000),
            LLMExecutionError(category),
        ]
    )

    with pytest.raises(LLMExecutionError) as error:
        run_pipeline_decision(
            client,
            pipeline=_PIPELINE,
            market_snapshot={"top_signals": {}},
        )

    assert error.value.category is category
    assert len(client.messages.calls) == 2


def test_run_pipeline_decision_truncation_retry_propagates_client_faults():
    # Only a provider-classified unusable reply degrades; a client fault on
    # the retry is a real error and must surface for diagnostics.
    client = _PipelineClient(
        [
            _PipelineResponse(_truncated_json(), output_tokens=2000),
            TypeError("client serialization failed"),
        ]
    )

    with pytest.raises(TypeError, match="client serialization failed"):
        run_pipeline_decision(
            client,
            pipeline=_PIPELINE,
            market_snapshot={"top_signals": {}},
        )


def test_run_pipeline_decision_retries_at_most_once_per_step():
    # The response_invalid retry already ran with the recovery budget; a
    # truncated reply from it has nothing different left to send.
    client = _PipelineClient(
        [
            LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID),
            _PipelineResponse(_truncated_json(), input_tokens=13, output_tokens=2000),
        ]
    )

    decision, usage, calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
    )

    assert decision is None
    assert usage == (13, 2000)
    assert calls == 1
    assert len(client.messages.calls) == 2


def test_run_pipeline_decision_retries_thinking_only_reply_at_the_ceiling():
    # A reasoning model that spent the whole first reply thinking, and hit the
    # ceiling doing it, is exactly what the provider's stop reason is for.
    client = _PipelineClient(
        [
            _ThinkingOnlyResponse(input_tokens=13, output_tokens=2000),
            _PipelineResponse('{"orders": []}', input_tokens=11, output_tokens=4),
        ]
    )

    decision, usage, calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
    )

    assert decision == {"actions": []}
    assert usage == (24, 2004)
    assert calls == 2
    assert client.messages.calls[1]["max_tokens"] == RECOVERY_MAX_OUTPUT_TOKENS


def test_run_pipeline_decision_thinking_only_reply_below_the_ceiling_ends_cleanly():
    # No text and no ceiling: nothing to retry, but the call was real and
    # billed, so the step ends with None and its usage rather than raising.
    client = _PipelineClient([_ThinkingOnlyResponse(input_tokens=13, output_tokens=40)])

    decision, usage, calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
    )

    assert decision is None
    assert usage == (13, 40)
    assert calls == 1
    assert len(client.messages.calls) == 1


# ---------------------------------------------------------------------------
# Post-trade steps on a hosted (fail_closed) client (#574): a provider outage
# or an empty reply holds the prompts and is recorded; every other execution
# failure stays fatal.
# ---------------------------------------------------------------------------

_POST_TRADE_STEPS = [
    {"id": "review", "label": "Review", "presetKey": "post_trade_analysis"}
]
_POST_TRADE_CONTEXT = {"trading_day": "2026-04-15"}
_PATCH_REPLY = (
    '{"summary": "tightened", "prompt_problems": [], "prompt_patches": '
    '[{"step_id": "decision", "new_prompt": "Choose carefully."}]}'
)


class _HostedPipelineClient(_PipelineClient):
    fail_closed = True


def _run_post_trade(client, steps=_POST_TRADE_STEPS):
    from dashboard.backend.infrastructure.llm.pipeline_runner import (
        run_post_trade_analysis,
    )

    return run_post_trade_analysis(
        client,
        post_trade_steps=steps,
        episode_context=_POST_TRADE_CONTEXT,
        decision_pipeline=_PIPELINE,
        model="test-model",
    )


def _skipped(reason, *, step=1, step_id="review", label="Review"):
    return {"step": step, "step_id": step_id, "label": label, "reason": reason}


@pytest.mark.parametrize(
    "category",
    [
        ExecutionErrorCategory.PROVIDER_TIMEOUT,
        ExecutionErrorCategory.PROVIDER_UNAVAILABLE,
        # The execution layer's empty reply: a reasoning model whose thinking
        # ate the ceiling. Released like an outage, so holding is unbilled.
        ExecutionErrorCategory.RESPONSE_INVALID,
    ],
)
def test_post_trade_holdable_failure_holds_prompts_on_hosted_client(category, capsys):
    client = _HostedPipelineClient([LLMExecutionError(category)])

    patched, record, tokens, calls = _run_post_trade(client)

    assert patched == _PIPELINE
    assert tokens == (0, 0)
    assert calls == 0
    assert record["applied_patches"] == []
    assert record["skipped_steps"] == [_skipped(category.value)]
    assert len(client.messages.calls) == 1
    out = capsys.readouterr().out
    # ``ERROR: llm.`` is the prefix the parent relays to the service log live.
    assert (
        f"ERROR: llm.post_trade_skipped category={category.value} step=1 step_id=review"
        in out.splitlines()
    )
    # An infrastructure failure must not be reported as a model output problem.
    assert "unparseable JSON" not in out


@pytest.mark.parametrize(
    "category",
    [
        ExecutionErrorCategory.BILLING_FAILED,
        ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED,
        ExecutionErrorCategory.CREDENTIAL_INVALID,
        ExecutionErrorCategory.CREDENTIAL_MISSING,
        ExecutionErrorCategory.ACCOUNT_RESTRICTED,
        ExecutionErrorCategory.USAGE_UNAVAILABLE,
    ],
)
def test_post_trade_other_execution_failures_stay_fatal(category):
    client = _HostedPipelineClient([LLMExecutionError(category)])

    with pytest.raises(LLMExecutionError) as excinfo:
        _run_post_trade(client)

    assert excinfo.value.category is category


def test_post_trade_non_execution_error_on_hosted_client_stays_fatal():
    client = _HostedPipelineClient([RuntimeError("boom")])

    with pytest.raises(RuntimeError):
        _run_post_trade(client)


def test_post_trade_thinking_only_reply_holds_with_usage_on_hosted_client():
    """A reply with no text block is unparseable, not a fault (cf. the decision path)."""
    client = _HostedPipelineClient([_ThinkingOnlyResponse(input_tokens=5, output_tokens=7)])

    patched, record, tokens, calls = _run_post_trade(client)

    assert patched == _PIPELINE
    assert tokens == (5, 7)
    assert calls == 1
    assert record["skipped_steps"] == [_skipped("no_text")]


def test_post_trade_unparseable_reply_is_recorded_as_such():
    client = _HostedPipelineClient([_PipelineResponse("not json at all")])

    patched, record, _tokens, calls = _run_post_trade(client)

    assert patched == _PIPELINE
    assert calls == 1
    assert record["skipped_steps"] == [_skipped("unparseable")]


_TWO_POST_TRADE_STEPS = [
    {"id": "review", "label": "Review", "presetKey": "post_trade_analysis"},
    {"id": "audit", "label": "Audit", "presetKey": "post_trade_analysis"},
]


@pytest.mark.parametrize(
    "category",
    [ExecutionErrorCategory.PROVIDER_TIMEOUT, ExecutionErrorCategory.PROVIDER_UNAVAILABLE],
)
def test_post_trade_outage_skips_the_rest_of_the_day(category):
    """Each further request would pay the full retry and failover wait again."""
    client = _HostedPipelineClient([LLMExecutionError(category)])

    patched, record, _tokens, calls = _run_post_trade(client, _TWO_POST_TRADE_STEPS)

    assert len(client.messages.calls) == 1
    assert calls == 0
    assert patched == _PIPELINE
    assert record["skipped_steps"] == [
        _skipped(category.value),
        _skipped("not_attempted", step=2, step_id="audit", label="Audit"),
    ]


def test_post_trade_empty_reply_does_not_skip_the_next_step():
    """An empty reply is about that prompt, not the provider: the next step still runs."""
    client = _HostedPipelineClient(
        [
            LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID),
            _PipelineResponse(_PATCH_REPLY),
        ]
    )

    patched, record, _tokens, calls = _run_post_trade(client, _TWO_POST_TRADE_STEPS)

    assert len(client.messages.calls) == 2
    assert calls == 1
    assert patched[0]["prompt"] == "Choose carefully."
    assert record["summary"] == "tightened"
    assert record["skipped_steps"] == [_skipped("response_invalid")]


def test_post_trade_success_records_no_skipped_steps():
    client = _HostedPipelineClient([_PipelineResponse(_PATCH_REPLY)])

    patched, record, _tokens, calls = _run_post_trade(client)

    assert calls == 1
    assert patched[0]["prompt"] == "Choose carefully."
    assert record["skipped_steps"] == []


@pytest.mark.parametrize(
    ("step_id", "expected"),
    [("review", "review"), (7, "7"), ("复盘", "-"), ("a b=c", "-"), (None, "-")],
)
def test_post_trade_skipped_line_never_prints_the_label(step_id, expected, capsys):
    """The position and a validated id identify the step; the user-authored label never does."""
    steps = [
        {"id": step_id, "label": "复盘 分析=1", "presetKey": "post_trade_analysis"}
    ]
    client = _HostedPipelineClient(
        [LLMExecutionError(ExecutionErrorCategory.PROVIDER_TIMEOUT)]
    )

    _run_post_trade(client, steps)

    lines = capsys.readouterr().out.splitlines()
    assert (
        f"ERROR: llm.post_trade_skipped category=provider_timeout step=1 step_id={expected}"
        in lines
    )
    # The step number matches the human-readable line printed just before it.
    assert any(line.startswith("📉 Post-trade analysis 1/1: 复盘 分析=1") for line in lines)


def test_post_trade_outages_leave_the_decision_step_strike_intact():
    """A held post-trade step must not spend the decision-step outage budget.

    The budget here is one strike. After two days of post-trade outages, a
    decision-step outage must still be absorbed rather than abort the run --
    the property the "no strike" design exists for, checked through the real
    ``_run_daily_post_trade`` and the real strike accounting.
    """
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester
    from dashboard.backend.domain.backtesting.portfolio_manager import (
        LLMDecisionError,
        PortfolioManager,
    )

    manager = PortfolioManager(100000)
    manager.strict_llm_total_steps = 49
    assert manager.strict_llm_fallback_budget() == 1
    client = _HostedPipelineClient(
        [
            LLMExecutionError(ExecutionErrorCategory.PROVIDER_TIMEOUT),
            LLMExecutionError(ExecutionErrorCategory.PROVIDER_UNAVAILABLE),
        ]
    )
    backtester = SimpleNamespace(
        use_llm=True,
        llm_client=client,
        model="test-model",
        pipeline=list(_PIPELINE) + list(_POST_TRADE_STEPS),
        prompt_adaptations=[],
        _current_equity=lambda _manager: 100000.0,
    )

    for day in ("2026-04-15", "2026-04-16"):
        HourlyBacktester._run_daily_post_trade(
            backtester,
            manager=manager,
            day_episode={"trading_day": day, "day_start_equity": 100000.0},
            post_trade_steps=_POST_TRADE_STEPS,
        )

    assert manager.strict_llm_fallbacks == 0
    assert manager.llm_calls == 0
    assert [step["id"] for step in backtester.pipeline] == ["decision", "review"]
    assert [len(r["skipped_steps"]) for r in backtester.prompt_adaptations] == [1, 1]
    held = manager._absorb_strict_llm_failure(
        {}, LLMDecisionError("decision-step outage"), hold=True
    )
    assert held == {"actions": []}
