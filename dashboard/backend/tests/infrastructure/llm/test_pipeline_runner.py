"""Tests for sub-agent pipeline backtest execution."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
)
from dashboard.backend.infrastructure.llm.pipeline_runner import (
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
    def __init__(self, text, input_tokens=7, output_tokens=3):
        self.content = [SimpleNamespace(type="text", text=text)]
        self.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


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


def test_run_pipeline_decision_retries_response_invalid_once():
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
    assert client.messages.calls[1]["reasoning_effort"] == "none"
    assert client.messages.calls[0]["messages"] == client.messages.calls[1]["messages"]


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


def test_run_pipeline_decision_preserves_response_invalid_for_legacy_client():
    class _LegacyMessages:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if "reasoning_effort" in kwargs:
                raise TypeError("create() got an unexpected keyword argument")
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


@pytest.mark.parametrize(
    "category",
    [
        ExecutionErrorCategory.CREDENTIAL_INVALID,
        ExecutionErrorCategory.PROVIDER_UNAVAILABLE,
        ExecutionErrorCategory.PROVIDER_TIMEOUT,
        ExecutionErrorCategory.BILLING_FAILED,
        ExecutionErrorCategory.USAGE_UNAVAILABLE,
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
