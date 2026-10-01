"""The execution client imposes the catalog's sampling policy on every call.

The policy is resolved from the signed handoff's ``model_id`` inside
``AnthropicCompatibleExecutionClient``, so no call site above it -- pipeline
step, recovery retry, post-trade review, or one written later -- has to
thread anything, and none can send something else by forgetting to. These
tests drive the real pipeline runner through the real client to pin that.
"""
import pytest

from dashboard.backend.domain.backtesting import engine as engine_module
from dashboard.backend.domain.model_providers.execution_catalog import (
    ATL_EXECUTION_MODELS,
)
from dashboard.backend.infrastructure.llm.execution.client import (
    AnthropicCompatibleExecutionClient,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
)
from dashboard.backend.infrastructure.llm.execution.handoff import ExecutionHandoff
from dashboard.backend.infrastructure.llm.execution.models import (
    BillingEvidence,
    BillingMode,
    LLMExecutionResult,
    LLMUsage,
)
from dashboard.backend.infrastructure.llm.pipeline_runner import (
    run_pipeline_decision,
    run_post_trade_analysis,
)

_PIPELINE = [
    {
        "id": "decision",
        "label": "Decision",
        "prompt": "Choose the action.",
        "outputFormat": '{"orders": []}',
    }
]
_POST_TRADE_STEPS = [
    {"id": "review", "label": "Review", "presetKey": "post_trade_analysis"}
]
_PATCH_REPLY = (
    '{"summary": "tightened", "prompt_problems": [], "prompt_patches": '
    '[{"step_id": "decision", "new_prompt": "Choose carefully."}]}'
)


class _Service:
    """Answers each request in turn; an exception in the queue is raised."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _result(model_id, text='{"orders": []}', *, provider_id="commonstack", wire=None):
    return LLMExecutionResult(
        text=text,
        provider_id=provider_id,
        requested_provider_id="commonstack",
        model_id=model_id,
        usage=LLMUsage(input_tokens=10, output_tokens=5),
        billing=BillingEvidence(
            billing_source=BillingMode.PLATFORM_CREDITS,
            usage_authority="provider_reported_cost",
            provider_cost_usd=0.01,
        ),
        sampling_wire=wire,
    )


def _client(model_id, replies):
    handoff = ExecutionHandoff(
        user_id=7,
        run_id="run-sampling",
        billing_mode=BillingMode.PLATFORM_CREDITS,
        provider_id="commonstack",
        provider_ids=("commonstack", "openrouter"),
        model_id=model_id,
        prompt_digest="a" * 64,
        nonce="n" * 16,
        issued_at=1,
        expires_at=2,
    )
    service = _Service(replies)
    return AnthropicCompatibleExecutionClient(
        execution_service=service, handoff=handoff
    ), service


@pytest.mark.parametrize("model", ATL_EXECUTION_MODELS, ids=lambda m: m.catalog_id)
def test_every_catalog_model_gets_its_policy_on_a_pipeline_step(model):
    client, service = _client(model.catalog_id, [_result(model.catalog_id)])

    run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
        model=model.catalog_id,
    )

    (request,) = service.requests
    assert request.temperature == model.sampling.temperature
    assert request.reasoning_effort == model.sampling.reasoning_effort


def test_the_recovery_retry_carries_the_same_policy():
    """A retry that dropped the policy would be a different request."""
    model_id = "deepseek/deepseek-v4-pro"
    client, service = _client(
        model_id,
        [
            LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID),
            _result(model_id),
        ],
    )

    run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
        model=model_id,
    )

    assert len(service.requests) == 2
    first, retry = service.requests
    assert retry.usage_policy.max_output_tokens > first.usage_policy.max_output_tokens
    assert [(r.temperature, r.reasoning_effort) for r in service.requests] == [
        (0.0, "none"),
        (0.0, "none"),
    ]


def test_the_post_trade_review_carries_the_policy():
    """The call #594's stand-in `self` reached: nothing threads it any more."""
    model_id = "qwen/qwen3.7-plus"
    client, service = _client(model_id, [_result(model_id, _PATCH_REPLY)])

    run_post_trade_analysis(
        client,
        post_trade_steps=_POST_TRADE_STEPS,
        episode_context={"trading_day": "2026-04-15"},
        decision_pipeline=_PIPELINE,
        model=model_id,
    )

    (request,) = service.requests
    assert (request.temperature, request.reasoning_effort) == (0.0, "none")


def test_the_policy_overrides_whatever_the_caller_passed():
    """Both halves at once: a caller's temperature on GPT-5.5 is a 400."""
    model_id = "openai/gpt-5.5"
    client, service = _client(model_id, [_result(model_id, "BUY")])

    client.messages.create(
        model=model_id,
        max_tokens=100,
        temperature=0.7,
        reasoning_effort="high",
        messages=[{"role": "user", "content": "Trade now"}],
    )

    (request,) = service.requests
    assert (request.temperature, request.reasoning_effort) == (None, "low")


def test_provider_default_sends_nothing_even_when_the_caller_asks():
    model_id = "google/gemini-3.1-pro-preview"
    client, service = _client(model_id, [_result(model_id, "BUY")])

    client.messages.create(
        model=model_id,
        max_tokens=100,
        temperature=0.3,
        messages=[{"role": "user", "content": "Trade now"}],
    )

    (request,) = service.requests
    assert (request.temperature, request.reasoning_effort) == (None, None)
    assert client.sampling_record()["policy"] == "provider_default"


def test_the_record_names_the_policy_and_each_lanes_wire_shape():
    """A failover lane sends the same policy in a different, unprobed shape."""
    model_id = "deepseek/deepseek-v4-pro"
    client, _service = _client(
        model_id,
        [
            _result(model_id, "BUY", wire="temperature=0.0;thinking=disabled"),
            _result(
                model_id,
                "BUY",
                provider_id="openrouter",
                wire="temperature=0.0;reasoning.effort=none,enabled=false",
            ),
        ],
    )
    for _ in range(2):
        client.messages.create(
            model=model_id,
            max_tokens=100,
            messages=[{"role": "user", "content": "Trade now"}],
        )

    assert client.sampling_record() == {
        "temperature": 0.0,
        "reasoning_effort": "none",
        "policy": "pinned_v1",
        "model": model_id,
        "wire": {
            "commonstack": "temperature=0.0;thinking=disabled",
            "openrouter": "temperature=0.0;reasoning.effort=none,enabled=false",
        },
    }


def test_a_run_with_no_completed_call_still_records_its_policy():
    client, _service = _client("anthropic/claude-haiku-4-5", [])
    record = client.sampling_record()
    assert record["policy"] == "pinned_v1"
    assert record["temperature"] == 0.0
    assert record["wire"] == {}


def test_engine_records_what_the_execution_client_reports():
    backtester = engine_module.HourlyBacktester.__new__(engine_module.HourlyBacktester)
    client, _service = _client("deepseek/deepseek-v4-pro", [])
    backtester.execution_client = client
    backtester.model = "deepseek/deepseek-v4-pro"

    assert backtester._llm_sampling_metadata() == client.sampling_record()


def test_engine_without_an_execution_client_records_provider_default():
    """A CLI or in-process run on a plain SDK client pinned nothing."""
    backtester = engine_module.HourlyBacktester.__new__(engine_module.HourlyBacktester)
    backtester.execution_client = None
    backtester.model = "claude-haiku-4-5"

    assert backtester._llm_sampling_metadata() == {
        "temperature": None,
        "reasoning_effort": None,
        "policy": "provider_default",
        "model": "claude-haiku-4-5",
        "wire": {},
    }


def test_no_sampling_rides_the_child_argv():
    """The policy derives from the signed model id; argv carries none of it.

    Unsigned argv values could disagree with the signed handoff, and a CLI run
    without a handoff reached the legacy OpenRouter client, which pairs a
    temperature with an Anthropic thinking block that rejects it.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[5]
    script = (root / "dashboard/scripts/backtest_hourly_agent.py").read_text()
    router = (root / "dashboard/backend/api/routers/backtests.py").read_text()
    for source in (script, router):
        assert "--llm-temperature" not in source
        assert "--llm-reasoning-effort" not in source
