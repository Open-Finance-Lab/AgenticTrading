from dashboard.backend.infrastructure.llm.execution.client import (
    AnthropicCompatibleExecutionClient,
)
from dashboard.backend.infrastructure.llm.execution.handoff import ExecutionHandoff
from dashboard.backend.infrastructure.llm.execution.models import (
    BillingEvidence,
    BillingMode,
    LLMExecutionResult,
    LLMRunEvidence,
    LLMUsage,
    PricingSnapshot,
)


class _FakeExecutionService:
    def __init__(self, results):
        self.results = list(results)
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        return self.results.pop(0)


def _handoff(billing_mode=BillingMode.PLATFORM_CREDITS):
    return ExecutionHandoff(
        user_id=7,
        run_id="run-evidence",
        billing_mode=billing_mode,
        provider_id="openrouter",
        model_id="openai/gpt-5.5",
        prompt_digest="a" * 64,
        nonce="n" * 16,
        issued_at=1,
        expires_at=2,
    )


def _result(
    *,
    billing_mode=BillingMode.PLATFORM_CREDITS,
    input_tokens=10,
    output_tokens=5,
    usage_available=True,
    provider_cost_usd=0.01,
    estimated_cost_usd=0.009,
    debited_micro=9_000,
    outstanding_micro=0,
    finish_reason=None,
    provider_id="openrouter",
    requested_provider_id="openrouter",
    credential_key_last_four="1234",
):
    snapshot = PricingSnapshot(
        provider_id=provider_id,
        model_id="openai/gpt-5.5",
        input_usd_per_million_tokens=5,
        output_usd_per_million_tokens=30,
        source_version="test-pricing-v1",
    )
    return LLMExecutionResult(
        text="BUY",
        provider_id=provider_id,
        requested_provider_id=requested_provider_id,
        model_id="openai/gpt-5.5",
        credential_id="credential-safe-id",
        credential_key_last_four=credential_key_last_four,
        usage=LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_available=usage_available,
        ),
        billing=BillingEvidence(
            billing_source=billing_mode,
            usage_authority=(
                "provider_reported_cost" if usage_available else "unavailable"
            ),
            provider_cost_usd=provider_cost_usd,
            estimated_cost_usd=estimated_cost_usd,
            pricing_snapshot=snapshot,
            provider_cost_credits_micro=debited_micro + outstanding_micro,
            debited_credits_micro=debited_micro,
            outstanding_credits_micro=outstanding_micro,
        ),
        finish_reason=finish_reason,
    )


def _create(client):
    return client.messages.create(
        model="openai/gpt-5.5",
        max_tokens=100,
        messages=[{"role": "user", "content": "Trade now"}],
    )


def test_client_preserves_compatibility_response_and_aggregates_evidence():
    service = _FakeExecutionService([
        _result(),
        _result(
            input_tokens=20,
            output_tokens=8,
            provider_cost_usd=0.02,
            estimated_cost_usd=0.019,
            debited_micro=15_000,
            outstanding_micro=5_000,
        ),
    ])
    client = AnthropicCompatibleExecutionClient(
        execution_service=service,
        handoff=_handoff(),
    )

    response = _create(client)
    _create(client)
    summary = client.execution_summary()

    assert response.content[0].text == "BUY"
    assert response.model == "openai/gpt-5.5"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.stop_reason is None
    assert [request.call_index for request in service.requests] == [0, 1]
    assert summary.billing_mode is BillingMode.PLATFORM_CREDITS
    assert summary.provider_id == "openrouter"
    assert summary.requested_provider_id == "openrouter"
    assert summary.provider_ids == ("openrouter",)
    assert summary.provider_mixed is False
    assert summary.model_id == "openai/gpt-5.5"
    assert summary.credential_id == "credential-safe-id"
    assert summary.credential_key_last_four == "1234"
    assert summary.call_count == 2
    assert summary.input_tokens == 30
    assert summary.output_tokens == 13
    assert summary.total_tokens == 43
    assert summary.usage_available is True
    assert summary.provider_cost_usd == 0.03
    assert summary.estimated_cost_usd == 0.028
    assert summary.debited_credits_micro == 24_000
    assert summary.outstanding_credits_micro == 5_000
    assert summary.outcome == "settled_overage"


def test_execution_summary_records_mixed_actual_providers():
    service = _FakeExecutionService(
        [
            _result(
                provider_id="openrouter",
                requested_provider_id="openrouter",
                credential_key_last_four="1111",
            ),
            _result(
                provider_id="commonstack",
                requested_provider_id="openrouter",
                credential_key_last_four="2222",
            ),
        ]
    )
    client = AnthropicCompatibleExecutionClient(
        execution_service=service,
        handoff=_handoff(),
    )

    _create(client)
    _create(client)
    summary = client.execution_summary()

    assert summary.requested_provider_id == "openrouter"
    assert summary.provider_id == "mixed"
    assert summary.provider_ids == ("openrouter", "commonstack")
    assert summary.provider_mixed is True
    assert summary.credential_id is None
    assert summary.credential_key_last_four is None
    assert summary.pricing_snapshot is None


def test_legacy_run_evidence_infers_uniform_provider_attribution():
    legacy = LLMRunEvidence.model_validate(
        {
            "billing_mode": "platform_credits",
            "provider_id": "openrouter",
            "model_id": "openai/gpt-5.5",
            "call_count": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "usage_available": True,
            "debited_credits_micro": 1,
            "outstanding_credits_micro": 0,
            "outcome": "settled",
        }
    )

    assert legacy.requested_provider_id == "openrouter"
    assert legacy.provider_ids == ("openrouter",)
    assert legacy.provider_mixed is False


def test_byok_summary_does_not_report_missing_usage_as_zero():
    service = _FakeExecutionService([
        _result(
            billing_mode=BillingMode.BYOK,
            input_tokens=0,
            output_tokens=0,
            usage_available=False,
            provider_cost_usd=None,
            estimated_cost_usd=None,
            debited_micro=0,
        )
    ])
    client = AnthropicCompatibleExecutionClient(
        execution_service=service,
        handoff=_handoff(BillingMode.BYOK),
    )

    _create(client)
    summary = client.execution_summary()

    assert summary.usage_available is False
    assert summary.provider_cost_usd is None
    assert summary.estimated_cost_usd is None
    assert summary.outcome == "unavailable"



def test_client_exposes_provider_stop_reason_like_the_sdk():
    service = _FakeExecutionService([_result(finish_reason="max_tokens")])
    client = AnthropicCompatibleExecutionClient(
        execution_service=service,
        handoff=_handoff(),
    )

    response = _create(client)

    assert response.stop_reason == "max_tokens"
