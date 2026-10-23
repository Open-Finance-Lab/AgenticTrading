import pytest

from dashboard.backend.infrastructure.llm.execution.handoff import (
    ExecutionHandoffError,
    consume_execution_handoff,
    create_execution_handoff,
)


@pytest.mark.parametrize("token", ["é.abc", "abc.é"], ids=["payload", "signature"])
def test_a_non_ascii_token_is_a_handoff_error(token):
    # The worker catches only ExecutionHandoffError; a raw UnicodeEncodeError
    # from the payload half reached it as a traceback (#659).
    with pytest.raises(ExecutionHandoffError):
        consume_execution_handoff(token, now=1_001)


def test_handoff_round_trips_ordered_provider_candidates():
    payload = create_execution_handoff(
        user_id=1,
        run_id="run-1",
        billing_mode="platform_credits",
        provider_id="openrouter",
        provider_ids=("openrouter", "commonstack"),
        model_id="qwen/qwen3.7-plus",
        now=1_000,
    )

    handoff = consume_execution_handoff(payload, now=1_001)

    assert handoff.provider_id == "openrouter"
    assert handoff.provider_ids == ("openrouter", "commonstack")


def test_handoff_rejects_duplicate_or_misordered_candidates():
    with pytest.raises(ValueError):
        create_execution_handoff(
            user_id=1,
            run_id="run-1",
            billing_mode="platform_credits",
            provider_id="openrouter",
            provider_ids=("commonstack", "openrouter"),
            model_id="qwen/qwen3.7-plus",
        )

    with pytest.raises(ValueError):
        create_execution_handoff(
            user_id=1,
            run_id="run-1",
            billing_mode="platform_credits",
            provider_id="openrouter",
            provider_ids=("openrouter", "openrouter"),
            model_id="qwen/qwen3.7-plus",
        )
