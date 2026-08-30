# OpenRouter Response Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover one empty or truncated model response per pipeline step with a single larger-output retry that preserves the provider's reasoning mode, while preserving fail-closed behavior and existing credit settlement.

**Architecture:** Keep response classification in the OpenAI-compatible execution adapter. Add a small request helper in the sequential pipeline runner; the first call uses the current client defaults, and an empty or structurally truncated response triggers one retry with `max_tokens` raised to the recovery ceiling. The retry does not override reasoning, so the provider's configured/default reasoning remains active. Parsing, decision normalization, portfolio behavior, and billing remain in their existing layers.

**Tech Stack:** Python 3.12, Pydantic execution models, Anthropic-shaped client protocol, pytest, existing OpenRouter/OpenAI-compatible adapter.

**Spec:** `docs/superpowers/specs/2026-08-29-openrouter-empty-response-recovery-design.md`

## Global Constraints

- Preserve the selected provider, model, prompt, billing mode, and run identity for both attempts; a recovery attempt may raise only its output-token ceiling.
- Retry at most once per pipeline step for `response_invalid` or a detected truncated JSON response.
- The recovery attempt preserves the provider's reasoning configuration and raises only the output-token ceiling.
- Credentials, provider errors, timeouts, billing/usage failures, and invalid business JSON are not retried.
- Each attempt uses the existing independent reservation/release lifecycle and a unique `call_index`.
- Do not log response bodies, reasoning text, API keys, or provider headers.
- Do not change frontend state, Render memory, worker concurrency, deployment settings, database schema, pricing, or credit policy.
- Use deterministic fake clients and responses; no network calls or real credentials.

### Task 1: Keep explicit OpenRouter reasoning controls provider-safe

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/execution/adapters/openai.py:108-118`
- Test: `dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py`

**Interfaces:**
- Consumes: `LLMExecutionRequest.reasoning_effort` and `ProviderRecord.adapter_type`.
- Produces: OpenRouter `chat.completions.create()` kwargs with a provider-safe `extra_body.reasoning` object; other OpenAI-compatible providers retain their current payload.

- [ ] **Step 1: Extend the adapter test request factory and write the failing contract test**

Update `_request()` to accept an optional `reasoning_effort` and add:

```python
def test_openrouter_reasoning_none_disables_reasoning(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _openai_response("qwen/qwen3.7-plus")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )
    monkeypatch.setattr(
        openai_module,
        "build_safe_http_client",
        lambda *_args, **_kwargs: _Closable(),
    )
    adapter = openai_module.OpenRouterAdapter(client_factory=lambda **_kwargs: client)

    adapter.complete(
        _request("openrouter", "qwen/qwen3.7-plus", reasoning_effort="none"),
        _credential("openrouter"),
        _provider("openrouter", "openrouter", "https://openrouter.ai/api/v1"),
    )

    assert captured["extra_body"] == {
        "reasoning": {"effort": "none", "enabled": False, "exclude": True}
    }
```

- [ ] **Step 2: Run the new test and verify the baseline fails**

Run:

```bash
python -m pytest dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py::test_openrouter_reasoning_none_disables_reasoning -q
```

Expected: FAIL because the current adapter forwards only `{"effort": "none"}`.

- [ ] **Step 3: Implement the minimal OpenRouter-only payload adjustment**

Replace the current reasoning block in `OpenAIExecutionAdapter.complete()` with:

```python
if request.reasoning_effort and provider.adapter_type in {
    "openrouter",
    "openai_compatible",
}:
    effort = request.reasoning_effort.strip().lower()
    reasoning = {"effort": request.reasoning_effort}
    if provider.adapter_type == "openrouter" and effort in {
        "none",
        "off",
        "false",
        "0",
        "disabled",
    }:
        reasoning.update({"enabled": False, "exclude": True})
    kwargs["extra_body"] = {"reasoning": reasoning}
```

This keeps normal configured effort values unchanged and makes an explicitly requested reasoning override provider-safe. Automatic recovery does not use this override; it preserves reasoning and changes only the output ceiling.

- [ ] **Step 4: Run the adapter route tests**

Run:

```bash
python -m pytest dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py -q
```

Expected: PASS, including the existing OpenAI, OpenRouter, Anthropic, and Gemini route tests.

- [ ] **Step 5: Commit the adapter contract**

```bash
git add dashboard/backend/infrastructure/llm/execution/adapters/openai.py dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py
git commit -m "fix: preserve OpenRouter reasoning controls"
```

### Task 2: Specify pipeline recovery behavior with deterministic tests

**Files:**
- Modify: `dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py`

**Interfaces:**
- Consumes: `run_pipeline_decision(client, pipeline, market_snapshot, model)`.
- Produces: tests that define the retry request shape, result accounting, retry limit, and non-retry boundaries before production code changes.

- [ ] **Step 1: Add fake response and sequenced client helpers**

Add these test-only helpers:

```python
from types import SimpleNamespace

from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
)
from dashboard.backend.infrastructure.llm.pipeline_runner import run_pipeline_decision


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
```

- [ ] **Step 2: Write the retry-success, retry-exhaustion, and non-retry tests**

Use one decision step and a valid empty order envelope:

```python
_PIPELINE = [{
    "id": "decision",
    "label": "Decision",
    "prompt": "Choose the action.",
    "outputFormat": '{"orders": []}',
}]


def test_run_pipeline_decision_retries_response_invalid_once():
    client = _PipelineClient([
        LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID),
        _PipelineResponse('{"orders": []}', input_tokens=11, output_tokens=4),
    ])

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
    client = _PipelineClient([
        LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID),
        LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID),
    ])

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
```

- [ ] **Step 3: Run the new pipeline tests and verify the baseline fails**

Run:

```bash
python -m pytest dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py -q
```

Expected: the existing normalization tests pass, while the new retry tests fail because `run_pipeline_decision()` currently performs only one unguarded model call.

### Task 3: Implement bounded response recovery with reasoning preserved

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/pipeline_runner.py:15-24,462-531`

**Interfaces:**
- Consumes: existing `client.messages.create()` Anthropic-shaped protocol and `ExecutionErrorCategory.RESPONSE_INVALID`.
- Produces: unchanged `run_pipeline_decision()` return type, with at most one retry request carrying the recovery `max_tokens` ceiling and no reasoning override.

- [ ] **Step 1: Add the execution-error imports and a private request helper**

Import `ExecutionErrorCategory` and `LLMExecutionError`, then add:

```python
def _create_pipeline_response(
    client,
    *,
    model: str,
    prompt: str,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
):
    request = {
        "model": model,
        "max_tokens": (
            DEFAULT_MAX_OUTPUT_TOKENS
            if max_tokens is None
            else max_tokens
        ),
        "system": PIPELINE_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    return client.messages.create(**request)
```

- [ ] **Step 2: Replace the direct decision-step call with one bounded recovery**

Inside `run_pipeline_decision()`, resolve `resolved_model = model or LLM_MODEL_NAME`, then use:

```python
    try:
        response = _create_pipeline_response(
            client,
            model=resolved_model,
            prompt=prompt,
        )
    except LLMExecutionError as first_error:
        if first_error.category != ExecutionErrorCategory.RESPONSE_INVALID:
            raise
        print(
            "   ⚠️  Empty model response; retrying with reasoning preserved "
            f"and max_tokens={RECOVERY_MAX_OUTPUT_TOKENS}"
        )
        response = _create_pipeline_response(
            client,
            model=resolved_model,
            prompt=prompt,
            max_tokens=RECOVERY_MAX_OUTPUT_TOKENS,
        )
```

Keep `llm_calls += 1`, usage extraction, parsing, and normalization immediately after this block. This means failed attempts do not become successful billable calls, while a successful retry follows the existing accounting path.

- [ ] **Step 3: Run the pipeline test module**

Run:

```bash
python -m pytest dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py -q
```

Expected: PASS for retry success, one-retry exhaustion, non-retryable errors, invalid JSON, and the existing pipeline normalization tests.

- [ ] **Step 4: Commit the pipeline recovery**

```bash
git add dashboard/backend/infrastructure/llm/pipeline_runner.py dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py
git commit -m "fix: recover OpenRouter empty pipeline responses"
```

### Task 4: Verify the complete regression surface

**Files:**
- Test: `dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py`
- Test: `dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py`
- Test: `dashboard/backend/tests/infrastructure/llm/test_execution_client.py`
- Test: `dashboard/backend/tests/llm/test_backtest_harness.py`
- Test: `dashboard/backend/tests/backtesting/test_portfolio_manager_move.py`

**Interfaces:**
- Consumes: the adapter payload contract and pipeline retry implementation from Tasks 1-3.
- Produces: evidence that OpenRouter routing, unified execution, legacy harness behavior, and strict portfolio handling remain compatible.

- [ ] **Step 1: Run the focused execution and pipeline tests**

```bash
python -m pytest \
  dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py \
  dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py \
  dashboard/backend/tests/infrastructure/llm/test_execution_client.py -q
```

Expected: PASS with no network access and no real credential use.

- [ ] **Step 2: Run related LLM and strict backtesting tests**

```bash
python -m pytest \
  dashboard/backend/tests/llm/test_backtest_harness.py \
  dashboard/backend/tests/llm/test_providers.py \
  dashboard/backend/tests/backtesting/test_portfolio_manager_move.py -q
```

Expected: PASS; existing non-pipeline rescue behavior and strict fallback boundaries remain unchanged.

- [ ] **Step 3: Run repository hygiene checks**

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors, no untracked secrets, and only the spec/plan plus the scoped adapter, pipeline, and test changes.

- [ ] **Step 4: Push the fix branch for deployment verification**

```bash
git push -u origin fix/openrouter-empty-response-recovery
```

Expected: the branch contains both implementation commits after the already-pushed design commit; Render must deploy the new head before a real Qwen run can validate recovery.
