# Backtest Pinned Sampling (Track B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every model call a dashboard backtest makes carries a pinned sampling policy chosen per catalog model, the run records what it asked for, the results panel shows it, and a rerun of one configuration is measured against the first run.

**Architecture:** The policy lives on the execution catalog (`CatalogModel.sampling`) and rides the resolved route the endpoint already preflights. It reaches the child as two argv flags, then engine → portfolio manager → pipeline runner → the worker client, which already validates `temperature` and `reasoning_effort`. The request builder adds each value only when set, so the CLI's plain Anthropic SDK client keeps working. The engine writes `llm_sampling` into `agent_runs.metadata`; the results panel renders it as a Sampling row. A dev script diffs two runs' decisions.

**Tech Stack:** Python 3 / FastAPI backend (`dashboard.backend` package), vanilla JS frontend lifted into `node -e` by `dashboard/backend/tests/_frontend_source.py`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-20-backtest-speed-and-trust-design.md` (Track B sections, including the **⚠ changed** note: the policy is a per-model table, not a provider flag).

## Global Constraints

- Run everything from the repo root. Tests: `pytest dashboard/backend/tests/<file> -v`. Node must be on `PATH` for the node-harness tests (they skip without it).
- The policy table is exactly: Claude Haiku 4.5 and Claude Sonnet 4.6 → temperature 0; GPT-5.5 → reasoning effort `low`, **no temperature**; Gemini 3.1 Pro Preview → temperature 0; DeepSeek V4 Pro and Qwen3.7 Plus → temperature 0 **and** reasoning effort `low`.
- A sampling value is sent **only when set**. An unset value must leave the request byte-identical to today's.
- Copy never says "deterministic". The row says *Pinned · …*, *Provider default*, or *Not recorded*.
- Every change to `dashboard/frontend/app.js` bumps the `app.js?v=N` pin in **five** files (`dashboard/frontend/app.html` plus the four tests from `grep -rln "app.js?v=" dashboard/backend/tests/*.py`). `styles.css` is not touched.
- Commit messages: `type: summary`, ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Branch: `feat/backtest-pinned-sampling`, cut from `origin/main` **after** Track A's PR (`docs/superpowers/plans/2026-09-20-backtest-visible-start.md`) has merged.
- Line numbers below were read on the pre-merge tree; each edit also quotes the code it replaces, so grep the quote if a number has drifted.

---

### Task 1: The policy lives on the catalog

**Files:**
- Modify: `dashboard/backend/domain/model_providers/execution_catalog.py` (dataclasses at `:14-25`, entries at `:27-51`, `list_execution_model_routes` at `:80-97`, `__all__` at `:114`)
- Test: `dashboard/backend/tests/test_execution_catalog_sampling.py` (new)

**Interfaces:**
- Produces: `SamplingPolicy(temperature: float | None, reasoning_effort: str | None)` (frozen dataclass); constants `PINNED_TEMPERATURE`, `PINNED_REASONING_LOW`, `PINNED_BOTH`; `CatalogModel.sampling: SamplingPolicy` (default `PINNED_TEMPERATURE`); `ExecutionModelRoute.sampling: SamplingPolicy` (default `PINNED_TEMPERATURE`, so existing `ExecutionModelRoute(...)` constructions in tests keep working).

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_execution_catalog_sampling.py`:

```python
"""Every dashboard-launchable model carries a pinned sampling policy.

The pipeline request builder sent model, max_tokens, system and messages and
nothing else, so two runs of one configuration were two draws from the
provider's default sampler, 49 bars deep, each prompt embedding the previous
bar's draw. The policy is per *model*, not per provider: the provider-level
`reasoning` capability cannot say whether this model rejects temperature
(OpenAI reasoning models do) or ignores it while thinking (DeepSeek does).
"""
import pytest

from dashboard.backend.domain.model_providers.execution_catalog import (
    ATL_EXECUTION_MODELS,
    PINNED_BOTH,
    PINNED_REASONING_LOW,
    PINNED_TEMPERATURE,
    SamplingPolicy,
    list_execution_model_routes,
)
from dashboard.backend.domain.model_providers.models import ProviderRecord

_EXPECTED = {
    "anthropic/claude-haiku-4-5": PINNED_TEMPERATURE,
    "anthropic/claude-sonnet-4-6": PINNED_TEMPERATURE,
    "openai/gpt-5.5": PINNED_REASONING_LOW,
    "google/gemini-3.1-pro-preview": PINNED_TEMPERATURE,
    "deepseek/deepseek-v4-pro": PINNED_BOTH,
    "qwen/qwen3.7-plus": PINNED_BOTH,
}


def test_the_table_covers_the_catalog_exactly():
    assert {m.catalog_id for m in ATL_EXECUTION_MODELS} == set(_EXPECTED)


@pytest.mark.parametrize("catalog_id", sorted(_EXPECTED))
def test_each_model_pins_the_documented_policy(catalog_id):
    model = next(m for m in ATL_EXECUTION_MODELS if m.catalog_id == catalog_id)
    assert model.sampling == _EXPECTED[catalog_id]
    assert model.sampling.temperature is not None or model.sampling.reasoning_effort


def test_the_three_policies_are_what_they_say():
    assert PINNED_TEMPERATURE == SamplingPolicy(temperature=0.0, reasoning_effort=None)
    assert PINNED_REASONING_LOW == SamplingPolicy(temperature=None, reasoning_effort="low")
    assert PINNED_BOTH == SamplingPolicy(temperature=0.0, reasoning_effort="low")


def test_routes_carry_their_models_policy():
    provider = ProviderRecord(
        provider_id="openrouter",
        display_name="OpenRouter",
        adapter_type="openrouter",
        approved_base_url="https://openrouter.ai/api/v1",
    )
    by_id = {r.catalog_id: r for r in list_execution_model_routes(provider)}
    assert by_id["openai/gpt-5.5"].sampling == PINNED_REASONING_LOW
    assert by_id["deepseek/deepseek-v4-pro"].sampling == PINNED_BOTH
    assert by_id["anthropic/claude-sonnet-4-6"].sampling == PINNED_TEMPERATURE
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_execution_catalog_sampling.py -v`
Expected: FAIL with `ImportError: cannot import name 'PINNED_BOTH'`.

- [ ] **Step 3: Add the policy to the catalog**

In `dashboard/backend/domain/model_providers/execution_catalog.py`, replace the two dataclasses (`:14-25`)

```python
@dataclass(frozen=True)
class CatalogModel:
    catalog_id: str
    label: str
    vendor: str


@dataclass(frozen=True)
class ExecutionModelRoute:
    catalog_id: str
    label: str
    provider_model_id: str
```

with:

```python
@dataclass(frozen=True)
class SamplingPolicy:
    """What every backtest call asks the model to sample with.

    Per model, not per provider: the provider-level ``reasoning`` capability
    cannot say whether *this* model rejects a temperature (OpenAI reasoning
    models return 400) or ignores it while thinking (DeepSeek). A value of
    ``None`` is *not sent*, which keeps the request byte-identical to what
    every run before this sent for that field.
    """

    temperature: float | None
    reasoning_effort: str | None


PINNED_TEMPERATURE = SamplingPolicy(temperature=0.0, reasoning_effort=None)
PINNED_REASONING_LOW = SamplingPolicy(temperature=None, reasoning_effort="low")
PINNED_BOTH = SamplingPolicy(temperature=0.0, reasoning_effort="low")


@dataclass(frozen=True)
class CatalogModel:
    catalog_id: str
    label: str
    vendor: str
    sampling: SamplingPolicy = PINNED_TEMPERATURE


@dataclass(frozen=True)
class ExecutionModelRoute:
    catalog_id: str
    label: str
    provider_model_id: str
    sampling: SamplingPolicy = PINNED_TEMPERATURE
```

Replace the catalog entries (`:27-51`) with:

```python
ATL_EXECUTION_MODELS = (
    CatalogModel(
        "anthropic/claude-haiku-4-5",
        "Claude Haiku 4.5",
        "anthropic",
    ),
    CatalogModel(
        "anthropic/claude-sonnet-4-6",
        "Claude Sonnet 4.6",
        "anthropic",
    ),
    # OpenAI reasoning models reject a non-default temperature outright.
    CatalogModel("openai/gpt-5.5", "GPT-5.5", "openai", PINNED_REASONING_LOW),
    CatalogModel(
        "google/gemini-3.1-pro-preview",
        "Gemini 3.1 Pro Preview",
        "google",
    ),
    # Thinking models behind OpenRouter: temperature is accepted and ignored
    # while thinking; the effort bound is what keeps the reply inside the
    # 2000-token ceiling instead of spending it all on the thinking block.
    CatalogModel(
        "deepseek/deepseek-v4-pro",
        "DeepSeek V4 Pro",
        "deepseek",
        PINNED_BOTH,
    ),
    CatalogModel("qwen/qwen3.7-plus", "Qwen3.7 Plus", "qwen", PINNED_BOTH),
)
```

In `list_execution_model_routes`, add `sampling=model.sampling,` to the `ExecutionModelRoute(` construction after `provider_model_id=provider_model_id,`.

Extend `__all__` with `"PINNED_BOTH"`, `"PINNED_REASONING_LOW"`, `"PINNED_TEMPERATURE"`, `"SamplingPolicy"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_execution_catalog_sampling.py dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py dashboard/backend/tests/test_market_data_features.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/model_providers/execution_catalog.py dashboard/backend/tests/test_execution_catalog_sampling.py
git commit -m "feat: pin a sampling policy per catalog model

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The pipeline runner sends the policy on every attempt

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/pipeline_runner.py` (`_create_pipeline_response` at `:505-523`, `_retry_with_recovery_budget` at `:526-540`, `run_pipeline_decision` signature at `:693-699` and its three call sites at `:746`, `:758`, `:780`)
- Test: `dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py` (append)

**Interfaces:**
- Produces: `run_pipeline_decision(client, *, pipeline, market_snapshot, model=None, temperature: Optional[float] = None, reasoning_effort: Optional[str] = None)`; `_create_pipeline_response(client, *, model, prompt, max_tokens=None, temperature=None, reasoning_effort=None)`; `_retry_with_recovery_budget(client, *, model, prompt, temperature=None, reasoning_effort=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py`:

```python
def test_run_pipeline_decision_sends_pinned_sampling_on_both_attempts():
    """The recovery retry is the same request at a higher ceiling; a retry
    that dropped the sampling would be a different request."""
    client = _PipelineClient(
        [
            LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID),
            _PipelineResponse('{"orders": []}'),
        ]
    )

    decision, _usage, _calls, _steps = run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
        model="deepseek/deepseek-v4-pro",
        temperature=0.0,
        reasoning_effort="low",
    )

    assert decision == {"actions": []}
    assert len(client.messages.calls) == 2
    for call in client.messages.calls:
        assert call["temperature"] == 0.0
        assert call["reasoning_effort"] == "low"


def test_truncation_retry_keeps_the_sampling():
    client = _PipelineClient(
        [
            _PipelineResponse(_truncated_json(), input_tokens=13, output_tokens=2000),
            _PipelineResponse('{"orders": []}', input_tokens=11, output_tokens=4),
        ]
    )

    run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
        model="google/gemini-3.1-pro-preview",
        temperature=0.0,
    )

    assert len(client.messages.calls) == 2
    for call in client.messages.calls:
        assert call["temperature"] == 0.0
        assert "reasoning_effort" not in call


def test_run_pipeline_decision_sends_only_the_set_half_of_a_policy():
    client = _PipelineClient([_PipelineResponse('{"orders": []}')])

    run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
        model="openai/gpt-5.5",
        reasoning_effort="low",
    )

    (call,) = client.messages.calls
    assert call["reasoning_effort"] == "low"
    assert "temperature" not in call


def test_run_pipeline_decision_default_request_shape_is_unchanged():
    """Unset means absent: the CLI's real Anthropic SDK client rejects an
    unknown reasoning_effort kwarg, and a None temperature is not a request."""
    client = _PipelineClient([_PipelineResponse('{"orders": []}')])

    run_pipeline_decision(
        client,
        pipeline=_PIPELINE,
        market_snapshot={"top_signals": {}},
        model="qwen/qwen3.7-plus",
    )

    (call,) = client.messages.calls
    assert set(call) == {"model", "max_tokens", "system", "messages"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py -k "sampling or truncation_retry_keeps or default_request_shape" -v`
Expected: FAIL with `TypeError: run_pipeline_decision() got an unexpected keyword argument 'temperature'` (the last test passes already; that is fine).

- [ ] **Step 3: Thread the values through the request builder**

Replace `_create_pipeline_response` (`:505-523`) with:

```python
def _create_pipeline_response(
    client,
    *,
    model: str,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
):
    """Create one pipeline request, optionally with a recovery output budget.

    Sampling values are added only when set. ``None`` keeps the request
    byte-identical to what every run before the pinned policy sent, and the
    CLI's plain Anthropic SDK client rejects an unknown ``reasoning_effort``
    kwarg outright.
    """
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
    if temperature is not None:
        request["temperature"] = temperature
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    return client.messages.create(**request)
```

Replace `_retry_with_recovery_budget` (`:526-540`) with:

```python
def _retry_with_recovery_budget(
    client,
    *,
    model: str,
    prompt: str,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
):
    """Second attempt for a step whose first attempt was unusable.

    The same request, reasoning preserved, with the output ceiling raised to
    ``RECOVERY_MAX_OUTPUT_TOKENS`` so a reasoning-heavy model has room for
    both its thinking and the final JSON. Both recovery paths send exactly
    this, which is why a step never gets a third attempt: a reply that is
    still unusable after it has nothing different left to ask for. The
    sampling travels with it for the same reason -- it is the same request.
    """
    return _create_pipeline_response(
        client,
        model=model,
        prompt=prompt,
        max_tokens=RECOVERY_MAX_OUTPUT_TOKENS,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
```

In `run_pipeline_decision`, extend the signature (`:693-699`):

```python
def run_pipeline_decision(
    client,
    *,
    pipeline: List[Dict[str, Any]],
    market_snapshot: Dict[str, Any],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Tuple[int, int], int, List[Dict[str, Any]]]:
```

and add `temperature=temperature, reasoning_effort=reasoning_effort,` to each of the three calls inside it:

```python
            response = _create_pipeline_response(
                client,
                model=request_model,
                prompt=prompt,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
```

```python
            response = _retry_with_recovery_budget(
                client,
                model=request_model,
                prompt=prompt,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
```

```python
                retry_response = _retry_with_recovery_budget(
                    client,
                    model=request_model,
                    prompt=prompt,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py -v`
Expected: all pass, including the existing `"reasoning_effort" not in client.messages.calls[...]` assertions.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/infrastructure/llm/pipeline_runner.py dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py
git commit -m "feat: carry pinned sampling through every pipeline attempt

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Harness, portfolio manager and engine thread it; metadata records it

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/backtest_harness.py` (`request_trading_decision` at `:198-222`)
- Modify: `dashboard/backend/domain/backtesting/portfolio_manager.py` (`make_trading_decision_with_llm` signature at `:244-256`; the `run_pipeline_decision(` call at `:462-468`; the two `_request_trading_decision(` calls at `:525-539`)
- Modify: `dashboard/backend/domain/backtesting/engine.py` (constructor kwargs at `:180-205`; `self.model = ...` at `:236`; the `make_trading_decision_with_llm(` call at `:1483-1491`; `_agent_run_metadata` at `:1008-1052`)
- Test: `dashboard/backend/tests/llm/test_backtest_harness.py` (append), `dashboard/backend/tests/test_agent_runs_metadata.py` (edit `test_engine_llm_run_metadata_snapshot` at `:131-163`, append), `dashboard/backend/tests/test_backtest_sampling_wiring.py` (new)

**Interfaces:**
- Consumes: `run_pipeline_decision(..., temperature=, reasoning_effort=)` from Task 2.
- Produces: `request_trading_decision(..., reasoning_effort: Optional[str] = None)`; `PortfolioManager.make_trading_decision_with_llm(..., temperature=None, reasoning_effort=None, ...)`; `HourlyBacktester(..., llm_temperature: Optional[float] = None, llm_reasoning_effort: Optional[str] = None)` with attributes of the same names; `HourlyBacktester._llm_sampling_metadata() -> Dict`; `agent_runs.metadata["llm_sampling"] = {"temperature", "reasoning_effort", "policy": "pinned_v1" | "provider_default", "catalog_id"}` on every LLM run.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/llm/test_backtest_harness.py`:

```python
def test_request_sends_reasoning_effort_only_when_set():
    client = _FakeClient(_FakeResponse('{"actions": []}'))
    harness.request_trading_decision(client, prompt="HELLO", reasoning_effort="low")
    assert client.captured["reasoning_effort"] == "low"

    client = _FakeClient(_FakeResponse('{"actions": []}'))
    harness.request_trading_decision(client, prompt="HELLO")
    assert "reasoning_effort" not in client.captured
```

In `dashboard/backend/tests/test_agent_runs_metadata.py`, inside `test_engine_llm_run_metadata_snapshot`, change the first expected dict (the `use_llm = True` case) to:

```python
    assert backtester._agent_run_metadata() == {
        "data_source": "alpaca",
        "symbols": ["AAPL", "MSFT"],
        "native_currency": "USD",
        "reporting_currency": "USD",
        "lot_size": 1,
        "llm_max_output_tokens": 777,
        "llm_sampling": {
            "temperature": None,
            "reasoning_effort": None,
            "policy": "provider_default",
            "catalog_id": None,
        },
    }
```

(the `use_llm = False` dict is unchanged), and append:

```python
def test_engine_records_the_pinned_sampling(monkeypatch):
    """A row that does not say what it sent cannot be reproduced. `policy`
    names the runs that pinned nothing, so an *absent* key on an older row
    reads as 'not recorded', never as 'default'."""
    import dashboard.backend.domain.backtesting.engine as engine_mod

    backtester = engine_mod.HourlyBacktester.__new__(engine_mod.HourlyBacktester)
    backtester.prompt_adaptations = []
    backtester.initial_pipeline = None
    backtester.pipeline = None
    backtester.symbols = ["AAPL"]
    backtester.data_source = "alpaca"
    backtester.use_llm = True
    backtester.model = "deepseek/deepseek-v4-pro"
    backtester.llm_temperature = 0.0
    backtester.llm_reasoning_effort = "low"
    monkeypatch.setattr(engine_mod.llm_harness, "DEFAULT_MAX_OUTPUT_TOKENS", 2000)

    assert backtester._agent_run_metadata()["llm_sampling"] == {
        "temperature": 0.0,
        "reasoning_effort": "low",
        "policy": "pinned_v1",
        "catalog_id": "deepseek/deepseek-v4-pro",
    }
```

Create `dashboard/backend/tests/test_backtest_sampling_wiring.py`:

```python
"""The sampling policy reaches the model call from the engine, on both branches.

Pinned by source shape, the way test_agent_runs_metadata pins the metadata
call site: the decision method needs a full market snapshot to reach either
branch, and a dropped keyword between two files is exactly the defect this
work fixes (the pipeline branch never forwarded temperature although the
single-prompt branch did).
"""
import inspect

from dashboard.backend.domain.backtesting import engine, portfolio_manager


def test_engine_forwards_both_values_to_the_manager():
    src = inspect.getsource(engine.HourlyBacktester.run_agent_backtest)
    assert "temperature=self.llm_temperature," in src
    assert "reasoning_effort=self.llm_reasoning_effort," in src


def test_manager_forwards_both_values_to_the_pipeline_runner():
    src = inspect.getsource(portfolio_manager.PortfolioManager.make_trading_decision_with_llm)
    pipeline_call = src[src.index("run_pipeline_decision("):]
    pipeline_call = pipeline_call[: pipeline_call.index(")")]
    assert "temperature=temperature," in pipeline_call
    assert "reasoning_effort=reasoning_effort," in pipeline_call


def test_manager_forwards_reasoning_effort_on_the_single_prompt_branch():
    src = inspect.getsource(portfolio_manager.PortfolioManager.make_trading_decision_with_llm)
    assert src.count("_request_trading_decision(") == 2
    assert src.count("reasoning_effort=reasoning_effort,") == 3


def test_engine_accepts_the_two_kwargs():
    params = inspect.signature(engine.HourlyBacktester.__init__).parameters
    assert params["llm_temperature"].default is None
    assert params["llm_reasoning_effort"].default is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/llm/test_backtest_harness.py::test_request_sends_reasoning_effort_only_when_set dashboard/backend/tests/test_agent_runs_metadata.py dashboard/backend/tests/test_backtest_sampling_wiring.py -v`
Expected: FAIL — `TypeError: request_trading_decision() got an unexpected keyword argument 'reasoning_effort'`, the snapshot dict mismatch, and `KeyError: 'llm_temperature'`.

- [ ] **Step 3: Harness**

In `dashboard/backend/infrastructure/llm/backtest_harness.py`, add `reasoning_effort: Optional[str] = None,` to `request_trading_decision`'s keyword parameters after `temperature: Optional[float] = None,`, and after

```python
    if temperature is not None:
        request_kwargs["temperature"] = temperature
```

add:

```python
    if reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = reasoning_effort
```

- [ ] **Step 4: Portfolio manager**

In `make_trading_decision_with_llm` (`:244-256`), add `reasoning_effort: Optional[str] = None,` directly after `temperature: Optional[float] = None,`.

Change the pipeline call (`:462-468`) to:

```python
                    run_pipeline_decision(
                        llm_client,
                        pipeline=pipeline,
                        market_snapshot=market_snapshot,
                        model=model,
                        temperature=temperature,
                        reasoning_effort=reasoning_effort,
                    )
```

In both `_request_trading_decision(` calls on the single-prompt branch (`:525` and `:534`), add `reasoning_effort=reasoning_effort,` directly after the `temperature=temperature,` line.

- [ ] **Step 5: Engine**

In `HourlyBacktester.__init__` (`:180`), append after `launched_at: Optional[float] = None,` (added by Track A):

```python
        llm_temperature: Optional[float] = None,
        llm_reasoning_effort: Optional[str] = None,
```

Directly after `self.model = model or default_model_name()` (`:236`), add:

```python
        # The pinned sampling policy for this run (execution_catalog.SamplingPolicy),
        # handed over by the parent as argv. None means "not sent".
        self.llm_temperature = llm_temperature
        self.llm_reasoning_effort = llm_reasoning_effort
```

Change the decision call (`:1483-1491`) to:

```python
                    decision = manager.make_trading_decision_with_llm(
                        state,
                        self.llm_client,
                        mode=self.mode,
                        model=self.model,
                        strategy_prompt=self.strategy_prompt,
                        pipeline=self.pipeline,
                        temperature=self.llm_temperature,
                        reasoning_effort=self.llm_reasoning_effort,
                        market_context=self._llm_market_context(),
                        strict_llm=self.strict_llm,
                    )
```

In `_agent_run_metadata`, change

```python
        if self.use_llm:
            meta["llm_max_output_tokens"] = llm_harness.DEFAULT_MAX_OUTPUT_TOKENS
```

to:

```python
        if self.use_llm:
            meta["llm_max_output_tokens"] = llm_harness.DEFAULT_MAX_OUTPUT_TOKENS
            meta["llm_sampling"] = self._llm_sampling_metadata()
```

and add the helper directly above `def _agent_run_metadata(`:

```python
    def _llm_sampling_metadata(self) -> Dict:
        """What this run asked every model call to sample with.

        Recorded beside ``llm_max_output_tokens`` for the same reason: the
        request shape is what makes two runs of one configuration comparable,
        and a row that does not say what it sent cannot be reproduced.
        ``provider_default`` names the runs that pinned nothing, so an absent
        key on an older row reads as "not recorded", never as "default".
        getattr throughout: tests and legacy tools build the engine with
        __new__.
        """
        temperature = getattr(self, "llm_temperature", None)
        reasoning_effort = getattr(self, "llm_reasoning_effort", None)
        pinned = temperature is not None or bool(reasoning_effort)
        return {
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "policy": "pinned_v1" if pinned else "provider_default",
            "catalog_id": getattr(self, "model", None),
        }
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/llm/test_backtest_harness.py dashboard/backend/tests/test_agent_runs_metadata.py dashboard/backend/tests/test_backtest_sampling_wiring.py dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py dashboard/backend/tests/test_backtest_run_provenance.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/infrastructure/llm/backtest_harness.py dashboard/backend/domain/backtesting/portfolio_manager.py dashboard/backend/domain/backtesting/engine.py dashboard/backend/tests/llm/test_backtest_harness.py dashboard/backend/tests/test_agent_runs_metadata.py dashboard/backend/tests/test_backtest_sampling_wiring.py
git commit -m "feat: thread the sampling policy to every backtest model call

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The parent decides the policy and hands it to the child

**Files:**
- Modify: `dashboard/scripts/backtest_hourly_agent.py` (argparse after `--launched-at`; the `HourlyBacktester(` call at `:443`)
- Modify: `dashboard/backend/api/routers/backtests.py` — `run_backtest_background` signature (`:1381-1400`) and argv after the `--model` line (`:1528-1529`); `run_backtest_endpoint`: after `execution_handoff_payload: Optional[str] = None` (`:3046`), after `execution_handoff_payload = create_execution_handoff(...)` (`:3126-3143`), and the thread `kwargs={...}` (`:3179-3199`)
- Test: `dashboard/backend/tests/test_backtest_sampling_argv.py` (new)

**Interfaces:**
- Consumes: `ExecutionModelRoute.sampling` from Task 1; `HourlyBacktester(llm_temperature=, llm_reasoning_effort=)` from Task 3.
- Produces: child argv `--llm-temperature <float repr>` and `--llm-reasoning-effort <lowercase str>`, each present only when its value is set; `run_backtest_background(..., llm_temperature: Optional[float] = None, llm_reasoning_effort: Optional[str] = None)`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_backtest_sampling_argv.py`:

```python
"""The parent resolves the sampling policy and the child receives it as argv.

The values are not secrets, so they ride beside --model rather than inside
the signed handoff envelope. Each flag is present only when its half of the
policy is set: an absent flag is how "not sent" reaches the request builder.
"""
import subprocess
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
import dashboard.backend.api.routers.backtests as backtests
from dashboard.backend.domain.model_providers.execution_catalog import (
    PINNED_BOTH,
    PINNED_REASONING_LOW,
    ExecutionModelRoute,
)
from dashboard.backend.infrastructure.market_data.profiles import A_SHARE_DEMO_6
from dashboard.backend.infrastructure.market_data.provider import IFIND_ASHARE
from dashboard.backend.tests._fake_child import FakeChild

REAL_RUN_BACKTEST_BACKGROUND = backtests.run_backtest_background


class _Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _Preflight:
    def __init__(self, sampling):
        self.sampling = sampling

    def preflight_execution_model(self, provider_id, catalog_model_id):
        return ExecutionModelRoute(
            catalog_id=catalog_model_id,
            label=catalog_model_id,
            provider_model_id=catalog_model_id,
            sampling=self.sampling,
        )

    def preflight_user_default_credential(self, user_id, provider_id):
        return None


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    backtests._backtest_rate_limiter.reset()
    backtests.backtest_status.update({
        "running": False, "error": None, "runs_count": 0,
        "started_at": None, "progress_file": None, "live_run_id": None,
    })
    yield
    backtests._backtest_rate_limiter.reset()


def _capture_command(monkeypatch, **kwargs):
    captured = {}

    def fake_popen(command, **_kwargs):
        captured["command"] = command
        return FakeChild()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(backtests.db, "get_runs_by_mode", lambda mode: [])
    REAL_RUN_BACKTEST_BACKGROUND(
        "2026-04-01", "2026-04-08", "session-id",
        decision_source="llm", model="deepseek/deepseek-v4-pro",
        strategy_prompt="momentum", **kwargs,
    )
    return captured["command"]


def test_both_flags_ride_argv_when_both_are_set(monkeypatch):
    command = _capture_command(monkeypatch, llm_temperature=0.0, llm_reasoning_effort="low")
    assert command[command.index("--llm-temperature") + 1] == "0.0"
    assert command[command.index("--llm-reasoning-effort") + 1] == "low"


def test_an_unset_half_is_absent_from_argv(monkeypatch):
    command = _capture_command(monkeypatch, llm_reasoning_effort="LOW")
    assert "--llm-temperature" not in command
    assert command[command.index("--llm-reasoning-effort") + 1] == "low"


def test_no_policy_means_no_flags(monkeypatch):
    command = _capture_command(monkeypatch)
    assert "--llm-temperature" not in command
    assert "--llm-reasoning-effort" not in command


@pytest.mark.parametrize(
    ("sampling", "expected"),
    [
        (PINNED_REASONING_LOW, {"llm_temperature": None, "llm_reasoning_effort": "low"}),
        (PINNED_BOTH, {"llm_temperature": 0.0, "llm_reasoning_effort": "low"}),
    ],
)
def test_endpoint_hands_the_routes_policy_to_the_launcher(monkeypatch, sampling, expected):
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    monkeypatch.setattr(backtests, "get_model_provider_service", lambda: _Preflight(sampling))
    monkeypatch.setattr(
        "dashboard.backend.api.dependencies._optional_user",
        lambda *_args, **_kwargs: {"id": 7},
    )
    spy = _Spy()
    monkeypatch.setattr(backtests, "run_backtest_background", spy)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-15",
            "data_source": IFIND_ASHARE,
            "universe": A_SHARE_DEMO_6,
            "timeframe": "60m",
            "decision_source": "llm",
            "billing_mode": "byok",
            "provider_id": "openrouter",
            "model": "openai/gpt-5.5",
            "strategy_prompt": "A-share momentum",
        },
        headers={"X-Session-Id": str(uuid.uuid4())},
    )

    assert response.status_code == 200, response.text
    (_args, kwargs), = spy.calls
    assert {k: kwargs[k] for k in expected} == expected


def test_script_accepts_the_two_flags(tmp_path):
    import os

    result = subprocess.run(
        [sys.executable, "dashboard/scripts/backtest_hourly_agent.py", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "DATABASE_PATH": str(tmp_path / "backtest.db")},
    )
    assert result.returncode == 0, result.stderr
    assert "--llm-temperature" in result.stdout
    assert "--llm-reasoning-effort" in result.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_backtest_sampling_argv.py -v`
Expected: FAIL — `TypeError: run_backtest_background() got an unexpected keyword argument 'llm_temperature'`, the endpoint test with `KeyError: 'llm_temperature'`, and the `--help` assertion.

- [ ] **Step 3: Script arguments**

In `dashboard/scripts/backtest_hourly_agent.py`, directly after the `--launched-at` argument, add:

```python
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature sent with every model call. Dashboard runs "
            "pin 0 for models that accept it (execution_catalog.SamplingPolicy)."
        ),
    )
    parser.add_argument(
        "--llm-reasoning-effort",
        default=None,
        help=(
            "Reasoning effort sent with every model call. Worker runs only: the "
            "plain Anthropic SDK client used without --execution-handoff-stdin "
            "rejects the kwarg."
        ),
    )
```

In the `backtester = HourlyBacktester(` call, add after `launched_at=args.launched_at,`:

```python
        llm_temperature=args.llm_temperature,
        llm_reasoning_effort=args.llm_reasoning_effort,
```

- [ ] **Step 4: Launcher argv**

In `run_backtest_background` (`:1381`), add two keyword parameters after `universe_selection: Optional[Dict[str, Any]] = None,`:

```python
    llm_temperature: Optional[float] = None,
    llm_reasoning_effort: Optional[str] = None,
```

Directly after

```python
        if uses_llm and model and model.strip():
            cmd += ["--model", model.strip()]
```

add:

```python
        # Each half of the sampling policy rides only when set: an absent flag
        # is how "not sent" reaches the request builder, and neither value is
        # a secret, so argv rather than the signed handoff.
        if uses_llm and llm_temperature is not None:
            cmd += ["--llm-temperature", repr(float(llm_temperature))]
        if uses_llm and llm_reasoning_effort and str(llm_reasoning_effort).strip():
            cmd += ["--llm-reasoning-effort", str(llm_reasoning_effort).strip().lower()]
```

- [ ] **Step 5: Endpoint**

In `run_backtest_endpoint`, directly after `execution_handoff_payload: Optional[str] = None` (`:3046`), add:

```python
    llm_sampling: Optional[SamplingPolicy] = None
```

and import `SamplingPolicy` beside the module's existing `execution_catalog` imports (`grep -n "execution_catalog import" dashboard/backend/api/routers/backtests.py`; add `SamplingPolicy` to that list).

Directly after the closing `)` of `execution_handoff_payload = create_execution_handoff(...)` (`:3143`), still inside the LLM block, add:

```python
        llm_sampling = route.sampling
```

In the thread `kwargs={...}` (`:3179`), add after `"execution_handoff_payload": execution_handoff_payload,`:

```python
            "llm_temperature": llm_sampling.temperature if llm_sampling else None,
            "llm_reasoning_effort": llm_sampling.reasoning_effort if llm_sampling else None,
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtest_sampling_argv.py dashboard/backend/tests/test_market_data_features.py dashboard/backend/tests/test_backtest_launch_phases.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add dashboard/scripts/backtest_hourly_agent.py dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtest_sampling_argv.py
git commit -m "feat: hand the route's sampling policy to the backtest child

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: The native OpenAI adapter sends reasoning effort

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/execution/adapters/openai.py:113-131`
- Test: `dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py` (append)

**Interfaces:**
- Produces: for `provider.adapter_type == "openai"`, `reasoning_effort` is sent as the top-level Chat Completions parameter; OpenRouter / openai_compatible keep `extra_body.reasoning`; Anthropic and Gemini adapters keep ignoring it.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py`:

```python
def test_native_openai_sends_reasoning_effort_as_a_top_level_parameter(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _openai_response("gpt-5.5")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )
    monkeypatch.setattr(
        openai_module, "build_safe_http_client", lambda *_args, **_kwargs: _Closable()
    )
    adapter = openai_module.OpenAIAdapter(client_factory=lambda **_kwargs: client)

    adapter.complete(
        _request("openai", "openai/gpt-5.5", reasoning_effort="low"),
        _credential("openai"),
        _provider("openai", "openai", "https://api.openai.com/v1"),
    )

    assert captured["reasoning_effort"] == "low"
    assert "extra_body" not in captured
    assert "temperature" not in captured


def test_native_openai_sends_nothing_extra_when_no_effort_is_requested(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _openai_response("gpt-5.5")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )
    monkeypatch.setattr(
        openai_module, "build_safe_http_client", lambda *_args, **_kwargs: _Closable()
    )
    adapter = openai_module.OpenAIAdapter(client_factory=lambda **_kwargs: client)

    adapter.complete(
        _request("openai", "openai/gpt-5.5"),
        _credential("openai"),
        _provider("openai", "openai", "https://api.openai.com/v1"),
    )

    assert "reasoning_effort" not in captured
    assert "extra_body" not in captured
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py -k native_openai -v`
Expected: the first FAILs with `KeyError: 'reasoning_effort'`; the second passes already.

- [ ] **Step 3: Add the native branch**

In `dashboard/backend/infrastructure/llm/execution/adapters/openai.py`, directly after the existing block that ends with

```python
                kwargs["extra_body"] = {
                    "reasoning": reasoning,
                }
```

add:

```python
            elif request.reasoning_effort and provider.adapter_type == "openai":
                # Chat Completions takes it as a top-level parameter; only
                # reasoning models accept it, and only the catalog's
                # reasoning-only policy ever asks for it here.
                kwargs["reasoning_effort"] = request.reasoning_effort.strip().lower()
```

(the preceding `if request.reasoning_effort and provider.adapter_type in {...}:` is unchanged; this is its `elif`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py dashboard/backend/tests/infrastructure/llm/adapters -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/infrastructure/llm/execution/adapters/openai.py dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py
git commit -m "feat: send reasoning_effort natively to OpenAI reasoning models

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: The results panel shows the Sampling row

**Files:**
- Modify: `dashboard/frontend/app.html:1212` region (the `backtestConfigProvenanceRow` cell)
- Modify: `dashboard/frontend/app.js` — new `formatBacktestSampling` beside `formatBacktestMarketDataProvenance` (`:9697`); `renderBacktestRunConfig` where `provenanceLabel` is computed (`:9735`) and where `provenanceRow` is toggled (`:9942-9956`)
- Modify: the five `app.js?v=` pins
- Test: `dashboard/backend/tests/test_backtest_sampling_row.py` (new)

**Interfaces:**
- Consumes: `agent_runs.metadata.llm_sampling` and `llm_max_output_tokens` from Task 3.
- Produces: `function formatBacktestSampling(sampling) -> string`; DOM ids `backtestConfigSamplingRow`, `backtestConfigSampling`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_backtest_sampling_row.py`:

```python
"""The results panel says what sampling a run asked for.

Three states, never a fourth: *Pinned · …* when the run recorded a policy,
*Provider default* when it recorded that it pinned nothing, *Not recorded*
for an LLM run written before the field existed. The row is hidden for
rule-based runs, which had no sampler. The copy never says "deterministic":
providers are not, at temperature 0 least of all with a mixture-of-experts
model; what is pinned is the request.
"""
import json
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import APP_HTML, fn_body

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _format(sampling_js: str) -> str:
    script = "\n".join(
        [
            fn_body("function formatBacktestSampling("),
            f"console.log(JSON.stringify(formatBacktestSampling({sampling_js})));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_temperature_only():
    assert _format("{temperature: 0, reasoning_effort: null, policy: 'pinned_v1'}") == (
        "Pinned · temperature 0"
    )


def test_reasoning_only_names_the_ignored_temperature():
    assert _format("{temperature: null, reasoning_effort: 'low', policy: 'pinned_v1'}") == (
        "Pinned · reasoning effort low (this model ignores temperature)"
    )


def test_both():
    assert _format("{temperature: 0, reasoning_effort: 'low', policy: 'pinned_v1'}") == (
        "Pinned · temperature 0 · reasoning effort low"
    )


def test_provider_default():
    assert _format("{temperature: null, reasoning_effort: null, policy: 'provider_default'}") == (
        "Provider default"
    )


def test_not_recorded():
    assert _format("null") == "Not recorded"
    assert _format("undefined") == "Not recorded"


def test_copy_never_claims_determinism():
    body = fn_body("function formatBacktestSampling(")
    assert "determin" not in body.lower()


def test_markup_and_renderer_carry_the_row():
    assert 'id="backtestConfigSamplingRow"' in APP_HTML
    assert 'id="backtestConfigSampling"' in APP_HTML
    body = fn_body("function renderBacktestRunConfig(")
    assert "formatBacktestSampling(" in body
    assert "backtestConfigSamplingRow" in body
    # Hidden for rule-based runs: an LLM run is one that recorded a ceiling.
    assert "metadata.llm_max_output_tokens" in body
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_backtest_sampling_row.py -v`
Expected: FAIL — `fn_body` cannot find `function formatBacktestSampling(`.

- [ ] **Step 3: Markup**

In `dashboard/frontend/app.html`, directly after the `backtestConfigProvenanceRow` cell (`:1212` region), add:

```html
                                <div class="backtest-config-detail-cell" id="backtestConfigSamplingRow" hidden><span>Sampling</span><strong id="backtestConfigSampling">—</strong></div>
```

- [ ] **Step 4: Formatter**

In `dashboard/frontend/app.js`, directly after `formatBacktestMarketDataProvenance` (`:9697-9705`), add:

```js
/**
 * The Sampling row's text. `null`/`undefined` is an LLM run written before
 * the engine recorded its policy, so it reads "Not recorded" -- an unrecorded
 * field is unknown, not a default. Never "deterministic": what was pinned is
 * the request, not the provider.
 */
function formatBacktestSampling(sampling) {
    if (!sampling || typeof sampling !== 'object') return 'Not recorded';
    const temperature = Number(sampling.temperature);
    const hasTemperature = sampling.temperature !== null
        && sampling.temperature !== undefined
        && Number.isFinite(temperature);
    const effort = typeof sampling.reasoning_effort === 'string'
        ? sampling.reasoning_effort.trim()
        : '';
    const parts = [];
    if (hasTemperature) parts.push(`temperature ${temperature}`);
    if (effort) parts.push(`reasoning effort ${effort}`);
    if (!parts.length) return 'Provider default';
    const note = !hasTemperature && effort ? ' (this model ignores temperature)' : '';
    return `Pinned · ${parts.join(' · ')}${note}`;
}
```

- [ ] **Step 5: Renderer**

In `renderBacktestRunConfig`, directly after the `provenanceLabel` computation (the `const provenanceLabel = formatBacktestMarketDataProvenance({ ... });` statement near `:9735`), add:

```js
    // Only for a run that used a model: rule-based runs had no sampler, and
    // "Not recorded" beside one would read as an accusation. The ceiling is
    // written on every LLM run and never on a rule-based one.
    const usedModel = Boolean(
        metadata.llm_sampling || metadata.llm_max_output_tokens !== undefined
    );
    const samplingLabel = !running && usedModel
        ? formatBacktestSampling(run?.llm_sampling ?? metadata.llm_sampling ?? null)
        : null;
```

Directly after the three `provenanceRow` lines

```js
    const provenanceRow = document.getElementById('backtestConfigProvenanceRow');
    ...
    if (provenanceRow) provenanceRow.hidden = !provenanceLabel;
```

add:

```js
    const samplingRow = document.getElementById('backtestConfigSamplingRow');
    if (samplingRow) samplingRow.hidden = !samplingLabel;
    if (samplingLabel) {
        setBacktestConfigText('backtestConfigSampling', samplingLabel);
    }
```

- [ ] **Step 6: Bump the cache-buster in five files**

```bash
grep -rn "app.js?v=" dashboard/frontend/app.html dashboard/backend/tests/*.py
```

Replace `app.js?v=N` with `app.js?v=N+1` in every file listed; re-run the grep to confirm one number, five files.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtest_sampling_row.py dashboard/backend/tests/test_byok_backtest_frontend.py dashboard/backend/tests/test_minute_data_frontend.py dashboard/backend/tests/test_backtest_run_provenance.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_backtest_comparison_frontend.py dashboard/backend/tests/test_analytics_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add dashboard/frontend/app.js dashboard/frontend/app.html dashboard/backend/tests/
git commit -m "feat: show the sampling a backtest asked for

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: A script that says where two runs first disagree

**Files:**
- Create: `dashboard/scripts/diff_backtest_runs.py`
- Test: `dashboard/backend/tests/test_diff_backtest_runs.py` (new)

**Interfaces:**
- Produces: `compare_decisions(a: list[dict], b: list[dict]) -> dict` and `compare_runs(run_a: str, run_b: str) -> dict` (keys: `steps_compared`, `steps_a`, `steps_b`, `divergent_steps`, `first_divergence`, `run_a`, `run_b`, `final_equity_a`, `final_equity_b`, `final_equity_gap_pct`, `sampling_a`, `sampling_b`); CLI `python dashboard/scripts/diff_backtest_runs.py <run_a> <run_b>` printing that dict as JSON.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_diff_backtest_runs.py`:

```python
"""Where do two backtests of one configuration first disagree?

The proof for pinned sampling is a number, not a promise: the first bar at
which two runs of one configuration diverge, how many bars diverge, and how
far the final equity moves. This script reads both off the tables the
dashboard already writes.
"""
import importlib.util
import sys
from pathlib import Path

from dashboard.backend.database import BacktestDatabase

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_script():
    path = _SCRIPTS_DIR / "diff_backtest_runs.py"
    spec = importlib.util.spec_from_file_location("diff_backtest_runs_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SCRIPTS_DIR))
    return module


def _decision(step, actions):
    return {
        "step_index": step,
        "timestamp": f"2026-09-0{1 + step // 7}T1{step % 7}:00:00",
        "decision_source": "llm",
        "actions_submitted": actions,
        "actions_executed": len(actions),
    }


def _seed(db, run_id, decisions, final_equity):
    db.insert_run(
        run_id=run_id,
        session_id="diff-session",
        agent_name="diff-agent",
        mode="backtest",
        start_date="2026-09-01",
        end_date="2026-09-08",
        initial_equity=100000.0,
        final_equity=final_equity,
        metadata={"llm_sampling": {"temperature": 0.0, "reasoning_effort": None}},
    )
    db.insert_decisions(run_id, decisions)


def test_compare_decisions_finds_the_first_divergent_bar():
    module = _load_script()
    a = [_decision(0, []), _decision(1, [{"symbol": "AAPL", "side": "buy"}]), _decision(2, [])]
    b = [_decision(0, []), _decision(1, []), _decision(2, [{"symbol": "MSFT", "side": "buy"}])]

    report = module.compare_decisions(a, b)

    assert report["steps_compared"] == 3
    assert report["divergent_steps"] == 2
    assert report["first_divergence"] == {"step_index": 1, "timestamp": a[1]["timestamp"]}


def test_compare_decisions_ignores_key_order_inside_an_action():
    module = _load_script()
    a = [_decision(0, [{"symbol": "AAPL", "side": "buy"}])]
    b = [_decision(0, [{"side": "buy", "symbol": "AAPL"}])]
    assert module.compare_decisions(a, b)["divergent_steps"] == 0


def test_compare_runs_reads_both_rows(tmp_path, monkeypatch):
    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed(db, "run_a", [_decision(0, []), _decision(1, [{"symbol": "AAPL", "side": "buy"}])], 101000.0)
    _seed(db, "run_b", [_decision(0, []), _decision(1, [])], 99990.0)

    report = module.compare_runs("run_a", "run_b")

    assert report["run_a"] == "run_a"
    assert report["divergent_steps"] == 1
    assert report["first_divergence"]["step_index"] == 1
    assert report["final_equity_a"] == 101000.0
    assert report["final_equity_b"] == 99990.0
    assert round(report["final_equity_gap_pct"], 4) == -1.0
    assert report["sampling_a"] == {"temperature": 0.0, "reasoning_effort": None}


def test_compare_runs_refuses_an_unknown_run(tmp_path, monkeypatch):
    import pytest

    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed(db, "run_a", [_decision(0, [])], 100000.0)

    with pytest.raises(SystemExit, match="run_zzz"):
        module.compare_runs("run_a", "run_zzz")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_diff_backtest_runs.py -v`
Expected: FAIL — `AssertionError` from `_load_script` (the file does not exist).

- [ ] **Step 3: Write the script**

Create `dashboard/scripts/diff_backtest_runs.py`:

```python
#!/usr/bin/env python3
"""Where do two backtests of one configuration first disagree?

Prints, as JSON, the first divergent bar, how many bars diverged, and the
final-equity gap between two saved runs, plus the sampling each run recorded.
Reads ``backtest_decisions`` and ``agent_runs`` through the same ``db`` the
dashboard uses, so it works against local SQLite or, with
``AGENT_RUNS_DATABASE_URL`` set, against Postgres.

    python dashboard/scripts/diff_backtest_runs.py <run_id_a> <run_id_b>

The number this exists for is *later and smaller*, not zero: providers are
not deterministic at temperature 0, and each bar's prompt embeds the previous
bar's answer, so one different draw is carried to the end of the run.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

if not __package__:
    from _bootstrap import ensure_repo_root

    ensure_repo_root()

from dashboard.backend.database import db  # noqa: E402


def _normalised_actions(entry: Dict[str, Any]) -> str:
    return json.dumps(entry.get("actions_submitted") or [], sort_keys=True)


def compare_decisions(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bar-by-bar comparison of two decision logs, in step order."""
    divergent = 0
    first: Optional[Dict[str, Any]] = None
    for x, y in zip(a, b):
        if _normalised_actions(x) != _normalised_actions(y):
            divergent += 1
            if first is None:
                first = {
                    "step_index": int(x.get("step_index", 0)),
                    "timestamp": x.get("timestamp"),
                }
    return {
        "steps_compared": min(len(a), len(b)),
        "steps_a": len(a),
        "steps_b": len(b),
        "divergent_steps": divergent,
        "first_divergence": first,
    }


def compare_runs(run_a: str, run_b: str) -> Dict[str, Any]:
    row_a = db.get_run(run_a)
    row_b = db.get_run(run_b)
    missing = [rid for rid, row in ((run_a, row_a), (run_b, row_b)) if row is None]
    if missing:
        raise SystemExit(f"unknown run id(s): {', '.join(missing)}")
    report = compare_decisions(db.get_decisions(run_a), db.get_decisions(run_b))
    final_a = row_a.get("final_equity")
    final_b = row_b.get("final_equity")
    gap_pct = None
    if final_a and final_b is not None:
        gap_pct = 100.0 * (float(final_b) - float(final_a)) / float(final_a)
    report.update(
        {
            "run_a": run_a,
            "run_b": run_b,
            "final_equity_a": final_a,
            "final_equity_b": final_b,
            "final_equity_gap_pct": gap_pct,
            "sampling_a": (row_a.get("metadata") or {}).get("llm_sampling"),
            "sampling_b": (row_b.get("metadata") or {}).get("llm_sampling"),
        }
    )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Where do two backtests of one configuration first disagree?"
    )
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    args = parser.parse_args(argv)
    print(json.dumps(compare_runs(args.run_a, args.run_b), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_diff_backtest_runs.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/scripts/diff_backtest_runs.py dashboard/backend/tests/test_diff_backtest_runs.py
git commit -m "feat: add a script that diffs two backtest runs bar by bar

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Document, measure, verify, PR

**Files:**
- Modify: `CLAUDE.md` (the **LLM backtest billing** bullet)
- Modify: this plan's **Final verification** section

- [ ] **Step 1: CLAUDE.md**

Append this paragraph to the end of the `LLM backtest billing — the live path` bullet in `CLAUDE.md`:

```markdown
**Sampling is pinned per catalog model** (`domain/model_providers/execution_catalog.py:SamplingPolicy`): Claude and Gemini get temperature 0; GPT-5.5 gets reasoning effort `low` and **no temperature** (OpenAI reasoning models reject one); DeepSeek V4 Pro and Qwen3.7 Plus get both. The route the endpoint preflights carries the policy, it rides the child's argv as `--llm-temperature` / `--llm-reasoning-effort` (each only when set; the plain Anthropic SDK client on the CLI path rejects an unknown `reasoning_effort`), and `pipeline_runner._create_pipeline_response` adds each value only when set so an unset half leaves the request byte-identical. The engine records what it asked for as `agent_runs.metadata.llm_sampling` (`policy: pinned_v1 | provider_default`), and the results panel's **Sampling** row reads *Pinned · …*, *Provider default*, or *Not recorded* for rows written before the field existed. ⚠ **Pinned is not deterministic** — providers are not, at temperature 0 least of all with a mixture-of-experts model, and each bar's prompt embeds the previous bar's answer — so the claim on screen is the request, and the spread is measured with `dashboard/scripts/diff_backtest_runs.py <run_a> <run_b>` (first divergent bar, divergent bar count, final-equity gap). The measured before/after is in `docs/superpowers/plans/2026-09-20-backtest-pinned-sampling.md`.
```

Commit:

```bash
git add CLAUDE.md
git commit -m "docs: describe the pinned sampling policy

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 2: Measure before and after, locally**

"Before" runs pin nothing (no flags). "After" runs pass the CLI flags the parent would derive. The CLI path uses the Anthropic SDK client, so the model is Claude; the point is the spread, not the model. Write `<scratchpad>/rerun.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
set -a; source dashboard/.env; set +a
TAG="${1:?before|after}"; N="${2:?run number}"
EXTRA=()
if [ "$TAG" = "after" ]; then EXTRA=(--llm-temperature 0); fi
python3 dashboard/scripts/backtest_hourly_agent.py \
  --start 2026-09-08 --end 2026-09-16 --session-id "rerun-$TAG" \
  --use-llm --run-id "rerun_${TAG}_$N" "${EXTRA[@]}"
```

Run it six times: `bash <scratchpad>/rerun.sh before 1`, `... before 2`, `... before 3`, then `after 1..3`. Then:

```bash
for p in "before_1 before_2" "before_1 before_3" "before_2 before_3" "after_1 after_2" "after_1 after_3" "after_2 after_3"; do
  set -- $p
  python3 dashboard/scripts/diff_backtest_runs.py "rerun_$1" "rerun_$2" | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["run_a"], r["run_b"], "first", (r["first_divergence"] or {}).get("step_index"), "divergent", r["divergent_steps"], "/", r["steps_compared"], "gap%", r["final_equity_gap_pct"])'
done
```

Record the six lines in the **Final verification** table. Six LLM runs of 49 bars at one step each is roughly 300 model calls; that is the cost of the number.

- [ ] **Step 3: Full suite and hygiene**

Run: `pytest dashboard/backend/tests/ -q` → green.
Run: `git status --short dashboard/storage/data/backtest.db` → empty (the reruns above write to whatever `DATABASE_PATH` the shell has; if the seed DB changed, `git checkout -- dashboard/storage/data/backtest.db`).
Run: `grep -rn "app.js?v=\|styles.css?v=" dashboard/frontend/app.html dashboard/backend/tests/*.py` → one number per asset.

- [ ] **Step 4: Record and open the PR**

Fill the table, commit the plan update, then:

```bash
git push -u origin feat/backtest-pinned-sampling
gh pr create --title "feat: pin backtest sampling and show it" --body "$(cat <<'EOF'
Track B of docs/superpowers/specs/2026-09-20-backtest-speed-and-trust-design.md.

- per-model sampling policy on the execution catalog; rides the route → argv → engine → pipeline runner → worker client
- values sent only when set; CLI Anthropic client unaffected
- `agent_runs.metadata.llm_sampling` recorded; Sampling row on the results panel
- `dashboard/scripts/diff_backtest_runs.py` + measured 3x3 rerun spread in the plan

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Final verification

| pair | first divergent bar | divergent bars / compared | final-equity gap % |
|---|---|---|---|
| before 1 vs 2 | | | |
| before 1 vs 3 | | | |
| before 2 vs 3 | | | |
| after 1 vs 2 | | | |
| after 1 vs 3 | | | |
| after 2 vs 3 | | | |

Model used for the reruns: _(the CLI's Anthropic default)_. Bars per run: 49 (7 weekdays × 7 hourly bars).

Full suite: `pytest dashboard/backend/tests/ -q` → _result_.
Seed DB: clean.
Cache-busters: one number per asset, five files.

## Out of scope (from the spec)

Track C (per-bar deadline, failover timeout, snapshot rebuild off the hot path), Track D (re-run and compare in the UI), the cadence and result-contract doors, changing the modal's default model. If the after-runs still show a completed-vs-failed split on the DeepSeek route in prod, that last item becomes a product decision to raise separately.

## User-facing docs to check after this ships

`docs/source/lab/operating_modes.rst` and `docs/source/lab/key_features.rst`: neither currently makes a reproducibility claim. Once the Sampling row is live, re-read both and file a docs follow-up if a sentence should now name it.
