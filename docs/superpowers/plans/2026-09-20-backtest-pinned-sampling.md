# Backtest Pinned Sampling (Track B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every model call a dashboard backtest makes carries a pinned sampling policy chosen per catalog model, the run records what it asked for, the results panel shows it, and a rerun of one configuration is measured against the first run.

**Architecture:** The policy lives on the execution catalog (`CatalogModel.sampling`) and rides the resolved route the endpoint already preflights. It reaches the child as two argv flags, then engine → portfolio manager → pipeline runner → the worker client, which already validates `temperature` and `reasoning_effort`. The request builder adds each value only when set, so the CLI's plain Anthropic SDK client keeps working. The engine writes `llm_sampling` into `agent_runs.metadata`; the results panel renders it as a Sampling row. A dev script diffs two runs bar by bar and reports which axis it measured (`basis`: the decision log when one exists, otherwise the equity curve — and for every run this plan measures it is the equity curve, because the pipeline runtime writes no decision log at all).

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
- **Line numbers below are read against `c68076ac` — `origin/main` plus this set's two docs commits, byte-identical to `origin/main` under `dashboard/`, which is the same reference the spec names — and this branch is cut two merges later: the largest drift of the three documents in this set.** The spec and the Track A plan each sit one merge from their anchors; this one sits behind *both* the spec's sequencing step 0 (#474's `feat/474-backtest-timeout-outcome`) **and** Track A's PR on top of it. The spec's own line-reference paragraph accounts for itself and for Track A; it does not cover this plan, so the accounting is here. Measured 2026-09-20 against that branch's merge-base: #474 changes `api/routers/backtests.py` by **+168/−10** and `frontend/app.js` by **+123/−15**, and touches nothing else this plan edits — `app.html` only in its two cache-buster `?v=` tags, both far below the markup this plan touches, so `app.html:1257` survives it. (Those two tags are `styles.css` at `:16` and the `app.js` tag at `:2701` on #474's branch — `:2342` on `main`, which has since moved it. The `13 and 2698` an earlier draft carried were the unified-diff **hunk headers**, not the changed lines; `git diff` prints the context start, and three lines of context is exactly the error. A number read off a hunk header is wrong in a paragraph whose only job is getting numbers right.) Track A then edits `backtests.py`, `app.js`, `engine.py` and `backtest_hourly_agent.py` again.
- **So: grep for the `def <name>(` / `function <name>(` line, never jump to the number.** Every number below is a hint about where to start grepping, not an address. Two of the four moving files are stale by both merges (`backtests.py`, `app.js`) and two by Track A alone (`engine.py`, `backtest_hourly_agent.py`); the **Files** entry for each carries its anchor text for that reason, and the quoted text beats the number wherever the two disagree. Already demonstrated, not hypothetical: `backtests.py:1381` is `def run_backtest_background(` here and `def _read_progress_file(` on the post-#474 tree — where the real signature has moved to `:1434`, and where Track A Task 3 Step 3 inserts `PROGRESS_PHASE_MESSAGES` + `_progress_message` directly above `def _read_progress_file(`. A reader who jumped to `:1381` would land inside Track A's new block and edit that. The files neither merge touches — `execution_catalog.py`, `pipeline_runner.py`, `backtest_harness.py`, `portfolio_manager.py`, `adapters/openai.py` and every test module this plan edits — should still land on their numbers; check anyway. The one test anchor worth a second look is the `ExecutionModelRoute(` stub list in Task 1's **Interfaces** (`test_backtests_router.py:348/383/407`): #474 adds 322 lines to that file (+322/−4) and Track A appends to it again, but both land far below `:407`, so the three survive — verified 2026-09-20 on `feat/474-backtest-timeout-outcome`, where they are still at `:348/383/407`.

---

### Task 1: The policy lives on the catalog

**Files:**
- Modify: `dashboard/backend/domain/model_providers/execution_catalog.py` (dataclasses at `:14-25`, entries at `:28-51`, `list_execution_model_routes` at `:80-96`, `__all__` at `:114`)
- Test: `dashboard/backend/tests/test_execution_catalog_sampling.py` (new)

**Interfaces:**
- Produces: `SamplingPolicy(temperature: float | None, reasoning_effort: str | None)` (frozen dataclass); constants `PINNED_TEMPERATURE`, `PINNED_REASONING_LOW`, `PINNED_BOTH`; `CatalogModel.sampling: SamplingPolicy` — **no default**, every catalog row states its policy at the construction site; `ExecutionModelRoute.sampling: SamplingPolicy | None = None` — a route nobody resolved from the catalog carries *no* policy rather than a pinned one, which is what the six existing `ExecutionModelRoute(...)` stubs in tests get and what the endpoint already renders as two absent flags.

The two fields are deliberately treated differently, and the asymmetry is the point. `CatalogModel` is the **declaration**: the catalog is the single owner of the policy table, and **all six** of its rows are written `CatalogModel("vendor/id", "Label", "vendor")` today — three positional arguments and nothing else, verified 2026-09-20 against `execution_catalog.py:28-51`. A `PINNED_TEMPERATURE` default would pin `temperature=0` on the next OpenAI reasoning model added in that house style — a model that answers 400 to any non-default temperature, so every backtest on that route would fail at the provider with nothing local to see. The spec already says it: a model not in the catalog cannot be launched from the dashboard, "so there is no default row" (the spec's **⚠ changed: the policy is a per-model table, not a flag** section, under the catalog table; cited by heading plus quoted sentence rather than by line, because the spec is being edited in the same pass as this plan). Re-derived 2026-09-20 with `grep -n "^#" docs/superpowers/specs/2026-09-20-backtest-speed-and-trust-design.md`: that `###` heading and the `## Track B — sampling pinned, recorded, shown` above it are the only two the sentence lives under, and **Track B — Sampling policy** — which an earlier draft of this paragraph named, having picked the citation *for* its robustness — is not a heading the spec has ever had. `ExecutionModelRoute` is a **derived value**: `list_execution_model_routes` always fills it from the model, so the only constructions that omit it are the six test stubs standing in for a preflight (`test_backtests_router.py:348/383/407`, `test_credit_metering.py:35`, `test_market_data_features.py:70`, `integration/test_ifind_ashare_backtest.py:195` — all keyword-style, all three fields, verified 2026-09-20). For those, `None` is the truthful answer, and the endpoint's `llm_sampling.temperature if llm_sampling else None` already turns it into "no flags". Requiring it there would make six stubs restate a policy they are not testing — the kind of noise that gets copy-pasted wrong — and would buy nothing that Step 1's `test_every_route_the_catalog_builds_carries_a_policy` does not already buy.

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
    CatalogModel,
    ExecutionModelRoute,
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


def test_a_catalog_row_cannot_forget_to_state_its_policy():
    """`sampling` has no default, and the missing default *is* the guard.

    Every row in the catalog was written `CatalogModel("vendor/id", "Label",
    "vendor")` before this change -- three positional arguments and nothing
    else -- so that is the form the next one gets copied into. With a
    `PINNED_TEMPERATURE` default, the next OpenAI reasoning
    model added that way silently pins `temperature=0` on a route
    that returns 400 for any non-default temperature -- every backtest on it
    failing at the provider, with nothing local to see and a metadata row
    confidently reporting `policy: pinned_v1`. The table test above does not
    catch it either: whoever adds the row also adds it to `_EXPECTED`, and if
    they copy the default they wrote by accident, the assertion agrees with
    them. Only the construction site can ask the question.
    """
    with pytest.raises(TypeError):
        CatalogModel("openai/o5", "o5", "openai")


def test_a_hand_built_route_claims_no_policy():
    """`ExecutionModelRoute.sampling` defaults to None, not to a policy.

    Only `list_execution_model_routes` builds a real route, and it always
    fills this from the model. The constructions that omit it are the six test
    stubs standing in for a preflight, and for those `None` is the truthful
    answer: nothing was resolved. A `PINNED_TEMPERATURE` default would have
    each of them claim a pinning nobody asked for, and the endpoint would then
    hand the child `--llm-temperature 0` for a route it never looked up.
    """
    route = ExecutionModelRoute(
        catalog_id="openai/gpt-5.5",
        label="GPT-5.5",
        provider_model_id="gpt-5.5",
    )
    assert route.sampling is None


def test_every_route_the_catalog_builds_carries_a_policy():
    """That None is reachable by hand only; the real builder never emits it."""
    provider = ProviderRecord(
        provider_id="openrouter",
        display_name="OpenRouter",
        adapter_type="openrouter",
        approved_base_url="https://openrouter.ai/api/v1",
    )
    routes = list_execution_model_routes(provider)
    assert routes
    assert all(route.sampling is not None for route in routes)
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
    # No default, deliberately. Every row below used to be three positional
    # arguments and nothing else, which is the form the next one gets copied
    # into; a default here would pin temperature 0 on the next OpenAI
    # reasoning model added that way -- a model that returns 400 for any
    # non-default temperature. The spec's rule is that a model not in this
    # catalog cannot be launched at all, "so there is no default row";
    # leaving the field required is that rule with a compiler behind it.
    sampling: SamplingPolicy


@dataclass(frozen=True)
class ExecutionModelRoute:
    catalog_id: str
    label: str
    provider_model_id: str
    # None, not a policy. Only `list_execution_model_routes` builds a real
    # route and it always fills this from the model; the constructions that
    # omit it are test stubs standing in for a preflight, and for those the
    # honest answer is "nothing was resolved". A pinned default would have a
    # hand-built route claim a policy nobody chose, and the endpoint would
    # then send `--llm-temperature 0` for a route it never looked up.
    sampling: SamplingPolicy | None = None
```

`SamplingPolicy | None` needs no `Optional` import: the module opens with `from __future__ import annotations`, and already writes `str | None` at `:63`.

Replace the catalog entries (`:28-51`) with:

```python
ATL_EXECUTION_MODELS = (
    CatalogModel(
        "anthropic/claude-haiku-4-5",
        "Claude Haiku 4.5",
        "anthropic",
        PINNED_TEMPERATURE,
    ),
    CatalogModel(
        "anthropic/claude-sonnet-4-6",
        "Claude Sonnet 4.6",
        "anthropic",
        PINNED_TEMPERATURE,
    ),
    # OpenAI reasoning models reject a non-default temperature outright.
    CatalogModel("openai/gpt-5.5", "GPT-5.5", "openai", PINNED_REASONING_LOW),
    CatalogModel(
        "google/gemini-3.1-pro-preview",
        "Gemini 3.1 Pro Preview",
        "google",
        PINNED_TEMPERATURE,
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

Three rows gain an explicit `PINNED_TEMPERATURE` that a default would have supplied. That is the visible cost of removing the default, and it is the whole benefit: the three rows that pin a temperature now say so where someone copying one of them will read it.

In `list_execution_model_routes` (`:80-96`), add `sampling=model.sampling,` to the `ExecutionModelRoute(` construction (`:90-94`) after `provider_model_id=provider_model_id,`. This is the only production construction of a route, so it is the only place the `None` default is not taken.

**No test stub is edited.** The six existing `ExecutionModelRoute(...)` constructions (`test_backtests_router.py:348/383/407`, `test_credit_metering.py:35`, `test_market_data_features.py:70`, `integration/test_ifind_ashare_backtest.py:195`) keep compiling and now carry `sampling=None`, which is what they mean — none of them is standing in for a catalog lookup.

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

Run: `pytest dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py -k "sampling or set_half or default_request_shape" -v`
Expected: 4 selected. The first two FAIL with `TypeError: run_pipeline_decision() got an unexpected keyword argument 'temperature'`; `test_run_pipeline_decision_sends_only_the_set_half_of_a_policy` FAILs with `... unexpected keyword argument 'reasoning_effort'`, since it passes that one alone; `test_run_pipeline_decision_default_request_shape_is_unchanged` passes already, and that is fine.

**Count the selection before trusting the red.** An earlier draft used `-k "sampling or truncation_retry_keeps or default_request_shape"`, which silently selected **three** of the four: `test_run_pipeline_decision_sends_only_the_set_half_of_a_policy` matches none of those substrings, nor the module name. That is the one case covering the reasoning-effort-without-temperature shape — the whole `PINNED_REASONING_LOW` row of the policy table, and the only new test covering a policy class on its own — so it would have been written, never watched fail, and marked green in Step 4 with no evidence it ever tested anything. Track A Task 4 Step 3 states the rule this breaks: a repair nobody watched fail is a repair nobody has tested. `set_half` is the substring that picks it up; verified 2026-09-20 that none of the 30 existing `test_*` in this module matches any of the three (the nearest are `truncation_retry_degrades` and `truncation_retry_propagates`), so the expression selects exactly the four new cases.

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
- Modify: `dashboard/backend/domain/backtesting/portfolio_manager.py` (`make_trading_decision_with_llm` signature at `:244-255`; the `run_pipeline_decision(` call at `:462-468`; **three** `_request_trading_decision(` calls at `:525-532`, `:534-540` and `:575-582`). Three, not two — re-derived 2026-09-20 with `grep -n "_request_trading_decision(" dashboard/backend/domain/backtesting/portfolio_manager.py`; all three are inside this one method, which runs to `:803` (the next `def` is `_record_llm_usage` at `:804`).
- Modify: `dashboard/backend/domain/backtesting/engine.py` — one merge stale rather than two (#474 does not touch this file; Track A does). Grep the quoted line, not the number, at each of the four sites: the `def __init__(` keyword block (`:180-205`); `        self.model = model or default_model_name()` (`:235` — `:236` is `self.execution_client = execution_client`, and an earlier draft of this plan named `:236` at both of its mentions, which would have inserted the two `llm_*` attributes one line below the line the prose describes); the `make_trading_decision_with_llm(` call (`:1483-1491`); and `    def _agent_run_metadata(self) -> Dict:` (`:1008`) down to its `meta["llm_max_output_tokens"] = …` line (`:1052`), which is where the new key lands — the method itself runs to `:1135`, so `:1052` is the insertion point and not the end of it
- Test: `dashboard/backend/tests/llm/test_backtest_harness.py` (append), `dashboard/backend/tests/test_agent_runs_metadata.py` (edit `test_engine_llm_run_metadata_snapshot` at `:130-162`, append), `dashboard/backend/tests/test_backtest_sampling_wiring.py` (new)

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
"""The sampling policy reaches the model call from the engine, on every branch.

Pinned by source shape, the way test_agent_runs_metadata pins the metadata
call site: the decision method needs a full market snapshot to reach either
branch, and a dropped keyword between two files is exactly the defect this
work fixes (the pipeline branch never forwarded temperature although the
single-prompt branch did).

Read by AST, and asserted **universally rather than by count**. An earlier
draft of this file asserted `src.count("_request_trading_decision(") == 2` and
`src.count("reasoning_effort=reasoning_effort,") == 3`. There are three call
sites, not two -- the third is the truncation-recovery retry -- so the numbers
were wrong and the test could never have gone green. But the numbers are the
smaller half of the problem: a count is the wrong *shape* of guard here. It
goes red when someone adds a correct fourth call site, and it stays green when
someone adds an incorrect one that happens to keep the total. "Every model
call in this method forwards both values" is the property the code has to
have, so it is the property the test states, and the number never appears.
The non-empty check is the other half of that trade: a universal assertion
over an empty set passes, so a rename that makes the query match nothing must
fail loudly rather than quietly cover nothing.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

from dashboard.backend.domain.backtesting import engine, portfolio_manager


def _callee(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls(method, *names: str) -> list[ast.Call]:
    """Every call to one of `names` inside `method`, as AST nodes.

    `textwrap.dedent` because `inspect.getsource` of a method keeps the class
    indentation, which `ast.parse` refuses outright.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _callee(node) in set(names)
    ]
    assert found, (
        f"no call to {sorted(names)} in {method.__qualname__} -- the query is "
        "stale, not the code. Fix this helper before trusting anything below: "
        "a 'for every call' assertion over nothing passes."
    )
    return found


def _kwargs(call: ast.Call) -> dict:
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}


def test_engine_forwards_both_values_to_the_manager():
    for call in _calls(
        engine.HourlyBacktester.run_agent_backtest,
        "make_trading_decision_with_llm",
    ):
        kwargs = _kwargs(call)
        assert kwargs.get("temperature") == "self.llm_temperature"
        assert kwargs.get("reasoning_effort") == "self.llm_reasoning_effort"


def test_every_model_call_in_the_manager_forwards_both_values():
    """One policy governs both branches, and both branches are every call.

    The pipeline branch (`run_pipeline_decision`) and the single-prompt branch
    (`_request_trading_decision`) are the only places this method reaches a
    model. Whatever the current count of call sites, each one has to carry the
    same two values, because the alternative is a step that silently changes
    sampler mid-run while the metadata still reports the pinned policy.
    """
    for call in _calls(
        portfolio_manager.PortfolioManager.make_trading_decision_with_llm,
        "_request_trading_decision",
        "run_pipeline_decision",
    ):
        kwargs = _kwargs(call)
        assert kwargs.get("temperature") == "temperature", (
            f"{_callee(call)} at line {call.lineno} of the method drops temperature"
        )
        assert kwargs.get("reasoning_effort") == "reasoning_effort", (
            f"{_callee(call)} at line {call.lineno} of the method drops reasoning_effort"
        )


def test_a_recovery_call_is_the_same_request_at_a_higher_ceiling():
    """A recovery call may differ from the ordinary one by max_tokens alone.

    Two of them raise the ceiling: the final rescue after the empty-reply
    retries are spent, and the truncation retry after the parse fails. The
    truncation one fires on exactly the reasoning-heavy models the policy pins
    an effort for, so a recovery that re-asked without the sampling would be a
    *different request* on the runs most likely to need it -- the thing the
    spec forbids when it says the recovery retry "carries the same sampling,
    because it is the same request at a higher ceiling".

    Stated as an equality against the ordinary call rather than as two
    `in` checks, so a third value added to one path and not the other is
    caught without this test having to learn its name.
    """
    calls = _calls(
        portfolio_manager.PortfolioManager.make_trading_decision_with_llm,
        "_request_trading_decision",
    )
    recovery = [
        c for c in calls
        if _kwargs(c).get("max_tokens") == "RECOVERY_MAX_OUTPUT_TOKENS"
    ]
    ordinary = [c for c in calls if c not in recovery]
    assert recovery, "no recovery call found -- the ceiling constant was renamed"
    assert ordinary, "no ordinary call found -- every call now raises the ceiling"
    for call in recovery:
        assert _kwargs(call) == {
            **_kwargs(ordinary[0]),
            "max_tokens": "RECOVERY_MAX_OUTPUT_TOKENS",
        }


def test_engine_accepts_the_two_kwargs():
    params = inspect.signature(engine.HourlyBacktester.__init__).parameters
    assert params["llm_temperature"].default is None
    assert params["llm_reasoning_effort"].default is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/llm/test_backtest_harness.py::test_request_sends_reasoning_effort_only_when_set dashboard/backend/tests/test_agent_runs_metadata.py dashboard/backend/tests/test_backtest_sampling_wiring.py -v`
Expected: FAIL — `TypeError: request_trading_decision() got an unexpected keyword argument 'reasoning_effort'`, the snapshot dict mismatch, `KeyError: 'llm_temperature'`, and from the wiring file `AssertionError: run_pipeline_decision at line N of the method drops temperature` (the AST guards name the offending call site rather than a total).

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

In `make_trading_decision_with_llm` (`:244-255`), add `reasoning_effort: Optional[str] = None,` directly after `temperature: Optional[float] = None,` (`:252`).

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

In **all three** `_request_trading_decision(` calls on the single-prompt branch, add `reasoning_effort=reasoning_effort,` directly after the `temperature=temperature,` line. Three, not two — re-derived 2026-09-20 with `grep -n "_request_trading_decision(" dashboard/backend/domain/backtesting/portfolio_manager.py`, which reports `:525`, `:534` and `:575`, all inside this one method:

- `:525-532` — the **final rescue** call, taken once the empty-reply retries are spent. Already carries `max_tokens=RECOVERY_MAX_OUTPUT_TOKENS`.
- `:534-540` — the **ordinary** attempt, every other time round that loop.
- `:575-582` — the **truncation-recovery retry**, reached after `_parse_llm_response` returns `None` and `truncation_reason` names a reply the provider cut at the ceiling. Also already carries `max_tokens=RECOVERY_MAX_OUTPUT_TOKENS`.

The third is the one an earlier draft of this plan missed, and it is the site that matters most. It fires on precisely the reasoning-heavy models the policy pins an effort for — GPT-5.5, DeepSeek V4 Pro, Qwen3.7 Plus — so leaving it unpatched would re-ask *a different request* on the runs most likely to reach it, which is exactly what the spec forbids when it says the recovery retry "carries the same sampling, because it is the same request at a higher ceiling" (the spec's **The wire** section). Worse than the nondeterminism itself: the step would silently change sampler mid-run while `agent_runs.metadata.llm_sampling` went on reporting `policy: pinned_v1`, so the row would claim a pinning the run did not have.

After the edit the two recovery calls differ from the ordinary one by `max_tokens` and nothing else, which is the property Step 1's `test_a_recovery_call_is_the_same_request_at_a_higher_ceiling` asserts — an equality rather than a count, so a fourth call site added later has to satisfy the same rule instead of breaking a number.

- [ ] **Step 5: Engine**

In `HourlyBacktester.__init__` (`:180`), append after `startup_clock: Optional[Dict[str, float]] = None,` — Track A adds two keywords here, `launched_at` then `startup_clock`, and `startup_clock` is the last one:

```python
        llm_temperature: Optional[float] = None,
        llm_reasoning_effort: Optional[str] = None,
```

Directly after `self.model = model or default_model_name()` (`:235`), add:

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
- Modify: `dashboard/scripts/backtest_hourly_agent.py` — one merge stale (Track A only). Three quoted anchors, all shifted down by whatever Track A's `CHILD_ENTERED_AT` preamble above `import sys`, its `IMPORTS_DONE_AT` stamp and its two new argparse arguments add above them: argparse directly after the `--launched-at` `add_argument`; a flag-coherence check directly above `    execution_handoff = None` (`:271` on `main`); the `    backtester = HourlyBacktester(` call (`:443` on `main`)
- Modify: `dashboard/backend/api/routers/backtests.py` — **two merges stale; grep every one of these, do not jump.** `def run_backtest_background(` and its signature (`:1381-1400` on `main`, `:1434` on the post-#474 tree — and `:1381` *there* is `def _read_progress_file(`, which is also where Track A Task 3 Step 3 inserts `PROGRESS_PHASE_MESSAGES` + `_progress_message`, so a reader who trusts the old number edits Track A's new block instead); argv directly after `            cmd += ["--model", model.strip()]` (`:1529` on `main`, `:1589` post-#474, and shifted down again by Track A's launch-clock hoist — the `launched_at = time.time()` block Track A Task 2 Step 4 writes into `:1499-1509`, which is *above* this line. Track A's own `--launched-at` argv edit is not what moves it: that goes into the `cmd += ["--run-id", …, "--progress-file", …]` line at `:1594`, immediately **below** this insertion point, so the two land adjacent and neither displaces the other); then in `run_backtest_endpoint`, after `    execution_handoff_payload: Optional[str] = None` (`:3046`), after `        execution_handoff_payload = create_execution_handoff(` (`:3126-3143`), and inside the thread's `        kwargs={` block (`:3179-3199`)
- Test: `dashboard/backend/tests/test_backtest_sampling_argv.py` (new)

**Interfaces:**
- Consumes: `ExecutionModelRoute.sampling` from Task 1; `HourlyBacktester(llm_temperature=, llm_reasoning_effort=)` from Task 3.
- Produces: child argv `--llm-temperature <float repr>` and `--llm-reasoning-effort <lowercase str>`, each present only when its value is set; `run_backtest_background(..., llm_temperature: Optional[float] = None, llm_reasoning_effort: Optional[str] = None)`. `--llm-reasoning-effort` without `--execution-handoff-stdin` is a **CLI error (exit 2)**, not a run.

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
    PINNED_TEMPERATURE,
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


def test_an_unset_temperature_is_absent_from_argv(monkeypatch):
    """The `PINNED_REASONING_LOW` shape -- GPT-5.5, which rejects a
    temperature outright, so the absent flag is the whole point of the row.
    Also the one case that watches the `.lower()` normalisation."""
    command = _capture_command(monkeypatch, llm_reasoning_effort="LOW")
    assert "--llm-temperature" not in command
    assert command[command.index("--llm-reasoning-effort") + 1] == "low"


def test_an_unset_reasoning_effort_is_absent_from_argv(monkeypatch):
    """The `PINNED_TEMPERATURE` shape, and the mirror is not free.

    Three of the catalog's six rows carry it -- both Claude models and
    Gemini 3.1 Pro Preview -- so this is the argv a dashboard backtest emits
    most often, and it was the one policy class the argv layer never saw.
    It does not follow from the case above, because the two halves are built
    by different code in Step 4: the temperature branch guards on
    `is not None` and renders `repr(float(...))`, the effort branch guards on
    truthiness and renders `.strip().lower()`. Only a witness per half can
    catch one of them growing a guard that swallows a set value.
    """
    command = _capture_command(monkeypatch, llm_temperature=0.0)
    assert command[command.index("--llm-temperature") + 1] == "0.0"
    assert "--llm-reasoning-effort" not in command


def test_no_policy_means_no_flags(monkeypatch):
    command = _capture_command(monkeypatch)
    assert "--llm-temperature" not in command
    assert "--llm-reasoning-effort" not in command


@pytest.mark.parametrize(
    ("sampling", "expected"),
    [
        (PINNED_TEMPERATURE, {"llm_temperature": 0.0, "llm_reasoning_effort": None}),
        (PINNED_REASONING_LOW, {"llm_temperature": None, "llm_reasoning_effort": "low"}),
        (PINNED_BOTH, {"llm_temperature": 0.0, "llm_reasoning_effort": "low"}),
    ],
)
def test_endpoint_hands_the_routes_policy_to_the_launcher(monkeypatch, sampling, expected):
    """All three policy shapes, on the route rather than in the catalog.

    The posted `model` stays `openai/gpt-5.5` across the sweep and the
    `_Preflight` stub answers with the parametrized policy regardless: what
    this asserts is that the endpoint forwards **whatever the route carried**,
    not that gpt-5.5 carries any particular thing. Which row carries which
    policy is Task 1's
    `test_each_model_pins_the_documented_policy`; asserting it twice would
    mean a table change reddens two files and someone edits the nearer one.
    """
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


def test_reasoning_effort_without_a_handoff_is_refused(tmp_path):
    """The flag is worker-only, and that is enforced rather than documented.

    Without `--execution-handoff-stdin` the engine builds `make_llm_client()`
    -- the plain Anthropic SDK -- and `request_trading_decision` would hand
    `reasoning_effort` to `messages.create()`, which raises `TypeError` for an
    unknown kwarg. That would land on bar 1, after the bar fetch and the
    indicator pass have already been paid for, with a help string as the only
    thing that ever said not to.

    DEVNULL on stdin and a timeout on purpose: if the guard is ever dropped,
    this process goes on to attempt a real run, and the case should fail on
    the missing message rather than block on a read or a network call.
    """
    import os

    result = subprocess.run(
        [
            sys.executable,
            "dashboard/scripts/backtest_hourly_agent.py",
            "--use-llm",
            "--llm-reasoning-effort",
            "low",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "DATABASE_PATH": str(tmp_path / "backtest.db")},
    )

    assert result.returncode == 2, result.stdout
    assert "--llm-reasoning-effort requires --execution-handoff-stdin" in result.stderr


def test_temperature_alone_is_not_swept_up_by_that_refusal(tmp_path):
    """`--llm-temperature` is legal on every path and must stay legal.

    `request_trading_decision` has always accepted a temperature, so the flag
    means something even without a handoff. This run still stops at exit 2 --
    but at the *pre-existing* rule about who pays for the call ("explicit LLM
    execution requires a signed execution handoff", raised for
    `decision_source=llm` on the default pipeline runtime), not at the
    sampling rule. Asserting on which message comes back is what keeps the two
    distinguishable: they refuse the same command for different reasons, and
    only one of them is ours to relax.
    """
    import os

    result = subprocess.run(
        [
            sys.executable,
            "dashboard/scripts/backtest_hourly_agent.py",
            "--use-llm",
            "--llm-temperature",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "DATABASE_PATH": str(tmp_path / "backtest.db")},
    )

    assert result.returncode == 2, result.stdout
    assert "--llm-reasoning-effort" not in result.stderr
    assert "signed execution handoff" in result.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_backtest_sampling_argv.py -v`
Expected: FAIL — `TypeError: run_backtest_background() got an unexpected keyword argument 'llm_temperature'`, the endpoint test with `KeyError: 'llm_temperature'`, the `--help` assertion, and `test_reasoning_effort_without_a_handoff_is_refused` on `assert 2 == 2` passing but the stderr assertion failing (the flag does not exist yet, so argparse reports `unrecognized arguments` instead of the refusal).

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
            "Reasoning effort sent with every model call. Requires "
            "--execution-handoff-stdin: without a handoff the engine builds "
            "the plain Anthropic SDK client, which rejects the kwarg."
        ),
    )
```

Then enforce that sentence, because a help string is not a guard. Directly above `execution_handoff = None` (`:271`), add:

```python
    # A help string is not a guard. Without a handoff the engine builds
    # make_llm_client() -- the plain Anthropic SDK -- and
    # request_trading_decision would pass `reasoning_effort` straight to
    # messages.create(), which raises TypeError on an unknown kwarg. That
    # lands on bar 1, after the bar fetch and the indicator pass.
    #
    # The nearest thing to a guard today is ~57 lines below and is a different
    # rule: LLM decisions on the pipeline runtime already refuse a missing
    # handoff ("explicit LLM execution requires a signed execution handoff"),
    # which is about who pays for the call. It is skipped entirely for
    # --runtime-type ai_hedge_fund, where this flag parses and is then
    # silently ignored, and it is a billing rule somebody may well relax.
    # Neither of those should decide whether a sampling kwarg is safe to send,
    # so refuse at the flag, where the person reading the flag is.
    #
    # --llm-temperature needs no such check: request_trading_decision has
    # accepted a temperature since long before this plan, on every client.
    if args.llm_reasoning_effort and not args.execution_handoff_stdin:
        parser.error(
            "--llm-reasoning-effort requires --execution-handoff-stdin "
            "(the plain Anthropic SDK client rejects the kwarg)"
        )
```

The dashboard is unaffected: `run_backtest_background` derives both flags from `route.sampling`, and the endpoint resolves a route only inside the `decision_source == llm and runtime_type == pipeline` block that also mints the handoff (`backtests.py:3046-3143`) — so the two always travel together. The refusal bites only a hand-typed CLI invocation, which is exactly who it is for.

In the `backtester = HourlyBacktester(` call, add after the `startup_clock={...}` block Track A adds below `launched_at=args.launched_at,`:

```python
        llm_temperature=args.llm_temperature,
        llm_reasoning_effort=args.llm_reasoning_effort,
```

- [ ] **Step 4: Launcher argv**

In `run_backtest_background` — `grep -n "^def run_backtest_background(" dashboard/backend/api/routers/backtests.py` to find it, because `:1381` is the pre-#474 number and that line is `def _read_progress_file(` by the time this plan runs (Global Constraints) — add two keyword parameters after `universe_selection: Optional[Dict[str, Any]] = None,`:

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

The `if llm_sampling` is doing real work now that `ExecutionModelRoute.sampling` defaults to `None` (Task 1): the block above runs only for `decision_source == llm` on the pipeline runtime, so `llm_sampling` is `None` on every other request, and a route built by something other than `list_execution_model_routes` carries no policy either. Both cases become two absent flags and a run whose metadata records `policy: provider_default` — an accurate statement of what was sent, rather than a claim inherited from a default nobody chose.

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
- Modify: `dashboard/frontend/app.html:1257` (directly after the `backtestConfigProvenanceRow` cell — re-derived 2026-09-20 with `grep -n backtestConfigProvenanceRow dashboard/frontend/app.html`; the `:1212` an earlier draft carried was 45 lines stale). This is the one anchor in this task that **does** survive both merges: #474 changes `app.html` only in its two cache-buster `?v=` tags — `:16` and, on its branch, `:2701` — and Track A adds no markup, so the number holds (`backtestConfigProvenanceRow` is at `:1257` on the merge-base, on `main` and on #474's branch alike, verified 2026-09-20). Verify with the same grep rather than assuming the exception generalises to `app.js` below
- Modify: `dashboard/frontend/app.js` — **two merges stale; grep the `function <name>(` line, never jump to the number.** #474 moves all four of these **down** by 108, and Track A's `BACKTEST_PHASE_LABELS` + `formatBacktestPhase` block moves them down again. **Measure a sibling branch's drift against the merge-base, never against `main`** — an earlier draft compared `main`'s numbers to `feat/474-backtest-timeout-outcome`'s (9690→9636) and read the 54-line *difference* as #474 moving the anchors **up**, which is the wrong direction and the wrong magnitude. `app.js` is the one file in this set where `main` has its own commits since the merge-base, so that subtraction nets two independent drifts against each other and the sign flips. Measured 2026-09-20 with `git merge-base origin/main origin/feat/474-backtest-timeout-outcome` = `77461f3`: `formatBacktestMarketDataProvenance` sits at `:9528` there, at `:9690` on `main` (+162 of `main`'s own work above it, `+230/−41` overall) and at `:9636` on #474's branch (+108 of #474's). The spec's sequencing step 0 rebases #474 **onto** `main`, so the tree this task runs against carries both and the four anchors land near `:9798 / :9808 / :9843 / :10052` — the `main` numbers below plus 108, before Track A adds more. The anchors: new `formatBacktestSampling` beside `function formatBacktestMarketDataProvenance(provenance) {` (`:9690-9698` on `main`); `function renderBacktestRunConfig(` (`:9700`), where `    const provenanceLabel = formatBacktestMarketDataProvenance({` is computed (`:9735`) and where `    const provenanceRow = document.getElementById('backtestConfigProvenanceRow');` is toggled (`:9944-9956`)
- Modify: the five `app.js?v=` pins
- Test: `dashboard/backend/tests/test_backtest_sampling_row.py` (new)

**Interfaces:**
- Consumes: `agent_runs.metadata.llm_sampling` and `llm_max_output_tokens` from Task 3.
- Produces: `function formatBacktestSampling(sampling) -> string`; DOM ids `backtestConfigSamplingRow`, `backtestConfigSampling`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_backtest_sampling_row.py`:

```python
"""The results panel says what sampling a run asked for.

One state per *shape* of recorded request. Three of those shapes are pinned
ones, as the policy table stands: *Pinned · temperature 0*, *Pinned ·
reasoning effort low (no temperature sent)*, and *Pinned · temperature 0 ·
reasoning effort low*. Beside them sit *Provider default*, for a run that
recorded pinning nothing, and *Not recorded*, for an LLM run written before
the field existed.

This file enumerates the shapes and never asserts how many there are. The
count moves with the policy table and with `SamplingPolicy` itself -- a new
catalog row reusing an existing policy adds no shape, while a third field on
that dataclass (a `top_p`, say) adds several at once -- so a test that
counted would go red on a change that broke nothing. Not hypothetical: this
docstring's own first draft opened "Three states, never a fourth" above a
module already testing five. The spec makes the rule explicit: "the count
moves with that table, so nothing downstream should assert a number."

The row is hidden for rule-based runs, which had no sampler. The copy never
says "deterministic": providers are not, at temperature 0 least of all with
a mixture-of-experts model; what is pinned is the request.
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


def test_reasoning_only_says_the_temperature_was_not_sent():
    """The note explains the *absence*, and says nothing about the model.

    An earlier draft rendered "(this model ignores temperature)" on exactly
    this row -- which is the one policy (`PINNED_REASONING_LOW`) assigned to
    the one model that does the opposite: GPT-5.5, which returns 400 for a
    non-default temperature, which is precisely why the catalog withholds one
    (the spec's catalog table says the same). The models that genuinely
    ignore temperature while thinking are DeepSeek V4 Pro and Qwen3.7 Plus,
    and they carry `PINNED_BOTH`, so the sentence rendered on every run it
    could not be about.

    The replacement makes no claim about the provider at all: an absent
    temperature could otherwise read as a half-recorded row, and "not sent" is
    the one thing the recorded request can actually support.
    """
    assert _format("{temperature: null, reasoning_effort: 'low', policy: 'pinned_v1'}") == (
        "Pinned · reasoning effort low (no temperature sent)"
    )


def test_both():
    assert _format("{temperature: 0, reasoning_effort: 'low', policy: 'pinned_v1'}") == (
        "Pinned · temperature 0 · reasoning effort low"
    )


def test_both_carries_no_caveat_about_what_the_model_does_with_it():
    """No "(ignored while thinking)" here, even though today it would be true.

    DeepSeek V4 Pro and Qwen3.7 Plus are the only `PINNED_BOTH` rows and both
    do ignore temperature while thinking -- but this formatter reads the
    *shape* of a recorded request, not the catalog, so it would be asserting
    that of any future both-set row as well, including one for a model that
    honours it. The row's one job is to say what the run asked for. That
    "temperature 0" is a request and not an outcome is a claim about every row
    on this panel, and it is made where it can stay true: the copy rule that
    this cell never says "deterministic", the CLAUDE.md caveat, and
    `diff_backtest_runs.py`, which measures the spread instead of asserting
    it.
    """
    assert "(" not in _format(
        "{temperature: 0, reasoning_effort: 'low', policy: 'pinned_v1'}"
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


@pytest.mark.parametrize("claim", ["ignor", "reject", "honour", "honor"])
def test_copy_never_says_what_the_provider_did_with_the_value(claim):
    """The formatter reads a request shape; it cannot see a provider.

    This is the guard for the specific mistake that shipped once: a note
    reading "this model ignores temperature", rendered on the one policy given
    to the one model that refuses temperature outright. Any word in this list
    is a claim about behaviour the recorded request does not observe, and the
    row has no way to be wrong about a fact it never states.

    `fn_body` slices from the `function` keyword, so the JSDoc above the
    function is not scanned -- explanations belong there, and this list does
    not censor them.
    """
    body = fn_body("function formatBacktestSampling(")
    assert claim not in body.lower()


def test_markup_and_renderer_carry_the_row():
    assert 'id="backtestConfigSamplingRow"' in APP_HTML
    assert 'id="backtestConfigSampling"' in APP_HTML
    body = fn_body("function renderBacktestRunConfig(")
    assert "formatBacktestSampling(" in body
    assert "backtestConfigSamplingRow" in body
    # Hidden for rule-based runs: an LLM run is one that recorded a ceiling.
    assert "metadata.llm_max_output_tokens" in body
    # Written unconditionally. A write guarded on the label leaves the
    # previous run's text in the hidden cell, ready to be painted under the
    # next run's heading by anything that unhides the row first. The literal
    # is the cell's own markup default, so the two agree.
    assert (
        "setBacktestConfigText('backtestConfigSampling', samplingLabel || '—')"
        in body
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_backtest_sampling_row.py -v`
Expected: FAIL — `fn_body` cannot find `function formatBacktestSampling(`.

- [ ] **Step 3: Markup**

In `dashboard/frontend/app.html`, directly after the `backtestConfigProvenanceRow` cell (`:1257`), add:

```html
                                <div class="backtest-config-detail-cell" id="backtestConfigSamplingRow" hidden><span>Sampling</span><strong id="backtestConfigSampling">—</strong></div>
```

- [ ] **Step 4: Formatter**

In `dashboard/frontend/app.js`, directly after `function formatBacktestMarketDataProvenance(provenance) {` and its body (`:9690-9698` pre-#474; grep the `function` line rather than jumping), add:

```js
/**
 * The Sampling row's text. `null`/`undefined` is an LLM run written before
 * the engine recorded its policy, so it reads "Not recorded" -- an unrecorded
 * field is unknown, not a default. Never "deterministic": what was pinned is
 * the request, not the provider.
 *
 * Every word here is a fact about the *request*, because a recorded request
 * is all this function is given. It sees no catalog and no provider, so it
 * cannot say what a model did with a value it was sent -- and the one time
 * this row tried to, it said "this model ignores temperature" on the single
 * policy (reasoning effort, no temperature) belonging to the single model
 * that refuses a temperature outright, while the two models that really do
 * ignore one carry both values and got no note at all. The general caveat
 * that a pinned request is not a reproducible run lives where it stays true
 * for every row: the CLAUDE.md bullet, and diff_backtest_runs.py, which
 * measures the spread rather than asserting there is none.
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
    // Names why the temperature half is *absent*, which is the only thing a
    // recorded request can support. Without it an effort-only row is hard to
    // tell from a half-recorded one, and this panel already has a separate
    // state for "we do not know".
    const note = effort && !hasTemperature ? ' (no temperature sent)' : '';
    return `Pinned · ${parts.join(' · ')}${note}`;
}
```

Why the note sits on that branch and not the other one, since an earlier draft had it backwards. `PINNED_REASONING_LOW` — effort set, temperature `null` — is assigned to `openai/gpt-5.5` and nothing else, under the catalog's own comment *"OpenAI reasoning models reject a non-default temperature outright"* (the spec's catalog table row for `openai/gpt-5.5` says the same). So the temperature is missing from that row because it was deliberately withheld, and saying so is the useful thing. `PINNED_BOTH` — both set — is assigned to `deepseek/deepseek-v4-pro` and `qwen/qwen3.7-plus`, under the comment *"temperature is accepted and ignored while thinking"*: those are the two models that ignore it, and they get no note, because this function reads the shape of a request rather than the catalog and would be making the same claim about any future both-set row, including one for a model that honours the value.

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
    // Written unconditionally, unlike the three rows above it. Those guard
    // the write on their label, so hiding a row leaves the previous run's
    // text inside it: open an LLM run, then pick a rule-based one in the same
    // session, and #backtestConfigSampling still holds "Pinned · temperature
    // 0" behind `hidden`. Nothing reveals it today -- and nothing has to, for
    // it to be wrong the moment any later path unhides the row before the
    // text is set, which paints one run's sampling under another run's
    // heading. The write is a single textContent assignment; there is nothing
    // to buy by skipping it. '—' is the cell's own markup default, so a
    // hidden row holds exactly what the page shipped with.
    setBacktestConfigText('backtestConfigSampling', samplingLabel || '—');
```

Deliberately *not* applied to the three rows above: they have the same flaw, they are not this plan's to fix, and changing four call sites in a diff about sampling buries the one that matters. Worth a follow-up issue rather than a silent widening here.

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
- Produces: `compare_decisions(a: list[dict], b: list[dict]) -> dict`, `compare_equity(a: list[dict], b: list[dict]) -> dict`, and `compare_runs(run_a: str, run_b: str) -> dict` (keys: `steps_compared`, `steps_a`, `steps_b`, `divergent_steps`, `first_divergence`, `decisions_recorded`, `basis`, `equity_points_compared`, `equity_points_a`, `equity_points_b`, `divergent_equity_points`, `first_equity_divergence`, `run_a`, `run_b`, `final_equity_a`, `final_equity_b`, `final_equity_gap_pct`, `sampling_a`, `sampling_b`); CLI `python dashboard/scripts/diff_backtest_runs.py <run_a> <run_b>` printing that dict as JSON.

**Two axes, because the obvious one is not always written.** `run_agent_backtest` writes `equity_timeseries` and `trades` for every run, but calls `db.insert_decisions` only when `self.runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE` (`engine.py:1774-1775`); the only other writer in the backend is the external-agent surface (`external_run_service.py:891`) — verified 2026-09-20 with `grep -rn insert_decisions dashboard/`. So a **pipeline**-runtime backtest, which is every run Task 8 measures, has no `backtest_decisions` rows at all, and a decisions-only script would have answered `divergent_steps: 0, first_divergence: null` for all six pairs: the strongest claim it can make, produced from a table nobody wrote to. `decisions_recorded` and a `None` (never `0`) are how that reads as *unmeasured*, and the equity curve is the axis that carries the number.

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


def _point(step, equity):
    return {
        "timestamp": f"2026-09-0{1 + step // 7}T1{step % 7}:00:00",
        "equity": equity,
        "cash": equity,
        "positions_value": 0.0,
    }


def _seed(db, run_id, decisions, final_equity, equity=()):
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
    if decisions:
        db.insert_decisions(run_id, decisions)
    if equity:
        db.insert_equity_points(
            run_id, [_point(i, value) for i, value in enumerate(equity)]
        )


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


def test_compare_equity_finds_the_first_divergent_bar():
    module = _load_script()
    a = [_point(0, 100000.0), _point(1, 100500.0), _point(2, 101000.0)]
    b = [_point(0, 100000.0), _point(1, 100500.0), _point(2, 100900.0)]

    report = module.compare_equity(a, b)

    assert report["equity_points_compared"] == 3
    assert report["divergent_equity_points"] == 1
    assert report["first_equity_divergence"] == {
        "index": 2,
        "timestamp": a[2]["timestamp"],
    }


def test_compare_runs_reads_both_rows(tmp_path, monkeypatch):
    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed(
        db,
        "run_a",
        [_decision(0, []), _decision(1, [{"symbol": "AAPL", "side": "buy"}])],
        101000.0,
        equity=[100000.0, 101000.0],
    )
    _seed(
        db,
        "run_b",
        [_decision(0, []), _decision(1, [])],
        99990.0,
        equity=[100000.0, 99990.0],
    )

    report = module.compare_runs("run_a", "run_b")

    assert report["run_a"] == "run_a"
    assert report["decisions_recorded"] is True
    assert report["basis"] == "decisions"
    assert report["divergent_steps"] == 1
    assert report["first_divergence"]["step_index"] == 1
    assert report["divergent_equity_points"] == 1
    assert report["final_equity_a"] == 101000.0
    assert report["final_equity_b"] == 99990.0
    assert round(report["final_equity_gap_pct"], 4) == -1.0
    assert report["sampling_a"] == {"temperature": 0.0, "reasoning_effort": None}


def test_an_absent_decision_log_is_unmeasured_not_agreement(tmp_path, monkeypatch):
    """A pipeline-runtime backtest writes no backtest_decisions rows at all.

    `run_agent_backtest` calls `db.insert_decisions` only for the AI Hedge
    Fund runtime (`engine.py:1774-1775`); the other writer in the backend is
    the external-agent surface. Every run Task 8 measures is a pipeline run,
    so both logs come back empty -- and `divergent_steps: 0` out of an empty
    log is this script announcing that two runs agreed on every bar because it
    had no bars to look at. That number would then be copied into the Final
    verification table as the headline result of the whole track.
    """
    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed(db, "run_a", [], 101000.0, equity=[100000.0, 100500.0, 101000.0])
    _seed(db, "run_b", [], 99990.0, equity=[100000.0, 100500.0, 99990.0])

    report = module.compare_runs("run_a", "run_b")

    assert report["decisions_recorded"] is False
    assert report["basis"] == "equity"
    assert report["steps_compared"] is None
    assert report["divergent_steps"] is None
    assert report["first_divergence"] is None
    assert report["divergent_equity_points"] == 1
    assert report["first_equity_divergence"]["index"] == 2


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
Expected: FAIL — `FileNotFoundError` from `_load_script`, raised by `spec.loader.exec_module(module)` (the file does not exist). Not the `assert spec and spec.loader` above it: `spec_from_file_location` never stats the path, so for any `.py` name it hands back a spec with a `SourceFileLoader` whether or not anything is there, and the miss surfaces only when the loader reads the bytes. Verified 2026-09-21 by running `_load_script`'s own sequence against the not-yet-written path: the spec comes back with `origin` set to the missing file and a `SourceFileLoader`, and `exec_module` raises out of `get_data`. All six cases in the module fail this way, since each one opens with `_load_script()`.

- [ ] **Step 3: Write the script**

Create `dashboard/scripts/diff_backtest_runs.py`:

```python
#!/usr/bin/env python3
"""Where do two backtests of one configuration first disagree?

Prints, as JSON, the first divergent bar, how many bars diverged, and the
final-equity gap between two saved runs, plus the sampling each run recorded.
Reads ``agent_runs``, ``equity_timeseries`` and ``backtest_decisions``
through the same ``db`` the dashboard uses, so it works against local SQLite
or, with ``AGENT_RUNS_DATABASE_URL`` set, against Postgres.

    python dashboard/scripts/diff_backtest_runs.py <run_id_a> <run_id_b>

**Two axes, and ``basis`` says which one answered.** The decision log is the
sharper of the two, but it is not always written: ``run_agent_backtest``
calls ``db.insert_decisions`` only on the AI Hedge Fund runtime
(``engine.py:1774-1775``), and the external-agent surface is the only other
writer. A pipeline-runtime backtest -- the ordinary dashboard run -- has no
rows there, so the decision fields come back ``None`` rather than ``0`` and
the equity curve, which every run writes, carries the number. A zero out of
an empty table is the strongest claim this script can make, and it is the one
claim it must never make by accident.

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


def _normalised_point(point: Dict[str, Any]) -> str:
    return json.dumps([str(point.get("timestamp")), point.get("equity")])


def compare_equity(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bar-by-bar comparison of two equity curves, in stored order.

    The axis that always exists: ``run_agent_backtest`` writes
    ``equity_timeseries`` for every run, unconditionally.

    Exact comparison, no tolerance. Two runs whose decisions agreed ran the
    same arithmetic over the same bars, so any difference at all is real, and
    a tolerance would swallow exactly the smallest and earliest divergence
    this script exists to find.
    """
    divergent = 0
    first: Optional[Dict[str, Any]] = None
    for index, (x, y) in enumerate(zip(a, b)):
        if _normalised_point(x) != _normalised_point(y):
            divergent += 1
            if first is None:
                first = {"index": index, "timestamp": x.get("timestamp")}
    return {
        "equity_points_compared": min(len(a), len(b)),
        "equity_points_a": len(a),
        "equity_points_b": len(b),
        "divergent_equity_points": divergent,
        "first_equity_divergence": first,
    }


#: What the decision axis reports when there is no decision log to read.
#: ``None``, never ``0``: ``backtest_decisions`` is written only by the AI
#: Hedge Fund runtime and the external-agent surface, so an empty log on a
#: pipeline run means *unmeasured*. A ``0`` there is this script asserting
#: that two runs agreed on every bar, out of a table nobody wrote to -- and
#: that number is the headline of the table it gets pasted into.
_NO_DECISION_LOG = {
    "steps_compared": None,
    "divergent_steps": None,
    "first_divergence": None,
}


def compare_runs(run_a: str, run_b: str) -> Dict[str, Any]:
    row_a = db.get_run(run_a)
    row_b = db.get_run(run_b)
    missing = [rid for rid, row in ((run_a, row_a), (run_b, row_b)) if row is None]
    if missing:
        raise SystemExit(f"unknown run id(s): {', '.join(missing)}")
    decisions_a = db.get_decisions(run_a)
    decisions_b = db.get_decisions(run_b)
    decisions_recorded = bool(decisions_a) and bool(decisions_b)
    if decisions_recorded:
        report = compare_decisions(decisions_a, decisions_b)
    else:
        report = dict(_NO_DECISION_LOG)
        report["steps_a"] = len(decisions_a)
        report["steps_b"] = len(decisions_b)
    report.update(
        compare_equity(db.get_equity_curve(run_a), db.get_equity_curve(run_b))
    )
    final_a = row_a.get("final_equity")
    final_b = row_b.get("final_equity")
    gap_pct = None
    if final_a and final_b is not None:
        gap_pct = 100.0 * (float(final_b) - float(final_a)) / float(final_a)
    report.update(
        {
            "run_a": run_a,
            "run_b": run_b,
            "decisions_recorded": decisions_recorded,
            "basis": "decisions" if decisions_recorded else "equity",
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
**Sampling is pinned per catalog model** (`domain/model_providers/execution_catalog.py:SamplingPolicy`): Claude and Gemini get temperature 0; GPT-5.5 gets reasoning effort `low` and **no temperature** (OpenAI reasoning models reject one); DeepSeek V4 Pro and Qwen3.7 Plus get both. The route the endpoint preflights carries the policy (`CatalogModel.sampling` has **no default** — a new catalog row must state one, because a defaulted `temperature 0` on an OpenAI reasoning model 400s every call; `ExecutionModelRoute.sampling` defaults to `None`, so a hand-built route claims nothing), it rides the child's argv as `--llm-temperature` / `--llm-reasoning-effort` (each only when set; `--llm-reasoning-effort` is refused outside `--execution-handoff-stdin` at exit 2, because without a handoff the engine builds the plain Anthropic SDK client, which rejects the kwarg), and `pipeline_runner._create_pipeline_response` adds each value only when set so an unset half leaves the request byte-identical. Every model call on both branches carries it, including the truncation-recovery retry — the same request at a higher ceiling is still the same request. The engine records what it asked for as `agent_runs.metadata.llm_sampling` (`policy: pinned_v1 | provider_default`), and the results panel's **Sampling** row reads *Pinned · …*, *Provider default*, or *Not recorded* for rows written before the field existed. ⚠ **Pinned is not deterministic** — providers are not, at temperature 0 least of all with a mixture-of-experts model, and each bar's prompt embeds the previous bar's answer — so the claim on screen is the request, and the spread is measured with `dashboard/scripts/diff_backtest_runs.py <run_a> <run_b>` (first divergent bar, divergent bar count, final-equity gap). It reports a `basis`: a pipeline-runtime backtest writes no `backtest_decisions` rows (`engine.py` writes them for the AI Hedge Fund runtime only), so the decision fields come back `None` rather than `0` and the equity curve carries the number. The measured before/after is in `docs/superpowers/plans/2026-09-20-backtest-pinned-sampling.md`.
```

Commit:

```bash
git add CLAUDE.md
git commit -m "docs: describe the pinned sampling policy

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 2: Measure before and after, locally**

"Before" runs pin nothing. "After" runs pin `temperature=0`, which is the half of the policy this path can carry — `request_trading_decision` has always accepted a temperature, while `reasoning_effort` is worker-only and refused outside a handoff (Task 4, Step 3). The client is whatever `make_llm_client()` resolves (CommonStack, or native Anthropic when only `ANTHROPIC_API_KEY` is set), so the model is Claude; the point is the spread, not the model.

**Not through `backtest_hourly_agent.py`, and an earlier draft of this plan was wrong about that.** Verified 2026-09-20 by running it: `python3 dashboard/scripts/backtest_hourly_agent.py --start 2026-09-08 --end 2026-09-16 --use-llm --run-id probe_1` exits **2** with `error: explicit LLM execution requires a signed execution handoff`. The launcher refuses `decision_source=llm` on the pipeline runtime — its default — unless `--execution-handoff-stdin` is supplied, and minting a valid handoff locally needs an account id, a registered provider row and a stored credential. That refusal is a rule about *who pays for the call*. The engine has no equivalent: with `execution_client=None` it builds `make_llm_client()` and takes the single-prompt branch, which is the same `_request_trading_decision` path the leaderboard's `llm_agent` runs on and the one `--llm-temperature` exists to feed. So drive the engine and skip the launcher. The old shell driver could not have produced a single row of the table below.

Write `<scratchpad>/rerun.py`:

```python
#!/usr/bin/env python3
"""One LLM backtest of one fixed configuration, run in-process.

    DATABASE_PATH=<scratchpad>/rerun.db PYTHONPATH=. python3 rerun.py before 1

DATABASE_PATH is not optional here. Unset, `db` resolves to the committed
seed backtest.db and six runs write themselves into a tracked file.
"""
import sys

from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.infrastructure.market_data.profiles import (
    LLM_DECISION_SOURCE,
)


def main(argv):
    tag, number = argv[1], argv[2]
    if tag not in {"before", "after"}:
        raise SystemExit("usage: rerun.py before|after <n>")
    backtester = HourlyBacktester(
        "2026-09-08",
        "2026-09-16",
        f"rerun-{tag}",
        use_llm=True,
        decision_source=LLM_DECISION_SOURCE,
        live_run_id=f"rerun_{tag}_{number}",
        # The only difference between the two arms.
        llm_temperature=0.0 if tag == "after" else None,
    )
    # Fail loudly here or this script measures nothing. With no
    # execution_client and the default ALPACA source, `self.strict_llm` is
    # False (engine.py:311-317), so BOTH LLM-unavailable paths degrade
    # instead of raising: no SDK sets decision_source to rule_based (:348,
    # :363), and a make_llm_client() that returns None prints a warning and
    # sets use_llm = False (:375-386). A degraded run still completes, still
    # writes a row, and still exits 0 -- and six rule-based runs of one
    # configuration are bit-identical, so all six diff pairs come back with
    # zero divergent points and a 0.00% gap. That is exactly the shape of a
    # triumphant result for pinned sampling. Same defect class the `basis`
    # field guards against one task earlier: a zero out of an axis nobody
    # wrote to.
    if not backtester.use_llm or backtester.decision_source != LLM_DECISION_SOURCE:
        raise SystemExit(
            "rerun.py: no LLM client resolved -- this run would be rule-based, "
            "and six rule-based runs agree perfectly for the wrong reason. Set "
            "COMMONSTACK_API_KEY / OPENROUTER_API_KEY / ANTHROPIC_API_KEY."
        )
    # Printed by the script rather than read off a banner: the engine's
    # `✅ LLM initialized (model=...)` line is emitted only on the branch that
    # succeeded, so its absence is the signal -- and an absence is the one
    # thing an operator scrolling a log does not notice. The client class
    # separates CommonStack from native Anthropic, which the model slug alone
    # does not.
    print(f"client={type(backtester.llm_client).__name__} model={backtester.model}")
    backtester.load_data()
    backtester.calculate_indicators()
    run_id, _curve = backtester.run_agent_backtest()
    print(run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

Run it six times:

```bash
export DATABASE_PATH="<scratchpad>/rerun.db"
set -a; source dashboard/.env; set +a
for tag in before after; do
  for n in 1 2 3; do PYTHONPATH=. python3 <scratchpad>/rerun.py "$tag" "$n"; done
done
```

Each run prints one `client=… model=…` line before it fetches a bar, and all six must be identical — a run that resolved a different client drew from a different sampler, and the spread it contributes is not the spread this table measures. If any run instead exits non-zero with `no LLM client resolved`, the environment is wrong; **do not** drop the guard and record the run anyway. Its whole purpose is that a degraded run is silent: it completes, writes a row, exits 0, and six rule-based runs of one configuration are bit-identical, so all six pairs below would read `0` divergent points and a `0.00%` gap — the shape of a perfect result for pinned sampling, produced by a path that never called a model.

Then, with the same `DATABASE_PATH` exported:

```bash
for p in "before_1 before_2" "before_1 before_3" "before_2 before_3" "after_1 after_2" "after_1 after_3" "after_2 after_3"; do
  set -- $p
  python3 dashboard/scripts/diff_backtest_runs.py "rerun_$1" "rerun_$2" | python3 -c '
import json, sys
r = json.load(sys.stdin)
on_decisions = r["decisions_recorded"]
first = (r["first_divergence"] or r["first_equity_divergence"] or {})
print(r["run_a"], r["run_b"],
      "basis", r["basis"],
      "first", first.get("step_index", first.get("index")),
      "divergent", r["divergent_steps"] if on_decisions else r["divergent_equity_points"],
      "/", r["steps_compared"] if on_decisions else r["equity_points_compared"],
      "gap%", r["final_equity_gap_pct"])'
done
```

Every pair should print `basis equity`: these are pipeline-runtime runs and nothing writes their decision log (see Task 7's **Interfaces**). A pair printing `basis decisions` means the runtime changed under you — read the number, then work out why before recording it.

Record the six lines in the **Final verification** table. Six LLM runs of 49 bars at one step each is roughly 300 model calls; that is the cost of the number.

- [ ] **Step 3: Full suite and hygiene**

Run: `pytest dashboard/backend/tests/ -q` → green.
Run: `git status --short dashboard/storage/data/backtest.db` → empty. Step 2 exports `DATABASE_PATH` at a scratch file so the six runs cannot touch the committed seed DB, but *importing* any store module runs `CREATE TABLE IF NOT EXISTS` against whatever `DATABASE_PATH` is set at the time, so check anyway; if it changed, `git checkout -- dashboard/storage/data/backtest.db`.
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

| pair | basis | first divergent bar | divergent bars / compared | final-equity gap % |
|---|---|---|---|---|
| before 1 vs 2 | | | | |
| before 1 vs 3 | | | | |
| before 2 vs 3 | | | | |
| after 1 vs 2 | | | | |
| after 1 vs 3 | | | | |
| after 2 vs 3 | | | | |

**A blank cell is not a result**, and neither is a `0` in a column whose `basis` says the axis was not written. Expect `basis: equity` on all six rows — these are pipeline-runtime runs and nothing writes their `backtest_decisions` (Task 7, **Interfaces**), which is why `divergent_steps` comes back `None` rather than `0` and why the bar count in that column is the equity curve's. If a row shows `basis: decisions`, say in the table which runtime produced it. The rows can also be trusted to be LLM rows at all only because `rerun.py` refuses to start when no client resolved (Step 2): without that guard, six rule-based runs of one configuration are bit-identical, and this table would fill with six zeroes and a `0.00%` gap — the best-looking result the page can print, from a path that never called a model.

Model used for the reruns: _(whatever `make_llm_client()` resolved — copy the `client=… model=…` line `rerun.py` prints before each of the six, and say so if they are not all identical)_. Take it from there rather than from the run summary: `run_agent_backtest` does print the slug (`engine.py:1781`), but from one `print` whose two branches differ only by a `✅ LLM enabled` / `❌ fallback` marker, and reading past that marker is precisely how a rule-based run gets recorded as an LLM one. Bars per run: 49 (7 weekdays × 7 hourly bars). Driver: `<scratchpad>/rerun.py` in-process, **not** `backtest_hourly_agent.py`, which refuses an LLM run without a signed handoff (Task 8, Step 2).

Full suite: `pytest dashboard/backend/tests/ -q` → _result_.
Seed DB: clean.
Cache-busters: one number per asset, **five** files (`dashboard/frontend/app.html` plus `test_frontend_fast_boot.py`, `test_backtest_comparison_frontend.py`, `test_analytics_frontend.py`, `test_admin_analytics_frontend.py`) — re-derived 2026-09-20 with `grep -rln "app.js?v=" dashboard/frontend/app.html dashboard/backend/tests/*.py`, unchanged by this plan. The count is per-document and moves when a test file starts or stops loading `app.js`; grep it, never carry it forward.

## Out of scope (from the spec)

Track C (per-bar deadline, failover timeout, snapshot rebuild off the hot path), Track D (re-run and compare in the UI), the cadence and result-contract doors, changing the modal's default model. If the after-runs still show a completed-vs-failed split on the DeepSeek route in prod, that last item becomes a product decision to raise separately.

## User-facing docs to check after this ships

`docs/source/lab/operating_modes.rst` and `docs/source/lab/key_features.rst`: neither currently makes a reproducibility claim. Once the Sampling row is live, re-read both and file a docs follow-up if a sentence should now name it.
