# Pipeline LLM Backtest Window Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refuse a `runtime_type="pipeline"` + `decision_source="llm"` backtest whose window and pipeline width cannot finish inside the fixed 3600-second parent budget, and lower `MAX_BACKTEST_DAYS` from 31 to 14.

**Architecture:** Mirror `_enforce_ai_hedge_fund_window` (issue #308) on the pipeline path: a pre-flight 422 in the `/backtest/run` handler, placed after `_validate_backtest_params` and before the rate limiter, the slot ledger and the billing/credential preflight. The parent budget itself does **not** change — this is a refusal, not a resize. The refusal is computed from an **estimated** LLM-call count (`bars × pipeline decision steps + trading days × post-trade steps`) against an **estimated** per-call duration, because unlike the hosted path there is no enforced per-bar ceiling to invert. The 3600-second timeout remains the real backstop.

**Tech Stack:** Python 3, FastAPI router (`dashboard/backend/api/routers/backtests.py`), pytest, vanilla JavaScript (`dashboard/frontend/app.js`, no build step), static HTML (`dashboard/frontend/app.html`), Node-based frontend source harnesses.

**Spec:** No separate design doc — this plan implements **checkbox 1 of GitHub issue #474** ("Size the pipeline budget from the window, or refuse an uncompletable one the way `_enforce_ai_hedge_fund_window` does (422 naming the bound *and* the request)"), resolved during brainstorming to the *refuse* branch. Read issue #474 alongside this plan. Items 2-5 of #474 are **out of scope** and are planned separately.

---

## Global Constraints

- **Do not change `PIPELINE_SUBPROCESS_TIMEOUT_SECONDS`** (3600). The 60-minute budget is a four-way contract — `backtests.py:1706`, `app.js:20` `BACKTEST_POLL_MAX_SECONDS`, `app.html:1288` `"limit: 60 minutes"`, `app.js:8515` `"Timed out after 60 minutes."` — pinned by `test_app_copy_register.py:200,204`. Moving the server half alone ships a strict regression (cards self-clear while the server still runs and still bills). Raising it is #474 item 5's territory, deliberately deferred.
- **Do not touch `app.js`'s `BACKTEST_POLL_MAX_SECONDS`, the two "60 minutes" copy strings, or `app.html:1288`.** Nothing in this plan moves the timeout.
- **Do not change the AI Hedge Fund path.** `_enforce_ai_hedge_fund_window`, `MAX_AI_HEDGE_FUND_TRADING_DAYS`, `_ai_hedge_fund_trading_days_ceiling` and `_backtest_subprocess_timeout`'s hosted branch stay exactly as they are. Issue #474 declares hosted out of scope.
- **Do not change `MAX_SUBPROCESS_TIMEOUT_SECONDS` (14400) or `SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS` (600).**
- **Do not make the backtest faster.** LLM call count, hourly decision cadence, pipeline semantics and retry policy are unchanged.
- **Do not raise or lower `MAX_ACTIVE_DASHBOARD_BACKTESTS`.** That is issue #473, a spend decision.
- **Every operator-set integer is parsed defensively at module scope with a log-and-fallback, never a bare `int()`.** CLAUDE.md records that a bare `int()` at module scope *in this exact module* once killed app boot on a typo'd env value.
- **The refusal must name both levers.** The bound is a product of window length × pipeline width, so a message naming only the window is a dead end for a user whose real problem is a wide pipeline.
- Preserve sanitized errors, platform-credit settlement, backtest-slot release and browser running-entry cleanup.
- No real API keys, database credentials, `.superpowers/` or `work/` in changes.

## Why this is an estimate and not an inverted ceiling

Read this before writing the comment in Task 4; it is the single thing most likely to be
"corrected" wrongly later.

`#308` could size the hosted budget because the hosted path **enforces** a per-step timeout:
`_backtest_subprocess_timeout` multiplies `resolve_step_timeout_seconds()`
(`AI_HEDGE_FUND_TIMEOUT_SECONDS`, default 300s) by `_estimated_decision_days`, and
`_ai_hedge_fund_trading_days_ceiling()` inverts that same formula — which is why its comment
can claim the bound and the budget "cannot disagree".

The pipeline path has no such ceiling to invert. `run_pipeline_decision`
(`infrastructure/llm/pipeline_runner.py:693`) issues **one LLM call per pipeline decision
step, per bar**, each with up to one retry, and the only wall-clock bound anywhere is the
httpx read timeout `build_safe_http_client(timeout_seconds=60.0)`
(`infrastructure/llm/execution/adapters/base.py:218`), which every provider adapter takes as
its default. A refusal built on that worst case — `3000 / (60 × 4)` ≈ 12 bars, under 2
trading days — would refuse essentially every run.

So `PIPELINE_SECONDS_PER_LLM_CALL` is a **calibrated estimate of typical per-call latency**,
not a guarantee. A slow reasoning model can still exhaust the 3600-second budget and be
killed; that failure is unchanged by this plan and is #474 item 2's subject. What this
preflight removes is the *obviously* uncompletable case, cheaply and before any spend.

**Two asymmetries that must be stated in code comments, because both invert the intuition
inherited from `#308`:**

1. `#308`'s comment says over-provisioning is the safe direction — true for a **budget**.
   Here we compute a **refusal**, so the safe direction is the opposite: *under*-estimating
   the work lets a marginal run through (and the timeout catches it), while *over*-estimating
   refuses a run that would have succeeded. A false refusal on the declared onboarding task
   is the worse failure.
2. `_estimated_decision_days` is deliberately an **upper** bound (weekdays, ignoring
   holidays) because #308 used it to size a budget. This plan reuses it unchanged — accepting
   that it errs slightly toward refusing — because a second, disagreeing day-count in the same
   module is worse than a known-conservative shared one.

## Arithmetic this plan is built on

Verified against the code during brainstorming; reproduce these numbers if you change a constant.

```
usable budget = PIPELINE_SUBPROCESS_TIMEOUT_SECONDS - SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS
              = 3600 - 600 = 3000 s
max calls     = 3000 // PIPELINE_SECONDS_PER_LLM_CALL = 3000 // 15 = 200

calls ≈ bars × max(1, len(decision_steps)) + trading_days × len(post_trade_steps)
bars   = _estimated_decision_days(start, end) × PIPELINE_DECISION_BARS_PER_TRADING_DAY
```

| Window | Weekdays | Bars | 1-step | 4-step ("Retrieval → Signal → Execution → Risk", `app.html:1462`) |
|---|---|---|---|---|
| Modal default `2026-04-15` → `2026-04-23` | 7 | 49 | 49 ✅ | 196 ✅ (98% of 200) |
| New max, 14 calendar days | 10 | 70 | 70 ✅ | 280 ❌ refused |

**Record this finding, do not hide it:** at 15 s/call the *shipped default window running the
advertised 4-module pipeline* sits at 98% of budget. It passes by 4 calls. That is not an
argument against the constant — it is evidence that the fixed 3600-second budget is undersized
for the product being advertised, which is why the operator override exists and why #474
item 5 remains open.

## File Map

- **Modify `dashboard/backend/api/routers/backtests.py`**
  - `:2286` — `MAX_BACKTEST_DAYS` 31 → 14.
  - New import of `split_pipeline` from `infrastructure.llm.pipeline_runner` (verified importable, no circular-import risk: the router already imports from `infrastructure.llm.execution.*`).
  - New module-scope block after `_enforce_ai_hedge_fund_window` (ends `:1907`): `PIPELINE_DECISION_BARS_PER_TRADING_DAY`, `_DEFAULT_PIPELINE_SECONDS_PER_LLM_CALL`, `_pipeline_seconds_per_llm_call()`, `PIPELINE_SECONDS_PER_LLM_CALL`, `_max_pipeline_llm_calls()`, `_estimated_pipeline_llm_calls()`, `_enforce_pipeline_llm_window()`.
  - `:2750` — call `_enforce_pipeline_llm_window(...)` immediately after `_validate_backtest_params`.
- **Modify `dashboard/frontend/app.js:9869`** — the mirrored `const MAX_BACKTEST_DAYS = 31;` → `14`.
- **Modify `dashboard/frontend/app.html:334`** — `"up to 31 days"` → `"up to 14 days"`.
- **Modify `dashboard/backend/tests/test_backtests_router.py`** — new tests for the constant, the call estimator, the env override and the guard.
- **Modify `CLAUDE.md`** — a `PIPELINE_SECONDS_PER_LLM_CALL` bullet in the env section.

`test_onboarding_flow_ux.py` needs **no edit**: `test_the_client_backtest_span_check_mirrors_the_server_constant` (`:233`) and `test_the_period_helper_states_the_limit_rather_than_inviting_the_error` (`:241`) both read the Python constant and assert the JS and HTML agree, so they fail automatically when only the backend moves and pass once all three are aligned. That is the failing test for Task 1.

## Verified facts (do not re-derive)

- **The leaderboard is unaffected by `MAX_BACKTEST_DAYS`.** `dashboard/config/leaderboard.json` runs `2026-04-15` → `2026-05-15` (30 days), but `_validate_backtest_params` is called only from the `/backtest/run` handler (`backtests.py:2724` and `:2750`); the board runs through `deploy_model_run` in `domain/leaderboard/service.py` and never touches the router validator.
- **A precedent exists.** `docs/superpowers/plans/2026-07-21-demo1-discord-talk.md` Task 2 proposed `MAX_BACKTEST_DAYS = 14` for a related reason (the Discord bot stops polling at ~11 minutes). It was never landed; the constant is still 31.
- **Post-trade steps run once per trading day, not per bar.** `split_pipeline` (`pipeline_runner.py:89`) separates them on `presetKey == "post_trade_analysis"`; `run_post_trade_analysis` is called from `engine.py:1250` at a day boundary. Counting them per bar would overstate a post-trade pipeline by ~7×.
- **All market profiles use a 60m decision timeframe** (`infrastructure/market_data/profiles.py`), and `_validate_backtest_params`' caller pins `timeframe` to the profile (`backtests.py:2425`), so decision cadence is not caller-controlled.
- **`_market_hours_only`** (`engine.py:1179`) keeps ~7 hourly bars/day for US (`local.hour > 9 and < 16`, plus the 9:30 and 16:00 edges) and fewer for CN (`profile.market == "CN"`, two sessions). 7 is the figure this plan uses.
- **`pipeline` is already `None` on a rule-based run** (dropped at `:2714`/`:2733`), so the guard's `decision_source == "llm"` condition is belt-and-braces, not load-bearing.

---

## Task 1: Lower the backtest window cap to 14 days

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py:2286`
- Modify: `dashboard/frontend/app.js:9869`
- Modify: `dashboard/frontend/app.html:334`
- Test: `dashboard/backend/tests/test_backtests_router.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `dashboard.backend.api.routers.backtests.MAX_BACKTEST_DAYS == 14`. Task 4's message arithmetic and Task 5's route test assume this value.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_backtests_router.py`:

```python
def test_backtest_window_cap_is_two_weeks():
    """A dashboard backtest is capped at a fortnight (issue #474 item 1).

    31 days was set when the parent budget was the only bound and nothing
    counted LLM calls. A 31-day window is ~22 weekdays x ~7 hourly bars = ~154
    decision bars, and a multi-step pipeline multiplies that by its step count
    -- far past what the fixed 3600s budget can finish. Two weeks matches the
    fortnight the Live Trading Leaderboard's seasons already use, so the
    product has one window vocabulary rather than two.
    """
    assert bt.MAX_BACKTEST_DAYS == 14
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest dashboard/backend/tests/test_backtests_router.py::test_backtest_window_cap_is_two_weeks -v
```

Expected: FAIL — `assert 31 == 14`.

- [ ] **Step 3: Change the backend constant**

In `dashboard/backend/api/routers/backtests.py`, replace the `MAX_BACKTEST_DAYS` line (`:2286`):

```python
# Two weeks, matching the fortnight the Live Trading Leaderboard's seasons use
# so the product has one window vocabulary. It was 31, set when the fixed
# parent budget was the only bound: a 31-day window is ~22 weekdays x ~7 hourly
# bars = ~154 decision bars, and a pipeline multiplies that by its step count,
# so the UI permitted a window roughly 3x the modal default against a constant
# budget with no preflight (issue #474 item 1). This bound is the calendar half
# of the answer; _enforce_pipeline_llm_window below is the work-volume half,
# because window length alone does not say how much a run costs.
MAX_BACKTEST_DAYS = 14
```

- [ ] **Step 4: Run the backend test to verify it passes, and confirm the mirror tests now fail**

```bash
pytest dashboard/backend/tests/test_backtests_router.py::test_backtest_window_cap_is_two_weeks -v
pytest dashboard/backend/tests/test_onboarding_flow_ux.py -v -k "span_check or period_helper"
```

Expected: the first PASSES. The second FAILS both cases — `test_the_client_backtest_span_check_mirrors_the_server_constant` (`assert 31 == 14`) and `test_the_period_helper_states_the_limit_rather_than_inviting_the_error` (`"up to 14 days" in APP_HTML` is False). **This is the point** — those tests exist to catch exactly this drift.

- [ ] **Step 5: Update the JavaScript mirror**

In `dashboard/frontend/app.js`, inside `async function runBacktest`, change line 9869:

```javascript
    const MAX_BACKTEST_DAYS = 14;
```

Leave the comment above it unchanged — it already explains why the mirror exists.

- [ ] **Step 6: Update the modal helper copy**

In `dashboard/frontend/app.html`, line 334:

```html
                    <p class="control-helper">Pre-filled with a recent window that has full market data. You can pick any window up to 14 days.</p>
```

Do **not** change the pre-filled `value="2026-04-15"` / `2026-04-23` dates below it — that window is 8 calendar days and stays legal.

- [ ] **Step 7: Run the full mirror suite to verify it passes**

```bash
pytest dashboard/backend/tests/test_onboarding_flow_ux.py -v
pytest dashboard/backend/tests/test_app_copy_register.py -v
```

Expected: PASS. `test_app_copy_register` must stay green — it asserts the two "60 minutes" strings, which this plan does not touch.

- [ ] **Step 8: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py dashboard/frontend/app.js dashboard/frontend/app.html dashboard/backend/tests/test_backtests_router.py
git commit -m "fix: cap the backtest window at two weeks"
```

---

## Task 2: Estimate a pipeline run's LLM call count

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py` (new import; new block after `_enforce_ai_hedge_fund_window`, which ends at `:1907`)
- Test: `dashboard/backend/tests/test_backtests_router.py`

**Interfaces:**
- Consumes: `_estimated_decision_days(start_date: str, end_date: str) -> int` (existing, `:1713`); `split_pipeline(pipeline) -> tuple[list[dict], list[dict]]` from `dashboard.backend.infrastructure.llm.pipeline_runner`.
- Produces:
  - `PIPELINE_DECISION_BARS_PER_TRADING_DAY: int` (7)
  - `_estimated_pipeline_llm_calls(start_date: str, end_date: str, pipeline: Optional[List[Dict[str, Any]]]) -> int`

  Task 4 calls `_estimated_pipeline_llm_calls` and reads `PIPELINE_DECISION_BARS_PER_TRADING_DAY` for its message.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_backtests_router.py`:

```python
def test_pipeline_call_estimate_counts_one_call_per_bar_with_no_pipeline():
    """A single-prompt LLM run calls the model once per hourly bar.

    ``max(1, len(decision_steps))`` is what makes the no-pipeline path cost
    anything at all -- ``split_pipeline(None)`` returns two empty lists, and a
    bare multiplication would estimate zero calls for the most common run
    there is.
    """
    # 2026-04-15 (Wed) -> 2026-04-23 (Thu) is 7 weekdays.
    assert bt._estimated_decision_days("2026-04-15", "2026-04-23") == 7
    assert (
        bt._estimated_pipeline_llm_calls("2026-04-15", "2026-04-23", None)
        == 7 * bt.PIPELINE_DECISION_BARS_PER_TRADING_DAY
    )


def test_pipeline_call_estimate_multiplies_by_decision_steps():
    """Each decision step is its own model call, per bar.

    ``run_pipeline_decision`` loops the decision steps inside the hourly loop,
    so a four-module pipeline costs 4x a single prompt over the same window --
    which is the whole reason window length alone cannot bound a run.
    """
    pipeline = [{"label": f"Step {i}"} for i in range(4)]
    assert (
        bt._estimated_pipeline_llm_calls("2026-04-15", "2026-04-23", pipeline)
        == 7 * bt.PIPELINE_DECISION_BARS_PER_TRADING_DAY * 4
    )


def test_pipeline_call_estimate_charges_post_trade_steps_per_day_not_per_bar():
    """Post-trade steps fire at a day boundary, not every bar.

    ``engine.py`` calls ``run_post_trade_analysis`` once per trading day, so
    counting them per bar would overstate a post-trade pipeline by ~7x and
    refuse windows that finish comfortably.
    """
    days = 7
    bars = days * bt.PIPELINE_DECISION_BARS_PER_TRADING_DAY
    decision_only = [{"label": "Signal"}]
    with_post_trade = decision_only + [{"presetKey": "post_trade_analysis"}]

    assert (
        bt._estimated_pipeline_llm_calls("2026-04-15", "2026-04-23", decision_only)
        == bars
    )
    assert (
        bt._estimated_pipeline_llm_calls("2026-04-15", "2026-04-23", with_post_trade)
        == bars + days
    )


def test_pipeline_call_estimate_is_zero_for_an_unusable_range():
    """Unparseable or inverted dates estimate zero rather than raising.

    ``_validate_backtest_params`` already 422s those before the guard runs, so
    this only has to avoid becoming a second, competing date validator with its
    own error surface.
    """
    assert bt._estimated_pipeline_llm_calls("not-a-date", "also-bad", None) == 0
    assert bt._estimated_pipeline_llm_calls("2026-04-23", "2026-04-15", None) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_backtests_router.py -v -k "pipeline_call_estimate"
```

Expected: FAIL — `AttributeError: module ... has no attribute '_estimated_pipeline_llm_calls'`.

- [ ] **Step 3: Add the import**

In `dashboard/backend/api/routers/backtests.py`, beside the other `infrastructure.llm` imports (near `:74-79`, the `execution.*` group):

```python
from dashboard.backend.infrastructure.llm.pipeline_runner import split_pipeline
```

- [ ] **Step 4: Write the estimator**

Insert immediately after `_enforce_ai_hedge_fund_window` ends (`:1907`), before the `# Running the child` banner:

```python
# ============================================================================
# Pipeline LLM window preflight (issue #474, item 1)
# ============================================================================
#
# The hosted guard above can invert its own budget because the hosted path
# ENFORCES a per-step timeout: _backtest_subprocess_timeout multiplies
# resolve_step_timeout_seconds() by a day count, and
# _ai_hedge_fund_trading_days_ceiling() inverts that same product, which is why
# its docstring can say the bound and the budget cannot disagree.
#
# Nothing enforces a per-bar ceiling on the pipeline path. run_pipeline_decision
# issues one model call per pipeline decision step per hourly bar, each with up
# to one retry, and the only wall-clock bound anywhere is the 60s httpx read
# timeout every provider adapter takes by default. A refusal built on that worst
# case -- 3000 / (60 * 4) = about 12 bars, under two trading days -- would refuse
# essentially every run. So the number below is a CALIBRATED ESTIMATE of typical
# per-call latency, not an inverted ceiling, and the 3600s timeout stays the real
# backstop: a slow reasoning model can still exhaust the budget and be killed.
# What this preflight removes is the obviously uncompletable case, before any
# spend and while the user still has the dates on screen.
#
# ⚠ The safe direction is INVERTED relative to _backtest_subprocess_timeout.
# That function over-provisions on purpose, because it sizes a BUDGET. This one
# computes a REFUSAL: under-estimating the work lets a marginal run through and
# the timeout catches it, while over-estimating refuses a run that would have
# succeeded -- on the declared onboarding task, where a new user gets one pass.
# Do not "tighten" these estimates toward the worst case.

# Hourly decision bars in one trading day. US is the widest session
# (_market_hours_only keeps roughly 09:30 through 16:00); CN runs two shorter
# sessions and costs less. One number for both, taken from the wider market,
# because a per-profile table here would be a second place for the bar cadence
# to be wrong and every profile already pins a 60m decision timeframe.
PIPELINE_DECISION_BARS_PER_TRADING_DAY = 7


def _estimated_pipeline_llm_calls(
    start_date: str, end_date: str, pipeline: Optional[List[Dict[str, Any]]]
) -> int:
    """Upper-bound the model calls a pipeline LLM backtest will make.

    Decision steps run inside the hourly loop; post-trade steps run once at a
    trading-day boundary (``run_post_trade_analysis``). Counting post-trade
    steps per bar would overstate such a pipeline by about 7x and refuse
    windows that finish comfortably.

    ``max(1, ...)`` is load-bearing: ``split_pipeline(None)`` returns two empty
    lists, and the single-prompt path -- the most common run there is -- would
    otherwise estimate zero calls and skip the guard entirely.
    """
    trading_days = _estimated_decision_days(start_date, end_date)
    if trading_days <= 0:
        # _validate_backtest_params already answered a bad range with a 422
        # before this runs. Returning 0 keeps this from becoming a second,
        # competing date validator with its own error surface.
        return 0
    decision_steps, post_trade_steps = split_pipeline(pipeline)
    bars = trading_days * PIPELINE_DECISION_BARS_PER_TRADING_DAY
    return bars * max(1, len(decision_steps)) + trading_days * len(post_trade_steps)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_backtests_router.py -v -k "pipeline_call_estimate"
```

Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtests_router.py
git commit -m "feat: estimate a pipeline backtest's model call count"
```

---

## Task 3: Make the per-call latency estimate operator-tunable

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py` (append to the Task 2 block)
- Test: `dashboard/backend/tests/test_backtests_router.py`
- Modify: `dashboard/backend/tests/conftest.py`

**Interfaces:**
- Consumes: `PIPELINE_SUBPROCESS_TIMEOUT_SECONDS` (`:1706`), `SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS` (`:1708`).
- Produces:
  - `_DEFAULT_PIPELINE_SECONDS_PER_LLM_CALL: int` (15)
  - `_MAX_PIPELINE_SECONDS_PER_LLM_CALL: int` (300)
  - `_pipeline_seconds_per_llm_call() -> int`
  - `PIPELINE_SECONDS_PER_LLM_CALL: int` — module-scope result of the above
  - `_max_pipeline_llm_calls() -> int`

  Task 4 calls `_max_pipeline_llm_calls()` and reads `PIPELINE_SECONDS_PER_LLM_CALL`.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_backtests_router.py`:

```python
def test_pipeline_seconds_per_call_default_and_derived_call_budget():
    """15s/call over the usable budget leaves 200 calls.

    Usable is the parent budget minus the overhead constant, because data load,
    baseline generation and persistence sit outside the decision loop and their
    share is already reserved.
    """
    assert bt._DEFAULT_PIPELINE_SECONDS_PER_LLM_CALL == 15
    assert bt.PIPELINE_SECONDS_PER_LLM_CALL == 15

    usable = (
        bt.PIPELINE_SUBPROCESS_TIMEOUT_SECONDS
        - bt.SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS
    )
    assert usable == 3000
    assert bt._max_pipeline_llm_calls() == 200


@pytest.mark.parametrize("raw", ["", "   ", "abc", "12.5", "0", "-4", "301"])
def test_pipeline_seconds_per_call_falls_back_on_junk(monkeypatch, raw, capsys):
    """A mistyped env value logs and falls back; it never kills app boot.

    CLAUDE.md records that a bare int() at module scope in this very module once
    took the whole app down on a typo. 0 is refused specifically because it
    would make the call budget a ZeroDivisionError rather than a bound.
    """
    monkeypatch.setenv("PIPELINE_SECONDS_PER_LLM_CALL", raw)
    assert (
        bt._pipeline_seconds_per_llm_call()
        == bt._DEFAULT_PIPELINE_SECONDS_PER_LLM_CALL
    )


def test_pipeline_seconds_per_call_honours_a_valid_override(monkeypatch):
    """An operator who has measured real latency tunes this without a deploy."""
    monkeypatch.setenv("PIPELINE_SECONDS_PER_LLM_CALL", "30")
    assert bt._pipeline_seconds_per_llm_call() == 30

    monkeypatch.setenv("PIPELINE_SECONDS_PER_LLM_CALL", "1")
    assert bt._pipeline_seconds_per_llm_call() == 1

    monkeypatch.setenv("PIPELINE_SECONDS_PER_LLM_CALL", "300")
    assert bt._pipeline_seconds_per_llm_call() == 300
```

Ensure `import pytest` is present at the top of the file (it already is; do not add a duplicate).

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_backtests_router.py -v -k "pipeline_seconds_per_call"
```

Expected: FAIL — `AttributeError: module ... has no attribute '_DEFAULT_PIPELINE_SECONDS_PER_LLM_CALL'`.

- [ ] **Step 3: Write the config reader**

Append to the block added in Task 2, directly after `_estimated_pipeline_llm_calls`:

```python
# Typical wall-clock for one model call on this deployment. NOT enforced
# anywhere -- see the banner above. 15s is calibrated for a model that spends a
# little time reasoning; a fast completion model finishes in a few seconds and a
# slow reasoning model can exceed the 60s httpx read timeout, which is why this
# is an operator dial rather than a literal.
#
# Consequence worth knowing before changing it: at 15s the shipped modal default
# window (7 weekdays, 49 bars) running the four-module pipeline advertised at
# app.html:1462 costs 196 of the 200 available calls -- it passes by four. That
# is not an argument for a smaller number here; it is evidence that the fixed
# 3600s budget is undersized for the advertised product (issue #474, item 5).
_DEFAULT_PIPELINE_SECONDS_PER_LLM_CALL = 15
# A value this high already refuses almost every window (200 -> 10 calls). Past
# it the setting stops expressing latency and starts silently disabling the
# lane, which deserves its own explicit switch rather than a large number here.
_MAX_PIPELINE_SECONDS_PER_LLM_CALL = 300


def _pipeline_seconds_per_llm_call() -> int:
    """Seconds one model call is assumed to take on this deployment.

    Parsed defensively for the reason CLAUDE.md records about this very module:
    an operator-set integer read with a bare ``int()`` at module scope once
    killed app boot on a typo. Junk, zero, negative and out-of-range values log
    and fall back -- the whole app must not fail to start because one optional
    estimate was mistyped in a web form. Zero is refused specifically because it
    would turn ``_max_pipeline_llm_calls`` into a ``ZeroDivisionError`` at the
    top of the request path.
    """
    default = _DEFAULT_PIPELINE_SECONDS_PER_LLM_CALL
    raw = os.getenv("PIPELINE_SECONDS_PER_LLM_CALL")
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        print(
            "PIPELINE_SECONDS_PER_LLM_CALL is not an integer "
            f"({raw!r}); using {default}",
            flush=True,
        )
        return default
    if value < 1 or value > _MAX_PIPELINE_SECONDS_PER_LLM_CALL:
        print(
            f"PIPELINE_SECONDS_PER_LLM_CALL is out of range ({value}; allowed "
            f"1-{_MAX_PIPELINE_SECONDS_PER_LLM_CALL}); using {default}",
            flush=True,
        )
        return default
    return value


PIPELINE_SECONDS_PER_LLM_CALL = _pipeline_seconds_per_llm_call()


def _max_pipeline_llm_calls() -> int:
    """Model calls the fixed parent budget has room for.

    Read through a function rather than frozen into a module constant so a test
    that monkeypatches ``PIPELINE_SECONDS_PER_LLM_CALL`` sees its own value --
    the same reason ``_enforce_ai_hedge_fund_window`` reads its module-level
    limit at call time instead of closing over it.
    """
    usable = max(
        0,
        PIPELINE_SUBPROCESS_TIMEOUT_SECONDS - SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS,
    )
    return usable // max(1, PIPELINE_SECONDS_PER_LLM_CALL)
```

> **Note for the implementer:** `_max_pipeline_llm_calls` reads the *module attribute*
> `PIPELINE_SECONDS_PER_LLM_CALL`, not the function. Task 4's tests monkeypatch that
> attribute; if you inline `_pipeline_seconds_per_llm_call()` here instead, those tests
> silently read the environment and stop testing what they claim to.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_backtests_router.py -v -k "pipeline_seconds_per_call"
```

Expected: PASS (9 cases — 2 plain plus 7 parametrized).

- [ ] **Step 5: Strip the variable in the test harness**

In `dashboard/backend/tests/conftest.py`, find the block that deletes operator-set variables (the one already handling `LLM_ESCALATE_CEILING_ON_RETRY`, `IFIND_*`, `ADMIN_BOOTSTRAP_SECRET`, `CONTENT_DATABASE_URL`, `AGENT_RUNS_DATABASE_URL`, `RENDER`) and add `PIPELINE_SECONDS_PER_LLM_CALL` alongside them, matching the surrounding style exactly. Add a short comment in the file's existing idiom:

```python
    # A developer configured like prod would otherwise have a different call
    # budget than the suite asserts, and read the resulting refusals as
    # unrelated failures.
```

- [ ] **Step 6: Run the full router suite to verify nothing regressed**

```bash
pytest dashboard/backend/tests/test_backtests_router.py -v
```

Expected: PASS, no new failures.

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/conftest.py
git commit -m "feat: make the pipeline per-call latency estimate tunable"
```

---

## Task 4: Refuse an uncompletable pipeline LLM window

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py` (append to the Task 2/3 block)
- Test: `dashboard/backend/tests/test_backtests_router.py`

**Interfaces:**
- Consumes: `_estimated_pipeline_llm_calls`, `_max_pipeline_llm_calls`, `PIPELINE_SECONDS_PER_LLM_CALL`, `PIPELINE_DECISION_BARS_PER_TRADING_DAY`, `_estimated_decision_days`, `split_pipeline`, `PIPELINE_RUNTIME_TYPE`, `LLM_DECISION_SOURCE` (all already imported or defined).
- Produces: `_enforce_pipeline_llm_window(runtime_type: str, decision_source: str, start_date: str, end_date: str, pipeline: Optional[List[Dict[str, Any]]]) -> None` — raises `HTTPException(422)` or returns `None`. Task 5 calls it.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_backtests_router.py`:

```python
def test_pipeline_window_guard_allows_the_shipped_default_window():
    """The modal's own pre-filled window must not 422 on the onboarding path.

    2026-04-15 -> 2026-04-23 with the advertised four-module pipeline is 196 of
    200 calls. It passes by four -- deliberately pinned, so that anyone who
    shrinks the budget or raises the per-call estimate sees exactly which run
    they just made illegal.
    """
    pipeline = [{"label": f"Step {i}"} for i in range(4)]
    assert (
        bt._enforce_pipeline_llm_window(
            "pipeline", "llm", "2026-04-15", "2026-04-23", pipeline
        )
        is None
    )


def test_pipeline_window_guard_refuses_a_wide_pipeline_over_a_legal_window():
    """A window inside MAX_BACKTEST_DAYS can still be uncompletable.

    This is the case MAX_BACKTEST_DAYS cannot express: 14 calendar days is
    legal, but four decision steps over it is 280 calls against a 200-call
    budget. Window length alone does not say how much a run costs.
    """
    pipeline = [{"label": f"Step {i}"} for i in range(4)]
    with pytest.raises(HTTPException) as excinfo:
        bt._enforce_pipeline_llm_window(
            "pipeline", "llm", "2026-04-06", "2026-04-19", pipeline
        )
    assert excinfo.value.status_code == 422


def test_pipeline_window_guard_names_both_levers_and_both_numbers():
    """A refusal naming one lever is a dead end for a user blocked by the other.

    The bound is a product of window length and pipeline width, so the message
    has to offer both exits -- and name the estimate AND the budget, the way
    _enforce_ai_hedge_fund_window names the bound and the request.
    """
    pipeline = [{"label": f"Step {i}"} for i in range(4)]
    with pytest.raises(HTTPException) as excinfo:
        bt._enforce_pipeline_llm_window(
            "pipeline", "llm", "2026-04-06", "2026-04-19", pipeline
        )
    detail = excinfo.value.detail
    assert "280" in detail          # what this request needs
    assert "200" in detail          # what the deployment allows
    assert "shorten" in detail.lower()
    assert "step" in detail.lower()


def test_pipeline_window_guard_allows_a_single_prompt_over_the_full_window():
    """The no-pipeline path costs one call per bar and fits the full fortnight.

    70 calls of 200. Lowering MAX_BACKTEST_DAYS must not be read as having
    made the simple path marginal too.
    """
    assert (
        bt._enforce_pipeline_llm_window(
            "pipeline", "llm", "2026-04-06", "2026-04-19", None
        )
        is None
    )


def test_pipeline_window_guard_ignores_runs_it_does_not_govern():
    """Rule-based runs and the hosted runtime are out of scope.

    A rule-based run makes no model calls at all, and the hosted path has its
    own window bound sized from its own enforced per-step timeout. Applying a
    pipeline-shaped estimate to either would refuse runs on arithmetic that
    does not describe them.
    """
    wide = [{"label": f"Step {i}"} for i in range(4)]
    assert (
        bt._enforce_pipeline_llm_window(
            "pipeline", "rule_based", "2026-04-06", "2026-04-19", wide
        )
        is None
    )
    assert (
        bt._enforce_pipeline_llm_window(
            "ai_hedge_fund", "llm", "2026-04-06", "2026-04-19", wide
        )
        is None
    )


def test_pipeline_window_guard_tracks_the_operator_override(monkeypatch):
    """Raising the per-call estimate shrinks what is runnable, immediately."""
    pipeline = [{"label": f"Step {i}"} for i in range(4)]
    monkeypatch.setattr(bt, "PIPELINE_SECONDS_PER_LLM_CALL", 30)
    assert bt._max_pipeline_llm_calls() == 100
    with pytest.raises(HTTPException) as excinfo:
        bt._enforce_pipeline_llm_window(
            "pipeline", "llm", "2026-04-15", "2026-04-23", pipeline
        )
    assert excinfo.value.status_code == 422
```

Confirm `HTTPException` is imported in the test module; if not, add `from fastapi import HTTPException` beside the existing imports.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_backtests_router.py -v -k "pipeline_window_guard"
```

Expected: FAIL — `AttributeError: module ... has no attribute '_enforce_pipeline_llm_window'`.

- [ ] **Step 3: Verify the arithmetic the tests assert**

Before implementing, confirm the two windows are what the tests claim:

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from dashboard.backend.api.routers import backtests as bt
for a, b in (('2026-04-15','2026-04-23'), ('2026-04-06','2026-04-19')):
    print(a, b, 'weekdays=', bt._estimated_decision_days(a, b))
"
```

Expected: `2026-04-15 2026-04-23 weekdays= 7` and `2026-04-06 2026-04-19 weekdays= 10`. If either differs, fix the **test** numbers to match the real day count before continuing — do not adjust the implementation to fit a wrong assertion.

> ⚠ **This command mutates the committed seed database.** Importing a store module runs `CREATE TABLE IF NOT EXISTS` against `DATABASE_PATH`, and a bare `python3 -c` outside pytest has no `conftest.py` redirecting that to a temp file — so `dashboard/storage/data/backtest.db` comes back dirty. Confirmed while writing this plan. Immediately after running it:
>
> ```bash
> git status --short && git checkout -- dashboard/storage/data/backtest.db
> ```
>
> Never let that file into a commit. Prefer running the check under pytest if you would rather not touch it at all.

- [ ] **Step 4: Write the guard**

Append to the block added in Tasks 2 and 3:

```python
def _enforce_pipeline_llm_window(
    runtime_type: str,
    decision_source: str,
    start_date: str,
    end_date: str,
    pipeline: Optional[List[Dict[str, Any]]],
) -> None:
    """Refuse a pipeline LLM run the fixed parent budget cannot finish.

    Only 422, never 503: unlike the hosted guard there is no "turned off here"
    state to report, and every refusal this raises is one the caller can act on
    -- by shortening the window or by removing pipeline steps.

    Both exits are named because the bound is a PRODUCT of the two. A user whose
    window is already short and whose pipeline is wide is told to shorten the
    window by a message that names only dates, and has no way to discover the
    real lever. ``MAX_BACKTEST_DAYS`` is the calendar half of this bound; this
    is the half that knows what the window costs.
    """
    if runtime_type != PIPELINE_RUNTIME_TYPE:
        return
    if decision_source != LLM_DECISION_SOURCE:
        return
    estimated = _estimated_pipeline_llm_calls(start_date, end_date, pipeline)
    allowed = _max_pipeline_llm_calls()
    if estimated <= allowed:
        return

    trading_days = _estimated_decision_days(start_date, end_date)
    decision_steps, _post_trade_steps = split_pipeline(pipeline)
    steps = max(1, len(decision_steps))
    minutes = PIPELINE_SUBPROCESS_TIMEOUT_SECONDS // 60
    raise HTTPException(
        status_code=422,
        detail=(
            f"This run needs about {estimated} model calls "
            f"({trading_days} trading days x "
            f"{PIPELINE_DECISION_BARS_PER_TRADING_DAY} hourly bars x "
            f"{steps} pipeline step(s)), and a backtest has room for about "
            f"{allowed} within its {minutes}-minute limit. Shorten the date "
            "range, or use fewer pipeline steps, and run it again."
        ),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_backtests_router.py -v -k "pipeline_window_guard"
```

Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtests_router.py
git commit -m "feat: refuse a pipeline LLM window the budget cannot finish"
```

---

## Task 5: Wire the guard into `/backtest/run`

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py:2750` (immediately after the second `_validate_backtest_params` call)
- Test: `dashboard/backend/tests/test_backtests_router.py`

**Interfaces:**
- Consumes: `_enforce_pipeline_llm_window` from Task 4.
- Produces: nothing new; the route now answers 422 for an uncompletable pipeline LLM window.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_backtests_router.py`. Follow the existing route tests in this file for the client fixture and the session header they use — reuse that fixture rather than building a new one:

```python
def test_backtest_run_refuses_an_uncompletable_pipeline_window(client):
    """The 422 arrives before the rate limiter, the slot ledger and any spend.

    A refusal that costs a rate-limit token or a concurrency slot punishes the
    user for a request the server was always going to decline -- and one that
    arrives after the billing preflight has touched the credential store has
    already done work on a run that cannot happen.
    """
    response = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-04-06",
            "end_date": "2026-04-19",
            "decision_source": "llm",
            "model": "claude-haiku-4-5-20251001",
            "billing_mode": "byok",
            "pipeline": [{"label": f"Step {i}"} for i in range(4)],
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "280" in detail
    assert "shorten" in detail.lower()


def test_backtest_run_still_refuses_an_over_long_window_on_dates_alone(client):
    """MAX_BACKTEST_DAYS keeps its own distinct message.

    The two bounds answer different questions and a user who picked a
    three-week window should be told the window is too long, not handed call
    arithmetic about a pipeline they may not have configured.
    """
    response = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-30",
            "decision_source": "llm",
            "model": "claude-haiku-4-5-20251001",
            "billing_mode": "byok",
        },
    )
    assert response.status_code == 422
    assert "max 14 days" in response.json()["detail"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_backtests_router.py -v -k "backtest_run_refuses or backtest_run_still_refuses"
```

Expected: the first FAILS (the run is accepted, or fails later for another reason); the second should already PASS from Task 1. If the second fails, the Task 1 copy change is incomplete — fix that before continuing.

- [ ] **Step 3: Call the guard in the handler**

In `dashboard/backend/api/routers/backtests.py`, find the second `_validate_backtest_params` call (`:2750`, preceded by the comment `# Validate before taking rate-limit capacity or scheduling the worker.`) and insert directly beneath it:

```python
    # After the date validation above, so this never has to be a second date
    # validator, and before the rate limiter, the slot ledger and the billing
    # preflight below -- for the reason _enforce_ai_hedge_fund_window is placed
    # ahead of its credential lookup: a run this deployment cannot finish is
    # refused whether or not the caller's credentials are good, and there is no
    # reason to spend a rate-limit token, a concurrency slot or a trip to the
    # secret store to say so.
    _enforce_pipeline_llm_window(
        runtime_type, resolved_decision_source, start_date, end_date, pipeline
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_backtests_router.py -v -k "backtest_run_refuses or backtest_run_still_refuses"
```

Expected: PASS.

- [ ] **Step 5: Run the full backend suite**

```bash
pytest dashboard/backend/tests/ -q
```

Expected: PASS end-to-end. Per CLAUDE.md the suite is green on a fresh run, so **any** red result here is a real regression introduced by this work — do not dismiss one as pre-existing. Two specific things to confirm are still green: `test_onboarding_flow_ux.py` (the JS/HTML mirrors) and `test_app_copy_register.py` (the untouched "60 minutes" strings).

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtests_router.py
git commit -m "fix: preflight the pipeline LLM window on /backtest/run"
```

---

## Task 6: Document the new bound

**Files:**
- Modify: `CLAUDE.md` (env section, after the `LLM_ESCALATE_CEILING_ON_RETRY` bullet)

**Interfaces:**
- Consumes: the constants and behaviour from Tasks 1-5.
- Produces: nothing executable.

- [ ] **Step 1: Add the environment bullet**

Insert into the env list in `CLAUDE.md`, matching the surrounding bullets' style (they explain the *why* and the failure they prevent, not just the syntax):

```markdown
- `PIPELINE_SECONDS_PER_LLM_CALL` (optional, default **15**, range 1–300): the assumed wall-clock of one model call, used only by `_enforce_pipeline_llm_window` in `api/routers/backtests.py` to refuse a `runtime_type="pipeline"` + `decision_source="llm"` window the fixed `PIPELINE_SUBPROCESS_TIMEOUT_SECONDS` (3600) cannot finish. The estimate is `bars × pipeline decision steps + trading days × post-trade steps`, against `(3600 − 600) // this` calls — 200 at the default. ⚠ **This is a calibrated estimate, not an inverted ceiling, and the difference is the whole design.** `_enforce_ai_hedge_fund_window` can invert its budget because the hosted path *enforces* `AI_HEDGE_FUND_TIMEOUT_SECONDS` per step; nothing enforces a per-bar ceiling on the pipeline path, where the only wall-clock bound is the 60s httpx read timeout every provider adapter takes by default (`infrastructure/llm/execution/adapters/base.py`). A refusal built on that worst case would allow under two trading days, so the 3600s timeout remains the real backstop and a slow reasoning model can still be killed mid-run (issue #474 item 2). The safe direction is therefore **inverted** relative to `_backtest_subprocess_timeout`, which over-provisions on purpose: this computes a *refusal*, so under-estimating lets a marginal run through while over-estimating refuses one that would have worked — on the declared onboarding task. Do not "tighten" it toward the worst case. Junk, zero, negative or out-of-range values log and fall back rather than raising at import; **an unparseable value read with a bare `int()` at module scope in this very module once killed app boot**. Worth knowing before tuning: at the default, the shipped modal window (`app.html`, 7 weekdays → 49 bars) running the four-module pipeline advertised at `app.html:1462` costs **196 of 200 calls** — it passes by four, which is evidence the 3600s budget is undersized for the advertised product rather than an argument for a smaller number here. `tests/conftest.py` strips it.
```

- [ ] **Step 2: Update the `MAX_BACKTEST_DAYS` mention in the credits module docstring**

`dashboard/backend/domain/entitlements/credits.py:19` says `MAX_BACKTEST_DAYS` "bounds that ratio at ~31x". Change `~31x` → `~14x`. The second mention at `:98` names the constant without a number and needs **no** edit. That module is superseded and on no request path (see CLAUDE.md's `CREDITS_METERING_ENABLED` bullet) — this is a factual correction to its prose only, **not** a re-wiring of the metering it describes. ⚠ CLAUDE.md warns explicitly against reading "only tests call it" as "spend is unmetered" and re-wiring the debit; that inference has already been drawn once and was wrong.

```bash
grep -n "31x\|MAX_BACKTEST_DAYS" dashboard/backend/domain/entitlements/credits.py
```

- [ ] **Step 3: Verify no stale "31 days" remains**

```bash
grep -rn "31 days\|MAX_BACKTEST_DAYS = 31\|31x" \
  dashboard/backend/api dashboard/backend/domain dashboard/frontend/app.js dashboard/frontend/app.html CLAUDE.md
```

Expected: no hits. Hits inside `docs/superpowers/plans/` are historical records of past sessions and must be left alone.

- [ ] **Step 4: Run the suite once more**

```bash
pytest dashboard/backend/tests/ -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md dashboard/backend/domain/entitlements/credits.py
git commit -m "docs: record the pipeline window preflight and the 14-day cap"
```

---

## Branch and PR

Per CLAUDE.md's merge discipline:

- The current checkout is on `fix/308-prod-memory-rationale`, whose work is unrelated and may already be consumed by a PR. **Cut a new branch before the first commit** — `git switch -c fix/474-pipeline-window-preflight` off an up-to-date `main` — and check `gh pr list --head fix/308-prod-memory-rationale --state all` first; GitHub silently orphans commits pushed behind a merged PR.
- Parallel sessions share this checkout. Re-verify the branch immediately before each commit.
- PR title short, in the repo's convention: `fix: preflight the pipeline backtest window`.
- The PR body must **not** claim to close #474 — this plan implements checkbox 1 of five. Reference it as "addresses item 1 of #474" so the issue stays open for items 2-5. A `Closes #474` line here would auto-close four unbuilt items.

## Out of scope — do not drift into these

Each is a separate item of #474 with its own plan:

- **Item 2** — distinguishing a timeout from a failure in the error surface, and not settling spend for a run that returned nothing.
- **Item 3** — showing projected calls and an estimated cost range in the modal before ▶ Run Backtest. (Task 2's `_estimated_pipeline_llm_calls` is the natural server-side input for it; leave it unexposed for now rather than adding an endpoint nobody calls.)
- **Item 4** — an actionable retry hint on the `MAX_ACTIVE_DASHBOARD_BACKTESTS` capacity refusal.
- **Item 5** — margin between `BACKTEST_POLL_MAX_SECONDS` and the server budget, and any raise of the 3600s budget itself.
- **Issue #473** — raising the concurrency cap. A spend decision.
