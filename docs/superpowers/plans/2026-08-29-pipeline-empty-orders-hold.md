# Pipeline Empty Orders Hold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treat explicitly empty pipeline decision envelopes as valid model-driven HOLD decisions without accepting malformed non-empty payloads.

**Architecture:** Keep normalization at the existing `pipeline_output_to_decision()` boundary. Preserve non-empty envelope priority, distinguish empty supported lists from unusable non-empty lists, and return `{"actions": []}` only when no unusable non-empty list was seen. Verify the boundary with unit contracts and the strict pipeline path with a portfolio-manager integration test.

**Tech Stack:** Python 3, pytest, existing hourly backtest pipeline and portfolio manager.

**Spec:** `docs/superpowers/specs/2026-08-29-pipeline-empty-orders-hold-design.md`

## Global Constraints

- Support empty `actions`, `orders`, and `risk_actions` envelopes.
- Preserve existing normalization and priority for non-empty envelopes.
- Keep malformed, unsupported, incorrectly typed, and unusable non-empty payloads invalid.
- Do not change prompts, providers, billing, retries, fallback budgets, frontend behavior, concurrency, or deployment limits.
- Add no dependencies.

---

### Task 1: Define and implement the empty-envelope contract

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/pipeline_runner.py:145-208`
- Test: `dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py`

**Interfaces:**
- Consumes: `pipeline_output_to_decision(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]`.
- Produces: the same signature, with each explicitly empty supported list normalized to `{"actions": []}`.

- [ ] **Step 1: Write failing empty-envelope tests**

Add `import pytest` and these tests below the existing non-empty action tests:

```python
@pytest.mark.parametrize(
    "parsed",
    [{"actions": []}, {"orders": []}, {"risk_actions": []}],
)
def test_pipeline_output_to_decision_empty_envelope_is_hold(parsed):
    assert pipeline_output_to_decision(parsed) == {"actions": []}


@pytest.mark.parametrize(
    "parsed",
    [
        {},
        {"orders": None},
        {"orders": "not-a-list"},
        {"orders": ["not-an-order"]},
        {"actions": [], "orders": ["not-an-order"]},
    ],
)
def test_pipeline_output_to_decision_rejects_invalid_payload(parsed):
    assert pipeline_output_to_decision(parsed) is None
```

- [ ] **Step 2: Run the focused module and verify RED**

Run:

```bash
python -m pytest dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py -q
```

Expected: the empty-envelope cases fail because the current function returns `None`; existing non-empty cases pass.

- [ ] **Step 3: Implement explicit empty-envelope handling**

Add state at the top of `pipeline_output_to_decision()`:

```python
saw_empty_envelope = False
saw_unusable_nonempty_envelope = False
```

Replace the `actions` truthiness branch with:

```python
actions = parsed.get("actions")
if isinstance(actions, list):
    if actions:
        return {"actions": actions}
    saw_empty_envelope = True
```

For `orders`, retain the existing normalization loop inside the non-empty branch and mark both outcomes explicitly:

```python
orders = parsed.get("orders")
if isinstance(orders, list):
    if not orders:
        saw_empty_envelope = True
    else:
        normalized = []
        for order in orders:
            if not isinstance(order, dict):
                continue
            side = str(order.get("side") or order.get("action") or "hold").lower()
            if side not in ("buy", "sell", "hold"):
                side = "hold"
            qty = order.get("qty", order.get("quantity", order.get("position_size", 0)))
            try:
                position_size = int(qty)
            except (TypeError, ValueError):
                position_size = 0
            normalized.append(
                {
                    "action": side,
                    "symbol": order.get("symbol"),
                    "confidence": float(order.get("confidence", 0.75) or 0.75),
                    "reasoning": order.get("reason") or order.get("rationale") or "",
                    "position_size": position_size,
                    "stop_loss_price": order.get("stop_loss_price"),
                    "take_profit_price": order.get("take_profit_price"),
                }
            )
        if normalized:
            return {"actions": normalized}
        saw_unusable_nonempty_envelope = True
```

Replace the `risk_actions` branch with the full empty/non-empty split:

```python
risk_actions = parsed.get("risk_actions")
if isinstance(risk_actions, list):
    if not risk_actions:
        saw_empty_envelope = True
    else:
        normalized = []
        for risk in risk_actions:
            if not isinstance(risk, dict):
                continue
            action_type = str(risk.get("action") or "hold").lower()
            if action_type in ("stop_loss", "take_profit", "trail"):
                side = "sell"
            elif action_type == "hold":
                side = "hold"
            else:
                side = (
                    action_type
                    if action_type in ("buy", "sell", "hold")
                    else "hold"
                )
            size_pct = float(risk.get("size_pct", 1.0) or 1.0)
            normalized.append(
                {
                    "action": side,
                    "symbol": risk.get("symbol"),
                    "confidence": 0.8,
                    "reasoning": (
                        risk.get("reason")
                        or risk.get("rationale")
                        or action_type
                    ),
                    "position_size": (
                        max(1, int(round(size_pct * 100)))
                        if side == "sell"
                        else 0
                    ),
                }
            )
        if normalized:
            return {"actions": normalized}
        saw_unusable_nonempty_envelope = True
```

Finish the function with:

```python
if saw_unusable_nonempty_envelope:
    return None
if saw_empty_envelope:
    return {"actions": []}
return None
```

- [ ] **Step 4: Run the focused module and verify GREEN**

Run:

```bash
python -m pytest dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py -q
```

Expected: every existing and new pipeline runner test passes.

---

### Task 2: Prove strict pipeline backtests preserve model HOLD

**Files:**
- Test: `dashboard/backend/tests/backtesting/test_portfolio_manager_move.py`

**Interfaces:**
- Consumes: `PortfolioManager.make_trading_decision_with_llm(..., pipeline: list[dict], strict_llm: bool) -> Dict` and Task 1's normalization contract.
- Produces: regression coverage proving `orders: []` returns `{"actions": []}`, increments `llm_decisions`, and does not raise `LLMDecisionError`.

- [ ] **Step 1: Add the strict pipeline regression test**

Add after `test_strict_llm_accepts_empty_actions_as_model_hold`:

```python
def test_strict_pipeline_accepts_empty_orders_as_model_hold():
    pm = CanonicalPortfolioManager(100000)
    client = _FakeClient(_FakeResp('{"orders": []}', _FakeUsage(3, 2)))

    result = pm.make_trading_decision_with_llm(
        _llm_state(),
        client,
        pipeline=[
            {
                "label": "Trading instruction",
                "prompt": "Return the orders for this bar.",
                "outputFormat": '{"orders": []}',
            }
        ],
        strict_llm=True,
    )

    assert result == {"actions": []}
    assert pm.llm_calls == 1
    assert pm.llm_decisions == 1
```

- [ ] **Step 2: Run the strict pipeline regression test**

Run:

```bash
python -m pytest dashboard/backend/tests/backtesting/test_portfolio_manager_move.py::test_strict_pipeline_accepts_empty_orders_as_model_hold -q
```

Expected: PASS with no rule-based fallback.

- [ ] **Step 3: Run the related regression suite**

Run:

```bash
python -m pytest dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py dashboard/backend/tests/backtesting/test_portfolio_manager_move.py -q
```

Expected: both modules pass with no regression in non-empty decisions, strict error handling, or fallback accounting.

- [ ] **Step 4: Check the diff and secrets**

Run:

```bash
git diff --check
git diff --stat origin/main...HEAD
git diff origin/main...HEAD | rg -n 'rnd_[A-Za-z0-9]{16,}|RENDER_API_KEY=.+|OPENAI_API_KEY=.+'
```

Expected: no whitespace errors, only the spec/plan/function/tests changed, and the credential scan returns no matches.

- [ ] **Step 5: Commit the implementation**

Run:

```bash
git add dashboard/backend/infrastructure/llm/pipeline_runner.py dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py dashboard/backend/tests/backtesting/test_portfolio_manager_move.py
git commit -m "fix(backtest): accept empty pipeline orders as hold"
```

- [ ] **Step 6: Push and create the pull request**

Run:

```bash
git push -u origin fix/pipeline-empty-orders-hold
gh pr create --repo Open-Finance-Lab/AgenticTrading --base main --head fix/pipeline-empty-orders-hold --title "fix(backtest): accept empty pipeline orders as hold" --body-file /tmp/pipeline-empty-orders-hold-pr.md
```

The PR body must include the reproduced Gemini/Qwen behavior, the normalization contract, the exact test commands, and explicit exclusions for retry, memory, concurrency, and frontend timeout work.
