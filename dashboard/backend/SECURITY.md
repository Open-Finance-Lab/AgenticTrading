# 🔐 LLM Security Implementation Guide

**Status:** ✅ Implemented  
**Last Updated:** May 7, 2026  
**Security Level:** Strict (No external API access to LLM)

---

## Quick Summary

The backend now provides **safe LLM integration** that:
- ✅ Prevents websearch, browser, and HTTP tool access
- ✅ Rejects tool_calls and function_calls
- ✅ Validates trading decisions against strict schema
- ✅ Enforces portfolio constraints
- ✅ Logs all decisions for audit trail

**Zero exposure of:** API keys, secrets, direct Alpaca access, raw market data APIs.

---

## Files

| File | Purpose |
|------|---------|
| `infrastructure/llm/validator.py` | Validation engine (schema, constraints, tool rejection) |
| `llm_integration_example.py` | Example endpoint to add to app.py |
| `tests/test_llm_validator.py` | 50+ unit tests (all cases including security) |

---

## Running Tests

```bash
# Install pytest if not present
pip install pytest

# Run all validator tests (from the repo root)
pytest dashboard/backend/tests/test_llm_validator.py -v

# Run specific test category
pytest dashboard/backend/tests/test_llm_validator.py::TestToolCallingRejection -v

# Check test coverage
pytest dashboard/backend/tests/test_llm_validator.py --cov=dashboard.backend.infrastructure.llm.validator
```

**Expected:** All tests pass ✅

---

## Using the Validator

### In Your Code

```python
from dashboard.backend.infrastructure.llm.validator import validate_llm_response

# Get response from LLM
raw_response = llm.call(prompt)  # JSON string from LLM

# Validate it
decision = validate_llm_response(
    raw_response,           # LLM response
    portfolio_state,        # Current portfolio
    current_prices          # Market prices
)

if decision is None:
    # Invalid response - already logged
    return {"action": "hold"}  # Safe fallback
else:
    # Valid decision - execute it
    return execute_trade(decision)
```

### Response Schema

The LLM **must** respond with JSON matching this schema:

```json
{
  "action": "buy|sell|hold",
  "symbol": "<DJIA stock>",
  "confidence": 0.0-1.0,
  "reasoning": "<max 500 chars>",
  "position_size": <int ≥ 0>,
  "stop_loss_price": <optional float>,
  "take_profit_price": <optional float>
}
```

**Valid example:**
```json
{
  "action": "buy",
  "symbol": "AAPL",
  "confidence": 0.75,
  "reasoning": "RSI oversold at 30, price below SMA20, good setup",
  "position_size": 100
}
```

---

## Exemption: the backtest engine's LLM loop (incl. custom `strategy_prompt`)

The hourly backtest engine (`domain/backtesting/portfolio_manager.py`,
`make_trading_decision_with_llm`) — including runs driven by a user's
free-form `strategy_prompt` (a My Agents agent's prompt) — does **not** route parsed
decisions through `validate_llm_response`. It predates the validator and its
behavior is intentionally preserved byte-for-byte for backtest reproducibility.

It applies these constraints inline instead:

| Constraint | Enforcement |
|------------|-------------|
| JSON-only, no tools | No request sends `tools`; text blocks only (`extract_response_text`, which skips `tool_use` and thinking blocks); prompt scaffold pins the output contract; response is parsed as JSON only (`parse_llm_response`), never executed |
| Symbol allowlist | Symbols outside the run's universe (`_allowed_set`: `DJIA_30` unless the run names a universe) skipped; a strict run rejects the whole batch |
| Action count | Capped at one action per universe symbol (`len(allowed_symbols)`); excess entries ignored, or the whole batch rejected on a strict run |
| Per-order size | `position_size > MAX_ORDER_SHARES` (10k, same constant as the validator) skipped; unparseable/non-finite/boolean sizes skipped; a buy size that rounds to 0 shares or is not positive skipped, never sized from confidence (only a missing/null/0 size is) |
| Cash bound | Buys must fit in available cash; sells limited to held shares |
| Confidence floor | Actions with confidence < 0.3 skipped |

A custom `strategy_prompt` replaces only the *strategy* section of the prompt;
the market snapshot, symbol allowlist, and strict JSON output contract are
appended by fixed scaffolding (`create_custom_prompt`) and cannot be overridden
by the user text. If you need the stricter schema validation (tool-call
rejection, audit logging), route through `validate_llm_response` as below —
new endpoints must not copy this exemption.

### Not covered: the Trading Algo backtest

The Trading Algo panel (`/api/algo/execute` → `dashboard/scripts/backtest_custom_algo.py`)
runs its own loop, `CustomAlgoPortfolioManager.make_trading_decision_with_custom_algo`,
and the table above does **not** apply to it. It reads `response.content[0].text`
instead of calling `extract_response_text`, parses with its own `find('{')`/`rfind('}')`
plus `fix_json_formatting` instead of `parse_llm_response`, keeps at most 8 actions on
`DJIA_30` names with confidence ≥ 0.3, and never applies `MAX_ORDER_SHARES`: a buy is
bounded only by available cash. Bring it onto the shared helpers before relying on the
table for it.

---

## Integration: Adding LLM Endpoint to Backend

Use `llm_integration_example.py` as a template. Here's the quick version:

### 1. Import at top of `app.py`

```python
from llm_integration_example import SafeTradingLLMIntegration
import os

# Initialize once at startup
llm_integration = SafeTradingLLMIntegration(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
```

### 2. Add endpoint

```python
@app.post("/api/llm-trading-decision")
async def get_llm_trading_decision(request: Request):
    """Get trading decision from LLM (safe, no tools)"""
    session_id = request.state.session_id
    
    # Get market snapshot
    market_snapshot = prepare_market_snapshot()
    portfolio_state = get_portfolio_state(session_id)
    current_prices = get_current_prices()
    
    # Call LLM (handles validation internally)
    decision = await llm_integration.get_trading_decision(
        session_id=session_id,
        market_snapshot=market_snapshot,
        portfolio_state=portfolio_state,
        current_prices=current_prices
    )
    
    if not decision:
        return {
            "success": False,
            "error": "LLM decision validation failed"
        }
    
    # Execute trade (backend only)
    result = execute_trade(session_id, decision, current_prices)
    
    return {
        "success": result["success"],
        "decision": decision,
        "trade_id": result.get("trade_id")
    }
```

### 3. Environment Variable

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or in `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## What Gets Validated

### ✅ Accepts

- Valid action: `buy`, `sell`, `hold`
- Valid DJIA symbols: AAPL, MSFT, JPM, V, JNJ, etc.
- Confidence: 0.0 to 1.0 (float)
- Position size: 0 to 10,000 shares
- Reasoning: 5-500 characters
- Stop loss/take profit: Optional floats

### ❌ Rejects

- Invalid JSON
- Missing required fields
- Tool_use/tool_calls blocks (security critical)
- Invalid symbols (non-DJIA)
- Confidence out of range
- Negative position sizes
- Position size > 10,000
- Insufficient cash for BUY
- Non-existent position for SELL
- Position size exceeds max

---

## Constraint Enforcement

The validator checks portfolio constraints:

```python
{
    "cash_available": 50000,      # Can't buy more than this
    "max_position_size": 5000,    # Per-stock limit
    "max_daily_trades": 20,       # Per-day limit
    "max_position_value": 50000,  # Per-position limit
    "min_confidence": 0.3         # Auto-downgrade to HOLD if below
}
```

**Example violations:**
- BUY 500 shares @ $150 = $75,000 > $50,000 cash → REJECTED
- Position size 6,000 > 5,000 max → REJECTED
- Confidence 0.25 < 0.3 min → Downgraded to HOLD

---

## Security Features

### 1. Tool Calling Prevention

The validator explicitly checks for tool_use/tool_calls in responses:

```python
if "tool_use" in raw_response or "tool_calls" in raw_response:
    logger.error("🚨 SECURITY: LLM attempted tool calling!")
    return None  # REJECT
```

### 2. No Tool Exposure

When calling Claude, **do NOT include tools**:

```python
# ✅ CORRECT: No tools parameter
response = client.messages.create(
    model="claude-opus-4-1",
    max_tokens=500,
    system="You CANNOT use tools or access APIs",
    messages=[...],
    # NOT including: tools=[...]
)

# ❌ WRONG: Exposing tools
response = client.messages.create(
    ...,
    tools=[web_search, browser, alpaca_api],  # DO NOT DO THIS
)
```

### 3. Data Sanitization

Only send to LLM:
```python
{
    "timestamp": "2026-05-07T14:30:00Z",
    "portfolio": {
        "cash": 50000,
        "positions": [{"symbol": "AAPL", "shares": 100}],
        "total_equity": 100000
    },
    "signals": {
        "AAPL": {"price": 150, "rsi": 35, "trend": "down"},
        ...
    }
}
```

Never send:
- API keys, tokens, secrets
- Database credentials
- Raw error traces
- System configuration
- External API URLs

### 4. Audit Trail

All decisions logged:

```
2026-05-07T14:30:00 📋 AUDIT: success decision for session abc123...
  "decision": {"action": "buy", "symbol": "AAPL", ...}
  "execution_status": "success"
```

---

## Testing

### Run All Tests

```bash
pytest dashboard/backend/tests/test_llm_validator.py -v
```

Output (node IDs are shown with the full repo-root path):
```
dashboard/backend/tests/test_llm_validator.py::TestValidResponses::test_valid_buy_decision PASSED
dashboard/backend/tests/test_llm_validator.py::TestValidResponses::test_valid_sell_decision PASSED
dashboard/backend/tests/test_llm_validator.py::TestValidResponses::test_valid_hold_decision PASSED
dashboard/backend/tests/test_llm_validator.py::TestInvalidJSON::test_malformed_json PASSED
dashboard/backend/tests/test_llm_validator.py::TestToolCallingRejection::test_tool_use_block_rejected PASSED
dashboard/backend/tests/test_llm_validator.py::TestToolCallingRejection::test_tool_calls_attribute_rejected PASSED
dashboard/backend/tests/test_llm_validator.py::TestToolCallingRejection::test_function_calls_rejected PASSED
dashboard/backend/tests/test_llm_validator.py::TestToolCallingRejection::test_api_access_attempt_rejected PASSED
dashboard/backend/tests/test_llm_validator.py::TestInvalidSchema::test_invalid_symbol PASSED
dashboard/backend/tests/test_llm_validator.py::TestInvalidSchema::test_confidence_out_of_range PASSED
dashboard/backend/tests/test_llm_validator.py::TestPortfolioConstraints::test_insufficient_cash PASSED
... (35+ more tests)

====== 50 passed in 1.23s ======
```

### Test Tool Calling Rejection

This is the most critical test. Ensure it passes:

```bash
pytest dashboard/backend/tests/test_llm_validator.py::TestToolCallingRejection -v
```

Must see:
```
test_tool_use_block_rejected PASSED
test_tool_calls_attribute_rejected PASSED
test_function_calls_rejected PASSED
test_api_access_attempt_rejected PASSED
```

---

## Common Issues

### Issue: "LLM attempted tool calling"

**Cause:** You included `tools=[...]` in the Claude API call

**Fix:** Remove tools parameter entirely
```python
# ❌ Wrong
response = client.messages.create(..., tools=[web_search])

# ✅ Right
response = client.messages.create(...)  # No tools
```

### Issue: "Invalid JSON from LLM"

**Cause:** LLM responded with markdown or explanation

**Fix:** Update system prompt to enforce JSON-only responses
```python
system="Respond with ONLY valid JSON. No markdown, no explanations."
```

### Issue: "Schema validation failed"

**Cause:** Missing required field in LLM response

**Example response that fails:**
```json
{
  "action": "buy",
  "symbol": "AAPL"
  // Missing: confidence, reasoning, position_size
}
```

**Fix:** Ensure LLM response includes all required fields (see schema above)

### Issue: "Position size exceeds max"

**Cause:** LLM suggested position larger than allowed

**Fix:** This is automatic - validator downgrades to HOLD. Check portfolio constraints in code.

---

## Logs & Monitoring

### Where Are Decisions Logged?

Check the application logs for `AUDIT:` entries:

```bash
grep "AUDIT:" logs/app.log | tail -20
```

### What to Monitor

1. **Tool calling attempts** (security alerts):
   ```
   🚨 SECURITY: LLM attempted tool calling!
   ```

2. **Validation failures** (debug):
   ```
   ❌ Schema validation failed: ...
   ```

3. **Constraint violations** (business logic):
   ```
   ❌ BUY: Insufficient cash
   ```

4. **Successful decisions** (normal):
   ```
   ✅ LLM decision validated: BUY AAPL 100 @ 0.75 confidence
   ```

---

## Compliance & Audit

Every LLM decision is logged with:
- Timestamp
- Session ID
- Decision details (action, symbol, confidence, etc.)
- Execution status (success/rejected/failed)
- Raw LLM response preview

Use for:
- Regulatory compliance
- Risk management review
- Debugging issues
- Auditing unauthorized attempts

---

## Troubleshooting Checklist

- [ ] Tests pass: `pytest dashboard/backend/tests/test_llm_validator.py -v`
- [ ] Environment variable set: `echo $ANTHROPIC_API_KEY`
- [ ] Anthropic SDK installed: `pip list | grep anthropic`
- [ ] No tools in API call (removed `tools=[...]`)
- [ ] System prompt enforces JSON-only
- [ ] Response schema matches specification
- [ ] Logs show validation success
- [ ] Audit trail populated

---

## Next Steps

1. **Run tests:** `pytest dashboard/backend/tests/test_llm_validator.py -v`
2. **Integrate:** Use `llm_integration_example.py` as template
3. **Test endpoint:** Make sample API call to `/api/llm-trading-decision`
4. **Monitor:** Check logs for `AUDIT:` entries

---

## References

- **infrastructure/llm/validator.py** — Source code for validation
- **llm_integration_example.py** — Integration pattern
- **tests/test_llm_validator.py** — Test coverage (50+ tests)

---

**Questions?** Check the source files above.

**Security concern?** Update `infrastructure/llm/validator.py` and re-run tests.

**New constraints?** Modify `PortfolioConstraints` and add tests.
