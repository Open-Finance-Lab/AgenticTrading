"""
Example: Safe LLM Integration for Trading Decisions

This demonstrates the CORRECT way to integrate an LLM for trading decisions:
1. Backend owns all API credentials and market data
2. LLM receives only sanitized snapshots
3. LLM returns JSON decision (no tool calling)
4. Backend validates response before execution
5. All decisions are logged for audit trail

DO NOT expose websearch, browser, or HTTP tools to the LLM.
"""

import json
import logging
import os
from typing import Dict, Optional, Any

# NOTE: This requires anthropic library
# pip install anthropic

from anthropic import Anthropic

# Import validation (from the same backend)
from dashboard.backend.infrastructure.llm.validator import (
    validate_llm_response,
    create_safe_prompt,
    log_audit_trail
)

logger = logging.getLogger(__name__)


class SafeTradingLLMIntegration:
    """
    Safe LLM integration for trading decisions.
    
    Enforces:
    - No tool calling
    - No web access
    - JSON response only
    - Strict schema validation
    - Portfolio constraints
    """
    
    def __init__(self, api_key: str):
        """Initialize LLM client (Claude via Anthropic SDK)"""
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-opus-4-1"  # Latest, most capable Claude
    
    async def get_trading_decision(
        self,
        session_id: str,
        market_snapshot: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        current_prices: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Get a safe trading decision from the LLM.
        
        Args:
            session_id: User session ID
            market_snapshot: Sanitized market data (OHLCV + indicators)
            portfolio_state: Current portfolio with constraints
            current_prices: Current market prices
        
        Returns:
            Validated trading decision or None if invalid
        
        Process:
            1. Create safe prompt (no tool calling)
            2. Call LLM with explicit "no tools" constraint
            3. Parse and validate response
            4. Check portfolio constraints
            5. Log decision to audit trail
            6. Return decision or None
        """
        
        try:
            # ================================================================
            # STEP 1: Create safe prompt
            # ================================================================
            prompt = create_safe_prompt(market_snapshot)
            
            logger.info(
                f"📤 Sending market snapshot to LLM (session {session_id[:8]}...)",
                extra={"market_keys": list(market_snapshot.keys())}
            )
            
            # ================================================================
            # STEP 2: Call LLM with STRICT constraints (NO TOOLS)
            # ================================================================
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,  # Small for JSON response only
                system="""You are a trading decision advisor.
                
CRITICAL: You CANNOT use tools, functions, or access any APIs.
You CANNOT make any tool_use blocks.
You MUST respond with ONLY valid JSON matching the schema provided.
If you cannot make a valid decision, respond with: {"action": "hold"}""",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                # IMPORTANT: NOT including tools parameter
                # This prevents the model from attempting tool calling
            )
            
            # ================================================================
            # STEP 3: Extract response text
            # ================================================================
            llm_response = response.content[0].text
            
            logger.info(
                f"📥 Received LLM response (preview: {llm_response[:100]}...)",
                extra={"session_id": session_id}
            )
            
            # ================================================================
            # STEP 4: Validate response (CRITICAL SECURITY CHECK)
            # ================================================================
            decision = validate_llm_response(
                llm_response,
                portfolio_state,
                current_prices
            )
            
            if decision is None:
                logger.warning(
                    f"❌ LLM response validation failed (session {session_id[:8]})",
                    extra={"raw_response_preview": llm_response[:200]}
                )
                log_audit_trail(
                    session_id,
                    None,
                    llm_response,
                    "rejected"
                )
                return None
            
            # ================================================================
            # STEP 5: Log decision to audit trail
            # ================================================================
            log_audit_trail(
                session_id,
                decision,
                llm_response,
                "success"
            )
            
            logger.info(
                f"✅ LLM decision validated and logged",
                extra={
                    "decision": decision.model_dump(),
                    "session_id": session_id
                }
            )
            
            return decision.model_dump()
        
        except Exception as e:
            logger.error(
                f"❌ LLM integration error: {e}",
                extra={"session_id": session_id, "error": str(e)},
                exc_info=True
            )
            log_audit_trail(
                session_id,
                None,
                str(e),
                "failed",
                error_msg=str(e)
            )
            return None


# ============================================================================
# EXAMPLE: Backend Endpoint (to be added to app.py)
# ============================================================================

EXAMPLE_ENDPOINT = '''
# Add this to dashboard/backend/app.py

from dashboard.backend.llm_integration_example import SafeTradingLLMIntegration
import os

# Initialize LLM integration (once at startup)
llm_integration = SafeTradingLLMIntegration(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

@app.post("/api/llm-trading-decision")
async def get_llm_trading_decision(request: Request):
    """
    Get a trading decision from Claude.
    
    Safe endpoint that:
    1. Fetches market data from backend database
    2. Prepares sanitized snapshot
    3. Calls LLM with strict constraints
    4. Validates response
    5. Executes trade if valid
    
    Returns:
        {"success": bool, "decision": {...}, "error": str}
    """
    session_id = request.state.session_id
    
    try:
        # Step 1: Get current market snapshot (backend owns this)
        market_snapshot = prepare_market_snapshot()  # Already defined
        
        # Step 2: Get portfolio state
        portfolio_state = get_portfolio_state(session_id)
        
        # Step 3: Get current prices
        current_prices = get_current_prices()
        
        # Step 4: Call LLM (safe, no tools)
        decision = await llm_integration.get_trading_decision(
            session_id=session_id,
            market_snapshot=market_snapshot,
            portfolio_state=portfolio_state,
            current_prices=current_prices
        )
        
        if not decision:
            return {
                "success": False,
                "error": "LLM decision validation failed",
                "decision": None
            }
        
        # Step 5: Execute trade (backend does this, NOT the LLM)
        trade_result = execute_trade(
            session_id=session_id,
            decision=decision,
            current_prices=current_prices
        )
        
        return {
            "success": trade_result["success"],
            "decision": decision,
            "trade_result": trade_result.get("trade_id")
        }
    
    except Exception as e:
        logger.error(f"LLM trading endpoint error: {e}")
        return {
            "success": False,
            "error": str(e),
            "decision": None
        }
'''

# ============================================================================
# WHAT NOT TO DO (Anti-patterns)
# ============================================================================

ANTI_PATTERN_1 = '''
# ❌ DO NOT: Give LLM direct API access

from alpaca.trading.client import TradingClient

@app.post("/api/trade")
async def trade_with_llm(request: Request):
    # ❌ WRONG: LLM decides AND executes
    decision = llm.get_decision(market_data)
    
    # ❌ WRONG: Passing Alpaca client to LLM
    client = TradingClient(api_key, secret_key)
    result = llm.execute_with_tools([
        Tool(name="place_order", func=client.submit_order)
    ])
    
    return result
'''

ANTI_PATTERN_2 = '''
# ❌ DO NOT: Expose websearch tools

decision = llm.get_decision(
    market_data,
    tools=[
        web_search,  # ❌ DANGEROUS
        browser,     # ❌ DANGEROUS
        http_request # ❌ DANGEROUS
    ]
)
'''

ANTI_PATTERN_3 = '''
# ❌ DO NOT: Send secrets to LLM

prompt = f"""
You have access to:
- API Key: {api_key}  # ❌ NEVER
- Database: {db_password}  # ❌ NEVER
- Trading account: {account_id}  # ❌ NEVER

Make trading decisions and execute them.
"""
'''

ANTI_PATTERN_4 = '''
# ❌ DO NOT: Trust LLM response without validation

decision = llm.get_decision(market_data)
execute_trade(decision)  # ❌ No validation!

# Instead, do:
if validate_decision(decision):
    execute_trade(decision)
'''

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

SETUP_INSTRUCTIONS = '''
# 1. Install dependencies
pip install anthropic

# 2. Set environment variable (in .env or shell)
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run tests
pytest dashboard/backend/tests/test_llm_validator.py -v

# 4. Check audit logs
tail -f logs/trading_audit.log

# 5. Monitor LLM decisions
# Look for "AUDIT:" entries in logs to see all LLM decisions
'''

# ============================================================================
# TESTING THE INTEGRATION
# ============================================================================

async def test_llm_integration():
    """Test the LLM integration with a sample market snapshot"""
    
    # Initialize
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return
    
    llm = SafeTradingLLMIntegration(api_key)
    
    # Sample data
    session_id = "test-session-123"
    market_snapshot = {
        "timestamp": "2026-05-07T14:30:00Z",
        "portfolio": {
            "cash": 50000,
            "positions": [
                {"symbol": "AAPL", "shares": 100, "value": 15000}
            ],
            "total_equity": 65000
        },
        "signals": {
            "AAPL": {
                "price": 150.00,
                "rsi": 35,  # Oversold
                "macd": 0.5,
                "sma20": 155.00,
                "trend": "downtrend"
            },
            "MSFT": {
                "price": 420.00,
                "rsi": 72,  # Overbought
                "macd": -0.2,
                "sma20": 410.00,
                "trend": "uptrend"
            }
        }
    }
    
    portfolio_state = {
        "cash": 50000,
        "positions": [{"symbol": "AAPL", "shares": 100}],
        "constraints": {
            "cash_available": 50000,
            "max_position_size": 5000,
            "max_daily_trades": 20,
            "max_position_value": 50000,
            "min_confidence": 0.3
        }
    }
    
    current_prices = {
        "AAPL": 150.00,
        "MSFT": 420.00,
        "JPM": 180.00,
        "V": 290.00
    }
    
    # Call LLM
    print("🤖 Calling LLM for trading decision...")
    decision = await llm.get_trading_decision(
        session_id=session_id,
        market_snapshot=market_snapshot,
        portfolio_state=portfolio_state,
        current_prices=current_prices
    )
    
    if decision:
        print(f"✅ Decision received: {json.dumps(decision, indent=2)}")
    else:
        print("❌ LLM decision validation failed")


if __name__ == "__main__":
    print("🔐 Safe LLM Trading Integration Example")
    print("=" * 70)
    print("\nSetup:")
    print(SETUP_INSTRUCTIONS)
    print("\nRun test:")
    print("  python dashboard/backend/llm_integration_example.py")
