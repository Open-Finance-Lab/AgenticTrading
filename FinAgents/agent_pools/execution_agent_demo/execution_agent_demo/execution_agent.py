
import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExecutionAgent")

# Try to import OpenAI Agents SDK
try:
    from agents import Agent, Runner, function_tool
except ImportError:
    logger.warning("OpenAI Agents SDK not found. Using mock implementation.")
    
    def function_tool(func):
        func.is_tool = True
        return func

    class Agent:
        def __init__(self, name, instructions, model="gpt-4o-mini", tools=None):
            self.name = name
            self.instructions = instructions
            self.model = model
            self.tools = tools or []

        def run(self, request, context=None):
            # Mock execution logic
            logger.info(f"Mock Agent {self.name} received request: {request}")
            return "Mock execution completed."

    class Runner:
        @staticmethod
        def run_sync(agent, request, context=None):
            return agent.run(request, context)

# Try to import Alpaca SDK
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    logger.warning("Alpaca SDK not found. Using mock implementation.")
    ALPACA_AVAILABLE = False
    OrderSide = Any
    TimeInForce = Any
    OrderType = Any

# Pydantic Models for Tools
class OrderRequest(BaseModel):
    symbol: str
    qty: float
    side: str  # 'buy' or 'sell'

class ExecutionResult(BaseModel):
    execution_results: List[Dict[str, str]]

class AccountSummary(BaseModel):
    buying_power: float
    cash: float
    portfolio_value: float
    currency: str

class Position(BaseModel):
    symbol: str
    qty: float
    market_value: float
    current_price: float

class AlpacaService:
    """
    Service layer to interact with Alpaca API.
    """
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.client = None
        self.is_mock = False
        
        if not api_key or "MOCK" in api_key:
             self.is_mock = True
             logger.info("Using Mock Alpaca Service (invalid or mock keys detected)")
             return

        if ALPACA_AVAILABLE and api_key:
            try:
                self.client = TradingClient(api_key, secret_key, paper=paper)
                logger.info(f"Connected to Alpaca {'Paper' if paper else 'Live'} Trading")
            except Exception as e:
                logger.error(f"Failed to connect to Alpaca: {e}")
                self.is_mock = True
        else:
            logger.warning("Alpaca client not initialized (missing keys or library)")
            self.is_mock = True

    def get_account(self):
        if not self.is_mock and self.client:
            try:
                return self.client.get_account()
            except Exception as e:
                logger.warning(f"Alpaca API error: {e}. Falling back to mock.")
        
        # Mock Data
        return type('obj', (object,), {
            "buying_power": "100000.00",
            "cash": "50000.00",
            "portfolio_value": "150000.00",
            "currency": "USD",
            "status": "ACTIVE"
        })

    def get_positions(self):
        if not self.is_mock and self.client:
             try:
                return self.client.get_all_positions()
             except Exception:
                pass
        return []

    def place_order(self, symbol: str, qty: float, side: str, type: str = "market", time_in_force: str = "day"):
        if not self.is_mock and self.client:
            try:
                order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY
                )
                return self.client.submit_order(order_data=req)
            except Exception as e:
                 logger.error(f"Order failed: {e}")
                 return {"error": str(e)}
        
        logger.info(f"MOCK ORDER: {side} {qty} {symbol}")
        return {"id": f"mock_{symbol}", "symbol": symbol, "status": "filled", "qty": qty, "side": side}

    def cancel_all_orders(self):
        if not self.is_mock and self.client:
            return self.client.cancel_orders()
        logger.info("MOCK: Cancelled all orders")

# Define Tools
alpaca_service = None  # Will be initialized

@function_tool
def get_account_summary() -> str:
    """
    Get the current account summary including buying power, cash, and portfolio value.
    Returns a JSON string with the account details.
    """
    if not alpaca_service:
        return json.dumps({"error": "Alpaca service not initialized"})
    
    try:
        acct = alpaca_service.get_account()
        # Handle both object and dict (mock)
        if hasattr(acct, 'buying_power'):
             data = {
                "buying_power": float(acct.buying_power),
                "cash": float(acct.cash),
                "portfolio_value": float(acct.portfolio_value),
                "currency": acct.currency
            }
        else:
            data = acct
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": str(e)})

@function_tool
def get_current_positions() -> str:
    """
    Get all current open positions.
    Returns a JSON string list of positions.
    """
    if not alpaca_service:
        return "[]"
    
    try:
        positions = alpaca_service.get_positions()
        result = []
        for p in positions:
            if hasattr(p, 'symbol'):
                 result.append({
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "current_price": float(p.current_price)
                })
            else:
                result.append(p) # Mock
        return json.dumps(result)
    except Exception as e:
        return json.dumps([{"error": str(e)}])

@function_tool
def execute_orders(orders: List[OrderRequest]) -> str:
    """
    Execute a list of orders.
    
    Args:
        orders: List of OrderRequest objects, each containing 'symbol', 'qty', 'side' ('buy' or 'sell').
    """
    if not alpaca_service:
        return json.dumps({"error": "Alpaca service not initialized"})
    
    results = []
    for order in orders:
        try:
            # Access Pydantic model attributes
            symbol = order.symbol
            qty = order.qty
            side = order.side
            
            if qty > 0:
                res = alpaca_service.place_order(symbol, qty, side)
                results.append({"symbol": symbol, "status": "submitted", "details": str(res)})
            else:
                results.append({"symbol": symbol, "status": "skipped", "reason": "qty <= 0"})
        except Exception as e:
            results.append({"symbol": order.symbol, "status": "failed", "error": str(e)})
            
    return json.dumps({"execution_results": results})

class ExecutionAgent:
    def __init__(self, alpaca_api_key: str, alpaca_secret_key: str, paper: bool = True):
        global alpaca_service
        alpaca_service = AlpacaService(alpaca_api_key, alpaca_secret_key, paper)
        
        self.agent = Agent(
            name="ExecutionAgent",
            instructions="""
            You are an Execution Agent responsible for executing trades on the Alpaca platform.
            You receive portfolio rebalancing instructions (target weights or specific orders).
            
            Your capabilities:
            1. Check account status (buying power, cash).
            2. Check current positions.
            3. Execute a batch of orders.
            
            When receiving a 'target_portfolio' (weights), you should:
            1. Get current positions.
            2. Calculate the difference (orders needed) to reach the target weights.
            3. Execute the necessary buy/sell orders.
            
            Always check for sufficient buying power before placing buy orders.
            Report back the execution results.
            """,
            tools=[get_account_summary, get_current_positions, execute_orders]
        )

    def run(self, instruction: str, context: Optional[Dict] = None):
        return Runner.run_sync(self.agent, instruction, context=context)

