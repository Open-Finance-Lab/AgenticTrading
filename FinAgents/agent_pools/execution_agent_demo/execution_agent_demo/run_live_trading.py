
import os
import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables from .env file
current_dir = Path(__file__).parent.resolve()
load_dotenv(current_dir / ".env")

# Add sibling directories to path to import other agents
current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
portfolio_agent_dir = parent_dir / "portfolio_agent_demo"
sys.path.append(str(portfolio_agent_dir))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LiveTrading")

# Import Execution Agent
try:
    from execution_agent import ExecutionAgent
except ImportError:
    sys.path.append(str(current_dir))
    from execution_agent import ExecutionAgent

# Import PortfolioAgent
try:
    from portfolio_agent import PortfolioAgent
except ImportError as e:
    logger.warning(f"Could not import PortfolioAgent: {e}. Using Mock.")
    class PortfolioAgent:
        def __init__(self, mode="backtest"): pass
        def inference(self, alpha_signals, risk_signals, transaction_costs, current_portfolio=None):
            # Mock balanced portfolio
            return {
                "target_weights": {k: 1.0/len(alpha_signals) for k in alpha_signals if alpha_signals[k] > 0},
                "risk_adjustment": {"risk_level": "LOW"}
            }

# Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PK_MOCK_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "SK_MOCK_SECRET")
PAPER_TRADING = True
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

def fetch_market_data(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch current market price for symbols.
    Uses Alpaca Market Data API if available.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        
        if ALPACA_API_KEY and ALPACA_SECRET_KEY and not ALPACA_API_KEY.startswith("PK_MOCK"):
            client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            request_params = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = client.get_stock_latest_quote(request_params)
            
            prices = {}
            for symbol, quote in quotes.items():
                prices[symbol] = quote.ask_price # Use ask price as proxy for buy
            return prices
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to fetch real data: {e}")

    # Mock prices for demo if no API key or failure
    import random
    return {s: 100.0 + random.uniform(-5, 5) for s in symbols}

def generate_alpha_signals(prices: Dict[str, float]) -> Dict[str, float]:
    """
    Generate simple alpha signals based on random noise (Mock) 
    or simple logic for demo purposes.
    """
    import random
    # Mock: Random alpha between -0.05 and 0.05
    return {s: random.uniform(-0.05, 0.05) for s in prices}

def generate_risk_signals() -> Dict[str, Any]:
    """
    Generate mock risk signals.
    """
    return {
        "overall_risk_level": "LOW",
        "risk_score": 0.1
    }

def main():
    logger.info("Starting Live Trading Demo...")
    
    # Initialize Agents
    exec_agent = ExecutionAgent(ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING)
    try:
        port_agent = PortfolioAgent(mode="paper_trading")
    except TypeError:
        # Handle case if PortfolioAgent doesn't accept mode (mock or old version)
        port_agent = PortfolioAgent()
    
    logger.info("Agents initialized.")
    
    try:
        while True:
            logger.info("\n--- Starting Trading Cycle ---")
            
            # 1. Fetch Data
            market_prices = fetch_market_data(SYMBOLS)
            logger.info(f"Market Data: {market_prices}")
            
            # 2. Generate Signals
            alpha_signals = generate_alpha_signals(market_prices)
            risk_signals = generate_risk_signals()
            
            logger.info(f"Alpha Signals: {json.dumps(alpha_signals)}")
            
            # 3. Portfolio Construction
            transaction_costs = {"fixed_cost": 1.0, "slippage": 0.0001}
            portfolio_result = port_agent.inference(
                alpha_signals=alpha_signals,
                risk_signals=risk_signals,
                transaction_costs=transaction_costs
            )
            
            target_weights = portfolio_result.get("target_weights", {})
            logger.info(f"Target Portfolio Weights: {json.dumps(target_weights)}")
            
            if not target_weights:
                logger.info("No target weights generated. Skipping execution.")
                time.sleep(5)
                continue

            # 4. Execution
            # Convert weights to instructions or pass directly
            # Logic: "Adjust portfolio to match these weights"
            
            # For this demo, we'll construct a prompt for the Execution Agent
            # But since we have tools, we can also calculate the diff here or let the agent do it.
            # The Execution Agent has `get_current_positions` and `execute_orders`.
            # We will ask it to "Rebalance portfolio to these weights: ..."
            
            instruction = f"""
            Current market prices are: {json.dumps(market_prices)}.
            Target portfolio weights are: {json.dumps(target_weights)}.
            Total capital should be based on current equity.
            Please rebalance the portfolio to match these targets.
            1. Check current positions and equity.
            2. Calculate target quantity for each asset (Target Value / Price).
            3. Execute BUY/SELL orders to reach targets.
            """
            
            logger.info("Sending instruction to Execution Agent...")
            result = exec_agent.run(instruction)
            logger.info(f"Execution Result: {result}")
            
            # Check Account
            # exec_agent.run("Check account status")
            
            logger.info("--- Cycle Complete. Waiting... ---")
            time.sleep(10) # Run every 10 seconds for demo
            
            # Break for single run demo (loop can be enabled)
            break 

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.error(f"Error in trading loop: {e}", exc_info=True)

if __name__ == "__main__":
    main()

