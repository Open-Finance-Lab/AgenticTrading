
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import inspect
from typing import Any, Dict, Optional, Callable, List, Union

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Orchestrator")

# Add agent paths to sys.path
current_dir = Path(__file__).parent
agent_pools_dir = current_dir.parent / "agent_pools"

sys.path.append(str(agent_pools_dir))
sys.path.append(str(agent_pools_dir / "alpha_agent_pool"))
sys.path.append(str(agent_pools_dir / "alpha_agent_demo"))
sys.path.append(str(agent_pools_dir / "risk_agent_demo"))
sys.path.append(str(agent_pools_dir / "portfolio_agent_demo"))
sys.path.append(str(agent_pools_dir / "execution_agent_demo" / "execution_agent_demo"))
sys.path.append(str(agent_pools_dir / "backtest_agent"))

# Import Agents
try:
    from alpha_signal_agent import AlphaSignalAgent
    from risk_signal_agent import RiskSignalAgent
    from portfolio_agent import PortfolioAgent
    from execution_agent import ExecutionAgent
    from backtest_agent import BacktestAgent
except ImportError as e:
    logger.error(f"Failed to import agents: {e}")
    sys.exit(1)

# Import OpenAI Agents SDK
try:
    from agents import Agent, Runner, function_tool
except ImportError:
    logger.error("openai-agents-sdk not found. Please install it.")
    sys.exit(1)

# ------------------------------------------------------------------------------
# Data Client
# ------------------------------------------------------------------------------
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    logger.warning("alpaca-py not installed. Data fetching will be mocked.")
    StockHistoricalDataClient = None

# ------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------
class Orchestrator:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not found. Agents might fail.")
            
        # Initialize Sub-Agents
        # Updated to use GPT-4o as requested
        self.alpha_agent = AlphaSignalAgent(name="AlphaCore", model="gpt-4o")
        self.risk_agent = RiskSignalAgent(name="RiskCore", model="gpt-4o")
        self.portfolio_agent = PortfolioAgent(name="PortfolioCore", model="gpt-4o")
        
        self.execution_agent = ExecutionAgent(
            alpaca_api_key=self.api_key, 
            alpaca_secret_key=self.secret_key, 
            paper=True
        )
        self.backtest_agent = BacktestAgent()
        
        # Data Client
        if StockHistoricalDataClient and self.api_key:
            self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        else:
            self.data_client = None
            
        # Pipeline Context (Shared Memory)
        self.pipeline_context = {}
        
        # Initialize Manager with Agent-as-Tool pattern
        self._initialize_manager_agent()

    def _patch_agent_to_support_as_tool(self, agent_instance):
        """
        Monkey-patch the agent instance to support 'as_tool' 
        if it was initialized with the fallback Agent class.
        """
        if not hasattr(agent_instance, 'as_tool'):
            # Bind the as_tool method from our local Agent class to this instance
            import types
            agent_instance.as_tool = types.MethodType(Agent.as_tool, agent_instance)

    def _initialize_manager_agent(self):
        """Initialize the Manager Agent using the Agent-as-Tool pattern"""
        
        # 1. Prepare Sub-Agents
        # We need to patch them because they might be using the minimal Agent class from their own files
        self._patch_agent_to_support_as_tool(self.alpha_agent.agent)
        self._patch_agent_to_support_as_tool(self.risk_agent.agent)
        self._patch_agent_to_support_as_tool(self.portfolio_agent.agent)
        self._patch_agent_to_support_as_tool(self.execution_agent.agent)
        self._patch_agent_to_support_as_tool(self.backtest_agent) # BacktestAgent inherits from Agent
        
        # 2. Define Helper Tools (Hosted Tools Equivalent)
        @function_tool
        def fetch_market_data(symbol: str, start_date: str, end_date: str) -> str:
            """Fetch and store market data in shared context."""
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                data = self.fetch_data(symbol, start_dt, end_dt)
                
                self.pipeline_context['market_data'] = data
                self.pipeline_context['symbol'] = symbol
                self.pipeline_context['dates'] = (start_date, end_date)
                
                return f"Data fetched for {symbol}. stored in context."
            except Exception as e:
                return f"Error: {e}"

        # 3. Create Manager Agent with Agents as Tools
        # This matches the "Agent as Tool" pattern from the docs
        
        self.manager_agent = Agent(
            name="OrchestratorAgent",
            instructions=(
                "You are a trading strategy manager. You use the tools given to you to execute the pipeline. "
                "1. Fetch data first. "
                "2. Ask Alpha Agent to analyze. "
                "3. Ask Risk Agent to assess. "
                "4. Ask Portfolio Agent to construct portfolio. "
                "5. Ask Execution Agent to trade OR Backtest Agent to simulate."
            ),
            tools=[
                # Helper Tool
                fetch_market_data,
                
                # Agents as Tools
                self.alpha_agent.agent.as_tool(
                    tool_name="ask_alpha_agent",
                    tool_description="Ask Alpha Agent to generate signals. Requires market data."
                ),
                self.risk_agent.agent.as_tool(
                    tool_name="ask_risk_agent",
                    tool_description="Ask Risk Agent to assess market risks."
                ),
                self.portfolio_agent.agent.as_tool(
                    tool_name="ask_portfolio_agent",
                    tool_description="Ask Portfolio Agent to generate target weights."
                ),
                self.execution_agent.agent.as_tool(
                    tool_name="ask_execution_agent",
                    tool_description="Ask Execution Agent to execute trades."
                ),
                self.backtest_agent.as_tool(
                    tool_name="ask_backtest_agent",
                    tool_description="Ask Backtest Agent to run simulation."
                )
            ]
        )

    def fetch_data(self, symbols: Union[str, List[str]], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical data from Alpaca or Mock"""
        if isinstance(symbols, str):
            symbols = [symbols]
            
        logger.info(f"Fetching data for {symbols} from {start_date} to {end_date}")
        
        if self.data_client:
            try:
                request_params = StockBarsRequest(
                    symbol_or_symbols=symbols,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date
                )
                bars = self.data_client.get_stock_bars(request_params)
                df = bars.df.reset_index()
                df = df.rename(columns={'timestamp': 'date'})
                df['date'] = df['date'].dt.tz_localize(None)
                return df
            except Exception as e:
                logger.error(f"Alpaca data fetch failed: {e}. Using fallback.")
        
        # Try YFinance (Real Data Fallback)
        try:
            import yfinance as yf
            import logging
            # Suppress yfinance noise
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            
            all_dfs = []
            for symbol in symbols:
                # Fetch data
                try:
                    df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
                    
                    if df.empty:
                        continue
                        
                    # Handle MultiIndex columns (yfinance > 0.2)
                    if isinstance(df.columns, pd.MultiIndex):
                        if df.columns.nlevels > 1:
                            df.columns = df.columns.droplevel(1)
                            
                    df = df.reset_index()
                    # Standardize columns
                    df.columns = [str(c).lower() for c in df.columns]
                    
                    # Ensure 'date' column
                    if 'date' not in df.columns:
                        date_cols = [c for c in df.columns if 'date' in c]
                        if date_cols:
                            df = df.rename(columns={date_cols[0]: 'date'})
                    
                    # Keep only standard columns to ensure clean concat
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    valid_cols = [c for c in required_cols if c in df.columns]
                    
                    if 'date' in df.columns:
                        df = df[['date'] + valid_cols].copy()
                        df['symbol'] = symbol
                        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                        all_dfs.append(df)
                        
                except Exception as e:
                    logger.warning(f"Failed to process {symbol}: {e}")
                    continue
            
            if all_dfs:
                logger.info(f"✅ Successfully fetched real data via yfinance for {len(all_dfs)} symbols")
                return pd.concat(all_dfs, ignore_index=True)
                
        except ImportError:
            logger.warning("yfinance not installed. Falling back to mock data.")
        except Exception as e:
            logger.error(f"yfinance fetch failed: {e}")
        
        # Mock Data
        all_dfs = []
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n = len(dates)
        
        for symbol in symbols:
            # Generate regime-switching price path to force signal changes
            # Include start_date in seed to ensure different data for different periods
            seed_val = (hash(symbol) + int(start_date.timestamp())) % (2**32)
            np.random.seed(seed_val)
            
            # Construct regimes
            p1 = n // 3
            p2 = n // 3
            p3 = n - p1 - p2
            
            # Regime 1: Bullish (Low Vol)
            r1 = np.random.normal(0.001, 0.01, p1)
            # Regime 2: Bearish (High Vol)
            r2 = np.random.normal(-0.0015, 0.02, p2)
            # Regime 3: Rebound (Med Vol)
            r3 = np.random.normal(0.0005, 0.015, p3)
            
            rets = np.concatenate([r1, r2, r3])
            price = 100.0 * np.cumprod(1 + rets)
            
            df = pd.DataFrame({
                'date': dates, 'symbol': symbol,
                'open': price, 'high': price*1.01, 'low': price*0.99, 'close': price,
                'volume': np.random.randint(1000, 10000, n)
            })
            all_dfs.append(df)
            
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    def run_pipeline(self, symbol: Union[str, List[str]], start_date: str, end_date: str, mode: str = "backtest",
                     total_capital: float = 100000.0, rebalance_freq: int = 5) -> Dict[str, Any]:
        """
        Run the complete investment pipeline: Data -> Alpha -> Risk -> Portfolio -> Backtest
        """
        # Handle both string and list of symbols
        symbols = [symbol] if isinstance(symbol, str) else symbol
        symbol_str = ", ".join(symbols)
        
        logger.info(f"Running pipeline for {symbol_str} from {start_date} to {end_date}")
        
        # 1. Data Fetching
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # ROLLING WINDOW: Fetch 1 year prior for training to avoid leakage
            lookback_days = 365
            fetch_start_dt = start_dt - timedelta(days=lookback_days)
            logger.info(f"Fetching extended data from {fetch_start_dt.date()} to {end_dt.date()} for Rolling Training...")
            
            full_data = self.fetch_data(symbols, fetch_start_dt, end_dt)
            
            if full_data.empty:
                return {"status": "error", "message": "No data fetched"}
            
            # Split into Train/Test
            if 'date' in full_data.columns:
                full_data['date'] = pd.to_datetime(full_data['date']).dt.tz_localize(None)
                train_data = full_data[full_data['date'] < start_dt].copy()
                test_data = full_data[full_data['date'] >= start_dt].copy()
                
                logger.info(f"Data Split - Train: {len(train_data)} rows ({fetch_start_dt.date()} to {start_dt.date()}), Test: {len(test_data)} rows ({start_dt.date()} to {end_dt.date()})")
            else:
                logger.warning("Date column missing. Using full data as test (Risk of Leakage).")
                train_data = None
                test_data = full_data

            # 2. Alpha Generation
            # Define some default factors/indicators for the demo
            factors = [
                {"factor_name": "momentum_20", "factor_type": "technical", "calculation_method": "expression", "expression": "close / Ref(close, 20) - 1", "lookback_period": 20}
            ]
            indicators = ["RSI", "MACD", "Bollinger"]
            
            alpha_result = self.alpha_agent.generate_signals_from_data(
                data=test_data, # Predict on Test
                factors=factors, 
                indicators=indicators, 
                model_type="linear", 
                signal_threshold=0.0,
                train_data=train_data # Train on History
            )
            
            if alpha_result["status"] != "success":
                return {"status": "error", "message": f"Alpha generation failed: {alpha_result.get('message')}"}
            
            # 3. Risk Analysis
            risk_result = self.risk_agent.generate_risk_signals_from_data(test_data)
            
            if risk_result["status"] != "success":
                logger.warning(f"Risk analysis failed: {risk_result.get('message')}")
                
            # 4. Backtest
            # The backtest agent expects predictions as pd.Series with MultiIndex (datetime, instrument)
            predictions_dict = alpha_result.get("signals", {})
            
            # Convert dict to Series
            if not predictions_dict:
                 return {"status": "error", "message": "No signals generated"}
            
            signals_series = pd.Series(predictions_dict)
            
            # Ensure MultiIndex (date, symbol)
            if not isinstance(signals_series.index, pd.MultiIndex):
                # If index is just date (single symbol case)
                if len(symbols) == 1:
                    signals_series.index = pd.MultiIndex.from_product([signals_series.index, [symbols[0]]], names=['datetime', 'instrument'])
                else:
                    # If we have multiple symbols but single index, this implies AlphaAgent return format issue
                    # But AlphaAgent should return MultiIndex for multiple symbols
                    # We assume signals keys are already (date, symbol) tuples if multi-asset
                    signals_series.index.names = ['datetime', 'instrument']
            else:
                # Ensure names are correct
                signals_series.index.names = ['datetime', 'instrument']

            backtest_result = self.backtest_agent.run_simple_backtest_paper_interface(
                predictions=signals_series,
                start_time=start_date,
                end_time=end_date,
                investment_horizon=rebalance_freq,
                total_capital=total_capital,
                market_data=test_data, # Backtest on Test Data only
                plot_results=False 
            )
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def run_inference_rolling_week(self, symbol: Union[str, List[str]], start_date: str, end_date: str, 
                                  lookback_days: int = 365, rebalance_freq: int = 5, total_capital: float = 100000.0) -> Dict[str, Any]:
        """
        Run inference in a strict 'World Model' fashion (Rolling Week):
        - Incrementally reveal data week by week.
        - Retrain/Predict for the upcoming week.
        - Simulate weekly rebalancing.
        """
        symbols = [symbol] if isinstance(symbol, str) else symbol
        logger.info(f"🚀 Starting World Model Inference (Rolling Week) for {symbols} ({start_date} to {end_date})")
        
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # 1. Fetch All Data (The "World" has data, but Agent sees it incrementally)
            fetch_start = start_dt - timedelta(days=lookback_days)
            full_data = self.fetch_data(symbols, fetch_start, end_dt)
            
            if full_data.empty or 'date' not in full_data.columns:
                return {"status": "error", "message": "Data fetch failed"}
                
            full_data['date'] = pd.to_datetime(full_data['date']).dt.tz_localize(None)
            
            # 2. Weekly Loop
            current_dt = start_dt
            all_signals_list = []
            
            while current_dt < end_dt:
                next_week_dt = current_dt + timedelta(weeks=1)
                if next_week_dt > end_dt: next_week_dt = end_dt
                
                # Define Data Slices
                # Train: [current_dt - lookback, current_dt)
                # Test: [current_dt, next_week_dt)
                
                train_start = current_dt - timedelta(days=lookback_days)
                train_data = full_data[
                    (full_data['date'] >= train_start) & 
                    (full_data['date'] < current_dt)
                ].copy()
                
                test_data = full_data[
                    (full_data['date'] >= current_dt) & 
                    (full_data['date'] < next_week_dt)
                ].copy()
                
                if test_data.empty:
                    current_dt = next_week_dt
                    continue
                    
                logger.info(f"📅 Rolling Step: {current_dt.date()} -> {next_week_dt.date()} (Train: {len(train_data)}, Test: {len(test_data)})")
                
                # Generate Signals
                factors = [
                    {"factor_name": "momentum_20", "factor_type": "technical", "calculation_method": "expression", "expression": "close / Ref(close, 4) - 1", "lookback_period": 4}
                ]
                indicators = ["RSI", "MACD", "Bollinger"]
                
                alpha_result = self.alpha_agent.generate_signals_from_data(
                    data=test_data, 
                    train_data=train_data,
                    factors=factors,
                    indicators=indicators,
                    model_type="linear",
                    signal_threshold=0.0
                )
                
                if alpha_result['status'] == 'success':
                    sigs = alpha_result.get('signals', {})
                    if sigs:
                        sig_series = pd.Series(sigs)
                        all_signals_list.append(sig_series)
                
                # Step Forward
                current_dt = next_week_dt
                
            # 3. Aggregate & Backtest
            if not all_signals_list:
                return {"status": "error", "message": "No signals generated during rolling inference"}
                
            # Concat all weekly signal series
            full_signals_series = pd.concat(all_signals_list)
            
            # Ensure MultiIndex
            if not isinstance(full_signals_series.index, pd.MultiIndex):
                 if len(symbols) == 1:
                     full_signals_series.index = pd.MultiIndex.from_product([full_signals_series.index, [symbols[0]]], names=['datetime', 'instrument'])
                 else:
                     full_signals_series.index.names = ['datetime', 'instrument']
            else:
                 full_signals_series.index.names = ['datetime', 'instrument']

            # Filter market_data to start_date -> end_date
            backtest_market_data = full_data[full_data['date'] >= start_dt].copy()
            
            backtest_result = self.backtest_agent.run_simple_backtest_paper_interface(
                predictions=full_signals_series,
                start_time=start_date,
                end_time=end_date,
                investment_horizon=rebalance_freq,
                total_capital=total_capital,
                market_data=backtest_market_data,
                plot_results=False
            )
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Rolling inference error: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def optimize_agent_prompts(self, agent_name: str, performance_metric: str, current_value: float, target_value: float) -> str:
        """
        Optimize agent instructions using a Meta-Agent approach.
        """
        logger.info(f"Optimizing prompts for {agent_name}. Current {performance_metric}: {current_value:.2f}, Target: {target_value}")
        
        # Identify the agent
        target_agent = None
        if agent_name == "Alpha":
            target_agent = self.alpha_agent.agent
        elif agent_name == "Risk":
            target_agent = self.risk_agent.agent
        elif agent_name == "Portfolio":
            target_agent = self.portfolio_agent.agent
            
        if not target_agent:
            logger.error(f"Agent {agent_name} not found")
            return "Optimization failed: Agent not found"
            
        # Construct the optimization prompt
        meta_prompt = f"""
        You are a Meta-Agent optimizing trading agents.
        The {agent_name} Agent is underperforming.
        Metric: {performance_metric}
        Current Value: {current_value}
        Target Value: {target_value}
        
        Current Instructions:
        {target_agent.instructions}
        
        Please rewrite the instructions to improve performance. Focus on:
        1. More robust signal generation
        2. Better risk management
        3. Adapting to market conditions
        
        Return ONLY the new instructions.
        """
        
        # Call LLM if available
        new_instructions = target_agent.instructions # Default to current
        
        if hasattr(target_agent, 'client') and target_agent.client:
            try:
                response = target_agent.client.chat.completions.create(
                    model="gpt-4o", # Use strong model for optimization
                    messages=[{"role": "user", "content": meta_prompt}]
                )
                new_instructions = response.choices[0].message.content
                logger.info("Prompts optimized using LLM")
            except Exception as e:
                logger.warning(f"LLM optimization failed: {e}. Appending refinement rule.")
                new_instructions += f"\n\nRefinement ({datetime.now().date()}): Be more conservative in signal generation when volatility is high."
        else:
            logger.info("LLM not available. Appending refinement rule.")
            new_instructions += f"\n\nRefinement ({datetime.now().date()}): Focus on trend-following and reduce position size during drawdowns."
            
        # Update the agent's instructions
        target_agent.instructions = new_instructions
        return new_instructions

    def run_agentic_pipeline(self, user_request: str):
        """Run the pipeline using the Manager Agent"""
        logger.info(f"Manager Agent processing: {user_request}")
        return Runner.run_sync(self.manager_agent, user_request)

if __name__ == "__main__":
    orchestrator = Orchestrator()
    
    print("\n--- Agentic Pipeline Demo (Agent-as-Tool Pattern) ---")
    # Simulate a user request that triggers the agents
    request = "Fetch data for AAPL, MSFT (2023-01-01 to 2023-06-01) and then ask Alpha Agent to analyze it."
    
    result = orchestrator.run_agentic_pipeline(request)
    print("\nFinal Result:")
    print(result)
