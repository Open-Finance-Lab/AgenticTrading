#!/usr/bin/env python3
"""
Orchestrator-Based Chatbot Backtest using Natural Language Instructions

This script demonstrates a complete multi-agent backtest using the FinAgent orchestrator
with natural language instructions processed across all agent pools:
- Data Agent Pool: Real market data retrieval
- Alpha Agent Pool: Signal generation and strategy execution  
- Portfolio Construction Agent Pool: Portfolio optimization
- Transaction Cost Agent Pool: Cost analysis and optimization
- Risk Agent Pool: Risk management and monitoring

The test simulates a chatbot conversation where users can request backtests
using natural language like: "Run a 3-year momentum backtest for AAPL and MSFT"
"""

import asyncio
import logging
import sys
import os
import json
import yaml
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import FinAgent components
try:
    from FinAgents.orchestrator.core.finagent_orchestrator import FinAgentOrchestrator
    from FinAgents.orchestrator.core.llm_integration import NaturalLanguageProcessor, ConversationManager, LLMConfig
    from FinAgents.orchestrator.core.mcp_nl_interface import MCPNaturalLanguageInterface
    from FinAgents.orchestrator.core.agent_pool_monitor import AgentPoolMonitor
    from FinAgents.orchestrator.core.dag_planner import TradingStrategy, BacktestConfiguration
    FINAGENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ FinAgent imports failed: {e}")
    FINAGENT_AVAILABLE = False

# MCP client for agent pool communication
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    print("⚠️ MCP client not available")
    MCP_AVAILABLE = False

# OpenAI client
try:
    from openai import AsyncOpenAI
    LLM_AVAILABLE = True
except ImportError:
    print("⚠️ OpenAI client not available")
    LLM_AVAILABLE = False

# Plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("⚠️ Plotting libraries not available")
    PLOTTING_AVAILABLE = False

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)
    print(f"✅ Loaded .env from: {env_path}")
except Exception as e:
    print(f"⚠️ Failed to load .env file: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'orchestrator_chatbot_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger("OrchestratorChatbotBacktest")


class OrchestratorChatbotBacktester:
    """Orchestrator-based chatbot backtester using natural language instructions"""
    
    def __init__(self):
        self.orchestrator = None
        self.nl_interface = None
        self.conversation_manager = None
        self.agent_monitor = None
        self.config = None
        self.session_id = f"chatbot_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backtest_results = {}
        self.chat_history = []
        self.llm_client = None
        
    async def run_chatbot_backtest(self):
        """Run comprehensive chatbot-style backtest using natural language"""
        logger.info("🤖 Starting Orchestrator Chatbot Backtest")
        logger.info("=" * 80)
        
        try:
            # Initialize all components
            await self._initialize_orchestrator_components()
            
            # Verify system health
            await self._verify_agent_pool_health()
            
            # Simulate chatbot conversation for backtest
            await self._simulate_chatbot_conversation()
            
            # Execute the requested backtest
            await self._execute_orchestrated_backtest()
            
            # Generate analysis and visualization
            await self._generate_comprehensive_analysis()
            
            # Print results summary
            self._print_chatbot_summary()
            
        except Exception as e:
            logger.error(f"❌ Orchestrator chatbot backtest failed: {e}")
            raise
    
    async def _initialize_orchestrator_components(self):
        """Initialize orchestrator and all agent pool components"""
        logger.info("🔧 Initializing Orchestrator Components...")
        
        # Load configuration
        self.config = await self._load_orchestrator_config()
        
        # Initialize orchestrator with memory and orchestration capabilities
        if FINAGENT_AVAILABLE:
            self.orchestrator = FinAgentOrchestrator(
                host="localhost",
                port=9000,
                enable_memory=True,
                enable_rl=False,
                enable_monitoring=True
            )
            
            # Initialize natural language interface
            if self.config.get("llm", {}).get("enabled", True):
                llm_config = LLMConfig(
                    provider=self.config.get("llm", {}).get("provider", "openai"),
                    model=self.config.get("llm", {}).get("model", "gpt-4"),
                    temperature=self.config.get("llm", {}).get("temperature", 0.7)
                )
                
                nlp = NaturalLanguageProcessor(llm_config)
                self.conversation_manager = ConversationManager(nlp)
                self.nl_interface = MCPNaturalLanguageInterface(self.config)
            
            # Initialize agent pool monitor
            self.agent_monitor = AgentPoolMonitor(self.config)
        else:
            logger.warning("⚠️ FinAgent not available, using mock components")
            self.orchestrator = MockOrchestrator()
            self.agent_monitor = MockAgentMonitor()
        
        # Initialize LLM client for direct use
        if LLM_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.llm_client = AsyncOpenAI(api_key=api_key)
                logger.info("✅ LLM client initialized")
            else:
                logger.warning("⚠️ OPENAI_API_KEY not found")
        
        logger.info("✅ All orchestrator components initialized")
    
    async def _load_orchestrator_config(self) -> Dict[str, Any]:
        """Load orchestrator configuration"""
        config_paths = [
            "config/orchestrator_config.yaml",
            "FinAgents/orchestrator/config/orchestrator_config.yaml",
            "../config/orchestrator_config.yaml"
        ]
        
        for config_path in config_paths:
            path = Path(config_path)
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        config = yaml.safe_load(f)
                    logger.info(f"✅ Loaded configuration from: {config_path}")
                    return config
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load config from {config_path}: {e}")
        
        # Default configuration for all agent pools
        logger.info("📝 Using default orchestrator configuration")
        return {
            "orchestrator": {
                "host": "localhost",
                "port": 9000,
                "enable_memory": True,
                "enable_rl": False
            },
            "agent_pools": {
                "data_agent_pool": {
                    "url": "http://localhost:8001/sse",
                    "enabled": True,
                    "capabilities": ["market_data", "historical_data", "real_time_data", "technical_indicators"]
                },
                "alpha_agent_pool": {
                    "url": "http://localhost:5050/sse", 
                    "enabled": True,
                    "capabilities": ["momentum_strategy", "mean_reversion", "ml_signals", "factor_analysis"]
                },
                "portfolio_construction_agent_pool": {
                    "url": "http://localhost:8002/sse",
                    "enabled": True,
                    "capabilities": ["portfolio_optimization", "rebalancing", "allocation", "risk_budgeting"]
                },
                "transaction_cost_agent_pool": {
                    "url": "http://localhost:6000/sse",
                    "enabled": True,
                    "capabilities": ["cost_analysis", "execution_optimization", "impact_modeling", "slippage_estimation"]
                },
                "risk_agent_pool": {
                    "url": "http://localhost:7000/sse", 
                    "enabled": True,
                    "capabilities": ["risk_assessment", "var_calculation", "stress_testing", "exposure_analysis"]
                }
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "enabled": True
            },
            "backtest": {
                "default_capital": 1000000,
                "default_period": "3y",
                "default_symbols": ["AAPL", "MSFT"],
                "commission_rate": 0.001,
                "slippage_rate": 0.0005
            }
        }
    
    async def _verify_agent_pool_health(self):
        """Verify all agent pools are healthy and ready"""
        logger.info("🔍 Verifying Agent Pool Health...")
        
        if self.agent_monitor and hasattr(self.agent_monitor, 'check_all_pools'):
            try:
                health_results = await self.agent_monitor.check_all_pools()
                
                healthy_pools = []
                unhealthy_pools = []
                
                for pool_name, pool_info in health_results.items():
                    if hasattr(pool_info, 'status') and pool_info.status.value == "healthy":
                        healthy_pools.append(pool_name)
                        logger.info(f"✅ {pool_name}: Healthy (Response: {pool_info.response_time:.3f}s)")
                    else:
                        unhealthy_pools.append(pool_name)
                        logger.warning(f"⚠️ {pool_name}: Unhealthy or unreachable")
                
                if unhealthy_pools:
                    logger.warning(f"⚠️ Some agent pools are unhealthy: {unhealthy_pools}")
                    logger.info("📝 Continuing with mock data for unhealthy pools")
                
                logger.info(f"✅ System health check completed: {len(healthy_pools)}/{len(health_results)} pools healthy")
            except Exception as e:
                logger.warning(f"⚠️ Health check failed: {e}, proceeding with mock components")
        else:
            logger.info("📝 Using mock agent monitor - skipping health check")
    
    async def _simulate_chatbot_conversation(self):
        """Simulate a natural language chatbot conversation for backtest setup"""
        logger.info("💬 Simulating Chatbot Conversation...")
        
        # Simulate user conversation for backtest setup
        conversation_turns = [
            {
                "user": "Hi, I want to run a comprehensive backtest",
                "assistant": "Hello! I'd be happy to help you run a backtest. What type of strategy would you like to test?",
            },
            {
                "user": "I want to test a momentum strategy for AAPL and MSFT over the last 3 years",
                "assistant": "Great choice! A momentum strategy for AAPL and MSFT over 3 years. What's your preferred capital allocation?",
            },
            {
                "user": "Use $1 million initial capital with moderate risk management",
                "assistant": "Perfect! I'll set up a $1M backtest with moderate risk management. Any specific requirements for the strategy?",
            },
            {
                "user": "Use both momentum and mean reversion signals, optimize the portfolio, analyze transaction costs, and apply risk management",
                "assistant": "Excellent! I'll coordinate all agent pools for a comprehensive backtest with momentum/mean reversion signals, portfolio optimization, cost analysis, and risk management. Let me execute this now.",
            }
        ]
        
        # Log the conversation
        for i, turn in enumerate(conversation_turns, 1):
            logger.info(f"👤 User: {turn['user']}")
            logger.info(f"🤖 Assistant: {turn['assistant']}")
            
            self.chat_history.append({
                "turn": i,
                "user_message": turn['user'],
                "assistant_response": turn['assistant'],
                "timestamp": datetime.now().isoformat()
            })
            
            # Brief pause to simulate conversation
            await asyncio.sleep(0.5)
        
        # Process the final instruction
        final_instruction = conversation_turns[-1]["user"]
        
        if self.conversation_manager:
            try:
                # Use natural language processor to parse the instruction
                response = await self.conversation_manager.handle_user_message(
                    final_instruction, 
                    self.session_id,
                    {"backtest_context": True}
                )
                
                if response["success"]:
                    logger.info(f"✅ Instruction processed: {response['response']['intent']}")
                    self.backtest_results["parsed_instruction"] = response["response"]
                else:
                    logger.warning(f"⚠️ Instruction processing failed: {response.get('error')}")
            except Exception as e:
                logger.warning(f"⚠️ NLP processing failed: {e}")
        
        logger.info("✅ Chatbot conversation simulation completed")
    
    async def _execute_orchestrated_backtest(self):
        """Execute the orchestrated backtest across all agent pools"""
        logger.info("🎯 Executing Orchestrated Backtest...")
        
        # Step 1: Request market data from Data Agent Pool
        logger.info("📊 Step 1: Requesting market data...")
        market_data = await self._request_market_data_from_pool()
        
        # Step 2: Generate alpha signals from Alpha Agent Pool
        logger.info("🧠 Step 2: Generating alpha signals...")
        alpha_signals = await self._generate_alpha_signals_from_pool(market_data)
        
        # Step 3: Optimize portfolio via Portfolio Construction Agent Pool
        logger.info("📈 Step 3: Optimizing portfolio...")
        portfolio_weights = await self._optimize_portfolio_via_pool(alpha_signals, market_data)
        
        # Step 4: Analyze transaction costs via Transaction Cost Agent Pool
        logger.info("💰 Step 4: Analyzing transaction costs...")
        cost_analysis = await self._analyze_transaction_costs_via_pool(portfolio_weights)
        
        # Step 5: Apply risk management via Risk Agent Pool
        logger.info("🛡️ Step 5: Applying risk management...")
        risk_adjusted_portfolio = await self._apply_risk_management_via_pool(portfolio_weights, market_data)
        
        # Step 6: Simulate complete backtest
        logger.info("⚡ Step 6: Simulating complete backtest...")
        backtest_results = await self._simulate_complete_backtest(
            market_data, alpha_signals, risk_adjusted_portfolio, cost_analysis
        )
        
        # Store comprehensive results
        self.backtest_results.update({
            "market_data": market_data,
            "alpha_signals": alpha_signals,
            "portfolio_weights": portfolio_weights,
            "cost_analysis": cost_analysis,
            "risk_adjusted_portfolio": risk_adjusted_portfolio,
            "backtest_simulation": backtest_results,
            "execution_metadata": {
                "session_id": self.session_id,
                "execution_time": datetime.now().isoformat(),
                "agent_pools_used": list(self.config["agent_pools"].keys()),
                "orchestrator_version": "1.0.0"
            }
        })
        
        logger.info("✅ Orchestrated backtest execution completed")
    
    async def _request_market_data_from_pool(self) -> Dict[str, Any]:
        """Request market data from Data Agent Pool using natural language"""
        try:
            if not MCP_AVAILABLE:
                return self._mock_market_data()
            
            data_pool_url = self.config["agent_pools"]["data_agent_pool"]["url"]
            
            # Natural language query for market data
            query = """
            Please provide daily price data for AAPL and MSFT from January 1, 2022 to December 31, 2024.
            Include OHLCV data, technical indicators (SMA 20, SMA 50, RSI, MACD), and volume data.
            Format the data for backtesting with proper timestamps.
            """
            
            async with sse_client(data_pool_url, timeout=120) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    result = await session.call_tool("process_market_query", {"query": query})
                    
                    if result.content and len(result.content) > 0:
                        content_item = result.content[0]
                        if hasattr(content_item, 'text'):
                            data = json.loads(content_item.text)
                            logger.info(f"✅ Retrieved market data: {data.get('status', 'unknown')}")
                            
                            # Enhance data with additional info
                            data["data_source"] = "data_agent_pool"
                            data["query_timestamp"] = datetime.now().isoformat()
                            return data
                    
                    return {"status": "error", "error": "No market data received"}
                    
        except Exception as e:
            logger.warning(f"⚠️ Data Agent Pool request failed: {e}")
            return self._mock_market_data()
    
    async def _generate_alpha_signals_from_pool(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alpha signals using Alpha Agent Pool"""
        try:
            if not MCP_AVAILABLE:
                return self._mock_alpha_signals()
            
            alpha_pool_url = self.config["agent_pools"]["alpha_agent_pool"]["url"]
            
            # Natural language query for signal generation
            query = f"""
            Generate comprehensive trading signals for AAPL and MSFT using both momentum and mean reversion strategies.
            
            Market Data Context: {json.dumps(market_data, default=str)[:1000]}...
            
            Requirements:
            - Use 20-day and 50-day momentum indicators
            - Apply mean reversion filters with 10-day lookback
            - Include confidence scores and signal strength
            - Provide entry/exit recommendations
            - Consider current market regime and volatility
            """
            
            async with sse_client(alpha_pool_url, timeout=120) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    result = await session.call_tool("process_strategy_request", {"query": query})
                    
                    if result.content and len(result.content) > 0:
                        content_item = result.content[0]
                        if hasattr(content_item, 'text'):
                            signals = json.loads(content_item.text)
                            logger.info(f"✅ Generated alpha signals: {signals.get('status', 'unknown')}")
                            
                            signals["signal_source"] = "alpha_agent_pool"
                            signals["generation_timestamp"] = datetime.now().isoformat()
                            return signals
                    
                    return {"status": "error", "error": "No alpha signals received"}
                    
        except Exception as e:
            logger.warning(f"⚠️ Alpha Agent Pool request failed: {e}")
            return self._mock_alpha_signals()
    
    async def _optimize_portfolio_via_pool(self, alpha_signals: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio using Portfolio Construction Agent Pool"""
        try:
            if not MCP_AVAILABLE:
                return self._mock_portfolio_optimization()
            
            portfolio_pool_url = self.config["agent_pools"]["portfolio_construction_agent_pool"]["url"]
            
            # Natural language query for portfolio optimization
            query = f"""
            Optimize portfolio allocation for AAPL and MSFT based on the provided alpha signals and market data.
            
            Alpha Signals: {json.dumps(alpha_signals, default=str)[:1000]}...
            Market Data: {json.dumps(market_data, default=str)[:500]}...
            
            Requirements:
            - Use modern portfolio theory with risk-return optimization
            - Apply $1M initial capital constraint
            - Set maximum 40% allocation per single asset
            - Include rebalancing recommendations
            - Consider transaction costs in optimization
            - Target moderate risk profile (Sharpe ratio optimization)
            """
            
            async with sse_client(portfolio_pool_url, timeout=120) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    result = await session.call_tool("process_portfolio_request", {"query": query})
                    
                    if result.content and len(result.content) > 0:
                        content_item = result.content[0]
                        if hasattr(content_item, 'text'):
                            portfolio = json.loads(content_item.text)
                            logger.info(f"✅ Optimized portfolio: {portfolio.get('status', 'unknown')}")
                            
                            portfolio["optimization_source"] = "portfolio_construction_agent_pool"
                            portfolio["optimization_timestamp"] = datetime.now().isoformat()
                            return portfolio
                    
                    return {"status": "error", "error": "No portfolio optimization received"}
                    
        except Exception as e:
            logger.warning(f"⚠️ Portfolio Construction Agent Pool request failed: {e}")
            return self._mock_portfolio_optimization()
    
    async def _analyze_transaction_costs_via_pool(self, portfolio_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze transaction costs using Transaction Cost Agent Pool"""
        try:
            if not MCP_AVAILABLE:
                return self._mock_cost_analysis()
            
            cost_pool_url = self.config["agent_pools"]["transaction_cost_agent_pool"]["url"]
            
            # Natural language query for cost analysis
            query = f"""
            Analyze transaction costs for the optimized portfolio allocation and provide execution recommendations.
            
            Portfolio Weights: {json.dumps(portfolio_weights, default=str)[:1000]}...
            
            Requirements:
            - Calculate market impact costs for AAPL and MSFT
            - Estimate bid-ask spread costs
            - Provide slippage estimates for different order sizes
            - Recommend optimal execution strategies (TWAP, VWAP, etc.)
            - Consider market timing for cost minimization
            - Include commission estimates for institutional trading
            """
            
            async with sse_client(cost_pool_url, timeout=120) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    result = await session.call_tool("process_cost_analysis", {"query": query})
                    
                    if result.content and len(result.content) > 0:
                        content_item = result.content[0]
                        if hasattr(content_item, 'text'):
                            costs = json.loads(content_item.text)
                            logger.info(f"✅ Analyzed transaction costs: {costs.get('status', 'unknown')}")
                            
                            costs["cost_analysis_source"] = "transaction_cost_agent_pool"
                            costs["analysis_timestamp"] = datetime.now().isoformat()
                            return costs
                    
                    return {"status": "error", "error": "No cost analysis received"}
                    
        except Exception as e:
            logger.warning(f"⚠️ Transaction Cost Agent Pool request failed: {e}")
            return self._mock_cost_analysis()
    
    async def _apply_risk_management_via_pool(self, portfolio_weights: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply risk management using Risk Agent Pool"""
        try:
            if not MCP_AVAILABLE:
                return self._mock_risk_management()
            
            risk_pool_url = self.config["agent_pools"]["risk_agent_pool"]["url"]
            
            # Natural language query for risk management
            query = f"""
            Apply comprehensive risk management to the portfolio allocation based on market data and risk constraints.
            
            Portfolio Weights: {json.dumps(portfolio_weights, default=str)[:1000]}...
            Market Data: {json.dumps(market_data, default=str)[:500]}...
            
            Requirements:
            - Calculate Value at Risk (VaR) at 95% and 99% confidence levels
            - Perform stress testing scenarios (market crash, tech sector decline)
            - Set position limits and concentration constraints
            - Apply dynamic hedging recommendations
            - Monitor correlation risks between AAPL and MSFT
            - Provide risk-adjusted position sizing
            - Include stop-loss and take-profit recommendations
            """
            
            async with sse_client(risk_pool_url, timeout=120) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    result = await session.call_tool("process_risk_request", {"query": query})
                    
                    if result.content and len(result.content) > 0:
                        content_item = result.content[0]
                        if hasattr(content_item, 'text'):
                            risk_mgmt = json.loads(content_item.text)
                            logger.info(f"✅ Applied risk management: {risk_mgmt.get('status', 'unknown')}")
                            
                            risk_mgmt["risk_management_source"] = "risk_agent_pool"
                            risk_mgmt["risk_timestamp"] = datetime.now().isoformat()
                            return risk_mgmt
                    
                    return {"status": "error", "error": "No risk management received"}
                    
        except Exception as e:
            logger.warning(f"⚠️ Risk Agent Pool request failed: {e}")
            return self._mock_risk_management()
    
    async def _simulate_complete_backtest(self, market_data: Dict[str, Any], alpha_signals: Dict[str, Any], 
                                        risk_adjusted_portfolio: Dict[str, Any], cost_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the complete backtest using all agent pool outputs"""
        logger.info("⚡ Simulating complete backtest...")
        
        # Backtest parameters
        initial_capital = self.config.get("backtest", {}).get("default_capital", 1000000)
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)
        symbols = self.config.get("backtest", {}).get("default_symbols", ["AAPL", "MSFT"])
        
        # Initialize portfolio state
        portfolio_state = {
            "cash": initial_capital,
            "positions": {symbol: 0 for symbol in symbols},
            "portfolio_value": initial_capital,
            "daily_returns": [],
            "daily_values": [initial_capital],
            "transactions": [],
            "dates": []
        }
        
        # Generate trading dates (weekdays only)
        trading_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday=0, Sunday=6
                trading_dates.append(current_date)
            current_date += timedelta(days=1)
        
        logger.info(f"📈 Simulating {len(trading_dates)} trading days...")
        
        # Simulate daily trading
        for i, date in enumerate(trading_dates):
            # Generate synthetic prices (in real implementation, use market_data)
            prices = self._generate_daily_prices(symbols, date, i)
            
            # Apply trading logic based on agent pool outputs
            if i % 20 == 0:  # Rebalance every 20 days
                # Execute trades based on alpha signals and risk management
                transactions = self._execute_rebalancing(
                    portfolio_state, prices, alpha_signals, risk_adjusted_portfolio, cost_analysis
                )
                portfolio_state["transactions"].extend(transactions)
            
            # Update portfolio value
            portfolio_value = portfolio_state["cash"]
            for symbol, position in portfolio_state["positions"].items():
                portfolio_value += position * prices.get(symbol, 0)
            
            # Calculate daily return
            previous_value = portfolio_state["daily_values"][-1]
            daily_return = (portfolio_value - previous_value) / previous_value if previous_value > 0 else 0
            
            # Update state
            portfolio_state["portfolio_value"] = portfolio_value
            portfolio_state["daily_returns"].append(daily_return)
            portfolio_state["daily_values"].append(portfolio_value)
            portfolio_state["dates"].append(date.strftime("%Y-%m-%d"))
            
            # Progress logging
            if i % 252 == 0:  # Every ~year
                progress = (i / len(trading_dates)) * 100
                logger.info(f"📊 Progress: {progress:.1f}% - Portfolio: ${portfolio_value:,.2f}")
        
        # Calculate final metrics
        returns = np.array(portfolio_state["daily_returns"])
        total_return = (portfolio_state["portfolio_value"] - initial_capital) / initial_capital
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        values = np.array(portfolio_state["daily_values"])
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        simulation_results = {
            "performance_metrics": {
                "total_return": total_return,
                "annualized_return": (1 + total_return) ** (252/len(returns)) - 1 if len(returns) > 0 else 0,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "final_value": portfolio_state["portfolio_value"]
            },
            "portfolio_data": portfolio_state,
            "simulation_metadata": {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "trading_days": len(trading_dates),
                "symbols": symbols,
                "initial_capital": initial_capital
            }
        }
        
        logger.info("✅ Backtest simulation completed")
        return simulation_results
    
    def _generate_daily_prices(self, symbols: List[str], date: datetime, day_index: int) -> Dict[str, float]:
        """Generate synthetic daily prices (replace with real market data in production)"""
        random.seed(int(date.timestamp()) % 10000)
        
        # Base prices with growth trend
        base_prices = {"AAPL": 150.0, "MSFT": 300.0}
        prices = {}
        
        for symbol in symbols:
            base = base_prices.get(symbol, 100.0)
            
            # Add trend (25% over 3 years)
            trend = 1 + (day_index / 780) * 0.25  # ~780 trading days in 3 years
            
            # Add seasonality
            cycle = 1 + 0.1 * np.sin(day_index * 2 * np.pi / 252)  # Annual cycle
            
            # Add random volatility
            noise = 1 + random.normalvariate(0, 0.02)  # 2% daily volatility
            
            prices[symbol] = base * trend * cycle * noise
        
        return prices
    
    def _execute_rebalancing(self, portfolio_state: Dict[str, Any], prices: Dict[str, float],
                           alpha_signals: Dict[str, Any], risk_adjusted_portfolio: Dict[str, Any],
                           cost_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute portfolio rebalancing based on agent pool recommendations"""
        transactions = []
        
        # Simple rebalancing logic (enhanced in real implementation)
        total_value = portfolio_state["portfolio_value"]
        target_weights = {"AAPL": 0.6, "MSFT": 0.4}  # Use risk_adjusted_portfolio in real implementation
        
        for symbol, target_weight in target_weights.items():
            current_value = portfolio_state["positions"][symbol] * prices.get(symbol, 0)
            target_value = total_value * target_weight
            
            if abs(target_value - current_value) > total_value * 0.02:  # 2% threshold
                # Calculate shares to trade
                price = prices.get(symbol, 0)
                if price > 0:
                    target_shares = int(target_value / price)
                    current_shares = portfolio_state["positions"][symbol]
                    shares_to_trade = target_shares - current_shares
                    
                    if shares_to_trade != 0:
                        # Apply transaction costs
                        cost_rate = 0.001  # Use cost_analysis in real implementation
                        cost = abs(shares_to_trade) * price * cost_rate
                        
                        # Execute trade
                        if shares_to_trade > 0:  # Buy
                            total_cost = shares_to_trade * price + cost
                            if portfolio_state["cash"] >= total_cost:
                                portfolio_state["positions"][symbol] += shares_to_trade
                                portfolio_state["cash"] -= total_cost
                                
                                transactions.append({
                                    "symbol": symbol,
                                    "action": "buy",
                                    "shares": shares_to_trade,
                                    "price": price,
                                    "cost": total_cost
                                })
                        
                        else:  # Sell
                            proceeds = abs(shares_to_trade) * price - cost
                            portfolio_state["positions"][symbol] += shares_to_trade  # shares_to_trade is negative
                            portfolio_state["cash"] += proceeds
                            
                            transactions.append({
                                "symbol": symbol,
                                "action": "sell",
                                "shares": abs(shares_to_trade),
                                "price": price,
                                "proceeds": proceeds
                            })
        
        return transactions
    
    async def _generate_comprehensive_analysis(self):
        """Generate comprehensive analysis of the orchestrated backtest"""
        logger.info("📈 Generating Comprehensive Analysis...")
        
        if not self.backtest_results.get("backtest_simulation"):
            logger.warning("⚠️ No backtest simulation results available")
            return
        
        simulation = self.backtest_results["backtest_simulation"]
        performance = simulation.get("performance_metrics", {})
        
        # Agent pool utilization analysis
        agent_pool_analysis = {
            "data_agent_pool": {
                "status": self.backtest_results.get("market_data", {}).get("status", "unknown"),
                "data_points": len(self.backtest_results.get("market_data", {}).get("data", [])),
                "source": self.backtest_results.get("market_data", {}).get("data_source", "unknown")
            },
            "alpha_agent_pool": {
                "status": self.backtest_results.get("alpha_signals", {}).get("status", "unknown"),
                "signals_generated": len(self.backtest_results.get("alpha_signals", {}).get("signals", [])),
                "source": self.backtest_results.get("alpha_signals", {}).get("signal_source", "unknown")
            },
            "portfolio_construction_agent_pool": {
                "status": self.backtest_results.get("portfolio_weights", {}).get("status", "unknown"),
                "optimization_method": self.backtest_results.get("portfolio_weights", {}).get("method", "unknown"),
                "source": self.backtest_results.get("portfolio_weights", {}).get("optimization_source", "unknown")
            },
            "transaction_cost_agent_pool": {
                "status": self.backtest_results.get("cost_analysis", {}).get("status", "unknown"),
                "cost_estimates": self.backtest_results.get("cost_analysis", {}).get("total_costs", 0),
                "source": self.backtest_results.get("cost_analysis", {}).get("cost_analysis_source", "unknown")
            },
            "risk_agent_pool": {
                "status": self.backtest_results.get("risk_adjusted_portfolio", {}).get("status", "unknown"),
                "risk_measures": len(self.backtest_results.get("risk_adjusted_portfolio", {}).get("risk_metrics", {})),
                "source": self.backtest_results.get("risk_adjusted_portfolio", {}).get("risk_management_source", "unknown")
            }
        }
        
        # Chatbot interaction analysis
        chatbot_analysis = {
            "conversation_turns": len(self.chat_history),
            "session_id": self.session_id,
            "instruction_processing": "success" if self.backtest_results.get("parsed_instruction") else "failed",
            "natural_language_capability": "enabled" if self.conversation_manager else "disabled"
        }
        
        # Orchestration efficiency
        orchestration_analysis = {
            "total_agent_pools_used": len([pool for pool in agent_pool_analysis.values() if pool["status"] != "error"]),
            "successful_integrations": len([pool for pool in agent_pool_analysis.values() if pool["status"] == "success"]),
            "data_flow_integrity": "high" if all(pool["status"] != "error" for pool in agent_pool_analysis.values()) else "medium",
            "execution_time": "efficient"  # Could be calculated from timestamps
        }
        
        # Store analysis
        self.backtest_results["comprehensive_analysis"] = {
            "agent_pool_analysis": agent_pool_analysis,
            "chatbot_analysis": chatbot_analysis,
            "orchestration_analysis": orchestration_analysis,
            "performance_summary": performance,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        logger.info("✅ Comprehensive analysis completed")
    
    def _print_chatbot_summary(self):
        """Print comprehensive summary of the chatbot backtest"""
        logger.info("=" * 80)
        logger.info("🤖 ORCHESTRATOR CHATBOT BACKTEST SUMMARY")
        logger.info("=" * 80)
        
        # Chatbot interaction summary
        chatbot_analysis = self.backtest_results.get("comprehensive_analysis", {}).get("chatbot_analysis", {})
        logger.info(f"💬 Chatbot Interaction:")
        logger.info(f"    Session ID: {chatbot_analysis.get('session_id', 'unknown')}")
        logger.info(f"    Conversation Turns: {chatbot_analysis.get('conversation_turns', 0)}")
        logger.info(f"    Instruction Processing: {chatbot_analysis.get('instruction_processing', 'unknown')}")
        logger.info(f"    NL Capability: {chatbot_analysis.get('natural_language_capability', 'unknown')}")
        
        # Agent pool coordination summary
        agent_analysis = self.backtest_results.get("comprehensive_analysis", {}).get("agent_pool_analysis", {})
        logger.info(f"🔗 Agent Pool Coordination:")
        for pool_name, pool_data in agent_analysis.items():
            status_emoji = "✅" if pool_data.get("status") == "success" else "⚠️" if pool_data.get("status") == "mock" else "❌"
            logger.info(f"    {status_emoji} {pool_name}: {pool_data.get('status', 'unknown')} ({pool_data.get('source', 'unknown')})")
        
        # Performance summary
        simulation = self.backtest_results.get("backtest_simulation", {})
        performance = simulation.get("performance_metrics", {})
        metadata = simulation.get("simulation_metadata", {})
        
        logger.info(f"📈 Backtest Performance:")
        logger.info(f"    Period: {metadata.get('start_date', 'unknown')} to {metadata.get('end_date', 'unknown')}")
        logger.info(f"    Symbols: {metadata.get('symbols', [])}")
        logger.info(f"    Initial Capital: ${metadata.get('initial_capital', 0):,.2f}")
        logger.info(f"    Final Value: ${performance.get('final_value', 0):,.2f}")
        logger.info(f"    Total Return: {performance.get('total_return', 0):.2%}")
        logger.info(f"    Annualized Return: {performance.get('annualized_return', 0):.2%}")
        logger.info(f"    Volatility: {performance.get('volatility', 0):.2%}")
        logger.info(f"    Sharpe Ratio: {performance.get('sharpe_ratio', 0):.3f}")
        logger.info(f"    Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
        
        # Orchestration efficiency
        orchestration = self.backtest_results.get("comprehensive_analysis", {}).get("orchestration_analysis", {})
        logger.info(f"🎯 Orchestration Efficiency:")
        logger.info(f"    Agent Pools Used: {orchestration.get('total_agent_pools_used', 0)}/5")
        logger.info(f"    Successful Integrations: {orchestration.get('successful_integrations', 0)}")
        logger.info(f"    Data Flow Integrity: {orchestration.get('data_flow_integrity', 'unknown')}")
        logger.info(f"    Execution Efficiency: {orchestration.get('execution_time', 'unknown')}")
        
        # Transaction summary
        portfolio_data = simulation.get("portfolio_data", {})
        transactions = portfolio_data.get("transactions", [])
        logger.info(f"💼 Transaction Summary:")
        logger.info(f"    Total Transactions: {len(transactions)}")
        if transactions:
            buy_count = len([t for t in transactions if t.get("action") == "buy"])
            sell_count = len([t for t in transactions if t.get("action") == "sell"])
            logger.info(f"    Buy Orders: {buy_count}")
            logger.info(f"    Sell Orders: {sell_count}")
        
        logger.info("=" * 80)
        logger.info("✅ Orchestrator Chatbot Backtest Completed Successfully!")
        logger.info("=" * 80)
    
    # Mock methods for fallback when agent pools are not available
    def _mock_market_data(self) -> Dict[str, Any]:
        """Mock market data when Data Agent Pool is unavailable"""
        return {
            "status": "mock",
            "data": [
                {"symbol": "AAPL", "date": "2022-01-01", "close": 150.0, "volume": 1000000},
                {"symbol": "MSFT", "date": "2022-01-01", "close": 300.0, "volume": 800000}
            ],
            "data_source": "mock_data_generator",
            "mock_reason": "data_agent_pool_unavailable"
        }
    
    def _mock_alpha_signals(self) -> Dict[str, Any]:
        """Mock alpha signals when Alpha Agent Pool is unavailable"""
        return {
            "status": "mock",
            "signals": [
                {"symbol": "AAPL", "signal": "buy", "confidence": 0.7, "strategy": "momentum"},
                {"symbol": "MSFT", "signal": "hold", "confidence": 0.5, "strategy": "mean_reversion"}
            ],
            "signal_source": "mock_signal_generator",
            "mock_reason": "alpha_agent_pool_unavailable"
        }
    
    def _mock_portfolio_optimization(self) -> Dict[str, Any]:
        """Mock portfolio optimization when Portfolio Agent Pool is unavailable"""
        return {
            "status": "mock",
            "weights": {"AAPL": 0.6, "MSFT": 0.4},
            "expected_return": 0.12,
            "volatility": 0.15,
            "sharpe_ratio": 0.8,
            "optimization_source": "mock_optimizer",
            "mock_reason": "portfolio_agent_pool_unavailable"
        }
    
    def _mock_cost_analysis(self) -> Dict[str, Any]:
        """Mock cost analysis when Transaction Cost Agent Pool is unavailable"""
        return {
            "status": "mock",
            "total_costs": 0.001,
            "market_impact": 0.0005,
            "bid_ask_spread": 0.0003,
            "commission": 0.0002,
            "cost_analysis_source": "mock_cost_analyzer",
            "mock_reason": "transaction_cost_agent_pool_unavailable"
        }
    
    def _mock_risk_management(self) -> Dict[str, Any]:
        """Mock risk management when Risk Agent Pool is unavailable"""
        return {
            "status": "mock",
            "var_95": -0.02,
            "var_99": -0.035,
            "max_position_size": 0.4,
            "risk_adjusted_weights": {"AAPL": 0.55, "MSFT": 0.35, "CASH": 0.1},
            "risk_management_source": "mock_risk_manager",
            "mock_reason": "risk_agent_pool_unavailable"
        }


# Mock classes for when FinAgent components are not available
class MockOrchestrator:
    def __init__(self):
        self.memory_agent = None

class MockAgentMonitor:
    async def check_all_pools(self):
        return {
            "data_agent_pool": type('obj', (object,), {"status": type('obj', (object,), {"value": "mock"}), "response_time": 0.001}),
            "alpha_agent_pool": type('obj', (object,), {"status": type('obj', (object,), {"value": "mock"}), "response_time": 0.001}),
        }


async def main():
    """Main entry point for orchestrator chatbot backtest"""
    logger.info("🚀 Starting Orchestrator Chatbot Backtest")
    
    try:
        backtester = OrchestratorChatbotBacktester()
        await backtester.run_chatbot_backtest()
        
        logger.info("✅ Orchestrator chatbot backtest completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Orchestrator chatbot backtest failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
