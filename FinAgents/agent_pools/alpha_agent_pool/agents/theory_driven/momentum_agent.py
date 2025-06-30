# agent_pools/alpha_agent_pool/agents/theory_driven/momentum_agent.py

from mcp.server.fastmcp import FastMCP, Context as MCPContext
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from schema.theory_driven_schema import (
    MomentumAgentConfig, MomentumSignalRequest, AlphaStrategyFlow, MarketContext, Decision, Action, PerformanceFeedback, Metadata
)
from typing import List, Dict, Any, Optional
import asyncio
import sys
import argparse
import json
import os
from datetime import datetime
import hashlib
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)
    print(f"✅ Loaded .env from: {env_path}")
except ImportError:
    print("⚠️ python-dotenv not installed. Please install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Failed to load .env file: {e}")

# LLM Integration
try:
    import openai
    from openai import AsyncOpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    AsyncOpenAI = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MomentumAgent:
    def __init__(self, config: MomentumAgentConfig):
        """
        Initialize the MomentumAgent with LLM integration for intelligent signal generation.
        Args:
            config (MomentumAgentConfig): Configuration object for the agent.
        """
        self.config = config
        self.agent = FastMCP("MomentumAlphaAgent")
        
        # Initialize LLM client
        self.llm_client = None
        self._initialize_llm()
        
        # Path to the shared memory JSON file (used by AlphaAgentPool)
        self.memory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../memory_unit.json"))
        # Path to store strategy signal flow
        self.signal_flow_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../momentum_signal_flow.json"))
        
        # Clear memory and signal flow files on each agent restart
        for path in [self.signal_flow_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
                    
        self._register_tools()

    def _initialize_llm(self):
        """Initialize LLM client for intelligent analysis"""
        if not LLM_AVAILABLE:
            logger.warning("OpenAI library not available. Using fallback logic.")
            return
            
        try:
            # Try to get API key from environment
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.llm_client = AsyncOpenAI(api_key=api_key)
                logger.info("✅ LLM client initialized successfully")
            else:
                logger.warning("OPENAI_API_KEY not found in environment. Using fallback logic.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")

    async def _analyze_market_with_llm(self, symbol: str, prices: List[float], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze market conditions and generate trading signals"""
        if not self.llm_client:
            return self._fallback_analysis(symbol, prices)
            
        try:
            # Prepare market data for LLM analysis
            price_data = {
                "symbol": symbol,
                "current_price": prices[-1] if prices else 0,
                "price_history": prices[-20:] if len(prices) >= 20 else prices,
                "price_change_1d": ((prices[-1] - prices[-2]) / prices[-2] * 100) if len(prices) >= 2 else 0,
                "price_change_5d": ((prices[-1] - prices[-6]) / prices[-6] * 100) if len(prices) >= 6 else 0,
                "price_change_20d": ((prices[-1] - prices[-20]) / prices[-20] * 100) if len(prices) >= 20 else 0,
                "sma_10": sum(prices[-10:]) / min(10, len(prices)) if prices else 0,
                "sma_20": sum(prices[-20:]) / min(20, len(prices)) if prices else 0,
                "volatility": self._calculate_volatility(prices),
                "momentum_score": self._calculate_momentum(prices)
            }
            
            # Create comprehensive prompt for LLM analysis
            prompt = f"""
            You are an expert quantitative analyst specializing in momentum trading strategies. 
            Analyze the following market data for {symbol} and provide a trading recommendation.

            MARKET DATA:
            - Symbol: {symbol}
            - Current Price: ${price_data['current_price']:.2f}
            - 1-Day Change: {price_data['price_change_1d']:.2f}%
            - 5-Day Change: {price_data['price_change_5d']:.2f}%
            - 20-Day Change: {price_data['price_change_20d']:.2f}%
            - 10-Day SMA: ${price_data['sma_10']:.2f}
            - 20-Day SMA: ${price_data['sma_20']:.2f}
            - Volatility: {price_data['volatility']:.4f}
            - Momentum Score: {price_data['momentum_score']:.4f}
            - Price History (last 10): {[round(p, 2) for p in prices[-10:]] if len(prices) >= 10 else prices}

            ANALYSIS REQUIREMENTS:
            1. Assess momentum strength and direction
            2. Evaluate price trends and moving average signals
            3. Consider volatility and risk factors
            4. Provide clear BUY/SELL/HOLD recommendation
            5. Estimate confidence level (0.0 to 1.0)
            6. Predict expected return (as decimal, e.g., 0.05 for 5%)
            7. Assess risk level (as decimal, e.g., 0.02 for 2% risk)
            8. Suggest position sizing weight (-1.0 to 1.0)

            Respond ONLY with a valid JSON object in this exact format:
            {{
                "signal": "BUY|SELL|HOLD",
                "confidence": 0.0-1.0,
                "reasoning": "detailed explanation of your analysis",
                "predicted_return": decimal_value,
                "risk_estimate": decimal_value,
                "execution_weight": decimal_value,
                "market_regime": "bullish|bearish|neutral|volatile",
                "key_factors": ["factor1", "factor2", "factor3"]
            }}
            """
            
            # Query LLM - use o4-mini model
            response = await self.llm_client.chat.completions.create(
                model="o4-mini",  # Using o4-mini model as requested
                messages=[
                    {"role": "system", "content": "You are a professional quantitative analyst. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1000  # o4-mini requires sufficient tokens for full response
            )
            
            # Parse LLM response
            llm_analysis = json.loads(response.choices[0].message.content.strip())
            
            # Validate and normalize LLM output
            signal = llm_analysis.get("signal", "HOLD").upper()
            if signal not in ["BUY", "SELL", "HOLD"]:
                signal = "HOLD"
                
            confidence = max(0.0, min(1.0, float(llm_analysis.get("confidence", 0.0))))
            predicted_return = float(llm_analysis.get("predicted_return", 0.0))
            risk_estimate = max(0.001, float(llm_analysis.get("risk_estimate", 0.02)))
            execution_weight = max(-1.0, min(1.0, float(llm_analysis.get("execution_weight", 0.0))))
            
            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": llm_analysis.get("reasoning", "LLM-based momentum analysis"),
                "predicted_return": predicted_return,
                "risk_estimate": risk_estimate,
                "execution_weight": execution_weight,
                "market_regime": llm_analysis.get("market_regime", "neutral"),
                "key_factors": llm_analysis.get("key_factors", []),
                "analysis_source": "llm",
                "llm_model": "o4-mini"
            }
            
        except Exception as e:
            logger.error(f"LLM analysis failed for {symbol}: {e}")
            return self._fallback_analysis(symbol, prices)

    def _fallback_analysis(self, symbol: str, prices: List[float]) -> Dict[str, Any]:
        """Fallback analysis when LLM is unavailable"""
        if not prices or len(prices) < 2:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reasoning": "Insufficient price data for analysis",
                "predicted_return": 0.0,
                "risk_estimate": 0.02,
                "execution_weight": 0.0,
                "analysis_source": "fallback"
            }
        
        # Simple momentum analysis
        momentum_score = self._calculate_momentum(prices)
        volatility = self._calculate_volatility(prices)
        
        # Generate signal based on momentum
        if momentum_score > 0.05:  # 5% momentum threshold
            signal = "BUY"
            confidence = min(momentum_score, 0.8)
            predicted_return = momentum_score * 0.5  # Conservative estimate
            execution_weight = min(momentum_score * 2, 0.5)
        elif momentum_score < -0.05:
            signal = "SELL"
            confidence = min(abs(momentum_score), 0.8)
            predicted_return = momentum_score * 0.5
            execution_weight = max(momentum_score * 2, -0.5)
        else:
            signal = "HOLD"
            confidence = 0.0
            predicted_return = 0.0
            execution_weight = 0.0
        
        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": f"Fallback momentum analysis: {momentum_score:.4f} momentum score, {volatility:.4f} volatility",
            "predicted_return": predicted_return,
            "risk_estimate": max(volatility, 0.01),
            "execution_weight": execution_weight,
            "analysis_source": "fallback"
        }

    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate price momentum"""
        if len(prices) < 2:
            return 0.0
        
        # Use 20-day momentum if available, otherwise use available period
        lookback = min(20, len(prices) - 1)
        if lookback <= 0:
            return 0.0
            
        current_price = prices[-1]
        old_price = prices[-(lookback + 1)]
        
        return (current_price - old_price) / old_price if old_price != 0 else 0.0

    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.02  # Default volatility
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return 0.02
        
        # Calculate standard deviation of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance ** 0.5

    def _read_memory(self, key):
        if not os.path.exists(self.memory_path):
            return None
        try:
            with open(self.memory_path, 'r') as f:
                data = json.load(f)
            return data.get(key)
        except Exception:
            return None

    def _write_signal_flow(self, flow_obj):
        # Append the strategy flow object to a json file as a list
        if os.path.exists(self.signal_flow_path):
            try:
                with open(self.signal_flow_path, 'r') as f:
                    flow = json.load(f)
            except Exception:
                flow = []
        else:
            flow = []
        flow.append(flow_obj)
        with open(self.signal_flow_path, 'w') as f:
            json.dump(flow, f, indent=2)

    def _register_tools(self):
        """
        Register the agent's tools with the FastMCP server with LLM-powered analysis.
        """
        @self.agent.tool()
        async def generate_signal(request: MomentumSignalRequest, ctx: MCPContext = None) -> dict:
            """
            Generate a sophisticated alpha strategy flow using LLM analysis and real market data.
            """
            symbol = request.symbol
            price_list = request.price_list
            
            # Get price data
            if price_list is None:
                closes = []
                for i in range(self.config.strategy.window):
                    key = f"{symbol}_close_2024-01-{str(31-i).zfill(2)} 05:00:00"
                    val = self._read_memory(key)
                    if val is not None:
                        closes.insert(0, float(val))
                prices = closes if closes else self._generate_synthetic_prices(symbol, self.config.strategy.window)
            else:
                prices = price_list
            
            # Use LLM for intelligent analysis
            market_context_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "price_count": len(prices)
            }
            
            analysis_result = await self._analyze_market_with_llm(symbol, prices, market_context_data)
            
            # Build comprehensive output using LLM insights
            now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            code_hash = hashlib.sha256((str(prices) + analysis_result["signal"]).encode()).hexdigest()
            
            # Determine market regime
            regime_map = {
                "BUY": analysis_result.get("market_regime", "bullish_trend"),
                "SELL": analysis_result.get("market_regime", "bearish_trend"),
                "HOLD": analysis_result.get("market_regime", "neutral")
            }
            
            flow_obj = AlphaStrategyFlow(
                alpha_id="momentum_llm_v4",
                version="2025.06.29-llm",
                timestamp=now,
                market_context=MarketContext(
                    symbol=symbol,
                    regime_tag=regime_map[analysis_result["signal"]],
                    input_features={
                        "price_current": prices[-1] if prices else 0,
                        "price_20d_ago": prices[0] if len(prices) >= 20 else (prices[0] if prices else 0),
                        "sma_10": sum(prices[-10:]) / min(10, len(prices)) if prices else 0,
                        "sma_20": sum(prices[-20:]) / min(20, len(prices)) if prices else 0,
                        "momentum_score": self._calculate_momentum(prices),
                        "volatility": self._calculate_volatility(prices),
                        "analysis_source": analysis_result.get("analysis_source", "unknown"),
                        "llm_factors": analysis_result.get("key_factors", [])
                    }
                ),
                decision=Decision(
                    signal=analysis_result["signal"],
                    confidence=analysis_result["confidence"],
                    reasoning=analysis_result["reasoning"],
                    predicted_return=analysis_result["predicted_return"],
                    risk_estimate=analysis_result["risk_estimate"],
                    signal_type="directional_llm",
                    asset_scope=[symbol]
                ),
                action=Action(
                    execution_weight=analysis_result["execution_weight"],
                    order_type="market",
                    order_price=prices[-1] if prices else 0,
                    execution_delay="T+0"
                ),
                performance_feedback=PerformanceFeedback(
                    status="pending",
                    evaluation_link=None
                ),
                metadata=Metadata(
                    generator_agent="momentum_llm_agent",
                    strategy_prompt="LLM-powered momentum analysis with sophisticated market regime detection",
                    code_hash=f"sha256:{code_hash}",
                    context_id=f"llm_dag_{now[:10].replace('-', '')}_{now[11:13]}"
                )
            )
            
            # Write to strategy flow file
            self._write_signal_flow(flow_obj.dict())
            return flow_obj.dict()

        @self.agent.tool()
        async def analyze_market_sentiment(symbol: str, lookback_days: int = 20) -> dict:
            """
            Analyze market sentiment using LLM for the given symbol.
            """
            try:
                # Get price data
                prices = []
                for i in range(lookback_days):
                    key = f"{symbol}_close_2024-01-{str(31-i).zfill(2)} 05:00:00"
                    val = self._read_memory(key)
                    if val is not None:
                        prices.insert(0, float(val))
                
                if not prices:
                    prices = self._generate_synthetic_prices(symbol, lookback_days)
                
                # Use LLM for sentiment analysis
                analysis = await self._analyze_market_with_llm(symbol, prices, {})
                
                return {
                    "symbol": symbol,
                    "sentiment": analysis.get("market_regime", "neutral"),
                    "confidence": analysis.get("confidence", 0.0),
                    "key_factors": analysis.get("key_factors", []),
                    "reasoning": analysis.get("reasoning", ""),
                    "timestamp": datetime.now().isoformat(),
                    "analysis_source": analysis.get("analysis_source", "llm")
                }
                
            except Exception as e:
                return {
                    "symbol": symbol,
                    "sentiment": "unknown",
                    "confidence": 0.0,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

    def _generate_synthetic_prices(self, symbol: str, window: int) -> List[float]:
        """
        Generate realistic synthetic price series for testing when real data is unavailable.
        """
        import random
        
        # Use symbol hash to ensure consistent "random" data for same symbol
        random.seed(hash(symbol) % 1000)
        
        base = random.uniform(90, 150)  # Starting price range
        prices = []
        
        for i in range(window):
            # Add some realistic market behavior
            daily_return = random.normalvariate(0.001, 0.02)  # 0.1% mean return, 2% volatility
            base *= (1 + daily_return)
            base = max(base, 1.0)  # Ensure price doesn't go negative
            prices.append(round(base, 2))
        
        return prices

    def start_mcp_server(self, port: int = None, host: str = "0.0.0.0", transport: str = "http"):
        """
        Start the MCP server for the MomentumAgent with LLM capabilities.
        """
        port = port or self.config.execution.port
        self.agent.settings.host = host
        self.agent.settings.port = port
        
        logger.info(f"Starting LLM-powered MomentumAgent MCP server on {host}:{port} (transport={transport})")
        if self.llm_client:
            logger.info("🤖 LLM integration enabled - using intelligent analysis")
        else:
            logger.warning("⚠️ LLM not available - using fallback analysis")
            
        self.agent.run(transport=transport)

    def __repr__(self):
        """
        Return a string representation of the enhanced MomentumAgent.
        """
        llm_status = "LLM-enabled" if self.llm_client else "fallback-mode"
        return f"<MomentumAgent id={self.config.agent_id} port={self.config.execution.port} {llm_status}>"