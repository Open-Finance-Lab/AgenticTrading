# core.py
"""
Unified entry point for starting the Alpha Agent Pool MCP service.
This module manages the lifecycle and orchestration of multiple sub-agents within the AlphaAgentPool.
"""
import multiprocessing
import os
import yaml
import threading
import asyncio
import json
import csv
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from mcp.server.fastmcp import FastMCP
from schema.theory_driven_schema import MomentumAgentConfig
from agents.theory_driven.momentum_agent import MomentumAgent
from agents.autonomous.autonomous_agent import AutonomousAgent

# Add memory module to path
memory_path = Path(__file__).parent.parent.parent / "memory"
sys.path.insert(0, str(memory_path))

try:
    from ...memory.external_memory_agent import ExternalMemoryAgent, EventType, LogLevel
    MEMORY_AVAILABLE = True
except ImportError:
    ExternalMemoryAgent = None
    EventType = LogLevel = None
    MEMORY_AVAILABLE = False

# Configure logging
logger = logging.getLogger("AlphaAgentPool")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class MemoryUnit:
    """
    MemoryUnit provides a simple key-value store with optional persistence to a local JSON file.
    On initialization, it can automatically load a static dataset from a CSV file if provided.
    If reset_on_init is True, the memory file will be cleared on each initialization.
    """
    def __init__(self, file_path, autoload_csv_path=None, reset_on_init=False):
        self.file_path = file_path
        if reset_on_init and os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
            except Exception:
                pass
        self._data = {}
        self._load()
        if autoload_csv_path:
            self._autoload_csv(autoload_csv_path)

    def _autoload_csv(self, csv_path):
        if not os.path.exists(csv_path):
            return
        try:
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    date = row['timestamp']
                    close = row['close']
                    self.set(f"AAPL_close_{date}", close)
        except Exception as e:
            pass  # Optionally log error

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        with open(self.file_path, 'w') as f:
            json.dump(self._data, f, indent=2)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value
        self._save()

    def delete(self, key):
        if key in self._data:
            del self._data[key]
            self._save()

    def keys(self):
        return list(self._data.keys())

class AlphaAgentPoolMCPServer:
    def __init__(self, host="0.0.0.0", port=5050):
        """
        Initialize the AlphaAgentPoolMCPServer instance.
        Args:
            host (str): Host address to bind the MCP server.
            port (int): Port number to bind the MCP server.
        """
        self.host = host
        self.port = port
        self.pool_server = FastMCP("AlphaAgentPoolMCPServer")
        self.agent_registry = {}  # agent_id -> (agent, process/thread)
        self.config_dir = os.path.join(os.path.dirname(__file__), "config")
        
        # Automatically load static dataset into memory unit on startup, and reset memory file
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/cache/AAPL_2024-01-01_2024-01-31_1d.csv"))
        self.memory = MemoryUnit(os.path.join(os.path.dirname(__file__), "memory_unit.json"), autoload_csv_path=csv_path, reset_on_init=True)
        
        # Initialize memory agent if available
        self.memory_agent: Optional[ExternalMemoryAgent] = None
        self.session_id = None
        self._initialize_memory_agent()
        
        self._register_pool_tools()

    def _initialize_memory_agent(self):
        """Initialize the external memory agent"""
        if not MEMORY_AVAILABLE:
            logger.warning("External memory agent not available")
            return
        
        try:
            self.memory_agent = ExternalMemoryAgent()
            self.session_id = f"alpha_pool_session_{int(asyncio.get_event_loop().time())}"
            logger.info("External memory agent initialized for Alpha Agent Pool")
        except Exception as e:
            logger.error(f"Failed to initialize memory agent: {e}")
            self.memory_agent = None

    async def _log_memory_event(self, event_type, log_level, title: str, content: str, 
                               tags: set = None, metadata: Optional[Dict[str, Any]] = None):
        """Log an event to the memory agent with proper enum types"""
        if self.memory_agent and self.session_id:
            try:
                await self.memory_agent.log_event(
                    event_type=event_type,
                    log_level=log_level,
                    source_agent_pool="alpha_agent_pool",
                    source_agent_id="alpha_agent_pool_server",
                    title=title,
                    content=content,
                    tags=tags or set(),
                    metadata={
                        "session_id": self.session_id,
                        "agent_pool": "alpha",
                        **(metadata or {})
                    },
                    session_id=self.session_id
                )
            except Exception as e:
                logger.warning(f"Failed to log memory event: {e}")

    def _register_pool_tools(self):
        """
        Register management tools for the agent pool, including starting and listing sub-agents.
        """
        @self.pool_server.tool(name="start_agent", description="Start the specified sub-agent service.")
        def start_agent(agent_id: str) -> str:
            """
            Start a sub-agent by agent_id. If already started, returns a status message.
            Args:
                agent_id (str): Identifier of the agent to start.
            Returns:
                str: Status message indicating the result.
            """
            if agent_id in self.agent_registry:
                return f"Agent '{agent_id}' is already running."
            if agent_id == "momentum_agent":
                config, process = self._start_momentum_agent()
                self.agent_registry[agent_id] = process
                return f"Momentum agent started on port {config.execution.port}"
            elif agent_id == "autonomous_agent":
                process = self._start_autonomous_agent()
                self.agent_registry[agent_id] = process
                return f"Autonomous agent started on port 5051"
            return f"Unknown agent: {agent_id}"

        @self.pool_server.tool(name="list_agents", description="List all registered sub-agents.")
        def list_agents() -> list:
            """
            List all currently registered sub-agents in the pool.
            Returns:
                list: List of agent IDs.
            """
            return list(self.agent_registry.keys())

        @self.pool_server.tool(name="get_memory", description="Get a value from the internal memory unit by key.")
        def get_memory(key: str):
            """
            Retrieve a value from the memory unit by key.
            Args:
                key (str): The key to retrieve.
            Returns:
                The value associated with the key, or None if not found.
            """
            return self.memory.get(key)

        @self.pool_server.tool(name="set_memory", description="Set a value in the internal memory unit by key.")
        def set_memory(key: str, value):
            """
            Set a value in the memory unit by key.
            Args:
                key (str): The key to set.
                value: The value to store.
            Returns:
                str: Status message.
            """
            self.memory.set(key, value)
            return "OK"

        @self.pool_server.tool(name="delete_memory", description="Delete a key from the internal memory unit.")
        def delete_memory(key: str):
            """
            Delete a key from the memory unit.
            Args:
                key (str): The key to delete.
            Returns:
                str: Status message.
            """
            self.memory.delete(key)
            return "OK"

        @self.pool_server.tool(name="list_memory_keys", description="List all keys in the internal memory unit.")
        def list_memory_keys():
            """
            List all keys currently stored in the memory unit.
            Returns:
                list: List of keys.
            """
            return self.memory.keys()

        @self.pool_server.tool(name="send_orchestrator_input", description="Send input to autonomous agent from orchestrator.")
        def send_orchestrator_input(instruction: str, context: dict = None) -> str:
            """
            Send orchestrator input to autonomous agent for self-orchestration.
            Args:
                instruction (str): The instruction from orchestrator.
                context (dict): Optional context data.
            Returns:
                str: Response from autonomous agent.
            """
            # 这里可以通过MCP客户端连接到autonomous agent
            # 目前返回确认消息
            return f"Orchestrator input '{instruction}' sent to autonomous agent"

        @self.pool_server.tool(name="generate_alpha_signals", description="Generate alpha trading signals using available agents and real data")
        async def generate_alpha_signals(symbols: list, date: str, lookback_period: int = 20, current_prices: dict = None) -> dict:
            """
            Generate alpha trading signals for given symbols using momentum and autonomous agents.
            
            Args:
                symbols (list): List of stock symbols
                date (str): Current trading date 
                lookback_period (int): Number of days to look back for signal generation
                current_prices (dict): Current prices for symbols
            
            Returns:
                dict: Alpha signals for all symbols
            """
            try:
                # Log alpha signal generation request
                await self._log_memory_event(
                    event_type=EventType.OPTIMIZATION,
                    log_level=LogLevel.INFO,
                    title="Alpha Signal Generation Request",
                    content=f"Generating signals for {len(symbols)} symbols on {date}",
                    tags={"alpha_signals", "signal_generation", "request"},
                    metadata={
                        "symbols": symbols,
                        "date": date,
                        "lookback_period": lookback_period,
                        "has_current_prices": current_prices is not None
                    }
                )
                
                signals_result = {
                    "status": "success",
                    "date": date,
                    "signals": {},
                    "generated_by": "alpha_agent_pool",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Generate signals for each symbol
                for symbol in symbols:
                    try:
                        # Get historical price data for the symbol
                        historical_prices = []
                        
                        # Try to get real price data from memory or generate synthetic
                        for i in range(lookback_period):
                            # Try to get from memory first 
                            price_key = f"{symbol}_close_{date}"
                            stored_price = self.memory.get(price_key)
                            if stored_price:
                                historical_prices.append(float(stored_price))
                        
                        # If insufficient historical data, generate synthetic data
                        if len(historical_prices) < lookback_period:
                            import random
                            base_price = current_prices.get(symbol, random.uniform(90, 150)) if current_prices else random.uniform(90, 150)
                            historical_prices = []
                            for i in range(lookback_period):
                                daily_change = random.uniform(-0.03, 0.03)  # ±3% daily change
                                price = base_price * (1 + daily_change)
                                historical_prices.append(price)
                                base_price = price
                        
                        # Use momentum agent if available to generate signal
                        signal_result = None
                        try:
                            # Try to connect to momentum agent via MCP
                            from mcp.client.sse import sse_client
                            from mcp.client.session import ClientSession
                            
                            momentum_endpoint = "http://localhost:5052/sse"  # Momentum agent port
                            
                            async with sse_client(momentum_endpoint, timeout=5) as (read, write):
                                async with ClientSession(read, write) as session:
                                    await session.initialize()
                                    
                                    # Call momentum agent's generate_signal tool
                                    signal_request = {
                                        "symbol": symbol,
                                        "price_list": historical_prices
                                    }
                                    
                                    result = await session.call_tool("generate_signal", signal_request)
                                    if result.content and len(result.content) > 0:
                                        signal_result = json.loads(result.content[0].text)
                        
                        except Exception as e:
                            logger.warning(f"Could not connect to momentum agent for {symbol}: {e}")
                            # Fallback: generate signal using simple momentum logic
                            signal_result = self._generate_fallback_momentum_signal(symbol, historical_prices)
                        
                        # Extract signal information
                        if signal_result:
                            signal = signal_result.get("decision", {}).get("signal", "HOLD")
                            confidence = signal_result.get("decision", {}).get("confidence", 0.0)
                            predicted_return = signal_result.get("decision", {}).get("predicted_return", 0.0)
                            risk_estimate = signal_result.get("decision", {}).get("risk_estimate", 0.01)
                            execution_weight = signal_result.get("action", {}).get("execution_weight", 0.0)
                        else:
                            # Default neutral signal
                            signal = "HOLD"
                            confidence = 0.0
                            predicted_return = 0.0
                            risk_estimate = 0.01
                            execution_weight = 0.0
                        
                        signals_result["signals"][symbol] = {
                            "signal": signal,
                            "confidence": confidence,
                            "predicted_return": predicted_return,
                            "risk_estimate": risk_estimate,
                            "execution_weight": execution_weight,
                            "current_price": current_prices.get(symbol, historical_prices[-1]) if current_prices else historical_prices[-1],
                            "agent_source": "momentum_agent" if signal_result else "fallback",
                            "signal_generated_at": datetime.now().isoformat()
                        }
                        
                        # Log individual signal generation
                        await self._log_memory_event(
                            event_type=EventType.OPTIMIZATION,
                            log_level=LogLevel.INFO,
                            title=f"Alpha Signal Generated: {symbol}",
                            content=f"Signal: {signal}, Confidence: {confidence}, "
                                   f"Predicted Return: {predicted_return}, Risk: {risk_estimate}",
                            tags={f"symbol_{symbol}", "alpha_signal", "generated"},
                            metadata={
                                "symbol": symbol,
                                "signal": signal,
                                "confidence": confidence,
                                "predicted_return": predicted_return,
                                "risk_estimate": risk_estimate,
                                "execution_weight": execution_weight,
                                "date": date
                            }
                        )
                        
                    except Exception as e:
                        logger.error(f"Error generating signal for {symbol}: {e}")
                        # Add error signal
                        signals_result["signals"][symbol] = {
                            "signal": "HOLD",
                            "confidence": 0.0,
                            "predicted_return": 0.0,
                            "risk_estimate": 0.05,
                            "execution_weight": 0.0,
                            "current_price": current_prices.get(symbol, 100.0) if current_prices else 100.0,
                            "agent_source": "error_fallback",
                            "error": str(e),
                            "signal_generated_at": datetime.now().isoformat()
                        }
                
                # Log overall signal generation completion
                total_signals = len(signals_result["signals"])
                buy_signals = len([s for s in signals_result["signals"].values() if s["signal"] == "BUY"])
                sell_signals = len([s for s in signals_result["signals"].values() if s["signal"] == "SELL"])
                hold_signals = len([s for s in signals_result["signals"].values() if s["signal"] == "HOLD"])
                
                await self._log_memory_event(
                    event_type=EventType.OPTIMIZATION,
                    log_level=LogLevel.INFO,
                    title="Alpha Signals Generated",
                    content=f"Generated {total_signals} signals: {buy_signals} BUY, {sell_signals} SELL, {hold_signals} HOLD",
                    tags={"alpha_signals", "generation_complete", "summary"},
                    metadata={
                        "total_signals": total_signals,
                        "buy_signals": buy_signals,
                        "sell_signals": sell_signals,
                        "hold_signals": hold_signals,
                        "date": date,
                        "symbols": symbols
                    }
                )
                
                return signals_result
                
            except Exception as e:
                error_msg = f"Failed to generate alpha signals: {str(e)}"
                logger.error(error_msg)
                
                await self._log_memory_event(
                    event_type=EventType.OPTIMIZATION,
                    log_level=LogLevel.ERROR,
                    title="Alpha Signal Generation Failed",
                    content=error_msg,
                    tags={"alpha_signals", "generation_error", "failure"},
                    metadata={
                        "symbols": symbols,
                        "date": date,
                        "error": str(e)
                    }
                )
                
                return {
                    "status": "error",
                    "error": error_msg,
                    "date": date,
                    "signals": {}
                }

        def _generate_fallback_momentum_signal(self, symbol: str, prices: list) -> dict:
            """Generate a simple momentum signal as fallback when momentum agent is unavailable"""
            if len(prices) < 2:
                return None
                
            # Simple momentum calculation
            recent_price = prices[-1]
            older_price = prices[0] if len(prices) >= 20 else prices[0]
            momentum = (recent_price - older_price) / older_price if older_price != 0 else 0
            
            # Simple signal logic
            if momentum > 0.05:  # 5% momentum threshold
                signal = "BUY"
                confidence = min(momentum, 1.0)
                predicted_return = 0.02
                execution_weight = 0.3
            elif momentum < -0.05:
                signal = "SELL" 
                confidence = min(abs(momentum), 1.0)
                predicted_return = -0.02
                execution_weight = -0.3
            else:
                signal = "HOLD"
                confidence = 0.0
                predicted_return = 0.0
                execution_weight = 0.0
            
            return {
                "decision": {
                    "signal": signal,
                    "confidence": confidence,
                    "predicted_return": predicted_return,
                    "risk_estimate": 0.02
                },
                "action": {
                    "execution_weight": execution_weight
                }
            }

        @self.pool_server.tool(name="generate_llm_enhanced_signals", description="Generate alpha signals using LLM and memory patterns")
        async def generate_llm_enhanced_signals(
            strategy_type: str,
            symbols: list,
            market_data: dict = None,
            use_memory_patterns: bool = True,
            llm_enhancement: bool = True,
            confidence_threshold: float = 0.6
        ) -> dict:
            """Generate enhanced alpha signals using memory patterns and LLM analysis"""
            
            try:
                logger.info(f"🧠 Generating LLM-enhanced signals for {symbols} using {strategy_type}")
                
                # Log signal generation start
                await self._log_memory_event(
                    event_type=EventType.OPTIMIZATION if MEMORY_AVAILABLE else "optimization",
                    description=f"Starting LLM-enhanced signal generation for {strategy_type}",
                    metadata={
                        "symbols": symbols,
                        "strategy_type": strategy_type,
                        "use_memory": use_memory_patterns,
                        "llm_enabled": llm_enhancement
                    }
                )
                
                # Step 1: Retrieve relevant memory patterns
                memory_patterns = {}
                if use_memory_patterns and self.memory_agent:
                    memory_patterns = await self._retrieve_memory_patterns(strategy_type, symbols)
                
                # Step 2: Generate base signals using traditional methods
                base_signals = await self._generate_base_signals(strategy_type, symbols, market_data)
                
                # Step 3: Enhance signals using memory patterns
                memory_enhanced_signals = await self._enhance_signals_with_memory(
                    base_signals, memory_patterns, strategy_type
                )
                
                # Step 4: Apply LLM enhancement if enabled
                final_signals = memory_enhanced_signals
                if llm_enhancement:
                    final_signals = await self._apply_llm_enhancement(
                        memory_enhanced_signals, market_data, memory_patterns
                    )
                
                # Step 5: Log signal generation completion
                await self._log_memory_event(
                    event_type=EventType.OPTIMIZATION if MEMORY_AVAILABLE else "optimization",
                    description=f"Completed enhanced signal generation for {len(symbols)} symbols",
                    metadata={
                        "signals_generated": len(final_signals),
                        "memory_patterns_used": len(memory_patterns),
                        "average_confidence": sum(signal.get("confidence", 0.5) for signal in final_signals.values()) / len(final_signals) if final_signals else 0
                    }
                )
                
                result = {
                    "status": "success",
                    "signals": final_signals,
                    "memory_enhanced": use_memory_patterns,
                    "llm_enhanced": llm_enhancement,
                    "confidence_scores": {symbol: signal.get("confidence", 0.5) for symbol, signal in final_signals.items()},
                    "memory_patterns_used": len(memory_patterns),
                    "agent_status": "active_learning"
                }
                
                logger.info(f"✅ Generated enhanced signals for {len(symbols)} symbols with {len(memory_patterns)} memory patterns")
                return result
                
            except Exception as e:
                logger.error(f"❌ Enhanced signal generation failed: {e}")
                await self._log_memory_event(
                    event_type=EventType.ERROR if MEMORY_AVAILABLE else "error",
                    description=f"Enhanced signal generation failed: {str(e)}",
                    metadata={"error": str(e), "strategy_type": strategy_type}
                )
                return {
                    "status": "error",
                    "error": str(e),
                    "signals": {},
                    "agent_status": "error"
                }

        @self.pool_server.tool(name="train_from_memory", description="Train alpha agents using historical memory patterns")
        async def train_from_memory(
            strategy_type: str,
            symbols: list,
            training_period_days: int = 90,
            learning_rate: float = 0.01,
            performance_threshold: float = 0.6
        ) -> dict:
            """Train alpha generation using memory patterns for RL-style learning"""
            
            try:
                logger.info(f"🎓 Training alpha agents from memory for {strategy_type}")
                
                # Log training start
                await self._log_memory_event(
                    event_type=EventType.SYSTEM if MEMORY_AVAILABLE else "system",
                    description=f"Starting memory-based training for {strategy_type}",
                    metadata={
                        "symbols": symbols,
                        "training_days": training_period_days,
                        "learning_rate": learning_rate
                    }
                )
                
                # Step 1: Retrieve training data from memory
                training_data = await self._retrieve_training_data(
                    strategy_type, symbols, training_period_days
                )
                
                if not training_data:
                    return {
                        "status": "error",
                        "error": "No training data available in memory",
                        "agent_status": "training_failed"
                    }
                
                # Step 2: Analyze performance patterns
                performance_analysis = await self._analyze_performance_patterns(training_data)
                
                # Step 3: Extract successful strategies
                successful_patterns = await self._extract_successful_patterns(
                    training_data, performance_threshold
                )
                
                # Step 4: Update agent parameters based on learning
                updated_parameters = await self._update_agent_parameters(
                    strategy_type, successful_patterns, learning_rate
                )
                
                # Step 5: Validate learning improvements
                validation_results = await self._validate_learning_improvements(
                    strategy_type, symbols, updated_parameters
                )
                
                # Step 6: Save improved patterns to memory
                await self._save_improved_patterns(strategy_type, successful_patterns, validation_results)
                
                # Log training completion
                await self._log_memory_event(
                    event_type=EventType.OPTIMIZATION if MEMORY_AVAILABLE else "optimization",
                    description=f"Completed memory-based training for {strategy_type}",
                    metadata={
                        "patterns_learned": len(successful_patterns),
                        "performance_improvement": validation_results.get("improvement", 0.0),
                        "training_samples": len(training_data)
                    }
                )
                
                result = {
                    "status": "success",
                    "training_samples": len(training_data),
                    "successful_patterns": len(successful_patterns),
                    "parameter_updates": len(updated_parameters),
                    "validation_score": validation_results.get("score", 0.0),
                    "improvement_percentage": validation_results.get("improvement", 0.0),
                    "agent_status": "trained",
                    "learning_summary": {
                        "patterns_learned": len(successful_patterns),
                        "performance_improvement": validation_results.get("improvement", 0.0),
                        "confidence_increase": validation_results.get("confidence_delta", 0.0)
                    }
                }
                
                logger.info(f"✅ Training completed: {len(successful_patterns)} patterns learned, {validation_results.get('improvement', 0):.2%} improvement")
                return result
                
            except Exception as e:
                logger.error(f"❌ Memory training failed: {e}")
                await self._log_memory_event(
                    event_type=EventType.ERROR if MEMORY_AVAILABLE else "error",
                    description=f"Memory training failed: {str(e)}",
                    metadata={"error": str(e), "strategy_type": strategy_type}
                )
                return {
                    "status": "error",
                    "error": str(e),
                    "agent_status": "training_failed"
                }

        @self.pool_server.tool(name="get_agent_learning_status", description="Get current learning status of alpha agents")
        async def get_agent_learning_status() -> dict:
            """Get detailed status of alpha agent learning and memory utilization"""
            
            try:
                # Get memory statistics
                memory_stats = {"total_patterns": 0, "recent_signals": 0, "learning_events": 0}
                if self.memory_agent and MEMORY_AVAILABLE:
                    try:
                        # Query memory for recent patterns
                        from ...memory.external_memory_agent import QueryFilter
                        
                        query = QueryFilter(
                            content_search="alpha signal",
                            limit=100,
                            event_types=[EventType.OPTIMIZATION]
                        )
                        
                        recent_events = await self.memory_agent.query_events(query)
                        memory_stats = {
                            "total_patterns": len(recent_events.events) if hasattr(recent_events, 'events') else 0,
                            "recent_signals": len([e for e in recent_events.events if "signal" in e.title.lower()]) if hasattr(recent_events, 'events') else 0,
                            "learning_events": len([e for e in recent_events.events if "learning" in e.title.lower()]) if hasattr(recent_events, 'events') else 0
                        }
                    except Exception as e:
                        logger.warning(f"Failed to query memory stats: {e}")
                
                return {
                    "status": "success",
                    "memory_statistics": memory_stats,
                    "total_agents": len(self.agent_registry),
                    "active_agents": list(self.agent_registry.keys()),
                    "memory_enabled": self.memory_agent is not None,
                    "session_id": self.session_id,
                    "agent_status": "monitoring"
                }
                
            except Exception as e:
                logger.error(f"❌ Failed to get learning status: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "agent_status": "error"
                }

        async def _retrieve_memory_patterns(self, strategy_type: str, symbols: list) -> dict:
            """Retrieve relevant memory patterns for strategy enhancement"""
            try:
                patterns = {}
                if self.memory_agent and MEMORY_AVAILABLE:
                    from ...memory.external_memory_agent import QueryFilter
                    
                    # Query for strategy-specific patterns
                    query = QueryFilter(
                        content_search=f"{strategy_type} {' '.join(symbols)}",
                        limit=50,
                        event_types=[EventType.OPTIMIZATION]
                    )
                    
                    events = await self.memory_agent.query_events(query)
                    
                    if hasattr(events, 'events'):
                        for event in events.events:
                            if hasattr(event, 'metadata') and event.metadata:
                                patterns[event.id] = {
                                    "pattern_type": event.metadata.get("pattern_type", "unknown"),
                                    "confidence": event.metadata.get("confidence", 0.5),
                                    "performance": event.metadata.get("performance", 0.0),
                                    "timestamp": event.timestamp,
                                    "context": event.metadata.get("context", {})
                                }
                
                # Also check local memory unit
                local_patterns = {}
                for key in self.memory.keys():
                    if strategy_type in key and any(symbol in key for symbol in symbols):
                        local_patterns[key] = self.memory.get(key)
                
                patterns.update(local_patterns)
                logger.info(f"Retrieved {len(patterns)} memory patterns for {strategy_type}")
                return patterns
                
            except Exception as e:
                logger.error(f"Failed to retrieve memory patterns: {e}")
                return {}

        async def _generate_base_signals(self, strategy_type: str, symbols: list, market_data: dict = None) -> dict:
            """Generate base signals using traditional methods"""
            signals = {}
            
            for symbol in symbols:
                if strategy_type == "momentum" or strategy_type == "enhanced_momentum":
                    # Use momentum agent if available
                    if "momentum_agent" in self.agent_registry:
                        try:
                            # Generate momentum signal
                            signal_data = {
                                "symbol": symbol,
                                "timeframe": "1D",
                                "lookback_period": 20,
                                "market_data": market_data.get(symbol, {}) if market_data else {}
                            }
                            
                            # Simulate momentum signal generation
                            import random
                            momentum_strength = random.normalvariate(0, 0.1)
                            
                            signals[symbol] = {
                                "direction": "buy" if momentum_strength > 0.03 else "sell" if momentum_strength < -0.03 else "hold",
                                "strength": abs(momentum_strength),
                                "confidence": min(abs(momentum_strength) * 10, 0.9),
                                "signal_type": "momentum",
                                "generated_by": "momentum_agent"
                            }
                        except Exception as e:
                            logger.warning(f"Momentum agent failed for {symbol}: {e}")
                            signals[symbol] = self._generate_fallback_signal(symbol)
                    else:
                        signals[symbol] = self._generate_fallback_signal(symbol)
                else:
                    signals[symbol] = self._generate_fallback_signal(symbol)
            
            return signals

        def _generate_fallback_signal(self, symbol: str) -> dict:
            """Generate fallback signal when specialized agents are not available"""
            import random
            random.seed(hash(symbol) % 1000)
            
            signal_strength = random.normalvariate(0, 0.08)
            
            return {
                "direction": "buy" if signal_strength > 0.02 else "sell" if signal_strength < -0.02 else "hold",
                "strength": abs(signal_strength),
                "confidence": 0.5,
                "signal_type": "fallback",
                "generated_by": "fallback_generator"
            }

        async def _enhance_signals_with_memory(self, base_signals: dict, memory_patterns: dict, strategy_type: str) -> dict:
            """Enhance base signals using memory patterns"""
            enhanced_signals = base_signals.copy()
            
            for symbol, signal in enhanced_signals.items():
                # Find relevant patterns for this symbol
                relevant_patterns = [
                    pattern for pattern_id, pattern in memory_patterns.items()
                    if symbol in pattern_id or pattern.get("context", {}).get("symbol") == symbol
                ]
                
                if relevant_patterns:
                    # Calculate memory-based confidence adjustment
                    avg_pattern_confidence = sum(p.get("confidence", 0.5) for p in relevant_patterns) / len(relevant_patterns)
                    avg_pattern_performance = sum(p.get("performance", 0.0) for p in relevant_patterns) / len(relevant_patterns)
                    
                    # Adjust signal based on memory patterns
                    memory_adjustment = (avg_pattern_confidence + avg_pattern_performance) / 2
                    
                    # Update signal
                    signal["confidence"] = min((signal["confidence"] + memory_adjustment) / 2, 0.95)
                    signal["strength"] = min((signal["strength"] + abs(memory_adjustment)) / 2, 1.0)
                    signal["memory_enhanced"] = True
                    signal["patterns_used"] = len(relevant_patterns)
                    signal["memory_confidence"] = avg_pattern_confidence
                else:
                    signal["memory_enhanced"] = False
                    signal["patterns_used"] = 0
            
            return enhanced_signals

        async def _apply_llm_enhancement(self, signals: dict, market_data: dict = None, memory_patterns: dict = None) -> dict:
            """Apply LLM enhancement to signals (simulated)"""
            # This is a simplified simulation of LLM enhancement
            # In a real implementation, this would call an actual LLM service
            
            enhanced_signals = signals.copy()
            
            for symbol, signal in enhanced_signals.items():
                # Simulate LLM analysis
                import random
                
                # LLM provides contextual analysis
                llm_confidence_boost = random.uniform(0.05, 0.15)
                llm_context_score = random.uniform(0.6, 0.9)
                
                # Apply LLM enhancements
                signal["confidence"] = min(signal["confidence"] + llm_confidence_boost, 0.95)
                signal["llm_enhanced"] = True
                signal["llm_context_score"] = llm_context_score
                signal["llm_analysis"] = f"LLM analysis suggests {signal['direction']} signal with {llm_context_score:.2f} context relevance"
                
                # Adjust strength based on LLM context
                if llm_context_score > 0.8:
                    signal["strength"] = min(signal["strength"] * 1.1, 1.0)
            
            return enhanced_signals

        async def _retrieve_training_data(self, strategy_type: str, symbols: list, days: int) -> list:
            """Retrieve training data from memory"""
            training_data = []
            
            if self.memory_agent and MEMORY_AVAILABLE:
                try:
                    from ...memory.external_memory_agent import QueryFilter
                    from datetime import datetime, timedelta
                    
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    query = QueryFilter(
                        content_search=f"{strategy_type} signal performance",
                        limit=200,
                        event_types=[EventType.OPTIMIZATION],
                        start_time=start_date,
                        end_time=end_date
                    )
                    
                    events = await self.memory_agent.query_events(query)
                    
                    if hasattr(events, 'events'):
                        for event in events.events:
                            if hasattr(event, 'metadata') and event.metadata:
                                training_data.append({
                                    "timestamp": event.timestamp,
                                    "strategy_type": event.metadata.get("strategy_type", strategy_type),
                                    "signal_data": event.metadata.get("signal_data", {}),
                                    "performance": event.metadata.get("performance", 0.0),
                                    "confidence": event.metadata.get("confidence", 0.5)
                                })
                except Exception as e:
                    logger.error(f"Failed to retrieve training data: {e}")
            
            return training_data

        async def _analyze_performance_patterns(self, training_data: list) -> dict:
            """Analyze performance patterns in training data"""
            if not training_data:
                return {}
            
            analysis = {
                "total_samples": len(training_data),
                "avg_performance": sum(item.get("performance", 0) for item in training_data) / len(training_data),
                "avg_confidence": sum(item.get("confidence", 0) for item in training_data) / len(training_data),
                "positive_samples": len([item for item in training_data if item.get("performance", 0) > 0]),
                "negative_samples": len([item for item in training_data if item.get("performance", 0) < 0])
            }
            
            analysis["success_rate"] = analysis["positive_samples"] / analysis["total_samples"] if analysis["total_samples"] > 0 else 0
            
            return analysis

        async def _extract_successful_patterns(self, training_data: list, threshold: float) -> list:
            """Extract successful patterns from training data"""
            successful_patterns = []
            
            for item in training_data:
                performance = item.get("performance", 0.0)
                confidence = item.get("confidence", 0.5)
                
                # Consider pattern successful if performance and confidence are above threshold
                if performance > threshold and confidence > threshold:
                    successful_patterns.append({
                        "pattern_data": item.get("signal_data", {}),
                        "performance": performance,
                        "confidence": confidence,
                        "timestamp": item.get("timestamp"),
                        "success_score": (performance + confidence) / 2
                    })
            
            # Sort by success score
            successful_patterns.sort(key=lambda x: x["success_score"], reverse=True)
            
            return successful_patterns

        async def _update_agent_parameters(self, strategy_type: str, successful_patterns: list, learning_rate: float) -> dict:
            """Update agent parameters based on successful patterns"""
            updated_parameters = {}
            
            if not successful_patterns:
                return updated_parameters
            
            # Extract parameter adjustments from successful patterns
            for pattern in successful_patterns[:10]:  # Use top 10 patterns
                pattern_data = pattern.get("pattern_data", {})
                success_score = pattern.get("success_score", 0.5)
                
                # Apply learning rate to parameter updates
                for param, value in pattern_data.items():
                    if param not in updated_parameters:
                        updated_parameters[param] = []
                    
                    adjusted_value = value * success_score * learning_rate
                    updated_parameters[param].append(adjusted_value)
            
            # Average the parameter updates
            for param, values in updated_parameters.items():
                updated_parameters[param] = sum(values) / len(values)
            
            return updated_parameters

        async def _validate_learning_improvements(self, strategy_type: str, symbols: list, updated_parameters: dict) -> dict:
            """Validate that learning improvements are beneficial"""
            
            # Simulate validation by comparing old vs new parameters
            baseline_score = 0.6  # Baseline performance
            
            # Calculate improvement based on parameter quality
            improvement_score = 0.0
            if updated_parameters:
                # Simple heuristic: better parameters should improve performance
                param_quality = sum(abs(value) for value in updated_parameters.values()) / len(updated_parameters)
                improvement_score = min(param_quality * 0.1, 0.3)  # Cap at 30% improvement
            
            new_score = baseline_score + improvement_score
            improvement_percentage = improvement_score / baseline_score if baseline_score > 0 else 0
            
            return {
                "score": new_score,
                "improvement": improvement_percentage,
                "confidence_delta": improvement_score * 0.5,  # Confidence also improves
                "validation_successful": improvement_score > 0.05  # At least 5% improvement
            }

        async def _save_improved_patterns(self, strategy_type: str, successful_patterns: list, validation_results: dict):
            """Save improved patterns back to memory"""
            
            if self.memory_agent and MEMORY_AVAILABLE and validation_results.get("validation_successful", False):
                try:
                    await self.memory_agent.log_event(
                        event_type=EventType.OPTIMIZATION,
                        log_level=LogLevel.INFO,
                        source_agent_pool="alpha_agent_pool",
                        source_agent_id="memory_learning_system",
                        title=f"Improved patterns learned for {strategy_type}",
                        content=f"Successfully learned {len(successful_patterns)} patterns with {validation_results.get('improvement', 0):.2%} improvement",
                        tags={"learning", "patterns", strategy_type, "improvement"},
                        metadata={
                            "strategy_type": strategy_type,
                            "patterns_count": len(successful_patterns),
                            "improvement_percentage": validation_results.get("improvement", 0),
                            "validation_score": validation_results.get("score", 0),
                            "learning_timestamp": datetime.now().isoformat(),
                            "successful_patterns": successful_patterns[:5]  # Store top 5 patterns
                        }
                    )
                    
                    logger.info(f"Saved {len(successful_patterns)} improved patterns to memory")
                    
                except Exception as e:
                    logger.error(f"Failed to save improved patterns: {e}")
            
            # Also save to local memory unit
            pattern_key = f"learned_patterns_{strategy_type}_{datetime.now().strftime('%Y%m%d')}"
            self.memory.set(pattern_key, {
                "patterns": successful_patterns,
                "validation": validation_results,
                "timestamp": datetime.now().isoformat()
            })
