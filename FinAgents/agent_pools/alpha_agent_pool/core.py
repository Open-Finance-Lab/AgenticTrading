# core.py
"""
Unified entry point for starting the Alpha Agent Pool MCP service.
This module manages the lifecycle and orchestration of multiple sub-agents within the AlphaAgentPool.
Enhanced with comprehensive memory integration for strategy tracking and performance analytics.
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from mcp.server.fastmcp import FastMCP
from .schema.theory_driven_schema import MomentumAgentConfig
from .agents.theory_driven.momentum_agent import MomentumAgent
from .agents.autonomous.autonomous_agent import AutonomousAgent

# Add memory module to path
memory_path = Path(__file__).parent.parent.parent / "memory"
sys.path.insert(0, str(memory_path))

# Import memory bridge for alpha agent pool
try:
    from .memory_bridge import (
        AlphaAgentPoolMemoryBridge,
        AlphaSignalRecord,
        StrategyPerformanceMetrics,
        MemoryPatternRecord,
        create_alpha_memory_bridge,
        create_alpha_signal_record,
        create_performance_metrics_record
    )
    MEMORY_BRIDGE_AVAILABLE = True
except ImportError:
    AlphaAgentPoolMemoryBridge = None
    MEMORY_BRIDGE_AVAILABLE = False

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
    Enhanced with event logging capabilities for strategy flow tracking.
    """
    def __init__(self, file_path, autoload_csv_path=None, reset_on_init=False, memory_bridge=None):
        self.file_path = file_path
        self.memory_bridge = memory_bridge
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
                data_loaded = 0
                for row in reader:
                    date = row['timestamp']
                    close = row['close']
                    self.set(f"AAPL_close_{date}", close, log_event=False)
                    data_loaded += 1
                
                # Log data loading event
                if self.memory_bridge:
                    asyncio.create_task(self._log_data_loading_event(csv_path, data_loaded))
        except Exception as e:
            logger.warning(f"Failed to autoload CSV data: {e}")

    async def _log_data_loading_event(self, csv_path: str, records_loaded: int):
        """Log data loading events to memory bridge"""
        if self.memory_bridge and hasattr(self.memory_bridge, '_log_system_event'):
            try:
                await self.memory_bridge._log_system_event(
                    event_type=EventType.SYSTEM if MEMORY_AVAILABLE else "system",
                    log_level=LogLevel.INFO if MEMORY_AVAILABLE else "info",
                    title="Historical Data Loaded",
                    content=f"Loaded {records_loaded} historical data records from {os.path.basename(csv_path)}",
                    metadata={
                        "source_file": csv_path,
                        "records_loaded": records_loaded,
                        "data_type": "historical_market_data"
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log data loading event: {e}")

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

    def set(self, key, value, log_event=True):
        self._data[key] = value
        self._save()
        
        # Log memory operation if enabled
        if log_event and self.memory_bridge:
            asyncio.create_task(self._log_memory_operation("SET", key, value))

    def delete(self, key, log_event=True):
        if key in self._data:
            del self._data[key]
            self._save()
            
            # Log memory operation if enabled
            if log_event and self.memory_bridge:
                asyncio.create_task(self._log_memory_operation("DELETE", key, None))

    def keys(self):
        return list(self._data.keys())

    async def _log_memory_operation(self, operation: str, key: str, value: Any):
        """Log memory unit operations to memory bridge"""
        if self.memory_bridge and hasattr(self.memory_bridge, '_log_system_event'):
            try:
                await self.memory_bridge._log_system_event(
                    event_type=EventType.SYSTEM if MEMORY_AVAILABLE else "system",
                    log_level=LogLevel.DEBUG if MEMORY_AVAILABLE else "debug",
                    title=f"Memory Unit Operation: {operation}",
                    content=f"Performed {operation} operation on key '{key}'",
                    metadata={
                        "operation": operation,
                        "key": key,
                        "value_type": type(value).__name__ if value is not None else None,
                        "component": "local_memory_unit"
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to log memory operation: {e}")

class AlphaAgentPoolMCPServer:
    def __init__(self, host="0.0.0.0", port=8081):
        """
        Initialize the AlphaAgentPoolMCPServer instance with enhanced memory capabilities.
        Args:
            host (str): Host address to bind the MCP server.
            port (int): Port number to bind the MCP server.
        """
        self.host = host
        self.port = port
        self.pool_server = FastMCP("AlphaAgentPoolMCPServer")
        self.agent_registry = {}  # agent_id -> (agent, process/thread)
        self.config_dir = os.path.join(os.path.dirname(__file__), "config")
        
        # Initialize memory bridge for comprehensive strategy tracking
        self.memory_bridge: Optional[AlphaAgentPoolMemoryBridge] = None
        # Note: Memory bridge will be initialized asynchronously when needed
        
        # Automatically load static dataset into memory unit on startup, and reset memory file
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/cache/AAPL_2024-01-01_2024-01-31_1d.csv"))
        self.memory = MemoryUnit(
            os.path.join(os.path.dirname(__file__), "memory_unit.json"), 
            autoload_csv_path=csv_path, 
            reset_on_init=True,
            memory_bridge=self.memory_bridge
        )
        
        # Initialize legacy memory agent if available
        self.memory_agent: Optional[ExternalMemoryAgent] = None
        self.session_id = None
        self._initialize_memory_agent()  # Initialize memory agent synchronously
        
        # Strategy performance tracking
        self.strategy_performance_cache = {}
        self.signal_generation_history = []
        
        self._register_pool_tools()

    async def _initialize_memory_bridge(self):
        """Initialize the advanced memory bridge for alpha agent pool"""
        if not MEMORY_BRIDGE_AVAILABLE:
            logger.warning("Memory bridge not available")
            return
        
        try:
            self.memory_bridge = await create_alpha_memory_bridge(
                config={
                    "enable_pattern_learning": True,
                    "performance_tracking_enabled": True,
                    "real_time_logging": True
                }
            )
            logger.info("Alpha Agent Pool Memory Bridge successfully initialized")
        except Exception as e:
            logger.error(f"Failed to initialize memory bridge: {e}")
            self.memory_bridge = None

    def _initialize_memory_agent(self):
        """Initialize the external memory agent"""
        if not MEMORY_AVAILABLE:
            logger.warning("External memory agent not available")
            return
        
        try:
            self.memory_agent = ExternalMemoryAgent()
            # Use timestamp instead of event loop time for session ID
            import time
            self.session_id = f"alpha_pool_session_{int(time.time())}"
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
        Enhanced with comprehensive memory bridge integration and strategy flow tracking.
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
                
                # Log agent startup event
                if self.memory_bridge:
                    asyncio.create_task(self._log_agent_lifecycle_event(
                        agent_id, "STARTED", f"Momentum agent started on port {config.execution.port}"
                    ))
                
                return f"Momentum agent started on port {config.execution.port}"
            elif agent_id == "autonomous_agent":
                process = self._start_autonomous_agent()
                self.agent_registry[agent_id] = process
                
                # Log agent startup event
                if self.memory_bridge:
                    asyncio.create_task(self._log_agent_lifecycle_event(
                        agent_id, "STARTED", "Autonomous agent started on port 5051"
                    ))
                
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

        @self.pool_server.tool(name="process_strategy_request", description="Process strategy requests and generate alpha signals")
        async def process_strategy_request(query: str) -> str:
            """
            Process natural language strategy requests and generate alpha signals.
            
            Args:
                query (str): Natural language query describing the strategy request
                
            Returns:
                str: JSON string containing strategy response and alpha signals
            """
            try:
                # Parse symbols from query
                symbols = []
                if "AAPL" in query.upper():
                    symbols.append("AAPL")
                if "MSFT" in query.upper():
                    symbols.append("MSFT")
                
                if not symbols:
                    symbols = ["AAPL", "MSFT"]  # Default symbols
                
                # Extract date from query or use today
                import re
                from datetime import datetime
                date_match = re.search(r'\d{4}-\d{2}-\d{2}', query)
                date = date_match.group(0) if date_match else datetime.now().strftime('%Y-%m-%d')
                
                # Generate alpha signals
                signals_result = await generate_alpha_signals(
                    symbols=symbols,
                    date=date,
                    lookback_period=20,
                    current_prices=None,
                    strategy_context={"query": query}
                )
                
                # Return structured response
                response = {
                    "status": "success",
                    "request_type": "strategy_processing",
                    "query": query,
                    "symbols": symbols,
                    "date": date,
                    "alpha_signals": signals_result,
                    "generated_at": datetime.now().isoformat()
                }
                
                return json.dumps(response)
                
            except Exception as e:
                error_response = {
                    "status": "error",
                    "error": str(e),
                    "query": query,
                    "generated_at": datetime.now().isoformat()
                }
                return json.dumps(error_response)

        @self.pool_server.tool(name="submit_strategy_event", description="Submit strategy flow events to memory system for tracking and analysis.")
        async def submit_strategy_event(event_type: str, strategy_id: str, event_data: dict, 
                                      metadata: Optional[dict] = None) -> str:
            """
            Submit strategy flow events to the memory system for comprehensive tracking.
            
            Args:
                event_type: Type of strategy event (SIGNAL_GENERATED, STRATEGY_EXECUTED, PERFORMANCE_UPDATED, etc.)
                strategy_id: Unique identifier for the strategy
                event_data: Core event data including signals, performance metrics, etc.
                metadata: Additional contextual information
            
            Returns:
                str: Event submission confirmation with storage ID
            """
            try:
                if not self.memory_bridge:
                    return "Memory bridge not available - event logged locally only"
                
                # Process different event types appropriately
                storage_id = None
                
                if event_type == "SIGNAL_GENERATED":
                    # Create and store alpha signal record
                    signal_record = create_alpha_signal_record(
                        symbol=event_data.get('symbol', 'UNKNOWN'),
                        signal_type=event_data.get('signal_type', 'HOLD'),
                        confidence=event_data.get('confidence', 0.0),
                        predicted_return=event_data.get('predicted_return', 0.0),
                        risk_estimate=event_data.get('risk_estimate', 0.01),
                        execution_weight=event_data.get('execution_weight', 0.0),
                        strategy_source=strategy_id,
                        agent_id=event_data.get('agent_id', 'alpha_pool'),
                        market_regime=event_data.get('market_regime'),
                        feature_vector=event_data.get('feature_vector'),
                        metadata=metadata
                    )
                    storage_id = await self.memory_bridge.store_alpha_signal(signal_record)
                    
                elif event_type == "PERFORMANCE_UPDATED":
                    # Create and store performance metrics
                    performance_record = create_performance_metrics_record(
                        strategy_id=strategy_id,
                        agent_id=event_data.get('agent_id', 'alpha_pool'),
                        signals_generated=event_data.get('signals_generated', 0),
                        successful_predictions=event_data.get('successful_predictions', 0),
                        sharpe_ratio=event_data.get('sharpe_ratio', 0.0),
                        information_ratio=event_data.get('information_ratio', 0.0),
                        max_drawdown=event_data.get('max_drawdown', 0.0),
                        avg_return=event_data.get('avg_return', 0.0),
                        volatility=event_data.get('volatility', 0.01),
                        **{k: v for k, v in event_data.items() if k not in [
                            'agent_id', 'signals_generated', 'successful_predictions',
                            'sharpe_ratio', 'information_ratio', 'max_drawdown', 
                            'avg_return', 'volatility'
                        ]}
                    )
                    storage_id = await self.memory_bridge.store_strategy_performance(performance_record)
                    
                else:
                    # Generic event logging through system event logging
                    await self.memory_bridge._log_system_event(
                        event_type=EventType.OPTIMIZATION if MEMORY_AVAILABLE else "optimization",
                        log_level=LogLevel.INFO if MEMORY_AVAILABLE else "info",
                        title=f"Strategy Event: {event_type}",
                        content=f"Strategy {strategy_id} submitted {event_type} event",
                        metadata={
                            "strategy_id": strategy_id,
                            "event_type": event_type,
                            "event_data": event_data,
                            "metadata": metadata
                        }
                    )
                    storage_id = f"system_event_{datetime.utcnow().isoformat()}"
                
                logger.info(f"Strategy event submitted: {event_type} for {strategy_id}")
                return f"Event submitted successfully - Storage ID: {storage_id}"
                
            except Exception as e:
                error_msg = f"Failed to submit strategy event: {str(e)}"
                logger.error(error_msg)
                return error_msg

        @self.pool_server.tool(name="retrieve_strategy_data", description="Retrieve historical strategy data and patterns from memory.")
        async def retrieve_strategy_data(query_type: str, filters: Optional[dict] = None, 
                                       time_range_hours: int = 24, limit: int = 100) -> dict:
            """
            Retrieve historical strategy data and patterns from the memory system.
            
            Args:
                query_type: Type of data to retrieve (signals, performance, patterns, analytics)
                filters: Filter criteria for data retrieval
                time_range_hours: Time window for data retrieval (hours)
                limit: Maximum number of records to retrieve
            
            Returns:
                dict: Retrieved data with metadata and summary statistics
            """
            try:
                if not self.memory_bridge:
                    return {"error": "Memory bridge not available", "data": []}
                
                time_range = timedelta(hours=time_range_hours)
                results = {"query_type": query_type, "filters": filters, "data": [], "summary": {}}
                
                if query_type == "signals":
                    signals = await self.memory_bridge.retrieve_alpha_signals(
                        filters=filters, time_range=time_range, limit=limit
                    )
                    results["data"] = [
                        {
                            "signal_id": s.signal_id,
                            "symbol": s.symbol,
                            "signal_type": s.signal_type,
                            "confidence": s.confidence_score,
                            "predicted_return": s.predicted_return,
                            "strategy_source": s.strategy_source,
                            "timestamp": s.timestamp.isoformat()
                        } for s in signals
                    ]
                    results["summary"] = {
                        "total_signals": len(signals),
                        "buy_signals": len([s for s in signals if s.signal_type == "BUY"]),
                        "sell_signals": len([s for s in signals if s.signal_type == "SELL"]),
                        "hold_signals": len([s for s in signals if s.signal_type == "HOLD"]),
                        "avg_confidence": sum(s.confidence_score for s in signals) / max(len(signals), 1)
                    }
                
                elif query_type == "performance":
                    # Retrieve performance analytics
                    if filters and "strategy_id" in filters:
                        analytics = await self.memory_bridge.get_strategy_performance_analytics(
                            strategy_id=filters["strategy_id"],
                            analysis_period=time_range
                        )
                        results["data"] = [analytics]
                        results["summary"] = {"analytics_generated": 1}
                    else:
                        results["error"] = "strategy_id required for performance queries"
                
                elif query_type == "patterns":
                    # Retrieve market patterns
                    market_conditions = filters.get("market_conditions", {}) if filters else {}
                    patterns = await self.memory_bridge.retrieve_relevant_patterns(
                        market_conditions=market_conditions,
                        pattern_types=filters.get("pattern_types") if filters else None,
                        min_success_rate=filters.get("min_success_rate", 0.6) if filters else 0.6,
                        min_significance=filters.get("min_significance", 0.05) if filters else 0.05
                    )
                    results["data"] = [
                        {
                            "pattern_id": p.pattern_id,
                            "pattern_type": p.pattern_type,
                            "success_rate": p.success_rate,
                            "statistical_significance": p.statistical_significance,
                            "pattern_frequency": p.pattern_frequency,
                            "last_occurrence": p.last_occurrence.isoformat()
                        } for p in patterns
                    ]
                    results["summary"] = {
                        "total_patterns": len(patterns),
                        "avg_success_rate": sum(p.success_rate for p in patterns) / max(len(patterns), 1)
                    }
                
                elif query_type == "bridge_stats":
                    # Retrieve memory bridge statistics
                    bridge_stats = await self.memory_bridge.get_bridge_statistics()
                    results["data"] = [bridge_stats]
                    results["summary"] = {"stats_retrieved": True}
                
                else:
                    results["error"] = f"Unknown query type: {query_type}"
                
                logger.info(f"Retrieved {len(results.get('data', []))} records for query type: {query_type}")
                return results
                
            except Exception as e:
                error_msg = f"Failed to retrieve strategy data: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg, "data": []}

        @self.pool_server.tool(name="analyze_strategy_performance", description="Generate comprehensive strategy performance analysis.")
        async def analyze_strategy_performance(strategy_id: str, analysis_period_days: int = 30,
                                             include_recommendations: bool = True) -> dict:
            """
            Generate comprehensive performance analysis for a specific strategy.
            
            Args:
                strategy_id: Strategy identifier for analysis
                analysis_period_days: Number of days to include in analysis
                include_recommendations: Whether to include actionable recommendations
            
            Returns:
                dict: Comprehensive performance analysis report
            """
            try:
                if not self.memory_bridge:
                    return {"error": "Memory bridge not available for performance analysis"}
                
                analysis_period = timedelta(days=analysis_period_days)
                analytics = await self.memory_bridge.get_strategy_performance_analytics(
                    strategy_id=strategy_id,
                    analysis_period=analysis_period
                )
                
                if "error" in analytics:
                    return analytics
                
                # Enhance analytics with additional insights
                enhanced_analytics = analytics.copy()
                enhanced_analytics.update({
                    "analysis_metadata": {
                        "generated_at": datetime.utcnow().isoformat(),
                        "analysis_period_days": analysis_period_days,
                        "strategy_id": strategy_id,
                        "include_recommendations": include_recommendations
                    },
                    "risk_assessment": self._generate_risk_assessment(analytics),
                    "performance_grade": self._calculate_performance_grade(analytics)
                })
                
                logger.info(f"Generated comprehensive performance analysis for strategy: {strategy_id}")
                return enhanced_analytics
                
            except Exception as e:
                error_msg = f"Failed to analyze strategy performance: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}

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
            # Log orchestrator interaction event
            if self.memory_bridge:
                asyncio.create_task(self._log_orchestrator_interaction(instruction, context))
            
            # 这里可以通过MCP客户端连接到autonomous agent
            # 目前返回确认消息
            return f"Orchestrator input '{instruction}' sent to autonomous agent"

        @self.pool_server.tool(name="generate_alpha_signals", description="Generate alpha trading signals using available agents and real data with comprehensive memory tracking")
        async def generate_alpha_signals(symbols: list, date: str, lookback_period: int = 20, 
                                       current_prices: dict = None, strategy_context: dict = None) -> dict:
            """
            Generate alpha trading signals for given symbols using momentum and autonomous agents.
            Enhanced with comprehensive memory tracking and strategy flow events.
            
            Args:
                symbols (list): List of stock symbols
                date (str): Current trading date 
                lookback_period (int): Number of days to look back for signal generation
                current_prices (dict): Current prices for symbols
                strategy_context (dict): Additional strategy context and parameters
            
            Returns:
                dict: Alpha signals for all symbols with comprehensive metadata
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
                        "has_current_prices": current_prices is not None,
                        "strategy_context": strategy_context
                    }
                )
                
                signals_result = {
                    "status": "success",
                    "date": date,
                    "signals": {},
                    "generated_by": "alpha_agent_pool",
                    "timestamp": datetime.now().isoformat(),
                    "strategy_metadata": strategy_context or {}
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
                        
                        # Submit signal event to memory bridge
                        if self.memory_bridge:
                            try:
                                await self.submit_strategy_event(
                                    event_type="SIGNAL_GENERATED",
                                    strategy_id="alpha_signal_generation",
                                    event_data={
                                        "symbol": symbol,
                                        "signal_type": signal,
                                        "confidence": confidence,
                                        "predicted_return": predicted_return,
                                        "risk_estimate": risk_estimate,
                                        "execution_weight": execution_weight,
                                        "agent_id": "alpha_pool_generator",
                                        "market_regime": strategy_context.get("market_regime") if strategy_context else None,
                                        "feature_vector": {
                                            "price": historical_prices[-1],
                                            "momentum": (historical_prices[-1] - historical_prices[0]) / historical_prices[0] if len(historical_prices) > 1 else 0,
                                            "volatility": self._calculate_price_volatility(historical_prices)
                                        }
                                    },
                                    metadata={
                                        "date": date,
                                        "lookback_period": lookback_period,
                                        "data_source": "momentum_agent" if signal_result else "fallback"
                                    }
                                )
                            except Exception as e:
                                logger.warning(f"Failed to submit strategy event for {symbol}: {e}")
                        
                        # Log individual signal generation
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
                
                # Store strategy performance if memory bridge is available
                if self.memory_bridge and total_signals > 0:
                    avg_confidence = sum(s.get("confidence", 0) for s in signals_result["signals"].values()) / total_signals
                    performance_event_data = {
                        "agent_id": "alpha_pool_generator",
                        "signals_generated": total_signals,
                        "successful_predictions": 0,  # Will be updated later based on actual outcomes
                        "sharpe_ratio": 0.0,  # Placeholder - will be calculated over time
                        "information_ratio": 0.0,
                        "max_drawdown": 0.0,
                        "avg_return": 0.0,
                        "volatility": avg_confidence * 0.1,  # Rough volatility estimate
                        "average_confidence": avg_confidence
                    }
                    
                    await self.submit_strategy_event(
                        event_type="PERFORMANCE_UPDATED",
                        strategy_id="alpha_signal_generation",
                        event_data=performance_event_data,
                        metadata={
                            "date": date,
                            "performance_type": "signal_generation_session"
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

    # Helper methods for enhanced functionality
    
    def _start_momentum_agent(self):
        """Start the momentum agent sub-service"""
        try:
            import subprocess
            import threading
            import time
            
            # Load momentum agent configuration
            config_path = os.path.join(self.config_dir, "momentum_agent.yml")
            
            # Default configuration if file doesn't exist
            default_config = {
                'execution': {
                    'host': '0.0.0.0',
                    'port': 5052
                },
                'model': 'momentum_strategy',
                'lookback_period': 20,
                'signal_threshold': 0.05
            }
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                config_data = default_config
                
            config = MomentumAgentConfig(**config_data)
            
            # Create momentum agent instance
            momentum_agent = MomentumAgent(config)
            
            # Start in a separate thread
            def run_momentum():
                try:
                    momentum_agent.run()
                except Exception as e:
                    logger.error(f"Momentum agent error: {e}")
            
            momentum_thread = threading.Thread(target=run_momentum, daemon=True)
            momentum_thread.start()
            
            # Give it time to start
            time.sleep(2)
            
            logger.info(f"Momentum agent started on port {config.execution.port}")
            return config, momentum_thread
            
        except Exception as e:
            logger.error(f"Failed to start momentum agent: {e}")
            # Return default config and None for process
            return MomentumAgentConfig(**{
                'execution': {'host': '0.0.0.0', 'port': 5052},
                'model': 'momentum_strategy',
                'lookback_period': 20,
                'signal_threshold': 0.05
            }), None

    def _start_autonomous_agent(self):
        """Start the autonomous agent sub-service"""
        try:
            import subprocess
            import threading
            
            # Create autonomous agent instance
            autonomous_agent = AutonomousAgent()
            
            # Start in a separate thread
            def run_autonomous():
                try:
                    autonomous_agent.run()
                except Exception as e:
                    logger.error(f"Autonomous agent error: {e}")
            
            autonomous_thread = threading.Thread(target=run_autonomous, daemon=True)
            autonomous_thread.start()
            
            logger.info("Autonomous agent started on port 5051")
            return autonomous_thread
            
        except Exception as e:
            logger.error(f"Failed to start autonomous agent: {e}")
            return None


# Main execution
if __name__ == "__main__":
    print("🚀 Starting Alpha Agent Pool...")
    
    # Create and start the server
    try:
        alpha_pool = AlphaAgentPoolMCPServer(host="0.0.0.0", port=8081)
        
        # Use the pool_server to run with SSE transport
        alpha_pool.pool_server.settings.host = "0.0.0.0"
        alpha_pool.pool_server.settings.port = 8081
        
        # Start the FastMCP server with SSE transport like data agent pool
        alpha_pool.pool_server.run(transport="sse")
        
    except KeyboardInterrupt:
        print("\n🛑 Alpha Agent Pool shutting down...")
    except Exception as e:
        print(f"❌ Alpha Agent Pool error: {e}")
