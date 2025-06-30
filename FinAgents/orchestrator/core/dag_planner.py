"""
FinAgent Orchestration Core System with LLM-Enhanced DAG Planning

This module implements the top-level orchestrator for the FinAgent ecosystem, providing:
- LLM-enhanced DAG-based task planning and execution
- Dynamic strategy decomposition via natural language processing
- Multi-agent pool coordination (Data, Alpha, Execution agents)
- Memory-enhanced reinforcement learning capabilities
- Sandbox testing environment
- Comprehensive backtesting framework

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Orchestrator Core                        │
    │  ┌─────────────────┐    ┌──────────────────────────────────┐ │
    │  │ LLM-Enhanced    │────│      Execution Engine           │ │
    │  │ DAG Planner     │    │                                  │ │
    │  └─────────────────┘    └──────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────┘
                                      │
    ┌─────────────────────────────────────────────────────────────┐
    │                  Agent Pool Layer                           │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │ Data Agent  │  │ Alpha Agent │  │ Execution Agent     │  │
    │  │ Pool        │  │ Pool        │  │ Pool                │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
                                      │
    ┌─────────────────────────────────────────────────────────────┐
    │                 Memory & RL Layer                           │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │ Memory      │  │ RL Policy   │  │ Backtesting         │  │
    │  │ Agent       │  │ Engine      │  │ Engine              │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘

Author: FinAgent Team
Version: 2.0.0 (LLM-Enhanced)
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import networkx as nx
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.client.sse import sse_client
from mcp import ClientSession

# Import LLM integration
from .llm_integration import NaturalLanguageProcessor, LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("FinAgentOrchestrator")


class TaskStatus(Enum):
    """Enumeration for task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentPoolType(Enum):
    """Enumeration for different agent pool types"""
    DATA_AGENT_POOL = "data_agent_pool"
    ALPHA_AGENT_POOL = "alpha_agent_pool"
    EXECUTION_AGENT_POOL = "execution_agent_pool"
    MEMORY_AGENT = "memory_agent"


@dataclass
class TaskNode:
    """
    Represents a task node in the execution DAG.
    
    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task (data_fetch, signal_generation, etc.)
        agent_pool: Target agent pool for execution
        tool_name: Specific tool/method to invoke
        parameters: Task execution parameters
        dependencies: List of prerequisite task IDs
        status: Current execution status
        result: Task execution result
        metadata: Additional task metadata
        created_at: Task creation timestamp
        started_at: Task execution start timestamp
        completed_at: Task completion timestamp
        error_message: Error details if task failed
    """
    task_id: str
    task_type: str
    agent_pool: AgentPoolType
    tool_name: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task node to dictionary representation"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key in ['created_at', 'started_at', 'completed_at']:
            if data[key] is not None:
                data[key] = data[key].isoformat()
        # Convert enum values to strings
        data['status'] = data['status'].value
        data['agent_pool'] = data['agent_pool'].value
        return data


@dataclass
class TradingStrategy:
    """
    Represents a trading strategy for backtesting and execution.
    
    Attributes:
        strategy_id: Unique strategy identifier
        name: Human-readable strategy name
        strategy_type: Type of strategy (momentum, mean_reversion, etc.)
        description: Strategy description
        parameters: Strategy-specific parameters
        symbols: List of trading symbols
        timeframe: Trading timeframe
        risk_parameters: Risk management parameters
        memory_context: Context for memory-based learning
        created_at: Strategy creation timestamp
    """
    name: str
    strategy_type: str
    symbols: List[str]
    timeframe: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    risk_parameters: Dict[str, Any] = field(default_factory=dict)
    memory_context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BacktestConfiguration:
    """
    Configuration for backtesting scenarios.
    
    Attributes:
        config_id: Unique configuration identifier
        strategy: Trading strategy to backtest
        start_date: Backtesting start date
        end_date: Backtesting end date
        initial_capital: Starting capital amount
        commission_rate: Transaction commission rate
        slippage_rate: Market slippage rate
        benchmark_symbol: Benchmark for performance comparison
        rl_enabled: Whether to enable RL-based optimization
        memory_enabled: Whether to use memory agent for context
    """
    config_id: str
    strategy: TradingStrategy
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float = 0.001
    slippage_rate: float = 0.001
    benchmark_symbol: str = "SPY"
    rl_enabled: bool = True
    memory_enabled: bool = True


class DAGPlanner:
    """
    Advanced LLM-Enhanced DAG planner for decomposing complex financial strategies into executable task graphs.
    
    This planner uses natural language processing and domain knowledge to create optimized
    execution plans that leverage all available agent pools efficiently. The LLM integration
    provides dynamic strategy decomposition and adaptive planning capabilities.
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.dag = nx.DiGraph()
        self.task_registry: Dict[str, TaskNode] = {}
        self.active_agents: Dict[str, Dict[str, Any]] = {}  # Track active agents
        
        # Initialize LLM for enhanced planning
        self.llm_enabled = False
        if llm_config:
            try:
                self.nlp = NaturalLanguageProcessor(llm_config)
                self.llm_enabled = True
                logger.info("✅ LLM-enhanced DAG planning enabled")
            except Exception as e:
                logger.warning(f"⚠️ LLM initialization failed, using traditional planning: {e}")
                self.llm_enabled = False
        else:
            logger.info("Using traditional DAG planning")
            
        # Memory-enhanced strategy patterns
        self.strategy_patterns = {
            "momentum": {
                "description": "Momentum-based strategies that follow trends",
                "required_data": ["price_data", "volume_data", "technical_indicators"],
                "alpha_signals": ["momentum_signals", "trend_following"],
                "risk_controls": ["position_sizing", "stop_loss", "volatility_check"]
            },
            "mean_reversion": {
                "description": "Mean reversion strategies that exploit price reversals",
                "required_data": ["price_data", "statistical_indicators", "volatility_data"],
                "alpha_signals": ["mean_reversion_signals", "pair_trading"],
                "risk_controls": ["position_sizing", "correlation_check", "liquidity_check"]
            },
            "enhanced_momentum": {
                "description": "AI-enhanced momentum with memory-based learning",
                "required_data": ["price_data", "volume_data", "technical_indicators", "market_sentiment"],
                "alpha_signals": ["llm_momentum_signals", "memory_enhanced_signals", "adaptive_signals"],
                "risk_controls": ["dynamic_position_sizing", "adaptive_stop_loss", "multi_factor_risk"]
            }
        }
        self.strategy_templates = self._load_strategy_templates()
        
        # Initialize LLM integration for dynamic planning
        if llm_config:
            self.nlp = NaturalLanguageProcessor(llm_config)
            self.llm_enabled = True
            logger.info("LLM-enhanced DAG planning enabled")
        else:
            self.nlp = None
            self.llm_enabled = False
            logger.info("Using traditional DAG planning")

    def _load_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined strategy templates for common trading scenarios"""
        return {
            "momentum_strategy": {
                "description": "Momentum-based trading strategy using technical indicators",
                "required_data": ["price_data", "volume_data", "technical_indicators"],
                "alpha_signals": ["momentum_signal", "volume_confirmation"],
                "execution_type": "market_orders",
                "risk_factors": ["market_volatility", "position_concentration"],
                "nlp_keywords": ["momentum", "trend", "breakout", "moving average"]
            },
            "mean_reversion": {
                "description": "Mean reversion strategy with statistical arbitrage",
                "required_data": ["price_data", "statistical_features"],
                "alpha_signals": ["mean_reversion_signal", "volatility_filter"],
                "execution_type": "limit_orders",
                "risk_factors": ["mean_reversion_failure", "volatility_expansion"],
                "nlp_keywords": ["mean reversion", "oversold", "overbought", "statistical arbitrage"]
            },
            "pairs_trading": {
                "description": "Statistical arbitrage between correlated pairs",
                "required_data": ["pair_price_data", "correlation_analysis"],
                "alpha_signals": ["spread_signal", "cointegration_test"],
                "execution_type": "simultaneous_orders",
                "risk_factors": ["correlation_breakdown", "execution_risk"],
                "nlp_keywords": ["pairs trading", "spread", "correlation", "cointegration"]
            },
            "machine_learning": {
                "description": "ML-based predictive trading strategy",
                "required_data": ["feature_matrix", "training_data", "market_regime"],
                "alpha_signals": ["ml_prediction", "confidence_score"],
                "execution_type": "adaptive_orders",
                "risk_factors": ["model_drift", "overfitting", "regime_change"],
                "nlp_keywords": ["machine learning", "neural network", "prediction", "algorithm"]
            }
        }

    async def plan_strategy_from_description(self, strategy_description: str, context: Dict[str, Any] = None) -> nx.DiGraph:
        """
        Create execution DAG from natural language strategy description using LLM.
        
        Args:
            strategy_description: Natural language description of the strategy
            context: Additional context for planning (market conditions, constraints, etc.)
            
        Returns:
            nx.DiGraph: Execution DAG with optimized task dependencies
        """
        if not self.llm_enabled:
            raise ValueError("LLM functionality not enabled. Please provide llm_config during initialization.")
        
        logger.info(f"🧠 Planning strategy from description: {strategy_description}")
        
        # Prepare LLM prompt for strategy decomposition
        prompt = self._create_strategy_planning_prompt(strategy_description, context)
        
        try:
            # Get LLM response for strategy decomposition
            llm_response = await self.nlp.process_natural_language(prompt)
            
            # Parse LLM response into structured plan
            strategy_plan = self._parse_llm_strategy_plan(llm_response)
            
            # Create execution DAG from parsed plan
            dag = await self._create_dag_from_plan(strategy_plan)
            
            logger.info(f"✅ LLM-enhanced strategy plan created with {len(dag.nodes)} tasks")
            return dag
            
        except Exception as e:
            logger.error(f"❌ LLM strategy planning failed: {e}")
            # Fallback to traditional planning
            return await self._create_fallback_dag(strategy_description, context)

    def _create_strategy_planning_prompt(self, description: str, context: Dict[str, Any] = None) -> str:
        """Create a detailed prompt for LLM strategy planning"""
        
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"
        
        return f"""
You are a financial strategy planning expert. Decompose the following trading strategy into a structured execution plan.

Strategy Description: {description}
{context_str}

Available Agent Pools:
1. DATA_AGENT_POOL: Fetches market data, technical indicators, fundamental data
2. ALPHA_AGENT_POOL: Generates trading signals, performs analysis, uses memory for learning
3. EXECUTION_AGENT_POOL: Executes trades, manages orders, portfolio management
4. MEMORY_AGENT: Stores and retrieves trading patterns, learns from history

Please provide a structured plan in JSON format with the following structure:
{{
    "strategy_type": "momentum|mean_reversion|ml_based|enhanced_momentum",
    "description": "Brief strategy description",
    "agent_workflow": [
        {{
            "step": 1,
            "agent_pool": "DATA_AGENT_POOL|ALPHA_AGENT_POOL|EXECUTION_AGENT_POOL|MEMORY_AGENT",
            "task_type": "fetch_data|generate_signals|execute_trades|retrieve_patterns",
            "tool_name": "specific_tool_name",
            "parameters": {{}},
            "dependencies": [],
            "memory_integration": "how this step uses or updates memory",
            "llm_enhancement": "how LLM improves this step"
        }}
    ],
    "risk_controls": ["list of risk management steps"],
    "memory_learning_strategy": "how the strategy learns and improves over time"
}}

Focus on:
1. Memory-driven alpha signal generation
2. LLM-enhanced decision making
3. Continuous learning from trading patterns
4. Risk management integration
5. Agent coordination and communication
"""

    def _parse_llm_strategy_plan(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response into structured strategy plan"""
        try:
            # Extract JSON from LLM response
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No valid JSON found in LLM response")
            
            json_str = llm_response[start_idx:end_idx]
            strategy_plan = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["strategy_type", "agent_workflow"]
            for field in required_fields:
                if field not in strategy_plan:
                    raise ValueError(f"Missing required field: {field}")
            
            return strategy_plan
            
        except Exception as e:
            logger.error(f"Failed to parse LLM strategy plan: {e}")
            # Return fallback plan
            return self._create_fallback_plan()

    async def _create_dag_from_plan(self, strategy_plan: Dict[str, Any]) -> nx.DiGraph:
        """Create execution DAG from parsed strategy plan"""
        
        dag = nx.DiGraph()
        task_nodes = []
        
        # Track active agents
        self.active_agents = {
            "data_agents": [],
            "alpha_agents": [],
            "execution_agents": [],
            "memory_agents": []
        }
        
        for step in strategy_plan.get("agent_workflow", []):
            # Create task node
            task_id = f"task_{step['step']}_{uuid.uuid4().hex[:8]}"
            
            task_node = TaskNode(
                task_id=task_id,
                task_type=step.get("task_type", "unknown"),
                agent_pool=AgentPoolType(step["agent_pool"].lower()),
                tool_name=step.get("tool_name", "default_tool"),
                parameters=step.get("parameters", {}),
                dependencies=step.get("dependencies", []),
                metadata={
                    "step_number": step["step"],
                    "memory_integration": step.get("memory_integration", ""),
                    "llm_enhancement": step.get("llm_enhancement", ""),
                    "strategy_type": strategy_plan.get("strategy_type", "unknown")
                }
            )
            
            # Track active agent
            agent_pool_name = step["agent_pool"].lower()
            if "data" in agent_pool_name:
                self.active_agents["data_agents"].append({
                    "task_id": task_id,
                    "tool_name": step.get("tool_name"),
                    "status": "planned"
                })
            elif "alpha" in agent_pool_name:
                self.active_agents["alpha_agents"].append({
                    "task_id": task_id,
                    "tool_name": step.get("tool_name"),
                    "memory_enhanced": True,
                    "llm_enabled": True,
                    "status": "planned"
                })
            elif "execution" in agent_pool_name:
                self.active_agents["execution_agents"].append({
                    "task_id": task_id,
                    "tool_name": step.get("tool_name"),
                    "status": "planned"
                })
            elif "memory" in agent_pool_name:
                self.active_agents["memory_agents"].append({
                    "task_id": task_id,
                    "tool_name": step.get("tool_name"),
                    "status": "planned"
                })
            
            task_nodes.append(task_node)
            self.task_registry[task_id] = task_node
            dag.add_node(task_id, task=task_node)
        
        # Add dependencies
        for task_node in task_nodes:
            for dep_step in task_node.dependencies:
                # Find dependency task
                dep_task = None
                for other_task in task_nodes:
                    if other_task.metadata.get("step_number") == dep_step:
                        dep_task = other_task
                        break
                
                if dep_task:
                    dag.add_edge(dep_task.task_id, task_node.task_id)
        
        return dag

    def get_active_agents_status(self) -> Dict[str, Any]:
        """Get status of all active agents"""
        return {
            "active_agents": self.active_agents,
            "total_data_agents": len(self.active_agents.get("data_agents", [])),
            "total_alpha_agents": len(self.active_agents.get("alpha_agents", [])),
            "total_execution_agents": len(self.active_agents.get("execution_agents", [])),
            "total_memory_agents": len(self.active_agents.get("memory_agents", [])),
            "llm_enabled_agents": len([a for a in self.active_agents.get("alpha_agents", []) if a.get("llm_enabled")]),
            "memory_enhanced_agents": len([a for a in self.active_agents.get("alpha_agents", []) if a.get("memory_enhanced")])
        }

    def update_agent_status(self, task_id: str, new_status: str):
        """Update the status of a specific agent"""
        for agent_type, agents in self.active_agents.items():
            for agent in agents:
                if agent.get("task_id") == task_id:
                    agent["status"] = new_status
                    agent["updated_at"] = datetime.now().isoformat()
                    break

    async def create_memory_enhanced_alpha_plan(self, strategy: TradingStrategy, memory_context: Dict[str, Any] = None) -> nx.DiGraph:
        """
        Create a specialized DAG for memory-enhanced alpha generation with RL-like learning.
        
        This method creates a plan that:
        1. Retrieves relevant patterns from memory
        2. Uses LLM for signal enhancement
        3. Learns from trading outcomes
        4. Adapts strategy parameters dynamically
        """
        
        logger.info(f"🧠 Creating memory-enhanced alpha plan for {strategy.name}")
        
        dag = nx.DiGraph()
        task_nodes = []
        
        # Initialize active agents tracking
        self.active_agents = {
            "data_agents": [],
            "alpha_agents": [],
            "memory_agents": [],
            "llm_agents": []
        }
        
        # Step 1: Memory Pattern Retrieval
        memory_task = TaskNode(
            task_id=f"memory_retrieval_{uuid.uuid4().hex[:8]}",
            task_type="memory_pattern_retrieval",
            agent_pool=AgentPoolType.MEMORY_AGENT,
            tool_name="retrieve_trading_patterns",
            parameters={
                "strategy_type": strategy.strategy_type,
                "symbols": strategy.symbols,
                "lookback_days": 90,
                "pattern_types": ["momentum", "mean_reversion", "volatility"],
                "minimum_confidence": 0.7
            },
            metadata={
                "purpose": "Retrieve historical patterns for alpha enhancement",
                "memory_integration": "Primary memory access for pattern learning"
            }
        )
        task_nodes.append(memory_task)
        self.active_agents["memory_agents"].append({
            "task_id": memory_task.task_id,
            "tool_name": "retrieve_trading_patterns",
            "status": "planned"
        })
        
        # Step 2: Enhanced Data Fetching with Memory Context
        data_task = TaskNode(
            task_id=f"enhanced_data_fetch_{uuid.uuid4().hex[:8]}",
            task_type="memory_contextualized_data_fetch",
            agent_pool=AgentPoolType.DATA_AGENT_POOL,
            tool_name="fetch_contextualized_market_data",
            parameters={
                "symbols": strategy.symbols,
                "timeframe": strategy.timeframe,
                "lookback_period": 60,
                "include_technical_indicators": True,
                "include_market_sentiment": True,
                "memory_guided_features": True
            },
            dependencies=[memory_task.task_id],
            metadata={
                "purpose": "Fetch market data enhanced by memory insights",
                "memory_integration": "Uses memory patterns to select relevant features"
            }
        )
        task_nodes.append(data_task)
        self.active_agents["data_agents"].append({
            "task_id": data_task.task_id,
            "tool_name": "fetch_contextualized_market_data",
            "status": "planned"
        })
        
        # Step 3: LLM-Enhanced Alpha Signal Generation
        alpha_task = TaskNode(
            task_id=f"llm_alpha_generation_{uuid.uuid4().hex[:8]}",
            task_type="llm_enhanced_alpha_generation",
            agent_pool=AgentPoolType.ALPHA_AGENT_POOL,
            tool_name="generate_llm_enhanced_signals",
            parameters={
                "strategy_type": strategy.strategy_type,
                "symbols": strategy.symbols,
                "use_memory_patterns": True,
                "llm_enhancement": True,
                "confidence_threshold": 0.6,
                "ensemble_methods": ["momentum_llm", "sentiment_llm", "pattern_llm"],
                "adaptive_parameters": strategy.parameters
            },
            dependencies=[memory_task.task_id, data_task.task_id],
            metadata={
                "purpose": "Generate alpha signals using LLM and memory",
                "llm_enhancement": "LLM processes market data and memory patterns for signal generation",
                "memory_integration": "Incorporates learned patterns into signal logic"
            }
        )
        task_nodes.append(alpha_task)
        self.active_agents["alpha_agents"].append({
            "task_id": alpha_task.task_id,
            "tool_name": "generate_llm_enhanced_signals",
            "llm_enabled": True,
            "memory_enhanced": True,
            "status": "planned"
        })
        
        # Step 4: Memory Learning and Feedback Loop
        learning_task = TaskNode(
            task_id=f"memory_learning_{uuid.uuid4().hex[:8]}",
            task_type="memory_learning_update",
            agent_pool=AgentPoolType.MEMORY_AGENT,
            tool_name="update_trading_patterns",
            parameters={
                "learning_mode": "reinforcement_style",
                "pattern_feedback": True,
                "strategy_adaptation": True,
                "performance_weighting": True
            },
            dependencies=[alpha_task.task_id],
            metadata={
                "purpose": "Learn from trading outcomes and update memory",
                "memory_integration": "Primary learning mechanism for strategy improvement",
                "rl_simulation": "Simulates RL-like learning without complex RL infrastructure"
            }
        )
        task_nodes.append(learning_task)
        self.active_agents["memory_agents"].append({
            "task_id": learning_task.task_id,
            "tool_name": "update_trading_patterns",
            "learning_enabled": True,
            "status": "planned"
        })
        
        # Add all tasks to DAG
        for task in task_nodes:
            self.task_registry[task.task_id] = task
            dag.add_node(task.task_id, task=task)
        
        # Add dependencies to DAG
        for task in task_nodes:
            for dep_id in task.dependencies:
                dag.add_edge(dep_id, task.task_id)
        
        logger.info(f"✅ Memory-enhanced alpha plan created with {len(task_nodes)} tasks")
        logger.info(f"🤖 Active agents: {len(self.active_agents['alpha_agents'])} alpha, {len(self.active_agents['memory_agents'])} memory")
        
        return dag

    def _create_fallback_plan(self) -> Dict[str, Any]:
        """Create fallback plan when LLM parsing fails"""
        return {
            "strategy_type": "momentum",
            "description": "Basic momentum strategy",
            "agent_workflow": [
                {
                    "step": 1,
                    "agent_pool": "DATA_AGENT_POOL",
                    "task_type": "fetch_data",
                    "tool_name": "get_market_data",
                    "parameters": {},
                    "dependencies": []
                },
                {
                    "step": 2,
                    "agent_pool": "ALPHA_AGENT_POOL", 
                    "task_type": "generate_signals",
                    "tool_name": "momentum_signals",
                    "parameters": {},
                    "dependencies": [1]
                }
            ]
        }

    async def _create_fallback_dag(self, description: str, context: Dict[str, Any] = None) -> nx.DiGraph:
        """Create fallback DAG when LLM planning fails"""
        fallback_plan = self._create_fallback_plan()
        return await self._create_dag_from_plan(fallback_plan)
