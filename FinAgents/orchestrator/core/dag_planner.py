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
        
        logger.info(f"Planning strategy from description: {strategy_description}")
        
        try:
            # Use LLM to parse strategy description
            planning_prompt = self._create_strategy_planning_prompt(strategy_description, context)
            
            response = await self.nlp.process_natural_language_request(
                planning_prompt,
                f"strategy_planning_{datetime.now().timestamp()}",
                {"strategy_templates": self.strategy_templates, "context": context}
            )
            
            if response["success"]:
                parsed_strategy = self._parse_llm_strategy_response(response["response"])
                
                # Convert to TradingStrategy object
                trading_strategy = self._create_trading_strategy_from_llm(parsed_strategy)
                
                # Plan execution using enhanced strategy
                return await self.plan_enhanced_strategy_execution(trading_strategy, parsed_strategy)
            else:
                raise ValueError(f"Failed to parse strategy description: {response.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error in LLM strategy planning: {e}")
            # Fallback to template-based planning
            return await self._fallback_strategy_planning(strategy_description)

    def _create_strategy_planning_prompt(self, description: str, context: Dict[str, Any]) -> str:
        """Create a comprehensive prompt for LLM strategy planning"""
        return f"""
Analyze the following trading strategy description and create a detailed execution plan:

STRATEGY DESCRIPTION: {description}

CONTEXT: {json.dumps(context, indent=2) if context else 'No additional context provided'}

AVAILABLE STRATEGY TEMPLATES:
{json.dumps(self.strategy_templates, indent=2)}

Please analyze the strategy and provide a detailed execution plan in the following JSON format:

{{
    "strategy_type": "identified_strategy_type",
    "confidence": 0.0-1.0,
    "strategy_name": "generated_strategy_name",
    "symbols": ["list", "of", "symbols"],
    "timeframe": "time_horizon",
    "parameters": {{
        "specific_strategy_parameters": "values"
    }},
    "required_data_sources": ["list", "of", "data", "requirements"],
    "alpha_generation_methods": ["list", "of", "signal", "methods"],
    "risk_management": {{
        "position_sizing": "method",
        "stop_loss": "percentage",
        "max_exposure": "percentage"
    }},
    "execution_requirements": {{
        "order_type": "market/limit/adaptive",
        "timing_constraints": "any_specific_timing",
        "slippage_tolerance": "percentage"
    }},
    "performance_metrics": ["list", "of", "metrics", "to", "track"],
    "dag_structure": {{
        "data_tasks": ["task1", "task2"],
        "alpha_tasks": ["task1", "task2"],
        "risk_tasks": ["task1", "task2"],
        "execution_tasks": ["task1", "task2"]
    }},
    "estimated_complexity": "low/medium/high",
    "estimated_duration": "execution_time_estimate",
    "dependencies": ["external", "dependencies"],
    "recommendations": ["optimization", "suggestions"]
}}

Focus on creating a practical, executable plan that leverages the available agent pools efficiently.
"""

    def _parse_llm_strategy_response(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate LLM strategy planning response"""
        try:
            # Extract the actual strategy plan from the LLM response
            strategy_plan = llm_response.get("parameters", {})
            
            # Validate required fields
            required_fields = ["strategy_type", "strategy_name", "symbols"]
            for field in required_fields:
                if field not in strategy_plan:
                    strategy_plan[field] = f"default_{field}"
            
            # Ensure symbols is a list
            if isinstance(strategy_plan.get("symbols"), str):
                strategy_plan["symbols"] = [strategy_plan["symbols"]]
            
            return strategy_plan
            
        except Exception as e:
            logger.error(f"Error parsing LLM strategy response: {e}")
            # Return a basic momentum strategy as fallback
            return {
                "strategy_type": "momentum_strategy",
                "strategy_name": "LLM_Generated_Momentum",
                "symbols": ["AAPL", "GOOGL"],
                "timeframe": "1D",
                "parameters": {"lookback_period": 20}
            }

    def _create_trading_strategy_from_llm(self, parsed_strategy: Dict[str, Any]) -> TradingStrategy:
        """Create TradingStrategy object from LLM-parsed strategy"""
        return TradingStrategy(
            name=parsed_strategy.get("strategy_name", "LLM_Strategy"),
            strategy_type=parsed_strategy.get("strategy_type", "momentum_strategy"),
            symbols=parsed_strategy.get("symbols", ["AAPL"]),
            timeframe=parsed_strategy.get("timeframe", "1D"),
            parameters=parsed_strategy.get("parameters", {}),
            created_at=datetime.now()
        )

    async def plan_enhanced_strategy_execution(self, strategy: TradingStrategy, llm_enhancement: Dict[str, Any]) -> nx.DiGraph:
        """
        Create execution DAG for a strategy with LLM enhancements.
        
        Args:
            strategy: Basic trading strategy
            llm_enhancement: Additional planning details from LLM
            
        Returns:
            nx.DiGraph: Enhanced execution DAG
        """
        logger.info(f"Planning enhanced execution for strategy: {strategy.name}")
        
        # Clear previous plan
        self.dag.clear()
        self.task_registry.clear()
        
        # Use LLM-enhanced planning if available
        if "dag_structure" in llm_enhancement:
            dag_structure = llm_enhancement["dag_structure"]
            
            # Create data tasks based on LLM recommendations
            data_tasks = await self._plan_enhanced_data_tasks(strategy, dag_structure.get("data_tasks", []))
            
            # Create alpha tasks with LLM-suggested methods
            alpha_tasks = await self._plan_enhanced_alpha_tasks(strategy, data_tasks, dag_structure.get("alpha_tasks", []))
            
            # Create risk management tasks
            risk_tasks = await self._plan_enhanced_risk_tasks(strategy, alpha_tasks, llm_enhancement.get("risk_management", {}))
            
            # Create execution tasks
            execution_tasks = await self._plan_enhanced_execution_tasks(strategy, alpha_tasks + risk_tasks, llm_enhancement.get("execution_requirements", {}))
            
        else:
            # Fallback to traditional planning
            data_tasks = await self._plan_data_tasks(strategy)
            alpha_tasks = await self._plan_alpha_tasks(strategy, data_tasks)
            memory_tasks = await self._plan_memory_tasks(strategy, data_tasks + alpha_tasks)
            execution_tasks = await self._plan_execution_tasks(strategy, alpha_tasks + memory_tasks)
        
        # Validate DAG
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("Generated DAG contains cycles - invalid execution plan")
        
        logger.info(f"Generated enhanced DAG with {len(self.task_registry)} tasks")
        return self.dag

    async def _plan_enhanced_data_tasks(self, strategy: TradingStrategy, recommended_tasks: List[str]) -> List[str]:
        """Plan data tasks with LLM recommendations"""
        tasks = []
        
        # Enhanced market data fetching
        for symbol in strategy.symbols:
            task_id = f"fetch_market_data_{symbol}"
            task = TaskNode(
                task_id=task_id,
                task_type="data_acquisition",
                agent_pool=AgentPoolType.DATA_AGENT_POOL,
                tool_name="fetch_market_data",
                parameters={
                    "symbol": symbol,
                    "timeframe": strategy.timeframe,
                    "lookback_period": strategy.parameters.get("lookback_period", "1Y"),
                    "enhanced_features": True  # LLM-suggested enhancement
                },
                metadata={"symbol": symbol, "priority": "high", "llm_enhanced": True}
            )
            self._add_task(task)
            tasks.append(task_id)

        # LLM-recommended additional data sources
        if "alternative_data" in recommended_tasks:
            for symbol in strategy.symbols:
                task_id = f"fetch_alternative_data_{symbol}"
                task = TaskNode(
                    task_id=task_id,
                    task_type="alternative_data",
                    agent_pool=AgentPoolType.DATA_AGENT_POOL,
                    tool_name="fetch_alternative_data",
                    parameters={"symbol": symbol, "data_types": ["news", "sentiment", "options_flow"]},
                    metadata={"symbol": symbol, "priority": "medium", "llm_recommended": True}
                )
                self._add_task(task)
                tasks.append(task_id)

        return tasks

    async def _plan_enhanced_alpha_tasks(self, strategy: TradingStrategy, data_task_deps: List[str], recommended_methods: List[str]) -> List[str]:
        """Plan alpha generation tasks with LLM enhancements"""
        tasks = []
        
        # Enhanced signal generation
        for method in recommended_methods:
            task_id = f"generate_{method}_signal"
            task = TaskNode(
                task_id=task_id,
                task_type="alpha_generation",
                agent_pool=AgentPoolType.ALPHA_AGENT_POOL,
                tool_name="generate_enhanced_signal",
                parameters={
                    "method": method,
                    "symbols": strategy.symbols,
                    "enhanced_features": True
                },
                dependencies=data_task_deps,
                metadata={"signal_type": method, "llm_enhanced": True}
            )
            self._add_task(task)
            tasks.append(task_id)
        
        # LLM-suggested ensemble method
        if len(recommended_methods) > 1:
            task_id = "ensemble_signal_combination"
            task = TaskNode(
                task_id=task_id,
                task_type="signal_ensemble",
                agent_pool=AgentPoolType.ALPHA_AGENT_POOL,
                tool_name="combine_signals",
                parameters={"method": "llm_weighted_ensemble"},
                dependencies=tasks,
                metadata={"ensemble_type": "llm_enhanced"}
            )
            self._add_task(task)
            tasks.append(task_id)
        
        return tasks

    async def _plan_enhanced_risk_tasks(self, strategy: TradingStrategy, alpha_deps: List[str], risk_management: Dict[str, Any]) -> List[str]:
        """Plan risk management tasks with LLM recommendations"""
        tasks = []
        
        # Enhanced risk assessment
        task_id = "enhanced_risk_assessment"
        task = TaskNode(
            task_id=task_id,
            task_type="risk_assessment",
            agent_pool=AgentPoolType.EXECUTION_AGENT_POOL,  # Assuming risk is handled by execution pool
            tool_name="assess_portfolio_risk",
            parameters={
                "symbols": strategy.symbols,
                "risk_model": "llm_enhanced",
                "position_sizing": risk_management.get("position_sizing", "equal_weight"),
                "max_exposure": risk_management.get("max_exposure", 0.1)
            },
            dependencies=alpha_deps,
            metadata={"risk_level": "enhanced", "llm_optimized": True}
        )
        self._add_task(task)
        tasks.append(task_id)
        
        return tasks

    async def _plan_enhanced_execution_tasks(self, strategy: TradingStrategy, signal_deps: List[str], execution_req: Dict[str, Any]) -> List[str]:
        """Plan execution tasks with LLM-optimized parameters"""
        tasks = []
        
        # Enhanced order execution
        task_id = "execute_enhanced_orders"
        task = TaskNode(
            task_id=task_id,
            task_type="order_execution",
            agent_pool=AgentPoolType.EXECUTION_AGENT_POOL,
            tool_name="execute_optimal_orders",
            parameters={
                "symbols": strategy.symbols,
                "order_type": execution_req.get("order_type", "adaptive"),
                "execution_algorithm": "llm_optimized",
                "slippage_tolerance": execution_req.get("slippage_tolerance", 0.01)
            },
            dependencies=signal_deps,
            metadata={"execution_style": "llm_enhanced"}
        )
        self._add_task(task)
        tasks.append(task_id)
        
        return tasks

    async def _fallback_strategy_planning(self, description: str) -> nx.DiGraph:
        """Fallback planning when LLM is not available"""
        logger.warning("Using fallback strategy planning due to LLM unavailability")
        
        # Simple keyword-based strategy detection
        description_lower = description.lower()
        
        if any(keyword in description_lower for keyword in ["momentum", "trend", "breakout"]):
            strategy_type = "momentum_strategy"
        elif any(keyword in description_lower for keyword in ["mean reversion", "oversold", "overbought"]):
            strategy_type = "mean_reversion"
        elif any(keyword in description_lower for keyword in ["pairs", "spread", "correlation"]):
            strategy_type = "pairs_trading"
        else:
            strategy_type = "momentum_strategy"  # Default
        
        # Create basic strategy
        strategy = TradingStrategy(
            name=f"Fallback_{strategy_type}",
            strategy_type=strategy_type,
            symbols=["AAPL", "GOOGL"],  # Default symbols
            timeframe="1D",
            parameters={"lookback_period": 20}
        )
        
        return await self.plan_strategy_execution(strategy)

    async def plan_strategy_execution(self, strategy: TradingStrategy) -> nx.DiGraph:
        """
        Create execution DAG for a trading strategy.
        
        Args:
            strategy: Trading strategy to plan
            
        Returns:
            nx.DiGraph: Execution DAG with optimized task dependencies
        """
        logger.info(f"Planning execution for strategy: {strategy.name}")
        
        # Clear previous plan
        self.dag.clear()
        self.task_registry.clear()
        
        # Create data acquisition tasks
        data_tasks = await self._plan_data_tasks(strategy)
        
        # Create alpha generation tasks
        alpha_tasks = await self._plan_alpha_tasks(strategy, data_tasks)
        
        # Create memory integration tasks
        memory_tasks = await self._plan_memory_tasks(strategy, data_tasks + alpha_tasks)
        
        # Create execution tasks
        execution_tasks = await self._plan_execution_tasks(strategy, alpha_tasks + memory_tasks)
        
        # Validate DAG
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("Generated DAG contains cycles - invalid execution plan")
        
        logger.info(f"Generated DAG with {len(self.task_registry)} tasks")
        return self.dag

    async def _plan_data_tasks(self, strategy: TradingStrategy) -> List[str]:
        """Plan data acquisition and preprocessing tasks"""
        tasks = []
        
        # Market data fetching
        for symbol in strategy.symbols:
            task_id = f"fetch_market_data_{symbol}"
            task = TaskNode(
                task_id=task_id,
                task_type="data_acquisition",
                agent_pool=AgentPoolType.DATA_AGENT_POOL,
                tool_name="fetch_market_data",
                parameters={
                    "symbol": symbol,
                    "timeframe": strategy.timeframe,
                    "lookback_period": strategy.parameters.get("lookback_period", "1Y")
                },
                metadata={"symbol": symbol, "priority": "high"}
            )
            self._add_task(task)
            tasks.append(task_id)

        # Technical indicators calculation
        if strategy.parameters.get("use_technical_indicators", True):
            for symbol in strategy.symbols:
                task_id = f"calculate_indicators_{symbol}"
                task = TaskNode(
                    task_id=task_id,
                    task_type="feature_engineering",
                    agent_pool=AgentPoolType.DATA_AGENT_POOL,
                    tool_name="calculate_technical_indicators",
                    parameters={
                        "symbol": symbol,
                        "indicators": strategy.parameters.get("indicators", ["SMA", "RSI", "MACD"])
                    },
                    dependencies=[f"fetch_market_data_{symbol}"],
                    metadata={"symbol": symbol, "priority": "medium"}
                )
                self._add_task(task)
                tasks.append(task_id)

        return tasks

    async def _plan_alpha_tasks(self, strategy: TradingStrategy, data_task_deps: List[str]) -> List[str]:
        """Plan alpha signal generation tasks"""
        tasks = []
        
        # Signal generation for each symbol
        for symbol in strategy.symbols:
            task_id = f"generate_alpha_signal_{symbol}"
            dependencies = [t for t in data_task_deps if symbol in t]
            
            task = TaskNode(
                task_id=task_id,
                task_type="signal_generation",
                agent_pool=AgentPoolType.ALPHA_AGENT_POOL,
                tool_name="generate_trading_signal",
                parameters={
                    "symbol": symbol,
                    "strategy_type": strategy.parameters.get("signal_type", "momentum"),
                    "lookback_window": strategy.parameters.get("signal_lookback", 20)
                },
                dependencies=dependencies,
                metadata={"symbol": symbol, "priority": "high"}
            )
            self._add_task(task)
            tasks.append(task_id)

        # Portfolio-level signal aggregation
        if len(strategy.symbols) > 1:
            task_id = "aggregate_portfolio_signals"
            task = TaskNode(
                task_id=task_id,
                task_type="signal_aggregation",
                agent_pool=AgentPoolType.ALPHA_AGENT_POOL,
                tool_name="aggregate_signals",
                parameters={
                    "symbols": strategy.symbols,
                    "aggregation_method": strategy.parameters.get("aggregation", "equal_weight")
                },
                dependencies=[f"generate_alpha_signal_{s}" for s in strategy.symbols],
                metadata={"priority": "high"}
            )
            self._add_task(task)
            tasks.append(task_id)

        return tasks

    async def _plan_memory_tasks(self, strategy: TradingStrategy, upstream_deps: List[str]) -> List[str]:
        """Plan memory integration and learning tasks"""
        tasks = []
        
        # Historical context retrieval
        task_id = "retrieve_historical_context"
        task = TaskNode(
            task_id=task_id,
            task_type="memory_retrieval",
            agent_pool=AgentPoolType.MEMORY_AGENT,
            tool_name="retrieve_similar_scenarios",
            parameters={
                "strategy_context": strategy.memory_context,
                "symbols": strategy.symbols,
                "lookback_days": 90
            },
            dependencies=[],  # Can run in parallel with data tasks
            metadata={"priority": "medium"}
        )
        self._add_task(task)
        tasks.append(task_id)

        # Memory-enhanced signal refinement
        task_id = "refine_signals_with_memory"
        task = TaskNode(
            task_id=task_id,
            task_type="memory_enhancement",
            agent_pool=AgentPoolType.MEMORY_AGENT,
            tool_name="enhance_signals_with_memory",
            parameters={
                "strategy_id": strategy.strategy_id,
                "symbols": strategy.symbols
            },
            dependencies=upstream_deps + ["retrieve_historical_context"],
            metadata={"priority": "high"}
        )
        self._add_task(task)
        tasks.append(task_id)

        return tasks

    async def _plan_execution_tasks(self, strategy: TradingStrategy, signal_deps: List[str]) -> List[str]:
        """Plan trade execution tasks"""
        tasks = []
        
        # Risk assessment
        task_id = "assess_portfolio_risk"
        task = TaskNode(
            task_id=task_id,
            task_type="risk_assessment",
            agent_pool=AgentPoolType.EXECUTION_AGENT_POOL,
            tool_name="calculate_portfolio_risk",
            parameters={
                "symbols": strategy.symbols,
                "risk_limits": strategy.risk_parameters
            },
            dependencies=signal_deps,
            metadata={"priority": "critical"}
        )
        self._add_task(task)
        tasks.append(task_id)

        # Order generation
        task_id = "generate_orders"
        task = TaskNode(
            task_id=task_id,
            task_type="order_generation",
            agent_pool=AgentPoolType.EXECUTION_AGENT_POOL,
            tool_name="create_trading_orders",
            parameters={
                "symbols": strategy.symbols,
                "execution_style": strategy.parameters.get("execution_style", "gradual")
            },
            dependencies=signal_deps + ["assess_portfolio_risk"],
            metadata={"priority": "critical"}
        )
        self._add_task(task)
        tasks.append(task_id)

        return tasks

    def _add_task(self, task: TaskNode) -> None:
        """Add task to DAG and registry"""
        self.task_registry[task.task_id] = task
        self.dag.add_node(task.task_id, **task.to_dict())
        
        # Add dependency edges
        for dep in task.dependencies:
            if dep in self.task_registry:
                self.dag.add_edge(dep, task.task_id)

    def get_execution_order(self) -> List[str]:
        """Get topologically sorted execution order"""
        return list(nx.topological_sort(self.dag))

    def get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready for execution (all dependencies completed)"""
        ready_tasks = []
        for task_id in self.task_registry:
            task = self.task_registry[task_id]
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                deps_completed = all(
                    self.task_registry[dep].status == TaskStatus.COMPLETED
                    for dep in task.dependencies
                    if dep in self.task_registry
                )
                if deps_completed:
                    ready_tasks.append(task_id)
        return ready_tasks

    def visualize_dag(self, output_path: Optional[str] = None) -> str:
        """Generate DAG visualization"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            plt.figure(figsize=(15, 10))
            
            # Create layout
            pos = nx.spring_layout(self.dag, k=3, iterations=50)
            
            # Color nodes by agent pool
            color_map = {
                AgentPoolType.DATA_AGENT_POOL: 'lightblue',
                AgentPoolType.ALPHA_AGENT_POOL: 'lightgreen',
                AgentPoolType.EXECUTION_AGENT_POOL: 'lightcoral',
                AgentPoolType.MEMORY_AGENT: 'lightyellow'
            }
            
            node_colors = []
            for task_id in self.dag.nodes():
                task = self.task_registry[task_id]
                node_colors.append(color_map.get(task.agent_pool, 'lightgray'))
            
            # Draw graph
            nx.draw(self.dag, pos, 
                   node_color=node_colors,
                   node_size=3000,
                   font_size=8,
                   font_weight='bold',
                   arrows=True,
                   arrowsize=20,
                   edge_color='gray',
                   alpha=0.8)
            
            # Add labels
            labels = {task_id: task_id.replace('_', '\\n') for task_id in self.dag.nodes()}
            nx.draw_networkx_labels(self.dag, pos, labels, font_size=6)
            
            # Add legend
            legend_elements = [
                mpatches.Patch(color='lightblue', label='Data Agent Pool'),
                mpatches.Patch(color='lightgreen', label='Alpha Agent Pool'),
                mpatches.Patch(color='lightcoral', label='Execution Agent Pool'),
                mpatches.Patch(color='lightyellow', label='Memory Agent')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.title("FinAgent Orchestration DAG", fontsize=16, fontweight='bold')
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"DAG visualization saved to {output_path}")
            else:
                output_path = f"dag_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            return output_path
            
        except ImportError:
            logger.warning("matplotlib not available, skipping DAG visualization")
            return ""


if __name__ == "__main__":
    # Example usage
    planner = DAGPlanner()
    
    # Create sample strategy
    strategy = TradingStrategy(
        strategy_id="momentum_001",
        name="Multi-Asset Momentum Strategy",
        description="Momentum strategy across multiple assets with memory enhancement",
        parameters={
            "signal_type": "momentum",
            "lookback_period": "6M",
            "indicators": ["SMA", "RSI", "MACD"],
            "signal_lookback": 20
        },
        symbols=["AAPL", "MSFT", "GOOGL"],
        timeframe="1d",
        risk_parameters={"max_position_size": 0.1, "stop_loss": 0.05}
    )
    
    # Generate execution plan
    dag = asyncio.run(planner.plan_strategy_execution(strategy))
    print(f"Generated DAG with {len(dag.nodes)} tasks")
    print("Execution order:", planner.get_execution_order())
