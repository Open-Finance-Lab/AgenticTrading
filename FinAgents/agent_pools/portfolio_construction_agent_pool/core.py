"""
Portfolio Construction Agent Pool Core Module

This module implements the core functionality for portfolio construction using
multi-agent inputs from alpha generation, risk management, and transaction cost
analysis. It provides sophisticated optimization algorithms and natural language
processing for investment portfolio management.

Author: Jifeng Li
Created: 2025-06-30
License: openMDW
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# LangGraph imports for ReAct agents
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated

from FinAgents.agent_pools.portfolio_construction_agent_pool.memory_bridge import (
    PortfolioConstructionMemoryBridge,
    PortfolioRecord,
    OptimizationResult,
    PortfolioMetrics,
    PortfolioPosition,
    OptimizationType,
    PortfolioStatus,
    create_portfolio_record,
    create_optimization_result,
    create_portfolio_metrics_record
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryUnit:
    """
    Local memory unit for Portfolio Construction Agent Pool internal operations.
    
    This unit maintains local state and coordinates with the external memory bridge
    for persistent storage and cross-agent communication.
    """
    
    def __init__(self, pool_id: str, storage_path: str):
        """
        Initialize the memory unit for portfolio construction operations.
        
        Args:
            pool_id: Unique identifier for the portfolio construction pool
            storage_path: Local storage path for temporary data
        """
        self.pool_id = pool_id
        self.storage_path = Path(storage_path)
        self.memory_data = {}
        self.events_log = []
        self.memory_bridge = None
        
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Internal state tracking
        self.active_portfolios = {}
        self.optimization_cache = {}
        self.market_data_cache = {}
        
        logger.info(f"Portfolio Construction Memory Unit initialized: {pool_id}")
    
    async def initialize(self):
        """Initialize the memory unit and establish connections."""
        self.file_path = self.storage_path / f"{self.pool_id}_memory.json"
        
        # Load existing data if available
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    saved_data = json.load(f)
                
                # Load different data sections
                self.memory_data = saved_data.get("memory_data", {})
                self.events_log = saved_data.get("events_log", [])
                self.active_portfolios = saved_data.get("active_portfolios", {})
                self.optimization_cache = saved_data.get("optimization_cache", {})
                
                logger.info(f"Loaded existing memory data: {len(self.memory_data)} entries, {len(self.events_log)} events")
            except Exception as e:
                logger.warning(f"Failed to load existing memory data: {e}")
                # Initialize with empty data if loading fails
                self.memory_data = {}
                self.events_log = []
                self.active_portfolios = {}
                self.optimization_cache = {}
        else:
            logger.info(f"No existing memory file found, starting with empty memory")
    
    async def close(self):
        """Close the memory unit and save state."""
        try:
            await self._save_to_file()
            logger.info("Portfolio Construction Memory Unit state saved successfully")
        except Exception as e:
            logger.error(f"Failed to save memory unit state: {e}")
    
    def set(self, key: str, value: Any):
        """Store data in memory unit."""
        self.memory_data[key] = value
        self._log_operation("SET", key, value)
    
    def get(self, key: str) -> Any:
        """Retrieve data from memory unit."""
        value = self.memory_data.get(key)
        self._log_operation("GET", key, value)
        return value
    
    def delete(self, key: str):
        """Delete data from memory unit."""
        if key in self.memory_data:
            del self.memory_data[key]
            self._log_operation("DELETE", key, None)
    
    def keys(self) -> List[str]:
        """Get all keys in memory unit."""
        return list(self.memory_data.keys())
    
    async def record_portfolio_event(self, event_data: Dict[str, Any]):
        """Record portfolio-related events."""
        event_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        event_data["pool_id"] = self.pool_id
        self.events_log.append(event_data)
        
        # Also store in memory_data for persistence
        event_key = f"event_{len(self.events_log)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.memory_data[event_key] = event_data
        
        # Save to file immediately for important events
        try:
            await self._save_to_file()
        except Exception as e:
            logger.warning(f"Failed to save event to file: {e}")
        
        # Forward to memory bridge if available
        if self.memory_bridge:
            try:
                await self.memory_bridge._log_system_event(
                    event_type="PORTFOLIO_EVENT",
                    log_level="INFO",
                    title=event_data.get("event_type", "Portfolio Event"),
                    content=json.dumps(event_data),
                    metadata={"pool_id": self.pool_id}
                )
            except Exception as e:
                logger.warning(f"Failed to forward event to memory bridge: {e}")
    
    def _log_operation(self, operation: str, key: str, value: Any):
        """Log memory operations for debugging."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "key": key,
            "value_type": type(value).__name__ if value is not None else "None"
        }
        self.events_log.append(log_entry)
        
        # Store operation log in memory_data
        op_key = f"operation_{len(self.events_log)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.memory_data[op_key] = log_entry
    
    async def _save_to_file(self):
        """Save current state to file."""
        if hasattr(self, 'file_path'):
            try:
                # Create a comprehensive data structure to save
                save_data = {
                    "pool_id": self.pool_id,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "memory_data": self.memory_data,
                    "events_log": self.events_log[-100:],  # Keep last 100 events
                    "active_portfolios": self.active_portfolios,
                    "optimization_cache": self.optimization_cache,
                    "statistics": {
                        "total_events": len(self.events_log),
                        "total_memory_entries": len(self.memory_data)
                    }
                }
                
                with open(self.file_path, 'w') as f:
                    json.dump(save_data, f, indent=2, default=str)
                logger.debug(f"Memory unit data saved: {len(save_data['memory_data'])} entries")
            except Exception as e:
                logger.error(f"Failed to save memory unit data: {e}")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory unit statistics."""
        return {
            "pool_id": self.pool_id,
            "total_events": len(self.events_log),
            "total_memory_entries": len(self.memory_data),
            "active_portfolios_count": len(self.active_portfolios),
            "cached_optimizations": len(self.optimization_cache),
            "storage_path": str(self.storage_path),
            "file_path": str(getattr(self, 'file_path', 'Not initialized'))
        }


class AgentRegistry:
    """
    Registry for managing portfolio construction agents and their capabilities.
    """
    
    def __init__(self):
        """Initialize the agent registry."""
        self.agents = {}
        self.agent_capabilities = {}
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Register default portfolio construction agents."""
        default_agents = {
            "mean_variance_optimizer": {
                "description": "Classical Markowitz mean-variance optimization",
                "capabilities": ["mean_variance", "efficient_frontier", "risk_budgeting"],
                "optimization_types": [OptimizationType.MEAN_VARIANCE]
            },
            "black_litterman_optimizer": {
                "description": "Black-Litterman model with Bayesian approach",
                "capabilities": ["black_litterman", "view_incorporation", "uncertainty_modeling"],
                "optimization_types": [OptimizationType.BLACK_LITTERMAN]
            },
            "risk_parity_optimizer": {
                "description": "Risk parity and equal risk contribution optimization",
                "capabilities": ["risk_parity", "equal_risk_contribution", "volatility_targeting"],
                "optimization_types": [OptimizationType.RISK_PARITY]
            },
            "factor_optimizer": {
                "description": "Factor-based portfolio construction and optimization",
                "capabilities": ["factor_modeling", "factor_exposure", "style_analysis"],
                "optimization_types": [OptimizationType.FACTOR_BASED]
            },
            "robust_optimizer": {
                "description": "Robust optimization under uncertainty",
                "capabilities": ["robust_optimization", "worst_case_scenarios", "ambiguity_aversion"],
                "optimization_types": [OptimizationType.ROBUST_OPTIMIZATION]
            },
            "rebalancing_agent": {
                "description": "Portfolio rebalancing and maintenance",
                "capabilities": ["rebalancing", "drift_monitoring", "transaction_optimization"],
                "optimization_types": ["maintenance"]
            },
            "constraint_manager": {
                "description": "Investment constraint management and validation",
                "capabilities": ["constraint_validation", "compliance_checking", "limit_monitoring"],
                "optimization_types": ["validation"]
            },
            "performance_analyzer": {
                "description": "Portfolio performance analysis and attribution",
                "capabilities": ["performance_analysis", "attribution", "benchmarking"],
                "optimization_types": ["analysis"]
            }
        }
        
        for agent_name, agent_info in default_agents.items():
            self.agents[agent_name] = agent_info
            self.agent_capabilities[agent_name] = agent_info["capabilities"]
        
        logger.info(f"Registered {len(self.agents)} portfolio construction agents")
    
    def register_agent(self, agent_name: str, agent_info: Dict[str, Any]):
        """Register a new agent."""
        self.agents[agent_name] = agent_info
        self.agent_capabilities[agent_name] = agent_info.get("capabilities", [])
        logger.info(f"Registered new agent: {agent_name}")
    
    def get_agent(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent information."""
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """List all registered agents."""
        return list(self.agents.keys())
    
    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents with specific capability."""
        matching_agents = []
        for agent_name, capabilities in self.agent_capabilities.items():
            if capability in capabilities:
                matching_agents.append(agent_name)
        return matching_agents
    
    def find_agents_by_optimization_type(self, optimization_type: OptimizationType) -> List[str]:
        """Find agents that support specific optimization type."""
        matching_agents = []
        for agent_name, agent_info in self.agents.items():
            if optimization_type in agent_info.get("optimization_types", []):
                matching_agents.append(agent_name)
        return matching_agents


class PortfolioConstructionMCPServer:
    """
    MCP (Model Context Protocol) Server for Portfolio Construction Agent Pool.
    
    This server provides structured interfaces for portfolio construction operations
    and integrates with external systems via standardized protocols.
    """
    
    def __init__(self, agent_pool: 'PortfolioConstructionAgentPool'):
        """
        Initialize the MCP server.
        
        Args:
            agent_pool: Reference to the portfolio construction agent pool
        """
        self.agent_pool = agent_pool
        self.server_id = f"portfolio_construction_mcp_{uuid.uuid4().hex[:8]}"
        self.active_sessions = {}
        
        logger.info(f"Portfolio Construction MCP Server initialized: {self.server_id}")
    
    async def handle_portfolio_construction_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle portfolio construction requests via MCP protocol.
        
        Args:
            request: Portfolio construction request data
            
        Returns:
            Dict[str, Any]: Portfolio construction response
        """
        try:
            request_type = request.get("type", "portfolio_optimization")
            
            if request_type == "portfolio_optimization":
                return await self._handle_optimization_request(request)
            elif request_type == "portfolio_analysis":
                return await self._handle_analysis_request(request)
            elif request_type == "rebalancing":
                return await self._handle_rebalancing_request(request)
            elif request_type == "performance_evaluation":
                return await self._handle_performance_request(request)
            else:
                return {
                    "status": "error",
                    "error": f"Unknown request type: {request_type}",
                    "supported_types": ["portfolio_optimization", "portfolio_analysis", 
                                      "rebalancing", "performance_evaluation"]
                }
        
        except Exception as e:
            logger.error(f"Failed to handle portfolio construction request: {e}")
            return {
                "status": "error",
                "error": str(e),
                "request_id": request.get("request_id")
            }
    
    async def _handle_optimization_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle portfolio optimization requests."""
        optimization_params = request.get("parameters", {})
        
        # Extract optimization parameters
        investment_universe = optimization_params.get("investment_universe", [])
        optimization_type = optimization_params.get("optimization_type", "mean_variance")
        constraints = optimization_params.get("constraints", {})
        objective = optimization_params.get("objective", "maximize_sharpe")
        
        # Perform optimization
        result = await self.agent_pool.execute_structured_task({
            "task_type": "portfolio_optimization",
            "agent_type": f"{optimization_type}_optimizer",
            "parameters": {
                "investment_universe": investment_universe,
                "optimization_type": optimization_type,
                "constraints": constraints,
                "objective": objective
            }
        })
        
        return {
            "status": "success",
            "request_id": request.get("request_id"),
            "optimization_result": result,
            "server_id": self.server_id
        }
    
    async def _handle_analysis_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle portfolio analysis requests."""
        analysis_params = request.get("parameters", {})
        portfolio_id = analysis_params.get("portfolio_id")
        
        if not portfolio_id:
            return {
                "status": "error",
                "error": "Portfolio ID required for analysis",
                "request_id": request.get("request_id")
            }
        
        # Perform analysis
        result = await self.agent_pool.execute_structured_task({
            "task_type": "portfolio_analysis",
            "agent_type": "performance_analyzer",
            "parameters": {
                "portfolio_id": portfolio_id,
                "analysis_type": analysis_params.get("analysis_type", "comprehensive")
            }
        })
        
        return {
            "status": "success",
            "request_id": request.get("request_id"),
            "analysis_result": result,
            "server_id": self.server_id
        }
    
    async def _handle_rebalancing_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle portfolio rebalancing requests."""
        rebalancing_params = request.get("parameters", {})
        portfolio_id = rebalancing_params.get("portfolio_id")
        
        # Perform rebalancing
        result = await self.agent_pool.execute_structured_task({
            "task_type": "portfolio_rebalancing",
            "agent_type": "rebalancing_agent",
            "parameters": {
                "portfolio_id": portfolio_id,
                "rebalancing_threshold": rebalancing_params.get("threshold", 0.05),
                "transaction_cost_model": rebalancing_params.get("cost_model", "linear")
            }
        })
        
        return {
            "status": "success",
            "request_id": request.get("request_id"),
            "rebalancing_result": result,
            "server_id": self.server_id
        }
    
    async def _handle_performance_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle portfolio performance evaluation requests."""
        performance_params = request.get("parameters", {})
        portfolio_id = performance_params.get("portfolio_id")
        
        # Get performance analytics
        analytics = await self.agent_pool.memory_bridge.get_portfolio_performance_analytics(
            portfolio_id=portfolio_id,
            time_range=timedelta(days=performance_params.get("days", 30))
        )
        
        return {
            "status": "success",
            "request_id": request.get("request_id"),
            "performance_analytics": analytics,
            "server_id": self.server_id
        }


class PortfolioConstructionAgentPool:
    """
    Portfolio Construction Agent Pool - Multi-Agent Portfolio Optimization System.
    
    This pool integrates alpha signals, risk metrics, and transaction costs from
    various agent pools to construct optimal investment portfolios using advanced
    optimization techniques and machine learning approaches.
    """
    
    def __init__(self, 
                 openai_api_key: str,
                 external_memory_config: Optional[Dict[str, Any]] = None,
                 enable_real_time_monitoring: bool = True,
                 pool_id: str = None):
        """
        Initialize the Portfolio Construction Agent Pool.
        
        Args:
            openai_api_key: OpenAI API key for natural language processing
            external_memory_config: Configuration for external memory integration
            enable_real_time_monitoring: Enable real-time portfolio monitoring
            pool_id: Unique identifier for this pool instance
        """
        self.pool_id = pool_id or f"portfolio_construction_pool_{uuid.uuid4().hex[:8]}"
        self.openai_api_key = openai_api_key
        self.external_memory_config = external_memory_config or {}
        self.enable_real_time_monitoring = enable_real_time_monitoring
        
        # Initialize core components
        self.agent_registry = AgentRegistry()
        self.memory_bridge = PortfolioConstructionMemoryBridge(
            external_memory_config=external_memory_config,
            enable_real_time_monitoring=enable_real_time_monitoring
        )
        
        # Set pool_id for memory bridge
        self.memory_bridge.pool_id = self.pool_id
        
        # Initialize OpenAI client
        self.openai_client = None
        self._initialize_openai_client()
        
        # Initialize LangGraph agents (after OpenAI client)
        self.langgraph_agents = {}
        self._initialize_langgraph_agents()
        
        # Initialize memory unit with correct storage path
        memory_storage_path = Path("./FinAgents/memory/portfolio_construction_memory_storage")
        self.memory_unit = MemoryUnit(
            pool_id=self.pool_id,
            storage_path=str(memory_storage_path)
        )
        
        # Connect memory unit to memory bridge
        self.memory_unit.memory_bridge = self.memory_bridge
        
        # Initialize MCP server
        self.mcp_server = PortfolioConstructionMCPServer(self)
        
        # Pool statistics
        self.pool_statistics = {
            'portfolios_created': 0,
            'optimizations_performed': 0,
            'total_value_managed': 0.0,
            'active_portfolios': 0,
            'last_activity': datetime.now(timezone.utc)
        }
        
        logger.info(f"Portfolio Construction Agent Pool initialized: {self.pool_id}")
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client for natural language processing."""
        try:
            import openai
            self.openai_client = openai.AsyncOpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI client initialized for Portfolio Construction Agent Pool")
        except ImportError:
            logger.warning("OpenAI package not available. Natural language processing will be limited.")
            self.openai_client = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
    
    def _initialize_langgraph_agents(self):
        """Initialize LangGraph-based agents for portfolio construction."""
        try:
            agent_types = [
                "mean_variance_optimizer",
                "black_litterman_optimizer", 
                "risk_parity_optimizer",
                "factor_optimizer",
                "robust_optimizer"
            ]
            
            for agent_type in agent_types:
                agent = PortfolioConstructionAgent(self, agent_type)
                self.langgraph_agents[agent_type] = agent
                logger.info(f"Initialized LangGraph agent: {agent_type}")
            
            logger.info(f"Initialized {len(self.langgraph_agents)} LangGraph portfolio agents")
            
        except Exception as e:
            logger.warning(f"Failed to initialize LangGraph agents: {e}")
            self.langgraph_agents = {}
    
    async def initialize(self):
        """Initialize the portfolio construction agent pool."""
        try:
            # Initialize memory components
            await self.memory_bridge.initialize()
            await self.memory_unit.initialize()
            
            # Log initialization
            await self.memory_unit.record_portfolio_event({
                "event_type": "POOL_INITIALIZED",
                "pool_id": self.pool_id,
                "agents_count": len(self.agent_registry.list_agents()),
                "features": {
                    "real_time_monitoring": self.enable_real_time_monitoring,
                    "external_memory": self.memory_bridge.external_memory_agent is not None,
                    "natural_language": self.openai_client is not None
                }
            })
            
            logger.info(f"Portfolio Construction Agent Pool fully initialized: {self.pool_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Portfolio Construction Agent Pool: {e}")
            raise
    
    async def close(self):
        """Close the portfolio construction agent pool and clean up resources."""
        try:
            await self.memory_bridge.close()
            await self.memory_unit.close()
            logger.info(f"Portfolio Construction Agent Pool closed: {self.pool_id}")
        except Exception as e:
            logger.error(f"Error closing Portfolio Construction Agent Pool: {e}")
    
    async def process_natural_language_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process natural language input for portfolio construction requests.
        
        Args:
            user_input: Natural language description of portfolio requirements
            
        Returns:
            Dict[str, Any]: Processed request with structured parameters
        """
        try:
            if not self.openai_client:
                return {
                    "status": "error",
                    "error": "Natural language processing not available"
                }
            
            # Create prompt for portfolio construction interpretation
            system_prompt = """
            You are a portfolio construction assistant. Convert natural language requests 
            into structured portfolio construction parameters. Return JSON with the following structure:
            {
                "task_type": "portfolio_optimization",
                "optimization_type": "mean_variance|black_litterman|risk_parity|factor_based",
                "investment_universe": ["AAPL", "GOOGL", ...],
                "objective": "maximize_sharpe|minimize_risk|maximize_return",
                "constraints": {
                    "max_weight": 0.1,
                    "min_weight": 0.0,
                    "sector_limits": {},
                    "risk_budget": 0.15
                },
                "benchmark": "SPY",
                "time_horizon": "daily|weekly|monthly",
                "rebalancing_frequency": "monthly"
            }
            """
            
            user_prompt = f"Portfolio construction request: {user_input}"
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            # Parse the response
            content = response.choices[0].message.content
            try:
                structured_request = json.loads(content)
                
                # Add metadata
                structured_request["original_input"] = user_input
                structured_request["processed_at"] = datetime.now(timezone.utc).isoformat()
                structured_request["pool_id"] = self.pool_id
                
                return {
                    "status": "success",
                    "structured_request": structured_request
                }
                
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "error": "Failed to parse structured request from natural language",
                    "raw_response": content
                }
        
        except Exception as e:
            logger.error(f"Failed to process natural language input: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def retrieve_multi_agent_signals(self, 
                                         investment_universe: List[str],
                                         time_horizon: str = "daily") -> Dict[str, Any]:
        """
        Retrieve and integrate signals from alpha, risk, and transaction cost agent pools.
        
        Args:
            investment_universe: List of assets to analyze
            time_horizon: Time horizon for analysis
            
        Returns:
            Dict[str, Any]: Integrated multi-agent signals
        """
        try:
            # Use memory bridge to retrieve multi-agent inputs
            multi_agent_inputs = await self.memory_bridge.retrieve_multi_agent_inputs(
                investment_universe=investment_universe,
                time_horizon=time_horizon
            )
            
            # Process and structure the inputs for portfolio construction
            structured_inputs = {
                "investment_universe": investment_universe,
                "time_horizon": time_horizon,
                "alpha_signals": self._process_alpha_signals(multi_agent_inputs.get("alpha_signals", [])),
                "risk_metrics": self._process_risk_metrics(multi_agent_inputs.get("risk_metrics", [])),
                "transaction_costs": self._process_transaction_costs(multi_agent_inputs.get("transaction_costs", [])),
                "data_quality": {
                    "alpha_signals_available": len(multi_agent_inputs.get("alpha_signals", [])),
                    "risk_analyses_available": len(multi_agent_inputs.get("risk_metrics", [])),
                    "cost_analyses_available": len(multi_agent_inputs.get("transaction_costs", [])),
                    "coverage_ratio": self._calculate_coverage_ratio(investment_universe, multi_agent_inputs)
                }
            }
            
            logger.info(f"Retrieved multi-agent signals for {len(investment_universe)} assets")
            return structured_inputs
            
        except Exception as e:
            logger.error(f"Failed to retrieve multi-agent signals: {e}")
            return {
                "investment_universe": investment_universe,
                "time_horizon": time_horizon,
                "alpha_signals": {},
                "risk_metrics": {},
                "transaction_costs": {},
                "error": str(e)
            }
    
    async def execute_structured_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute structured portfolio construction tasks.
        
        Args:
            task: Structured task definition
            
        Returns:
            Dict[str, Any]: Task execution result
        """
        try:
            task_type = task.get("task_type")
            agent_type = task.get("agent_type")
            parameters = task.get("parameters", {})
            use_langgraph = parameters.get("use_langgraph", True)  # Default to using LangGraph
            
            # Log task execution
            await self.memory_unit.record_portfolio_event({
                "event_type": "TASK_STARTED",
                "task_type": task_type,
                "agent_type": agent_type,
                "use_langgraph": use_langgraph,
                "parameters": parameters
            })
            
            # Use LangGraph agents if available and requested
            if use_langgraph and task_type == "portfolio_optimization" and agent_type in self.langgraph_agents:
                result = await self.execute_langgraph_task(task)
            else:
                # Fall back to traditional implementation
                if task_type == "portfolio_optimization":
                    result = await self._execute_optimization_task(agent_type, parameters)
                elif task_type == "portfolio_analysis":
                    result = await self._execute_analysis_task(agent_type, parameters)
                elif task_type == "portfolio_rebalancing":
                    result = await self._execute_rebalancing_task(agent_type, parameters)
                elif task_type == "performance_evaluation":
                    result = await self._execute_performance_task(agent_type, parameters)
                else:
                    result = {
                        "status": "error",
                        "error": f"Unknown task type: {task_type}"
                    }
            
            # Log task completion
            await self.memory_unit.record_portfolio_event({
                "event_type": "TASK_COMPLETED",
                "task_type": task_type,
                "agent_type": agent_type,
                "result_status": result.get("status"),
                "execution_time": result.get("execution_time"),
                "use_langgraph": use_langgraph
            })
            
            # Update statistics
            self.pool_statistics['last_activity'] = datetime.now(timezone.utc)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute structured task: {e}")
            return {
                "status": "error",
                "error": str(e),
                "task": task
            }
    
    async def _execute_optimization_task(self, agent_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute portfolio optimization task."""
        start_time = datetime.now()
        
        try:
            # Get multi-agent inputs
            investment_universe = parameters.get("investment_universe", [])
            optimization_type = parameters.get("optimization_type", "mean_variance")
            
            if not investment_universe:
                return {
                    "status": "error",
                    "error": "Investment universe required for optimization"
                }
            
            # Retrieve multi-agent signals
            signals = await self.retrieve_multi_agent_signals(investment_universe)
            
            # Perform optimization based on type
            if optimization_type == "mean_variance":
                optimization_result = await self._mean_variance_optimization(signals, parameters)
            elif optimization_type == "black_litterman":
                optimization_result = await self._black_litterman_optimization(signals, parameters)
            elif optimization_type == "risk_parity":
                optimization_result = await self._risk_parity_optimization(signals, parameters)
            elif optimization_type == "factor_based":
                optimization_result = await self._factor_based_optimization(signals, parameters)
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported optimization type: {optimization_type}"
                }
            
            # Store optimization result
            if optimization_result.get("status") == "success":
                await self.memory_bridge.store_optimization_result(optimization_result["result"])
                self.pool_statistics['optimizations_performed'] += 1
            
            execution_time = (datetime.now() - start_time).total_seconds()
            optimization_result["execution_time"] = execution_time
            
            return optimization_result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _execute_analysis_task(self, agent_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute portfolio analysis task."""
        try:
            portfolio_id = parameters.get("portfolio_id")
            
            if not portfolio_id:
                return {
                    "status": "error",
                    "error": "Portfolio ID required for analysis"
                }
            
            # Get portfolio performance analytics
            analytics = await self.memory_bridge.get_portfolio_performance_analytics(
                portfolio_id=portfolio_id
            )
            
            return {
                "status": "success",
                "portfolio_id": portfolio_id,
                "analytics": analytics,
                "analysis_agent": agent_type
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _execute_rebalancing_task(self, agent_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute portfolio rebalancing task."""
        try:
            portfolio_id = parameters.get("portfolio_id")
            threshold = parameters.get("rebalancing_threshold", 0.05)
            
            # Implement rebalancing logic
            rebalancing_result = {
                "portfolio_id": portfolio_id,
                "rebalancing_threshold": threshold,
                "trades_required": [],
                "estimated_costs": 0.0,
                "rebalancing_needed": False
            }
            
            return {
                "status": "success",
                "result": rebalancing_result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _execute_performance_task(self, agent_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute portfolio performance evaluation task."""
        try:
            portfolio_id = parameters.get("portfolio_id")
            
            # Get comprehensive performance metrics
            analytics = await self.memory_bridge.get_portfolio_performance_analytics(
                portfolio_id=portfolio_id,
                time_range=timedelta(days=parameters.get("evaluation_days", 30))
            )
            
            return {
                "status": "success",
                "portfolio_id": portfolio_id,
                "performance_analytics": analytics
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    # Optimization implementations
    async def _mean_variance_optimization(self, signals: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Implement mean-variance optimization."""
        try:
            # Extract expected returns and covariance matrix from signals
            investment_universe = signals["investment_universe"]
            alpha_signals = signals["alpha_signals"]
            risk_metrics = signals["risk_metrics"]
            
            # Build expected returns vector
            expected_returns = {}
            for asset in investment_universe:
                asset_alpha = alpha_signals.get(asset, {})
                expected_returns[asset] = asset_alpha.get("expected_return", 0.08)  # Default 8%
            
            # Build covariance matrix from risk metrics
            # This is simplified - in practice, would use more sophisticated risk modeling
            variances = {}
            for asset in investment_universe:
                asset_risk = risk_metrics.get(asset, {})
                variances[asset] = (asset_risk.get("volatility", 0.20)) ** 2  # Default 20% vol
            
            # Simple equal-weight optimization (placeholder for actual optimization)
            n_assets = len(investment_universe)
            optimal_weights = {asset: 1.0 / n_assets for asset in investment_universe}
            
            # Calculate portfolio metrics
            portfolio_return = sum(expected_returns[asset] * weight 
                                 for asset, weight in optimal_weights.items())
            portfolio_variance = sum(variances[asset] * (weight ** 2) 
                                   for asset, weight in optimal_weights.items())
            portfolio_volatility = portfolio_variance ** 0.5
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility  # Assuming 2% risk-free rate
            
            # Create optimization result
            optimization_result = create_optimization_result(
                optimization_type=OptimizationType.MEAN_VARIANCE,
                portfolio_id=str(uuid.uuid4()),
                optimal_weights=optimal_weights,
                input_signals=signals,
                expected_metrics={
                    "expected_return": portfolio_return,
                    "expected_volatility": portfolio_volatility,
                    "sharpe_ratio": sharpe_ratio
                },
                optimization_status="success",
                objective_value=sharpe_ratio,
                optimization_time_seconds=0.1
            )
            
            return {
                "status": "success",
                "optimization_type": "mean_variance",
                "result": optimization_result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _black_litterman_optimization(self, signals: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Implement Black-Litterman optimization."""
        # Placeholder implementation
        return {
            "status": "success",
            "optimization_type": "black_litterman",
            "result": create_optimization_result(
                optimization_type=OptimizationType.BLACK_LITTERMAN,
                portfolio_id=str(uuid.uuid4()),
                optimal_weights={asset: 1.0/len(signals["investment_universe"]) 
                               for asset in signals["investment_universe"]},
                input_signals=signals
            )
        }
    
    async def _risk_parity_optimization(self, signals: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Implement risk parity optimization."""
        # Placeholder implementation
        return {
            "status": "success",
            "optimization_type": "risk_parity",
            "result": create_optimization_result(
                optimization_type=OptimizationType.RISK_PARITY,
                portfolio_id=str(uuid.uuid4()),
                optimal_weights={asset: 1.0/len(signals["investment_universe"]) 
                               for asset in signals["investment_universe"]},
                input_signals=signals
            )
        }
    
    async def _factor_based_optimization(self, signals: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Implement factor-based optimization."""
        # Placeholder implementation
        return {
            "status": "success",
            "optimization_type": "factor_based",
            "result": create_optimization_result(
                optimization_type=OptimizationType.FACTOR_BASED,
                portfolio_id=str(uuid.uuid4()),
                optimal_weights={asset: 1.0/len(signals["investment_universe"]) 
                               for asset in signals["investment_universe"]},
                input_signals=signals
            )
        }
    
    # Helper methods
    def _process_alpha_signals(self, alpha_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process alpha signals for portfolio construction."""
        processed = {}
        for signal in alpha_signals:
            asset = signal.get("symbol") or signal.get("asset_id")
            if asset:
                processed[asset] = {
                    "expected_return": signal.get("predicted_return", 0.08),
                    "confidence": signal.get("confidence_score", 0.5),
                    "signal_strength": signal.get("strength", 0.0),
                    "signal_type": signal.get("signal_type", "neutral")
                }
        return processed
    
    def _process_risk_metrics(self, risk_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process risk metrics for portfolio construction."""
        processed = {}
        for metric in risk_metrics:
            asset = metric.get("asset_id") or metric.get("symbol")
            if asset:
                processed[asset] = {
                    "volatility": metric.get("volatility", 0.20),
                    "var_95": metric.get("var_95", 0.05),
                    "beta": metric.get("beta", 1.0),
                    "correlation": metric.get("correlation", {})
                }
        return processed
    
    def _process_transaction_costs(self, transaction_costs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process transaction costs for portfolio construction."""
        processed = {}
        for cost in transaction_costs:
            asset = cost.get("asset_id") or cost.get("symbol")
            if asset:
                processed[asset] = {
                    "bid_ask_spread": cost.get("bid_ask_spread", 0.001),
                    "market_impact": cost.get("market_impact", 0.001),
                    "commission": cost.get("commission", 0.0005),
                    "total_cost": cost.get("total_cost", 0.0025)
                }
        return processed
    
    def _calculate_coverage_ratio(self, investment_universe: List[str], multi_agent_inputs: Dict[str, Any]) -> float:
        """Calculate data coverage ratio for investment universe."""
        if not investment_universe:
            return 0.0
        
        covered_assets = set()
        
        # Check alpha signals coverage
        for signal in multi_agent_inputs.get("alpha_signals", []):
            asset = signal.get("symbol") or signal.get("asset_id")
            if asset in investment_universe:
                covered_assets.add(asset)
        
        # Check risk metrics coverage
        for metric in multi_agent_inputs.get("risk_metrics", []):
            asset = metric.get("asset_id") or metric.get("symbol")
            if asset in investment_universe:
                covered_assets.add(asset)
        
        return len(covered_assets) / len(investment_universe)
    
    async def execute_langgraph_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using LangGraph agents."""
        try:
            agent_type = task.get("agent_type", "mean_variance_optimizer")
            parameters = task.get("parameters", {})
            
            # Get the appropriate LangGraph agent
            agent = self.langgraph_agents.get(agent_type)
            if not agent:
                return {
                    "status": "error",
                    "error": f"LangGraph agent not available: {agent_type}",
                    "available_agents": list(self.langgraph_agents.keys())
                }
            
            # Prepare request for the agent
            request = {
                "user_input": parameters.get("user_input", f"Construct portfolio using {agent_type}"),
                "investment_universe": parameters.get("investment_universe", []),
                "optimization_params": {
                    "optimization_type": parameters.get("optimization_type", agent_type.replace("_optimizer", "")),
                    "constraints": parameters.get("constraints", {}),
                    "objective": parameters.get("objective", "maximize_sharpe")
                }
            }
            
            # Execute portfolio construction
            result = await agent.construct_portfolio(request)
            
            # Log the execution
            await self.memory_unit.record_portfolio_event({
                "event_type": "LANGGRAPH_TASK_EXECUTED",
                "agent_type": agent_type,
                "status": result.get("status"),
                "thread_id": result.get("thread_id")
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute LangGraph task: {e}")
            return {
                "status": "error",
                "error": str(e),
                "task": task
            }
    
    async def process_natural_language_with_langgraph(self, user_input: str, 
                                                    investment_universe: List[str] = None) -> Dict[str, Any]:
        """Process natural language input using LangGraph agents."""
        try:
            # Default investment universe if not provided
            if not investment_universe:
                investment_universe = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
            
            # Determine the best agent type based on user input
            user_input_lower = user_input.lower()
            
            if "risk parity" in user_input_lower or "equal risk" in user_input_lower:
                agent_type = "risk_parity_optimizer"
            elif "black litterman" in user_input_lower or "bayesian" in user_input_lower:
                agent_type = "black_litterman_optimizer"
            elif "factor" in user_input_lower or "momentum" in user_input_lower or "value" in user_input_lower:
                agent_type = "factor_optimizer"
            elif "robust" in user_input_lower or "uncertainty" in user_input_lower:
                agent_type = "robust_optimizer"
            else:
                agent_type = "mean_variance_optimizer"  # Default
            
            # Create task for LangGraph agent
            task = {
                "task_type": "portfolio_optimization",
                "agent_type": agent_type,
                "parameters": {
                    "user_input": user_input,
                    "investment_universe": investment_universe,
                    "optimization_type": agent_type.replace("_optimizer", ""),
                    "constraints": {
                        "max_weight": 0.40,
                        "min_weight": 0.02
                    }
                }
            }
            
            # Execute using LangGraph
            result = await self.execute_langgraph_task(task)
            
            # Add natural language context
            result["original_input"] = user_input
            result["selected_agent"] = agent_type
            result["investment_universe"] = investment_universe
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process natural language with LangGraph: {e}")
            return {
                "status": "error",
                "error": str(e),
                "original_input": user_input
            }
    
    async def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory status for debugging."""
        try:
            memory_stats = self.memory_unit.get_memory_statistics()
            
            # Add pool-level statistics
            pool_stats = {
                "pool_id": self.pool_id,
                "pool_statistics": self.pool_statistics,
                "langgraph_agents": list(self.langgraph_agents.keys()),
                "openai_available": self.openai_client is not None,
                "memory_bridge_session": getattr(self.memory_bridge, 'session_id', 'Not initialized'),
                "memory_unit_stats": memory_stats
            }
            
            return pool_stats
            
        except Exception as e:
            logger.error(f"Failed to get memory status: {e}")
            return {
                "error": str(e),
                "pool_id": self.pool_id
            }
    
    async def save_memory_immediately(self):
        """Force save memory unit data immediately."""
        try:
            await self.memory_unit._save_to_file()
            logger.info("Memory unit data saved successfully")
            return {"status": "success", "message": "Memory saved"}
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            return {"status": "error", "error": str(e)}


# LangGraph State and Tools for Portfolio Construction Agents

class PortfolioAgentState(TypedDict):
    """State for portfolio construction agents using LangGraph."""
    messages: Annotated[List[BaseMessage], add_messages]
    portfolio_data: Dict[str, Any]
    investment_universe: List[str]
    optimization_params: Dict[str, Any]
    alpha_signals: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    transaction_costs: Dict[str, Any]
    optimization_result: Optional[Dict[str, Any]]
    current_step: str
    error_message: Optional[str]
    iteration_count: int


class PortfolioTools:
    """Tools for portfolio construction agents to interact with market data and optimization."""
    
    def __init__(self, agent_pool: 'PortfolioConstructionAgentPool'):
        self.agent_pool = agent_pool
    
    def create_market_data_tool(self) -> Tool:
        """Create tool for retrieving market data."""
        
        def get_market_data(symbols: str) -> str:
            """Retrieve market data for given symbols.
            
            Args:
                symbols: Comma-separated list of stock symbols (e.g., "AAPL,GOOGL,MSFT")
            
            Returns:
                JSON string with market data including prices, volatility, returns
            """
            try:
                symbol_list = [s.strip().upper() for s in symbols.split(',')]
                
                # Mock market data - in production, this would call real data APIs
                market_data = {}
                for symbol in symbol_list:
                    market_data[symbol] = {
                        "current_price": np.random.uniform(50, 500),
                        "daily_return": np.random.normal(0.0008, 0.02),
                        "volatility": np.random.uniform(0.15, 0.40),
                        "volume": np.random.randint(1000000, 10000000),
                        "beta": np.random.uniform(0.5, 2.0),
                        "market_cap": np.random.uniform(1e9, 1e12),
                        "sector": np.random.choice(["Technology", "Healthcare", "Finance", "Consumer", "Energy"])
                    }
                
                return json.dumps(market_data, indent=2)
                
            except Exception as e:
                return f"Error retrieving market data: {str(e)}"
        
        return Tool(
            name="get_market_data",
            description="Retrieve current market data for stocks including prices, volatility, returns, and fundamentals",
            func=get_market_data
        )
    
    def create_alpha_signals_tool(self) -> Tool:
        """Create tool for retrieving alpha signals."""
        
        def get_alpha_signals(symbols: str, time_horizon: str = "daily") -> str:
            """Retrieve alpha signals for given symbols.
            
            Args:
                symbols: Comma-separated list of stock symbols
                time_horizon: Time horizon for signals ("daily", "weekly", "monthly")
            
            Returns:
                JSON string with alpha signals and confidence scores
            """
            try:
                symbol_list = [s.strip().upper() for s in symbols.split(',')]
                
                # Mock alpha signals - in production, this would query the alpha agent pool
                alpha_signals = {}
                for symbol in symbol_list:
                    signal_strength = np.random.uniform(-2, 2)
                    alpha_signals[symbol] = {
                        "signal_type": "BUY" if signal_strength > 0.5 else "SELL" if signal_strength < -0.5 else "HOLD",
                        "signal_strength": signal_strength,
                        "confidence_score": np.random.uniform(0.3, 0.9),
                        "expected_return": np.random.normal(0.08, 0.03),
                        "time_horizon": time_horizon,
                        "source_models": ["momentum", "mean_reversion", "ml_pattern"],
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }
                
                return json.dumps(alpha_signals, indent=2)
                
            except Exception as e:
                return f"Error retrieving alpha signals: {str(e)}"
        
        return Tool(
            name="get_alpha_signals",
            description="Retrieve alpha generation signals from various quantitative models for portfolio construction",
            func=get_alpha_signals
        )
    
    def create_risk_analysis_tool(self) -> Tool:
        """Create tool for risk analysis."""
        
        def analyze_portfolio_risk(portfolio_weights: str, symbols: str) -> str:
            """Analyze portfolio risk metrics.
            
            Args:
                portfolio_weights: JSON string with asset weights (e.g., '{"AAPL": 0.3, "GOOGL": 0.7}')
                symbols: Comma-separated list of symbols to analyze
            
            Returns:
                JSON string with comprehensive risk metrics
            """
            try:
                weights = json.loads(portfolio_weights)
                symbol_list = [s.strip().upper() for s in symbols.split(',')]
                
                # Calculate portfolio risk metrics
                portfolio_volatility = np.sqrt(sum(weights.get(symbol, 0)**2 * np.random.uniform(0.15, 0.40)**2 
                                                 for symbol in symbol_list))
                
                risk_metrics = {
                    "portfolio_volatility": portfolio_volatility,
                    "value_at_risk_95": portfolio_volatility * 1.65,  # Assuming normal distribution
                    "expected_shortfall": portfolio_volatility * 2.33,
                    "maximum_drawdown": np.random.uniform(0.10, 0.30),
                    "beta": sum(weights.get(symbol, 0) * np.random.uniform(0.5, 2.0) for symbol in symbol_list),
                    "tracking_error": np.random.uniform(0.02, 0.08),
                    "information_ratio": np.random.uniform(-0.5, 1.5),
                    "sharpe_ratio": np.random.uniform(0.5, 2.0),
                    "correlation_matrix": {f"{s1}_{s2}": np.random.uniform(-0.5, 0.8) 
                                         for s1 in symbol_list for s2 in symbol_list if s1 != s2}
                }
                
                return json.dumps(risk_metrics, indent=2)
                
            except Exception as e:
                return f"Error analyzing portfolio risk: {str(e)}"
        
        return Tool(
            name="analyze_portfolio_risk",
            description="Analyze comprehensive risk metrics for a given portfolio allocation",
            func=analyze_portfolio_risk
        )
    
    def create_optimization_tool(self) -> Tool:
        """Create tool for portfolio optimization."""
        
        def optimize_portfolio(optimization_type: str, symbols: str, constraints: str = "{}") -> str:
            """Optimize portfolio allocation.
            
            Args:
                optimization_type: Type of optimization ("mean_variance", "risk_parity", "black_litterman")
                symbols: Comma-separated list of symbols
                constraints: JSON string with optimization constraints
            
            Returns:
                JSON string with optimal weights and performance metrics
            """
            try:
                symbol_list = [s.strip().upper() for s in symbols.split(',')]
                constraints_dict = json.loads(constraints) if constraints else {}
                
                # Simple optimization implementation
                n_assets = len(symbol_list)
                
                if optimization_type == "equal_weight":
                    weights = {symbol: 1.0/n_assets for symbol in symbol_list}
                elif optimization_type == "risk_parity":
                    # Risk parity approximation
                    risks = {symbol: np.random.uniform(0.15, 0.40) for symbol in symbol_list}
                    inv_risks = {symbol: 1.0/risk for symbol, risk in risks.items()}
                    total_inv_risk = sum(inv_risks.values())
                    weights = {symbol: inv_risk/total_inv_risk for symbol, inv_risk in inv_risks.items()}
                elif optimization_type == "mean_variance":
                    # Simplified mean-variance optimization
                    returns = {symbol: np.random.normal(0.08, 0.03) for symbol in symbol_list}
                    risks = {symbol: np.random.uniform(0.15, 0.40) for symbol in symbol_list}
                    
                    # Simple optimization: weight by return/risk ratio
                    ratios = {symbol: returns[symbol]/risks[symbol] for symbol in symbol_list}
                    total_ratio = sum(ratios.values())
                    weights = {symbol: ratio/total_ratio for symbol, ratio in ratios.items()}
                else:
                    weights = {symbol: 1.0/n_assets for symbol in symbol_list}
                
                # Apply constraints
                max_weight = constraints_dict.get("max_weight", 1.0)
                min_weight = constraints_dict.get("min_weight", 0.0)
                
                # Clamp weights to constraints and renormalize
                for symbol in weights:
                    weights[symbol] = max(min_weight, min(max_weight, weights[symbol]))
                
                total_weight = sum(weights.values())
                weights = {symbol: weight/total_weight for symbol, weight in weights.items()}
                
                # Calculate portfolio metrics
                portfolio_return = sum(weights[symbol] * np.random.normal(0.08, 0.03) for symbol in symbol_list)
                portfolio_risk = np.sqrt(sum(weights[symbol]**2 * np.random.uniform(0.15, 0.40)**2 for symbol in symbol_list))
                sharpe_ratio = (portfolio_return - 0.02) / portfolio_risk  # Assuming 2% risk-free rate
                
                result = {
                    "optimization_type": optimization_type,
                    "optimal_weights": weights,
                    "portfolio_metrics": {
                        "expected_return": portfolio_return,
                        "portfolio_risk": portfolio_risk,
                        "sharpe_ratio": sharpe_ratio,
                        "total_weight": sum(weights.values())
                    },
                    "constraints_applied": constraints_dict,
                    "optimization_timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                return json.dumps(result, indent=2)
                
            except Exception as e:
                return f"Error optimizing portfolio: {str(e)}"
        
        return Tool(
            name="optimize_portfolio",
            description="Optimize portfolio allocation using various optimization techniques and constraints",
            func=optimize_portfolio
        )
    
    def get_all_tools(self) -> List[Tool]:
        """Get all available tools for portfolio construction."""
        return [
            self.create_market_data_tool(),
            self.create_alpha_signals_tool(), 
            self.create_risk_analysis_tool(),
            self.create_optimization_tool()
        ]


class PortfolioConstructionAgent:
    """LangGraph-based Portfolio Construction Agent with ReAct capabilities."""
    
    def __init__(self, agent_pool: 'PortfolioConstructionAgentPool', agent_type: str = "mean_variance_optimizer"):
        self.agent_pool = agent_pool
        self.agent_type = agent_type
        self.tools = PortfolioTools(agent_pool)
        self.llm = None
        self.graph = None
        self.memory = MemorySaver()
        
        # Initialize LLM if OpenAI is available
        if agent_pool.openai_client:
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.1,
                api_key=agent_pool.openai_api_key
            )
            self._create_agent_graph()
    
    def _create_agent_graph(self):
        """Create the LangGraph workflow for portfolio construction."""
        
        # Create the graph
        workflow = StateGraph(PortfolioAgentState)
        
        # Add nodes
        workflow.add_node("reasoning", self._reasoning_node)
        workflow.add_node("tool_calling", ToolNode(self.tools.get_all_tools()))
        workflow.add_node("optimization", self._optimization_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("final_output", self._final_output_node)
        
        # Add edges
        workflow.add_edge(START, "reasoning")
        workflow.add_conditional_edges(
            "reasoning",
            self._should_use_tools,
            {
                "use_tools": "tool_calling",
                "optimize": "optimization",
                "validate": "validation",
                "finish": "final_output"
            }
        )
        workflow.add_edge("tool_calling", "reasoning")
        workflow.add_edge("optimization", "validation")
        workflow.add_conditional_edges(
            "validation",
            self._validation_check,
            {
                "retry": "reasoning",
                "finish": "final_output"
            }
        )
        workflow.add_edge("final_output", END)
        
        # Compile the graph
        self.graph = workflow.compile(checkpointer=self.memory)
    
    async def _reasoning_node(self, state: PortfolioAgentState) -> PortfolioAgentState:
        """Reasoning node that decides what action to take next."""
        try:
            messages = state["messages"]
            current_step = state.get("current_step", "start")
            iteration_count = state.get("iteration_count", 0)
            
            # Create system message for reasoning
            system_prompt = f"""
            You are a portfolio construction agent of type '{self.agent_type}'. Your goal is to construct optimal portfolios using quantitative methods.
            
            Current step: {current_step}
            Iteration: {iteration_count}
            
            Available tools:
            1. get_market_data - Retrieve market data for stocks
            2. get_alpha_signals - Get alpha generation signals 
            3. analyze_portfolio_risk - Analyze portfolio risk metrics
            4. optimize_portfolio - Optimize portfolio allocation
            
            Based on the user request and current state, decide what to do next:
            - If you need market data or alpha signals, use the appropriate tools
            - If you have enough data, proceed with optimization
            - If you have optimization results, validate them
            - If everything is complete, provide final output
            
            Think step by step and explain your reasoning clearly.
            """
            
            # Add system message if not present
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages = [SystemMessage(content=system_prompt)] + messages
            
            # Get response from LLM
            response = await self.llm.ainvoke(messages)
            
            # Update state
            state["messages"] = messages + [response]
            state["iteration_count"] = iteration_count + 1
            
            # Determine current step based on response
            response_content = response.content.lower()
            if "market data" in response_content or "get_market_data" in response_content:
                state["current_step"] = "need_market_data"
            elif "alpha signal" in response_content or "get_alpha_signals" in response_content:
                state["current_step"] = "need_alpha_signals"
            elif "risk analysis" in response_content or "analyze_portfolio_risk" in response_content:
                state["current_step"] = "need_risk_analysis"
            elif "optimize" in response_content or "optimization" in response_content:
                state["current_step"] = "ready_to_optimize"
            elif "validate" in response_content or "validation" in response_content:
                state["current_step"] = "ready_to_validate"
            elif "complete" in response_content or "finished" in response_content:
                state["current_step"] = "complete"
            else:
                state["current_step"] = "reasoning"
            
            return state
            
        except Exception as e:
            state["error_message"] = str(e)
            state["current_step"] = "error"
            return state
    
    def _should_use_tools(self, state: PortfolioAgentState) -> str:
        """Decide whether to use tools, optimize, validate, or finish."""
        current_step = state.get("current_step", "start")
        iteration_count = state.get("iteration_count", 0)
        
        # Prevent infinite loops
        if iteration_count > 15:
            return "finish"
        
        if current_step in ["need_market_data", "need_alpha_signals", "need_risk_analysis"]:
            return "use_tools"
        elif current_step == "ready_to_optimize":
            return "optimize"
        elif current_step == "ready_to_validate":
            return "validate"
        elif current_step in ["complete", "error"]:
            return "finish"
        else:
            # For start and reasoning states, go to optimization after a few iterations
            if iteration_count > 3:
                return "optimize"
            else:
                return "use_tools"
    
    async def _optimization_node(self, state: PortfolioAgentState) -> PortfolioAgentState:
        """Perform portfolio optimization."""
        try:
            investment_universe = state.get("investment_universe", [])
            optimization_params = state.get("optimization_params", {})
            
            if not investment_universe:
                state["error_message"] = "No investment universe specified"
                state["current_step"] = "error"
                return state
            
            # Extract optimization parameters
            optimization_type = optimization_params.get("optimization_type", self.agent_type.replace("_optimizer", ""))
            constraints = optimization_params.get("constraints", {})
            
            # Use the optimization tool
            optimize_tool = self.tools.create_optimization_tool()
            symbols_str = ",".join(investment_universe)
            constraints_str = json.dumps(constraints)
            
            result = optimize_tool.func(optimization_type, symbols_str, constraints_str)
            
            # Parse result
            try:
                optimization_result = json.loads(result)
                state["optimization_result"] = optimization_result
                state["current_step"] = "ready_to_validate"
                
                # Add result message
                result_message = AIMessage(content=f"Portfolio optimization completed using {optimization_type} method. Results: {result}")
                state["messages"].append(result_message)
                
            except json.JSONDecodeError:
                state["error_message"] = f"Failed to parse optimization result: {result}"
                state["current_step"] = "error"
            
            return state
            
        except Exception as e:
            state["error_message"] = str(e)
            state["current_step"] = "error"
            return state
    
    async def _validation_node(self, state: PortfolioAgentState) -> PortfolioAgentState:
        """Validate optimization results."""
        try:
            optimization_result = state.get("optimization_result")
            
            if not optimization_result:
                state["error_message"] = "No optimization result to validate"
                state["current_step"] = "error"
                return state
            
            # Perform validation checks
            optimal_weights = optimization_result.get("optimal_weights", {})
            portfolio_metrics = optimization_result.get("portfolio_metrics", {})
            
            validation_results = {
                "weights_sum_to_one": abs(sum(optimal_weights.values()) - 1.0) < 0.01,
                "no_negative_weights": all(w >= 0 for w in optimal_weights.values()),
                "reasonable_sharpe_ratio": 0 < portfolio_metrics.get("sharpe_ratio", 0) < 5,
                "reasonable_risk": 0.05 < portfolio_metrics.get("portfolio_risk", 0) < 0.50
            }
            
            # Check if all validations pass
            all_valid = all(validation_results.values())
            
            if all_valid:
                state["current_step"] = "complete"
                validation_message = AIMessage(content=f"Portfolio validation successful. All checks passed: {validation_results}")
            else:
                state["current_step"] = "retry"
                validation_message = AIMessage(content=f"Portfolio validation failed. Issues found: {validation_results}")
            
            state["messages"].append(validation_message)
            return state
            
        except Exception as e:
            state["error_message"] = str(e)
            state["current_step"] = "error"
            return state
    
    def _validation_check(self, state: PortfolioAgentState) -> str:
        """Check validation results."""
        current_step = state.get("current_step", "")
        iteration_count = state.get("iteration_count", 0)
        
        # Prevent infinite loops
        if iteration_count > 10:
            return "finish"
        
        if current_step == "complete":
            return "finish"
        elif current_step == "retry":
            return "retry"
        else:
            return "finish"
    
    async def _final_output_node(self, state: PortfolioAgentState) -> PortfolioAgentState:
        """Generate final output."""
        try:
            optimization_result = state.get("optimization_result")
            error_message = state.get("error_message")
            
            if error_message:
                final_message = AIMessage(content=f"Portfolio construction failed with error: {error_message}")
            elif optimization_result:
                final_message = AIMessage(content=f"Portfolio construction completed successfully. Final result: {json.dumps(optimization_result, indent=2)}")
            else:
                final_message = AIMessage(content="Portfolio construction completed but no results available.")
            
            state["messages"].append(final_message)
            state["current_step"] = "finished"
            
            return state
            
        except Exception as e:
            error_message = AIMessage(content=f"Error in final output generation: {str(e)}")
            state["messages"].append(error_message)
            state["current_step"] = "error"
            return state
    
    async def construct_portfolio(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to construct portfolio using LangGraph agent."""
        try:
            # Extract request parameters
            investment_universe = request.get("investment_universe", [])
            optimization_params = request.get("optimization_params", {})
            user_input = request.get("user_input", "Construct an optimal portfolio")
            
            # Try LangGraph implementation first
            if self.graph:
                try:
                    # Create initial state
                    initial_state = PortfolioAgentState(
                        messages=[HumanMessage(content=user_input)],
                        portfolio_data={},
                        investment_universe=investment_universe,
                        optimization_params=optimization_params,
                        alpha_signals={},
                        risk_metrics={},
                        transaction_costs={},
                        optimization_result=None,
                        current_step="start",
                        error_message=None,
                        iteration_count=0
                    )
                    
                    # Create a unique thread ID for this conversation
                    thread_id = str(uuid.uuid4())
                    config = {"configurable": {"thread_id": thread_id}}
                    
                    # Run the agent
                    final_state = await self.graph.ainvoke(initial_state, config=config)
                    
                    # Extract results
                    optimization_result = final_state.get("optimization_result")
                    error_message = final_state.get("error_message")
                    messages = final_state.get("messages", [])
                    
                    if error_message:
                        raise Exception(f"LangGraph error: {error_message}")
                    elif optimization_result:
                        return {
                            "status": "success",
                            "optimization_result": optimization_result,
                            "messages": [msg.content for msg in messages],
                            "agent_type": self.agent_type,
                            "thread_id": thread_id
                        }
                    else:
                        raise Exception("No optimization result generated")
                        
                except Exception as langgraph_error:
                    logger.warning(f"LangGraph failed, using fallback: {langgraph_error}")
                    # Fall through to fallback
            
            # Fallback: Simple portfolio optimization
            logger.info("Using simple optimization fallback")
            
            if not investment_universe:
                return {
                    "status": "error",
                    "error": "No investment universe provided"
                }
            
            num_assets = len(investment_universe)
            base_weight = 1.0 / num_assets
            
            # Create portfolio weights
            portfolio_weights = {}
            for symbol in investment_universe:
                portfolio_weights[symbol] = base_weight
            
            # Simple risk metrics calculation
            portfolio_volatility = 0.15  # Assumed 15% portfolio volatility
            expected_return = 0.08  # Assumed 8% expected return
            
            optimization_result = {
                "portfolio_weights": portfolio_weights,
                "expected_return": expected_return,
                "portfolio_volatility": portfolio_volatility,
                "sharpe_ratio": expected_return / portfolio_volatility,
                "optimization_method": "equal_weight_fallback"
            }
            
            return {
                "status": "success",
                "optimization_result": optimization_result,
                "agent_type": self.agent_type,
                "messages": ["Portfolio optimized using equal-weight fallback method"]
            }
            
        except Exception as e:
            logger.error(f"Portfolio construction agent failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent_type": self.agent_type
            }


# Main execution - Portfolio Construction Agent Pool MCP Server
if __name__ == "__main__":
    print("🚀 Starting Portfolio Construction Agent Pool...")
    
    try:
        from mcp.server.fastmcp import FastMCP
        import os
        
        # Create FastMCP server
        portfolio_server = FastMCP("PortfolioConstructionAgentPool")
        
        # Create portfolio construction agent pool first
        portfolio_pool = PortfolioConstructionAgentPool(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            external_memory_config={
                "memory_agent_url": os.getenv("MEMORY_AGENT_URL", "http://localhost:8001"),
                "enable_external_memory": True
            },
            enable_real_time_monitoring=True,
            pool_id="portfolio_pool_main"
        )
        
        # Create portfolio construction agent with the pool
        portfolio_agent = PortfolioConstructionAgent(portfolio_pool)
        
        @portfolio_server.tool(name="process_strategy_request", description="Process portfolio optimization strategy request")
        async def process_strategy_request(request: dict) -> dict:
            """Process portfolio optimization strategy request from orchestrator"""
            try:
                logger.info("Processing portfolio optimization strategy request")
                
                # Extract request details
                symbols = request.get('symbols', ['AAPL', 'MSFT'])
                date = request.get('date', datetime.now().strftime('%Y-%m-%d'))
                alpha_signals = request.get('alpha_signals', {})
                risk_constraints = request.get('risk_constraints', {})
                transaction_costs = request.get('transaction_costs', {})
                
                # Create portfolio optimization request
                optimization_query = f"""
                Optimize portfolio for symbols: {symbols}
                Date: {date}
                Alpha signals: {alpha_signals}
                Risk constraints: {risk_constraints}
                Transaction costs: {transaction_costs}
                
                Please provide optimal portfolio weights and risk metrics.
                """
                
                # Process through portfolio agent's construct_portfolio method
                result = await portfolio_agent.construct_portfolio({
                    "investment_universe": symbols,
                    "optimization_params": {
                        "risk_constraints": risk_constraints,
                        "transaction_costs": transaction_costs
                    },
                    "user_input": optimization_query
                })
                
                logger.info("Portfolio optimization completed successfully")
                
                # Extract results from the portfolio agent response
                optimization_result = result.get("optimization_result", {})
                if isinstance(optimization_result, dict):
                    portfolio_weights = optimization_result.get("portfolio_weights", {})
                    expected_return = optimization_result.get("expected_return", 0.0)
                    portfolio_volatility = optimization_result.get("portfolio_volatility", 0.0)
                    sharpe_ratio = optimization_result.get("sharpe_ratio", 0.0)
                else:
                    portfolio_weights = {}
                    expected_return = 0.0
                    portfolio_volatility = 0.0
                    sharpe_ratio = 0.0
                
                return {
                    "status": "success",
                    "portfolio_weights": portfolio_weights,
                    "risk_metrics": {
                        "portfolio_volatility": portfolio_volatility,
                        "sharpe_ratio": sharpe_ratio
                    },
                    "expected_return": expected_return,
                    "portfolio_risk": portfolio_volatility,
                    "agent_source": "portfolio_construction_agent",
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Portfolio optimization failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "agent_source": "portfolio_construction_agent",
                    "timestamp": datetime.now().isoformat()
                }

        @portfolio_server.tool(name="ping", description="Health check ping")
        def ping() -> str:
            return "pong"

        @portfolio_server.tool(name="status", description="Get portfolio agent status")
        def status() -> dict:
            return {
                "status": "running",
                "agent_type": "portfolio_construction",
                "port": 8083,
                "capabilities": [
                    "portfolio_optimization",
                    "risk_management", 
                    "weight_allocation",
                    "constraint_handling"
                ]
            }
        
        # Configure and start server
        portfolio_server.settings.host = "0.0.0.0"
        portfolio_server.settings.port = 8083
        
        logger.info("Starting Portfolio Construction Agent Pool on port 8083...")
        portfolio_server.run(transport="sse")
        
    except KeyboardInterrupt:
        print("\n🛑 Portfolio Construction Agent Pool shutting down...")
    except Exception as e:
        print(f"❌ Portfolio Construction Agent Pool error: {e}")
        import traceback
        traceback.print_exc()