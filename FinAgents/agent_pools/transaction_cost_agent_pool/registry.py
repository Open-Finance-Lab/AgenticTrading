"""
Transaction Cost Agent Pool - Agent Registry System

This module implements a comprehensive agent registration and discovery system
for transaction cost analysis agents. It provides centralized management of
agent configurations, capabilities, and lifecycle states.

Key Features:
- Dynamic agent registration and discovery
- Configuration-driven agent initialization
- Capability-based agent selection
- Performance monitoring integration

Author: FinAgent Development Team
License: OpenMDW
"""

import logging
import json
import os
import yaml
from typing import Dict, Any, List, Optional, Type, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Initialize logger
logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Enumeration of supported agent types in the transaction cost pool."""
    PRE_TRADE = "pre_trade"
    POST_TRADE = "post_trade"
    OPTIMIZATION = "optimization"
    RISK_ADJUSTED = "risk_adjusted"

class AgentStatus(Enum):
    """Enumeration of possible agent status states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class AgentCapability:
    """
    Represents a specific capability of a transaction cost agent.
    
    Attributes:
        name (str): Capability name
        description (str): Detailed capability description
        input_types (List[str]): Supported input data types
        output_types (List[str]): Produced output data types
        performance_metrics (Dict[str, Any]): Performance characteristics
    """
    name: str
    description: str
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentConfiguration:
    """
    Comprehensive configuration for a transaction cost agent.
    
    Attributes:
        agent_id (str): Unique agent identifier
        agent_type (AgentType): Agent category
        class_name (str): Agent implementation class name
        module_path (str): Python module path for agent class
        description (str): Agent description
        capabilities (List[AgentCapability]): Agent capabilities
        config_params (Dict[str, Any]): Configuration parameters
        dependencies (List[str]): Required dependencies
        resource_requirements (Dict[str, Any]): Resource requirements
        status (AgentStatus): Current agent status
        created_at (str): Creation timestamp
        last_updated (str): Last update timestamp
    """
    agent_id: str
    agent_type: AgentType
    class_name: str
    module_path: str
    description: str = ""
    capabilities: List[AgentCapability] = field(default_factory=list)
    config_params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    status: AgentStatus = AgentStatus.UNINITIALIZED
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class AgentRegistry:
    """
    Centralized registry for transaction cost agents.
    
    This class manages the registration, discovery, and lifecycle of all
    transaction cost agents within the pool. It provides:
    
    - Dynamic agent registration and deregistration
    - Capability-based agent discovery
    - Configuration management and validation
    - Performance monitoring integration
    - Dependency resolution
    """

    def __init__(self):
        """Initialize the agent registry with empty state."""
        self._agents: Dict[str, AgentConfiguration] = {}
        self._agent_instances: Dict[str, Any] = {}
        self._capability_index: Dict[str, List[str]] = {}
        self._type_index: Dict[AgentType, List[str]] = {
            agent_type: [] for agent_type in AgentType
        }
        
        logger.info("Agent registry initialized")

    def __len__(self) -> int:
        """
        Return the number of registered agents.
        
        Returns:
            int: Number of agents in the registry
        """
        return len(self._agents)

    def register_agent(
        self,
        agent_id: str,
        agent_type: AgentType,
        class_name: str,
        module_path: str,
        description: str = "",
        capabilities: Optional[List[AgentCapability]] = None,
        config_params: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        resource_requirements: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a new transaction cost agent.
        
        Args:
            agent_id (str): Unique agent identifier
            agent_type (AgentType): Agent category
            class_name (str): Agent implementation class name
            module_path (str): Python module path
            description (str): Agent description
            capabilities (Optional[List[AgentCapability]]): Agent capabilities
            config_params (Optional[Dict[str, Any]]): Configuration parameters
            dependencies (Optional[List[str]]): Required dependencies
            resource_requirements (Optional[Dict[str, Any]]): Resource requirements
        
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            if agent_id in self._agents:
                logger.warning(f"Agent {agent_id} already registered, updating configuration")
            
            # Create agent configuration
            config = AgentConfiguration(
                agent_id=agent_id,
                agent_type=agent_type,
                class_name=class_name,
                module_path=module_path,
                description=description,
                capabilities=capabilities or [],
                config_params=config_params or {},
                dependencies=dependencies or [],
                resource_requirements=resource_requirements or {}
            )
            
            # Register in main registry
            self._agents[agent_id] = config
            
            # Update type index
            if agent_id not in self._type_index[agent_type]:
                self._type_index[agent_type].append(agent_id)
            
            # Update capability index
            for capability in config.capabilities:
                if capability.name not in self._capability_index:
                    self._capability_index[capability.name] = []
                if agent_id not in self._capability_index[capability.name]:
                    self._capability_index[capability.name].append(agent_id)
            
            logger.info(f"Successfully registered agent: {agent_id} ({agent_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {str(e)}")
            return False

    def deregister_agent(self, agent_id: str) -> bool:
        """
        Deregister an agent from the registry.
        
        Args:
            agent_id (str): Agent identifier to deregister
        
        Returns:
            bool: True if deregistration successful, False otherwise
        """
        try:
            if agent_id not in self._agents:
                logger.warning(f"Agent {agent_id} not found in registry")
                return False
            
            config = self._agents[agent_id]
            
            # Remove from type index
            if agent_id in self._type_index[config.agent_type]:
                self._type_index[config.agent_type].remove(agent_id)
            
            # Remove from capability index
            for capability in config.capabilities:
                if (capability.name in self._capability_index and 
                    agent_id in self._capability_index[capability.name]):
                    self._capability_index[capability.name].remove(agent_id)
            
            # Remove from main registry
            del self._agents[agent_id]
            
            # Remove instance if exists
            if agent_id in self._agent_instances:
                del self._agent_instances[agent_id]
            
            logger.info(f"Successfully deregistered agent: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister agent {agent_id}: {str(e)}")
            return False

    def get_agent(self, agent_id: str) -> Optional[AgentConfiguration]:
        """
        Retrieve agent configuration by ID.
        
        Args:
            agent_id (str): Agent identifier
        
        Returns:
            Optional[AgentConfiguration]: Agent configuration or None if not found
        """
        return self._agents.get(agent_id)

    def get_agents_by_type(self, agent_type: AgentType) -> List[AgentConfiguration]:
        """
        Retrieve all agents of a specific type.
        
        Args:
            agent_type (AgentType): Agent type to filter by
        
        Returns:
            List[AgentConfiguration]: List of matching agent configurations
        """
        agent_ids = self._type_index.get(agent_type, [])
        return [self._agents[agent_id] for agent_id in agent_ids if agent_id in self._agents]

    def get_agents_by_capability(self, capability_name: str) -> List[AgentConfiguration]:
        """
        Retrieve agents that support a specific capability.
        
        Args:
            capability_name (str): Capability name to search for
        
        Returns:
            List[AgentConfiguration]: List of agents with the specified capability
        """
        agent_ids = self._capability_index.get(capability_name, [])
        return [self._agents[agent_id] for agent_id in agent_ids if agent_id in self._agents]

    def find_best_agent(
        self,
        capability_name: str,
        performance_criteria: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentConfiguration]:
        """
        Find the best agent for a specific capability based on performance criteria.
        
        Args:
            capability_name (str): Required capability
            performance_criteria (Optional[Dict[str, Any]]): Performance requirements
        
        Returns:
            Optional[AgentConfiguration]: Best matching agent or None
        """
        try:
            candidates = self.get_agents_by_capability(capability_name)
            
            if not candidates:
                return None
            
            if not performance_criteria:
                # Return first available agent
                return candidates[0]
            
            # Score agents based on performance criteria
            best_agent = None
            best_score = float('-inf')
            
            for agent in candidates:
                score = self._calculate_agent_score(agent, performance_criteria)
                if score > best_score:
                    best_score = score
                    best_agent = agent
            
            return best_agent
            
        except Exception as e:
            logger.error(f"Error finding best agent for {capability_name}: {str(e)}")
            return None

    def _calculate_agent_score(
        self,
        agent: AgentConfiguration,
        criteria: Dict[str, Any]
    ) -> float:
        """
        Calculate an agent's score based on performance criteria.
        
        Args:
            agent (AgentConfiguration): Agent to score
            criteria (Dict[str, Any]): Scoring criteria
        
        Returns:
            float: Agent score
        """
        try:
            score = 0.0
            
            # Example scoring logic - can be extended
            if agent.status == AgentStatus.RUNNING:
                score += 10.0
            elif agent.status == AgentStatus.INITIALIZED:
                score += 5.0
            
            # Score based on capability performance metrics
            for capability in agent.capabilities:
                metrics = capability.performance_metrics
                if 'accuracy' in metrics and 'accuracy' in criteria:
                    score += metrics['accuracy'] * criteria.get('accuracy_weight', 1.0)
                if 'latency' in metrics and 'latency' in criteria:
                    # Lower latency is better
                    target_latency = criteria.get('target_latency', 100)
                    if metrics['latency'] < target_latency:
                        score += (target_latency - metrics['latency']) / target_latency * 10
            
            return score
            
        except Exception as e:
            logger.warning(f"Error calculating agent score: {str(e)}")
            return 0.0

    def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """
        Update the status of a registered agent.
        
        Args:
            agent_id (str): Agent identifier
            status (AgentStatus): New status
        
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            if agent_id in self._agents:
                self._agents[agent_id].status = status
                self._agents[agent_id].last_updated = datetime.utcnow().isoformat()
                logger.debug(f"Updated agent {agent_id} status to {status.value}")
                return True
            else:
                logger.warning(f"Agent {agent_id} not found for status update")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update agent status: {str(e)}")
            return False

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive registry statistics.
        
        Returns:
            Dict[str, Any]: Registry statistics and metrics
        """
        try:
            stats = {
                "total_agents": len(self._agents),
                "agents_by_type": {
                    agent_type.value: len(agent_ids)
                    for agent_type, agent_ids in self._type_index.items()
                },
                "agents_by_status": {},
                "capabilities": list(self._capability_index.keys()),
                "total_capabilities": len(self._capability_index),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Count agents by status
            status_counts = {}
            for agent in self._agents.values():
                status = agent.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            stats["agents_by_status"] = status_counts
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating registry stats: {str(e)}")
            return {"error": str(e)}

    def export_configuration(self, file_path: str) -> bool:
        """
        Export current registry configuration to file.
        
        Args:
            file_path (str): Output file path
        
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            export_data = {
                "registry_version": "1.0.0",
                "exported_at": datetime.utcnow().isoformat(),
                "agents": {}
            }
            
            for agent_id, config in self._agents.items():
                export_data["agents"][agent_id] = {
                    "agent_type": config.agent_type.value,
                    "class_name": config.class_name,
                    "module_path": config.module_path,
                    "description": config.description,
                    "capabilities": [
                        {
                            "name": cap.name,
                            "description": cap.description,
                            "input_types": cap.input_types,
                            "output_types": cap.output_types,
                            "performance_metrics": cap.performance_metrics
                        }
                        for cap in config.capabilities
                    ],
                    "config_params": config.config_params,
                    "dependencies": config.dependencies,
                    "resource_requirements": config.resource_requirements
                }
            
            with open(file_path, 'w') as f:
                if file_path.endswith('.json'):
                    json.dump(export_data, f, indent=2)
                elif file_path.endswith(('.yaml', '.yml')):
                    yaml.dump(export_data, f, default_flow_style=False)
                else:
                    raise ValueError("Unsupported file format")
            
            logger.info(f"Registry configuration exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {str(e)}")
            return False

    def load_configuration(self, file_path: str) -> bool:
        """
        Load registry configuration from file.
        
        Args:
            file_path (str): Configuration file path
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Configuration file not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    config_data = json.load(f)
                elif file_path.endswith(('.yaml', '.yml')):
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError("Unsupported file format")
            
            # Load agents from configuration
            for agent_id, agent_config in config_data.get("agents", {}).items():
                capabilities = [
                    AgentCapability(
                        name=cap["name"],
                        description=cap["description"],
                        input_types=cap.get("input_types", []),
                        output_types=cap.get("output_types", []),
                        performance_metrics=cap.get("performance_metrics", {})
                    )
                    for cap in agent_config.get("capabilities", [])
                ]
                
                self.register_agent(
                    agent_id=agent_id,
                    agent_type=AgentType(agent_config["agent_type"]),
                    class_name=agent_config["class_name"],
                    module_path=agent_config["module_path"],
                    description=agent_config.get("description", ""),
                    capabilities=capabilities,
                    config_params=agent_config.get("config_params", {}),
                    dependencies=agent_config.get("dependencies", []),
                    resource_requirements=agent_config.get("resource_requirements", {})
                )
            
            logger.info(f"Registry configuration loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return False


# Global registry instance
AGENT_REGISTRY = AgentRegistry()

# Convenience functions
def register_agent(agent_id: str, **kwargs) -> bool:
    """Register an agent using the global registry."""
    return AGENT_REGISTRY.register_agent(agent_id, **kwargs)

def get_agent(agent_id: str) -> Optional[AgentConfiguration]:
    """Get an agent configuration from the global registry."""
    return AGENT_REGISTRY.get_agent(agent_id)

def get_agents_by_type(agent_type: AgentType) -> List[AgentConfiguration]:
    """Get agents by type from the global registry."""
    return AGENT_REGISTRY.get_agents_by_type(agent_type)

def find_best_agent(capability_name: str, performance_criteria: Optional[Dict[str, Any]] = None) -> Optional[AgentConfiguration]:
    """Find the best agent for a capability using the global registry."""
    return AGENT_REGISTRY.find_best_agent(capability_name, performance_criteria)

# Pre-register default agents
def preload_default_agents():
    """
    Pre-register default transaction cost agents.
    
    This function registers the standard set of transaction cost agents
    that are available in the system by default.
    """
    logger.info("Preloading default transaction cost agents...")
    
    # Pre-trade agents
    register_agent(
        agent_id="cost_predictor",
        agent_type=AgentType.PRE_TRADE,
        class_name="CostPredictor",
        module_path="transaction_cost_agent_pool.agents.pre_trade.cost_predictor",
        description="Advanced transaction cost prediction using machine learning models",
        capabilities=[
            AgentCapability(
                name="cost_estimation",
                description="Estimate transaction costs for various order types and market conditions",
                input_types=["trade_specification", "market_data"],
                output_types=["cost_breakdown", "confidence_intervals"],
                performance_metrics={"accuracy": 0.92, "latency": 15.5}
            )
        ]
    )
    
    register_agent(
        agent_id="impact_estimator",
        agent_type=AgentType.PRE_TRADE,
        class_name="ImpactEstimator",
        module_path="transaction_cost_agent_pool.agents.pre_trade.impact_estimator",
        description="Market impact estimation for large orders",
        capabilities=[
            AgentCapability(
                name="market_impact_estimation",
                description="Estimate market impact for various order sizes and execution strategies",
                input_types=["order_specification", "market_microstructure"],
                output_types=["impact_forecast", "temporary_permanent_breakdown"],
                performance_metrics={"accuracy": 0.88, "latency": 12.3}
            )
        ]
    )
    
    # Post-trade agents
    register_agent(
        agent_id="execution_analyzer",
        agent_type=AgentType.POST_TRADE,
        class_name="ExecutionAnalyzer",
        module_path="transaction_cost_agent_pool.agents.post_trade.execution_analyzer",
        description="Comprehensive post-trade execution quality analysis",
        capabilities=[
            AgentCapability(
                name="execution_quality_analysis",
                description="Analyze execution quality against various benchmarks",
                input_types=["execution_data", "benchmark_data"],
                output_types=["quality_metrics", "attribution_analysis"],
                performance_metrics={"processing_speed": 1000, "latency": 8.7}
            )
        ]
    )
    
    # Optimization agents
    register_agent(
        agent_id="portfolio_optimizer",
        agent_type=AgentType.OPTIMIZATION,
        class_name="PortfolioOptimizer",
        module_path="transaction_cost_agent_pool.agents.optimization.portfolio_optimizer",
        description="Portfolio-level execution optimization",
        capabilities=[
            AgentCapability(
                name="portfolio_optimization",
                description="Optimize portfolio execution to minimize total transaction costs",
                input_types=["portfolio_trades", "optimization_constraints"],
                output_types=["optimized_execution_plan", "cost_savings_estimate"],
                performance_metrics={"optimization_quality": 0.94, "computation_time": 25.1}
            )
        ]
    )
    
    # Risk-adjusted agents
    register_agent(
        agent_id="var_adjusted_cost",
        agent_type=AgentType.RISK_ADJUSTED,
        class_name="VaRAdjustedCostAnalyzer",
        module_path="transaction_cost_agent_pool.agents.risk_adjusted.var_adjusted_cost",
        description="Value-at-Risk adjusted transaction cost analysis",
        capabilities=[
            AgentCapability(
                name="risk_adjusted_cost_analysis",
                description="Calculate transaction costs adjusted for market risk exposure",
                input_types=["trade_data", "risk_parameters"],
                output_types=["risk_adjusted_metrics", "var_decomposition"],
                performance_metrics={"model_accuracy": 0.91, "calculation_speed": 500}
            )
        ]
    )
    
    logger.info(f"Preloaded {len(AGENT_REGISTRY._agents)} default agents")

# Initialize default agents on module import
preload_default_agents()
