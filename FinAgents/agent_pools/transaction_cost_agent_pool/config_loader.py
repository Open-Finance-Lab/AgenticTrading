"""
Configuration Loader for Transaction Cost Agent Pool

This module provides configuration management for the transaction cost agent pool,
including loading settings from files, environment variables, and dynamic updates.

Configuration Categories:
- Agent-specific settings
- Market data sources
- Execution venues
- Cost models
- Optimization parameters
- Risk management settings

Author: FinAgent Development Team
Created: 2025-06-25
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    
    agent_id: str
    agent_type: str
    enabled: bool = True
    update_frequency_minutes: int = 60
    max_concurrent_requests: int = 10
    timeout_seconds: int = 30
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VenueConfig:
    """Configuration for execution venues."""
    
    venue_id: str
    venue_name: str
    enabled: bool = True
    fee_structure: Dict[str, float] = field(default_factory=dict)
    market_hours: Dict[str, str] = field(default_factory=dict)
    supported_order_types: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    api_endpoints: Dict[str, str] = field(default_factory=dict)


@dataclass
class CostModelConfig:
    """Configuration for cost models."""
    
    model_id: str
    model_type: str
    enabled: bool = True
    update_frequency_hours: int = 24
    parameters: Dict[str, Any] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""
    
    algorithm_id: str
    algorithm_type: str
    enabled: bool = True
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    time_limit_seconds: int = 300
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    
    max_position_size: float = 1000000.0
    max_daily_cost: float = 10000.0
    max_market_impact_bps: float = 50.0
    stop_loss_threshold: float = 0.05
    var_limit_95: float = 100000.0
    stress_test_scenarios: List[str] = field(default_factory=list)
    monitoring_frequency_minutes: int = 5


@dataclass
class TransactionCostPoolConfig:
    """Complete configuration for transaction cost agent pool."""
    
    pool_id: str = "transaction_cost_pool"
    enabled: bool = True
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    venues: Dict[str, VenueConfig] = field(default_factory=dict)
    cost_models: Dict[str, CostModelConfig] = field(default_factory=dict)
    optimization: Dict[str, OptimizationConfig] = field(default_factory=dict)
    risk_management: RiskConfig = field(default_factory=RiskConfig)
    data_sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    logging_config: Dict[str, Any] = field(default_factory=dict)


class ConfigurationLoader:
    """
    Configuration loader for transaction cost agent pool.
    
    Loads configuration from multiple sources and provides
    dynamic configuration updates.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "config"
        self.config_cache = {}
        self.last_loaded = {}
        
    def load_config(self, config_name: str = "main") -> TransactionCostPoolConfig:
        """
        Load configuration from files and environment variables.
        
        Args:
            config_name: Name of the configuration to load
            
        Returns:
            Complete configuration object
        """
        try:
            # Load base configuration
            base_config = self._load_base_config(config_name)
            
            # Load agent configurations
            agent_configs = self._load_agent_configs()
            
            # Load venue configurations
            venue_configs = self._load_venue_configs()
            
            # Load cost model configurations
            cost_model_configs = self._load_cost_model_configs()
            
            # Load optimization configurations
            optimization_configs = self._load_optimization_configs()
            
            # Load risk management configuration
            risk_config = self._load_risk_config()
            
            # Apply environment variable overrides
            self._apply_env_overrides(base_config)
            
            # Combine all configurations
            complete_config = TransactionCostPoolConfig(
                pool_id=base_config.get("pool_id", "transaction_cost_pool"),
                enabled=base_config.get("enabled", True),
                agents=agent_configs,
                venues=venue_configs,
                cost_models=cost_model_configs,
                optimization=optimization_configs,
                risk_management=risk_config,
                data_sources=base_config.get("data_sources", {}),
                logging_config=base_config.get("logging", {})
            )
            
            # Cache the configuration
            self.config_cache[config_name] = complete_config
            self.last_loaded[config_name] = os.path.getmtime(self.config_dir / f"{config_name}.yaml")
            
            logger.info(f"Loaded configuration '{config_name}' successfully")
            return complete_config
            
        except Exception as e:
            logger.error(f"Failed to load configuration '{config_name}': {e}")
            # Return default configuration
            return self._get_default_config()
    
    def _load_base_config(self, config_name: str) -> Dict[str, Any]:
        """Load base configuration from file."""
        
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        
        # Try JSON format
        config_file = self.config_dir / f"{config_name}.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        
        logger.warning(f"No configuration file found for '{config_name}', using defaults")
        return {}
    
    def _load_agent_configs(self) -> Dict[str, AgentConfig]:
        """Load agent configurations."""
        
        agent_configs = {}
        
        # Default agent configurations
        default_agents = {
            "market_impact_estimator": {
                "agent_type": "market_impact_estimator",
                "enabled": True,
                "update_frequency_minutes": 30,
                "parameters": {
                    "alpha": 0.6,
                    "beta": 0.4,
                    "gamma": 0.2
                }
            },
            "venue_cost_analyzer": {
                "agent_type": "venue_cost_analyzer",
                "enabled": True,
                "update_frequency_minutes": 60,
                "parameters": {
                    "historical_window_days": 30,
                    "confidence_level": 0.95
                }
            },
            "execution_performance_monitor": {
                "agent_type": "execution_performance_monitor",
                "enabled": True,
                "update_frequency_minutes": 15,
                "parameters": {
                    "alert_threshold_bps": 10.0,
                    "monitoring_window_minutes": 60
                }
            },
            "cost_optimizer": {
                "agent_type": "cost_optimizer",
                "enabled": True,
                "update_frequency_minutes": 120,
                "parameters": {
                    "optimization_horizon_hours": 4,
                    "max_iterations": 1000
                }
            },
            "timing_optimizer": {
                "agent_type": "timing_optimizer",
                "enabled": True,
                "update_frequency_minutes": 30,
                "parameters": {
                    "max_execution_windows": 6,
                    "min_window_size_minutes": 15
                }
            }
        }
        
        # Load from file if exists
        agents_file = self.config_dir / "agents.yaml"
        if agents_file.exists():
            with open(agents_file, 'r') as f:
                file_agents = yaml.safe_load(f) or {}
                default_agents.update(file_agents)
        
        # Convert to AgentConfig objects
        for agent_id, config_data in default_agents.items():
            agent_configs[agent_id] = AgentConfig(
                agent_id=agent_id,
                agent_type=config_data.get("agent_type", agent_id),
                enabled=config_data.get("enabled", True),
                update_frequency_minutes=config_data.get("update_frequency_minutes", 60),
                max_concurrent_requests=config_data.get("max_concurrent_requests", 10),
                timeout_seconds=config_data.get("timeout_seconds", 30),
                parameters=config_data.get("parameters", {})
            )
        
        return agent_configs
    
    def _load_venue_configs(self) -> Dict[str, VenueConfig]:
        """Load venue configurations."""
        
        venue_configs = {}
        
        # Default venue configurations
        default_venues = {
            "NYSE": {
                "venue_name": "New York Stock Exchange",
                "enabled": True,
                "fee_structure": {
                    "add_liquidity": -0.003,
                    "remove_liquidity": 0.003,
                    "regulatory_fee": 0.0000221
                },
                "market_hours": {
                    "open": "09:30",
                    "close": "16:00",
                    "timezone": "America/New_York"
                },
                "supported_order_types": ["LIMIT", "MARKET", "STOP", "STOP_LIMIT"],
                "latency_ms": 0.5
            },
            "NASDAQ": {
                "venue_name": "NASDAQ",
                "enabled": True,
                "fee_structure": {
                    "add_liquidity": -0.002,
                    "remove_liquidity": 0.0035,
                    "regulatory_fee": 0.0000221
                },
                "market_hours": {
                    "open": "09:30",
                    "close": "16:00",
                    "timezone": "America/New_York"
                },
                "supported_order_types": ["LIMIT", "MARKET", "STOP", "STOP_LIMIT"],
                "latency_ms": 0.4
            },
            "ARCA": {
                "venue_name": "NYSE Arca",
                "enabled": True,
                "fee_structure": {
                    "add_liquidity": -0.0025,
                    "remove_liquidity": 0.0032,
                    "regulatory_fee": 0.0000221
                },
                "market_hours": {
                    "open": "09:30",
                    "close": "16:00",
                    "timezone": "America/New_York"
                },
                "supported_order_types": ["LIMIT", "MARKET"],
                "latency_ms": 0.6
            }
        }
        
        # Load from file if exists
        venues_file = self.config_dir / "venues.yaml"
        if venues_file.exists():
            with open(venues_file, 'r') as f:
                file_venues = yaml.safe_load(f) or {}
                default_venues.update(file_venues)
        
        # Convert to VenueConfig objects
        for venue_id, config_data in default_venues.items():
            venue_configs[venue_id] = VenueConfig(
                venue_id=venue_id,
                venue_name=config_data.get("venue_name", venue_id),
                enabled=config_data.get("enabled", True),
                fee_structure=config_data.get("fee_structure", {}),
                market_hours=config_data.get("market_hours", {}),
                supported_order_types=config_data.get("supported_order_types", []),
                latency_ms=config_data.get("latency_ms", 0.0),
                api_endpoints=config_data.get("api_endpoints", {})
            )
        
        return venue_configs
    
    def _load_cost_model_configs(self) -> Dict[str, CostModelConfig]:
        """Load cost model configurations."""
        
        cost_model_configs = {}
        
        # Default cost model configurations
        default_models = {
            "linear_impact_model": {
                "model_type": "linear_impact",
                "enabled": True,
                "update_frequency_hours": 24,
                "parameters": {
                    "alpha": 0.5,
                    "beta": 0.3,
                    "decay_rate": 0.1
                },
                "data_sources": ["execution_history", "market_data"],
                "validation_rules": {
                    "min_data_points": 100,
                    "max_parameter_change": 0.5
                }
            },
            "square_root_model": {
                "model_type": "square_root",
                "enabled": True,
                "update_frequency_hours": 12,
                "parameters": {
                    "temporary_impact_coeff": 0.314,
                    "permanent_impact_coeff": 0.142,
                    "volatility_factor": 1.0
                },
                "data_sources": ["execution_history", "volatility_data"],
                "validation_rules": {
                    "min_data_points": 200,
                    "max_parameter_change": 0.3
                }
            }
        }
        
        # Load from file if exists
        models_file = self.config_dir / "cost_models.yaml"
        if models_file.exists():
            with open(models_file, 'r') as f:
                file_models = yaml.safe_load(f) or {}
                default_models.update(file_models)
        
        # Convert to CostModelConfig objects
        for model_id, config_data in default_models.items():
            cost_model_configs[model_id] = CostModelConfig(
                model_id=model_id,
                model_type=config_data.get("model_type", model_id),
                enabled=config_data.get("enabled", True),
                update_frequency_hours=config_data.get("update_frequency_hours", 24),
                parameters=config_data.get("parameters", {}),
                data_sources=config_data.get("data_sources", []),
                validation_rules=config_data.get("validation_rules", {})
            )
        
        return cost_model_configs
    
    def _load_optimization_configs(self) -> Dict[str, OptimizationConfig]:
        """Load optimization configurations."""
        
        optimization_configs = {}
        
        # Default optimization configurations
        default_optimizations = {
            "cost_minimization": {
                "algorithm_type": "gradient_descent",
                "enabled": True,
                "max_iterations": 1000,
                "convergence_threshold": 1e-6,
                "time_limit_seconds": 300,
                "parameters": {
                    "learning_rate": 0.01,
                    "momentum": 0.9,
                    "regularization": 0.001
                },
                "constraints": {
                    "max_market_impact_bps": 50.0,
                    "min_fill_rate": 0.95
                }
            },
            "venue_routing": {
                "algorithm_type": "mixed_integer_programming",
                "enabled": True,
                "max_iterations": 500,
                "convergence_threshold": 1e-4,
                "time_limit_seconds": 180,
                "parameters": {
                    "cost_weight": 0.7,
                    "speed_weight": 0.3
                },
                "constraints": {
                    "max_venues_per_order": 3,
                    "min_order_size": 100
                }
            }
        }
        
        # Load from file if exists
        optimization_file = self.config_dir / "optimization.yaml"
        if optimization_file.exists():
            with open(optimization_file, 'r') as f:
                file_optimizations = yaml.safe_load(f) or {}
                default_optimizations.update(file_optimizations)
        
        # Convert to OptimizationConfig objects
        for opt_id, config_data in default_optimizations.items():
            optimization_configs[opt_id] = OptimizationConfig(
                algorithm_id=opt_id,
                algorithm_type=config_data.get("algorithm_type", opt_id),
                enabled=config_data.get("enabled", True),
                max_iterations=config_data.get("max_iterations", 1000),
                convergence_threshold=config_data.get("convergence_threshold", 1e-6),
                time_limit_seconds=config_data.get("time_limit_seconds", 300),
                parameters=config_data.get("parameters", {}),
                constraints=config_data.get("constraints", {})
            )
        
        return optimization_configs
    
    def _load_risk_config(self) -> RiskConfig:
        """Load risk management configuration."""
        
        # Default risk configuration
        default_risk = {
            "max_position_size": 1000000.0,
            "max_daily_cost": 10000.0,
            "max_market_impact_bps": 50.0,
            "stop_loss_threshold": 0.05,
            "var_limit_95": 100000.0,
            "stress_test_scenarios": [
                "market_crash",
                "volatility_spike",
                "liquidity_crisis"
            ],
            "monitoring_frequency_minutes": 5
        }
        
        # Load from file if exists
        risk_file = self.config_dir / "risk.yaml"
        if risk_file.exists():
            with open(risk_file, 'r') as f:
                file_risk = yaml.safe_load(f) or {}
                default_risk.update(file_risk)
        
        return RiskConfig(**default_risk)
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> None:
        """Apply environment variable overrides to configuration."""
        
        # Environment variable mappings
        env_mappings = {
            "TC_POOL_ENABLED": ("enabled", bool),
            "TC_POOL_ID": ("pool_id", str),
            "TC_MAX_POSITION_SIZE": ("risk_management.max_position_size", float),
            "TC_MAX_DAILY_COST": ("risk_management.max_daily_cost", float),
            "TC_LOG_LEVEL": ("logging.level", str),
        }
        
        for env_var, (config_path, config_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Convert to appropriate type
                    if config_type == bool:
                        typed_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif config_type == float:
                        typed_value = float(env_value)
                    elif config_type == int:
                        typed_value = int(env_value)
                    else:
                        typed_value = env_value
                    
                    # Set nested configuration value
                    self._set_nested_config(config, config_path, typed_value)
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable {env_var}={env_value}: {e}")
    
    def _set_nested_config(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested configuration value."""
        
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _get_default_config(self) -> TransactionCostPoolConfig:
        """Get default configuration when loading fails."""
        
        return TransactionCostPoolConfig(
            pool_id="transaction_cost_pool",
            enabled=True,
            agents={},
            venues={},
            cost_models={},
            optimization={},
            risk_management=RiskConfig(),
            data_sources={},
            logging_config={"level": "INFO"}
        )
    
    def reload_config(self, config_name: str = "main") -> TransactionCostPoolConfig:
        """
        Reload configuration if files have changed.
        
        Args:
            config_name: Name of the configuration to reload
            
        Returns:
            Reloaded configuration or cached version if unchanged
        """
        try:
            config_file = self.config_dir / f"{config_name}.yaml"
            
            if config_file.exists():
                current_mtime = os.path.getmtime(config_file)
                last_mtime = self.last_loaded.get(config_name, 0)
                
                if current_mtime > last_mtime:
                    logger.info(f"Configuration '{config_name}' has changed, reloading...")
                    return self.load_config(config_name)
            
            # Return cached version
            return self.config_cache.get(config_name, self._get_default_config())
            
        except Exception as e:
            logger.error(f"Failed to reload configuration '{config_name}': {e}")
            return self.config_cache.get(config_name, self._get_default_config())
    
    def validate_config(self, config: TransactionCostPoolConfig) -> List[str]:
        """
        Validate configuration for consistency and completeness.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate pool-level settings
        if not config.pool_id:
            errors.append("Pool ID cannot be empty")
        
        # Validate agents
        for agent_id, agent_config in config.agents.items():
            if not agent_config.agent_type:
                errors.append(f"Agent '{agent_id}' missing agent_type")
            
            if agent_config.update_frequency_minutes <= 0:
                errors.append(f"Agent '{agent_id}' has invalid update frequency")
        
        # Validate venues
        for venue_id, venue_config in config.venues.items():
            if not venue_config.venue_name:
                errors.append(f"Venue '{venue_id}' missing venue_name")
            
            if venue_config.latency_ms < 0:
                errors.append(f"Venue '{venue_id}' has negative latency")
        
        # Validate risk management
        if config.risk_management.max_position_size <= 0:
            errors.append("Max position size must be positive")
        
        if config.risk_management.max_daily_cost <= 0:
            errors.append("Max daily cost must be positive")
        
        return errors


# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    config_loader = ConfigurationLoader()
    
    # Load configuration
    config = config_loader.load_config("main")
    
    print("=== Transaction Cost Pool Configuration ===")
    print(f"Pool ID: {config.pool_id}")
    print(f"Enabled: {config.enabled}")
    print(f"Number of Agents: {len(config.agents)}")
    print(f"Number of Venues: {len(config.venues)}")
    print(f"Number of Cost Models: {len(config.cost_models)}")
    
    # Validate configuration
    errors = config_loader.validate_config(config)
    if errors:
        print("\n=== Configuration Errors ===")
        for error in errors:
            print(f"- {error}")
    else:
        print("\n✓ Configuration is valid")
    
    # Test reload
    reloaded_config = config_loader.reload_config("main")
    print(f"\n✓ Configuration reloaded: {reloaded_config.pool_id}")
