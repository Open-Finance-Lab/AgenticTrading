"""
Configuration Schema for Alpha Agent Pool
=========================================

This module defines the configuration schemas used by alpha agents for
strategy development, factor research, and parameter management.

Author: FinAgent Development Team
Created: 2025-07-25
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json

class AgentType(Enum):
    """Agent type enumeration."""
    THEORY_DRIVEN = "theory_driven"
    EMPIRICAL = "empirical"
    AUTONOMOUS = "autonomous"
    HYBRID = "hybrid"

class FactorCategory(Enum):
    """Alpha factor category enumeration."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    MACRO = "macro"
    SENTIMENT = "sentiment"

@dataclass
class AgentConfig:
    """Base configuration for all alpha agents."""
    agent_id: str
    name: str
    agent_type: AgentType
    description: str = ""
    enabled: bool = True
    
    # Performance parameters
    max_concurrent_requests: int = 10
    timeout_seconds: int = 300
    retry_attempts: int = 3
    
    # Memory integration
    memory_integration: bool = True
    memory_server_url: str = "http://127.0.0.1:8000"
    a2a_memory_url: str = "http://127.0.0.1:8002"
    
    # Logging configuration
    log_level: str = "INFO"
    log_to_file: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create configuration from dictionary."""
        if 'agent_type' in data and isinstance(data['agent_type'], str):
            data['agent_type'] = AgentType(data['agent_type'])
        return cls(**data)

@dataclass
class MomentumAgentConfig(AgentConfig):
    """Configuration for momentum-based alpha agents."""
    agent_type: AgentType = AgentType.THEORY_DRIVEN
    
    # Momentum-specific parameters
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    momentum_threshold: float = 0.02
    rebalance_frequency: str = "daily"
    
    # Risk management
    max_position_size: float = 0.05
    stop_loss_threshold: float = -0.10
    volatility_scaling: bool = True
    
    # Factor generation
    generate_cross_sectional: bool = True
    generate_time_series: bool = True
    normalize_factors: bool = True
    
    # Backtesting
    benchmark_symbol: str = "SPY"
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005

@dataclass
class MeanReversionAgentConfig(AgentConfig):
    """Configuration for mean reversion alpha agents."""
    agent_type: AgentType = AgentType.THEORY_DRIVEN
    
    # Mean reversion parameters
    lookback_window: int = 20
    mean_reversion_threshold: float = 2.0  # Standard deviations
    holding_period: int = 5
    
    # Technical indicators
    use_bollinger_bands: bool = True
    use_rsi: bool = True
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    
    # Risk management
    max_positions: int = 10
    position_sizing: str = "equal_weight"  # or "volatility_adjusted"
    max_drawdown: float = 0.15

@dataclass
class EmpiricalAgentConfig(AgentConfig):
    """Configuration for empirical/data-mining agents."""
    agent_type: AgentType = AgentType.EMPIRICAL
    
    # Data mining parameters
    min_pattern_confidence: float = 0.6
    max_pattern_complexity: int = 5
    feature_selection_method: str = "mutual_info"
    
    # Machine learning
    model_type: str = "random_forest"  # or "xgboost", "lightgbm", "neural_network"
    train_test_split: float = 0.8
    cross_validation_folds: int = 5
    hyperparameter_tuning: bool = True
    
    # Pattern recognition
    detect_regime_changes: bool = True
    seasonal_adjustment: bool = True
    outlier_detection: bool = True
    
    # Feature engineering
    generate_technical_features: bool = True
    generate_statistical_features: bool = True
    feature_importance_threshold: float = 0.01

@dataclass
class AutonomousAgentConfig(AgentConfig):
    """Configuration for autonomous self-directed agents."""
    agent_type: AgentType = AgentType.AUTONOMOUS
    
    # Autonomy parameters
    exploration_rate: float = 0.1
    learning_rate: float = 0.01
    memory_capacity: int = 10000
    
    # Strategy discovery
    max_strategy_complexity: int = 3
    strategy_evaluation_period: int = 252  # Trading days
    min_strategy_performance: float = 0.05  # Minimum annualized return
    
    # Reinforcement learning
    reward_function: str = "risk_adjusted_return"  # or "sharpe_ratio", "sortino_ratio"
    discount_factor: float = 0.95
    experience_replay: bool = True
    
    # Adaptation
    adapt_to_market_regime: bool = True
    regime_detection_method: str = "hmm"  # or "clustering", "changepoint"
    adaptation_speed: float = 0.05

@dataclass
class AlphaPoolConfig:
    """Configuration for the entire Alpha Agent Pool."""
    
    # Pool-level settings
    pool_id: str = "alpha_agent_pool"
    version: str = "1.0.0"
    max_concurrent_agents: int = 5
    
    # Server configuration
    host: str = "127.0.0.1"
    port: int = 8081
    debug_mode: bool = False
    
    # Agent configurations
    momentum_agent: MomentumAgentConfig = field(default_factory=lambda: MomentumAgentConfig(
        agent_id="momentum_001",
        name="Primary Momentum Agent"
    ))
    
    mean_reversion_agent: MeanReversionAgentConfig = field(default_factory=lambda: MeanReversionAgentConfig(
        agent_id="mean_reversion_001", 
        name="Primary Mean Reversion Agent"
    ))
    
    empirical_agent: EmpiricalAgentConfig = field(default_factory=lambda: EmpiricalAgentConfig(
        agent_id="empirical_001",
        name="Data Mining Agent"
    ))
    
    autonomous_agent: AutonomousAgentConfig = field(default_factory=lambda: AutonomousAgentConfig(
        agent_id="autonomous_001",
        name="Self-Directed Agent" 
    ))
    
    # Memory and storage
    memory_config: Dict[str, Any] = field(default_factory=lambda: {
        "legacy_memory_url": "http://127.0.0.1:8000",
        "a2a_memory_url": "http://127.0.0.1:8002",
        "enable_persistent_storage": True,
        "storage_backend": "neo4j",
        "max_memory_entries": 100000
    })
    
    # Quality assurance
    qa_config: Dict[str, Any] = field(default_factory=lambda: {
        "enable_backtesting": True,
        "enable_performance_monitoring": True,
        "min_sharpe_ratio": 0.5,
        "max_drawdown": 0.2,
        "performance_check_frequency": "daily"
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif isinstance(value, dict):
                result[key] = value
            else:
                result[key] = value
        return result
    
    def save_to_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def save_to_json(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'AlphaPoolConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'AlphaPoolConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlphaPoolConfig':
        """Create configuration from dictionary."""
        # Convert nested configurations
        if 'momentum_agent' in data:
            data['momentum_agent'] = MomentumAgentConfig.from_dict(data['momentum_agent'])
        if 'mean_reversion_agent' in data:
            data['mean_reversion_agent'] = MeanReversionAgentConfig.from_dict(data['mean_reversion_agent'])
        if 'empirical_agent' in data:
            data['empirical_agent'] = EmpiricalAgentConfig.from_dict(data['empirical_agent'])
        if 'autonomous_agent' in data:
            data['autonomous_agent'] = AutonomousAgentConfig.from_dict(data['autonomous_agent'])
        
        return cls(**data)

# Factory functions for creating common configurations
def create_default_config() -> AlphaPoolConfig:
    """Create default alpha pool configuration."""
    return AlphaPoolConfig()

def create_research_config() -> AlphaPoolConfig:
    """Create configuration optimized for research."""
    config = AlphaPoolConfig()
    config.debug_mode = True
    config.momentum_agent.generate_cross_sectional = True
    config.momentum_agent.generate_time_series = True
    config.empirical_agent.hyperparameter_tuning = True
    config.autonomous_agent.exploration_rate = 0.2
    return config

def create_production_config() -> AlphaPoolConfig:
    """Create configuration optimized for production."""
    config = AlphaPoolConfig()
    config.debug_mode = False
    config.max_concurrent_agents = 10
    config.qa_config["enable_performance_monitoring"] = True
    config.qa_config["performance_check_frequency"] = "realtime"
    return config

# Configuration validation
def validate_config(config: AlphaPoolConfig) -> List[str]:
    """Validate configuration and return list of warnings/errors."""
    warnings = []
    
    # Check port conflicts
    if config.port == 8000 or config.port == 8002:
        warnings.append(f"Port {config.port} conflicts with memory services")
    
    # Check agent parameters
    if config.momentum_agent.max_position_size > 0.1:
        warnings.append("Momentum agent position size > 10% may be risky")
    
    if config.mean_reversion_agent.max_drawdown > 0.25:
        warnings.append("Mean reversion agent max drawdown > 25% is high")
    
    # Check memory configuration
    if not config.memory_config.get("enable_persistent_storage", False):
        warnings.append("Persistent storage disabled - results may be lost")
    
    return warnings

__all__ = [
    'AgentType',
    'FactorCategory', 
    'AgentConfig',
    'MomentumAgentConfig',
    'MeanReversionAgentConfig',
    'EmpiricalAgentConfig',
    'AutonomousAgentConfig',
    'AlphaPoolConfig',
    'create_default_config',
    'create_research_config',
    'create_production_config',
    'validate_config'
]
