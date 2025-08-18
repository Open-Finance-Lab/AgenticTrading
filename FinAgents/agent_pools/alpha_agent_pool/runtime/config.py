"""Configuration management for Alpha Agent Pool."""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry policies."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0


@dataclass
class BackpressureConfig:
    """Configuration for backpressure policies."""
    max_queue_size: int = 100
    warning_threshold: float = 0.8
    rate_limit_rps: float = 10.0


@dataclass
class ObservabilityConfig:
    """Configuration for observability features."""
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_detailed_logging: bool = False
    metrics_port: int = 9090
    log_level: str = "INFO"


@dataclass
class StrategyConfig:
    """Configuration for strategy execution."""
    default_timeout: float = 30.0
    warmup_enabled: bool = True
    max_concurrent_strategies: int = 10


@dataclass
class FeatureConfig:
    """Configuration for feature retrieval."""
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000
    fetch_timeout: float = 15.0


@dataclass
class MemoryConfig:
    """Configuration for A2A memory coordination."""
    memory_url: str = "http://127.0.0.1:8010"
    connection_timeout: float = 10.0
    retry_attempts: int = 3


@dataclass
class McpConfig:
    """Configuration for MCP server/client."""
    server_host: str = "0.0.0.0"
    server_port: int = 8081
    client_timeout: float = 30.0
    transport: str = "stdio"


@dataclass
class AlphaPoolConfig:
    """Main configuration for Alpha Agent Pool."""
    # Core settings
    pool_id: str = "alpha_pool_default"
    environment: str = "development"
    
    # Component configurations
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    backpressure: BackpressureConfig = field(default_factory=BackpressureConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    
    # Runtime settings
    enable_enhanced_lifecycle: bool = True
    worker_threads: int = 4
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.retry.max_attempts < 1:
            raise ValueError("retry.max_attempts must be >= 1")
        
        if self.circuit_breaker.failure_threshold < 1:
            raise ValueError("circuit_breaker.failure_threshold must be >= 1")
        
        if self.backpressure.max_queue_size < 1:
            raise ValueError("backpressure.max_queue_size must be >= 1")
        
        if self.mcp.server_port < 1 or self.mcp.server_port > 65535:
            raise ValueError("mcp.server_port must be between 1 and 65535")
        
        if self.worker_threads < 1:
            raise ValueError("worker_threads must be >= 1")


class ConfigLoader:
    """Base class for configuration loaders."""
    
    def load(self, config_class: Type[T]) -> T:
        """Load configuration of specified type."""
        raise NotImplementedError


class EnvironmentConfigLoader(ConfigLoader):
    """Load configuration from environment variables."""
    
    def __init__(self, prefix: str = "ALPHA_POOL_"):
        self.prefix = prefix
    
    def load(self, config_class: Type[T]) -> T:
        """Load configuration from environment variables."""
        # This is a simplified implementation
        # In practice, you'd use a library like pydantic or create a more sophisticated mapper
        env_values = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                config_key = key[len(self.prefix):].lower()
                env_values[config_key] = value
        
        # Create default instance and update with env values
        config = config_class()
        
        # Simple mapping for common fields
        if hasattr(config, 'pool_id') and 'pool_id' in env_values:
            config.pool_id = env_values['pool_id']
        
        if hasattr(config, 'environment') and 'environment' in env_values:
            config.environment = env_values['environment']
        
        return config


class YamlConfigLoader(ConfigLoader):
    """Load configuration from YAML file."""
    
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
    
    def load(self, config_class: Type[T]) -> T:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return self._dict_to_dataclass(config_data, config_class)
    
    def _dict_to_dataclass(self, data: Dict[str, Any], config_class: Type[T]) -> T:
        """Convert dictionary to dataclass instance."""
        # This is a simplified implementation
        # In practice, you'd use a library like dacite or cattrs for proper conversion
        
        if not data:
            return config_class()
        
        # Get field types from dataclass
        import dataclasses
        
        if not dataclasses.is_dataclass(config_class):
            return config_class(**data)
        
        field_types = {f.name: f.type for f in dataclasses.fields(config_class)}
        converted_data = {}
        
        for key, value in data.items():
            if key in field_types:
                field_type = field_types[key]
                
                # Handle nested dataclasses
                if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
                    converted_data[key] = self._dict_to_dataclass(value, field_type)
                else:
                    converted_data[key] = value
        
        return config_class(**converted_data)


class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, loader: ConfigLoader):
        self.loader = loader
        self._config: Optional[AlphaPoolConfig] = None
    
    def load_config(self) -> AlphaPoolConfig:
        """Load and validate configuration."""
        self._config = self.loader.load(AlphaPoolConfig)
        self._config.validate()
        return self._config
    
    @property
    def config(self) -> AlphaPoolConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload_config(self) -> AlphaPoolConfig:
        """Reload configuration from source."""
        self._config = None
        return self.load_config()


def create_config_manager(
    config_path: Optional[Union[str, Path]] = None,
    use_env: bool = True
) -> ConfigManager:
    """Create a configuration manager with appropriate loader.
    
    Args:
        config_path: Path to YAML config file (optional)
        use_env: Whether to use environment variables
        
    Returns:
        Configured ConfigManager instance
    """
    if config_path:
        loader = YamlConfigLoader(config_path)
    elif use_env:
        loader = EnvironmentConfigLoader()
    else:
        # Use default configuration
        class DefaultLoader(ConfigLoader):
            def load(self, config_class: Type[T]) -> T:
                return config_class()
        
        loader = DefaultLoader()
    
    return ConfigManager(loader)
