"""
FinAgent Configuration Manager

Centralized configuration management system for different deployment scenarios.
This manager handles configuration for development, testing, staging, and production
environments across all FinAgent memory components.

Features:
- Environment-specific configurations
- Server-specific settings (MCP, A2A, Memory)
- Database configuration management
- Deployment scenario handling
- Configuration validation and defaults

Author: FinAgent Team
License: Open Source
"""

# ═══════════════════════════════════════════════════════════════════════════════════
# IMPORTS AND DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════════

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION ENUMS AND DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════════

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class ServerType(Enum):
    """Types of FinAgent servers"""
    MEMORY = "memory"
    MCP = "mcp" 
    A2A = "a2a"
    ALL = "all"

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "finagent123"
    database: str = "finagent"
    max_connection_pool_size: int = 100
    connection_timeout: int = 30
    max_retry_time: int = 30
    
    # Neo4j specific settings
    encrypted: bool = False
    trust: str = "TRUST_ALL_CERTIFICATES"
    
    # Memory and indexing settings
    enable_intelligent_indexing: bool = True
    memory_cache_size: int = 1000
    semantic_search_enabled: bool = True

@dataclass 
class ServerConfig:
    """Server configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"
    cors_enabled: bool = True
    cors_origins: List[str] = None
    
    # Performance settings
    workers: int = 1
    max_connections: int = 1000
    keepalive_timeout: int = 5
    
    # Security settings
    api_key_required: bool = False
    api_key: Optional[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]

@dataclass
class MemoryConfig:
    """Memory system configuration"""
    graph_memory_enabled: bool = True
    vector_memory_enabled: bool = True
    cache_enabled: bool = True
    
    # Graph memory settings
    max_nodes: int = 10000
    max_relationships: int = 50000
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 24
    
    # Vector memory settings  
    vector_dimension: int = 384
    similarity_threshold: float = 0.7
    max_vectors: int = 10000
    
    # Intelligent indexing
    indexing_enabled: bool = True
    auto_index_creation: bool = True
    index_update_interval: int = 300  # seconds

@dataclass
class MCPConfig:
    """MCP server specific configuration"""
    transport_type: str = "stdio"
    protocol_version: str = "2024-11-05"
    
    # Tool configuration
    enable_all_tools: bool = True
    enabled_tools: List[str] = None
    tool_timeout: int = 30
    
    # Performance settings
    max_concurrent_operations: int = 10
    operation_timeout: int = 60
    
    def __post_init__(self):
        if self.enabled_tools is None:
            self.enabled_tools = []

@dataclass
class A2AConfig:
    """A2A server specific configuration"""
    websocket_enabled: bool = True
    real_time_enabled: bool = True
    
    # Message settings
    max_message_history: int = 1000
    message_retention_hours: int = 24
    broadcast_enabled: bool = True
    
    # Agent management
    max_concurrent_agents: int = 100
    agent_timeout_minutes: int = 30
    heartbeat_interval: int = 30

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = False
    file_path: str = "logs/finagent.log"
    max_file_size: str = "10MB"
    backup_count: int = 5
    
    # Component-specific logging
    database_log_level: str = "WARNING"
    server_log_level: str = "INFO"
    memory_log_level: str = "INFO"

@dataclass
class PortConfig:
    """Port configuration for different servers"""
    memory_server: int = 8000
    mcp_server: int = 8001
    a2a_server: int = 8002

@dataclass
class DatabaseInitConfig:
    """Database initialization configuration"""
    auto_create_indexes: bool = True
    auto_create_constraints: bool = True
    clear_on_startup: bool = False
    seed_data_enabled: bool = False
    backup_before_init: bool = True
    required_indexes: List[Dict[str, Any]] = None
    required_constraints: List[Dict[str, Any]] = None
    seed_data: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.required_indexes is None:
            self.required_indexes = []
        if self.required_constraints is None:
            self.required_constraints = []
        if self.seed_data is None:
            self.seed_data = []

@dataclass
class FinAgentConfig:
    """Complete FinAgent configuration"""
    environment: Environment = Environment.DEVELOPMENT
    
    # Component configurations
    database: DatabaseConfig = None
    server: ServerConfig = None
    memory: MemoryConfig = None
    mcp: MCPConfig = None
    a2a: A2AConfig = None
    logging: LoggingConfig = None
    ports: PortConfig = None
    database_init: DatabaseInitConfig = None
    
    # Global settings
    project_name: str = "FinAgent"
    version: str = "2.0.0"
    api_prefix: str = "/api/v1"
    
    def __post_init__(self):
        # Initialize sub-configurations if not provided
        if self.database is None:
            self.database = DatabaseConfig()
        if self.server is None:
            self.server = ServerConfig()
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.mcp is None:
            self.mcp = MCPConfig()
        if self.a2a is None:
            self.a2a = A2AConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.ports is None:
            self.ports = PortConfig()
        if self.database_init is None:
            self.database_init = DatabaseInitConfig()

# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION MANAGER IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════════

class ConfigurationManager:
    """
    Centralized configuration management for FinAgent deployment scenarios.
    Handles loading, validation, and environment-specific configurations.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        self._configs: Dict[Environment, FinAgentConfig] = {}
        self._current_environment = Environment.DEVELOPMENT
        self._load_configurations()
    
    def _load_configurations(self):
        """Load configurations for all environments."""
        try:
            # Load default configuration
            default_config = self._get_default_config()
            
            # Load environment-specific configurations
            for env in Environment:
                config_file = self.config_dir / f"{env.value}.yaml"
                
                if config_file.exists():
                    env_config = self._load_config_file(config_file, default_config)
                else:
                    env_config = self._get_environment_defaults(env, default_config)
                
                self._configs[env] = env_config
                
        except Exception as e:
            print(f"⚠️ Configuration loading failed: {e}")
            # Fall back to default configurations
            self._load_default_configurations()
    
    def _load_default_configurations(self):
        """Load default configurations for all environments."""
        default_config = self._get_default_config()
        
        for env in Environment:
            self._configs[env] = self._get_environment_defaults(env, default_config)
    
    def _get_default_config(self) -> FinAgentConfig:
        """Get base default configuration."""
        return FinAgentConfig(
            environment=Environment.DEVELOPMENT,
            database=DatabaseConfig(),
            server=ServerConfig(),
            memory=MemoryConfig(),
            mcp=MCPConfig(),
            a2a=A2AConfig(),
            logging=LoggingConfig()
        )
    
    def _get_environment_defaults(self, env: Environment, base_config: FinAgentConfig) -> FinAgentConfig:
        """Get environment-specific default configuration."""
        config = FinAgentConfig(
            environment=env,
            database=DatabaseConfig(),
            server=ServerConfig(),
            memory=MemoryConfig(),
            mcp=MCPConfig(),
            a2a=A2AConfig(),
            logging=LoggingConfig(),
            ports=PortConfig(),
            database_init=DatabaseInitConfig()
        )
        
        # Environment-specific adjustments
        if env == Environment.DEVELOPMENT:
            config.server.port = 8000
            config.server.debug = True
            config.logging.level = "DEBUG"
            config.logging.file_enabled = True
            config.database.max_connection_pool_size = 10
            config.ports.memory_server = 8000
            config.ports.mcp_server = 8001
            config.ports.a2a_server = 8002
            
        elif env == Environment.TESTING:
            config.server.port = 8001
            config.server.debug = True
            config.logging.level = "INFO"
            config.database.database = "finagent_test"
            config.memory.max_nodes = 1000
            config.memory.max_relationships = 5000
            config.ports.memory_server = 8001
            config.ports.mcp_server = 8011
            config.ports.a2a_server = 8021
            config.database_init.clear_on_startup = True
            config.database_init.seed_data_enabled = True
            
        elif env == Environment.STAGING:
            config.server.port = 8000
            config.server.debug = False
            config.logging.level = "INFO"
            config.logging.file_enabled = True
            config.database.max_connection_pool_size = 50
            config.server.api_key_required = True
            config.ports.memory_server = 8000
            config.ports.mcp_server = 8001
            config.ports.a2a_server = 8002
            
        elif env == Environment.PRODUCTION:
            config.server.port = 8000
            config.server.debug = False
            config.logging.level = "WARNING"
            config.logging.file_enabled = True
            config.database.max_connection_pool_size = 100
            config.server.api_key_required = True
            config.server.cors_origins = []  # Restrict CORS in production
            config.server.workers = 4
            config.database.encrypted = True
            config.database.trust = "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"
            config.ports.memory_server = 8000
            config.ports.mcp_server = 8001
            config.ports.a2a_server = 8002
            
        return config
    
    def _load_config_file(self, config_file: Path, base_config: FinAgentConfig) -> FinAgentConfig:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Merge with base configuration
            return self._merge_configs(base_config, config_data)
            
        except Exception as e:
            print(f"⚠️ Failed to load config file {config_file}: {e}")
            return base_config
    
    def _merge_configs(self, base_config: FinAgentConfig, config_data: Dict[str, Any]) -> FinAgentConfig:
        """Merge configuration data with base configuration."""
        try:
            # Convert base config to dict
            config_dict = asdict(base_config)
            
            # Recursively merge configuration data
            self._deep_merge(config_dict, config_data)
            
            # Convert back to FinAgentConfig
            return self._dict_to_config(config_dict)
            
        except Exception as e:
            print(f"⚠️ Config merge failed: {e}")
            return base_config
    
    def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Recursively merge two dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> FinAgentConfig:
        """Convert dictionary to FinAgentConfig object."""
        # Convert environment string to enum
        if 'environment' in config_dict and isinstance(config_dict['environment'], str):
            config_dict['environment'] = Environment(config_dict['environment'])
        
        # Convert sub-configs
        if 'database' in config_dict:
            config_dict['database'] = DatabaseConfig(**config_dict['database'])
        if 'server' in config_dict:
            config_dict['server'] = ServerConfig(**config_dict['server'])
        if 'memory' in config_dict:
            config_dict['memory'] = MemoryConfig(**config_dict['memory'])
        if 'mcp' in config_dict:
            config_dict['mcp'] = MCPConfig(**config_dict['mcp'])
        if 'a2a' in config_dict:
            config_dict['a2a'] = A2AConfig(**config_dict['a2a'])
        if 'logging' in config_dict:
            config_dict['logging'] = LoggingConfig(**config_dict['logging'])
        if 'ports' in config_dict:
            config_dict['ports'] = PortConfig(**config_dict['ports'])
        if 'database_init' in config_dict:
            config_dict['database_init'] = DatabaseInitConfig(**config_dict['database_init'])
        
        return FinAgentConfig(**config_dict)
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PUBLIC INTERFACE METHODS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def set_environment(self, environment: Union[Environment, str]):
        """Set the current environment."""
        if isinstance(environment, str):
            environment = Environment(environment)
        
        if environment in self._configs:
            self._current_environment = environment
        else:
            raise ValueError(f"Environment {environment} not configured")
    
    def get_config(self, environment: Optional[Environment] = None) -> FinAgentConfig:
        """Get configuration for specified environment or current environment."""
        env = environment or self._current_environment
        return self._configs.get(env, self._get_default_config())
    
    def get_database_config(self, environment: Optional[Environment] = None) -> DatabaseConfig:
        """Get database configuration."""
        return self.get_config(environment).database
    
    def get_server_config(self, server_type: ServerType, environment: Optional[Environment] = None) -> ServerConfig:
        """Get server configuration for specific server type."""
        config = self.get_config(environment)
        base_config = config.server
        
        # Adjust port based on server type using ports configuration
        if server_type == ServerType.MEMORY:
            base_config.port = config.ports.memory_server
        elif server_type == ServerType.MCP:
            base_config.port = config.ports.mcp_server
        elif server_type == ServerType.A2A:
            base_config.port = config.ports.a2a_server
        
        return base_config
    
    def get_port_config(self, environment: Optional[Environment] = None) -> PortConfig:
        """Get port configuration."""
        return self.get_config(environment).ports
    
    def get_database_init_config(self, environment: Optional[Environment] = None) -> DatabaseInitConfig:
        """Get database initialization configuration."""
        return self.get_config(environment).database_init
    
    def get_memory_config(self, environment: Optional[Environment] = None) -> MemoryConfig:
        """Get memory configuration."""
        return self.get_config(environment).memory
    
    def get_mcp_config(self, environment: Optional[Environment] = None) -> MCPConfig:
        """Get MCP configuration."""
        return self.get_config(environment).mcp
    
    def get_a2a_config(self, environment: Optional[Environment] = None) -> A2AConfig:
        """Get A2A configuration."""
        return self.get_config(environment).a2a
    
    def get_logging_config(self, environment: Optional[Environment] = None) -> LoggingConfig:
        """Get logging configuration."""
        return self.get_config(environment).logging
    
    def save_config(self, environment: Environment, config: FinAgentConfig):
        """Save configuration to file."""
        try:
            config_file = self.config_dir / f"{environment.value}.yaml"
            config_dict = asdict(config)
            
            # Convert enums to strings for YAML serialization
            config_dict['environment'] = config.environment.value
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            # Update in-memory configuration
            self._configs[environment] = config
            
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration: {e}")
    
    def export_config(self, environment: Environment, format: str = "yaml") -> str:
        """Export configuration in specified format."""
        config = self.get_config(environment)
        config_dict = asdict(config)
        config_dict['environment'] = config.environment.value
        
        if format.lower() == "yaml":
            return yaml.dump(config_dict, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            return json.dumps(config_dict, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def validate_config(self, config: FinAgentConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate database configuration
        if not config.database.uri:
            issues.append("Database URI is required")
        if not config.database.username:
            issues.append("Database username is required")
        if not config.database.password:
            issues.append("Database password is required")
        
        # Validate server configuration  
        if config.server.port < 1 or config.server.port > 65535:
            issues.append("Server port must be between 1 and 65535")
        
        # Validate memory configuration
        if config.memory.max_nodes <= 0:
            issues.append("Max nodes must be positive")
        if config.memory.max_relationships <= 0:
            issues.append("Max relationships must be positive")
        
        return issues
    
    def list_environments(self) -> List[Environment]:
        """List all configured environments."""
        return list(self._configs.keys())
    
    def get_current_environment(self) -> Environment:
        """Get current active environment."""
        return self._current_environment

# ═══════════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS FOR EASY ACCESS
# ═══════════════════════════════════════════════════════════════════════════════════

# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None

def get_config_manager(config_dir: Optional[str] = None) -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_dir)
    return _config_manager

def get_database_config(environment: Optional[Environment] = None) -> Dict[str, Any]:
    """Get database configuration as dictionary."""
    config = get_config_manager().get_database_config(environment)
    return {
        "uri": config.uri,
        "username": config.username,
        "password": config.password,
        "database": config.database
    }

def get_server_config_dict(server_type: ServerType, environment: Optional[Environment] = None) -> Dict[str, Any]:
    """Get server configuration as dictionary."""
    config = get_config_manager().get_server_config(server_type, environment)
    return asdict(config)

def create_default_configs():
    """Create default configuration files for all environments."""
    config_manager = get_config_manager()
    
    for env in Environment:
        try:
            config_file = config_manager.config_dir / f"{env.value}.yaml"
            if not config_file.exists():
                config = config_manager.get_config(env)
                config_manager.save_config(env, config)
                print(f"✅ Created default config for {env.value}")
        except Exception as e:
            print(f"❌ Failed to create config for {env.value}: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT DETECTION AND AUTO-CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════

def detect_environment() -> Environment:
    """Detect current environment from environment variables."""
    env_var = os.getenv("FINAGENT_ENV", "development").lower()
    
    env_mapping = {
        "dev": Environment.DEVELOPMENT,
        "development": Environment.DEVELOPMENT,
        "test": Environment.TESTING,
        "testing": Environment.TESTING,
        "stage": Environment.STAGING,
        "staging": Environment.STAGING,
        "prod": Environment.PRODUCTION,
        "production": Environment.PRODUCTION
    }
    
    return env_mapping.get(env_var, Environment.DEVELOPMENT)

def auto_configure() -> FinAgentConfig:
    """Auto-configure based on detected environment."""
    environment = detect_environment()
    config_manager = get_config_manager()
    config_manager.set_environment(environment)
    
    print(f"🔧 Auto-configured for {environment.value} environment")
    return config_manager.get_config()

# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════════

def print_config_summary(environment: Optional[Environment] = None):
    """Print configuration summary."""
    config_manager = get_config_manager()
    config = config_manager.get_config(environment)
    
    print("\n" + "="*80)
    print("🔧 FINAGENT CONFIGURATION SUMMARY")
    print("="*80)
    print(f"📋 Environment: {config.environment.value}")
    print(f"🏷️  Project: {config.project_name} v{config.version}")
    print("\n🗄️  Database Configuration:")
    print(f"   📍 URI: {config.database.uri}")
    print(f"   👤 Username: {config.database.username}")
    print(f"   🗄️  Database: {config.database.database}")
    print(f"   🔗 Pool Size: {config.database.max_connection_pool_size}")
    print(f"   🧠 Intelligent Indexing: {'✅' if config.database.enable_intelligent_indexing else '❌'}")
    print("\n🌐 Server Configuration:")
    print(f"   🏠 Host: {config.server.host}")
    print(f"   🚪 Port: {config.server.port}")
    print(f"   🐛 Debug: {'✅' if config.server.debug else '❌'}")
    print(f"   📝 Log Level: {config.logging.level}")
    print(f"   🔐 API Key Required: {'✅' if config.server.api_key_required else '❌'}")
    print("\n💾 Memory Configuration:")
    print(f"   📊 Max Nodes: {config.memory.max_nodes:,}")
    print(f"   🔗 Max Relationships: {config.memory.max_relationships:,}")
    print(f"   🧠 Vector Memory: {'✅' if config.memory.vector_memory_enabled else '❌'}")
    print(f"   🔍 Semantic Search: {'✅' if config.database.semantic_search_enabled else '❌'}")
    print("\n📡 Protocol Configuration:")
    print(f"   🔧 MCP Transport: {config.mcp.transport_type}")
    print(f"   🌐 A2A WebSocket: {'✅' if config.a2a.websocket_enabled else '❌'}")
    print(f"   📨 Max Message History: {config.a2a.max_message_history}")
    print("="*80)

if __name__ == "__main__":
    # Example usage
    print("🔧 FinAgent Configuration Manager")
    
    # Create default configs
    create_default_configs()
    
    # Auto-configure
    config = auto_configure()
    
    # Print summary
    print_config_summary()
