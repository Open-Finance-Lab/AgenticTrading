from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import yaml
import logging
from .agent_config import AlphaAgentConfig, AgentType, DataSource, SignalRule, RiskParameter

logger = logging.getLogger(__name__)

class CommonConfig(BaseModel):
    """Common configuration shared by all agents"""
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "min_confidence": 0.7,
            "max_position_size": 0.1,
            "max_drawdown": 0.05
        }
    )
    data_sources: List[DataSource] = Field(default_factory=list)

class AgentConfigSchema(BaseModel):
    """Schema for individual agent configuration"""
    agent_id: str
    agent_type: AgentType
    description: str
    data_sources: List[DataSource]
    parameters: Dict[str, Any]
    signal_rules: List[SignalRule]
    risk_parameters: List[RiskParameter]

class AlphaConfigManager:
    """Configuration manager for alpha agents"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the configuration manager
        
        Args:
            config_dir: Directory containing configuration files. If None, uses default path.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        
        self.config_dir = Path(config_dir)
        self._common_config: Optional[CommonConfig] = None
        self._agent_configs: Optional[Dict[str, AlphaAgentConfig]] = None
        
        # Define expected config files
        self.common_config_file = self.config_dir / "common.yaml"
        self.agent_config_files = {
            "technical_agent": self.config_dir / "technical_agent.yaml",
            "event_agent": self.config_dir / "event_agent.yaml",
            "ml_agent": self.config_dir / "ml_agent.yaml"
        }
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load a YAML configuration file
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Dict containing the configuration
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_common_config(self) -> CommonConfig:
        """
        Load the common configuration
        
        Returns:
            CommonConfig object containing common configuration
        """
        if self._common_config is None:
            config_data = self._load_yaml_file(self.common_config_file)
            self._common_config = CommonConfig(**config_data)
        return self._common_config
    
    def _merge_common_config(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge common configuration with agent-specific configuration
        
        Args:
            agent_config: Agent-specific configuration
            
        Returns:
            Merged configuration
        """
        common = self._load_common_config()
        
        # Merge parameters
        agent_config['parameters'] = {
            **common.parameters,
            **agent_config.get('parameters', {})
        }
        
        # Merge data sources
        agent_config['data_sources'] = (
            [ds.dict() for ds in common.data_sources] + 
            agent_config.get('data_sources', [])
        )
        
        return agent_config
    
    def get_agent_configs(self) -> Dict[str, AlphaAgentConfig]:
        """
        Get all agent configurations as AlphaAgentConfig objects
        
        Returns:
            Dictionary mapping agent_id to AlphaAgentConfig
        """
        if self._agent_configs is not None:
            return self._agent_configs
            
        self._agent_configs = {}
        
        # Process each agent configuration file
        for agent_id, config_file in self.agent_config_files.items():
            try:
                # Load agent-specific config
                agent_config = self._load_yaml_file(config_file)
                
                # Validate against schema
                agent_schema = AgentConfigSchema(**agent_config)
                
                # Merge with common config
                merged_config = self._merge_common_config(agent_config)
                
                # Convert to AlphaAgentConfig
                config = AlphaAgentConfig(
                    agent_id=merged_config['agent_id'],
                    agent_type=AgentType[merged_config['agent_type']],
                    description=merged_config['description'],
                    data_sources=[
                        DataSource(**ds) for ds in merged_config['data_sources']
                    ],
                    parameters=merged_config['parameters'],
                    signal_rules=[
                        SignalRule(**rule) for rule in merged_config['signal_rules']
                    ],
                    risk_parameters=[
                        RiskParameter(**param) for param in merged_config['risk_parameters']
                    ]
                )
                
                self._agent_configs[agent_id] = config
                
            except Exception as e:
                logger.error(f"Failed to load configuration for {agent_id}: {str(e)}")
                raise
            
        return self._agent_configs

# Create a singleton instance
config_manager = AlphaConfigManager() 