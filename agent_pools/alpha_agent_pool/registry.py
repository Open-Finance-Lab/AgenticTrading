import logging
from typing import Dict, Type, Optional
from abc import ABC, abstractmethod
from .schema.agent_config import AlphaAgentConfig, AgentType
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class AlphaAgent(ABC):
    """Base class for all alpha generation agents"""
    
    def __init__(self, config: AlphaAgentConfig):
        self.config = config
        self.validate_config()
        
    def validate_config(self) -> None:
        """Validate agent configuration"""
        if not self.config.validate():
            raise ValueError(f"Invalid configuration for agent {self.config.agent_id}")
            
    @abstractmethod
    async def generate_alpha(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alpha signals from input data"""
        pass
    
    @abstractmethod
    async def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate generated signals"""
        pass

class AlphaAgentRegistry:
    """Registry for managing alpha generation agents"""
    
    def __init__(self):
        self._agents: Dict[str, Type[AlphaAgent]] = {}
        self._instances: Dict[str, AlphaAgent] = {}
        
    def register(self, agent_id: str, agent_class: Type[AlphaAgent]) -> None:
        """Register a new alpha agent class"""
        if agent_id in self._agents:
            raise ValueError(f"Agent {agent_id} already registered")
        self._agents[agent_id] = agent_class
        logger.info(f"Registered alpha agent: {agent_id}")
        
    def get_agent_class(self, agent_id: str) -> Optional[Type[AlphaAgent]]:
        """Get agent class by ID"""
        return self._agents.get(agent_id)
        
    def create_agent(self, config: AlphaAgentConfig) -> AlphaAgent:
        """Create a new agent instance"""
        agent_class = self.get_agent_class(config.agent_id)
        if not agent_class:
            raise ValueError(f"Agent {config.agent_id} not found in registry")
            
        agent = agent_class(config)
        self._instances[config.agent_id] = agent
        logger.info(f"Created instance of alpha agent: {config.agent_id}")
        return agent
        
    def get_agent_instance(self, agent_id: str) -> Optional[AlphaAgent]:
        """Get existing agent instance by ID"""
        return self._instances.get(agent_id)
        
    def list_agents(self) -> Dict[str, AgentType]:
        """List all registered agents"""
        return {
            agent_id: agent_class.config.agent_type
            for agent_id, agent_class in self._agents.items()
        }
        
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the registry"""
        if agent_id in self._agents:
            del self._agents[agent_id]
            if agent_id in self._instances:
                del self._instances[agent_id]
            logger.info(f"Removed alpha agent: {agent_id}")
            
# Global registry instance
alpha_registry = AlphaAgentRegistry() 