"""
Base classes for data agents.

This module contains the base interfaces and common functionality for all data agents,
separated from registry to avoid circular imports.
"""
from typing import Dict, Any, Callable
import logging


class BaseAgent:
    """
    Base interface for all data agents in the FinAgent ecosystem.
    
    Provides common functionality:
    - Configuration management
    - Dynamic method execution
    - Logging setup
    """
    
    def __init__(self, config: dict):
        """
        Initialize base agent with configuration.
        
        Args:
            config: Agent configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(self, function_name: str, inputs: dict) -> Any:
        """
        Dispatch execution to agent methods dynamically.
        
        Args:
            function_name: Name of the method to call
            inputs: Dictionary of arguments to pass to the method
            
        Returns:
            Result of the method execution
            
        Raises:
            AttributeError: If the method doesn't exist
        """
        method: Callable = getattr(self, function_name, None)
        if not method:
            raise AttributeError(f"Function '{function_name}' not found in agent {self.__class__.__name__}.")
        return method(**inputs)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get basic information about the agent.
        
        Returns:
            dict: Agent information including class name and config
        """
        return {
            "agent_class": self.__class__.__name__,
            "config": self.config
        }
