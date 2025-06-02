from enum import Enum
from typing import Dict, Any, List
from dataclasses import dataclass, field

class TaskStatus(Enum):
    """Enumeration of task statuses"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentType(Enum):
    """Enumeration of agent types"""
    DATA = "data_agent"
    ALPHA = "alpha_agent"
    RISK = "risk_agent"
    TRANSACTION = "transaction_agent"
    PORTFOLIO = "portfolio_agent"
    EXECUTION = "execution_agent"
    ATTRIBUTION = "attribution_agent"
    BACKTEST = "backtest_agent"

@dataclass
class TaskDefinition:
    """Task definition class
    
    Defines the basic properties and configuration of a task.
    """
    name: str
    description: str
    agent_type: AgentType
    required_parameters: List[str] = field(default_factory=list)
    optional_parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 300  # Default timeout in seconds
    retry_count: int = 3  # Default number of retries
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate if task parameters meet requirements
        
        Args:
            parameters: Dictionary of task parameters
            
        Returns:
            bool: Whether parameters are valid
        """
        # Check required parameters
        for param in self.required_parameters:
            if param not in parameters:
                return False
        return True

@dataclass
class TaskResult:
    """Task execution result class"""
    task_id: str
    status: TaskStatus
    output: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    execution_time: float = 0.0
    
    @property
    def is_success(self) -> bool:
        """Whether task executed successfully"""
        return self.status == TaskStatus.COMPLETED
    
    @property
    def has_error(self) -> bool:
        """Whether task encountered an error"""
        return bool(self.error) 