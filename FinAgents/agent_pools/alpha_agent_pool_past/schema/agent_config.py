from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field

class AgentType(str, Enum):
    TECHNICAL = "TECHNICAL"
    FUNDAMENTAL = "FUNDAMENTAL"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    ML_BASED = "ML_BASED"

class DataSource(BaseModel):
    name: str
    type: str
    description: Optional[str] = None

class SignalRule(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class RiskParameter(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class AlphaAgentConfig(BaseModel):
    agent_id: str
    agent_type: AgentType
    description: str
    data_sources: List[DataSource]
    parameters: Dict[str, Any]
    signal_rules: List[SignalRule]
    risk_parameters: List[RiskParameter]

    def validate(self) -> bool:
        """Validate the configuration"""
        if not self.agent_id or not isinstance(self.agent_id, str):
            return False
        if not isinstance(self.agent_type, AgentType):
            return False
        if not self.parameters or not isinstance(self.parameters, dict):
            return False
        if not self.data_sources or not isinstance(self.data_sources, list):
            return False
        if not self.signal_rules or not isinstance(self.signal_rules, list):
            return False
        if not self.risk_parameters or not isinstance(self.risk_parameters, list):
            return False
        return True 