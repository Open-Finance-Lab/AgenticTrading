from pydantic import BaseModel, Field
from typing import Optional, List

class StrategyConfig(BaseModel):
    name: str = Field(..., description="Strategy name, e.g. momentum")
    window: int = Field(..., description="Window length for signal computation")
    threshold: float = Field(..., description="Threshold for strategy signal activation")

class ExecutionConfig(BaseModel):
    port: int = Field(..., description="Port for MCP server to listen on")
    mode: str = Field("mcp_server", description="Execution mode")
    timeout: int = Field(20, description="Request timeout in seconds")

class AuthConfig(BaseModel):
    api_key: Optional[str] = None
    secret_key: Optional[str] = None

class AgentMetadata(BaseModel):
    author: Optional[str] = "Unknown"
    description: Optional[str] = None
    tags: Optional[List[str]] = []

class AlphaAgentConfig(BaseModel):
    agent_id: str
    strategy: StrategyConfig
    execution: ExecutionConfig
    authentication: AuthConfig
    metadata: Optional[AgentMetadata] = None
    llm_enabled: bool = False