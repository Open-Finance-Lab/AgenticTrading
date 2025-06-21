# agent_pools/alpha_agent_pool/schema/theory_driven_schema.py

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# === Input Model ===
class MomentumSignalRequest(BaseModel):
    """
    Request model for generating a momentum trading signal.
    Attributes:
        symbol (str): The stock or asset symbol for which the signal is requested.
        price_list (Optional[List[float]]): Optional list of historical prices. If not provided, the agent will generate synthetic data.
    """
    symbol: str
    price_list: Optional[List[float]] = None

# === Output Model ===
class MomentumSignal(BaseModel):
    """
    Response model representing the generated momentum trading signal.
    Attributes:
        symbol (str): The stock or asset symbol.
        score (float): The confidence score of the signal.
        signal (Literal["buy", "sell", "hold"]): The trading action recommended by the agent.
        momentum (float): The calculated momentum value.
    """
    symbol: str
    score: float
    signal: Literal["buy", "sell", "hold"]
    momentum: float

# === Strategy Configuration Model ===
class StrategyConfig(BaseModel):
    """
    Configuration model for the momentum trading strategy.
    Attributes:
        window (int): Number of past periods to evaluate for momentum calculation.
        threshold (float): Threshold value for detecting significant momentum.
    """
    window: int = Field(default=10, description="Number of past periods to evaluate")
    threshold: float = Field(default=0.02, description="Threshold for detecting momentum")

# === Agent Execution Configuration Model ===
class ExecutionConfig(BaseModel):
    """
    Configuration model for agent execution parameters.
    Attributes:
        port (int): Port number to run the MCP server on.
    """
    port: int = Field(default=5050, description="Port to run the MCP server")

class MomentumAgentConfig(BaseModel):
    """
    Aggregated configuration model for the MomentumAgent, including strategy and execution settings.
    Attributes:
        agent_id (str): Unique identifier for the agent instance.
        strategy (StrategyConfig): Strategy configuration parameters.
        execution (ExecutionConfig): Execution configuration parameters.
    """
    agent_id: str
    strategy: StrategyConfig
    execution: ExecutionConfig