from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

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

# === Output Model (Full Strategy Flow) ===
class MarketContext(BaseModel):
    symbol: str
    regime_tag: Optional[str] = None
    input_features: Dict[str, Any]

class Decision(BaseModel):
    signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    reasoning: str
    predicted_return: float
    risk_estimate: float
    signal_type: Literal["directional", "allocation", "ranking"]
    asset_scope: List[str]

class Action(BaseModel):
    execution_weight: float
    order_type: Literal["market", "limit", "stop"]
    order_price: float
    execution_delay: str

class PerformanceFeedback(BaseModel):
    status: Literal["pending", "evaluated"]
    evaluation_link: Optional[str] = None

class Metadata(BaseModel):
    generator_agent: str
    strategy_prompt: str
    code_hash: str
    context_id: str

class AlphaStrategyFlow(BaseModel):
    alpha_id: str
    version: str
    timestamp: str
    market_context: MarketContext
    decision: Decision
    action: Action
    performance_feedback: PerformanceFeedback
    metadata: Metadata

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