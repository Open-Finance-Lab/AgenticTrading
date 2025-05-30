from pydantic import BaseModel
from typing import Optional

class APIEndpoints(BaseModel):
    spot_price: Optional[str]
    ohlcv: Optional[str]

class APIConfig(BaseModel):
    base_url: str
    endpoints: APIEndpoints
    default_interval: Optional[str]

class AuthConfig(BaseModel):
    api_key: Optional[str]
    secret_key: Optional[str]

class ConstraintsConfig(BaseModel):
    timeout: int = 5
    rate_limit_per_minute: int = 60

class AgentConfig(BaseModel):
    agent_id: str
    api: APIConfig
    authentication: AuthConfig
    constraints: ConstraintsConfig
