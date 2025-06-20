
from pydantic import BaseModel
from typing import Optional, Dict

class APISettings(BaseModel):
    base_url: str
    endpoints: Dict[str, str]

class AuthSettings(BaseModel):
    api_key: str
    secret_key: Optional[str] = None

class Constraints(BaseModel):
    timeout: int
    rate_limit_per_minute: int

class NewsAPIConfig(BaseModel):
    agent_id: str
    api: APISettings
    authentication: AuthSettings
    constraints: Constraints

class RSSConfig(BaseModel):
    agent_id: str
    api: APISettings
    constraints: Constraints

class AlphaVantageConfig(BaseModel):
    agent_id: str
    api: APISettings
    authentication: AuthSettings
    constraints: Constraints