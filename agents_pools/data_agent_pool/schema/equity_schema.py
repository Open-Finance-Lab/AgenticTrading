
from pydantic import BaseModel
from typing import Optional, Dict

class APISettings(BaseModel):
    base_url: str
    endpoints: Dict[str, str]
    default_interval: str

class AuthSettings(BaseModel):
    api_key: str
    secret_key: str

class Constraints(BaseModel):
    timeout: int
    rate_limit_per_minute: int

class AlpacaConfig(BaseModel):
    agent_id: str
    api: APISettings
    authentication: AuthSettings
    constraints: Constraints

class IEXConfig(BaseModel):
    agent_id: str
    api: APISettings
    authentication: AuthSettings
    constraints: Constraints
