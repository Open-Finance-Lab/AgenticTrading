# config_loader.py

import os
import yaml
from schema.agent_config import AgentConfig
from pydantic import ValidationError



CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

def load_agent_config(agent_id: str) -> AgentConfig:
    config_path = os.path.join(CONFIG_DIR, f"{agent_id.replace('_agent', '')}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file for agent: {config_path}")
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    return AgentConfig(**raw)

try:
    config = load_agent_config("binance_agent")
except ValidationError as e:
    print(f"❌ Config validation failed: {e}")