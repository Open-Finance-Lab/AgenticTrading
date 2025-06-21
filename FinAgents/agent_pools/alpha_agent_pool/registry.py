# registry.py

import logging
import os
import yaml
from fastapi import FastAPI
from typing import Dict, Callable, Any

from agent_pools.alpha_agent_pool.schema.theory_driven_schema import MomentumAgentConfig
from agent_pools.alpha_agent_pool.agent_service import MomentumAgentService

# === Base Interface for All Alpha Agents ===
class BaseAlphaAgent:
    def __init__(self, config: dict):
        self.config = config

    def execute(self, function_name: str, inputs: dict) -> Any:
        method: Callable = getattr(self, function_name, None)
        if not method:
            raise AttributeError(f"Function '{function_name}' not found in agent.")
        return method(**inputs)

# === Global Registry ===
ALPHA_AGENT_REGISTRY: Dict[str, BaseAlphaAgent] = {}
FASTAPI_SUBAPPS: Dict[str, FastAPI] = {}

# === Config Loader ===
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

def load_config(agent_id: str) -> dict:
    config_file = os.path.join(CONFIG_DIR, f"{agent_id.replace('_agent', '')}.yaml")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file for agent '{agent_id}' not found: {config_file}")
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

# === Agent Mounting ===
def mount_momentum_agent(app: FastAPI, agent_id: str):
    cfg_dict = load_config(agent_id)
    config = MomentumAgentConfig(**cfg_dict)
    service = MomentumAgentService.from_schema(config)
    ALPHA_AGENT_REGISTRY[agent_id] = service  # service implements BaseAlphaAgent-like interface
    FASTAPI_SUBAPPS[agent_id] = service.app
    app.mount(f"/{agent_id}", service.app)
    logging.info(f"Mounted {agent_id} at /{agent_id}")

# === Unified FastAPI App ===
app = FastAPI(title="Alpha Agent MCP Registry", version="1.0")

# === Register & Mount All Agents ===
def register_all_agents():
    # For now, only mount momentum_agent
    mount_momentum_agent(app, "momentum_agent")

register_all_agents()