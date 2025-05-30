
import yaml
import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

def load_agent_config(agent_id):
    config_path = os.path.join(CONFIG_DIR, f"{agent_id.replace('_agent', '')}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file for agent '{agent_id}' not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Example:
# config = load_agent_config("binance_agent")
# print(config["api"]["base_url"])
