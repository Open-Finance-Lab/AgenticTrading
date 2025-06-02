# mcp_server.py

from datetime import datetime
from mcp.server.fastmcp import FastMCP
from registry import AGENT_REGISTRY, preload_default_agents

# Preload all agents at startup
preload_default_agents()

# Create a FastMCP instance (stateless HTTP for streamable transport)
mcp = FastMCP("Data Agent Pool", stateless_http=True)

# -----------------------------------------------------------------------------
# Tool: agent.execute
# Executes a registered agent's function and returns its result
# -----------------------------------------------------------------------------
@mcp.tool(
    name="agent.execute",
    description="Execute a function on a registered agent"
)
def agent_execute(agent_id: str, function: str, input: dict) -> dict:
    agent = AGENT_REGISTRY.get(agent_id)
    if not agent:
        raise ValueError(f"Unknown agent_id: {agent_id}")
    try:
        result = agent.execute(function, input)
        return {"result": result, "meta": {"status": "success"}}
    except Exception as e:
        raise RuntimeError(f"Execution failed for {agent_id}.{function}: {str(e)}")

# -----------------------------------------------------------------------------
# Resource: register://{agent_id}
# Dynamically registers a new agent; returns registration info
# -----------------------------------------------------------------------------
@mcp.resource("register://{agent_id}")
def register_agent(agent_id: str) -> dict:
    timestamp = datetime.utcnow().isoformat()
    AGENT_REGISTRY[agent_id] = {
        "tools": ["example"],
        "registered_at": timestamp
    }
    # Log to console for visibility
    print(f"[REGISTER] agent_id={agent_id} at {timestamp}")
    return {
        "message":      f"Agent '{agent_id}' registered",
        "registered_at": timestamp
    }

# -----------------------------------------------------------------------------
# Resource: heartbeat://{agent_id}
# Records a heartbeat for a given agent; returns last heartbeat time
# -----------------------------------------------------------------------------
@mcp.resource("heartbeat://{agent_id}")
def heartbeat(agent_id: str) -> dict:
    if agent_id not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent_id: {agent_id}")
    timestamp = datetime.utcnow().isoformat()
    AGENT_REGISTRY[agent_id]["last_heartbeat"] = timestamp
    print(f"[HEARTBEAT] agent_id={agent_id} at {timestamp}")
    return {
        "message":         f"Heartbeat received for '{agent_id}'",
        "last_heartbeat":  timestamp
    }

# -----------------------------------------------------------------------------
# Export the ASGI app for streamable HTTP transport
# -----------------------------------------------------------------------------
app = mcp.streamable_http_app()