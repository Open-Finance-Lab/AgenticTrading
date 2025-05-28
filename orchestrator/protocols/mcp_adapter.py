"""
MCP (Model Context Protocol) FastMCP adapter implementation.
"""
from mcp.server.fastmcp import FastMCP, Context

# Create FastMCP server instance
mcp = FastMCP("FinAgent-MCP-Server")

# Example: Register a static resource
@mcp.resource("config://app")
def get_config() -> str:
    """Return static configuration info"""
    return "FinAgent Orchestration configuration data"

# Example: Register a dynamic resource
@mcp.resource("user://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    return f"Profile data for user {user_id}"

# Example: Register a tool (callable by LLM)
@mcp.tool()
def execute_task(task_name: str, arguments: dict) -> dict:
    """Basic interface for task execution (sync example)"""
    # TODO: Implement actual task scheduling logic
    print(f"[INFO] Executing task: {task_name} with arguments: {arguments}")
    return {"status": "success", "result": f"Executed {task_name}"}

# Example: Register an async tool
@mcp.tool()
async def async_tool_demo(param: str) -> str:
    print(f"[INFO] Async tool received param: {param}")
    return f"Async tool received param: {param}"

# Example: Register a prompt
@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review the following code:\n\n{code}"

# Server entry point
if __name__ == "__main__":
    print("[INFO] Starting FinAgent MCP Server...")
    mcp.run() 