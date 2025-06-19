from fastapi import FastAPI
from .mcp_server import mcp

app = FastAPI(
    title="Alpha Agent Pool",
    description="MCP server for alpha generation agents",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Alpha Agent Pool MCP server is running",
        "status": "active"
    }

# Mount MCP server
app.mount("/mcp", mcp.app) 