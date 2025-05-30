from fastapi import FastAPI
from mcp_server import mcp  # <- your FastMCP instance

app = FastAPI()

@app.get("/")
def root():
    return {"message": "MCP root is live."}

# Mount MCP
app.mount("/mcp", mcp.streamable_http_app())