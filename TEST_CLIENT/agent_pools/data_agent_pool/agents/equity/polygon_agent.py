from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel
import asyncio

polygon_mcp_server = MultiServerMCPClient(
{
    "polygon_mcp": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http",
    }
}
)

async def create_tools():
    tools = await polygon_mcp_server.get_tools()
    return tools

class ResponseFormat(BaseModel):
    """Response format for the agent."""
    result: str

tools = asyncio.run(create_tools())

agent = create_react_agent(
    ChatOpenAI(model="o4-mini"),
    tools=tools,
    response_format=ResponseFormat,
)