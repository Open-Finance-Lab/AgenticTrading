from agent_pools.data_agent_pool.agents.equity.mcp_adapter import mcp

if __name__ == "__main__":
    # Use streamable-http transport for production-grade deployments
    mcp.run(transport="streamable-http")