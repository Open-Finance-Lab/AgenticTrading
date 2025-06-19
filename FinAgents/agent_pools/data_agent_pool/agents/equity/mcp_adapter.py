from mcp.server.fastmcp import FastMCP

class MCPAdapter:
    """
    Adapter to expose an agent as an MCP server.
    Automatically registers agent tools as MCP endpoints.
    """
    def __init__(self, agent, name=None, stateless_http=True):
        self.agent = agent
        self.mcp = FastMCP(name or f"{agent.__class__.__name__}-MCP", stateless_http=stateless_http)
        self._register_tools()

    def _register_tools(self):
        @self.mcp.tool(description="Health check for MCP server")
        def health_check() -> dict:
            import logging
            logging.info("health_check called")
            return {"status": "ok"}
        
        # Register fetch_market_data
        @self.mcp.tool(description="Fetch historical market data")
        def fetch_market_data(symbol: str, start: str, end: str, interval: str = "1d", force_refresh: bool = False) -> dict:
            """
            Retrieve historical OHLCV market data for a specified symbol and date range.
            This tool is suitable for quantitative analysis, backtesting, and reporting.
            All parameters must be provided in ISO format (YYYY-MM-DD).
            """
            df = self.agent.fetch(symbol=symbol, start=start, end=end, interval=interval, force_refresh=force_refresh)
            return df.reset_index().to_dict(orient="records")

        # Register analyze_company
        @self.mcp.tool(description="Get company information")
        def analyze_company(symbol: str) -> dict:
            """
            Retrieve comprehensive company information for the specified stock symbol.
            This includes business profile, sector, industry, and key financial metrics.
            """
            return self.agent.get_company_info(symbol)

        # Register identify_leaders
        @self.mcp.tool(description="Get top tickers")
        def identify_leaders(n: int = 5) -> list:
            """
            Identify the top N tickers based on trading volume and volatility.
            This tool is useful for market screening and leader board generation.
            """
            return self.agent.get_top_tickers(n)

        # Register process_intent if available
        if hasattr(self.agent, "process_intent"):
            @self.mcp.tool(description="Process natural language intent")
            async def process_intent(query: str) -> dict:
                """
                Process a natural language query and return a structured execution plan and results.
                This tool leverages LLM capabilities for advanced workflow orchestration.
                """
                return await self.agent.process_intent(query)

    def run(self, **kwargs):
        """Start the MCP server."""
        self.mcp.run(transport="streamable-http", **kwargs)