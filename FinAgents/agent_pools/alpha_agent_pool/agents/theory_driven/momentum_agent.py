# agent_pools/alpha_agent_pool/agents/theory_driven/momentum_agent.py

from mcp.server.fastmcp import FastMCP, Context as MCPContext
from ...schema.theory_driven_schema import (
    MomentumAgentConfig, MomentumSignal, MomentumSignalRequest
)
from typing import List
import random
import asyncio
import sys
import argparse

class MomentumAgent:
    def __init__(self, config: MomentumAgentConfig):
        """
        Initialize the MomentumAgent with the specified configuration.
        Args:
            config (MomentumAgentConfig): Configuration object for the agent.
        """
        self.config = config
        self.agent = FastMCP("MomentumAlphaAgent")
        self._register_tools()

    def _register_tools(self):
        """
        Register the agent's tools with the FastMCP server. This includes the signal generation endpoint.
        """
        @self.agent.tool()
        async def generate_signal(request: MomentumSignalRequest, ctx: MCPContext = None) -> MomentumSignal:
            """
            Generate a momentum-based trading signal based on the provided price series.
            Args:
                request (MomentumSignalRequest): The request containing the symbol and price list.
                ctx (MCPContext, optional): The MCP context object.
            Returns:
                MomentumSignal: The generated trading signal with score and momentum value.
            """
            symbol = request.symbol
            price_list = request.price_list

            if price_list is None:
                prices = self.get_price_series(symbol, self.config.strategy.window)
            else:
                prices = price_list

            threshold = self.config.strategy.threshold
            momentum = prices[-1] - prices[0]

            if momentum > threshold:
                score = +1.0
                signal = "buy"
            elif momentum < -threshold:
                score = -1.0
                signal = "sell"
            else:
                score = 0.0
                signal = "hold"
                
            return MomentumSignal(
                symbol=symbol,
                score=score,
                signal=signal,
                momentum=round(momentum, 4)
            )

    def get_price_series(self, symbol: str, window: int) -> List[float]:
        """
        Generate a synthetic price series for a given symbol and window size.
        Args:
            symbol (str): The stock or asset symbol.
            window (int): The number of price points to generate.
        Returns:
            List[float]: A list of simulated price values.
        """
        base = random.uniform(90, 110)
        return [base + random.uniform(-2, 2) for _ in range(window)]

    def start_mcp_server(self, port: int = None, host: str = "0.0.0.0", transport: str = "http"):
        """
        Start the MCP server for the MomentumAgent. Supports specifying port, host, and transport protocol (http/sse/stdio).
        Args:
            port (int, optional): The port to bind the server to. Defaults to the value in config.
            host (str, optional): The host address to bind. Defaults to "0.0.0.0".
            transport (str, optional): The transport protocol to use. Defaults to "http".
        """
        port = port or self.config.execution.port
        self.agent.settings.host = host
        self.agent.settings.port = port
        print(f"Starting MomentumAgent MCP server on {host}:{port} (transport={transport}) ...")
        self.agent.run(transport=transport)

    def __repr__(self):
        """
        Return a string representation of the MomentumAgent instance.
        Returns:
            str: String representation including agent ID and port.
        """
        return f"<MomentumAgent id={self.config.agent_id} port={self.config.execution.port}>"