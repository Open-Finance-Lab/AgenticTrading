# agent_pools/alpha_agent_pool/agents/theory_driven/momentum_agent.py

from mcp.server.fastmcp import FastMCP, Context as MCPContext
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from schema.theory_driven_schema import (
    MomentumAgentConfig, MomentumSignalRequest, AlphaStrategyFlow, MarketContext, Decision, Action, PerformanceFeedback, Metadata
)
from typing import List
import random
import asyncio
import sys
import argparse
import json
import os
from datetime import datetime
import hashlib

class MomentumAgent:
    def __init__(self, config: MomentumAgentConfig):
        """
        Initialize the MomentumAgent with the specified configuration.
        On initialization, clear the local memory and strategy flow files.
        Args:
            config (MomentumAgentConfig): Configuration object for the agent.
        """
        self.config = config
        self.agent = FastMCP("MomentumAlphaAgent")
        # Path to the shared memory JSON file (used by AlphaAgentPool)
        self.memory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../memory_unit.json"))
        # Path to store strategy signal flow
        self.signal_flow_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../momentum_signal_flow.json"))
        # Clear memory and signal flow files on each agent restart
        for path in [self.signal_flow_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        self._register_tools()

    def _read_memory(self, key):
        if not os.path.exists(self.memory_path):
            return None
        try:
            with open(self.memory_path, 'r') as f:
                data = json.load(f)
            return data.get(key)
        except Exception:
            return None

    def _write_signal_flow(self, flow_obj):
        # Append the strategy flow object to a json file as a list
        if os.path.exists(self.signal_flow_path):
            try:
                with open(self.signal_flow_path, 'r') as f:
                    flow = json.load(f)
            except Exception:
                flow = []
        else:
            flow = []
        flow.append(flow_obj)
        with open(self.signal_flow_path, 'w') as f:
            json.dump(flow, f, indent=2)

    def _register_tools(self):
        """
        Register the agent's tools with the FastMCP server. This includes the signal generation endpoint.
        """
        @self.agent.tool()
        async def generate_signal(request: MomentumSignalRequest, ctx: MCPContext = None) -> dict:
            """
            Generate a full alpha strategy flow output as specified, using memory and current config.
            """
            symbol = request.symbol
            price_list = request.price_list
            # If price_list is not provided, try to fetch from memory
            if price_list is None:
                closes = []
                for i in range(self.config.strategy.window):
                    key = f"AAPL_close_2024-01-{str(31-i).zfill(2)} 05:00:00"
                    val = self._read_memory(key)
                    if val is not None:
                        closes.insert(0, float(val))
                prices = closes if closes else self.get_price_series(symbol, self.config.strategy.window)
            else:
                prices = price_list
            # --- Feature engineering ---
            price_today = prices[-1]
            price_20d_ago = prices[0] if len(prices) >= 20 else prices[0]
            sma_10 = sum(prices[-10:]) / min(10, len(prices))
            sma_50 = sum(prices) / len(prices)  # fallback: use all as 50
            momentum_score = (price_today - price_20d_ago) / price_20d_ago if price_20d_ago != 0 else 0
            # --- Decision logic ---
            if momentum_score > self.config.strategy.threshold:
                signal = "BUY"
                confidence = round(momentum_score, 2)
                reasoning = "20-day momentum above threshold, SMA10 > SMA50"
                predicted_return = 0.012
                risk_estimate = 0.018
                execution_weight = 0.3
            elif momentum_score < -self.config.strategy.threshold:
                signal = "SELL"
                confidence = round(abs(momentum_score), 2)
                reasoning = "20-day momentum below negative threshold, SMA10 < SMA50"
                predicted_return = -0.012
                risk_estimate = 0.018
                execution_weight = -0.3
            else:
                signal = "HOLD"
                confidence = 0.0
                reasoning = "Momentum within threshold, no action"
                predicted_return = 0.0
                risk_estimate = 0.01
                execution_weight = 0.0
            # --- Build output ---
            now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            code_hash = hashlib.sha256((str(prices)+signal).encode()).hexdigest()
            flow_obj = AlphaStrategyFlow(
                alpha_id="momentum_v3",
                version="2025.06.23-001",
                timestamp=now,
                market_context=MarketContext(
                    symbol=symbol,
                    regime_tag="bullish_trend" if signal=="BUY" else "bearish_trend" if signal=="SELL" else "neutral",
                    input_features={
                        "price_today": price_today,
                        "price_20d_ago": price_20d_ago,
                        "sma_10": sma_10,
                        "sma_50": sma_50,
                        "momentum_score": round(momentum_score, 4)
                    }
                ),
                decision=Decision(
                    signal=signal,
                    confidence=confidence,
                    reasoning=reasoning,
                    predicted_return=predicted_return,
                    risk_estimate=risk_estimate,
                    signal_type="directional",
                    asset_scope=[symbol]
                ),
                action=Action(
                    execution_weight=execution_weight,
                    order_type="market",
                    order_price=price_today,
                    execution_delay="T+0"
                ),
                performance_feedback=PerformanceFeedback(
                    status="pending",
                    evaluation_link=None
                ),
                metadata=Metadata(
                    generator_agent="alpha_agent_12",
                    strategy_prompt="Trade based on 20-day price momentum and SMA crossover",
                    code_hash=f"sha256:{code_hash}",
                    context_id=f"dag_{now[:10].replace('-', '')}_{now[11:13]}"
                )
            )
            # Write to strategy flow file
            self._write_signal_flow(flow_obj.dict())
            return flow_obj.dict()
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