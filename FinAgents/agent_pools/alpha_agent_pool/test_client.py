import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession
from .schema.agent_config import AlphaAgentConfig, AgentType

logger = logging.getLogger(__name__)

class AlphaTestClient:
    """Test client for alpha generation agents"""
    
    def __init__(self, url: str = "http://localhost:8001/mcp"):
        self.url = url
        
    async def test_technical_agent(self):
        """Test technical analysis agent"""
        # Generate test data
        test_data = self._generate_ohlcv_data()
        
        # Test agent
        async with streamablehttp_client(self.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                # Execute agent
                result = await session.call_tool(
                    "agent.execute",
                    {
                        "agent_id": "technical_agent",
                        "function": "generate_alpha",
                        "input": test_data
                    }
                )
                return result
                
    async def test_event_agent(self):
        """Test event-driven agent"""
        # Generate test events
        test_events = self._generate_test_events()
        
        # Test agent
        async with streamablehttp_client(self.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                # Execute agent
                result = await session.call_tool(
                    "agent.execute",
                    {
                        "agent_id": "event_agent",
                        "function": "generate_alpha",
                        "input": {"events": test_events}
                    }
                )
                return result
                
    async def test_ml_agent(self):
        """Test ML-based agent"""
        # Generate test features
        test_features = self._generate_test_features()
        
        # Test agent
        async with streamablehttp_client(self.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                # Execute agent
                result = await session.call_tool(
                    "agent.execute",
                    {
                        "agent_id": "ml_agent",
                        "function": "generate_alpha",
                        "input": test_features
                    }
                )
                return result
                
    def _generate_ohlcv_data(self, n_points: int = 100) -> Dict[str, Any]:
        """Generate synthetic OHLCV data"""
        dates = pd.date_range(end=datetime.now(), periods=n_points, freq='1H')
        prices = np.random.normal(100, 1, n_points).cumsum() + 1000
        volumes = np.random.lognormal(10, 1, n_points)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.02, n_points)),
            'low': prices * (1 - np.random.uniform(0, 0.02, n_points)),
            'close': prices * (1 + np.random.normal(0, 0.01, n_points)),
            'volume': volumes
        })
        
        return {"ohlcv": df.values.tolist()}
        
    def _generate_test_events(self, n_events: int = 5) -> List[Dict[str, Any]]:
        """Generate synthetic test events"""
        event_types = ["earnings", "news", "analyst_rating", "insider_trading"]
        events = []
        
        for _ in range(n_events):
            event_type = np.random.choice(event_types)
            timestamp = (datetime.now() - timedelta(hours=np.random.uniform(0, 24))).isoformat()
            impact = np.random.uniform(0.1, 0.9)
            
            events.append({
                "type": event_type,
                "timestamp": timestamp,
                "impact": impact,
                "description": f"Test {event_type} event"
            })
            
        return events
        
    def _generate_test_features(self) -> Dict[str, Any]:
        """Generate synthetic test features for ML agent"""
        return {
            "rsi": np.random.uniform(0, 100),
            "macd": np.random.normal(0, 1),
            "volume_ma": np.random.lognormal(10, 1),
            "price_volatility": np.random.uniform(0, 0.1)
        }
        
async def main():
    """Main test function"""
    client = AlphaTestClient()
    
    # Test technical agent
    print("\nTesting Technical Analysis Agent...")
    tech_result = await client.test_technical_agent()
    print("Technical Agent Result:", tech_result)
    
    # Test event agent
    print("\nTesting Event-Driven Agent...")
    event_result = await client.test_event_agent()
    print("Event Agent Result:", event_result)
    
    # Test ML agent
    print("\nTesting ML-based Agent...")
    ml_result = await client.test_ml_agent()
    print("ML Agent Result:", ml_result)
    
if __name__ == "__main__":
    asyncio.run(main()) 