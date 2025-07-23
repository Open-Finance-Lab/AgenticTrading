#!/usr/bin/env python3
"""
FinAgent A2A Integration Final Test

This script performs comprehensive testing of the A2A integration with enhanced
relationship creation and visualization capabilities.
"""

import asyncio
import aiohttp
import json
from datetime import datetime

async def test_a2a_comprehensive_integration():
    """Comprehensive A2A integration test with relationship verification"""
    
    print("🚀 FinAgent A2A Integration Final Test")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:8011"
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Multiple signal types
        print("\n📡 Test 1: Multiple Signal Types")
        signals = [
            {
                "sender_agent_id": "final_test_alpha_001",
                "target_agent_id": "memory_agent",
                "signal_type": "buy_signal",
                "signal_data": {
                    "symbol": "TSLA",
                    "price": 245.80,
                    "volume": 2500000,
                    "confidence": 0.92,
                    "reasoning": "Strong earnings beat + technical breakout"
                },
                "priority": "high"
            },
            {
                "sender_agent_id": "final_test_alpha_001",
                "target_agent_id": "memory_agent",
                "signal_type": "sell_signal",
                "signal_data": {
                    "symbol": "NVDA",
                    "price": 458.30,
                    "volume": 1800000,
                    "confidence": 0.78,
                    "reasoning": "Profit taking at resistance level"
                },
                "priority": "medium"
            },
            {
                "sender_agent_id": "final_test_momentum_002",
                "target_agent_id": "memory_agent",
                "signal_type": "hold_signal",
                "signal_data": {
                    "symbol": "AAPL",
                    "price": 186.20,
                    "volume": 950000,
                    "confidence": 0.65,
                    "reasoning": "Consolidation phase, awaiting catalyst"
                },
                "priority": "low"
            }
        ]
        
        signal_ids = []
        for i, signal in enumerate(signals, 1):
            async with session.post(f"{base_url}/a2a/signals/transmit", json=signal) as response:
                result = await response.json()
                if response.status == 200:
                    signal_ids.append(result["signal_id"])
                    print(f"  ✅ Signal {i}: {signal['signal_type']} for {signal['signal_data']['symbol']} stored")
                else:
                    print(f"  ❌ Signal {i} failed: {result}")
        
        # Test 2: Advanced strategy sharing
        print("\n📊 Test 2: Advanced Strategy Sharing")
        strategies = [
            {
                "sender_agent_id": "final_test_strategy_001",
                "target_agent_id": "memory_agent",
                "strategy_type": "multi_factor_momentum",
                "strategy_data": {
                    "factors": ["price_momentum", "volume_momentum", "earnings_momentum"],
                    "weights": [0.4, 0.3, 0.3],
                    "rebalance_frequency": "daily",
                    "risk_management": {
                        "max_position_size": 0.05,
                        "stop_loss": -0.02,
                        "take_profit": 0.04
                    }
                },
                "performance_metrics": {
                    "returns": 0.24,
                    "volatility": 0.15,
                    "sharpe_ratio": 1.6,
                    "max_drawdown": -0.12,
                    "win_rate": 0.68,
                    "avg_holding_period": 3.2
                },
                "sharing_permission": "collaborative"
            },
            {
                "sender_agent_id": "final_test_strategy_002",
                "target_agent_id": "memory_agent",
                "strategy_type": "mean_reversion_enhanced",
                "strategy_data": {
                    "lookback_period": 14,
                    "z_score_threshold": 2.0,
                    "volatility_filter": True,
                    "market_regime_filter": True
                },
                "performance_metrics": {
                    "returns": 0.08,
                    "volatility": 0.09,
                    "sharpe_ratio": 0.89,
                    "max_drawdown": -0.05,
                    "win_rate": 0.58
                },
                "sharing_permission": "read_only"
            }
        ]
        
        strategy_ids = []
        for i, strategy in enumerate(strategies, 1):
            async with session.post(f"{base_url}/a2a/strategies/share", json=strategy) as response:
                result = await response.json()
                if response.status == 200:
                    strategy_ids.append(result["strategy_id"])
                    print(f"  ✅ Strategy {i}: {strategy['strategy_type']} with {strategy['performance_metrics']['returns']:.1%} returns stored")
                else:
                    print(f"  ❌ Strategy {i} failed: {result}")
        
        # Test 3: Data query and retrieval
        print("\n🔍 Test 3: Data Query and Retrieval")
        query_payload = {
            "query_type": "performance_analysis",
            "filters": {
                "memory_type": "strategy",
                "min_returns": 0.05
            },
            "requesting_agent_id": "final_test_analyzer_001"
        }
        
        async with session.post(f"{base_url}/a2a/data/query", json=query_payload) as response:
            result = await response.json()
            if response.status == 200:
                results = result.get('results', result.get('data', []))
                print(f"  ✅ Query returned {len(results)} high-performance strategies")
                for strategy in results[:3]:  # Show first 3
                    if isinstance(strategy, dict) and 'content' in strategy:
                        content = json.loads(strategy['content']) if isinstance(strategy['content'], str) else strategy['content']
                        returns = content.get('performance_metrics', {}).get('returns', 0)
                        print(f"    - Strategy: {content.get('strategy_type', 'Unknown')} ({returns:.1%} returns)")
                    else:
                        print(f"    - Raw result: {str(strategy)[:100]}...")
            else:
                print(f"  ❌ Query failed: {result}")
        
        # Test 4: System health and capabilities
        print("\n🏥 Test 4: System Health and Capabilities")
        
        async with session.get(f"{base_url}/a2a/health") as response:
            health = await response.json()
            if response.status == 200:
                messages = health.get('messages_processed', health.get('message_count', 0))
                print(f"  ✅ Health: {health['status']} - {messages} messages processed")
            else:
                print(f"  ❌ Health check failed")
        
        async with session.get(f"{base_url}/a2a/capabilities") as response:
            capabilities = await response.json()
            if response.status == 200:
                caps = capabilities.get('capabilities', capabilities.get('supported_operations', []))
                print(f"  ✅ Capabilities: {len(caps)} operations available")
                for cap in caps:
                    if isinstance(cap, dict):
                        print(f"    - {cap.get('name', 'Unknown')}: {cap.get('description', 'No description')}")
                    else:
                        print(f"    - {cap}")
            else:
                print(f"  ❌ Capabilities check failed")
        
        async with session.get(f"{base_url}/a2a/status") as response:
            status = await response.json()
            if response.status == 200:
                print(f"  ✅ Status: {status['memory_count']} memories, {status['active_connections']} connections")
            else:
                print(f"  ❌ Status check failed")
    
    print("\n🎯 A2A Integration Test Complete!")
    print("\n💡 Neo4j Desktop Visualization:")
    print("   Run this query to see the enhanced network:")
    print("   MATCH (n)-[r]->(m) WHERE NOT type(r) = 'SIMILAR_SIGNAL' RETURN n, r, m LIMIT 100")
    print("\n📊 Specific relationship queries:")
    print("   1. Agent networks: MATCH (a:Agent)-[:CREATED]->(m:Memory) RETURN a, m")
    print("   2. Symbol targeting: MATCH (m:Memory)-[:TARGETS]->(s:Symbol) RETURN m, s")
    print("   3. Performance levels: MATCH (m:Memory)-[:HAS_PERFORMANCE]->(p:PerformanceLevel) RETURN m, p")

if __name__ == "__main__":
    asyncio.run(test_a2a_comprehensive_integration())
