#!/usr/bin/env python3
"""
A2A Server Test Suite
====================

Tests for the FinAgent A2A (Agent-to-Agent) Server
This test suite validates A2A server functionality, endpoints, and communication features.

Requirements:
- A2A server running on localhost:8002
- Conda agent environment activated
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import aiohttp
    import pytest
    HTTP_CLIENT_AVAILABLE = True
except ImportError:
    HTTP_CLIENT_AVAILABLE = False
    print("⚠️  aiohttp not available. Install with: pip install aiohttp pytest")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class A2AServerTester:
    """Test suite for A2A Server functionality."""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        """Async context manager entry."""
        if HTTP_CLIENT_AVAILABLE:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def log_test_result(self, test_name: str, success: bool, message: str = ""):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}: {message}")
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message
        })
    
    async def test_server_health(self) -> bool:
        """Test A2A server health endpoint."""
        test_name = "A2A Server Health Check"
        
        try:
            if not self.session:
                self.log_test_result(test_name, False, "HTTP client not available")
                return False
            
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test_result(test_name, True, f"Server healthy - {data}")
                    return True
                else:
                    self.log_test_result(test_name, False, f"HTTP {response.status}")
                    return False
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Connection error: {e}")
            return False
    
    async def test_server_status(self) -> bool:
        """Test A2A server status endpoint."""
        test_name = "A2A Server Status Check"
        
        try:
            if not self.session:
                self.log_test_result(test_name, False, "HTTP client not available")
                return False
            
            async with self.session.get(f"{self.base_url}/status") as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test_result(test_name, True, f"Status: {data}")
                    return True
                else:
                    self.log_test_result(test_name, False, f"HTTP {response.status}")
                    return False
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
            return False
    
    async def test_send_signal(self) -> bool:
        """Test A2A signal sending endpoint."""
        test_name = "A2A Send Signal"
        
        try:
            if not self.session:
                self.log_test_result(test_name, False, "HTTP client not available")
                return False
            
            # Test signal data
            signal_data = {
                "sender_id": "test_agent_1",
                "recipient_id": "test_agent_2",
                "signal_type": "test_signal",
                "payload": {
                    "message": "Test signal from A2A test suite",
                    "timestamp": datetime.now().isoformat(),
                    "test_data": {"key": "value"}
                },
                "priority": "normal"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/signals/send",
                json=signal_data
            ) as response:
                if response.status in [200, 201]:
                    data = await response.json()
                    self.log_test_result(test_name, True, f"Signal sent: {data}")
                    return True
                else:
                    self.log_test_result(test_name, False, f"HTTP {response.status}")
                    return False
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
            return False
    
    async def test_signal_history(self) -> bool:
        """Test A2A signal history endpoint."""
        test_name = "A2A Signal History"
        
        try:
            if not self.session:
                self.log_test_result(test_name, False, "HTTP client not available")
                return False
            
            # Test with query parameters
            params = {
                "agent_id": "test_agent_1",
                "limit": 10
            }
            
            async with self.session.get(
                f"{self.base_url}/api/v1/signals/history",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test_result(test_name, True, f"History retrieved: {len(data.get('signals', []))} records")
                    return True
                else:
                    self.log_test_result(test_name, False, f"HTTP {response.status}")
                    return False
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
            return False
    
    async def test_share_strategy(self) -> bool:
        """Test A2A strategy sharing endpoint."""
        test_name = "A2A Share Strategy"
        
        try:
            if not self.session:
                self.log_test_result(test_name, False, "HTTP client not available")
                return False
            
            # Test strategy data
            strategy_data = {
                "agent_id": "test_strategy_agent",
                "strategy_name": "Test Strategy",
                "strategy_type": "test",
                "parameters": {
                    "param1": "value1",
                    "param2": 42,
                    "param3": True
                },
                "performance_metrics": {
                    "returns": 0.15,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": 0.05
                },
                "description": "This is a test strategy for A2A validation"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/strategies/share",
                json=strategy_data
            ) as response:
                if response.status in [200, 201]:
                    data = await response.json()
                    self.log_test_result(test_name, True, f"Strategy shared: {data}")
                    return True
                else:
                    self.log_test_result(test_name, False, f"HTTP {response.status}")
                    return False
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
            return False
    
    async def test_list_strategies(self) -> bool:
        """Test A2A strategy listing endpoint."""
        test_name = "A2A List Strategies"
        
        try:
            if not self.session:
                self.log_test_result(test_name, False, "HTTP client not available")
                return False
            
            async with self.session.get(f"{self.base_url}/api/v1/strategies/list") as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test_result(test_name, True, f"Strategies listed: {len(data.get('strategies', []))} found")
                    return True
                else:
                    self.log_test_result(test_name, False, f"HTTP {response.status}")
                    return False
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
            return False
    
    async def test_share_memory(self) -> bool:
        """Test A2A memory sharing endpoint."""
        test_name = "A2A Share Memory"
        
        try:
            if not self.session:
                self.log_test_result(test_name, False, "HTTP client not available")
                return False
            
            # Test memory data
            memory_data = {
                "sender_id": "test_memory_agent",
                "memory_type": "experience",
                "content": {
                    "title": "Test Memory Share",
                    "description": "This is a test memory for A2A validation",
                    "data": {
                        "market_analysis": "Test analysis data",
                        "insights": ["insight1", "insight2"],
                        "timestamp": datetime.now().isoformat()
                    }
                },
                "tags": ["test", "a2a", "memory"],
                "importance": 0.7
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/memory/share",
                json=memory_data
            ) as response:
                if response.status in [200, 201]:
                    data = await response.json()
                    self.log_test_result(test_name, True, f"Memory shared: {data}")
                    return True
                else:
                    self.log_test_result(test_name, False, f"HTTP {response.status}")
                    return False
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
            return False
    
    async def test_analytics_summary(self) -> bool:
        """Test A2A analytics summary endpoint."""
        test_name = "A2A Analytics Summary"
        
        try:
            if not self.session:
                self.log_test_result(test_name, False, "HTTP client not available")
                return False
            
            async with self.session.get(f"{self.base_url}/api/v1/analytics/summary") as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test_result(test_name, True, f"Analytics: {data}")
                    return True
                else:
                    self.log_test_result(test_name, False, f"HTTP {response.status}")
                    return False
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all A2A server tests."""
        logger.info("🧪 Starting A2A Server Test Suite")
        logger.info("=" * 50)
        
        # Run individual tests
        tests = [
            self.test_server_health(),
            self.test_server_status(),
            self.test_send_signal(),
            self.test_signal_history(),
            self.test_share_strategy(),
            self.test_list_strategies(),
            self.test_share_memory(),
            self.test_analytics_summary()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Calculate summary
        passed = sum(1 for result in results if result is True)
        total = len(results)
        
        logger.info("=" * 50)
        logger.info(f"📊 A2A Test Summary: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All A2A tests PASSED!")
        else:
            logger.warning(f"⚠️  {total - passed} A2A tests FAILED")
        
        return {
            "summary": f"{passed}/{total} tests passed",
            "passed": passed,
            "total": total,
            "success_rate": passed / total if total > 0 else 0,
            "results": self.test_results
        }

async def main():
    """Main test execution function."""
    print("🤝 FinAgent A2A Server Test Suite")
    print("=" * 50)
    
    if not HTTP_CLIENT_AVAILABLE:
        print("❌ Required dependencies not available")
        print("   Install with: pip install aiohttp pytest")
        return
    
    # Test A2A server
    async with A2AServerTester() as a2a_tester:
        a2a_results = await a2a_tester.run_all_tests()
    
    print("\n" + "=" * 50)
    print("🏁 A2A Test Suite Complete")
    print("=" * 50)
    
    return a2a_results

if __name__ == "__main__":
    asyncio.run(main())
