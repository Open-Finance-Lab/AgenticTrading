#!/usr/bin/env python3
"""
MCP Server Test Suite
====================

Tests for the FinAgent MCP (Model Context Protocol) Server
This test suite validates MCP server functionality, tools, and endpoints.

Requirements:
- MCP server running on localhost:8001
- Conda agent environment activated
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

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

class MCPServerTester:
    """Test suite for MCP Server functionality."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
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
        """Test MCP server health endpoint."""
        test_name = "MCP Server Health Check"
        
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
    
    async def test_mcp_tools_endpoint(self) -> bool:
        """Test MCP tools listing endpoint."""
        test_name = "MCP Tools Endpoint"
        
        try:
            if not self.session:
                self.log_test_result(test_name, False, "HTTP client not available")
                return False
            
            # Try to access MCP tools endpoint
            async with self.session.get(f"{self.base_url}/mcp/tools") as response:
                if response.status == 200:
                    data = await response.json()
                    tools = data.get('tools', [])
                    self.log_test_result(test_name, True, f"Found {len(tools)} tools")
                    return True
                else:
                    self.log_test_result(test_name, False, f"HTTP {response.status}")
                    return False
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
            return False
    
    async def test_mcp_store_memory_tool(self) -> bool:
        """Test MCP store_memory tool."""
        test_name = "MCP Store Memory Tool"
        
        try:
            if not self.session:
                self.log_test_result(test_name, False, "HTTP client not available")
                return False
            
            # Test data for storing memory
            test_memory = {
                "query": "Test financial analysis query",
                "keywords": ["test", "financial", "analysis"],
                "summary": "This is a test memory for MCP validation",
                "agent_id": "mcp_test_agent",
                "event_type": "TEST_EVENT",
                "log_level": "INFO"
            }
            
            # Try to call store_memory tool
            tool_request = {
                "method": "tools/call",
                "params": {
                    "name": "store_memory",
                    "arguments": test_memory
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/mcp/call",
                json=tool_request
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test_result(test_name, True, f"Memory stored: {data}")
                    return True
                else:
                    # Try alternative endpoint structure
                    async with self.session.post(
                        f"{self.base_url}/tools/store_memory",
                        json=test_memory
                    ) as alt_response:
                        if alt_response.status == 200:
                            data = await alt_response.json()
                            self.log_test_result(test_name, True, f"Memory stored (alt): {data}")
                            return True
                        else:
                            self.log_test_result(test_name, False, f"HTTP {response.status}, Alt: {alt_response.status}")
                            return False
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
            return False
    
    async def test_mcp_retrieve_memory_tool(self) -> bool:
        """Test MCP retrieve_memory tool."""
        test_name = "MCP Retrieve Memory Tool"
        
        try:
            if not self.session:
                self.log_test_result(test_name, False, "HTTP client not available")
                return False
            
            # Test data for retrieving memory
            search_params = {
                "search_query": "financial analysis",
                "limit": 5
            }
            
            # Try to call retrieve_memory tool
            tool_request = {
                "method": "tools/call",
                "params": {
                    "name": "retrieve_memory",
                    "arguments": search_params
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/mcp/call",
                json=tool_request
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test_result(test_name, True, f"Memory retrieved: {data}")
                    return True
                else:
                    # Try alternative endpoint
                    async with self.session.post(
                        f"{self.base_url}/tools/retrieve_memory",
                        json=search_params
                    ) as alt_response:
                        if alt_response.status == 200:
                            data = await alt_response.json()
                            self.log_test_result(test_name, True, f"Memory retrieved (alt): {data}")
                            return True
                        else:
                            self.log_test_result(test_name, False, f"HTTP {response.status}, Alt: {alt_response.status}")
                            return False
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Error: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all MCP server tests."""
        logger.info("🧪 Starting MCP Server Test Suite")
        logger.info("=" * 50)
        
        # Run individual tests
        tests = [
            self.test_server_health(),
            self.test_mcp_tools_endpoint(),
            self.test_mcp_store_memory_tool(),
            self.test_mcp_retrieve_memory_tool()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Calculate summary
        passed = sum(1 for result in results if result is True)
        total = len(results)
        
        logger.info("=" * 50)
        logger.info(f"📊 MCP Test Summary: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All MCP tests PASSED!")
        else:
            logger.warning(f"⚠️  {total - passed} MCP tests FAILED")
        
        return {
            "summary": f"{passed}/{total} tests passed",
            "passed": passed,
            "total": total,
            "success_rate": passed / total if total > 0 else 0,
            "results": self.test_results
        }

async def main():
    """Main test execution function."""
    print("🔗 FinAgent MCP Server Test Suite")
    print("=" * 50)
    
    if not HTTP_CLIENT_AVAILABLE:
        print("❌ Required dependencies not available")
        print("   Install with: pip install aiohttp pytest")
        return
    
    # Test MCP server
    async with MCPServerTester() as mcp_tester:
        mcp_results = await mcp_tester.run_all_tests()
    
    print("\n" + "=" * 50)
    print("🏁 MCP Test Suite Complete")
    print("=" * 50)
    
    return mcp_results

if __name__ == "__main__":
    asyncio.run(main())
