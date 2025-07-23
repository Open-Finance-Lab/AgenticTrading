#!/usr/bin/env python3
"""
FinAgent Services Integration Test Runner
=========================================

Comprehensive test suite for MCP and A2A services.
This runner executes all tests and provides a consolidated report.

Usage:
    python test_services_runner.py [--mcp-only] [--a2a-only] [--simple]
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from test_mcp_server import MCPServerTester
    from test_a2a_server import A2AServerTester
    MCP_TEST_AVAILABLE = True
    A2A_TEST_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    MCP_TEST_AVAILABLE = False
    A2A_TEST_AVAILABLE = False

# Simple HTTP client for basic tests
try:
    import urllib.request
    import urllib.parse
    import json
    SIMPLE_HTTP_AVAILABLE = True
except ImportError:
    SIMPLE_HTTP_AVAILABLE = False

class SimpleServiceTester:
    """Simple service tester using only standard library."""
    
    def __init__(self):
        self.test_results = []
    
    def test_url(self, name: str, url: str) -> bool:
        """Test if a URL is accessible."""
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    data = response.read().decode('utf-8')
                    print(f"✅ {name}: HTTP 200 - {url}")
                    return True
                else:
                    print(f"❌ {name}: HTTP {response.status} - {url}")
                    return False
        except Exception as e:
            print(f"❌ {name}: Connection failed - {e}")
            return False
    
    def run_simple_tests(self):
        """Run simple connectivity tests."""
        print("🔍 Running Simple Connectivity Tests")
        print("=" * 50)
        
        tests = [
            ("Memory Server Health", "http://localhost:8000/health"),
            ("MCP Server (Port Check)", "http://localhost:8001/"),
            ("A2A Server Health", "http://localhost:8002/health"),
            ("A2A Server Status", "http://localhost:8002/status"),
        ]
        
        passed = 0
        total = len(tests)
        
        for name, url in tests:
            if self.test_url(name, url):
                passed += 1
        
        print("=" * 50)
        print(f"📊 Simple Tests: {passed}/{total} passed")
        
        return {
            "summary": f"{passed}/{total} tests passed",
            "passed": passed,
            "total": total,
            "success_rate": passed / total if total > 0 else 0
        }

async def run_comprehensive_tests(mcp_only=False, a2a_only=False):
    """Run comprehensive tests for both services."""
    print("🚀 FinAgent Services Integration Test Runner")
    print("=" * 60)
    
    all_results = {}
    
    # Test MCP Server
    if not a2a_only and MCP_TEST_AVAILABLE:
        print("\n🔗 Testing MCP Server...")
        try:
            async with MCPServerTester() as mcp_tester:
                mcp_results = await mcp_tester.run_all_tests()
                all_results['mcp'] = mcp_results
        except Exception as e:
            print(f"❌ MCP tests failed: {e}")
            all_results['mcp'] = {"error": str(e)}
    
    # Test A2A Server
    if not mcp_only and A2A_TEST_AVAILABLE:
        print("\n🤝 Testing A2A Server...")
        try:
            async with A2AServerTester() as a2a_tester:
                a2a_results = await a2a_tester.run_all_tests()
                all_results['a2a'] = a2a_results
        except Exception as e:
            print(f"❌ A2A tests failed: {e}")
            all_results['a2a'] = {"error": str(e)}
    
    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    total_passed = 0
    total_tests = 0
    
    for service, results in all_results.items():
        if 'error' in results:
            print(f"❌ {service.upper()}: Error - {results['error']}")
        else:
            passed = results.get('passed', 0)
            total = results.get('total', 0)
            rate = results.get('success_rate', 0)
            total_passed += passed
            total_tests += total
            
            status = "✅" if rate == 1.0 else "⚠️" if rate > 0.5 else "❌"
            print(f"{status} {service.upper()}: {passed}/{total} tests passed ({rate:.1%})")
    
    if total_tests > 0:
        overall_rate = total_passed / total_tests
        overall_status = "✅" if overall_rate == 1.0 else "⚠️" if overall_rate > 0.5 else "❌"
        print(f"\n{overall_status} OVERALL: {total_passed}/{total_tests} tests passed ({overall_rate:.1%})")
    
    print("=" * 60)
    
    if total_passed == total_tests and total_tests > 0:
        print("🎉 ALL TESTS PASSED! Services are working correctly.")
    elif total_passed > 0:
        print("⚠️  Some tests passed. Check individual service results above.")
    else:
        print("❌ No tests passed. Check service availability and configuration.")
    
    return all_results

def run_simple_tests():
    """Run simple connectivity tests."""
    tester = SimpleServiceTester()
    return tester.run_simple_tests()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FinAgent Services Test Runner")
    parser.add_argument("--mcp-only", action="store_true", help="Test only MCP server")
    parser.add_argument("--a2a-only", action="store_true", help="Test only A2A server")
    parser.add_argument("--simple", action="store_true", help="Run simple connectivity tests only")
    
    args = parser.parse_args()
    
    if args.simple or (not MCP_TEST_AVAILABLE and not A2A_TEST_AVAILABLE):
        if not MCP_TEST_AVAILABLE or not A2A_TEST_AVAILABLE:
            print("⚠️  Advanced testing not available. Running simple tests.")
            print("   Install dependencies: pip install aiohttp pytest")
        
        run_simple_tests()
    else:
        asyncio.run(run_comprehensive_tests(args.mcp_only, args.a2a_only))

if __name__ == "__main__":
    main()
