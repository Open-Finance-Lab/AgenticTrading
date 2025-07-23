#!/usr/bin/env python3
"""
Quick Service Health Checker
============================

A lightweight service health checker that uses only Python standard library.
Tests basic connectivity and health endpoints for MCP and A2A services.

Usage:
    python quick_health_check.py
"""

import urllib.request
import urllib.parse
import json
import socket
import sys
from datetime import datetime
from typing import Dict, Any

class QuickHealthChecker:
    """Quick health checker for FinAgent services."""
    
    def __init__(self):
        self.results = []
    
    def check_port(self, host: str, port: int, timeout: int = 3) -> bool:
        """Check if a port is open."""
        try:
            with socket.create_connection((host, port), timeout):
                return True
        except (socket.timeout, socket.error):
            return False
    
    def check_http_endpoint(self, name: str, url: str, timeout: int = 5) -> Dict[str, Any]:
        """Check HTTP endpoint and return detailed results."""
        result = {
            "name": name,
            "url": url,
            "success": False,
            "status_code": None,
            "response_data": None,
            "error": None,
            "response_time": None
        }
        
        start_time = datetime.now()
        
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'FinAgent-HealthChecker/1.0')
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                end_time = datetime.now()
                result["response_time"] = (end_time - start_time).total_seconds()
                result["status_code"] = response.status
                
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        data = json.loads(response.read().decode('utf-8'))
                        result["response_data"] = data
                    else:
                        result["response_data"] = response.read().decode('utf-8')[:200]
                    
                    result["success"] = True
                else:
                    result["error"] = f"HTTP {response.status}"
                    
        except urllib.error.HTTPError as e:
            result["status_code"] = e.code
            result["error"] = f"HTTP {e.code}: {e.reason}"
        except urllib.error.URLError as e:
            result["error"] = f"URL Error: {e.reason}"
        except socket.timeout:
            result["error"] = "Timeout"
        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
        
        self.results.append(result)
        return result
    
    def print_result(self, result: Dict[str, Any]):
        """Print formatted test result."""
        status = "✅" if result["success"] else "❌"
        name = result["name"]
        url = result["url"]
        
        print(f"{status} {name}")
        print(f"   URL: {url}")
        
        if result["success"]:
            response_time = result.get("response_time", 0)
            print(f"   Status: HTTP {result['status_code']} ({response_time:.3f}s)")
            
            if result["response_data"]:
                if isinstance(result["response_data"], dict):
                    # Pretty print JSON response
                    for key, value in result["response_data"].items():
                        if isinstance(value, str) and len(value) > 50:
                            value = value[:50] + "..."
                        print(f"   {key}: {value}")
                else:
                    print(f"   Response: {result['response_data']}")
        else:
            print(f"   Error: {result['error']}")
        
        print()
    
    def run_health_checks(self):
        """Run all health checks."""
        print("🏥 FinAgent Services Health Check")
        print("=" * 50)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Define test endpoints
        endpoints = [
            ("Memory Server Health", "http://localhost:8000/health"),
            ("MCP Server (Port 8001)", "http://localhost:8001/"),
            ("MCP Server Health", "http://localhost:8001/health"),
            ("A2A Server Health", "http://localhost:8002/health"),
            ("A2A Server Status", "http://localhost:8002/status"),
        ]
        
        # Check port connectivity first
        print("🔌 Port Connectivity Check:")
        ports = [
            ("Memory Server", "localhost", 8000),
            ("MCP Server", "localhost", 8001),
            ("A2A Server", "localhost", 8002),
        ]
        
        for name, host, port in ports:
            if self.check_port(host, port):
                print(f"✅ {name} (:{port}) - Port is open")
            else:
                print(f"❌ {name} (:{port}) - Port is closed or not responding")
        
        print("\n🌐 HTTP Endpoint Tests:")
        
        # Test each endpoint
        passed = 0
        for name, url in endpoints:
            result = self.check_http_endpoint(name, url)
            self.print_result(result)
            if result["success"]:
                passed += 1
        
        # Summary
        total = len(endpoints)
        success_rate = passed / total if total > 0 else 0
        
        print("=" * 50)
        print(f"📊 Health Check Summary: {passed}/{total} endpoints healthy ({success_rate:.1%})")
        
        if success_rate == 1.0:
            print("🎉 All services are healthy!")
        elif success_rate >= 0.5:
            print("⚠️  Some services may have issues. Check details above.")
        else:
            print("❌ Multiple service issues detected. Check service status.")
        
        print("=" * 50)
        
        return {
            "passed": passed,
            "total": total,
            "success_rate": success_rate,
            "results": self.results
        }

def main():
    """Main entry point."""
    checker = QuickHealthChecker()
    return checker.run_health_checks()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Health check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Health check failed: {e}")
        sys.exit(1)
