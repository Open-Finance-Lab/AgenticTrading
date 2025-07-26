#!/usr/bin/env python3
"""
A2A Health Check Tool
=====================

A tool to properly check the health of A2A servers using the correct A2A protocol
instead of HTTP GET requests that cause 405 errors.

Author: FinAgent Development Team
Created: 2025-07-25
"""

import asyncio
import httpx
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class A2AHealthChecker:
    """
    A2A-compliant health checker that uses proper JSON-RPC format
    instead of HTTP GET requests.
    """
    
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.http_client = httpx.AsyncClient(timeout=timeout)
    
    async def check_a2a_server_health(self, server_url: str) -> Dict[str, Any]:
        """
        Check A2A server health using proper A2A protocol.
        
        Args:
            server_url: A2A server URL (e.g., http://localhost:8002)
            
        Returns:
            Health check results
        """
        health_result = {
            "server_url": server_url,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "unknown",
            "response_time_ms": None,
            "error": None
        }
        
        try:
            start_time = datetime.utcnow()
            
            # Try A2A protocol health check using correct JSON-RPC format
            health_request = {
                "jsonrpc": "2.0",
                "id": f"health_check_{int(datetime.utcnow().timestamp())}",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": '{"action": "health", "timestamp": "' + datetime.utcnow().isoformat() + '"}'
                            }
                        ]
                    }
                }
            }
            
            logger.info(f"🏥 Checking A2A server health: {server_url}")
            
            # Send health check request to A2A server root endpoint
            response = await self.http_client.post(
                server_url,
                json=health_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
            
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000
            health_result["response_time_ms"] = round(response_time, 2)
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    health_result["status"] = "healthy"
                    health_result["response_data"] = response_data
                    logger.info(f"✅ A2A server is healthy: {server_url}")
                except json.JSONDecodeError:
                    health_result["status"] = "responsive_but_invalid_json"
                    health_result["response_text"] = response.text[:200]
                    logger.warning(f"⚠️ A2A server responded but returned invalid JSON")
            else:
                health_result["status"] = f"http_error_{response.status_code}"
                health_result["response_text"] = response.text[:200]
                logger.warning(f"⚠️ A2A server returned HTTP {response.status_code}")
                
        except httpx.ConnectError:
            health_result["status"] = "connection_failed"
            health_result["error"] = "Cannot connect to server"
            logger.error(f"❌ Cannot connect to A2A server: {server_url}")
            
        except httpx.TimeoutException:
            health_result["status"] = "timeout"
            health_result["error"] = f"Request timed out after {self.timeout}s"
            logger.error(f"⏰ A2A server health check timed out: {server_url}")
            
        except Exception as e:
            health_result["status"] = "error"
            health_result["error"] = str(e)
            logger.error(f"❌ A2A server health check failed: {e}")
        
        return health_result
    
    async def check_legacy_memory_server_health(self, server_url: str) -> Dict[str, Any]:
        """
        Check legacy memory server health using HTTP GET (works for uvicorn servers).
        
        Args:
            server_url: Memory server URL (e.g., http://localhost:8000)
            
        Returns:
            Health check results
        """
        health_result = {
            "server_url": server_url,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "unknown",
            "response_time_ms": None,
            "error": None
        }
        
        try:
            start_time = datetime.utcnow()
            
            logger.info(f"🏥 Checking legacy memory server health: {server_url}")
            
            # Try simple GET request for legacy servers
            response = await self.http_client.get(f"{server_url}/health")
            
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000
            health_result["response_time_ms"] = round(response_time, 2)
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    health_result["status"] = "healthy"
                    health_result["response_data"] = response_data
                    logger.info(f"✅ Legacy memory server is healthy: {server_url}")
                except json.JSONDecodeError:
                    health_result["status"] = "responsive_but_invalid_json"
                    health_result["response_text"] = response.text[:200]
                    logger.warning(f"⚠️ Legacy server responded but returned invalid JSON")
            else:
                health_result["status"] = f"http_error_{response.status_code}"
                health_result["response_text"] = response.text[:200]
                logger.warning(f"⚠️ Legacy server returned HTTP {response.status_code}")
                
        except httpx.ConnectError:
            health_result["status"] = "connection_failed"
            health_result["error"] = "Cannot connect to server"
            logger.error(f"❌ Cannot connect to legacy memory server: {server_url}")
            
        except httpx.TimeoutException:
            health_result["status"] = "timeout"
            health_result["error"] = f"Request timed out after {self.timeout}s"
            logger.error(f"⏰ Legacy memory server health check timed out: {server_url}")
            
        except Exception as e:
            health_result["status"] = "error"
            health_result["error"] = str(e)
            logger.error(f"❌ Legacy memory server health check failed: {e}")
        
        return health_result
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on all memory services.
        
        Returns:
            Complete health check results
        """
        logger.info("🔍 Starting comprehensive memory services health check...")
        
        # Define servers to check
        servers = {
            "alpha_agent_pool": {
                "url": "http://127.0.0.1:8081",
                "type": "mcp_sse",
                "description": "Alpha Agent Pool MCP Server"
            },
            "a2a_memory_server": {
                "url": "http://127.0.0.1:8002", 
                "type": "a2a",
                "description": "A2A Memory Server"
            },
            "legacy_memory_server": {
                "url": "http://127.0.0.1:8000",
                "type": "legacy",
                "description": "Legacy Memory Server"
            }
        }
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "unknown",
            "servers": {},
            "summary": {
                "total_servers": len(servers),
                "healthy_servers": 0,
                "unhealthy_servers": 0,
                "errors": []
            }
        }
        
        # Check each server
        for server_name, server_info in servers.items():
            logger.info(f"🔍 Checking {server_info['description']}...")
            
            if server_info["type"] == "a2a":
                health_result = await self.check_a2a_server_health(server_info["url"])
            elif server_info["type"] == "legacy":
                health_result = await self.check_legacy_memory_server_health(server_info["url"])
            else:
                # For MCP SSE servers, just check basic connectivity
                health_result = await self._check_basic_connectivity(server_info["url"])
            
            health_result["server_type"] = server_info["type"]
            health_result["description"] = server_info["description"]
            results["servers"][server_name] = health_result
            
            if health_result["status"] in ["healthy", "responsive_but_invalid_json"]:
                results["summary"]["healthy_servers"] += 1
            else:
                results["summary"]["unhealthy_servers"] += 1
                results["summary"]["errors"].append(f"{server_name}: {health_result.get('error', health_result['status'])}")
        
        # Determine overall status
        if results["summary"]["healthy_servers"] == results["summary"]["total_servers"]:
            results["overall_status"] = "all_healthy"
        elif results["summary"]["healthy_servers"] > 0:
            results["overall_status"] = "partially_healthy"
        else:
            results["overall_status"] = "all_unhealthy"
        
        return results
    
    async def _check_basic_connectivity(self, server_url: str) -> Dict[str, Any]:
        """Basic connectivity check for MCP servers."""
        health_result = {
            "server_url": server_url,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "unknown",
            "response_time_ms": None,
            "error": None
        }
        
        try:
            start_time = datetime.utcnow()
            response = await self.http_client.get(f"{server_url}/", timeout=5.0)
            end_time = datetime.utcnow()
            
            response_time = (end_time - start_time).total_seconds() * 1000
            health_result["response_time_ms"] = round(response_time, 2)
            
            # Accept any response as connectivity proof
            if response.status_code in [200, 404, 405]:
                health_result["status"] = "responsive"
                logger.info(f"✅ Server is responsive: {server_url}")
            else:
                health_result["status"] = f"http_error_{response.status_code}"
                health_result["response_text"] = response.text[:200]
                
        except Exception as e:
            health_result["status"] = "error"
            health_result["error"] = str(e)
            logger.error(f"❌ Basic connectivity check failed: {e}")
        
        return health_result
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()

async def main():
    """Main function for command line usage."""
    print("🏥 A2A Health Check Tool")
    print("=" * 50)
    
    checker = A2AHealthChecker()
    
    try:
        results = await checker.comprehensive_health_check()
        
        print(f"\n📊 Health Check Results ({results['timestamp']})")
        print("=" * 50)
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Healthy Servers: {results['summary']['healthy_servers']}/{results['summary']['total_servers']}")
        
        print("\n🖥️  Server Details:")
        for server_name, health_data in results["servers"].items():
            status_icon = "✅" if health_data["status"] in ["healthy", "responsive"] else "❌"
            print(f"  {status_icon} {health_data['description']}")
            print(f"     URL: {health_data['server_url']}")
            print(f"     Status: {health_data['status']}")
            if health_data["response_time_ms"]:
                print(f"     Response Time: {health_data['response_time_ms']}ms")
            if health_data.get("error"):
                print(f"     Error: {health_data['error']}")
            print()
        
        if results['summary']['errors']:
            print("⚠️  Errors Found:")
            for error in results['summary']['errors']:
                print(f"  • {error}")
        
        print(f"\n✅ Health check completed!")
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        logger.error(f"Health check error: {e}")
    
    finally:
        await checker.close()

if __name__ == "__main__":
    asyncio.run(main())
