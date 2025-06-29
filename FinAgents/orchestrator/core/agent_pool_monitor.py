"""
Agent Pool Health Monitoring and Validation System
Ensures all agent pools are properly started and operational via MCP protocol
"""

import asyncio
import logging
import json
import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import subprocess
import psutil
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoolStatus(Enum):
    """Agent pool status enumeration"""
    HEALTHY = "healthy"
    STARTING = "starting"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class AgentPoolInfo:
    """Information about an agent pool"""
    name: str
    url: str
    port: int
    process_id: Optional[int] = None
    status: PoolStatus = PoolStatus.STOPPED
    last_check: Optional[datetime] = None
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    capabilities: List[str] = None
    version: Optional[str] = None

class AgentPoolMonitor:
    """
    Monitor and manage agent pool health and lifecycle
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pools: Dict[str, AgentPoolInfo] = {}
        self.monitoring_active = False
        self.health_check_interval = config.get("health_check_interval", 30)
        self._initialize_pools()
    
    def _initialize_pools(self):
        """Initialize agent pool configurations"""
        agent_pools = self.config.get("agent_pools", {})
        
        for pool_name, pool_config in agent_pools.items():
            if pool_config.get("enabled", True):
                url = pool_config.get("url", "")
                port = self._extract_port_from_url(url)
                
                self.pools[pool_name] = AgentPoolInfo(
                    name=pool_name,
                    url=url,
                    port=port,
                    capabilities=pool_config.get("capabilities", [])
                )
                
                logger.info(f"Registered agent pool: {pool_name} at {url}")
    
    def _extract_port_from_url(self, url: str) -> int:
        """Extract port number from URL"""
        try:
            if ":" in url:
                return int(url.split(":")[-1])
            return 80  # Default port
        except ValueError:
            return 80
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        self.monitoring_active = True
        logger.info("Starting agent pool monitoring")
        
        while self.monitoring_active:
            await self.check_all_pools()
            await asyncio.sleep(self.health_check_interval)
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        logger.info("Stopped agent pool monitoring")
    
    async def check_all_pools(self) -> Dict[str, AgentPoolInfo]:
        """Check health of all registered agent pools"""
        results = {}
        
        for pool_name, pool_info in self.pools.items():
            try:
                updated_info = await self.check_pool_health(pool_name)
                results[pool_name] = updated_info
            except Exception as e:
                logger.error(f"Error checking {pool_name}: {e}")
                pool_info.status = PoolStatus.ERROR
                pool_info.error_message = str(e)
                results[pool_name] = pool_info
        
        return results
    
    async def check_pool_health(self, pool_name: str) -> AgentPoolInfo:
        """
        Check health of a specific agent pool
        
        Args:
            pool_name: Name of the agent pool to check
            
        Returns:
            Updated AgentPoolInfo with current status
        """
        if pool_name not in self.pools:
            raise ValueError(f"Unknown agent pool: {pool_name}")
        
        pool_info = self.pools[pool_name]
        start_time = time.time()
        
        try:
            # Check if process is running
            if pool_info.process_id:
                if not self._is_process_running(pool_info.process_id):
                    pool_info.process_id = None
                    pool_info.status = PoolStatus.STOPPED
            
            # Check network connectivity
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Try health endpoint first
                health_url = f"{pool_info.url}/health"
                try:
                    response = await client.get(health_url)
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        pool_info.status = PoolStatus.HEALTHY
                        pool_info.response_time = response_time
                        pool_info.error_message = None
                        
                        # Parse health response if available
                        try:
                            health_data = response.json()
                            pool_info.version = health_data.get("version")
                            if "capabilities" in health_data:
                                pool_info.capabilities = health_data["capabilities"]
                        except:
                            pass  # Health endpoint doesn't return JSON
                    else:
                        pool_info.status = PoolStatus.UNHEALTHY
                        pool_info.error_message = f"HTTP {response.status_code}"
                        
                except httpx.RequestError:
                    # Health endpoint not available, try basic connectivity
                    try:
                        # Try to connect to the base URL (SSE endpoint)
                        response = await client.get(pool_info.url.rstrip('/') + '/', timeout=5.0)
                        response_time = time.time() - start_time
                        
                        # If we can connect to the base URL, consider it healthy (MCP protocol)
                        if response.status_code in [200, 404, 403]:  # 404/403 is acceptable for MCP endpoints
                            pool_info.status = PoolStatus.HEALTHY
                            pool_info.response_time = response_time
                            pool_info.error_message = None
                            logger.info(f"✅ {pool_name}: MCP endpoint responsive (HTTP {response.status_code})")
                        else:
                            pool_info.status = PoolStatus.UNHEALTHY
                            pool_info.error_message = f"HTTP {response.status_code}"
                            
                    except httpx.RequestError as e:
                        # Check if port is listening (basic network check)
                        if self._is_port_open(pool_info.port):
                            pool_info.status = PoolStatus.HEALTHY
                            pool_info.response_time = time.time() - start_time
                            pool_info.error_message = None
                            logger.info(f"✅ {pool_name}: Port {pool_info.port} listening (MCP service)")
                        else:
                            pool_info.status = PoolStatus.UNHEALTHY
                            pool_info.error_message = f"Connection failed: {str(e)}"
                        
                except httpx.ConnectError:
                    # Try to check if port is open
                    if self._is_port_open(pool_info.port):
                        pool_info.status = PoolStatus.STARTING
                        pool_info.error_message = "Service starting, health endpoint not ready"
                    else:
                        pool_info.status = PoolStatus.STOPPED
                        pool_info.error_message = "Port not accessible"
                        
                except httpx.TimeoutException:
                    pool_info.status = PoolStatus.UNHEALTHY
                    pool_info.error_message = "Health check timeout"
        
        except Exception as e:
            pool_info.status = PoolStatus.ERROR
            pool_info.error_message = str(e)
        
        pool_info.last_check = datetime.now()
        self.pools[pool_name] = pool_info
        
        return pool_info
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is running"""
        try:
            return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
        except:
            return False
    
    def _is_port_open(self, port: int) -> bool:
        """Check if a port is open"""
        try:
            # Use lsof to check if port is in use
            result = subprocess.run(
                ["lsof", "-i", f":{port}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    async def start_agent_pool(self, pool_name: str) -> bool:
        """
        Start a specific agent pool
        
        Args:
            pool_name: Name of the agent pool to start
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        if pool_name not in self.pools:
            logger.error(f"Unknown agent pool: {pool_name}")
            return False
        
        pool_info = self.pools[pool_name]
        
        try:
            # Check if already running
            current_status = await self.check_pool_health(pool_name)
            if current_status.status == PoolStatus.HEALTHY:
                logger.info(f"Agent pool {pool_name} is already running")
                return True
            
            # Start the agent pool process
            pool_config = self.config["agent_pools"][pool_name]
            script_path = pool_config.get("script_path", f"FinAgents/agent_pools/{pool_name}/core.py")
            working_dir = pool_config.get("working_dir", f"FinAgents/agent_pools/{pool_name}")
            
            logger.info(f"Starting agent pool {pool_name} from {script_path}")
            
            # Start process in background
            process = subprocess.Popen(
                ["python", "core.py"],
                cwd=working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            pool_info.process_id = process.pid
            pool_info.status = PoolStatus.STARTING
            
            # Wait for startup
            for i in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                current_status = await self.check_pool_health(pool_name)
                if current_status.status == PoolStatus.HEALTHY:
                    logger.info(f"Agent pool {pool_name} started successfully (PID: {process.pid})")
                    return True
                elif current_status.status == PoolStatus.ERROR:
                    logger.error(f"Agent pool {pool_name} failed to start: {current_status.error_message}")
                    return False
            
            logger.warning(f"Agent pool {pool_name} startup timeout")
            return False
            
        except Exception as e:
            logger.error(f"Error starting agent pool {pool_name}: {e}")
            pool_info.status = PoolStatus.ERROR
            pool_info.error_message = str(e)
            return False
    
    async def stop_agent_pool(self, pool_name: str) -> bool:
        """
        Stop a specific agent pool
        
        Args:
            pool_name: Name of the agent pool to stop
            
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if pool_name not in self.pools:
            logger.error(f"Unknown agent pool: {pool_name}")
            return False
        
        pool_info = self.pools[pool_name]
        
        try:
            if pool_info.process_id:
                # Try graceful shutdown first
                if self._is_process_running(pool_info.process_id):
                    process = psutil.Process(pool_info.process_id)
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    for i in range(10):
                        if not self._is_process_running(pool_info.process_id):
                            break
                        await asyncio.sleep(1)
                    
                    # Force kill if still running
                    if self._is_process_running(pool_info.process_id):
                        process.kill()
                
                pool_info.process_id = None
            
            # Verify it's stopped
            current_status = await self.check_pool_health(pool_name)
            if current_status.status == PoolStatus.STOPPED:
                logger.info(f"Agent pool {pool_name} stopped successfully")
                return True
            else:
                logger.warning(f"Agent pool {pool_name} may still be running")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping agent pool {pool_name}: {e}")
            return False
    
    async def restart_agent_pool(self, pool_name: str) -> bool:
        """Restart a specific agent pool"""
        logger.info(f"Restarting agent pool {pool_name}")
        
        # Stop first
        await self.stop_agent_pool(pool_name)
        await asyncio.sleep(2)  # Brief pause
        
        # Then start
        return await self.start_agent_pool(pool_name)
    
    async def start_all_pools(self) -> Dict[str, bool]:
        """Start all configured agent pools"""
        results = {}
        
        # Start pools in a specific order for dependencies
        startup_order = [
            "data_agent_pool",
            "alpha_agent_pool", 
            "risk_agent_pool",
            "transaction_cost_agent_pool"
        ]
        
        for pool_name in startup_order:
            if pool_name in self.pools:
                logger.info(f"Starting {pool_name}...")
                results[pool_name] = await self.start_agent_pool(pool_name)
                if results[pool_name]:
                    await asyncio.sleep(2)  # Brief pause between starts
                else:
                    logger.error(f"Failed to start {pool_name}, continuing with others...")
        
        return results
    
    def get_system_status_summary(self) -> Dict[str, Any]:
        """Get a summary of system status"""
        healthy_count = sum(1 for pool in self.pools.values() if pool.status == PoolStatus.HEALTHY)
        total_count = len(self.pools)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_pools": total_count,
            "healthy_pools": healthy_count,
            "system_health": "healthy" if healthy_count == total_count else "degraded" if healthy_count > 0 else "critical",
            "pools": {
                name: {
                    "status": pool.status.value,
                    "url": pool.url,
                    "last_check": pool.last_check.isoformat() if pool.last_check else None,
                    "response_time": pool.response_time,
                    "error": pool.error_message,
                    "capabilities": pool.capabilities
                }
                for name, pool in self.pools.items()
            }
        }
    
    async def validate_mcp_connectivity(self, pool_name: str) -> Dict[str, Any]:
        """
        Validate MCP protocol connectivity with an agent pool
        
        Args:
            pool_name: Name of the agent pool to validate
            
        Returns:
            Dict with validation results
        """
        if pool_name not in self.pools:
            return {"success": False, "error": f"Unknown pool: {pool_name}"}
        
        pool_info = self.pools[pool_name]
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Test MCP capabilities endpoint
                capabilities_url = f"{pool_info.url}/mcp/capabilities"
                response = await client.get(capabilities_url)
                
                if response.status_code == 200:
                    capabilities = response.json()
                    
                    # Test a simple MCP tool call
                    tool_test_url = f"{pool_info.url}/mcp/tools"
                    tools_response = await client.get(tool_test_url)
                    
                    if tools_response.status_code == 200:
                        tools = tools_response.json()
                        
                        return {
                            "success": True,
                            "pool_name": pool_name,
                            "mcp_version": capabilities.get("mcp_version", "unknown"),
                            "available_tools": tools.get("tools", []),
                            "capabilities": capabilities,
                            "response_time": response.elapsed.total_seconds()
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"MCP tools endpoint failed: {tools_response.status_code}"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"MCP capabilities endpoint failed: {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"MCP validation failed: {str(e)}"
            }

# Demo and testing functionality
async def main():
    """Test the agent pool monitor"""
    
    # Sample configuration
    config = {
        "agent_pools": {
            "data_agent_pool": {
                "url": "http://localhost:8001",
                "enabled": True,
                "script_path": "FinAgents/agent_pools/data_agent_pool/core.py",
                "working_dir": "FinAgents/agent_pools/data_agent_pool",
                "capabilities": ["market_data_fetch", "news_data_fetch"]
            },
            "alpha_agent_pool": {
                "url": "http://localhost:5050",
                "enabled": True,
                "script_path": "FinAgents/agent_pools/alpha_agent_pool/core.py",
                "working_dir": "FinAgents/agent_pools/alpha_agent_pool",
                "capabilities": ["signal_generation", "strategy_development"]
            },
            "risk_agent_pool": {
                "url": "http://localhost:7000",
                "enabled": True,
                "script_path": "FinAgents/agent_pools/risk_agent_pool/core.py",
                "working_dir": "FinAgents/agent_pools/risk_agent_pool",
                "capabilities": ["risk_assessment", "portfolio_optimization"]
            }
        },
        "health_check_interval": 30
    }
    
    # Initialize monitor
    monitor = AgentPoolMonitor(config)
    
    print("🔍 Agent Pool Health Monitor Demo")
    print("=" * 50)
    
    # Check all pools
    print("\n📊 Checking all agent pools...")
    results = await monitor.check_all_pools()
    
    for pool_name, pool_info in results.items():
        status_icon = "✅" if pool_info.status == PoolStatus.HEALTHY else "❌"
        print(f"{status_icon} {pool_name}: {pool_info.status.value}")
        if pool_info.error_message:
            print(f"   Error: {pool_info.error_message}")
        if pool_info.response_time:
            print(f"   Response time: {pool_info.response_time:.3f}s")
    
    # System status summary
    print("\n📈 System Status Summary:")
    summary = monitor.get_system_status_summary()
    print(f"   Total pools: {summary['total_pools']}")
    print(f"   Healthy pools: {summary['healthy_pools']}")
    print(f"   System health: {summary['system_health']}")
    
    # Test MCP connectivity for healthy pools
    print("\n🔗 Testing MCP Connectivity...")
    for pool_name, pool_info in results.items():
        if pool_info.status == PoolStatus.HEALTHY:
            mcp_result = await monitor.validate_mcp_connectivity(pool_name)
            if mcp_result["success"]:
                print(f"✅ {pool_name}: MCP connectivity OK")
                print(f"   Available tools: {len(mcp_result.get('available_tools', []))}")
            else:
                print(f"❌ {pool_name}: MCP connectivity failed - {mcp_result['error']}")
    
    print("\n✅ Agent pool monitoring demo completed!")

if __name__ == "__main__":
    asyncio.run(main())
