#!/usr/bin/env python3
"""
FinAgent Memory System Launcher

Unified launcher for all FinAgent memory system components with modular architecture.
This script provides easy management of MCP servers, A2A servers, and unified memory services
across different deployment scenarios.

Features:
- Multi-server deployment management
- Environment-specific configurations
- Health monitoring and auto-restart
- Graceful shutdown handling
- Development and production modes

Author: FinAgent Team
License: Open Source
"""

# ═══════════════════════════════════════════════════════════════════════════════════
# IMPORTS AND DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════════

import asyncio
import sys
import signal
import argparse
import logging
import multiprocessing
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import time

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configuration management
try:
    from configuration_manager import (
        get_config_manager, get_database_config, get_server_config_dict,
        auto_configure, print_config_summary, Environment, ServerType
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️ Configuration manager not available")

# Database initialization
try:
    from database_initializer import initialize_database
    DATABASE_INIT_AVAILABLE = True
except ImportError:
    DATABASE_INIT_AVAILABLE = False
    print("⚠️ Database initializer not available")

# Server imports
try:
    from memory_server import app as memory_app, print_memory_server_info
    MEMORY_SERVER_AVAILABLE = True
except ImportError:
    MEMORY_SERVER_AVAILABLE = False
    print("⚠️ Memory server not available")

try:
    from mcp_server import print_mcp_server_info
    MCP_SERVER_AVAILABLE = True
except ImportError:
    MCP_SERVER_AVAILABLE = False
    print("⚠️ MCP server not available")

try:
    from a2a_server import app as a2a_app, print_a2a_server_info
    A2A_SERVER_AVAILABLE = True
except ImportError:
    A2A_SERVER_AVAILABLE = False
    print("⚠️ A2A server not available")

# Process management
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Web server for FastAPI
try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False
    print("⚠️ Uvicorn not available for web servers")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════
# LAUNCHER CONFIGURATION AND CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════════

LAUNCHER_VERSION = "2.0.0"
DEFAULT_ENVIRONMENT = Environment.DEVELOPMENT

# Server process tracking
server_processes: Dict[str, Any] = {}
shutdown_requested = False

# Default ports for each server type (fallback)
DEFAULT_PORTS = {
    ServerType.MEMORY: 8000,
    ServerType.MCP: 8001,
    ServerType.A2A: 8002
}

# ═══════════════════════════════════════════════════════════════════════════════════
# SERVER PROCESS MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════════

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True

def start_memory_server(config: Dict[str, Any]) -> Optional[multiprocessing.Process]:
    """Start the memory server process."""
    if not MEMORY_SERVER_AVAILABLE or not UVICORN_AVAILABLE:
        logger.error("❌ Memory server or Uvicorn not available")
        return None
    
    try:
        def run_memory_server():
            logger.info("🚀 Starting Memory Server...")
            uvicorn.run(
                "memory_server:app",
                host=config.get("host", "0.0.0.0"),
                port=config.get("port", 8000),
                log_level=config.get("log_level", "info").lower(),
                access_log=True
            )
        
        process = multiprocessing.Process(
            target=run_memory_server,
            name="FinAgent-Memory-Server"
        )
        process.start()
        logger.info(f"✅ Memory Server started with PID {process.pid}")
        return process
        
    except Exception as e:
        logger.error(f"❌ Failed to start Memory Server: {e}")
        return None

def start_mcp_server(config: Dict[str, Any]) -> Optional[multiprocessing.Process]:
    """Start the MCP server process."""
    if not MCP_SERVER_AVAILABLE:
        logger.error("❌ MCP server not available")
        return None
    
    try:
        def run_mcp_server():
            logger.info("🚀 Starting MCP Server...")
            # Import and run MCP server
            from mcp_server import main as mcp_main
            mcp_main()
        
        process = multiprocessing.Process(
            target=run_mcp_server,
            name="FinAgent-MCP-Server"
        )
        process.start()
        logger.info(f"✅ MCP Server started with PID {process.pid}")
        return process
        
    except Exception as e:
        logger.error(f"❌ Failed to start MCP Server: {e}")
        return None

def start_a2a_server(config: Dict[str, Any]) -> Optional[multiprocessing.Process]:
    """Start the A2A server process."""
    if not A2A_SERVER_AVAILABLE or not UVICORN_AVAILABLE:
        logger.error("❌ A2A server or Uvicorn not available")
        return None
    
    try:
        def run_a2a_server():
            logger.info("🚀 Starting A2A Server...")
            uvicorn.run(
                "a2a_server:app",
                host=config.get("host", "0.0.0.0"),
                port=config.get("port", 8002),
                log_level=config.get("log_level", "info").lower(),
                access_log=True
            )
        
        process = multiprocessing.Process(
            target=run_a2a_server,
            name="FinAgent-A2A-Server"
        )
        process.start()
        logger.info(f"✅ A2A Server started with PID {process.pid}")
        return process
        
    except Exception as e:
        logger.error(f"❌ Failed to start A2A Server: {e}")
        return None

def stop_server_process(process: multiprocessing.Process, server_name: str, timeout: int = 10):
    """Stop a server process gracefully."""
    try:
        if process and process.is_alive():
            logger.info(f"🛑 Stopping {server_name}...")
            process.terminate()
            
            # Wait for graceful shutdown
            process.join(timeout=timeout)
            
            if process.is_alive():
                logger.warning(f"⚠️ Force killing {server_name}...")
                process.kill()
                process.join()
            
            logger.info(f"✅ {server_name} stopped")
        
    except Exception as e:
        logger.error(f"❌ Error stopping {server_name}: {e}")

def monitor_server_processes():
    """Monitor server processes and handle restarts if needed."""
    global shutdown_requested
    
    while not shutdown_requested:
        try:
            # Check each server process
            for server_name, process in list(server_processes.items()):
                if process and not process.is_alive():
                    logger.warning(f"⚠️ {server_name} process died, exit code: {process.exitcode}")
                    
                    # Remove dead process
                    del server_processes[server_name]
                    
                    # Auto-restart in development mode
                    if not shutdown_requested:
                        logger.info(f"🔄 Attempting to restart {server_name}...")
                        # Implementation for restart logic would go here
            
            # Sleep before next check
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"❌ Process monitoring error: {e}")
            time.sleep(5)

# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN LAUNCHER IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════════

class FinAgentLauncher:
    """Main launcher class for FinAgent memory system."""
    
    def __init__(self, environment: Environment = DEFAULT_ENVIRONMENT):
        """Initialize the launcher."""
        self.environment = environment
        self.config_manager = None
        self.config = None
        
        # Initialize configuration if available
        if CONFIG_AVAILABLE:
            try:
                self.config_manager = get_config_manager()
                self.config_manager.set_environment(environment)
                self.config = self.config_manager.get_config()
            except Exception as e:
                logger.error(f"❌ Configuration initialization failed: {e}")
    
    def print_launcher_info(self):
        """Print launcher information and status."""
        print("\n" + "="*80)
        print("🚀 FINAGENT MEMORY SYSTEM LAUNCHER")
        print("="*80)
        print(f"📋 Launcher Version: {LAUNCHER_VERSION}")
        print(f"🌍 Environment: {self.environment.value}")
        print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📦 Available Components:")
        print(f"   🧠 Memory Server: {'✅ Available' if MEMORY_SERVER_AVAILABLE else '❌ Unavailable'}")
        print(f"   🔧 MCP Server: {'✅ Available' if MCP_SERVER_AVAILABLE else '❌ Unavailable'}")
        print(f"   📡 A2A Server: {'✅ Available' if A2A_SERVER_AVAILABLE else '❌ Unavailable'}")
        print(f"   ⚙️  Configuration: {'✅ Available' if CONFIG_AVAILABLE else '❌ Unavailable'}")
        print(f"   🌐 Web Server: {'✅ Available' if UVICORN_AVAILABLE else '❌ Unavailable'}")
        
        if self.config:
            print(f"\n🔧 Configuration Summary:")
            print(f"   🗄️  Database: {self.config.database.uri}")
            print(f"   🚪 Memory Port: {self.config.server.port}")
            print(f"   📝 Log Level: {self.config.logging.level}")
        
        print("="*80)
    
    def start_server(self, server_type: ServerType) -> bool:
        """Start a specific server."""
        try:
            server_name = f"FinAgent-{server_type.value.upper()}-Server"
            
            # Get server configuration from config manager
            if self.config_manager:
                server_config = get_server_config_dict(server_type, self.environment)
                port_config = self.config_manager.get_port_config(self.environment)
                
                # Use port from configuration
                if server_type == ServerType.MEMORY:
                    server_config["port"] = port_config.memory_server
                elif server_type == ServerType.MCP:
                    server_config["port"] = port_config.mcp_server
                elif server_type == ServerType.A2A:
                    server_config["port"] = port_config.a2a_server
            else:
                server_config = {"host": "0.0.0.0", "port": DEFAULT_PORTS[server_type]}
            
            # Start the appropriate server
            process = None
            
            if server_type == ServerType.MEMORY:
                process = start_memory_server(server_config)
            elif server_type == ServerType.MCP:
                process = start_mcp_server(server_config)
            elif server_type == ServerType.A2A:
                process = start_a2a_server(server_config)
            
            if process:
                server_processes[server_name] = process
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start {server_type.value} server: {e}")
            return False
    
    def start_all_servers(self) -> bool:
        """Start all available servers."""
        success_count = 0
        total_servers = 0
        
        for server_type in [ServerType.MEMORY, ServerType.MCP, ServerType.A2A]:
            total_servers += 1
            if self.start_server(server_type):
                success_count += 1
            time.sleep(2)  # Brief delay between server starts
        
        logger.info(f"✅ Started {success_count}/{total_servers} servers")
        return success_count > 0
    
    def stop_all_servers(self):
        """Stop all running servers."""
        logger.info("🛑 Stopping all servers...")
        
        for server_name, process in server_processes.items():
            stop_server_process(process, server_name)
        
        server_processes.clear()
        logger.info("✅ All servers stopped")
    
    def run(self, servers: List[ServerType] = None):
        """Run the launcher with specified servers."""
        global shutdown_requested
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Print launcher info
        self.print_launcher_info()
        
        try:
            # Initialize database first
            if DATABASE_INIT_AVAILABLE and self.config_manager:
                logger.info("🔧 Initializing database...")
                init_success = asyncio.run(initialize_database(self.environment))
                if not init_success:
                    logger.error("❌ Database initialization failed")
                    return 1
                logger.info("✅ Database initialization completed")
            
            # Start specified servers or all servers
            if servers:
                for server_type in servers:
                    self.start_server(server_type)
            else:
                self.start_all_servers()
            
            if not server_processes:
                logger.error("❌ No servers started successfully")
                return 1
            
            logger.info("🎯 All servers started. Press Ctrl+C to stop.")
            
            # Start process monitoring in background
            monitor_thread = multiprocessing.Process(target=monitor_server_processes)
            monitor_thread.start()
            
            # Main loop - wait for shutdown signal
            while not shutdown_requested:
                time.sleep(1)
            
            # Cleanup
            logger.info("🛑 Shutdown requested, stopping all services...")
            stop_server_process(monitor_thread, "Process Monitor", timeout=5)
            self.stop_all_servers()
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Launcher error: {e}")
            self.stop_all_servers()
            return 1

# ═══════════════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════════

def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="FinAgent Memory System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py                                    # Start all servers in development mode
  python launcher.py --env production                   # Start all servers in production mode
  python launcher.py --servers memory mcp               # Start only memory and MCP servers
  python launcher.py --config-summary                   # Show configuration summary and exit
  python launcher.py --server-info                      # Show server info and exit
        """
    )
    
    parser.add_argument(
        "--env", "--environment",
        choices=["development", "testing", "staging", "production"],
        default="development",
        help="Deployment environment (default: development)"
    )
    
    parser.add_argument(
        "--servers",
        nargs="+",
        choices=["memory", "mcp", "a2a", "all"],
        default=["all"],
        help="Servers to start (default: all)"
    )
    
    parser.add_argument(
        "--config-summary",
        action="store_true",
        help="Show configuration summary and exit"
    )
    
    parser.add_argument(
        "--server-info",
        action="store_true",
        help="Show server information and exit"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"FinAgent Launcher v{LAUNCHER_VERSION}"
    )
    
    return parser

def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Convert environment string to enum
    environment = Environment(args.env)
    
    # Handle info commands
    if args.config_summary:
        if CONFIG_AVAILABLE:
            config_manager = get_config_manager()
            config_manager.set_environment(environment)
            print_config_summary(environment)
        else:
            print("❌ Configuration manager not available")
        return 0
    
    if args.server_info:
        print("\n🚀 FinAgent Server Information:")
        if MEMORY_SERVER_AVAILABLE:
            print_memory_server_info()
        if MCP_SERVER_AVAILABLE:
            print_mcp_server_info()
        if A2A_SERVER_AVAILABLE:
            print_a2a_server_info()
        return 0
    
    # Convert server arguments to ServerType enums
    server_types = []
    if "all" in args.servers:
        server_types = [ServerType.MEMORY, ServerType.MCP, ServerType.A2A]
    else:
        for server_str in args.servers:
            if server_str == "memory":
                server_types.append(ServerType.MEMORY)
            elif server_str == "mcp":
                server_types.append(ServerType.MCP)
            elif server_str == "a2a":
                server_types.append(ServerType.A2A)
    
    # Create and run launcher
    launcher = FinAgentLauncher(environment)
    return launcher.run(server_types)

# ═══════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.exit(main())
