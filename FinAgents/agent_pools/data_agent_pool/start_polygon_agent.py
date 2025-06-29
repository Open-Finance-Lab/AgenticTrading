#!/usr/bin/env python3
"""
Standalone Polygon Agent MCP Server

This script starts the PolygonAgent as an independent MCP server that provides
natural language interface for market data queries.

Usage:
    python start_polygon_agent.py [--port 8002] [--host 0.0.0.0]

The server exposes the following tools:
- process_market_query: Natural language market data queries
- fetch_market_data: Direct market data fetching
- get_company_info: Company information lookup
- health_check: Agent health status
"""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from FinAgents.agent_pools.data_agent_pool.agents.equity.polygon_agent import PolygonAgent
from FinAgents.agent_pools.data_agent_pool.schema.equity_schema import PolygonConfig


def load_polygon_config():
    """Load polygon agent configuration from YAML file."""
    config_path = project_root / "FinAgents" / "agent_pools" / "data_agent_pool" / "config" / "polygon.yaml"
    
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        # Create a minimal config if none exists
        config_data = {
            "agent_id": "polygon_agent",
            "api": {
                "api_key": os.getenv("POLYGON_API_KEY", "demo_key_for_testing"),
                "base_url": "https://api.polygon.io"
            },
            "llm_enabled": False
        }
        return PolygonConfig(**config_data)
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        print(f"Loaded config from {config_path}")
        
        # Convert the YAML structure to match PolygonConfig expected format
        formatted_config = {
            "agent_id": config_data.get("agent_id", "polygon_agent"),
            "api": {
                "base_url": config_data.get("api", {}).get("base_url", "https://api.polygon.io"),
                "endpoints": config_data.get("api", {}).get("endpoints", {
                    "ohlcv": "/v2/aggs/ticker/{symbol}/range",
                    "company_info": "/v3/reference/tickers/{symbol}"
                }),
                "default_interval": config_data.get("api", {}).get("default_interval", "1d")
            },
            "authentication": {
                "api_key": config_data.get("authentication", {}).get("api_key", "demo"),
                "secret_key": config_data.get("authentication", {}).get("secret_key", "")
            },
            "constraints": {
                "timeout": config_data.get("constraints", {}).get("timeout", 30),
                "rate_limit_per_minute": config_data.get("constraints", {}).get("rate_limit_per_minute", 120)
            },
            "llm_enabled": config_data.get("llm_enabled", False)
        }
        
        print(f"API Key configured: {'Yes' if formatted_config['authentication']['api_key'] != 'demo' else 'No (using demo)'}")
        print(f"LLM enabled: {formatted_config['llm_enabled']}")
        
        return PolygonConfig(**formatted_config)
        
    except Exception as e:
        print(f"Error loading config file: {e}")
        # Fallback to default config
        config_data = {
            "agent_id": "polygon_agent", 
            "api": {
                "base_url": "https://api.polygon.io",
                "endpoints": {
                    "ohlcv": "/v2/aggs/ticker/{symbol}/range",
                    "company_info": "/v3/reference/tickers/{symbol}"
                },
                "default_interval": "1d"
            },
            "authentication": {
                "api_key": os.getenv("POLYGON_API_KEY", "demo_key_for_testing"),
                "secret_key": ""
            },
            "constraints": {
                "timeout": 30,
                "rate_limit_per_minute": 120
            },
            "llm_enabled": False
        }
        return PolygonConfig(**config_data)


def main():
    parser = argparse.ArgumentParser(description="Start PolygonAgent MCP Server")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind the server to (default: 8002)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--transport", type=str, default="sse", choices=["http", "sse", "stdio"], 
                        help="Transport protocol (default: sse)")
    
    args = parser.parse_args()
    
    print("Loading PolygonAgent configuration...")
    config = load_polygon_config()
    
    print("Initializing PolygonAgent...")
    agent = PolygonAgent(config)
    
    print(f"Starting PolygonAgent MCP server on {args.host}:{args.port}")
    print(f"Transport: {args.transport}")
    print("Press Ctrl+C to stop the server")
    
    try:
        agent.start_mcp_server(port=args.port, host=args.host, transport=args.transport)
    except KeyboardInterrupt:
        print("\nShutting down PolygonAgent MCP server...")
    except Exception as e:
        print(f"Error starting PolygonAgent MCP server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
