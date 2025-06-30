#!/usr/bin/env python3
"""
启动Data Agent Pool服务器用于测试

这个脚本启动Data Agent Pool MCP服务器，使得测试可以通过MCP客户端获取真实的市场数据。
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Data Agent Pool
from FinAgents.agent_pools.data_agent_pool.core_new import DataAgentPoolMCPServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataAgentPoolStarter")

async def start_data_agent_pool():
    """启动Data Agent Pool服务器"""
    logger.info("🚀 启动Data Agent Pool服务器...")
    
    try:
        # 创建服务器实例
        server = DataAgentPoolMCPServer(host="0.0.0.0", port=8001)
        
        # 启动服务器
        logger.info("🔧 正在启动MCP服务器在端口8001...")
        await server.pool_server.run()
        
    except Exception as e:
        logger.error(f"❌ 启动服务器失败: {e}")
        raise

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("🏁 Data Agent Pool服务器启动器")
    logger.info("=" * 60)
    logger.info("📊 此服务器将为测试提供真实的市场数据")
    logger.info("🔗 服务器地址: http://localhost:8001/sse")
    logger.info("📈 支持的符号: AAPL, MSFT")
    logger.info("🗂️ 数据源: Polygon.io")
    logger.info("=" * 60)
    
    try:
        asyncio.run(start_data_agent_pool())
    except KeyboardInterrupt:
        logger.info("👋 收到中断信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"❌ 服务器运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
