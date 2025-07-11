#!/usr/bin/env python3
"""
Alpha Agent Pool 功能清单工具
快速连接并列出Alpha Agent Pool的所有可用功能
"""

import asyncio
import json
import sys
import logging
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# 配置简单日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def list_alpha_pool_functions(endpoint: str = "http://localhost:8081/sse"):
    """列出Alpha Agent Pool的所有功能"""
    
    logger.info("🔍 Alpha Agent Pool 功能清单")
    logger.info("=" * 50)
    
    try:
        # 连接到服务器
        logger.info(f"正在连接到: {endpoint}")
        
        async with sse_client(endpoint, timeout=10) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("✅ 连接成功")
                
                # 获取所有可用工具
                tools = await session.list_tools()
                
                logger.info(f"\n📋 发现 {len(tools.tools)} 个可用功能:")
                logger.info("-" * 50)
                
                # 按功能分类
                function_categories = {
                    "内存管理": [],
                    "代理管理": [],
                    "信号生成": [],
                    "策略处理": [],
                    "数据检索": [],
                    "性能分析": [],
                    "系统交互": []
                }
                
                for i, tool in enumerate(tools.tools, 1):
                    # 分类工具
                    if "memory" in tool.name.lower():
                        category = "内存管理"
                    elif "agent" in tool.name.lower():
                        category = "代理管理"
                    elif "signal" in tool.name.lower() or "alpha" in tool.name.lower():
                        category = "信号生成"
                    elif "strategy" in tool.name.lower() and ("request" in tool.name.lower() or "process" in tool.name.lower()):
                        category = "策略处理"
                    elif "retrieve" in tool.name.lower() or "data" in tool.name.lower():
                        category = "数据检索"
                    elif "analyze" in tool.name.lower() or "performance" in tool.name.lower():
                        category = "性能分析"
                    else:
                        category = "系统交互"
                    
                    function_categories[category].append({
                        "name": tool.name,
                        "description": tool.description,
                        "index": i
                    })
                
                # 打印分类后的功能
                for category, functions in function_categories.items():
                    if functions:
                        logger.info(f"\n🔧 {category}:")
                        for func in functions:
                            logger.info(f"  {func['index']:2d}. {func['name']}")
                            logger.info(f"      📝 {func['description']}")
                
                # 功能总结
                logger.info("\n" + "=" * 50)
                logger.info("📊 功能总结:")
                logger.info(f"  🧠 内存管理: {len(function_categories['内存管理'])} 个功能")
                logger.info(f"  🤖 代理管理: {len(function_categories['代理管理'])} 个功能")
                logger.info(f"  📈 信号生成: {len(function_categories['信号生成'])} 个功能")
                logger.info(f"  🎯 策略处理: {len(function_categories['策略处理'])} 个功能")
                logger.info(f"  🔍 数据检索: {len(function_categories['数据检索'])} 个功能")
                logger.info(f"  📊 性能分析: {len(function_categories['性能分析'])} 个功能")
                logger.info(f"  🔗 系统交互: {len(function_categories['系统交互'])} 个功能")
                logger.info(f"  📋 总计: {len(tools.tools)} 个MCP工具")
                
                # 核心能力说明
                logger.info("\n🎯 Alpha Agent Pool 核心能力:")
                logger.info("  • 生成基于动量的alpha交易信号")
                logger.info("  • 管理多个子代理(momentum, autonomous)")
                logger.info("  • 处理自然语言策略查询")
                logger.info("  • 记录和追踪策略事件")
                logger.info("  • 分析策略历史性能")
                logger.info("  • 提供内存存储和检索")
                logger.info("  • 支持与orchestrator的交互")
                
                return tools.tools
                
    except Exception as e:
        logger.error(f"❌ 连接失败: {e}")
        logger.error("请确保Alpha Agent Pool服务器正在运行:")
        logger.error("  python3 core.py")
        return []

async def quick_functionality_test():
    """快速功能测试"""
    logger.info("\n🧪 快速功能测试")
    logger.info("-" * 30)
    
    endpoint = "http://localhost:8081/sse"
    
    try:
        async with sse_client(endpoint, timeout=5) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 测试基本功能
                logger.info("1. 测试内存功能...")
                result = await session.call_tool("set_memory", {"key": "test", "value": "ok"})
                logger.info(f"   ✅ 设置内存: {result.content[0].text if result.content else 'Success'}")
                
                result = await session.call_tool("get_memory", {"key": "test"})
                logger.info(f"   ✅ 获取内存: {result.content[0].text if result.content else 'None'}")
                
                logger.info("2. 测试代理列表...")
                result = await session.call_tool("list_agents", {})
                agents = json.loads(result.content[0].text) if result.content else []
                logger.info(f"   ✅ 当前代理: {agents}")
                
                logger.info("3. 测试信号生成...")
                result = await session.call_tool("generate_alpha_signals", {
                    "symbols": ["AAPL"],
                    "date": "2024-01-15",
                    "lookback_period": 10
                })
                if result.content:
                    signals = json.loads(result.content[0].text)
                    status = signals.get('status', 'unknown')
                    logger.info(f"   ✅ 信号生成状态: {status}")
                
                logger.info("✅ 快速测试完成")
                
    except Exception as e:
        logger.error(f"❌ 快速测试失败: {e}")

async def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 运行快速测试
        await quick_functionality_test()
    else:
        # 只列出功能
        await list_alpha_pool_functions()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 已取消")
    except Exception as e:
        logger.error(f"❌ 运行错误: {e}")
