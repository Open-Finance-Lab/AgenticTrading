# test_fetch_tool.py

from agent_pools.data_agent_pool.core import DataAgentPool
from agent_pools.data_agent_pool.tools.fetch_tool import fetch_ticker_data_tool

# 实例化 Agent Pool（内部已包含 agent 调用机制）
pool = DataAgentPool()

# 构建 LangGraph Tool
tool = fetch_ticker_data_tool(pool)

# 构造输入测试样例
input_data = {
    "symbol": "AAPL",
    "start": "2024-02-01",
    "end": "2024-02-02",
    "interval": "1h"
}

# 执行 tool.func
output = tool.func(input_data)
print("✅ Tool Output:", output)