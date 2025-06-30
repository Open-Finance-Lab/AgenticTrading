#!/bin/bash
# 一键运行真实数据回测
# 此脚本将启动Data Agent Pool服务器并运行回测测试

echo "🚀 启动真实数据LLM增强回测"
echo "================================"

# 检查环境变量
if [ -z "$POLYGON_API_KEY" ]; then
    echo "❌ 错误: POLYGON_API_KEY环境变量未设置"
    echo "   请运行: export POLYGON_API_KEY='your-api-key'"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 错误: OPENAI_API_KEY环境变量未设置"
    echo "   请运行: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

echo "✅ 环境变量检查通过"

# 启动Data Agent Pool服务器（后台运行）
echo "🔧 启动Data Agent Pool服务器..."
python start_data_agent_pool.py &
DATA_AGENT_PID=$!

# 等待服务器启动
echo "⏳ 等待服务器启动（5秒）..."
sleep 5

# 检查服务器是否仍在运行
if ! kill -0 $DATA_AGENT_PID 2>/dev/null; then
    echo "❌ Data Agent Pool服务器启动失败"
    exit 1
fi

echo "✅ Data Agent Pool服务器启动成功 (PID: $DATA_AGENT_PID)"

# 运行回测测试
echo "📊 运行真实数据回测测试..."
python tests/test_simple_llm_backtest.py

# 获取测试退出码
TEST_EXIT_CODE=$?

# 清理：停止Data Agent Pool服务器
echo "🧹 清理：停止Data Agent Pool服务器..."
kill $DATA_AGENT_PID 2>/dev/null
wait $DATA_AGENT_PID 2>/dev/null

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ 回测测试成功完成"
else
    echo "❌ 回测测试失败"
fi

exit $TEST_EXIT_CODE
