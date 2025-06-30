#!/bin/bash
"""
启动和运行 FinAgent Orchestrator Chatbot 回测

这个脚本演示了如何使用 FinAgent orchestrator 通过自然语言指令执行全面的多agent回测。
包括以下agent pools的协调：
- Data Agent Pool: 市场数据获取
- Alpha Agent Pool: 信号生成和策略执行  
- Portfolio Construction Agent Pool: 投资组合优化
- Transaction Cost Agent Pool: 成本分析和优化
- Risk Agent Pool: 风险管理和监控

该测试模拟chatbot对话，用户可以使用自然语言请求回测，如：
"为AAPL和MSFT运行3年动量回测"
"""

echo "🚀 启动 FinAgent Orchestrator Chatbot 回测"
echo "==============================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未找到，请安装Python"
    exit 1
fi

# 检查是否在正确的目录
if [ ! -f "tests/test_orchestrator_chatbot_backtest.py" ]; then
    echo "❌ 请在FinAgent-Orchestration项目根目录下运行此脚本"
    exit 1
fi

# 检查.env文件
if [ ! -f ".env" ]; then
    echo "⚠️ .env文件未找到，某些功能可能受限"
fi

echo "📋 运行前检查："
echo "   ✅ Python环境"
echo "   ✅ 项目结构"
echo "   ✅ 测试文件"

echo ""
echo "🤖 启动Orchestrator Chatbot回测..."
echo "   📊 Data Agent Pool: localhost:8001"
echo "   🧠 Alpha Agent Pool: localhost:5050" 
echo "   📈 Portfolio Agent Pool: localhost:8002"
echo "   💰 Transaction Cost Agent Pool: localhost:6000"
echo "   🛡️ Risk Agent Pool: localhost:7000"
echo ""
echo "注意：如果agent pools未运行，系统将使用mock数据继续测试"
echo ""

# 运行测试
python tests/test_orchestrator_chatbot_backtest.py

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Orchestrator Chatbot回测成功完成！"
    echo ""
    echo "📋 测试展示了："
    echo "   💬 自然语言chatbot对话接口"
    echo "   🎯 Multi-agent orchestration协调"
    echo "   📊 综合回测模拟（AAPL和MSFT）"
    echo "   📈 性能指标和分析"
    echo "   🔍 Agent pool健康检查和故障转移"
    echo ""
    echo "查看日志文件获取详细信息："
    echo "   📄 orchestrator_chatbot_backtest_*.log"
else
    echo ""
    echo "❌ 测试执行失败，请检查日志获取详细错误信息"
    exit 1
fi
