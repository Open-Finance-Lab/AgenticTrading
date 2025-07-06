#!/bin/bash
# 测试启动单个FinAgent Pool服务器

echo "🧪 测试启动单个FinAgent服务"
echo "================================"

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "📁 项目根目录: ${PROJECT_ROOT}"

# 设置PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
echo "🐍 PYTHONPATH设置为: ${PYTHONPATH}"

# 创建PID文件目录
mkdir -p "${PROJECT_ROOT}/logs"

# 切换到项目根目录
cd "${PROJECT_ROOT}"

echo "🔧 测试启动 data_agent_pool (端口 8001)..."

# 检查端口是否被占用
if lsof -i:8001 &> /dev/null; then
    echo "⚠️ 端口 8001 已被占用"
    exit 1
fi

# 测试模块导入
echo "🔍 测试模块导入..."
if ! python -c "from FinAgents.agent_pools.data_agent_pool import core; print('模块导入成功')" 2>/dev/null; then
    echo "❌ 模块导入失败"
    exit 1
fi

echo "✅ 模块导入测试通过"

# 启动服务
echo "🚀 启动服务..."
python -m FinAgents.agent_pools.data_agent_pool.core &
pid=$!
echo "服务PID: ${pid}"

# 等待服务启动
sleep 5

# 检查服务状态
if kill -0 ${pid} 2>/dev/null; then
    echo "✅ 服务启动成功"
    echo "🛑 停止测试服务..."
    kill ${pid}
    echo "✅ 测试完成"
else
    echo "❌ 服务启动失败"
    exit 1
fi
