#!/bin/bash
# 停止所有FinAgent Pool服务器

echo "🛑 停止FinAgent生态系统"
echo "=========================="

# 函数：停止agent pool
stop_agent_pool() {
    local name=$1
    local pid_file="logs/${name}.pid"
    
    if [ -f "${pid_file}" ]; then
        local pid=$(cat ${pid_file})
        if kill -0 ${pid} 2>/dev/null; then
            echo "🛑 停止 ${name} (PID: ${pid})..."
            kill ${pid}
            
            # 等待进程结束
            for i in {1..10}; do
                if ! kill -0 ${pid} 2>/dev/null; then
                    echo "✅ ${name} 已停止"
                    break
                fi
                sleep 1
            done
            
            # 如果进程仍在运行，强制杀死
            if kill -0 ${pid} 2>/dev/null; then
                echo "⚠️ 强制停止 ${name}..."
                kill -9 ${pid}
                echo "✅ ${name} 已强制停止"
            fi
        else
            echo "⚠️ ${name} 进程不存在"
        fi
        
        # 删除PID文件
        rm -f ${pid_file}
    else
        echo "⚠️ ${name} PID文件不存在"
    fi
}

# 停止所有agent pools
stop_agent_pool "data_agent_pool"
stop_agent_pool "alpha_agent_pool"
stop_agent_pool "portfolio_agent_pool"
stop_agent_pool "transaction_cost_agent_pool"
stop_agent_pool "risk_agent_pool"

echo ""
echo "🧹 清理临时文件..."
rm -f logs/*.pid

echo ""
echo "✅ 所有FinAgent服务已停止"
