#!/bin/bash
# 修复Agent Pool导入问题的脚本

echo "🔧 修复FinAgent Pool导入问题..."

# 1. 修复Alpha Agent Pool
echo "📝 修复Alpha Agent Pool导入..."
sed -i '' 's|from schema\.theory_driven_schema|from FinAgents.agent_pools.alpha_agent_pool.schema.theory_driven_schema|g' FinAgents/agent_pools/alpha_agent_pool/core.py
sed -i '' 's|from agents\.theory_driven\.momentum_agent|from FinAgents.agent_pools.alpha_agent_pool.agents.theory_driven.momentum_agent|g' FinAgents/agent_pools/alpha_agent_pool/core.py
sed -i '' 's|from agents\.autonomous\.autonomous_agent|from FinAgents.agent_pools.alpha_agent_pool.agents.autonomous.autonomous_agent|g' FinAgents/agent_pools/alpha_agent_pool/core.py

# 2. 修复Portfolio Agent Pool  
echo "📝 修复Portfolio Agent Pool导入..."
sed -i '' 's|from \.memory_bridge|from FinAgents.agent_pools.portfolio_construction_agent_pool.memory_bridge|g' FinAgents/agent_pools/portfolio_construction_agent_pool/core.py
sed -i '' 's|from \.registry|from FinAgents.agent_pools.portfolio_construction_agent_pool.registry|g' FinAgents/agent_pools/portfolio_construction_agent_pool/core.py

# 3. 修复Risk Agent Pool
echo "📝 修复Risk Agent Pool导入..."
sed -i '' 's|from \.registry|from FinAgents.agent_pools.risk_agent_pool.registry|g' FinAgents/agent_pools/risk_agent_pool/core.py
sed -i '' 's|from \.memory_bridge|from FinAgents.agent_pools.risk_agent_pool.memory_bridge|g' FinAgents/agent_pools/risk_agent_pool/core.py

echo "✅ 导入修复完成！"
echo ""
echo "🚀 现在重新启动agent pools..."
