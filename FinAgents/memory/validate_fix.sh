#!/bin/bash

# Quick validation script for the fixed LLM research service command
echo "🧪 Validating Fixed LLM Research Command"
echo "========================================"

# Set up environment
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agent

echo "✅ Environment activated"
echo "📁 Current directory: $(pwd)"

# Test the fixed Python command
echo ""
echo "🔬 Testing Python command:"
echo "PYTHONPATH=/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration:\$PYTHONPATH python -c \"from FinAgents.memory.llm_research_service import llm_research_service; import asyncio; asyncio.run(llm_research_service.analyze_memory_patterns([]))\""

echo ""
echo "⚡ Executing command..."
PYTHONPATH=/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration:$PYTHONPATH python -c "
try:
    from FinAgents.memory.llm_research_service import llm_research_service
    import asyncio
    print('✅ Import successful')
    print('🧠 Running analyze_memory_patterns...')
    asyncio.run(llm_research_service.analyze_memory_patterns([]))
    print('✅ Command executed successfully')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo ""
echo "📝 Log files location:"
ls -la logs/

echo ""
echo "🎉 Validation complete!"
