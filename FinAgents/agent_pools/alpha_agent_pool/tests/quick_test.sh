#!/bin/bash

# Quick test - only process first 5 rows of data
echo "🚀 Quick strategy signal generation test (process only a few data points)..."

cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/tests

# Create a small test data file
TEST_DATA="/tmp/test_aapl_small.csv"
echo "timestamp,volume,vw,open,close,high,low,t,trades,vwap,pre_market,after_market" > $TEST_DATA
head -6 /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/data/cache/AAPL_2022-01-01_2024-12-31_1d.csv | tail -5 >> $TEST_DATA

echo "📊 Test data file: $TEST_DATA"
echo "📋 Data content:"
cat $TEST_DATA
echo ""

echo "🔄 Generating strategy signals (only 5 data points)..."
python generate_strategy_signals.py \
    --dataset_path "$TEST_DATA" \
    --symbol "AAPL" \
    --lookback 10 \
    --output "quick_test_output.json"

echo ""
echo "🎯 Quick test complete!"

# Show result
if [ -f "quick_test_output.json" ]; then
    echo "✅ Output file generated successfully:"
    cat quick_test_output.json
else
    echo "❌ Output file not generated"
fi
