#!/bin/bash

# Strategy Signal Generator Test Script
# Strategy Signal Generator Test Script

echo "🚀 Starting strategy signal generation test..."

## Set test parameters
SCRIPT_DIR="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/tests"
DATA_PATH="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/data/cache"
SYMBOL="AAPL"
LOOKBACK=10
DAYS_TO_TEST=30
OUTPUT="test_strategy_flow_AAPL_small.json"

echo "📋 Test parameters:"
echo "  Data path: $DATA_PATH"
echo "  Symbol: $SYMBOL"
echo "  Lookback: $LOOKBACK days"
echo "  Test data: $DAYS_TO_TEST days"
echo "  Output file: $OUTPUT"
echo ""

## Create small test data file (first 30 days only)
SMALL_CSV_FILE="$SCRIPT_DIR/test_data_small.csv"
ORIGINAL_CSV="$DATA_PATH/${SYMBOL}_2022-01-01_2024-12-31_1d.csv"

echo "🔧 Creating small test data file..."
head -1 "$ORIGINAL_CSV" > "$SMALL_CSV_FILE"  # Copy header
tail -n +2 "$ORIGINAL_CSV" | head -$DAYS_TO_TEST >> "$SMALL_CSV_FILE"  # Get first 30 rows

echo "✅ Small test data file created: $SMALL_CSV_FILE"
echo "📊 Test data line count: $(wc -l < "$SMALL_CSV_FILE")"
echo ""

## Check if test data file exists
if [ ! -f "$SMALL_CSV_FILE" ]; then
    echo "❌ Failed to create test data file: $SMALL_CSV_FILE"
    exit 1
fi

## Change to script directory
cd "$SCRIPT_DIR"

echo "🔄 Generating strategy signals..."
echo "Command: python generate_strategy_signals.py --dataset_path \"$SMALL_CSV_FILE\" --symbol \"$SYMBOL\" --lookback $LOOKBACK --output \"$OUTPUT\""
echo ""

## Run strategy signal generator (using small test file)
python generate_strategy_signals.py \
    --dataset_path "$SMALL_CSV_FILE" \
    --symbol "$SYMBOL" \
    --lookback $LOOKBACK \
    --output "$OUTPUT"

echo ""
# Check result
RESULT_CODE=$?
echo ""
echo "🏁 Test finished, exit code: $RESULT_CODE"

if [ $RESULT_CODE -eq 0 ]; then
    echo "✅ Strategy signal generation succeeded!"
    
    # Check output file
    if [ -f "$OUTPUT" ]; then
        echo "📁 Output file: $OUTPUT"
        echo "📊 File size: $(ls -lh "$OUTPUT" | awk '{print $5}')"
        echo "🔍 First few lines:"
        head -20 "$OUTPUT"
    else
        echo "⚠️ Output file not found: $OUTPUT"
    fi
else
    echo "❌ Strategy signal generation failed!"
fi

echo "🎯 Test complete!"

