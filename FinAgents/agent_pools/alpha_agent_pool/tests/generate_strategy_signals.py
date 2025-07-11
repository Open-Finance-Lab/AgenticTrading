#!/usr/bin/env python3
"""
Strategy Signal Generator
Generates new strategy signal files from market data using Alpha Agent Pool.
This file focuses solely on signal generation and saves results to strategy flow files.
"""
import asyncio
import json
import sys
import logging
import csv
from datetime import datetime
from typing import List, Dict, Any
import argparse
import os

# Add project root directory to Python path to ensure mcp module can be found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from mcp import ClientSession
from mcp.client.sse import sse_client

# Import schema classes for proper signal flow generation (not needed for raw signal flow)
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from schema.theory_driven_schema import (
#     AlphaStrategyFlow, MarketContext, Decision, Action, PerformanceFeedback, Metadata
# )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_csv_dates(csv_path: str, date_col: str = "timestamp", price_col: str = "close") -> list:
    """Load market data from CSV file for signal generation"""
    data = []
    try:
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if date_col in row and price_col in row and row[price_col]:
                    try:
                        # Extract date part from 'YYYY-MM-DD HH:MM:SS' format timestamp
                        date_obj = datetime.strptime(row[date_col], '%Y-%m-%d %H:%M:%S')
                        date_str = date_obj.strftime('%Y-%m-%d')
                        price = float(row[price_col])
                        data.append({"date": date_str, "price": price})
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Cannot parse date or price from row {row}: {e}")
                        continue
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
    return data


class AlphaSignalGenerator:
    """Client for generating strategy signals using Alpha Agent Pool"""
    
    def __init__(self, host="localhost", port=8081):
        self.base_url = f"http://{host}:{port}/sse"

    async def _call_tool(self, tool_name: str, params: dict) -> Any:
        """Generic tool calling function for signal generation"""
        import traceback
        try:
            logger.info(f"🔧 Calling tool '{tool_name}' with params: {params}")
            async with sse_client(self.base_url, timeout=60) as (read, write):  # Increase timeout
                async with ClientSession(read, write) as session:
                    logger.info(f"📡 Session initialized, calling tool...")
                    await session.initialize()
                    response_parts = await session.call_tool(tool_name, params)
                    logger.info(f"📨 Received response from tool '{tool_name}'")

                    if response_parts is None:
                        logger.warning(f"Received empty response from tool '{tool_name}'")
                        return None

                    # Handle response format
                    part = response_parts
                    if isinstance(response_parts, list):
                        if not response_parts:
                            logger.warning(f"Received empty list response from tool '{tool_name}'")
                            return None
                        part = response_parts[0]

                    # Extract content
                    content_str = None
                    if hasattr(part, 'content') and isinstance(part.content, list):
                        for c in part.content:
                            if hasattr(c, 'type') and c.type == 'text' and hasattr(c, 'text'):
                                content_str = c.text
                                break
                    if not content_str and hasattr(part, 'text'):
                        content_str = part.text

                    if content_str:
                        try:
                            data = json.loads(content_str)
                            logger.info(f"✅ Successfully parsed JSON response from '{tool_name}'")
                        except json.JSONDecodeError:
                            logger.info(f"📄 Received non-JSON response from '{tool_name}', returning as string")
                            return content_str

                        # 原封不动返回任何格式的数据
                        return data
                    
                    logger.warning(f"Unexpected response format from tool '{tool_name}': {response_parts}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}': {e}\n{traceback.format_exc()}")
            return None

    async def generate_alpha_signals(self, symbols: list, date: str, lookback_period: int, price: float = None) -> dict:
        """Generate alpha signals using agent pool"""
        params = {
            "symbols": symbols,
            "date": date,
            "lookback_period": lookback_period,
            "price": price
        }
        return await self._call_tool("generate_alpha_signals", params)

    async def generate_signal(self, symbol: str, price_list: List[float] = None) -> dict:
        """Call the momentum agent's generate_signal tool directly"""
        params = {"symbol": symbol}
        if price_list is not None:
            params["price_list"] = price_list
        return await self._call_tool("generate_signal", params)


async def generate_strategy_flow(csv_path: str, symbol: str, lookback_days: int, output_file: str = None):
    """Generate strategy signal flow from market data"""
    logger.info(f"\n🎯 Generating strategy signals for {symbol} from {csv_path}")
    
    # Load market data
    market_data = load_csv_dates(csv_path, date_col="timestamp", price_col="close")
    logger.info(f"Loaded {len(market_data)} data points")
    
    if not market_data:
        logger.error(f"No data loaded from {csv_path}")
        return None

    # Initialize signal generator
    generator = AlphaSignalGenerator()
    
    # Test agent availability
    logger.info("🧪 Testing agent connectivity...")
    try:
        test_signal = await generator.generate_alpha_signals([symbol], "2024-01-01", lookback_days, 100.0)
        if test_signal:
            logger.info("✅ Agent connectivity confirmed")
        else:
            logger.warning("⚠️ Agent may not be responding properly")
    except Exception as e:
        logger.error(f"❌ Agent connectivity test failed: {e}")
        return None

    # Generate signals for each data point
    logger.info(f"🔄 Generating signals for {len(market_data)} data points...")
    strategy_flows = []

    for i, data_point in enumerate(market_data):
        date = data_point["date"]
        price = data_point["price"]
        
        logger.info(f"Processing {i+1}/{len(market_data)}: {date} @ ${price:.2f}")
        
        # Generate signal for this data point
        signals = await generator.generate_alpha_signals(
            symbols=[symbol],
            date=date,
            lookback_period=lookback_days,
            price=float(price)
        )
        
        if signals is not None:
            # Receive signal as-is, no assumptions or transformation
            strategy_flows.append(signals)
            logger.info(f"  ✅ Signal received for {date}")
        else:
            logger.warning(f"  ❌ Failed to generate signal for {date} - aborting")
            # Abort if no signal flow received
            break

    # Save strategy flow
    if output_file is None:
        output_file = f"strategy_flow_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(strategy_flows, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n📁 Strategy flow saved to: {output_file}")
    logger.info(f"📊 Total signals generated: {len(strategy_flows)}")
    
    return {
        "output_file": output_file,
        "total_signals": len(strategy_flows),
        "strategy_flows": strategy_flows
    }


async def main():
    """Main function for strategy signal generation"""
    parser = argparse.ArgumentParser(description="Generate Strategy Signals from Market Data")
    parser.add_argument("--dataset_path", type=str, required=True, 
                       help="Path to the dataset CSV file or directory")
    parser.add_argument("--symbol", type=str, default="AAPL", 
                       help="Stock symbol to generate signals for")
    parser.add_argument("--lookback", type=int, default=30, 
                       help="Lookback period for momentum analysis")
    parser.add_argument("--output", type=str, 
                       help="Output file name for strategy flow (auto-generated if not provided)")
    
    args = parser.parse_args()

    # Determine CSV file path
    if os.path.isdir(args.dataset_path):
        csv_file = os.path.join(args.dataset_path, f"{args.symbol}_2022-01-01_2024-12-31_1d.csv")
    else:
        csv_file = args.dataset_path

    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        return

    # Generate strategy signals
    result = await generate_strategy_flow(
        csv_path=csv_file,
        symbol=args.symbol,
        lookback_days=args.lookback,
        output_file=args.output
    )
    
    if result:
        logger.info(f"\n✅ Strategy signal generation completed successfully!")
        logger.info(f"📄 Output file: {result['output_file']}")
        logger.info(f"📊 Total signals: {result['total_signals']}")
        logger.info(f"🎯 Use the output file with execute_strategy_trades.py to run backtests")
    else:
        logger.error("❌ Strategy signal generation failed")


if __name__ == "__main__":
    asyncio.run(main())
