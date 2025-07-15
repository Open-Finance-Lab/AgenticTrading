#!/usr/bin/env python3
"""
Alpha Agent Pool MCP Client - SSE Transport
Test all Alpha Agent Pool functionality using SSE channel
This client will connect to Alpha Agent Pool MCP server and test all available tools and functions.
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


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_csv_dates(csv_path: str, date_col: str = "timestamp", price_col: str = "close") -> list:
    """Load backtest dates and price data from CSV file"""
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

class AlphaAgentPoolClient:
    """Async client for Alpha Agent Pool using ClientSession and sse_client"""
    def __init__(self, host="localhost", port=8081):
        self.base_url = f"http://{host}:{port}/sse"

    async def _call_tool(self, tool_name: str, params: dict) -> Any:
        """Generic tool calling function, compatible with momentum agent return format and enhanced exception output"""
        import traceback
        try:
            async with sse_client(self.base_url, timeout=30) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    response_parts = await session.call_tool(tool_name, params)

                    if response_parts is None:
                        logger.warning(f"Received empty response from tool '{tool_name}'")
                        return None

                    # Compatible with CallToolResult or list
                    part = response_parts
                    if isinstance(response_parts, list):
                        if not response_parts:
                            logger.warning(f"Received empty list response from tool '{tool_name}'")
                            return None
                        part = response_parts[0]

                    # Priority compatibility with content=[TextContent(...)] structure
                    content_str = None
                    if hasattr(part, 'content') and isinstance(part.content, list):
                        for c in part.content:
                            # Compatible with TextContent(type='text', text=...)
                            if hasattr(c, 'type') and c.type == 'text' and hasattr(c, 'text'):
                                content_str = c.text
                                break
                    if not content_str and hasattr(part, 'text'):
                        content_str = part.text

                    if content_str:
                        try:
                            data = json.loads(content_str)
                        except json.JSONDecodeError:
                            return content_str

                        # Compatible with momentum agent return format
                        if tool_name == "generate_alpha_signals":
                            # If already in standard format, return directly
                            if isinstance(data, dict) and "status" in data and "alpha_signals" in data:
                                return data
                            # momentum agent format adaptation
                            if isinstance(data, dict) and "decision" in data:
                                # Build standard format
                                symbol = None
                                if "market_context" in data and "symbol" in data["market_context"]:
                                    symbol = data["market_context"]["symbol"]
                                signal = data["decision"].get("signal")
                                # Compatible with asset_scope
                                asset_scope = data["decision"].get("asset_scope")
                                if asset_scope and isinstance(asset_scope, list) and len(asset_scope) > 0:
                                    symbol = asset_scope[0]
                                # Build alpha_signals
                                alpha_signals = {
                                    "signals": {
                                        symbol or "UNKNOWN": {
                                            "signal": signal,
                                            "confidence": data["decision"].get("confidence"),
                                            "reasoning": data["decision"].get("reasoning"),
                                            "raw": data
                                        }
                                    }
                                }
                                return {
                                    "status": "success",
                                    "alpha_signals": alpha_signals
                                }
                        return data
                    logger.warning(f"Unexpected response format from tool '{tool_name}': {response_parts}")
                    return None
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}': {e}\n{traceback.format_exc()}")
            return None

    async def process_strategy_request(self, query: str) -> str:
        """Process strategy request"""
        return await self._call_tool("process_strategy_request", {"query": query})

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

async def run_backtest_with_agent_pool(csv_path: str, symbol: str, lookback_days: int):
    """Run backtest using Alpha Agent Pool"""
    logger.info("\n🚦 Alpha Agent Pool backtest: %s", symbol)
    
    # Load backtest dates and prices from CSV
    market_data = load_csv_dates(csv_path, date_col="timestamp", price_col="close")
    logger.info("Number of data points: %d", len(market_data))
    if not market_data:
        logger.error("Failed to load any data from %s", csv_path)
        return

    # Initialize Alpha Agent Pool client
    client = AlphaAgentPoolClient()

    # Test planner functionality before backtest
    logger.info("\n🧪 Testing Command Planner functionality...")
    try:
        list_agents_response = await client.process_strategy_request("list agents")
        logger.info("Response to `list agents` command: %s", list_agents_response)
        
        # Check if any agent is running
        if list_agents_response and "No agents are currently running" in list_agents_response.get("planner_output", ""):
             logger.warning("No agent is running. Backtest may not generate signals.")

    except Exception as e:
        logger.error(f"Error testing planner functionality: {e}")


    # Run backtest using 'generate_alpha_signals' tool
    logger.info("Starting backtest using 'generate_alpha_signals'...")
    
    signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}

    for data_point in market_data:
        date = data_point["date"]
        price = data_point["price"]
        
        signals = await client.generate_alpha_signals(
            symbols=[symbol],
            date=date,
            lookback_period=lookback_days,
            price=float(price)  # Pass price
        )
        
        logger.info("Raw response from generate_alpha_signals for date %s: %s", date, signals)

        if signals and signals.get("status") == "success":
            signal_data = signals.get("alpha_signals", {}).get("signals", {})
            for s, data in signal_data.items():
                signal_counts[data["signal"]] += 1
        else:
            logger.warning("Failed to generate signal for date %s or generation failed: %s", date, signals.get("message", "No details") if signals else "No response")

    logger.info("\nSignal statistics: BUY=%d, SELL=%d, HOLD=%d", signal_counts["BUY"], signal_counts["SELL"], signal_counts["HOLD"])

async def main():
    """
    Main function to run tests for Alpha Agent Pool client.
    All comments and docstrings are in English. This function also tests the new RL backtest/update method.
    """
    parser = argparse.ArgumentParser(description="Alpha Agent Pool Test Client")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset CSV file or directory.")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol to test.")
    parser.add_argument("--lookback", type=int, default=30, help="Lookback period for momentum.")
    parser.add_argument("--strategy_flow", type=str, help="Path to existing strategy flow JSON file to execute trades from.")
    parser.add_argument("--initial_cash", type=float, default=100000, help="Initial cash for trading simulation.")
    args = parser.parse_args()


    # If strategy flow file is provided, execute trades directly from it
    if args.strategy_flow:
        logger.info(f"🎯 Executing trades from existing strategy flow: {args.strategy_flow}")
        result = await execute_trades_from_strategy_flow(
            strategy_flow_file=args.strategy_flow,
            symbol=args.symbol,
            initial_cash=args.initial_cash
        )
        return

    # If no dataset path provided and no strategy flow, show help
    if not args.dataset_path:
        parser.print_help()
        logger.error("--dataset_path is required when not using --strategy_flow")
        return

    client = AlphaAgentPoolClient()

    # --- 1. Test process_strategy_request tool ---
    logger.info("\n--- [Test 1] Test process_strategy_request tool ---")
    response = await client.process_strategy_request("test")
    logger.info(f"Response: {response}")

    # --- 2. Test generate_alpha_signals tool ---
    logger.info("\n--- [Test 2] Test generate_alpha_signals tool ---")
    signals = await client.generate_alpha_signals([args.symbol], "2023-01-01", args.lookback)
    logger.info(f"Signals: {signals}")

    # --- 3. Use planner to list running agents ---
    logger.info("\n--- [Test 3] Use Planner to list running agents ---")
    planner_response = await client.process_strategy_request("list agents")
    if planner_response:
        logger.info(f"Planner response: {planner_response}")
    else:
        logger.error("Failed to get response from planner")

    # --- 4. Test RL backtest and update method ---
    logger.info(f"\n--- [Test 4] Run RL backtest and update for {args.symbol} ---")

    # Determine if input is a single file or directory
    if os.path.isdir(args.dataset_path):
        csv_file = os.path.join(args.dataset_path, f"{args.symbol}_2022-01-01_2024-12-31_1d.csv")
    else:
        csv_file = args.dataset_path

    # Load market data
    backtest_data = load_csv_dates(csv_file)
    if not backtest_data:
        logger.error(f"Failed to load backtest data from {csv_file}. Aborting.")
        return

    # --- NEW RL BACKTEST/UPDATE TEST ---
    # This tests the agent's RL learning loop and feedback-driven optimization
    logger.info(f"\n--- [Test 5] Testing RL backtest and policy update ---")
    try:
        # The following assumes the agent pool exposes a tool named 'run_rl_backtest_and_update'
        rl_result = await client._call_tool(
            "run_rl_backtest_and_update",
            {
                "symbol": args.symbol,
                "market_data": backtest_data,
                "lookback_period": args.lookback,
                "initial_cash": args.initial_cash
            }
        )
        logger.info(f"RL backtest and update result: {rl_result}")
    except Exception as e:
        logger.error(f"Error running RL backtest and update: {e}")

    # Optionally, you can add more assertions or checks here to validate RL learning impact

async def execute_trades_from_strategy_flow(strategy_flow_file: str, symbol: str, initial_cash: float = 100000):
    """
    Execute trades based on existing strategy flow file and generate detailed trading report
    """
    logger.info(f"\n🔄 Executing trades from strategy flow: {strategy_flow_file}")
    
    # Load strategy flows from file
    try:
        with open(strategy_flow_file, 'r', encoding='utf-8') as f:
            strategy_flows = json.load(f)
        logger.info(f"Loaded {len(strategy_flows)} strategy flow entries")
    except FileNotFoundError:
        logger.error(f"Strategy flow file not found: {strategy_flow_file}")
        return
    except Exception as e:
        logger.error(f"Error loading strategy flow: {e}")
        return
    
    if not strategy_flows:
        logger.error("No strategy flows found in file")
        return
    
    # Initialize trading state
    cash = initial_cash
    position = 0.0  # Number of shares held
    equity_curve = []
    trade_log = []
    daily_returns = []
    
    # Trading parameters
    min_trade_threshold = 0.05  # Minimum position change to trigger trade (5%)
    max_position_ratio = 0.95   # Maximum portfolio allocation to single position
    
    logger.info(f"Starting trading simulation with ${initial_cash:,.2f} initial capital")
    logger.info(f"Trading parameters: min_threshold={min_trade_threshold}, max_position={max_position_ratio}")
    
    for i, flow in enumerate(strategy_flows):
        date = flow.get("date", f"day_{i}")
        price = flow.get("price", 0)
        
        # Extract signal information
        signal = flow.get("signal", "HOLD")
        confidence = flow.get("confidence", 0.0)
        execution_weight = flow.get("execution_weight", 0.0)
        reasoning = flow.get("reasoning", "No reasoning provided")
        
        # Calculate current portfolio metrics
        portfolio_value = cash + position * price
        current_position_ratio = (position * price) / portfolio_value if portfolio_value > 0 else 0
        
        # Determine target position based on signal and execution weight
        target_position_ratio = 0.0
        
        if signal == "BUY":
            # Use execution_weight as target position ratio, capped at max_position_ratio
            target_position_ratio = min(max_position_ratio, max(0.0, execution_weight))
            
        elif signal == "SELL":
            # For SELL signals, reduce position based on confidence
            if confidence > 0.7:
                target_position_ratio = 0.0  # Close position for high confidence sells
            elif confidence > 0.5:
                # Partial reduction for medium confidence
                target_position_ratio = current_position_ratio * 0.5
            else:
                # Small reduction for low confidence
                target_position_ratio = current_position_ratio * 0.8
                
        elif signal == "HOLD":
            # Maintain current position
            target_position_ratio = current_position_ratio
        
        # Calculate position change required
        position_change = abs(target_position_ratio - current_position_ratio)
        
        # Execute trade if change is significant enough
        if position_change > min_trade_threshold:
            target_shares = (portfolio_value * target_position_ratio) / price if price > 0 else 0
            shares_to_trade = target_shares - position
            
            trade_executed = False
            
            if shares_to_trade > 0.01:  # BUY ORDER
                # Calculate how many shares we can afford
                max_affordable = cash / price
                actual_shares = min(shares_to_trade, max_affordable)
                
                if actual_shares > 0.01 and cash >= actual_shares * price:
                    trade_cost = actual_shares * price
                    position += actual_shares
                    cash -= trade_cost
                    trade_executed = True
                    
                    trade_record = {
                        "date": date,
                        "action": "BUY",
                        "shares": round(actual_shares, 2),
                        "price": round(price, 2),
                        "cost": round(trade_cost, 2),
                        "signal": signal,
                        "confidence": round(confidence, 3),
                        "execution_weight": round(execution_weight, 3),
                        "reasoning": reasoning,
                        "portfolio_value_before": round(portfolio_value, 2),
                        "cash_after": round(cash, 2),
                        "position_after": round(position, 2),
                        "target_ratio": round(target_position_ratio, 3),
                        "actual_ratio": round((position * price) / (cash + position * price), 3)
                    }
                    trade_log.append(trade_record)
                    
                    logger.info(f"📈 BUY  | {date} | {actual_shares:.2f} shares @ ${price:.2f} | Cost: ${trade_cost:.2f} | Confidence: {confidence:.2f}")
                    
            elif shares_to_trade < -0.01:  # SELL ORDER
                shares_to_sell = min(abs(shares_to_trade), position)
                
                if shares_to_sell > 0.01:
                    proceeds = shares_to_sell * price
                    position -= shares_to_sell
                    cash += proceeds
                    trade_executed = True
                    
                    trade_record = {
                        "date": date,
                        "action": "SELL",
                        "shares": round(shares_to_sell, 2),
                        "price": round(price, 2),
                        "proceeds": round(proceeds, 2),
                        "signal": signal,
                        "confidence": round(confidence, 3),
                        "execution_weight": round(execution_weight, 3),
                        "reasoning": reasoning,
                        "portfolio_value_before": round(portfolio_value, 2),
                        "cash_after": round(cash, 2),
                        "position_after": round(position, 2),
                        "target_ratio": round(target_position_ratio, 3),
                        "actual_ratio": round((position * price) / (cash + position * price) if (cash + position * price) > 0 else 0, 3)
                    }
                    trade_log.append(trade_record)
                    
                    logger.info(f"📉 SELL | {date} | {shares_to_sell:.2f} shares @ ${price:.2f} | Proceeds: ${proceeds:.2f} | Confidence: {confidence:.2f}")
            
            if not trade_executed and position_change > min_trade_threshold:
                logger.debug(f"⏸️ No trade executed for {date} despite signal {signal} (insufficient funds or position)")
        
        # Record daily portfolio value and returns
        final_portfolio_value = cash + position * price
        equity_curve.append(final_portfolio_value)
        
        if i > 0:
            daily_return = (final_portfolio_value - equity_curve[i-1]) / equity_curve[i-1]
            daily_returns.append(daily_return)
    
    # Generate comprehensive trading report
    logger.info(f"\n" + "="*60)
    logger.info(f"📊 TRADING EXECUTION REPORT - {symbol}")
    logger.info(f"="*60)
    
    # Basic portfolio metrics
    final_value = equity_curve[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100
    
    logger.info(f"💰 Portfolio Performance:")
    logger.info(f"   Initial Capital:     ${initial_cash:,.2f}")
    logger.info(f"   Final Portfolio:     ${final_value:,.2f}")
    logger.info(f"   Total Return:        {total_return:.2f}%")
    logger.info(f"   Final Cash:          ${cash:,.2f}")
    logger.info(f"   Final Position:      {position:.2f} shares (${position * strategy_flows[-1]['price']:,.2f})")
    
    # Trading activity summary
    buy_trades = [t for t in trade_log if t['action'] == 'BUY']
    sell_trades = [t for t in trade_log if t['action'] == 'SELL']
    
    logger.info(f"🔄 Trading Activity:")
    logger.info(f"   Total Trades:        {len(trade_log)}")
    logger.info(f"   Buy Orders:          {len(buy_trades)}")
    logger.info(f"   Sell Orders:         {len(sell_trades)}")
    
    if trade_log:
        total_volume = sum(t.get('cost', 0) + t.get('proceeds', 0) for t in trade_log)
        logger.info(f"   Total Volume:        ${total_volume:,.2f}")
        logger.info(f"   First Trade:         {trade_log[0]['date']} - {trade_log[0]['action']}")
        logger.info(f"   Last Trade:          {trade_log[-1]['date']} - {trade_log[-1]['action']}")
    
    # Calculate risk metrics
    if len(daily_returns) > 1:
        import numpy as np
        returns_array = np.array(daily_returns)
        
        # Annualized metrics
        avg_daily_return = np.mean(returns_array)
        volatility = np.std(returns_array) * np.sqrt(252) * 100
        sharpe_ratio = avg_daily_return / (np.std(returns_array) + 1e-8) * np.sqrt(252)
        
        # Drawdown calculation
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        logger.info(f"📈 Risk Metrics:")
        logger.info(f"   Volatility:          {volatility:.2f}%")
        logger.info(f"   Sharpe Ratio:        {sharpe_ratio:.3f}")
        logger.info(f"   Max Drawdown:        {max_drawdown:.2f}%")
        logger.info(f"   Win Rate:            {(returns_array > 0).mean() * 100:.1f}%")
    
    # Signal distribution analysis
    signal_counts = {}
    for flow in strategy_flows:
        signal = flow.get("signal", "UNKNOWN")
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
    
    logger.info(f"📊 Signal Distribution:")
    for signal, count in signal_counts.items():
        pct = count / len(strategy_flows) * 100
        logger.info(f"   {signal:4s}: {count:3d} ({pct:.1f}%)")
    
    logger.info(f"="*60)
    
    # Save detailed trading log
    trade_log_file = f"detailed_trade_log_{symbol}.json"
    with open(trade_log_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "symbol": symbol,
                "initial_cash": initial_cash,
                "final_value": final_value,
                "total_return_pct": total_return,
                "total_trades": len(trade_log),
                "buy_trades": len(buy_trades),
                "sell_trades": len(sell_trades)
            },
            "trades": trade_log,
            "equity_curve": equity_curve,
            "daily_returns": daily_returns
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📁 Detailed trading log saved to: {trade_log_file}")
    
    # Create visualization
    create_trading_visualization(strategy_flows, trade_log, equity_curve, symbol)
    
    return {
        "final_value": final_value,
        "total_return": total_return,
        "trade_log": trade_log,
        "equity_curve": equity_curve
    }

def create_trading_visualization(strategy_flows, trade_log, equity_curve, symbol):
    """Create comprehensive trading visualization"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        # Prepare data
        dates = [datetime.strptime(flow["date"], "%Y-%m-%d") for flow in strategy_flows]
        prices = [flow["price"] for flow in strategy_flows]
        signals = [flow.get("signal", "HOLD") for flow in strategy_flows]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Trading Analysis Dashboard - {symbol}', fontsize=16, fontweight='bold')
        
        # Plot 1: Stock Price with Signals
        ax1.plot(dates, prices, label='Stock Price', linewidth=2, color='black', alpha=0.8)
        
        # Color background based on signals
        for i, signal in enumerate(signals):
            start_date = dates[i]
            end_date = dates[min(i+1, len(dates)-1)]
            
            if signal == "BUY":
                ax1.axvspan(start_date, end_date, alpha=0.3, color='green')
            elif signal == "SELL":
                ax1.axvspan(start_date, end_date, alpha=0.3, color='red')
            elif signal == "HOLD":
                ax1.axvspan(start_date, end_date, alpha=0.1, color='gray')
        
        # Mark trades
        for trade in trade_log:
            trade_date = datetime.strptime(trade['date'], "%Y-%m-%d")
            if trade_date in dates:
                idx = dates.index(trade_date)
                if trade['action'] == 'BUY':
                    ax1.scatter(trade_date, prices[idx], color='darkgreen', marker='^', s=100, zorder=5, edgecolors='white')
                else:
                    ax1.scatter(trade_date, prices[idx], color='darkred', marker='v', s=100, zorder=5, edgecolors='white')
        
        ax1.set_title('Stock Price with Signals & Trades')
        ax1.set_ylabel('Price ($)')
        ax1.legend(['Price', 'Buy Signal', 'Sell Signal', 'Hold Signal'])
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Plot 2: Portfolio Equity Curve
        ax2.plot(dates, equity_curve, label='Portfolio Value', linewidth=3, color='blue')
        ax2.set_title('Portfolio Equity Curve')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Add performance annotation
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
        ax2.text(0.02, 0.98, f'Total Return: {total_return:.1f}%', 
                transform=ax2.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot 3: Signal Distribution
        signal_counts = {}
        for flow in strategy_flows:
            signal = flow.get("signal", "UNKNOWN")
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
        signals_list = list(signal_counts.keys())
        counts = list(signal_counts.values())
        colors_list = [colors.get(s, 'blue') for s in signals_list]
        
        ax3.pie(counts, labels=signals_list, colors=colors_list, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Signal Distribution')
        
        # Plot 4: Trade Timeline
        if trade_log:
            trade_dates = [datetime.strptime(t['date'], "%Y-%m-%d") for t in trade_log]
            trade_values = [t.get('cost', 0) if t['action'] == 'BUY' else -t.get('proceeds', 0) for t in trade_log]
            trade_colors = ['green' if t['action'] == 'BUY' else 'red' for t in trade_log]
            
            bars = ax4.bar(trade_dates, trade_values, color=trade_colors, alpha=0.7, width=1)
            ax4.set_title('Trade Timeline (Green=Buy, Red=Sell)')
            ax4.set_ylabel('Trade Value ($)')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            # Add trade count annotation
            buy_count = len([t for t in trade_log if t['action'] == 'BUY'])
            sell_count = len([t for t in trade_log if t['action'] == 'SELL'])
            ax4.text(0.02, 0.98, f'Trades: {buy_count} Buys, {sell_count} Sells', 
                    transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No Trades Executed', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Trade Timeline')
        
        # Format x-axes
        for ax in [ax1, ax2, ax4]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"trading_analysis_{symbol}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Trading analysis chart saved to: {filename}")
        
        plt.show()
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualization")
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")

if __name__ == "__main__":
    asyncio.run(main())
