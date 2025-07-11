#!/usr/bin/env python3
"""
Strategy Trade Executor
Executes trades directly from existing strategy signal files.
This file focuses solely on trade execution and backtesting from pre-generated signals.
No additional logic or signal modification - pure execution of strategy flow decisions.
"""
import json
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any
import argparse
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrategyTradeExecutor:
    """Execute trades based on pre-generated strategy flow signals"""
    
    def __init__(self, initial_cash: float = 100000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0.0  # Number of shares held
        self.equity_curve = []
        self.trade_log = []
        self.daily_returns = []
        
        # Trading parameters - pure execution focused
        self.min_trade_threshold = 0.03  # Minimum position change to trigger trade (3%)
        self.max_position_ratio = 0.95   # Maximum portfolio allocation
        
    def load_strategy_flow(self, strategy_flow_file: str) -> List[Dict[str, Any]]:
        """Load strategy flow from JSON file"""
        try:
            with open(strategy_flow_file, 'r', encoding='utf-8') as f:
                strategy_flows = json.load(f)
            logger.info(f"📁 Loaded {len(strategy_flows)} strategy flow entries from {strategy_flow_file}")
            return strategy_flows
        except FileNotFoundError:
            logger.error(f"❌ Strategy flow file not found: {strategy_flow_file}")
            return []
        except Exception as e:
            logger.error(f"❌ Error loading strategy flow: {e}")
            return []

    def execute_trades(self, strategy_flows: List[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        """Execute trades based strictly on strategy flow decisions"""
        if not strategy_flows:
            logger.error("No strategy flows provided for execution")
            return {}
        
        logger.info(f"🔄 Executing trades for {symbol} with ${self.initial_cash:,.2f} initial capital")
        logger.info(f"⚙️ Trading parameters: min_threshold={self.min_trade_threshold}, max_position={self.max_position_ratio}")
        
        for i, flow in enumerate(strategy_flows):
            date = flow.get("date", f"day_{i}")
            price = flow.get("price", 0)
            
            # Extract signal information directly from strategy flow
            signal = flow.get("signal", "HOLD")
            confidence = flow.get("confidence", 0.0)
            execution_weight = flow.get("execution_weight", confidence)
            reasoning = flow.get("reasoning", "No reasoning provided")
            
            # Calculate current portfolio metrics
            portfolio_value = self.cash + self.position * price
            current_position_ratio = (self.position * price) / portfolio_value if portfolio_value > 0 else 0
            
            # Determine target position based STRICTLY on strategy flow signals
            target_position_ratio = self._calculate_target_position(
                signal, execution_weight, confidence, current_position_ratio
            )
            
            # Calculate position change required
            position_change = abs(target_position_ratio - current_position_ratio)
            
            # Execute trade if change is significant enough
            if position_change > self.min_trade_threshold:
                self._execute_trade(
                    date, price, signal, confidence, execution_weight, reasoning,
                    target_position_ratio, portfolio_value
                )
            
            # Record daily portfolio value and returns
            final_portfolio_value = self.cash + self.position * price
            self.equity_curve.append(final_portfolio_value)
            
            if i > 0:
                daily_return = (final_portfolio_value - self.equity_curve[i-1]) / self.equity_curve[i-1]
                self.daily_returns.append(daily_return)
        
        # Generate execution report
        return self._generate_execution_report(symbol, strategy_flows)

    def _calculate_target_position(self, signal: str, execution_weight: float, confidence: float, 
                                 current_position_ratio: float) -> float:
        """Calculate target position ratio based strictly on signal parameters"""
        target_position_ratio = 0.0
        
        if signal == "BUY":
            # Use execution_weight directly as target position ratio
            target_position_ratio = min(self.max_position_ratio, max(0.0, execution_weight))
            
        elif signal == "SELL":
            # Reduce position based on confidence level
            if confidence > 0.7:
                target_position_ratio = 0.0  # Close position for high confidence sells
            elif confidence > 0.5:
                target_position_ratio = current_position_ratio * 0.5  # 50% reduction
            elif confidence > 0.3:
                target_position_ratio = current_position_ratio * 0.7  # 30% reduction
            else:
                target_position_ratio = current_position_ratio * 0.9  # 10% reduction
                
        elif signal == "HOLD":
            # Maintain current position exactly as strategy flow indicates
            target_position_ratio = current_position_ratio
        
        return target_position_ratio

    def _execute_trade(self, date: str, price: float, signal: str, confidence: float, 
                      execution_weight: float, reasoning: str, target_position_ratio: float, 
                      portfolio_value: float) -> bool:
        """Execute individual trade based on strategy parameters"""
        target_shares = (portfolio_value * target_position_ratio) / price if price > 0 else 0
        shares_to_trade = target_shares - self.position
        
        trade_executed = False
        
        if shares_to_trade > 0.01:  # BUY ORDER
            max_affordable = self.cash / price
            actual_shares = min(shares_to_trade, max_affordable)
            
            if actual_shares > 0.01 and self.cash >= actual_shares * price:
                trade_cost = actual_shares * price
                self.position += actual_shares
                self.cash -= trade_cost
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
                    "cash_after": round(self.cash, 2),
                    "position_after": round(self.position, 2),
                    "target_ratio": round(target_position_ratio, 3),
                    "actual_ratio": round((self.position * price) / (self.cash + self.position * price), 3)
                }
                self.trade_log.append(trade_record)
                
                logger.info(f"📈 BUY  | {date} | {actual_shares:.2f} shares @ ${price:.2f} | Cost: ${trade_cost:.2f}")
                
        elif shares_to_trade < -0.01:  # SELL ORDER
            shares_to_sell = min(abs(shares_to_trade), self.position)
            
            if shares_to_sell > 0.01:
                proceeds = shares_to_sell * price
                self.position -= shares_to_sell
                self.cash += proceeds
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
                    "cash_after": round(self.cash, 2),
                    "position_after": round(self.position, 2),
                    "target_ratio": round(target_position_ratio, 3),
                    "actual_ratio": round((self.position * price) / (self.cash + self.position * price) if (self.cash + self.position * price) > 0 else 0, 3)
                }
                self.trade_log.append(trade_record)
                
                logger.info(f"📉 SELL | {date} | {shares_to_sell:.2f} shares @ ${price:.2f} | Proceeds: ${proceeds:.2f}")
        
        return trade_executed

    def _generate_execution_report(self, symbol: str, strategy_flows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive execution report"""
        logger.info(f"\n" + "="*60)
        logger.info(f"📊 TRADE EXECUTION REPORT - {symbol}")
        logger.info(f"="*60)
        
        # Basic portfolio metrics
        final_value = self.equity_curve[-1] if self.equity_curve else self.initial_cash
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        
        logger.info(f"💰 Portfolio Performance:")
        logger.info(f"   Initial Capital:     ${self.initial_cash:,.2f}")
        logger.info(f"   Final Portfolio:     ${final_value:,.2f}")
        logger.info(f"   Total Return:        {total_return:.2f}%")
        logger.info(f"   Final Cash:          ${self.cash:,.2f}")
        
        if strategy_flows:
            final_price = strategy_flows[-1].get('price', 0)
            logger.info(f"   Final Position:      {self.position:.2f} shares (${self.position * final_price:,.2f})")
        
        # Trading activity summary
        buy_trades = [t for t in self.trade_log if t['action'] == 'BUY']
        sell_trades = [t for t in self.trade_log if t['action'] == 'SELL']
        
        logger.info(f"🔄 Trading Activity:")
        logger.info(f"   Total Trades:        {len(self.trade_log)}")
        logger.info(f"   Buy Orders:          {len(buy_trades)}")
        logger.info(f"   Sell Orders:         {len(sell_trades)}")
        
        if self.trade_log:
            total_volume = sum(t.get('cost', 0) + t.get('proceeds', 0) for t in self.trade_log)
            logger.info(f"   Total Volume:        ${total_volume:,.2f}")
            logger.info(f"   First Trade:         {self.trade_log[0]['date']} - {self.trade_log[0]['action']}")
            logger.info(f"   Last Trade:          {self.trade_log[-1]['date']} - {self.trade_log[-1]['action']}")
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics()
        logger.info(f"📈 Risk Metrics:")
        for metric, value in risk_metrics.items():
            logger.info(f"   {metric}: {value}")
        
        # Signal distribution analysis
        signal_counts = {}
        for flow in strategy_flows:
            signal = flow.get("signal", "UNKNOWN")
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        logger.info(f"📊 Signal Distribution:")
        for signal, count in signal_counts.items():
            pct = count / len(strategy_flows) * 100 if strategy_flows else 0
            logger.info(f"   {signal:4s}: {count:3d} ({pct:.1f}%)")
        
        logger.info(f"="*60)
        
        return {
            "final_value": final_value,
            "total_return": total_return,
            "trade_log": self.trade_log,
            "equity_curve": self.equity_curve,
            "signal_counts": signal_counts,
            "risk_metrics": risk_metrics
        }

    def _calculate_risk_metrics(self) -> Dict[str, str]:
        """Calculate risk and performance metrics"""
        if len(self.daily_returns) < 2:
            return {"Insufficient data": "for risk calculation"}
        
        try:
            import numpy as np
            returns_array = np.array(self.daily_returns)
            
            # Annualized metrics
            avg_daily_return = np.mean(returns_array)
            volatility = np.std(returns_array) * np.sqrt(252) * 100
            sharpe_ratio = avg_daily_return / (np.std(returns_array) + 1e-8) * np.sqrt(252)
            
            # Drawdown calculation
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown) * 100
            
            return {
                "Volatility": f"{volatility:.2f}%",
                "Sharpe Ratio": f"{sharpe_ratio:.3f}",
                "Max Drawdown": f"{max_drawdown:.2f}%",
                "Win Rate": f"{(returns_array > 0).mean() * 100:.1f}%"
            }
        except ImportError:
            return {"NumPy not available": "for detailed risk metrics"}
        except Exception as e:
            return {"Error calculating metrics": str(e)}

    def save_execution_results(self, symbol: str, output_prefix: str = None) -> str:
        """Save detailed execution results to file"""
        if output_prefix is None:
            output_prefix = f"execution_results_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_file = f"{output_prefix}.json"
        
        results = {
            "execution_summary": {
                "symbol": symbol,
                "initial_cash": self.initial_cash,
                "final_value": self.equity_curve[-1] if self.equity_curve else self.initial_cash,
                "total_return_pct": ((self.equity_curve[-1] if self.equity_curve else self.initial_cash) - self.initial_cash) / self.initial_cash * 100,
                "total_trades": len(self.trade_log),
                "buy_trades": len([t for t in self.trade_log if t['action'] == 'BUY']),
                "sell_trades": len([t for t in self.trade_log if t['action'] == 'SELL'])
            },
            "trades": self.trade_log,
            "equity_curve": self.equity_curve,
            "daily_returns": self.daily_returns
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Execution results saved to: {output_file}")
        return output_file

    def create_visualization(self, strategy_flows: List[Dict[str, Any]], symbol: str):
        """Create trading visualization if matplotlib is available"""
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
            fig.suptitle(f'Strategy Execution Analysis - {symbol}', fontsize=16, fontweight='bold')
            
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
            
            # Mark executed trades
            for trade in self.trade_log:
                trade_date = datetime.strptime(trade['date'], "%Y-%m-%d")
                if trade_date in dates:
                    idx = dates.index(trade_date)
                    if trade['action'] == 'BUY':
                        ax1.scatter(trade_date, prices[idx], color='darkgreen', marker='^', s=100, zorder=5, edgecolors='white')
                    else:
                        ax1.scatter(trade_date, prices[idx], color='darkred', marker='v', s=100, zorder=5, edgecolors='white')
            
            ax1.set_title('Stock Price with Strategy Signals & Executed Trades')
            ax1.set_ylabel('Price ($)')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            # Plot 2: Portfolio Equity Curve
            ax2.plot(dates, self.equity_curve, label='Portfolio Value', linewidth=3, color='blue')
            ax2.set_title('Portfolio Equity Curve')
            ax2.set_ylabel('Portfolio Value ($)')
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            # Add performance annotation
            total_return = (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0] * 100
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
            ax3.set_title('Strategy Signal Distribution')
            
            # Plot 4: Trade Timeline
            if self.trade_log:
                trade_dates = [datetime.strptime(t['date'], "%Y-%m-%d") for t in self.trade_log]
                trade_values = [t.get('cost', 0) if t['action'] == 'BUY' else -t.get('proceeds', 0) for t in self.trade_log]
                trade_colors = ['green' if t['action'] == 'BUY' else 'red' for t in self.trade_log]
                
                ax4.bar(trade_dates, trade_values, color=trade_colors, alpha=0.7, width=1)
                ax4.set_title('Executed Trade Timeline (Green=Buy, Red=Sell)')
                ax4.set_ylabel('Trade Value ($)')
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax4.grid(True, alpha=0.3)
                ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                
                # Add trade count annotation
                buy_count = len([t for t in self.trade_log if t['action'] == 'BUY'])
                sell_count = len([t for t in self.trade_log if t['action'] == 'SELL'])
                ax4.text(0.02, 0.98, f'Executed: {buy_count} Buys, {sell_count} Sells', 
                        transform=ax4.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            else:
                ax4.text(0.5, 0.5, 'No Trades Executed', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
                ax4.set_title('Executed Trade Timeline')
            
            # Format x-axes
            for ax in [ax1, ax2, ax4]:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            filename = f"strategy_execution_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Execution analysis chart saved to: {filename}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping visualization")
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")


async def main():
    """Main function for strategy trade execution"""
    parser = argparse.ArgumentParser(description="Execute Trades from Strategy Signal Files")
    parser.add_argument("--strategy_flow", type=str, required=True,
                       help="Path to strategy flow JSON file")
    parser.add_argument("--symbol", type=str, default="AAPL",
                       help="Stock symbol for the strategy")
    parser.add_argument("--initial_cash", type=float, default=100000,
                       help="Initial cash for trading simulation")
    parser.add_argument("--output", type=str,
                       help="Output file prefix for execution results")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization of execution results")
    
    args = parser.parse_args()

    if not os.path.exists(args.strategy_flow):
        logger.error(f"❌ Strategy flow file not found: {args.strategy_flow}")
        return

    # Initialize trade executor
    executor = StrategyTradeExecutor(initial_cash=args.initial_cash)
    
    # Load strategy flow
    strategy_flows = executor.load_strategy_flow(args.strategy_flow)
    if not strategy_flows:
        logger.error("❌ No strategy flows loaded")
        return
    
    # Execute trades
    logger.info(f"🎯 Executing trades for {args.symbol} from {args.strategy_flow}")
    results = executor.execute_trades(strategy_flows, args.symbol)
    
    if results:
        # Save results
        output_file = executor.save_execution_results(args.symbol, args.output)
        
        # Create visualization if requested
        if args.visualize:
            executor.create_visualization(strategy_flows, args.symbol)
        
        logger.info(f"\n✅ Trade execution completed successfully!")
        logger.info(f"📄 Results saved to: {output_file}")
        logger.info(f"📊 Final Portfolio Value: ${results['final_value']:,.2f}")
        logger.info(f"📈 Total Return: {results['total_return']:.2f}%")
        logger.info(f"🔄 Total Trades: {len(results['trade_log'])}")
    else:
        logger.error("❌ Trade execution failed")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
