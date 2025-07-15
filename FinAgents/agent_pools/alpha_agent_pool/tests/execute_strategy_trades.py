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
# Backtesting framework
import backtrader as bt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Backtrader strategy for executing trades from strategy flow
class StrategyFlowBacktest(bt.Strategy):
    params = (
        ('signal_data', None),
    )
    """
    Backtrader strategy that executes trades based on pre-generated strategy flow signals.
    """
    def __init__(self):
        self.signal_data = self.params.signal_data  # list of signals
        self.order = None
        self.bar_index = 0
        self.log_records = []  # 每步记录
        self.prev_value = None
        self.stop_loss = 0.05           # 自动止损阈值（5%）
        self.entry_price = None         # 记录买入价格

    def next(self):
        """
        仅止亏，其余完全按策略流执行
        - BUY: 全仓买入
        - SELL: 全仓清仓
        - HOLD: 保持原仓位
        - 持仓期间如亏损超过stop_loss则强制清仓
        """
        if self.bar_index >= len(self.signal_data):
            self.bar_index += 1
            return
        signal_info = self.signal_data[self.bar_index]
        self.bar_index += 1
        signal = signal_info.get("signal", "HOLD")
        confidence = signal_info.get("confidence", 0.0)
        predicted_return = signal_info.get("predicted_return", None)
        price = self.datas[0].close[0]
        position = self.position.size

        # 按策略流决定仓位
        if signal == "BUY":
            target_ratio = 1.0
        elif signal == "SELL":
            target_ratio = 0.0
        else:
            target_ratio = (position * price) / self.broker.getvalue() if self.broker.getvalue() > 0 else 0.0

        # 持仓止亏机制
        if position > 0:
            if self.entry_price is None:
                self.entry_price = price
            pnl_ratio = (price - self.entry_price) / self.entry_price if self.entry_price else 0.0
            if pnl_ratio <= -self.stop_loss:
                target_ratio = 0.0
        else:
            self.entry_price = None

        # 计算目标持仓股数
        portfolio_value = self.broker.getvalue()
        target_shares = int(portfolio_value * target_ratio / price)
        shares_to_trade = target_shares - position

        # 执行交易
        if shares_to_trade > 0:
            self.order = self.buy(size=shares_to_trade)
        elif shares_to_trade < 0:
            self.order = self.sell(size=abs(shares_to_trade))
        # else: no trade

        # 记录前一天资金
        if self.prev_value is None:
            self.prev_value = portfolio_value

        # 记录每日收益
        cur_value = self.broker.getvalue()
        daily_return = (cur_value - self.prev_value) / self.prev_value if self.prev_value else 0.0
        self.prev_value = cur_value

        self.log_records.append({
            'bar_index': self.bar_index,
            'signal': signal,
            'confidence': confidence,
            'predicted_return': predicted_return,
            'price': price,
            'position': position,
            'target_ratio': target_ratio,
            'portfolio_value': cur_value,
            'daily_return': daily_return
        })

    def stop(self):
        # 回测结束后计算IC/IR等因子和常用绩效指标
        import numpy as np
        predicted_returns = []
        actual_returns = []
        confidences = []
        portfolio_values = []
        daily_returns = []
        for rec in self.log_records:
            if rec['predicted_return'] is not None:
                predicted_returns.append(rec['predicted_return'])
                actual_returns.append(rec['daily_return'])
                confidences.append(rec['confidence'])
            portfolio_values.append(rec['portfolio_value'])
            daily_returns.append(rec['daily_return'])
        # IC/IR
        ic = np.corrcoef(predicted_returns, actual_returns)[0,1] if len(predicted_returns) > 1 else None
        ic_mean = np.mean(ic) if ic is not None else None
        ic_std = np.std(ic) if ic is not None else None
        ir = ic_mean / ic_std if ic_mean is not None and ic_std and ic_std != 0 else None
        # CR% (累计收益率)
        if portfolio_values:
            cr = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        else:
            cr = None
        # ARR% (年化收益率)
        n_days = len(portfolio_values)
        if n_days > 1 and portfolio_values[0] > 0:
            arr = (portfolio_values[-1] / portfolio_values[0]) ** (252 / n_days) - 1
        else:
            arr = None
        # SR (夏普比率)
        if daily_returns and np.std(daily_returns) != 0:
            sr = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sr = None
        # MDD% (最大回撤)
        if portfolio_values:
            peak = np.maximum.accumulate(portfolio_values)
            drawdowns = (portfolio_values - peak) / peak
            mdd = drawdowns.min()
        else:
            mdd = None
        self.ic = ic
        self.ir = ir
        self.cr = cr
        self.arr = arr
        self.sr = sr
        self.mdd = mdd
        self.log_records_summary = {
            'ic': ic,
            'ir': ir,
            'cr': cr,
            'arr': arr,
            'sr': sr,
            'mdd': mdd,
            'log_records': self.log_records
        }

## No longer needed: strategy_flow_to_dataframe



def main():
    """
    Main function for strategy trade execution using Backtrader backtesting framework.
    """
    parser = argparse.ArgumentParser(description="Execute Trades from Strategy Signal Files (Backtrader)")
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
    parser.add_argument("--market_data", type=str, required=True,
                       help="Path to market data CSV file")

    args = parser.parse_args()

    if not os.path.exists(args.strategy_flow):
        logger.error(f"❌ Strategy flow file not found: {args.strategy_flow}")
        return


    # Parse strategy flow signals
    signal_map = parse_strategy_flow(args.strategy_flow)

    # Load market data CSV
    import pandas as pd
    market_df = pd.read_csv(args.market_data, parse_dates=['timestamp'])
    market_df.set_index('timestamp', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in market_df.columns:
            market_df[col] = market_df['vw'] if 'vw' in market_df.columns else 1
    market_df['volume'] = market_df['volume'].fillna(1)
    bt_df = market_df[['open', 'high', 'low', 'close', 'volume']].copy()

    # Create Backtrader data feed
    import backtrader.feeds as btfeeds
    data = btfeeds.PandasData(dataname=bt_df)

    # Set up Backtrader Cerebro engine
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(args.initial_cash)
    cerebro.adddata(data)
    cerebro.addstrategy(StrategyFlowBacktest, signal_data=signal_map)

    logger.info(f"🎯 Running Backtrader backtest for {args.symbol} using {args.market_data} and {args.strategy_flow}")
    # Run backtest
    result = cerebro.run()
    final_value = cerebro.broker.getvalue()
    logger.info(f"✅ Backtest completed. Final Portfolio Value: ${final_value:,.2f}")

    # 获取详细日志和因子值
    strat = result[0]
    ic = getattr(strat, 'ic', None)
    ir = getattr(strat, 'ir', None)
    cr = getattr(strat, 'cr', None)
    arr = getattr(strat, 'arr', None)
    sr = getattr(strat, 'sr', None)
    mdd = getattr(strat, 'mdd', None)
    log_records = getattr(strat, 'log_records', [])

    # Visualization (Backtrader plot)
    if args.visualize:
        cerebro.plot(style='candlestick')

    # Save results (详细因子与日志)
    output_file = args.output or f"bt_execution_results_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results = {
        "symbol": args.symbol,
        "initial_cash": args.initial_cash,
        "final_value": final_value,
        "strategy_flow_file": args.strategy_flow,
        "market_data_file": args.market_data,
        "ic": ic,
        "ir": ir,
        "cr": cr,
        "arr": arr,
        "sr": sr,
        "mdd": mdd,
        "log_records": log_records
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"📁 Backtest results saved to: {output_file}")


def parse_strategy_flow(strategy_flow_path):
    with open(strategy_flow_path, 'r', encoding='utf-8') as f:
        flow = json.load(f)
    signals = []
    for entry in flow:
        try:
            signal_info = entry['alpha_signals']['signals']['AAPL']
            signal = signal_info['decision']['signal']
            weight = signal_info['action'].get('execution_weight', signal_info['decision'].get('confidence', 0.0))
            confidence = signal_info['decision'].get('confidence', 0.0)
            predicted_return = signal_info['decision'].get('predicted_return', None)
            signals.append({'signal': signal, 'execution_weight': weight, 'confidence': confidence, 'predicted_return': predicted_return})
        except Exception as e:
            continue
    return signals


if __name__ == "__main__":
    main()
