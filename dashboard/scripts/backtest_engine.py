#!/usr/bin/env python3
"""
Backtesting Engine for LLM Trading Agents

Simulates trading performance of different strategies:
1. Clawdy (AI Agent) - Smart trading with committee approval
2. Buy & Hold - Buy and hold baseline
3. DJIA Index - Market index tracking
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass, asdict
import numpy as np
from enum import Enum

# ============================================================================
# Data Classes
# ============================================================================

class TradeType(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: str
    symbol: str
    side: TradeType
    price: float
    quantity: int
    reason: str
    agent: str
    
    def __post_init__(self):
        if isinstance(self.side, str):
            self.side = TradeType[self.side]


@dataclass
class EquitySnapshot:
    """Daily equity snapshot."""
    timestamp: str
    equity: float
    cash: float
    positions_value: float
    daily_return: float
    cumulative_return: float


@dataclass
class BacktestMetrics:
    """Performance metrics."""
    total_return: float  # %
    annual_return: float  # %
    sharpe_ratio: float
    max_drawdown: float  # %
    win_rate: float  # %
    num_trades: int
    avg_trade_return: float  # %
    best_trade: float  # %
    worst_trade: float  # %


class PortfolioState:
    """Tracks portfolio during backtest."""
    
    def __init__(self, initial_capital: float, agent_name: str):
        self.initial_capital = initial_capital
        self.agent_name = agent_name
        self.cash = initial_capital
        self.positions: Dict[str, int] = {}  # symbol -> quantity
        self.trades: List[Trade] = []
        self.equity_history: List[EquitySnapshot] = []
        self.price_cache: Dict[str, float] = {}  # Current prices for quick lookup
        self.returns: List[float] = []  # Daily returns for Sharpe calculation
    
    def get_position_value(self, prices: Dict[str, float]) -> float:
        """Get current value of all positions."""
        value = 0.0
        for symbol, qty in self.positions.items():
            price = prices.get(symbol, self.price_cache.get(symbol, 0))
            value += qty * price
        return value
    
    def get_total_equity(self, prices: Dict[str, float]) -> float:
        """Get total account equity (cash + positions)."""
        return self.cash + self.get_position_value(prices)
    
    def buy(self, symbol: str, quantity: int, price: float, reason: str):
        """Execute buy order."""
        cost = quantity * price
        if cost > self.cash:
            # Reduce quantity to fit budget
            quantity = int(self.cash / price)
            if quantity == 0:
                return False
            cost = quantity * price
        
        self.cash -= cost
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        
        self.trades.append(Trade(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            side=TradeType.BUY,
            price=price,
            quantity=quantity,
            reason=reason,
            agent=self.agent_name
        ))
        
        return True
    
    def sell(self, symbol: str, quantity: int, price: float, reason: str):
        """Execute sell order."""
        held = self.positions.get(symbol, 0)
        if quantity > held:
            quantity = held
        
        if quantity == 0:
            return False
        
        revenue = quantity * price
        self.cash += revenue
        self.positions[symbol] -= quantity
        
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        
        self.trades.append(Trade(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            side=TradeType.SELL,
            price=price,
            quantity=quantity,
            reason=reason,
            agent=self.agent_name
        ))
        
        return True
    
    def record_daily_equity(self, timestamp: str, prices: Dict[str, float]):
        """Record equity at end of day."""
        total = self.get_total_equity(prices)
        positions_value = self.get_position_value(prices)
        
        prev_equity = self.equity_history[-1].equity if self.equity_history else self.initial_capital
        daily_return = ((total - prev_equity) / prev_equity) * 100 if prev_equity > 0 else 0
        
        cumulative_return = ((total - self.initial_capital) / self.initial_capital) * 100
        
        self.equity_history.append(EquitySnapshot(
            timestamp=timestamp,
            equity=total,
            cash=self.cash,
            positions_value=positions_value,
            daily_return=daily_return,
            cumulative_return=cumulative_return
        ))
        
        self.returns.append(daily_return / 100)  # Store as decimal for Sharpe
        self.price_cache = prices.copy()


class BacktestEngine:
    """Core backtesting engine."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.portfolios: Dict[str, PortfolioState] = {}
    
    def register_agent(self, agent_name: str):
        """Register a trading agent."""
        self.portfolios[agent_name] = PortfolioState(self.initial_capital, agent_name)
    
    def execute_trade(self, agent_name: str, symbol: str, side: str, quantity: int, 
                     price: float, reason: str):
        """Execute a trade for an agent."""
        if agent_name not in self.portfolios:
            raise ValueError(f"Agent {agent_name} not registered")
        
        portfolio = self.portfolios[agent_name]
        
        if side == "BUY":
            portfolio.buy(symbol, quantity, price, reason)
        elif side == "SELL":
            portfolio.sell(symbol, quantity, price, reason)
    
    def record_daily_snapshot(self, timestamp: str, prices: Dict[str, float]):
        """Record daily snapshot for all agents."""
        for portfolio in self.portfolios.values():
            portfolio.record_daily_equity(timestamp, prices)
    
    def calculate_metrics(self, agent_name: str) -> BacktestMetrics:
        """Calculate performance metrics for an agent."""
        if agent_name not in self.portfolios:
            raise ValueError(f"Agent {agent_name} not registered")
        
        portfolio = self.portfolios[agent_name]
        history = portfolio.equity_history
        
        if len(history) < 2:
            return BacktestMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Total return
        final_equity = history[-1].equity
        initial = portfolio.initial_capital
        total_return = ((final_equity - initial) / initial) * 100
        
        # Annual return (assuming ~252 trading days)
        days = len(history)
        years = days / 252
        annual_return = (((final_equity / initial) ** (1 / years)) - 1) * 100 if years > 0 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        returns = np.array(portfolio.returns)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        equities = np.array([snap.equity for snap in history])
        running_max = np.maximum.accumulate(equities)
        drawdowns = (equities - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100
        
        # Win rate and trade stats
        trades = portfolio.trades
        buy_trades = [t for t in trades if t.side == TradeType.BUY]
        
        profitable_trades = 0
        total_trade_return = 0
        best_return = 0
        worst_return = 0
        
        for i, buy_trade in enumerate(buy_trades):
            # Find corresponding sell
            sell_trades = [t for t in trades[trades.index(buy_trade)+1:] 
                          if t.symbol == buy_trade.symbol and t.side == TradeType.SELL]
            
            if sell_trades:
                sell = sell_trades[0]
                trade_return = ((sell.price - buy_trade.price) / buy_trade.price) * 100
                
                if trade_return > 0:
                    profitable_trades += 1
                
                total_trade_return += trade_return
                best_return = max(best_return, trade_return)
                worst_return = min(worst_return, trade_return)
        
        win_rate = (profitable_trades / len(buy_trades) * 100) if buy_trades else 0
        avg_trade_return = (total_trade_return / len(buy_trades)) if buy_trades else 0
        
        return BacktestMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            num_trades=len(trades),
            avg_trade_return=avg_trade_return,
            best_trade=best_return,
            worst_trade=worst_return
        )
    
    def get_equity_curve(self, agent_name: str) -> List[Dict]:
        """Get equity curve as list of dicts."""
        if agent_name not in self.portfolios:
            raise ValueError(f"Agent {agent_name} not registered")
        
        return [asdict(snap) for snap in self.portfolios[agent_name].equity_history]
    
    def export_results(self, output_path: Path):
        """Export backtest results to JSON."""
        results = {}
        
        for agent_name, portfolio in self.portfolios.items():
            metrics = self.calculate_metrics(agent_name)
            
            results[agent_name] = {
                "metrics": asdict(metrics),
                "equity_curve": self.get_equity_curve(agent_name),
                "trades": [
                    {
                        "timestamp": t.timestamp,
                        "symbol": t.symbol,
                        "side": t.side.value,
                        "price": t.price,
                        "quantity": t.quantity,
                        "reason": t.reason,
                        "agent": t.agent
                    }
                    for t in portfolio.trades
                ]
            }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        
        return results


# ============================================================================
# Utility Functions
# ============================================================================

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from daily returns."""
    returns = np.array(returns)
    if len(returns) < 2:
        return 0
    
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    
    if excess_returns.std() == 0:
        return 0
    
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown from equity curve."""
    equities = np.array(equity_curve)
    running_max = np.maximum.accumulate(equities)
    drawdown = (equities - running_max) / running_max
    return np.min(drawdown)


if __name__ == "__main__":
    print("Backtesting Engine loaded. Use this in your strategy scripts.")
