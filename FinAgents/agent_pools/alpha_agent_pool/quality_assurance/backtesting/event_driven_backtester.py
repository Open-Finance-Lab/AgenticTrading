"""
Event-Driven Backtesting Framework for Alpha Agent Pool

This module implements a comprehensive event-driven backtesting system that
validates alpha strategies using event-based execution rather than time-based.
The framework focuses on transaction costs, market impact, and realistic
execution assumptions for quantitative trading strategies.

Key Features:
1. Event-driven execution engine with realistic latency modeling
2. Transaction cost analysis with market impact estimation
3. Cross-sectional alpha factor evaluation
4. Performance attribution and risk decomposition
5. Reinforcement learning policy validation

Author: FinAgent Quality Assurance Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of market events"""
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    PRICE_UPDATE = "price_update"
    SIGNAL_GENERATION = "signal_generation"
    ORDER_EXECUTION = "order_execution"
    RISK_CHECK = "risk_check"

@dataclass
class MarketEvent:
    """Market event container"""
    timestamp: datetime
    event_type: EventType
    symbol: str
    data: Dict[str, Any]
    priority: int = 1  # Lower numbers = higher priority

@dataclass 
class OrderEvent:
    """Order execution event"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market', 'limit'
    price: Optional[float] = None
    
@dataclass
class FillEvent:
    """Order fill event"""
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    fill_price: float
    commission: float
    market_impact: float

@dataclass
class PositionState:
    """Current position state"""
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    sortino_ratio: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    hit_rate: float = 0.0
    average_trade_pnl: float = 0.0
    trade_count: int = 0

class TransactionCostModel:
    """Transaction cost modeling with market impact"""
    
    def __init__(
        self,
        fixed_cost: float = 1.0,
        percentage_cost: float = 0.001,
        market_impact_factor: float = 0.0001
    ):
        """
        Initialize transaction cost model.
        
        Args:
            fixed_cost: Fixed cost per trade
            percentage_cost: Percentage cost of trade value
            market_impact_factor: Market impact as function of trade size
        """
        self.fixed_cost = fixed_cost
        self.percentage_cost = percentage_cost
        self.market_impact_factor = market_impact_factor
    
    def calculate_cost(
        self, 
        trade_value: float, 
        volume_participation: float = 0.01
    ) -> Tuple[float, float]:
        """
        Calculate transaction costs and market impact.
        
        Args:
            trade_value: Dollar value of trade
            volume_participation: Fraction of daily volume
            
        Returns:
            Tuple of (commission_cost, market_impact_cost)
        """
        commission = self.fixed_cost + abs(trade_value) * self.percentage_cost
        market_impact = abs(trade_value) * self.market_impact_factor * np.sqrt(volume_participation)
        
        return commission, market_impact

class EventDrivenBacktester:
    """
    Event-driven backtesting engine for alpha strategy validation.
    
    This backtester processes market events sequentially and executes
    trades based on alpha signals, providing realistic execution assumptions.
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        cost_model: Optional[TransactionCostModel] = None,
        max_position_size: float = 0.05,  # 5% max position per symbol
        leverage: float = 1.0
    ):
        """
        Initialize event-driven backtester.
        
        Args:
            initial_capital: Starting capital
            cost_model: Transaction cost model
            max_position_size: Maximum position size as fraction of capital
            leverage: Maximum leverage allowed
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()
        self.max_position_size = max_position_size
        self.leverage = leverage
        
        # Portfolio state
        self.positions: Dict[str, PositionState] = {}
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        
        # Event queues
        self.event_queue: List[MarketEvent] = []
        self.order_queue: List[OrderEvent] = []
        
        # Performance tracking
        self.trade_history: List[FillEvent] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        self.signal_history: List[Dict[str, Any]] = []
        
        # Risk limits
        self.position_limits: Dict[str, float] = {}
        
    def add_market_data(self, price_data: pd.DataFrame) -> None:
        """
        Add market data as price update events.
        
        Args:
            price_data: DataFrame with OHLCV data
        """
        for timestamp, row in price_data.iterrows():
            for symbol in price_data.columns:
                if symbol in row and not pd.isna(row[symbol]):
                    event = MarketEvent(
                        timestamp=timestamp,
                        event_type=EventType.PRICE_UPDATE,
                        symbol=symbol.split('_')[0] if '_' in symbol else symbol,
                        data={'price': row[symbol]},
                        priority=2
                    )
                    self.event_queue.append(event)
        
        # Sort events by timestamp and priority
        self.event_queue.sort(key=lambda x: (x.timestamp, x.priority))
    
    def add_alpha_signals(self, signals: pd.DataFrame) -> None:
        """
        Add alpha signals as signal generation events.
        
        Args:
            signals: DataFrame with alpha signals (dates x symbols)
        """
        for timestamp, row in signals.iterrows():
            for symbol in signals.columns:
                if symbol in row and not pd.isna(row[symbol]):
                    event = MarketEvent(
                        timestamp=timestamp,
                        event_type=EventType.SIGNAL_GENERATION,
                        symbol=symbol,
                        data={'signal': row[symbol]},
                        priority=1
                    )
                    self.event_queue.append(event)
        
        # Re-sort events after adding signals
        self.event_queue.sort(key=lambda x: (x.timestamp, x.priority))
    
    def process_signal_event(self, event: MarketEvent) -> None:
        """Process alpha signal generation event."""
        symbol = event.symbol
        signal = event.data['signal']
        timestamp = event.timestamp
        
        # Record signal for analysis
        self.signal_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'signal': signal
        })
        
        # Convert signal to target position
        target_value = signal * self.max_position_size * self.portfolio_value
        current_position = self.positions.get(symbol, PositionState(symbol)).quantity
        
        # Get current price (look for recent price update)
        current_price = self._get_current_price(symbol, timestamp)
        if current_price is None:
            logger.warning(f"No price available for {symbol} at {timestamp}")
            return
        
        # Calculate required trade
        target_quantity = target_value / current_price
        trade_quantity = target_quantity - current_position
        
        if abs(trade_quantity) > 0.001:  # Minimum trade threshold
            # Create order event
            order = OrderEvent(
                timestamp=timestamp,
                symbol=symbol,
                side='buy' if trade_quantity > 0 else 'sell',
                quantity=abs(trade_quantity),
                order_type='market'
            )
            self.order_queue.append(order)
    
    def process_order_event(self, order: OrderEvent) -> None:
        """Process order execution with transaction costs."""
        symbol = order.symbol
        timestamp = order.timestamp
        
        # Get execution price
        execution_price = self._get_current_price(symbol, timestamp)
        if execution_price is None:
            logger.warning(f"Cannot execute order for {symbol} - no price available")
            return
        
        # Calculate transaction costs
        trade_value = order.quantity * execution_price
        commission, market_impact = self.cost_model.calculate_cost(trade_value)
        
        # Adjust execution price for market impact
        if order.side == 'buy':
            fill_price = execution_price * (1 + market_impact / trade_value)
            quantity = order.quantity
        else:
            fill_price = execution_price * (1 - market_impact / trade_value)
            quantity = -order.quantity
        
        # Check capital constraints
        required_capital = abs(quantity * fill_price) + commission
        if required_capital > self.cash and order.side == 'buy':
            logger.warning(f"Insufficient capital for {symbol} order")
            return
        
        # Execute trade
        fill_event = FillEvent(
            timestamp=timestamp,
            symbol=symbol,
            side=order.side,
            quantity=abs(quantity),
            fill_price=fill_price,
            commission=commission,
            market_impact=market_impact
        )
        
        self._update_position(fill_event, quantity)
        self.trade_history.append(fill_event)
    
    def _update_position(self, fill: FillEvent, signed_quantity: float) -> None:
        """Update position state after trade execution."""
        symbol = fill.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = PositionState(symbol)
        
        position = self.positions[symbol]
        
        # Update position quantity and average cost
        old_quantity = position.quantity
        new_quantity = old_quantity + signed_quantity
        
        if new_quantity == 0:
            # Position closed
            position.realized_pnl += (fill.fill_price - position.avg_cost) * (-old_quantity)
            position.avg_cost = 0.0
        elif old_quantity == 0:
            # New position
            position.avg_cost = fill.fill_price
        elif np.sign(old_quantity) == np.sign(signed_quantity):
            # Adding to position
            total_cost = old_quantity * position.avg_cost + signed_quantity * fill.fill_price
            position.avg_cost = total_cost / new_quantity
        else:
            # Reducing position
            realized_quantity = min(abs(signed_quantity), abs(old_quantity))
            position.realized_pnl += (fill.fill_price - position.avg_cost) * realized_quantity * np.sign(signed_quantity)
        
        position.quantity = new_quantity
        
        # Update cash
        self.cash -= signed_quantity * fill.fill_price + fill.commission
    
    def _get_current_price(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """Get most recent price for symbol."""
        # Look for price events before or at this timestamp
        relevant_events = [
            e for e in self.event_queue 
            if e.symbol == symbol 
            and e.event_type == EventType.PRICE_UPDATE
            and e.timestamp <= timestamp
        ]
        
        if relevant_events:
            latest_event = max(relevant_events, key=lambda x: x.timestamp)
            return latest_event.data['price']
        
        return None
    
    def run_backtest(self) -> PerformanceMetrics:
        """
        Run the complete event-driven backtest.
        
        Returns:
            PerformanceMetrics with comprehensive performance analysis
        """
        logger.info(f"Starting event-driven backtest with {len(self.event_queue)} events")
        
        current_timestamp = None
        
        # Process all events in chronological order
        for event in self.event_queue:
            # Process any pending orders at timestamp boundaries
            if current_timestamp != event.timestamp:
                self._process_pending_orders(current_timestamp)
                self._update_portfolio_value(event.timestamp)
                current_timestamp = event.timestamp
            
            # Process market event
            if event.event_type == EventType.SIGNAL_GENERATION:
                self.process_signal_event(event)
        
        # Process final orders
        self._process_pending_orders(current_timestamp)
        
        # Calculate final performance metrics
        return self._calculate_performance_metrics()
    
    def _process_pending_orders(self, timestamp: Optional[datetime]) -> None:
        """Process all pending orders at current timestamp."""
        if timestamp is None:
            return
            
        orders_to_process = [
            order for order in self.order_queue
            if order.timestamp <= timestamp
        ]
        
        for order in orders_to_process:
            self.process_order_event(order)
            self.order_queue.remove(order)
    
    def _update_portfolio_value(self, timestamp: datetime) -> None:
        """Update portfolio valuation."""
        total_value = self.cash
        
        # Add position values
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                current_price = self._get_current_price(symbol, timestamp)
                if current_price:
                    position_value = position.quantity * current_price
                    position.unrealized_pnl = position_value - position.quantity * position.avg_cost
                    total_value += position_value
        
        self.portfolio_value = total_value
        
        # Record portfolio state
        self.portfolio_history.append({
            'timestamp': timestamp,
            'portfolio_value': total_value,
            'cash': self.cash,
            'positions': {k: v.quantity for k, v in self.positions.items()}
        })
    
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if not self.portfolio_history:
            return PerformanceMetrics()
        
        # Extract portfolio values
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('timestamp', inplace=True)
        
        values = portfolio_df['portfolio_value']
        returns = values.pct_change().dropna()
        
        if len(returns) < 2:
            return PerformanceMetrics()
        
        # Basic metrics
        total_return = (values.iloc[-1] / values.iloc[0]) - 1
        
        # Risk-adjusted metrics
        mean_return = returns.mean()
        return_std = returns.std()
        sharpe_ratio = mean_return / return_std * np.sqrt(252) if return_std > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        trade_pnls = [
            (t.quantity * t.fill_price * (-1 if t.side == 'sell' else 1))
            for t in self.trade_history
        ]
        
        win_trades = [pnl for pnl in trade_pnls if pnl > 0]
        loss_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        win_rate = len(win_trades) / len(trade_pnls) if trade_pnls else 0
        profit_factor = (sum(win_trades) / abs(sum(loss_trades))) if loss_trades else 0
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = mean_return / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = mean_return * 252 / abs(max_drawdown) if max_drawdown < 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            average_trade_pnl=np.mean(trade_pnls) if trade_pnls else 0,
            trade_count=len(trade_pnls)
        )

class AlphaFactorEvaluator:
    """
    Specialized evaluator for alpha factors using event-driven backtesting.
    
    This class combines factor quality assessment with realistic backtesting
    to provide comprehensive alpha factor validation.
    """
    
    def __init__(
        self,
        backtester: EventDrivenBacktester,
        benchmark_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize alpha factor evaluator.
        
        Args:
            backtester: Event-driven backtester instance
            benchmark_data: Benchmark returns for relative performance analysis
        """
        self.backtester = backtester
        self.benchmark_data = benchmark_data
    
    def evaluate_factor_strategy(
        self,
        factor_data: pd.DataFrame,
        price_data: pd.DataFrame,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate alpha factor using comprehensive backtesting.
        
        Args:
            factor_data: Alpha factor values (dates x symbols)
            price_data: Price data for backtesting
            strategy_params: Strategy-specific parameters
            
        Returns:
            Dictionary with evaluation results
        """
        # Convert factor values to position signals
        factor_signals = self._factor_to_signals(factor_data, strategy_params)
        
        # Run backtest
        self.backtester.add_market_data(price_data)
        self.backtester.add_alpha_signals(factor_signals)
        
        performance_metrics = self.backtester.run_backtest()
        
        # Additional factor-specific analysis
        factor_analysis = self._analyze_factor_characteristics(
            factor_data, self.backtester.signal_history
        )
        
        return {
            'performance_metrics': performance_metrics,
            'factor_analysis': factor_analysis,
            'trade_history': self.backtester.trade_history,
            'portfolio_history': self.backtester.portfolio_history
        }
    
    def _factor_to_signals(
        self,
        factor_data: pd.DataFrame,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Convert factor values to position signals."""
        params = strategy_params or {}
        
        # Default signal generation: rank-based
        ranking_method = params.get('ranking_method', 'cross_sectional')
        signal_cap = params.get('signal_cap', 1.0)
        
        signals = pd.DataFrame(index=factor_data.index, columns=factor_data.columns)
        
        for date in factor_data.index:
            factor_cross_section = factor_data.loc[date].dropna()
            
            if len(factor_cross_section) > 0:
                if ranking_method == 'cross_sectional':
                    # Cross-sectional ranking
                    ranks = factor_cross_section.rank(pct=True)
                    # Convert to signals [-1, 1]
                    date_signals = (ranks - 0.5) * 2 * signal_cap
                else:
                    # Direct factor values (normalized)
                    date_signals = factor_cross_section / factor_cross_section.abs().max() * signal_cap
                
                signals.loc[date, date_signals.index] = date_signals
        
        return signals.fillna(0)
    
    def _analyze_factor_characteristics(
        self,
        factor_data: pd.DataFrame,
        signal_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze factor characteristics from backtest results."""
        
        # Convert signal history to DataFrame
        signals_df = pd.DataFrame(signal_history)
        if signals_df.empty:
            return {}
        
        signals_pivot = signals_df.pivot(
            index='timestamp', 
            columns='symbol', 
            values='signal'
        )
        
        # Factor turnover analysis
        turnover_rates = []
        for i in range(1, len(signals_pivot)):
            prev_signals = signals_pivot.iloc[i-1].dropna()
            curr_signals = signals_pivot.iloc[i].dropna()
            
            common_symbols = prev_signals.index.intersection(curr_signals.index)
            if len(common_symbols) > 0:
                signal_changes = (prev_signals[common_symbols] - curr_signals[common_symbols]).abs()
                turnover_rate = signal_changes.mean()
                turnover_rates.append(turnover_rate)
        
        # Factor concentration analysis
        concentration_scores = []
        for _, row in signals_pivot.iterrows():
            active_signals = row.dropna()
            if len(active_signals) > 0:
                # Calculate Herfindahl index for concentration
                weights = active_signals.abs() / active_signals.abs().sum()
                herfindahl = (weights ** 2).sum()
                concentration_scores.append(herfindahl)
        
        return {
            'average_turnover': np.mean(turnover_rates) if turnover_rates else 0,
            'average_concentration': np.mean(concentration_scores) if concentration_scores else 0,
            'signal_count_mean': signals_pivot.count(axis=1).mean(),
            'signal_strength_mean': signals_pivot.abs().mean(axis=1).mean()
        }
