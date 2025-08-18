"""
Mean Reversion Agent for Alpha Strategy Generation

This agent implements mean reversion trading strategies based on statistical analysis
and market regime detection. It identifies securities that have deviated significantly
from their historical mean and generates signals for potential reversion opportunities.

Academic Framework:
Based on established mean reversion theories in quantitative finance, including
Ornstein-Uhlenbeck processes and statistical arbitrage methodologies.

Author: FinAgent Research Team
Created: 2025-07-25
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MeanReversionSignal:
    """Data structure for mean reversion trading signals."""
    symbol: str
    signal_strength: float
    current_price: float
    mean_price: float
    z_score: float
    lookback_period: int
    confidence: float
    timestamp: datetime
    expected_return: Optional[float] = None
    risk_metrics: Optional[Dict[str, float]] = None

class MeanReversionAgent:
    """
    Mean Reversion Agent for generating alpha signals based on statistical mean reversion.
    
    This agent analyzes price series for mean reversion opportunities using:
    - Z-score analysis
    - Rolling mean calculations
    - Volatility-adjusted signals
    - Risk-based position sizing
    """
    
    def __init__(self, coordinator=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Mean Reversion Agent with configuration parameters.
        
        Args:
            coordinator: Agent coordinator for cross-agent communication
            config: Configuration dictionary containing:
                - lookback_window: Period for mean calculation (default: 20)
                - z_threshold: Z-score threshold for signal generation (default: 2.0)
                - min_periods: Minimum periods required for calculation (default: 10)
                - volatility_adjustment: Whether to adjust for volatility (default: True)
        """
        self.coordinator = coordinator
        self.config = config or {}
        self.lookback_window = self.config.get('lookback_window', 20)
        self.z_threshold = self.config.get('z_threshold', 2.0)
        self.min_periods = self.config.get('min_periods', 10)
        self.volatility_adjustment = self.config.get('volatility_adjustment', True)
        
        logger.info(f"MeanReversionAgent initialized with lookback_window={self.lookback_window}, "
                   f"z_threshold={self.z_threshold}")
    
    async def initialize(self):
        """Initialize the agent asynchronously."""
        logger.info("🔧 Initializing Mean Reversion Agent")
        # Add any async initialization logic here
        logger.info("✅ Mean Reversion Agent initialization completed")
    
    async def get_health_status(self) -> str:
        """Get agent health status."""
        return "healthy"
    
    async def shutdown(self):
        """Shutdown the agent."""
        logger.info("🛑 Shutting down Mean Reversion Agent")
    
    async def discover_mean_reversion_factors(self, symbols: List[str], lookback_period: int = 20) -> Dict[str, Any]:
        """Discover mean reversion factors for given symbols."""
        logger.info(f"🔍 Discovering mean reversion factors for {len(symbols)} symbols")
        
        # Mock implementation for testing
        factors_discovered = []
        for i, symbol in enumerate(symbols):
            factors_discovered.append({
                "symbol": symbol,
                "factor_name": f"mean_reversion_{symbol.lower()}",
                "category": "mean_reversion",
                "z_score": 2.1 + (i * 0.2),
                "strength": 0.65 + (i * 0.1),
                "confidence": 0.80 - (i * 0.05)
            })
        
        return {
            "agent_id": "mean_reversion_agent",
            "factors_discovered": factors_discovered,
            "performance": {
                "factors_found": len(factors_discovered),
                "execution_duration": 0.5,
                "success_rate": 0.95
            }
        }
    
    def calculate_mean_reversion_signal(self, 
                                      price_data: pd.Series, 
                                      symbol: str) -> Optional[MeanReversionSignal]:
        """
        Calculate mean reversion signal for a given price series.
        
        Args:
            price_data: Time series of price data
            symbol: Stock symbol
            
        Returns:
            MeanReversionSignal object if signal is generated, None otherwise
        """
        try:
            if len(price_data) < self.min_periods:
                logger.warning(f"Insufficient data for {symbol}: {len(price_data)} periods")
                return None
            
            # Calculate rolling statistics
            rolling_mean = price_data.rolling(window=self.lookback_window, min_periods=self.min_periods).mean()
            rolling_std = price_data.rolling(window=self.lookback_window, min_periods=self.min_periods).std()
            
            # Get latest values
            current_price = price_data.iloc[-1]
            mean_price = rolling_mean.iloc[-1]
            std_price = rolling_std.iloc[-1]
            
            if pd.isna(mean_price) or pd.isna(std_price) or std_price == 0:
                logger.warning(f"Invalid statistics for {symbol}")
                return None
            
            # Calculate Z-score
            z_score = (current_price - mean_price) / std_price
            
            # Generate signal only if Z-score exceeds threshold
            if abs(z_score) < self.z_threshold:
                return None
            
            # Calculate signal strength (negative for mean reversion)
            signal_strength = -np.sign(z_score) * min(abs(z_score) / self.z_threshold, 3.0)
            
            # Calculate confidence based on Z-score magnitude
            confidence = min(abs(z_score) / self.z_threshold, 1.0)
            
            # Risk metrics
            volatility = rolling_std.iloc[-1] / mean_price if mean_price != 0 else 0
            risk_metrics = {
                'volatility': volatility,
                'z_score_magnitude': abs(z_score),
                'price_deviation': abs(current_price - mean_price) / mean_price if mean_price != 0 else 0
            }
            
            # Expected return estimate (simple mean reversion expectation)
            expected_return = (mean_price - current_price) / current_price if current_price != 0 else 0
            
            return MeanReversionSignal(
                symbol=symbol,
                signal_strength=signal_strength,
                current_price=current_price,
                mean_price=mean_price,
                z_score=z_score,
                lookback_period=self.lookback_window,
                confidence=confidence,
                timestamp=datetime.now(),
                expected_return=expected_return,
                risk_metrics=risk_metrics
            )
            
        except Exception as e:
            logger.error(f"Error calculating mean reversion signal for {symbol}: {e}")
            return None
    
    def generate_portfolio_signals(self, 
                                 price_data_dict: Dict[str, pd.Series]) -> List[MeanReversionSignal]:
        """
        Generate mean reversion signals for a portfolio of securities.
        
        Args:
            price_data_dict: Dictionary mapping symbols to price series
            
        Returns:
            List of MeanReversionSignal objects
        """
        signals = []
        
        for symbol, price_data in price_data_dict.items():
            signal = self.calculate_mean_reversion_signal(price_data, symbol)
            if signal:
                signals.append(signal)
                logger.info(f"Generated mean reversion signal for {symbol}: "
                           f"strength={signal.signal_strength:.3f}, z_score={signal.z_score:.3f}")
        
        # Sort by signal strength (absolute value)
        signals.sort(key=lambda x: abs(x.signal_strength), reverse=True)
        
        return signals
    
    def get_strategy_metadata(self) -> Dict[str, Any]:
        """Get metadata about the mean reversion strategy."""
        return {
            'strategy_type': 'mean_reversion',
            'lookback_window': self.lookback_window,
            'z_threshold': self.z_threshold,
            'min_periods': self.min_periods,
            'volatility_adjustment': self.volatility_adjustment,
            'description': 'Statistical mean reversion strategy using Z-score analysis'
        }

# Factory function for compatibility
def create_mean_reversion_agent(config: Optional[Dict[str, Any]] = None) -> MeanReversionAgent:
    """Create and return a MeanReversionAgent instance."""
    return MeanReversionAgent(config)

# For backward compatibility and testing
if __name__ == "__main__":
    # Simple test
    agent = MeanReversionAgent()
    
    # Generate test data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)  # Random walk with drift
    price_series = pd.Series(prices, index=dates)
    
    # Test signal generation
    signal = agent.calculate_mean_reversion_signal(price_series, 'TEST')
    if signal:
        print(f"Test signal generated: {signal}")
    else:
        print("No signal generated for test data")
