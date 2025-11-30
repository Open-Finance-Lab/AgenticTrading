"""
Standard Factor Calculator for Backtesting Framework
Handles calculation of various financial factors and indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from data_interfaces import FactorInput, FactorInterface


class StandardFactorCalculator(FactorInterface):
    """Standard factor calculator supporting multiple calculation methods"""
    
    def calculate(self, data: pd.DataFrame, factor_input: FactorInput) -> pd.Series:
        """Calculate factor based on input specification"""
        
        if factor_input.calculation_method == "expression":
            return self._calculate_expression_factor(data, factor_input)
        elif factor_input.calculation_method == "function":
            return self._calculate_function_factor(data, factor_input)
        else:
            # Default momentum factor
            return self._calculate_momentum_factor(data, factor_input.lookback_period)
    
    def _calculate_expression_factor(self, data: pd.DataFrame, factor_input: FactorInput) -> pd.Series:
        """Unified factor calculation method for all factor types"""
        
        # Convert to pivot format for vectorized calculation
        close_prices = data.pivot(index='date', columns='symbol', values='close')
        high_prices = data.pivot(index='date', columns='symbol', values='high')
        low_prices = data.pivot(index='date', columns='symbol', values='low')
        volume = data.pivot(index='date', columns='symbol', values='volume')
        
        # Calculate raw factor values using unified approach
        factor_values = self._unified_factor_calculation(
            close_prices, high_prices, low_prices, volume, factor_input
        )
        
        # Apply factor processing pipeline (standardization and neutralization)
        factor_values = self._process_factor(factor_values, factor_input)
        
        # Convert back to multi-index series
        result = factor_values.stack()
        result.index.names = ['date', 'symbol']
        
        return result
    
    def _unified_factor_calculation(self, close_prices: pd.DataFrame, high_prices: pd.DataFrame, 
                                  low_prices: pd.DataFrame, volume: pd.DataFrame, 
                                  factor_input: FactorInput) -> pd.DataFrame:
        """Unified calculation method for all factor types"""
        
        factor_name = factor_input.factor_name.lower()
        lookback = factor_input.lookback_period
        
        # Pre-calculate common components with proper handling of missing data
        returns = close_prices.pct_change(fill_method=None)  # Don't fill missing values automatically
        log_returns = np.log(close_prices / close_prices.shift(1))
        
        # === MOMENTUM FACTORS ===
        if "momentum" in factor_name:
            # Use exact matching with lookback_period to avoid name conflicts
            if lookback == 1:
                # 1-hour momentum: simple returns
                factor_values = returns
            elif lookback == 4:
                # 4-hour momentum: cumulative returns over 4 periods
                factor_values = close_prices.pct_change(4)
            elif lookback == 24:
                # 24-hour momentum: cumulative returns over 24 periods
                factor_values = close_prices.pct_change(24)
            else:
                # General momentum with volatility adjustment for other periods
                raw_returns = close_prices.pct_change(lookback)
                volatility = returns.rolling(lookback).std()
                factor_values = raw_returns / (volatility + 0.001)
                
        # === VOLATILITY FACTORS ===
        elif "volatility" in factor_name:
            if "2h" in factor_name or lookback == 2:
                # 2-hour volatility
                factor_values = returns.rolling(2).std()
            elif "realized_vol" in factor_name:
                # Realized volatility using high-low range
                factor_values = np.log(high_prices / low_prices).rolling(lookback).mean()
            else:
                # Standard rolling volatility
                factor_values = returns.rolling(lookback).std()
                
        # === TECHNICAL INDICATORS ===
        elif "rsi" in factor_name:
            # RSI with specified period
            period = 6 if "6h" in factor_name else lookback
            factor_values = self._calculate_rsi(close_prices, period)
            
        elif "bollinger" in factor_name:
            # Bollinger Band position
            period = 12 if "12h" in factor_name else lookback
            factor_values = self._calculate_bollinger_position(close_prices, period)
            
        elif "sma" in factor_name:
            # Simple Moving Average ratio
            sma = close_prices.rolling(lookback).mean()
            factor_values = close_prices / sma - 1
            
        elif "ema" in factor_name:
            # Exponential Moving Average ratio
            ema = close_prices.ewm(span=lookback).mean()
            factor_values = close_prices / ema - 1
            
        # === VOLUME FACTORS ===
        elif "volume" in factor_name:
            if "momentum" in factor_name:
                # Volume momentum
                period = 3 if "3h" in factor_name else lookback
                volume_ma = volume.rolling(period).mean()
                factor_values = (volume / volume_ma - 1)
            elif "price_volume" in factor_name:
                # Price-volume correlation
                factor_values = returns.rolling(lookback).corr(volume.pct_change())
            else:
                # Volume ratio
                volume_ma = volume.rolling(lookback).mean()
                factor_values = volume / volume_ma - 1
                
        # === PRICE PATTERN FACTORS ===
        elif "acceleration" in factor_name:
            # Price acceleration (second derivative)
            if lookback == 1:
                factor_values = returns.diff()
            else:
                momentum = close_prices.pct_change(lookback)
                factor_values = momentum.diff()
                
        elif "reversal" in factor_name:
            # Mean reversion factor
            returns_ma = returns.rolling(lookback).mean()
            factor_values = -returns_ma  # Negative for reversal
            
        elif "trend" in factor_name:
            # Trend strength using linear regression slope
            factor_values = self._calculate_trend_slope(close_prices, lookback)
            
        # === CROSS-SECTIONAL FACTORS ===
        elif "relative_strength" in factor_name:
            # Relative strength vs market
            market_return = returns.mean(axis=1)  # Equal-weighted market
            factor_values = returns.subtract(market_return, axis=0)
            
        elif "beta" in factor_name:
            # Rolling beta vs market
            market_return = returns.mean(axis=1)
            factor_values = self._calculate_rolling_beta(returns, market_return, lookback)
            
        # === DEFAULT CASE ===
        else:
            # Default to simple momentum
            factor_values = close_prices.pct_change(lookback)
        
        return factor_values.fillna(0).replace([np.inf, -np.inf], 0)
    
    def _calculate_trend_slope(self, prices: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate trend slope using linear regression"""
        
        def slope(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0] if not np.isnan(y).all() else 0
        
        return prices.rolling(window).apply(slope)
    
    def _calculate_rolling_beta(self, returns: pd.DataFrame, market_returns: pd.Series, window: int) -> pd.DataFrame:
        """Calculate rolling beta vs market"""
        
        def beta(stock_returns):
            if len(stock_returns) < 2 or len(market_returns) < len(stock_returns):
                return 0
            market_slice = market_returns.iloc[-len(stock_returns):]
            covariance = np.cov(stock_returns, market_slice)[0, 1]
            market_variance = np.var(market_slice)
            return covariance / market_variance if market_variance > 0 else 0
        
        return returns.rolling(window).apply(beta)
    
    def _calculate_bollinger_position(self, prices: pd.DataFrame, period: int) -> pd.DataFrame:
        """Calculate Bollinger Band position factor"""
        
        # Calculate moving average and standard deviation
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        # Calculate upper and lower bands
        upper_band = ma + 2 * std
        lower_band = ma - 2 * std
        
        # Calculate position within bands (-1 to 1)
        bollinger_position = (prices - ma) / (std + 0.001)  # Normalized position
        
        return bollinger_position.clip(-2, 2)  # Clip extreme values
    
    def _process_factor(self, raw_factor: pd.DataFrame, factor_input: FactorInput) -> pd.DataFrame:
        """Apply selective factor processing to preserve factor characteristics"""
        
        processed_factor = raw_factor.copy()
        
        # Apply enhanced winsorization to handle extreme outliers better
        processed_factor = self._winsorize_factor(processed_factor, quantile=0.01)  # More aggressive winsorization
        
        # Additional outlier treatment for specific factor types
        factor_name = factor_input.factor_name.lower()
        if 'momentum' in factor_name or 'acceleration' in factor_name:
            # Additional clipping for momentum-based factors to reduce noise
            for col in processed_factor.columns:
                processed_factor[col] = processed_factor[col].clip(
                    processed_factor[col].quantile(0.05), 
                    processed_factor[col].quantile(0.95)
                )
        
        # Skip cross-sectional neutralization and standardization to preserve factor diversity
        # This allows different factor types to maintain their unique characteristics
        # which is essential for effective machine learning
        
        return processed_factor
    
    def _winsorize_factor(self, factor: pd.DataFrame, quantile: float = 0.01) -> pd.DataFrame:
        """Winsorize factor values to remove extreme outliers"""
        
        winsorized = factor.copy()
        
        # Apply winsorization date by date (cross-sectional)
        for date in factor.index:
            date_values = factor.loc[date]
            if date_values.notna().sum() > 2:  # Need at least 3 valid values
                lower_bound = date_values.quantile(quantile)
                upper_bound = date_values.quantile(1 - quantile)
                winsorized.loc[date] = date_values.clip(lower_bound, upper_bound)
        
        return winsorized
    
    def _neutralize_factor(self, factor: pd.DataFrame, factor_input: FactorInput) -> pd.DataFrame:
        """Apply market neutralization to remove market beta"""
        
        neutralized = factor.copy()
        
        # Cross-sectional demeaning (market neutral)
        for date in factor.index:
            date_values = factor.loc[date]
            if date_values.notna().sum() > 1:
                # Remove cross-sectional mean (market neutralization)
                mean_value = date_values.mean()
                neutralized.loc[date] = date_values - mean_value
        
        return neutralized
    
    def _standardize_factor(self, factor: pd.DataFrame) -> pd.DataFrame:
        """Standardize factor to unit variance"""
        
        standardized = factor.copy()
        
        # Cross-sectional standardization
        for date in factor.index:
            date_values = factor.loc[date]
            if date_values.notna().sum() > 1:
                std_value = date_values.std()
                if std_value > 0:
                    standardized.loc[date] = date_values / std_value
                else:
                    standardized.loc[date] = 0
        
        return standardized
    
    def _calculate_function_factor(self, data: pd.DataFrame, factor_input: FactorInput) -> pd.Series:
        """Calculate factor using function"""
        # Placeholder for custom function calculations
        return self._calculate_momentum_factor(data, factor_input.lookback_period)
    
    def _calculate_momentum_factor(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate simple momentum factor"""
        
        close_prices = data.pivot(index='date', columns='symbol', values='close')
        momentum = close_prices.pct_change(period)
        
        result = momentum.stack()
        result.index.names = ['date', 'symbol']
        
        return result
    
    def _calculate_rsi(self, prices: pd.DataFrame, period: int) -> pd.DataFrame:
        """Calculate RSI factor with momentum component to reduce correlation with Bollinger"""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Add momentum component to differentiate from Bollinger (mean reversion)
        # RSI momentum: rate of change of RSI itself
        rsi_momentum = rsi.diff(2)  # 2-period RSI change
        
        # Combine traditional RSI with momentum component (70% RSI, 30% momentum)
        normalized_rsi = (rsi - 50) / 50  # Traditional RSI [-1, 1]
        normalized_momentum = rsi_momentum / 10  # Scale momentum component
        
        combined_rsi = 0.7 * normalized_rsi + 0.3 * normalized_momentum
        
        return combined_rsi.clip(-2, 2)  # Clip extreme values
    
    def validate_factor(self, factor_values: pd.Series) -> Dict[str, Any]:
        """Validate factor values"""
        
        factor_clean = factor_values.dropna()
        
        return {
            'total_values': len(factor_values),
            'valid_values': len(factor_clean),
            'valid_ratio': len(factor_clean) / len(factor_values) if len(factor_values) > 0 else 0,
            'mean': factor_clean.mean() if len(factor_clean) > 0 else 0,
            'std': factor_clean.std() if len(factor_clean) > 0 else 0,
            'min': factor_clean.min() if len(factor_clean) > 0 else 0,
            'max': factor_clean.max() if len(factor_clean) > 0 else 0
        }
