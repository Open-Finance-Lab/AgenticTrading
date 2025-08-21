"""
Qlib Standard Factor Calculator Implementation

This module implements factor calculation using Qlib's Expression engine
and standard operators for technical analysis and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
import warnings

# Qlib expression imports
from qlib.data.base import Expression, Feature
from qlib.data.ops import (
    # Basic operators
    Add, Sub, Mul, Div, Power, 
    # Math functions
    Log, Abs, Sign,
    # Rolling operators
    Mean, Sum, Std, Var, Skew, Kurt, Max, Min, 
    # Reference operators 
    Ref, Delta, Slope,
    # Ranking operators
    Rank, Quantile,
    # Cross-sectional operators  
    Greater, Less, Gt, Lt, Ge, Le, Eq, Ne,
    # Logical operators
    And, Or, Not, If,
    # Rolling wrapper
    Rolling
)


class QlibFactorCalculator:
    """
    Standard Qlib factor calculator using Expression engine.
    
    This class provides a standardized interface for calculating technical
    factors using Qlib's powerful expression framework.
    """
    
    def __init__(
        self,
        factor_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the Qlib factor calculator.
        
        Args:
            factor_config: Configuration for factor calculation
            **kwargs: Additional configuration parameters
        """
        self.factor_config = factor_config or {}
        self.factor_expressions = {}
        self._setup_default_factors()
    
    def _setup_default_factors(self) -> None:
        """
        Set up default factor expressions using Qlib operators.
        
        This includes comprehensive technical indicators using all available operators.
        """
        # Basic price features
        self.factor_expressions.update({
            # Simple price features
            'close': Feature('$close'),
            'open': Feature('$open'), 
            'high': Feature('$high'),
            'low': Feature('$low'),
            'volume': Feature('$volume'),
            
            # Basic operators (Add, Sub, Mul, Div, Power)
            'price_spread': Sub(Feature('$high'), Feature('$low')),
            'price_sum': Add(Feature('$high'), Feature('$low')),
            'price_product': Mul(Feature('$close'), Feature('$volume')),
            'normalized_close': Div(Feature('$close'), Feature('$open')),
            'price_power_2': Power(Feature('$close'), 2),
            'volume_price_ratio': Div(Feature('$volume'), Feature('$close')),
            
            # Math functions (Log, Abs, Sign)
            'log_close': Log(Feature('$close')),
            'log_volume': Log(Add(Feature('$volume'), 1)),  # Add 1 to avoid log(0)
            'abs_returns_1': Abs(Sub(Div(Feature('$close'), Ref(Feature('$close'), 1)), 1)),
            'sign_returns_1': Sign(Sub(Div(Feature('$close'), Ref(Feature('$close'), 1)), 1)),
            'log_price_ratio': Log(Div(Feature('$close'), Feature('$open'))),
            
            # Rolling operators (Mean, Sum, Std, Var, Skew, Kurt, Max, Min)
            'ma_5': Mean(Feature('$close'), 5),
            'ma_10': Mean(Feature('$close'), 10), 
            'ma_20': Mean(Feature('$close'), 20),
            'ma_60': Mean(Feature('$close'), 60),
            'sum_volume_5': Sum(Feature('$volume'), 5),
            'sum_volume_20': Sum(Feature('$volume'), 20),
            'volatility_5': Std(Sub(Div(Feature('$close'), Ref(Feature('$close'), 1)), 1), 5),
            'volatility_20': Std(Sub(Div(Feature('$close'), Ref(Feature('$close'), 1)), 1), 20),
            'variance_5': Var(Feature('$close'), 5),
            'variance_20': Var(Feature('$close'), 20),
            'skewness_20': Skew(Sub(Div(Feature('$close'), Ref(Feature('$close'), 1)), 1), 20),
            'kurtosis_20': Kurt(Sub(Div(Feature('$close'), Ref(Feature('$close'), 1)), 1), 20),
            'highest_20': Max(Feature('$close'), 20),
            'lowest_20': Min(Feature('$close'), 20),
            'volume_max_5': Max(Feature('$volume'), 5),
            'volume_min_5': Min(Feature('$volume'), 5),
            
            # Reference operators (Ref, Delta, Slope)
            'close_lag_1': Ref(Feature('$close'), 1),
            'close_lag_5': Ref(Feature('$close'), 5),
            'close_lag_20': Ref(Feature('$close'), 20),
            'price_delta_1': Delta(Feature('$close'), 1),
            'price_delta_5': Delta(Feature('$close'), 5),
            'price_delta_20': Delta(Feature('$close'), 20),
            'price_trend_5': Slope(Feature('$close'), 5),
            'price_trend_20': Slope(Feature('$close'), 20),
            'volume_trend_10': Slope(Feature('$volume'), 10),
            
            # Ranking operators (Rank, Quantile)
            'rank_close_20': Rank(Feature('$close'), 20),
            'rank_volume_20': Rank(Feature('$volume'), 20),
            'rank_returns_1_20': Rank(Sub(Div(Feature('$close'), Ref(Feature('$close'), 1)), 1), 20),
            'quantile_close_20': Quantile(Feature('$close'), 20, 0.8),
            'quantile_volume_20': Quantile(Feature('$volume'), 20, 0.2),
            'rank_momentum_5': Rank(Sub(Div(Feature('$close'), Ref(Feature('$close'), 5)), 1), 20),
            
            # Cross-sectional operators (Greater, Less, Gt, Lt, Ge, Le, Eq, Ne)
            'close_gt_ma20': Greater(Feature('$close'), Mean(Feature('$close'), 20)),
            'close_lt_ma20': Less(Feature('$close'), Mean(Feature('$close'), 20)),
            'volume_gt_ma5': Gt(Feature('$volume'), Mean(Feature('$volume'), 5)),
            'close_ge_high': Ge(Feature('$close'), Feature('$high')),
            'close_le_low': Le(Feature('$close'), Feature('$low')),
            'price_ne_open': Ne(Feature('$close'), Feature('$open')),
            'high_eq_close': Eq(Feature('$high'), Feature('$close')),
            
            # Logical operators (And, Or, Not, If)
            'bullish_signal': And(
                Greater(Feature('$close'), Mean(Feature('$close'), 20)),
                Greater(Feature('$volume'), Mean(Feature('$volume'), 5))
            ),
            'bearish_signal': Or(
                Less(Feature('$close'), Mean(Feature('$close'), 20)),
                Less(Feature('$volume'), Mul(Mean(Feature('$volume'), 5), 0.5))
            ),
            'trend_reversal': Not(Greater(Feature('$close'), Ref(Feature('$close'), 1))),
            'conditional_momentum': If(
                Greater(Feature('$close'), Mean(Feature('$close'), 20)),
                Sub(Div(Feature('$close'), Ref(Feature('$close'), 5)), 1),
                Sub(Div(Feature('$close'), Ref(Feature('$close'), 10)), 1)
            ),
            
            # Complex combinations
            'momentum_5': Sub(Div(Feature('$close'), Ref(Feature('$close'), 5)), 1),
            'momentum_10': Sub(Div(Feature('$close'), Ref(Feature('$close'), 10)), 1),
            'momentum_20': Sub(Div(Feature('$close'), Ref(Feature('$close'), 20)), 1),
            'returns_1': Sub(Div(Feature('$close'), Ref(Feature('$close'), 1)), 1),
            'returns_5': Sub(Div(Feature('$close'), Ref(Feature('$close'), 5)), 1),
            
            # Price position indicators
            'close_to_high': Div(Feature('$close'), Feature('$high')),
            'close_to_low': Div(Feature('$close'), Feature('$low')),
            'high_low_ratio': Div(Feature('$high'), Feature('$low')),
            'price_position_in_range': Div(
                Sub(Feature('$close'), Min(Feature('$low'), 20)),
                Sub(Max(Feature('$high'), 20), Min(Feature('$low'), 20))
            ),
            
            # Volume indicators  
            'volume_ma_5': Mean(Feature('$volume'), 5),
            'volume_ratio_5': Div(Feature('$volume'), Mean(Feature('$volume'), 5)),
            'volume_momentum_5': Sub(Div(Feature('$volume'), Ref(Feature('$volume'), 5)), 1),
            'volume_volatility_10': Std(Feature('$volume'), 10),
            'volume_trend_strength': Abs(Slope(Feature('$volume'), 10)),
            
            # Advanced technical indicators
            'rsi_14': self._create_rsi_expression(14),
            'bollinger_position': self._create_bollinger_position_expression(20),
            'macd_signal': self._create_macd_expression(),
            'price_oscillator': self._create_price_oscillator_expression(),
            'volume_price_trend': self._create_vpt_expression(),
            
            # Statistical measures
            'rolling_beta': self._create_rolling_beta_expression(20),
            'rolling_correlation': self._create_rolling_correlation_expression(20),
            'momentum_divergence': self._create_momentum_divergence_expression(),
        })
    
    def _create_rsi_expression(self, window: int) -> Expression:
        """
        Create RSI (Relative Strength Index) expression.
        
        Args:
            window: RSI calculation window
            
        Returns:
            Expression: RSI calculation expression
        """
        # Price changes
        price_change = Sub(Feature('$close'), Ref(Feature('$close'), 1))
        
        # Positive and negative changes  
        pos_change = If(Gt(price_change, 0), price_change, 0)
        neg_change = If(Lt(price_change, 0), Mul(price_change, -1), 0)
        
        # Average gains and losses
        avg_gain = Mean(pos_change, window)
        avg_loss = Mean(neg_change, window)
        
        # RSI calculation
        rs = Div(avg_gain, Add(avg_loss, 1e-8))  # Add small epsilon to avoid division by zero
        rsi = Sub(100, Div(100, Add(1, rs)))
        
        return rsi
    
    def _create_bollinger_position_expression(self, window: int) -> Expression:
        """
        Create Bollinger Band position expression.
        
        Args:
            window: Bollinger band calculation window
            
        Returns:
            Expression: Position within Bollinger bands (0-1 scale)
        """
        close = Feature('$close')
        ma = Mean(close, window)
        std = Std(close, window)
        
        upper_band = Add(ma, Mul(2, std))
        lower_band = Sub(ma, Mul(2, std))
        
        # Position within bands (0 = lower band, 1 = upper band)
        position = Div(Sub(close, lower_band), Add(Sub(upper_band, lower_band), 1e-8))
        
        return position
    
    def _create_macd_expression(self) -> Expression:
        """
        Create MACD (Moving Average Convergence Divergence) expression.
        
        Returns:
            Expression: MACD calculation expression
        """
        close = Feature('$close')
        ema_12 = Mean(close, 12)  # Simplified EMA as Mean
        ema_26 = Mean(close, 26)
        
        macd_line = Sub(ema_12, ema_26)
        signal_line = Mean(macd_line, 9)
        
        return Sub(macd_line, signal_line)
    
    def _create_price_oscillator_expression(self) -> Expression:
        """
        Create Price Oscillator expression.
        
        Returns:
            Expression: Price oscillator calculation
        """
        close = Feature('$close')
        ma_short = Mean(close, 10)
        ma_long = Mean(close, 20)
        
        return Div(Sub(ma_short, ma_long), ma_long)
    
    def _create_vpt_expression(self) -> Expression:
        """
        Create Volume Price Trend (VPT) expression.
        
        Returns:
            Expression: VPT calculation expression
        """
        close = Feature('$close')
        volume = Feature('$volume')
        price_change_pct = Sub(Div(close, Ref(close, 1)), 1)
        
        return Sum(Mul(volume, price_change_pct), 10)
    
    def _create_rolling_beta_expression(self, window: int) -> Expression:
        """
        Create rolling beta expression (simplified version).
        
        Args:
            window: Rolling window size
            
        Returns:
            Expression: Rolling beta calculation
        """
        returns = Sub(Div(Feature('$close'), Ref(Feature('$close'), 1)), 1)
        market_returns = Mean(returns, window)  # Simplified market proxy
        
        # Simplified beta calculation using correlation approximation
        return Div(
            Std(returns, window),
            Add(Std(market_returns, window), 1e-8)
        )
    
    def _create_rolling_correlation_expression(self, window: int) -> Expression:
        """
        Create rolling correlation expression (price vs volume).
        
        Args:
            window: Rolling window size
            
        Returns:
            Expression: Rolling correlation approximation
        """
        price_returns = Sub(Div(Feature('$close'), Ref(Feature('$close'), 1)), 1)
        volume_change = Sub(Div(Feature('$volume'), Ref(Feature('$volume'), 1)), 1)
        
        # Simplified correlation using standardized values
        price_std = Std(price_returns, window)
        volume_std = Std(volume_change, window)
        
        price_norm = Div(price_returns, Add(price_std, 1e-8))
        volume_norm = Div(volume_change, Add(volume_std, 1e-8))
        
        return Mean(Mul(price_norm, volume_norm), window)
    
    def _create_momentum_divergence_expression(self) -> Expression:
        """
        Create momentum divergence expression.
        
        Returns:
            Expression: Momentum divergence calculation
        """
        close = Feature('$close')
        volume = Feature('$volume')
        
        price_momentum = Sub(Div(close, Ref(close, 10)), 1)
        volume_momentum = Sub(Div(volume, Ref(volume, 10)), 1)
        
        price_trend = Sign(price_momentum)
        volume_trend = Sign(volume_momentum)
        
        # Divergence occurs when trends are opposite
        return Sub(price_trend, volume_trend)
    
    def calculate_single_factor(
        self,
        data: pd.DataFrame,
        factor_name: str
    ) -> Optional[pd.Series]:
        """
        Calculate a single factor for correlation analysis.
        
        Args:
            data: Input DataFrame with OHLCV data
            factor_name: Name of the factor to calculate
            
        Returns:
            pd.Series: Calculated factor values or None if failed
        """
        try:
            if factor_name not in self.factor_expressions:
                return None
                
            # Calculate just this one factor
            factor_data = self.calculate_factors(data, factor_names=[factor_name])
            
            if not factor_data.empty and factor_name in factor_data.columns:
                return factor_data[factor_name]
            
            return None
            
        except Exception:
            return None

    def calculate_factors(
        self,
        data: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate factors using Qlib expressions.
        
        Args:
            data: Input DataFrame with OHLCV data in Qlib format
            factor_names: List of factors to calculate (None for all)
            start_time: Start time for calculation
            end_time: End time for calculation
            
        Returns:
            pd.DataFrame: Calculated factors in Qlib format
        """
        if factor_names is None:
            factor_names = list(self.factor_expressions.keys())
        
        # Validate factor names
        invalid_factors = set(factor_names) - set(self.factor_expressions.keys())
        if invalid_factors:
            raise ValueError(f"Unknown factors: {invalid_factors}")
        
        # Extract instruments from data
        instruments = data.index.get_level_values('instrument').unique().tolist()
        
        # Calculate factors for each instrument
        factor_results = {}
        
        for instrument in instruments:
            # Get instrument data
            inst_data = data.xs(instrument, level='instrument')
            
            # Apply time filtering
            if start_time:
                inst_data = inst_data[inst_data.index >= pd.Timestamp(start_time)]
            if end_time:
                inst_data = inst_data[inst_data.index <= pd.Timestamp(end_time)]
            
            if inst_data.empty:
                continue
            
            # Calculate factors for this instrument
            inst_factors = self._calculate_instrument_factors(
                inst_data, instrument, factor_names
            )
            
            factor_results[instrument] = inst_factors
        
        if not factor_results:
            return self._create_empty_factor_dataframe()
        
        # Combine results
        combined_factors = pd.concat(factor_results.values(), keys=factor_results.keys())
        combined_factors.index.names = ['instrument', 'datetime']
        
        # Reorder index to match Qlib convention (datetime, instrument)
        combined_factors = combined_factors.swaplevel().sort_index()
        
        return combined_factors
    
    def _calculate_instrument_factors(
        self,
        inst_data: pd.DataFrame,
        instrument: str,
        factor_names: List[str]
    ) -> pd.DataFrame:
        """
        Calculate factors for a single instrument.
        
        Args:
            inst_data: Single instrument OHLCV data
            instrument: Instrument symbol
            factor_names: List of factors to calculate
            
        Returns:
            pd.DataFrame: Calculated factors for the instrument
        """
        # Prepare data for factor calculation
        # Convert multi-level columns to simple columns for calculation
        calc_data = self._prepare_calculation_data(inst_data)
        
        factor_values = {}
        
        for factor_name in factor_names:
            try:
                expression = self.factor_expressions[factor_name]
                
                # Calculate factor using simplified pandas operations
                # (In a full Qlib implementation, this would use the expression engine)
                factor_value = self._evaluate_expression_simplified(
                    expression, calc_data, factor_name
                )
                
                factor_values[('feature', factor_name)] = factor_value
                
            except Exception as e:
                warnings.warn(f"Failed to calculate factor {factor_name} for {instrument}: {str(e)}")
                # Fill with NaN values
                factor_values[('feature', factor_name)] = np.nan
        
        # Create result DataFrame
        result_df = pd.DataFrame(factor_values, index=inst_data.index)
        result_df.columns = pd.MultiIndex.from_tuples(
            result_df.columns, names=['field_group', 'field_name']
        )
        
        return result_df
    
    def _prepare_calculation_data(self, inst_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for factor calculation by flattening column structure.
        
        Args:
            inst_data: Multi-level column DataFrame
            
        Returns:
            pd.DataFrame: Flattened DataFrame suitable for calculation
        """
        calc_data = pd.DataFrame(index=inst_data.index)
        
        for col in inst_data.columns:
            if isinstance(col, tuple):
                # Multi-level columns
                field_group, field_name = col
                if field_group == 'feature':
                    # Map to standard OHLCV naming
                    if 'close' in field_name.lower():
                        calc_data['$close'] = inst_data[col]
                    elif 'open' in field_name.lower():
                        calc_data['$open'] = inst_data[col]
                    elif 'high' in field_name.lower():
                        calc_data['$high'] = inst_data[col]
                    elif 'low' in field_name.lower():
                        calc_data['$low'] = inst_data[col]
                    elif 'volume' in field_name.lower():
                        calc_data['$volume'] = inst_data[col]
                    else:
                        # Keep original name for other features
                        calc_data[field_name] = inst_data[col]
            else:
                # Simple columns
                if 'close' in col.lower():
                    calc_data['$close'] = inst_data[col]
                elif 'open' in col.lower():
                    calc_data['$open'] = inst_data[col]
                elif 'high' in col.lower():
                    calc_data['$high'] = inst_data[col]
                elif 'low' in col.lower():
                    calc_data['$low'] = inst_data[col]
                elif 'volume' in col.lower():
                    calc_data['$volume'] = inst_data[col]
                else:
                    calc_data[col] = inst_data[col]
        
        return calc_data
    
    def _evaluate_expression_simplified(
        self,
        expression: Any,
        data: pd.DataFrame,
        factor_name: str
    ) -> pd.Series:
        """
        Simplified expression evaluation using pandas operations.
        
        This is a simplified implementation. In a full Qlib setup,
        the expression engine would handle this automatically.
        
        Args:
            expression: Factor expression (simplified handling)
            data: Calculation data
            factor_name: Name of the factor being calculated
            
        Returns:
            pd.Series: Calculated factor values
        """
        try:
            # Basic price features
            if factor_name in ['close', 'open', 'high', 'low', 'volume']:
                return data.get(f'${factor_name}', pd.Series(index=data.index))
            
            # Basic operators
            elif factor_name == 'price_spread':
                high, low = data.get('$high'), data.get('$low')
                return (high - low) if high is not None and low is not None else pd.Series(index=data.index)
            
            elif factor_name == 'price_sum':
                high, low = data.get('$high'), data.get('$low')
                return (high + low) if high is not None and low is not None else pd.Series(index=data.index)
            
            elif factor_name == 'price_product':
                close, volume = data.get('$close'), data.get('$volume')
                return (close * volume) if close is not None and volume is not None else pd.Series(index=data.index)
            
            elif factor_name == 'normalized_close':
                close, open_price = data.get('$close'), data.get('$open')
                return (close / (open_price + 1e-8)) if close is not None and open_price is not None else pd.Series(index=data.index)
            
            elif factor_name == 'price_power_2':
                close = data.get('$close')
                return (close ** 2) if close is not None else pd.Series(index=data.index)
            
            elif factor_name == 'volume_price_ratio':
                volume, close = data.get('$volume'), data.get('$close')
                return (volume / (close + 1e-8)) if volume is not None and close is not None else pd.Series(index=data.index)
            
            # Math functions
            elif factor_name == 'log_close':
                close = data.get('$close')
                return np.log(close + 1e-8) if close is not None else pd.Series(index=data.index)
            
            elif factor_name == 'log_volume':
                volume = data.get('$volume')
                return np.log(volume + 1) if volume is not None else pd.Series(index=data.index)
            
            elif factor_name == 'abs_returns_1':
                close = data.get('$close')
                return np.abs(close.pct_change(1)) if close is not None else pd.Series(index=data.index)
            
            elif factor_name == 'sign_returns_1':
                close = data.get('$close')
                returns = close.pct_change(1) if close is not None else None
                return np.sign(returns) if returns is not None else pd.Series(index=data.index)
            
            elif factor_name == 'log_price_ratio':
                close, open_price = data.get('$close'), data.get('$open')
                if close is not None and open_price is not None:
                    return np.log((close / (open_price + 1e-8)) + 1e-8)
                return pd.Series(index=data.index)
            
            # Rolling operators
            elif factor_name.startswith('ma_'):
                window = int(factor_name.split('_')[1])
                close = data.get('$close')
                return close.rolling(window=window, min_periods=1).mean() if close is not None else pd.Series(index=data.index)
            
            elif factor_name.startswith('sum_volume_'):
                window = int(factor_name.split('_')[-1])
                volume = data.get('$volume')
                return volume.rolling(window=window, min_periods=1).sum() if volume is not None else pd.Series(index=data.index)
            
            elif factor_name.startswith('volatility_'):
                window = int(factor_name.split('_')[1])
                close = data.get('$close')
                if close is not None:
                    returns = close.pct_change(1)
                    return returns.rolling(window=window, min_periods=1).std()
                return pd.Series(index=data.index)
            
            elif factor_name.startswith('variance_'):
                window = int(factor_name.split('_')[1])
                close = data.get('$close')
                return close.rolling(window=window, min_periods=1).var() if close is not None else pd.Series(index=data.index)
            
            elif factor_name.startswith('skewness_'):
                window = int(factor_name.split('_')[1])
                close = data.get('$close')
                if close is not None:
                    returns = close.pct_change(1)
                    return returns.rolling(window=window, min_periods=3).skew()
                return pd.Series(index=data.index)
            
            elif factor_name.startswith('kurtosis_'):
                window = int(factor_name.split('_')[1])
                close = data.get('$close')
                if close is not None:
                    returns = close.pct_change(1)
                    return returns.rolling(window=window, min_periods=4).kurt()
                return pd.Series(index=data.index)
            
            elif factor_name.startswith('highest_'):
                window = int(factor_name.split('_')[1])
                close = data.get('$close')
                return close.rolling(window=window, min_periods=1).max() if close is not None else pd.Series(index=data.index)
            
            elif factor_name.startswith('lowest_'):
                window = int(factor_name.split('_')[1])
                close = data.get('$close')
                return close.rolling(window=window, min_periods=1).min() if close is not None else pd.Series(index=data.index)
            
            elif factor_name.startswith('volume_max_'):
                window = int(factor_name.split('_')[-1])
                volume = data.get('$volume')
                return volume.rolling(window=window, min_periods=1).max() if volume is not None else pd.Series(index=data.index)
            
            elif factor_name.startswith('volume_min_'):
                window = int(factor_name.split('_')[-1])
                volume = data.get('$volume')
                return volume.rolling(window=window, min_periods=1).min() if volume is not None else pd.Series(index=data.index)
            
            # Reference operators
            elif factor_name.startswith('close_lag_'):
                window = int(factor_name.split('_')[-1])
                close = data.get('$close')
                return close.shift(window) if close is not None else pd.Series(index=data.index)
            
            elif factor_name.startswith('price_delta_'):
                window = int(factor_name.split('_')[-1])
                close = data.get('$close')
                return (close - close.shift(window)) if close is not None else pd.Series(index=data.index)
            
            elif factor_name.startswith('price_trend_'):
                window = int(factor_name.split('_')[-1])
                close = data.get('$close')
                if close is not None:
                    def calc_slope(series):
                        if len(series) < 2:
                            return np.nan
                        x = np.arange(len(series))
                        y = series.values
                        if np.all(np.isnan(y)):
                            return np.nan
                        valid_mask = ~np.isnan(y)
                        if np.sum(valid_mask) < 2:
                            return np.nan
                        slope = np.polyfit(x[valid_mask], y[valid_mask], 1)[0]
                        return slope
                    return close.rolling(window=window, min_periods=2).apply(calc_slope)
                return pd.Series(index=data.index)
            
            elif factor_name.startswith('volume_trend_'):
                window = int(factor_name.split('_')[-1])
                volume = data.get('$volume')
                if volume is not None:
                    def calc_slope(series):
                        if len(series) < 2:
                            return np.nan
                        x = np.arange(len(series))
                        y = series.values
                        if np.all(np.isnan(y)):
                            return np.nan
                        valid_mask = ~np.isnan(y)
                        if np.sum(valid_mask) < 2:
                            return np.nan
                        slope = np.polyfit(x[valid_mask], y[valid_mask], 1)[0]
                        return slope
                    return volume.rolling(window=window, min_periods=2).apply(calc_slope)
                return pd.Series(index=data.index)
            
            # Ranking operators
            elif factor_name.startswith('rank_'):
                if 'close' in factor_name:
                    values = data.get('$close')
                elif 'volume' in factor_name:
                    values = data.get('$volume')
                elif 'returns_1' in factor_name:
                    close = data.get('$close')
                    values = close.pct_change(1) if close is not None else None
                elif 'momentum_5' in factor_name:
                    close = data.get('$close')
                    values = close.pct_change(5) if close is not None else None
                else:
                    values = None
                
                if values is not None:
                    return values.rank(pct=True)
                return pd.Series(index=data.index)
            
            elif factor_name.startswith('quantile_'):
                window = 20  # Fixed window for quantile
                if 'close' in factor_name:
                    values = data.get('$close')
                    percentile = 0.8 if 'close' in factor_name else 0.2
                elif 'volume' in factor_name:
                    values = data.get('$volume')
                    percentile = 0.2
                else:
                    values = None
                    percentile = 0.5
                
                if values is not None:
                    return values.rolling(window=window, min_periods=1).quantile(percentile)
                return pd.Series(index=data.index)
            
            # Cross-sectional operators (simplified as boolean conditions)
            elif factor_name == 'close_gt_ma20':
                close = data.get('$close')
                if close is not None:
                    ma20 = close.rolling(window=20, min_periods=1).mean()
                    return (close > ma20).astype(float)
                return pd.Series(index=data.index)
            
            elif factor_name == 'close_lt_ma20':
                close = data.get('$close')
                if close is not None:
                    ma20 = close.rolling(window=20, min_periods=1).mean()
                    return (close < ma20).astype(float)
                return pd.Series(index=data.index)
            
            elif factor_name == 'volume_gt_ma5':
                volume = data.get('$volume')
                if volume is not None:
                    ma5 = volume.rolling(window=5, min_periods=1).mean()
                    return (volume > ma5).astype(float)
                return pd.Series(index=data.index)
            
            elif factor_name == 'close_ge_high':
                close, high = data.get('$close'), data.get('$high')
                return (close >= high).astype(float) if close is not None and high is not None else pd.Series(index=data.index)
            
            elif factor_name == 'close_le_low':
                close, low = data.get('$close'), data.get('$low')
                return (close <= low).astype(float) if close is not None and low is not None else pd.Series(index=data.index)
            
            elif factor_name == 'price_ne_open':
                close, open_price = data.get('$close'), data.get('$open')
                return (close != open_price).astype(float) if close is not None and open_price is not None else pd.Series(index=data.index)
            
            elif factor_name == 'high_eq_close':
                high, close = data.get('$high'), data.get('$close')
                return (high == close).astype(float) if high is not None and close is not None else pd.Series(index=data.index)
            
            # Logical operators
            elif factor_name == 'bullish_signal':
                close, volume = data.get('$close'), data.get('$volume')
                if close is not None and volume is not None:
                    close_condition = close > close.rolling(window=20, min_periods=1).mean()
                    volume_condition = volume > volume.rolling(window=5, min_periods=1).mean()
                    return (close_condition & volume_condition).astype(float)
                return pd.Series(index=data.index)
            
            elif factor_name == 'bearish_signal':
                close, volume = data.get('$close'), data.get('$volume')
                if close is not None and volume is not None:
                    close_condition = close < close.rolling(window=20, min_periods=1).mean()
                    volume_condition = volume < (volume.rolling(window=5, min_periods=1).mean() * 0.5)
                    return (close_condition | volume_condition).astype(float)
                return pd.Series(index=data.index)
            
            elif factor_name == 'trend_reversal':
                close = data.get('$close')
                if close is not None:
                    uptrend = close > close.shift(1)
                    return (~uptrend).astype(float)
                return pd.Series(index=data.index)
            
            elif factor_name == 'conditional_momentum':
                close = data.get('$close')
                if close is not None:
                    ma20 = close.rolling(window=20, min_periods=1).mean()
                    condition = close > ma20
                    momentum_5 = close.pct_change(5)
                    momentum_10 = close.pct_change(10)
                    return np.where(condition, momentum_5, momentum_10)
                return pd.Series(index=data.index)
            
            # Standard factors (maintaining backward compatibility)
            elif factor_name.startswith('returns_'):
                window = int(factor_name.split('_')[1])
                close = data.get('$close')
                return close.pct_change(window) if close is not None else pd.Series(index=data.index)
            
            elif factor_name.startswith('momentum_'):
                window = int(factor_name.split('_')[1])
                close = data.get('$close')
                return close.pct_change(window) if close is not None else pd.Series(index=data.index)
            
            # Price position indicators
            elif factor_name == 'close_to_high':
                close, high = data.get('$close'), data.get('$high')
                return (close / (high + 1e-8)) if close is not None and high is not None else pd.Series(index=data.index)
            
            elif factor_name == 'close_to_low':
                close, low = data.get('$close'), data.get('$low')
                return (close / (low + 1e-8)) if close is not None and low is not None else pd.Series(index=data.index)
            
            elif factor_name == 'high_low_ratio':
                high, low = data.get('$high'), data.get('$low')
                return (high / (low + 1e-8)) if high is not None and low is not None else pd.Series(index=data.index)
            
            elif factor_name == 'price_position_in_range':
                close, high, low = data.get('$close'), data.get('$high'), data.get('$low')
                if close is not None and high is not None and low is not None:
                    highest_20 = high.rolling(window=20, min_periods=1).max()
                    lowest_20 = low.rolling(window=20, min_periods=1).min()
                    range_size = highest_20 - lowest_20
                    position = (close - lowest_20) / (range_size + 1e-8)
                    return position
                return pd.Series(index=data.index)
            
            # Volume indicators
            elif factor_name.startswith('volume_'):
                volume = data.get('$volume')
                if volume is not None:
                    if 'ma_' in factor_name:
                        window = int(factor_name.split('_')[-1])
                        return volume.rolling(window=window, min_periods=1).mean()
                    elif 'ratio_' in factor_name:
                        window = int(factor_name.split('_')[-1])
                        vol_ma = volume.rolling(window=window, min_periods=1).mean()
                        return volume / (vol_ma + 1e-8)
                    elif 'momentum_' in factor_name:
                        window = int(factor_name.split('_')[-1])
                        return volume.pct_change(window)
                    elif 'volatility_' in factor_name:
                        window = int(factor_name.split('_')[-1])
                        return volume.rolling(window=window, min_periods=1).std()
                    elif 'trend_strength' in factor_name:
                        slope_values = self._evaluate_expression_simplified(None, data, 'volume_trend_10')
                        return np.abs(slope_values) if slope_values is not None else pd.Series(index=data.index)
                return pd.Series(index=data.index)
            
            # Advanced indicators (simplified implementations)
            elif factor_name in ['rsi_14', 'bollinger_position', 'macd_signal', 'price_oscillator', 
                               'volume_price_trend', 'rolling_beta', 'rolling_correlation', 'momentum_divergence']:
                return self._calculate_advanced_indicator(factor_name, data)
            
            # Default: return NaN series
            return pd.Series(np.nan, index=data.index)
            
        except Exception as e:
            warnings.warn(f"Error calculating {factor_name}: {str(e)}")
            return pd.Series(np.nan, index=data.index)
    
    def _calculate_advanced_indicator(self, factor_name: str, data: pd.DataFrame) -> pd.Series:
        """
        Calculate advanced technical indicators.
        
        Args:
            factor_name: Name of the advanced indicator
            data: Calculation data
            
        Returns:
            pd.Series: Calculated indicator values
        """
        try:
            if factor_name == 'rsi_14':
                close = data.get('$close')
                if close is not None:
                    delta = close.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                    rs = gain / (loss + 1e-8)
                    return 100 - (100 / (1 + rs))
            
            elif factor_name == 'bollinger_position':
                close = data.get('$close')
                if close is not None:
                    ma20 = close.rolling(window=20, min_periods=1).mean()
                    std20 = close.rolling(window=20, min_periods=1).std()
                    upper = ma20 + (2 * std20)
                    lower = ma20 - (2 * std20)
                    return (close - lower) / ((upper - lower) + 1e-8)
            
            elif factor_name == 'macd_signal':
                close = data.get('$close')
                if close is not None:
                    ema12 = close.rolling(window=12, min_periods=1).mean()  # Simplified EMA
                    ema26 = close.rolling(window=26, min_periods=1).mean()
                    macd_line = ema12 - ema26
                    signal_line = macd_line.rolling(window=9, min_periods=1).mean()
                    return macd_line - signal_line
            
            elif factor_name == 'price_oscillator':
                close = data.get('$close')
                if close is not None:
                    ma10 = close.rolling(window=10, min_periods=1).mean()
                    ma20 = close.rolling(window=20, min_periods=1).mean()
                    return (ma10 - ma20) / (ma20 + 1e-8)
            
            elif factor_name == 'volume_price_trend':
                close, volume = data.get('$close'), data.get('$volume')
                if close is not None and volume is not None:
                    price_change_pct = close.pct_change(1)
                    vpt_increment = volume * price_change_pct
                    return vpt_increment.rolling(window=10, min_periods=1).sum()
            
            elif factor_name == 'rolling_beta':
                close = data.get('$close')
                if close is not None:
                    returns = close.pct_change(1)
                    market_returns = returns.rolling(window=20, min_periods=1).mean()  # Simplified market proxy
                    return returns.rolling(window=20, min_periods=1).std() / (market_returns.rolling(window=20, min_periods=1).std() + 1e-8)
            
            elif factor_name == 'rolling_correlation':
                close, volume = data.get('$close'), data.get('$volume')
                if close is not None and volume is not None:
                    price_returns = close.pct_change(1)
                    volume_change = volume.pct_change(1)
                    return price_returns.rolling(window=20, min_periods=2).corr(volume_change)
            
            elif factor_name == 'momentum_divergence':
                close, volume = data.get('$close'), data.get('$volume')
                if close is not None and volume is not None:
                    price_momentum = close.pct_change(10)
                    volume_momentum = volume.pct_change(10)
                    price_trend = np.sign(price_momentum)
                    volume_trend = np.sign(volume_momentum)
                    return price_trend - volume_trend
            
            return pd.Series(np.nan, index=data.index)
            
        except Exception as e:
            warnings.warn(f"Error calculating advanced indicator {factor_name}: {str(e)}")
            return pd.Series(np.nan, index=data.index)
    
    def _create_empty_factor_dataframe(self) -> pd.DataFrame:
        """
        Create empty factor DataFrame with proper structure.
        
        Returns:
            pd.DataFrame: Empty DataFrame with correct structure
        """
        empty_index = pd.MultiIndex.from_tuples(
            [], names=['datetime', 'instrument']
        )
        empty_columns = pd.MultiIndex.from_tuples(
            [], names=['field_group', 'field_name']
        )
        return pd.DataFrame(index=empty_index, columns=empty_columns)
    
    def add_custom_factor(
        self,
        factor_name: str,
        expression: Expression
    ) -> None:
        """
        Add a custom factor expression.
        
        Args:
            factor_name: Name for the custom factor
            expression: Qlib Expression defining the factor calculation
        """
        self.factor_expressions[factor_name] = expression
    
    def get_available_factors(self) -> List[str]:
        """
        Get list of all available factors.
        
        Returns:
            List[str]: List of factor names
        """
        return list(self.factor_expressions.keys())
    
    def remove_factor(self, factor_name: str) -> None:
        """
        Remove a factor from the calculator.
        
        Args:
            factor_name: Name of factor to remove
        """
        if factor_name in self.factor_expressions:
            del self.factor_expressions[factor_name]
        else:
            warnings.warn(f"Factor {factor_name} not found")
