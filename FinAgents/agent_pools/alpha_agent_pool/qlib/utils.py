"""
Utility Function@dataclass
class QlibConfig:

    # Configuration class for Qlib backtesting pipeline

    
    # Data configuration
    provider_uri: str = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data"
    instruments: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"])
    freq: str = "day"
    
    # Basic market data fields for US market data
    basic_fields: List[str] = field(default_factory=lambda: [
        "Open", "High", "Low", "Close", "Volume"
    ])tion for Qlib Backtesting Pipeline

This module provides utility classes and functions for data processing,
configuration management, and result processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    from .interfaces import EvaluationMetrics
except ImportError:
    from interfaces import EvaluationMetrics


@dataclass
class QlibConfig:
    """
    Configuration class for Qlib backtesting pipeline
    """
    
    # Data configuration
    provider_uri: str = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data"
    instruments: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"])
    freq: str = "day"
    
    # Basic market data fields for US market data
    basic_fields: List[str] = field(default_factory=lambda: [
        "Open", "High", "Low", "Close", "Volume"
    ])
    
    # Date ranges - updated for your data timeframe
    train_start_date: str = "2022-08-01"
    train_end_date: str = "2023-12-31"
    valid_start_date: str = "2024-01-01"
    valid_end_date: str = "2024-06-30"
    test_start_date: str = "2024-07-01"
    test_end_date: str = "2024-12-31"
    
    # Backtesting parameters for US market
    benchmark: str = "SPY"
    rebalance_frequency: str = "daily"
    
    # # Date ranges
    # train_start_date: str = "2008-01-01"
    # train_end_date: str = "2014-12-31"
    # valid_start_date: str = "2015-01-01"
    # valid_end_date: str = "2016-12-31"
    # test_start_date: str = "2017-01-01"
    # test_end_date: str = "2020-12-31"
    
    # # Backtesting parameters
    # benchmark: str = "SH000300"
    # rebalance_frequency: str = "daily"
    
    # Risk management
    max_position_size: float = 0.1
    transaction_cost: float = 0.003
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QlibConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> 'QlibConfig':
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'provider_uri': self.provider_uri,
            'instruments': self.instruments,
            'freq': self.freq,
            'basic_fields': self.basic_fields,
            'train_start_date': self.train_start_date,
            'train_end_date': self.train_end_date,
            'valid_start_date': self.valid_start_date,
            'valid_end_date': self.valid_end_date,
            'test_start_date': self.test_start_date,
            'test_end_date': self.test_end_date,
            'benchmark': self.benchmark,
            'rebalance_frequency': self.rebalance_frequency,
            'max_position_size': self.max_position_size,
            'transaction_cost': self.transaction_cost
        }
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save config to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class DataProcessor:
    """
    Data processing utilities for preparing market data and features
    """
    
    def __init__(self, config: QlibConfig):
        """
        Initialize data processor
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess market data
        
        Args:
            data: Raw market data
            
        Returns:
            Cleaned data
        """
        # Remove infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill missing values (up to 5 periods)
        data = data.fillna(method='ffill', limit=5)
        
        # Drop remaining missing values
        data = data.dropna()
        
        # Remove outliers (beyond 3 standard deviations)
        for col in data.select_dtypes(include=[np.number]).columns:
            mean = data[col].mean()
            std = data[col].std()
            data = data[
                (data[col] >= mean - 3 * std) & 
                (data[col] <= mean + 3 * std)
            ]
        
        return data
    
    def add_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add return calculations to market data
        
        Args:
            data: Market data with price information
            
        Returns:
            Data with added return columns
        """
        data = data.copy()
        
        # Identify price column
        price_col = None
        for col in ['$close', 'close', '$adj_close']:
            if col in data.columns:
                price_col = col
                break
        
        if price_col is None:
            raise ValueError("No price column found in data")
        
        # Calculate returns
        data['returns'] = data[price_col].pct_change()
        data['returns_1d'] = data['returns']
        data['returns_5d'] = data[price_col].pct_change(periods=5)
        data['returns_20d'] = data[price_col].pct_change(periods=20)
        
        return data
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicator features from market data
        
        Args:
            data: Market data with OHLCV
            
        Returns:
            Data with technical features added
        """
        features = data.copy()
        
        # Price-based features
        if '$close' in features.columns:
            close_col = '$close'
            high_col = '$high'
            low_col = '$low'
            volume_col = '$volume'
        else:
            close_col = 'close'
            high_col = 'high'
            low_col = 'low'
            volume_col = 'volume'
        
        # Moving averages
        for window in [5, 10, 20, 60]:
            features[f'ma_{window}'] = features[close_col].rolling(window).mean()
            features[f'ma_ratio_{window}'] = features[close_col] / features[f'ma_{window}']
        
        # Volatility
        features['volatility_20'] = features[close_col].pct_change().rolling(20).std()
        
        # Price momentum
        for period in [1, 5, 10, 20]:
            features[f'momentum_{period}'] = features[close_col].pct_change(periods=period)
        
        # Volume features
        if volume_col in features.columns:
            features['volume_ma_20'] = features[volume_col].rolling(20).mean()
            features['volume_ratio'] = features[volume_col] / features['volume_ma_20']
        
        # High-Low spread
        if high_col in features.columns and low_col in features.columns:
            features['hl_spread'] = (features[high_col] - features[low_col]) / features[close_col]
        
        # Relative Strength Index (simplified)
        delta = features[close_col].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        return features
    
    def create_targets(self, data: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """
        Create target variables for model training
        
        Args:
            data: Market data
            horizon: Forward-looking horizon in periods
            
        Returns:
            DataFrame with target variables
        """
        targets = pd.DataFrame(index=data.index)
        
        # Identify price column
        price_col = None
        for col in ['$close', 'close', '$adj_close']:
            if col in data.columns:
                price_col = col
                break
        
        if price_col is None:
            raise ValueError("No price column found in data")
        
        # Forward returns as targets
        targets['targets'] = data[price_col].pct_change(periods=horizon).shift(-horizon)
        
        # Binary classification targets (direction)
        targets['targets_binary'] = (targets['targets'] > 0).astype(int)
        
        # Quantile-based targets
        targets['targets_quantile'] = targets['targets'].rolling(252).rank(pct=True)
        
        return targets[['targets']]  # Return only the main target for simplicity


class ResultProcessor:
    """
    Result processing utilities for calculating performance metrics
    """
    
    def __init__(self):
        """Initialize result processor"""
        pass
    
    def calculate_metrics(self, returns: pd.Series) -> EvaluationMetrics:
        """
        Calculate comprehensive performance metrics from returns
        
        Args:
            returns: Time series of strategy returns
            
        Returns:
            EvaluationMetrics object with calculated metrics
        """
        if returns.empty or returns.isna().all():
            return EvaluationMetrics(
                annual_return=0.0,
                cumulative_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                downside_risk=0.0,
                calmar_ratio=0.0
            )
        
        # Clean returns
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return EvaluationMetrics(
                annual_return=0.0,
                cumulative_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                downside_risk=0.0,
                calmar_ratio=0.0
            )
        
        # Basic return metrics
        cumulative_return = (1 + returns_clean).prod() - 1
        annual_return = (1 + returns_clean.mean()) ** 252 - 1
        volatility = returns_clean.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0.0
        
        # Maximum drawdown
        cumulative = (1 + returns_clean).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Downside risk
        negative_returns = returns_clean[returns_clean < 0]
        downside_risk = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.0
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return EvaluationMetrics(
            annual_return=annual_return,
            cumulative_return=cumulative_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            downside_risk=downside_risk,
            calmar_ratio=calmar_ratio
        )
    
    def calculate_ic_metrics(self, factor_values: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate Information Coefficient metrics
        
        Args:
            factor_values: Factor values
            returns: Forward returns
            
        Returns:
            Dictionary with IC metrics
        """
        # Align data
        aligned_data = pd.concat([factor_values, returns], axis=1).dropna()
        
        if len(aligned_data) < 10:
            return {
                'ic_mean': 0.0,
                'ic_std': 0.0,
                'ic_ir': 0.0,
                'rank_ic': 0.0
            }
        
        factor_col = aligned_data.columns[0]
        return_col = aligned_data.columns[1]
        
        # Rolling IC calculation
        rolling_ic = aligned_data[factor_col].rolling(20).corr(aligned_data[return_col])
        rolling_ic = rolling_ic.dropna()
        
        # Rank IC (Spearman correlation)
        rank_ic = aligned_data[factor_col].corr(aligned_data[return_col], method='spearman')
        
        return {
            'ic_mean': rolling_ic.mean(),
            'ic_std': rolling_ic.std(),
            'ic_ir': rolling_ic.mean() / rolling_ic.std() if rolling_ic.std() > 0 else 0.0,
            'rank_ic': rank_ic if not pd.isna(rank_ic) else 0.0
        }
    
    def generate_performance_report(self, 
                                   metrics: EvaluationMetrics,
                                   name: str = "Strategy") -> str:
        """
        Generate a formatted performance report
        
        Args:
            metrics: Performance metrics
            name: Name of the strategy/factor/model
            
        Returns:
            Formatted report string
        """
        report = f"""
{name} Performance Report
{'=' * (len(name) + 20)}

Return Metrics:
- Annual Return: {metrics.annual_return:.2%}
- Cumulative Return: {metrics.cumulative_return:.2%}
- Volatility: {metrics.volatility:.2%}

Risk Metrics:
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}
- Maximum Drawdown: {metrics.max_drawdown:.2%}
- Downside Risk: {metrics.downside_risk:.2%}
- Calmar Ratio: {metrics.calmar_ratio:.2f}
"""
        
        # Add factor-specific metrics if available
        if metrics.ic_mean is not None:
            report += f"""
Factor Metrics:
- IC Mean: {metrics.ic_mean:.4f}
- IC Std: {metrics.ic_std:.4f}
- IC IR: {metrics.ic_ir:.2f}
- Rank IC: {metrics.rank_ic:.4f}
"""
        
        # Add model-specific metrics if available
        if metrics.accuracy is not None:
            report += f"""
Model Metrics:
- Accuracy: {metrics.accuracy:.2%}
"""
            if metrics.precision is not None:
                report += f"- Precision: {metrics.precision:.2%}\n"
            if metrics.recall is not None:
                report += f"- Recall: {metrics.recall:.2%}\n"
        
        return report


# Utility functions
def load_example_config() -> QlibConfig:
    """Load example configuration for testing with US market data"""
    return QlibConfig(
        provider_uri="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data",
        instruments=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        train_start_date="2022-08-01",
        train_end_date="2023-12-31",
        test_start_date="2024-01-01",
        test_end_date="2024-12-31"
    )


def validate_date_format(date_str: str) -> bool:
    """
    Validate date string format (YYYY-MM-DD)
    
    Args:
        date_str: Date string to validate
        
    Returns:
        True if valid format, False otherwise
    """
    try:
        pd.to_datetime(date_str, format='%Y-%m-%d')
        return True
    except ValueError:
        return False


def create_date_range(start_date: str, end_date: str, freq: str = 'D') -> pd.DatetimeIndex:
    """
    Create date range for backtesting
    
    Args:
        start_date: Start date string
        end_date: End date string
        freq: Frequency string
        
    Returns:
        DatetimeIndex for the specified range
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)
