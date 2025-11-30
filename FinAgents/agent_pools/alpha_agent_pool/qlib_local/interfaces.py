"""
Interface Standards for Qlib Backtesting Pipeline

This module defines the standard interfaces that all backtesting components must implement.
Following these interfaces ensures consistency and interoperability across the pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class EvaluationMetrics:
    """Standard evaluation metrics for backtesting results"""
    
    # Return metrics
    annual_return: float
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Risk metrics
    volatility: float
    downside_risk: float
    calmar_ratio: float
    
    # Factor-specific metrics (optional)
    ic_mean: Optional[float] = None
    ic_std: Optional[float] = None
    ic_ir: Optional[float] = None
    rank_ic: Optional[float] = None
    
    # Model-specific metrics (optional)
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class BacktestInterface(ABC):
    """Base interface for all backtesting components"""
    
    @abstractmethod
    def prepare_data(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """
        Prepare data for backtesting
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            **kwargs: Additional parameters for data preparation
            
        Returns:
            Prepared data as pandas DataFrame
        """
        pass
    
    @abstractmethod
    def run_backtest(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Execute the backtesting process
        
        Args:
            data: Prepared data for backtesting
            **kwargs: Additional parameters for backtesting
            
        Returns:
            Dictionary containing backtest results
        """
        pass
    
    @abstractmethod
    def evaluate_performance(self, results: Dict[str, Any]) -> EvaluationMetrics:
        """
        Evaluate performance from backtest results
        
        Args:
            results: Raw backtest results
            
        Returns:
            Standardized evaluation metrics
        """
        pass


class FactorInterface(ABC):
    """Interface for factor-related operations"""
    
    @abstractmethod
    def calculate_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate factor values from market data
        
        Args:
            data: Market data with OHLCV information
            
        Returns:
            Factor values as pandas Series with datetime index
        """
        pass
    
    @abstractmethod
    def validate_factor(self, factor_values: pd.Series) -> bool:
        """
        Validate factor values for quality and completeness
        
        Args:
            factor_values: Calculated factor values
            
        Returns:
            True if factor passes validation, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def factor_name(self) -> str:
        """Return the name of the factor"""
        pass
    
    @property
    @abstractmethod
    def factor_description(self) -> str:
        """Return description of what the factor measures"""
        pass


class ModelInterface(ABC):
    """Interface for model-related operations"""
    
    @abstractmethod
    def train(self, features: pd.DataFrame, targets: pd.Series, **kwargs) -> None:
        """
        Train the model on provided data
        
        Args:
            features: Feature matrix for training
            targets: Target values for training
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate predictions using the trained model
        
        Args:
            features: Feature matrix for prediction
            
        Returns:
            Model predictions as pandas Series
        """
        pass
    
    @abstractmethod
    def validate_model(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """
        Validate model performance on validation data
        
        Args:
            features: Validation feature matrix
            targets: Validation target values
            
        Returns:
            Dictionary of validation metrics
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model"""
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the type of model (e.g., 'Tabular', 'TimeSeries')"""
        pass


class AcceptanceCriteria(ABC):
    """Interface for defining acceptance criteria for factors and models"""
    
    @abstractmethod
    def evaluate_factor(self, metrics: EvaluationMetrics) -> bool:
        """
        Determine if a factor meets acceptance criteria
        
        Args:
            metrics: Factor performance metrics
            
        Returns:
            True if factor is accepted, False otherwise
        """
        pass
    
    @abstractmethod
    def evaluate_model(self, metrics: EvaluationMetrics) -> bool:
        """
        Determine if a model meets acceptance criteria
        
        Args:
            metrics: Model performance metrics
            
        Returns:
            True if model is accepted, False otherwise
        """
        pass
    
    @abstractmethod
    def get_criteria_description(self) -> Dict[str, str]:
        """
        Return description of acceptance criteria
        
        Returns:
            Dictionary describing the acceptance criteria
        """
        pass


# Standard configurations
class StandardAcceptanceCriteria(AcceptanceCriteria):
    """Default acceptance criteria implementation"""
    
    def __init__(self, 
                 min_sharpe_ratio: float = 1.0,
                 max_drawdown_threshold: float = 0.2,
                 min_ic_mean: float = 0.02,
                 min_annual_return: float = 0.05):
        self.min_sharpe_ratio = min_sharpe_ratio
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_ic_mean = min_ic_mean
        self.min_annual_return = min_annual_return
    
    def evaluate_factor(self, metrics: EvaluationMetrics) -> bool:
        """Evaluate factor based on IC and risk metrics"""
        if metrics.ic_mean is None:
            return False
        
        return (
            metrics.ic_mean >= self.min_ic_mean and
            abs(metrics.max_drawdown) <= self.max_drawdown_threshold and
            metrics.sharpe_ratio >= self.min_sharpe_ratio * 0.5  # Relaxed for factors
        )
    
    def evaluate_model(self, metrics: EvaluationMetrics) -> bool:
        """Evaluate model based on return and risk metrics"""
        return (
            metrics.annual_return >= self.min_annual_return and
            metrics.sharpe_ratio >= self.min_sharpe_ratio and
            abs(metrics.max_drawdown) <= self.max_drawdown_threshold
        )
    
    def get_criteria_description(self) -> Dict[str, str]:
        return {
            "factor_criteria": f"IC >= {self.min_ic_mean}, Max DD <= {self.max_drawdown_threshold}, Sharpe >= {self.min_sharpe_ratio * 0.5}",
            "model_criteria": f"Annual Return >= {self.min_annual_return}, Sharpe >= {self.min_sharpe_ratio}, Max DD <= {self.max_drawdown_threshold}"
        }
