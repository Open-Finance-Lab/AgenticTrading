"""
Factor Backtesting Pipeline

This module implements factor-specific backtesting functionality.
It evaluates individual factors and factor combinations for their predictive power.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import qlib
from qlib.data import D
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config

try:
    from .interfaces import BacktestInterface, FactorInterface, EvaluationMetrics, AcceptanceCriteria
    from .utils import QlibConfig, DataProcessor, ResultProcessor
except ImportError:
    from interfaces import BacktestInterface, FactorInterface, EvaluationMetrics, AcceptanceCriteria
    from utils import QlibConfig, DataProcessor, ResultProcessor


class FactorBacktester(BacktestInterface):
    """
    Backtester specifically designed for evaluating factors
    """
    
    def __init__(self, 
                 config: QlibConfig,
                 acceptance_criteria: Optional[AcceptanceCriteria] = None):
        """
        Initialize Factor Backtester
        
        Args:
            config: Qlib configuration object
            acceptance_criteria: Criteria for accepting factors
        """
        self.config = config
        self.acceptance_criteria = acceptance_criteria
        self.data_processor = DataProcessor(config)
        self.result_processor = ResultProcessor()
        
        # Initialize Qlib
        qlib.init(provider_uri=config.provider_uri, region=REG_CN)
    
    def prepare_data(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """
        Prepare market data for factor backtesting
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional parameters (instruments, fields, etc.)
            
        Returns:
            Prepared market data with price and volume information
        """
        instruments = kwargs.get('instruments', self.config.instruments)
        fields = kwargs.get('fields', self.config.basic_fields)
        
        # Get basic market data
        data = D.features(
            instruments=instruments,
            fields=fields,
            start_time=start_date,
            end_time=end_date,
            freq=self.config.freq
        )
        
        # Process and clean data
        data = self.data_processor.clean_data(data)
        data = self.data_processor.add_returns(data)
        
        return data
    
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    factor: FactorInterface,
                    **kwargs) -> Dict[str, Any]:
        """
        Run backtesting for a specific factor
        
        Args:
            data: Market data prepared for backtesting
            factor: Factor to be backtested
            **kwargs: Additional backtesting parameters
            
        Returns:
            Dictionary containing backtest results
        """
        try:
            # Calculate factor values
            factor_values = factor.calculate_factor(data)
            
            # Validate factor
            if not factor.validate_factor(factor_values):
                return {
                    'success': False,
                    'error': 'Factor validation failed',
                    'factor_name': factor.factor_name
                }
            
            # Calculate factor returns and performance metrics
            results = self._calculate_factor_performance(data, factor_values, **kwargs)
            results['factor_name'] = factor.factor_name
            results['factor_description'] = factor.factor_description
            results['success'] = True
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'factor_name': factor.factor_name
            }
    
    def evaluate_performance(self, results: Dict[str, Any]) -> EvaluationMetrics:
        """
        Evaluate factor performance and create standardized metrics
        
        Args:
            results: Raw backtest results from run_backtest
            
        Returns:
            Standardized evaluation metrics
        """
        if not results.get('success', False):
            # Return empty metrics for failed backtests
            return EvaluationMetrics(
                annual_return=0.0,
                cumulative_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                downside_risk=0.0,
                calmar_ratio=0.0
            )
        
        # Extract performance metrics
        returns = results.get('returns', pd.Series())
        ic_series = results.get('ic_series', pd.Series())
        
        # Calculate standard metrics
        metrics = self.result_processor.calculate_metrics(returns)
        
        # Add factor-specific metrics
        if not ic_series.empty:
            metrics.ic_mean = ic_series.mean()
            metrics.ic_std = ic_series.std()
            metrics.ic_ir = metrics.ic_mean / metrics.ic_std if metrics.ic_std > 0 else 0.0
            metrics.rank_ic = results.get('rank_ic', 0.0)
        
        return metrics
    
    def _calculate_factor_performance(self, 
                                    data: pd.DataFrame,
                                    factor_values: pd.Series,
                                    **kwargs) -> Dict[str, Any]:
        """
        Calculate comprehensive factor performance metrics
        
        Args:
            data: Market data
            factor_values: Calculated factor values
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with factor performance metrics
        """
        # Align factor values with returns data
        returns_data = data['returns'] if 'returns' in data.columns else data.pct_change()
        
        # Calculate Information Coefficient (IC)
        ic_series = self._calculate_ic(factor_values, returns_data)
        
        # Calculate Rank IC
        rank_ic = self._calculate_rank_ic(factor_values, returns_data)
        
        # Generate factor-based strategy returns
        strategy_returns = self._generate_strategy_returns(factor_values, returns_data, **kwargs)
        
        return {
            'ic_series': ic_series,
            'rank_ic': rank_ic,
            'returns': strategy_returns,
            'factor_values': factor_values,
            'raw_data': data
        }
    
    def _calculate_ic(self, factor_values: pd.Series, returns: pd.Series) -> pd.Series:
        """Calculate Information Coefficient between factor and returns"""
        # Align indices
        common_idx = factor_values.index.intersection(returns.index)
        factor_aligned = factor_values.loc[common_idx]
        returns_aligned = returns.loc[common_idx]
        
        # Remove NaN values
        valid_mask = ~(factor_aligned.isna() | returns_aligned.isna())
        factor_clean = factor_aligned[valid_mask]
        returns_clean = returns_aligned[valid_mask]
        
        if len(factor_clean) < 10:  # Minimum data requirement
            return pd.Series(dtype=float)
        
        # Calculate rolling IC
        ic_series = factor_clean.rolling(window=20).corr(returns_clean)
        
        return ic_series.dropna()
    
    def _calculate_rank_ic(self, factor_values: pd.Series, returns: pd.Series) -> float:
        """Calculate Rank Information Coefficient"""
        # Align and clean data
        common_idx = factor_values.index.intersection(returns.index)
        factor_aligned = factor_values.loc[common_idx]
        returns_aligned = returns.loc[common_idx]
        
        valid_mask = ~(factor_aligned.isna() | returns_aligned.isna())
        factor_clean = factor_aligned[valid_mask]
        returns_clean = returns_aligned[valid_mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        # Calculate Spearman correlation (rank correlation)
        rank_ic = factor_clean.corr(returns_clean, method='spearman')
        
        return rank_ic if not pd.isna(rank_ic) else 0.0
    
    def _generate_strategy_returns(self, 
                                 factor_values: pd.Series, 
                                 returns: pd.Series,
                                 **kwargs) -> pd.Series:
        """
        Generate strategy returns based on factor values
        
        Args:
            factor_values: Factor values for position sizing
            returns: Asset returns
            **kwargs: Strategy parameters
            
        Returns:
            Strategy returns series
        """
        # Simple long-short strategy based on factor quintiles
        strategy_type = kwargs.get('strategy_type', 'long_short')
        rebalance_freq = kwargs.get('rebalance_freq', 20)  # Days
        
        # Align data
        common_idx = factor_values.index.intersection(returns.index)
        factor_aligned = factor_values.loc[common_idx]
        returns_aligned = returns.loc[common_idx]
        
        # Create position signals based on factor values
        if strategy_type == 'long_only':
            # Long positions for top quintile
            positions = (factor_aligned.rank(pct=True) > 0.8).astype(float)
        elif strategy_type == 'long_short':
            # Long top quintile, short bottom quintile
            positions = pd.Series(index=factor_aligned.index, dtype=float)
            positions[factor_aligned.rank(pct=True) > 0.8] = 1.0
            positions[factor_aligned.rank(pct=True) < 0.2] = -1.0
            positions = positions.fillna(0.0)
        else:
            # Equal weighted based on factor rank
            positions = factor_aligned.rank(pct=True) - 0.5
        
        # Calculate strategy returns
        strategy_returns = (positions.shift(1) * returns_aligned).fillna(0.0)
        
        return strategy_returns


class FactorEvaluator:
    """
    Evaluator for comprehensive factor analysis and acceptance testing
    """
    
    def __init__(self, acceptance_criteria: AcceptanceCriteria):
        """
        Initialize Factor Evaluator
        
        Args:
            acceptance_criteria: Criteria for factor acceptance
        """
        self.acceptance_criteria = acceptance_criteria
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate_factor(self, 
                       backtester: FactorBacktester,
                       factor: FactorInterface,
                       start_date: str,
                       end_date: str,
                       **kwargs) -> Dict[str, Any]:
        """
        Comprehensive factor evaluation
        
        Args:
            backtester: Factor backtester instance
            factor: Factor to evaluate
            start_date: Evaluation start date
            end_date: Evaluation end date
            **kwargs: Additional evaluation parameters
            
        Returns:
            Complete evaluation results including acceptance decision
        """
        # Prepare data
        data = backtester.prepare_data(start_date, end_date, **kwargs)
        
        # Run backtest
        backtest_results = backtester.run_backtest(data, factor, **kwargs)
        
        # Evaluate performance
        metrics = backtester.evaluate_performance(backtest_results)
        
        # Make acceptance decision
        is_accepted = self.acceptance_criteria.evaluate_factor(metrics)
        
        # Compile evaluation results
        evaluation_result = {
            'factor_name': factor.factor_name,
            'factor_description': factor.factor_description,
            'evaluation_period': f"{start_date} to {end_date}",
            'metrics': metrics,
            'is_accepted': is_accepted,
            'backtest_results': backtest_results,
            'acceptance_criteria': self.acceptance_criteria.get_criteria_description(),
            'timestamp': pd.Timestamp.now()
        }
        
        # Store in history
        self.evaluation_history.append(evaluation_result)
        
        return evaluation_result
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """
        Get summary of all factor evaluations
        
        Returns:
            DataFrame with evaluation summary
        """
        if not self.evaluation_history:
            return pd.DataFrame()
        
        summary_data = []
        for eval_result in self.evaluation_history:
            metrics = eval_result['metrics']
            summary_data.append({
                'factor_name': eval_result['factor_name'],
                'is_accepted': eval_result['is_accepted'],
                'annual_return': metrics.annual_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'ic_mean': metrics.ic_mean,
                'ic_ir': metrics.ic_ir,
                'evaluation_date': eval_result['timestamp']
            })
        
        return pd.DataFrame(summary_data)
    
    def get_accepted_factors(self) -> List[str]:
        """
        Get list of accepted factor names
        
        Returns:
            List of factor names that passed acceptance criteria
        """
        return [
            eval_result['factor_name'] 
            for eval_result in self.evaluation_history 
            if eval_result['is_accepted']
        ]
    
    def export_results(self, filepath: Union[str, Path]) -> None:
        """
        Export evaluation results to file
        
        Args:
            filepath: Path to save results
        """
        summary_df = self.get_evaluation_summary()
        summary_df.to_csv(filepath, index=False)
        print(f"Factor evaluation results exported to {filepath}")


# Example factor implementation
class ExampleMomentumFactor(FactorInterface):
    """Example implementation of a momentum factor"""
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
    
    def calculate_factor(self, data: pd.DataFrame) -> pd.Series:
        """Calculate momentum factor as price change over lookback period"""
        if 'close' in data.columns:
            prices = data['close']
        elif '$close' in data.columns:
            prices = data['$close']
        else:
            raise ValueError("No close price column found in data")
        
        momentum = prices.pct_change(periods=self.lookback_period)
        return momentum
    
    def validate_factor(self, factor_values: pd.Series) -> bool:
        """Validate momentum factor values"""
        # Check for sufficient non-NaN values
        valid_ratio = factor_values.dropna().shape[0] / factor_values.shape[0]
        
        # Check for reasonable value range (momentum should be bounded)
        value_range = factor_values.quantile(0.95) - factor_values.quantile(0.05)
        
        return valid_ratio > 0.7 and value_range > 0.01
    
    @property
    def factor_name(self) -> str:
        return f"Momentum_{self.lookback_period}D"
    
    @property
    def factor_description(self) -> str:
        return f"Price momentum calculated over {self.lookback_period} days"
