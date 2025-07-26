"""
Alpha Factor Metrics and Performance Analytics

This module provides comprehensive alpha factor quality metrics and
risk-adjusted performance measures specifically designed for quantitative
factor evaluation in agent-based trading systems.

Key Metrics Categories:
1. Information Content Metrics (IC, IR, Hit Rate)
2. Risk-Adjusted Performance Metrics (Sharpe, Sortino, Calmar)
3. Cross-Sectional Factor Metrics (Factor Decay, Turnover, Concentration)
4. Implementation Metrics (Transaction Costs, Market Impact)

Author: FinAgent Quality Assurance Team
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings

logger = logging.getLogger(__name__)

@dataclass
class AlphaMetricsResult:
    """Container for alpha factor metrics results"""
    metric_name: str
    value: float
    percentile_rank: Optional[float] = None
    significance_level: Optional[float] = None
    interpretation: str = ""

class AlphaFactorMetrics:
    """
    Comprehensive alpha factor quality metrics calculator.
    
    This class implements industry-standard and academic metrics for
    evaluating alpha factor quality, stability, and implementation feasibility.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize alpha factor metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        self.scaler = StandardScaler()
    
    def information_coefficient(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        method: str = 'spearman'
    ) -> pd.Series:
        """
        Calculate Information Coefficient between factor and forward returns.
        
        Args:
            factor_values: Factor values (dates x symbols)
            forward_returns: Forward returns (dates x symbols)
            method: Correlation method ('spearman' or 'pearson')
            
        Returns:
            Time series of IC values
        """
        ic_series = []
        
        for date in factor_values.index:
            if date in forward_returns.index:
                factor_cs = factor_values.loc[date].dropna()
                return_cs = forward_returns.loc[date].dropna()
                
                common_symbols = factor_cs.index.intersection(return_cs.index)
                
                if len(common_symbols) >= 10:
                    factor_vals = factor_cs[common_symbols]
                    return_vals = return_cs[common_symbols]
                    
                    try:
                        if method == 'spearman':
                            ic, _ = stats.spearmanr(factor_vals, return_vals)
                        else:
                            ic, _ = stats.pearsonr(factor_vals, return_vals)
                        
                        ic_series.append((date, ic if not np.isnan(ic) else 0.0))
                    except Exception as e:
                        logger.warning(f"IC calculation failed for {date}: {e}")
                        ic_series.append((date, 0.0))
        
        if ic_series:
            ic_df = pd.DataFrame(ic_series, columns=['date', 'ic'])
            return ic_df.set_index('date')['ic']
        else:
            return pd.Series(dtype=float)
    
    def information_ratio(self, ic_series: pd.Series) -> float:
        """
        Calculate Information Ratio from IC time series.
        
        Args:
            ic_series: Time series of IC values
            
        Returns:
            Information Ratio (IC_mean / IC_std)
        """
        if len(ic_series) < 2:
            return 0.0
        
        ic_clean = ic_series.dropna()
        if ic_clean.std() == 0:
            return 0.0
        
        return ic_clean.mean() / ic_clean.std()
    
    def ic_decay_analysis(
        self,
        factor_values: pd.DataFrame,
        returns_data: pd.DataFrame,
        max_horizon: int = 20
    ) -> Dict[int, float]:
        """
        Analyze Information Coefficient decay over time horizons.
        
        Args:
            factor_values: Factor values over time
            returns_data: Daily returns data
            max_horizon: Maximum horizon days to analyze
            
        Returns:
            Dictionary mapping horizon to IC values
        """
        decay_results = {}
        
        for horizon in range(1, max_horizon + 1):
            # Calculate forward returns for this horizon
            forward_returns = returns_data.shift(-horizon).rolling(
                window=horizon, min_periods=horizon
            ).sum()
            
            ic_series = self.information_coefficient(factor_values, forward_returns)
            decay_results[horizon] = ic_series.mean() if not ic_series.empty else 0.0
        
        return decay_results
    
    def factor_autocorrelation(
        self,
        factor_values: pd.DataFrame,
        max_lags: int = 10
    ) -> Dict[int, float]:
        """
        Calculate factor autocorrelation to assess persistence.
        
        Args:
            factor_values: Factor values over time
            max_lags: Maximum number of lags to analyze
            
        Returns:
            Dictionary mapping lag to autocorrelation values
        """
        autocorr_results = {}
        
        # Calculate cross-sectional averages
        factor_means = factor_values.mean(axis=1, skipna=True)
        
        for lag in range(1, max_lags + 1):
            if len(factor_means) > lag:
                corr_val = factor_means.autocorr(lag=lag)
                autocorr_results[lag] = corr_val if not np.isnan(corr_val) else 0.0
            else:
                autocorr_results[lag] = 0.0
        
        return autocorr_results
    
    def factor_concentration(
        self,
        factor_values: pd.DataFrame,
        method: str = 'herfindahl'
    ) -> pd.Series:
        """
        Calculate factor concentration over time.
        
        Args:
            factor_values: Factor values (dates x symbols)
            method: Concentration measure ('herfindahl', 'gini', 'max_weight')
            
        Returns:
            Time series of concentration values
        """
        concentration_series = []
        
        for date in factor_values.index:
            factor_cs = factor_values.loc[date].dropna()
            
            if len(factor_cs) > 0:
                # Use absolute values for concentration
                abs_factors = factor_cs.abs()
                weights = abs_factors / abs_factors.sum()
                
                if method == 'herfindahl':
                    concentration = (weights ** 2).sum()
                elif method == 'gini':
                    # Gini coefficient calculation
                    sorted_weights = weights.sort_values()
                    n = len(sorted_weights)
                    cumsum = sorted_weights.cumsum()
                    concentration = (2 * (np.arange(1, n + 1) * sorted_weights).sum()) / (n * sorted_weights.sum()) - (n + 1) / n
                elif method == 'max_weight':
                    concentration = weights.max()
                else:
                    concentration = 0.0
                
                concentration_series.append((date, concentration))
        
        if concentration_series:
            conc_df = pd.DataFrame(concentration_series, columns=['date', 'concentration'])
            return conc_df.set_index('date')['concentration']
        else:
            return pd.Series(dtype=float)
    
    def factor_turnover(
        self,
        factor_values: pd.DataFrame,
        method: str = 'gross'
    ) -> pd.Series:
        """
        Calculate factor turnover over time.
        
        Args:
            factor_values: Factor values (dates x symbols)
            method: Turnover calculation method ('gross', 'net')
            
        Returns:
            Time series of turnover values
        """
        turnover_series = []
        
        for i in range(1, len(factor_values)):
            prev_date = factor_values.index[i-1]
            curr_date = factor_values.index[i]
            
            prev_factors = factor_values.loc[prev_date].dropna()
            curr_factors = factor_values.loc[curr_date].dropna()
            
            common_symbols = prev_factors.index.intersection(curr_factors.index)
            
            if len(common_symbols) > 0:
                prev_weights = prev_factors[common_symbols]
                curr_weights = curr_factors[common_symbols]
                
                # Normalize to unit portfolio
                prev_weights = prev_weights / prev_weights.abs().sum()
                curr_weights = curr_weights / curr_weights.abs().sum()
                
                if method == 'gross':
                    turnover = (prev_weights - curr_weights).abs().sum()
                else:  # net
                    turnover = abs((prev_weights - curr_weights).sum())
                
                turnover_series.append((curr_date, turnover))
        
        if turnover_series:
            turnover_df = pd.DataFrame(turnover_series, columns=['date', 'turnover'])
            return turnover_df.set_index('date')['turnover']
        else:
            return pd.Series(dtype=float)

class RiskAdjustedMetrics:
    """
    Risk-adjusted performance metrics for alpha strategies.
    
    Implements comprehensive risk-adjusted performance measures
    commonly used in quantitative finance and academic research.
    """
    
    def __init__(self, risk_free_rate: float = 0.02, trading_days: int = 252):
        """
        Initialize risk-adjusted metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate
            trading_days: Number of trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
    
    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / self.trading_days
        return excess_returns.mean() / returns.std() * np.sqrt(self.trading_days)
    
    def sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio using downside deviation."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / self.trading_days
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / downside_returns.std() * np.sqrt(self.trading_days)
    
    def calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(returns) < 2:
            return 0.0
        
        annual_return = returns.mean() * self.trading_days
        max_dd = self.maximum_drawdown(returns)
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / abs(max_dd)
    
    def maximum_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) < 2:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def value_at_risk(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk at specified confidence level."""
        if len(returns) < 2:
            return 0.0
        
        return returns.quantile(confidence_level)
    
    def expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(returns) < 2:
            return 0.0
        
        var_threshold = self.value_at_risk(returns, confidence_level)
        tail_returns = returns[returns <= var_threshold]
        
        return tail_returns.mean() if len(tail_returns) > 0 else 0.0
    
    def tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error against benchmark."""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
        
        excess_returns = returns - benchmark_returns
        return excess_returns.std() * np.sqrt(self.trading_days)
    
    def information_ratio_vs_benchmark(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate Information Ratio vs benchmark."""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
        
        excess_returns = returns - benchmark_returns
        te = self.tracking_error(returns, benchmark_returns)
        
        if te == 0:
            return 0.0
        
        return excess_returns.mean() * self.trading_days / te

class CrossSectionalMetrics:
    """
    Cross-sectional analysis metrics for alpha factors.
    
    Provides metrics specifically designed for cross-sectional
    alpha factor analysis and portfolio construction validation.
    """
    
    def __init__(self, num_quantiles: int = 5):
        """
        Initialize cross-sectional metrics calculator.
        
        Args:
            num_quantiles: Number of quantiles for cross-sectional analysis
        """
        self.num_quantiles = num_quantiles
    
    def quantile_analysis(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Perform quantile-based cross-sectional analysis.
        
        Args:
            factor_values: Factor values (dates x symbols)
            forward_returns: Forward returns (dates x symbols)
            
        Returns:
            Dictionary with quantile analysis results
        """
        quantile_returns = {i: [] for i in range(self.num_quantiles)}
        quantile_counts = {i: [] for i in range(self.num_quantiles)}
        
        for date in factor_values.index:
            if date in forward_returns.index:
                factor_cs = factor_values.loc[date].dropna()
                return_cs = forward_returns.loc[date].dropna()
                
                common_symbols = factor_cs.index.intersection(return_cs.index)
                
                if len(common_symbols) >= self.num_quantiles * 2:
                    factor_vals = factor_cs[common_symbols]
                    return_vals = return_cs[common_symbols]
                    
                    # Create quantile rankings
                    try:
                        quantiles = pd.qcut(
                            factor_vals,
                            self.num_quantiles,
                            labels=False,
                            duplicates='drop'
                        )
                        
                        for q in range(self.num_quantiles):
                            mask = quantiles == q
                            if mask.sum() > 0:
                                q_returns = return_vals[mask]
                                quantile_returns[q].append(q_returns.mean())
                                quantile_counts[q].append(mask.sum())
                    except Exception as e:
                        logger.warning(f"Quantile analysis failed for {date}: {e}")
        
        # Calculate quantile statistics
        results = {}
        for q in range(self.num_quantiles):
            if quantile_returns[q]:
                results[f'quantile_{q}_mean_return'] = np.mean(quantile_returns[q])
                results[f'quantile_{q}_std_return'] = np.std(quantile_returns[q])
                results[f'quantile_{q}_sharpe'] = (
                    np.mean(quantile_returns[q]) / np.std(quantile_returns[q])
                    if np.std(quantile_returns[q]) > 0 else 0
                )
                results[f'quantile_{q}_avg_count'] = np.mean(quantile_counts[q])
        
        # Long-short portfolio analysis
        if quantile_returns[self.num_quantiles-1] and quantile_returns[0]:
            long_returns = quantile_returns[self.num_quantiles-1]
            short_returns = quantile_returns[0]
            
            min_length = min(len(long_returns), len(short_returns))
            if min_length > 0:
                ls_returns = [long_returns[i] - short_returns[i] for i in range(min_length)]
                results['long_short_mean_return'] = np.mean(ls_returns)
                results['long_short_std_return'] = np.std(ls_returns)
                results['long_short_sharpe'] = (
                    np.mean(ls_returns) / np.std(ls_returns)
                    if np.std(ls_returns) > 0 else 0
                )
        
        return results
    
    def cross_sectional_correlation_analysis(
        self,
        factor_values: pd.DataFrame,
        other_factors: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, float]:
        """
        Analyze cross-sectional correlations between factors.
        
        Args:
            factor_values: Primary factor values
            other_factors: Dictionary of other factors to compare against
            
        Returns:
            Dictionary with correlation analysis results
        """
        if other_factors is None:
            return {}
        
        correlation_results = {}
        
        for other_name, other_factor in other_factors.items():
            correlations = []
            
            for date in factor_values.index:
                if date in other_factor.index:
                    factor_cs = factor_values.loc[date].dropna()
                    other_cs = other_factor.loc[date].dropna()
                    
                    common_symbols = factor_cs.index.intersection(other_cs.index)
                    
                    if len(common_symbols) >= 10:
                        try:
                            corr, _ = stats.spearmanr(
                                factor_cs[common_symbols],
                                other_cs[common_symbols]
                            )
                            if not np.isnan(corr):
                                correlations.append(corr)
                        except Exception as e:
                            logger.warning(f"Correlation calculation failed: {e}")
            
            if correlations:
                correlation_results[f'correlation_with_{other_name}'] = np.mean(correlations)
                correlation_results[f'correlation_std_with_{other_name}'] = np.std(correlations)
        
        return correlation_results
    
    def factor_breadth_analysis(
        self,
        factor_values: pd.DataFrame,
        significance_threshold: float = 0.02
    ) -> Dict[str, float]:
        """
        Analyze factor breadth (fraction of universe with significant factor exposure).
        
        Args:
            factor_values: Factor values (dates x symbols)
            significance_threshold: Threshold for significant factor exposure
            
        Returns:
            Dictionary with breadth analysis results
        """
        breadth_series = []
        
        for date in factor_values.index:
            factor_cs = factor_values.loc[date].dropna()
            
            if len(factor_cs) > 0:
                significant_count = (factor_cs.abs() > significance_threshold).sum()
                breadth = significant_count / len(factor_cs)
                breadth_series.append(breadth)
        
        if breadth_series:
            return {
                'average_breadth': np.mean(breadth_series),
                'breadth_stability': np.std(breadth_series),
                'min_breadth': np.min(breadth_series),
                'max_breadth': np.max(breadth_series)
            }
        else:
            return {
                'average_breadth': 0.0,
                'breadth_stability': 0.0,
                'min_breadth': 0.0,
                'max_breadth': 0.0
            }
