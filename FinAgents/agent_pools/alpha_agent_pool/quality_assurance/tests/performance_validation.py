"""
Performance Validation Testing Framework

This module provides comprehensive performance validation tests for alpha
strategies and reinforcement learning updates. It focuses on statistical
significance, economic significance, and implementation feasibility.

Key Testing Areas:
1. Strategy Performance Statistical Validation
2. Risk-Adjusted Return Analysis
3. Implementation Cost Assessment
4. Reinforcement Learning Policy Validation
5. Out-of-Sample Performance Testing

Author: FinAgent Quality Assurance Team
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Import statsmodels for Ljung-Box test
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available, some statistical tests will be skipped")

# Import configuration management
try:
    from ..config_manager import get_validation_thresholds, ValidationThresholds
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    warnings.warn("Configuration manager not available, using hardcoded defaults")

logger = logging.getLogger(__name__)

@dataclass
class PerformanceValidationResult:
    """Container for performance validation results"""
    test_name: str
    is_valid: bool
    confidence_level: float
    performance_metrics: Dict[str, float]
    statistical_tests: Dict[str, Any]
    economic_significance: Dict[str, float]
    risk_assessment: Dict[str, float]
    recommendations: List[str] = field(default_factory=list)

class PerformanceValidationTests:
    """
    Comprehensive performance validation testing framework.
    
    This class provides statistical and economic validation of alpha strategies,
    ensuring they meet industry standards for production deployment.
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        min_sharpe_ratio: float = 1.0,
        max_drawdown_threshold: float = 0.15,
        min_trade_count: int = 100,
        validation_config: Optional[ValidationThresholds] = None
    ):
        """
        Initialize performance validation framework.
        
        Args:
            confidence_level: Statistical confidence level for tests
            min_sharpe_ratio: Minimum acceptable Sharpe ratio
            max_drawdown_threshold: Maximum acceptable drawdown
            min_trade_count: Minimum number of trades for validation
            validation_config: Custom validation configuration
        """
        self.confidence_level = confidence_level
        self.min_sharpe_ratio = min_sharpe_ratio
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_trade_count = min_trade_count
        self.alpha_level = 1 - confidence_level
        
        # Load validation thresholds from configuration or use defaults
        if validation_config:
            self.thresholds = validation_config.__dict__
        elif CONFIG_AVAILABLE:
            self.thresholds = get_validation_thresholds()
        else:
            # Fallback to default thresholds if configuration not available
            self.thresholds = {
                'min_annual_return': 0.05,
                'max_sharpe_degradation': 0.5,
                'min_oos_sharpe_factor': 0.7,
                'max_drawdown_increase': 0.05,
                'min_sharpe_degradation_threshold': 0.3,
                'min_economic_value': 0.03,
                'min_cv_sharpe_factor': 0.8,
                'min_cv_consistency': 0.7,
                'trading_days_per_year': 252,
                'min_data_points_shapiro': 8,
                'min_data_points_ljung_box': 10,
                'ljung_box_max_lags_factor': 4,
                'var_confidence_level': 0.05,
                'degradation_components_count': 3
            }
    
    def validate_strategy_performance(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        transaction_costs: Optional[pd.Series] = None
    ) -> PerformanceValidationResult:
        """
        Comprehensive strategy performance validation.
        
        Args:
            strategy_returns: Strategy return time series
            benchmark_returns: Benchmark return time series
            transaction_costs: Transaction cost time series
            
        Returns:
            PerformanceValidationResult with validation outcome
        """
        test_name = "strategy_performance_validation"
        
        # Apply transaction costs if provided
        if transaction_costs is not None:
            net_returns = strategy_returns - transaction_costs
        else:
            net_returns = strategy_returns
        
        # Basic performance metrics
        performance_metrics = self._calculate_performance_metrics(net_returns)
        
        # Statistical tests
        statistical_tests = self._run_statistical_tests(
            net_returns, benchmark_returns
        )
        
        # Economic significance assessment
        economic_significance = self._assess_economic_significance(
            net_returns, transaction_costs
        )
        
        # Risk assessment
        risk_assessment = self._assess_risk_metrics(net_returns)
        
        # Overall validation decision
        is_valid, recommendations = self._make_validation_decision(
            performance_metrics,
            statistical_tests,
            economic_significance,
            risk_assessment
        )
        
        return PerformanceValidationResult(
            test_name=test_name,
            is_valid=is_valid,
            confidence_level=self.confidence_level,
            performance_metrics=performance_metrics,
            statistical_tests=statistical_tests,
            economic_significance=economic_significance,
            risk_assessment=risk_assessment,
            recommendations=recommendations
        )
    
    def validate_out_of_sample_performance(
        self,
        in_sample_returns: pd.Series,
        out_of_sample_returns: pd.Series,
        factor_data: Optional[pd.DataFrame] = None
    ) -> PerformanceValidationResult:
        """
        Validate out-of-sample performance consistency.
        
        Args:
            in_sample_returns: In-sample strategy returns
            out_of_sample_returns: Out-of-sample strategy returns
            factor_data: Factor values for stability analysis
            
        Returns:
            PerformanceValidationResult for out-of-sample validation
        """
        test_name = "out_of_sample_validation"
        
        # Calculate performance metrics for both periods
        is_metrics = self._calculate_performance_metrics(in_sample_returns)
        oos_metrics = self._calculate_performance_metrics(out_of_sample_returns)
        
        # Performance degradation analysis
        degradation_analysis = self._analyze_performance_degradation(
            is_metrics, oos_metrics
        )
        
        # Statistical tests for performance consistency
        consistency_tests = self._test_performance_consistency(
            in_sample_returns, out_of_sample_returns
        )
        
        # Factor stability analysis (if factor data provided)
        stability_analysis = {}
        if factor_data is not None:
            stability_analysis = self._analyze_factor_stability_oos(
                factor_data, in_sample_returns.index[-1]
            )
        
        # Economic significance in out-of-sample period
        oos_economic_significance = self._assess_economic_significance(
            out_of_sample_returns
        )
        
        # Validation decision
        performance_metrics = {
            "in_sample": is_metrics,
            "out_of_sample": oos_metrics,
            "degradation": degradation_analysis
        }
        
        is_valid = (
            degradation_analysis["sharpe_degradation"] < self.thresholds['max_sharpe_degradation'] and
            oos_metrics["sharpe_ratio"] > self.min_sharpe_ratio * self.thresholds['min_oos_sharpe_factor'] and
            consistency_tests["performance_consistency_pvalue"] > self.alpha_level
        )
        
        recommendations = self._generate_oos_recommendations(
            degradation_analysis, consistency_tests, oos_economic_significance
        )
        
        return PerformanceValidationResult(
            test_name=test_name,
            is_valid=is_valid,
            confidence_level=self.confidence_level,
            performance_metrics=performance_metrics,
            statistical_tests=consistency_tests,
            economic_significance=oos_economic_significance,
            risk_assessment=stability_analysis,
            recommendations=recommendations
        )
    
    def validate_reinforcement_learning_updates(
        self,
        pre_rl_returns: pd.Series,
        post_rl_returns: pd.Series,
        rl_training_metrics: Dict[str, Any]
    ) -> PerformanceValidationResult:
        """
        Validate reinforcement learning policy updates.
        
        Args:
            pre_rl_returns: Returns before RL update
            post_rl_returns: Returns after RL update
            rl_training_metrics: RL training performance metrics
            
        Returns:
            PerformanceValidationResult for RL validation
        """
        test_name = "reinforcement_learning_validation"
        
        # Performance improvement analysis
        improvement_analysis = self._analyze_rl_improvement(
            pre_rl_returns, post_rl_returns
        )
        
        # Training convergence validation
        convergence_validation = self._validate_rl_convergence(
            rl_training_metrics
        )
        
        # Overfitting detection
        overfitting_tests = self._detect_rl_overfitting(
            rl_training_metrics, post_rl_returns
        )
        
        # Policy stability assessment
        stability_metrics = self._assess_rl_policy_stability(
            post_rl_returns, rl_training_metrics
        )
        
        # Economic value of RL improvement
        rl_economic_value = self._calculate_rl_economic_value(
            pre_rl_returns, post_rl_returns
        )
        
        # Validation decision
        is_valid = (
            improvement_analysis["performance_improvement"] > 0 and
            improvement_analysis["improvement_significance"] < self.alpha_level and
            convergence_validation["converged"] and
            not overfitting_tests["overfitting_detected"]
        )
        
        recommendations = self._generate_rl_recommendations(
            improvement_analysis,
            convergence_validation,
            overfitting_tests,
            stability_metrics
        )
        
        return PerformanceValidationResult(
            test_name=test_name,
            is_valid=is_valid,
            confidence_level=self.confidence_level,
            performance_metrics=improvement_analysis,
            statistical_tests=convergence_validation,
            economic_significance=rl_economic_value,
            risk_assessment=stability_metrics,
            recommendations=recommendations
        )
    
    def validate_cross_validation_performance(
        self,
        returns_data: pd.Series,
        factor_data: pd.DataFrame,
        cv_folds: int = 5
    ) -> PerformanceValidationResult:
        """
        Validate strategy using time series cross-validation.
        
        Args:
            returns_data: Strategy returns time series
            factor_data: Factor values for cross-validation
            cv_folds: Number of cross-validation folds
            
        Returns:
            PerformanceValidationResult for cross-validation
        """
        test_name = "cross_validation_performance"
        
        # Perform time series cross-validation
        cv_results = self._perform_time_series_cv(
            returns_data, factor_data, cv_folds
        )
        
        # Analyze cross-validation stability
        cv_stability = self._analyze_cv_stability(cv_results)
        
        # Calculate cross-validation performance metrics
        cv_performance = self._calculate_cv_performance_metrics(cv_results)
        
        # Statistical significance of cross-validation results
        cv_statistical_tests = self._test_cv_significance(cv_results)
        
        # Economic significance across folds
        cv_economic_significance = self._assess_cv_economic_significance(cv_results)
        
        # Validation decision
        is_valid = (
            cv_performance["mean_sharpe"] > self.min_sharpe_ratio * self.thresholds['min_cv_sharpe_factor'] and
            cv_stability["sharpe_consistency"] > self.thresholds['min_cv_consistency'] and
            cv_statistical_tests["consistent_performance_pvalue"] > self.alpha_level
        )
        
        recommendations = self._generate_cv_recommendations(
            cv_performance, cv_stability, cv_statistical_tests
        )
        
        return PerformanceValidationResult(
            test_name=test_name,
            is_valid=is_valid,
            confidence_level=self.confidence_level,
            performance_metrics=cv_performance,
            statistical_tests=cv_statistical_tests,
            economic_significance=cv_economic_significance,
            risk_assessment=cv_stability,
            recommendations=recommendations
        )
    
    # Helper methods for performance calculations
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(returns) < 2:
            return {metric: 0.0 for metric in [
                "total_return", "annualized_return", "volatility",
                "sharpe_ratio", "max_drawdown", "calmar_ratio"
            ]}
        
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns).prod() ** (self.thresholds['trading_days_per_year'] / len(returns)) - 1
        volatility = returns.std() * np.sqrt(self.thresholds['trading_days_per_year'])
        
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio
        }
    
    def _run_statistical_tests(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Run statistical significance tests."""
        tests = {}
        
        # Test for non-zero mean return
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        tests["mean_return_ttest"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < self.alpha_level
        }
        
        # Normality test
        if len(returns) >= self.thresholds['min_data_points_shapiro']:  # Minimum for Shapiro-Wilk
            shapiro_stat, shapiro_p = stats.shapiro(returns)
            tests["normality_test"] = {
                "statistic": shapiro_stat,
                "p_value": shapiro_p,
                "normal": shapiro_p > self.alpha_level
            }
        
        # Serial correlation test
        if STATSMODELS_AVAILABLE and len(returns) > self.thresholds['min_data_points_ljung_box']:
            max_lags = min(10, len(returns) // self.thresholds['ljung_box_max_lags_factor'])
            try:
                ljung_box_result = acorr_ljungbox(
                    returns, lags=max_lags, return_df=True
                )
                ljung_box_stat = ljung_box_result['lb_stat'].iloc[-1]
                ljung_box_p = ljung_box_result['lb_pvalue'].iloc[-1]
                tests["serial_correlation"] = {
                    "statistic": ljung_box_stat,
                    "p_value": ljung_box_p,
                    "no_autocorr": ljung_box_p > self.alpha_level
                }
            except Exception as e:
                logger.debug(f"Ljung-Box test failed: {e}")
                tests["serial_correlation"] = {
                    "statistic": np.nan,
                    "p_value": 1.0,
                    "no_autocorr": True
                }
        else:
            tests["serial_correlation"] = {
                "statistic": np.nan,
                "p_value": 1.0,
                "no_autocorr": True
            }
        
        # Benchmark comparison (if provided)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            excess_returns = returns - benchmark_returns
            t_stat_excess, p_value_excess = stats.ttest_1samp(excess_returns, 0)
            tests["excess_return_test"] = {
                "t_statistic": t_stat_excess,
                "p_value": p_value_excess,
                "significant_outperformance": (
                    p_value_excess < self.alpha_level and t_stat_excess > 0
                )
            }
        
        return tests
    
    def _assess_economic_significance(
        self,
        returns: pd.Series,
        transaction_costs: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Assess economic significance of strategy performance."""
        if len(returns) < 2:
            return {"economic_value": 0.0, "cost_adjusted_value": 0.0}
        
        # Calculate economic value (dollar value of performance)
        annualized_return = (1 + returns).prod() ** (self.thresholds['trading_days_per_year'] / len(returns)) - 1
        economic_value = annualized_return  # As fraction of capital
        
        # Cost-adjusted economic value
        if transaction_costs is not None:
            net_returns = returns - transaction_costs
            net_annualized = (1 + net_returns).prod() ** (self.thresholds['trading_days_per_year'] / len(net_returns)) - 1
            cost_adjusted_value = net_annualized
            total_costs = transaction_costs.sum() * (self.thresholds['trading_days_per_year'] / len(transaction_costs))
        else:
            cost_adjusted_value = economic_value
            total_costs = 0.0
        
        return {
            "economic_value": economic_value,
            "cost_adjusted_value": cost_adjusted_value,
            "annual_transaction_costs": total_costs,
            "cost_drag": economic_value - cost_adjusted_value
        }
    
    def _assess_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Assess comprehensive risk metrics."""
        if len(returns) < 2:
            return {metric: 0.0 for metric in [
                "var_95", "expected_shortfall", "downside_deviation", "skewness", "kurtosis"
            ]}
        
        # Value at Risk (configurable confidence level)
        var_95 = returns.quantile(self.thresholds['var_confidence_level'])
        
        # Expected Shortfall
        tail_returns = returns[returns <= var_95]
        expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_95
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Higher moments
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return {
            "var_95": var_95,
            "expected_shortfall": expected_shortfall,
            "downside_deviation": downside_deviation,
            "skewness": skewness,
            "kurtosis": kurtosis
        }
    
    def _make_validation_decision(
        self,
        performance_metrics: Dict[str, float],
        statistical_tests: Dict[str, Any],
        economic_significance: Dict[str, float],
        risk_assessment: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """Make overall validation decision with recommendations."""
        
        validation_criteria = []
        recommendations = []
        
        # Performance criteria
        if performance_metrics["sharpe_ratio"] >= self.min_sharpe_ratio:
            validation_criteria.append(True)
        else:
            validation_criteria.append(False)
            recommendations.append(
                f"Sharpe ratio {performance_metrics['sharpe_ratio']:.2f} "
                f"below minimum threshold {self.min_sharpe_ratio}"
            )
        
        # Risk criteria
        if abs(performance_metrics["max_drawdown"]) <= self.max_drawdown_threshold:
            validation_criteria.append(True)
        else:
            validation_criteria.append(False)
            recommendations.append(
                f"Maximum drawdown {abs(performance_metrics['max_drawdown']):.2%} "
                f"exceeds threshold {self.max_drawdown_threshold:.2%}"
            )
        
        # Statistical significance
        if statistical_tests.get("mean_return_ttest", {}).get("significant", False):
            validation_criteria.append(True)
        else:
            validation_criteria.append(False)
            recommendations.append("Strategy returns not statistically significant")
        
        # Economic significance
        if economic_significance["cost_adjusted_value"] > self.thresholds['min_annual_return']:
            validation_criteria.append(True)
        else:
            validation_criteria.append(False)
            recommendations.append("Strategy lacks economic significance after costs")
        
        # Overall validation (require all criteria to pass)
        is_valid = all(validation_criteria)
        
        if is_valid:
            recommendations.append("Strategy passes all validation criteria")
        
        return is_valid, recommendations
    
    # Additional helper methods for specific validation types
    
    def _analyze_performance_degradation(
        self,
        is_metrics: Dict[str, float],
        oos_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Analyze performance degradation from in-sample to out-of-sample."""
        
        sharpe_degradation = (
            (is_metrics["sharpe_ratio"] - oos_metrics["sharpe_ratio"]) /
            is_metrics["sharpe_ratio"] if is_metrics["sharpe_ratio"] != 0 else 0
        )
        
        return_degradation = (
            (is_metrics["annualized_return"] - oos_metrics["annualized_return"]) /
            is_metrics["annualized_return"] if is_metrics["annualized_return"] != 0 else 0
        )
        
        drawdown_increase = oos_metrics["max_drawdown"] - is_metrics["max_drawdown"]
        
        return {
            "sharpe_degradation": sharpe_degradation,
            "return_degradation": return_degradation,
            "drawdown_increase": drawdown_increase,
            "overall_degradation_score": (
                sharpe_degradation + return_degradation + abs(drawdown_increase)
            ) / self.thresholds['degradation_components_count']
        }
    
    def _test_performance_consistency(
        self,
        in_sample_returns: pd.Series,
        out_of_sample_returns: pd.Series
    ) -> Dict[str, Any]:
        """Test statistical consistency between in-sample and out-of-sample performance."""
        
        # Two-sample t-test for mean returns
        t_stat, p_value = stats.ttest_ind(in_sample_returns, out_of_sample_returns)
        
        # F-test for variance equality
        f_stat = in_sample_returns.var() / out_of_sample_returns.var()
        f_p_value = 2 * min(
            stats.f.cdf(f_stat, len(in_sample_returns)-1, len(out_of_sample_returns)-1),
            1 - stats.f.cdf(f_stat, len(in_sample_returns)-1, len(out_of_sample_returns)-1)
        )
        
        return {
            "mean_consistency_ttest": {
                "t_statistic": t_stat,
                "p_value": p_value,
                "means_consistent": p_value > self.alpha_level
            },
            "variance_consistency_ftest": {
                "f_statistic": f_stat,
                "p_value": f_p_value,
                "variances_consistent": f_p_value > self.alpha_level
            },
            "performance_consistency_pvalue": min(p_value, f_p_value)
        }
    
    def _generate_oos_recommendations(
        self,
        degradation_analysis: Dict[str, float],
        consistency_tests: Dict[str, Any],
        economic_significance: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations for out-of-sample validation."""
        
        recommendations = []
        
        if degradation_analysis["sharpe_degradation"] > self.thresholds['min_sharpe_degradation_threshold']:
            recommendations.append(
                "Significant Sharpe ratio degradation detected - consider model recalibration"
            )
        
        if degradation_analysis["drawdown_increase"] > self.thresholds['max_drawdown_increase']:
            recommendations.append(
                "Increased drawdown in out-of-sample period - review risk management"
            )
        
        if not consistency_tests["mean_consistency_ttest"]["means_consistent"]:
            recommendations.append(
                "Mean returns inconsistent between periods - potential regime change"
            )
        
        if economic_significance["cost_adjusted_value"] < self.thresholds['min_economic_value']:
            recommendations.append(
                "Low out-of-sample economic value - reassess implementation costs"
            )
        
        if not recommendations:
            recommendations.append("Out-of-sample performance acceptable")
        
        return recommendations
    
    # Reinforcement Learning specific methods
    
    def _analyze_rl_improvement(
        self,
        pre_rl_returns: pd.Series,
        post_rl_returns: pd.Series
    ) -> Dict[str, Any]:
        """Analyze performance improvement from RL updates."""
        
        pre_metrics = self._calculate_performance_metrics(pre_rl_returns)
        post_metrics = self._calculate_performance_metrics(post_rl_returns)
        
        # Performance improvement metrics
        sharpe_improvement = post_metrics["sharpe_ratio"] - pre_metrics["sharpe_ratio"]
        return_improvement = post_metrics["annualized_return"] - pre_metrics["annualized_return"]
        
        # Statistical significance of improvement
        t_stat, p_value = stats.ttest_ind(post_rl_returns, pre_rl_returns)
        
        return {
            "performance_improvement": sharpe_improvement,
            "return_improvement": return_improvement,
            "improvement_significance": p_value,
            "improvement_t_stat": t_stat,
            "pre_rl_metrics": pre_metrics,
            "post_rl_metrics": post_metrics
        }
    
    def _validate_rl_convergence(
        self,
        rl_training_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate RL training convergence."""
        
        convergence_results = {
            "converged": False,
            "convergence_epoch": None,
            "final_loss": None,
            "convergence_stability": 0.0
        }
        
        if "training_losses" in rl_training_metrics:
            losses = rl_training_metrics["training_losses"]
            
            # Check for convergence (loss stabilization)
            if len(losses) > 10:
                recent_losses = losses[-10:]
                loss_std = np.std(recent_losses)
                loss_mean = np.mean(recent_losses)
                
                # Convergence if relative standard deviation is low
                if loss_mean > 0 and (loss_std / loss_mean) < 0.1:
                    convergence_results["converged"] = True
                    convergence_results["convergence_epoch"] = len(losses) - 10
                    convergence_results["convergence_stability"] = 1.0 - (loss_std / loss_mean)
                
                convergence_results["final_loss"] = losses[-1]
        
        return convergence_results
    
    def _detect_rl_overfitting(
        self,
        rl_training_metrics: Dict[str, Any],
        post_rl_returns: pd.Series
    ) -> Dict[str, Any]:
        """Detect overfitting in RL training."""
        
        overfitting_tests = {
            "overfitting_detected": False,
            "train_val_gap": 0.0,
            "performance_degradation": False,
            "complexity_penalty": 0.0
        }
        
        # Check training vs validation loss gap
        if "training_losses" in rl_training_metrics and "validation_losses" in rl_training_metrics:
            train_losses = rl_training_metrics["training_losses"]
            val_losses = rl_training_metrics["validation_losses"]
            
            if len(train_losses) > 0 and len(val_losses) > 0:
                final_train_loss = train_losses[-1]
                final_val_loss = val_losses[-1]
                
                if final_val_loss > final_train_loss * 1.5:  # 50% gap threshold
                    overfitting_tests["overfitting_detected"] = True
                
                overfitting_tests["train_val_gap"] = final_val_loss - final_train_loss
        
        # Check for performance degradation on new data
        if len(post_rl_returns) > 20:
            recent_performance = post_rl_returns.tail(10).mean()
            earlier_performance = post_rl_returns.head(10).mean()
            
            if recent_performance < earlier_performance * 0.8:  # 20% degradation
                overfitting_tests["performance_degradation"] = True
                overfitting_tests["overfitting_detected"] = True
        
        return overfitting_tests
    
    def _assess_rl_policy_stability(
        self,
        post_rl_returns: pd.Series,
        rl_training_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess RL policy stability."""
        
        stability_metrics = {
            "return_volatility": post_rl_returns.std(),
            "policy_consistency": 0.0,
            "action_diversity": 0.0
        }
        
        # Policy consistency based on return patterns
        if len(post_rl_returns) > 30:
            rolling_sharpe = []
            window_size = 10
            
            for i in range(window_size, len(post_rl_returns)):
                window_returns = post_rl_returns.iloc[i-window_size:i]
                if window_returns.std() > 0:
                    sharpe = window_returns.mean() / window_returns.std()
                    rolling_sharpe.append(sharpe)
            
            if rolling_sharpe:
                stability_metrics["policy_consistency"] = 1.0 - (np.std(rolling_sharpe) / (np.mean(rolling_sharpe) + 1e-8))
        
        # Action diversity from training metrics
        if "action_distribution" in rl_training_metrics:
            action_dist = rl_training_metrics["action_distribution"]
            if isinstance(action_dist, list) and len(action_dist) > 0:
                # Calculate entropy as measure of diversity
                probs = np.array(action_dist) / np.sum(action_dist)
                probs = probs[probs > 0]  # Remove zeros
                entropy = -np.sum(probs * np.log(probs))
                stability_metrics["action_diversity"] = entropy
        
        return stability_metrics
    
    def _calculate_rl_economic_value(
        self,
        pre_rl_returns: pd.Series,
        post_rl_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate economic value of RL improvement."""
        
        pre_economic = self._assess_economic_significance(pre_rl_returns)
        post_economic = self._assess_economic_significance(post_rl_returns)
        
        value_improvement = post_economic["economic_value"] - pre_economic["economic_value"]
        
        return {
            "value_improvement": value_improvement,
            "pre_rl_value": pre_economic["economic_value"],
            "post_rl_value": post_economic["economic_value"],
            "improvement_ratio": value_improvement / (abs(pre_economic["economic_value"]) + 1e-8)
        }
    
    def _generate_rl_recommendations(
        self,
        improvement_analysis: Dict[str, Any],
        convergence_validation: Dict[str, Any],
        overfitting_tests: Dict[str, Any],
        stability_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate RL-specific recommendations."""
        
        recommendations = []
        
        if improvement_analysis["performance_improvement"] <= 0:
            recommendations.append("RL update did not improve performance - review reward function and training data")
        
        if not convergence_validation["converged"]:
            recommendations.append("RL training did not converge - increase training epochs or adjust learning rate")
        
        if overfitting_tests["overfitting_detected"]:
            recommendations.append("RL overfitting detected - implement regularization or reduce model complexity")
        
        if stability_metrics["policy_consistency"] < 0.5:
            recommendations.append("RL policy shows instability - consider ensemble methods or experience replay")
        
        if improvement_analysis["improvement_significance"] > self.alpha_level:
            recommendations.append("RL improvement not statistically significant - collect more training data")
        
        if not recommendations:
            recommendations.append("RL update successful - ready for deployment")
        
        return recommendations
    
    # Cross-validation specific methods
    
    def _perform_time_series_cv(
        self,
        returns_data: pd.Series,
        factor_data: pd.DataFrame,
        cv_folds: int
    ) -> List[Dict[str, Any]]:
        """Perform time series cross-validation."""
        
        cv_results = []
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Align data
        common_index = returns_data.index.intersection(factor_data.index)
        returns_aligned = returns_data.loc[common_index]
        factors_aligned = factor_data.loc[common_index]
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(returns_aligned)):
            train_returns = returns_aligned.iloc[train_idx]
            test_returns = returns_aligned.iloc[test_idx]
            
            train_factors = factors_aligned.iloc[train_idx]
            test_factors = factors_aligned.iloc[test_idx]
            
            # Calculate metrics for this fold
            train_metrics = self._calculate_performance_metrics(train_returns)
            test_metrics = self._calculate_performance_metrics(test_returns)
            
            # Factor statistics
            factor_ic = {}
            for col in factors_aligned.columns:
                if len(test_factors[col].dropna()) > 10 and len(test_returns) > 10:
                    try:
                        ic, _ = stats.spearmanr(test_factors[col].dropna(), test_returns.loc[test_factors[col].dropna().index])
                        factor_ic[col] = ic if not np.isnan(ic) else 0.0
                    except:
                        factor_ic[col] = 0.0
            
            cv_results.append({
                "fold": fold_idx,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "factor_ic": factor_ic,
                "test_period": (test_returns.index[0], test_returns.index[-1])
            })
        
        return cv_results
    
    def _analyze_cv_stability(
        self,
        cv_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze cross-validation stability."""
        
        if not cv_results:
            return {"sharpe_consistency": 0.0, "return_consistency": 0.0}
        
        # Extract metrics across folds
        test_sharpes = [result["test_metrics"]["sharpe_ratio"] for result in cv_results]
        test_returns = [result["test_metrics"]["annualized_return"] for result in cv_results]
        
        # Calculate consistency (inverse of coefficient of variation)
        sharpe_consistency = 1.0 - (np.std(test_sharpes) / (np.mean(test_sharpes) + 1e-8))
        return_consistency = 1.0 - (np.std(test_returns) / (np.mean(test_returns) + 1e-8))
        
        return {
            "sharpe_consistency": max(0.0, sharpe_consistency),
            "return_consistency": max(0.0, return_consistency),
            "sharpe_std": np.std(test_sharpes),
            "return_std": np.std(test_returns)
        }
    
    def _calculate_cv_performance_metrics(
        self,
        cv_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate aggregate CV performance metrics."""
        
        if not cv_results:
            return {"mean_sharpe": 0.0, "mean_return": 0.0}
        
        test_sharpes = [result["test_metrics"]["sharpe_ratio"] for result in cv_results]
        test_returns = [result["test_metrics"]["annualized_return"] for result in cv_results]
        test_drawdowns = [result["test_metrics"]["max_drawdown"] for result in cv_results]
        
        return {
            "mean_sharpe": np.mean(test_sharpes),
            "mean_return": np.mean(test_returns),
            "mean_drawdown": np.mean(test_drawdowns),
            "min_sharpe": np.min(test_sharpes),
            "max_sharpe": np.max(test_sharpes),
            "sharpe_range": np.max(test_sharpes) - np.min(test_sharpes)
        }
    
    def _test_cv_significance(
        self,
        cv_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test statistical significance of CV results."""
        
        if not cv_results:
            return {"consistent_performance_pvalue": 1.0}
        
        test_sharpes = [result["test_metrics"]["sharpe_ratio"] for result in cv_results]
        
        # Test if mean Sharpe is significantly different from zero
        t_stat, p_value = stats.ttest_1samp(test_sharpes, 0)
        
        # Test for consistency across folds (low variance)
        consistency_test = stats.chi2.sf(np.var(test_sharpes) * len(test_sharpes), len(test_sharpes) - 1)
        
        return {
            "mean_sharpe_ttest": {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < self.alpha_level
            },
            "consistency_test_pvalue": consistency_test,
            "consistent_performance_pvalue": min(p_value, consistency_test)
        }
    
    def _assess_cv_economic_significance(
        self,
        cv_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Assess economic significance across CV folds."""
        
        if not cv_results:
            return {"mean_economic_value": 0.0}
        
        economic_values = [result["test_metrics"]["annualized_return"] for result in cv_results]
        
        return {
            "mean_economic_value": np.mean(economic_values),
            "min_economic_value": np.min(economic_values),
            "economic_value_consistency": 1.0 - (np.std(economic_values) / (np.mean(economic_values) + 1e-8))
        }
    
    def _generate_cv_recommendations(
        self,
        cv_performance: Dict[str, float],
        cv_stability: Dict[str, float],
        cv_statistical_tests: Dict[str, Any]
    ) -> List[str]:
        """Generate cross-validation specific recommendations."""
        
        recommendations = []
        
        if cv_performance["mean_sharpe"] < self.min_sharpe_ratio:
            recommendations.append(f"Mean CV Sharpe ratio {cv_performance['mean_sharpe']:.2f} below threshold")
        
        if cv_stability["sharpe_consistency"] < self.thresholds['min_cv_consistency']:
            recommendations.append("Low consistency across CV folds - strategy may be unstable")
        
        if cv_performance["sharpe_range"] > 1.0:
            recommendations.append("Large Sharpe ratio range across folds - consider regime-aware modeling")
        
        if not cv_statistical_tests["mean_sharpe_ttest"]["significant"]:
            recommendations.append("CV performance not statistically significant")
        
        if not recommendations:
            recommendations.append("Cross-validation results are acceptable")
        
        return recommendations
    
    def _analyze_factor_stability_oos(
        self,
        factor_data: pd.DataFrame,
        split_date: pd.Timestamp
    ) -> Dict[str, float]:
        """Analyze factor stability in out-of-sample period."""
        
        # Split factor data
        in_sample_factors = factor_data.loc[:split_date]
        out_sample_factors = factor_data.loc[split_date:]
        
        stability_metrics = {}
        
        for col in factor_data.columns:
            is_factor = in_sample_factors[col].dropna()
            oos_factor = out_sample_factors[col].dropna()
            
            if len(is_factor) > 10 and len(oos_factor) > 10:
                # Calculate correlation between IS and OOS factor distributions
                try:
                    corr, _ = stats.pearsonr(
                        is_factor.rolling(20).mean().dropna(),
                        oos_factor.rolling(20).mean().dropna()[:len(is_factor.rolling(20).mean().dropna())]
                    )
                    stability_metrics[f"{col}_stability"] = corr if not np.isnan(corr) else 0.0
                except:
                    stability_metrics[f"{col}_stability"] = 0.0
        
        return stability_metrics
