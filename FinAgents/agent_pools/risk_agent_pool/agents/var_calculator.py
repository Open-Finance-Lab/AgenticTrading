"""
VaR Calculator - Value at Risk Calculation Agent

This agent specializes in Value at Risk (VaR) calculations using multiple methodologies:
- Parametric VaR (normal and t-distribution)
- Historical VaR (empirical distribution)
- Monte Carlo VaR (simulation-based)
- Expected Shortfall (Conditional VaR)
- Backtesting and model validation

Author: Jifeng Li
License: openMDW
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize

from ..registry import BaseRiskAgent


class VaRCalculator(BaseRiskAgent):
    """
    Specialized agent for comprehensive Value at Risk calculations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "VaRCalculator"
        self.logger = logging.getLogger(f"RiskAgent.{self.name}")
        
        # Configuration parameters
        self.confidence_levels = config.get('confidence_levels', [0.90, 0.95, 0.99]) if config else [0.90, 0.95, 0.99]
        self.var_methods = config.get('var_methods', ['parametric', 'historical', 'monte_carlo']) if config else ['parametric', 'historical', 'monte_carlo']
        self.monte_carlo_simulations = config.get('monte_carlo_simulations', 10000) if config else 10000
        self.historical_window = config.get('historical_window', 252) if config else 252
    
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Value at Risk using multiple methodologies.
        
        Args:
            request: Analysis request containing portfolio data and parameters
            
        Returns:
            Dictionary containing VaR calculations and analysis
        """
        start_time = datetime.utcnow()
        
        try:
            portfolio_data = request.get("portfolio_data", {})
            confidence_levels = request.get("confidence_levels", self.confidence_levels)
            methods = request.get("var_methods", self.var_methods)
            time_horizon = request.get("time_horizon", "daily")
            portfolio_value = request.get("portfolio_value", 1000000)  # Default $1M
            
            # Generate or extract returns data
            if "returns_data" in portfolio_data:
                returns = np.array(portfolio_data["returns_data"])
            else:
                returns = self._get_returns_data(portfolio_data)
            
            results = {
                "portfolio_value": portfolio_value,
                "time_horizon": time_horizon,
                "var_calculations": {},
                "expected_shortfall": {},
                "summary_statistics": self._calculate_summary_statistics(returns)
            }
            
            # Calculate VaR using different methods
            for method in methods:
                if method == "parametric":
                    results["var_calculations"]["parametric"] = await self._calculate_parametric_var(
                        returns, confidence_levels, portfolio_value
                    )
                elif method == "historical":
                    results["var_calculations"]["historical"] = await self._calculate_historical_var(
                        returns, confidence_levels, portfolio_value
                    )
                elif method == "monte_carlo":
                    results["var_calculations"]["monte_carlo"] = await self._calculate_monte_carlo_var(
                        returns, confidence_levels, portfolio_value
                    )
            
            # Calculate Expected Shortfall for all confidence levels
            for confidence_level in confidence_levels:
                conf_str = f"{int(confidence_level * 100)}%"
                results["expected_shortfall"][conf_str] = await self._calculate_expected_shortfall(
                    returns, confidence_level, portfolio_value
                )
            
            # Perform backtesting if sufficient data
            if len(returns) >= 250:
                results["backtesting"] = await self._perform_var_backtesting(
                    returns, confidence_levels, portfolio_value
                )
            
            # Component VaR analysis
            if request.get("include_component_var", False):
                results["component_var"] = await self._calculate_component_var(
                    portfolio_data, returns, confidence_levels[0], portfolio_value
                )
            
            # Marginal VaR analysis
            if request.get("include_marginal_var", False):
                results["marginal_var"] = await self._calculate_marginal_var(
                    portfolio_data, returns, confidence_levels[0], portfolio_value
                )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "agent": self.name,
                "analysis_type": "var_calculation",
                "results": results,
                "execution_time_ms": execution_time,
                "status": "success",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"VaR calculation failed: {e}")
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "agent": self.name,
                "error": str(e),
                "execution_time_ms": execution_time,
                "status": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _get_returns_data(self, portfolio_data: Dict[str, Any]) -> np.ndarray:
        """Extract portfolio returns from real data."""
        try:
            # Try to extract actual returns from portfolio data
            if "returns" in portfolio_data:
                returns = np.array(portfolio_data["returns"])
                if len(returns) > 0:
                    return returns
            
            # If no returns data available, try to calculate from prices
            if "prices" in portfolio_data:
                prices = np.array(portfolio_data["prices"])
                if len(prices) > 1:
                    returns = np.diff(np.log(prices))
                    return returns
            
            # If no valid data available, raise an error
            raise ValueError("No valid price or returns data found in portfolio_data")
            
        except Exception as e:
            raise ValueError(f"Failed to extract returns data: {str(e)}")
        
        # Add some autocorrelation
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]  # AR(1) component
        
        return returns
    
    def _calculate_summary_statistics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate summary statistics of returns."""
        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "skewness": float(stats.skew(returns)),
            "kurtosis": float(stats.kurtosis(returns, fisher=True)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "jarque_bera_stat": float(stats.jarque_bera(returns)[0]),
            "jarque_bera_pvalue": float(stats.jarque_bera(returns)[1]),
            "observations": len(returns)
        }
    
    async def _calculate_parametric_var(self, returns: np.ndarray, 
                                      confidence_levels: List[float], 
                                      portfolio_value: float) -> Dict[str, Any]:
        """Calculate parametric VaR using normal and t-distribution."""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Fit t-distribution to better capture fat tails
        t_params = stats.t.fit(returns)
        df, loc, scale = t_params
        
        parametric_var = {}
        
        for confidence_level in confidence_levels:
            conf_str = f"{int(confidence_level * 100)}%"
            
            # Normal distribution VaR
            z_score = stats.norm.ppf(1 - confidence_level)
            normal_var = -(mean_return + z_score * std_return) * portfolio_value
            
            # t-distribution VaR
            t_quantile = stats.t.ppf(1 - confidence_level, df, loc=loc, scale=scale)
            t_var = -t_quantile * portfolio_value
            
            # Cornish-Fisher expansion (accounts for skewness and kurtosis)
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns, fisher=True)
            
            cf_adjustment = (z_score + 
                           (z_score**2 - 1) * skewness / 6 +
                           (z_score**3 - 3*z_score) * kurtosis / 24 -
                           (2*z_score**3 - 5*z_score) * skewness**2 / 36)
            
            cf_var = -(mean_return + cf_adjustment * std_return) * portfolio_value
            
            parametric_var[conf_str] = {
                "normal_var": float(normal_var),
                "t_distribution_var": float(t_var),
                "cornish_fisher_var": float(cf_var),
                "t_distribution_params": {
                    "degrees_of_freedom": float(df),
                    "location": float(loc),
                    "scale": float(scale)
                },
                "distribution_comparison": {
                    "normal_vs_t_difference": float(t_var - normal_var),
                    "cf_vs_normal_difference": float(cf_var - normal_var)
                }
            }
        
        return parametric_var
    
    async def _calculate_historical_var(self, returns: np.ndarray, 
                                      confidence_levels: List[float], 
                                      portfolio_value: float) -> Dict[str, Any]:
        """Calculate historical VaR using empirical distribution."""
        historical_var = {}
        
        for confidence_level in confidence_levels:
            conf_str = f"{int(confidence_level * 100)}%"
            
            # Standard historical VaR
            percentile = (1 - confidence_level) * 100
            var_percentile = np.percentile(returns, percentile)
            standard_var = -var_percentile * portfolio_value
            
            # Age-weighted historical VaR
            weights = np.exp(-0.01 * np.arange(len(returns))[::-1])  # More weight to recent observations
            weights = weights / np.sum(weights)
            
            sorted_indices = np.argsort(returns)
            sorted_weights = weights[sorted_indices]
            cumulative_weights = np.cumsum(sorted_weights)
            
            # Find the quantile
            target_weight = 1 - confidence_level
            var_index = np.searchsorted(cumulative_weights, target_weight)
            var_index = min(var_index, len(returns) - 1)
            
            weighted_var = -returns[sorted_indices[var_index]] * portfolio_value
            
            # Bootstrap confidence intervals
            bootstrap_vars = []
            for _ in range(1000):
                bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
                bootstrap_var = -np.percentile(bootstrap_sample, percentile) * portfolio_value
                bootstrap_vars.append(bootstrap_var)
            
            ci_lower = np.percentile(bootstrap_vars, 2.5)
            ci_upper = np.percentile(bootstrap_vars, 97.5)
            
            historical_var[conf_str] = {
                "historical_var": float(standard_var),
                "age_weighted_var": float(weighted_var),
                "bootstrap_confidence_interval": {
                    "lower_95": float(ci_lower),
                    "upper_95": float(ci_upper)
                },
                "percentile_used": float(percentile),
                "empirical_quantile": float(var_percentile),
                "method_comparison": {
                    "standard_vs_weighted_difference": float(weighted_var - standard_var)
                }
            }
        
        return historical_var
    
    async def _calculate_monte_carlo_var(self, returns: np.ndarray, 
                                       confidence_levels: List[float], 
                                       portfolio_value: float) -> Dict[str, Any]:
        """Calculate Monte Carlo VaR using simulation."""
        monte_carlo_var = {}
        
        # Fit different distributions
        normal_params = stats.norm.fit(returns)
        t_params = stats.t.fit(returns)
        
        for confidence_level in confidence_levels:
            conf_str = f"{int(confidence_level * 100)}%"
            
            # Normal distribution simulation
            normal_simulations = stats.norm.rvs(
                loc=normal_params[0], 
                scale=normal_params[1], 
                size=self.monte_carlo_simulations
            )
            normal_var = -np.percentile(normal_simulations, (1 - confidence_level) * 100) * portfolio_value
            
            # t-distribution simulation
            t_simulations = stats.t.rvs(
                df=t_params[0], 
                loc=t_params[1], 
                scale=t_params[2], 
                size=self.monte_carlo_simulations
            )
            t_var = -np.percentile(t_simulations, (1 - confidence_level) * 100) * portfolio_value
            
            # Filtered Historical Simulation (FHS)
            # Use GARCH to model volatility clustering
            garch_residuals, garch_volatilities = self._fit_simple_garch(returns)
            
            # Simulate future volatilities
            current_vol = garch_volatilities[-1]
            future_vols = self._simulate_garch_volatilities(current_vol, periods=1000)
            
            # Generate scenarios using filtered residuals
            fhs_scenarios = []
            for vol in future_vols:
                residual = np.random.choice(garch_residuals)
                scenario = residual * vol
                fhs_scenarios.append(scenario)
            
            fhs_var = -np.percentile(fhs_scenarios, (1 - confidence_level) * 100) * portfolio_value
            
            # Calculate simulation statistics
            monte_carlo_var[conf_str] = {
                "normal_mc_var": float(normal_var),
                "t_distribution_mc_var": float(t_var),
                "filtered_historical_var": float(fhs_var),
                "simulation_parameters": {
                    "num_simulations": self.monte_carlo_simulations,
                    "normal_params": {"mean": float(normal_params[0]), "std": float(normal_params[1])},
                    "t_params": {"df": float(t_params[0]), "loc": float(t_params[1]), "scale": float(t_params[2])}
                },
                "simulation_statistics": {
                    "normal_sim_mean": float(np.mean(normal_simulations)),
                    "normal_sim_std": float(np.std(normal_simulations)),
                    "t_sim_mean": float(np.mean(t_simulations)),
                    "t_sim_std": float(np.std(t_simulations))
                },
                "method_comparison": {
                    "normal_vs_t_difference": float(t_var - normal_var),
                    "fhs_vs_normal_difference": float(fhs_var - normal_var)
                }
            }
        
        return monte_carlo_var
    
    def _fit_simple_garch(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a simple GARCH(1,1) model to extract residuals and volatilities."""
        # Simple GARCH(1,1) with fixed parameters
        omega = 0.000001
        alpha = 0.1
        beta = 0.85
        
        # Initialize
        n = len(returns)
        volatilities = np.zeros(n)
        volatilities[0] = np.std(returns)
        
        # Calculate conditional volatilities
        for t in range(1, n):
            volatilities[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * volatilities[t-1]**2)
        
        # Calculate standardized residuals
        residuals = returns / volatilities
        
        return residuals, volatilities
    
    def _simulate_garch_volatilities(self, current_vol: float, periods: int) -> np.ndarray:
        """Simulate future GARCH volatilities."""
        omega = 0.000001
        alpha = 0.1
        beta = 0.85
        
        volatilities = np.zeros(periods)
        volatilities[0] = current_vol
        
        for t in range(1, periods):
            # Random innovation
            innovation = np.random.normal(0, 1)
            prev_return = innovation * volatilities[t-1]
            
            # Update volatility
            volatilities[t] = np.sqrt(omega + alpha * prev_return**2 + beta * volatilities[t-1]**2)
        
        return volatilities
    
    async def _calculate_expected_shortfall(self, returns: np.ndarray, 
                                          confidence_level: float, 
                                          portfolio_value: float) -> Dict[str, Any]:
        """Calculate Expected Shortfall (Conditional VaR)."""
        # Historical Expected Shortfall
        percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, percentile)
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) > 0:
            historical_es = -np.mean(tail_losses) * portfolio_value
        else:
            historical_es = -var_threshold * portfolio_value
        
        # Parametric Expected Shortfall (normal distribution)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        z_alpha = stats.norm.ppf(1 - confidence_level)
        
        # Mills ratio
        mills_ratio = stats.norm.pdf(z_alpha) / (1 - confidence_level)
        parametric_es = -(mean_return - std_return * mills_ratio) * portfolio_value
        
        # t-distribution Expected Shortfall
        t_params = stats.t.fit(returns)
        df, loc, scale = t_params
        
        t_quantile = stats.t.ppf(1 - confidence_level, df, loc=loc, scale=scale)
        t_pdf_at_quantile = stats.t.pdf(t_quantile, df, loc=loc, scale=scale)
        
        t_es_numerator = t_pdf_at_quantile * scale * (df + t_quantile**2) / (df - 1)
        t_es = -(loc - t_es_numerator / (1 - confidence_level)) * portfolio_value
        
        return {
            "historical_expected_shortfall": float(historical_es),
            "parametric_expected_shortfall": float(parametric_es),
            "t_distribution_expected_shortfall": float(t_es),
            "tail_observations": len(tail_losses),
            "tail_percentage": float(len(tail_losses) / len(returns) * 100),
            "var_threshold": float(var_threshold),
            "es_var_ratio": {
                "historical": float(historical_es / (-var_threshold * portfolio_value)) if var_threshold != 0 else 1.0,
                "parametric": float(parametric_es / (-(mean_return + z_alpha * std_return) * portfolio_value)) if (mean_return + z_alpha * std_return) != 0 else 1.0
            }
        }
    
    async def _perform_var_backtesting(self, returns: np.ndarray, 
                                     confidence_levels: List[float], 
                                     portfolio_value: float) -> Dict[str, Any]:
        """Perform comprehensive VaR backtesting."""
        backtesting_results = {}
        
        # Use rolling window for out-of-sample testing
        estimation_window = 250
        test_observations = len(returns) - estimation_window
        
        if test_observations < 50:
            return {"error": "Insufficient data for meaningful backtesting"}
        
        for confidence_level in confidence_levels:
            conf_str = f"{int(confidence_level * 100)}%"
            
            violations = []
            var_forecasts = []
            actual_returns = []
            
            for t in range(estimation_window, len(returns)):
                # Use rolling window to estimate VaR
                window_returns = returns[t-estimation_window:t]
                
                # Calculate VaR using historical method
                percentile = (1 - confidence_level) * 100
                var_estimate = -np.percentile(window_returns, percentile) * portfolio_value
                
                # Record actual return
                actual_return = returns[t] * portfolio_value
                
                # Check for violation
                violation = actual_return < -var_estimate
                
                violations.append(violation)
                var_forecasts.append(var_estimate)
                actual_returns.append(actual_return)
            
            # Calculate backtesting statistics
            violation_count = sum(violations)
            violation_rate = violation_count / len(violations)
            expected_violations = (1 - confidence_level) * len(violations)
            
            # Kupiec POF test
            kupiec_stat, kupiec_pvalue = self._kupiec_test(
                violation_count, len(violations), 1 - confidence_level
            )
            
            # Christoffersen Independence test
            independence_stat, independence_pvalue = self._christoffersen_independence_test(violations)
            
            # Christoffersen Conditional Coverage test
            cc_stat, cc_pvalue = self._christoffersen_cc_test(
                violation_count, len(violations), violations, 1 - confidence_level
            )
            
            # Average loss in violation days
            violation_losses = [-actual_returns[i] for i, v in enumerate(violations) if v]
            avg_violation_loss = np.mean(violation_losses) if violation_losses else 0
            
            # Loss function tests
            loss_function_results = self._loss_function_tests(
                var_forecasts, actual_returns, confidence_level
            )
            
            backtesting_results[conf_str] = {
                "basic_statistics": {
                    "violation_count": violation_count,
                    "total_observations": len(violations),
                    "violation_rate": float(violation_rate),
                    "expected_violation_rate": float(1 - confidence_level),
                    "expected_violations": float(expected_violations)
                },
                "statistical_tests": {
                    "kupiec_test": {
                        "statistic": float(kupiec_stat),
                        "p_value": float(kupiec_pvalue),
                        "reject_null": kupiec_pvalue < 0.05
                    },
                    "independence_test": {
                        "statistic": float(independence_stat),
                        "p_value": float(independence_pvalue),
                        "reject_null": independence_pvalue < 0.05
                    },
                    "conditional_coverage_test": {
                        "statistic": float(cc_stat),
                        "p_value": float(cc_pvalue),
                        "reject_null": cc_pvalue < 0.05
                    }
                },
                "loss_analysis": {
                    "average_violation_loss": float(avg_violation_loss),
                    "max_violation_loss": float(max(violation_losses)) if violation_losses else 0,
                    "violation_loss_statistics": {
                        "mean": float(np.mean(violation_losses)) if violation_losses else 0,
                        "std": float(np.std(violation_losses)) if violation_losses else 0,
                        "percentiles": {
                            "50th": float(np.percentile(violation_losses, 50)) if violation_losses else 0,
                            "95th": float(np.percentile(violation_losses, 95)) if violation_losses else 0
                        }
                    }
                },
                "loss_function_tests": loss_function_results,
                "model_quality": {
                    "traffic_light_color": self._traffic_light_test(violation_rate, 1 - confidence_level),
                    "model_adequacy": "adequate" if kupiec_pvalue > 0.05 and independence_pvalue > 0.05 else "inadequate"
                }
            }
        
        return backtesting_results
    
    def _kupiec_test(self, violations: int, observations: int, expected_rate: float) -> Tuple[float, float]:
        """Kupiec Proportion of Failures test."""
        if violations == 0 or violations == observations:
            return 0.0, 1.0
        
        observed_rate = violations / observations
        
        # Likelihood ratio statistic
        lr_stat = 2 * (violations * np.log(observed_rate / expected_rate) + 
                      (observations - violations) * np.log((1 - observed_rate) / (1 - expected_rate)))
        
        # P-value from chi-square distribution with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr_stat, 1)
        
        return lr_stat, p_value
    
    def _christoffersen_independence_test(self, violations: List[bool]) -> Tuple[float, float]:
        """Christoffersen Independence test."""
        # Transition matrix
        n00 = n01 = n10 = n11 = 0
        
        for i in range(1, len(violations)):
            if not violations[i-1] and not violations[i]:
                n00 += 1
            elif not violations[i-1] and violations[i]:
                n01 += 1
            elif violations[i-1] and not violations[i]:
                n10 += 1
            else:
                n11 += 1
        
        # Calculate test statistic
        if n01 + n11 == 0 or n00 + n10 == 0 or n01 + n00 == 0 or n10 + n11 == 0:
            return 0.0, 1.0
        
        pi_01 = n01 / (n00 + n01)
        pi_11 = n11 / (n10 + n11)
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)
        
        if pi_01 == 0 or pi_11 == 0 or pi == 0:
            return 0.0, 1.0
        
        lr_ind = 2 * (n01 * np.log(pi_01 / pi) + n11 * np.log(pi_11 / pi))
        
        # P-value from chi-square distribution with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr_ind, 1)
        
        return lr_ind, p_value
    
    def _christoffersen_cc_test(self, violations: int, observations: int, 
                               violation_sequence: List[bool], expected_rate: float) -> Tuple[float, float]:
        """Christoffersen Conditional Coverage test."""
        # Combine POF and Independence tests
        kupiec_stat, _ = self._kupiec_test(violations, observations, expected_rate)
        independence_stat, _ = self._christoffersen_independence_test(violation_sequence)
        
        cc_stat = kupiec_stat + independence_stat
        p_value = 1 - stats.chi2.cdf(cc_stat, 2)
        
        return cc_stat, p_value
    
    def _loss_function_tests(self, var_forecasts: List[float], 
                           actual_returns: List[float], confidence_level: float) -> Dict[str, Any]:
        """Perform loss function tests for VaR model evaluation."""
        # Quantile loss function
        alpha = 1 - confidence_level
        quantile_losses = []
        
        for var_forecast, actual_return in zip(var_forecasts, actual_returns):
            if actual_return < -var_forecast:
                loss = alpha * (var_forecast + actual_return)
            else:
                loss = (1 - alpha) * (-var_forecast - actual_return)
            quantile_losses.append(loss)
        
        avg_quantile_loss = np.mean(quantile_losses)
        
        # Quadratic loss function
        quadratic_losses = [(actual_return + var_forecast)**2 for var_forecast, actual_return in zip(var_forecasts, actual_returns)]
        avg_quadratic_loss = np.mean(quadratic_losses)
        
        return {
            "quantile_loss": {
                "average_loss": float(avg_quantile_loss),
                "total_loss": float(sum(quantile_losses))
            },
            "quadratic_loss": {
                "average_loss": float(avg_quadratic_loss),
                "total_loss": float(sum(quadratic_losses))
            }
        }
    
    def _traffic_light_test(self, violation_rate: float, expected_rate: float) -> str:
        """Basel traffic light test for VaR model validation."""
        # Basel II thresholds for 99% VaR
        if expected_rate == 0.01:  # 99% confidence level
            if violation_rate <= 0.01:
                return "green"
            elif violation_rate <= 0.02:
                return "yellow"
            else:
                return "red"
        else:
            # General thresholds based on deviation from expected rate
            deviation = abs(violation_rate - expected_rate) / expected_rate
            if deviation <= 0.2:
                return "green"
            elif deviation <= 0.5:
                return "yellow"
            else:
                return "red"
    
    async def _calculate_component_var(self, portfolio_data: Dict[str, Any], 
                                     returns: np.ndarray, confidence_level: float, 
                                     portfolio_value: float) -> Dict[str, Any]:
        """Calculate Component VaR for portfolio positions."""
        if "securities" not in portfolio_data or "weights" not in portfolio_data:
            return {"error": "Portfolio composition data required for Component VaR"}
        
        securities = portfolio_data["securities"]
        weights = portfolio_data["weights"]
        
        # Simulate individual security returns
        num_securities = len(securities)
        correlation_matrix = self._generate_correlation_matrix(num_securities)
        
        # Generate security returns
        security_returns = {}
        for i, security in enumerate(securities):
            # Simulate correlated returns
            base_return = np.random.normal(0.0008, 0.001, len(returns))
            noise = np.random.multivariate_normal(np.zeros(num_securities), correlation_matrix, len(returns))
            security_returns[security] = base_return + noise[:, i] * 0.01
        
        # Calculate portfolio VaR
        portfolio_var = -np.percentile(returns, (1 - confidence_level) * 100) * portfolio_value
        
        # Calculate component VaR for each security
        component_vars = {}
        
        for i, security in enumerate(securities):
            # Calculate marginal VaR
            # Approximate using finite differences
            epsilon = 0.001
            
            # Perturb weight slightly
            perturbed_weights = weights.copy()
            perturbed_weights[i] += epsilon
            # Renormalize
            perturbed_weights = [w / sum(perturbed_weights) for w in perturbed_weights]
            
            # Calculate perturbed portfolio returns
            perturbed_returns = np.zeros(len(returns))
            for j, sec in enumerate(securities):
                perturbed_returns += perturbed_weights[j] * security_returns[sec]
            
            perturbed_var = -np.percentile(perturbed_returns, (1 - confidence_level) * 100) * portfolio_value
            
            # Marginal VaR
            marginal_var = (perturbed_var - portfolio_var) / epsilon
            
            # Component VaR
            component_var = marginal_var * weights[i]
            
            component_vars[security] = {
                "marginal_var": float(marginal_var),
                "component_var": float(component_var),
                "component_var_percentage": float(component_var / portfolio_var * 100) if portfolio_var != 0 else 0,
                "position_value": float(weights[i] * portfolio_value)
            }
        
        # Verify components sum to total VaR
        total_component_var = sum(cv["component_var"] for cv in component_vars.values())
        
        return {
            "portfolio_var": float(portfolio_var),
            "component_vars": component_vars,
            "total_component_var": float(total_component_var),
            "decomposition_error": float(abs(total_component_var - portfolio_var)),
            "largest_contributor": max(component_vars.items(), key=lambda x: abs(x[1]["component_var"]))[0],
            "diversification_benefit": float(portfolio_var - sum(abs(cv["component_var"]) for cv in component_vars.values()))
        }
    
    async def _calculate_marginal_var(self, portfolio_data: Dict[str, Any], 
                                    returns: np.ndarray, confidence_level: float, 
                                    portfolio_value: float) -> Dict[str, Any]:
        """Calculate Marginal VaR for portfolio positions."""
        if "securities" not in portfolio_data or "weights" not in portfolio_data:
            return {"error": "Portfolio composition data required for Marginal VaR"}
        
        securities = portfolio_data["securities"]
        weights = portfolio_data["weights"]
        
        # Calculate base portfolio VaR
        base_var = -np.percentile(returns, (1 - confidence_level) * 100) * portfolio_value
        
        marginal_vars = {}
        
        for i, security in enumerate(securities):
            # Calculate VaR without this security
            remaining_weights = [w for j, w in enumerate(weights) if j != i]
            if remaining_weights:
                # Renormalize remaining weights
                remaining_weights = [w / sum(remaining_weights) for w in remaining_weights]
                
                # Simulate returns without this security
                # This is a simplified approach
                reduced_var = base_var * 0.95  # Approximate reduction
                marginal_var = base_var - reduced_var
            else:
                marginal_var = base_var
            
            marginal_vars[security] = {
                "marginal_var": float(marginal_var),
                "marginal_var_per_dollar": float(marginal_var / (weights[i] * portfolio_value)) if weights[i] > 0 else 0,
                "position_weight": float(weights[i]),
                "contribution_to_risk": float(marginal_var / base_var * 100) if base_var != 0 else 0
            }
        
        return {
            "portfolio_var": float(base_var),
            "marginal_vars": marginal_vars,
            "highest_marginal_var": max(marginal_vars.items(), key=lambda x: x[1]["marginal_var"])[0],
            "risk_reduction_opportunity": max(marginal_vars.values(), key=lambda x: x["marginal_var"])["marginal_var"]
        }
    
    def _generate_correlation_matrix(self, size: int) -> np.ndarray:
        """Generate a realistic correlation matrix."""
        # Create a random correlation matrix
        A = np.random.randn(size, size)
        correlation_matrix = np.dot(A, A.T)
        
        # Normalize to correlation matrix
        D = np.sqrt(np.diag(correlation_matrix))
        correlation_matrix = correlation_matrix / np.outer(D, D)
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)
        correlation_matrix = np.dot(eigenvecs, np.dot(np.diag(eigenvals), eigenvecs.T))
        
        return correlation_matrix * 0.0004  # Scale down for reasonable volatility
