"""
Market Risk Analyzer - Advanced Market Risk Analysis Agent

This agent specializes in comprehensive market risk analysis including:
- Portfolio volatility calculation
- Beta analysis and systematic risk
- Value at Risk (VaR) estimation
- Maximum Drawdown analysis
- Correlation and dependency analysis

Author: Jifeng Li
License: openMDW
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from scipy import stats
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

from ..registry import BaseRiskAgent
from ..memory_bridge import RiskAnalysisRecord


class MarketRiskAnalyzer(BaseRiskAgent):
    """
    Advanced market risk analyzer with multiple risk calculation methodologies.
    
    This agent provides comprehensive market risk analysis including volatility,
    VaR, beta, correlation analysis, and stress testing capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "MarketRiskAnalyzer"
        self.logger = logging.getLogger(f"RiskAgent.{self.name}")
        
        # Configuration parameters
        self.confidence_levels = config.get('confidence_levels', [0.95, 0.99]) if config else [0.95, 0.99]
        self.lookback_periods = config.get('lookback_periods', [30, 60, 252]) if config else [30, 60, 252]
        self.var_methods = config.get('var_methods', ['parametric', 'historical', 'monte_carlo']) if config else ['parametric', 'historical', 'monte_carlo']
        
        # Model parameters
        self.monte_carlo_simulations = 10000
        self.stress_test_scenarios = 1000
    
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive market risk analysis.
        
        Args:
            request: Analysis request containing portfolio data and parameters
            
        Returns:
            Dictionary containing comprehensive market risk metrics
        """
        start_time = datetime.utcnow()
        
        try:
            # Extract request parameters
            portfolio_data = request.get("portfolio_data", {})
            risk_measures = request.get("risk_measures", ["volatility", "var", "beta"])
            time_horizon = request.get("time_horizon", "daily")
            confidence_levels = request.get("confidence_levels", self.confidence_levels)
            
            # Validate input data
            if not self._validate_portfolio_data(portfolio_data):
                return {
                    "agent": self.name,
                    "status": "error",
                    "error": "Invalid portfolio data provided",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Initialize results structure
            results = {
                "portfolio_summary": self._get_portfolio_summary(portfolio_data),
                "risk_metrics": {}
            }
            
            # Calculate requested risk measures
            if "volatility" in risk_measures:
                results["risk_metrics"]["volatility"] = await self._calculate_volatility_metrics(
                    portfolio_data, time_horizon
                )
            
            if "var" in risk_measures:
                results["risk_metrics"]["var"] = await self._calculate_var_metrics(
                    portfolio_data, time_horizon, confidence_levels
                )
            
            if "beta" in risk_measures:
                results["risk_metrics"]["beta"] = await self._calculate_beta_metrics(
                    portfolio_data
                )
            
            if "correlation" in risk_measures:
                results["risk_metrics"]["correlation"] = await self._calculate_correlation_metrics(
                    portfolio_data
                )
            
            if "drawdown" in risk_measures:
                results["risk_metrics"]["drawdown"] = await self._calculate_drawdown_metrics(
                    portfolio_data
                )
            
            # Add stress testing if requested
            if request.get("include_stress_test", False):
                results["stress_test"] = await self._perform_stress_testing(
                    portfolio_data, time_horizon
                )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Create analysis record for memory storage
            analysis_record = RiskAnalysisRecord(
                id=f"market_risk_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                agent_name=self.name,
                risk_type="market",
                analysis_type="comprehensive",
                input_parameters=request,
                results=results,
                timestamp=datetime.utcnow().isoformat(),
                execution_time_ms=execution_time,
                status="success"
            )
            
            return {
                "agent": self.name,
                "risk_type": "market",
                "analysis_type": "comprehensive",
                "results": results,
                "execution_time_ms": execution_time,
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "analysis_record": analysis_record
            }
            
        except Exception as e:
            self.logger.error(f"Market risk analysis failed: {e}")
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "agent": self.name,
                "risk_type": "market",
                "error": str(e),
                "execution_time_ms": execution_time,
                "status": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _validate_portfolio_data(self, portfolio_data: Dict[str, Any]) -> bool:
        """Validate the portfolio data structure and content."""
        try:
            # Check for required fields
            required_fields = ["securities", "weights"]
            for field in required_fields:
                if field not in portfolio_data:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate securities and weights
            securities = portfolio_data["securities"]
            weights = portfolio_data["weights"]
            
            if not isinstance(securities, list) or not isinstance(weights, list):
                self.logger.error("Securities and weights must be lists")
                return False
            
            if len(securities) != len(weights):
                self.logger.error("Securities and weights must have the same length")
                return False
            
            # Check if weights sum to approximately 1
            weight_sum = sum(weights)
            if abs(weight_sum - 1.0) > 0.01:
                self.logger.warning(f"Portfolio weights sum to {weight_sum}, not 1.0")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Portfolio data validation failed: {e}")
            return False
    
    def _get_portfolio_summary(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate portfolio summary statistics."""
        try:
            securities = portfolio_data["securities"]
            weights = portfolio_data["weights"]
            
            return {
                "num_securities": len(securities),
                "weight_distribution": {
                    "min_weight": min(weights),
                    "max_weight": max(weights),
                    "mean_weight": np.mean(weights),
                    "weight_concentration": self._calculate_herfindahl_index(weights)
                },
                "portfolio_composition": dict(zip(securities, weights))
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio summary calculation failed: {e}")
            return {}
    
    def _calculate_herfindahl_index(self, weights: List[float]) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration measurement."""
        return sum(w**2 for w in weights)
    
    async def _calculate_volatility_metrics(self, portfolio_data: Dict[str, Any], 
                                          time_horizon: str) -> Dict[str, Any]:
        """Calculate comprehensive volatility metrics."""
        try:
            # Simulate historical returns for demonstration
            # In production, this would fetch actual market data
            returns_data = self._simulate_returns_data(portfolio_data, 252)
            
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(returns_data, portfolio_data["weights"])
            
            # Calculate volatility for different periods
            volatility_metrics = {}
            
            for period in self.lookback_periods:
                if len(portfolio_returns) >= period:
                    period_returns = portfolio_returns[-period:]
                    
                    # Historical volatility
                    historical_vol = np.std(period_returns) * np.sqrt(252) if time_horizon == "daily" else np.std(period_returns)
                    
                    # EWMA volatility
                    ewma_vol = self._calculate_ewma_volatility(period_returns)
                    
                    # GARCH volatility (simplified)
                    garch_vol = self._calculate_garch_volatility(period_returns)
                    
                    volatility_metrics[f"{period}_day"] = {
                        "historical_volatility": float(historical_vol),
                        "ewma_volatility": float(ewma_vol),
                        "garch_volatility": float(garch_vol),
                        "volatility_percentiles": {
                            "5th": float(np.percentile(period_returns, 5)),
                            "25th": float(np.percentile(period_returns, 25)),
                            "75th": float(np.percentile(period_returns, 75)),
                            "95th": float(np.percentile(period_returns, 95))
                        }
                    }
            
            # Add current volatility estimate
            current_volatility = np.std(portfolio_returns[-30:]) * np.sqrt(252) if len(portfolio_returns) >= 30 else 0
            
            return {
                "current_volatility": float(current_volatility),
                "volatility_by_period": volatility_metrics,
                "volatility_regime": self._classify_volatility_regime(current_volatility),
                "volatility_forecast": await self._forecast_volatility(portfolio_returns)
            }
            
        except Exception as e:
            self.logger.error(f"Volatility calculation failed: {e}")
            return {"error": str(e)}
    
    def _simulate_returns_data(self, portfolio_data: Dict[str, Any], periods: int) -> pd.DataFrame:
        """Simulate historical returns data for demonstration purposes."""
        # In production, this would fetch actual market data from a data provider
        securities = portfolio_data["securities"]
        
        # Generate correlated random returns
        num_securities = len(securities)
        correlation_matrix = self._generate_correlation_matrix(num_securities)
        
        # Generate returns using multivariate normal distribution
        mean_returns = np.random.normal(0.0008, 0.0002, num_securities)  # Daily returns around 0.08% with variation
        returns = np.random.multivariate_normal(mean_returns, correlation_matrix * 0.0004, periods)
        
        return pd.DataFrame(returns, columns=securities)
    
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
        eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive eigenvalues
        correlation_matrix = np.dot(eigenvecs, np.dot(np.diag(eigenvals), eigenvecs.T))
        
        return correlation_matrix
    
    def _calculate_portfolio_returns(self, returns_data: pd.DataFrame, weights: List[float]) -> np.ndarray:
        """Calculate portfolio returns from individual security returns."""
        weights_array = np.array(weights)
        portfolio_returns = np.dot(returns_data.values, weights_array)
        return portfolio_returns
    
    def _calculate_ewma_volatility(self, returns: np.ndarray, lambda_param: float = 0.94) -> float:
        """Calculate Exponentially Weighted Moving Average volatility."""
        if len(returns) < 2:
            return 0.0
        
        squared_returns = returns**2
        weights = np.array([(1 - lambda_param) * (lambda_param**i) for i in range(len(squared_returns))])
        weights = weights[::-1]  # Reverse so most recent gets highest weight
        weights = weights / weights.sum()  # Normalize
        
        ewma_variance = np.dot(weights, squared_returns)
        return np.sqrt(ewma_variance * 252)  # Annualized
    
    def _calculate_garch_volatility(self, returns: np.ndarray) -> float:
        """Calculate GARCH(1,1) volatility (simplified implementation)."""
        if len(returns) < 10:
            return np.std(returns) * np.sqrt(252)
        
        # Simplified GARCH(1,1) with fixed parameters
        omega = 0.000001
        alpha = 0.1
        beta = 0.85
        
        variance = np.var(returns)
        for ret in returns:
            variance = omega + alpha * (ret**2) + beta * variance
        
        return np.sqrt(variance * 252)  # Annualized
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify current volatility regime."""
        if volatility < 0.15:
            return "low_volatility"
        elif volatility < 0.25:
            return "normal_volatility"
        elif volatility < 0.35:
            return "high_volatility"
        else:
            return "extreme_volatility"
    
    async def _forecast_volatility(self, returns: np.ndarray) -> Dict[str, Any]:
        """Forecast future volatility using multiple models."""
        try:
            if len(returns) < 30:
                return {"error": "Insufficient data for volatility forecasting"}
            
            # Simple forecasts using different methods
            current_vol = np.std(returns[-30:]) * np.sqrt(252)
            trend_vol = self._calculate_volatility_trend(returns)
            mean_reversion_vol = self._calculate_mean_reversion_forecast(returns)
            
            return {
                "1_day_forecast": float(current_vol * 1.02),  # Slight persistence
                "5_day_forecast": float(current_vol * 1.05),
                "30_day_forecast": float(mean_reversion_vol),
                "trend_component": float(trend_vol),
                "mean_reversion_component": float(mean_reversion_vol),
                "forecast_confidence": 0.75  # Placeholder confidence
            }
            
        except Exception as e:
            self.logger.error(f"Volatility forecasting failed: {e}")
            return {"error": str(e)}
    
    def _calculate_volatility_trend(self, returns: np.ndarray) -> float:
        """Calculate volatility trend component."""
        if len(returns) < 60:
            return np.std(returns) * np.sqrt(252)
        
        # Calculate rolling volatilities
        window = 30
        rolling_vols = []
        for i in range(window, len(returns)):
            vol = np.std(returns[i-window:i]) * np.sqrt(252)
            rolling_vols.append(vol)
        
        # Calculate trend
        if len(rolling_vols) > 1:
            x = np.arange(len(rolling_vols))
            slope, _, _, _, _ = stats.linregress(x, rolling_vols)
            return rolling_vols[-1] + slope * 5  # 5-day trend projection
        
        return rolling_vols[-1] if rolling_vols else np.std(returns) * np.sqrt(252)
    
    def _calculate_mean_reversion_forecast(self, returns: np.ndarray) -> float:
        """Calculate mean reversion volatility forecast."""
        # Long-term volatility
        long_term_vol = np.std(returns) * np.sqrt(252)
        
        # Current volatility
        current_vol = np.std(returns[-30:]) * np.sqrt(252) if len(returns) >= 30 else long_term_vol
        
        # Mean reversion with half-life of 30 days
        decay_factor = 0.5**(1/30)
        mean_reversion_vol = long_term_vol + (current_vol - long_term_vol) * (decay_factor**30)
        
        return mean_reversion_vol
    
    async def _calculate_var_metrics(self, portfolio_data: Dict[str, Any], 
                                   time_horizon: str, confidence_levels: List[float]) -> Dict[str, Any]:
        """Calculate Value at Risk using multiple methodologies."""
        try:
            # Simulate returns data
            returns_data = self._simulate_returns_data(portfolio_data, 252)
            portfolio_returns = self._calculate_portfolio_returns(returns_data, portfolio_data["weights"])
            
            var_results = {}
            
            for method in self.var_methods:
                var_results[method] = {}
                
                for confidence_level in confidence_levels:
                    if method == "parametric":
                        var_value = self._calculate_parametric_var(portfolio_returns, confidence_level)
                    elif method == "historical":
                        var_value = self._calculate_historical_var(portfolio_returns, confidence_level)
                    elif method == "monte_carlo":
                        var_value = await self._calculate_monte_carlo_var(portfolio_returns, confidence_level)
                    else:
                        var_value = 0.0
                    
                    var_results[method][f"var_{int(confidence_level*100)}"] = float(var_value)
            
            # Calculate Expected Shortfall (CVaR)
            cvar_results = {}
            for confidence_level in confidence_levels:
                cvar_value = self._calculate_expected_shortfall(portfolio_returns, confidence_level)
                cvar_results[f"es_{int(confidence_level*100)}"] = float(cvar_value)
            
            # Add backtesting results
            backtesting = self._perform_var_backtesting(portfolio_returns, confidence_levels)
            
            return {
                "var_estimates": var_results,
                "expected_shortfall": cvar_results,
                "backtesting_results": backtesting,
                "var_attribution": await self._calculate_var_attribution(portfolio_data, portfolio_returns)
            }
            
        except Exception as e:
            self.logger.error(f"VaR calculation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_parametric_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        z_score = stats.norm.ppf(1 - confidence_level)
        
        var = -(mean_return + z_score * std_return)
        return var
    
    def _calculate_historical_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate historical VaR using empirical distribution."""
        if len(returns) == 0:
            return 0.0
        
        percentile = (1 - confidence_level) * 100
        var = -np.percentile(returns, percentile)
        return var
    
    async def _calculate_monte_carlo_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Monte Carlo VaR."""
        if len(returns) == 0:
            return 0.0
        
        # Simulate future returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        simulated_returns = np.random.normal(mean_return, std_return, self.monte_carlo_simulations)
        
        percentile = (1 - confidence_level) * 100
        var = -np.percentile(simulated_returns, percentile)
        return var
    
    def _calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(returns) == 0:
            return 0.0
        
        var_threshold = -self._calculate_historical_var(returns, confidence_level)
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) > 0:
            expected_shortfall = -np.mean(tail_losses)
        else:
            expected_shortfall = -var_threshold
        
        return expected_shortfall
    
    def _perform_var_backtesting(self, returns: np.ndarray, confidence_levels: List[float]) -> Dict[str, Any]:
        """Perform VaR backtesting using Kupiec test."""
        backtesting_results = {}
        
        if len(returns) < 100:
            return {"error": "Insufficient data for backtesting"}
        
        # Use last 100 observations for backtesting
        test_returns = returns[-100:]
        
        for confidence_level in confidence_levels:
            # Calculate VaR for each day using rolling window
            violations = 0
            for i in range(50, len(test_returns)):  # Start from day 50 to have enough history
                historical_returns = test_returns[:i]
                var_estimate = self._calculate_historical_var(historical_returns, confidence_level)
                actual_return = test_returns[i]
                
                if actual_return < -var_estimate:
                    violations += 1
            
            total_observations = len(test_returns) - 50
            violation_rate = violations / total_observations
            expected_violation_rate = 1 - confidence_level
            
            # Kupiec test statistic
            if violation_rate > 0 and violation_rate < 1:
                lr_stat = 2 * (violations * np.log(violation_rate / expected_violation_rate) + 
                              (total_observations - violations) * np.log((1 - violation_rate) / (1 - expected_violation_rate)))
            else:
                lr_stat = 0
            
            backtesting_results[f"confidence_{int(confidence_level*100)}"] = {
                "violations": violations,
                "total_observations": total_observations,
                "violation_rate": violation_rate,
                "expected_violation_rate": expected_violation_rate,
                "kupiec_lr_statistic": lr_stat,
                "test_result": "pass" if abs(violation_rate - expected_violation_rate) < 0.05 else "fail"
            }
        
        return backtesting_results
    
    async def _calculate_var_attribution(self, portfolio_data: Dict[str, Any], 
                                       portfolio_returns: np.ndarray) -> Dict[str, Any]:
        """Calculate VaR attribution by security."""
        try:
            securities = portfolio_data["securities"]
            weights = portfolio_data["weights"]
            
            # Simulate individual security returns
            returns_data = self._simulate_returns_data(portfolio_data, 252)
            
            # Calculate marginal VaR for each security
            var_attribution = {}
            base_var = self._calculate_historical_var(portfolio_returns, 0.95)
            
            for i, security in enumerate(securities):
                # Calculate VaR without this security (rebalancing remaining weights)
                remaining_weights = [w for j, w in enumerate(weights) if j != i]
                if remaining_weights:
                    remaining_weights = [w / sum(remaining_weights) for w in remaining_weights]
                    remaining_securities = [s for j, s in enumerate(securities) if j != i]
                    remaining_returns_data = returns_data[remaining_securities]
                    
                    reduced_portfolio_returns = self._calculate_portfolio_returns(
                        remaining_returns_data, remaining_weights
                    )
                    reduced_var = self._calculate_historical_var(reduced_portfolio_returns, 0.95)
                    
                    marginal_var = base_var - reduced_var
                    component_var = marginal_var * weights[i]
                    
                    var_attribution[security] = {
                        "marginal_var": float(marginal_var),
                        "component_var": float(component_var),
                        "var_contribution_percent": float(component_var / base_var * 100) if base_var != 0 else 0
                    }
            
            return var_attribution
            
        except Exception as e:
            self.logger.error(f"VaR attribution calculation failed: {e}")
            return {"error": str(e)}
    
    async def _calculate_beta_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio beta and related metrics."""
        try:
            # Simulate market and portfolio returns
            market_returns = np.random.normal(0.0008, 0.012, 252)  # Market returns
            returns_data = self._simulate_returns_data(portfolio_data, 252)
            portfolio_returns = self._calculate_portfolio_returns(returns_data, portfolio_data["weights"])
            
            # Calculate beta
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance != 0 else 0
            
            # Calculate alpha (Jensen's alpha)
            portfolio_mean = np.mean(portfolio_returns)
            market_mean = np.mean(market_returns)
            risk_free_rate = 0.02 / 252  # Assume 2% annual risk-free rate
            alpha = portfolio_mean - (risk_free_rate + beta * (market_mean - risk_free_rate))
            
            # Calculate R-squared
            correlation = np.corrcoef(portfolio_returns, market_returns)[0, 1]
            r_squared = correlation**2
            
            # Calculate systematic and idiosyncratic risk
            portfolio_variance = np.var(portfolio_returns)
            systematic_variance = beta**2 * market_variance
            idiosyncratic_variance = portfolio_variance - systematic_variance
            
            return {
                "portfolio_beta": float(beta),
                "jensen_alpha": float(alpha * 252),  # Annualized
                "r_squared": float(r_squared),
                "correlation_with_market": float(correlation),
                "systematic_risk": float(np.sqrt(systematic_variance) * np.sqrt(252)),  # Annualized
                "idiosyncratic_risk": float(np.sqrt(idiosyncratic_variance) * np.sqrt(252)),  # Annualized
                "total_risk": float(np.sqrt(portfolio_variance) * np.sqrt(252)),  # Annualized
                "risk_decomposition": {
                    "systematic_risk_contribution": float(systematic_variance / portfolio_variance * 100),
                    "idiosyncratic_risk_contribution": float(idiosyncratic_variance / portfolio_variance * 100)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Beta calculation failed: {e}")
            return {"error": str(e)}
    
    async def _calculate_correlation_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlation and dependency metrics."""
        try:
            # Simulate returns data
            returns_data = self._simulate_returns_data(portfolio_data, 252)
            
            # Calculate correlation matrix
            correlation_matrix = returns_data.corr()
            
            # Calculate portfolio correlation metrics
            correlations = correlation_matrix.values
            upper_triangle = correlations[np.triu_indices_from(correlations, k=1)]
            
            # Principal component analysis
            eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix.values)
            eigenvalues = eigenvalues[::-1]  # Sort in descending order
            explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
            
            return {
                "correlation_statistics": {
                    "mean_correlation": float(np.mean(upper_triangle)),
                    "median_correlation": float(np.median(upper_triangle)),
                    "max_correlation": float(np.max(upper_triangle)),
                    "min_correlation": float(np.min(upper_triangle)),
                    "correlation_std": float(np.std(upper_triangle))
                },
                "principal_components": {
                    "eigenvalues": eigenvalues.tolist(),
                    "explained_variance_ratio": explained_variance_ratio.tolist(),
                    "num_components_95_variance": int(np.argmax(np.cumsum(explained_variance_ratio) >= 0.95) + 1)
                },
                "correlation_matrix": correlation_matrix.round(3).to_dict(),
                "diversification_metrics": {
                    "diversification_ratio": self._calculate_diversification_ratio(
                        correlation_matrix.values, portfolio_data["weights"]
                    ),
                    "effective_number_of_assets": self._calculate_effective_number_of_assets(
                        correlation_matrix.values, portfolio_data["weights"]
                    )
                }
            }
            
        except Exception as e:
            self.logger.error(f"Correlation calculation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_diversification_ratio(self, correlation_matrix: np.ndarray, weights: List[float]) -> float:
        """Calculate the diversification ratio."""
        weights_array = np.array(weights)
        
        # Weighted average of individual volatilities
        individual_vols = np.sqrt(np.diag(correlation_matrix))
        weighted_avg_vol = np.dot(weights_array, individual_vols)
        
        # Portfolio volatility
        portfolio_variance = np.dot(weights_array, np.dot(correlation_matrix, weights_array))
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Diversification ratio
        if portfolio_vol > 0:
            diversification_ratio = weighted_avg_vol / portfolio_vol
        else:
            diversification_ratio = 1.0
        
        return float(diversification_ratio)
    
    def _calculate_effective_number_of_assets(self, correlation_matrix: np.ndarray, weights: List[float]) -> float:
        """Calculate the effective number of assets in the portfolio."""
        weights_array = np.array(weights)
        
        # Portfolio variance
        portfolio_variance = np.dot(weights_array, np.dot(correlation_matrix, weights_array))
        
        # Sum of squared weighted volatilities
        individual_vols = np.sqrt(np.diag(correlation_matrix))
        weighted_vol_squared_sum = np.sum((weights_array * individual_vols)**2)
        
        # Effective number of assets
        if weighted_vol_squared_sum > 0:
            effective_assets = portfolio_variance / weighted_vol_squared_sum
        else:
            effective_assets = 1.0
        
        return float(effective_assets)
    
    async def _calculate_drawdown_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate drawdown metrics including maximum drawdown."""
        try:
            # Simulate portfolio returns
            returns_data = self._simulate_returns_data(portfolio_data, 252)
            portfolio_returns = self._calculate_portfolio_returns(returns_data, portfolio_data["weights"])
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative_returns)
            
            # Calculate drawdowns
            drawdowns = (cumulative_returns - running_max) / running_max
            
            # Find maximum drawdown
            max_drawdown = np.min(drawdowns)
            max_drawdown_index = np.argmin(drawdowns)
            
            # Calculate drawdown duration
            drawdown_start_index = np.argmax(cumulative_returns[:max_drawdown_index+1])
            drawdown_duration = max_drawdown_index - drawdown_start_index
            
            # Calculate recovery time
            recovery_index = None
            for i in range(max_drawdown_index, len(cumulative_returns)):
                if cumulative_returns[i] >= running_max[max_drawdown_index]:
                    recovery_index = i
                    break
            
            recovery_time = recovery_index - max_drawdown_index if recovery_index else None
            
            # Calculate other drawdown statistics
            drawdown_percentiles = np.percentile(drawdowns, [5, 10, 25, 50, 75, 90, 95])
            
            return {
                "maximum_drawdown": float(max_drawdown),
                "max_drawdown_duration_days": int(drawdown_duration),
                "recovery_time_days": int(recovery_time) if recovery_time else None,
                "current_drawdown": float(drawdowns[-1]),
                "drawdown_percentiles": {
                    "5th": float(drawdown_percentiles[0]),
                    "10th": float(drawdown_percentiles[1]),
                    "25th": float(drawdown_percentiles[2]),
                    "50th": float(drawdown_percentiles[3]),
                    "75th": float(drawdown_percentiles[4]),
                    "90th": float(drawdown_percentiles[5]),
                    "95th": float(drawdown_percentiles[6])
                },
                "average_drawdown": float(np.mean(drawdowns[drawdowns < 0])) if np.any(drawdowns < 0) else 0.0,
                "drawdown_frequency": int(np.sum(drawdowns < -0.05)),  # Number of drawdowns > 5%
                "calmar_ratio": float(np.mean(portfolio_returns) * 252 / abs(max_drawdown)) if max_drawdown != 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Drawdown calculation failed: {e}")
            return {"error": str(e)}
    
    async def _perform_stress_testing(self, portfolio_data: Dict[str, Any], 
                                    time_horizon: str) -> Dict[str, Any]:
        """Perform comprehensive stress testing scenarios."""
        try:
            # Define stress scenarios
            stress_scenarios = {
                "market_crash": {"market_shock": -0.20, "volatility_shock": 2.0},
                "interest_rate_shock": {"rate_shock": 0.02, "volatility_shock": 1.5},
                "credit_crisis": {"credit_shock": 0.05, "correlation_shock": 1.8},
                "liquidity_crisis": {"liquidity_shock": 0.10, "bid_ask_shock": 3.0},
                "tail_risk_scenario": {"tail_shock": -0.35, "volatility_shock": 3.0}
            }
            
            # Simulate base portfolio returns
            returns_data = self._simulate_returns_data(portfolio_data, 252)
            portfolio_returns = self._calculate_portfolio_returns(returns_data, portfolio_data["weights"])
            
            stress_results = {}
            
            for scenario_name, scenario_params in stress_scenarios.items():
                # Apply stress scenario
                stressed_returns = self._apply_stress_scenario(portfolio_returns, scenario_params)
                
                # Calculate stressed metrics
                stressed_metrics = {
                    "stressed_var_95": float(self._calculate_historical_var(stressed_returns, 0.95)),
                    "stressed_var_99": float(self._calculate_historical_var(stressed_returns, 0.99)),
                    "stressed_volatility": float(np.std(stressed_returns) * np.sqrt(252)),
                    "stressed_return": float(np.mean(stressed_returns) * 252),
                    "worst_case_loss": float(np.min(stressed_returns)),
                    "probability_of_loss": float(np.sum(stressed_returns < 0) / len(stressed_returns))
                }
                
                stress_results[scenario_name] = stressed_metrics
            
            # Add reverse stress testing
            reverse_stress = await self._perform_reverse_stress_testing(portfolio_returns)
            stress_results["reverse_stress_testing"] = reverse_stress
            
            return stress_results
            
        except Exception as e:
            self.logger.error(f"Stress testing failed: {e}")
            return {"error": str(e)}
    
    def _apply_stress_scenario(self, returns: np.ndarray, scenario_params: Dict[str, float]) -> np.ndarray:
        """Apply stress scenario to portfolio returns."""
        stressed_returns = returns.copy()
        
        # Apply market shock
        if "market_shock" in scenario_params:
            market_shock = scenario_params["market_shock"]
            stressed_returns = stressed_returns + market_shock / 252  # Distribute shock over year
        
        # Apply volatility shock
        if "volatility_shock" in scenario_params:
            vol_multiplier = scenario_params["volatility_shock"]
            current_vol = np.std(returns)
            vol_adjustment = (vol_multiplier - 1) * current_vol
            noise = np.random.normal(0, vol_adjustment, len(returns))
            stressed_returns = stressed_returns + noise
        
        # Apply other shocks as needed
        # This is a simplified implementation
        
        return stressed_returns
    
    async def _perform_reverse_stress_testing(self, returns: np.ndarray) -> Dict[str, Any]:
        """Perform reverse stress testing to find scenarios that cause specific losses."""
        try:
            target_losses = [-0.10, -0.15, -0.20, -0.25]  # Target loss levels
            reverse_stress_results = {}
            
            current_vol = np.std(returns)
            current_mean = np.mean(returns)
            
            for target_loss in target_losses:
                # Calculate required shock to achieve target loss
                required_shock = target_loss - current_mean
                
                # Calculate probability of such shock
                shock_probability = 1 - stats.norm.cdf(abs(required_shock) / current_vol)
                
                # Estimate market conditions needed
                vol_multiplier = abs(required_shock) / (2 * current_vol) + 1
                
                reverse_stress_results[f"loss_{abs(int(target_loss*100))}pct"] = {
                    "required_shock": float(required_shock),
                    "shock_probability": float(shock_probability),
                    "estimated_vol_multiplier": float(vol_multiplier),
                    "scenario_description": f"Market conditions requiring {vol_multiplier:.1f}x volatility increase"
                }
            
            return reverse_stress_results
            
        except Exception as e:
            self.logger.error(f"Reverse stress testing failed: {e}")
            return {"error": str(e)}
