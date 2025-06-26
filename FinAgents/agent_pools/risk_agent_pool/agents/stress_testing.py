"""
Stress Testing Agent

Author: Jifeng Li
License: openMDW
Description: Comprehensive stress testing framework for portfolio and risk management,
             including scenario generation, sensitivity analysis, and regulatory stress tests.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StressTestType(Enum):
    """Types of stress tests"""
    HISTORICAL_SCENARIO = "historical_scenario"
    HYPOTHETICAL_SCENARIO = "hypothetical_scenario"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    MONTE_CARLO = "monte_carlo"
    REGULATORY = "regulatory"
    REVERSE_STRESS = "reverse_stress"


class SeverityLevel(Enum):
    """Stress test severity levels"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"


@dataclass
class StressScenario:
    """Data class for stress test scenarios"""
    scenario_id: str
    name: str
    description: str
    stress_type: StressTestType
    severity: SeverityLevel
    risk_factors: Dict[str, float]  # factor_name -> shock_value
    probability: Optional[float] = None
    time_horizon_days: int = 1
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StressTestResult:
    """Results from stress testing"""
    scenario_id: str
    portfolio_value_change: float
    portfolio_value_change_pct: float
    var_change: float
    component_impacts: Dict[str, float]
    risk_metrics: Dict[str, float]
    breach_indicators: Dict[str, bool]
    execution_time: float
    timestamp: datetime


@dataclass
class PortfolioPosition:
    """Portfolio position for stress testing"""
    asset_id: str
    quantity: float
    current_price: float
    currency: str = "USD"
    asset_type: str = "equity"
    sector: Optional[str] = None
    country: Optional[str] = None
    duration: Optional[float] = None  # For bonds
    beta: Optional[float] = None  # For equities


class StressTester:
    """
    Comprehensive stress testing engine
    
    Provides:
    - Historical scenario replay
    - Hypothetical scenario analysis
    - Sensitivity analysis
    - Monte Carlo stress testing
    - Regulatory stress tests (CCAR, EBA, etc.)
    - Reverse stress testing
    """
    
    def __init__(self):
        """Initialize the stress tester"""
        self.scenarios: Dict[str, StressScenario] = {}
        self.results_history: List[StressTestResult] = []
        self.risk_factor_correlations: Optional[np.ndarray] = None
        self.factor_names: List[str] = []
        
        # Load predefined scenarios
        self._load_predefined_scenarios()
    
    def _load_predefined_scenarios(self):
        """Load predefined stress scenarios"""
        # 2008 Financial Crisis scenario
        self.add_scenario(StressScenario(
            scenario_id="2008_financial_crisis",
            name="2008 Financial Crisis",
            description="Historical replay of 2008 financial crisis conditions",
            stress_type=StressTestType.HISTORICAL_SCENARIO,
            severity=SeverityLevel.EXTREME,
            risk_factors={
                "equity_market": -0.40,  # 40% equity market decline
                "credit_spreads": 0.35,  # 350 bps credit spread widening
                "interest_rates": -0.015,  # 150 bps rate decline
                "volatility": 0.80,  # 80% volatility increase
                "liquidity": -0.60,  # 60% liquidity reduction
                "real_estate": -0.25,  # 25% real estate decline
                "commodities": -0.30  # 30% commodity decline
            },
            probability=0.01,  # 1% annual probability
            time_horizon_days=252  # 1 year
        ))
        
        # COVID-19 pandemic scenario
        self.add_scenario(StressScenario(
            scenario_id="covid19_pandemic",
            name="COVID-19 Pandemic",
            description="Economic stress from global pandemic and lockdowns",
            stress_type=StressTestType.HISTORICAL_SCENARIO,
            severity=SeverityLevel.SEVERE,
            risk_factors={
                "equity_market": -0.35,
                "credit_spreads": 0.25,
                "interest_rates": -0.02,
                "volatility": 1.20,
                "oil_prices": -0.65,
                "travel_sector": -0.70,
                "technology_sector": 0.15,
                "healthcare_sector": 0.05
            },
            probability=0.02,
            time_horizon_days=90
        ))
        
        # Interest rate shock scenario
        self.add_scenario(StressScenario(
            scenario_id="rate_shock_up",
            name="Interest Rate Shock Up",
            description="Rapid 300 bps interest rate increase",
            stress_type=StressTestType.HYPOTHETICAL_SCENARIO,
            severity=SeverityLevel.SEVERE,
            risk_factors={
                "interest_rates": 0.03,  # 300 bps increase
                "bond_prices": -0.15,  # Duration-adjusted bond decline
                "equity_market": -0.20,  # Equity decline due to discount rate impact
                "credit_spreads": 0.15,  # Credit spread widening
                "currency_usd": 0.10,  # USD strengthening
                "real_estate": -0.18  # Real estate decline
            },
            probability=0.05,
            time_horizon_days=30
        ))
    
    def add_scenario(self, scenario: StressScenario):
        """Add a stress scenario to the library"""
        self.scenarios[scenario.scenario_id] = scenario
        logger.info(f"Added stress scenario: {scenario.name}")
    
    def set_risk_factor_correlations(
        self,
        correlation_matrix: np.ndarray,
        factor_names: List[str]
    ):
        """Set risk factor correlation matrix for Monte Carlo simulations"""
        self.risk_factor_correlations = correlation_matrix
        self.factor_names = factor_names
        logger.info(f"Set risk factor correlations for {len(factor_names)} factors")
    
    async def run_stress_test(
        self,
        scenario_id: str,
        portfolio: List[PortfolioPosition],
        pricing_functions: Dict[str, Callable] = None
    ) -> StressTestResult:
        """
        Run a stress test for a given scenario and portfolio
        
        Args:
            scenario_id: ID of the stress scenario
            portfolio: List of portfolio positions
            pricing_functions: Custom pricing functions for different asset types
            
        Returns:
            StressTestResult: Results of the stress test
        """
        start_time = datetime.now()
        
        try:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                raise ValueError(f"Scenario {scenario_id} not found")
            
            # Calculate baseline portfolio value
            baseline_value = sum(pos.quantity * pos.current_price for pos in portfolio)
            
            # Apply stress scenario
            stressed_portfolio_value = 0.0
            component_impacts = {}
            
            for position in portfolio:
                # Get asset-specific stress factors
                stressed_price = self._apply_stress_to_position(position, scenario)
                stressed_value = position.quantity * stressed_price
                original_value = position.quantity * position.current_price
                
                stressed_portfolio_value += stressed_value
                component_impacts[position.asset_id] = stressed_value - original_value
            
            # Calculate portfolio-level impacts
            portfolio_value_change = stressed_portfolio_value - baseline_value
            portfolio_value_change_pct = (portfolio_value_change / baseline_value * 100) if baseline_value != 0 else 0
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_stress_risk_metrics(
                portfolio, scenario, baseline_value, stressed_portfolio_value
            )
            
            # Check breach indicators
            breach_indicators = self._check_breach_indicators(
                portfolio_value_change_pct, risk_metrics
            )
            
            # Create result
            result = StressTestResult(
                scenario_id=scenario_id,
                portfolio_value_change=portfolio_value_change,
                portfolio_value_change_pct=portfolio_value_change_pct,
                var_change=risk_metrics.get('var_change', 0.0),
                component_impacts=component_impacts,
                risk_metrics=risk_metrics,
                breach_indicators=breach_indicators,
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now()
            )
            
            self.results_history.append(result)
            
            logger.info(f"Completed stress test {scenario_id}: {portfolio_value_change_pct:.2f}% portfolio impact")
            return result
            
        except Exception as e:
            logger.error(f"Error running stress test {scenario_id}: {str(e)}")
            raise
    
    async def run_sensitivity_analysis(
        self,
        portfolio: List[PortfolioPosition],
        risk_factor: str,
        shock_range: Tuple[float, float],
        num_points: int = 21
    ) -> Dict[str, Any]:
        """
        Run sensitivity analysis for a single risk factor
        
        Args:
            portfolio: Portfolio positions
            risk_factor: Risk factor to analyze
            shock_range: (min_shock, max_shock) range
            num_points: Number of data points
            
        Returns:
            Dict containing sensitivity analysis results
        """
        try:
            baseline_value = sum(pos.quantity * pos.current_price for pos in portfolio)
            
            # Generate shock values
            shock_values = np.linspace(shock_range[0], shock_range[1], num_points)
            portfolio_values = []
            
            for shock in shock_values:
                # Create temporary scenario
                temp_scenario = StressScenario(
                    scenario_id=f"sensitivity_{risk_factor}",
                    name=f"Sensitivity Analysis - {risk_factor}",
                    description=f"Sensitivity test for {risk_factor}",
                    stress_type=StressTestType.SENSITIVITY_ANALYSIS,
                    severity=SeverityLevel.MODERATE,
                    risk_factors={risk_factor: shock}
                )
                
                # Calculate stressed portfolio value
                stressed_value = 0.0
                for position in portfolio:
                    stressed_price = self._apply_stress_to_position(position, temp_scenario)
                    stressed_value += position.quantity * stressed_price
                
                portfolio_values.append(stressed_value)
            
            # Calculate sensitivity metrics
            value_changes = [(val - baseline_value) for val in portfolio_values]
            value_changes_pct = [(val / baseline_value - 1) * 100 for val in portfolio_values]
            
            # Find linear sensitivity (derivative at zero shock)
            zero_index = num_points // 2  # Middle point should be zero shock
            if zero_index < len(value_changes) - 1:
                sensitivity = (value_changes[zero_index + 1] - value_changes[zero_index - 1]) / (shock_values[zero_index + 1] - shock_values[zero_index - 1])
            else:
                sensitivity = 0.0
            
            return {
                'risk_factor': risk_factor,
                'shock_values': shock_values.tolist(),
                'portfolio_values': portfolio_values,
                'value_changes': value_changes,
                'value_changes_pct': value_changes_pct,
                'baseline_value': baseline_value,
                'linear_sensitivity': sensitivity,
                'max_loss': min(value_changes),
                'max_gain': max(value_changes),
                'shock_range': shock_range
            }
            
        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {str(e)}")
            raise
    
    async def run_monte_carlo_stress(
        self,
        portfolio: List[PortfolioPosition],
        num_simulations: int = 10000,
        time_horizon_days: int = 1,
        confidence_levels: List[float] = [0.95, 0.99, 0.999]
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo stress testing
        
        Args:
            portfolio: Portfolio positions
            num_simulations: Number of Monte Carlo simulations
            time_horizon_days: Time horizon for simulations
            confidence_levels: Confidence levels for VaR calculations
            
        Returns:
            Dict containing Monte Carlo stress test results
        """
        try:
            if self.risk_factor_correlations is None:
                logger.warning("No risk factor correlations set, using independent factors")
                correlation_matrix = np.eye(len(self.factor_names))
            else:
                correlation_matrix = self.risk_factor_correlations
            
            baseline_value = sum(pos.quantity * pos.current_price for pos in portfolio)
            portfolio_returns = []
            
            # Generate correlated random shocks
            for _ in range(num_simulations):
                # Generate correlated normal random variables
                if len(self.factor_names) > 0:
                    random_vars = np.random.multivariate_normal(
                        mean=np.zeros(len(self.factor_names)),
                        cov=correlation_matrix
                    )
                    
                    # Scale by time horizon
                    time_scaling = np.sqrt(time_horizon_days / 252)  # Assuming 252 trading days per year
                    random_vars *= time_scaling
                    
                    # Create scenario with random shocks
                    risk_factors = {
                        factor: shock for factor, shock in zip(self.factor_names, random_vars)
                    }
                else:
                    risk_factors = {}
                
                temp_scenario = StressScenario(
                    scenario_id="monte_carlo_sim",
                    name="Monte Carlo Simulation",
                    description="Random stress scenario",
                    stress_type=StressTestType.MONTE_CARLO,
                    severity=SeverityLevel.MODERATE,
                    risk_factors=risk_factors
                )
                
                # Calculate stressed portfolio value
                stressed_value = 0.0
                for position in portfolio:
                    stressed_price = self._apply_stress_to_position(position, temp_scenario)
                    stressed_value += position.quantity * stressed_price
                
                portfolio_return = (stressed_value - baseline_value) / baseline_value
                portfolio_returns.append(portfolio_return)
            
            # Calculate statistics
            portfolio_returns = np.array(portfolio_returns)
            
            # Calculate VaR and ES for different confidence levels
            var_results = {}
            es_results = {}
            
            for conf_level in confidence_levels:
                var_percentile = (1 - conf_level) * 100
                var_value = np.percentile(portfolio_returns, var_percentile)
                
                # Expected Shortfall (Conditional VaR)
                tail_losses = portfolio_returns[portfolio_returns <= var_value]
                es_value = np.mean(tail_losses) if len(tail_losses) > 0 else var_value
                
                var_results[f"VaR_{conf_level}"] = var_value * baseline_value
                es_results[f"ES_{conf_level}"] = es_value * baseline_value
            
            return {
                'num_simulations': num_simulations,
                'time_horizon_days': time_horizon_days,
                'baseline_value': baseline_value,
                'portfolio_returns': portfolio_returns.tolist(),
                'mean_return': float(np.mean(portfolio_returns)),
                'std_return': float(np.std(portfolio_returns)),
                'min_return': float(np.min(portfolio_returns)),
                'max_return': float(np.max(portfolio_returns)),
                'var_results': var_results,
                'es_results': es_results,
                'percentiles': {
                    '1%': float(np.percentile(portfolio_returns, 1)),
                    '5%': float(np.percentile(portfolio_returns, 5)),
                    '10%': float(np.percentile(portfolio_returns, 10)),
                    '90%': float(np.percentile(portfolio_returns, 90)),
                    '95%': float(np.percentile(portfolio_returns, 95)),
                    '99%': float(np.percentile(portfolio_returns, 99))
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo stress testing: {str(e)}")
            raise
    
    async def run_reverse_stress_test(
        self,
        portfolio: List[PortfolioPosition],
        target_loss_pct: float,
        max_iterations: int = 1000,
        tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """
        Run reverse stress test to find scenarios that cause a target loss
        
        Args:
            portfolio: Portfolio positions
            target_loss_pct: Target loss percentage (positive value)
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dict containing reverse stress test results
        """
        try:
            baseline_value = sum(pos.quantity * pos.current_price for pos in portfolio)
            target_loss_amount = baseline_value * (target_loss_pct / 100)
            
            # Simple iterative approach to find stress factors
            # In practice, this could use more sophisticated optimization
            best_scenario = None
            best_loss_diff = float('inf')
            
            for iteration in range(max_iterations):
                # Generate random stress factors
                stress_factors = {}
                for factor in self.factor_names:
                    # Random stress between -50% and +50%
                    stress_factors[factor] = np.random.uniform(-0.5, 0.5)
                
                # Create test scenario
                test_scenario = StressScenario(
                    scenario_id=f"reverse_test_{iteration}",
                    name="Reverse Stress Test",
                    description="Generated scenario for reverse stress testing",
                    stress_type=StressTestType.REVERSE_STRESS,
                    severity=SeverityLevel.MODERATE,
                    risk_factors=stress_factors
                )
                
                # Calculate portfolio loss
                stressed_value = 0.0
                for position in portfolio:
                    stressed_price = self._apply_stress_to_position(position, test_scenario)
                    stressed_value += position.quantity * stressed_price
                
                actual_loss = baseline_value - stressed_value
                loss_diff = abs(actual_loss - target_loss_amount)
                
                if loss_diff < best_loss_diff:
                    best_loss_diff = loss_diff
                    best_scenario = test_scenario
                    
                    # Check convergence
                    if loss_diff < tolerance * target_loss_amount:
                        break
            
            if best_scenario:
                # Calculate final results with best scenario
                final_result = await self.run_stress_test(
                    best_scenario.scenario_id,
                    portfolio
                )
                
                return {
                    'target_loss_pct': target_loss_pct,
                    'target_loss_amount': target_loss_amount,
                    'achieved_loss_pct': abs(final_result.portfolio_value_change_pct),
                    'achieved_loss_amount': abs(final_result.portfolio_value_change),
                    'stress_factors': best_scenario.risk_factors,
                    'iterations_used': iteration + 1,
                    'convergence_achieved': best_loss_diff < tolerance * target_loss_amount,
                    'scenario': best_scenario
                }
            else:
                raise ValueError("Could not find suitable stress scenario")
                
        except Exception as e:
            logger.error(f"Error in reverse stress testing: {str(e)}")
            raise
    
    def _apply_stress_to_position(
        self,
        position: PortfolioPosition,
        scenario: StressScenario
    ) -> float:
        """Apply stress scenario to individual position"""
        stressed_price = position.current_price
        
        # Apply general market stress factors
        if "equity_market" in scenario.risk_factors and position.asset_type == "equity":
            stressed_price *= (1 + scenario.risk_factors["equity_market"])
        
        if "bond_prices" in scenario.risk_factors and position.asset_type == "bond":
            stressed_price *= (1 + scenario.risk_factors["bond_prices"])
        
        if "interest_rates" in scenario.risk_factors and position.duration:
            # Duration-based bond price sensitivity
            rate_shock = scenario.risk_factors["interest_rates"]
            duration_impact = -position.duration * rate_shock
            stressed_price *= (1 + duration_impact)
        
        # Apply sector-specific stress factors
        if position.sector:
            sector_factor = f"{position.sector.lower()}_sector"
            if sector_factor in scenario.risk_factors:
                stressed_price *= (1 + scenario.risk_factors[sector_factor])
        
        # Apply country-specific stress factors
        if position.country:
            country_factor = f"{position.country.lower()}_market"
            if country_factor in scenario.risk_factors:
                stressed_price *= (1 + scenario.risk_factors[country_factor])
        
        # Apply currency stress factors
        if position.currency != "USD":
            currency_factor = f"currency_{position.currency.lower()}"
            if currency_factor in scenario.risk_factors:
                stressed_price *= (1 + scenario.risk_factors[currency_factor])
        
        return max(stressed_price, 0.0)  # Ensure non-negative prices
    
    async def _calculate_stress_risk_metrics(
        self,
        portfolio: List[PortfolioPosition],
        scenario: StressScenario,
        baseline_value: float,
        stressed_value: float
    ) -> Dict[str, float]:
        """Calculate additional risk metrics for stress test"""
        try:
            # Calculate basic metrics
            portfolio_return = (stressed_value - baseline_value) / baseline_value
            
            # Estimate VaR change (simplified calculation)
            # In practice, this would require full VaR recalculation
            var_multiplier = 1.0
            if abs(portfolio_return) > 0.1:  # 10% threshold
                var_multiplier = 1.5  # Assume VaR increases by 50% in stress
            elif abs(portfolio_return) > 0.05:  # 5% threshold
                var_multiplier = 1.2  # Assume VaR increases by 20% in stress
            
            baseline_var = baseline_value * 0.05  # Assume 5% baseline VaR
            stressed_var = baseline_var * var_multiplier
            
            return {
                'portfolio_return': portfolio_return,
                'var_change': stressed_var - baseline_var,
                'var_multiplier': var_multiplier,
                'concentration_risk': self._calculate_concentration_risk(portfolio),
                'liquidity_risk_score': self._calculate_liquidity_risk_score(portfolio, scenario)
            }
            
        except Exception as e:
            logger.error(f"Error calculating stress risk metrics: {str(e)}")
            return {}
    
    def _calculate_concentration_risk(self, portfolio: List[PortfolioPosition]) -> float:
        """Calculate portfolio concentration risk score"""
        total_value = sum(pos.quantity * pos.current_price for pos in portfolio)
        
        if total_value == 0:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index for concentration
        weights = [(pos.quantity * pos.current_price / total_value) ** 2 for pos in portfolio]
        hhi = sum(weights)
        
        # Normalize to 0-1 scale (1 = maximum concentration)
        return hhi
    
    def _calculate_liquidity_risk_score(
        self,
        portfolio: List[PortfolioPosition],
        scenario: StressScenario
    ) -> float:
        """Calculate liquidity risk score under stress"""
        liquidity_stress = scenario.risk_factors.get("liquidity", 0.0)
        
        # Simple liquidity scoring based on asset types
        liquidity_scores = {
            "cash": 1.0,
            "government_bond": 0.9,
            "corporate_bond": 0.7,
            "equity": 0.6,
            "commodity": 0.4,
            "real_estate": 0.2,
            "private_equity": 0.1
        }
        
        total_value = sum(pos.quantity * pos.current_price for pos in portfolio)
        if total_value == 0:
            return 0.0
        
        weighted_liquidity = 0.0
        for pos in portfolio:
            position_value = pos.quantity * pos.current_price
            weight = position_value / total_value
            asset_liquidity = liquidity_scores.get(pos.asset_type, 0.5)
            
            # Apply stress factor
            stressed_liquidity = asset_liquidity * (1 + liquidity_stress)
            weighted_liquidity += weight * max(0.0, min(1.0, stressed_liquidity))
        
        return 1.0 - weighted_liquidity  # Return risk score (higher = more risky)
    
    def _check_breach_indicators(
        self,
        portfolio_loss_pct: float,
        risk_metrics: Dict[str, float]
    ) -> Dict[str, bool]:
        """Check if stress test results breach risk limits"""
        breaches = {}
        
        # Portfolio loss thresholds
        breaches['severe_loss_breach'] = portfolio_loss_pct < -20.0  # 20% loss threshold
        breaches['extreme_loss_breach'] = portfolio_loss_pct < -30.0  # 30% loss threshold
        
        # Concentration risk threshold
        concentration_risk = risk_metrics.get('concentration_risk', 0.0)
        breaches['concentration_breach'] = concentration_risk > 0.25  # 25% concentration limit
        
        # Liquidity risk threshold
        liquidity_risk = risk_metrics.get('liquidity_risk_score', 0.0)
        breaches['liquidity_breach'] = liquidity_risk > 0.5  # 50% liquidity risk limit
        
        return breaches
    
    def get_scenario_library(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all available stress scenarios"""
        library = {}
        for scenario_id, scenario in self.scenarios.items():
            library[scenario_id] = {
                'name': scenario.name,
                'description': scenario.description,
                'type': scenario.stress_type.value,
                'severity': scenario.severity.value,
                'probability': scenario.probability,
                'time_horizon_days': scenario.time_horizon_days,
                'risk_factors': list(scenario.risk_factors.keys())
            }
        return library
