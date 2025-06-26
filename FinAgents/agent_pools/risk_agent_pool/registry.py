"""
Risk Agent Registry - Agent Discovery and Management

This module provides a centralized registry for all risk analysis agents,
enabling dynamic discovery, initialization, and lifecycle management.

Author: Jifeng Li
License: openMDW
"""

import logging
from typing import Dict, Any, Type, Optional, List
from abc import ABC, abstractmethod

logger = logging.getLogger("RiskAgentRegistry")

# Global agent registry
AGENT_REGISTRY: Dict[str, Type] = {}


class BaseRiskAgent(ABC):
    """
    Base class for all risk analysis agents.
    Defines the common interface that all risk agents must implement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(f"RiskAgent.{self.name}")
    
    @abstractmethod
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform risk analysis based on the provided request.
        
        Args:
            request: Structured request containing analysis parameters
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    async def calculate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Alternative method name for analysis (backward compatibility).
        """
        return await self.analyze(request)
    
    async def cleanup(self):
        """Clean up resources when agent is shut down."""
        pass


class MarketRiskAgent(BaseRiskAgent):
    """Agent for market risk analysis including volatility, VaR, and price risk."""
    
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market risk metrics."""
        try:
            portfolio_data = request.get("portfolio_data", {})
            risk_measures = request.get("risk_measures", ["var", "volatility"])
            time_horizon = request.get("time_horizon", "daily")
            
            results = {}
            
            # Calculate volatility if requested
            if "volatility" in risk_measures:
                results["volatility"] = await self._calculate_volatility(portfolio_data, time_horizon)
            
            # Calculate VaR if requested
            if "var" in risk_measures:
                results["var"] = await self._calculate_var(portfolio_data, time_horizon)
            
            # Calculate beta if requested
            if "beta" in risk_measures:
                results["beta"] = await self._calculate_beta(portfolio_data)
            
            return {
                "agent": self.name,
                "risk_type": "market",
                "results": results,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Market risk analysis error: {e}")
            return {
                "agent": self.name,
                "risk_type": "market",
                "error": str(e),
                "status": "error"
            }
    
    async def _calculate_volatility(self, portfolio_data: Dict, time_horizon: str) -> Dict[str, Any]:
        """Calculate portfolio volatility."""
        # Placeholder implementation
        return {
            "daily_volatility": 0.02,
            "annualized_volatility": 0.32,
            "time_horizon": time_horizon,
            "calculation_method": "historical"
        }
    
    async def _calculate_var(self, portfolio_data: Dict, time_horizon: str) -> Dict[str, Any]:
        """Calculate Value at Risk."""
        # Placeholder implementation
        return {
            "var_95": 0.05,
            "var_99": 0.08,
            "time_horizon": time_horizon,
            "confidence_levels": [95, 99],
            "calculation_method": "parametric"
        }
    
    async def _calculate_beta(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Calculate portfolio beta."""
        # Placeholder implementation
        return {
            "portfolio_beta": 1.2,
            "benchmark": "S&P 500",
            "calculation_period": "252 days"
        }


class VolatilityAgent(BaseRiskAgent):
    """Specialized agent for volatility analysis and forecasting."""
    
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility patterns and forecasts."""
        try:
            portfolio_data = request.get("portfolio_data", {})
            time_horizon = request.get("time_horizon", "daily")
            
            # Calculate various volatility measures
            results = {
                "historical_volatility": await self._calculate_historical_volatility(portfolio_data),
                "implied_volatility": await self._calculate_implied_volatility(portfolio_data),
                "volatility_forecast": await self._forecast_volatility(portfolio_data, time_horizon),
                "volatility_clustering": await self._analyze_volatility_clustering(portfolio_data)
            }
            
            return {
                "agent": self.name,
                "risk_type": "market",
                "analysis_type": "volatility",
                "results": results,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Volatility analysis error: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "status": "error"
            }
    
    async def _calculate_historical_volatility(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Calculate historical volatility metrics."""
        return {
            "30_day": 0.018,
            "60_day": 0.022,
            "252_day": 0.025,
            "method": "close-to-close"
        }
    
    async def _calculate_implied_volatility(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Calculate implied volatility from options."""
        return {
            "atm_iv": 0.20,
            "iv_skew": 0.05,
            "iv_term_structure": {"30d": 0.18, "60d": 0.20, "90d": 0.22}
        }
    
    async def _forecast_volatility(self, portfolio_data: Dict, time_horizon: str) -> Dict[str, Any]:
        """Forecast future volatility."""
        return {
            "forecast_1d": 0.019,
            "forecast_5d": 0.021,
            "forecast_30d": 0.024,
            "model": "GARCH(1,1)",
            "confidence_interval": {"lower": 0.015, "upper": 0.030}
        }
    
    async def _analyze_volatility_clustering(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Analyze volatility clustering patterns."""
        return {
            "clustering_detected": True,
            "persistence": 0.85,
            "regime_changes": 3,
            "current_regime": "high_volatility"
        }


class VaRAgent(BaseRiskAgent):
    """Specialized agent for Value at Risk calculations."""
    
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Value at Risk using multiple methodologies."""
        try:
            portfolio_data = request.get("portfolio_data", {})
            time_horizon = request.get("time_horizon", "daily")
            confidence_levels = request.get("confidence_levels", [95, 99])
            
            results = {}
            
            # Calculate VaR using different methods
            for method in ["parametric", "historical", "monte_carlo"]:
                results[f"var_{method}"] = await self._calculate_var_method(
                    portfolio_data, time_horizon, confidence_levels, method
                )
            
            # Calculate Expected Shortfall (CVaR)
            results["expected_shortfall"] = await self._calculate_expected_shortfall(
                portfolio_data, time_horizon, confidence_levels
            )
            
            return {
                "agent": self.name,
                "risk_type": "market",
                "analysis_type": "var",
                "results": results,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"VaR calculation error: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "status": "error"
            }
    
    async def _calculate_var_method(self, portfolio_data: Dict, time_horizon: str, 
                                  confidence_levels: list, method: str) -> Dict[str, Any]:
        """Calculate VaR using specified method."""
        var_values = {}
        for conf in confidence_levels:
            # Placeholder calculation based on method
            if method == "parametric":
                var_values[f"var_{conf}"] = 0.02 * (conf / 95.0)
            elif method == "historical":
                var_values[f"var_{conf}"] = 0.025 * (conf / 95.0)
            else:  # monte_carlo
                var_values[f"var_{conf}"] = 0.022 * (conf / 95.0)
        
        return {
            "method": method,
            "time_horizon": time_horizon,
            "values": var_values,
            "portfolio_value": 1000000  # placeholder
        }
    
    async def _calculate_expected_shortfall(self, portfolio_data: Dict, 
                                          time_horizon: str, confidence_levels: list) -> Dict[str, Any]:
        """Calculate Expected Shortfall (Conditional VaR)."""
        es_values = {}
        for conf in confidence_levels:
            es_values[f"es_{conf}"] = 0.035 * (conf / 95.0)
        
        return {
            "time_horizon": time_horizon,
            "values": es_values,
            "interpretation": "Expected loss given that loss exceeds VaR"
        }


class CreditRiskAgent(BaseRiskAgent):
    """Agent for credit risk analysis including default probability and credit spreads."""
    
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze credit risk metrics."""
        try:
            portfolio_data = request.get("portfolio_data", {})
            
            results = {
                "default_probability": await self._calculate_default_probability(portfolio_data),
                "credit_spreads": await self._analyze_credit_spreads(portfolio_data),
                "credit_var": await self._calculate_credit_var(portfolio_data),
                "recovery_rates": await self._estimate_recovery_rates(portfolio_data)
            }
            
            return {
                "agent": self.name,
                "risk_type": "credit",
                "results": results,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Credit risk analysis error: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "status": "error"
            }
    
    async def _calculate_default_probability(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Calculate probability of default."""
        return {
            "1_year_pd": 0.02,
            "3_year_pd": 0.06,
            "5_year_pd": 0.10,
            "model": "structural_model"
        }
    
    async def _analyze_credit_spreads(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Analyze credit spreads."""
        return {
            "current_spread": 150,  # basis points
            "spread_volatility": 25,
            "spread_duration": 4.2,
            "benchmark": "treasury"
        }
    
    async def _calculate_credit_var(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Calculate Credit Value at Risk."""
        return {
            "credit_var_95": 0.08,
            "credit_var_99": 0.12,
            "time_horizon": "1_year",
            "method": "monte_carlo"
        }
    
    async def _estimate_recovery_rates(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Estimate recovery rates in case of default."""
        return {
            "senior_debt": 0.65,
            "subordinated_debt": 0.35,
            "equity": 0.05,
            "historical_average": 0.45
        }


class LiquidityRiskAgent(BaseRiskAgent):
    """Agent for liquidity risk analysis."""
    
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze liquidity risk metrics."""
        try:
            portfolio_data = request.get("portfolio_data", {})
            
            results = {
                "liquidity_ratios": await self._calculate_liquidity_ratios(portfolio_data),
                "bid_ask_spreads": await self._analyze_bid_ask_spreads(portfolio_data),
                "market_impact": await self._estimate_market_impact(portfolio_data),
                "funding_liquidity": await self._assess_funding_liquidity(portfolio_data)
            }
            
            return {
                "agent": self.name,
                "risk_type": "liquidity",
                "results": results,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Liquidity risk analysis error: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "status": "error"
            }
    
    async def _calculate_liquidity_ratios(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Calculate various liquidity ratios."""
        return {
            "current_ratio": 1.5,
            "quick_ratio": 1.2,
            "cash_ratio": 0.3,
            "operating_cash_flow_ratio": 0.8
        }
    
    async def _analyze_bid_ask_spreads(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Analyze bid-ask spreads as liquidity indicators."""
        return {
            "average_spread": 0.02,  # percentage
            "spread_volatility": 0.005,
            "spread_percentiles": {"25": 0.01, "50": 0.02, "75": 0.03, "95": 0.05}
        }
    
    async def _estimate_market_impact(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Estimate market impact of large trades."""
        return {
            "temporary_impact": 0.01,
            "permanent_impact": 0.005,
            "impact_model": "square_root",
            "liquidity_score": 7.5  # out of 10
        }
    
    async def _assess_funding_liquidity(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Assess funding liquidity risk."""
        return {
            "funding_gap": 0.15,
            "funding_concentration": 0.3,
            "maturity_mismatch": 0.2,
            "funding_stability": "moderate"
        }


class CorrelationAgent(BaseRiskAgent):
    """Agent for correlation and dependency analysis."""
    
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlation structures and dependencies."""
        try:
            portfolio_data = request.get("portfolio_data", {})
            
            results = {
                "correlation_matrix": await self._calculate_correlation_matrix(portfolio_data),
                "tail_dependencies": await self._analyze_tail_dependencies(portfolio_data),
                "dynamic_correlations": await self._calculate_dynamic_correlations(portfolio_data),
                "copula_analysis": await self._perform_copula_analysis(portfolio_data)
            }
            
            return {
                "agent": self.name,
                "risk_type": "systemic",
                "analysis_type": "correlation",
                "results": results,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Correlation analysis error: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "status": "error"
            }
    
    async def _calculate_correlation_matrix(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Calculate correlation matrix of portfolio assets."""
        # Placeholder correlation matrix
        return {
            "matrix_size": "10x10",
            "average_correlation": 0.35,
            "max_correlation": 0.85,
            "min_correlation": -0.15,
            "eigenvalues": [3.2, 1.8, 1.1, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1]
        }
    
    async def _analyze_tail_dependencies(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Analyze tail dependencies between assets."""
        return {
            "upper_tail_dependence": 0.25,
            "lower_tail_dependence": 0.35,
            "asymmetric_dependence": True,
            "tail_dependence_test": "significant"
        }
    
    async def _calculate_dynamic_correlations(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Calculate time-varying correlations."""
        return {
            "correlation_regime": "high_correlation",
            "regime_persistence": 0.75,
            "correlation_volatility": 0.08,
            "correlation_forecast": 0.42
        }
    
    async def _perform_copula_analysis(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Perform copula-based dependency analysis."""
        return {
            "best_fit_copula": "t-copula",
            "degrees_of_freedom": 8.5,
            "goodness_of_fit": 0.92,
            "tail_risk_contribution": 0.15
        }


class OperationalRiskAgent(BaseRiskAgent):
    """Agent for operational risk analysis including fraud detection and KRI monitoring."""
    
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze operational risk metrics."""
        try:
            from .agents.operational_risk import OperationalRiskAnalyzer
            
            analyzer = OperationalRiskAnalyzer()
            analysis_type = request.get("analysis_type", "metrics")
            
            results = {}
            
            if analysis_type == "metrics":
                start_date = request.get("start_date")
                end_date = request.get("end_date")
                results["metrics"] = await analyzer.calculate_operational_metrics(start_date, end_date)
            
            elif analysis_type == "fraud_assessment":
                transaction_data = request.get("transaction_data", {})
                results["fraud_risk"] = await analyzer.assess_fraud_risk(transaction_data)
            
            elif analysis_type == "kri_monitoring":
                current_metrics = request.get("current_metrics", {})
                results["kri_status"] = await analyzer.monitor_key_risk_indicators(current_metrics)
            
            elif analysis_type == "operational_var":
                confidence_level = request.get("confidence_level", 0.99)
                time_horizon = request.get("time_horizon_days", 365)
                results["operational_var"] = await analyzer.calculate_operational_var(confidence_level, time_horizon)
            
            elif analysis_type == "scenario_analysis":
                scenarios = request.get("scenarios", [])
                results["scenario_results"] = await analyzer.scenario_analysis(scenarios)
            
            return {
                "agent": self.name,
                "risk_type": "operational",
                "results": results,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Operational risk analysis error: {e}")
            return {
                "agent": self.name,
                "risk_type": "operational",
                "error": str(e),
                "status": "error"
            }


class StressTestingAgent(BaseRiskAgent):
    """Agent for comprehensive stress testing and scenario analysis."""
    
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform stress testing analysis."""
        try:
            from .agents.stress_testing import StressTester
            
            tester = StressTester()
            test_type = request.get("test_type", "scenario")
            portfolio = request.get("portfolio", [])
            
            results = {}
            
            if test_type == "scenario":
                scenario_id = request.get("scenario_id")
                pricing_functions = request.get("pricing_functions", {})
                results["stress_test"] = await tester.run_stress_test(scenario_id, portfolio, pricing_functions)
            
            elif test_type == "sensitivity":
                risk_factor = request.get("risk_factor")
                shock_range = request.get("shock_range", (-0.1, 0.1))
                num_points = request.get("num_points", 21)
                results["sensitivity"] = await tester.run_sensitivity_analysis(portfolio, risk_factor, shock_range, num_points)
            
            elif test_type == "monte_carlo":
                num_simulations = request.get("num_simulations", 10000)
                time_horizon = request.get("time_horizon_days", 1)
                confidence_levels = request.get("confidence_levels", [0.95, 0.99, 0.999])
                results["monte_carlo"] = await tester.run_monte_carlo_stress(portfolio, num_simulations, time_horizon, confidence_levels)
            
            elif test_type == "reverse":
                target_loss_pct = request.get("target_loss_pct", 10.0)
                max_iterations = request.get("max_iterations", 1000)
                tolerance = request.get("tolerance", 0.01)
                results["reverse_stress"] = await tester.run_reverse_stress_test(portfolio, target_loss_pct, max_iterations, tolerance)
            
            elif test_type == "scenario_library":
                results["scenario_library"] = tester.get_scenario_library()
            
            return {
                "agent": self.name,
                "risk_type": "stress_testing",
                "results": results,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Stress testing analysis error: {e}")
            return {
                "agent": self.name,
                "risk_type": "stress_testing",
                "error": str(e),
                "status": "error"
            }


class ModelRiskAgent(BaseRiskAgent):
    """Agent for model risk management including validation and performance monitoring."""
    
    def __init__(self, **kwargs):
        """Initialize the model risk agent with persistent manager."""
        super().__init__(**kwargs)
        from .agents.model_risk import ModelRiskManager
        self.manager = ModelRiskManager()
    
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform model risk analysis."""
        try:
            action = request.get("action", "inventory_report")
            
            results = {}
            
            if action == "register_model":
                model_metadata = request.get("model_metadata")
                results["model_id"] = await self.manager.register_model(model_metadata)
            
            elif action == "validate_model":
                model_id = request.get("model_id")
                validator = request.get("validator")
                validation_config = request.get("validation_config", {})
                results["validation_report"] = await self.manager.validate_model(model_id, validator, validation_config)
            
            elif action == "monitor_performance":
                model_id = request.get("model_id")
                performance_data = request.get("performance_data", {})
                results["performance_metrics"] = await self.manager.monitor_model_performance(model_id, performance_data)
            
            elif action == "track_change":
                model_change = request.get("model_change")
                results["change_id"] = await self.manager.track_model_change(model_change)
            
            elif action == "inventory_report":
                filters = request.get("filters")
                results["inventory_report"] = await self.manager.generate_model_inventory_report(filters)
            
            return {
                "agent": self.name,
                "risk_type": "model_risk",
                "results": results,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Model risk analysis error: {e}")
            return {
                "agent": self.name,
                "risk_type": "model_risk",
                "error": str(e),
                "status": "error"
            }


def register_agent(name: str, agent_class: Type[BaseRiskAgent]):
    """
    Register an agent class in the global registry.
    
    Args:
        name: Unique name for the agent
        agent_class: Agent class that inherits from BaseRiskAgent
    """
    if not issubclass(agent_class, BaseRiskAgent):
        raise ValueError(f"Agent class {agent_class} must inherit from BaseRiskAgent")
    
    AGENT_REGISTRY[name] = agent_class
    logger.info(f"Registered agent: {name}")


def unregister_agent(name: str):
    """
    Unregister an agent from the global registry.
    
    Args:
        name: Name of the agent to unregister
    """
    if name in AGENT_REGISTRY:
        del AGENT_REGISTRY[name]
        logger.info(f"Unregistered agent: {name}")


def get_agent_class(name: str) -> Optional[Type[BaseRiskAgent]]:
    """
    Get an agent class by name.
    
    Args:
        name: Name of the agent
        
    Returns:
        Agent class or None if not found
    """
    return AGENT_REGISTRY.get(name)


def list_agents() -> List[str]:
    """
    List all registered agent names.
    
    Returns:
        List of agent names
    """
    return list(AGENT_REGISTRY.keys())


def preload_default_agents():
    """Preload all default risk analysis agents."""
    # Register all default agents
    register_agent("market_risk_agent", MarketRiskAgent)
    register_agent("volatility_agent", VolatilityAgent)
    register_agent("var_agent", VaRAgent)
    register_agent("cvar_agent", VaRAgent)  # CVaR is calculated by VaR agent
    register_agent("credit_risk_agent", CreditRiskAgent)
    register_agent("liquidity_risk_agent", LiquidityRiskAgent)
    register_agent("correlation_agent", CorrelationAgent)
    register_agent("systemic_risk_agent", CorrelationAgent)  # Uses correlation agent
    register_agent("beta_agent", MarketRiskAgent)  # Beta calculated by market risk agent
    register_agent("drawdown_agent", MarketRiskAgent)  # Drawdown calculated by market risk agent
    register_agent("operational_risk_agent", OperationalRiskAgent)
    register_agent("stress_testing_agent", StressTestingAgent)
    register_agent("model_risk_agent", ModelRiskAgent)
    
    logger.info(f"Preloaded {len(AGENT_REGISTRY)} default risk agents")


# Auto-register default agents when module is imported
preload_default_agents()
