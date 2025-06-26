"""
Risk-Adjusted Transaction Cost Analyzer

This agent analyzes transaction costs in the context of portfolio risk,
providing risk-adjusted cost metrics and recommendations.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum

from ...schema.cost_models import TransactionCostBreakdown, CostEstimate
from ...schema.execution_schema import TradeExecution


logger = logging.getLogger(__name__)


class RiskMetric(Enum):
    """Different risk metrics for analysis."""
    VALUE_AT_RISK = "value_at_risk"
    EXPECTED_SHORTFALL = "expected_shortfall"
    VOLATILITY = "volatility"
    BETA = "beta"
    SHARPE_RATIO = "sharpe_ratio"
    INFORMATION_RATIO = "information_ratio"
    MAX_DRAWDOWN = "max_drawdown"


@dataclass
class RiskAdjustedMetrics:
    """Risk-adjusted transaction cost metrics."""
    cost_per_unit_risk: float
    risk_adjusted_cost_bps: float
    cost_volatility_ratio: float
    risk_contribution_pct: float
    sharpe_impact: float
    var_impact_bps: float
    expected_shortfall_impact_bps: float
    confidence_level: float


@dataclass
class RiskCostAnalysisResult:
    """Result of risk-adjusted cost analysis."""
    analysis_id: str
    timestamp: datetime
    total_portfolio_value: float
    risk_adjusted_metrics: RiskAdjustedMetrics
    cost_risk_attribution: Dict[str, float]
    risk_budgeting: Dict[str, float]
    optimization_recommendations: List[str]
    stress_test_results: Dict[str, Dict[str, float]]
    scenario_analysis: Dict[str, Dict[str, float]]


class RiskCostAnalyzer:
    """
    Advanced risk-adjusted transaction cost analyzer.
    
    This analyzer provides comprehensive risk-adjusted cost analysis including:
    - Risk-adjusted cost metrics calculation
    - Cost attribution in risk space
    - Portfolio impact analysis
    - Risk budgeting for transactions
    - Stress testing and scenario analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the risk-cost analyzer.
        
        Args:
            config: Configuration parameters for the analyzer
        """
        self.config = config or {}
        self.risk_models = {}
        self.benchmark_data = {}
        
        # Risk parameters
        self.confidence_levels = self.config.get('confidence_levels', [0.95, 0.99])
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% annually
        self.lookback_days = self.config.get('lookback_days', 252)  # 1 year
        
        # Analysis parameters
        self.stress_scenarios = self.config.get('stress_scenarios', [
            'market_crash', 'volatility_spike', 'liquidity_crisis', 'sector_rotation'
        ])
        
    async def analyze_risk_adjusted_costs(
        self, 
        executions: List[TradeExecution],
        portfolio_positions: Optional[Dict[str, Dict[str, float]]] = None,
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
        analysis_id: Optional[str] = None
    ) -> RiskCostAnalysisResult:
        """
        Perform comprehensive risk-adjusted transaction cost analysis.
        
        Args:
            executions: List of trade executions to analyze
            portfolio_positions: Current portfolio positions {symbol: {position: float, value: float}}
            market_data: Historical market data for risk calculations
            analysis_id: Unique identifier for this analysis
            
        Returns:
            Comprehensive risk-adjusted cost analysis results
        """
        try:
            analysis_id = analysis_id or f"risk_cost_analysis_{datetime.utcnow().isoformat()}"
            logger.info(f"Starting risk-adjusted cost analysis {analysis_id}")
            
            # Prepare data
            execution_data = self._prepare_execution_data(executions)
            portfolio_data = portfolio_positions or self._estimate_portfolio_positions(executions)
            
            # Calculate portfolio risk metrics
            portfolio_risk = await self._calculate_portfolio_risk(portfolio_data, market_data)
            
            # Calculate risk-adjusted cost metrics
            risk_adjusted_metrics = await self._calculate_risk_adjusted_metrics(
                execution_data, portfolio_risk, market_data
            )
            
            # Perform cost-risk attribution
            cost_risk_attribution = await self._perform_cost_risk_attribution(
                execution_data, portfolio_risk
            )
            
            # Calculate risk budgeting
            risk_budgeting = await self._calculate_risk_budgeting(
                execution_data, portfolio_data, portfolio_risk
            )
            
            # Generate optimization recommendations
            recommendations = await self._generate_risk_optimization_recommendations(
                risk_adjusted_metrics, cost_risk_attribution, risk_budgeting
            )
            
            # Perform stress testing
            stress_results = await self._perform_stress_testing(
                execution_data, portfolio_data, market_data
            )
            
            # Scenario analysis
            scenario_results = await self._perform_scenario_analysis(
                execution_data, portfolio_data, market_data
            )
            
            # Calculate total portfolio value
            total_portfolio_value = sum(
                pos_data.get('value', 0) for pos_data in portfolio_data.values()
            )
            
            # Compile results
            result = RiskCostAnalysisResult(
                analysis_id=analysis_id,
                timestamp=datetime.utcnow(),
                total_portfolio_value=total_portfolio_value,
                risk_adjusted_metrics=risk_adjusted_metrics,
                cost_risk_attribution=cost_risk_attribution,
                risk_budgeting=risk_budgeting,
                optimization_recommendations=recommendations,
                stress_test_results=stress_results,
                scenario_analysis=scenario_results
            )
            
            logger.info(f"Risk-adjusted cost analysis {analysis_id} completed")
            return result
            
        except Exception as e:
            logger.error(f"Error in risk-adjusted cost analysis: {str(e)}")
            raise
    
    def _prepare_execution_data(self, executions: List[TradeExecution]) -> pd.DataFrame:
        """Prepare execution data for risk analysis."""
        
        data = []
        for execution in executions:
            market_value = execution.quantity * execution.executed_price
            cost_bps = (execution.total_cost / market_value) * 10000 if market_value > 0 else 0
            
            data.append({
                'execution_id': execution.execution_id,
                'symbol': execution.symbol,
                'side': execution.side,
                'quantity': execution.quantity,
                'executed_price': execution.executed_price,
                'market_value': market_value,
                'total_cost': execution.total_cost,
                'cost_bps': cost_bps,
                'execution_time': execution.execution_time
            })
        
        return pd.DataFrame(data)
    
    def _estimate_portfolio_positions(self, executions: List[TradeExecution]) -> Dict[str, Dict[str, float]]:
        """Estimate portfolio positions from executions (simplified)."""
        
        positions = {}
        for execution in executions:
            symbol = execution.symbol
            market_value = execution.quantity * execution.executed_price
            
            if symbol not in positions:
                positions[symbol] = {'position': 0, 'value': 0}
            
            if execution.side.upper() == 'BUY':
                positions[symbol]['position'] += execution.quantity
                positions[symbol]['value'] += market_value
            else:
                positions[symbol]['position'] -= execution.quantity
                positions[symbol]['value'] -= market_value
        
        return positions
    
    async def _calculate_portfolio_risk(
        self, 
        portfolio_data: Dict[str, Dict[str, float]],
        market_data: Optional[Dict[str, pd.DataFrame]]
    ) -> Dict[str, float]:
        """Calculate portfolio risk metrics."""
        
        risk_metrics = {
            'portfolio_volatility': 0.15,  # 15% annual volatility (estimated)
            'portfolio_beta': 1.0,         # Market beta
            'portfolio_var_95': 0.0,       # 95% VaR
            'portfolio_var_99': 0.0,       # 99% VaR
            'expected_shortfall_95': 0.0,  # 95% Expected Shortfall
            'max_drawdown': 0.0,           # Maximum drawdown
            'correlation_matrix': {},      # Symbol correlations
            'individual_volatilities': {}   # Individual symbol volatilities
        }
        
        # Calculate individual symbol metrics
        total_portfolio_value = sum(pos_data.get('value', 0) for pos_data in portfolio_data.values())
        
        for symbol, pos_data in portfolio_data.items():
            weight = pos_data.get('value', 0) / total_portfolio_value if total_portfolio_value > 0 else 0
            
            # Estimate individual volatility (simplified)
            individual_vol = np.random.normal(0.25, 0.1)  # Random volatility around 25%
            individual_vol = max(0.05, min(0.80, individual_vol))  # Clip to reasonable range
            
            risk_metrics['individual_volatilities'][symbol] = individual_vol
        
        # Estimate portfolio-level metrics
        if len(portfolio_data) > 1:
            # Simplified portfolio volatility calculation
            avg_volatility = np.mean(list(risk_metrics['individual_volatilities'].values()))
            avg_correlation = 0.3  # Assume 30% average correlation
            diversification_benefit = 1 - avg_correlation * (len(portfolio_data) - 1) / len(portfolio_data)
            risk_metrics['portfolio_volatility'] = avg_volatility * np.sqrt(diversification_benefit)
        elif len(portfolio_data) == 1:
            risk_metrics['portfolio_volatility'] = list(risk_metrics['individual_volatilities'].values())[0]
        
        # Calculate VaR and Expected Shortfall (simplified)
        daily_volatility = risk_metrics['portfolio_volatility'] / np.sqrt(252)
        
        # 95% VaR (1-day)
        risk_metrics['portfolio_var_95'] = daily_volatility * 1.645 * total_portfolio_value
        
        # 99% VaR (1-day)
        risk_metrics['portfolio_var_99'] = daily_volatility * 2.326 * total_portfolio_value
        
        # Expected Shortfall (simplified)
        risk_metrics['expected_shortfall_95'] = risk_metrics['portfolio_var_95'] * 1.3
        
        # Maximum drawdown (estimated)
        risk_metrics['max_drawdown'] = risk_metrics['portfolio_volatility'] * 2.5
        
        return risk_metrics
    
    async def _calculate_risk_adjusted_metrics(
        self, 
        execution_data: pd.DataFrame,
        portfolio_risk: Dict[str, float],
        market_data: Optional[Dict[str, pd.DataFrame]]
    ) -> RiskAdjustedMetrics:
        """Calculate risk-adjusted transaction cost metrics."""
        
        if len(execution_data) == 0:
            return RiskAdjustedMetrics(
                cost_per_unit_risk=0.0,
                risk_adjusted_cost_bps=0.0,
                cost_volatility_ratio=0.0,
                risk_contribution_pct=0.0,
                sharpe_impact=0.0,
                var_impact_bps=0.0,
                expected_shortfall_impact_bps=0.0,
                confidence_level=0.8
            )
        
        # Basic cost metrics
        total_cost = execution_data['total_cost'].sum()
        total_market_value = execution_data['market_value'].sum()
        avg_cost_bps = execution_data['cost_bps'].mean()
        cost_volatility = execution_data['cost_bps'].std()
        
        # Portfolio risk metrics
        portfolio_volatility = portfolio_risk.get('portfolio_volatility', 0.15)
        portfolio_var_95 = portfolio_risk.get('portfolio_var_95', 0.0)
        expected_shortfall_95 = portfolio_risk.get('expected_shortfall_95', 0.0)
        
        # Calculate risk-adjusted metrics
        
        # Cost per unit of risk
        if portfolio_volatility > 0:
            cost_per_unit_risk = avg_cost_bps / (portfolio_volatility * 10000)  # Cost bps per vol bps
        else:
            cost_per_unit_risk = 0.0
        
        # Risk-adjusted cost (cost scaled by portfolio risk)
        risk_adjusted_cost_bps = avg_cost_bps * (1 + portfolio_volatility)
        
        # Cost-volatility ratio
        if cost_volatility > 0:
            cost_volatility_ratio = avg_cost_bps / cost_volatility
        else:
            cost_volatility_ratio = float('inf')
        
        # Risk contribution of transaction costs
        if total_market_value > 0:
            cost_as_pct_of_portfolio = total_cost / total_market_value
            # Estimate risk contribution (simplified)
            risk_contribution_pct = cost_as_pct_of_portfolio * 100 * 2  # Rough multiplier
        else:
            risk_contribution_pct = 0.0
        
        # Sharpe ratio impact (simplified)
        # Transaction costs reduce returns, impacting Sharpe ratio
        annual_cost_drag = avg_cost_bps / 10000  # Convert to decimal
        sharpe_impact = -annual_cost_drag / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        # VaR impact
        var_impact_bps = (total_cost / portfolio_var_95) * 10000 if portfolio_var_95 > 0 else 0.0
        
        # Expected Shortfall impact
        es_impact_bps = (total_cost / expected_shortfall_95) * 10000 if expected_shortfall_95 > 0 else 0.0
        
        # Confidence level (based on data quality and model assumptions)
        confidence_level = min(0.95, 0.7 + len(execution_data) * 0.01)  # Higher confidence with more data
        
        return RiskAdjustedMetrics(
            cost_per_unit_risk=cost_per_unit_risk,
            risk_adjusted_cost_bps=risk_adjusted_cost_bps,
            cost_volatility_ratio=cost_volatility_ratio,
            risk_contribution_pct=min(100.0, risk_contribution_pct),
            sharpe_impact=sharpe_impact,
            var_impact_bps=var_impact_bps,
            expected_shortfall_impact_bps=es_impact_bps,
            confidence_level=confidence_level
        )
    
    async def _perform_cost_risk_attribution(
        self, 
        execution_data: pd.DataFrame,
        portfolio_risk: Dict[str, float]
    ) -> Dict[str, float]:
        """Perform cost attribution in risk space."""
        
        attribution = {
            'systematic_risk_cost': 0.0,
            'idiosyncratic_risk_cost': 0.0,
            'timing_risk_cost': 0.0,
            'liquidity_risk_cost': 0.0,
            'execution_risk_cost': 0.0
        }
        
        if len(execution_data) == 0:
            return attribution
        
        total_cost = execution_data['total_cost'].sum()
        portfolio_beta = portfolio_risk.get('portfolio_beta', 1.0)
        
        # Systematic risk cost (market beta related)
        attribution['systematic_risk_cost'] = total_cost * portfolio_beta * 0.6
        
        # Idiosyncratic risk cost (stock-specific)
        attribution['idiosyncratic_risk_cost'] = total_cost * (1 - portfolio_beta * 0.6)
        
        # Timing risk cost (based on execution time spread)
        if len(execution_data) > 1:
            time_spread_hours = (execution_data['execution_time'].max() - 
                               execution_data['execution_time'].min()).total_seconds() / 3600
            timing_risk_factor = min(0.3, time_spread_hours / 24)  # Max 30% for full day
            attribution['timing_risk_cost'] = total_cost * timing_risk_factor
        
        # Liquidity risk cost (estimated from cost volatility)
        cost_volatility = execution_data['cost_bps'].std()
        if cost_volatility > 0:
            liquidity_risk_factor = min(0.4, cost_volatility / 20)  # Normalize by 20 bps
            attribution['liquidity_risk_cost'] = total_cost * liquidity_risk_factor
        
        # Execution risk cost (residual)
        assigned_cost = (attribution['systematic_risk_cost'] + 
                        attribution['idiosyncratic_risk_cost'] + 
                        attribution['timing_risk_cost'] + 
                        attribution['liquidity_risk_cost'])
        attribution['execution_risk_cost'] = max(0, total_cost - assigned_cost)
        
        return attribution
    
    async def _calculate_risk_budgeting(
        self, 
        execution_data: pd.DataFrame,
        portfolio_data: Dict[str, Dict[str, float]],
        portfolio_risk: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate risk budgeting for transaction costs."""
        
        budgeting = {
            'cost_risk_budget_bps': 0.0,
            'utilized_risk_budget_pct': 0.0,
            'remaining_risk_budget_bps': 0.0,
            'risk_budget_efficiency': 0.0,
            'recommended_adjustments': {}
        }
        
        # Set risk budget (e.g., 2% of portfolio volatility)
        portfolio_volatility = portfolio_risk.get('portfolio_volatility', 0.15)
        cost_risk_budget_bps = portfolio_volatility * 200  # 2% of volatility as bps
        
        # Calculate utilized budget
        if len(execution_data) > 0:
            actual_cost_bps = execution_data['cost_bps'].mean()
            utilized_pct = (actual_cost_bps / cost_risk_budget_bps) * 100 if cost_risk_budget_bps > 0 else 0
        else:
            actual_cost_bps = 0
            utilized_pct = 0
        
        remaining_budget_bps = max(0, cost_risk_budget_bps - actual_cost_bps)
        
        # Calculate efficiency (return per unit of risk budget used)
        if actual_cost_bps > 0:
            # Estimate efficiency based on execution quality
            efficiency = 1.0 / actual_cost_bps  # Higher cost = lower efficiency
        else:
            efficiency = 0.0
        
        budgeting.update({
            'cost_risk_budget_bps': cost_risk_budget_bps,
            'utilized_risk_budget_pct': utilized_pct,
            'remaining_risk_budget_bps': remaining_budget_bps,
            'risk_budget_efficiency': efficiency
        })
        
        # Recommended adjustments by symbol
        for symbol in execution_data['symbol'].unique():
            symbol_data = execution_data[execution_data['symbol'] == symbol]
            symbol_cost = symbol_data['cost_bps'].mean()
            
            if symbol_cost > cost_risk_budget_bps * 1.5:  # 150% of budget
                budgeting['recommended_adjustments'][symbol] = "Reduce position size or improve execution"
            elif symbol_cost < cost_risk_budget_bps * 0.3:  # 30% of budget
                budgeting['recommended_adjustments'][symbol] = "Consider increasing position size"
        
        return budgeting
    
    async def _generate_risk_optimization_recommendations(
        self, 
        risk_metrics: RiskAdjustedMetrics,
        cost_attribution: Dict[str, float],
        risk_budgeting: Dict[str, float]
    ) -> List[str]:
        """Generate risk-based optimization recommendations."""
        
        recommendations = []
        
        # Risk-adjusted cost recommendations
        if risk_metrics.risk_adjusted_cost_bps > 10.0:
            recommendations.append(
                f"Risk-adjusted costs ({risk_metrics.risk_adjusted_cost_bps:.1f} bps) are high - "
                "consider reducing portfolio volatility or improving execution efficiency"
            )
        
        # Cost per unit risk recommendations
        if risk_metrics.cost_per_unit_risk > 0.5:
            recommendations.append(
                f"Cost per unit of risk ({risk_metrics.cost_per_unit_risk:.3f}) is high - "
                "optimize risk-return trade-off in execution strategies"
            )
        
        # Sharpe impact recommendations
        if risk_metrics.sharpe_impact < -0.1:
            recommendations.append(
                f"Transaction costs significantly impact Sharpe ratio ({risk_metrics.sharpe_impact:.3f}) - "
                "prioritize cost reduction to improve risk-adjusted returns"
            )
        
        # Risk budgeting recommendations
        if risk_budgeting.get('utilized_risk_budget_pct', 0) > 80:
            recommendations.append(
                "Over 80% of risk budget utilized - consider more efficient execution strategies"
            )
        elif risk_budgeting.get('utilized_risk_budget_pct', 0) < 30:
            recommendations.append(
                "Risk budget under-utilized - may have opportunity for more aggressive execution"
            )
        
        # Risk attribution recommendations
        total_attribution = sum(cost_attribution.values())
        if total_attribution > 0:
            liquidity_pct = cost_attribution.get('liquidity_risk_cost', 0) / total_attribution
            if liquidity_pct > 0.4:
                recommendations.append(
                    "Liquidity risk dominates cost structure - focus on liquidity optimization"
                )
            
            timing_pct = cost_attribution.get('timing_risk_cost', 0) / total_attribution
            if timing_pct > 0.3:
                recommendations.append(
                    "Timing risk is significant - implement better execution timing strategies"
                )
        
        return recommendations if recommendations else [
            "Risk-adjusted transaction costs are within acceptable ranges"
        ]
    
    async def _perform_stress_testing(
        self, 
        execution_data: pd.DataFrame,
        portfolio_data: Dict[str, Dict[str, float]],
        market_data: Optional[Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, float]]:
        """Perform stress testing on transaction costs."""
        
        stress_results = {}
        
        # Market crash scenario
        stress_results['market_crash'] = {
            'cost_increase_bps': 3.0,  # 3 bps increase in stressed conditions
            'var_impact_multiplier': 2.0,
            'liquidity_impact': 0.4,   # 40% reduction in liquidity
            'probability': 0.05        # 5% probability
        }
        
        # Volatility spike scenario
        stress_results['volatility_spike'] = {
            'cost_increase_bps': 2.0,
            'var_impact_multiplier': 1.5,
            'liquidity_impact': 0.2,
            'probability': 0.15
        }
        
        # Liquidity crisis scenario
        stress_results['liquidity_crisis'] = {
            'cost_increase_bps': 5.0,
            'var_impact_multiplier': 1.2,
            'liquidity_impact': 0.6,
            'probability': 0.02
        }
        
        # Sector rotation scenario
        stress_results['sector_rotation'] = {
            'cost_increase_bps': 1.5,
            'var_impact_multiplier': 1.1,
            'liquidity_impact': 0.1,
            'probability': 0.25
        }
        
        return stress_results
    
    async def _perform_scenario_analysis(
        self, 
        execution_data: pd.DataFrame,
        portfolio_data: Dict[str, Dict[str, float]],
        market_data: Optional[Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, float]]:
        """Perform scenario analysis on risk-adjusted costs."""
        
        scenarios = {}
        
        # Bull market scenario
        scenarios['bull_market'] = {
            'cost_change_bps': -0.5,     # Lower costs due to better liquidity
            'risk_change_pct': -10.0,    # 10% reduction in risk
            'sharpe_impact': 0.02,       # Positive impact on Sharpe
            'scenario_probability': 0.3
        }
        
        # Bear market scenario
        scenarios['bear_market'] = {
            'cost_change_bps': 2.0,      # Higher costs due to stress
            'risk_change_pct': 25.0,     # 25% increase in risk
            'sharpe_impact': -0.05,      # Negative impact on Sharpe
            'scenario_probability': 0.2
        }
        
        # High volatility scenario
        scenarios['high_volatility'] = {
            'cost_change_bps': 1.5,
            'risk_change_pct': 40.0,
            'sharpe_impact': -0.08,
            'scenario_probability': 0.15
        }
        
        # Low volatility scenario
        scenarios['low_volatility'] = {
            'cost_change_bps': -1.0,
            'risk_change_pct': -20.0,
            'sharpe_impact': 0.03,
            'scenario_probability': 0.25
        }
        
        # Base case scenario
        scenarios['base_case'] = {
            'cost_change_bps': 0.0,
            'risk_change_pct': 0.0,
            'sharpe_impact': 0.0,
            'scenario_probability': 0.4
        }
        
        return scenarios


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from datetime import datetime, timedelta
    
    async def test_risk_cost_analyzer():
        """Test the risk-cost analyzer with sample data."""
        
        # Create sample executions
        sample_executions = [
            TradeExecution(
                execution_id="exec_001",
                symbol="AAPL",
                side="BUY",
                quantity=1000,
                executed_price=150.25,
                benchmark_price=150.20,
                execution_time=datetime.utcnow() - timedelta(hours=2),
                venue="NYSE",
                order_type="LIMIT",
                total_cost=45.0
            ),
            TradeExecution(
                execution_id="exec_002",
                symbol="GOOGL",
                side="SELL",
                quantity=500,
                executed_price=2750.80,
                benchmark_price=2751.00,
                execution_time=datetime.utcnow() - timedelta(hours=1),
                venue="NASDAQ",
                order_type="MARKET",
                total_cost=82.5
            ),
            TradeExecution(
                execution_id="exec_003",
                symbol="MSFT",
                side="BUY",
                quantity=750,
                executed_price=340.10,
                benchmark_price=340.05,
                execution_time=datetime.utcnow() - timedelta(minutes=30),
                venue="ARCA",
                order_type="LIMIT",
                total_cost=38.0
            )
        ]
        
        # Create sample portfolio positions
        portfolio_positions = {
            "AAPL": {"position": 5000, "value": 750000},
            "GOOGL": {"position": 200, "value": 550000},
            "MSFT": {"position": 1500, "value": 510000},
            "AMZN": {"position": 300, "value": 450000}
        }
        
        # Initialize analyzer and run analysis
        analyzer = RiskCostAnalyzer()
        result = await analyzer.analyze_risk_adjusted_costs(
            sample_executions, 
            portfolio_positions
        )
        
        print("=== Risk-Adjusted Cost Analysis Results ===")
        print(f"Analysis ID: {result.analysis_id}")
        print(f"Total Portfolio Value: ${result.total_portfolio_value:,.2f}")
        
        print("\n=== Risk-Adjusted Metrics ===")
        metrics = result.risk_adjusted_metrics
        print(f"Cost per Unit Risk: {metrics.cost_per_unit_risk:.4f}")
        print(f"Risk-Adjusted Cost: {metrics.risk_adjusted_cost_bps:.2f} bps")
        print(f"Cost-Volatility Ratio: {metrics.cost_volatility_ratio:.2f}")
        print(f"Risk Contribution: {metrics.risk_contribution_pct:.2f}%")
        print(f"Sharpe Impact: {metrics.sharpe_impact:.4f}")
        print(f"VaR Impact: {metrics.var_impact_bps:.2f} bps")
        print(f"Confidence Level: {metrics.confidence_level:.1%}")
        
        print("\n=== Cost-Risk Attribution ===")
        for component, value in result.cost_risk_attribution.items():
            print(f"{component.replace('_', ' ').title()}: ${value:,.2f}")
        
        print("\n=== Risk Budgeting ===")
        budgeting = result.risk_budgeting
        print(f"Risk Budget: {budgeting['cost_risk_budget_bps']:.1f} bps")
        print(f"Utilized: {budgeting['utilized_risk_budget_pct']:.1f}%")
        print(f"Remaining: {budgeting['remaining_risk_budget_bps']:.1f} bps")
        print(f"Efficiency: {budgeting['risk_budget_efficiency']:.3f}")
        
        print("\n=== Optimization Recommendations ===")
        for i, rec in enumerate(result.optimization_recommendations, 1):
            print(f"{i}. {rec}")
        
        print("\n=== Stress Test Results ===")
        for scenario, results in result.stress_test_results.items():
            print(f"{scenario.replace('_', ' ').title()}:")
            print(f"  Cost Increase: {results['cost_increase_bps']:.1f} bps")
            print(f"  VaR Impact: {results['var_impact_multiplier']:.1f}x")
            print(f"  Probability: {results['probability']:.1%}")
    
    # Run the test
    asyncio.run(test_risk_cost_analyzer())
