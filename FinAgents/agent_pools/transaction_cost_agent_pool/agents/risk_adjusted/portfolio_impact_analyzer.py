"""
Portfolio Impact Analyzer

This agent analyzes the impact of transaction costs on overall portfolio performance,
providing portfolio-level insights and optimization recommendations.
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


class PortfolioMetric(Enum):
    """Portfolio performance metrics."""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    ALPHA = "alpha"
    BETA = "beta"
    INFORMATION_RATIO = "information_ratio"
    TRACKING_ERROR = "tracking_error"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"


@dataclass
class PortfolioImpactMetrics:
    """Portfolio impact metrics from transaction costs."""
    total_cost_drag_bps: float
    annual_cost_drag_pct: float
    sharpe_ratio_impact: float
    alpha_impact_bps: float
    tracking_error_impact_bps: float
    information_ratio_impact: float
    cost_efficiency_ratio: float
    net_performance_impact_pct: float


@dataclass
class SectorImpactAnalysis:
    """Sector-level impact analysis."""
    sector_name: str
    total_cost_bps: float
    cost_as_pct_of_sector_value: float
    relative_cost_efficiency: float
    sector_weight_in_portfolio: float
    impact_on_portfolio_performance: float


@dataclass
class PortfolioOptimizationInsights:
    """Portfolio optimization insights."""
    rebalancing_frequency_recommendation: str
    cost_efficient_rebalancing_threshold: float
    optimal_trade_size_distribution: Dict[str, float]
    cost_budget_allocation: Dict[str, float]
    risk_adjusted_cost_targets: Dict[str, float]


@dataclass
class PortfolioImpactResult:
    """Result of portfolio impact analysis."""
    analysis_id: str
    timestamp: datetime
    portfolio_value: float
    impact_metrics: PortfolioImpactMetrics
    sector_analysis: List[SectorImpactAnalysis]
    time_series_impact: Dict[str, List[float]]
    optimization_insights: PortfolioOptimizationInsights
    benchmark_comparison: Dict[str, float]
    recommendations: List[str]


class PortfolioImpactAnalyzer:
    """
    Advanced portfolio impact analyzer for transaction costs.
    
    This analyzer provides comprehensive portfolio-level analysis including:
    - Portfolio performance impact measurement
    - Sector-level cost analysis
    - Time-series impact tracking
    - Cost efficiency optimization
    - Benchmark relative analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the portfolio impact analyzer.
        
        Args:
            config: Configuration parameters for the analyzer
        """
        self.config = config or {}
        self.benchmark_data = {}
        self.sector_mappings = {}
        
        # Analysis parameters
        self.analysis_horizon_days = self.config.get('analysis_horizon_days', 252)  # 1 year
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% annually
        self.benchmark_return = self.config.get('benchmark_return', 0.10)  # 10% annually
        
        # Initialize sector mappings
        self._initialize_sector_mappings()
        
        # Rebalancing parameters
        self.rebalancing_cost_threshold = self.config.get('rebalancing_cost_threshold', 0.25)  # 25 bps
        self.optimal_trade_sizes = self.config.get('optimal_trade_sizes', {
            'large_cap': 10000,
            'mid_cap': 5000,
            'small_cap': 2000
        })
        
    def _initialize_sector_mappings(self):
        """Initialize sector mappings for stocks."""
        self.sector_mappings = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'JPM': 'Financials',
            'BAC': 'Financials',
            'JNJ': 'Healthcare',
            'PFE': 'Healthcare',
            'XOM': 'Energy',
            'CVX': 'Energy'
        }
    
    async def analyze_portfolio_impact(
        self, 
        executions: List[TradeExecution],
        portfolio_positions: Dict[str, Dict[str, float]],
        historical_returns: Optional[Dict[str, pd.Series]] = None,
        benchmark_returns: Optional[pd.Series] = None,
        analysis_id: Optional[str] = None
    ) -> PortfolioImpactResult:
        """
        Analyze the impact of transaction costs on portfolio performance.
        
        Args:
            executions: List of trade executions to analyze
            portfolio_positions: Current portfolio positions
            historical_returns: Historical returns data by symbol
            benchmark_returns: Benchmark return series
            analysis_id: Unique identifier for this analysis
            
        Returns:
            Comprehensive portfolio impact analysis results
        """
        try:
            analysis_id = analysis_id or f"portfolio_impact_{datetime.utcnow().isoformat()}"
            logger.info(f"Starting portfolio impact analysis {analysis_id}")
            
            # Calculate portfolio value
            portfolio_value = sum(pos['value'] for pos in portfolio_positions.values())
            
            # Prepare execution and portfolio data
            execution_data = self._prepare_execution_data(executions)
            portfolio_weights = self._calculate_portfolio_weights(portfolio_positions)
            
            # Calculate portfolio impact metrics
            impact_metrics = await self._calculate_portfolio_impact_metrics(
                execution_data, portfolio_positions, historical_returns
            )
            
            # Perform sector-level analysis
            sector_analysis = await self._perform_sector_analysis(
                execution_data, portfolio_positions, portfolio_weights
            )
            
            # Calculate time-series impact
            time_series_impact = await self._calculate_time_series_impact(
                execution_data, historical_returns
            )
            
            # Generate optimization insights
            optimization_insights = await self._generate_optimization_insights(
                execution_data, portfolio_positions, impact_metrics
            )
            
            # Compare to benchmarks
            benchmark_comparison = await self._compare_to_benchmarks(
                impact_metrics, benchmark_returns
            )
            
            # Generate recommendations
            recommendations = await self._generate_portfolio_recommendations(
                impact_metrics, sector_analysis, optimization_insights
            )
            
            # Compile results
            result = PortfolioImpactResult(
                analysis_id=analysis_id,
                timestamp=datetime.utcnow(),
                portfolio_value=portfolio_value,
                impact_metrics=impact_metrics,
                sector_analysis=sector_analysis,
                time_series_impact=time_series_impact,
                optimization_insights=optimization_insights,
                benchmark_comparison=benchmark_comparison,
                recommendations=recommendations
            )
            
            logger.info(f"Portfolio impact analysis {analysis_id} completed")
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio impact analysis: {str(e)}")
            raise
    
    def _prepare_execution_data(self, executions: List[TradeExecution]) -> pd.DataFrame:
        """Prepare execution data for analysis."""
        
        data = []
        for execution in executions:
            market_value = execution.quantity * execution.executed_price
            cost_bps = (execution.total_cost / market_value) * 10000 if market_value > 0 else 0
            
            # Get sector mapping
            sector = self.sector_mappings.get(execution.symbol, 'Other')
            
            data.append({
                'execution_id': execution.execution_id,
                'symbol': execution.symbol,
                'sector': sector,
                'side': execution.side,
                'quantity': execution.quantity,
                'executed_price': execution.executed_price,
                'market_value': market_value,
                'total_cost': execution.total_cost,
                'cost_bps': cost_bps,
                'execution_time': execution.execution_time
            })
        
        return pd.DataFrame(data)
    
    def _calculate_portfolio_weights(self, portfolio_positions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate portfolio weights."""
        
        total_value = sum(pos['value'] for pos in portfolio_positions.values())
        
        return {
            symbol: pos['value'] / total_value if total_value > 0 else 0
            for symbol, pos in portfolio_positions.items()
        }
    
    async def _calculate_portfolio_impact_metrics(
        self, 
        execution_data: pd.DataFrame,
        portfolio_positions: Dict[str, Dict[str, float]],
        historical_returns: Optional[Dict[str, pd.Series]]
    ) -> PortfolioImpactMetrics:
        """Calculate portfolio-level impact metrics."""
        
        portfolio_value = sum(pos['value'] for pos in portfolio_positions.values())
        
        if len(execution_data) == 0 or portfolio_value == 0:
            return PortfolioImpactMetrics(
                total_cost_drag_bps=0.0,
                annual_cost_drag_pct=0.0,
                sharpe_ratio_impact=0.0,
                alpha_impact_bps=0.0,
                tracking_error_impact_bps=0.0,
                information_ratio_impact=0.0,
                cost_efficiency_ratio=0.0,
                net_performance_impact_pct=0.0
            )
        
        # Calculate total cost impact
        total_transaction_costs = execution_data['total_cost'].sum()
        total_cost_drag_bps = (total_transaction_costs / portfolio_value) * 10000
        
        # Annualize cost drag
        trading_days_per_year = 252
        days_covered = (execution_data['execution_time'].max() - 
                       execution_data['execution_time'].min()).days
        if days_covered > 0:
            annualization_factor = trading_days_per_year / days_covered
            annual_cost_drag_pct = (total_cost_drag_bps / 10000) * annualization_factor
        else:
            annual_cost_drag_pct = total_cost_drag_bps / 10000
        
        # Estimate portfolio volatility (simplified)
        if historical_returns:
            # Calculate portfolio volatility from historical returns
            portfolio_volatility = self._estimate_portfolio_volatility(
                historical_returns, portfolio_positions
            )
        else:
            portfolio_volatility = 0.15  # Default 15% annual volatility
        
        # Sharpe ratio impact
        sharpe_ratio_impact = -annual_cost_drag_pct / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Alpha impact (costs reduce alpha)
        alpha_impact_bps = -total_cost_drag_bps
        
        # Tracking error impact (costs can increase tracking error)
        tracking_error_impact_bps = total_cost_drag_bps * 0.3  # Estimated component
        
        # Information ratio impact
        if tracking_error_impact_bps > 0:
            information_ratio_impact = alpha_impact_bps / tracking_error_impact_bps
        else:
            information_ratio_impact = 0.0
        
        # Cost efficiency ratio (portfolio value per unit of cost)
        cost_efficiency_ratio = portfolio_value / total_transaction_costs if total_transaction_costs > 0 else 0
        
        # Net performance impact
        net_performance_impact_pct = -annual_cost_drag_pct  # Negative because costs reduce performance
        
        return PortfolioImpactMetrics(
            total_cost_drag_bps=total_cost_drag_bps,
            annual_cost_drag_pct=annual_cost_drag_pct,
            sharpe_ratio_impact=sharpe_ratio_impact,
            alpha_impact_bps=alpha_impact_bps,
            tracking_error_impact_bps=tracking_error_impact_bps,
            information_ratio_impact=information_ratio_impact,
            cost_efficiency_ratio=cost_efficiency_ratio,
            net_performance_impact_pct=net_performance_impact_pct
        )
    
    def _estimate_portfolio_volatility(
        self, 
        historical_returns: Dict[str, pd.Series],
        portfolio_positions: Dict[str, Dict[str, float]]
    ) -> float:
        """Estimate portfolio volatility from historical returns."""
        
        # Simplified portfolio volatility calculation
        portfolio_weights = self._calculate_portfolio_weights(portfolio_positions)
        
        # Calculate weighted average volatility
        weighted_volatility = 0.0
        total_weight = 0.0
        
        for symbol, weight in portfolio_weights.items():
            if symbol in historical_returns and len(historical_returns[symbol]) > 0:
                symbol_volatility = historical_returns[symbol].std() * np.sqrt(252)  # Annualize
                weighted_volatility += weight * symbol_volatility
                total_weight += weight
        
        if total_weight > 0:
            avg_volatility = weighted_volatility / total_weight
            # Apply diversification benefit (simplified)
            diversification_factor = 0.8  # Assume 20% diversification benefit
            return avg_volatility * diversification_factor
        else:
            return 0.15  # Default volatility
    
    async def _perform_sector_analysis(
        self, 
        execution_data: pd.DataFrame,
        portfolio_positions: Dict[str, Dict[str, float]],
        portfolio_weights: Dict[str, float]
    ) -> List[SectorImpactAnalysis]:
        """Perform sector-level cost analysis."""
        
        sector_analysis = []
        
        if len(execution_data) == 0:
            return sector_analysis
        
        # Group by sector
        sector_groups = execution_data.groupby('sector')
        
        # Calculate portfolio value by sector
        portfolio_value_by_sector = {}
        for symbol, weight in portfolio_weights.items():
            sector = self.sector_mappings.get(symbol, 'Other')
            if sector not in portfolio_value_by_sector:
                portfolio_value_by_sector[sector] = 0
            portfolio_value_by_sector[sector] += weight
        
        total_portfolio_value = sum(pos['value'] for pos in portfolio_positions.values())
        
        for sector, group in sector_groups:
            # Calculate sector metrics
            sector_total_cost = group['total_cost'].sum()
            sector_total_value = group['market_value'].sum()
            sector_cost_bps = (sector_total_cost / sector_total_value) * 10000 if sector_total_value > 0 else 0
            
            # Calculate sector weight in portfolio
            sector_weight = portfolio_value_by_sector.get(sector, 0)
            
            # Cost as percentage of sector portfolio value
            sector_portfolio_value = sector_weight * total_portfolio_value
            cost_as_pct_of_sector = (sector_total_cost / sector_portfolio_value) * 100 if sector_portfolio_value > 0 else 0
            
            # Relative cost efficiency (lower is better)
            avg_cost_bps = execution_data['cost_bps'].mean()
            relative_efficiency = sector_cost_bps / avg_cost_bps if avg_cost_bps > 0 else 1.0
            
            # Impact on portfolio performance
            portfolio_impact = sector_weight * (sector_cost_bps / 10000) * 100  # As percentage
            
            sector_analysis.append(SectorImpactAnalysis(
                sector_name=sector,
                total_cost_bps=sector_cost_bps,
                cost_as_pct_of_sector_value=cost_as_pct_of_sector,
                relative_cost_efficiency=relative_efficiency,
                sector_weight_in_portfolio=sector_weight,
                impact_on_portfolio_performance=portfolio_impact
            ))
        
        # Sort by impact
        sector_analysis.sort(key=lambda x: x.impact_on_portfolio_performance, reverse=True)
        
        return sector_analysis
    
    async def _calculate_time_series_impact(
        self, 
        execution_data: pd.DataFrame,
        historical_returns: Optional[Dict[str, pd.Series]]
    ) -> Dict[str, List[float]]:
        """Calculate time-series impact of transaction costs."""
        
        time_series_impact = {
            'daily_cost_drag': [],
            'cumulative_cost_impact': [],
            'cost_volatility': [],
            'performance_attribution': []
        }
        
        if len(execution_data) == 0:
            return time_series_impact
        
        # Group executions by date
        execution_data['date'] = execution_data['execution_time'].dt.date
        daily_costs = execution_data.groupby('date')['total_cost'].sum()
        
        # Calculate daily cost drag
        portfolio_value = execution_data['market_value'].sum()  # Simplified
        daily_cost_drag = (daily_costs / portfolio_value * 10000).tolist()  # In bps
        
        # Calculate cumulative impact
        cumulative_impact = np.cumsum(daily_cost_drag).tolist()
        
        # Calculate rolling cost volatility (30-day window)
        if len(daily_cost_drag) >= 30:
            cost_volatility = pd.Series(daily_cost_drag).rolling(30).std().fillna(0).tolist()
        else:
            cost_volatility = [0] * len(daily_cost_drag)
        
        # Performance attribution (simplified)
        performance_attribution = [-cost for cost in daily_cost_drag]  # Negative impact
        
        time_series_impact.update({
            'daily_cost_drag': daily_cost_drag,
            'cumulative_cost_impact': cumulative_impact,
            'cost_volatility': cost_volatility,
            'performance_attribution': performance_attribution
        })
        
        return time_series_impact
    
    async def _generate_optimization_insights(
        self, 
        execution_data: pd.DataFrame,
        portfolio_positions: Dict[str, Dict[str, float]],
        impact_metrics: PortfolioImpactMetrics
    ) -> PortfolioOptimizationInsights:
        """Generate portfolio optimization insights."""
        
        # Rebalancing frequency recommendation
        if impact_metrics.annual_cost_drag_pct > 0.5:  # 50 bps
            rebalancing_freq = "monthly"
        elif impact_metrics.annual_cost_drag_pct > 0.25:  # 25 bps
            rebalancing_freq = "quarterly"
        else:
            rebalancing_freq = "semi-annually"
        
        # Cost-efficient rebalancing threshold
        cost_efficient_threshold = max(0.1, impact_metrics.total_cost_drag_bps * 2)  # 2x current cost
        
        # Optimal trade size distribution
        if len(execution_data) > 0:
            size_percentiles = execution_data['quantity'].quantile([0.25, 0.5, 0.75]).to_dict()
            optimal_sizes = {
                'small_trades': size_percentiles[0.25],
                'medium_trades': size_percentiles[0.5],
                'large_trades': size_percentiles[0.75]
            }
        else:
            optimal_sizes = {'small_trades': 1000, 'medium_trades': 5000, 'large_trades': 10000}
        
        # Cost budget allocation by sector
        cost_budget = {}
        if len(execution_data) > 0:
            sector_costs = execution_data.groupby('sector')['total_cost'].sum()
            total_cost = sector_costs.sum()
            for sector, cost in sector_costs.items():
                cost_budget[sector] = (cost / total_cost) * 100 if total_cost > 0 else 0
        
        # Risk-adjusted cost targets
        risk_adjusted_targets = {}
        portfolio_vol = 0.15  # Assumed portfolio volatility
        for symbol in portfolio_positions.keys():
            # Set target based on position size and estimated volatility
            symbol_vol = np.random.normal(0.25, 0.1)  # Simplified volatility estimate
            symbol_vol = max(0.1, min(0.5, symbol_vol))
            risk_adjusted_targets[symbol] = (symbol_vol / portfolio_vol) * 5.0  # Target in bps
        
        return PortfolioOptimizationInsights(
            rebalancing_frequency_recommendation=rebalancing_freq,
            cost_efficient_rebalancing_threshold=cost_efficient_threshold,
            optimal_trade_size_distribution=optimal_sizes,
            cost_budget_allocation=cost_budget,
            risk_adjusted_cost_targets=risk_adjusted_targets
        )
    
    async def _compare_to_benchmarks(
        self, 
        impact_metrics: PortfolioImpactMetrics,
        benchmark_returns: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Compare portfolio cost impact to benchmarks."""
        
        comparison = {}
        
        # Industry benchmarks (simplified)
        industry_benchmarks = {
            'institutional_average_cost_bps': 8.0,
            'retail_average_cost_bps': 15.0,
            'best_practice_cost_bps': 4.0,
            'passive_fund_cost_bps': 2.0
        }
        
        current_cost = impact_metrics.total_cost_drag_bps
        
        for benchmark_name, benchmark_value in industry_benchmarks.items():
            comparison[f'vs_{benchmark_name}'] = current_cost - benchmark_value
        
        # Sharpe ratio comparison
        comparison['sharpe_impact_vs_no_cost'] = impact_metrics.sharpe_ratio_impact
        
        # Alpha impact comparison
        comparison['alpha_impact_vs_benchmark'] = impact_metrics.alpha_impact_bps
        
        return comparison
    
    async def _generate_portfolio_recommendations(
        self, 
        impact_metrics: PortfolioImpactMetrics,
        sector_analysis: List[SectorImpactAnalysis],
        optimization_insights: PortfolioOptimizationInsights
    ) -> List[str]:
        """Generate portfolio-level recommendations."""
        
        recommendations = []
        
        # Overall cost impact recommendations
        if impact_metrics.annual_cost_drag_pct > 0.5:
            recommendations.append(
                f"Annual cost drag of {impact_metrics.annual_cost_drag_pct:.2%} is high - "
                "implement comprehensive cost reduction strategies"
            )
        
        # Sharpe ratio impact recommendations
        if impact_metrics.sharpe_ratio_impact < -0.1:
            recommendations.append(
                f"Transaction costs significantly impact Sharpe ratio ({impact_metrics.sharpe_ratio_impact:.3f}) - "
                "prioritize execution efficiency improvements"
            )
        
        # Sector-specific recommendations
        if sector_analysis:
            worst_sector = max(sector_analysis, key=lambda x: x.relative_cost_efficiency)
            if worst_sector.relative_cost_efficiency > 1.5:
                recommendations.append(
                    f"{worst_sector.sector_name} sector shows poor cost efficiency "
                    f"({worst_sector.relative_cost_efficiency:.1f}x average) - "
                    "focus optimization efforts on this sector"
                )
            
            # High impact sectors
            high_impact_sectors = [s for s in sector_analysis if s.impact_on_portfolio_performance > 0.1]
            if high_impact_sectors:
                sector_names = [s.sector_name for s in high_impact_sectors]
                recommendations.append(
                    f"High portfolio impact from {', '.join(sector_names)} sectors - "
                    "consider sector-specific execution strategies"
                )
        
        # Rebalancing recommendations
        recommendations.append(
            f"Optimize rebalancing frequency to {optimization_insights.rebalancing_frequency_recommendation} "
            f"with threshold of {optimization_insights.cost_efficient_rebalancing_threshold:.1f} bps"
        )
        
        # Trade size recommendations
        optimal_sizes = optimization_insights.optimal_trade_size_distribution
        if 'large_trades' in optimal_sizes and 'small_trades' in optimal_sizes:
            size_ratio = optimal_sizes['large_trades'] / optimal_sizes['small_trades']
            if size_ratio > 20:
                recommendations.append(
                    "Large variation in trade sizes detected - implement size-based execution strategies"
                )
        
        # Cost efficiency recommendations
        if impact_metrics.cost_efficiency_ratio < 1000:  # Less than 1000:1 ratio
            recommendations.append(
                f"Cost efficiency ratio ({impact_metrics.cost_efficiency_ratio:.0f}:1) is low - "
                "review overall trading strategy"
            )
        
        return recommendations if recommendations else [
            "Portfolio transaction cost impact is within acceptable ranges"
        ]


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from datetime import datetime, timedelta
    
    async def test_portfolio_impact_analyzer():
        """Test the portfolio impact analyzer with sample data."""
        
        # Create sample executions
        sample_executions = [
            TradeExecution(
                execution_id="exec_001",
                symbol="AAPL",
                side="BUY",
                quantity=1000,
                executed_price=150.25,
                benchmark_price=150.20,
                execution_time=datetime.utcnow() - timedelta(days=5),
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
                execution_time=datetime.utcnow() - timedelta(days=3),
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
                execution_time=datetime.utcnow() - timedelta(days=1),
                venue="ARCA",
                order_type="LIMIT",
                total_cost=38.0
            ),
            TradeExecution(
                execution_id="exec_004",
                symbol="JPM",
                side="BUY",
                quantity=1200,
                executed_price=165.50,
                benchmark_price=165.45,
                execution_time=datetime.utcnow() - timedelta(hours=6),
                venue="NYSE",
                order_type="LIMIT",
                total_cost=59.0
            )
        ]
        
        # Create sample portfolio positions
        portfolio_positions = {
            "AAPL": {"position": 5000, "value": 750000},
            "GOOGL": {"position": 200, "value": 550000},
            "MSFT": {"position": 1500, "value": 510000},
            "JPM": {"position": 3000, "value": 495000},
            "AMZN": {"position": 300, "value": 450000}
        }
        
        # Initialize analyzer and run analysis
        analyzer = PortfolioImpactAnalyzer()
        result = await analyzer.analyze_portfolio_impact(
            sample_executions, 
            portfolio_positions
        )
        
        print("=== Portfolio Impact Analysis Results ===")
        print(f"Analysis ID: {result.analysis_id}")
        print(f"Portfolio Value: ${result.portfolio_value:,.2f}")
        
        print("\n=== Portfolio Impact Metrics ===")
        metrics = result.impact_metrics
        print(f"Total Cost Drag: {metrics.total_cost_drag_bps:.2f} bps")
        print(f"Annual Cost Drag: {metrics.annual_cost_drag_pct:.2%}")
        print(f"Sharpe Ratio Impact: {metrics.sharpe_ratio_impact:.4f}")
        print(f"Alpha Impact: {metrics.alpha_impact_bps:.2f} bps")
        print(f"Cost Efficiency Ratio: {metrics.cost_efficiency_ratio:.0f}:1")
        print(f"Net Performance Impact: {metrics.net_performance_impact_pct:.2%}")
        
        print("\n=== Sector Analysis ===")
        for sector in result.sector_analysis:
            print(f"{sector.sector_name}:")
            print(f"  Total Cost: {sector.total_cost_bps:.2f} bps")
            print(f"  Portfolio Weight: {sector.sector_weight_in_portfolio:.1%}")
            print(f"  Cost Efficiency: {sector.relative_cost_efficiency:.2f}x")
            print(f"  Portfolio Impact: {sector.impact_on_portfolio_performance:.3f}%")
        
        print("\n=== Optimization Insights ===")
        insights = result.optimization_insights
        print(f"Recommended Rebalancing: {insights.rebalancing_frequency_recommendation}")
        print(f"Cost-Efficient Threshold: {insights.cost_efficient_rebalancing_threshold:.1f} bps")
        print("Optimal Trade Sizes:")
        for size_type, size in insights.optimal_trade_size_distribution.items():
            print(f"  {size_type}: {size:,.0f} shares")
        
        print("\n=== Benchmark Comparison ===")
        for benchmark, difference in result.benchmark_comparison.items():
            print(f"{benchmark}: {difference:+.2f} bps")
        
        print("\n=== Recommendations ===")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"{i}. {rec}")
    
    # Run the test
    asyncio.run(test_portfolio_impact_analyzer())
