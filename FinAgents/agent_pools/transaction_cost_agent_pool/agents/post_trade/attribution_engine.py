"""
Post-Trade Attribution Engine

This agent performs detailed attribution analysis of transaction costs,
identifying the sources and drivers of execution performance.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum

from ...schema.execution_schema import TradeExecution
from ...schema.cost_models import TransactionCostBreakdown


logger = logging.getLogger(__name__)


class AttributionCategory(Enum):
    """Categories for cost attribution analysis."""
    MARKET_IMPACT = "market_impact"
    TIMING_COST = "timing_cost"
    SPREAD_COST = "spread_cost"
    COMMISSION = "commission"
    FEES = "fees"
    OPPORTUNITY_COST = "opportunity_cost"
    VENUE_SELECTION = "venue_selection"
    ORDER_TYPE = "order_type"
    EXECUTION_TIMING = "execution_timing"


@dataclass
class AttributionComponent:
    """Single component of cost attribution."""
    category: AttributionCategory
    cost_bps: float
    percentage_contribution: float
    description: str
    confidence_level: float = 0.8


@dataclass
class AttributionResult:
    """Result of comprehensive attribution analysis."""
    analysis_id: str
    timestamp: datetime
    total_cost_bps: float
    attribution_components: List[AttributionComponent]
    factor_analysis: Dict[str, Dict[str, float]]
    performance_drivers: List[str]
    benchmark_comparison: Dict[str, float]
    risk_attribution: Dict[str, float]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AttributionEngine:
    """
    Advanced attribution engine for transaction cost analysis.
    
    This engine provides comprehensive attribution analysis including:
    - Multi-factor cost decomposition
    - Performance driver identification  
    - Risk factor attribution
    - Benchmark relative analysis
    - Strategy effectiveness measurement
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the attribution engine.
        
        Args:
            config: Configuration parameters for the engine
        """
        self.config = config or {}
        self.attribution_cache = {}
        self.benchmark_data = {}
        self.factor_models = {}
        
        # Attribution parameters
        self.attribution_window = self.config.get('attribution_window_days', 30)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.min_observations = self.config.get('min_observations', 10)
        
        # Factor model parameters
        self.risk_factors = self.config.get('risk_factors', [
            'market_volatility', 'bid_ask_spread', 'trading_volume',
            'time_of_day', 'market_stress', 'liquidity_conditions'
        ])
        
        # Benchmark settings
        self.benchmark_universe = self.config.get('benchmark_universe', {
            'peer_execution': 'industry_average',
            'theoretical_optimal': 'zero_cost',
            'historical_performance': 'rolling_average'
        })
        
    async def perform_attribution(
        self, 
        executions: List[TradeExecution],
        market_data: Optional[Dict[str, Any]] = None,
        benchmark_data: Optional[Dict[str, Any]] = None,
        analysis_id: Optional[str] = None
    ) -> AttributionResult:
        """
        Perform comprehensive transaction cost attribution analysis.
        
        Args:
            executions: List of trade executions to analyze
            market_data: Market data for factor analysis
            benchmark_data: Benchmark data for comparison
            analysis_id: Unique identifier for this analysis
            
        Returns:
            Detailed attribution analysis results
        """
        try:
            analysis_id = analysis_id or f"attribution_{datetime.utcnow().isoformat()}"
            logger.info(f"Starting attribution analysis {analysis_id} for {len(executions)} trades")
            
            # Prepare execution data
            executions_df = self._prepare_execution_data(executions)
            
            # Prepare market data
            market_df = self._prepare_market_data(market_data, executions_df)
            
            # Perform cost decomposition
            cost_components = await self._decompose_transaction_costs(executions_df, market_df)
            
            # Perform factor analysis
            factor_analysis = await self._perform_factor_analysis(executions_df, market_df)
            
            # Identify performance drivers
            performance_drivers = await self._identify_performance_drivers(
                executions_df, market_df, factor_analysis
            )
            
            # Compare to benchmarks
            benchmark_comparison = await self._compare_to_benchmarks(
                executions_df, benchmark_data
            )
            
            # Perform risk attribution
            risk_attribution = await self._perform_risk_attribution(
                executions_df, market_df, factor_analysis
            )
            
            # Generate recommendations
            recommendations = await self._generate_attribution_recommendations(
                cost_components, factor_analysis, benchmark_comparison
            )
            
            # Calculate total cost
            total_cost_bps = sum(comp.cost_bps for comp in cost_components)
            
            # Compile results
            result = AttributionResult(
                analysis_id=analysis_id,
                timestamp=datetime.utcnow(),
                total_cost_bps=total_cost_bps,
                attribution_components=cost_components,
                factor_analysis=factor_analysis,
                performance_drivers=performance_drivers,
                benchmark_comparison=benchmark_comparison,
                risk_attribution=risk_attribution,
                recommendations=recommendations,
                metadata={
                    'execution_count': len(executions),
                    'analysis_period': self.attribution_window,
                    'confidence_threshold': self.confidence_threshold
                }
            )
            
            logger.info(f"Attribution analysis {analysis_id} completed")
            return result
            
        except Exception as e:
            logger.error(f"Error in attribution analysis: {str(e)}")
            raise
    
    def _prepare_execution_data(self, executions: List[TradeExecution]) -> pd.DataFrame:
        """Prepare execution data for attribution analysis."""
        
        data = []
        for execution in executions:
            # Calculate basic metrics
            market_value = execution.quantity * execution.executed_price
            price_impact = (execution.executed_price - execution.benchmark_price) / execution.benchmark_price
            
            data.append({
                'execution_id': execution.execution_id,
                'symbol': execution.symbol,
                'side': execution.side,
                'quantity': execution.quantity,
                'executed_price': execution.executed_price,
                'benchmark_price': execution.benchmark_price,
                'price_impact': price_impact,
                'price_impact_bps': price_impact * 10000,
                'execution_time': execution.execution_time,
                'venue': execution.venue,
                'order_type': execution.order_type,
                'total_cost': execution.total_cost,
                'market_value': market_value,
                'cost_bps': (execution.total_cost / market_value) * 10000 if market_value > 0 else 0
            })
        
        df = pd.DataFrame(data)
        
        # Add time-based features
        df['execution_hour'] = df['execution_time'].dt.hour
        df['execution_day'] = df['execution_time'].dt.dayofweek
        df['execution_minute'] = df['execution_time'].dt.minute
        
        # Add size categories
        df['order_size_category'] = pd.qcut(
            df['quantity'], 
            q=4, 
            labels=['Small', 'Medium', 'Large', 'XLarge'],
            duplicates='drop'
        )
        
        # Add value categories
        df['order_value_category'] = pd.qcut(
            df['market_value'], 
            q=4, 
            labels=['Low', 'Medium', 'High', 'VeryHigh'],
            duplicates='drop'
        )
        
        return df
    
    def _prepare_market_data(
        self, 
        market_data: Optional[Dict[str, Any]], 
        executions_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Prepare market data for factor analysis."""
        
        if not market_data:
            # Generate synthetic market data for demonstration
            market_df = pd.DataFrame({
                'timestamp': executions_df['execution_time'],
                'market_volatility': np.random.normal(0.15, 0.05, len(executions_df)).clip(0.05, 0.5),
                'bid_ask_spread_bps': np.random.normal(8, 3, len(executions_df)).clip(1, 25),
                'trading_volume': np.random.lognormal(15, 1, len(executions_df)),
                'market_stress_index': np.random.normal(0.3, 0.2, len(executions_df)).clip(0, 1)
            })
            return market_df
        
        # Process provided market data
        try:
            market_df = pd.DataFrame(market_data)
            # Align with execution timestamps
            return market_df
        except Exception as e:
            logger.warning(f"Could not process market data: {e}")
            return None
    
    async def _decompose_transaction_costs(
        self, 
        executions_df: pd.DataFrame,
        market_df: Optional[pd.DataFrame]
    ) -> List[AttributionComponent]:
        """Decompose transaction costs into attribution components."""
        
        components = []
        total_market_value = executions_df['market_value'].sum()
        
        if total_market_value == 0:
            return components
        
        # Market Impact Component
        market_impact_cost = abs(executions_df['price_impact_bps'].mean())
        market_impact_pct = market_impact_cost / executions_df['cost_bps'].mean() * 100 if executions_df['cost_bps'].mean() > 0 else 0
        
        components.append(AttributionComponent(
            category=AttributionCategory.MARKET_IMPACT,
            cost_bps=market_impact_cost,
            percentage_contribution=min(market_impact_pct, 100),
            description=f"Average market impact of {market_impact_cost:.2f} bps per trade",
            confidence_level=0.85
        ))
        
        # Commission and Fees Component
        avg_commission_bps = executions_df['cost_bps'].mean() - market_impact_cost
        commission_pct = avg_commission_bps / executions_df['cost_bps'].mean() * 100 if executions_df['cost_bps'].mean() > 0 else 0
        
        components.append(AttributionComponent(
            category=AttributionCategory.COMMISSION,
            cost_bps=max(0, avg_commission_bps),
            percentage_contribution=max(0, min(commission_pct, 100)),
            description=f"Average commission and fees of {max(0, avg_commission_bps):.2f} bps per trade",
            confidence_level=0.95
        ))
        
        # Timing Cost Component (estimated from price volatility)
        if len(executions_df) > 1:
            timing_cost = executions_df['price_impact_bps'].std() * 0.5  # Approximation
            timing_pct = timing_cost / executions_df['cost_bps'].mean() * 100 if executions_df['cost_bps'].mean() > 0 else 0
            
            components.append(AttributionComponent(
                category=AttributionCategory.TIMING_COST,
                cost_bps=timing_cost,
                percentage_contribution=min(timing_pct, 30),  # Cap at reasonable level
                description=f"Estimated timing cost of {timing_cost:.2f} bps from execution delays",
                confidence_level=0.6
            ))
        
        # Venue Selection Impact
        if len(executions_df['venue'].unique()) > 1:
            venue_performance = executions_df.groupby('venue')['cost_bps'].mean()
            best_venue_cost = venue_performance.min()
            avg_venue_cost = venue_performance.mean()
            venue_selection_cost = avg_venue_cost - best_venue_cost
            venue_pct = venue_selection_cost / executions_df['cost_bps'].mean() * 100 if executions_df['cost_bps'].mean() > 0 else 0
            
            if venue_selection_cost > 0.5:  # Only if significant
                components.append(AttributionComponent(
                    category=AttributionCategory.VENUE_SELECTION,
                    cost_bps=venue_selection_cost,
                    percentage_contribution=min(venue_pct, 25),
                    description=f"Venue selection cost of {venue_selection_cost:.2f} bps from suboptimal routing",
                    confidence_level=0.7
                ))
        
        # Order Type Impact
        if len(executions_df['order_type'].unique()) > 1:
            order_type_performance = executions_df.groupby('order_type')['cost_bps'].mean()
            if 'LIMIT' in order_type_performance.index and 'MARKET' in order_type_performance.index:
                order_type_cost = order_type_performance['MARKET'] - order_type_performance['LIMIT']
                order_type_pct = order_type_cost / executions_df['cost_bps'].mean() * 100 if executions_df['cost_bps'].mean() > 0 else 0
                
                if order_type_cost > 1.0:  # Only if significant
                    components.append(AttributionComponent(
                        category=AttributionCategory.ORDER_TYPE,
                        cost_bps=order_type_cost,
                        percentage_contribution=min(order_type_pct, 20),
                        description=f"Order type impact of {order_type_cost:.2f} bps from market vs limit orders",
                        confidence_level=0.8
                    ))
        
        return components
    
    async def _perform_factor_analysis(
        self, 
        executions_df: pd.DataFrame,
        market_df: Optional[pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """Perform multi-factor analysis of execution costs."""
        
        factor_analysis = {
            'time_factors': {},
            'size_factors': {},
            'venue_factors': {},
            'market_factors': {},
            'correlations': {}
        }
        
        # Time factor analysis
        hourly_costs = executions_df.groupby('execution_hour')['cost_bps'].agg(['mean', 'std', 'count'])
        factor_analysis['time_factors'] = {
            'hourly_variation': hourly_costs['std'].mean(),
            'peak_hour_cost': hourly_costs['mean'].max(),
            'off_peak_hour_cost': hourly_costs['mean'].min(),
            'time_effect_bps': hourly_costs['mean'].max() - hourly_costs['mean'].min()
        }
        
        # Size factor analysis
        if 'order_size_category' in executions_df.columns:
            size_costs = executions_df.groupby('order_size_category')['cost_bps'].mean()
            factor_analysis['size_factors'] = {
                'size_effect_exists': size_costs.max() - size_costs.min() > 2.0,
                'large_order_premium_bps': size_costs.get('XLarge', 0) - size_costs.get('Small', 0),
                'size_correlation': executions_df['quantity'].corr(executions_df['cost_bps'])
            }
        
        # Venue factor analysis
        venue_costs = executions_df.groupby('venue')['cost_bps'].agg(['mean', 'std', 'count'])
        if len(venue_costs) > 1:
            factor_analysis['venue_factors'] = {
                'venue_effect_bps': venue_costs['mean'].max() - venue_costs['mean'].min(),
                'best_venue': venue_costs['mean'].idxmin(),
                'worst_venue': venue_costs['mean'].idxmax(),
                'venue_consistency': 1 / (venue_costs['std'].mean() + 1e-6)  # Higher is better
            }
        
        # Market factor analysis (if market data available)
        if market_df is not None and len(market_df) > 0:
            # Align market data with executions
            aligned_data = executions_df.copy()
            
            if 'market_volatility' in market_df.columns:
                factor_analysis['market_factors']['volatility_correlation'] = \
                    market_df['market_volatility'].corr(aligned_data['cost_bps'])
            
            if 'bid_ask_spread_bps' in market_df.columns:
                factor_analysis['market_factors']['spread_correlation'] = \
                    market_df['bid_ask_spread_bps'].corr(aligned_data['cost_bps'])
        
        # Cross-factor correlations
        numeric_cols = ['quantity', 'execution_hour', 'price_impact_bps']
        for col in numeric_cols:
            if col in executions_df.columns:
                corr = executions_df['cost_bps'].corr(executions_df[col])
                if abs(corr) > 0.1:  # Only meaningful correlations
                    factor_analysis['correlations'][col] = corr
        
        return factor_analysis
    
    async def _identify_performance_drivers(
        self, 
        executions_df: pd.DataFrame,
        market_df: Optional[pd.DataFrame],
        factor_analysis: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Identify key drivers of execution performance."""
        
        drivers = []
        
        # Time-based drivers
        time_factors = factor_analysis.get('time_factors', {})
        if time_factors.get('time_effect_bps', 0) > 3:
            drivers.append(f"Execution timing drives {time_factors['time_effect_bps']:.1f} bps cost variation")
        
        # Size-based drivers
        size_factors = factor_analysis.get('size_factors', {})
        if size_factors.get('size_effect_exists', False):
            premium = size_factors.get('large_order_premium_bps', 0)
            if premium > 2:
                drivers.append(f"Large orders incur {premium:.1f} bps premium over small orders")
        
        # Venue-based drivers
        venue_factors = factor_analysis.get('venue_factors', {})
        if venue_factors.get('venue_effect_bps', 0) > 2:
            best_venue = venue_factors.get('best_venue', 'Unknown')
            worst_venue = venue_factors.get('worst_venue', 'Unknown')
            effect = venue_factors['venue_effect_bps']
            drivers.append(f"Venue selection drives {effect:.1f} bps difference ({best_venue} vs {worst_venue})")
        
        # Market condition drivers
        market_factors = factor_analysis.get('market_factors', {})
        vol_corr = market_factors.get('volatility_correlation', 0)
        if abs(vol_corr) > 0.3:
            direction = "increases" if vol_corr > 0 else "decreases"
            drivers.append(f"Market volatility strongly {direction} execution costs (correlation: {vol_corr:.2f})")
        
        # Correlation-based drivers
        correlations = factor_analysis.get('correlations', {})
        for factor, corr in correlations.items():
            if abs(corr) > 0.4:
                direction = "positively" if corr > 0 else "negatively"
                drivers.append(f"{factor.replace('_', ' ').title()} {direction} correlated with costs ({corr:.2f})")
        
        return drivers if drivers else ["No dominant performance drivers identified"]
    
    async def _compare_to_benchmarks(
        self, 
        executions_df: pd.DataFrame,
        benchmark_data: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compare execution performance to various benchmarks."""
        
        comparison = {}
        avg_cost = executions_df['cost_bps'].mean()
        
        # Historical benchmark (rolling average)
        comparison['vs_historical_avg'] = 0.0  # Placeholder - would use historical data
        
        # Industry benchmark (simplified)
        industry_avg_cost = 6.0  # Typical industry average in bps
        comparison['vs_industry_avg'] = avg_cost - industry_avg_cost
        
        # Theoretical optimal (zero cost)
        comparison['vs_theoretical_optimal'] = avg_cost
        
        # Best practice benchmark
        best_practice_cost = 3.0  # Best-in-class execution cost
        comparison['vs_best_practice'] = avg_cost - best_practice_cost
        
        # Venue-specific benchmarks
        if len(executions_df['venue'].unique()) > 1:
            venue_costs = executions_df.groupby('venue')['cost_bps'].mean()
            best_venue_cost = venue_costs.min()
            comparison['vs_best_venue'] = avg_cost - best_venue_cost
        
        # Time-period benchmarks
        if len(executions_df) > 10:
            recent_trades = executions_df.tail(len(executions_df) // 2)
            earlier_trades = executions_df.head(len(executions_df) // 2)
            comparison['recent_vs_earlier'] = recent_trades['cost_bps'].mean() - earlier_trades['cost_bps'].mean()
        
        return comparison
    
    async def _perform_risk_attribution(
        self, 
        executions_df: pd.DataFrame,
        market_df: Optional[pd.DataFrame],
        factor_analysis: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Perform risk attribution analysis."""
        
        risk_attribution = {}
        
        # Cost volatility risk
        cost_volatility = executions_df['cost_bps'].std()
        risk_attribution['cost_volatility_risk'] = cost_volatility
        
        # Timing risk (variance due to execution time)
        if 'execution_hour' in executions_df.columns:
            hourly_variance = executions_df.groupby('execution_hour')['cost_bps'].var().mean()
            risk_attribution['timing_risk'] = hourly_variance ** 0.5
        
        # Venue concentration risk
        venue_counts = executions_df['venue'].value_counts(normalize=True)
        venue_concentration = (venue_counts ** 2).sum()  # Herfindahl index
        risk_attribution['venue_concentration_risk'] = venue_concentration
        
        # Size concentration risk
        if 'order_size_category' in executions_df.columns:
            size_counts = executions_df['order_size_category'].value_counts(normalize=True)
            size_concentration = (size_counts ** 2).sum()
            risk_attribution['size_concentration_risk'] = size_concentration
        
        # Market impact risk (tail risk)
        impact_95th = executions_df['price_impact_bps'].quantile(0.95)
        risk_attribution['market_impact_tail_risk'] = impact_95th
        
        # Execution cost VaR (95th percentile)
        cost_var_95 = executions_df['cost_bps'].quantile(0.95)
        risk_attribution['cost_var_95'] = cost_var_95
        
        return risk_attribution
    
    async def _generate_attribution_recommendations(
        self, 
        cost_components: List[AttributionComponent],
        factor_analysis: Dict[str, Dict[str, float]],
        benchmark_comparison: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on attribution analysis."""
        
        recommendations = []
        
        # Component-based recommendations
        for component in cost_components:
            if component.cost_bps > 5 and component.confidence_level > 0.7:
                if component.category == AttributionCategory.MARKET_IMPACT:
                    recommendations.append(
                        f"Market impact of {component.cost_bps:.1f} bps is significant - "
                        "consider order splitting and algorithmic execution strategies"
                    )
                elif component.category == AttributionCategory.VENUE_SELECTION:
                    recommendations.append(
                        f"Venue selection costs {component.cost_bps:.1f} bps - "
                        "optimize routing to best-performing venues"
                    )
                elif component.category == AttributionCategory.ORDER_TYPE:
                    recommendations.append(
                        f"Order type selection costs {component.cost_bps:.1f} bps - "
                        "consider increasing use of limit orders where appropriate"
                    )
        
        # Factor-based recommendations
        time_factors = factor_analysis.get('time_factors', {})
        if time_factors.get('time_effect_bps', 0) > 3:
            recommendations.append(
                "Significant time-of-day effect detected - implement time-based execution strategies"
            )
        
        venue_factors = factor_analysis.get('venue_factors', {})
        if venue_factors.get('venue_effect_bps', 0) > 2:
            best_venue = venue_factors.get('best_venue', 'top-performing venue')
            recommendations.append(f"Route more orders to {best_venue} for optimal execution")
        
        # Benchmark-based recommendations
        if benchmark_comparison.get('vs_industry_avg', 0) > 2:
            recommendations.append(
                "Execution costs are above industry average - review overall execution strategy"
            )
        
        if benchmark_comparison.get('vs_best_practice', 0) > 3:
            recommendations.append(
                "Significant gap vs best practice - consider advanced execution algorithms"
            )
        
        return recommendations if recommendations else [
            "Execution performance is well-attributed - continue monitoring key factors"
        ]


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from datetime import datetime, timedelta
    
    async def test_attribution_engine():
        """Test the attribution engine with sample data."""
        
        # Create sample executions
        sample_executions = [
            TradeExecution(
                execution_id="exec_001",
                symbol="AAPL",
                side="BUY",
                quantity=1000,
                executed_price=150.30,
                benchmark_price=150.20,
                execution_time=datetime.utcnow() - timedelta(hours=3),
                venue="NYSE",
                order_type="MARKET",
                total_cost=95.0
            ),
            TradeExecution(
                execution_id="exec_002",
                symbol="GOOGL",
                side="SELL",
                quantity=500,
                executed_price=2750.50,
                benchmark_price=2755.00,
                execution_time=datetime.utcnow() - timedelta(hours=2),
                venue="NASDAQ",
                order_type="LIMIT",
                total_cost=125.0
            ),
            TradeExecution(
                execution_id="exec_003",
                symbol="MSFT",
                side="BUY",
                quantity=1500,
                executed_price=340.10,
                benchmark_price=340.05,
                execution_time=datetime.utcnow() - timedelta(hours=1),
                venue="ARCA",
                order_type="MARKET",
                total_cost=75.0
            )
        ]
        
        # Initialize engine and run analysis
        engine = AttributionEngine()
        result = await engine.perform_attribution(sample_executions)
        
        print("=== Attribution Analysis Results ===")
        print(f"Analysis ID: {result.analysis_id}")
        print(f"Total Cost: {result.total_cost_bps:.2f} bps")
        
        print("\n=== Cost Attribution Components ===")
        for component in result.attribution_components:
            print(f"- {component.category.value}: {component.cost_bps:.2f} bps "
                  f"({component.percentage_contribution:.1f}%) "
                  f"[Confidence: {component.confidence_level:.1%}]")
            print(f"  {component.description}")
        
        print("\n=== Performance Drivers ===")
        for driver in result.performance_drivers:
            print(f"- {driver}")
        
        print("\n=== Benchmark Comparison ===")
        for benchmark, difference in result.benchmark_comparison.items():
            print(f"- {benchmark}: {difference:+.2f} bps")
        
        print("\n=== Recommendations ===")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"{i}. {rec}")
    
    # Run the test
    asyncio.run(test_attribution_engine())
