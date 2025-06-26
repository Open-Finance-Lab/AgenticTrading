"""
Post-Trade Execution Analysis Agent

This agent analyzes completed trades to evaluate execution quality,
identify patterns, and provide insights for future optimization.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Temporary imports with fallback for missing models
try:
    from ...schema.execution_schema import (
        ExecutionAnalysisRequest,
        ExecutionAnalysisResult,
        ExecutionQualityMetrics,
        TradeExecution,
        ExecutionBenchmark
    )
except ImportError:
    # Create placeholder classes if schema models are not available
    class ExecutionAnalysisRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class ExecutionAnalysisResult:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class ExecutionQualityMetrics:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class TradeExecution:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class ExecutionBenchmark:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

try:
    from ...schema.cost_models import TransactionCostBreakdown
except ImportError:
    class TransactionCostBreakdown:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


logger = logging.getLogger(__name__)


class ExecutionAnalyzer:
    """
    Analyzes post-trade execution data to evaluate performance and identify optimization opportunities.
    
    This agent provides comprehensive analysis of trade execution quality, including:
    - Implementation shortfall analysis
    - Execution cost breakdown
    - Performance attribution
    - Benchmark comparisons
    - Pattern identification
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the execution analyzer.
        
        Args:
            config: Configuration parameters for the analyzer
        """
        self.config = config or {}
        self.analysis_cache = {}
        self.benchmark_data = {}
        
        # Analysis parameters
        self.lookback_periods = self.config.get('lookback_periods', [30, 90, 180])
        self.benchmark_sources = self.config.get('benchmark_sources', ['TWAP', 'VWAP', 'Arrival'])
        self.quality_thresholds = self.config.get('quality_thresholds', {
            'excellent': 0.02,
            'good': 0.05,
            'acceptable': 0.10,
            'poor': 0.20
        })
        
    async def analyze_execution(self, request: ExecutionAnalysisRequest) -> ExecutionAnalysisResult:
        """
        Perform comprehensive execution analysis.
        
        Args:
            request: Analysis request containing execution data and parameters
            
        Returns:
            Detailed execution analysis results
        """
        try:
            logger.info(f"Starting execution analysis for {len(request.executions)} trades")
            
            # Prepare execution data
            executions_df = self._prepare_execution_data(request.executions)
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(
                executions_df, request.benchmark_type
            )
            
            # Perform implementation shortfall analysis
            shortfall_analysis = await self._analyze_implementation_shortfall(
                executions_df, request.benchmark_prices
            )
            
            # Analyze execution patterns
            pattern_analysis = await self._analyze_execution_patterns(executions_df)
            
            # Generate cost attribution
            cost_attribution = await self._analyze_cost_attribution(executions_df)
            
            # Create recommendations
            recommendations = await self._generate_recommendations(
                quality_metrics, pattern_analysis, cost_attribution
            )
            
            # Compile results
            result = ExecutionAnalysisResult(
                request_id=request.request_id,
                analysis_timestamp=datetime.utcnow(),
                executions_analyzed=len(request.executions),
                quality_metrics=quality_metrics,
                shortfall_analysis=shortfall_analysis,
                pattern_insights=pattern_analysis,
                cost_attribution=cost_attribution,
                recommendations=recommendations,
                benchmark_comparison=await self._compare_to_benchmarks(
                    executions_df, request.benchmark_type
                )
            )
            
            logger.info(f"Execution analysis completed for request {request.request_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in execution analysis: {str(e)}")
            raise
    
    def _prepare_execution_data(self, executions: List[TradeExecution]) -> pd.DataFrame:
        """Prepare execution data for analysis."""
        data = []
        for execution in executions:
            data.append({
                'execution_id': execution.execution_id,
                'symbol': execution.symbol,
                'side': execution.side,
                'quantity': execution.quantity,
                'executed_price': execution.executed_price,
                'benchmark_price': execution.benchmark_price,
                'execution_time': execution.execution_time,
                'venue': execution.venue,
                'order_type': execution.order_type,
                'total_cost': execution.total_cost,
                'market_value': execution.quantity * execution.executed_price
            })
        
        df = pd.DataFrame(data)
        
        # Calculate derived metrics
        df['price_deviation'] = (df['executed_price'] - df['benchmark_price']) / df['benchmark_price']
        df['cost_bps'] = (df['total_cost'] / df['market_value']) * 10000
        df['execution_hour'] = df['execution_time'].dt.hour
        df['execution_day'] = df['execution_time'].dt.dayofweek
        
        return df
    
    async def _calculate_quality_metrics(
        self, 
        executions_df: pd.DataFrame, 
        benchmark_type: str
    ) -> ExecutionQualityMetrics:
        """Calculate comprehensive execution quality metrics."""
        
        # Implementation shortfall components
        market_impact = executions_df['price_deviation'].abs().mean()
        timing_cost = executions_df.groupby('symbol')['price_deviation'].std().mean()
        opportunity_cost = self._calculate_opportunity_cost(executions_df)
        
        # Execution costs
        avg_cost_bps = executions_df['cost_bps'].mean()
        cost_volatility = executions_df['cost_bps'].std()
        
        # Fill rates and completion metrics
        fill_rate = len(executions_df) / len(executions_df)  # Assuming all provided are filled
        completion_rate = executions_df.groupby('symbol')['quantity'].sum().mean()
        
        # Quality scoring
        quality_score = self._calculate_quality_score(
            market_impact, avg_cost_bps, cost_volatility
        )
        
        return ExecutionQualityMetrics(
            implementation_shortfall=market_impact + timing_cost + opportunity_cost,
            market_impact_bps=market_impact * 10000,
            timing_cost_bps=timing_cost * 10000,
            opportunity_cost_bps=opportunity_cost * 10000,
            average_cost_bps=avg_cost_bps,
            cost_volatility_bps=cost_volatility,
            fill_rate=fill_rate,
            completion_rate=completion_rate,
            quality_score=quality_score,
            benchmark_type=benchmark_type
        )
    
    async def _analyze_implementation_shortfall(
        self, 
        executions_df: pd.DataFrame,
        benchmark_prices: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Analyze implementation shortfall components."""
        
        shortfall_analysis = {
            'total_shortfall_bps': 0.0,
            'components': {
                'market_impact': 0.0,
                'timing_cost': 0.0,
                'opportunity_cost': 0.0,
                'fees_and_commissions': 0.0
            },
            'by_symbol': {},
            'by_venue': {},
            'by_time_period': {}
        }
        
        # Calculate by symbol
        for symbol in executions_df['symbol'].unique():
            symbol_data = executions_df[executions_df['symbol'] == symbol]
            symbol_shortfall = symbol_data['price_deviation'].mean() * 10000
            
            shortfall_analysis['by_symbol'][symbol] = {
                'shortfall_bps': symbol_shortfall,
                'trade_count': len(symbol_data),
                'total_value': symbol_data['market_value'].sum()
            }
        
        # Calculate by venue
        venue_analysis = executions_df.groupby('venue').agg({
            'price_deviation': 'mean',
            'cost_bps': 'mean',
            'market_value': 'sum'
        }).to_dict('index')
        
        for venue, metrics in venue_analysis.items():
            shortfall_analysis['by_venue'][venue] = {
                'shortfall_bps': metrics['price_deviation'] * 10000,
                'avg_cost_bps': metrics['cost_bps'],
                'total_value': metrics['market_value']
            }
        
        # Time-based analysis
        time_analysis = executions_df.groupby('execution_hour').agg({
            'price_deviation': 'mean',
            'cost_bps': 'mean'
        }).to_dict('index')
        
        shortfall_analysis['by_time_period'] = {
            f"hour_{hour}": {
                'shortfall_bps': metrics['price_deviation'] * 10000,
                'avg_cost_bps': metrics['cost_bps']
            }
            for hour, metrics in time_analysis.items()
        }
        
        return shortfall_analysis
    
    async def _analyze_execution_patterns(self, executions_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify patterns in execution data."""
        
        patterns = {
            'temporal_patterns': {},
            'venue_patterns': {},
            'size_patterns': {},
            'volatility_patterns': {}
        }
        
        # Temporal patterns
        hourly_performance = executions_df.groupby('execution_hour').agg({
            'cost_bps': ['mean', 'std'],
            'price_deviation': 'mean'
        }).round(4)
        
        patterns['temporal_patterns'] = {
            'best_hours': hourly_performance[('cost_bps', 'mean')].nsmallest(3).to_dict(),
            'worst_hours': hourly_performance[('cost_bps', 'mean')].nlargest(3).to_dict(),
            'most_volatile_hours': hourly_performance[('cost_bps', 'std')].nlargest(3).to_dict()
        }
        
        # Venue patterns
        venue_performance = executions_df.groupby('venue').agg({
            'cost_bps': ['mean', 'count'],
            'price_deviation': 'mean'
        }).round(4)
        
        patterns['venue_patterns'] = {
            'best_venues': venue_performance[('cost_bps', 'mean')].nsmallest(3).to_dict(),
            'most_used_venues': venue_performance[('cost_bps', 'count')].nlargest(3).to_dict()
        }
        
        # Size patterns
        executions_df['size_category'] = pd.cut(
            executions_df['market_value'], 
            bins=5, 
            labels=['XS', 'S', 'M', 'L', 'XL']
        )
        
        size_performance = executions_df.groupby('size_category').agg({
            'cost_bps': ['mean', 'std'],
            'price_deviation': 'mean'
        }).round(4)
        
        patterns['size_patterns'] = size_performance.to_dict()
        
        return patterns
    
    async def _analyze_cost_attribution(self, executions_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cost attribution across different factors."""
        
        attribution = {
            'total_costs': executions_df['total_cost'].sum(),
            'average_cost_bps': executions_df['cost_bps'].mean(),
            'cost_breakdown': {
                'market_impact': 0.0,
                'spread_cost': 0.0,
                'commission': 0.0,
                'fees': 0.0,
                'timing_cost': 0.0
            },
            'variance_attribution': {},
            'performance_drivers': []
        }
        
        # Estimate cost components (simplified)
        total_market_value = executions_df['market_value'].sum()
        
        # Market impact estimation
        market_impact_cost = abs(executions_df['price_deviation'] * executions_df['market_value']).sum()
        attribution['cost_breakdown']['market_impact'] = (market_impact_cost / total_market_value) * 10000
        
        # Commission and fees (estimated from total cost)
        commission_fees = executions_df['total_cost'].sum() - market_impact_cost
        attribution['cost_breakdown']['commission'] = (commission_fees / total_market_value) * 10000
        
        # Performance drivers analysis
        if executions_df['cost_bps'].std() > 0:
            correlation_analysis = {}
            for col in ['market_value', 'execution_hour', 'price_deviation']:
                if col in executions_df.columns:
                    corr = executions_df['cost_bps'].corr(executions_df[col])
                    if abs(corr) > 0.3:  # Significant correlation
                        correlation_analysis[col] = corr
            
            attribution['performance_drivers'] = [
                f"{col}: {corr:.3f}" for col, corr in correlation_analysis.items()
            ]
        
        return attribution
    
    async def _generate_recommendations(
        self, 
        quality_metrics: ExecutionQualityMetrics,
        pattern_analysis: Dict[str, Any],
        cost_attribution: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        
        recommendations = []
        
        # Quality-based recommendations
        if quality_metrics.market_impact_bps > 10:
            recommendations.append(
                "Consider breaking large orders into smaller sizes to reduce market impact"
            )
        
        if quality_metrics.cost_volatility_bps > 5:
            recommendations.append(
                "Execution costs show high volatility - implement more consistent execution strategies"
            )
        
        # Pattern-based recommendations
        temporal_patterns = pattern_analysis.get('temporal_patterns', {})
        if 'best_hours' in temporal_patterns:
            best_hours = list(temporal_patterns['best_hours'].keys())[:2]
            recommendations.append(
                f"Optimize execution timing - best performance observed during hours {best_hours}"
            )
        
        # Venue-based recommendations
        venue_patterns = pattern_analysis.get('venue_patterns', {})
        if 'best_venues' in venue_patterns:
            best_venues = list(venue_patterns['best_venues'].keys())[:2]
            recommendations.append(
                f"Consider increasing allocation to top-performing venues: {best_venues}"
            )
        
        # Cost attribution recommendations
        if cost_attribution['cost_breakdown']['market_impact'] > 5:
            recommendations.append(
                "Market impact is a significant cost driver - consider algorithmic execution strategies"
            )
        
        # Default recommendation if no specific issues found
        if not recommendations:
            recommendations.append(
                "Execution performance is within acceptable ranges - continue monitoring"
            )
        
        return recommendations
    
    async def _compare_to_benchmarks(
        self, 
        executions_df: pd.DataFrame,
        benchmark_type: str
    ) -> Dict[str, Any]:
        """Compare execution performance to industry benchmarks."""
        
        comparison = {
            'benchmark_type': benchmark_type,
            'performance_vs_benchmark': {},
            'percentile_ranking': {},
            'industry_comparison': {}
        }
        
        # Simplified benchmark comparison
        avg_cost = executions_df['cost_bps'].mean()
        
        # Industry benchmarks (simplified)
        industry_benchmarks = {
            'TWAP': {'good': 3.0, 'average': 5.0, 'poor': 8.0},
            'VWAP': {'good': 2.5, 'average': 4.5, 'poor': 7.0},
            'Arrival': {'good': 4.0, 'average': 6.0, 'poor': 9.0}
        }
        
        if benchmark_type in industry_benchmarks:
            benchmarks = industry_benchmarks[benchmark_type]
            
            if avg_cost <= benchmarks['good']:
                performance = 'Excellent'
                percentile = 90
            elif avg_cost <= benchmarks['average']:
                performance = 'Good'
                percentile = 70
            elif avg_cost <= benchmarks['poor']:
                performance = 'Average'
                percentile = 50
            else:
                performance = 'Below Average'
                percentile = 25
            
            comparison['performance_vs_benchmark'] = {
                'rating': performance,
                'cost_vs_good': avg_cost - benchmarks['good'],
                'cost_vs_average': avg_cost - benchmarks['average']
            }
            
            comparison['percentile_ranking'] = {
                'estimated_percentile': percentile,
                'interpretation': f"Better than {percentile}% of similar executions"
            }
        
        return comparison
    
    def _calculate_opportunity_cost(self, executions_df: pd.DataFrame) -> float:
        """Calculate opportunity cost from delayed execution."""
        # Simplified opportunity cost calculation
        # In practice, this would consider the price movement during execution delay
        return executions_df['price_deviation'].abs().mean() * 0.3  # Approximate component
    
    def _calculate_quality_score(
        self, 
        market_impact: float, 
        avg_cost_bps: float, 
        cost_volatility: float
    ) -> float:
        """Calculate overall execution quality score (0-100)."""
        
        # Normalize metrics to 0-1 scale
        impact_score = max(0, 1 - (market_impact * 100))  # Lower is better
        cost_score = max(0, 1 - (avg_cost_bps / 10))      # Lower is better
        consistency_score = max(0, 1 - (cost_volatility / 5))  # Lower is better
        
        # Weighted average
        quality_score = (
            impact_score * 0.4 + 
            cost_score * 0.4 + 
            consistency_score * 0.2
        ) * 100
        
        return min(100, max(0, quality_score))


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from datetime import datetime, timedelta
    
    async def test_execution_analyzer():
        """Test the execution analyzer with sample data."""
        
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
            )
        ]
        
        # Create analysis request
        request = ExecutionAnalysisRequest(
            request_id="analysis_001",
            executions=sample_executions,
            benchmark_type="VWAP",
            analysis_period_days=30
        )
        
        # Initialize analyzer and run analysis
        analyzer = ExecutionAnalyzer()
        result = await analyzer.analyze_execution(request)
        
        print("=== Execution Analysis Results ===")
        print(f"Quality Score: {result.quality_metrics.quality_score:.2f}")
        print(f"Average Cost: {result.quality_metrics.average_cost_bps:.2f} bps")
        print(f"Market Impact: {result.quality_metrics.market_impact_bps:.2f} bps")
        print(f"Implementation Shortfall: {result.quality_metrics.implementation_shortfall:.4f}")
        print("\nRecommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"{i}. {rec}")
    
    # Run the test
    asyncio.run(test_execution_analyzer())
