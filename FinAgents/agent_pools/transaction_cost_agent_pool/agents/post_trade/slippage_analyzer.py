"""
Post-Trade Slippage Analysis Agent

This agent analyzes slippage patterns in executed trades to identify
causes and provide recommendations for minimizing future slippage.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ...schema.execution_schema import TradeExecution
from ...schema.cost_models import TransactionCostBreakdown


logger = logging.getLogger(__name__)


@dataclass
class SlippageAnalysisResult:
    """Result of slippage analysis."""
    analysis_id: str
    timestamp: datetime
    total_slippage_bps: float
    average_slippage_bps: float
    slippage_volatility: float
    worst_slippage_trades: List[Dict[str, Any]]
    slippage_by_symbol: Dict[str, float]
    slippage_by_venue: Dict[str, float]
    slippage_by_time: Dict[str, float]
    contributing_factors: List[str]
    recommendations: List[str]
    risk_metrics: Dict[str, float]


class SlippageAnalyzer:
    """
    Analyzes slippage in trade executions to identify patterns and optimization opportunities.
    
    This agent provides comprehensive slippage analysis including:
    - Price slippage measurement and attribution
    - Timing slippage analysis
    - Volume-weighted slippage assessment
    - Market condition impact analysis
    - Venue and routing optimization insights
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the slippage analyzer.
        
        Args:
            config: Configuration parameters for the analyzer
        """
        self.config = config or {}
        self.analysis_cache = {}
        self.market_data_cache = {}
        
        # Analysis parameters
        self.slippage_thresholds = self.config.get('slippage_thresholds', {
            'low': 2.0,      # bps
            'medium': 5.0,   # bps
            'high': 10.0,    # bps
            'extreme': 20.0  # bps
        })
        
        self.time_buckets = self.config.get('time_buckets', {
            'market_open': (9, 10),
            'morning': (10, 12),
            'midday': (12, 14),
            'afternoon': (14, 16),
            'market_close': (16, 17)
        })
        
        self.volume_percentiles = self.config.get('volume_percentiles', [25, 50, 75, 90])
        
    async def analyze_slippage(
        self, 
        executions: List[TradeExecution],
        reference_prices: Optional[Dict[str, Dict[str, float]]] = None,
        analysis_id: Optional[str] = None
    ) -> SlippageAnalysisResult:
        """
        Perform comprehensive slippage analysis.
        
        Args:
            executions: List of trade executions to analyze
            reference_prices: Reference prices for slippage calculation
            analysis_id: Unique identifier for this analysis
            
        Returns:
            Comprehensive slippage analysis results
        """
        try:
            analysis_id = analysis_id or f"slippage_analysis_{datetime.utcnow().isoformat()}"
            logger.info(f"Starting slippage analysis {analysis_id} for {len(executions)} trades")
            
            # Prepare execution data
            executions_df = self._prepare_execution_data(executions, reference_prices)
            
            # Calculate slippage metrics
            slippage_metrics = await self._calculate_slippage_metrics(executions_df)
            
            # Analyze slippage patterns
            pattern_analysis = await self._analyze_slippage_patterns(executions_df)
            
            # Identify contributing factors
            contributing_factors = await self._identify_contributing_factors(executions_df)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(executions_df)
            
            # Generate recommendations
            recommendations = await self._generate_slippage_recommendations(
                executions_df, slippage_metrics, pattern_analysis
            )
            
            # Identify worst performing trades
            worst_trades = self._identify_worst_trades(executions_df, top_n=10)
            
            # Compile results
            result = SlippageAnalysisResult(
                analysis_id=analysis_id,
                timestamp=datetime.utcnow(),
                total_slippage_bps=slippage_metrics['total_slippage_bps'],
                average_slippage_bps=slippage_metrics['average_slippage_bps'],
                slippage_volatility=slippage_metrics['slippage_volatility'],
                worst_slippage_trades=worst_trades,
                slippage_by_symbol=pattern_analysis['by_symbol'],
                slippage_by_venue=pattern_analysis['by_venue'],
                slippage_by_time=pattern_analysis['by_time'],
                contributing_factors=contributing_factors,
                recommendations=recommendations,
                risk_metrics=risk_metrics
            )
            
            logger.info(f"Slippage analysis {analysis_id} completed")
            return result
            
        except Exception as e:
            logger.error(f"Error in slippage analysis: {str(e)}")
            raise
    
    def _prepare_execution_data(
        self, 
        executions: List[TradeExecution],
        reference_prices: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        """Prepare execution data for slippage analysis."""
        
        data = []
        for execution in executions:
            # Use benchmark price or reference price
            reference_price = execution.benchmark_price
            if reference_prices and execution.symbol in reference_prices:
                ref_data = reference_prices[execution.symbol]
                if 'arrival_price' in ref_data:
                    reference_price = ref_data['arrival_price']
            
            # Calculate slippage
            if execution.side.upper() == 'BUY':
                slippage = execution.executed_price - reference_price
            else:
                slippage = reference_price - execution.executed_price
            
            slippage_bps = (slippage / reference_price) * 10000 if reference_price != 0 else 0
            
            data.append({
                'execution_id': execution.execution_id,
                'symbol': execution.symbol,
                'side': execution.side,
                'quantity': execution.quantity,
                'executed_price': execution.executed_price,
                'reference_price': reference_price,
                'slippage': slippage,
                'slippage_bps': slippage_bps,
                'execution_time': execution.execution_time,
                'venue': execution.venue,
                'order_type': execution.order_type,
                'market_value': execution.quantity * execution.executed_price,
                'total_cost': execution.total_cost
            })
        
        df = pd.DataFrame(data)
        
        # Add derived fields
        df['execution_hour'] = df['execution_time'].dt.hour
        df['execution_day'] = df['execution_time'].dt.dayofweek
        df['time_bucket'] = df['execution_hour'].apply(self._categorize_time)
        df['slippage_category'] = df['slippage_bps'].apply(self._categorize_slippage)
        df['volume_percentile'] = pd.qcut(df['quantity'], q=4, labels=['Small', 'Medium', 'Large', 'XLarge'])
        
        return df
    
    async def _calculate_slippage_metrics(self, executions_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive slippage metrics."""
        
        metrics = {}
        
        # Basic slippage metrics
        metrics['total_slippage_bps'] = executions_df['slippage_bps'].sum()
        metrics['average_slippage_bps'] = executions_df['slippage_bps'].mean()
        metrics['median_slippage_bps'] = executions_df['slippage_bps'].median()
        metrics['slippage_volatility'] = executions_df['slippage_bps'].std()
        
        # Percentile metrics
        for percentile in [75, 90, 95, 99]:
            metrics[f'slippage_{percentile}th_percentile'] = executions_df['slippage_bps'].quantile(percentile/100)
        
        # Value-weighted metrics
        total_value = executions_df['market_value'].sum()
        if total_value > 0:
            metrics['value_weighted_slippage_bps'] = (
                (executions_df['slippage_bps'] * executions_df['market_value']).sum() / total_value
            )
        
        # Positive vs negative slippage
        positive_slippage = executions_df[executions_df['slippage_bps'] > 0]
        negative_slippage = executions_df[executions_df['slippage_bps'] < 0]
        
        metrics['positive_slippage_rate'] = len(positive_slippage) / len(executions_df) if len(executions_df) > 0 else 0
        metrics['average_positive_slippage_bps'] = positive_slippage['slippage_bps'].mean() if len(positive_slippage) > 0 else 0
        metrics['average_negative_slippage_bps'] = negative_slippage['slippage_bps'].mean() if len(negative_slippage) > 0 else 0
        
        # Extreme slippage metrics
        extreme_threshold = self.slippage_thresholds['extreme']
        extreme_slippage = executions_df[executions_df['slippage_bps'].abs() > extreme_threshold]
        metrics['extreme_slippage_rate'] = len(extreme_slippage) / len(executions_df) if len(executions_df) > 0 else 0
        
        return metrics
    
    async def _analyze_slippage_patterns(self, executions_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze slippage patterns across different dimensions."""
        
        patterns = {
            'by_symbol': {},
            'by_venue': {},
            'by_time': {},
            'by_order_type': {},
            'by_volume': {},
            'by_side': {}
        }
        
        # By symbol
        symbol_analysis = executions_df.groupby('symbol').agg({
            'slippage_bps': ['mean', 'std', 'count'],
            'market_value': 'sum'
        }).round(4)
        
        for symbol in symbol_analysis.index:
            patterns['by_symbol'][symbol] = {
                'avg_slippage_bps': symbol_analysis.loc[symbol, ('slippage_bps', 'mean')],
                'slippage_volatility': symbol_analysis.loc[symbol, ('slippage_bps', 'std')],
                'trade_count': symbol_analysis.loc[symbol, ('slippage_bps', 'count')],
                'total_value': symbol_analysis.loc[symbol, ('market_value', 'sum')]
            }
        
        # By venue
        venue_analysis = executions_df.groupby('venue').agg({
            'slippage_bps': ['mean', 'std', 'count']
        }).round(4)
        
        for venue in venue_analysis.index:
            patterns['by_venue'][venue] = {
                'avg_slippage_bps': venue_analysis.loc[venue, ('slippage_bps', 'mean')],
                'slippage_volatility': venue_analysis.loc[venue, ('slippage_bps', 'std')],
                'trade_count': venue_analysis.loc[venue, ('slippage_bps', 'count')]
            }
        
        # By time bucket
        time_analysis = executions_df.groupby('time_bucket').agg({
            'slippage_bps': ['mean', 'std', 'count']
        }).round(4)
        
        for time_bucket in time_analysis.index:
            patterns['by_time'][time_bucket] = {
                'avg_slippage_bps': time_analysis.loc[time_bucket, ('slippage_bps', 'mean')],
                'slippage_volatility': time_analysis.loc[time_bucket, ('slippage_bps', 'std')],
                'trade_count': time_analysis.loc[time_bucket, ('slippage_bps', 'count')]
            }
        
        # By order type
        order_type_analysis = executions_df.groupby('order_type').agg({
            'slippage_bps': ['mean', 'std', 'count']
        }).round(4)
        
        for order_type in order_type_analysis.index:
            patterns['by_order_type'][order_type] = {
                'avg_slippage_bps': order_type_analysis.loc[order_type, ('slippage_bps', 'mean')],
                'slippage_volatility': order_type_analysis.loc[order_type, ('slippage_bps', 'std')],
                'trade_count': order_type_analysis.loc[order_type, ('slippage_bps', 'count')]
            }
        
        # By volume percentile
        volume_analysis = executions_df.groupby('volume_percentile').agg({
            'slippage_bps': ['mean', 'std', 'count']
        }).round(4)
        
        for volume_cat in volume_analysis.index:
            patterns['by_volume'][str(volume_cat)] = {
                'avg_slippage_bps': volume_analysis.loc[volume_cat, ('slippage_bps', 'mean')],
                'slippage_volatility': volume_analysis.loc[volume_cat, ('slippage_bps', 'std')],
                'trade_count': volume_analysis.loc[volume_cat, ('slippage_bps', 'count')]
            }
        
        # By side
        side_analysis = executions_df.groupby('side').agg({
            'slippage_bps': ['mean', 'std', 'count']
        }).round(4)
        
        for side in side_analysis.index:
            patterns['by_side'][side] = {
                'avg_slippage_bps': side_analysis.loc[side, ('slippage_bps', 'mean')],
                'slippage_volatility': side_analysis.loc[side, ('slippage_bps', 'std')],
                'trade_count': side_analysis.loc[side, ('slippage_bps', 'count')]
            }
        
        return patterns
    
    async def _identify_contributing_factors(self, executions_df: pd.DataFrame) -> List[str]:
        """Identify factors contributing to slippage."""
        
        factors = []
        
        # Correlation analysis
        numeric_cols = ['quantity', 'execution_hour', 'market_value']
        correlations = {}
        
        for col in numeric_cols:
            if col in executions_df.columns:
                corr = executions_df['slippage_bps'].corr(executions_df[col])
                if abs(corr) > 0.3:  # Significant correlation
                    correlations[col] = corr
        
        # Market timing factors
        if 'execution_hour' in correlations:
            if correlations['execution_hour'] > 0.3:
                factors.append("Later execution times correlate with higher slippage")
            elif correlations['execution_hour'] < -0.3:
                factors.append("Earlier execution times correlate with higher slippage")
        
        # Volume factors
        if 'quantity' in correlations:
            if correlations['quantity'] > 0.3:
                factors.append("Larger order sizes correlate with higher slippage")
            elif correlations['quantity'] < -0.3:
                factors.append("Smaller order sizes correlate with higher slippage")
        
        # Order type analysis
        if 'order_type' in executions_df.columns:
            order_type_slippage = executions_df.groupby('order_type')['slippage_bps'].mean()
            if 'MARKET' in order_type_slippage.index and 'LIMIT' in order_type_slippage.index:
                market_slippage = order_type_slippage['MARKET']
                limit_slippage = order_type_slippage['LIMIT']
                if market_slippage > limit_slippage + 2:
                    factors.append("Market orders show significantly higher slippage than limit orders")
        
        # Venue concentration
        venue_counts = executions_df['venue'].value_counts()
        if len(venue_counts) > 1:
            dominant_venue_pct = venue_counts.iloc[0] / len(executions_df)
            if dominant_venue_pct > 0.8:
                factors.append("High concentration in single venue may limit execution quality")
        
        # Time concentration
        time_concentration = executions_df['time_bucket'].value_counts()
        if len(time_concentration) > 1:
            dominant_time_pct = time_concentration.iloc[0] / len(executions_df)
            if dominant_time_pct > 0.6:
                factors.append("High concentration in specific time periods may impact slippage")
        
        # Extreme slippage events
        extreme_threshold = self.slippage_thresholds['extreme']
        extreme_events = executions_df[executions_df['slippage_bps'].abs() > extreme_threshold]
        if len(extreme_events) > 0:
            extreme_rate = len(extreme_events) / len(executions_df)
            if extreme_rate > 0.05:  # More than 5%
                factors.append(f"High rate of extreme slippage events ({extreme_rate:.1%})")
        
        return factors if factors else ["No significant contributing factors identified"]
    
    async def _calculate_risk_metrics(self, executions_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate slippage risk metrics."""
        
        risk_metrics = {}
        
        # Value at Risk (VaR) - 95th percentile
        risk_metrics['slippage_var_95'] = executions_df['slippage_bps'].quantile(0.95)
        risk_metrics['slippage_var_99'] = executions_df['slippage_bps'].quantile(0.99)
        
        # Expected Shortfall (Conditional VaR)
        var_95 = risk_metrics['slippage_var_95']
        tail_losses = executions_df[executions_df['slippage_bps'] > var_95]['slippage_bps']
        risk_metrics['expected_shortfall_95'] = tail_losses.mean() if len(tail_losses) > 0 else 0
        
        # Maximum slippage
        risk_metrics['max_slippage_bps'] = executions_df['slippage_bps'].max()
        risk_metrics['min_slippage_bps'] = executions_df['slippage_bps'].min()
        
        # Volatility metrics
        risk_metrics['slippage_volatility'] = executions_df['slippage_bps'].std()
        risk_metrics['slippage_skewness'] = executions_df['slippage_bps'].skew()
        risk_metrics['slippage_kurtosis'] = executions_df['slippage_bps'].kurtosis()
        
        # Tracking error
        benchmark_slippage = 0  # Assuming zero as benchmark
        risk_metrics['tracking_error'] = ((executions_df['slippage_bps'] - benchmark_slippage) ** 2).mean() ** 0.5
        
        return risk_metrics
    
    async def _generate_slippage_recommendations(
        self, 
        executions_df: pd.DataFrame,
        slippage_metrics: Dict[str, float],
        pattern_analysis: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate recommendations to reduce slippage."""
        
        recommendations = []
        
        # Overall slippage level
        avg_slippage = slippage_metrics['average_slippage_bps']
        if avg_slippage > self.slippage_thresholds['high']:
            recommendations.append(
                f"Average slippage ({avg_slippage:.1f} bps) is high - consider algorithmic execution strategies"
            )
        
        # Volatility recommendations
        if slippage_metrics['slippage_volatility'] > 10:
            recommendations.append(
                "High slippage volatility detected - implement more consistent execution timing"
            )
        
        # Time-based recommendations
        time_patterns = pattern_analysis.get('by_time', {})
        if time_patterns:
            best_time = min(time_patterns.items(), key=lambda x: x[1]['avg_slippage_bps'])
            worst_time = max(time_patterns.items(), key=lambda x: x[1]['avg_slippage_bps'])
            
            if best_time[1]['avg_slippage_bps'] < worst_time[1]['avg_slippage_bps'] - 3:
                recommendations.append(
                    f"Consider executing more trades during {best_time[0]} period "
                    f"(avg slippage: {best_time[1]['avg_slippage_bps']:.1f} bps vs "
                    f"{worst_time[1]['avg_slippage_bps']:.1f} bps in {worst_time[0]})"
                )
        
        # Venue recommendations
        venue_patterns = pattern_analysis.get('by_venue', {})
        if len(venue_patterns) > 1:
            best_venue = min(venue_patterns.items(), key=lambda x: x[1]['avg_slippage_bps'])
            worst_venue = max(venue_patterns.items(), key=lambda x: x[1]['avg_slippage_bps'])
            
            if best_venue[1]['avg_slippage_bps'] < worst_venue[1]['avg_slippage_bps'] - 2:
                recommendations.append(
                    f"Consider routing more orders to {best_venue[0]} "
                    f"(avg slippage: {best_venue[1]['avg_slippage_bps']:.1f} bps)"
                )
        
        # Order type recommendations
        order_type_patterns = pattern_analysis.get('by_order_type', {})
        if 'MARKET' in order_type_patterns and 'LIMIT' in order_type_patterns:
            market_slippage = order_type_patterns['MARKET']['avg_slippage_bps']
            limit_slippage = order_type_patterns['LIMIT']['avg_slippage_bps']
            
            if market_slippage > limit_slippage + 3:
                recommendations.append(
                    f"Market orders show {market_slippage - limit_slippage:.1f} bps higher slippage "
                    "than limit orders - consider increasing limit order usage"
                )
        
        # Volume recommendations
        volume_patterns = pattern_analysis.get('by_volume', {})
        if 'XLarge' in volume_patterns and 'Small' in volume_patterns:
            large_slippage = volume_patterns['XLarge']['avg_slippage_bps']
            small_slippage = volume_patterns['Small']['avg_slippage_bps']
            
            if large_slippage > small_slippage + 5:
                recommendations.append(
                    f"Large orders show {large_slippage - small_slippage:.1f} bps higher slippage - "
                    "consider order splitting strategies"
                )
        
        # Extreme slippage recommendations
        extreme_rate = slippage_metrics.get('extreme_slippage_rate', 0)
        if extreme_rate > 0.1:  # More than 10%
            recommendations.append(
                f"High rate of extreme slippage events ({extreme_rate:.1%}) - "
                "implement better risk controls and pre-trade analysis"
            )
        
        return recommendations if recommendations else [
            "Slippage performance is within acceptable ranges - continue monitoring"
        ]
    
    def _identify_worst_trades(self, executions_df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
        """Identify trades with worst slippage performance."""
        
        worst_trades = executions_df.nlargest(top_n, 'slippage_bps')
        
        return [
            {
                'execution_id': row['execution_id'],
                'symbol': row['symbol'],
                'slippage_bps': row['slippage_bps'],
                'executed_price': row['executed_price'],
                'reference_price': row['reference_price'],
                'quantity': row['quantity'],
                'venue': row['venue'],
                'execution_time': row['execution_time'].isoformat(),
                'order_type': row['order_type']
            }
            for _, row in worst_trades.iterrows()
        ]
    
    def _categorize_time(self, hour: int) -> str:
        """Categorize execution time into buckets."""
        for bucket, (start, end) in self.time_buckets.items():
            if start <= hour < end:
                return bucket
        return 'other'
    
    def _categorize_slippage(self, slippage_bps: float) -> str:
        """Categorize slippage into severity levels."""
        abs_slippage = abs(slippage_bps)
        
        if abs_slippage <= self.slippage_thresholds['low']:
            return 'low'
        elif abs_slippage <= self.slippage_thresholds['medium']:
            return 'medium'
        elif abs_slippage <= self.slippage_thresholds['high']:
            return 'high'
        else:
            return 'extreme'


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from datetime import datetime, timedelta
    
    async def test_slippage_analyzer():
        """Test the slippage analyzer with sample data."""
        
        # Create sample executions with varying slippage
        sample_executions = [
            TradeExecution(
                execution_id="exec_001",
                symbol="AAPL",
                side="BUY",
                quantity=1000,
                executed_price=150.30,  # 10 bps slippage
                benchmark_price=150.15,
                execution_time=datetime.utcnow() - timedelta(hours=3),
                venue="NYSE",
                order_type="MARKET",
                total_cost=45.0
            ),
            TradeExecution(
                execution_id="exec_002",
                symbol="GOOGL",
                side="SELL",
                quantity=500,
                executed_price=2750.50,  # 18 bps slippage
                benchmark_price=2755.00,
                execution_time=datetime.utcnow() - timedelta(hours=2),
                venue="NASDAQ",
                order_type="MARKET",
                total_cost=82.5
            ),
            TradeExecution(
                execution_id="exec_003",
                symbol="MSFT",
                side="BUY",
                quantity=2000,
                executed_price=340.05,  # 1.5 bps slippage
                benchmark_price=340.00,
                execution_time=datetime.utcnow() - timedelta(hours=1),
                venue="ARCA",
                order_type="LIMIT",
                total_cost=68.0
            )
        ]
        
        # Initialize analyzer and run analysis
        analyzer = SlippageAnalyzer()
        result = await analyzer.analyze_slippage(sample_executions)
        
        print("=== Slippage Analysis Results ===")
        print(f"Analysis ID: {result.analysis_id}")
        print(f"Average Slippage: {result.average_slippage_bps:.2f} bps")
        print(f"Total Slippage: {result.total_slippage_bps:.2f} bps")
        print(f"Slippage Volatility: {result.slippage_volatility:.2f} bps")
        
        print("\n=== Slippage by Symbol ===")
        for symbol, slippage in result.slippage_by_symbol.items():
            print(f"{symbol}: {slippage['avg_slippage_bps']:.2f} bps")
        
        print("\n=== Contributing Factors ===")
        for factor in result.contributing_factors:
            print(f"- {factor}")
        
        print("\n=== Recommendations ===")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"{i}. {rec}")
        
        print("\n=== Worst Trades ===")
        for trade in result.worst_slippage_trades[:3]:  # Top 3
            print(f"- {trade['symbol']}: {trade['slippage_bps']:.2f} bps "
                  f"({trade['execution_id']})")
    
    # Run the test
    asyncio.run(test_slippage_analyzer())
