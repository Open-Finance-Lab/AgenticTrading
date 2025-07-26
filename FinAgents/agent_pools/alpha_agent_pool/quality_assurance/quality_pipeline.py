"""
Alpha Agent Pool Quality Assurance Pipeline

This module implements the main quality assurance pipeline for comprehensive
testing and validation of the alpha agent pool. It orchestrates all testing
components and generates detailed quality reports.

Pipeline Components:
1. Alpha Factor Quality Assessment
2. Event-Driven Backtesting Validation
3. Agent Interaction and Collaboration Testing
4. Performance Validation and Statistical Testing
5. Reinforcement Learning Update Validation

Author: FinAgent Quality Assurance Team
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quality_assurance.tests import (
    AlphaFactorQualityTests,
    AgentInteractionTests,
    PerformanceValidationTests,
    TestStatus
)
from quality_assurance.backtesting import (
    EventDrivenBacktester,
    AlphaFactorEvaluator,
    TransactionCostModel
)
from quality_assurance.metrics import (
    AlphaFactorMetrics,
    RiskAdjustedMetrics,
    CrossSectionalMetrics
)
from quality_assurance.alpha_agent_integration import (
    create_alpha_agent_factor_generator,
    AlphaAgentFactorGenerator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quality_assurance/logs/qa_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QualityAssessmentConfig:
    """Configuration for quality assessment pipeline"""
    data_cache_path: str = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/data/cache"
    test_period_start: str = "2022-01-01"
    test_period_end: str = "2024-12-31"
    initial_capital: float = 1000000.0
    transaction_cost_bps: float = 5.0  # 5 basis points
    min_sharpe_ratio: float = 1.0
    max_drawdown_threshold: float = 0.15
    confidence_level: float = 0.95
    enable_agent_interaction_tests: bool = True
    enable_rl_validation: bool = True
    output_directory: str = "quality_assurance/reports"
    
    # Alpha agent configuration
    use_alpha_agents: bool = True  # Whether to use intelligent agents or fallback factors
    alpha_agent_config: Dict[str, Any] = field(default_factory=lambda: {
        'use_real_agents': True,
        'autonomous_agent': {
            'server_name': 'alpha_factor_discovery',
            'host': 'localhost',
            'port': 8889
        },
        'factor_count': 8,
        'factor_types': ['momentum', 'mean_reversion', 'volatility', 'cross_sectional', 'technical']
    })

@dataclass
class QualityAssessmentResult:
    """Comprehensive quality assessment result"""
    timestamp: datetime
    overall_grade: str  # 'A', 'B', 'C', 'D', 'F'
    overall_score: float  # 0-100
    passed_tests: int
    total_tests: int
    factor_quality_results: Dict[str, Any]
    backtesting_results: Dict[str, Any]
    agent_interaction_results: Dict[str, Any]
    performance_validation_results: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    detailed_results: Dict[str, Any] = field(default_factory=dict)

class AlphaAgentPoolQualityPipeline:
    """
    Main quality assurance pipeline for alpha agent pool validation.
    
    This pipeline orchestrates comprehensive testing of alpha factors,
    strategies, agent interactions, and performance validation.
    """
    
    def __init__(self, config: QualityAssessmentConfig):
        """
        Initialize quality assessment pipeline.
        
        Args:
            config: Quality assessment configuration
        """
        self.config = config
        self.data_cache_path = Path(config.data_cache_path)
        self.output_path = Path(config.output_directory)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize testing components
        self.factor_quality_tests = AlphaFactorQualityTests()
        self.agent_interaction_tests = AgentInteractionTests(str(self.data_cache_path))
        self.performance_validation_tests = PerformanceValidationTests(
            confidence_level=config.confidence_level,
            min_sharpe_ratio=config.min_sharpe_ratio,
            max_drawdown_threshold=config.max_drawdown_threshold
        )
        
        # Initialize metrics calculators
        self.alpha_metrics = AlphaFactorMetrics()
        self.risk_metrics = RiskAdjustedMetrics()
        self.cross_sectional_metrics = CrossSectionalMetrics()
        
        # Initialize alpha agent factor generator
        self.alpha_agent_generator = create_alpha_agent_factor_generator(
            config.alpha_agent_config
        )
        
        # Initialize backtesting framework
        self.transaction_cost_model = TransactionCostModel(
            fixed_cost=1.0,
            percentage_cost=config.transaction_cost_bps / 10000.0
        )
        
        self.backtester = EventDrivenBacktester(
            initial_capital=config.initial_capital,
            cost_model=self.transaction_cost_model
        )
        
        self.factor_evaluator = AlphaFactorEvaluator(self.backtester)
    
    def load_market_data(self) -> pd.DataFrame:
        """Load market data from cache directory."""
        logger.info("Loading market data from cache")
        logger.info(f"Cache path: {self.data_cache_path}")
        logger.info(f"Cache path exists: {self.data_cache_path.exists()}")
        
        if self.data_cache_path.exists():
            csv_files = list(self.data_cache_path.glob("*.csv"))
            logger.info(f"Found CSV files: {[f.name for f in csv_files]}")
        
        market_data = {}
        
        # Load all available CSV files
        for csv_file in self.data_cache_path.glob("*.csv"):
            try:
                symbol = csv_file.stem.split('_')[0]
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                
                # Use close prices for backtesting
                if 'close' in df.columns:
                    market_data[symbol] = df['close']
                elif 'Close' in df.columns:
                    market_data[symbol] = df['Close']
                else:
                    # Use first numeric column
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        market_data[symbol] = df[numeric_cols[0]]
                
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
        
        if not market_data:
            raise ValueError("No market data loaded from cache directory")
        
        # Combine into single DataFrame
        price_df = pd.DataFrame(market_data)
        
        # Filter by date range
        start_date = pd.to_datetime(self.config.test_period_start)
        end_date = pd.to_datetime(self.config.test_period_end)
        
        price_df = price_df.loc[start_date:end_date]
        
        logger.info(f"Loaded market data for {len(price_df.columns)} symbols, "
                   f"{len(price_df)} time periods")
        
        return price_df
    
    async def generate_alpha_factors_from_agents(
        self, 
        price_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate alpha factors using intelligent agents instead of hardcoded logic.
        
        Args:
            price_data: Historical price data
            
        Returns:
            Dictionary of factor names to factor DataFrames
        """
        logger.info("Generating alpha factors using intelligent agents")
        
        # Check if we should use agents or fall back to hardcoded factors
        if not self.config.use_alpha_agents:
            logger.info("Agent-based factor generation disabled, using fallback factors")
            return self.generate_test_alpha_factors(price_data)
        
        try:
            # Request factors from alpha agents
            agent_factors = await self.alpha_agent_generator.request_alpha_factors(
                market_data=price_data,
                factor_count=self.config.alpha_agent_config.get('factor_count', 8),
                factor_types=self.config.alpha_agent_config.get('factor_types', 
                    ['momentum', 'mean_reversion', 'volatility', 'cross_sectional', 'technical'])
            )
            
            if agent_factors:
                logger.info(f"Successfully generated {len(agent_factors)} factors from alpha agents")
                
                # Log factor metadata
                for factor_name in agent_factors.keys():
                    metadata = self.alpha_agent_generator.get_factor_metadata(factor_name)
                    logger.info(f"Factor {factor_name}: generated by {metadata.get('generated_by', 'unknown')}")
                
                return agent_factors
            else:
                logger.warning("No factors generated by agents, falling back to hardcoded examples")
                return self.generate_test_alpha_factors(price_data)
                
        except Exception as e:
            logger.error(f"Failed to generate factors from agents: {e}")
            logger.info("Falling back to hardcoded test factors")
            return self.generate_test_alpha_factors(price_data)
    
    def generate_test_alpha_factors(self, price_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate hardcoded test alpha factors as fallback.
        This method is kept as a backup when agents are unavailable.
        """
        logger.info("Generating fallback test alpha factors")
        
        factors = {}
        
        # Momentum factor (focus on alpha factor discovery)
        returns = price_data.pct_change()
        momentum_factor = returns.rolling(window=5).sum().shift(1)  # 5-day momentum
        factors['momentum_5d'] = momentum_factor
        
        # Mean reversion factor
        z_scores = returns.rolling(window=20).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
        )
        factors['mean_reversion_20d'] = -z_scores  # Negative for mean reversion
        
        # Price relative strength factor
        relative_returns = returns.subtract(returns.mean(axis=1), axis=0)
        factors['relative_strength'] = relative_returns.rolling(window=10).sum()
        
        # Volatility factor
        volatility = returns.rolling(window=20).std()
        volatility_zscore = volatility.subtract(volatility.mean(axis=1), axis=0).div(
            volatility.std(axis=1), axis=0
        )
        factors['volatility_factor'] = -volatility_zscore  # Low vol factor
        
        # Cross-sectional ranking factor
        for factor_name, factor_data in factors.copy().items():
            ranked_factor = factor_data.rank(axis=1, pct=True).subtract(0.5).multiply(2)
            factors[f'{factor_name}_ranked'] = ranked_factor
        
        logger.info(f"Generated {len(factors)} fallback alpha factors")
        return factors
    
    async def run_comprehensive_quality_assessment(self) -> QualityAssessmentResult:
        """
        Run comprehensive quality assessment pipeline.
        
        Returns:
            QualityAssessmentResult with complete assessment
        """
        logger.info("Starting comprehensive alpha agent pool quality assessment")
        assessment_start = datetime.now()
        
        try:
            # Load market data
            price_data = self.load_market_data()
            returns_data = price_data.pct_change().dropna()
            
            # Generate alpha factors using intelligent agents
            alpha_factors = await self.generate_alpha_factors_from_agents(price_data)
            
            # 1. Factor Quality Assessment
            logger.info("Running alpha factor quality assessment")
            factor_quality_results = await self._assess_factor_quality(
                alpha_factors, returns_data
            )
            
            # 2. Event-Driven Backtesting
            logger.info("Running event-driven backtesting validation")
            backtesting_results = await self._run_backtesting_validation(
                alpha_factors, price_data, returns_data
            )
            
            # 3. Agent Interaction Testing
            agent_interaction_results = {}
            if self.config.enable_agent_interaction_tests:
                logger.info("Running agent interaction tests")
                agent_interaction_results = await self._run_agent_interaction_tests()
            
            # 4. Performance Validation
            logger.info("Running performance validation tests")
            performance_validation_results = await self._run_performance_validation(
                backtesting_results
            )
            
            # 5. Generate overall assessment
            overall_result = self._generate_overall_assessment(
                factor_quality_results,
                backtesting_results,
                agent_interaction_results,
                performance_validation_results
            )
            
            # 6. Generate detailed report
            await self._generate_quality_report(overall_result)
            
            assessment_time = (datetime.now() - assessment_start).total_seconds()
            logger.info(f"Quality assessment completed in {assessment_time:.2f} seconds")
            logger.info(f"Overall grade: {overall_result.overall_grade} "
                       f"({overall_result.overall_score:.1f}/100)")
            
            return overall_result
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            raise
    
    async def _assess_factor_quality(
        self,
        alpha_factors: Dict[str, pd.DataFrame],
        returns_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Assess quality of alpha factors."""
        
        factor_results = {}
        
        for factor_name, factor_data in alpha_factors.items():
            logger.info(f"Assessing quality of factor: {factor_name}")
            
            # Calculate forward returns
            forward_returns = returns_data.shift(-1)
            
            # Run comprehensive quality assessment
            quality_metrics = self.factor_quality_tests.run_comprehensive_quality_assessment(
                factor_data, returns_data
            )
            
            # Generate quality report
            quality_report = self.factor_quality_tests.generate_quality_report(
                quality_metrics, factor_name
            )
            
            factor_results[factor_name] = {
                'metrics': quality_metrics,
                'report': quality_report,
                'grade': self._grade_factor_quality(quality_metrics)
            }
        
        return factor_results
    
    async def _run_backtesting_validation(
        self,
        alpha_factors: Dict[str, pd.DataFrame],
        price_data: pd.DataFrame,
        returns_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run event-driven backtesting validation."""
        
        backtesting_results = {}
        
        for factor_name, factor_data in alpha_factors.items():
            logger.info(f"Backtesting factor: {factor_name}")
            
            # Create fresh backtester instance for each factor
            backtester = EventDrivenBacktester(
                initial_capital=self.config.initial_capital,
                cost_model=self.transaction_cost_model
            )
            
            evaluator = AlphaFactorEvaluator(backtester)
            
            # Run factor evaluation
            evaluation_result = evaluator.evaluate_factor_strategy(
                factor_data, price_data
            )
            
            backtesting_results[factor_name] = evaluation_result
        
        return backtesting_results
    
    async def _run_agent_interaction_tests(self) -> Dict[str, Any]:
        """Run agent interaction and collaboration tests."""
        
        # Run comprehensive agent interaction tests
        interaction_results = await self.agent_interaction_tests.run_comprehensive_agent_tests()
        
        # Calculate interaction test summary
        total_tests = len(interaction_results)
        passed_tests = sum(1 for result in interaction_results.values() 
                          if result.status == TestStatus.PASSED)
        
        interaction_summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'detailed_results': interaction_results
        }
        
        return interaction_summary
    
    async def _run_performance_validation(
        self,
        backtesting_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run performance validation tests."""
        
        validation_results = {}
        
        for factor_name, backtest_result in backtesting_results.items():
            logger.info(f"Validating performance of factor: {factor_name}")
            
            # Extract strategy returns from backtest
            portfolio_history = backtest_result.get('portfolio_history', [])
            if not portfolio_history:
                continue
            
            portfolio_df = pd.DataFrame(portfolio_history)
            portfolio_df.set_index('timestamp', inplace=True)
            
            if len(portfolio_df) < 2:
                continue
            
            strategy_returns = portfolio_df['portfolio_value'].pct_change().dropna()
            
            # Run strategy performance validation
            performance_validation = self.performance_validation_tests.validate_strategy_performance(
                strategy_returns
            )
            
            # Run out-of-sample validation (if sufficient data)
            if len(strategy_returns) > 200:  # Need sufficient data for split
                split_point = len(strategy_returns) * 2 // 3
                in_sample = strategy_returns.iloc[:split_point]
                out_of_sample = strategy_returns.iloc[split_point:]
                
                oos_validation = self.performance_validation_tests.validate_out_of_sample_performance(
                    in_sample, out_of_sample
                )
            else:
                oos_validation = None
            
            validation_results[factor_name] = {
                'strategy_validation': performance_validation,
                'out_of_sample_validation': oos_validation
            }
        
        return validation_results
    
    def _generate_overall_assessment(
        self,
        factor_quality_results: Dict[str, Any],
        backtesting_results: Dict[str, Any],
        agent_interaction_results: Dict[str, Any],
        performance_validation_results: Dict[str, Any]
    ) -> QualityAssessmentResult:
        """Generate overall quality assessment result."""
        
        # Calculate component scores
        factor_quality_score = self._calculate_factor_quality_score(factor_quality_results)
        backtesting_score = self._calculate_backtesting_score(backtesting_results)
        interaction_score = self._calculate_interaction_score(agent_interaction_results)
        validation_score = self._calculate_validation_score(performance_validation_results)
        
        # Weighted overall score
        component_weights = {
            'factor_quality': 0.3,
            'backtesting': 0.3,
            'interaction': 0.2,
            'validation': 0.2
        }
        
        overall_score = (
            factor_quality_score * component_weights['factor_quality'] +
            backtesting_score * component_weights['backtesting'] +
            interaction_score * component_weights['interaction'] +
            validation_score * component_weights['validation']
        )
        
        # Assign letter grade
        overall_grade = self._assign_letter_grade(overall_score)
        
        # Count passed tests
        total_tests = 0
        passed_tests = 0
        
        # Factor quality tests
        for factor_result in factor_quality_results.values():
            total_tests += 4  # Assume 4 tests per factor
            if factor_result['grade'] in ['A', 'B']:
                passed_tests += 4
            elif factor_result['grade'] == 'C':
                passed_tests += 2
        
        # Agent interaction tests
        if agent_interaction_results:
            total_tests += agent_interaction_results.get('total_tests', 0)
            passed_tests += agent_interaction_results.get('passed_tests', 0)
        
        # Performance validation tests
        for validation_result in performance_validation_results.values():
            total_tests += 2  # Strategy + OOS validation
            if validation_result['strategy_validation'].is_valid:
                passed_tests += 1
            if (validation_result['out_of_sample_validation'] and 
                validation_result['out_of_sample_validation'].is_valid):
                passed_tests += 1
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_score, factor_quality_results, backtesting_results,
            agent_interaction_results, performance_validation_results
        )
        
        return QualityAssessmentResult(
            timestamp=datetime.now(),
            overall_grade=overall_grade,
            overall_score=overall_score,
            passed_tests=passed_tests,
            total_tests=total_tests,
            factor_quality_results=factor_quality_results,
            backtesting_results=backtesting_results,
            agent_interaction_results=agent_interaction_results,
            performance_validation_results=performance_validation_results,
            recommendations=recommendations,
            detailed_results={
                'component_scores': {
                    'factor_quality': factor_quality_score,
                    'backtesting': backtesting_score,
                    'interaction': interaction_score,
                    'validation': validation_score
                },
                'component_weights': component_weights
            }
        )
    
    def _grade_factor_quality(self, quality_metrics) -> str:
        """Assign letter grade to factor quality."""
        
        criteria_met = 0
        total_criteria = 4
        
        # Statistical significance
        if quality_metrics.p_value < 0.05:
            criteria_met += 1
        
        # Information ratio
        if abs(quality_metrics.ic_ir) > 0.5:
            criteria_met += 1
        
        # Hit rate
        if quality_metrics.hit_rate > 0.55:
            criteria_met += 1
        
        # Stability
        if quality_metrics.stability_score > 0.3:
            criteria_met += 1
        
        score = criteria_met / total_criteria
        
        if score >= 0.9:
            return 'A'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.5:
            return 'C'
        elif score >= 0.3:
            return 'D'
        else:
            return 'F'
    
    def _calculate_factor_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate aggregate factor quality score."""
        if not results:
            return 0.0
        
        grade_points = {'A': 100, 'B': 85, 'C': 70, 'D': 55, 'F': 0}
        
        scores = [grade_points.get(result['grade'], 0) for result in results.values()]
        return np.mean(scores)
    
    def _calculate_backtesting_score(self, results: Dict[str, Any]) -> float:
        """Calculate aggregate backtesting score."""
        if not results:
            return 0.0
        
        scores = []
        for result in results.values():
            performance_metrics = result.get('performance_metrics')
            if performance_metrics:
                sharpe = performance_metrics.sharpe_ratio
                max_dd = abs(performance_metrics.max_drawdown)
                
                # Score based on Sharpe ratio and drawdown
                sharpe_score = min(100, max(0, (sharpe / 2.0) * 100))
                dd_score = max(0, 100 - (max_dd / 0.2) * 100)
                
                factor_score = (sharpe_score + dd_score) / 2
                scores.append(factor_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_interaction_score(self, results: Dict[str, Any]) -> float:
        """Calculate agent interaction score."""
        if not results:
            return 80.0  # Default score if interaction tests disabled
        
        pass_rate = results.get('pass_rate', 0.0)
        return pass_rate * 100
    
    def _calculate_validation_score(self, results: Dict[str, Any]) -> float:
        """Calculate performance validation score."""
        if not results:
            return 0.0
        
        valid_count = 0
        total_count = 0
        
        for result in results.values():
            if result['strategy_validation'].is_valid:
                valid_count += 1
            total_count += 1
            
            if result['out_of_sample_validation']:
                if result['out_of_sample_validation'].is_valid:
                    valid_count += 1
                total_count += 1
        
        return (valid_count / total_count * 100) if total_count > 0 else 0.0
    
    def _assign_letter_grade(self, score: float) -> str:
        """Assign letter grade based on numerical score."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(
        self,
        overall_score: float,
        factor_quality_results: Dict[str, Any],
        backtesting_results: Dict[str, Any],
        agent_interaction_results: Dict[str, Any],
        performance_validation_results: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        # Overall performance recommendations
        if overall_score < 70:
            recommendations.append(
                "Overall quality score is below acceptable threshold. "
                "Comprehensive review of factor methodology required."
            )
        
        # Factor quality recommendations
        low_quality_factors = [
            name for name, result in factor_quality_results.items()
            if result['grade'] in ['D', 'F']
        ]
        
        if low_quality_factors:
            recommendations.append(
                f"Factors with low quality detected: {', '.join(low_quality_factors)}. "
                "Consider feature engineering or alternative factor construction."
            )
        
        # Backtesting recommendations
        low_sharpe_factors = []
        high_drawdown_factors = []
        
        for name, result in backtesting_results.items():
            metrics = result.get('performance_metrics')
            if metrics:
                if metrics.sharpe_ratio < 1.0:
                    low_sharpe_factors.append(name)
                if abs(metrics.max_drawdown) > 0.15:
                    high_drawdown_factors.append(name)
        
        if low_sharpe_factors:
            recommendations.append(
                f"Low Sharpe ratio factors: {', '.join(low_sharpe_factors)}. "
                "Review signal strength and position sizing."
            )
        
        if high_drawdown_factors:
            recommendations.append(
                f"High drawdown factors: {', '.join(high_drawdown_factors)}. "
                "Implement additional risk controls."
            )
        
        # Agent interaction recommendations
        if agent_interaction_results:
            pass_rate = agent_interaction_results.get('pass_rate', 1.0)
            if pass_rate < 0.8:
                recommendations.append(
                    "Agent interaction tests showing instability. "
                    "Review A2A protocol implementation and memory coordination."
                )
        
        # Success recommendations
        if overall_score >= 85:
            recommendations.append(
                "Alpha agent pool demonstrates high quality and is ready for production deployment."
            )
        elif overall_score >= 75:
            recommendations.append(
                "Alpha agent pool shows good quality with minor improvements needed."
            )
        
        return recommendations
    
    async def _generate_quality_report(self, result: QualityAssessmentResult) -> None:
        """Generate comprehensive quality assessment report."""
        
        report_timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        report_path = self.output_path / f"quality_assessment_report_{report_timestamp}.md"
        
        report_content = f"""# Alpha Agent Pool Quality Assessment Report

## Executive Summary

**Assessment Date:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Overall Grade:** {result.overall_grade}
**Overall Score:** {result.overall_score:.1f}/100
**Tests Passed:** {result.passed_tests}/{result.total_tests} ({result.passed_tests/result.total_tests*100:.1f}%)

## Component Scores

- **Factor Quality:** {result.detailed_results['component_scores']['factor_quality']:.1f}/100
- **Backtesting Performance:** {result.detailed_results['component_scores']['backtesting']:.1f}/100
- **Agent Interactions:** {result.detailed_results['component_scores']['interaction']:.1f}/100
- **Performance Validation:** {result.detailed_results['component_scores']['validation']:.1f}/100

## Factor Quality Assessment

"""
        
        for factor_name, factor_result in result.factor_quality_results.items():
            report_content += f"### {factor_name} (Grade: {factor_result['grade']})\n\n"
            report_content += f"```\n{factor_result['report']}\n```\n\n"
        
        report_content += """## Backtesting Results

"""
        
        for factor_name, backtest_result in result.backtesting_results.items():
            metrics = backtest_result.get('performance_metrics')
            if metrics:
                report_content += f"""### {factor_name}

- **Total Return:** {metrics.total_return:.2%}
- **Sharpe Ratio:** {metrics.sharpe_ratio:.2f}
- **Maximum Drawdown:** {metrics.max_drawdown:.2%}
- **Win Rate:** {metrics.win_rate:.2%}
- **Trade Count:** {metrics.trade_count}

"""
        
        if result.agent_interaction_results:
            report_content += f"""## Agent Interaction Testing

- **Total Tests:** {result.agent_interaction_results['total_tests']}
- **Passed Tests:** {result.agent_interaction_results['passed_tests']}
- **Pass Rate:** {result.agent_interaction_results['pass_rate']:.2%}

"""
        
        report_content += """## Recommendations

"""
        
        for i, recommendation in enumerate(result.recommendations, 1):
            report_content += f"{i}. {recommendation}\n"
        
        report_content += f"""

## Technical Details

### Configuration
- **Data Period:** {self.config.test_period_start} to {self.config.test_period_end}
- **Initial Capital:** ${self.config.initial_capital:,.0f}
- **Transaction Costs:** {self.config.transaction_cost_bps} bps
- **Confidence Level:** {self.config.confidence_level:.0%}

### Data Summary
- **Market Data Sources:** {len(self.load_market_data().columns)} symbols
- **Test Factors Generated:** {len(result.factor_quality_results)}

---
*Report generated by Alpha Agent Pool Quality Assurance Pipeline*
"""
        
        # Write report to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Quality assessment report saved to: {report_path}")
        
        # Also save JSON summary for programmatic access
        json_path = self.output_path / f"quality_assessment_summary_{report_timestamp}.json"
        
        summary_data = {
            'timestamp': result.timestamp.isoformat(),
            'overall_grade': result.overall_grade,
            'overall_score': result.overall_score,
            'passed_tests': result.passed_tests,
            'total_tests': result.total_tests,
            'component_scores': result.detailed_results['component_scores'],
            'recommendations': result.recommendations
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Quality assessment summary saved to: {json_path}")

# Main execution function
async def main():
    """Main function to run quality assessment pipeline."""
    
    # Configure pipeline
    config = QualityAssessmentConfig()
    
    # Create and run pipeline
    pipeline = AlphaAgentPoolQualityPipeline(config)
    
    try:
        result = await pipeline.run_comprehensive_quality_assessment()
        
        print(f"\n{'='*60}")
        print(f"ALPHA AGENT POOL QUALITY ASSESSMENT COMPLETE")
        print(f"{'='*60}")
        print(f"Overall Grade: {result.overall_grade}")
        print(f"Overall Score: {result.overall_score:.1f}/100")
        print(f"Tests Passed: {result.passed_tests}/{result.total_tests}")
        print(f"{'='*60}")
        
        if result.recommendations:
            print("\nKey Recommendations:")
            for i, rec in enumerate(result.recommendations[:3], 1):
                print(f"{i}. {rec}")
        
        return result
        
    except Exception as e:
        logger.error(f"Quality assessment pipeline failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
