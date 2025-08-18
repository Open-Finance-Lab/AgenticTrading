#!/usr/bin/env python3
"""
Alpha Agent Pool Quality Acceptance Testing Script

This script serves as the main entry point for running comprehensive
quality assurance tests on the alpha agent pool. It validates alpha
factors, backtesting performance, agent interactions, and provides
a final acceptance decision for production deployment.

Usage:
    python run_quality_acceptance.py [--config CONFIG_FILE]

Author: FinAgent Quality Assurance Team
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quality_assurance.quality_pipeline import (
    AlphaAgentPoolQualityPipeline,
    QualityAssessmentConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print quality acceptance testing banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ALPHA AGENT POOL QUALITY ACCEPTANCE                      ║
║                         Comprehensive Testing Suite                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  • Alpha Factor Quality Assessment                                          ║
║  • Event-Driven Backtesting Validation                                     ║
║  • Agent Interaction & Collaboration Testing                               ║
║  • Performance Validation & Statistical Testing                            ║
║  • Reinforcement Learning Update Validation                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def load_custom_config(config_path: str) -> QualityAssessmentConfig:
    """Load custom configuration from file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return QualityAssessmentConfig(**config_data)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        logger.info("Using default configuration")
        return QualityAssessmentConfig()

def print_results_summary(result):
    """Print formatted results summary."""
    
    print(f"\n{'='*80}")
    print(f"                        QUALITY ASSESSMENT RESULTS")
    print(f"{'='*80}")
    
    # Overall results
    print(f"\n📊 OVERALL ASSESSMENT")
    print(f"   Grade:          {result.overall_grade}")
    print(f"   Score:          {result.overall_score:.1f}/100")
    print(f"   Tests Passed:   {result.passed_tests}/{result.total_tests} ({result.passed_tests/result.total_tests*100:.1f}%)")
    
    # Component scores
    component_scores = result.detailed_results['component_scores']
    print(f"\n📈 COMPONENT SCORES")
    print(f"   Factor Quality:       {component_scores['factor_quality']:.1f}/100")
    print(f"   Backtesting:          {component_scores['backtesting']:.1f}/100")
    print(f"   Agent Interactions:   {component_scores['interaction']:.1f}/100")
    print(f"   Performance Valid.:   {component_scores['validation']:.1f}/100")
    
    # Factor quality breakdown
    print(f"\n🔍 FACTOR QUALITY BREAKDOWN")
    for factor_name, factor_result in result.factor_quality_results.items():
        grade = factor_result['grade']
        emoji = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴', 'F': '⚫'}.get(grade, '❓')
        print(f"   {factor_name:<25} {emoji} Grade {grade}")
    
    # Backtesting summary
    print(f"\n⚡ BACKTESTING PERFORMANCE")
    for factor_name, backtest_result in result.backtesting_results.items():
        metrics = backtest_result.get('performance_metrics')
        if metrics:
            sharpe_emoji = '🟢' if metrics.sharpe_ratio >= 1.0 else '🟡' if metrics.sharpe_ratio >= 0.5 else '🔴'
            dd_emoji = '🟢' if abs(metrics.max_drawdown) <= 0.15 else '🟡' if abs(metrics.max_drawdown) <= 0.25 else '🔴'
            print(f"   {factor_name:<25} {sharpe_emoji} Sharpe: {metrics.sharpe_ratio:.2f}  {dd_emoji} DD: {metrics.max_drawdown:.1%}")
    
    # Agent interactions (if tested)
    if result.agent_interaction_results:
        interaction_emoji = '🟢' if result.agent_interaction_results['pass_rate'] >= 0.8 else '🟡' if result.agent_interaction_results['pass_rate'] >= 0.6 else '🔴'
        print(f"\n🤝 AGENT INTERACTIONS")
        print(f"   Test Results:    {interaction_emoji} {result.agent_interaction_results['passed_tests']}/{result.agent_interaction_results['total_tests']} ({result.agent_interaction_results['pass_rate']:.1%})")
    
    # Recommendations
    print(f"\n💡 KEY RECOMMENDATIONS")
    for i, rec in enumerate(result.recommendations[:5], 1):
        print(f"   {i}. {rec}")
    
    # Final verdict
    print(f"\n{'='*80}")
    if result.overall_grade in ['A', 'B']:
        verdict_emoji = '✅'
        verdict_text = "ACCEPTED FOR PRODUCTION"
        verdict_color = '\033[92m'  # Green
    elif result.overall_grade == 'C':
        verdict_emoji = '⚠️'
        verdict_text = "CONDITIONAL ACCEPTANCE - IMPROVEMENTS NEEDED"
        verdict_color = '\033[93m'  # Yellow
    else:
        verdict_emoji = '❌'
        verdict_text = "REJECTED - MAJOR ISSUES DETECTED"
        verdict_color = '\033[91m'  # Red
    
    reset_color = '\033[0m'
    
    print(f"{verdict_color}              {verdict_emoji} {verdict_text} {verdict_emoji}{reset_color}")
    print(f"{'='*80}")

async def run_quality_acceptance(config: QualityAssessmentConfig) -> bool:
    """
    Run complete quality acceptance testing.
    
    Args:
        config: Quality assessment configuration
        
    Returns:
        True if acceptance criteria met, False otherwise
    """
    
    print_banner()
    
    # Initialize pipeline
    print("🔧 Initializing Quality Assurance Pipeline...")
    pipeline = AlphaAgentPoolQualityPipeline(config)
    
    # Run comprehensive assessment
    print("🚀 Starting Comprehensive Quality Assessment...")
    print(f"   Data Period: {config.test_period_start} to {config.test_period_end}")
    print(f"   Initial Capital: ${config.initial_capital:,.0f}")
    print(f"   Transaction Costs: {config.transaction_cost_bps} bps")
    print(f"   Confidence Level: {config.confidence_level:.0%}")
    print()
    
    try:
        # Run assessment
        result = await pipeline.run_comprehensive_quality_assessment()
        
        # Print results
        print_results_summary(result)
        
        # Determine acceptance
        acceptance_threshold = 75.0  # Minimum score for acceptance
        is_accepted = result.overall_score >= acceptance_threshold and result.overall_grade in ['A', 'B', 'C']
        
        # Log final decision
        if is_accepted:
            logger.info(f"Alpha Agent Pool ACCEPTED for production (Score: {result.overall_score:.1f}, Grade: {result.overall_grade})")
        else:
            logger.warning(f"Alpha Agent Pool REJECTED (Score: {result.overall_score:.1f}, Grade: {result.overall_grade})")
        
        return is_accepted
        
    except Exception as e:
        logger.error(f"Quality acceptance testing failed: {e}")
        print(f"\n❌ QUALITY ACCEPTANCE TESTING FAILED")
        print(f"   Error: {e}")
        print(f"   Alpha Agent Pool REJECTED due to testing failure")
        return False

def main():
    """Main entry point for quality acceptance testing."""
    
    parser = argparse.ArgumentParser(
        description="Alpha Agent Pool Quality Acceptance Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_quality_acceptance.py                    # Use default configuration
  python run_quality_acceptance.py --config qa.json  # Use custom configuration
  
Configuration File Format (JSON):
  {
    "data_cache_path": "/path/to/data/cache",
    "test_period_start": "2022-01-01",
    "test_period_end": "2024-12-31",
    "initial_capital": 1000000.0,
    "transaction_cost_bps": 5.0,
    "min_sharpe_ratio": 1.0,
    "max_drawdown_threshold": 0.15,
    "confidence_level": 0.95
  }
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file (JSON format)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/data',
        help='Path to data directory (default: relative data path)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='quality_assurance/reports',
        help='Output directory for reports (default: quality_assurance/reports)'
    )
    
    parser.add_argument(
        '--disable-agent-tests',
        action='store_true',
        help='Disable agent interaction tests'
    )
    
    parser.add_argument(
        '--disable-rl-tests',
        action='store_true',
        help='Disable reinforcement learning validation tests'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        config = load_custom_config(args.config)
    else:
        config = QualityAssessmentConfig()
    
    # Apply command line overrides
    if args.data_path:
        config.data_cache_path = os.path.join(args.data_path, 'cache')
    config.output_directory = args.output_dir
    config.enable_agent_interaction_tests = not args.disable_agent_tests
    config.enable_rl_validation = not args.disable_rl_tests
    
    # Ensure output directory exists
    Path(config.output_directory).mkdir(parents=True, exist_ok=True)
    
    # Run quality acceptance testing
    try:
        is_accepted = asyncio.run(run_quality_acceptance(config))
        
        # Exit with appropriate code
        exit_code = 0 if is_accepted else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Quality acceptance testing interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\n\n❌ Unexpected error during quality acceptance testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
