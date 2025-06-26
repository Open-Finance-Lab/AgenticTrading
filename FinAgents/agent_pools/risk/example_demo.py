"""
Risk Agent Pool Example - Comprehensive Risk Analysis Demo

Author: Jifeng Li
License: openMDW
Description: Example script demonstrating the capabilities of the Risk Agent Pool
             including market risk, credit risk, operational risk, and stress testing.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RiskAgentPoolDemo:
    """
    Demonstration of Risk Agent Pool capabilities
    """
    
    def __init__(self):
        self.risk_pool = None
        
    async def setup(self):
        """Initialize the risk agent pool"""
        try:
            # Import the risk agent pool
            from FinAgents.agent_pools.risk_agent_pool import RiskAgentPool
            
            # Initialize with configuration
            self.risk_pool = RiskAgentPool(
                openai_api_key="your_openai_api_key",  # Replace with actual key
                external_memory_config={
                    "host": "localhost",
                    "port": 8000,
                    "cache_enabled": True,
                    "cache_ttl": 3600
                },
                mcp_server_config={
                    "host": "localhost",
                    "port": 3000
                }
            )
            
            # Start the MCP server
            await self.risk_pool.start()
            
            logger.info("Risk Agent Pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Risk Agent Pool: {e}")
            raise
    
    async def demo_natural_language_processing(self):
        """Demonstrate natural language risk analysis requests"""
        logger.info("\n=== Natural Language Processing Demo ===")
        
        natural_language_requests = [
            "Calculate the 95% VaR for my technology stock portfolio",
            "Assess credit risk for a borrower with FICO score 720 and DTI 0.35",
            "Run a stress test using the 2008 financial crisis scenario",
            "Check for fraud risk in a $50,000 international wire transfer",
            "Monitor operational risk indicators for system downtime",
            "Validate my VaR pricing model and check performance metrics"
        ]
        
        for i, request in enumerate(natural_language_requests, 1):
            try:
                logger.info(f"\nRequest {i}: {request}")
                
                result = await self.risk_pool.process_orchestrator_input(request)
                
                if result.get("status") == "success":
                    logger.info(f"✓ Analysis completed: {result.get('summary', 'No summary')}")
                else:
                    logger.warning(f"⚠ Analysis failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ Request failed: {e}")
    
    async def demo_market_risk_analysis(self):
        """Demonstrate comprehensive market risk analysis"""
        logger.info("\n=== Market Risk Analysis Demo ===")
        
        # Sample portfolio data
        portfolio_data = {
            "positions": [
                {
                    "asset_id": "AAPL",
                    "quantity": 1000,
                    "current_price": 150.0,
                    "currency": "USD",
                    "asset_type": "equity",
                    "sector": "technology",
                    "beta": 1.2
                },
                {
                    "asset_id": "GOOGL",
                    "quantity": 500,
                    "current_price": 2800.0,
                    "currency": "USD", 
                    "asset_type": "equity",
                    "sector": "technology",
                    "beta": 1.1
                },
                {
                    "asset_id": "BOND_10Y",
                    "quantity": 10000,
                    "current_price": 98.5,
                    "currency": "USD",
                    "asset_type": "bond",
                    "duration": 8.2,
                    "credit_rating": "AAA"
                }
            ],
            "returns_data": {
                "AAPL": [0.02, -0.01, 0.015, -0.008, 0.025, -0.012, 0.018],
                "GOOGL": [0.015, -0.005, 0.022, -0.015, 0.020, -0.008, 0.012],
                "BOND_10Y": [0.002, -0.001, 0.001, 0.003, -0.002, 0.001, 0.002]
            }
        }
        
        # Risk analysis requests
        risk_analyses = [
            {
                "name": "Value at Risk (VaR)",
                "task": {
                    "task_type": "risk_analysis",
                    "agent_type": "market_risk_agent",
                    "parameters": {
                        "portfolio_data": portfolio_data,
                        "risk_measures": ["var"],
                        "confidence_levels": [0.95, 0.99],
                        "time_horizon": "daily"
                    }
                }
            },
            {
                "name": "Volatility Analysis",
                "task": {
                    "task_type": "risk_analysis",
                    "agent_type": "volatility_agent",
                    "parameters": {
                        "portfolio_data": portfolio_data,
                        "analysis_type": "portfolio_volatility",
                        "method": "historical"
                    }
                }
            },
            {
                "name": "Portfolio Beta",
                "task": {
                    "task_type": "risk_analysis",
                    "agent_type": "market_risk_agent",
                    "parameters": {
                        "portfolio_data": portfolio_data,
                        "risk_measures": ["beta"],
                        "benchmark": "S&P500"
                    }
                }
            }
        ]
        
        for analysis in risk_analyses:
            try:
                logger.info(f"\nRunning {analysis['name']}...")
                
                result = await self.risk_pool.execute_structured_task(analysis['task'])
                
                if result.get("status") == "success":
                    logger.info(f"✓ {analysis['name']} completed")
                    self._print_results(result.get("results", {}))
                else:
                    logger.warning(f"⚠ {analysis['name']} failed: {result.get('error')}")
                    
            except Exception as e:
                logger.error(f"❌ {analysis['name']} error: {e}")
    
    async def demo_credit_risk_analysis(self):
        """Demonstrate credit risk analysis"""
        logger.info("\n=== Credit Risk Analysis Demo ===")
        
        # Sample borrower data
        borrower_scenarios = [
            {
                "name": "Prime Borrower",
                "data": {
                    "borrower_data": {
                        "credit_score": 780,
                        "debt_to_income": 0.25,
                        "annual_income": 120000,
                        "employment_history": 8,
                        "payment_history": 0.98
                    },
                    "loan_data": {
                        "loan_amount": 400000,
                        "loan_type": "mortgage",
                        "term_years": 30,
                        "ltv_ratio": 0.75
                    }
                }
            },
            {
                "name": "Subprime Borrower",
                "data": {
                    "borrower_data": {
                        "credit_score": 620,
                        "debt_to_income": 0.45,
                        "annual_income": 55000,
                        "employment_history": 2,
                        "payment_history": 0.85
                    },
                    "loan_data": {
                        "loan_amount": 200000,
                        "loan_type": "mortgage",
                        "term_years": 30,
                        "ltv_ratio": 0.90
                    }
                }
            }
        ]
        
        credit_analyses = ["pd_estimation", "lgd_modeling", "ead_calculation"]
        
        for scenario in borrower_scenarios:
            logger.info(f"\nAnalyzing {scenario['name']}:")
            
            for analysis_type in credit_analyses:
                try:
                    task = {
                        "task_type": "risk_analysis",
                        "agent_type": "credit_risk_agent",
                        "parameters": {
                            "borrower_data": scenario["data"]["borrower_data"],
                            "loan_data": scenario["data"]["loan_data"],
                            "analysis_type": analysis_type
                        }
                    }
                    
                    result = await self.risk_pool.execute_structured_task(task)
                    
                    if result.get("status") == "success":
                        logger.info(f"  ✓ {analysis_type.upper()} completed")
                    else:
                        logger.warning(f"  ⚠ {analysis_type.upper()} failed")
                        
                except Exception as e:
                    logger.error(f"  ❌ {analysis_type.upper()} error: {e}")
    
    async def demo_operational_risk_analysis(self):
        """Demonstrate operational risk analysis"""
        logger.info("\n=== Operational Risk Analysis Demo ===")
        
        # Fraud detection demo
        suspicious_transactions = [
            {
                "name": "Large International Transfer",
                "data": {
                    "amount": 75000,
                    "user_id": "user_12345",
                    "location": "foreign_country",
                    "timestamp": datetime.now(),
                    "recent_transaction_count": 20,
                    "deviates_from_pattern": True
                }
            },
            {
                "name": "Normal Transaction",
                "data": {
                    "amount": 500,
                    "user_id": "user_67890",
                    "location": "domestic",
                    "timestamp": datetime.now(),
                    "recent_transaction_count": 3,
                    "deviates_from_pattern": False
                }
            }
        ]
        
        for transaction in suspicious_transactions:
            try:
                logger.info(f"\nAssessing fraud risk for: {transaction['name']}")
                
                task = {
                    "task_type": "risk_analysis",
                    "agent_type": "operational_risk_agent",
                    "parameters": {
                        "analysis_type": "fraud_assessment",
                        "transaction_data": transaction["data"]
                    }
                }
                
                result = await self.risk_pool.execute_structured_task(task)
                
                if result.get("status") == "success":
                    fraud_result = result["results"]["fraud_risk"]
                    logger.info(f"  Risk Level: {fraud_result['risk_level']}")
                    logger.info(f"  Risk Score: {fraud_result['risk_score']:.2f}")
                    logger.info(f"  Requires Review: {fraud_result['requires_review']}")
                    
            except Exception as e:
                logger.error(f"  ❌ Fraud assessment error: {e}")
        
        # KRI monitoring demo
        try:
            logger.info("\nMonitoring Key Risk Indicators...")
            
            task = {
                "task_type": "risk_analysis",
                "agent_type": "operational_risk_agent",
                "parameters": {
                    "analysis_type": "kri_monitoring",
                    "current_metrics": {
                        "system_downtime_hours": 30,  # Threshold breach
                        "failed_transactions_pct": 0.03,  # Normal
                        "staff_turnover_rate": 0.20,  # Threshold breach
                        "compliance_violations": 2,  # Normal
                        "fraud_incidents_monthly": 15  # Threshold breach
                    }
                }
            }
            
            result = await self.risk_pool.execute_structured_task(task)
            
            if result.get("status") == "success":
                kri_result = result["results"]["kri_status"]
                logger.info(f"  Overall Status: {kri_result['overall_status']}")
                logger.info(f"  Total Breaches: {kri_result['total_breaches']}")
                
                for alert in kri_result.get("alerts", []):
                    logger.warning(f"  ⚠ Alert: {alert['kri']} - {alert['severity']}")
                    
        except Exception as e:
            logger.error(f"❌ KRI monitoring error: {e}")
    
    async def demo_stress_testing(self):
        """Demonstrate stress testing capabilities"""
        logger.info("\n=== Stress Testing Demo ===")
        
        # Portfolio for stress testing
        from FinAgents.agent_pools.risk_agent_pool.agents.stress_testing import PortfolioPosition
        
        portfolio = [
            PortfolioPosition(
                asset_id="AAPL",
                quantity=1000,
                current_price=150.0,
                currency="USD",
                asset_type="equity",
                sector="technology",
                beta=1.2
            ),
            PortfolioPosition(
                asset_id="BOND_10Y",
                quantity=10000,
                current_price=98.5,
                currency="USD",
                asset_type="bond",
                duration=8.2
            )
        ]
        
        stress_tests = [
            {
                "name": "2008 Financial Crisis Scenario",
                "task": {
                    "task_type": "risk_analysis",
                    "agent_type": "stress_testing_agent",
                    "parameters": {
                        "test_type": "scenario",
                        "scenario_id": "2008_financial_crisis",
                        "portfolio": portfolio
                    }
                }
            },
            {
                "name": "Interest Rate Sensitivity",
                "task": {
                    "task_type": "risk_analysis",
                    "agent_type": "stress_testing_agent",
                    "parameters": {
                        "test_type": "sensitivity",
                        "risk_factor": "interest_rates",
                        "shock_range": (-0.03, 0.03),
                        "portfolio": portfolio
                    }
                }
            },
            {
                "name": "Monte Carlo Stress Test",
                "task": {
                    "task_type": "risk_analysis",
                    "agent_type": "stress_testing_agent",
                    "parameters": {
                        "test_type": "monte_carlo",
                        "num_simulations": 1000,
                        "time_horizon_days": 1,
                        "portfolio": portfolio
                    }
                }
            }
        ]
        
        for test in stress_tests:
            try:
                logger.info(f"\nRunning {test['name']}...")
                
                result = await self.risk_pool.execute_structured_task(test['task'])
                
                if result.get("status") == "success":
                    logger.info(f"✓ {test['name']} completed")
                    self._print_stress_test_results(test['name'], result.get("results", {}))
                else:
                    logger.warning(f"⚠ {test['name']} failed: {result.get('error')}")
                    
            except Exception as e:
                logger.error(f"❌ {test['name']} error: {e}")
    
    async def demo_model_risk_management(self):
        """Demonstrate model risk management"""
        logger.info("\n=== Model Risk Management Demo ===")
        
        try:
            # Model registration
            from FinAgents.agent_pools.risk_agent_pool.agents.model_risk import ModelMetadata, ModelType, ModelStatus
            
            model_metadata = ModelMetadata(
                model_id="VAR_MODEL_001",
                name="Monte Carlo VaR Model",
                model_type=ModelType.RISK,
                version="2.1",
                developer="Quantitative Risk Team",
                business_owner="Risk Management",
                description="Monte Carlo simulation model for Value at Risk calculation",
                purpose="Portfolio risk measurement and regulatory reporting",
                status=ModelStatus.DEVELOPMENT,
                created_date=datetime.now(),
                last_updated=datetime.now(),
                criticality_level="high",
                regulatory_classification="regulatory",
                tags=["var", "monte_carlo", "regulatory"]
            )
            
            logger.info("\nRegistering new risk model...")
            
            register_task = {
                "task_type": "risk_analysis",
                "agent_type": "model_risk_agent",
                "parameters": {
                    "action": "register_model",
                    "model_metadata": model_metadata
                }
            }
            
            result = await self.risk_pool.execute_structured_task(register_task)
            
            if result.get("status") == "success":
                model_id = result["results"]["model_id"]
                logger.info(f"✓ Model registered with ID: {model_id}")
                
                # Model validation
                logger.info("\nValidating model...")
                
                validation_task = {
                    "task_type": "risk_analysis",
                    "agent_type": "model_risk_agent",
                    "parameters": {
                        "action": "validate_model",
                        "model_id": model_id,
                        "validator": "Model Validation Team",
                        "validation_config": {
                            "methodology": "Comprehensive validation framework",
                            "data_quality": {"min_quality_score": 0.9},
                            "accuracy_tests": {"min_accuracy": 0.85},
                            "stability_tests": {"min_stability": 0.8},
                            "bias_tests": {"protected_characteristics": ["age", "gender"]},
                            "benchmark_tests": {"benchmarks": ["previous_model", "industry_standard"]}
                        }
                    }
                }
                
                result = await self.risk_pool.execute_structured_task(validation_task)
                
                if result.get("status") == "success":
                    validation_report = result["results"]["validation_report"]
                    logger.info(f"✓ Model validation completed: {validation_report.result.value}")
                    logger.info(f"  Findings: {len(validation_report.findings)} issues identified")
                    logger.info(f"  Recommendations: {len(validation_report.recommendations)} recommendations")
                
        except Exception as e:
            logger.error(f"❌ Model risk management error: {e}")
    
    async def demo_comprehensive_portfolio_analysis(self):
        """Demonstrate comprehensive portfolio risk analysis"""
        logger.info("\n=== Comprehensive Portfolio Risk Analysis ===")
        
        # Large diversified portfolio
        portfolio_data = {
            "positions": [
                # Technology stocks
                {"asset_id": "AAPL", "quantity": 1000, "current_price": 150.0, "sector": "technology"},
                {"asset_id": "GOOGL", "quantity": 500, "current_price": 2800.0, "sector": "technology"},
                {"asset_id": "MSFT", "quantity": 800, "current_price": 350.0, "sector": "technology"},
                
                # Financial stocks
                {"asset_id": "JPM", "quantity": 1200, "current_price": 140.0, "sector": "financial"},
                {"asset_id": "BAC", "quantity": 2000, "current_price": 35.0, "sector": "financial"},
                
                # Government bonds
                {"asset_id": "US_10Y", "quantity": 50000, "current_price": 98.5, "asset_type": "bond"},
                {"asset_id": "US_30Y", "quantity": 30000, "current_price": 95.2, "asset_type": "bond"},
                
                # Corporate bonds
                {"asset_id": "CORP_AAA", "quantity": 20000, "current_price": 99.1, "asset_type": "bond"},
            ]
        }
        
        # Run multiple risk analyses
        comprehensive_analyses = [
            "Calculate portfolio VaR at 95% and 99% confidence levels",
            "Assess concentration risk across sectors",
            "Run stress test with 2008 financial crisis scenario",
            "Analyze interest rate sensitivity for bond holdings",
            "Calculate portfolio beta and tracking error"
        ]
        
        for i, analysis in enumerate(comprehensive_analyses, 1):
            try:
                logger.info(f"\n{i}. {analysis}")
                
                result = await self.risk_pool.process_orchestrator_input(analysis)
                
                if result.get("status") == "success":
                    logger.info(f"   ✓ Completed successfully")
                else:
                    logger.warning(f"   ⚠ Analysis incomplete: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"   ❌ Analysis failed: {e}")
    
    def _print_results(self, results: Dict[str, Any]):
        """Print analysis results in a readable format"""
        for key, value in results.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")
    
    def _print_stress_test_results(self, test_name: str, results: Dict[str, Any]):
        """Print stress test results"""
        if test_name == "2008 Financial Crisis Scenario":
            if "stress_test" in results:
                stress_result = results["stress_test"]
                logger.info(f"  Portfolio Impact: {stress_result.portfolio_value_change_pct:.2f}%")
                logger.info(f"  Value Change: ${stress_result.portfolio_value_change:,.2f}")
        
        elif test_name == "Interest Rate Sensitivity":
            if "sensitivity" in results:
                sensitivity_result = results["sensitivity"]
                logger.info(f"  Linear Sensitivity: {sensitivity_result['linear_sensitivity']:.2f}")
                logger.info(f"  Max Loss: ${sensitivity_result['max_loss']:,.2f}")
        
        elif test_name == "Monte Carlo Stress Test":
            if "monte_carlo" in results:
                mc_result = results["monte_carlo"]
                logger.info(f"  Mean Return: {mc_result['mean_return']:.4f}")
                logger.info(f"  VaR 95%: ${mc_result['var_results'].get('VaR_0.95', 0):,.2f}")
                logger.info(f"  VaR 99%: ${mc_result['var_results'].get('VaR_0.99', 0):,.2f}")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.risk_pool:
            await self.risk_pool.stop()
            logger.info("Risk Agent Pool stopped")


async def main():
    """Main demonstration function"""
    demo = RiskAgentPoolDemo()
    
    try:
        # Initialize the risk agent pool
        await demo.setup()
        
        # Run demonstrations
        await demo.demo_natural_language_processing()
        await demo.demo_market_risk_analysis()
        await demo.demo_credit_risk_analysis()
        await demo.demo_operational_risk_analysis()
        await demo.demo_stress_testing()
        await demo.demo_model_risk_management()
        await demo.demo_comprehensive_portfolio_analysis()
        
        logger.info("\n🎉 Risk Agent Pool demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
    
    finally:
        # Clean up
        await demo.cleanup()


if __name__ == "__main__":
    """Run the demonstration"""
    print("🚀 Starting Risk Agent Pool Demonstration...")
    print("=" * 60)
    
    # Run the demo
    asyncio.run(main())
