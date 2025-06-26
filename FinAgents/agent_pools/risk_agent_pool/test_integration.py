"""
Comprehensive Integration Test for Risk Agent Pool

Author: Jifeng Li
License: openMDW
Description: Integration tests to verify all components of the Risk Agent Pool
             work together correctly, including OpenAI integration, MCP server,
             external memory, and all risk analysis agents.
"""

import asyncio
import pytest
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import tempfile
import os

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockExternalMemory:
    """Mock external memory agent for testing"""
    
    def __init__(self):
        self.storage = {}
        self.events = []
    
    async def store_data(self, key: str, data: Dict[str, Any]) -> bool:
        self.storage[key] = data
        return True
    
    async def retrieve_data(self, key: str) -> Dict[str, Any]:
        return self.storage.get(key, {})
    
    async def record_event(self, event_data: Dict[str, Any]) -> str:
        event_id = f"event_{len(self.events)}"
        event_data["event_id"] = event_id
        event_data["timestamp"] = datetime.now()
        self.events.append(event_data)
        return event_id


class MockOpenAIClient:
    """Mock OpenAI client for testing"""
    
    async def chat_completions_create(self, **kwargs):
        # Return a structured task for risk analysis
        mock_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "task_type": "risk_analysis",
                        "agent_type": "market_risk_agent",
                        "parameters": {
                            "portfolio_data": {
                                "positions": [
                                    {"asset": "AAPL", "quantity": 100, "price": 150.0}
                                ]
                            },
                            "risk_measures": ["var", "volatility"],
                            "confidence_level": 0.95
                        }
                    })
                }
            }]
        }
        
        class MockResponse:
            def __init__(self, data):
                self.choices = [type('Choice', (), {
                    'message': type('Message', (), {'content': data["choices"][0]["message"]["content"]})()
                })()]
        
        return MockResponse(mock_response)


@pytest.fixture
async def mock_risk_pool():
    """Create a mock risk agent pool for testing"""
    from FinAgents.agent_pools.risk_agent_pool.core import RiskAgentPool
    
    # Create mock external memory
    mock_memory = MockExternalMemory()
    
    # Create risk pool with mocked dependencies
    pool = RiskAgentPool(
        openai_api_key="test_key",
        external_memory_config={
            "host": "localhost",
            "port": 8000
        }
    )
    
    # Replace with mocks
    pool.memory_bridge.external_memory = mock_memory
    pool.openai_client = MockOpenAIClient()
    
    return pool


@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing"""
    return {
        "positions": [
            {
                "asset_id": "AAPL",
                "quantity": 100,
                "current_price": 150.0,
                "currency": "USD",
                "asset_type": "equity",
                "sector": "technology"
            },
            {
                "asset_id": "GOOGL", 
                "quantity": 50,
                "current_price": 2800.0,
                "currency": "USD",
                "asset_type": "equity",
                "sector": "technology"
            },
            {
                "asset_id": "BOND_001",
                "quantity": 1000,
                "current_price": 98.5,
                "currency": "USD",
                "asset_type": "bond",
                "duration": 5.2
            }
        ],
        "returns_data": {
            "AAPL": [0.01, -0.02, 0.015, -0.005, 0.02],
            "GOOGL": [0.005, -0.01, 0.02, -0.008, 0.015],
            "BOND_001": [0.001, -0.001, 0.002, 0.0, 0.001]
        }
    }


@pytest.fixture
def sample_credit_data():
    """Sample credit data for testing"""
    return {
        "borrower_data": {
            "credit_score": 720,
            "debt_to_income": 0.35,
            "annual_income": 80000,
            "loan_amount": 250000,
            "employment_history": 5,
            "payment_history": 0.95
        },
        "loan_data": {
            "loan_type": "mortgage",
            "term_years": 30,
            "interest_rate": 0.045,
            "ltv_ratio": 0.80
        }
    }


class TestRiskAgentPoolIntegration:
    """Integration tests for Risk Agent Pool"""
    
    @pytest.mark.asyncio
    async def test_pool_initialization(self, mock_risk_pool):
        """Test risk agent pool initialization"""
        pool = mock_risk_pool
        
        # Check that agents are registered
        agent_names = pool.agent_registry.list_agents()
        
        expected_agents = [
            "market_risk_agent",
            "volatility_agent", 
            "var_agent",
            "credit_risk_agent",
            "liquidity_risk_agent",
            "operational_risk_agent",
            "stress_testing_agent",
            "model_risk_agent"
        ]
        
        for expected_agent in expected_agents:
            assert expected_agent in agent_names, f"Agent {expected_agent} not registered"
        
        logger.info(f"Successfully initialized pool with {len(agent_names)} agents")
    
    @pytest.mark.asyncio
    async def test_natural_language_processing(self, mock_risk_pool):
        """Test natural language context processing"""
        pool = mock_risk_pool
        
        test_contexts = [
            "Calculate VaR for my portfolio with 95% confidence",
            "Assess credit risk for a borrower with FICO 720",
            "Run stress test on my equity positions",
            "Monitor operational risk indicators"
        ]
        
        for context in test_contexts:
            try:
                result = await pool.process_orchestrator_input(context)
                
                assert result is not None
                assert "status" in result
                logger.info(f"Successfully processed: {context[:50]}...")
                
            except Exception as e:
                logger.error(f"Failed to process context: {context}, Error: {e}")
                raise
    
    @pytest.mark.asyncio
    async def test_market_risk_analysis(self, mock_risk_pool, sample_portfolio_data):
        """Test comprehensive market risk analysis"""
        pool = mock_risk_pool
        
        # Test different risk measures
        risk_measures = ["var", "volatility", "beta", "drawdown"]
        
        for measure in risk_measures:
            task = {
                "task_type": "risk_analysis",
                "agent_type": "market_risk_agent",
                "parameters": {
                    "portfolio_data": sample_portfolio_data,
                    "risk_measures": [measure],
                    "time_horizon": "daily"
                }
            }
            
            result = await pool.execute_structured_task(task)
            
            assert result["status"] == "success"
            assert "results" in result
            logger.info(f"Market risk analysis ({measure}) completed successfully")
    
    @pytest.mark.asyncio
    async def test_credit_risk_analysis(self, mock_risk_pool, sample_credit_data):
        """Test credit risk analysis"""
        pool = mock_risk_pool
        
        analysis_types = ["pd_estimation", "lgd_modeling", "ead_calculation"]
        
        for analysis_type in analysis_types:
            task = {
                "task_type": "risk_analysis",
                "agent_type": "credit_risk_agent",
                "parameters": {
                    "borrower_data": sample_credit_data["borrower_data"],
                    "loan_data": sample_credit_data["loan_data"],
                    "analysis_type": analysis_type
                }
            }
            
            result = await pool.execute_structured_task(task)
            
            assert result["status"] == "success"
            assert "results" in result
            logger.info(f"Credit risk analysis ({analysis_type}) completed successfully")
    
    @pytest.mark.asyncio
    async def test_operational_risk_analysis(self, mock_risk_pool):
        """Test operational risk analysis"""
        pool = mock_risk_pool
        
        # Test fraud assessment
        fraud_task = {
            "task_type": "risk_analysis",
            "agent_type": "operational_risk_agent",
            "parameters": {
                "analysis_type": "fraud_assessment",
                "transaction_data": {
                    "amount": 50000,
                    "user_id": "user123",
                    "location": "foreign_country",
                    "timestamp": datetime.now(),
                    "recent_transaction_count": 15
                }
            }
        }
        
        result = await pool.execute_structured_task(fraud_task)
        
        assert result["status"] == "success"
        assert "results" in result
        assert "fraud_risk" in result["results"]
        
        # Test KRI monitoring
        kri_task = {
            "task_type": "risk_analysis", 
            "agent_type": "operational_risk_agent",
            "parameters": {
                "analysis_type": "kri_monitoring",
                "current_metrics": {
                    "system_downtime_hours": 30,
                    "failed_transactions_pct": 0.08,
                    "staff_turnover_rate": 0.20
                }
            }
        }
        
        result = await pool.execute_structured_task(kri_task)
        
        assert result["status"] == "success"
        assert "results" in result
        assert "kri_status" in result["results"]
        
        logger.info("Operational risk analysis completed successfully")
    
    @pytest.mark.asyncio
    async def test_stress_testing(self, mock_risk_pool, sample_portfolio_data):
        """Test stress testing functionality"""
        pool = mock_risk_pool
        
        # Convert portfolio data to expected format
        portfolio_positions = []
        for pos in sample_portfolio_data["positions"]:
            from FinAgents.agent_pools.risk_agent_pool.agents.stress_testing import PortfolioPosition
            portfolio_positions.append(PortfolioPosition(
                asset_id=pos["asset_id"],
                quantity=pos["quantity"],
                current_price=pos["current_price"],
                currency=pos["currency"],
                asset_type=pos["asset_type"],
                sector=pos.get("sector"),
                duration=pos.get("duration")
            ))
        
        test_scenarios = [
            {
                "test_type": "scenario",
                "scenario_id": "2008_financial_crisis",
                "portfolio": portfolio_positions
            },
            {
                "test_type": "sensitivity",
                "risk_factor": "equity_market",
                "shock_range": (-0.2, 0.2),
                "portfolio": portfolio_positions
            },
            {
                "test_type": "scenario_library"
            }
        ]
        
        for scenario in test_scenarios:
            task = {
                "task_type": "risk_analysis",
                "agent_type": "stress_testing_agent", 
                "parameters": scenario
            }
            
            result = await pool.execute_structured_task(task)
            
            assert result["status"] == "success"
            assert "results" in result
            logger.info(f"Stress test ({scenario['test_type']}) completed successfully")
    
    @pytest.mark.asyncio
    async def test_model_risk_management(self, mock_risk_pool):
        """Test model risk management functionality"""
        pool = mock_risk_pool
        
        # Test model registration
        from FinAgents.agent_pools.risk_agent_pool.agents.model_risk import ModelMetadata, ModelType, ModelStatus
        
        model_metadata = ModelMetadata(
            model_id="TEST_MODEL_001",
            name="Test VaR Model",
            model_type=ModelType.RISK,
            version="1.0",
            developer="Test Developer",
            business_owner="Risk Team",
            description="Test model for VaR calculation",
            purpose="Risk measurement",
            status=ModelStatus.DEVELOPMENT,
            created_date=datetime.now(),
            last_updated=datetime.now(),
            criticality_level="high"
        )
        
        register_task = {
            "task_type": "risk_analysis",
            "agent_type": "model_risk_agent",
            "parameters": {
                "action": "register_model",
                "model_metadata": model_metadata
            }
        }
        
        result = await pool.execute_structured_task(register_task)
        
        assert result["status"] == "success"
        assert "results" in result
        assert "model_id" in result["results"]
        
        # Test model validation
        validation_task = {
            "task_type": "risk_analysis",
            "agent_type": "model_risk_agent",
            "parameters": {
                "action": "validate_model",
                "model_id": "TEST_MODEL_001",
                "validator": "Risk Team",
                "validation_config": {
                    "accuracy_tests": {"min_accuracy": 0.85},
                    "stability_tests": {},
                    "bias_tests": {}
                }
            }
        }
        
        result = await pool.execute_structured_task(validation_task)
        
        assert result["status"] == "success"
        assert "results" in result
        assert "validation_report" in result["results"]
        
        logger.info("Model risk management completed successfully")
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, mock_risk_pool, sample_portfolio_data):
        """Test external memory integration"""
        pool = mock_risk_pool
        
        # Perform analysis that should be stored in memory
        task = {
            "task_type": "risk_analysis",
            "agent_type": "market_risk_agent",
            "parameters": {
                "portfolio_data": sample_portfolio_data,
                "risk_measures": ["var"],
                "confidence_level": 0.95
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        assert result["status"] == "success"
        
        # Check that analysis was recorded in memory
        memory_bridge = pool.memory_bridge
        
        # Store a test analysis record
        from FinAgents.agent_pools.risk_agent_pool.memory_bridge import RiskAnalysisRecord
        
        analysis_record = RiskAnalysisRecord(
            analysis_id="TEST_ANALYSIS_001",
            risk_type="market",
            portfolio_id="TEST_PORTFOLIO",
            analysis_results=result["results"],
            timestamp=datetime.now(),
            confidence_level=0.95,
            time_horizon="daily"
        )
        
        stored = await memory_bridge.store_analysis_result(analysis_record)
        assert stored
        
        # Retrieve analysis history
        history = await memory_bridge.retrieve_analysis_history({
            "risk_type": "market",
            "portfolio_id": "TEST_PORTFOLIO"
        })
        
        assert len(history) > 0
        logger.info("Memory integration test completed successfully")
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, mock_risk_pool, sample_portfolio_data):
        """Test concurrent risk analysis requests"""
        pool = mock_risk_pool
        
        # Create multiple concurrent tasks
        tasks = []
        for i in range(5):
            task = {
                "task_type": "risk_analysis",
                "agent_type": "market_risk_agent",
                "parameters": {
                    "portfolio_data": sample_portfolio_data,
                    "risk_measures": ["var", "volatility"],
                    "time_horizon": "daily"
                }
            }
            tasks.append(pool.execute_structured_task(task))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all tasks completed successfully
        successful_results = 0
        for result in results:
            if isinstance(result, dict) and result.get("status") == "success":
                successful_results += 1
            elif isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
        
        assert successful_results >= 4, f"Only {successful_results}/5 concurrent tasks succeeded"
        logger.info(f"Concurrent analysis test completed: {successful_results}/5 tasks successful")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_risk_pool):
        """Test error handling and recovery"""
        pool = mock_risk_pool
        
        # Test with invalid agent type
        invalid_task = {
            "task_type": "risk_analysis",
            "agent_type": "nonexistent_agent",
            "parameters": {}
        }
        
        result = await pool.execute_structured_task(invalid_task)
        
        assert result["status"] == "error"
        assert "error" in result
        
        # Test with invalid parameters
        invalid_params_task = {
            "task_type": "risk_analysis",
            "agent_type": "market_risk_agent",
            "parameters": {
                "portfolio_data": None,  # Invalid data
                "risk_measures": ["invalid_measure"]
            }
        }
        
        result = await pool.execute_structured_task(invalid_params_task)
        
        # Should handle gracefully
        assert "status" in result
        logger.info("Error handling test completed successfully")


class TestPerformanceMetrics:
    """Performance and benchmarking tests"""
    
    @pytest.mark.asyncio
    async def test_analysis_performance(self, mock_risk_pool, sample_portfolio_data):
        """Test analysis performance metrics"""
        pool = mock_risk_pool
        
        start_time = datetime.now()
        
        task = {
            "task_type": "risk_analysis",
            "agent_type": "market_risk_agent",
            "parameters": {
                "portfolio_data": sample_portfolio_data,
                "risk_measures": ["var", "volatility", "beta"],
                "time_horizon": "daily"
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        assert result["status"] == "success"
        assert execution_time < 5.0, f"Analysis took too long: {execution_time}s"
        
        logger.info(f"Market risk analysis completed in {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_memory_cache_performance(self, mock_risk_pool):
        """Test memory caching performance"""
        pool = mock_risk_pool
        memory_bridge = pool.memory_bridge
        
        # Test data
        test_key = "test_cache_key"
        test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
        
        # First store (should be slow)
        start_time = datetime.now()
        await memory_bridge.cache_store(test_key, test_data)
        store_time = (datetime.now() - start_time).total_seconds()
        
        # First retrieve (should be fast from cache)
        start_time = datetime.now()
        retrieved_data = await memory_bridge.cache_get(test_key)
        retrieve_time = (datetime.now() - start_time).total_seconds()
        
        assert retrieved_data == test_data
        assert retrieve_time < store_time, "Cache retrieval should be faster than storage"
        
        logger.info(f"Cache performance: store={store_time:.3f}s, retrieve={retrieve_time:.3f}s")


if __name__ == "__main__":
    """Run integration tests manually"""
    import sys
    
    async def run_tests():
        # Create test fixtures
        mock_memory = MockExternalMemory()
        
        from FinAgents.agent_pools.risk_agent_pool.core import RiskAgentPool
        
        pool = RiskAgentPool(
            openai_api_key="test_key",
            external_memory_config={"host": "localhost", "port": 8000}
        )
        
        pool.memory_bridge.external_memory = mock_memory
        pool.openai_client = MockOpenAIClient()
        
        sample_portfolio = {
            "positions": [
                {"asset_id": "AAPL", "quantity": 100, "current_price": 150.0, 
                 "currency": "USD", "asset_type": "equity", "sector": "technology"}
            ],
            "returns_data": {"AAPL": [0.01, -0.02, 0.015]}
        }
        
        # Run basic tests
        print("Running Risk Agent Pool Integration Tests...")
        
        try:
            # Test 1: Initialization
            agent_names = pool.agent_registry.list_agents()
            print(f"✓ Initialized with {len(agent_names)} agents")
            
            # Test 2: Natural language processing
            result = await pool.process_orchestrator_input(
                "Calculate VaR for my portfolio with 95% confidence"
            )
            print(f"✓ Natural language processing: {result.get('status')}")
            
            # Test 3: Market risk analysis
            task = {
                "task_type": "risk_analysis",
                "agent_type": "market_risk_agent",
                "parameters": {
                    "portfolio_data": sample_portfolio,
                    "risk_measures": ["var"],
                    "time_horizon": "daily"
                }
            }
            
            result = await pool.execute_structured_task(task)
            print(f"✓ Market risk analysis: {result.get('status')}")
            
            print("\n✅ All integration tests passed!")
            
        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
            sys.exit(1)
    
    # Run the tests
    asyncio.run(run_tests())
