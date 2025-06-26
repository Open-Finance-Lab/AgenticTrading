"""
Test Fixtures for Risk Agent Pool Tests

Author: Jifeng Li
License: openMDW

This module provides reusable test fixtures including mock objects,
sample data, and test utilities for Risk Agent Pool testing.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, MagicMock


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
                "sector": "technology",
                "beta": 1.2
            },
            {
                "asset_id": "GOOGL", 
                "quantity": 50,
                "current_price": 2800.0,
                "currency": "USD",
                "asset_type": "equity",
                "sector": "technology",
                "beta": 1.1
            },
            {
                "asset_id": "BOND_001",
                "quantity": 1000,
                "current_price": 98.5,
                "currency": "USD",
                "asset_type": "bond",
                "duration": 5.2,
                "credit_rating": "AAA"
            },
            {
                "asset_id": "JPY_FX",
                "quantity": 10000,
                "current_price": 0.0067,
                "currency": "JPY",
                "asset_type": "fx",
                "volatility": 0.12
            }
        ],
        "returns_data": {
            "AAPL": [0.01, -0.02, 0.015, -0.005, 0.02, 0.008, -0.012, 0.018],
            "GOOGL": [0.005, -0.01, 0.02, -0.008, 0.015, 0.003, -0.007, 0.011],
            "BOND_001": [0.001, -0.001, 0.002, 0.0, 0.001, -0.0005, 0.0015, 0.0008],
            "JPY_FX": [0.002, -0.003, 0.001, 0.004, -0.002, 0.0015, -0.001, 0.002]
        },
        "correlation_matrix": [
            [1.0, 0.7, 0.2, -0.1],
            [0.7, 1.0, 0.15, -0.05], 
            [0.2, 0.15, 1.0, 0.3],
            [-0.1, -0.05, 0.3, 1.0]
        ]
    }


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
        "prices": [150.0, 2800.0, 300.0, 3200.0, 800.0],
        "volumes": [50000000, 1200000, 30000000, 2800000, 25000000],
        "volatilities": [0.25, 0.30, 0.20, 0.35, 0.55],
        "betas": [1.2, 1.1, 0.9, 1.3, 1.8],
        "correlations": {
            "AAPL": {"GOOGL": 0.7, "MSFT": 0.8, "AMZN": 0.6, "TSLA": 0.4},
            "GOOGL": {"AAPL": 0.7, "MSFT": 0.6, "AMZN": 0.8, "TSLA": 0.3},
            "MSFT": {"AAPL": 0.8, "GOOGL": 0.6, "AMZN": 0.5, "TSLA": 0.2},
            "AMZN": {"AAPL": 0.6, "GOOGL": 0.8, "MSFT": 0.5, "TSLA": 0.4},
            "TSLA": {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.2, "AMZN": 0.4}
        },
        "sectors": {
            "AAPL": "technology",
            "GOOGL": "technology", 
            "MSFT": "technology",
            "AMZN": "consumer_discretionary",
            "TSLA": "consumer_discretionary"
        },
        "market_cap": {
            "AAPL": 2500000000000,
            "GOOGL": 1800000000000,
            "MSFT": 2300000000000,
            "AMZN": 1600000000000,
            "TSLA": 800000000000
        },
        "timestamp": datetime.now().isoformat()
    }


@pytest.fixture 
def sample_volatility_data():
    """Sample volatility data for testing"""
    import numpy as np
    
    # Generate sample price data for volatility calculation
    np.random.seed(42)  # For reproducible test data
    base_price = 100.0
    num_days = 252  # Trading days in a year
    
    # Generate realistic price movements
    daily_returns = np.random.normal(0.0008, 0.02, num_days)  # ~20% annual vol
    prices = [base_price]
    
    for ret in daily_returns:
        prices.append(prices[-1] * (1 + ret))
    
    return {
        "symbol": "AAPL",
        "prices": prices,
        "returns": daily_returns.tolist(),
        "dates": [(datetime.now() - timedelta(days=i)).isoformat() for i in range(num_days + 1)],
        "historical_volatility": 0.20,
        "implied_volatility": 0.22,
        "garch_volatility": 0.21,
        "volatility_smile": {
            "strikes": [90, 95, 100, 105, 110],
            "implied_vols": [0.25, 0.23, 0.22, 0.24, 0.26]
        }
    }


@pytest.fixture
def sample_risk_context():
    """Sample risk analysis context for testing"""
    return {
        "portfolio": {
            "id": "PORTFOLIO_001",
            "name": "Test Portfolio", 
            "positions": [
                {"symbol": "AAPL", "quantity": 1000, "market_value": 150000},
                {"symbol": "GOOGL", "quantity": 500, "market_value": 125000},
                {"symbol": "MSFT", "quantity": 750, "market_value": 225000}
            ],
            "total_value": 1000000,
            "currency": "USD",
            "last_updated": datetime.now().isoformat()
        },
        "market_conditions": {
            "vix": 18.5,
            "interest_rates": {"fed_rate": 0.05, "10yr_treasury": 0.045},
            "credit_spreads": {"investment_grade": 0.012, "high_yield": 0.035},
            "fx_rates": {"EURUSD": 1.08, "GBPUSD": 1.25}
        },
        "risk_preferences": {
            "confidence_level": 0.95,
            "time_horizon_days": 1,
            "risk_measure": "VaR"
        },
        "analysis_type": "comprehensive"
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
            "payment_history": 0.95,
            "existing_debt": 28000,
            "assets": 150000
        },
        "loan_data": {
            "loan_type": "mortgage",
            "term_years": 30,
            "interest_rate": 0.045,
            "ltv_ratio": 0.80,
            "loan_purpose": "home_purchase",
            "property_type": "single_family",
            "property_value": 312500
        },
        "market_data": {
            "risk_free_rate": 0.025,
            "credit_spread": 0.015,
            "market_volatility": 0.18,
            "housing_index": 1.05
        }
    }


@pytest.fixture
def sample_transaction_data():
    """Sample transaction data for fraud testing"""
    return {
        "transaction_id": "TXN_001",
        "timestamp": datetime.now(),
        "amount": 25000.0,
        "currency": "USD",
        "user_id": "USER_001",
        "recent_transaction_count": 12,
        "location": "high_risk_country_1",
        "deviates_from_pattern": True,
        "account_from": "ACC_001",
        "account_to": "ACC_002",
        "transaction_type": "wire_transfer",
        "channel": "online",
        "risk_indicators": {
            "unusual_amount": True,
            "off_hours": True,
            "new_merchant": True,
            "velocity_trigger": True
        }
    }


@pytest.fixture
def sample_stress_portfolio():
    """Sample portfolio data for stress testing"""
    from FinAgents.agent_pools.risk_agent_pool.agents.stress_testing import PortfolioPosition
    
    return [
        PortfolioPosition(
            asset_id="AAPL",
            quantity=1000.0,
            current_price=150.0,
            currency="USD",
            asset_type="equity",
            sector="technology",
            beta=1.2
        ),
        PortfolioPosition(
            asset_id="GOOGL",
            quantity=500.0,
            current_price=2800.0,
            currency="USD",
            asset_type="equity",
            sector="technology", 
            beta=1.1
        ),
        PortfolioPosition(
            asset_id="JPM",
            quantity=2000.0,
            current_price=140.0,
            currency="USD",
            asset_type="equity",
            sector="financial",
            beta=1.3
        ),
        PortfolioPosition(
            asset_id="BOND_10Y",
            quantity=5000.0,
            current_price=95.0,
            currency="USD",
            asset_type="bond",
            sector="government",
            duration=8.5,
            beta=0.1
        ),
        PortfolioPosition(
            asset_id="CORP_BOND_001",
            quantity=3000.0,
            current_price=98.5,
            currency="USD",
            asset_type="bond",
            sector="corporate",
            duration=5.2,
            beta=0.3
        )
    ]


@pytest.fixture
def sample_model_metadata():
    """Sample model metadata for model risk testing"""
    from FinAgents.agent_pools.risk_agent_pool.agents.model_risk import ModelMetadata, ModelType, ModelStatus
    from datetime import datetime
    
    return ModelMetadata(
        model_id="RISK_MODEL_001",
        name="VaR Calculation Model",
        model_type=ModelType.RISK,
        version="1.2.0",
        developer="Risk Analytics Team",
        business_owner="Risk Management",
        description="Monte Carlo simulation based VaR model for equity and fixed income portfolios",
        purpose="Calculate daily VaR for trading portfolios",
        status=ModelStatus.APPROVED,
        created_date=datetime(2024, 1, 1),
        last_updated=datetime(2024, 6, 1),
        approval_date=datetime(2024, 2, 15),
        regulatory_classification="regulatory",
        criticality_level="high",
        documentation_links=["http://models.internal/risk_model_001"],
        dependencies=["market_data_feed", "portfolio_system"],
        data_sources=["bloomberg", "internal_positions"],
        tags=["var", "risk", "trading", "regulatory"]
    )


@pytest.fixture 
def sample_validation_config():
    """Sample validation configuration for model testing"""
    return {
        "validation_id": "VAL_001",
        "validation_type": "comprehensive",
        "test_suite": {
            "backtesting": {
                "enabled": True,
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "frequency": "daily",
                "confidence_levels": [0.95, 0.99],
                "exception_threshold": 0.05
            },
            "sensitivity_analysis": {
                "enabled": True,
                "risk_factors": ["equity_vol", "interest_rates", "credit_spreads"],
                "shock_sizes": [0.01, 0.05, 0.10, 0.25]
            },
            "stress_testing": {
                "enabled": True,
                "scenarios": [
                    "2008_financial_crisis",
                    "covid_2020",
                    "interest_rate_shock"
                ]
            },
            "benchmarking": {
                "enabled": True,
                "benchmark_models": ["historical_simulation", "parametric_var"],
                "comparison_metrics": ["accuracy", "coverage", "independence"]
            }
        },
        "data_quality_checks": {
            "completeness": True,
            "accuracy": True,
            "consistency": True,
            "timeliness": True
        },
        "performance_criteria": {
            "accuracy_threshold": 0.95,
            "coverage_threshold": 0.95,
            "independence_pvalue": 0.05,
            "max_runtime_seconds": 300
        },
        "documentation_requirements": [
            "model_specification",
            "validation_report",
            "performance_metrics",
            "limitation_analysis"
        ]
    }


class MockExternalMemory:
    """Mock external memory agent for testing"""
    
    def __init__(self):
        self.storage = {}
        self.events = []
        self.call_count = 0
    
    async def store_data(self, key: str, data: Dict[str, Any]) -> bool:
        """Mock store data method"""
        self.call_count += 1
        self.storage[key] = {
            "data": data,
            "timestamp": datetime.now(),
            "call_count": self.call_count
        }
        return True
    
    async def retrieve_data(self, key: str) -> Dict[str, Any]:
        """Mock retrieve data method"""
        self.call_count += 1
        stored_item = self.storage.get(key, {})
        return stored_item.get("data", {})
    
    async def record_event(self, event_data: Dict[str, Any]) -> str:
        """Mock record event method"""
        self.call_count += 1
        event_id = f"event_{len(self.events) + 1}"
        event_data["event_id"] = event_id
        event_data["timestamp"] = datetime.now()
        self.events.append(event_data)
        return event_id
    
    async def query_events(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock query events method"""
        self.call_count += 1
        # Simple filtering implementation
        filtered_events = []
        for event in self.events:
            match = True
            for key, value in filters.items():
                if key in event and event[key] != value:
                    match = False
                    break
            if match:
                filtered_events.append(event)
        return filtered_events
    
    def get_call_count(self) -> int:
        """Get total number of calls made to this mock"""
        return self.call_count
    
    def reset(self):
        """Reset mock state"""
        self.storage.clear()
        self.events.clear()
        self.call_count = 0


class MockOpenAIClient:
    """Mock OpenAI client for testing"""
    
    def __init__(self):
        self.call_count = 0
        self.responses = {}
        self.default_response = {
            "task_type": "risk_analysis",
            "agent_type": "market_risk_agent",
            "parameters": {
                "portfolio_data": {"test": "data"},
                "risk_measures": ["var", "volatility"],
                "confidence_level": 0.95
            }
        }
    
    def set_response(self, context: str, response: Dict[str, Any]):
        """Set specific response for a context"""
        self.responses[context] = response
    
    async def chat_completions_create(self, **kwargs):
        """Mock chat completion method"""
        self.call_count += 1
        
        # Extract user message from kwargs
        messages = kwargs.get("messages", [])
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Find matching response or use default
        response_data = self.default_response
        for context, response in self.responses.items():
            if context.lower() in user_message.lower():
                response_data = response
                break
        
        # Create mock response object
        class MockChoice:
            def __init__(self, content):
                self.message = MockMessage(content)
        
        class MockMessage:
            def __init__(self, content):
                self.content = json.dumps(content)
        
        class MockResponse:
            def __init__(self, data):
                self.choices = [MockChoice(data)]
        
        return MockResponse(response_data)
    
    def get_call_count(self) -> int:
        """Get total number of calls made to this mock"""
        return self.call_count
    
    def reset(self):
        """Reset mock state"""
        self.call_count = 0
        self.responses.clear()


@pytest.fixture
def mock_external_memory():
    """Fixture providing mock external memory"""
    return MockExternalMemory()


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = AsyncMock()
    
    # Mock chat completions
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"test": "response"}'
    
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    return mock_client


@pytest.fixture 
def mock_memory_bridge():
    """Mock memory bridge for testing"""
    mock_bridge = AsyncMock()
    
    # Mock memory operations
    mock_bridge.store_analysis_result = AsyncMock(return_value="test_analysis_id")
    mock_bridge.get_historical_analysis = AsyncMock(return_value=[])
    mock_bridge.store_market_data = AsyncMock(return_value="test_market_id")
    mock_bridge.get_market_data = AsyncMock(return_value=[])
    mock_bridge.log_event = AsyncMock(return_value="test_event_id")
    mock_bridge.get_event_history = AsyncMock(return_value=[])
    mock_bridge.cache_set = AsyncMock()
    mock_bridge.cache_get = AsyncMock(return_value=None)
    mock_bridge.cache_delete = AsyncMock()
    mock_bridge.health_check = AsyncMock(return_value={"status": "healthy"})
    
    return mock_bridge


@pytest.fixture
def mock_mcp_server():
    """Mock MCP server for testing"""
    mock_server = MagicMock()
    mock_server.serve = AsyncMock()
    mock_server.stop = AsyncMock()
    mock_server.is_running = False
    
    return mock_server


@pytest.fixture
async def mock_risk_pool(mock_external_memory, mock_openai_client):
    """Fixture providing mock risk agent pool"""
    from FinAgents.agent_pools.risk_agent_pool.core import RiskAgentPool
    
    # Create risk pool with mocked dependencies
    pool = RiskAgentPool(
        openai_api_key="test_key",
        external_memory_config={
            "host": "localhost",
            "port": 8000
        }
    )
    
    # Replace with mocks
    pool.memory_bridge.external_memory = mock_external_memory
    pool.openai_client = mock_openai_client
    
    return pool


@pytest.fixture
def sample_kri_metrics():
    """Sample KRI metrics for operational risk testing"""
    return {
        "system_downtime_hours": 2.5,
        "failed_transactions_pct": 0.02,
        "staff_turnover_rate": 0.08,
        "compliance_violations": 1,
        "fraud_incidents_monthly": 3,
        "customer_complaints": 15,
        "processing_errors": 8,
        "security_incidents": 0
    }


@pytest.fixture
def performance_test_data():
    """Large dataset for performance testing"""
    import random
    
    # Generate large portfolio for performance testing
    assets = []
    for i in range(100):  # 100 assets
        assets.append({
            "asset_id": f"ASSET_{i:03d}",
            "quantity": random.randint(10, 1000),
            "current_price": random.uniform(10, 500),
            "currency": "USD",
            "asset_type": random.choice(["equity", "bond", "commodity", "fx"]),
            "sector": random.choice(["technology", "finance", "healthcare", "energy"]),
            "beta": random.uniform(0.5, 2.0)
        })
    
    # Generate returns data
    returns_data = {}
    for asset in assets:
        returns_data[asset["asset_id"]] = [
            random.gauss(0, 0.02) for _ in range(252)  # One year of daily returns
        ]
    
    return {
        "positions": assets,
        "returns_data": returns_data
    }


class TestDataGenerator:
    """Utility class for generating test data"""
    
    @staticmethod
    def generate_random_portfolio(num_assets: int = 10) -> Dict[str, Any]:
        """Generate random portfolio data"""
        import random
        
        assets = []
        returns_data = {}
        
        for i in range(num_assets):
            asset_id = f"ASSET_{i:03d}"
            assets.append({
                "asset_id": asset_id,
                "quantity": random.randint(10, 1000),
                "current_price": random.uniform(10, 500),
                "currency": "USD",
                "asset_type": random.choice(["equity", "bond"]),
                "sector": random.choice(["technology", "finance", "healthcare"]),
                "beta": random.uniform(0.5, 2.0)
            })
            
            returns_data[asset_id] = [
                random.gauss(0, 0.02) for _ in range(30)  # 30 days of returns
            ]
        
        return {
            "positions": assets,
            "returns_data": returns_data
        }
    
    @staticmethod
    def generate_correlation_matrix(size: int) -> List[List[float]]:
        """Generate a valid correlation matrix"""
        import numpy as np
        
        # Generate random symmetric positive definite matrix
        A = np.random.randn(size, size)
        cov_matrix = np.dot(A, A.T)
        
        # Convert to correlation matrix
        D = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(D, D)
        
        return corr_matrix.tolist()


@pytest.fixture
def data_generator():
    """Fixture providing test data generator"""
    return TestDataGenerator()


# Async test utilities
class AsyncTestUtils:
    """Utilities for async testing"""
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 5.0):
        """Run coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise AssertionError(f"Test timed out after {timeout} seconds")
    
    @staticmethod
    async def assert_eventually(check_func, timeout: float = 5.0, interval: float = 0.1):
        """Assert that condition becomes true within timeout"""
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            if await check_func():
                return
            await asyncio.sleep(interval)
        
        raise AssertionError(f"Condition not met within {timeout} seconds")


@pytest.fixture
def async_utils():
    """Fixture providing async test utilities"""
    return AsyncTestUtils()
