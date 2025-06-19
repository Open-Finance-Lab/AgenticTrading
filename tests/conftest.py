import pytest
import logging

@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests"""
    logging.basicConfig(level=logging.DEBUG)

@pytest.fixture(scope="session")
def mock_registry():
    """Provide mock agent registry for tests"""
    return {
        "binance": Mock(),
        "equity_provider": Mock(),
        "news_service": Mock()
    }