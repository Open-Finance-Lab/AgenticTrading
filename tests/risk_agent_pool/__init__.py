"""
Risk Agent Pool Test Suite

Author: Jifeng Li
License: openMDW

This module contains comprehensive tests for the Risk Agent Pool including
unit tests, integration tests, and performance tests.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any

# Test configuration
pytest_plugins = ['pytest_asyncio']

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test data fixtures and utilities will be imported from other modules
from .fixtures import *
from .utils import *

__all__ = [
    'fixtures',
    'utils'
]
