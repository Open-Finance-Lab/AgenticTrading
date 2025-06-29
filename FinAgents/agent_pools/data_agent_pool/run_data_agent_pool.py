"""
Academic Data Agent Pool Launcher

This script launches the Data Agent Pool to sequentially crawl historical data for 30 stocks over the past three years using the Polygon API. Due to the API's restriction on parallel requests, all data acquisition is performed in a strictly sequential manner to ensure compliance and data integrity.

Key Features:
- Sequential data crawling for 30 equities
- Three-year historical window
- Robust error handling and logging
- Designed for academic research and reproducibility

Author: Jifeng Li
License: openMDW
"""

import os
import time
import logging
from datetime import datetime, timedelta
from FinAgents.agent_pools.data_agent_pool.core import DataAgentPool

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("DataAgentPoolLauncher")

# List of 30 target stock symbols (example: S&P 500 constituents or user-defined)
STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH",
    "HD", "PG", "MA", "DIS", "BAC", "XOM", "VZ", "ADBE", "CMCSA", "KO",
    "PFE", "CSCO", "PEP", "T", "ABT", "MRK", "WMT", "INTC", "CVX", "MCD"
]

# Polygon API key (set as environment variable for security)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    raise EnvironmentError("POLYGON_API_KEY environment variable not set.")

# Define the historical window (last 3 years)
END_DATE = datetime.utcnow().date()
START_DATE = END_DATE - timedelta(days=3*365)

# Initialize the Data Agent Pool
pool = DataAgentPool({
    "polygon_api_key": POLYGON_API_KEY,
    "sequential_mode": True,  # Enforce sequential crawling
    "log_level": "INFO"
})

logger.info(f"Starting sequential data crawl for {len(STOCK_SYMBOLS)} stocks from {START_DATE} to {END_DATE}.")

for symbol in STOCK_SYMBOLS:
    try:
        logger.info(f"Crawling data for {symbol}...")
        result = pool.crawl_historical_data(
            symbol=symbol,
            start_date=START_DATE,
            end_date=END_DATE,
            frequency="day"
        )
        logger.info(f"Completed: {symbol} ({len(result['data'])} records)")
        # Optional: Persist or process result['data'] as needed
        time.sleep(1.5)  # Respect API rate limits (adjust as required)
    except Exception as e:
        logger.error(f"Error crawling {symbol}: {e}")
        time.sleep(3)  # Backoff on error

logger.info("Data crawling for all stocks completed.")
