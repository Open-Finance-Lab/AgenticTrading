Execution and Trading
=====================

The FinAgent framework supports both simulated execution (Backtesting) and live/paper trading via the **Execution Agent**. This dual capability allows strategies to be validated historically before being deployed to real markets.

Backtesting (Simulated Execution)
---------------------------------

Backtesting is primarily handled by the **Orchestrator** and the **Alpha Agent**'s internal simulation logic. It is designed for high-speed validation of strategies over long historical periods using vectorized operations where possible.

**Mechanism:**

1.  **Signal Generation**: The Alpha Agent processes historical data (e.g., from Qlib or local CSVs) and generates a series of trading signals $S_t \in \{-1, 0, 1\}$ for the entire period.
2.  **Vectorized / Event Loop Simulation**: 
    *   **Vectorized (Fast)**: For simple strategies, signals are multiplied by returns to estimate performance: $R_{strategy} = S_{t-1} \times R_t - C_{trans}$.
    *   **Event-Driven**: The Orchestrator steps through the signals, simulating portfolio changes, applying transaction costs, and tracking equity curves.
3.  **Paper Interface**: The backtester mimics the interface of a live exchange, accepting "orders" and returning "fills" based on historical prices.

**Assumptions & Constraints:**
*   **Execution Price**: Often assumes execution at the *Close* or *Next Open* price.
*   **Slippage**: Can be modeled as a fixed basis point cost or a function of volatility.
*   **Market Impact**: Currently minimal in the vectorized engine, but supported in the event-driven mode.

**Key Components:**

*   **Orchestrator**: Manages the time-stepping and data feeding.
*   **Alpha Agent**: Generates the core signals.
*   **Performance Metrics**: Calculates Sharpe Ratio, Max Drawdown, and Total Return.

Paper Trading (Live Execution)
------------------------------

Paper trading (and live trading) is managed by the dedicated **Execution Agent**. This agent acts as the bridge between the AI decision-making layer and external brokerage APIs (specifically **Alpaca**).

**Execution Agent Architecture:**

.. code-block:: text

    +---------------------+       +-----------------------+       +-----------------------+
    |                     |       |                       |       |                       |
    |  Portfolio Agent    |------>|   Execution Agent     |------>|   Brokerage API       |
    |  (Target Weights)   |       |   (Trade Executor)    |       |   (e.g., Alpaca)      |
    |                     |       |                       |       |                       |
    +---------------------+       +-----------+-----------+       +-----------+-----------+
                                              |                               ^
                                              | Uses Tools                    |
                                              v                               |
                                  +-----------------------+                   |
                                  |  - get_account_summary|-------------------+
                                  |  - get_positions      |-------------------+
                                  |  - execute_orders     |-------------------+
                                  +-----------------------+

**Capabilities:**

*   **Account Monitoring**: Checks available buying power (cash) and current portfolio value (`get_account_summary`).
*   **Position Management**: Retrieves currently held assets to calculate necessary rebalancing trades (`get_current_positions`).
*   **Order Execution**: Converts high-level instructions (e.g., "Rebalance to 50% AAPL") into specific market or limit orders and submits them to the broker (`execute_orders`).
*   **Safety Checks**: Ensures sufficient funds before placing buy orders to prevent rejections (Pre-Trade Compliance).

**Configuration:**

The Execution Agent is initialized with API keys and a mode flag. It gracefully falls back to a "Mock Mode" if keys are missing, ensuring the pipeline can be tested without external dependencies.

.. code-block:: python

   # Example Initialization
   execution_agent = ExecutionAgent(
       alpaca_api_key="...",
       alpaca_secret_key="...",
       paper=True  # Set to False for real money trading
   )

**Workflow:**

1.  **Instruction**: Receives a target portfolio $W_{target}$ or specific trade list from the Portfolio Agent.
2.  **State Sync**: Queries the broker for current positions $W_{current}$ and cash.
3.  **Diff Calculation**: Calculates the difference $\Delta W = W_{target} - W_{current}$.
4.  **Execution**: Generates and submits the necessary orders (Buy/Sell) to align the portfolio.
5.  **Feedback**: Returns the execution result (filled orders, errors) to the Orchestrator for logging and state update.

This separation ensures that the *strategy logic* (Alpha/Risk/Portfolio) remains decoupled from the *execution mechanics* (API calls, order types), allowing for easier switching between brokers or simulation modes.
