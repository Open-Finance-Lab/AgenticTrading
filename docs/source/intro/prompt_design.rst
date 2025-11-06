==================
Prompt Design
==================

This section describes the prompt engineering framework for the Agentic Trading system.
The design follows a modular **Data → Alpha → Risk → Portfolio → Backtest** pipeline, 
where each agent receives structured inputs and produces standardized JSON outputs.

Pipeline Overview
==================

The agentic trading workflow consists of five specialized agents:

.. list-table:: Agent Pipeline
   :widths: 20 25 30 25
   :header-rows: 1

   * - Agent
     - Input
     - Core Task
     - Output
   * - **DataAgent**
      - Asset types, time range, scale, data type
      - Fetch, clean, and organize market data
      - Standardized dataset + pointer + event_id
   * - **AlphaAgent**
     - Historical OHLCV + alpha factors
     - Mine and quantify expected excess return
     - ``expected_excess_return``
   * - **RiskAgent**
     - Alpha output + covariance matrix
     - Evaluate risk, volatility, constraints
     - ``risk_score``, adjusted weights
   * - **PortfolioAgent**
     - Alpha + Risk signals
     - Optimize portfolio weights
     - Investment weights, allocation plan
   * - **BacktestAgent**
     - Portfolio weights + execution rules
     - Simulate historical trading
     - Return, volatility, Sharpe ratio

Design Principles
==================

1. **Modularity**: Each agent has a single responsibility and can be independently tested.
2. **Standardization**: All agents communicate via JSON with predefined schemas.
3. **Reproducibility**: Prompts explicitly define lookback windows, constraints, and objectives.
4. **Cross-Asset Generality**: The same structure applies to both stock and cryptocurrency tasks.


1. Data Agent Prompt
======================

**Objective**: Fetch, clean, and organize market data for downstream agents.

Prompt Template
-----------------

.. code-block:: text

   You are a Data Agent responsible for fetching and organizing market data.
   Your goal is to retrieve, validate, and prepare standardized datasets
   for use by downstream agents (Alpha, Risk, Portfolio, Backtest).

   Input:
   - Asset types: ["stocks", "crypto", "futures", "options"]
   - Time range: start_date, end_date (ISO8601 format)
   - Scale: ["1min", "5min", "15min", "1h", "1d"]
   - Data type: ["OHLCV", "LOB", "trades", "fundamentals", "alternative"]
   - Symbols: List of tickers or contracts
   - Additional filters: exchanges, data quality requirements

   Task:
   1. Fetch data from appropriate sources (APIs, databases, files).
   2. Validate data quality (missing values, outliers, timestamps).
   3. Apply standardization (timezone normalization, schema alignment).
   4. Store processed data and generate access pointer.
   5. Return metadata with event_id for tracking and reproducibility.

   Output Example:
   {
     "event_id": "DATA_20250106_143052_ABC123",
     "status": "success",
     "dataset_info": {
       "asset_type": "stocks",
       "symbols": ["AAPL", "MSFT", "GOOGL"],
       "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
       "scale": "1d",
       "data_type": "OHLCV",
       "total_records": 756,
       "missing_rate": 0.002
     },
     "data_pointer": {
       "storage_type": "file",
       "path": "/data/cache/stocks_OHLCV_1d_20240101_20250101.parquet",
       "format": "parquet",
       "schema_version": "v2.1"
     },
     "quality_checks": {
       "timestamp_continuity": "pass",
       "price_sanity": "pass",
       "volume_outliers": "flagged_3_records",
       "data_completeness": 0.998
     },
     "metadata": {
       "source": "polygon.io",
       "fetch_timestamp": "2025-01-06T14:30:52Z",
       "processing_time_sec": 4.23,
       "cache_enabled": true
     }
   }

Key Parameters
---------------

- **Asset Types**: 
   - ``stocks``: Equity securities (NYSE, NASDAQ, etc.)
   - ``crypto``: Cryptocurrencies (BTC, ETH, etc.)
   - ``futures``: Futures contracts
   - ``options``: Option contracts

- **Time Range**: ISO8601 format (YYYY-MM-DDTHH:MM:SSZ)
   - ``start_date``: Beginning of data window
   - ``end_date``: End of data window

- **Scale**: Data frequency
   - Intraday: ``1min``, ``5min``, ``15min``, ``1h``
   - Daily: ``1d``
   - Weekly/Monthly: ``1w``, ``1M``

- **Data Types**:
   - ``OHLCV``: Open, High, Low, Close, Volume
   - ``LOB``: Limit Order Book (Level 1/2/3)
   - ``trades``: Tick-by-tick trade data
   - ``fundamentals``: Financial statements, ratios
   - ``alternative``: Sentiment, news, social media

- **Output Schema**: 
   - ``event_id``: Unique identifier (format: DATA_YYYYMMDD_HHMMSS_HASH)
   - ``data_pointer``: Storage location and access information
   - ``dataset_info``: Metadata about the dataset
   - ``quality_checks``: Data validation results

Implementation Details

The Data Agent follows a multi-stage pipeline:

1. **Fetch Stage**:
   - Connect to data sources (Polygon, Yahoo Finance, Binance, etc.)
   - Handle API rate limits and retries
   - Support batch fetching for multiple symbols
   - Cache results to avoid redundant requests

2. **Validation Stage**:
   - Check timestamp continuity and timezone consistency
   - Detect and flag price/volume outliers using statistical methods
   - Identify missing data and forward-fill where appropriate
   - Validate schema compliance

3. **Standardization Stage**:
   - Normalize all timestamps to UTC
   - Align column names to standard schema
   - Convert data types (float64 for prices, int64 for volume)
   - Add metadata columns (source, fetch_time, quality_score)

4. **Storage Stage**:
   - Save to efficient format (Parquet for columnar, HDF5 for time-series)
   - Generate data pointer with access information
   - Update data catalog/registry

5. **Event Tracking**:
   - Generate unique ``event_id`` for reproducibility
   - Log fetch parameters and results
   - Enable data lineage tracking

Data Quality Metrics
---------------------

.. math::

   	ext{completeness} = 1 - \frac{\text{missing\_records}}{\text{total\_expected\_records}}

.. math::

   	ext{quality\_score} = w_1 \cdot \text{completeness} + w_2 \cdot \text{continuity} + w_3 \cdot \text{sanity}

where:

- ``completeness``: Ratio of present vs. expected records
- ``continuity``: Timestamp gaps within acceptable range
- ``sanity``: Price/volume within expected statistical bounds

Error Handling
---------------

The Data Agent handles common failure modes:

- **API Failures**: Retry with exponential backoff, fallback to cached data
- **Partial Data**: Return partial results with warning, mark incomplete periods
- **Schema Mismatch**: Attempt automatic mapping, flag unmapped fields
- **Outliers**: Flag but don't remove (downstream agents decide treatment)

Example Use Cases
------------------

**Stock Portfolio (Daily OHLCV)**:

.. code-block:: json

   {
     "asset_type": "stocks",
     "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
     "time_range": {"start": "2022-01-01", "end": "2025-01-01"},
     "scale": "1d",
     "data_type": "OHLCV"
   }

**Crypto High-Frequency (1min LOB)**:

.. code-block:: json

   {
     "asset_type": "crypto",
     "symbols": ["BTC-USD", "ETH-USD"],
     "time_range": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-02T00:00:00Z"},
     "scale": "1min",
     "data_type": "LOB",
     "lob_depth": 10
   }

**Multi-Asset with Fundamentals**:

.. code-block:: json

   {
     "asset_type": "stocks",
     "symbols": ["AAPL", "MSFT"],
     "time_range": {"start": "2023-01-01", "end": "2024-12-31"},
     "scale": "1d",
     "data_type": ["OHLCV", "fundamentals"],
     "fundamentals_fields": ["pe_ratio", "market_cap", "revenue"]
   }


2. Alpha Agent Prompt
======================

**Objective**: Estimate the expected excess return of each asset based on mined alpha factors.

Prompt Template
-----------------

.. code-block:: text

   You are an Alpha Agent analyzing financial time series and factor data.
   Your goal is to estimate the expected excess return of each asset
   based on mined alpha factors.

   Input:
   - Historical OHLCV data and derived alpha factors (e.g., momentum, volatility, value)
   - Lookback window: 60 bars
   - Target horizon: 20 bars ahead

   Task:
   1. Compute factor-weighted signals.
   2. Predict expected excess return for each asset.
   3. Output JSON with {"ticker","expected_excess_return","confidence"}.

   Output Example:
   {
     "AAPL": {"expected_excess_return": 0.015, "confidence": 0.82},
     "MSFT": {"expected_excess_return": 0.009, "confidence": 0.78},
     "TSLA": {"expected_excess_return": -0.006, "confidence": 0.73}
   }

Key Parameters
---------------

- **Lookback Window**: 60 trading days (≈3 months)
- **Prediction Horizon**: 20 trading days (≈1 month)
- **Alpha Factors**: Momentum, mean reversion, volatility, volume, fundamental ratios
- **Output Schema**: ``{ticker: {expected_excess_return: float, confidence: float}}``

Implementation Details
-----------------------

The Alpha Agent computes a weighted combination of alpha factors:

.. math::

   \alpha_i = \sum_{k=1}^{K} w_k \cdot f_{i,k}

where :math:`f_{i,k}` is the k-th factor for asset i, and :math:`w_k` are learned weights.

Extended Interface & Output Modes
---------------------------------

The Alpha Agent supports two output paradigms aligned with downstream usage:

1. **Cumulative Horizon Prediction (Default)**: Single expected excess return over the next ``H`` bars. Stable and directly consumable by Portfolio optimization.
2. **Path Prediction (Optional)**: Multi-step sequence :math:`[\hat{r}_{t+1}, \hat{r}_{t+2}, \dots, \hat{r}_{t+H}]` for execution timing or intra-horizon scheduling.

Selection Guidance:

- Use cumulative only when Portfolio holds weights static for horizon H.
- Enable path output only if Execution layer consumes temporal shape (e.g. front/back-loaded signal).

Additional Input Fields (Optional):

.. code-block:: json

    {
       "horizon_bars": 20,
       "lookback_bars": 60,
       "factors_ref": "s3://.../factors.parquet",
       "returns_ref": "s3://.../returns.parquet",
       "regime_signal": "calm"
    }

Augmented Output Schema (Minimal):

.. code-block:: json

    {
       "AAPL": {
          "y_pred_h": 0.015,
          "confidence": 0.82,
          "q05": 0.004,
          "q50": 0.013,
          "q95": 0.026,
          "path_ref": null,
          "shape_tag": null
       }
    }

Where:

- ``y_pred_h``: Expected cumulative excess return next H bars.
- ``confidence``: Calibrated score (0–1) from historical hit-rate or probabilistic model.
- ``q05/q50/q95``: Optional uncertainty quantiles (enable for robust allocation).
- ``path_ref``: Pointer to external multi-step prediction table (only when path mode active).
- ``shape_tag``: Temporal morphology label (e.g., "front_loaded", "back_loaded", "u_shape").

Failure / Degradation Modes:

- If path generation exceeds latency budget, fall back to cumulative only and set ``path_ref=null``.
- If factor coverage < threshold (e.g., 90%), reduce confidence proportionally.

Policy Trace (Optional Logging):

.. code-block:: json

    {
       "policy_trace": {
          "mode": "cumulative",
          "regime": "calm",
          "factor_count": 12,
          "uncertainty_enabled": true
       }
    }
3. Risk Agent Prompt
======================

**Objective**: Evaluate and adjust alpha signals based on risk exposures and constraints.

Prompt Template
-----------------

.. code-block:: text

   You are a Risk Agent responsible for evaluating and adjusting alpha signals.
   You receive expected excess returns and estimate the corresponding risk exposures.

   Input:
   - AlphaAgent output (expected_excess_return)
   - Historical covariance matrix Σ
   - Factor exposure matrix B
   - Risk constraints: target_vol = 0.15, max_leverage = 2.0, beta_neutral = true

   Task:
   1. Estimate volatility and systematic risk per asset.
   2. Compute risk-adjusted score (alpha / σ).
   3. Return adjusted risk signal and suggested scaling factor.

   Output Example:
   {
     "AAPL": {"risk_score": 0.011, "volatility": 0.14, "risk_scaler": 0.91},
     "MSFT": {"risk_score": 0.007, "volatility": 0.12, "risk_scaler": 0.86},
     "TSLA": {"risk_score": -0.004, "volatility": 0.21, "risk_scaler": 0.73}
   }

Extended Input Interface
------------------------

In addition to alpha, covariance (Σ), factor exposure (B), and constraints, the Risk Agent can consume richer contextual metadata to drive adaptive tooling:

.. code-block:: json

    {
       "alpha_ref": "s3://.../pred_table.parquet",
       "sigma_ref": "s3://.../Sigma.parquet",
       "exposure_ref": "s3://.../B.parquet",
       "returns_ref": "s3://.../returns.parquet",
       "liquidity_ref": "s3://.../liq.parquet",
       "group_map_ref": "s3://.../sector_map.parquet",
       "horizon_bars": 20,
       "constraints": {
          "target_vol": 0.15,
          "gross_leverage_max": 2.0,
          "beta_neutral": true,
          "group_limits": {"sector": 0.3}
       },
       "regime_signal": "turbulent",
       "cost_model": {"turnover_bp": 10, "impact_fn": "square_root"}
    }

Tool Catalog (Function-Calling / MCP)
-------------------------------------

Estimation / Computation:

- ``risk.compute_sigma(returns_ref, horizon_bars, method)`` → ``sigma_ref``
- ``risk.compute_vol(returns_ref, horizon_bars, method)`` → per-asset volatility table
- ``risk.factor_exposure(prices_or_features_ref, model)`` → ``exposure_ref``
- ``risk.var_cvar(returns_ref, horizon_bars, method, alpha)`` → {"var": float, "cvar": float, "ref": str}

Neutralization / Scaling / Gating:

- ``risk.neutralize(alpha_ref, exposure_ref, scheme)`` → ``alpha_neutral_ref``
- ``risk.scale_by_vol(alpha_ref|alpha_neutral_ref, vol_table_ref, target_vol)`` → ``scaled_scores_ref``, ``scalers_ref``
- ``risk.enforce_limits(scores_ref, constraints, liquidity_ref, group_map_ref)`` → ``gated_scores_ref``, ``violations``

Optimization / Diagnostics:

- ``risk.solve_qp(scores_ref, sigma_ref, constraints, cost_model)`` → ``weights_ref``, ``duals``, ``status``
- ``risk.regime_detect(returns_ref)`` → {"regime_label": str, "evidence": dict}
- ``risk.stress_scenarios(weights_ref, scenarios_ref)`` → ``stress_report_ref``

Method Values:

- ``method``: ``"ewma" | "garch" | "lw_shrink" | "oas" | "pca_factor"``
- ``scheme``: ``"beta_neutral" | "sector_neutral" | "multi_factor_neutral"``

Policy & Decision Rules
------------------------

- Covariance selection: if ``regime_signal == turbulent`` → use ``oas`` + shorter lookback; else ``ledoit_wolf``.
- Volatility estimation: intraday or high realized vol → ``ewma(λ=0.94)``; else ``garch(1,1)``.
- Neutralization order: ``beta → sector → style (optional)``.
- Gating priority: feasibility > leverage > group_limits > liquidity.
- Infeasible optimization: relax secondary constraints (e.g., group limits +10%) and re-solve; keep leverage & beta strict.
- Cost integration: incorporate turnover_bp into effective scores or as penalty term in QP (L1/L2 regularization).
- Fallback on deadline breach: perform simple volatility scaling + soft group limits, skip QP.

Execution Sequence (Minimal Deterministic Flow):

1. Compute / update Σ^H and per-asset σ^H.
2. Neutralize alpha (beta → sector → style).
3. Volatility scale to target effective risk.
4. Enforce limits & liquidity gating.
5. (Optional) Solve preliminary QP for candidate weights.
6. Produce diagnostics + artifacts + trace.

Extended Output Schema
-----------------------

Artifacts & References:

.. code-block:: json

    {
       "per_asset_ref": "s3://.../risk_per_asset.parquet",
       "scaled_scores_ref": "s3://.../scaled_scores.parquet",
       "gated_scores_ref": "s3://.../gated_scores.parquet",
       "weights_prelim_ref": "s3://.../weights_qp.parquet",
       "portfolio_diagnostics": {
          "forecast_vol_est": 0.142,
          "target_vol": 0.150,
          "gross_leverage": 1.82,
          "net_beta": 0.01,
          "qp_status": "optimal",
          "violations": []
       },
       "tool_calls": [
          {"fn": "risk.compute_sigma", "method": "oas", "lookback": "252d", "rt_ms": 183},
          {"fn": "risk.scale_by_vol", "target_vol": 0.15, "rt_ms": 21},
          {"fn": "risk.enforce_limits", "rules": ["gross<=2.0", "beta≈0", "sector<=0.3"], "rt_ms": 12}
       ],
       "policy_trace": {
          "regime": "turbulent",
          "sigma_method": "oas",
          "neutral_order": ["beta", "sector"],
          "fallback_used": false
       }
    }

Per-Asset Table Columns (referenced by ``per_asset_ref``):

- ``risk_score``
- ``volatility``
- ``risk_scaler``
- ``beta`` (post-neutralization residual)
- ``group`` (e.g., sector label)
- ``liquidity_gate`` (bool or enum: pass / reduced / blocked)

Failure & Uncertainty Handling:

- Infeasible QP → set ``qp_status="infeasible"``; include ``infeasible_reason`` & ``relax_suggestion``.
- Provide optional ``vol_ci`` and ``var_cvar_ci`` when extreme regime detected.
- Allow streaming partial outputs: set ``partial=true`` for early scaled scores before gating/QP.
- Include ``deadline_ms`` in inputs; if exceeded, record fallback path in ``policy_trace.fallback_used=true``.

Prompt Snippet (English for tool selection logic):

.. code-block:: text

    If optimization is infeasible, relax secondary constraints (group limits) by 10% and re-solve. Keep leverage and beta neutrality hard. Log all violations and duals. On timeout, skip QP and emit scaled_scores_ref + diagnostics with fallback_used=true.

Risk Metrics
-------------

**Risk Score**: Risk-adjusted expected return

.. math::

   \text{risk\_score}_i = \frac{\alpha_i}{\sigma_i}

where :math:`\sigma_i` is the estimated volatility of asset i.

**Risk Scaler**: Adjustment factor to meet portfolio-level constraints

.. math::

   \text{scaler}_i = \min\left(1, \frac{\sigma_{\text{target}}}{\sigma_i}\right)

Constraints
------------

- **Target Volatility**: 15% annualized
- **Max Leverage**: 2.0 (gross exposure ≤ 2× capital)
- **Beta Neutral**: Net market beta ≈ 0


4. Portfolio Agent Prompt
===========================

**Objective**: Combine alpha and risk signals into actionable portfolio weights.

Prompt Template
-----------------

.. code-block:: text

   You are a Portfolio Agent combining alpha and risk signals into actionable portfolio weights.

   Input:
   - AlphaAgent expected_excess_return
   - RiskAgent risk_score and volatility
   - Constraints: gross_leverage ≤ 2, position_limit ≤ 0.25, sector_neutral = true

   Task:
   1. Optimize portfolio weights w to maximize Sharpe ratio:
      maximize (wᵀα) / sqrt(wᵀΣw)
   2. Apply constraints and normalization.
   3. Output final allocation plan and turnover suggestion.

   Output Example:
   {
     "weights": {"AAPL": 0.20, "MSFT": 0.15, "TSLA": -0.05},
     "expected_portfolio_return": 0.012,
     "expected_portfolio_vol": 0.14,
     "target_sharpe": 0.85,
     "execution_suggestion": "rebalance_monthly, strategies ....."
   }

Optimization Objective
-----------------------

The Portfolio Agent solves a mean-variance optimization problem:

.. math::

   \max_w \frac{w^T \alpha}{\sqrt{w^T \Sigma w}}

subject to:

.. math::

   \begin{aligned}
   &\sum_i |w_i| \leq 2 \quad &\text{(gross leverage)} \\
   &|w_i| \leq 0.25 \quad &\text{(position limit)} \\
   &\sum_i w_i \cdot \beta_i = 0 \quad &\text{(beta neutral)} \\
   &w^T \Sigma w \leq \sigma_{\text{target}}^2 \quad &\text{(volatility target)}
   \end{aligned}

Turnover Control
-----------------

To reduce transaction costs, the agent applies a turnover penalty:

.. math::

   \text{turnover\_cost} = \lambda \sum_i |w_i^{\text{new}} - w_i^{\text{old}}|

where :math:`\lambda` is the transaction cost rate (default: 10 bps).


5. Backtest Agent Prompt
==========================

**Objective**: Simulate portfolio performance and report risk-adjusted metrics.

Prompt Template
-----------------

.. code-block:: text

   You are a Backtest Agent simulating the performance of a trading portfolio.

   Input:
   - PortfolioAgent weights and rebalance rules
   - Historical market data (prices, costs, slippage)
   - Backtest period: 2024-01-01 → 2025-01-01

   Task:
   1. Apply daily rebalancing based on given weights.
   2. Compute cumulative return, volatility, Sharpe ratio, and drawdown.
   3. Generate textual performance summary and JSON metrics.

   Output Example:
   {
     "cumulative_return": 0.238,
     "annual_volatility": 0.148,
     "sharpe_ratio": 1.61,
     "max_drawdown": 0.056,
     "summary": "The strategy achieved a 23.8% annual return with a Sharpe of 1.61."
   }

Performance Metrics
--------------------

**Cumulative Return**:

.. math::

   R_{\text{cum}} = \prod_{t=1}^{T} (1 + r_t) - 1

**Sharpe Ratio**:

.. math::

   \text{Sharpe} = \frac{\mathbb{E}[r_t - r_f]}{\sigma(r_t)}

**Maximum Drawdown**:

.. math::

   \text{MDD} = \max_{t} \left(\frac{\max_{s \leq t} V_s - V_t}{\max_{s \leq t} V_s}\right)

where :math:`V_t` is the portfolio value at time t.

Execution Model
----------------

- **Rebalance Frequency**: Daily at market close
- **Transaction Costs**: 10 bps per trade + slippage
- **Slippage Model**: :math:`\text{slippage} = 0.05\% \times \sqrt{\text{trade\_size}/\text{ADV}}`


6. BTC Agent Prompt (Cryptocurrency)
======================================

**Objective**: Apply the same Alpha–Risk–Portfolio–Backtest framework to minute-level BTC-USDT trading.

Prompt Template
-----------------

.. code-block:: text

   You are a Crypto Trading Agent for BTC-USDT using minute-level data.
   You follow the same Alpha → Risk → Portfolio → Backtest structure.

   Input:
   - Price, volume, EMA, RSI, MACD
   - Funding rate, open interest, volatility index
   - Position info: qty, leverage, unrealized_pnl

   Task:
   1. Predict short-horizon excess return from features.
   2. Adjust signal based on volatility and funding risk.
   3. Allocate position within leverage/risk limits.
   4. Evaluate via rolling-minute backtest.

   Output Example:
   {
     "alpha_signal": 0.012,
     "risk_adjustment": 0.86,
     "position_size": 0.42,
     "expected_pnl_next_hour": 0.009,
     "decision": "hold"
   }

Cryptocurrency-Specific Features
----------------------------------

1. **Funding Rate Risk**: Adjusts position size to avoid excessive funding payments
2. **Liquidation Risk**: Ensures leverage never exceeds safe threshold based on volatility
3. **24/7 Trading**: No market close; continuous position monitoring
4. **Higher Frequency**: 1-minute bars instead of daily bars

Risk Adjustments
-----------------

.. math::

   \text{position\_size} = \text{base\_size} \times \frac{\sigma_{\text{target}}}{\sigma_{\text{realized}}} \times (1 - \text{funding\_penalty})

where:

- :math:`\sigma_{\text{realized}}` is the rolling 1-hour realized volatility


7. Portfolio ↔ Execution Agent Interface
=====================================

Purpose
--------

Define a simple, reliable contract from PortfolioAgent (target weights) to ExecutionAgent (orders), aligned with our execution operating mode: LIMIT orders anchored at mid-price, DAY time-in-force, and TWAP slicing for large notionals.

Contract Overview
------------------

- Inputs (from PortfolioAgent to ExecutionAgent)
   - ``rebalance_id``: unique idempotency token (string)
   - ``timestamp``: ISO8601 time when weights are produced
   - ``effective_time``: when orders should start (e.g., next market open)
   - ``portfolio_value``: float, USD
   - ``current_positions``: {symbol: shares}
   - ``prices``: {symbol: {"mid": float, "last": float}}
   - ``target_weights``: {symbol: float} where sum(weights) ≤ 1.0 (cash = 1 − Σw)
   - ``constraints`` (optional):
      - ``max_position_weight`` (default 0.30)
      - ``min_trade_value`` (default 100.0 USD)
      - ``min_weight_change`` (default 0.002, i.e., 20 bps)
      - ``round_lot`` (default 1 share)
      - ``max_participation_rate`` (default 0.1)
      - ``cash_buffer`` (default 0.0)
   - ``execution_prefs`` (optional):
      - ``default_order_type``: "LIMIT" | "MARKET" (default "LIMIT")
      - ``price_anchor``: "mid" | "last" | "bid" | "ask" (default "mid")
      - ``tif``: "DAY" (default)
      - ``twap_threshold_notional``: e.g., 100000.0 USD
      - ``twap_slices``: e.g., 4
      - ``twap_interval_sec``: e.g., 600
   - ``cost_params`` (optional):
      - ``commission_bps``: e.g., 10
      - ``slippage_model``: "linear" | "kyle"
      - ``slippage_rate``: e.g., 0.0005

- Outputs (from ExecutionAgent back to Portfolio/Orchestrator)
   - ``orders``: list of OrderSpec
   - ``ack``: {"status": "accepted"|"rejected"|"queued", "reason"?: str}
   - ``errors``: [] (if any symbol rejected)
   - ``execution_report_refs``: optional mapping {parent_order_id → analysis_endpoint or report_id}

Mapping Weights → Orders
-------------------------

For each symbol i:

.. code-block:: text

    target_shares_i = floor(((portfolio_value * target_weight_i) / price_i) / round_lot) * round_lot
    delta_shares_i  = target_shares_i − current_shares_i

Filter tiny trades:

- Ignore if |delta_shares_i| × price_i < ``min_trade_value``
- Ignore if |target_weight_i − current_weight_i| < ``min_weight_change``

Order construction:

- side = BUY if delta_shares_i > 0 else SELL
- order_type = execution_prefs.default_order_type (default LIMIT)
- limit_price = price_anchor(mid by default)
- tif = execution_prefs.tif (DAY)
- large notional (|delta_shares_i| × price_i ≥ twap_threshold_notional) → use ``algo = TWAP`` with slices/interval

OrderSpec (JSON)
-----------------

.. code-block:: json

    {
       "symbol": "AAPL",
       "side": "SELL",
       "quantity": 336,
       "order_type": "LIMIT",
       "limit_price": 175.50,
       "time_in_force": "DAY",
       "algo": {"type": "TWAP", "slices": 4, "interval_sec": 600},
       "participation_limit": 0.10,
       "parent_order_id": "REB_20231229_AAPL",
       "rebalance_id": "REB_20231229",
       "expire_at": "2023-12-29T20:00:00Z",
       "risk_checks": {"price_band_bps": 100, "notional_cap": 250000.0}
    }

Acknowledgements & States
--------------------------

- ``accepted``: order(s) queued or submitted
- ``rejected``: symbol invalid, below notional/lot, violates constraints (include reason)
- ``partial`` / ``filled`` handled in execution reports

Idempotency & Retries
----------------------

- ``rebalance_id`` ensures duplicate submissions are ignored or reconciled
- If child slices time out, fallback policy:
   - Retry once; if still unfilled > 70% of delta, optionally switch remaining to MARKET near close or cancel remainder

Execution Reports (Post-Trade)
--------------------------------

Use the existing models in codebase for post-trade analysis:

- ``FinAgents/agent_pools/transaction_cost_agent_pool/schema/execution_schema.py``
   - ``ExecutionReport``, ``TradeExecution``, ``QualityMetrics``

Each ``OrderSpec.parent_order_id`` should map to ``TradeExecution.parent_order_id`` for attribution.

Minimal Envelope Example (7-stock weekly rebalance)
---------------------------------------------------

.. code-block:: json

    {
       "rebalance_id": "REB_2023W52",
       "timestamp": "2023-12-29T14:59:00Z",
       "effective_time": "2023-12-29T15:30:00Z",
       "portfolio_value": 1285000.0,
       "current_positions": {"AAPL": 1800, "MSFT": 5500, "GOOGL": 900, "AMZN": 700, "TSLA": 750, "META": 800, "NVDA": 600},
       "prices": {"AAPL": {"mid": 175.50}, "MSFT": {"mid": 370.25}, "TSLA": {"mid": 255.10}, "GOOGL": {"mid": 141.80}, "AMZN": {"mid": 153.20}, "META": {"mid": 351.00}, "NVDA": {"mid": 486.00}},
       "target_weights": {"AAPL": 0.20, "MSFT": 0.22, "GOOGL": 0.15, "AMZN": 0.12, "TSLA": 0.10, "META": 0.13, "NVDA": 0.08},
       "constraints": {"max_position_weight": 0.30, "min_trade_value": 100.0, "min_weight_change": 0.002, "round_lot": 1, "max_participation_rate": 0.10, "cash_buffer": 0.00},
       "execution_prefs": {"default_order_type": "LIMIT", "price_anchor": "mid", "tif": "DAY", "twap_threshold_notional": 100000.0, "twap_slices": 4, "twap_interval_sec": 600},
       "cost_params": {"commission_bps": 10, "slippage_model": "linear", "slippage_rate": 0.0005}
    }

Instruction Templates
----------------------

- PortfolioAgent → "Produce target_weights with the envelope fields above (rebalance_id, portfolio_value, constraints, execution_prefs). Ensure weights satisfy constraints and include current_positions snapshot."
- ExecutionAgent → "Given the envelope, compute delta shares per symbol with rounding and thresholds, then emit OrderSpec list. Use LIMIT@mid with DAY TIF; apply TWAP when notional ≥ threshold. Return ack and link future ExecutionReport IDs."

Prompt Flow Summary
====================

.. list-table:: Complete Prompt Structure
   :widths: 15 35 25 25
   :header-rows: 1

   * - Agent
     - Core Prompt Instruction
     - Key Output Fields
     - Constraints
      * - **Data**
         - "Fetch and organize market data; validate quality."
         - ``event_id``, ``data_pointer``, ``dataset_info``
         - Quality > 99%, Schema v2.1
   * - **Alpha**
     - "Estimate expected excess return based on alpha factors."
     - ``expected_excess_return``, ``confidence``
     - Lookback=60, Horizon=20
   * - **Risk**
     - "Evaluate volatility and scale exposure under constraints."
     - ``risk_score``, ``volatility``, ``risk_scaler``
     - ``target_vol=0.15``, ``beta_neutral``
   * - **Portfolio**
     - "Optimize weights to maximize risk-adjusted return."
     - ``weights``, ``expected_return``, ``target_sharpe``
     - ``gross_leverage≤2``, ``position≤0.25``
   * - **Backtest**
     - "Simulate portfolio and report performance metrics."
     - ``sharpe_ratio``, ``max_drawdown``, ``summary``
     - Transaction cost=10bps


Methodology Statement (For Paper)
===================================

.. note::

       **Prompt-Driven Agentic Trading Pipeline: Data → Alpha → Risk → Portfolio → Backtest**

       Each trading prompt follows a unified Data–Alpha–Risk–Portfolio–Backtest framework.
       The Data Agent fetches and standardizes market data, ensuring quality and reproducibility.
   The Alpha Agent computes the future expected excess return based on mined alpha factors.
   The Risk Agent evaluates volatility and systematic exposure, adjusting signal magnitude.
   The Portfolio Agent integrates alpha and risk signals into an optimized asset allocation 
   under leverage and exposure constraints.
   Finally, the Backtest Agent simulates trading outcomes, reporting Sharpe ratio, 
   drawdown, and turnover statistics.
   
   This modular prompt structure ensures **interpretability**, **reproducibility**, 
   and **cross-asset generality** across both stock and crypto tasks.


Reproducibility
=================

All prompts are version-controlled and logged with:

- **Model**: GPT-4 / Claude-3.5
- **Temperature**: 0.1 (deterministic)
- **Max Tokens**: 2048
- **Seed**: Fixed per experiment

Each agent prompt includes explicit:

- Input data schema
- Output JSON schema  
- Parameter specifications (windows, constraints, thresholds)

This enables exact replication of agent behavior across runs.



