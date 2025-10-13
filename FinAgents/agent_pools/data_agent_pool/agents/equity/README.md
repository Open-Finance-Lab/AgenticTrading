# Polygon Agent: Design and Logic

## Overview

The `PolygonAgent` is a modular, extensible data agent designed to interface with the Polygon.io financial data API. It is implemented as part of a multi-agent orchestration framework for quantitative research and financial data analysis. The agent is capable of both direct API-driven data retrieval and natural language-driven intent parsing via large language models (LLMs).

## Architecture

### 1. Configuration

The agent is initialized with a strict configuration schema (`PolygonConfig`), which enforces the presence of all required API parameters, including the API key, base URL, rate limits, and a boolean flag (`llm_enabled`) that controls whether LLM-based intent parsing is enabled.

### 2. Tool Registration

Upon initialization, the agent registers a set of callable tools, each corresponding to a core data operation:
- `fetch_market_data`: Retrieves historical OHLCV data for a given symbol and interval.
- `analyze_company`: Fetches detailed company information.
- `identify_leaders`: Selects top-performing tickers based on volume and volatility.

These tools are exposed to the LLM for dynamic plan generation and execution.

### 3. LLM Integration

If `llm_enabled` is set to `True`, the agent initializes an LLM interface with a carefully engineered system prompt. This prompt instructs the LLM to output execution plans as valid JSON objects, supporting both single-step and multi-step (sequential or parallel) workflows. The prompt restricts the LLM to only use the registered tool names, ensuring robust and interpretable plan generation.

### 4. Intent Parsing

Natural language queries are processed by the LLM, which returns a structured execution plan. The agent parses this plan, validating its structure and ensuring all required fields are present. The plan may specify a single tool invocation or a sequence of steps, each with its own parameters.

### 5. Strategy Execution

The agent executes the plan by mapping each step to the corresponding registered tool. Both synchronous and asynchronous tool functions are supported. Results from each step are aggregated and returned to the user, along with metadata describing the execution context.

### 6. Data Enrichment

For certain intervals (e.g., daily or hourly), the agent enriches the returned market data with additional metrics such as VWAP, pre-market, and after-market prices by making supplementary API calls.

## Key Features

- **Strict Configuration Validation:** Ensures all required parameters are present at initialization.
- **LLM-Driven Planning:** Supports flexible, natural language-driven workflows with robust prompt engineering.
- **Multi-Step Execution:** Handles both single and multi-step plans, with support for sequential or parallel execution modes.
- **Extensible Tooling:** New data operations can be added as tools and exposed to the LLM with minimal code changes.
- **Error Handling:** Provides clear error messages for missing configuration, invalid plans, or unsupported tool invocations.

## Example Workflow

1. **User Query:**  
   "Get daily price data for AAPL and MSFT for January 2024, and also provide company information for both."

2. **LLM Plan Generation:**  
   The LLM outputs a JSON plan with multiple steps, each specifying a tool and parameters.

3. **Plan Parsing and Validation:**  
   The agent parses the plan, checks for required fields, and validates tool names.

4. **Strategy Execution:**  
   Each step is executed in order (or in parallel, if specified), and results are aggregated.

5. **Result Delivery:**  
   The agent returns the results and execution metadata to the user.

## Extensibility

The agent is designed for easy extension. To add new data operations, implement the corresponding method, register it as a tool, and update the system prompt to expose the new tool to the LLM.

---

**This architecture enables robust, interpretable, and flexible financial data workflows, bridging API-driven and natural language-driven paradigms in quantitative research.**

## Using the yfinance MCP server locally

This directory also contains a simple Model Context Protocol (MCP) server backed by
`yfinance`. Combine it with the OpenAI Agents Python SDK to let an agent call the
server's `get_stock_metric` and `get_historical_data` tools.

1. Install dependencies (the OpenAI SDK provides the `agents` package)::

   pip install openai yfinance

2. Set your OpenAI API key in the environment::

   export OPENAI_API_KEY="sk-..."

3. Run the demo script, which spawns the MCP server over stdio and issues an
   example query through an agent::

   python use_yfinance_mcp.py --symbol AAPL --metric marketCap --period 6mo

Override the agent's request by setting `YFINANCE_AGENT_PROMPT` before launching
the script. The CLI supports `--start`/`--end` to specify explicit date ranges
and now instructs the agent to store the CSV on the MCP server side. When the
managed prompt is used, the tool writes the data to disk (default `./outputs`,
configurable via `--output-dir`) and the agent replies with JSON summarising the
completed steps, metric value, and the saved file path. The helper script parses
that JSON, confirms the tasks, and prints a short preview of the stored CSV.