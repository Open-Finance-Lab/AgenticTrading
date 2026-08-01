#!/usr/bin/env python3
"""
Profile-driven Hourly Backtest with Agent Decision Making

The agent manages a portfolio across the selected market profile.
Each hour, the agent analyzes market data and technical indicators and decides:
- What positions to buy
- What positions to sell  
- What positions to hold

This generates a realistic equity curve based on agent decision-making.

Uses REAL Alpaca hourly data with forward-filled price cache for missing bars.

Usage:
    python3 backtest_hourly_agent.py --start 2026-03-01 --end 2026-04-23
"""

import sys
import json
import argparse
from pathlib import Path

# Bootstrap for non-package execution contexts: when this module is run directly
# as a file (``python dashboard/scripts/backtest_hourly_agent.py``) or imported
# flat by the backend (``import backtest_hourly_agent`` after the backend adds
# SCRIPTS_DIR to the import path), the repository root is not necessarily
# importable, so the canonical ``dashboard.backend.*`` imports below would fail.
# In those cases
# ``__package__`` is empty and we use the shared script bootstrap helper. When
# this module is imported as ``dashboard.scripts.backtest_hourly_agent`` the repo
# root is already importable and no bootstrap (or extra sys.path entry) is needed.
if not __package__:
    from _bootstrap import ensure_repo_root

    ensure_repo_root()

from dashboard.backend.paths import CREDENTIALS_DIR
from dashboard.backend.database import db
from dashboard.backend.infrastructure.llm.validator import DJIA_30

# Optional: LLM integration. Phase 2C2 moved the Anthropic SDK import, the
# default model name, and the LLM request/parse workflow into the canonical
# harness at dashboard.backend.infrastructure.llm.backtest_harness. These symbols
# are re-exported here so existing consumers (engines/strategies/llm_agent.py,
# backtest_custom_algo.py, and bha.* callers) keep working unchanged.
from dashboard.backend.infrastructure.llm.backtest_harness import (
    Anthropic,
    HAS_ANTHROPIC,
    LLM_MODEL_NAME,
)

try:
    import pandas_ta as ta
except ImportError:
    print("Installing pandas_ta...")
    import subprocess
    subprocess.check_call(["pip", "install", "pandas_ta"])
    import pandas_ta as ta

# ---------------------------------------------------------------------------
# Phase 2A extraction: the implementations below now live under the canonical
# dashboard.backend.* packages and are re-exported here so this script's public
# compatibility surface (and the three backend callers that import this module)
# stays unchanged. pandas_ta is imported above first so the features module can
# rely on it being available.
# ---------------------------------------------------------------------------
from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.domain.backtesting.metrics import (
    calculate_sharpe,
    calculate_max_drawdown,
)
from dashboard.backend.infrastructure.llm.decision_parsing import fix_json_formatting
from dashboard.backend.infrastructure.market_data.alpaca_bars import AlpacaDataLoader
from dashboard.backend.infrastructure.market_data.provider import (
    ALPACA,
    IFIND_ASHARE,
    VNPY_SIMULATION,
)
from dashboard.backend.infrastructure.market_data.profiles import (
    LLM_DECISION_SOURCE,
    RULE_BASED_DECISION_SOURCE,
    get_market_profile,
    resolve_decision_source,
)
from dashboard.backend.domain.backtesting.constants import INITIAL_CAPITAL
from dashboard.backend.domain.agents.runtime import (
    AI_HEDGE_FUND_RUNTIME_TYPE,
    PIPELINE_RUNTIME_TYPE,
)

# DJIA_30 is imported from validator (the single source of truth, guarded by
# tests/test_djia30_universe.py) rather than redefined here.

# ============================================================================
# JSON Parsing Utilities
# ============================================================================
# `fix_json_formatting` now lives in
# dashboard.backend.infrastructure.llm.decision_parsing and is re-exported above.


# ============================================================================
# LLM Model Configuration
# ============================================================================
# `LLM_MODEL_NAME` now lives in
# dashboard.backend.infrastructure.llm.backtest_harness and is re-exported above.

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_START = "2026-03-01"
DEFAULT_END = "2026-04-13"
# `INITIAL_CAPITAL` now lives in
# dashboard.backend.domain.backtesting.constants and is re-exported above.
TIMEFRAME = "1h"  # Hourly


# ============================================================================
# Data Loader - Alpaca API
# ============================================================================
# `AlpacaDataLoader` now lives in
# dashboard.backend.infrastructure.market_data.alpaca_bars and is re-exported above.


# ============================================================================
# Technical Indicators
# ============================================================================
# `TechnicalIndicators` now lives in
# dashboard.backend.domain.backtesting.features and is re-exported above.


# ============================================================================
# Portfolio Manager with Agent Decision Logic
# ============================================================================

# `PortfolioManager` now lives in
# dashboard.backend.domain.backtesting.portfolio_manager and is re-exported
# here so the legacy public path (bha.PortfolioManager) and existing
# subclasses (e.g. backtest_custom_algo) keep working unchanged.
from dashboard.backend.domain.backtesting.portfolio_manager import (
    PortfolioManager,
)


# ============================================================================
# Backtester
# ============================================================================

# `HourlyBacktester` now lives in
# dashboard.backend.domain.backtesting.engine and is re-exported here so the
# legacy public path (bha.HourlyBacktester), main() below, and existing
# subclasses (e.g. backtest_custom_algo) keep working unchanged.
from dashboard.backend.domain.backtesting.engine import HourlyBacktester


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hourly backtest with Agent vs Baselines"
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END, help="End date (YYYY-MM-DD)")
    parser.add_argument("--session-id", default="legacy-demo-session", help="Session ID for isolation")
    parser.add_argument("--clear", action="store_true", help="Clear all data first")
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--use-llm", dest="use_llm", action="store_true", help="Use the profile's default decision source (legacy compatibility)")
    llm_group.add_argument("--no-llm", dest="use_llm", action="store_false", help="Disable LLM, use rule-based logic")
    parser.set_defaults(use_llm=None)
    parser.add_argument(
        "--decision-source",
        choices=[RULE_BASED_DECISION_SOURCE, LLM_DECISION_SOURCE],
        default=None,
        help="Explicit decision source; takes precedence over the profile default",
    )
    parser.add_argument("--mode", default="safe_trading", choices=["safe_trading", "buy_and_hold"], help="Agent mode: 'safe_trading' (risk management) or 'buy_and_hold' (debug)")
    parser.add_argument("--strategy-prompt-file", default=None, help="Path to a UTF-8 file with a free-form strategy prompt that REPLACES the built-in agent prompt for this run")
    parser.add_argument("--pipeline-file", default=None, help="Path to a UTF-8 JSON file with the sub-agent pipeline steps for this run")
    parser.add_argument(
        "--runtime-type",
        default=PIPELINE_RUNTIME_TYPE,
        choices=[PIPELINE_RUNTIME_TYPE, AI_HEDGE_FUND_RUNTIME_TYPE],
        help="Hosted agent runtime (default: pipeline)",
    )
    parser.add_argument(
        "--runtime-config-file",
        default=None,
        help="Path to the hosted runtime's non-secret JSON configuration",
    )
    parser.add_argument("--model", default=None, help="Override the LLM model id (e.g. anthropic/claude-haiku-4-5). Defaults to the gateway-appropriate slug.")
    parser.add_argument("--run-id", default=None, help="Preset run id (used for live progress + DB row)")
    parser.add_argument("--progress-file", default=None, help="Path to write incremental equity snapshots for live dashboard charting")
    parser.add_argument(
        "--data-source",
        default=ALPACA,
        choices=[ALPACA, VNPY_SIMULATION, IFIND_ASHARE],
        help="Market-data provider (default: alpaca)",
    )
    parser.add_argument(
        "--universe",
        default=None,
        help="Fixed universe key bound to the selected market-data provider",
    )
    parser.add_argument(
        "--timeframe",
        default=None,
        help="Fixed bar timeframe bound to the selected market-data provider",
    )
    parser.add_argument("--initial-capital", type=float, default=None, help="Starting capital for this backtest (defaults to INITIAL_CAPITAL)")
    parser.add_argument(
        "--assets",
        default=None,
        help="Comma-separated tickers for the tradeable universe (default: full DJIA_30)",
    )
    
    args = parser.parse_args()
    try:
        market_profile = get_market_profile(args.data_source, args.universe)
    except ValueError as exc:
        parser.error(str(exc))
        return
    if args.timeframe is not None and args.timeframe != market_profile.timeframe:
        parser.error(
            f"--data-source {args.data_source} requires "
            f"--timeframe {market_profile.timeframe}"
        )
    if args.decision_source is not None and args.use_llm is not None:
        flag_source = (
            LLM_DECISION_SOURCE if args.use_llm else RULE_BASED_DECISION_SOURCE
        )
        if flag_source != args.decision_source:
            parser.error(
                f"--decision-source {args.decision_source} conflicts with "
                f"{'--use-llm' if args.use_llm else '--no-llm'}"
            )
    legacy_use_llm = True if args.use_llm is None else args.use_llm
    requested_decision_source = (
        args.decision_source
        if args.decision_source is not None
        else (None if legacy_use_llm else RULE_BASED_DECISION_SOURCE)
    )
    try:
        decision_source = resolve_decision_source(
            market_profile,
            requested_decision_source,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return
    
    session_id = args.session_id

    # Optional free-form strategy prompt (read from a file to avoid shell escaping).
    strategy_prompt = None
    if args.strategy_prompt_file:
        try:
            strategy_prompt = Path(args.strategy_prompt_file).read_text(encoding="utf-8").strip() or None
        except OSError as exc:
            print(f"⚠️  Could not read --strategy-prompt-file ({args.strategy_prompt_file}): {exc}")
            strategy_prompt = None

    pipeline = None
    if args.pipeline_file:
        try:
            raw_pipeline = json.loads(Path(args.pipeline_file).read_text(encoding="utf-8"))
            if isinstance(raw_pipeline, list) and raw_pipeline:
                pipeline = raw_pipeline
            else:
                print(f"⚠️  --pipeline-file must contain a non-empty JSON array; ignoring")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"⚠️  Could not read --pipeline-file ({args.pipeline_file}): {exc}")
            pipeline = None

    runtime_config = {}
    if args.runtime_config_file:
        try:
            raw_runtime_config = json.loads(
                Path(args.runtime_config_file).read_text(encoding="utf-8")
            )
            if isinstance(raw_runtime_config, dict):
                runtime_config = raw_runtime_config
            else:
                parser.error("--runtime-config-file must contain a JSON object")
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"Could not read --runtime-config-file: {exc}")
    
    # Validate and swap dates if backwards
    from datetime import datetime as dt_parser
    try:
        start = dt_parser.strptime(args.start, "%Y-%m-%d")
        end = dt_parser.strptime(args.end, "%Y-%m-%d")
        
        if start > end:
            print(f"⚠️  Dates were backwards ({args.start} > {args.end}). Swapping...\n")
            args.start, args.end = args.end, args.start
    except ValueError:
        pass  # Invalid format, let it error naturally
    
    if args.clear:
        print("🗑️ Clearing all existing data...\n")
        db.clear_all()
    
    symbols = None
    if args.assets:
        symbols = [s.strip().upper() for s in args.assets.split(",") if s.strip()]
        if not symbols:
            symbols = None

    print(f"\n🚀 Hourly Agent Backtest Framework")
    print(f"{'='*70}")
    print(f"Period: {args.start} → {args.end}")
    print(f"Session: {session_id[:8]}...")
    effective_symbols = (
        list(market_profile.symbols)
        if args.data_source == IFIND_ASHARE
        else (symbols or list(DJIA_30))
    )
    universe_label = f"{len(effective_symbols)} ({market_profile.universe})"
    print(f"Stocks: {universe_label}")
    print(f"Universe: {', '.join(effective_symbols)}")
    print(f"Data source: {args.data_source}")
    print(f"Decision source: {decision_source}")
    print(f"Trading: Hourly (Agent decisions based on indicators)")
    capital = float(args.initial_capital) if args.initial_capital is not None else float(INITIAL_CAPITAL)
    print(f"Capital: ${capital:,.0f}")
    
    # Show mode
    mode_display = (
        "AI Hedge Fund"
        if args.runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE
        else (
            "Sub-agent Pipeline"
            if pipeline
            else (
                "Custom Prompt"
                if strategy_prompt
                else args.mode.replace("_", " ").title()
            )
        )
    )
    print(f"Mode: {mode_display}")
    if pipeline:
        print(f"Sub-agent pipeline: {len(pipeline)} step(s)")
    if strategy_prompt and not pipeline:
        print(f"Custom strategy prompt: {len(strategy_prompt)} chars")
    print(f"{'='*70}\n")
    
    # Initialize backtester (with LLM if available and enabled)
    # Note: dates are validated in __init__ if they somehow got reversed again
    backtester = HourlyBacktester(
        args.start,
        args.end,
        session_id,
        use_llm=decision_source == LLM_DECISION_SOURCE,
        mode=args.mode,
        strategy_prompt=strategy_prompt,
        model=args.model,
        pipeline=pipeline,
        live_run_id=args.run_id,
        progress_file=args.progress_file,
        data_source=args.data_source,
        initial_capital=capital,
        symbols=symbols,
        universe=market_profile.universe,
        decision_source=decision_source,
        runtime_type=args.runtime_type,
        runtime_config=runtime_config,
    )
    
    if args.runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE:
        print("🧠 Using hosted AI Hedge Fund runtime for trading decisions\n")
    elif backtester.use_llm:
        print(f"🧠 Using {LLM_MODEL_NAME} for trading decisions (Mode: {mode_display})\n")
    else:
        print("⚙️  Using rule-based logic for trading decisions\n")
    
    # Step 1: Load data
    print(f"1️⃣ Loading historical hourly data from {args.data_source}...")
    backtester.load_data()
    
    # Step 2: Calculate indicators
    print("\n2️⃣ Calculating technical indicators...")
    backtester.calculate_indicators()
    
    # DEBUG: Show loaded symbols
    print(f"\n📊 DEBUG - Loaded Symbols:")
    print(f"   Total symbols loaded: {len(backtester.all_data)}")
    print(f"   Symbols: {', '.join(sorted(backtester.all_data.keys())[:10])}{'...' if len(backtester.all_data) > 10 else ''}")
    print(f"   Agent universe: {', '.join(backtester.symbols)}")
    print(f"   Baselines will use: {market_profile.benchmark}")
    print(f"   Loaded bars for: {len(backtester.all_data)} symbols")
    
    # Step 3: Run backtests
    print("\n3️⃣ Running backtests...\n")
    
    agent_id, agent_eq = backtester.run_agent_backtest()
    
    # DEBUG: Show what agent bought
    print(f"\n📋 DEBUG - Agent Holdings Summary:")
    if agent_eq:
        agent_final = agent_eq[-1]
        print(f"   Final equity: ${agent_final['equity']:,.0f}")
    
    bh_id, bh_eq = backtester.run_buyhold_baseline()
    
    # DEBUG: Show what baseline bought
    print(f"\n📋 DEBUG - Baseline Holdings Summary:")
    if bh_eq:
        bh_final = bh_eq[-1]
        print(f"   Final equity: ${bh_final['equity']:,.0f}")
    
    if market_profile.index_baseline_enabled:
        djia_id, djia_eq = backtester.run_djia_baseline()
    else:
        djia_id, djia_eq = None, []

    db.update_run_baselines(
        agent_id,
        djia_run_id=djia_id,
        buyhold_run_id=bh_id,
    )
    
    # Summary
    print(f"{'='*70}")
    print(f"✅ All backtests complete!")
    print(f"{'='*70}")
    print(f"\nRun IDs:")
    print(f"  • Agent: {agent_id}")
    print(f"  • Buy & Hold: {bh_id}")
    if djia_id:
        print(f"  • DJIA Index: {djia_id}")
    print(f"\n📊 Dashboard: python3 dashboard/backend/app.py → http://localhost:8000")


if __name__ == "__main__":
    from dashboard.backend.infrastructure.market_data.alpaca_bars import (
        MarketDataUnavailableError,
    )

    try:
        main()
    except MarketDataUnavailableError as exc:
        # Library code raises (so server threads can catch it); the CLI
        # boundary is where the process exit belongs.
        print(f"❌ {exc}")
        sys.exit(1)
