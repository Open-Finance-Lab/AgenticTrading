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
import importlib.util
import subprocess
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
#
# Bound by assignment from a module alias rather than a bare `from ... import X`
# so each re-export is explicit and static analysis sees it as used -- the
# convention external_run_service.py already documents. Do not collapse back:
# `py/unused-import` is intra-file only, so it cannot see the cross-module
# contract these three exist to satisfy.
import dashboard.backend.infrastructure.llm.backtest_harness as _backtest_harness

Anthropic = _backtest_harness.Anthropic
HAS_ANTHROPIC = _backtest_harness.HAS_ANTHROPIC
LLM_MODEL_NAME = _backtest_harness.LLM_MODEL_NAME

# A presence probe, not a use: dashboard.backend.domain.backtesting.features
# imports pandas_ta itself. This script installs it first so that import cannot
# fail part-way through a run. find_spec rather than a try/except import because
# nothing here needs the module object -- only the answer to "is it installed?".
if importlib.util.find_spec("pandas_ta") is None:
    print("Installing pandas_ta...")
    # sys.executable, not a bare "pip": the script may well be running under a
    # venv whose pip is not first on PATH, and installing into the wrong
    # interpreter would leave the import below still failing.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta"])
    # The probe above populated the path-finder's directory caches with a miss;
    # drop them so the module just written to site-packages is discoverable by
    # the features import below rather than on the next process start.
    importlib.invalidate_caches()

# ---------------------------------------------------------------------------
# Phase 2A extraction: the implementations below now live under the canonical
# dashboard.backend.* packages and are re-exported here so this script's public
# compatibility surface (and the three backend callers that import this module)
# stays unchanged. pandas_ta is imported above first so the features module can
# rely on it being available.
# ---------------------------------------------------------------------------
#
# Same explicit-assignment form as the harness re-exports above, for the same
# reason: these five are referenced only through `bha.<name>` from other
# modules, which `py/unused-import` cannot see. The guard suites
# (test_backtest_compatibility, test_canonical_consumers) assert each one is the
# canonical object, so deleting any of them reddens those tests.
import dashboard.backend.domain.backtesting.features as _features
import dashboard.backend.domain.backtesting.metrics as _metrics
import dashboard.backend.infrastructure.llm.decision_parsing as _decision_parsing
import dashboard.backend.infrastructure.market_data.alpaca_bars as _alpaca_bars

TechnicalIndicators = _features.TechnicalIndicators
calculate_sharpe = _metrics.calculate_sharpe
calculate_max_drawdown = _metrics.calculate_max_drawdown
fix_json_formatting = _decision_parsing.fix_json_formatting
AlpacaDataLoader = _alpaca_bars.AlpacaDataLoader

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
# subclasses (e.g. backtest_custom_algo) keep working unchanged. Explicit
# assignment, as above.
import dashboard.backend.domain.backtesting.portfolio_manager as _portfolio_manager

PortfolioManager = _portfolio_manager.PortfolioManager


# ============================================================================
# Backtester
# ============================================================================

# `HourlyBacktester` now lives in
# dashboard.backend.domain.backtesting.engine and is re-exported here so the
# legacy public path (bha.HourlyBacktester), main() below, and existing
# subclasses (e.g. backtest_custom_algo) keep working unchanged.
from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.infrastructure.llm.execution.handoff import (
    ExecutionHandoffError,
    consume_execution_handoff,
)
from dashboard.backend.infrastructure.llm.execution.client import (
    AnthropicCompatibleExecutionClient,
)
from dashboard.backend.infrastructure.llm.execution.errors import LLMExecutionError
from dashboard.backend.infrastructure.llm.execution.service import LLMExecutionService
from dashboard.backend.domain.credits.service import credits_service
from dashboard.backend.domain.model_providers.service import get_model_provider_service


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
    parser.add_argument(
        "--execution-handoff-stdin",
        action="store_true",
        help="Read one signed, secret-free execution handoff from stdin",
    )
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
    execution_handoff = None
    if args.execution_handoff_stdin:
        try:
            execution_handoff = consume_execution_handoff(sys.stdin.read())
        except ExecutionHandoffError:
            parser.error("invalid execution handoff")
        if args.run_id != execution_handoff.run_id:
            parser.error("execution handoff run id does not match --run-id")
        if args.model != execution_handoff.model_id:
            parser.error("execution handoff model does not match --model")
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

    if execution_handoff is not None:
        if decision_source != LLM_DECISION_SOURCE:
            parser.error("execution handoff requires decision_source='llm'")
        print(
            "Execution handoff verified: "
            f"provider={execution_handoff.provider_id}, "
            f"model={execution_handoff.model_id}, "
            f"billing_mode={execution_handoff.billing_mode.value}"
        )
    elif (
        decision_source == LLM_DECISION_SOURCE
        and args.runtime_type == PIPELINE_RUNTIME_TYPE
    ):
        parser.error("explicit LLM execution requires a signed execution handoff")

    execution_service = None
    execution_client = None
    if execution_handoff is not None:
        execution_service = LLMExecutionService(
            providers=get_model_provider_service(),
            credits=credits_service,
        )
        execution_client = AnthropicCompatibleExecutionClient(
            execution_service=execution_service,
            handoff=execution_handoff,
        )
    
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
        execution_client=execution_client,
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
    
    try:
        agent_id, agent_eq = backtester.run_agent_backtest()
    finally:
        if execution_service is not None and execution_handoff is not None:
            try:
                execution_service.finalize_run(
                    execution_handoff.run_id,
                    billing_mode=execution_handoff.billing_mode,
                )
            except LLMExecutionError as exc:
                print(f"❌ LLM execution finalization failed: {exc.safe_message}")
                raise
    
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


def _close_pools() -> None:
    """Close shared Postgres pools before this short-lived CLI exits."""
    # Avoid importing psycopg on paths that never selected a Postgres store.
    # The registry is already loaded when a pooled store was used.
    pool_module = sys.modules.get("dashboard.backend.db_pool")
    if pool_module is not None:
        pool_module.close_all_pools()


if __name__ == "__main__":
    # Reached through the module alias bound at the top rather than a second
    # `from` import of the same module: importing one module both ways in one
    # file is its own CodeQL finding (py/import-and-import-from), and the alias
    # already exists.
    try:
        try:
            main()
        except _alpaca_bars.MarketDataUnavailableError as exc:
            # Library code raises (so server threads can catch it); the CLI
            # boundary is where the process exit belongs.
            print(f"❌ {exc}")
            sys.exit(1)
    finally:
        _close_pools()
