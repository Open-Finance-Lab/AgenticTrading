"""Canonical starter configuration for newly created built-in agents.

The agent Configure screen has a single mode: one plain-language trading
instruction, stored as a **one-step pipeline** whose ``presetKey`` is
``simple_instruction``. There is no separate "simple" storage format.

These constants are mirrored in ``dashboard/frontend/app.js`` (the browser needs
them to recognise a server-seeded pipeline as the editable simple kind). The two
copies are pinned together by ``tests/test_agent_starter_defaults.py`` — if they
drift, ``isSimplePipeline()`` stops matching and every default agent renders the
"saving replaces your custom pipeline" warning it should never show.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

SIMPLE_INSTRUCTION_PRESET_KEY = "simple_instruction"

# The trading-actions contract every simple-mode agent emits.
SIMPLE_INSTRUCTION_OUTPUT_FORMAT = (
    'JSON: { "orders": [{ "symbol": "...", "side": "buy|sell|hold", '
    '"qty": number, "order_type": "market|limit", "limit_price": number|null, '
    '"reason": "..." }] }'
)

SIMPLE_INSTRUCTION_LABEL = "Trading instruction"

# Seeded into every new built-in agent so a user can sign up and immediately run
# a meaningful backtest without opening Configure first.
#
# Worded for how the engine actually executes, because the model is told none of
# it: orders fill in whole shares, and a sell always closes the whole position.
# The closing "Orders:" paragraph exists because a short run tolerates almost no
# malformed replies before it aborts. Those replies are checked by
# pipeline_runner.pipeline_output_to_decision and the strict_llm block in
# portfolio_manager.py, not by infrastructure/llm/validator.py. Staying
# invested and holding is the design goal: LLM traders most often lose to
# buy-and-hold by sitting in cash and over-trading.
#
# Known gaps between this text and the engine. Close them in the engine, not
# here: the wording is what the A/B measured, so rewording it means a new run.
# - "Listed stocks" are the per-bar snapshot, not the run's universe. Since
#   #541 a pipeline agent's snapshot is the whole universe up to 30 names
#   (Mag7, the Dow, the 30-name pools); above 30 it is the 12 best names by
#   trend score plus holdings, announced by a ``universe_note``, so rule 1
#   cannot spread across all of a larger pool.
# - Rule 7's 0 is how macd and macd_signal warm up: both read 0 for a run's
#   first 33 bars (features.py). RSI warms up at a neutral 50, harmless while
#   rule 4 uses it only as an upper gate, and sma20 at the mean of the closes
#   so far, so neither shows 0. Dashboard backtests fetch no warm-up history
#   (#540), so rules 3 and 4 run without MACD for 33 of the onboarding
#   window's 49 bars.
# - Rules 5 and 6 are advice: nothing in the engine caps a position's share of
#   the account or, A-share T+1 aside, stops a same-day round trip.
#
# Three copies must match exactly: this one, app.js's mirror and the seven LLM
# cards in config/marketplace.json (each pinned by a test). Seeding is
# write-once, so a change reaches new agents and new clones only.
DEFAULT_STARTER_INSTRUCTION = (
    "Manage this account like a disciplined portfolio manager. The goal is to "
    "keep pace with, and ideally beat, simply buying equal amounts of every "
    "listed stock and holding them.\n\n"
    "1. Stay invested. At the start (all cash), buy roughly equal dollar "
    "amounts of as many listed stocks as the cash allows, keeping about 3% in "
    "cash. Skip a stock if one share costs more than a third of the account.\n"
    "2. Holding is the default. Most hours the right move is to change "
    "nothing. Never trade on small moves.\n"
    "3. Sell a stock only when its trend has clearly broken: price at least "
    "2% below its 20-hour average (sma20) AND momentum (macd) below its "
    "signal line (macd_signal). A sell always closes the whole position.\n"
    "4. Reinvest cash quickly. When cash is above 10% of the account, buy the "
    "stock you own the least of among those with price above sma20, macd "
    "above macd_signal and RSI below 75. If none qualifies, buy the stock you "
    "own the least of anyway.\n"
    "5. Keep any one stock under 35% of the account, and do not add to a "
    "stock that is already above 25%.\n"
    "6. Do not buy back a stock you sold in the last day, or sell one you "
    "bought in the last day (check recent_trades).\n"
    "7. An indicator showing 0 does not have enough history yet: ignore it.\n\n"
    "Orders: list each stock at most once, use whole-share quantities, and "
    "keep the total cost of all buys within available cash. If you make no "
    'trades, return one "hold" order for any listed stock. Keep each reason '
    "under 15 words."
)

def starter_agent_description(name: str) -> str:
    """Card copy for a pre-created prompted-model starter."""
    return (
        f"A {name} starter — open it to edit the trading instruction "
        "and run a backtest."
    )


# Pre-created Prompted Models cards for a brand-new account. Mirrored in
# ``dashboard/frontend/app.js`` (guest fallback POST). Signup provisions
# server-side so a stale browser localStorage guard cannot skip the set.
STARTER_AGENTS: tuple[Dict[str, str], ...] = (
    {
        "name": "DeepSeek V4 Pro",
        "model_name": "deepseek/deepseek-v4-pro",
        "description": starter_agent_description("DeepSeek V4 Pro"),
    },
    {
        "name": "GPT-5.5",
        "model_name": "openai/gpt-5.5",
        "description": starter_agent_description("GPT-5.5"),
    },
    {
        "name": "Claude Sonnet 4.6",
        "model_name": "anthropic/claude-sonnet-4-6",
        "description": starter_agent_description("Claude Sonnet 4.6"),
    },
)

# First-card aliases: tests and the original single-starter call sites.
STARTER_AGENT_NAME = STARTER_AGENTS[0]["name"]
STARTER_AGENT_MODEL = STARTER_AGENTS[0]["model_name"]
STARTER_AGENT_DESCRIPTION = STARTER_AGENTS[0]["description"]


def default_starter_pipeline() -> List[Dict[str, Any]]:
    """The one-step pipeline a new built-in agent starts with."""
    return [
        {
            "id": f"sub_starter_{uuid.uuid4().hex[:8]}",
            "presetKey": SIMPLE_INSTRUCTION_PRESET_KEY,
            "label": SIMPLE_INSTRUCTION_LABEL,
            "prompt": DEFAULT_STARTER_INSTRUCTION,
            "outputFormat": SIMPLE_INSTRUCTION_OUTPUT_FORMAT,
        }
    ]


# pipeline_runner.POST_TRADE_PRESET_KEY. Not imported: this module is a leaf the
# frontend mirror and the seeding path read, and pipeline_runner pulls in the
# LLM execution layer. tests/test_agent_starter_defaults.py pins the two equal.
_POST_TRADE_PRESET_KEY = "post_trade_analysis"


def _decision_steps(pipeline: Any) -> List[Dict[str, Any]]:
    if not isinstance(pipeline, list):
        return []
    return [
        step
        for step in pipeline
        if isinstance(step, dict) and step.get("presetKey") != _POST_TRADE_PRESET_KEY
    ]


def has_trading_instruction(pipeline: Any) -> bool:
    """Whether any decision step tells the model what to do.

    A pipeline whose decision steps all have blank prompts is accepted by the
    agent PATCH and the backtest body alike, and would otherwise send the model
    a step with no task at all.
    """
    return any(
        str(step.get("prompt") or "").strip() for step in _decision_steps(pipeline)
    )


def effective_pipeline(
    pipeline: Any,
    strategy_prompt: Optional[str] = None,
) -> Any:
    """The pipeline an LLM run executes once "empty means default" is applied.

    An empty trading instruction means the platform default: the starter
    instruction every new agent is seeded with, i.e. the text Configure shows
    under "See the default instruction". "Empty" is no pipeline, or one whose
    decision steps carry no prompt; its post-trade steps are kept. A
    ``strategy_prompt`` is an instruction in its own right, and an explicit
    ``[]`` or a non-list is returned untouched so the caller's validator can
    refuse it as malformed.

    The substitute is the exact pipeline a starter agent runs, so it costs what
    a starter agent costs -- on a universe above 12 names that is the full
    snapshot and the recovery output ceiling from the first call
    (pipeline_runner.DEFAULT_CEILING_SNAPSHOT_SYMBOLS), not the single-prompt
    path's shortlist. That parity is the point: the disclosure promises this
    strategy, so the run has to be the one a starter agent would produce.

    Owned here rather than in a route so every consumer of an agent's pipeline
    resolves "empty" the same way. ``robinhood_live_service`` does not call it
    yet: its empty-instruction fallback trades real money and is a separate
    decision.
    """
    if pipeline is None:
        if (strategy_prompt or "").strip():
            return None
        return default_starter_pipeline()
    if not isinstance(pipeline, list) or not pipeline:
        return pipeline
    if has_trading_instruction(pipeline):
        return pipeline
    post_trade = [
        step
        for step in pipeline
        if isinstance(step, dict) and step.get("presetKey") == _POST_TRADE_PRESET_KEY
    ]
    return default_starter_pipeline() + post_trade


def is_default_instruction_pipeline(pipeline: Any) -> bool:
    """Whether a recorded pipeline's decision side is exactly the default.

    True for an empty-instruction run and for a starter agent whose instruction
    was never edited: both ran DEFAULT_STARTER_INSTRUCTION. Judged on content,
    never on step ids, which seeding mints at random.
    """
    steps = _decision_steps(pipeline)
    return (
        len(steps) == 1
        and str(steps[0].get("prompt") or "").strip()
        == DEFAULT_STARTER_INSTRUCTION.strip()
    )
