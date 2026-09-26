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
from typing import Any, Dict, List

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
# - "Listed stocks" are the per-bar snapshot, not the run's universe:
#   make_trading_decision_with_llm shows the 12 best names by trend score plus
#   holdings. The seven-name onboarding sleeve fits whole; a 30-name Dow run
#   shows 12 of 30, so rule 1 cannot spread across all of them.
# - Rule 7's 0 is how sma20, macd and macd_signal warm up, with two exceptions.
#   RSI warms up at a neutral 50, harmless while rule 4 uses it only as an
#   upper gate. And in a window shorter than 20 bars, sma20 is the whole
#   window's mean close, future bars included (features.py; dashboard
#   backtests fetch no warm-up history). A week-long onboarding window clears it.
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
