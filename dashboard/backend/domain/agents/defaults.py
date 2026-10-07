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
from typing import Any, Dict, List, Optional, Tuple

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


# The post-trade split is owned here and imported by pipeline_runner (which
# re-exports it), so "is this a decision step?" has one answer for the runner,
# the route and effective_pipeline below. This module stays a leaf: it imports
# nothing from the backend.
POST_TRADE_PRESET_KEY = "post_trade_analysis"


def is_post_trade_step(step: Any) -> bool:
    return isinstance(step, dict) and step.get("presetKey") == POST_TRADE_PRESET_KEY


def split_pipeline(
    pipeline: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a mixed pipeline into hourly decision steps and post-trade steps."""
    decision_steps: List[Dict[str, Any]] = []
    post_trade_steps: List[Dict[str, Any]] = []
    if not pipeline or not isinstance(pipeline, (list, tuple)):
        return decision_steps, post_trade_steps
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        if is_post_trade_step(step):
            post_trade_steps.append(step)
        else:
            decision_steps.append(step)
    return decision_steps, post_trade_steps


def _has_prompt(step: Dict[str, Any]) -> bool:
    return bool(str(step.get("prompt") or "").strip())


def has_trading_instruction(pipeline: Any) -> bool:
    """Whether any decision step tells the model what to do.

    A pipeline whose decision steps all have blank prompts is accepted by the
    agent PATCH and the backtest body alike, and would otherwise send the model
    a step with no task at all.
    """
    decision_steps, _ = split_pipeline(pipeline)
    return any(_has_prompt(step) for step in decision_steps)


def blank_decision_step_numbers(pipeline: Any) -> List[int]:
    """1-based positions of blank decision steps in a pipeline that has an
    instruction elsewhere.

    effective_pipeline replaces an all-blank decision side; a partly blank one
    cannot be repaired the same way (which step's task would the default
    take?), and run as-is it bills a call that carries no task -- for the last
    step, the one whose orders execute. The route refuses it instead.
    """
    if not has_trading_instruction(pipeline):
        return []
    return [
        index + 1
        for index, step in enumerate(pipeline)
        if isinstance(step, dict) and not is_post_trade_step(step) and not _has_prompt(step)
    ]


def _instruction_step(prompt: str) -> Dict[str, Any]:
    step = default_starter_pipeline()[0]
    step["prompt"] = prompt
    return step


def effective_pipeline(
    pipeline: Any,
    strategy_prompt: Optional[str] = None,
) -> Any:
    """The pipeline an LLM run executes once "empty means default" is applied.

    An empty trading instruction means the platform default: the starter
    instruction every new agent is seeded with, i.e. the text Configure shows
    under "See the default instruction". "Empty" is no pipeline, or one whose
    decision steps all carry no prompt; its post-trade steps are kept. An
    explicit ``[]`` or a non-list is returned untouched so the caller's
    validator can refuse it as malformed.

    A ``strategy_prompt`` is an instruction in its own right and fills an
    empty decision side in place of the default, on both empty shapes. With no
    post-trade steps to keep, that is ``None`` -- the single-prompt path the
    worker takes for a strategy_prompt and no pipeline. With post-trade steps,
    it becomes the instruction step, so those steps still run.

    The default substitute is the exact pipeline a starter agent runs, so it
    costs what a starter agent costs -- on a universe above 12 names that is
    the full snapshot and the recovery output ceiling from the first call
    (pipeline_runner.DEFAULT_CEILING_SNAPSHOT_SYMBOLS), not the single-prompt
    path's shortlist. That parity is the point: the disclosure promises this
    strategy, so the run has to be the one a starter agent would produce. The
    Run Backtest modal says so when it previews the default.

    The only caller today is the dashboard ``/backtest/run`` route. The one
    other reader of an agent's pipeline, ``robinhood_live_service``, does not
    call it: its empty-instruction fallback trades real money and is a separate
    decision. The protocol and /api/v2 surfaces run external agents, which have
    no stored pipeline, and the leaderboard runs its curated entries with no
    instruction on purpose (see the route's comment).
    """
    if pipeline is not None and (not isinstance(pipeline, list) or not pipeline):
        return pipeline
    if has_trading_instruction(pipeline):
        return pipeline
    _, post_trade = split_pipeline(pipeline)
    instruction = (strategy_prompt or "").strip()
    if instruction:
        if not post_trade:
            return None
        return [_instruction_step(instruction)] + post_trade
    return default_starter_pipeline() + post_trade


def is_default_instruction_pipeline(pipeline: Any) -> bool:
    """Whether a recorded pipeline's decision side is exactly the default.

    True for an empty-instruction run and for a starter agent whose instruction
    was never edited: both ran DEFAULT_STARTER_INSTRUCTION. Judged on what
    reaches the model -- the step's label, prompt and output format, the three
    fields pipeline_runner._build_step_prompt sends -- and never on step ids,
    which seeding mints at random. A step that reuses the default wording under
    its own output contract ran a different request and is not the default.
    """
    steps, _ = split_pipeline(pipeline)
    if len(steps) != 1:
        return False
    step = steps[0]
    return all(
        str(step.get(field) or "").strip() == expected.strip()
        for field, expected in (
            ("label", SIMPLE_INSTRUCTION_LABEL),
            ("prompt", DEFAULT_STARTER_INSTRUCTION),
            ("outputFormat", SIMPLE_INSTRUCTION_OUTPUT_FORMAT),
        )
    )
