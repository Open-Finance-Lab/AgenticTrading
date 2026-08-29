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
DEFAULT_STARTER_INSTRUCTION = (
    "Spread the money across a few of the strongest available stocks. Buy on "
    "meaningful dips, take profits after strong run-ups, and never put "
    "everything into one stock."
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
