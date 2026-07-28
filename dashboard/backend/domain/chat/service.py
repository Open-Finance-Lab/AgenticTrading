from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

from anthropic import APIError, AsyncAnthropic
from dotenv import load_dotenv

from dashboard.backend.infrastructure.llm.backtest_harness import (
    COMMONSTACK_MODEL_NAME,
    LLM_MODEL_NAME,
)


load_dotenv()


# CommonStack is the "model we host": one key reaches frontier models behind an
# Anthropic-compatible endpoint. When COMMONSTACK_API_KEY is set the chat client
# routes through it (and must use the gateway slug); otherwise it falls back to
# native Anthropic.
COMMONSTACK_BASE_URL = os.getenv("COMMONSTACK_BASE_URL", "https://api.commonstack.ai")


def require_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value


def resolve_chat_model() -> str:
    """Model id matching the client ``get_claude_client`` builds.

    CommonStack expects ``provider/model`` slugs. Prefer ``CHAT_MODEL`` when set;
    otherwise use ``COMMONSTACK_MODEL_NAME`` (DeepSeek by default — Anthropic
    slugs on CommonStack have been observed returning a canned greeting with
    ~10 ``input_tokens`` while ignoring the request body).
    """
    if os.getenv("COMMONSTACK_API_KEY"):
        return os.getenv("CHAT_MODEL") or COMMONSTACK_MODEL_NAME
    return os.getenv("ANTHROPIC_MODEL", LLM_MODEL_NAME)


# Lazily-constructed Anthropic-compatible client.
#
# Importing this module must not require credentials or build a network client;
# the client is created on first use via ``get_claude_client`` so that import
# stays side-effect free and test/runtime configuration is resolved on demand.
_claude_client: AsyncAnthropic | None = None


def get_claude_client() -> AsyncAnthropic:
    """Return the shared chat client, constructing it on first use.

    Prefers CommonStack (the hosted gateway) when ``COMMONSTACK_API_KEY`` is set;
    otherwise uses native Anthropic via ``ANTHROPIC_API_KEY``.
    """
    global _claude_client

    if _claude_client is None:
        commonstack_key = os.getenv("COMMONSTACK_API_KEY")
        if commonstack_key:
            _claude_client = AsyncAnthropic(
                api_key=commonstack_key,
                base_url=COMMONSTACK_BASE_URL,
            )
        else:
            _claude_client = AsyncAnthropic(api_key=require_env("ANTHROPIC_API_KEY"))

    return _claude_client


# CommonStack Anthropic-provider stub (2026-07): ignores body, returns this
# greeting with ~10 input_tokens. Refuse and fall back to a working provider.
_STUB_ASSISTANT_GREETING = "Hi! How can I help you today?"
_COMMONSTACK_CHAT_FALLBACK_MODELS = (
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
)


def _is_stub_assistant_reply(
    text: str,
    *,
    input_tokens: int | None,
    prompt_chars: int,
) -> bool:
    if (text or "").strip() == _STUB_ASSISTANT_GREETING:
        return True
    # Body ignored by gateway: long prompt but almost no input tokens billed.
    if input_tokens is not None and prompt_chars > 80 and input_tokens < 20:
        return True
    return False


def _chat_model_candidates(preferred: str | None) -> list[str]:
    """Ordered models to try for chat/strategy synthesis."""
    primary = (preferred or "").strip() or resolve_chat_model()
    candidates = [primary]
    if os.getenv("COMMONSTACK_API_KEY"):
        for model in _COMMONSTACK_CHAT_FALLBACK_MODELS:
            if model not in candidates:
                candidates.append(model)
    return candidates


def _stub_reply_error(model: str, *, action: str) -> RuntimeError:
    """Error raised when every candidate model produced a stub reply.

    Only mentions CommonStack/CHAT_MODEL when a CommonStack key is actually in
    play; the native-Anthropic path has neither, so that advice would confuse.
    """
    if os.getenv("COMMONSTACK_API_KEY"):
        return RuntimeError(
            f"Hosted model {model!r} ignored the {action} request (returned a "
            f"canned greeting). Set CHAT_MODEL to a working CommonStack slug "
            f"(e.g. deepseek/deepseek-v4-pro) or pick an agent whose model is "
            f"not on the broken Anthropic route."
        )
    return RuntimeError(
        f"Model {model!r} returned a canned greeting instead of a real "
        f"{action} reply. Check ANTHROPIC_MODEL or the selected agent's model."
    )


# Temporary MVP memory.
#
# Key:
#   (platform_user_id, agent_id)
#
# Value:
#   Claude-compatible conversation messages
#
# This will eventually be replaced with persistent database storage.
conversation_history: dict[
    tuple[str, str],
    list[dict[str, Any]],
] = defaultdict(list)


SYSTEM_PROMPT = """
You are the conversational assistant for Agentic Trading Lab.

Agentic Trading Lab helps users experiment with LLM-based trading agents,
including backtesting, paper trading, strategy configuration, performance
evaluation, and decision analysis.

This Discord integration is currently an early chat prototype.

Do not claim that you:
- executed a trade,
- changed a saved strategy,
- accessed a portfolio,
- ran a backtest,
- retrieved live market data,

unless the application provides an actual tool result confirming that action.

Provide educational and research-oriented assistance. Clearly distinguish
general information from personalized financial advice.
""".strip()


def extract_text(response: Any) -> str:
    parts: list[str] = []

    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)

    return "\n".join(parts).strip()


async def chat_with_agent(
    *,
    user_id: str,
    agent_id: str,
    message: str,
    model: str | None = None,
) -> str:
    """
    Send a message to an Agentic Trading Lab agent.

    This function is the main integration boundary. The Discord bot should
    not call Anthropic directly.

    Tries ``model`` (or ``resolve_chat_model()``) first, then CommonStack
    fallbacks if the primary provider returns a known stub greeting or a
    request error.

    Future implementation:
    - authenticate the platform user,
    - verify agent ownership,
    - retrieve durable memory,
    - load the selected agent configuration,
    - expose approved trading tools,
    - save messages and tool results.
    """
    cleaned_message = message.strip()

    if not cleaned_message:
        raise ValueError("Message cannot be empty.")

    key = (user_id, agent_id)
    history = conversation_history[key]

    history.append(
        {
            "role": "user",
            "content": cleaned_message,
        }
    )

    # Keep only the latest six user-assistant exchanges for the MVP.
    if len(history) > 12:
        del history[:-12]

    prompt_chars = len(SYSTEM_PROMPT) + sum(
        len(str(m.get("content") or "")) for m in history
    )

    try:
        client = get_claude_client()
        candidates = _chat_model_candidates(model)
        last_stub_model: str | None = None
        answer = ""
        for index, candidate in enumerate(candidates):
            is_last = index == len(candidates) - 1
            try:
                response = await client.messages.create(
                    model=candidate,
                    max_tokens=1200,
                    system=SYSTEM_PROMPT,
                    messages=history,
                )
            except APIError:
                if is_last:
                    raise
                print(
                    f"chat: model={candidate!r} request failed; trying fallback"
                )
                continue

            reply = extract_text(response)
            usage = getattr(response, "usage", None)
            input_tokens = (
                getattr(usage, "input_tokens", None) if usage is not None else None
            )
            if reply and _is_stub_assistant_reply(
                reply, input_tokens=input_tokens, prompt_chars=prompt_chars
            ):
                last_stub_model = candidate
                print(
                    f"chat: stub reply from model={candidate!r} "
                    f"input_tokens={input_tokens}; trying fallback"
                )
                continue

            answer = reply
            break
        else:
            raise _stub_reply_error(last_stub_model, action="chat")
    except Exception:
        # Avoid retaining a user message that never received an answer.
        if history and history[-1]["role"] == "user":
            history.pop()

        raise

    if not answer:
        answer = "Claude returned an empty response."

    history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    if len(history) > 12:
        del history[:-12]

    return answer


def reset_agent_conversation(
    *,
    user_id: str,
    agent_id: str,
) -> None:
    key = (user_id, agent_id)
    conversation_history.pop(key, None)


# System prompt for compiling a conversation/idea into a single, self-contained
# free-form strategy prompt. The output is fed to the backtest agent each hour;
# the backtest engine appends the market snapshot + JSON output contract, so this
# must NOT specify any output format.
STRATEGY_SYNTH_SYSTEM = """You are a trading-strategy compiler for Agentic Trading Lab.

Read the conversation and/or idea, then output a SINGLE, self-contained trading
strategy prompt that an LLM trading agent will follow each market hour to trade
DJIA stocks in a backtest.

Output rules:
- Output ONLY the strategy prompt text. No preamble, no markdown headers, no JSON.
- Be concrete about entry rules, exit rules, position sizing, and risk, grounded
  in the signals the agent will have: price, SMA20, SMA50, MACD, RSI, recent
  momentum, current holdings, and cash.
- Do NOT describe any output/JSON format; the system adds that automatically.
- Do NOT invent data sources the agent cannot see (no live news/Twitter/APIs).
- Keep it under ~250 words and directly actionable.
""".strip()


async def synthesize_strategy_prompt(
    *,
    user_id: str,
    agent_id: str,
    extra: str | None = None,
    model: str | None = None,
) -> str:
    """Compile a user's conversation (+ optional extra text) into one strategy prompt.

    Uses the hosted chat model. Pulls the user's existing conversation history
    (from prior ``chat_with_agent`` turns) and an optional ``extra`` instruction,
    and returns a single free-form strategy prompt suitable for
    ``POST /backtest/run`` (``strategy_prompt``) — no JSON, no formatting.

    ``model`` should be the selected agent's model when available (same as
    ``/ask``); otherwise ``resolve_chat_model()`` is used, with CommonStack
    fallbacks if the primary provider returns a known stub greeting or a
    request error.
    """
    key = (user_id, agent_id)
    history = list(conversation_history[key])

    final_instruction = (
        "Compile everything above into the final strategy prompt now. "
        "Output only the strategy prompt text."
    )
    if extra and extra.strip():
        final_instruction = (
            f"Strategy idea / requirements:\n{extra.strip()}\n\n" + final_instruction
        )

    if not history and not (extra and extra.strip()):
        raise ValueError(
            "Nothing to compile: chat about your strategy first, or provide a description."
        )

    messages = history + [{"role": "user", "content": final_instruction}]
    prompt_chars = len(STRATEGY_SYNTH_SYSTEM) + sum(
        len(str(m.get("content") or "")) for m in messages
    )

    client = get_claude_client()
    candidates = _chat_model_candidates(model)
    last_stub_model: str | None = None
    for index, candidate in enumerate(candidates):
        is_last = index == len(candidates) - 1
        try:
            response = await client.messages.create(
                model=candidate,
                max_tokens=900,
                system=STRATEGY_SYNTH_SYSTEM,
                messages=messages,
            )
        except APIError:
            if is_last:
                raise
            print(
                f"strategy synth: model={candidate!r} request failed; trying fallback"
            )
            continue

        strategy = extract_text(response).strip()
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
        if not strategy:
            continue
        if _is_stub_assistant_reply(
            strategy, input_tokens=input_tokens, prompt_chars=prompt_chars
        ):
            last_stub_model = candidate
            print(
                f"strategy synth: stub reply from model={candidate!r} "
                f"input_tokens={input_tokens}; trying fallback"
            )
            continue
        return strategy

    if last_stub_model:
        raise _stub_reply_error(last_stub_model, action="strategy")
    raise RuntimeError("The model returned an empty strategy prompt.")
