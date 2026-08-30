"""LLM backtest harness: external model-interaction infrastructure.

Extracted (Phase 2C2) from ``PortfolioManager.make_trading_decision_with_llm`` in
``dashboard/scripts/backtest_hourly_agent.py``. This module owns the *external
LLM infrastructure* concerns only:

* the optional Anthropic SDK import (``Anthropic`` / ``HAS_ANTHROPIC``);
* the default model name (``LLM_MODEL_NAME``);
* the system prompt and request parameters;
* Anthropic client invocation / prompt submission;
* response-text extraction;
* token-usage extraction;
* response parsing with the existing JSON-repair fallbacks.

Business rules (market-snapshot construction, action conversion, position
sizing, rule-based fallback) and portfolio/manager-state mutation deliberately
remain in the legacy ``PortfolioManager`` wrapper. Behavior here is byte-for-byte
identical to the original inline logic.

This module reuses the already-extracted ``fix_json_formatting`` from
``dashboard.backend.infrastructure.llm.decision_parsing`` (no duplication) and is
domain/infrastructure-only: it must NOT import FastAPI, API routers, the database
singleton, Alpaca clients, dashboard scripts, ``PortfolioManager``, or
``HourlyBacktester``. Importing it is safe without an Anthropic API key (the SDK
import is optional and there is no import-time network or credential access).
"""

import json
import os
from typing import Dict, Iterator, Optional, Tuple

from dashboard.backend.infrastructure.llm.decision_parsing import fix_json_formatting
from dashboard.backend.infrastructure.llm.providers import (
    anthropic_native,
    commonstack,
    openrouter,
)
from dashboard.backend.infrastructure.llm.providers import (
    default_model_name as _providers_default_model_name,
)
from dashboard.backend.infrastructure.llm.providers import (
    make_llm_client as _providers_make_llm_client,
)

# Optional: LLM integration. Mirrors the original optional-dependency behavior
# (no API key required at import time; only the SDK presence is detected).
try:
    # Reached by attribute rather than `from anthropic import Anthropic`: the
    # name exists here only to be re-exported (backtest_hourly_agent and the
    # engines read it through this module), which `py/unused-import` cannot see
    # because the rule is intra-file.
    import anthropic

    Anthropic = anthropic.Anthropic
    HAS_ANTHROPIC = True
except (ImportError, AttributeError):
    # AttributeError as well as ImportError: attribute access is what makes the
    # re-export visible to `py/unused-import`, but it also moves the "SDK is
    # unusable" signal from ImportError to AttributeError. This module is on the
    # backend's import path, so an unhandled one would not just disable LLM
    # trading -- it would take the app's import down with it.
    # Bind ``Anthropic`` to None (instead of leaving it undefined) so the legacy
    # script can unconditionally re-export it; consumers always guard on
    # ``HAS_ANTHROPIC`` before using the client.
    Anthropic = None
    HAS_ANTHROPIC = False
    print("⚠️  Anthropic SDK not installed. Fallback to rule-based trading.")
    print("   To enable LLM: pip install anthropic")

# Default model name (model selection). Re-exported by the legacy script.
LLM_MODEL_NAME = anthropic_native.DEFAULT_MODEL

# Same default model, but as the CommonStack gateway slug. CommonStack expects
# ``provider/model`` ids, so when routing through CommonStack we must send the
# gateway slug rather than a native Anthropic id (see the integration report's
# "slug/gateway coupling" note). Default is DeepSeek (see commonstack.py).
COMMONSTACK_MODEL_NAME = commonstack.DEFAULT_MODEL

# CommonStack / OpenRouter base URLs (env-overridable). Re-exported for
# callers that still read these constants (e.g. chat service).
COMMONSTACK_BASE_URL = commonstack.base_url()
OPENROUTER_MODEL_NAME = openrouter.DEFAULT_MODEL
OPENROUTER_BASE_URL = openrouter.base_url()


class LLMConfigurationError(RuntimeError):
    """Raised when an explicitly requested LLM cannot be initialized."""


def make_llm_client(
    integration: Optional[str] = None,
    *,
    reasoning_effort: Optional[str] = None,
):
    """Create an Anthropic-compatible client for trading decisions.

    ``integration`` selects a gateway (``commonstack``, ``openrouter``, or
    ``anthropic``). When omitted, prefers CommonStack if ``COMMONSTACK_API_KEY``
    is set, otherwise native Anthropic — OpenRouter is opt-in only. Returns
    ``None`` when the SDK or the chosen integration's key is unavailable, so
    callers fall back to rule-based trading.
    """
    return _providers_make_llm_client(
        integration,
        reasoning_effort=reasoning_effort,
    )


def default_model_name(integration: Optional[str] = None) -> str:
    """Return the default model id matching the client ``make_llm_client`` builds.

    Pass the same ``integration`` used for the client so the slug matches the
    gateway (CommonStack / OpenRouter ``provider/model`` vs native Anthropic id).
    Callers can still override with an explicit model id.
    """
    return _providers_default_model_name(integration)

# System prompt sent on every request. Preserved exactly from the original
# inline string (do not "improve" it).
SYSTEM_PROMPT = """You are an expert quantitative trading advisor analyzing DJIA stocks.

You have deep knowledge of:
- Technical analysis (RSI, MACD, Bollinger Bands, Moving Averages)
- Indicator interpretation and confluence
- Risk management and position sizing
- Trading psychology and market microstructure

IMPORTANT INSTRUCTIONS:
1. Analyze EACH stock signal provided (don't skip any)
2. For each stock, decide: BUY, SELL, or HOLD
3. Always include a confidence score (0.0-1.0)
4. Return a JSON object with an "actions" array containing one entry per stock
5. Even if you decide HOLD, include it in the actions array
6. Respond with ONLY valid JSON - no explanations outside JSON

Make precise, actionable trading decisions based on the technical indicators provided."""

A_SHARE_SYSTEM_PROMPT = """You are an expert quantitative trading advisor analyzing Chinese A-share stocks.

This is a historical paper backtest using 60m bars in the Asia/Shanghai timezone.
It never sends real orders or uses real money.

You have deep knowledge of:
- Technical analysis (RSI, MACD, Bollinger Bands, Moving Averages)
- Indicator interpretation and confluence
- Risk management and position sizing
- Trading psychology and market microstructure

IMPORTANT INSTRUCTIONS:
1. Analyze EACH stock signal provided (don't skip any)
2. For each stock, decide: BUY, SELL, or HOLD
3. Always include a confidence score (0.0-1.0)
4. Return a JSON object with an "actions" array containing one entry per stock
5. Even if you decide HOLD, include it in the actions array
6. BUY and SELL position_size values must be positive multiples of 100 shares
7. Respond with ONLY valid JSON - no explanations outside JSON

Make precise, actionable trading decisions based on the technical indicators provided."""


def system_prompt_for_market(market_context: Optional[Dict] = None) -> str:
    """Return a backend-owned system prompt for the selected market."""
    market = str((market_context or {}).get("market") or "").strip().upper()
    if market == "CN":
        return A_SHARE_SYSTEM_PROMPT
    return SYSTEM_PROMPT


# Per-request output-token ceiling. Defaults to 2000 (unchanged) but can be
# lowered via env (e.g. LLM_MAX_OUTPUT_TOKENS=600) to cap spend in small demos.
def _parse_max_output_tokens(raw: Optional[str]) -> int:
    """Parse ``LLM_MAX_OUTPUT_TOKENS`` defensively.

    A malformed or non-positive value must not crash the module at import time
    (this constant is evaluated on ``import``, so a bad env var would take down
    every backtest CLI and the backend). Falls back to 2000 with a warning.
    """
    if raw is None:
        return 2000
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        print(
            f"⚠️ Ignoring invalid LLM_MAX_OUTPUT_TOKENS={raw!r} "
            f"(expected a positive integer); using default 2000"
        )
        return 2000
    return value


DEFAULT_MAX_OUTPUT_TOKENS = _parse_max_output_tokens(os.getenv("LLM_MAX_OUTPUT_TOKENS"))


def request_trading_decision(
    client,
    *,
    prompt: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    market_context: Optional[Dict] = None,
):
    """Submit the prompt to the Anthropic client and return the raw response.

    Mirrors the original ``llm_client.messages.create(...)`` call exactly: same
    model resolution (``model or LLM_MODEL_NAME``), ``max_tokens=2000``, system
    prompt, and single user message. Exceptions are intentionally NOT caught here
    so the legacy wrapper's outer handler can fall back to rule-based logic,
    exactly as before. Temperature is omitted unless explicitly configured.
    """
    request_kwargs = {
        "model": model or LLM_MODEL_NAME,
        "max_tokens": max_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
        "system": system_prompt_for_market(market_context),
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    return client.messages.create(**request_kwargs)


def extract_response_text(response) -> str:
    """Extract assistant text from an Anthropic-shaped messages response.

    Historically this was ``response.content[0].text``. Reasoning / "thinking"
    models (e.g. NVIDIA Nemotron via OpenRouter) often put a ``ThinkingBlock``
    first — that object has no ``.text``, so indexing ``[0]`` crashes and the
    portfolio manager falls back to rule-based trading. Prefer every block that
    exposes usable text (``type == "text"`` or a ``text`` attribute); join them
    if the model split the answer across multiple text blocks.
    """
    blocks = getattr(response, "content", None) or []
    texts: list[str] = []
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type is not None and block_type != "text":
            # Skip thinking / tool_use / redacted_thinking / etc.
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            texts.append(text)
    if texts:
        return "\n".join(texts)
    # Last resort: some SDKs expose a top-level string; preserve a clear error
    # if nothing usable is present (caller falls back to rule-based).
    raise AttributeError(
        "No text content block in LLM response "
        f"(content types: {[getattr(b, 'type', type(b).__name__) for b in blocks]})"
    )

def extract_token_usage(response):
    """Return ``(input_tokens, output_tokens)`` deltas from a provider response.

    Mirrors the original usage reads: ``getattr(response, "usage", None)`` then
    ``int(getattr(usage, "input_tokens", 0) or 0)`` etc., returning ``(0, 0)``
    when no usage object is present. The caller is responsible for applying these
    to manager state and incrementing the call counter (kept in the wrapper).
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def _structural_braces(text: str) -> Iterator[Tuple[int, str]]:
    """Yield ``(index, brace)`` for each ``{`` / ``}`` outside a string literal.

    Tracks JSON string boundaries, including backslash-escaped quotes, so a
    brace inside a ``reason`` value counts as data rather than structure.
    """
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{}":
            yield index, char


def _structural_brace_counts(text: str) -> Tuple[int, int]:
    """``(open, close)`` counts of the braces that are actually structure."""
    braces = [brace for _, brace in _structural_braces(text)]
    open_count = braces.count("{")
    return open_count, len(braces) - open_count


def _trim_extra_closing_braces(text: str) -> str:
    """Cut at the last structural ``}`` until closers no longer outnumber openers.

    Each pass removes at least one closer, so the loop always terminates.
    """
    while True:
        braces = list(_structural_braces(text))
        closers = [index for index, brace in braces if brace == "}"]
        if len(closers) <= len(braces) - len(closers):
            return text
        text = text[: closers[-1]]


def parse_llm_response(llm_response: str) -> Optional[Dict]:
    """Parse the LLM response into a decision dict, or ``None`` on failure.

    Replicates the original STEP-3 parsing exactly: strip markdown fences, slice
    from the first ``{`` to the last ``}``, ``json.loads`` with two escalating
    ``fix_json_formatting`` / bracket-balancing repair attempts, and the same
    printed diagnostics. Returns the parsed ``decision`` dict on success.

    Returns ``None`` in every case where the original method returned
    ``{"actions": []}`` (no JSON found, unrecoverable parse error, or any other
    exception in the parse block), so the caller can return ``{"actions": []}``
    and preserve behavior.
    """
    print(f"\n📫 Parsing LLM response...")
    print(f"   Raw response (first 300 chars): {llm_response[:300]}")

    try:
        # Extract JSON from response
        # First, strip markdown code fences if present
        response_cleaned = llm_response
        if '```json' in response_cleaned:
            response_cleaned = response_cleaned.replace('```json', '').replace('```', '')
        elif '```' in response_cleaned:
            response_cleaned = response_cleaned.replace('```', '')

        start = response_cleaned.find('{')
        end = response_cleaned.rfind('}') + 1
        if start < 0 or end <= 0:
            print(f"   ❌ No JSON found in response")
            print(f"   Full response: {response_cleaned[:500]}")
            return None

        json_str = response_cleaned[start:end]

        # Try to parse
        try:
            decision = json.loads(json_str)
            print(f"   ✅ JSON parsed successfully")
        except json.JSONDecodeError as e:
            # Try to fix common formatting issues
            print(f"   ⚠️  Initial parse failed: {e}")
            print(f"   Attempting to fix JSON formatting...")

            json_str_fixed = fix_json_formatting(json_str)
            try:
                decision = json.loads(json_str_fixed)
                print(f"   ✅ JSON fixed and parsed successfully!")
            except json.JSONDecodeError as e2:
                print(f"   ❌ Still failed after fix: {e2}")
                print(f"   Error at line {e2.lineno}, column {e2.colno}")

                # Show detailed context around error
                lines = json_str_fixed.split('\n')
                if e2.lineno <= len(lines):
                    start = max(0, e2.lineno - 3)
                    end = min(len(lines), e2.lineno + 2)
                    print(f"\n   Context around error (lines {start+1}-{end}):")
                    for i in range(start, end):
                        marker = ">> " if i == e2.lineno - 1 else "   "
                        print(f"   {marker}{i+1:3d}: {lines[i][:70]}")

                # Try one more aggressive fix
                print(f"\n   Attempting second fix attempt (validate structure)...")
                try:
                    # Count opening vs closing braces outside string values:
                    # ``str.count`` also saw a ``}`` inside a "reason", and
                    # the trim below then stripped a real closer with the
                    # stray one, turning a parseable decision into None.
                    open_count, close_count = _structural_brace_counts(json_str_fixed)
                    if open_count != close_count:
                        print(f"   Bracket mismatch: {open_count} open, {close_count} close")
                        # Remove extra closing braces from the end, one per
                        # pass. (An older form re-appended the brace it had
                        # just cut, so the counts never changed and the loop
                        # spun forever on a reply with one stray ``}``.)
                        json_str_fixed = _trim_extra_closing_braces(json_str_fixed)
                        print(f"   Removed extra closing brackets")

                    decision = json.loads(json_str_fixed)
                    print(f"   ✅ JSON fixed after structure cleanup!")
                except json.JSONDecodeError as e3:
                    print(f"   ❌ Cannot fix: {e3}")
                    return None

        print(f"   Actions from LLM: {len(decision.get('actions', []))}")
        return decision

    except (json.JSONDecodeError, ValueError, Exception) as e:
        print(f"   ❌ Failed to parse JSON: {e}")
        print(f"   LLM response: {llm_response[:500]}...")
        return None
