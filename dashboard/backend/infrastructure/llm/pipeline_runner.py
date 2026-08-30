"""Sequential sub-agent pipeline executor for hourly backtest decisions.

Each decision-pipeline step is one LLM call. Step 1 receives the market
snapshot; later steps receive upstream JSON outputs. The final decision step
should emit either ``actions`` (standard trading contract) or ``orders`` /
``risk_actions`` which are normalized into ``actions`` for the existing
portfolio executor.

Steps with ``presetKey == "post_trade_analysis"`` are stripped from the hourly
decision path and executed once per trading day after trades settle.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from dashboard.backend.infrastructure.llm.backtest_harness import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    LLM_MODEL_NAME,
    extract_response_text,
    extract_token_usage,
    parse_llm_response,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
)

POST_TRADE_PRESET_KEY = "post_trade_analysis"

PIPELINE_SYSTEM_PROMPT = """You are a sub-agent in a multi-step trading pipeline.
Follow your task instructions precisely.
Return ONLY valid JSON matching the required output format.
No markdown, no code fences, no explanatory text outside the JSON."""

POST_TRADE_SYSTEM_PROMPT = """You are the post-trade analysis sub-agent.
Review one trading day's episode and improve upstream decision-step prompts.
Return ONLY valid JSON matching the required output format.
No markdown, no code fences, no explanatory text outside the JSON.
Only revise prompts for the listed decision steps. Never invent trades or prices
beyond the episode context."""

# Keep normal calls at the configured ceiling, but give a recovery attempt room
# for both reasoning tokens and the final JSON. This is intentionally scoped to
# retries so reasoning stays enabled without doubling every successful call.
RECOVERY_MAX_OUTPUT_TOKENS = max(DEFAULT_MAX_OUTPUT_TOKENS, 4096)


def is_post_trade_step(step: Any) -> bool:
    return isinstance(step, dict) and step.get("presetKey") == POST_TRADE_PRESET_KEY


def split_pipeline(
    pipeline: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a mixed pipeline into hourly decision steps and post-trade steps."""
    decision_steps: List[Dict[str, Any]] = []
    post_trade_steps: List[Dict[str, Any]] = []
    if not pipeline:
        return decision_steps, post_trade_steps
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        if is_post_trade_step(step):
            post_trade_steps.append(step)
        else:
            decision_steps.append(step)
    return decision_steps, post_trade_steps


def trading_day_key(timestamp: Any) -> str:
    """Calendar-day key for day-boundary post-trade triggers."""
    if timestamp is None:
        return ""
    if hasattr(timestamp, "date"):
        try:
            return timestamp.date().isoformat()
        except Exception:
            pass
    text = str(timestamp)
    if "T" in text:
        return text.split("T", 1)[0][:10]
    return text[:10]


def is_last_bar_of_trading_day(
    timestamps: List[Any],
    index: int,
) -> bool:
    """True when ``timestamps[index]`` is the last bar of its calendar day."""
    if not timestamps or index < 0 or index >= len(timestamps):
        return False
    if index == len(timestamps) - 1:
        return True
    return trading_day_key(timestamps[index]) != trading_day_key(timestamps[index + 1])


def _build_step_prompt(
    *,
    step_index: int,
    step: Dict[str, Any],
    market_snapshot: Dict[str, Any],
    prior_outputs: List[Dict[str, Any]],
    is_last: bool,
) -> str:
    label = (step.get("label") or f"Step {step_index + 1}").strip()
    task = (step.get("prompt") or "").strip()
    output_format = (step.get("outputFormat") or "").strip()

    parts = [
        f"=== SUB-AGENT: {label} ===",
        task,
        "",
        "=== REQUIRED OUTPUT FORMAT ===",
        output_format or '{"output": "..."}',
    ]

    if step_index == 0:
        parts.extend(
            [
                "",
                "=== MARKET SNAPSHOT ===",
                json.dumps(market_snapshot, indent=2),
            ]
        )

    if prior_outputs:
        parts.extend(
            [
                "",
                "=== UPSTREAM PIPELINE OUTPUTS ===",
                json.dumps(prior_outputs, indent=2),
            ]
        )

    if is_last:
        parts.extend(
            [
                "",
                "=== EXECUTION RULES ===",
                "- Trade ONLY symbols listed in the market snapshot.",
                "- Only SELL symbols that appear in current_holdings.",
                "- Respect available cash for buy orders.",
                "- Use integer share quantities.",
            ]
        )

    parts.extend(["", "Return ONLY valid JSON matching the required output format."])
    return "\n".join(parts)


def pipeline_output_to_decision(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a pipeline step's JSON into the standard ``actions`` decision."""
    if not isinstance(parsed, dict):
        return None

    saw_empty_envelope = False
    saw_unusable_nonempty_envelope = False

    actions = parsed.get("actions")
    if isinstance(actions, list):
        if actions:
            return {"actions": actions}
        saw_empty_envelope = True

    orders = parsed.get("orders")
    if isinstance(orders, list):
        if not orders:
            saw_empty_envelope = True
        else:
            normalized = []
            for order in orders:
                if not isinstance(order, dict):
                    continue
                side = str(order.get("side") or order.get("action") or "hold").lower()
                if side not in ("buy", "sell", "hold"):
                    side = "hold"
                qty = order.get(
                    "qty", order.get("quantity", order.get("position_size", 0))
                )
                try:
                    position_size = int(qty)
                except (TypeError, ValueError):
                    position_size = 0
                normalized.append(
                    {
                        "action": side,
                        "symbol": order.get("symbol"),
                        "confidence": float(order.get("confidence", 0.75) or 0.75),
                        "reasoning": order.get("reason")
                        or order.get("rationale")
                        or "",
                        "position_size": position_size,
                        "stop_loss_price": order.get("stop_loss_price"),
                        "take_profit_price": order.get("take_profit_price"),
                    }
                )
            if normalized:
                return {"actions": normalized}
            saw_unusable_nonempty_envelope = True

    risk_actions = parsed.get("risk_actions")
    if isinstance(risk_actions, list):
        if not risk_actions:
            saw_empty_envelope = True
        else:
            normalized = []
            for risk in risk_actions:
                if not isinstance(risk, dict):
                    continue
                action_type = str(risk.get("action") or "hold").lower()
                if action_type in ("stop_loss", "take_profit", "trail"):
                    side = "sell"
                elif action_type == "hold":
                    side = "hold"
                else:
                    side = (
                        action_type
                        if action_type in ("buy", "sell", "hold")
                        else "hold"
                    )
                size_pct = float(risk.get("size_pct", 1.0) or 1.0)
                normalized.append(
                    {
                        "action": side,
                        "symbol": risk.get("symbol"),
                        "confidence": 0.8,
                        "reasoning": risk.get("reason")
                        or risk.get("rationale")
                        or action_type,
                        "position_size": (
                            max(1, int(round(size_pct * 100)))
                            if side == "sell"
                            else 0
                        ),
                    }
                )
            if normalized:
                return {"actions": normalized}
            saw_unusable_nonempty_envelope = True

    if saw_unusable_nonempty_envelope:
        return None
    if saw_empty_envelope:
        return {"actions": []}
    return None


def apply_prompt_patches(
    decision_pipeline: List[Dict[str, Any]],
    patches: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply ``prompt_patches`` onto decision steps. Returns (new_pipeline, applied)."""
    updated = copy.deepcopy(decision_pipeline or [])
    if not isinstance(patches, list) or not patches:
        return updated, []

    by_id = {
        str(step.get("id")): step
        for step in updated
        if isinstance(step, dict) and step.get("id") is not None
    }
    by_preset: Dict[str, List[Dict[str, Any]]] = {}
    for step in updated:
        if not isinstance(step, dict):
            continue
        key = step.get("presetKey")
        if key:
            by_preset.setdefault(str(key), []).append(step)

    applied: List[Dict[str, Any]] = []
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        new_prompt = patch.get("new_prompt")
        if not isinstance(new_prompt, str) or not new_prompt.strip():
            continue
        if patch.get("presetKey") == POST_TRADE_PRESET_KEY:
            continue

        target = None
        step_id = patch.get("step_id")
        if step_id is not None and str(step_id) in by_id:
            target = by_id[str(step_id)]
        else:
            preset = patch.get("presetKey")
            candidates = by_preset.get(str(preset), []) if preset else []
            if len(candidates) == 1:
                target = candidates[0]

        if target is None or is_post_trade_step(target):
            continue

        old_prompt = target.get("prompt")
        target["prompt"] = new_prompt.strip()
        applied.append(
            {
                "step_id": target.get("id"),
                "presetKey": target.get("presetKey"),
                "label": target.get("label"),
                "old_prompt": old_prompt,
                "new_prompt": target["prompt"],
                "change_rationale": patch.get("change_rationale") or "",
            }
        )
    return updated, applied


def _serialize_day_trades(trades: List[Dict[str, Any]], *, limit: int = 40) -> List[Dict[str, Any]]:
    serialized = []
    for trade in (trades or [])[-limit:]:
        if not isinstance(trade, dict):
            continue
        ts = trade.get("timestamp")
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat()
        serialized.append(
            {
                "timestamp": ts,
                "symbol": trade.get("symbol"),
                "side": trade.get("side") or trade.get("action"),
                "quantity": trade.get("shares") or trade.get("quantity"),
                "price": trade.get("price"),
                "reason": (trade.get("reason") or "")[:240],
            }
        )
    return serialized


def _prompt_catalog(decision_pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    catalog = []
    for step in decision_pipeline or []:
        if not isinstance(step, dict):
            continue
        catalog.append(
            {
                "step_id": step.get("id"),
                "presetKey": step.get("presetKey"),
                "label": step.get("label"),
                "prompt": step.get("prompt"),
            }
        )
    return catalog


def _build_post_trade_prompt(
    *,
    step: Dict[str, Any],
    episode_context: Dict[str, Any],
    decision_pipeline: List[Dict[str, Any]],
) -> str:
    label = (step.get("label") or "Post-trade Analysis").strip()
    task = (step.get("prompt") or "").strip()
    output_format = (step.get("outputFormat") or "").strip()
    context = {
        "trading_day": episode_context.get("trading_day"),
        "day_start_equity": episode_context.get("day_start_equity"),
        "day_end_equity": episode_context.get("day_end_equity"),
        "day_return": episode_context.get("day_return"),
        "trade_count": episode_context.get("trade_count"),
        "trades": _serialize_day_trades(episode_context.get("trades") or []),
        "latest_step_outputs": episode_context.get("latest_step_outputs") or [],
        "decision_prompts": _prompt_catalog(decision_pipeline),
    }
    return "\n".join(
        [
            f"=== SUB-AGENT: {label} ===",
            task,
            "",
            "=== REQUIRED OUTPUT FORMAT ===",
            output_format
            or (
                'JSON: { "summary": "...", "prompt_problems": [], "prompt_patches": [] }'
            ),
            "",
            "=== DAY EPISODE CONTEXT ===",
            json.dumps(context, indent=2, default=str),
            "",
            "Return ONLY valid JSON matching the required output format.",
        ]
    )


def run_post_trade_analysis(
    client,
    *,
    post_trade_steps: List[Dict[str, Any]],
    episode_context: Dict[str, Any],
    decision_pipeline: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Tuple[int, int], int]:
    """Run daily post-trade LLM analysis and patch decision prompts.

    Returns ``(new_decision_pipeline, analysis_record, (in_tokens, out_tokens), llm_calls)``.
    On failure, returns the original decision pipeline unchanged.
    """
    if not post_trade_steps:
        return list(decision_pipeline or []), {}, (0, 0), 0

    total_in = 0
    total_out = 0
    llm_calls = 0
    working = copy.deepcopy(decision_pipeline or [])
    last_parsed: Dict[str, Any] = {}
    applied: List[Dict[str, Any]] = []

    for index, step in enumerate(post_trade_steps):
        if not isinstance(step, dict):
            continue
        prompt = _build_post_trade_prompt(
            step=step,
            episode_context=episode_context,
            decision_pipeline=working,
        )
        label = step.get("label") or f"Post-trade {index + 1}"
        print(f"\n📉 Post-trade analysis: {label} (day={episode_context.get('trading_day')})")

        try:
            response = client.messages.create(
                model=model or LLM_MODEL_NAME,
                max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                system=POST_TRADE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            llm_calls += 1
            in_delta, out_delta = extract_token_usage(response)
            total_in += in_delta
            total_out += out_delta
            parsed = parse_llm_response(extract_response_text(response))
        except Exception as exc:
            print(f"   ⚠️  Post-trade analysis failed: {exc}")
            if getattr(client, "fail_closed", False):
                raise
            parsed = None

        if not isinstance(parsed, dict):
            print("   ⚠️  Post-trade returned unparseable JSON; keeping prompts unchanged")
            continue

        last_parsed = parsed
        working, applied_now = apply_prompt_patches(working, parsed.get("prompt_patches"))
        applied.extend(applied_now)
        if applied_now:
            print(f"   ✅ Applied {len(applied_now)} prompt patch(es)")
        else:
            print("   ℹ️  No prompt patches applied")

    record = {
        "trading_day": episode_context.get("trading_day"),
        "day_start_equity": episode_context.get("day_start_equity"),
        "day_end_equity": episode_context.get("day_end_equity"),
        "day_return": episode_context.get("day_return"),
        "trade_count": episode_context.get("trade_count"),
        "summary": last_parsed.get("summary") if last_parsed else None,
        "prompt_problems": last_parsed.get("prompt_problems") if last_parsed else [],
        "applied_patches": applied,
    }
    return working, record, (total_in, total_out), llm_calls


def recombine_pipeline(
    decision_steps: List[Dict[str, Any]],
    post_trade_steps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Decision steps first, then post-trade steps (UI-friendly ordering)."""
    return list(decision_steps or []) + list(post_trade_steps or [])


def _create_pipeline_response(
    client,
    *,
    model: str,
    prompt: str,
    max_tokens: Optional[int] = None,
):
    """Create one pipeline request, optionally with a recovery output budget."""
    request = {
        "model": model,
        "max_tokens": (
            DEFAULT_MAX_OUTPUT_TOKENS
            if max_tokens is None
            else max_tokens
        ),
        "system": PIPELINE_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    return client.messages.create(**request)


def _retry_with_recovery_budget(client, *, model: str, prompt: str):
    """Second attempt for a step whose first attempt was unusable.

    The same request, reasoning preserved, with the output ceiling raised to
    ``RECOVERY_MAX_OUTPUT_TOKENS`` so a reasoning-heavy model has room for
    both its thinking and the final JSON. Both recovery paths send exactly
    this, which is why a step never gets a third attempt: a reply that is
    still unusable after it has nothing different left to ask for.
    """
    return _create_pipeline_response(
        client,
        model=model,
        prompt=prompt,
        max_tokens=RECOVERY_MAX_OUTPUT_TOKENS,
    )


class _StepUsage:
    """Calls and token deltas for the responses a pipeline actually received.

    Only responses that came back are recorded — an attempt that raised was
    released by the execution service and never billed.
    """

    __slots__ = ("calls", "input_tokens", "output_tokens")

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, response) -> int:
        """Count one received response; returns its output-token delta."""
        self.calls += 1
        in_delta, out_delta = extract_token_usage(response)
        self.input_tokens += in_delta
        self.output_tokens += out_delta
        return out_delta

    @property
    def totals(self) -> Tuple[int, int]:
        return self.input_tokens, self.output_tokens


# Provider-neutral spellings of "the reply stopped because it hit the output
# ceiling": Anthropic ``stop_reason``, OpenAI/OpenRouter ``finish_reason`` and
# Gemini ``finishReason`` (the execution adapters normalise the last two).
_OUTPUT_CEILING_STOP_REASONS = frozenset({"max_tokens", "length"})

# Below this many characters a structurally unclosed reply is treated as an
# ordinary malformed answer rather than a truncation, so short garbage does
# not buy a second billed call. Reasoning can consume most of the output
# budget and leave a surprisingly short JSON prefix (a production failure was
# ~170 chars), so the floor is small; it only applies to the structural
# fallback — a provider-reported stop reason or an output-token count at the
# ceiling fires regardless of length (and of script: CJK output reaches the
# ceiling at a fraction of these characters).
_TRUNCATION_MIN_CHARS = 64

# A structurally unclosed reply is only worth a retry when it had started the
# decision envelope, so unrelated malformed JSON is not treated as a truncation.
_DECISION_ENVELOPE_KEYS = ("actions", "orders", "risk_actions")

_CODE_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)
# The first brace that actually opens an object. A ``{`` inside a prose
# preamble ("Analysis {see below") would otherwise count as an unclosed
# delimiter and buy a retry for a reply that is complete.
_OBJECT_START = re.compile(r'\{\s*"')


def _hit_output_ceiling(response, output_tokens: int, max_tokens: int) -> bool:
    """True when the provider itself says the reply stopped at ``max_tokens``.

    Prefers the explicit stop/finish reason where the response carries one
    (the raw Anthropic SDK and the execution client both expose
    ``stop_reason``); falls back to the reported output-token count reaching
    the ceiling the request asked for. Either signal is exact where the
    structural scan below is a guess.
    """
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason is None:
        stop_reason = getattr(response, "finish_reason", None)
    if isinstance(stop_reason, str):
        if stop_reason.strip().lower() in _OUTPUT_CEILING_STOP_REASONS:
            return True
    return output_tokens >= max_tokens


def _looks_like_truncated_json(response_text: str) -> bool:
    """Structural fallback: a decision object whose delimiters never close.

    Provider adapters already classify an empty response as
    ``response_invalid``, and ``_hit_output_ceiling`` covers every response
    that reports its own stop reason or usage. This scan is for the rest: a
    reply that reaches the parser with text, begins a decision envelope, and
    stops before the delimiters close. It tolerates exactly what
    ``parse_llm_response`` tolerates — code fences of any case and prose
    before the object — so a truncation the parser would have seen is not
    hidden from the retry by its preamble; the scan starts at the first
    ``{"`` so a brace *inside* that preamble is not mistaken for the object.
    A *mismatched* closer (``}`` closing a ``[``) is a malformed reply, not a
    cut-off one, and is deliberately not retried; neither is a short unclosed
    fragment or one that never named a decision key.
    """
    text = _CODE_FENCE.sub("", response_text)
    opener = _OBJECT_START.search(text)
    if opener is None:
        return False
    text = text[opener.start():].strip()
    if len(text) < _TRUNCATION_MIN_CHARS:
        return False
    if not any(f'"{key}"' in text for key in _DECISION_ENVELOPE_KEYS):
        return False

    stack: list[str] = []
    in_string = False
    escaped = False
    matching = {"}": "{", "]": "["}
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or stack[-1] != matching[char]:
                return False
            stack.pop()
    return bool(stack)


def truncation_reason(response, output_tokens: int, response_text: str) -> Optional[str]:
    """Why an unparseable first reply is worth one more attempt, or ``None``.

    Shared with the non-pipeline decision path in
    ``domain/backtesting/portfolio_manager.py``, which spends the same single
    recovery budget on the same two signals.
    """
    if _hit_output_ceiling(response, output_tokens, DEFAULT_MAX_OUTPUT_TOKENS):
        return "stopped at the output ceiling"
    if _looks_like_truncated_json(response_text):
        return "is structurally incomplete"
    return None


def response_text_or_none(response) -> Optional[str]:
    """Text of a response, or ``None`` when it carries no text block.

    A reasoning model can spend the whole reply on a thinking block; that is
    an unparseable reply, not a fault, so it must not raise past the usage
    the step has already recorded for it.
    """
    try:
        return extract_response_text(response)
    except AttributeError as exc:
        if "No text content" not in str(exc):
            raise
        return None


def run_pipeline_decision(
    client,
    *,
    pipeline: List[Dict[str, Any]],
    market_snapshot: Dict[str, Any],
    model: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Tuple[int, int], int, List[Dict[str, Any]]]:
    """Execute decision pipeline steps sequentially.

    Post-trade steps are ignored here. Returns
    ``(decision_dict_or_none, (input_tokens, output_tokens), llm_calls, step_outputs)``.

    Each step gets at most **one** second attempt (``_retry_with_recovery_budget``),
    and only when the first attempt was unusable: the provider rejected it as
    ``response_invalid``, or it came back unparseable *and* truncated (see
    ``truncation_reason``). A reply with no text block at all is unparseable
    by definition: it is retried when the provider reports the ceiling, and
    otherwise ends the step cleanly with its usage intact. The two triggers
    share that single budget on
    purpose — the recovery attempt is the same request either way, so a
    truncated recovery reply has nothing different left to send. A truncation
    retry that fails in a way the provider would also have classified as an
    unusable reply degrades to the first attempt's outcome (``None`` for this
    step) rather than raising an error the provider never produced, so the
    usage already recorded for the step — a real, billed call — is returned to
    the caller instead of being lost with the exception.
    """
    decision_steps, _post_trade_steps = split_pipeline(pipeline)
    if not decision_steps:
        return None, (0, 0), 0, []

    prior_outputs: List[Dict[str, Any]] = []
    usage = _StepUsage()
    last_parsed: Optional[Dict[str, Any]] = None
    request_model = model or LLM_MODEL_NAME

    for index, step in enumerate(decision_steps):
        if not isinstance(step, dict):
            print(f"   ⚠️  Pipeline step {index + 1} is invalid; aborting pipeline")
            return None, usage.totals, usage.calls, prior_outputs

        prompt = _build_step_prompt(
            step_index=index,
            step=step,
            market_snapshot=market_snapshot,
            prior_outputs=prior_outputs,
            is_last=(index == len(decision_steps) - 1),
        )
        label = step.get("label") or f"Step {index + 1}"
        print(f"\n🔗 Pipeline step {index + 1}/{len(decision_steps)}: {label}")

        retried = False
        try:
            response = _create_pipeline_response(
                client,
                model=request_model,
                prompt=prompt,
            )
        except LLMExecutionError as first_error:
            if first_error.category is not ExecutionErrorCategory.RESPONSE_INVALID:
                raise
            print(
                "   ⚠️  Empty model response; retrying with reasoning preserved "
                f"and max_tokens={RECOVERY_MAX_OUTPUT_TOKENS}"
            )
            response = _retry_with_recovery_budget(
                client, model=request_model, prompt=prompt
            )
            retried = True
        out_delta = usage.record(response)

        response_text = response_text_or_none(response)
        if response_text is None:
            print("   ⚠️  Pipeline response carried no text block")
            parsed = None
        else:
            parsed = parse_llm_response(response_text)
        reason = None
        if parsed is None and not retried:
            reason = truncation_reason(response, out_delta, response_text or "")
        if reason is not None:
            print(
                f"   ⚠️  Pipeline response {reason}; retrying once with reasoning "
                f"preserved and max_tokens={RECOVERY_MAX_OUTPUT_TOKENS}"
            )
            retry_response = None
            try:
                retry_response = _retry_with_recovery_budget(
                    client, model=request_model, prompt=prompt
                )
            except LLMExecutionError as retry_error:
                # An empty second reply is the same class of outcome as the
                # truncated first one; anything else is an infrastructure
                # failure the first attempt would have raised too.
                if retry_error.category is not ExecutionErrorCategory.RESPONSE_INVALID:
                    raise
                print("   ⚠️  Retry returned no usable response; keeping the first attempt")
            if retry_response is not None:
                usage.record(retry_response)
                retry_text = response_text_or_none(retry_response)
                if retry_text is None:
                    print("   ⚠️  Retry returned no text block; keeping the first attempt")
                else:
                    parsed = parse_llm_response(retry_text)
        if parsed is None:
            print(f"   ❌ Pipeline step {index + 1} returned unparseable JSON")
            return None, usage.totals, usage.calls, prior_outputs

        prior_outputs.append(
            {
                "step": index + 1,
                "label": label,
                "presetKey": step.get("presetKey"),
                "id": step.get("id"),
                "output": parsed,
            }
        )
        last_parsed = parsed

    decision = pipeline_output_to_decision(last_parsed or {})
    if decision is None:
        print("   ❌ Final pipeline output could not be converted to trading actions")
        return None, usage.totals, usage.calls, prior_outputs

    return decision, usage.totals, usage.calls, prior_outputs
