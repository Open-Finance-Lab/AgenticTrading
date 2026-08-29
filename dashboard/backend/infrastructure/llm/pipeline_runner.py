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
from typing import Any, Dict, List, Optional, Tuple

from dashboard.backend.infrastructure.llm.backtest_harness import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    LLM_MODEL_NAME,
    extract_response_text,
    extract_token_usage,
    parse_llm_response,
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
    """
    decision_steps, _post_trade_steps = split_pipeline(pipeline)
    if not decision_steps:
        return None, (0, 0), 0, []

    prior_outputs: List[Dict[str, Any]] = []
    total_in = 0
    total_out = 0
    llm_calls = 0
    last_parsed: Optional[Dict[str, Any]] = None

    for index, step in enumerate(decision_steps):
        if not isinstance(step, dict):
            print(f"   ⚠️  Pipeline step {index + 1} is invalid; aborting pipeline")
            return None, (total_in, total_out), llm_calls, prior_outputs

        prompt = _build_step_prompt(
            step_index=index,
            step=step,
            market_snapshot=market_snapshot,
            prior_outputs=prior_outputs,
            is_last=(index == len(decision_steps) - 1),
        )
        label = step.get("label") or f"Step {index + 1}"
        print(f"\n🔗 Pipeline step {index + 1}/{len(decision_steps)}: {label}")

        response = client.messages.create(
            model=model or LLM_MODEL_NAME,
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            system=PIPELINE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        llm_calls += 1
        in_delta, out_delta = extract_token_usage(response)
        total_in += in_delta
        total_out += out_delta

        parsed = parse_llm_response(extract_response_text(response))
        if parsed is None:
            print(f"   ❌ Pipeline step {index + 1} returned unparseable JSON")
            return None, (total_in, total_out), llm_calls, prior_outputs

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
        return None, (total_in, total_out), llm_calls, prior_outputs

    return decision, (total_in, total_out), llm_calls, prior_outputs
