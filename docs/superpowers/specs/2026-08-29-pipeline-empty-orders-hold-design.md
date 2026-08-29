# Pipeline Empty Orders Hold Design

## Problem

The hourly backtest pipeline treats a valid empty decision envelope as an
unparseable model decision. Both Gemini and Qwen have returned valid JSON such
as:

```json
{"orders": []}
```

An empty order list means that the model chose not to trade during the current
bar. `pipeline_output_to_decision()` currently checks list truthiness, so an
empty list falls through to `None`. Strict LLM backtests then raise
`LLMDecisionError` and discard the run even though parsing succeeded.

## Goal

Normalize an explicitly empty supported pipeline decision envelope to
`{"actions": []}` so the backtest records a model-driven hold and continues.

## Scope

This change covers the three decision envelopes already supported by the
pipeline runner:

- `{"actions": []}`
- `{"orders": []}`
- `{"risk_actions": []}`

The change is limited to pipeline decision normalization and its contract
tests. It does not change prompts, provider adapters, billing, strict-LLM
fallback budgets, retry policy, frontend polling, concurrency, or Render
resource limits.

## Normalization Contract

`pipeline_output_to_decision(parsed)` keeps the current non-empty envelope
priority and conversion behavior:

1. A non-empty `actions` list is returned as the standard action envelope.
2. Otherwise, a non-empty `orders` list is normalized into actions.
3. Otherwise, a non-empty `risk_actions` list is normalized into actions.
4. If a non-empty `orders` or `risk_actions` list contains no normalizable
   records, return `None`; an empty sibling field must not hide that invalid
   payload.
5. If no supported list is non-empty and at least one supported field is
   explicitly an empty list, return `{"actions": []}`.
6. Return `None` for non-dictionary input, missing supported fields, or payloads
   whose supported fields are only non-list values.

The empty result is a model-driven HOLD, not a rule-based fallback. The
existing portfolio manager therefore counts it as an LLM decision without
executing trades.

## Testing

Add focused contract tests that prove:

- each supported empty envelope normalizes to `{"actions": []}`;
- missing and incorrectly typed envelopes remain invalid;
- a non-empty orders envelope with no valid records remains invalid;
- an empty envelope cannot mask a non-empty invalid orders envelope;
- `run_pipeline_decision()` accepts an `orders: []` provider response and
  returns a valid empty action decision instead of `None`.

Run the pipeline runner test module and the strict portfolio-manager tests to
verify both the normalization boundary and the downstream strict-LLM behavior.

## Success Criteria

- The reproduced Gemini and Qwen `{"orders": []}` response continues the
  backtest as a HOLD.
- Malformed or unsupported responses still fail closed.
- Existing non-empty pipeline decisions are unchanged.
- No frontend or deployment behavior changes in this pull request.
