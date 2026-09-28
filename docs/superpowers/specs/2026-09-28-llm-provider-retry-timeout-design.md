# LLM provider timeouts and retries: one generation per attempt

> **Status: IMPLEMENTED** on `fix/llm-provider-timeout`. This is S2's
> no-regeneration half from `2026-09-23-llm-backtest-step-latency-design.md`.
> Code: `infrastructure/llm/execution/adapters/base.py` (timeouts, `RetryHint`
> mapping), `execution/service.py` (the retry loop), and the OpenAI/Anthropic
> adapters (`max_retries=0`). Where this document and the code disagree, the
> code is current.

## 1. Incident

On 2026-09-28, DeepSeek V4 on ATL Credits, run `agent_20260928_024706_cbda3555`,
recorded these reservation durations: 112s, 100s, 58s, 176s, then 185s and
`provider_timeout`. That ended the run under `fail_closed`.

The cause is three layers stacking:

1. `build_safe_http_client` used `httpx.Timeout(60.0, connect=8.0)`. A
   non-streaming provider sends no byte until the completion is done, so the
   read timeout acts as a whole-generation deadline. 60s sat just under what
   DeepSeek V4 needs to fill a 2000-token ceiling at its slowest healthy rate,
   about 33 tok/s.
2. openai 1.101 adopts an `http_client`'s timeout when it differs from httpx's
   default. It then retries under `max_retries=2` inside a single
   `except httpx.TimeoutException` (`_base_client.py:963-1000`), which treats a
   read timeout exactly like a connect error. anthropic 0.95 has the same
   Stainless loop.
3. Neither SDK sends an idempotency key (`_idempotency_header = None`), so each
   replay is a fresh, billed generation. All replays happen inside one ATL
   reservation, and only the last one is settled.

Reading the numbers with that in mind:
- 100s = a 60s generation that was abandoned, plus a regenerated one.
- 176s = 60 + 60 + 56.
- 185s = 3 × 60s + backoff.

The "only bound is the 60s read timeout" statement in `backtests.py` and
CLAUDE.md was really about 185s and three generations.

A second defect hid the first: `_execute_with_platform_failover` raised the
**last** candidate's error. So the CommonStack timeout, followed by quota-dead
OpenRouter's 402, surfaced as `provider_quota_exhausted`.

## 2. Invariant

**One reservation row covers at most one physical provider attempt.** An attempt
that might generate never shares a row with another attempt.

## 3. Policy

| Failure | Same-provider repeat | Why |
|---|---|---|
| DNS, connect, TLS, write or pool (`RetryHint.PRE_SEND`) | yes, at any elapsed time | nothing reached the provider |
| 408/409/429/5xx, `x-should-retry: true`, or a dropped connection (`REJECTED`) | yes, only if it failed within **15s** | a fast refusal generated nothing; CommonStack's 2026-09-27 500s came ~24s in, *after* generation |
| read timeout, or a timeout of unknown phase (`NONE`) | **never** | a whole generation was in flight and would be billed again |
| other 4xx, 401/403, 402, quota bodies | never | lane state, or the caller's error |
| RESPONSE_INVALID, USAGE_UNAVAILABLE, BILLING_FAILED, ACCOUNT_RESTRICTED | never | not provider availability |

- **Limits.** At most 2 repeats per candidate. Backoff is 4s then 12s. A numeric
  `Retry-After` / `retry-after-ms` is honoured up to 30s; above that the call
  fails over instead of waiting.
- **Counter.** Every attempt, repeat or failover, takes the next `attempt_index`
  of its call. The reservation key is `(user, run, call, attempt)`, and reusing
  an index returns the already-released row, which raises BILLING_FAILED.
  Pinned by `test_reused_attempt_index_fails_billing`.
- **Hints change no categories.** An error built without a hint means "never
  repeat". That covers every scripted test adapter and Gemini's status branch,
  so their behaviour is unchanged.
- **Error raised.** When every candidate fails, the error raised is the first
  timeout or unavailable in attempt order.

## 4. Constants

| Name | Value | Derivation |
|---|---|---|
| `LLM_PROVIDER_READ_TIMEOUT_SECONDS` | 180 (30–600) | About the old per-candidate worst case (185s), so a hang never waits longer. It covers the 4096-token recovery ceiling down to ~23 tok/s. |
| connect / write / pool | 8 / 60 / 60 | Unchanged. |
| `SDK_MAX_RETRIES` | 0 | See §1. |
| `MAX_SAME_PROVIDER_RETRIES` | 2 | Keeps the SDK's old attempt count, but only for the cheap class. |
| `FAST_FAILURE_SECONDS` | 15 | Gateway refusals arrive in under 2s. The fastest DeepSeek completion seen is ~24s. |
| `SAME_PROVIDER_BACKOFF_SECONDS` | (4, 12) | Paid only on failure paths; 16s is ~0.5% of the 3000s budget. |
| `RETRY_AFTER_CAP_SECONDS` | 30 | Beyond this, failover is the better use of the time. |

**A flat deadline, not one derived from `max_tokens`.** The alternative was
`clamp(30 + max_tokens / 20, 60, 300)`. It would stretch a hung 4096-token
recovery call to 235s. S3 must revisit this if the first-attempt ceiling rises
above ~4096.

## 5. Budget

Per logical call, with CommonStack live and OpenRouter dead at ~1s:

| Shape | Before | After |
|---|---|---|
| Healthy 60–180s generation | Regenerated, up to 185s and 3 generations; may fail | 1 generation, completes |
| Hang | ~185s and 3 generations; surfaced as QUOTA | 180s and 1 generation; surfaces as TIMEOUT |
| Slow 500 at ~24s | ~73s and 3 generations | ~24s and 1 generation, then failover |
| Fast 502, then success | ~1s backoff | 4s backoff |

Two live candidates double each row.

## 6. Observability

Each failed timeout/unavailable attempt prints one flushed line:

```
ERROR: llm.provider_attempt_failed run=… call=… attempt=… provider=… model=… billing=… category=… phase=<connect|read|write|pool|-> status=<int|-> hint=<pre_send|rejected|none> elapsed_s=… read_timeout_s=… next=<retry|provider|none>
```

- **Relay.** `_drain_stream` relays it live because it starts with
  `ERROR: llm.`, which matters on the timeout-kill path that never dumps the
  capture.
- **No exception text.** Every value is a validated identifier or a number, so
  `_redact_credentials` has nothing to rewrite.
- **Reading it:**
  - `phase=read elapsed_s≈T`: a hang, or T is too short.
  - `status=504` with `elapsed_s` > 15: an intermediary is cutting requests
    below T. This is the main unknown, since the client never waited past 60s
    before.

## 7. Post-deploy check (read-only)

On the next DeepSeek ATL Credits run:
- In `credit_llm_reservations`, single attempts land in the old 57–66s and
  100–125s dead zones.
- `failure_reason='provider_timeout'` rows cluster near 180s.
- `attempt_index=0` is still `commonstack`.
- The Render logs API with `text=llm.provider_attempt_failed` shows how
  CommonStack behaves past 60s.

## 8. Out of scope

- **Streaming.** CommonStack does not document `stream_options.include_usage`,
  and settlement fails closed on missing usage, so a wrong guess is a total
  outage. CommonStack also bills cancelled generations, so streaming would not
  recover that spend. Needs a live, billed probe per provider first.
- **Per-run provider cooldown.** Moot while OpenRouter is dead (#523).
- **`portfolio_manager` fail_closed and the strike budget.** Draft #543. This
  change only makes the categories it absorbs truthful.
- **`PIPELINE_SECONDS_PER_LLM_CALL` and the 3600s budget.** S4. Only their
  justifying comments changed here.
- **httpcore `retries=`.** No log line and no reservation, and it would
  multiply with the service retry. `PinnedNetworkBackend` already walks every
  resolved IP.
- **The billed-empty-reply gap.** A RESPONSE_INVALID reservation is released
  at zero debit while the provider bills about 2000 tokens. Fixing it needs
  credits-store twin changes and a who-pays decision.
- **Legacy validator adapters, the chat/algo Anthropic clients (still 600s × 3),
  and the AI Hedge Fund runtime.** Not on the billed path.
- **`EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS`.** The 60s protocol step deadline
  is a different timer.
