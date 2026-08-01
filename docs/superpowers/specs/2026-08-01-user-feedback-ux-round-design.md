# User-feedback UX round — design

**Date:** 2026-08-01
**Source:** real tester feedback on the live product (three items, verbatim intent preserved below)
**Status:** design approved; implementation plan written and reviewed (2026-08-01) — no
blockers, seven source-verified corrections folded back into this document and the plan

---

## Origin

A tester used the live product and reported three things. Translated:

1. **Creating a built-in agent looks broken for ~5 seconds.** Clicking *Create built-in
   agent* produced no visible change — no button label change, no spinner, no success
   confirmation. The agent was created correctly and the cash allocation was right. This is
   an interaction-feedback defect, not a creation defect. Asked for: lock the button, show
   `Creating…`, confirm success.
2. **A long backtest gives no sense of position.** At 3m05s the page still showed only
   `Backtesting` and an elapsed timer — no percentage, no estimate, no way to tell running
   from stuck, no cancel.
3. **A first-time visitor cannot quickly tell what the platform is for.** Products like
   Robinhood and Carry state early, with icons and short lines: what this is, what problem it
   solves, how it differs, how to start. The tester listed seven selling points the product
   has but does not say out loud.

## Audience (settled with the requester)

The reader we are writing the landing for: **has money to spare, is not deep into trading.**
They want to test a market idea at low cost and low friction, and eventually deploy real
capital. This audience definition drives every copy decision in Workstream C.

---

## What the code actually says

Investigated before designing. Three findings changed the shape of the work.

### Finding 1 — the backtest progress data already exists; only the surface drops it

The backend has emitted step-level progress the whole time:

```python
# dashboard/backend/api/routers/backtests.py:1272
pct = min(99, round(100 * step / total))
message = f"Backtest running… step {step}/{total} ({pct}%)"
```

fed by a progress file the engine rewrites on every step
(`domain/backtesting/engine.py:287`, `_publish_live_progress`). The frontend already has a
full percentage panel (`app.js:4607`, markup `app.html:1080`).

That panel only paints when `viewingLive` is true **and** it lives under the Backtest tab.
After launching, the user lands on **My Agents**, where `renderAgentRunningBody`
(`app.js:782`) renders an intentionally indeterminate bar. The user watched the one surface
that discards the data.

**This reverses a documented non-goal, and the reason matters.**
`2026-07-29-my-agents-capital-instruction-running-design.md` states:

> The running indicator is a pulsing dot plus an **indeterminate** animated bar —
> deliberately not a percentage, since no honest completion estimate exists.

The premise is false: `step`/`total_steps` were already being written per step when that was
written. We are correcting a factual error, not relitigating a taste decision. Recorded here
so the next reader does not revert it back.

Carried forward from that spec: the card's animation must keep its
`@media (prefers-reduced-motion: reduce)` fallbacks. The determinate bar inherits that rule.

### Finding 2 — cancel does not exist anywhere, and is expensive

`run_backtest_background` uses blocking `subprocess.run()` (`backtests.py:477`). Cancel
requires `Popen` + a stored handle + `terminate()` + a new route + a `cancelled ≠ error`
status path — in code that already carries a documented completion-detection race (PR #163).
`backtest_status` is a single process-global dict: one backtest server-wide.

**Decision: deferred to a tracked issue.** Not in this round.

### Finding 3 — one of the seven selling points is not a shippable claim

| Selling point | Reality | Verdict |
|---|---|---|
| Create agents in natural language | `agentEditorSimpleInstruction` — *"Tell the agent how to trade in plain language"* (`app.html:972`) | real; **absent from landing** |
| No trading program to write | same path, no code required | real; absent from landing |
| Choose different LLMs | `builtinAgentModel` select; 7 LLM entries on the board | real; absent from landing |
| Historical backtests | core engine | real; landing covers (`02 — Test`) |
| **Paper trading** | **No order-submission route exists.** `AlpacaPaperTradingClient` has **no order-submission method**. `PaperTradingSession.add_trade` has **zero production callers**. `execution/paper_backend.py` raises `NotImplementedError`. | **not a shippable claim** |
| Compare agent performance | leaderboard + baselines + H6 guard | real; landing covers (`03 — Race`) |
| Connect External Agents | `/api/v1` protocol + PyPI SDK + `/api/v2` | real; **absent from landing** |

**Say it as "no order-submission route exists", not as "`/paper/*` is read-only".** The
conclusion is right; that particular shorthand is not. `POST /paper/start-session`
(`api/routers/paper_trading.py:253`) *does* write — a run row via `db.insert_run(...)`. It
places no order, reads only `client.get_account()`, and no frontend code calls it, so nothing
about the verdict changes. But a PR body that says "read-only" hands a reviewer a
thirty-second refutation of the one claim this workstream is built on. Use the wording that
survives the grep.

The published docs already say the same thing —
`docs/source/lab/operating_modes.rst:8`: *"Monitoring only for now — an agent cannot yet
trade this account, so **Run Paper Trading** on an agent card is disabled."* Two independent
sources agree. Landing copy must not contradict them.

Real-capital execution **does** exist (`execution/robinhood_live_service.py` — risk gates,
per-order notional cap, idempotency, audit log) but is disarmed by default
(`ROBINHOOD_EXECUTE` defaults to `false`). Not claimed in this round.

---

## Decisions

Settled with the requester before design; recorded so the plan does not relitigate them.

| Question | Decision |
|---|---|
| How far does the landing change go? | Hero (`Talk to Agents` / `Test Trading Ideas`) **stays frozen**. Everything below it is reworked. |
| What does the funnel promise as its endpoint? | **Proven-on-history.** Describe → prove → rank. Live deployment named once as what's next. Paper trading not claimed. |
| Cancel a running backtest? | **Deferred to an issue.** Progress, ETA and staleness ship now. |
| Success confirmation style for agent creation? | A minimal toast. Not `alert()` — a blocking modal for a *success* is worse than the silence it replaces. |
| Ship as one PR or several? | Three independent PRs (A, B, C). |

## Non-goals

- Cancelling a running backtest (deferred; see Finding 2).
- Building agent-driven paper trading. `execution/paper_backend.py` stays a stub.
- Any change to the Hero section or the `Talk → Test → Race` section *structure*. Workstream C
  changes copy and adds one band; it does not reorder or delete sections.
- Reducing agent-creation latency as a committed deliverable (one investigation item only —
  see Workstream A).
- The C+D funnel pivot (on-site NL intake, lifecycle home). Separate workstream, needs
  sign-off.
- **Any documentation update.** Docs are coordinated separately by the requester and are
  never edited by the implementation PRs, even where this work makes them stale. Staleness is
  recorded (C7, Open follow-ups) so it is visible rather than silent.

---

## Workstream A — create-agent interaction feedback

**Surface:** `/app`, frontend only. **File:** `dashboard/frontend/app.js`,
`app.html`, `styles.css`.

`submitCreateBuiltinAgent` (`app.js:1812`) already sets `submitBtn.disabled = true` at
`:1838`. Everything the tester missed is what is *not* there.

### A1. Pending button state

A small shared pair, `setButtonPending(btn, label)` / `restoreButton(btn)`:

- stash the original label in a `data-` key
- `disabled = true`, `aria-busy = "true"`
- swap text to `Creating…` with a CSS spinner glyph
- restore in `finally`

Shared rather than inline because the same gap exists on other async submits; wired to
create-built-in only in this round, so the diff stays honest about what was verified.

### A2. Confirm on POST resolution, not after the grid refresh

Current order (`app.js:1848-1850`): close modal → `applyActiveAgent` → `await loadAgents()`,
with the button restored in `finally` *after* the refresh. Reorder so the close and the
success confirmation fire on the POST result. `loadAgents()` continues behind the
confirmation. `finally` still restores the button; harmless once the modal is closed.

### A3. Success toast

No toast system exists in `/app` — `alert()` appears 18 times and is the current convention.
Add a minimal one:

- `.app-toast`, `role="status"`, `aria-live="polite"`, ~4s auto-dismiss
- ~30 lines JS + ~25 lines CSS
- **distinct from `.home-live-toast`** (`styles.css:6792`), which is the Home live-decision
  widget in the same shared stylesheet. Visually matched, semantically separate.
- respects `prefers-reduced-motion` for its enter/exit transition

Message on success: agent name + that it was created, e.g. `"Momentum Alpha" created`.

### A4. Locate the new agent

After `loadAgents()` resolves, scroll the new agent's card into view and flash it briefly.
Answers "did it work?" positionally, not just textually.

The card element itself carries no `data-agent-id` today — all ten occurrences in `app.js` are
*buttons inside* a card. Tag the card, and scope the lookup to `.agent-card[data-agent-id]`:
the unscoped attribute selector matches every child button too and would scroll six to eight
times per creation.

### A5. Latency — investigation only, not a deliverable

`create_agent` (`agents.py:129`) calls `get_or_create_portfolio` at `:176` after
`ensure_cash_for_new_agent` at `:153` has already touched the portfolio. Possibly one
redundant Neon round-trip. `ensure_cash_for_new_agent` only runs when `cash > 0`, so the
paths are not obviously equivalent — **investigate and report; change only if provably
safe.** Do not block the workstream on it.

### A6. Error handling

Unchanged in shape: the existing `errorEl` path stays. The pending state must be cleared on
the error path too — that is what `finally` is for, and the test asserts it.

### A7. Tests

Repo H8 source-guard style, at the **tests root** — `dashboard/backend/tests/test_frontend_*.py`
/ `test_my_agents_*_ui.py`. That is the dominant convention: thirteen frontend source-guards
live at the root against two under `tests/integrations/`, and the two nearest analogues to
this work (`test_my_agents_card_ui.py`, `test_frontend_bundle_integrity.py`) are both at the
root. `tests/integrations/` holds cross-surface wiring checks (Discord, docs commands), not
frontend markup guards.

1. the submit handler sets a pending label before `await`
2. a toast element exists in `app.html` with `role="status"` and `aria-live="polite"`
3. the success path invokes the toast helper
4. the error path restores the button (assert the `finally` restore is present)

---

## Workstream B — backtest progress on the surface the user is standing on

**Surfaces:** `dashboard/backend/api/routers/backtests.py`, `dashboard/frontend/app.js`,
`styles.css`.

### B1. Backend — one new datum

`_read_backtest_progress` (`backtests.py:324`) currently returns the parsed payload. Add the
progress file's modification time as `progress_updated_at` (epoch seconds) so the client can
compute staleness. Everything else already ships.

`stat()` and `read_text()` are separate syscalls; a file rewritten between them yields an
mtime slightly older than the payload. Harmless at a 120s staleness threshold — note it, do
not engineer around it.

Existing failure behaviour is preserved: an unreadable or malformed progress file returns
`None` and the status payload simply omits `progress`, exactly as today.

### B2. Card — determinate when known, indeterminate when not

`renderAgentRunningBody` (`app.js:782`):

- when `step` and `total_steps` are known → determinate bar + `step N/M` + `%`
- when they are not (run start, file not yet written) → today's indeterminate bar
- rewrite the stale comment at `:775`; state that the estimate is now honest and why

The fallback is not a nicety: the progress file does not exist for the first moments of every
run, so the indeterminate state is a normal state, not an error state.

### B3. In-place patching

`refreshRunningAgentCards()` (`app.js:3356`) already patches the elapsed timer without
re-rendering the grid. Extend the same mechanism to step, percentage and ETA. Do not
introduce a second update path.

**One consequence to close, not to accept.** The polled progress is a single module-level
object (correct — `backtest_status` is one process-global, so one backtest runs server-wide)
merged into *every* entry of the `sessionStorage` running map. Today a stranded entry costs
only a wrong elapsed timer; once it carries step/percent/ETA, a stranded entry renders the
*next* run's numbers. The finished branch of the poller already clears the map
(`app.js:4907`); **the 10-minute timeout branch (`:4942`) does not**, and that is the leak —
agent A times out, agent B starts, A's card shows B's progress until `getAgentBacktestRunning`
expires it at 600s. Clear on timeout too, following the convention the finished branch set.
The remaining window — a tab closed mid-run and reopened, no poller alive to clear — is
bounded by the same 600s expiry and does not justify per-agent progress keying.

### B4. ETA

`remaining = (elapsed / step) × (total − step)`.

- suppressed until `step >= 3` — the first estimates are wild
- rendered coarse: `~2m left`, `<1m left`
- never rendered when `step` or `total_steps` is missing

A precise-looking ETA that jumps around reads as broken; a coarse one that drifts reads as an
estimate.

### B5. Staleness — answering "is it stuck?"

If `now − progress_updated_at > 120s`, show: **"No progress for {N} — long model steps can do
this."**, where `{N}` is the *actual* stale interval rendered coarsely (`2m`, `5m`, …), not
the literal threshold. A message frozen at "2m" while the real gap grows to ten would be
worse than no message.

Neutral by design. We know the file is stale; we do not know the run is stuck, and an LLM
pipeline step can legitimately take minutes. Claiming "stuck" would be the same class of
error as the fabricated Performance Drivers card.

### B6. Surface consistency

Mirror the same numbers into the Backtest-tab panel via `updateBacktestRunProgress`
(`app.js:4607`) so the two surfaces never disagree. This changes the numbers the panel shows,
not when it shows them.

The gate is on the call *site*, not the function: of the seven callers, four sit behind a
`viewingLive` / `isViewingLiveBacktest(...)` check (`:4891`, `:4917`, `:4926`, `:4947`) and
three do not (`:4806`, `:4831`, `:5491`). Only the live-poll site (`:4891`) gains the new
fields; the rest are launch, error, completion and timeout paths where an ETA is noise, and
they stay correct unedited because every added parameter defaults to `null`.

### B7. Deferred — cancel

**Filed as [issue #273](https://github.com/Open-Finance-Lab/AgenticTrading/issues/273).** Not
in this round. Covers `POST /backtest/cancel`, `subprocess.run` → `Popen` + stored handle +
`terminate()`, `cancelled` as a status distinct from `error`, preservation of the existing
timeout path, and the PR #163 completion-detection race as the hazard to design around.

### B8. Tests

1. backend: status payload carries `progress_updated_at` when a progress file exists; omits
   `progress` entirely when the file is missing or malformed
2. pure-function unit tests for the ETA formatter (suppression below `step 3`, coarse
   buckets, missing-input handling) and the staleness formatter — both are pure, so TDD
   applies directly
3. source-guard: the card markup renders step/total when present and falls back otherwise
4. source-guard: the `prefers-reduced-motion` fallback still covers the bar

---

## Workstream C — landing below-fold rework

**Source:** `dashboard/landing/src/**`. **Shipped artifact:**
`dashboard/frontend/index.html` + `dashboard/frontend/assets/`.

### C1. Structure

```
[ Hero — FROZEN, untouched ]
─────────────────────────────
▸ NEW BAND: "Why you should care"
    problem statement
    01 Describe it   02 Prove it   03 Rank it
    secondary row: pick the model · bring your own agent · Discord
─────────────────────────────
[ 01 — Talk  ]  copy reframed, structure kept
[ 02 — Test  ]  copy touched lightly
[ 03 — Race  ]  unchanged
[ FooterCTA  ]
```

The new band reuses the existing `01/02/03` mono-label pattern from Talk/Test/Race so it
reads as one system rather than a bolted-on marketing block. It sits above them and names the
same three acts — the band is the scannable summary, the sections are the detail.

### C2. Copy — the band

Lead with the problem, not the features:

> **You have an idea about the market. Testing it properly is the expensive part.**
> Normally that means writing code, buying data, and waiting months to find out you were
> wrong. Here it costs one sentence and a few minutes.

Three steps, icon + heading + one line each:

| Step | Heading | Line |
|---|---|---|
| 01 | Describe it in plain English | No code, no formulas. Write how you want to trade the way you would explain it to a person. |
| 02 | Prove it on real market data | Real prices, real market hours, measured against buy-and-hold and the index — so you learn whether the idea was good, not whether it felt good. |
| 03 | See how it ranks | Same window, same rules as everyone else's agents. |

Secondary row, three compact items — this is where the remaining verified selling points go
without the section becoming a feature dump:

- **Pick the model** — same idea, different brains: Claude, GPT, Gemini, DeepSeek, Qwen.
  Verified against the live select at `app.html:781` (Claude Haiku 4.5, Claude Sonnet 4.6,
  GPT-5.5, Gemini 3.1 Pro Preview, DeepSeek V4 Pro, Qwen3.7 Plus). Naming families rather
  than versions keeps the copy from going stale on every model bump — but if a family is
  ever dropped from that select, this line goes stale silently.
- **Bring your own agent** — Python SDK and an API, if you would rather write the code.
- **Talk to it on Discord** — if you would rather just chat.

### C3. Copy — `01 — Talk` reframe

Currently Discord-first: heading *"Talk to agents on Discord"*, steps *"Join the server →
Talk to the agent → Get your backtest result"*, CTA into Discord.

For this audience, "join a Discord server" is the friction the section is supposed to remove.
The app has had on-site plain-English authoring since the agent editor shipped
(`app.html:972`). Reframe so the **on-site instruction field is the primary path** and
Discord is the alternative. Structure, `DiscordMock` visual, and the `#talk` anchor stay —
this is a copy and CTA-emphasis change.

The hidden `#landing-stats` anchor inside `Talk.tsx:10` is Hero's scroll target and **must
survive**. If the new band is inserted above Talk, the scroll target should move to the band
so the first scroll lands on the value proposition.

### C4. What must not appear

- **Paper trading**, in any form implying an agent trades an account. Contradicts the code
  and `operating_modes.rst:8`.
- **Any "deploy real capital" promise.** `ROBINHOOD_EXECUTE` defaults to `false`.
- Live deployment appears exactly once, labelled as what is next — not as a capability.

### C5. Build discipline — the real risk in this workstream

`dashboard/frontend/index.html` is **not** Vite output. It is a hand-patched artifact
carrying ~370 lines Vite cannot emit: the auth-gate script, `#landingAuthModal`,
`<style id="landing-auth-patch">`, and the delegated `[data-landing-auth]` handler. Copying
`dist/index.html` over it — the obvious refresh — **silently kills every landing CTA with no
console error and a passing page load.** Tracked as issue #225.

Procedure:

1. `cd dashboard/landing && npm ci && npm run build` (~90s + ~3s; node 22 / npm 11 verified
   present)
2. verify the emitted asset hash equals the committed filename — the build is
   byte-reproducible on this toolchain, so a mismatch means the source and bundle disagree
3. re-apply the auth patch by hand
4. run `dashboard/backend/tests/test_frontend_bundle_integrity.py` — 4 checks including one
   `data-landing-auth` per emitter and every `lib/cta.ts` label present in the bundle
5. any new CTA in the band needs its own `data-landing-auth` attribute, which moves the
   emitter count and must be reflected in the guard

### C6. Footer links

`FooterCTA.tsx` ships `Terms`, `Privacy` and `Documentation` as dead `#` anchors — flagged in
the 2026-07-25 UI audit and still open. `Documentation` points at the real docs site.
`Terms` and `Privacy` have no destination to point at, so they are **removed** rather than
left dead: a link that goes nowhere is worse than an absent one, and inventing placeholder
pages is out of scope.

Included here because C already pays the bundle-rebuild cost; on its own this would not
justify a rebuild.

### C7. Docs — followup, NOT part of the implementation

`docs/landing-narrative-copy.md` is the copy deck this work supersedes: its "Hero — **Frozen
— do not change**" note stays true, but its stated tone (*"No feature dumps"*) and the
Talk/Test/Race-only arc do not survive Workstream C.

**It is not updated by the implementation PR.** Doc updates are coordinated separately by the
requester. Recorded here, and surfaced as an explicit session followup, so the deck and the
shipped page do not drift silently.

### C8. Tests

1. bundle integrity guard passes (C5 step 4)
2. source-guard: the band exists, carries the three step headings, and contains no
   paper-trading or real-capital claim — a string-absence assertion, cheap and durable
3. source-guard: `#landing-stats` still resolves to an element that exists
4. source-guard: no `href="#"` remains in `FooterCTA.tsx` (C6)

---

## Sequencing

Three independent PRs, no shared state:

| PR | Scope | Risk |
|---|---|---|
| **A** | create-agent feedback (`app.js`, `app.html`, `styles.css`) | low — additive, guarded |
| **B** | backtest progress (`backtests.py`, `app.js`, `styles.css`) | low-medium — reverses a documented non-goal on a corrected premise |
| **C** | landing below-fold (`landing/src/**` + rebuilt bundle) | **highest** — the bundle refresh has a silent-failure mode |

A and B both touch `app.js` but in disjoint regions (`:1812` vs `:782`/`:3356`/`:4607`).
Order is not enforced; C is independent of both.

## Open follow-ups

- **Cancel a running backtest** — filed as
  [issue #273](https://github.com/Open-Finance-Lab/AgenticTrading/issues/273). Out of scope
  here; see B7.
- **Dead footer links** — now **in scope** as C6, not a follow-up.
- **`docs/landing-narrative-copy.md`** — superseded by Workstream C, but **deliberately not
  updated by the implementation**. The requester coordinates doc updates separately. See C7.
- **`docs/source/lab/*.rst`** — light scan found the paper-trading claims accurate. No action;
  recorded so the next reader does not re-scan.
