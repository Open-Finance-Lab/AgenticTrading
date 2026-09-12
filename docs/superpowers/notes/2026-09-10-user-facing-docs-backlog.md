# User-facing docs backlog — opened 2026-09-10

Running list of **hosted / user-facing** documentation that the shipped code has
outrun. Nothing here is edited piecemeal: the whole list lands in one pass, on
one branch, when it is signed off. Add to it as work merges; do not fix items
individually.

Scope is `docs/source/lab/*.rst` (the Sphinx manual) and `README.md`. Agent-facing
notes (`docs/superpowers/`, `docs/architecture/`) are out of scope — they are
maintained inline with the code that changes them.

Every entry below was re-verified against source on 2026-09-10, on `main` at
`f5f803c3`. Where a suspected item turned out to be correct it is recorded under
**Checked, not stale** rather than deleted, so the same false lead is not chased
twice. Entries added after that date carry their own verification anchor
instead, because the code they describe is not on `main` yet - see entry 6.

---

## 1 + 1b. Wrong capital numbers - LANDED 2026-09-10 (zero-capital PR)

Both entries are **fixed and out of this backlog**. They were the only items the
zero-capital change itself made wrong, so they shipped on that branch rather than
waiting for the batch - a batch that landed the ceiling fix alone would have left
a freshly-false floor behind it.

- `docs/source/lab/getting_started.rst` - "Minimum **$1**, maximum **$10,000**"
  is now "Minimum **$0**, maximum **$3,000**", with the ceiling tied to the Paper
  Trading cap it matches (`MAX_AGENT_CASH_ALLOCATION`) and a paragraph separating
  a typed `0` (honoured; a real, flat, 0.00% run) from an *empty* field ("not
  configured"; mirrors the paper sleeve, floor $1,000). The `$0`-sleeve paragraph
  further down now states that asymmetry as the deliberate choice it is.
- `docs/api/agent-environment-protocol-v1.md` - the `10000` cap is now the real
  `0`-`3000` range, plus a paragraph saying `0` is legal rather than rejected and
  what a zero-capital run actually does.

The `$10,000` red herring is worth remembering for the rest of this batch: it is
a real number in the product (`DEFAULT_PORTFOLIO_EQUITY`, the account-level
budget) sitting two paragraphs away, which is almost certainly how it got here.

## 2. Undocumented surface: Credits & Billing

**Files:** nowhere. `grep -ri credits docs/source/lab/` returns one unrelated hit
("runs are credited to your agents" in `accounts.rst:9`).

**Shipped and user-visible:** a whole top-level page — `#creditsView`
(`app.html:1892`), reachable from the account menu ("Credits & Billing",
`app.html:263`) and from the `credits` nav route (`app.html:32,114`). It carries
three tabs — **API Keys**, **Credits**, **Activity** (`app.html:1908-1910`) — a
**Test Mode** badge, and Stripe-test-card purchase.

A signed-out visitor gets a "Sign in to manage Credits" empty state
(`app.html:1913`), so this is an account-gated feature that `accounts.rst`
should introduce and a new page should cover.

**Must state plainly:** purchased Credits **buy nothing yet**. They land in
`credit_ledger_entries`, while the metered path spends `user_entitlements.credits`,
and no code path connects the two — pinned by
`test_balance_does_not_claim_credits_are_spendable`. Documentation that implies a
purchased balance is spendable would contradict a test that exists specifically to
stop that claim.

## 3. Undocumented surface: BYOK vs Platform Credits

**Files:** nowhere.

**Shipped:** a billing-mode choice on the run path — `data-billing-mode="byok"`
and `data-billing-mode="platform_credits"` (`app.html:438,444`), with the
"Use ATL Credits" button at `app.html:447`. The Credits page's own footnote
states the rule: *"BYOK runs use your provider account and do not deduct ATL
Credits"* (`app.html:2068`).

This is the first thing a cost-conscious user needs to understand and it exists
only as one line of footnote text inside the app. It belongs in
`getting_started.rst` (the run-a-backtest flow) and in the new Credits page.

## 4. Incomplete: the Asset Universe is now a builder, not a picker

**File:** `docs/source/lab/getting_started.rst:9`
**Says:** "In the dialog set the **Period** and **Asset Universe**, then click
**Run Backtest**."

**Actual:** the Asset Universe control is a two-tab component (`app.html:377-415`)
— **Built-in** vs **Custom**, where Custom is a chip-based universe builder
("Add companies to create your custom universe", `app.html:415`), plus a
collapsible roster preview (`#backtestUniversePreview`, `:395`). There is also a
separate registered **A-share universe** panel (`#ifindAshareUniverse`, `:362`)
with its own selector.

Shipped across #441/#442/#443/#446. One sentence describing a dropdown no longer
covers it.

## 5. Stale: the Leaderboard is two boards now

**File:** `docs/source/lab/operating_modes.rst:13-14`
**Says:** one board — "Standardized comparison of agents against buy-and-hold and
index baselines over a fixed window."

**Actual:** since #352 the dashboard serves the **Competition Leaderboard** (the
fixed historical window this sentence describes) *and* the **Live Trading
Leaderboard**, which renders a **Season 0 preview** — real Competition curves under
season chrome, with a banner saying nothing has advanced, because the season
engine does not exist yet.

The existing sentence is not wrong about the Competition board; it is silent about
the second one, which is the one a visitor sees under a "Live" label and will
misread as live results. The H6 sentence that follows it is still accurate.

**Careful:** do not write copy implying either board takes user entries. Neither
does — every row is built from the curated roster in
`dashboard/config/leaderboard.json`, and `api/routers/leaderboard.py` exposes no
submission route.

## 6. Incomplete: the first-loop checklist on My Agents

**File:** `docs/source/lab/getting_started.rst:7-10` - the four numbered steps of
"Run a backtest in the dashboard".

**Shipped in #451** (anchors verified 2026-09-11 on `feat/user-feedback-batch-d`
at `90d44aae`, not yet on `main`): a **Get your first result** panel -
`#onboardingChecklist` (`app.html:989`), painted by `renderOnboardingChecklist()`
(`app.js:1957`) - sitting directly above the agent grid, with two steps: **Run
your first backtest** and **See your results**.

Step 1 of the manual sends the reader to **My Agents** and step 2 to an agent's
card. The panel is between them, and it is the first thing on that screen for
precisely the reader this page is written for, so the walkthrough now skips past
a visible element rather than naming it.

**Must state plainly - it is derived state, not a to-do list.** Every tick comes
from data the page already holds (`deriveOnboardingChecklist`, `app.js:1891`);
nothing is persisted. There is no dismiss control and no way to bring the panel
back: it shows only while the roster has loaded and **no** agent has a completed
run, and it goes away for good on the first finished backtest. Copy inviting the
reader to dismiss it, finish it later, or reopen it would describe a control that
does not exist.

**Careful:** do not write "create an agent" as one of its steps. Signup
provisions starter agents (`api/auth.py` -> `provision_starter_agents`) and the
client re-provisions them for guests, so a step keyed on agent existence would
arrive pre-ticked; it was left out for that reason, and documenting it back in
would teach a reader that the ticks mean nothing.

---

## Checked, not stale

- **`agentic-trading-lab.vercel.app` as the "Live app" URL**
  (`overview.rst:6`, `getting_started.rst:7`). Verified working: Vercel serves the
  static frontend and **proxies the API to Render** — `dashboard/frontend/vercel.json`
  rewrites `/api`, `/paper`, `/backtest`, `/runs`, `/config`, `/admin`, `/ticker`,
  `/health` and `/compare` to `https://agentictrading.onrender.com`, and its CSP
  allowlists that origin in `connect-src`. Both `/` and `/app` return 200 on both
  hosts. The Vercel URL is correct user-facing guidance; leave it.

- **"charts the agent against buy-and-hold and DJIA"** (`getting_started.rst:11-13`)
  — still matches the benchmark comparison shipped in #415.

- **Agent Supermarket naming** — `marketplace.rst`, `getting_started.rst` and
  `live_trading.rst` were all renamed on 2026-08-27 alongside #409. Current.

- **PR #447's own changes** (capital-field guard, chart height floor, board
  palette) — cosmetic and behavioural-guard only; they change no documented number
  or flow. Item 1 above predates #447 and is unrelated to it.

---

## Landing the batch

One branch, one PR, `docs:` prefix. Items 2 and 3 likely want a new
`docs/source/lab/credits.rst` added to the `lab/index.rst` toctree rather than
being wedged into `accounts.rst`. Items 1, 4, 5 and 6 are edits in place - and
4 and 6 touch the same four numbered steps in `getting_started.rst`, so write
them together rather than as two passes over one list.

Sphinx deps are optional and live in `requirements-sphinx.txt` — install those
before building to check the toctree.
