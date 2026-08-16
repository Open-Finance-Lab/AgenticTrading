# Participatory Competition — Phase 0 (Gate) + Phase 1 (Evidence) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> # ⚠ Status: reconciled 2026-08-15 — DO NOT EXECUTE TASKS 5, 7 OR 8 AS WRITTEN
>
> This plan was written on 2026-08-09. Six days later PR #352 (`7db504d`) and
> PR #357 (`60fa01f`) shipped a leaderboard redesign and a leaderboard-first
> landing page. Neither was written against this plan, and between them they
> delivered part of Phase 1 by other means — in one case in the **opposite**
> direction.
>
> **Nothing in Phase 0 has been implemented.** Verified 2026-08-15 against
> `origin/main` @ `88c7b8c`, and across every ref in the repo: no branch anywhere
> adds `strategy_prompt` to `llm_agent.py`.
>
> ## Task status
>
> | # | Task | Status | Evidence checked 2026-08-15 |
> |---|---|---|---|
> | 1 | `LLMAgentStrategy` forwards `strategy_prompt` | **DONE — on a branch, NOT on `main`** | Built 2026-08-16. `self.strategy_prompt` at `llm_agent.py:85`, loop extracted to `_run_decision_loop`. Lives on `worktree-phase0-instruction-gate`, **PR #366 open** — anything depending on it must wait for that merge |
> | 2 | Instruction-sensitivity probe script | **DONE — same branch/PR** | `dashboard/scripts/probe_instruction_sensitivity.py`, 567 lines. Gained `--initial-capital` and a capital-resolution guard that refuses to spend when one median share exceeds 1% of equity |
> | 3 | Run the probe, record the gate | ✅ **DONE — GATE PASSES** | Ran 2026-08-16 at `initial_capital=100000`. `aggressive_momentum` **+3.83%** vs `control_nonsense` **+0.33% / +0.13%** → signal **3.61pp** over a **0.20pp** noise floor, **18.1×**; all cleared H6. **DeepSeek V4 Pro is the pinned model.** Write-up: `docs/superpowers/probe-results/2026-08-09-instruction-sensitivity.md`. Spend $3.61 of the $4.97 sanction |
> | 4 | Six Open Track seed entries | **TODO — unblocked by the gate; read the caveat below** | `leaderboard.json` still holds exactly 12 entries (5 baselines + 7 Model); no `label: "Open Track"`, `authored_by` or `strategy_prompt` anywhere. Also **depends on Task 1, which is not on `main`** (PR #366) |
> | 5 | `landing/src/lib/leaderboard.ts` | **BLOCKED — premise withdrawn** | Needed only if the landing page fetches live data. #357 chose labelled sample data instead. Do not build until that decision is reopened |
> | 6 | `landing/src/lib/analytics.ts` | **TODO — still valid** | No `track()` import anywhere in `landing/src/`; `<Analytics />` is mount-only. Independent of the board-data decision |
> | 7 | `Race` renders the live board | **SUPERSEDED — and partly reversed** | See the task's own status note. Its guard test would now *delete* a guard #357 added deliberately |
> | 8 | Promote the board above the fold | **DONE, by different means — one loose end** | `BoardPreview` mounted at `Hero.tsx:144`. `Race` never moved and is still last. Loose end: `FooterCTA.tsx:10` still says `Talk → Test → Race` |
> | 9 | Social share card (`og:image`) | **TODO — valid, unchanged** | `index.html` still declares `twitter:card="summary_large_image"` with no image — the exact "worst of both" state the task describes |
> | 10 | Community→Agent Marketplace, Teams→Traders | **TODO — valid, line numbers moved** | `app.html:243` `Community`, `:1691` `<h2>Community</h2>`, `:1433` `Participating Teams`, `:1439` `Season hasn't started yet` |
> | 11 | Rebuild the shipped landing bundle | **MECHANICAL — only if landing source changes** | #357 already rebuilt it (`index-XgaRai2O.js`). Re-run only after touching `dashboard/landing/src/` |
> | 12 | Full verification and PR | **Rewrite** — scope is now Tasks 1–4, 6, 9, 10 | — |
>
> **What is actually left of Phase 1:** the seed field (1–4), funnel events (6),
> `og:image` (9), and the two renames (10). That is a materially smaller PR than
> the one this plan describes, and it no longer touches the landing page's data
> path at all.
>
> **The gate caveat (2026-08-16).** Phase 0 passed, but it measured
> instruction-vs-**control**, not instruction-vs-instruction: `contrarian_reversion`
> was cut to stay inside the spend sanction, so the leg shows that a real
> instruction separates from nonsense — not that two *plausible* instructions
> separate from each other. Task 4 seeds six Open Track entries that are all
> plausible, and Phase 2 ranks user entries against each other. **That ranking is
> the untested case.** Two further facts from the leg bear on it directly:
> `temperature=0` still leaves **0.20pp** of run-to-run variance, so margins
> inside ~0.2pp are not real and a re-run is not guaranteed to reproduce a rank;
> and Nemotron is **unmeasured, not failed** — its leg ran at the wrong capital
> base and was never redone, so nothing here licenses a claim about small models.
> About $1.40 of the sanction remains, enough for `contrarian_reversion` plus a
> no-`strategy_prompt` anchor.
>
> **Line numbers throughout this document are as of 2026-08-09** unless a task's
> status note gives a newer one. Re-verify before editing.
>
> ## Read these three before starting Phase 2
>
> - **`docs/superpowers/specs/2026-08-15-live-trading-leaderboard-ui.md`** — the
>   newer board-side spec. Authoritative over the companion design doc on the
>   boards' UI and payload contract; that doc stays authoritative on user entry.
> - **Issue #354** — the Live Trading season engine (C8), already filed.
> - **Issue #355** — two open design questions that **explicitly block this PR and
>   Phase 2**: whether the qualifier gate survives now that the practice board is
>   unranked, and what `instruction_sha256` config-freeze means for user-owned
>   editable entries.
>
> **Phase 0 (Tasks 1–3) is not blocked by #355.** It measures whether instructions
> move returns, which is upstream of every question in that issue and answers none
> of them. It can proceed the moment the spend is approved.

**Goal:** Prove that a trading instruction measurably changes a pinned LLM's backtest return, then ship a landing page whose hero is the real leaderboard showing most models losing to buy-and-hold, with a CTA inviting visitors to beat them.

> **Half of this goal is met and half changed shape (2026-08-15).** The board *is*
> the hero and the CTA *is* in place — #357 did that. But the landing hero shows
> labelled **illustrative** numbers, not the real board; the real board is on the
> `/app` home screen. The first clause — proving an instruction moves the return —
> is untouched and remains the whole point of Phase 0.

**Architecture:** Phase 0 adds one two-line capability to `LLMAgentStrategy` (`strategy_prompt` reaches the prompt builder) and uses it to run a 12-run sensitivity probe across two models. If the probe passes, the five surviving instructions become permanent house-authored seed entries in `leaderboard.json`, and Phase 1 wires the landing page's existing `Race` section to the live leaderboard API, promotes it above the fold, and instruments the funnel. **No new database tables, no new routes, no user-submitted entries.**

**Tech Stack:** Python 3.13 / FastAPI / pytest (backend); vanilla JS + Chart.js (`/app`); Vite + React + recharts + Tailwind (`dashboard/landing`, built into `dashboard/frontend`); SQLite + optional Neon Postgres; Vercel (frontend) + Render (backend).

**Spec:** `docs/superpowers/specs/2026-08-09-participatory-competition-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Never `git add -A` in this repo.** A bare backend import runs lazy `CREATE TABLE`/`ALTER` statements against the committed prod seed database `dashboard/storage/data/backtest.db`. Always stage explicit paths. Before every commit, run `git status --short` and confirm `dashboard/storage/data/` is absent.
- **Run everything from the repo root.** The backend is the `dashboard.backend` package; `python dashboard/backend/app.py` does not work.
- **Route-contract freeze golden sets must be updated in the same commit as any new route.** The sets are `EXPECTED_LEADERBOARD_ROUTES` (`dashboard/backend/tests/test_router_move.py:112`) and `EXPECTED_FULL_CONTRACT` (`dashboard/backend/tests/test_app_composition.py:56`). They drift independently. *Phase 1 adds no routes, so this should not fire — if it does, you have added a route this plan did not call for.*
- **Postgres twin parity:** `test_store_twin_parity.py` compares column **names** only and cannot see f-string DDL. `NOT NULL` and type divergence are invisible to it and must be checked by hand. *Phase 1 adds no tables.*
- **Any test importing an optional dep (`vnpy`, `discord`) must `importorskip`.** An unguarded import raises during *collection*, and a collection error aborts the whole pytest session — 0 tests run, and the deploy hook that gates on backend tests never fires.
- **Frontend cache-busters:** every touched `/app` asset needs its own `?v=` bump in `dashboard/frontend/app.html` — they are versioned independently, not globally. **The landing bundle needs none**: Vite asset names are content-hashed, and the hash is the cache bust.
- **Merging to Open-Finance-Lab `main` auto-deploys prod.** A CI job hits the Render deploy hook once backend tests pass on `main`.
- ~~**PR #326 is open and touches `dashboard/frontend/js/leaderboard.js`.** Rebase onto it before starting Phase 1 frontend work; do not resolve conflicts by discarding its hover-gate fixes.~~ **Void — #326 merged 2026-08-09.** The live hazard is now different: `js/leaderboard.js` went 1,037 → 1,600 lines across #326, #352 and #357, so every line reference into it in this plan is stale.
- **Cache-buster collisions are the recurring merge hazard on this repo.** Two PRs bumping the same `?v=` produce a conflict that resolves silently to the *lower* number if taken carelessly. Resolve upward, always.
- **`OPENROUTER_API_KEY` must be set locally** for Phase 0. Nemotron and DeepSeek are reached through OpenRouter, which is never auto-selected (`providers/__init__.py:11`). Unset, every probe run silently falls back to rule-based and the probe measures nothing.
- **Copy rule:** user-facing text says **"instruction"**, never "strategy" (`domain/strategies/` is an existing product noun) and never "prompt". Entrants are **"traders"**, never "teams".
- **Never claim the market is easy to beat.** Headline numbers are stated against the house instruction's curve; beating the market is a separate, scarce badge.

---

## File Structure

**Phase 0**

| File | Responsibility |
|---|---|
| `dashboard/backend/domain/leaderboard/strategies/llm_agent.py` (modify) | Accept and forward `strategy_prompt` |
| `dashboard/backend/tests/domain/leaderboard/test_llm_agent_instruction.py` (create) | Guard that the instruction reaches the call and that omission preserves today's behaviour |
| `dashboard/scripts/probe_instruction_sensitivity.py` (create) | Standalone probe: N instructions × M models, prints a spread table. Builds configs in memory so the probe never pollutes `leaderboard.json` |
| `docs/superpowers/probe-results/2026-08-09-instruction-sensitivity.md` (create) | The recorded gate decision |

**Phase 1**

| File | Responsibility |
|---|---|
| `dashboard/config/leaderboard.json` (modify) | Six new house-authored Open Track entries carrying `strategy_prompt` |
| `dashboard/backend/tests/domain/leaderboard/test_seed_entries.py` (create) | Guard seed entries are labelled, pinned to Nemotron, and carry instructions |
| `dashboard/landing/src/lib/leaderboard.ts` (create) | Typed fetch + shaping of `/api/v1/leaderboard` for recharts. Sole owner of the API contract on the landing side |
| `dashboard/landing/src/lib/analytics.ts` (create) | Thin `trackEvent()` wrapper over `@vercel/analytics`, with the event-name union |
| `dashboard/landing/src/components/home/Race.tsx` (modify) | Render live data; own its loading/error/empty states |
| `dashboard/landing/src/pages/landing-page.tsx` (modify) | Promote `Race` above `WhyCare` |
| `dashboard/landing/src/components/home/Talk.tsx`, `Test.tsx`, `FooterCTA.tsx` (modify) | Storyline renumber |
| `dashboard/frontend/index.html` (modify) | `og:image`; rebuilt bundle asset hashes; auth layer preserved verbatim |
| `dashboard/frontend/images/og-card.png` (create) | Static 1200×630 share image |
| `dashboard/frontend/app.html` (modify) | Community→Agent Marketplace; Teams→Traders |
| `dashboard/backend/tests/test_frontend_marketplace_placement.py` (modify) | Follow the rename |
| `dashboard/backend/tests/test_landing_race_copy.py` (create) | Guard the CTA sentence and the absence of "Illustrative" |

---

# PHASE 0 — The Gate

**This is not a PR. It is a decision.** The entire design rests on an unverified
assumption: that better instructions produce better returns on a 30B nano model.
If instruction quality does not move the return, the leaderboard ranks noise and
Phase 2 must not be built.

> **Premise correction, 2026-08-15.** This paragraph originally justified the
> gate by saying the model's prompt "is dominated by an unconditional
> `SAFE_TRADING_PROMPT` (`validator.py:545-664`)". **That is not what the code
> does.** A caller-supplied instruction *replaces* the `SAFE_TRADING_PROMPT`
> strategy body; only a fixed execution-contract scaffold is concatenated after
> it — `CUSTOM_STRATEGY_OUTPUT_CONTRACT` (`validator.py:667-711`), joined by
> `create_custom_prompt` (`:730`), in place since `edc186b`, 2026-06-26. The
> instruction owns the whole strategy slot.
>
> The gate still matters — the open question is whether a nano model *acts* on
> instruction content — but two things change in how you read the result:
>
> - **A flat spread is more damning than this plan assumed.** There is no
>   prompt-dilution fix to try next; "the model ignores the instruction" is what
>   is left.
> - **The control is the load-bearing measurement**, not the spread. A nonsense
>   control still yields a valid run, so a control landing mid-pack means the
>   model responds to *having* a strategy body rather than to its content — which
>   passes a naive spread check while the board ranks noise. Task 3's gate table
>   already encodes this; do not relax it.

Task 1's code ships in Phase 1 regardless. Tasks 2–3 are throwaway measurement.

---

### Task 1: `LLMAgentStrategy` accepts an instruction (change C1)

**Files:**
- Modify: `dashboard/backend/domain/leaderboard/strategies/llm_agent.py:45-89` and `:176-182`
- Test: `dashboard/backend/tests/domain/leaderboard/test_llm_agent_instruction.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `LLMAgentStrategy(config)` now reads `config["strategy_prompt"]` into `self.strategy_prompt: str | None` (empty/whitespace-only → `None`) and passes it as the `strategy_prompt=` keyword to `PortfolioManager.make_trading_decision_with_llm`. Every later task and both later phases depend on this exact attribute name and this exact keyword.

**Why this is two lines:** `make_trading_decision_with_llm` already declares
`strategy_prompt: str = None` (`portfolio_manager.py:233` — **`:245` as of
2026-08-15**) and already threads it to
`create_prompt(custom_prompt=strategy_prompt)` (`:453` — **`:462-465`**). The
house path simply never passes it, so `custom_prompt` is always `None`. No
downstream signature changes.

> **Re-verified 2026-08-15 and still true.** The wiring is intact, the method is
> still one method with one `if pipeline:` branch (`:440`), and `llm_agent.py`
> still passes neither argument. `self.temperature` remains at `llm_agent.py:81`,
> exactly where Step 3 says to insert. This task is unaffected by #352/#357 and
> is the one piece of Phase 0/1 that can be executed exactly as written.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/domain/leaderboard/test_llm_agent_instruction.py`:

```python
"""C1: an Open Track entry's instruction must reach the shared prompt builder.

`make_trading_decision_with_llm` already accepts `strategy_prompt` and threads it
into `create_prompt(custom_prompt=...)`. The house path never passed it, so every
leaderboard entry ran the bare SAFE_TRADING_PROMPT. These guard that the wire is
connected and, just as importantly, that omitting the key preserves today's
behaviour for the seven published Model Track entries.
"""

import pytest

from dashboard.backend.domain.leaderboard.strategies.llm_agent import LLMAgentStrategy

BASE_CONFIG = {
    "strategy": "llm_agent",
    "model_id": "nvidia/nemotron-3-nano-30b-a3b",
    "integration": "openrouter",
    "temperature": 0,
    "reasoning_effort": "none",
    "mode": "safe_trading",
    "symbols": [],
}


def test_instruction_is_read_from_config():
    strategy = LLMAgentStrategy({**BASE_CONFIG, "strategy_prompt": "Buy the dip."})
    assert strategy.strategy_prompt == "Buy the dip."


def test_missing_instruction_is_none_not_empty_string():
    """The published Model Track entries carry no `strategy_prompt` key.

    `None` and `""` are NOT interchangeable downstream: `create_prompt` branches on
    truthiness, so an empty string would take the same branch as None today but
    silently diverge if that branch is ever tightened to `is not None`.
    """
    strategy = LLMAgentStrategy(dict(BASE_CONFIG))
    assert strategy.strategy_prompt is None


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_blank_instruction_collapses_to_none(blank):
    strategy = LLMAgentStrategy({**BASE_CONFIG, "strategy_prompt": blank})
    assert strategy.strategy_prompt is None


def test_instruction_is_stripped():
    strategy = LLMAgentStrategy({**BASE_CONFIG, "strategy_prompt": "  Hold cash.  "})
    assert strategy.strategy_prompt == "Hold cash."


def test_instruction_is_passed_to_the_decision_call(monkeypatch):
    """The attribute existing is not the contract — reaching the call site is."""
    import dashboard.backend.domain.leaderboard.strategies.llm_agent as mod

    seen = {}

    class FakeManager:
        cash = 10_000.0
        trades = []
        equity_history = [{"equity": 10_000.0}]
        llm_calls = 0
        llm_decisions = 0
        input_tokens = 0
        output_tokens = 0

        def __init__(self, **kwargs):
            pass

        def get_portfolio_state(self, market_data, price_cache, ts):
            return {}

        def make_trading_decision_with_llm(self, state, client, **kwargs):
            seen.update(kwargs)
            return {"actions": []}

        def execute_actions(self, actions, market_data, ts):
            pass

        def update_equity(self, market_data, price_cache, ts):
            pass

        def get_equity_curve(self):
            return [{"timestamp": "2026-04-15T14:00:00", "equity": 10_000.0}]

    monkeypatch.setattr(mod, "PortfolioManager", FakeManager)

    strategy = LLMAgentStrategy({**BASE_CONFIG, "strategy_prompt": "Rotate weekly."})
    strategy._run_decision_loop(
        client=object(),
        timestamps=["2026-04-15T14:00:00"],
        symbols=["AAPL"],
        data={},
        price_cache={},
        initial_capital=10_000.0,
    )

    assert seen.get("strategy_prompt") == "Rotate weekly."
```

> **Note for the implementer:** the last test calls `_run_decision_loop`, which
> does **not exist yet**. Extracting the loop currently inlined in `run()`
> (`llm_agent.py:145-200`) into a named method is part of Step 3 — the loop is
> otherwise unreachable without live market data and an LLM client, and an
> untestable call site is exactly where a silently-dropped keyword hides. Keep
> the extraction pure: move the code, change nothing else.

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest dashboard/backend/tests/domain/leaderboard/test_llm_agent_instruction.py -v
```

Expected: `test_instruction_is_read_from_config` FAILS with
`AttributeError: 'LLMAgentStrategy' object has no attribute 'strategy_prompt'`,
and `test_instruction_is_passed_to_the_decision_call` FAILS with
`AttributeError: ... has no attribute '_run_decision_loop'`.

- [ ] **Step 3: Implement**

In `llm_agent.py`, immediately after `self.temperature = temperature` (`:81`),
add:

```python
        # C1: the Open Track's competing variable. Absent on the seven published
        # Model Track entries, which must keep producing their existing curves —
        # so blank collapses to None rather than "".
        self.strategy_prompt = (self.config.get("strategy_prompt") or "").strip() or None
```

Then extract the decision loop from `run()` into a method, and add the keyword.
The loop currently at `:165-187` becomes:

```python
    def _run_decision_loop(
        self,
        client,
        timestamps,
        symbols,
        data,
        price_cache,
        initial_capital,
    ):
        """One decision per timestamp, executed against a fresh PortfolioManager.

        Extracted from run() so the strategy_prompt hand-off is reachable in a
        test without live bars or an LLM client.
        """
        profile = get_market_profile(ALPACA)
        manager = PortfolioManager(
            initial_capital=initial_capital,
            t_plus_one_enabled=profile.t_plus_one_enabled,
        )
        total = len(timestamps)

        for i, ts in enumerate(timestamps):
            market_data = {}
            for sym in symbols:
                df = data.get(sym)
                if df is not None and ts in df.index:
                    market_data[sym] = df.loc[ts]

            state = manager.get_portfolio_state(market_data, price_cache, ts)
            state["timestamp"] = ts

            if client is not None:
                decision = manager.make_trading_decision_with_llm(
                    state,
                    client,
                    mode=self.mode,
                    model=self.model_id or default_model_name(self.integration),
                    strategy_prompt=self.strategy_prompt,
                    temperature=self.temperature,
                )
            else:
                decision = manager.make_trading_decision(state)

            manager.execute_actions(decision.get("actions", []), market_data, ts)
            manager.update_equity(market_data, price_cache, ts)

            if (i + 1) % 25 == 0 or (i + 1) == total:
                equity = (
                    manager.equity_history[-1]["equity"]
                    if manager.equity_history
                    else initial_capital
                )
                print(f"      step {i + 1}/{total} · equity ${equity:,.0f} · calls {manager.llm_calls}")

        return manager
```

Replace the inlined loop in `run()` with a call to it, keeping the existing
post-loop bookkeeping (`curve`, `_num_trades`, `llm_calls`, `llm_decisions`,
`decision_steps`, token counters) reading from the returned `manager`.

- [ ] **Step 4: Run the new test**

```bash
pytest dashboard/backend/tests/domain/leaderboard/test_llm_agent_instruction.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Run the leaderboard suite to prove the extraction changed nothing**

```bash
pytest dashboard/backend/tests/domain/leaderboard/ dashboard/backend/tests/test_leaderboard_api.py -v
```

Expected: all pass. **A failure here means the extraction was not pure** — the
published Model Track curves depend on this loop.

- [ ] **Step 6: Commit**

```bash
git status --short   # confirm dashboard/storage/data/ is NOT listed
git add dashboard/backend/domain/leaderboard/strategies/llm_agent.py \
        dashboard/backend/tests/domain/leaderboard/test_llm_agent_instruction.py
git commit -m "feat(leaderboard): LLMAgentStrategy forwards strategy_prompt (C1)"
```

---

### Task 2: The instruction-sensitivity probe script

**Files:**
- Create: `dashboard/scripts/probe_instruction_sensitivity.py`

**Interfaces:**
- Consumes: `LLMAgentStrategy.strategy_prompt` from Task 1.
- Produces: `PROBE_INSTRUCTIONS: list[tuple[str, str]]` — `(slug, instruction_text)`. Task 4 imports this exact name to build the seed config entries, so the five non-control instructions must never be edited in one place only.

**Why a script and not config entries:** the probe runs 6 instructions × 2 models
= 12 runs, but only 5 Nemotron instructions survive into `leaderboard.json`.
Adding 12 temporary entries to a config file that feeds the prod board is an
invitation to commit them by accident.

- [ ] **Step 1: Write the script**

Create `dashboard/scripts/probe_instruction_sensitivity.py`:

```python
"""Does a trading instruction actually change what the model does?

The participatory-competition design assumes instruction quality maps to return
on a pinned model. Nobody has measured it. This runs deliberately opposite
instructions through the leaderboard harness and prints the spread.

    python dashboard/scripts/probe_instruction_sensitivity.py

Costs about $4.97 at 2026-08 prices (6 instructions x 2 models, ~160 LLM calls
each). Requires OPENROUTER_API_KEY -- without it every run silently falls back to
rule-based and the spread is meaningless, so we assert on it up front.

GATE: if the spread across these instructions is under ~1pp on both models, the
instruction axis does not exist and Phase 2 must not be built.
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dashboard.backend.domain.leaderboard.service import load_leaderboard_config
from dashboard.backend.domain.leaderboard.strategies.registry import get_strategy

# The five that become the Phase 1 seed field, plus one control that never ships.
PROBE_INSTRUCTIONS: list[tuple[str, str]] = [
    (
        "aggressive_momentum",
        "Concentrate in the strongest recent performers. Add to positions that "
        "are rising and cut losers quickly. Prefer a small number of large "
        "positions over broad diversification.",
    ),
    (
        "defensive_cash",
        "Preserve capital above all. Hold a large cash position, buy only when a "
        "stock is clearly oversold, and take profits early. Never hold more than "
        "half the portfolio in equities.",
    ),
    (
        "equal_weight_hold",
        "Spread the money evenly across many of the available stocks on the first "
        "opportunity, then hold. Do not react to short-term moves.",
    ),
    (
        "contrarian_reversion",
        "Buy what has fallen the most and sell what has risen the most. Assume "
        "prices revert toward their recent average.",
    ),
    (
        "verbose_analytical",
        "Before each decision, weigh trend, momentum and valuation signals against "
        "each other. Act only when at least two signals agree. Explain the reason "
        "for every order.",
    ),
    # CONTROL — never seeded to the board. If this scores like the others, the
    # model is ignoring the instruction and the axis is dead.
    (
        "control_nonsense",
        "The weather is pleasant today. Consider the colour blue. Bananas are a "
        "type of fruit that grows in warm climates.",
    ),
]

PROBE_MODELS: list[tuple[str, dict]] = [
    (
        "nemotron",
        {
            "model_id": "nvidia/nemotron-3-nano-30b-a3b",
            "integration": "openrouter",
            "temperature": 0,
            "reasoning_effort": "none",
        },
    ),
    (
        "deepseek",
        {
            "model_id": "deepseek/deepseek-v4-pro",
            "integration": "openrouter",
            "temperature": 0,
            "reasoning_effort": "none",
        },
    ),
]

SEEDABLE = [slug for slug, _ in PROBE_INSTRUCTIONS if not slug.startswith("control_")]


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("FATAL: OPENROUTER_API_KEY is unset. Every run would silently fall")
        print("back to rule-based and the probe would measure nothing.")
        return 2

    cfg = load_leaderboard_config()
    results: dict[str, dict[str, float]] = {}

    for model_slug, model_cfg in PROBE_MODELS:
        results[model_slug] = {}
        for slug, instruction in PROBE_INSTRUCTIONS:
            print(f"\n=== {model_slug} / {slug} ===")
            strategy = get_strategy(
                {
                    "strategy": "llm_agent",
                    "mode": "safe_trading",
                    "symbols": [],
                    "strategy_prompt": instruction,
                    **model_cfg,
                }
            )
            curve = strategy.run(
                start_date=cfg["start_date"],
                end_date=cfg["end_date"],
                reference_start_date=cfg["reference_start_date"],
                initial_capital=cfg["initial_capital"],
            )
            first = curve[0]["equity"]
            last = curve[-1]["equity"]
            ret = (last / first - 1.0) * 100.0
            coverage = (
                strategy.llm_decisions / strategy.decision_steps
                if strategy.decision_steps
                else 0.0
            )
            results[model_slug][slug] = ret
            print(f"  return {ret:+.2f}%  llm coverage {coverage:.1%}")
            if coverage < 0.95:
                print("  WARNING: below the H6 threshold — this run could not publish.")

    print("\n" + "=" * 60)
    print("SPREAD")
    print("=" * 60)
    verdict_pass = False
    for model_slug in results:
        rets = list(results[model_slug].values())
        spread = max(rets) - min(rets)
        stdev = statistics.pstdev(rets)
        print(f"\n{model_slug}: spread {spread:.2f}pp, stdev {stdev:.2f}pp")
        for slug, ret in sorted(results[model_slug].items(), key=lambda kv: -kv[1]):
            marker = "  (control)" if slug.startswith("control_") else ""
            print(f"   {ret:+7.2f}%  {slug}{marker}")
        if spread >= 1.0:
            verdict_pass = True

    print("\n" + "=" * 60)
    if verdict_pass:
        print("GATE: PASS — at least one model separates instructions by >=1pp.")
    else:
        print("GATE: FAIL — no model separates instructions. Do NOT build Phase 2.")
    print("=" * 60)
    return 0 if verdict_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the script's own wiring without spending money**

```bash
python -c "
from dashboard.scripts.probe_instruction_sensitivity import PROBE_INSTRUCTIONS, SEEDABLE
assert len(PROBE_INSTRUCTIONS) == 6, PROBE_INSTRUCTIONS
assert len(SEEDABLE) == 5, SEEDABLE
assert all(t.strip() for _, t in PROBE_INSTRUCTIONS)
print('OK', SEEDABLE)
"
```

Expected: `OK ['aggressive_momentum', 'defensive_cash', 'equal_weight_hold', 'contrarian_reversion', 'verbose_analytical']`

- [ ] **Step 3: Confirm the fatal guard fires**

```bash
env -u OPENROUTER_API_KEY python dashboard/scripts/probe_instruction_sensitivity.py; echo "exit=$?"
```

Expected: the FATAL message and `exit=2`. **Do not skip this** — a probe that
silently measures rule-based fallback would produce a confident, meaningless
verdict.

- [ ] **Step 4: Commit**

```bash
git status --short
git add dashboard/scripts/probe_instruction_sensitivity.py
git commit -m "chore(leaderboard): instruction-sensitivity probe script"
```

---

### Task 3: Run the probe and record the gate decision

**Files:**
- Create: `docs/superpowers/probe-results/2026-08-09-instruction-sensitivity.md`

**Interfaces:**
- Consumes: Task 2's script.
- Produces: the pass/fail decision, and — on a Nemotron-fails/DeepSeek-passes result — the pinned model for every later task.

**This task spends real money (~$4.97).** Confirm with the repo owner before
running.

- [ ] **Step 1: Run the probe**

```bash
python dashboard/scripts/probe_instruction_sensitivity.py 2>&1 | tee /tmp/probe.log
```

Runtime is roughly 30–60 minutes: 12 runs × ~160 sequential LLM calls.

- [ ] **Step 2: Record the result**

Create `docs/superpowers/probe-results/2026-08-09-instruction-sensitivity.md`
containing: the full spread table from stdout, the H6 coverage per run, the
verdict, and — critically — **where the control landed**. A control that scores
mid-pack is the single most informative outcome: it means the model responds to
*having* an instruction rather than to its content, which passes a naive spread
check while the leaderboard still ranks noise.

- [ ] **Step 3: Apply the gate**

| Outcome | Action |
|---|---|
| Nemotron spread ≥1pp **and** control is an outlier | **PASS.** Proceed to Phase 1 as written. |
| Nemotron flat, DeepSeek spread ≥1pp | **PASS with contingency.** Pin DeepSeek. Per the spec, the budget holds and the grant shrinks: Competition attempts 5 → 1. Update the spec's cost tables and every `nvidia/nemotron-3-nano-30b-a3b` reference in Task 4 before proceeding. |
| Both flat, **or** the control scores mid-pack on both | **FAIL. Stop.** The instruction axis does not exist. Phase 1's hero chart would advertise a competition that cannot be won on merit. Report back rather than proceeding. |

- [ ] **Step 4: Commit the result**

```bash
git status --short
git add docs/superpowers/probe-results/2026-08-09-instruction-sensitivity.md
git commit -m "docs: instruction-sensitivity probe results and gate decision"
```

---

# PHASE 1 — Evidence and Measurement (one PR)

Ships the hook without shipping the competition: the landing hero becomes the
real board, the Open Track gets a visible field of house-authored entries, and
the funnel is instrumented. **No user can enter yet, and the UI must not pretend
otherwise** — the `#homeGetStartedBtn` dead-CTA failure is the thing this design
exists to fix, so a CTA that leads nowhere is a Phase 1 bug, not a Phase 2
placeholder.

**Before starting:** `git fetch origin && git rebase origin/main`. ~~and rebase
onto PR #326 if it has not yet merged.~~ (#326 merged 2026-08-09.)

> **Scope reduced 2026-08-15.** Phase 1 as described below is larger than what is
> left to do. #352/#357 delivered the above-the-fold board and rewrote the
> landing's data story; Tasks 5, 7 and 8 are withdrawn, superseded or done. The
> remaining PR is **Tasks 1–4, 6, 9 and 10**, and it no longer touches the
> landing page's data path. The paragraph below still describes the phase's
> *purpose* correctly, and the dead-CTA principle in it is worth keeping — it is
> now the standard the shipped page is being held to, not a change this phase
> makes.

---

### Task 4: Seed the Open Track with six house-authored entries

**Files:**
- Modify: `dashboard/config/leaderboard.json`
- Test: `dashboard/backend/tests/domain/leaderboard/test_seed_entries.py` (create)

**Interfaces:**
- Consumes: `PROBE_INSTRUCTIONS`/`SEEDABLE` (Task 2), `strategy_prompt` support (Task 1).
- Produces: six config entries with `id` values `open_house_reference`, `open_seed_aggressive_momentum`, `open_seed_defensive_cash`, `open_seed_equal_weight_hold`, `open_seed_contrarian_reversion`, `open_seed_verbose_analytical`, each carrying `"label": "Open Track"` and `"authored_by": "Agentic Trading Lab"`. Task 5 filters the API payload on `label == "Open Track"`.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/domain/leaderboard/test_seed_entries.py`:

```python
"""Phase 1 ships an Open Track field of house-authored entries.

A board with one row is not a board, and an unlabelled house entry sitting among
what will later be user entries is dishonest. These guard both.
"""

import json
from pathlib import Path

import pytest

CONFIG = json.loads(
    (Path(__file__).resolve().parents[4] / "config" / "leaderboard.json").read_text(
        encoding="utf-8"
    )
)
ENTRIES = CONFIG["strategies"]
OPEN_TRACK = [e for e in ENTRIES if e.get("label") == "Open Track"]

PINNED_MODEL_ID = "nvidia/nemotron-3-nano-30b-a3b"

# Deliberately literal rather than imported from the probe script:
# dashboard/scripts/ is NOT a Python package (no __init__.py), so
# `from dashboard.scripts.probe_instruction_sensitivity import SEEDABLE` raises
# ModuleNotFoundError at collection — and a collection error aborts the whole
# pytest session, not just this file. The config is the source of truth once
# seeded; the probe script is throwaway measurement.
SEEDABLE = [
    "aggressive_momentum",
    "defensive_cash",
    "equal_weight_hold",
    "contrarian_reversion",
    "verbose_analytical",
]


def test_open_track_has_a_house_reference_plus_five_seeds():
    assert len(OPEN_TRACK) == 6, [e["id"] for e in OPEN_TRACK]
    assert any(e["id"] == "open_house_reference" for e in OPEN_TRACK)


def test_every_seed_slug_from_the_probe_is_present():
    seeded = {e["id"] for e in OPEN_TRACK}
    for slug in SEEDABLE:
        assert f"open_seed_{slug}" in seeded, slug


def test_every_open_track_entry_carries_an_instruction():
    for entry in OPEN_TRACK:
        assert entry.get("strategy_prompt", "").strip(), entry["id"]


def test_every_open_track_entry_is_labelled_house_authored():
    """Seeds must never be mistakable for user entries."""
    for entry in OPEN_TRACK:
        assert entry.get("authored_by") == "Agentic Trading Lab", entry["id"]


def test_every_open_track_entry_is_pinned_to_the_season_model():
    """A silent switch to a frontier model would multiply spend ~190x."""
    for entry in OPEN_TRACK:
        assert entry["strategy"] == "llm_agent", entry["id"]
        assert entry["model_id"] == PINNED_MODEL_ID, entry["id"]
        assert entry["integration"] == "openrouter", entry["id"]
        assert entry["temperature"] == 0, entry["id"]


def test_model_track_entries_still_carry_no_instruction():
    """The seven published curves must not change. C1 reads this key; if one
    appeared on a Model Track entry, its next re-deploy would produce a
    different curve under the same published name."""
    model_track = [e for e in ENTRIES if e.get("label") == "Model"]
    assert len(model_track) == 7, [e["id"] for e in model_track]
    for entry in model_track:
        assert "strategy_prompt" not in entry, entry["id"]


@pytest.mark.parametrize("entry", OPEN_TRACK, ids=lambda e: e["id"])
def test_open_track_ids_do_not_collide_with_the_lb_run_prefix(entry):
    assert not entry["id"].startswith("lb_"), entry["id"]
```

- [ ] **Step 2: Run it to verify it fails**

```bash
pytest dashboard/backend/tests/domain/leaderboard/test_seed_entries.py -v
```

Expected: FAIL — `assert 0 == 6`.

- [ ] **Step 3: Add the six entries**

Append to the `strategies` array in `dashboard/config/leaderboard.json`. The
house reference uses the **existing** house behaviour expressed as an explicit
instruction — this is what makes it the bar being challenged:

```json
    {
      "id": "open_house_reference",
      "name": "House instruction",
      "label": "Open Track",
      "authored_by": "Agentic Trading Lab",
      "model": "Nemotron 3 Nano 30B",
      "provider": "NVIDIA",
      "strategy": "llm_agent",
      "integration": "openrouter",
      "model_id": "nvidia/nemotron-3-nano-30b-a3b",
      "reasoning_effort": "none",
      "temperature": 0,
      "mode": "safe_trading",
      "symbols": [],
      "auto_compute": false,
      "strategy_prompt": "Manage the portfolio prudently. Diversify across strong stocks, size positions sensibly, and avoid concentrating risk in any single name."
    },
    {
      "id": "open_seed_aggressive_momentum",
      "name": "Aggressive momentum",
      "label": "Open Track",
      "authored_by": "Agentic Trading Lab",
      "model": "Nemotron 3 Nano 30B",
      "provider": "NVIDIA",
      "strategy": "llm_agent",
      "integration": "openrouter",
      "model_id": "nvidia/nemotron-3-nano-30b-a3b",
      "reasoning_effort": "none",
      "temperature": 0,
      "mode": "safe_trading",
      "symbols": [],
      "auto_compute": false,
      "strategy_prompt": "Concentrate in the strongest recent performers. Add to positions that are rising and cut losers quickly. Prefer a small number of large positions over broad diversification."
    }
```

…and the same shape for `open_seed_defensive_cash`, `open_seed_equal_weight_hold`,
`open_seed_contrarian_reversion` and `open_seed_verbose_analytical`, with
`strategy_prompt` copied **verbatim** from `PROBE_INSTRUCTIONS` and `name` set to
"Defensive cash", "Equal weight & hold", "Contrarian reversion" and
"Weighted signals" respectively.

> **Verbatim matters:** the probe's measured returns are only evidence for these
> entries if the strings are byte-identical. If you retype rather than copy, the
> recorded probe results no longer describe what ships.

- [ ] **Step 4: Run the test**

```bash
pytest dashboard/backend/tests/domain/leaderboard/test_seed_entries.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Deploy the six curves**

```bash
for id in open_house_reference open_seed_aggressive_momentum open_seed_defensive_cash \
          open_seed_equal_weight_hold open_seed_contrarian_reversion open_seed_verbose_analytical; do
  python dashboard/scripts/deploy_leaderboard_model.py --entry "$id"
done
```

Cost: ~$0.43, or $0 if you reuse the Phase 0 Nemotron runs. Each must report a
published curve; an H6 rejection here means that instruction cannot seed the
board — drop it and note why in the probe-results doc rather than lowering the
threshold.

- [ ] **Step 6: Verify the API serves them and the seed DB was not mutated**

```bash
git status --short   # dashboard/storage/data/ MUST NOT appear
python -c "
from dashboard.backend.domain.leaderboard.service import get_leaderboard
p = get_leaderboard()
open_track = [e for e in p['entries'] if e.get('team_badge') == 'Open Track' or 'open_' in str(e.get('entry_id'))]
print(len(open_track), [e.get('entry_id') for e in open_track])
"
```

- [ ] **Step 7: Verify the existing `/app` board still styles correctly**

The six new entries appear on the existing `/app` Competition board too, carrying
a `team_badge` value (`Open Track`) that `LEADERBOARD_STYLES` has no preset for.
That is safe by design — `getSeriesStyle` (`js/leaderboard.js:99-106`) falls
through to `{ color: getTeamColor(entry_id), kind: 'team' }`, which is the
correct tier for an entrant. Confirm it rather than assume it:

```bash
uvicorn dashboard.backend.app:app --reload
```

Load `http://localhost:8000/app?view=leaderboard` and confirm the six entries
render as solid coloured `team`-tier lines with distinct colours, appear in the
curve picker, and do not collide visually with the seven grey model lines.

- [ ] **Step 8: Commit**

```bash
git status --short
git add dashboard/config/leaderboard.json \
        dashboard/backend/tests/domain/leaderboard/test_seed_entries.py
git commit -m "feat(leaderboard): seed the Open Track with six house-authored entries"
```

---

### Task 5: Typed leaderboard client for the landing page

> ## ⛔ BLOCKED 2026-08-15 — premise withdrawn, do not build
>
> This module exists solely to let the landing page fetch the live board. PR #357
> decided the landing page ships **labelled sample data** instead
> (`BoardPreview.tsx`), and put the live board on the `/app` home screen, which
> fetches through `home-page.js:1518` and needs nothing from here.
>
> There is a real argument for that choice, not just an accident: the landing is
> served from Vercel while the API is on Render's **free tier, which spins down**,
> so an above-the-fold cross-origin fetch to a cold backend is the weakest
> possible first impression.
>
> **Build this only if the sample-vs-live decision is deliberately reopened** —
> and if it is, budget for the cold-start problem first, because a spinner or an
> error card above the fold is worse than an honest labelled sample. Tasks 6, 9
> and 10 do not depend on this one.

**Files:**
- Create: `dashboard/landing/src/lib/leaderboard.ts`

**Interfaces:**
- Consumes: `GET /api/v1/leaderboard` (unchanged; no auth).
- Produces:
  - `type BoardEntry = { entryId: string; name: string; badge: string; rank: number; returnPct: number; curve: { t: string; equity: number }[] }`
  - `type BoardState = { status: "loading" | "ok" | "error" | "empty"; entries: BoardEntry[]; message?: string }`
  - `fetchBoard(signal?: AbortSignal): Promise<BoardEntry[]>`
  - `toRechartsSeries(entries: BoardEntry[]): { rows: Record<string, number | string>[]; keys: string[] }`
  - `selectHeroSeries(entries: BoardEntry[]): BoardEntry[]`

  Task 6 imports all five names exactly as spelled here.

**Isolation requirement:** this module owns the API contract for the landing
side. `Race.tsx` must not call `fetch` directly, so a contract change touches one
file.

- [ ] **Step 1: Write the module**

Create `dashboard/landing/src/lib/leaderboard.ts`:

```ts
/**
 * The landing page's sole consumer of GET /api/v1/leaderboard.
 *
 * Same-origin by design: the Vercel rewrite proxies /api/* to Render, and the
 * CSP on both origins allows connect-src 'self'. MarketTicker.tsx hardcodes
 * https://agentictrading.onrender.com instead — do not copy that; it defeats the
 * rewrite and pins the frontend to one backend hostname.
 */

export type BoardEntry = {
  entryId: string;
  name: string;
  badge: string;
  rank: number;
  returnPct: number;
  curve: { t: string; equity: number }[];
};

export type BoardState = {
  status: "loading" | "ok" | "error" | "empty";
  entries: BoardEntry[];
  message?: string;
};

const FETCH_TIMEOUT_MS = 45_000;

export async function fetchBoard(signal?: AbortSignal): Promise<BoardEntry[]> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  signal?.addEventListener("abort", () => controller.abort());
  try {
    const res = await fetch("/api/v1/leaderboard?period=contest", {
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const raw = Array.isArray(data?.entries) ? data.entries : [];
    return raw
      .filter((e: any) => Array.isArray(e?.equity_curve) && e.equity_curve.length > 0)
      .map((e: any) => ({
        entryId: String(e.entry_id ?? ""),
        name: String(e.model ?? e.team_name ?? e.entry_id ?? "Entry"),
        badge: String(e.team_badge ?? ""),
        rank: Number(e.rank ?? 0),
        returnPct: Number(e.cumulative_return ?? 0),
        curve: e.equity_curve.map((p: any) => ({
          t: String(p.timestamp),
          equity: Number(p.equity),
        })),
      }));
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * The hero shows at most 8 lines: the market baseline, the house instruction,
 * and the top Open Track entries. The seven Model Track models are returned too
 * but the chart renders them as one thin grey cluster — see Race.tsx.
 */
export function selectHeroSeries(entries: BoardEntry[]): BoardEntry[] {
  const byId = (id: string) => entries.find((e) => e.entryId === id);
  const baseline = byId("buy_hold_djia") ?? byId("djia_index");
  const house = byId("open_house_reference");
  const openTrack = entries
    .filter((e) => e.entryId.startsWith("open_seed_"))
    .sort((a, b) => b.returnPct - a.returnPct)
    .slice(0, 3);
  const models = entries.filter((e) => e.badge === "Model");
  return [...models, ...(baseline ? [baseline] : []), ...(house ? [house] : []), ...openTrack];
}

/**
 * recharts wants one row per x-value with a key per series. Curves are merged on
 * the timestamp string rather than by array index: entries tick at different
 * minutes, so index-merging silently misaligns them (the same reason
 * buildEquityCurvesFromEntries in js/leaderboard.js merges on a time key).
 */
export function toRechartsSeries(entries: BoardEntry[]): {
  rows: Record<string, number | string>[];
  keys: string[];
} {
  const axis = new Set<string>();
  for (const e of entries) for (const p of e.curve) axis.add(p.t);
  const sorted = [...axis].sort();
  const keys = entries.map((e) => e.entryId);
  const rows = sorted.map((t) => {
    const row: Record<string, number | string> = { t };
    for (const e of entries) {
      const point = e.curve.find((p) => p.t === t);
      if (point) row[e.entryId] = point.equity;
    }
    return row;
  });
  return { rows, keys };
}
```

- [ ] **Step 2: Typecheck**

```bash
cd dashboard/landing && npx tsc --noEmit
```

Expected: no errors. (Run `npm install` first if `node_modules` is absent.)

- [ ] **Step 3: Commit**

```bash
git status --short
git add dashboard/landing/src/lib/leaderboard.ts
git commit -m "feat(landing): typed leaderboard client"
```

---

### Task 6: Funnel instrumentation

**Files:**
- Create: `dashboard/landing/src/lib/analytics.ts`

**Interfaces:**
- Consumes: `@vercel/analytics` (already a dependency, `package.json:75`; already mounted at `App.tsx:30`).
- Produces: `trackEvent(name: FunnelEvent, props?: Record<string, string | number>): void` and `type FunnelEvent = "landing_view" | "board_interact" | "cta_click"`. Task 7 calls `trackEvent`.

**Deviation from the spec, deliberate:** the spec calls for a *first-party
beacon* in Phase 1. A beacon needs somewhere to write, and Phase 1's defining
constraint is **no new tables**. Rather than smuggle a table into the phase that
was scoped to avoid them, Phase 1 uses the `@vercel/analytics` custom-event API
that is already installed and already mounted, and Phase 2 adds the first-party
beacon alongside the tables it is already creating. If you disagree with this
trade, the alternative is pulling one `analytics_events` table plus a
`POST /api/v1/events` route (and its two golden-set updates) into Phase 1.

**Scope note:** the spec lists six funnel events. Three are landing-side and
ship here. `signup_complete`, `instruction_saved` and `attempt_submitted` are
`/app`-side, where the React bundle does not run — they arrive with Phase 2's
first-party beacon, alongside the tables that phase already adds. Phase 1
therefore under-counts ad-blocking visitors; that is acceptable because Phase 1's
question is comparative (does the new hero convert better than the old page) and
the undercount applies equally to both.

- [ ] **Step 1: Verify custom events are actually available**

Open the Vercel project dashboard → Analytics. Confirm **Web Analytics is
enabled** and that custom events are included on the current plan.

**If custom events are not available, stop and report back** rather than
shipping calls that silently no-op — the fallback is Phase 2's first-party
beacon pulled forward, which is a different task with a backend route and a
golden-set update.

- [ ] **Step 2: Write the module**

Create `dashboard/landing/src/lib/analytics.ts`:

```ts
import { track } from "@vercel/analytics";

/**
 * The landing half of the Phase 1 funnel. Deliberately three events, not ten:
 * the question is "does the board convert a visitor into a signup", and every
 * event beyond that is noise nobody will read.
 *
 * The /app half (signup_complete, instruction_saved, attempt_submitted) lands
 * with Phase 2's first-party beacon — the React bundle does not run on /app.
 */
export type FunnelEvent = "landing_view" | "board_interact" | "cta_click";

export function trackEvent(
  name: FunnelEvent,
  props?: Record<string, string | number>,
): void {
  try {
    track(name, props);
  } catch {
    // Analytics must never break the page. A blocked or failed beacon is an
    // expected condition, not an error worth surfacing.
  }
}
```

- [ ] **Step 3: Typecheck and commit**

```bash
cd dashboard/landing && npx tsc --noEmit
cd ../.. && git status --short
git add dashboard/landing/src/lib/analytics.ts
git commit -m "feat(landing): funnel event wrapper"
```

---

### Task 7: `Race` renders the live board

> ## ⛔ SUPERSEDED 2026-08-15 — and its guard test is now *wrong*
>
> PR #357 resolved this task in the opposite direction, deliberately.
>
> **What shipped instead:** the chart moved out of `Race.tsx` into a new
> `BoardPreview.tsx` mounted in the hero, and it renders `SAMPLE_CURVES` /
> `SAMPLE_STANDINGS` — hardcoded illustrative numbers — under a visible
> "Illustrative example" badge. `Race.tsx` keeps the full standings table and
> imports `SAMPLE_STANDINGS` from `BoardPreview`.
>
> **Why the guard below must not be written as specified.** Step 1 creates
> `test_landing_race_copy.py` asserting `"Illustrative" not in RACE` and that
> `SAMPLE_CURVES`/`SAMPLE_STANDINGS` are gone. Main now ships the inverse
> assertion:
> `test_landing_copy_register.py::test_illustrative_example_label_appears_at_least_twice`
> counts `"Illustrative example"` in the **shipped bundle** and requires ≥ 2.
> Implementing this task means deleting that guard. That is a decision to make
> explicitly with the person who added it, not a step to execute.
>
> **One detail worth preserving if this is ever revisited.** `BoardPreview.tsx`
> spells the label out as a literal in each card rather than sharing an exported
> constant, and says so in a comment. That is not sloppiness: esbuild collapses a
> shared constant to a single string literal, so the DRY version renders the label
> on both cards while the bundle-level guard drops from 3 hits to 1. Do not
> "clean it up".
>
> The parts of this task that are *not* superseded — the CTA naming the beatable
> number rather than the market, and the ban on "strategy"/"teams" in user-facing
> copy — remain live requirements and are already honoured by the shipped copy.

**Files:**
- Modify: `dashboard/landing/src/components/home/Race.tsx`
- Test: `dashboard/backend/tests/test_landing_race_copy.py` (create)

**Interfaces:**
- Consumes: `fetchBoard`, `selectHeroSeries`, `toRechartsSeries`, `BoardState` (Task 5); `trackEvent` (Task 6).
- Produces: a `Race` section that owns its own loading, error and empty states.

**Isolation requirement (spec §Isolation contract):** a failed board fetch must
render *inside this section only*. It must never throw to a parent, blank the
page, or block the rest of the landing from rendering.

- [ ] **Step 1: Write the copy guard first**

Create `dashboard/backend/tests/test_landing_race_copy.py`:

```python
"""Guards on the shipped landing source for the Phase 1 hook.

Asserted against the React source rather than the built bundle: the bundle is
minified and its identifiers are mangled, so a source assertion is the only one
that survives a rebuild. test_frontend_bundle_integrity.py separately guards that
the bundle and source agree on CTA labels.
"""

from pathlib import Path

RACE = (
    Path(__file__).resolve().parents[2]
    / "landing"
    / "src"
    / "components"
    / "home"
    / "Race.tsx"
).read_text(encoding="utf-8")


def test_the_illustrative_badge_is_gone():
    """The data is real now. Leaving the badge would understate it; leaving the
    badge AND real data would be a lie in the other direction."""
    assert "Illustrative" not in RACE


def test_sample_data_is_gone():
    for dead in ("SAMPLE_CURVES", "SAMPLE_STANDINGS"):
        assert dead not in RACE, dead


def test_the_cta_names_the_beatable_number_not_the_market():
    """Spec: rank against the house instruction, never imply the market is easy
    to beat."""
    assert "buy-and-hold" in RACE
    assert "instruction" in RACE


def test_copy_avoids_the_reserved_product_nouns():
    """'strategy' collides with domain/strategies/; 'teams' contradicts one
    entrant = one person."""
    for line in RACE.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("<p", "<h2", "<h3", "<li", "{\"")):
            continue
        assert "your strategy" not in stripped.lower(), stripped
        assert "teams" not in stripped.lower(), stripped


def test_race_does_not_fetch_directly():
    """The API contract lives in lib/leaderboard.ts so a change touches one file."""
    assert "fetch(" not in RACE
    assert "from \"@/lib/leaderboard\"" in RACE


def test_race_renders_its_own_error_state():
    """Isolation contract: one failed fetch must not blank the page."""
    assert "error" in RACE
```

- [ ] **Step 2: Run it to verify it fails**

```bash
pytest dashboard/backend/tests/test_landing_race_copy.py -v
```

Expected: multiple failures — `SAMPLE_CURVES` present, "Illustrative" present.

- [ ] **Step 3: Rewrite `Race.tsx`**

Replace the sample constants and the component body. Keep the existing
`Button`/`PRIMARY_LANDING_CTA` usage, the card chrome and the Tailwind classes;
change the data source, the copy, and add the states:

```tsx
import { useEffect, useMemo, useRef, useState } from "react";
import { Medal } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { PRIMARY_LANDING_CTA } from "@/lib/cta";
import {
  fetchBoard, selectHeroSeries, toRechartsSeries,
  type BoardEntry, type BoardState,
} from "@/lib/leaderboard";
import { trackEvent } from "@/lib/analytics";

/** Model Track entries render as one thin grey cluster; Open Track entries are bold. */
const OPEN_TRACK_COLORS = ["#22d3ee", "#a78bfa", "#fbbf24"];
const HOUSE_COLOR = "#f97316";
const BASELINE_COLOR = "#94a3b8";
const MODEL_CLUSTER_COLOR = "#475569";

function styleFor(entry: BoardEntry, openTrackIndex: number) {
  if (entry.entryId === "open_house_reference") {
    return { stroke: HOUSE_COLOR, strokeWidth: 2.5, dash: undefined, opacity: 1 };
  }
  if (entry.entryId.startsWith("open_seed_")) {
    return {
      stroke: OPEN_TRACK_COLORS[openTrackIndex % OPEN_TRACK_COLORS.length],
      strokeWidth: 2.5, dash: undefined, opacity: 1,
    };
  }
  if (entry.badge === "Model") {
    return { stroke: MODEL_CLUSTER_COLOR, strokeWidth: 1, dash: undefined, opacity: 0.35 };
  }
  return { stroke: BASELINE_COLOR, strokeWidth: 1.5, dash: "4 4", opacity: 0.9 };
}

export function Race() {
  const [state, setState] = useState<BoardState>({ status: "loading", entries: [] });
  const interacted = useRef(false);

  useEffect(() => {
    let cancelled = false;
    fetchBoard()
      .then((entries) => {
        if (cancelled) return;
        setState(
          entries.length
            ? { status: "ok", entries }
            : { status: "empty", entries: [], message: "The board has no published entries yet." },
        );
      })
      .catch(() => {
        if (cancelled) return;
        setState({
          status: "error",
          entries: [],
          message: "The leaderboard is temporarily unavailable.",
        });
      });
    return () => { cancelled = true; };
  }, []);

  const hero = useMemo(() => selectHeroSeries(state.entries), [state.entries]);
  const { rows } = useMemo(() => toRechartsSeries(hero), [hero]);

  const standings = useMemo(
    () =>
      hero
        .filter((e) => e.entryId.startsWith("open_") || e.badge === "Model")
        .sort((a, b) => b.returnPct - a.returnPct)
        .slice(0, 5),
    [hero],
  );

  function onChartInteract() {
    if (interacted.current) return;
    interacted.current = true;
    trackEvent("board_interact");
  }

  let openTrackSeen = 0;

  return (
    <section id="race" className="py-24 bg-muted/20 border-y border-border scroll-mt-40">
      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-12 items-start mb-12">
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-3">
              Most models lost to buy-and-hold
            </h2>
            <p className="text-foreground/80 mb-6 text-lg">
              We gave seven frontier models the same house instruction and one month of
              real market data. Six of them finished behind simply buying and holding.
              Can you beat them with a better instruction?
            </p>
            <ul className="space-y-2 mb-8 text-sm text-foreground/80">
              <li>· Real hourly market data — no real money at risk</li>
              <li>· One pinned model, so the instruction is the only variable</li>
              <li>· Same window, same starting cash, same rules for every entry</li>
            </ul>
            <Button
              size="lg"
              type="button"
              data-landing-auth={PRIMARY_LANDING_CTA.authMode}
              onClick={() => trackEvent("cta_click", { placement: "race" })}
              className="bg-primary text-primary-foreground hover:bg-primary/90"
            >
              {PRIMARY_LANDING_CTA.label}
            </Button>
          </div>

          <div className="bg-card border border-card-border rounded-xl shadow-xl p-6">
            <div className="flex items-center justify-between mb-2 border-b border-border pb-4 gap-3">
              <h3 className="text-xl font-bold flex items-center gap-2 min-w-0">
                <Medal className="w-5 h-5 text-primary shrink-0" />
                Standings
              </h3>
            </div>
            <div className="space-y-2 mt-4">
              {state.status === "loading" && (
                <p className="text-sm text-muted-foreground p-3">Loading the board…</p>
              )}
              {(state.status === "error" || state.status === "empty") && (
                <p className="text-sm text-muted-foreground p-3">{state.message}</p>
              )}
              {state.status === "ok" && standings.map((item, i) => {
                const isOpen = item.entryId.startsWith("open_");
                return (
                  <div
                    key={item.entryId}
                    className={`grid grid-cols-12 items-center p-3 border rounded-lg ${
                      isOpen ? "bg-primary/10 border-primary/40" : "bg-background border-border"
                    }`}
                  >
                    <div className="col-span-2 font-mono font-bold text-muted-foreground">#{i + 1}</div>
                    <div className="col-span-7 font-medium truncate pr-2">
                      {item.name}
                      <span className="ml-2 text-xs font-mono text-muted-foreground">
                        {isOpen ? "instruction" : "model"}
                      </span>
                    </div>
                    <div className={`col-span-3 text-right font-mono font-bold ${
                      item.returnPct >= 0 ? "text-positive" : "text-destructive"
                    }`}>
                      {item.returnPct >= 0 ? "+" : ""}{item.returnPct.toFixed(2)}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="bg-card border border-card-border rounded-xl shadow-xl p-6 md:p-8">
          <h3 className="text-lg font-bold mb-6">Leaderboard</h3>
          <div className="h-[320px] md:h-[400px] w-full" onMouseEnter={onChartInteract}>
            {state.status !== "ok" ? (
              <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
                {state.status === "loading" ? "Loading the board…" : state.message}
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                  <XAxis dataKey="t" stroke="hsl(var(--muted-foreground))" fontSize={12}
                         tickLine={false} axisLine={false} tickFormatter={(v: string) => v.slice(5, 10)} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false}
                         axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => `$${v}`} />
                  <Tooltip contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    borderColor: "hsl(var(--border))",
                    borderRadius: "8px",
                  }} />
                  {hero.map((entry) => {
                    const isOpenSeed = entry.entryId.startsWith("open_seed_");
                    const style = styleFor(entry, isOpenSeed ? openTrackSeen++ : 0);
                    return (
                      <Line
                        key={entry.entryId}
                        type="linear"
                        dataKey={entry.entryId}
                        name={entry.name}
                        stroke={style.stroke}
                        strokeWidth={style.strokeWidth}
                        strokeDasharray={style.dash}
                        strokeOpacity={style.opacity}
                        connectNulls
                        dot={false}
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
```

- [ ] **Step 4: Run the guard and the typechecker**

```bash
pytest dashboard/backend/tests/test_landing_race_copy.py -v
cd dashboard/landing && npx tsc --noEmit
```

Expected: 6 passed; no TS errors.

- [ ] **Step 5: Verify against a live backend**

```bash
# terminal 1
uvicorn dashboard.backend.app:app --reload
# terminal 2
cd dashboard/landing && npm run dev
```

Confirm, in the browser: the chart renders real curves; the Model Track lines are
a faint grey cluster while Open Track lines are bold; **stop the backend and
reload** — the Race section must show its own error text while every other
section still renders.

- [ ] **Step 6: Commit**

```bash
git status --short
git add dashboard/landing/src/components/home/Race.tsx \
        dashboard/backend/tests/test_landing_race_copy.py
git commit -m "feat(landing): Race renders the live leaderboard"
```

---

### Task 8: Promote the board above the fold and renumber the storyline

> ## ✅ DONE 2026-08-15 by different means — one loose end remains
>
> PR #357 achieved the goal without the reorder this task specifies.
>
> | Step | Status |
> |---|---|
> | Move `Race` under `Hero` in `landing-page.tsx` | **Not done, and not needed.** The chart was extracted into `BoardPreview` and mounted at `Hero.tsx:144` instead. Section order is unchanged: `Hero → WhyCare → Talk → Test → Race` |
> | `Talk` keeps `01`, `Test` keeps `02` | **Done** — `Talk.tsx:13`, `Test.tsx:143` |
> | The board drops its `03 — Race` line | **Done** — no `03 —` remains anywhere in `landing/src/` |
> | `FooterCTA.tsx:10`: `Talk → Test → Race` → `Race → Talk → Test` | **⚠ NOT DONE** |
>
> **The loose end is real and small.** `FooterCTA.tsx:10` is now the only place on
> the page asserting a sequence the page no longer has — the board is the frame,
> and `Race` lost its number, but the footer still narrates `Talk → Test → Race`.
> Worth fixing whenever the landing source is next touched; it forces a bundle
> rebuild (Task 11), so it is not worth a PR of its own.
>
> **The sign-off this task required never happened.** The `01/02/03` sequence is
> Allan's copy, and #357 renumbered it without asking. That is water under the
> bridge, but if the footer line is changed, mention it rather than slipping a
> second unreviewed copy edit past the same person.

**Files:**
- Modify: `dashboard/landing/src/pages/landing-page.tsx:18-22`
- Modify: `dashboard/landing/src/components/home/Talk.tsx:12`
- Modify: `dashboard/landing/src/components/home/Test.tsx:143`
- Modify: `dashboard/landing/src/components/home/FooterCTA.tsx:10`

**Interfaces:**
- Consumes: Task 7's `Race`.
- Produces: section order `Hero → Race → WhyCare → Talk → Test`.

**Sign-off required:** the `01 — Talk / 02 — Test / 03 — Race` sequence is
Allan's copy (`WhyCare.tsx:5` comments on the numbering deliberately, and
`FooterCTA.tsx:10` restates it). **Confirm with the repo owner before
committing.** Promoting Race makes it the frame rather than step three, so it
loses its number and Talk/Test keep 01/02 — the sequence is not deleted.

- [ ] **Step 1: Reorder**

In `landing-page.tsx`, move `<Race />` to sit directly after `<Hero />`:

```tsx
      <main>
        <Hero />
        <Race />
        <WhyCare />
        <Talk />
        <Test />
      </main>
```

- [ ] **Step 2: Renumber**

`Race.tsx` already dropped its `03 — Race` line in Task 7. Leave `Talk.tsx:12`
(`01 — Talk`) and `Test.tsx:143` (`02 — Test`) as they are — they remain correct.
In `FooterCTA.tsx:10`, change `Talk → Test → Race` to `Race → Talk → Test`.

- [ ] **Step 3: Verify no orphaned numbering**

```bash
command grep -rn "03 —\|Talk → Test → Race" dashboard/landing/src/
```

Expected: no output.

- [ ] **Step 4: Visual check**

`npm run dev`, then confirm the chart is visible without scrolling at 1280×800,
and that the `#race` anchor in `Navbar.tsx:9` still scrolls correctly.

- [ ] **Step 5: Commit**

```bash
git status --short
git add dashboard/landing/src/pages/landing-page.tsx \
        dashboard/landing/src/components/home/FooterCTA.tsx
git commit -m "feat(landing): promote the leaderboard above the fold"
```

---

### Task 9: Social share card

> **✅ Still valid, unchanged, and still worth doing. Re-verified 2026-08-15:**
> `index.html` declares `twitter:card="summary_large_image"` and supplies no
> image — the exact state this task describes. `dashboard/frontend/images/`
> contains five logo variants plus `snapshot.png`, and no `og-card.png`. The
> `<head>` runs to `:22`, with the description/OG/Twitter meta tags at `:7-14`;
> insert after `:14` as the task says.
>
> One addition: if the card is produced by screenshotting the hero, note the hero
> now shows **illustrative** numbers. A share card built from sample data and
> presented as a result would be the dishonesty the badge exists to prevent.
> Either shoot the `/app` board (live data) or keep the card to product name +
> hook sentence with no numbers on it.

**Files:**
- Create: `dashboard/frontend/images/og-card.png` (1200×630)
- Modify: `dashboard/frontend/index.html:9-14`

**Interfaces:**
- Consumes: nothing.
- Produces: `og:image` / `twitter:image` meta tags.

Today `index.html:12` declares `twitter:card="summary_large_image"` and supplies
**no image** — the worst of both, since it requests a large card and gives the
crawler nothing. Dynamic per-entry images are rejected in the spec: Render's free
tier spins down and link-preview crawlers abandon in seconds, so most first
shares would render nothing.

- [ ] **Step 1: Produce the image**

A 1200×630 PNG carrying the product name and the hook sentence
("Most models lost to buy-and-hold"). Screenshot the new hero chart at
1200×630 if no designed asset exists. Keep it under 300 KB.

- [ ] **Step 2: Add the tags**

In `index.html`, after line 14:

```html
    <meta property="og:image" content="https://agentictrading.com/images/og-card.png" />
    <meta property="og:image:width" content="1200" />
    <meta property="og:image:height" content="630" />
    <meta name="twitter:image" content="https://agentictrading.com/images/og-card.png" />
```

> Absolute URLs are required — crawlers do not resolve relative `og:image`
> paths. Replace the host with the production domain if it differs from
> `agentictrading.com`; check the Vercel project's assigned domain first.

- [ ] **Step 3: Verify the asset resolves**

```bash
python - <<'PY'
from pathlib import Path
html = Path("dashboard/frontend/index.html").read_text(encoding="utf-8")
assert 'og:image' in html and 'twitter:image' in html
assert Path("dashboard/frontend/images/og-card.png").is_file()
print("OK", Path("dashboard/frontend/images/og-card.png").stat().st_size, "bytes")
PY
```

- [ ] **Step 4: Commit**

```bash
git status --short
git add dashboard/frontend/index.html dashboard/frontend/images/og-card.png
git commit -m "feat(landing): social share card"
```

---

### Task 10: Rename Community → Agent Marketplace, Teams → Traders

> **✅ Still valid. Every line number below moved — use these instead
> (verified 2026-08-15 against `origin/main` @ `88c7b8c`):**
>
> | This task says | Actual | Current content |
> |---|---|---|
> | `app.html:195` | **`:243`** | `<button class="mode-btn" data-mode="community">Community</button>` |
> | `app.html:1600` | **`:1691`** | `<h2 class="page-title">Community</h2>` |
> | `app.html:1403-1410` | **`:1428-1439`** | the participants empty state |
> | `Participating Teams` subtab | **`:1433`** | `data-competition-tab="participants">Participating Teams<` |
>
> **The subtab roster changed under this task.** PR #352 replaced the Daily
> Leaderboard subtab with Live Trading, so the four are now `leaderboard`,
> `live`, `participants`, `about` (`:1431-1434`). This does not change what Task
> 10 renames, but Step 3's empty-state replacement copy needs a rewrite: it
> currently says *"The Ranking board shows AI models and baseline strategies
> only"*, which describes one board when there are now two. Suggested:
>
> > `No traders yet` / `Traders will appear here when the season opens. Both
> > boards currently show AI models, baseline strategies and house-authored
> > instructions.`
>
> Also note `:1439` still reads *"Season hasn't started yet"* — which is now
> **accidentally accurate** for Live Trading's Season 0, on a panel that belongs
> to the Competition tab. Rename it anyway; two boards make "the season" ambiguous
> and that ambiguity is worth removing, not preserving.

**Files:**
- Modify: `dashboard/frontend/app.html:195`, `:1600`, `:1403-1410`
- Modify: `dashboard/backend/tests/test_frontend_marketplace_placement.py:117-137`

**Interfaces:**
- Consumes: nothing.
- Produces: user-visible strings only.

**Do not rename** the `community` page key, `#communityView`, the `NAV_VIEW_MAP`
entries (`app.html:28-44`) or the `nav-state` localStorage value — live bookmarks
break for no user-visible gain. `?view=marketplace` already exists as a legacy
alias pointing at `community`, so this rename makes that alias the accurate one.

- [ ] **Step 1: Update the guard first**

`test_frontend_marketplace_placement.py::test_community_page_header_matches_the_nav_button`
(`:117-137`) asserts the nav button text equals the page `<h2>`. It passes only
if **both** move. Add an explicit assertion that the new label is in place:

```python
def test_nav_and_page_title_say_agent_marketplace():
    """They must move together — the pre-existing matching guard passes if both
    stay 'Community', so it cannot detect a half-done rename on its own."""
    from ._frontend_source import APP_HTML

    assert 'data-mode="community">Agent Marketplace<' in APP_HTML
    assert '<h2 class="page-title">Agent Marketplace</h2>' in APP_HTML
```

- [ ] **Step 2: Run it to verify it fails**

```bash
pytest dashboard/backend/tests/test_frontend_marketplace_placement.py -v
```

Expected: the new test FAILS.

- [ ] **Step 3: Apply the renames**

| File:line | From | To |
|---|---|---|
| `app.html:195` | `data-mode="community">Community<` | `data-mode="community">Agent Marketplace<` |
| `app.html:1600` | `<h2 class="page-title">Community</h2>` | `<h2 class="page-title">Agent Marketplace</h2>` |
| `app.html:1406` | `Season hasn't started yet` | `No traders yet` |
| `app.html:1407` | `Participating teams will appear here when the contest season opens. The Ranking board shows AI models and baseline strategies only.` | `Traders will appear here when the season opens. The board currently shows AI models, baseline strategies and house-authored instructions.` |
| `app.html` (subtab) | `data-competition-tab="participants">Participating Teams<` | `data-competition-tab="participants">Participating Traders<` |

Leave `data-competition-tab="participants"` and `#competitionParticipantsPanel`
unchanged — identifiers, not copy.

- [ ] **Step 4: Bump the cache-buster**

`app.html` itself is not versioned, but any touched asset is. This task edits
only `app.html`, so **no `?v=` bump is required.** Confirm you did not edit
`app.js`, `styles.css` or `js/leaderboard.js`:

```bash
git diff --name-only
```

Expected: only `app.html` and the test file.

- [ ] **Step 5: Run the frontend guards**

```bash
pytest dashboard/backend/tests/test_frontend_marketplace_placement.py -v
command grep -c "Participating Teams" dashboard/frontend/app.html   # expect 0
```

- [ ] **Step 6: Commit**

```bash
git status --short
git add dashboard/frontend/app.html \
        dashboard/backend/tests/test_frontend_marketplace_placement.py
git commit -m "ux(app): Agent Marketplace and Participating Traders"
```

---

### Task 11: Rebuild the shipped landing bundle

> **Mechanical, and conditional. Run this only if `dashboard/landing/src/` was
> touched.** With Tasks 5, 7 and 8 withdrawn or done, the reduced Phase 1 (1–4,
> 6, 9, 10) changes React source only in Task 6 (`lib/analytics.ts`), so the
> rebuild is needed only if that module is actually imported by a component. Task
> 9 edits `index.html`'s `<head>` by hand and Task 10 edits `app.html`, neither of
> which is Vite output.
>
> #357 already rebuilt the bundle (current: `index-XgaRai2O.js`,
> `index-BbJpUuy3.css`). The four hand-written auth blocks survived that rebuild,
> which is evidence the procedure below works — follow it exactly rather than
> improvising, and let `test_frontend_bundle_integrity.py` confirm it.

**Files:**
- Modify: `dashboard/frontend/index.html` (asset hashes only)
- Modify/Create/Delete: `dashboard/frontend/assets/*`

**Interfaces:**
- Consumes: Tasks 6–9.
- Produces: the deployed landing page.

The shipped page is the Vite build **plus four hand-written auth blocks with no
React counterpart**. Losing any of them breaks signup silently — the page renders
and the button does nothing.

- [ ] **Step 1: Build**

```bash
cd dashboard/landing && npm run build
```

- [ ] **Step 2: Copy the new assets and remove the superseded ones**

```bash
cp dist/public/assets/* ../frontend/assets/
# then delete the PREVIOUS index-*.js / index-*.css from ../frontend/assets/
```

- [ ] **Step 3: Repoint the tags, preserving the auth layer**

In `dashboard/frontend/index.html`, update the `<script>` and `<link>` to the new
content-hashed filenames. **Keep verbatim:** the auth-gate `<script>` in `<head>`,
the `#landingAuthModal` markup, `<style id="landing-auth-patch">`, and the
end-of-body auth `<script>`. No `?v=` cache-buster — the content hash is the bust.

- [ ] **Step 4: Run the integrity guard**

```bash
pytest dashboard/backend/tests/test_frontend_bundle_integrity.py -v
```

Expected: all pass. This checks every `/assets/...` reference resolves, no
superseded bundle is left behind, the four auth markers survive, and the shipped
bundle carries the current CTA label from `lib/cta.ts`.

- [ ] **Step 5: Browser verification**

Load `/` and confirm: each section renders exactly once; the board is above the
fold; the modal opens in **signup** mode from every **Start Free** button and in
**login** mode from the navbar **Sign in** (widen past 1024px — that control is
`lg:`-gated); the only console error is the `/_vercel/insights/script.js` 404,
which is expected off Vercel.

- [ ] **Step 6: Commit**

```bash
git status --short
git add dashboard/frontend/index.html dashboard/frontend/assets
git commit -m "build(landing): refresh shipped bundle"
```

---

### Task 12: Full verification and PR

> **The PR body in Step 4 is stale — rewrite it.** It advertises "`Race` renders
> the live board, promoted above the fold", which is now either superseded or
> already shipped. The reduced Phase 1 ships: C1, the six seed entries, funnel
> events, `og:image`, and the two `/app` renames. Say that instead, and keep the
> title short per the repo's convention.
>
> The verification steps themselves (Steps 1–3, 5) are unchanged and still
> correct. Step 2's expectation holds with extra force: the reduced scope adds no
> routes, so a golden-set failure means something unintended was added.

- [ ] **Step 1: Full backend suite**

```bash
pytest dashboard/backend/tests/ -q
```

Expected: green. A red test is a real regression — the suite is green
end-to-end. **Exception:** if `test_deleted_shim_is_not_importable` fails with
`DID NOT RAISE ModuleNotFoundError`, that is stale bytecode, not a regression:

```bash
rm -rf dashboard/backend/engines dashboard/backend/services
```

- [ ] **Step 2: Confirm the route contract did not move**

```bash
pytest dashboard/backend/tests/test_router_move.py dashboard/backend/tests/test_app_composition.py -q
```

Expected: green **without** editing any golden set. Phase 1 adds no routes; if
these fail, a route was added that this plan did not call for.

- [ ] **Step 3: Confirm the prod seed DB is untouched**

```bash
git status --short
git diff --stat origin/main -- dashboard/storage/
```

Expected: no output from the second command.

- [ ] **Step 4: Open the PR**

```bash
git push -u origin feat/participatory-competition-phase-1
gh pr create --title "feat: leaderboard-first landing + Open Track seed field" --body "$(cat <<'EOF'
Phase 1 of the participatory competition design. Display only — no user entries yet.

- C1: `LLMAgentStrategy` forwards `strategy_prompt`
- Six house-authored Open Track seed entries, gated on the instruction-sensitivity probe
- `Race` renders the live board, promoted above the fold
- Funnel events, `og:image`, Community→Agent Marketplace, Teams→Traders

Spec: `docs/superpowers/specs/2026-08-09-participatory-competition-design.md`
Probe: `docs/superpowers/probe-results/2026-08-09-instruction-sensitivity.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Post-merge prod verification**

Merging to Open-Finance-Lab `main` auto-deploys. After the deploy completes:

```bash
curl -s https://agentictrading.onrender.com/api/v1/leaderboard | python -c "
import json,sys
p=json.load(sys.stdin)
ids=[e.get('entry_id') for e in p['entries']]
print('open track:', [i for i in ids if str(i).startswith('open_')])
"
```

Expected: six `open_*` ids. **If empty, `OPENROUTER_API_KEY` is unset in the
Render dashboard** — the seeds' curves come from the committed DB, so an empty
result means the deploy reset it and the entries could not recompute.

---

## Deferred: Phases 2 and 3

Each gets its own plan, written when its prerequisites resolve.

**Phase 2 (Competition qualifier)** is blocked on the Task 3 gate. Its pinned
model, attempt grant and cost tables all change if the probe forces DeepSeek, so
detailed tasks written now would be discarded. It covers C3/C4/C5/C7,
`leaderboard_entries` + `leaderboard_attempts` with Postgres twins, the attempt
ledger, email verification at signup, the checklist card, the Submit action, the
two-worker queue, and graduation into Loop A.

**Phase 3 (Live Trading seasons)** is blocked on Phase 2 — Live Trading entry
requires a completed Competition attempt. It covers C8, `forward_positions`,
derived **two-week** seasons, the season-close email, the share-card download,
and the notification channel abstraction with Discord stubbed.

Four items to carry into planning:

1. **Worker concurrency against issue #202**, which reports blocking sync I/O on
   exactly the leaderboard routes. The spec says two workers; that must be
   measured, not assumed.
2. **`BREVO_API_KEY` becomes a hard signup dependency** once verification gates
   account creation. It must be set in Render *before* that PR merges, or every
   signup breaks.
3. **Added 2026-08-15 — consider pulling C8 out of Phase 3 and landing it
   first.** The Live Trading board is already in prod as a Season 0 preview that
   has never advanced. C8 against the existing house roster needs no user
   entries, no ledger, and nothing from Phase 2, and it does not depend on the
   Phase 0 gate — whether *instructions* move returns is a separate question from
   whether the board can advance. The Phase 2→3 ordering constraint is about user
   entry, and a house-only advance has no users in it. See §Rollout in the spec
   for the invariants it must respect and the cost question it must answer first.
4. **Added 2026-08-15 — the instruction-lock trade-off is reopened.** The
   two-week cadence doubles how long a user waits to correct a bad instruction,
   and the original "user's call" decision rested on a one-week season. Resolve
   it in Phase 3 planning (one pre-advance edit? a mid-season re-entry slot?)
   rather than inheriting a decision made under different terms.
