# Asset-Class Shelves — Design

**Date:** 2026-08-05
**Status:** approved (owner, 2026-08-05)
**Supersedes:** the shelf taxonomy shipped in PR #311 (`cd421fc`) + PR #313 (`5fbece2`), both merged earlier today.

## Problem

My Agents ships four sections that classify agents along **three different axes at once**:

| Section | Axis it actually uses |
|---|---|
| Prompting LLMs | *how the agent decides* (hosted AI vs. code) |
| U.S. Stock Trading | *geography* |
| China A-Share Trading | *geography* |
| For Developers: Connected Agents | *integration type* |

There is no single question a user can ask themselves to know which section an agent belongs on. The owner's instruction is that the organizing principle should be **what the agent trades** — stocks, crypto, futures — not where it trades.

Two presentation defects compound it:

1. **The sections have no visual weight.** `.agents-category` (`styles.css:9289`) is five declarations: a `margin-top`, an `<h3>`, a `<p>`. No panel, no frame, no background. Against the bordered cards beneath them the headers read as loose prose — the owner's description was "just like a plain html page written with a couple `<p></p>`".
2. **The empty-state "Community" call to action is a bare hyperlink.** `app.js:1382` emits `<a href="#" class="agents-empty-community-link">Community</a>`, and that class has **no CSS rule anywhere**, so it inherits generic anchor styling. It is the primary path from an empty shelf to the content that fills it, and it does not look actionable.

## Capability truth (gates every shelf in this design)

Verified at source, not assumed:

- `_MARKET_PROFILES` (`dashboard/backend/infrastructure/market_data/profiles.py:83`) holds exactly four profiles. Every one is equities: `(alpaca, djia_30)` and `(vnpy_simulation, djia_30)` are `market="US"`; `(ifind_ashare, a_share_demo_6)` and `(ifind_ashare, csi300_sample_20_2026h2)` are `market="CN"`.
- **Crypto is a price ticker only.** `quotes.py:456` `get_crypto_quote` fetches BTC/ETH/XRP/SOL from CoinGecko for display. `dashboard/backend/domain/backtesting/` contains zero references to `quotes`.
- **Futures exist nowhere.** `vnpy_simulation` is synthetic *US equity* bars, not contracts. There is no contract, margin, or expiry modelling in the repo.

So **stocks is the only asset class the engine can backtest**, and that is not expected to change within this piece of work.

## Owner decision

Presented three options; the owner chose **locked Crypto and Futures shelves** — rendered, visibly unavailable, non-interactive.

This **knowingly reverses** a finding from the 2026-08-04 persona research recorded in `docs/superpowers/plans/2026-08-04-audience-language-and-shelves.md` (Global Constraints, "Capability truth"), which concluded that a clickable shelf with nothing behind it "would convert caution into distrust" and directed: *"No Crypto or Futures shelf, tag, or copy in v1 — cut, not 'coming soon'."*

The reversal is deliberate and is the owner's call. The mitigation this design commits to is that the locked shelves are **inert, not merely empty**: no grid, no chips, no focus stop, `aria-disabled`, dashed and muted — they must read as *not built yet*, never as *built and broken*.

## Design

### 1. Shelves

| # | Shelf | State | Contents |
|---|---|---|---|
| 1 | **Stocks** | live | every built-in agent; market chips `All` · `U.S.` · `China A-Share` filter within it |
| 2 | **Crypto** | locked | none, ever |
| 3 | **Futures** | locked | none, ever |
| 4 | **Connected Agents** | live | every non-`builtin` agent (developer surface) |

**"Prompting LLMs" retires as a section.** It is not an asset class; it describes how an agent decides. That axis moves onto the card as a badge — **Hosted AI** vs. **Your own code**. This reverses owner decision #3 in the 2026-08-04 plan ("*'Prompting LLMs' ships verbatim*"), raised explicitly at approval time and accepted.

Consequence to preserve: Prompting LLMs was also the **onboarding surface** — the shelf a brand-new user's auto-provisioned agent lands on. Stocks inherits that role. Its empty state must therefore keep the onboarding voice ("You don't have any agents yet. Create one and test your first trading idea."), not the "add one from Community" voice used by the old market shelves.

### 2. Data model — no new column, no migration

The stored slugs `us_stocks` and `cn_ashares` **already encode both axes** (asset = stocks, market = US/CN). The market chips are derived from them. Nothing new is stored.

- `AgentCategory` (`taxonomy.py`) becomes `Literal["us_stocks", "cn_ashares"]`. `prompting_llms` is dropped.
- **The locked shelves get no slug at all.** Nothing can ever be assigned to Crypto or Futures, so they require no `Literal` member, no DDL, no store-twin change, no `openapi.json` change. They are frontend chrome. This is the single largest cost saving in the design and the reason locked-and-inert beats real-but-empty.
- **Legacy `prompting_llms` rows** (only agents created or cloned since PR #313 went live at 14:01 UTC today) need no backfill. `agentShelfKey` (`app.js:520`) already falls back for unrecognised slugs; that fallback repoints to Stocks. The read path is safe: `category` is typed only on *request* models (`agents.py:60`, applied at `:85`/`:118`); the response builds a plain dict (`:290`), so a retired slug in an old row passes through reads without a validation error.
**Shelf keys and category slugs stop being the same thing.** Today they are 1:1 — `AGENT_SHELVES`' keys *are* the category slugs, and `SHELF_LABELS` is derived from the shelf titles. Stocks holding two categories breaks that, so the two concepts separate:

- `AGENT_SHELVES` keys become shelf ids: `stocks`, `crypto`, `futures`, `external`. Only `stocks` and `external` receive agents.
- A new `MARKET_LABELS` map (`us_stocks` → `"U.S."`, `cn_ashares` → `"China A-Share"`) becomes the single category-slug → display-name source, consumed by the Stocks market chips, the Community chips, the Community card submeta, and the Configure picker (`window.AGENT_SHELF_LABELS` re-points at it).
- `test_shelf_labels_map_is_derived_from_agent_shelves_not_duplicated` encodes the old 1:1 assumption and must be re-pointed at `MARKET_LABELS`. Its *intent* — one declaration, never hand-typed twice — is preserved, and remains binding.

`AGENT_CATEGORY_ORDER`'s declaration-order-is-shelf-order contract (documented in `taxonomy.py`) still holds; it now orders the **market chips** within Stocks rather than top-level sections.

**A built-in with a legacy, blank, or unrecognised category** renders on Stocks and appears under the `All` chip only — it matches no market chip, because the platform genuinely does not know which market it targets. It is never hidden.

**The count pill reports the shelf's total**, ignoring the active market chip and the search box, so the number does not move underfoot while filtering. Chip and search filtering affect the grid only.

### 3. Visual treatment

`.agents-category` becomes a real container:

- Panel on `--bg-surface`, `1px solid var(--border-color)`, rounded, padded.
- A header strip — asset icon, title, right-aligned count pill ("7 agents") — separated from the grid by a hairline rule.
- The subtitle stays, as secondary text inside the header strip.

**Locked shelves** (`Crypto`, `Futures`): dashed border, muted foreground, lock glyph, a right-aligned "Not yet available" tag, one line of plain description, `aria-disabled="true"`, no grid element, no chips, no footer, not focusable, no hover affordance.

**Market chips** inside Stocks reuse the existing Community chip styling so the two surfaces visibly match — the same taxonomy should look the same wherever it appears.

### 4. Community control

`app.js:1382` becomes a real `<button type="button">` styled as a small secondary button, keeping the existing `data-community-category` attribute and its event delegation. Applies to the Stocks empty state.

The top-nav `Community` tab is **left alone**: it is already a `<button>` element, and `.mode-btn`'s borderless/transparent treatment is a deliberate tab pattern with an `.active` state. Recorded here because the owner's report ("the button 'community' is also a hyperlink rather than a button") could have meant either; the empty-state anchor is the one that is literally an `<a>`.

### 5. Ripple

Follows automatically from `AGENT_SHELVES` — no separate edit, and guarded by an existing test that forbids re-declaring the labels:

- Community category chips (`app.js:1858`)
- Configure screen's shelf `<select>` (`js/agent-editor.js:583`, via `window.AGENT_SHELF_LABELS`)

Requires explicit edits:

- `dashboard/config/marketplace.json` — the two `prompting_llms` templates (`balanced-starter`, `momentum-scout`) both run DJIA, so both become `us_stocks`.
- `dashboard/backend/domain/agents/taxonomy.py` + `tests/test_agent_taxonomy.py`.
- `dashboard/frontend/app.html` — four `agents-category` sections replaced by two live sections plus two locked rows; per-shelf element ids change.
- `dashboard/frontend/app.js` — `AGENT_SHELVES`, `agentShelfKey` fallback, market-chip state, empty-state copy, the Community button.
- `dashboard/frontend/styles.css` — the panel treatment and the locked-row treatment.
- `dashboard/backend/tests/test_frontend_shelves.py` — pins the four current headers, the four shelf keys, and the four-section count; must be rewritten alongside.
- `app.html` cache-buster bump (currently `app.js?v=` — read at implementation time; it has moved several times this week).

### 6. Testing

Same shape as the shelves already use — pytest source guards over the shipped frontend via `dashboard/backend/tests/_frontend_source.py`, comment-stripped, scoped to the narrowest branch.

- Backend: `AGENT_CATEGORIES == {"us_stocks", "cn_ashares"}`; `coerce_category("prompting_llms")` raises; `normalize_category("prompting_llms") is None`; create/patch with `prompting_llms` → 422.
- Frontend guards: the two live shelf headers and the two locked rows present; "Prompting LLMs", "U.S. Stock Trading", "China A-Share Trading" absent as section headers; `AGENT_SHELVES` carries the live keys; locked rows carry `aria-disabled`; the empty-state emits `<button` and not `<a href="#"` for the Community control; `DEFAULT_AGENT_PROVISION_GUARD_PREFIX` byte-identical to `main`.
- Marketplace: every template `category` ∈ `AGENT_CATEGORIES`; call `reload_marketplace_catalog()` in any test touching the catalog (it is `lru_cache`d).
- Full suite green before the PR opens.

## Repo invariants this work must not break

Carried forward from the 2026-08-04 plan, all still live:

- **Seed DB:** never `git add -A`. The committed `dashboard/storage/data/backtest.db` is the prod database; a bare backend import runs lazy `ALTER`s against it. Check `git status` shows no `backtest.db` / `-wal` change before every commit.
- **Zero new routes.** `category` rides the existing create/patch bodies. `test_app_composition.py::EXPECTED_FULL_CONTRACT` and `test_router_move.py::EXPECTED_AGENT_ROUTES` must stay untouched and green.
- **Never touch** `DEFAULT_AGENT_PROVISION_GUARD_PREFIX` / `defaultAgentProvisionGuardKey()` (`app.js:230–251`) — a prefix change re-provisions duplicate agents for every existing user.
- **`patchAgent`'s truthy `[]`** (`js/agent-editor.js:823`) is deliberate — it clears the pipeline. Never "tidy" it.
- **`#modelSelect` keeps its 9 options** (`test_ifind_ashare_frontend.py`, `test_vnpy_simulation_frontend.py` pin it).
- **Store twins** change in the same commit with literal-string DDL — not engaged by this design (no column change), but binding if that assumption breaks.
- Source-guard tests strip comments before `not in` assertions and scope to the narrowest branch.

## Out of scope

- **User-facing docs go stale again.** PR #313 refreshed `docs/source/lab/*` for the shelf names this design retires. Per the owner's docs policy these are not edited mid-session; they are surfaced as an explicit follow-up. RTD has no CI build — run `sphinx-build -n -E` locally when they are edited.
- Building actual crypto or futures support: a real bar source, a `MarketProfile`, 24/7 session handling, and (for futures) contract/margin/expiry modelling. None of it exists; the locked shelves are a roadmap signal, not a stub.
- Landing-page copy (`dashboard/landing/src`) — untouched by this change.
