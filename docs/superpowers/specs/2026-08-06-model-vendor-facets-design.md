# Model-vendor facets, catalog expansion, and the model-vocabulary fix

**Date:** 2026-08-06
**Status:** Design approved, pending implementation plan

## Goal

Tempt users into creating more agents and running them, by making the *model* an
explicit, browsable dimension of the platform — and by offering two concrete
"make another one" actions built on it.

The professor's suggestion was to shelve by closed-source model / open-source
model / vendor. This design adopts that as a **facet on Community**, expressed
primarily as vendor with licence as a card badge, plus the catalog content and
the conversion hooks needed to make the facet non-decorative.

## Background: what already exists

Shelving today (shipped in #311 → #313 → #314) has one axis, **market**:

- `dashboard/backend/domain/agents/taxonomy.py` — `AgentCategory =
  Literal["us_stocks", "cn_ashares"]`, a stored column, published into
  `openapi.json`.
- `app.js` `MARKET_LABELS` — slug → display name, the single source for the
  Community chips, the Stocks-shelf chips, card submeta and the Configure picker.
- My Agents renders two live shelves (`Stocks`, `For Developers: Connected
  Agents`) plus two locked inert rows (Crypto, Futures). Market is a chip row
  *inside* Stocks, not a shelf.

The market axis needed a database column because nothing in an agent record
implies its market. **The model axis needs none**: `model_name` is already on
every agent and every marketplace template. This taxonomy is a pure derivation —
no migration, no Postgres-twin parity work, no `openapi` enum deploy gate.

`app.js:1902` already carries `MODEL_PROVIDER_LABELS`, a prefix table mapping
`anthropic/` → `"Powered by Claude"` across eight vendors. It renders card
submeta today and has never been used to filter.

### The content problem

The Community catalog is six Claude templates and one Nemotron. A vendor facet
built on it today would show two populated chips. The Create/Configure picker
(`app.html:799`), by contrast, already offers five vendors — so users can
*already* build across the vendor spread; only the catalog is a monoculture.

## Non-goals

- No vendor chip row on My Agents. A user with 3–8 agents does not need vendor
  filtering, and the existing market chip row there is already lightly used. Add
  it if and when fleets get large.
- No new shelf *rows*. Shelves multiply badly — 2 markets × 3 vendor groups is 6
  rows — and `AGENT_SHELVES`' own comment records that a mis-listed shelf trips
  the "some grid is missing" guard and silently aborts the entire My Agents
  render.
- No `model_name` whitelist on the API. See §3.
- No crypto/futures work. Those rows stay locked and inert.

---

## 1. The model-vocabulary fix (prerequisite)

### The defect

`app.html:360` holds a nine-option model `<select>` (`#modelSelect`). It is
**not** dead markup: `syncBacktestModelFieldMode` (`app.js:1655`) unhides it when
the market-data source is iFinD A-share (`IFIND_ASHARE_SOURCE = 'ifind_ashare'`),
where it doubles as the rule-based-vs-LLM decision-source picker. Everywhere else
it is hidden behind a read-only echo of the agent's saved model.

Its option list has drifted from the platform's real model set. It offers six
models that do not exist here — `claude-opus-4.7`, `gpt-5.2`, `gpt-5-mini`,
`deepseek-v4-flash`, `gemini-3.5-flash`, `gemini-2.5-pro` — and omits four that
do: `gpt-5.5`, `gemini-3.1-pro-preview`, `qwen3.7-plus`, and the Nemotron used by
the AI Hedge Fund runtime.

There is no *vocabulary* conflict — `normalizeBacktestModelId` (`app.js:1615`)
strips the provider prefix and rewrites `-4-5` → `-4.5`, so
`anthropic/claude-haiku-4-5` correctly matches `claude-haiku-4.5`. The defect is
membership, and it has a silent-failure mode:

`resolveBacktestModelRequest` (`app.js:1640`) guards against submitting a value
left behind by a previously-selected agent, but the guard is
`!backtestModelPickerIsLiveControl()` — it only fires on the hidden path. On the
**live iFinD path**, an agent whose `model_name` matches no option leaves
`syncModelSelectFromAgent` a no-op, and the run submits whatever the last agent
left in the select. No error, no warning, wrong model in the run record.

This blocks §4: `qwen3.7-plus` matches no option, and iFinD A-share is the
A-share backtest path, so the proposed Qwen A-share template would hit exactly
this combination.

### The fix

Introduce a single frontend source of truth for runnable models:

```js
const SUPPORTED_MODELS = [
  { slug: 'anthropic/claude-haiku-4-5',        label: 'Claude Haiku 4.5',       vendor: 'anthropic' },
  { slug: 'anthropic/claude-sonnet-4-6',       label: 'Claude Sonnet 4.6',      vendor: 'anthropic' },
  { slug: 'openai/gpt-5.5',                    label: 'GPT-5.5',                vendor: 'openai'    },
  { slug: 'google/gemini-3.1-pro-preview',     label: 'Gemini 3.1 Pro Preview', vendor: 'google'    },
  { slug: 'deepseek/deepseek-v4-pro',          label: 'DeepSeek V4 Pro',        vendor: 'deepseek'  },
  { slug: 'qwen/qwen3.7-plus',                 label: 'Qwen3.7 Plus',           vendor: 'qwen'      },
];
```

Both `<select>` elements are built from it: the Configure picker
(`app.html:799`) and the backtest picker (`app.html:360`). The backtest one keeps
its existing `rule_based` option prepend and its `disabled`/hint behaviour in
`syncIFindModelControl` unchanged.

`normalizeBacktestModelId` is **kept**, not deleted. Stored `model_name` values
in the wild are messy (`'GPT-5.5'`, `'local-model'`, `'rule-based'`,
`'rule-based-demo'`, `'gpt-5.2'` on pre-existing agents), so the normalizer stays
as the bridge for legacy values. It simply stops being load-bearing for current
ones.

**Close the silent-leftover hole.** `syncModelSelectFromAgent` gains: when the
agent's model matches no option *and* the picker is a live control, inject an
option carrying the agent's actual model and select it, rather than leaving a
previous agent's value in place. An unrepresentable model must submit itself, or
visibly fail — never a stale neighbour's value.

The AI Hedge Fund runtime's Nemotron is deliberately **not** in
`SUPPORTED_MODELS`: it is not user-selectable, it is a property of a hosted
runtime, and `syncBacktestModelFieldMode:1663` already renders that case as
`"AI Hedge Fund — hosted runtime"`. It appears in the vendor table (§2) so its
card gets a chip and a badge, but not in the picker.

---

## 2. The vendor taxonomy

`MODEL_PROVIDER_LABELS` is promoted from a label lookup to the source of truth
for vendor identity and licence:

```js
const MODEL_VENDORS = [
  { key: 'anthropic', prefix: 'anthropic/',      label: 'Claude',   licence: 'closed' },
  { key: 'openai',    prefix: 'openai/',         label: 'GPT',      licence: 'closed' },
  { key: 'google',    prefix: 'google/',         label: 'Gemini',   licence: 'closed' },
  { key: 'deepseek',  prefix: 'deepseek/',       label: 'DeepSeek', licence: 'open'   },
  { key: 'qwen',      prefix: 'qwen/',           label: 'Qwen',     licence: 'open'   },
  { key: 'nvidia',    prefix: 'nvidia/nemotron', label: 'Nemotron', licence: 'open'   },
  { key: 'meta',      prefix: 'meta-llama/',     label: 'Llama',    licence: 'open'   },
  { key: 'xai',       prefix: 'x-ai/',           label: 'Grok',     licence: 'closed' },
];
```

Declaration order is chip order, mirroring how `MARKET_LABELS`' key order mirrors
the `AgentCategory` Literal.

`formatModelProviderLabel` becomes a derivation over this table. Its output stays
byte-identical (`"Powered by Claude"`, unmatched → `"AI-powered"`, raw slug never
leaked), so existing assertions continue to hold.

A new `agentVendorKey(agent)` mirrors `agentMarketKey`'s contract exactly:
**unmatched → `''`**, meaning the agent stays visible under `All` and is excluded
only by an explicit vendor chip. That policy is already documented and tested at
`app.js:529-539`; reusing it verbatim introduces no new failure mode.

Matching is by prefix, not exact slug, so a new model version under a known
vendor needs no table entry.

---

## 3. Navigation: stacked facets

Community gets a **second chip row** beneath the existing market row. The two
filters AND together (`U.S.` + `DeepSeek`).

The one deliberate asymmetry with the market chips: **vendor chips render only
for vendors present in the loaded catalog.** Market chips are hardcoded from
`MARKET_LABELS` because that is a closed, backend-validated enum. Vendors are
open-ended, and hardcoding all eight would ship permanently-empty chips. Chips
are therefore `MODEL_VENDORS` ∩ vendors-present, in `MODEL_VENDORS` order.

Chip rows are built once and then only toggle state, following
`renderMarketplaceCategoryChips`' existing pattern — the grid re-renders on every
keystroke of the search box, and rebuilding `innerHTML` per keystroke would blow
away the focused chip.

Three empty states must stay distinguishable (a concern the codebase already
records at `app.js:1388`):

| State | Message |
|---|---|
| Search matched nothing | existing search-empty copy |
| A single chip has nothing on it | existing chip-empty copy |
| **Facet combination empty** (new) | "No templates match both filters" + a **Clear filters** action |

Selecting a vendor chip resets the grid's page index, matching
`setAgentMarketFilter`'s existing behaviour.

**Why facets rather than shelves:** facets combine multiplicatively without
multiplying rows. A third axis later is one more chip row, not twelve more
shelves.

### Licence badge

Cards on open-weight models get a small `Open-source model` badge. Closed models
get **nothing** — absence is not a negative claim, and a "Closed" label reads as
a warning about someone else's product. Licence lives in `MODEL_VENDORS` so it
can never drift from the vendor it describes.

---

## 4. Catalog expansion: four new templates

Content only — no code. Added to `dashboard/config/marketplace.json` in the
existing single-step `pipeline` shape, in the same plain-language register as
Balanced Starter.

| Template | `model_name` | `category` |
|---|---|---|
| Contrarian Dip Buyer — buys oversold names, trims into strength | `openai/gpt-5.5` | `us_stocks` |
| Sector Rotator — concentrates into the leading sectors | `google/gemini-3.1-pro-preview` | `us_stocks` |
| Volatility Guard — cuts exposure when volatility spikes | `deepseek/deepseek-v4-pro` | `us_stocks` |
| A-Share Momentum (T+1) — rides leaders, respects T+1 settlement | `qwen/qwen3.7-plus` | `cn_ashares` |

Result: catalog 7 → 11, all six live vendor chips populated, `cn_ashares` 1 → 2.

Two pairings are deliberate rather than arbitrary:

- **Qwen on A-shares.** Qwen is Alibaba's model and `cn_ashares` is the thinnest
  shelf; one card fills both gaps and the two axes reinforce instead of
  competing. The template must respect T+1 (see the A-share T+1 work in #272 /
  #289) — same-day-bought shares are not sellable.
- **DeepSeek on Volatility Guard.** DeepSeek is the only LLM on the leaderboard
  that beat the passive baselines, so this card ships with a verifiable claim
  behind it rather than novelty. It gets the most prompt care of the four.

Prompts are straightforward, not elaborate — the point is populated, runnable
variety, not strategy research.

---

## 5. Hook A — choose the model when cloning (Community)

**Split button, not a dialog on the primary click.** `Add to My Agents` keeps its
exact current one-click behaviour using the template's default model; a secondary
`Choose model ▾` affordance sits beside it, offering `SUPPORTED_MODELS` with the
template's default preselected. Model is the **only** thing it changes — no
rename, no capital, no pipeline edits. Those already have a home in Configure,
and a second half-Configure inside a clone menu is how two editing surfaces start
drifting apart.

Rationale: the primary CTA is the conversion click. Putting a picker in front of
it adds friction to precisely the interaction we most want to succeed. The split
button exposes the choice without taxing the default path.

### Backend

`CloneMarketplaceBody` (`api/routers/agents.py:239`) gains
`model_name: Optional[str]`, threaded into `clone_marketplace_template`
(`domain/agents/service.py:385`) to override the hardcoded
`template.get("model_name")` at line 409. Blank or omitted → template default.

**No whitelist.** `create_agent` does not validate `model_name` today, so
validating only on clone would be inconsistent, and a `Literal` here would drag
in the `openapi` enum deploy gate that #313 had to discharge. This follows the
lenient-category precedent documented two lines below, at `service.py:422`: an
unrecognized value must not reject the clone. This adds no attack surface that
`POST /agents` and `PATCH /agents/{id}` do not already have.

---

## 6. Hook B — "Run on another model" (My Agents)

On an agent card whose status badge is `BACKTESTED` or `PAPER TRADING` — i.e. the
user has already demonstrated intent — a `Run on another model` action clones
that agent onto a different model, named `<name> (DeepSeek)`.

It offers `SUPPORTED_MODELS` **minus the agent's current model** — an entry that
duplicates an agent onto the model it already runs is a no-op the user has to
reason about. If the agent's current model is not in `SUPPORTED_MODELS` (a legacy
or hosted-runtime value), the full list is offered and nothing is filtered out.
The generated name uses the vendor label, and collides freely: two clones onto
DeepSeek both read `<name> (DeepSeek)`. Names are not unique anywhere else in
this product either, and de-duplicating them would mean a lookup for a cosmetic
gain.

This is the highest-yield hook: it turns one agent into three and one run into
three, among users who have already succeeded once, and it is independent of the
Community catalog.

### Backend

New endpoint `POST /api/agents/{agent_id}/duplicate`, body `{ model_name,
name? }`. Server-side so the pipeline copy is atomic and the ownership check is
one path. It reuses `create_agent` plus the pipeline write — the same two steps
`clone_marketplace_template` already performs — and applies the same
`portfolio_service.ensure_cash_for_new_agent` guard, so an over-allocated user
gets the existing 400 rather than a half-created agent.

Ownership is enforced via `_require_owner_context`, matching every other
mutating agent route.

**It does not auto-launch a backtest.** It lands the user on the new agent with
Run primed. Auto-firing spends LLM credits on a click the user did not frame as
"run"; silently costing money is not a thing to be clever about.

---

## 7. Testing

Frontend guards follow the existing `dashboard/backend/tests/_frontend_source.py`
source-parsing pattern used by `test_frontend_shelves.py`.

- **Every `model_name` in `marketplace.json` matches a `MODEL_VENDORS` prefix.**
  The highest-value guard: it catches a future template silently landing under
  "AI-powered" with no chip and no badge, which is otherwise invisible.
- **Every `SUPPORTED_MODELS` slug matches a `MODEL_VENDORS` prefix**, and every
  `marketplace.json` `model_name` is either in `SUPPORTED_MODELS` or is a hosted
  runtime's model. Catches the §1 drift recurring.
- Licence classification pinned per vendor — a wrong badge is a factual claim
  about someone else's product.
- Vendor chips are derived, not hardcoded (assert no literal vendor list in the
  chip builder).
- Both `<select>` elements are built from `SUPPORTED_MODELS` (assert no literal
  `<option value="claude-` in `app.html`).
- Facet-combination-empty state renders its own copy, distinct from the two
  existing empty states.
- `syncModelSelectFromAgent` injects-and-selects for an unrepresentable model on
  the live path, and never submits a leftover value — the §1 regression test.
- Backend: clone honours `model_name`, ignores blank, does not 422 on an unknown
  value; duplicate copies the pipeline, enforces ownership, and enforces the cash
  guard.

No store signature changes are expected (no new column), so no Postgres-twin
parity work — but if any store call gains a field, **both twins** change together
per `test_store_twin_parity.py`.

Cache busters: `app.js`, `styles.css`, and `agent-editor.js` query strings all
bump.

---

## 8. Delivery: four PRs

| # | Scope | Depends on |
|---|---|---|
| 1 | Model-vocabulary fix (§1) — frontend only, bug fix | — |
| 2 | Backend: clone `model_name` override + duplicate endpoint (§5, §6) | — |
| 3 | Content: four new templates (§4) | 1 (Qwen path) |
| 4 | Frontend: vendor taxonomy, facets, badge, both hooks (§2, §3, §5, §6) | 2 **live in prod**, and 1 |

1 and 2 are independent and can proceed in parallel.

PR 4 must not merge until PR 2's endpoints are actually serving in production.
Render lags Vercel by 10–40 minutes and merging to `main` auto-deploys, so the
gate is verified by probing prod's `/openapi.json` for the new `duplicate` route
and the `model_name` field on `CloneMarketplaceBody` — not by PR 2 showing as
merged. This is the same gate #313 discharged against #311.

Per repo convention, if PR 4 opens before that gate clears it opens as a
**draft** with the gate as an imperative in the first line of the body. `main`
has no branch protection and open clean PRs get merged by others.
