# Asset-Class Shelves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-axis My Agents from geography (U.S. / China A-Share) to asset class (Stocks live; Crypto and Futures locked), give each shelf a real visual container, and turn the empty-shelf "Community" link into a button.

**Architecture:** The stored category slugs (`us_stocks`, `cn_ashares`) already encode both asset and market, so nothing new is persisted — the asset axis becomes one live `stocks` shelf, and the market axis becomes a chip row inside it. Crypto and Futures are static, inert HTML rows with no JS behaviour and no category slug, because no engine can run them. `SHELF_LABELS` is replaced by `MARKET_LABELS`, the single declaration of category-slug → display-name consumed by the market chips, the Community chips, the agent-card submeta, and the Configure picker.

**Tech Stack:** Python 3 / FastAPI / Pydantic `Literal` (backend taxonomy), vanilla JS + static HTML/CSS with no build step (frontend), pytest source-guard tests over the shipped frontend.

**Spec:** `docs/superpowers/specs/2026-08-05-asset-class-shelves-design.md`

**Worktree:** `.claude/worktrees/asset-shelves`, branch `ux/asset-class-shelves`. All paths below are relative to that worktree root. Run pytest from the worktree root.

## Global Constraints

- **Never `git add -A` in this repo.** A bare backend import runs lazy `ALTER`s against the committed production seed DB `dashboard/storage/data/backtest.db`. Every commit stages named paths only, and `git status` must show no `backtest.db` / `backtest.db-wal` change before committing.
- **Zero new routes.** `test_app_composition.py::EXPECTED_FULL_CONTRACT` and `test_router_move.py::EXPECTED_AGENT_ROUTES` must stay untouched and green.
- **Never touch `DEFAULT_AGENT_PROVISION_GUARD_PREFIX` / `defaultAgentProvisionGuardKey()`** (`dashboard/frontend/app.js:230–251`). A changed prefix re-provisions a duplicate starter agent for every existing user.
- **Never "tidy" `patchAgent`'s truthy `[]`** (`dashboard/frontend/js/agent-editor.js:823`) — it deliberately clears the pipeline.
- **`#modelSelect` keeps its 9 options** (pinned by `test_ifind_ashare_frontend.py`, `test_vnpy_simulation_frontend.py`).
- **Source-guard tests strip comments before `not in` assertions** and scope to the narrowest branch (function body, or one section of `app.html`) — never the whole file.
- **No store-twin work.** This change adds no column and no DDL. If that assumption ever breaks, both twins change in the same commit with literal-string DDL.
- **The frontend is dark-theme only** (`:root` tokens; zero `data-theme="light"` / `prefers-color-scheme` rules in `styles.css`). Do not add light-mode variants.
- **Locked shelves have no category slug, ever.** `coerce_category("crypto")` and `coerce_category("futures")` must keep raising. A slug would make them selectable in Configure and cloneable from Community while no engine can run them.
- **Verbatim copy** (exact strings, do not paraphrase):
  - Stocks title: `Stocks`
  - Stocks subtitle: `Trade U.S. blue-chip and Chinese A-share stocks, tested hour by hour on real market data.`
  - Crypto title: `Crypto`; subtitle: `Round-the-clock crypto backtesting isn't built yet. Nothing here can be run.`
  - Futures title: `Futures`; subtitle: `Futures contracts aren't built yet. Nothing here can be run.`
  - Locked tag: `Not yet available`
  - Connected shelf keeps its shipped title verbatim: `For Developers: Connected Agents`
  - Market labels: `U.S.` and `China A-Share`
  - Card decision-axis labels: `Hosted AI` and `Your own code`

### Deviation from the spec, and why

Spec §2 says `AGENT_SHELVES`' keys become `stocks`, `crypto`, `futures`, `external`. **This plan puts only the two live shelves (`stocks`, `external`) in `AGENT_SHELVES`**; Crypto and Futures are static HTML only.

Reason: `AGENT_SHELVES` exists to drive rendering. Five call sites iterate it (`agentGridPage` init ×2, `renderAgentsError`, `renderAgentCategories`, the chip builder), and locked shelves have no grid/footer/empty element. Listing them would force a `!shelf.locked` filter at every site, and one missed filter makes `renderAgentCategories`' `if (shelves.some(({ grid }) => !grid)) return;` fire — silently aborting the entire My Agents render with no error. The spec's intent (one declaration of shelf order, locked rows are pure chrome) is preserved; the locked rows' order is their order in `app.html`, and Task 3's guards pin them there.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `dashboard/backend/domain/agents/taxonomy.py` | The `AgentCategory` `Literal` — now markets, not shelves | 1 |
| `dashboard/config/marketplace.json` | Template categories; the two `prompting_llms` entries move to `us_stocks` | 1 |
| `dashboard/backend/tests/test_agent_taxonomy.py` | Whitelist + coercer guards; locked-shelf-has-no-slug guard | 1 |
| `dashboard/backend/tests/test_agents_api.py` | Catalog ordering guard (leading shelf changed) | 1 |
| `dashboard/frontend/app.js` (config block, ~476–546) | `MARKET_LABELS`, `AGENT_SHELVES`, `agentShelfKey`, `agentMarketKey`, `agentTypeLabel` | 2 |
| `dashboard/frontend/js/agent-editor.js` | Configure picker's load-failure fallback labels | 2 |
| `dashboard/frontend/app.html` (`#agentsCategories`, editor field, cache-busters) | Shelf markup: 2 live sections + 2 locked rows | 3 |
| `dashboard/backend/tests/test_frontend_fast_boot.py` | Skeleton-per-grid guard (grid ids changed) | 3 |
| `dashboard/frontend/app.js` (render loop + Community chips) | Market filtering, count pill, empty states, Community button | 4 |
| `dashboard/backend/tests/test_frontend_shelves.py` | The whole shelf guard suite — rewritten across tasks 2–4 | 2, 3, 4 |
| `dashboard/frontend/styles.css` | Panel treatment, locked rows, count pill, Community button | 5 |

---

## Task 1: Backend taxonomy — retire `prompting_llms`, keep two markets

**Files:**
- Modify: `dashboard/backend/domain/agents/taxonomy.py:1-30`
- Modify: `dashboard/config/marketplace.json` (2 values)
- Test: `dashboard/backend/tests/test_agent_taxonomy.py`
- Test: `dashboard/backend/tests/test_agents_api.py:698-726`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `AgentCategory = Literal["us_stocks", "cn_ashares"]`; `AGENT_CATEGORY_ORDER == ("us_stocks", "cn_ashares")`; `AGENT_CATEGORIES == frozenset({"us_stocks", "cn_ashares"})`. `coerce_category`, `normalize_category`, `category_sort_rank` keep their existing signatures and behaviour. Task 2's `MARKET_LABELS` mirrors `AGENT_CATEGORY_ORDER`'s declaration order.

- [ ] **Step 1: Update the whitelist test to the two markets**

In `dashboard/backend/tests/test_agent_taxonomy.py`, replace line 14:

```python
def test_categories_whitelist():
    assert AGENT_CATEGORIES == {"us_stocks", "cn_ashares"}
```

- [ ] **Step 2: Add `prompting_llms` to the rejected-values list**

In the same file, replace the `test_coerce_rejects_unknown` parametrize list so the retired slug is pinned as rejected:

```python
@pytest.mark.parametrize(
    "bad", ["crypto", "futures", "prompting_llms", "Foundation", "us stocks"]
)
def test_coerce_rejects_unknown(bad):
    with pytest.raises(ValueError, match="unknown category"):
        coerce_category(bad)
```

- [ ] **Step 3: Add the locked-shelf and catalog-migration guards**

Append to the end of `dashboard/backend/tests/test_agent_taxonomy.py`:

```python
# --- locked shelves + catalog migration ------------------------------------


@pytest.mark.parametrize("locked", ["crypto", "futures"])
def test_locked_shelves_have_no_category_slug(locked):
    """Crypto and Futures are inert rows in the frontend only.

    Nothing can be assigned to them -- no bar source, no MarketProfile, no
    engine support -- so they deliberately get no slug. A member here would
    make them selectable in Configure and cloneable from Community while no
    backtest could ever run.
    """
    assert locked not in AGENT_CATEGORIES
    with pytest.raises(ValueError, match="unknown category"):
        coerce_category(locked)


def test_no_template_is_left_on_the_retired_prompting_llms_slug():
    """A template stranded on the retired slug would 422 on any later PATCH and
    render under Community's "General" fallback label rather than a market chip.
    The catalog is `lru_cache`d, so this reloads it before reading.
    """
    import dashboard.backend.domain.agents.marketplace as marketplace_mod

    marketplace_mod.reload_marketplace_catalog()
    slugs = {t.get("category") for t in marketplace_mod.list_marketplace_templates()}
    assert "prompting_llms" not in slugs
    assert slugs <= AGENT_CATEGORIES
```

- [ ] **Step 4: Run the taxonomy tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_agent_taxonomy.py -q`
Expected: FAIL — `test_categories_whitelist` (set still contains `prompting_llms`), `test_coerce_rejects_unknown[prompting_llms]` (DID NOT RAISE), `test_no_template_is_left_on_the_retired_prompting_llms_slug` (two templates still on it).

- [ ] **Step 5: Narrow the `Literal` and correct the docstrings**

In `dashboard/backend/domain/agents/taxonomy.py`, replace lines 12–30 (from the `` ``AGENT_CATEGORIES`` is derived `` paragraph through `AGENT_CATEGORIES = frozenset(...)`) with:

```python
``AGENT_CATEGORIES`` is derived from ``AgentCategory`` rather than declared
alongside it: the ``Literal`` is what Pydantic validates against and what FastAPI
publishes into ``openapi.json``, so making it the single source keeps the runtime
whitelist and the published contract from drifting apart.

The ``Literal``'s *declaration order* is load-bearing too: it is the market
order inside My Agents' Stocks shelf, mirroring ``MARKET_LABELS`` in
``dashboard/frontend/app.js``. Reorder the members and the Community listing
reorders with it (see :func:`category_sort_rank`).

These are *markets*, not asset classes. My Agents shelves by what an agent
trades (Stocks, Crypto, Futures) and only Stocks is live, so "crypto" and
"futures" are deliberately absent here: those shelves are locked, inert rows in
the frontend with nothing assignable to them. Adding a member would make them
selectable in Configure and cloneable from Community while no bar source,
``MarketProfile`` or engine support exists behind either.
"""
from typing import Literal, Optional, Tuple, get_args

AgentCategory = Literal["us_stocks", "cn_ashares"]

#: Market display order. ``get_args`` preserves the ``Literal``'s declaration
#: order, so this is the one place the ordering is stated -- alphabetical slug
#: order is *not* the intended order ("cn_ashares" would lead).
AGENT_CATEGORY_ORDER: Tuple[str, ...] = get_args(AgentCategory)

AGENT_CATEGORIES = frozenset(AGENT_CATEGORY_ORDER)
```

- [ ] **Step 6: Recategorize the two `prompting_llms` templates**

Both run DJIA-30 (U.S. equities). In `dashboard/config/marketplace.json`, change `"category": "prompting_llms"` to `"category": "us_stocks"` at line 8 (`balanced-starter`) and line 26 (`momentum-scout`). These are the only two occurrences.

Verify exactly two lines changed and no other key moved:

```bash
git diff --stat dashboard/config/marketplace.json
command grep -c '"prompting_llms"' dashboard/config/marketplace.json   # expect 0
```

- [ ] **Step 7: Fix the catalog-ordering test's leading shelf**

`test_marketplace_listing_is_ordered_by_shelf_not_by_slug` in `dashboard/backend/tests/test_agents_api.py` asserts the retired slug leads the listing. Replace its docstring's slug-order sentence and the two order assertions:

```python
def test_marketplace_listing_is_ordered_by_shelf_not_by_slug():
    """Community cards group by market in *declaration* order, not slug order.

    The recategorization onto slugs quietly changed which card leads the page:
    ``sorted`` on the raw value orders cn_ashares < us_stocks, so the A-share
    template became card #1 on a U.S.-focused product. Nothing caught it because
    no test asserted order. ``category_sort_rank`` keys on the AgentCategory
    Literal's declaration order instead, which is also the order MARKET_LABELS
    renders the market chips in, so the two surfaces agree.
    """
```

and, further down in the same function:

```python
    # The U.S. market leads; uncategorized templates never do.
    assert templates[0]["category"] == AGENT_CATEGORY_ORDER[0] == "us_stocks"
    assert templates[-1]["category"] == "cn_ashares"
```

- [ ] **Step 8: Run the backend tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_agent_taxonomy.py dashboard/backend/tests/test_agents_api.py -q`
Expected: PASS. `test_agents_api.py`'s openapi guard (`assert enums == [AGENT_CATEGORIES]`, ~line 1248) follows the `Literal` automatically and must stay green without edits — if it fails, the `Literal` edit is wrong, not the test.

- [ ] **Step 9: Confirm the seed DB is untouched, then commit**

```bash
git status --short
# must NOT list dashboard/storage/data/backtest.db or backtest.db-wal
git add dashboard/backend/domain/agents/taxonomy.py \
        dashboard/config/marketplace.json \
        dashboard/backend/tests/test_agent_taxonomy.py \
        dashboard/backend/tests/test_agents_api.py
git commit -m "refactor(agents): categories are markets, not shelves"
```

---

## Task 2: Frontend config layer — `MARKET_LABELS` replaces `SHELF_LABELS`

**Files:**
- Modify: `dashboard/frontend/app.js:472-536` (config block), `:712-714` (`agentTypeLabel`), `:1805-1872` (Community chip layer), `:1931` (card category label), `:6828` (`navigateToPage` reset)
- Modify: `dashboard/frontend/js/agent-editor.js:576-580`
- Test: `dashboard/backend/tests/test_frontend_shelves.py`

**Why the Community chip layer moves in this task, not Task 4:** four call sites read `SHELF_LABELS` (app.js ~1845, ~1860, ~1931, ~6828). Deleting the declaration without them is a `ReferenceError` on every Community render, and `renderMarketplaceCategoryChips` needs more than a rename — built from `AGENT_SHELVES` it would emit a single, meaningless "Stocks" chip once Task 2 lands. Repointing them here keeps every commit's app runnable.

**Interfaces:**
- Consumes: Task 1's `AGENT_CATEGORY_ORDER` order (`us_stocks` before `cn_ashares`), mirrored by `MARKET_LABELS`' key order.
- Produces, for Tasks 3–4:
  - `MARKET_LABELS: {us_stocks: 'U.S.', cn_ashares: 'China A-Share'}` — insertion-ordered, so `Object.entries` yields chip order.
  - `window.AGENT_SHELF_LABELS === MARKET_LABELS` (the export name is unchanged so `agent-editor.js` needs no rewiring).
  - `AGENT_SHELVES: [{key:'stocks', title:'Stocks', match}, {key:'external', title:'For Developers: Connected Agents', match}]`
  - `agentShelfKey(agent) -> 'stocks' | 'external'`
  - `agentMarketKey(agent) -> 'us_stocks' | 'cn_ashares' | ''`
  - `let agentMarketFilter = 'all'` (module-level, mutable)
  - `agentTypeLabel(agent) -> 'Hosted AI' | 'Your own code'`
  - `shelfIdSuffix` unchanged: `'stocks' -> 'Stocks'`, `'external' -> 'External'`

- [ ] **Step 1: Write the failing guards**

In `dashboard/backend/tests/test_frontend_shelves.py`, **delete** these four existing tests — `test_agent_shelves_config_has_the_four_shelf_keys` (~line 134), `test_shelf_labels_map_is_derived_from_agent_shelves_not_duplicated` (~line 221), `test_render_marketplace_category_chips_covers_all_plus_the_three_categories` (~line 253) and `test_navigate_to_page_resets_chip_filter_on_plain_community_entry` (~line 297) — and add this block where the first of them was (~line 134, which is after `_strip_js_comments`' definition at ~line 121 and so can call it):

```python
def test_agent_shelves_config_holds_only_the_two_live_shelves():
    """`AGENT_SHELVES` drives rendering, so it lists only shelves that have a
    grid to render into. Crypto and Futures are locked, inert rows in app.html
    with no grid/footer/empty element -- listing them here would force a
    `locked` filter at all five iteration sites, and one missed filter trips
    renderAgentCategories' "some grid is missing" guard, silently aborting the
    whole My Agents render.
    """
    config = js_const("AGENT_SHELVES")
    for key in ("stocks", "external"):
        assert f"key: '{key}'" in config, key
    for absent in ("crypto", "futures", "prompting_llms", "us_stocks", "cn_ashares"):
        assert f"key: '{absent}'" not in config, absent


def test_market_labels_is_the_only_declaration_of_the_market_names():
    """One map, four consumers: the Stocks market chips, the Community category
    chips, the agent-card submeta, and the Configure picker. A second hand-typed
    copy would let one surface drift from the others -- which is exactly what
    the retired SHELF_LABELS existed to prevent, so its name must be gone too,
    not merely unused.
    """
    decl = js_const("MARKET_LABELS")
    assert "us_stocks: 'U.S.'" in decl
    assert "cn_ashares: 'China A-Share'" in decl
    assert "SHELF_LABELS" not in _strip_js_comments(APP_JS)


def test_market_labels_is_what_the_configure_picker_reads():
    """The <select>'s options come from app.js, not a second option list in the
    editor. The export NAME is unchanged (window.AGENT_SHELF_LABELS) so the
    editor needs no rewiring; only what it points at changed.
    """
    assert "window.AGENT_SHELF_LABELS = MARKET_LABELS;" in APP_JS


def test_card_submeta_carries_the_decision_axis_the_retired_shelf_used_to():
    """"Prompting LLMs" is gone as a section, so the how-does-it-decide axis it
    carried moves onto the card. 'Built-in'/'External' named the plumbing;
    'Hosted AI'/'Your own code' names what the user actually gets.
    """
    body = _strip_js_comments(fn_body("function agentTypeLabel("))
    assert "'Hosted AI'" in body
    assert "'Your own code'" in body
    assert "'Built-in'" not in body


def test_render_marketplace_category_chips_is_built_from_the_shared_label_map():
    """The chip row is built from MARKET_LABELS rather than a second hardcoded
    list, plus an 'all' chip that isn't a category at all. It is no longer built
    from AGENT_SHELVES: Community filters templates by *market*, and there is
    now one Stocks shelf holding both markets, so the shelf list and the chip
    list are different things -- built from AGENT_SHELVES this row would emit a
    single, meaningless "Stocks" chip that matches no template.
    """
    body = _strip_js_comments(fn_body("function renderMarketplaceCategoryChips()"))
    assert "MARKET_LABELS" in body
    assert "'all'" in body
    for label in ("U.S.", "China A-Share"):
        assert label not in body, f"{label!r} hardcoded instead of read from MARKET_LABELS"


def test_navigate_to_page_resets_chip_filter_on_plain_community_entry():
    """A category set by one Community visit must not leak into a later,
    unrelated visit made through the plain nav tab -- the most common entry
    path. navigateToPage is the one choke point every Community entry funnels
    through, so the reset belongs there: 'all' unless an explicit
    `communityCategory` option says otherwise. Signature passed to fn_body stops
    at the opening paren, not `(page, options = {})` -- that default value's own
    `{}` would otherwise be mistaken for the function body by fn_body's brace
    matcher.
    """
    body = _strip_js_comments(fn_body("function navigateToPage("))
    assert (
        "marketplaceCategoryFilter = MARKET_LABELS[options.communityCategory] "
        "? options.communityCategory : 'all';"
    ) in body
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest dashboard/backend/tests/test_frontend_shelves.py -q`
Expected: FAIL — `js_const("MARKET_LABELS")` raises or returns nothing, `AGENT_SHELVES` still carries the four old keys, `agentTypeLabel` still returns `'Built-in'`, and both Community-layer tests still see `SHELF_LABELS`.

- [ ] **Step 3: Rewrite the app.js config block**

Replace `dashboard/frontend/app.js` lines 472–536 — the whole run from the `// Shelf to use for an uncategorized agent…` comment through the `let agentGridPage = …` line — with:

```js
// Legacy runtime -> market, for an uncategorized agent whose runtime already
// implies one. Every agent cloned before shelving shipped carries
// `category: null`, and the hosted AI Hedge Fund runtime is a U.S. stock
// strategy. Keyed on `runtime_type` rather than backfilled in SQL because the
// fallback also covers rows written by an older backend that doesn't send
// `category` at all, which a one-shot migration cannot. New clones stamp the
// column and never reach this table.
const LEGACY_RUNTIME_MARKET = { ai_hedge_fund: 'us_stocks' };

/** Category slug -> market display name. The single place these strings are
 * written: the Stocks shelf's market chips, the Community category chips, the
 * agent-card submeta and the Configure picker all read this map, so renaming a
 * market is one edit. Key order is chip order and mirrors the AgentCategory
 * Literal's declaration order in dashboard/backend/domain/agents/taxonomy.py.
 *
 * Markets, not asset classes: My Agents shelves by what an agent trades, and
 * Stocks is the only asset class the engine can backtest, so both entries here
 * live under it. */
const MARKET_LABELS = {
  us_stocks: 'U.S.',
  cn_ashares: 'China A-Share',
};

// Exported for js/agent-editor.js, which builds the Configure screen's market
// <select> from this rather than a second hardcoded option list. agent-editor.js
// is loaded *before* app.js, so it must read this at call time (when the editor
// opens), never at its own module-init time -- the same rule window.API follows.
window.AGENT_SHELF_LABELS = MARKET_LABELS;

// My Agents' JS-driven sections, in display order. `match` delegates to
// agentShelfKey so every agent resolves to exactly one shelf by construction
// rather than by predicates staying mutually exclusive as they're edited.
//
// Crypto and Futures are deliberately NOT here. They are locked, inert rows in
// app.html with no grid, footer or empty-state element, so nothing in this file
// may try to address them: listing them would force a `locked` filter at every
// site that iterates this array, and one missed filter trips
// renderAgentCategories' "some grid is missing" guard, silently aborting the
// entire My Agents render. Their order is their order in app.html.
const AGENT_SHELVES = [
  { key: 'stocks', title: 'Stocks',
    match: (a) => agentShelfKey(a) === 'stocks' },
  { key: 'external', title: 'For Developers: Connected Agents',
    match: (a) => agentShelfKey(a) === 'external' },
];

/** The single shelf an agent renders under. Exactly one value per agent, so no
 * agent can be double-counted or dropped off every shelf.
 *
 * Stocks is the only asset class the backtest engine supports (every entry in
 * the backend's _MARKET_PROFILES is equities), so every built-in lands there
 * regardless of category, and connected agents split off by `agent_type`. The
 * market an agent trades is a separate axis -- see agentMarketKey. */
function agentShelfKey(agent) {
  if (!agent || agent.agent_type !== 'builtin') return 'external';
  return 'stocks';
}

/** Market a built-in agent trades, or '' when the platform genuinely doesn't
 * know -- a NULL/blank category, or a slug from a newer or older backend.
 *
 * '' is not a bug and must never hide the agent: those agents stay on Stocks
 * under the All chip and are excluded only by an explicit market filter, which
 * is the honest outcome when the market is unknown. */
function agentMarketKey(agent) {
  const slug = String(agent?.category || '').trim().toLowerCase();
  if (MARKET_LABELS[slug]) return slug;
  return LEGACY_RUNTIME_MARKET[String(agent?.runtime_type || '').trim().toLowerCase()] || '';
}

/** 'all' or one of MARKET_LABELS' keys. Narrows the Stocks shelf's grid only --
 * never its count pill, which reports what the shelf holds. */
let agentMarketFilter = 'all';

/** 'us_stocks' -> 'UsStocks' -- app.html's per-shelf element id suffix (agentsGrid<Suffix> etc). */
function shelfIdSuffix(shelfKey) {
  return String(shelfKey)
    .split('_')
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join('');
}

/** Per-shelf page index (0-based), keyed by AGENT_SHELVES' `key`. Reset on search change. */
let agentGridPage = Object.fromEntries(AGENT_SHELVES.map((shelf) => [shelf.key, 0]));
```

- [ ] **Step 4: Rename the card's decision-axis label**

Replace `dashboard/frontend/app.js:712-714`:

```js
/** How the agent decides, shown on the card submeta. This is the axis the
 * retired "Prompting LLMs" section used to carry: the platform runs hosted
 * models for built-ins, while a connected agent runs the user's own program.
 * 'Built-in'/'External' named the plumbing; these name what the user gets. */
function agentTypeLabel(agent) {
  return agent.agent_type === 'builtin' ? 'Hosted AI' : 'Your own code';
}
```

- [ ] **Step 5: Point the editor's load-failure fallback at the markets**

Replace `dashboard/frontend/js/agent-editor.js:576-580`:

```js
  const SHELF_LABELS_FALLBACK = {
    us_stocks: 'U.S.',
    cn_ashares: 'China A-Share',
  };
```

- [ ] **Step 6: Repoint the Community chip layer at `MARKET_LABELS`**

Four edits in `dashboard/frontend/app.js`:

1. `~line 1805`, the `marketplaceCategoryFilter` doc comment:

```js
/** 'all' or one of MARKET_LABELS' keys. Set by the chip row and by the Stocks
 * shelf's empty-state Community button (via navigateToPage's options). */
let marketplaceCategoryFilter = 'all';
```

2. `~line 1845`, inside `setMarketplaceCategoryFilter` (the rest of the function, including its long doc comment, is unchanged):

```js
  marketplaceCategoryFilter = MARKET_LABELS[category] ? category : 'all';
```

3. `~line 1849-1866`, `renderMarketplaceCategoryChips`'s doc comment and `chips` array:

```js
/** Chip row above the marketplace grid: 'All' plus one chip per market, built
 * from MARKET_LABELS rather than a second hardcoded list. Built from the label
 * map rather than AGENT_SHELVES because Community filters templates by
 * *market*, and there is now one Stocks shelf holding both markets -- the shelf
 * list and the chip list are different things. */
function renderMarketplaceCategoryChips() {
  const container = document.getElementById('marketplaceCategoryChips');
  if (!container) return;
  const chips = [
    { key: 'all', label: 'All' },
    ...Object.entries(MARKET_LABELS).map(([key, label]) => ({ key, label })),
  ];
```

The rest of the function — the build-once block and the state toggle — is unchanged.

4. `~line 1931`, in `renderMarketplaceGrid`:

```js
    const categoryLabel = MARKET_LABELS[String(template.category || '').toLowerCase()] || 'General';
```

- [ ] **Step 7: Repoint `navigateToPage`'s filter reset**

`~line 6828`, inside `navigateToPage`:

```js
            marketplaceCategoryFilter = MARKET_LABELS[options.communityCategory] ? options.communityCategory : 'all';
```

- [ ] **Step 8: Run the guards and confirm no stale reference survives**

Run: `pytest dashboard/backend/tests/test_frontend_shelves.py -q`
Expected: the six tests from Step 1 PASS. Other tests in the file still fail — they pin app.html markup (Task 3) and My Agents render-loop strings (Task 4). That is expected here.

```bash
command grep -n "SHELF_LABELS" dashboard/frontend/app.js          # expect no matches
command grep -n "LEGACY_RUNTIME_SHELF" dashboard/frontend/app.js  # expect no matches
```

Both must return nothing. (`SHELF_LABELS_FALLBACK` in `agent-editor.js` is a different file and stays.) A surviving match is a `ReferenceError` waiting on the next Community render, not a cosmetic leftover.

- [ ] **Step 9: Commit**

```bash
git status --short   # no backtest.db / -wal
git add dashboard/frontend/app.js dashboard/frontend/js/agent-editor.js \
        dashboard/backend/tests/test_frontend_shelves.py
git commit -m "refactor(agents): MARKET_LABELS replaces SHELF_LABELS"
```

---

## Task 3: `app.html` — two live sections, two locked rows

**Files:**
- Modify: `dashboard/frontend/app.html:899-951` (`#agentsCategories`), `:976-979` (editor field), `:16` + `:1707-1708` (cache-busters)
- Test: `dashboard/backend/tests/test_frontend_shelves.py`
- Test: `dashboard/backend/tests/test_frontend_fast_boot.py:140-152`

**Interfaces:**
- Consumes: Task 2's `shelfIdSuffix` mapping — element ids must be `agentsGridStocks` / `agentsGridFooterStocks` / `agentsEmptyStocks` / `agentsCountStocks`, and the same with the `External` suffix.
- Produces, for Task 4: `#agentsMarketChips` (empty container, JS fills it), `#agentsCountStocks` and `#agentsCountExternal` (`hidden` until JS writes them), and locked sections carrying `data-category="crypto"` / `"futures"` with **no** grid, footer, empty, count or chip elements.

- [ ] **Step 1: Write the failing markup guards**

In `dashboard/backend/tests/test_frontend_shelves.py`, replace `_HEADERS_AND_SUBTITLES` (lines 33–50), `_SHELF_SUFFIX_TO_CATEGORY_SLUG` (57–65), `test_all_four_shelf_headers_and_subtitles_are_present` (68–71) and `test_four_agents_category_sections_with_distinct_shelf_ids` (86–99) with:

```python
_LIVE_SHELVES = [
    (
        "Stocks",
        "Trade U.S. blue-chip and Chinese A-share stocks, tested hour by hour on real market data.",
    ),
    (
        "For Developers: Connected Agents",
        "Run your own trading program against our backtests. Requires an access key.",
    ),
]

_LOCKED_SHELVES = [
    (
        "crypto",
        "Crypto",
        "Round-the-clock crypto backtesting isn't built yet. Nothing here can be run.",
    ),
    (
        "futures",
        "Futures",
        "Futures contracts aren't built yet. Nothing here can be run.",
    ),
]

# Live shelf id suffix -> the AGENT_SHELVES key it corresponds to. Only these
# two are addressed from JS; the locked rows have no ids at all.
_SHELF_SUFFIX_TO_KEY = {"Stocks": "stocks", "External": "external"}

_RETIRED_SECTION_HEADERS = (
    "Prompting LLMs",
    "U.S. Stock Trading",
    "China A-Share Trading",
)
```

and:

```python
def test_live_shelf_headers_and_subtitles_are_present():
    for header, subtitle in _LIVE_SHELVES:
        assert header in _HTML, f"missing shelf header: {header!r}"
        assert subtitle in _HTML, f"missing shelf subtitle: {subtitle!r}"


def test_geographic_section_headers_are_retired():
    """The taxonomy is asset class now. These three named the old axes -- two
    geographic, one about how the agent decides -- and must not survive as
    section headers anywhere on the page.
    """
    for header in _RETIRED_SECTION_HEADERS:
        assert f">{header}</h3>" not in _HTML, header


def test_two_live_sections_with_distinct_shelf_ids():
    """Each live shelf gets its own grid/footer/empty/count id so the render
    loop can address them uniformly: `agentsGrid<Shelf>` /
    `agentsGridFooter<Shelf>` / `agentsEmpty<Shelf>` / `agentsCount<Shelf>`,
    where `<Shelf>` is shelfIdSuffix's PascalCase form of the AGENT_SHELVES key.
    """
    for suffix, key in _SHELF_SUFFIX_TO_KEY.items():
        assert f'data-category="{key}"' in _HTML, key
        assert f'id="agentsGrid{suffix}"' in _HTML, suffix
        assert f'id="agentsGridFooter{suffix}"' in _HTML, suffix
        assert f'id="agentsEmpty{suffix}"' in _HTML, suffix
        assert f'id="agentsCount{suffix}"' in _HTML, suffix


def test_retired_shelf_ids_are_gone():
    """A leftover id would silently double-register an element the render loop
    no longer expects to find.
    """
    for suffix in ("Builtin", "PromptingLlms", "UsStocks", "CnAshares"):
        assert f'id="agentsGrid{suffix}"' not in _HTML, suffix
        assert f'id="agentsGridFooter{suffix}"' not in _HTML, suffix
        assert f'id="agentsEmpty{suffix}"' not in _HTML, suffix


def test_locked_shelves_are_rendered_inert_not_empty():
    """Crypto and Futures have no bar source, no MarketProfile and no engine
    support. They must read as "not built yet", never as "built and broken", so
    they carry aria-disabled and the locked class -- and, critically, none of
    the grid/footer/empty/count/chip elements the render loop addresses. A grid
    element here would make renderAgentCategories' missing-grid guard the only
    thing standing between a stray id and a page that renders nothing.
    """
    for slug, title, subtitle in _LOCKED_SHELVES:
        section_at = _HTML.index(f'data-category="{slug}"')
        section = _HTML[section_at : _HTML.index("</section>", section_at)]
        assert 'class="agents-category agents-category--locked"' in _HTML[
            max(0, section_at - 120) : section_at
        ], slug
        assert 'aria-disabled="true"' in _HTML[max(0, section_at - 120) : section_at + 120], slug
        assert f">{title}</h3>" in section, title
        assert subtitle in section, subtitle
        assert "Not yet available" in section, slug
        assert "agents-grid" not in section, slug
        assert "agentsCount" not in section, slug
        assert "agentsEmpty" not in section, slug


def test_market_chip_container_is_inside_the_stocks_shelf():
    """The chips filter the Stocks shelf, and they ride #agentsCategories'
    existing delegated click handler -- so they must live inside that container,
    not in the page toolbar above it.
    """
    stocks_at = _HTML.index('data-category="stocks"')
    stocks = _HTML[stocks_at : _HTML.index("</section>", stocks_at)]
    assert 'id="agentsMarketChips"' in stocks
    assert _HTML.index('id="agentsCategories"') < stocks_at
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest dashboard/backend/tests/test_frontend_shelves.py -q -k "shelf or locked or market_chip or geographic"`
Expected: FAIL — app.html still ships the four old sections.

- [ ] **Step 3: Replace the `#agentsCategories` markup**

Replace `dashboard/frontend/app.html` lines 899–951 in full with:

```html
                <div id="agentsCategories">
                    <section class="agents-category" data-category="stocks">
                        <div class="agents-category-head">
                            <div class="agents-category-heading">
                                <span class="agents-category-icon" aria-hidden="true">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
                                </span>
                                <h3 class="agents-category-title">Stocks</h3>
                                <span id="agentsCountStocks" class="agents-category-count" hidden></span>
                            </div>
                            <p class="agents-category-sub">Trade U.S. blue-chip and Chinese A-share stocks, tested hour by hour on real market data.</p>
                            <!-- Filled by renderAgentMarketChips(); the chips ride
                                 #agentsCategories' delegated click handler. -->
                            <div id="agentsMarketChips" class="marketplace-category-chips agents-market-chips" role="group" aria-label="Filter by market"></div>
                        </div>
                        <div id="agentsGridStocks" class="agents-grid">
                            <!-- Loading skeletons: visible pre-JS and while the first
                                 agents fetch runs. No removal code needed — every render
                                 path overwrites this grid's innerHTML (pinned by
                                 test_frontend_fast_boot.py). -->
                            <div class="section-card agent-card agent-card--skeleton" aria-hidden="true"></div>
                            <div class="section-card agent-card agent-card--skeleton" aria-hidden="true"></div>
                            <div class="section-card agent-card agent-card--skeleton" aria-hidden="true"></div>
                        </div>
                        <div id="agentsGridFooterStocks" class="agents-grid-footer" hidden></div>
                        <p id="agentsEmptyStocks" class="control-helper" hidden>You don't have any agents yet. Create one and test your first trading idea.</p>
                    </section>
                    <!-- Locked shelves. Crypto is a display-only price ticker
                         (quotes.py, never wired to the engine) and futures exist
                         nowhere in the repo, so these rows are inert on purpose:
                         no grid, no chips, no focus stop, nothing clickable. They
                         must read as "not built yet", never as "built and broken".
                         Adding a category slug for either would make them
                         selectable in Configure and cloneable from Community. -->
                    <section class="agents-category agents-category--locked" data-category="crypto" aria-disabled="true">
                        <div class="agents-category-head">
                            <div class="agents-category-heading">
                                <span class="agents-category-icon" aria-hidden="true">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
                                </span>
                                <h3 class="agents-category-title">Crypto</h3>
                                <span class="agents-category-tag">Not yet available</span>
                            </div>
                            <p class="agents-category-sub">Round-the-clock crypto backtesting isn't built yet. Nothing here can be run.</p>
                        </div>
                    </section>
                    <section class="agents-category agents-category--locked" data-category="futures" aria-disabled="true">
                        <div class="agents-category-head">
                            <div class="agents-category-heading">
                                <span class="agents-category-icon" aria-hidden="true">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
                                </span>
                                <h3 class="agents-category-title">Futures</h3>
                                <span class="agents-category-tag">Not yet available</span>
                            </div>
                            <p class="agents-category-sub">Futures contracts aren't built yet. Nothing here can be run.</p>
                        </div>
                    </section>
                    <section class="agents-category" data-category="external">
                        <div class="agents-category-head">
                            <div class="agents-category-heading">
                                <span class="agents-category-icon" aria-hidden="true">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 18 6-6-6-6"/><path d="m8 6-6 6 6 6"/></svg>
                                </span>
                                <h3 class="agents-category-title">For Developers: Connected Agents</h3>
                                <span id="agentsCountExternal" class="agents-category-count" hidden></span>
                            </div>
                            <p class="agents-category-sub">Run your own trading program against our backtests. Requires an access key.</p>
                        </div>
                        <div id="agentsGridExternal" class="agents-grid">
                            <div class="section-card agent-card agent-card--skeleton" aria-hidden="true"></div>
                        </div>
                        <div id="agentsGridFooterExternal" class="agents-grid-footer" hidden></div>
                        <p id="agentsEmptyExternal" class="control-helper" hidden>No agents match your search.</p>
                    </section>
                </div>
```

- [ ] **Step 4: Relabel the Configure picker for markets**

The picker sets which market chip an agent falls under, not which section it lives in. Replace `dashboard/frontend/app.html:977-978`:

```html
                    <span class="agent-editor-model-label">Market</span>
                    <select id="agentEditorCategorySelect" class="agent-editor-model-select" aria-label="Which market this agent trades"></select>
```

- [ ] **Step 5: Update the fast-boot skeleton guard for the new grid ids**

Replace the `grid_ids` tuple and its comment in `dashboard/backend/tests/test_frontend_fast_boot.py:140-152`:

```python
def test_agents_grid_ships_loading_skeleton():
    # Pre-JS, My Agents must show placeholder cards instead of a blank panel,
    # in every shelf that renders agents. Only two do: Stocks (the one live
    # asset class) and External. The Crypto/Futures rows are locked and have no
    # grid at all, so there is nothing to skeleton there.
    grid_ids = (
        "agentsGridStocks",
        "agentsGridExternal",
    )
```

- [ ] **Step 6: Bump the cache-busters**

`app.html:16` → `<link rel="stylesheet" href="styles.css?v=83">`
`app.html:1707` → `<script src="js/agent-editor.js?v=26" defer></script>`
`app.html:1708` → `<script src="app.js?v=71" defer></script>`

- [ ] **Step 7: Run the markup guards to verify they pass**

Run: `pytest dashboard/backend/tests/test_frontend_shelves.py dashboard/backend/tests/test_frontend_fast_boot.py -q`
Expected: `test_frontend_fast_boot.py` fully PASS; in `test_frontend_shelves.py` every test added in Steps 1 and in Task 2 PASSes. Tests pinning render-loop strings (`test_navigate_to_page_resets_chip_filter_on_plain_community_entry`, `test_render_marketplace_category_chips_*`) still fail — Task 4 fixes them.

- [ ] **Step 8: Commit**

```bash
git status --short   # no backtest.db / -wal
git add dashboard/frontend/app.html \
        dashboard/backend/tests/test_frontend_shelves.py \
        dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "feat(agents): asset-class shelf markup with locked rows"
```

---

## Task 4: `app.js` render loop — market chips, count pill, Community button

**Files:**
- Modify: `dashboard/frontend/app.js:1241-1256` (card submeta), `:1360-1432` (empty states + render loop), `:6942-6960` (delegated click handler)
- Test: `dashboard/backend/tests/test_frontend_shelves.py`

The Community chip layer and `navigateToPage`'s filter reset were already repointed in Task 2 — do not touch them again here.

**Interfaces:**
- Consumes: Task 2's `MARKET_LABELS`, `AGENT_SHELVES`, `agentShelfKey`, `agentMarketKey`, `agentMarketFilter`, `shelfIdSuffix`; Task 3's `#agentsMarketChips`, `#agentsCountStocks`, `#agentsCountExternal`.
- Produces: `renderAgentMarketChips()`, `setAgentMarketFilter(market)`, `stocksEmptyHtml({searching, marketFilter})`, `communityShelfButtonHtml(category)` (emits `<button class="agents-empty-community-btn">`, styled in Task 5).

- [ ] **Step 1: Write the failing guards**

Append to `dashboard/backend/tests/test_frontend_shelves.py`:

```python
def test_empty_shelf_community_cta_is_a_button_not_a_bare_anchor():
    """This is the primary path off an empty shelf, and it shipped as
    `<a href="#">` against a class with no CSS rule anywhere -- so it inherited
    plain link styling and did not read as actionable. The dataset hook the
    delegated handler reads is unchanged; only the element and its class are.
    """
    body = _strip_js_comments(fn_body("function communityShelfButtonHtml("))
    assert "<button type=\"button\"" in body
    assert "agents-empty-community-btn" in body
    assert "data-community-category=" in body
    assert 'href="#"' not in body
    assert "agents-empty-community-link" not in _strip_js_comments(APP_JS)


def test_market_chips_filter_the_grid_but_never_the_count_pill():
    """The pill reports what the shelf HOLDS, read from the unfiltered roster
    (`allAgents`), not what is on screen. A number that moved while you typed in
    the search box or clicked a chip would read as agents disappearing.
    """
    body = _strip_js_comments(fn_body("function renderAgentCategories("))
    assert "allAgents.filter(shelf.match).length" in body
    assert "agentMarketKey(a) === agentMarketFilter" in body


def test_market_chip_selection_resets_pagination():
    """The page index is per-shelf, so a page-3 position under 'All' would land
    past the end of a narrower market's single page -- an empty grid with a
    "Page 3 of 1" footer. applyAgentFilters() resets it; applyAgentFilters(false)
    would not.
    """
    body = _strip_js_comments(fn_body("function setAgentMarketFilter("))
    assert "applyAgentFilters()" in body


def test_stocks_empty_state_distinguishes_search_chip_and_truly_empty():
    """Three distinct cases, deliberately worded apart. Collapsing them would
    tell a user who is mid-search, or who clicked a market chip, that they own
    no agents at all -- and Stocks is the onboarding surface now (it inherited
    that role from the retired Prompting LLMs shelf), so its true-empty copy is
    the create-your-first voice, not the add-from-Community voice.
    """
    body = _strip_js_comments(fn_body("function stocksEmptyHtml("))
    assert "No agents match your search." in body
    assert "You don't have any agents yet." in body
    assert "communityShelfButtonHtml" in body
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest dashboard/backend/tests/test_frontend_shelves.py -q`
Expected: FAIL — `stocksEmptyHtml`, `communityShelfButtonHtml`, `setAgentMarketFilter` and the count-pill/market-filter lines don't exist yet, so `fn_body` raises on each.

- [ ] **Step 3: Show the market on the agent card**

Under one Stocks shelf, `All` mixes U.S. and A-share agents; without this the card cannot say which. Replace `dashboard/frontend/app.js:1244-1245`:

```js
    const model = escapeHtml(formatAgentModelLabel(agent.model_name));
    const type = escapeHtml(agentTypeLabel(agent));
    // Under the All chip the Stocks shelf mixes markets, so the card has to
    // say which. Omitted rather than guessed when agentMarketKey returns ''.
    const market = MARKET_LABELS[agentMarketKey(agent)];
```

and the submeta line inside the same `card.innerHTML` template (currently `<p class="agent-card-submeta">${model} · ${type}</p>`):

```html
            <p class="agent-card-submeta">${model} · ${type}${market ? ` · ${escapeHtml(market)}` : ''}</p>
```

- [ ] **Step 4: Replace the empty-state helpers**

Replace `dashboard/frontend/app.js` lines 1362–1383 — from the `// Empty-state HTML shown when a shelf has zero agents…` comment through the end of `communityShelfLinkHtml` — with:

```js
// Empty-state HTML for the Stocks shelf. Three cases, deliberately worded
// apart: a live search hiding everything, a market chip with nothing on it yet,
// and a genuinely empty shelf. Collapsing them would tell a searching or
// filtering user they own no agents. Stocks is also the onboarding surface (it
// inherited that role from the retired Prompting LLMs shelf), so the true-empty
// case keeps the create-your-first voice rather than the Community-upsell voice
// the old market shelves used.
//
// External renders a placeholder CARD instead (renderExternalPlaceholderCard),
// so it has no entry here.
function stocksEmptyHtml({ searching, marketFilter }) {
  if (searching) return 'No agents match your search.';
  if (marketFilter !== 'all') {
    const label = escapeHtml(MARKET_LABELS[marketFilter] || '');
    return `No ${label} agents yet. Add a ready-made ${label} strategy from ${communityShelfButtonHtml(marketFilter)}.`;
  }
  return `You don't have any agents yet. Create one and test your first trading idea, or browse ready-made strategies in ${communityShelfButtonHtml('all')}.`;
}

// A real <button>, not an <a href="#">: this is the primary path off an empty
// shelf, and as an anchor it matched no CSS rule anywhere in styles.css, so it
// inherited plain link styling and did not read as actionable.
//
// data-community-category is read by #agentsCategories' delegated click
// handler, which routes it through navigateToPage's options so the matching
// Community chip is pre-selected. 'all' is a valid value there -- navigateToPage
// falls it back to 'all' because it isn't a MARKET_LABELS key.
function communityShelfButtonHtml(category) {
  return `<button type="button" class="agents-empty-community-btn" data-community-category="${escapeHtml(category)}">Community</button>`;
}
```

- [ ] **Step 5: Add the market chip row and rewrite the render loop**

Replace `dashboard/frontend/app.js`'s `renderAgentCategories` (from `function renderAgentCategories(agents) {` through its closing brace, currently lines 1385–1432) with:

```js
/** The Stocks shelf's market filter row: 'All' plus one chip per MARKET_LABELS
 * entry, reusing the Community chip classes so the same taxonomy looks the same
 * on both surfaces.
 *
 * Built once, then only toggled. This runs from renderAgentCategories, which is
 * bound to the search box's `input` event -- rebuilding innerHTML per keystroke
 * would blow away the focused chip on every character typed. */
function renderAgentMarketChips() {
  const container = document.getElementById('agentsMarketChips');
  if (!container) return;
  const chips = [
    { key: 'all', label: 'All' },
    ...Object.entries(MARKET_LABELS).map(([key, label]) => ({ key, label })),
  ];
  const existing = container.querySelectorAll('[data-agent-market]');
  if (existing.length !== chips.length) {
    container.innerHTML = chips
      .map((chip) => `<button type="button" class="marketplace-category-chip" data-agent-market="${escapeHtml(chip.key)}" aria-pressed="false">${escapeHtml(chip.label)}</button>`)
      .join('');
  }
  container.querySelectorAll('[data-agent-market]').forEach((button) => {
    const active = button.dataset.agentMarket === agentMarketFilter;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}

/** Select a market chip and re-render. Resets pagination: the page index is
 * per-shelf, so a page-3 position under 'All' would land past the end of a
 * narrower market's single page -- an empty grid under a "Page 3 of 1" footer.
 * An unrecognized market falls back to 'all' rather than filtering to a chip
 * that doesn't exist. */
function setAgentMarketFilter(market) {
  agentMarketFilter = MARKET_LABELS[market] ? market : 'all';
  applyAgentFilters();
}

function renderAgentCategories(agents) {
  const errorEl = document.getElementById('agentsErrorState');
  const shelves = AGENT_SHELVES.map((shelf) => {
    const suffix = shelfIdSuffix(shelf.key);
    return {
      shelf,
      grid: document.getElementById(`agentsGrid${suffix}`),
      emptyEl: document.getElementById(`agentsEmpty${suffix}`),
      countEl: document.getElementById(`agentsCount${suffix}`),
    };
  });
  if (shelves.some(({ grid }) => !grid)) return;

  if (errorEl) errorEl.hidden = true; // a successful render clears any prior error

  renderAgentMarketChips();

  const defaultId = getDefaultAgentId();
  const pinDefaultFirst = (list) =>
    [...list].sort((a, b) => (b.agent_id === defaultId) - (a.agent_id === defaultId));

  // A live search narrows every shelf: distinguish "no agents at all"
  // (onboarding / Community upsell) from "none match your search" so we
  // never mis-say a shelf is empty when a search term is just hiding its
  // agents, and never surface the External onboarding card as a search result.
  const searching = !!(document.getElementById('agentSearchInput')?.value || '').trim();

  shelves.forEach(({ shelf, grid, emptyEl, countEl }) => {
    // The pill counts what the shelf HOLDS, read from the unfiltered roster --
    // not what is currently on screen. A number that moved while you typed or
    // clicked a chip would read as agents disappearing.
    if (countEl) {
      const held = allAgents.filter(shelf.match).length;
      countEl.hidden = held === 0;
      countEl.textContent = `${held} agent${held === 1 ? '' : 's'}`;
    }

    let matched = pinDefaultFirst(agents.filter(shelf.match));
    if (shelf.key === 'stocks' && agentMarketFilter !== 'all') {
      // agentMarketKey returns '' for a NULL/blank/unknown category, so those
      // agents match no chip and appear under All only -- visible, but never
      // filed under a market the platform can't actually vouch for.
      matched = matched.filter((a) => agentMarketKey(a) === agentMarketFilter);
    }
    renderAgentCards(grid, matched, shelf.key);

    if (shelf.key === 'external') {
      if (matched.length > 0) {
        if (emptyEl) emptyEl.hidden = true;
      } else if (searching) {
        if (emptyEl) {
          emptyEl.hidden = false;
          emptyEl.textContent = 'No agents match your search.';
        }
      } else {
        if (emptyEl) emptyEl.hidden = true;
        renderExternalPlaceholderCard(grid);
      }
      return;
    }

    if (!emptyEl) return;
    emptyEl.hidden = matched.length > 0;
    if (matched.length === 0) {
      emptyEl.innerHTML = stocksEmptyHtml({ searching, marketFilter: agentMarketFilter });
    }
  });
}
```

- [ ] **Step 6: Bind the market chips to the existing delegated handler**

`~line 6942`, in the `#agentsCategories` delegated click handler, insert the market-chip branch as the **first** branch, above the `communityLink` branch:

```js
    document.getElementById('agentsCategories')?.addEventListener('click', (event) => {
      const marketChip = event.target.closest('[data-agent-market]');
      if (marketChip) {
        setAgentMarketFilter(marketChip.dataset.agentMarket);
        return;
      }
      const communityLink = event.target.closest('[data-community-category]');
```

The rest of the handler is unchanged. The Community branch already calls `event.preventDefault()`; that is inert on a `type="button"` element but harmless, so leave it.

- [ ] **Step 7: Run the full shelf suite and confirm nothing stale survives**

Run: `pytest dashboard/backend/tests/test_frontend_shelves.py -q`
Expected: PASS, whole file.

```bash
command grep -n "agents-empty-community-link" dashboard/frontend/app.js  # expect no matches
command grep -n "communityShelfLinkHtml" dashboard/frontend/app.js       # expect no matches
command grep -n "shelfOnboardingEmptyHtml" dashboard/frontend/app.js     # expect no matches
```

All three must return nothing.

- [ ] **Step 8: Commit**

```bash
git status --short   # no backtest.db / -wal
git add dashboard/frontend/app.js dashboard/backend/tests/test_frontend_shelves.py
git commit -m "feat(agents): market chips and Community button on shelves"
```


## Task 5: `styles.css` — panels, locked rows, and the Community button

**Files:**
- Modify: `dashboard/frontend/styles.css:9288-9302`
- Test: `dashboard/backend/tests/test_frontend_shelves.py`

**Interfaces:**
- Consumes: Task 3's classes (`.agents-category--locked`, `.agents-category-heading`, `.agents-category-icon`, `.agents-category-count`, `.agents-category-tag`, `.agents-market-chips`) and Task 4's `.agents-empty-community-btn`.
- Produces: nothing downstream. Existing `.marketplace-category-chip` rules (styles.css:9468) are reused as-is by the market chips — do not duplicate or fork them.

Layering note: the page is `--bg-primary: #0a0e27`, the new panel is `--bg-surface: #0f1328`, and the cards inside it are `--bg-card: #131a35`. Each step is lighter than the one beneath, so the panel reads as a container without any extra shadow. The frontend is dark-theme only — no light-mode variants.

- [ ] **Step 1: Write the failing style guards**

Append to `dashboard/backend/tests/test_frontend_shelves.py`:

```python
_STYLES_PATH = (
    Path(__file__).resolve().parents[2] / "frontend" / "styles.css"
)
_STYLES = re.sub(r"/\*.*?\*/", "", _STYLES_PATH.read_text(encoding="utf-8"), flags=re.DOTALL)


def _rule(selector: str) -> str:
    """The declaration block for `selector`'s first rule, comments stripped.

    Scoped to one block so a check for "this shelf has a border" cannot be
    satisfied by an unrelated rule elsewhere in a 9,000-line stylesheet.
    """
    at = _STYLES.index(selector + " {")
    return _STYLES[at : _STYLES.index("}", at)]


def test_shelf_sections_are_real_panels_not_loose_prose():
    """The shipped shelves were five declarations -- a margin, an <h3>, a <p> --
    sitting above bordered cards, so the headers read as page copy rather than
    as a container the cards belong to. A panel needs all three of a border, a
    radius and a background to separate from the page beneath it.
    """
    rule = _rule(".agents-category")
    assert "border:" in rule
    assert "border-radius:" in rule
    assert "background:" in rule
    assert "padding:" in rule


def test_locked_shelves_are_visually_disabled_not_just_empty():
    """Dashed and muted, so the row reads as "not built yet" rather than as a
    live shelf that failed to load its cards.
    """
    rule = _rule(".agents-category--locked")
    assert "border-style: dashed" in rule
    assert "opacity:" in rule


def test_nothing_inside_a_locked_shelf_is_clickable():
    """A hover or pointer affordance on a row with nothing behind it converts
    caution into distrust -- the exact failure the locked treatment exists to
    avoid.
    """
    assert ".agents-category--locked * { pointer-events: none; }" in _STYLES


def test_community_cta_button_has_a_rule_of_its_own():
    """The class it replaces (`agents-empty-community-link`) had no rule
    anywhere in this file, which is why the CTA rendered as a plain hyperlink.
    A hover state is what makes it read as pressable.
    """
    assert ".agents-empty-community-btn {" in _STYLES
    assert ".agents-empty-community-btn:hover {" in _STYLES
    assert "agents-empty-community-link" not in _STYLES


def test_market_chips_reuse_the_community_chip_rules():
    """The same taxonomy must look the same on both surfaces, so the chips share
    `.marketplace-category-chip` rather than getting a forked copy that can
    drift. `.agents-market-chips` exists only to space the row inside the shelf
    head.
    """
    assert ".marketplace-category-chip {" in _STYLES
    assert not re.search(r"\.agents-market-chip\s*\{", _STYLES)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest dashboard/backend/tests/test_frontend_shelves.py -q -k "panel or locked or community_cta or market_chips_reuse"`
Expected: FAIL — `.agents-category` has no `border:`, and none of the new selectors exist.

- [ ] **Step 3: Replace the shelf styles**

Replace `dashboard/frontend/styles.css` lines 9288–9302 (the `/* My Agents — category rows (Demo 1) */` block, through `.agents-category-sub`'s closing brace) with:

```css
/* My Agents — asset-class shelves. A real panel, not loose prose: these sections
   sit directly above bordered cards, so an unframed <h3>/<p> pair read as page
   copy rather than as a container the cards belong to. The page is --bg-primary,
   the panel --bg-surface, the cards --bg-card: each step lighter than the one
   beneath, so no shadow is needed to separate them. */
.agents-category {
    margin-top: 24px;
    padding: 18px 18px 20px;
    border: 1px solid var(--border-color);
    border-radius: 14px;
    background: var(--bg-surface);
}
.agents-category:first-child { margin-top: 8px; }
.agents-category-head {
    margin-bottom: 16px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border-color);
}
.agents-category-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}
.agents-category-icon {
    display: inline-flex;
    flex-shrink: 0;
    width: 18px;
    height: 18px;
    color: var(--info-color);
}
.agents-category-icon svg { width: 100%; height: 100%; }
.agents-category-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
}
/* Reports what the shelf holds, not what is on screen — see
   renderAgentCategories. `margin-left: auto` parks it at the right edge of the
   heading row without a spacer element. */
.agents-category-count {
    margin-left: auto;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    background: rgba(148, 163, 184, 0.12);
    border: 1px solid var(--border-color);
    white-space: nowrap;
}
.agents-category-sub {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 0;
}
/* The chips themselves are .marketplace-category-chip — shared with Community
   on purpose, so the same taxonomy looks the same on both surfaces. This rule
   only spaces the row inside the shelf head. */
.agents-market-chips { margin: 12px 0 0; }

/* Locked shelves (Crypto, Futures). Neither asset class has a bar source, a
   MarketProfile or any engine support, so the row must read as "not built yet",
   never as "built and broken": dashed and muted, with no grid, no chips and
   nothing clickable inside it. */
.agents-category--locked {
    border-style: dashed;
    background: transparent;
    opacity: 0.62;
}
.agents-category--locked .agents-category-head {
    margin-bottom: 0;
    padding-bottom: 0;
    border-bottom: 0;
}
.agents-category--locked .agents-category-icon,
.agents-category--locked .agents-category-title {
    color: var(--text-secondary);
}
/* Belt and braces with aria-disabled: no hover, no cursor change, no focus
   stop, even if a future edit puts an interactive element in one of these rows. */
.agents-category--locked * { pointer-events: none; }
.agents-category-tag {
    margin-left: auto;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: var(--text-secondary);
    border: 1px dashed var(--border-color);
    white-space: nowrap;
}

/* Empty-shelf Community call to action. This shipped as an <a href="#"> whose
   class matched no rule in this file, so the primary path off an empty shelf
   inherited plain link styling and did not read as actionable. */
.agents-empty-community-btn {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    margin: 0 2px;
    border-radius: 999px;
    font-family: inherit;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-primary);
    background: rgba(0, 191, 255, 0.12);
    border: 1px solid var(--info-color);
    cursor: pointer;
    transition: all 0.15s;
}
.agents-empty-community-btn:hover {
    color: var(--text-on-accent);
    background-color: var(--info-color);
}
```

- [ ] **Step 4: Run the style guards to verify they pass**

Run: `pytest dashboard/backend/tests/test_frontend_shelves.py -q`
Expected: PASS, whole file.

- [ ] **Step 5: Run the full backend suite**

Run: `pytest dashboard/backend/tests/ -q`
Expected: PASS (skips are fine — `vnpy`/`discord` are optional deps, and the `@pg_only` tier fails open locally with no Postgres). Any failure here is a real regression, not a pre-existing one.

If `test_deleted_shim_is_not_importable` fails with `DID NOT RAISE ModuleNotFoundError`, that is stale bytecode, not a regression: `rm -rf dashboard/backend/engines dashboard/backend/services` and re-run.

- [ ] **Step 6: Commit**

```bash
git status --short   # no backtest.db / -wal
git add dashboard/frontend/styles.css dashboard/backend/tests/test_frontend_shelves.py
git commit -m "style(agents): shelf panels, locked rows, Community button"
```

- [ ] **Step 7: Sanity-check the rendered page**

```bash
python3 -m http.server 8123 --directory dashboard/frontend
```

Open `http://localhost:8123/app.html`. The API calls will fail (no backend) and the error state will show — that is expected and is itself worth checking. Confirm by eye:

1. Four sections, visibly framed and separated: **Stocks**, **Crypto** (dashed/muted, "Not yet available"), **Futures** (same), **For Developers: Connected Agents**.
2. No "Prompting LLMs", "U.S. Stock Trading" or "China A-Share Trading" heading anywhere.
3. Under Stocks: `All` · `U.S.` · `China A-Share` chips, styled identically to Community's chip row. Clicking one moves the active state.
4. Nothing inside Crypto/Futures highlights, changes the cursor, or takes focus via Tab.
5. The page does not scroll horizontally at a 390px-wide viewport.

Stop the server when done.

---

## Follow-ups (do NOT act on during implementation — surface at session end)

- **User-facing docs go stale.** PR #313 refreshed `docs/source/lab/*` for the shelf names this change retires. Per the owner's docs policy these are not edited mid-session. RTD has no CI build — run `sphinx-build -n -E` locally whenever they are eventually edited.
- **`agent-editor.js`'s Configure picker** now offers markets, not sections. If a future task adds a real third market, it lands in `MARKET_LABELS` + the backend `Literal` and every surface follows from those two edits.

## Self-Review

**Spec coverage.** §1 shelves → Tasks 2 (config) + 3 (markup); "Prompting LLMs" retirement + card badge → Task 2 Step 4; Stocks inherits the onboarding voice → Task 4 Step 4. §2 data model → Task 1 (Literal, no DDL, no migration) + Task 2 (`MARKET_LABELS`, shelf-key separation); legacy categories under `All` only → Task 2's `agentMarketKey` doc + Task 4's filter comment; count pill ignores chip and search → Task 4 Step 5 + its guard. §3 visual treatment → Task 5. §4 Community button → Task 4 Step 4 + Task 5; the top-nav `Community` tab is deliberately untouched (already a `<button>`; `.mode-btn`'s flat treatment is the tab pattern). §5 ripple → all five tasks, one file each. §6 testing → guards in every task, full suite at Task 5 Step 5.

**One sequencing correction found in review:** the four `SHELF_LABELS` reads in the Community chip layer moved from Task 4 into Task 2. Deleting the declaration in Task 2 while its readers survived would have left every Community render throwing a `ReferenceError` across two intermediate commits, and `renderMarketplaceCategoryChips` needed a real rewrite (not a rename) because `AGENT_SHELVES` no longer carries market keys.

**One deliberate spec deviation**, argued in Global Constraints: locked shelves stay out of `AGENT_SHELVES`.

**One addition beyond the spec**, Task 4 Step 3: the agent card shows its market. Under one Stocks shelf the `All` view mixes U.S. and A-share agents with no way to tell them apart, which would lose information the geographic shelves carried. It reads `MARKET_LABELS`, so it adds a consumer, not a declaration.

**Type consistency.** `MARKET_LABELS`, `AGENT_SHELVES`, `agentShelfKey`, `agentMarketKey`, `agentMarketFilter`, `setAgentMarketFilter`, `renderAgentMarketChips`, `stocksEmptyHtml`, `communityShelfButtonHtml`, `shelfIdSuffix` are spelled identically in the definitions (Tasks 2, 4) and every consumer (Tasks 3, 4, 5). Element ids follow `shelfIdSuffix` exactly: `stocks` → `Stocks`, `external` → `External`.
