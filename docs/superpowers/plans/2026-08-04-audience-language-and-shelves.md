# Audience Language & My Agents Shelves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite all user-facing copy for the real target audience (wealthy, mostly older or executive-level, non-technical, serious about money) and reorganize My Agents + Community around a shared category taxonomy ("shelves"), with "Prompting LLMs" as the top section replacing "Foundation Agents".

**Architecture:** Three independent PRs. PR A edits landing copy in `dashboard/landing/src` and rebuilds the shipped bundle. PR B adds a nullable `category` column to the agents store (both SQLite/Postgres twins), an optional `category` body field on agent create/patch (zero new routes), and a category stamp in the marketplace clone flow. PR C lands the shelf UI in `app.html`/`app.js`, the Community category chips, the recategorized + new `marketplace.json` templates, and the app-side copy sweep. PR B deploys before PR C (Render lags Vercel 10–40 min; the frontend NULL-fallback keeps the UI correct against an old backend).

**Tech Stack:** Vanilla JS + static HTML (`dashboard/frontend`, no build step), Vite/React landing (`dashboard/landing`), FastAPI + SQLite/Postgres twins, pytest source-guard tests (`dashboard/backend/tests/_frontend_source.py` helpers).

## Global Constraints

**Voice & tone (binding for every string in this plan and every future string):**
- Register: a private banker explaining a capability to a longtime client. Short declarative sentences, concrete nouns. The test for every sentence: *would a private banker say this out loud about the client's money?*
- Finance vocabulary is trust **currency** (keep: backtest, Sharpe ratio, max drawdown, long-only, paper trading w/ gloss, Magnificent 7). Technology vocabulary is trust **cost**.
- Anti-flippancy: never describe trading with *game, play, toy, fun, magic, just, only takes, brains*. Speed claims must name the rigor behind them. Canonical counterexample: "Here it costs one sentence and a few minutes."
- Glossary (banned → replacement): LLM→"AI model"/"AI-powered" (sole exception: the category name "Prompting LLMs" + its description, verbatim); prompt (noun)→"trading instruction"/"Instruction"; bare "model"→"AI model"; frontier model→"leading AI model"; API key→"access key" in app UI (developer bridge once: "(the API key in the SDK and docs)"); session→drop or "sign-in"; pipeline→"multi-step strategy"; token cost→"AI cost"; deterministic→"repeatable"; managed provider/runtime→"hosted and managed by Agentic Trading Lab"; deploy job→describe the outcome; sleeve→plain sentence; brains→"AI models"; localhost→"a desktop computer"; bare Discord→"our Discord community" on first use per surface.
- "agent" is KEPT (it's the brand); gloss once per surface: "an AI trading assistant that follows your written instruction."
- Canonical no-real-money sentence (never shorten — Robinhood live trading exists as opt-in; never say "real-time" — bars are hourly): **"Every test here uses simulated money. Real money is involved only if you explicitly connect a brokerage account and turn on live trading."**

**Capability truth (gates every claim and every shelf):** backtestable = US stocks (Alpaca hourly, DJIA-30) + China A-shares (iFinD hourly, real T+1 ledger) only. Crypto is a display-only ticker (`GET /ticker`, `quotes.py` — never wired to the engine). "Futures"/vn.py (`vnpy_simulation.py`) is synthetic US-equity test data. **No Crypto or Futures shelf, tag, or copy in v1 — cut, not "coming soon".**

**Repo invariants (violating any is a plan failure):**
- Store twins change in the **same commit** (`domain/agents/repository.py` + `repository_postgres.py`); DDL as **literal strings only** (f-string ALTERs are invisible to the parity guard); guard compares column NAMES only, so hand-match nullability/type. `test_store_twin_parity.py` needs **no edits** — it parses DDL from source.
- **Zero new routes.** `category` rides existing create/patch bodies. Do NOT touch `test_app_composition.py::EXPECTED_FULL_CONTRACT` or `test_router_move.py::EXPECTED_AGENT_ROUTES`; if they redden, the change grew a route and must shrink back.
- `patchAgent`'s truthy `[]` (`dashboard/frontend/js/agent-editor.js:823`): `if (pipeline) payload.pipeline = ...` — `[]` truthy **on purpose** (clears the pipeline). Never "tidy" it.
- Capital validators: `cash_allocation` `ge=0` vs `backtest_allocation` `ge=1` — different on purpose; leave them alone.
- Never touch `DEFAULT_AGENT_PROVISION_GUARD_PREFIX` / `defaultAgentProvisionGuardKey()` (`app.js:230–251`) — display-name rename is safe; a prefix change re-provisions duplicates for every user.
- Seed DB: never `git add -A`; the new `category` ALTER makes the lazy-ALTER-mutates-committed-DB trap live on this exact branch. Check `git status` shows no `backtest.db`/`-wal` changes before every commit.
- Marketplace catalog is `lru_cache`d — tests touching it call `reload_marketplace_catalog()`.
- `#modelSelect` stays in the DOM with its 9 options (`test_ifind_ashare_frontend.py`, `test_vnpy_simulation_frontend.py` pin it).
- Landing: edit `dashboard/landing/src` only, `npm run build`, ship the rebuilt output; `test_frontend_bundle_integrity.py` guards the shipped bundle — read it before the first landing commit and keep it green. Never hand-edit `frontend/index.html`'s built parts.
- Frontend source-guard tests: strip comments before `not in` assertions; scope to the narrowest branch (`tests/_frontend_source.py` helpers).
- H6 leaderboard guard untouched; nothing here may alter `llm_decisions` accounting.
- Cache-buster: `app.html:1676` is `app.js?v=62` at plan commit time (it moved twice on 2026-08-04 alone — the #284/#285 auth PRs bumped it) → PR C bumps to the next free number after re-reading the current value. Open PR #289 (A-share lot execution) touches `app.html`/`app.js` but does not line-collide; re-check `gh pr list --state open` at implementation time anyway.
- The fridge/shelf metaphor never appears in product copy — UI says "sections"/"categories"; "shelf" lives only in code comments and this doc.

## Owner decisions (RESOLVED 2026-08-04)

1. **RESOLVED — footer identity:** the operating entity is **SecureFinAI Lab**; use a standard open-source footer (exact copy in Task A3; the contact path is the GitHub repository link). All three personas ranked "who runs this?" the #1 trust-killer. (Privacy/terms links remain tracked as issue #278.)
2. **RESOLVED — AI Hedge Fund attribution:** keep "based on the open-source AI Hedge Fund project by virattt" in the card description (plan default confirmed).
3. **RESOLVED — "Prompting LLMs"** ships verbatim. Recorded tension: the 70+ persona still stumbles on "LLMs"; mitigation is the acronym-free helper copy under the section, and an optional later tooltip gloss.
4. **DEFERRED — optional layout follow-ups (not in this plan):** replacing the landing hero's `agent-playground.exe` terminal chrome with a plain "conversation" card, and demoting the "Join Discord" CTA below "Get Started" — both flagged by personas, both layout (not copy) changes; revisit after v1.
5. **CONFIRMED — docs updates are deferred until after implementation** (see Coordination follow-ups §1).

---

# PR A — Landing copy (`dashboard/landing/src`)

### Task A1: Landing copy edits + rebuild

**Files:**
- Modify: `dashboard/landing/src/components/home/WhyCare.tsx`, `Talk.tsx`, `Test.tsx`, `Hero.tsx`, `Race.tsx`, `Navbar.tsx` (Discord label only), `dashboard/landing/index.html` (meta description)
- Rebuild: `cd dashboard/landing && npm run build` → ships as `dashboard/frontend/index.html` + `assets/`
- Test: `dashboard/backend/tests/test_landing_value_band.py` (existing), `test_frontend_bundle_integrity.py` (existing), plus new guards below

**Interfaces:** none consumed/produced; independent of PRs B/C.

Copy table (every row is exact; current text verified against source 2026-08-04):

| File:line | Current | Proposed |
|---|---|---|
| WhyCare.tsx:62–64 | "Normally that means writing code, buying data, and waiting months to find out you were wrong. Here it costs one sentence and a few minutes." | "Normally that means writing code, buying data, and waiting months to find out you were wrong. Here you get a rigorous, data-backed test in minutes — no code, no data subscription, no waiting months." |
| WhyCare.tsx:27–29 | "Pick the model / Same idea, different brains: Claude, GPT, Gemini, DeepSeek, Qwen." | "Pick the AI model / Same idea, different AI models — Claude, GPT, Gemini, and more, all available to try." |
| WhyCare.tsx:32–34 | "Bring your own agent / A Python SDK and an API, if you would rather write the code." | "For developers: bring your own agent / A Python toolkit (SDK), if you would rather write the code." |
| Talk.tsx:13 | "Describe your idea, in a sentence" | "Describe your idea in plain language" |
| Test.tsx:169–172 | "Est. token cost" | "Est. AI cost" |
| Hero.tsx:281–288 | metric row (Return +41.2% / Sharpe 1.31 / Win Rate 68% / Max DD −9.4%) with no label at point of display | same metrics + the words "Illustrative example" rendered directly on/beside the metric row |
| Hero.tsx (under headline) | — | add one line: "Every test here uses simulated money. Real money is involved only if you explicitly connect a brokerage account and turn on live trading." |
| Race.tsx:16–30 (rendered ~:130) | `SAMPLE_CURVES`/`SAMPLE_STANDINGS` presented with a "7d ago → Now" live-race framing, no label | add "Illustrative example" visibly on the chart/standings block (same treatment as Hero) |
| Navbar.tsx + Hero CTA | "Join Discord" | "Join our Discord community" (label only; demotion is a follow-up, see Owner decision 4) |
| landing `index.html:6–7` meta | "…Build and backtest LLM trading agents in the lab." | "…Test AI trading agents on real market data — no code required." |
| paper-trading first mention on landing (Test.tsx section) | bare "paper trading" | first use reads "paper trading — practice trading with simulated money at live market prices" |

- [ ] **Step 1: Write failing source guards** — add `dashboard/backend/tests/test_landing_copy_register.py` using `_frontend_source.py` helpers, asserting against the **shipped** `dashboard/frontend/index.html` + `assets/*.js` (comment-stripped): `"one sentence and a few minutes"` absent; `"different brains"` absent; `"Est. token cost"` absent; `"Illustrative example"` present at least twice; the canonical no-real-money sentence present.
- [ ] **Step 2: Run them; confirm they FAIL** — `pytest dashboard/backend/tests/test_landing_copy_register.py -v`
- [ ] **Step 3: Apply the table above in `dashboard/landing/src`** (source only, never the bundle).
- [ ] **Step 4: Rebuild + ship** — `cd dashboard/landing && npm run build`, copy the output per the procedure `test_frontend_bundle_integrity.py` encodes (read that test first; it is the contract for how the bundle ships).
- [ ] **Step 5: Run** `pytest dashboard/backend/tests/test_landing_copy_register.py dashboard/backend/tests/test_frontend_bundle_integrity.py dashboard/backend/tests/test_landing_value_band.py -v` → all PASS.
- [ ] **Step 6: Commit** (verify `git status` shows no `backtest.db`/`-wal` mutation): `git commit -m "ux(landing): serious-register copy — no flippancy, no AI jargon, illustrative labels"`

### Task A2: Auth modal error copy

**Files:** Modify `dashboard/landing/src` auth modal source (the string ships in `frontend/index.html:384`); Test: extend `test_landing_copy_register.py`.

- [ ] **Step 1:** Guard: shipped bundle contains "Something went wrong. Please try again." and not a bare "Something went wrong." Run → FAIL.
- [ ] **Step 2:** Change "Something went wrong." → "Something went wrong. Please try again." in the auth modal source; rebuild as in A1 Step 4.
- [ ] **Step 3:** Run guards + bundle integrity → PASS. Commit: `ux(landing): auth error gives a next step`.

### Task A3: Footer identity line

**Files:** Modify `dashboard/landing/src/components/home/FooterCTA.tsx:25`; Test: extend `test_landing_copy_register.py`.

Current: `"© 2026 Agentic Trading Lab. All rights reserved."`
Proposed: `"© 2026 SecureFinAI Lab · Agentic Trading Lab is an open-source research platform by SecureFinAI Lab, organizer of the SecureFinAI Contest 2026 · GitHub"` — where "GitHub" links to the repository (the standard open-source contact path). Match the surrounding footer's existing link markup for the GitHub anchor.

- [ ] **Step 1:** Failing guard: shipped bundle contains "© 2026 SecureFinAI Lab" and "open-source research platform". **Step 2:** FAIL → edit source → rebuild per A1 Step 4 → guards + bundle integrity PASS. **Step 3: Commit** `ux(landing): footer names SecureFinAI Lab as the operating entity`.

---

# PR B — Backend category taxonomy (deploys before PR C)

### Task B1: Taxonomy constant + normalizer

**Files:**
- Create: `dashboard/backend/domain/agents/taxonomy.py`
- Test: `dashboard/backend/tests/test_agent_taxonomy.py`

**Interfaces:**
- Produces: `AGENT_CATEGORIES: frozenset[str]` = `{"prompting_llms", "us_stocks", "cn_ashares"}`; `normalize_category(value) -> Optional[str]` (lenient: unknown/legacy → `None`). Consumed by Tasks B3 and B4.

- [ ] **Step 1: Failing test:**

```python
from dashboard.backend.domain.agents.taxonomy import AGENT_CATEGORIES, normalize_category

def test_categories_whitelist():
    assert AGENT_CATEGORIES == {"prompting_llms", "us_stocks", "cn_ashares"}

def test_normalize_valid_passthrough_and_case():
    assert normalize_category("us_stocks") == "us_stocks"
    assert normalize_category(" US_STOCKS ") == "us_stocks"

def test_normalize_unknown_and_legacy_to_none():
    assert normalize_category("Foundation") is None   # legacy marketplace value
    assert normalize_category("Hosted") is None
    assert normalize_category("") is None
    assert normalize_category(None) is None
```

- [ ] **Step 2:** `pytest dashboard/backend/tests/test_agent_taxonomy.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement:**

```python
"""Shared agent category whitelist ("shelves"). Slugs are stored; display names live in the frontend."""
from typing import Optional

AGENT_CATEGORIES = frozenset({"prompting_llms", "us_stocks", "cn_ashares"})


def normalize_category(value: object) -> Optional[str]:
    if value is None:
        return None
    slug = str(value).strip().lower()
    return slug if slug in AGENT_CATEGORIES else None
```

- [ ] **Step 4:** Rerun → PASS. **Step 5: Commit** `feat(agents): category taxonomy whitelist`.

### Task B2: `category` column on both store twins (ONE commit)

**Files:**
- Modify: `dashboard/backend/domain/agents/repository.py` (CREATE TABLE ~112–129; lazy-migration `PRAGMA table_info` block ~143–186; `_public_agent()` ~70–91; `create_agent()` param + INSERT ~197–256; `update_agent()` writable fields)
- Modify: `dashboard/backend/domain/agents/repository_postgres.py` (CREATE TABLE ~69–89 **and** `ADD COLUMN IF NOT EXISTS` ~97–131 — per the comment block at 44–58 a new column must appear in BOTH; mirror `create_agent`/`update_agent` signatures)
- Test: extend `dashboard/backend/tests/test_agents_api.py` (SQLite path; reuse its existing client/store fixtures)

**Interfaces:**
- Produces: `create_agent(..., category: Optional[str] = None)`, `update_agent` accepting `category` (None clears), `_public_agent()` dicts carrying `"category"`. Consumed by B3/B4/C3.

- [ ] **Step 1: Failing tests** (SQLite twin; parity guard covers the Postgres twin's DDL automatically):

```python
def test_agent_row_carries_category(agent_store):
    agent = agent_store.create_agent(name="Cat Test", agent_type="builtin", category="us_stocks")
    assert agent["category"] == "us_stocks"

def test_category_defaults_to_none_and_survives_update_clear(agent_store):
    agent = agent_store.create_agent(name="Cat Default", agent_type="builtin")
    assert agent["category"] is None
    agent_store.update_agent(agent["agent_id"], category="cn_ashares")
    assert agent_store.get_agent(agent["agent_id"])["category"] == "cn_ashares"
    agent_store.update_agent(agent["agent_id"], category=None)
    assert agent_store.get_agent(agent["agent_id"])["category"] is None

def test_category_column_added_to_preexisting_table(tmp_path):
    # simulate a pre-upgrade DB: table without the column, then re-init triggers the lazy ALTER
    ...  # follow the existing lazy-migration test pattern in this file for prior columns
```

(Adapt fixture names to what `test_agents_api.py` actually uses; the third test copies the established prior-column migration test pattern in the same file.)

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement in BOTH twins: `category TEXT` in each CREATE TABLE; SQLite guarded literal `ALTER TABLE external_agents ADD COLUMN category TEXT` in the PRAGMA-probe block; Postgres literal `ALTER TABLE external_agents ADD COLUMN IF NOT EXISTS category TEXT`; add to `_public_agent()`, `create_agent()`, `update_agent()` in both.
- [ ] **Step 4:** `pytest dashboard/backend/tests/test_agents_api.py dashboard/backend/tests/test_store_twin_parity.py -v` → PASS (parity test needs no edits — it parses the DDL you just wrote; if it fails, your DDL is f-stringed or diverged).
- [ ] **Step 5:** Confirm `git status` shows no seed-DB mutation. **Commit** (single commit, both twins): `feat(agents): nullable category column on both store twins`.

### Task B3: Optional `category` on create/patch APIs (zero new routes)

**Files:**
- Modify: `dashboard/backend/api/routers/agents.py` (create + patch request models and their handlers)
- Test: extend `dashboard/backend/tests/test_agents_api.py`

**Interfaces:**
- Consumes: B1's `AGENT_CATEGORIES`/`normalize_category`, B2's store params.
- Produces: `POST /api/v1/agents` + `PATCH /api/v1/agents/{agent_id}` accepting optional `category`; unknown → 422; `GET /api/v1/agents` / `/agents/builtin` echo it via `_public_agent()`.

- [ ] **Step 1: Failing tests:**

```python
def test_create_agent_with_category_round_trips(client, auth_headers):
    r = client.post("/api/v1/agents", json={"name": "Shelved", "agent_type": "builtin",
                                            "category": "us_stocks"}, headers=auth_headers)
    assert r.status_code in (200, 201)
    listed = client.get("/api/v1/agents", headers=auth_headers).json()
    assert any(a.get("category") == "us_stocks" for a in listed["agents"])

def test_create_agent_unknown_category_422(client, auth_headers):
    r = client.post("/api/v1/agents", json={"name": "Bad Shelf", "agent_type": "builtin",
                                            "category": "crypto"}, headers=auth_headers)
    assert r.status_code == 422

def test_patch_category_and_patch_to_null(client, auth_headers, existing_agent):
    ok = client.patch(f"/api/v1/agents/{existing_agent}", json={"category": "cn_ashares"}, headers=auth_headers)
    assert ok.status_code == 200 and ok.json()["category"] == "cn_ashares"
    cleared = client.patch(f"/api/v1/agents/{existing_agent}", json={"category": None}, headers=auth_headers)
    assert cleared.status_code == 200 and cleared.json()["category"] is None
```

(Adapt fixtures to the file's existing client/auth/agent fixtures.)

- [ ] **Step 2:** Run → FAIL. **Step 3:** Add `category: Optional[str] = None` to both request models; in handlers: `if body.category is not None and body.category not in AGENT_CATEGORIES: raise HTTPException(422, ...)` (strict at the API; `normalize_category` stays for lenient internal callers). Do not touch `cash_allocation`/`backtest_allocation` validators.
- [ ] **Step 4:** Run the new tests **and** `pytest dashboard/backend/tests/test_app_composition.py dashboard/backend/tests/test_router_move.py -v` → all PASS (route freezes untouched proves zero new routes).
- [ ] **Step 5: Commit** `feat(agents): optional category on create/patch (whitelist-validated)`.

### Task B4: Clone stamps category (mall → fridge)

**Files:**
- Modify: `dashboard/backend/domain/agents/service.py::clone_marketplace_template` (~373–413)
- Test: extend the marketplace tests beside the existing clone tests

**Interfaces:** Consumes B1 + B2. Produces: cloned agents carrying the template's normalized category.

- [ ] **Step 1: Failing test** (remember `reload_marketplace_catalog()` — the catalog is `lru_cache`d; use a monkeypatched catalog fixture with `"category": "us_stocks"`):

```python
def test_clone_stamps_normalized_category(client, auth_headers, categorized_catalog):
    r = client.post("/api/v1/agents/marketplace/balanced-starter/clone", headers=auth_headers)
    assert r.status_code in (200, 201)
    assert r.json()["category"] == "us_stocks"

def test_clone_legacy_category_stamps_none(client, auth_headers):  # live catalog still says "Foundation"
    r = client.post("/api/v1/agents/marketplace/momentum-scout/clone", headers=auth_headers)
    assert r.json()["category"] is None
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** In `clone_marketplace_template`, pass `category=normalize_category(template.get("category"))` into `create_agent`.
- [ ] **Step 4:** Run → PASS. **Step 5: Commit** `feat(agents): marketplace clone stamps category`. Open PR B; after merge, **wait for the Render deploy to go live** (probe: `GET /api/v1/agents/builtin` response rows contain a `category` key) before merging PR C.

---

# PR C — Frontend shelves, Community chips, data, app copy sweep

### Task C1: `dashboard/config/marketplace.json` — recategorize + fix copy + 3 new seed templates

**Files:** Modify `dashboard/config/marketplace.json`. Test: existing marketplace tests + B4's tests keep passing (`reload_marketplace_catalog()` in any new test).

Edits to existing templates:
- `balanced-starter`: `category` `"Foundation"` → `"prompting_llms"`; description "A simple foundation agent that…" → "A simple starter agent that diversifies across strong stocks, buys dips, and takes profits after run-ups."
- `momentum-scout`: `category` → `"prompting_llms"` (description already clean).
- `pipeline-analyst`: `category` `"Advanced"` → `"us_stocks"`; description "A three-step pipeline: …" → "A three-step strategy: gather market facts, convert them into signals, then produce executable orders."; tags `["pipeline", "multi-step"]` → `["multi-step strategy", "official template"]`.
- `ai-hedge-fund`: `category` `"Hosted"` → `"us_stocks"`; description → "A hosted panel of AI analysts that weigh in on every trade, run through Agentic Trading Lab's long-only backtest engine. Based on the open-source AI Hedge Fund project by virattt." (attribution KEPT — Owner decision 2); tags `["multi-agent", "fundamentals", "first-party"]` → `["analyst team", "fundamentals", "official template"]`; author stays "virattt / Agentic Trading Lab".
- `balanced-starter`/`momentum-scout` tags: `["starter", "diversified"]` / `["momentum", "trend"]` — keep (already plain).

New templates (same shape as `balanced-starter`; `presetKey: "simple_instruction"`, same `outputFormat` string verbatim):

```json
{
  "template_id": "blue-chip-steady",
  "name": "Blue-Chip Steady",
  "model_name": "anthropic/claude-haiku-4-5",
  "description": "Buy and hold a handful of the strongest Dow companies, selling only when a position deteriorates badly. Mirrors the buy-and-hold benchmark on our leaderboard.",
  "category": "us_stocks",
  "tags": ["buy and hold", "blue chips"],
  "author": "Agentic Trading Lab",
  "pipeline": [{
    "id": "sub_blue_chip_steady",
    "presetKey": "simple_instruction",
    "label": "Trading instruction",
    "prompt": "Buy and hold a handful of the strongest Dow companies. Sell only if a company's position deteriorates badly. Do not chase short-term moves.",
    "outputFormat": "JSON: { \"orders\": [{ \"symbol\": \"...\", \"side\": \"buy|sell|hold\", \"qty\": number, \"order_type\": \"market|limit\", \"limit_price\": number|null, \"reason\": \"...\" }] }"
  }]
},
{
  "template_id": "even-split-dow",
  "name": "Even-Split Dow",
  "model_name": "anthropic/claude-haiku-4-5",
  "description": "Spread the money evenly across all available Dow stocks and keep the split even. Mirrors the equal-weight benchmark on our leaderboard.",
  "category": "us_stocks",
  "tags": ["equal weight", "diversified"],
  "author": "Agentic Trading Lab",
  "pipeline": [{
    "id": "sub_even_split_dow",
    "presetKey": "simple_instruction",
    "label": "Trading instruction",
    "prompt": "Spread the money evenly across all available Dow stocks and keep the split even, rebalancing when any position drifts far from its equal share.",
    "outputFormat": "JSON: { \"orders\": [{ \"symbol\": \"...\", \"side\": \"buy|sell|hold\", \"qty\": number, \"order_type\": \"market|limit\", \"limit_price\": number|null, \"reason\": \"...\" }] }"
  }]
},
{
  "template_id": "ashare-steady-t1",
  "name": "A-Share Steady (T+1)",
  "model_name": "anthropic/claude-haiku-4-5",
  "description": "A patient strategy for Chinese A-shares, built for that market's rule that shares bought today cannot be sold until the next trading day.",
  "category": "cn_ashares",
  "tags": ["a-shares", "patient"],
  "author": "Agentic Trading Lab",
  "pipeline": [{
    "id": "sub_ashare_steady",
    "presetKey": "simple_instruction",
    "label": "Trading instruction",
    "prompt": "Trade the available Chinese A-share stocks patiently. Because shares bought today cannot be sold until the next trading day, avoid quick in-and-out trades; build positions you are willing to hold overnight.",
    "outputFormat": "JSON: { \"orders\": [{ \"symbol\": \"...\", \"side\": \"buy|sell|hold\", \"qty\": number, \"order_type\": \"market|limit\", \"limit_price\": number|null, \"reason\": \"...\" }] }"
  }]
}
```

Registry mapping is now complete and documented: `buy_hold`→Blue-Chip Steady, `equal_weight_index`→Even-Split Dow, `equal_weight_buyhold` (the A-share benchmark, `registry.py:19–25`)→A-Share Steady (T+1) mirrors its spirit; `mean_variance`/`market_index` are math/index engines with no prompt equivalent; `llm_agent` *is* the builtin mechanism. Nobody needs to "finish" this mapping later.

- [ ] **Step 1:** Failing test: catalog loads, `template_id`s unique, every `category` ∈ `AGENT_CATEGORIES` ∪ `{None}`, 7 templates. **Step 2:** FAIL → apply edits → PASS (with `reload_marketplace_catalog()`). **Step 3:** Update B4's `test_clone_legacy_category_stamps_none` (live catalog no longer has legacy values — repoint it at a monkeypatched legacy fixture). **Step 4: Commit** `feat(community): shelf categories + three honest seed templates`.

### Task C2: My Agents shelf sections (`app.html`) + app copy riding this surface

**Files:** Modify `dashboard/frontend/app.html` (~897–926: the two `agents-category` blocks). Test: new `dashboard/backend/tests/test_frontend_shelves.py` (source guards, `_frontend_source.py` helpers).

Replace the two sections (Foundation Agents / External Agents) with four static `agents-category` blocks patterned on the existing markup, new per-shelf grid/empty-state/footer IDs:

| # | Header | Subtitle under header |
|---|---|---|
| 1 | Prompting LLMs | Prompt state-of-the-art LLMs to backtest on real market data. |
| 2 | U.S. Stock Trading | Ready-made strategies for U.S. blue-chip stocks, tested hour by hour on real market data. |
| 3 | China A-Share Trading | Strategies for Chinese A-share stocks, following that market's own next-day (T+1) trading rules. |
| 4 | For Developers: Connected Agents | Run your own trading program against our backtests. Requires an access key. |

The old subtitle **"A prompting game — give it money, an instruction, and a model, then backtest."** is deleted here — it was the #1 trust-killer in all three persona walkthroughs; this task must not ship without removing it. Also add near the capital/live-trading controls area of My Agents the canonical no-real-money sentence (Global Constraints) as a quiet helper line.

- [ ] **Step 1:** Failing guards: comment-stripped `app.html` contains all four headers + subtitles; contains neither "Foundation Agents" nor "A prompting game"; contains the canonical no-real-money sentence. **Step 2:** FAIL → edit → PASS. **Step 3: Commit** `ux(agents): four shelf sections replace Foundation/External split`.

### Task C3: Shelf rendering (`app.js`)

**Files:** Modify `dashboard/frontend/app.js` — `renderAgentCategories` (~1298–1360), `agentGridPage` (471, reset at 1021), empty states (1325–1326), default agent name (1566), placeholder card (1349–1352). Test: extend `test_frontend_shelves.py`.

**Interfaces:** Consumes `category` from `GET /api/v1/agents` (B2/B3); NULL-fallback keeps working against an old backend.

Add the declarative config and generalize the existing two-grid render into a loop over it:

```js
const AGENT_SHELVES = [
  { key: 'prompting_llms', title: 'Prompting LLMs',
    match: (a) => a.agent_type === 'builtin' && (!a.category || a.category === 'prompting_llms') },
  { key: 'us_stocks', title: 'U.S. Stock Trading',
    match: (a) => a.agent_type === 'builtin' && a.category === 'us_stocks' },
  { key: 'cn_ashares', title: 'China A-Share Trading',
    match: (a) => a.agent_type === 'builtin' && a.category === 'cn_ashares' },
  { key: 'external', title: 'For Developers: Connected Agents',
    match: (a) => a.agent_type !== 'builtin' },
];
```

Predicates are mutually exclusive by construction; each agent renders on exactly one shelf; legacy `category=NULL` builtins (all existing prod rows) land on Prompting LLMs — no backfill. Re-key `agentGridPage` by shelf key (footer ←/→ handlers at ~6702–6703 work unchanged). Search narrows all shelves and resets pagination (existing behavior, looped). Empty states (all shelves share "No agents match your search." when searching):

- Prompting LLMs: "You don't have any agents yet. Create one and test your first trading idea." (always visible — onboarding surface; replaces `app.js:1325–1326`)
- U.S. Stock Trading: "Nothing here yet. Add a ready-made U.S. stock strategy from Community." ("Community" links to the Community page with this category chip pre-selected — C4)
- China A-Share Trading: "Nothing here yet. Add an A-share strategy from Community." (same link pattern)
- For Developers: keep the placeholder card, retitled "Connect your own trading program" / "For developers: run your own trading program against our backtests using an access key."

Rename `'My Foundation Agent'` → `'My Trading Agent'` (`app.js:1566`) — display name only; `ensureDefaultFoundationAgent()` flow and guard keys untouched (Global Constraints); the starter lands on the top shelf via the NULL fallback, so provisioning needs no category change.

- [ ] **Step 1:** Failing guards: `AGENT_SHELVES` present with the four keys; "No foundation agents" absent; "My Foundation Agent" absent; "My Trading Agent" present; `DEFAULT_AGENT_PROVISION_GUARD_PREFIX` byte-identical to `main`'s value. **Step 2:** FAIL → implement → PASS. **Step 3:** Manual check: `uvicorn dashboard.backend.app:app` with a scratch `DATABASE_PATH` copy, load `/app?view=playground`, confirm four sections render and the starter agent sits under Prompting LLMs. **Step 4: Commit** `feat(agents): shelf-based rendering with NULL-fallback`.

### Task C4: Community chips + CTA verb + label map

**Files:** Modify `dashboard/frontend/app.js` marketplace render (~1700–1800: chips, CTA ~1757, submeta ~1768). Test: extend `test_frontend_shelves.py`.

- Chip row above the grid: `All` (default) · `Prompting LLMs` · `U.S. Stock Trading` · `China A-Share Trading` — labels from a shared `SHELF_LABELS` map derived from `AGENT_SHELVES`; chips AND text search compose; unknown/legacy categories appear under `All` only (after C1 there are none at ship time; during the Render deploy lag the old catalog behaves this way by design).
- Unify the CTA: `app.js:1757` renders "Copy to My Agents" for one card and "Add to My Agents" for the rest → **"Add to My Agents"** everywhere (canonical per PR #253).
- Card submeta (~1768) currently prints raw `template.category` and the raw model slug: route category through `SHELF_LABELS` (never show a slug); map model slugs to friendly names via one shared table — `anthropic/*`→"Powered by Claude …", `nvidia/nemotron*`→"Powered by NVIDIA Nemotron", `deepseek/*`→"Powered by DeepSeek", `openai/*`→"Powered by GPT" — full slug only in expandable details.
- Fallback description `"Open agent template."` (~1774) → `"No description provided yet."`
- Wire C3's empty-shelf "Community" links to open Community with the matching chip pre-selected (in-memory state or view param — no route/API change).
- Add near the Add to My Agents flow (hosted/live-capable templates) the canonical no-real-money sentence once per page.

- [ ] **Step 1:** Failing guards: "Copy to My Agents" absent; single CTA string present; `SHELF_LABELS` present; raw-slug regex (`nvidia/|anthropic/`) absent from the rendered-submeta template-literal branch (narrowest-branch scoping). **Step 2:** FAIL → implement → PASS + manual chip/filter check as in C3 Step 3. **Step 3: Commit** `feat(community): category chips share the My Agents taxonomy`.

### Task C5: App copy sweep (modals, toasts, Competition, Account, Playground)

**Files:** Modify `dashboard/frontend/app.html`, `dashboard/frontend/app.js`. Test: `dashboard/backend/tests/test_app_copy_register.py` (new; guards on the highest-risk strings marked ★).

| File:line | Current | Proposed |
|---|---|---|
| ★ app.html addAgent modal subtitle | "Register an external agent to get a persistent session and API key. Log in to keep agents across devices and sessions." | "Register your own agent and receive a secure access key. Sign in to keep your agents across devices." |
| app.html addAgent option | "Hosted agent powered by a frontier model — chat with it on Discord" | "Hosted agent powered by a leading AI model — you can also chat with it in our Discord community" |
| app.html addAgent option | "Get an API key and session for your backtest client" | "For developers: get an access key to connect your own program" |
| app.html createBuiltinAgent subtitle | "A hosted agent powered by a frontier model. It appears on your agent cards here and becomes selectable from the Agentic Trading Discord server with `/agent`." | "A hosted agent powered by a leading AI model. It appears on your agent cards here and can also be used from our Discord community with `/agent`." |
| app.html builtin placeholder | "What is this agent's edge?" | "What makes this agent different?" |
| app.html agentCredentials subtitle | "Your agent is ready. Use the API key below to connect your trading client to Agentic Trading Lab." | "Your agent is ready. Use the access key below to connect your own program to Agentic Trading Lab. (This is the API key in the SDK and docs.)" |
| app.html credentials label | "API key" | "Access key" |
| app.js:960 | "New API key" | "New access key" |
| app.js:1257 | "Create a new API key for "X"? The current key will stop working immediately." | "Create a new access key for "X"? The current key stops working right away — any connected program must switch to the new key." |
| app.js:2020–2021 | "New API key created" / "…Update your client — the old key no longer works." | "New access key created" / "…Update your program — the old key no longer works." |
| app.html agentEditor label | "Managed provider / model" ("Managed by Agentic Trading Lab") | "Hosted AI model" ("Managed for you by Agentic Trading Lab") |
| ★ app.html agentEditor capital note | "Reserved from My Portfolio. Real sleeve." | "Reserved from your My Portfolio balance while this agent paper-trades. Backtests use a separate simulated amount and never touch it." (My Portfolio is the simulated $10k ledger — never say "real money" here) |
| app.html agentEditor replace note | "This agent currently uses a custom multi-step pipeline. Saving this instruction replaces it." | "This agent currently uses a custom multi-step strategy. Saving this instruction replaces it." |
| app.html aihedge title + text | "AI Hedge Fund analyst committee" / "Choose the upstream analysts that compose this agent's strategy. The hosted OpenRouter model and runtime controls are managed by Agentic Trading Lab." | "AI Hedge Fund analyst panel" / "Choose the analysts that shape this agent's strategy. The AI model and its settings are hosted and managed by Agentic Trading Lab." |
| app.html label | "Financial Datasets API key" | "Financial Datasets access key (from financialdatasets.ai)" |
| app.html credential note | "Encrypted in credential storage. The value is never returned to this browser after saving." | "Encrypted and stored securely. For your protection, it is never shown again after you save." |
| app.html backtest hint | "Multi-step agent pipelines can take several minutes (timeout: 10 min)." | "Multi-step strategies can take several minutes (limit: 10 minutes)." |
| app.html home leaderboard column | "Model" | "AI Model" |
| app.html home + community | "Join Discord" | "Join our Discord community" |
| app.html homePlaygroundTitle | "agent-playground.exe" | "Example: a conversation with your agent" |
| app.html resource subtitles | "Guides and API reference" / "Chat with other builders" / "Open source and examples" | "Guides and reference documentation" / "Talk strategy with other members" / "Source code and worked examples" |
| ★ app.html backtest config row | "Prompt" | "Instruction" |
| app.html backtest config row | "Decision source" | "Decision method" |
| app.html market data notice | "vn.py simulated bars · deterministic · no LLM calls" | "Simulated practice data — repeatable results, rule-based decisions only (no AI)" |
| app.html Sharpe tooltip | "Annualized for hourly data (sqrt(252*6.5))" | "Risk-adjusted return, annualized from hourly results." |
| app.html Competition contest header | "SecureFinAI Contest 2026" (relationship to ATL unstated) | add one line beneath the contest title: "Organized by SecureFinAI Lab with Agentic Trading Lab." |
| ★ app.html Competition About | "…a paper-trading competition for LLM-powered agents." | "…a paper-trading competition for AI-powered agents." |
| ★ app.html Competition About | "LLM entries are added when the daily deploy job runs." | "New AI entries appear when the leaderboard next updates." (the daily job is unwired — issue #145; never say "automatically each trading day") |
| app.html Competition About | "…comparing provided LLM models, baseline strategies, and market indices." | "…comparing leading AI models, baseline strategies, and market indices." |
| app.html Participants empty | "The Ranking board shows models and baselines only." | "The Ranking board shows AI models and baseline strategies only." |
| app.html account description | "Signed-in profile and session." | "Your profile and sign-in details." |
| app.js:3185 | "Robinhood connection failed (reason). Use localhost and a desktop browser." | "Robinhood connection failed. Please try again on a desktop computer." |
| app.js:1264 | "Failed to create new API key" | "Couldn't create a new access key. Please try again." |
| app.js:3104 | "Could not start Discord linking. Are you signed in?" | "Couldn't start Discord linking. Please sign in and try again." |
| app.js:1799 | "Failed to add template" | "Couldn't add this template. Please try again." |
| app.js:1292 | "Failed to delete agent" | "Couldn't delete the agent. Please try again." |
| Paper Trading tab first "paper trading" mention | bare "paper trading" | gloss identical to landing: "paper trading — practice trading with simulated money at live market prices" |
| Keep unchanged | "Magnificent 7" (+ its "7 major tech companies" subtitle), Sharpe/drawdown/long-only vocabulary, sign-in form labels, Built-in/External API values | established finance/press terms and tested-clean strings; `agent_type` API values are NOT copy — never rename them |

Line numbers are pre-C2/C3 anchors — locate by quoted string, not by number.

- [ ] **Step 1:** Failing guards on the ★ rows (comment-stripped): "prompting game" absent (belt-and-braces with C2), "Real sleeve." absent, "LLM-powered" absent from app.html, "daily deploy job" absent, addAgent subtitle's "persistent session and API key" absent, backtest config "Prompt" label replaced. **Step 2:** FAIL → apply the table → PASS. **Step 3: Commit** `ux(app): plain-language register across modals, toasts, competition, account`.

### Task C6: Cache-buster + final verification

**Files:** Modify `dashboard/frontend/app.html:1676`.

- [ ] **Step 1:** `gh pr list --state open --json number,title` — confirm no open PR bumps `app.js?v=`.
- [ ] **Step 2:** Read the current value at `app.html:1676` (`v=62` at plan commit time; it moves often) → bump to the next number.
- [ ] **Step 3:** Full suite: `pytest dashboard/backend/tests/ -v` → green end-to-end (a red test IS a regression — except phantom `test_deleted_shim_is_not_importable` failures, which mean stale `__pycache__`: `rm -rf dashboard/backend/engines dashboard/backend/services`).
- [ ] **Step 4:** `git status` — no seed-DB/WAL mutation. **Commit** `chore(frontend): bump app.js cache-buster to v=61`. Open PR C (after PR B is live on Render — see B4 Step 5).

---

## Coordination follow-ups (not code in this plan)

1. **User-facing docs go stale the moment this ships** (owner-coordinated per docs policy; do NOT edit mid-session): `docs/source/lab/marketplace.rst` (lines 18, 36, 41 hardcode "Foundation"), `docs/source/lab/getting_started.rst:8` ("Use one of the Foundation Agents"), `docs/source/lab/key_features.rst:8` + `README.md:56` ("Agent Marketplace" phrasing — verify against the shipped Community UI), SDK README "API key" wording (keep the term for developers; add the "access key" bridge sentence). RTD has no CI build — run `sphinx-build -n -E` locally when these are edited.
2. **Staged (build later, never tease in UI):** Crypto shelf (needs a real crypto bar source, a `MarketProfile`, 24/7 session support, one runnable seed template) and Futures shelf (all that plus contract/margin/expiry modeling, which exists nowhere). Category picker in Configure (`agent-editor.js` — mind the truthy-`[]` invariant at :801) and Run-Backtest data-source pre-selection from category.
3. **Landing hero terminal-chrome swap + Discord CTA demotion** — Owner decision 4.

## Evidence base (session artifacts, temp — will not survive cleanup)

Three persona walkthroughs (Harold 72 retired owner / Diane 58 CEO / Marcus 49 ex-banker) against live screenshots, two design docs, and two adversarial verifications under `/tmp/claude-1000/-mnt-d-github-agent-trading-lab/5eeaed17-7b24-4a16-8c3e-205f0a375d42/scratchpad/audit/`. Unanimous persona verdicts baked in above: "A prompting game" was the #1 trust-killer; missing "who runs this?" identity is the #1 missing reassurance; a clickable Crypto/Futures shelf with nothing behind it would convert caution into distrust.
