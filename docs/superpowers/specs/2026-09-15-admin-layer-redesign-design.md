# Admin Layer Redesign

**Date:** 2026-09-15
**Status:** Design settled in a structured design-review session (a "grilling": one question at a time until each decision has a reason) held on the evening of 2026-09-15 US Eastern and recorded in the project notes under 2026-09-16 (24 of 25 questions). The two items that session left open, the document disposition in §1 and the PR boundaries in §13, are applied here as proposed and are ratified by merging this document. The session's written handoff, referred to below as "the handoff", is the summary this document was drafted from; every factual claim in it was re-verified at source on 2026-09-15 and the corrections are folded in.
**Delivery:** Seven pull requests: PR 0 (burner kill), PR C (the `/admin` page, frontend-first), PR T (store twin), PR A (rewrite), PR B (read paths), PR D (contract re-cut), and this documentation PR. Re-ordered 2026-09-16: C moved ahead of T/A/B so the advisor settles *what to show and how* on a live page before the backend is locked to it, and the re-cut became its own PR after B (§13, D19). Plans: `docs/superpowers/plans/2026-09-15-admin-layer-redesign-*.md`.
**Supersedes:** `2026-09-12-user-analytics-architecture-design.md` and `2026-09-03-admin-user-value-analytics-design.md` (both deleted by this PR). Narrows `2026-08-26-admin-user-analytics-design.md`. Aligns `2026-09-13-admin-user-groups-design.md`. See §1.
**Audience:** HaoXiang, who wrote the new overview page, and anyone maintaining the admin layer. This document is written to be read on its own. Where it relies on a repo convention it explains the convention (§0).

---

## 0. How to read this document

This is the single authoritative design for the administrator layer of Agentic Trading Lab: the analytics data model, the daily job that fills it, the nine admin analytics endpoints, and the `/admin` page that renders them. It replaces four overlapping design documents written between 2026-08-26 and 2026-09-13, none of which was ever retired and none of which any code file references.

If you are working on the **page**, read §2, §7, §8, §9 and §13. If you are working on the **backend**, read §4, §6, §10, §11, §12 and §13. §5 lists every decision with its reasoning, so that nobody "fixes" a deliberate choice back to its previous state.

Terms this document uses that are specific to this repository:

| Term | Meaning here |
|---|---|
| **Burner** | A background mechanism that spends an unbounded resource budget. Here: the two analytics code paths that exhausted the Postgres egress quota on 2026-09-11 (§2.2). |
| **Twin** | Every persistence store has two implementations with identical public methods: a SQLite class (local dev, tests, and ephemeral prod fallback) and a `*_postgres.py` sibling selected when the matching `*_DATABASE_URL` is set. `tests/test_store_twin_parity.py` asserts the pairs stay in step. |
| **users DB / runs DB** | Two separate Neon Postgres projects. `USERS_DATABASE_URL` holds accounts, sessions, the credit ledger, analytics tables and `user_group`. `AGENT_RUNS_DATABASE_URL` holds backtest run history (`agent_runs`, trades, equity). No code path joins across them. |
| **Rollups** | `analytics_daily_rollups`: anonymous per-day counters keyed by metric, event, billing mode, provider, model, outcome and error category. No user id, so no privacy horizon; retained indefinitely. |
| **Reaper tick** | A 60-second background thread in the web process that expires stale protocol runs and writes run heartbeats. Analytics maintenance was bolted onto it; §6 moves it off. |
| **IIFE module + `?v=N`** | The dashboard has no build step. Each frontend module is a plain script wrapped in an immediately-invoked function that exposes one global (`window.AdminTabs`, …), loaded by a `<script src="js/x.js?v=N">` tag whose `N` is bumped by hand to bust caches. Two dozen pytest modules lift functions out of these files with `tests/_frontend_source.py` and execute them under `node -e`. Inline `<script>` blocks are invisible to that harness. |
| **`require_admin`** | The FastAPI dependency that rejects a non-admin session with 403. It is applied at router level to every `/api/admin/*` route and is the only real admin gate in the system. |
| **Render / Vercel** | The backend runs on Render as a Docker service. The static frontend (`dashboard/frontend/`) is deployed twice: served by the Render process itself, and independently by Vercel from the same directory. Anything the page does before it calls the API therefore runs on a static host with no session access. |
| **Lifecycle segment** | The behavioural label the 09-03 design gave each user (`new`, `growing`, `core`, `at_risk`, `dormant`, …; §15 carries the definitions). Distinct from the legacy five **user states** it replaced and from the operational state (`blocked` / `needs_attention` / `healthy`). |
| **`user_group`** | The admin-editable source label shipped in PR #465: `internal / invited / organic / competition / partner / unknown`. The only categorical axis besides role and tier. |

---

## 1. Supersession

| Document | Disposition | What carries forward |
|---|---|---|
| `docs/superpowers/specs/2026-09-12-user-analytics-architecture-design.md` | **Absorbed and deleted.** | Its architecture, carried into §6 verbatim where unchanged, with the amendments in §6.9 and §12 marked in place. Its "Why now", freshness contract, data model, daily job, read paths, event-log discipline, read budget and field-visibility sections all survive under this document. |
| `docs/superpowers/plans/2026-09-12-user-analytics-architecture.md` | **Replaced and deleted** (never executed). | Its B and C task bodies are the base text of the PR A and PR B plans, edited for the corrections in §12. Its A tasks are not built; Requirement 1 of the handoff reduced them to PR 0 (§13). |
| `docs/superpowers/specs/2026-09-03-admin-user-value-analytics-design.md` | **Absorbed and deleted.** | Its definitions (activation, meaningful behaviour, inactivity clock, lifecycle segments and explainability, operational state, commercial value, retention cohorts) and its loading/empty/error and accessibility contract, carried into §15 unchanged. Its storage design ("current snapshot", "user-level daily history") was already overridden by the 09-12 document and is not carried. |
| `docs/superpowers/specs/2026-08-26-admin-user-analytics-design.md` | **Kept, narrowed.** | Collection Scope, Event Contract, Privacy and Security, and Retention remain authoritative there; this document defers to them (changing the event allowlist, the 1 KB property cap or the 180-day raw retention is a non-goal here). Its product structure, storage model, metrics, five explainable user states, freshness, delivery plan and testing sections are deleted from that file so that nothing in it looks live that is not. |
| `docs/superpowers/specs/2026-09-13-admin-user-groups-design.md` | **Kept, aligned.** | The `user_group` axis is a separate, shipped axis; this document defers to it for the taxonomy, the edit route and the group badge precedence. That file gains one paragraph: the 09-12 `cohort` axis was never built and is struck; `user_group` is the sole third axis. |
| Executed plans `2026-08-26-admin-user-analytics-{foundation,metrics,ui}.md`, `2026-09-03-admin-user-value-analytics.md`, `2026-09-13-admin-user-groups.md` | **Kept, one-line pointer added.** | They record why shipped code looks the way it does. Each gains a header line pointing here as the current design. |

The test used to decide each row: *does the architecture design defer to this document, or override it?* Defer means it describes a separate axis and is kept. Override means its content is absorbed and the file is retired. Leaving a retired document in place is what produced four partially-contradictory specs in three weeks, so absorbed documents are deleted, not marked.

---

## 2. Why this redesign exists

### 2.1 The request, and what the audit found

The request was to clean up the admin analytics page, which had been rushed for a CEO demo. PR #467 (merged 2026-09-13) shipped that page at `/admin-analytics` as a deliberate synthetic preview and said so in its body: every number on it is a hardcoded literal, and its only network call is the `/api/auth/me` probe that bounces non-admins to `/app`.

Auditing the surrounding code showed the demo had not filled a gap. It had **displaced a working system**. Three analytics surfaces coexist on `main`:

| Surface | State |
|---|---|
| `/admin-analytics` (`dashboard/frontend/admin-analytics.html`) | 100 % static mock. Sixteen sequential inline `<script>` blocks, several of which overwrite earlier ones (§8.1). |
| The in-app panel in `app.html`, driven by `js/admin-analytics.js` and `js/admin-analytics-value.js` (about 2,400 lines) | **Fully wired and working** over nine live endpoints. Its scripts are still loaded on every `/app` visit. |
| `admin-analytics-legacy-overview` in `app.html` | Carries the `hidden` attribute. No code anywhere clears it. |

One line in `js/admin-tabs.js`, added by #467, calls `window.location.replace('/admin-analytics')` when an admin enters the admin view on the default tab. The comment beside it states the intent: leave the in-app panel in place but never show it. The working dashboard therefore has no route to it.

Two consequences follow that are invisible from the page that caused them:

- **The phantom fetch.** `window.location.replace()` *schedules* a navigation; it does not stop the current synchronous tick. `app.js` calls `AdminTabs.onEnter()`, `AdminAnalytics.onEnter()` and `AdminAnalyticsValue.onEnter()` as independent `if` blocks (after it has already called `loadAdminStats()` and `loadAdminUsers()`), so the later calls still run; `AdminAnalyticsValue.onEnter()` finds no tab in the URL, defaults to `analytics`, and fires `refreshPrimary()` → `Promise.allSettled([fetchLifecycle(), fetchPriorityUsers(), fetchGroups()])`. **Three real analytics GETs, plus the console's stats and user-list loads, fire per admin visit** into a page that is already navigating away.
- **Two thousand lines of the only working reference for the endpoint contracts are one tidy-up away from deletion**, because nothing on screen uses them.

Fixing the page first would have wired a better face onto that. FlyM1ss made the call, against his own opening request, that **the architecture is corrected before any UI work**. This document is the result.

### 2.2 Both outage burners are still live

On 2026-09-11 every route on the users/content Postgres project, including login, returned 500 for about five hours. The Neon free-tier egress quota had been exhausted by analytics maintenance. The 2026-09-12 design correctly diagnosed two burners and prescribed a stopgap. **The stopgap never landed**: PR #455 merged the documents only. On `main` today:

- `domain/analytics/states.py` still selects a user's snapshot as stale after **15 minutes**, and the reaper sweeps every 60 seconds.
- `domain/analytics/service.py` still constructs the service with `project_snapshots=True`, so every accepted lifecycle event recomputes the caller's snapshot **synchronously inside the request**, including one `protocol_runs` query per agent the user owns (an N+1 against the local run store, at ten agents per user). Flipping that flag alone is not enough: `instrumentation.py::_emit`, the only production caller, has its own fallback that recomputes the snapshot whenever the service does **not** project. Both halves are switched off together.

Prod survives this only because it moved to Neon Launch, which bills compute hours instead of capping egress. The failure changed shape from an outage into a monthly bill, which is exactly why it was survivable enough to sit for five days. PR 0 (§13) kills both burners before anything else lands.

### 2.3 Scale target

Hundreds to a few thousand users; thousands to **tens of thousands of agents**, roughly ten agents per user. The platform is agent-dominated, and any table or job keyed by agent must be sized for that ratio. It is explicitly **not** hyperscale: no sharding, no partitioning, no external queue, no second process. Everything in §6 runs inside the single Render web process against two Neon projects.

---

## 3. Goals and non-goals

**Goals**

1. Remove the per-user recompute mechanism entirely, so the cost of analytics is bounded by construction and pinned by tests (carried from 09-12).
2. Put one admin page at `/admin`, built on the repo's testable module convention, that renders only numbers the backend can serve with a named source, freshness tier and read cost.
3. Make the admin layer look like the rest of the repository: stores with twins, domains that read each other through services, routers on FastAPI's declarative surface, periodic work on a thread that owns it.
4. Retire the document debt: one design, one set of plans.

**Non-goals** (unchanged from 09-12 unless noted)

- The outreach sender, templates or opt-out surface; only the transition log it will consume.
- The user-facing My Usage page; only the field-visibility rule.
- A subscription system; only the tier resolver seam.
- A `viewer` role.
- Durable storage for protocol runs.
- Changing the event allowlist, the 1 KB property cap, or 180-day raw retention (those live in the 08-26 document).
- Sub-day cross-user **analytics** aggregates. Live operational counters read from process state are not analytics aggregates (§5, D16).
- Porting the old console's Users (account management and grant pool), Providers and Activity tabs into `/admin` **in these PRs**. It is the committed end state (D5) and a named follow-up, not part of PR C.
- Hyperscale (§2.3).

---

## 4. What prod does today

Verified at source against `main` @ `c3bbf2ed` on 2026-09-15. Line numbers move; the file and function names do not.

### 4.1 The three surfaces

| Surface | Files | Reads | State |
|---|---|---|---|
| Static preview page, `/admin-analytics` | `dashboard/frontend/admin-analytics.html` (727 lines, 16 inline `<script>` blocks) | `GET /api/auth/me` only | Every number is a literal. Executed under jsdom, the final DOM differs from the markup in five places (§8.1). Served by `app.py::serve_admin_analytics` on Render and by Vercel `cleanUrls`; listed in `middleware.py::EXEMPT_PATHS`. |
| In-app Analytics panel (value overview) | `app.html` `#adminAnalyticsValueOverview`; `js/admin-analytics-value.js` (1,210 lines) | `/lifecycle`, `/retention`, `/commercial`, `/operational`, `/users`, `/groups` | Working. Unreachable through the UI: `AdminTabs.setTab` redirects on admin intent. |
| In-app Analytics panel (legacy overview + profile) | `app.html` `.admin-analytics-legacy-overview` (`hidden`, nothing clears it) and `#adminAnalyticsProfile`; `js/admin-analytics.js` (1,209 lines) | `/overview`, `/users`, `/users/{id}`, `/users/{id}/activity` | The overview half is dead markup. The **profile** half is live and is opened from the value overview's evidence dialog. |

The rest of the old console is untouched and live: **Users** (`js/admin-credits.js`: grant pool, account management, `PATCH /api/admin/users/{id}` including `user_group`), **Providers** (`js/admin-model-providers.js`), **Activity** (`js/admin-credits.js`: grant audit trail). `admin-tabs.js` owns the four-tab rail and the redirect.

### 4.2 The redirect and the phantom fetches

`js/admin-tabs.js::setTab` calls `window.location.replace('/admin-analytics')` when the resolved tab is `analytics` and `leave` is true. `leave` is true on admin intent only (entering the admin view via `onEnter`, or clicking the tab); PR #468 made page-load and popstate pass `leave: false` after #467 had bounced every `/app` view. `test_admin_tabs_redirect.py` pins those four paths under `node -e` with its own DOM stub.

`app.js::navigateToPage`'s `page === 'admin'` branch then runs, in order: `loadAdminStats()`, `loadAdminUsers()`, `AdminTabs.onEnter()` (which schedules the redirect), `AdminAnalytics.onEnter()`, `AdminAnalyticsValue.onEnter()`, `AdminModelProviders.onEnter()`, `AdminCredits.onEnter()`. Each is an independent `if (window.X)` block; none checks whether the previous one navigated away. `AdminAnalyticsValue.onEnter()` reads no `adminTab` from the URL, defaults to `analytics`, and calls `refreshPrimary()` → `Promise.allSettled([fetchLifecycle(), fetchPriorityUsers(), fetchGroups()])`. **Per admin visit, three analytics GETs plus the console's own stats and user-list loads are issued into a page that is already navigating away.** (`AdminAnalytics.onEnter()` only fetches when a profile deep link is in the URL.)

### 4.3 The burners

- `domain/analytics/states.py::repair_stale_snapshots(stale_after=timedelta(minutes=15), limit=100)` and its sibling `repair_stale_value_snapshots`. Both are called by `maintenance.py::run_analytics_maintenance` with no override, which `app.py` registers with `register_reaper_sweep`. The reaper runs every `RUN_REAPER_INTERVAL_SECONDS` (60). The first re-reads each stale user's full 180-day event history through `calculate_user_state`; the second does the same through `recalculate_user_snapshots` → `_calculate_user_value_snapshot`.
- `domain/analytics/service.py::_build_analytics_service` constructs the singleton with `project_snapshots=True`. `record_server_event` then calls `recalculate_user_snapshots` synchronously, in the request, for every accepted lifecycle event (one backtest emits four). The recompute includes `value_repository.py::_run_health`, which lists the user's agents from the content database and then queries `protocol_runs` **once per agent** through `domain/runs/repository.py::run_store` (local SQLite on `DATABASE_PATH`, not the runs Neon project as the handoff assumed). `instrumentation.py::_emit` carries a fallback that performs the same recompute when `project_snapshots` is false, so both must be disarmed.

### 4.4 The nine endpoints and their sources

All under `/api/admin/analytics`, router-level `Depends(require_admin)`, all plain `def` handlers. Response models live in `query_service.py` (legacy) and `value_queries.py` (value axis), not in `models.py`.

| Route | Response model | Reads today | Bounded? |
|---|---|---|---|
| `GET /overview` | `AnalyticsOverview` | raw `analytics_events` for the filter window (no width cap), rollups for past days, `user_analytics_snapshots` full scan for `user_state_counts`, `users` full scan | window only |
| `GET /lifecycle` | `LifecycleAnalyticsResponse` | `user_analytics_snapshots` (chunked by id), `user_lifecycle_daily_snapshots`, rollups fallback, credit ledger for `paid_users` | 180-day cap, 365-day history scan |
| `GET /retention` | `RetentionAnalyticsResponse` | snapshots for cohort membership, **raw `analytics_events` scan** cohort-start → end+35d, credit activity, daily snapshots | window only |
| `GET /commercial` | `CommercialAnalyticsResponse` | credit ledger (lifetime + window), balances; `platform_model_cost` borrowed from the legacy overview | mostly |
| `GET /operational` | `OperationalAnalyticsResponse` | snapshots for state counts; everything else delegates to the legacy overview scan | as overview |
| `GET /groups` | `GroupAnalyticsResponse` | `users` full scan grouped by `user_group`, raw event scan for runs and cost, ledger for `paid_users` | window only |
| `GET /users` | `PaginatedValueUsers` | `users` full scan, snapshots, ledger (fixed 30-day window); **filters, sort and pagination run in Python over the whole population** | no |
| `GET /users/{id}` | `ValueUserProfile` | one snapshot row, ledger for one user, raw events for one user in the window, daily snapshots for one user | per user, ≤180 days |
| `GET /users/{id}/activity` | `AnalyticsActivityPage` | per-user keyset-paginated events for `timeline`/`runs`/`usage`; **`sessions` reads the user's entire experience-event history** | per page, except sessions |

Also: `POST /api/analytics/events` (ingestion, any signed-in user, 8 KiB body cap, 120/300 s per user) and `GET /api/admin/stats` (`users`, `admins`, `agents`, `active_dashboard_backtests`, `credits_metering_enabled`, `default_credits`; the fourth from the in-process slot ledger).

Dead code on the same router: `_user_filters()`, `AnalyticsUserFilters`, `PaginatedUsers`, `AnalyticsQueryService.list_users` — a complete parallel users-list stack no route reaches. The hand-rolled `_query_values()` helper is the only hand-parsed query string in the repository. `_raise_service_error` maps any unrecognised exception to `HTTPException(503) from None`; the file contains no `print` or logger, so a bad SQL statement and an exhausted pool are indistinguishable in prod logs. `admin_analytics` is absent from `BLOCKING_IO_ROUTER_MODULES` (`tests/test_event_loop_threadpool.py`).

### 4.5 Tables

| Table | Key | Swept? | Note |
|---|---|---|---|
| `analytics_events` | `sequence`; `event_id` unique | 180 days | raw log; allowlist in `models.py` |
| `analytics_daily_rollups` | `(rollup_date, metric_name, event_name, billing_mode, provider_id, model_id, outcome, error_category, user_state)` | **never** | anonymous; grows one row per distinct key per day |
| `user_analytics_snapshots` | `user_id` | never | legacy `status` (5-state, CHECK-constrained) **and** the value columns (`lifecycle_*`, `operational_*`, `activated_at`, `last_meaningful_activity_at`, `active_days_30d`, `successful_backtests_30d`) on one row |
| `user_lifecycle_daily_snapshots` | `(snapshot_date, user_id)` | 180 days | the per-user daily history the movement charts read |
| `analytics_projection_jobs` | `job_name` | never | **already exists**; `cursor`, `status`, window — the day claim extends it |
| `analytics_subject_settings` | `user_id` | never | exclusion flag |
| `admin_analytics_access_log` | `sequence` | 180 days | who viewed whom |

`ValueAnalyticsStore` (`value_repository.py`, 14 public methods) owns no table of its own and has 21 `is_postgres` branches and no `*_postgres.py` twin. `tests/test_store_twin_parity.py::_TWINS` registers 11 distinct pairs (the `AnalyticsStore` pair is listed twice) and discovers twins by `rglob("*_postgres.py")`, so a store with no twin file is never examined.

### 4.6 Guards that exist, and what they do not cover

- `test_app_composition.py` freezes `admin_analytics` routes as `(method, path, name)` triples and the whole app as `(method, path)` pairs, including `("GET", "/admin-analytics")`. `test_router_move.py` does not cover this router at all. Neither asserts a response shape.
- `test_admin_analytics_frontend.py` and `test_admin_analytics_value_frontend.py` are static text assertions against the JS source and `app.html`; they execute no JavaScript. `test_app_lifecycle_and_cache_versions_are_wired` pins the exact `?v=` of `app.js` (131), `styles.css` (140), `admin-analytics.js` (6), `admin-analytics-value.js` (5) and `admin-tabs.js` (7).
- No test exercises `middleware.py::is_exempt('/admin-analytics')`; `test_vercel_cache_headers.py` does not cover the two `/admin-analytics` header rules. Breaking either breaks the live page with a green suite.
- `api/auth.py` has no mapping for `psycopg.OperationalError`, `sqlite3.OperationalError` or `PoolTimeout` anywhere: `get_current_user`, `login`, `signup` and `me` let them propagate as bare 500s. That is how the September outage presented.

---

## 5. Decisions

Each decision names what it rules out and why, because a decision without its reasoning reads as an oversight to the next person and gets reverted. Numbering is for cross-reference only.

### Framing

**D1 — Architecture first, UI second.** This document pins the architecture of both the backend and the frontend. The new HTML that exists today is a UI *proposal*; it does not constrain the data model or the module structure. *Why:* the demo displaced a working system (§2.1); a better face on the same wiring would have preserved every defect.

**D2 — The survival rule.** A UI element survives only if this document can name its **data source, freshness tier, and read cost**. An element that fails is recorded in §8 as **cut, with the reason**. *Why:* the CEO saw the mock, and someone will ask where a panel went. A cut without a recorded reason reads as an oversight and gets rebuilt.

### Surface

**D3 — `/admin-analytics` replaces the old admin console entirely.** It is what an admin lands on from Profile → Admin. *Why:* two admin surfaces with different shells is the state that produced the redirect in the first place.

**D4 — The route is renamed to `/admin`.** *Why:* the page stops being an analytics page and becomes the admin console; a name that says otherwise invites a second console.

**D5 — Absorption is staged, and full absorption is the committed end state.** PR C absorbs **Analytics**, which includes the users list and the per-user profile (`/users` and `/users/{id}` are analytics routes). The old console's **Users** tab (account management fed by `GET /api/admin/credits/users` and `PATCH /api/admin/users/{id}`, plus the grant pool), **Providers** and **Activity** stay where they are and port in a later PR; `/admin` links to them. *Why:* those three tabs have their own routers, stores, pagination and tests, none of them analytics, and would double PR C's size for no analytics benefit. The end state is stated here so that "later" does not silently become "never".

### Frontend architecture

**D6 — Standalone page with external JS modules on the IIFE + `?v=N` convention. No inline scripts.** *Why:* the dashboard has no build step, so the only way any test in this repository can execute a line of frontend JavaScript is `tests/_frontend_source.py` lifting a named function out of an external file and running it under `node -e`. Two dozen pytest modules do exactly that. Inline scripts are invisible to that entire layer; today **no test in the repository can execute a single line of the shipped admin page**.

**D7 — Keep the client-side shell gate, and say why it is not an oversight.** The page probes `/api/auth/me` and bounces non-admins to `/app`. *Why:* Vercel serves `dashboard/frontend` as static files. A server-side gate for the HTML would exist only on the Render copy, and the two deployments would diverge. The **real** gate is `require_admin` on every `/api/admin/*` route: a non-admin who defeats the shell gate sees an empty page and 403s. The shell gate is a courtesy redirect, and the document says so because it looks like a security omission to a reader who does not know about the Vercel deployment.

**D8 — Harvest the existing wired JavaScript per need; delete it at the end of the UI PR.** *Why:* `admin-analytics.js` and `admin-analytics-value.js` are the only working reference for the endpoint contracts, and they already solve stale-response guarding (`requestSeq`), URL-backed filter state, lazy disclosures, the evidence dialog, and per-section availability. They are not kept as modules because their DOM targets are the in-app panel this design retires; they are mined, then removed in the same PR so that the redirect, the hidden legacy block and the phantom fetch cannot survive it.

### Axes

**D9 — `user_group` replaces `cohort` entirely, and `cohort` is struck explicitly.** The 09-12 design's `cohort` (a free-text slug on `users`, validated `^[a-z0-9_-]{1,32}$`, with the rule that `paid` and `admin` are never cohort values; in the plan a `users.cohort` column on both twins, `normalize_cohort`, a text input, a `<datalist>` of existing values and a Users-tab column in task C6) was never implemented. PR #465 shipped `user_group` with six fixed values and an edit route. *Why:* two categorical labels on one user is two owners of one fact. The strike is explicit, naming every artefact of the old plan, because an axis that is merely omitted gets built by the next reader who finds it there. The word "cohort" survives in one legitimate sense: `RetentionCohort.cohort_week`, the activation-week bucket of the retention grid.

**D10 — Activation week is the intake axis.** "September intake" is `RetentionCohort.cohort_week`, computed from behaviour, and needs no stored label. *Why:* the only remaining argument for a free-text cohort was intake tracking; the retention grid already answers it.

**D11 — The group badge is server-computed.** Nothing computes it today: `user_group` is a plain column and no precedence logic exists anywhere in the backend or frontend. PR B adds `resolve_group_badge` (the 09-12 plan's name) as a unit-tested pure function; PR D puts `group_badge` on every user payload in the contract re-cut; PR C renders the string it is given (`—` while the field is absent). The 09-12 plan's instruction (task C6) is kept verbatim: *"Do not recompute the precedence in JavaScript: the server owns that rule, and two owners of one rule is how the season-number badge and its banner came to disagree."* (The precedent is the public leaderboard's season badge, whose number and banner were once computed in two places; `CLAUDE.md`, "Two leaderboards".)

### Consumer priority

**D12 — The overview answers questions in this order, top to bottom:**

1. **Platform scale** — total users, total agents, backtests running now. Already served by `GET /api/admin/stats`. Zero new backend work.
2. **Activity and behaviour** — what users have done and are doing.
3. **Activation and funnel** — where the product leaks.
4. **Cost control** — who burns operator-funded spend.
5. **Group comparison** — how the six sources differ.
6. **Retention outreach** (future) — the transition log only.

The 09-12 document ranked cost control **first**. *Why it moved:* that ranking was written the day after an outage caused by analytics cost, when cost was the live question. For the audience this page serves (the supervisor, and a CEO who has seen it once), the first question is "how big is this and is it alive", then "what are people doing". Cost control is a real consumer and stays on the page; it is fourth because the entitlement lever it feeds is pulled monthly, not daily. Recorded so the next reader does not restore it as a correction.

### Metrics

**D13 — `top_failure_categories` is kept and served from `analytics_daily_rollups` on the `error_category` dimension.** *Why:* the 09-12 rewrite silently deleted it (zero mentions across its 4,930 lines of spec and plan), while both frontends render it. The rollups already key on `error_category`, so keeping it costs no schema change and no raw-event read.

**D14 — Billing-lane mix (BYOK vs platform credits) is kept, from the rollups' `billing_mode` dimension.** Same reasoning as D13.

**D15 — Top product page, country and device are dropped from display, and named here.** *Why:* none has a home in the daily facts, and the 09-12 plan punted with "list whatever you drop in the PR body". The *collection* of those event properties is a separate decision and a named follow-up (§14), because the 08-26 document owns the event allowlist and changing it is a non-goal here. Dropping the display without deciding the collection would be a silence; this is a decision.

### Live operations

**D16 — Live operations stay on the page, served from process state via their own route, never from `/api/admin/analytics/*`.** *Why:* the 09-12 non-goal forbids sub-day **analytics** aggregates. Backtests-running-now is not an analytics aggregate; it is a counter in the in-process slot ledger, and `GET /api/admin/stats` already returns it. Reading it costs nothing and touches no analytics table.

**D17 — General rule: a live number comes from the component that owns the state, never from a projection of it.** *Why:* this is the generalised lesson of the outage, which was caused by materialising labels from a copy of the facts. `/api/admin/stats` independently arrived at the same rule.

Two caveats the page must carry: the slot ledger is **per-process**, so a second replica under-reports (prod runs one instance today); and the mock's "12 / 24 slots" is fiction — the real ceiling is `MAX_ACTIVE_DASHBOARD_BACKTESTS`, default **5**.

### API contracts

**D18 — The nine endpoint shapes are held byte-identical through PR C, through PR A, and through PR B except for D20.** PR C adds no field: it renders §9's fields when a response carries them and "Awaiting data source" when it does not, so the page's node tests join the conformance oracle rather than replacing it. *Why:* this is a **testing** argument, not an API-design one. Admin analytics has **zero external consumers**: not the PyPI SDK (hand-written stdlib on `/api/v1`), not the Discord bot, not the landing site. There is no compatibility obligation, so cleanliness is free to buy at any time. What holding the shapes buys is that the existing wired frontend and its tests become a free conformance oracle while the data model underneath is replaced: a moved number has exactly one suspect.

**D19 — The contract is re-cut once, in PR D, after PR B and after the new frontend has shipped.** (Amended 2026-09-16; it read "in PR C, alongside the new frontend".) *Why:* the two arguments still hold — the testing argument expires the moment the new page exists, and re-cutting earlier would mean two frontends tracking one moving contract. What changed is the advisor's request to settle *what to show and how* on a live page before the backend is locked to it. So the page ships first, reading today's contract and rendering the target shape from fixture copies (`tests/fixtures/admin_analytics/target/`), and the one re-cut lands after B has re-sourced every read — shaped by the page as reviewed, not by a design guess. The cost is one interval in which §9's slots read "Awaiting data source": visible and named, which is why absent-versus-empty is a rule the page carries (PR C plan, "Interim contract") rather than a blank chart. Note that the route-contract guards (`test_app_composition.py`: `(method, path, name)` triples per router and `(method, path)` pairs app-wide) protect the route **set**, not response shapes, which is how a whole dead users-list stack survived under them.

**D20 — The one permitted break before PR D: the five-state vocabulary leaves the contract in PR B.** That is `user_state_counts` on `/overview` and the `status` filter on `/users` — one vocabulary, two appearances. *Why:* it ships the legacy states alongside the lifecycle segments that replaced them. PR B deletes the five-state model (it is the PR that moves every read path off the snapshot tables; deleting the model in PR A, before the reads move, would leave prod reading dropped tables — see §13). A shim faking a dead vocabulary from a live one is worse than the break, because it makes the field look maintained. The only sender of `status` is the legacy attention queue in `admin-analytics.js`, dead markup since #467; its query and the test that pins the parameter name go in the same PR. `users_needing_attention` on `/overview` keeps its shape, re-sourced from operational state (`blocked` / `needs_attention` are the same words in both models).

### Delivery

**D21 — PR letters are renumbered: PR A is the rewrite.** The 09-12 document's throwaway "PR A" stopgap is reduced to the two value changes, one guard and one error mapping in PR 0 (§13). *Why:* everything else in the old PR A (a shared stale constant, a one-read recompute, a spy-store budget test for the sweep) is scaffolding for a code path PR A deletes.

**D22 — The `ValueAnalyticsStore` Postgres twin is extracted before PR A, as its own PR, together with an absence-direction check in `test_store_twin_parity.py`.** *Why:* the store has 21 dialect branches and no twin, and the parity guard enumerates existing `*_postgres.py` files, so a store with none is **invisible** to it, not exempt. The 09-12 plan tells its implementer the guard will catch a one-sided method, which is true of every store except the one receiving twelve new methods. Extracting now is a day; after PR A it is several. The absence check (grep for `is_postgres` / `hasattr(..., "database_url")` outside a registered twin, assert zero) is what stops this recurring.

**D23 — Inside PR A: the daily job moves to its own worker thread; ledger and run aggregates go on the owning stores; `backfill.py` is fixed in the same pass.** *Why:* the run-heartbeat thread exists to keep `heartbeat_at` fresh against a 300-second staleness rule; a whole-population, eight-step, three-database batch on that thread starves the heartbeat and is only bounded by a test that counts queries, not wall-clock. The two hand-rolled cross-domain readers in `value_repository.py` and the original one in `backfill.py` are the same idiom; fixing two of three regrows the third.

**D24 — Hygiene is folded into PR A, not shipped as a separate PR.** Items: give `_raise_service_error` a `print` and drop `from None`; register `admin_analytics` in `BLOCKING_IO_ROUTER_MODULES`; delete the dead `_user_filters` / `AnalyticsUserFilters` / `PaginatedUsers` / `list_users` path. *Why:* this repository has no required checks and merges opportunistically. A hygiene-only PR is the one that sits until it conflicts.

---

## 6. Target architecture — backend

Carried forward from the 2026-09-12 design. Paragraphs marked **Amended** differ from that document; everything else is its text, lightly re-punctuated. The 09-12 document is deleted by this PR, so this section is now the only copy.

### 6.1 Principle

Per-user facts are derived from the authoritative tables that already hold them. The event log is read only in bounded one-day or paginated windows. Lifecycle labels are computed at read time from stored timestamps. All cross-user aggregates come from one daily fact table written once per UTC day. **No periodic per-user recompute exists anywhere.** The mechanism that caused the outage was labels materialised from a copy of the facts; this architecture has no such copy.

### 6.2 Three axes

| Axis | Values | Source | Writable by |
|---|---|---|---|
| role | `user`, `admin` | `users.role` | admin (existing) |
| tier | `unpaid`, `starter`, `invested`, `high_value` | `commercial_tier()` over the credit ledger | nobody; derived |
| **user_group** (**Amended**, was `cohort`) | `internal`, `invited`, `organic`, `competition`, `partner`, `unknown` (default) | `users.user_group`, shipped in PR #465 | admin, via the existing groups edit route |

Rules:

- A user has exactly one `user_group` (**Amended**: the cohort was nullable, "at most one"; `user_group` is `NOT NULL DEFAULT 'unknown'`). Groups do not overlap with role or tier; "paid" and "admin" are never group values.
- The tier resolver is a pure function in the credits domain. When a subscription system exists it changes the resolver's input, not analytics.
- **Display precedence** for the single "group" badge (owned by the 09-13 document, restated): `admin` if the role is admin, else the `user_group` when it is not `unknown`, else `paid` when the tier is anything but `unpaid`, else `free`. The server computes it; the client renders the string it is given (D11).
- Every analytics filter and chart accepts any combination of the three axes.
- **Amended:** there is no free-text cohort, no cohort validator, no cohort suggestion list. Intake is the activation week (D10).

### 6.3 Freshness contract

Three tiers.

**Live** (read at request time, always scoped to one user or one ledger):

- Credit balance and entitlements. Read from the ledger, not analytics.
- One user's event timeline, paginated.
- One user's current lifecycle segment, computed from stored timestamps, the trailing-30-day sums from the daily fact table, and today's date.
- One user's operational state (`blocked`, `needs_attention`, `healthy`), a fixed handful of small queries.

**Daily** (from `user_daily_facts` and `analytics_daily_rollups`, complete through the previous UTC day):

- Every cross-user number: overview counts, activation funnel, daily active users, segment distribution and movement, retention grid, cost trends, group and tier comparisons, priority list, failure categories, billing-lane mix.
- Per-user trailing windows on the profile: runs and spend over the past 7 and 30 days, labelled "as of yesterday".

The overview may additionally read the **current** UTC day from raw events as one bounded one-day scan, so "today" is not blank. That is the only raw-event read outside the paginated timeline. No per-user "today" numbers are shown.

**Live operations** (**Amended**, new tier): the counters `GET /api/admin/stats` already serves — backtests running now and the slot ceiling from process memory, total users and total agents from one `COUNT` each on the owning store. Never from an analytics table and never from a projection (D16, D17). Per-process scope for the memory counters.

Consequence to accept: a user's third successful backtest today promotes them to Core tomorrow. Segments are day-granular by definition, so this is consistent rather than late.

### 6.4 Sources of truth

| Fact | Authoritative table | Database |
|---|---|---|
| account, role, `user_group` (**Amended**, was `cohort`), created_at | `users` | users |
| sessions, last seen | `auth_sessions` | users |
| dashboard runs, model cost, tokens | `agent_runs` (+ new `owner_user_id`) | runs |
| purchases, refunds, consumption | `credit_ledger_entries`, `credit_llm_usage_entries` | users |
| discrete product actions, page views | `analytics_events` | users |
| protocol and v2 run outcomes | `analytics_events` (`backtest_*`) until protocol runs are durable | users |
| backtests running now, active slots (**Amended**, new row) | in-process slot ledger (`count_active_dashboard_backtests()`) | none (process memory) |

Analytics tables stay in the users database. Every read path joins `users` for role and exclusion, the data is small at the §2.3 scale, and the 2026-09-11 blast radius was an unbounded loop, not co-location. The auth surface maps a user-store `OperationalError` (including `PoolTimeout`) to 503 with an `ERROR: auth.user_store_unavailable` log line so the next outage is visible instead of a bare 500 (PR 0).

**`agent_runs.owner_user_id`.** Nullable, written at run creation on both twins from the authenticated caller. Not backfilled: rows before the column stay unattributed. It is the source for per-user model cost (`est_cost_usd`, `input_tokens`, `output_tokens`). Run *counts* come from events so the three run surfaces share one column set. The plan confirmed this column is genuinely required: `model_usage_recorded` carries `cost_micro_usd` only on the historical backfill path; the live path emits `credits_reserved` / `credits_settled` / `credits_refunded`, which measure the user's own credit spend, not operator-funded cost. Operator cost lives in `agent_runs.est_cost_usd` and nowhere else.

### 6.5 `user_activity` (replaces `user_analytics_snapshots`)

One row per user, maintained by event ingestion itself. No recompute path exists.

```text
user_id                       PK, FK users.id
activated_at                  first accepted backtest_completed; set once
last_meaningful_activity_at   GREATEST(existing, event.occurred_at) for every
                              accepted event in the lifecycle activity set
updated_at
```

Credit activity already reaches this row through the `model_usage_recorded` and `credits_*` events the credits service emits, which are in the meaningful-activity set. The daily job's ledger step corrects the timestamps from the ledger itself, so a dropped event cannot leave a paying user looking inactive. The meaningful-activity set is `_LIFECYCLE_ACTIVITY_EVENTS` in `domain/analytics/lifecycle.py`, unchanged.

The legacy 08-26 five-state columns (`status`, `reason_code`, `human_readable_reason`, `evidence_event_ids_json`), their calculator (`calculate_user_state`), and their repair pass are deleted. The 09-03 value columns (`lifecycle_*`, `operational_*`, `active_days_30d`, `successful_backtests_30d`, `calculated_at`) are dropped: segment and operational state are computed at read time or read from the daily table, and the 30-day counts are sums over it.

**This table is never swept** (§11). It is current state, not history.

### 6.6 Lifecycle at read time

`calculate_lifecycle` keeps its rules and reason codes (§15) but takes stored inputs instead of an event list:

```text
calculate_lifecycle(
    created_at, activated_at, last_meaningful_activity_at,
    active_days_30d, successful_backtests_30d, today
) -> LifecycleResult
```

`active_days_30d` and `successful_backtests_30d` are `SUM` over the user's `user_daily_facts` rows for the 30 UTC dates ending yesterday. Evidence in the explanation cites the timestamps and sums, not event IDs.

### 6.7 `user_daily_facts` (replaces `user_lifecycle_daily_snapshots`)

At most one row per user per UTC day, written for the previous day.

```text
snapshot_date         PK part
user_id               PK part, FK users.id
lifecycle_segment     as of the end of snapshot_date
lifecycle_reason_code
operational_state
operational_reason_code   (Amended: new; the reason behind a blocked / needs_attention
                           state, so "why are users blocked" is a grouped read, not a
                           per-user one)
tier                  commercial_tier() as of the end of snapshot_date
user_group            users.user_group as of the write   (Amended: was `cohort`, nullable)
active                1 when the user had any accepted event that day
runs_requested
runs_completed
runs_failed
runs_cancelled
operator_cost_micro   SUM(agent_runs.est_cost_usd) for operator-funded runs,
                      converted to micro-USD
own_spend_micro       SUM(credit_llm_usage_entries) consumed that day
data_quality          'complete' | 'partial'
calculated_at
```

Rules:

- Admins and users excluded via `analytics_subject_settings` get no rows, matching today's aggregate exclusion. Their profiles still work from live data.
- `partial` marks a row for which at least one source was unavailable, or a migrated row without run and cost columns. The UI keeps labelling such periods "Incomplete data" rather than zero.
- Retention is 180 days because the row carries a user id. Before rows expire the retention coordinator writes the anonymous long-term rollups (segment counts, transitions, and run and cost totals keyed by tier and `user_group`) into `analytics_daily_rollups`, which is retained indefinitely. **Amended — the encoding:** that table's key has no tier or group column, so the dimension name rides in `metric_name` and the dimension value in the `user_state` column, exactly as today's `lifecycle_segment_count` / `lifecycle_transition` rows already do; the new metric names are `runs_by_tier`, `runs_by_user_group`, `operator_cost_by_tier`, `operator_cost_by_user_group`, `own_spend_by_tier`, `own_spend_by_user_group`. No DDL change. This write is extended in PR B (the 09-12 plan's task C4); `rollup_day` stops writing the five-state `user_state_count` rows in the same PR.
- `runs_failed` is a bare count. **Failure *reasons* are not on this table and do not need to be**: the rollups carry `error_category` per day (D13).

### 6.8 `lifecycle_transitions` (outreach hook)

Appended by the daily job when a user's segment on `snapshot_date` differs from the previous stored day.

```text
transition_id      PK
user_id            FK users.id
snapshot_date
from_segment
to_segment
inactive_days      at the transition
data_quality       'complete' | 'partial', from the day it was derived from
created_at
```

`UNIQUE (user_id, snapshot_date)`, written with `ON CONFLICT … DO UPDATE`. The constraint is there to stop a duplicate when the daily job retries a day, not to make the first write final: the day a transition is rewritten is the day the first attempt was wrong, and `DO NOTHING` would leave this table permanently disagreeing with the `user_daily_facts` row the retry did correct.

`data_quality` carries the quality of the day the transition was derived from, so a consumer can tell a transition computed from a complete day from one computed from a day whose ledger or run source was unavailable. Without it, "activated user reached fourteen inactive days" can fire on evidence the job itself marked incomplete.

No consumer exists in this design. The future outreach rule "activated user reached fourteen inactive days" is one `SELECT` over this table plus an `outreach_log` of its own, and fires at most once per inactivity episode because each episode produces exactly one `growing → at_risk` row.

Retained 180 days, expired by the same retention sweep that expires `user_daily_facts`. This table carries a user id and is the easiest one in the design to forget, because nothing reads it yet.

### 6.9 The daily job

**Amended — scheduling.** One run per UTC day, driven by **its own worker thread** (`domain/analytics/daily_job.py`, started from `app.py` beside the reaper, with its own interval and its own stop event). It is *not* a step of the reaper tick. *Why:* the reaper thread's job is to keep run heartbeats fresh against `RUN_HEARTBEAT_STALE_SECONDS` (300 s); a whole-population batch across three databases on that thread would let a slow analytics night mark live runs as orphaned. The thread is still inside the single web process (§2.3): no external scheduler, no second service.

**Due check** (**Amended** — the claim and the cursor are two fields, so that "never twice" and "retry on failure" can both be true). Each tick of the worker reads the `analytics_daily_facts` row in `analytics_projection_jobs`, a table that already exists with `cursor`, `status ∈ {pending, running, complete}` and `updated_at`. If `cursor` already equals yesterday's date the tick does nothing else. Otherwise it **claims** the day with a compare-and-set that moves `status` from `pending`/`complete` to `running` and stamps `updated_at`; a second process whose compare-and-set finds `running` does nothing. The **cursor** advances to `D` and `status` returns to `complete` only after step 6 succeeds. A `running` claim whose `updated_at` is older than two hours is treated as abandoned and may be re-claimed, so a crash mid-job retries on the next tick rather than never. This replaces the process-local `_last_rollup_day` guard.

**Steps, for the previous UTC day `D`, each one set-based query over all users:**

1. Roll up raw events for `D` into `analytics_daily_rollups` (existing `rollup_day`; **Amended:** its call moves out of `run_analytics_maintenance` into this job in PR A so the two never double-write, and its five-state `user_state_count` rows stop in PR B).
2. One one-day event scan for `D`: per user, `active`, run outcome counts, and `activated_at` / `last_meaningful_activity_at` corrections if ingestion missed any (defensive, idempotent).
3. One `agent_runs` query grouped by `owner_user_id` for runs ending on `D`: operator cost. **Amended:** this is a public method on the run-history store (`BacktestDatabase` / `PostgresBacktestDatabase`), not SQL written inside the analytics package.
4. One ledger query grouped by user for `D`: own spend and the lifetime net purchase that feeds `commercial_tier()`. **Amended:** this is a public method on `CreditsStore` / `PostgresCreditsStore`, and `backfill.py`'s existing hand-rolled ledger read is moved onto the same method in the same task.
5. A fixed set of grouped queries for operational state (credits billing state, provider credential facts, the enabled-provider and platform-credential sets, agent ownership, and terminal runs in the trailing 24 hours), the same predicates the current `get_operational_facts` uses, written once for the population rather than five-plus queries per user in a loop. It is more than three queries and cannot be fewer: `protocol_runs` and `external_agents` live in different databases, so the run count needs two statements and a fold in application code. What matters is that the count is fixed, not that it is small. **Amended:** each query is a public method on the domain that owns the table.
6. Compute the fact row for `D` and upsert `user_daily_facts`, then write `lifecycle_transitions` where the segment changed. **Amended — what the row carries:** `lifecycle_segment` and `lifecycle_reason_code` from the read-time calculator; `operational_state` and `operational_reason_code` from `calculate_operational_state` over step 5's population facts; `tier` from step 4's lifetime net purchase; `user_group` read from `users` at the write; `active` and the four run counts from step 2; `operator_cost_micro` from step 3; `own_spend_micro` from step 4. Two properties of this step are load-bearing:
   - **Evidence is clamped to the end of `D`.** `user_activity` is one row per user overwritten in place; it describes now, not the end of an earlier day, and the segment rule refuses evidence newer than its `as_of`. A user who acted after midnight therefore aborts the step unless their activation and activity timestamps are clamped, with the fact table supplying the last active date on or before `D`.
   - **The 30-day window includes `D`.** The sums come from a table this step is about to write, so summing "the last 30 days of existing facts" silently means 29. `D`'s own contribution comes from step 2's result, in memory, so the stored segment equals what the read path computes the next morning.
7. Recompute any day in the last week whose events arrived after that day's facts were written, comparing `received_at` against the stored `calculated_at`. Events legitimately arrive late (the frontend route accepts `occurred_at` up to 24 hours old, and a run finishing at 23:59 is appended after midnight), and the day-claim cursor means nothing would ever revisit them otherwise. Both fact rows and transitions are upserts so a recompute corrects rather than duplicates.
8. Run the retention coordinator (existing) if due. **It never touches `user_activity`** (§11).

Every step is wrapped individually; a failed step marks that day's rows `partial` and logs `WARNING: analytics.daily_facts.<step>_failed category=<exception class>`, never a message body. A failed day is retried on the next tick because the cursor is only advanced past `D` after step 6 succeeds. Step 7 is a separate safety net and not a substitute: it covers the day that *succeeded* and whose evidence changed afterwards.

**Migration of existing history.** Eight weeks of `user_lifecycle_daily_snapshots` are copied into `user_daily_facts` with run, cost and tier columns NULL and `data_quality='partial'`, and `user_activity` is seeded from `user_analytics_snapshots` (§11). **Amended — timing:** the copy and the seed run in PR A, when the new tables appear; the old tables are dropped in PR B, after every read path has moved (§13). The seed is idempotent (`MIN` on activation, `MAX` on activity) and PR B re-runs it immediately before the drop, because the legacy snapshot row keeps being maintained between the two PRs. Movement charts keep their history.

### 6.10 Read paths

| Route | Today | After PR B |
|---|---|---|
| `GET /overview` | live scan of raw events for the filter window | rollups for completed days plus one one-day raw scan for today; `top_failure_categories` and billing-lane mix from rollups |
| `GET /lifecycle` | snapshot table plus daily lifecycle table | **Amended:** segment counts and movement from yesterday's `user_daily_facts` and `lifecycle_transitions`; headline `activated_users` from `user_activity`; read-time segments are for the one-user profile only (a read-time loop over every user is what §2.3 rules out) |
| `GET /retention` | live scan of raw events per activation week | `user_daily_facts.active` grouped by activation week |
| `GET /commercial` | live ledger aggregation | unchanged; ledger is the source |
| `GET /operational` | snapshot columns | yesterday's `operational_state` from `user_daily_facts` for lists; live for one profile |
| `GET /groups` (**Amended**, new row) | `users` grouped by `user_group`, raw event scan for runs and cost, ledger for `paid_users` | user counts from `users`; run counts and operator cost from `user_daily_facts` grouped by `user_group`; `paid_users` from the ledger |
| `GET /users` | snapshot listing | `user_activity` joined to `users` and yesterday's facts; filters on `role`, `commercial_tier`, `user_group`, `lifecycle_segment`, `operational_state`, with tier and operational state taken from yesterday's facts row; **Amended:** filtering, sorting and pagination happen in SQL, not in Python over the whole population, and the filters become declared FastAPI query parameters in PR D |
| `GET /users/{id}` | snapshot plus per-section queries | live segment and operational state; 7-day and 30-day sums from facts |
| `GET /users/{id}/activity` | paginated raw events; sessions section unbounded | timeline unchanged; sessions section gains a 30-day window |
| `GET /api/admin/stats` (**Amended**, new row) | process state + two counts | unchanged but for one added constant (§10.3); the live-ops source (D16) |

All routes keep the central admin dependency, the access log, and display-safe responses.

### 6.11 Event log discipline

These rules govern `analytics_events` from PR A onward and are enforced by a test in `tests/test_architecture_boundaries.py` that scans the analytics package source:

1. The allowlist stays closed and the 1 KB property cap stays.
2. Raw rows are retained 180 days; aggregates live in rollups.
3. No code path reads one user's full event history except the paginated timeline route.
4. Every other `list_events` call passes a window of at most one UTC day.
5. Cross-user aggregates read `analytics_daily_rollups` or `user_daily_facts`, never raw events, except the overview's current-day scan.
6. No maintenance step loops over users issuing per-user queries.
7. **Amended (new):** the analytics package never opens another domain's connection. `_get_connection()` is called only on `self`. Other domains' tables are read through public methods on the store that owns them.

### 6.12 Read budget

Pinned by a test that wraps **every store the job touches** — analytics, credits, run-history, users, agents, providers (**Amended**: a spy on the analytics store alone cannot see the reads §6.9 moved onto the owning domains) — in a counting spy and drives the daily job through 24 simulated hours, at 200 and again at 400 synthetic users:

| Path | Budget |
|---|---|
| worker tick when the day is not due | 1 query (`analytics_projection_jobs`) |
| daily job, per UTC day | a constant number of queries, independent of user count, none parameterised by a single user id |
| daily job, per UTC day (**Amended**, new) | wall-clock under a fixed bound at 200 users, and the 400-user run no more than 2.5× the 200-user run, so a constant-count step that scans an unindexed table or hides a quadratic fold is caught |
| admin overview page load | at most 1 raw-event query, windowed to the current UTC day |
| admin user profile load | at most 1 raw-event query (timeline page) |

The PR A plan fixes the daily-job constant from the final step list and the test pins that exact number. The test fails on the first user-id-parameterised query inside the daily job regardless of count, and it fails if doubling the synthetic user count changes the query count at all.

### 6.13 Field visibility (My Usage hook)

The per-user metric struct is built once by `AnalyticsQueryService.get_user_metrics(user_id)` and filtered by audience:

| Field | Admin | User (future) |
|---|---|---|
| runs by outcome, 7d and 30d | yes | yes |
| own spend, operator-funded cost | yes | yes |
| tier, credit balance | yes | yes |
| active days | yes | no |
| lifecycle segment, reasons, evidence | yes | no |
| operational state and evidence | yes | no |
| `user_group`, group badge | yes | no |
| access log | yes | no |

A future `GET /api/me/usage` is the same call under the session dependency with the user column of this table. It is not built here.

### 6.14 Ownership boundaries (new)

A convention audit — twenty-five independent review agents, each checking one aspect of the admin and analytics code against the conventions the rest of the repository follows, every finding then re-verified by a second agent trying to refute it — found five root causes behind thirteen misalignments. The architecture states the rule each one broke:

| Rule | Broken today by | Fixed in |
|---|---|---|
| A domain reads another domain's **service**, never its **tables**. `_get_connection()` is called only on `self`. | `value_repository.py` (two readers into the credit ledger), `backfill.py` (one) | PR A |
| Every store has a Postgres twin, and the parity guard can see a store with **no** twin. | `ValueAnalyticsStore` (21 dialect branches, no twin) | PR T |
| Periodic work owns its scheduler. | analytics maintenance on the run-heartbeat thread | PR A |
| Routers sit on FastAPI's declarative surface: query params are declared, blocking-IO routers are registered, unknown exceptions are logged before they become 503. | hand-rolled `_query_values()`, `_raise_service_error` with `from None` and no `print`, `admin_analytics` absent from `BLOCKING_IO_ROUTER_MODULES` | PR A (logging, registration, dead path), PR D (declared params) |
| One frontend surface per feature. | the static preview forked from the in-app panel, neither retired | PR C |

### 6.15 Statements of fact about the code being replaced

Recorded by the 09-12 document while its plan was being written; kept because they explain the shape of PR 0 and PR A.

1. The sixty-second sweep was one of two burners. `AnalyticsService.record_server_event` also recomputes the caller's snapshot synchronously, in the request, for every accepted lifecycle event, and each recompute reads that user's full 180-day history twice. One backtest emits four progress events, so a single run cost eight full-history scans before the sweep ran at all. This is the larger of the two. It is the same defect — a label materialised from a copy of the facts — and the same fix: ingestion maintains `user_activity`, and nothing recomputes. (**Amended**, PR labels per D21:) PR 0 switches it off, including the `instrumentation.py` fallback; PR B deletes it.
2. The overview's current-day read is not bounded today. `rollup_current_day` reads a 30-day trailing window and `rollup_day` a 31-day one, because the conversion and repeat-rate formulas need trailing context. "One bounded one-day scan" is a narrowing to implement in PR B, not a property to preserve. Context comes from the rollups, which already hold it; a metric that cannot be computed that way is shown as of yesterday rather than widening the scan.
3. `GET /users/{id}/activity`'s sessions section scans a user's entire experience-event history with no time window. PR B gives it a 30-day window. The timeline, runs and usage sections are unchanged.

---

## 7. Target architecture — frontend

### 7.1 One page, `/admin`

`dashboard/frontend/admin.html`, a standalone document served at **`/admin`** on both hosts:

- **Render:** `app.py` gains `@app.get("/admin")` returning `FileResponse(frontend_path / "admin.html")`, plus `@app.get("/admin.css")` beside the existing `/styles.css` route (every static file is an explicit route; there is no static mount). `/admin-analytics` becomes a 308 redirect to `/admin` for one release, then goes.
- **Vercel:** `cleanUrls` serves `admin.html` at `/admin`. The existing rewrite `/admin/:path*` → Render is **deleted**: its only purpose was `DELETE /admin/runs/{run_id}`, which no frontend code calls, and whether `:path*` also matches the bare `/admin` differs between path-to-regexp versions, so leaving it in place makes the page's reachability depend on a router version nobody controls. The bare `/admin` already receives `Cache-Control: no-store` from the catch-all header rule, which is the right policy for an admin shell. A `redirects` entry sends `/admin-analytics` to `/admin`. (**Amended** by PR C review F5 — what shipped differs on both counts, and this paragraph as written tells a later reader to undo the fix:) deleting the rewrite outright was the reachability rule written one notch too wide. It also unproxied `DELETE /admin/runs/{run_id}`, which `api/routers/admin.py` does serve, leaving a backend admin route with no path to it from the deployed frontend host — the same route the `admin` arm of the no-store header rule was written for. What ships is the narrower `/admin/runs/:path*`: a **required literal segment** after `/admin` means no path-to-regexp version can fold the bare `/admin` into a pattern that demands `/runs/` next, so the version-dependence this paragraph is about cannot return. Pinned by `test_no_rewrite_can_claim_the_admin_page`, which guards the property (nothing may claim the page) rather than the prefix. Separately, `/admin` and `/admin.html` ship `public, max-age=0, must-revalidate`, not the `no-store` claimed above: a later `headers` entry overrides the catch-all, deliberately, so the shell caches like every other shell. It carries no data (D7), so `no-store` would buy nothing. Pinned by `test_admin_shell_must_revalidate_like_every_other_shell` and `test_admin_overrides_follow_the_api_no_store_rule`.
- **Middleware:** `middleware.py::EXEMPT_PATHS` gains `/admin`; `/admin-analytics` stays exempt until its redirect route is removed (an exact-match exemption that is dropped early would 400 the redirect before it can fire). Without the exemption a page load returns a 400 JSON body demanding `X-Session-Id`. A test now pins both entries.

The HTML is public on both hosts and **carries no data**. Every number arrives from a `require_admin`-gated endpoint after the shell gate has run (D7).

### 7.2 Modules

Four IIFE modules, flat under `js/` (the `/js/{file_name}` route serves one path segment), each exposing one global, loaded with `defer` and a `?v=N` cache-buster pinned by a test:

| File | Global | Owns |
|---|---|---|
| `js/admin-shell.js` | `window.AdminShell` | the `/api/auth/me` gate; hash routing (`#overview`, `#sources`, `#retention`, `#credits`, `#lifecycle`, `#health`, `#users`, `#users/{id}`; there is no `#live` route — the live row has no detail page); range and filter state ↔ URL; the shared `request()` with a `requestSeq` counter per surface; the shared loading / empty / error / stale / "Incomplete data" helpers; the rules and evidence dialogs with return-focus |
| `js/admin-live.js` | `window.AdminLive` | the live-operations row, from `GET /api/admin/stats` only |
| `js/admin-overview.js` | `window.AdminOverview` | the nine overview panels and their detail routes, from the analytics endpoints |
| `js/admin-users.js` | `window.AdminUsers` | the users list and the profile (Overview / Timeline / Runs / Usage / Sessions, lazy, cursor-paged). Account management is **not** ported (D5): the profile's "Open account management" action deep-links to `/app?view=admin&adminTab=users` with the user pre-filled, the same hand-off `AdminTabs.openAccountManagement` performs today |
| `js/admin-providers.js` | `AdminProviders` | The approved provider registry and the platform credential. *(Amended 2026-09-17: added by the navigation consolidation; the page's only write surface, reaching the network through `AdminShell.write`.)* |

Ownership, panel by panel:

| Overview panel (§8.2) | Module | Opens into |
|---|---|---|
| Live row (users, agents, backtests running, slot ceiling) | `AdminLive` | — (no detail page) |
| Users needing attention | `AdminOverview` | `#health` (failure categories, top operational reasons) |
| Active users | `AdminOverview` | `#users` |
| Activation progress | `AdminOverview` | `#sources` (the six-group table with activation per group) |
| Where users come from | `AdminOverview` | `#sources` |
| Users coming back | `AdminOverview` | `#retention` |
| Users reaching value | `AdminOverview` | `#lifecycle` |
| User lifecycle | `AdminOverview` | `#lifecycle` |
| Credits usage | `AdminOverview` | `#credits` |
| Revenue | `AdminOverview` | `#credits` |
| — | `AdminUsers` | `#users`, `#users/{id}` |

`admin.css?v=N` holds the page's styles, externalised from the mock's inline `<style>`, including the ATL header and ticker mirror HaoXiang built. It does not import `styles.css`; the admin page must not inherit a 15,000-line stylesheet to get a header.

Charts stay as the mock draws them: CSS bars and generated SVG, no Chart.js. Every renderer is a **pure function from payload to markup** so it can be lifted with `fn_body` and executed under `node -e` against the committed fixtures in `tests/fixtures/admin_analytics/`, which already round-trip through the real Pydantic response models. That is the whole point of D6: the page's behaviour becomes testable by the same harness as the rest of the frontend.

### 7.3 The gate

`admin-shell.js` probes `GET /api/auth/me` with credentials; a non-2xx, a missing user, or a role other than `admin` sends the browser to `/app`. Until the probe resolves the page shows the shell with empty panels, never sample numbers. This is the courtesy redirect of D7, and the doc says so in a comment beside it.

### 7.4 Harvest map

What is mined from the wired modules before they are deleted, and where it lands:

| Pattern | Today | Destination |
|---|---|---|
| Stale-response guard: `const seq = ++state.requestSeq; … if (seq !== state.requestSeq) return;` per surface, bumped on auth loss | `admin-analytics.js::loadOverview`, `loadAttention`, `loadProfile`, `loadProfileSection`; `admin-analytics-value.js::refreshPrimary` | `AdminShell.request` + per-module counters |
| URL-backed state: `readUrlFilters` / `replaceAnalyticsUrl` / `pushProfileUrl` (`pushState` on profile open so Back closes it); `analyticsPanel` remembering open disclosures | `admin-analytics.js:119-221`; `admin-analytics-value.js:200-226` | `AdminShell` state |
| Lazy disclosures: fetch once on first open, cache until reset; "Load more" cursor paging | `admin-analytics.js::selectProfileSection`, `loadProfileSection`; `admin-analytics-value.js::ensureDisclosureLoaded` | `AdminUsers` (profile tabs), `AdminOverview` (detail routes) |
| Evidence dialog with return-focus map, Escape and backdrop close | `admin-analytics-value.js:135-148, 639-665` | `AdminShell` dialogs |
| Per-section availability (`availability[name].status`), row-level `data_quality === 'partial'` → "Incomplete data", stale-but-shown (`keepStaleData`) | `admin-analytics.js:244-246`; `admin-analytics-value.js::availabilityIncomplete`, `applySettledSection` | `AdminShell` helpers |
| Access lost (401/403) → refresh auth user and leave | `handleAccessLost` in both files | `AdminShell` |
| Rules dialog content (`LIFECYCLE_RULES`, `OPERATIONAL_RULES`) | `admin-analytics-value.js:54-66` | `AdminShell` rules dialog (§15 is the source of truth for the copy) |
| Exact endpoint query names and the read-only / `textContent`-only discipline pinned by `test_client_uses_exact_pr2_endpoints_and_query_names` and `test_analytics_is_read_only_and_uses_safe_dom_rendering` | `test_admin_analytics_frontend.py` | re-pinned against the new modules |

### 7.5 What PR C removes, and what the old console keeps

Removed: `admin-analytics.html`; `js/admin-analytics.js`; `js/admin-analytics-value.js`; the `#adminPanelAnalytics` section, the hidden legacy block, `#adminAnalyticsProfile`, and the two analytics dialogs in `app.html`; the `.admin-analytics-*` and `admin-value-*` rule families in `styles.css`; the redirect branch in `admin-tabs.js`; the `AdminAnalytics` / `AdminAnalyticsValue` calls in `app.js` (including the refresh button's `AdminAnalytics.refresh()`, which today reaches `AdminAnalyticsValue.refresh()` indirectly).

The old console (`/app?view=admin`) keeps **Users** (account management and the grant pool), **Providers** and **Activity** (`admin-credits.js`, `admin-model-providers.js`, and `app.js`'s `#adminStats` strip). *(Amended 2026-09-17: Providers ported to `/admin` in the navigation consolidation; `?adminTab=providers` now redirects to `/admin#providers`. **Users** is relabelled **Account Management** per N6 of the 09-17 design — label only: `data-admin-tab="users"`, the panel id and the `?adminTab=users` value are unchanged, so no URL breaks.)*
Its rail loses only the Analytics tab; `DEFAULT_TAB` becomes `users`; Profile → Admin navigates to `/admin`. The new page's aside links to `/app?view=admin&adminTab=users`, `…=providers` and `…=activity` until the follow-up port (D5). *(Amended 2026-09-17: `…=providers` removed — it is `#providers` on this page now.)*

### 7.6 Testing contract for the page

- Every module has a node-driven test that lifts its renderers with `fn_body` and feeds them the committed fixtures (today's shapes) and, for the §9 fields, the target-shape copies under `fixtures/admin_analytics/target/` until PR D folds them in; each §9 slot also has an absent-field test. The test fails if a renderer touches `innerHTML` with unescaped payload text.
- `admin.html` has a source-shape guard: every `<script>` carries `src`; no inline scripts; the gate module loads first; all `?v=` values match the pinned set; and **no element inside a panel region contains a numeric or percentage literal** (placeholders are `—`), which is the mock's actual failure mode and the property "carries no data" rests on.
- `test_admin_tabs_redirect.py` is rewritten: entering the admin view no longer navigates anywhere; the Analytics rail button is gone.
- `test_app_composition.py` route contract gains `("GET", "/admin")`, `("GET", "/admin.css")` and the redirect; loses `("GET", "/admin-analytics")` after the redirect release.
- New: `test_middleware_exemptions.py` pins `is_exempt("/admin")` and, until the redirect goes, `is_exempt("/admin-analytics")`; `test_vercel_cache_headers.py` gains the `/admin` cases and asserts the `/admin/:path*` rewrite is absent.

---

## 8. Surface inventory

### 8.1 How the mock actually renders

The page was executed under jsdom, block by block. Five facts a reader of the markup would get wrong:

1. The **taxonomy** on screen is the shipped six groups (Internal 22 % · Invited 18 % · Organic 22 % · Competition 12 % · Partner 18 % · Unknown 8 %), written by the sixth script block. The markup's `Community / Student / Friend referral` never renders — except in the `#profiles` table and the `#profile` identity line, which the sixth block did not touch.
2. The **activation funnel** has five implementations; the last block wins with a "layered route" (80 → 60 → 50 → 42, "% remain"). Two of the five are dead code that query a class an earlier block renamed.
3. The **"What is blocking users?" panel** is deleted at runtime by the fourth block; its bars reappear only inside the `#health` detail route.
4. The **live health** section's sparklines, run lanes and "Workload profile" composite are all destroyed; what renders is a 2×2 tile grid: Online now 12 · Active runs 8 · Queued 3 · Blocked users 1.
5. The five rows are **regrouped** by the ninth block into a different pairing and order.

Every number is a literal; the footer reads *Layout proposal · synthetic data · no live account changes*.

### 8.2 Survival table

Applying D2 to every element of the **final** DOM. "Freshness" is a §6.3 tier. "Cost" is what one page load costs the backend after PR B.

| Element (mock) | Verdict | Source after PR B/C | Freshness | Cost |
|---|---|---|---|---|
| Header, product nav, aside (Analytics subnav · Users · Providers · Activity) | keep | none | — | 0 |
| "Sample data" label, footer notice | **cut** | — | — | replaced by a freshness legend: "Daily figures complete through <yesterday UTC>; live tiles are this instance's process state" |
| Range picker 1W / 1M / 1Y | keep | `from`/`to` on every route | daily | 0 |
| Range picker **1D** | **cut** | — | — | no cross-user source finer than a day exists (non-goal); "today" appears only as the overview's one-day scan |
| Filter: Source | keep | `user_group` param on `/users`, `/groups` | — | 0 |
| Filter: Cohort (September / August intake) | **cut** | — | — | no stored label (D9, D10); intake is the retention grid's activation week |
| Filter: Lifecycle stage | keep | `lifecycle_segment` on `/users` (today's parameter name, kept in PR D's re-cut) | — | 0 |
| Filter: Paid / Unpaid | keep, relabelled **Tier** (4 values) | `commercial_tier` on `/users`, `tier_counts` on `/commercial` | — | 0 |
| Filter: "Internal accounts" checkbox | keep, relabelled **Include internal accounts** | `include_internal` | — | 0; the old label collides with the `Internal` group |
| **Live row** — Online now | **cut** | — | — | no process-state source; the only candidate, `auth_sessions.last_seen_at`, is written at most every 10 minutes and is unindexed, so the number would be neither live nor cheap (§14) |
| Live row — Active runs | keep, relabelled **Backtests running · this instance** | `stats.active_dashboard_backtests` | live | 0 (process memory) |
| Live row — slot ceiling ("12 / 24 slots" in the mock) | keep, real value | `stats.max_active_dashboard_backtests` (new; default 5) | constant | 0 |
| Live row — **Queued** | **cut** | — | — | the platform has no queue: over-capacity requests are refused, not queued |
| Live row — Blocked users | **cut from the live row** (duplicate) | survives in the attention panel | daily | — |
| Live row — **Total users**, **Total agents** (absent from the mock) | **added** (D12 #1) | `stats.users`, `stats.agents` | live | 2 COUNT queries, already made today |
| Live detail page (`#live`): eight run lanes all reading "running", queue-pressure chart, "Current blockers" | **cut**; the live row has no detail page and no `#live` route | — | — | the lanes carry no information beyond the count (a per-run list is a follow-up, §14); there is no queue; blockers are the daily operational reasons, shown in `#health` |
| Users needing attention (7 · Blocked 1 · Needs attention 4 · At risk 2) | keep | `/operational.operational_state_counts`, `/lifecycle.segment_counts.at_risk`; list from `/users?priority=true` | daily | 3 grouped reads over yesterday's facts |
| Attention note: top blocking reason | keep | `/operational.top_operational_reasons` (new; from `user_daily_facts.operational_reason_code`, added in PR A) | daily | 1 grouped read |
| Active users (32; bars by day) | keep | `/overview.daily_active_users`, `active_users_7d` | daily + today | rollups rows for the range + one one-day scan |
| Activation progress (80 → 60 → 50 → 42; 52.5 %) | keep | `/overview.activation_funnel`, `first_success_conversion` | daily + today | same call; the "% remain" is presentation over the counts |
| Where users come from (donut, 100 users) | keep | `/groups.groups[].users` | daily | 1 grouped read |
| Users coming back (W0 / W1 / W2 / W4 by activation week; 66.7 %) | keep | `/retention.cohorts`, `summary_week_1` | daily | 1 grouped read over facts per grid |
| Users reaching value (not yet · once · repeat; 52) | keep | sums of `/groups.successful_run_users`, `repeat_users` (or `/lifecycle.headline.activated_users`) | daily | same `/groups` call |
| User lifecycle (6-segment bar, 80) | keep | `/lifecycle.segment_counts` | daily | 1 grouped read |
| Credits usage (10.260 settled; Platform vs BYOK by day) | keep | total: `/commercial.selected_period.consumed_micro` (ledger); per-day lanes: `/overview.billing_lane_mix` (new; rollups `billing_mode`) | live (ledger) + daily | 1 ledger aggregate via `CreditsStore` + rollups rows |
| Revenue (15.000 purchased; line by day) | keep | total: `/commercial.selected_period.purchased_micro`; series: `/commercial.purchased_by_day` (new; `CreditsStore.sum_ledger_by_day`) | live | 1 grouped ledger read |
| Hidden duplicate legend under the credits chart (`.inline-key`, `display:none`) | **cut** | — | — | dead markup; the visible legend inside the chart is the one that survives |
| "Admin Grants excluded from revenue" copy | keep | §15 Commercial Value | — | 0 |
| `#sources` detail (six-row table) | keep | `/groups` | daily | as above |
| `#retention`, `#lifecycle`, `#credits` details | keep | `/retention`, `/lifecycle` (+ rules table from §15), `/commercial` | daily / live | as above |
| `#health`: failed runs, success rate, failure-category table | keep | `/operational.failed_runs`, `backtest_success_rate`, `top_failure_categories` (D13, from rollups `error_category`) | daily + today | rollups rows |
| `#health`: "Affected users" column | **cut** | — | — | rollups are anonymous; a per-category user count would need a raw scan |
| `#profiles` → **`#users`**: list (user · source · lifecycle · operational · last active) | keep | `/users` items, which gain `user_group`, `role`, `group_badge`, `last_meaningful_activity_at` in PR D's re-cut (the page renders the two it shows as `—` until then); filters `role`, `commercial_tier`, `user_group`, `lifecycle_segment`, `operational_state` | daily (list) | 1 SQL-filtered, SQL-paginated query (not Python) |
| `#profile` → `#users/{id}`: badges, blocker, milestones, tabs | keep | `/users/{id}`, `/users/{id}/activity` | live | ≤ 1 raw-event query per tab page |
| Profile identity "Community · September intake" | **replaced** | group badge + activation week | — | — |
| Profile facts Region / Device / Browser / top page (wired dashboard) | **cut** (D15) | — | — | no home in the daily facts; collection decision is a follow-up |
| Rules dialog ("How states are determined") | keep | static, from §15 | — | 0 |
| Orphan routes `#usage`, `#revenue` | **cut** | — | — | already unreachable; merged into `#credits` |

### 8.3 Additions the survival rule forces

Three things the mock does not show, added because D12 puts them first: total users, total agents, and the real slot ceiling. One thing the mock shows as fiction that becomes real: the live row's caveat label, which reads *this instance* rather than *Sample snapshot*.

---

## 9. Metrics: kept, added, dropped

**Kept, re-sourced (PR B):** `top_failure_categories` (raw scan → rollups `error_category`); `activation_funnel`, `daily_active_users`, `active_users_7d`, `first_success_conversion`, `repeat_run_rate`, `backtest_success_rate` (raw scan → rollups + one-day scan); `segment_counts`, `operational_state_counts` (snapshots → yesterday's facts); retention cells (raw scan → facts grouped by activation week); `/groups` run counts (raw scan → facts).

**Added (PR C renders them wherever a response carries them, from target-shape fixtures, and shows "Awaiting data source" where it does not; PR B computes and unit-tests the service methods; PR D puts the fields on the response models in the re-cut):**

| Field | Route | Source | Why |
|---|---|---|---|
| `billing_lane_mix` (per day, platform vs BYOK run counts) | `/overview` | rollups `billing_mode` | D14 |
| `top_operational_reasons` | `/operational` | `user_daily_facts.operational_reason_code` (new column, PR A) | the attention note; "why are users blocked" without per-user reads |
| `purchased_by_day`, `consumed_by_day` | `/commercial` | `CreditsStore.sum_ledger_by_day` (both twins) | revenue and credits charts |
| `user_group`, `role`, `group_badge`, `last_meaningful_activity_at` | `/users` items, `/users/{id}` | `users`, `resolve_group_badge` (§6.2) | D11; the users list is the group view |
| `max_active_dashboard_backtests` | `/api/admin/stats` | the parsed constant | live row's real ceiling. Added and exposed in PR C; this route is outside the D18 freeze |

**Dropped:**

| Field | Where | When | Why |
|---|---|---|---|
| `user_state_counts` | `/overview` | PR B | five-state vocabulary deleted (D20) |
| `status` filter (five-state) | `/users` | PR B | same vocabulary; the only sender is the dead legacy attention queue, whose query and test pin go with it |
| `country_code`, `device_category`, `browser_family`, `top_product_page` | `/users/{id}` display | PR C | D15; fields remain in the payload until the collection decision |
| `credits_metering_enabled`, `default_credits` | `/api/admin/stats` | unchanged | still read by the old console's Users panel, which stays (D5) |

---

## 10. API contract policy

Consolidating D18–D20 into rules an implementer can check:

1. **PR 0, PR T, PR A:** every response of the nine `/api/admin/analytics/*` routes is byte-identical to `main` for identical inputs. PR A changes no read path at all.
2. **PR B:** byte-identical **except** that the five-state vocabulary leaves: `user_state_counts` is removed from `GET /overview` and the `status` query parameter from `GET /users` (D20). The existing `tests/test_admin_analytics_api.py`, `test_admin_analytics_frontend.py` and `test_admin_analytics_value_frontend.py` are the conformance oracle and are edited only for those two items.
3. **PR C** changes no analytics route: every response is byte-identical to `main` for identical inputs, whether C lands before or after B. The page sends only query names today's parser allows (`from`, `to`, `include_internal`, `user_group`, `lifecycle_segment`, `commercial_tier`, `q`, `priority`, `limit`, `offset`, `section`, `cursor`) and renders §9's fields when present, "Awaiting data source" when absent. Its target-shape fixtures under `tests/fixtures/admin_analytics/target/` are the proposed contract, pinned to differ from the committed fixtures by exactly the §9 fields.
4. **PR D** re-cuts the contract once, after B and C. New fields from §9 land; dead fields go; the axis filters keep today's parameter names (`role` new, `commercial_tier`, `user_group`, `lifecycle_segment`, `operational_state`) and become declared FastAPI query parameters so they appear in `app.openapi()`; the hand-rolled `_query_values()` helper is deleted. The committed fixtures gain the fields and C's `target/` copies are deleted in the same PR.
5. **Live operations.** `GET /api/admin/stats` gains `max_active_dashboard_backtests` (the parsed constant) in PR C. It keeps its six existing keys, reads process state and two owning stores, and never imports the analytics domain. These are the only sub-day numbers on the page.
6. **`POST /api/analytics/events`** (ingestion) is unchanged throughout; its allowlist is owned by the 08-26 document.

---

## 11. Retention and the tables that outlive it

| Table | Carries a user id | Retention | Owner of the rule |
|---|---|---|---|
| `analytics_events` | yes | 180 days (08-26) | retention sweep |
| `user_daily_facts` | yes | 180 days | retention sweep, after long-term rollup |
| `lifecycle_transitions` | yes | 180 days | retention sweep (easy to forget: nothing reads it yet) |
| `analytics_daily_rollups` | **no** | **indefinite** | nobody sweeps it, by design |
| `user_activity` (created in PR A) | yes | **never swept** — see below | ingestion (one row per user, overwritten in place) |
| `user_analytics_snapshots` (today) | yes | never swept today | holds `activated_at` until PR B drops it; see the migration note |
| `admin_analytics_access_log` | yes (viewer and viewed) | 180 days (08-26) | retention sweep |
| `analytics_subject_settings` | yes | **never** | a setting, like `user_activity`; one row per excluded user |
| `agent_runs.owner_user_id` (added in PR A) | yes | follows `agent_runs` (no sweep; run history is durable by design) | run-history store |

**The `user_activity` exception, stated in words.** Lifetime metrics ("activated ever", "first activation date") survive solely because `activated_at` is set once, on a one-row-per-user table that is never swept. Today that table is `user_analytics_snapshots`; after PR A it is `user_activity`. The 09-12 plan states a blanket rule that any table carrying a user id is retained 180 days, and `user_activity` is keyed by user id. It is the exception: **`user_activity` is a current-state row, not history, and the retention sweep must never touch it.** A future tidy-up that "completes" the 180-day rule over every user-keyed table would delete the only lifetime evidence in the system with every test still green. PR A pins this with a test that runs the sweep past the horizon and asserts `user_activity` row counts are unchanged.

**The migration must carry `activated_at` across.** The 09-12 plan's task B9 copies eight weeks of daily snapshots into `user_daily_facts` and drops the value columns from `user_analytics_snapshots`, but nothing in it seeds `user_activity` from the old row. Ingestion (task B4) only writes `user_activity` for events that arrive after the deploy, and the daily job's one-day scan cannot see an activation from last year. Without a seed, every existing user's activation date is lost on the night the old table is dropped, while `test_activated_at_holds_the_earliest_success` stays green. PR A therefore seeds `user_activity` from `user_analytics_snapshots` (`activated_at`, `last_meaningful_activity_at`) when it creates the table, PR B re-runs the same idempotent seed immediately before dropping the old table, and a test seeds a snapshot row, runs the migration, and asserts the timestamps survive.

**The permanent table is the rollups, not the facts.** Failure categories, billing-lane mix and per-provider/model outcome counts live in `analytics_daily_rollups`, which is anonymous and therefore has no privacy horizon. This is why D13 and D14 are free.

---

## 12. Corrections to the 2026-09-12 plan

The 09-12 plan is the base text for the PR A and PR B plans. A 25-agent convention audit found that, as written, it makes three things worse, and the new plans correct each:

1. **Cross-domain raw SQL would go from two readers to four.** Task B7 correctly adds batched public methods to the owning domains; task B8 step 3 then puts `aggregate_ledger_for_day` and `aggregate_operator_cost_for_day` on `ValueAnalyticsStore` with hand-written SQL against `credit_ledger_entries`, `credit_llm_usage_entries` and `agent_runs`. **Correction:** those two aggregates are methods on `CreditsStore` / `PostgresCreditsStore` and on the run-history store, and `backfill.py`'s existing hand-rolled ledger read is moved onto the same methods in the same task. After PR A, `rg "_get_connection" domain/analytics` returns only the analytics stores' own connections.
2. **About twelve dual-dialect methods would land on the untwinned class**, while the plan's own text (`plan:2978`) assures the implementer that parity "compares the public method set and every signature, so a method on one side only fails the build". **Correction:** PR T extracts `PostgresValueAnalyticsStore`, registers it, and adds the absence-direction check; PR A then adds its methods to both twins and the guard really does cover them.
3. **PR C's frontend tasks target the dead surface.** The plan was written the same day #467 shipped and never saw it; it would build axis filters and the group badge into the in-app panel no admin can open, and `test_admin_analytics_value_frontend.py` would go green. Measurable staleness: it records `admin-tabs.js?v=5` / `styles.css?v=139`; `main` is at `?v=7` / `?v=140`. **Correction:** the UI plan is rewritten from scratch against §7 and §8; task C6 is not carried forward.

Two further amendments, from the same audit:

4. **The daily job leaves the heartbeat thread** (D23). The 09-12 design scheduled it "from the existing run-reaper tick". It keeps the compare-and-set day claim and gains its own worker thread with its own interval.
5. **The read-budget test bounds wall-clock as well as query count.** The 09-12 test asserts a constant query count independent of user count. It keeps that and adds an upper bound on the daily job's wall-clock at the synthetic population, so a step that is one query but scans an unindexed table is caught.
6. **The migration seeds `user_activity` from `user_analytics_snapshots`** (§11). The 09-12 plan drops the old row's value columns without copying `activated_at` anywhere; existing users would lose their activation dates on the first night. PR A seeds on creation, PR B re-seeds before the drop, with a test.
7. **The old model is dropped only after the reads have moved.** The 09-12 plan deleted the snapshot tables and `states.py` in its data-model PR (task B9) while its read-paths PR (C1–C5) was the one that stopped reading them, so prod would have read dropped tables between the two. Here PR A only creates; PR B rewires every read path and then drops, in the same PR.
8. **`_TWINS` lists the `AnalyticsStore` pair twice.** Harmless today (both instances pass or fail together) but it is the list PR T extends; PR T removes the duplicate so the registry reads as the eleven distinct pairs it is.

---

## 13. Delivery

Seven pull requests, each leaving prod working on its own. Boundaries are drawn where a reviewer could reject one PR while approving its neighbour.

| PR | Name | Content | Must not |
|---|---|---|---|
| **0** | Burner kill | `states.py` stale window 15 min → 24 h. `service.py::_build_analytics_service`: `project_snapshots=False`, **and** `instrumentation.py::_emit`'s fallback recompute disarmed (it fires exactly when the service does not project, so the flag alone would move the burner, not remove it). `AdminTabs.onEnter()` returns whether it scheduled the redirect, and `app.js`'s admin branch calls it **first** and returns early when it did, so neither analytics module nor the stats and user-list loads fire into a departing page. `api/auth.py` maps `psycopg.OperationalError`, `sqlite3.OperationalError` and `PoolTimeout` to 503 with an `ERROR: auth.user_store_unavailable` log line on `get_current_user`, `login`, `signup` and `me` (the 09-12 plan's task A4, the one A task that survives). One `CLAUDE.md` gotcha. Tests for each; `app.js?v`, `admin-tabs.js?v` bumped and the pins updated. | Build any other part of the old stopgap: no shared stale constant, no one-read recompute, no spy-store sweep budget. |
| **T** | Store twin | Extract `domain/analytics/value_repository_postgres.py` (`PostgresValueAnalyticsStore`) from the 21 `is_postgres` / 3 `hasattr(…, "database_url")` branches; register it in `_TWINS` and remove that list's duplicate `AnalyticsStore` entry; add the absence-direction check to `test_store_twin_parity.py` (grep the backend for `is_postgres` and `hasattr(*, "database_url")` outside a registered twin; assert zero **unexplained** hits — the non-twin helpers that branch over an injected, already-twinned base store (`states.py`, `rollups.py`, `query_service.py`, `lifecycle_backfill.py`, `backfill.py`, and the two `credits_base` checks left in `value_repository.py`) are named in an allowlist with a reason each, and a second assertion fails on any allowlist entry whose file no longer branches, so the list can only shrink). Behaviour-preserving. Green on the CI Postgres tier before merge. | Change any query, response or table. |
| **A** | The rewrite: create | The 09-12 plan's B1, B3–B8 and B10 with §12 corrections; B2 (`cohort`) struck; B9's drop deferred to PR B. New tables on both twins (`user_daily_facts` with `user_group` and `operational_reason_code`, `user_activity`, `lifecycle_transitions`); `agent_runs.owner_user_id`; ingestion-maintained `user_activity`, **seeded from `user_analytics_snapshots`** on creation; read-time lifecycle calculator (not yet wired to routes); the day claim on the existing `analytics_projection_jobs`; population operational signals via owning-domain methods; the daily job on its own worker thread, which takes over `rollup_day` and the retention coordinator from `run_analytics_maintenance` (the reaper sweep keeps only the throttled snapshot repairs until PR B); ledger/run aggregates on `CreditsStore` / `PostgresCreditsStore` / run store, `backfill.py` moved onto them and its absence-allowlist entry removed; copy of eight weeks of history into `user_daily_facts`; hygiene (D24); the `user_activity` sweep-exception test; budget (queries + wall-clock, all stores spied) and event-log-discipline tests. **Every existing read path and table is left running.** | Change any response shape. Drop any table or column. Touch the frontend. Add `cohort`. |
| **B** | The rewrite: read paths, then delete | The 09-12 plan's C1–C5 amended: overview, lifecycle, retention, operational, users and profile routes onto `user_daily_facts`, `user_activity` and rollups, with `/users` filtered and paginated in SQL; `users_needing_attention` re-sourced from operational state; `top_failure_categories` from rollups `error_category`; the overview's current-day scan narrowed to one UTC day; the activity route's 30-day sessions window; service methods for `billing_lane_mix`, `top_operational_reasons`, `purchased_by_day` / `consumed_by_day` (`CreditsStore.sum_ledger_by_day`), `resolve_group_badge`, `get_user_metrics` with the audience filter — unit-tested, **not on the response models yet**; long-term rollups by tier and `user_group` (encoding in §6.7). Then the deletions B9 planned: re-seed `user_activity`, drop `user_lifecycle_daily_snapshots` and `user_analytics_snapshots`, delete `states.py`, `lifecycle_backfill.py`, the repair pass, `run_analytics_maintenance` and its reaper registration, the five-state vocabulary everywhere including `rollup_day`'s `user_state_count` rows, `user_state_counts` and the `status` filter (D20), and the dead `status` query in `admin-analytics.js` with its test pin. Shapes otherwise held. | Re-cut the contract. Touch the frontend beyond the dead `status` query. |
| **C** | The `/admin` page | `admin.html` + `admin.css` + four IIFE modules (§7.2) built from the mock's surviving elements (§8); client shell gate; live row from `GET /api/admin/stats`, which gains `max_active_dashboard_backtests`; users list and profile absorbed (account management stays in the old console, D5); **no contract change** (D18): the page renders §9's fields when a response carries them and "Awaiting data source" when it does not, from target-shape fixtures under `tests/fixtures/admin_analytics/target/` (today's fixtures plus exactly the §9 fields, pinned); D15's display fields dropped from the page; route rename in `app.py` (+ `/admin.css`, 308 from `/admin-analytics`), `vercel.json` (`/admin/:path*` rewrite deleted, redirect added), `middleware.py` (`/admin` added, `/admin-analytics` kept until the redirect goes), `admin-tabs.js` and tests; deletion of `admin-analytics.html`, `admin-analytics.js`, `admin-analytics-value.js`, the analytics panel, legacy block, profile article and dialogs in `app.html`, their CSS, and the redirect; old console rail loses the Analytics tab only, `DEFAULT_TAB` = `users`; node-driven tests for every module plus the middleware-exemption and `vercel.json` guards (§7.6). | Port Users, Providers or Activity. Compute the group badge client-side. Add inline scripts. Add a `#live` route. Change an analytics response model, query parameter or store. Send a query name today's parser does not allow. *(Amended 2026-09-17: this prohibition bound **PR C only**. Providers was ported to `/admin` by the 09-17 navigation consolidation, which executes D5's staged absorption ahead of schedule. Account Management and Activity remain unported.)* |
| **D** | Contract re-cut | The one re-cut (D19): §9's fields on the response models, populated from B's service methods; the `role` filter; axis filters as declared FastAPI query params, `_query_values()` deleted, every 422 body byte-identical; the committed fixtures gain the fields, C's `target/` copies and `test_admin_target_fixtures.py` are deleted, and the two renderer test modules read the committed fixtures again. No frontend module changes — the page already renders every field, and its absent-field branches stay as the fail-visible guard. | Touch a store or a table. Change a JavaScript file. Rename a query parameter. |
| **docs** | This document | This design, the five plans, and the disposition in §1. | — |

Ordering constraints (re-ordered 2026-09-16): 0 first and alone. **C next** — it depends on nothing but 0 (it reads today's routes), and putting it in front of the advisor before T/A/B is the point of the re-ordering. T before A. A before B; B must not land until A's daily job has written at least eight days of `user_daily_facts` in prod, so that the read paths it switches to have data behind them (the copied history covers the movement charts; the sums the segment rule needs come from the new rows). D after B (it calls B's service methods) and after C (it retires C's target fixtures). The docs PR can land any time after 0; it should land before C so that C's reviewers read the current design. Until B lands, the new page drives today's raw-scan read paths — admin-only and acceptable at §2.3 scale, named in C's plan.

---

## 14. Follow-ups named here so they are not silences

- **Collection decision for page, country and device** (D15): whether to keep collecting those event properties at all, owned by the 08-26 document's allowlist. Decide after PR C ships; today they are collected and not displayed.
- **Activity and Account Management absorption into `/admin`** (D5). *(Amended 2026-09-17: Providers landed in the navigation consolidation; the two that
  remain are one 712-line module and port together.)*
- **Multi-replica live counters** (D17 caveat): the slot ledger is per-process. If `numInstances` ever exceeds 1, the live-ops route needs a shared counter or must label itself per-instance. Not a problem today.
- **User-facing administrator documentation**: the published tree has no page on the admin console, segments or groups. Nothing is stale because nothing exists; a page is owed once PR C ships.
- **Route-contract guard depth**: the freeze guard protects the route set, not shapes. After PR D re-cuts the contract, consider snapshotting response schemas from `app.openapi()` so a dead field cannot survive silently again.
- **A per-run list for the live row** (§8.2): the slot ledger holds per-slot state, so "which backtests are running" is answerable from process memory. Cut from PR C as decoration; add when someone asks for it.
- **An "online now" tile** (§8.2): would need an index on `auth_sessions.last_seen_at` on both twins and an honest label ("sessions seen in the last 15 minutes", given the 10-minute write throttle in `session_tokens.py`). Cut from PR C; decide once the page exists.
- **A tracking issue.** No GitHub issue tracks the outage stopgap, the rewrite, the static preview or the redirect. This document is the tracking artefact until one is filed; filing one assigns work to others and is FlyM1ss's call.
- **`CLAUDE.md`** says nothing about `/admin-analytics`, `user_group`, the nine endpoints or the outage mechanism. The documentation PR adds an "Admin layer" pointer to this document; PR 0 and PR A update it as they change the facts.
- **`analytics_daily_rollups` growth**: unbounded by design (anonymous, one row per distinct key per day). At the §2.3 scale that is tens of thousands of rows a year, not a concern; noted so nobody adds a sweep to "fix" it without reading D13/D14.

---

## 15. Definitions carried forward

The following is the 2026-09-03 design's text, carried unchanged so that this document stands alone. Headings are shifted one level. One bracketed editorial note marks the single sentence the rest of this document overrides.

### 15.1 Definitions

#### Activation

A user activates at the first server-authoritative `backtest_completed` event.
This timestamp is stable after it is first observed. A failed, cancelled,
queued, or merely started backtest does not activate a user.

#### Meaningful behavior

Lifecycle activity counts only intentional product behavior that represents
setup, execution, or commercial use:

- creating or materially configuring an Agent;
- saving, verifying, re-verifying, or selecting a model API credential;
- requesting or progressing a backtest, including a terminal failure;
- purchasing ATL Credits; and
- consuming ATL Credits through model execution.

Passive page visits, sign-in, token refresh, polling, browser refresh, and
session heartbeat events do not count. An automatic or administrator-assigned
Grant also does not count because it is not an action by the user.

An **active day** is a distinct UTC calendar day containing at least one
meaningful behavior. Multiple events on the same day count once.

#### Inactivity clock

The inactivity clock starts from the most recent meaningful behavior. If none
exists, it starts from the account creation timestamp. `inactive_days` is the
difference between that timestamp's UTC date and the calculation UTC date.
Values from 0 through 7 are recent, 8 through 29 are At risk, and 30 or more
are Dormant. This removes sub-day gaps and makes the result independent of a
browser timezone.

### 15.2 Lifecycle Segments

Each non-excluded user receives exactly one current lifecycle segment.

| Segment | Deterministic rule |
| --- | --- |
| `New` | UTC account age is 0 through 6 days and no successful backtest exists. |
| `Onboarding` | No successful backtest exists, and the user is not New, At risk, or Dormant. |
| `Growing` | A successful backtest exists, `inactive_days` is at most 7, and Core criteria are not met. |
| `Core` | In the current UTC date plus preceding 29 UTC dates, the user has at least 3 active days and at least 3 successful backtests, and `inactive_days` is at most 7. |
| `At risk` | `inactive_days` is 8 through 29. |
| `Dormant` | `inactive_days` is at least 30. |

Evaluation is deterministic:

1. `Dormant` and `At risk` are evaluated from the inactivity clock first.
2. A non-inactive, unactivated user is `New` while account age is less than 7
   days, then `Onboarding`.
3. A non-inactive, activated user is `Core` when the Core thresholds are met,
   otherwise `Growing`.

At risk and Dormant reasons include one of these reason qualifiers:

- `never_activated`
- `previously_activated`

The current identity always uses these fixed rolling windows. Changing the page
date filter never changes a user's current segment.

#### Explainability

Every lifecycle result includes:

```text
lifecycle_segment
lifecycle_reason_code
lifecycle_reason
lifecycle_evidence
calculated_at
```

`lifecycle_evidence` contains display-safe facts rather than a private event
payload. Examples include:

```text
4 active days in the trailing 30 days
6 successful backtests in the trailing 30 days
last meaningful activity 2 days ago
first successful backtest on 2026-08-21
```

The interface exposes the rules in three places:

- a concise tooltip on each lifecycle label;
- a keyboard-accessible `How segments work` side panel containing all rules;
- user-specific evidence in the priority-user panel and User Analytics Profile.

### 15.3 Operational State

Operational state is calculated separately from lifecycle and has this
precedence:

1. `Blocked`
2. `Needs attention`
3. `Healthy`

`Blocked` represents an unresolved condition that currently prevents a core
action, including an account Credits restriction, missing usable billing lane,
or disabled selected provider. `Needs attention` represents an actionable but
not necessarily blocking condition, including an invalid default credential,
three consecutive failed terminal runs within 24 hours, or a run beyond its
safe deadline. `Healthy` means no supported current blocker or attention rule
matched; it is not a promise that every external provider is available.

Every operational state returns its own reason code, human-readable reason,
evidence, and calculation time. Lifecycle and operational badges are shown
side by side where both are relevant.

The existing mixed `status` field and filters remain temporarily available to
old consumers. New Admin Analytics UI code must use `lifecycle_segment` and
`operational_state`; it must not infer one from legacy `status`.
*[Editorial note: the first sentence is superseded by D20. The legacy `status`
field and filter are deleted in PR A; the second sentence stands.]*

### 15.4 Commercial Value

Commercial value is based on lifetime net purchased ATL Credits:

```text
net_purchased_micro = settled purchase entries - settled refund entries
```

The Credits ledger is authoritative. Model consumption does not reduce
lifetime net purchases, and Admin Grant assignment or reclaim never contributes
to it.

| Tier | Lifetime net purchase |
| --- | ---: |
| `Unpaid` | exactly $0 |
| `Starter` | greater than $0 and less than $5 |
| `Invested` | at least $5 and less than $20 |
| `High value` | at least $20 |

Refunds can move a user to a lower tier. The displayed value uses the existing
one-dollar-to-one-ATL-Credit accounting contract.

The Commercial Value section also shows these independent measures without
combining them into revenue:

- ATL Credits consumed in the selected period;
- current Grant, Purchased, and total available balance;
- platform model cost in the selected period; and
- Admin Grant activity, clearly labelled as non-revenue.

### 15.5 Retention Cohorts

Retention starts at the first successful backtest rather than signup.

- Cohort membership uses the UTC Monday-to-Sunday week containing activation.
- That activation week is Week 0.
- Week 1, Week 2, and Week 4 retention require at least one meaningful behavior
  in the corresponding later UTC calendar week.
- A target week enters the denominator only after the full target week has
  elapsed.
- An immature cell is unavailable, never zero.
- Summary percentages are `retained eligible users / all eligible users` across
  mature cohorts; the cohort table also shows each weekly cohort independently.

The date filter selects activation cohort weeks and trend coverage. It does not
reassign users to a historical current lifecycle segment.

### 15.6 Loading, Empty, and Error States

Lifecycle, Retention, Commercial, Operational, and Users are independent query
boundaries. A failure in one never blanks another.

- Stable-size skeletons prevent the layout from moving during loading.
- Empty data says what is empty; it never resembles a loading or error state.
- A failed section shows `This section is temporarily unavailable` and a local
  retry action.
- Incomplete historical evidence shows coverage dates and `Incomplete data`.
- A stale successful response may remain visible with a stale indicator while
  its refresh fails.
- Authentication or authorization loss follows the existing Admin exit path.
- Server and browser errors remain display-safe and never include SQL, raw
  events, provider response bodies, secrets, or stack traces.

Analytics calculation, snapshot, daily history, aggregate rollup, or backfill
failure never changes authentication, Agent, backtest, provider, Credits, or
payment outcomes.

### 15.7 Accessibility and Responsive Behavior

- The Admin rail uses a vertical ARIA tablist and complete keyboard navigation.
- Narrow layouts retain a left icon rail with accessible names and tooltips.
- Segment controls are buttons with selected state and visible focus.
- Collapsed analysis sections use buttons or native disclosure semantics with
  correct expanded state.
- The rules and quick-evidence side panels have dialog names, focus containment,
  Escape-to-close, backdrop close, and focus return to the opener.
- Charts provide descriptive labels and equivalent hidden data tables.
- Color is never the only signal for a segment, trend, quality state, or error.
- Tables keep column headers and become horizontally scrollable only when a
  vertical mobile representation would lose meaning.
- Text, controls, and badges must not overlap at supported desktop and mobile
  widths.
