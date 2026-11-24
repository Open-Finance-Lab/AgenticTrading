# Portfolio Management demo subtab — design

Date: 2026-11-04, revised 2026-11-05. Status: draft for owner review.

## 1. Intent

A recorded run of an agentic weekly portfolio rebalance (proposer, solver, code checks, certificate, critic, gate) is
played back step by step on one page. The page already exists as a standalone build ("weekly-v2", case 2) and the
owner has approved its content. This design puts that page inside ATL as a subtab of the My Agents page.

**Why it goes in ATL.** It is shown as a keynote-style live demo at a talk on 2026-11-06.
- At the talk, the presenter switches from the slides to the browser and opens ATL. They land on, or click into, the
  My Agents page, click **Portfolio Management**, and start talking straight away, clicking the step arrows to move
  through the stages. Then they switch back to the slides.
- Beforehand, the presenters and the PI rehearse on the same page and click around in it themselves.
- Anyone who opens ATL can follow the same path. Showing it inside the product, not as a separate file, is the point.

**After the talk it stays as is.** It becomes a permanent, public, static page. There is no plan to turn it into a
feature users run; it may grow into one only if future funded work calls for it. The lab's main engineering focus
stays elsewhere (agent traces), so the page must cost nothing to leave alone.

## 2. Goals

1. A subtab labelled **Portfolio Management**, placed **before My Agents** in the My Agents page's subtab row
   (`.playground-subtabs`, `dashboard/frontend/app.html` near line 954). The page inside keeps its own heading,
   "Weekly Portfolio Rebalancing".
2. The page looks and behaves like the approved standalone build's shell view (the published artifact, Version 6).
   - Content is the same: steps, captions, numbers, the top step controls and the loop header's ← ▶ Play → controls,
     the Details tabs (Proposer · Solver · Checks · Certificate · Critic), the About panel, and all labels.
   - Two things go: the build's stand-in ATL chrome, and stage view (§7).
   - This is a port, not a redesign.
3. It opens ready to present. It shows step 1 the first time the tab opens in a page load, and again after a reload.
   Switching to another tab and back resumes where it was.
4. It is reachable by clicking the tab and by a direct link, `/app?view=portfolio-management`. A reload keeps the
   presenter on the tab.
5. Every other ATL page behaves exactly as before: same default tab, ticker, keys, URLs, styles, history and
   scrolling.

## 3. Non-goals

- Running anything live: no LLM call, no solver, no backend route, no fetch, no spend. The page reads only its
  bundled recording.
- Case 1 (the 5% cap / client approval run). Case 2 only.
- A phone or tablet layout (desktop and projector widths only), a light theme (ATL is dark-only), analytics page
  views for the new tab, regenerating the recording.
- Stage view, the build's projector mode. The owner dropped it on 2026-11-05, and the presenter uses the normal page.
- A recorded ticker. ATL's real ticker stays as it is on every page.
- Editing user-facing docs (see §8).

## 4. Anti-goals (a failure even if it works)

- The new tab becomes the landing tab. My Agents stays the default for every visitor and every fallback.
- Any ATL page changes behaviour because the demo's code is loaded: a swallowed Space or arrow key, a rewritten URL
  or `history.state`, a restyled element, a timer running in the background.
- Copy that implies real trading or a live run. The page must not say "deterministic", "live", "paper trading",
  "audit" or "sample" in visible copy, and must never use the name "Certified Rebalance". The recorded-run badge and
  provenance chips stay as approved.
- Content drifting from the approved build: no new numbers, captions or claims slip in during the port.

## 5. Constraints (verified on origin/main 5a49d00a, 2026-11-04)

**Subtab mechanics.**
- The subtab row is three hard-coded buttons (agents, backtest, paper); the agents button carries a static `active`
  class.
- The default tab is the literal `'agents'`: `let playgroundTab = "agents"` in `app.js` (about line 6509), plus about
  11 `|| 'agents'` fallbacks and the boot script default (`app.html` about line 131). Button order decides nothing, so a
  first-placed button does not become the default and nothing needs to change to keep My Agents the default.
- A new tab must be wired in all of these, or it flashes My Agents on first paint, falls into the agents branch, or
  stays visible on other pages:
  - `NAV_VIEW_MAP` (`app.html` about lines 28–56);
  - the boot nav-state sanitiser (`app.html` about lines 63–68 and 120–132);
  - the boot CSS per-tab rules (`app.html` about lines 157–164);
  - `showPlaygroundPanel` (`app.js` about line 11563): an explicit branch, not the `else`, which runs `loadAgents()`;
  - `navigateToPage`'s hide list (`app.js` about lines 11755–11770);
  - `viewParamForNavState` (`app.js` about lines 11282–11300), keeping its `return 'agents'` default intact.
- Nav state persists in localStorage and `?view=`, and back/forward restores from `history.state`. ATL's boot and
  `resolveInitialNavigation` read `location.hash` as a legacy view name, so a hash equal to a `NAV_VIEW_MAP` key
  navigates.
- Leaving a subtab only sets `display:none`; ATL tears nothing down. The page must clean up after itself.

**Serving and security.**
- CSP (`dashboard/backend/middleware.py` about lines 126–131, `dashboard/frontend/vercel.json` line 9) allows
  `'self' 'unsafe-inline'` scripts and styles, Google Fonts, and `data:` images, with no `'unsafe-eval'`. The Vercel
  copy is the prod one, and its `connect-src` and `img-src` are narrower. The standalone build already uses no eval,
  no inline handlers and no innerHTML, so CSP is not a blocker.
- `/js/{file}.js` is a generic route (`app.py` about lines 572–582), so new JS files need no route. There is **no
  route for a new CSS file** (only `styles.css`, `admin.css`, `admin-console.css`), and `test_app_composition.py`
  freezes the route list. Do not add routes.
- Vercel caches `/js/*` for an hour; the `?v=N` cache-buster is what ships a change.

**Tests and pins.**
- Editing `app.js` means bumping `app.js?v=` (164 on main) in `app.html` and in every test that pins it. Grep, never
  recall: `test_frontend_fast_boot.py` owns the floors, and exact pins also sit in `test_admin_analytics_frontend.py`,
  `test_analytics_frontend.py`, `test_backtest_comparison_frontend.py` and others.
- Order-sensitive script pins: analytics after app.js before agent-editor; comparison helper before app.js;
  credit-format before credits and admin-credits. Place new tags after `app.js` and outside pinned pairs.
- `test_frontend_api_base.py` scans every frontend file: no API base, no `location.origin` use.
- No CI step parses `app.js` (or a new JS file) whole; run `node --check` on each.

**Open PRs.** Draft #676 changes the same `app.js`, `styles.css` and `agent-editor.js` pins and will conflict; it
needs a rebase anyway. #708 touches only the leaderboard pins.

**Theme and look.** ATL is dark-only with `:root` tokens in `styles.css`, plus Inter and JetBrains Mono from Google
Fonts in `app.html`. The demo uses the same palette under its own token names.

**Deploy.** Merging to `main` deploys prod: Vercel ships the frontend, and Render follows once `Backend tests` pass.
The `main` ruleset requires one approving review and a CodeQL result with no new high alerts.

## 6. Decided

| Item | Decision | By |
|---|---|---|
| Tab label | Portfolio Management | owner, 2026-11-04 |
| Placement | first button, before My Agents, in the My Agents page's subtab row | owner, 2026-11-03/04 |
| Page heading | Weekly Portfolio Rebalancing | owner, 2026-11-03 |
| Content | case 2 recording, Version 6 of the standalone build, unchanged | owner approved the build, 2026-11-04 |
| Lifespan | permanent, static, as is | owner, 2026-11-04 |
| Default tab | stays My Agents (no change to defaults) | follows from the anti-goal |
| Slug | `portfolio-management`: `data-playground-tab`, `?view=`, file names | our call |
| Files | `dashboard/frontend/js/portfolio-management.js` (page) and `js/portfolio-management-data.js` (recording) | our call |
| Stage view | dropped; shell view only | owner, 2026-11-05 |
| Ticker | ATL's real ticker stays; the build's stand-in ticker and its data line go | owner, 2026-11-05 |
| Review | the owner reviews the PR | owner, 2026-11-05 |
| Target browser | Chrome or Edge on the presenter laptops; parity renders use Windows headless Chrome | our call |
| `data.js` | ported unchanged, including keys the embed no longer reads (such as `ticker`) | our call |
| Fallback | the published standalone page, plus `dist/rebalance.html` saved on each presenter laptop | our call |
| Persistence | the tab persists like the others, so a reload keeps the presenter on it | our call |

Notes on our calls:
- **Ticker data line.** Dropping the stand-in ticker's data line ("Yahoo Finance · split-adjusted closes of Fri Sep 4,
  2026") loses nothing. The page header already says "this run: the decision of Fri Sep 4, 2026", and the evidence
  rows and About panel name the Yahoo source.
- **Persistence** means a visitor who last viewed the demo returns to it on their next `/app` visit. New visitors are
  unaffected.

## 7. The port: what changes in the standalone build

Source (outside this repo): `D:\AgenticTradingLab\Events\IBM-Nov-6\atl-demo\weekly-v2`
- `src/markup.html`, `src/app.js`, `src/style.css` (handwritten);
- `src/data.js`, generated by `content/build_case2.py` from recorder run `20261104T050945Z-3a2f26`. Do not regenerate.

The approved reference rendering is the published standalone page (Version 6).

Required changes:
1. **Remove the stand-in ATL chrome and stage view.**
   - Chrome: the demo's own header, nav, ticker, ticker info and subtab bar (markup lines 2–32), their CSS,
     `renderTicker`, and the logo data URI if nothing else uses it. ATL's real chrome, ticker included, replaces them.
   - Stage view: the "Stage view" button (`#modeBtn`), `S.mode`, the `data-mode` attribute, and every
     `[data-mode="stage"]` rule.
   - The hash prefix `stage|shell` goes with them (see 5).
2. **One global at most.** `data.js` declares `var REC`. Expose the recording under one namespaced global (or none),
   and keep `app.js`'s IIFE.
3. **Mount and unmount.** Initialise on first show of the panel, not at script load. All document and window
   listeners and all timers follow the panel:
   - registered on show, removed on hide: `keydown`, `hashchange`, `resize`, `load`, `fonts.ready`;
   - stopped on hide: autoplay and the typing and phase timers, so playback pauses.

   ATL calls the hooks from `showPlaygroundPanel` and `navigateToPage`. Scripts load with `defer`; window `load` may
   already have fired.
4. **Keys only when the page is showing.** Arrows, PageUp/PageDown, Space and Esc act only while the Portfolio
   Management tab is active, and never when focus is in a form control or outside the panel's purpose (for example
   ATL's header search). Never `preventDefault` otherwise.
5. **The URL belongs to ATL.**
   - Never write `location.hash` or call `replaceState` with a state that drops ATL's `history.state`.
   - Step and tab state lives in memory. A step deep link is optional and must not match a `NAV_VIEW_MAP` key or
     disturb `?view=`.
6. **Scope every style** under the panel root (`#playgroundPortfolioManagementPanel` or a `pm-` prefix):
   - no rules on `html`, `body`, `*`, `button`, `[hidden]` or `:focus-visible` outside the root;
   - tokens move from `:root` to the root;
   - `color-scheme` is scoped;
   - colliding class names (`chip`, `open`, `is-active`, `subtab-bar`, `sr-only`, `up`, `down`, `idle`, and generic
     ones like `tab`, `seg`, `btn`) are renamed or confined.
7. **About overlay.** It is `position: fixed; inset: 0; z-index: 40` (`style.css` line 378), which would sit under
   ATL's header. Raise it above ATL's chrome or confine it to the panel, and keep Esc and the close button working.
   The Details section stays inline, as in shell view.
8. **Delivery.** Ship as the two JS files under `dashboard/frontend/js/`, loaded after `app.js` with `?v=1`, with the
   data file before the page file. The CSS can be appended to `styles.css` (bump its pin) or carried by the page's JS.
   It must not need a new route, and a direct link must not flash unstyled content.
9. **Provenance headers.** Each new file states that it is generated from the standalone build: source, recorder run
   id, date. The data file says "generated, do not edit".

## 8. Left to the builder

- How the CSS is delivered (§7.8) and how mount hooks are wired.
- Whether to keep the 15 static inline `style=""` attributes (allowed by CSP) or turn them into classes.
- Whether a step deep link exists, and its shape.
- The test design. Minimum:
  - a pytest that pins the new button's position, `NAV_VIEW_MAP` entry and inverse, boot CSS and the
    `showPlaygroundPanel` branch;
  - a check that loading the page's scripts adds no document-level key effect while another tab is active;
  - `node --check` on the new files;
  - the full frontend suite green. (`test_frontend_no_mock_data.py` reads only `app.js` and `app.html`, so the
    recorded-data file cannot trip it.)
- Visual parity checks:
  - Windows headless Chrome renders of the ATL panel against the standalone build's shell view, at steps 1, 5, 6, 7
    and 9, at 1536 and 1920 wide (WSL Chromium does not work here);
  - an interactive pass by a person, because headless renders cannot click.

## 9. Out of scope, logged

- **User-facing docs to update later** (owner coordinates; do not edit in this work):
  `docs/source/lab/getting_started.rst` describes the My Agents page and its subtabs (about lines 7–14, 56–59,
  104–120) without the new tab.
- The talk deck's slide 6 describes "a 10-for-1 split into a −90% day"; the recorded case is a 2-for-1 split
  showing −48.73%. The deck is not owned here.

## 10. Shipping

The owner reviews the PR and decides when to merge. Merging deploys prod, so the rehearsal and the talk run on the
live site.
