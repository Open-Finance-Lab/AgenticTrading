# Admin console port — finishing PR C

Status: in progress, 2026-09-18. Branch family: `feat/admin-console-port` (cut from
`origin/main` at `05794351`, the squash merge of PR #488).

Supersedes nothing. This is the **PR2** that `admin.html:75-76` names in a comment
and that the PR C plan
(`docs/superpowers/plans/2026-09-15-admin-layer-redesign-prC-admin-page.md`)
deferred under design note N2, **plus** one requirement the design did not
anticipate (§0 below).

---

## Why this exists

Two user reports, one cause:

> "still broken nav. when i click account management or activity it jumps from
> /admin to /app. and the /app domain excludes the provider page, which admins
> can see when they are on the /admin subdomain"

> "i do want the top nav bar to remain the same across all pages no matter admins
> are on the admin page or my agents page so the newer header from the /admin
> subdomain pages will have to go, and be replaced by the top nav bar used
> everywhere else."

The `/admin` rail paints four visually identical `class="admin-tab"` entries, but
only two of them (Analytics, Providers) are in-page hash routes. Account Management
(`admin.html:77`) and Activity (`admin.html:79`) are hard links to
`/app?view=admin&adminTab=…`. Meanwhile PR C moved Providers the *other* way
(`RETIRED_TABS` in `admin-tabs.js:22`), so `/app`'s console is down to two tabs.
The rail advertises one console; the functions live on two surfaces and neither is
complete.

**This is not a regression.** It is the deferral in `admin.html:75-76` becoming
visible now that the other half of the split moved. The fix is to finish the move.

---

## Ordering note — PR D is blocked, and this doc records why

FlyM1ss asked for "the entire PR C (and D) complete before moving on to PR A and B."
**PR D cannot be reordered ahead of T/A/B.** Its own Task 1 Step 0 is a hard stop:

> "PR 0, PR C, PR T, PR A and PR B are merged before this branch is cut. Task 1
> Step 0 is the hard stop: `user_state_counts` / `legacy_status` still in the
> domain means PR B has not merged; no `tests/fixtures/admin_analytics/target/`
> means PR C has not merged." — prD plan:17

PR D ships no store, no table, no DDL — "this PR reads what PR B shipped"
(prD plan:20). Its entire job is wiring the §9 fields to service methods PR B
introduces (`billing_lane_mix`, `top_operational_reasons`, `purchased_by_day` /
`consumed_by_day`, `resolve_group_badge`) and deleting PR C's target fixtures. Run
it now and it reads from nothing.

So the executable reading of the instruction is: **finish PR C (this doc), then
T → A → B, then D.** The six "Awaiting data source" slots FlyM1ss is waiting on
are filled by **B**, not by D.

---

## Findings this doc carries forward (derived at source 2026-09-18; do not re-derive)

### The port is smaller than `admin.html:75-76` implies — two of the four parts are dead

| Part | Where | Verdict |
|---|---|---|
| `admin-credits.js` (712 lines): grant pool, credits users table, Activity | `app.html:2128-2284` + dialog `2286-2305` | **Port it.** This is the real work. |
| `app.js::loadAdminStats` → `#adminStats` | `app.js:4253` | **Do not port.** `admin-live.js` already renders users / agents / running-backtests from the same `/api/admin/stats` on `/admin`'s Overview. Only the `admins` counter and the `credits_metering_enabled` note are unique, and that flag is superseded (root CLAUDE.md). |
| `app.js::loadAdminUsers` + `_renderAdminPager` → `#adminUsersBody` etc. | `app.js:4314`, `:4280` | **Do not port — delete.** They render into `#adminLegacyUsersPanel`, which is `hidden` in `app.html:2237` with **nothing anywhere unhiding it** (grep: the only two hits are that line and one test). Dead UI superseded by `admin-credits.js`'s Account Management table. |
| `app.js::_handleAdminAccessLost` | `app.js:4298` | Still called by live app.js paths (`:4390`, `:4444`, `:4535`). **Leave it**; only the admin-view call sites go. |

`admin-users.js` (`/admin` analytics users, `/api/admin/analytics/users` only) and
`admin-credits.js` (account management, `/api/admin/credits/*` + `/api/admin/users/{id}`)
**do not overlap**. The port is a move, not a merge.

### `admin-credits.js` has exactly six cross-surface seams

`AdminShell` already exports a drop-in replacement for each.

| # | Seam | Line | Replacement |
|---|---|---|---|
| 1 | `window.CreditFormat` destructure | `:5` | unchanged — `credit-format.js` already loads on `/admin` |
| 2 | `isAdmin()` via `window.getStoredAuthUser` | `:35-37` | `AdminShell.state.admin` (set by `gate()`) |
| 3 | `window.API.request` guard + call | `:40-43` | `AdminShell.request` / `AdminShell.write` |
| 4 | `navigateToPage('home')` on access loss | `:121` | `AdminShell.handleAccessLost(error)` |
| 5 | `window.getStoredAuthUser?.()` | `:269`, `:639` | `AdminShell.user()` |
| 6 | `window.AdminTabs?.onEnter()` | `:693` | drop — routing is the shell's now |

Entry point becomes the `admin:route` event, replacing the `DOMContentLoaded` +
`dataset.navPage === 'admin'` check at the tail of the IIFE.

⚠ `admin-credits.js:118` has its own local copy of the access-lost logic and
conflates 403-CSRF with 403-role-loss — the same defect PR #488 fixed in
`admin-shell.js`. Seam 4 fixes it by construction (the shell's version is the one
that learned the difference). Do not port the local copy forward.

### The 25 element IDs `admin-credits.js` requires

All 25 live in `app.html`; **none** are in `admin.html`, so nothing collides on
arrival:

`adminCreditsActivityBody`, `adminCreditsActivityMoreBtn`, `adminCreditsAllocated`,
`adminCreditsPoolAvailable`, `adminCreditsPoolSummary`, `adminCreditsPoolTotal`,
`adminCreditsRefreshBtn`, `adminCreditsStatus`, `adminCreditsUserCount`,
`adminCreditsUserQuery`, `adminCreditsUsersBody`, `adminCreditsUserSearch`,
`adminCreditsUsersNextBtn`, `adminCreditsUsersPrevBtn`, `adminCreditsUsersRange`,
`adminGrantPoolAmount`, `adminGrantPoolForm`, `adminGrantPoolReason`,
`adminGrantReason`, `adminGrantReasonCancel`, `adminGrantReasonClose`,
`adminGrantReasonDialog`, `adminGrantReasonForm`, `adminGrantReasonStatus`,
`adminGrantReasonSummary`.

Plus two ring nodes the summary renderer touches: `adminCreditsPoolRingAvailable`,
`adminCreditsPoolRingAllocated`.

### Icons the ported markup needs

`admin.html`'s sprite has 7 symbols. The move adds **`#icon-wallet`** and
**`#icon-search`**; the header (§0) adds **`#icon-github`**, **`#icon-discord`**,
**`#icon-chevron-right`**. All five are copyable verbatim from `app.html`'s sprite.
A `<use>` with no matching `<symbol>` renders an empty box without raising, so a
missed one is silent — check all five.

### Cache-buster pins, re-derived by grep 2026-09-18

Per the standing memory note: **grep, never recall.** As of `05794351`:

| Asset | Pinned in |
|---|---|
| `admin-shell.js?v=4` | `admin.html`; `test_admin_page_shell.py` (×2); `test_admin_analytics_frontend.py` |
| `admin.css?v=3` | `admin.html` |
| `admin-credits.js?v=6` | `app.html`; `test_frontend_fast_boot.py`; `test_credit_format_frontend.py`; `test_admin_credits_frontend.py` |
| `admin-tabs.js?v=12` | `app.html`; `test_admin_credits_frontend.py`; `test_admin_analytics_frontend.py` |
| `app.js?v=134` | `app.html`; `test_frontend_fast_boot.py`; `test_backtest_comparison_frontend.py`; `test_analytics_frontend.py`; `test_admin_analytics_frontend.py` |

Re-grep at the start of every chunk — other branches move these.

---

## §0 — Chunk C1: header parity

**Goal:** `/admin` draws the same top bar as every other page.

`admin.html:35-61` is a bespoke header (`.brand`, `.product-nav`, `.account-btn`)
built for the standalone mock. Replace it with `app.html:221-270`'s `.header`.

Markup deltas that are legitimate and must stay enumerable (a guard test pins the
list):

1. `data-mode` **buttons → anchors** with real hrefs. `/app`'s copy is driven by
   `app.js`, which does not load here.
2. `#authSignInBtn` **dropped** — `gate()` already bounces a non-admin, so the
   signed-out branch is unreachable on this page.
3. Account-menu entries are `<a href>` rather than app.js-driven `<button>`s
   (`admin.html` already does this today).
4. The Admin entry is **present and unhidden**, pointing at `/admin` — only admins
   reach this page.
5. `#accountMenuLogoutError` is `/admin`-only (added by PR #488): the console is
   precisely the page you must not be left sitting on believing you signed out.

CSS: **port, do not import.** `styles.css` is 13,752 lines and collides with
`admin.css` on `.auth-btn`, `.account-menu*`, `.credits-*` and the bare
`button`/`a` rules. Retype the `.header` family into `admin.css` and alias the
app-side custom-property names on `:root` so the ported declarations copy
unchanged — substituted tokens are how two copies of one header become two
headers. Source ranges in `styles.css` at `05794351`:
`95-370` (header + brand + account button), `649-682` (`.mode-toggle`/`.mode-btn`),
`5826-5840` (`.primary-nav`, `.nav-menu-toggle`), `10597-10700` (account menu),
and the responsive blocks at `2903`, `2917`, `2951`, `2957`, `2981`, `3077`, `10041`.

Two resets the port needs that `/app` does not: `admin.css` gives every bare
`button` a 40px floor, so `.auth-account-btn` and `.nav-menu-toggle` need
`min-height:0` or they stand ~10px taller here than there.

JS: `admin-shell.js` looks up `accountBtn` / `accountAvatar` / `accountLabel`
(`:546`, `:547`, `:557`, `:702`). App's markup names them `authAccountBtn` /
`authAvatar` / `authUserLabel`. **Rename to match app's** — identical IDs are what
lets the guard test compare the two blocks — and add a `#navMenuToggle` handler
that toggles `.open` on `#primaryNav` (app.js's behaviour; `/admin` has no app.js).

Delete from `admin.css`: the bare `header{}` rule and `.brand`, `.brand img`,
`.product-nav` (lines 28-31) and their two media-query mentions (`374`, `375`).
`.workspace`'s `calc(100vh - 80px)` is sized to the old 80px bar; the new one is
85px (10px padding ×2 + 64px logo + 1px border) — put it in a custom property
rather than a second literal.

**New test:** `test_admin_header_parity.py` — read both documents, extract the
`<header …>…</header>` block from each, normalise the enumerated deltas above, and
assert what remains matches. This is the only kind of test that can catch the two
headers drifting, since neither file has a build step or a module graph.

**Cache-busters:** `admin.css` and `admin-shell.js` both change → bump both pins
everywhere the table above lists them.

---

## §1 — Chunk C2: Account Management + Activity markup on /admin

Move `app.html:2128-2284` (`#adminView`) and `app.html:2286-2305`
(`<dialog id="adminGrantReasonDialog">`) into `admin.html`.

`<dialog id="creditsRefundDialog">` (`app.html:2307-2331`) belongs to the Credits
page, **not** the admin view — leave it where it is.

Drop on the way across:

- The `#adminView` page-header, `#adminError`, `#adminSuccess` and `#adminRefreshBtn`
  — `/admin` has its own page chrome and per-panel retry.
- The `#adminTabs` nav (`:2140-2143`) — the rail replaces it.
- `#adminStats` (`:2147-2152`) — `admin-live.js` already renders it (see Findings).
- `#adminLegacyUsersPanel` (`:2237-2263`) — dead (see Findings).

Land the remainder as two sections keyed the way the shell's other views are:
`#accountManagementView` (grant pool + credits users table) and `#activityView`
(the grant audit trail).

`admin-shell.js`:
- `ROUTES` gains `'account-management'` and `'activity'`. **Not** `ANALYTICS_ROUTES`
  — they are not analytics, and `syncRail` lights the Analytics parent off that list.
- `route()` gains `showView('accountManagement', …)` and `showView('activity', …)`,
  and both join Providers in the `pageControls` / `freshnessLegend` suppression at
  `:622-623`: the range group and filter form describe daily analytics figures and
  scope nothing on these two pages.

`admin.html` rail: `admin.html:77` and `:79` become `href="#account-management"` and
`href="#activity"`, and the `PR2` comment at `:75-76` goes.

CSS: port the `.admin-credits-*` / `.admin-stat*` / `.admin-grant-reason-dialog`
families from `styles.css` into `admin.css`, same discipline as §0.

---

## §2 — Chunk C3: rewire admin-credits.js onto AdminShell

The six seams in the Findings table. Drop the file's `DOMContentLoaded` tail and
enter on `admin:route` for the two new routes, matching `admin-live.js`'s shape.

Verify under the DOM stub (`tests/_admin_dom_stub.py`), not by reading: the module
is an IIFE over `window`/`document` and `node` is present in CI.

---

## §3 — Chunk C4: retire `/app?view=admin`

- `app.html`: delete `#adminView` + `#adminGrantReasonDialog`, and the
  `js/admin-credits.js` and `js/admin-tabs.js` script tags.
- `app.js`: delete `loadAdminUsers`, `_renderAdminPager`, `adminUsersPage` and the
  `#adminUsersBody` listeners; delete `loadAdminStats` and `setAdminCreditsNote`;
  keep `_handleAdminAccessLost` (still used at `:4390`, `:4444`, `:4535`).
- `?view=admin` follows Providers' precedent (N5, `admin-tabs.js:22`): an explicit
  `window.location.replace('/admin')`, not a silent fallback to Home. Gate it on
  admin intent the way `admin-tabs.js:131` learned to, so a stale link does not
  yank a signed-out visitor off the page they asked for.
- `#accountMenuAdminBtn` in `app.js` points at `/admin`.
- `js/admin-tabs.js` and its `test_admin_tabs_redirect.py` are deleted outright once
  no tab remains — but only if nothing else imports the file. Grep first.

---

## §4 — Chunk C5: test + cache-buster sweep

- Re-grep every `?v=` pin (table above) and reconcile.
- Update the source-shape guards that reference the moved markup:
  `test_admin_credits_frontend.py` (it slices `app.html` between `id="adminView"`
  and `id="adminLegacyUsersPanel"` — both move/vanish), `test_frontend_fast_boot.py`,
  `test_credit_format_frontend.py`, `test_admin_analytics_frontend.py`,
  `test_admin_page_shell.py`.
- Full suite green: `pytest dashboard/backend/tests/ -q`. Baseline at `05794351`
  is 4912 passed.

---

## Global constraints

- **Never `git add -A` / `.` / bare `-u`.** A bare backend import rewrites
  `dashboard/storage/data/backtest.db`; stage named files only.
- `node` must be on PATH or the whole JS-behaviour layer **skips rather than fails**.
  Confirmed present (v24.15.0).
- No inline scripts in `admin.html` (design D6) — pytest can only execute frontend
  code it can lift from an external file.
- The `PENDING` / `fieldPending` "Awaiting data source" branches stay. They are PR
  B/D's to remove, and they are the fail-visible contract.
