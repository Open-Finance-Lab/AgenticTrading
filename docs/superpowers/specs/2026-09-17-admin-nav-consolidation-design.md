# Admin Navigation Consolidation

**Date:** 2026-09-17
**Status:** approved, not yet implemented
**Relationship to prior work:** this document does not supersede
`2026-09-15-admin-layer-redesign-design.md`. It *executes* that design's **D5**
("absorption is staged, and full absorption is the committed end state") ahead of
schedule for one of the three deferred tabs, and amends three of its paragraphs
where what ships will differ from what it says. The amendments are listed in §10
so a later reader of the 09-15 document is not told to undo this one.

---

## 1. Why this exists

PR #487 shipped `/admin` (the standalone console: Analytics, the users list and
per-user profiles). D5 deliberately left **Account Management**, **Providers**
and **Activity** in the old in-app console at `/app?view=admin`, with `/admin`
linking out to them.

The result, on screen, is two admin surfaces with different shells, different
navigation idioms, and a full page load between them. Every cross-link in
`/admin`'s aside is an external `href` into a differently-styled application.
The user-visible complaint is "the analytics page still lives on a separate page
than the users, providers and activity page", and the navigation problems
observed downstream all descend from that one split.

Three concrete defects, each verified against `origin/main` at `8df40dbe`:

1. **`/admin` has no account menu.** `admin.html:20-23` is brand plus four
   product links. Searching `admin.html` for `accountMenu`, `ticker` or `logout`
   returns zero matches. An admin on `/admin` cannot see who they are signed in
   as, reach Profile or Account, or log out. This is a live bug independent of
   the split.
2. **Two navigation idioms.** The old console's rail
   (`styles.css:13282-13360`) is a sticky card of 43px icon+label pills at
   13px/600 weight. `/admin`'s aside (`admin.css`, `aside a` and
   `.analytics-subnav a`) is a flat list of text links whose subnav runs at
   12px with no icons. The larger, iconed idiom is the better one and is what
   the rest of the application uses.
3. **One word, two owners.** `/admin`'s Analytics subnav has a **Users** entry
   (`#users`, the read-only analytics list and profiles, absorbed by PR C) while
   the link out to the old console is already labelled **Account management**.
   The old console calls that same destination **Users**. Two different things
   named "Users" across two pages is the artefact of a feature cut in half.

## 2. Goals

- `/admin` gains the application's navigation idiom: icons, the larger type
  scale, the sticky rail card.
- `/admin` gains an account menu, closing defect 1.
- **Providers** moves into `/admin` and out of `/app`, reducing the split from
  three tabs to two.
- The naming collision is resolved in favour of **Account Management**.
- The read-only discipline `/admin` ships today survives contact with a write
  surface **as a property**, not as a test that still runs and no longer
  protects anything.

## 3. Non-goals

- **Account Management and Activity do not move in this PR.** They are one
  module (§4.3) and port together in PR2.
- No backend change. Every endpoint this work touches already exists and is
  already `require_admin`-gated.
- No analytics contract change. **D18 and D19 stand**: the contract is re-cut
  once, in PR D, after PR B. Nothing here touches a response model.
- No change to what the Providers panel *does*. It is moved and rewired, not
  redesigned.

## 4. What prod does today

### 4.1 The two surfaces

| | `/admin` | `/app?view=admin` |
|---|---|---|
| Shell | `admin.html`, `admin.css` (274 lines, own tokens) | `app.html`, `styles.css` (13,990 lines) |
| Header | brand + 4 product links | GitHub, Discord, brand, 4 links, ticker strip, account chip |
| Nav | flat text links, 12px subnav | sticky rail card, 18px icons, 13px/600 labels |
| Routing | hash (`#overview` … `#users/{id}`) | `?view=admin&adminTab=` + in-place panel swap |
| Writes | none | roles, quotas, grant pool, providers, credentials |

`app.js:5397-5402` sends Profile → Admin to `/admin`. Nothing in `app.html`
links back to `/admin`; the only return path is the account menu, which `/admin`
does not have.

### 4.2 `admin.css` does not import `styles.css`

By design, and the file says so in its own header comment. The two stylesheets
also use different token names — `admin.css` defines `--line`, `--muted`,
`--accent`, `--text`, `--surface` at `:root`; `styles.css` uses
`--border-color`, `--text-secondary`, `--text-primary`, `--bg-card`. Any rule
that moves between them is **retyped against the destination's tokens**, never
linked and never copied verbatim.

### 4.3 Activity is not a separate panel

`js/admin-credits.js` is one 712-line IIFE exposing one global
(`window.AdminCredits`) that owns three surfaces at once:

| Surface | Endpoints | Verbs |
|---|---|---|
| Grant Pool | `/api/admin/credits/grant-pool`, `…/grants/{op}` | GET, POST |
| Accounts table | `/api/admin/credits/users`, `/api/admin/users/{id}`, `…/accounts/{id}/reinstate` | GET, PATCH, POST |
| **Activity** (audit trail) | `/api/admin/credits/activity` | GET |

The Activity tab's markup (`app.html:2317-2333`) is `adminCreditsActivityBody`
and `adminCreditsActivityMoreBtn`, both driven by that module. "Port Activity,
defer Account Management" is therefore not a cut along a module boundary, and
this design does not attempt one.

### 4.4 Providers already fits `/admin`'s conventions

`js/admin-model-providers.js` (260 lines) is already an IIFE exposing one global,
already renders with `textContent`, and already builds its icons with
`createElementNS` + `<use>` rather than `innerHTML`. It depends on the host page
at exactly two seams:

| Line | Dependency |
|---|---|
| `:15` | `window.API.request(path, options)` — defined in `app.js` |
| `:252` | `window.getStoredAuthUser()` — defined in `app.js` |

It also calls `window.confirm` at `:130` before revoking a platform key.

### 4.5 The guards PR C shipped

`dashboard/backend/tests/test_admin_page_modules.py` concatenates every module
`admin.html` loads into one string `ALL` and asserts over it:

```python
def test_every_request_is_a_credentialed_get_made_by_the_shell():
    assert set(re.findall(r"method:\s*'(\w+)'", ALL)) == {"GET"}
    for verb in ("POST", "PATCH", "PUT", "DELETE"):
        assert f"method: '{verb}'" not in ALL

def test_prohibited_field_names_and_local_storage_are_absent():
    for prohibited in ("api_key", ..., "credential_ciphertext", ...):
        assert prohibited not in ALL
```

`admin-model-providers.js` does `PUT` (`:176`, `:199`), `POST` (`:215`) and
`DELETE` (`:229`), and contains the literal `api_key` once. Dropping it into the
module set fails both tests.

## 5. Decisions

**N1 — Port the visual idiom, not the ARIA semantics.** The rail's *appearance*
moves; its `role="tab"` / `aria-selected` markup does not. The old console swaps
panels in place, where a tablist is correct. `/admin` navigates by hash and
`#users/{id}` is a real, linkable, back-button-able location, so its rail entries
stay `<a href="#route">` and carry `aria-current="page"`. *Why:* `role="tab"` on
a control that changes the URL misreports the interaction to assistive
technology, and anchors give middle-click, copy-link and keyboard handling with
no JavaScript. Copying the markup wholesale because it is the thing being
imitated is the mistake this decision exists to prevent.

**N2 — Providers ports; Account Management and Activity do not.** *Why:* §4.3.
Splitting a 712-line module mid-file to move one of its three surfaces would
leave the old console running a half-gutted file for a release, for no gain the
user can see. PR2 moves `admin-credits.js` whole and deletes `/app?view=admin`.

**N3 — The read-only property is kept and its subject is narrowed.** The
analytics modules stay pinned GET-only and free of credential field names. Write
modules are enumerated **by name with their exact permitted verb set**, and gain
a new pin that no credential-bearing field reaches a DOM sink. *Why:* the `==
{"GET"}` assertion over every module concatenated was true of the page **as
built** — a pure reader cannot mutate and cannot leak a secret. A write surface
ends that, and the only two honest responses are to narrow the subject or to
abandon the property. Deleting the verb list and dropping `api_key` from the
prohibited names would leave two tests that run, pass, and protect nothing —
the failure mode `CLAUDE.md` names "fail-closed is not fail-visible". An exact
verb set rather than a subset means adding a verb is a test failure rather than
a silent pass.

**N4 — `AdminShell` gains a second, separately named request function.**
`request(path)` stays hardcoded `method: 'GET'`. `write(path, {method, body})`
is the only write path on the page. *Why:* N3 needs a boundary a test can see.
Adding an options bag to `request()` would make "does this module write?"
un-greppable, which is the property N3 is asserting. Two functions with
different names make it a one-line assertion.

**N5 — `/app?view=admin&adminTab=providers` redirects to `/admin#providers`.**
It does not silently fall back to the accounts tab. *Why:* the URL is in
browser histories and possibly in someone's bookmarks. `admin-tabs.js` already
normalises unknown tabs to the default, so the fallback would be silent and
would land the user on a different page than the one they asked for. The
`/admin-analytics` → `/admin` 308 is the precedent for paying this courtesy.

**N6 — The old console's "Users" tab is relabelled "Account Management" in this
PR**, even though its port is PR2. *Why:* the two surfaces coexist for one
release and must not disagree about what the destination is called. The rename
is label-only; `data-admin-tab="users"`, the panel id and the query-parameter
value are unchanged, so no URL breaks and `admin-tabs.js` needs no new
normalisation.

**N7 — The account menu omits an "Admin" item and carries no ticker.** *Why:*
the Admin item would link to the page it is on. The ticker is a product-shell
element with a live market feed; an admin console has no use for it and it would
drag another module and its polling into a page whose whole point is that it
loads five files and no framework.

**N8 — `gate()` retains the user object it already fetches.** It currently
discards it, setting only `state.admin = true` (`admin-shell.js:311-325`). The
account menu needs the display name and email, and the page is already paying
for the request. *Why recorded:* so that the next reader does not add a second
`/api/auth/me` call for the menu.

## 6. Target — the rail

`admin.html`'s `<aside>` becomes:

```
┌─ Administration ──────────────┐
│ ▐ [chart]    Analytics     ⌄  │   .admin-tab, aria-expanded
│                Overview       │   .analytics-subnav, 13px
│                User sources   │
│                Retention      │
│                Credits & revenue │
│                User lifecycle │
│                System health  │
│                Users          │
│   [users]    Account Management │  → /app?view=admin&adminTab=users
│   [network]  Providers        │   → #providers   (ported)
│   [activity] Activity         │   → /app?view=admin&adminTab=activity
└───────────────────────────────┘
```

- Entries are `<a>` with `aria-current="page"` on the active one (N1). The
  Analytics parent keeps its existing `aria-expanded` disclosure behaviour,
  which `admin-shell.js:bindControls` already wires.
- Icons come from a new `<svg hidden>` sprite block in `admin.html`, holding
  `icon-chart`, `icon-users`, `icon-network`, `icon-activity` and `icon-refresh`
  copied from `app.html:186-193`. All five already exist; none need drawing.
  The sprite is markup, not script, so the page's "no inline scripts" guard
  (D6) is unaffected.
- CSS is retyped into `admin.css` against its tokens (§4.2): `.admin-rail` →
  the aside's card treatment, `.admin-tab` → the entry pill, `.admin-tab svg`,
  the `::before` active bar, hover/focus, and the `max-width: 700px` collapse to
  icon-only with visually-hidden labels.
- The Analytics subnav rises from 12px to the rail's 13px and keeps its
  indent rule.

## 7. Target — the account menu

A new header control in `admin.html`, owned by `admin-shell.js` (it already
holds the user after N8):

| Item | Action |
|---|---|
| identity — display name, email | not interactive |
| Account | `/app?view=account` |
| Credits & Billing | `/app?view=credits` |
| Log out | `AdminShell.write('/api/auth/logout', {method:'POST'})`, then `/app` |

The `.account-menu*` rules (12 selectors) are retyped into `admin.css`. The menu
renders only after the gate resolves; before that the header shows no chip,
never a placeholder name.

## 8. Target — the Providers port

| Item | From | To |
|---|---|---|
| Markup | `app.html:2266-2316` | `admin.html`, new `<section id="providers">` |
| Module | `js/admin-model-providers.js` | `js/admin-providers.js`, `window.AdminProviders` |
| Route | `?adminTab=providers` | `#providers`, added to `ROUTES` in `admin-shell.js:7` |
| Reads | `window.API.request(p)` | `AdminShell.request(p)` |
| Writes | `window.API.request(p, {method})` | `AdminShell.write(p, {method, body})` |
| Auth | `window.getStoredAuthUser()` | `AdminShell.user()` |

Removed from the old console in the same PR: the Providers rail button
(`app.html:2142`), the panel, the `AdminModelProviders` calls in `app.js`, and
`'providers'` from `ALLOWED_TABS` in `admin-tabs.js`. `admin-tabs.js` gains the
N5 redirect.

`window.confirm` at `:130` is kept for this PR. Replacing it with
`AdminShell.openDialog` is a real improvement and a real scope increase; it is
recorded in §11 rather than smuggled in.

### 8.1 The CSS the move drags along

The providers markup reuses six class families. Counted against `styles.css`:

| Family | Selectors |
|---|---|
| `.admin-provider-*` | 34 |
| `.admin-platform-key-*` | 11 |
| `.auth-btn*` | 30 |
| `.account-menu*` (§7) | 12 |
| `.admin-tab` / `.admin-rail` (§6) | 15 |
| `credits-*` shared vocabulary (`section-kicker`, `admin-badge`, `icon-btn`, `key-field`, `muted`, `status`) | 19 |
| `.control-select` | 5 |
| **Total** | **126** |

Roughly 350-450 lines into a file that is 274 lines today, so `admin.css` about
doubles. This is the standing tax on the two-page split and is paid once more in
PR2. It is recorded here because "just import `styles.css`" is the obvious
objection and the 09-15 design already considered and rejected it (§4.2).

## 9. The guard rescope

`test_admin_page_modules.py` splits its module set:

```python
READ_MODULES  = ("admin-shell.js", "admin-live.js", "admin-overview.js", "admin-users.js")
WRITE_MODULES = {"admin-providers.js": {"PUT", "POST", "DELETE"}}
```

| Test | Becomes |
|---|---|
| `test_every_request_is_a_credentialed_get_made_by_the_shell` | over `READ_MODULES` only; additionally asserts no read module references `AdminShell.write` |
| new `test_write_modules_declare_their_exact_verbs` | for each write module, the verb set found `==` its declared set — exact, so a new verb fails |
| new `test_only_the_shell_owns_fetch` | `fetch(` appears in `admin-shell.js` only; `request` stays GET-hardcoded |
| `test_prohibited_field_names_and_local_storage_are_absent` | unchanged over `READ_MODULES`; over write modules the credential names are permitted **as request-body keys only** |
| new `test_no_credential_field_reaches_a_dom_sink` | in write modules, no line matching a credential identifier (`api_key`, `secret`, `credential`, `token`, `password`) also assigns to `textContent` / `innerHTML` / `.value`, or is an argument to `setAttribute` / `append*` / `createTextNode` |
| `test_each_module_is_an_iife_exposing_exactly_one_global` | extended to the new module |
| `test_every_module_admin_html_loads_exists_and_nothing_else_is_loaded` | extended; `?v=` pins bumped |

`localStorage` / `sessionStorage` stay banned across **both** sets.

## 10. Amendments to the 2026-09-15 design

To be applied to `2026-09-15-admin-layer-redesign-design.md` in this PR, each
marked `(Amended 2026-09-17)` in place rather than rewritten, so the original
reasoning stays legible:

1. **§7.2, the module table** — gains `js/admin-providers.js` / `AdminProviders`.
   The `admin-users.js` row's parenthetical about account management not being
   ported stands; it is still true.
2. **§7.5** — "The old console keeps **Users**, **Providers** and **Activity**"
   becomes Users (relabelled Account Management) and Activity. The sentence
   "The new page's aside links to `…=users`, `…=providers` and `…=activity`
   until the follow-up port" loses `providers`.
3. **§13, the PR C row's "Must not" column** — currently reads "Port Users,
   Providers or Activity." A reader hitting that line after this PR would
   conclude the port was a violation. It gains a note that the prohibition
   bound PR C only, and that Providers ported in the 09-17 consolidation.
4. **§14, follow-ups** — "Providers and Activity absorption into `/admin` (D5)"
   becomes Activity and Account Management only.

D5 itself needs no amendment: staged absorption with full absorption as the
committed end state is exactly what this is.

## 11. Follow-ups

- **PR2: Account Management and Activity.** `admin-credits.js` ported whole,
  `/app?view=admin` deleted, `adminView` removed from `app.html`, the ~475-line
  admin block removed from `app.js` (`:4065-4540`), `admin-tabs.js` deleted.
  This is the end state D5 committed to.
- **`window.confirm` → `AdminShell.openDialog`** in the providers module
  (§8). The shell already owns a dialog with return-focus; the native
  confirm is the last un-styled, un-testable interaction on the page.
- **A back-link from `/app?view=admin` to `/admin`** is deliberately *not*
  added: PR2 deletes that view, and a link added now would be deleted before
  anyone relied on it. Named so its absence reads as a decision.
- **The `credits-*` vocabulary is now duplicated** across `styles.css` and
  `admin.css`. That is accepted for two releases. When PR2 lands and
  `/app?view=admin` is gone, the `admin-*` and `credits-admin-*` families in
  `styles.css` become dead and should be deleted in that PR, not left to rot.

## 12. Testing contract

- `test_admin_providers_frontend.py` — new, node-driven, lifting renderers with
  `fn_body` against fixtures, in the idiom of the other five admin frontend test
  modules.
- `test_admin_page_shell.py` — rail source-shape guards: every top-level rail
  entry has an icon and a label; the sprite defines every symbol the rail
  and the providers panel reference; the account menu exists and carries no
  hard-coded identity text.
- `test_admin_tabs_redirect.py` — rewritten for N5: `?adminTab=providers`
  navigates to `/admin#providers`; `users` and `activity` still do not navigate.
  The existing five-way parametrize at `:84` gains the new expectation rather
  than being replaced.
- `test_app_composition.py` — unchanged route set; `/admin` and `/admin.css`
  already registered.
- Cache-busters bump `?v=1` → `?v=2` on every changed file. **Project memory
  records that these pins span five test files, not the one the docs name** —
  all five are updated in the same commit.
- Full backend suite green before the PR opens.

---

*Design approved in session 2026-09-17. Implementation follows via
`superpowers:writing-plans`.*
