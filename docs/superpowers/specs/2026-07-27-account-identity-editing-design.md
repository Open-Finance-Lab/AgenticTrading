# Account identity editing — display name, email change, logout placement

Status: design approved 2026-07-27. Supersedes nothing; extends the Phase 1 work in
`2026-07-22-profile-system-enhancements-design.md`.

Lets a signed-in user edit the two identity fields the dashboard has always shown but
never let them change — **display name** and **email address** — and moves the
account-page **log out** control to the bottom of the page in a destructive-red
treatment. Email change is protected by the current password *and* two 6-character
confirmation codes: one to the current address, one to the new address.

## Context

Profile Phase 1 (PR #171, merged `40ae414`) shipped the account dropdown, password
change, the NIST-style password policy, and avatar upload. It deliberately left the two
identity fields read-only, because email change needs an email provider and none exists.

Current state as of `origin/main` @ `116874e`:

- `users.display_name` is a `NOT NULL` column that already exists (`users.py:135–180`),
  written once at signup by `create_user` and never updated. There is **no**
  `update_display_name` and **no** `update_email` in either store twin.
- There is **no table for short-lived codes or tokens** of any kind. `auth_sessions`
  (7-day opaque `secrets.token_urlsafe(32)`) is the only token store.
- There is **no email-sending capability anywhere in the repo** — no provider client, no
  dependency, no env var. Issue #187 has been open on exactly this prerequisite since
  Phase 1.
- The account page (`app.html:1496–1541`) shows display name and email as read-only
  `.account-row`s, with the logout button third from top in an `.account-actions`
  wrapper, above the Profile-photo and Change-password sections.

## Decisions

**D1 — Email provider: Brevo, single verified sender.** Free tier is 300 emails/day and
is unambiguously permanent; SendGrid's once-permanent 100/day tier became a 60-day trial
in 2025 and would quietly expire. Brevo also supports verifying a single sender address,
which matters because `PUBLIC_APP_URL` is a `vercel.app` subdomain and there is no domain
under our control to authenticate. Called by plain HTTPS POST to
`https://api.brevo.com/v3/smtp/email`, no SDK — matching the Phase 2 spec's
no-new-dependency stance.

Two limitations accepted explicitly:

- When the configured sender is a free-mail address (gmail.com, yahoo.com), Brevo rewrites
  the visible From address to a compliant one, so recipients see a Brevo-owned sender
  rather than the literal configured address. A deliverability workaround on Brevo's side,
  not a bug in this design.
- More consequentially, an *unauthenticated* single sender gets materially worse inbox
  placement under the Gmail/Yahoo/Microsoft bulk-sender rules in force since 2024. Codes
  landing in spam is the likeliest first-day complaint, so the stage `old` and stage `new`
  UI copy both say to check the spam folder. This is a risk to observe on the first live
  send, not a quantified one.

Verifying a real domain later removes both and requires no code change.

**D2 — Two codes, not one.** The brief specified a code to the *original* address. That
alone is the right security control — it stops an attacker holding only a stolen password
— but it leaves an unrecoverable failure mode: a user who typos the new address ends up
with an account email they do not own, and because the *next* change would send its code
to that same wrong address, they can never correct it. There is no admin UI, so recovery
would be a manual Neon edit.

So the flow keeps the original-address code as the authorization step and adds a second
code to the new address as a reachability proof. The change commits only when both are
entered. Rejected alternative: a client-side "confirm new email" field, which does not
survive a user who makes the same typo twice, and copy-paste makes that likely.

**D3 — One stage-driven verify endpoint, not two.** The server already knows which stage
is outstanding; exposing separate `verify-current` and `confirm` endpoints would only give
the client a way to call the wrong one.

**D4 — Non-blocking HTTP.** Auth routes are `async def` (`auth.py:128` onward). The one
existing outbound-HTTP precedent, `api/discord_oauth.py:122,138`, uses a *synchronous*
`httpx.Client(timeout=20.0)` — but the enclosing `exchange_code_for_access_token` (`:112`)
and `fetch_discord_user` (`:137`) are plain `def`s, and the async route awaits them through
`asyncio.to_thread` (`auth.py:270–277`) precisely so the blocking call stays off the event
loop, with a comment saying so. That is the rule this decision follows, not a precedent to
avoid. Two ways to honour it — wrap a sync client in `asyncio.to_thread`, or use an async
client directly. `httpx==0.28.1` is already a pinned direct dependency
(`requirements.txt:21`), so the mail call uses `httpx.AsyncClient` awaited from the async
route: one less thread hop, and the sender is `async def` anyway. On a single free-tier
Render worker a provider request blocking the loop would freeze the entire API. Issue #202
tracks the routes where this rule *is* violated today; `discord_oauth` is not one of them,
and both stay out of scope.

**D5 — Fail-visible mail.** `send_email()` returns `bool`. An unconfigured provider or a
non-2xx response `print()`s an ERROR line and the route returns **503** — never a
`{"status": "ok"}` for a code the user will never receive. `print()`, not `logging`,
because logger output from `dashboard.backend.*` is invisible under the deployed uvicorn
config. This is CLAUDE.md's "fail-closed is not fail-visible" rule applied at the point
where it would otherwise be violated.

**D6 — Issue #185 is fixed, not worked around.** `api/auth.py:12` does
`from dashboard.backend.users import ... user_store ...`, capturing the singleton into the
router's namespace at import time, so `test_auth.py`'s fixture (`:23–28`) patches a store
the routes never read. Every auth-route test today runs against the process-wide store.
Email uniqueness tests are the most exposed case — a leaked shared store produces
cross-test 409s that look like real failures, or worse, passes that mean nothing. Fixed by
importing the module and reading `users.user_store` at call time.

The issue names a second import-time binding, `api/routers/discord.py:11`, and it is fixed
the same way in the same commit so #185 closes without a loose end. A third site,
`api/dependencies.py:28`, is an in-function deferred import that already resolves at call
time and needs no change. The correct two-binding test pattern already exists as
`pg_client` (`test_users_postgres.py:70–83`).

**One existing test's premise is the bug being fixed.**
`test_change_password_revocation_failure_still_succeeds` carries an in-file comment
(`test_auth.py:236–237`) reading *"the `client` fixture only reassigns
users_module.user_store; api/auth.py may still hold the original singleton binding"*. That
test must be re-read and re-run deliberately in this commit, not allowed to break or pass
incidentally.

The same discipline applies to the new mailer: routes call `email.sender.send_email(...)`
through the module, never a name bound at import, so there is exactly one place to patch
and patching it works.

**D7 — Changing the password cancels a pending email change.** Narrow but real: a user who
suspects compromise changes their password, and an attacker's in-flight email change
should die with it. `cancel_email_change(user_id)` is called next to the existing
`delete_other_sessions` in the change-password route, best-effort, so the policy lives
where the session-revocation policy already lives rather than inside a store method.

**D8 — No "your email was changed" notice.** With D2, an attacker must already control
the original inbox to complete a change, so a warning sent to that inbox reaches only the
attacker. A third email and third template for zero security value.

## Part A — display name

**Store, both twins.** `update_display_name(user_id, display_name) -> dict` next to
`set_avatar` (`users.py:316`, `users_postgres.py:210`): `UPDATE` then return
`public_user(...)`; `.strip()` the value exactly as `create_user` does (`users.py:193`);
raise `ValueError("user_not_found")` on a missing row.

**Route.** `PUT /api/auth/display-name`, body `{"display_name": "..."}`, response
`{"user": {...}}`. `Field(min_length=1, max_length=100)` matching `SignupRequest:38`.
Empty-after-strip is rejected with 400 — issue #167 is that exact bug on the agents
surface (whitespace-only `name` stored as an empty string) and it is not repeated here.

No password required: a display name is not an authentication factor, and no major
platform gates it behind one. No new column, no migration, no uniqueness constraint
(none exists today and adding one would break existing duplicate names).

## Part B — email change

### Schema

New table `email_change_requests` in both twins, created in the existing lazy
`_init_schema` blocks (`users.py:135–180`, `users_postgres.py:39–88`). One row is one
in-flight change; at most one active row per user, enforced by deleting any prior row for
that user before insert.

| column | SQLite | Postgres | notes |
|---|---|---|---|
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | `SERIAL PRIMARY KEY` | |
| `user_id` | `INTEGER NOT NULL` | `INTEGER NOT NULL` | FK → `users(id)` `ON DELETE CASCADE` |
| `new_email` | `TEXT NOT NULL` | `TEXT NOT NULL` | normalized `.strip().lower()` at write time |
| `stage` | `TEXT NOT NULL` | `TEXT NOT NULL` | `'old'` or `'new'` |
| `code_hash` | `TEXT NOT NULL` | `TEXT NOT NULL` | SHA-256 hex of the uppercased code |
| `attempts` | `INTEGER NOT NULL DEFAULT 0` | same | wrong-code counter |
| `created_at` | `TIMESTAMP` | `TEXT NOT NULL` (ISO) | store-native, matching each twin's convention |
| `expires_at` | `TIMESTAMP NOT NULL` | `TEXT NOT NULL` | `created_at` + 15 min |
| `used_at` | `TIMESTAMP` nullable | `TEXT` nullable | set on commit |
| `cancelled_at` | `TIMESTAMP` nullable | `TEXT` nullable | set on cancel (see "Cancel does not reset the cooldown" below) — never deleted, so `created_at` stays intact for the cooldown read |

Plus `CREATE INDEX ... ON email_change_requests(user_id)`.

Normalizing `new_email` to lowercase is **mandatory, not stylistic**: SQLite's
`users.email` is `UNIQUE COLLATE NOCASE` (`users.py:139`) but the Postgres twin is plain
`UNIQUE` (`users_postgres.py:43`). All case-insensitivity in prod is an artifact of callers
lowercasing first. A write path that forgets would let `A@x.com` and `a@x.com` coexist in
prod while being rejected locally — twin drift that a SQLite-only test run cannot see.

### Code format

Six characters from `23456789ABCDEFGHJKMNPQRSTUVWXYZ` — 31 symbols, with `0`, `O`, `1`,
`I` and `L` removed so a code cannot be misread off a phone screen (note: original text listed 32 but that was unsatisfiable against the test). 31⁶ ≈ 8.9 × 10⁸,
which is safe given a 5-attempt cap and a 15-minute expiry. Generated with
`secrets.choice`, compared case-insensitively (input uppercased before hashing).

Stored as SHA-256 hex, not bcrypt: the code is a short-lived high-entropy secret, bcrypt
would add cost to every attempt, and SHA-256 is the same choice the approved Phase 2
design made for reset tokens. The hash is not an offline-attack defense (10⁹ candidates
is trivially searchable, and an attacker with database read access could simply rewrite
the `users.email` row); it prevents a *casual* read — a log line, a backup, a support
query — from yielding a live code.

### Store methods, both twins

- `create_email_change_request(user_id, new_email, code_hash) -> dict` — deletes any
  prior row for the user, inserts at `stage='old'`.
- `get_active_email_change(user_id) -> dict | None` — `used_at IS NULL` and not expired.
- `advance_email_change(request_id, code_hash) -> dict` — sets `stage='new'`, replaces
  `code_hash`, resets `attempts`, refreshes `expires_at`.
- `record_email_change_attempt(request_id) -> int` — increments and returns `attempts`.
- `mark_email_change_used(request_id) -> None` — sets `used_at`; does not delete the row.
- `cancel_email_change(user_id) -> None` — sets `cancelled_at`; does not delete the row.
- `update_email(user_id, new_email) -> dict` — normalizes, updates, returns
  `public_user(...)`. Maps `sqlite3.IntegrityError` / `psycopg.errors.UniqueViolation` to
  `ValueError("email_already_registered")` exactly as `create_user` does
  (`users.py:197`, `users_postgres.py:104`).
- `last_email_change_request_at(user_id) -> str | None` — for the cooldown.

### Cancel does not reset the cooldown

`cancel_email_change` deactivates the row (`cancelled_at`) instead of deleting it, the same
move `mark_email_change_used` already makes for a *completed* change. Deleting would also
erase the row `last_email_change_request_at` reads, and `DELETE /api/auth/email-change`
requires only a valid session, not the password — so a caller who already knows the
account's password could loop request (send, password-gated) → cancel (wipe the cooldown
clock, session-only) → request again, with the cooldown never enforced. That mail-bombs the
original address and burns the platform's shared 300-emails/day Brevo quota in well under a
minute. The 5-wrong-attempts path in `POST /api/auth/email-change/verify` calls the same
`cancel_email_change`, so keeping the row closes that trigger too, with one change instead
of two.

`get_active_email_change` excludes a row once either `used_at` or `cancelled_at` is set, so
a cancelled (or completed) request cannot be resumed — only its timestamp survives, purely
to keep the cooldown clock honest.

### Rate limiting (revised 2026-07-28)

The 60-second cooldown above bounds one *cycle*, but each cycle can send two messages: one
to the account's own address, and — once the first code is verified — one to an address the
requester chose. One authenticated account could therefore still drain the shared
300-emails/day Brevo quota in about 2.5 hours, half of it aimed at a third party from our
sending domain. Two further limits close that, and encode the product policy that an email
address is not something to churn:

| Constant | Window | Keyed on | Purpose |
|---|---|---|---|
| `EMAIL_CHANGE_MIN_INTERVAL_DAYS = 7` | 7 days | last **completed** change (`used_at`) | Product policy. Also protects anything later bound to an account (entitlements, paid plans) from being shed or inherited by address-hopping. |
| `EMAIL_CHANGE_MAX_REQUESTS_PER_DAY = 3` | rolling 24 h | request `created_at` | Caps one account at 6 messages/day, so draining the shared quota needs ~50 accounts rather than one. |
| `EMAIL_CHANGE_COOLDOWN_SECONDS = 60` | 60 s | newest request | Unchanged. |

Keying the 7-day limit on a **completed** change rather than on a *request* is the whole
point: a 7-second-to-7-day widening of the existing cooldown would lock a user out for a
week because they mistyped the new address once. The daily cap is what bounds abuse, since
the abuse path (request → read own code → verify → mail a third party → cancel → repeat)
never completes.

**`email_change_requests` therefore becomes append-only.** `create_email_change_request`
supersedes prior in-flight rows (`cancelled_at`) instead of `DELETE`-ing them — a delete
would erase both the `used_at` the 7-day limit reads and the `created_at` rows the daily cap
counts, so the act of making a request would clear both limits. `cancel_email_change` is
scoped to still-active rows for the same reason: stamping `cancelled_at` over a completed
change would misrepresent it in what is now an audit trail.

Two store reads support this, in both twins:

- `last_email_change_completed_at(user_id) -> str | None` — ordered by `used_at`, not `id`,
  since a row created earlier can be completed later.
- `email_change_request_times_since(user_id, since) -> list[str]` — oldest first, so the
  caller can report *when* the rolling window frees rather than only that it is full.

`Retry-After` carries the wait that is actually left, not the window's width; at seven days
those differ enough that a client honouring the header would sit out the whole period over a
wait with minutes to run.

**Not closed by this:** the quota is shared across accounts, so ~50 accounts still drains it,
and signup is unlimited. Tracked separately.

### Routes

All under the existing `/api/auth` router; the pending change is modeled as a singleton
resource.

| Route | Body | Behavior |
|---|---|---|
| `POST /api/auth/email-change` | `{current_password, new_email}` | Verify password (400 on mismatch) → reject same-as-current (400) → reject already-registered (409) → the three rate limits (429, see "Rate limiting" above) → create request → mail code to the **original** address. 503 if the mail send fails. |
| `POST /api/auth/email-change/verify` | `{code}` | Stage-driven. Correct code at `stage='old'` → advance to `'new'`, mail a fresh code to the **new** address, return `{"stage": "new", "new_email": "..."}`. Correct code at `stage='new'` → commit, return `{"status": "ok", "user": {...}}`. Wrong code → increment attempts, 400; at 5 attempts the request is deleted and the response says to start over. No active request → 400. |
| `GET /api/auth/email-change` | – | `{"pending": bool, "stage": ..., "new_email": ..., "expires_at": ...}` so a page reload does not strand the user mid-flow. |
| `DELETE /api/auth/email-change` | – | Cancel — deactivates, does not delete (see above), so the 60-second cooldown still applies to the next request. Also serves as the resend path: cancel, then restart, which re-verifies the password. |

Request models reuse the existing `_normalize_email` (`auth.py:29`) as a
`@field_validator("email")`, matching `SignupRequest:41–44`.

The duplicate-email check at request time is an account-enumeration oracle, and that is
accepted: `POST /api/auth/signup` already returns 409 on a duplicate, unauthenticated and
unlimited, so this path is strictly narrower, not a new exposure. It runs **before** the
cooldown check (see the order below), so the cooldown does not bound it — what bounds it is
that this path additionally requires a valid session and the account's own password, which
signup does not. Failing early beats walking a user through two codes only to 409 at
commit. The commit-time check remains as the TOCTOU backstop.

Password verification runs **before** every rate-limit check, so a mistyped password does
not burn any allowance. Full order for `POST /api/auth/email-change`: password →
same-as-current → already-registered → 7-day interval → daily cap → 60-second cooldown →
generate code → send → persist. The three limits are ordered widest window first, so
whichever fires reports the longest wait that actually applies rather than the shortest.

### Send before persist

Both mail-sending steps **attempt the send first and write state only if it succeeded.**
Without this, D5's 503 leaves the user worse off than a plain failure:

- On `POST /api/auth/email-change`, a persisted-then-failed send would burn the 60-second
  cooldown for a code that does not exist.
- On the stage `old` → `new` transition, it is worse: the request would sit at stage `new`
  awaiting a code that was never delivered, while the original-address code that *did*
  arrive is no longer accepted. The user's only exit is Cancel, and nothing on screen would
  explain why.

So the verify endpoint generates the stage-2 code, sends it, and only then writes
`stage='new'` with the new `code_hash`. A failed send returns 503 with the request
untouched at stage `old` — the code the user already has stays valid and they can simply
resubmit it.

### On commit

`update_email(...)`, then `delete_other_sessions(user_id, keep_token=<caller's>)` — an
email change is an identity change, so other sessions end while the caller stays signed
in. Best-effort and non-fatal, mirroring change-password (`auth.py:183–200` — rationale
comment 183–190, `try`/`except`/`print` 191–199, `return {"status": "ok"}` at 200), which
already documents that the durable write and the revocation are separate transactions and
that a revocation failure is logged rather than fatal.

That existing handler prints **`WARNING:`**, not `ERROR:`, because by then the durable
write has already succeeded. The revocation print here mirrors `WARNING:` for the same
reason; D5's `ERROR:` is reserved for the mail failures, where the user genuinely gets
nothing. Do not level-shift either one.

### Mail module

New `dashboard/backend/infrastructure/email/sender.py` — a provider-neutral
`async def send_email(to: str, subject: str, text_body: str) -> bool` whose body talks to
Brevo. Sits under `infrastructure/` alongside `brokers/alpaca_paper.py`, the existing
isolated third-party HTTP adapter. Like `brokers/`, `llm/` and `market_data/`, the new
`email/` subpackage carries its own `__init__.py`.

Naming the package `email` does **not** shadow the standard library: Python 3 uses
absolute imports, so a bare `import email` anywhere still resolves to stdlib. Noted here
to pre-empt the review question.

Brevo contract (verified current, 2026): `POST https://api.brevo.com/v3/smtp/email`,
auth header `api-key`, body
`{"sender": {"name": ..., "email": ...}, "to": [{"email": ...}], "subject": ..., "textContent": ...}`.

Configuration, all read at call time so tests can set and unset freely:

- `BREVO_API_KEY` — API key; unset means the feature is off.
- `ACCOUNT_EMAIL_FROM` — the Brevo-verified sender address.
- `ACCOUNT_EMAIL_FROM_NAME` — optional, defaults to `Agentic Trading Lab`.

Returns `False` after `print("ERROR: ...")` when unconfigured, when the POST raises, or on
a non-2xx response (logging status and a truncated body). 10-second timeout. Never raises.

## Part C — logout placement

Pure markup and CSS; no JavaScript and no backend change.

The `.account-actions` wrapper (`app.html:1505–1507`) is cut from its current position —
third from the top, above Profile photo — and re-inserted as the last child of
`#accountSignedIn`, wrapped in its own `.account-section` so it picks up the section
divider (`border-top`) that separates it from Change password.

`id="authLogoutBtn"` is preserved verbatim, so the existing handler keeps working
untouched. There is exactly one literal reference in `app.js` —
`const logoutBtn = document.getElementById('authLogoutBtn');` (`:2517`) — and one use of
that local, `logoutBtn?.addEventListener('click', …)` (`:2556–2558`); both sit inside
`initAuthUI()` (`:2513–2647`).

The class changes from `auth-btn auth-btn-secondary` to `auth-btn auth-btn-danger`, a new
rule mirroring `.agent-delete-btn` (`styles.css:6999–7009`):

```css
.auth-btn-danger {
    border-color: rgba(248, 113, 113, 0.55);
    background: rgba(248, 113, 113, 0.1);
    color: #f87171;
}
.auth-btn-danger:hover {
    border-color: #f87171;
    background: rgba(248, 113, 113, 0.2);
    color: #fca5a5;
}
```

A new class rather than reusing `.agent-delete-btn`, which is agent-scoped; the codebase
already makes this exact "same look, own class" split between
`.account-menu-item--danger` and `.agent-menu-item--danger`. Note the repo carries **two**
reds: `#f87171` for destructive controls and `--danger-color: #ff4141` (`styles.css:31`)
for negative P&L. Logout takes the destructive-control red. `.auth-btn-danger` has zero
occurrences in `styles.css` today, so the name is free.

**Placement matters and is not optional.** The button keeps its `auth-btn` class, and
`styles.css` already defines a bare `.auth-btn:hover` (`:384–387`) setting
`border-color`/`color` to `--info-color`. `.auth-btn:hover` and `.auth-btn-danger:hover`
have *identical* specificity (0,2,0), as do `.auth-btn` (`:361–370`) and
`.auth-btn-danger` (0,1,0) — so source order alone decides the winner. The new rules must
be appended **after line 387**. Inserted anywhere earlier, the button silently reverts to
the info-blue hover and the red looks broken only on mouseover, which is exactly the kind
of defect a screenshot taken at rest would miss.

No `confirm()` — logout is not destructive and is one-click today.

**The header-dropdown logout (`#accountMenuLogoutBtn`, `app.html:180`) is unchanged.** The
brief targeted the account-page button; removing the dropdown item would also make
`docs/source/lab/accounts.rst:31–32` factually wrong.

## Frontend

Inside `#accountSignedIn`, final order:

```
account-row      Display name   →  #accountDisplayName    unchanged (summary)
account-row      Email          →  #accountEmail          unchanged (summary)
account-section  Display name   →  NEW   input + Save
account-section  Email address  →  NEW   two-step form
account-section  Profile photo  →  unchanged
account-section  Change password→  unchanged
account-section  Log out        →  MOVED here, red
```

The two read-only rows stay as the at-a-glance summary and keep working with the existing
`applyUpdatedUser` refresh path; the new sections are the editors. Both new sections follow
the Change-password block's structure — an `.auth-form` with `.auth-field` labels, a
`<p class="auth-error" hidden>`, a `<p class="account-success" hidden>`, and an
`auth-btn auth-btn-primary` submit.

Email section states, driven by `GET /api/auth/email-change` on render:

- **idle** — new email + current password + `Send code`.
- **stage `old`** — "We sent a 6-character code to `<current address>`. Check your spam
  folder if it doesn't arrive." + code input + `Verify` + `Cancel`.
- **stage `new`** — "Code sent to `<new address>`. Enter it to finish — check spam if it
  doesn't arrive." + code input + `Confirm` + `Cancel`.

The spam-folder line is not filler: per D1, an unauthenticated single sender has
materially degraded inbox placement, and a code that silently lands in spam is
indistinguishable to the user from a code that was never sent.

No address masking: the user owns both addresses, the current one is printed two rows
above, and they just typed the new one.

JS follows the existing shape — `AuthAPI.*` methods beside `changePassword`
(`app.js:2002–2007`, inside the `AuthAPI` object at `:1951–2023`), an
`initEmailChangeForm()` / `initDisplayNameForm()` pair modeled on `initChangePasswordForm()`
(`:2213–2257`), registered in the initializer block at the tail of `initAuthUI()`
(`:2639–2646`, which today calls `initChangePasswordForm(); initAvatarControls();
refreshAuthUser();`), with `applyUpdatedUser(data.user)` (`:2133–2137`) on success so the
summary rows and the header dropdown refresh together. `initAuthUI()` itself is invoked
once from `DOMContentLoaded` (`:2677`).

Cache-bust bumps: `styles.css?v=64` → `65` (`app.html:12`) and `app.js?v=47` → `48`
(`app.html:1555`). **Re-read both lines at branch time** rather than trusting these
numbers — they moved from 63/44 to 64/47 between this spec being drafted and PR #227
landing on `main` afterward, and they are a standing merge-conflict magnet across
concurrent PRs. Bump to one above whatever `main` actually carries; do not assume the
numbers above are still one below current `main` by the time you branch.

## Testing

- **Password policy / code helpers (unit).** Code alphabet and length; ambiguous
  characters absent; case-insensitive comparison; hash stability.
- **Store twins.** `update_display_name` (strip, not-found); `update_email` (uniqueness →
  `ValueError`, lowercase normalization); request-table lazy creation; expiry; attempt
  counter; cancel. Postgres twins under `@pg_only` (`test_users_postgres.py:24–27`), using
  the destructive-fixture localhost guard `require_local_postgres_url`, which is
  *imported* at `:20` and defined in `dashboard/backend/tests/_postgres_testing.py:21`.
- **API (`test_auth.py`).** Full happy path across both stages; wrong password; wrong
  code; attempt cap; expired code; cooldown → 429; **cooldown survives cancel-then-resend**
  (the DELETE route needs only a session, not the password — this is the regression test
  for the cancel/cooldown bypass); new email already registered at request
  (409) and at commit (409); same-as-current → 400; cancel; `GET` pending state;
  unauthenticated → 401 on every new route; other sessions revoked while the caller's
  survives; password change cancels a pending request (D7); **provider unconfigured → 503
  with the ERROR asserted on `capsys`, not `caplog`**. Copy the fixture pattern from
  `pg_client` (`test_users_postgres.py:70–83`), not from `client` (`test_auth.py:23–28`) —
  the latter is the #185 bug. `_signup_and_token` is at `:142` and
  `test_change_password_happy_path` (`:151–171`) is the closest template.
- **Mail sender.** `httpx.AsyncClient` monkeypatched inside the sender module; the
  coroutine driven by `asyncio.run()` so no `pytest-asyncio` dependency is added. Asserts
  request payload shape, the timeout, and that a non-2xx returns `False` after printing
  ERROR.
- **Route-contract freeze.** The five new route tuples go into `EXPECTED_FULL_CONTRACT`
  (`test_app_composition.py:53–166`; the existing auth entries — nine at this spec's
  `116874e` base, twelve as of PR #227's three `/api/auth/robinhood/*` additions — are
  `:68–79`; re-grep before editing, `main` may have moved further by execution time) **in
  the same commit** — otherwise `test_full_route_contract_unchanged` (`:237`) turns red on
  every open PR, not just this one. That set is the **only** frozen list in the suite that
  enumerates `/api/auth/*`: `test_router_move.py` holds nine `EXPECTED_*` sets but none is
  an auth set (its "auth" references are `protocol_auth`, the `/api/v1` agent-key path).
  One known edit, not an open-ended hunt.
- **Frontend.** The node-under-pytest brace-matched-extraction pattern
  (`test_frontend_portfolio_panel.py:35–52`, with the node-availability skip at `:30–32`
  and the `_run_node` subprocess helper at `:55–60`; `test_frontend_xss_guards.py:36` is
  the second exemplar) is the only harness available. At minimum: the logout button is the
  last child of `#accountSignedIn` and carries `auth-btn-danger`.

## Delivery

1. **Operator step, before merge.** Sender decided: `ACCOUNT_EMAIL_FROM=flymiss.privateserver@gmail.com`
   (a personal Gmail address, not a domain we control — accepting the D1 free-mail-sender
   caveats: Brevo rewrites the visible From and inbox placement is worse than an
   authenticated domain). `BREVO_API_KEY` is issued and set locally in the (gitignored)
   `dashboard/.env` for dev/internal testing; it still needs to be set in the **Render
   dashboard** before this can work in prod — `render.yaml` is documentation, not the
   mechanism. Unset in prod means email change 503s while display name and logout work
   normally.
2. Branch `feat/account-identity-editing`, worktree `/mnt/d/github/atl-wt-account`, cut
   from `origin/main` @ `116874e`. `origin/main` has since advanced (PR #227, #234), both
   touching the exact files Parts A–C edit here — rebase onto current `origin/main` before
   opening the PR; see the plan's Task 12, Step 1.
3. `main` has no branch protection and the observed norm is that any collaborator merges
   any open PR at any moment. The PR therefore **opens as a draft** with
   `DO NOT MERGE until BREVO_API_KEY / ACCOUNT_EMAIL_FROM are set in Render` as the
   **first line of the body**. A comment is not a gate.
4. **Seed-DB caution.** Running the app locally in SQLite mode lazily creates
   `email_change_requests` inside the committed `dashboard/storage/data/backtest.db`, and
   the write can hide in the untracked `-wal` sidecar until a checkpoint folds it into the
   binary. Check `git status` on that file before every commit.
5. **`@pg_only` fails open.** `TEST_POSTGRES_URL` is unset locally and no local Postgres
   is available, so the Postgres tier silently skips. Verify it by grepping the CI job log,
   not by a green local run.

## Issues

An exhaustive search of open **and** closed issues for "display name", "change email",
"email address", "logout", "account settings", "username", "edit profile", "sign out
button" and "account menu" found **no** existing issue covering any of parts A, B or C.
This work closes exactly one issue, and it is an incidental one.

- **Closes #185** — fixed per D6 at both import-time binding sites, not worked around.
- **Does not close #187** (password reset via email). This dissolves its stated blocker:
  the mail module and provider credentials will exist. Comment there at merge rather than
  closing — and flag that D1 selects **Brevo**, not the Resend that #187 and the Phase 2
  spec both name, so its env-var list (`RESEND_API_KEY`, `RESET_EMAIL_FROM`) needs
  updating before anyone builds against it.
- **References #167** (whitespace-only PATCH stores an empty string) as precedent only. It
  is confined to the agents router and does not close.
- **References #202** (event-loop blocking) as the rule D4 obeys. Does not close, and per
  D4 `discord_oauth` is not one of its instances.
- **#172** (six Phase 1 polish items) is untouched. All six were re-read: the closest,
  account-menu ARIA/focus, is the same surface but a different concern.

## User-facing docs follow-ups (coordinate separately, not part of the PR)

- `docs/source/lab/accounts.rst:35–46` — "Manage your profile" documents only password
  change and profile photo. Becomes **incomplete**, not wrong; needs display-name and
  email-change paragraphs, including where each code is sent.
- `docs/source/lab/accounts.rst:20` — describes display name as something set at signup and
  nowhere says it is editable later.
- `docs/source/lab/accounts.rst:31–32` — stays accurate, since the dropdown logout is
  unchanged. Noted so a later reader does not "fix" it.
- Pre-existing and unrelated: the ReadTheDocs protocol page and the PyPI README still
  state a 30-second decision deadline; it is 60 seconds.
