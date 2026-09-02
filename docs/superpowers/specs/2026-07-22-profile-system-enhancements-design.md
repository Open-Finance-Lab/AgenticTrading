# Profile System Enhancements — Design

**Date:** 2026-07-22
**Status:** Approved (brainstorming session with Felix, 2026-07-22)
**Scope:** Dashboard user accounts — header profile dropdown with logout, password change,
password strength policy, avatar upload (Phase 1); password reset via email (Phase 2).

## Context

The dashboard has working email/password auth (`dashboard/backend/users.py` +
`users_postgres.py`, bearer tokens in localStorage, bcrypt hashing, SQLite/Postgres twin
stores selected by `USERS_DATABASE_URL`) but almost no account-management surface:

- `POST /api/auth/logout` exists, and a logout button exists — but only on the Account
  page (`app.html` `#authLogoutBtn`). The header identity button (`#authAccountBtn`)
  just navigates to the Account page.
- Password validation is length-only (min 8, max 128, `api/auth.py`).
- No change-password, no password reset, no avatar. No email-sending infrastructure of
  any kind exists in the repo.
- Prod runs on disk-less Render free tier; the durable user store in prod is Neon
  Postgres via `USERS_DATABASE_URL` (account-persistence fix, PR #134).

## Phasing

- **Phase 1 (one PR, no new infrastructure):** header profile dropdown + logout,
  password change, password strength policy, avatar upload.
- **Phase 2 (separate PR, requires email provider signup + Render env vars):**
  password reset via email. Fully designed here; ships when the operator creates the
  provider account.

Existing accounts are never forced to change passwords; the policy applies to new
passwords only (signup, change, reset).

## Phase 1

### 1. Header profile dropdown (frontend only)

Replace `#authAccountBtn`'s navigate-on-click with a dropdown toggle:

- **Trigger:** avatar thumbnail (initials circle fallback) + display name + caret.
- **Menu:** non-interactive "signed in as" row (display name + email), then
  **Account** (navigates to the existing Account page) and **Log out**.
- **Behavior:** click toggles; outside-click and `Escape` close; `aria-expanded` /
  `aria-haspopup` on the trigger. Log out calls the existing `logoutUser()` in
  `app.js` (POSTs `/api/auth/logout`, clears `auth-token`/`auth-user` from
  localStorage, redirects home). The Account-page logout button remains.
- **Implementation:** reuse the `.agent-menu-dropdown` pattern in `styles.css`
  (absolute-positioned panel anchored to the trigger, right-aligned). No new
  dependencies; vanilla JS like the rest of `app.js`.

### 2. Password change

**Backend — `POST /api/auth/change-password`** (Bearer auth required):

```json
{ "current_password": "...", "new_password": "..." }
```

- 401 if the token is missing/invalid (existing `get_current_user` dependency).
- 400 `{"detail": "Current password is incorrect."}` if verification fails.
- 400 with the violation list if the new password fails the strength policy (§3).
- On success: bcrypt-rehash (this also silently upgrades legacy PBKDF2 hashes),
  update `password_hash`, then **delete all the user's other sessions** via a new
  store method `delete_other_sessions(user_id, keep_token)` implemented in **both**
  `UserStore` (SQLite) and `PostgresUserStore`. The current session stays valid.
  Returns `{"status": "ok"}`.

**Frontend:** "Change password" section on the Account page — current / new /
confirm fields, inline errors, success confirmation. Confirm-mismatch is caught
client-side.

### 3. Password strength policy (shared validator)

New module `dashboard/backend/password_policy.py`:

```python
def validate_new_password(password: str, email: str) -> list[str]:
    """Return human-readable violations; empty list means acceptable."""
```

Rules (NIST 800-63B style — no character-class composition rules):

1. Length ≥ 8 and ≤ 128 (existing bounds, now centralized).
2. Not in the vendored common-password blocklist:
   `dashboard/backend/common_passwords.txt` — the first 1,000 entries of length
   ≥ 8 from SecLists' `10-million-password-list-top-10000` (shorter entries are
   already dead under rule 1, so they are filtered out before taking 1,000).
   One per line, compared case-insensitively, loaded once at import; provenance
   noted in a header comment; the loader skips `#` comment lines and blanks.
3. Must not contain the email local-part (the part before `@`) when that part is
   ≥ 3 characters, compared case-insensitively.

Wired into **signup** and **change-password** (and later reset). Violations return
as HTTP 400 with the list joined into `detail`.

**Frontend feedback:** mirror rules 1 and 3 client-side for live feedback as the
user types; blocklist violations (rule 2) surface from the server response on
submit — this avoids shipping the password list to the browser.

### 4. Avatar

**Storage:** new nullable `avatar` TEXT column on `users` holding a full data URI
(`data:image/jpeg;base64,...`). Added by lazy migration in both stores, mirroring
the `discord_user_id` pattern (SQLite: `ALTER TABLE` guarded by `PRAGMA
table_info`; Postgres: `ADD COLUMN IF NOT EXISTS`). No object storage, no image
library, no disk — the value rides the existing durable Postgres path in prod.

**Client side (does the heavy lifting):** file picker on the Account page →
reject inputs > 10 MB → canvas downscale to fit 256×256 → JPEG at ~0.85 quality →
data URI. If the result exceeds 100 KB decoded, retry once at lower quality; if
still over, show an error.

**Server side (re-validates everything — never trust the client):**

- `PUT /api/auth/avatar` (Bearer): body `{"avatar": "<data-uri>"}`.
  Validation: data-URI prefix with mime in {`image/jpeg`, `image/png`,
  `image/webp`}; base64 decodes cleanly; **decoded bytes' magic numbers match the
  declared mime** (JPEG `FF D8 FF`, PNG `89 50 4E 47`, WebP `RIFF....WEBP` —
  stdlib byte checks, no Pillow); decoded size ≤ 100 KB. Returns the updated user.
- `DELETE /api/auth/avatar` (Bearer): sets the column NULL. Returns the updated user.
- `avatar` is included in the user payload of `/api/auth/me`, login, and signup
  responses so the header/dropdown can render it.

**Display:** header trigger + dropdown show the avatar (or initials circle);
Account page shows a larger preview with upload/remove controls.

**CSP:** the implementation must verify `CSPHeaderMiddleware`'s `img-src` allows
`data:` URIs and add it if absent — otherwise avatars silently won't render.

## Phase 2 — Password reset via email

> **Superseded (2026-09-01):** this section predates the Brevo sender and the
> `verification_codes.py` extraction. The shipped design — emailed 6-character
> code via Brevo, not a Resend link — is
> `2026-09-01-password-reset-design.md`. Kept for history; do not implement
> from this section.

**Provider: Resend**, called via plain HTTPS POST (`https://api.resend.com/emails`,
no SDK dependency). Config via env vars set in the Render dashboard:

- `RESEND_API_KEY` — API key.
- `RESET_EMAIL_FROM` — sender address (Resend-verified domain or their onboarding
  sender during testing).
- `PUBLIC_APP_URL` (already read by the app) — base for reset links.

**Fail-visible, not fail-silent:** if config is missing, `forgot-password` still
returns the generic success response (no enumeration signal) but emits
`print("ERROR: password reset requested but RESEND_API_KEY unset — email not sent")`
— `print()`, not `logging`, because logger output is invisible under the deployed
config.

**Schema:** new `password_reset_tokens` table in both stores:

| column | type | notes |
|---|---|---|
| `token_hash` | TEXT PRIMARY KEY | SHA-256 hex of the raw token; raw token never stored |
| `user_id` | INTEGER, FK → users.id, CASCADE | |
| `created_at` | timestamp | store-native format, matching each store's conventions |
| `expires_at` | timestamp | created_at + 30 minutes |
| `used_at` | timestamp, nullable | set on successful reset; non-null ⇒ unusable |

**Endpoints:**

- `POST /api/auth/forgot-password` `{"email": "..."}` → always `{"status": "ok"}`
  whether or not the account exists (no account enumeration). If the account
  exists: enforce a per-email cooldown (one request per 5 minutes, checked against
  `created_at` of that user's latest token), generate `secrets.token_urlsafe(32)`,
  store its hash, email a link `{PUBLIC_APP_URL}/app?reset_token=<raw>`.
- `POST /api/auth/reset-password` `{"token": "...", "new_password": "..."}` →
  hash the token, look up an unused, unexpired row; 400 on failure. Validate the
  new password against the policy (§3), bcrypt-rehash, mark the token used, and
  **delete all of the user's sessions** (unlike change-password there is no
  session to keep). Returns `{"status": "ok"}`; the user signs in again.

**Frontend:** "Forgot password?" link in the login modal → email-entry state →
generic confirmation. `/app?reset_token=...` opens a new-password form (with the
same live policy feedback), then redirects to sign-in.

## Security summary

- Session invalidation: change-password keeps only the current session; reset
  keeps none. (Change-password revocation is **best-effort** — it runs as a
  separate transaction after the durable password write and is wrapped so a
  revocation failure is logged, not fatal; see the Phase 1 plan's Task 3.)
- No account enumeration from `forgot-password`; reset tokens stored hashed,
  30-minute expiry, single-use.
- Avatar: server-side mime allowlist + magic-number verification + 100 KB cap;
  rendered only via `<img src>` (data URIs carry no script context there).
- Strength policy server-side is authoritative; client-side mirroring is UX only.

## Testing

- **Unit:** `password_policy` (each rule, boundary lengths, blocklist hit,
  email-local-part hit, empty email edge).
- **API (`test_auth.py`):** change-password happy path; wrong current password;
  weak new password; other sessions invalidated while current survives; avatar
  PUT/DELETE happy paths; avatar rejections (bad mime, magic-number mismatch,
  oversize, malformed base64); avatar present in `/auth/me`.
- **Postgres twins:** mirror store-level coverage under `@pg_only` in
  `test_users_postgres.py` (new column migration on an existing table, new store
  methods). Destructive fixtures must call the localhost guard
  (`require_local_postgres_url`) per the repo convention.
- **Route-contract freeze:** add each new route tuple to `EXPECTED_FULL_CONTRACT`
  in `test_app_composition.py` (per-phase: Phase 1 adds `change-password` +
  avatar PUT/DELETE; Phase 2 adds `forgot-password` + `reset-password`).
- **Phase 2:** token expiry, single-use, cooldown, enumeration-safety (identical
  response for unknown email), unset-provider fail-visible path (assert on
  `capsys`, not `caplog`).
- Frontend has no JS test framework; verify the dropdown, password change, and
  avatar flows manually (Playwright walkthrough, as done for PR #165).

## Delivery & sequencing

- Branch: `feat/profile-system` (cut from `main` at #160). PR #165 (My Agents)
  is open and heavily touches `app.html`/`app.js`/`styles.css`; **rebase this
  branch onto `main` after #165 merges before starting frontend work**, or open
  Phase 1 as a draft until then.
- Phase 1 is one PR (spec + backend + frontend + tests). Phase 2 is a separate
  PR, opened only when the Resend account and Render env vars exist — its PR must
  state that gate in the first line of the body.
- **Seed-DB caution:** running the app or suite locally in SQLite mode lazily
  `ALTER`s the committed `dashboard/storage/data/backtest.db` (users live there
  locally). Do not commit those mutations; check `git status` on the seed DB
  before committing (WAL/ALTER trap).

## User-facing docs follow-ups (not in scope of the PRs; coordinate separately)

Adding these features makes the following docs stale or more incomplete:

- No account-management docs exist at all (Account page is undocumented) — needs
  a section covering sign-up/sign-in, profile, password change/reset, logout.
- `docs/source/lab/getting_started.rst` — never mentions user accounts.
- `docs/source/lab/external_agents.rst` — conflates user signup with agent
  registration; profile features widen the confusion.
- `docs/discord-bot-instructions.md` + `docs/architecture/discord-to-backtest.md`
  — "Sign in on the website" with no pointer to account management.
