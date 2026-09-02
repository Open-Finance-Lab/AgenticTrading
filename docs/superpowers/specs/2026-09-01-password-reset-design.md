# Password Reset via Email Code — Design

**Status:** Approved (design approved in-session 2026-09-01, then adversarially
reviewed against source; supersedes the "Phase 2 — Password reset via email"
section of `2026-07-22-profile-system-enhancements-design.md`)

**Date:** 2026-09-01

**Closes:** #187

## Context

A user who forgets their password has no self-service path today — the only
credential flows are the authenticated `change-password` and email-change
routes. Issue #187 tracked this as "blocked on an email provider"; that blocker
dissolved when the Brevo sender (`infrastructure/email/sender.py`,
`BREVO_API_KEY`/`ACCOUNT_EMAIL_FROM`) shipped for the email-change flow. The
July spec's Phase 2 section designed this feature around Resend and an emailed
link; both choices predate what the codebase actually grew:

- `verification_codes.py`'s own docstring names "the password-reset flow
  (#187)" as its intended second consumer — the 6-char-code machinery was
  extracted for exactly this.
- The email-change flow battle-tested the request → code → confirm UX end to
  end (attempt caps, TTLs, cooldowns, send-before-persist, spam-folder copy).

This design therefore uses an **emailed 6-character code entered in the same
modal**, not a link. No new env vars; the flow reuses the Brevo credentials the
email-change flow already requires.

## Goals

1. A signed-out user can reset a forgotten password from the login modal:
   enter email → receive a 6-char code → enter code + new password → sign in.
2. No account enumeration through responses: status and body are identical
   whether or not the submitted email has an account, and `forgot-password`'s
   **timing** is uniform too (all account-dependent work happens after the
   response). `reset-password`'s residual timing difference (one extra store
   read/write on the real-account path) is accepted — account existence
   already leaks through signup's documented 409, so this discloses nothing
   new; see Accepted gaps.
3. Codes are single-use and expire in 15 minutes; the wrong-attempt cap (5)
   and single-use property hold **under concurrency** (atomic SQL, not
   read-then-write — see Storage).
4. A completed reset durably sets the new password, then best-effort revokes
   **all** of the account's sessions and cancels any pending email-change
   (compromise-response, rule D7). Best-effort matches the existing
   change-password/email-change convention: revocation failure is a `WARNING`
   print, never a 500 after the durable write.
5. Misconfigured email (Brevo vars unset) fails visibly (503 + `ERROR` print),
   never as a silent success. A server-wide send budget bounds anonymous
   drain of the shared Brevo quota.
6. Full deterministic test coverage, including the Postgres twin.

## Non-goals

- No link-based reset (`reset_token` URLs, `PUBLIC_APP_URL` dependency) — the
  July spec's shape is explicitly superseded.
- No account lockout, CAPTCHA, or SMS second factor.
- No "resend code" affordance in v1: the way back is "Back to sign in" →
  "Forgot password?" again, and the cooldown governs how soon that produces a
  new code. (A resend button is a UX nicety the email-change flow also lives
  without.)
- No duplication of the reset UI into the landing page's hand-inlined auth
  modal — the landing link deep-links to `/app` (see Frontend).
- No user-facing docs edits in the implementation PR
  (`docs/source/lab/accounts.rst` follow-up is coordinated separately; see
  the follow-ups section).
- Login does **not** cancel a pending reset request (industry norm: a reset
  request must not lock out or be voided by normal use). Only change-password,
  email-change commit, and the reset flow itself cancel it.

## Backend

Two routes in `dashboard/backend/api/auth.py`, mounted like every other auth
route via the existing `router` → `api/router.py` → `app.py` chain.

Request models follow the existing auth-model conventions (explicit `Field`
bounds + the `_normalize_email` validator, as on `LoginRequest`):

```python
class ForgotPasswordRequest(BaseModel):
    email: str            # _normalize_email; same bounds as LoginRequest.email

class ResetPasswordRequest(BaseModel):
    email: str            # _normalize_email; same bounds as LoginRequest.email
    code: str             # Field(min_length=1, max_length=16); hash_code trims/upcases
    new_password: str     # same bounds as ChangePasswordRequest.new_password
```

A syntactically malformed email 422s at the model layer (both routes). This is
a deliberate, accepted exception to response uniformity: the 422 is
shape-keyed, not account-keyed — a malformed address cannot have an account,
so it discloses nothing.

### `POST /api/auth/forgot-password` — request a code

Handler is `async def`, and answers in this order:

1. **In-process rate limits** (429 + `Retry-After` via `rate_limited_error`):
   a per-client limiter and a per-submitted-email limiter, both charged with
   `.allow()` on every accepted request (there is no success/failure split to
   exempt). Both keys are existence-blind — the email key is the *typed*
   normalized address, so a 429 reveals nothing about accounts. Built with
   the existing `_build_limiter` pattern:
   - `_FORGOT_IP_LIMITER = _build_limiter("AUTH_FORGOT_IP", 30, 3600)`
   - `_FORGOT_EMAIL_LIMITER = _build_limiter("AUTH_FORGOT_EMAIL", 5, 3600)`
   429 detail (both): `"Too many reset requests. Please wait before trying
   again."`
2. **`email_configured()` check → 503** with the email-change flow's copy
   ("Could not send the confirmation email. Please try again later."), plus
   its own operator line — `email_configured()` has no side effects, so the
   route must print it:
   `print("ERROR: password reset requested but BREVO_API_KEY / ACCOUNT_EMAIL_FROM are not set -- returning 503")`.
   This fires *before any account lookup*, for every caller identically — it
   is config-shaped information, not account-shaped, and it keeps a
   Brevo-unconfigured deploy fail-visible instead of silently 200ing.
   (Deliberate deviation from the July spec's silent-200-with-print, which
   predates the repo's "answer 503 rather than half-working" convention.)
3. **Immediate generic `200 {"status": "ok"}`** for everyone else. All real
   work happens in a FastAPI `BackgroundTasks` task (first use of
   `BackgroundTasks` in this repo), so response status *and latency* are
   uniform — the login path's `verify_password_for_account` lesson (a
   work-only-for-real-accounts branch is a timing oracle even when the body
   is uniform) applied to the email send.

The background task (`async`; store calls wrapped in `asyncio.to_thread`,
`await email_sender.send_email(...)` directly):

1. `get_user_by_email` — unknown → print `auth.reset_skipped reason=unknown
   domain=...`, done.
2. **DB-backed cooldown and daily cap**, enforced silently (still 200 —
   an inline 429 keyed on account state would be an enumeration oracle):
   skip (with `reason=cooldown` / `reason=daily_cap`) if the account's latest
   request row is younger than `PASSWORD_RESET_COOLDOWN_SECONDS`, or if
   `PASSWORD_RESET_MAX_REQUESTS_PER_DAY` rows exist in the trailing 24h.
   Reads are **status-blind** (cancelled/used rows still count), exactly like
   `last_email_change_request_at` — so a cross-flow cancellation does *not*
   reset the cooldown clock, and cancelling can never be used to mint codes
   faster. These are the durable backstop behind the in-process limiters
   (which reset on redeploy).
3. **Global send budget**: `_FORGOT_GLOBAL_LIMITER =
   _build_limiter("AUTH_FORGOT_GLOBAL", 10, 3600)`, charged via `.allow()`
   with a single fixed key immediately before sending. Over budget → skip
   (`reason=global_cap`, printed as a `WARNING`), still nothing to the
   caller. This bounds the one genuinely new attack surface this feature
   opens: an unauthenticated caller iterating *known account* addresses to
   drain the shared Brevo quota and take down all transactional email
   (email-change never had this exposure — it sits behind a session + the
   account's own password). 10/hour caps worst-case drain at ~240/day, under
   Brevo's free-tier ~300/day, while being far above legitimate reset volume
   at current account counts; env-tunable like every limiter here.
4. `generate_code()` + `send_email(...)`. **Send before persist** (the
   email-change invariant): a failed send prints an `ERROR:` line and persists
   nothing, so the user's retry is not cooldown-blocked.
5. On send success: `create_password_reset_request(user_id, hash_code(code))`
   — atomically cancels any prior active row and inserts (one active request
   per account; see Storage). Print `auth.reset_requested domain=...`.

`TestClient` executes background tasks before the response returns to the test
caller (verified against starlette source), so tests stay deterministic.

### `POST /api/auth/reset-password` — redeem the code

Handler is a plain `def` (threadpool — it does store I/O and one bcrypt hash),
pinned in `BLOCKING_IO_HANDLERS`. Limiter mechanics mirror login's exactly
(`auth.py:478-501`): a loose per-client `.check()` before any work → 429; on
**every failure outcome below** (unknown email, no active row, wrong code),
charge `.record()` on the IP limiter and `.allow()` on the per-email limiter —
`.allow()` is what makes the per-email budget actually enforce (a bare
`.record()` never blocks), so a hammered address starts answering 429 on
subsequent attempts:

- `_RESET_IP_LIMITER = _build_limiter("AUTH_RESET_IP", 30, 900)`
- `_RESET_EMAIL_LIMITER = _build_limiter("AUTH_RESET_EMAIL", 10, 900)`
- 429 detail (both): `"Too many attempts. Please wait before trying again."`

Order:

1. Per-client `.check()` → 429.
2. Unknown email, or no active request row → **uniform generic 400**
   `RESET_FAILURE_DETAIL = "Invalid or expired code."` (one constant for all
   failure branches, mirroring `LOGIN_FAILURE_DETAIL`'s role). Expiry is not
   a separate branch: `get_active_password_reset` folds it into "no active
   row", exactly like `get_active_email_change` (the store, not the route,
   owns expiry).
3. Code compare: `hmac.compare_digest(hash_code(payload.code),
   row["code_hash"])` — mismatch → `record_password_reset_attempt(row_id)`,
   an **atomic conditional increment** whose return value tells the route the
   cap was hit (reaching `PASSWORD_RESET_MAX_ATTEMPTS` cancels the request —
   email-change precedent, but enforced in SQL so a concurrent burst cannot
   exceed the budget); same generic 400.
4. Only after the code passes: `validate_new_password(new_password, email)` —
   violations → 400 with the policy detail (the same
   `{"detail": {"code": ..., "violations": [...]}}` shape change-password
   uses). The request row is untouched, so the user resubmits the same
   still-valid code with a better password. (The policy 400 is
   distinguishable from the generic 400 only by a caller who already
   presented the correct code — i.e. who has already won; it leaks nothing.)
5. Commit: **`mark_password_reset_used(row_id)` first — an atomic
   compare-and-swap** (`... SET used_at = ? WHERE id = ? AND used_at IS NULL
   AND cancelled_at IS NULL`, acting on rowcount); a losing concurrent redeem
   gets the generic 400. Only the CAS winner proceeds to the durable
   `update_password(user_id, new_password)`. Consuming the code before the
   password write is the safe failure order: a crash between the two burns a
   code (user re-requests) instead of leaving a live code after a state
   change. Then **best-effort** (try/except + `WARNING` print, never a 500
   after the durable write): `delete_other_sessions(user_id,
   keep_token=None)` — *all* sessions, there is no session to keep — and
   `cancel_email_change(user_id)` (D7: a reset is plausibly a compromise
   response).
6. Print `auth.reset_completed domain=...`; return `{"status": "ok"}`. The
   user signs in with the new password (no auto-login: minting a session for
   an email-only proof would skip the fresh-password check login performs).

### Cross-flow cancellation (symmetry with D7)

- `change-password` (authenticated) additionally calls
  `cancel_password_reset(user_id)` — best-effort, same try/except pattern it
  already uses for `cancel_email_change`.
- `email-change/verify`'s final commit also calls
  `cancel_password_reset(user_id)` — a code mailed to the *old* address must
  not survive the address changing.

### Storage

New append-only table `password_reset_requests` in **both**
`users.py::UserStore` and `users_postgres.py::PostgresUserStore`, mirroring
`email_change_requests` (including its per-user index — add
`idx_password_reset_requests_user_id`):

| column | type | notes |
|---|---|---|
| `id` | INTEGER PK / SERIAL | |
| `user_id` | INTEGER NOT NULL, FK → users(id) ON DELETE CASCADE | declared-but-unenforced on SQLite, enforced on Postgres, like the existing tables |
| `code_hash` | TEXT NOT NULL | `verification_codes.hash_code` output; raw code never stored |
| `attempts` | INTEGER NOT NULL DEFAULT 0 | |
| `created_at` | timestamp | store-native format |
| `expires_at` | timestamp | `created_at` + `PASSWORD_RESET_TTL_MINUTES` |
| `used_at` | timestamp NULL | set on successful reset |
| `cancelled_at` | timestamp NULL | set by cancel paths; rows are never deleted |

Store methods, identical signatures on both twins (the `UserStore` /
`PostgresUserStore` pair is already registered in
`test_store_twin_parity.py`'s `_TWINS`, so signature and schema parity are
machine-checked — keep every DDL string a plain literal, no f-strings):

- `create_password_reset_request(user_id, code_hash)` — cancels any active
  row and inserts **in one transaction on one connection**, returns the new
  row. (Two racing creates must not leave the earlier-delivered code alive:
  last write wins cleanly, and the loser's code fails as "no active row".)
- `get_active_password_reset(user_id)` — latest row with `used_at IS NULL AND
  cancelled_at IS NULL`, returning `None` when there is no such row **or the
  row is expired** — expiry lives in the store, mirroring
  `get_active_email_change` (`users.py:1107-1123`); the route never reads
  `expires_at`.
- `record_password_reset_attempt(request_id)` — atomic conditional increment
  (`UPDATE ... SET attempts = attempts + 1 WHERE id = ? AND attempts < ?`
  style, cancelling on cap), returning enough for the route to distinguish
  "counted" from "cap reached". Mirror the email-change increment's exact
  decomposition where it is already atomic; upgrade it here regardless — the
  5-attempt cap is stated as a guarantee, so it must survive concurrency.
- `mark_password_reset_used(request_id)` — the CAS described in commit step 5,
  returning whether this call won.
- `cancel_password_reset(user_id)`
- `last_password_reset_request_at(user_id)` and
  `password_reset_request_times_since(user_id, since)` — the cooldown /
  daily-cap reads, mirroring `last_email_change_request_at` /
  `email_change_request_times_since` (`users.py:1205-1272`), status-blind.

The cooldown/daily-cap check-then-act in the background task is **not**
atomic and is accepted as racy: the inline per-email `.allow()` limiter is
the hard bound on issuance cadence (do not remove it on the theory that the
DB caps suffice — they are the durable backstop, not the enforcement).

### Constants (in `users.py`, beside the email-change block)

```python
PASSWORD_RESET_TTL_MINUTES = 15
PASSWORD_RESET_MAX_ATTEMPTS = 5
PASSWORD_RESET_COOLDOWN_SECONDS = 300   # July spec's one-request-per-5-minutes
PASSWORD_RESET_MAX_REQUESTS_PER_DAY = 5
```

### Email copy

Follow the `_email_change_*_body` f-string style; plain text.

- Subject: `Your Agentic Trading Lab password reset code`
- Body:

  ```
  Someone requested a password reset for the Agentic Trading Lab account
  belonging to this address.

  Your reset code: {code}

  Enter it on the password reset screen along with your new password. The
  code expires in {PASSWORD_RESET_TTL_MINUTES} minutes and can be used once.

  If you didn't request this, you can ignore this email — your password has
  not been changed.
  ```

### Logging

`print()`, never `logging`; never the raw address, only
`_email_domain(email)`. Events: `auth.reset_requested`, `auth.reset_skipped
reason=unknown|cooldown|daily_cap|global_cap`, the step-2 `ERROR` line when
unconfigured, an `ERROR` on send failure (from the caller, alongside
`sender.py`'s own line), `auth.reset_completed`, `WARNING` on best-effort
revocation failure. (These lines are operator-side only; correlating
`reason=` values with domains over time is inside the operator trust
boundary, like every existing `auth.*` line.)

## Frontend

Canonical UI in `app.html`/`app.js` only.

- **"Forgot password?"** — `<button type="button" id="authForgotPasswordBtn"
  class="auth-link-btn">` between the password field and `#authError` in
  `#authForm`; visible only when `authMode === 'login'`, toggled inside
  `setAuthMode()` exactly like `#authDisplayNameField` is for signup.
- **Reset mode** — a third `setAuthMode('reset')` mode in `#authModal`,
  two-stage in one form (the email-change pattern: stage held in a closure
  variable, steps toggled via `hidden`):
  - Stage 1: email entry (reuses `#authEmail`; `#authPassword`'s field and
    the forgot link are hidden in reset mode), submit label "Send code".
  - Stage 2: confirmation copy set via `textContent` (never `innerHTML`) —
    "We sent a 6-character code to `fel•••@gmail.com` — it expires in
    15 minutes. Check your spam folder too." — plus **dedicated stage-2
    inputs** in their own step container: `#resetCodeInput`
    (`maxlength="6" autocomplete="one-time-code"`) and `#resetNewPassword`
    (`autocomplete="new-password"`), the latter with its own `input` listener
    reusing `localPasswordViolations()` for live policy hints (the existing
    `#authPassword` hint listener is hard-gated to signup mode — widen
    nothing; wire the new input separately). Submit label "Reset password".
  - The masked address is the **user's own typed input** (mask helper: first
    3 chars of the local part — 1 if the local part is ≤3 chars — then
    `•••@` + domain). Masking stored account data pre-submission would be an
    enumeration oracle; masking their own input is pure reassurance.
  - **Submit wiring**: the shared `#authForm` submit handler gets a
    `authMode === 'reset'` branch at the top — before the existing
    `if (!email || !password) return;` guard (which would silently no-op
    stage 1) and completely separate from the login/signup success path
    (`setAuthState`/`navigateToPage('agents')`/`claimAgentsForUser()` must
    not run for a reset).
  - **Back affordance**: in reset mode `#authSwitchBtn` reads "Back to sign
    in" and calls `setAuthMode('login')`; any mode switch resets the stage
    closure.
  - Success: toast, `setAuthMode('login')` with the email prefilled.
  - 400/422/429/503 surface through the existing `#authError` convention
    (`data?.detail || data?.error`).
- **State hygiene**: a module-level `resetPasswordResetForm` rebound to the
  closure's reset function and called from `clearAuthState()` — the exact
  `resetEmailChangeForm` pattern, so a second user on the same tab never
  resumes a half-finished reset.
- **`AuthAPI`**: `requestPasswordReset(email)` and
  `resetPassword(email, code, newPassword)` wrappers.
- **Deep link**: `openAuthFromUrl()` learns `auth=reset`; the landing page's
  "Forgot password?" (in its hand-inlined modal, login mode only) is a plain
  link to `/app?auth=reset` — no reset logic duplicated into the hand-patched
  bundle.
- **Cache busters**: bump `app.js?v=` (and `styles.css?v=` if styles change)
  in `app.html`.

## Guards this change must update

- `EXPECTED_FULL_CONTRACT` (`test_app_composition.py`): add
  `("POST", "/api/auth/forgot-password")` and
  `("POST", "/api/auth/reset-password")`.
- `_CSRF_EXEMPT_PATHS` (`csrf.py`): add both paths — unauthenticated browser
  POSTs, same stale-cookie lockout class the login/signup exemption exists
  for. The Origin/Referer check still applies to them (it runs for exempt
  paths too).
- `BLOCKING_IO_HANDLERS` (`test_event_loop_threadpool.py`): add
  `("dashboard.backend.api.auth", "reset_password")`. `forgot_password` stays
  `async def` with all store I/O inside the background task via
  `asyncio.to_thread` (the signup/login offload idiom).
- Twin parity: automatic via the registered `UserStore` pair; new table is a
  `CREATE TABLE` on both sides (the lazy-`ALTER` mirror rule only bites for
  columns added to existing tables later).
- `test_frontend_account_page.py`'s account-card order list is untouched (the
  reset UI lives in the auth modal, not the account view).

## Accepted gaps (deliberate, reviewed)

- **`reset-password` timing**: the real-account failure path costs one extra
  store read (+ a write on wrong-code) vs the unknown-email path. Not leveled
  with dummy work: account existence is already a documented accepted
  disclosure via signup's 409, so this adds nothing an attacker lacks. If
  signup's stance ever tightens, revisit here too.
- **DB cooldown/daily-cap races**: check-then-act, bounded by the atomic
  inline limiters (see Storage). The one-active-row invariant itself is NOT
  in this list — `create_password_reset_request` is transactional.
- **Cross-flow cancel races**: `cancel_password_reset` /
  `cancel_email_change` are best-effort and not atomic with the other flow's
  commit; a sufficiently precise interleaving of a reset and an email-change
  commit can let one outlive the other's cancel. Both flows already revoke
  sessions at commit, which bounds the damage; accepted as-is rather than
  introducing cross-table transactions into the twins.
- **In-process limiters reset on redeploy** (including the global send
  budget): the standing repo stance — in-process budgets bound naive abuse,
  durable per-account caps hold regardless; deploys are merge-triggered and
  infrequent.

## Testing

Backend (`test_auth.py`, existing fixtures: `client`, `sent_emails`,
`_code_from`, backdate helpers; new backdate helper for
`password_reset_requests` rows):

- Happy path end to end: request → code from `sent_emails` → reset → old
  password rejected at login, new one accepted.
- **Enumeration**: known vs unknown email produce byte-identical
  forgot-password responses; unknown email sends no mail and writes no row.
- 503 when Brevo unconfigured (no `sent_emails` fixture; assert the route's
  own `ERROR` line via `capsys`) — identical for known and unknown emails.
- Send failure (`sent_emails.fail_sends()`): 200, nothing persisted, retry
  after `resume_sends()` succeeds immediately (no cooldown burned).
- Cooldown and daily cap: second request inside 300s sends nothing (still
  200, and the `reason=cooldown` line is pinned via `capsys`); backdated rows
  re-enable; 6th request in a day sends nothing; a cancelled row still gates
  the cooldown (status-blind read).
- Global send budget: with the global limiter monkeypatched to a tiny window,
  over-budget requests still 200 but send nothing.
- Wrong code ×5 cancels the request; the correct code then fails.
- Expired code fails (backdated `expires_at`; asserts the store folds expiry
  into "no active row"); single-use (second redeem fails); code is
  case/whitespace-insensitive (via `hash_code`).
- Weak new password → 400 policy detail, code still redeemable.
- Successful reset revokes **all** sessions (a pre-reset token no longer
  works) and cancels a pending email-change.
- Best-effort commit: with `delete_other_sessions` monkeypatched to raise,
  the reset still returns 200 and the password is changed (the
  `test_change_password_revocation_failure_still_succeeds` pattern).
- Cross-flow cancellation is verified **end-to-end**: after change-password
  (and after an email-change commit), the previously-issued reset code
  actually fails redemption — not merely that a cancel call fired.
- Rate limiters: 429 + `Retry-After` on both routes via the
  `auth_rate_limit_clock` monkeypatch pattern, including a case that
  exercises the **per-email** reset limiter specifically (distinct IPs /
  client keys, same email → 429), so a dead per-email budget can't pass.
- Route/guard freezes updated (`EXPECTED_FULL_CONTRACT`, CSRF exemption
  covered by a request-with-stale-cookie test).

Store level: lifecycle + append-only tests in `test_users_store.py` — pinning
the atomic semantics: `mark_password_reset_used` CAS (second caller loses),
`record_password_reset_attempt` cap under concurrent increments,
`create_password_reset_request` leaving exactly one active row — mirrored
`@pg_only` in `test_users_postgres.py` (destructive fixtures keep the
`require_local_postgres_url` guard; the local run skips this tier — verify on
CI).

Frontend source-shape (`test_frontend_account_page.py` idioms —
`_frontend_source.fn_body`, raw substring pins):

- Forgot link exists, hidden outside login mode (`setAuthMode` pin).
- The reset branch in the form handler precedes the shared
  `if (!email || !password)` guard.
- `clearAuthState` calls `resetPasswordResetForm`.
- Stage-2 copy mentions the spam folder (the email-change copy-pin pattern).
- Cache-buster bump test stays green (`>=` floors).

## Delivery

- Branch `feat/password-reset` off `main`; single PR; body notes it **closes
  #187** and why the issue's Resend/env-var blocker no longer applies (Brevo
  vars already power email-change in prod; if absent the new route 503s
  visibly rather than half-working — no pre-merge operator step).
- No new required env vars; the five `AUTH_FORGOT_*`/`AUTH_RESET_*` limiter
  knobs are optional overrides with printed defaults.

## User-facing docs follow-ups (not in scope of the PR; coordinate separately)

- `docs/source/lab/accounts.rst`: add a "Reset your password" section after
  "Sign in and out" (link location, code delivery/expiry, spam-folder note,
  all-sessions sign-out effect).
