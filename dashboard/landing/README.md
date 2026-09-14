# Landing page

The page served at `/`, for **both** signed-out and signed-in visitors. React +
Vite + Tailwind v4.

It used to be signed-out only: the auth gate in the shipped `index.html`
`location.replace`d anyone carrying a valid session to `/app`, so `/` and
/app's Home screen were free to drift into two different products, and did. The
redirect is gone. What changes between the two audiences is the CTAs —
`src/lib/cta.ts` holds both pairs and `src/components/home/LandingCTA.tsx`
chooses — and nothing else.

Originally exported from a Replit **pnpm monorepo**; it has since been made a
self-contained standalone app so it builds with plain `npm` (no workspace, no
`pnpm-workspace.yaml` catalog). See "History" below for what that entailed.

## Structure

- `src/` — React + Vite + Tailwind source
  - `src/lib/utils.ts` — the shadcn `cn()` helper (imported as `@/lib/utils`)
  - `attached_assets/` — build-time image imports (`@assets/…`), e.g. the logo
- `public/` — static passthrough assets (favicon, robots.txt)
- Built static output is copied into `../frontend/` for deployment (see below).

## Build (standalone)

```bash
cd dashboard/landing
npm install            # plain npm; uses the committed package-lock.json
npm run build          # → dist/public/  (BASE_PATH defaults to /, PORT to 5173)
npm run typecheck      # tsc --noEmit (optional; currently clean)
npm run dev            # local dev server
```

`PORT` / `BASE_PATH` are optional (they default); set them to override, e.g.
`BASE_PATH=/some/sub/path npm run build`.

## ⚠️ Refreshing the shipped `../frontend/` bundle keeps a small auth layer

The production landing page (`dashboard/frontend/index.html`, served at `/`) is the
Vite `index.html` **plus a small inline auth layer** that can't live in the static
React bundle. That layer is, by design, all that remains hand-written in
`index.html`:

- an **auth-gate `<script>`** in `<head>` that revalidates a cached `auth-user`
  against `/api/auth/me` and holds the page invisible until it answers (runs
  before React, to avoid a content flash). It does **not** redirect: it decides
  whether the cached profile is real, clears it if not, and dispatches a
  `landing-auth-change` event so an already-mounted React tree re-reads it —
  `storage` does not fire in the tab that wrote, so without that dispatch an
  expired session keeps rendering the signed-in CTAs;
- the **`#landingAuthModal`** markup + its `<style id="landing-auth-patch">` and
  end-of-body `<script>` — a signup/sign-in modal that talks to `/api/auth/*` and
  a click delegation that funnels the landing's CTAs into it.

Everything else the landing shows — the nav (with its native `data-landing-auth`
**Start Free** and **Sign in** buttons), the Discord-prompt and paper-trading sections, the
fixed-height agent playground — is now rendered **natively by the React source**.
(An earlier shipped bundle injected those via a `MutationObserver` patch; the source
has since absorbed them, and that obsolete patch machinery was removed so it can't
duplicate sections or fight the native grid nav.)

So a bundle refresh is now close to mechanical:

```bash
npm run build
cp dist/public/assets/* ../frontend/assets/     # new content-hashed JS/CSS/img
# remove the superseded ../frontend/assets/index-*.{js,css} + atl-logo-*.png
# In ../frontend/index.html: point the <script>/<link> at the NEW asset
#   filenames. KEEP the auth-gate <script>, #landingAuthModal markup,
#   <style id="landing-auth-patch">, and the end-of-body auth <script>.
```

The asset filenames are content-hashed, so **no `?v=` cache-buster is needed** —
a new build produces a new name, which is the cache bust. (Older revisions of this
file told you to bump one; the shipped `index.html` has carried none since the
`frontend/` refresh, and adding one now would just be noise.)

`dashboard/backend/tests/test_frontend_bundle_integrity.py` guards the mechanical
half of the steps above in CI: every `/assets/...` reference in `index.html` must
resolve, no superseded bundle may be left behind, and the four hand-written auth
markers must still be present. It does **not** rebuild the bundle — Vite's content
hashes move with toolchain versions, so a reproducibility check would be flaky.

The auth layer binds to native `data-landing-auth` buttons + CTA labels, so it no
longer depends on any patch-injected DOM. Still: it touches the live signup/login
flow, so verify in a browser (or headlessly — load `/`, confirm each section renders
once, the modal opens on every **Start Free** button *in signup mode* and on the
navbar's **Sign in** button *in login mode*, and the only console error is
the `/_vercel/insights/script.js` 404, which exists off Vercel and is expected) before
shipping.

**Check the signed-in half too**, which is the half no anonymous smoke test
reaches: with a valid session, `/` must render rather than bounce, every CTA must
read **Test a trading idea** + **Join our Discord community**, and both must
NAVIGATE. A signed-in CTA that opens the signup modal instead is the failure mode
to watch for — the delegated handler in `index.html` matches on the
`data-landing-auth` attribute *and* on the literal labels `Start Free` /
`Get Started`, and either match calls `preventDefault()`, so it will swallow an
`<a>` as readily as a `<button>`.
`test_no_signed_in_cta_is_hijacked_by_the_delegated_handler` pins both rules
against the real `index.html`, so this is a browser confirmation rather than the
only line of defence.

Then sign out from `/app`: it `location.replace`s back to `/`, and the CTAs must
already read **Start Free** — `logoutUser` clears the cached profile before the
hop for exactly that reason. Note the **Sign in** control is `lg:`-gated — widen the viewport past
1024px or it legitimately will not be in the DOM's layout. Longer term, folding the auth modal + gate into the React source would
remove even this remnant, making the build output *exactly* the shipped page.

## History — why this failed to build from a clean clone

The Replit export was severed from its monorepo, leaving it unbuildable:

- `package.json` used pnpm-only `catalog:` / `workspace:*` version specifiers with
  no `pnpm-workspace.yaml` to resolve them → replaced with concrete versions.
- `@workspace/api-client-react` (`workspace:*`) pointed at a sibling package that
  isn't in this repo and nothing in `src/` imported → dropped.
- `tsconfig.json` extended a missing `../../tsconfig.base.json` and referenced a
  missing `../../lib/api-client-react` → made self-contained.
- `vite.config.ts` threw unless `PORT`/`BASE_PATH` were set and imported Replit-only
  plugins → env made optional, Replit plugins dropped.
- `src/lib/utils.ts` (the `cn` helper every `ui/` component imports) was absent →
  restored. The repo-root `.gitignore` also had a blanket `lib/` (Python) rule that
  silently un-tracked it → negated for this path.
- `typescript` wasn't a declared dependency → added.
