/** Opens the shipped landing signup modal (see dashboard/frontend/index.html). */
export const LANDING_AUTH_MODE = "signup" as const;

/**
 * The only other mode the shipped modal recognises. Both of its coercion sites
 * (`setAuthMode` and the delegated click handler) compare against this exact
 * string and fall back to signup on anything else, so a typo here downgrades the
 * control silently — `test_frontend_bundle_integrity.py` pins the pair instead.
 */
export const LANDING_LOGIN_MODE = "login" as const;

export const PRIMARY_LANDING_CTA = {
  label: "Start Free",
  authMode: LANDING_AUTH_MODE,
} as const;

/** Navbar companion to Start Free — white text link, opens the same modal in login mode. */
export const LANDING_SIGN_IN_CTA = {
  label: "Sign in",
  authMode: LANDING_LOGIN_MODE,
} as const;

/** THE SIGNED-IN HALF OF THE PAGE. `/` used to redirect a signed-in visitor
 *  straight to `/app` (dashboard/frontend/index.html), so the landing had
 *  exactly one audience and its CTAs had exactly one job: open the signup
 *  modal. The redirect is gone — one homepage now serves both states — and
 *  these are what the same buttons say to someone who has already signed up.
 *
 *  `href`, NOT a modal mode, and that difference is the whole shape of these
 *  two constants: a signed-out CTA stays on this page and opens the hand-written
 *  modal, a signed-in one leaves for the app. Anything with an `authMode` is a
 *  modal trigger; anything with an `href` is a link.
 *
 *  TWO RULES THE SHIPPED PAGE ENFORCES FROM OUTSIDE THIS FILE, both in
 *  dashboard/frontend/index.html's delegated click handler:
 *
 *    1. A signed-in CTA must never carry `data-landing-auth`. That attribute is
 *       the handler's primary match; it calls `preventDefault()` and opens the
 *       signup modal, so an <a> carrying it would swallow its own navigation
 *       and offer an account to someone who has one.
 *    2. A signed-in CTA's label must never be exactly "Start Free" or
 *       "Get Started". Those two strings are the handler's LABEL FALLBACK for
 *       CTAs that forgot the attribute — a text match on an element with no
 *       attribute at all — and they are compared after `.trim()`, so they hit
 *       an <a> just as hard. Neither label below is either string, and neither
 *       may be renamed into one.
 */
export const SIGNED_IN_PRIMARY_CTA = {
  label: "Test a trading idea",
  /** THE NAVBAR'S LABEL BELOW `lg`, and it exists for a width constraint rather
   *  than for tone.
   *
   *  `.landing-header` centres the brand by OVERLAYING it across all three grid
   *  columns (index.css), so a wider CTA cluster paints over `.brand-title`
   *  instead of pushing it. The navbar's own comment measures the effect: 65px
   *  of extra button width moved the collision threshold from ~684px to ~814px.
   *
   *  Below `lg` the signed-out cluster is "Start Free" ALONE -- "Sign in" is
   *  `hidden lg:inline-block` -- so the signed-in cluster has no 65px of
   *  departing button to spend there, and "Test a trading idea" is ~55px wider
   *  than "Start Free" outright. That lands the collision threshold around
   *  790-820px: iPad portrait at 768px, garbled, signed in only, with no test
   *  rendering the navbar to catch it.
   *
   *  "My agents" is within a couple of pixels of "Start Free" and names the page
   *  the href actually opens. At `lg` and above the full label fits, because
   *  that is exactly the band where "Sign in" leaves and gives its width back. */
  shortLabel: "My agents",
  /** The same destination /app's own signed-in CTA reaches
   *  (`initHomeGetStarted` -> `navigateToPage('playground', {playgroundTab:
   *  'agents'})`, home-page.js:620) and the same one `goToDashboardLoggedIn`
   *  sends a visitor to the moment they sign up. `?view=` beats saved
   *  `nav-state` on boot, so this needs no localStorage write of its own —
   *  which is why it can be a plain link rather than a click handler. */
  href: "/app?view=agents",
} as const;

/** The invite the whole product uses — app.js's `DISCORD_SERVER_URL`, the /app
 *  header, and the signed-in home screen's own second CTA all point here.
 *  Duplicated rather than imported because `/app` has no build step to share a
 *  constant across the split; if the invite is ever rotated, this is the copy
 *  that lives on the other side of that line. */
export const LANDING_DISCORD_CTA = {
  label: "Join our Discord community",
  href: "https://discord.gg/9HnQ6XDG98",
} as const;

/** The one signed-in control that is neither a modal trigger nor a link.
 *
 *  IT HAS NO `authMode` AND NO `href`, and both absences are deliberate. No
 *  `authMode`, because rule 1 above bans `data-landing-auth` on anything a
 *  signed-in visitor sees. No `href`, because signing out is a POST with a
 *  side effect and a link that performs one is wrong in every way that matters
 *  — it is prefetchable, middle-clickable into a second tab, and reachable by
 *  a crawler. `AccountControl.tsx` renders it as a `<button type="button">`.
 *
 *  Rule 2 still applies with full force even though this is not a link: "Sign
 *  out" is not "Start Free" or "Get Started", and must not be renamed into
 *  either, or index.html's LABEL FALLBACK would open the signup modal on top of
 *  a sign-out click. `test_frontend_bundle_integrity.py` pins that for every
 *  signed-in CTA constant in this file, link or not. */
export const LANDING_SIGN_OUT_CTA = {
  label: "Sign out",
} as const;
