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
