/** Opens the shipped landing signup modal (see dashboard/frontend/index.html). */
export const LANDING_AUTH_MODE = "signup" as const;

export const PRIMARY_LANDING_CTA = {
  label: "Start Free",
  authMode: LANDING_AUTH_MODE,
} as const;

/** Navbar companion to Start Free — white text link, opens the same modal in login mode. */
export const LANDING_SIGN_IN_CTA = {
  label: "Sign in",
  authMode: "login",
} as const;
