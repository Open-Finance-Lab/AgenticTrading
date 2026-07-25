/** Opens the shipped landing signup modal (see dashboard/frontend/index.html). */
export const LANDING_AUTH_MODE = "signup" as const;

export const PRIMARY_LANDING_CTA = {
  label: "Start Free",
  authMode: LANDING_AUTH_MODE,
} as const;
