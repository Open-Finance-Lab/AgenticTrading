/** Landing CTAs funnel anonymous visitors to the app signup modal. */
export const SIGNUP_HREF = "/app?auth=signup";
export const DASHBOARD_HREF = "/app?view=home";

export function primaryLandingCta(signedIn: boolean): { href: string; label: string } {
  if (signedIn) {
    return { href: DASHBOARD_HREF, label: "Open Dashboard" };
  }
  return { href: SIGNUP_HREF, label: "Start Free" };
}
