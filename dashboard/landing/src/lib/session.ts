import { useEffect, useState } from "react";

const AUTH_USER_KEY = "auth-user";

/** Dispatched on `window` by the pre-hydration auth gate in
 *  dashboard/frontend/index.html once `/api/auth/me` has answered.
 *
 *  IT EXISTS BECAUSE `storage` DOES NOT FIRE IN THE TAB THAT WROTE, which is
 *  the only tab that matters here. The gate reads the cached `auth-user`,
 *  revalidates it against the cookie session, and on a 401 calls its
 *  `clearAuth()` — a same-document `localStorage.removeItem`, which notifies
 *  nothing. Every listener below would keep reporting the stale cached profile
 *  until a reload.
 *
 *  That was invisible while `/` redirected a signed-in visitor to `/app`: the
 *  CTAs had one state, nothing read this hook, and the comment on
 *  `useSignedIn` said so. Serving both states off one page is what makes it
 *  reachable — an expired session would be offered "Test a trading idea",
 *  linking to an app that bounces them back to sign in, which is the same
 *  wrong-audience CTA this change exists to remove, one turn further on.
 *
 *  A plain `Event`, not a `CustomEvent` carrying the user: the gate and this
 *  hook already agree on where the truth lives (`localStorage[AUTH_USER_KEY]`),
 *  and a payload would give them a second, overridable one. The event says
 *  "re-read", nothing more. */
export const LANDING_AUTH_EVENT = "landing-auth-change";

/** True when a cached auth-user profile is present (session cookie is HttpOnly). */
export function hasAuthToken(): boolean {
  try {
    return Boolean(localStorage.getItem(AUTH_USER_KEY));
  } catch {
    return false;
  }
}

/** Reactive signed-in flag. Every landing CTA branches on this.
 *
 *  THE INITIAL READ IS THE CACHED PROFILE, deliberately optimistic: it is
 *  synchronous, so a returning visitor's first paint already carries the
 *  signed-in CTAs rather than flashing "Start Free" and swapping. The gate's
 *  revalidation lands within the same fetch that would have redirected them,
 *  and corrects the optimism through the event below if the session is dead.
 *
 *  BOTH LISTENERS, and they answer different questions. `storage` is the OTHER
 *  tab — sign out in one tab and this one follows. `LANDING_AUTH_EVENT` is
 *  THIS tab, which `storage` is specified never to notify; without it the one
 *  case that matters, the gate clearing an expired session on this very page,
 *  is the one case nothing hears. */
export function useSignedIn(): boolean {
  const [signedIn, setSignedIn] = useState(hasAuthToken);

  useEffect(() => {
    const sync = () => setSignedIn(hasAuthToken());
    sync();
    window.addEventListener("storage", sync);
    window.addEventListener(LANDING_AUTH_EVENT, sync);
    return () => {
      window.removeEventListener("storage", sync);
      window.removeEventListener(LANDING_AUTH_EVENT, sync);
    };
  }, []);

  return signedIn;
}
