import { useEffect, useState } from "react";

const AUTH_TOKEN_KEY = "auth-token";
const AUTH_USER_KEY = "auth-user";

/** SAME RULE AS THE GATE IN dashboard/frontend/index.html, restated rather than
 *  imported because that file has no build step to share a constant with this
 *  one — the same split `LANDING_AUTH_EVENT` and `LANDING_DISCORD_CTA` already
 *  straddle. On localhost the API is the dev server's own origin; in prod the
 *  page is served from Vercel and the API from Render behind a same-origin
 *  rewrite, so a RELATIVE path is the only one that reaches it. Getting this
 *  wrong fails in exactly one direction — a cross-origin POST that drops the
 *  session cookie and leaves the server session alive after a "successful"
 *  sign-out — which is why it is derived here once rather than inlined. */
function apiBase(): string {
  const host = window.location.hostname;
  return host === "localhost" || host === "127.0.0.1" ? window.location.origin : "";
}

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

/** The cached display name, or null. Same key, same tolerance to a missing or
 *  corrupt value as `hasAuthToken` — a profile that will not parse is treated
 *  as absent rather than thrown, because this runs during render.
 *
 *  `display_name || email` MATCHES /app's OWN HEADER (app.js), so the two
 *  surfaces name the same visitor the same way. An account that never set a
 *  display name shows its email; one that never had either shows nothing and
 *  the caller falls back to a generic label rather than printing "undefined". */
function readAuthUserName(): string | null {
  try {
    const raw = localStorage.getItem(AUTH_USER_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { display_name?: unknown; email?: unknown } | null;
    if (!parsed || typeof parsed !== "object") return null;
    const name =
      (typeof parsed.display_name === "string" && parsed.display_name.trim()) ||
      (typeof parsed.email === "string" && parsed.email.trim()) ||
      "";
    return name || null;
  } catch {
    return null;
  }
}

/** Reactive cached display name. Listens on the SAME TWO events as
 *  `useSignedIn` and for the same two reasons — `storage` for another tab,
 *  `LANDING_AUTH_EVENT` for this one — so the name and the CTAs can never end
 *  up describing different sessions. */
export function useAuthUser(): string | null {
  const [name, setName] = useState<string | null>(readAuthUserName);

  useEffect(() => {
    const sync = () => setName(readAuthUserName());
    sync();
    window.addEventListener("storage", sync);
    window.addEventListener(LANDING_AUTH_EVENT, sync);
    return () => {
      window.removeEventListener("storage", sync);
      window.removeEventListener(LANDING_AUTH_EVENT, sync);
    };
  }, []);

  return name;
}

/** Sign out from the landing page, without leaving it.
 *
 *  THE LOCAL CLEAR IS IN THE `finally`, which is the same ordering
 *  `logoutUser()` in app.js settles on and for the same reason: a failed
 *  `/api/auth/logout` must still sign the visitor out of THIS tab. The opposite
 *  ordering leaves someone who clicked "Sign out" looking at signed-in CTAs
 *  because a request failed — the one outcome that reads as "it ignored me".
 *
 *  IT DOES NOT NAVIGATE, and that is the difference from /app's version. /app
 *  hops to `/` because its signed-in shell has nothing to show a signed-out
 *  visitor; this page has, and it is already on it. The event below swaps every
 *  CTA in place, which is the whole point of serving one homepage to both
 *  audiences — a redirect here would be a round trip back to the same URL.
 *
 *  `credentials: "include"` IS LOAD-BEARING: the session lives in an HttpOnly
 *  cookie that a default `fetch` would not send, so without it the request
 *  succeeds, clears nothing server-side, and the session stays valid until it
 *  expires. */
export async function signOutLanding(): Promise<void> {
  try {
    await fetch(apiBase() + "/api/auth/logout", {
      method: "POST",
      credentials: "include",
    });
  } catch {
    /* Network failure signs the tab out anyway — see the `finally` note above. */
  } finally {
    try {
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(AUTH_USER_KEY);
    } catch {
      /* ignore */
    }
    // Same-document removeItem notifies nothing; this is the notification.
    try {
      window.dispatchEvent(new Event(LANDING_AUTH_EVENT));
    } catch {
      /* ignore */
    }
  }
}
