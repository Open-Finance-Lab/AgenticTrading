import { useState } from "react";
import { LANDING_SIGN_OUT_CTA } from "@/lib/cta";
import { signOutLanding, useAuthUser, useSignedIn } from "@/lib/session";

/** THE WAY OUT OF THE SIGNED-IN HOMEPAGE.
 *
 *  `/` used to redirect a signed-in visitor to `/app`, so the landing page had
 *  no signed-in state and needed no sign-out: the only surface that could show
 *  you as signed in was the one that could sign you out. Serving both states
 *  off one page removed the redirect and, with it, the only place the control
 *  lived — a visitor landing on `/` saw their own CTAs ("Test a trading idea")
 *  with nothing on the page able to end the session, and had to open `/app`
 *  purely to leave it.
 *
 *  IT ALSO ANSWERS "WHO AM I", which is the half a bare button would miss. On a
 *  shared or handed-over browser the cached profile is the only thing that says
 *  whose session these CTAs belong to; "Signed in as <name>" makes the answer
 *  visible next to the control that changes it, rather than only after a click.
 *
 *  RENDERS NOTHING WHEN SIGNED OUT — not a disabled control, not a placeholder.
 *  The anonymous page is the one with traffic on it and this component must not
 *  be able to alter it; `null` is the only return that guarantees that. */
export function AccountControl({ className = "" }: { className?: string }) {
  const signedIn = useSignedIn();
  const name = useAuthUser();
  // IN-FLIGHT GUARD, not a spinner. `signOutLanding` is a POST, and a
  // double-click would send two — the second racing the first's own
  // `LANDING_AUTH_EVENT`, which unmounts this component mid-request. Disabling
  // is enough: the control disappears on success, so there is no "done" state
  // to render.
  const [busy, setBusy] = useState(false);

  if (!signedIn) return null;

  const signOut = async () => {
    if (busy) return;
    setBusy(true);
    try {
      await signOutLanding();
    } finally {
      // Reached on failure. On success the event above has already flipped
      // `signedIn` and this component is gone, so the write is a no-op rather
      // than a leak — React tolerates it, and dropping the `finally` would pin
      // `busy` forever on the one path where the button still has to work.
      setBusy(false);
    }
  };

  return (
    <div className={`flex items-center gap-3 text-sm ${className}`}>
      {/* `truncate` with a `max-w`: an email is the fallback display name, and a
          long one would otherwise push the sign-out button off a narrow
          container — the same overlay/width hazard the navbar documents. */}
      <span className="text-muted-foreground truncate max-w-[14rem]">
        Signed in as{" "}
        <span className="text-foreground font-medium">{name ?? "your account"}</span>
      </span>
      {/* A `<button type="button">`, never an <a>: this is a POST with a side
          effect. `type` is explicit because the default inside a <form> is
          `submit`, and this component does not control where it is mounted.
          No `data-landing-auth` — see the rules on LANDING_SIGN_OUT_CTA. */}
      <button
        type="button"
        onClick={signOut}
        disabled={busy}
        className="underline underline-offset-4 text-muted-foreground hover:text-foreground disabled:opacity-60 disabled:cursor-not-allowed"
      >
        {LANDING_SIGN_OUT_CTA.label}
      </button>
    </div>
  );
}
