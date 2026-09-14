import { Button } from "@/components/ui/button";
import {
  LANDING_DISCORD_CTA,
  PRIMARY_LANDING_CTA,
  SIGNED_IN_PRIMARY_CTA,
} from "@/lib/cta";
import { useSignedIn } from "@/lib/session";

/** ONE COMPONENT FOR ALL SIX BODY CTAs, so they cannot drift apart.
 *
 *  Before the landing served signed-in visitors, every one of these was the
 *  same four lines of JSX copied into six files, and that was survivable
 *  because there was one state to render. There are two now, and six
 *  hand-written branches is six chances for one section to keep offering a free
 *  account to someone who already has one — the complaint that started this
 *  change, one level down. `test_frontend_bundle_integrity.py` counts the
 *  shipped labels; this is what makes that count mean something.
 *
 *  SIGNED OUT, THE DOM IS BYTE-FOR-BYTE WHAT IT WAS: a single <Button> carrying
 *  `data-landing-auth`. That is deliberate rather than incidental — the
 *  anonymous page is the one with traffic on it, and this change should not be
 *  able to alter it. Every measured class string the callers pass still lands
 *  on that same element.
 *
 *  SIGNED IN, IT BECOMES TWO, and they are wrapped rather than returned as a
 *  fragment. TWO of the six callers already wrap their CTA in a flex row (Hero,
 *  FooterCTA) and FOUR do not (WhyCare, Talk, Test, Race) — a bare fragment
 *  would lay the pair out correctly in the first group and butt them together
 *  with no gap in the second. The wrapper costs the signed-out case
 *  nothing because the signed-out case never renders it.
 */
export function LandingCTA({
  size = "lg",
  className = "",
  secondaryClassName = "",
}: {
  size?: "default" | "sm" | "lg" | "icon";
  /** Passed through to the primary button in BOTH states — these are the
   *  per-section measured class strings (`w-full sm:w-auto`, `h-12 px-8`,
   *  glow), and a signed-in button that dropped them would be a different
   *  size from the one it replaced. */
  className?: string;
  secondaryClassName?: string;
}) {
  const signedIn = useSignedIn();

  if (!signedIn) {
    return (
      <Button
        size={size}
        type="button"
        data-landing-auth={PRIMARY_LANDING_CTA.authMode}
        className={className}
      >
        {PRIMARY_LANDING_CTA.label}
      </Button>
    );
  }

  return (
    <div className="flex flex-col sm:flex-row items-center gap-4">
      {/* `asChild`, so the rendered element is an <a> and not a <button> with a
          click handler. A real link is what makes this work with middle-click,
          ⌘-click and "copy link address" — and, more to the point here, it is
          what keeps the destination visible in the markup rather than buried in
          a handler the delegated listener in index.html would have to be
          trusted not to preventDefault. */}
      <Button asChild size={size} className={className}>
        <a href={SIGNED_IN_PRIMARY_CTA.href}>{SIGNED_IN_PRIMARY_CTA.label}</a>
      </Button>
      <Button asChild size={size} variant="outline" className={secondaryClassName}>
        <a href={LANDING_DISCORD_CTA.href} target="_blank" rel="noopener noreferrer">
          {LANDING_DISCORD_CTA.label}
        </a>
      </Button>
    </div>
  );
}
