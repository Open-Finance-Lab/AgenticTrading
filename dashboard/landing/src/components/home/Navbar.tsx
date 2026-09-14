import { Link } from "wouter";
import atlLogo from "@assets/atltransparent.png";
import {
  LANDING_SIGN_IN_CTA,
  PRIMARY_LANDING_CTA,
  SIGNED_IN_PRIMARY_CTA,
} from "@/lib/cta";
import { useSignedIn } from "@/lib/session";

const NAV_LINKS = [
  { href: "#why", label: "Why" },
  { href: "#talk", label: "Talk" },
  { href: "#test", label: "Test" },
  { href: "#race", label: "Race" },
] as const;

/** Same 3-column chrome as dashboard `.header` so the brand sits on the viewport center. */
export function Navbar() {
  const signedIn = useSignedIn();
  return (
    <nav className="landing-header border-b border-border bg-background/80 backdrop-blur-md">
      <div className="hidden md:flex items-center gap-3 text-[15px] font-semibold text-muted-foreground min-w-0">
        {NAV_LINKS.map((link) => (
          <a key={link.href} href={link.href} className="hover:text-foreground transition-colors whitespace-nowrap">
            {link.label}
          </a>
        ))}
      </div>
      <Link href="/" className="brand-lockup">
        <div className="brand-logo">
          <img src={atlLogo} alt="" />
        </div>
        <span className="brand-title">Agentic Trading Lab</span>
      </Link>
      <div className="flex items-center justify-end gap-4 min-w-0">
        {/*
          Hidden below `lg` on purpose. `.landing-header` centres the brand by
          *overlaying* it across all three columns (index.css), so a wider CTA
          cluster covers `.brand-title` rather than pushing it: the 65px this
          button adds drags the collision threshold from ~684px up to ~814px,
          garbling the navbar on iPad portrait. `md:` is 768px — inside that
          band — so `lg:` is the first safe breakpoint. Login stays reachable
          below it via the modal's own "Already have an account?" switch.
        */}
        {signedIn ? null : (
          <button
            type="button"
            data-landing-auth={LANDING_SIGN_IN_CTA.authMode}
            className="hidden lg:inline-block text-[15px] font-semibold text-foreground hover:text-foreground/80 transition-colors whitespace-nowrap"
          >
            {LANDING_SIGN_IN_CTA.label}
          </button>
        )}
        {/* TWO LABELS, ONE LINK, BANDED ON THE SAME BREAKPOINT "Sign in" USES,
            because the width note above applies to the signed-in cluster too and
            the obvious arithmetic for it is wrong.

            The tempting version: "Sign in" leaves (-65px) and "Start Free"
            widens to "Test a trading idea" (+~55px), so the signed-in cluster
            is narrower. That is true only at >=lg. "Sign in" is
            `hidden lg:inline-block`, so BELOW 1024px it was never rendered and
            there is no 65px to give back -- the signed-out cluster there is
            "Start Free" alone, and a single wider button is a straight +55px.
            Using this file's own measurement (65px of width moved the collision
            threshold ~684px -> ~814px, about 2px of threshold per px of width),
            +55px lands the threshold around 790-820px. iPad portrait is 768.
            `.landing-header` paints the brand UNDER the CTA cluster rather than
            being pushed by it, so the failure is "Agentic Trading Lab" with a
            button over it -- silent, signed-in only, and nothing in the suite
            renders this navbar.

            So the full label ships exactly where the width for it does, and
            below that the link carries `shortLabel` ("My agents"), which is
            within a couple of pixels of "Start Free" and names the page the href
            opens. One <a>, so there is one destination and one accessible name
            per band rather than two controls to keep in step.

            Neither label may be "Start Free" or "Get Started", and this element
            carries no data-landing-auth -- see the two rules in lib/cta.ts.
            `test_no_signed_in_cta_is_hijacked_by_the_delegated_handler` reads
            both labels out of that file and checks both. The class string is the
            signed-out button's, unchanged, so the control keeps its measured
            height and padding. */}
        {signedIn ? (
          <a
            href={SIGNED_IN_PRIMARY_CTA.href}
            className="inline-flex items-center justify-center rounded-md text-[15px] font-semibold h-10 px-5 bg-primary text-primary-foreground hover:bg-primary/90 transition-colors whitespace-nowrap"
          >
            <span className="lg:hidden">{SIGNED_IN_PRIMARY_CTA.shortLabel}</span>
            <span className="hidden lg:inline">{SIGNED_IN_PRIMARY_CTA.label}</span>
          </a>
        ) : (
          <button
            type="button"
            data-landing-auth={PRIMARY_LANDING_CTA.authMode}
            className="inline-flex items-center justify-center rounded-md text-[15px] font-semibold h-10 px-5 bg-primary text-primary-foreground hover:bg-primary/90 transition-colors whitespace-nowrap"
          >
            {PRIMARY_LANDING_CTA.label}
          </button>
        )}
      </div>
    </nav>
  );
}
