import { Link } from "wouter";
import atlLogo from "@assets/atltransparent.png";
import { LANDING_SIGN_IN_CTA, PRIMARY_LANDING_CTA } from "@/lib/cta";

const NAV_LINKS = [
  { href: "#why", label: "Why" },
  { href: "#talk", label: "Talk" },
  { href: "#test", label: "Test" },
  { href: "#race", label: "Race" },
] as const;

/** Same 3-column chrome as dashboard `.header` so the brand sits on the viewport center. */
export function Navbar() {
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
        <button
          type="button"
          data-landing-auth={LANDING_SIGN_IN_CTA.authMode}
          className="text-[15px] font-semibold text-foreground hover:text-foreground/80 transition-colors whitespace-nowrap"
        >
          {LANDING_SIGN_IN_CTA.label}
        </button>
        <button
          type="button"
          data-landing-auth={PRIMARY_LANDING_CTA.authMode}
          className="inline-flex items-center justify-center rounded-md text-[15px] font-semibold h-10 px-5 bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          {PRIMARY_LANDING_CTA.label}
        </button>
      </div>
    </nav>
  );
}
