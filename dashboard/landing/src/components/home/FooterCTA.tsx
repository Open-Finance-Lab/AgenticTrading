import { Button } from "@/components/ui/button";
import { PRIMARY_LANDING_CTA } from "@/lib/cta";

export function FooterCTA() {
  return (
    <footer className="py-24 relative overflow-hidden text-center border-t border-border">
      <div className="absolute inset-0 bg-grid-pattern opacity-10 [mask-image:radial-gradient(ellipse_at_center,black,transparent_70%)]" />
      <div className="container mx-auto px-6 relative z-10">
        <p className="text-sm font-mono tracking-wide text-muted-foreground mb-4">
          Talk → Test → Race
        </p>
        <h2 className="text-4xl md:text-5xl font-bold tracking-tighter mb-10">Ready to test your first idea?</h2>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Button
            size="lg"
            type="button"
            data-landing-auth={PRIMARY_LANDING_CTA.authMode}
            className="w-full sm:w-auto bg-primary text-primary-foreground hover:bg-primary/90 text-base h-12 px-8"
          >
            {PRIMARY_LANDING_CTA.label}
          </Button>
        </div>

        <div className="mt-24 pt-8 border-t border-border flex flex-col md:flex-row justify-between items-center text-sm text-muted-foreground">
          <div>
            © 2026 SecureFinAI Lab · Agentic Trading Lab is an open-source
            research platform by SecureFinAI Lab, organizer of the
            SecureFinAI Contest 2026 ·{" "}
            <a
              href="https://github.com/Open-Finance-Lab/AgenticTrading"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-foreground"
            >
              GitHub
            </a>
          </div>
          <div className="flex gap-6 mt-4 md:mt-0">
            <a
              href="https://finagent-orchestration.readthedocs.io/en/latest/"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-foreground"
            >
              Documentation
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
