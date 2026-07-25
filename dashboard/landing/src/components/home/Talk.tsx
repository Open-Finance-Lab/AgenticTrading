import { Button } from "@/components/ui/button";
import { MessageSquare, Bot, Hash } from "lucide-react";
import { DiscordMock } from "./DiscordMock";
import { PRIMARY_LANDING_CTA } from "@/lib/cta";

export function Talk() {
  return (
    <section id="talk" className="py-24 bg-muted/20 border-y border-border scroll-mt-40">
      {/* Hero scroll target — do not remove; Hero.tsx still anchors here */}
      <div id="landing-stats" className="h-0 w-0 overflow-hidden" aria-hidden="true" />

      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.25fr)] gap-10 xl:gap-14 items-center">
          <div>
            <p className="text-base md:text-lg font-mono tracking-wide text-primary mb-3">01 — Talk</p>
            <h2 className="text-3xl md:text-4xl font-bold mb-3">Talk to agents on Discord</h2>
            <p className="text-foreground/80 mb-8 text-lg">
              Describe your trading idea. The agent runs it.
            </p>
            <ol className="space-y-3 mb-8 text-sm text-foreground/80">
              <li className="flex items-start gap-3">
                <Hash className="w-4 h-4 text-primary mt-0.5 shrink-0" />
                <span><span className="text-foreground font-medium">1.</span> Join the server</span>
              </li>
              <li className="flex items-start gap-3">
                <MessageSquare className="w-4 h-4 text-primary mt-0.5 shrink-0" />
                <span><span className="text-foreground font-medium">2.</span> Talk to the agent</span>
              </li>
              <li className="flex items-start gap-3">
                <Bot className="w-4 h-4 text-primary mt-0.5 shrink-0" />
                <span><span className="text-foreground font-medium">3.</span> Get your backtest result</span>
              </li>
            </ol>
            <Button
              size="lg"
              type="button"
              data-landing-auth={PRIMARY_LANDING_CTA.authMode}
              className="bg-primary text-primary-foreground hover:bg-primary/90"
            >
              {PRIMARY_LANDING_CTA.label}
            </Button>
          </div>

          <DiscordMock />
        </div>
      </div>
    </section>
  );
}
