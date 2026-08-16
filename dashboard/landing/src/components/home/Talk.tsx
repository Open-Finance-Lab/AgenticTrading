import { Button } from "@/components/ui/button";
import { DiscordMock } from "./DiscordMock";
import { ChatWindow } from "./ChatSimulation";
import { PRIMARY_LANDING_CTA } from "@/lib/cta";

export function Talk() {
  return (
    <section id="talk" className="py-24 bg-muted/20 border-y border-border scroll-mt-40">
      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.25fr)] gap-10 xl:gap-14 items-center">
          <div>
            <p className="text-base md:text-lg font-mono tracking-wide text-primary mb-3">01 — Talk</p>
            <h2 className="text-3xl md:text-4xl font-bold mb-3">Describe your idea in plain language</h2>
            <p className="text-foreground/80 mb-8 text-lg">
              Write how you want to trade. The agent follows it, hour by hour.
            </p>
            {/* The three-step list is gone: it restated WhyCare's three acts
                one screen later, and its Discord step duplicated the Discord
                section below, which shows the thing rather than describing it.
                The lucide import went with it — those three icons had no other
                use in this file, and an unused import is a noUnusedLocals build
                failure, not a lint nit. */}
            <Button
              size="lg"
              type="button"
              data-landing-auth={PRIMARY_LANDING_CTA.authMode}
              className="bg-primary text-primary-foreground hover:bg-primary/90"
            >
              {PRIMARY_LANDING_CTA.label}
            </Button>
          </div>

          {/* Moved down from the hero, which now shows the board. This is the
              act the conversation actually illustrates — it was demonstrating
              "talk to an agent" one section above the section titled Talk. */}
          <ChatWindow />
        </div>

        <div className="mt-16 lg:mt-20 grid lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.25fr)] gap-10 xl:gap-14 items-center">
          <div>
            <h3 className="text-2xl font-bold mb-3">Or say it in our Discord community</h3>
            <p className="text-foreground/80 text-lg">
              Same agent, same instruction — answered in a channel instead of a dashboard.
            </p>
          </div>
          <DiscordMock />
        </div>
      </div>
    </section>
  );
}
