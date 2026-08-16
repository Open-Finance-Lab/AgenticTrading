import { MessageSquare, LineChart, Trophy, Cpu, Code2, Hash } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PRIMARY_LANDING_CTA } from "@/lib/cta";

/** Unnumbered on purpose: Talk/Test below run the numbered sequence, and a
 *  second one here made the summary compete with the narrative it introduces.
 *  Race lost its own number when the board moved into the hero — the board is
 *  the first thing on the page now, so numbering it third described the wrong
 *  page. (Do not write that number as a quoted string anywhere in this file:
 *  test_band_runs_no_second_step_sequence greps for one.) */
const ACTS = [
  {
    icon: MessageSquare,
    title: "Describe it in plain English",
    body: "No code, no formulas — write it the way you would explain it to a person.",
  },
  {
    icon: LineChart,
    title: "Prove it on real market data",
    body: "Real prices and real market hours, measured against buy-and-hold and the index.",
  },
  {
    icon: Trophy,
    title: "See how it ranks",
    // Not "everyone else's agents": no user agent is on any board, and the
    // roster is curated (`dashboard/config/leaderboard.json`). The comparison
    // that actually exists is against the AI models and the passive baselines.
    body: "The same days and the same starting capital as every AI model on the board.",
  },
] as const;

const EXTRAS = [
  {
    icon: Cpu,
    title: "Pick the AI model",
    body: "Same idea, different AI models — Claude, GPT, Gemini, and more, all available to try.",
  },
  {
    icon: Code2,
    title: "For developers: bring your own agent",
    body: "A Python toolkit (SDK), if you would rather write the code.",
  },
  {
    icon: Hash,
    title: "Talk to it in our Discord community",
    body: "If you would rather have a conversation.",
  },
] as const;

export function WhyCare() {
  return (
    <section id="why" className="py-24 scroll-mt-40">
      {/* Hero's scroll target — moved here from Talk so the first scroll lands
          on the value proposition rather than past it. Hero.tsx still anchors
          to #landing-stats; do not rename without updating it.

          scroll-mt-40 is NOT redundant with the section's: scroll-margin is
          read off the element scrollIntoView() targets and is not inherited,
          so without it here the headline parks under the fixed .landing-chrome
          (120px). Deleting it as a duplicate reintroduces that. */}
      <div id="landing-stats" className="h-0 w-0 overflow-hidden scroll-mt-40" aria-hidden="true" />

      <div className="container mx-auto px-6">
        <div className="max-w-3xl mb-14">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 tracking-tight">
            You have an idea about the market.
            <span className="block text-[#22d3ee]">Testing it properly is the expensive part.</span>
          </h2>
          <p className="text-foreground/80 text-lg">
            Normally that means writing code, buying data, and waiting months to find out you were
            wrong.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-16">
          {ACTS.map(({ icon: Icon, title, body }) => (
            <div key={title}>
              <Icon className="w-6 h-6 text-primary mb-3" aria-hidden="true" />
              <h3 className="text-lg font-bold mb-2">{title}</h3>
              <p className="text-sm text-foreground/70 leading-relaxed">{body}</p>
            </div>
          ))}
        </div>

        <div className="grid sm:grid-cols-3 gap-6 pt-10 border-t border-border">
          {EXTRAS.map(({ icon: Icon, title, body }) => (
            <div key={title} className="flex gap-3">
              <Icon className="w-5 h-5 text-primary shrink-0 mt-0.5" aria-hidden="true" />
              <div>
                <h4 className="text-sm font-semibold text-foreground mb-1">{title}</h4>
                <p className="text-sm text-foreground/60 leading-relaxed">{body}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-14">
          <Button
            size="lg"
            type="button"
            data-landing-auth={PRIMARY_LANDING_CTA.authMode}
            className="bg-primary text-primary-foreground hover:bg-primary/90 text-base h-12 px-8"
          >
            {PRIMARY_LANDING_CTA.label}
          </Button>
        </div>
      </div>
    </section>
  );
}
