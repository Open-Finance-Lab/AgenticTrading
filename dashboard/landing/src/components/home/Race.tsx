import { Medal, CalendarClock, TrendingUp, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
// No storyline import here on purpose: the Talk → Test story agent belongs to a
// backtest run report (Test.tsx), not to a board.
import { PRIMARY_LANDING_CTA } from "@/lib/cta";
// The sample rows are gone: this table and the hero card render the SAME live
// Competition board, from one fetch. Real numbers in the hero above invented
// ones here, on the same page, is worse than either alone.
import { useLeaderboard } from "@/lib/useLeaderboard";

/** Three facts, in the order a sceptic asks for them: what was held equal, what
 *  the other board is, and what disqualifies a result. Icons carry the shape so
 *  the list reads at a glance — a timer for a fixed window, a rising line for a
 *  board that moves forward, a shield for the rule that withholds publication. */
const BOARD_RULES = [
  {
    icon: CalendarClock,
    text: "Competition: one fixed window of market history — the same days and the same starting capital for every contender.",
  },
  {
    icon: TrendingUp,
    text: "Live Trading Leaderboard: designed to move forward one trading session at a time, in two-week seasons.",
  },
  {
    icon: ShieldCheck,
    text: "Published only if the AI model itself made at least 95% of the decisions.",
  },
] as const;

export function Race() {
  const board = useLeaderboard();
  // The standings table is a board, not the home CHART rank list -- it INCLUDES
  // buy_hold_djia and djia_index alongside the 7 models, deliberately different
  // from /app's models-only rank row. Three reasons: (1) the dashboard's own
  // Competition Leaderboard tab ranks all twelve entries including baselines --
  // it is the home CHART rank list, not this table, that is models-only; (2)
  // the chart on this page already draws both baselines as dashed curves, so a
  // row-less curve would be a dangling reference with nothing to name it; (3)
  // most of the models lost to buy-and-hold, and a models-only table would
  // silently make the page more flattering than the truth -- the exact failure
  // the copy guards in this file exist to prevent. `selectBoardEntries` already
  // seeds `standings` with both baselines; do not add a filter here that drops
  // them back out.
  const standings = board.status === "ready" ? board.data.standings : [];
  return (
    <section id="race" className="py-24 bg-muted/20 border-y border-border scroll-mt-40">
      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-12 items-start">
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-3">What the AI models actually returned</h2>
            <p className="text-foreground/80 mb-6 text-lg">
              Seven leading AI models traded the same days with simulated money, ranked against
              buy-and-hold and the index. Only one finished ahead of both.
            </p>
            <ul className="space-y-3 mb-4 text-sm text-foreground/80">
              {BOARD_RULES.map(({ icon: Icon, text }) => (
                <li key={text} className="flex items-start gap-3">
                  <Icon className="w-4 h-4 text-primary mt-0.5 shrink-0" aria-hidden="true" />
                  <span>{text}</span>
                </li>
              ))}
            </ul>
            {/* "Live" names the direction the board runs, not brokered execution, and
                Season 0 is a shakedown with no nightly advance deployed yet. Both are
                stated on the board's own About card; saying it here too keeps the
                landing from selling a standing that does not exist. */}
            <p className="text-xs text-muted-foreground mb-8">
              The Live Trading Leaderboard is in preview for Season 0. It has not moved forward a
              session yet, and nothing on it is a record. Season 1 is the first that counts.
            </p>
            <Button
              size="lg"
              type="button"
              data-landing-auth={PRIMARY_LANDING_CTA.authMode}
              className="bg-primary text-primary-foreground hover:bg-primary/90"
            >
              {PRIMARY_LANDING_CTA.label}
            </Button>
          </div>

          <div className="bg-card border border-card-border rounded-xl shadow-xl p-6">
            <div className="flex items-center justify-between mb-2 border-b border-border pb-4 gap-3">
              <h3 className="text-xl font-bold flex items-center gap-2 min-w-0">
                <Medal className="w-5 h-5 text-primary shrink-0" aria-hidden="true" />
                Competition Standings
              </h3>
              {/* Literal, not a shared constant — see the note in
                  BoardPreview.tsx: the guard counts occurrences in the minified
                  bundle. */}
              <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded shrink-0">
                {board.status === "ready" && board.data.windowLabel
                  ? `Competition window · ${board.data.windowLabel}`
                  : "Competition window"}
              </span>
            </div>
            <div className="space-y-2 mt-4">
              <div className="grid grid-cols-12 text-xs font-mono text-muted-foreground pb-2 px-2">
                <div className="col-span-2">Rank</div>
                <div className="col-span-7">AI model</div>
                <div className="col-span-3 text-right">Return</div>
              </div>
              {board.status === "loading" ? (
                <p className="px-2 py-6 text-sm text-muted-foreground">Loading the board…</p>
              ) : board.status === "error" ? (
                // Names the failure. Absent and broken must not render the same.
                <p className="px-2 py-6 text-sm text-muted-foreground">
                  The standings didn&apos;t load ({board.message}). Reload to try again.
                </p>
              ) : (
                standings.map((item, index) => (
                  <div
                    key={item.key}
                    className={`grid grid-cols-12 items-center p-3 border rounded-lg ${
                      index === 0
                        ? "bg-primary/10 border-primary/40"
                        : "bg-background border-border"
                    }`}
                  >
                    <div className="col-span-2 font-mono font-bold text-muted-foreground">
                      #{index + 1}
                    </div>
                    <div
                      className={`col-span-7 font-medium truncate pr-2 ${
                        index === 0 ? "text-primary" : "text-foreground"
                      }`}
                    >
                      {item.name}
                    </div>
                    <div
                      className={`col-span-3 text-right font-mono font-bold ${
                        index === 0
                          ? "text-primary"
                          : item.ret.startsWith("-")
                            ? "text-destructive"
                            : "text-positive"
                      }`}
                    >
                      {item.ret}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
