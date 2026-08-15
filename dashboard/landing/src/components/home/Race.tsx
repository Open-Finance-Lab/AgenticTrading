import { Medal } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";
// No storyline import here on purpose: the Talk → Test story agent belongs to a
// backtest run report (Test.tsx), not to a board. See SAMPLE_CURVES below.
import { PRIMARY_LANDING_CTA } from "@/lib/cta";

/** Sample curves for the Live Trading Leaderboard, which advances one session at a
 *  time — hence a relative day axis rather than a fixed contest month. Illustrative:
 *  the card carries a visible label saying so.
 *
 *  Every series here is a board entry from the curated
 *  `dashboard/config/leaderboard.json` roster. There is deliberately no "yours"
 *  curve and no user agent in the standings: no user agent is on any board, and
 *  drawing one — highlighted, at rank 2, as the thickest line — sold the entry
 *  flow the copy beside it was rewritten to stop promising. A picture makes that
 *  promise more vividly than the sentence that was removed.
 *
 *  The shape is also not a clean sweep on purpose. On the real board exactly one
 *  model finished ahead of the passive baselines, so an illustration in which
 *  every model beats buy-and-hold would be its own false claim about what these
 *  agents do. */
const SAMPLE_CURVES = [
  { day: "7d ago", deepseek: 1000, claude: 1000, gpt: 1000, buyHold: 1000, djia: 1000 },
  { day: "5d ago", deepseek: 1061, claude: 1035, gpt: 1012, buyHold: 1018, djia: 1008 },
  { day: "3d ago", deepseek: 1094, claude: 1012, gpt: 986, buyHold: 1005, djia: 995 },
  { day: "2d ago", deepseek: 1128, claude: 1026, gpt: 972, buyHold: 1032, djia: 1014 },
  { day: "Yesterday", deepseek: 1186, claude: 1019, gpt: 991, buyHold: 1048, djia: 1022 },
  { day: "Now", deepseek: 1210, claude: 1014, gpt: 985, buyHold: 1055, djia: 1028 },
];

const SAMPLE_STANDINGS = [
  { rank: 1, name: "DeepSeek V4 Pro", ret: "+21.0%", highlight: true },
  { rank: 2, name: "Buy & Hold", ret: "+5.5%", highlight: false },
  { rank: 3, name: "DJIA", ret: "+2.8%", highlight: false },
  { rank: 4, name: "Claude Sonnet 4.6", ret: "+1.4%", highlight: false },
  { rank: 5, name: "GPT-5.5", ret: "-1.5%", highlight: false },
];

const LINE_COLORS = {
  deepseek: "#a78bfa",
  claude: "#fbbf24",
  gpt: "#22d3ee",
  buyHold: "#94a3b8",
  djia: "#64748b",
} as const;

export function Race() {
  return (
    <section id="race" className="py-24 bg-muted/20 border-y border-border scroll-mt-40">
      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-12 items-start mb-12">
          <div>
            <p className="text-base md:text-lg font-mono tracking-wide text-primary mb-3">03 — Race</p>
            <h2 className="text-3xl md:text-4xl font-bold mb-3">See where the bar is</h2>
            <p className="text-foreground/80 mb-6 text-lg">
              Leading AI models, ranked head to head against passive baselines — simulated
              trading on real market data, with no broker and no capital at risk.
            </p>
            <ul className="space-y-2 mb-4 text-sm text-foreground/80">
              <li>· Competition: one fixed window, identical for every entry</li>
              <li>· Live Trading Leaderboard: runs forward in two-week seasons</li>
              <li>· Published only if the AI model actually drove the run</li>
            </ul>
            {/* "Live" names the direction the board runs, not brokered execution, and
                Season 0 is a shakedown with no nightly advance deployed yet. Both are
                stated on the board's own About card; saying it here too keeps the
                landing from selling a standing that does not exist. */}
            <p className="text-xs text-muted-foreground mb-8">
              The Live Trading Leaderboard is in preview for Season 0 while the nightly
              advance ships. Season 1 is the first that counts.
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
                <Medal className="w-5 h-5 text-primary shrink-0" />
                Standings
              </h3>
              <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded shrink-0">Illustrative example</span>
            </div>
            <div className="space-y-2 mt-4">
              <div className="grid grid-cols-12 text-xs font-mono text-muted-foreground pb-2 px-2">
                <div className="col-span-2">Rank</div>
                <div className="col-span-7">Entry</div>
                <div className="col-span-3 text-right">Return</div>
              </div>
              {SAMPLE_STANDINGS.map((item) => (
                <div
                  key={item.rank}
                  className={`grid grid-cols-12 items-center p-3 border rounded-lg ${
                    item.highlight
                      ? "bg-primary/10 border-primary/40"
                      : "bg-background border-border"
                  }`}
                >
                  <div className="col-span-2 font-mono font-bold text-muted-foreground">#{item.rank}</div>
                  <div className={`col-span-7 font-medium truncate pr-2 ${item.highlight ? "text-primary" : "text-foreground"}`}>
                    {item.name}
                  </div>
                  <div
                    className={`col-span-3 text-right font-mono font-bold ${
                      item.highlight
                        ? "text-primary"
                        : item.ret.startsWith("-")
                          ? "text-destructive"
                          : "text-positive"
                    }`}
                  >
                    {item.ret}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="bg-card border border-card-border rounded-xl shadow-xl p-6 md:p-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-6">
            <h3 className="text-lg font-bold">Leaderboard</h3>
            <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded w-fit shrink-0">Illustrative example</span>
          </div>

          <div className="h-[320px] md:h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={SAMPLE_CURVES} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                <XAxis dataKey="day" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                  domain={[960, 1240]}
                  tickFormatter={(v) => `$${v}`}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: "hsl(var(--card))", borderColor: "hsl(var(--border))", borderRadius: "8px" }}
                />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 12 }} />
                <Line type="linear" dataKey="deepseek" name="DeepSeek V4 Pro" stroke={LINE_COLORS.deepseek} strokeWidth={3} dot={false} />
                <Line type="linear" dataKey="claude" name="Claude Sonnet 4.6" stroke={LINE_COLORS.claude} strokeWidth={2} dot={false} />
                <Line type="linear" dataKey="gpt" name="GPT-5.5" stroke={LINE_COLORS.gpt} strokeWidth={2} dot={false} />
                <Line type="linear" dataKey="buyHold" name="Buy & Hold" stroke={LINE_COLORS.buyHold} strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
                <Line type="linear" dataKey="djia" name="DJIA" stroke={LINE_COLORS.djia} strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
}
