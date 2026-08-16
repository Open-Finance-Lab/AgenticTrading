import { LineChart as LineChartIcon } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

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
 *  agents do — and the prose beside it now states that 1-of-7 fact outright.
 *
 *  Lives here rather than in Race.tsx because the card moved into the hero: the
 *  board is the first thing on the page, not the third act's illustration. */
export const SAMPLE_CURVES = [
  { day: "7d ago", deepseek: 1000, claude: 1000, gpt: 1000, buyHold: 1000, djia: 1000 },
  { day: "5d ago", deepseek: 1061, claude: 1035, gpt: 1012, buyHold: 1018, djia: 1008 },
  { day: "3d ago", deepseek: 1094, claude: 1012, gpt: 986, buyHold: 1005, djia: 995 },
  { day: "2d ago", deepseek: 1128, claude: 1026, gpt: 972, buyHold: 1032, djia: 1014 },
  { day: "Yesterday", deepseek: 1186, claude: 1019, gpt: 991, buyHold: 1048, djia: 1022 },
  { day: "Now", deepseek: 1210, claude: 1014, gpt: 985, buyHold: 1055, djia: 1028 },
];

export const LINE_COLORS = {
  deepseek: "#a78bfa",
  claude: "#fbbf24",
  gpt: "#22d3ee",
  buyHold: "#94a3b8",
  djia: "#64748b",
} as const;

/** `swatch` is the identity link between a row and its curve. The chart ships
 *  no Recharts <Legend>: at this card's width a five-item legend wraps to two
 *  rows and pushes the plot area down, and a legend and a standings table
 *  listing the same five names is the same information twice. Colouring the
 *  rows makes the table the key.
 *
 *  Which is also why rank 1 no longer renders its name in the accent colour.
 *  The accent is cyan, DeepSeek's curve is purple, and cyan is GPT-5.5's curve
 *  — so highlighting the leader in cyan pointed at the wrong line. */
export const SAMPLE_STANDINGS = [
  { rank: 1, name: "DeepSeek V4 Pro", ret: "+21.0%", highlight: true, swatch: LINE_COLORS.deepseek },
  { rank: 2, name: "Buy & Hold", ret: "+5.5%", highlight: false, swatch: LINE_COLORS.buyHold },
  { rank: 3, name: "DJIA", ret: "+2.8%", highlight: false, swatch: LINE_COLORS.djia },
  { rank: 4, name: "Claude Sonnet 4.6", ret: "+1.4%", highlight: false, swatch: LINE_COLORS.claude },
  { rank: 5, name: "GPT-5.5", ret: "-1.5%", highlight: false, swatch: LINE_COLORS.gpt },
];

/** NOTE: every card built on these numbers spells "Illustrative example" out as
 *  a literal, deliberately, rather than sharing one exported constant.
 *  `test_landing_copy_register.py::test_illustrative_example_label_appears_at_least_twice`
 *  counts occurrences in the *shipped bundle*, and esbuild collapses a shared
 *  constant to a single string literal — so the DRY version renders the label on
 *  both cards while reading as one, and the guard drops from 3 hits to 1. Keep
 *  the duplication. */

/**
 * The hero's right-hand card. Deliberately compact: it exists so the board is
 * on screen before any scroll, not to replace the full standings under
 * `#race`. Chart first, then the top five — a visitor should see the shape
 * before they read a single number.
 */
export function BoardPreview() {
  return (
    <div className="bg-card border border-card-border rounded-xl shadow-2xl overflow-hidden flex flex-col">
      <div className="px-5 pt-5 pb-4 border-b border-border">
        <div className="flex items-start justify-between gap-3 mb-2">
          <h2 className="text-xl font-bold flex items-center gap-2 min-w-0">
            <LineChartIcon className="w-5 h-5 text-primary shrink-0" aria-hidden="true" />
            Where the AI models stand
          </h2>
          <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded shrink-0">
            Illustrative example
          </span>
        </div>
        {/* One line at this card width, and that is load-bearing: the chart's
            clamp subtracts this bar's height. Two lines here invalidates the
            390 constant below and the card goes half-visible without anything
            failing. Naming no window is also deliberate — the sample curves do
            not come from the range the old wording claimed. */}
        <p className="text-sm text-foreground/65 leading-relaxed">
          Each line is one AI model&apos;s account value. Dashed lines are buy-and-hold and the index.
        </p>
      </div>

      {/* An inline style, not an arbitrary Tailwind value: the formula's commas
          and parentheses get mangled by the arbitrary-value parser, and this
          constant is load-bearing enough to want readable in source.

          390 = the card's own non-chart height (~227px: caption bar, chip
          strip, detail line, padding) + 120px --landing-chrome-height + ~43px
          fold margin. RE-DERIVE IT if the caption or chip strip changes height.
          A shared clamp with /app was measured and rejected: it puts this card
          25-46px below the fold at four ordinary laptop heights. */}
      <div
        className="w-full px-3 pt-4"
        style={{ height: "clamp(300px, calc(100dvh - 390px), 520px)" }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={SAMPLE_CURVES} margin={{ top: 4, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
            <XAxis dataKey="day" stroke="hsl(var(--muted-foreground))" fontSize={14} tickLine={false} axisLine={false} />
            <YAxis
              stroke="hsl(var(--muted-foreground))"
              fontSize={14}
              tickLine={false}
              axisLine={false}
              domain={[960, 1240]}
              width={44}
              tickFormatter={(v) => `$${v}`}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "hsl(var(--card))", borderColor: "hsl(var(--border))", borderRadius: "8px" }}
            />
            <Line type="linear" dataKey="deepseek" name="DeepSeek V4 Pro" stroke={LINE_COLORS.deepseek} strokeWidth={3} dot={false} />
            <Line type="linear" dataKey="claude" name="Claude Sonnet 4.6" stroke={LINE_COLORS.claude} strokeWidth={2} dot={false} />
            <Line type="linear" dataKey="gpt" name="GPT-5.5" stroke={LINE_COLORS.gpt} strokeWidth={2} dot={false} />
            <Line type="linear" dataKey="buyHold" name="Buy & Hold" stroke={LINE_COLORS.buyHold} strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
            <Line type="linear" dataKey="djia" name="DJIA" stroke={LINE_COLORS.djia} strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="px-5 pb-5 pt-3">
        {/* DEMOTION, NOT DELETION. The chart ships no Recharts legend — at this
            card's width a five-item one wraps to two rows and pushes the plot
            area down — so this strip is the only thing linking a curve's colour
            to a model's name. Delete it and five unnamed lines are left. The
            full standings, with ranks, live in Race.tsx, which is the detail
            home.

            Kept in THIS FILE deliberately: test_landing_copy_register.py scopes
            its corpus to files containing the standings constant and requires a
            Recharts data key in it, and the only ones are on the line elements
            above. Splitting this strip into its own component reddens that
            guard though nothing was deleted. */}
        <div className="flex flex-nowrap items-center gap-x-4 gap-y-2 overflow-hidden text-base">
          {SAMPLE_STANDINGS.map((item) => (
            <span key={item.rank} className="flex items-center gap-2 whitespace-nowrap">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                style={{ backgroundColor: item.swatch }}
                aria-hidden="true"
              />
              <span className="font-medium text-foreground">{item.name}</span>
              <span
                className={`font-mono font-bold ${
                  item.ret.startsWith("-") ? "text-destructive" : "text-positive"
                }`}
              >
                {item.ret}
              </span>
            </span>
          ))}
        </div>
        <p className="mt-3 text-sm text-foreground/65">
          Account value over the competition window.
        </p>
      </div>
    </div>
  );
}
