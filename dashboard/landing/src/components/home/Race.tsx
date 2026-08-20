import { Medal, CalendarClock, TrendingUp, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
// No storyline import here on purpose: the Talk → Test story agent belongs to a
// backtest run report (Test.tsx), not to a board.
import { PRIMARY_LANDING_CTA } from "@/lib/cta";
// The sample rows are gone: this table and the hero card render the SAME live
// Competition board, from one fetch. Real numbers in the hero above invented
// ones here, on the same page, is worse than either alone.
import { useLeaderboard } from "@/lib/useLeaderboard";
import { boardHeadlineCounts, standingsCoverage, type BoardStanding } from "@/lib/leaderboard";

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

/** Spelled out, to match the register of the sentence they sit in. Past ten the
 *  digits are used -- the curated roster is nowhere near that, and a word list
 *  that runs out silently is worse than one that visibly stops. */
const COUNT_WORDS = [
  "No",
  "One",
  "Two",
  "Three",
  "Four",
  "Five",
  "Six",
  "Seven",
  "Eight",
  "Nine",
  "Ten",
] as const;

function countWord(n: number): string {
  return n >= 0 && n < COUNT_WORDS.length ? COUNT_WORDS[n] : String(n);
}

/** The section's opening claim, DERIVED FROM THE BOARD IT SITS BESIDE.
 *
 *  This was "Seven leading AI models traded the same days with simulated money,
 *  ranked against buy-and-hold and the index. Only one finished ahead of both."
 *  -- two hardcoded facts printed directly above a table that is now live off
 *  the same payload, with nothing holding them together. An eighth `llm_agent`
 *  entry in dashboard/config/leaderboard.json is the documented way the roster
 *  reached seven, and it would have left the sentence saying "Seven" beside
 *  eight rows; a re-run that put a second model ahead of buy-and-hold would
 *  have falsified the second half with the counter-evidence rendered beside it.
 *  On the page's most checkable claim, on the highest-traffic anonymous
 *  surface.
 *
 *  THE WORDS LIVE HERE, THE COUNTS COME FROM lib. `boardHeadlineCounts` returns
 *  numbers only, because `test_no_landing_component_claims_brokered_or_real_capital_trading`
 *  scans `components/home/*.tsx` and not `lib/` -- copy moved into the library
 *  would leave that scan, which the guard's own docstring names as the one
 *  thing this class of copy reliably does.
 *
 *  NO COUNT AND NO OUTCOME WHEN THERE IS NO BOARD. Loading, an error and a 200
 *  carrying no models all reach this with nothing to count, and a sentence
 *  asserting a tally over an empty table is the confident-frame-over-nothing
 *  shape the branches under the table exist to remove. The fallback drops to
 *  the present tense and claims neither number. */
function headlineSentence(standings: BoardStanding[]): string {
  const { models, baselines, ahead } = boardHeadlineCounts(standings);
  if (!models) {
    return "Leading AI models trade the same days with simulated money, ranked against buy-and-hold and the index.";
  }
  const subject = models === 1 ? "leading AI model" : "leading AI models";
  const against = baselines === 2 ? "buy-and-hold and the index" : "the reference baselines";
  const opening = `${countWord(models)} ${subject} traded the same days with simulated money, ranked against ${against}.`;
  if (!baselines) return opening;
  const both = baselines === 2 ? "both" : "all of them";
  if (!ahead) return `${opening} None finished ahead of ${both}.`;
  if (ahead === models) return `${opening} Every one of them finished ahead of ${both}.`;
  if (ahead === 1) return `${opening} Only one finished ahead of ${both}.`;
  return `${opening} ${countWord(ahead)} finished ahead of ${both}.`;
}

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
  // A 200 is not a board: `get_leaderboard` skips any strategy with no
  // cached run and still answers 200, so a payload with no entries -- or
  // with the two baselines and none of the seven models -- is an ordinary
  // SUCCESSFUL response that `board.status` cannot tell from a full board.
  // Rendering the Rank/Contender/Return header over that is the
  // fail-closed-is-not-fail-visible shape, on the page's most checkable
  // claim. Do not answer it with invented rows.
  const coverage = standingsCoverage(standings);
  return (
    <section id="race" className="py-24 bg-muted/20 border-y border-border scroll-mt-40">
      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-12 items-start">
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-3">What the AI models actually returned</h2>
            <p className="text-foreground/80 mb-6 text-lg">{headlineSentence(standings)}</p>
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
            {/* WRAPS, and the chip may not out-size the row — the same one fix
                BoardPreview.tsx carries, for the same measured defect, because
                this card took the same chip in the same commit and did not get
                repaired with it. "Illustrative example" was 19 characters,
                "Competition window · 2026-04-15 → 2026-05-15" is 44, and the
                chip carried `shrink-0`. Measured at 390x844: the <h3> went
                109px -> 0px WIDE while still rendering 56px tall, so its text
                overflowed under the chip's own `bg-muted` and the heading read
                "Standings" with "Competition" painted over it; the chip ran
                58.2px past the card's inner right edge; and at 360x800 the
                document gained 25px of horizontal scroll. Nothing failed — no
                scrollbar warning, no ellipsis, no console error. Do not put
                `shrink-0` back, and do not put either class behind a `lg:`
                prefix: the measurements above are all BELOW 1024. */}
            <div className="flex flex-wrap items-center justify-between mb-2 border-b border-border pb-4 gap-3">
              <h3 className="text-xl font-bold flex items-center gap-2 min-w-0">
                <Medal className="w-5 h-5 text-primary shrink-0" aria-hidden="true" />
                Competition Standings
              </h3>
              {/* Literal, not a shared constant — see the note in
                  BoardPreview.tsx: the guard counts occurrences in the minified
                  bundle. */}
              <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded max-w-full">
                {board.status === "ready" && board.data.windowLabel
                  ? `Competition window · ${board.data.windowLabel}`
                  : "Competition window"}
              </span>
            </div>
            <div className="space-y-2 mt-4">
              <div className="grid grid-cols-12 text-xs font-mono text-muted-foreground pb-2 px-2">
                <div className="col-span-2">Rank</div>
                {/* NOT "AI model". This table deliberately ranks buy_hold_djia
                    and djia_index alongside the models (see the note beside
                    `standings` above), and most of the models lost to
                    buy-and-hold -- so on the live board the `#1` row IS a
                    benchmark, rendered in the brand accent, under a header
                    naming it an AI model, below a heading reading "What the AI
                    models actually returned". Three signals all saying the
                    passive index is the leading model. "Contender" is this
                    section's own word for the mixed field (see BOARD_RULES),
                    and the per-row tag below is the table's equivalent of the
                    dash pattern the chart uses to mark the same two curves. The
                    accent stays on the true leader: that buy-and-hold won is
                    the honest, unflattering fact this card exists to show. */}
                <div className="col-span-7">Contender</div>
                <div className="col-span-3 text-right">Return</div>
              </div>
              {board.status === "loading" ? (
                <p className="px-2 py-6 text-sm text-muted-foreground">Loading the board…</p>
              ) : board.status === "error" ? (
                // Names the failure. Absent and broken must not render the same.
                <p className="px-2 py-6 text-sm text-muted-foreground">
                  The standings didn&apos;t load ({board.message}). Reload to try again.
                </p>
              ) : coverage === "empty" ? (
                <p className="px-2 py-6 text-sm text-muted-foreground">
                  The standings came back empty. The request succeeded and carried no entries —
                  nothing here is a result. Reload to try again.
                </p>
              ) : (
                <>
                  {coverage === "baselines-only" ? (
                    // The reachable half, and the one that looks plausible: all
                    // seven LLM entries carry `auto_compute: false` while the
                    // baselines auto-recompute, so a contest-window edit misses
                    // cache on all twelve, rebuilds the two baselines and never
                    // rebuilds the models. The Contender header, the per-row
                    // benchmark tag and the derived headline each stop
                    // over-claiming on their own now; what none of them can say
                    // is that the ABSENCE was not intended. That is this line:
                    // these two rows are all that came back, not the field.
                    <p className="px-2 pb-3 text-sm text-muted-foreground">
                      No AI model results came back this time — the rows below are the reference
                      baselines only.
                    </p>
                  ) : null}
                  {standings.map((item, index) => (
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
                      {/* `min-w-0` + `truncate` on the NAME and `shrink-0` on
                          the tag, not `truncate` on the cell: a tag inside a
                          truncating block is the first thing clipped, and this
                          card has shipped exactly that failure twice with no
                          scrollbar, no ellipsis and nothing failing. */}
                      <div className="col-span-7 flex items-center gap-2 min-w-0 pr-2">
                        <span
                          className={`font-medium truncate ${
                            index === 0 ? "text-primary" : "text-foreground"
                          }`}
                        >
                          {item.name}
                        </span>
                        {item.isModel ? null : (
                          <span className="shrink-0 rounded border border-border px-1 font-mono text-[10px] uppercase leading-4 tracking-wide text-muted-foreground">
                            Benchmark
                          </span>
                        )}
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
                  ))}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
