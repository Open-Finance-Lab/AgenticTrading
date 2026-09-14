import { Medal, CalendarClock, TrendingUp, ShieldCheck } from "lucide-react";
import { LandingCTA } from "./LandingCTA";
// No storyline import here on purpose: the Talk → Test story agent belongs to a
// backtest run report (Test.tsx), not to a board.
// Still reads the board, and now for one sentence rather than a table. The
// note this replaces said "this table and the hero card render the SAME live
// Competition board, from one fetch" -- which was the fix for a worse state
// (invented rows here under real numbers there) and became the argument for
// deleting the table outright: two renderings of one payload, four screens
// apart, is a duplication whichever way they agree. The hero card kept the
// rows; this section kept the claim about them, and the claim is still derived
// from the same single fetch.
import { useLeaderboard } from "@/lib/useLeaderboard";
import { boardHeadlineCounts, type BoardStanding } from "@/lib/leaderboard";

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
  // The ONLY thing this section still derives from the board, and the whole
  // reason it still reads it: the sentence below counts the models, the
  // baselines, and how many beat every baseline.
  //
  // THE TABLE THAT USED TO SIT HERE IS GONE, moved into the hero card at the
  // top of the page (BoardPreview.tsx) where it replaced a chip strip. It was
  // never a second board -- it was the same nine rows off the same fetch,
  // four screens apart from the chart they describe. The reasoning that kept
  // its baselines on the board travelled with it and is restated there; so did
  // its coverage branches, its per-row `Benchmark` tag, and the "no AI model
  // results came back" line. Nothing was dropped in the move except the
  // duplication.
  const standings = board.status === "ready" ? board.data.standings : [];
  return (
    <section id="race" className="py-24 bg-muted/20 border-y border-border scroll-mt-40">
      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-12 items-start">
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-3">What the AI models actually returned</h2>
            <p className="text-foreground/80 mb-6 text-lg">{headlineSentence(standings)}</p>
            {/* The one thing removing the table costs: the rows were also how a
                reader found the rows. The hero card is above this section, not
                below it, so this points back up rather than down — and it names
                the card by what it shows rather than by a link, because the
                board is on screen when the page opens and a visitor who
                scrolled here has already passed it. */}
            {/* NAMES ONLY WHAT SURVIVES EVERY WIDTH. The first draft read
                "with its ending value, return and Sharpe", which is false on a
                phone: BoardPreview hides both of those columns below `sm`, so
                the sentence itemised two things a mobile reader cannot find on
                the page it points at. The ranking and the benchmarks are there
                at every width. */}
            <p className="text-foreground/70 mb-6">
              Every contender is ranked in the board at the top of this page, benchmarks included.
            </p>
            {/* "Live" names the direction the board runs, not brokered execution, and
                Season 0 is a shakedown with no nightly advance deployed yet. Both are
                stated on the board's own About card; saying it here too keeps the
                landing from selling a standing that does not exist. */}
            <p className="text-xs text-muted-foreground mb-8">
              The Live Trading Leaderboard is in preview for Season 0. It has not moved forward a
              session yet, and nothing on it is a record. Season 1 is the first that counts.
            </p>
            <LandingCTA size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90" />
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
                prefix: the measurements above are all BELOW 1024.

                THE CARD'S CONTENTS CHANGED AND THE HEADER ROW DID NOT, on
                purpose: the chip is a provenance statement about the window the
                sentence's counts were taken over, which is still exactly what
                this card is about now that it holds the rules rather than the
                rows. */}
            <div className="flex flex-wrap items-center justify-between mb-2 border-b border-border pb-4 gap-3">
              <h3 className="text-xl font-bold flex items-center gap-2 min-w-0">
                <Medal className="w-5 h-5 text-primary shrink-0" aria-hidden="true" />
                How the board works
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
            {/* Three facts, in the order a sceptic asks for them. They moved out
                of the copy column and into this card when the rows left it: the
                column would otherwise have carried the whole section alone
                beside an empty half, and the rules are what the reader needs in
                order to trust the numbers one screen up. */}
            <ul className="space-y-4 mt-4 text-sm text-foreground/80">
              {BOARD_RULES.map(({ icon: Icon, text }) => (
                <li key={text} className="flex items-start gap-3">
                  <Icon className="w-4 h-4 text-primary mt-0.5 shrink-0" aria-hidden="true" />
                  <span>{text}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
