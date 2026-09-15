import { LineChart as LineChartIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Customized,
} from "recharts";
import { useLeaderboard } from "@/lib/useLeaderboard";
import {
  chartCoverage,
  formatAxisDate,
  formatPercent,
  formatTooltipDate,
  standingsCoverage,
  type BoardSeries,
  type BoardStanding,
} from "@/lib/leaderboard";
import { frameLayout, measureTextWidth } from "@/lib/boardFrame";
import { EndpointRail } from "./EndpointRail";

/** Matches `fontSize={14}` on both axes below. The y-axis reserve is measured
 *  in it rather than guessed: `width={56}` was measured correctly against
 *  `$1030` at 11px, the tick font later moved to 14px, and four of five labels
 *  lost their leading `$` with nothing failing. */
const AXIS_TICK_FONT = "14px Inter, system-ui, sans-serif";

/** Breathing room between the widest Y tick and the plot, added to the measured
 *  text width. Recharts takes `yAxisWidth` as the whole axis band -- tick text
 *  AND its gap -- so a width of exactly the text sets the ticks flush against
 *  the curves. Local on purpose: unlike the gutter constants in `boardFrame`
 *  this one mirrors nothing in `js/leaderboard.js`, whose left axis is drawn by
 *  Chart.js and padded by Chart.js. */
const AXIS_TICK_GUTTER_PX = 12;

/** One decimal on the axis, two in the tooltip and the pills.
 *
 *  Same split screen 0 makes, for the same reason: an axis tick is a scale
 *  marker with no neighbour to match, and over a domain under eight percentage
 *  points zero decimals renders duplicate labels while two renders noise. The
 *  tooltip and the chips sit beside each other and must agree, so both are two.
 */
function axisTick(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

/** Rows Recharts can plot: one object per timestamp, one column per curve. */
function toRows(times: string[], series: BoardSeries[]) {
  return times.map((t, i) => {
    const row: Record<string, string | number | null> = { t };
    series.forEach((s) => { row[s.key] = s.values[i]; });
    return row;
  });
}

/** The plotted range, padded.
 *
 *  Derived, because the hardcoded `[960, 1240]` it replaces was a dollar domain
 *  for fabricated curves. The real board spans about -0.43% to +7.49%, which is
 *  visually flat next to nof1's -34%..+34% -- and that is the honest picture.
 *  Do not widen the padding to manufacture a fan-out that did not happen. */
function percentDomain(series: BoardSeries[]): [number, number] {
  // TRACKED IN THE SCAN, NOT SPREAD INTO Math.min/Math.max. A spread is an
  // argument-count-bounded CALL rather than a scan: nine series over the hourly
  // contest window is ~1,400 arguments today, but that count is the product of
  // the window length and the roster size, both of which live in
  // dashboard/config/leaderboard.json. A longer window past the engine's
  // argument limit throws `RangeError: Maximum call stack size exceeded` inside
  // a useMemo and takes the whole hero card down rather than degrading. One
  // pass costs nothing and has no ceiling.
  let lo = Infinity;
  let hi = -Infinity;
  series.forEach((s) =>
    s.values.forEach((v) => {
      if (v == null || !Number.isFinite(v)) return;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }),
  );
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [-0.05, 0.05];
  const pad = Math.max((hi - lo) * 0.12, 0.005);
  return [lo - pad, hi + pad];
}

/** ONE TEMPLATE, TWO CONSUMERS, and it has to stay one.
 *
 *  The header row and the data rows are separate elements with independent
 *  class strings, so a column added to one and not the other unaligns the table
 *  from its own header -- silently, because both halves still render. The /app
 *  original cannot have that bug: styles.css:8667-8671 joins the two selectors
 *  into a single rule. This constant is that rule. Do not inline it into either
 *  consumer "for readability".
 *
 *  The track widths are the /app HERO card's (styles.css:8669), not the
 *  dashboard tile's. The tile is a third of a screen wide and gives the value
 *  column 72px, which the comment beside it records as too narrow for the words
 *  "Ending value"; this card is two-thirds of the hero, same as that one.
 *
 *  BELOW `sm` THE TABLE IS THREE COLUMNS, and the tracks change with the cells.
 *  `hidden` is `display: none`, so a hidden cell stops occupying its grid
 *  column -- but the TRACK survives, and two dead rails at 96px and 62px take
 *  158px of a 285px card. The narrow template drops them, so rank, name and
 *  return get the whole width. Which two columns go is the answer to "what does
 *  a phone reader lose least by losing": the ranking and the return are the
 *  claim, the ending value restates the return against a known base, and Sharpe
 *  is the one number that needs a caption to mean anything. */
const RANK_GRID =
  "grid grid-cols-[26px_minmax(0,1fr)_78px] sm:grid-cols-[26px_minmax(0,1.4fr)_96px_78px_62px] gap-2 items-center";

/** Gold, silver, bronze on a near-black foreground -- the literal hex from
 *  styles.css:8128-8130, because these three are a recognised visual signature
 *  rather than theme colours: a visitor who signs up meets the same three
 *  badges on the signed-in board. `--positive`/`--destructive` stay themed for
 *  the same reason in reverse; those are semantic. */
const MEDAL_CLASS = [
  "bg-[#fbbf24] text-[#041018]",
  "bg-[#94a3b8] text-[#041018]",
  "bg-[#d97706] text-[#041018]",
];

/** `>= 0` IS GREEN, matching /app's `ret >= 0 ? 'positive' : 'negative'`
 *  (home-page.js:1863) -- deliberately NOT the `> 0` that decides the `+`
 *  prefix in `formatPercent`. The two thresholds look like a contradiction and
 *  are not: an exactly-flat run is not a loss, so it is not painted as one, but
 *  it also did not gain, so it is not signed as one.
 *
 *  READ OFF THE NUMBER, NOT OFF `ret.startsWith("-")` as the chip strip did.
 *  `ret` is `'—'` (U+2014) when the return is non-finite, and an em dash does
 *  not start with a hyphen-minus -- so the strip painted a MISSING return in
 *  the same green it paints a profit. The third state gets its own muted tone
 *  here instead. */
function returnTone(row: BoardStanding): string {
  if (!Number.isFinite(row.cumulativeReturn)) return "text-muted-foreground";
  return row.cumulativeReturn < 0 ? "text-destructive" : "text-positive";
}

/**
 * The hero's right-hand card. Deliberately compact: it exists so the board is
 * on screen before any scroll, not to replace the full standings under
 * `#race`. Chart first, then the standings — a visitor should see the shape
 * before they read a single number.
 *
 * The curves are the LIVE Competition board, the same one the signed-in Home
 * screen draws and selected by the same rule: every model entry plus exactly two
 * reference baselines. Seven model curves with nothing to judge them against is
 * the failure that rule exists to prevent, and it is no less true here than on
 * screen 0.
 */
export function BoardPreview() {
  const board = useLeaderboard();
  const chartRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  // The gutter is a FRACTION of the rendered width, so the width has to be
  // observed. Recharts' own <ResponsiveContainer> knows it but does not hand it
  // to the parent, and `margin` is a prop on <LineChart>, which is the parent's
  // to set.
  useEffect(() => {
    const el = chartRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) setSize({ width: rect.width, height: rect.height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const data = board.status === "ready" ? board.data : null;
  const series = data?.series ?? [];
  const standings = data?.standings ?? [];
  // A 200 is not a board. `get_leaderboard` skips any strategy with no cached
  // run and still answers 200, so an empty payload and a baselines-only one
  // are ordinary SUCCESSFUL responses that `board.status` cannot tell apart
  // from a full board -- see chartCoverage's own note.
  const coverage = chartCoverage(series);
  // A SECOND, GENUINELY DIFFERENT QUESTION, not a convenience alias.
  // `chartCoverage` reads `series` and `standingsCoverage` reads `standings`,
  // and the two sets can disagree: `buildBoardData` pushes every selected entry
  // to `standings` and only reaches `series.push` past
  // `if (!values.some(v => v != null)) return`, so a model with no drawable
  // curve is in one and not the other. Answering the table's branch with the
  // chart's rule is how this card came to print "No AI model results came back"
  // above a strip listing seven models — the caption's own note describes that
  // failure from the other side. The rule moved here from Race.tsx with the
  // table it governs.
  const tableCoverage = standingsCoverage(standings);

  // NULL PROTOTYPE, not `Object.fromEntries`. Not a live bug today, and worth
  // being exact about which: both readers index this by a key that is already
  // known to be present -- `series[].key` below, and `String(item.dataKey)` in
  // EndpointRail, which Recharts took from those same series -- so the subset
  // note under this comment is what keeps every lookup an own property, and
  // the prototype is never consulted.
  //
  // It is the CONSEQUENCE OF THAT NOTE BEING WRONG that this changes. On a
  // plain object a miss does not read as a miss: `constructor`, `toString` and
  // `valueOf` answer from Object.prototype with a function, and the rail's
  // `?? ""` cannot catch it because a function is not nullish -- so a `series`
  // entry that ever escapes `standings` stops being a blank pill and becomes a
  // stringified function, measured into the gutter as a label. That converts a
  // future invariant break from visible-and-obvious into rendered-and-wrong,
  // for a roster whose entry_ids come from config rather than from code. A
  // dictionary with no prototype has nothing to inherit, so the miss stays a
  // miss, and it costs one line to buy both readers out of the question.
  const valueByKey = useMemo(() => {
    const byKey: Record<string, string> = Object.create(null);
    for (const s of standings) byKey[s.key] = s.ret;
    return byKey;
  }, [standings]);

  // MEASURED OVER `series`, NOT `standings`, because the rail draws `series`.
  // The two are not the same set and never can be: `buildBoardData` pushes
  // every selected entry to `standings` unconditionally and only reaches
  // `series.push` past `if (!values.some(v => v != null)) return`, so a
  // curve-less model is in one and not the other -- the same asymmetry the
  // caption's `chartCoverage` note describes. Measuring `standings` therefore
  // reserved gutter for pills the rail never paints, and paid for it twice: the
  // plot lost width to a phantom label, and `boardLabelBlockWidth` could push
  // the floor past BOARD_GUTTER_MAX_FRACTION and degrade the WHOLE rail to
  // arrow-only -- dropping the labels of curves that would have fitted, because
  // of a name belonging to a curve that does not exist. Every `series` entry
  // has a `standings` row (the subset runs that way, not the other), so the
  // lookup below is total.
  const frame = useMemo(
    () =>
      frameLayout({
        width: size.width,
        height: size.height,
        labels: series.map((s) => ({ name: s.name, value: valueByKey[s.key] })),
      }),
    [size.width, size.height, series, valueByKey],
  );

  const domain = useMemo(() => percentDomain(series), [series]);
  const yAxisWidth = useMemo(() => {
    const widest = Math.max(
      measureTextWidth(axisTick(domain[0]), AXIS_TICK_FONT),
      measureTextWidth(axisTick(domain[1]), AXIS_TICK_FONT),
    );
    return Math.ceil(widest) + AXIS_TICK_GUTTER_PX;
  }, [domain]);

  const rows = useMemo(() => toRows(data?.times ?? [], series), [data, series]);

  return (
    // `data-testid` ON THE ROOT BECAUSE THE MEASUREMENT PASS ALREADY QUERIES IT.
    // dashboard/scripts/verify_chart_first_layout.py selects
    // `[data-testid="board-preview"]` and, finding nothing, falls back to
    // guessing the card by its `.rounded-xl` class -- a fallback that goes stale
    // in SILENCE, because a class-based guess keeps matching some element and
    // keeps reporting a height for it. The reserves derived below are checked
    // against whatever that guess returns, so the one script able to catch an
    // over-optimistic reserve was measuring an element nobody had pinned.
    <div
      className="bg-card border border-card-border rounded-xl shadow-2xl overflow-hidden flex flex-col"
      data-testid="board-preview"
    >
      <div className="px-5 pt-5 pb-4 border-b border-border">
        {/* WRAPS, and the chip may not out-size the row. Both halves are one
            fix for one measured defect, and it is the window label above that
            caused it: "Illustrative example" was 19 characters, "Competition
            window · 2026-04-15 → 2026-05-15" is 44, and the chip carried
            `shrink-0`. At 390px the chip's max-content width is 332.8px inside
            a 285px row, so it ran 38.8px past the card's right edge — and the
            card is `overflow-hidden`, so the window's end date was simply cut
            off. The same non-shrinking chip squeezed the <h2> beside it to
            width ZERO, which still rendered 112px tall (four lines of nothing)
            and put 112px of pure damage into the reserve measured below.
            Both were invisible to every guard: no scrollbar, no ellipsis,
            nothing failing — the clipping failure this card has now shipped
            twice. Measured after: title 285px wide and 56px tall, chip 285px
            and wrapped to two lines, nothing past the card edge. */}
        <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-2 mb-2">
          <h2 className="text-xl font-bold flex items-center gap-2 min-w-0">
            <LineChartIcon className="w-5 h-5 text-primary shrink-0" aria-hidden="true" />
            Where the AI models stand
          </h2>
          {/* Was "Illustrative example". The data is no longer illustrative, and
              that label on real numbers is its own false claim. What replaces it
              is the window the chart actually draws, off the payload -- so the
              chip is now a provenance statement rather than a disclaimer, and it
              is what keeps the forward arrow below from reading as a claim that
              this window is still running. */}
          <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded max-w-full">
            {data?.windowLabel ? `Competition window · ${data.windowLabel}` : "Competition window"}
          </span>
        </div>
        {/* One line at this card width, and that is load-bearing: the chart's
            clamp subtracts this bar's height. Two lines here invalidates the
            reserves below and the card goes half-visible without anything
            failing. */}
        <p className="text-sm text-foreground/65 leading-relaxed">
          {/* SCOPED TO THE CHART, because `chartCoverage` is what drives it.
              `chartCoverage` and `standingsCoverage` are deliberately different
              questions -- a model with no drawable curve reaches `standings`
              and never reaches `series` -- so in that state this caption said
              "No AI model results came back" while the chip strip 300px below
              listed seven model names with their returns and Race showed a full
              table. The chart branch may only make a claim about the chart. */}
          {coverage === "empty"
            ? "No curves came back for this window."
            : coverage === "baselines-only"
              ? "No AI model curves were drawn — the dashed lines are buy-and-hold and the index."
              : "Each line is one AI model's return. Dashed lines are buy-and-hold and the index."}
        </p>
      </div>

      {/* The formula stays an inline style — its commas and parentheses get
          mangled by Tailwind's arbitrary-VALUE parser — while the one number
          that has to change per breakpoint rides an arbitrary PROPERTY, which
          does take a responsive prefix.

          ── RE-DERIVED when the chip strip became the ranking table ──────────
          Both numbers moved, because the block under the chart changed height
          and these reserves are nothing but that height plus the card's top.
          The rule is unchanged: `reserve = ceil10(cardTop + nonChart) + 10`,
          taken at the NARROWEST width of each band with the board READY.

          What left, and what arrived, at each band:

            lg+ (measured at 1024x768, where the strip took FOUR rows)
              out: strip 120 (4 rows x 24 + 3 x 8) + caption 32 = 152
              in : caption 20 + head 26.4 (14.4 + mt-3 12) + list 156
                   (148 + mt-2 8) = 202.4
              nonChart 313.75 - 152 + 202.4 = 364.15
              reserve = ceil10(136 + 364.15) + 10 = 520

            below (measured at 360x800, where the strip took EIGHT rows)
              out: strip 248 (8 x 24 + 7 x 8) + caption 52 = 300
              in : caption 40 + head 26.4 + list 156 = 222.4
              nonChart 583.25 - 300 + 222.4 = 505.65
              reserve = ceil10(132 + 505.65) + 10 = 650  (unchanged)

            THE TYPE BUMP COST THE ROWS NOTHING AND THE HEAD 1.2px. Raising the
            row from 13.5px to 16px moved no height at all, because the row is
            as tall as its 22px badge and 16 x 1.35 = 21.6 is still under it
            (see the badge-dominance note below) -- the list block is 156 on
            both sides of the change. Only the head moved, 13.2 -> 14.4, and
            1.2px is enough to cross a ceil10 boundary at lg (500.15 rounds up
            to 510, +10 = 520) and not enough to cross one below (637.65 rounds
            up to 640 either way). A one-band move out of a 1.2px input is
            exactly what the trailing +10 exists to absorb, which is why only
            one of the two constants here changed.

            THE CAPTION WRAPS AT 360 AND DOES NOT AT 1024, which is why the two
            bands do not share an `in` figure. "Return over the competition
            window, hour by hour." is ~317px of 14px Inter against a 272px card
            interior at 360 (360 - 48 px-6 - 40 px-5), so it takes two lines
            there and one from about 406px up. An earlier draft of this block
            claimed the new content "does not wrap" and reused the lg figure at
            both bands; it reached 650 anyway, because the same 20px was also
            missing from the `out` side -- a cancelling pair, which is the worst
            way for a derivation to be right. Both sides are stated at their
            real height now, so re-measuring one of them cannot introduce 20px
            of error into a constant whose step is 10.

            THE HEAD'S LINE-HEIGHT IS PINNED, NOT INHERITED. `text-[12px]`
            compiles to `font-size: 12px` and nothing else, so the row would
            otherwise take Tailwind preflight's `html { line-height: 1.5 }` and
            stand 18px tall against the 14.4 this derivation assumes -- 3.6px,
            which is three times the 1.2px that already moved the lg constant a
            full ceil10 step. `leading-[1.2]` on that row makes 14.4 a property
            of the markup rather than of a default that can change under it.
            The same reasoning is why the ROW below is `text-[16px]` and not
            `text-base`: the two set the same font-size, but `text-base` also
            sets `line-height: 1.5rem` = 24px -- past the 22px badge, so it
            would silently make every row 30px instead of 28 and put the list
            18px over the height this block derives for it.

          THE FIT IS NOW FLAT AT 19.85px RATHER THAN ONE TIGHT POINT, and the
          list clamp is what buys that. Card bottom at >=lg is

            136 + 208.15 + (list + 8) + chart
              list  = clamp(96, 100dvh - 632, 148)
              chart = clamp(260, 100dvh - 520, 520)

          and from 728 to 1040 tall EXACTLY ONE of those two clamps is in its
          linear middle while the other sits on a bound, so the sum tracks the
          viewport 1:1 and the slack is a constant 19.85px:

            768  list 136, chart 260 (floored) -> 748.15   19.85 above the fold
            780  list 148 (capped), chart 260  -> 760.15   19.85
            800  list 148, chart 280           -> 780.15   19.85
            1040 list 148, chart 520 (capped)  -> 1020.15  19.85
            1080 list 148, chart 520           -> 1020.15  59.85

          1024x768 is therefore no longer the one viewport the card only just
          clears; it clears by the same margin as every other. RAISING THE
          RESERVE ALONE COULD NOT HAVE DONE THIS: at 768 the chart is already on
          its 260 floor, so the reserve is not in the expression there at all,
          and the only lever at that viewport is the list's own clamp. That is
          why 620 became 632 (768 - 632 = 136, one row and change less list)
          while the base reserve did not move. Below 728 both clamps sit on
          their lower bounds, the expression stops tracking the viewport, and
          the card hangs again -- that band belongs to the 260px-floor note
          further down and always has.

          ⚠ DERIVED ARITHMETICALLY FROM THE MEASURED FIGURES ABOVE, NOT
          RE-MEASURED IN A BROWSER. The two inputs that carry over unchanged —
          cardTop and the header block — are measurements; the strip heights
          removed and the table height added are computed from the pitches this
          file and styles.css already state (24px row + 8px gap; 16px/1.35
          text; 22px rank badge + 3px padding-block top and bottom = 28px
          row). THE ROW IS AS TALL AS ITS BADGE, NOT AS ITS TEXT: the row is a
          grid with items-center, so its height is the tallest cell, and the
          22px badge beats the 21.6px line of 16px/1.35 text. That 0.4px of
          headroom is the whole reason the readability bump was free, and it is
          equally the reason the next one will not be: 17px/1.35 is 22.95 and
          every row grows. Nine rows are
          therefore 9 x 28 + 8 x 8 = 316px, and the 148px viewport shows four
          rows and a sliver of the fifth. RE-MEASURE at 1024x768 and 360x800 at
          the next opportunity and correct these two numbers if they disagree.
          The failure mode is a silently half-visible card, not a broken build,
          so nothing will tell you.

          THE TABLE IS WHY THIS IS NOW STABLE. The old note below records two
          reserves that had to be re-derived every time the strip's row count
          moved — five entries became nine, four rows became five between 1024
          and 1279, and a number measured at 1440 was wrong for the whole band
          it governed. A fixed-height scrolling list has no row count: the
          roster can grow to twenty and this block stays 201.2px tall. Adding a
          COLUMN still costs nothing; changing `max-h-[148px]`, the row's font
          size or its padding is what forces a re-derivation.

          ── The original derivation, kept because it is the measured half ────

          TWO RESERVES, BOTH MEASURED, because the card's non-chart height is
          not one number: beside the copy at >=lg it is one thing, stacked at
          390px wide the title, the chip and the caption all wrap. One constant
          cannot serve both, and the desktop one applied to a phone put the card
          77px past the fold.

          The figures the arithmetic above starts from:

            lg+   460 = ceil10(136 + 313.75 @1024x768) + 10 -> 10.25px slack
            below 730 = ceil10(132 + 583.25 @360x800)  + 10 -> floor-bound

          The trailing +10 is not padding-by-taste: rounding alone left 0.25px
          of fold slack at 1024, which is a number that survives one browser and
          no other.

          MEASURE THE lg RESERVE AT 1024, NOT AT 1440. This is what an earlier
          390 got wrong and what nothing caught: `lg:` binds from 1024 up, but
          390 was derived at 1440 where nonChart is 249.75. Between 1024 and
          1279 the chip strip took FOUR rows instead of three and nonChart was
          313.75, so the card hung below the fold across that whole band —
          every 1280-wide-and-under laptop — while the 1280+ viewports the
          number was checked against passed with room to spare.

          THE 260px FLOOR, NOT THE RESERVE, IS WHAT BINDS ON A PHONE, and no
          value here can change that: the card still needs more than a 844px
          viewport once the floor is in, so its last rows sit below the fold at
          every reserve. Dropping the floor to ~183 is the only thing that would
          pull it up, and that trades the chart the hero exists to show for its
          own fallback key: the chart itself already ends well above the fold
          there. Left as derived deliberately. The table makes this case
          strictly better than the strip did — 504.45 of non-chart against the
          old 583.25, so ~79px less of the card hangs — but better is not
          fixed, and the floor is still what decides.

          RE-DERIVE BOTH AGAIN if the caption, the title, the table head or the
          list's `max-h` changes height, and re-derive at the NARROWEST width of
          each band. The failure mode is a silently half-visible card, not a
          broken build. */}
      <div
        ref={chartRef}
        className="w-full px-3 pt-4 [--board-chart-reserve:650px] lg:[--board-chart-reserve:520px]"
        style={{
          height: "clamp(260px, calc(100dvh - var(--board-chart-reserve)), 520px)",
        }}
      >
        {board.status === "loading" ? (
          // Deliberate, not a stall. Render's free tier cold-starts in 30-60s,
          // so this is what the first visitor of the day sees.
          <div className="h-full w-full rounded-lg bg-muted/40 animate-pulse" aria-hidden="true" />
        ) : board.status === "error" ? (
          // A chart-shaped message that NAMES the failure. Explicitly not a
          // permanent shimmer and explicitly not a fallback to sample curves:
          // either would make "the backend is down" and "the backend is fine"
          // render near-identically.
          <div className="h-full w-full rounded-lg border border-border bg-muted/20 flex flex-col items-center justify-center gap-2 px-6 text-center">
            <p className="text-sm text-foreground/80">The leaderboard didn&apos;t load.</p>
            <p className="text-xs font-mono text-muted-foreground">{board.message}</p>
            <p className="text-xs text-muted-foreground">
              The board itself is fine — reload to try again.
            </p>
          </div>
        ) : coverage === "empty" ? (
          // A 200 that carried nothing to draw. Without this branch the card
          // rendered its whole frame over it -- a percent axis labelled
          // -5.0%..5.0% off percentDomain's hardcoded fallback, a scale no run
          // produced, under the axis arrow, the title, the window chip and a
          // caption naming the competition window. Confident, silent, wrong.
          // The fix is to SAY the board is empty; substituting curves is the
          // bug this card exists to remove.
          <div className="h-full w-full rounded-lg border border-border bg-muted/20 flex flex-col items-center justify-center gap-2 px-6 text-center">
            <p className="text-sm text-foreground/80">The board came back empty.</p>
            <p className="text-xs text-muted-foreground">
              The request succeeded and carried no curves. Nothing here is a result — reload to
              try again.
            </p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows} margin={{ top: 4, right: frame.gutter, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
              <XAxis
                dataKey="t"
                stroke="hsl(var(--muted-foreground))"
                fontSize={14}
                tickLine={false}
                axisLine={false}
                minTickGap={48}
                tickFormatter={formatAxisDate}
              />
              <YAxis
                stroke="hsl(var(--muted-foreground))"
                fontSize={14}
                tickLine={false}
                axisLine={false}
                domain={domain}
                width={yAxisWidth}
                tickFormatter={axisTick}
              />
              {/* `labelFormatter` is the tooltip HEADER and it is a separate
                  wire from the axis: recharts renders the raw category value
                  -- the `t` column, i.e. `timeKey()` output -- unless this
                  prop is given, and `XAxis.tickFormatter` never reaches it. So
                  the axis fix left the hero printing `2026-04-15T14:00` above
                  an axis correctly reading `Apr 15`. Same string /app shows. */}
              <Tooltip
                contentStyle={{ backgroundColor: "hsl(var(--card))", borderColor: "hsl(var(--border))", borderRadius: "8px" }}
                labelFormatter={formatTooltipDate}
                formatter={(value: number | string) =>
                  formatPercent(Number(value), 2)
                }
              />
              {series.map((s) => (
                <Line
                  key={s.key}
                  type="linear"
                  dataKey={s.key}
                  name={s.name}
                  stroke={s.color}
                  strokeWidth={s.isBaseline ? 1.5 : 2}
                  strokeDasharray={s.dash}
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
              ))}
              {/* Last, so it paints over the curves. `valueByKey`/`drawLabels`/
                  `gap` reach the rail because Recharts clones a <Customized>
                  child with the chart's own props and state spread OVER the
                  element's -- so an extra prop must not collide with a chart
                  prop or state key. These three do not. */}
              <Customized
                component={EndpointRail}
                valueByKey={valueByKey}
                drawLabels={frame.drawLabels}
                gap={frame.gap}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="px-5 pb-5 pt-3">
        {/* Names the axis directly above it, and only that. The axis is percent
            now — see the plan's §6 — so a caption about "account value" would
            describe a chart that is not there. By the same rule it is withheld
            when nothing was drawn: on an empty 200 there IS no axis, and a
            caption dating one is the confident-frame-over-nothing claim the
            empty branch above exists to remove.

            IT MOVED ABOVE THE TABLE, and the move is what the sentence is for
            rather than tidying: it names THE AXIS, and an axis caption with a
            five-column ranking between it and the axis names whatever happens
            to sit above it instead. Do not push it back to the foot of the
            card. */}
        {coverage === "empty" ? null : (
          <p className="text-sm text-foreground/65">
            Return over the competition window, hour by hour.
          </p>
        )}

        {/* THE SIGNED-IN BOARD'S RANK TABLE, PORTED. It replaces a wrapping
            chip strip that carried the same colour-to-name link one entry per
            line, with no number on it but the return.

            It inherits the strip's two jobs and keeps both. The chart ships no
            Recharts <Legend> — a nine-item one wraps to three rows at this
            width and pushes the plot area down — so this is still the only
            thing linking a curve's colour to a name; and it is still the
            fallback when the endpoint rail declines to draw, on a card too
            narrow or too short for the gutter labels or under a Recharts
            internal that moved beneath EndpointRail. Every row carries the
            swatch, so neither job was traded for the columns.

            NINE ROWS, NOT SIX: every model AND the two reference baselines,
            tagged. /app's rank list filters to models and is right to, because
            its chart is a different chart. This one draws nine curves, so a
            models-only table leaves buy_hold_djia and djia_index as two unnamed
            dashed lines — the dangling reference the strip existed to prevent.
            The reasoning is Race.tsx's, which ranked the same mixed field for
            the same reasons; it moved here with the table. Do not add a filter
            that drops them back out.

            AND THAT IS WHY THE COLUMN SAYS "Contender". On the live board the
            #1 row IS a benchmark — most of the models lost to buy-and-hold — so
            a header reading "AI Model" would line up the column name, the row
            accent and the section heading behind the claim that the passive
            index is the leading AI model. /app's header can say "AI Model"
            because /app's list contains only models. */}
        {/* THE HEAD IS INSIDE THE STATUS BRANCH, not above it. It used to sit
            outside every branch, so the five-column frame "# | Contender |
            Ending value | Return | Sharpe" was painted over the loading
            shimmer, over the error message and over the empty-board message —
            a confident frame with nothing under it, which is precisely the
            failure the branches inside the <ol> below exist to remove and which
            lib/leaderboard.ts's own docstring bans by name ("Race drew its
            Rank/AI model/Return header over zero rows").

            `tableCoverage`, NOT `coverage`. The two answer genuinely different
            questions — see the note where they are computed — and the table's
            branch must be decided by the table's rule, or this head reappears
            over a list that is itself printing "the standings came back empty".

            WITHHOLDING IT MAKES THE CARD SHORTER THAN THE DERIVATION BELOW
            ASSUMES, which is the safe direction: the reserves are computed with
            the 26.4px head block present, so a state that drops it has 26.4px
            MORE fold slack, never less. */}
        {board.status === "ready" && tableCoverage !== "empty" ? (
          <div
            className={`${RANK_GRID} mt-3 text-[12px] leading-[1.2] text-muted-foreground px-0.5`}
            data-testid="board-rank-head"
          >
            <span>#</span>
            <span>Contender</span>
            <span className="hidden sm:block text-right">Ending value</span>
            <span className="text-right">Return</span>
            <span
              className="hidden sm:block text-right"
              title="Risk-adjusted return, annualized from hourly results."
            >
              Sharpe
            </span>
          </div>
        ) : null}

        {board.status === "ready" && tableCoverage === "baselines-only" ? (
          // The reachable half, and the one that looks plausible: the LLM
          // entries carry `auto_compute: false` while the baselines
          // auto-recompute, so a contest-window edit misses cache on all twelve,
          // rebuilds the two baselines and never rebuilds the models. The
          // Contender header and the per-row tag each stop over-claiming on
          // their own; what neither can say is that the ABSENCE was not
          // intended. That is this line, and it is the one thing allowed to grow
          // this block past the height derived above — a degraded board may
          // spend a few pixels of fold to say so.
          <p className="mt-2 text-xs text-muted-foreground">
            No AI model results came back this time — the rows below are the reference baselines
            only.
          </p>
        ) : null}

        {/* THE SCROLL IS THE DESIGN, and `max-h` + `overflow-y-auto` is only
            half of it; the reserve derivation on the chart container above is
            the other half. Nine rows at this pitch run ~316px, and the card
            cannot spend that beside a chart this size without hanging below the
            fold — the exact failure those reserves exist to prevent. A fixed
            viewport onto a scrolling list spends a known 148px instead, and the
            148 is not taste: it is the number the reserves were re-derived
            against, so moving it means re-deriving both.

            `list-none` is belt-and-braces over Tailwind's preflight: this is an
            <ol>, and a reset that stops applying puts numerals in front of rows
            that already carry a rank badge.

            THE HEIGHT IS A CLAMP, NOT A CONSTANT, and the third number is the
            one the reserve above was derived against. 148px is what the list
            takes wherever the card fits; below that it gives height back rather
            than pushing the card through the fold.

              max-h: clamp(96px, calc(100dvh - 620px), 148px)

            620 is chosen so the shrink begins exactly where the fit runs out:
            at 100dvh = 768 the expression is 148 (the full list), and below 768
            it falls, which is the band where the chart has already hit its
            260px floor and can give nothing back. It buys a real range --
            re-running the same arithmetic at 716px of viewport gives a card of
            706.95px against 716, where a fixed 148 would have given 758.95
            against 716 and hung 43px under. The floor of 96px is ~3 rows: below
            that the list stops being a list, so the card is allowed to exceed
            the fold instead, exactly as it does today on a phone.

            IT ALSO MAKES THE DERIVED RESERVES FAIL SAFE. Those two constants
            were computed arithmetically rather than measured in a browser, and
            148 is the largest value this expression can take -- so if either is
            a little optimistic, the error lands on a viewport where this clamp
            is already handing height back, rather than on one where nothing is.

            NOT `aria-hidden`, which is what /app puts on its own header row --
            but be precise about what that buys, because it is less than it
            looks. The header is a plain <div> with no `role="columnheader"` and
            no aria relationship to the rows, so it does not LABEL them; it adds
            one announcement of the column names ahead of the list, which helps
            a reader going through the page linearly and does nothing for one who
            jumps straight into the list by rotor. Real table semantics
            (role="table"/"row"/"columnheader"/"cell") would label them, and are
            the honest fix if this needs to be better. Un-hiding costs nothing
            and helps the common case, so it stays; it is not a solved problem.

            `role="list"` IS NOT REDUNDANT ON AN <ol>. Safari/VoiceOver drops the
            implicit list role -- and with it the item count -- as soon as
            `list-style` is `none`, which `list-none` sets. Without it the
            aria-label above announces on an element VoiceOver no longer treats
            as a list, which is the platform most of this page's mobile screen
            reader traffic is on. */}
        <ol
          className="board-rank-scroll mt-2 max-h-[clamp(96px,calc(100dvh-632px),148px)] overflow-y-auto overflow-x-hidden list-none m-0 p-0 flex flex-col gap-2"
          role="list"
          aria-label="Competition standings"
          data-testid="board-rank-list"
        >
          {board.status === "loading" ? (
            // Deliberate, not a stall — the same 30-60s free-tier cold start the
            // chart's shimmer covers.
            <li className="px-0.5 py-4 text-sm text-muted-foreground">Loading the standings…</li>
          ) : board.status === "error" ? (
            // Names the failure. Absent and broken must not render the same.
            <li className="px-0.5 py-4 text-sm text-muted-foreground">
              The standings didn&apos;t load ({board.message}). Reload to try again.
            </li>
          ) : tableCoverage === "empty" ? (
            <li className="px-0.5 py-4 text-sm text-muted-foreground">
              The standings came back empty. The request succeeded and carried no entries —
              nothing here is a result. Reload to try again.
            </li>
          ) : (
            standings.map((item) => (
              <li key={item.key} className={`${RANK_GRID} text-[16px] leading-[1.35] py-[3px]`}>
                <span
                  className={`w-[22px] h-[22px] rounded-full grid place-items-center text-[12px] font-bold ${
                    item.rank >= 1 && item.rank <= MEDAL_CLASS.length
                      ? MEDAL_CLASS[item.rank - 1]
                      : "bg-muted text-muted-foreground"
                  }`}
                >
                  {item.rank}
                </span>
                {/* The swatch lives INSIDE this cell, never as a sixth child of
                    the grid: a sixth child shifts every later cell one column
                    right and unaligns the rows from their own header — the same
                    warning styles.css:8104-8111 carries for the same markup.
                    `min-w-0` is what lets `truncate` engage, and the tag takes
                    `shrink-0` rather than the cell taking `truncate`: a tag
                    inside a truncating block is the first thing clipped, which
                    Race.tsx shipped twice. */}
                <span className="flex items-center gap-2 min-w-0 overflow-hidden">
                  <span
                    className="w-[9px] h-[9px] rounded-full shrink-0"
                    style={{ backgroundColor: item.color }}
                    aria-hidden="true"
                  />
                  <span className="truncate font-semibold text-foreground">{item.name}</span>
                  {item.isModel ? null : (
                    <span className="shrink-0 rounded border border-border px-1 font-mono text-[10px] uppercase leading-4 tracking-wide text-muted-foreground">
                      Benchmark
                    </span>
                  )}
                </span>
                <span className="hidden sm:block text-right font-mono text-sm text-muted-foreground tabular-nums">
                  {item.endingValue}
                </span>
                <span className={`text-right font-mono font-bold tabular-nums ${returnTone(item)}`}>
                  {item.ret}
                </span>
                <span className="hidden sm:block text-right font-mono text-muted-foreground tabular-nums">
                  {item.sharpe}
                </span>
              </li>
            ))
          )}
        </ol>
      </div>
    </div>
  );
}
