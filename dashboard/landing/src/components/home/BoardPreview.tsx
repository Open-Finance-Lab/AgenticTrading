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
  type BoardSeries,
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
    <div className="bg-card border border-card-border rounded-xl shadow-2xl overflow-hidden flex flex-col">
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

          TWO RESERVES, BOTH MEASURED, because the card's non-chart height is
          not one number: beside the copy at >=lg it is one thing, stacked at
          390px wide the title, the chip and the caption all wrap AND the chip
          strip runs to several rows. One constant cannot serve both, and the
          desktop one applied to a phone put the card 77px past the fold.

          RE-DERIVED in a browser for live data, and BOTH NUMBERS MOVED. The
          rule is `reserve = ceil10(cardTop + nonChart) + 10`, measured at the
          NARROWEST width of the band with the board READY (the loading state
          is one shimmer div and measures nothing):

            lg+   460 = ceil10(136 + 313.75 @1024x768) + 10 -> 10.25px slack
            below 730 = ceil10(132 + 583.25 @360x800)  + 10 -> floor-bound

          The trailing +10 is not padding-by-taste: rounding alone left 0.25px
          of fold slack at 1024, which is a number that survives one browser and
          no other.

          MEASURE THE lg RESERVE AT 1024, NOT AT 1440. This is what the old 390
          got wrong and what nothing caught: `lg:` binds from 1024 up, but 390
          was derived at 1440 where nonChart is 249.75. Between 1024 and 1279
          the chip strip takes FOUR rows instead of three and nonChart is
          313.75, so the card hung below the fold across that whole band --
          every 1280-wide-and-under laptop -- while the 1280+ viewports the
          number was checked against passed with room to spare.

          Those two figures were "five instead of four" and 309.75 until
          2026-08-20. Both were wrong, and the second contradicted this
          comment's OWN derivation four lines above it, which has always read
          313.75 and is the number that produced the shipped 460. Re-measured
          against the live payload (nine chips, not the five this was first
          written for): 1024x768 -> 4 rows, nonChart 313.75, cardTop 136;
          1280x800 and 1440x900 -> 3 rows, nonChart 249.75. The row PITCH is
          32px (24px row + 8px `gap-y-2`), so one extra row is 32 of the 64px
          gap between the bands and the wrapping title/chip bar is the rest.
          Nothing shipped moves: ceil10(136 + 313.75) + 10 and
          ceil10(136 + 309.75) + 10 are both 460, which is exactly why a wrong
          number could sit here this long -- the constant it justifies is
          insensitive to it, so only reading the two numbers against each other
          catches it.

          THE 260px FLOOR, NOT THE RESERVE, IS WHAT BINDS ON A PHONE, and no
          value here can change that: at 390x844 the card needs 920.5px
          (132 + 528.5 + the 260 floor) against 844 of viewport, so the last
          ~77px -- the tail of the chip strip -- sits below the fold at every
          reserve. Dropping the floor to ~183 is the only thing that would pull
          it up, and that trades the chart the hero exists to show for its own
          fallback key: the chart itself already ends at y=586, well above the
          fold. Left as measured deliberately. The reserve still earns its
          value on every stacked viewport tall enough for the floor to clear
          (390x1000 fits); below that the floor decides and the
          reserve is inert. Deliberately no slack figure here: three
          independent measurements of that viewport gave 37.5, 49.5 and 69.5px,
          so the direction is solid and the number is not. Re-measure rather
          than trusting a figure written down once.

          RE-DERIVE BOTH AGAIN if the caption, the title or the chip strip
          changes height, and re-derive at the NARROWEST width of each band.
          The failure mode is a silently half-visible card, not a broken
          build. */}
      <div
        ref={chartRef}
        className="w-full px-3 pt-4 [--board-chart-reserve:730px] lg:[--board-chart-reserve:460px]"
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
        {/* DEMOTION, NOT DELETION, and now doing two jobs. The chart ships no
            Recharts <Legend> — at this card's width a nine-item one wraps to
            three rows and pushes the plot area down — so this strip is the only
            thing linking a curve's colour to a model's name. It is ALSO the
            fallback whenever the endpoint rail declines to draw: a card too
            narrow or too short for the gutter labels, or a Recharts internal
            that moved under EndpointRail. Delete it and nine unnamed lines are
            left. The full standings, with ranks, live in Race.tsx.

            WRAPS, and must. `flex-nowrap` + `overflow-hidden` silently cut
            entries off the end whenever the strip was narrower than its
            content: measured scrollWidth 910 against clientWidth 285 at 390
            (four of five chips gone, leaving one model to key five drawn
            curves), 663 at 768, 895 at 1024 — so the whole lg band and every
            phone. No scrollbar, no ellipsis, nothing failing. The pressure is
            higher now, not lower: five entries became nine. */}
        <div
          data-testid="board-chip-strip"
          className="flex flex-wrap items-center gap-x-4 gap-y-2 text-base"
        >
          {standings.map((item) => (
            <span key={item.key} className="flex items-center gap-2 whitespace-nowrap">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                style={{ backgroundColor: item.color }}
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
        {/* Names the axis directly above it, and only that. The axis is percent
            now — see the plan's §6 — so a caption about "account value" would
            describe a chart that is not there. By the same rule it is withheld
            when nothing was drawn: on an empty 200 there IS no axis, and a
            caption dating one is the confident-frame-over-nothing claim the
            empty branch above exists to remove. */}
        {coverage === "empty" ? null : (
          <p className="mt-3 text-sm text-foreground/65">
            Return over the competition window, hour by hour.
          </p>
        )}
      </div>
    </div>
  );
}
