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
import { formatAxisDate, formatPercent, type BoardSeries } from "@/lib/leaderboard";
import { frameLayout, measureTextWidth } from "@/lib/boardFrame";
import { EndpointRail } from "./EndpointRail";

/** Matches `fontSize={14}` on both axes below. The y-axis reserve is measured
 *  in it rather than guessed: `width={56}` was measured correctly against
 *  `$1030` at 11px, the tick font later moved to 14px, and four of five labels
 *  lost their leading `$` with nothing failing. */
const AXIS_TICK_FONT = "14px Inter, system-ui, sans-serif";

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
  const values: number[] = [];
  series.forEach((s) => s.values.forEach((v) => { if (v != null) values.push(v); }));
  if (!values.length) return [-0.05, 0.05];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
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

  const frame = useMemo(
    () =>
      frameLayout({
        width: size.width,
        height: size.height,
        labels: standings.map((s) => ({ name: s.name, value: s.ret })),
      }),
    [size.width, size.height, standings],
  );

  const domain = useMemo(() => percentDomain(series), [series]);
  const yAxisWidth = useMemo(() => {
    const widest = Math.max(
      measureTextWidth(axisTick(domain[0]), AXIS_TICK_FONT),
      measureTextWidth(axisTick(domain[1]), AXIS_TICK_FONT),
    );
    return Math.ceil(widest) + 12;
  }, [domain]);

  const rows = useMemo(() => toRows(data?.times ?? [], series), [data, series]);
  const valueByKey = useMemo(
    () => Object.fromEntries(standings.map((s) => [s.key, s.ret])),
    [standings],
  );

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
          Each line is one AI model&apos;s return. Dashed lines are buy-and-hold and the index.
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
          the chip strip takes five rows instead of four and nonChart is
          309.75, so the card hung 55.75px BELOW THE FOLD across that whole
          band -- every 1280-wide-and-under laptop -- while the 1280+ viewports
          the number was checked against passed with 4.25px to spare.

          THE 260px FLOOR, NOT THE RESERVE, IS WHAT BINDS ON A PHONE, and no
          value here can change that: at 390x844 the card needs 920.5px
          (132 + 528.5 + the 260 floor) against 844 of viewport, so the last
          ~77px -- the tail of the chip strip -- sits below the fold at every
          reserve. Dropping the floor to ~183 is the only thing that would pull
          it up, and that trades the chart the hero exists to show for its own
          fallback key: the chart itself already ends at y=586, well above the
          fold. Left as measured deliberately. The reserve still earns its
          value on every stacked viewport tall enough for the floor to clear
          (>=990dvh: 390x1000 fits with 49.5px to spare); below that the
          floor decides and the reserve is inert.

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
              <Tooltip
                contentStyle={{ backgroundColor: "hsl(var(--card))", borderColor: "hsl(var(--border))", borderRadius: "8px" }}
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
            describe a chart that is not there. */}
        <p className="mt-3 text-sm text-foreground/65">
          Return over the competition window, hour by hour.
        </p>
      </div>
    </div>
  );
}
