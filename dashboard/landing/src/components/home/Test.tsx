import { Button } from "@/components/ui/button";
import { ArrowUpRight, ArrowDownRight, Clock } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  ReferenceLine,
} from "recharts";
import { PRIMARY_LANDING_CTA } from "@/lib/cta";
import { STORY_AGENT_NAME, STORY_DECISIONS, STORY_SPECS } from "./storyline";

const C0 = STORY_SPECS.initialCapitalNum;

/** Regular-session sample hours (ET): 7 points per trading day. */
const SESSION_HOURS = [10, 11, 12, 13, 14, 15, 16] as const;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

type EquityPoint = {
  t: string;
  dayLabel: string;
  hour: number;
  isDayOpen: boolean;
  agent: number;
  djia: number;
  spy: number;
  buyHold: number;
};

/** Deterministic noise in [-1, 1] from integer seed. */
function noise(seed: number) {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}

/**
 * Hourly equity for the illustrative window in STORY_SPECS.timePeriod
 * (weekdays only). Paths drift toward the published end levels; agent
 * outperforms baselines.
 *
 * KEEP THESE DATES AND STORY_SPECS.timePeriod IN STEP -- the x-axis day labels
 * come from here while the settings card above states the window, so a drift
 * between them puts two different months inside one figure. They must also stay
 * OFF the contest window in dashboard/config/leaderboard.json: see the note on
 * STORY_SPECS for why a real window here is a falsifiable claim now that the
 * live board is four screens up.
 */
function buildHourlyEquity(): EquityPoint[] {
  const start = new Date(Date.UTC(2025, 8, 3));
  const end = new Date(Date.UTC(2025, 9, 3));
  const tradingDays: Date[] = [];
  for (let d = new Date(start); d <= end; d.setUTCDate(d.getUTCDate() + 1)) {
    const wd = d.getUTCDay();
    if (wd === 0 || wd === 6) continue;
    tradingDays.push(new Date(d));
  }

  const n = tradingDays.length * SESSION_HOURS.length;
  const points: EquityPoint[] = [];
  let i = 0;

  for (const day of tradingDays) {
    const dayLabel = `${MONTHS[day.getUTCMonth()]} ${day.getUTCDate()}`;
    for (let h = 0; h < SESSION_HOURS.length; h += 1) {
      const hour = SESSION_HOURS[h];
      const progress = i / Math.max(1, n - 1);
      // Target terminals: agent 11420, spy ~10480, djia ~10310, buyHold ~10490
      const agentBase = C0 + progress * 1420;
      const spyBase = C0 + progress * 480;
      const djiaBase = C0 + progress * 310;
      const bhBase = C0 + progress * 490;
      // Intraday + day noise (linear segments, not smoothed)
      const agent =
        agentBase +
        noise(i * 3 + 1) * 55 +
        noise(day.getUTCDate() * 7 + h) * 35 +
        Math.sin(progress * Math.PI * 4) * 90;
      const spy = spyBase + noise(i * 5 + 2) * 28 + Math.sin(progress * Math.PI * 2) * 40;
      const djia = djiaBase + noise(i * 7 + 3) * 24 + Math.cos(progress * Math.PI * 2.2) * 35;
      const buyHold = bhBase + noise(i * 11 + 4) * 30 + Math.sin(progress * Math.PI * 3) * 45;

      points.push({
        t: `${dayLabel} ${String(hour).padStart(2, "0")}:00`,
        dayLabel,
        hour,
        isDayOpen: h === 0,
        agent: Math.round(agent * 10) / 10,
        djia: Math.round(djia * 10) / 10,
        spy: Math.round(spy * 10) / 10,
        buyHold: Math.round(buyHold * 10) / 10,
      });
      i += 1;
    }
  }

  // Pin endpoints exactly for metric consistency
  if (points.length) {
    points[0] = { ...points[0], agent: C0, djia: C0, spy: C0, buyHold: C0 };
    const last = points[points.length - 1];
    points[points.length - 1] = {
      ...last,
      agent: 11420,
      djia: 10310,
      spy: 10480,
      buyHold: 10490,
    };
  }

  return points;
}

const EQUITY = buildHourlyEquity();

/** Show ~1 day label every few trading days (first hour of that day). */
const DAY_TICK_EVERY = 3;
const dayOpenIndices = EQUITY.map((p, idx) => (p.isDayOpen ? idx : -1)).filter((idx) => idx >= 0);
const xTickIndices = new Set(
  dayOpenIndices.filter((_, k) => k % DAY_TICK_EVERY === 0 || k === dayOpenIndices.length - 1),
);

const LINE = {
  agent: "#22d3ee",
  djia: "#94a3b8",
  spy: "#64748b",
  buyHold: "#a78bfa",
} as const;

const SETTINGS: { label: string; value: string }[] = [
  { label: "Initial capital", value: STORY_SPECS.initialCapital },
  { label: "Time period", value: `${STORY_SPECS.timePeriodLabel} · ${STORY_SPECS.timePeriod}` },
  { label: "Universe", value: STORY_SPECS.universe },
  { label: "Baselines", value: STORY_SPECS.baselines.join(" · ") },
  { label: "AI model", value: STORY_SPECS.model },
  { label: "Est. AI cost", value: STORY_SPECS.estTokenCost },
];

function money(v: number) {
  return `$${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

export function Test() {
  return (
    <section id="test" className="py-24 relative scroll-mt-40">
      <div className="container mx-auto px-6">
        <div className="mb-10 max-w-3xl">
          <p className="text-base md:text-lg font-mono tracking-wide text-primary mb-3">02 — Test</p>
          <h2 className="text-3xl md:text-4xl font-bold mb-3">Test your trading idea</h2>
        </div>

        <figure className="bg-card border border-card-border rounded-xl shadow-2xl overflow-hidden mb-10">
          {/* 1. Trading Performance */}
          <figcaption className="flex flex-wrap items-start justify-between gap-3 px-6 md:px-8 pt-6 md:pt-8 pb-5 border-b border-border">
            <div className="min-w-0">
              <h3 className="text-lg md:text-xl font-bold text-foreground">
                Trading Performance
              </h3>
            </div>
            <span className="text-xs font-mono text-muted-foreground bg-muted px-2.5 py-1 rounded shrink-0">
              Illustrative
            </span>
          </figcaption>

          <div className="p-6 md:p-8 space-y-8">
            {/* 2. Experiment settings — 6 modules */}
            <div>
              <h4 className="text-sm font-mono tracking-wide text-foreground/70 mb-4">
                Experiment settings
              </h4>
              <dl className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {SETTINGS.map((s) => (
                  <div key={s.label} className="rounded-lg border border-border bg-background/80 px-4 py-3">
                    <dt className="text-xs font-mono tracking-wide text-foreground/60 mb-1">
                      {s.label}
                    </dt>
                    <dd className="text-base font-medium text-foreground leading-snug">{s.value}</dd>
                  </div>
                ))}
              </dl>
            </div>

            {/* 3. Chart — hourly, linear */}
            <div className="h-[360px] md:h-[440px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={EQUITY} margin={{ top: 12, right: 16, left: 4, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                  <XAxis
                    dataKey="t"
                    stroke="hsl(var(--foreground))"
                    fontSize={13}
                    tickLine={false}
                    axisLine={false}
                    interval={0}
                    minTickGap={28}
                    ticks={EQUITY.filter((_, idx) => xTickIndices.has(idx)).map((p) => p.t)}
                    tickFormatter={(value: string) => {
                      const pt = EQUITY.find((p) => p.t === value);
                      return pt?.dayLabel ?? "";
                    }}
                  />
                  <YAxis
                    stroke="hsl(var(--foreground))"
                    fontSize={13}
                    tickLine={false}
                    axisLine={false}
                    domain={[9600, 11600]}
                    ticks={[10000, 10500, 11000, 11500]}
                    tickFormatter={(v) => `$${(v / 1000).toFixed(1)}k`}
                    width={52}
                  />
                  <ReferenceLine
                    y={C0}
                    stroke="hsl(var(--muted-foreground))"
                    strokeDasharray="4 4"
                    strokeOpacity={0.45}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      borderColor: "hsl(var(--border))",
                      borderRadius: "8px",
                      fontSize: "13px",
                    }}
                    formatter={(v: number, name: string) => [money(v), name]}
                    labelFormatter={(label) => String(label)}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: "13px", paddingTop: "8px" }}
                    iconType="plainline"
                  />
                  <Line
                    type="linear"
                    dataKey="agent"
                    name={STORY_AGENT_NAME}
                    stroke={LINE.agent}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                    activeDot={{ r: 3 }}
                  />
                  <Line
                    type="linear"
                    dataKey="spy"
                    name="S&P 500"
                    stroke={LINE.spy}
                    strokeWidth={1.5}
                    strokeDasharray="6 4"
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="linear"
                    dataKey="djia"
                    name="DJIA"
                    stroke={LINE.djia}
                    strokeWidth={1.5}
                    strokeDasharray="2 3"
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="linear"
                    dataKey="buyHold"
                    name="Buy & Hold"
                    stroke={LINE.buyHold}
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* 4. Metrics */}
            <div>
              <h4 className="text-sm font-mono tracking-wide text-foreground/70 mb-4">
                Metrics
              </h4>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <Metric label="Total return" value={STORY_SPECS.returnPct} tone="positive" />
                <Metric label="Sharpe ratio" value={STORY_SPECS.sharpe} />
                <Metric label="Max drawdown" value={STORY_SPECS.maxDd} tone="destructive" />
                <Metric label="vs Buy & Hold" value={STORY_SPECS.vsBuyHold} tone="positive" />
              </div>
            </div>
          </div>
        </figure>

        <div className="mb-10">
          <div className="flex items-baseline justify-between gap-3 mb-4">
            <h3 className="text-base font-mono tracking-wide text-foreground/80">
              Decision log · selected steps
            </h3>
            <span className="text-sm text-foreground/60 font-mono">
              {STORY_SPECS.trades} trades total
            </span>
          </div>
          <div className="border border-card-border rounded-xl overflow-hidden bg-card">
            <div className="hidden md:grid grid-cols-12 gap-2 px-4 py-2.5 text-xs font-mono tracking-wide text-foreground/60 border-b border-border bg-muted/30">
              <div className="col-span-2">Step / time</div>
              <div className="col-span-2">Action</div>
              <div className="col-span-2">Size</div>
              <div className="col-span-6">Rationale</div>
            </div>
            {STORY_DECISIONS.map((d) => (
              <div
                key={`${d.step}-${d.symbol}`}
                className="grid md:grid-cols-12 gap-2 px-4 py-3.5 border-b border-border last:border-b-0 items-start"
              >
                <div className="md:col-span-2 font-mono text-sm text-foreground/70">
                  <div className="text-foreground font-semibold">#{d.step}</div>
                  <div>{d.time}</div>
                </div>
                <div className="md:col-span-2 flex items-center gap-2">
                  <span
                    className={
                      d.type === "positive"
                        ? "w-7 h-7 rounded-md flex items-center justify-center bg-emerald-500/10 text-emerald-400 shrink-0"
                        : d.type === "destructive"
                          ? "w-7 h-7 rounded-md flex items-center justify-center bg-red-500/10 text-red-400 shrink-0"
                          : "w-7 h-7 rounded-md flex items-center justify-center bg-slate-500/10 text-slate-400 shrink-0"
                    }
                  >
                    {d.action === "BUY" ? (
                      <ArrowUpRight className="w-3.5 h-3.5" />
                    ) : d.action === "SELL" ? (
                      <ArrowDownRight className="w-3.5 h-3.5" />
                    ) : (
                      <Clock className="w-3.5 h-3.5" />
                    )}
                  </span>
                  <div className="text-sm font-semibold">
                    <span
                      className={
                        d.type === "positive"
                          ? "text-positive"
                          : d.type === "destructive"
                            ? "text-destructive"
                            : "text-muted-foreground"
                      }
                    >
                      {d.action.charAt(0) + d.action.slice(1).toLowerCase()}
                    </span>{" "}
                    {d.symbol}
                  </div>
                </div>
                <div className="md:col-span-2 font-mono text-sm text-foreground/70">
                  {d.shares} sh @ {d.price}
                </div>
                <div className="md:col-span-6 text-sm text-foreground/80 leading-relaxed">{d.detail}</div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <Button
            size="lg"
            type="button"
            data-landing-auth={PRIMARY_LANDING_CTA.authMode}
            className="bg-primary text-primary-foreground glow-primary hover:bg-primary/90"
          >
            {PRIMARY_LANDING_CTA.label}
          </Button>
        </div>
      </div>
    </section>
  );
}

function Metric({ label, value, tone }: { label: string; value: string; tone?: "positive" | "destructive" }) {
  const valueClass =
    tone === "positive" ? "text-positive" : tone === "destructive" ? "text-destructive" : "text-foreground";
  return (
    <div className="p-3.5 border border-border rounded-lg bg-background">
      <div className="text-sm text-foreground/70 mb-1">{label}</div>
      <div className={`text-2xl font-bold font-mono tabular-nums ${valueClass}`}>{value}</div>
    </div>
  );
}
