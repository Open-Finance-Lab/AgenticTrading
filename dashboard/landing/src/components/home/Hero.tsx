import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Terminal, Bot, User, Search, LineChart, CheckCircle2, ChevronDown } from "lucide-react";
import { useState, useEffect } from "react";
import { PRIMARY_LANDING_CTA } from "@/lib/cta";

const HEADLINE_LINE_1 = ["Talk", "to", "Agents"] as const;
const HEADLINE_LINE_2 = ["Test", "Trading", "Ideas"] as const;
/** Per-word fade cadence — slower reads clearer on first paint. */
const WORD_STAGGER = 0.18;
const WORD_DURATION = 0.7;
/** Quiet beat after line 1 finishes before line 2 starts. */
const LINE_GAP = 0.65;
const LINE1_START = 0.1;
const EASE = [0.22, 1, 0.36, 1] as const;

function Word({
  children,
  delay,
  className = "",
}: {
  children: string;
  delay: number;
  className?: string;
}) {
  return (
    <motion.span
      className={`inline-block ${className}`}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: WORD_DURATION, ease: EASE, delay }}
    >
      {children}
    </motion.span>
  );
}

function HeadlineWords({
  words,
  startDelay,
  wordClassName = "",
}: {
  words: readonly string[];
  startDelay: number;
  wordClassName?: string;
}) {
  return (
    <span className="inline">
      {words.map((word, i) => (
        <span key={`${word}-${i}`}>
          {i > 0 ? " " : null}
          <Word delay={startDelay + i * WORD_STAGGER} className={wordClassName}>
            {word}
          </Word>
        </span>
      ))}
    </span>
  );
}

export function Hero() {
  const [hintHidden, setHintHidden] = useState(false);
  // Start line 2 only after line 1's last word has finished + LINE_GAP.
  const line2Delay =
    LINE1_START +
    (HEADLINE_LINE_1.length - 1) * WORD_STAGGER +
    WORD_DURATION +
    LINE_GAP;
  const ctaDelay =
    line2Delay +
    (HEADLINE_LINE_2.length - 1) * WORD_STAGGER +
    WORD_DURATION +
    0.25;

  useEffect(() => {
    const onScroll = () => {
      setHintHidden(window.scrollY > 48);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const scrollToNext = () => {
    document.getElementById("landing-stats")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <section className="relative min-h-[100dvh] flex items-start overflow-hidden landing-hero pb-20 md:pb-24">
      <div className="absolute inset-0 bg-grid-pattern opacity-30 [mask-image:radial-gradient(ellipse_at_center,black,transparent_80%)]" />

      <div className="container mx-auto px-6 relative z-10 flex flex-col lg:flex-row items-center gap-12 lg:gap-16 lg:min-h-[calc(100dvh-var(--landing-chrome-height)-4rem)]">
        <div className="flex-1 text-center lg:text-left">
          <h1 className="mb-8 max-w-xl text-[clamp(2.85rem,3.9vw,4.25rem)] font-extrabold leading-[1.05] tracking-[-0.04em] text-[#e5e7eb] mx-auto lg:mx-0">
            <span className="block">
              <HeadlineWords words={HEADLINE_LINE_1} startDelay={LINE1_START} />
            </span>
            <span className="block mt-[0.42em] text-[#22d3ee]">
              <HeadlineWords words={HEADLINE_LINE_2} startDelay={line2Delay} />
            </span>
          </h1>
          <motion.div
            className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: ctaDelay }}
          >
            <Button
              size="lg"
              type="button"
              data-landing-auth={PRIMARY_LANDING_CTA.authMode}
              className="w-full sm:w-auto bg-primary text-primary-foreground glow-primary hover:bg-primary/90 text-base h-12 px-8"
            >
              {PRIMARY_LANDING_CTA.label}
            </Button>
          </motion.div>
        </div>

        <motion.div
          className="flex-1 w-full max-w-2xl shrink-0"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.7, delay: 0.3 }}
        >
          <div className="bg-card border border-card-border rounded-xl shadow-2xl overflow-hidden flex flex-col h-[480px] min-h-[480px] max-h-[480px]">
            <div className="h-10 shrink-0 bg-muted/50 border-b border-border flex items-center px-4 gap-2">
              <div className="w-3 h-3 rounded-full bg-destructive/80" />
              <div className="w-3 h-3 rounded-full bg-secondary-border" />
              <div className="w-3 h-3 rounded-full bg-positive/80" />
              <div className="ml-4 text-xs font-mono text-muted-foreground flex items-center gap-2">
                <Terminal className="w-3 h-3" />
                agent-playground.exe
              </div>
            </div>
            <div className="flex-1 min-h-0 p-4 font-mono text-sm overflow-y-auto overflow-x-hidden flex flex-col gap-3">
              <ChatSimulation />
            </div>
          </div>
        </motion.div>
      </div>

      <button
        type="button"
        className={`landing-scroll-hint${hintHidden ? " is-hidden" : ""}`}
        aria-label="Scroll for more"
        onClick={scrollToNext}
      >
        <span className="landing-scroll-hint-label">Scroll</span>
        <ChevronDown className="landing-scroll-hint-icon" aria-hidden="true" />
      </button>
    </section>
  );
}

/** User messages sit on the RIGHT. */
function UserBubble({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-3 flex-row-reverse">
      <div className="w-6 h-6 rounded bg-primary/20 text-primary flex items-center justify-center shrink-0">
        <User className="w-3 h-3" />
      </div>
      <div className="bg-primary/15 border border-primary/25 p-3 rounded-l-lg rounded-br-lg text-foreground max-w-[88%]">
        {children}
      </div>
    </div>
  );
}

/** Agent messages sit on the LEFT. */
function AgentBubble({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <div className="w-6 h-6 rounded bg-secondary-border flex items-center justify-center shrink-0">
        <Bot className="w-3 h-3 text-muted-foreground" />
      </div>
      <div className="bg-card border border-card-border p-3 rounded-r-lg rounded-bl-lg text-muted-foreground max-w-[88%]">
        {children}
      </div>
    </div>
  );
}

function EquityCurve() {
  const points = "0,42 20,40 40,38 60,34 80,36 100,30 120,28 140,22 160,24 180,16 200,12 220,14 240,8";
  return (
    <svg viewBox="0 0 240 50" className="w-full h-14 mt-2 mb-1" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <linearGradient id="heroEquityFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`0,50 ${points} 240,50`} fill="url(#heroEquityFill)" />
      <polyline points={points} fill="none" stroke="#22d3ee" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

function FadeIn({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: "easeOut", delay }}
    >
      {children}
    </motion.div>
  );
}

function ChatSimulation() {
  const [step, setStep] = useState(0);
  const [cycle, setCycle] = useState(0);

  // Advance steps, then pause briefly and replay from the start.
  useEffect(() => {
    if (step < 4) {
      const timer = setTimeout(() => setStep((s) => s + 1), 2200);
      return () => clearTimeout(timer);
    }
    const timer = setTimeout(() => {
      setStep(0);
      setCycle((c) => c + 1);
    }, 3200);
    return () => clearTimeout(timer);
  }, [step]);

  return (
    <div key={cycle} className="flex flex-col gap-3">
      <FadeIn>
        <UserBubble>
          I want to follow Warren Buffett. If Berkshire makes a move, copy the move and tell me how it goes.
        </UserBubble>
      </FadeIn>

      {step >= 1 && (
        <FadeIn>
          <AgentBubble>
            <div className="flex items-center gap-2 text-foreground mb-2">
              <Search className="w-3.5 h-3.5 text-primary" />
              <span>Fetching Berkshire Hathaway 13F filings...</span>
            </div>
            <ul className="space-y-1.5 text-xs leading-relaxed">
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-positive shrink-0 mt-0.5" />
                <span>Q1 2025 → +$OXY +$SIRI −$PARA</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-positive shrink-0 mt-0.5" />
                <span>Q4 2024 → +$OXY +$VZ −$HP</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-positive shrink-0 mt-0.5" />
                <span>Copy-trade rules set · 6 tickers tracked</span>
              </li>
            </ul>
          </AgentBubble>
        </FadeIn>
      )}

      {step >= 2 && (
        <FadeIn>
          <AgentBubble>
            <div className="flex items-center gap-2 text-foreground mb-1.5">
              <LineChart className="w-3.5 h-3.5 text-primary" />
              <span>Running backtest · 24 months, $10k start...</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <CheckCircle2 className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
              <span>22 trades · avg hold 63 days</span>
            </div>
          </AgentBubble>
        </FadeIn>
      )}

      {step >= 3 && (
        <FadeIn>
          <AgentBubble>
            <div className="flex items-center gap-2 mb-1 text-positive">
              <LineChart className="w-4 h-4" />
              <span className="font-bold">Backtest Complete</span>
            </div>
            <EquityCurve />
            <div className="grid grid-cols-4 gap-2 text-xs mt-1">
              <div>Return <span className="text-positive font-semibold">+41.2%</span></div>
              <div>Sharpe <span className="text-foreground font-semibold">1.31</span></div>
              <div>Win Rate <span className="text-foreground font-semibold">68%</span></div>
              <div>Max DD <span className="text-destructive font-semibold">-9.4%</span></div>
            </div>
          </AgentBubble>
        </FadeIn>
      )}

      {step >= 4 && (
        <FadeIn>
          <AgentBubble>
            <p className="text-foreground mb-3">
              Looks solid. Want me to run this in{" "}
              <span className="text-primary font-semibold">paper trading</span>{" "}
              and alert you when Berkshire&apos;s next 13F drops?
            </p>
            <div className="flex flex-wrap gap-2">
              <span className="inline-flex items-center rounded-md border border-primary/50 bg-primary/10 px-3 py-1.5 text-xs font-semibold text-primary">
                Yes, start now
              </span>
              <span className="inline-flex items-center rounded-md border border-border px-3 py-1.5 text-xs font-semibold text-muted-foreground">
                Not yet
              </span>
            </div>
          </AgentBubble>
        </FadeIn>
      )}
    </div>
  );
}
