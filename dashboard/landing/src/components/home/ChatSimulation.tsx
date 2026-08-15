import { motion } from "framer-motion";
import { Bot, User, Search, LineChart, CheckCircle2, MessageSquare } from "lucide-react";
import { useState, useEffect } from "react";

/**
 * The agent conversation demo. Lived inside Hero.tsx until the board took the
 * hero's right column; it now runs under the Talk act, which is the beat it
 * actually illustrates.
 *
 * Step 4's simulated-money gloss is one of the two sentences allowlisted by
 * `test_landing_copy_register.py::_CLAIM_DISCLAIMERS`. Keep it byte-identical —
 * an allowlist entry that stops matching silently re-arms the brokered-claim
 * ban against a sentence that was always fine.
 *
 * (This comment deliberately does not spell the banned phrase out: the scan
 * reads the whole file, comments included, and an explanatory mention reddens
 * the guard exactly as a real claim would.)
 */

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
            <div className="flex items-center justify-between gap-2 mb-1">
              <div className="flex items-center gap-2 text-positive">
                <LineChart className="w-4 h-4" />
                <span className="font-bold">Backtest Complete</span>
              </div>
              <span className="text-xs font-mono text-muted-foreground">Illustrative example</span>
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
              <span className="text-primary font-semibold">paper trading</span>
              {" "}— practice trading with simulated money at live market prices —
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

/**
 * The framed card. The title bar used to read `agent-playground.exe` — a
 * filename is developer furniture on a page written for people who trade, and
 * the app-side twin (`app.html`) was already reworded away from it.
 */
export function ChatWindow() {
  return (
    <div className="bg-card border border-card-border rounded-xl shadow-2xl overflow-hidden flex flex-col h-[480px] min-h-[480px] max-h-[480px]">
      <div className="h-10 shrink-0 bg-muted/50 border-b border-border flex items-center px-4 gap-2">
        <div className="w-3 h-3 rounded-full bg-destructive/80" />
        <div className="w-3 h-3 rounded-full bg-secondary-border" />
        <div className="w-3 h-3 rounded-full bg-positive/80" />
        <div className="ml-4 text-xs font-mono text-muted-foreground flex items-center gap-2">
          <MessageSquare className="w-3 h-3" />
          Example: a conversation with your agent
        </div>
      </div>
      <div className="flex-1 min-h-0 p-4 font-mono text-sm overflow-y-auto overflow-x-hidden flex flex-col gap-3">
        <ChatSimulation />
      </div>
    </div>
  );
}
