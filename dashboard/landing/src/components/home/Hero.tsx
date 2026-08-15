import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ChevronDown } from "lucide-react";
import { useState, useEffect } from "react";
import { PRIMARY_LANDING_CTA } from "@/lib/cta";
import { BoardPreview } from "./BoardPreview";

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
          <h1 className="mb-6 max-w-xl text-[clamp(2.85rem,3.9vw,4.25rem)] font-extrabold leading-[1.05] tracking-[-0.04em] text-[#e5e7eb] mx-auto lg:mx-0">
            <span className="block">
              <HeadlineWords words={HEADLINE_LINE_1} startDelay={LINE1_START} />
            </span>
            <span className="block mt-[0.42em] text-[#22d3ee]">
              <HeadlineWords words={HEADLINE_LINE_2} startDelay={line2Delay} />
            </span>
          </h1>
          {/* The one-per-surface gloss on "agent". The headline uses the word
              before anything else on the page defines it, and the board beside
              it is the only other thing above the fold — so the definition has
              to land here or not at all. */}
          <p className="max-w-xl mx-auto lg:mx-0 mb-5 text-base text-foreground/85 leading-relaxed">
            Write your trading idea in plain English. An agent is an AI trading assistant that
            follows your written instruction — it trades the idea hour by hour, measured against
            buy-and-hold and the index.
          </p>
          <p className="max-w-xl mx-auto lg:mx-0 mb-8 text-sm text-foreground/75">
            Every test here uses simulated money. Real money is involved only if you explicitly
            connect a brokerage account and turn on live trading.
          </p>
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

        {/* The board, not a product screenshot. It used to sit four screens down
            under #race, which meant the one piece of evidence on the page was
            the last thing anyone saw. The full standings and the rules that
            govern them still live there; this is the same numbers, above the
            fold, so nobody has to scroll to find out what is being measured. */}
        <motion.div
          className="flex-1 w-full max-w-2xl shrink-0"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.7, delay: 0.3 }}
        >
          <BoardPreview />
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
