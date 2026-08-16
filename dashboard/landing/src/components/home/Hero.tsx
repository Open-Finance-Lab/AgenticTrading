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
        {/* Ordered with `order-*`, never by moving the board component above
            this block in source: that would put the board's <h2> ahead of the
            page's only <h1> and open the document outline on the board's
            title. */}
        {/* `lg:grow`, not `lg:grow-0`. The board takes
            `lg:ms-[calc((100%-100vw)/2)]`, which frees the container's left
            gutter as flex free space -- 152px at 1920 -- and with grow-0 on
            both columns nothing claimed it, so the copy column stopped short of
            the container's right edge for no stated reason. The board keeps
            grow-0 so it stays exactly the declared 2/3; the copy absorbs the
            slack the negative margin created.

            `order-last` alone: the `lg:order-last` that used to sit beside it
            restated an unconditional base class, and the guard asserted both,
            so the dead prefix was pinned in place. */}
        <div className="flex-1 lg:basis-1/3 lg:grow order-last text-center lg:text-left">
          <h1 className="mb-6 max-w-xl text-[clamp(2.85rem,3.9vw,4.25rem)] font-extrabold leading-[1.05] tracking-[-0.04em] text-[#e5e7eb] mx-auto lg:mx-0">
            <span className="block">
              <HeadlineWords words={HEADLINE_LINE_1} startDelay={LINE1_START} />
            </span>
            <span className="block mt-[0.42em] text-[#22d3ee]">
              <HeadlineWords words={HEADLINE_LINE_2} startDelay={line2Delay} />
            </span>
          </h1>
          {/* The one-per-surface gloss on "agent", then the invitation. The
              headline uses the word before anything else on the page defines it,
              and the board beside it is the only other thing above the fold — so
              the definition has to land here or not at all. Two short sentences
              and no more: at 1/3 column width a third clause wraps and makes the
              copy column taller than the card, which pushes the card's bottom
              edge below the fold (the hero row is `items-center`). */}
          <p className="max-w-xl mx-auto lg:mx-0 mb-5 text-base text-foreground/85 leading-relaxed">
            Agents here are AI trading assistants that follow your written instruction. You may
            write your trading strategy in plain English.
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
          {/* The challenge the board raises, and the one line of small print
              that keeps it from reading as an invitation to risk anything. The
              second line is pinned verbatim by the _CLAIM_DISCLAIMERS allowlist,
              whose staleness check fails if the wording drifts: it names the
              exact phrase the brokered-claim scan bans, in order to deny it, so
              a reword that leaves the allowlist behind re-arms the ban on the
              disclaimer itself. Below the CTA on purpose — it answers the
              question the button raises rather than competing with it. */}
          <p className="max-w-xl mx-auto lg:mx-0 mt-5 text-sm text-foreground/75">
            Can you beat the strategies and baselines on the left?
          </p>
          <p className="max-w-xl mx-auto lg:mx-0 mt-1.5 text-xs text-foreground/60">
            No real money. Simulated money only.
          </p>
        </div>

        {/* The board, not a product screenshot. It used to sit four screens down
            under #race, which meant the one piece of evidence on the page was
            the last thing anyone saw. The full standings and the rules that
            govern them still live there; this is the same numbers, above the
            fold, so nobody has to scroll to find out what is being measured. */}
        {/* The 672px card cap is gone deliberately: that is card width, and two
            thirds of a 1280px container is 853px, so leaving it on makes every
            other change here cosmetic.

            The negative inline-start margin escapes the shared
            `container mx-auto px-6` on one edge only — a class removal cannot
            do it, because that same div owns the hero's min-height contract.
            It is a >=1300px effect: the container's left gutter measures 185px
            at 1920 and 73px at 1440, but 0px at 1280 and below, where the 2/3
            split carries the layout alone. `lg:ps-6` puts back the 24px the
            container's padding was giving this edge, so the chart runs flush to
            the viewport without its axis labels touching it. */}
        <motion.div
          className="w-full flex-1 lg:basis-2/3 lg:grow-0 shrink-0 order-first lg:ms-[calc((100%-100vw)/2)] lg:ps-6"
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
