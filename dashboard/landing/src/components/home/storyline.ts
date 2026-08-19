/** Shared demo storyline across Talk → Test → Race. Keep phrases identical. */

/**
 * Refined strategy prompt shown after Discord brainstorming (Talk).
 * Hero keeps its own frozen casual line; this is the Lab-ready formulation.
 */
export const STORY_PROMPT =
  "Systematically mirror material Berkshire Hathaway 13F position changes: enter and exit holdings when filings disclose significant increases or reductions. Evaluate over a one-month window with $10,000 starting capital; report return, Sharpe ratio, and maximum drawdown versus DJIA, S&P 500, and equal-weight buy-and-hold baselines.";

export const STORY_AGENT_NAME = "Alpha";

/** Experiment settings for the Test (Part 2) run report.
 *
 *  THE MODEL AND THE WINDOW ARE DELIBERATE PLACEHOLDERS, and that is the whole
 *  of the constraint on this block. Everything here is fabricated -- the figure
 *  carries an "Illustrative" chip saying so -- which was unremarkable while
 *  nothing else on the page could be checked against it. The live Competition
 *  board now sits four screens up, so naming a REAL roster model over the REAL
 *  contest window turned a fabricated `+14.2%` into a claim a visitor falsifies
 *  by scrolling: the board publishes that same model's actual return over that
 *  same window from that same $10,000 base.
 *
 *  Breaking either half breaks the comparison; the window is the cheaper half,
 *  and both are broken here. Nothing else changed -- the numbers, the dollar
 *  axis and the layout are untouched, because they are not what made this
 *  falsifiable. Pinned by test_the_illustrative_run_report_* in
 *  dashboard/backend/tests/test_landing_copy_register.py, which derives the
 *  banned roster from dashboard/config/leaderboard.json rather than hardcoding
 *  it, so an eighth LLM entry extends the ban by itself.
 *
 *  Reversible in one commit if the product owner would rather name a real
 *  model: put the name and the window back and delete those guards. What must
 *  not ship is a real model plus a real window plus a fabricated return on the
 *  same scroll as the live board. */
export const STORY_SPECS = {
  timePeriod: "Sep 3 – Oct 3, 2025",
  timePeriodLabel: "1 month",
  initialCapital: "$10,000",
  initialCapitalNum: 10_000,
  universe: "DJIA 30",
  baselines: ["DJIA", "S&P 500", "Buy & Hold"] as const,
  model: "Example model",
  estTokenCost: "$0.38",
  estTokens: "412k in · 28k out",
  returnPct: "+14.2%",
  sharpe: "1.84",
  maxDd: "-8.6%",
  vsBuyHold: "+4.3%",
  trades: 22,
  avgHoldDays: 63,
  /** @deprecated use timePeriod — kept for older refs */
  window: "Sep 3 – Oct 3, 2025",
} as const;

export const STORY_DECISIONS = [
  {
    step: 12,
    time: "Sep 10 · 14:00 ET",
    action: "BUY" as const,
    symbol: "OXY",
    shares: 48,
    price: "$62.40",
    detail: "Material increase in latest 13F — mirrored entry at open of next session",
    type: "positive" as const,
  },
  {
    step: 41,
    time: "Sep 19 · 15:00 ET",
    action: "HOLD" as const,
    symbol: "AAPL",
    shares: 36,
    price: "$198.20",
    detail: "No material Berkshire change this step — position unchanged",
    type: "muted" as const,
  },
  {
    step: 58,
    time: "Sep 26 · 14:00 ET",
    action: "SELL" as const,
    symbol: "PARA",
    shares: 120,
    price: "$11.85",
    detail: "Material reduction in 13F — mirrored full exit",
    type: "destructive" as const,
  },
];
