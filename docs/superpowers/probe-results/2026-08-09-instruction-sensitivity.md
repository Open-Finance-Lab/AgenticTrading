# Phase 0 probe: does a trading instruction change a pinned LLM's return?

**Date run:** 2026-08-15 (Nemotron), 2026-08-16 (DeepSeek)
**Plan:** `docs/superpowers/plans/2026-08-09-participatory-competition-phase-0-1.md`, Task 3
**Script:** `dashboard/scripts/probe_instruction_sensitivity.py` (commit `6ce95bb`)
**Verdict:** 🚫 **NO RESULT. Both legs are invalid — they ran at the wrong capital base.**
**Spend so far:** $2.2431 of the plan's $4.97 sanction.

---

## Read this first: the probe measured nothing, and an earlier draft of this file said otherwise

Both legs ran at `initial_capital = $10,000`. **Every published board curve they are compared
against was computed at `$100,000`.** At $10,000, a single DJIA share is 2.49% of the portfolio
while the effect under test is ~0.6pp — the measuring instrument is four times coarser than the
thing it is measuring.

An earlier version of this document reported "GATE FAILS on both models — do not build Phase 2."
That conclusion is withdrawn. It was not merely unsupported; it was the kind of confident wrong
answer this probe was written to prevent, and it survived a full write-up because nobody checked
the resolution of the apparatus against the size of the effect.

## What is for the gate

Phase 2 lets users compete by writing a trading instruction. That only means something if the
instruction measurably moves the result. If it does not, the leaderboard ranks noise and the
competition cannot be won on merit. The probe runs one pinned model over one fixed window with
different instructions and asks whether the returns separate.

## The capital mismatch

`dashboard/config/leaderboard.json` carried `initial_capital: 100000` when the 12 board runs were
computed (2026-07-05 and 2026-07-12). It was then changed twice, **after** those runs existed:

| Commit | Date | Change |
|---|---|---|
| `0cfc8fb` | 2026-06-18 | introduced `initial_capital: 100000` |
| `ea1bf2b` | 2026-07-13 | `100000` → `1000` |
| `1dd5816` | 2026-07-20 | `1000` → `10000` ("Scale leaderboard **display** capital to $10k") |

The value is **not** display-only. `domain/leaderboard/service.py:764` and `:1085` pass it
straight into `strategy_impl.run(bars, start_date, end_date, initial_capital)`. The probe reads
the live config, so it correctly used $10k — and thereby produced curves that cannot be compared
with anything on the board.

Confirmed against the seed DB: all 12 contest-window rows carry `initial_equity = 100000.0`.

## Why $10,000 destroys the measurement

Share counts are integers, so the capital base sets how finely a portfolio can express an
allocation. DJIA prices at the window open (fetched live from Alpaca): **min $45.40, median
$249.40, mean $272.56, max $910.92**.

| | **$10,000** (probe) | **$100,000** (board) |
|---|---:|---:|
| one median share, as % of equity | **2.49%** | 0.25% |
| one priciest share, as % of equity | **9.11%** | 0.91% |
| DJIA names unbuyable at equal weight | **6 / 30** | 0 / 30 |
| median shares affordable per name | **1** | 13 |
| cash stranded by integer rounding | **34.6%** | 4.4% |

Three consequences, in order of importance:

1. **There is no size dimension.** With a median of one affordable share per name, every position
   is binary — own it or don't. An instruction cannot express "tilt harder into momentum"; it can
   only change *which* names are held. The largest axis an instruction acts on is quantized away.
2. **The noise floor is set by arithmetic, not by the model.** Two runs differing by one median
   share differ by 2.49pp of final equity. The "noise floor" this probe reported was 1.18pp —
   *below* one share quantum, i.e. entirely consistent with runs that differed by a single
   position, and unusable as evidence about model determinism.
3. **Every run was forced negative.** ~35% of capital is stranded by rounding in a window where
   the market rose ~5%. All four DeepSeek runs returned between −0.25% and −1.85% while the
   passive baselines returned +2.24% to +5.95%. That gap is the cash drag, not the instruction.

Quantization compresses signal while leaving noise intact, so the signal-to-noise ratio can only
get worse at $10k — this is not a case where the error might cancel.

## The runs as executed (a record of the invalid leg, not evidence)

**DeepSeek V4 Pro** via CommonStack, `temperature=0`, 161 decision steps, **$10,000**, window
2026-04-15 → 2026-05-15. All four cleared H6 coverage.

| Run | Instruction | Return | Coverage | Calls | Decisions | Trades | Cost |
|---|---|---:|---:|---:|---:|---:|---:|
| d1 | `aggressive_momentum` | −0.25% | 98.1% | 184 | 158 | 410 | $0.397 |
| d2 | `contrarian_reversion` | −0.91% | 96.3% | 203 | 155 | 652 | $0.538 |
| d3 | `control_nonsense` **A** | −0.66% | 98.1% | 180 | 158 | 644 | $0.453 |
| d4 | `control_nonsense` **B** | −1.85% | 99.4% | 184 | 160 | 635 | $0.470 |

Computed spread figures, **quoted only to show why they cannot be used**: noise (same instruction
twice) 1.18pp, signal (two different instructions) 0.66pp, naive 4-run spread 1.59pp. Against a
2.49pp share quantum, none of these numbers resolve anything.

**Nemotron 3 Nano 30B** via OpenRouter, six instructions, same window, also **$10,000**, spread
0.63pp, $0.3555. Same invalidity.

## What survives the capital problem

Three findings do not depend on return, and are worth carrying into Phase 2 regardless:

**1. Instructions separate *behaviour* strongly, even when return says nothing.** Trade counts on
Nemotron ranged 0 → 517, and `defensive_cash` held exactly $10,000.00 for all 161 steps — literal,
perfect compliance. **Instruction compliance is not instruction performance**, and only the second
is what a leaderboard ranks. A Phase 2 that rewards compliance would be measuring the wrong thing.

**2. An instruction can silently disqualify itself under H6 while being the most expensive run.**
`equal_weight_hold` ("spread the money evenly across many of the available stocks") makes the
model emit one action per DJIA symbol; the response exceeded `LLM_MAX_OUTPUT_TOKENS` and arrived
truncated (`Expecting ',' delimiter: line 209 column 6`, `Bracket mismatch: 24 open, 23 close`).
18 steps failed all three repair attempts and fell back to rule-based → 89.4% coverage, under
`MIN_LLM_DECISION_COVERAGE = 0.95`. It cost **$0.0733, the most of any run in that leg, for the
least usable output** — truncated calls bill in full. If users can write instructions, some will
write ones that induce verbose output, pay the most, and be rejected with no explanation. Phase 2
must surface this at authoring time, not publish time.

(At $10k this instruction is additionally infeasible *by arithmetic* — 6 of 30 names cannot be
bought at all — so some of the overflow may be the model fighting an impossible request.)

**3. Truncated calls are a real and invisible cost.** `llm_calls > llm_decisions` in every
DeepSeek run (184/158, 203/155, 180/158, 184/160): **120 billed calls across the leg produced no
decision.** For contrast, the published DeepSeek run made exactly 161 calls for 161 steps — zero
retries. The retry burst is a property of the custom-prompt path, and it is unbudgeted.

**4. `temperature=0` did not pin the outcome.** Two identical control runs landed 1.18pp apart.
How much of that is model nondeterminism versus the share quantum is unknown — separating them is
the main reason to re-run.

## Defect found in the probe itself — now fixed

The probe read `initial_capital` from the live config and never checked whether that base was
fine enough to resolve the effect it was measuring. Fixed in this commit:

- **`--initial-capital`** overrides the config value, and the effective capital (plus its source)
  is printed in the run header instead of being an invisible default.
- **`_check_capital_resolution`** runs after the bar fetch and **before the first billable call**.
  It refuses to spend when one median share exceeds `MAX_SHARE_FRACTION_PCT` (1.0%) of equity,
  reporting the share fraction and the unbuyable-symbol count. `--allow-coarse-capital` overrides
  it for anyone deliberately measuring the coarse regime.

Verified against the real window: **blocks at $10,000** (2.49%, 6/30 unbuyable), **allows at
$100,000** (0.25%, 0/30), and returns `EXIT_CONFIG` without crashing on empty bars or zero
capital. The guard would have refused both legs before a cent was spent.

Still open, and the more general form: a probe that quotes a stored run should **assert the
stored run's parameters match its own** (`initial_equity`, data feed, window) rather than trusting
that the config which produced them is the config it reads. Same shape as the feed-drift trap
documented under `ALPACA_DATA_FEED` in `CLAUDE.md`.

## Related repo defect (not filed — needs the owner's go-ahead)

`_find_cached_run` (`domain/leaderboard/service.py:615`) keys on
`(mode, start_date, end_date, llm_model)` — **not** `initial_equity`. Combined with the config
drift above:

- A forced refresh (`POST /api/v1/leaderboard/refresh?force=true`) recomputes at $10k, silently
  replacing curves computed at $100k.
- A *partial* refresh leaves $100k rows ranking head-to-head against $10k rows on one board.

Display is unaffected — `service.py:1205` reads each row's stored `initial_equity` — so this stays
invisible until someone refreshes. Note this is dormant, not live: the daily cron is paused as of
PR #352 and `LEADERBOARD_DAILY_AUTO_DEPLOY` is off.

## Gate application

| Plan's outcome row | Status |
|---|---|
| Nemotron spread ≥1pp **and** control an outlier | ⛔ Cannot be evaluated — wrong capital base |
| Nemotron flat, DeepSeek spread ≥1pp | ⛔ Cannot be evaluated — wrong capital base |
| Both flat, **or** control mid-pack on both | ⛔ Cannot be claimed |

**Phase 0 has no result.** Do not proceed to Phase 1/2, and do not cancel Phase 2 on this
evidence either — nothing here bears on the question.

## To finish Phase 0

Re-run the DeepSeek leg at `initial_capital = 100000`, keeping the duplicated control:

```bash
# one instruction per process, as before, so a crash cannot lose the whole leg
python dashboard/scripts/probe_instruction_sensitivity.py \
  --models deepseek --initial-capital 100000 \
  --instructions aggressive_momentum --out probe-e1.json
# ... likewise contrarian_reversion, then control_nonsense TWICE (e3, e4)
```

The run header now prints the capital and its source, and the resolution guard refuses to spend
if the base is too coarse — so a repeat of this mistake fails closed instead of billing.

- **Measured cost:** ~$0.47/run × 4 = **~$1.86**. Phase 0 total would reach ~$4.10, inside the
  plan's $4.97 sanction.
- **Keep both controls.** Running the control twice is the only thing that revealed the noise
  floor, and at $100k it is the only way to separate model nondeterminism from share granularity.
- **Consider also running one leg with no `strategy_prompt`** (i.e. stock `SAFE_TRADING_PROMPT`)
  as an upper anchor. The published run scored +7.49% at default temperature; an anchor at $100k
  and `temperature=0` isolates how much of the gap is prompt-replacement versus temperature.

## Methodological lesson

The failure was not the $10k value. It was that **the resolution of the measurement was never
computed before paying for it.** One share was 2.49% of the portfolio; the effect under test was
~0.6pp. That comparison takes five minutes and costs nothing, and it invalidates the entire
experiment before a single API call.

It was caught only because the gap between the published DeepSeek curve (+7.49%) and every probe
run (−0.25% to −1.85%) was chased down instead of being written off as "different prompt." An
unexplained 8pp sitting next to a 0.66pp "signal" is the finding, not a footnote — whenever a
control comparison is an order of magnitude larger than the effect, the apparatus is the suspect.
