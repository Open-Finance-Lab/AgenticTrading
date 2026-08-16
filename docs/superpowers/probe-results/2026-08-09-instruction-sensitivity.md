# Phase 0 probe: does a trading instruction change a pinned LLM's return?

**Date run:** 2026-08-15 (Nemotron @ $10k, invalid), 2026-08-16 (DeepSeek @ $10k, invalid;
DeepSeek @ **$100k**, the result)
**Plan:** `docs/superpowers/plans/2026-08-09-participatory-competition-phase-0-1.md`, Task 3
**Script:** `dashboard/scripts/probe_instruction_sensitivity.py` (commit `9388465`)
**Verdict:** ✅ **GATE PASSES** on DeepSeek V4 Pro. A trading instruction moves the return by
**3.61pp** against a **0.20pp** run-to-run noise floor — signal/noise **18×**.
**Spend:** $3.6100 of the plan's $4.97 sanction ($2.2431 on the invalid legs, $1.3669 on the valid one).

---

## The result

Re-run at the board's capital base, `initial_capital = $100,000`, DeepSeek V4 Pro, `temperature=0`,
161 decision steps, window 2026-04-15 → 2026-05-15. The control was run **twice** so that the
noise floor is measured rather than assumed.

| Run | Instruction | Return | Coverage | Calls | Decisions | Trades | Cost |
|---|---|---:|---:|---:|---:|---:|---:|
| e1 | `aggressive_momentum` | **+3.83%** | 100.0% | 185 | 161 | 656 | $0.4427 |
| e3 | `control_nonsense` **A** | +0.33% | 100.0% | 177 | 161 | 712 | $0.4489 |
| e4 | `control_nonsense` **B** | +0.13% | 99.4% | 185 | 160 | 661 | $0.4753 |

| Measure | Value |
|---|---:|
| noise floor — same instruction, two runs | **0.20pp** |
| signal — seeded vs. control mean (+0.23%) | **3.61pp** |
| signal / noise | **18.1×** |
| naive spread (max − min), plan's ≥1pp threshold | 3.71pp |
| seeded return outside the control band? | **yes** |

All three cleared H6 (`MIN_LLM_DECISION_COVERAGE = 0.95`), so every curve here is publishable.

> **The returns above were computed with the probe's pre-review denominator** —
> `curve[-1]/curve[0] − 1`, where `curve[0]` is equity *after* the first step's trades — and
> the review of this PR replaced it with the board's own definition,
> `(final − initial_capital) / initial_capital`. The old denominator differs per instruction
> (an instruction that deploys capital immediately gets a smaller base than one that holds
> cash), so a slice of the measured spread was apparatus rather than signal. The raw curves
> were not persisted to the repo, so these three figures **cannot be re-derived without
> re-running the leg**. At $100k the first-step drift is a fraction of a percent against a
> 3.61pp signal and a 0.20pp noise floor, so the 18× verdict is not in question — but treat
> the exact decimals as definition-dependent, and quote future runs off the corrected metric.
> This is the same class of error as the capital mismatch below, caught before it mattered
> rather than after.

The pass holds on the strict reading, not just the naive one. The plan warns that a spread can
come from *having* an instruction rather than from its content; here the two controls — same
nonsense instruction, same seed, same temperature — land 0.20pp apart, and the seeded run sits
**18× that distance away and outside their band**. On the invalid $10k leg the opposite was true:
one control fell *inside* the seeded range.

The returns are also coherent against the rest of the board. `aggressive_momentum` at +3.83% sits
inside the passive baseline band for this window (+2.24% to +5.95%), while both controls badly
underperform passive despite ~700 trades each — nonsense instruction produces churn without edge,
which is what a working instrument should show.

**Two limits, stated plainly.** The noise floor is one pair, so it is an estimate with no error
bar of its own; and `contrarian_reversion` was dropped to stay inside the sanction, so this
measures instruction-vs-control, **not** instruction-vs-instruction. Neither weakens the gate as
the plan posed it — "does an instruction move the return at all" is answered — but ranking two
*good* instructions against each other is untested, and Phase 2 depends on that.

## Read this first: an earlier draft of this file said the gate fails

That conclusion is withdrawn, and the numbers above are why. Both earlier legs ran at
`initial_capital = $10,000` while **every published board curve they were compared against was
computed at `$100,000`.** At $10,000 a single DJIA share is 2.49% of the portfolio, against an
effect under test of ~0.6pp — the measuring instrument was four times coarser than the thing it
was measuring, and it reported a null.

It was not merely unsupported; it was the kind of confident wrong answer this probe was written to
prevent, and it survived a full write-up because nobody checked the resolution of the apparatus
against the size of the effect. The record of that failure is kept below deliberately.

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

Four findings do not depend on return, and are worth carrying into Phase 2 regardless:

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

**4. `temperature=0` did not pin the outcome, at either capital base.** Two identical control runs
landed 1.18pp apart at $10k and **0.20pp apart at $100k**. The re-run settles the attribution the
$10k leg could not: most of that 1.18pp was share granularity, but 0.20pp of genuine run-to-run
variance survives at a capital base where rounding is negligible. `temperature=0` buys a small
noise floor, not a reproducible one — so a Phase 2 leaderboard cannot treat a re-run of the same
entry as guaranteed to reproduce its rank, and margins inside ~0.2pp are not real.

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

### Further defects, found reviewing this PR and fixed in it

The capital guard above was the first pass. A review of the script found nine more, in three
families — all fixed in this branch:

**It could still have dirtied the prod seed database.** The `DATABASE_PATH` redirect used
`os.environ.setdefault` *after* `load_dotenv`, and `.env.example` ships
`DATABASE_PATH=dashboard/storage/data/backtest.db` **uncommented** (line 204). On any checkout
whose `.env` came from that template the setdefault was a no-op and the backend's lazy
`CREATE TABLE`/`ALTER` ran against the committed prod DB — issue #244's exact trap. Reproduced by
checksum (the file genuinely changed), then fixed by forcing the assignment, matching
`backfill_runs_to_postgres.py:117`.

**It could lose runs it had already paid for.** `_write` ran only after the whole loop, so an
exception in any single run discarded every completed one; and `_write` itself raised on a
missing parent directory. Both now: persist after every completed run, `mkdir(parents=True)`, and
a `_write` that dumps to stdout rather than throwing. A failed run is recorded and the leg
continues, with the verdict forced to INCONCLUSIVE.

**The gate could pass on a dead axis.** The `≥1pp` test used the spread across *all* runs, which
includes the control — so five identical seeded instructions plus one far-off control cleared it
and printed "separates instructions", the precise outcome the control exists to detect. Spread is
now measured over seeded runs alone; **every** control must clear the margin (was: any); and when
only one seeded instruction ran, PASS prints an explicit note that instruction-vs-instruction is
untested. Also fixed: the resolution guard failed *open and silently* when no price could be read,
`opens[len//2]` was the upper-middle rather than the median, and the sampled price came from
`reference_start_date` — a month before the window whose resolution was being judged.

Still open, and the more general form: a probe that quotes a stored run should **assert the
stored run's parameters match its own** (`initial_equity`, data feed, window) rather than trusting
that the config which produced them is the config it reads. Same shape as the feed-drift trap
documented under `ALPACA_DATA_FEED` in `CLAUDE.md`.

## Related repo defect — filed as issue #365

`_find_cached_run` (`domain/leaderboard/service.py:615`) keys on
`(mode, start_date, end_date, llm_model)` — **not** `initial_equity`. Combined with the config
drift above and the `auto_compute` split:

- the **5 baselines** (`auto_compute` unset → true) are recomputed by `?refresh=true` at the
  config's **$10k**;
- the **7 LLM entries** (`auto_compute: false`) are written only by `deploy_model_run()`, so they
  stay at **$100k**.

One refresh therefore leaves the board comparing $10k baselines against $100k model curves — the
exact comparison the board exists to make — with `equal_weight_djia` and `buy_hold_djia` carrying
~35% stranded cash, making the models look better than they are.

Display is unaffected — `service.py:1205` reads each row's stored `initial_equity` — so this stays
invisible until someone refreshes. Dormant, not live: the daily cron is paused as of PR #352 and
`LEADERBOARD_DAILY_AUTO_DEPLOY` is off.

**`strategy_prompt` lands in the same hole, and this PR is what puts it there.** The key does not
include it either, so editing an Open Track entry's instruction and redeploying returns the cached
row — publishing the *old* instruction's curve under the *new* instruction's name. Nothing sets
the key in `leaderboard.json` yet, so it is latent rather than live, and widening the cache key is
#365's fix rather than a second one. What this PR does do is close the *auditability* half:
`_llm_run_metadata` now records the instruction the strategy actually ran with, so a substituted
curve is at least detectable after the fact instead of leaving no trace at all.

## Gate application

| Plan's outcome row | Status |
|---|---|
| Nemotron spread ≥1pp **and** control an outlier | ⛔ Not evaluated — the Nemotron leg ran at $10k and was not re-run |
| Nemotron flat, DeepSeek spread ≥1pp | ✅ **This is the outcome.** DeepSeek separates at 3.71pp; Nemotron is unmeasured, not flat |
| Both flat, **or** control mid-pack on both | ❌ Refuted for DeepSeek — the control is not mid-pack, it is the floor |

**Phase 0 passes, and DeepSeek V4 Pro is the pinned model** for every later task, which is the
disposition the plan specifies for this row. Say "unmeasured" rather than "fails" about Nemotron:
its leg was invalidated by capital, not by a null result, and nothing here licenses a claim about
small models. If Phase 2 ever wants a second model on the board, that leg has to be paid for.

## What is still needed before Phase 2

The gate answers *whether* an instruction moves the return. Phase 2 ranks instructions against
each other, and that is a strictly harder question this leg does not answer:

- **Instruction-vs-instruction separation is untested.** `contrarian_reversion` was cut for
  budget. Two *plausible* strategies may well land inside each other's noise even though nonsense
  is separable from momentum — which is the case that decides whether a board of user entries is
  a ranking or a lottery.
- **The noise floor rests on one pair.** 0.20pp from two runs is enough to clear an 18× margin;
  it is not enough to size a leaderboard's tie-breaking.
- **The default-prompt anchor was never run.** The published DeepSeek run scored +7.49% at default
  temperature and no `strategy_prompt`; this leg's seeded run scored +3.83% at `temperature=0`.
  How much of that 3.7pp is prompt-replacement versus temperature is still unattributed, and it
  matters because Phase 2 replaces the house prompt for every entrant.

Cost for those, at measured prices: **~$0.46/run**.

**Note on the estimate in the previous draft.** It said "budget ~$0.70–0.80/run" by scaling from
the published $100k run's $0.756. Actual was **$0.4556/run** — the leg came in at $1.3669 against
a $2.10–2.40 forecast. So the earlier 1.63× "more capital costs more" inference does not hold
either; two consecutive cost predictions here were wrong in opposite directions, and per-run cost
on this path should be read off a measured run rather than extrapolated from a stored one.

## Methodological lesson

The failure was not the $10k value. It was that **the resolution of the measurement was never
computed before paying for it.** One share was 2.49% of the portfolio; the effect under test was
~0.6pp. That comparison takes five minutes and costs nothing, and it invalidates the entire
experiment before a single API call.

It was caught only because the gap between the published DeepSeek curve (+7.49%) and every probe
run (−0.25% to −1.85%) was chased down instead of being written off as "different prompt." An
unexplained 8pp sitting next to a 0.66pp "signal" is the finding, not a footnote — whenever a
control comparison is an order of magnitude larger than the effect, the apparatus is the suspect.

The correction did not merely widen the error bars, it **reversed the verdict**: the same model,
the same window and the same instructions went from a 0.55× signal-to-noise null to an 18× pass
on a capital change alone. A coarse instrument does not return a noisy version of the right
answer — here it returned a confident wrong one, and the whole participatory competition would
have been cancelled on it for $2.24.
