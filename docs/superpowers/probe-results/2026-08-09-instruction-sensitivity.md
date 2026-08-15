# Phase 0 probe: does a trading instruction change a pinned LLM's return?

**Date run:** 2026-08-15
**Plan:** `docs/superpowers/plans/2026-08-09-participatory-competition-phase-0-1.md`, Task 3
**Script:** `dashboard/scripts/probe_instruction_sensitivity.py` (commit `ded41af`)
**Verdict:** ⚠ **Nemotron leg FAILS. The gate is UNDETERMINED — the DeepSeek leg was not run.**

---

## What the gate is for

Phase 2 lets users compete by writing a trading instruction. That only means
something if the instruction measurably moves the result. If it does not, the
leaderboard ranks noise and the competition cannot be won on merit. This probe
runs one pinned model over one fixed window with six different instructions and
asks whether the returns separate.

## Setup

| | |
|---|---|
| Model | `nvidia/nemotron-3-nano-30b-a3b` via OpenRouter, `temperature=0` |
| Window | 2026-04-15 → 2026-05-15 (the contest window, `dashboard/config/leaderboard.json`) |
| Steps | 161 hourly decisions per run, 6 runs |
| Capital | $10,000 |
| Actual spend | **$0.3555** (plan estimated $4.97 for both models; only the Nemotron leg was run) |

Run as three parallel processes of two instructions each; results merged from
`probe-g{1,2,3}.json`. Each run used its own fresh `PortfolioManager` over one
shared bar fetch.

## Results

| Instruction | Return | H6 coverage | Trades | Cost | Valid |
|---|---:|---:|---:|---:|:--:|
| **`control_nonsense`** ★ | **+0.35%** | 97.5% | 76 | $0.0445 | ✅ |
| `verbose_analytical` | +0.34% | 98.8% | 82 | $0.0541 | ✅ |
| `contrarian_reversion` | +0.18% | 98.8% | 446 | $0.0552 | ✅ |
| `equal_weight_hold` | +0.05% | **89.4%** | 10 | $0.0733 | ❌ **INVALID** |
| `defensive_cash` | +0.00% | 98.8% | 0 | $0.0589 | ✅ |
| `aggressive_momentum` | −0.28% | 99.4% | 517 | $0.0694 | ✅ |

★ = the control. Its text is deliberately meaningless: *"The weather is pleasant
today. Consider the colour blue. Bananas are a type of fruit."*

- **Spread (all 6):** 0.63pp
- **Spread (5 valid):** 0.63pp
- **Stdev (valid):** 0.26pp
- **Threshold:** ≥1.00pp → **not met**

## Where the control landed — the load-bearing result

**The nonsense instruction finished first**, 0.01pp above the best real
instruction (`verbose_analytical`, +0.34%) and with a near-identical trade count
(76 vs 82).

The plan's failure condition is worded as "the control scores mid-pack". This
outcome is *not* mid-pack — it is rank 1 of 5 — and that is worse, not better.
A mid-pack control means the model responds to *having* an instruction rather
than to its content. A control that ties the best seeded instruction means the
five carefully written instructions failed to beat gibberish on return.

**A caution about the script's own outlier test.** It computes
`control < min(seeded) or control > max(seeded)` — a pure boundary check — so it
printed `OUTLIER (good)` for a 0.01pp margin, which inverts the meaning. The
merge step reports the *margin* instead. Any future reading of this gate should
use the margin, not the boolean. (Same class of defect as the
`payload.period !== 'live'` banner check in `leaderboard.js`: a test that looks
correct and passes on the exact case it exists to catch.)

## Behaviour separated even though return did not

| Instruction | Trades |
|---|---:|
| `aggressive_momentum` | 517 |
| `contrarian_reversion` | 446 |
| `verbose_analytical` | 82 |
| `control_nonsense` | 76 |
| `equal_weight_hold` | 10 |
| `defensive_cash` | **0** |

`defensive_cash` held equity at exactly $10,000.00 for all 161 steps — the model
followed "stay in cash" literally and perfectly. The instruction axis clearly
reaches the model's *behaviour*. It just does not reach its *return* on this
model over this window.

This is the finding to carry forward: **instruction compliance is not the same
measurement as instruction performance**, and only the second one is what a
leaderboard ranks.

## Two hazards this probe surfaced for Phase 2

**1. An instruction can silently disqualify itself under H6.**
`equal_weight_hold` ("spread the money evenly across many of the available
stocks") makes the model emit one action per DJIA symbol. The response exceeds
`LLM_MAX_OUTPUT_TOKENS` and arrives truncated —
`Expecting ',' delimiter: line 209 column 6 (char 7204)`,
`Bracket mismatch: 24 open, 23 close`. 18 steps failed all three repair attempts
and fell back to rule-based, dropping coverage to 89.4%, under
`MIN_LLM_DECISION_COVERAGE = 0.95`.

Note what that costs: **$0.0733, the most expensive run in the set, for the least
usable output.** Truncated calls bill in full. If users can write instructions,
some will write ones that induce verbose output, pay the most, and be rejected by
the integrity guard with no explanation. Phase 2 needs to surface this at
authoring time, not at publish time.

**2. There is no run-to-run noise estimate.** Every number here is a single
sample at `temperature=0`. A 0.63pp spread cannot be separated from model
nondeterminism without a repeat. One repeated instruction costs ~$0.07.

## Context: the window is not flat, and the model is the worst on the board

Baselines over this exact window, from `agent_runs`:

| Baseline | Return |
|---|---:|
| `spy_index` | +5.95% |
| `equal_weight_djia` | +5.04% |
| `buy_hold_djia` | +4.87% |
| `djia_index` | +2.24% |

The market rose ~5%. The window has headroom; a 0.63pp spread is not an artifact
of a dead tape.

Published model entries on the same window span **+7.49% (DeepSeek V4 Pro) to
−0.22% (Nemotron 3 Nano 30B)** — a 7.71pp range across *model identity*, versus
0.63pp across *instruction* on Nemotron. Model choice moved return ~12× more than
instruction did.

**This is a confound, and it is why the gate is undetermined rather than failed.**
Nemotron is the weakest entry on the board — below every passive baseline, and
the only LLM entry with a negative return. Its own published `SAFE_TRADING_PROMPT`
run scored −0.22%; all six probe instructions cluster within ~0.6pp of that.
Concluding "instructions do not move returns" from the least capable available
model would be testing whether coaching improves marathon times on a subject who
cannot run. A weak model's *outcome* can be insensitive to instruction while its
*behaviour* is highly sensitive — which is exactly the 0-to-517-trade result above.

## Gate application

| Plan's outcome row | Status |
|---|---|
| Nemotron spread ≥1pp **and** control an outlier | ❌ Not met — spread 0.63pp, control margin 0.01pp |
| Nemotron flat, DeepSeek spread ≥1pp | **⏸ Untested** — this is the open branch |
| Both flat, **or** control mid-pack on both | ⏸ Cannot be claimed — DeepSeek not measured |

**Do not proceed to Phase 1/2 on this evidence.** Nor should Phase 2 be
cancelled on it: the plan's own table routes a flat Nemotron to the DeepSeek
leg, and that leg is the one with a plausible chance of separating, since
DeepSeek is the only entry that beat the passive baselines.

**Estimated cost of the DeepSeek leg:** $0.756/run.
- All 6 instructions: **~$4.54**
- Minimum informative subset — `aggressive_momentum`, `defensive_cash`,
  `control_nonsense` (max activity, zero activity, control): **~$2.27**

Awaiting the repo owner's decision on that spend.
