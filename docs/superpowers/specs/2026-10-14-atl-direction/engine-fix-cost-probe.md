# Backtest engine rewrite: cost probe (My Agents, US equities)

## 1. Bottom line

1. **This is a rough probe.** It rests on 11 judgement estimates, each reviewed once. Only K1, K4 and K5 were prototyped, on scratch copies. B was neither reviewed nor prototyped. Two critic passes checked arithmetic and coverage; the last pass's notes are folded in. Treat every figure as ±50%. Produced 2026-10-13 by workflow `wf_1d7d708f-895` against `origin/main` @ `2dc8203a`.
2. **A, patch in place:** **39 / 59 / 100** engineer-days (low / likely / high), after de-duplication and including the differential report. This uses the *measured* reading: the K1 prototype showed the daily cases pass through the adapter without D16. On the brief's case map, which requires D16, A costs **41 / 62 / 105**. Production code is about 3.6–6.0k lines.
3. **B, own core:** **43 / 62 / 96** as estimated. Applying the upward correction the reviews gave A gives about **45 / 66 / 101**. Production code is about 4.1–6.5k lines. B compares like-for-like with A's measured reading, because B's core passes the daily cases without D16.
4. **Calibration:** at the team's median past throughput (358 production lines per active day, plus 25–30% rework), the same code takes about **17 days for A and 19 for B**. The judgement figures imply about 81 lines a day. That is below the historical 25th percentile of about 135. At that 25th-percentile rate the calibrated high (about 58 for A, 63 for B) roughly equals the judgement likely. So the day figures may be about 3.5× too high; the ratios between options hold either way.
5. **Adopt vs build:** an adopted engine could replace about 11 likely days of A (19%), or 9–11 days of B (about 15–18%). The research plan puts a 2-week spike before any adapter or port. Adoption is therefore **unlikely to pay back on this scope**, unless the spike is mostly calendar time. The adapter, the shims and dependency upkeep are unpriced, and they only widen the gap.
6. **If we build:** use B's design, and start with about 15 days of work that is the same under A, B and adoption.

## 2. Per-cluster estimates

How to read the table:
- **Days** are the reconciled figures from the reviews.
- **Cases alone** uses the brief's case map plus three additions I verified: T05 needs D1, O01 and O02 need D1, and S01 needs D14.
- **"(measured)"** marks extra cases that the recorded failure lines show would also flip.
- **Engine share** is the fraction of the cluster's work an adopted engine would do.

| Cluster | Defects | Cases alone | Cases with others (needs) | Days L / Lk / H | Prod / test LOC | Conf. | Engine share |
|---|---|---|---|---|---|---|---|
| K1 Fill timing, calendar | D2, D19, D4 | none by map (T02, T02d measured) | T05 (K3, K5); F02, F03 (K4, plus D16 on map); CA03 (K7, by map) | 2.2 / 3.2 / 6.5 | 155–300 / 280–520 | high (D4 med) | 0.40 |
| K2 Executor cash, liquidity | D5, D6, D7, D11 | C01, C03, L01 (C02, C04 measured) | C02, C04 (K3 by map); F05 (K4, K5); C01/DL (K10); C04/DL (K3, K10) | 3.2 / 4.75 / 7.75 | 255–445 / 390–690 | med (D11 low) | 0.41 |
| K3 Order lifecycle | D8 | T03, T04 | C02, C04 by map (K2); T05 (K1, K5); C04/DL (K2, K10) | 2 / 3.5 / 6 | 120–230 / 190–360 | med | 0.30 |
| K4 US costs | D10 | none | F01, F04 (K5); F05 (K2, K5); F02, F03 (K1, plus D16 on map) | 3 / 4.7 / 6.8 | 310–460 / 450–650 | med | 0.25 |
| K5 Ledger, marking | D1, D3, D14 | M01, M02 | F01, F04, F05, CA01, S01 (D and DL), O01, O02, T05 | 1.75 / 3 / 5.5 | 128–275 / 230–550 | high | 0.50 |
| K6 Sharpe, frontend recompute | D15, D20 | M03, M03b | — (D20 has no case) | 4 / 5.5 / 9 | 330–540 / 450–730 | med | 0 |
| K7 Corporate actions | D13 | CA02 (CA03 measured, by coincidence) | CA01 (K5); CA03 (K1 by map); CA04 (K9) | 5.5 / 8 / 12 | 750–1,200 / 500–850 | med-low | 0.20 |
| K8 Fractional, target weight | D9 | S02 | S01, S01/DL (K5) | 5 / 7.5 / 11 | 450–600 / 700–900 | low | 0.20* |
| K9 Order types (core) | D12 | O03, O04 | O01, O02 (K5); CA04 (K7) | 4.4 / 6.3 / 10.3 | 450–700† / 450–700 | med | 0.40 |
| K10 LLM translator | D18 | none | C01/DL (K2); C04/DL (K2, K3) | 2.5 / 4 / 7.5 | 115–225 / 190–330 | med-low | 0.06 |
| K11 Integration, cutover | cross-cutting | none (needed to ship) | — | 8.3 / 12 / 20.5 | 810–1,320 / 1,240–2,170 | med | 0 |
| **Base sum** | | | | **41.85 / 62.45 / 102.85** | **3.9–6.3k / 5.1–8.5k** | | |

\* My estimate; the K8 probe returned no share. † LOC from before the review; the review added 1.3 days without adding LOC.

**Options kept out of the base.** Both A and B exclude these. D16 is the exception to "unmeasured": the brief's map requires it, the measured reading does not (§3, §6).

| Option | Days | LOC (prod / test) |
|---|---|---|
| D16 daily timeframe (required on the brief's map only) | 2.0 / 3.2 / 5.0 | 150–260 / 200–320 |
| D12.enforce (stop-loss / take-profit enforcement) | 2.25 / 3.5 / 5.5 | 200–320 / 300–420 |
| D12.disclaim (stop advertising the fields) | 0.25 / 0.4 / 0.75 | 20–60 / 20–50 |
| K1 extra session past `end_date` | +1.5–3 | 80–130 prod |
| Full D4 fix (only if `vnpy_simulation` stays) | +1.5–2 | ≈80 prod |
| Exact gross curve by tape replay | +1–2 | — |
| Port the custom-algo script to the new loop | +1.5–3 | — |

**Review adjustments I accepted**

- **K1**
  - **D2, +0.4.** The probe's fix ("read equity before the catch-up") drops the 15:35–15:55 marks. It also still shows the post-trade review a fill at the next open. The correct order is: mark the same day, run the review, execute, then mark the next open.
  - **D19, +0.1.** Needs a half-day test with extended-hours bars, because `is_in_session` needs the date.
  - **D4, −0.2.** The narrow guard can never fire on a reachable US + Alpaca path.
  - **D16, +0.2.** D16 is now an option, not in the base.
  - **T05 also needs D1.** The prototype left exactly one failure: equity 990 vs 1000.
  - **Sequencing, no days.** D2 ships only with or after K5.
- **K2**
  - +0.25 frontend reason labels and +0.25 run provenance. Both are real, and both are removed in §3 as duplicates.
  - +0.1 adapter policy injection.
  - −0.25 participation-cap test churn: the synthetic volumes in tests sit far above a 10% cap.
  - Gate the new path on `intraday_mode`, not on market. `vnpy_simulation` is a US market with a 60m source. No days.
- **K3** (net about zero; 2 / 3.5 / 6 stands)
  - +0.25 cache-buster bump.
  - +0.15 guard for hold and zero-share actions at the new `execution.py:526` event.
  - +0.2 decision on the hosted-runtime sentinel step.
  - −0.4 drop `order_outcome_counts`: no case or policy needs it.
  - The "+1 LLM call per run" risk is 0 against today, because the 16:00 decision is already a step.
- **K4**
  - Five additions: metadata snapshot pin, a cost axis in `diff_backtest_runs.py`, fee-rate sourcing, docs, review rounds. They itemise to +1.1, but the reviewer moved the total by +0.7. I kept its 4.7 (listed as unreconciled in §7).
  - The production US profile must attach after K2. In the scratch run, F05's full-cash buy turned into a rejection once fees existed without P5.
  - The DJIA index stays uncosted. This corrects a mis-statement and costs no days.
- **K5**
  - **D1, 1.0 → 0.6, and production LOC 40–90 → 15–40.** A pre-mark loop of about 14 lines flipped M01's equity, initial-equity and drawdown checks on a scratch copy.
  - **D14, 2.5 → 2.0.** Neither the display row nor a Decimal ledger is needed for M02 or P9.
  - **D3 → 0.3.** An explicit initial-equity stamp is needed outside intraday mode.
  - +0.25 day-end snapshot (shared with K1) and +0.1 default for an old Live-board snapshot (shared with K11). Both are removed in §3.
- **K6** (likely +0.5)
  - Opt-in metrics-grid switch, net +0.35. I verified `baseline_worker.py:136-146`: step sessions build `HourlyBacktester` and run both baselines.
  - +0.1 flip M03 and M03b to PASS instead of deleting them; the harness requires a prediction for every case.
  - A refresh guard for the risk-free table.
  - Two items moved only the low and high bounds: the risk-free rate for legacy rows in chart-data (+0.25) and methodology text (+0.4).
  - Engine share 0.1 → 0: the research keeps metrics ATL-owned.
- **K7**
  - Additions:
    - +0.25 a fourth call site, before `get_portfolio_state` at `engine.py:2198`.
    - +0.25 make the forward-filled price cache split-aware.
    - +0.75 split-adjusted baseline bars. Strike this if the owner prefers raw baselines.
    - +0.25 a gate so `CustomAlgoBacktester` does not inherit the change.
    - +0.25 Alpaca probing and a recorded wire fixture.
    - +0.5 review buffer.
  - Removals:
    - −0.4 no on-disk events cache in v1: a cached empty answer would hide an outage for a week.
    - −0.1 the cancel stub belongs to K9.
    - −0.1 overlap with K11.
  - CA03 flips with K7 alone by coincidence: it checks only cash, positions and equity.
- **K8**
  - I took the review's 5 / 7.5 / 11 whole. The probe returned no itemised work, so the review's blind re-derivation is the only priced source.
  - Persistence (+1.5) is removed in §3, because K11's INT-1 owns it.
  - S01 also needs D14. Verified: the S01/DL entry in `known_failures.json` has a `realized_pnl … actual None` line.
- **K9 (core), +1.3 likely** (per-defect 5.0 → 6.3)
  - Items:
    - +0.75 run the order sweep before the state read at `engine.py:2191`.
    - +0.3 per-action fill price.
    - +0.35 attribute working-order fills on the decision tape.
    - +0.2 frontend, removed in §3.
  - The items sum to +1.6. The review shows no offset for the remaining 0.3 (§7).
  - O01 and O02 need the D1 reorder. Verified: O01 checks equity 1005 at HC(DAY,1) (`cases.py:667`, `:675`).
  - O04 does not need K8: the order book rejects the fractional order before the executor sees it.
- **K10**
  - Additions:
    - +0.4 pipeline orders envelope: a missing quantity becomes 0 at `pipeline_runner.py:236-248`.
    - +0.3 measure the production SELL-size distribution.
    - +0.25 run-level metric.
    - +0.25 conformance re-record.
  - −0.75 drop `position_fraction`: no shipped template emits `risk_actions`.
  - K10 must follow K2: a forwarded −5 sell mints shares at `execution.py:812-826`.
  - K10's LOC in the table is from before the review, as K9's is.
- **K11**
  - Additions:
    - +0.25 INT-1, the first `ALTER COLUMN TYPE` in the Postgres twin.
    - +0.1 extra cache-buster pins.
    - +0.2 custom-algo smoke test.
    - +0.3 Live-board snapshot restore compatibility.
  - Reductions:
    - INT-3 −0.45, cut down to the pooling key only: nothing recomputes cached leaderboard rows automatically.
    - INT-9 and INT-12 −0.4: they overlap the per-cluster re-records.

**Rejected or not applied**

- **K11 review: INT-3 engine share "1.0 conditional".** Rejected. An adopted engine does not do this work. The work disappears whenever the old executor is left untouched, so I book it as an A-only isolation cost (§4), not as an adoption saving.
- **K9 review: cluster totals of 6.5 / 9.8 / 15.5.** These are below the review's own per-defect sums (6.65 / 9.8 / 15.8). I used the per-defect figures throughout: core 4.4 / 6.3 / 10.3, enforce 2.25 / 3.5 / 5.5.
- **K8 review: −0.5 qty_step overlap.** Not subtracted again in §3. The review's components sum to about 7.6 at their lows (about 8.1 at midpoints) before review rounds, against a 7.5 total, so the overlap already looks netted.
- **Evidence that removes defects from the case map.** This covers D16 for T02, T02d, F02 and F03; D8 for C02/D and C04/D; and D2 for CA03. It is not applied to the map curve, which the brief fixes. It is applied in full to the measured curve (§5).
  - I verified the C02/C04 part: their `known_failures.json` entries carry only D6/D7 symptoms and no invariant-7 line.
  - The headline uses the measured reading only because that is where A and B are like-for-like (§6).

## 3. De-duplication

| Step | Low | Likely | High |
|---|---|---|---|
| Sum of the 11 reviewed clusters (K9 at its per-defect sums) | 46.10 | 69.15 | 113.35 |
| − D16 daily timeframe (option) | −2.00 | −3.20 | −5.00 |
| − D12.enforce (option) | −2.25 | −3.50 | −5.50 |
| **= Base sum (§2)** | **41.85** | **62.45** | **102.85** |
| − overlaps 1–11 below | −5.25 | −6.70 | −8.10 |
| **= A fix work, de-duplicated** | **36.60** | **55.75** | **94.75** |
| + differential report (research §8 Phase 2; no A cluster priced it; B's figure) | +2.50 | +3.50 | +5.00 |
| **= A, measured reading (comparable with B)** | **39.10** | **59.25** | **99.75** |
| + D16, required on the brief's map | +2.00 | +3.20 | +5.00 |
| **= A on the brief's map** | **41.10** | **62.45** | **104.75** |

**Overlaps**, each counted once (low / likely / high). The credit goes to whichever cluster lands second in §5.

1. K3 and K9 build one order registry and one set of cancelled/expired plumbing; credited to K9: 0.75 / 1.0 / 1.5.
2. K8's fractional persistence is the same work as K11 INT-1, which owns it: 1.5 / 1.5 / 1.5.
3. K8's qty_step plumbing: 0 / 0 / 0. It looks already netted inside K8 (§2). It was 0.25 / 0.5 / 0.75 in the previous draft.
4. K1 and K5 both price the day-end snapshot; credited to K1: 0.25.
5. K2's run provenance is K11 INT-2: 0.25.
6. Frontend work moves to K11 INT-6: K2 0.25 + K3 0.65 + K4 0.4 + K8 0.2 + K9 0.2 = 1.7 likely (1.0 / 2.0). The K3 and K4 parts are my split of their buckets.
7. K4's fee columns in both store twins are K11 INT-4: 0.5 / 0.6 / 0.75.
8. Run-metadata allow-list blocks move to K11 INT-5: K4 0.3 + K6 0.3 + K5 0.15 = 0.75 (0.5 / 1.0). This is my split.
9. K5's +0.1 restore of an old Live-board snapshot is the same work as K11's +0.3: 0.1.
10. INT-5 and INT-8 items that nothing reaches: K6's design is additive, with an opt-in switch, so `baselines.py:98` and custom-algo `:353` stay on the legacy Sharpe. K3 dropped the `order_outcomes` tally. Credit: 0.3 / 0.4 / 0.5.
11. One opt-in flag can serve K6's metrics switch, K7's custom-algo gate and part of K11's smoke test; credited to K6: 0.1 / 0.15 / 0.25.

**LOC.** Removing about 0.4–0.5k duplicated production lines, and adding about 0.1–0.2k for the differential tooling (which B's integration LOC already includes), gives A about **3.6–6.0k production and 4.7–7.9k test**. K9's +1.3 review days carry no LOC. On the map reading, add D16's 150–260 / 200–320.

**B's own A figure is a coincidence, not a cross-check.** B's estimate put a comparable A at about 39 / 60 / 100. It got there by removing only 3–4 days of overlap and adding no differential. I remove 6.7 days and add 3.5, and the differences cancel. On B's own basis, an A that also included the differential would be about 63.

**B for comparison**
- As estimated: 42.95 / 62.1 / 96.5. That is components 26.85 / 38.1 / 59 plus integration 16.1 / 24 / 37.5 (arithmetic checked).
- The reviews raised the A-cluster sums from 40.95 / 61.8 / 103.25 to 46.1 / 69.15 / 113.35: **+12.6% / +11.9% / +9.8%**.
- I applied that to B's unreviewed lines, leaving out B's own review-rounds line because that line already is a review buffer. The base is C1–C7, C12, C13, the conformance adapter, the test port and the performance check: 19.1 / 28.5 / 44.5. This adds 2.4 / 3.4 / 4.35, giving **B ≈ 45.4 / 65.5 / 100.9**. This is my calibration, not a review.

**Calibration cross-check.** The rates are the quartiles of the calibration input's own 16 per-PR values: about 135 (25th percentile), 358 (median) and about 500 (75th percentile). The input's stated 110 / 300 / 490 understate its own data. Trimming the #506 outlier gives 120 / 336 / 493 and a likely of about 18 for A and 20 for B, the same conclusion.

| | Prod LOC | LOC-based days (500 / 358 / 135 per day, + rework 25 / 27.5 / 30%) | Judgement days | Implied LOC/day (likely) |
|---|---|---|---|---|
| A (measured) | 3.6–6.0k | 9.0 / 17.1 / 57.8 | 39 / 59 / 100 | ≈81 |
| B | 4.1–6.5k | 10.3 / 18.9 / 62.6 | 43 / 62 / 96 | ≈85 |

The gap is not reconciled, for three reasons:
- The calibration counts only days with commits. Design, review and production verification are unrecorded.
- Most calibration PRs are single-subsystem work. This work cuts across shared code that tests pin heavily.
- The estimates' test-to-production ratio is about 1.3, against the repo's 2.08. If anything, that says the test work is under-scoped.
- The calibration's own caveat cuts the other way: its LOC per active day is agent throughput from parallel sessions sharing a day, so it may understate one engineer's effort per line.

## 4. Engine-internal vs ATL-side

| | Low | Likely | High | Share (likely) |
|---|---|---|---|---|
| A total (measured) | 39.1 | 59.25 | 99.75 | |
| A engine-internal (share × de-duplicated cluster days) | 7.0 | 11.0 | 19.0 | 19% |
| A ATL-side | 32.1 | 48.25 | 80.75 | 81% |
| — of which A-only isolation (K2 policy scaffold ≈1.0, INT-3 0.8) | | ≈1.8 | | 3% |
| B total (as estimated) | 42.95 | 62.1 | 96.5 | |
| B engine-internal, undiscounted (C2–C7) | 9.1 | 13.75 | 22.0 | 22% |
| — replaceable by adoption, per B's own gap analysis | | 9–11 | | ≈15–18% |
| B ATL-side | 33.85 | 48.35 | 74.5 | 78% |

**How A's engine-internal figure is computed.** Each cluster's engine share is applied to its days after the §3 credits. The credited frontend, persistence, allow-list and provenance items moved to K11, which has a share of 0. For example, the likely is K9 5.1 × 0.4 + K2 4.25 × 0.41 + K7 8.0 × 0.2 + …, which comes to 11.0.

Some ATL-side items that the reviews added are still weighted by a share, such as K7's split-adjusted baselines and fixture work. That makes the figure about 0.2 days high. The shares are judgement; no probe read AKQuant's source.

A's 11.0 and B's 9–11 are the comparable "what adoption would do" figures. B's 13.75 is the undiscounted core.

**Against research §8.** The research lays out these phases:
- Phase 0: a spike on a throwaway branch, 2 weeks.
- Phase 1: SimulatorPort, the AKQuant adapter, market packs, drivers, writer, compatibility facades, and the 83 engine test files ported as facade acceptance tests.
- Phase 2: the differential against the current engine.
- Phase 3: shadow runs at production shape.
- Phase 4: cutover.
- Phase 5: decommission.

It gives no duration except the spike's 2 weeks.

The port, drivers, writer, facades, test port, differential and cutover are all ATL-side. B already prices their My Agents-scoped versions (C1, C12, C13 and the integration items). Adoption swaps out only C2–C7. In their place it adds the spike, the AKQuant adapter, and dependency upkeep. It also adds shims for the gaps the research lists:
- P5, P8 and P15;
- the volume-cap remainder, which AKQuant carries over where the owner chose to cancel it;
- zero-fee defaults;
- the stale mark on a split day.

**Break-even.** If 2 weeks means 10 working days, the spike alone costs about as much as the replaceable work (about 11 days for A, 9–11 for B), before the adapter and shims. Several of the spike's kill criteria (`…adoption-research.md:281-293`) test things outside this scope:
- 100 in-process sessions;
- the A-share 10送3转2派1 event;
- Live-board snapshot restore.

So much of the spike buys evidence that this rewrite does not need.

On the calibrated reading the ceiling shrinks further. C2–C7 is 1.32–2.04k production lines, about 6 days at the median rate, and A's 11.0 scales to about 3.

Adoption could pay back only in two cases:
- The spike is mostly calendar waiting. Upstream turnaround is one of its own kill criteria.
- The scope widens to A-share, the leaderboard and step sessions, where T+1, lots, limits and snapshots carry more of the load. This probe did not price that wider scope.

## 5. Fix-order curve (A)

Prerequisites:
- K5 goes before K1 and K9.
- K3 goes before or with K1.
- K2 goes before K4's production profile, before K9 (sells-first across merged fills), and before K10.
- K3 goes before K9 (one registry).
- INT-2 and INT-3 go before the first merge that changes results.

A case counts only when every defect it maps to is fixed. Marginal days are the cluster days after the §3 credits. `cases.py:1105-1170` holds 30 D predictions and 3 DL predictions, so the curve counts **33**.

**Map curve (the brief's map plus the three verified additions; D16 lands with K1)**

| # | Cluster | Marginal days | Cases flipped | Passing /33 | Cum. days | Cases/day |
|---|---|---|---|---|---|---|
| 0 | today | — | T01 | 1 | 0 | — |
| 1 | K11 ship gate (INT-2 stamp 1.0, INT-3 pool key 0.8) | 1.8 | — | 1 | 1.8 | 0 |
| 2 | K5 | 2.75 | M01, M02 | 3 | 4.55 | 0.73 |
| 3 | K3 | 2.85 | T03, T04 | 5 | 7.4 | 0.70 |
| 4 | K2 | 4.25 | C01, C02, C03, C04, L01 | 10 | 11.65 | 1.18 |
| 5 | K4 | 3.4 | F01, F04, F05 | 13 | 15.05 | 0.88 |
| 6 | K1 + D16 | 6.15 | T02, T02d, T05, F02, F03 | 18 | 21.2 | 0.81 |
| 7 | K9 core | 5.1 | O01–O04 | 22 | 26.3 | 0.78 |
| 8 | K8 | 5.8 | S01, S02, S01/DL | 25 | 32.1 | 0.52 |
| 9 | K10 | 4.0 | C01/DL, C04/DL | 27 | 36.1 | 0.50 |
| 10 | K7 | 8.0 | CA01–CA04 | 31 | 44.1 | 0.50 |
| 11 | K6 | 5.05 | M03, M03b | 33 | 49.15 | 0.40 |
| 12 | rest of K11 | 9.8 | — | 33 | 58.95 | 0 |
| + | differential report | 3.5 | — | 33 | 62.45 | |

**Measured curve (all three map removals applied, then re-sorted)**

| # | Cluster | Marginal days | Cases flipped | Passing /33 | Cum. days | Cases/day |
|---|---|---|---|---|---|---|
| 1 | K11 ship gate | 1.8 | — | 1 | 1.8 | 0 |
| 2 | K2 | 4.25 | C01, C02, C03, C04, L01 | 6 | 6.05 | 1.18 |
| 3 | K5 | 2.75 | M01, M02 | 8 | 8.8 | 0.73 |
| 4 | K4 | 3.4 | F01, F04, F05 | 11 | 12.2 | 0.88 |
| 5 | K3 | 2.85 | T03, T04 | 13 | 15.05 | 0.70 |
| 6 | K1 | 2.95 | T02, T02d, T05, F02, F03 | 18 | 18.0 | 1.69 |
| 7–11 | K9, K8, K10, K7, K6 (same as map) | 5.1 / 5.8 / 4.0 / 8.0 / 5.05 | as above | 22 / 25 / 27 / 31 / 33 | 23.1 / 28.9 / 32.9 / 40.9 / 45.95 | 0.78 / 0.52 / 0.50 / 0.50 / 0.40 |
| 12, + | rest of K11, differential | 9.8, 3.5 | — | 33 | 55.75, 59.25 | |

Notes on the curves:
- **Map ordering.** K2 alone flips only 3 cases (0.71 per day) against K3's 0.70. K3 goes first because it gates K1, and it lets K2 flip 5 cases.
- **Measured ordering.** Once D8 is off C02/C04, K2 leads outright. C04/DL keeps K3 conservatively, because the evidence covers path D only.
- **Ties.** K10 and K7 tie at 0.50 per day.
- **Suite-passing is not shippable.** The curve back-loads K11. Shipping step 5 to production also needs about 2.35–2.45 days of frontend, column and metadata work that §3 moved into K11: K3 0.65, K2 0.25 (frontend only; its provenance is already in the step-1 ship gate), K4 1.3 (including the US cost formatter, without which every US run renders yen), and K5 0.15–0.25. That puts the first shippable 13 cases at about 17.5 days.
- **Order is constraint-driven, not greedy by cases/day.** K5 goes before K1 and K9, and K3 before K1.

## 6. A vs B

**The totals do not separate the two options.** Like-for-like, A's measured reading is 59 likely, against 62 for B as estimated (66 calibrated). The ranges overlap almost fully, and the LOC is similar. On the brief's map A is 62, but that adds D16 product plumbing that B also lacks: B passes the daily cases through its cadence-agnostic core instead. So the map figure is not like-for-like.

Where the two diverge:

- **Shipping.**
  - A flips 6 cases by day 6 and 18 by day 18 on the measured reading (on the map: 3 by day 4.6, 18 by day 21).
  - B can start flipping suite cases through the SimulatorPort after C1–C7 plus its adapter, at about 17 likely days.
  - B ships nothing to production before the core, facade, writer, adapter and test port exist: at least 28 days (C1–C7 15, C12 2.75, C13 5.5, adapter 2.25, test port 2.5). That is a floor. It leaves out the integration A's 17.5 includes: persistence 2.25, engine stamp and allow-list 2.25, frontend up to 4.5, tape v2 1.25 and the C9 fee tables 1.0. A realistic first safe ship is about **35–39 likely days**. Then everything lands in one discontinuity.
- **Shared code.**
  - K2, K5 and K8 edit `execution.py` and `portfolio_manager.py`. These files are also used by the leaderboard (`llm_agent.py:249`), step sessions (`external_run_service.py:797`), A-share and the custom-algo script.
  - A therefore needs opt-in policy branches plus the INT-3 pooling key.
  - K5's weighted-average cost and K8's fractional ledger change what step-session agents see, by design.
  - B leaves `execution.py`, `portfolio.py` and the `PortfolioManager` ledger byte-identical. It still edits the LLM translator in `portfolio_manager.py` (C14), which the leaderboard shares, so it needs the same `llm_agent.py:238` isolation flag as A's K10.
- **Loop surgery.** K1, K3, K5, K7 and K9 all edit the run-loop block (`engine.py` around lines 2185–2440), with hard ordering between them. B replaces that block once, with one clock that owns P1–P4.
- **Recorded intent.** The decision-tape spec says fees, slippage and latency are not retrofitted into the current engine (`docs/superpowers/specs/2026-10-08-backtest-decision-tape-design.md:108-118`). A contradicts that; B honours it.
- **Leaderboard** (impact only, not priced).
  - Under A it becomes a hybrid of the new executor and old same-close fills, unless it is isolated.
  - Under B it stays fully legacy until it moves onto SimulatorPort, which is the same seam an adopted engine would plug into.
  - Either way, `_find_cached_run` needs an engine key before the leaderboard migrates.
- **What B costs.**
  - Two executors, two fill models and two Sharpe definitions must be maintained until the leaderboard and step sessions move.
  - C13, the facade, is the largest single risk at 4–8 days. It covers the hosted runtime, the strict-LLM budget, the H6 counters, the decision tape, progress phases and SIGTERM partial results.
  - A is partly prototyped; B is not prototyped at all.

**Preference if we build: B's design, staged.**

1. **Start with the option-neutral work, about 15.35 likely days.** It ships incrementally on the current engine and is needed under A, B and adoption, so a spike could run in parallel without wasting any of it.

   | Item | Likely days |
   |---|---|
   | D19 calendar | 1.6 |
   | Dated US fee tables | 1.0 |
   | P19 Sharpe, risk-free rate, and server-side Sharpe for D20 | ≈5.5 |
   | Corporate-action sourcing and back-adjusted indicators | 4.0 |
   | Fractional and fee persistence | 2.25 |
   | INT-2 engine stamp | 1.0 |

2. **Then build C2–C7 behind SimulatorPort.** Follow with the facade, the translator (K10 / C14) and the differential report.

**When A wins instead.** A is preferable only if the leaderboard and step sessions will never move, and the goal is early case flips. The cheapest path is 13 of 33 suite-passing in about 15 days on either reading (map: K5 → K3 → K2 → K4; measured: K2 → K5 → K4 → K3), or about 17.5 days to a shippable state (B's first safe ship is about 35–39).

## 7. What would move the estimate most

| Factor | Swing (likely days) | Cheapest way to firm it up | Cost |
|---|---|---|---|
| Throughput: judgement (≈81 LOC/day) vs history (358) | about −42 (59 → ≈17) | Build K5 for real (prototyped, smallest, flips M01/M02), or B's C2 + C6 slice, and time the active days against the estimate | ≈3 d, kept |
| B core and facade have no prototype | C2–C7 9–22; C13 4–8 | A 1–2 day prototype of C2 + C4 + C6, driven by the existing adapter on T02, C03, L01 and M01; this also measures Decimal speed. Then list HourlyBacktester's lifecycle features against the tests that cover them | 2–3 d |
| K11 integration (12, high 20.5): frontend work in a 13.7k-line `app.js` that CI never parses whole and nobody checks in a browser | ±8 | Ship the INT-2 stamp first; every option needs it | 1 d, kept |
| Open owner decisions | ≈8–10 combined | One session to decide them (list below) | <1 d |
| Alpaca corporate-actions facts: which date the filter uses, nullable `payable_date`, Basic-plan access | K7 5.5–12 | One live call; record the wire fixture | 0.25 d |
| K8 was priced blind | ±3 | Re-run the K8 probe | 0.5 d |
| Production SELL-size distribution | 0.5–3.5 | Query decision tapes in `agent_traces` | 0.3 d |
| Neon `trades` size, for the quantity column change (INT-1) | small, but takes a lock in production | One `COUNT(*)` | minutes |

The open owner decisions, with prices where known:
- D16 in or out: 3.2 days. This is also the map-vs-measured gap.
- D12 enforce vs disclaim: 3.5 vs 0.4 days.
- Split-adjusted baselines: 0.75 days.
- K10 prompt change: +0.5–1 days plus billable re-runs.
- Isolation approach, 3a vs 3b.
- Whether stop-limit orders are in scope: +0.5 days.
- What happens to GTC orders at the end of data.
- Whether the final bar spends an LLM call.
- Whether the volume cap or the cash check applies first.
- The order of a split and a dividend on the same day.

**What I could not reconcile**

- **Case count.** The conformance spec's tally says 28 cases (spec lines 22 and 712). The brief's own counts (1 PASS, 22 FAIL, 7 N/E) and the 30 D predictions in `cases.py` give 30. I used 30 plus 3 DL entries, 33 in total.
- **Map vs evidence.** The map curve is the primary fix order. The headline total uses the measured reading, because only there are A and B like-for-like. Both totals are shown.
- **Review arithmetic.**
  - K4's items sum to +1.1, but its total moved +0.7. I kept 4.7.
  - K6's items sum to +1.1, but its likely moved +0.5, with the rest in the bounds.
  - K9's items sum to +1.6 against a +1.3 per-defect move, and its cluster totals sit below its per-defect sums.
  - K8's components exceed its total, so the qty_step overlap looks netted. This is not confirmed (±0.5). K8's low inherits the review's unexplained compression: its components minus persistence come to about 5.6–6.1 against the 3.5 used here, so A's low is about 2 days understated (≈41, not 39). K2's ExecutionPolicy scaffold also threads `qty_step`, so 0.25–0.5 days of K8 plumbing may be counted twice.
  - Small uncredited overlaps of about 0.5–1 day total: K6's Sharpe tooltip vs INT-6, K5's metadata asserts vs INT-2, and decision-tape edits priced in K3, K8, K9 and K10 as well as INT-7.
  - About 0.4 days of K2's policy scaffold is both share-weighted as engine-internal and booked as A-only isolation in §4.
- **Judgement vs calibration.** The two differ by about 3.5×, and the test-to-production ratio (1.3 vs 2.08) is unexplained.
- **My own inferences and guesses.** B's review uplift is my inference, not a review. The engine shares are guesses, applied to de-duplicated days and about 0.2 days high.
- **Research §8** gives no duration beyond the spike.
- **Prototype results.** The K1, K4 and K5 results rest on the probe agents' scratch runs. I ran nothing. I only read `known_failures.json`, `cases.py`, `baseline_worker.py` and research §8 to settle disputes.

Tree: a read-only `origin/main` worktree @ 2dc8203a (read-only).
