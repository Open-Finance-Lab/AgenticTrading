1. **Drop #346 from the "duplicates the rewrite" risk (B1 Risks, G16).** #346 is the A-share corporate-action issue, and A-share was dropped from the rewrite's scope on 10-07 (RA line 3, CS line 14). Only #461, US costs, duplicates conformance group F. The US corporate-action gap is D13 and has no filed issue. Fix: say that #461 duplicates group F, that #346 stays independent and out of scope, and that G16 can close #461 against the suite but not #346.

2. **Conflict 4 and the Q6 "Against" bullet overstate the 10-08 rule.** The recorded decision (EP §6, citing the decision-tape spec at lines 108-118) covers only fees and slippage not being retrofitted. The Q6 fixes are early close, trade memory in `snapshot_state`, the D12 disclaimer and Sharpe. None of them is a fee or slippage retrofit. LB also labels this whole Stage 0 set (tape, early close, trade memory, D12 disclaimer, P19 Sharpe, engine key, frozen bar archive) "needed under every option". The brief's phrase "code the rewrite deletes" contradicts that. Fix: downgrade the conflict to "possible tension, not a conflict". Add LB's option-independence claim to Q6, and drop "against the 10-08 no-retrofit rule".

3. **The nof1 status and methodology claims are stated more firmly than the gatherer supports (A, C2, Q1).**
   - "No public run since Dec 2025" rests on TradeRank, a competitor. It was checked on Aug 6, and the Aug 27 recheck returned 403. Sept/Oct is only an absence of search hits, and the gatherer says to confirm in a real browser.
   - "Methodology unpublished" (C2) is really "none retrieved". nof1.ai, alphaarena.ai, /blog and /leaderboard were 404, 429 or 402.
   - A dropped fact bears directly on Q1: Bitget says Season 2 "will include human traders and their self-developed models" [low]. That would overlap ATL's instruction-vs-instruction niche.
   - Fix: write "none found as of Aug 6, unconfirmed since", "methodology not retrieved", and add the Season 2 human-traders line to Q1 "Against".

4. **C1 overstates the peer bar in three places.**
   - "Fill no earlier than next data point: table stakes" lists Nautilus. The simulator-bar gatherer says Nautilus on_bar orders settle against bar N's close. Backtrader's `coc` and TradingView's `process_orders_on_close` are documented same-close options. LEAN's wait-for-next-bar rule is stated for limit orders only. Fix: reframe the bar as "a documented, overridable fill rule". The ATL gap is that same-close is the undocumented, non-overridable default.
   - "Credibility comes from documented assumptions, not non-zero defaults [high]" is the gatherer's own inference, and Zipline's default is 5 bps slippage plus $0.001/share. Downgrade the tag to inference and qualify the claim.
   - "LEAN, Nautilus, Backtrader and Alpaca all publish a 'what we don't model' list [high]" is directly verified only for Alpaca and Nautilus. Downgrade the tag.

5. **Q1–Q7 labels collide.** The brief uses Q1–Q5 for the owner's CS decisions (B2, §D) and Q1–Q7 for round-1 questions (§F, Conflicts 2, 4 and 6). Conflict 6 says "the Q5 proposal is ~$100/mo", and no gatherer contains that figure. The only $100 on record is the CommonStack total cap (July), which is not monthly. Fix: rename the owner decisions "CS-Q1..Q5" and the round-1 questions "R1..R7". Source the ~$100/mo figure or remove it. Q5 "$100 … double the prior figure" has no support as written.

6. **"No owner decision ratifies B" may contradict the owner's own memory index (A, Conflict 8, G9).** The memory entry reads "rewrite authorized 2026-10-07 … verdict = build own core step-first (B)". The engine gatherer calls B an analysis recommendation. Fix: say "recorded in memory as the verdict; no explicit ratification found; confirm", and not "no decision".

7. **"No LLM median beats the passive baselines" (A) needs a caveat.** #604's body says the baselines and the GPT-5.5, Gemini and Sonnet rows are July vintage and nothing re-runs them. Memory says medians exist only for Qwen, DeepSeek and Haiku. So the claim covers three models against a stale baseline set. Add the caveat to A and G11.

8. **"Three surfaces on one legacy engine" (A) is imprecise.** My Agents runs on `HourlyBacktester`. The Contest and Live boards run `llm_agent.py` with `PortfolioManager`. The shared parts are `execution.py` and `portfolio_manager.py`. Most (b) and (c) cells in the engine gatherer's defect table are "likely", not verified. Fix: say "shared executor and portfolio manager" and note that board exposure to D1, D3 and D5–D9 is inferred.

9. **Small accuracy fixes.**
   - "D1–D20 are all open" includes D4 (non-intraday only) and D17 (out of scope). D20 is not in the case file. Say "no defect is marked fixed".
   - #594 and #596 are already merged. They are not future conflicts. Only #573 and #600 remain open, and #543 is missing from the list.
   - "Merges #625–#631 are unrecorded": the central-db gatherer says #630 is recorded.
   - Q7 and G16 propose a draft or `blocked` gate for #626. It is an external author's PR, so use the `blocked` label plus a comment. The owner cannot convert it to draft.
   - Q5 lists "real-time SIP $99/mo" as a cost of hourly cadence. LB says hourly appends run on settled bars and cost only a Render worker. Mark the $99 as conditional on real-time data.

10. **Dropped gatherer facts that bear on the goals.**
    - #128: the vnpy simulation goes flat after about 78 bars, against a default window of about 90 days. This is a simulator-validity issue.
    - #123: a run does not record its news source, which a reproducibility claim needs.
    - The tester screenshot shows the Competition window ending 05-15 while the last plotted point is 09-02. A skeptical reader will notice. Add it to E.
    - SEC Section 31 went from $0.00 to $20.60 per million on 2026-04-04. This matters for the "dated SEC/TAF" decision in Q3, and for deciding which contest dates fall before or after the rate change.
    - Season 1's Nemotron temperature-0 pin is unimplemented (participatory spec). This bears on roster decision #8.
    - EP's break-even caveat: adoption pays back only if the spike is mostly calendar wait. This bears on G9.

11. **Checked and correct, no change needed.**
    - The cited line numbers all match the engine-probe tree: `profiles.py:144`, `decision_tape.py:48`, `backtest-comparison.js:65`, `live.py:56/74-77/283/1517`, the `app.html` lines, `validator.py:86-87`, `portfolio_manager.py:194/445`, `leaderboard.json:10-11`, and the `CLAUDE.md` lines 71/217/227/232/243/259.
    - 12 `*_postgres.py` modules do call `init_schema_unless_worker`.
    - The Phase 0 floor is 0.20pp for DeepSeek V4 Pro at temperature 0 on the same 04-15 → 05-15 window, so the comparison to the 12.6pp spread is valid. The probe itself calls it one pair with no error bar.
    - 07c0ab0d dropped GPT-5.5 because it was "most of the nightly token spend".
    - HEAD 2dc8203a is "docs: recover engine rewrite phase 1 records".
    - LB supports the 5.8× perps figure, the 20–28 day shorts cost, the 2.5-day tape, the 3.9-day fix sum (1+1+0.4+1.5), the 28–44 calibrated range (B 28–32 to Nautilus 41–44), and the 2.5–4.5 h late cron.
    - The 63 points (9 sessions × 7 nodes), 2026-11-27 as the half day, and the roughly 25-day S2 increment all check out.
