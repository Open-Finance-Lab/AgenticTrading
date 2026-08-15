# Landing narrative — copy + build checklist

> **Status (2026-08):** this is the original design checklist for the
> Talk → Test → Race narrative, kept for the rationale and the kill-list. The
> shipped copy has since evolved past it (the Talk act now leads with
> "describe your idea" — the Discord mock survives as its secondary visual —
> and the 2026-08-04 audience-language pass rewrote the register throughout). For what the live
> page actually says, the source of truth is `dashboard/landing/src` plus the
> guards in `dashboard/backend/tests/test_landing_copy_register.py` — update
> those, not this file, when copy changes.

Story arc: **Talk (Discord) → Test (backtest) → Race (contest)**  
Tone: short lines, one job per section. No feature dumps.

---

## Copy deck

### Nav
| Slot | Copy |
|------|------|
| Links | Talk · Test · Race |
| CTA | Get Started → Discord (same as Hero primary) |

### Hero
**Headline and CTAs frozen.** Keep `Talk to Agents` / `Test Trading Ideas` and `Start Free`.
Scroll target `#landing-stats` is preserved as a hidden anchor inside WhyCare.

**The visual is no longer frozen (2026-08-15).** The right column was the agent
conversation demo; it is now the board — `BoardPreview.tsx`, a compact equity
chart plus the top five standings, above the fold at 1440×900. The conversation
demo moved down to the Talk act (`ChatSimulation.tsx`), which is the beat it
illustrates. The hero also gained one supporting line under the headline: the
one-per-surface gloss on "agent", because the headline uses the word before
anything else on the page defines it and the board is now the only other thing
above the fold.

Why: the leaderboard was the page's only piece of evidence and sat roughly four
screens down, so the last thing a visitor saw was the one thing that would have
convinced them.

### 01 — Talk
| Slot | Copy |
|------|------|
| Label | 01 — Talk |
| H2 | Talk to agents on Discord |
| Body (1 line) | Describe your trading idea. The agent runs it. |
| Steps | 1. Join the server · 2. Talk to the agent · 3. Get your backtest result |
| Primary CTA | Join Discord |

**Right visual:** Discord channel mock (`DiscordMock`) — server rail + `#agent-trading-lab` + APP agent thread (not chat bubbles).

**Mock dialogue (keep short)**
- You: `I want to follow Warren Buffett. If Berkshire makes a move, copy the move and tell me how it goes.`
- Agent: clarify → rules → backtest embed (`+14.2%` · Sharpe · See full result ↓)

### 02 — Test
| Slot | Copy |
|------|------|
| Label | 02 — Test |
| H2 | Test your trading idea |
| Body (1 line) | Full agent run with fixed experiment settings + baselines. |
| Figure | Equity: Alpha vs DJIA / S&P 500 / Buy & Hold |
| Experiment settings | Initial capital · Time period (1 month) · Universe (DJIA 30) · Baselines · AI model · Est. AI cost (dollar figure only, no raw token counts) — **no prompt field** |
| Metrics | Return · Sharpe · Max DD · vs Buy & Hold · trades / avg hold |
| Log | Decision log with step, size, rationale |
| Primary CTA | Race this agent ↓ |

### Race (unnumbered since 2026-08-15)
| Slot | Copy |
|------|------|
| Label | **none** — the `03 — Race` mono-label was dropped when the board moved into the hero. Talk (01) and Test (02) keep theirs; the board is the first thing on the page, so numbering it third described the wrong page. |
| H2 | What the AI models actually returned |
| Body (1 line) | Seven leading AI models traded the same days with simulated money, ranked against buy-and-hold and the index. Only one finished ahead of both. |
| Rules (3 bullets max, each with an icon) | Competition: one fixed window of market history — the same days and the same starting capital for every contender. (`CalendarClock`) · Live Trading Leaderboard: designed to move forward one trading session at a time, in two-week seasons. (`TrendingUp`) · Published only if the AI model itself made at least 95% of the decisions. (`ShieldCheck`) |
| Preview note | The Live Trading Leaderboard is in preview for Season 0. It has not moved forward a session yet, and nothing on it is a record. Season 1 is the first that counts. |
| Standings card | Competition Standings (`Illustrative example` chip) |
| Primary CTA | Start Free (`PRIMARY_LANDING_CTA`) |

**Two counts in the body are facts, not flourishes.** "Seven" is the LLM entry
count in `dashboard/config/leaderboard.json`; "only one finished ahead of both"
is the real result (DeepSeek V4 Pro). Re-check both against that file when the
roster changes — the sentence is the page's main credibility move precisely
because it volunteers an unflattering number.

**Why the numbered label went, not the bullets.** The rules are the section's
whole job: they are what makes a return figure mean anything. What stopped
making sense was calling the board "act three" on a page that opens with it.

**No chart here any more.** The equity chart lives in the hero
(`BoardPreview.tsx`); Race keeps the full standings table and the rules. The
sample rows are shared — `Race.tsx` imports `SAMPLE_STANDINGS` — so the two
cards cannot drift.

**Why the H2 is not "Race your agent" (2026-08-15).** Board entries come from the
curated `dashboard/config/leaderboard.json` roster, so **no user agent is on any
board** — "race your agent" / "watch your agent climb" described an entry flow that
does not exist. It is also two boards now, not one, and only one of them is even
notionally live. The act still lands: the board is the bar you are testing against
in 02. Revisit when the season entry flow ships (PR #328 frontier).

### Footer
| Slot | Copy |
|------|------|
| Line | Talk → Test → Race |
| CTAs | Join Discord · Open Leaderboard |

### App-side twin (`app.html` Home screen 0)

Not part of the marketing page, but written to the same register and changed in
the same pass. Signed-in visitors never see `/` at all — `index.html` redirects
them to `/app` — and Home's pager screen 0 was a second copy of this hero,
"Talk to Agents / Test Trading Ideas" over a **Get Started** button, shown to
people who had already got started. It now carries the real Competition
Leaderboard (`#homeModuleRanking`, moved off the screen-1 dashboard grid), and
the CTA reads by sign-in state: *Create a free account* signed out, *Test a
trading idea* signed in.

### Kill / avoid
- Fake stats (“Agents Online”, etc.)
- “From Idea to Execution” / Talk·Test·**Trade**
- “Live Network”, “tick-level”, “Season 4” fake names
- Long paragraphs under any H2

---

## Component checklist

### Phase A — IA + copy (structure)
| Action | File |
|--------|------|
| Reorder: Hero → Talk → Test → Race → Footer | `landing-page.tsx` |
| Nav anchors → Talk / Test / Race | `Navbar.tsx` |
| Hero: subline + chips; CTAs; scroll → `#talk` | `Hero.tsx` |
| Promote Discord section → `#talk` | rename/reuse `DiscordPrompt.tsx` → `Talk.tsx` |
| Merge backtest + short decision strip → `#test` | `Backtesting.tsx` → `Test.tsx` |
| Contest section → `#race`; kill fake rows | `Community.tsx` → `Race.tsx` |
| Footer: 3-beat line + CTAs | `FooterCTA.tsx` |
| **Delete** | `StatsBar.tsx`, `HowItWorks.tsx` |
| **Delete or fold** | `ActivityFeed.tsx` → 3–5 rows inside Test |
| **Demote** | `PaperTradingDeploy.tsx` → one footnote under Test |

### Phase B — proof (data)
| Action | Source |
|--------|--------|
| Race board | `GET /api/v1/leaderboard?period=contest` |
| Contest dates | response + `config/leaderboard.json` |
| Test equity/metrics | `defaults.json` run IDs / seed DB export → fixture |
| Test decision log | same run’s decisions (API or static fixture) |
| API fail | skeleton + “Unavailable” — never fake ranks |

### Phase C — Talk polish
| Action | Source |
|--------|--------|
| Hero visual → Discord-shaped | real agent transcript or labeled demo |
| Talk mock = same script as Hero (or shorter) | Discord export, scrubbed |
| Commands link | Discord agent docs / README |

### Phase D — tracking
| Event | Where |
|-------|-------|
| `hero_cta_discord_click` / `hero_cta_leaderboard_click` | Hero |
| `hero_chip_click` `{beat}` | Hero chips |
| `section_talk_view` / `section_test_view` / `section_race_view` | IO ≥50% |
| `talk_cta_discord_click` | Talk |
| `test_open_run_click` | Test |
| `race_leaderboard_loaded` / `race_leaderboard_error` | Race |
| `race_row_click` `{id}` | Race |
| `race_cta_lab_click` / `race_cta_discord_click` | Race |

### Done when
- [ ] Page order matches Talk → Test → Race
- [ ] No fake stats / fake leaderboard
- [ ] Each section ≤ 1 H2 + 1 line body + 1 visual + 1–2 CTAs
- [ ] Race loads real API (or honest empty state)
- [ ] Events wired on all CTAs + section views
