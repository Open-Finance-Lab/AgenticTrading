/* Portfolio Management (a My Agents subtab): Weekly Portfolio Rebalancing, case 2, playing one recorded run.
   Ported on 2026-11-05 from the standalone build weekly-v2 Version 6 (src/app.js, src/style.css, src/markup.html);
   recorder run 20261104T050945Z-3a2f26, whose recording is js/portfolio-management-data.js. This file is now the
   source; the published standalone page is the reference rendering. app.js calls show(panel) and hide() from
   showPlaygroundPanel and navigateToPage; the page lives in a shadow root on #playgroundPortfolioManagementPanel,
   so its styles and ids stay inside the panel and ATL's stay out.
   Every number, check result and model reply comes from the recording: DeepSeek V4 Pro (proposer) and Claude
   Sonnet 4.6 (critic), Gurobi 13.0.3, our code checks. Plain-words lines are ours. Vanilla JS, no eval, no inline
   handlers. */
(function () {
  'use strict';
  var CSS = String.raw`/* Weekly Portfolio Rebalancing inside the My Agents page: a three-column page, Portfolio 30% / Your rules 38% / Chat 32%,
   the loop strip and the Explanation under the first two. Single dark theme on purpose: this page lives inside ATL's dark product.
   Everything here applies inside the panel's shadow root only. */
:host {
  color-scheme: dark;
  --bg: #0a0e27;
  --surface: #0f1328;
  --card: #131a35;
  --hover: #1a2047;
  --code-bg: #0c1130;
  --text: #e5e7eb;
  --text-body: #d1d5db;
  --text-2: #9ca3af;
  --muted: #6b7280;
  --border: #1f2937;
  --border-2: #2a3557;

  --success: #00c851;
  --success-bg: rgba(0, 200, 81, 0.12);
  --success-line: rgba(0, 200, 81, 0.5);
  --danger: #ff4141;
  --danger-bg: rgba(255, 65, 65, 0.12);
  --danger-line: rgba(255, 65, 65, 0.55);
  --amber: #ffa500;
  --amber-bg: rgba(255, 165, 0, 0.11);
  --amber-line: rgba(255, 165, 0, 0.55);
  --info: #00bfff;
  --info-bg: rgba(0, 191, 255, 0.1);
  --info-line: rgba(0, 191, 255, 0.5);
  --accent: #22d3ee;
  --accent-bg: rgba(34, 211, 238, 0.1);
  --accent-line: rgba(34, 211, 238, 0.55);
  --on-accent: #041018;

  /* provenance labels */
  --l-model: #b69cff;
  --l-model-bg: rgba(182, 156, 255, 0.12);
  --l-model-line: rgba(182, 156, 255, 0.5);
  --l-code: #22d3ee;
  --l-code-bg: rgba(34, 211, 238, 0.1);
  --l-code-line: rgba(34, 211, 238, 0.5);
  --l-team: #f472b6;
  --l-team-bg: rgba(244, 114, 182, 0.11);
  --l-team-line: rgba(244, 114, 182, 0.6);
  --l-sim: #a7b4c8;
  --l-sim-bg: rgba(167, 180, 200, 0.1);
  --l-sim-line: rgba(167, 180, 200, 0.6);

  /* pie wedges: validated on #131a35 (dataviz validator, adjacent pairs); others is a deliberate low-chroma slot */
  --w1: #3987e5; --w2: #c98500; --w3: #9085e9; --w4: #2aa3b5;
  --w5: #d55181; --w6: #5c7cfa; --w7: #d95926; --w8: #8a6ee0;
  --w-others: #2e3a5e;

  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
  --font-mono: 'JetBrains Mono', 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', monospace;
  --radius: 8px;
  --gutter-l: max(16px, env(safe-area-inset-left, 0px));
  --gutter-r: max(16px, env(safe-area-inset-right, 0px));
}

*, *::before, *::after { box-sizing: border-box; }
:host { all: initial; display: block; color: var(--text); font-family: var(--font-sans); font-size: 16px; line-height: 1.45; -webkit-font-smoothing: antialiased; color-scheme: dark; }
button, input { font-family: inherit; font-size: inherit; }
button { cursor: pointer; }
[hidden] { display: none !important; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.mono { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.num { font-variant-numeric: tabular-nums; }
.nowrap { white-space: nowrap; }
.sr-only { position: absolute !important; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; } }

/* ---------- provenance chips ---------- */
.chip { display: inline-flex; align-items: center; gap: 5px; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; line-height: 1.5; border: 1px solid var(--border-2); color: var(--text-2); white-space: nowrap; vertical-align: middle; }
.chip--model { color: var(--l-model); background: var(--l-model-bg); border-color: var(--l-model-line); }
.chip--code { color: var(--l-code); background: var(--l-code-bg); border-color: var(--l-code-line); }
.chip--team { color: var(--l-team); background: var(--l-team-bg); border-style: dashed; border-color: var(--l-team-line); }
.chip--sim { color: var(--l-sim); background: var(--l-sim-bg); border-style: dotted; border-color: var(--l-sim-line); }
.chip--amber { color: var(--amber); background: var(--amber-bg); border-color: var(--amber-line); }
.chip--green { color: var(--success); background: var(--success-bg); border-color: var(--success-line); }
.chip--red { color: var(--danger); background: var(--danger-bg); border-color: var(--danger-line); }
.chip--blue { color: var(--info); background: var(--info-bg); border-color: var(--info-line); }
.chip--cyan { color: var(--accent); background: var(--accent-bg); border-color: var(--accent-line); }
.chip--mono { font-family: var(--font-mono); font-weight: 500; }
.chip--sm { font-size: 11px; padding: 0 6px; }

/* ---------- page ---------- */
.page { display: flex; flex-direction: column; min-height: 0; }
.ph { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; margin-bottom: 12px; }
.ph-title { margin: 0; font-size: 24px; font-weight: 700; letter-spacing: -0.01em; line-height: 1.2; }
.ph-sub { margin: 4px 0 0; color: var(--text-2); font-size: 14px; }
.ph-right { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
.badge-rec { display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border-radius: 6px; border: 1.5px solid var(--success); color: var(--success); font-family: var(--font-mono); font-size: 13px; font-weight: 600; letter-spacing: 0.02em; white-space: nowrap; }
.badge-rec::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--success); }
.btn { display: inline-flex; align-items: center; gap: 6px; padding: 7px 12px; border-radius: 6px; border: 1px solid var(--border-2); background: var(--card); color: var(--text); font-size: 14px; font-weight: 600; white-space: nowrap; }
.btn:hover { background: var(--hover); }
.btn--ghost { background: transparent; color: var(--text-2); }
.btn--primary { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
.btn--primary:hover { background: #67e8f9; }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }
.play { display: flex; align-items: center; gap: 6px; padding: 4px; border: 1px solid var(--border); border-radius: 8px; background: var(--card); }
.play .btn { padding: 6px 10px; border: 0; background: transparent; }
.play .btn--primary { background: var(--accent); min-width: 84px; justify-content: center; }
.play-count { font-family: var(--font-mono); font-size: 13px; color: var(--text-2); padding: 0 8px; white-space: nowrap; }
.play--loop { padding: 3px; }
.play--loop .btn { padding: 4px 9px; font-size: 13px; }
.play--loop .btn--primary { min-width: 78px; }
.play--loop .play-count { font-size: 12px; padding: 0 6px; }

.grid { display: grid; grid-template-columns: 30fr 38fr 32fr; grid-template-rows: auto auto auto; gap: 12px; min-height: 0; }
.box { background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 12px 16px 14px; min-width: 0; min-height: 0; display: flex; flex-direction: column; position: relative; }
.box-h { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
.box-h h2 { margin: 0; font-size: 16px; font-weight: 700; }
.box-h .sub { color: var(--text-2); font-size: 14px; }
.box-h .spacer { flex: 1; }
.box-portfolio { grid-column: 1; grid-row: 1; }
.box-rules { grid-column: 2; grid-row: 1; }
.box-chat { grid-column: 3; grid-row: 1 / 4; background: var(--surface); }
.box-loop { grid-column: 1 / 3; grid-row: 2; }
.box-explain { grid-column: 1 / 3; grid-row: 3; background: transparent; border-color: transparent; padding: 4px 6px 0; }

/* ---------- portfolio ---------- */
.state-chip { font-family: var(--font-mono); font-size: 12px; font-weight: 700; letter-spacing: 0.06em; padding: 1px 8px; border-radius: 4px; border: 1px solid var(--border-2); color: var(--text-2); }
.state-chip[data-state="planned"] { color: var(--info); border-color: var(--info-line); background: var(--info-bg); }
.state-chip[data-state="traded"] { color: var(--success); border-color: var(--success-line); background: var(--success-bg); }
.pie-wrap { flex: 1; min-height: 0; display: grid; place-items: center; }
.pie { width: 100%; height: 100%; max-height: 440px; display: block; }
.pie .w { transition: opacity 0.4s ease; }
.pie .lbl { font-family: var(--font-mono); font-size: 14px; font-weight: 600; fill: var(--text); }
.pie .lbl .pct { fill: var(--text-2); font-weight: 500; }
.pie .lbl .over { fill: var(--danger); font-weight: 700; }
.pie .lead { stroke: var(--lead); stroke-width: 2px; vector-effect: non-scaling-stroke; fill: none; }
.pie .lead.hot { stroke: var(--danger-line); }
.pie .c-state { font-size: 20px; font-weight: 700; fill: var(--text); }
.pie .c-date { font-family: var(--font-mono); font-size: 12px; fill: var(--text-2); }
.pie .hatch-outline { fill: none; stroke: var(--danger); stroke-width: 1.5; }
.tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 8px; }
.tile { padding: 8px 10px; border-radius: 6px; background: var(--surface); border: 1px solid var(--border); min-width: 0; }
.tile .t-k { font-size: 12px; color: var(--text-2); }
.tile .t-v { font-size: 16px; font-weight: 700; line-height: 1.3; }
.tile .t-l { font-size: 12px; color: var(--muted); }
.tile[data-tone="red"] { border-color: var(--danger-line); background: var(--danger-bg); }
.tile[data-tone="red"] .t-v { color: var(--danger); }
.tile[data-tone="amber"] { border-color: var(--amber-line); }
.tile[data-tone="amber"] .t-l { color: var(--amber); }

/* ---------- rules ---------- */
.ver-chip { font-family: var(--font-mono); font-size: 12px; padding: 1px 8px; border-radius: 4px; border: 1px solid var(--border-2); color: var(--text); background: var(--surface); white-space: nowrap; }
.ver-chip[data-tone="amber"] { color: var(--amber); border-color: var(--amber-line); background: var(--amber-bg); }
.tag { font-size: 11px; font-weight: 600; padding: 0 6px; border-radius: 4px; border: 1px solid var(--border-2); color: var(--text-2); white-space: nowrap; }
.tag--amber { color: var(--amber); border-color: var(--amber-line); background: var(--amber-bg); }
.rules { width: 100%; border-collapse: collapse; font-size: 14px; }
.rules th { text-align: left; font-size: 11px; font-weight: 600; letter-spacing: 0.06em; color: var(--muted); padding: 2px 8px 6px; border-bottom: 1px solid var(--border); white-space: nowrap; }
.rules td { padding: 6px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
.rules tr:last-child td { border-bottom: 0; }
.rules .r-id { font-family: var(--font-mono); font-size: 12px; color: var(--text-2); padding-right: 0; width: 34px; }
.rules .r-w { color: var(--text); width: 44%; }
.rules .r-w q { quotes: "\201C" "\201D"; }
.rules .r-e { color: var(--text-body); }
.rules .r-s { text-align: right; width: 1%; white-space: nowrap; }
.rules .r-foot-row td { padding: 0 8px 7px 42px; font-size: 12px; color: var(--muted); line-height: 1.35; }
.rules .r-note { display: block; font-size: 12px; color: var(--muted); margin-top: 2px; }
.rules .r-foot { display: block; font-size: 12px; color: var(--muted); margin-top: 3px; line-height: 1.35; }
.rules tr[data-conf="1"] td { background: var(--amber-bg); }
.rules tr[data-conf="1"] td:first-child { box-shadow: inset 3px 0 0 var(--amber); }
.st { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 600; }
.st--ok { color: var(--success); }
.st--bad { color: var(--danger); }
.st--ask { color: var(--amber); }
.conf-line { margin: 8px 0 0; padding: 8px 10px; border-radius: 6px; border: 1px solid var(--amber-line); background: var(--amber-bg); font-size: 13px; color: var(--text); line-height: 1.4; }
.conf-line b { color: var(--amber); font-weight: 700; }
.rules-foot { margin: 8px 0 0; font-size: 12px; color: var(--muted); line-height: 1.4; }

/* ---------- chat ---------- */
.box-chat { padding: 0; }
.chat-inner { position: absolute; inset: 0; display: flex; flex-direction: column; }
.chat-h { padding: 12px 16px 8px; border-bottom: 1px solid var(--border); }
.thread { flex: 1; min-height: 0; overflow-y: auto; padding: 12px 16px; display: flex; flex-direction: column; gap: 12px; }
.msg { max-width: 92%; animation: slidein 0.35s ease both; }
.msg.old { opacity: 0.6; }
@keyframes slidein { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
.msg--client { align-self: flex-end; text-align: right; }
.msg--client .bub { display: inline-block; text-align: left; padding: 9px 13px; border-radius: 12px 12px 3px 12px; background: #1e2a4a; color: var(--text); font-size: 15px; line-height: 1.4; }
.msg .meta { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text-2); margin-bottom: 4px; }
.msg--client .meta { justify-content: flex-end; }
.msg .meta .who { font-weight: 700; color: var(--text); }
.msg--proposer .meta .who { color: var(--l-model); }
.msg .meta .when { font-family: var(--font-mono); color: var(--muted); }
.msg--proposer { align-self: flex-start; }
.msg--proposer .bub { padding: 10px 14px; border-radius: 12px 12px 12px 3px; background: var(--card); border: 1px solid var(--border-2); color: var(--text); font-size: 15px; line-height: 1.45; }
.msg--sys { align-self: center; max-width: 100%; text-align: center; font-family: var(--font-mono); font-size: 12.5px; color: var(--text-2); display: flex; align-items: center; gap: 8px; justify-content: center; padding: 2px 0; }
.msg--sys svg { width: 14px; height: 14px; flex: 0 0 auto; stroke: currentColor; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
.msg--sys[data-tone="amber"] { color: var(--amber); }
.msg--sys[data-tone="green"] { color: var(--success); }
.msg--sys[data-tone="red"] { color: var(--danger); }
.msg--sys[data-tone="blue"] { color: var(--info); }
.redact { text-decoration: underline; text-decoration-style: dotted; text-underline-offset: 3px; text-decoration-color: var(--l-code); }
.lock { display: inline-flex; align-items: center; gap: 3px; margin-left: 3px; vertical-align: 1px; font-size: 10px; color: var(--l-code); font-family: var(--font-mono); }
.lock svg { width: 10px; height: 10px; stroke: currentColor; fill: none; stroke-width: 2; }
.seen { font-size: 12px; color: var(--muted); margin-top: 5px; line-height: 1.35; }
.opts { display: flex; flex-direction: column; gap: 6px; margin-top: 10px; }
.opt { display: flex; gap: 10px; align-items: flex-start; text-align: left; padding: 8px 12px; border-radius: 8px; border: 1px solid var(--border-2); background: var(--surface); color: var(--text); font-size: 14px; line-height: 1.35; cursor: default; }
.opt .k { font-family: var(--font-mono); font-weight: 700; color: var(--text-2); }
.opt.picked { border-color: var(--amber); background: var(--amber-bg); }
.opt.picked .k { color: var(--amber); }
.opt.dim { opacity: 0.45; }
.plain { margin-top: 6px; font-size: 13px; color: var(--text-2); line-height: 1.4; }
.chat-foot { padding: 10px 16px 12px; border-top: 1px solid var(--border); }
.compose { display: flex; gap: 8px; }
.compose input { flex: 1; min-width: 0; padding: 9px 12px; border-radius: 8px; border: 1px solid var(--border-2); background: var(--bg); color: var(--text); font-size: 14px; }
.compose input:disabled { opacity: 0.7; }
.compose .send { width: 40px; border-radius: 8px; border: 0; background: var(--accent); color: var(--on-accent); display: grid; place-items: center; }
.compose .send:disabled { opacity: 0.4; cursor: not-allowed; }
.compose .send svg { width: 18px; height: 18px; stroke: currentColor; fill: none; stroke-width: 2.2; }
.chat-note { margin: 6px 0 0; font-size: 12px; color: var(--muted); }
.typing { display: inline-block; }

/* ---------- loop strip ---------- */
.box-loop { padding-bottom: 12px; }
.loop-h { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.loop-h h2 { margin: 0; font-size: 16px; font-weight: 700; }
.round { font-family: var(--font-mono); font-size: 13px; color: var(--text-2); }
.loop-wrap { position: relative; }
.lanes { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; overflow: visible; }
.lanes path { fill: none; stroke-width: 2; }
.lanes .ret { stroke: var(--danger); }
.lanes .arc { stroke: var(--amber); stroke-dasharray: 5 4; }
.lanes text { font-family: var(--font-sans); font-size: 12.5px; font-weight: 600; }
.lanes .ret-t { fill: var(--danger); }
.lanes .arc-t { fill: var(--amber); }
.lanes .arrow-ret { fill: var(--danger); }
.lanes .arrow-arc { fill: var(--amber); }
.loop { display: grid; grid-template-columns: 1fr 26px 1.25fr 26px 1.15fr 26px 1.15fr 26px 1fr; align-items: stretch; }
.grp { grid-row: 1; font-size: 11px; font-weight: 700; letter-spacing: 0.1em; color: var(--muted); text-align: center; padding: 20px 0 4px; }
.lbox { grid-row: 2; border: 1px solid var(--border-2); border-radius: var(--radius); background: var(--surface); padding: 8px 11px 8px; min-width: 0; display: flex; flex-direction: column; gap: 4px; transition: border-color 0.25s ease, box-shadow 0.25s ease; }
.lbox[data-tab] { cursor: pointer; }
.lbox[data-tab]:hover { background: var(--hover); }
.lbox.is-active { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent), 0 0 18px rgba(34, 211, 238, 0.18); }
.lbox[data-tone="green"] { border-color: var(--success-line); }
.lbox[data-tone="red"] { border-color: var(--danger-line); }
.lbox[data-tone="amber"] { border-color: var(--amber-line); }
.lbox[data-tone="blue"] { border-color: var(--info-line); }
.lbox-t { font-size: 15px; font-weight: 700; display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.lbox-s { font-size: 12px; color: var(--text-2); margin-bottom: 2px; }
.lrow { font-size: 12.5px; line-height: 1.35; color: var(--text-body); display: flex; gap: 7px; align-items: flex-start; }
.lrow .k { font-family: var(--font-mono); font-size: 10.5px; font-weight: 700; letter-spacing: 0.04em; padding: 1px 5px; border-radius: 3px; border: 1px solid var(--border-2); color: var(--text-2); flex: 0 0 auto; margin-top: 2px; }
.lrow .k--code { color: var(--l-code); border-color: var(--l-code-line); }
.lrow .k--model { color: var(--l-model); border-color: var(--l-model-line); }
.lrow[data-tone="red"] { color: var(--danger); }
.lrow[data-tone="green"] { color: var(--success); }
.lrow[data-tone="amber"] { color: var(--amber); }
.lrow[data-tone="blue"] { color: var(--info); }
.lrow[data-tone="cyan"] { color: var(--accent); }
.lrow.idle { color: var(--muted); }
.lopts { display: flex; flex-direction: column; gap: 2px; margin: 2px 0 0 0; font-family: var(--font-mono); font-size: 12px; }
.lopts div { color: var(--text-2); }
.lopts div[data-ok="0"] { color: var(--danger); }
.lopts div[data-ok="1"] { color: var(--text-2); }
.lbox-foot { margin-top: auto; padding-top: 6px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.arrow { grid-row: 2; display: grid; place-items: center; color: var(--border-2); }
.arrow svg { width: 22px; height: 16px; stroke: currentColor; fill: none; stroke-width: 2; }
.arrow.hot { color: var(--accent); }
.arrow.green { color: var(--success); }
.under { margin: 6px 0 0; font-size: 13px; color: var(--amber); line-height: 1.4; display: flex; gap: 8px; align-items: baseline; }
.under .chip { flex: 0 0 auto; }
.verdict { font-family: var(--font-mono); font-size: 11px; font-weight: 700; letter-spacing: 0.08em; padding: 1px 7px; border-radius: 4px; }
.verdict--obj { color: var(--danger); background: var(--danger-bg); border: 1px solid var(--danger-line); }
.verdict--pass { color: var(--success); background: var(--success-bg); border: 1px solid var(--success-line); }
.nothing { font-family: var(--font-mono); font-size: 11px; font-weight: 700; letter-spacing: 0.08em; color: var(--info); border: 1px solid var(--info-line); background: var(--info-bg); padding: 2px 8px; border-radius: 4px; }

/* ---------- explanation ---------- */
.ex-title { margin: 0; font-size: 18px; font-weight: 700; }
.ex-line { margin: 4px 0 0; font-size: 17px; color: var(--text); line-height: 1.4; max-width: 70ch; }
.ex-next { margin: 4px 0 0; font-family: var(--font-mono); font-size: 13px; color: var(--muted); }

/* ---------- details ---------- */
.details { background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); margin-top: 12px; }
.dt-h { display: flex; align-items: center; gap: 10px; padding: 10px 16px; border-bottom: 1px solid var(--border); }
.dt-h h2 { margin: 0; font-size: 16px; font-weight: 700; margin-right: 6px; }
.tabs { display: flex; gap: 4px; }
.tabs button { padding: 6px 12px; border-radius: 6px; border: 1px solid transparent; background: transparent; color: var(--text-2); font-size: 14px; font-weight: 600; }
.tabs button.is-active { background: var(--surface); border-color: var(--border-2); color: var(--text); }
/* certificate: the recorded JSON file as a collapsible tree */
.cj-bar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 0 0 8px; }
.cj-bar .ck-title { margin: 0; }
.cj-bar .spacer { flex: 1; }
.seg { display: inline-flex; gap: 2px; padding: 2px; border: 1px solid var(--border-2); border-radius: 6px; }
.seg button { padding: 2px 8px; border: 0; border-radius: 4px; background: transparent; color: var(--text-2); font-family: var(--font-mono); font-size: 12px; }
.seg button.is-active { background: var(--hover); color: var(--text); }
.cj { font-family: var(--font-mono); font-size: 12.5px; line-height: 1.55; background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px 10px 8px; color: var(--text-2); overflow-x: auto; }
.cj .j-row { position: relative; padding-left: 14px; white-space: pre-wrap; overflow-wrap: anywhere; border-radius: 3px; }
.cj .j-in { margin-left: 16px; }
.cj details > summary { list-style: none; cursor: pointer; }
.cj details > summary::-webkit-details-marker { display: none; }
.cj details > summary::before { content: "▸"; position: absolute; left: 1px; color: var(--muted); }
.cj details[open] > summary::before { content: "▾"; }
.cj details[open] > summary .j-hint { display: none; }
.cj details:not([open]) > .j-close { display: none; }
.cj .j-k { color: var(--j-key); }
.cj .j-s { color: var(--j-str); }
.cj .j-n { color: var(--j-num); }
.cj .j-l { color: var(--j-lit); }
.cj .j-hint { color: var(--muted); font-style: italic; }
.cj .j-flow { padding-left: 14px; }
.cj .j-ent { display: inline-block; margin-right: 12px; white-space: nowrap; border-radius: 3px; }
.cj .j-hl--red { background: var(--danger-bg); box-shadow: inset 2px 0 0 var(--danger); }
.cj .j-hl--green { background: var(--success-bg); box-shadow: inset 2px 0 0 var(--success); }
.cj .j-ent.j-hl--red, .cj .j-ent.j-hl--green { padding: 0 4px; box-shadow: none; outline: 1px solid currentColor; }
.cj .j-ent.j-hl--red { color: var(--danger); }
.cj .j-ent.j-hl--green { color: var(--success); }
.dt-body { padding: 14px 16px 16px; font-size: 14px; }
.dt-foot { margin: 14px 0 0; padding-top: 10px; border-top: 1px solid var(--border); font-size: 13px; color: var(--text-2); }
.ck { display: grid; grid-template-columns: 24px 44px 1fr; gap: 6px 10px; align-items: baseline; }
.ck > div { padding: 4px 0; border-bottom: 1px solid var(--border); }
.ck .ck-s { font-weight: 700; }
.ck .ck-id { font-family: var(--font-mono); font-size: 12px; color: var(--text-2); }
.ck [data-s="ok"] { color: var(--success); }
.ck [data-s="bad"] { color: var(--danger); }
.ck [data-s="cav"] { color: var(--amber); }
.ck-title { font-size: 13px; color: var(--text-2); margin: 10px 0 6px; font-weight: 600; }
.ck-title:first-child { margin-top: 0; }
.raw { font-family: var(--font-mono); font-size: 12.5px; line-height: 1.5; white-space: pre-wrap; overflow-wrap: anywhere; background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; margin: 0; color: var(--text-body); max-height: 9.2em; overflow: hidden; position: relative; }
.raw.open { max-height: none; }
.raw-more { margin: 6px 0 0; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; font-size: 14px; }
.kv .k { color: var(--text-2); }
.kv .v { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.eq { font-family: var(--font-mono); font-size: 13px; color: var(--text-2); margin: 8px 0 12px; }
.dt-sec { margin: 0 0 14px; }
.dt-sec h3 { margin: 0 0 6px; font-size: 14px; font-weight: 700; display: flex; gap: 8px; align-items: center; }
.tbl { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.tbl th { text-align: left; font-weight: 600; color: var(--muted); font-size: 11px; letter-spacing: 0.06em; padding: 4px 8px; border-bottom: 1px solid var(--border); }
.tbl td { padding: 5px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
.tbl td.n { font-family: var(--font-mono); font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
.tbl tr[data-tone="red"] td { color: var(--danger); }
.tbl tr[data-tone="green"] td { color: var(--success); }
.tbl tr[data-tone="amber"] td { color: var(--amber); }
.empty { color: var(--muted); font-size: 14px; }
.orders { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.dt-x { margin-left: auto; }

/* ---------- about ---------- */
.about { position: fixed; inset: 0; z-index: 1000; background: rgba(5, 8, 24, 0.72); display: grid; place-items: center; padding: 24px; }
.about-panel { width: min(1100px, 100%); max-height: calc(100vh - 48px); overflow: auto; background: var(--card); border: 1px solid var(--border-2); border-radius: 12px; padding: 20px 24px 24px; }
.about-h { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.about-h h2 { margin: 0; font-size: 20px; }
.ab-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px 22px; font-size: 14px; }
.ab { min-width: 0; }
.ab h3 { margin: 0 0 6px; font-size: 14px; font-weight: 700; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.ab p { margin: 0 0 6px; color: var(--text-body); line-height: 1.45; }
.ab ul { margin: 0; padding-left: 18px; color: var(--text-body); line-height: 1.45; }
.ab li { margin: 0 0 3px; }
.ab .wide { grid-column: 1 / -1; }
.legend { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.legend div { font-size: 13px; color: var(--text-2); line-height: 1.4; }
.legend .chip { display: table; margin-bottom: 4px; }
.ab-last { grid-column: 1 / -1; font-size: 13px; color: var(--text-2); border-top: 1px solid var(--border); padding-top: 10px; }

/* ---------- case 2: labels, translation, evidence ---------- */
:host { --p1: #60a5fa; --p2: #a78bfa; --p3: #a3e635; --p4: #fb923c; --p5: #94a3b8; --w9: #c6b34a; --lead: #46547f; --j-key: #7dd3fc; --j-str: #a5d6a7; --j-num: #fbbf77; --j-lit: #c4b5fd; }
.badge-mock { display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border-radius: 6px; border: 1.5px dashed var(--amber); color: var(--amber); font-family: var(--font-mono); font-size: 13px; font-weight: 600; letter-spacing: 0.02em; white-space: nowrap; }
.chip--stand { color: var(--l-model); background: transparent; border-style: dashed; border-color: var(--l-model-line); }
.chip--plant { color: var(--danger); background: var(--danger-bg); border-style: dashed; border-color: var(--danger-line); }
.tile[data-tone="green"] .t-v { color: var(--success); }
.pie .lbl .over--amber { fill: var(--amber); }
.pie .lead.hot--amber { stroke: var(--amber-line); }
.pie .lead-dot { fill: var(--text-2); }
.pie .lead-dot.hot { fill: var(--danger); }
.pie .lead-dot.hot--amber { fill: var(--amber); }

.tr-req { margin: 0 0 8px; padding: 8px 10px; border-radius: 6px; background: var(--surface); border: 1px solid var(--border); font-size: 14px; line-height: 1.55; }
.tr-req-k { display: block; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; color: var(--muted); text-transform: uppercase; margin-bottom: 2px; }
.tr-none { color: var(--muted); font-size: 13px; }
.tr-q { quotes: "\201C" "\201D"; color: var(--text-body); }
.hl { --c: var(--p1); color: var(--text); background: color-mix(in srgb, var(--c) 16%, transparent); border-bottom: 2px solid var(--c); padding: 0 2px; border-radius: 2px; }
.hl[data-n="2"] { --c: var(--p2); } .hl[data-n="3"] { --c: var(--p3); } .hl[data-n="4"] { --c: var(--p4); } .hl[data-n="5"] { --c: var(--p5); }
.hl sup { font-family: var(--font-mono); font-size: 10px; font-weight: 700; color: var(--c); margin-left: 2px; }
.pii { display: inline-flex; align-items: center; gap: 3px; padding: 0 6px; margin: 0 1px; border-radius: 4px; border: 1px dashed var(--l-code-line); color: var(--l-code); font-family: var(--font-mono); font-size: 11.5px; vertical-align: 1px; }
.pii svg { width: 10px; height: 10px; stroke: currentColor; fill: none; stroke-width: 2; }
.trans td { padding: 5px 7px; }
.trans .t-n { width: 26px; padding-right: 0; }
.nb { --c: var(--p1); display: inline-grid; place-items: center; width: 20px; height: 20px; border-radius: 50%; font-family: var(--font-mono); font-size: 11px; font-weight: 700; color: #0a0e27; background: var(--c); }
.nb[data-n="2"] { --c: var(--p2); } .nb[data-n="3"] { --c: var(--p3); } .nb[data-n="4"] { --c: var(--p4); } .nb[data-n="5"] { --c: var(--p5); }
.nb--mand { background: transparent; color: var(--muted); border: 1px solid var(--border-2); }
.trans .t-w { width: 33%; color: var(--text); }
.trans .t-mq { color: var(--text-2); } .trans .t-mq b { font-family: var(--font-mono); font-size: 11px; color: var(--muted); font-weight: 600; } .trans .t-mq q { quotes: "\201C" "\201D"; }
.t-mand { display: block; font-size: 11.5px; color: var(--muted); margin-top: 2px; line-height: 1.3; }
.trans .t-p { color: var(--text-body); }
.trans .t-num { font-family: var(--font-mono); font-size: 13px; white-space: nowrap; color: var(--text); }
.t-was { display: block; font-family: var(--font-sans); font-size: 11px; color: var(--muted); white-space: normal; }
.num-warn { color: var(--amber); }
.trans .t-f { width: 1%; }
.trans .t-empty { color: var(--muted); }
.src { display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: 11.5px; font-weight: 600; white-space: nowrap; border: 1px solid var(--border-2); color: var(--text-2); }
.src[data-kind="mand"] { color: var(--l-team); border-color: var(--l-team-line); border-style: dashed; }
.src[data-kind="data"] { color: var(--l-code); border-color: var(--l-code-line); background: var(--l-code-bg); }
.src[data-kind="profile"] { color: var(--p5); border-style: dotted; }
.src[data-kind="bad"] { color: var(--danger); border-color: var(--danger); background: var(--danger-bg); }
.src[data-kind="warn"] { color: var(--amber); border-color: var(--amber-line); background: var(--amber-bg); }
.src[data-kind="fixed"] { color: var(--success); border-color: var(--success-line); background: var(--success-bg); }
.trans tr[data-tone="red"] td { background: var(--danger-bg); }
.trans tr[data-tone="red"] td:first-child { box-shadow: inset 3px 0 0 var(--danger); }
.trans tr.reveal { animation: rowin 0.45s ease both; }
@keyframes rowin { from { opacity: 0; transform: translateX(-8px); } to { opacity: 1; transform: none; } }

.lp-ev { margin-top: 8px; padding: 8px 12px 6px; border-radius: 6px; border: 1px solid var(--border); background: var(--surface); }
.ev-h { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
.ev-t { font-size: 13px; font-weight: 700; }
.ev-lg { display: flex; gap: 14px; font-size: 12px; color: var(--text-2); margin-left: auto; }
.lg::before { content: ""; display: inline-block; width: 16px; height: 0; border-top: 2px solid; vertical-align: middle; margin-right: 6px; }
.lg--fed::before { border-color: var(--danger); } .lg--src::before { border-color: var(--l-code); border-top-style: dashed; }
.ko { width: 100%; height: 96px; display: block; }
.ko .grid { stroke: var(--border); stroke-width: 1; }
.ko .ax { font-family: var(--font-mono); font-size: 11px; fill: var(--muted); }
.ko .fed { fill: none; stroke: var(--danger); stroke-width: 2; stroke-linejoin: round; }
.ko .src { fill: none; stroke: var(--l-code); stroke-width: 1.6; stroke-dasharray: 5 4; }
.ko .cliff { fill: var(--danger); }
.ko .ann { font-family: var(--font-sans); font-size: 12.5px; font-weight: 600; fill: var(--danger); }
.cmp { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.cmp th { text-align: right; font-size: 11px; font-weight: 600; letter-spacing: 0.04em; color: var(--muted); padding: 2px 8px 4px; border-bottom: 1px solid var(--border); }
.cmp th:first-child { text-align: left; }
.cmp td { padding: 3px 8px; border-bottom: 1px solid var(--border); }
.cmp td.n { text-align: right; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.cmp tr:last-child td { border-bottom: 0; }
.cmp tr[data-r="1"] td { color: var(--danger); }
.cmp tr[data-r="2"] td { color: var(--success); }
.legend { grid-template-columns: repeat(5, 1fr); }

/* ---------- modes ---------- */
.wr[data-mode="shell"] .dt-x { display: none; }
.wr[data-mode="shell"] .box-chat { grid-row: 1; }
.wr[data-mode="shell"] .box-loop, .wr[data-mode="shell"] .box-explain { grid-column: 1 / 4; }
.wr[data-mode="shell"] .grid { grid-template-rows: auto auto auto; }
.wr[data-mode="shell"] .thread { min-height: 0; }

/* ---------- 1440 and narrower ---------- */
@media (max-width: 1500px) {
  .ph-title { font-size: 22px; }
  .rules { font-size: 13.5px; }
  .rules .r-w { width: 44%; }
  .msg--client .bub, .msg--proposer .bub { font-size: 14px; }
  .ex-line { font-size: 16px; }
  .lrow { font-size: 12.5px; }
  .loop { grid-template-columns: 1fr 22px 1.3fr 22px 1.15fr 22px 1.15fr 22px 1fr; }
  .box-h .chip { font-size: 11px; padding: 0 6px; }
}
@media (max-width: 1100px) {
  .grid { grid-template-columns: 1fr 1fr; }
  .box-chat { grid-column: 1 / 3; grid-row: 4; min-height: 520px; }
  .wr[data-mode="shell"] .box-chat { grid-column: 1 / 3; grid-row: 4; }
  .box-loop, .box-explain { grid-column: 1 / 3; }
}
@media (max-width: 760px) {
  .grid { grid-template-columns: 1fr; }
  .box-portfolio, .box-rules, .box-chat, .box-loop, .box-explain { grid-column: 1; grid-row: auto; }
  .loop { grid-template-columns: 1fr; }
  .arrow, .grp { display: none; }
  .lanes { display: none; }
  .ab-grid, .orders, .legend { grid-template-columns: 1fr; }
}
.trans .t-w { width: auto; white-space: nowrap; }
.trans .t-num { white-space: nowrap; }
.trans .t-mq { white-space: normal; font-size: 12.5px; }
@media (max-width: 1500px) {
  .trans { font-size: 13px; }
  .trans .t-w { white-space: normal; }
  .src { white-space: normal; font-size: 11px; }
  .trans td { padding: 4px 5px; }
}
.ab.wide { grid-column: 1 / -1; }

/* case 2, recorded */
.badge-rec { display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border-radius: 6px; border: 1.5px solid var(--l-model-line); color: var(--l-model); background: var(--l-model-bg); font-family: var(--font-mono); font-size: 13px; font-weight: 600; letter-spacing: 0.02em; white-space: nowrap; }
.ev-note { margin: 4px 0 0; font-size: 11.5px; color: var(--muted); }
.sm-line { margin: 4px 0; font-size: 13px; line-height: 1.45; }
.ab-note { margin: 8px 0 0; font-size: 13px; color: var(--muted); line-height: 1.5; }
.ck b { font-weight: 600; }
.cmp tr[data-r=""] td { color: inherit; }
.ev-note--red { color: var(--danger); font-weight: 600; }
`;
  var MARKUP = String.raw`<div class="wr" id="wr" data-mode="shell">
  <div class="page" id="page">

    <div class="ph">
      <div>
        <h1 class="ph-title">Weekly Portfolio Rebalancing</h1>
        <p class="ph-sub">Dow 30 + APH · every Friday after the close · this run: the decision of <span class="nowrap">Fri Sep 4, 2026</span></p>
      </div>
      <div class="ph-right">
        <span class="badge-rec" title="Recorded Nov 4, 2026: DeepSeek V4 Pro and Claude Sonnet 4.6 (real API calls), Gurobi 13.0.3 and our code checks. This page plays run 1 of 5.">RECORDED RUN · 1 of 5 · Nov 4</span>
        <button type="button" class="btn" id="aboutBtn">About this run</button>
        <div class="play" role="group" aria-label="Playback">
          <button type="button" class="btn" data-act="prev" aria-label="Back one step">←</button>
          <button type="button" class="btn btn--primary" data-act="play">▶ Play</button>
          <button type="button" class="btn" data-act="next" aria-label="Next step">→</button>
          <span class="play-count" data-count>step 1 / 9</span>
        </div>
      </div>
    </div>

    <div class="grid">
      <section class="box box-portfolio" aria-labelledby="pf-h">
        <div class="box-h">
          <h2 id="pf-h">Portfolio</h2>
          <span class="state-chip" id="pf-state" data-state="held">HELD</span>
          <span class="sub" id="pf-sub">Held · Fri Sep 4 close</span>
          <span class="spacer"></span>
          <span class="chip chip--sim">Simulated</span>
        </div>
        <div class="pie-wrap"><svg class="pie" id="pie" viewBox="14 -6 532 262" role="img" aria-label="Portfolio weights"></svg></div>
        <div class="tiles" id="tiles"></div>
      </section>

      <section class="box box-rules box-tr" aria-labelledby="tr-h">
        <div class="box-h">
          <h2 id="tr-h">Your words → the problem</h2>
          <span class="ver-chip" id="tr-ver">mandate of Aug 28</span>
          <span class="tag" id="tr-tag" hidden>rules unchanged</span>
          <span class="spacer"></span>
          <span class="chip chip--team chip--sm" title="The client, the message and the mandate were written by our team">Written by our team</span>
          <span class="chip chip--model chip--sm" title="The phrase-to-term mapping is the proposer's recorded output">Model · recorded</span>
          <span class="chip chip--code chip--sm" title="Numbers and sources are computed by our code">Code &amp; solver</span>
        </div>
        <div class="tr-req" id="tr-req"></div>
        <table class="rules trans" id="trans">
          <thead><tr><th></th><th>YOUR WORDS</th><th>IN THE PROBLEM</th><th>NUMBER</th><th>COMES FROM</th></tr></thead>
          <tbody id="tr-body"></tbody>
        </table>
        <p class="rules-foot" id="tr-foot"></p>
      </section>

      <section class="box box-chat" aria-labelledby="ch-h">
        <div class="chat-inner">
          <div class="chat-h box-h" style="margin-bottom:0">
            <h2 id="ch-h">Chat</h2>
            <span class="sub">your agent</span>
            <span class="spacer"></span>
            <span class="chip chip--model" title="The proposer's message is its recorded output">Proposer · DeepSeek V4 Pro</span>
          </div>
          <div class="thread" id="thread" aria-live="polite"></div>
          <div class="chat-foot">
            <div class="compose">
              <input type="text" id="composer" placeholder="Message your agent…" aria-label="Message your agent" disabled>
              <button type="button" class="send" id="sendBtn" aria-label="Send" disabled><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7"/></svg></button>
            </div>
            <p class="chat-note">Playing recorded run 1 of 5 (Nov 4, 2026)</p>
          </div>
        </div>
      </section>

      <section class="box box-loop" aria-labelledby="lp-h">
        <div class="loop-h">
          <h2 id="lp-h">This week's rebalance</h2>
          <span class="nothing" id="lp-nothing" hidden>NOTHING TRADES</span>
          <span class="spacer" style="flex:1"></span>
          <span class="round" id="lp-round"></span>
          <div class="play play--loop" role="group" aria-label="Playback">
            <button type="button" class="btn" data-act="prev" aria-label="Back one step">←</button>
            <button type="button" class="btn btn--primary" data-act="play">▶ Play</button>
            <button type="button" class="btn" data-act="next" aria-label="Next step">→</button>
            <span class="play-count" data-count>step 1 / 9</span>
          </div>
        </div>
        <div class="loop-wrap">
          <div class="loop" id="loop">
            <div class="grp" style="grid-column:1">REQUEST</div>
            <div class="grp" style="grid-column:3">PROPOSAL</div>
            <div class="grp" style="grid-column:5 / 8">CERTIFICATION</div>
            <div class="grp" style="grid-column:9">DECISION</div>
            <div class="lbox" id="b-client" style="grid-column:1"><div class="lbox-t">Client</div><div class="lbox-s">you</div><div class="lrows"></div><div class="lbox-foot"><span class="chip chip--team chip--sm">Written by our team</span></div></div>
            <div class="arrow" id="a1" style="grid-column:2"><svg viewBox="0 0 22 16"><path d="M1 8h18M14 3l5 5-5 5"/></svg></div>
            <div class="lbox" id="b-proposer" style="grid-column:3"><div class="lbox-t">Proposer</div><div class="lbox-s">DeepSeek V4 Pro</div><div class="lrows"></div><div class="lbox-foot"><span class="chip chip--model chip--sm">Model · recorded</span><span class="chip chip--code chip--sm">Gurobi 13.0.3</span></div></div>
            <div class="arrow" id="a2" style="grid-column:4"><svg viewBox="0 0 22 16"><path d="M1 8h18M14 3l5 5-5 5"/></svg></div>
            <div class="lbox" id="b-checks" style="grid-column:5"><div class="lbox-t">Checks</div><div class="lbox-s">our code, 15 checks</div><div class="lrows"></div><div class="lbox-foot"><span class="chip chip--code chip--sm">Code &amp; solver</span></div></div>
            <div class="arrow" id="a3" style="grid-column:6"><svg viewBox="0 0 22 16"><path d="M1 8h18M14 3l5 5-5 5"/></svg></div>
            <div class="lbox" id="b-critic" style="grid-column:7"><div class="lbox-t">Critic</div><div class="lbox-s">Claude Sonnet 4.6</div><div class="lrows"></div><div class="lbox-foot"><span class="chip chip--model chip--sm">Model · recorded</span></div></div>
            <div class="arrow" id="a4" style="grid-column:8"><svg viewBox="0 0 22 16"><path d="M1 8h18M14 3l5 5-5 5"/></svg></div>
            <div class="lbox" id="b-gate" style="grid-column:9"><div class="lbox-t">Gate</div><div class="lbox-s">decides</div><div class="lrows"></div><div class="lbox-foot"><span class="chip chip--code chip--sm">Code &amp; solver</span></div></div>
          </div>
          <svg class="lanes" id="lanes" aria-hidden="true"></svg>
        </div>
        <p class="under" id="lp-under" hidden></p>
        <div class="lp-ev" id="lp-ev" hidden></div>
      </section>

      <section class="box box-explain" aria-labelledby="ex-title">
        <h2 class="ex-title" id="ex-title"></h2>
        <p class="ex-line" id="ex-line"></p>
        <p class="ex-next" id="ex-next"></p>
      </section>
    </div>

    <section class="details" id="details" aria-labelledby="dt-h">
      <div class="dt-h">
        <h2 id="dt-h">Details</h2>
        <div class="tabs" role="tablist">
          <button type="button" role="tab" data-tab="proposer" class="is-active" aria-selected="true">Proposer</button>
          <button type="button" role="tab" data-tab="solver" aria-selected="false">Solver</button>
          <button type="button" role="tab" data-tab="checks" aria-selected="false">Checks</button>
          <button type="button" role="tab" data-tab="certificate" aria-selected="false">Certificate</button>
          <button type="button" role="tab" data-tab="critic" aria-selected="false">Critic</button>
        </div>
      </div>
      <div class="dt-body" id="dt-body"></div>
    </section>
  </div>

  <div class="about" id="about" hidden role="dialog" aria-modal="true" aria-labelledby="about-h">
    <div class="about-panel">
      <div class="about-h">
        <h2 id="about-h">About this run</h2>
        <span class="badge-rec" title="Recorded Nov 4, 2026: DeepSeek V4 Pro and Claude Sonnet 4.6 (real API calls), Gurobi 13.0.3 and our code checks. This page plays run 1 of 5.">RECORDED RUN · 1 of 5 · Nov 4</span>
        <span class="spacer" style="flex:1"></span>
        <button type="button" class="btn" id="aboutClose">Close</button>
      </div>
      <div class="ab-grid">
        <div class="ab wide">
          <h3>What the labels mean</h3>
          <div class="legend">
            <div><span class="chip chip--code">Code &amp; solver</span>What our code computed or Gurobi solved in the recorded run: every weight, risk figure, check result and order on this page.</div>
            <div><span class="chip chip--model">Model · recorded</span>A reply of the proposer (DeepSeek V4 Pro) or the critic (Claude Sonnet 4.6), as the API returned it on Nov 4, 2026.</div>
            <div><span class="chip chip--plant">Feed rebuilt from Yahoo</span>The firm’s price feed: as-traded prices, rebuilt as Yahoo’s split-adjusted close times every split factor Yahoo records. No keyed vendor feed was used.</div>
            <div><span class="chip chip--team">Written by our team</span>The client, the message, the mandate, the choice to add APH to the list, and every plain-words line.</div>
            <div><span class="chip chip--sim">Simulated</span>A $100,000 account bought at equal weight on Mon Aug 31, and its trades. Not a real account.</div>
          </div>
        </div>
        <div class="ab wide">
          <h3>The runs <span class="chip chip--code chip--sm">Code &amp; solver</span></h3>
          <div id="ab-runs"></div>
          <p class="ab-note">Eleven earlier real runs ran on earlier code. In them the critic caught two wrong client messages that code’s number check had passed: one promised “later planning” for details the problem had set aside, one called an estimated 1.90% monthly return a target. Both came from our prompts, which we then fixed.</p>
        </div>
        <div class="ab">
          <h3>What is real, what is not</h3>
          <ul>
            <li><b>Real and connected:</b> both models (API calls through CommonStack), Gurobi 13.0.3, the redaction, every code check (format, mandate values, units, sources, your words, prices D1, corporate actions D2, re-solve, limits, model dates, personal details), the gate, the certificates and the recording.</li>
            <li><b>Real data:</b> APH’s 2-for-1 split (announced Aug 6, record date Aug 17, shares paid Sep 2, split-adjusted trading from Sep 3), Yahoo’s split record and its prices.</li>
            <li><b>Rebuilt:</b> the feed as delivered. <b>Same vendor:</b> the reference prices D1 compares with, so D1 cannot miss here; in production it would be a second vendor.</li>
            <li><b>Scripted:</b> the client, the message, the mandate. <b>Simulated:</b> the account and the fills.</li>
          </ul>
        </div>
        <div class="ab">
          <h3>What the critic added here</h3>
          <p>Code found the split, stopped the round and built the adjusted prices. In five runs with the critic switched off, code’s own objection led the proposer to the same fix and the same eight orders. The critic explained the fault, tied it to the plan and the return floor, and reviewed the two HON spin-offs code could not verify. Its clearest value showed in the client message, not in the data.</p>
        </div>
        <div class="ab">
          <h3>Point in time</h3>
          <ul>
            <li><b>Data:</b> a date filter at the Fri Sep 4 close on files downloaded Nov 3; the executor alone reads the Tue Sep 8 opens. No model input contains a later date.</li>
            <li><b>Models:</b> the critic’s training data ends Jan 2026 (documented). DeepSeek publishes no cutoff; its last V4 Pro update was Aug 13, after APH announced the split on Aug 6 and before the split itself.</li>
          </ul>
        </div>
        <div class="ab">
          <h3>Limits and what is planned</h3>
          <ul>
            <li>One scenario, chosen because APH split that week. Expected returns are 252-day averages, not forecasts.</li>
            <li>The data checks run at the gate, after the proposer binds; production would also run them when data arrive.</li>
            <li>When the record lacks the split, the proposer tried to answer “no clean data, nothing can trade”; the output format has no such answer yet, so the run held after three rounds. Planned: a data hold that goes to a person.</li>
            <li>Planned: a second price vendor, the raw Alpaca bars ATL already fetches, and a news feed with publish times.</li>
          </ul>
        </div>
      </div>
      
<div class="ab-grid ab-grid--last">
        <p class="ab-last">Passing these checks would not establish forecast accuracy, profitability, or comprehensive legal compliance.</p>
      </div>
    </div>
  </div>
  <p class="sr-only" id="live" aria-live="polite"></p>
</div>`;
  var shadow = null, panel = null, active = false;
  var $ = function (id) { return shadow.getElementById(id); };
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function svgEl(tag, attrs) {
    var e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs) if (Object.prototype.hasOwnProperty.call(attrs, k)) e.setAttribute(k, attrs[k]);
    return e;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function chip(cls, text) { return el('span', 'chip ' + cls, text); }
  function pc(x, k) { return (100 * x).toFixed(k == null ? 2 : k) + '%'; }
  function sg(v) { return (v < 0 ? '\u2212' : '+') + Math.abs(v).toFixed(2) + '%'; }
  var LAST = 9;
  var R = window.PortfolioManagementRecording;
  if (!R) return;

  /* ------------------------------------------------------------------ data derived from the recording */
  var WCOLOR = { APH: 1, CVX: 2, KO: 3, JNJ: 4, IBM: 5, NKE: 6, SHW: 7, GS: 8 };
  var CF = R.cf, W1 = R.w1, W2 = R.w2;
  var APH_HELD = R.pies.held.wedges[0][1], APH_R1 = R.pies.plan1.wedges[0][1];
  var FLOOR1 = pc(W1.floor_monthly), FLOOR2 = pc(W2.floor_monthly);
  var N1 = R.checks.n1, N2 = R.checks.n2;
  var HON_MAX = Math.max.apply(null, R.hon.map(function (h) { return h.max_pp; }));

  var STEPS = [
    { title: 'Your portfolio, your mandate, your list',
      line: 'A simulated $100,000 account bought the 31 stocks of your list in equal amounts on Mon Aug 31. The list is the Dow 30 plus Amphenol (APH), added for this case because APH split 2-for-1 on Thu Sep 3. The first weekly rebalance runs after the close on Fri Sep 4.',
      next: 'your message' },
    { title: 'Your message: personal details stay out',
      line: 'You ask for low risk with a target return, sector limits and a turnover cap, and you mention your age, family, savings and email. Code removes those four details before any model reads the message.',
      next: 'the proposer reads it' },
    { title: 'From your words to the problem',
      line: 'The proposer maps each phrase to one part of the problem. Your words carry no numbers, so each number comes from your mandate or from the data. Your retirement date goes to your profile, not into this week’s problem.',
      next: 'the data at Friday’s close' },
    { title: 'The data at Friday’s close',
      line: 'The firm’s price feed delivers prices as traded, with no adjustment step. APH split 2-for-1 on Thursday, so in the feed it fell from $' + R.aph.fed_before.toFixed(2) + ' to $' + R.aph.fed_after.toFixed(2) + ': ' + sg(R.aph.r_fed) + ' in one day. Adjusted for the split, it rose ' + R.aph.r_adj.toFixed(2) + '%.',
      next: 'Gurobi solves on these prices' },
    { title: 'Round 1: the problem, solved on the feed as delivered',
      line: 'The proposer binds the risk and return estimates built from the feed. To the solver APH now looks like a stock that just halved, so the plan sells ' + CF.aph_sold + ' of your ' + CF.aph_shares_held + ' APH shares. The same prices lower your return floor from ' + FLOOR2 + ' to ' + FLOOR1 + ' a month.',
      next: 'the code checks' },
    { title: 'Caught by code: nothing trades',
      line: 'Code compares every daily price move with reference prices. Three days disagree, and each matches a corporate action on record: APH’s split and two HON spin-offs. The gate stops the round, and code builds prices adjusted for those three events.',
      next: 'the critic’s review' },
    { title: 'The critic’s evidence and feedback',
      line: 'The critic, Claude Sonnet 4.6, reads code’s results and writes the objection: firm policy bars prices that fail the check, so use the adjusted estimates, with a return floor of ' + (100 * W2.floor_monthly).toFixed(3) + '% a month. In five runs with the critic switched off, code’s own objection led to the same fix.',
      next: 'the proposer revises' },
    { title: 'The revision: same words, same limits, adjusted prices',
      line: 'Only the data changes. APH stays at ' + APH_HELD.toFixed(2) + '%, risk falls from ' + pc(R.held_risk) + ' to ' + pc(W2.risk_monthly) + ' a month, and the floor is ' + FLOOR2 + ' again. Had round 1 traded, the risk would have been about the same; the cost was ' + CF.aph_sold + ' APH shares sold for no reason.',
      next: 'final checks and trade' },
    { title: 'Final checks, then trade',
      line: 'All ' + N2.n + ' checks pass. Two HON adjustments use the vendor’s factors, which code cannot verify; leaving either out moves no weight by more than ' + HON_MAX.toFixed(2) + ' points, and the critic passes. The gate trades ' + R.fills.n_orders + ' simulated orders at Tuesday’s open (Monday is Labor Day).',
      next: 'end of the case' }
  ];

  var CHAT = [
    { step: 1, kind: 'sys', icon: 'cal', text: 'Weekly rebalance · first run Fri Sep 4 after the close' },
    { step: 2, kind: 'client', when: R.msg.when, redact: true },
    { step: 2, kind: 'sys', icon: 'lock', text: 'Removed before any model call: age, household, savings amount, email.' },
    { step: 4, kind: 'sys', icon: 'cal', text: 'Fri Sep 4 close · APH split 2-for-1 on Thu · the feed shows ' + sg(R.aph.r_fed) },
    { step: 5, kind: 'sys', icon: 'gear', text: 'Gurobi · solved · the plan sells ' + CF.aph_sold + ' of ' + CF.aph_shares_held + ' APH shares' },
    { step: 6, kind: 'sys', icon: 'x', tone: 'red', text: 'Code · price check D1 failed on 3 days · nothing trades' },
    { step: 7, kind: 'sys', icon: 'x', tone: 'red', text: 'Critic · objection · use the adjusted prices' },
    { step: 8, kind: 'sys', icon: 'check', tone: 'green', text: 'Round 2 · adjusted prices · checks and critic pass' },
    { step: 9, kind: 'proposer', when: R.sent.at, text: R.sent.text },
    { step: 9, phase: 'b', kind: 'sys', icon: 'receipt', tone: 'green', text: 'Tue Sep 8 open · ' + R.fills.n_orders + ' orders · cash left ' + R.fills.cash_after_pct.toFixed(2) + '% · simulated' }
  ];

  var ICONS = {
    cal: 'M3 5h18v16H3zM3 10h18M8 3v4M16 3v4',
    lock: 'M5 11h14v10H5zM8 11V7a4 4 0 0 1 8 0v4',
    gear: 'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1 7 17M17 7l2.1-2.1',
    check: 'M20 6 9 17l-5-5',
    x: 'M6 6l12 12M18 6 6 18',
    receipt: 'M5 3h14v18l-2-1.5L15 21l-2-1.5L11 21l-2-1.5L7 21l-2-1.5zM8 8h8M8 12h8M8 16h5'
  };
  function icon(name) {
    var s = svgEl('svg', { viewBox: '0 0 24 24', 'aria-hidden': 'true' });
    s.appendChild(svgEl('path', { d: ICONS[name] }));
    return s;
  }

  /* ------------------------------------------------------------------ state */
  var S = { step: 1, phase: 'a', tab: 'proposer', tabPinned: false, certRound: 0, about: false, playing: false, timer: null, phaseTimer: null, typeTimer: null, sendTimer: null };

  /* ------------------------------------------------------------------ pie */
  function arcPath(cx, cy, Ro, r, a0, a1) {
    var large = (a1 - a0) > Math.PI ? 1 : 0;
    var x0 = cx + Ro * Math.cos(a0), y0 = cy + Ro * Math.sin(a0), x1 = cx + Ro * Math.cos(a1), y1 = cy + Ro * Math.sin(a1);
    var ix0 = cx + r * Math.cos(a1), iy0 = cy + r * Math.sin(a1), ix1 = cx + r * Math.cos(a0), iy1 = cy + r * Math.sin(a0);
    return 'M' + x0.toFixed(2) + ',' + y0.toFixed(2) + ' A' + Ro + ',' + Ro + ' 0 ' + large + ' 1 ' + x1.toFixed(2) + ',' + y1.toFixed(2) +
      ' L' + ix0.toFixed(2) + ',' + iy0.toFixed(2) + ' A' + r + ',' + r + ' 0 ' + large + ' 0 ' + ix1.toFixed(2) + ',' + iy1.toFixed(2) + ' Z';
  }
  function renderPie(stateKey, cutTone) {
    var P = R.pies[stateKey], svg = $('pie');
    clear(svg);
    var defs = svgEl('defs', {});
    var pat = svgEl('pattern', { id: 'hatch', patternUnits: 'userSpaceOnUse', width: '7', height: '7', patternTransform: 'rotate(45)' });
    pat.appendChild(svgEl('line', { x1: '0', y1: '0', x2: '0', y2: '7', stroke: '#ff4141', 'stroke-width': '2.2' }));
    defs.appendChild(pat);
    svg.appendChild(defs);
    var cx = 280, cy = 128, Ro = 118, r = 70;   // viewBox 14 -6 532 262 (markup): content only, so the ring scales up
    var g = svgEl('g', { class: 'w' });
    var named = 0;
    var items = P.wedges.map(function (w) { named += w[1]; return { t: w[0], v: w[1], cut: P.cut === w[0] }; });
    items.push({ t: 'others', v: Math.round((100 - named) * 10) / 10, others: true });
    var a = -Math.PI / 2, labels = [];
    items.forEach(function (it) {
      var a1 = a + it.v / 100 * Math.PI * 2;
      var d = arcPath(cx, cy, Ro, r, a, a1);
      var fill = it.others ? 'var(--w-others)' : 'var(--w' + WCOLOR[it.t] + ')';
      var p = svgEl('path', { d: d, stroke: '#131a35', 'stroke-width': '2' });
      p.setAttribute('style', 'fill:' + fill);
      g.appendChild(p);
      if (it.cut && cutTone === 'red') {
        g.appendChild(svgEl('path', { d: d, fill: 'url(#hatch)', stroke: 'none' }));
        g.appendChild(svgEl('path', { d: d, class: 'hatch-outline' }));
      }
      labels.push({ it: it, ang: (a + a1) / 2 });
      a = a1;
    });
    svg.appendChild(g);
    var side = { right: [], left: [] };
    labels.forEach(function (L) {
      var c = Math.cos(L.ang), s = Math.sin(L.ang);
      // the line starts inside its own slice and leaves it along the slice's mid-angle, then bends to the label
      L.px = cx + (Ro - 8) * c; L.py = cy + (Ro - 8) * s;
      L.ax = cx + (Ro + 8) * c; L.ay = cy + (Ro + 8) * s;
      L.ey = L.ay;
      L.right = c >= 0;
      L.y = L.ey;
      (L.right ? side.right : side.left).push(L);
    });
    ['right', 'left'].forEach(function (k) {
      var arr = side[k].sort(function (p, q) { return p.y - q.y; }), gap = 18, top = 12, bot = 248, i;
      if (!arr.length) return;
      if (arr[0].y < top) arr[0].y = top;
      for (i = 1; i < arr.length; i++) if (arr[i].y < arr[i - 1].y + gap) arr[i].y = arr[i - 1].y + gap;
      if (arr[arr.length - 1].y > bot) {
        arr[arr.length - 1].y = bot;
        for (i = arr.length - 2; i >= 0; i--) if (arr[i].y > arr[i + 1].y - gap) arr[i].y = arr[i + 1].y - gap;
      }
    });
    var gl = svgEl('g', { class: 'labels' });
    labels.forEach(function (L) {
      var ex = L.right ? cx + Ro + 22 : cx - Ro - 22;   // one elbow column per side
      var lx = L.right ? cx + Ro + 34 : cx - Ro - 34;
      var tx = L.right ? lx + 4 : lx - 4;
      var hot = L.it.cut && cutTone, hc = hot ? ' hot hot--' + cutTone : '';
      gl.appendChild(svgEl('polyline', { class: 'lead' + hc, points: L.px.toFixed(1) + ',' + L.py.toFixed(1) + ' ' + L.ax.toFixed(1) + ',' + L.ay.toFixed(1) + ' ' + ex + ',' + L.y.toFixed(1) + ' ' + lx + ',' + L.y.toFixed(1) }));
      gl.appendChild(svgEl('circle', { class: 'lead-dot' + hc, cx: L.px.toFixed(1), cy: L.py.toFixed(1), r: Math.max(1.3, Math.min(2.6, (Ro - 8) * L.it.v / 100 * Math.PI - 0.4)).toFixed(1) }));   // fits a thin slice
      var t = svgEl('text', { class: 'lbl', x: tx, y: (L.y + 4).toFixed(1), 'text-anchor': L.right ? 'start' : 'end' });
      var ts = svgEl('tspan', {}); ts.textContent = L.it.t; t.appendChild(ts);
      var tp = svgEl('tspan', { class: 'pct' }); tp.textContent = ' ' + L.it.v.toFixed(L.it.others ? 1 : 2) + '%'; t.appendChild(tp);
      if (hot) { var to = svgEl('tspan', { class: 'over over--' + cutTone }); to.textContent = ' ↓'; t.appendChild(to); }
      gl.appendChild(t);
    });
    svg.appendChild(gl);
    var c1 = svgEl('text', { class: 'c-state', x: cx, y: cy - 2, 'text-anchor': 'middle' }); c1.textContent = P.center[0]; svg.appendChild(c1);
    var c2 = svgEl('text', { class: 'c-date', x: cx, y: cy + 18, 'text-anchor': 'middle' }); c2.textContent = P.center[1]; svg.appendChild(c2);
    $('pf-state').textContent = P.chip; $('pf-state').setAttribute('data-state', stateKey === 'held' ? 'held' : stateKey === 'traded' ? 'traded' : 'planned');
    $('pf-sub').textContent = P.sub;
  }
  function renderTiles(key) {
    var host = $('tiles'); clear(host);
    R.tiles[key].forEach(function (t) {
      var d = el('div', 'tile'); if (t[3]) d.setAttribute('data-tone', t[3]);
      d.appendChild(el('div', 't-k', t[0])); d.appendChild(el('div', 't-v num', t[1])); d.appendChild(el('div', 't-l', t[2]));
      host.appendChild(d);
    });
  }
  function renderPortfolio() {
    var st = S.step, key = 'held', tiles = 'held', cut = null;
    if (st >= 5 && st <= 7) { key = 'plan1'; tiles = st >= 6 ? 'plan1bad' : 'plan1'; cut = st >= 6 ? 'red' : 'amber'; }
    if (st === 8) { key = 'plan2'; tiles = 'plan2'; }
    if (st === 9) { key = S.phase === 'b' ? 'traded' : 'plan2'; tiles = S.phase === 'b' ? 'traded' : 'plan2'; }
    renderPie(key, cut); renderTiles(tiles);
  }

  /* ------------------------------------------------------------------ translation: your words -> the problem */
  function srcChip(T) {
    var st = S.step, s = el('span', 'src');
    var txt, kind = 'mand';
    if (T.src === 'prices') {
      kind = 'data'; txt = 'data · the feed to Sep 4';
      if (st === 6 || st === 7) { kind = 'bad'; txt = 'APH ✕ split read as a loss'; }
      if (st >= 8) { kind = 'fixed'; txt = 'adjusted prices ✓'; }
    } else if (T.src === 'floor') {
      kind = 'data'; txt = 'mandate M5 + the feed';
      if (st === 6 || st === 7) { kind = 'warn'; txt = 'M5 + the same prices'; }
      if (st >= 8) { kind = 'fixed'; txt = 'M5 + adjusted prices ✓'; }
    } else if (T.src === 'profile') {
      kind = 'profile'; txt = 'your profile';
    } else {
      txt = 'mandate ' + T.src;
    }
    s.setAttribute('data-kind', kind); s.textContent = txt;
    return s;
  }
  function numCell(T) {
    var st = S.step, td = el('td', 't-num');
    if (T.num === 'floor') {
      td.appendChild(el('span', (st === 6 || st === 7) ? 'num-warn' : null, (st >= 8 ? FLOOR2 : FLOOR1) + '/mo'));
      if (st >= 8) td.appendChild(el('span', 't-was', 'was ' + FLOOR1));
    } else td.textContent = T.num;
    return td;
  }
  function renderTranslate() {
    var st = S.step, body = $('tr-body'); clear(body);
    $('tr-tag').hidden = st !== 8;
    var rq = $('tr-req'); clear(rq);
    rq.classList.remove('is-min');
    if (st === 1) rq.appendChild(el('span', 'tr-none', 'No message this week yet. Each Friday the proposer writes the week’s problem from your words and your mandate.'));
    else {
      rq.appendChild(el('span', 'tr-req-k', 'What the proposer reads'));
      rq.classList.toggle('is-min', st >= 4);
      var q = el('q', 'tr-q');
      R.msg.seen.forEach(function (p) {
        if (p[1] === 'pii') { var x = el('span', 'pii'); x.appendChild(icon('lock')); x.appendChild(el('span', null, p[0])); q.appendChild(x); }
        else if (p[1]) { var h = el('span', 'hl', p[0]); h.setAttribute('data-n', String(p[1])); if (st >= 3) h.appendChild(el('sup', null, String(p[1]))); q.appendChild(h); }
        else q.appendChild(document.createTextNode(p[0]));
      });
      rq.appendChild(q);
    }
    R.terms.forEach(function (T, idx) {
      if (T.from && st < 3) return;
      if (st === 1 && !T.mand) return;
      var tr = el('tr');
      if (st === 3) { tr.style.animationDelay = (idx * 0.35) + 's'; tr.className = 'reveal'; }
      if (T.src === 'prices' && (st === 6 || st === 7)) tr.setAttribute('data-tone', 'red');
      if (T.src === 'profile' && st === 6) tr.className = 'st-hide';   // stage step 6: the D1 table takes this room
      var said = T.phrase && st >= 2;
      var tn = el('td', 't-n'); var b = el('span', 'nb' + (said ? '' : ' nb--mand'), said ? T.n : '·'); if (said) b.setAttribute('data-n', T.n); tn.appendChild(b); tr.appendChild(tn);
      var tw = el('td', 't-w');
      if (said) { var ph = el('span', 'hl', T.phrase); ph.setAttribute('data-n', T.n); tw.appendChild(ph); }
      else if (T.mand) tw.appendChild(el('span', 't-mq', st >= 2 ? T.mand : 'your mandate: fully invested in your list, no stock above 10%, no sector above 30%, trade at most 20% a week, lowest risk with an expected return at least an equal-weight portfolio’s'));
      else tw.appendChild(el('span', 't-mq', '—'));
      tr.appendChild(tw);
      if (st >= 3) {
        tr.appendChild(el('td', 't-p', T.prob));
        tr.appendChild(numCell(T));
        var tf = el('td', 't-f'); tf.appendChild(srcChip(T)); tr.appendChild(tf);
      } else { var e = el('td', 't-empty', '—'); e.colSpan = 3; tr.appendChild(e); }
      body.appendChild(tr);
    });
    $('tr-foot').classList.toggle('st-hide', st === 6);
    $('tr-foot').textContent = st >= 3 ? 'Your words carry no numbers: each number has a source, and a source can be wrong. Code adds: long only, no borrowing.'
      : 'Your mandate, signed Aug 28 when the account opened. Each Friday the proposer writes the week’s problem from it and from your message.';
  }

  /* ------------------------------------------------------------------ chat */
  function clientBubble(m) {
    var wrap = el('div', 'msg msg--client');
    var meta = el('div', 'meta'); meta.appendChild(chip('chip--team chip--sm', 'Written by our team')); meta.appendChild(el('span', 'who', 'You')); meta.appendChild(el('span', 'when', m.when)); wrap.appendChild(meta);
    var b = el('div', 'bub');
    R.msg.parts.forEach(function (p) {
      if (!p[1]) { b.appendChild(document.createTextNode(p[0])); return; }
      b.appendChild(el('span', 'redact', p[0]));
      var lk = el('span', 'lock'); lk.appendChild(icon('lock')); lk.appendChild(el('span', null, p[1])); b.appendChild(lk);
    });
    wrap.appendChild(b);
    return wrap;
  }
  function proposerMsg(m) {
    var wrap = el('div', 'msg msg--proposer');
    var meta = el('div', 'meta'); meta.appendChild(el('span', 'who', 'Proposer · DeepSeek V4 Pro')); meta.appendChild(chip('chip--model chip--sm', 'Model · recorded')); meta.appendChild(el('span', 'when', m.when)); wrap.appendChild(meta);
    wrap.appendChild(el('div', 'bub', m.text));
    wrap.appendChild(el('div', 'seen', 'Numbers checked by code; wording reviewed by the critic before sending.'));
    return wrap;
  }
  function sysLine(m) {
    var d = el('div', 'msg msg--sys'); if (m.tone) d.setAttribute('data-tone', m.tone);
    d.appendChild(icon(m.icon)); d.appendChild(el('span', null, m.text));
    d.title = 'Code & solver';
    return d;
  }
  function visibleChat() {
    return CHAT.filter(function (m) { return m.step < S.step || (m.step === S.step && (!m.phase || m.phase === S.phase)); });
  }
  function renderChat(animateNew) {
    var host = $('thread'); clear(host);
    var list = visibleChat();
    list.forEach(function (m, i) {
      var node = m.kind === 'client' ? clientBubble(m) : m.kind === 'proposer' ? proposerMsg(m) : sysLine(m);
      if (i < list.length - 5 && m.kind !== 'client') node.className += ' old';
      if (!animateNew || m.step !== S.step) node.style.animation = 'none';
      host.appendChild(node);
    });
    scrollBottom(animateNew);
  }
  function scrollBottom(smooth) { var host = $('thread'); try { host.scrollTo({ top: host.scrollHeight, behavior: smooth ? 'smooth' : 'auto' }); } catch (e) { host.scrollTop = host.scrollHeight; } requestAnimationFrame(function () { host.scrollTop = host.scrollHeight; }); }
  function typeThenSend(text, done) {
    var inp = $('composer'), i = 0;
    clearInterval(S.typeTimer);
    inp.value = '';
    S.typeTimer = setInterval(function () {
      i += 2; inp.value = text.slice(0, i);
      inp.scrollLeft = inp.scrollWidth;
      if (i >= text.length) { clearInterval(S.typeTimer); S.typeTimer = null; S.sendTimer = setTimeout(function () { S.sendTimer = null; inp.value = ''; done(); }, 350); }
    }, 22);
  }

  /* ------------------------------------------------------------------ loop strip */
  function row(tone, k, text, kcls) {
    var d = el('div', 'lrow'); if (tone) d.setAttribute('data-tone', tone);
    if (k) d.appendChild(el('span', 'k ' + (kcls || ''), k));
    d.appendChild(el('span', null, text));
    return d;
  }
  function idle(text) { return el('div', 'lrow idle', text || '—'); }
  function loopState() {
    var st = S.step, ph = S.phase, L = { active: null, tone: {}, rows: {}, round: '', nothing: false, lane: null, under: null, ev: null, rebuilt: false };
    L.rows.client = [idle('no message yet')];
    L.rows.proposer = [idle()]; L.rows.checks = [idle()]; L.rows.critic = [idle()]; L.rows.gate = [idle()];
    if (st >= 2) L.rows.client = [row(st === 2 ? 'cyan' : null, null, 'message · ' + R.msg.when), row(null, 'CODE', R.msg.removed.length + ' personal details removed', 'k--code')];
    if (st === 2) L.active = 'client';
    if (st >= 3 && st <= 7) {
      L.rows.proposer = [row(null, 'LLM', '4 phrases → 4 terms; the other numbers from your mandate', 'k--model')];
      if (st >= 5) L.rows.proposer.push(row(null, 'GUROBI', 'solved on the feed · APH ' + APH_HELD.toFixed(2) + '% → ' + APH_R1.toFixed(2) + '%', 'k--code'));
      if (st === 3 || st === 5) L.active = 'proposer';
    }
    if (st >= 8) {
      L.rows.proposer = [row(null, 'LLM', 'round 2: same terms, estimates on adjusted prices', 'k--model'), row(null, 'GUROBI', 'solved · APH stays ' + APH_HELD.toFixed(2) + '% · risk ' + pc(W2.risk_monthly), 'k--code')];
    }
    if (st === 4) { L.ev = 'chart'; L.under = 'A split is not a loss: each share became two. The feed has no adjustment step, so its history now shows a one-day crash.'; L.rebuilt = true; }
    if (st === 6 || st === 7) {
      L.rows.checks = [row('red', 'D1', '✕ 3 days disagree with reference prices'), row('green', 'D2', '✓ each matches a corporate action on record'), row(null, null, N1.pass + ' of ' + N1.n + ' pass')];
      L.tone.checks = 'red'; L.hash = R.cert.r1 + '…';
      L.rows.gate = [row('amber', null, 'Back to the proposer · round 2')]; L.tone.gate = 'amber'; L.nothing = true;
    }
    if (st === 6) {
      L.active = 'checks'; L.ev = 'd1';
    }
    if (st === 7) {
      L.rows.critic = [row('red', 'LLM', 'Firm policy bars the feed’s estimates: bind the adjusted ones; floor ' + FLOOR2 + '.', 'k--model')];
      L.tone.critic = 'red'; L.active = 'critic'; L.verdict = 'obj'; L.lane = 'return';
      L.under = 'The critic reads code’s results, so it explains and ties the evidence to your plan and your floor. It did not find the split by itself.';
    }
    if (st >= 8) {
      L.rows.checks = [row('green', null, N2.pass + ' of ' + N2.n + ' pass')]; L.tone.checks = 'green'; L.hash = R.cert.r2 + '…';
      L.rows.critic = [row('green', 'LLM', 'HON exceptions immaterial; every clause matches the mandate.', 'k--model')];
      L.tone.critic = 'green'; L.verdict = 'pass';
      L.rows.gate = [row('green', null, 'Trade · ' + R.fills.n_orders + ' orders at Tuesday’s open (simulated)')];
      if (st === 9 && ph === 'b') L.rows.gate.push(row('green', null, 'After-fill checks: 4 of 4 pass · cash left ' + R.fills.cash_after_pct.toFixed(2) + '%'));
      L.tone.gate = 'green'; L.active = st === 8 ? 'critic' : 'gate';
      L.ev = 'compare';
    }
    if (st >= 5 && st <= 7) L.round = 'round 1';
    if (st >= 8) L.round = 'round 2';
    return L;
  }
  var BOXES = ['client', 'proposer', 'checks', 'critic', 'gate'];
  function renderLoop() {
    var L = loopState();
    BOXES.forEach(function (b, i) {
      var box = $('b-' + b), rows = box.querySelector('.lrows'); clear(rows);
      L.rows[b].forEach(function (r) { rows.appendChild(r); });
      box.classList.toggle('is-active', L.active === b);
      if (L.tone[b]) box.setAttribute('data-tone', L.tone[b]); else box.removeAttribute('data-tone');
      var foot = box.querySelector('.lbox-foot');
      var old = foot.querySelector('.dyn'); if (old) foot.removeChild(old);
      if (b === 'checks' && L.hash) foot.appendChild(chip('chip--mono chip--sm dyn', 'certificate ' + L.hash));
      var t = box.querySelector('.lbox-t'); var oldv = t.querySelector('.verdict'); if (oldv) t.removeChild(oldv);
      if (b === 'critic' && L.verdict) t.appendChild(el('span', 'verdict ' + (L.verdict === 'obj' ? 'verdict--obj' : 'verdict--pass'), L.verdict === 'obj' ? 'OBJECTION' : 'PASS'));
      if (i > 0) { var ar = $('a' + i); ar.classList.toggle('hot', L.active === b && !!L.active); ar.classList.toggle('green', b === 'gate' && S.step >= 8); }
    });
    $('lp-round').textContent = L.round;
    $('lp-nothing').hidden = !L.nothing;
    var u = $('lp-under'); clear(u);
    if (L.under) {
      u.hidden = false; u.appendChild(chip('chip--team chip--sm', 'Written by our team')); u.appendChild(el('span', null, L.under));
      if (L.rebuilt) u.appendChild(chip('chip--plant chip--sm', 'Feed rebuilt from Yahoo'));
    } else u.hidden = true;
    renderEvidence(L.ev);
    drawLanes(L.lane);
  }
  function renderEvidence(kind) {
    var host = $('lp-ev'); clear(host);
    host.hidden = !kind;
    if (kind === 'chart') host.appendChild(aphChart(Math.max(600, host.clientWidth - 26), 96));
    if (kind === 'd1') host.appendChild(d1Table());
    if (kind === 'compare') host.appendChild(compareTable());
  }
  function aphChart(W, H) {
    var A = R.aph, wrap = el('div', 'ev-chart');
    var head = el('div', 'ev-h');
    head.appendChild(el('span', 'ev-t', 'APH’s price over the 252 days the estimates use'));
    var lg = el('span', 'ev-lg');
    lg.appendChild(el('span', 'lg lg--fed', 'the feed, as delivered')); lg.appendChild(el('span', 'lg lg--src', 'adjusted for the split'));
    head.appendChild(lg); head.appendChild(chip('chip--code chip--sm', 'Code & solver'));
    wrap.appendChild(head);
    var padL = 40, padR = 8, padT = 8, padB = 16, n = A.dates.length;
    var all = A.fed.concat(A.adj), ymin = Math.floor(Math.min.apply(null, all) / 25) * 25, ymax = Math.ceil(Math.max.apply(null, all) / 25) * 25;
    var x = function (i) { return padL + i * (W - padL - padR) / (n - 1); };
    var y = function (v) { return padT + (ymax - v) / (ymax - ymin) * (H - padT - padB); };
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'ko', role: 'img', 'aria-label': 'APH prices: the feed as delivered halves on Sep 3; adjusted for the split, the price rises' });
    [50, 100, 150].filter(function (v) { return v > ymin && v < ymax; }).forEach(function (v) {
      svg.appendChild(svgEl('line', { x1: padL, x2: W - padR, y1: y(v), y2: y(v), class: 'grid' }));
      var t = svgEl('text', { x: padL - 6, y: y(v) + 4, 'text-anchor': 'end', class: 'ax' }); t.textContent = '$' + v; svg.appendChild(t);
    });
    var pts = function (arr) { return arr.map(function (v, i) { return x(i).toFixed(1) + ',' + y(v).toFixed(1); }).join(' '); };
    svg.appendChild(svgEl('polyline', { points: pts(A.adj), class: 'src' }));
    svg.appendChild(svgEl('polyline', { points: pts(A.fed), class: 'fed' }));
    var k = A.dates.indexOf(A.split);
    svg.appendChild(svgEl('circle', { cx: x(k), cy: y(A.fed[k]), r: 4, class: 'cliff' }));
    [[0, 'Sep 2025'], [n - 1, 'Sep 4, 2026']].forEach(function (p, j) {
      var t = svgEl('text', { x: x(p[0]), y: H - 3, 'text-anchor': j === 0 ? 'start' : 'end', class: 'ax' }); t.textContent = p[1]; svg.appendChild(t);
    });
    wrap.appendChild(svg);
    wrap.appendChild(el('p', 'ev-note ev-note--red', 'Thu Sep 3: ' + sg(A.r_fed) + ' in the feed, ' + sg(A.r_adj) + ' adjusted. Yahoo’s record: a 2-for-1 split that day (announced Aug 6).'));
    return wrap;
  }
  function d1Table() {
    var wrap = el('div', 'ev-cmp');
    var head = el('div', 'ev-h'); head.appendChild(el('span', 'ev-t', 'Price check D1: the days the feed disagrees with reference prices')); head.appendChild(chip('chip--code chip--sm', 'Code & solver')); wrap.appendChild(head);
    var t = el('table', 'cmp'); var th = el('thead'); var trh = el('tr');
    ['Stock and day', 'In the feed', 'Reference', 'On record (D2)'].forEach(function (h) { trh.appendChild(el('th', null, h)); }); th.appendChild(trh); t.appendChild(th);
    var tb = el('tbody');
    var rec = {}; R.d2.forEach(function (x) { rec[x.ticker + x.date] = x; });
    R.d1.slice().sort(function (p, q) { return q.gap_pp - p.gap_pp; }).forEach(function (r) {
      var tr = el('tr'); tr.setAttribute('data-r', r.ticker === 'APH' ? '1' : '');
      var x = rec[r.ticker + r.date];
      tr.appendChild(el('td', null, r.ticker + ' · ' + r.date));
      tr.appendChild(el('td', 'n', sg(r.return_pct)));
      tr.appendChild(el('td', 'n', sg(r.reference_pct)));
      tr.appendChild(el('td', 'n', x ? (x.kind === 'split' ? x.ratio + ' split' : x.ratio + ' spin-off') : 'none'));
      tb.appendChild(tr);
    });
    t.appendChild(tb); wrap.appendChild(t);
    wrap.appendChild(el('p', 'ev-note', 'Written by our team: the reference prices come from the same vendor as the rebuilt feed, so here D1 cannot miss. With APH’s split removed from the record, a control run held after 3 rounds and nothing traded.'));
    return wrap;
  }
  function compareTable() {
    var C = R.compare, wrap = el('div', 'ev-cmp');
    var head = el('div', 'ev-h'); head.appendChild(el('span', 'ev-t', 'What the split did to round 1, and what round 2 does')); head.appendChild(chip('chip--code chip--sm', 'Code & solver')); wrap.appendChild(head);
    var t = el('table', 'cmp'); var th = el('thead'); var trh = el('tr');
    C.head.forEach(function (h) { trh.appendChild(el('th', null, h)); }); th.appendChild(trh); t.appendChild(th);
    var tb = el('tbody');
    C.rows.forEach(function (r, i) {
      var tr = el('tr'); tr.setAttribute('data-r', String(i));
      r.forEach(function (c, j) { tr.appendChild(el('td', j ? 'n' : null, c)); });
      tb.appendChild(tr);
    });
    t.appendChild(tb); wrap.appendChild(t);
    wrap.appendChild(el('p', 'ev-note', C.note));
    return wrap;
  }
  function drawLanes(kind) {
    var svg = $('lanes'); clear(svg);
    var wrap = svg.parentNode; wrap.style.paddingBottom = kind === 'return' ? '24px' : '0';
    if (!kind) return;
    var base = wrap.getBoundingClientRect();
    function box(id) { var r = $(id).getBoundingClientRect(); return { x: r.left - base.left, y: r.top - base.top, w: r.width, h: r.height }; }
    svg.setAttribute('viewBox', '0 0 ' + base.width + ' ' + base.height);
    var c = box('b-critic'), p = box('b-proposer'), yy = Math.max(c.y + c.h, p.y + p.h) + 14;
    var x0 = c.x + c.w / 2, x1 = p.x + p.w / 2;
    svg.appendChild(svgEl('path', { class: 'ret', d: 'M' + x0 + ',' + (c.y + c.h) + ' V' + yy + ' H' + x1 + ' V' + (p.y + p.h + 7) }));
    svg.appendChild(svgEl('path', { class: 'arrow-ret', d: 'M' + (x1 - 5) + ',' + (p.y + p.h + 8) + ' L' + x1 + ',' + (p.y + p.h + 1) + ' L' + (x1 + 5) + ',' + (p.y + p.h + 8) + ' Z' }));
    var t = svgEl('text', { class: 'ret-t', x: (x0 + x1) / 2, y: yy - 5, 'text-anchor': 'middle' }); t.textContent = '↩ code and critic objections · round 2'; svg.appendChild(t);
  }

  /* ------------------------------------------------------------------ explanation */
  function renderExplain() {
    var st = S.step, E = STEPS[st - 1];
    $('ex-title').textContent = E.title;
    $('ex-line').textContent = E.line;
    $('ex-next').textContent = 'next → ' + E.next;
    each('[data-count]', function (c) { c.textContent = 'step ' + st + ' / ' + LAST; });
    $('live').textContent = 'Step ' + st + ': ' + E.title + '. ' + E.line;
    each('[data-act="prev"]', function (b) { b.disabled = st === 1; });
    each('[data-act="next"]', function (b) { b.disabled = st === LAST && S.phase === 'b'; });
  }

  /* ------------------------------------------------------------------ details */
  function checksTable(rows) {
    var g = el('div', 'ck');
    rows.forEach(function (r) {
      var s = el('div', 'ck-s', r[1] === 'ok' ? '✓' : r[1] === 'bad' ? '✕' : '●'); s.setAttribute('data-s', r[1]); g.appendChild(s);
      var id = el('div', 'ck-id', r[0]); id.title = r[2]; g.appendChild(id);
      var tx = el('div'); tx.appendChild(el('b', null, r[2] + ' · ')); tx.appendChild(document.createTextNode(r[3])); g.appendChild(tx);
    });
    return g;
  }
  function rawBlock(text) {
    var wrap = el('div');
    var pre = el('pre', 'raw', text); wrap.appendChild(pre);
    if (text.split('\n').length > 6 || text.length > 520) {
      var b = el('button', 'btn btn--ghost raw-more', 'show all'); b.type = 'button';
      b.addEventListener('click', function () { var o = pre.classList.toggle('open'); b.textContent = o ? 'show less' : 'show all'; });
      wrap.appendChild(b);
    } else pre.classList.add('open');
    return wrap;
  }
  /* a JSON document as a collapsible tree: short values inline, the rest folded except along marked paths */
  function jsonTree(doc, marks) {
    var root = el('div', 'cj');
    function lit(v, parent) {
      if (v === null || typeof v === 'boolean') parent.appendChild(el('span', 'j-l', String(v)));
      else if (typeof v === 'number') parent.appendChild(el('span', 'j-n', String(v)));
      else if (typeof v === 'string') parent.appendChild(el('span', 'j-s', JSON.stringify(v)));
      else if (Array.isArray(v)) { parent.appendChild(document.createTextNode('[')); v.forEach(function (x, i) { if (i) parent.appendChild(document.createTextNode(', ')); lit(x, parent); }); parent.appendChild(document.createTextNode(']')); }
      else { parent.appendChild(document.createTextNode('{ ')); Object.keys(v).forEach(function (k, i) { if (i) parent.appendChild(document.createTextNode(', ')); parent.appendChild(el('span', 'j-k', JSON.stringify(k))); parent.appendChild(document.createTextNode(': ')); lit(v[k], parent); }); parent.appendChild(document.createTextNode(' }')); }
    }
    function marked(path) { return marks[path]; }
    function onPath(path) { return Object.keys(marks).some(function (m) { return m.indexOf(path + '.') === 0; }); }
    function keyed(parent, key) { if (key !== null) { parent.appendChild(el('span', 'j-k', JSON.stringify(key))); parent.appendChild(document.createTextNode(': ')); } }
    function node(key, v, path, last, host, inh) {
      var comma = last ? '' : ',', isObj = v && typeof v === 'object', tone = marked(path) || inh;
      if (!isObj || JSON.stringify(v).length <= 72) {
        var r = el('div', 'j-row' + (tone ? ' j-hl--' + tone : '')); keyed(r, key); lit(v, r); r.appendChild(document.createTextNode(comma)); host.appendChild(r); return;
      }
      var arr = Array.isArray(v), keys = arr ? v.map(function (_, i) { return i; }) : Object.keys(v);
      var d = el('details'); d.open = path === '' || onPath(path) || !!tone;
      var s = el('summary', 'j-row' + (tone ? ' j-hl--' + tone : '')); keyed(s, key); s.appendChild(document.createTextNode(arr ? '[' : '{'));
      s.appendChild(el('span', 'j-hint', ' ' + keys.length + (arr ? (keys.length === 1 ? ' item ' : ' items ') : ' keys ') + (arr ? ']' : '}') + comma)); d.appendChild(s);
      var flat = !arr && keys.length > 8 && keys.every(function (k) { return typeof v[k] === 'number'; });
      var inner = el('div', flat ? 'j-flow' : 'j-in');
      keys.forEach(function (k, i) {
        var p = path ? path + (arr ? '[' + k + ']' : '.' + k) : String(k);
        if (flat) { var t = marked(p) || tone, e = el('span', 'j-ent' + (t ? ' j-hl--' + t : '')); keyed(e, k); lit(v[k], e); if (i < keys.length - 1) e.appendChild(document.createTextNode(',')); inner.appendChild(e); }
        else node(arr ? null : k, v[k], p, i === keys.length - 1, inner, tone);
      });
      d.appendChild(inner); d.appendChild(el('div', 'j-row j-close' + (tone ? ' j-hl--' + tone : ''), (arr ? ']' : '}') + comma));
      host.appendChild(d);
    }
    node(null, doc, '', true, root);
    return root;
  }
  function sec(title, chipEl) { var s = el('div', 'dt-sec'); var h = el('h3', null, title); if (chipEl) h.appendChild(chipEl); s.appendChild(h); return s; }
  function countLine(r, n) { return 'Round ' + r + ' · ' + n.n + ' code checks · ' + n.pass + ' pass' + (n.fail.length ? ' · ' + n.fail.join(', ') + ' failed' : '') + ' · certificate ' + (r === 1 ? R.cert.r1 : R.cert.r2) + '…'; }
  function renderDetails() {
    var host = $('dt-body'); clear(host);
    var st = S.step;
    if (!S.tabPinned) S.tab = st <= 4 || st === 8 ? 'proposer' : st === 5 ? 'solver' : st === 7 ? 'critic' : 'certificate';
    if (st < 8) S.certRound = 0;
    var tab = S.tab;
    $('details').querySelectorAll('.tabs button').forEach(function (b) { var on = b.getAttribute('data-tab') === tab; b.classList.toggle('is-active', on); b.setAttribute('aria-selected', on ? 'true' : 'false'); });
    if (tab === 'proposer') {
      var hp = el('p', 'ck-title'); hp.appendChild(document.createTextNode('DeepSeek V4 Pro · ')); hp.appendChild(chip('chip--model chip--sm', 'Model · recorded')); hp.appendChild(document.createTextNode(' the problem as the API returned it')); host.appendChild(hp);
      if (st < 3) host.appendChild(el('p', 'empty', 'The proposer reads your message, with the personal details removed, at step 3.'));
      else {
        var p1 = sec('Round 1 · binds the feed as delivered'); p1.appendChild(rawBlock(R.proposer.r1)); host.appendChild(p1);
        if (st >= 8) { var p2 = sec('Round 2 · binds the adjusted prices'); p2.appendChild(rawBlock(R.proposer.r2)); host.appendChild(p2); }
      }
    }
    if (tab === 'checks') {
      if (st <= 5) host.appendChild(el('p', 'empty', 'Code checks run on the proposer’s problem, the data it binds and the solver’s plan. They run next, at step 6.'));
      else {
        var r = st >= 8 ? 2 : 1;
        host.appendChild(el('p', 'ck-title', countLine(r, r === 1 ? N1 : N2)));
        host.appendChild(checksTable(r === 1 ? R.checks.r1 : R.checks.r2));
        if (st === 9 && S.phase === 'b') { host.appendChild(el('p', 'ck-title', 'After the fill · Tue Sep 8 open')); host.appendChild(checksTable(R.checks.fill)); }
        var c2 = sec('What the data checks found', chip('chip--code chip--sm', 'Code & solver'));
        var kv = el('div', 'kv');
        R.d1.forEach(function (r) { kv.appendChild(el('div', 'k', r.ticker + ' ' + r.date)); kv.appendChild(el('div', 'v', 'feed ' + r.return_pct.toFixed(2) + '% · reference ' + r.reference_pct.toFixed(2) + '% · gap ' + r.gap_pp.toFixed(2) + ' points')); });
        kv.appendChild(el('div', 'k', 'Quarantine')); kv.appendChild(el('div', 'v', R.offered));
        R.hon.forEach(function (x) { kv.appendChild(el('div', 'k', 'HON ' + x.ratio)); kv.appendChild(el('div', 'v', x.plain)); });
        c2.appendChild(kv); c2.style.marginTop = '14px'; host.appendChild(c2);
      }
      host.appendChild(el('p', 'dt-foot', 'Each line is the text our code wrote in the recorded run. ● P4 passes with a caveat: the proposer’s training-data end is bounded by its release date, not published.'));
    }
    if (tab === 'certificate') {
      if (st < 6) host.appendChild(el('p', 'empty', 'Code issues a certificate for each round once the proposer’s problem is solved. The first appears at step 6.'));
      else {
        var cr = S.certRound && st >= 8 ? S.certRound : st >= 8 ? 2 : 1, key = 'r' + cr, doc = R.cert_doc[key];
        var bar = el('div', 'cj-bar');
        var ht = el('p', 'ck-title', 'Round ' + cr + ' · certificate ' + R.cert[key + '_full'].slice(0, 16) + '… '); ht.appendChild(chip('chip--code chip--sm', 'Code & solver')); ht.appendChild(document.createTextNode(' the file as recorded')); bar.appendChild(ht);
        bar.appendChild(el('span', 'spacer'));
        if (st >= 8) {
          var sw = el('div', 'seg'); sw.setAttribute('role', 'group'); sw.setAttribute('aria-label', 'Certificate round');
          [1, 2].forEach(function (n) { var b = el('button', n === cr ? 'is-active' : null, 'round ' + n + ' · ' + R.cert['r' + n]); b.type = 'button'; b.addEventListener('click', function () { S.certRound = n; renderDetails(); }); sw.appendChild(b); });
          bar.appendChild(sw);
        }
        var ex = el('button', 'btn btn--ghost raw-more', 'expand all'); ex.type = 'button'; ex.style.margin = '0'; bar.appendChild(ex);
        host.appendChild(bar);
        var tone = cr === 1 ? 'red' : 'green';
        var tree = jsonTree(doc, { 'data.prices': tone, 'problem.compiled.clauses.M5.r_star': tone, 'outcome.weights_pct.APH': tone });
        ex.addEventListener('click', function () { var all = tree.querySelectorAll('details'), open = ex.textContent === 'expand all'; all.forEach(function (d) { d.open = open; }); ex.textContent = open ? 'collapse' : 'expand all'; });
        host.appendChild(tree);
        host.appendChild(el('p', 'dt-foot', (cr === 1
          ? 'Marked: the prices this round binds (the feed as delivered), the return floor computed from them (1.39% a month) and APH’s planned weight. '
          : 'Marked: the adjusted prices round 2 binds, the return floor on them (1.53% a month) and APH’s weight, unchanged. ') +
          'The file is named by its SHA-256 hash, which code computes over the whole certificate; building this page recomputed both hashes and they match. “parent” is null: this decision does not follow from an earlier certificate.'));
      }
    }
    if (tab === 'critic') {
      var h = el('p', 'ck-title'); h.appendChild(document.createTextNode('Claude Sonnet 4.6 · ')); h.appendChild(chip('chip--model chip--sm', 'Model · recorded')); h.appendChild(document.createTextNode(' the reply as the API returned it')); host.appendChild(h);
      if (st < 7) host.appendChild(el('p', 'empty', 'The critic reads the certificate, the solver log and the code checks. It answers at step 7.'));
      else {
        var s1 = sec('Round 1 · objection'); s1.appendChild(rawBlock(R.critic.r1)); host.appendChild(s1);
        if (st >= 8) { var s2 = sec('Round 2 · pass'); s2.appendChild(rawBlock(R.critic.r2)); host.appendChild(s2); }
        if (st === 9) {
          var s3 = sec('The message to the client · checked by code, reviewed by the critic');
          R.summary.forEach(function (x) {
            if (x.who === 'proposer') s3.appendChild(el('p', 'sm-line', 'Proposer (' + x.purpose + '): “' + x.text + '”'));
            else if (x.who === 'code') s3.appendChild(el('p', 'sm-line', 'Code S1: ' + x.plain));
            else s3.appendChild(el('p', 'sm-line', 'Critic: ' + (x.verdict === 'pass' ? 'pass.' : 'objects: ' + x.objections.map(function (o) { return o.reason + ' Change: ' + o.requested_change; }).join(' '))));
          });
          host.appendChild(s3);
        }
      }
    }
    if (tab === 'solver') {
      if (st < 5) host.appendChild(el('p', 'empty', 'Gurobi runs after the proposer writes the problem, at step 5.'));
      else {
        var c1 = sec(st >= 8 ? 'Round 1 against round 2' : 'Round 1 · solved on the feed as delivered', chip('chip--code chip--sm', 'Code & solver'));
        var t = el('table', 'tbl'); var th = el('thead'); var trh = el('tr'); R.compare.head.forEach(function (c) { trh.appendChild(el('th', null, c.toUpperCase())); }); th.appendChild(trh); t.appendChild(th);
        var tb = el('tbody'); R.compare.rows.forEach(function (rr, i) { if (i === 2 && st < 8) return; var tr = el('tr'); rr.forEach(function (c, j) { tr.appendChild(el('td', j ? 'n' : null, c)); }); tb.appendChild(tr); }); t.appendChild(tb);
        c1.appendChild(t); c1.appendChild(el('p', 'eq', R.compare.note + ' Risk = monthly volatility of the planned weights, √(wᵀΣw), Σ from 252 daily returns (Ledoit–Wolf), ×21.')); host.appendChild(c1);
      }
    }
  }

  /* ------------------------------------------------------------------ about: run tables from the recording */
  function renderAbout() {
    var host = $('ab-runs'); if (!host || host.getAttribute('data-done')) return;
    host.setAttribute('data-done', '1');
    var t = el('table', 'tbl'); var th = el('thead'); var trh = el('tr');
    ['RUNS', 'FEEDBACK', 'ROUNDS', 'DECISION', 'ORDERS', 'APH TRADED', 'MESSAGE TO CLIENT'].forEach(function (c) { trh.appendChild(el('th', null, c)); }); th.appendChild(trh); t.appendChild(th);
    var tb = el('tbody');
    function agg(list, label, arm) {
      var tr = el('tr');
      var same = function (f) { var v = list.map(f); return v.every(function (x) { return x === v[0]; }) ? String(v[0]) : v.join(', '); };
      [label, arm, same(function (r) { return r.rounds; }), same(function (r) { return r.final; }), same(function (r) { return r.orders; }),
        same(function (r) { return r.aph_traded ? 'yes' : 'no'; }), list.every(function (r) { return r.message === 'no trade'; }) ? 'no trade' : list.filter(function (r) { return r.message === 'sent'; }).length + ' of ' + list.length + ' sent']
        .forEach(function (c, j) { tr.appendChild(el('td', j > 1 ? 'n' : null, c)); });
      tb.appendChild(tr);
    }
    agg(R.runs.final, '5 recorded runs (this page: run 1)', 'code + critic');
    agg(R.runs.code_only, '5 runs, critic switched off', 'code only');
    agg(R.runs.control, '1 control: APH split removed from the record', 'code + critic');
    t.appendChild(tb); host.appendChild(t);
  }

  /* ------------------------------------------------------------------ navigation */
  function render(opts) {
    opts = opts || {};
    renderPortfolio(); renderTranslate(); renderLoop(); renderExplain(); renderDetails();
    renderChat(!!opts.animate);
    $('about').hidden = !S.about;
    if (S.about) renderAbout();
  }
  function goto(step, phase, opts) {
    opts = opts || {};
    clearTimeout(S.phaseTimer); S.phaseTimer = null; clearInterval(S.typeTimer); S.typeTimer = null; $('composer').value = '';
    var forward = step > S.step;
    S.step = step; S.phase = phase || 'a';
    var clientMsg = CHAT.filter(function (m) { return m.step === step && m.kind === 'client'; })[0];
    if (opts.live && forward && clientMsg) {
      S.step = step - 1; render({ animate: false }); S.step = step;
      renderExplain();
      typeThenSend(R.msg.parts.map(function (p) { return p[0]; }).join(''), function () { render({ animate: true }); });
    } else render({ animate: !!opts.live });
    if (step === LAST && S.phase === 'a' && !opts.hold) S.phaseTimer = setTimeout(function () { S.phaseTimer = null; S.phase = 'b'; render({ animate: true }); }, 1600);
  }
  function next() { if (S.step < LAST) goto(S.step + 1, 'a', { live: true }); else if (S.phase === 'a') goto(LAST, 'b'); }
  function prev() { if (S.step > 1) goto(S.step - 1, 'a'); }
  function setPlaying(on) {
    S.playing = on; clearInterval(S.timer);
    each('[data-act="play"]', function (b) { b.textContent = on ? '⏸ Pause' : '▶ Play'; });
    if (on) S.timer = setInterval(function () { if (S.step >= LAST) { setPlaying(false); return; } next(); }, 9000);
  }
  /* ------------------------------------------------------------------ wiring */
  function each(sel, fn) { Array.prototype.forEach.call(shadow.querySelectorAll(sel), fn); }
  function openTab(t) {
    S.tab = t; S.tabPinned = true; render();
    $('details').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  function bindElements() {
    each('[data-act="next"]', function (b) { b.addEventListener('click', function () { setPlaying(false); next(); }); });
    each('[data-act="prev"]', function (b) { b.addEventListener('click', function () { setPlaying(false); prev(); }); });
    each('[data-act="play"]', function (b) { b.addEventListener('click', function () { if (!S.playing && S.step >= LAST) goto(1, 'a'); setPlaying(!S.playing); }); });
    [['proposer', 'proposer'], ['checks', 'certificate'], ['critic', 'critic']].forEach(function (p) {
      var box = $('b-' + p[0]);
      box.setAttribute('data-tab', p[1]); box.setAttribute('role', 'button'); box.tabIndex = 0;
      box.title = 'Open ' + p[1].charAt(0).toUpperCase() + p[1].slice(1) + ' in Details';
      box.addEventListener('click', function () { openTab(p[1]); });
      box.addEventListener('keydown', function (e) { if (e.key === 'Enter') { e.preventDefault(); openTab(p[1]); } });
    });
    $('aboutBtn').addEventListener('click', function () { S.about = true; render(); });
    $('aboutClose').addEventListener('click', function () { S.about = false; render(); });
    $('about').addEventListener('click', function (e) { if (e.target === $('about')) { S.about = false; render(); } });
    $('details').querySelectorAll('.tabs button').forEach(function (b) { b.addEventListener('click', function () { S.tab = b.getAttribute('data-tab'); S.tabPinned = true; render(); }); });
  }

  /* ------------------------------------------------------------------ mount (ATL's Portfolio Management subtab) */
  // Which keys the page takes. Null leaves the key to the browser and to ATL: a form control,
  // a modifier, focus outside the panel (other than on ATL's own tab buttons, where a click on
  // this tab leaves it), Space on a focused button or link, or a hidden panel.
  function keyAction(e, host, activeEl) {
    if (!host || !host.offsetParent) return null;
    if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return null;
    var path = e.composedPath ? e.composedPath() : [];
    var origin = path[0] || e.target;
    var tag = origin && origin.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (origin && origin.isContentEditable)) return null;
    if (activeEl && activeEl !== document.body && activeEl !== document.documentElement && activeEl !== host
      && !(host.contains && host.contains(activeEl))
      && !(activeEl.closest && activeEl.closest('.subtab-btn, .mode-btn, [data-playground-tab]'))) return null;
    if (e.key === ' ') return tag === 'BUTTON' || tag === 'SUMMARY' || tag === 'A' ? null : 'toggle';
    if (e.key === 'ArrowRight' || e.key === 'PageDown') return 'next';
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') return 'prev';
    if (e.key === 'Escape') return 'escape';
    return null;
  }
  function onKeydown(e) {
    if (!panel || !panel.offsetParent) return;
    var action = keyAction(e, panel, document.activeElement);
    if (!action) return;
    if (action === 'next') { e.preventDefault(); setPlaying(false); next(); }
    else if (action === 'prev') { e.preventDefault(); setPlaying(false); prev(); }
    else if (action === 'toggle') { e.preventDefault(); setPlaying(!S.playing); }
    else { S.about = false; render(); }
  }
  function redraw() { if (active) { scrollBottom(false); drawLanes(loopState().lane); } }
  function onResize() { drawLanes(loopState().lane); }
  function onWindowLoad() { redraw(); }

  // First call builds the page in a shadow root on the panel; every call while hidden takes the
  // document keys and window events back and redraws once laid out.
  function show(target) {
    if (!target) return;
    if (!shadow) {
      panel = target;
      shadow = panel.attachShadow({ mode: 'open' });
      var style = document.createElement('style');
      style.textContent = CSS;
      shadow.appendChild(style);
      var tpl = document.createElement('template');
      tpl.innerHTML = MARKUP;
      shadow.appendChild(tpl.content);
      bindElements();
      // Optional rehearsal link, read once and never written: #pm-step-5, #pm-step-9a, #pm-step-9b.
      var m = /^#pm-step-([1-9])([ab])?$/.exec(location.hash || '');
      if (m) goto(+m[1], m[2] || 'a', { hold: !!m[2] }); else goto(1, 'a');
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(redraw);
    }
    if (active) return;
    active = true;
    document.addEventListener('keydown', onKeydown);
    window.addEventListener('resize', onResize);
    if (document.readyState !== 'complete') window.addEventListener('load', onWindowLoad);
    requestAnimationFrame(redraw);
  }
  // Leaving the tab pauses the page: autoplay stops, a half-typed message or the step-9 pause
  // settles, and the document keys and window events go back to ATL.
  function hide() {
    if (!active) return;
    active = false;
    setPlaying(false);
    if (S.typeTimer || S.sendTimer) {
      clearInterval(S.typeTimer); clearTimeout(S.sendTimer);
      S.typeTimer = null; S.sendTimer = null;
      $('composer').value = '';
      render();
    }
    if (S.phaseTimer) { clearTimeout(S.phaseTimer); S.phaseTimer = null; S.phase = 'b'; render(); }
    document.removeEventListener('keydown', onKeydown);
    window.removeEventListener('resize', onResize);
    window.removeEventListener('load', onWindowLoad);
  }

  window.PortfolioManagementDemo = { show: show, hide: hide, keyAction: keyAction };
  // A reload or direct link onto this tab: app.js navigates only after its sign-in requests, so
  // paint now rather than leave the panel empty until then.
  var root = document.documentElement;
  if (root && root.getAttribute('data-nav-page') === 'playground'
    && root.getAttribute('data-nav-playground-tab') === 'portfolio-management') {
    show(document.getElementById('playgroundPortfolioManagementPanel'));
  }
})();
