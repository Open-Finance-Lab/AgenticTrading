"""Does a trading instruction actually change what the model does?

The participatory-competition design assumes instruction quality maps to return
on a pinned model. Nobody has measured it. This runs deliberately opposite
instructions through the leaderboard harness and prints the spread.

    # free: check wiring, spend nothing
    python dashboard/scripts/probe_instruction_sensitivity.py --list-instructions

    # ~$0.43: the cheap half, and decisive on its own if it passes
    python dashboard/scripts/probe_instruction_sensitivity.py \
        --models nemotron --initial-capital 100000

    # ~$4.54: only needed if nemotron comes back flat (the contingency branch)
    python dashboard/scripts/probe_instruction_sensitivity.py \
        --models deepseek --initial-capital 100000

`--initial-capital 100000` is not optional in practice. leaderboard.json currently
carries `initial_capital: 10000`, which the resolution guard below refuses (one
median DJIA share is 2.49% of equity, above the 1% ceiling) — so without the flag
these commands exit EXIT_CONFIG before spending anything. $100k is also the base
every published board curve was computed at, which is what makes a probe return
comparable with one. The flag stays explicit rather than defaulting to 100000
because the config value is the thing under suspicion; see the capital-mismatch
section of the write-up.

Cost is read off the seven published contest runs, which recorded $0.0715
(nemotron) and $0.756 (deepseek) per 160-step run over this exact window, so
six instructions cost ~$0.43 and ~$4.54 respectively. Running both is the
plan's ~$4.97 — but the gate table only needs deepseek when nemotron fails, so
`--models` defaults to nemotron alone.

Treat those as an UPPER bound, not a forecast. They extrapolate from a
default-prompt board run, and this script exercises the custom-prompt path,
which bills differently: the 2026-08-16 deepseek leg came in at $0.4556/run
against a $0.70-0.80 prediction, after an earlier prediction erred the other
way. Deliberately not lowered — an estimate that under-shoots is the one that
overspends. Read per-run cost off a measured run; see
docs/superpowers/probe-results/2026-08-09-instruction-sensitivity.md.

GATE: if the spread across these instructions is under ~1pp, the instruction
axis does not exist and Phase 2 must not be built. See the gate table in
docs/superpowers/plans/2026-08-09-participatory-competition-phase-0-1.md.

WHY A FALLBACK RUN INVALIDATES EVERYTHING: without a usable API key, every run
silently falls back to rule-based trading and produces the *same* curve. That
is a spread of 0.00pp — indistinguishable from a real "the model ignores the
instruction" result, and it would kill the design over a billing problem. Any
run that did not clear the H6 coverage threshold makes the verdict
INCONCLUSIVE (exit 3), never FAIL.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from pathlib import Path

from dotenv import load_dotenv

DASHBOARD_DIR = Path(__file__).resolve().parent.parent

# Bootstrap for direct-file execution (``python dashboard/scripts/probe_...py``).
# When imported as ``dashboard.scripts.probe_instruction_sensitivity`` — which the
# gate-logic tests do, so the verdict that decides whether Phase 2 gets built is
# not left untested — the repo root is already importable and __package__ is
# truthy, so this is skipped. Same pattern as backfill_runs_to_postgres.py:125.
if not __package__:
    from _bootstrap import ensure_repo_root

    ensure_repo_root()

# Load secrets from dashboard/.env then repo root .env, exactly as the deploy
# script does. Without a key every run falls back to rule-based (see above).
load_dotenv(DASHBOARD_DIR / ".env")
load_dotenv(DASHBOARD_DIR.parent / ".env")

# The probe reads config and market data but writes no run rows, while importing
# the backend runs lazy CREATE TABLE/ALTER against DATABASE_PATH. Point that at a
# throwaway file *before* the backend imports so the committed prod seed database
# at dashboard/storage/data/backtest.db cannot be mutated by running this.
#
# FORCE, not setdefault. The load_dotenv calls above have already run, and
# .env.example ships `DATABASE_PATH=dashboard/storage/data/backtest.db`
# *uncommented* (line 204) — so on any checkout whose .env was copied from it, a
# setdefault sees the key already present, does nothing, and aims the lazy
# migrations straight at the committed seed DB (issue #244, plus an untracked
# -wal sidecar that hides the ALTERs). An ambient DATABASE_PATH in the operator's
# shell does the same. This script never needs the real value for anything.
# Same reasoning, same fix as backfill_runs_to_postgres.py:117.
os.environ["DATABASE_PATH"] = str(
    Path(tempfile.gettempdir()) / "probe_instruction_sensitivity.db"
)

from dashboard.backend.domain.leaderboard.baselines import (  # noqa: E402
    fetch_hourly_bars,
)
from dashboard.backend.domain.leaderboard.service import (  # noqa: E402
    MIN_LLM_DECISION_COVERAGE,
    load_leaderboard_config,
)
from dashboard.backend.domain.leaderboard.strategies import get_strategy  # noqa: E402
from dashboard.backend.domain.leaderboard.strategies._common import (  # noqa: E402
    reference_start_date,
)
from dashboard.backend.domain.leaderboard.strategies._common import (  # noqa: E402
    parse_config_date,
    timestamp_date,
)
from dashboard.backend.infrastructure.llm import token_cost  # noqa: E402
from dashboard.backend.infrastructure.llm.providers import (  # noqa: E402
    KNOWN_INTEGRATIONS,
)

# The five that become the Phase 1 seed field, plus one control that never ships.
PROBE_INSTRUCTIONS: list[tuple[str, str]] = [
    (
        "aggressive_momentum",
        "Concentrate in the strongest recent performers. Add to positions that "
        "are rising and cut losers quickly. Prefer a small number of large "
        "positions over broad diversification.",
    ),
    (
        "defensive_cash",
        "Preserve capital above all. Hold a large cash position, buy only when a "
        "stock is clearly oversold, and take profits early. Never hold more than "
        "half the portfolio in equities.",
    ),
    (
        "equal_weight_hold",
        "Spread the money evenly across many of the available stocks on the first "
        "opportunity, then hold. Do not react to short-term moves.",
    ),
    (
        "contrarian_reversion",
        "Buy what has fallen the most and sell what has risen the most. Assume "
        "prices revert toward their recent average.",
    ),
    (
        "verbose_analytical",
        "Before each decision, weigh trend, momentum and valuation signals against "
        "each other. Act only when at least two signals agree. Explain the reason "
        "for every order.",
    ),
    # CONTROL — never seeded to the board. If this scores like the others, the
    # model is ignoring the instruction and the axis is dead.
    (
        "control_nonsense",
        "The weather is pleasant today. Consider the colour blue. Bananas are a "
        "type of fruit that grows in warm climates.",
    ),
]

# Gateways match dashboard/config/leaderboard.json so a probe return is
# comparable with the published curve for the same model. Override with
# --integration when only one gateway's key is available.
PROBE_MODELS: dict[str, dict] = {
    "nemotron": {
        "model_id": "nvidia/nemotron-3-nano-30b-a3b",
        "integration": "openrouter",
        "temperature": 0,
        "reasoning_effort": "none",
    },
    "deepseek": {
        "model_id": "deepseek/deepseek-v4-pro",
        "integration": "commonstack",
        "temperature": 0,
        "reasoning_effort": "none",
    },
}

# Recorded cost per 160-step contest run, from the published agent_runs rows.
COST_PER_RUN_USD = {"nemotron": 0.0715, "deepseek": 0.7561}

KEY_BY_INTEGRATION = {
    "openrouter": "OPENROUTER_API_KEY",
    "commonstack": "COMMONSTACK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
# Fail loudly if a provider is added without telling this probe which key it needs,
# rather than letting an unknown gateway slip past the credential preflight.
assert set(KEY_BY_INTEGRATION) == set(KNOWN_INTEGRATIONS), (
    f"KEY_BY_INTEGRATION {sorted(KEY_BY_INTEGRATION)} has drifted from "
    f"KNOWN_INTEGRATIONS {sorted(KNOWN_INTEGRATIONS)}"
)

SEEDABLE = [slug for slug, _ in PROBE_INSTRUCTIONS if not slug.startswith("control_")]
CONTROLS = [slug for slug, _ in PROBE_INSTRUCTIONS if slug.startswith("control_")]

# How far outside the seeded range a control must land before it counts as
# separated. A placeholder floor, deliberately declared rather than left implicit
# at 0: the honest bar is the *measured* run-to-run noise of one instruction, so
# run a control twice and require the margin to clear that gap. Until a leg does
# that, treat a margin near this constant as "not separated" rather than a pass.
CONTROL_MARGIN_PP = 0.25

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_CONFIG = 2
EXIT_INCONCLUSIVE = 3


# A probe is only as good as its resolution. Share counts are integers, so the
# capital base decides how finely a portfolio can express an allocation — and
# therefore how small an instruction-driven difference the probe can even see.
# The 2026-08 legs were run at $10k against board curves computed at $100k: one
# median DJIA share was 2.49% of equity, 6 of 30 names were unbuyable, and every
# position collapsed to a 0-or-1-share decision. The instrument was ~4x coarser
# than the ~0.6pp effect, and both legs had to be thrown away. This check costs
# nothing and runs before the first billable call.
MAX_SHARE_FRACTION_PCT = 1.0


def _bar_date(ts):
    """Calendar date of a bar timestamp, tolerating a tz-naive frame.

    ``timestamp_date`` calls ``.astimezone``, which pandas *raises* on for a
    tz-naive Timestamp rather than assuming a zone. Live Alpaca bars are UTC-aware
    so the production path never hits it, but the caller's blanket ``except``
    would read that raise as "no price available" and refuse to spend against a
    perfectly good frame — a guard that fails closed on its own helper.
    """
    try:
        return timestamp_date(ts)
    except (TypeError, ValueError):
        return parse_config_date(str(ts)[:10])


def _window_open_price(df, window_open) -> float | None:
    """Close of the first bar on or after the contest window opens.

    Deliberately not ``iloc[0]``: bars are fetched from ``reference_start_date``,
    a calendar month before the window, so row 0 is a price the probe never
    trades at. Judging the capital base off it measures the resolution of a
    window that was never run.
    """
    if df is None or len(df) == 0 or "close" not in getattr(df, "columns", []):
        return None
    closes = df["close"]
    for pos, ts in enumerate(df.index):
        try:
            if _bar_date(ts) < window_open:
                continue
            price = float(closes.iloc[pos])
        except Exception:
            continue
        if price > 0:
            return price
    return None


def _check_capital_resolution(
    bars: dict, initial_capital: float, start_date: str
) -> str | None:
    """Return a problem description when one share is too large a slice of equity."""
    if initial_capital <= 0:
        return f"initial capital must be positive; got ${initial_capital:,.2f}"

    window_open = parse_config_date(start_date)
    prices = [
        price
        for price in (_window_open_price(df, window_open) for df in bars.values())
        if price is not None
    ]
    if not prices:
        # Fail VISIBLE, not open. Returning None here made "the base is fine" and
        # "no price could be read at all" produce the identical silent log, and
        # the very next thing main() does is spend money. A partial Alpaca
        # outage, or a frame without a lowercase `close`, lands here.
        return (
            f"could not read an opening price for any of the {len(bars)} fetched symbols "
            f"on or after {start_date}, so the capital base was never checked. Refusing "
            f"to spend against an instrument of unknown resolution."
        )

    # statistics.median, not opens[len//2] — the latter is the upper-middle value
    # for an even count, and it skews the guard toward refusing.
    median = statistics.median(prices)
    frac = median / initial_capital * 100
    per_name = initial_capital / len(prices)
    unbuyable = sum(1 for p in prices if p > per_name)

    print(
        f"  resolution   : median share at {start_date} (${median:,.2f}) = {frac:.2f}% of "
        f"${initial_capital:,.0f}; {unbuyable}/{len(prices)} names unbuyable at equal weight"
    )
    if frac <= MAX_SHARE_FRACTION_PCT:
        return None
    return (
        f"capital base too coarse to measure an instruction effect. One median share "
        f"(${median:,.2f}) is {frac:.2f}% of ${initial_capital:,.0f} — above the "
        f"{MAX_SHARE_FRACTION_PCT}% ceiling — and {unbuyable}/{len(prices)} symbols cannot "
        f"be bought at all at equal weight. Run-to-run differences of a single share would "
        f"swamp the effect under test."
    )


def _preflight_keys(model_slugs: list[str], integration_override: str | None) -> list[str]:
    """Return a list of human-readable problems; empty means good to spend."""
    problems = []
    for slug in model_slugs:
        integration = integration_override or PROBE_MODELS[slug]["integration"]
        env_key = KEY_BY_INTEGRATION[integration]
        if not os.environ.get(env_key):
            problems.append(
                f"{slug}: needs {env_key} for integration {integration!r} — unset"
            )
    return problems


def _run_one(
    *,
    instruction: str,
    model_cfg: dict,
    bars: dict,
    start_date: str,
    end_date: str,
    initial_capital: float,
) -> dict:
    strategy = get_strategy(
        {
            "strategy": "llm_agent",
            "mode": "safe_trading",
            "symbols": [],
            "strategy_prompt": instruction,
            **model_cfg,
        }
    )
    curve = strategy.run(bars, start_date, end_date, initial_capital)
    if not curve:
        raise RuntimeError("no equity curve produced — check the window and bars")

    # NOT curve[0]: _run_decision_loop calls execute_actions *before* the first
    # update_equity, so curve[0] is already post-trade and differs per
    # instruction — an instruction that deploys capital immediately gets a
    # smaller denominator than one that holds cash, and the probe would report
    # that difference as instruction signal. `initial_capital` is also the board's
    # own definition (calc_metrics, baselines.py:93), so probe returns and
    # published returns are the same quantity and can be quoted side by side.
    first_step_equity = curve[0]["equity"]
    last = curve[-1]["equity"]
    steps = int(getattr(strategy, "decision_steps", 0) or 0)
    decisions = int(getattr(strategy, "llm_decisions", 0) or 0)
    coverage = decisions / steps if steps else 0.0
    input_tokens = int(getattr(strategy, "input_tokens", 0) or 0)
    output_tokens = int(getattr(strategy, "output_tokens", 0) or 0)

    return {
        "return_pct": (last - initial_capital) / initial_capital * 100.0,
        "first_step_equity": first_step_equity,
        "coverage": coverage,
        "used_llm": bool(getattr(strategy, "used_llm", False)),
        "llm_calls": int(getattr(strategy, "llm_calls", 0) or 0),
        "llm_decisions": decisions,
        "decision_steps": steps,
        "num_trades": strategy.num_trades(),
        "final_equity": last,
        "cost_usd": token_cost.estimate_cost_usd(
            getattr(strategy, "model_id", None), input_tokens, output_tokens
        )
        or 0.0,
        # A run is only evidence if the model actually drove it. Anything else is
        # a rule-based curve wearing the model's name.
        "valid": bool(getattr(strategy, "used_llm", False))
        and coverage >= MIN_LLM_DECISION_COVERAGE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure whether a trading instruction moves a model's return"
    )
    parser.add_argument(
        "--models",
        default="nemotron",
        help="Comma-separated subset of "
        f"{','.join(PROBE_MODELS)} (default: nemotron — the cheap half, and "
        "decisive on its own if it passes)",
    )
    parser.add_argument(
        "--integration",
        default=None,
        choices=sorted(KNOWN_INTEGRATIONS),
        help="Override the gateway for every selected model (use when only one "
        "gateway's key is available). Default: each model's board gateway.",
    )
    parser.add_argument("--start", default=None, help="Override window start (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Override window end (YYYY-MM-DD)")
    parser.add_argument(
        "--instructions",
        default=None,
        help="Comma-separated subset of instruction slugs (default: all six). A "
        "smoke test should keep at least one control.",
    )
    parser.add_argument(
        "--out", default=None, help="Write the full result set to this JSON path"
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=None,
        help="Override the capital base (default: leaderboard.json's initial_capital). "
        "Set this to match the capital the published runs were computed at — see the "
        "resolution guard below; the config value has drifted from the board before.",
    )
    parser.add_argument(
        "--allow-coarse-capital",
        action="store_true",
        help="Run even when one share is a large fraction of equity. Only use this "
        "when you are deliberately measuring the coarse regime.",
    )
    parser.add_argument(
        "--list-instructions",
        action="store_true",
        help="Print the instruction set and exit without spending anything",
    )
    args = parser.parse_args()

    if args.list_instructions:
        for slug, text in PROBE_INSTRUCTIONS:
            tag = "CONTROL " if slug.startswith("control_") else "seedable"
            print(f"  [{tag}] {slug}: {text[:60]}...")
        print(f"\nseedable={SEEDABLE}")
        print(f"controls={CONTROLS}")
        return EXIT_PASS

    model_slugs = [s.strip() for s in args.models.split(",") if s.strip()]
    unknown = [s for s in model_slugs if s not in PROBE_MODELS]
    if unknown:
        print(f"FATAL: unknown model(s) {unknown}. Known: {list(PROBE_MODELS)}")
        return EXIT_CONFIG

    instructions = PROBE_INSTRUCTIONS
    if args.instructions:
        wanted = {s.strip() for s in args.instructions.split(",") if s.strip()}
        instructions = [(s, t) for s, t in PROBE_INSTRUCTIONS if s in wanted]
        missing = wanted - {s for s, _ in instructions}
        if missing:
            print(f"FATAL: unknown instruction slug(s) {sorted(missing)}")
            return EXIT_CONFIG
        # `missing` is empty when `wanted` is empty, so a value like "," slips
        # past the check above, runs nothing, and used to surface as a max()
        # traceback out of _report rather than the EXIT_CONFIG everything else
        # here returns.
        if not instructions:
            print(
                f"FATAL: --instructions {args.instructions!r} selected no instructions."
            )
            return EXIT_CONFIG

    problems = _preflight_keys(model_slugs, args.integration)
    if problems:
        print("FATAL: missing API credentials. Every run would silently fall back")
        print("to rule-based trading and the probe would measure nothing:\n")
        for p in problems:
            print(f"  - {p}")
        print("\nSet the key(s) in dashboard/.env or the environment, or pick a")
        print("different gateway with --integration.")
        return EXIT_CONFIG

    cfg = load_leaderboard_config()
    start_date = args.start or cfg["start_date"]
    end_date = args.end or cfg["end_date"]
    # The config value has drifted from the capital the board was computed at
    # before (leaderboard.json went 100000 → 1000 → 10000 in 2026-07, after the
    # published runs were written), so make the effective value explicit in the
    # header rather than letting it be an invisible default.
    initial_capital = (
        float(args.initial_capital)
        if args.initial_capital is not None
        else float(cfg["initial_capital"])
    )
    if initial_capital <= 0:
        print(f"FATAL: --initial-capital must be positive; got {initial_capital!r}")
        return EXIT_CONFIG

    est = sum(COST_PER_RUN_USD.get(m, 0.0) for m in model_slugs) * len(instructions)
    print("=" * 64)
    print("INSTRUCTION-SENSITIVITY PROBE")
    print("=" * 64)
    print(f"  window       : {start_date} → {end_date}")
    print(
        f"  capital      : ${initial_capital:,.0f}"
        + (" (--initial-capital)" if args.initial_capital is not None else " (from config)")
    )
    print(f"  models       : {', '.join(model_slugs)}")
    print(f"  instructions : {len(instructions)} ({len(model_slugs) * len(instructions)} runs)")
    print(f"  est. cost    : ~${est:.2f} (at recorded contest-run prices)")
    if args.integration:
        print(f"  gateway      : {args.integration} (overriding board defaults)")

    # One fetch for every run: the bars are identical across instructions and
    # models, and re-fetching 30 symbols twelve times is slow and rate-limited.
    bars_start = reference_start_date(start_date, cfg)
    if bars_start > start_date:
        bars_start = start_date
    probe_symbols = get_strategy(
        {"strategy": "llm_agent", "symbols": [], **PROBE_MODELS[model_slugs[0]]}
    ).required_symbols()
    print(f"\n  fetching bars {bars_start} → {end_date} for {len(probe_symbols)} symbols...")
    bars = fetch_hourly_bars(probe_symbols, bars_start, end_date)
    if not bars:
        print(f"FATAL: no market data for {bars_start} → {end_date}")
        return EXIT_CONFIG
    print(f"  got {len(bars)} symbols")

    resolution_problem = _check_capital_resolution(bars, initial_capital, start_date)
    if resolution_problem and not args.allow_coarse_capital:
        print(f"\nFATAL: {resolution_problem}")
        print("\nRe-run with --initial-capital matching the published runs, or pass")
        print("--allow-coarse-capital if the coarse regime is what you mean to measure.")
        return EXIT_CONFIG

    results: dict[str, dict[str, dict]] = {}
    failures: list[str] = []
    spent = 0.0

    # Defined BEFORE the run loop so it can be called after every completed run.
    # Persisting only after the loop protected against a _report bug but not
    # against _run_one raising — a gateway timeout on run 6 of 6 discarded the
    # five runs already paid for, which is the same money lost by a different
    # door. It must also never raise for the same reason.
    def _write(exit_code: int | None) -> None:
        if not args.out:
            return
        payload = {
            "window": {"start_date": start_date, "end_date": end_date},
            "initial_capital": initial_capital,
            "models": {m: PROBE_MODELS[m] for m in model_slugs},
            "integration_override": args.integration,
            "results": results,
            "failures": failures,
            "total_cost_usd": spent,
            "exit_code": exit_code,
        }
        try:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 — a bad --out must not cost a leg
            print(f"\nWARNING: could not write {args.out}: {type(exc).__name__}: {exc}")
            print("Dumping the payload to stdout so the paid-for runs survive:\n")
            print(json.dumps(payload, indent=2))

    for model_slug in model_slugs:
        model_cfg = dict(PROBE_MODELS[model_slug])
        if args.integration:
            model_cfg["integration"] = args.integration
        results[model_slug] = {}
        for slug, instruction in instructions:
            print(f"\n=== {model_slug} / {slug} ===")
            try:
                row = _run_one(
                    instruction=instruction,
                    model_cfg=model_cfg,
                    bars=bars,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                )
            except KeyboardInterrupt:
                print("\nInterrupted — persisting the runs already paid for.")
                _write(None)
                return EXIT_INCONCLUSIVE
            except Exception as exc:  # noqa: BLE001 — one bad run must not void the leg
                print(f"  ⚠ RUN FAILED: {type(exc).__name__}: {exc}")
                failures.append(f"{model_slug}/{slug}: {type(exc).__name__}: {exc}")
                _write(None)
                continue
            results[model_slug][slug] = row
            spent += row["cost_usd"]
            print(
                f"  return {row['return_pct']:+.2f}%  coverage {row['coverage']:.1%}  "
                f"trades {row['num_trades']}  cost ${row['cost_usd']:.4f}  "
                f"(running ${spent:.2f})"
            )
            _write(None)
            if not row["valid"]:
                print(
                    "  ⚠ INVALID: the model did not drive this run "
                    f"(used_llm={row['used_llm']}, coverage below "
                    f"{MIN_LLM_DECISION_COVERAGE:.0%}). It is not evidence."
                )

    exit_code = _report(results, spent, failures)
    _write(exit_code)
    if args.out:
        print(f"\nWrote {args.out}")

    return exit_code


def _report(
    results: dict[str, dict[str, dict]], spent: float, failures: list[str]
) -> int:
    print("\n" + "=" * 64)
    print("SPREAD")
    print("=" * 64)

    any_pass = False
    any_invalid = False
    # Whether ANY model in this invocation carried both a seeded instruction and
    # a control — i.e. whether a gate is even answerable here. A single-instruction
    # shard cannot answer it, and must not print a verdict that says it did.
    any_comparable = False
    # Whether any passing model had too few seeded instructions to say anything
    # about instruction-vs-instruction — the question Phase 2 actually rests on.
    axis_untested = False

    for model_slug, rows in results.items():
        if not rows:
            # Zero completed runs for this model. max() on the empty dict below
            # would raise and take the whole report down with it.
            print(f"\n{model_slug}: no completed runs — nothing to compare.")
            continue
        invalid = [s for s, r in rows.items() if not r["valid"]]
        any_invalid = any_invalid or bool(invalid)

        rets = {s: r["return_pct"] for s, r in rows.items()}
        spread = max(rets.values()) - min(rets.values())
        stdev = statistics.pstdev(list(rets.values()))
        print(f"\n{model_slug}: spread {spread:.2f}pp (all runs), stdev {stdev:.2f}pp")
        for slug, ret in sorted(rets.items(), key=lambda kv: -kv[1]):
            marker = "  (control)" if slug.startswith("control_") else ""
            flag = "" if rows[slug]["valid"] else "  ⚠ INVALID"
            print(f"   {ret:+7.2f}%  {slug}{marker}{flag}")

        if invalid:
            print(f"  ⚠ {len(invalid)} run(s) were not model-driven: {invalid}")
            print("    This model's numbers are not evidence either way.")
            continue

        # The control is the load-bearing measurement, not the spread: a nonsense
        # instruction landing mid-pack means the model responds to *having* a
        # strategy body rather than to its content — which passes a naive spread
        # check while the board ranks noise.
        seeded = [r for s, r in rets.items() if not s.startswith("control_")]
        controls = {s: r for s, r in rets.items() if s.startswith("control_")}
        if controls and not seeded:
            print("  control(s) ran, but no seeded instruction in this invocation —")
            print("  a control is only meaningful *positioned against* seeded returns.")
            print("  Merge with the seeded runs before reading any gate off this.")
            continue
        if not controls:
            continue
        any_comparable = True

        # Measured over the SEEDED runs alone. The all-runs `spread` above
        # includes the control, so using it collapsed the gate's two conditions
        # into one: five identical seeded instructions plus one far-off control
        # produced a 3.5pp "spread", cleared the >=1pp test, and printed
        # "separates instructions" for a dead axis — exactly what the control
        # exists to catch.
        seeded_spread = max(seeded) - min(seeded) if len(seeded) > 1 else None
        if seeded_spread is None:
            print(
                f"  seeded spread: n/a — only {len(seeded)} seeded instruction ran, so "
                "instruction-vs-instruction is UNTESTED"
            )
        else:
            print(
                f"  seeded spread: {seeded_spread:.2f}pp across {len(seeded)} instructions"
            )

        margins: dict[str, float] = {}
        for cslug, cret in controls.items():
            # Margin, not a bare boundary test. `cret < min(seeded)` is true when
            # the control beats the best seeded run by 0.01pp — noise wearing the
            # word OUTLIER, which is how the Nemotron leg printed "OUTLIER (good)"
            # for a result that separates nothing. Same defect class as the
            # `payload.period !== 'live'` banner check in leaderboard.js: a
            # boundary that inverts its own meaning at small margins.
            margin = max(min(seeded) - cret, cret - max(seeded))
            margins[cslug] = margin
            rank = sorted(rets.values(), reverse=True).index(cret) + 1
            verdict = (
                "OUTLIER (good)"
                if margin >= CONTROL_MARGIN_PP
                else f"NOT SEPARATED (margin {margin:+.2f}pp < {CONTROL_MARGIN_PP}pp)"
            )
            print(
                f"  control {cslug}: {cret:+.2f}% — rank {rank}/{len(rets)}, {verdict}"
            )

        # ALL controls, not any. One nonsense instruction landing mid-pack IS the
        # dead-axis result, and a second control landing clear does not cancel it.
        all_separated = all(m >= CONTROL_MARGIN_PP for m in margins.values())
        # Distance from the seeded band to the *nearest* control — the actual
        # "an instruction moved the return by this much" quantity.
        signal = min(margins.values())
        axis_ok = seeded_spread is None or seeded_spread >= 1.0

        if all_separated and signal >= 1.0 and axis_ok:
            any_pass = True
            axis_untested = axis_untested or seeded_spread is None
        elif all_separated and signal >= 1.0 and not axis_ok:
            print(
                f"  ⚠ controls separate ({signal:.2f}pp), but the seeded instructions do "
                f"not separate from each other ({seeded_spread:.2f}pp < 1pp): the model "
                "responds to HAVING an instruction, not to its content. A Phase 2 board "
                "built on this ranks noise."
            )

    print("\n" + "=" * 64)
    print(f"Total spend this run: ${spent:.2f}")
    if failures:
        print(f"GATE: INCONCLUSIVE — {len(failures)} run(s) failed outright:")
        for failure in failures:
            print(f"  - {failure}")
        print("The instruction set is incomplete, so no verdict is available from it.")
        print("=" * 64)
        return EXIT_INCONCLUSIVE
    if any_invalid:
        print("GATE: INCONCLUSIVE — at least one run was not driven by the model.")
        print("Fix credentials/billing and re-run. Do NOT read this as a FAIL:")
        print("rule-based fallback produces identical curves, i.e. a 0.00pp spread.")
        print("=" * 64)
        return EXIT_INCONCLUSIVE
    if any_pass:
        print("GATE: PASS — for at least one model, EVERY control lands >=1pp outside")
        print("the seeded range, and the seeded instructions do not collapse together.")
        if axis_untested:
            print("")
            print("NOTE: only one seeded instruction ran, so this licenses exactly")
            print("'an instruction moves the return' — NOT 'instructions can be ranked")
            print("against each other', which is what Phase 2 actually needs. Run two")
            print("plausible instructions against each other before building on it.")
        print("=" * 64)
        return EXIT_PASS
    if not any_comparable:
        # A shard — one instruction per process, so the leg's runs can proceed
        # independently. Printing FAIL here would leave every shard log asserting
        # "do NOT build Phase 2" on evidence that cannot support any verdict, and
        # a confidently-wrong FAIL is the failure mode this probe exists to avoid.
        print("GATE: N/A — this invocation has no seeded/control pair to compare.")
        print("It is a shard, not a verdict. Merge the leg's outputs and read the")
        print("gate off the combined set.")
        print("=" * 64)
        return EXIT_INCONCLUSIVE
    print("GATE: FAIL — no model both separates instructions and isolates the")
    print("control. Per the gate table, do NOT build Phase 2.")
    print("=" * 64)
    return EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
