"""Stdlib bridge executed by the isolated TradingAgents interpreter.

Runs as a *script* under a separate interpreter, never as part of the backend
process, so it imports nothing from ``dashboard.backend``. Two packages must be
reachable from that interpreter: ``tradingagents`` (installed into the isolated
environment) and ``agentictrading`` (ATL's own SDK, whose
``TradingAgentsDecisionGenerator`` owns the decision logic and is reused here
rather than reimplemented — the adapter supplies it on ``PYTHONPATH``).

One invocation is one billed decision: a single symbol on a single analysis
date. Model and provider are platform-owned and validated here rather than
trusted from the payload, because TradingAgents accepts any model name on
OpenRouter and silently defaults to ``llm_provider='openai'`` — neither upstream
nor the SDK would reject a crafted config, so this is the last place that can.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Platform-managed gateways. The *deployment* picks one (see the adapter's
# ``select_managed_gateway``); a payload may only ever *match* an entry, never
# introduce one -- so provider, model and endpoint stay platform-owned even if
# a crafted config reaches this process.
#
# CommonStack has no provider id of its own upstream, so ATL's gateway rides
# the generic ``openai_compatible`` endpoint, which requires an explicit URL.
MANAGED_GATEWAYS: Dict[str, Dict[str, Any]] = {
    "openrouter": {
        "llm_provider": "openrouter",
        "deep_think_llm": "nvidia/nemotron-3-nano-30b-a3b",
        "quick_think_llm": "nvidia/nemotron-3-nano-30b-a3b",
        # The openrouter provider carries its own base URL.
        "backend_url": None,
    },
    "commonstack": {
        "llm_provider": "openai_compatible",
        "deep_think_llm": "anthropic/claude-haiku-4-5",
        "quick_think_llm": "anthropic/claude-haiku-4-5",
        "backend_url": "https://api.commonstack.ai/v1",
    },
}

DEFAULT_MANAGED_GATEWAY = "openrouter"

# Debate structure, pinned by ATL rather than inherited from upstream's
# DEFAULT_CONFIG. These two decide how many LLM calls one decision costs -- the
# graph exits on ``count >= 2 * debate`` and ``count >= 3 * risk`` -- so an
# upstream default change would otherwise silently alter both runtime and spend
# with nothing in ATL's own records to show it. 1/1 is upstream's current
# default and what every measurement so far was taken against, so sending them
# explicitly changes no behaviour; it only makes the structure recorded.
MANAGED_MAX_DEBATE_ROUNDS = 1
MANAGED_MAX_RISK_DISCUSS_ROUNDS = 1

# Named aliases for the default gateway's pin. The adapter reports these to ATL
# and the tests build payloads from them, so they must stay in this module --
# the value ATL reports and the value the bridge accepts cannot be allowed to
# drift apart.
MANAGED_LLM_PROVIDER = MANAGED_GATEWAYS[DEFAULT_MANAGED_GATEWAY]["llm_provider"]
MANAGED_DEEP_THINK_MODEL = MANAGED_GATEWAYS[DEFAULT_MANAGED_GATEWAY]["deep_think_llm"]
MANAGED_QUICK_THINK_MODEL = MANAGED_GATEWAYS[DEFAULT_MANAGED_GATEWAY][
    "quick_think_llm"
]

# TradingAgents' own analyst registry at the pinned revision
# (``tradingagents/graph/setup.py``). Analyst composition is the one
# strategy-level choice an ATL agent persists.
SUPPORTED_ANALYSTS = ("market", "social", "news", "fundamentals")

# The debate transcript is unbounded, upstream-controlled text. ATL keeps a
# digest of it and this much of a readable excerpt; the rest never crosses.
MAX_EXCERPT_LENGTH = 500

# Non-secret TradingAgents knobs the caller may set. Everything else — result
# and cache directories, project paths, anything credential-shaped — stays at
# the upstream default, so a payload cannot redirect writes or smuggle keys
# into the run.
_ALLOWED_CONFIG_KEYS = (
    "max_debate_rounds",
    "max_risk_discuss_rounds",
    "temperature",
    "llm_max_retries",
    "output_language",
    "data_vendors",
    "tool_vendors",
)


_COMMIT_RE = re.compile(r"([0-9a-f]{40})")

# Upstream's five-tier scale, and the label pattern its parser looks for.
# Both mirror ``tradingagents/agents/utils/rating.py`` at the pinned revision.
_RATING_SCALE = frozenset({"buy", "overweight", "hold", "underweight", "sell"})
# Upstream's own pattern, copied exactly. Used ONLY to predict what upstream's
# first pass will do -- a narrower copy here would mispredict it, which is the
# one thing this mirror must not do.
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)
# Deliberately stricter, and used for a different question: did the model
# *state* a rating in label form? Anchored at the start of the line and
# requiring the word "rating" there, so prose that merely mentions a rating is
# not read as stating one. Upstream's pattern is far looser -- its `[:\-]`
# makes the hyphen in "Our rating is Buy - a medium-term call." produce the
# capture "a" -- and treating that as a stated rating refused a decision
# upstream had resolved correctly.
_STATED_RATING_RE = re.compile(
    r"^\s*\**\s*(?:final|overall|revised|recommended|portfolio|committee)?\s*"
    r"rating\s*\**\s*[:\-]\s*\**\s*([A-Za-z][A-Za-z ]*)",
    re.IGNORECASE,
)


def unsupported_rating_label(raw: str) -> str:
    """The out-of-scale rating the text explicitly states, or ``""``.

    Guards one specific, direction-reversing failure in upstream's parser.
    ``parse_rating`` works in two passes: an explicit ``Rating: X`` label
    first, then the first five-tier word appearing *anywhere* in the text. The
    passes are not exclusive -- a label whose value is outside the scale
    matches the regex but fails the set test, so the first pass falls through
    silently and the second pass answers with whatever rating word happens to
    appear earliest in the prose.

    "Strong Buy" is the case that matters, because models emit it constantly
    and it is not on the scale. Given::

        ... the bear case recommends Sell.
        Final Rating: Strong Buy

    pass one sees ``Rating: Strong``, rejects "strong", and pass two returns
    **Sell** -- from the sentence describing the argument that lost. An
    explicitly bullish decision becomes a sell order, with a rating that
    validates cleanly all the way through: ATL's own rating/action cross-check
    cannot see it, because both values come from that one parse.

    So this is checked here, against the raw text, before the rating is
    trusted. It deliberately flags *only* the stated-but-unsupported case. A
    text with no rating label at all also goes through pass two, but refusing
    every unlabelled decision would refuse most legitimate output; "the model
    named a rating we do not support, so do not guess" is a rule that can be
    defended.

    The real fix is upstream: pass one should stop looking once it has found a
    label. That is a change to a shared module and to trade direction, so it
    is escalated rather than made here.

    ⚠ **Pass one has to be mirrored before complaining, not skipped.** The
    first version of this function returned on the first out-of-scale label it
    saw, which is not what upstream does: upstream's pass one *keeps scanning*
    and answers with the first label value that IS on the scale. So a
    perfectly good decision was refused whenever any later line merely
    mentioned a rating -- and the regex's ``[:\\-]`` makes that common, because
    a hyphen counts::

        Rating: Buy                                      -> upstream: 'Buy'
        The rating reflects a risk-adjusted view.         -> flagged 'adjusted'

        Final Rating: Buy / Risk rating: Elevated         -> flagged 'Elevated'
        Rating: Overweight / Rating confidence: 8/10      -> flagged '8'

    Each of those cost a fully billed decision and held the step, and the
    third such day would abort the run. Any hyphenated word on a line
    containing "rating", and any secondary label (``Risk rating:``,
    ``ESG rating:``), was enough. Found by independent review, reproduced, and
    fixed here by asking the real question: *did upstream's explicit-label
    pass succeed?* If it did, there is no contradiction to find.
    """
    lines = str(raw or "").splitlines()
    # Pass one, exactly as upstream runs it. If it resolves, the rating came
    # from a label the model stated and ATL supports: nothing to refuse.
    for line in lines:
        match = _RATING_LABEL_RE.search(line)
        if match and match.group(1).lower() in _RATING_SCALE:
            return ""
    # Pass one fell through, so upstream answered from prose. If the model
    # nonetheless *stated* a rating in label form, it stated one we do not
    # support -- and what upstream traded on came from somewhere else.
    for line in lines:
        match = _STATED_RATING_RE.match(line)
        if match:
            stated = match.group(1).strip()
            # Only the first word is compared against the scale upstream used;
            # the rest is kept so the message can name "Strong Buy" rather
            # than "Strong".
            if stated.split()[0].lower() not in _RATING_SCALE:
                return stated
    return ""


def _commit_from_url(url: Any) -> str:
    """Return the 40-hex revision embedded in an install URL, or ``""``.

    Only a full revision identifies a build; a short sha or a PyPI/branch
    install has nothing unambiguous to report, and inventing one would be worse
    than recording nothing.
    """
    match = _COMMIT_RE.search(str(url or ""))
    return match.group(1) if match else ""


def _upstream_commit() -> str:
    """Return the exact upstream revision this environment installed.

    ``importlib.metadata.version`` reports the *tag* (0.3.1) and cannot tell two
    builds apart -- ``requirements-tradingagents.txt`` pins a commit six commits
    past that tag, so two different revisions would produce identical ATL
    records. pip and uv write the real source into ``direct_url.json``, which is
    the only in-process witness of which revision actually ran.

    Never fatal: a missing or unreadable file means "unknown revision", not a
    failed decision.
    """
    try:
        import importlib.metadata as importlib_metadata

        raw = importlib_metadata.distribution("tradingagents").read_text(
            "direct_url.json"
        )
        return _commit_from_url(json.loads(raw or "{}").get("url"))
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return ""


class BridgeInputError(ValueError):
    """Raised when the caller's payload cannot safely become an analysis."""


class BridgeOutputError(ValueError):
    """Raised when upstream's own answer cannot safely become an order.

    Separate from ``BridgeInputError`` because the parent classifies failures
    by their text to decide whether to abort the run or hold the step
    (``classify_runtime_failure``). A rejected payload is ATL's own bug and
    identical on every step, so it aborts. An answer this one call produced is
    a property of that generation, so the step holds and the run continues.
    """


def _managed_config_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fail closed on anything but one of the platform's managed gateways.

    The payload has to match a ``MANAGED_GATEWAYS`` entry *as a whole*: a
    provider borrowed from one entry with models from another is not a gateway
    this platform operates, and neither is an arbitrary model behind a provider
    that happens to accept any model name.
    """
    config = payload.get("config")
    if not isinstance(config, dict):
        raise BridgeInputError(
            "TradingAgents bridge requires a platform-managed gateway config"
        )
    provider = str(config.get("llm_provider") or "").strip()
    deep_think = str(config.get("deep_think_llm") or "").strip()
    quick_think = str(config.get("quick_think_llm") or "").strip()
    for candidate in MANAGED_GATEWAYS.values():
        if (
            provider == candidate["llm_provider"]
            and deep_think == candidate["deep_think_llm"]
            and quick_think == candidate["quick_think_llm"]
        ):
            gateway = candidate
            break
    else:
        raise BridgeInputError(
            "TradingAgents bridge requires the platform-managed gateway models"
        )

    resolved: Dict[str, Any] = {
        "llm_provider": gateway["llm_provider"],
        "deep_think_llm": gateway["deep_think_llm"],
        "quick_think_llm": gateway["quick_think_llm"],
        # Taken from the table, never echoed from the payload: the endpoint is
        # the one value a crafted config would most want to move.
        "backend_url": gateway["backend_url"],
    }
    for key in _ALLOWED_CONFIG_KEYS:
        if key in config:
            resolved[key] = config[key]
    return resolved


def _single_analysis_date(payload: Dict[str, Any]) -> str:
    """Return the one date this invocation may analyse.

    One call is one billed multi-agent run. A payload carrying several dates
    would turn a single backtest step into that many of them.
    """
    dates = payload.get("analysis_dates")
    if not isinstance(dates, list) or len(dates) != 1:
        raise BridgeInputError(
            "TradingAgents bridge requires exactly one analysis date per call"
        )
    return str(dates[0])


def _selected_analysts(payload: Dict[str, Any]) -> List[str]:
    analysts = payload.get("selected_analysts")
    if not isinstance(analysts, list) or not analysts:
        raise BridgeInputError("TradingAgents bridge requires at least one analyst")
    names = [str(item).strip() for item in analysts]
    if len(set(names)) != len(names):
        raise BridgeInputError("TradingAgents analysts must not repeat")
    unsupported = sorted(set(names) - set(SUPPORTED_ANALYSTS))
    if unsupported:
        raise BridgeInputError(f"unsupported TradingAgents analysts: {unsupported}")
    return names


def _disable_dotenv_loading() -> None:
    """Prevent any dependency from discovering a checkout's or home ``.env``.

    Applied with ``setattr`` because this is a deliberate runtime monkeypatch,
    not a module-level rebinding: this file only ever executes as the entry
    point of a *separate* interpreter, so it can never reach ATL's own
    ``load_dotenv`` call sites. Tolerates python-dotenv's absence — it is a
    transitive dependency of the isolated environment, not a guarantee.
    """
    try:
        import dotenv
        from dotenv import main as dotenv_main
    except ImportError:
        return

    def disabled_load_dotenv(*_args, **_kwargs) -> bool:
        return False

    setattr(dotenv, "load_dotenv", disabled_load_dotenv)
    setattr(dotenv_main, "load_dotenv", disabled_load_dotenv)


def _sanitized_message(error: BaseException) -> str:
    """Return credential-free failure text, or a marker if it cannot be made so.

    Prefixed with the exception class, which is the only part of a failure
    that survives the process boundary intact and is never a secret. The
    parent classifies this text to decide whether to abort the run or hold the
    step (``classify_runtime_failure``), and ``sanitize_error_message``
    returns ``str(error)`` alone -- so without the prefix, a deterministic
    fault like the bridge's own fail-closed refusal was indistinguishable from
    a rate limit and the run paid to rediscover it once per trading day.
    """
    try:
        from agentictrading.integrations._tradingagents_core import (
            sanitize_error_message,
        )

        return f"{type(error).__name__}: {sanitize_error_message(error)}"
    except Exception:  # noqa: BLE001 - never print unredacted upstream text
        return f"{type(error).__name__}: detail withheld (sanitizer unavailable)"


def _decide(payload: Dict[str, Any]) -> Dict[str, Any]:
    _disable_dotenv_loading()
    # Resolves only inside the isolated environment, never ATL's own.
    from agentictrading.integrations._tradingagents_core import (
        TradingAgentsDecisionGenerator,
    )

    analysis_date = _single_analysis_date(payload)
    analysts = _selected_analysts(payload)
    config = _managed_config_from_payload(payload)

    # One date per call, so the first failure is this call's failure: stopping
    # at one keeps the generator's detailed (already sanitized) message instead
    # of collapsing to its generic all-dates-failed text.
    generator = TradingAgentsDecisionGenerator(max_consecutive_failures=1)
    artifact = generator.generate(
        symbol=str(payload.get("symbol") or ""),
        analysis_dates=[analysis_date],
        config=config,
        selected_analysts=analysts,
    )
    record = artifact.decisions[0]
    # Checked against the raw text, not the parsed rating, and only on a record
    # the SDK already accepted: see unsupported_rating_label for why the two
    # cannot disagree once parsing is done. Raised rather than downgraded to a
    # hold, so the step is visibly refused instead of quietly becoming a
    # no-trade day that looks like the model's own call.
    if record.status == "valid":
        stated = unsupported_rating_label(record.raw_final_trade_decision)
        if stated:
            raise BridgeOutputError(
                f"TradingAgents stated an unsupported rating {stated!r}; "
                f"upstream's parser answered {record.rating!r} from elsewhere "
                "in the text, which may be the opposite direction"
            )
    return {
        "schema_version": artifact.schema_version,
        "manifest": artifact.manifest,
        # Outside the manifest on purpose: the manifest's shape is the SDK's
        # contract, shared with the CLI and the replay path, and this is a fact
        # about the *installed environment* rather than about the analysis.
        "upstream_commit": _upstream_commit(),
        "decision": {
            "analysis_date": record.analysis_date,
            "rating": record.rating,
            "atl_action": record.atl_action,
            "status": record.status,
            "attempts": record.attempts,
            "raw_sha256": record.raw_sha256,
            "raw_excerpt": record.raw_final_trade_decision[:MAX_EXCERPT_LENGTH],
            # Carried so an error record is reportable as what it is. With
            # ``max_consecutive_failures=1`` the generator raises on the first
            # failed date instead of returning one of these, so this is the
            # defensive half: without it, a record that ever did arrive here
            # reached the adapter as rating="" and was reported as an
            # unsupported rating rather than as the upstream failure it is.
            # Already sanitized by the SDK (``sanitize_error_message``); the
            # adapter bounds the length again on the way into the audit row.
            "error_type": record.error_type or "",
            "error_message": record.error_message or "",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        output = _decide(payload)
    except Exception as exc:  # noqa: BLE001 - the traceback is the leak
        # `generate()` re-raises initialization failures with `from exc`, so
        # letting this propagate would print the original exception -- and any
        # credential in it -- to the stderr the adapter captures and persists.
        sys.stderr.write(_sanitized_message(exc) + "\n")
        raise SystemExit(1) from None

    # Written only on success: a stale file would let the adapter read a
    # previous step's decision as this step's.
    Path(args.output).write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    main()
