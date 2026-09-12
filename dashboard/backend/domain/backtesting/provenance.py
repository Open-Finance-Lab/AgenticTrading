"""What actually drove a finished backtest's decisions (issue #169).

A dashboard backtest could fall back to rule-based logic on *every* step and
still report "Backtest completed successfully". Two independent layers made
that possible, and this module is the second half of closing both.

**The run row was already honest and nothing read it.** ``engine.py`` defaults
``llm_model`` to ``"rule-based"`` and promotes it to the model's name only
after a call succeeds, so a run where every step held already persisted the
honest label -- ``/backtest/status`` simply never looked at it.

**The counter that catches the worst case was never persisted.** A run whose
every step made a billed call whose response failed to parse has
``llm_calls == decision_steps`` and ``llm_model == "claude-..."`` while the
model drove nothing at all. Only ``llm_decisions`` separates that from a clean
run, and ``agent_runs`` had no column for it -- so the one component that could
tell them apart, the leaderboard's H6 guard, could not: it runs in-process and
dashboard backtests are subprocesses.

Hence ``llm_calls`` is never a coverage numerator here. It is the *billing*
counter: it ticks on a truncated or unparseable response that then traded
rule-based, which is exactly the run this module exists to name.

The threshold is imported from the leaderboard domain rather than restated.
Two owners of one number disagree eventually, and "the dashboard and the
leaderboard disagree about whether the model drove this run" is the same defect
one layer up.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from dashboard.backend.domain.leaderboard.service import MIN_LLM_DECISION_COVERAGE
from dashboard.backend.infrastructure.market_data.profiles import (
    LLM_DECISION_SOURCE,
    RULE_BASED_DECISION_SOURCE,
)

# The observed verdict deliberately reuses the *requested* decision-source
# vocabulary, so "what you asked for" and "what you got" are comparable with
# ``==`` instead of through a translation table that can drift.
DECISION_PROVENANCE_LLM = LLM_DECISION_SOURCE
DECISION_PROVENANCE_RULE_BASED = RULE_BASED_DECISION_SOURCE
DECISION_PROVENANCE_PARTIAL = "partial"
#: The row predates the counter, so no verdict can be drawn from it. Distinct
#: from ``rule_based`` on purpose -- see ``run_decision_provenance``.
DECISION_PROVENANCE_UNKNOWN = "unknown"

#: Metadata key carrying the coverage denominator, and the witness that a row
#: records ``llm_decisions`` at all. See ``run_decision_provenance``.
DECISION_STEPS_KEY = "decision_steps"

# Labels the engine persists when no model drove the run. "rule-based" is the
# one it actually writes (engine.py); the rest are defensive, since a caller
# with no model configured can leave the column empty or null.
_RULE_BASED_MODEL_LABELS = frozenset({"", "rule-based", "rule_based", "none", "null"})


def _as_int(value: Any) -> int:
    """``value`` as a count, treating anything unreadable as zero."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def classify_decision_provenance(
    *,
    llm_model: Optional[str],
    llm_calls: Optional[int],
    llm_decisions: Optional[int],
    decision_steps: Optional[int] = None,
) -> str:
    """What drove this run: ``llm``, ``partial``, ``rule_based`` or ``unknown``.

    The load-bearing case is ``llm_calls == decision_steps`` with
    ``llm_decisions == 0``: every step made a billed call, every response was
    unusable, every step traded rule-based. A coverage check keyed on
    ``llm_calls`` reports that run as 100% model-driven, which is why the two
    counters exist separately.

    ``llm_decisions=None`` means "not recorded", and answers ``unknown`` rather
    than ``rule_based``. A run written before the column existed reads back as
    0 like any other, and accusing every historical run of a fallback it did
    not have would be the same kind of lie in the other direction.

    ``decision_steps`` is the denominator the leaderboard's H6 guard uses --
    steps the model was *asked* to decide. Falling back to ``llm_calls`` when
    it is unknown keeps the total-fallback case correct (0 decisions out of
    anything is still rule-based) but understates a run that skipped the model
    on some steps entirely, so record it where you can.
    """
    if (llm_model or "").strip().lower() in _RULE_BASED_MODEL_LABELS:
        return DECISION_PROVENANCE_RULE_BASED

    calls = _as_int(llm_calls)
    if calls <= 0:
        # A model name beside zero calls is a rule-based curve wearing that
        # model's name -- the same shape the H6 guard refuses to publish.
        return DECISION_PROVENANCE_RULE_BASED

    if llm_decisions is None:
        return DECISION_PROVENANCE_UNKNOWN

    decisions = _as_int(llm_decisions)
    if decisions <= 0:
        return DECISION_PROVENANCE_RULE_BASED

    steps = _as_int(decision_steps)
    denominator = steps if steps > 0 else calls
    # Same comparison as the H6 guard, inverted: it raises when
    # ``llm_decisions < MIN_LLM_DECISION_COVERAGE * decision_steps``.
    if decisions >= MIN_LLM_DECISION_COVERAGE * denominator:
        return DECISION_PROVENANCE_LLM
    return DECISION_PROVENANCE_PARTIAL


def run_decision_provenance(run: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Provenance block for one persisted ``agent_runs`` row, or None.

    Returns both facts side by side, because the bug is precisely that they can
    differ: ``decision_source`` is what the caller *asked* for (the meaning that
    name already carries in ``POST /backtest/run``'s response and in this row's
    own metadata) and ``decision_provenance`` is what actually happened.
    ``decision_fallback`` is the two disagreeing -- the run asked for the model
    and did not get it.

    ``metadata.decision_steps`` is the witness that this row records
    ``llm_decisions``. The column was added with ``DEFAULT 0``, so every row
    written before it existed reads back as 0 and a bare 0 cannot distinguish
    "the model drove nothing" from "nobody was counting". The metadata key is
    written by the same code path that writes the column, so its presence is
    the one thing that can.
    """
    if not run:
        return None

    metadata = run.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}

    recorded = DECISION_STEPS_KEY in metadata
    decision_steps = _as_int(metadata.get(DECISION_STEPS_KEY)) if recorded else None
    llm_decisions = _as_int(run.get("llm_decisions")) if recorded else None

    llm_model = run.get("llm_model")
    llm_calls = _as_int(run.get("llm_calls"))
    verdict = classify_decision_provenance(
        llm_model=llm_model,
        llm_calls=llm_calls,
        llm_decisions=llm_decisions,
        decision_steps=decision_steps,
    )

    requested = metadata.get("decision_source")
    requested = str(requested) if requested else None
    # Only claimed when the request is on record. Inferring intent from
    # ``llm_model`` cannot work here: a total fallback persists the honest
    # "rule-based" label, so the model's name is gone exactly when a fallback
    # happened -- which would report the worst case as "no fallback".
    fallback = requested == LLM_DECISION_SOURCE and verdict in (
        DECISION_PROVENANCE_RULE_BASED,
        DECISION_PROVENANCE_PARTIAL,
    )

    block: Dict[str, Any] = {
        "decision_provenance": verdict,
        "decision_source": requested,
        "decision_fallback": fallback,
        "llm_model": llm_model,
        "llm_calls": llm_calls,
        "llm_decisions": llm_decisions,
        "decision_steps": decision_steps,
    }
    block["decision_note"] = decision_provenance_note(block)
    return block


def decision_provenance_note(provenance: Mapping[str, Any]) -> Optional[str]:
    """Short clause for a surface that has already said the run finished.

    Single owner of this copy. The browser renders whatever this returns rather
    than composing its own sentence out of the same counters, because the same
    fields worded twice is two owners of one message, and they disagree the
    moment either side's wording moves.

    None for a clean run and for ``unknown``: a run the model drove needs no
    note, and a row written before the counters existed has nothing to report
    -- saying anything there would be inventing a finding out of missing data.
    """
    verdict = provenance.get("decision_provenance")
    decisions = _as_int(provenance.get("llm_decisions"))
    steps = _as_int(provenance.get("decision_steps"))

    if verdict == DECISION_PROVENANCE_PARTIAL:
        scale = f" only {decisions} of {steps} steps" if steps else " only part of it"
        return (
            f"The model drove{scale} — the rest fell back to rule-based logic."
        )
    if verdict == DECISION_PROVENANCE_RULE_BASED:
        if provenance.get("decision_fallback"):
            return (
                "No step used the model — every decision fell back to "
                "rule-based logic."
            )
        return "Rule-based strategy — no model decisions."
    return None


def describe_decision_provenance(provenance: Mapping[str, Any]) -> str:
    """The completion message for one run, as a whole sentence."""
    note = decision_provenance_note(provenance)
    return f"Backtest completed. {note}" if note else "Backtest completed successfully"
