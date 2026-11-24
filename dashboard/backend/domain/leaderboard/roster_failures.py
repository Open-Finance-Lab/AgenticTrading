"""Per-entry failure bookkeeping for the leaderboard refresh loops.

``refresh_live_leaderboard`` and ``refresh_daily_leaderboard`` deploy each LLM
entry in turn and catch ``Exception`` per entry, so one model's failure does not
leave the rest of the roster untraded. Both hand the failure to
``RosterFailures``, which logs it, records it, and decides when to stop.

Stopping is what a bare catch-all gets wrong. A fault in code every entry shares
(the checkpoint write, the run metadata, the database) raises for each model in
turn, and on the live path it raises after the session's model calls were
billed. Catching it and moving on bills the whole roster for nothing, and the
refresh retries the window every night until someone fixes the fault. So two
entries in a row failing with the same error is read as that case: the rest of
the roster is recorded as not attempted. One entry's failure, or two different
ones, is still isolated, and an entry that actually trades in between resets the
count, because it proves the shared path works.
"""

from __future__ import annotations

import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

# The types the per-entry catch named before it widened to ``Exception``. An H6
# refusal (``LeaderboardFallbackError``), a session gap (``LiveSessionGapError``)
# and missing bars are all ``RuntimeError``s. Anything else used to escape the
# loop with a full traceback, and still gets one here: those are the bugs.
EXPECTED_ENTRY_ERRORS: Tuple[type, ...] = (ValueError, RuntimeError)

# Consecutive entries failing with the same error before the rest are skipped.
REPEATS_BEFORE_STOP = 2


def _type_name(exc: BaseException) -> str:
    """The class name, module-qualified unless it is a builtin or this repo's.

    Bare names collide across the outages an operator has to tell apart:
    ``sqlite3.OperationalError`` and ``psycopg.OperationalError``, or the
    builtin ``ConnectionError`` and ``requests.exceptions.ConnectionError``.
    """
    cls = type(exc)
    module = cls.__module__ or ""
    if module == "builtins" or module.startswith("dashboard."):
        return cls.__qualname__
    return f"{module}.{cls.__qualname__}"


def _one_line(exc: BaseException) -> str:
    # Driver messages span lines (psycopg's "server closed the connection
    # unexpectedly\n\tThis probably means..."), and a log search on the ERROR
    # line returns only its first.
    return " ".join(str(exc).split())


def describe_error(exc: BaseException) -> str:
    """``Type: message`` on one line, or the bare type when there is no message
    (``TimeoutError()``), so the record never reads as an empty string."""
    message = _one_line(exc)
    name = _type_name(exc)
    return f"{name}: {message}" if message else name


class RosterFailures:
    """What one refresh's roster loop failed on, and whether to keep going."""

    def __init__(self, board: str) -> None:
        self.board = board
        self.failures: List[Dict[str, Any]] = []
        self._streak_key: Optional[Tuple[str, str]] = None
        self._streak: List[str] = []
        self._streak_type = ""

    @property
    def stopped(self) -> bool:
        return len(self._streak) >= REPEATS_BEFORE_STOP

    def failed(self, entry_id: str, exc: Exception) -> None:
        """Log and record one entry's failure."""
        error = describe_error(exc)
        print(
            f"ERROR: {self.board}.model_deploy_failed entry={entry_id} error={error}",
            flush=True,
        )
        if not isinstance(exc, EXPECTED_ENTRY_ERRORS):
            print(
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                end="",
                flush=True,
            )
        self.failures.append({"entry_id": entry_id, "error": error})
        # The entry's own id is masked so a shared fault whose message names
        # the entry (a session gap, a missing checkpoint) still compares equal.
        # Whole tokens only: an id inside a run id or another word stays put.
        cls = type(exc)
        key = (
            f"{cls.__module__}.{cls.__qualname__}",
            re.sub(
                rf"(?<![\w-]){re.escape(entry_id)}(?![\w-])", "<entry>", _one_line(exc)
            ),
        )
        if key != self._streak_key:
            self._streak_key, self._streak = key, []
        self._streak.append(entry_id)
        self._streak_type = _type_name(exc)

    def traded(self, row: Dict[str, Any]) -> None:
        """An entry deployed. A cached row ran none of the shared path, so only
        one that traded resets the streak."""
        if not row.get("cached"):
            self._streak_key, self._streak = None, []

    def skip(self, entry_id: str) -> None:
        """Record an entry left undeployed because the roster stopped."""
        after = ",".join(self._streak)
        print(
            f"ERROR: {self.board}.model_not_attempted entry={entry_id} "
            f"after={after} error={self._streak_type}",
            flush=True,
        )
        self.failures.append(
            {
                "entry_id": entry_id,
                "error": (
                    f"not attempted: {' and '.join(self._streak)} failed in a "
                    f"row with the same {self._streak_type}"
                ),
            }
        )
