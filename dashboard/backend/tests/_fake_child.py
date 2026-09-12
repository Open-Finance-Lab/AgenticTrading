"""A ``Popen`` stand-in for the backtest parent's side of the contract.

``run_backtest_background`` launches its child with ``subprocess.Popen`` rather
than ``subprocess.run`` so that ``POST /backtest/cancel`` has a handle to signal
and so the parent never buffers the whole child log (issues #273 and #308).
That moved four behaviours from the stdlib into this repo's own code — the
wall-clock budget, the stream drain, the SIGTERM/SIGKILL escalation and the
handle's lifetime on the slot — and every one of them is now something a test
has to be able to drive.

Shared rather than copied because a fake that quietly diverges between two test
modules is how one of them stops testing the thing it names.
"""

from __future__ import annotations

import io
import subprocess
from typing import List, Optional


class RecordingPipe:
    """A write side that keeps what was written (``io.StringIO`` discards it)."""

    def __init__(self) -> None:
        self.writes: List[str] = []
        self.closed = False

    def write(self, data: str) -> int:
        self.writes.append(data)
        return len(data)

    def close(self) -> None:
        self.closed = True

    @property
    def value(self) -> str:
        return "".join(self.writes)


class FakeChild:
    """Minimal ``Popen`` stand-in.

    ``timeout_waits`` is how many leading ``wait()`` calls raise
    ``TimeoutExpired``. 1 models a child that overran the parent's budget and
    then died to SIGTERM; 2 models one that ignored SIGTERM too and has to be
    killed, which is the only way to exercise the escalation.
    """

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        timeout_waits: int = 0,
    ) -> None:
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.stdin = RecordingPipe()
        self.returncode = returncode
        self.terminated = 0
        self.killed = 0
        self.wait_timeouts: List[Optional[float]] = []
        self._timeout_waits = int(timeout_waits)

    def wait(self, timeout: Optional[float] = None) -> int:
        self.wait_timeouts.append(timeout)
        if len(self.wait_timeouts) <= self._timeout_waits:
            raise subprocess.TimeoutExpired("fake-backtest-child", timeout or 0)
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1

    def kill(self) -> None:
        self.killed += 1
