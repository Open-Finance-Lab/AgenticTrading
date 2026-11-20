"""Windows Job Objects: one handle that can kill a child and everything it started.

POSIX stops a backtest tree with ``killpg`` on the child's process group, and
that reaches a grandchild even after the child itself has exited. Windows has
no process groups to kill. ``Popen.kill`` is TerminateProcess on one pid, and
``taskkill /T`` walks the tree from a live parent, so neither reaches a
grandchild whose parent already exited -- the case that matters, since the
child obeys CTRL_BREAK and exits inside the grace period while a grandchild
that ignored it keeps running (#688).

A Job Object is the Windows equivalent of that group: a process assigned to a
job stays in it, every process it creates afterwards joins it, and
``TerminateJobObject`` ends every member that is still alive, whether or not
the process that was assigned has exited.

The job is created without ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. That flag
would also kill the tree whenever the handle closes: at the end of a normal
run, and when the web server restarts (``uvicorn --reload``). POSIX does
neither, so the tree is ended only by an explicit ``terminate_job``, on the
same stop path where POSIX runs ``killpg``.

Off Windows every function is a no-op: ``assign_kill_job`` returns None.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

_IS_WINDOWS = sys.platform == "win32"
# Exit code given to members still alive at the sweep. 1 matches
# ``Popen.kill()`` on Windows, so the run's outcome is labelled as before.
_TERMINATED_EXIT_CODE = 1

_kernel32: Any = None
_warned = False


def _api() -> Any:
    """kernel32 with typed signatures, loaded on first use (Windows only)."""
    global _kernel32
    if _kernel32 is None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        _kernel32 = kernel32
    return _kernel32


def _warn_once(message: str) -> None:
    global _warned
    if not _warned:
        _warned = True
        print(f"WARNING: process_jobs: {message}", flush=True)


def assign_kill_job(process: Any) -> Optional[int]:
    """Put a just-started child in a new job; the job handle, or None.

    None off Windows, for an object without a process handle (a test stub),
    and when Windows refuses the assignment (a parent job that forbids nesting,
    before Windows 8). The caller then keeps its old escalation,
    ``Popen.kill``, which reaches the child alone.

    Only processes the child creates after this call join the job. The window
    is the child interpreter's start-up, which spawns nothing.
    """
    if not _IS_WINDOWS:
        return None
    handle = getattr(process, "_handle", None)
    if handle is None:
        return None
    try:
        api = _api()
        job = api.CreateJobObjectW(None, None)
        if not job:
            _warn_once("CreateJobObjectW failed; a stopped run's grandchildren may survive")
            return None
        if not api.AssignProcessToJobObject(job, int(handle)):
            api.CloseHandle(job)
            _warn_once(
                "AssignProcessToJobObject was refused; a stopped run's "
                "grandchildren may survive"
            )
            return None
        return int(job)
    except (OSError, AttributeError, ValueError, TypeError) as exc:
        _warn_once(f"job objects unavailable ({exc}); a stopped run's grandchildren may survive")
        return None


def terminate_job(job: Optional[int]) -> bool:
    """End every process still in ``job``. True when Windows accepted the call."""
    if job is None or not _IS_WINDOWS:
        return False
    try:
        return bool(_api().TerminateJobObject(job, _TERMINATED_EXIT_CODE))
    except (OSError, AttributeError, ValueError, TypeError):
        return False


def close_job(job: Optional[int]) -> None:
    """Release the job handle. Members keep running: the job has no kill-on-close."""
    if job is None or not _IS_WINDOWS:
        return
    try:
        _api().CloseHandle(job)
    except (OSError, AttributeError, ValueError, TypeError):
        pass


__all__ = ["assign_kill_job", "close_job", "terminate_job"]
