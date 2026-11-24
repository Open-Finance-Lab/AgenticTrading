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

The child must be in the job before it runs any code: only processes created
after the assignment join, and a child assigned after ``Popen`` returns may
already have started its own children (a Windows venv's ``Scripts\\python.exe``
is a launcher that starts the base interpreter at once). So the caller starts
the child with ``SUSPENDED_START_FLAG`` and hands it to ``start_in_kill_job``,
which assigns it and only then lets it run (#717).

Off Windows every function is a no-op: ``SUSPENDED_START_FLAG`` is 0 and
``start_in_kill_job`` returns None.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

_IS_WINDOWS = sys.platform == "win32"
# Exit code given to members still alive at the sweep. 1 matches
# ``Popen.kill()`` on Windows, so the run's outcome is labelled as before.
_TERMINATED_EXIT_CODE = 1
# CREATE_SUSPENDED: the child's first thread waits until ``start_in_kill_job``
# resumes it. 0 off Windows, where ``Popen`` refuses any creation flag.
SUSPENDED_START_FLAG = 0x00000004 if _IS_WINDOWS else 0

_kernel32: Any = None
_ntdll: Any = None
# One warning per failure kind, so a harmless first refusal cannot hide a
# later, different failure for the life of the server.
_warned: set = set()


def _api() -> Any:
    """kernel32 with typed signatures, loaded on first use (Windows only)."""
    global _kernel32
    if _kernel32 is None:
        import ctypes.wintypes

        wintypes = ctypes.wintypes
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


def _nt() -> Any:
    """ntdll with ``NtResumeProcess`` typed, loaded on first use (Windows only).

    ``Popen`` closes the child's thread handle, so ``ResumeThread`` has nothing
    to act on; ``NtResumeProcess`` resumes through the process handle it keeps.
    Undocumented but stable since XP, and what ``psutil.Process.resume`` uses.
    """
    global _ntdll
    if _ntdll is None:
        import ctypes.wintypes

        ntdll = ctypes.WinDLL("ntdll")
        ntdll.NtResumeProcess.argtypes = (ctypes.wintypes.HANDLE,)
        ntdll.NtResumeProcess.restype = ctypes.c_long  # NTSTATUS; 0 is success
        _ntdll = ntdll
    return _ntdll


def _warn_once(kind: str, message: str) -> None:
    if kind not in _warned:
        _warned.add(kind)
        print(f"WARNING: process_jobs: {message}", flush=True)


def assign_kill_job(process: Any) -> Optional[int]:
    """Put a just-started child in a new job; the job handle, or None.

    None off Windows, for an object without a process handle (a test stub),
    and when Windows refuses the assignment (a parent job that forbids nesting,
    before Windows 8). The caller then keeps its old escalation,
    ``Popen.kill``, which reaches the child alone.

    Only processes the child creates after this call join the job, so on its
    own this races the child's start-up. Launch callers go through
    ``start_in_kill_job`` with a suspended child, which closes that window.

    Never raises: any failure falls back to None, so a launched child is
    always handed back to the caller that owns it.
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
            _warn_once(
                "create", "CreateJobObjectW failed; a stopped run's grandchildren may survive"
            )
            return None
        if not api.AssignProcessToJobObject(job, int(handle)):
            api.CloseHandle(job)
            # A child that already exited (a bad argument, an import error)
            # is refused too, and has no tree left to lose: not a warning.
            if process.poll() is None:
                _warn_once(
                    "assign",
                    "AssignProcessToJobObject was refused; a stopped run's "
                    "grandchildren may survive",
                )
            return None
        return int(job)
    except Exception as exc:  # incl. ctypes.ArgumentError: never strand the child
        _warn_once(
            "unavailable",
            f"job objects unavailable ({exc}); a stopped run's grandchildren may survive",
        )
        return None


def start_in_kill_job(process: Any) -> Optional[int]:
    """Assign a child started with ``SUSPENDED_START_FLAG`` to a job, then resume it.

    Returns the job handle, or None when there is no job (as ``assign_kill_job``);
    the child is resumed either way, since a refused assignment only costs the
    sweep, not the run. Assigning first means everything the child ever starts
    is in the job, however fast it starts it (#717).

    Raises OSError if a live child cannot be resumed, after killing it: a child
    left suspended would hold its slot silently until the run's timeout.
    """
    job = assign_kill_job(process)
    if not _IS_WINDOWS:
        return job
    handle = getattr(process, "_handle", None)
    if handle is None:
        return job
    try:
        status = _nt().NtResumeProcess(int(handle))
        failure = None if status == 0 else f"NTSTATUS 0x{status & 0xFFFFFFFF:08X}"
    except Exception as exc:  # a missing export or bad handle: same outcome
        failure = str(exc)
    if failure is None:
        return job
    if process.poll() is not None:
        # Already gone (killed from outside while suspended): nothing to hang.
        # Its exit code tells the run what happened, better than this would.
        return job
    try:
        process.kill()  # TerminateProcess works on a suspended process
    except OSError:
        pass
    close_job(job)
    raise OSError(f"could not resume the backtest child ({failure}); it was killed")


def terminate_job(job: Optional[int]) -> bool:
    """End every process still in ``job``. True when Windows accepted the call."""
    if job is None or not _IS_WINDOWS:
        return False
    try:
        return bool(_api().TerminateJobObject(job, _TERMINATED_EXIT_CODE))
    except Exception:  # the sweep is best effort; the stop path must finish
        return False


def close_job(job: Optional[int]) -> None:
    """Release the job handle. Members keep running: the job has no kill-on-close."""
    if job is None or not _IS_WINDOWS:
        return
    try:
        _api().CloseHandle(job)
    except Exception:  # runs from a finalizer: nothing left to report to
        pass


__all__ = [
    "SUSPENDED_START_FLAG",
    "assign_kill_job",
    "close_job",
    "start_in_kill_job",
    "terminate_job",
]
