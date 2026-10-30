"""The backtest child's launch and stop path, pinned for Windows on a Linux CI.

PR #653 fixed the one line that kept a Windows server from starting a
dashboard backtest at all (``.venv/bin/python3`` does not exist there), and
in doing so made reachable every POSIX assumption that came after it: a
``signal.SIGKILL`` lookup that raised, pipes decoded in the ANSI code page,
``terminate()`` as a hard kill, a kill label keyed on negative exit codes.
CI never runs Windows, so each case here reproduces the Windows condition
on Linux -- a ``signal`` module without SIGKILL, a ``subprocess`` module
with CREATE_NEW_PROCESS_GROUP, an ``os.name`` of ``nt`` -- rather than
skipping where it does not apply.
"""

from __future__ import annotations

import ast
import io
import os
import re
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import dashboard.backend.api.routers.backtests as bt
import dashboard.backend.domain.backtesting.algo_service as algo_service
import dashboard.backend.paths as paths
from dashboard.backend.infrastructure.ai_hedge_fund import adapter as ahf_adapter
from dashboard.backend.tests._fake_child import FakeChild

_BACKEND_DIR = Path(bt.__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean_slots():
    bt._reset_slots_for_tests()
    yield
    bt._reset_slots_for_tests()


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    path.chmod(0o755)
    return path


def _as_os(monkeypatch, name: str) -> None:
    """``os.name`` as ``paths`` sees it, without changing it for pathlib."""
    monkeypatch.setattr(paths, "os", SimpleNamespace(name=name))


def _run_parent(monkeypatch, child, *, run_id):
    """Drive the real ``run_backtest_background`` against ``child``.

    Returns the Popen call (argv plus kwargs) and the trace labels written.
    """
    session_id = str(uuid.uuid4())
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=None
        )
        is None
    )
    calls = []
    labelled = []

    def fake_popen(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        return child

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        bt.trace_service,
        "fail_trace_if_running",
        lambda rid, code: labelled.append((rid, code)) or True,
    )
    bt.run_backtest_background(
        start_date="2026-01-01",
        end_date="2026-01-02",
        session_id=session_id,
        live_run_id=run_id,
        decision_source="rule_based",
    )
    return calls, labelled


# ===========================================================================
# Which interpreter runs the child
# ===========================================================================


def test_venv_python_probes_this_os_layout_only(tmp_path, monkeypatch):
    """The layout comes from ``os.name``; the file has to actually be there.

    A WSL-built ``.venv`` in a checkout Windows also serves has ``bin/`` and no
    ``Scripts/``, and the reverse: ``.venv`` exists either way, which is all the
    old check looked at. A Windows ``Scripts/python.exe`` is a real file on
    POSIX and still must not be returned there -- it cannot run.
    """
    venv = tmp_path / ".venv"
    assert paths.venv_python(venv) is None  # no venv at all

    windows_exe = _touch(venv / "Scripts" / "python.exe")
    _as_os(monkeypatch, "posix")
    assert paths.venv_python(venv) is None
    posix_exe = _touch(venv / "bin" / "python3")
    assert paths.venv_python(venv) == posix_exe

    _as_os(monkeypatch, "nt")
    assert paths.venv_python(venv) == windows_exe
    windows_exe.unlink()
    assert paths.venv_python(venv) is None  # bin/python3 is not a Windows exe
    msys_exe = _touch(venv / "bin" / "python.exe")
    assert paths.venv_python(venv) == msys_exe


def test_child_python_is_the_server_interpreter_even_beside_a_repo_venv(
    tmp_path, monkeypatch
):
    """The root cause, not the symptom: the server's interpreter already
    imported everything the child needs, so a ``.venv`` in the repo root --
    stale, partial, or built by the other OS -- must not outrank it."""
    venv = tmp_path / ".venv"
    _touch(venv / "bin" / "python3")
    _touch(venv / "Scripts" / "python.exe")
    monkeypatch.setattr(paths, "VENV_DIR", venv)

    assert paths.child_python() == sys.executable


def test_child_python_falls_back_to_the_venv_only_without_sys_executable(
    tmp_path, monkeypatch
):
    venv = tmp_path / ".venv"
    monkeypatch.setattr(paths, "VENV_DIR", venv)
    monkeypatch.setattr(paths, "sys", SimpleNamespace(executable=""))
    _as_os(monkeypatch, "posix")

    with pytest.raises(RuntimeError, match="No Python interpreter"):
        paths.child_python()

    exe = _touch(venv / "bin" / "python3")
    assert paths.child_python() == str(exe)


def test_dashboard_backtest_launches_on_child_python_with_utf8_pipes(monkeypatch):
    """Both halves of the pipe contract, and the interpreter, at the one
    call site that matters: what the real worker hands ``Popen``."""
    monkeypatch.setattr(bt, "child_python", lambda: "/opt/server/python")
    calls, _ = _run_parent(
        monkeypatch, FakeChild(returncode=0, stdout="ok\n"), run_id="agent_win_launch"
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["cmd"][0] == "/opt/server/python"
    # Parent decodes UTF-8 and can never raise on a byte...
    assert (call["encoding"], call["errors"]) == ("utf-8", "replace")
    # ...and the child is told to write UTF-8, overriding whatever the
    # operator's environment said.
    assert call["env"]["PYTHONIOENCODING"] == "utf-8:backslashreplace"


def test_algo_backtest_launches_on_child_python(tmp_path, monkeypatch):
    """The Trading Algo panel shares the resolver instead of keeping a copy
    that already disagreed with the dashboard's (it looked in ``dashboard/.venv``)."""
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")

    monkeypatch.setattr(algo_service, "child_python", lambda: "/opt/server/python")
    monkeypatch.setattr(algo_service.subprocess, "run", fake_run)
    monkeypatch.setattr(algo_service, "algo_status", {"running": True, "error": None})

    algo_service._run_backtest_background(
        tmp_path / "algo.json", "sess", "team", "sub", "2026-01-01", "2026-01-02"
    )

    assert seen["argv"][0] == "/opt/server/python"
    assert algo_service.algo_status["running"] is False


def test_algo_resolver_failure_still_releases_the_single_flight(tmp_path, monkeypatch):
    """Resolved inside the try: a raise outside it skipped the ``finally``
    and left the global ``algo_status`` claiming "running" until a restart."""

    def no_interpreter():
        raise RuntimeError("No Python interpreter for backtest subprocesses")

    monkeypatch.setattr(algo_service, "child_python", no_interpreter)
    monkeypatch.setattr(algo_service, "algo_status", {"running": True, "error": None})

    algo_service._run_backtest_background(
        tmp_path / "algo.json", "sess", "team", "sub", "2026-01-01", "2026-01-02"
    )

    assert algo_service.algo_status["running"] is False
    assert "No Python interpreter" in algo_service.algo_status["error"]


def test_ai_hedge_fund_runtime_is_found_in_either_venv_layout(tmp_path, monkeypatch):
    runtime = tmp_path / ".ai-hedge-fund-venv"
    monkeypatch.setattr(ahf_adapter, "_LOCAL_RUNTIME_DIR", runtime)
    runner = ahf_adapter.AiHedgeFundSubprocessRunner({})

    _as_os(monkeypatch, "nt")
    with pytest.raises(ahf_adapter.AiHedgeFundConfigurationError):
        runner._python_executable()
    windows_exe = _touch(runtime / "Scripts" / "python.exe")
    assert runner._python_executable() == windows_exe

    _as_os(monkeypatch, "posix")
    posix_exe = _touch(runtime / "bin" / "python")
    assert runner._python_executable() == posix_exe


def test_only_paths_spells_out_a_venv_interpreter():
    """One owner for where a venv keeps its interpreter.

    The dashboard runner, the Trading Algo runner and the AI Hedge Fund
    adapter each had their own spelling, and two of them disagreed about both
    the layout and which ``.venv`` to look in.
    """
    spelled = re.compile(r"""["'](?:Scripts|bin)["']\s*/\s*["']python""")
    offenders = [
        str(path.relative_to(_BACKEND_DIR))
        for path in _BACKEND_DIR.rglob("*.py")
        if "tests" not in path.relative_to(_BACKEND_DIR).parts
        and path.name != "paths.py"
        and spelled.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


# ===========================================================================
# Stopping the child on a platform without SIGKILL
# ===========================================================================


def test_grace_escalation_survives_a_signal_module_without_sigkill(monkeypatch):
    """Windows' ``signal`` has no SIGKILL. The lookup used to sit in an
    argument list, outside ``_killpg``'s guard, and raise AttributeError
    from every caller after the handle had already been killed."""
    monkeypatch.delattr(signal, "SIGKILL")
    swept = []
    monkeypatch.setattr(bt, "_resolve_child_pgid", lambda process: 5150)
    monkeypatch.setattr(bt, "_killpg", lambda pgid, sig: swept.append(sig) or True)
    stubborn = FakeChild(timeout_waits=1)

    bt._kill_backtest_process_after_grace(stubborn, grace_seconds=0)

    assert stubborn.killed == 1
    assert swept == []


def test_a_timeout_without_sigkill_still_finalizes_as_timed_out(monkeypatch):
    """The user-visible half: before the guard this run landed in the generic
    arm with "module 'signal' has no attribute 'SIGKILL'" as its error and
    none of the timeout outcome (status, budget, label)."""
    monkeypatch.delattr(signal, "SIGKILL")
    run_id = "agent_win_timeout"

    _, labelled = _run_parent(
        monkeypatch, FakeChild(stdout="run header line\n", timeout_waits=1), run_id=run_id
    )

    assert bt._recent_slots[run_id]["timed_out"] is True
    assert bt._recent_slots[run_id]["error"] is None
    assert labelled == [(run_id, "run_timed_out")]


def test_a_prelaunch_cancel_without_sigkill_raises_cancelled(monkeypatch):
    """Not AttributeError: the worker's generic arm would finalize a slot the
    cancel route already closed and overwrite ``cancelled`` with an error."""
    monkeypatch.delattr(signal, "SIGKILL")
    run_id = "agent_win_prelaunch_cancel"
    session_id = str(uuid.uuid4())
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=None
        )
        is None
    )
    accepted, _ = bt._cancel_backtest_slot(
        live_run_id=run_id, session_id=session_id, user_id=None
    )
    assert accepted is True
    child = FakeChild()
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: child)

    with pytest.raises(bt._BacktestCancelled):
        bt._run_backtest_subprocess(
            ["python", "-c", "pass"],
            cwd=".",
            env={},
            stdin_payload="",
            timeout=60,
            live_run_id=run_id,
        )
    assert child.terminated == 1


# ===========================================================================
# Windows' graceful stop: a console-group break before TerminateProcess
# ===========================================================================


class _WindowsChild(FakeChild):
    pid = 4321

    def __init__(self, *, exited: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._exited = exited

    def poll(self):
        return self.returncode if self._exited else None


@pytest.fixture
def windows_signals(monkeypatch):
    """CTRL_BREAK_EVENT present, as on Windows, with ``os.kill`` recorded."""
    monkeypatch.setattr(signal, "CTRL_BREAK_EVENT", 1, raising=False)
    sent = []
    monkeypatch.setattr(bt.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    return sent


def test_windows_stop_breaks_the_console_group_instead_of_terminating(windows_signals):
    child = _WindowsChild()
    bt._CONSOLE_GROUP_LEADERS.add(child)

    assert bt._signal_backtest_process(child) is True

    assert windows_signals == [(4321, signal.CTRL_BREAK_EVENT)]
    # TerminateProcess would have skipped the child's ``finally``.
    assert child.terminated == 0


def test_windows_break_is_only_sent_to_a_group_this_module_created(windows_signals):
    """An id that leads no console group may be broadcast to the whole
    console, uvicorn included, so an unregistered handle gets the handle
    call and nothing else."""
    child = _WindowsChild()

    assert bt._signal_backtest_process(child) is True

    assert windows_signals == []
    assert child.terminated == 1


def test_windows_break_skips_a_child_that_already_exited(windows_signals):
    child = _WindowsChild(exited=True)
    bt._CONSOLE_GROUP_LEADERS.add(child)

    bt._signal_backtest_process(child)

    assert windows_signals == []
    assert child.terminated == 1


def test_windows_break_falls_back_to_terminate_without_a_console(monkeypatch):
    """GenerateConsoleCtrlEvent needs a console; a service has none."""
    monkeypatch.setattr(signal, "CTRL_BREAK_EVENT", 1, raising=False)

    def no_console(pid, sig):
        raise OSError(6, "The handle is invalid")

    monkeypatch.setattr(bt.os, "kill", no_console)
    child = _WindowsChild()
    bt._CONSOLE_GROUP_LEADERS.add(child)

    assert bt._signal_backtest_process(child) is True
    assert child.terminated == 1


@pytest.mark.parametrize("flag", [0x200, None], ids=["windows", "posix"])
def test_the_child_leads_a_console_group_only_where_one_exists(monkeypatch, flag):
    if flag is None:
        monkeypatch.delattr(subprocess, "CREATE_NEW_PROCESS_GROUP", raising=False)
    else:
        monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", flag, raising=False)
    seen = {}
    child = FakeChild()

    def fake_popen(cmd, **kwargs):
        seen.update(kwargs)
        return child

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    bt._run_backtest_subprocess(
        ["python", "-c", "pass"],
        cwd=".",
        env={},
        stdin_payload="",
        timeout=60,
        live_run_id=None,
    )

    # A non-zero creationflags is refused off Windows, so 0 there.
    assert seen["creationflags"] == (flag or 0)
    assert (child in bt._CONSOLE_GROUP_LEADERS) is bool(flag)


def test_the_child_script_unwinds_on_sigbreak_where_it_exists():
    """The receiving end of the break. Without a handler the child dies
    where it stands, which is no better than TerminateProcess."""
    script = (_BACKEND_DIR.parent / "scripts" / "backtest_hourly_agent.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(script)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
    )
    body = ast.unparse(main)
    assert 'if hasattr(signal, "SIGBREAK"):' in body.replace("'", '"')
    assert "signal.signal(signal.SIGBREAK, _exit_on_sigterm)" in body


# ===========================================================================
# Encoding on the pipes and on the parent's own stdout
# ===========================================================================


def test_undecodable_child_bytes_cannot_stop_the_reader():
    """A real child writing a byte UTF-8 cannot decode, then more output.

    Under a strict decode the reader thread swallowed the error as a closed
    pipe and stopped, which on a chatty child means a full pipe and a hang.
    The tail must still arrive.
    """
    write = "import sys; sys.stdout.buffer.write(b'\\xe2\\x8f\\xb1 phase\\n\\xff tail\\n')"
    outcome = bt._run_backtest_subprocess(
        [sys.executable, "-c", write],
        cwd=".",
        env=dict(os.environ),
        stdin_payload="",
        timeout=60,
        live_run_id=None,
    )

    assert outcome.returncode == 0
    assert "⏱ phase" in outcome.stdout
    assert "� tail" in outcome.stdout


def test_unencodable_server_output_degrades_instead_of_raising():
    from dashboard.backend import app as app_module

    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    with pytest.raises(UnicodeEncodeError):
        print("\U0001f680 Background", file=stream)

    app_module._never_fail_on_unencodable_output([stream])
    print("\U0001f680 Background", file=stream)
    stream.flush()

    assert stream.buffer.getvalue().endswith(b"\\U0001f680 Background\n")


def test_app_applies_the_stdio_guard_at_import():
    source = (_BACKEND_DIR / "app.py").read_text(encoding="utf-8")
    calls = [
        node.value.func.id
        for node in ast.parse(source).body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    ]
    assert "_never_fail_on_unencodable_output" in calls


# ===========================================================================
# Labelling a child some other process stopped
# ===========================================================================


@pytest.mark.parametrize(
    ("returncode", "sigbreak", "killed"),
    [
        (-9, None, True),  # POSIX signal death: a cgroup OOM SIGKILL
        (128 + 15, None, True),  # the child's SIGTERM handler, via SystemExit
        (128 + 21, 21, True),  # its SIGBREAK handler, on Windows
        (128 + 21, None, False),  # 149 means nothing where SIGBREAK does not exist
        (1, 21, False),  # TerminateProcess: indistinguishable from a crash
        (0, 21, False),
    ],
)
def test_stopped_by_signal(monkeypatch, returncode, sigbreak, killed):
    if sigbreak is None:
        monkeypatch.delattr(signal, "SIGBREAK", raising=False)
    else:
        monkeypatch.setattr(signal, "SIGBREAK", sigbreak, raising=False)

    assert bt._stopped_by_signal(returncode) is killed


def test_a_child_its_handler_unwound_is_labelled_killed(monkeypatch):
    """A sign check alone read the handler's ``128 + N`` exit as a failure,
    and on Windows it is the only way a stop signal shows up at all."""
    run_id = "agent_win_handled_stop"
    _, labelled = _run_parent(
        monkeypatch, FakeChild(returncode=128 + 15), run_id=run_id
    )
    assert labelled == [(run_id, "run_killed")]
