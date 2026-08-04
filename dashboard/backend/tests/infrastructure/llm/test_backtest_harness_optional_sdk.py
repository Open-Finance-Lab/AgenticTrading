"""The optional Anthropic SDK import must degrade, never crash the import.

``backtest_harness`` is on the backend's import path, so anything that escapes
its optional-dependency handler does not merely disable LLM trading -- it takes
the whole app's import down with it. The module binds the client class by
attribute (``anthropic.Anthropic``) so the name is visibly *used* for
``py/unused-import``; that form raises ``AttributeError`` where the older
``from anthropic import Anthropic`` raised ``ImportError``, and only the latter
was handled.

Run in a subprocess, like ``test_canonical_consumers`` does: the SDK stub has to
be installed before the module is first imported, and reloading it in-process
would rebind functions that other suites assert the identity of.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]

_HARNESS = "dashboard.backend.infrastructure.llm.backtest_harness"


def _import_with_sdk_stub(body: str) -> dict:
    """Import the harness with ``sys.modules['anthropic']`` replaced by a stub."""
    code = (
        "import sys, types, json\n"
        "stub = types.ModuleType('anthropic')\n"
        f"{body}\n"
        "sys.modules['anthropic'] = stub\n"
        f"import importlib; h = importlib.import_module({_HARNESS!r})\n"
        "print(json.dumps({'has': h.HAS_ANTHROPIC, 'client_is_none': h.Anthropic is None}))\n"
    )
    with tempfile.TemporaryDirectory(prefix="atl_sdk_") as tmp:
        env = {**os.environ, "DATABASE_PATH": os.path.join(tmp, "t.db")}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
    assert proc.returncode == 0, f"importing the harness failed:\n{proc.stderr}"
    import json

    last = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")][-1]
    return json.loads(last)


def test_sdk_without_the_client_class_degrades():
    """An `anthropic` package that imports but exposes no `Anthropic`."""
    res = _import_with_sdk_stub("pass  # stub deliberately has no .Anthropic")

    assert res["has"] is False
    assert res["client_is_none"] is True


def test_sdk_with_the_client_class_is_detected():
    """The success path still binds the real class and flips the flag."""
    res = _import_with_sdk_stub("stub.Anthropic = type('Anthropic', (), {})")

    assert res["has"] is True
    assert res["client_is_none"] is False
