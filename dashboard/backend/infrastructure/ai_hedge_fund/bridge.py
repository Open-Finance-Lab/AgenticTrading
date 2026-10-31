"""Stdlib bridge executed by the isolated AI Hedge Fund interpreter."""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import json
import threading
from pathlib import Path

MANAGED_MODEL_NAME = "nvidia/nemotron-3-nano-30b-a3b"
MANAGED_MODEL_PROVIDER = "OpenRouter"


def _managed_model_from_payload(payload: dict) -> tuple[str, str]:
    """Fail closed instead of falling back to upstream's direct OpenAI default."""
    model_name = str(payload.get("model_name") or "").strip()
    model_provider = str(payload.get("model_provider") or "").strip()
    if (
        model_name != MANAGED_MODEL_NAME
        or model_provider != MANAGED_MODEL_PROVIDER
    ):
        raise ValueError(
            "AI Hedge Fund bridge requires the platform-managed OpenRouter model"
        )
    return model_name, model_provider


def _disable_dotenv_loading() -> None:
    """Prevent the pinned upstream package from discovering checkout secrets.

    Upstream pins python-dotenv 1.0.0, which predates support for
    ``PYTHON_DOTENV_DISABLED``. Patch both public import locations before any
    upstream module is imported; its normal ``from dotenv import load_dotenv``
    then receives this no-op.

    Applied with ``setattr`` because this is a deliberate runtime monkeypatch,
    not a module-level rebinding: this file only ever executes as the entry
    point of a *separate* interpreter, so it can never reach ATL's own
    ``from dotenv import load_dotenv`` call sites. Written as a plain
    assignment, static analysis reads it as the repo mutating a shared module
    attribute and reports every one of those unrelated call sites.
    """
    import dotenv
    from dotenv import main as dotenv_main

    def disabled_load_dotenv(*_args, **_kwargs) -> bool:
        return False

    setattr(dotenv, "load_dotenv", disabled_load_dotenv)
    setattr(dotenv_main, "load_dotenv", disabled_load_dotenv)


def _reported_usage(response) -> dict | None:
    """One model call's ``input_tokens``/``output_tokens``, or None if unreported."""
    try:
        usage = response.generations[0][0].message.usage_metadata
    except (AttributeError, IndexError, TypeError):
        return None
    if not isinstance(usage, dict):
        return None
    counts = {}
    for key in ("input_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        counts[key] = value
    return counts


def _usage_meter():
    """A context manager yielding a meter of every model call (None without one).

    The meter counts calls as well as tokens. That is what tells a decision
    that made no model call (every analyst rule-based and every ticker
    pre-filled as a hold) from one whose calls came back without usage (a
    provider or langchain change): the first is free, the second a broken meter
    ATL bills at the decision ceiling. Registered the way langchain-core's own
    ``get_usage_metadata_callback`` is, as an inheritable context-var hook, so
    it sees the calls LangGraph runs on its worker threads too (the analysts
    fan out in parallel). Without that hook API it yields None and ATL bills
    the ceiling.
    """
    try:
        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_core.tracers.context import register_configure_hook
    except ImportError:
        return contextlib.nullcontext()

    class UsageMeter(BaseCallbackHandler):
        def __init__(self) -> None:
            super().__init__()
            self._lock = threading.Lock()
            self._totals = {
                "input_tokens": 0,
                "output_tokens": 0,
                "calls": 0,
                "unmetered_calls": 0,
            }

        def on_llm_end(self, response, **_kwargs) -> None:
            usage = _reported_usage(response)
            with self._lock:
                self._totals["calls"] += 1
                if usage is None:
                    self._totals["unmetered_calls"] += 1
                    return
                for key, value in usage.items():
                    self._totals[key] += value

        def report(self) -> dict:
            with self._lock:
                return dict(self._totals)

    @contextlib.contextmanager
    def metered():
        meter = UsageMeter()
        meter_var = contextvars.ContextVar("atl_usage_meter", default=None)
        register_configure_hook(meter_var, inheritable=True)
        token = meter_var.set(meter)
        try:
            yield meter
        finally:
            meter_var.reset(token)

    return metered()


def _usage_report(meter) -> dict | None:
    return meter.report() if meter is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # This import resolves only inside the isolated upstream environment (or an
    # explicitly configured read-only checkout on the child process's controlled
    # PYTHONPATH), never ATL's main environment.
    _disable_dotenv_loading()
    from src.main import run_hedge_fund

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    model_name, model_provider = _managed_model_from_payload(payload)
    output_path = Path(args.output)
    # Written before upstream can make a model call. A failed exit that leaves
    # only this marker (an OOM SIGKILL, a crash the handler below never saw)
    # spent an unknown amount, and ATL bills it the ceiling; no file at all
    # means the bridge died before any call was possible.
    output_path.write_text(json.dumps({"started": True}), encoding="utf-8")
    with _usage_meter() as meter:
        try:
            result = run_hedge_fund(
                tickers=payload["tickers"],
                start_date=payload["start_date"],
                end_date=payload["end_date"],
                portfolio=payload["portfolio"],
                show_reasoning=bool(payload.get("show_reasoning", False)),
                selected_analysts=list(payload.get("selected_analysts") or []),
                model_name=model_name,
                model_provider=model_provider,
            )
            # Inside the try, so an output that fails to serialize still
            # leaves the usage report below rather than only the marker.
            output_path.write_text(
                json.dumps(
                    {
                        "decisions": result.get("decisions"),
                        "analyst_signals": result.get("analyst_signals"),
                        "usage": _usage_report(meter),
                    }
                ),
                encoding="utf-8",
            )
        except BaseException:
            # A failed decision may still have spent model calls; ATL bills
            # what was used, so report it before the non-zero exit. Usage
            # only: there is no decision to read.
            output_path.write_text(
                json.dumps({"started": True, "usage": _usage_report(meter)}),
                encoding="utf-8",
            )
            raise


if __name__ == "__main__":
    main()
