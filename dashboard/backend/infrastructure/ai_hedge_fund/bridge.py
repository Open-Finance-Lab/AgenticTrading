"""Stdlib bridge executed by the isolated AI Hedge Fund interpreter."""

from __future__ import annotations

import argparse
import contextlib
import json
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


def _usage_counter():
    """A context manager that totals every model call's tokens, or ``None``.

    LangChain's ``get_usage_metadata_callback`` registers an inheritable
    context-var handler, so it sees the calls LangGraph runs on its worker
    threads too (the analysts fan out in parallel). Absent in an older
    langchain-core; the caller then reports no usage and ATL bills the
    decision's reservation ceiling instead.
    """
    try:
        from langchain_core.callbacks import get_usage_metadata_callback
    except ImportError:
        return None
    return get_usage_metadata_callback()


def _usage_totals(counter) -> dict | None:
    by_model = getattr(counter, "usage_metadata", None)
    if not isinstance(by_model, dict):
        return None
    totals = {"input_tokens": 0, "output_tokens": 0}
    for usage in by_model.values():
        for key in totals:
            value = usage.get(key) if isinstance(usage, dict) else None
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                totals[key] += value
    return totals


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
    counter = _usage_counter()
    output_path = Path(args.output)
    # Written before upstream can make a model call. A failed exit that leaves
    # only this marker (an OOM SIGKILL, a crash the handler below never saw)
    # spent an unknown amount, and ATL bills it the ceiling; no file at all
    # means the bridge died before any call was possible.
    output_path.write_text(json.dumps({"started": True}), encoding="utf-8")
    handler = None
    try:
        with counter if counter is not None else contextlib.nullcontext() as handler:
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
        # Inside the try, so an output that fails to serialize still leaves
        # the usage report below rather than only the marker.
        output_path.write_text(
            json.dumps(
                {
                    "decisions": result.get("decisions"),
                    "analyst_signals": result.get("analyst_signals"),
                    "usage": _usage_totals(handler),
                }
            ),
            encoding="utf-8",
        )
    except BaseException:
        # A failed decision may still have spent model calls; ATL bills what
        # was used, so report it before the non-zero exit. Usage only: there
        # is no decision to read.
        output_path.write_text(
            json.dumps({"started": True, "usage": _usage_totals(handler)}),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
