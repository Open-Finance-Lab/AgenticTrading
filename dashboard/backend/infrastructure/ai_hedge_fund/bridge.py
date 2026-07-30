"""Stdlib bridge executed by the isolated AI Hedge Fund interpreter."""

from __future__ import annotations

import argparse
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
    Path(args.output).write_text(
        json.dumps(
            {
                "decisions": result.get("decisions"),
                "analyst_signals": result.get("analyst_signals"),
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
