"""Stdlib bridge executed by the isolated AI Hedge Fund interpreter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _disable_dotenv_loading() -> None:
    """Prevent the pinned upstream package from discovering checkout secrets.

    Upstream pins python-dotenv 1.0.0, which predates support for
    ``PYTHON_DOTENV_DISABLED``. Patch both public import locations before any
    upstream module is imported; its normal ``from dotenv import load_dotenv``
    then receives this no-op.
    """
    import dotenv
    from dotenv import main as dotenv_main

    def disabled_load_dotenv(*_args, **_kwargs) -> bool:
        return False

    dotenv.load_dotenv = disabled_load_dotenv
    dotenv_main.load_dotenv = disabled_load_dotenv


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
    result = run_hedge_fund(
        tickers=payload["tickers"],
        start_date=payload["start_date"],
        end_date=payload["end_date"],
        portfolio=payload["portfolio"],
        show_reasoning=bool(payload.get("show_reasoning", False)),
        selected_analysts=list(payload.get("selected_analysts") or []),
        model_name=str(payload.get("model_name") or "gpt-4.1"),
        model_provider=str(payload.get("model_provider") or "OpenAI"),
    )
    Path(args.output).write_text(
        json.dumps({"decisions": result.get("decisions")}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
