#!/usr/bin/env python3
"""Run a trusted local vn.py CtaTemplate strategy in an ATL backtest."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urljoin

from agentictrading import ATLClient, __version__ as atl_sdk_version
from agentictrading.integrations.vnpy_cta import (
    VnpyCtaAdapter,
    VnpyCtaATLRunner,
    VnpyCtaRunSummary,
    VnpyCtaRuntime,
    build_safe_manifest,
    sanitize_error_message,
)


DEFAULT_STRATEGY = "vnpy_cta_double_ma_strategy:DoubleMaStrategy"
REQUIRED_ENVIRONMENT = ("ATL_BASE_URL", "ATL_API_KEY", "ATL_AGENT_VERSION_ID")
DISPLAY_DIAGNOSTICS = (
    "warmup_hold",
    "strategy_hold",
    "decision_submitted",
    "fills",
    "error_hold",
    "unsupported_actions",
    "local_rejections",
    "atl_rejections",
    "timed_out_orders",
    "timeout_hold",
    "fatal_data_error",
    "run_error",
    "terminal_bar_skipped",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local vn.py CTA strategy against ATL hourly AAPL data."
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help="trusted local strategy as module:Class (default: %(default)s)",
    )
    parser.add_argument(
        "--settings-file",
        help="JSON object containing CtaTemplate settings",
    )
    parser.add_argument("--symbol", default="AAPL", help="MVP symbol: AAPL")
    parser.add_argument("--start", required=True, help="start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="end date (YYYY-MM-DD)")
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=None,
        help="ATL starting cash; the current environment accepts its fixed value only",
    )
    parser.add_argument(
        "--output",
        help="audit artifact JSON path (default: ~/.agentictrading/vnpy-cta/runs/)",
    )
    return parser


def validate_inputs(symbol: str, start: str, end: str) -> str:
    normalized_symbol = str(symbol).strip().upper()
    if normalized_symbol != "AAPL":
        raise ValueError("vn.py CTA MVP supports AAPL only")
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError as exc:
        raise ValueError("start and end must use YYYY-MM-DD") from exc
    if end_date < start_date:
        raise ValueError("end date must not be before start date")
    return normalized_symbol


def required_environment(environ: Mapping[str, str]) -> dict[str, str]:
    missing = [name for name in REQUIRED_ENVIRONMENT if not environ.get(name, "").strip()]
    if missing:
        raise ValueError("missing required environment variables: " + ", ".join(missing))
    return {name: environ[name].strip() for name in REQUIRED_ENVIRONMENT}


def load_settings(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}
    settings_path = Path(path).expanduser()
    try:
        value = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read settings JSON: {settings_path}") from exc
    if not isinstance(value, dict):
        raise ValueError("settings file must contain one JSON object")
    return value


def load_strategy_class(spec: str) -> type:
    module_name, separator, class_name = str(spec).strip().partition(":")
    if not separator or not module_name or not class_name or ":" in class_name:
        raise ValueError("strategy must use module:Class format")
    module = importlib.import_module(module_name)
    strategy_class = getattr(module, class_name, None)
    if not isinstance(strategy_class, type):
        raise ValueError(f"strategy class not found: {spec}")
    return strategy_class


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"vnpy-cta-{stamp}-{uuid.uuid4().hex[:8]}.json"
    return Path.home() / ".agentictrading" / "vnpy-cta" / "runs" / name


def build_manifest(
    *, strategy_spec: str, settings: Mapping[str, Any], runtime: Any
) -> dict[str, Any]:
    module_name, _, class_name = strategy_spec.partition(":")
    return build_safe_manifest(
        {
            "strategy": {
                "module": module_name,
                "class": class_name,
                "settings": dict(settings),
            },
            "versions": {
                "agentictrading": atl_sdk_version,
                "vnpy": runtime.bindings.vnpy_version,
                "vnpy_ctastrategy": runtime.bindings.cta_version,
            },
        }
    )


def format_summary(summary: VnpyCtaRunSummary, *, base_url: str) -> str:
    compare_url = summary.compare_url
    result_url = urljoin(base_url.rstrip("/") + "/", compare_url) if compare_url else "not returned"
    lines = [
        "=== vn.py CTA -> ATL result ===",
        f"run_id: {summary.run_id or 'unknown'}",
        f"result_url: {result_url}",
        f"clean: {str(summary.clean).lower()}",
        "metrics:",
    ]
    if summary.metrics:
        lines.extend(f"  {key}: {value}" for key, value in sorted(summary.metrics.items()))
    else:
        lines.append("  none")

    lines.append("diagnostics:")
    shown = set()
    for key in DISPLAY_DIAGNOSTICS:
        lines.append(f"  {key}: {summary.diagnostics.get(key, 0)}")
        shown.add(key)
    for key in sorted(set(summary.diagnostics) - shown):
        lines.append(f"  {key}: {summary.diagnostics[key]}")
    lines.extend(
        [
            f"artifact: {summary.artifact_path}",
            f"artifact_sha256: {summary.artifact_sha256}",
        ]
    )
    return "\n".join(lines)


def run_cli(
    argv: Optional[Sequence[str]] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> VnpyCtaRunSummary:
    args = build_parser().parse_args(argv)
    symbol = validate_inputs(args.symbol, args.start, args.end)
    if args.initial_cash is not None and args.initial_cash <= 0:
        raise ValueError("initial cash must be positive")
    env = required_environment(environ if environ is not None else os.environ)
    settings = load_settings(args.settings_file)
    output = Path(args.output).expanduser() if args.output else default_output_path()

    # Importing this spec executes trusted local code with the current user's rights.
    strategy_class = load_strategy_class(args.strategy)
    runtime = VnpyCtaRuntime(strategy_class, symbol=symbol, settings=settings)
    manifest = build_manifest(
        strategy_spec=args.strategy,
        settings=settings,
        runtime=runtime,
    )
    adapter = VnpyCtaAdapter(runtime, symbol=symbol, manifest=manifest)

    client = ATLClient(base_url=env["ATL_BASE_URL"], api_key=env["ATL_API_KEY"])
    runner = VnpyCtaATLRunner(client, adapter, artifact_path=output)
    summary = runner.run_backtest(
        agent_version_id=env["ATL_AGENT_VERSION_ID"],
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.initial_cash,
    )
    print(format_summary(summary, base_url=env["ATL_BASE_URL"]))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        run_cli(argv)
    except Exception as exc:
        print(f"vn.py CTA backtest failed: {sanitize_error_message(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
