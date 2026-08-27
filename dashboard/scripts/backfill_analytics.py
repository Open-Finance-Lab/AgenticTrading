#!/usr/bin/env python3
"""Reconstruct up to 180 days of authoritative Analytics events."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from _bootstrap import ensure_repo_root


def _before(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--before must be an ISO-8601 timestamp with a timezone"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError(
            "--before must be an ISO-8601 timestamp with a timezone"
        )
    return parsed.astimezone(timezone.utc)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill safe server-authoritative Analytics history.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="History window in days (1-180; default: 180).",
    )
    parser.add_argument(
        "--before",
        type=_before,
        help="Inclusive ISO-8601 cutoff with timezone (default: now).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count without writing events, snapshots, or rollups.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ensure_repo_root()
    from dashboard.backend.domain.analytics.backfill import backfill_analytics

    try:
        report = backfill_analytics(
            days=args.days,
            before=args.before,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        _parser().error(str(exc))
        return 2
    safe_counts = report.model_dump(exclude={"source_event_ids"})
    print(json.dumps(safe_counts, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
