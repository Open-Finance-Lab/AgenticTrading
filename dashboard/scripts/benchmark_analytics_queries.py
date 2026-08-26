#!/usr/bin/env python3
"""Benchmark Admin Analytics queries on deterministic disposable SQLite data."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from _bootstrap import ensure_repo_root


BENCHMARK_NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
OVERVIEW_P95_LIMIT_SECONDS = 1.0
PROFILE_P95_LIMIT_SECONDS = 0.5


def _bounded_integer(name: str, minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be from {minimum} through {maximum}"
            )
        return parsed

    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Admin Analytics overview and initial profile queries "
            "using deterministic disposable SQLite data."
        ),
    )
    parser.add_argument(
        "--users",
        type=_bounded_integer("--users", 1, 100_000),
        default=10_000,
    )
    parser.add_argument(
        "--events-per-user",
        type=_bounded_integer("--events-per-user", 1, 1_000),
        default=40,
    )
    parser.add_argument(
        "--days",
        type=_bounded_integer("--days", 1, 180),
        default=180,
    )
    parser.add_argument(
        "--iterations",
        type=_bounded_integer("--iterations", 1, 100),
        default=20,
    )
    return parser


def _event_id(sequence: int) -> str:
    return f"10000000-0000-4000-8000-{sequence:012x}"


def _session_id(user_id: int) -> str:
    return f"20000000-0000-4000-8000-{user_id:012x}"


def _event_shape(index: int) -> tuple[str, str]:
    return (
        ("page_viewed", "experience"),
        ("backtest_completed", "run"),
        ("backtest_failed", "run"),
        ("model_usage_recorded", "resource"),
        ("credits_settled", "resource"),
        ("credential_verified", "credential"),
        ("agent_created", "agent"),
        ("backtest_started", "run"),
    )[index % 8]


def _source_record_type(group: str) -> str:
    return {
        "run": "run",
        "resource": "run",
        "credential": "credential",
        "agent": "agent",
    }[group]


def _seed(path: Path, *, users: int, events_per_user: int, days: int) -> None:
    from dashboard.backend.domain.analytics.repository import (
        AnalyticsStore,
        _EVENT_COLUMNS,
    )
    from dashboard.backend.users import UserStore

    UserStore(path)
    AnalyticsStore(path)
    now_iso = BENCHMARK_NOW.isoformat()
    day_users: dict[str, set[int]] = defaultdict(set)
    completed_by_day: Counter[str] = Counter()
    failed_by_day: Counter[str] = Counter()
    platform_cost_by_day: Counter[str] = Counter()
    input_tokens_by_day: Counter[str] = Counter()
    output_tokens_by_day: Counter[str] = Counter()

    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("PRAGMA synchronous = OFF")
        conn.executemany(
            """
            INSERT INTO users (
                id, email, display_name, password_hash, role, created_at
            ) VALUES (?, ?, ?, 'benchmark-only', 'user', ?)
            """,
            (
                (
                    user_id,
                    f"user-{user_id:05d}@example.test",
                    f"Benchmark User {user_id:05d}",
                    (BENCHMARK_NOW - timedelta(days=user_id % days)).isoformat(),
                )
                for user_id in range(1, users + 1)
            ),
        )
        conn.executemany(
            """
            INSERT INTO user_analytics_snapshots (
                user_id, status, reason_code, human_readable_reason,
                evidence_event_ids_json, calculated_at
            ) VALUES (?, ?, ?, ?, '[]', ?)
            """,
            (
                (
                    user_id,
                    "needs_attention" if user_id % 20 == 0 else "active",
                    (
                        "invalid_default_credential"
                        if user_id % 20 == 0
                        else "recent_successful_run_and_activity"
                    ),
                    (
                        "The default model credential is invalid."
                        if user_id % 20 == 0
                        else "A successful run and recent meaningful activity are present."
                    ),
                    now_iso,
                )
                for user_id in range(1, users + 1)
            ),
        )

        placeholders = ", ".join("?" for _ in _EVENT_COLUMNS)
        insert_event = (
            f"INSERT INTO analytics_events ({', '.join(_EVENT_COLUMNS)}) "
            f"VALUES ({placeholders})"
        )
        batch: list[tuple[object, ...]] = []
        sequence = 0
        for user_id in range(1, users + 1):
            for index in range(events_per_user):
                sequence += 1
                event_name, event_group = _event_shape(index)
                day_offset = (user_id * 17 + index * 29) % days
                day = BENCHMARK_NOW.date() - timedelta(days=day_offset)
                seconds = (user_id * 131 + index * 977) % (12 * 60 * 60)
                occurred_at = datetime.combine(
                    day,
                    datetime_time.min,
                    tzinfo=timezone.utc,
                ) + timedelta(seconds=seconds)
                day_key = day.isoformat()
                day_users[day_key].add(user_id)

                is_frontend = event_group == "experience"
                source_record_id = f"{user_id}-{index}"
                source_event_id = (
                    None
                    if is_frontend
                    else f"benchmark:{event_name}:{source_record_id}"
                )
                record_type = (
                    None if is_frontend else _source_record_type(event_group)
                )
                correlation_id = (
                    f"benchmark-run-{user_id}-{index // 8}"
                    if event_group in {"run", "resource"}
                    else None
                )
                provider_id = None
                model_id = None
                billing_mode = None
                outcome = None
                error_category = None
                properties = "{}"
                if event_name == "backtest_completed":
                    outcome = "succeeded"
                    completed_by_day[day_key] += 1
                elif event_name == "backtest_failed":
                    outcome = "failed"
                    error_category = "provider_timeout"
                    failed_by_day[day_key] += 1
                elif event_name == "model_usage_recorded":
                    provider_id = "openrouter"
                    model_id = "openai/gpt-5.5"
                    billing_mode = "platform_credits"
                    outcome = "succeeded"
                    input_tokens = 100 + index
                    output_tokens = 25 + index
                    cost_micro = 1_000 + index
                    input_tokens_by_day[day_key] += input_tokens
                    output_tokens_by_day[day_key] += output_tokens
                    platform_cost_by_day[day_key] += cost_micro
                    properties = json.dumps(
                        {
                            "cost_micro_usd": cost_micro,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                elif event_name == "credits_settled":
                    billing_mode = "platform_credits"
                    properties = '{"amount_micro":100,"bucket":"grant"}'
                elif event_name == "credential_verified":
                    provider_id = "openrouter"

                batch.append(
                    (
                        _event_id(sequence),
                        1,
                        event_name,
                        event_group,
                        user_id,
                        _session_id(user_id) if is_frontend else None,
                        occurred_at.isoformat(),
                        (occurred_at + timedelta(seconds=1)).isoformat(),
                        "frontend" if is_frontend else "server",
                        source_event_id,
                        record_type,
                        source_record_id if not is_frontend else None,
                        correlation_id,
                        "agents" if is_frontend else None,
                        provider_id,
                        model_id,
                        billing_mode,
                        outcome,
                        error_category,
                        "US" if is_frontend else None,
                        "desktop" if is_frontend else None,
                        "Chrome" if is_frontend else None,
                        None,
                        properties,
                    )
                )
                if len(batch) == 5_000:
                    conn.executemany(insert_event, batch)
                    batch.clear()
        if batch:
            conn.executemany(insert_event, batch)

        rollup_rows: list[tuple[object, ...]] = []
        for day_key in sorted(day_users):
            dimensions = ("", "", "", "", "", "")
            rollup_rows.extend(
                [
                    (
                        day_key,
                        "daily_active_users",
                        *dimensions,
                        "",
                        len(day_users[day_key]),
                        0,
                        now_iso,
                    ),
                    (
                        day_key,
                        "terminal_completed",
                        *dimensions,
                        "",
                        completed_by_day[day_key],
                        0,
                        now_iso,
                    ),
                    (
                        day_key,
                        "terminal_failed",
                        *dimensions,
                        "",
                        failed_by_day[day_key],
                        0,
                        now_iso,
                    ),
                    (
                        day_key,
                        "input_tokens",
                        *dimensions,
                        "",
                        input_tokens_by_day[day_key],
                        0,
                        now_iso,
                    ),
                    (
                        day_key,
                        "output_tokens",
                        *dimensions,
                        "",
                        output_tokens_by_day[day_key],
                        0,
                        now_iso,
                    ),
                    (
                        day_key,
                        "platform_model_cost_usd",
                        "",
                        "platform_credits",
                        "",
                        "",
                        "",
                        "",
                        "",
                        0,
                        platform_cost_by_day[day_key],
                        now_iso,
                    ),
                ]
            )
        conn.executemany(
            """
            INSERT INTO analytics_daily_rollups (
                rollup_date, metric_name, event_name, billing_mode,
                provider_id, model_id, outcome, error_category, user_state,
                value_count, value_sum_micro, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rollup_rows,
        )


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]


def _measure(callback, iterations: int) -> list[float]:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        callback()
        samples.append(time.perf_counter() - started)
    return samples


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ensure_repo_root()
    with TemporaryDirectory(prefix="analytics-query-benchmark-") as tmpdir:
        path = Path(tmpdir) / "analytics-benchmark.db"
        env_names = (
            "DATABASE_PATH",
            "USERS_DATABASE_URL",
            "CONTENT_DATABASE_URL",
            "AGENT_RUNS_DATABASE_URL",
        )
        previous_env = {name: os.environ.get(name) for name in env_names}
        os.environ["DATABASE_PATH"] = str(path)
        for name in env_names[1:]:
            os.environ.pop(name, None)
        try:
            from dashboard.backend.domain.analytics.metrics import (
                AnalyticsMetricFilters,
            )
            from dashboard.backend.domain.analytics.query_service import (
                AnalyticsQueryService,
            )
            from dashboard.backend.domain.analytics.repository import AnalyticsStore
            from dashboard.backend.users import UserStore

            seed_started = time.perf_counter()
            _seed(
                path,
                users=args.users,
                events_per_user=args.events_per_user,
                days=args.days,
            )
            seed_seconds = time.perf_counter() - seed_started
            service = AnalyticsQueryService(
                store=AnalyticsStore(path),
                user_store=UserStore(path),
            )
            filters = AnalyticsMetricFilters(
                start=BENCHMARK_NOW - timedelta(days=min(args.days, 30)),
                end=BENCHMARK_NOW + timedelta(seconds=1),
            )
            profile_user_id = max(1, args.users // 2)

            service.get_overview(filters=filters, now=BENCHMARK_NOW)
            service.get_user_profile(user_id=profile_user_id, now=BENCHMARK_NOW)
            overview_samples = _measure(
                lambda: service.get_overview(
                    filters=filters,
                    now=BENCHMARK_NOW,
                ),
                args.iterations,
            )
            profile_samples = _measure(
                lambda: service.get_user_profile(
                    user_id=profile_user_id,
                    now=BENCHMARK_NOW,
                ),
                args.iterations,
            )
        finally:
            for name, value in previous_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    overview_p50 = _percentile(overview_samples, 0.50)
    overview_p95 = _percentile(overview_samples, 0.95)
    profile_p50 = _percentile(profile_samples, 0.50)
    profile_p95 = _percentile(profile_samples, 0.95)
    passed = (
        overview_p95 <= OVERVIEW_P95_LIMIT_SECONDS
        and profile_p95 <= PROFILE_P95_LIMIT_SECONDS
    )
    print(
        json.dumps(
            {
                "days": args.days,
                "events": args.users * args.events_per_user,
                "events_per_user": args.events_per_user,
                "iterations": args.iterations,
                "overview": {
                    "p50_seconds": round(overview_p50, 6),
                    "p95_seconds": round(overview_p95, 6),
                    "target_p95_seconds": OVERVIEW_P95_LIMIT_SECONDS,
                },
                "passed": passed,
                "profile": {
                    "p50_seconds": round(profile_p50, 6),
                    "p95_seconds": round(profile_p95, 6),
                    "target_p95_seconds": PROFILE_P95_LIMIT_SECONDS,
                },
                "seed_seconds": round(seed_seconds, 3),
                "users": args.users,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
