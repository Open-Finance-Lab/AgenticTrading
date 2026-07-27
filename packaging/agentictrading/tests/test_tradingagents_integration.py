"""Tests for the optional TradingAgents -> ATL client-side integration."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from agentictrading import (
    ATLConflictError,
    ATLRunFailedError,
    ATLValidationError,
    ExecutionResult,
    Observation,
    Run,
    RunResult,
    Step,
)
from agentictrading.integrations.tradingagents import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactValidationError,
    TradingAgentsDecisionGenerator,
    TradingAgentsDecisionArtifact,
    TradingAgentsDecisionRecord,
    TradingAgentsDependencyError,
    TradingAgentsGenerationError,
    TradingAgentsATLRunner,
    TradingAgentsATLRunOutcome,
    TradingAgentsReplayDiagnostics,
    TradingAgentsReplayIncompleteError,
    TradingAgentsReplayPlanner,
    TradingAgentsReplayValidationError,
    TradingAgentsVersionError,
    build_safe_manifest,
    default_decision_artifact_path,
    load_decision_artifact,
    map_rating,
    sanitize_error_message,
    save_decision_artifact,
    sha256_text,
)


def _record(
    date: str = "2026-04-03",
    *,
    rating: str = "Buy",
    action: str = "BUY",
) -> TradingAgentsDecisionRecord:
    raw = f"**Rating**: {rating}\n\nA concise portfolio decision."
    return TradingAgentsDecisionRecord(
        analysis_date=date,
        rating=rating,
        atl_action=action,
        status="valid",
        attempts=1,
        raw_final_trade_decision=raw,
        raw_sha256=sha256_text(raw),
    )


def _manifest(**overrides):
    manifest = build_safe_manifest(
        symbol="AAPL",
        tradingagents_version="0.3.1",
        config={
            "llm_provider": "openai",
            "deep_think_llm": "gpt-test-deep",
            "quick_think_llm": "gpt-test-quick",
            "max_debate_rounds": 1,
            "data_vendors": {"core_stock_apis": "yfinance"},
        },
        selected_analysts=("market", "news"),
        created_at="2026-07-26T12:00:00Z",
    )
    manifest.update(overrides)
    return manifest


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        ("Buy", "BUY"),
        ("Overweight", "BUY"),
        ("Hold", "HOLD"),
        ("Underweight", "SELL"),
        ("Sell", "SELL"),
        (" buy ", "BUY"),
    ],
)
def test_maps_tradingagents_five_tier_rating(rating, expected):
    assert map_rating(rating) == expected


def test_unknown_rating_is_not_silently_converted_to_hold():
    with pytest.raises(ArtifactValidationError, match="rating"):
        map_rating("")
    with pytest.raises(ArtifactValidationError, match="rating"):
        map_rating("Strong Buy")


def test_artifact_round_trip_and_file_hash(tmp_path):
    artifact = TradingAgentsDecisionArtifact(
        manifest=_manifest(),
        decisions=(
            _record("2026-04-03", rating="Buy", action="BUY"),
            TradingAgentsDecisionRecord(
                analysis_date="2026-04-10",
                rating="Hold",
                atl_action="HOLD",
                status="valid",
                attempts=1,
                raw_final_trade_decision="**Rating**: Hold\n\n持有理由。",
                raw_sha256=sha256_text("**Rating**: Hold\n\n持有理由。"),
            ),
        ),
    )
    path = tmp_path / "aapl.json"

    digest = save_decision_artifact(artifact, path)

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    # The digest is ATL run provenance, so it must identify the content and not
    # the platform: no newline translation may reach the file on any OS.
    assert b"\r\n" not in path.read_bytes()
    assert load_decision_artifact(path) == artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert "持有理由" in payload["decisions"][1]["raw_final_trade_decision"]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda p: p.update(schema_version="future-v9"), "schema"),
        (lambda p: p["manifest"].update(symbol=""), "symbol"),
        (lambda p: p["decisions"][0].update(status="mystery"), "status"),
        (lambda p: p["decisions"][0].update(attempts=0), "attempts"),
        (lambda p: p["decisions"][0].update(raw_sha256="bad"), "raw_sha256"),
    ],
)
def test_load_rejects_invalid_artifact_fields(tmp_path, mutator, message):
    artifact = TradingAgentsDecisionArtifact(
        manifest=_manifest(), decisions=(_record(),)
    )
    path = tmp_path / "bad.json"
    save_decision_artifact(artifact, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match=message):
        load_decision_artifact(path)


def test_load_rejects_malformed_json_and_empty_decisions(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="JSON"):
        load_decision_artifact(broken)

    empty = tmp_path / "empty.json"
    empty.write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "manifest": _manifest(),
                "decisions": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactValidationError, match="decision"):
        load_decision_artifact(empty)


def test_artifact_requires_unique_sorted_iso_dates():
    with pytest.raises(ArtifactValidationError, match="sorted"):
        TradingAgentsDecisionArtifact(
            manifest=_manifest(),
            decisions=(_record("2026-04-10"), _record("2026-04-03")),
        )
    with pytest.raises(ArtifactValidationError, match="unique"):
        TradingAgentsDecisionArtifact(
            manifest=_manifest(),
            decisions=(_record("2026-04-03"), _record("2026-04-03")),
        )
    with pytest.raises(ArtifactValidationError, match="analysis_date"):
        TradingAgentsDecisionArtifact(
            manifest=_manifest(), decisions=(_record("04/03/2026"),)
        )


_ERROR_RECORD_FIELDS = dict(
    analysis_date="2026-04-03",
    status="error",
    attempts=2,
    raw_final_trade_decision="",
    raw_sha256=sha256_text(""),
    error_type="RuntimeError",
    error_message="provider failed",
)


def test_error_record_has_explicit_hold_and_sanitized_error():
    record = TradingAgentsDecisionRecord(
        rating="", atl_action="HOLD", **_ERROR_RECORD_FIELDS
    )
    assert record.atl_action == "HOLD"
    assert record.error_type == "RuntimeError"


@pytest.mark.parametrize(
    ("rating", "atl_action", "message"),
    [
        # An errored analysis never traded, so it cannot claim a direction.
        ("", "BUY", "HOLD"),
        # It produced no opinion either, so it cannot carry a rating that
        # replay or downstream reporting would read as one.
        ("Buy", "HOLD", "rating"),
    ],
)
def test_error_record_rejects_action_or_rating(rating, atl_action, message):
    with pytest.raises(ArtifactValidationError, match=message):
        TradingAgentsDecisionRecord(
            rating=rating, atl_action=atl_action, **_ERROR_RECORD_FIELDS
        )


def test_safe_manifest_excludes_nested_credentials_and_scrubs_values():
    manifest = build_safe_manifest(
        symbol="AAPL",
        tradingagents_version="0.3.1",
        selected_analysts=("market",),
        created_at="2026-07-26T12:00:00Z",
        config={
            "llm_provider": "openai",
            "deep_think_llm": "gpt-test",
            "api_key": "sk-top-secret",
            "data_vendors": {
                "core_stock_apis": "yfinance",
                "provider_token": "nested-secret",
            },
            "backend_url": "https://user:password@example.com/v1",
            "unlisted_internal_path": "/Users/private/project",
        },
    )
    serialized = json.dumps(manifest, sort_keys=True)

    assert "sk-top-secret" not in serialized
    assert "nested-secret" not in serialized
    assert "password" not in serialized
    assert "/Users/private/project" not in serialized
    assert manifest["llm_provider"] == "openai"
    assert manifest["data_vendors"] == {"core_stock_apis": "yfinance"}
    assert len(manifest["safe_config_sha256"]) == 64


def test_artifact_rejects_manually_supplied_sensitive_manifest():
    unsafe = _manifest()
    unsafe["api_key"] = "sk-manual-secret"
    with pytest.raises(ArtifactValidationError, match="sensitive"):
        TradingAgentsDecisionArtifact(
            manifest=unsafe,
            decisions=(_record(),),
        )


def test_error_sanitizer_removes_common_secret_shapes_and_caps_length():
    message = (
        "OPENAI_API_KEY=sk-abc123 Authorization: Bearer bearer-secret "
        "https://alice:hunter2@example.com/v1 " + "x" * 1_000
    )
    cleaned = sanitize_error_message(message)

    assert "sk-abc123" not in cleaned
    assert "bearer-secret" not in cleaned
    assert "hunter2" not in cleaned
    assert len(cleaned) <= 300


def test_default_artifact_path_follows_the_storage_convention():
    """The default location is library contract, not a CLI detail.

    Every front-end that writes decisions has to agree on where they land, so
    this is asserted directly rather than only through a CLI run that always
    passes --output explicitly.
    """
    path = default_decision_artifact_path("aapl")
    assert path.parent == Path.home() / ".agentictrading" / "tradingagents" / "decisions"
    assert path.name.startswith("aapl-")
    assert path.suffix == ".json"
    # Case and surrounding whitespace must not produce a second directory.
    assert default_decision_artifact_path("  AAPL ").parent == path.parent
    assert default_decision_artifact_path("  AAPL ").name.startswith("aapl-")
    with pytest.raises(ArtifactValidationError):
        default_decision_artifact_path("   ")


def test_artifact_module_does_not_import_tradingagents():
    assert not any(
        name == "tradingagents" or name.startswith("tradingagents.")
        for name in sys.modules
    )


# ---------------------------------------------------------------------------
# Local TradingAgents generation (all dependencies are injected in tests)
# ---------------------------------------------------------------------------


class _FakeGraph:
    def __init__(self, responses):
        self.responses = {
            date: list(values) for date, values in responses.items()
        }
        self.calls = []

    def propagate(self, symbol, analysis_date):
        self.calls.append((symbol, analysis_date))
        value = self.responses[analysis_date].pop(0)
        if isinstance(value, BaseException):
            raise value
        return {"final_trade_decision": value}, "ignored-upstream-default"


def _generator(graph, *, parser=None, version="0.3.1"):
    return TradingAgentsDecisionGenerator(
        graph_factory=lambda **kwargs: graph,
        rating_parser=parser or (lambda raw, default="": raw.split(":", 1)[-1].strip()),
        version_resolver=lambda: version,
        clock=lambda: "2026-07-26T12:00:00Z",
    )


def test_generator_builds_graph_once_and_calls_each_date_in_order():
    graph = _FakeGraph(
        {
            "2026-04-03": ["Rating: Buy"],
            "2026-04-10": ["Rating: Hold"],
        }
    )
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return graph

    generator = TradingAgentsDecisionGenerator(
        graph_factory=factory,
        rating_parser=lambda raw, default="": raw.split(":", 1)[-1].strip(),
        version_resolver=lambda: "0.3.1",
        clock=lambda: "2026-07-26T12:00:00Z",
    )
    artifact = generator.generate(
        symbol="aapl",
        analysis_dates=("2026-04-03", "2026-04-10"),
        config={"llm_provider": "openai", "deep_think_llm": "gpt-test"},
        selected_analysts=("market", "news"),
    )

    assert len(factory_calls) == 1
    assert factory_calls[0]["selected_analysts"] == ("market", "news")
    assert factory_calls[0]["config"]["llm_provider"] == "openai"
    assert graph.calls == [
        ("AAPL", "2026-04-03"),
        ("AAPL", "2026-04-10"),
    ]
    assert [record.rating for record in artifact.decisions] == ["Buy", "Hold"]
    assert [record.atl_action for record in artifact.decisions] == ["BUY", "HOLD"]
    assert all(record.status == "valid" for record in artifact.decisions)
    assert artifact.manifest["tradingagents_version"] == "0.3.1"
    assert artifact.manifest["symbol"] == "AAPL"


def test_generator_passes_empty_default_to_upstream_rating_parser():
    graph = _FakeGraph({"2026-04-03": ["**Rating**: Hold"]})
    calls = []

    def parser(raw, default="not-empty"):
        calls.append((raw, default))
        return "Hold"

    artifact = _generator(graph, parser=parser).generate(
        symbol="AAPL",
        analysis_dates=("2026-04-03",),
        config={},
        selected_analysts=("market",),
    )

    assert calls == [("**Rating**: Hold", "")]
    assert artifact.decisions[0].rating == "Hold"
    assert artifact.decisions[0].status == "valid"


def test_generator_retries_once_then_succeeds():
    graph = _FakeGraph(
        {"2026-04-03": [RuntimeError("temporary provider error"), "Rating: Sell"]}
    )
    artifact = _generator(graph).generate(
        symbol="AAPL",
        analysis_dates=("2026-04-03",),
        config={},
        selected_analysts=("market",),
    )

    record = artifact.decisions[0]
    assert record.status == "valid"
    assert record.attempts == 2
    assert record.rating == "Sell"
    assert graph.calls == [("AAPL", "2026-04-03")] * 2


def test_generator_records_partial_failure_as_error_hold():
    graph = _FakeGraph(
        {
            "2026-04-03": ["Rating: Buy"],
            "2026-04-10": [
                RuntimeError("OPENAI_API_KEY=sk-secret failed"),
                RuntimeError("Bearer token-secret still failed"),
            ],
        }
    )
    artifact = _generator(graph).generate(
        symbol="AAPL",
        analysis_dates=("2026-04-03", "2026-04-10"),
        config={"api_key": "sk-manifest-secret", "llm_provider": "openai"},
        selected_analysts=("market",),
    )

    failed = artifact.decisions[1]
    assert failed.status == "error"
    assert failed.atl_action == "HOLD"
    assert failed.attempts == 2
    assert failed.error_type == "RuntimeError"
    assert "token-secret" not in failed.error_message
    assert "sk-manifest-secret" not in json.dumps(artifact.manifest)


def test_generator_retries_unparseable_output_instead_of_defaulting_to_hold():
    graph = _FakeGraph(
        {"2026-04-03": ["No rating in this response", "Still no rating"]}
    )
    generator = _generator(graph, parser=lambda raw, default="": default)

    with pytest.raises(TradingAgentsGenerationError, match="all analysis dates"):
        generator.generate(
            symbol="AAPL",
            analysis_dates=("2026-04-03",),
            config={},
            selected_analysts=("market",),
        )

    assert graph.calls == [("AAPL", "2026-04-03")] * 2


def test_generator_refuses_to_return_artifact_when_every_date_fails():
    graph = _FakeGraph(
        {
            "2026-04-03": [RuntimeError("down"), RuntimeError("still down")],
            "2026-04-10": [RuntimeError("down"), RuntimeError("still down")],
        }
    )
    with pytest.raises(TradingAgentsGenerationError, match="all analysis dates"):
        _generator(graph).generate(
            symbol="AAPL",
            analysis_dates=("2026-04-03", "2026-04-10"),
            config={},
            selected_analysts=("market",),
        )


def test_generator_stops_paying_for_dates_after_a_failure_streak():
    """Three dead dates in a row is a systemic fault, not three bad dates.

    Each failed date has already burned two full multi-agent analyses, so the
    remaining dates must not be attempted once the cause is clearly not
    date-specific.
    """
    graph = _FakeGraph(
        {
            "2026-04-03": ["Rating: Buy"],
            "2026-04-06": [RuntimeError("down"), RuntimeError("down")],
            "2026-04-07": [RuntimeError("down"), RuntimeError("down")],
            "2026-04-08": [RuntimeError("down"), RuntimeError("down")],
            "2026-04-09": ["Rating: Buy"],
        }
    )

    with pytest.raises(TradingAgentsGenerationError, match="consecutive"):
        _generator(graph).generate(
            symbol="AAPL",
            analysis_dates=(
                "2026-04-03",
                "2026-04-06",
                "2026-04-07",
                "2026-04-08",
                "2026-04-09",
            ),
            config={},
            selected_analysts=("market",),
        )

    assert ("AAPL", "2026-04-09") not in graph.calls


def test_generator_failure_streak_resets_on_a_successful_date():
    graph = _FakeGraph(
        {
            "2026-04-03": [RuntimeError("down"), RuntimeError("down")],
            "2026-04-06": [RuntimeError("down"), RuntimeError("down")],
            "2026-04-07": ["Rating: Buy"],
            "2026-04-08": [RuntimeError("down"), RuntimeError("down")],
        }
    )

    artifact = _generator(graph).generate(
        symbol="AAPL",
        analysis_dates=("2026-04-03", "2026-04-06", "2026-04-07", "2026-04-08"),
        config={},
        selected_analysts=("market",),
    )

    assert [record.status for record in artifact.decisions] == [
        "error",
        "error",
        "valid",
        "error",
    ]


def test_generator_rejects_a_non_positive_failure_budget():
    with pytest.raises(ArtifactValidationError, match="max_consecutive_failures"):
        TradingAgentsDecisionGenerator(max_consecutive_failures=0)


@pytest.mark.parametrize("version", ["0.2.9", "0.4.0", "1.0.0", "unknown"])
def test_generator_rejects_unverified_tradingagents_versions(version):
    graph = _FakeGraph({"2026-04-03": ["Rating: Buy"]})
    with pytest.raises(TradingAgentsVersionError, match="0.3"):
        _generator(graph, version=version).generate(
            symbol="AAPL",
            analysis_dates=("2026-04-03",),
            config={},
            selected_analysts=("market",),
        )
    assert graph.calls == []


def test_default_generator_loads_dependency_only_when_generate_is_called(monkeypatch):
    generator = TradingAgentsDecisionGenerator()
    assert not any(
        name == "tradingagents" or name.startswith("tradingagents.")
        for name in sys.modules
    )

    real_import = __import__

    def blocked_import(name, *args, **kwargs):
        if name == "tradingagents" or name.startswith("tradingagents."):
            raise ModuleNotFoundError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    with pytest.raises(TradingAgentsDependencyError, match="git clone"):
        generator.generate(
            symbol="AAPL",
            analysis_dates=("2026-04-03",),
            config={},
            selected_analysts=("market",),
        )


# ---------------------------------------------------------------------------
# Pure T+1 replay planner
# ---------------------------------------------------------------------------


def _artifact(*records):
    return TradingAgentsDecisionArtifact(
        manifest=_manifest(symbol="AAPL"),
        decisions=tuple(records or (_record(),)),
    )


def _step(
    timestamp: str,
    *,
    sequence: int = 0,
    price=100.0,
    equity=1_000.0,
    positions=None,
    allowed_symbols=None,
    max_position_weight=0.25,
    step_id=None,
):
    features = {} if price is None else {"AAPL": {"price": price}}
    observation = Observation(
        market={"features": features},
        portfolio={
            "cash": equity,
            "equity": equity,
            "positions": list(positions or []),
        },
    )
    constraints = {}
    if allowed_symbols is not None:
        constraints["allowed_symbols"] = allowed_symbols
    else:
        constraints["allowed_symbols"] = ["AAPL"]
    if max_position_weight is not None:
        constraints["max_position_weight"] = max_position_weight
    return Step(
        status="awaiting_decision",
        run_id="run_1",
        step_id=step_id or f"step_{sequence}",
        sequence=sequence,
        timestamp=timestamp,
        observation=observation,
        constraints=constraints,
    )


def _accepted(planner, step):
    """Propose a decision and accept it, as a successful ATL submission does."""
    decision = planner.decision_for_step(step)
    planner.commit(step)
    return decision


def _error_record(date="2026-04-03"):
    return TradingAgentsDecisionRecord(
        analysis_date=date,
        rating="",
        atl_action="HOLD",
        status="error",
        attempts=2,
        raw_final_trade_decision="",
        raw_sha256=sha256_text(""),
        error_type="RuntimeError",
        error_message="provider unavailable",
    )


def test_replay_waits_until_first_new_york_trading_date_after_analysis():
    planner = TradingAgentsReplayPlanner(_artifact(_record()), "a" * 64)

    same_day = _accepted(
        planner, _step("2026-04-03T14:00:00+00:00", sequence=0)
    )
    monday_open = _accepted(
        planner, _step("2026-04-06T13:30:00+00:00", sequence=1)
    )
    monday_later = _accepted(
        planner, _step("2026-04-06T14:30:00+00:00", sequence=2)
    )

    assert same_day.orders == []
    assert len(monday_open.orders) == 1
    assert monday_open.orders[0].side == "buy"
    assert monday_later.orders == []
    diagnostics = planner.finalize()
    assert diagnostics.processed_dates == ("2026-04-03",)
    assert diagnostics.unprocessed_dates == ()
    assert diagnostics.buy_orders == 1
    assert diagnostics.passive_holds == 2


def test_replay_same_step_is_idempotent():
    planner = TradingAgentsReplayPlanner(_artifact(_record()), "b" * 64)
    step = _step("2026-04-06T13:30:00+00:00", step_id="same")

    first = planner.decision_for_step(step)
    second = planner.decision_for_step(step)

    assert second is first
    planner.commit(step)
    assert planner.decision_for_step(step) is first
    assert planner.finalize().buy_orders == 1


def test_replay_uses_latest_when_multiple_records_become_eligible():
    planner = TradingAgentsReplayPlanner(
        _artifact(
            _record("2026-04-03", rating="Buy", action="BUY"),
            _record("2026-04-10", rating="Sell", action="SELL"),
        ),
        "c" * 64,
    )
    decision = _accepted(
        planner,
        _step(
            "2026-04-13T13:30:00+00:00",
            positions=[{"symbol": "AAPL", "quantity": 2}],
        ),
    )

    assert decision.orders[0].side == "sell"
    assert decision.orders[0].quantity == 2
    diagnostics = planner.finalize()
    assert diagnostics.superseded == 1
    assert diagnostics.processed_dates == ("2026-04-03", "2026-04-10")


def test_replay_error_record_is_an_explicit_generation_error_hold():
    planner = TradingAgentsReplayPlanner(_artifact(_error_record()), "d" * 64)

    decision = _accepted(planner, _step("2026-04-06T13:30:00+00:00"))

    assert decision.orders == []
    assert "generation_error" in decision.rationale
    assert "2026-04-03" in decision.rationale
    assert planner.finalize().error_holds == 1


def test_replay_model_hold_is_distinct_from_passive_hold():
    record = _record("2026-04-03", rating="Hold", action="HOLD")
    planner = TradingAgentsReplayPlanner(_artifact(record), "e" * 64)

    decision = _accepted(planner, _step("2026-04-06T13:30:00+00:00"))

    assert decision.orders == []
    assert "rating=Hold" in decision.rationale
    diagnostics = planner.finalize()
    assert diagnostics.model_holds == 1
    assert diagnostics.passive_holds == 0


def test_replay_buy_only_tops_up_to_environment_weight_cap():
    planner = TradingAgentsReplayPlanner(_artifact(_record()), "f" * 64)
    decision = planner.decision_for_step(
        _step(
            "2026-04-06T13:30:00+00:00",
            price=100,
            equity=1_000,
            positions=[{"symbol": "AAPL", "quantity": 1}],
        )
    )

    order = decision.orders[0]
    assert order.symbol == "AAPL"
    assert order.side == "buy"
    assert order.quantity_type == "shares"
    assert order.quantity == 1  # floor(1000 * .25 / 100) - one already held
    assert order.order_type == "market"
    assert "rating=Buy" in decision.rationale
    assert "artifact=" + "f" * 12 in decision.rationale


@pytest.mark.parametrize(
    ("price", "held", "reason", "counter"),
    [
        (100, 2, "already_at_target", "constraint_holds"),
        (300, 0, "price_too_high_for_target", "price_too_high_holds"),
        (None, 0, "missing_price", "constraint_holds"),
    ],
)
def test_replay_buy_constraint_holds_have_specific_reason(
    price, held, reason, counter
):
    planner = TradingAgentsReplayPlanner(_artifact(_record()), "1" * 64)
    positions = [] if not held else [{"symbol": "AAPL", "quantity": held}]

    decision = _accepted(
        planner,
        _step(
            "2026-04-06T13:30:00+00:00",
            price=price,
            positions=positions,
        ),
    )

    assert decision.orders == []
    assert reason in decision.rationale
    diagnostics = planner.finalize()
    assert getattr(diagnostics, counter) == 1


def test_replay_price_too_high_is_counted_apart_from_other_constraint_holds():
    """A signal priced out of the cap must not read as an ordinary hold.

    $1,000 of equity at a 25% cap buys nothing above $250/share, so the BUY is
    unexecutable by arithmetic. The rationale has to name both numbers and the
    counter has to stay distinct, or the run is indistinguishable from a
    strategy that simply chose to stay in cash.
    """
    planner = TradingAgentsReplayPlanner(_artifact(_record()), "7" * 64)

    decision = _accepted(
        planner, _step("2026-04-06T13:30:00+00:00", price=430.0, equity=1_000.0)
    )

    assert decision.orders == []
    assert "price=430.00" in decision.rationale
    assert "max_position_budget=250.00" in decision.rationale
    diagnostics = planner.finalize()
    assert diagnostics.price_too_high_holds == 1
    assert diagnostics.constraint_holds == 0
    assert diagnostics.buy_orders == 0


def test_replay_sell_closes_all_shares_and_never_shorts():
    sell = _record("2026-04-03", rating="Underweight", action="SELL")
    planner = TradingAgentsReplayPlanner(_artifact(sell), "2" * 64)
    decision = _accepted(
        planner,
        _step(
            "2026-04-06T13:30:00+00:00",
            positions=[{"symbol": "AAPL", "quantity": 3}],
        ),
    )
    assert decision.orders[0].side == "sell"
    assert decision.orders[0].quantity == 3

    empty_planner = TradingAgentsReplayPlanner(_artifact(sell), "3" * 64)
    empty = _accepted(empty_planner, _step("2026-04-06T13:30:00+00:00"))
    assert empty.orders == []
    assert "sell_without_position" in empty.rationale
    diagnostics = empty_planner.finalize()
    assert diagnostics.sell_orders == 0
    assert diagnostics.constraint_holds == 1


@pytest.mark.parametrize(
    ("step", "message"),
    [
        (
            _step(
                "2026-04-06T13:30:00+00:00",
                allowed_symbols=["MSFT"],
            ),
            "allowed_symbols",
        ),
        (
            _step(
                "2026-04-06T13:30:00+00:00",
                max_position_weight=None,
            ),
            "max_position_weight",
        ),
        (
            _step("2026-04-06T13:30:00"),
            "timezone",
        ),
    ],
)
def test_replay_fails_loudly_on_invalid_step_contract(step, message):
    planner = TradingAgentsReplayPlanner(_artifact(_record()), "4" * 64)
    with pytest.raises(TradingAgentsReplayValidationError, match=message):
        planner.decision_for_step(step)


def test_replay_explains_how_to_fix_a_missing_timezone_database(monkeypatch):
    """Windows ships no IANA database; a bare lookup error is not actionable."""
    from zoneinfo import ZoneInfoNotFoundError

    from agentictrading.integrations import _tradingagents_replay as replay_module

    def _no_database(key):
        raise ZoneInfoNotFoundError(key)

    monkeypatch.setattr(replay_module, "ZoneInfo", _no_database)
    planner = TradingAgentsReplayPlanner(_artifact(_record()), "c" * 64)

    with pytest.raises(TradingAgentsReplayValidationError, match="pip install tzdata"):
        planner.decision_for_step(_step("2026-04-06T13:30:00+00:00"))


def test_sdk_declares_tzdata_where_zoneinfo_has_no_system_database():
    """Guard the dependency the replay planner's zoneinfo use now requires."""
    pyproject = (
        Path(__file__).resolve().parents[1] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    declared = re.search(
        r"^dependencies\s*=\s*\[(.*?)\]", pyproject, re.MULTILINE | re.DOTALL
    )

    assert declared is not None
    assert "tzdata" in declared.group(1)
    assert "Windows" in declared.group(1)


def test_replay_finalize_lists_unprocessed_analysis_dates():
    planner = TradingAgentsReplayPlanner(
        _artifact(
            _record("2026-04-03"),
            _record("2026-04-10", rating="Hold", action="HOLD"),
        ),
        "5" * 64,
    )
    _accepted(planner, _step("2026-04-06T13:30:00+00:00"))

    diagnostics = planner.finalize()

    assert diagnostics.processed_dates == ("2026-04-03",)
    assert diagnostics.unprocessed_dates == ("2026-04-10",)


def test_replay_uncommitted_proposal_consumes_nothing():
    """A proposal ATL never accepted must leave the artifact untouched.

    Consuming the record at proposal time would report an order that was never
    executed while permanently dropping the signal behind it.
    """
    planner = TradingAgentsReplayPlanner(_artifact(_record()), "8" * 64)
    step = _step("2026-04-06T13:30:00+00:00")

    decision = planner.decision_for_step(step)
    assert decision.orders[0].side == "buy"

    planner.discard(step)

    diagnostics = planner.finalize()
    assert diagnostics.buy_orders == 0
    assert diagnostics.processed_dates == ()
    assert diagnostics.unprocessed_dates == ("2026-04-03",)
    assert diagnostics.autoheld_steps == 1


def test_replay_discarded_record_executes_on_the_next_step():
    planner = TradingAgentsReplayPlanner(_artifact(_record()), "9" * 64)
    autoheld = _step("2026-04-06T13:30:00+00:00", sequence=0)

    planner.decision_for_step(autoheld)
    planner.discard(autoheld)
    retried = _accepted(planner, _step("2026-04-06T14:30:00+00:00", sequence=1))

    assert retried.orders[0].side == "buy"
    diagnostics = planner.finalize()
    assert diagnostics.buy_orders == 1
    assert diagnostics.autoheld_steps == 1
    assert diagnostics.unprocessed_dates == ()


def test_replay_discard_rolls_back_superseded_records_too():
    planner = TradingAgentsReplayPlanner(
        _artifact(
            _record("2026-04-03", rating="Buy", action="BUY"),
            _record("2026-04-10", rating="Hold", action="HOLD"),
        ),
        "b" * 64,
    )
    step = _step("2026-04-13T13:30:00+00:00")

    planner.decision_for_step(step)
    planner.discard(step)

    diagnostics = planner.finalize()
    assert diagnostics.superseded == 0
    assert diagnostics.unprocessed_dates == ("2026-04-03", "2026-04-10")


# ---------------------------------------------------------------------------
# ATLClient orchestration loop (fake client, no network)
# ---------------------------------------------------------------------------


class _FakeATLClient:
    def __init__(self, steps, *, result=None, execution_results=None):
        self.steps = list(steps)
        self.result = result or RunResult(
            run_id="run_1",
            status="completed",
            metrics={"total_return": 0.05, "timeout_holds": 0},
            raw={"compare_url": "http://atl.local/app?run=run_1"},
        )
        self.execution_results = list(execution_results or [])
        self.created = []
        self.submitted = []
        self.waits = []
        self.environment_calls = 0

    def list_environments(self):
        self.environment_calls += 1
        return [
            {
                "environment_id": "us-equity-hourly-v1",
                "universe": ["AAPL", "MSFT"],
            }
        ]

    def create_run(self, agent_version_id, **kwargs):
        self.created.append((agent_version_id, kwargs))
        return Run(run_id="run_1", status="created")

    def get_next_step(self, run_id):
        assert run_id == "run_1"
        return self.steps.pop(0)

    def submit_decision(self, run_id, step_id, decision):
        self.submitted.append((run_id, step_id, decision))
        if self.execution_results:
            result = self.execution_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return ExecutionResult(
            run_id=run_id,
            step_id=step_id,
            accepted=True,
            validation={"passed": True, "rejections": []},
            fills=[],
            run_status="running",
        )

    def get_run_result(self, run_id):
        assert run_id == "run_1"
        return self.result

    def wait(self, seconds):
        self.waits.append(seconds)


def _completed_step():
    return Step(status="completed", run_id="run_1", result_run_id="ext_1")


def test_atl_runner_drives_full_loop_and_returns_structured_outcome():
    artifact = _artifact(
        _record("2026-04-03", rating="Buy", action="BUY"),
        _record("2026-04-10", rating="Hold", action="HOLD"),
    )
    executions = [
        ExecutionResult(
            run_id="run_1",
            step_id="step_0",
            accepted=True,
            validation={"passed": True, "rejections": []},
            fills=[{"symbol": "AAPL", "side": "buy", "filled_quantity": 2}],
            run_status="running",
        ),
        ExecutionResult(
            run_id="run_1",
            step_id="step_1",
            accepted=True,
            validation={
                "passed": False,
                "rejections": [{"reason": "test_rejection"}],
            },
            fills=[],
            run_status="running",
        ),
    ]
    client = _FakeATLClient(
        [
            Step(status="loading", run_id="run_1"),
            _step("2026-04-06T13:30:00+00:00", sequence=0),
            _step("2026-04-13T13:30:00+00:00", sequence=1),
            _completed_step(),
        ],
        execution_results=executions,
    )

    outcome = TradingAgentsATLRunner(client).run_backtest(
        artifact=artifact,
        artifact_sha256="a" * 64,
        agent_version_id="agv_1",
        start_date="2026-04-06",
        end_date="2026-04-14",
        poll_interval=0.01,
    )

    assert isinstance(outcome, TradingAgentsATLRunOutcome)
    assert outcome.result.metrics["total_return"] == 0.05
    assert outcome.run_id == "run_1"
    assert outcome.compare_url == "http://atl.local/app?run=run_1"
    assert outcome.fills_count == 1
    assert outcome.rejections == ({"reason": "test_rejection"},)
    assert outcome.replay.buy_orders == 1
    assert outcome.replay.model_holds == 1
    assert outcome.timeout_holds == 0
    assert client.waits == [0.01]
    assert len(client.submitted) == 2

    agent_version_id, create_kwargs = client.created[0]
    assert agent_version_id == "agv_1"
    assert create_kwargs["environment_id"] == "us-equity-hourly-v1"
    assert create_kwargs["symbols"] == ["AAPL"]
    integration = create_kwargs["config"]["integration"]
    assert integration["id"] == "tradingagents"
    assert integration["artifact_sha256"] == "a" * 64
    assert integration["analysis_dates"] == ["2026-04-03", "2026-04-10"]
    assert integration["decision_data_source"] == "tradingagents_configured_vendors"
    assert integration["execution_data_source"] == "atl_alpaca"
    serialized = json.dumps(create_kwargs["config"])
    assert "raw_final_trade_decision" not in serialized
    assert "/Users/" not in serialized


def test_atl_runner_incomplete_replay_raises_with_run_id_and_dates():
    artifact = _artifact(
        _record("2026-04-03"),
        _record("2026-04-10", rating="Hold", action="HOLD"),
    )
    client = _FakeATLClient(
        [_step("2026-04-06T13:30:00+00:00"), _completed_step()]
    )

    with pytest.raises(TradingAgentsReplayIncompleteError) as error:
        TradingAgentsATLRunner(client).run_backtest(
            artifact=artifact,
            artifact_sha256="b" * 64,
            agent_version_id="agv_1",
            start_date="2026-04-06",
            end_date="2026-04-11",
        )

    assert error.value.run_id == "run_1"
    assert error.value.analysis_dates == ("2026-04-10",)
    assert error.value.result.metrics["total_return"] == 0.05


def test_atl_runner_attaches_run_id_to_api_errors():
    client = _FakeATLClient(
        [_step("2026-04-06T13:30:00+00:00")],
        execution_results=[ATLValidationError("rejected", code="bad_order")],
    )

    with pytest.raises(ATLValidationError) as error:
        TradingAgentsATLRunner(client).run_backtest(
            artifact=_artifact(_record()),
            artifact_sha256="c" * 64,
            agent_version_id="agv_1",
            start_date="2026-04-06",
            end_date="2026-04-07",
        )

    assert error.value.run_id == "run_1"


def test_atl_runner_retries_auto_held_record_on_the_next_step():
    """An auto-held step must not consume the signal it failed to submit."""
    client = _FakeATLClient(
        [
            _step("2026-04-06T13:30:00+00:00", sequence=0),
            _step("2026-04-06T14:30:00+00:00", sequence=1),
            _completed_step(),
        ],
        result=RunResult(
            run_id="run_1",
            status="completed",
            metrics={"timeout_holds": 1},
        ),
        execution_results=[
            ATLConflictError(
                "late", code="decision_deadline_exceeded", status_code=409
            )
        ],
    )

    outcome = TradingAgentsATLRunner(client).run_backtest(
        artifact=_artifact(_record()),
        artifact_sha256="d" * 64,
        agent_version_id="agv_1",
        start_date="2026-04-06",
        end_date="2026-04-07",
    )

    assert outcome.timeout_holds == 1
    assert outcome.autoheld_steps == 1
    # The BUY landed on the retry, and it is counted exactly once.
    assert outcome.replay.buy_orders == 1
    assert outcome.replay.processed_dates == ("2026-04-03",)
    assert outcome.replay.unprocessed_dates == ()
    assert len(client.submitted) == 2


def test_atl_runner_auto_hold_on_the_last_step_reports_an_incomplete_replay():
    """A dropped signal has to fail loudly instead of counting as an order."""
    client = _FakeATLClient(
        [_step("2026-04-06T13:30:00+00:00"), _completed_step()],
        result=RunResult(
            run_id="run_1",
            status="completed",
            metrics={"timeout_holds": 1},
        ),
        execution_results=[
            ATLConflictError(
                "late", code="decision_deadline_exceeded", status_code=409
            )
        ],
    )

    with pytest.raises(TradingAgentsReplayIncompleteError) as error:
        TradingAgentsATLRunner(client).run_backtest(
            artifact=_artifact(_record()),
            artifact_sha256="d" * 64,
            agent_version_id="agv_1",
            start_date="2026-04-06",
            end_date="2026-04-07",
        )

    assert error.value.analysis_dates == ("2026-04-03",)
    diagnostics = error.value.diagnostics
    assert diagnostics.buy_orders == 0
    assert diagnostics.autoheld_steps == 1


def test_atl_runner_attaches_run_id_to_replay_validation_errors():
    """A mid-run contract break must name the run it leaves open."""
    client = _FakeATLClient(
        [_step("2026-04-06T13:30:00+00:00", allowed_symbols=["MSFT"])]
    )

    with pytest.raises(TradingAgentsReplayValidationError) as error:
        TradingAgentsATLRunner(client).run_backtest(
            artifact=_artifact(_record()),
            artifact_sha256="d" * 64,
            agent_version_id="agv_1",
            start_date="2026-04-06",
            end_date="2026-04-07",
        )

    assert error.value.run_id == "run_1"
    assert "run_1" in str(error.value)


def test_atl_runner_failed_step_raises_run_failed_error():
    client = _FakeATLClient(
        [Step(status="failed", run_id="run_1", message="engine failed")]
    )
    with pytest.raises(ATLRunFailedError, match="engine failed") as error:
        TradingAgentsATLRunner(client).run_backtest(
            artifact=_artifact(_record()),
            artifact_sha256="e" * 64,
            agent_version_id="agv_1",
            start_date="2026-04-06",
            end_date="2026-04-07",
        )
    assert error.value.run_id == "run_1"


def test_atl_runner_rejects_invalid_input_before_creating_run():
    all_error = _artifact(_error_record())
    client = _FakeATLClient([])
    with pytest.raises(TradingAgentsGenerationError, match="valid"):
        TradingAgentsATLRunner(client).run_backtest(
            artifact=all_error,
            artifact_sha256="f" * 64,
            agent_version_id="agv_1",
            start_date="2026-04-06",
            end_date="2026-04-07",
        )
    assert client.environment_calls == 0
    assert client.created == []

    with pytest.raises(ArtifactValidationError, match="start_date"):
        TradingAgentsATLRunner(client).run_backtest(
            artifact=_artifact(_record()),
            artifact_sha256="f" * 64,
            agent_version_id="agv_1",
            start_date="2026-04-07",
            end_date="2026-04-06",
        )
    assert client.created == []


def test_atl_runner_preflights_symbol_against_environment():
    client = _FakeATLClient([])
    client.list_environments = lambda: [
        {"environment_id": "us-equity-hourly-v1", "universe": ["MSFT"]}
    ]

    with pytest.raises(TradingAgentsReplayValidationError, match="universe"):
        TradingAgentsATLRunner(client).run_backtest(
            artifact=_artifact(_record()),
            artifact_sha256="1" * 64,
            agent_version_id="agv_1",
            start_date="2026-04-06",
            end_date="2026-04-07",
        )
    assert client.created == []


# ---------------------------------------------------------------------------
# Example CLI
# ---------------------------------------------------------------------------


def _load_cli_module():
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "dashboard" / "examples" / "tradingagents_atl_backtest.py"
    spec = importlib.util.spec_from_file_location("tradingagents_atl_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _outcome_with(**replay_overrides):
    """A completed single-BUY outcome, with any diagnostic counter overridden."""
    counters = {
        "processed_dates": ("2026-04-03",),
        "unprocessed_dates": (),
        "buy_orders": 0,
        "sell_orders": 0,
        "model_holds": 0,
        "error_holds": 0,
        "passive_holds": 0,
        "constraint_holds": 0,
        "superseded": 0,
    }
    counters.update(replay_overrides)
    return TradingAgentsATLRunOutcome(
        result=RunResult(
            run_id="run_1",
            status="completed",
            metrics={"total_return": 0.0, "timeout_holds": 0},
        ),
        replay=TradingAgentsReplayDiagnostics(**counters),
        fills=(),
        rejections=(),
    )


def test_cli_parser_accepts_repeated_explicit_analysis_dates():
    cli = _load_cli_module()
    args = cli.build_parser().parse_args(
        [
            "--symbol",
            "AAPL",
            "--analysis-date",
            "2026-04-03",
            "--analysis-date",
            "2026-04-10",
            "--start-date",
            "2026-04-06",
            "--end-date",
            "2026-04-14",
        ]
    )

    assert args.symbol == "AAPL"
    assert args.analysis_dates == ["2026-04-03", "2026-04-10"]
    assert args.decisions_file is None
    assert "--analysis-date" in cli.build_parser().format_help()


def test_cli_requires_generation_dates_unless_replaying_file(tmp_path):
    cli = _load_cli_module()
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--symbol",
            "AAPL",
            "--start-date",
            "2026-04-06",
            "--end-date",
            "2026-04-14",
        ]
    )
    with pytest.raises(cli.CLIConfigurationError, match="analysis-date"):
        cli.validate_args(args)

    replay = parser.parse_args(
        [
            "--symbol",
            "AAPL",
            "--decisions-file",
            str(tmp_path / "artifact.json"),
            "--start-date",
            "2026-04-06",
            "--end-date",
            "2026-04-14",
        ]
    )
    cli.validate_args(replay)


def test_cli_requires_atl_credentials_from_environment():
    cli = _load_cli_module()
    args = cli.build_parser().parse_args(
        [
            "--symbol",
            "AAPL",
            "--decisions-file",
            "artifact.json",
            "--start-date",
            "2026-04-06",
            "--end-date",
            "2026-04-14",
        ]
    )

    with pytest.raises(cli.CLIConfigurationError, match="ATL_API_KEY"):
        cli.resolve_atl_settings(args, environ={})
    with pytest.raises(cli.CLIConfigurationError, match="ATL_BASE_URL"):
        cli.resolve_atl_settings(args, environ={"ATL_API_KEY": "ag_test"})
    with pytest.raises(cli.CLIConfigurationError, match="ATL_AGENT_VERSION_ID"):
        cli.resolve_atl_settings(
            args,
            environ={"ATL_API_KEY": "ag_test", "ATL_BASE_URL": "http://atl"},
        )


def test_cli_decisions_file_replay_never_constructs_generator(tmp_path):
    cli = _load_cli_module()
    artifact = _artifact(_record())
    artifact_path = tmp_path / "artifact.json"
    artifact_hash = save_decision_artifact(artifact, artifact_path)
    args = cli.build_parser().parse_args(
        [
            "--symbol",
            "AAPL",
            "--decisions-file",
            str(artifact_path),
            "--start-date",
            "2026-04-06",
            "--end-date",
            "2026-04-07",
        ]
    )
    expected = _outcome_with(buy_orders=1)
    runner_calls = []

    class FakeRunner:
        def validate_symbol(self, symbol):
            assert symbol == "AAPL"

        def run_backtest(self, **kwargs):
            runner_calls.append(kwargs)
            return expected

    result = cli.run_from_args(
        args,
        environ={
            "ATL_API_KEY": "ag_test",
            "ATL_BASE_URL": "http://atl.local",
            "ATL_AGENT_VERSION_ID": "agv_test",
        },
        client_factory=lambda **kwargs: ("client", kwargs),
        runner_factory=lambda client: FakeRunner(),
        generator_factory=lambda: pytest.fail(
            "generator must not be constructed for --decisions-file"
        ),
    )

    assert result.outcome is expected
    assert result.artifact_path == artifact_path
    assert result.artifact_sha256 == artifact_hash
    assert runner_calls[0]["artifact"] == artifact


def test_cli_generation_writes_artifact_and_forwards_safe_model_config(tmp_path):
    cli = _load_cli_module()
    output = tmp_path / "generated.json"
    args = cli.build_parser().parse_args(
        [
            "--symbol",
            "aapl",
            "--analysis-date",
            "2026-04-03",
            "--start-date",
            "2026-04-06",
            "--end-date",
            "2026-04-07",
            "--output",
            str(output),
            "--llm-provider",
            "openai",
            "--deep-think-llm",
            "deep-test",
            "--quick-think-llm",
            "quick-test",
            "--selected-analyst",
            "market",
        ]
    )
    generation_calls = []
    runner_calls = []
    events = []

    class FakeGenerator:
        def generate(self, **kwargs):
            assert events == ["preflight"]
            generation_calls.append(kwargs)
            return _artifact(_record())

    class FakeRunner:
        def validate_symbol(self, symbol):
            assert symbol == "AAPL"
            events.append("preflight")

        def run_backtest(self, **kwargs):
            runner_calls.append(kwargs)
            return _outcome_with(buy_orders=1)

    result = cli.run_from_args(
        args,
        environ={
            "ATL_API_KEY": "ag_test",
            "ATL_BASE_URL": "http://atl.local",
            "ATL_AGENT_VERSION_ID": "agv_test",
        },
        client_factory=lambda **kwargs: object(),
        runner_factory=lambda client: FakeRunner(),
        generator_factory=FakeGenerator,
    )

    assert output.is_file()
    assert result.artifact_path == output
    assert generation_calls == [
        {
            "symbol": "AAPL",
            "analysis_dates": ("2026-04-03",),
            "config": {
                "llm_provider": "openai",
                "deep_think_llm": "deep-test",
                "quick_think_llm": "quick-test",
            },
            "selected_analysts": ("market",),
        }
    ]
    assert runner_calls[0]["artifact_sha256"] == result.artifact_sha256


def test_cli_summary_warns_when_every_buy_was_priced_out(capsys):
    """A 0% run caused by the position cap must not read as a model decision."""
    cli = _load_cli_module()
    result = cli.CommandResult(
        artifact_path=Path("artifact.json"),
        artifact_sha256="a" * 64,
        outcome=_outcome_with(price_too_high_holds=2, autoheld_steps=1),
    )

    cli.print_summary(result)

    captured = capsys.readouterr()
    assert "price_too_high=2" in captured.out
    assert "autoheld_steps=1" in captured.out
    assert "WARNING" in captured.err
    assert "$250" in captured.err
    assert "auto-held 1 step" in captured.err


def test_cli_summary_stays_quiet_when_orders_actually_executed(capsys):
    cli = _load_cli_module()
    result = cli.CommandResult(
        artifact_path=Path("artifact.json"),
        artifact_sha256="a" * 64,
        outcome=_outcome_with(buy_orders=1),
    )

    cli.print_summary(result)

    assert capsys.readouterr().err == ""


def test_cli_reports_the_run_id_left_open_by_a_mid_run_abort(capsys):
    cli = _load_cli_module()
    argv = [
        "--symbol",
        "AAPL",
        "--decisions-file",
        "artifact.json",
        "--start-date",
        "2026-04-06",
        "--end-date",
        "2026-04-07",
    ]

    def _abort(*args, **kwargs):
        raise TradingAgentsReplayValidationError(
            "AAPL is missing from Step constraints.allowed_symbols"
        ).with_run_id("run_42")

    original = cli.run_from_args
    cli.run_from_args = _abort
    try:
        assert cli.main(argv) == 1
    finally:
        cli.run_from_args = original

    assert "Aborted ATL run id: run_42" in capsys.readouterr().err
