"""Tests for the optional TradingAgents -> ATL client-side integration."""

from __future__ import annotations

import hashlib
import json
import sys

import pytest

from agentictrading.integrations.tradingagents import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactValidationError,
    TradingAgentsDecisionGenerator,
    TradingAgentsDecisionArtifact,
    TradingAgentsDecisionRecord,
    TradingAgentsDependencyError,
    TradingAgentsGenerationError,
    TradingAgentsVersionError,
    build_safe_manifest,
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


def test_error_record_has_explicit_hold_and_sanitized_error():
    record = TradingAgentsDecisionRecord(
        analysis_date="2026-04-03",
        rating="",
        atl_action="HOLD",
        status="error",
        attempts=2,
        raw_final_trade_decision="",
        raw_sha256=sha256_text(""),
        error_type="RuntimeError",
        error_message="provider failed",
    )
    assert record.atl_action == "HOLD"
    assert record.error_type == "RuntimeError"

    with pytest.raises(ArtifactValidationError, match="HOLD"):
        TradingAgentsDecisionRecord(
            analysis_date="2026-04-03",
            rating="",
            atl_action="BUY",
            status="error",
            attempts=2,
            raw_final_trade_decision="",
            raw_sha256=sha256_text(""),
            error_type="RuntimeError",
            error_message="provider failed",
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
