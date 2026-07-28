from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from agentictrading.integrations.vnpy_cta import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactValidationError,
    CapturedCtaOrder,
    VnpyBindings,
    VnpyCtaAdapter,
    VnpyCtaAuditArtifact,
    VnpyCtaAuditRecord,
    VnpyCtaCompatibilityError,
    VnpyCtaDependencyError,
    VnpyCtaDataError,
    VnpyCtaRuntime,
    VnpyCtaATLRunner,
    VnpyCtaRunSummary,
    build_audit_artifact,
    build_safe_manifest,
    load_audit_artifact,
    map_captured_order,
    load_vnpy_bindings,
    sanitize_error_message,
    save_audit_artifact,
)
from agentictrading import ATLAPIError, ATLConflictError
from agentictrading.models import ExecutionResult, Observation, Run, RunResult, Step


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_example(filename: str, module_name: str):
    path = _REPO_ROOT / "dashboard" / "examples" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _captured(**overrides) -> CapturedCtaOrder:
    values = {
        "order_id": "atl-cta-1",
        "timestamp": "2026-04-15T10:00:00-04:00",
        "symbol": "AAPL",
        "direction": "long",
        "offset": "open",
        "price": 101.25,
        "volume": 2.0,
        "stop": False,
        "lock": False,
        "net": False,
    }
    values.update(overrides)
    return CapturedCtaOrder(**values)


def _record(sequence: int = 0, *, status: str = "strategy_hold") -> VnpyCtaAuditRecord:
    return VnpyCtaAuditRecord(
        sequence=sequence,
        observation_timestamp=f"2026-04-15T{10 + sequence:02d}:00:00-04:00",
        signal_timestamp=None,
        status=status,
        bar={"symbol": "AAPL", "close": 100.0 + sequence},
    )


def test_public_import_does_not_load_optional_vnpy_dependencies():
    sys.modules.pop("vnpy", None)
    sys.modules.pop("vnpy_ctastrategy", None)

    module = importlib.reload(importlib.import_module("agentictrading.integrations.vnpy_cta"))

    assert module.ARTIFACT_SCHEMA_VERSION == "vnpy-cta-atl-v1"
    assert "vnpy" not in sys.modules
    assert "vnpy_ctastrategy" not in sys.modules


def test_maps_long_open_to_market_buy_and_audits_limit_price():
    mapped = map_captured_order(_captured(), symbol="AAPL", current_position=0)

    assert mapped.status == "mapped"
    assert mapped.reason is None
    assert mapped.warnings == ("limit_price_not_enforced",)
    assert mapped.order is not None
    assert mapped.order.to_dict() == {
        "symbol": "AAPL",
        "side": "buy",
        "quantity_type": "shares",
        "quantity": 2,
        "order_type": "market",
    }


def test_maps_short_close_to_sell_only_with_enough_position():
    captured = _captured(direction="short", offset="close", volume=3)

    mapped = map_captured_order(captured, symbol="AAPL", current_position=5)
    rejected = map_captured_order(captured, symbol="AAPL", current_position=2)

    assert mapped.order is not None
    assert mapped.order.side == "sell"
    assert mapped.order.quantity == 3
    assert rejected.order is None
    assert rejected.status == "local_rejection"
    assert rejected.reason == "sell_exceeds_position"


@pytest.mark.parametrize("volume", [0, -1, 1.5, math.nan, math.inf])
def test_rejects_non_positive_non_integer_or_non_finite_volume(volume):
    mapped = map_captured_order(
        _captured(volume=volume), symbol="AAPL", current_position=0
    )

    assert mapped.order is None
    assert mapped.status == "local_rejection"
    assert mapped.reason == "invalid_share_volume"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"direction": "short", "offset": "open"}, "short_not_supported"),
        ({"direction": "long", "offset": "close"}, "cover_not_supported"),
        ({"stop": True}, "stop_not_supported"),
        ({"lock": True}, "lock_not_supported"),
        ({"net": True}, "net_not_supported"),
    ],
)
def test_unsupported_cta_actions_are_explicit(overrides, reason):
    mapped = map_captured_order(
        _captured(**overrides), symbol="AAPL", current_position=5
    )

    assert mapped.order is None
    assert mapped.status == "unsupported_action"
    assert mapped.reason == reason


def test_rejects_wrong_symbol_and_invalid_audit_price():
    wrong_symbol = map_captured_order(
        _captured(symbol="MSFT"), symbol="AAPL", current_position=0
    )
    invalid_price = map_captured_order(
        _captured(price=math.nan), symbol="AAPL", current_position=0
    )

    assert (wrong_symbol.status, wrong_symbol.reason) == (
        "local_rejection",
        "symbol_mismatch",
    )
    assert (invalid_price.status, invalid_price.reason) == (
        "local_rejection",
        "invalid_order_price",
    )


def test_safe_manifest_and_error_message_remove_credentials():
    manifest = build_safe_manifest(
        {
            "strategy": "demo:DoubleMaStrategy",
            "settings": {
                "fast_window": 5,
                "api_key": "sk-super-secret",
                "nested": {"password": "open-sesame"},
            },
            "endpoint": "https://user:password@example.com/path",
            "authorization": "Bearer top-secret-token",
        }
    )
    error = sanitize_error_message(
        "request failed ATL_API_KEY=abc123 Bearer bearer-secret "
        "https://alice:pw@example.com/private"
    )
    encoded = json.dumps(manifest, sort_keys=True)

    assert manifest["strategy"] == "demo:DoubleMaStrategy"
    assert manifest["settings"]["fast_window"] == 5
    assert "sk-super-secret" not in encoded
    assert "open-sesame" not in encoded
    assert "password@example" not in encoded
    assert "top-secret-token" not in encoded
    assert "abc123" not in error
    assert "bearer-secret" not in error
    assert "alice:pw" not in error


def test_error_message_redacts_colon_and_json_style_credentials():
    """The equals-only regex missed header-dump and JSON-embedded credentials
    (e.g. an HTTP client's ``X-API-Key: <key>`` header repr, or a stringified
    JSON payload) — both are plausible shapes for a credential to leak into
    an exception's ``str()``.
    """
    error = sanitize_error_message(
        "request failed headers={'X-API-Key': 'ag_live_abc123'} "
        'body={"api_key": "sk-json-secret", "symbol": "AAPL"}'
    )
    assert "ag_live_abc123" not in error
    assert "sk-json-secret" not in error
    assert "AAPL" in error  # non-sensitive fields must survive redaction


def test_artifact_round_trip_has_stable_summary_and_sha256(tmp_path):
    artifact = build_audit_artifact(
        manifest=build_safe_manifest({"strategy": "demo:DoubleMaStrategy"}),
        records=(
            _record(0, status="warmup_hold"),
            _record(1, status="strategy_hold"),
            _record(2, status="error_hold"),
            _record(3, status="timeout_hold"),
        ),
    )
    path = tmp_path / "audit.json"

    digest = save_audit_artifact(artifact, path)
    loaded = load_audit_artifact(path)

    assert len(digest) == 64
    assert loaded == artifact
    assert loaded.schema_version == ARTIFACT_SCHEMA_VERSION
    assert loaded.summary == {
        "error_hold": 1,
        "strategy_hold": 1,
        "timeout_hold": 1,
        "total_records": 4,
        "warmup_hold": 1,
    }
    assert save_audit_artifact(loaded, path) == digest


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        json.dumps([]),
        json.dumps(
            {
                "schema_version": "unknown-v9",
                "manifest": {},
                "records": [],
                "summary": {"total_records": 0},
            }
        ),
    ],
)
def test_artifact_loader_rejects_invalid_json_shape_or_schema(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ArtifactValidationError):
        load_audit_artifact(path)


def test_artifact_rejects_unknown_status_and_duplicate_sequences():
    with pytest.raises(ArtifactValidationError, match="status"):
        _record(status="mystery")

    with pytest.raises(ArtifactValidationError, match="duplicate sequence"):
        VnpyCtaAuditArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            manifest={},
            records=(_record(0), _record(0)),
            summary={"strategy_hold": 2, "total_records": 2},
        )


class _Direction(Enum):
    LONG = "long"
    SHORT = "short"


class _Offset(Enum):
    OPEN = "open"
    CLOSE = "close"


class _Exchange(Enum):
    SMART = "SMART"


class _Interval(Enum):
    HOUR = "1h"


class _Status(Enum):
    ALLTRADED = "all traded"
    PARTTRADED = "part traded"
    REJECTED = "rejected"


class _OrderType(Enum):
    MARKET = "market"


class _EngineType(Enum):
    BACKTESTING = "backtesting"


class _DataObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _CtaTemplate:
    pass


class _RecordingStrategy(_CtaTemplate):
    def __init__(self, engine, strategy_name, vt_symbol, setting):
        self.cta_engine = engine
        self.strategy_name = strategy_name
        self.vt_symbol = vt_symbol
        self.setting = setting
        self.inited = False
        self.trading = False
        self.pos = 0
        self.events = []

    def on_init(self):
        self.events.append("init")
        self.cta_engine.load_bar(self.vt_symbol, 10, _Interval.HOUR, self.on_bar)

    def on_start(self):
        self.events.append("start")

    def on_stop(self):
        self.events.append("stop")

    def on_bar(self, bar):
        self.events.append(("bar", bar))
        self.cta_engine.send_order(
            self,
            _Direction.LONG,
            _Offset.OPEN,
            bar.close_price,
            2,
            False,
            False,
            False,
        )

    def on_order(self, order):
        self.events.append(("order", order))

    def on_trade(self, trade):
        self.events.append(("trade", trade))


def _fake_bindings(
    *, vnpy_version: str = "4.4.0", cta_version: str = "1.3.0"
) -> VnpyBindings:
    return VnpyBindings(
        vnpy_version=vnpy_version,
        cta_version=cta_version,
        CtaTemplate=_CtaTemplate,
        EngineType=_EngineType,
        BarData=_DataObject,
        OrderData=_DataObject,
        TradeData=_DataObject,
        Direction=_Direction,
        Offset=_Offset,
        Exchange=_Exchange,
        Interval=_Interval,
        Status=_Status,
        OrderType=_OrderType,
    )


def _bar_payload(hour: int = 10):
    return {
        "timestamp": f"2026-04-15T{hour:02d}:00:00-04:00",
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.5,
        "volume": 1_000_000.0,
    }


def test_load_vnpy_bindings_reports_missing_optional_dependency():
    def missing_import(name):
        raise ModuleNotFoundError(name)

    with pytest.raises(VnpyCtaDependencyError, match=r"agentictrading\[vnpy\]"):
        load_vnpy_bindings(import_module=missing_import)


@pytest.mark.parametrize(
    ("vnpy_version", "cta_version"),
    [("4.3.9", "1.3.0"), ("5.0.0", "1.3.0"), ("4.4.0", "1.2.9"), ("4.4.0", "2.0.0")],
)
def test_runtime_rejects_incompatible_vnpy_versions(vnpy_version, cta_version):
    with pytest.raises(VnpyCtaCompatibilityError):
        VnpyCtaRuntime(
            _RecordingStrategy,
            symbol="AAPL",
            settings={},
            bindings=_fake_bindings(
                vnpy_version=vnpy_version, cta_version=cta_version
            ),
        )


def test_runtime_runs_lifecycle_builds_bar_and_captures_order():
    runtime = VnpyCtaRuntime(
        _RecordingStrategy,
        symbol="AAPL",
        settings={"fast_window": 5},
        bindings=_fake_bindings(),
    )

    runtime.start()
    captured = runtime.on_bar(_bar_payload())

    assert runtime.strategy.inited is True
    assert runtime.strategy.trading is True
    assert runtime.strategy.vt_symbol == "AAPL.SMART"
    assert runtime.strategy.setting == {"fast_window": 5}
    assert runtime.strategy.events[:2] == ["init", "start"]
    bar = runtime.strategy.events[2][1]
    assert bar.gateway_name == "ATL"
    assert bar.symbol == "AAPL"
    assert bar.exchange is _Exchange.SMART
    assert bar.interval is _Interval.HOUR
    assert bar.datetime.isoformat() == _bar_payload()["timestamp"]
    assert bar.close_price == 101.5
    assert captured == (
        _captured(order_id="atl-cta-1", price=101.5, volume=2),
    )
    assert runtime.engine.drain_captured_orders() == ()
    assert any(event["type"] == "history_preload_unavailable" for event in runtime.events)

    runtime.stop()
    assert runtime.strategy.events[-1] == "stop"
    assert runtime.strategy.trading is False


def test_runtime_syncs_position_and_emits_fill_callbacks():
    runtime = VnpyCtaRuntime(
        _RecordingStrategy,
        symbol="AAPL",
        settings={},
        bindings=_fake_bindings(),
    )
    runtime.start()
    captured = runtime.on_bar(_bar_payload())
    result = ExecutionResult(
        accepted=True,
        fills=[
            {
                "symbol": "AAPL",
                "side": "buy",
                "requested_quantity": 2,
                "filled_quantity": 2,
                "fill_price": 102.0,
            }
        ],
        validation={"passed": True, "warnings": [], "rejections": []},
    )

    runtime.apply_execution_result(
        result,
        captured_orders=captured,
        timestamp="2026-04-15T11:00:00-04:00",
    )

    order = next(
        event[1]
        for event in runtime.strategy.events
        if isinstance(event, tuple) and event[0] == "order"
    )
    trade = next(
        event[1]
        for event in runtime.strategy.events
        if isinstance(event, tuple) and event[0] == "trade"
    )
    assert order.orderid == "atl-cta-1"
    assert order.status is _Status.ALLTRADED
    assert order.traded == 2
    assert trade.orderid == "atl-cta-1"
    assert trade.price == 102.0
    assert runtime.strategy.pos == 2

    runtime.sync_position(7)
    assert runtime.strategy.pos == 7


class _TwoOrderStrategy(_CtaTemplate):
    """Emits two same-side orders of different sizes from a single on_bar()."""

    def __init__(self, engine, strategy_name, vt_symbol, setting):
        self.cta_engine = engine
        self.pos = 0
        self.events = []

    def on_init(self):
        return None

    def on_start(self):
        return None

    def on_stop(self):
        return None

    def on_bar(self, bar):
        self.events.append(("bar", bar))
        self.cta_engine.send_order(
            self, _Direction.LONG, _Offset.OPEN, bar.close_price, 1, False, False, False
        )
        self.cta_engine.send_order(
            self, _Direction.LONG, _Offset.OPEN, bar.close_price, 3, False, False, False
        )

    def on_order(self, order):
        self.events.append(("order", order))

    def on_trade(self, trade):
        self.events.append(("trade", trade))


def test_apply_execution_result_matches_multiple_same_side_orders_by_quantity():
    """(symbol, side) alone can't disambiguate two same-side captured orders;
    ATL's orders-v1 wire schema has no client order id, so requested_quantity
    is the only extra signal available — verify it's used to avoid pairing
    fills to the wrong captured order.
    """
    runtime = VnpyCtaRuntime(
        _TwoOrderStrategy,
        symbol="AAPL",
        settings={},
        bindings=_fake_bindings(),
    )
    runtime.start()
    captured = runtime.on_bar(_bar_payload())
    assert [order.volume for order in captured] == [1, 3]

    # Fills listed in the OPPOSITE order from submission: a pure (symbol, side)
    # first-match would pair the 1-share order with the 3-share fill.
    result = ExecutionResult(
        accepted=True,
        fills=[
            {
                "symbol": "AAPL",
                "side": "buy",
                "requested_quantity": 3,
                "filled_quantity": 3,
                "fill_price": 103.0,
            },
            {
                "symbol": "AAPL",
                "side": "buy",
                "requested_quantity": 1,
                "filled_quantity": 1,
                "fill_price": 101.0,
            },
        ],
        validation={"passed": True, "warnings": [], "rejections": []},
    )

    runtime.apply_execution_result(
        result,
        captured_orders=captured,
        timestamp="2026-04-15T11:00:00-04:00",
    )

    trades = [event[1] for event in runtime.strategy.events if event[0] == "trade"]
    assert len(trades) == 2
    by_order_id = {trade.orderid: trade for trade in trades}
    assert by_order_id[captured[0].order_id].volume == 1
    assert by_order_id[captured[0].order_id].price == 101.0
    assert by_order_id[captured[1].order_id].volume == 3
    assert by_order_id[captured[1].order_id].price == 103.0
    assert runtime.strategy.pos == 4


def test_runtime_emits_rejected_order_callback_without_trade():
    runtime = VnpyCtaRuntime(
        _RecordingStrategy,
        symbol="AAPL",
        settings={},
        bindings=_fake_bindings(),
    )
    runtime.start()
    captured = runtime.on_bar(_bar_payload())

    runtime.reject_captured_order(
        captured[0],
        reason="exceeds_max_position_weight",
        timestamp="2026-04-15T11:00:00-04:00",
    )

    order_events = [
        event[1]
        for event in runtime.strategy.events
        if isinstance(event, tuple) and event[0] == "order"
    ]
    trade_events = [
        event[1]
        for event in runtime.strategy.events
        if isinstance(event, tuple) and event[0] == "trade"
    ]
    assert len(order_events) == 1
    assert order_events[0].status is _Status.REJECTED
    assert trade_events == []
    assert runtime.strategy.pos == 0


@pytest.mark.skipif(
    importlib.util.find_spec("vnpy") is None
    or importlib.util.find_spec("vnpy_ctastrategy") is None,
    reason="optional vn.py CTA dependencies are not installed",
)
def test_runtime_constructs_real_vnpy_objects_when_optional_extra_is_installed():
    bindings = load_vnpy_bindings()

    class RealStrategy(bindings.CtaTemplate):
        def on_init(self):
            return None

    runtime = VnpyCtaRuntime(
        RealStrategy,
        symbol="AAPL",
        settings={},
        bindings=bindings,
    )
    runtime.start()

    captured = runtime.on_bar(_bar_payload())
    bar = runtime._bar_data(_bar_payload())

    assert captured == ()
    assert isinstance(bar, bindings.BarData)
    assert bar.vt_symbol == "AAPL.SMART"
    runtime.stop()


class _AdapterEngine:
    def drain_captured_orders(self):
        return ()


class _AdapterRuntime:
    def __init__(self, order_specs=()):
        self.engine = _AdapterEngine()
        self.order_specs = list(order_specs)
        self.started = False
        self.stopped = False
        self.position_syncs = []
        self.bars = []
        self.applied = []
        self.rejected = []
        self.raise_on_bar = None
        self.strategy = SimpleNamespace(__class__=SimpleNamespace(__name__="FakeStrategy"))

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def sync_position(self, quantity):
        self.position_syncs.append(quantity)

    def on_bar(self, bar):
        self.bars.append(dict(bar))
        if self.raise_on_bar is not None:
            raise self.raise_on_bar
        specs = self.order_specs.pop(0) if self.order_specs else ()
        return tuple(
            _captured(
                order_id=f"atl-cta-{index + 1}",
                timestamp=bar["timestamp"],
                price=bar["close"],
                **spec,
            )
            for index, spec in enumerate(specs)
        )

    def apply_execution_result(self, result, *, captured_orders, timestamp):
        self.applied.append((result, tuple(captured_orders), timestamp))

    def reject_captured_order(self, captured, *, reason, timestamp):
        self.rejected.append((captured, reason, timestamp))

    def drain_captured_orders(self):
        return self.engine.drain_captured_orders()


def _observation(hour: int, *, position: int = 0, bar_overrides=None) -> Observation:
    bar = _bar_payload(hour)
    if bar_overrides:
        bar.update(bar_overrides)
    positions = []
    if position:
        positions.append({"symbol": "AAPL", "quantity": position})
    return Observation(
        market={"bars": {"AAPL": bar}, "features": {}, "events": []},
        portfolio={"cash": 1_000.0, "equity": 1_000.0, "positions": positions},
    )


def _execution(*, fills=None, rejections=None) -> ExecutionResult:
    return ExecutionResult(
        accepted=True,
        fills=list(fills or []),
        validation={
            "passed": not rejections,
            "warnings": [],
            "rejections": list(rejections or []),
        },
        portfolio_after={"cash": 1_000.0, "equity": 1_000.0, "positions": []},
        run_status="running",
    )


def _adapter(runtime: _AdapterRuntime) -> VnpyCtaAdapter:
    return VnpyCtaAdapter(
        runtime,
        symbol="AAPL",
        manifest={"strategy": "tests:FakeStrategy"},
    )


def test_adapter_first_observation_is_warmup_hold():
    runtime = _AdapterRuntime()
    adapter = _adapter(runtime)

    decision = adapter.decide(_observation(10))

    assert runtime.started is True
    assert runtime.bars == []
    assert decision.orders == []
    assert decision.trace["status"] == "warmup_hold"
    assert decision.trace["observation_timestamp"] == _bar_payload(10)["timestamp"]
    adapter.on_execution_result(_execution())
    artifact = adapter.finalize_artifact()
    assert artifact.records[0].bar["timestamp"] == _bar_payload(10)["timestamp"]
    assert artifact.records[0].signal_bar == {}
    assert artifact.summary == {
        "total_records": 1,
        "warmup_hold": 1,
    }


def test_adapter_processes_only_previous_bar_and_syncs_current_position():
    runtime = _AdapterRuntime(order_specs=[({"volume": 2},)])
    adapter = _adapter(runtime)
    adapter.decide(_observation(10))
    adapter.on_execution_result(_execution())

    decision = adapter.decide(_observation(11, position=3))

    assert runtime.position_syncs == [0, 3]
    assert [bar["timestamp"] for bar in runtime.bars] == [
        _bar_payload(10)["timestamp"]
    ]
    assert decision.trace["signal_timestamp"] == _bar_payload(10)["timestamp"]
    assert decision.trace["observation_timestamp"] == _bar_payload(11)["timestamp"]
    assert [order.to_dict() for order in decision.orders] == [
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity_type": "shares",
            "quantity": 2,
            "order_type": "market",
        }
    ]
    record = adapter.finalize_artifact().records[-1]
    assert record.bar["timestamp"] == _bar_payload(11)["timestamp"]
    assert record.signal_bar["timestamp"] == _bar_payload(10)["timestamp"]


def test_adapter_execution_hook_updates_runtime_and_artifact():
    runtime = _AdapterRuntime(order_specs=[({"volume": 2},)])
    adapter = _adapter(runtime)
    adapter.decide(_observation(10))
    adapter.on_execution_result(_execution())
    adapter.decide(_observation(11))
    fill = {
        "symbol": "AAPL",
        "side": "buy",
        "requested_quantity": 2,
        "filled_quantity": 2,
        "fill_price": 102.0,
    }

    adapter.on_execution_result(_execution(fills=[fill]))
    artifact = adapter.finalize_artifact()

    assert len(runtime.applied) == 2
    assert runtime.applied[-1][2] == _bar_payload(11)["timestamp"]
    assert artifact.records[-1].execution["fills"] == [fill]
    assert artifact.records[-1].diagnostics == {"fills": 1}
    assert artifact.summary["fills"] == 1


def test_adapter_mixed_supported_and_unsupported_orders_are_audited():
    runtime = _AdapterRuntime(
        order_specs=[
            (
                {"volume": 1},
                {"direction": "short", "offset": "open", "volume": 1},
            )
        ]
    )
    adapter = _adapter(runtime)
    adapter.decide(_observation(10))
    adapter.on_execution_result(_execution())

    decision = adapter.decide(_observation(11, position=5))

    assert len(decision.orders) == 1
    assert decision.trace["status"] == "partial_submission"
    assert decision.trace["diagnostics"] == {"unsupported_actions": 1}
    assert len(runtime.rejected) == 1
    assert runtime.rejected[0][1] == "short_not_supported"


def test_adapter_strategy_exception_is_error_hold_not_strategy_hold():
    runtime = _AdapterRuntime()
    adapter = _adapter(runtime)
    adapter.decide(_observation(10))
    adapter.on_execution_result(_execution())
    runtime.raise_on_bar = RuntimeError("API_KEY=secret strategy failed")

    decision = adapter.decide(_observation(11))
    artifact = adapter.finalize_artifact()

    assert decision.orders == []
    assert decision.trace["status"] == "error_hold"
    assert "secret" not in artifact.records[-1].error
    assert runtime.rejected == []
    assert artifact.summary["error_hold"] == 1
    assert artifact.summary.get("strategy_hold", 0) == 0


@pytest.mark.parametrize(
    "observation",
    [
        Observation(market={"bars": {}}, portfolio={"positions": []}),
        _observation(10, bar_overrides={"timestamp": "2026-04-15T10:00:00"}),
        _observation(10, bar_overrides={"close": math.nan}),
    ],
)
def test_adapter_rejects_missing_naive_or_invalid_bars(observation):
    adapter = _adapter(_AdapterRuntime())

    with pytest.raises(VnpyCtaDataError):
        adapter.decide(observation)


def test_adapter_rejects_duplicate_observation_timestamp():
    adapter = _adapter(_AdapterRuntime())
    adapter.decide(_observation(10))
    adapter.on_execution_result(_execution())

    with pytest.raises(VnpyCtaDataError, match="duplicate"):
        adapter.decide(_observation(10))


def test_adapter_marks_missing_execution_hook_as_timeout_on_next_step():
    runtime = _AdapterRuntime()
    adapter = _adapter(runtime)
    adapter.decide(_observation(10))

    adapter.decide(_observation(11))
    artifact = adapter.finalize_artifact()

    assert artifact.records[0].status == "timeout_hold"
    assert artifact.summary["timeout_hold"] == 1


def test_adapter_completion_stops_runtime_and_records_terminal_bar():
    runtime = _AdapterRuntime()
    adapter = _adapter(runtime)
    adapter.decide(_observation(10))
    adapter.on_execution_result(_execution())

    adapter.on_run_completed(RunResult(run_id="run_test", status="completed"))
    artifact = adapter.finalize_artifact()

    assert runtime.stopped is True
    assert artifact.records[-1].status == "terminal_bar_skipped"
    assert artifact.records[-1].signal_timestamp == _bar_payload(10)["timestamp"]
    assert artifact.summary["terminal_bar_skipped"] == 1


def test_artifact_summary_aggregates_record_diagnostics():
    record = VnpyCtaAuditRecord(
        sequence=0,
        observation_timestamp=_bar_payload(10)["timestamp"],
        status="partial_submission",
        diagnostics={"unsupported_actions": 2, "local_rejections": 1},
    )

    artifact = build_audit_artifact(manifest={}, records=[record])

    assert artifact.summary == {
        "local_rejections": 1,
        "partial_submission": 1,
        "total_records": 1,
        "unsupported_actions": 2,
    }


class _RunnerClient:
    def __init__(self, steps, *, result=None, conflict_code=None, next_error=None):
        self.steps = list(steps)
        self.result = result or RunResult(
            run_id="run_vnpy",
            result_run_id="ext_vnpy",
            status="completed",
            metrics={"total_return": 0.02, "num_trades": 1},
            equity_curve=[{"timestamp": "2026-04-15T11:00:00-04:00", "equity": 1020}],
        )
        self.conflict_code = conflict_code
        self.next_error = next_error
        self.create_calls = []
        self.submitted = []

    def create_run(self, agent_version_id, **kwargs):
        self.create_calls.append((agent_version_id, kwargs))
        return Run(run_id="run_vnpy", status="running")

    def get_next_step(self, run_id):
        if self.next_error is not None:
            raise self.next_error
        return self.steps.pop(0)

    def submit_decision(self, run_id, step_id, decision):
        self.submitted.append(decision)
        if self.conflict_code:
            code = self.conflict_code
            self.conflict_code = None
            raise ATLConflictError("late", code=code)
        fills = []
        if decision.orders:
            order = decision.orders[0]
            fills = [
                {
                    "symbol": order.symbol,
                    "side": order.side,
                    "requested_quantity": order.quantity,
                    "filled_quantity": order.quantity,
                    "fill_price": 102.0,
                }
            ]
        return _execution(fills=fills)

    def get_run_result(self, run_id):
        return self.result

    def get_run_metrics(self, run_id):
        return self.result.metrics

    def wait(self, seconds):
        return None


def _awaiting_step(sequence: int, hour: int) -> Step:
    return Step(
        status="awaiting_decision",
        run_id="run_vnpy",
        step_id=f"step_{sequence}",
        sequence=sequence,
        timestamp=_bar_payload(hour)["timestamp"],
        observation=_observation(hour),
    )


def _completed_step() -> Step:
    return Step(status="completed", run_id="run_vnpy", result_run_id="ext_vnpy")


def test_vnpy_runner_completes_with_existing_agent_runner_and_safe_config(tmp_path):
    runtime = _AdapterRuntime(order_specs=[({"volume": 2},)])
    adapter = VnpyCtaAdapter(
        runtime,
        symbol="AAPL",
        manifest={
            "strategy": "tests:FakeStrategy",
            "settings": {"fast_window": 5, "api_key": "never-persist"},
        },
    )
    client = _RunnerClient(
        [_awaiting_step(0, 10), _awaiting_step(1, 11), _completed_step()]
    )
    artifact_path = tmp_path / "vnpy-run.json"
    runner = VnpyCtaATLRunner(client, adapter, artifact_path=artifact_path)

    summary = runner.run_backtest(
        agent_version_id="agv_vnpy",
        start_date="2026-04-15",
        end_date="2026-04-16",
        poll_interval=0.001,
    )

    assert isinstance(summary, VnpyCtaRunSummary)
    assert summary.run_id == "run_vnpy"
    assert summary.clean is True
    assert summary.metrics["total_return"] == 0.02
    assert len(summary.artifact_sha256) == 64
    assert artifact_path.exists()
    assert len(client.submitted) == 2
    _, create_kwargs = client.create_calls[0]
    assert create_kwargs["environment_id"] == "us-equity-hourly-v1"
    assert create_kwargs["symbols"] == ["AAPL"]
    encoded_config = json.dumps(create_kwargs["config"], sort_keys=True)
    assert create_kwargs["config"]["integration"] == "vnpy_cta"
    assert "never-persist" not in encoded_config


def test_vnpy_runner_marks_autoheld_timeout_unclean(tmp_path):
    adapter = _adapter(_AdapterRuntime())
    client = _RunnerClient(
        [_awaiting_step(0, 10), _completed_step()],
        conflict_code="decision_deadline_exceeded",
    )
    runner = VnpyCtaATLRunner(
        client, adapter, artifact_path=tmp_path / "timeout.json"
    )

    summary = runner.run_backtest(
        agent_version_id="agv_vnpy",
        start_date="2026-04-15",
        end_date="2026-04-16",
        poll_interval=0.001,
    )

    assert summary.clean is False
    assert summary.diagnostics["timeout_hold"] == 1


def test_vnpy_runner_persists_partial_artifact_and_reraises_api_error(tmp_path):
    artifact_path = tmp_path / "failed.json"
    adapter = _adapter(_AdapterRuntime())
    error = ATLAPIError("backend failed", code="backend_failed")
    client = _RunnerClient([], next_error=error)
    runner = VnpyCtaATLRunner(client, adapter, artifact_path=artifact_path)

    with pytest.raises(ATLAPIError) as raised:
        runner.run_backtest(
            agent_version_id="agv_vnpy",
            start_date="2026-04-15",
            end_date="2026-04-16",
        )

    assert raised.value is error
    assert raised.value.run_id == "run_vnpy"
    assert artifact_path.exists()
    artifact = load_audit_artifact(artifact_path)
    assert artifact.summary["run_error"] == 1
    assert artifact.summary.get("error_hold", 0) == 0


def test_vnpy_runner_classifies_bad_ohlcv_as_fatal_data_error(tmp_path):
    artifact_path = tmp_path / "bad-bar.json"
    adapter = _adapter(_AdapterRuntime())
    bad_step = _awaiting_step(0, 10)
    bad_step.observation.market["bars"]["AAPL"]["close"] = math.nan
    client = _RunnerClient([bad_step])
    runner = VnpyCtaATLRunner(client, adapter, artifact_path=artifact_path)

    with pytest.raises(VnpyCtaDataError):
        runner.run_backtest(
            agent_version_id="agv_vnpy",
            start_date="2026-04-15",
            end_date="2026-04-16",
        )

    artifact = load_audit_artifact(artifact_path)
    assert artifact.summary["fatal_data_error"] == 1
    assert artifact.summary.get("run_error", 0) == 0


@pytest.mark.parametrize(
    ("start", "end"),
    [("not-a-date", "2026-04-16"), ("2026-04-17", "2026-04-16")],
)
def test_vnpy_runner_rejects_bad_dates_before_creating_run(tmp_path, start, end):
    client = _RunnerClient([])
    runner = VnpyCtaATLRunner(
        client,
        _adapter(_AdapterRuntime()),
        artifact_path=tmp_path / "unused.json",
    )

    with pytest.raises(ValueError):
        runner.run_backtest(
            agent_version_id="agv_vnpy",
            start_date=start,
            end_date=end,
        )

    assert client.create_calls == []


def test_vnpy_cli_help_is_lazy_and_needs_no_credentials(tmp_path):
    script = _REPO_ROOT / "dashboard" / "examples" / "vnpy_cta_atl_backtest.py"
    blocked = tmp_path / "blocked-imports"
    blocked.mkdir()
    (blocked / "vnpy_ctastrategy.py").write_text(
        "raise RuntimeError('vnpy_ctastrategy imported during --help')\n",
        encoding="utf-8",
    )
    vnpy_package = blocked / "vnpy"
    vnpy_package.mkdir()
    (vnpy_package / "__init__.py").write_text(
        "raise RuntimeError('vnpy imported during --help')\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    source = str(_REPO_ROOT / "packaging" / "agentictrading" / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(blocked), source, env.get("PYTHONPATH")) if part
    )
    env.pop("ATL_BASE_URL", None)
    env.pop("ATL_API_KEY", None)
    env.pop("ATL_AGENT_VERSION_ID", None)

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    for option in (
        "--strategy",
        "--settings-file",
        "--symbol",
        "--start",
        "--end",
        "--initial-cash",
        "--output",
    ):
        assert option in result.stdout


def test_vnpy_cli_validates_inputs_environment_and_strategy_spec(tmp_path, monkeypatch):
    cli = _load_example("vnpy_cta_atl_backtest.py", "_test_vnpy_cta_cli_validation")

    assert cli.validate_inputs("aapl", "2026-04-15", "2026-04-16") == "AAPL"
    with pytest.raises(ValueError, match="AAPL only"):
        cli.validate_inputs("MSFT", "2026-04-15", "2026-04-16")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        cli.validate_inputs("AAPL", "bad-date", "2026-04-16")
    with pytest.raises(ValueError, match="end date"):
        cli.validate_inputs("AAPL", "2026-04-17", "2026-04-16")
    with pytest.raises(ValueError) as missing:
        cli.required_environment({})
    for name in ("ATL_BASE_URL", "ATL_API_KEY", "ATL_AGENT_VERSION_ID"):
        assert name in str(missing.value)

    module_path = tmp_path / "trusted_strategy.py"
    module_path.write_text("class TrustedStrategy:\n    pass\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    strategy = cli.load_strategy_class("trusted_strategy:TrustedStrategy")
    assert strategy.__name__ == "TrustedStrategy"
    with pytest.raises(ValueError, match="module:Class"):
        cli.load_strategy_class("trusted_strategy")


def test_vnpy_cli_initializes_strategy_before_client_and_redacts_settings(
    tmp_path, monkeypatch, capsys
):
    cli = _load_example("vnpy_cta_atl_backtest.py", "_test_vnpy_cta_cli_run")
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"fast_window": 2, "api_key": "settings-secret"}),
        encoding="utf-8",
    )
    events = []
    seen = {}

    class FakeRuntime:
        def __init__(self, strategy_class, *, symbol, settings):
            events.append("runtime")
            seen["runtime_settings"] = dict(settings)
            self.bindings = SimpleNamespace(vnpy_version="4.4.0", cta_version="1.3.0")

    class FakeAdapter:
        def __init__(self, runtime, *, symbol, manifest):
            events.append("adapter")
            seen["manifest"] = manifest

    class FakeClient:
        def __init__(self, *, base_url, api_key):
            events.append("client")
            assert api_key == "atl-secret"

    class FakeRunner:
        def __init__(self, client, adapter, *, artifact_path):
            events.append("runner")
            self.artifact_path = Path(artifact_path)

        def run_backtest(self, **kwargs):
            events.append("run")
            artifact = build_audit_artifact(manifest={}, records=())
            return VnpyCtaRunSummary(
                result=RunResult(
                    run_id="run_cli",
                    status="completed",
                    metrics={"total_return": 0.01},
                    raw={"compare_url": "/compare?run_ids=ext_cli"},
                ),
                artifact=artifact,
                artifact_path=self.artifact_path,
                artifact_sha256="a" * 64,
                clean=True,
            )

    def fake_load_strategy(spec):
        events.append("strategy_import")
        return type("Strategy", (), {})

    monkeypatch.setattr(cli, "load_strategy_class", fake_load_strategy)
    monkeypatch.setattr(cli, "VnpyCtaRuntime", FakeRuntime)
    monkeypatch.setattr(cli, "VnpyCtaAdapter", FakeAdapter)
    monkeypatch.setattr(cli, "ATLClient", FakeClient)
    monkeypatch.setattr(cli, "VnpyCtaATLRunner", FakeRunner)

    summary = cli.run_cli(
        [
            "--strategy",
            "trusted:Strategy",
            "--settings-file",
            str(settings_path),
            "--symbol",
            "AAPL",
            "--start",
            "2026-04-15",
            "--end",
            "2026-04-16",
            "--output",
            str(tmp_path / "audit.json"),
        ],
        environ={
            "ATL_BASE_URL": "https://atl.example",
            "ATL_API_KEY": "atl-secret",
            "ATL_AGENT_VERSION_ID": "agv_cli",
        },
    )

    assert summary.run_id == "run_cli"
    assert events == ["strategy_import", "runtime", "adapter", "client", "runner", "run"]
    assert seen["runtime_settings"]["api_key"] == "settings-secret"
    encoded_manifest = json.dumps(seen["manifest"], sort_keys=True)
    assert "settings-secret" not in encoded_manifest
    assert "atl-secret" not in capsys.readouterr().out


def test_vnpy_cli_summary_contains_result_metrics_diagnostics_and_artifact(tmp_path):
    cli = _load_example("vnpy_cta_atl_backtest.py", "_test_vnpy_cta_cli_summary")
    artifact = build_audit_artifact(
        manifest={}, records=(_record(status="strategy_hold"),)
    )
    summary = VnpyCtaRunSummary(
        result=RunResult(
            run_id="run_summary",
            status="completed",
            metrics={"total_return": 0.025, "sharpe_ratio": 1.2},
            raw={"compare_url": "/compare?run_ids=ext_summary"},
        ),
        artifact=artifact,
        artifact_path=tmp_path / "summary.json",
        artifact_sha256="b" * 64,
        clean=False,
    )

    output = cli.format_summary(summary, base_url="https://atl.example/api")

    assert "run_summary" in output
    assert "https://atl.example/compare?run_ids=ext_summary" in output
    assert "total_return: 0.025" in output
    assert "sharpe_ratio: 1.2" in output
    assert "strategy_hold: 1" in output
    assert "error_hold: 0" in output
    assert str(tmp_path / "summary.json") in output
    assert "b" * 64 in output


def test_vnpy_double_ma_example_streams_bars_and_emits_buy_then_sell(monkeypatch):
    class FakeCtaTemplate:
        def __init__(self, engine, strategy_name, vt_symbol, setting):
            self.cta_engine = engine
            self.strategy_name = strategy_name
            self.vt_symbol = vt_symbol
            self.pos = 0
            self.buy_calls = []
            self.sell_calls = []
            for name, value in setting.items():
                setattr(self, name, value)

        def buy(self, price, volume):
            self.buy_calls.append((price, volume))

        def sell(self, price, volume):
            self.sell_calls.append((price, volume))

        def write_log(self, message):
            return None

        def put_event(self):
            return None

    fake_cta = ModuleType("vnpy_ctastrategy")
    fake_cta.CtaTemplate = FakeCtaTemplate
    fake_vnpy = ModuleType("vnpy")
    fake_trader = ModuleType("vnpy.trader")
    fake_object = ModuleType("vnpy.trader.object")
    fake_object.BarData = SimpleNamespace
    monkeypatch.setitem(sys.modules, "vnpy", fake_vnpy)
    monkeypatch.setitem(sys.modules, "vnpy.trader", fake_trader)
    monkeypatch.setitem(sys.modules, "vnpy.trader.object", fake_object)
    monkeypatch.setitem(sys.modules, "vnpy_ctastrategy", fake_cta)
    example = _load_example(
        "vnpy_cta_double_ma_strategy.py", "_test_vnpy_double_ma_strategy"
    )
    strategy = example.DoubleMaStrategy(
        object(),
        "double_ma",
        "AAPL.SMART",
        {"fast_window": 2, "slow_window": 3, "fixed_size": 1},
    )

    for close in (3, 2, 1, 2):
        strategy.on_bar(SimpleNamespace(close_price=close))
    assert strategy.buy_calls == []

    strategy.on_bar(SimpleNamespace(close_price=3))
    assert strategy.buy_calls == [(3.0, 1)]
    strategy.pos = 1
    strategy.on_bar(SimpleNamespace(close_price=2))
    strategy.on_bar(SimpleNamespace(close_price=1))
    assert strategy.sell_calls == [(1.0, 1)]


def test_vnpy_cta_guide_covers_execution_contract_and_limits():
    guide = (_REPO_ROOT / "docs" / "integrations" / "vnpy-cta.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "ATL AAPL hourly OHLCV",
        "module:Class",
        "Alpaca key",
        "T+1",
        "25%",
        "load_bar",
        "TargetPosTemplate",
        "A-shares",
        "live trading",
        "artifact",
        "does not mean the strategy is profitable",
    ):
        assert required in guide

    external_agents = (
        _REPO_ROOT / "docs" / "source" / "lab" / "external_agents.rst"
    ).read_text(encoding="utf-8")
    assert "Run a local vn.py CTA strategy" in external_agents
    assert "docs/integrations/vnpy-cta.md" in external_agents


class _DeterministicCtaClient:
    def __init__(self):
        self.closes = (3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0)
        self.start = datetime(2026, 4, 15, 10, tzinfo=timezone.utc)
        self.index = 0
        self.position = 0
        self.minimum_position = 0
        self.orders = []
        self.rejections = []

    def create_run(self, agent_version_id, **kwargs):
        return Run(run_id="run_real_vnpy_offline", status="running")

    def _timestamp(self, index):
        return (self.start + timedelta(hours=index)).isoformat()

    def get_next_step(self, run_id):
        if self.index >= len(self.closes):
            return Step(
                status="completed",
                run_id=run_id,
                result_run_id="ext_real_vnpy_offline",
            )
        close = self.closes[self.index]
        timestamp = self._timestamp(self.index)
        positions = (
            [{"symbol": "AAPL", "quantity": self.position}]
            if self.position
            else []
        )
        observation = Observation(
            market={
                "bars": {
                    "AAPL": {
                        "timestamp": timestamp,
                        "open": close,
                        "high": close + 0.25,
                        "low": close - 0.25,
                        "close": close,
                        "volume": 10_000.0,
                    }
                },
                "features": {},
                "events": [],
            },
            portfolio={
                "cash": 1_000.0,
                "equity": 1_000.0,
                "positions": positions,
            },
        )
        return Step(
            status="awaiting_decision",
            run_id=run_id,
            step_id=f"step_{self.index}",
            sequence=self.index,
            timestamp=timestamp,
            observation=observation,
        )

    def submit_decision(self, run_id, step_id, decision):
        timestamp = self._timestamp(self.index)
        fills = []
        for order in decision.orders:
            quantity = int(order.quantity)
            if order.side == "buy":
                self.position += quantity
            elif order.side == "sell" and quantity <= self.position:
                self.position -= quantity
            else:
                rejection = {
                    "order": order.to_dict(),
                    "reason": "unexpected_offline_order",
                }
                self.rejections.append(rejection)
                continue
            fill = {
                "symbol": order.symbol,
                "side": order.side,
                "requested_quantity": quantity,
                "filled_quantity": quantity,
                "fill_price": self.closes[self.index],
            }
            fills.append(fill)
            self.orders.append(
                {
                    "side": order.side,
                    "quantity": quantity,
                    "signal_timestamp": decision.trace["signal_timestamp"],
                    "execution_timestamp": timestamp,
                }
            )
        self.minimum_position = min(self.minimum_position, self.position)
        self.index += 1
        positions = (
            [{"symbol": "AAPL", "quantity": self.position}]
            if self.position
            else []
        )
        return ExecutionResult(
            accepted=not self.rejections,
            fills=fills,
            validation={
                "passed": not self.rejections,
                "warnings": [],
                "rejections": list(self.rejections),
            },
            portfolio_after={
                "cash": 1_000.0,
                "equity": 1_000.0,
                "positions": positions,
            },
            run_status="running",
        )

    def get_run_result(self, run_id):
        return RunResult(
            run_id=run_id,
            result_run_id="ext_real_vnpy_offline",
            status="completed",
            metrics={"num_trades": len(self.orders), "total_return": 0.0},
            equity_curve=[
                {"timestamp": self._timestamp(index), "equity": 1_000.0}
                for index in range(len(self.closes))
            ],
        )

    def get_run_metrics(self, run_id):
        return {"num_trades": len(self.orders), "total_return": 0.0}

    def wait(self, seconds):
        return None


@pytest.mark.skipif(
    importlib.util.find_spec("vnpy") is None
    or importlib.util.find_spec("vnpy_ctastrategy") is None,
    reason="optional vn.py CTA dependencies are not installed",
)
def test_real_vnpy_double_ma_offline_loop_is_t1_long_only_and_deterministic(tmp_path):
    example = _load_example(
        "vnpy_cta_double_ma_strategy.py", "_test_real_vnpy_double_ma_strategy"
    )

    def run_once(name):
        runtime = VnpyCtaRuntime(
            example.DoubleMaStrategy,
            symbol="AAPL",
            settings={"fast_window": 2, "slow_window": 3, "fixed_size": 1},
        )
        adapter = VnpyCtaAdapter(
            runtime,
            symbol="AAPL",
            manifest={"strategy": "example:DoubleMaStrategy"},
        )
        client = _DeterministicCtaClient()
        summary = VnpyCtaATLRunner(
            client,
            adapter,
            artifact_path=tmp_path / f"{name}.json",
        ).run_backtest(
            agent_version_id="agv_real_vnpy_offline",
            start_date="2026-04-15",
            end_date="2026-04-16",
            poll_interval=0.001,
        )
        return client, summary

    first_client, first = run_once("first")
    second_client, second = run_once("second")

    assert [order["side"] for order in first_client.orders] == ["buy", "sell"]
    assert first_client.orders == second_client.orders
    assert first_client.minimum_position == 0
    assert first_client.position == 0
    assert first_client.rejections == []
    for order in first_client.orders:
        signal = datetime.fromisoformat(order["signal_timestamp"])
        execution = datetime.fromisoformat(order["execution_timestamp"])
        assert execution > signal

    for counter in (
        "error_hold",
        "timeout_hold",
        "unsupported_actions",
        "local_rejections",
        "atl_rejections",
    ):
        assert first.diagnostics.get(counter, 0) == 0
    assert first.clean is True
    assert first.diagnostics == second.diagnostics
    assert first.artifact.to_dict() == second.artifact.to_dict()
    assert first.artifact_sha256 == second.artifact_sha256
