"""Bridge tests for the hosted TradingAgents runtime.

The bridge is the money boundary: it is the last ATL code that runs before a
multi-agent TradingAgents analysis bills the platform's OpenRouter account, and
the first that runs after upstream returns text ATL has to trust. These tests
therefore exercise the *real* ``TradingAgentsDecisionGenerator`` — rating
mapping, artifact validation and the 0.3.x version guard all stay live — and
fake only the upstream LangGraph, which is the slow, billable, networked part.

``agentictrading`` is the repo's own SDK package but is not a backend
dependency, so its source tree is put on ``sys.path`` here the same way the
adapter puts it on the isolated interpreter's ``PYTHONPATH``.
"""

import json
import sys
from pathlib import Path

import pytest

_SDK_SRC = (
    Path(__file__).resolve().parents[5] / "packaging" / "agentictrading" / "src"
)
if str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

from agentictrading.integrations import _tradingagents_core  # noqa: E402

from dashboard.backend.infrastructure.tradingagents import bridge  # noqa: E402
from dashboard.backend.infrastructure.tradingagents.bridge import (  # noqa: E402
    MANAGED_DEEP_THINK_MODEL,
    MANAGED_LLM_PROVIDER,
    MANAGED_QUICK_THINK_MODEL,
    SUPPORTED_ANALYSTS,
    BridgeInputError,
    _managed_config_from_payload,
    _selected_analysts,
    _single_analysis_date,
)

# Hand-derived with hashlib, not with the code under test.
_TRANSCRIPT = "Analysts agree. FINAL TRANSACTION PROPOSAL: **BUY**"
_TRANSCRIPT_SHA256 = "afd306ffc3047d75fe1537dccedae4984c625fc660a246cc8d0c9709ad432b83"

_ANALYSIS_DATE = "2026-03-13"


def _payload(**overrides):
    payload = {
        "schema": "tradingagents-bridge-v1",
        "symbol": "AAPL",
        "analysis_dates": [_ANALYSIS_DATE],
        "selected_analysts": list(SUPPORTED_ANALYSTS),
        "config": {
            "llm_provider": MANAGED_LLM_PROVIDER,
            "deep_think_llm": MANAGED_DEEP_THINK_MODEL,
            "quick_think_llm": MANAGED_QUICK_THINK_MODEL,
        },
    }
    payload.update(overrides)
    return payload


class _FakeGraph:
    """Stands in for ``TradingAgentsGraph`` — the billable, networked part."""

    def __init__(self, *, transcript, propagate_error, calls):
        self._transcript = transcript
        self._propagate_error = propagate_error
        self._calls = calls

    def propagate(self, symbol, analysis_date):
        self._calls.append((symbol, analysis_date))
        if self._propagate_error is not None:
            raise self._propagate_error
        return {"final_trade_decision": self._transcript}, {}


def _install_fake_upstream(
    monkeypatch,
    *,
    rating="buy",
    transcript=_TRANSCRIPT,
    version="0.3.1",
    factory_error=None,
    propagate_error=None,
):
    """Replace only the upstream loader; the generator itself stays real."""
    seen = {"config": None, "selected_analysts": None, "propagate_calls": []}

    def graph_factory(*, selected_analysts, config):
        seen["selected_analysts"] = tuple(selected_analysts)
        seen["config"] = dict(config)
        if factory_error is not None:
            raise factory_error
        return _FakeGraph(
            transcript=transcript,
            propagate_error=propagate_error,
            calls=seen["propagate_calls"],
        )

    def rating_parser(raw, default=""):
        if rating is None:
            # The real algorithm, vendored and held to measured upstream
            # behaviour (see test_rating_contradiction.py). Passing a fixed
            # rating back instead -- which every case below did -- meant the
            # parser that decides every mapped order's direction was never
            # executed anywhere in this repository's suite.
            from dashboard.backend.tests.infrastructure.tradingagents.test_rating_contradiction import (  # noqa: E501
                upstream_parse_rating,
            )

            return upstream_parse_rating(raw, default=default)
        return rating

    monkeypatch.setattr(
        _tradingagents_core,
        "_load_tradingagents_components",
        lambda: (graph_factory, rating_parser, lambda: version),
    )
    return seen


def _run_bridge(monkeypatch, tmp_path, payload):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["bridge.py", "--input", str(input_path), "--output", str(output_path)],
    )
    bridge.main()
    return json.loads(output_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Fail-closed input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            "llm_provider": "openai",
            "deep_think_llm": MANAGED_DEEP_THINK_MODEL,
            "quick_think_llm": MANAGED_QUICK_THINK_MODEL,
        },
        {
            "llm_provider": MANAGED_LLM_PROVIDER,
            "deep_think_llm": "anthropic/claude-opus-5",
            "quick_think_llm": MANAGED_QUICK_THINK_MODEL,
        },
        {
            "llm_provider": MANAGED_LLM_PROVIDER,
            "deep_think_llm": MANAGED_DEEP_THINK_MODEL,
            "quick_think_llm": "openai/gpt-5.5",
        },
    ],
)
def test_bridge_refuses_any_llm_config_that_is_not_the_platform_pin(config):
    """TradingAgents accepts *any* model name on OpenRouter and silently
    defaults to ``llm_provider='openai'``, so this check is the only thing
    standing between a crafted payload and spend on an unpinned account."""
    with pytest.raises(BridgeInputError, match="platform-managed"):
        _managed_config_from_payload({"config": config})


def test_bridge_forwards_only_allowlisted_tradingagents_knobs():
    resolved = _managed_config_from_payload(
        {
            "config": {
                "llm_provider": MANAGED_LLM_PROVIDER,
                "deep_think_llm": MANAGED_DEEP_THINK_MODEL,
                "quick_think_llm": MANAGED_QUICK_THINK_MODEL,
                "max_debate_rounds": 2,
                "results_dir": "/etc",
                "OPENAI_API_KEY": "sk-smuggled",
            }
        }
    )

    assert resolved["max_debate_rounds"] == 2
    assert resolved["backend_url"] is None
    assert "results_dir" not in resolved
    assert "OPENAI_API_KEY" not in resolved


@pytest.mark.parametrize(
    "dates",
    [[], [_ANALYSIS_DATE, "2026-03-16"], "2026-03-13"],
)
def test_bridge_refuses_anything_but_exactly_one_analysis_date(dates):
    """One subprocess call is one billed decision. A payload carrying N dates
    would quietly turn a single backtest step into N multi-agent runs."""
    with pytest.raises(BridgeInputError, match="exactly one analysis date"):
        _single_analysis_date({"analysis_dates": dates})


@pytest.mark.parametrize(
    "analysts",
    [[], ["market", "risk_manager"], ["market", "market"], "market"],
)
def test_bridge_refuses_analysts_outside_the_supported_four(analysts):
    with pytest.raises(BridgeInputError, match="analyst"):
        _selected_analysts({"selected_analysts": analysts})


# --------------------------------------------------------------------------
# Decision path
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parsed_rating,expected_rating,expected_action",
    [
        ("buy", "Buy", "BUY"),
        ("overweight", "Overweight", "BUY"),
        ("hold", "Hold", "HOLD"),
        ("underweight", "Underweight", "SELL"),
        ("sell", "Sell", "SELL"),
    ],
)
def test_bridge_maps_every_tradingagents_rating_tier_to_its_atl_action(
    monkeypatch, tmp_path, parsed_rating, expected_rating, expected_action
):
    _install_fake_upstream(monkeypatch, rating=parsed_rating)

    output = _run_bridge(monkeypatch, tmp_path, _payload())

    assert output["decision"]["rating"] == expected_rating
    assert output["decision"]["atl_action"] == expected_action
    assert output["decision"]["analysis_date"] == _ANALYSIS_DATE
    assert output["decision"]["status"] == "valid"


def test_bridge_analyses_the_requested_symbol_on_the_requested_date(
    monkeypatch, tmp_path
):
    seen = _install_fake_upstream(monkeypatch)

    _run_bridge(monkeypatch, tmp_path, _payload(symbol="msft"))

    assert seen["propagate_calls"] == [("MSFT", _ANALYSIS_DATE)]


def test_bridge_hands_tradingagents_the_pinned_provider_and_models(
    monkeypatch, tmp_path
):
    seen = _install_fake_upstream(monkeypatch)

    _run_bridge(monkeypatch, tmp_path, _payload())

    assert seen["config"]["llm_provider"] == "openrouter"
    assert seen["config"]["deep_think_llm"] == MANAGED_DEEP_THINK_MODEL
    assert seen["config"]["quick_think_llm"] == MANAGED_QUICK_THINK_MODEL
    assert seen["selected_analysts"] == SUPPORTED_ANALYSTS


def test_bridge_reports_the_transcript_digest_without_the_transcript(
    monkeypatch, tmp_path
):
    """The debate transcript is unbounded, upstream-controlled text that would
    otherwise land in ATL run metadata. Only the digest and a capped excerpt
    may cross the boundary."""
    long_transcript = _TRANSCRIPT + " " + ("deliberation " * 200)
    _install_fake_upstream(monkeypatch, transcript=long_transcript)

    output = _run_bridge(monkeypatch, tmp_path, _payload())

    serialized = json.dumps(output)
    assert "raw_final_trade_decision" not in serialized
    assert long_transcript not in serialized
    assert len(output["decision"]["raw_excerpt"]) <= 500
    assert output["decision"]["raw_excerpt"] == long_transcript[:500]


def test_bridge_digest_identifies_the_transcript_it_summarized(monkeypatch, tmp_path):
    _install_fake_upstream(monkeypatch, transcript=_TRANSCRIPT)

    output = _run_bridge(monkeypatch, tmp_path, _payload())

    assert output["decision"]["raw_sha256"] == _TRANSCRIPT_SHA256


def test_bridge_manifest_records_the_upstream_version_it_ran(monkeypatch, tmp_path):
    _install_fake_upstream(monkeypatch, version="0.3.4")

    output = _run_bridge(monkeypatch, tmp_path, _payload())

    assert output["manifest"]["tradingagents_version"] == "0.3.4"
    assert output["manifest"]["symbol"] == "AAPL"
    assert output["schema_version"] == "tradingagents-atl-v1"


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_bridge_refuses_a_tradingagents_version_outside_the_supported_line(
    monkeypatch, tmp_path, capsys
):
    _install_fake_upstream(monkeypatch, version="0.2.9")

    with pytest.raises(SystemExit) as exc:
        _run_bridge(monkeypatch, tmp_path, _payload())

    assert exc.value.code == 1
    assert "0.3" in capsys.readouterr().err


def test_bridge_failure_reaches_stderr_without_the_credential_or_a_traceback(
    monkeypatch, tmp_path, capsys
):
    """``generate()`` re-raises initialization failures with ``from exc``, so a
    propagating exception would print the original traceback — and any key in
    it — to the stderr the adapter captures and persists."""
    _install_fake_upstream(
        monkeypatch,
        factory_error=RuntimeError("openrouter refused: sk-or-v1-topsecret"),
    )

    with pytest.raises(SystemExit) as exc:
        _run_bridge(monkeypatch, tmp_path, _payload())

    stderr = capsys.readouterr().err
    assert exc.value.code == 1
    assert "sk-or-v1-topsecret" not in stderr
    assert "Traceback" not in stderr
    assert "failed to initialize TradingAgents" in stderr


def test_bridge_writes_no_output_file_when_the_analysis_fails(
    monkeypatch, tmp_path
):
    """A stale or partial output file would let the adapter parse a previous
    step's decision as this step's."""
    _install_fake_upstream(
        monkeypatch, propagate_error=RuntimeError("upstream unavailable")
    )
    output_path = tmp_path / "output.json"

    with pytest.raises(SystemExit):
        _run_bridge(monkeypatch, tmp_path, _payload())

    assert not output_path.exists()


@pytest.fixture
def restore_dotenv():
    """``_disable_dotenv_loading`` patches a process-global module."""
    import dotenv
    from dotenv import main as dotenv_main

    public_loader = dotenv.load_dotenv
    module_loader = dotenv_main.load_dotenv
    try:
        yield dotenv, dotenv_main
    finally:
        dotenv.load_dotenv = public_loader
        dotenv_main.load_dotenv = module_loader


def test_bridge_blocks_dotenv_so_a_user_env_file_is_never_read(restore_dotenv):
    dotenv, dotenv_main = restore_dotenv

    bridge._disable_dotenv_loading()

    assert dotenv.load_dotenv() is False
    assert dotenv_main.load_dotenv() is False


def test_bridge_disables_dotenv_before_running_the_analysis(
    monkeypatch, tmp_path, restore_dotenv
):
    """The block is worthless if the decision path forgets to apply it: a
    ``.env`` in the working directory would then reach TradingAgents.

    The probe names a real file: a bare ``load_dotenv()`` returns False
    whenever it finds nothing, so it would pass with the guard removed.
    """
    dotenv, _dotenv_main = restore_dotenv
    env_file = tmp_path / "canary.env"
    env_file.write_text("ATL_DOTENV_CANARY=leaked\n", encoding="utf-8")
    _install_fake_upstream(monkeypatch)

    _run_bridge(monkeypatch, tmp_path, _payload())

    assert dotenv.load_dotenv(str(env_file)) is False


# ---------------------------------------------------------------------------
# Managed gateways (CommonStack)
# ---------------------------------------------------------------------------

def test_commonstack_is_a_managed_gateway_with_a_pinned_endpoint():
    """Prod pays with CommonStack, and upstream has no provider id for it, so
    ATL's gateway rides the generic OpenAI-compatible endpoint. That endpoint
    is the one config value a crafted payload would most want to move, so the
    bridge pins it from its own table rather than echoing the payload."""
    gateway = bridge.MANAGED_GATEWAYS["commonstack"]
    resolved = _managed_config_from_payload(
        {
            "config": {
                "llm_provider": gateway["llm_provider"],
                "deep_think_llm": gateway["deep_think_llm"],
                "quick_think_llm": gateway["quick_think_llm"],
                "backend_url": "https://attacker.example/v1",
            }
        }
    )
    assert resolved["llm_provider"] == "openai_compatible"
    assert resolved["backend_url"] == "https://api.commonstack.ai/v1"


def test_openrouter_gateway_still_pins_a_null_backend_url():
    """The openrouter provider carries its own base URL; pinning None keeps a
    payload from redirecting it."""
    resolved = _managed_config_from_payload(
        {
            "config": {
                "llm_provider": MANAGED_LLM_PROVIDER,
                "deep_think_llm": MANAGED_DEEP_THINK_MODEL,
                "quick_think_llm": MANAGED_QUICK_THINK_MODEL,
                "backend_url": "https://attacker.example/v1",
            }
        }
    )
    assert resolved["backend_url"] is None


def test_a_model_outside_every_managed_gateway_is_refused():
    """Adding a gateway must not widen the bridge into "any OpenAI-compatible
    model"; the pair still has to match one table entry exactly."""
    with pytest.raises(BridgeInputError, match="platform-managed"):
        _managed_config_from_payload(
            {
                "config": {
                    "llm_provider": "openai_compatible",
                    "deep_think_llm": "anthropic/claude-opus-5-5",
                    "quick_think_llm": "anthropic/claude-opus-5-5",
                }
            }
        )


def test_gateway_models_may_not_be_mixed_across_entries():
    """A provider from one entry with a model from another is not a gateway
    this platform operates."""
    with pytest.raises(BridgeInputError, match="platform-managed"):
        _managed_config_from_payload(
            {
                "config": {
                    "llm_provider": "openrouter",
                    "deep_think_llm": bridge.MANAGED_GATEWAYS["commonstack"][
                        "deep_think_llm"
                    ],
                    "quick_think_llm": bridge.MANAGED_GATEWAYS["commonstack"][
                        "quick_think_llm"
                    ],
                }
            }
        )


def test_the_measured_gateway_baselines_are_pinned():
    """Both measured baselines (290.5s and 414.9s) ran on **CommonStack with
    anthropic/claude-haiku-4-5**. The openrouter entry has NO measured
    baseline -- it only ever reached HTTP 402 (insufficient credits) and never
    produced a decision. Editing either pin in place would silently make the
    CommonStack runs incomparable, so a model switch has to ADD an entry."""
    assert bridge.MANAGED_GATEWAYS["openrouter"] == {
        "llm_provider": "openrouter",
        "deep_think_llm": "nvidia/nemotron-3-nano-30b-a3b",
        "quick_think_llm": "nvidia/nemotron-3-nano-30b-a3b",
        "backend_url": None,
    }
    assert bridge.MANAGED_GATEWAYS["commonstack"] == {
        "llm_provider": "openai_compatible",
        "deep_think_llm": "anthropic/claude-haiku-4-5",
        "quick_think_llm": "anthropic/claude-haiku-4-5",
        "backend_url": "https://api.commonstack.ai/v1",
    }


def test_a_payload_cannot_redirect_where_upstream_writes():
    """The deployment may move the trajectory directory; a payload may not.
    Forwarding TRADINGAGENTS_RESULTS_DIR must not turn into a path the caller
    controls -- that would let a crafted config write outside its sandbox."""
    gateway = bridge.MANAGED_GATEWAYS["commonstack"]
    resolved = _managed_config_from_payload(
        {
            "config": {
                "llm_provider": gateway["llm_provider"],
                "deep_think_llm": gateway["deep_think_llm"],
                "quick_think_llm": gateway["quick_think_llm"],
                "results_dir": "/etc",
                "data_cache_dir": "/etc",
                "memory_log_path": "/etc/passwd",
            }
        }
    )
    assert "results_dir" not in resolved
    assert "data_cache_dir" not in resolved
    assert "memory_log_path" not in resolved


# ---------------------------------------------------------------------------
# Exact upstream revision
# ---------------------------------------------------------------------------

def test_the_exact_upstream_commit_is_recovered_from_the_install_url():
    """``importlib.metadata.version`` reports the tag (0.3.1) and cannot tell
    two builds apart -- requirements pins a commit six commits past that tag,
    so two different revisions produce identical records. pip/uv write the real
    source into direct_url.json, which is the only in-process witness."""
    url = (
        "https://github.com/TauricResearch/TradingAgents/archive/"
        "a33fd4c0f134485a43553a2c23a63cb14adbd88f.tar.gz"
    )
    assert bridge._commit_from_url(url) == (
        "a33fd4c0f134485a43553a2c23a63cb14adbd88f"
    )


def test_a_url_without_a_commit_yields_no_false_revision():
    """A PyPI or branch install has no commit to report; inventing one would be
    worse than recording nothing."""
    assert bridge._commit_from_url("https://pypi.org/simple/tradingagents") == ""
    assert bridge._commit_from_url("") == ""
    assert bridge._commit_from_url(None) == ""


def test_a_short_or_malformed_sha_is_not_accepted():
    """Only a full 40-hex revision identifies a build unambiguously."""
    assert bridge._commit_from_url("http://x/archive/a33fd4c0.tar.gz") == ""

# --------------------------------------------------------------------------
# The real parsing algorithm, end to end through the bridge
# --------------------------------------------------------------------------


_CONTRADICTORY_TRANSCRIPT = (
    "After weighing both sides, the bear case recommends Sell.\n"
    "Final Rating: Strong Buy"
)


def test_the_real_parser_runs_and_a_clean_transcript_maps_as_expected(
    monkeypatch, tmp_path
):
    """`rating=None` puts the real algorithm on the path, not a stub."""
    _install_fake_upstream(
        monkeypatch,
        rating=None,
        transcript="Everything considered.\nFinal Rating: Overweight",
    )
    output = _run_bridge(monkeypatch, tmp_path, _payload())

    assert output["decision"]["rating"] == "Overweight"
    assert output["decision"]["atl_action"] == "BUY"


def test_a_contradictory_transcript_is_refused_rather_than_sold(
    monkeypatch, tmp_path
):
    """The direction-reversing case, driven through the whole bridge.

    The transcript states "Strong Buy" and upstream's parser answers "Sell"
    from the losing argument's sentence -- measured, not hypothesised (see
    test_rating_contradiction.py). Without the guard this leaves the bridge as
    a fully valid SELL record: rating "Sell", action "SELL", digest matching,
    cross-check satisfied, nothing downstream able to tell.
    """
    _install_fake_upstream(
        monkeypatch,
        rating=None,
        transcript=_CONTRADICTORY_TRANSCRIPT,
    )
    with pytest.raises(SystemExit):
        _run_bridge(monkeypatch, tmp_path, _payload())


def test_the_contradiction_would_otherwise_have_become_a_sell(
    monkeypatch, tmp_path
):
    """Shows what the guard prevents, so the case above is not circular.

    With the guard disabled the same transcript produces a valid SELL. This is
    the assertion that makes the refusal meaningful rather than a tautology
    about an exception being raised.
    """
    monkeypatch.setattr(bridge, "unsupported_rating_label", lambda _raw: "")
    _install_fake_upstream(
        monkeypatch,
        rating=None,
        transcript=_CONTRADICTORY_TRANSCRIPT,
    )
    output = _run_bridge(monkeypatch, tmp_path, _payload())

    assert output["decision"]["rating"] == "Sell"
    assert output["decision"]["atl_action"] == "SELL"

