"""The Live roster, checked against what its gateway serves.

Live-board calls go through CommonStack on ``COMMONSTACK_API_KEY``, outside the
Credits layer, so nothing at request time checks that a model id exists. An id
CommonStack does not serve fails every call, every night: Nemotron's OpenRouter
slug did exactly that until 2026-10-14. These cases pin the roster to the
gateway's own listing instead.
"""

import re
from pathlib import Path

import pytest

import dashboard.backend.domain.leaderboard.live as live
import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.infrastructure.llm import pricing
from dashboard.backend.infrastructure.llm.reasoning_controls import is_reasoning_off

# CommonStack GET /v1/models with the prod key on 2026-10-14: 86 models, none
# of them NVIDIA/Nemotron. Model id -> listed (input, output) USD per million
# tokens, copied for the ids this roster uses. Re-list before adding an id:
# the code's allowlists and pricing.py are not the listing.
COMMONSTACK_SERVED = {
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    "qwen/qwen3.7-plus": (0.40, 1.60),
    "anthropic/claude-haiku-4-5": (1.0, 5.0),
}
# After every live_from in leaderboard.json, so no entry is gated out.
_EVERY_JOIN_OPEN = "2099-12-01"


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(live, "_warned_live_config", set())


def _live_entries():
    """Every roster entry as the Live board runs it (overrides applied)."""
    return live.live_llm_entries(
        lb_service.load_leaderboard_config(), month_start=_EVERY_JOIN_OPEN
    )


def test_every_live_model_id_names_a_config_entry():
    """A typo in LIVE_MODEL_IDS silently drops that model from the board."""
    assert [e["id"] for e in _live_entries()] == list(live.LIVE_MODEL_IDS)


def test_every_live_entry_calls_a_model_commonstack_serves():
    for entry in _live_entries():
        assert entry["integration"] == "commonstack", entry["id"]
        assert entry["model_id"] in COMMONSTACK_SERVED, (entry["id"], entry["model_id"])


def test_live_entries_that_turn_thinking_off_pin_temperature_zero():
    for entry in _live_entries():
        if is_reasoning_off(entry.get("reasoning_effort")):
            assert entry.get("temperature") == 0, entry["id"]


def test_every_live_model_is_priced_at_its_listed_rate():
    """``est_cost_usd`` on every live row is priced from pricing.py. An id the
    table does not list is priced at ``_DEFAULT_PRICING``, which is a guess."""
    for entry in _live_entries():
        listed = COMMONSTACK_SERVED.get(entry["model_id"])
        assert listed is not None, (entry["id"], entry["model_id"])
        assert pricing.listed_price_for_model(entry["model_id"]) == pytest.approx(
            listed
        ), entry["id"]


def test_the_committed_live_config_raises_no_warning(capsys):
    """A malformed live_from or live_overrides is ignored with a WARNING at
    runtime, because the public GET must not fail on config. Committed, it is
    a bug: the entry would trade at once, or on its shared values."""
    _live_entries()
    assert "WARNING: live." not in capsys.readouterr().out


_REPO = Path(__file__).resolve().parents[5]
_ROSTER_MENTIONS = (
    _REPO / ".github" / "workflows" / "live-leaderboard.yml",
    _REPO / "dashboard" / "scripts" / "refresh_live_leaderboard.py",
)


@pytest.mark.parametrize("path", _ROSTER_MENTIONS, ids=lambda p: p.name)
def test_roster_mentions_name_the_constant_not_the_models(path):
    """The workflow comment and the CLI help both listed the roster ("2 models
    (DeepSeek V4 Pro, Nemotron 3 Nano)") and went stale on its first change."""
    text = path.read_text(encoding="utf-8")
    assert "LIVE_MODEL_IDS" in text
    assert "Nemotron" not in text
    assert not re.search(r"\b\d+ models\b", text)
