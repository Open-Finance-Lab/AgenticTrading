"""The frozen news snapshot (`demo: true`, api/routers/news.py) under node.

Both consumers of /api/news/signals run their real render code against the
committed snapshot: home-page.js (the Home Market News / Signals tabs) and
home-news-signals.js (the #homeNewsSignals section). Each must label the frozen
feed "Demo news" rather than present it as live, and show fixed dates instead
of relative times that keep growing on a paused feed.
"""

import json
import shutil
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import FRONTEND, _match_brace, fn_body, run_node

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

_HOME_JS = (FRONTEND / "home-page.js").read_text(encoding="utf-8")
_SIGNALS_JS = (FRONTEND / "home-news-signals.js").read_text(encoding="utf-8")
_SNAPSHOT = json.loads(
    (Path(__file__).resolve().parents[2] / "config" / "news_demo_snapshot.json").read_text(encoding="utf-8")
)


def _home_mock_news() -> str:
    start = _HOME_JS.index("const HOME_MOCK_NEWS = {")
    return _HOME_JS[start : _match_brace(_HOME_JS, _HOME_JS.index("{", start)) + 1] + ";"


_FAKE_DOM = """
const els = {};
function el(id) {
  if (!els[id]) els[id] = { id, hidden: false, textContent: '', innerHTML: '' };
  return els[id];
}
const document = { getElementById: el };
const window = {
  formatMarketEventRelativeTime: (ms, now) => 'REL',
};
"""


def _home_harness(payload: dict) -> str:
    fns = [fn_body(f"function {name}(", _HOME_JS) for name in (
        "homeEscape", "homeSafeUrl", "homeSentimentClass", "homeRelTime",
        "homeFeedTime", "setHomeMarketDemoBadge", "renderHomeMarketNews",
        "renderHomeMarketSignals",
    )]
    return "\n".join([_FAKE_DOM, *fns, _home_mock_news(), f"""
const payload = {json.dumps(payload)};
renderHomeMarketNews(payload);
const news = {{ status: {{ ...el('homeMarketStatus') }}, list: el('homeModuleNewsList').innerHTML,
                badge: el('homeMarketDemoBadge').hidden }};
renderHomeMarketSignals(payload);
console.log(JSON.stringify({{ news, signals: {{ status: {{ ...el('homeMarketSignalsStatus') }},
  list: el('homeModuleSignalsList').innerHTML }} }}));
"""])


def _live(payload: dict) -> dict:
    out = {k: v for k, v in payload.items() if k != "demo"}
    out["staleness_hours"] = 1.5
    return out


@pytest.fixture
def demo_payload():
    # What the loader serves: the file plus the two markers it owns.
    return {**_SNAPSHOT, "demo": True, "staleness_hours": None}


def test_home_news_labels_the_snapshot_demo_news(demo_payload):
    out = run_node(_home_harness(demo_payload))
    assert out["news"]["status"]["hidden"] is False
    assert out["news"]["status"]["textContent"] == "Demo news"
    # Real stories, not HOME_MOCK_NEWS: the mock "Demo data" badge stays hidden.
    assert out["news"]["badge"] is True
    assert _SNAPSHOT["feed"][0]["headline"].split()[0] in out["news"]["list"]
    assert out["signals"]["status"]["textContent"] == f"Demo news · {len(_SNAPSHOT['signals'])} signals"


def test_home_news_shows_fixed_dates_not_relative_times(demo_payload):
    out = run_node(_home_harness(demo_payload))
    assert "REL" not in out["news"]["list"]
    assert "Oct " in out["news"]["list"]


def test_home_live_feed_keeps_relative_times_and_updated_line(demo_payload):
    out = run_node(_home_harness(_live(demo_payload)))
    assert out["news"]["status"]["textContent"] == "Updated 1.5h ago"
    assert "REL" in out["news"]["list"]
    assert out["signals"]["status"]["textContent"] == f"{len(_SNAPSHOT['signals'])} signals"


def test_home_signals_read_the_live_sentiment_score(demo_payload):
    """Live signals carry `sentiment_score`; reading the mock's `score` rendered
    every one as 0.00."""
    sym, sig = next((s, v) for s, v in _SNAPSHOT["signals"].items() if v["sentiment_score"])
    out = run_node(_home_harness(demo_payload))
    assert f"{sym} · score {sig['sentiment_score']:.2f}" in out["signals"]["list"]


def test_home_mock_still_reads_its_score_field():
    out = run_node(_home_harness({"status": "unavailable"}))
    assert "AAPL · score 0.72" in out["signals"]["list"]
    assert out["news"]["badge"] is False  # mock data: "Demo data" badge shown


def _signals_harness(payload: dict) -> str:
    return "\n".join([_FAKE_DOM, """
function escapeHtml(v) {
  return String(v == null ? '' : v).replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
const API_BASE = '';
let served;
const API = { get: async () => served };
global.setInterval = () => 1;
""", f"served = {json.dumps(payload)};", _SIGNALS_JS, """
window.newsSignalsPanel.onShow();
setTimeout(() => console.log(JSON.stringify({
  updated: el('nnsUpdated').textContent, badge: el('nnsStatusBadge').hidden,
  feed: el('nnsFeedList').innerHTML,
})), 0);
"""])


def test_signals_section_labels_the_snapshot_demo_news(demo_payload):
    out = run_node(_signals_harness(demo_payload))
    assert out["updated"] == "Demo news"
    assert out["badge"] is True
    assert "REL" not in out["feed"] and "Oct " in out["feed"]


def test_signals_section_live_feed_unchanged(demo_payload):
    out = run_node(_signals_harness(_live(demo_payload)))
    assert out["updated"] == "Updated 1.5h ago"
    assert "REL" in out["feed"]
