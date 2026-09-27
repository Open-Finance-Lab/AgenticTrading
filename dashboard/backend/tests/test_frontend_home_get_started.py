"""Get Started on Home plays a stored model curve instead of opening My Agents."""

from pathlib import Path
import json
import re

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_HOME_JS = (_FRONTEND / "home-page.js").read_text(encoding="utf-8")
_DEMO_JSON = _FRONTEND / "data" / "home-demo-run.json"


def _strip_js_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", source, flags=re.MULTILINE)


def test_get_started_button_exists():
    assert 'id="homeGetStartedBtn"' in _APP_HTML
    assert "Get Started" in _APP_HTML


def test_get_started_plays_stored_curve_not_my_agents():
    body_start = _HOME_JS.index("function initHomeGetStarted")
    body = _HOME_JS[body_start : _HOME_JS.index("function homeDemoMoney")]
    source = _strip_js_comments(body)
    assert "startHomeDemoRun()" in source
    assert "playgroundTab: 'agents'" not in source
    assert "navigateToPage('playground'" not in source


def test_home_demo_covers_ranking_with_arrows():
    assert 'id="homeDemoRunPanel"' in _APP_HTML
    assert 'id="homeDemoRunChart"' in _APP_HTML
    assert 'id="homeBoardPrev"' in _APP_HTML
    assert 'id="homeBoardNext"' in _APP_HTML
    assert 'data-home-board-slide="rank"' in _APP_HTML
    assert "HOME_DEMO_RETURN_MS = 3000" in _HOME_JS
    assert "setHomeBoardSlide('demo'" in _HOME_JS
    assert "scheduleHomeDemoAutoReturn" in _HOME_JS
    assert "homeModuleRanking.hidden" not in _HOME_JS
    assert "board.hidden = true" not in _HOME_JS
    assert _DEMO_JSON.is_file()
    payload = json.loads(_DEMO_JSON.read_text(encoding="utf-8"))
    assert payload["model"] == "GPT-5.5"
    assert payload["points"]
    assert payload["points"][0]["equity"] == 10000.0
    assert payload["final_equity"] > payload["initial_equity"]


def test_home_demo_fetches_allowlisted_json():
    source = _strip_js_comments(_HOME_JS)
    assert "/data/home-demo-run.json" in source


def test_home_demo_json_is_public_without_session():
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    with TestClient(app) as client:
        response = client.get("/data/home-demo-run.json")
    assert response.status_code == 200
    assert response.json()["model"] == "GPT-5.5"


def test_data_route_serves_only_the_allowlisted_file():
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    with TestClient(app) as client:
        for path in (
            "/data/app.html",
            "/data/..%2Fapp.html",
            "/data/..%2F..%2Fconfig%2Fleaderboard.json",
        ):
            assert client.get(path).status_code == 404, path
