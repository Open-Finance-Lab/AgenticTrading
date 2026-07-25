"""Guard the A+B navigation/honesty friction cleanup.

The frontend is vanilla JS with no test harness, so these are source-level guards
(read the files as text) that run in CI and lock two fixes:

  A) The in-app Home no longer re-shows the marketing landing hero. Home opens
     straight on the dashboard, so "Get Started" on the landing lands the user on
     a real destination instead of a second identical hero + "Get Started".
  B) The backtest result panel no longer renders a fabricated "Performance
     Drivers" card (three hardcoded lines shown identically every run) or the two
     dead buttons that had no JS behind them. Real metrics are untouched.
"""

from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[3] / "frontend"
_APP_HTML = _FRONTEND / "app.html"
_HOME_JS = _FRONTEND / "home-page.js"


def test_app_home_does_not_duplicate_the_landing_hero():
    html = _APP_HTML.read_text(encoding="utf-8")
    # The redundant in-app hero screen and its "Get Started" CTA are gone.
    assert "homeScreenLanding" not in html
    assert "homeGetStartedBtn" not in html
    # The real dashboard screen (and the pager it lives in) still exist.
    assert "homeScreenDashboard" in html
    assert "homePagerTrack" in html


def test_home_page_js_drops_dead_hero_wiring():
    js = _HOME_JS.read_text(encoding="utf-8")
    # The hero-only demo animation and Get-Started wiring were removed with the hero.
    assert "initLandingPlaygroundChat" not in js
    assert "initHomeGetStarted" not in js
    assert "homePlaygroundChat" not in js
    # The dashboard wiring the Home still depends on stays put.
    assert "function initHomeModules" in js
    assert "refreshHomeModulesWhenReady" in js


def test_backtest_result_has_no_fabricated_performance_drivers():
    html = _APP_HTML.read_text(encoding="utf-8")
    # The fabricated, always-identical driver lines are gone (H8 spirit: no fake data).
    assert "Performance Drivers" not in html
    assert "driver-item" not in html
    assert "Lower slippage improved execution quality" not in html
    # The two dead buttons (no JS behind them) are gone.
    assert "view-details-btn" not in html
    assert "view-more-btn" not in html


def test_real_result_metrics_are_preserved():
    html = _APP_HTML.read_text(encoding="utf-8")
    # The genuine, data-driven metrics must survive the cleanup.
    for metric in ("max-drawdown", "sharpe"):
        assert f'data-metric="{metric}"' in html
