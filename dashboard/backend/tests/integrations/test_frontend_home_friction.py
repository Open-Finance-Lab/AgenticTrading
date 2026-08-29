"""Guard the backtest-result honesty friction cleanup.

The frontend is vanilla JS with no test harness, so these are source-level guards
(read the files as text) that run in CI and lock one fix:

  The backtest result panel no longer renders a fabricated "Performance
  Drivers" card (three hardcoded lines shown identically every run) or the two
  dead buttons that had no JS behind them. Real metrics are untouched.
"""

from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[3] / "frontend"
_APP_HTML = _FRONTEND / "app.html"


def test_backtest_result_has_no_fabricated_performance_drivers():
    html = _APP_HTML.read_text(encoding="utf-8")
    # The fabricated, always-identical driver lines are gone (H8 spirit: no fake data).
    assert "Performance Drivers" not in html
    assert "driver-item" not in html
    assert "Lower slippage improved execution quality" not in html
    # The two dead buttons (no JS behind them) are gone.
    assert "view-details-btn" not in html
    assert "view-more-btn" not in html


def test_real_result_comparison_is_preserved():
    html = _APP_HTML.read_text(encoding="utf-8")
    # The genuine, data-driven result surface must survive the cleanup.
    assert 'id="performanceComparison"' in html
    assert 'id="performanceComparisonBody"' in html
    assert 'src="js/backtest-comparison.js?v=1"' in html
