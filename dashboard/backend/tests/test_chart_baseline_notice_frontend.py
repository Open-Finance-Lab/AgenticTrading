"""Guards for the degraded-chart notice on the Playground backtest chart.

/app has no build step and no JS test toolchain, so these contracts are asserted
against the shipped source as text (see ``_frontend_source``).

What is being protected: when Yahoo is unreachable the chart loses its index
benchmarks, and *fewer lines* is not a message. Without the notice a degraded
chart is byte-for-byte the same experience as an agent that genuinely has no
benchmark — the absent-vs-broken collapse CLAUDE.md calls out.
"""

from ._frontend_source import APP_HTML, css_blocks, fn_body


def test_notice_element_ships_hidden():
    assert 'id="chartBaselineNotice"' in APP_HTML
    element = APP_HTML[APP_HTML.index('id="chartBaselineNotice"') :]
    element = element[: element.index("</p>")]
    # Hidden by default: the healthy path must not have to remember to hide it.
    assert "hidden" in element


def test_initialize_charts_reads_the_flag_and_toggles_the_notice():
    body = fn_body("function initializeCharts()")
    assert "chartBaselineNotice" in body
    assert "index_baselines_ok" in body
    assert "renderPerformanceComparison" in body
    # `!== false`, not falsy: a payload from an older backend omits the key
    # entirely, and `!payload.index_baselines_ok` would show the warning on
    # every healthy chart served during a frontend/backend deploy skew.
    assert "index_baselines_ok !== false" in body


def test_notice_has_an_explicit_hidden_rule():
    """Without it the class's own `display` beats the bare `hidden` attribute."""
    blocks = css_blocks(".chart-baseline-notice[hidden]")
    assert blocks, ".chart-baseline-notice[hidden] rule is missing from styles.css"
    assert any("display: none" in block for block in blocks)
