"""Guards for the Portfolio Management subtab (the recorded weekly-rebalancing demo).

The tab plays a fixed recording inside a Shadow DOM root (js/portfolio-management.js),
so these guards pin the three seams where it could leak into the rest of /app:
the subtab wiring (My Agents must stay the default), what loading the two scripts
registers on other tabs (nothing), and which keys the demo is allowed to take.
"""

import re
import shutil

import pytest

from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, fn_body, run_node, strip_comments

requires_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")


# --- Task 1: ATL wiring ---------------------------------------------------------


def test_subtab_sits_before_my_agents_and_is_not_active():
    row = APP_HTML[APP_HTML.index('<div class="subtab-bar playground-subtabs"'):]  # not the boot CSS
    row = row[: row.index("</div>")]
    tabs = re.findall(r'<button class="([^"]*)" data-playground-tab="([^"]+)"[^>]*>([^<]+)</button>', row)
    assert [(t[1], t[2]) for t in tabs[:2]] == [
        ("portfolio-management", "Portfolio Management"), ("agents", "My Agents")]
    assert "active" not in tabs[0][0].split() and "active" in tabs[1][0].split()


def test_view_map_routes_the_slug_and_reserves_no_pm_keys():
    assert "'portfolio-management': { page: 'playground', playgroundTab: 'portfolio-management' }," in APP_HTML
    nav_map = APP_HTML[APP_HTML.index("var NAV_VIEW_MAP"): APP_HTML.index("window.NAV_VIEW_MAP")]
    assert not re.search(r"^\s*'pm-", nav_map, re.M)  # the step link (#pm-step-N) must never route


@requires_node
def test_view_param_round_trips_and_keeps_the_agents_default():
    fn = fn_body("function viewParamForNavState(")
    out = run_node(fn + "; console.log(JSON.stringify(["
        "viewParamForNavState({page:'playground', playgroundTab:'portfolio-management'}),"
        "viewParamForNavState({page:'playground'}),"
        "viewParamForNavState({page:'playground', playgroundTab:'agents'})]))")
    assert out == ["portfolio-management", "agents", "agents"]


def test_default_tab_is_still_my_agents():
    assert 'let playgroundTab = "agents";' in APP_JS
    assert "nav.playgroundTab || 'agents'" in APP_HTML
    assert "playgroundTab: initial.playgroundTab || 'agents'" in fn_body("function applyInitialNavigation(")


def test_boot_css_paints_the_tab_without_flashing_my_agents():
    sel = 'html[data-nav-boot][data-nav-page="playground"][data-nav-playground-tab="portfolio-management"]'
    assert sel + ' .playground-subtabs [data-playground-tab="portfolio-management"]' in APP_HTML
    assert sel + ' #playgroundAgentsPanel' in APP_HTML  # joined to the existing display:none group
    assert sel + ' #playgroundPortfolioManagementPanel { display: block !important; }' in APP_HTML


def test_show_playground_panel_shows_the_demo_and_hides_it_elsewhere():
    body = strip_comments(fn_body("function showPlaygroundPanel("))
    branch = body[body.index("tab === 'portfolio-management') {"):]
    branch = branch[: branch.index("} else {")]
    assert "showPortfolioManagementDemo(" in branch and "loadAgents" not in branch
    assert "if (tab !== 'portfolio-management') window.PortfolioManagementDemo?.hide();" in body
    helper = strip_comments(fn_body("function showPortfolioManagementDemo("))
    assert ".show(panel)" in helper and "console.error" in helper
    assert "Portfolio Management did not load. Reload the page to try again." in helper  # fail-visible


def test_navigate_to_page_hides_the_panel_and_stops_the_demo_off_playground():
    body = strip_comments(fn_body("function navigateToPage("))
    assert "hide(document.getElementById('playgroundPortfolioManagementPanel'));" in body
    assert "if (page !== 'playground') window.PortfolioManagementDemo?.hide();" in body
