"""My Agents card: the capital figures, and paper trading switched off.

The card showed only the paper sleeve directly above a **Run Backtest** button,
which implied the figure was what the backtest would use -- it wasn't. Both
figures are now labelled side by side whenever paper trading is enabled.

Paper trading is switched off product-wide (`PAPER_TRADING_ENABLED = false`
in app.js; execution/paper_backend.py is still a stub), so the shipped card
shows the backtest figure alone and offers no paper-trading button at all.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _extract_function(src: str, name: str) -> str:
    for marker in (f"async function {name}(", f"function {name}("):
        start = src.find(marker)
        if start != -1:
            break
    else:
        raise AssertionError(f"{name} not found in app.js")
    depth = 0
    i = src.index("{", start)
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
        i += 1


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _harness(body: str, paper_enabled: bool = True) -> str:
    """Real functions lifted from app.js, with their few dependencies stubbed.

    ``paper_enabled`` defaults to True so the capital-resolution tests below
    keep seeing both figures; the shipped value is pinned separately.
    """
    return f"""
const PAPER_TRADING_ENABLED = {"true" if paper_enabled else "false"};
const MAX_BACKTEST_ALLOCATED_CAPITAL = 3000;
const DEFAULT_AGENT_CASH_ALLOCATION = 1000;
function escapeHtml(s) {{ return String(s); }}
function formatAgentCashAllocation(v) {{ return '$' + Number(v).toLocaleString(); }}
{_extract_function(_APP_JS, "resolveBacktestCapital")}
{_extract_function(_APP_JS, "renderAgentAllocatedCapitalHero")}
{body}
"""


def test_paper_trading_ships_switched_off():
    """The product decision itself: paper trading is disabled until it ships."""
    assert "const PAPER_TRADING_ENABLED = false;" in _APP_JS


def test_card_hides_the_paper_sleeve_when_paper_trading_is_off():
    out = _run_node(
        _harness(
            "console.log(renderAgentAllocatedCapitalHero("
            "{cash_allocation: 1000, backtest_allocation: 2500}));",
            paper_enabled=False,
        )
    )
    assert "Paper Trading" not in out
    assert "From My Portfolio" not in out
    assert "$1,000" not in out
    assert "Backtesting" in out
    assert "$2,500" in out
    assert "agent-card-capitals--single" in out


def test_card_shows_both_capitals():
    out = _run_node(
        _harness(
            "console.log(renderAgentAllocatedCapitalHero("
            "{cash_allocation: 1000, backtest_allocation: 2500}));"
        )
    )
    assert "Paper Trading" in out
    assert "Backtesting" in out
    assert "$1,000" in out
    assert "$2,500" in out


def test_card_backtest_capital_falls_back_to_the_sleeve():
    """An agent predating the column must not render a dash."""
    out = _run_node(
        _harness(
            "console.log(renderAgentAllocatedCapitalHero("
            "{cash_allocation: 2000, backtest_allocation: null}));"
        )
    )
    assert out.count("$2,000") == 2


def test_an_unset_backtest_capital_does_not_mirror_a_zero_paper_sleeve():
    """The two capitals diverge when the backtest one is UNSET -- by design.

    `cash_allocation` is `ge=0`: a $0 paper sleeve is a real, legal state and is
    shown honestly rather than padded to a default. `backtest_allocation` is
    `ge=0` too as of 2026-09-10, but this case is the NULL column -- "never
    configured" -- which mirrors the paper sleeve only when that sleeve is
    funded. A $0 sleeve is the ordinary state of someone who does not
    paper-trade at all, and mirroring it would silently zero the backtests of
    every such agent that never touched this field.
    """
    out = _run_node(
        _harness(
            "console.log(renderAgentAllocatedCapitalHero("
            "{cash_allocation: 0, backtest_allocation: null}));"
        )
    )
    assert "$0" in out
    assert "$1,000" in out


def test_a_saved_zero_backtest_capital_is_displayed_as_zero():
    """The other half of the split above, and the one that used to be lost.

    ``$0`` became a saveable backtest amount on 2026-09-10, but
    ``resolveBacktestCapital``'s ``value > 0`` still read it as absent -- so the
    card and the Run Backtest dialog rendered $1,000 (or the paper sleeve) over
    a setting the owner had explicitly chosen, after a save that reported
    success. Reopening Configure then wrote the displayed number back, which
    made the setting undo itself by being looked at.
    """
    out = _run_node(
        _harness(
            "console.log(renderAgentAllocatedCapitalHero("
            "{cash_allocation: 2000, backtest_allocation: 0}));"
        )
    )
    assert "$0" in out, "a saved $0 backtest capital was replaced by a fallback"
    assert "$1,000" not in out


def test_cards_offer_no_paper_trading_button():
    """Paper trading is switched off: no greyed "Run Paper Trading" button."""
    actions = _extract_function(_APP_JS, "renderAgentCardActions")
    assert ">Run Paper Trading<" not in actions


def test_status_badge_never_says_paper_trading_while_it_is_off():
    """A live/paper deployment flag (or the guest demo's is_live mock) must not
    resurrect the PAPER TRADING card while the feature is disabled."""
    fn = _extract_function(_APP_JS, "resolveAgentStatusBadge")
    out = _run_node(
        "const PAPER_TRADING_ENABLED = false;\n"
        + fn
        + "\nconsole.log(JSON.stringify(["
        "resolveAgentStatusBadge({is_live: true}).key,"
        "resolveAgentStatusBadge({deployment_status: 'paper', run_count: 1}).key,"
        "]));"
    )
    assert out.strip() == '["draft","backtested"]'


def test_run_backtest_lands_on_my_agents():
    """The whole point: the user sees the agent they just started."""
    run_backtest = _extract_function(_APP_JS, "runBacktest")
    assert "playgroundTab: 'agents'" in run_backtest
    assert "playgroundTab: 'backtest'" not in run_backtest


def test_running_state_survives_a_refresh():
    """Kept as a presence check only; the behaviour is covered elsewhere.

    This asserts the store exists, which is all a string match can do. Its
    arithmetic, its shape handling and the re-render-versus-patch decision are
    executed under node in `test_running_backtest_store.py` (issue #258). Do
    not grow this one -- a string check that looks like coverage is worse than
    an obvious presence check.
    """
    assert "sessionStorage" in _APP_JS
    assert "function markAgentBacktestRunning(" in _APP_JS
    assert "function clearAgentBacktestRunning(" in _APP_JS


def test_running_card_shows_an_indicator_and_elapsed_time():
    body = _extract_function(_APP_JS, "renderAgentRunningBody")
    assert "Backtesting" in body
    assert "agent-card-running-dot" in body
    assert "agent-card-running-bar" in body
    assert "formatBacktestElapsed" in body


def test_running_animation_respects_reduced_motion():
    """First continuously-animating element on the page."""
    css = (_FRONTEND / "styles.css").read_text(encoding="utf-8")
    start = css.index(".agent-card-running-dot")
    assert "prefers-reduced-motion" in css[start:]


def _css_rule(css: str, selector: str) -> str:
    start = css.index(selector)
    open_brace = css.index("{", start)
    depth = 0
    i = open_brace
    while i < len(css):
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
            if depth == 0:
                return css[start : i + 1]
        i += 1
    raise AssertionError(f"unclosed rule for {selector}")


def test_submeta_stays_one_line_so_cards_in_a_row_align():
    """A wrapping model line used to stagger Paper Trading / Configure
    across cards in the same grid row (high zoom, long names like
    'Nemotron 3 Nano 30b A3b · Hosted AI · U.S.').

    nowrap+ellipsis is not enough on its own: a grid item's min-width:auto
    lets the nowrap string grow the card, and a wrap that still happens
    (cached CSS, high zoom) must not change the box height.
    """
    css = (_FRONTEND / "styles.css").read_text(encoding="utf-8")
    card = _css_rule(css, ".agent-card,\n.participant-card {")
    assert "min-width: 0" in card
    identity = _css_rule(css, ".agent-card-identity-text {")
    assert "overflow: hidden" in identity
    rule = _css_rule(css, ".agent-card-submeta {")
    assert "white-space: nowrap" in rule
    assert "text-overflow: ellipsis" in rule
    assert "height: 1.35em" in rule
    assert "max-width: 100%" in rule
    assert "overflow-wrap: anywhere" not in rule
    placeholder = _css_rule(css, ".agent-card--placeholder .agent-card-submeta {")
    assert "white-space: normal" in placeholder
    assert "height: auto" in placeholder


def test_agent_card_submeta_exposes_full_line_on_hover():
    """When a market label is shown, title keeps the full string past ellipsis."""
    render = _extract_function(_APP_JS, "renderAgentCards")
    assert 'class="agent-card-submeta" title="' in render
    assert "overflow-wrap: anywhere" not in render
    assert "Hosted AI" not in render
