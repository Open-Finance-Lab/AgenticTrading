"""My Agents card: both capitals, and a signposted paper-trading affordance.

The card showed only the paper sleeve directly above a **Run Backtest** button,
which implied the figure was what the backtest would use -- it wasn't. Both
figures are now labelled side by side.

Run Paper Trading ships disabled: execution/paper_backend.py is still a stub
(Phase B), and a greyed button with no explanation reads as a bug.
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


def _harness(body: str) -> str:
    """Real functions lifted from app.js, with their few dependencies stubbed."""
    return f"""
const MAX_BACKTEST_ALLOCATED_CAPITAL = 10000;
const DEFAULT_AGENT_CASH_ALLOCATION = 1000;
function escapeHtml(s) {{ return String(s); }}
function formatAgentCashAllocation(v) {{ return '$' + Number(v).toLocaleString(); }}
{_extract_function(_APP_JS, "resolveBacktestCapital")}
{_extract_function(_APP_JS, "renderAgentAllocatedCapitalHero")}
{body}
"""


def test_card_shows_both_capitals():
    out = _run_node(
        _harness(
            "console.log(renderAgentAllocatedCapitalHero("
            "{cash_allocation: 1000, backtest_allocation: 5000}));"
        )
    )
    assert "Paper Trading" in out
    assert "Backtesting" in out
    assert "$1,000" in out
    assert "$5,000" in out


def test_card_backtest_capital_falls_back_to_the_sleeve():
    """An agent predating the column must not render a dash."""
    out = _run_node(
        _harness(
            "console.log(renderAgentAllocatedCapitalHero("
            "{cash_allocation: 2000, backtest_allocation: null}));"
        )
    )
    assert out.count("$2,000") == 2


def test_zero_paper_sleeve_displays_as_zero_but_backtest_capital_does_not():
    """The two capitals deliberately diverge at $0 -- this is not a bug.

    `cash_allocation` is `ge=0` server-side: a $0 paper sleeve is a real,
    legal state and must be shown honestly, not padded to a default. But
    `backtest_allocation` is `ge=1` server-side -- a backtest cannot run on
    $0 -- so `resolveBacktestCapital` treats a non-positive value as absent
    and falls through to the $1,000 default. Rendering these two the same
    way would either lie about a user's real (zero) money or hand a $0 into
    an API call that would 422.
    """
    out = _run_node(
        _harness(
            "console.log(renderAgentAllocatedCapitalHero("
            "{cash_allocation: 0, backtest_allocation: null}));"
        )
    )
    assert "$0" in out
    assert "$1,000" in out


def test_run_paper_trading_button_is_disabled_and_explained():
    actions = _extract_function(_APP_JS, "renderAgentCardActions")
    assert "Run Paper Trading" in actions
    assert "disabled" in actions
    assert "Paper trading is coming soon" in actions


def test_run_paper_trading_is_absent_from_live_paper_cards():
    """Paper cards show Open Agent; a second paper button would be nonsense."""
    actions = _extract_function(_APP_JS, "renderAgentCardActions")
    head, _, tail = actions.partition("if (statusKey === 'paper')")
    branch, _, rest = tail.partition("} else {")
    assert "Run Paper Trading" not in branch
