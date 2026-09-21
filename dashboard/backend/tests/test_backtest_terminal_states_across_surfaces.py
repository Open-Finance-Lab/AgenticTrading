"""Every backtest poller must know every terminal state the status route emits.

Three surfaces poll ``/backtest/status``: ``ensureBacktestPolling`` in app.js,
the Discord watcher in ``integrations/discord_bot.py``, and the standalone
``strategy.html`` share page served at ``/strategy``. A surface that does not
branch on a terminal shape does not fail loudly -- it falls through. The
Discord watcher spun out its entire polling budget and then reported "still
running" long after the answer existed; ``strategy.html`` did worse, exiting
its loop normally and then painting a PREVIOUS run's metrics under a green
"Backtest complete."

Source-shape guards, because that is the only kind that can catch this. A test
that feeds ONE surface a ``timed_out`` payload proves nothing about the two
surfaces nobody remembered to update -- which is exactly how the state added in
this PR reached two of the three.
"""

import re

import pytest

from dashboard.backend.tests._frontend_source import (
    FRONTEND,
    fn_body,
    strip_comments,
)

BACKEND = FRONTEND.parent / "backend"
DISCORD_SOURCE = (BACKEND / "integrations" / "discord_bot.py").read_text(
    encoding="utf-8"
)
STRATEGY_HTML = (FRONTEND / "strategy.html").read_text(encoding="utf-8")


def _strategy_poll_loop() -> str:
    """The `strategy.html` poll loop, comments stripped.

    Stripped because every branch below is also *described* in a comment right
    beside it; an un-stripped scan would be satisfied by the prose explaining
    the fix and would go on passing if the code were reverted.
    """
    scripts = re.findall(r"<script>(.*?)</script>", STRATEGY_HTML, re.DOTALL)
    assert scripts, "strategy.html has no inline <script> block"
    return strip_comments("\n".join(scripts))


def _discord_watcher_body() -> str:
    """`watch_and_deliver_backtest`, from the file rather than from an import.

    `discord.py` is an optional dependency (`requirements-discord.txt`), so
    importing the bot module here would skip this guard on CI -- which installs
    core requirements only, and is the run that has to catch a missed state.
    """
    start = DISCORD_SOURCE.index("async def watch_and_deliver_backtest(")
    rest = DISCORD_SOURCE[start + 1 :]
    end = re.search(r"\n(?:async def |def |@)", rest)
    body = rest[: end.start()] if end else rest
    # Python, so `strip_comments` (a JS scanner) does not apply; drop `#` lines
    # the same way and for the same reason.
    return "\n".join(
        line for line in body.split("\n") if not line.lstrip().startswith("#")
    )


TERMINAL_STATES = ("cancelled", "timed_out", "error")


@pytest.mark.parametrize("state", TERMINAL_STATES)
def test_the_browser_poller_branches_on_every_terminal_state(state):
    poller = strip_comments(fn_body("function ensureBacktestPolling("))
    assert f"status.{state}" in poller, (
        f"ensureBacktestPolling does not branch on `{state}`; a payload "
        f"carrying it falls through to whichever branch matches next"
    )


@pytest.mark.parametrize("state", TERMINAL_STATES)
def test_the_discord_watcher_branches_on_every_terminal_state(state):
    body = _discord_watcher_body()
    assert f'status.get("{state}")' in body, (
        f"the Discord watcher does not branch on `{state}`; the payload then "
        f"matches no shape, falls through with neither continue nor break, and "
        f"the for/else reports the run as still going"
    )


@pytest.mark.parametrize("state", TERMINAL_STATES)
def test_the_strategy_share_page_branches_on_every_terminal_state(state):
    loop = _strategy_poll_loop()
    assert f"st.{state}" in loop, (
        f"strategy.html does not branch on `{state}`; it keeps polling for the "
        f"rest of its budget and then renders /runs/latest/metrics anyway"
    )


def test_the_strategy_share_page_never_renders_metrics_it_did_not_see_complete():
    """The worst of the three failures, and the one a branch alone does not fix.

    Exiting the loop normally and exiting it on a success used to be the same
    path: both fell through to `/runs/latest/metrics`, so a run that never
    produced results painted whatever this session ran LAST under "Backtest
    complete." Branching on the new states shortens the wait; only refusing to
    render without an observed completion makes the outcome honest.
    """
    loop = _strategy_poll_loop()
    assert "completed = true" in loop
    assert "if (!completed)" in loop
    # The guard has to sit between the loop and the fetch, or it guards nothing.
    assert loop.index("if (!completed)") < loop.index("/runs/latest/metrics")


def test_a_timed_out_background_run_is_announced_like_a_cancelled_one():
    """The free outcome had the acknowledgement and the billed one had silence.

    `finishedFocused` is populated only for the run the Backtest panel is
    pinned to, so a run that timed out in the background cleared its My Agents
    card and told the user nothing -- not that it stopped, not why, and not
    that Credits were spent. A cancel, which costs nothing, already toasted.

    A source-shape guard for the same reason the cancel toast beside it is one:
    a payload-driven test supplies a focused run and never exercises the
    unfocused branch this is about.
    """
    poller = strip_comments(fn_body("function ensureBacktestPolling("))

    assert "status.timed_out && unwatched" in poller
    assert "showAppToast('Backtest stopped at the time limit" in poller
    # The cancel toast keeps its extra suppression; a timeout has none to keep,
    # because nothing announces a timeout locally.
    assert "status.cancelled && !announcedHere && unwatched" in poller
