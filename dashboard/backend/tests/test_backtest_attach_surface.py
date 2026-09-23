"""Source guards for attachToLiveBacktest -- the deep-link / reload-mid-run path.

/app has no build step and no JS test toolchain, so these assert against the
shipped source as text, the convention set by test_ai_hedge_fund_frontend.py and
followed by test_frontend_live_trading_board.py. Deliberately node-free: the
node-gated modules *skip* rather than fail on a machine with no node, and a
guard that silently stops running is worth nothing.

Both regressions pinned here were introduced by publishing the progress file
before the bar loop, and neither is visible from the diff that caused it --
which is the case for source guards rather than behavioural ones.

**The stale trading log.** Making the child publish a phase before the loop
flipped `progress` from null to truthy for the whole pre-loop window (~18s of a
~21s launch, nearly all `loading_bars`). `attachToLiveBacktest` branched on that
truthiness: `if (progress) { ... } else if (!alreadyLive) { clearTradingLog() }`.
The pre-loop payload carries no `trades` and no `order_events` -- `publish_phase`
builds it as `dict(last)` over an empty `last` -- so `updateLiveTradingLog`
early-returns on it and the clear never ran. Switching from a finished run to an
in-flight one left the finished run's fills on screen under a "Loading market
data…" header, attributed to the run just starting.

`publish_phase`'s own docstring reasons that "every existing reader of the file
(chart, trading log, ETA anchor) keeps its early return", which is true and does
not help: the defect was in a *caller* branching on the payload's existence, not
in a *reader* of its fields. Any `if (x)` where `x` means "has this started yet"
becomes wrong the moment the thing starts earlier, so the guard below pins the
branch to the same predicate the painter returns on.

**The self-contradicting phase label.** BACKTEST_PHASE_LABELS' docblock sets the
ownership rule: the phase *names* are the contract between the two surfaces, the
Backtest panel prints the server's own sentence, and the agent card owns the
short labels. The panel called `formatBacktestPhase` -- the card's vocabulary --
so attaching to a run in `first_decision` wrote "Waiting on first decision" and
the 1s poller replaced it with "Waiting on the first model decision… (49 decision
bars queued)" a tick later: one node rewording itself unprompted, on the first
screen a returning user sees.
"""

from dashboard.backend.tests._frontend_source import APP_JS, fn_body, strip_comments

_ATTACH = strip_comments(fn_body("function attachToLiveBacktest("))
_UPDATE_LOG = strip_comments(fn_body("function updateLiveTradingLog("))
_FOLD = strip_comments(fn_body("function advanceBacktestProgress("))


def test_panel_prints_the_servers_sentence_not_the_cards_label():
    """The panel's message is `serverMessage`, never the card's short label."""
    assert "message: serverMessage" in _ATTACH
    assert "formatBacktestPhase" not in _ATTACH, (
        "attachToLiveBacktest paints the Backtest panel, which prints the "
        "server's own sentence; formatBacktestPhase is the agent card's "
        "vocabulary. Mixing them makes this node reword itself one poll later."
    )


def test_the_server_sentence_actually_reaches_attach():
    """A `serverMessage` parameter nothing passes is the same bug, silently."""
    assert "{ serverMessage = '' } = {}" in _ATTACH, (
        "the parameter must keep its empty-string default: the launch path "
        "calls attachToLiveBacktest without it."
    )
    assert "{ serverMessage: statusMessage }" in APP_JS, (
        "the poller's status.message must be handed to attachToLiveBacktest; "
        "without the call site the parameter defaults to '' forever and the "
        "count fallback silently becomes the only path."
    )
    assert "statusMessage = status.message || ''" in APP_JS


def test_the_log_is_cleared_on_a_payload_that_paints_nothing():
    """Keyed on the records, not on the payload's truthiness."""
    assert "!hasTradingLogRecords(progress)" in _ATTACH, (
        "the clear must ask 'did anything paint the log?', not 'is there a "
        "payload?' -- a pre-loop phase payload is truthy and paints nothing."
    )
    assert "} else if (!alreadyLive) {" not in _ATTACH, (
        "the clear must not sit in an else-branch of `if (progress)`: that is "
        "the exact shape that stopped running when the child began publishing "
        "before the bar loop."
    )
    assert _ATTACH.count("clearTradingLog(") == 1


def test_alreadylive_still_suppresses_the_clear():
    """Re-attaching to the run already on screen must not wipe its own fills."""
    assert "if (!alreadyLive && !hasTradingLogRecords(progress)) {" in _ATTACH


def test_one_owner_for_the_trading_log_predicate():
    """The painter and the clear branch read the same function."""
    assert "hasTradingLogRecords(progress)" in _UPDATE_LOG
    assert "order_events" not in _UPDATE_LOG, (
        "a second inline Array.isArray(...) copy here is how the painter and "
        "the caller drift apart again."
    )


def test_the_fold_does_not_carry_a_raw_server_clock():
    """`phase_started_at` is a server wall clock; the card derives from age."""
    assert "phaseStartedAt" not in _FOLD
    assert "phase_started_at" not in _FOLD, (
        "elapsed figures come from progress_age_seconds, computed server side "
        "so client-clock skew cannot report a phase starting in the future."
    )
