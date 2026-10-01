"""The sampling policy reaches the model call from the engine, on every branch.

Pinned by source shape, the way test_agent_runs_metadata pins the metadata
call site: the decision method needs a full market snapshot to reach either
branch, and a dropped keyword between two files is exactly the defect this
work fixes (the pipeline branch never forwarded temperature although the
single-prompt branch did).

Read by AST, and asserted **universally rather than by count**. An earlier
draft of this file asserted `src.count("_request_trading_decision(") == 2` and
`src.count("reasoning_effort=reasoning_effort,") == 3`. There are three call
sites, not two -- the third is the truncation-recovery retry -- so the numbers
were wrong and the test could never have gone green. But the numbers are the
smaller half of the problem: a count is the wrong *shape* of guard here. It
goes red when someone adds a correct fourth call site, and it stays green when
someone adds an incorrect one that happens to keep the total. "Every model
call in this method forwards both values" is the property the code has to
have, so it is the property the test states, and the number never appears.
The non-empty check is the other half of that trade: a universal assertion
over an empty set passes, so a rename that makes the query match nothing must
fail loudly rather than quietly cover nothing.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

from dashboard.backend.domain.backtesting import engine, portfolio_manager


def _callee(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls(method, *names: str) -> list[ast.Call]:
    """Every call to one of `names` inside `method`, as AST nodes.

    `textwrap.dedent` because `inspect.getsource` of a method keeps the class
    indentation, which `ast.parse` refuses outright.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _callee(node) in set(names)
    ]
    assert found, (
        f"no call to {sorted(names)} in {method.__qualname__} -- the query is "
        "stale, not the code. Fix this helper before trusting anything below: "
        "a 'for every call' assertion over nothing passes."
    )
    return found


def _kwargs(call: ast.Call) -> dict:
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}


def test_engine_forwards_both_values_to_the_manager():
    for call in _calls(
        engine.HourlyBacktester.run_agent_backtest,
        "make_trading_decision_with_llm",
    ):
        kwargs = _kwargs(call)
        assert kwargs.get("temperature") == "self.llm_temperature"
        assert kwargs.get("reasoning_effort") == "self.llm_reasoning_effort"


def test_engine_forwards_both_values_to_the_post_trade_call():
    """The daily post-trade analysis is a model call too.

    It is not on the decision path, so it is easy to forget -- and forgetting
    it is not a cosmetic miss: for DeepSeek V4 Pro and Qwen3.7 Plus the
    pinned policy is thinking *off*, so an unpinned post-trade call goes out
    with thinking on (~100s) while the run's metadata still says `pinned_v1`.
    """
    for call in _calls(
        engine.HourlyBacktester._run_daily_post_trade,
        "run_post_trade_analysis",
    ):
        kwargs = _kwargs(call)
        assert kwargs.get("temperature") == "self.llm_temperature", (
            f"post-trade call at line {call.lineno} of the method drops temperature"
        )
        assert kwargs.get("reasoning_effort") == "self.llm_reasoning_effort", (
            f"post-trade call at line {call.lineno} of the method drops reasoning_effort"
        )


def test_every_model_call_in_the_manager_forwards_both_values():
    """One policy governs both branches, and both branches are every call.

    The pipeline branch (`run_pipeline_decision`) and the single-prompt branch
    (`_request_trading_decision`) are the only places this method reaches a
    model. Whatever the current count of call sites, each one has to carry the
    same two values, because the alternative is a step that silently changes
    sampler mid-run while the metadata still reports the pinned policy.
    """
    for call in _calls(
        portfolio_manager.PortfolioManager.make_trading_decision_with_llm,
        "_request_trading_decision",
        "run_pipeline_decision",
    ):
        kwargs = _kwargs(call)
        assert kwargs.get("temperature") == "temperature", (
            f"{_callee(call)} at line {call.lineno} of the method drops temperature"
        )
        assert kwargs.get("reasoning_effort") == "reasoning_effort", (
            f"{_callee(call)} at line {call.lineno} of the method drops reasoning_effort"
        )


def test_a_recovery_call_is_the_same_request_at_a_higher_ceiling():
    """A recovery call may differ from the ordinary one by max_tokens alone.

    Two of them raise the ceiling: the final rescue after the empty-reply
    retries are spent, and the truncation retry after the parse fails. The
    truncation one fires on exactly the models the policy pins an effort for,
    so a recovery that re-asked without the sampling would be a *different
    request* on the runs most likely to need it -- for DeepSeek V4 Pro and
    Qwen3.7 Plus, one with thinking switched back on -- the thing the
    spec forbids when it says the recovery retry "carries the same sampling,
    because it is the same request at a higher ceiling".

    Stated as an equality against the ordinary call rather than as two
    `in` checks, so a third value added to one path and not the other is
    caught without this test having to learn its name.
    """
    calls = _calls(
        portfolio_manager.PortfolioManager.make_trading_decision_with_llm,
        "_request_trading_decision",
    )
    recovery = [
        c for c in calls
        if _kwargs(c).get("max_tokens") == "RECOVERY_MAX_OUTPUT_TOKENS"
    ]
    ordinary = [c for c in calls if c not in recovery]
    assert recovery, "no recovery call found -- the ceiling constant was renamed"
    assert ordinary, "no ordinary call found -- every call now raises the ceiling"
    for call in recovery:
        assert _kwargs(call) == {
            **_kwargs(ordinary[0]),
            "max_tokens": "RECOVERY_MAX_OUTPUT_TOKENS",
        }


def test_engine_accepts_the_two_kwargs():
    params = inspect.signature(engine.HourlyBacktester.__init__).parameters
    assert params["llm_temperature"].default is None
    assert params["llm_reasoning_effort"].default is None
