"""Guards for issues #390 and #365 in ``get_leaderboard``.

**#390 — a NULL equity published as a real $0.**
``float(pt.get("equity") or 0)`` turned "nobody observed this hour" into "the
account held nothing". ``chart_equity_curve`` prepends an open tick at starting
capital, so the series read as a −100% loss, and ``align_equity_curves`` puts
every entry on one shared axis — so a single such point sets the y-range and
flattens every honest curve on the board. The fix is the wire format: the server
emits ``null`` and each surface null-fills it the way it already null-fills a
missing timestamp.

**#365 — one board, two capital bases.** The twelve published contest curves in
the committed seed database were all computed at **$100,000** (verified below,
against the file itself). ``leaderboard.json`` was later changed to
``initial_capital: 10000``, and because the cache key omitted seed capital the
two halves of the board diverged: ``auto_compute`` baselines recompute at the
new number while LLM entries stay cached at the old one. That is not a display
problem — a $10k account buys whole shares in a coarser quantum and pays
per-trade costs against a smaller base, so it is a *different run* of the same
strategy. Rescaling the dollar levels hides the difference and repairs nothing.
This branch aligns the config to $100,000, so the stored rows are correct as
they stand and no re-run is needed (#194 is blocked on LLM credits).

⚠ **Seed capital is a ranking key in the cache, never a filter, and the tests
below pin that as a spend control.** A config change that made rows *miss* would
not be quiet: ``ensure_leaderboard_runs`` answers a miss by recomputing on a
public unauthenticated GET, and ``_daily_models_status`` reports one as
*pending*, which is what ``maybe_schedule_daily_leaderboard_refresh`` acts on —
``refresh_daily_leaderboard`` then calls ``deploy_model_run`` for every
configured LLM entry. With ``LEADERBOARD_DAILY_AUTO_DEPLOY`` armed, one edit to
``leaderboard.json`` would buy a billable re-run of the whole board from an
anonymous request. ``_warn_on_capital_drift`` is the signal instead, following
``_warn_on_feed_drift``, which set that policy for the identical problem one
field over.

The frontend cases run the real extracted functions under node, following
test_frontend_leaderboard_hover.py.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import dashboard.backend.database as db_module
import dashboard.backend.domain.leaderboard.service as lb_service

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_LEADERBOARD_JS = (_FRONTEND / "js" / "leaderboard.js").read_text(encoding="utf-8")
_HOME_JS = (_FRONTEND / "home-page.js").read_text(encoding="utf-8")
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")

_SESSION = "leaderboard-contest"
_START = "2026-04-15"
_END = "2026-05-15"
# Read from the config rather than hardcoded: the $100,000 alignment lands as its
# own commit, and every case here must hold on both sides of it. A case that is
# *about* the alignment lives with that commit, not here.
_DISPLAY_CAPITAL = float(lb_service.load_leaderboard_config()["initial_capital"])
# A seed no board publishes, for the drift cases.
_OTHER_CAPITAL = _DISPLAY_CAPITAL / 10

# Captured before the `board` fixture stubs it out, so the two cases that are
# *about* the compute path can exercise the real one.
_REAL_ENSURE_LEADERBOARD_RUNS = lb_service.ensure_leaderboard_runs


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


class _Board:
    """A leaderboard database wired into the service, plus the two NULL shims.

    ⚠ **A NULL cannot be written through the normal API today**, and that is
    worth stating rather than working around silently:
    ``equity_timeseries.equity`` and ``agent_runs.initial_equity`` are declared
    ``NOT NULL`` in *both* backends (``database.py``, ``database_postgres.py``),
    and the committed seed database — which *is* prod's database — holds no NULL
    in either column today. So ``insert_equity_points`` raises rather than
    storing the shape these defects mishandle.

    That does not make the defects hypothetical, it makes the schema a *second*
    layer that happens to hold:

    * ``CREATE TABLE IF NOT EXISTS`` never tightens an existing table, so any
      database created under a looser schema keeps it — and the committed seed
      DB is itself proof that this drift happens: its ``equity_timeseries``
      declares ``cash`` and ``positions_value`` **nullable** (the current
      ``database.py`` declares both NOT NULL), and ``get_leaderboard`` scaled
      those two columns through exactly the same ``or 0``.
    * ``get_equity_curve`` is a plain row reader with no validation, so anything
      the column can hold reaches the payload.

    These shims therefore hand ``get_leaderboard`` the rows a looser database
    would, at the seam it actually reads them through.
    """

    def __init__(self, test_db, monkeypatch):
        self.db = test_db
        self._curves = {}
        self._seeds = {}
        self._metadata = {}
        self._finals = {}
        real_curve = test_db.get_equity_curve
        real_runs = test_db.get_runs_by_session

        def get_equity_curve(run_id):
            if run_id in self._curves:
                return [dict(pt) for pt in self._curves[run_id]]
            return real_curve(run_id)

        def get_runs_by_session(session_id):
            rows = real_runs(session_id)
            for row in rows:
                if row["run_id"] in self._seeds:
                    row["initial_equity"] = self._seeds[row["run_id"]]
                if row["run_id"] in self._metadata:
                    row["metadata"] = self._metadata[row["run_id"]]
                if row["run_id"] in self._finals:
                    row["final_equity"] = self._finals[row["run_id"]]
            return rows

        monkeypatch.setattr(test_db, "get_equity_curve", get_equity_curve)
        monkeypatch.setattr(test_db, "get_runs_by_session", get_runs_by_session)

    def seed_run(
        self,
        strategy_id,
        *,
        initial_equity,
        equities,
        run_id=None,
        final_equity=None,
        session_id=_SESSION,
    ):
        """One cached leaderboard run and its curve. ``equities`` may hold None."""
        run_id = run_id or (
            f"lb_{strategy_id}_{_START.replace('-', '')}_{_END.replace('-', '')}"
        )
        finite = [e for e in equities if e is not None]
        self.db.insert_run(
            run_id=run_id,
            session_id=session_id,
            agent_name="Agentic Trading Lab",
            mode="leaderboard",
            start_date=_START,
            end_date=_END,
            # A placeholder when the case under test is "no seed recorded": the
            # column is NOT NULL, so the NULL is applied on read (see above).
            initial_equity=initial_equity if initial_equity is not None else 1.0,
            final_equity=(
                final_equity
                if final_equity is not None
                else (finite[-1] if finite else None)
            ),
            total_return=0.05,
            sharpe_ratio=1.0,
            max_drawdown=-0.01,
            num_trades=1,
            llm_model=strategy_id,
        )
        if initial_equity is None:
            self._seeds[run_id] = None
        self._curves[run_id] = [
            {
                "timestamp": f"2026-04-{15 + i:02d}T14:00:00+00:00",
                "equity": equity,
                "cash": equity,
                "positions_value": 0 if equity is not None else None,
                "daily_return": 0,
            }
            for i, equity in enumerate(equities)
        ]
        return run_id

    def set_metadata(self, run_id, metadata):
        """What ``_llm_run_metadata`` would have written for this run (PR #366)."""
        self._metadata[run_id] = metadata

    def set_final_equity(self, run_id, value):
        """``agent_runs.final_equity`` is nullable in both backends."""
        self._finals[run_id] = value


@pytest.fixture
def board(tmp_path, monkeypatch):
    """An empty leaderboard database wired into the service.

    ``ensure_leaderboard_runs`` is stubbed out for the same reason
    test_leaderboard_api.py stubs it: unstubbed it reaches Alpaca for any
    strategy it cannot find cached, which is all of them here.
    """
    test_db = db_module.BacktestDatabase(db_path=tmp_path / "leaderboard.db")
    monkeypatch.setattr(db_module, "db", test_db)
    monkeypatch.setattr(lb_service, "db", test_db)
    monkeypatch.setattr(
        lb_service,
        "ensure_leaderboard_runs",
        lambda force_refresh=False, period="contest", config=None: {
            "session_id": _SESSION,
            "start_date": _START,
            "end_date": _END,
            "period": "contest",
            "created": 0,
            "refreshed_at": "2026-09-11T00:00:00+00:00",
        },
    )
    # One-shot warning state is module-level; a later test seeding the same
    # drift would otherwise see no line at all.
    monkeypatch.setattr(lb_service, "_warned_seed_mismatch", set())
    monkeypatch.setattr(lb_service, "_warned_capital_drift", set())
    return _Board(test_db, monkeypatch)


def _seed_run(board, *args, **kwargs):
    return board.seed_run(*args, **kwargs)


def _entry(payload, entry_id):
    return next((e for e in payload["entries"] if e["entry_id"] == entry_id), None)


# --------------------------------------------------------------------------
# #390 — a NULL equity is a gap, never a $0 account
# --------------------------------------------------------------------------


def test_a_null_equity_point_is_published_as_null_and_never_as_zero(board):
    _seed_run(board, "djia_index", initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL, None, _DISPLAY_CAPITAL * 1.01])

    entry = _entry(lb_service.get_leaderboard(), "djia_index")
    equities = [pt["equity"] for pt in entry["equity_curve"]]

    assert None in equities, (
        "a stored NULL equity must survive to the wire as null; `or 0` published "
        "an observation nobody made"
    )
    assert 0 not in equities and 0.0 not in equities


def test_a_null_equity_point_does_not_drag_the_shared_axis(board):
    """The damage is not one bad marker.

    ``align_equity_curves`` puts every entry on one axis, so a single −100%
    series sets the y-range and collapses the real board's spread into a sliver
    at the top. The floor asserted here is deliberately crude: anything near
    zero means a null became a dollar amount again.
    """
    _seed_run(board, "djia_index", initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL, None, _DISPLAY_CAPITAL * 1.01])
    _seed_run(board, "spy_index", initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL, _DISPLAY_CAPITAL * 1.005, _DISPLAY_CAPITAL * 1.012])

    payload = lb_service.get_leaderboard()
    plotted = [
        pt["equity"]
        for entry in payload["entries"]
        for pt in entry["equity_curve"]
        if pt["equity"] is not None
    ]

    assert plotted, "the board published no plottable points at all"
    assert min(plotted) > _DISPLAY_CAPITAL * 0.5, (
        f"lowest plotted equity {min(plotted)} — a missing observation has been "
        "rendered as a zero-dollar portfolio somewhere on the board"
    )


def test_cash_and_positions_value_are_nulled_too(board):
    """Not defence in depth — the reachable half.

    ``equity`` is ``NOT NULL`` in both backends, but the committed seed
    database's own ``equity_timeseries`` declares ``cash`` and
    ``positions_value`` **nullable** (``database.py`` declares them NOT NULL —
    ``CREATE TABLE IF NOT EXISTS`` never tightened the existing table). Those
    two columns went through exactly the same ``or 0``.
    """
    _seed_run(board, "djia_index", initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL, None])

    curve = _entry(lb_service.get_leaderboard(), "djia_index")["equity_curve"]
    gap = next(pt for pt in curve if pt["equity"] is None)

    assert gap["cash"] is None
    assert gap["positions_value"] is None


def test_a_real_zero_equity_is_still_published(board):
    """A blown account is not a missing observation and must not become a gap."""
    _seed_run(board, "djia_index", initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL, 0, 0])

    entry = _entry(lb_service.get_leaderboard(), "djia_index")
    equities = [pt["equity"] for pt in entry["equity_curve"]]

    assert 0.0 in equities
    assert equities.count(None) == 0


# --------------------------------------------------------------------------
# #390 — absent must not be byte-identical to broken
# --------------------------------------------------------------------------


def test_a_wholly_null_curve_logs_an_error_at_the_wholesale_boundary(board, capsys):
    _seed_run(board, "djia_index", initial_equity=_DISPLAY_CAPITAL, equities=[None, None, None])

    lb_service.get_leaderboard()
    out = capsys.readouterr().out

    assert "ERROR" in out and "djia_index" in out
    assert "broken, not absent" in out, (
        "a contract break must be distinguishable in the log from an empty "
        "window; a per-point warning cannot report a wholesale one"
    )


def test_a_curve_with_no_points_at_all_logs_an_error(board, capsys):
    """A cached run with no curve is a broken write, not an empty window.

    It publishes as a single opening tick at starting capital — a row that looks
    like a flat, uneventful month.
    """
    _seed_run(board, "djia_index", initial_equity=_DISPLAY_CAPITAL, equities=[])

    payload = lb_service.get_leaderboard()
    out = capsys.readouterr().out

    assert "ERROR" in out and "no equity points at all" in out
    assert len(_entry(payload, "djia_index")["equity_curve"]) == 1


def test_a_partial_gap_logs_one_aggregate_warning_not_one_line_per_point(board, capsys):
    _seed_run(
        board,
        "djia_index",
        initial_equity=_DISPLAY_CAPITAL,
        equities=[_DISPLAY_CAPITAL, None, None, _DISPLAY_CAPITAL * 1.01],
    )

    lb_service.get_leaderboard()
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "djia_index" in ln]
    gap_lines = [ln for ln in lines if "gaps" in ln]

    assert len(gap_lines) == 1, gap_lines
    assert "WARNING" in gap_lines[0] and "2 of 4" in gap_lines[0]


# --------------------------------------------------------------------------
# #365 — the seed comes from the run, never from the config
# --------------------------------------------------------------------------


def test_a_run_with_no_seed_and_no_usable_first_point_is_skipped_and_logged(board, capsys):
    """A published row is a claim, and here there is no honest number to make it with.

    The old fallback substituted the *config's* capital, which made ``scale``
    exactly 1.0 — so the entry shipped at whatever scale the run happened to
    have, labelled with the board's.
    """
    _seed_run(board, "djia_index", initial_equity=None, equities=[None, None])
    _seed_run(board, "spy_index", initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL, _DISPLAY_CAPITAL * 1.01])

    payload = lb_service.get_leaderboard()
    out = capsys.readouterr().out

    assert _entry(payload, "djia_index") is None, (
        "an entry whose seed nobody recorded must not be published at an "
        "unverified scale"
    )
    assert _entry(payload, "spy_index") is not None, "the rest of the board still ships"
    assert "ERROR" in out and "djia_index" in out and "#365" in out


def test_a_stored_zero_seed_does_not_collapse_to_config_capital(board, capsys):
    """`run.get("initial_equity") or config[...]` — `0.0` is falsy.

    The column is NOT NULL in both backends so the state cannot occur today, and
    that is exactly why it had to be tightened while the line was open: nothing
    else would ever have caught it. A zero seed is not a seed.
    """
    _seed_run(board, "djia_index", initial_equity=0.0, equities=[None, None])

    payload = lb_service.get_leaderboard()

    assert _entry(payload, "djia_index") is None
    assert "ERROR" in capsys.readouterr().out


def test_a_missing_seed_is_derived_from_the_curves_own_first_point(board):
    """The curve is a fact about the run; the config is a fact about the board."""
    _seed_run(
        board,
        "djia_index",
        initial_equity=None,
        equities=[_OTHER_CAPITAL, _OTHER_CAPITAL * 1.05],
        final_equity=_OTHER_CAPITAL * 1.05,
    )

    entry = _entry(lb_service.get_leaderboard(), "djia_index")
    equities = [pt["equity"] for pt in entry["equity_curve"] if pt["equity"] is not None]

    assert max(equities) == pytest.approx(_DISPLAY_CAPITAL * 1.05), (
        "a curve seeded at a tenth of the board must be rescaled from its own "
        f"first point, not published unscaled: got {max(equities)}"
    )
    assert entry["initial_equity"] == pytest.approx(_DISPLAY_CAPITAL)


def test_the_scaling_shim_says_it_is_a_shim():
    """Scaling is only honest for a scale-free strategy.

    Position sizing, whole-share quanta and per-trade costs make a $10k run a
    genuinely different run from a $100k one. Nothing in the code stops a reader
    treating the rescale as equivalence, so the comment has to.
    """
    source = Path(lb_service.__file__).read_text(encoding="utf-8")
    marker = "SCALING IS A COMPATIBILITY SHIM, NOT A RE-RUN"
    assert marker in source
    note = source[source.index(marker) : source.index(marker) + 900]
    assert "#194" in note, "the comment must point at the re-run that actually fixes it"


# --------------------------------------------------------------------------
# #365 criterion 3 — warn on mixed capital, on the feed-drift pattern
# --------------------------------------------------------------------------


def test_mixed_capital_within_one_board_is_warned_about(board, capsys):
    _seed_run(board, "djia_index", initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL])
    _seed_run(board, "spy_index", initial_equity=_OTHER_CAPITAL, equities=[_OTHER_CAPITAL])

    lb_service.get_leaderboard()
    # The board-level line specifically. The per-entry "not comparable with"
    # line below it reports the drop, which is a different fact.
    warnings = [
        ln
        for ln in capsys.readouterr().out.splitlines()
        if "Curves seeded at different capital" in ln
    ]

    assert len(warnings) == 1, warnings
    assert f"${_OTHER_CAPITAL:,.2f}" in warnings[0]
    assert f"${_DISPLAY_CAPITAL:,.2f}" in warnings[0]


def test_the_capital_warning_fires_once_per_process_like_the_feed_warning(board, capsys):
    """This runs on the hot cached path of a public endpoint.

    A line per page load buries itself, which is the reason ``_warn_on_feed_drift``
    dedupes and the reason this one copies it.
    """
    _seed_run(board, "djia_index", initial_equity=_OTHER_CAPITAL, equities=[_OTHER_CAPITAL])

    lb_service.get_leaderboard()
    lb_service.get_leaderboard()
    warnings = [
        ln
        for ln in capsys.readouterr().out.splitlines()
        if "Curves seeded at different capital" in ln
    ]

    assert len(warnings) == 1, warnings


def test_the_warning_sees_the_llm_entries_ensure_leaderboard_runs_cannot(board, capsys):
    """`ensure_leaderboard_runs` skips every `auto_compute: false` entry before it
    looks anything up, so its own `cached_runs` list is baselines only — and the
    LLM half is exactly where the stale capital lives. `get_leaderboard` is the
    only seam that sees the whole board.
    """
    _seed_run(board, "claude_haiku_4_5", initial_equity=_OTHER_CAPITAL, equities=[_OTHER_CAPITAL])

    lb_service.get_leaderboard()
    out = capsys.readouterr().out

    assert "Curves seeded at different capital" in out


# --------------------------------------------------------------------------
# #365 criterion 1 — the key distinguishes seeds, and a drifted row is inert
# --------------------------------------------------------------------------


def test_the_run_id_is_seed_free_so_one_window_holds_exactly_one_row(board):
    """The invariant that keeps a refresh from doubling the board.

    Putting the seed in ``_run_id`` was tried and reverted: ``insert_run`` is
    INSERT OR REPLACE on this id, and the twelve committed rows carry the
    *seed-free* id — so a suffixed id would not replace them, it would insert
    alongside them. The first force-refresh after such a change doubles every
    entry and orphans twelve equity curves nothing prunes.
    """
    import inspect

    assert lb_service._run_id("djia_index", _START, _END) == (
        "lb_djia_index_20260415_20260515"
    )
    params = inspect.signature(lb_service._run_id).parameters
    assert list(params) == ["strategy_id", "start_date", "end_date"], (
        "a seed-dependent run id inserts beside the committed rows instead of "
        "replacing them"
    )


def test_a_refresh_replaces_the_existing_row_rather_than_adding_one(board, monkeypatch):
    """End to end, against the real writer, at a seed the stored row disagrees with.

    The id must not vary with capital, so the second write lands on the first
    row. Two rows for one window would be worse than the drift they record.
    """

    class _Strategy:
        def num_trades(self):
            return 0

        def required_symbols(self):
            return []

        def run(self, bars, start_date, end_date, initial_capital):
            return [
                {
                    "timestamp": f"{start_date}T14:00:00+00:00",
                    "equity": initial_capital,
                    "cash": initial_capital,
                    "positions_value": 0,
                }
            ]

    _seed_run(board, "djia_index", initial_equity=_OTHER_CAPITAL, equities=[_OTHER_CAPITAL])
    monkeypatch.setattr(lb_service, "get_strategy", lambda entry: _Strategy())

    _REAL_ENSURE_LEADERBOARD_RUNS(
        force_refresh=True,
        config={
            "session_id": _SESSION,
            "start_date": _START,
            "end_date": _END,
            "initial_capital": _DISPLAY_CAPITAL,
            "period": "contest",
            "strategies": [
                {
                    "id": "djia_index",
                    "name": "Agentic Trading Lab",
                    "strategy": "market_index",
                }
            ],
        },
    )

    rows = [
        run
        for run in board.db.get_runs_by_session(_SESSION)
        if run.get("llm_model") == "djia_index"
    ]
    assert len(rows) == 1, [r["run_id"] for r in rows]
    assert rows[0]["initial_equity"] == pytest.approx(_DISPLAY_CAPITAL)


def test_two_rows_for_one_window_are_ranked_not_guessed_between(board):
    """Defensive, and honest about it.

    The id invariant above means one window holds one row, so this state is not
    reachable through the writers today. It is reachable through an imported or
    hand-repaired database, and ``created_at DESC`` order is not a reason to
    prefer one seed over another.
    """
    _seed_run(
        board,
        "djia_index",
        initial_equity=_OTHER_CAPITAL,
        equities=[_OTHER_CAPITAL],
        run_id="lb_djia_index_20260415_20260515",
    )
    _seed_run(
        board,
        "djia_index",
        initial_equity=_DISPLAY_CAPITAL,
        equities=[_DISPLAY_CAPITAL],
        run_id="lb_djia_index_20260415_20260515_imported",
    )

    at_other = lb_service._find_cached_run(
        "djia_index", _START, _END, _SESSION, _OTHER_CAPITAL
    )
    at_board = lb_service._find_cached_run(
        "djia_index", _START, _END, _SESSION, _DISPLAY_CAPITAL
    )

    assert at_other["initial_equity"] == pytest.approx(_OTHER_CAPITAL)
    assert at_board["initial_equity"] == pytest.approx(_DISPLAY_CAPITAL)


def test_a_drifted_row_is_still_a_cache_hit(board):
    """THE SEED IS A RANKING KEY, NOT A FILTER — and that is a spend control.

    Refusing the row would make it *missing*, and missing is what
    ``ensure_leaderboard_runs`` answers by recomputing and what
    ``_daily_models_status`` reports as pending. See the module docstring.
    """
    _seed_run(board, "djia_index", initial_equity=_OTHER_CAPITAL, equities=[_OTHER_CAPITAL])

    run, rank = lb_service._resolve_cached_run(
        "djia_index", _START, _END, _SESSION, _DISPLAY_CAPITAL
    )

    assert run is not None, "a drifted row must never read as a cache MISS"
    assert rank == lb_service._CACHE_STALE


def test_a_row_with_no_recorded_seed_is_a_cache_hit(board):
    _seed_run(board, "spy_index", initial_equity=None, equities=[_DISPLAY_CAPITAL])

    run, rank = lb_service._resolve_cached_run(
        "spy_index", _START, _END, _SESSION, _DISPLAY_CAPITAL
    )

    assert run is not None
    assert rank == lb_service._CACHE_UNRECORDED


def test_a_float_wobble_in_a_stored_seed_is_the_same_seed(board):
    """The committed board stores 100000.00000000003 for one entry."""
    _seed_run(
        board, "djia_index", initial_equity=_DISPLAY_CAPITAL + 3e-11, equities=[_DISPLAY_CAPITAL]
    )

    run, rank = lb_service._resolve_cached_run(
        "djia_index", _START, _END, _SESSION, _DISPLAY_CAPITAL
    )

    assert run is not None
    assert rank == lb_service._CACHE_MATCH


def test_an_exact_seed_beats_an_unrecorded_one_which_beats_a_drifted_one(board):
    for run_id, seed in (
        ("lb_djia_index_20260415_20260515", _OTHER_CAPITAL),
        ("lb_djia_index_20260415_20260515_x", None),
        ("lb_djia_index_20260415_20260515_y", _DISPLAY_CAPITAL),
    ):
        _seed_run(
            board,
            "djia_index",
            initial_equity=seed,
            equities=[seed or _DISPLAY_CAPITAL / 2],
            run_id=run_id,
        )

    run, rank = lb_service._resolve_cached_run(
        "djia_index", _START, _END, _SESSION, _DISPLAY_CAPITAL
    )
    assert rank == lb_service._CACHE_MATCH
    assert run["run_id"] == "lb_djia_index_20260415_20260515_y"


def test_a_drifted_baseline_is_never_recomputed_on_a_public_get(board, monkeypatch):
    """`ensure_leaderboard_runs` runs on every public GET of the board.

    Treating a config change as "missing" would refetch Alpaca on every page
    load for as long as the disagreement lasts — the failure
    ``_warn_on_feed_drift`` was written to avoid, one field over.
    """
    _seed_run(board, "djia_index", initial_equity=_OTHER_CAPITAL, equities=[_OTHER_CAPITAL])

    def _explode(*args, **kwargs):
        raise AssertionError("a drifted row must not trigger a recompute")

    monkeypatch.setattr(lb_service, "fetch_hourly_bars", _explode)
    monkeypatch.setattr(lb_service, "get_strategy", _explode)

    meta = _REAL_ENSURE_LEADERBOARD_RUNS(
        config={
            "session_id": _SESSION,
            "start_date": _START,
            "end_date": _END,
            "initial_capital": _DISPLAY_CAPITAL,
            "period": "contest",
            "strategies": [{"id": "djia_index", "strategy": "market_index"}],
        }
    )

    assert meta["cache_hit"] is True
    assert meta["created"] == 0


def test_a_drifted_model_is_never_redeployed_on_a_public_get(board, monkeypatch):
    """The billable path: `get_leaderboard(period="daily")` ->
    `maybe_schedule_daily_leaderboard_refresh` -> `refresh_daily_leaderboard`,
    which calls `deploy_model_run` for EVERY configured LLM entry. It short
    circuits only on a `_resolve_cached_run` hit, so a drifted row that read as
    a miss would buy a real LLM run from an anonymous request.
    """
    _seed_run(board, "claude_haiku_4_5", initial_equity=_OTHER_CAPITAL, equities=[_OTHER_CAPITAL])

    def _explode(*args, **kwargs):
        raise AssertionError("a drifted row must not trigger a billable deploy")

    monkeypatch.setattr(lb_service, "get_strategy", _explode)

    result = lb_service.deploy_model_run("claude_haiku_4_5")

    assert result["cached"] is True
    assert result["run_id"] == "lb_claude_haiku_4_5_20260415_20260515"


def test_a_drifted_model_is_counted_cached_not_pending(board, monkeypatch):
    """`models_pending > 0` is the whole trigger for the deploy loop above."""
    config = lb_service.resolve_leaderboard_config("contest")
    entries = lb_service.llm_leaderboard_entries(config)
    _seed_run(board, entries[0]["id"], initial_equity=_OTHER_CAPITAL, equities=[_OTHER_CAPITAL])

    index, drifted = lb_service._cached_run_index(
        _START, _END, _SESSION, _DISPLAY_CAPITAL
    )

    assert entries[0]["id"] in index, "drifted rows are cached, never missing"
    assert entries[0]["id"] in drifted, "and they are still reported as drifted"


def test_the_batched_index_agrees_with_the_scan(board):
    """``_cached_run_index`` exists only to be ``_resolve_cached_run`` in one query.

    Two predicates for one lookup is how the daily board's "models cached" count
    and the board it then renders end up disagreeing.
    """
    _seed_run(
        board,
        "djia_index",
        initial_equity=_OTHER_CAPITAL,
        equities=[_OTHER_CAPITAL],
        run_id="lb_djia_index_20260415_20260515",
    )
    _seed_run(
        board,
        "djia_index",
        initial_equity=_DISPLAY_CAPITAL,
        equities=[_DISPLAY_CAPITAL],
        run_id="lb_djia_index_20260415_20260515_100000",
    )
    _seed_run(board, "spy_index", initial_equity=None, equities=[_DISPLAY_CAPITAL])

    index, _ = lb_service._cached_run_index(_START, _END, _SESSION, _DISPLAY_CAPITAL)
    for strategy_id in ("djia_index", "spy_index"):
        scanned = lb_service._find_cached_run(
            strategy_id, _START, _END, _SESSION, _DISPLAY_CAPITAL
        )
        assert index[strategy_id]["run_id"] == scanned["run_id"], strategy_id


# --------------------------------------------------------------------------
# #365 — strategy_prompt, the second key dimension (PR #366)
# --------------------------------------------------------------------------


def test_a_changed_instruction_does_not_silently_reuse_the_old_curve(board):
    """The Open Track's competing variable rewrites the model's whole strategy.

    Recorded in ``agent_runs.metadata`` since PR #366, and omitted from the key
    until now — so editing an entry's instruction and redeploying returned the
    cached row.
    """
    run_id = _seed_run(board, "claude_haiku_4_5", initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL])
    board.set_metadata(run_id, {"strategy_prompt": "buy the dip"})

    same = lb_service._resolve_cached_run(
        "claude_haiku_4_5", _START, _END, _SESSION, _DISPLAY_CAPITAL, "buy the dip"
    )
    changed = lb_service._resolve_cached_run(
        "claude_haiku_4_5", _START, _END, _SESSION, _DISPLAY_CAPITAL, "sell the rip"
    )

    assert same[1] == lb_service._CACHE_MATCH
    assert changed[1] == lb_service._CACHE_STALE


def test_an_unrecorded_instruction_is_unknown_and_not_a_mismatch(board):
    """Every row in the committed seed database has ``metadata = NULL``.

    Reading that as "this run used no prompt" would mark all twelve drifted the
    moment any entry gains one.
    """
    _seed_run(board, "claude_haiku_4_5", initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL])

    run, rank = lb_service._resolve_cached_run(
        "claude_haiku_4_5", _START, _END, _SESSION, _DISPLAY_CAPITAL, "buy the dip"
    )

    assert run is not None
    assert rank == lb_service._CACHE_UNRECORDED


def test_a_prompt_the_config_does_not_set_is_not_compared(board):
    """``_llm_run_metadata`` resolves the prompt off the strategy impl, not only
    off ``leaderboard.json``, so a row can legitimately record one the config
    does not name. Comparing one-sidedly would drop it off the board.
    """
    run_id = _seed_run(board, "claude_haiku_4_5", initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL])
    board.set_metadata(run_id, {"strategy_prompt": "resolved from the impl"})

    run, rank = lb_service._resolve_cached_run(
        "claude_haiku_4_5", _START, _END, _SESSION, _DISPLAY_CAPITAL, None
    )

    assert run is not None
    assert rank == lb_service._CACHE_MATCH


# --------------------------------------------------------------------------
# #365 — one capital base per board, and never an empty board
# --------------------------------------------------------------------------


def test_a_config_typo_cannot_empty_the_board(board, capsys):
    """THE DEFECT IS MIXED CAPITAL, NOT CAPITAL THAT DISAGREES WITH CONFIG.

    Entries that all share one seed are mutually comparable: the board is
    internally consistent and merely mislabelled. Dropping every row because the
    config names a third number turns a labelling problem into an outage on the
    Competition Leaderboard, which CLAUDE.md calls the acquisition hook — from a
    one-character edit.
    """
    for entry_id in ("djia_index", "spy_index", "buy_hold_djia"):
        _seed_run(board, entry_id, initial_equity=_OTHER_CAPITAL, equities=[_OTHER_CAPITAL])

    # A third value: neither the config's nor the stored rows'.
    config = dict(lb_service.load_leaderboard_config())
    config["initial_capital"] = _OTHER_CAPITAL * 5
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(lb_service, "load_leaderboard_config", lambda: config)
        payload = lb_service.get_leaderboard()

    assert len(payload["entries"]) == 3, "a config typo must not empty the board"
    assert "Curves seeded at different capital" in capsys.readouterr().out


def test_a_minority_seed_is_dropped_rather_than_ranked_against_the_rest(board, capsys):
    """This is the state that actually corrupts a ranking.

    A $10k run and a $100k run of the same strategy are different runs — the
    smaller account buys whole shares in a coarser quantum — so ranking them
    together compares two things that were not measured the same way. The
    consistent majority publishes; the outlier does not.
    """
    for entry_id in ("djia_index", "spy_index"):
        _seed_run(board, entry_id, initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL])
    _seed_run(board, "buy_hold_djia", initial_equity=_OTHER_CAPITAL, equities=[_OTHER_CAPITAL])

    payload = lb_service.get_leaderboard()
    out = capsys.readouterr().out

    assert {e["entry_id"] for e in payload["entries"]} == {"djia_index", "spy_index"}
    assert "buy_hold_djia" in out and "not comparable with" in out


def test_the_majority_wins_even_when_the_config_agrees_with_the_minority(board):
    """The base is the board's, not the config's.

    Publishing the config's seed here would keep one entry and drop two, which
    is the empty-board failure in miniature — and it would rank nothing against
    anything.
    """
    for entry_id in ("djia_index", "spy_index"):
        _seed_run(board, entry_id, initial_equity=_OTHER_CAPITAL, equities=[_OTHER_CAPITAL])
    _seed_run(board, "buy_hold_djia", initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL])

    payload = lb_service.get_leaderboard()

    assert {e["entry_id"] for e in payload["entries"]} == {"djia_index", "spy_index"}


def test_a_tie_goes_to_the_seed_the_config_publishes(board):
    """Deterministic, rather than whichever seed the DB happened to yield first."""
    _seed_run(board, "djia_index", initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL])
    _seed_run(board, "spy_index", initial_equity=_OTHER_CAPITAL, equities=[_OTHER_CAPITAL])

    payload = lb_service.get_leaderboard()

    assert [e["entry_id"] for e in payload["entries"]] == ["djia_index"]


def test_the_seed_tolerance_is_not_exact_equality(board):
    """``100000.00000000003`` is a real stored value, not a hypothetical.

    An `==` here would split the committed board into two capital groups and
    drop eleven entries as a "minority".
    """
    source = Path(lb_service.__file__).read_text(encoding="utf-8")
    assert "_SEED_MATCH_TOLERANCE = 0.01" in source
    assert "100000.00000000003" in source, (
        "the comment naming the value that forces a tolerance must stay, or "
        "someone will tighten this to `==`"
    )


# --------------------------------------------------------------------------
# #365 — an unrecorded final equity is absent, not "finished level"
# --------------------------------------------------------------------------


def test_a_run_with_no_final_equity_publishes_null_not_starting_capital(board):
    """``portfolio_value = display_capital`` was the same defect class.

    A run that recorded no final equity was published as an account that ended
    exactly level — a number nobody measured, in the column the board ranks on.
    """
    run_id = _seed_run(
        board, "djia_index", initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL]
    )
    board.set_final_equity(run_id, None)

    entry = _entry(lb_service.get_leaderboard(), "djia_index")

    assert entry is not None, "the entry still publishes; only the value is absent"
    assert entry["portfolio_value"] is None


def test_an_absent_portfolio_value_still_ranks(board):
    """``_rank_sort_key`` falls back to ``cumulative_return`` for a None.

    Publishing null would be a regression if it made the entry unsortable.
    """
    run_id = _seed_run(
        board, "djia_index", initial_equity=_DISPLAY_CAPITAL, equities=[_DISPLAY_CAPITAL]
    )
    board.set_final_equity(run_id, None)
    _seed_run(
        board,
        "spy_index",
        initial_equity=_DISPLAY_CAPITAL,
        equities=[_DISPLAY_CAPITAL, _DISPLAY_CAPITAL * 1.05],
    )

    payload = lb_service.get_leaderboard()

    assert [e["rank"] for e in payload["entries"]] == [1, 2]


def test_both_tables_render_an_absent_portfolio_value_as_a_dash():
    """`formatLeaderboardNumber(null)` is `0.00`, so the table printed `$0.00`.

    Same conflation on the client as `or 0` was on the server, one column over.
    """
    body = _strip_comments(_LEADERBOARD_JS)
    home = _strip_comments(_HOME_JS)

    assert "formatLeaderboardMoneyOrDash(entry.portfolio_value)" in body
    assert "$${formatLeaderboardNumber(entry.portfolio_value)}" not in body
    # `Number(null)` is 0 and 0 is finite, so a bare `Number(value)` here never
    # saw the shape the guard below it was written for.
    home_fn = _extract_function(home, "homeFormatPortfolioValue")
    assert "const n = Number(value);" not in home_fn
    assert "Number.isFinite(n)" in home_fn


# --------------------------------------------------------------------------
# frontend — both readers of the changed wire format
# --------------------------------------------------------------------------


def _extract_function(source: str, name: str) -> str:
    """The source of ``function <name>(...) { ... }``, brace-matched.

    Extracted rather than restated so a rename or deletion fails these tests
    instead of leaving them green against a copy that no longer ships.
    """
    start = source.index(f"function {name}(")
    index = source.index("{", source.index(")", start))
    depth = 0
    while True:
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
        index += 1


def _run_node(*parts):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    proc = subprocess.run(
        [node, "-e", "\n".join(parts)], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _strip_comments(source: str) -> str:
    """Whole-line comments only — an inline strip would eat the tail of any URL."""
    import re

    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", source)


_CURVE_HARNESS = (
    _extract_function(_LEADERBOARD_JS, "finiteNumber"),
    _extract_function(_LEADERBOARD_JS, "chartTimeKey"),
    _extract_function(_LEADERBOARD_JS, "buildEquityCurvesFromEntries"),
    # Hoisted stub: the real one reaches a palette and module state this
    # harness has no use for.
    "function getSeriesStyle() { return { color: '#fff', dash: [], kind: 'model' }; }",
)

_TWO_ENTRIES = """
const entries = [
  { entry_id: 'a', model: 'A', initial_equity: 100000, equity_curve: [
      { timestamp: '2026-04-15T14:00', equity: 100000 },
      { timestamp: '2026-04-16T14:00', equity: %(bad)s },
      { timestamp: '2026-04-17T14:00', equity: 101000 },
  ]},
  { entry_id: 'b', model: 'B', initial_equity: 100000, equity_curve: [
      { timestamp: '2026-04-15T14:00', equity: 100000 },
      { timestamp: '2026-04-16T14:00', equity: 100500 },
      { timestamp: '2026-04-17T14:00', equity: 101200 },
  ]},
];
const built = buildEquityCurvesFromEntries(entries);
console.log(JSON.stringify({ a: built.curves.A, b: built.curves.B }));
"""


def test_app_chart_keeps_a_null_equity_null_instead_of_zeroing_it():
    """`Number(pt.equity) || 0` was the client half of #390.

    Null-filling the hour is what this function ALREADY does for a timestamp a
    series does not carry, which is why the fix is to stop coercing rather than
    to add a second rendering mode.
    """
    result = _run_node(*_CURVE_HARNESS, _TWO_ENTRIES % {"bad": "null"})

    assert result["a"] == [100000, None, 101000], (
        "the missing hour must stay null; zeroing it plots a -100% spike that "
        "sets the y-range for every series on the shared axis"
    )
    assert result["b"] == [100000, 100500, 101200]


@pytest.mark.parametrize("bad", ["null", "undefined", "''", "NaN", "'oops'"])
def test_app_chart_treats_every_unparseable_equity_as_a_gap(bad):
    """`Number.isFinite(Number(x))` alone would not do this: `Number(null)` is 0."""
    result = _run_node(*_CURVE_HARNESS, _TWO_ENTRIES % {"bad": bad})

    assert result["a"] == [100000, None, 101000], result["a"]


def test_app_chart_still_plots_a_real_zero():
    result = _run_node(*_CURVE_HARNESS, _TWO_ENTRIES % {"bad": "0"})

    assert result["a"] == [100000, 0, 101000]


def test_the_app_chart_reads_equity_through_the_landings_helper():
    """One wire format, one definition of "is this a number".

    ``finiteNumber`` is the landing's (``src/lib/leaderboard.ts``, PR #387),
    ported under the same name: two readers of one payload answering that
    question differently is how they drift.
    """
    body = _strip_comments(_LEADERBOARD_JS)
    assert "function finiteNumber(value)" in body
    assert "Number(pt.equity) || 0" not in body


def test_the_chart_tooltip_does_not_print_a_missing_hour_as_zero_dollars():
    """`(ds._raw && ds._raw[idx]) || 0` printed "$0.00" and "-100.00%"."""
    body = _strip_comments(_LEADERBOARD_JS)
    assert "finiteNumber(ds._raw ? ds._raw[idx] : null)" in body
    assert "(ds._raw && ds._raw[idx]) || 0" not in body


def test_the_marketplace_card_reads_a_null_equity_as_a_gap():
    """app.js is the second reader of this payload, and its guard was inert.

    `Number(point?.equity)` + `Number.isFinite` looks like a guard and is not:
    `Number(null)` is 0 and 0 is finite, so the single most likely malformed
    shape went straight through as a $0 account and drew the card's sparkline as
    a collapse to −100%.
    """
    result = _run_node(
        _extract_function(_APP_JS, "marketplaceIndexedPctSeries"),
        _extract_function(_APP_JS, "marketplaceLinePath"),
        """
const series = marketplaceIndexedPctSeries([
  { timestamp: 't0', equity: 100000 },
  { timestamp: 't1', equity: null },
  { timestamp: 't2', equity: 101000 },
]);
const path = marketplaceLinePath(series, (i) => i * 10, (p) => 100 - p);
console.log(JSON.stringify({ pcts: series.map((p) => p.pct), path }));
""",
    )

    assert result["pcts"][1] is None, (
        "a missing observation must not become a -100% point on the card"
    )
    assert result["pcts"][0] == pytest.approx(0.0)
    assert result["pcts"][2] == pytest.approx(1.0)
    assert result["path"].count("M") == 2, (
        f"the line must break at the gap rather than draw through it: {result['path']}"
    )


def test_the_marketplace_card_indexes_off_the_first_observed_point():
    """A leading gap must not make the base null and every later pct NaN."""
    result = _run_node(
        _extract_function(_APP_JS, "marketplaceIndexedPctSeries"),
        """
const series = marketplaceIndexedPctSeries([
  { timestamp: 't0', equity: null },
  { timestamp: 't1', equity: 100000 },
  { timestamp: 't2', equity: 101000 },
]);
console.log(JSON.stringify(series.map((p) => p.pct)));
""",
    )

    assert result[0] is None
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(1.0)


def test_a_derived_seed_that_differs_from_the_board_is_reported(board, capsys):
    """The one drift `_warn_on_capital_drift` cannot see.

    It scans rows for a *recorded* seed; this row has none, so its seed came
    from the curve's own first point and the board-level scan skips it.
    """
    _seed_run(board, "djia_index", initial_equity=None, equities=[_OTHER_CAPITAL, _OTHER_CAPITAL * 1.05])

    lb_service.get_leaderboard()
    lb_service.get_leaderboard()
    lines = [
        ln
        for ln in capsys.readouterr().out.splitlines()
        if "djia_index" in ln and "published seed" in ln
    ]

    assert len(lines) == 1, lines
    assert "#194" in lines[0]


def test_a_recorded_seed_is_not_reported_twice(board, capsys):
    """One condition, one line. `_warn_on_capital_drift` owns the recorded case."""
    _seed_run(board, "djia_index", initial_equity=_OTHER_CAPITAL, equities=[_OTHER_CAPITAL])

    lb_service.get_leaderboard()
    out = capsys.readouterr().out

    assert "not comparable" in out
    assert "published seed" not in out
