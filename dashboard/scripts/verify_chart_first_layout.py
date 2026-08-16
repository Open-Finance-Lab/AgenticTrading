"""Measure the chart-first layout on both surfaces, at the viewports that matter.

Run against a local backend (see the plan's Task 12 Step 1 for the scratch-DB
invocation):

    ~/.venvs/htmlpdf/bin/python dashboard/scripts/verify_chart_first_layout.py

Exits non-zero on the first failed assertion, printing every measurement so a
near-miss is legible rather than a bare traceback.

WHY A SCRIPT AND NOT A PYTEST CASE: this needs a running server and a real
browser, and it is a pre-merge measurement pass, not a CI gate. The values it
confirms are pinned separately by Task 11's source guards, which do run in CI.
"""

from __future__ import annotations

import json
import sys
import urllib.request

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

BASE = "http://localhost:8077"

# 1366x768 and 1280x720 are the two that falsified the first draft's heights.
# A list that only samples 900px-tall screens cannot see the bug this pass exists
# to catch. 390x844 is the stacked (below-lg) case.
VIEWPORTS = [
    (1280, 720),
    (1280, 800),
    (1366, 768),
    (1440, 768),
    (1440, 900),
    (1600, 900),
    (1920, 1080),
    (390, 844),
]

LG = 1024  # Tailwind's lg: breakpoint, where / stops stacking
PAGER_MIN = 1200  # below this /app stacks and the pager does not apply

failures: list[str] = []


def check(ok: bool, label: str, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: {detail}")
    if not ok:
        failures.append(f"{label}: {detail}")


def clamp(lo: float, preferred: float, hi: float) -> float:
    return max(lo, min(preferred, hi))


def measure_landing(page, width: int, height: int) -> None:
    # `load`, NOT `networkidle`: both surfaces poll in the background, so there is
    # no guarantee the network ever goes idle for 500ms. `networkidle` passed at
    # the first two viewports and then timed out at the third -- a wait that fails
    # on a coin-flip cannot tell a broken layout from a poll landing badly.
    page.goto(f"{BASE}/", wait_until="load")
    m = page.evaluate(
        """() => {
        const card = document.querySelector('[data-testid="board-preview"]')
            || document.querySelector('#hero .rounded-xl, header .rounded-xl')
            || document.querySelector('main .rounded-xl');
        const chartBox = card && card.querySelector('.recharts-responsive-container');
        const column = card && card.closest('div[class*="basis-2/3"], div[class*="lg:basis-2/3"]');
        const container = column && column.closest('.container');
        const chips = card && card.querySelector('.flex-nowrap');
        const r = (el) => el ? el.getBoundingClientRect() : null;
        return {
            card: r(card),
            chart: r(chartBox),
            column: r(column),
            container: r(container),
            chips: chips ? {scrollWidth: chips.scrollWidth, clientWidth: chips.clientWidth} : null,
            innerHeight: window.innerHeight,
            // getComputedStyle, never the `hidden` attribute: PR #357's clipping
            // bug was invisible to attribute probes.
            cardDisplay: card ? getComputedStyle(card).display : null,
        };
    }"""
    )

    check(m["card"] is not None, "/ card found", str(bool(m["card"])))
    if not m["card"]:
        return

    check(
        m["cardDisplay"] not in (None, "none"),
        "/ card is displayed",
        f"display={m['cardDisplay']}",
    )

    # The fold. The check the first draft lacked -- and the one that failed at
    # four viewports.
    bottom = m["card"]["y"] + m["card"]["height"]
    check(
        bottom <= m["innerHeight"] + 0.5,
        "/ card sits above the fold",
        f"bottom={bottom:.1f} innerHeight={m['innerHeight']}",
    )

    # The chart's own clamp: clamp(300px, calc(100dvh - 390px), 520px).
    # Reported as a FAILURE when the container is absent, never skipped: a
    # missing chart is the single worst outcome this pass exists to catch, and a
    # silent skip would render it as a clean run.
    check(m["chart"] is not None, "/ chart container found", str(bool(m["chart"])))
    if m["chart"]:
        expected = clamp(300.0, height - 390.0, 520.0)
        actual = m["chart"]["height"]
        check(
            abs(actual - expected) <= 2.0,
            "/ chart height matches its clamp",
            f"actual={actual:.1f} expected={expected:.1f}",
        )

    # Column width. THE DENOMINATOR IS THE CONTAINER, NOT THE VIEWPORT: this
    # same layout is 66.7% of the container but only 63.0-65.9% of the viewport,
    # so a guard that quietly switched denominators would sit within 3pp of its
    # own threshold. Guarded at 60% -- below the 2/3 target so gutters and
    # rounding cannot redden a correct layout, above 50% so a reverted split
    # still fails.
    if width >= LG and m["column"] and m["container"]:
        ratio = m["column"]["width"] / m["container"]["width"]
        check(
            ratio >= 0.60,
            "/ chart column >= 60% OF THE CONTAINER",
            f"ratio={ratio:.3f} column={m['column']['width']:.0f}"
            f" container={m['container']['width']:.0f}",
        )

    # Five chips, one row. Checked at 1440 specifically -- the width the strip
    # was designed against.
    if width == 1440 and m["chips"]:
        check(
            m["chips"]["scrollWidth"] <= m["chips"]["clientWidth"] + 1,
            "/ chip strip fits on one row",
            f"scrollWidth={m['chips']['scrollWidth']} clientWidth={m['chips']['clientWidth']}",
        )


def measure_app(page, width: int, height: int) -> None:
    page.goto(f"{BASE}/app", wait_until="load")
    # Wait on the module's own loading state, not a wall-clock guess: the
    # leaderboard arrives over fetch, so a fixed sleep is either wasteful or too
    # short.
    #
    # THE CONDITION IS "THE PLACEHOLDER CLEARED", NOT "THE LIST HAS CHILDREN".
    # app.html ships `<li class="home-module-rank-empty">Loading the
    # standings...</li>` as STATIC markup, so a children-count wait is satisfied
    # by the served HTML before a line of JS runs -- it returns instantly and
    # every measurement below reads the pre-fetch page. That produced a clean
    # sweep of /app failures that looked like a broken layout.
    #
    # This waits for a PRECONDITION (the module finished loading), never for the
    # postconditions being asserted (row count, chart height, swatch colours) --
    # otherwise the wait would guarantee its own assertions. A timeout is NOT
    # swallowed: the checks below then report the still-loading page as
    # failures, so "the API is down" stays distinguishable from "the layout is
    # wrong" instead of both rendering as a hang.
    _wait_for_module(page)
    m = page.evaluate(
        """() => {
        const screen = document.querySelector('#homeScreenLanding');
        const wrap = document.querySelector('.hm-rank-chart');
        const canvas = document.querySelector('#homeModuleRankChart');
        const list = document.querySelector('#homeModuleRankList');
        const rows = list ? Array.from(list.children) : [];
        const listBox = list ? list.getBoundingClientRect() : null;
        const chart = (canvas && window.Chart && window.Chart.getChart)
            ? window.Chart.getChart(canvas) : null;
        return {
            screen: screen
                ? {scrollHeight: screen.scrollHeight, clientHeight: screen.clientHeight}
                : null,
            chartHeight: wrap ? wrap.getBoundingClientRect().height : null,
            chartDisplay: wrap ? getComputedStyle(wrap).display : null,
            rowCount: rows.length,
            rowsInside: listBox
                ? rows.filter(r => r.getBoundingClientRect().bottom
                    <= listBox.bottom + 0.5).length
                : 0,
            rowBadges: rows.map(r => (r.textContent || '').includes('Baseline')),
            // The swatch must resolve to a real colour, and to the SAME colour
            // as its curve. A transparent swatch is the documented degraded
            // state (getSeriesStyle missing) and must not pass silently.
            rowSwatches: rows.map(r => {
                const name = r.querySelector('.home-module-rank-name');
                const sw = r.querySelector('.hm-rank-swatch');
                return {
                    label: name ? (name.textContent || '').trim() : null,
                    color: sw ? getComputedStyle(sw).backgroundColor : null,
                };
            }),
            datasets: chart
                ? chart.data.datasets.map(d => ({
                    label: d.label,
                    dash: (d.borderDash || []).length,
                    color: d.borderColor,
                  }))
                : null,
        };
    }"""
    )

    # The pager clips with overflow:hidden and NO scrollbar, so this is the only
    # way to see it. A height assertion on the panel alone cannot.
    if width >= PAGER_MIN and m["screen"]:
        overflow = m["screen"]["scrollHeight"] - m["screen"]["clientHeight"]
        check(
            overflow <= 1,
            "/app screen 0 does not clip",
            f"scrollHeight-clientHeight={overflow}",
        )

    check(
        m["chartHeight"] is not None,
        "/app chart wrapper found",
        str(m["chartHeight"] is not None),
    )
    if m["chartHeight"] is not None:
        expected = clamp(140.0, height * 0.26, 280.0)
        check(
            abs(m["chartHeight"] - expected) <= 2.0,
            "/app chart height matches its clamp",
            f"actual={m['chartHeight']:.1f} expected={expected:.1f}",
        )
        check(
            m["chartDisplay"] != "none",
            "/app chart is displayed",
            f"display={m['chartDisplay']}",
        )

    check(
        m["rowCount"] == 7,
        "/app renders all 7 models",
        f"rowCount={m['rowCount']}",
    )
    check(
        m["rowsInside"] == m["rowCount"],
        "/app every row is inside the list's visible box",
        f"{m['rowsInside']}/{m['rowCount']} visible",
    )
    # The list stays models-only, which is what keeps app.html's pinned
    # "AI models only - ranked by return" literally true.
    check(
        not any(m["rowBadges"]),
        "/app rank list carries no baseline rows",
        f"baseline rows={sum(m['rowBadges'])}",
    )

    # The swatch is the chart's ONLY key, so a blank or duplicated one is the
    # same failure as an unlabelled legend. Source-shape guards cannot see this:
    # they read the template string, not the resolved colour.
    swatches = [s for s in m["rowSwatches"] if s["label"]]
    if swatches:
        blank = [
            s["label"]
            for s in swatches
            if not s["color"] or "rgba(0, 0, 0, 0)" in s["color"]
        ]
        check(
            not blank,
            "/app every rank row has a resolved swatch colour",
            f"transparent={blank}" if blank else "all resolved",
        )
        colours = [s["color"] for s in swatches]
        check(
            len(set(colours)) == len(colours),
            "/app swatch colours are unique per row",
            f"{len(set(colours))} distinct across {len(colours)} rows",
        )

    # Likewise a hard check. `Chart.getChart(canvas)` returning undefined means
    # the chart never instantiated -- the documented degraded state, and exactly
    # what an `is not None` skip would wave through.
    check(
        m["datasets"] is not None,
        "/app Chart.js instance is live on the canvas",
        f"datasets={len(m['datasets']) if m['datasets'] else 'none'}",
    )
    if m["datasets"] is not None:
        labels = {d["label"]: d for d in m["datasets"]}
        for name in ("Buy & Hold", "DJIA"):
            present = name in labels
            check(present, f"/app chart carries the {name} baseline", str(present))
            if present:
                check(
                    labels[name]["dash"] > 0,
                    f"/app {name} is dashed",
                    f"borderDash length={labels[name]['dash']}",
                )

        # A row's swatch pointing at a different colour than its own curve is
        # worse than no swatch. Both sides read `getSeriesStyle`; this confirms
        # they actually agree once rendered.
        def _rgb(value: str) -> tuple[int, int, int] | None:
            if not value:
                return None
            if value.startswith("#") and len(value) == 7:
                return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))
            nums = [
                int(float(n))
                for n in value.replace("rgba(", "")
                .replace("rgb(", "")
                .rstrip(")")
                .split(",")[:3]
            ]
            return tuple(nums) if len(nums) == 3 else None

        mismatched = [
            s["label"]
            for s in swatches
            if s["label"] in labels
            and _rgb(s["color"]) != _rgb(labels[s["label"]]["color"])
        ]
        check(
            not mismatched,
            "/app each row's swatch matches its own curve colour",
            f"mismatched={mismatched}" if mismatched else "all match",
        )


def _wait_for_module(page) -> None:
    """Block until the leaderboard module has replaced its static placeholder."""
    try:
        page.wait_for_function(
            """() => {
                const list = document.querySelector('#homeModuleRankList');
                return !!list && list.children.length > 0
                    && !list.querySelector('.home-module-rank-empty');
            }""",
            timeout=20000,
        )
    except PlaywrightTimeoutError:
        print("  [warn] rank list still showing its placeholder; measuring as-is")


def measure_fallbacks(browser) -> None:
    """Force the three no-chart states and confirm none of them draws a chart.

    THE FIXTURES ARE THE REAL PAYLOAD, MUTATED -- never hand-written from the
    field names this repo's own frontend reads. A fixture built that way tests
    the consumer against itself and stays green through a producer rename; the
    news-adapter outage in CLAUDE.md is the standing example.

    The third state is the one a click-through pass skips. `unreachable` and
    `empty` both take the `sample` branch and return before the chart call, so
    they cannot distinguish "no chart because we bailed early" from "no chart
    because the series were empty". Only real entries carrying empty
    `equity_curve`s reach `renderHomeLeaderboardChart` and exercise its guard.
    """
    with urllib.request.urlopen(f"{BASE}/api/v1/leaderboard", timeout=180) as resp:
        real = json.loads(resp.read().decode())
    real_models = [e for e in real.get("entries", []) if e.get("is_model")]
    print(f"\n=== fallback states (real payload: {len(real_models)} model entries) ===")

    cases = [
        ("unreachable", None, True),
        ("empty", {**real, "entries": []}, True),
        (
            "curveless",
            {**real, "entries": [{**e, "equity_curve": []} for e in real.get("entries", [])]},
            False,
        ),
    ]

    # A FACTORY, not `def handler(route, _body=body)`. Playwright inspects the
    # handler's arity and passes `(route, request)` to any two-parameter
    # callable, so the default-argument closure idiom silently receives a
    # Request where the body belongs.
    def make_handler(body):
        def handler(route):
            if body is None:
                route.abort()
            else:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(body),
                )

        return handler

    for name, body, expect_sample in cases:
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.route("**/api/v1/leaderboard*", make_handler(body))
        page.goto(f"{BASE}/app", wait_until="load")
        _wait_for_module(page)
        page.wait_for_timeout(800)  # the chart call is synchronous after render
        m = page.evaluate(
            """() => {
            const list = document.querySelector('#homeModuleRankList');
            const note = document.getElementById('homeModuleRankSample');
            return {
                hasChart: !!document.getElementById('homeModuleRankChartWrap'),
                hasCanvas: !!document.querySelector('#homeModuleRankChart'),
                rows: list ? list.children.length : -1,
                noteVisible: !!note && !note.hidden,
                noteText: note ? (note.textContent || '').trim().slice(0, 48) : null,
            };
        }"""
        )
        print(f"  -- {name}")
        # The whole point of the design: no series means NO ELEMENT, because a
        # blank reserved box reads as a chart that failed.
        check(not m["hasChart"], f"/app [{name}] draws no chart wrapper", str(m))
        check(not m["hasCanvas"], f"/app [{name}] leaves no orphan canvas", str(m["hasCanvas"]))
        check(
            m["noteVisible"] == expect_sample,
            f"/app [{name}] sample note visible == {expect_sample}",
            f"visible={m['noteVisible']} text={m['noteText']!r}",
        )
        if not expect_sample:
            # Real rows, not the five-row mock: this state must stay legible as
            # "the board is fine, the curves are missing".
            check(
                m["rows"] == len(real_models),
                f"/app [{name}] still lists the real model rows",
                f"rows={m['rows']} expected={len(real_models)}",
            )
        page.close()


def main() -> int:
    mode = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "all"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        if mode in ("all", "layout"):
            run_viewport_sweep(browser)
        if mode in ("all", "fallbacks"):
            measure_fallbacks(browser)
        browser.close()

    print(f"\n{'-' * 60}")
    if failures:
        print(f"{len(failures)} FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all measurements pass")
    return 0


def run_viewport_sweep(browser) -> None:
    for width, height in VIEWPORTS:
        page = browser.new_page(viewport={"width": width, "height": height})
        print(f"\n=== {width}x{height} ===")
        # Contained per surface. An earlier run aborted mid-sweep on a
        # navigation timeout and lost the five viewports behind it -- which
        # reads as "we measured nothing" but LOOKS like a crash in the page.
        # A surface that blows up is one recorded FAIL, not a lost pass.
        for label, measure in (("/", measure_landing), ("/app", measure_app)):
            try:
                measure(page, width, height)
            except Exception as exc:  # noqa: BLE001 - report, don't abort
                check(False, f"{label} measured at {width}x{height}", repr(exc))
        page.close()


if __name__ == "__main__":
    sys.exit(main())
