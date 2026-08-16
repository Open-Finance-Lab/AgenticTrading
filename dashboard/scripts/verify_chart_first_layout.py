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
#
# 1024x900 and 1152x864 cover the rest of Tailwind's `lg` band. / stops stacking
# at 1024 but the hero's two columns are at their narrowest there, and a sweep
# that jumped 390 -> 1280 measured neither end of that band: the chip strip --
# the chart's only legend -- was clipping one to two of its five entries across
# the whole range, invisibly, because the only guard on it ran at 1440.
#
# 1201x760 and 1240x700 sit just above /app's 1200px pager threshold, the band
# where screen 0 has the least height and still clips rather than scrolls.
VIEWPORTS = [
    (1024, 900),
    (1152, 864),
    (1201, 760),
    (1240, 700),
    (1280, 720),
    (1280, 800),
    (1366, 768),
    (1440, 768),
    (1440, 900),
    (1600, 900),
    (1920, 1080),
    (390, 844),
    # The second phone size: the chip strip wraps to four rows here and five at
    # 390, so the stacked reserve has to clear the taller of the two.
    (414, 896),
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
    # WAIT OUT THE ENTRANCE ANIMATION, and key on the TRANSFORM, not on height.
    #
    # The hero card mounts under framer-motion with `initial={{scale: 0.95}}`,
    # delay 0.3s, duration 0.7s. getBoundingClientRect() reports the TRANSFORMED
    # box, so every measurement taken mid-animation is scaled: 494.0 against a
    # 520px clamp, 313.5 against 330, 431.3 against 454 -- all of them exactly
    # 0.95x, a uniform 5% error that reads as a subtle CSS discrepancy rather
    # than an instrumentation bug.
    #
    # A "has the height stopped changing" check does NOT catch it: during the
    # 0.3s delay the scale sits constant at 0.95, so two consecutive polls agree
    # and the wait exits on a stationary WRONG value. Only the transform reaching
    # identity actually means the animation finished.
    try:
        page.wait_for_function(
            """() => {
                const rc = document.querySelector('.recharts-responsive-container');
                if (!rc) return false;
                if (rc.getBoundingClientRect().height <= 0) return false;
                for (let el = rc; el && el !== document.body; el = el.parentElement) {
                    const t = getComputedStyle(el).transform;
                    if (t && t !== 'none' && t !== 'matrix(1, 0, 0, 1, 0, 0)') return false;
                }
                return true;
            }""",
            timeout=15000,
        )
    except PlaywrightTimeoutError:
        print("  [warn] landing entrance animation never settled; measuring as-is")
    m = page.evaluate(
        """() => {
        const card = document.querySelector('[data-testid="board-preview"]')
            || document.querySelector('#hero .rounded-xl, header .rounded-xl')
            || document.querySelector('main .rounded-xl');
        // THE WRAPPER, NOT THE RECHARTS CONTAINER. The clamp is an inline
        // style on the wrapper; the container inside it is shorter by the
        // wrapper's `pt-4` (16px). Measuring the child and comparing it to the
        // clamp bakes that padding into the expected value invisibly, so the
        // guard reads 16px low forever and would redden on a padding change
        // that left the clamp perfectly honoured.
        const rc = card && card.querySelector('.recharts-responsive-container');
        const chartBox = rc && rc.parentElement;
        const column = card && card.closest('div[class*="basis-2/3"], div[class*="lg:basis-2/3"]');
        const container = column && column.closest('.container');
        // A testid, not `.flex-nowrap`. The class was the thing under test, so
        // the selector went stale the moment the strip was fixed -- and a
        // querySelector that finds nothing makes this check SKIP, which prints
        // as a clean run.
        const chips = card && card.querySelector('[data-testid="board-chip-strip"]');
        const r = (el) => el ? el.getBoundingClientRect() : null;
        const stripBox = chips ? chips.getBoundingClientRect() : null;
        return {
            card: r(card),
            chart: r(chartBox),
            column: r(column),
            container: r(container),
            // Per-chip containment, not scrollWidth. Once the strip wraps it
            // can never overflow horizontally, so the old scrollWidth test
            // would pass forever regardless of what is visible; what matters is
            // that all five entries are inside the box.
            chips: chips ? {
                total: chips.children.length,
                inside: Array.from(chips.children).filter(c => {
                    const b = c.getBoundingClientRect();
                    return b.width > 0 && b.height > 0
                        && b.left >= stripBox.left - 0.5 && b.right <= stripBox.right + 0.5
                        && b.top >= stripBox.top - 0.5 && b.bottom <= stripBox.bottom + 0.5;
                }).length,
            } : null,
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

    # The chart's own clamp. The reserve is breakpoint-dependent -- 390px beside
    # the copy, 590px stacked -- because the card's non-chart height is 218-241px
    # at lg+ and 443px at phone width, where the title, chip and caption wrap and
    # the chip strip runs to five rows.
    reserve = 390.0 if width >= LG else 590.0
    # Reported as a FAILURE when the container is absent, never skipped: a
    # missing chart is the single worst outcome this pass exists to catch, and a
    # silent skip would render it as a clean run.
    check(m["chart"] is not None, "/ chart container found", str(bool(m["chart"])))
    if m["chart"]:
        expected = clamp(260.0, height - reserve, 520.0)
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

    # EVERY viewport, not just 1440. This ran at one width -- the one the strip
    # was designed against -- so it could not see that the strip was dropping
    # one to four of its five chips across the whole lg band and every phone.
    # The strip is the chart's only legend; a missing chip is a drawn curve
    # nothing on the page names.
    check(m["chips"] is not None, "/ chip strip found", str(bool(m["chips"])))
    if m["chips"]:
        check(
            m["chips"]["inside"] == m["chips"]["total"] and m["chips"]["total"] == 5,
            "/ every chip is visible inside the strip",
            f"{m['chips']['inside']}/{m['chips']['total']} inside",
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
        // THE HERO, NOT THE SCREEN. `.home-landing-hero` is the element that
        // carries `overflow: hidden` above 1200px, and it is `height: 100%` of
        // the screen -- so it absorbs its own overflow and #homeScreenLanding's
        // scrollHeight NEVER exceeds its clientHeight no matter how far the
        // panel overruns. Probing the screen reported a clean 0 while the panel
        // header and footer button were off-screen; that is how the clip
        // shipped through a measurement pass.
        const hero = document.querySelector('.home-landing-hero');
        const panel = document.querySelector('#homeModuleRanking');
        const wrap = document.querySelector('.hm-rank-chart');
        const canvas = document.querySelector('#homeModuleRankChart');
        const list = document.querySelector('#homeModuleRankList');
        const rows = list ? Array.from(list.children) : [];
        const listBox = list ? list.getBoundingClientRect() : null;
        const heroBox = hero ? hero.getBoundingClientRect() : null;
        const panelBox = panel ? panel.getBoundingClientRect() : null;
        const chart = (canvas && window.Chart && window.Chart.getChart)
            ? window.Chart.getChart(canvas) : null;
        return {
            screen: screen
                ? {scrollHeight: screen.scrollHeight, clientHeight: screen.clientHeight}
                : null,
            hero: hero
                ? {scrollHeight: hero.scrollHeight, clientHeight: hero.clientHeight,
                   overflowY: getComputedStyle(hero).overflowY}
                : null,
            // Signed, and reported both ways: a panel cut at the TOP is the
            // half a bottom-edge check misses, and this one was cut at both.
            panelAboveHero: (heroBox && panelBox) ? heroBox.top - panelBox.top : null,
            panelBelowHero: (heroBox && panelBox) ? panelBox.bottom - heroBox.bottom : null,
            chartHeight: wrap ? wrap.getBoundingClientRect().height : null,
            chartDisplay: wrap ? getComputedStyle(wrap).display : null,
            rowCount: rows.length,
            // Rows fully inside the list's own visible box. NOT a clip check --
            // the list is `overflow-y: auto`, so rows below its fold are
            // scrolled, not lost, and when the panel was overrunning the hero
            // this counted 7/7 for a list that was itself off-screen. It is a
            // FLOOR: a panel that squeezes the standings to one row has
            // defeated the screen even with nothing clipped.
            rowsInside: listBox
                ? rows.filter(r => r.getBoundingClientRect().bottom
                    <= listBox.bottom + 0.5).length
                : 0,
            // The rows that are not visible have to be reachable.
            listScrolls: list
                ? (list.scrollHeight <= list.clientHeight + 1
                   || ['auto', 'scroll'].includes(getComputedStyle(list).overflowY))
                : null,
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
    if width >= PAGER_MIN and m["hero"]:
        overflow = m["hero"]["scrollHeight"] - m["hero"]["clientHeight"]
        check(
            overflow <= 1,
            "/app screen 0 does not clip",
            f"hero scrollHeight-clientHeight={overflow}"
            f" (screen reports {m['screen']['scrollHeight'] - m['screen']['clientHeight']}"
            " -- the screen cannot see this)",
        )
        # The containment check, kept separate. An ancestor's scrollHeight is a
        # summary; these two name WHICH edge, and the top edge is the one a
        # bottom-only check waves through.
        check(
            m["panelAboveHero"] is not None and m["panelAboveHero"] <= 0.5,
            "/app the panel's top edge is inside the hero",
            f"cut by {m['panelAboveHero']:.1f}px"
            if m["panelAboveHero"] and m["panelAboveHero"] > 0.5
            else "inside",
        )
        check(
            m["panelBelowHero"] is not None and m["panelBelowHero"] <= 0.5,
            "/app the panel's bottom edge is inside the hero",
            f"cut by {m['panelBelowHero']:.1f}px"
            if m["panelBelowHero"] and m["panelBelowHero"] > 0.5
            else "inside",
        )

    check(
        m["chartHeight"] is not None,
        "/app chart wrapper found",
        str(m["chartHeight"] is not None),
    )
    if m["chartHeight"] is not None:
        # A RANGE, not equality. The chart is `flex: 0 1 auto` with a 132px
        # floor, so under a deficit -- 509px of panel against 637px of content
        # at 1240x700 -- it legitimately renders shorter than its clamp; that
        # is what stops the standings collapsing to one row. Equality here
        # would redden on the correct layout and, worse, was only satisfiable
        # before because the panel was overrunning the hero. What the clamp
        # still fixes is the CEILING.
        ceiling = clamp(140.0, height * 0.26, 280.0)
        check(
            132.0 - 0.5 <= m["chartHeight"] <= ceiling + 2.0,
            "/app chart height is within its clamp and above its floor",
            f"actual={m['chartHeight']:.1f} floor=132.0 ceiling={ceiling:.1f}",
        )
        # One anchor where there IS surplus, so a clamp quietly replaced by a
        # smaller formula cannot hide behind the range above.
        if (width, height) == (1920, 1080):
            check(
                abs(m["chartHeight"] - ceiling) <= 2.0,
                "/app chart takes its full clamp where the panel has room",
                f"actual={m['chartHeight']:.1f} expected={ceiling:.1f}",
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
    # A FLOOR, not equality. Seven rows plus the chart plus 253px of panel
    # chrome need 637px, and the hero row is 509px at 1240x700 -- so at short
    # viewports some rows are below the list's fold by design, which is what its
    # `overflow-y: auto` is for. Equality here only ever passed because the
    # panel was overrunning the hero: every row really was inside the LIST, and
    # the list was outside the SCREEN. What matters is that the standings are
    # not squeezed to nothing and that the remainder is reachable.
    check(
        m["rowsInside"] >= min(3, m["rowCount"]),
        "/app at least three standings rows are visible",
        f"{m['rowsInside']}/{m['rowCount']} visible",
    )
    check(
        bool(m["listScrolls"]),
        "/app rows below the list's fold are scrollable, not lost",
        f"listScrolls={m['listScrolls']}",
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

    measure_stale_chart(browser, len(real_models))


def measure_stale_chart(browser, model_count: int) -> None:
    """The state a fresh page per case CANNOT reach: a failure AFTER a success.

    Every case above opens a new page, so the panel has never drawn a chart when
    the fallback runs -- which means they confirm "no chart is created" and say
    nothing about "an existing chart is removed". Those are different code paths
    and only the second one is exercised in production: `onHomePageShow` calls
    `refreshHomeModules()` on every return to Home, and an IntersectionObserver
    calls it again, so the first failed refresh of a session always lands on a
    panel with nine real curves already on it.

    Left standing, those curves sat above five INVENTED sample rows, and because
    the mock roster is a different set of models each row's swatch keyed a
    different model's line than the one it named.
    """
    print("\n=== stale chart (success, then a failed refresh) ===")
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"{BASE}/app", wait_until="load")
    _wait_for_module(page)
    page.wait_for_timeout(800)
    before = page.evaluate(
        """() => {
        const c = document.querySelector('#homeModuleRankChart');
        const chart = (c && window.Chart && window.Chart.getChart)
            ? window.Chart.getChart(c) : null;
        return {hasChart: !!document.getElementById('homeModuleRankChartWrap'),
                datasets: chart ? chart.data.datasets.length : 0};
    }"""
    )
    # A precondition, asserted: if the chart never drew, everything below passes
    # for the wrong reason.
    check(
        before["hasChart"] and before["datasets"] > 0,
        "/app [stale] a real chart is on screen before the failed refresh",
        f"hasChart={before['hasChart']} datasets={before['datasets']}",
    )

    page.route("**/api/v1/leaderboard*", lambda route: route.abort())
    page.evaluate("() => window.refreshHomeModules && window.refreshHomeModules()")
    page.wait_for_timeout(2500)
    after = page.evaluate(
        """() => {
        const note = document.getElementById('homeModuleRankSample');
        const list = document.querySelector('#homeModuleRankList');
        return {hasChart: !!document.getElementById('homeModuleRankChartWrap'),
                hasCanvas: !!document.querySelector('#homeModuleRankChart'),
                noteVisible: !!note && !note.hidden,
                rows: list ? list.children.length : -1};
    }"""
    )
    check(
        not after["hasChart"] and not after["hasCanvas"],
        "/app [stale] the previous chart is torn down, not left standing",
        str(after),
    )
    check(
        after["noteVisible"] and after["rows"] != model_count,
        "/app [stale] the standings fall back and say so",
        f"noteVisible={after['noteVisible']} rows={after['rows']}",
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
