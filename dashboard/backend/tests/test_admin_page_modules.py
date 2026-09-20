"""Source-shape pins shared by the four /admin modules (design §7.4, §7.6, §13 row C).

Harvested from test_admin_analytics_frontend.py's `test_client_uses_exact_pr2_
endpoints_and_query_names` and `test_analytics_is_read_only_and_uses_safe_dom_
rendering`, re-targeted at js/admin-*.js, plus the pins the design's "Must not"
column for PR C asks for: no #live route, no Chart.js, no client-side group
badge, and none of the D15 display fields read.
"""

import re
from pathlib import Path

from dashboard.backend.domain.user_groups import (
    DEFAULT_USER_GROUP,
    USER_GROUP_LABELS,
    USER_GROUPS,
)
from dashboard.backend.tests._frontend_source import fn_body, strip_comments

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
# Design N3: the read-only property is kept and its subject is narrowed. The
# analytics modules stay pinned GET-only and free of credential field names.
# Write modules are enumerated by name with their EXACT permitted verb set --
# exact, not a subset, so adding a verb is a test failure rather than a silent
# pass. Deleting the verb list and dropping api_key from the prohibited names
# would have left two tests that run, pass, and protect nothing.
READ_MODULES = ("admin-shell.js", "admin-live.js", "admin-overview.js", "admin-users.js")
WRITE_MODULES = {
    "admin-providers.js": {"PUT", "POST", "DELETE"},
    # The absorbed credits console mutates groups, roles, grants and the pool.
    "admin-credits.js": {"PATCH", "POST"},
}
NAMES = READ_MODULES + tuple(WRITE_MODULES)
MODULES = {name: (FRONTEND / "js" / name).read_text(encoding="utf-8") for name in NAMES}
# The write-module scans below run on the stripped copy. A write module's header
# explains the rules it obeys, and every one of those sentences names a
# credential -- `strip_comments` is what keeps the guard asserting about the
# code rather than about the prose describing the code.
CODE = {name: strip_comments(source) for name, source in MODULES.items()}
READS = {name: MODULES[name] for name in READ_MODULES}
ALL = "\n".join(READS.values())
ADMIN_HTML = (FRONTEND / "admin.html").read_text(encoding="utf-8")
GLOBALS = {
    "admin-shell.js": "AdminShell",
    "admin-live.js": "AdminLive",
    "admin-overview.js": "AdminOverview",
    "admin-users.js": "AdminUsers",
    "admin-providers.js": "AdminProviders",
    "admin-credits.js": "AdminCredits",
}

CREDENTIAL = re.compile(r"\w*(?:api_key|secret|credential|token|password)\w*", re.I)
SINK_ASSIGN = re.compile(r"\.(?:textContent|innerHTML|value)\s*=\s*(?P<rhs>[^;]+);")
SINK_CALL = re.compile(r"\.(?:setAttribute|append|appendChild|replaceChildren|createTextNode)\s*\(")
EMPTY_RHS = {"''", '""', "``"}


def test_each_module_is_an_iife_exposing_exactly_one_global():
    for name, source in MODULES.items():
        assert source.lstrip().startswith("/**"), name
        assert "(function () {\n  'use strict';" in source, name
        assigned = set(re.findall(r"^\s*window\.(\w+) = ", source, re.M))
        assert assigned == {GLOBALS[name]}, (name, assigned)


def test_every_module_admin_html_loads_exists_and_nothing_else_is_loaded():
    srcs = re.findall(r'<script src="js/([^?"]+)\?v=\d+" defer></script>', ADMIN_HTML)
    assert set(srcs) == set(NAMES) | {"credit-format.js"} | {"admin-ticker.js"}
    for name in srcs:
        assert (FRONTEND / "js" / name).exists(), name


def test_rendering_is_text_content_only():
    for name, source in MODULES.items():
        for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function("):
            assert forbidden not in source, (name, forbidden)
        assert "textContent" in source, name


def test_every_request_is_a_credentialed_get_except_the_shells_own_write_path():
    """Every request through request()/gate() is a GET. The one exception is
    logout() (design §7, the 09-17 nav consolidation), and it does not issue a
    fetch of its own -- it goes through write(), the shell's single
    CSRF-bearing write path (N4), which is exactly the seam this test's other
    assertions exist to keep singular.

    Rescoped to READS (design N3): admin-providers.js is this page's first real
    write module and asserts its own exact verb set below
    (test_write_modules_declare_their_exact_verbs). Folding it into ALL here
    would blur "read module" back into "any module on the page", which is
    exactly the property this rescope exists to keep narrow rather than widen."""
    assert set(re.findall(r"method:\s*'(\w+)'", ALL)) == {"GET", "POST"}
    assert ALL.count("method: 'POST'") == 1
    assert "write('/api/auth/logout', { method: 'POST' })" in MODULES["admin-shell.js"]
    for name, source in READS.items():
        if name == "admin-shell.js":
            # request(), gate() and write() -- the third is the page's single
            # write path (N4) and is deliberately a separate named function so
            # this file can tell readers from writers by grep.
            assert source.count("fetch(") == 3
            assert "credentials: 'include'" in source
        else:
            assert "fetch(" not in source, name
            assert "XMLHttpRequest" not in source, name
            # A read module that reaches for the write path is a read module no
            # longer, and nothing else in this file would notice.
            assert "AdminShell.write" not in source, name
            assert "shell().write" not in source, name
    for verb in ("PATCH", "PUT", "DELETE"):
        assert f"method: '{verb}'" not in ALL


def test_only_the_shell_owns_fetch():
    for name, code in CODE.items():
        if name == "admin-shell.js":
            continue
        assert "fetch(" not in code, name
        assert "XMLHttpRequest" not in code, name
    # request() stays GET-hardcoded and gains no options bag (N4). The exact
    # signature, not just the name: `request(path, options = {})` would satisfy
    # a substring check while being precisely the change N4 forbids.
    assert "async function request(path) {" in CODE["admin-shell.js"]
    assert "method: 'GET'" in CODE["admin-shell.js"]


def test_write_modules_declare_their_exact_verbs():
    """Exact, not a subset: a module that grows a verb has grown a capability,
    and the point of enumerating them is that the growth is visible here."""
    for name, permitted in WRITE_MODULES.items():
        code = CODE[name]
        found = set(re.findall(r"method:\s*'(\w+)'", code))
        assert found == permitted, (name, found, permitted)
        assert "AdminShell" in code or "shell()" in code, name


def test_credential_names_in_a_write_module_appear_only_as_request_body_keys():
    for name in WRITE_MODULES:
        code = CODE[name]
        occurrences = [line for line in code.splitlines() if "api_key" in line]
        if name == "admin-credits.js":
            # Grants move Credits, never credentials: there is no api_key line
            # to police, and the prohibition below still applies.
            assert occurrences == [], (name, occurrences)
            continue
        assert len(occurrences) == 1, (name, occurrences)
        assert "JSON.stringify(" in occurrences[0], (name, occurrences[0].strip())
        for prohibited in (
            "session_id", "network_hash", "provider_response_body",
            "credential_ciphertext", "prompt", "strategy", "portfolio", "raw_user_agent",
        ):
            assert prohibited not in code, (name, prohibited)
        assert "localStorage" not in code, name
        assert "sessionStorage" not in code, name


def test_no_credential_field_reaches_a_dom_sink():
    """The pin a write surface needs and a read surface never did. A line may
    name a credential, or it may write to the DOM; not both. Clearing a field
    (`= ''`) is exempt -- that is the control, not the leak."""
    for name in WRITE_MODULES:
        for number, line in enumerate(CODE[name].splitlines(), start=1):
            if not CREDENTIAL.search(line):
                continue
            assigned = SINK_ASSIGN.search(line)
            if assigned:
                assert assigned.group("rhs").strip() in EMPTY_RHS, (name, number, line.strip())
            assert not SINK_CALL.search(line), (name, number, line.strip())


def test_the_write_path_carries_the_csrf_double_submit_header():
    """CsrfMiddleware 403s an unsafe method that carries a session cookie without
    a matching X-CSRF-Token. A fetch-stubbed behaviour test cannot see that, so
    the header is pinned in source too."""
    shell = MODULES["admin-shell.js"]
    assert "X-CSRF-Token" in shell
    assert "__Host-atl_csrf" in shell and "'atl_csrf'" in shell
    assert "document.cookie" in shell


def test_exact_endpoints_and_query_names():
    assert "/api/auth/me" in MODULES["admin-shell.js"]
    assert "/api/admin/stats" in MODULES["admin-live.js"]
    for endpoint in (
        "/api/admin/analytics/overview", "/api/admin/analytics/lifecycle", "/api/admin/analytics/retention",
        "/api/admin/analytics/commercial", "/api/admin/analytics/operational", "/api/admin/analytics/groups",
    ):
        assert endpoint in MODULES["admin-overview.js"], endpoint
    assert "/api/admin/analytics/users" in MODULES["admin-users.js"]
    assert "/activity?" in MODULES["admin-users.js"]
    for query_name in (
        "'from'", "'to'", "'include_internal'", "'user_group'", "'lifecycle_segment'",
        "'commercial_tier'", "'q'", "'priority'", "'limit'", "'offset'", "cursor",
    ):
        assert query_name in ALL, query_name
    assert "section, limit:" in MODULES["admin-users.js"]
    for rejected in ("'status'", "start_date", "provider_id'", "model_id'", "'cohort'", "analyticsUser", "adminTab=analytics"):
        assert rejected not in ALL, rejected


def test_no_live_route_and_no_chartjs():
    assert "#live" not in ALL and "#live" not in ADMIN_HTML
    routes_line = re.search(r"const ROUTES = \[([^\]]+)\];", MODULES["admin-shell.js"]).group(1)
    assert "'live'" not in routes_line
    assert "'usage'" not in routes_line and "'revenue'" not in routes_line and "'profiles'" not in routes_line
    for forbidden in ("window.Chart", "new Chart(", "chart.js", "cdn.jsdelivr.net"):
        assert forbidden not in ALL, forbidden
        assert forbidden not in ADMIN_HTML, forbidden
    assert "createElementNS" in MODULES["admin-overview.js"]  # the revenue line is generated SVG


def test_group_badge_is_never_computed_client_side():
    """D11: `group_badge` is rendered verbatim; the precedence rule has one owner, the server."""
    assert "group_badge" in MODULES["admin-users.js"]
    for name in ("admin-live.js", "admin-overview.js", "admin-users.js"):
        source = MODULES[name]
        for forbidden in ("role === 'admin'", "=== 'unpaid'", "'paid'", "'free'", "user_group ===", "!== 'unknown'"):
            assert forbidden not in source, (name, forbidden)
    # The shell's gate is the one place `role` is compared, and only against 'admin' for the redirect.
    assert MODULES["admin-shell.js"].count("account.role !== 'admin'") == 1


def test_d15_display_fields_are_not_read():
    for field in ("country_code", "device_category", "browser_family", "top_product_page"):
        assert field not in ALL, field


def test_prohibited_field_names_and_local_storage_are_absent():
    for prohibited in (
        "api_key", "session_id", "network_hash", "provider_response_body",
        "credential_ciphertext", "prompt", "strategy", "portfolio", "password", "raw_user_agent",
    ):
        assert prohibited not in ALL, prohibited
    assert "localStorage" not in ALL
    assert "sessionStorage" not in ALL


def test_formatting_goes_through_intl_and_the_shared_credit_formatter():
    shell = MODULES["admin-shell.js"]
    assert "window.CreditFormat.formatCreditsMicro(value)" in shell
    assert "Intl.NumberFormat" in shell and "Intl.DateTimeFormat" in shell
    for name in ("admin-live.js", "admin-overview.js", "admin-users.js"):
        assert "formatCreditsMicro" not in MODULES[name], name  # only via AdminShell.formatCredits
        assert ".toFixed(6)" not in MODULES[name], name


def test_renderers_can_be_lifted_with_fn_body():
    """§7.6: every renderer is a named function the shared harness can slice."""
    for signature in (
        "function renderAttention(", "function renderActiveUsers(", "function renderActivation(",
        "function renderSources(", "function renderRetention(", "function renderValue(",
        "function renderLifecycle(", "function renderCredits(", "function renderRevenue(",
        "function detailSources(", "function detailRetention(", "function detailCredits(",
        "function detailLifecycle(", "function detailHealth(",
    ):
        body = fn_body(signature, MODULES["admin-overview.js"])
        assert "innerHTML" not in body
    for signature in (
        "function renderUserRows(", "function renderPager(", "function renderEvidence(",
        "function renderProfileHeader(", "function renderProfileOverview(", "function renderActivityItems(",
    ):
        fn_body(signature, MODULES["admin-users.js"])
    body = fn_body("function renderTiles(", MODULES["admin-live.js"])
    assert "max_active_dashboard_backtests" in body


def test_client_group_taxonomies_match_the_python_source_of_truth():
    """Six hand-maintained copies across four files; this keeps them one list.

    /admin and /app have no build step, so the taxonomy in
    backend/domain/user_groups.py is re-typed in admin-shell.js (filter state),
    admin-credits.js (the editor select and its fallback), admin-overview.js
    (two swatch maps) and admin.html (the filter markup). Nothing at runtime
    notices when one copy is missed: a stale option writes a value the API
    rejects, a stale swatch map silently paints a slice steel. The pairing is
    asserted here -- the same move test_admin_page_shell.py makes for
    CSRF_FAILURE_CODE.

    Read off CODE (comments stripped), because several of those copies carry a
    comment naming the groups they mirror.
    """
    expected = list(USER_GROUPS)
    expected_labels = [USER_GROUP_LABELS[group] for group in expected]

    shell_list = re.search(r"const USER_GROUPS = \[([^\]]*)\];", CODE["admin-shell.js"])
    assert shell_list, "admin-shell.js no longer declares USER_GROUPS"
    assert re.findall(r"'([^']+)'", shell_list.group(1)) == expected

    options = re.search(
        r"const USER_GROUP_OPTIONS = Object\.freeze\(\[(.*?)\]\);",
        CODE["admin-credits.js"],
        re.S,
    )
    assert options, "admin-credits.js no longer declares USER_GROUP_OPTIONS"
    pairs = re.findall(r"\['([^']+)', '([^']+)'\]", options.group(1))
    assert [value for value, _ in pairs] == expected
    assert [label for _, label in pairs] == expected_labels
    # The fallback is a named group, not "the first option": coercing to
    # expected[0] would file every unclassified account as Internal.
    assert f"const DEFAULT_USER_GROUP = '{DEFAULT_USER_GROUP}';" in CODE["admin-credits.js"]

    for name in ("GROUP_CLASSES", "GROUP_COLORS"):
        block = re.search(
            rf"const {name} = Object\.freeze\(\{{(.*?)\}}\);",
            CODE["admin-overview.js"],
            re.S,
        )
        assert block, f"admin-overview.js no longer declares {name}"
        assert re.findall(r"(\w+):", block.group(1)) == expected, name

    select = re.search(r'<select id="filterGroup"[^>]*>(.*?)</select>', ADMIN_HTML, re.S)
    assert select, "admin.html no longer ships the group filter"
    markup = re.findall(r'<option value="([^"]*)">([^<]*)</option>', select.group(1))
    assert markup[0] == ("", "All sources"), markup[0]
    assert [value for value, _ in markup[1:]] == expected
    assert [label for _, label in markup[1:]] == expected_labels
