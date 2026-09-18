# Admin console port — deferred items

Deferrals recorded by the AFK dev loop while executing
`docs/superpowers/plans/2026-09-18-admin-console-port.md`. Each entry names the
defer rule (D1–D6) that fired and the evidence for it.

---

### §0.C1 — The ported header CSS has no drift guard

**Status: deferred 2026-09-18 in PR #490 (defer rule D2)**

**What:** `test_admin_header_parity.py` compares the two `<header>` **markup**
blocks and asserts they are the same header. The ~40 retyped CSS declarations in
`admin.css` (the `.header` family plus seven breakpoints, copied from
`styles.css`) have no equivalent check. An edit to `styles.css`'s `.header`
family — a padding change, a token swap, a new breakpoint — leaves `/admin`'s
copy behind with a fully green suite, which re-creates the reported bug ("the
newer header from the /admin subdomain pages will have to go") through the CSS
half rather than the markup half. The new test's docstring claims "this file is
the mechanism"; today that is true of the markup only.

**Why deferred (D2 — harness blindspot):** verifying it needs a harness that
does not exist here — something that parses both stylesheets, extracts a named
selector's declaration block from each, and diffs them modulo an allowlist of
intended divergences. That allowlist is not small and is not mechanical: the
port legitimately differs from source on `.nav-menu-toggle` and
`.auth-account-btn` (`min-height:0`, resetting this file's 40px bare-`button`
floor), `.primary-nav` and `.header-right` (each merged from two separate
`styles.css` blocks), `.account-menu-item:hover` and `.account-menu-item--danger`
(colours restated to beat this file's generic `a:hover,button:hover` rule), and
every rule `styles.css` writes expanded that `admin.css` writes minified. A
comparison that hardcodes those exceptions as string literals is a second copy
of the same drift problem one layer up. Getting it right is its own piece of
work, and shipping a version that silently matches nothing is worse than
shipping none — the failure mode this whole port is guarding against.

Not a budget deferral: the markup guard was built in this PR, and the CSS guard
was scoped and found to need a harness, not more time.

**Next-session entry point:** `dashboard/backend/tests/test_admin_header_parity.py`
— add a `test_the_ported_css_still_matches_styles_css` beside the markup guard.
Source blocks: `dashboard/frontend/styles.css:95-391` (header family),
`:649-682` (`.mode-toggle`/`.mode-btn`), `:5826-5840`
(`.primary-nav`/`.nav-menu-toggle`), `:10595-10692` (account dropdown), and the
breakpoints at `:2903`, `:2917`, `:2951`, `:2975`, `:2986`, `:3077`, `:10041`.
Ported copies: `dashboard/frontend/admin.css:58-85` and the seven `@media`
blocks that follow the "Universal top bar, responsive" comment. Re-run
`pytest dashboard/backend/tests/test_admin_header_parity.py -q`. Estimated
effort: half a chunk, most of it deciding what the allowlist is allowed to be.
