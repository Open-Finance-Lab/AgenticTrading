"""Guards for what the Docker build context is allowed to carry.

``.gitignore`` has no effect on a Docker build context. ``Dockerfile`` does
``COPY dashboard ./dashboard``, and ``app.py`` loads ``dashboard/.env`` at
import, so without a ``.dockerignore`` a local build bakes live credentials
into the image and the process then reads them as if an operator had set them.
That also makes the "no platform credential" path untestable against a locally
built image, because the key ships inside it.

Render builds from the Git clone, where ``.env`` was never committed, so
production images are unaffected -- these guards keep it that way and stop a
local image from becoming a credential leak if it is ever pushed.

**These checks evaluate the rules against real paths, not the pattern list.**
An earlier version asserted that certain strings appeared in the file, which
answers "is this pattern present" and not "what would Docker actually do".
Those two differ in both directions that matter here: a bare ``.env`` looks
present while leaving ``dashboard/.env`` in the image (Docker patterns are not
gitignore patterns -- one without a slash matches only at the context root),
and adding ``dashboard/`` would exclude the entire application while every
pattern assertion still passed.
"""

import fnmatch
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCKERIGNORE = _REPO_ROOT / ".dockerignore"


def _patterns() -> list[str]:
    text = _DOCKERIGNORE.read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _pattern_matches(pattern: str, path: str) -> bool:
    """Whether one ``.dockerignore`` pattern matches ``path``.

    Docker matches the context-relative path with Go's ``filepath.Match``
    semantics per component, plus ``**`` for "any number of components", and a
    pattern naming a directory excludes everything beneath it. Implemented here
    rather than taken from a library because the behaviour this file exists to
    pin *is* these semantics -- in particular that a pattern with no slash
    anchors at the context root.
    """
    pattern = pattern.rstrip("/")
    if not pattern:
        return False

    if "/" not in pattern:
        # Root-anchored: matches that name at the top level only. This is the
        # rule that makes a bare `.env` insufficient.
        head = path.split("/", 1)[0]
        return fnmatch.fnmatch(head, pattern)

    # Translate to a regex, treating `**` as "any number of path components"
    # and `*` as "anything but a separator".
    parts = []
    for segment in pattern.split("/"):
        if segment == "**":
            parts.append("(?:[^/]+/)*")
        else:
            escaped = re.escape(segment)
            escaped = escaped.replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
            parts.append(escaped + "/")
    regex = "".join(parts).rstrip("/")
    # A directory pattern also covers everything under it.
    return re.fullmatch(regex + "(/.*)?", path) is not None


def is_excluded(path: str, patterns: list[str] | None = None) -> bool:
    """Whether Docker would leave ``path`` out of the build context.

    Last matching pattern wins, and a ``!`` pattern re-includes.
    """
    rules = _patterns() if patterns is None else patterns
    excluded = False
    for rule in rules:
        if rule.startswith("!"):
            if _pattern_matches(rule[1:], path):
                excluded = False
        elif _pattern_matches(rule, path):
            excluded = True
    return excluded


# ---------------------------------------------------------------------------
# The matcher itself. If it is wrong, every case below is meaningless.
# ---------------------------------------------------------------------------

def test_the_matcher_anchors_a_slashless_pattern_at_the_context_root():
    assert _pattern_matches(".env", ".env")
    assert not _pattern_matches(".env", "dashboard/.env")
    assert _pattern_matches("**/.env", "dashboard/.env")
    assert _pattern_matches("**/.env", ".env")


def test_the_matcher_treats_a_directory_pattern_as_its_whole_subtree():
    assert _pattern_matches("credentials/", "credentials/alpaca.json")
    assert _pattern_matches("**/credentials/", "dashboard/credentials/x.json")
    assert _pattern_matches("dashboard", "dashboard/backend/app.py")


def test_the_matcher_lets_the_last_rule_win():
    rules = ["**/.env.*", "!**/.env.example"]
    assert is_excluded("dashboard/.env.local", rules)
    assert not is_excluded("dashboard/.env.example", rules)
    # Order matters, and the reverse order really is a different answer.
    assert is_excluded("dashboard/.env.example", list(reversed(rules)))


# ---------------------------------------------------------------------------
# Secrets must not enter the image.
# ---------------------------------------------------------------------------

def test_the_repository_has_a_dockerignore():
    assert _DOCKERIGNORE.is_file(), (
        "without .dockerignore, COPY dashboard ./dashboard ships dashboard/.env"
    )


def test_every_known_secret_path_is_excluded():
    """Real paths, including the nested one a bare pattern would miss.

    Verified the hard way when this shipped: adding only the bare patterns
    shrank the context from 157MB to 70MB and still shipped the credentials.
    """
    for path in (
        ".env",
        "dashboard/.env",
        ".env.local",
        "dashboard/.env.production",
        "credentials/alpaca.json",
        "dashboard/credentials/alpaca.json",
        "server.pem",
        "dashboard/backend/tls/server.key",
    ):
        assert is_excluded(path), path


def test_the_example_env_survives_the_exclusions():
    """Committed documentation, not a secret; the runbook points operators at
    it. It is re-included *after* the `.env.*` rule, which is the only order
    that works."""
    assert not is_excluded(".env.example")
    assert not is_excluded("dashboard/.env.example")


def test_the_isolated_venvs_are_not_copied_in():
    """The image builds its own under /opt; copying a host venv in would both
    bloat the context and shadow the interpreter the Dockerfile provisioned."""
    for path in (
        ".venv/bin/python",
        ".ai-hedge-fund-venv/bin/python",
        ".tradingagents-venv/bin/python",
        ".tradingagents-venv/lib/python3.12/site-packages/tradingagents/__init__.py",
    ):
        assert is_excluded(path), path


def test_vcs_and_cache_noise_is_excluded():
    for path in (
        ".git/config",
        ".github/workflows/ci.yml",
        "dashboard/backend/__pycache__/app.cpython-312.pyc",
        "packaging/agentictrading/src/agentictrading.egg-info/PKG-INFO",
    ):
        assert is_excluded(path), path


# ---------------------------------------------------------------------------
# The application must enter the image. This is the half that was missing:
# every exclusion above is satisfiable by excluding everything.
# ---------------------------------------------------------------------------

def test_the_application_itself_is_in_the_build_context():
    """Adding ``dashboard/`` to .dockerignore would pass every pattern
    assertion and ship an image with no application in it."""
    for path in (
        "dashboard/backend/app.py",
        "dashboard/backend/api/routers/backtests.py",
        "dashboard/backend/infrastructure/tradingagents/adapter.py",
        "dashboard/backend/infrastructure/tradingagents/bridge.py",
        "dashboard/frontend/app.html",
        "dashboard/frontend/app.js",
        "dashboard/frontend/js/agent-editor.js",
        "dashboard/scripts/backtest_hourly_agent.py",
        "packaging/agentictrading/src/agentictrading/__init__.py",
        "requirements.txt",
        "requirements-tradingagents.txt",
        "Dockerfile",
    ):
        assert not is_excluded(path), path


def test_the_configuration_the_app_reads_at_boot_is_in_the_build_context():
    for path in (
        "dashboard/config/marketplace.json",
        "dashboard/config/defaults.json",
        "dashboard/config/leaderboard.json",
    ):
        assert not is_excluded(path), path


def test_the_committed_seed_database_is_in_the_build_context():
    """``*.db`` must NOT be excluded, however much the comment above it
    suggests otherwise.

    ``dashboard/storage/data/backtest.db`` is committed and holds the seed runs
    ``dashboard/config/defaults.json`` references; on Render, where there is no
    persistent disk, that file is what ``DATABASE_PATH`` resets to on every
    deploy. An independent review proposed adding ``*.db`` on the grounds that
    it is "the extension this repo actually uses" -- which is true, and is
    exactly why adding it would break the image. A developer's working copy
    cannot be told from the seed by extension; this test exists so the
    plausible-looking fix fails loudly.
    """
    seed = _REPO_ROOT / "dashboard" / "storage" / "data" / "backtest.db"
    assert seed.is_file(), "the seed DB is committed; this test assumes it"
    assert not is_excluded("dashboard/storage/data/backtest.db")
    # The scratch forms around it stay excluded.
    assert is_excluded("dashboard/storage/data/backtest.db-journal")
    assert is_excluded("dashboard/storage/data/scratch.sqlite")
