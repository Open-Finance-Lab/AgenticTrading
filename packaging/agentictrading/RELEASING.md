# Releasing `agentictrading`

The package is published to PyPI automatically by
`.github/workflows/publish-pypi.yml` whenever a `v*` tag is pushed.

## One-time setup: PyPI Trusted Publishing (no token needed)

The workflow uses OIDC Trusted Publishing, so no API token is stored in GitHub.
Configure it once on PyPI:

1. Go to https://pypi.org/manage/project/agentictrading/settings/publishing/
   (Project → Settings → Publishing).
2. Add a new **GitHub** trusted publisher with:
   - **Owner:** `Open-Finance-Lab`
   - **Repository:** `AgenticTrading`
   - **Workflow name:** `publish-pypi.yml`
   - **Environment:** `pypi`

   The owner must be the repo the release tag is pushed to, because that is
   where the workflow runs and mints the OIDC token. A publisher registered
   against a fork (e.g. `Allan-Feng`) rejects it as an invalid publisher.
3. In `Open-Finance-Lab/AgenticTrading`, create an environment named `pypi`
   (Settings → Environments → New environment → `pypi`). Optionally add
   protection rules (required reviewers) for an approval gate before publish.

That's it — no `PYPI_API_TOKEN` secret is required.

> Prefer a token instead of OIDC? Add a repo secret `PYPI_API_TOKEN`, delete the
> `permissions: id-token: write` block + `environment:` from the `publish` job,
> and give the publish step:
> `with: { password: ${{ secrets.PYPI_API_TOKEN }} }`.

## Cutting a release

PyPI versions are immutable — every release needs a new version number. Run
every command below from the repo root.

1. Set the version in the single source of truth,
   `packaging/agentictrading/src/agentictrading/__init__.py`:

   ```python
   __version__ = "<new version>"
   ```

   It must be higher than the latest release on
   https://pypi.org/project/agentictrading/. If `__version__` already holds an
   unpublished number above that, keep it and go straight to step 3.

   (`pyproject.toml` reads this automatically via `dynamic = ["version"]`.)

2. Land the bump through a pull request, like any other change. Do not push it
   to `main` directly: that skips review, and every change on `main` deploys prod.

3. Tag the merged commit and push **only that tag**:

   ```bash
   git switch main && git pull --ff-only origin main
   VERSION="$(sed -nE 's/^__version__ = "([^"]+)"/\1/p' packaging/agentictrading/src/agentictrading/__init__.py)"
   git tag "v$VERSION"
   git push origin "v$VERSION"
   ```

   Not `git push --tags`: every `v*` tag that reaches the repo triggers a
   publish, so a stale local tag would start a second one.

4. The workflow builds, checks the tag matches `__version__`, and publishes to
   PyPI. Watch it under the repo's **Actions** tab.

## Manual fallback

```bash
cd packaging/agentictrading
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/*
python -m twine upload dist/*        # username __token__, password = PyPI token
```
