# Marketplace Card Test Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unblock the repository backend test gate by correcting the Marketplace card comparison test's normalization of the provider label.

**Architecture:** Keep production Marketplace markup unchanged. Update the focused Python/Node contract test so it removes the exact open-source badge and normalizes the provider label in the serialized card HTML, covering both visible text and the `title` tooltip while preserving byte-level comparison for every other field.

**Tech Stack:** Python, pytest, Node.js test subprocess, GitHub Actions backend test suite.

**Spec:** `docs/superpowers/specs/2026-08-28-marketplace-card-test-fix-design.md`

## Global Constraints

- Change only the Marketplace contract test; do not modify production UI.
- Preserve strict detection of any closed-card badge, marker, or structural difference.
- Do not touch Credits, Admin Analytics, PR #411 files, secrets, databases, `.superpowers/`, or `work/` artifacts.
- Use English code, filenames, and commit messages; communicate in Chinese.

### Task 1: Correct Provider Label Normalization

**Files:**
- Modify: `dashboard/backend/tests/test_frontend_model_facets.py:487-492`
- Test: `dashboard/backend/tests/test_frontend_model_facets.py::test_closed_card_differs_from_open_card_by_exactly_the_badge`

**Interfaces:**
- Consumes: the existing `openHtml`, `closedHtml`, and exact `BADGE` constants in the Node script.
- Produces: `equalAfterNormalizing === true` when the two cards differ only by the open-source badge and provider label.

- [ ] **Step 1: Write the failing test assertion fixture check**

  Keep the existing test unchanged initially and run it to document the baseline failure caused by the `title` attribute containing different provider labels.

- [ ] **Step 2: Run the focused test to verify the baseline failure**

  Run:

  ```bash
  python -m pytest dashboard/backend/tests/test_frontend_model_facets.py::test_closed_card_differs_from_open_card_by_exactly_the_badge -q
  ```

  Expected before the fix: failure at `equalAfterNormalizing` with the message that the closed card must be byte-identical after removing the badge.

- [ ] **Step 3: Normalize the provider label across serialized HTML**

  In the embedded Node script, replace the current one-occurrence substitutions:

  ```javascript
  const openWithoutBadge = openHtml.split(BADGE).join('')
    .replace('Powered by DeepSeek', 'POWERED_BY_MODEL');
  const closedNormalized = closedHtml.replace('Powered by Claude', 'POWERED_BY_MODEL');
  ```

  with global substitutions that cover both the visible text and the `title` attribute:

  ```javascript
  const openWithoutBadge = openHtml.split(BADGE).join('')
    .split('Powered by DeepSeek').join('POWERED_BY_MODEL');
  const closedNormalized = closedHtml
    .split('Powered by Claude').join('POWERED_BY_MODEL');
  ```

  Do not remove or normalize any other HTML, class, tag, whitespace, or card content.

- [ ] **Step 4: Run the focused test and module**

  Run:

  ```bash
  python -m pytest dashboard/backend/tests/test_frontend_model_facets.py::test_closed_card_differs_from_open_card_by_exactly_the_badge -q
  python -m pytest dashboard/backend/tests/test_frontend_model_facets.py -q
  ```

  Expected: both commands pass, including the existing checks that open cards have the badge and closed/unknown cards do not.

- [ ] **Step 5: Verify scope and commit**

  Run:

  ```bash
  git diff --check
  git status --short
  git diff -- dashboard/backend/tests/test_frontend_model_facets.py
  ```

  Confirm only the intended test and documentation files are present, then commit the implementation:

  ```bash
  git add dashboard/backend/tests/test_frontend_model_facets.py
  git commit -m "test: normalize marketplace provider labels"
  ```

### Task 2: Full Verification and Pull Request

**Files:**
- Verify: repository test and CI configuration; no additional source changes expected.

**Interfaces:**
- Consumes: the committed Marketplace test fix from Task 1.
- Produces: a pushed branch and pull request targeting `main`.

- [ ] **Step 1: Run the backend test suite used by CI**

  Run the repository's documented backend CI command from `.github/workflows/ci.yml` and record any unrelated failures separately. The Marketplace module and focused test must remain green.

- [ ] **Step 2: Confirm the final diff is isolated**

  Run:

  ```bash
  git diff origin/main...HEAD --stat
  git diff origin/main...HEAD --name-only
  ```

  Expected: the implementation diff contains only `dashboard/backend/tests/test_frontend_model_facets.py`; design and plan documentation are the only additional files in the branch.

- [ ] **Step 3: Push and open the PR**

  Push `fix/marketplace-card-test` to `origin` and open a PR targeting `main` with a title such as `test: fix marketplace card provider-label comparison`. Explain that the prior assertion compared a visible label and tooltip as different legitimate strings, and that production Marketplace markup is unchanged.
