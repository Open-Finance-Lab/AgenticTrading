# Marketplace Card Test Fix Design

## Goal

Restore the backend test suite's Marketplace card contract by making the test normalize the model-provider label everywhere it is rendered, including the accessibility tooltip, while preserving strict detection of any other card difference.

## Context

The test `test_closed_card_differs_from_open_card_by_exactly_the_badge` renders an open-weight and a closed-weight Marketplace card with identical fixture fields except `model_name`. The production card renders the provider label both as visible text and as the `title` attribute on `.agent-card-submeta`. The test currently normalizes only the visible text, so the expected comparison fails on the legitimate `title` value difference. This failure is present on `origin/main` and blocks unrelated pull requests.

## Design

- Change only `dashboard/backend/tests/test_frontend_model_facets.py`.
- Keep the production `renderMarketplaceGrid` implementation unchanged.
- Replace the provider label in the complete serialized HTML for both cards using the same normalization token. This covers visible text and the `title` attribute without stripping arbitrary markup.
- Continue removing only the exact open-source badge from the open card.
- Continue asserting that the closed card contains no licence-badge class and that the normalized cards are byte-identical. Any additional closed-card marker or structural difference must still fail the test.

## Verification

- Run the focused failing test and the complete `test_frontend_model_facets.py` module.
- Run the full backend suite used by CI when practical.
- Confirm the diff contains no production UI changes, secrets, database files, `.superpowers/`, or `work/` artifacts.

## Out of Scope

- Credits precision or run-level Activity aggregation.
- Admin Analytics UI and API changes.
- Marketplace card markup, styling, copy, or behavior.
