# Admin Acquisition and ATL Operations Analytics Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add a first-version acquisition and ATL operations view that tells an administrator where users came from, how many started and completed the core backtest task, who returned, what settled ATL Credits they consumed, and who showed verified purchase intent or paid.

**Architecture:** Keep the existing Analytics navigation and User Analytics Profile. Add a privacy-safe attribution record and immutable correction audit alongside the existing Analytics repository, capture source only from a signed controlled invite token, and compose acquisition groups from authoritative user, Analytics event, run, and Credits records. Extend the existing Admin Analytics contracts with optional acquisition filters and one independent acquisition panel, then render the group table, source/cohort filters, profile attribution editor, and drill-down navigation in the current vanilla JavaScript modules.

**Tech Stack:** Python 3, FastAPI, Pydantic v2, SQLite, PostgreSQL/psycopg, pytest, vanilla JavaScript, semantic HTML, CSS, Chart.js only where an existing Analytics chart already owns the surface.

**Spec:** docs/superpowers/specs/2026-09-06-admin-acquisition-operations-design.md (extends docs/superpowers/specs/2026-08-26-admin-user-analytics-design.md)

## Global Constraints

- Acquisition source is exactly student, community, friend, competition, or unknown; missing or invalid attribution resolves to unknown.
- Controlled invite tokens are verified only by the backend with the `ACQUISITION_INVITE_SIGNING_KEY` environment variable; the key is never sent to the browser, persisted, or included in fixtures.
- Operating cohorts are nullable lower-case slugs of 1 through 64 characters matching ^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$; raw URLs and free-form referral text are never stored.
- Source is captured only from a server-verified controlled invite/campaign token. Do not infer it from email, IP, page views, User-Agent, or other guesses.
- Lifecycle, source/cohort, blocked state, and paid state remain separate dimensions. Do not encode one axis into another.
- Date controls are 1D, 1W, 1M, and 1Y, interpreted as UTC half-open ranges. Lifetime progress fields do not change with the selected range.
- Active means a distinct user with at least one accepted backtest_requested in the selected range, regardless of success or failure.
- Task completed means a persisted, viewable successful backtest_completed exists in the user's full history.
- Repeat means at least two successful backtests on different UTC calendar dates in the user's full history.
- Runs per active user is accepted requests divided by active users; a zero-active group returns null and the UI renders an em dash.
- ATL Credits in acquisition groups are settled platform-credit debits only. Grants, reservations, refunds, and BYOK provider spend are excluded.
- Paid intent is a distinct user with a successful server-created checkout; Paid is a distinct user with a settled purchased-credit ledger entry. Admin grants are neither.
- Checkout-to-purchase conversion uses a seven-day observation window and excludes immature windows from the denominator; the UI labels incomplete coverage.
- Default Analytics queries exclude Admin and analytics_excluded accounts. include_internal=true is the only explicit override.
- Attribution corrections record actor, UTC timestamp, previous source/cohort, and new source/cohort. No correction reason is collected or returned.
- Every SQLite implementation has a matching PostgreSQL implementation and the same repository method contract.
- Analytics writes and reads are observational: an attribution or Analytics failure never fails signup, backtest execution, checkout creation, Credits settlement, or authentication.
- APIs return display-safe data only. Never persist, return, log, or fixture API keys, credentials, prompts, strategies, raw provider bodies, payment payloads, raw referral URLs, full IPs, or raw User-Agent values.
- The default acquisition table has no input-token, output-token, or total-token columns.
- Panel failures use the existing partial-availability contract; unavailable aggregate values are not silently converted to zero.
- Admin Analytics remains read-only except for the explicitly scoped Profile attribution correction action. Account role, Credits, provider, and other mutations remain in existing Admin surfaces.
- Preserve unrelated working-tree changes. Never stage or commit dashboard/storage/data/backtest.db, .superpowers/, work/, secrets, or analytics-layout-demo.html.
- Use synthetic fixtures and fake repositories only. No real API key, payment provider, production database, or production identity is required by tests.

## File Map

- dashboard/backend/domain/analytics/acquisition.py: source/cohort types, signed invite-token parsing, filter validation, and pure group formulas.
- dashboard/backend/domain/analytics/models.py: add the server event contract for checkout_started.
- dashboard/backend/domain/analytics/repository.py: SQLite attribution, correction-audit, and acquisition read/write methods.
- dashboard/backend/domain/analytics/repository_postgres.py: PostgreSQL twin of the attribution and correction-audit contract.
- dashboard/backend/domain/analytics/value_repository.py: batched acquisition joins over Analytics events, run evidence, and Credits ledger facts.
- dashboard/backend/domain/analytics/query_service.py: extend the legacy overview/user projections with acquisition filters and the stable acquisition_groups member.
- dashboard/backend/domain/analytics/value_queries.py: typed acquisition response models and value-module filtering/drill-down composition.
- dashboard/backend/api/auth.py: accept a signed invite token at signup and best-effort persist the resolved attribution.
- dashboard/backend/api/routers/credits.py: emit checkout_started after a checkout/order is successfully created.
- dashboard/backend/api/routers/admin_analytics.py: parse acquisition filters, expose attribution read/update, and pass filters to all supported Analytics queries.
- dashboard/frontend/app.js: preserve a controlled invite token when the authenticated signup form submits.
- dashboard/frontend/index.html: preserve the controlled invite token in the landing-page signup form.
- dashboard/frontend/app.html: add shared source/cohort/lifecycle/blocked/paid controls, the collapsible Acquisition groups panel, and Profile attribution fields.
- dashboard/frontend/js/admin-analytics-value.js: own acquisition panel state, requests, table rendering, filter URL state, group/metric drill-down, and partial errors.
- dashboard/frontend/js/admin-analytics.js: render Profile attribution metadata/editing, save corrections, and preserve back navigation.
- dashboard/frontend/styles.css: style the full-width group table, compact filters, attribution editor, and unavailable state without changing the persistent Admin rail.
- dashboard/backend/tests/domain/analytics/test_acquisition.py: pure source/cohort and formula tests.
- dashboard/backend/tests/domain/analytics/test_acquisition_repository.py: SQLite/PostgreSQL repository contract tests and privacy assertions.
- dashboard/backend/tests/test_auth_acquisition.py: signed invite capture and signup-failure isolation tests.
- dashboard/backend/tests/test_credits_checkout_analytics.py: checkout event idempotency and failure isolation tests.
- dashboard/backend/tests/test_admin_acquisition_api.py: API query, filter, correction, audit, and partial-error tests.
- dashboard/backend/tests/test_admin_acquisition_frontend.py: static UI contract, ARIA, URL, navigation, and prohibited-field tests.
- dashboard/backend/tests/fixtures/admin_analytics/acquisition.json: safe group response fixture with no token columns or sensitive fields.
- dashboard/backend/tests/fixtures/admin_analytics/acquisition_partial_error.json: safe independent-panel failure fixture.

---

### Task 1: Define Attribution Types and Pure Acquisition Rules

**Files:**
- Create: dashboard/backend/domain/analytics/acquisition.py
- Create: dashboard/backend/tests/domain/analytics/test_acquisition.py

**Interfaces:**
- Consumes: UTC datetimes, allowlisted Analytics event records, and display-safe integer facts supplied by later repository tasks.
- Produces: AcquisitionSource, AttributionMethod, AcquisitionAttribution, AcquisitionFilters, AcquisitionGroupKey, AcquisitionGroupFacts, normalize_cohort_slug(value), resolve_invite_token(token, signing_key, now), active_user_ids(events, start, end), repeat_user_ids(success_events), runs_per_active_user(runs, active), and checkout_window_is_mature(checkout_at, as_of).

- [ ] Step 1: Write failing validation and formula tests.

    @pytest.mark.parametrize("source", ["student", "community", "friend", "competition", "unknown"])
    def test_all_sources_are_accepted(source):
        assert AcquisitionAttribution(user_id=7, source=source, cohort=None).source == source

    def test_missing_source_defaults_to_unknown():
        assert AcquisitionAttribution(user_id=7).source == "unknown"

    def test_cohort_slug_is_lowercase_bounded_and_nullable():
        assert normalize_cohort_slug("discord-aug") == "discord-aug"
        assert normalize_cohort_slug(None) is None
        with pytest.raises(ValueError):
            normalize_cohort_slug("Discord/Aug")
        with pytest.raises(ValueError):
            normalize_cohort_slug("x" * 65)

    def test_active_counts_requests_even_when_terminal_event_failed():
        events = [event(7, "backtest_requested"), event(7, "backtest_failed")]
        assert active_user_ids(events, START, END) == {7}

    def test_repeat_requires_successes_on_different_utc_dates():
        successes = [
            event(7, "backtest_completed", at="2026-09-01T23:30:00+00:00"),
            event(7, "backtest_completed", at="2026-09-02T00:30:00+00:00"),
        ]
        assert repeat_user_ids(successes) == {7}

    def test_zero_active_group_has_no_division_by_zero():
        assert runs_per_active_user(runs=4, active=0) is None

- [ ] Step 2: Run the focused tests to verify the module is absent.

  Run: pytest -q dashboard/backend/tests/domain/analytics/test_acquisition.py

  Expected: FAIL during collection because dashboard.backend.domain.analytics.acquisition does not exist.

- [ ] Step 3: Implement frozen Pydantic models and deterministic formulas.

    AcquisitionSource = Literal["student", "community", "friend", "competition", "unknown"]
    AttributionMethod = Literal["invite", "manual", "unknown"]

    class AcquisitionAttribution(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        user_id: int = Field(gt=0)
        source: AcquisitionSource = "unknown"
        cohort: str | None = None
        method: AttributionMethod = "unknown"
        attributed_at: datetime | None = None
        original_source: AcquisitionSource = "unknown"
        original_cohort: str | None = None
        last_corrected_at: datetime | None = None
        last_corrected_by_admin_id: int | None = Field(default=None, gt=0)

    class AcquisitionFilters(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        source: AcquisitionSource | None = None
        cohort: str | None = None
        lifecycle: Literal["new", "active", "at_risk", "dormant"] | None = None
        blocked: bool | None = None
        paid: bool | None = None
        include_internal: bool = False

  Use UTC date comparisons for every formula. Treat a signed invite token as an opaque input whose verified payload contains only source and optional cohort. Invalid signatures, expired timestamps, unknown sources, and malformed cohorts resolve to unknown in signup rather than raising.

- [ ] Step 4: Add purchase-window boundary tests.

    def test_checkout_window_excludes_unmatured_rows():
        checkout = datetime(2026, 9, 1, 12, tzinfo=UTC)
        assert checkout_window_is_mature(checkout, checkout + timedelta(days=6, hours=23)) is False
        assert checkout_window_is_mature(checkout, checkout + timedelta(days=7)) is True

- [ ] Step 5: Run the focused tests and commit.

  Run: pytest -q dashboard/backend/tests/domain/analytics/test_acquisition.py

  Expected: PASS.

    git add dashboard/backend/domain/analytics/acquisition.py dashboard/backend/tests/domain/analytics/test_acquisition.py
    git commit -m "feat: define acquisition analytics rules"

### Task 2: Persist Attribution and Correction Audit With SQLite/PostgreSQL Parity

**Files:**
- Modify: dashboard/backend/domain/analytics/repository.py
- Modify: dashboard/backend/domain/analytics/repository_postgres.py
- Create: dashboard/backend/tests/domain/analytics/test_acquisition_repository.py

**Interfaces:**
- Consumes: AcquisitionAttribution, positive user IDs, and the existing Analytics repository connection helpers.
- Produces: AnalyticsStore.get_user_attribution(user_id) -> AcquisitionAttribution, AnalyticsStore.list_user_attributions(user_ids) -> dict[int, AcquisitionAttribution], AnalyticsStore.record_initial_attribution(attribution) -> AcquisitionAttribution, AnalyticsStore.update_user_attribution(user_id, source, cohort, actor_user_id, now) -> AcquisitionAttribution, and AnalyticsStore.list_attribution_audit(user_id, limit=50) -> list[dict[str, object]].

- [ ] Step 1: Write failing schema, default, correction, and parity tests.

    def test_missing_row_reads_as_unknown(tmp_path):
        store = AnalyticsStore(tmp_path / "analytics.db")
        assert store.get_user_attribution(7) is None

    def test_initial_attribution_preserves_original_values(store):
        row = store.record_initial_attribution(
            AcquisitionAttribution(user_id=7, source="community", cohort="discord-aug", method="invite")
        )
        assert row.original_source == "community"
        assert row.last_corrected_at is None

    def test_correction_is_atomic_and_has_no_reason_field(store):
        updated = store.update_user_attribution(
            7, source="competition", cohort="competition-2026", actor_user_id=99, now=NOW
        )
        assert updated.source == "competition"
        assert updated.last_corrected_by_admin_id == 99
        audit = store.list_attribution_audit(7)
        assert audit[0] == {
            "actor_user_id": 99,
            "subject_user_id": 7,
            "changed_at": NOW.isoformat(),
            "previous_source": "community",
            "new_source": "competition",
            "previous_cohort": "discord-aug",
            "new_cohort": "competition-2026",
        }
        assert "reason" not in audit[0]

- [ ] Step 2: Run the repository tests to verify the methods and tables are missing.

  Run: pytest -q dashboard/backend/tests/domain/analytics/test_acquisition_repository.py

  Expected: FAIL because the attribution tables and repository methods do not exist.

- [ ] Step 3: Add additive DDL and bounded projections.

  Add these tables to both repository DDL strings:

    CREATE TABLE IF NOT EXISTS user_acquisition_attributions (
        user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
        source TEXT NOT NULL DEFAULT 'unknown',
        cohort TEXT,
        method TEXT NOT NULL DEFAULT 'unknown',
        attributed_at TEXT,
        original_source TEXT NOT NULL DEFAULT 'unknown',
        original_cohort TEXT,
        last_corrected_at TEXT,
        last_corrected_by_admin_id INTEGER REFERENCES users(id) ON DELETE RESTRICT
    );

    CREATE TABLE IF NOT EXISTS admin_analytics_attribution_audit (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        actor_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
        subject_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
        changed_at TEXT NOT NULL,
        previous_source TEXT NOT NULL,
        new_source TEXT NOT NULL,
        previous_cohort TEXT,
        new_cohort TEXT
    );

  Use BIGSERIAL for the PostgreSQL audit sequence and its existing boolean/integer syntax. Validate source/cohort through Task 1 before SQL. Initial writes use `INSERT INTO user_acquisition_attributions (user_id, source, cohort, method, attributed_at, original_source, original_cohort) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (user_id) DO NOTHING` so retries cannot overwrite original attribution. Corrections update the current row and insert one audit row in the same transaction. Return only the six audit fields shown above.

- [ ] Step 4: Add migration and privacy assertions.

    def test_attribution_tables_are_idempotent_and_no_raw_token_column_exists(store):
        store._init_schema()
        store._init_schema()
        columns = table_columns(store, "user_acquisition_attributions")
        assert "invite_token" not in columns
        assert "referral_url" not in columns

- [ ] Step 5: Run SQLite and PostgreSQL contract suites and commit.

  Run: pytest -q dashboard/backend/tests/domain/analytics/test_acquisition_repository.py dashboard/backend/tests/domain/analytics/test_repository_contract.py dashboard/backend/tests/domain/analytics/test_repository_postgres.py

  Expected: PASS for the configured PostgreSQL contract environment; SQLite tests must always run locally.

    git add dashboard/backend/domain/analytics/repository.py dashboard/backend/domain/analytics/repository_postgres.py dashboard/backend/tests/domain/analytics/test_acquisition_repository.py
    git commit -m "feat: persist acquisition attribution audit"

### Task 3: Capture Controlled Source at Signup Without Blocking Account Creation

**Files:**
- Modify: dashboard/backend/api/auth.py
- Modify: dashboard/frontend/app.js
- Modify: dashboard/frontend/index.html
- Create: dashboard/backend/tests/test_auth_acquisition.py

**Interfaces:**
- Consumes: resolve_invite_token(token, signing_key, now), AnalyticsStore.record_initial_attribution, and the existing SignupRequest.
- Produces: SignupRequest.invite_token: str | None, not returned in AuthResponse, and best-effort capture_signup_attribution(user_id, invite_token, now).

- [ ] Step 1: Write failing signup and isolation tests.

    def test_valid_signed_invite_is_saved_without_persisting_the_token(client, analytics_store):
        token = signed_invite(source="student", cohort="fall-course")
        response = client.post("/api/auth/signup", json=signup_payload(invite_token=token))
        assert response.status_code == 200
        attribution = analytics_store.get_user_attribution(response.json()["user"]["id"])
        assert attribution.source == "student"
        assert attribution.cohort == "fall-course"
        assert token not in serialized_database(analytics_store)

    def test_missing_or_tampered_invite_defaults_to_unknown(client, analytics_store):
        response = client.post("/api/auth/signup", json=signup_payload(invite_token="tampered"))
        assert response.status_code == 200
        assert analytics_store.get_user_attribution(response.json()["user"]["id"]).source == "unknown"

    def test_attribution_write_failure_does_not_fail_signup(client, monkeypatch):
        monkeypatch.setattr(analytics_store, "record_initial_attribution", fail_once)
        assert client.post("/api/auth/signup", json=signup_payload()).status_code == 200

- [ ] Step 2: Run the focused tests to verify the request field and capture hook are absent.

  Run: pytest -q dashboard/backend/tests/test_auth_acquisition.py

  Expected: FAIL because invite_token and capture_signup_attribution do not exist.

- [ ] Step 3: Add strict token plumbing and best-effort capture.

  Add a maximum length of 512 to invite_token. The backend verifies HMAC and expiry, extracts only the allowlisted source/cohort, and calls record_initial_attribution after the user row commits. Catch only attribution persistence failures at this boundary, print a bounded category, and leave the account response unchanged. Do not log the token, payload, email, or display name.

  Update both signup callers to copy only the current page's invite query value into the JSON request. Never append it to Analytics URLs or include it in an auth response. A normal signup with no invite still records unknown when the best-effort store is available.

- [ ] Step 4: Run auth, user-store, and frontend source tests and commit.

  Run: pytest -q dashboard/backend/tests/test_auth_acquisition.py dashboard/backend/tests/test_users_store.py dashboard/backend/tests/test_admin_console_frontend.py

  Expected: PASS.

    git add dashboard/backend/api/auth.py dashboard/frontend/app.js dashboard/frontend/index.html dashboard/backend/tests/test_auth_acquisition.py
    git commit -m "feat: capture controlled acquisition source at signup"

### Task 4: Add Server-Authoritative Checkout Intent Instrumentation

**Files:**
- Modify: dashboard/backend/domain/analytics/models.py
- Modify: dashboard/backend/domain/analytics/instrumentation.py
- Modify: dashboard/backend/api/routers/credits.py
- Create: dashboard/backend/tests/test_credits_checkout_analytics.py

**Interfaces:**
- Consumes: successful CheckoutResult.order_id, authenticated user ID, and existing analytics_instrumentation.emit_resource_event.
- Produces: allowlisted server event checkout_started in event group resource, with source_event_id="checkout:{order_id}", source_record_type="credit_checkout", and source_record_id=order_id.

- [ ] Step 1: Write failing event-model and route tests.

    def test_checkout_started_is_a_supported_empty_property_server_event():
        draft = AnalyticsEventRecord(
            event_name="checkout_started",
            event_group="resource",
            event_source="server",
            user_id=7,
            source_event_id="checkout:ord-7",
            occurred_at=NOW,
            received_at=NOW,
            properties={},
        )
        assert draft.event_name == "checkout_started"
        assert draft.event_group == "resource"

    def test_checkout_route_emits_once_after_order_creation(client, fake_analytics):
        response = client.post("/api/credits/checkout-sessions", json=checkout_payload())
        assert response.status_code == 200
        assert fake_analytics.events == [("checkout_started", "checkout:ord-7")]

    def test_analytics_failure_does_not_turn_successful_checkout_into_5xx(client, monkeypatch):
        monkeypatch.setattr(analytics_instrumentation, "emit_resource_event", fail_once)
        assert client.post("/api/credits/checkout-sessions", json=checkout_payload()).status_code == 200

- [ ] Step 2: Run focused tests to verify the event is rejected.

  Run: pytest -q dashboard/backend/tests/test_credits_checkout_analytics.py

  Expected: FAIL because checkout_started is not in the event allowlist and the route emits no event.

- [ ] Step 3: Extend the allowlist and emit after the authoritative service result.

  Add checkout_started to ALLOWED_SERVER_EVENT_NAMES, EVENT_GROUP_BY_NAME, and _EMPTY_SERVER_EVENTS. Emit only after credits_service.create_checkout returns. Use the stable order ID as source record and idempotency reference; never store the checkout URL, provider payload, amount metadata, or payment details. Keep the existing response and error handling unchanged if Analytics is unavailable.

- [ ] Step 4: Run Credits integration and Analytics model tests and commit.

  Run: pytest -q dashboard/backend/tests/test_credits_checkout_analytics.py dashboard/backend/tests/integration/test_credits_checkout_flow.py dashboard/backend/tests/domain/analytics/test_models.py dashboard/backend/tests/domain/analytics/test_instrumentation.py

  Expected: PASS.

    git add dashboard/backend/domain/analytics/models.py dashboard/backend/domain/analytics/instrumentation.py dashboard/backend/api/routers/credits.py dashboard/backend/tests/test_credits_checkout_analytics.py
    git commit -m "feat: instrument checkout intent safely"

### Task 5: Build Acquisition Group Facts From Authoritative Records

**Files:**
- Modify: dashboard/backend/domain/analytics/value_repository.py
- Modify: dashboard/backend/domain/analytics/value_queries.py
- Modify: dashboard/backend/domain/analytics/query_service.py
- Modify: dashboard/backend/domain/analytics/metrics.py
- Create: dashboard/backend/tests/domain/analytics/test_acquisition_queries.py

**Interfaces:**
- Consumes: attribution rows, event records, current value snapshots, run terminal events, credit_llm_usage_entries, credit_ledger_entries, and checkout_started events.
- Produces: AcquisitionGroup, AcquisitionAnalyticsResponse, AnalyticsQueryService.get_overview(filters: AnalyticsMetricFilters, now: datetime | None = None).acquisition_groups, ValueAnalyticsQueryService.get_acquisition_groups(start: date, end: date, group_by: Literal["source", "cohort"], filters: AcquisitionFilters, now: datetime | None = None), and ValueAnalyticsStore.list_acquisition_facts(user_ids: Sequence[int], start: datetime, end: datetime) -> dict[int, AcquisitionGroupFacts].

- [ ] Step 1: Write failing formula, exclusion, and response-shape tests.

    def test_group_metrics_use_lifetime_progress_and_selected_range_activity(query):
        response = query.get_acquisition_groups(
            start=date(2026, 9, 1), end=date(2026, 9, 8),
            group_by="source", filters=AcquisitionFilters(), now=NOW,
        )
        community = next(row for row in response.groups if row.group.value == "community")
        assert community.users == 2
        assert community.active == 1
        assert community.task_completed == 2
        assert community.repeat == 1
        assert community.runs == 3
        assert community.runs_per_active_user == 3.0
        assert community.atl_credits_settled_micro == 420_000
        assert community.paid_intent == 1
        assert community.paid == 1

    def test_internal_and_excluded_accounts_are_omitted_by_default(query):
        response = query.get_acquisition_groups(
            start=date(2026, 9, 1), end=date(2026, 9, 8), group_by="source",
            filters=AcquisitionFilters(), now=NOW,
        )
        assert response.total_users == 2

    def test_default_group_response_has_no_token_fields():
        assert "input_tokens" not in AcquisitionGroup.model_fields
        assert "output_tokens" not in AcquisitionGroup.model_fields
        assert "total_tokens" not in AcquisitionGroup.model_fields

    def test_unmatured_checkout_window_is_partial_not_zero(query):
        response = query.get_acquisition_groups(
            start=date(2026, 9, 1), end=date(2026, 9, 8), group_by="source",
            filters=AcquisitionFilters(), now=CHECKOUT_PLUS_SIX_DAYS,
        )
        assert response.purchase_window_complete is False
        assert response.availability.available is True

- [ ] Step 2: Run the focused query tests to verify missing contracts fail.

  Run: pytest -q dashboard/backend/tests/domain/analytics/test_acquisition_queries.py

  Expected: FAIL during collection because acquisition response models and batched facts are missing.

- [ ] Step 3: Add typed response models and batched repository joins.

    class AcquisitionGroupKey(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        kind: Literal["source", "cohort"]
        value: str
        label: str

    class AcquisitionGroupFacts(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        active: bool = False
        task_completed: bool = False
        repeat: bool = False
        runs: int = Field(ge=0)
        atl_credits_settled_micro: int = Field(ge=0)
        paid_intent: bool = False
        paid: bool = False

    class AcquisitionGroup(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        group: AcquisitionGroupKey
        users: int = Field(ge=0)
        active: int = Field(ge=0)
        task_completed: int = Field(ge=0)
        repeat: int = Field(ge=0)
        runs: int = Field(ge=0)
        runs_per_active_user: float | None = Field(default=None, ge=0)
        atl_credits_settled_micro: int = Field(ge=0)
        paid_intent: int = Field(ge=0)
        paid: int = Field(ge=0)

    class AcquisitionAnalyticsResponse(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        groups: Sequence[AcquisitionGroup]
        group_by: Literal["source", "cohort"]
        purchase_window_complete: bool
        availability: PanelAvailability

  Use one batched attribution query per user chunk and one batched event/Credits query per date range. Count unique users for all user metrics. Count accepted request events for runs. For ATL Credits, sum only negative rows from `credit_llm_usage_entries`, excluding rows whose `operation_key` contains `:recovery:`; that ledger is already the platform-Credits usage ledger, so do not infer BYOK from model/token events. Never include reservations, grants, refunds, or BYOK spend. Join purchase intent only from checkout_started and paid only from settled purchased-credit ledger entries. Keep arithmetic in integer microcredits and expose atl_credits_settled_micro; UI formatting converts microcredits to the administrator's Credits display.

  Add acquisition_groups and acquisition availability to AnalyticsOverview without removing legacy fields. Reuse the existing PanelAvailability model from query_service. The group-by default is source; cohort uses the nullable cohort slug and maps null to Unknown cohort.

- [ ] Step 4: Add source/cohort/lifecycle/blocked/paid filtering to query composition.

  Extend AnalyticsMetricFilters, UserValueFilters, and value query methods with an AcquisitionFilters field. Apply source/cohort before aggregation, lifecycle and blocked against current snapshots, and paid against verified settled purchase facts. Keep the selected date range out of lifetime task_completed and repeat calculations. Preserve from/to compatibility while accepting canonical date_range=1d|1w|1m|1y shorthand; reject requests that mix shorthand with contradictory explicit dates.

- [ ] Step 5: Run query, value, and legacy Analytics tests and commit.

  Run: pytest -q dashboard/backend/tests/domain/analytics/test_acquisition_queries.py dashboard/backend/tests/domain/analytics/test_value_queries.py dashboard/backend/tests/test_admin_analytics_api.py

  Expected: PASS, including legacy overview fields and new acquisition fields.

    git add dashboard/backend/domain/analytics/value_repository.py dashboard/backend/domain/analytics/value_queries.py dashboard/backend/domain/analytics/query_service.py dashboard/backend/domain/analytics/metrics.py dashboard/backend/tests/domain/analytics/test_acquisition_queries.py
    git commit -m "feat: aggregate acquisition operations metrics"

### Task 6: Expose Filtered Overview, Users, Profile Attribution, and Audit APIs

**Files:**
- Modify: dashboard/backend/api/routers/admin_analytics.py
- Modify: dashboard/backend/domain/analytics/value_queries.py
- Create: dashboard/backend/tests/test_admin_acquisition_api.py

**Interfaces:**
- Consumes: Task 2 attribution methods, Task 5 AcquisitionAnalyticsResponse, and existing centralized require_admin.
- Produces:
  - GET /api/admin/analytics/overview?date_range=1d|1w|1m|1y&acquisition_source={source}&acquisition_cohort={slug}&lifecycle={lifecycle}&blocked=true|false&paid=true|false&include_internal=false
  - GET /api/admin/analytics/users with the same filters plus existing query/sort/pagination parameters.
  - GET /api/admin/analytics/users/{user_id} with acquisition profile fields.
  - PATCH /api/admin/analytics/users/{user_id}/attribution with {"source": "competition", "cohort": "competition-2026" | null}.

- [ ] Step 1: Write failing route and authorization tests.

    def test_overview_accepts_acquisition_filters_and_returns_groups(admin_client):
        response = admin_client.get(
            "/api/admin/analytics/overview?date_range=1w"
            "&acquisition_source=community&paid=false"
        )
        assert response.status_code == 200
        assert response.json()["acquisition_groups"][0]["group"]["value"] == "community"

    def test_invalid_source_or_cohort_is_422(admin_client):
        assert admin_client.get("/api/admin/analytics/overview?acquisition_source=ads").status_code == 422
        assert admin_client.get("/api/admin/analytics/overview?acquisition_cohort=Bad/Slug").status_code == 422

    def test_non_admin_cannot_read_or_edit_attribution(user_client):
        assert user_client.get("/api/admin/analytics/overview").status_code == 403
        assert user_client.patch(
            "/api/admin/analytics/users/7/attribution",
            json={"source": "friend", "cohort": None},
        ).status_code == 403

    def test_profile_update_returns_audit_safe_metadata(admin_client):
        response = admin_client.patch(
            "/api/admin/analytics/users/7/attribution",
            json={"source": "competition", "cohort": "competition-2026"},
        )
        assert response.status_code == 200
        body = response.json()["acquisition"]
        assert body["source"] == "competition"
        assert body["last_corrected_by_admin_id"] == 99
        assert "reason" not in body

- [ ] Step 2: Run API tests to verify parser and route failures.

  Run: pytest -q dashboard/backend/tests/test_admin_acquisition_api.py

  Expected: FAIL because the query allowlists, response member, and PATCH route do not exist.

- [ ] Step 3: Extend strict query parsing and response composition.

  Add source/cohort/date-range keys to the existing duplicate-key rejecting parser. Validate date_range against 1d|1w|1m|1y, resolve UTC start/end at request time, and pass typed AcquisitionFilters to every supported service. Keep existing from, to, billing_mode, provider, model, status, limit, offset, section, and cursor contracts working.

  Extend the profile model:

    class ValueUserProfile(AnalyticsUserProfile):
        acquisition: AcquisitionAttribution

  Implement the PATCH handler as Admin-only, validate the source/cohort pair, call the repository update in one transaction, and return the current projection plus original_source, last_corrected_at, and last_corrected_by_admin_id. Do not accept a reason field and do not expose raw audit history in the profile payload. A missing attribution row returns the default unknown projection.

- [ ] Step 4: Add partial-error isolation tests.

    def test_acquisition_failure_keeps_snapshot_and_attention_payload(admin_client, monkeypatch):
        monkeypatch.setattr("dashboard.backend.domain.analytics.value_queries.get_acquisition_groups", fail_once)
        payload = admin_client.get("/api/admin/analytics/overview").json()
        assert payload["availability"]["acquisition"]["available"] is False
        assert payload["availability"]["snapshot"]["available"] is True
        assert "users_needing_attention" in payload

- [ ] Step 5: Run the full Analytics API contract tests and commit.

  Run: pytest -q dashboard/backend/tests/test_admin_acquisition_api.py dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/domain/analytics/test_service.py

  Expected: PASS.

    git add dashboard/backend/api/routers/admin_analytics.py dashboard/backend/domain/analytics/value_queries.py dashboard/backend/tests/test_admin_acquisition_api.py
    git commit -m "feat: expose filtered acquisition analytics APIs"

### Task 7: Add Acquisition Controls, Group Table, and Drill-Down UI

**Files:**
- Modify: dashboard/frontend/app.html
- Modify: dashboard/frontend/js/admin-analytics-value.js
- Modify: dashboard/frontend/styles.css
- Create: dashboard/backend/tests/fixtures/admin_analytics/acquisition.json
- Create: dashboard/backend/tests/fixtures/admin_analytics/acquisition_partial_error.json
- Create: dashboard/backend/tests/test_admin_acquisition_frontend.py

**Interfaces:**
- Consumes: AcquisitionAnalyticsResponse, ValueUserProfile.acquisition, and the existing AdminAnalyticsValue URL/filter state.
- Produces: semantic controls with IDs adminAcquisitionSource, adminAcquisitionCohort, adminAcquisitionGroupBy, adminAcquisitionGroups, adminAcquisitionToggle, and accessible group/metric links into Users.

- [ ] Step 1: Write failing static contract tests.

    def test_acquisition_fixture_has_core_columns_and_no_tokens():
        payload = load_fixture("acquisition.json")
        assert {
            "group", "users", "active", "task_completed", "repeat", "runs",
            "runs_per_active_user", "atl_credits_settled_micro", "paid_intent", "paid",
        } <= payload["groups"][0]
        assert "input_tokens" not in walk_keys(payload)
        assert "output_tokens" not in walk_keys(payload)
        assert "total_tokens" not in walk_keys(payload)

    def test_markup_has_shared_filters_and_collapsible_panel():
        assert 'id="adminAcquisitionSource"' in APP_HTML
        assert 'id="adminAcquisitionCohort"' in APP_HTML
        assert 'id="adminAcquisitionGroupBy"' in APP_HTML
        assert 'id="adminAcquisitionToggle"' in APP_HTML
        assert 'id="adminAcquisitionGroups"' in APP_HTML

- [ ] Step 2: Run frontend contract tests to verify the markup and fixtures are absent.

  Run: pytest -q dashboard/backend/tests/test_admin_acquisition_frontend.py

  Expected: FAIL because the fixture and acquisition markup do not exist.

- [ ] Step 3: Add compact controls and the full-width table.

  Place Source, Operating cohort, Lifecycle, Blocked, and Paid controls in the existing Analytics filter row. Add a collapsed Acquisition groups disclosure below the core metrics. The default table headings are exactly:

    Group | Users | Active | Task completed | Repeat | Runs / user | ATL Credits | Paid intent | Paid

  Render counts as buttons/links only when available. A group link writes source/cohort filters and navigates to Users. A metric link adds the corresponding derived filter. Unknown values display Unknown; unavailable values display an em dash with the panel-level unavailable message. Render Credits from atl_credits_settled_micro through window.CreditFormat.

- [ ] Step 4: Add URL-backed state, independent loading, and keyboard behavior.

  Add URL keys analyticsAcquisitionSource, analyticsAcquisitionCohort, analyticsAcquisitionGroupBy, and analyticsAcquisitionOpen. Keep the existing 1D/1W/1M/1Y date selector and preserve all filters when opening Users or a Profile. Fetch acquisition data independently with a request sequence guard; a rejected acquisition request must not clear lifecycle, retention, commercial, operational, or attention panels. The disclosure button exposes aria-expanded, the table has a caption and scoped headings, and all filter controls are keyboard reachable.

- [ ] Step 5: Run frontend contracts and commit.

  Run: pytest -q dashboard/backend/tests/test_admin_acquisition_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_admin_analytics_value_frontend.py

  Expected: PASS.

    git add dashboard/frontend/app.html dashboard/frontend/js/admin-analytics-value.js dashboard/frontend/styles.css dashboard/backend/tests/fixtures/admin_analytics/acquisition.json dashboard/backend/tests/fixtures/admin_analytics/acquisition_partial_error.json dashboard/backend/tests/test_admin_acquisition_frontend.py
    git commit -m "feat: add acquisition groups analytics UI"

### Task 8: Add Attribution Display and Editing to User Analytics Profile

**Files:**
- Modify: dashboard/frontend/app.html
- Modify: dashboard/frontend/js/admin-analytics.js
- Modify: dashboard/frontend/styles.css
- Create: dashboard/backend/tests/test_admin_acquisition_profile_frontend.py

**Interfaces:**
- Consumes: GET /api/admin/analytics/users/{user_id} and PATCH /api/admin/analytics/users/{user_id}/attribution.
- Produces: Profile header fields for Source, Operating cohort, Attribution method, Attributed time, Original source, Last corrected time, and modifier ID; an Admin-only editor with source select, cohort input, Save action, and accessible status feedback.

- [ ] Step 1: Write failing profile markup and safety tests.

    def test_profile_shows_attribution_fields_and_save_control():
        for element_id in (
            "adminAnalyticsProfileSource",
            "adminAnalyticsProfileCohort",
            "adminAnalyticsProfileAttributionMethod",
            "adminAnalyticsProfileAttributionSave",
        ):
            assert f'id="{element_id}"' in APP_HTML

    def test_profile_source_has_no_reason_input_or_raw_referral_field():
        assert "correction reason" not in APP_HTML.lower()
        assert "referral_url" not in analytics_source()
        assert "invite_token" not in analytics_source()

- [ ] Step 2: Run profile frontend tests to verify the controls are absent.

  Run: pytest -q dashboard/backend/tests/test_admin_acquisition_profile_frontend.py

  Expected: FAIL because attribution fields and save behavior do not exist.

- [ ] Step 3: Render safe attribution metadata and implement the PATCH flow.

  Use textContent and existing safe DOM helpers. Display the current source prominently, show Edited only when a correction exists, and show original source, correction time, and modifier ID as audit metadata. Do not render a raw token, URL, payment payload, or arbitrary audit text. On save, disable the button, send only source/cohort, update the displayed projection from the response, announce success through aria-live, and keep the current Profile tab and parent URL intact on failure.

- [ ] Step 4: Verify Profile navigation and account-management separation.

  Add a visible Open account management link targeting the existing Users tab. Profile attribution editing is the only mutation on this page; role, Credits, provider, and account controls remain outside Analytics. Back returns to the previous filtered Users/Overview state, including source/cohort filters, pagination, and scroll restoration.

- [ ] Step 5: Run profile and Admin navigation tests and commit.

  Run: pytest -q dashboard/backend/tests/test_admin_acquisition_profile_frontend.py dashboard/backend/tests/test_admin_console_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py

  Expected: PASS.

    git add dashboard/frontend/app.html dashboard/frontend/js/admin-analytics.js dashboard/frontend/styles.css dashboard/backend/tests/test_admin_acquisition_profile_frontend.py
    git commit -m "feat: edit acquisition attribution in user profiles"

### Task 9: End-to-End Contract, Privacy, and Regression Verification

**Files:**
- Modify: dashboard/backend/tests/test_admin_acquisition_api.py
- Modify: dashboard/backend/tests/test_admin_acquisition_frontend.py
- Modify: dashboard/backend/tests/domain/analytics/test_acquisition_queries.py
- Modify: dashboard/backend/tests/domain/analytics/test_acquisition_repository.py
- Modify: docs/superpowers/specs/2026-09-06-admin-acquisition-operations-design.md only if implementation clarifies an already-approved contract

**Interfaces:**
- Consumes: all prior repository, API, and UI contracts.
- Produces: a deterministic acceptance suite proving attribution, activation, repeat use, Credits, paid intent, paid status, navigation, partial errors, and privacy boundaries agree.

- [ ] Step 1: Add the synthetic acceptance fixture.

    def test_admin_can_answer_the_acquisition_questions_with_one_fixture(admin_client):
        payload = admin_client.get(
            "/api/admin/analytics/overview?date_range=1w&acquisition_source=community"
        ).json()
        row = payload["acquisition_groups"][0]
        assert row["users"] >= row["active"] >= row["task_completed"] >= row["repeat"]
        assert row["runs_per_active_user"] is None or row["runs_per_active_user"] >= 0
        assert row["atl_credits_settled_micro"] >= 0
        assert row["paid_intent"] >= row["paid"]

- [ ] Step 2: Add privacy canaries and partial-panel assertions.

    def test_prohibited_values_never_cross_storage_api_or_html():
        prohibited = {
            "api_key", "password", "prompt", "strategy", "provider_response_body",
            "payment_payload", "referral_url", "invite_token", "input_tokens",
            "output_tokens", "total_tokens",
        }
        assert prohibited.isdisjoint(all_fixture_keys_and_frontend_literals())

  Assert that a failed acquisition query returns availability.acquisition.available == false while snapshot, attention, and Profile navigation still work. Assert that a failed attribution write leaves the existing source unchanged and does not fail an account operation.

- [ ] Step 3: Run the focused regression suite.

  Run:

    pytest -q \
      dashboard/backend/tests/domain/analytics/test_acquisition.py \
      dashboard/backend/tests/domain/analytics/test_acquisition_repository.py \
      dashboard/backend/tests/domain/analytics/test_acquisition_queries.py \
      dashboard/backend/tests/test_auth_acquisition.py \
      dashboard/backend/tests/test_credits_checkout_analytics.py \
      dashboard/backend/tests/test_admin_acquisition_api.py \
      dashboard/backend/tests/test_admin_acquisition_frontend.py \
      dashboard/backend/tests/test_admin_acquisition_profile_frontend.py

  Expected: PASS.

- [ ] Step 4: Run the broader Analytics and Credits regression suite.

  Run:

    pytest -q \
      dashboard/backend/tests/test_admin_analytics_api.py \
      dashboard/backend/tests/test_admin_analytics_frontend.py \
      dashboard/backend/tests/test_admin_analytics_value_frontend.py \
      dashboard/backend/tests/domain/analytics \
      dashboard/backend/tests/integration/test_credits_checkout_flow.py

  Expected: PASS with no changes to legacy response fields, Admin tab order, Credits behavior, or existing Profile tabs.

- [ ] Step 5: Inspect the diff and commit only implementation-owned files.

  Run:

    git status --short
    git diff --check
    git diff --stat

  Confirm that dashboard/storage/data/backtest.db, .superpowers/, work/, secrets, and analytics-layout-demo.html are not staged. Keep the implementation as the separate task commits listed above; do not squash unrelated worktree changes.

## Self-Review Checklist

- Spec coverage: Tasks 1-2 cover source/cohort validation, default unknown, bounded cohorts, attribution audit metadata, and SQLite/PostgreSQL parity. Task 3 covers controlled signup capture without inference or signup failure. Task 4 covers server-authoritative checkout intent and idempotency. Task 5 covers Active, Task completed, Repeat, Runs/user, settled ATL Credits, Paid intent, Paid, lifetime-vs-range semantics, no-token columns, and seven-day maturity. Task 6 covers shared filters, Admin authorization, Profile read/update contracts, internal exclusions, and partial errors. Tasks 7-8 cover the Overview card, Users/Profile drill-down, ARIA, keyboard navigation, URL state, visible attribution editing, and account-management separation. Task 9 covers privacy canaries and acceptance/regression behavior.
- Completeness scan: Every implementation step names a file, interface, validation rule, test command, or concrete response shape; no step relies on an unnamed future decision.
- Type consistency: AcquisitionAttribution, AcquisitionFilters, AcquisitionGroupKey, AcquisitionGroup, and AcquisitionAnalyticsResponse are introduced in Tasks 1/5 and consumed by the repository, query, API, and UI tasks with the same field names. The authoritative Credits field is consistently atl_credits_settled_micro; UI conversion is display-only.
- Scope check: This is intentionally a follow-up spanning backend contracts and UI. It does not silently claim that the existing PR 3 UI alone can provide source capture, purchase intent, or attribution editing.
