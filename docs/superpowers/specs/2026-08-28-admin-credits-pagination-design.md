# Admin Credits Pagination Design

## Goal

Make the Admin Credits `Account Management` table easier to scan by showing 25 accounts per page instead of 100.

## Scope

- Change only the Admin Credits account-management list (`/api/admin/credits/users` and its `admin-credits.js` consumer).
- Keep the existing offset pagination, search reset, range text, and Previous/Next controls.
- Leave the legacy Admin Users table and all other list sizes unchanged.

## Behavior

- The browser requests 25 accounts by default.
- The API defaults to 25 when callers omit `limit`; its existing validation range remains unchanged.
- The response `limit`, `offset`, and `total` remain authoritative.
- If accounts disappear and the current offset becomes empty, the client falls back to the last valid page.

## Verification

- Add a frontend contract assertion for the 25-account default.
- Add an API test proving the omitted-limit default is 25 and the returned range metadata matches.
- Run the Admin Credits frontend/API tests and `git diff --check`.
