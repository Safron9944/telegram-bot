# Admin Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace the live admin panel with the approved customer workspace,
content editor, and protected purchase records.

**Architecture:** Immutable purchase facts and separate manual grants determine
effective access. A shared admin shell mounts a customer master/detail workspace,
content workspace, or system workspace around existing specialized editors.

**Tech Stack:** FastAPI, aiogram, asyncpg/PostgreSQL, ES modules, scoped CSS.

## Task 1: Purchase storage and effective access

Files: `storage.py`, `access.py`, `sections.py`, `app.py`, optional focused
`commerce.py`; tests `tests/test_commerce.py` and existing access tests.

- [x] Add failing tests for `access_tier` selecting full access over a later
  attestation purchase and for paid sections surviving a false admin override.
- [x] Run `python -m unittest discover -s tests -p test_commerce.py -v`; confirm
  those tests fail because purchase provenance is absent.
- [x] Add payment, manual grant, and admin event tables with guarded migrations.
  Purchase identity is unique, unknown historical amounts stay null, and payment
  history has no cascading deletion from users.
- [x] Record successful Telegram payments and entitlement atomically, preserving
  existing legacy rights. Return paid/manual/legacy provenance from get_user.
- [x] Make access checks and visibility resolve paid rights first, without
  exposing unpublished draft banks to full-package holders.
- [x] Run commerce, access, section, and admin-control tests; confirm duplicate
  payments, refunds, preservation of legacy data, and package precedence.

## Task 2: Administrative commerce API

Files: backend files above, `tests/test_commerce.py`.

- [x] Write failing service tests: paid user + `enabled=False` manual grant
  keeps the purchase; non-admin receives 403; missing user receives 404;
  confirmed external payment with repeated request_id creates one record.
- [x] Implement the exact routes and payloads in the design's API section.
  Each mutating route returns `admin_user_detail` with authoritative state.
- [x] Add paged user summaries, payment search, and paged admin event history.
- [x] Add paged Stars reconciliation; distinguish incoming invoice payments
  from refunds and ignore unrelated transaction types and unknown payloads.
- [x] Protect legacy access routes and deletion of users with purchase evidence.
- [x] Run targeted tests and perform a spec compliance review before UI integration.

## Task 3: Customer workspace and shared shell

Files: new `static/js/admin_workspace.js`, `static/js/admin_customer_view.js`,
`static/js/admin_navigation.js`, `static/js/app.js`, `static/js/core/state.js`,
`static/styles/admin.css`; Node/browser tests.

- [x] Write browser regressions for selected customer state, paid purchase
  surviving gift revocation, paid-history rows, and failed writes retaining UI.
- [x] Replace admin landing with the customer workspace. Desktop shows list and
  inspector; mobile shows one at a time. Persistent navigation has Clients,
  Content, and System; specialized screens inherit it.
- [x] Render purchases, gifts, unknown legacy rights, and expandable history.
  Add grant/external payment/recovery review forms with reason, pending lock,
  retry-safe request_id and authoritative API refresh. No paid-access toggle.
- [x] Integrate paged payment history and explicit Stars reconciliation progress.
- [x] Preserve search/filter/page state, ignore stale customer responses, and
  prevent actions leaking to the next selected customer.

## Task 4: Content and system spaces

Files: new `static/js/admin_content_workspace.js`, existing section/settings
and editor modules, shared stylesheet and navigation.

- [x] Place material list and selected material settings in one workspace.
  Keep question editor, topics, ordering, deletion and import reachable.
- [x] Include full-package price alongside individual section prices.
- [x] Make settings/messages/activity system tools with consistent navigation.
- [x] Update all changed module and stylesheet cache keys to
  `20261014-admin-workspace-01`, including imports from existing modules.

## Task 5: Verification and documentation

Files: `tests/admin_workspace.browser.mjs`, `tests/fixtures/admin_workspace_preview.js`,
new focused tests, `PRODUCT.md`, `DESIGN.md`, these specifications.

Release verification: 230 Python tests and 13 Node tests pass. Browser checks
cover 360/1280 px, cross-view searches, keyboard focus, pending reentry,
delete/catalog selection, and partial Telegram reconciliation. Native PostgreSQL
SQL was additionally exercised with isolated PGlite through a local adapter;
the production asyncpg transport and production database were not used.
Spec and independent code reviews passed; visual finish disposition: `ship`.

- [x] Run `python -m unittest discover -s tests -v` with bytecode disabled.
- [x] Run meaningful Node helper tests and browser scenarios at 360/1280 px.
  Test purchase sources, gift revocation, external receipt, errors and races,
  search, invoice history, catalog edits, specialized editors, and system.
- [x] Inspect batched desktop/mobile captures, fix material findings once, and
  verify fixes. Review spec compliance before the independent code quality pass.
- [x] Document actual schema, migration limitations, operational reconciliation,
  and implemented visual/navigation rules. No fabricated historic payment facts.

## Task 6: Commit, push, merge, synchronize

- [ ] Check tracked diff and ensure no runtime files, secrets, or bytecode enter
  the commit. Use the git-pushing smart_commit script with accurate Codex footer.
- [ ] Push `codex/admin-workspace`, create a PR with concrete behavior and tests,
  attach the PR, inspect available checks, and merge the tested revision.
- [ ] Fast-forward local main to the merge commit and verify it matches origin/main.

## Review

All design requirements map to Tasks 1–5. Purchase facts, rights, and display
are separated; read-only legacy provenance never claims payment. Changes ship
together because the UI requires the new provenance API, and old endpoints
must protect purchases before the operator reaches the new panel.
