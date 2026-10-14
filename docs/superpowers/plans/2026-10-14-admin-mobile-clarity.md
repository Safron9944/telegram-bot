# Admin Mobile Clarity Implementation Plan

**Goal:** Correct the rejected mobile admin interface using the app palette,
compact readable records and Telegram-safe navigation.

**Architecture:** Keep purchase/grant persistence and existing workflows.
Add factual display fields to commerce summaries, pure client-row rendering,
and responsive shell styling. SDK inset CSS variables drive layout directly.

**Tech Stack:** ES modules, scoped CSS, existing FastAPI commerce summaries.

- [x] Add failing summary and browser regressions for the user screenshot:
  spacing, palette, single-column rows, safe areas, fixed navigation, honest copy.
- [x] Extend `commerce_summary`, add pure `clientListSummary` and row renderer,
  integrate mobile filter and compact shared shell across spaces/editors.
- [x] Replace rejected palette and mobile scaffolding; maintain focus,
  payment review, pending state, and desktop master/detail.
- [x] Run affected Python/Node and browser scenarios; inspect one batched set
  at mobile/desktop, resolve material findings together and confirm once.
- [x] Fresh independent visual review using the user's screenshot as evidence,
  targeted code review and update DESIGN.md/sidecar from the resulting UI.
Release sequence (authorized): commit/push, create and attach PR, inspect checks, merge tested head,
  fast-forward local main and verify it matches GitHub.

Verification: 231 Python tests and 18 Node tests pass. Mobile geometry, inset
changes, sources and controls verified at 360/390/640/1280 px, including
landscape; existing payment, gift-revoke, stale-response, focus, selection, and
partial-sync browser scenarios pass. Code review passed; visual review's sole
mixed-access finding was scored resolved, disposition ship.
