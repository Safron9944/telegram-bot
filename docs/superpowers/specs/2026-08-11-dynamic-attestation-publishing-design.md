# Dynamic Attestation Publishing Design

## Goal

Allow an administrator to turn a question bank extracted from an APK into a new published attestation section without changing application code. The first expected use is “Атестація посадових осіб — 2 етап”, while the same workflow must support later stages and other tests.

## Approved workflow

1. The administrator uploads an APK, XAPK, or APKS and selects a discovered question bank.
2. The existing importer decrypts and parses the bank, then shows its sections, questions, answer choices, correct answers, shuffle restrictions, duplicates, and validation errors.
3. The administrator enters the public section title and presses “Опублікувати”. Publication is never automatic immediately after parsing.
4. The server stores the bank durably in PostgreSQL and makes it available in the running application without a deployment or restart.
5. The Mini App adds a home-screen card for the published bank and uses the same section, 50-question block, random block, answer, feedback, result, mistake, and subscription behavior as Stage 1.
6. The admin question editor can browse and correct questions in every published bank.
7. Publishing the same source bank again replaces that bank atomically, records a backup/revision, and does not create a duplicate home card.

## Architecture

### Persistent catalog

Add PostgreSQL tables for published assessment banks, their questions, and immutable publication revisions. A bank has a stable slug, public title, source filename/hash, status, display order, access policy, timestamps, and question count. Questions retain their section/topic, source number, choices, correct indexes, and `shuffle_choices` flag.

Stage 1 remains available and is represented through the same runtime catalog. Its existing bundled JSON remains the safe bootstrap fallback until it has a database record. Dynamic banks are loaded from PostgreSQL during startup and immediately after a successful publication.

### Runtime question model

Generalize the Stage 1-only index into a collection keyed by bank slug. Each bank exposes sections and ordered question IDs. IDs are allocated by PostgreSQL and remain globally unique, so saved mistakes and active sessions continue to resolve questions correctly.

The generic start endpoint receives a bank slug, section, and block. It verifies publication state and access, builds the selected ordered block of up to 50 questions or a random set of up to 50, and starts the existing learning session with metadata identifying the bank. Block labels are generated from the actual section size, so banks do not need exactly 200 questions per section.

Choice rendering continues to use the current choice-order mechanism. Choices are shuffled only when `shuffle_choices` is true; the stored correct indexes stay mapped through that order. Questions marked as order-sensitive keep their original ordering.

### Publication service

The importer session remains temporary until publication. The publish endpoint validates administrator access, title, bank contents, unique source identity, four choices, correct-answer indexes, and non-empty sections. Invalid banks are rejected with a readable report.

Publication runs in one database transaction: lock the existing bank identity, save a revision of the previous version when present, upsert metadata, replace its questions, and commit. Only after commit is the in-memory catalog refreshed. A failure leaves the previous published version untouched.

### Mini App

The APK preview gains a title field and a primary “Опублікувати” button. After success it displays the created section and offers to open it.

The user home screen renders published assessment cards from the bootstrap catalog rather than hard-coding Stage 1. Selecting a card opens the same section/block screen currently used by Stage 1. Hidden or draft banks are omitted.

The admin editor receives a bank selector and reuses its current search, pagination, editing, revision, export, and send-to-chat behavior for the selected bank.

## Access and visibility

Every newly published assessment bank defaults to the same subscription requirement as Stage 1: access for the attestation/full tiers, no access during a trial. Publication also creates a visible home card. Existing global attestation visibility still hides all assessment cards when disabled.

## Error handling

- Unsupported encryption or malformed banks remain preview/import errors and cannot be published.
- Empty titles, empty sections, missing correct answers, and invalid indexes produce field-level messages.
- Duplicate source banks are treated as updates, not new cards.
- Concurrent publications of the same bank are serialized by a database lock.
- If runtime refresh fails after the transaction, the server reloads the catalog from PostgreSQL; persisted data remains valid.

## Testing

Tests cover schema creation, first publication, replacement with revision, rollback on invalid content, dynamic catalog serialization, access enforcement, generic block selection, shuffle and no-shuffle correctness, admin editing, mobile publish controls, and backward compatibility for Stage 1. The full existing test suite must pass before pushing to `main`.

## Out of scope

This version does not infer or break an entirely new unknown encryption algorithm. It publishes any bank that the importer can successfully parse. Support for a new encryption format remains an importer adapter task.
