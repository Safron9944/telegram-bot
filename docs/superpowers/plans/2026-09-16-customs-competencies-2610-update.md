# Customs Competencies 2610 Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the full 2,610-question customs competency block from the APK export while keeping a valid 800-question legislation block and deploying the result through `main`.

**Architecture:** Normalize the supplied APK JSON into the repository's flat question schema, store one compressed XZ bundle, and make startup load that bundle directly. Remove the obsolete fragmented Base85 payload so there is one authoritative bundled source.

**Tech Stack:** Python 3, JSON, LZMA/XZ, unittest, Git.

---

### Task 1: Lock the expected bundle behavior

**Files:**
- Modify: `tests/test_customs_competencies_bundle.py`

- [x] Add a focused assertion that the bundle has 3,410 unique IDs and exactly 2,610 unique competency keys.
- [x] Run `python -m unittest tests.test_customs_competencies_bundle -v` and confirm it fails because the current XZ payload is invalid.

### Task 2: Replace the competency payload and loader

**Files:**
- Modify: `customs_questions_update.py`
- Modify: `data/customs_competencies_1_9_1.json.xz`
- Modify: `questions_flat.json`
- Delete: `data/customs_testmsmo_v13.part*.b85`

- [x] Normalize the 800 legislation questions from `customs_all_questions_3410.json` and all 2,610 competency questions from `customs_competencies_2610.json` into `id`, `section`, `topic`, `ok`, `level`, `qnum`, `question`, `choices`, `correct`, and `correct_texts`.
- [x] Preserve both correct indexes for the one APK question containing the same correct answer twice.
- [x] Write the normalized list to `questions_flat.json` and its compact XZ form to the bundled data path.
- [x] Load startup rows from the XZ bundle and bump the bank version so deployed databases refresh.
- [x] Remove all obsolete Base85 fragments.

### Task 3: Verify and publish

**Files:**
- Test: `tests/test_customs_competencies_bundle.py`

- [x] Run only the focused bundle test and a syntax compilation of changed Python files.
- [x] Confirm 3,410 total questions, 800 legislation questions, 2,610 competency questions, 17 competency modules, and no duplicate IDs or logical keys.
- [ ] Commit with a conventional message and push `main` to `origin` using the repository's smart commit workflow.
