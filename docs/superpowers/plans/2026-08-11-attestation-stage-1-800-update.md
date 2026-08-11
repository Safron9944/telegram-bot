# Attestation Stage 1 800-Question Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stage-1 attestation bank with the supplied 800-question JSON while preserving correct-answer grading after option shuffling.

**Architecture:** Keep the existing single-file attestation source and replace it atomically rather than merging records. Reuse the existing `QuestionBank` loader and choice-order mapping, then validate both source mappings and shuffled display mappings.

**Tech Stack:** Python 3, JSON, `unittest`, Git

---

### Task 1: Replace the attestation source

**Files:**
- Modify: `attestation_stage_1.json`

- [ ] **Step 1: Validate the supplied source before copying**

Run a Python validation that asserts 800 records, unique IDs, four choices per question, one 1-based correct index, and an exact `correct_texts` match.

- [ ] **Step 2: Replace the tracked source atomically**

Copy `C:\Users\Lucky\Desktop\bot\attestation_stage_1_from_testms_800_2026-08-11.json` over `attestation_stage_1.json`; do not merge records or retain a second runtime file.

- [ ] **Step 3: Confirm exact replacement**

Compare SHA-256 hashes and parsed JSON content between the supplied and tracked files. Expected: identical hashes and content.

### Task 2: Verify loading, grading, and shuffling

**Files:**
- Test: `tests/test_attestation_stage_1.py`
- Test: `tests/test_choice_order.py`

- [ ] **Step 1: Load the replacement through `QuestionBank`**

Assert that all 800 attestation questions load without ID conflicts with `questions_flat.json`.

- [ ] **Step 2: Exercise shuffled answer mappings**

For every question and multiple deterministic shuffles, map the displayed answer back to its source index and assert it equals the JSON `correct` index and `correct_texts` value.

- [ ] **Step 3: Run the complete suite**

Run `python -m unittest discover -s tests -v`. Expected: all tests pass with zero failures.

### Task 3: Publish the update

**Files:**
- Modify: `attestation_stage_1.json`
- Create: `docs/superpowers/plans/2026-08-11-attestation-stage-1-800-update.md`

- [ ] **Step 1: Inspect the final diff**

Run `git diff --check`, inspect `git diff --stat`, and confirm no unrelated files are modified.

- [ ] **Step 2: Commit and push**

Create a conventional data-update commit and push the clean verified `main` to `origin/main`.
