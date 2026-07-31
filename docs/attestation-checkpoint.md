# First-stage attestation extraction checkpoint

Updated: 2026-07-31 10:07 (Europe/Kyiv)

## Stop point

Work was stopped immediately at the user's request during Task 12, Step 3: extracting and verifying the Constitution section from physical PDF pages 3–30. Step 3 is not complete. Do not start pages 31–128 until the Constitution checkpoint is complete.

## Saved state

- Branch: `main`.
- Source PDF: `C:\Users\Lucky\Desktop\2048\додаток_до_наказу_3125_Перелік_питань.pdf`.
- All 28 source-page PNG files for physical pages 3–30 are saved in `tmp\attestation-ocr`.
- One completed OCR cache file is saved: `tmp\attestation-ocr\page-017.json`. It contains six synthetic markers numbered 109–114; treat that numbering as unverified and regenerate the file if it does not agree with the visible source numbers.
- The isolated OCR runtime is `C:\ocr3125\Scripts\python.exe`.
- EasyOCR models are in `C:\ocr3125\models`; set `EASYOCR_MODULE_PATH=C:\ocr3125\models` and `PYTHONIOENCODING=utf-8`.
- The PDF table-structure recovery now detects row rules, the number column, answer bullets, wrapped lines, and bold-answer ink scores from pixels.
- On physical page 3, the structure check found question numbers 1–8 and exactly 30 visible answer bullets (question 8 continues on page 4).
- A full-page test grouped the page into 8 questions. The later bullet-column correction removed the false fifth-choice issue seen in that test.
- The focused extractor/quality tests passed: 12 tests.

## Resume sequence

1. Run the focused tests again before resuming OCR.
2. Inspect `page-017.json`; reuse it only if its synthetic numbers and choices still pass the current checks.
3. Continue OCR for pages 3–16 and 18–30. Use at most two EasyOCR workers on this 16 GB laptop; four workers caused memory pressure and were stopped.
4. Build candidates only for the `constitution` section and account for every source number 1–200.
5. Cross-check exact/fuzzy matches against `questions_flat.json`, `test_exam_questions.json`, and the configured database when available.
6. Put any truncated, low-confidence, structurally ambiguous, missing, or answer-conflicting item in the review queue; never publish it as verified.
7. Save Constitution-only checkpoint outputs under `data\checkpoints`: verified questions, review candidates, and the quality/accounting report.
8. Verify that all numbers 1–200 are represented exactly once across verified plus review records, then stop. Do not proceed to the Civil Service section in the same resumed step.

## Uncommitted implementation at stop

- `scripts/extract_attestation_questions.py`: expected-count accounting, duplicate consolidation, pixel-based table structure, deterministic line segmentation, page-number OCR, and structured recognition.
- `attestation_quality.py`: valid short Ukrainian words no longer trigger false suspicious-break warnings.
- `tests/test_attestation_extractor.py`: regression coverage for a bullet detected separately from its answer text.

No Constitution verified/review/report checkpoint JSON was produced yet.
