"""Apply reviewed OCR corrections to stage-1 attestation questions.

The correction bundles contain only high-confidence scan/OCR fixes. They are
applied both to JSON seed objects and once to existing PostgreSQL rows. The
six medium-confidence cases from the review are intentionally excluded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from questions import ATTESTATION_STAGE_1_SECTION

MIGRATION_KEY = "attestation-stage-1-ocr-2026-08-03-148"
FIX_GLOB = "attestation_stage_1_ocr_fixes_*.json"


def _load_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        return list(parsed) if isinstance(parsed, list) else []
    return []


def load_fixes(base_dir: Path | None = None) -> list[dict[str, Any]]:
    root = base_dir or Path(__file__).resolve().parent
    fixes: list[dict[str, Any]] = []
    for path in sorted(root.glob(FIX_GLOB)):
        data = json.loads(path.read_text(encoding="utf-8"))
        fixes.extend(data.get("fixes") or [])
    if len(fixes) != 148:
        raise RuntimeError(f"Expected 148 reviewed OCR fixes, found {len(fixes)}")
    ids = [int(item["id"]) for item in fixes]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate question IDs in OCR correction bundles")
    return fixes


def _replace_all(text: str, replacements: Iterable[list[str]]) -> tuple[str, bool]:
    result = text or ""
    changed = False
    for pair in replacements:
        if not isinstance(pair, list) or len(pair) != 2:
            raise RuntimeError(f"Invalid OCR replacement: {pair!r}")
        old, new = str(pair[0]), str(pair[1])
        if old in result:
            result = result.replace(old, new)
            changed = True
    return result, changed


def apply_fix_to_content(
    *,
    question: str,
    choices: list[str],
    correct_texts: list[str],
    fix: dict[str, Any],
) -> tuple[str, list[str], list[str], bool]:
    new_question = question or ""
    new_choices = [str(value) for value in choices]
    new_correct_texts = [str(value) for value in correct_texts]
    changed = False

    expected_question = fix.get("question_before")
    if expected_question is not None and new_question == str(expected_question):
        new_question = str(fix.get("question_after") or "")
        changed = True

    new_question, replaced = _replace_all(
        new_question,
        fix.get("question_replacements") or [],
    )
    changed = changed or replaced

    expected_answer = fix.get("answer_before")
    if expected_answer is not None:
        answer_after = str(fix.get("answer_after") or "")
        for index, value in enumerate(new_choices):
            if value == str(expected_answer):
                new_choices[index] = answer_after
                changed = True
        for index, value in enumerate(new_correct_texts):
            if value == str(expected_answer):
                new_correct_texts[index] = answer_after
                changed = True

    answer_replacements = fix.get("answer_replacements") or []
    for index, value in enumerate(new_choices):
        new_choices[index], replaced = _replace_all(value, answer_replacements)
        changed = changed or replaced
    for index, value in enumerate(new_correct_texts):
        new_correct_texts[index], replaced = _replace_all(value, answer_replacements)
        changed = changed or replaced

    return new_question, new_choices, new_correct_texts, changed


def apply_fixes_to_questions(questions: Iterable[Any]) -> int:
    fixes_by_id = {int(item["id"]): item for item in load_fixes()}
    changed_count = 0
    for question in questions:
        fix = fixes_by_id.get(int(question.id))
        if not fix:
            continue
        if (question.section or "").strip() != ATTESTATION_STAGE_1_SECTION:
            raise RuntimeError(f"OCR fix points to non-attestation question {question.id}")
        if (question.topic or "").strip() != str(fix.get("topic") or "").strip():
            raise RuntimeError(f"OCR fix topic mismatch for question {question.id}")
        if int(question.qnum or 0) != int(fix.get("qnum") or 0):
            raise RuntimeError(f"OCR fix number mismatch for question {question.id}")

        new_question, new_choices, new_correct_texts, changed = apply_fix_to_content(
            question=question.question,
            choices=list(question.choices or []),
            correct_texts=list(question.correct_texts or []),
            fix=fix,
        )
        if changed:
            question.question = new_question
            question.choices = new_choices
            question.correct_texts = new_correct_texts
            changed_count += 1
    return changed_count


async def apply_fixes_to_store(store: Any) -> int:
    if getattr(store, "_attestation_ocr_fixes_applied", False):
        return 0

    assert store.pool
    fixes = load_fixes()
    ids = [int(item["id"]) for item in fixes]
    fixes_by_id = {int(item["id"]): item for item in fixes}
    changed_count = 0

    async with store.pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS content_migrations (
                    migration_key TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    changed_count INT NOT NULL DEFAULT 0
                )
                """
            )
            await connection.fetchval(
                "SELECT pg_advisory_xact_lock(hashtext($1))",
                MIGRATION_KEY,
            )
            if await connection.fetchval(
                "SELECT 1 FROM content_migrations WHERE migration_key=$1",
                MIGRATION_KEY,
            ):
                store._attestation_ocr_fixes_applied = True
                return 0

            rows = await connection.fetch(
                """
                SELECT id, section, topic, qnum, question, choices, correct, correct_texts
                FROM questions
                WHERE id = ANY($1::int[])
                """,
                ids,
            )

            for row in rows:
                qid = int(row["id"])
                fix = fixes_by_id[qid]
                if (row["section"] or "").strip() != ATTESTATION_STAGE_1_SECTION:
                    raise RuntimeError(f"OCR fix points to non-attestation DB row {qid}")
                if (row["topic"] or "").strip() != str(fix.get("topic") or "").strip():
                    raise RuntimeError(f"OCR fix topic mismatch in DB for question {qid}")
                if int(row["qnum"] or 0) != int(fix.get("qnum") or 0):
                    raise RuntimeError(f"OCR fix number mismatch in DB for question {qid}")

                choices = [str(value) for value in _load_json_list(row["choices"])]
                correct = [int(value) for value in _load_json_list(row["correct"])]
                correct_texts = [str(value) for value in _load_json_list(row["correct_texts"])]
                before = {
                    "question": row["question"] or "",
                    "choices": choices,
                    "correct": correct,
                    "correct_texts": correct_texts,
                }

                new_question, new_choices, new_correct_texts, changed = apply_fix_to_content(
                    question=before["question"],
                    choices=choices,
                    correct_texts=correct_texts,
                    fix=fix,
                )
                if not changed:
                    continue

                after = {
                    "question": new_question,
                    "choices": new_choices,
                    "correct": correct,
                    "correct_texts": new_correct_texts,
                }
                version = await connection.fetchval(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM question_revisions WHERE qid=$1",
                    qid,
                )
                await connection.execute(
                    """
                    INSERT INTO question_revisions(qid, version, changed_by, before, after)
                    VALUES($1, $2, $3, $4::jsonb, $5::jsonb)
                    """,
                    qid,
                    int(version),
                    "system:ocr-2026-08-03",
                    json.dumps(before, ensure_ascii=False),
                    json.dumps(after, ensure_ascii=False),
                )
                await connection.execute(
                    """
                    UPDATE questions
                    SET question=$2,
                        choices=$3::jsonb,
                        correct_texts=$4::jsonb,
                        updated_at=now()
                    WHERE id=$1
                    """,
                    qid,
                    new_question,
                    json.dumps(new_choices, ensure_ascii=False),
                    json.dumps(new_correct_texts, ensure_ascii=False),
                )
                changed_count += 1

            await connection.execute(
                """
                INSERT INTO content_migrations(migration_key, changed_count)
                VALUES($1, $2)
                """,
                MIGRATION_KEY,
                changed_count,
            )

    store._attestation_ocr_fixes_applied = True
    return changed_count
