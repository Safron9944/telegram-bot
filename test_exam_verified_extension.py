"""Import the verified 2026-08-14 test-exam questions without duplicates.

The regular application seeds ``test_exam_questions.json`` during startup.
This extension patches that importer so:
- duplicates are detected by normalized question text, not only by number/source;
- the verified 2026-08-14 supplement is imported after the bundled bank;
- known equivalent questions are not duplicated;
- the one verified answer conflict (temporary import of pets) corrects the old row.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from storage import Storage


_VERIFIED_PATH = Path(__file__).resolve().parent / "customs_test_questions_answers_verified.json"
_VERIFIED_SOURCE = "Перевірено 14.08.2026"
_SKIP_VERIFIED_NUMBERS = {45, 59}

_PET_LEGACY_QUESTION = (
    "Домашні тварини (до 3 ссавців, 6 птахів, 20 риб) без забезпечення "
    "сплати митних платежів дозволені для:"
)
_PET_CORRECT_ANSWER = "тимчасового ввезення на митну територію України"
_PET_JUSTIFICATION = (
    "Перевірено 14.08.2026 за офіційною інформацією Держмитслужби України: "
    "для домашніх тварин у зазначеній кількості дозволяється тимчасове ввезення "
    "без надання забезпечення сплати митних платежів."
)

_ORIGINAL_IMPORT_TEST_EXAM_QUESTIONS = Storage.import_test_exam_questions


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("’", "'").replace("`", "'").replace("ʼ", "'")
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _verified_items() -> list[dict[str, Any]]:
    if not _VERIFIED_PATH.exists():
        return []

    payload = json.loads(_VERIFIED_PATH.read_text(encoding="utf-8"))
    raw_items = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(raw_items, list):
        return []

    items: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        number = raw.get("number")
        try:
            number_int = int(number)
        except (TypeError, ValueError):
            continue

        # №45 already exists with equivalent wording and the same answer.
        # №59 is represented by an older equivalent row whose answer is corrected below.
        if number_int in _SKIP_VERIFIED_NUMBERS:
            continue

        question = str(raw.get("question") or "").strip()
        correct_answer = str(raw.get("correct_answer") or "").strip()
        if not question or not correct_answer:
            continue

        items.append(
            {
                "num": f"№ {number_int}",
                "module": None,
                "question": question,
                "correct_answer": correct_answer,
                "justification": "",
                "source": _VERIFIED_SOURCE,
            }
        )
    return items


async def _import_without_question_duplicates(
    store: Storage,
    items: list[dict[str, Any]],
) -> int:
    assert store.pool
    inserted = 0

    async with store.pool.acquire() as con:
        existing_rows = await con.fetch(
            """
            SELECT id, num, source, question, correct_answer
            FROM test_exam_questions
            """
        )
        existing_questions = {
            _normalize_text(row["question"])
            for row in existing_rows
            if _normalize_text(row["question"])
        }
        existing_keys = {
            (str(row["num"] or ""), str(row["source"] or ""))
            for row in existing_rows
        }

        for raw in items:
            if not isinstance(raw, dict):
                continue

            question = str(raw.get("question") or "").strip()
            if not question:
                continue

            normalized_question = _normalize_text(question)
            key = (str(raw.get("num") or ""), str(raw.get("source") or ""))

            if normalized_question in existing_questions or key in existing_keys:
                continue

            await con.execute(
                """
                INSERT INTO test_exam_questions
                    (num, module, question, correct_answer, justification, source)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                raw.get("num"),
                raw.get("module"),
                question,
                str(raw.get("correct_answer") or "").strip(),
                str(raw.get("justification") or "").strip(),
                str(raw.get("source") or "").strip(),
            )
            inserted += 1
            existing_questions.add(normalized_question)
            existing_keys.add(key)

    return inserted


async def _apply_verified_answer_corrections(store: Storage) -> int:
    assert store.pool
    async with store.pool.acquire() as con:
        result = await con.execute(
            """
            UPDATE test_exam_questions
            SET correct_answer=$2,
                justification=$3
            WHERE question=$1
              AND correct_answer IS DISTINCT FROM $2
            """,
            _PET_LEGACY_QUESTION,
            _PET_CORRECT_ANSWER,
            _PET_JUSTIFICATION,
        )
    try:
        return int(result.rsplit(" ", 1)[-1])
    except (TypeError, ValueError):
        return 0


async def _import_test_exam_questions_with_verified(
    store: Storage,
    items: list[dict[str, Any]],
) -> int:
    inserted = await _import_without_question_duplicates(store, items)

    if not getattr(store, "_verified_test_exam_seed_loaded", False):
        inserted += await _import_without_question_duplicates(store, _verified_items())
        await _apply_verified_answer_corrections(store)
        store._verified_test_exam_seed_loaded = True

    return inserted


Storage.import_test_exam_questions = _import_test_exam_questions_with_verified
