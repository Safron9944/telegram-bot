"""Persist stage-1 attestation questions in PostgreSQL.

The application historically loaded ``attestation_stage_1.json`` only into
memory. The existing admin editor saves through the ``questions`` table, so
those questions could be opened but not updated. This bootstrap keeps the
JSON file as an insert-only seed and makes the database the source of truth.

Python imports ``sitecustomize`` automatically during normal interpreter
startup, before Uvicorn imports ``app``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from questions import ATTESTATION_STAGE_1_SECTION, QuestionBank
from storage import Storage


_ORIGINAL_FETCH_QUESTIONS = Storage.fetch_questions
_ORIGINAL_LOAD_ATTESTATION_STAGE_1 = QuestionBank.load_attestation_stage_1


def _attestation_path() -> Path:
    raw = (os.getenv("ATTESTATION_STAGE_1_PATH") or "attestation_stage_1.json").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def _seed_items(path: Path) -> list[Any]:
    bank = QuestionBank(str(path))
    bank.load()
    return [
        question
        for question in bank.by_id.values()
        if question.is_valid_mcq
        and (question.section or "").strip() == ATTESTATION_STAGE_1_SECTION
    ]


async def _seed_attestation_stage_1(store: Storage) -> None:
    if getattr(store, "_attestation_stage_1_seeded", False):
        return

    path = _attestation_path()
    if not path.exists():
        return

    items = _seed_items(path)
    if not items:
        raise RuntimeError(f"No valid stage-1 attestation questions found in {path}")

    assert store.pool
    ids = [int(question.id) for question in items]

    async with store.pool.acquire() as connection:
        async with connection.transaction():
            existing_rows = await connection.fetch(
                "SELECT id, section FROM questions WHERE id = ANY($1::int[])",
                ids,
            )
            existing_by_id = {int(row["id"]): (row["section"] or "") for row in existing_rows}

            collisions = [
                qid
                for qid, section in existing_by_id.items()
                if section.strip() != ATTESTATION_STAGE_1_SECTION
            ]
            if collisions:
                preview = ", ".join(str(qid) for qid in sorted(collisions)[:10])
                raise RuntimeError(
                    "Stage-1 attestation question IDs collide with another question bank: "
                    f"{preview}"
                )

            params = []
            for question in items:
                if int(question.id) in existing_by_id:
                    continue
                params.append(
                    (
                        int(question.id),
                        question.section or ATTESTATION_STAGE_1_SECTION,
                        question.topic or "",
                        question.ok,
                        int(question.level) if question.level is not None else None,
                        int(question.qnum) if question.qnum is not None else None,
                        question.question or "",
                        json.dumps(list(question.choices or []), ensure_ascii=False),
                        json.dumps([int(value) for value in (question.correct or [])], ensure_ascii=False),
                        json.dumps(list(question.correct_texts or []), ensure_ascii=False),
                    )
                )

            if params:
                await connection.executemany(
                    """
                    INSERT INTO questions (
                        id, section, topic, ok, level, qnum, question,
                        choices, correct, correct_texts, updated_at
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb,$10::jsonb,now())
                    ON CONFLICT (id) DO NOTHING
                    """,
                    params,
                )

    store._attestation_stage_1_seeded = True


async def _fetch_questions_with_attestation_seed(store: Storage) -> list[dict[str, Any]]:
    await _seed_attestation_stage_1(store)
    return await _ORIGINAL_FETCH_QUESTIONS(store)


def _load_attestation_stage_1_without_duplicates(bank: QuestionBank, path: str) -> None:
    already_loaded = any(
        (question.section or "").strip() == ATTESTATION_STAGE_1_SECTION
        for question in bank.by_id.values()
    )
    if already_loaded:
        return
    _ORIGINAL_LOAD_ATTESTATION_STAGE_1(bank, path)


Storage.fetch_questions = _fetch_questions_with_attestation_seed
QuestionBank.load_attestation_stage_1 = _load_attestation_stage_1_without_duplicates
