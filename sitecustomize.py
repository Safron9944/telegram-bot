"""Synchronize the verified stage-1 attestation bank with PostgreSQL.

The JSON file is the versioned source for deployment.  A content hash stored in
``settings`` makes synchronization run only once per JSON revision, so later
edits made through the admin panel are not overwritten on every restart.

``launcher.py`` imports this module before Uvicorn imports ``app``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from questions import ATTESTATION_STAGE_1_SECTION, QuestionBank
from storage import Storage


_ORIGINAL_FETCH_QUESTIONS = Storage.fetch_questions
_ORIGINAL_LOAD_ATTESTATION_STAGE_1 = QuestionBank.load_attestation_stage_1
_SEED_HASH_SETTING = "attestation_stage_1_seed_sha256"
_EXPECTED_QUESTION_COUNT = 800
_SYNC_LOCK_ID = 1_500_001


def _attestation_path() -> Path:
    raw = (os.getenv("ATTESTATION_STAGE_1_PATH") or "attestation_stage_1.json").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def _seed_items(path: Path) -> tuple[list[Any], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()

    bank = QuestionBank(str(path))
    bank.load()
    items = [
        question
        for question in bank.by_id.values()
        if question.is_valid_mcq
        and (question.section or "").strip() == ATTESTATION_STAGE_1_SECTION
    ]

    if len(items) != _EXPECTED_QUESTION_COUNT:
        raise RuntimeError(
            f"Expected {_EXPECTED_QUESTION_COUNT} valid stage-1 attestation questions "
            f"in {path}, found {len(items)}"
        )

    ids = [int(question.id) for question in items]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate stage-1 attestation question IDs found in {path}")

    return items, digest


async def _sync_attestation_stage_1(store: Storage) -> None:
    if getattr(store, "_attestation_stage_1_seeded", False):
        return

    path = _attestation_path()
    if not path.exists():
        raise RuntimeError(f"Stage-1 attestation bank not found: {path}")

    items, digest = _seed_items(path)
    ids = [int(question.id) for question in items]

    assert store.pool
    async with store.pool.acquire() as connection:
        async with connection.transaction():
            # Serialize startup synchronization when several app instances start together.
            await connection.fetchval("SELECT pg_advisory_xact_lock($1)", _SYNC_LOCK_ID)

            current_digest = await connection.fetchval(
                "SELECT value FROM settings WHERE key=$1",
                _SEED_HASH_SETTING,
            )
            if current_digest == digest:
                store._attestation_stage_1_seeded = True
                return

            existing_rows = await connection.fetch(
                "SELECT id, section FROM questions WHERE id = ANY($1::int[])",
                ids,
            )
            collisions = [
                int(row["id"])
                for row in existing_rows
                if (row["section"] or "").strip() != ATTESTATION_STAGE_1_SECTION
            ]
            if collisions:
                preview = ", ".join(str(qid) for qid in sorted(collisions)[:10])
                raise RuntimeError(
                    "Stage-1 attestation question IDs collide with another question bank: "
                    f"{preview}"
                )

            stale_rows = await connection.fetch(
                """
                SELECT id
                FROM questions
                WHERE section=$1
                  AND NOT (id = ANY($2::int[]))
                """,
                ATTESTATION_STAGE_1_SECTION,
                ids,
            )
            stale_ids = [int(row["id"]) for row in stale_rows]
            if stale_ids:
                # ``errors`` has no foreign key; clean it explicitly.
                await connection.execute(
                    "DELETE FROM errors WHERE qid = ANY($1::int[])",
                    stale_ids,
                )
                # ``question_revisions`` is removed by ON DELETE CASCADE.
                await connection.execute(
                    "DELETE FROM questions WHERE id = ANY($1::int[])",
                    stale_ids,
                )

            params = [
                (
                    int(question.id),
                    question.section or ATTESTATION_STAGE_1_SECTION,
                    question.topic or "",
                    question.ok,
                    int(question.level) if question.level is not None else None,
                    int(question.qnum) if question.qnum is not None else None,
                    question.question or "",
                    json.dumps(list(question.choices or []), ensure_ascii=False),
                    json.dumps(
                        [int(value) for value in (question.correct or [])],
                        ensure_ascii=False,
                    ),
                    json.dumps(list(question.correct_texts or []), ensure_ascii=False),
                )
                for question in items
            ]

            await connection.executemany(
                """
                INSERT INTO questions (
                    id, section, topic, ok, level, qnum, question,
                    choices, correct, correct_texts, updated_at
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb,$10::jsonb,now())
                ON CONFLICT (id) DO UPDATE SET
                    section=EXCLUDED.section,
                    topic=EXCLUDED.topic,
                    ok=EXCLUDED.ok,
                    level=EXCLUDED.level,
                    qnum=EXCLUDED.qnum,
                    question=EXCLUDED.question,
                    choices=EXCLUDED.choices,
                    correct=EXCLUDED.correct,
                    correct_texts=EXCLUDED.correct_texts,
                    updated_at=now()
                """,
                params,
            )

            await connection.execute(
                """
                INSERT INTO settings(key, value)
                VALUES ($1, $2)
                ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value
                """,
                _SEED_HASH_SETTING,
                digest,
            )

    store._attestation_stage_1_seeded = True


async def _fetch_questions_with_attestation_sync(store: Storage) -> list[dict[str, Any]]:
    await _sync_attestation_stage_1(store)
    return await _ORIGINAL_FETCH_QUESTIONS(store)


def _load_attestation_stage_1_without_duplicates(bank: QuestionBank, path: str) -> None:
    already_loaded = any(
        (question.section or "").strip() == ATTESTATION_STAGE_1_SECTION
        for question in bank.by_id.values()
    )
    if already_loaded:
        return
    _ORIGINAL_LOAD_ATTESTATION_STAGE_1(bank, path)


Storage.fetch_questions = _fetch_questions_with_attestation_sync
QuestionBank.load_attestation_stage_1 = _load_attestation_stage_1_without_duplicates

# Register dedicated admin endpoints before Uvicorn imports the FastAPI app.
import test_exam_verified_extension  # noqa: E402,F401
import admin_attestation_extension  # noqa: E402,F401
import admin_apk_import_extension  # noqa: E402,F401
