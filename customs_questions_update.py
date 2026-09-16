"""One-time replacement of the customs-competencies bank from APK 1.9.1."""

from __future__ import annotations

import json

from migrate_customs_competencies import BUNDLE_PATH, _load_bundle
from storage import Storage

_CUSTOMS_BANK_VERSION = "apk-1.9.1-competencies-2610-v2"
_CUSTOMS_BANK_SETTING = "customs_question_bank_version"
_EXPECTED_QUESTIONS = 3410
_ORIGINAL_STORAGE_INIT = Storage.init

def _parse_rows() -> list[dict]:
    rows = _load_bundle(BUNDLE_PATH)
    if len(rows) != _EXPECTED_QUESTIONS:
        raise RuntimeError(f"Unexpected customs question count: {len(rows)}")
    return rows


async def _replace_customs_questions(store: Storage) -> None:
    current = await store.get_setting(_CUSTOMS_BANK_SETTING, "")
    if current == _CUSTOMS_BANK_VERSION:
        return

    rows = _parse_rows()
    assert store.pool
    async with store.pool.acquire() as con:
        async with con.transaction():
            # The migration replaces the whole main customs bank. Old mistake
            # records reference previous question IDs and therefore are reset.
            await con.execute("DELETE FROM errors")
            await con.execute("DELETE FROM questions")
            await con.executemany(
                """
                INSERT INTO questions
                    (id, section, topic, ok, level, qnum, question, choices, correct, correct_texts, updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb,$10::jsonb,now())
                """,
                [
                    (
                        row["id"], row["section"], row["topic"], row["ok"], row["level"], row["qnum"],
                        row["question"],
                        json.dumps(row["choices"], ensure_ascii=False),
                        json.dumps(row["correct"], ensure_ascii=False),
                        json.dumps(row["correct_texts"], ensure_ascii=False),
                    )
                    for row in rows
                ],
            )
            await con.execute(
                """
                INSERT INTO settings(key, value) VALUES($1, $2)
                ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value
                """,
                _CUSTOMS_BANK_SETTING,
                _CUSTOMS_BANK_VERSION,
            )


async def _storage_init_with_customs_bank(self: Storage):
    await _ORIGINAL_STORAGE_INIT(self)
    await _replace_customs_questions(self)


Storage.init = _storage_init_with_customs_bank
