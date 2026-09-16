"""One-time replacement of the customs-competencies bank from APK 1.9.1."""

from __future__ import annotations

import base64
import bz2
import hashlib
import json
import re
from pathlib import Path

from apk_importer.testms import parse_testms_bank
from storage import Storage
from utils import OK_TITLES

_CUSTOMS_BANK_VERSION = "testmsmo-13-apk-1.9.1"
_CUSTOMS_BANK_SETTING = "customs_question_bank_version"
_CUSTOMS_BANK_PART_GLOB = "customs_testmsmo_v13.part*.b85"
_EXPECTED_QUESTIONS = 3410
_EXPECTED_LAW_QUESTIONS = 800
_EXPECTED_OK_QUESTIONS = 2610
_ORIGINAL_STORAGE_INIT = Storage.init

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_to_int(value: str) -> int:
    value = (value or "").strip().upper()
    total = previous = 0
    for char in reversed(value):
        current = _ROMAN_VALUES.get(char)
        if current is None:
            raise ValueError(f"Unsupported TestMS section code: {value}")
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total


def _bank_plaintext() -> tuple[str, str]:
    base = Path(__file__).resolve().parent / "data"
    parts = sorted(base.glob(_CUSTOMS_BANK_PART_GLOB))
    if not parts:
        raise RuntimeError("Bundled customs bank parts are missing.")
    encoded = b"".join(part.read_bytes().strip() for part in parts)
    packed = base64.b85decode(encoded)
    payload = bz2.decompress(packed)
    return payload.decode("utf-8"), hashlib.sha256(payload).hexdigest()


def _normalize_topic(value: str) -> str:
    return re.sub(r"\s*~?Рівень\s+[123]\s*$", "", str(value or "").strip(), flags=re.IGNORECASE).strip()


def _parse_rows() -> list[dict]:
    plaintext, source_hash = _bank_plaintext()
    bank = parse_testms_bank(
        plaintext,
        source="testmsmo.enc",
        source_hash=source_hash,
    )
    if bank.source_version != "13":
        raise RuntimeError(f"Unexpected customs bank version: {bank.source_version}")
    if len(bank.questions) != _EXPECTED_QUESTIONS:
        raise RuntimeError(f"Unexpected customs question count: {len(bank.questions)}")

    rows: list[dict] = []
    law_count = ok_count = 0
    for question_id, item in enumerate(bank.questions, start=1):
        section_code = item.source_key.rsplit(":", 2)[-2]
        section_number = _roman_to_int(section_code)

        if section_number <= 4:
            section = "Законодавство"
            topic = _normalize_topic(item.topic)
            ok = None
            level = None
            law_count += 1
        else:
            offset = section_number - 5
            module_number = offset // 3 + 1
            level = offset % 3 + 1
            ok = f"ОК-{module_number}"
            if ok not in OK_TITLES:
                raise RuntimeError(f"Unexpected operational competency: {ok}")
            section = "Операційні компетенції"
            topic = OK_TITLES[ok]
            ok_count += 1

        rows.append({
            "id": question_id,
            "section": section,
            "topic": topic,
            "ok": ok,
            "level": level,
            "qnum": int(item.qnum),
            "question": item.question,
            "choices": list(item.choices),
            "correct": [int(value) for value in item.correct],
            "correct_texts": list(item.correct_texts),
        })

    if law_count != _EXPECTED_LAW_QUESTIONS or ok_count != _EXPECTED_OK_QUESTIONS:
        raise RuntimeError(f"Unexpected customs split: law={law_count}, ok={ok_count}")
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
