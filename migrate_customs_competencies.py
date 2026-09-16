from __future__ import annotations

import asyncio
import hashlib
import json
import lzma
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from questions import LEGACY_ATTESTATION_STAGE_1_SECTION
from storage import Storage
from utils import clean_law_title, ok_extract_code

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "data" / "customs_competencies_1_9_1.json.xz"
SOURCE_SETTING_KEY = "customs_competencies_source_sha256"
SOURCE_LABEL = "apk-1.9.1:testmsmo-13"
EXPECTED_TOTAL = 3410
EXPECTED_LAW = 800
EXPECTED_OK = 2610
EXPECTED_OK_MODULES = 17


def _database_url() -> str:
    return (
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("POSTGRESQL_URL")
        or os.getenv("PGDATABASE_URL")
        or ""
    ).strip()


def _bundle_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalize_law_topic(value: Any) -> str:
    topic = clean_law_title(str(value or "").strip())
    topic = re.sub(r"^\s*(?:[IVXLCDM]+|\d+)\s*[.)]\s*", "", topic, flags=re.IGNORECASE)
    return " ".join(topic.split()).casefold()


def _question_key(item: dict[str, Any]) -> tuple[Any, ...] | None:
    ok = ok_extract_code(str(item.get("ok") or ""))
    try:
        qnum = int(item.get("qnum")) if item.get("qnum") is not None else None
    except (TypeError, ValueError):
        qnum = None
    if qnum is None:
        return None

    if ok:
        try:
            level = int(item.get("level"))
        except (TypeError, ValueError):
            return None
        return ("ok", ok, level, qnum)

    topic = _normalize_law_topic(item.get("topic") or item.get("section"))
    return ("law", topic, qnum) if topic else None


def _load_bundle(path: Path) -> list[dict[str, Any]]:
    compressed = path.read_bytes()
    raw = lzma.decompress(compressed)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("Customs competencies bundle must contain a JSON list.")
    items = [dict(item) for item in data if isinstance(item, dict)]
    if len(items) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Unexpected customs competencies question count: {len(items)} != {EXPECTED_TOTAL}"
        )

    law_count = sum(1 for item in items if not ok_extract_code(str(item.get("ok") or "")))
    ok_count = len(items) - law_count
    modules = {
        ok_extract_code(str(item.get("ok") or ""))
        for item in items
        if ok_extract_code(str(item.get("ok") or ""))
    }
    if (law_count, ok_count, len(modules)) != (EXPECTED_LAW, EXPECTED_OK, EXPECTED_OK_MODULES):
        raise RuntimeError(
            "Unexpected customs competencies structure: "
            f"law={law_count}, ok={ok_count}, modules={len(modules)}"
        )
    return items


def _assign_stable_ids(
    source_items: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    legacy_ids = {
        int(row["id"])
        for row in existing_rows
        if str(row.get("section") or "").strip() == LEGACY_ATTESTATION_STAGE_1_SECTION
    }
    main_rows = [
        row
        for row in existing_rows
        if str(row.get("section") or "").strip() != LEGACY_ATTESTATION_STAGE_1_SECTION
    ]
    existing_by_key: dict[tuple[Any, ...], list[int]] = {}
    for row in main_rows:
        key = _question_key(row)
        if key is not None:
            existing_by_key.setdefault(key, []).append(int(row["id"]))

    used_ids: set[int] = set()
    all_existing_ids = {int(row["id"]) for row in existing_rows}
    next_id = max(all_existing_ids | {EXPECTED_TOTAL}) + 1
    mapped: list[dict[str, Any]] = []

    for source in source_items:
        item = dict(source)
        key = _question_key(item)
        candidates = existing_by_key.get(key, []) if key is not None else []
        qid = next((value for value in candidates if value not in used_ids), None)

        preferred = int(item.get("id") or 0)
        if qid is None and preferred > 0 and preferred not in legacy_ids and preferred not in used_ids:
            qid = preferred

        if qid is None:
            while next_id in all_existing_ids or next_id in used_ids:
                next_id += 1
            qid = next_id
            next_id += 1

        item["id"] = int(qid)
        used_ids.add(int(qid))
        mapped.append(item)

    stale_main_ids = {int(row["id"]) for row in main_rows} - used_ids
    return mapped, used_ids, stale_main_ids


async def _delete_stale_main_questions(store: Storage, stale_ids: set[int]) -> int:
    if not stale_ids:
        return 0
    assert store.pool
    async with store.pool.acquire() as con:
        result = await con.execute(
            "DELETE FROM questions WHERE id = ANY($1::int[])",
            sorted(stale_ids),
        )
    try:
        return int(result.rsplit(" ", 1)[-1])
    except (TypeError, ValueError):
        return len(stale_ids)


async def _verify_database(store: Storage) -> None:
    rows = await store.fetch_questions()
    main_rows = [
        row
        for row in rows
        if str(row.get("section") or "").strip() != LEGACY_ATTESTATION_STAGE_1_SECTION
    ]
    law_count = sum(1 for row in main_rows if not ok_extract_code(str(row.get("ok") or "")))
    ok_count = len(main_rows) - law_count
    modules = {
        ok_extract_code(str(row.get("ok") or ""))
        for row in main_rows
        if ok_extract_code(str(row.get("ok") or ""))
    }
    if (len(main_rows), law_count, ok_count, len(modules)) != (
        EXPECTED_TOTAL,
        EXPECTED_LAW,
        EXPECTED_OK,
        EXPECTED_OK_MODULES,
    ):
        raise RuntimeError(
            "Customs competencies database verification failed: "
            f"total={len(main_rows)}, law={law_count}, ok={ok_count}, modules={len(modules)}"
        )


async def migrate() -> None:
    dsn = _database_url()
    if not dsn:
        raise RuntimeError("Set DATABASE_URL env var.")
    if not BUNDLE_PATH.exists():
        raise RuntimeError(f"Customs competencies bundle not found: {BUNDLE_PATH}")

    bundle_bytes = BUNDLE_PATH.read_bytes()
    fingerprint = _bundle_hash(bundle_bytes)
    source_items = _load_bundle(BUNDLE_PATH)

    store = Storage(dsn)
    await store.init()
    try:
        applied = await store.get_setting(SOURCE_SETTING_KEY, "")
        if applied == fingerprint:
            print(f"Customs competencies are current ({SOURCE_LABEL}, {EXPECTED_TOTAL} questions).")
            return

        existing_rows = await store.fetch_questions()
        mapped_items, _, stale_ids = _assign_stable_ids(source_items, existing_rows)

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".json",
                delete=False,
            ) as handle:
                json.dump(mapped_items, handle, ensure_ascii=False, separators=(",", ":"))
                temp_path = handle.name

            changed = await store.import_questions_from_json(
                temp_path,
                changed_by=SOURCE_LABEL,
                force=False,
            )
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

        removed = await _delete_stale_main_questions(store, stale_ids)
        await _verify_database(store)
        await store.set_setting(SOURCE_SETTING_KEY, fingerprint)
        await store.set_setting("customs_competencies_source_version", SOURCE_LABEL)

        print(
            "Customs competencies updated: "
            f"{EXPECTED_TOTAL} questions, {changed} inserted/changed, {removed} obsolete removed."
        )
    finally:
        if store.pool:
            await store.pool.close()


if __name__ == "__main__":
    asyncio.run(migrate())
