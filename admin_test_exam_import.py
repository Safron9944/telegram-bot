"""Manual JSON import for the admin ``Тестові питання`` screen.

Routes are registered explicitly from ``launcher.py`` after the main FastAPI
application is created, so this module does not patch FastAPI startup hooks.
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from typing import Any

from fastapi import Depends, FastAPI, File, Form, UploadFile

_MAX_FILE_BYTES = 5 * 1024 * 1024
_SIMILARITY_THRESHOLD = 0.965
_DEFAULT_SOURCE = "Імпорт JSON"


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("’", "'").replace("`", "'").replace("ʼ", "'")
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _clean_num(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return f"№ {value}"
    text = str(value).strip()
    if not text:
        return None
    match = re.fullmatch(r"(?:№\s*)?(\d+)", text)
    return f"№ {int(match.group(1))}" if match else text


async def _read_json(file: UploadFile) -> tuple[Any, int]:
    content = await file.read(_MAX_FILE_BYTES + 1)
    if len(content) > _MAX_FILE_BYTES:
        raise ValueError("Файл завеликий. Максимальний розмір — 5 МБ.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("JSON-файл повинен бути у кодуванні UTF-8.") from exc
    try:
        return json.loads(text), len(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некоректний JSON: рядок {exc.lineno}, колонка {exc.colno}.") from exc


def _extract_items(payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_items = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raise ValueError("JSON повинен містити масив questions або бути масивом питань.")

    items: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            invalid.append({"index": index, "reason": "Елемент не є об'єктом."})
            continue
        question = str(raw.get("question") or "").strip()
        answer = str(raw.get("correct_answer") or raw.get("answer") or "").strip()
        if not question or not answer:
            invalid.append(
                {
                    "index": index,
                    "number": raw.get("number") or raw.get("num"),
                    "question": question,
                    "reason": "Потрібні поля question і correct_answer.",
                }
            )
            continue
        items.append(
            {
                "import_index": index,
                "num": _clean_num(raw.get("number") if raw.get("number") is not None else raw.get("num")),
                "module": str(raw.get("module") or "").strip() or None,
                "question": question,
                "correct_answer": answer,
                "justification": str(raw.get("justification") or "").strip(),
                "source": str(raw.get("source") or _DEFAULT_SOURCE).strip() or _DEFAULT_SOURCE,
            }
        )
    return items, invalid


def _public_item(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if not k.startswith("_")}


def _public_existing(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "num": row.get("num"),
        "module": row.get("module"),
        "question": row.get("question") or "",
        "correct_answer": row.get("correct_answer") or "",
        "justification": row.get("justification") or "",
        "source": row.get("source") or "",
    }


def _best_similar(normalized: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    if len(normalized) < 24:
        return None, 0.0
    best = None
    best_ratio = 0.0
    for row in rows:
        candidate = row["_normalized_question"]
        if not candidate:
            continue
        if min(len(normalized), len(candidate)) / max(len(normalized), len(candidate)) < 0.82:
            continue
        ratio = difflib.SequenceMatcher(None, normalized, candidate, autojunk=False).ratio()
        if ratio > best_ratio:
            best = row
            best_ratio = ratio
    return (best, best_ratio) if best_ratio >= _SIMILARITY_THRESHOLD else (None, best_ratio)


async def _classify(runtime: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
    assert runtime.store.pool
    async with runtime.store.pool.acquire() as con:
        records = await con.fetch(
            """
            SELECT id, num, module, question, correct_answer, justification, source
            FROM test_exam_questions
            ORDER BY id
            """
        )

    existing: list[dict[str, Any]] = []
    exact: dict[str, dict[str, Any]] = {}
    for record in records:
        row = dict(record)
        normalized = _normalize(row.get("question"))
        row["_normalized_question"] = normalized
        existing.append(row)
        exact.setdefault(normalized, row)

    new_items: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    seen_file: dict[str, dict[str, Any]] = {}

    for item in items:
        q_norm = _normalize(item["question"])
        a_norm = _normalize(item["correct_answer"])
        item["_normalized_question"] = q_norm

        earlier = seen_file.get(q_norm)
        if earlier:
            if _normalize(earlier["correct_answer"]) == a_norm:
                duplicates.append(
                    {
                        "kind": "file_duplicate",
                        "match_type": "exact",
                        "imported": _public_item(item),
                        "existing": _public_item(earlier),
                        "similarity": 1.0,
                    }
                )
            else:
                conflicts.append(
                    {
                        "kind": "file_answer_conflict",
                        "match_type": "exact",
                        "imported": _public_item(item),
                        "existing": _public_item(earlier),
                        "similarity": 1.0,
                    }
                )
            continue
        seen_file[q_norm] = item

        matched = exact.get(q_norm)
        if matched:
            payload = {
                "imported": _public_item(item),
                "existing": _public_existing(matched),
                "similarity": 1.0,
                "match_type": "exact",
            }
            if _normalize(matched.get("correct_answer")) == a_norm:
                duplicates.append({"kind": "existing_duplicate", **payload})
            else:
                conflicts.append({"kind": "answer_conflict", **payload})
            continue

        similar, ratio = _best_similar(q_norm, existing)
        if similar:
            conflicts.append(
                {
                    "kind": "similar_question",
                    "match_type": "similar",
                    "imported": _public_item(item),
                    "existing": _public_existing(similar),
                    "answers_match": _normalize(similar.get("correct_answer")) == a_norm,
                    "similarity": round(ratio, 4),
                }
            )
            continue

        new_items.append(_public_item(item))

    return {
        "new": new_items,
        "duplicates": duplicates,
        "conflicts": conflicts,
        "existing_total": len(existing),
    }


def _parse_resolutions(raw: str) -> dict[int, str]:
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("Некоректні рішення щодо конфліктів.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Рішення щодо конфліктів повинні бути об'єктом.")
    result: dict[int, str] = {}
    for key, value in payload.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError("Некоректний номер конфлікту.") from exc
        decision = str(value)
        if decision not in {"keep_existing", "use_imported", "add_new"}:
            raise ValueError("Невідоме рішення щодо конфлікту.")
        result[index] = decision
    return result


async def _apply(runtime: Any, items: list[dict[str, Any]], decisions: dict[int, str]) -> dict[str, int]:
    preview = await _classify(runtime, items)
    file_conflicts = [c for c in preview["conflicts"] if c["kind"] == "file_answer_conflict"]
    if file_conflicts:
        raise ValueError("У самому JSON є однакові питання з різними відповідями. Виправ файл перед імпортом.")

    resolvable = [c for c in preview["conflicts"] if c["kind"] != "file_answer_conflict"]
    missing = [c for c in resolvable if int(c["imported"]["import_index"]) not in decisions]
    if missing:
        raise ValueError(f"Потрібно вирішити всі конфлікти: {len(missing)}.")

    item_by_index = {int(item["import_index"]): item for item in items}
    inserted = 0
    updated = 0
    kept = len(preview["duplicates"])

    assert runtime.store.pool
    async with runtime.store.pool.acquire() as con:
        async with con.transaction():
            for public in preview["new"]:
                item = item_by_index[int(public["import_index"])]
                await con.execute(
                    """
                    INSERT INTO test_exam_questions
                        (num, module, question, correct_answer, justification, source)
                    VALUES ($1,$2,$3,$4,$5,$6)
                    """,
                    item.get("num"), item.get("module"), item["question"], item["correct_answer"],
                    item.get("justification") or "", item.get("source") or _DEFAULT_SOURCE,
                )
                inserted += 1

            for conflict in resolvable:
                imported = item_by_index[int(conflict["imported"]["import_index"])]
                decision = decisions[int(imported["import_index"])]
                existing = conflict.get("existing") or {}

                if decision == "keep_existing":
                    kept += 1
                    continue
                if decision == "add_new":
                    if conflict.get("match_type") != "similar":
                        raise ValueError("Точний дублікат не можна додати як нове питання.")
                    await con.execute(
                        """
                        INSERT INTO test_exam_questions
                            (num, module, question, correct_answer, justification, source)
                        VALUES ($1,$2,$3,$4,$5,$6)
                        """,
                        imported.get("num"), imported.get("module"), imported["question"], imported["correct_answer"],
                        imported.get("justification") or "", imported.get("source") or _DEFAULT_SOURCE,
                    )
                    inserted += 1
                    continue

                existing_id = existing.get("id")
                if not existing_id:
                    raise ValueError("Не вдалося визначити наявне питання для оновлення.")
                await con.execute(
                    """
                    UPDATE test_exam_questions
                    SET correct_answer=$2,
                        justification=CASE WHEN $3 <> '' THEN $3 ELSE justification END,
                        source=CASE WHEN $4 <> '' THEN $4 ELSE source END
                    WHERE id=$1
                    """,
                    int(existing_id), imported["correct_answer"],
                    imported.get("justification") or "", imported.get("source") or _DEFAULT_SOURCE,
                )
                updated += 1

    return {"inserted": inserted, "updated": updated, "kept": kept}


def register_routes(
    app: FastAPI,
    *,
    get_auth_context: Any,
    get_runtime: Any,
    require_http: Any,
) -> None:
    if getattr(app.state, "_admin_test_exam_import_routes_installed", False):
        return

    @app.post("/api/admin/test-exam-questions/import/preview")
    async def preview_import(
        file: UploadFile = File(...),
        auth=Depends(get_auth_context),
        runtime=Depends(get_runtime),
    ):
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")
        try:
            payload, size = await _read_json(file)
            items, invalid = _extract_items(payload)
        except ValueError as exc:
            require_http(400, "invalid_test_import", str(exc))
        if not items:
            require_http(400, "empty_test_import", "У файлі немає придатних питань для імпорту.")
        classified = await _classify(runtime, items)
        return {
            "file_name": file.filename or "questions.json",
            "file_size": size,
            "valid_count": len(items),
            "invalid": invalid,
            "invalid_count": len(invalid),
            **classified,
            "new_count": len(classified["new"]),
            "duplicate_count": len(classified["duplicates"]),
            "conflict_count": len(classified["conflicts"]),
        }

    @app.post("/api/admin/test-exam-questions/import/apply")
    async def apply_import(
        file: UploadFile = File(...),
        resolutions: str = Form("{}"),
        auth=Depends(get_auth_context),
        runtime=Depends(get_runtime),
    ):
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")
        try:
            payload, _ = await _read_json(file)
            items, invalid = _extract_items(payload)
            result = await _apply(runtime, items, _parse_resolutions(resolutions))
        except ValueError as exc:
            require_http(400, "test_import_conflict", str(exc))
        return {"ok": True, **result, "invalid": len(invalid)}

    app.state._admin_test_exam_import_routes_installed = True
