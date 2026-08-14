"""Admin JSON import for the ``Тестові питання`` bank.

Adds a two-step import flow:
1. preview a JSON file and classify new questions, duplicates and answer conflicts;
2. apply the same file after the administrator explicitly resolves every conflict.

Duplicate detection is based on normalized question text. Very close wording is
shown as a manual conflict instead of being silently inserted or silently
skipped.
"""

from __future__ import annotations

import difflib
import functools
import inspect
import json
import re
import unicodedata
from typing import Any

from fastapi import Depends, FastAPI, File, Form, UploadFile


_ORIGINAL_FASTAPI_INIT = FastAPI.__init__
_PATCHED = False
_MAX_FILE_BYTES = 5 * 1024 * 1024
_SIMILARITY_THRESHOLD = 0.965
_DEFAULT_SOURCE = "Імпорт JSON"


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("’", "'").replace("`", "'").replace("ʼ", "'")
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _normalize_answer(value: Any) -> str:
    return _normalize_text(value)


def _clean_num(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return f"№ {value}"
    text = str(value).strip()
    if not text:
        return None
    digits = re.fullmatch(r"(?:№\s*)?(\d+)", text)
    if digits:
        return f"№ {int(digits.group(1))}"
    return text


def _extract_questions(payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(payload, dict):
        raw_items = payload.get("questions")
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = None

    if not isinstance(raw_items, list):
        raise ValueError("JSON повинен містити масив questions або бути масивом питань.")

    items: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            invalid.append({"index": index, "reason": "Елемент не є об'єктом."})
            continue

        question = str(raw.get("question") or "").strip()
        correct_answer = str(raw.get("correct_answer") or raw.get("answer") or "").strip()
        if not question or not correct_answer:
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
                "module": (str(raw.get("module") or "").strip() or None),
                "question": question,
                "correct_answer": correct_answer,
                "justification": str(raw.get("justification") or "").strip(),
                "source": str(raw.get("source") or _DEFAULT_SOURCE).strip() or _DEFAULT_SOURCE,
            }
        )

    return items, invalid


async def _read_payload(file: UploadFile) -> tuple[Any, int]:
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


def _best_similar_match(
    normalized_question: str,
    existing: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float]:
    if len(normalized_question) < 24:
        return None, 0.0

    best: dict[str, Any] | None = None
    best_ratio = 0.0
    q_len = len(normalized_question)
    for row in existing:
        candidate = row["_normalized_question"]
        if not candidate:
            continue
        c_len = len(candidate)
        if not c_len or min(q_len, c_len) / max(q_len, c_len) < 0.82:
            continue
        ratio = difflib.SequenceMatcher(None, normalized_question, candidate, autojunk=False).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = row
    if best_ratio >= _SIMILARITY_THRESHOLD:
        return best, best_ratio
    return None, best_ratio


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


def _public_imported(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if not key.startswith("_")}


async def _classify(runtime: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
    assert runtime.store.pool
    async with runtime.store.pool.acquire() as con:
        rows = await con.fetch(
            """
            SELECT id, num, module, question, correct_answer, justification, source
            FROM test_exam_questions
            ORDER BY id
            """
        )

    existing: list[dict[str, Any]] = []
    exact_map: dict[str, list[dict[str, Any]]] = {}
    for record in rows:
        row = dict(record)
        normalized = _normalize_text(row.get("question"))
        row["_normalized_question"] = normalized
        existing.append(row)
        if normalized:
            exact_map.setdefault(normalized, []).append(row)

    new_items: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    # Also deduplicate repeated questions inside the uploaded file itself.
    seen_import: dict[str, dict[str, Any]] = {}

    for item in items:
        normalized_question = _normalize_text(item["question"])
        normalized_answer = _normalize_answer(item["correct_answer"])
        item["_normalized_question"] = normalized_question

        prior_import = seen_import.get(normalized_question)
        if prior_import:
            if _normalize_answer(prior_import["correct_answer"]) == normalized_answer:
                duplicates.append(
                    {
                        "kind": "file_duplicate",
                        "imported": _public_imported(item),
                        "existing": _public_imported(prior_import),
                        "similarity": 1.0,
                    }
                )
            else:
                conflicts.append(
                    {
                        "kind": "file_answer_conflict",
                        "match_type": "exact",
                        "imported": _public_imported(item),
                        "existing": _public_imported(prior_import),
                        "similarity": 1.0,
                    }
                )
            continue
        seen_import[normalized_question] = item

        exact_candidates = exact_map.get(normalized_question) or []
        exact = exact_candidates[0] if exact_candidates else None
        if exact:
            if _normalize_answer(exact.get("correct_answer")) == normalized_answer:
                duplicates.append(
                    {
                        "kind": "existing_duplicate",
                        "imported": _public_imported(item),
                        "existing": _public_existing(exact),
                        "similarity": 1.0,
                    }
                )
            else:
                conflicts.append(
                    {
                        "kind": "answer_conflict",
                        "match_type": "exact",
                        "imported": _public_imported(item),
                        "existing": _public_existing(exact),
                        "similarity": 1.0,
                    }
                )
            continue

        similar, ratio = _best_similar_match(normalized_question, existing)
        if similar:
            conflicts.append(
                {
                    "kind": "similar_question",
                    "match_type": "similar",
                    "imported": _public_imported(item),
                    "existing": _public_existing(similar),
                    "answers_match": _normalize_answer(similar.get("correct_answer")) == normalized_answer,
                    "similarity": round(ratio, 4),
                }
            )
            continue

        new_items.append(_public_imported(item))

    return {
        "new": new_items,
        "duplicates": duplicates,
        "conflicts": conflicts,
        "existing_total": len(existing),
    }


def _parse_resolutions(raw: str) -> dict[int, str]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("Некоректний список рішень щодо конфліктів.") from exc
    if not isinstance(value, dict):
        raise ValueError("Рішення щодо конфліктів повинні бути об'єктом.")
    result: dict[int, str] = {}
    for key, decision in value.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError("Некоректний ідентифікатор конфлікту.") from exc
        decision = str(decision)
        if decision not in {"keep_existing", "use_imported", "add_new"}:
            raise ValueError("Невідоме рішення щодо конфлікту.")
        result[index] = decision
    return result


async def _apply_import(
    runtime: Any,
    items: list[dict[str, Any]],
    resolutions: dict[int, str],
) -> dict[str, int]:
    preview = await _classify(runtime, items)
    conflicts = preview["conflicts"]

    unresolved = [
        conflict
        for conflict in conflicts
        if int(conflict["imported"]["import_index"]) not in resolutions
    ]
    if unresolved:
        raise ValueError(f"Потрібно вирішити всі конфлікти: {len(unresolved)}.")

    item_by_index = {int(item["import_index"]): item for item in items}
    conflict_by_index = {
        int(conflict["imported"]["import_index"]): conflict for conflict in conflicts
    }

    inserted = 0
    updated = 0
    kept = len(preview["duplicates"])

    assert runtime.store.pool
    async with runtime.store.pool.acquire() as con:
        async with con.transaction():
            for public_item in preview["new"]:
                item = item_by_index[int(public_item["import_index"])]
                await con.execute(
                    """
                    INSERT INTO test_exam_questions
                        (num, module, question, correct_answer, justification, source)
                    VALUES ($1,$2,$3,$4,$5,$6)
                    """,
                    item.get("num"),
                    item.get("module"),
                    item["question"],
                    item["correct_answer"],
                    item.get("justification") or "",
                    item.get("source") or _DEFAULT_SOURCE,
                )
                inserted += 1

            for index, conflict in conflict_by_index.items():
                decision = resolutions[index]
                imported = item_by_index[index]
                match_type = conflict.get("match_type")
                existing_row = conflict.get("existing") or {}
                existing_id = existing_row.get("id")

                if decision == "keep_existing":
                    kept += 1
                    continue

                if decision == "add_new":
                    if match_type != "similar":
                        raise ValueError("Точний дублікат не можна додати як нове питання.")
                    await con.execute(
                        """
                        INSERT INTO test_exam_questions
                            (num, module, question, correct_answer, justification, source)
                        VALUES ($1,$2,$3,$4,$5,$6)
                        """,
                        imported.get("num"),
                        imported.get("module"),
                        imported["question"],
                        imported["correct_answer"],
                        imported.get("justification") or "",
                        imported.get("source") or _DEFAULT_SOURCE,
                    )
                    inserted += 1
                    continue

                if decision == "use_imported":
                    if not existing_id:
                        # Conflict only inside the uploaded file: keep the first item and
                        # use the selected answer on the row that will be inserted for it.
                        # Such duplicate-in-file conflicts are intentionally not auto-applied.
                        raise ValueError(
                            "Конфлікт усередині файлу потрібно виправити у JSON перед імпортом."
                        )
                    await con.execute(
                        """
                        UPDATE test_exam_questions
                        SET correct_answer=$2,
                            justification=CASE WHEN $3 <> '' THEN $3 ELSE justification END,
                            source=CASE WHEN $4 <> '' THEN $4 ELSE source END
                        WHERE id=$1
                        """,
                        int(existing_id),
                        imported["correct_answer"],
                        imported.get("justification") or "",
                        imported.get("source") or _DEFAULT_SOURCE,
                    )
                    updated += 1

    return {"inserted": inserted, "updated": updated, "kept": kept}


def _register_routes(app: FastAPI, module_globals: dict[str, Any]) -> None:
    if getattr(app.state, "_admin_test_exam_import_routes_installed", False):
        return

    get_auth_context = module_globals.get("get_auth_context")
    get_runtime = module_globals.get("get_runtime")
    require_http = module_globals.get("require_http")
    if not callable(get_auth_context) or not callable(get_runtime) or not callable(require_http):
        return

    @app.post("/api/admin/test-exam-questions/import/preview")
    async def api_admin_test_exam_import_preview(
        file: UploadFile = File(...),
        auth=Depends(get_auth_context),
        runtime=Depends(get_runtime),
    ):
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")
        try:
            payload, file_size = await _read_payload(file)
            items, invalid = _extract_questions(payload)
        except ValueError as exc:
            require_http(400, "invalid_test_import", str(exc))

        if not items:
            require_http(400, "empty_test_import", "У файлі немає придатних питань для імпорту.")

        classified = await _classify(runtime, items)
        return {
            "file_name": file.filename or "questions.json",
            "file_size": file_size,
            "valid_count": len(items),
            "invalid": invalid,
            "invalid_count": len(invalid),
            **classified,
            "new_count": len(classified["new"]),
            "duplicate_count": len(classified["duplicates"]),
            "conflict_count": len(classified["conflicts"]),
        }

    @app.post("/api/admin/test-exam-questions/import/apply")
    async def api_admin_test_exam_import_apply(
        file: UploadFile = File(...),
        resolutions: str = Form("{}"),
        auth=Depends(get_auth_context),
        runtime=Depends(get_runtime),
    ):
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")
        try:
            payload, _ = await _read_payload(file)
            items, invalid = _extract_questions(payload)
            decisions = _parse_resolutions(resolutions)
            result = await _apply_import(runtime, items, decisions)
        except ValueError as exc:
            require_http(400, "test_import_conflict", str(exc))

        return {
            "ok": True,
            **result,
            "invalid": len(invalid),
        }

    app.state._admin_test_exam_import_routes_installed = True


@functools.wraps(_ORIGINAL_FASTAPI_INIT)
def _fastapi_init_with_test_exam_import(self: FastAPI, *args, **kwargs) -> None:
    _ORIGINAL_FASTAPI_INIT(self, *args, **kwargs)
    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    module_globals = caller.f_globals if caller else {}
    if module_globals.get("__name__") == "app":
        _register_routes(self, module_globals)


def install() -> None:
    global _PATCHED
    if _PATCHED:
        return
    FastAPI.__init__ = _fastapi_init_with_test_exam_import
    _PATCHED = True


install()
