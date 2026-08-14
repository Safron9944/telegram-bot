"""Editable admin JSON import endpoints for the ``Тестові питання`` bank.

This extends the existing import flow with server-side application of edits made
in the Mini App before preview/apply. The original uploaded JSON stays unchanged;
edits are carried as a small JSON object keyed by the import item's index.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import Depends, FastAPI, File, Form, UploadFile

from admin_test_exam_import import (
    _apply_import,
    _classify,
    _clean_num,
    _extract_questions,
    _parse_resolutions,
    _read_payload,
)

_ALLOWED_EDIT_FIELDS = {
    "num",
    "module",
    "question",
    "correct_answer",
    "justification",
    "source",
}


def _parse_edits(raw: str) -> dict[int, dict[str, Any]]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("Некоректний список виправлень питань.") from exc
    if not isinstance(value, dict):
        raise ValueError("Виправлення питань повинні бути об'єктом.")

    result: dict[int, dict[str, Any]] = {}
    for raw_index, raw_edit in value.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("Некоректний номер редагованого питання.") from exc
        if not isinstance(raw_edit, dict):
            raise ValueError("Кожне виправлення питання повинно бути об'єктом.")

        unknown = set(raw_edit) - _ALLOWED_EDIT_FIELDS
        if unknown:
            raise ValueError("Передано невідоме поле редагування питання.")

        edit: dict[str, Any] = {}
        if "question" in raw_edit:
            question = str(raw_edit.get("question") or "").strip()
            if not question:
                raise ValueError("Текст питання не може бути порожнім.")
            edit["question"] = question
        if "correct_answer" in raw_edit:
            answer = str(raw_edit.get("correct_answer") or "").strip()
            if not answer:
                raise ValueError("Правильна відповідь не може бути порожньою.")
            edit["correct_answer"] = answer
        if "num" in raw_edit:
            edit["num"] = _clean_num(raw_edit.get("num"))
        for field in ("module", "justification", "source"):
            if field in raw_edit:
                text = str(raw_edit.get(field) or "").strip()
                edit[field] = (text or None) if field == "module" else text

        result[index] = edit
    return result


def _apply_edits_to_items(
    items: list[dict[str, Any]],
    edits: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_index = {int(item["import_index"]): item for item in items}
    missing = sorted(set(edits) - set(by_index))
    if missing:
        raise ValueError("Одне з редагованих питань більше не існує у файлі.")

    for index, edit in edits.items():
        by_index[index].update(edit)

    for item in items:
        if not str(item.get("question") or "").strip():
            raise ValueError("Текст питання не може бути порожнім.")
        if not str(item.get("correct_answer") or "").strip():
            raise ValueError("Правильна відповідь не може бути порожньою.")
    return items


def register_routes(
    app: FastAPI,
    *,
    get_auth_context,
    get_runtime,
    require_http,
) -> None:
    if getattr(app.state, "_admin_test_exam_editor_routes_installed", False):
        return

    @app.post("/api/admin/test-exam-questions/import/preview-edited")
    async def api_admin_test_exam_import_preview_edited(
        file: UploadFile = File(...),
        edits: str = Form("{}"),
        auth=Depends(get_auth_context),
        runtime=Depends(get_runtime),
    ):
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")
        try:
            payload, file_size = await _read_payload(file)
            items, invalid = _extract_questions(payload)
            parsed_edits = _parse_edits(edits)
            _apply_edits_to_items(items, parsed_edits)
        except ValueError as exc:
            require_http(400, "invalid_test_import_edit", str(exc))

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

    @app.post("/api/admin/test-exam-questions/import/apply-edited")
    async def api_admin_test_exam_import_apply_edited(
        file: UploadFile = File(...),
        edits: str = Form("{}"),
        resolutions: str = Form("{}"),
        auth=Depends(get_auth_context),
        runtime=Depends(get_runtime),
    ):
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")
        try:
            payload, _ = await _read_payload(file)
            items, invalid = _extract_questions(payload)
            if invalid:
                pass
            parsed_edits = _parse_edits(edits)
            _apply_edits_to_items(items, parsed_edits)
            parsed_resolutions = _parse_resolutions(resolutions)
            result = await _apply_import(runtime, items, parsed_resolutions)
        except ValueError as exc:
            require_http(400, "invalid_test_import_edit", str(exc))

        return {
            "ok": True,
            **result,
            "edited": len(parsed_edits),
        }

    app.state._admin_test_exam_editor_routes_installed = True
