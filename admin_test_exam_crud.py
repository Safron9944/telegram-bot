"""Simple CRUD API for admin test-exam questions."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from fastapi import Body, Depends, FastAPI, Query


_DEFAULT_SOURCE = "Вручну"


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


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Некоректні дані питання.")
    question = str(payload.get("question") or "").strip()
    answer = str(payload.get("correct_answer") or "").strip()
    if not question:
        raise ValueError("Введіть текст питання.")
    if not answer:
        raise ValueError("Введіть правильну відповідь.")
    if len(question) > 10000:
        raise ValueError("Текст питання занадто довгий.")
    if len(answer) > 10000:
        raise ValueError("Правильна відповідь занадто довга.")

    module = str(payload.get("module") or "").strip() or None
    justification = str(payload.get("justification") or "").strip()
    source = str(payload.get("source") or "").strip() or _DEFAULT_SOURCE

    return {
        "num": _clean_num(payload.get("num")),
        "module": module,
        "question": question,
        "correct_answer": answer,
        "justification": justification,
        "source": source,
    }


def _public(row: Any) -> dict[str, Any]:
    data = dict(row)
    return {
        "id": int(data["id"]),
        "num": data.get("num"),
        "module": data.get("module"),
        "question": data.get("question") or "",
        "correct_answer": data.get("correct_answer") or "",
        "justification": data.get("justification") or "",
        "source": data.get("source") or "",
        "created_at": data.get("created_at").isoformat() if data.get("created_at") else None,
    }


async def _find_duplicate(con: Any, question: str, *, exclude_id: int | None = None) -> dict[str, Any] | None:
    rows = await con.fetch(
        """
        SELECT id, num, module, question, correct_answer, justification, source, created_at
        FROM test_exam_questions
        ORDER BY id
        """
    )
    target = _normalize(question)
    for row in rows:
        row_id = int(row["id"])
        if exclude_id is not None and row_id == exclude_id:
            continue
        if _normalize(row.get("question")) == target:
            return _public(row)
    return None


async def _fetch_one(con: Any, question_id: int) -> Any:
    return await con.fetchrow(
        """
        SELECT id, num, module, question, correct_answer, justification, source, created_at
        FROM test_exam_questions
        WHERE id=$1
        """,
        int(question_id),
    )


def register_routes(
    app: FastAPI,
    *,
    get_auth_context: Any,
    get_runtime: Any,
    require_http: Any,
) -> None:
    if getattr(app.state, "_admin_test_exam_crud_routes_installed", False):
        return

    def require_admin(auth: Any) -> None:
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")

    @app.get("/api/admin/test-exam-questions/crud/lookup")
    async def lookup_question(
        question: str = Query(min_length=1),
        num: str = "",
        answer: str = "",
        module: str = "",
        auth=Depends(get_auth_context),
        runtime=Depends(get_runtime),
    ):
        require_admin(auth)
        assert runtime.store.pool
        async with runtime.store.pool.acquire() as con:
            rows = await con.fetch(
                """
                SELECT id, num, module, question, correct_answer, justification, source, created_at
                FROM test_exam_questions
                ORDER BY id
                """
            )

        target = _normalize(question)
        matches = [_public(row) for row in rows if _normalize(row.get("question")) == target]
        if not matches:
            require_http(404, "test_question_not_found", "Питання більше не знайдено в базі.")

        num_norm = _normalize(num)
        answer_norm = _normalize(answer)
        module_norm = _normalize(module)

        def score(item: dict[str, Any]) -> int:
            value = 0
            if num_norm and _normalize(item.get("num")) == num_norm:
                value += 3
            if answer_norm and _normalize(item.get("correct_answer")) == answer_norm:
                value += 2
            if module_norm and _normalize(item.get("module")) == module_norm:
                value += 1
            return value

        matches.sort(key=lambda item: (-score(item), item["id"]))
        return {"item": matches[0], "matches": len(matches)}

    @app.get("/api/admin/test-exam-questions/crud/{question_id}")
    async def get_question(
        question_id: int,
        auth=Depends(get_auth_context),
        runtime=Depends(get_runtime),
    ):
        require_admin(auth)
        assert runtime.store.pool
        async with runtime.store.pool.acquire() as con:
            row = await _fetch_one(con, question_id)
        if not row:
            require_http(404, "test_question_not_found", "Питання не знайдено.")
        return {"item": _public(row)}

    @app.post("/api/admin/test-exam-questions/crud")
    async def create_question(
        payload: dict[str, Any] = Body(...),
        auth=Depends(get_auth_context),
        runtime=Depends(get_runtime),
    ):
        require_admin(auth)
        try:
            clean = _clean_payload(payload)
        except ValueError as exc:
            require_http(400, "invalid_test_question", str(exc))

        assert runtime.store.pool
        async with runtime.store.pool.acquire() as con:
            async with con.transaction():
                duplicate = await _find_duplicate(con, clean["question"])
                if duplicate:
                    require_http(
                        409,
                        "duplicate_test_question",
                        f"Таке питання вже є в базі (ID {duplicate['id']}). Відкрийте його та відредагуйте.",
                    )
                row = await con.fetchrow(
                    """
                    INSERT INTO test_exam_questions
                        (num, module, question, correct_answer, justification, source)
                    VALUES ($1,$2,$3,$4,$5,$6)
                    RETURNING id, num, module, question, correct_answer, justification, source, created_at
                    """,
                    clean["num"],
                    clean["module"],
                    clean["question"],
                    clean["correct_answer"],
                    clean["justification"],
                    clean["source"],
                )
        return {"ok": True, "item": _public(row)}

    @app.patch("/api/admin/test-exam-questions/crud/{question_id}")
    async def update_question(
        question_id: int,
        payload: dict[str, Any] = Body(...),
        auth=Depends(get_auth_context),
        runtime=Depends(get_runtime),
    ):
        require_admin(auth)
        try:
            clean = _clean_payload(payload)
        except ValueError as exc:
            require_http(400, "invalid_test_question", str(exc))

        assert runtime.store.pool
        async with runtime.store.pool.acquire() as con:
            async with con.transaction():
                current = await _fetch_one(con, question_id)
                if not current:
                    require_http(404, "test_question_not_found", "Питання не знайдено.")

                duplicate = await _find_duplicate(con, clean["question"], exclude_id=question_id)
                if duplicate:
                    require_http(
                        409,
                        "duplicate_test_question",
                        f"Після зміни це питання дублює інше питання (ID {duplicate['id']}).",
                    )

                row = await con.fetchrow(
                    """
                    UPDATE test_exam_questions
                    SET num=$2,
                        module=$3,
                        question=$4,
                        correct_answer=$5,
                        justification=$6,
                        source=$7
                    WHERE id=$1
                    RETURNING id, num, module, question, correct_answer, justification, source, created_at
                    """,
                    int(question_id),
                    clean["num"],
                    clean["module"],
                    clean["question"],
                    clean["correct_answer"],
                    clean["justification"],
                    clean["source"],
                )
        return {"ok": True, "item": _public(row)}

    @app.delete("/api/admin/test-exam-questions/crud/{question_id}")
    async def delete_question(
        question_id: int,
        auth=Depends(get_auth_context),
        runtime=Depends(get_runtime),
    ):
        require_admin(auth)
        assert runtime.store.pool
        async with runtime.store.pool.acquire() as con:
            result = await con.execute("DELETE FROM test_exam_questions WHERE id=$1", int(question_id))
        if result == "DELETE 0":
            require_http(404, "test_question_not_found", "Питання не знайдено.")
        return {"ok": True, "id": int(question_id)}

    app.state._admin_test_exam_crud_routes_installed = True
