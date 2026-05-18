from __future__ import annotations

import base64
import hashlib
import html
import os
import sqlite3
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

from rijndael.cipher.method import rijndael

KEY = "lkirwf897+22#bbtrm8814z5qq=498j5"
IV = "741952hheeyy66#cs!9hjv887mxx7@8y"
CIPHER = rijndael(KEY, block_size=32)


def _sxor(a: str, b: str) -> str:
    return "".join(chr(ord(x) ^ ord(y)) for x, y in zip(a, b))


def clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def decrypt_answer(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        data = base64.b64decode(text)
    except Exception:
        return clean_text(text)

    try:
        encrypted = data.decode("latin1")
        prev = IV
        out: list[str] = []
        for index in range(0, len(encrypted), 32):
            block = encrypted[index:index + 32]
            if len(block) != 32:
                break
            out.append(_sxor(CIPHER.decrypt(block), prev))
            prev = block
        raw = "".join(out).encode("latin1")
        return clean_text(raw.rstrip(b"\x00").decode("utf-8", "replace"))
    except Exception:
        return clean_text(text)


def _fetch_case_meta(con: sqlite3.Connection) -> dict[str, str]:
    case_number = ""
    case_title = ""

    try:
        row = con.execute(
            "SELECT namber_spisok, name_spisok FROM x_spisok ORDER BY id_spisok LIMIT 1"
        ).fetchone()
        if row:
            case_number = clean_text(row[0])
            case_title = clean_text(row[1])
    except Exception:
        pass

    try:
        row = con.execute(
            "SELECT name_posada, name_test_test FROM test_test ORDER BY id_test_test LIMIT 1"
        ).fetchone()
        if row:
            case_number = case_number or clean_text(row[0])
            case_title = case_title or clean_text(row[1])
    except Exception:
        pass

    return {
        "case_number": case_number or "Без номера",
        "case_title": case_title or "Кейс без назви",
    }


def extract_case_from_keys_db(path: str | Path) -> dict[str, Any]:
    db_path = Path(path)
    digest = hashlib.sha256(db_path.read_bytes()).hexdigest()

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        meta = _fetch_case_meta(con)
        rows = con.execute(
            """
            SELECT v.id_vopros, v.name_test_vopros, v.descr_test_vopros, v.vid_vopros,
                   a.id_answer, a.name_answer, a.var_x, a.bal
            FROM test_vopros v
            LEFT JOIN test_answer a ON a.id_answer_vopros = v.id_vopros
            ORDER BY v.id_vopros, a.id_answer
            """
        ).fetchall()
    finally:
        con.close()

    all_answers: list[dict[str, Any]] = []
    qmap: "OrderedDict[int, dict[str, Any]]" = OrderedDict()

    for row in rows:
        qid = int(row["id_vopros"])
        question = clean_text(row["name_test_vopros"])
        description = clean_text(row["descr_test_vopros"])
        answer_id = row["id_answer"]
        answer_text = decrypt_answer(row["name_answer"]) if answer_id is not None else ""
        is_correct = bool(row["bal"] == 100 or row["var_x"] == 1)

        if qid not in qmap:
            qmap[qid] = {
                "source_question_id": qid,
                "position": len(qmap) + 1,
                "question": question,
                "description": description,
                "question_type": row["vid_vopros"],
                "answers": [],
                "correct_answers": [],
            }

        answer = {
            "answer_id": int(answer_id) if answer_id is not None else None,
            "answer_text": answer_text,
            "var_x": int(row["var_x"] or 0),
            "bal": int(row["bal"] or 0),
            "is_correct": is_correct,
        }
        if answer_id is not None:
            qmap[qid]["answers"].append(answer)
            all_answers.append({"question_id": qid, **answer})
        if is_correct:
            qmap[qid]["correct_answers"].append(answer_text)

    questions = list(qmap.values())
    for item in questions:
        correct = item.get("correct_answers") or []
        item["correct_answer"] = "\n".join(
            f"{idx + 1}) {text}" if len(correct) > 1 else text
            for idx, text in enumerate(correct)
        )
        item["correct_count"] = len(correct)

    return {
        **meta,
        "source_hash": digest,
        "questions": questions,
        "questions_count": len(questions),
        "answers_count": len(all_answers),
        "correct_count": sum(1 for item in all_answers if item.get("is_correct")),
    }


def extract_case_from_upload_bytes(data: bytes) -> dict[str, Any]:
    fd, name = tempfile.mkstemp(suffix="_Keys.db")
    os.close(fd)
    path = Path(name)
    try:
        path.write_bytes(data)
        return extract_case_from_keys_db(path)
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
