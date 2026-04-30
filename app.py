from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import urllib.parse
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message, WebAppInfo
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bot import (
    GROUP_URL,
    QuestionBank,
    Storage,
    access_status,
    clean_law_title,
    dt_to_iso,
    find_question_ids_by_title,
    get_admin_contact_url,
    iso_to_dt,
    now,
    ok_extract_code,
    ok_full_label,
    ok_sort_key,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"


def env_flag(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name, "1" if default else "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def parse_admin_ids(raw: str) -> set[int]:
    admin_ids: set[int] = set()
    for item in (raw or "").split(","):
        item = item.strip()
        if item.isdigit():
            admin_ids.add(int(item))
    return admin_ids


def resolve_questions_path() -> Path:
    raw = (os.getenv("QUESTIONS_PATH") or "questions_flat.json").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def resolve_webapp_url() -> str:
    explicit = (
        os.getenv("WEBAPP_URL")
        or os.getenv("PUBLIC_BASE_URL")
        or os.getenv("APP_BASE_URL")
        or ""
    ).strip()
    if explicit:
        return explicit.rstrip("/")

    railway_domain = (os.getenv("RAILWAY_PUBLIC_DOMAIN") or "").strip()
    if railway_domain:
        return f"https://{railway_domain.rstrip('/')}"

    return "http://localhost:8000"


def format_dt(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, str):
        value = iso_to_dt(value)
    if not value:
        return None
    return value.strftime("%d.%m.%Y %H:%M")


def access_payload(user: dict[str, Any]) -> dict[str, Any]:
    has_access, state = access_status(user)
    trial_end = format_dt(user.get("trial_end"))
    sub_end = format_dt(user.get("sub_end"))

    if has_access and state == "trial":
        label = f"Тріал до {trial_end}" if trial_end else "Тріал активний"
    elif has_access and state == "sub_infinite":
        label = "Підписка активна безстроково"
    elif has_access and state == "sub_active":
        label = f"Підписка до {sub_end}" if sub_end else "Підписка активна"
    elif state == "not_registered":
        label = "Користувача ще не зареєстровано"
    else:
        label = "Доступ завершився"

    return {
        "has_access": has_access,
        "state": state,
        "label": label,
        "trial_end": dt_to_iso(user.get("trial_end")),
        "sub_end": dt_to_iso(user.get("sub_end")),
        "sub_infinite": bool(user.get("sub_infinite")),
    }


def require_http(status_code: int, code: str, message: str) -> None:
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message})


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def serialize_question(q: Any, *, reveal_answers: bool = False) -> dict[str, Any]:
    correct = [int(x) for x in (q.correct or [])]
    correct_set = set(correct)

    choices = []
    for index, text in enumerate(q.choices or [], start=1):
        item: dict[str, Any] = {
            "index": index,
            "text": text,
        }
        if reveal_answers:
            item["is_correct"] = index in correct_set
        choices.append(item)

    payload = {
        "id": int(q.id),
        "section": q.section,
        "topic": q.topic,
        "ok": q.ok,
        "ok_label": ok_full_label(q.ok) if q.ok else None,
        "level": q.level,
        "qnum": q.qnum,
        "question": q.question,
        "choices": choices,
    }
    if reveal_answers:
        payload["correct"] = correct
        payload["correct_texts"] = list(q.correct_texts or [])
    return payload


def sort_law_groups(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key_fn(item: dict[str, Any]) -> tuple[int, Any]:
        key = str(item.get("key") or "")
        return (0, int(key)) if key.isdigit() else (1, key.lower())

    return sorted(items, key=key_fn)


def build_option_review(q: Any, chosen_index: int | None) -> list[dict[str, Any]]:
    correct_set = set(int(x) for x in (q.correct or []))
    items = []
    for idx, text in enumerate(q.choices or []):
        status = "plain"
        if (idx + 1) in correct_set:
            status = "correct"
        elif chosen_index is not None and idx == chosen_index:
            status = "chosen"
        items.append({"index": idx + 1, "text": text, "status": status})
    return items


def pretest_limit(meta: dict[str, Any]) -> int:
    if str(meta.get("kind") or "").lower() == "ok":
        if ok_extract_code(str(meta.get("module") or "")) == "ОК-17":
            return 70
    return 50


@dataclass
class RuntimeContext:
    store: Storage
    qb: QuestionBank
    admin_ids: set[int]
    bot_token: str
    webapp_url: str
    allow_dev_auth: bool
    auth_max_age_seconds: int
    bot: Bot | None = None
    dispatcher: Dispatcher | None = None
    polling_task: asyncio.Task | None = None


@dataclass
class AuthContext:
    telegram_user: dict[str, Any]
    user: dict[str, Any]
    user_id: int
    is_admin: bool


class OkModulesUpdate(BaseModel):
    modules: list[str] = Field(default_factory=list)


class StartLearningRequest(BaseModel):
    kind: Literal["law", "lawrand", "ok"]
    group_key: str | None = None
    part: int | None = 1
    module: str | None = None
    level: int | None = 1


class StartTestRequest(BaseModel):
    include_law: bool = True
    module_levels: dict[str, list[int]] = Field(default_factory=dict)


class SelectIndexRequest(BaseModel):
    index: int


class AnswerRequest(BaseModel):
    choice: int


class ReviewIndexRequest(BaseModel):
    index: int


class SubscriptionUpdateRequest(BaseModel):
    infinite: bool


class QuestionPatchRequest(BaseModel):
    question: str | None = None
    choices: list[str] | None = None
    correct: list[int] | None = None


def verify_init_data(init_data: str, bot_token: str, max_age_seconds: int) -> dict[str, Any]:
    parsed_pairs = urllib.parse.parse_qsl(init_data, keep_blank_values=True)
    if not parsed_pairs:
        require_http(401, "missing_init_data", "Не вистачає Telegram initData.")

    data = dict(parsed_pairs)
    received_hash = data.pop("hash", "")
    if not received_hash:
        require_http(401, "missing_hash", "Telegram initData не містить hash.")

    data_check_string = "\n".join(f"{key}={value}" for key, value in sorted(data.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    expected_hash = hmac.new(secret_key, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(received_hash, expected_hash):
        require_http(401, "bad_hash", "Не вдалося перевірити Telegram initData.")

    auth_date_raw = data.get("auth_date")
    if auth_date_raw and max_age_seconds > 0:
        try:
            auth_date = int(auth_date_raw)
        except ValueError:
            auth_date = 0
        if auth_date and (now().timestamp() - auth_date) > max_age_seconds:
            require_http(401, "expired_init_data", "Telegram initData застарів. Відкрийте Mini App ще раз.")

    user_raw = data.get("user")
    if not user_raw:
        require_http(401, "missing_user", "У Telegram initData немає користувача.")

    try:
        return json.loads(user_raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=401, detail={"code": "bad_user", "message": "Некоректний user у initData."}) from exc


async def get_runtime(request: Request) -> RuntimeContext:
    return request.app.state.runtime


async def get_auth_context(
    request: Request,
    runtime: RuntimeContext = Depends(get_runtime),
) -> AuthContext:
    debug_user_id = (
        request.headers.get("X-Debug-User-Id")
        or request.query_params.get("dev_user_id")
        or ""
    ).strip()

    telegram_user: dict[str, Any]
    if debug_user_id:
        if not runtime.allow_dev_auth:
            require_http(403, "debug_auth_disabled", "Debug-авторизацію вимкнено.")
        if not debug_user_id.isdigit():
            require_http(400, "bad_debug_user", "X-Debug-User-Id має бути числом.")
        telegram_user = {
            "id": int(debug_user_id),
            "first_name": request.headers.get("X-Debug-First-Name", "Dev"),
            "last_name": request.headers.get("X-Debug-Last-Name", "User"),
            "username": request.headers.get("X-Debug-Username", "dev_user"),
        }
    else:
        init_data = (
            request.headers.get("X-Telegram-Init-Data")
            or request.query_params.get("initData")
            or ""
        ).strip()
        if not runtime.bot_token:
            require_http(503, "missing_bot_token", "На сервері не задано BOT_TOKEN.")
        telegram_user = verify_init_data(init_data, runtime.bot_token, runtime.auth_max_age_seconds)

    user_id = int(telegram_user.get("id"))
    is_admin = user_id in runtime.admin_ids
    await runtime.store.ensure_user(
        user_id,
        is_admin=is_admin,
        first_name=telegram_user.get("first_name"),
        last_name=telegram_user.get("last_name"),
    )
    await runtime.store.start_trial_if_needed(
        user_id,
        first_name=telegram_user.get("first_name"),
        last_name=telegram_user.get("last_name"),
    )
    user = await runtime.store.get_user(user_id)

    return AuthContext(
        telegram_user=telegram_user,
        user=user,
        user_id=user_id,
        is_admin=is_admin,
    )


class MiniAppService:
    def __init__(self, runtime: RuntimeContext):
        self.runtime = runtime

    @property
    def store(self) -> Storage:
        return self.runtime.store

    @property
    def qb(self) -> QuestionBank:
        return self.runtime.qb

    async def get_state(self, user_id: int) -> dict[str, Any]:
        ui = await self.store.get_ui(user_id)
        return ui.get("state", {}) or {}

    async def set_state(self, user_id: int, state: dict[str, Any]) -> None:
        await self.store.set_state(user_id, state)

    def ensure_access(self, auth: AuthContext) -> None:
        if access_payload(auth.user)["has_access"]:
            return
        require_http(403, "access_expired", "Доступ завершився. Потрібна активна підписка.")

    def serialize_user(self, auth: AuthContext) -> dict[str, Any]:
        user = auth.user
        first_name = (user.get("first_name") or auth.telegram_user.get("first_name") or "").strip()
        last_name = (user.get("last_name") or auth.telegram_user.get("last_name") or "").strip()
        display_name = " ".join(x for x in [first_name, last_name] if x).strip() or f"ID {auth.user_id}"
        return {
            "id": auth.user_id,
            "first_name": first_name,
            "last_name": last_name,
            "display_name": display_name,
            "is_admin": auth.is_admin,
            "access": access_payload(user),
            "ok_modules": list(user.get("ok_modules", []) or []),
            "ok_last_levels": dict(user.get("ok_last_levels", {}) or {}),
            "created_at": dt_to_iso(user.get("created_at")),
        }

    def serialize_stats(self, raw: dict[str, Any]) -> dict[str, Any]:
        last = raw.get("last") or {}
        return {
            "count": int(raw.get("count", 0) or 0),
            "avg": float(raw.get("avg", 0.0) or 0.0),
            "last": {
                "finished_at": dt_to_iso(last.get("finished_at")),
                "finished_at_label": format_dt(last.get("finished_at")),
                "correct": int(last.get("correct", 0) or 0),
                "total": int(last.get("total", 0) or 0),
                "percent": float(last.get("percent", 0.0) or 0.0),
            }
            if last
            else None,
        }

    def serialize_catalog(self, auth: AuthContext) -> dict[str, Any]:
        law_groups = []
        for key, qids in self.qb.law_groups.items():
            law_groups.append(
                {
                    "key": key,
                    "title": clean_law_title(self.qb.law_group_title(key)),
                    "count": len(qids or []),
                }
            )

        user_modules = set(auth.user.get("ok_modules", []) or [])
        last_levels = dict(auth.user.get("ok_last_levels", {}) or {})
        ok_modules = []
        for name in sorted(self.qb.ok_modules.keys(), key=ok_sort_key):
            levels_map = self.qb.ok_modules.get(name, {}) or {}
            ok_modules.append(
                {
                    "name": name,
                    "code": ok_extract_code(name),
                    "label": ok_full_label(name),
                    "selected": name in user_modules,
                    "last_level": last_levels.get(name),
                    "levels": [
                        {
                            "level": int(level),
                            "count": len(levels_map.get(level, []) or []),
                        }
                        for level in sorted(levels_map.keys())
                    ],
                }
            )

        return {
            "law_groups": sort_law_groups(law_groups),
            "ok_modules": ok_modules,
            "counts": {
                "questions": len(self.qb.by_id),
                "law": len(self.qb.law),
                "ok_modules": len(self.qb.ok_modules),
            },
        }

    async def bootstrap(self, auth: AuthContext) -> dict[str, Any]:
        stats = await self.store.stats(auth.user_id)
        saved_view = await self.saved_view(auth)
        return {
            "user": self.serialize_user(auth),
            "links": {
                "group_url": GROUP_URL,
                "admin_url": get_admin_contact_url(self.runtime.admin_ids),
                "webapp_url": self.runtime.webapp_url,
            },
            "catalog": self.serialize_catalog(auth),
            "stats": self.serialize_stats(stats),
            "saved_view": saved_view,
        }

    async def saved_view(self, auth: AuthContext) -> dict[str, Any] | None:
        state = await self.get_state(auth.user_id)
        mode = str(state.get("mode") or "")
        if not mode:
            return None
        if mode == "pretest":
            return self.build_pretest_view(state, auth.is_admin)
        if mode in {"learn", "test", "mistakes"}:
            return await self.build_session_view(auth, state)
        if mode == "test_result":
            return self.build_test_result_view(state)
        if mode == "test_review":
            return await self.build_test_review_view(auth.user_id, state)
        return None

    def build_pretest_view(self, state: dict[str, Any], is_admin: bool) -> dict[str, Any]:
        qids = [int(x) for x in (state.get("qids", []) or [])]
        if not qids:
            return {"mode": "pretest", "screen": "empty", "message": "Немає питань для підготовки."}

        selected = clamp(int(state.get("selected", 0) or 0), 0, len(qids) - 1)
        qid = qids[selected]
        q = self.qb.by_id.get(qid)
        if not q:
            return {"mode": "pretest", "screen": "empty", "message": "Питання не знайдено."}

        return {
            "mode": "pretest",
            "screen": "preview",
            "header": state.get("header", ""),
            "meta": dict(state.get("meta", {}) or {}),
            "total": len(qids),
            "selected_index": selected,
            "question": serialize_question(q, reveal_answers=True),
            "is_admin": is_admin,
        }

    async def start_pretest(self, user_id: int, qids: list[int], header: str, meta: dict[str, Any]) -> None:
        limited = list(qids or [])[: pretest_limit(meta)]
        await self.set_state(
            user_id,
            {
                "mode": "pretest",
                "header": header,
                "qids": limited,
                "selected": 0,
                "meta": meta or {},
            },
        )

    async def start_learning_session(
        self,
        user_id: int,
        qids: list[int],
        header: str,
        meta: dict[str, Any],
    ) -> None:
        await self.set_state(
            user_id,
            {
                "mode": "learn",
                "header": header,
                "pending": list(qids or []),
                "skipped": [],
                "phase": "pending",
                "feedback": None,
                "current_qid": None,
                "correct_count": 0,
                "total": len(qids or []),
                "started_at": dt_to_iso(now()),
                "answers": {},
                "meta": meta or {},
            },
        )

    async def build_session_view(self, auth: AuthContext, state: dict[str, Any] | None = None) -> dict[str, Any]:
        state = state or await self.get_state(auth.user_id)
        mode = str(state.get("mode") or "")
        if mode not in {"learn", "test", "mistakes"}:
            return {"mode": "idle", "screen": "empty"}

        feedback = state.get("feedback")
        if feedback:
            qid = int(feedback.get("qid"))
            chosen = int(feedback.get("chosen"))
            q = self.qb.by_id.get(qid)
            if q:
                return {
                    "mode": mode,
                    "screen": "feedback",
                    "header": state.get("header", ""),
                    "question": {
                        "id": qid,
                        "question": q.question,
                        "options": build_option_review(q, chosen),
                    },
                    "actions": {"next": True, "can_edit": auth.is_admin},
                }

        pending = [int(x) for x in (state.get("pending", []) or [])]
        skipped = [int(x) for x in (state.get("skipped", []) or [])]
        phase = str(state.get("phase") or "pending")

        if not pending:
            if mode == "learn" and skipped:
                state["pending"] = skipped
                state["skipped"] = []
                state["phase"] = "skipped"
                await self.set_state(auth.user_id, state)
                return await self.build_session_view(auth, state)

            if mode == "test":
                return await self.finish_test(auth.user_id, state)
            if mode == "mistakes":
                return await self.finish_mistakes(auth.user_id, state)
            return await self.finish_learning(auth.user_id, state)

        while pending:
            q = self.qb.by_id.get(int(pending[0]))
            if q and (q.choices or []):
                break
            pending = pending[1:]

        if not pending:
            state["pending"] = []
            await self.set_state(auth.user_id, state)
            return await self.build_session_view(auth, state)

        qid = int(pending[0])
        q = self.qb.by_id[qid]

        state["pending"] = pending
        state["current_qid"] = qid
        await self.set_state(auth.user_id, state)

        total = int(state.get("total", len(pending)) or 0)
        current = (total - len(pending) + 1) if total else 1

        return {
            "mode": mode,
            "screen": "question",
            "header": state.get("header", ""),
            "progress": {"current": current, "total": total, "phase": phase},
            "question": serialize_question(q, reveal_answers=False),
            "actions": {"allow_skip": mode == "learn", "can_edit": auth.is_admin},
        }

    async def finish_learning(self, user_id: int, state: dict[str, Any]) -> dict[str, Any]:
        correct = int(state.get("correct_count", 0) or 0)
        total = int(state.get("total", 0) or 0)
        percent = (correct / total * 100.0) if total else 0.0
        meta = dict(state.get("meta", {}) or {})

        await self.set_state(user_id, {})
        return {
            "mode": "learn_result",
            "screen": "result",
            "summary": {
                "title": "Навчання завершено",
                "correct": correct,
                "total": total,
                "percent": percent,
                "meta": meta,
            },
        }

    async def finish_mistakes(self, user_id: int, state: dict[str, Any]) -> dict[str, Any]:
        answers = dict(state.get("answers", {}) or {})
        total = len(answers)
        correct_ids = [int(qid) for qid, ok in answers.items() if ok]
        wrong_ids = [int(qid) for qid, ok in answers.items() if not ok]

        for qid in correct_ids:
            await self.store.remove_mistake(user_id, qid)

        percent = (len(correct_ids) / total * 100.0) if total else 0.0
        await self.set_state(user_id, {})

        return {
            "mode": "mistakes_result",
            "screen": "result",
            "summary": {
                "title": "Робота над помилками завершена",
                "correct": len(correct_ids),
                "total": total,
                "percent": percent,
                "remaining": len(wrong_ids),
            },
        }

    async def finish_test(self, user_id: int, state: dict[str, Any]) -> dict[str, Any]:
        correct = int(state.get("correct_count", 0) or 0)
        total = int(state.get("total", 0) or 0)
        percent = (correct / total * 100.0) if total else 0.0
        passed = percent >= 60.0

        started_at = iso_to_dt(state.get("started_at")) or now()
        finished_at = now()
        await self.store.save_test(user_id, started_at, finished_at, total, correct)

        answers = dict(state.get("answers", {}) or {})
        chosen_map = dict(state.get("chosen", {}) or {})
        wrong_qids = [int(qid) for qid, ok in answers.items() if not ok]

        block_summary = []
        for name, qids in (state.get("test_blocks", {}) or {}).items():
            qid_list = [int(x) for x in (qids or [])]
            block_summary.append(
                {
                    "name": name,
                    "correct": sum(1 for qid in qid_list if answers.get(str(qid)) is True),
                    "total": len(qid_list),
                }
            )

        result_state = {
            "mode": "test_result",
            "summary": {
                "title": "Тестування завершено",
                "correct": correct,
                "total": total,
                "percent": percent,
                "passed": passed,
                "blocks": block_summary,
                "finished_at": dt_to_iso(finished_at),
            },
            "wrong_qids": wrong_qids,
            "chosen": chosen_map,
            "review_index": 0,
        }
        await self.set_state(user_id, result_state)
        return self.build_test_result_view(result_state)

    def build_test_result_view(self, state: dict[str, Any]) -> dict[str, Any]:
        summary = dict(state.get("summary", {}) or {})
        return {
            "mode": "test_result",
            "screen": "result",
            "summary": summary,
            "wrong_count": len(state.get("wrong_qids", []) or []),
        }

    async def build_test_review_view(self, user_id: int, state: dict[str, Any]) -> dict[str, Any]:
        wrong_qids = [int(x) for x in (state.get("wrong_qids", []) or [])]
        chosen_map = dict(state.get("chosen", {}) or {})

        while wrong_qids:
            index = clamp(int(state.get("review_index", 0) or 0), 0, len(wrong_qids) - 1)
            qid = wrong_qids[index]
            q = self.qb.by_id.get(qid)
            if q:
                chosen = chosen_map.get(str(qid))
                chosen_idx = int(chosen) if chosen is not None else None
                return {
                    "mode": "test_review",
                    "screen": "review",
                    "index": index,
                    "total": len(wrong_qids),
                    "question": {
                        "id": qid,
                        "question": q.question,
                        "options": build_option_review(q, chosen_idx),
                        "selected_missing": chosen is None,
                    },
                    "actions": {
                        "has_prev": index > 0,
                        "has_next": index < len(wrong_qids) - 1,
                    },
                }

            wrong_qids.pop(index)
            state["wrong_qids"] = wrong_qids
            state["review_index"] = min(index, max(0, len(wrong_qids) - 1))
            await self.set_state(user_id, state)

        return {"mode": "test_review", "screen": "empty", "message": "У цьому тесті немає помилок."}

    async def start_learning_flow(self, auth: AuthContext, payload: StartLearningRequest) -> dict[str, Any]:
        self.ensure_access(auth)

        if payload.kind == "law":
            group_key = (payload.group_key or "").strip()
            all_qids = list(self.qb.law_groups.get(group_key, []) or [])
            part = max(1, int(payload.part or 1))
            start = (part - 1) * 50
            qids = all_qids[start:start + 50]
            if not qids:
                require_http(404, "law_group_empty", "У цій частині немає питань.")
            title = self.qb.law_group_title(group_key)
            await self.start_pretest(auth.user_id, qids, clean_law_title(title), {"kind": "law", "group": group_key, "part": part})
            return self.build_pretest_view(await self.get_state(auth.user_id), auth.is_admin)

        if payload.kind == "lawrand":
            group_key = (payload.group_key or "").strip()
            all_qids = list(self.qb.law_groups.get(group_key, []) or [])
            if not all_qids:
                require_http(404, "law_group_empty", "У цьому розділі немає питань.")
            qids = self.qb.pick_random(all_qids, min(50, len(all_qids)))
            title = self.qb.law_group_title(group_key)
            await self.start_pretest(auth.user_id, qids, f"{clean_law_title(title)} • рандом", {"kind": "lawrand", "group": group_key})
            return self.build_pretest_view(await self.get_state(auth.user_id), auth.is_admin)

        module = (payload.module or "").strip()
        if not module:
            require_http(400, "missing_module", "Не вибрано модуль ОК.")
        user_modules = list(auth.user.get("ok_modules", []) or [])
        if module not in user_modules:
            require_http(403, "module_not_selected", "Цей модуль не обрано у вашому профілі.")

        level = int(payload.level or 1)
        levels_map = self.qb.ok_modules.get(module, {}) or {}
        if level not in levels_map:
            require_http(404, "level_not_found", "Для модуля не знайдено потрібний рівень.")

        await self.store.set_ok_last_level(auth.user_id, module, level)
        limit = 70 if ok_extract_code(module) == "ОК-17" else 50
        qids = list(levels_map.get(level, []) or [])[:limit]
        if not qids:
            require_http(404, "ok_level_empty", "Для цього рівня немає питань.")

        await self.start_pretest(auth.user_id, qids, f"{ok_full_label(module)} • Рівень {level}", {"kind": "ok", "module": module, "level": level})
        return self.build_pretest_view(await self.get_state(auth.user_id), auth.is_admin)

    async def set_ok_modules(self, auth: AuthContext, payload: OkModulesUpdate) -> dict[str, Any]:
        available = set(self.qb.ok_modules.keys())
        cleaned = [name for name in payload.modules if name in available]
        await self.store.set_ok_modules(auth.user_id, cleaned)
        auth.user = await self.store.get_user(auth.user_id)
        return {"user": self.serialize_user(auth), "catalog": self.serialize_catalog(auth)}

    async def start_test(self, auth: AuthContext, payload: StartTestRequest) -> dict[str, Any]:
        self.ensure_access(auth)

        include_law = bool(payload.include_law)
        law_qids = self.qb.pick_random(self.qb.law, 50) if include_law else []

        modules = sorted(list(auth.user.get("ok_modules", []) or []), key=ok_sort_key)
        last_levels = dict(auth.user.get("ok_last_levels", {}) or {})
        selected_map = dict(payload.module_levels or {})

        ok_qids: list[int] = []
        blocks: dict[str, list[int]] = {}
        if law_qids:
            blocks["Законодавство"] = list(law_qids)

        for module in modules:
            levels_map = self.qb.ok_modules.get(module, {}) or {}
            if not levels_map:
                continue
            available = sorted(levels_map.keys())
            raw = selected_map.get(module)
            if raw is None:
                raw = [int(last_levels.get(module, available[0]))]

            selected = []
            for level in raw:
                try:
                    selected.append(int(level))
                except (TypeError, ValueError):
                    continue
            selected = [level for level in sorted(set(selected)) if level in available]
            if raw == []:
                selected = []
            if not selected:
                continue

            pool: list[int] = []
            for level in selected:
                pool.extend(levels_map.get(level, []) or [])
            pool = list(dict.fromkeys(pool))

            picked = self.qb.pick_random(pool, 20)
            if picked:
                ok_qids.extend(picked)
                blocks[module] = list(picked)

        all_qids = list(law_qids) + ok_qids
        if not all_qids:
            require_http(400, "empty_test", "Оберіть хоча б один блок для тесту.")

        import random

        random.shuffle(all_qids)
        await self.set_state(
            auth.user_id,
            {
                "mode": "test",
                "header": "Тестування",
                "pending": all_qids,
                "skipped": [],
                "phase": "pending",
                "feedback": None,
                "current_qid": None,
                "correct_count": 0,
                "total": len(all_qids),
                "started_at": dt_to_iso(now()),
                "answers": {},
                "chosen": {},
                "test_blocks": blocks,
            },
        )
        return await self.build_session_view(auth)

    async def start_mistakes(self, auth: AuthContext) -> dict[str, Any]:
        self.ensure_access(auth)
        qids = await self.store.list_mistakes(auth.user_id)
        if not qids:
            return {"mode": "mistakes_empty", "screen": "empty", "message": "Поки що немає питань у блоці помилок."}

        await self.set_state(
            auth.user_id,
            {
                "mode": "mistakes",
                "header": "Робота над помилками",
                "pending": list(qids),
                "skipped": [],
                "phase": "pending",
                "feedback": None,
                "current_qid": None,
                "answers": {},
            },
        )
        return await self.build_session_view(auth)

    async def pretest_select(self, auth: AuthContext, payload: SelectIndexRequest) -> dict[str, Any]:
        self.ensure_access(auth)
        state = await self.get_state(auth.user_id)
        if state.get("mode") != "pretest":
            require_http(409, "no_pretest", "Немає активної підготовки.")
        qids = list(state.get("qids", []) or [])
        if not qids:
            require_http(409, "empty_pretest", "У підготовці немає питань.")
        state["selected"] = clamp(int(payload.index), 0, len(qids) - 1)
        await self.set_state(auth.user_id, state)
        return self.build_pretest_view(state, auth.is_admin)

    async def pretest_start(self, auth: AuthContext) -> dict[str, Any]:
        self.ensure_access(auth)
        state = await self.get_state(auth.user_id)
        if state.get("mode") != "pretest":
            require_http(409, "no_pretest", "Немає активної підготовки.")

        qids = list(state.get("qids", []) or [])
        if not qids:
            require_http(409, "empty_pretest", "У підготовці немає питань.")

        selected = clamp(int(state.get("selected", 0) or 0), 0, len(qids) - 1)
        ordered = qids[selected:] + qids[:selected]
        await self.start_learning_session(auth.user_id, ordered, state.get("header", "Навчання"), dict(state.get("meta", {}) or {}))
        return await self.build_session_view(auth)

    async def answer(self, auth: AuthContext, payload: AnswerRequest) -> dict[str, Any]:
        self.ensure_access(auth)
        state = await self.get_state(auth.user_id)
        mode = str(state.get("mode") or "")
        qid = state.get("current_qid")
        if mode not in {"learn", "test", "mistakes"} or not qid:
            require_http(409, "inactive_session", "Немає активної сесії.")

        q = self.qb.by_id.get(int(qid))
        if not q or not (q.choices or []):
            require_http(404, "question_missing", "Питання недоступне.")

        choice = int(payload.choice)
        if choice < 0 or choice >= len(q.choices or []):
            require_http(400, "bad_choice", "Невірний номер відповіді.")

        is_correct = (choice + 1) in set(int(x) for x in (q.correct or []))
        pending = [int(x) for x in (state.get("pending", []) or [])]
        if pending and int(pending[0]) == int(qid):
            pending = pending[1:]
        state["pending"] = pending

        if mode == "learn":
            if is_correct:
                state["correct_count"] = int(state.get("correct_count", 0) or 0) + 1
                state["feedback"] = None
            else:
                await self.store.bump_wrong(auth.user_id, int(qid))
                state["feedback"] = {"qid": int(qid), "chosen": choice}
            await self.set_state(auth.user_id, state)
            return await self.build_session_view(auth, state)

        if mode == "test":
            answers = dict(state.get("answers", {}) or {})
            chosen = dict(state.get("chosen", {}) or {})
            answers[str(qid)] = bool(is_correct)
            chosen[str(qid)] = choice
            state["answers"] = answers
            state["chosen"] = chosen
            if is_correct:
                state["correct_count"] = int(state.get("correct_count", 0) or 0) + 1
            await self.set_state(auth.user_id, state)
            return await self.build_session_view(auth, state)

        answers = dict(state.get("answers", {}) or {})
        answers[str(qid)] = bool(is_correct)
        state["answers"] = answers
        await self.set_state(auth.user_id, state)
        return await self.build_session_view(auth, state)

    async def skip(self, auth: AuthContext) -> dict[str, Any]:
        self.ensure_access(auth)
        state = await self.get_state(auth.user_id)
        if state.get("mode") != "learn":
            require_http(409, "skip_not_allowed", "Пропуск доступний лише в навчанні.")
        qid = state.get("current_qid")
        if not qid:
            require_http(409, "missing_current_qid", "Немає поточного питання.")

        pending = [int(x) for x in (state.get("pending", []) or [])]
        skipped = [int(x) for x in (state.get("skipped", []) or [])]
        if pending and int(pending[0]) == int(qid):
            pending = pending[1:]
        skipped.append(int(qid))

        state["pending"] = pending
        state["skipped"] = skipped
        state["feedback"] = None
        await self.set_state(auth.user_id, state)
        return await self.build_session_view(auth, state)

    async def feedback_next(self, auth: AuthContext) -> dict[str, Any]:
        self.ensure_access(auth)
        state = await self.get_state(auth.user_id)
        if state.get("mode") != "learn":
            require_http(409, "not_learning", "Поточна сесія не є навчанням.")
        state["feedback"] = None
        await self.set_state(auth.user_id, state)
        return await self.build_session_view(auth, state)

    async def leave_session(self, auth: AuthContext) -> dict[str, Any]:
        await self.set_state(auth.user_id, {})
        return {"mode": "idle", "screen": "empty"}

    async def open_test_review(self, auth: AuthContext) -> dict[str, Any]:
        state = await self.get_state(auth.user_id)
        wrong_qids = list(state.get("wrong_qids", []) or [])
        if not wrong_qids:
            require_http(409, "no_review", "У цьому тесті немає помилок.")
        state["mode"] = "test_review"
        state["review_index"] = 0
        await self.set_state(auth.user_id, state)
        return await self.build_test_review_view(auth.user_id, state)

    async def set_review_index(self, auth: AuthContext, payload: ReviewIndexRequest) -> dict[str, Any]:
        state = await self.get_state(auth.user_id)
        if str(state.get("mode") or "") not in {"test_review", "test_result"}:
            require_http(409, "no_review", "Немає активного перегляду помилок.")
        wrong_qids = list(state.get("wrong_qids", []) or [])
        if not wrong_qids:
            require_http(409, "no_review", "У цьому тесті немає помилок.")
        state["mode"] = "test_review"
        state["review_index"] = clamp(int(payload.index), 0, len(wrong_qids) - 1)
        await self.set_state(auth.user_id, state)
        return await self.build_test_review_view(auth.user_id, state)

    async def back_to_test_result(self, auth: AuthContext) -> dict[str, Any]:
        state = await self.get_state(auth.user_id)
        if not state.get("summary"):
            require_http(409, "no_test_result", "Немає результату тесту.")
        state["mode"] = "test_result"
        await self.set_state(auth.user_id, state)
        return self.build_test_result_view(state)

    async def list_admin_users(self, auth: AuthContext, offset: int, limit: int = 10) -> dict[str, Any]:
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")

        items = await self.store.list_users(offset, limit + 1)
        has_next = len(items) > limit
        page_items = items[:limit]
        counts = {"active": 0, "trial": 0, "expired": 0}
        serialized = []

        for user in page_items:
            access = access_payload(user)
            if access["state"] in {"sub_active", "sub_infinite"}:
                counts["active"] += 1
            elif access["state"] == "trial":
                counts["trial"] += 1
            else:
                counts["expired"] += 1
            serialized.append(
                {
                    "user_id": int(user.get("user_id")),
                    "first_name": user.get("first_name") or "",
                    "last_name": user.get("last_name") or "",
                    "display_name": " ".join(
                        x for x in [user.get("first_name") or "", user.get("last_name") or ""] if x
                    ).strip()
                    or "—",
                    "access": access,
                    "created_at": dt_to_iso(user.get("created_at")),
                }
            )

        return {
            "items": serialized,
            "offset": max(0, offset),
            "limit": limit,
            "has_next": has_next,
            "has_prev": offset > 0,
            "counts": counts,
        }

    async def admin_user_detail(self, auth: AuthContext, target_id: int) -> dict[str, Any]:
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")
        user = await self.store.get_user(target_id)
        if not user:
            require_http(404, "user_not_found", "Користувача не знайдено.")
        return {
            "user_id": target_id,
            "first_name": user.get("first_name") or "",
            "last_name": user.get("last_name") or "",
            "access": access_payload(user),
            "created_at": dt_to_iso(user.get("created_at")),
            "ok_modules": list(user.get("ok_modules", []) or []),
            "ok_last_levels": dict(user.get("ok_last_levels", {}) or {}),
        }

    async def admin_set_subscription(self, auth: AuthContext, target_id: int, infinite: bool) -> dict[str, Any]:
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")
        await self.store.set_subscription(target_id, None, infinite=infinite)
        return await self.admin_user_detail(auth, target_id)

    async def admin_questions_page(self, auth: AuthContext, page: int, page_size: int = 10) -> dict[str, Any]:
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")

        ids = sorted(int(qid) for qid in self.qb.by_id.keys())
        total = len(ids)
        if total == 0:
            return {"items": [], "page": 0, "pages": 0, "total": 0}

        pages = (total + page_size - 1) // page_size
        page = clamp(page, 0, pages - 1)
        start = page * page_size
        chunk = ids[start:start + page_size]

        items = []
        for qid in chunk:
            q = self.qb.by_id.get(qid)
            items.append({"id": qid, "question": (q.question or "").strip(), "topic": q.topic, "ok": q.ok, "level": q.level})

        return {"items": items, "page": page, "pages": pages, "total": total}

    async def admin_questions_search(self, auth: AuthContext, query: str, limit: int = 12) -> dict[str, Any]:
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")
        query = (query or "").strip()
        if len(query) < 3:
            require_http(400, "short_query", "Введіть щонайменше 3 символи для пошуку.")

        ids = find_question_ids_by_title(self.qb, query, limit=limit)
        items = []
        for qid in ids:
            q = self.qb.by_id.get(int(qid))
            if q:
                items.append({"id": int(qid), "question": q.question, "topic": q.topic, "ok": q.ok, "level": q.level})
        return {"query": query, "items": items}

    async def admin_question_detail(self, auth: AuthContext, qid: int) -> dict[str, Any]:
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")
        q = self.qb.by_id.get(int(qid))
        if not q:
            require_http(404, "question_not_found", "Питання не знайдено.")
        return {"question": serialize_question(q, reveal_answers=True)}

    async def admin_update_question(self, auth: AuthContext, qid: int, payload: QuestionPatchRequest) -> dict[str, Any]:
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")

        existing = await self.store.fetch_question(int(qid))
        if not existing:
            require_http(404, "question_not_found", "Питання не знайдено.")

        question_text = payload.question
        if question_text is not None:
            question_text = question_text.strip()
            if not question_text:
                require_http(400, "empty_question", "Текст питання не може бути порожнім.")

        choices = payload.choices
        if choices is not None:
            cleaned_choices = [str(item).strip() for item in choices if str(item).strip()]
            if len(cleaned_choices) < 2:
                require_http(400, "bad_choices", "Потрібно щонайменше 2 варіанти відповіді.")
            choices = cleaned_choices

        final_choices = choices if choices is not None else list(existing.get("choices") or [])
        correct = payload.correct
        if correct is not None:
            normalized_correct = []
            for value in correct:
                try:
                    normalized_correct.append(int(value))
                except (TypeError, ValueError):
                    continue
            correct = sorted(set(normalized_correct))

        final_correct = correct if correct is not None else [int(x) for x in (existing.get("correct") or [])]
        if not final_correct:
            require_http(400, "missing_correct", "Потрібно вибрати хоча б одну правильну відповідь.")
        if any(index < 1 or index > len(final_choices) for index in final_correct):
            require_http(400, "bad_correct", "Правильні відповіді виходять за межі варіантів.")

        after = await self.store.update_question_content(
            int(qid),
            question=question_text,
            choices=final_choices if choices is not None else None,
            correct=final_correct if (correct is not None or choices is not None) else None,
            changed_by=f"admin:{auth.user_id}",
        )
        if not after:
            require_http(404, "question_not_found", "Питання не знайдено.")

        q = self.qb.by_id.get(int(qid))
        if q:
            q.question = after.get("question") or q.question
            q.choices = list(after.get("choices") or q.choices)
            q.correct = [int(x) for x in (after.get("correct") or q.correct)]
            q.correct_texts = list(after.get("correct_texts") or q.correct_texts)

        return {"question": serialize_question(self.qb.by_id[int(qid)], reveal_answers=True)}


def build_bot_router(runtime: RuntimeContext) -> Router:
    router = Router()

    @router.message(F.text.startswith("/start"))
    async def start(message: Message) -> None:
        user_id = message.from_user.id
        is_admin = user_id in runtime.admin_ids
        await runtime.store.ensure_user(
            user_id,
            is_admin=is_admin,
            first_name=message.from_user.first_name,
            last_name=message.from_user.last_name,
        )
        await runtime.store.start_trial_if_needed(
            user_id,
            first_name=message.from_user.first_name,
            last_name=message.from_user.last_name,
        )

        text = (
            "<b>Mini App готовий</b>\n\n"
            "Уся користувацька частина й адмінка тепер відкриваються через вебінтерфейс."
        )
        markup = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="Відкрити Mini App", web_app=WebAppInfo(url=runtime.webapp_url))]
            ]
        )
        await message.answer(text, reply_markup=markup)

    return router


async def run_polling(runtime: RuntimeContext) -> None:
    if not runtime.bot:
        return
    assert runtime.dispatcher is not None
    await runtime.dispatcher.start_polling(runtime.bot)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bot_token = (os.getenv("BOT_TOKEN") or "").strip()
    admin_ids = parse_admin_ids(os.getenv("ADMIN_IDS", ""))

    dsn = (
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("POSTGRESQL_URL")
        or os.getenv("PGDATABASE_URL")
        or ""
    ).strip()
    if not dsn:
        raise RuntimeError("Set DATABASE_URL env var.")

    questions_path = resolve_questions_path()
    store = Storage(dsn)
    await store.init()

    auto_import = env_flag("QUESTIONS_AUTO_IMPORT", True)
    force_import = env_flag("QUESTIONS_FORCE_IMPORT", False)
    if auto_import:
        count = await store.questions_count()
        if force_import or count == 0:
            if questions_path.exists():
                await store.import_questions_from_json(str(questions_path), changed_by="bootstrap", force=force_import)
            elif count == 0:
                raise RuntimeError(f"Questions table is empty and file not found: {questions_path}")

    qb = QuestionBank(str(questions_path))
    await qb.load_from_db(store)
    if not qb.by_id:
        raise RuntimeError("No questions loaded from DB.")

    runtime = RuntimeContext(
        store=store,
        qb=qb,
        admin_ids=admin_ids,
        bot_token=bot_token,
        webapp_url=resolve_webapp_url(),
        allow_dev_auth=env_flag("ALLOW_DEV_AUTH", False),
        auth_max_age_seconds=int(os.getenv("WEBAPP_AUTH_MAX_AGE_SECONDS", "86400")),
    )

    if bot_token:
        runtime.bot = Bot(bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        runtime.dispatcher = Dispatcher()
        runtime.dispatcher.include_router(build_bot_router(runtime))
        runtime.polling_task = asyncio.create_task(run_polling(runtime))

    app.state.runtime = runtime
    try:
        yield
    finally:
        if runtime.polling_task:
            runtime.polling_task.cancel()
            with suppress(asyncio.CancelledError):
                await runtime.polling_task
        if runtime.bot:
            await runtime.bot.session.close()
        if runtime.store.pool:
            await runtime.store.pool.close()


app = FastAPI(title="Telegram Mini App", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/healthz")
async def healthcheck() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/bootstrap")
async def api_bootstrap(auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).bootstrap(auth)


@app.get("/api/stats")
async def api_stats(auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return MiniAppService(runtime).serialize_stats(await runtime.store.stats(auth.user_id))


@app.post("/api/preferences/ok-modules")
async def api_set_ok_modules(payload: OkModulesUpdate, auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).set_ok_modules(auth, payload)


@app.post("/api/learning/start")
async def api_learning_start(payload: StartLearningRequest, auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).start_learning_flow(auth, payload)


@app.post("/api/mistakes/start")
async def api_mistakes_start(auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).start_mistakes(auth)


@app.post("/api/pretest/select")
async def api_pretest_select(payload: SelectIndexRequest, auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).pretest_select(auth, payload)


@app.post("/api/pretest/start")
async def api_pretest_start(auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).pretest_start(auth)


@app.get("/api/session")
async def api_session(auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).saved_view(auth) or {"mode": "idle", "screen": "empty"}


@app.post("/api/session/answer")
async def api_session_answer(payload: AnswerRequest, auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).answer(auth, payload)


@app.post("/api/session/skip")
async def api_session_skip(auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).skip(auth)


@app.post("/api/session/next")
async def api_session_next(auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).feedback_next(auth)


@app.post("/api/session/leave")
async def api_session_leave(auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).leave_session(auth)


@app.post("/api/test/start")
async def api_test_start(payload: StartTestRequest, auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).start_test(auth, payload)


@app.post("/api/test/review/open")
async def api_test_review_open(auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).open_test_review(auth)


@app.post("/api/test/review/index")
async def api_test_review_index(payload: ReviewIndexRequest, auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).set_review_index(auth, payload)


@app.post("/api/test/review/back")
async def api_test_review_back(auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).back_to_test_result(auth)


@app.get("/api/admin/users")
async def api_admin_users(offset: int = 0, limit: int = 10, auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).list_admin_users(auth, max(0, offset), max(1, min(limit, 50)))


@app.get("/api/admin/users/{user_id}")
async def api_admin_user_detail(user_id: int, auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).admin_user_detail(auth, user_id)


@app.post("/api/admin/users/{user_id}/subscription")
async def api_admin_user_subscription(user_id: int, payload: SubscriptionUpdateRequest, auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).admin_set_subscription(auth, user_id, payload.infinite)


@app.get("/api/admin/questions")
async def api_admin_questions(page: int = 0, page_size: int = 10, auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).admin_questions_page(auth, page, max(1, min(page_size, 50)))


@app.get("/api/admin/questions/search")
async def api_admin_questions_search(q: str, limit: int = 12, auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).admin_questions_search(auth, q, max(1, min(limit, 50)))


@app.get("/api/admin/questions/{qid}")
async def api_admin_question_detail(qid: int, auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).admin_question_detail(auth, qid)


@app.patch("/api/admin/questions/{qid}")
async def api_admin_question_update(qid: int, payload: QuestionPatchRequest, auth: AuthContext = Depends(get_auth_context), runtime: RuntimeContext = Depends(get_runtime)):
    return await MiniAppService(runtime).admin_update_question(auth, qid, payload)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_FILE)
