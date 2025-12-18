from __future__ import annotations

import os
import json
import asyncio
import random
import hashlib
import re
import asyncpg
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from html import escape as hescape

from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    KeyboardButton,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder


try:
    TZ = ZoneInfo("Europe/Kyiv")
except Exception:
    # Fallback for environments without tzdata
    TZ = timezone.utc

def normalize_tme_url(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if s.startswith(("http://", "https://", "tg://")):
        return s
    if s.startswith("t.me/"):
        return "https://" + s
    if s.startswith("@"):
        return "https://t.me/" + s.lstrip("@")
    return s


# Links (can be overridden via env vars)
GROUP_URL = normalize_tme_url(os.getenv("GROUP_URL", "t.me/mytnytsia_test"))

# --- keys у state ---
ADMIN_PANEL_MSG_ID = "admin_panel_msg_id"
ADMIN_PANEL_CHAT_ID = "admin_panel_chat_id"

ADMIN_USERS_QUERY = "admin_users_query"
ADMIN_USERS_AWAITING = "awaiting"
ADMIN_USERS_BACK_OFFSET = "admin_users_back_offset"

def get_admin_contact_url(admin_ids: set[int]) -> str:
    """URL for 'contact admin' button.

    Priority:
    1) ADMIN_CONTACT_URL (full URL, 't.me/...', or '@username')
    2) ADMIN_USERNAME (username only or '@username')
    3) fallback to tg://user?id=<admin_id> (works in most Telegram clients)
    """
    url = normalize_tme_url(os.getenv("ADMIN_CONTACT_URL", ""))
    if url:
        return url

    username = (os.getenv("ADMIN_USERNAME", "") or "").strip().lstrip("@")
    if username:
        return f"https://t.me/{username}"

    if admin_ids:
        # Telegram deep link by numeric user id (client-side)
        admin_id = next(iter(admin_ids))
        return f"tg://user?id={admin_id}"

    return ""



def now() -> datetime:
    return datetime.now(TZ)


def dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def iso_to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s)


def clamp_callback(s: str, max_bytes: int = 64) -> str:
    """Return callback_data within Telegram's 1..64 bytes UTF-8 limit."""
    s = (s or "").strip()
    if not s:
        return "0"
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    return b[:max_bytes].decode("utf-8", errors="ignore")



def normalize_postgres_dsn(dsn: str) -> tuple[str, object | None]:
    """Normalize Railway/Heroku-style DATABASE_URL for asyncpg.

    - Accepts postgres:// or postgresql://
    - Extracts sslmode from query and maps it to asyncpg's ssl=... parameter
    """
    if not dsn:
        return dsn, None

    # asyncpg accepts both, but normalize to postgresql://
    if dsn.startswith("postgres://"):
        dsn = "postgresql://" + dsn[len("postgres://"):]

    parsed = urlparse(dsn)
    qs = parse_qs(parsed.query)

    sslmode = (qs.get("sslmode", [None])[0] or "").lower() if qs else ""
    ssl_param = None
    if sslmode in {"require", "verify-ca", "verify-full"}:
        ssl_param = True

    # Remove libpq-only params that asyncpg may not understand
    for k in ["sslmode"]:
        qs.pop(k, None)

    new_query = urlencode({k: v[0] for k, v in qs.items()}) if qs else ""
    cleaned = urlunparse(parsed._replace(query=new_query))

    return cleaned, ssl_param

# -------------------- DB --------------------

class Storage:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: asyncpg.Pool | None = None

    async def init(self):
        dsn, ssl_param = normalize_postgres_dsn(self.dsn)
        self.pool = await asyncpg.create_pool(dsn=dsn, ssl=ssl_param, min_size=1, max_size=10)

        async with self.pool.acquire() as con:
            await con.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    phone TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    is_admin INT DEFAULT 0,
                    trial_start TIMESTAMPTZ,
                    trial_end TIMESTAMPTZ,
                    sub_end TIMESTAMPTZ,
                    sub_infinite INT DEFAULT 0,
                    ok_modules_json TEXT DEFAULT '[]',
                    ok_last_levels_json TEXT DEFAULT '{}',
                    created_at TIMESTAMPTZ
                );
            """)

            # ---- migration: add new columns if DB already exists ----
            await con.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS first_name TEXT;")
            await con.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_name TEXT;")

            await con.execute("""
                CREATE TABLE IF NOT EXISTS ui_state (
                    user_id BIGINT PRIMARY KEY,
                    chat_id BIGINT,
                    main_message_id BIGINT,
                    state_json TEXT DEFAULT '{}'
                );
            """)
            await con.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    user_id BIGINT,
                    qid INT,
                    wrong_count INT DEFAULT 0,
                    in_mistakes INT DEFAULT 0,
                    PRIMARY KEY (user_id, qid)
                );
            """)
            await con.execute("""
                CREATE TABLE IF NOT EXISTS tests (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT,
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    total INT,
                    correct INT,
                    percent DOUBLE PRECISION
                );
            """)


            await con.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    id INT PRIMARY KEY,
                    section TEXT,
                    topic TEXT,
                    ok TEXT,
                    level INT,
                    qnum INT,
                    question TEXT NOT NULL,
                    choices JSONB NOT NULL,
                    correct JSONB NOT NULL,
                    correct_texts JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ DEFAULT now()
                );
            """)
            await con.execute("CREATE INDEX IF NOT EXISTS idx_questions_ok ON questions(ok);")
            await con.execute("CREATE INDEX IF NOT EXISTS idx_questions_level ON questions(level);")

            await con.execute("""
                CREATE TABLE IF NOT EXISTS question_revisions (
                    id BIGSERIAL PRIMARY KEY,
                    qid INT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
                    version INT NOT NULL,
                    changed_by TEXT,
                    changed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    before JSONB,
                    after JSONB NOT NULL
                );
            """)
            await con.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_question_revisions_qid_version ON question_revisions(qid, version);")
            await con.execute("CREATE INDEX IF NOT EXISTS idx_question_revisions_qid ON question_revisions(qid);")
    async def _fetchrow(self, sql: str, *params):
        assert self.pool
        async with self.pool.acquire() as con:
            return await con.fetchrow(sql, *params)

    async def _fetch(self, sql: str, *params):
        assert self.pool
        async with self.pool.acquire() as con:
            return await con.fetch(sql, *params)

    async def _exec(self, sql: str, *params):
        assert self.pool
        async with self.pool.acquire() as con:
            return await con.execute(sql, *params)

    async def ensure_user(
            self,
            user_id: int,
            is_admin: bool = False,
            first_name: Optional[str] = None,
            last_name: Optional[str] = None,
    ):
        # normalize: treat empty strings as NULL
        fn = (first_name or "").strip() or None
        ln = (last_name or "").strip() or None

        await self._exec("""
            INSERT INTO users (user_id, is_admin, created_at, first_name, last_name)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (user_id) DO UPDATE SET
                is_admin=EXCLUDED.is_admin,
                first_name=COALESCE(EXCLUDED.first_name, users.first_name),
                last_name=COALESCE(EXCLUDED.last_name, users.last_name)
        """, user_id, 1 if is_admin else 0, now(), fn, ln)

    async def show_registration_gate(bot: Bot, store: Storage, uid: int, chat_id: int,
                                     message: Optional[Message] = None):
        text, kb = screen_need_registration()
        await render_main(bot, store, uid, chat_id, text, kb, message=message)
        await show_contact_request(bot, store, uid, chat_id)

    async def get_user(self, user_id: int) -> dict:
        r = await self._fetchrow("SELECT * FROM users WHERE user_id=$1", user_id)
        if not r:
            return {}
        d = dict(r)
        d["ok_modules"] = json.loads(d.get("ok_modules_json") or "[]")
        d["ok_last_levels"] = json.loads(d.get("ok_last_levels_json") or "{}")
        return d

    async def set_phone_and_trial(
            self,
            user_id: int,
            phone: str,
            first_name: Optional[str] = None,
            last_name: Optional[str] = None,
    ):
        ts = now()
        te = ts + timedelta(days=3)
        fn = (first_name or "").strip() or None
        ln = (last_name or "").strip() or None

        await self._exec("""
            UPDATE users
            SET phone=$1, trial_start=$2, trial_end=$3,
                first_name=COALESCE($4, first_name),
                last_name=COALESCE($5, last_name)
            WHERE user_id=$6
        """, phone, ts, te, fn, ln, user_id)

    async def set_ok_modules(self, user_id: int, modules: list[str]):
        await self._exec("""
            UPDATE users SET ok_modules_json=$1 WHERE user_id=$2
        """, json.dumps(modules, ensure_ascii=False), user_id)

    async def set_ok_last_level(self, user_id: int, module: str, level: int):
        user = await self.get_user(user_id)
        mp = user.get("ok_last_levels", {}) or {}
        mp[str(module)] = int(level)
        await self._exec("""
            UPDATE users SET ok_last_levels_json=$1 WHERE user_id=$2
        """, json.dumps(mp, ensure_ascii=False), user_id)

    async def set_subscription(self, user_id: int, sub_end: datetime | None, infinite: bool):
        await self._exec("""
            UPDATE users SET sub_end=$1, sub_infinite=$2 WHERE user_id=$3
        """, sub_end, 1 if infinite else 0, user_id)

    async def get_ui(self, user_id: int) -> dict:
        r = await self._fetchrow("SELECT * FROM ui_state WHERE user_id=$1", user_id)
        if not r:
            return {}
        d = dict(r)
        d["state"] = json.loads(d.get("state_json") or "{}")
        return d

    async def set_ui(self, user_id: int, chat_id: int, main_message_id: int):
        await self._exec("""
            INSERT INTO ui_state (user_id, chat_id, main_message_id, state_json)
            VALUES ($1, $2, $3, '{}')
            ON CONFLICT (user_id) DO UPDATE SET chat_id=EXCLUDED.chat_id, main_message_id=EXCLUDED.main_message_id
        """, user_id, chat_id, main_message_id)

    async def set_state(self, user_id: int, state: dict):
        await self._exec("""
            INSERT INTO ui_state (user_id, chat_id, main_message_id, state_json)
            VALUES ($1, NULL, NULL, $2)
            ON CONFLICT (user_id) DO UPDATE SET state_json=EXCLUDED.state_json
        """, user_id, json.dumps(state, ensure_ascii=False))

    async def bump_wrong(self, user_id: int, qid: int) -> tuple[int, int]:
        r = await self._fetchrow("SELECT wrong_count, in_mistakes FROM errors WHERE user_id=$1 AND qid=$2", user_id, qid)
        if not r:
            await self._exec("INSERT INTO errors (user_id, qid, wrong_count, in_mistakes) VALUES ($1,$2,1,0)", user_id, qid)
            return 1, 0
        wc = int(r["wrong_count"]) + 1
        im = int(r["in_mistakes"])
        if wc >= 5:
            im = 1
        await self._exec("UPDATE errors SET wrong_count=$1, in_mistakes=$2 WHERE user_id=$3 AND qid=$4", wc, im, user_id, qid)
        return wc, im

    async def list_mistakes(self, user_id: int) -> list[int]:
        rows = await self._fetch("SELECT qid FROM errors WHERE user_id=$1 AND in_mistakes=1", user_id)
        return [int(r["qid"]) for r in rows]

    async def remove_mistake(self, user_id: int, qid: int):
        await self._exec("DELETE FROM errors WHERE user_id=$1 AND qid=$2", user_id, qid)

    async def save_test(self, user_id: int, started_at: datetime, finished_at: datetime, total: int, correct: int):
        percent = (correct / total * 100.0) if total else 0.0
        await self._exec("""
            INSERT INTO tests (user_id, started_at, finished_at, total, correct, percent)
            VALUES ($1,$2,$3,$4,$5,$6)
        """, user_id, started_at, finished_at, total, correct, percent)

    async def stats(self, user_id: int) -> dict:
        rows = await self._fetch("""
            SELECT * FROM tests WHERE user_id=$1 ORDER BY id DESC LIMIT 50
        """, user_id)
        if not rows:
            return {"count": 0, "avg": 0.0, "last": None}
        perc = [float(r["percent"]) for r in rows]
        last = dict(rows[0])
        return {"count": len(rows), "avg": sum(perc)/len(perc), "last": last}

    async def list_users(self, offset: int, limit: int) -> list[dict]:
        rows = await self._fetch("""
            SELECT user_id, phone, first_name, last_name, trial_end, sub_end, sub_infinite, created_at
            FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2
        """, limit, offset)
        return [dict(r) for r in rows]

    async def search_users_by_phone(self, phone_digits: str, offset: int, limit: int) -> list[dict]:
        q = "".join(ch for ch in (phone_digits or "").strip() if ch.isdigit())
        if not q:
            return []
        rows = await self._fetch("""
            SELECT user_id, phone, first_name, last_name, trial_end, sub_end, sub_infinite, created_at
            FROM users
            WHERE regexp_replace(COALESCE(phone,''), '\\D', '', 'g') LIKE '%' || $1 || '%'
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
        """, q, limit, offset)
        return [dict(r) for r in rows]
# -------------------- Questions (DB) --------------------

async def questions_count(self) -> int:
    r = await self._fetchrow("SELECT COUNT(*) AS c FROM questions")
    return int(r["c"]) if r else 0

async def fetch_questions(self) -> list[dict]:
    rows = await self._fetch("""
        SELECT id, section, topic, ok, level, qnum, question, choices, correct, correct_texts
        FROM questions
        ORDER BY id
    """)
    return [dict(r) for r in rows]

async def import_questions_from_json(
    self,
    path: str,
    *,
    changed_by: str = "import",
    force: bool = False,
) -> int:
    """
    Імпорт/синхронізація питань з JSON у таблицю questions.
    Підтримує повторний запуск:
    - якщо питання не змінилось — revision не створюємо
    - якщо force=True — перезаписуємо і створюємо revision завжди
    """
    if not path or not os.path.exists(path):
        raise RuntimeError(f"Questions file not found: {path}")

    qb = QuestionBank(path)
    qb.load()

    items = [q for q in qb.by_id.values() if q.is_valid_mcq]
    if not items:
        return 0

    def _norm_json(v: Any):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return v
        return v

    assert self.pool
    imported = 0

    async with self.pool.acquire() as con:
        async with con.transaction():
            for q in items:
                existing = await con.fetchrow(
                    """
                    SELECT id, section, topic, ok, level, qnum, question, choices, correct, correct_texts
                    FROM questions
                    WHERE id=$1
                    """,
                    int(q.id),
                )

                before = dict(existing) if existing else None
                if before:
                    before["choices"] = _norm_json(before.get("choices"))
                    before["correct"] = _norm_json(before.get("correct"))
                    before["correct_texts"] = _norm_json(before.get("correct_texts"))

                after = {
                    "id": int(q.id),
                    "section": q.section or "",
                    "topic": q.topic or "",
                    "ok": q.ok,
                    "level": int(q.level) if q.level is not None else None,
                    "qnum": int(q.qnum) if q.qnum is not None else None,
                    "question": q.question or "",
                    "choices": list(q.choices or []),
                    "correct": [int(x) for x in (q.correct or [])],
                    "correct_texts": list(q.correct_texts or []),
                }

                changed = True
                if before:
                    cmp_before = {k: before.get(k) for k in after.keys()}
                    changed = (cmp_before != after)

                if (not existing) or changed or force:
                    await con.execute(
                        """
                        INSERT INTO questions (id, section, topic, ok, level, qnum, question, choices, correct, correct_texts, updated_at)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb,$10::jsonb, now())
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
                        after["id"],
                        after["section"],
                        after["topic"],
                        after["ok"],
                        after["level"],
                        after["qnum"],
                        after["question"],
                        json.dumps(after["choices"], ensure_ascii=False),
                        json.dumps(after["correct"], ensure_ascii=False),
                        json.dumps(after["correct_texts"], ensure_ascii=False),
                    )

                    # revision
                    prev_ver = await con.fetchval(
                        "SELECT COALESCE(MAX(version), 0) FROM question_revisions WHERE qid=$1",
                        after["id"],
                    )
                    ver = int(prev_ver or 0) + 1

                    await con.execute(
                        """
                        INSERT INTO question_revisions (qid, version, changed_by, before, after)
                        VALUES ($1,$2,$3,$4::jsonb,$5::jsonb)
                        """,
                        after["id"],
                        ver,
                        (changed_by or None),
                        (json.dumps(before, ensure_ascii=False) if before else None),
                        json.dumps(after, ensure_ascii=False),
                    )

                    imported += 1

    return imported



# -------------------- Question bank --------------------

@dataclass
class Q:
    id: int
    section: str
    topic: str
    ok: Optional[str]
    level: Optional[int]
    qnum: Optional[int]          # <-- НОВЕ: порядковий номер питання (1..N)
    question: str
    choices: List[str]
    correct: List[int]
    correct_texts: List[str]

    @property
    def is_valid_mcq(self) -> bool:
        return bool(self.choices) and isinstance(self.correct, list) and len(self.correct) > 0


class QuestionBank:
    """Loads a questions file and нормалізує структуру до формату Q.

    Підтримує кілька форматів JSON:
    1) Плоский список питань (старий формат)
    2) {"questions": [...]}
    3) {"law": [...], "ok": ...} (де ok може бути list або dict модуль->рівні->питання)
    4) {"sections": [...]} (примітивна підтримка вкладених секцій)
    """

    def __init__(self, path: str):
        self.path = path
        self.by_id: Dict[int, Q] = {}
        self.law: List[int] = []
        self.law_groups: Dict[str, List[int]] = {}        # key -> qids
        self.ok_modules: Dict[str, Dict[int, List[int]]] = {}  # ok(module name) -> level -> qids

        # для UI: короткі ключі груп законодавства -> заголовок
        self._law_group_titles: Dict[str, str] = {}

    # ---------- public ----------
    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.by_id.clear()
        self.law.clear()
        self.law_groups.clear()
        self.ok_modules.clear()
        self._law_group_titles.clear()

        for item in self._iter_raw_questions(raw):
            norm = self._normalize_item(item)
            if not norm:
                continue

            q = Q(
                id=norm["id"],
                section=norm.get("section", ""),
                topic=norm.get("topic", ""),
                ok=norm.get("ok"),
                level=norm.get("level"),
                qnum=norm.get("qnum"),  # <-- НОВЕ
                question=norm.get("question", ""),
                choices=norm.get("choices", []),
                correct=norm.get("correct", []),
                correct_texts=norm.get("correct_texts", []),
            )

            # уникальний id (на випадок колізій при хешуванні)
            if q.id in self.by_id:
                nid = q.id
                while nid in self.by_id:
                    nid += 1
                q.id = nid

            self.by_id[q.id] = q

        # ---- indexes ----

        for qid, q in self.by_id.items():
            if not q.is_valid_mcq:
                continue

            # New flat DB logic:
            # - If there is an OK module -> it's an OK question
            # - Otherwise -> legislation ("Законодавство")
            sec = (q.section or "").lower()
            is_ok = bool(q.ok) or ("операцій" in sec and "компет" in sec)

            if not is_ok:
                self.law.append(qid)
                key = self._law_group_key(q.topic or q.section)
                self.law_groups.setdefault(key, []).append(qid)

            if is_ok:
                mod = q.ok or "ОК"
                self.ok_modules.setdefault(mod, {})
                lvl = int(q.level or 1)
                self.ok_modules[mod].setdefault(lvl, []).append(qid)

        def _ord_key(qid: int):
            qq = self.by_id.get(qid)
            n = getattr(qq, "qnum", None)
            # якщо номера нема — кидаємо в кінець, щоб не ламати порядок
            return (n if isinstance(n, int) else 10 ** 9, int(qid))

        # stable order (by question number)
        for k in self.law_groups:
            self.law_groups[k].sort(key=_ord_key)

        self.law.sort(key=_ord_key)

        for ok in self.ok_modules:
            for lvl in self.ok_modules[ok]:
                self.ok_modules[ok][lvl].sort(key=_ord_key)


async def load_from_db(self, store: "Storage"):
    """
    Завантаження питань з Postgres у памʼять (для швидкого UI).
    Очікується, що таблиця questions вже заповнена.
    """
    rows = await store.fetch_questions()

    self.by_id.clear()
    self.law.clear()
    self.law_groups.clear()
    self.ok_modules.clear()
    self._law_group_titles.clear()

    def _norm_json(v: Any):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return v
        return v

    for r in rows:
        rid = int(r.get("id"))
        q = Q(
            id=rid,
            section=(r.get("section") or ""),
            topic=(r.get("topic") or ""),
            ok=r.get("ok"),
            level=r.get("level"),
            qnum=r.get("qnum"),
            question=(r.get("question") or ""),
            choices=_norm_json(r.get("choices")) or [],
            correct=_norm_json(r.get("correct")) or [],
            correct_texts=_norm_json(r.get("correct_texts")) or [],
        )
        self.by_id[q.id] = q

    # ---- indexes ---- (same logic as .load())

    for qid, q in self.by_id.items():
        if not q.is_valid_mcq:
            continue

        sec = (q.section or "").lower()
        is_ok = bool(q.ok) or ("операцій" in sec and "компет" in sec)

        if not is_ok:
            self.law.append(qid)
            key = self._law_group_key(q.topic or q.section)
            self.law_groups.setdefault(key, []).append(qid)

        if is_ok:
            mod = q.ok or "ОК"
            self.ok_modules.setdefault(mod, {})
            lvl = int(q.level or 1)
            self.ok_modules[mod].setdefault(lvl, []).append(qid)

    def _ord_key(qid: int):
        qq = self.by_id.get(qid)
        n = getattr(qq, "qnum", None)
        return (n if isinstance(n, int) else 10 ** 9, int(qid))

    for k in self.law_groups:
        self.law_groups[k].sort(key=_ord_key)

    self.law.sort(key=_ord_key)

    for ok in self.ok_modules:
        for lvl in self.ok_modules[ok]:
            self.ok_modules[ok][lvl].sort(key=_ord_key)



    def law_group_title(self, key: str) -> str:
        # hashed key -> title
        if key in self._law_group_titles:
            return self._law_group_titles[key]

        # numeric key -> try to find any topic starting with "key."
        if key.isdigit():
            for qid in self.law_groups.get(key, []):
                t = (self.by_id[qid].topic or "").strip()
                if t.startswith(f"{key}."):
                    return t.split(".", 1)[1].strip()
        return key

    def pick_random(self, qids: List[int], n: int) -> List[int]:
        if len(qids) <= n:
            return list(qids)
        return random.sample(qids, n)

    # ---------- internals ----------
    def _iter_raw_questions(self, raw: Any):
        # 1) старий формат: list[dict]
        if isinstance(raw, list):
            for it in raw:
                if isinstance(it, dict):
                    yield it
            return

        if not isinstance(raw, dict):
            return

        # 2) {"questions": [...]}
        qlist = raw.get("questions") or raw.get("items")
        if isinstance(qlist, list):
            for it in qlist:
                if isinstance(it, dict):
                    yield it

        # 3) law / legislation
        # - list: [{"question":...}, ...]
        # - dict (new flat): {"<group title>": [ ... ], ...}
        law = raw.get("law") or raw.get("laws") or raw.get("legislation")
        if isinstance(law, list):
            for it in law:
                if isinstance(it, dict):
                    it = dict(it)
                    it.setdefault("section", "Законодавство")
                    yield it

        if isinstance(law, dict):
            for law_title, arr in law.items():
                if not isinstance(arr, list):
                    continue
                for it in arr:
                    if not isinstance(it, dict):
                        continue
                    it = dict(it)
                    it.setdefault("section", "Законодавство")
                    it.setdefault("topic", str(law_title))
                    yield it

        # 4) OK / operational competencies
        ok = (
            raw.get("ok")
            or raw.get("ok_questions")
            or raw.get("ok_modules")
            or raw.get("operational_competencies")
            or raw.get("operationalCompetencies")
        )

        if isinstance(ok, list):
            for it in ok:
                if isinstance(it, dict):
                    it = dict(it)
                    it.setdefault("section", "ОК")
                    yield it

        # ok як dict:
        # A) module -> (dict level->list | list)  [old]
        # B) module -> {"name": "...", "levels": {level->list}} [new]
        if isinstance(ok, dict):
            for module_name, v in ok.items():
                # new structure: {"name": "...", "levels": {...}}
                if isinstance(v, dict) and isinstance(v.get("levels"), dict):
                    module_title = v.get("name") or v.get("title") or v.get("module_name") or ""
                    levels_dict = v.get("levels") or {}
                    for lvl, arr in levels_dict.items():
                        if not isinstance(arr, list):
                            continue
                        for it in arr:
                            if not isinstance(it, dict):
                                continue
                            it = dict(it)
                            it.setdefault("section", "ОК")
                            it.setdefault("ok", str(module_name))
                            it.setdefault("level", lvl)
                            if module_title:
                                it.setdefault("topic", str(module_title))
                            yield it
                    continue

                # old structure: module -> dict(level -> list[dict])
                if isinstance(v, dict):
                    for lvl, arr in v.items():
                        if not isinstance(arr, list):
                            continue
                        for it in arr:
                            if not isinstance(it, dict):
                                continue
                            it = dict(it)
                            it.setdefault("section", "ОК")
                            it.setdefault("ok", str(module_name))
                            it.setdefault("level", lvl)
                            yield it
                elif isinstance(v, list):
                    for it in v:
                        if not isinstance(it, dict):
                            continue
                        it = dict(it)
                        it.setdefault("section", "ОК")
                        it.setdefault("ok", str(module_name))
                        yield it

        # 5) {"sections": [...]} (примітивна підтримка вкладених секцій)
        secs = raw.get("sections")
        if isinstance(secs, list):
            for sec in secs:
                if not isinstance(sec, dict):
                    continue
                sec_name = sec.get("name") or sec.get("title") or sec.get("section") or "Секція"
                sec_q = sec.get("questions") or sec.get("items")
                if isinstance(sec_q, list):
                    for it in sec_q:
                        if not isinstance(it, dict):
                            continue
                        it = dict(it)
                        it.setdefault("section", str(sec_name))
                        yield it
                # або: {"topics":[{"name":...,"questions":[...]}]}
                topics = sec.get("topics")
                if isinstance(topics, list):
                    for tp in topics:
                        if not isinstance(tp, dict):
                            continue
                        topic_name = tp.get("name") or tp.get("title") or tp.get("topic") or ""
                        tp_q = tp.get("questions") or tp.get("items")
                        if isinstance(tp_q, list):
                            for it in tp_q:
                                if not isinstance(it, dict):
                                    continue
                                it = dict(it)
                                it.setdefault("section", str(sec_name))
                                it.setdefault("topic", str(topic_name))
                                yield it

    def _normalize_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        # ---- question text ----
        qtext = (
                item.get("question_text")
                or item.get("questionText")
                or item.get("question")
                or item.get("q")
                or item.get("text")
                or item.get("title")
                or ""
        )
        qtext = str(qtext).strip()
        if not qtext:
            return None

        # ---- meta (section/topic/ok/level) ----
        section = str(item.get("section") or item.get("category") or item.get("type") or "").strip()
        topic = str(item.get("topic") or item.get("group") or item.get("chapter") or "").strip()
        if not topic:
            # для законодавства з твого формату section=Законодавство, topic=назва закону (виставляється вище)
            topic = section

        ok = item.get("ok") or item.get("module") or item.get("ok_module") or item.get("okModule")
        ok = str(ok).strip() if ok is not None else None
        if ok == "":
            ok = None

        # якщо окремо лежать ok_code / ok_name — підхопимо
        ok_code = item.get("ok_code") or item.get("okCode")
        ok_name = item.get("ok_name") or item.get("okName")
        if ok is None:
            if ok_code is not None and str(ok_code).strip():
                ok = str(ok_code).strip()
            elif ok_name is not None and str(ok_name).strip():
                ok = str(ok_name).strip()

        # ---- level (int) ----
        lvl_raw = item.get("level") or item.get("lvl") or item.get("difficulty") or item.get("diff")
        level: Optional[int] = None
        try:
            if lvl_raw is not None and str(lvl_raw).strip() != "":
                level = int(str(lvl_raw).strip())
        except Exception:
            level = None

        # ---- question number (order) ----
        qnum_raw = (
                item.get("question_number")
                or item.get("questionNumber")
                or item.get("number")  # твій формат
                or item.get("num")
                or item.get("no")
        )
        qnum: Optional[int] = None
        try:
            if qnum_raw is not None and str(qnum_raw).strip() != "":
                qnum = int(str(qnum_raw).strip())
        except Exception:
            qnum = None

        # ---- choices / answers ----
        choices_raw = (
                item.get("choices")
                or item.get("options")
                or item.get("answers")  # твій формат
                or item.get("variants")
                or item.get("variants_list")
                or []
        )

        # підтримка dict варіантів {"A": "...", "B": "..."}
        if isinstance(choices_raw, dict):
            choices_raw = list(choices_raw.values())

        choices: List[str] = []
        inferred_correct: List[int] = []

        if isinstance(choices_raw, list) and choices_raw and all(isinstance(x, dict) for x in choices_raw):
            # формат: [{"text": "...", "is_correct": true}, ...]
            for d in choices_raw:
                t = str(d.get("text") or d.get("answer") or d.get("value") or d.get("title") or "").strip()
                if not t:
                    continue
                choices.append(t)
                flag = d.get("is_correct")
                if isinstance(flag, bool) and flag:
                    inferred_correct.append(len(choices))  # 1-based
        elif isinstance(choices_raw, list):
            for ch in choices_raw:
                txt = str(ch).strip()
                if txt:
                    choices.append(txt)

        if not choices:
            return None

        # ---- correct (list[int]) ----
        correct: List[int] = []

        # 1) з answers[].is_correct
        if inferred_correct:
            correct = sorted(set(inferred_correct))

        # 2) correct_index / correct_answer_index (0-based)
        if not correct:
            idx0 = (
                    item.get("correct_answer_index")
                    or item.get("correctAnswerIndex")
                    or item.get("correct_index")  # твій формат
                    or item.get("correctIndex")
            )
            try:
                if idx0 is not None and str(idx0).strip() != "":
                    i0 = int(str(idx0).strip())
                    if 0 <= i0 < len(choices):
                        correct = [i0 + 1]
            except Exception:
                pass

        # 3) correct_answer_indices (0-based list)
        if not correct:
            idxs0 = item.get("correct_answer_indices") or item.get("correctAnswerIndices")
            if isinstance(idxs0, list):
                tmp: List[int] = []
                for v in idxs0:
                    try:
                        i0 = int(v)
                        if 0 <= i0 < len(choices):
                            tmp.append(i0 + 1)
                    except Exception:
                        continue
                if tmp:
                    correct = sorted(set(tmp))

        # 4) legacy: correct як 1-based числа/список/маска
        if not correct:
            correct_raw = (
                    item.get("correct")
                    or item.get("correct_answers")
                    or item.get("correctAnswers")
                    or item.get("right")
                    or item.get("right_answers")
                    or item.get("answer")
            )

            if isinstance(correct_raw, list) and correct_raw and all(isinstance(x, bool) for x in correct_raw):
                correct = [i + 1 for i, flag in enumerate(correct_raw) if flag]
            else:
                nums: List[int] = []
                if isinstance(correct_raw, int):
                    nums = [correct_raw]
                elif isinstance(correct_raw, str):
                    nums = [int(x) for x in re.findall(r"\d+", correct_raw)]
                elif isinstance(correct_raw, list):
                    for x in correct_raw:
                        if isinstance(x, int):
                            nums.append(x)
                        elif isinstance(x, str) and x.strip().isdigit():
                            nums.append(int(x.strip()))
                # вважаємо, що це 1-based (якщо файл інший)
                nums = [n for n in nums if 1 <= n <= len(choices)]
                if nums:
                    correct = sorted(set(nums))

        # ---- correct_texts ----
        correct_texts = item.get("correct_texts") or item.get("correctTexts") or []
        if isinstance(correct_texts, str):
            correct_texts = [correct_texts]
        if isinstance(correct_texts, list):
            correct_texts = [str(x).strip() for x in correct_texts if str(x).strip()]
        else:
            correct_texts = []

        if not correct_texts and correct:
            correct_texts = [choices[i - 1] for i in correct if 1 <= i <= len(choices)]

        # ---- ids ----
        raw_id = item.get("id") or item.get("uid") or item.get("qid") or item.get("question_id")
        qid = self._make_int_id(raw_id, fallback=(raw_id, section, topic, qnum, qtext))

        return {
            "id": int(qid),
            "section": section,
            "topic": topic,
            "ok": ok,
            "level": level,
            "qnum": qnum,
            "question": qtext,
            "choices": choices,
            "correct": correct,  # <-- завжди list[int]
            "correct_texts": correct_texts,
        }

        def _clean_ok_name(name: Any) -> str:
            s = str(name or "").strip()
            # remove trailing "Рівень N" to avoid duplicates with levels menu
            s = re.sub(r"\s*Рівень\s*\d+\s*$", "", s, flags=re.IGNORECASE).strip()
            return s

        if ok is None and (ok_code or ok_name):
            code = str(ok_code or "").strip()
            name = _clean_ok_name(ok_name)
            if code and name:
                ok = f"{code} — {name}"
            elif code:
                ok = code
            elif name:
                ok = name

        lvl_raw = item.get("level") or item.get("lvl") or item.get("difficulty")
        level: Optional[int] = None
        try:
            if lvl_raw is not None and str(lvl_raw).strip() != "":
                level = int(lvl_raw)
        except Exception:
            level = None

        # ---- choices/answers ----
        choices_raw = item.get("choices") or item.get("answers") or item.get("options") or item.get("variants") or []
        if isinstance(choices_raw, dict):
            choices_raw = list(choices_raw.values())

        choices: List[str] = []
        inferred_correct: List[int] = []

        if isinstance(choices_raw, list) and choices_raw and all(isinstance(x, dict) for x in choices_raw):
            # new format: [{"text": "...", "is_correct": true}, ...]
            for d in choices_raw:
                t = str(d.get("text") or d.get("answer") or d.get("value") or "").strip()
                if not t:
                    continue
                choices.append(t)
                flag = d.get("is_correct")
                if isinstance(flag, bool) and flag:
                    inferred_correct.append(len(choices))  # 1-based index after append
        else:
            if not isinstance(choices_raw, list):
                choices_raw = []
            choices = [str(x).strip() for x in choices_raw if str(x).strip()]

        # ---- correct ----
        correct_texts = item.get("correct_texts") or item.get("correctTexts") or []
        correct: List[int] = []

        # 1) from answers[].is_correct
        if inferred_correct:
            correct = inferred_correct[:]

        # 2) from correct_answer_index (0-based in the new DB)
        if not correct:
            idx0 = (
                item.get("correct_answer_index")
                or item.get("correctAnswerIndex")
                or item.get("correct_index")
                or item.get("correctIndex")
            )
            try:
                if idx0 is not None and str(idx0).strip() != "":
                    i0 = int(idx0)
                    if 0 <= i0 < len(choices):
                        correct = [i0 + 1]
            except Exception:
                pass

        # 3) from correct_answer_indices (0-based list)
        if not correct:
            idxs0 = item.get("correct_answer_indices") or item.get("correctAnswerIndices")
            if isinstance(idxs0, list):
                tmp: List[int] = []
                for v in idxs0:
                    try:
                        i0 = int(v)
                        if 0 <= i0 < len(choices):
                            tmp.append(i0 + 1)
                    except Exception:
                        continue
                if tmp:
                    correct = sorted(set(tmp))

        # 4) legacy formats: correct / answer / right / bool-mask / text match
        if not correct:
            correct_raw = (
                item.get("correct")
                or item.get("correct_answers")
                or item.get("correctAnswers")
                or item.get("right")
                or item.get("right_answers")
                or item.get("answer")
            )

            def _as_int_list(v: Any) -> List[int]:
                if v is None:
                    return []
                if isinstance(v, int):
                    return [v]
                if isinstance(v, str):
                    nums = re.findall(r"\d+", v)
                    return [int(x) for x in nums] if nums else []
                if isinstance(v, list):
                    nums: List[int] = []
                    for x in v:
                        if isinstance(x, bool):
                            continue
                        if isinstance(x, int):
                            nums.append(x)
                        elif isinstance(x, str) and x.strip().isdigit():
                            nums.append(int(x.strip()))
                    return nums
                return []

            if isinstance(correct_raw, list) and correct_raw and all(isinstance(x, bool) for x in correct_raw):
                correct = [i + 1 for i, flag in enumerate(correct_raw) if flag]
            else:
                correct = _as_int_list(correct_raw)

            # if correct заданий текстом/текстами
            if (not correct) and isinstance(correct_raw, (str, list)) and choices:
                cand_texts: List[str] = []
                if isinstance(correct_raw, str):
                    cand_texts = [correct_raw]
                elif isinstance(correct_raw, list):
                    cand_texts = [str(x) for x in correct_raw if x is not None]

                inferred_texts: List[str] = []
                for t in cand_texts:
                    t = str(t).strip()
                    if not t or t.isdigit():
                        continue
                    for i, ch in enumerate(choices):
                        if ch.strip() == t:
                            correct.append(i + 1)
                            inferred_texts.append(t)
                            break

                if inferred_texts and not correct_texts:
                    correct_texts = inferred_texts

        # correct_texts normalization / inference
        if isinstance(correct_texts, str):
            correct_texts = [correct_texts]
        if isinstance(correct_texts, list):
            correct_texts = [str(x).strip() for x in correct_texts if str(x).strip()]
        else:
            correct_texts = []

        if not correct_texts and correct and choices:
            correct_texts = [choices[i - 1] for i in correct if 1 <= i <= len(choices)]

        # ---- ids ----
        raw_id = item.get("id") or item.get("uid") or item.get("qid") or item.get("question_id")
        qid = self._make_int_id(raw_id, fallback=(raw_id, section, topic, item.get("question_number"), qtext))

        return {
            "id": int(qid),
            "section": section,
            "topic": topic,
            "ok": ok,
            "level": level,
            "question": qtext,
            "choices": choices,
            "correct": correct,
            "correct_texts": correct_texts,
        }

    def _law_group_key(self, topic: str) -> str:
        topic = (topic or "").strip()
        # classic: "1. ...", "2. ..."
        if len(topic) >= 2 and topic[0].isdigit() and topic[1] == ".":
            return topic.split(".", 1)[0].strip()  # "1", "2", ...
        if not topic:
            topic = "Законодавство"
        key = "t" + hashlib.sha1(topic.encode("utf-8")).hexdigest()[:10]  # ASCII, short
        self._law_group_titles.setdefault(key, topic)
        return key

    def _make_int_id(self, raw_id: Any, fallback: Any) -> int:
        """Return a deterministic INT32-safe id for a question.

        Postgres column `errors.qid` is INT (int4), so we must keep ids within
        [-2147483648..2147483647]. For non-numeric ids we use a SHA1-based hash
        truncated to 31 bits (positive).
        """
        # if the file contains a numeric id — use it (only if it fits int32)
        try:
            if raw_id is not None and str(raw_id).strip().lstrip("-").isdigit():
                v = int(str(raw_id).strip())
                if -2147483648 <= v <= 2147483647:
                    return v
        except Exception:
            pass

        # otherwise — deterministic 31-bit hash
        s = json.dumps(fallback, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha1(s.encode("utf-8")).digest()
        v = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        return v or 1

# -------------------- Access --------------------

def access_status(user: Dict[str, Any]) -> Tuple[bool, str]:
    # registered?
    if not user or not user.get("phone"):
        return False, "not_registered"

    t_end: Optional[datetime] = user.get("trial_end")
    s_end: Optional[datetime] = user.get("sub_end")
    inf: bool = bool(user.get("sub_infinite"))

    n = now()

    # ✅ ПІДПИСКА має пріоритет над тріалом
    if inf:
        return True, "sub_infinite"
    if s_end and n <= s_end:
        return True, "sub_active"
    if t_end and n <= t_end:
        return True, "trial"

    return False, "expired"



# -------------------- UI helpers --------------------

def kb_inline(
    buttons: List[Tuple[str, str]],
    row: int = 2,
    single_row_prefixes: Optional[Tuple[str, ...]] = ("⬅️", "🔁"),
    single_row_exact: Optional[Tuple[str, ...]] = ("⬅️ Назад", "⬅️ Меню"),
) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()

    main = []
    tail = []

    for text, data in buttons:
        item = (text, data)
        if (single_row_exact and text in single_row_exact) or (
            single_row_prefixes and text.startswith(single_row_prefixes)
        ):
            tail.append(item)      # ці кнопки підуть вниз по 1 в рядок
        else:
            main.append(item)      # решта — звичайне розкладання

    for text, data in main:
        b.button(text=text, callback_data=clamp_callback(data))
    b.adjust(row)

    # додаємо "хвіст" з кнопок, кожна окремим рядком
    for text, data in tail:
        b.button(text=text, callback_data=clamp_callback(data))
        b.adjust(1)

    return b.as_markup()


def fmt_access_line(user: Dict[str, Any]) -> str:
    ok, st = access_status(user)
    if not user.get("phone"):
        return "Статус: ❌ не зареєстровано"
    if ok and st == "trial":
        te = user.get("trial_end")
        return f"Статус: 🟡 тріал до {te.strftime('%d.%m.%Y %H:%M')}"
    if ok and st == "sub_infinite":
        return "Статус: 🟢 підписка (безкінечно)"
    if ok and st == "sub_active":
        se = user.get("sub_end")
        return f"Статус: 🟢 підписка до {se.strftime('%d.%m.%Y %H:%M')}"
    return "Статус: 🔴 доступ завершився"


def clean_law_title(title: str) -> str:
    """Remove boilerplate from law group titles (UI only)."""
    t = (title or "").strip()
    prefixes = [
        "Питання на перевірку знання ",
        "Питання на перевірку знань ",
        "Питання на перевірку знання",
        "Питання на перевірку знань",
    ]
    for p in prefixes:
        if t.startswith(p):
            t = t[len(p):].strip()
    return t


async def render_main(
    bot: Bot,
    store: Storage,
    user_id: int,
    chat_id: int,
    text: str,
    keyboard: Optional[InlineKeyboardMarkup],
    message: Optional[Message] = None,
):
    async def save_mid(mid: int):
        # ✅ set_ui очікує (user_id, chat_id, main_message_id)
        await store.set_ui(user_id, chat_id, mid)

    # 1) Якщо передали Message — редагуємо його
    if message:
        try:
            await message.edit_text(
                text,
                reply_markup=keyboard,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            await save_mid(message.message_id)
            return
        except TelegramBadRequest:
            pass

    # 2) Інакше редагуємо "головне" з ui
    ui = await store.get_ui(user_id) or {}
    mid = ui.get("main_message_id")

    if mid:
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=mid,
                text=text,
                reply_markup=keyboard,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            return
        except TelegramBadRequest:
            pass

    # 3) Якщо нема що редагувати — надсилаємо нове і запам'ятовуємо
    sent = await bot.send_message(
        chat_id,
        text,
        reply_markup=keyboard,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )
    await save_mid(sent.message_id)


# -------------------- Screens --------------------

def screen_need_registration() -> Tuple[str, InlineKeyboardMarkup]:
    text = (
        "Щоб користуватись ботом, потрібна реєстрація.\n\n"
        "Натисни кнопку внизу 👇"
    )
    # Кнопка «Поділитися номером» показується як ReplyKeyboard (request_contact),
    # тому тут залишаємо лише допоміжні кнопки.
    kb = kb_inline([
        ("❓ Допомога", "nav:help"),
    ], row=1)
    return text, kb


def screen_main_menu(user: Dict[str, Any], is_admin: bool) -> Tuple[str, InlineKeyboardMarkup]:
    FILL = "\u2800" * 30

    text = (
        "🏠 <b>Головне меню</b>\n"
        f"{FILL}\n"
        "Оберіть розділ:\n\n"
        "ℹ️ <b>Підказка</b>:\n"
        "• 📚 <b>Навчання</b> — розділ для вивчення матеріалів (законодавство та ОК-модулі).\n"
        "• 📝 <b>Тестування</b> — розділ для перевірки знань за законодавством і ОК-модулями."
    )

    rows = [
        [InlineKeyboardButton(text="📚 Навчання", callback_data="nav:learn")],
        [InlineKeyboardButton(text="📝 Тестування", callback_data="nav:test")],
        [InlineKeyboardButton(text="📊 Статистика", callback_data="nav:stats")],
        [InlineKeyboardButton(text="❓ Допомога", callback_data="nav:help")],
    ]
    if is_admin:
        rows.append([InlineKeyboardButton(text="🛠 Користувачі", callback_data="admin:users:0")])

    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def screen_help(admin_url: str, registered: bool) -> Tuple[str, InlineKeyboardMarkup]:
    text = (
        "❓ <b>Допомога</b>\n\n"
        "Тут ви можете:\n"
        "▪ приєднатися до Telegram-групи\n"
        "▪ звернутися до адміністратора\n"
    )
    if not registered:
        text += "▪ зареєструватися (поділитися номером)\n"

    b = InlineKeyboardBuilder()
    if GROUP_URL:
        b.button(text="🔗 Telegram-група", url=GROUP_URL)
    if admin_url:
        b.button(text="📩 Написати адміну", url=admin_url)

    if registered:
        b.button(text="⬅️ Меню", callback_data="nav:menu")
    else:
        b.button(text="⬅️ Реєстрація", callback_data="nav:reg")

    b.adjust(1)
    return text, b.as_markup()


def screen_no_access(user: Dict[str, Any], admin_url: str) -> Tuple[str, InlineKeyboardMarkup]:
    text = (
        "⛔️ <b>Підписка неактивна</b>\n"
        f"{fmt_access_line(user)}\n\n"
        "Термін дії підписки завершився.\n"
        "Для продовження вам надано доступ лише для звернення до адміністратора."
    )

    b = InlineKeyboardBuilder()
    if admin_url:
        b.button(text="📩 Написати адміну", url=admin_url)
    b.button(text="⬅️ Меню", callback_data="nav:menu")
    b.adjust(1)
    return text, b.as_markup()


def screen_learning_menu(user: Optional[Dict[str, Any]] = None) -> Tuple[str, InlineKeyboardMarkup]:
    FILL = "\u2800" * 30

    text = (
        "📚 <b>Навчання</b>\n"
        f"{FILL}\n"
        "Оберіть напрям:\n\n"
        "ℹ️ <b>Підказка</b>: розділ «Навчання» — це матеріали для підготовки: законодавство та навчальні модулі."
    )

    rows = [
        [InlineKeyboardButton(text="📜 Законодавство", callback_data="learn:law")],
        [InlineKeyboardButton(text="🧩 Операційні компетенції (ОК)", callback_data="learn:ok")],
        [InlineKeyboardButton(text="🧯 Робота над помилками", callback_data="learn:mistakes")],
        [InlineKeyboardButton(text="⬅️ Меню", callback_data="nav:menu")],
    ]
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def screen_law_groups(
    qb: QuestionBank,
    user: Optional[Dict[str, Any]] = None,
    page: int = 0,
    per_page: int = 8
) -> Tuple[str, InlineKeyboardMarkup]:
    user = user or {}

    FILL = "\u2800" * 30
    keys = list(qb.law_groups.keys())

    def key_sort(k: str):
        return (0, int(k)) if k.isdigit() else (1, k)

    keys.sort(key=key_sort)

    total_pages = max(1, (len(keys) + per_page - 1) // per_page)
    try:
        page = int(page)
    except Exception:
        page = 0
    page = max(0, min(page, total_pages - 1))

    start = page * per_page
    shown = keys[start:start + per_page]

    buttons: List[Tuple[str, str]] = []
    for k in shown:
        title = clean_law_title(qb.law_group_title(k))
        buttons.append((f"{k}. {title}" if k.isdigit() else title, f"lawgrp:{k}"))

    if total_pages > 1:
        if page > 0:
            buttons.append(("◀️ Попередні", f"lawpg:{page - 1}"))
        if page < total_pages - 1:
            buttons.append(("▶️ Наступні", f"lawpg:{page + 1}"))

    text = (
        "📜 <b>Законодавство</b>\n"
        f"{FILL}\n"
        "Оберіть розділ:"
    )

    buttons.append(("⬅️ Назад", "nav:learn"))
    kb = kb_inline(buttons, row=1)
    return text, kb


def screen_law_parts(group_key: str, qb: QuestionBank) -> Tuple[str, InlineKeyboardMarkup]:
    qids = qb.law_groups.get(group_key, [])
    total = len(qids)

    header = "📜 <b>Законодавство</b>"

    if total <= 5:
        text = (
            f"{header}\n\n"
            f"Питань: {total}\n"
            f"Почати?\n\n"
            f"ℹ️ <i>Під час тестування: при вірній відповіді система автоматично переходить до наступного питання, "
            f"при невірній — відображається екран з поясненням помилки.</i>"
        )
        kb = kb_inline([
            ("▶️ Почати", f"learn_start:law:{group_key}:1"),
            ("🎲 Рандомні", f"learn_start:lawrand:{group_key}"),
            ("⬅️ Назад", "learn:law"),
        ], row=1)
        return text, kb

    part_size = 50
    parts = []
    p = 1
    for i in range(0, total, part_size):
        a = i + 1
        b = min(i + part_size, total)
        parts.append((p, a, b))
        p += 1

    text = (
        f"{header}\n\n"
        f"Оберіть частину:\n\n"
        f"ℹ️ <i>Під час тестування: при вірній відповіді система автоматично переходить до наступного питання, "
        f"при невірній — відображається екран з поясненням помилки.</i>"
    )

    main_buttons = []
    main_buttons.append(("🎲 Рандомні 50", f"learn_start:lawrand:{group_key}"))
    for p, a, b in parts:
        main_buttons.append((f"{a}–{b}", f"learn_start:law:{group_key}:{p}"))

    kb_main = kb_inline(main_buttons, row=1)
    kb_back = kb_inline([("⬅️ Назад", "learn:law")], row=1)

    kb_main.inline_keyboard.extend(kb_back.inline_keyboard)
    return text, kb_main


OK_TITLES: dict[str, str] = {
    "ОК-1": "Кінологічне забезпечення",
    "ОК-2": "Технічні засоби здійснення митного контролю",
    "ОК-3": "Класифікація товарів",
    "ОК-4": "Митні платежі",
    "ОК-5": "Походження товарів",
    "ОК-6": "Митна вартість",
    "ОК-7": "Нетарифне регулювання",
    "ОК-8": "Контроль за міжнародними передачами стратегічних товарів",
    "ОК-9": "Авторизації",
    "ОК-10": "Митні процедури",
    "ОК-11": "Митний аудит",
    "ОК-12": "Митна статистика",
    "ОК-13": "Транзитні процедури",
    "ОК-14": "Протидія контрабанді та боротьба з порушеннями митних правил",
    "ОК-15": "Управління ризиками",
    "ОК-16": "Захист прав інтелектуальної власності",
    "ОК-17": "Підтримка митниці (організаційне забезпечення)",
}

_ok_code_any_re = re.compile(r"(?i)(?:\[\s*)?(?:ОК|OK)\s*[-–]?\s*(\d+)(?:\s*\])?")

def ok_extract_code(s: str) -> Optional[str]:
    """Extract normalized code like 'ОК-7' from 'ОК-7', '[ОК-7] ...', 'OK7', etc."""
    if not s:
        return None
    m = _ok_code_any_re.search(s.strip())
    if not m:
        return None
    try:
        n = int(m.group(1))
    except Exception:
        return None
    return f"ОК-{n}"

def ok_full_label(s: str) -> str:
    """UI label: '[ОК-7] <name>' if known, otherwise return original."""
    code = ok_extract_code(s) or (s or "").strip()
    if code in OK_TITLES:
        return f"[{code}] {OK_TITLES[code]}"
    return s

def _truncate(s: str, max_len: int) -> str:
    s = " ".join((s or "").split())
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s
    return s[: max(1, max_len - 1)].rstrip() + "…"


def _wrap_for_button(s: str, max_len: int = 34, max_lines: int = 2) -> str:
    """
    Робить "перенос" через \n (до max_lines рядків) і обрізає хвіст '…'.
    """
    s = " ".join((s or "").split())
    if not s:
        return ""
    if max_lines <= 1:
        return _truncate(s, max_len)

    if len(s) <= max_len:
        return s

    words = s.split(" ")
    lines = []
    cur = ""

    for w in words:
        cand = (cur + " " + w).strip() if cur else w
        if len(cand) <= max_len:
            cur = cand
            continue

        if cur:
            lines.append(cur)
        else:
            lines.append(_truncate(w, max_len))

        cur = ""
        if len(lines) >= max_lines - 1:
            break

    rest = " ".join(words[len(" ".join(lines).split()):]).strip() if lines else s
    if not cur:
        cur = rest
    else:
        cur = (cur + " " + rest).strip()

    lines.append(_truncate(cur, max_len))
    return "\n".join(lines[:max_lines])


def _split_ok_label(full: str) -> tuple[str, str]:
    """
    '[ОК-14] Назва...' -> ('[ОК-14]', 'Назва...')
    """
    full = (full or "").strip()
    m = re.match(r"^\s*(\[[^\]]+\])\s*(.*)$", full)
    if not m:
        return "", full
    return m.group(1).strip(), (m.group(2) or "").strip()


def ok_button_text(module: str, *, prefix: str = "", suffix: str = "", max_len: int = 34) -> str:
    """
    Формат для inline-кнопок: робимо перенос після коду (якщо є) + обрізання.
    """
    full = ok_full_label(module)
    code, name = _split_ok_label(full)

    if code:
        line1 = (prefix + code).strip()
        line2_raw = name
        if suffix:
            line2_raw = (f"{name} • {suffix}" if name else suffix).strip()
        line2 = _wrap_for_button(line2_raw, max_len=max_len, max_lines=1)
        return (line1 + "\n" + line2).strip()

    # fallback якщо немає коду в дужках
    raw = (prefix + full).strip()
    if suffix:
        raw = f"{raw} • {suffix}"
    return _wrap_for_button(raw, max_len=max_len, max_lines=2)


def ok_sort_key(name: str):
    code = ok_extract_code(name)
    if code:
        try:
            return (0, int(code.split("-", 1)[1]))
        except Exception:
            pass
    return (1, (name or "").lower())

def screen_ok_menu(
    user: Dict[str, Any],
    user_modules: List[str],
    qb: "QuestionBank"
) -> Tuple[str, InlineKeyboardMarkup]:
    FILL = "\u2800" * 30

    TITLE = "🧩 <b>Оцінювання рівня операційних митних компетенцій</b>\n"

    if not user_modules:
        text = (
            f"{TITLE}"
            f"{FILL}\n"
            "Тут можна проходити питання за вибраними модулями.\n"
            "Спочатку оберіть модулі (можна кілька)."
        )
        kb = kb_inline([
            ("✅ Обрати модулі", "okmods:pick"),
            ("⬅️ Назад", "nav:learn"),
        ], row=1)
        return text, kb

    text = (
        f"{TITLE}"
        f"{FILL}\n"
        "Оберіть модуль, щоб почати:"
    )

    buttons: List[Tuple[str, str]] = []

    pairs = [(i, m) for i, m in enumerate(user_modules) if m in qb.ok_modules]
    pairs.sort(key=lambda p: ok_sort_key(p[1]))
    for i, m in pairs:
        buttons.append((ok_button_text(m), f"okmodi:{i}"))

    buttons += [
        ("🔁 Змінити модулі", "okmods:pick"),
        ("⬅️ Назад", "nav:learn"),
    ]

    kb = kb_inline(buttons, row=1)
    return text, kb



def screen_ok_modules_pick(selected: List[str], all_mods: List[str]) -> Tuple[str, InlineKeyboardMarkup]:
    text = "🧩 <b>Оберіть модулі ОК</b>\n\nПозначте потрібні модулі:"
    b = InlineKeyboardBuilder()

    pairs = list(enumerate(all_mods))
    pairs.sort(key=lambda p: ok_sort_key(p[1]))

    for i, m in pairs:
        mark = "✅" if m in selected else "⬜️"
        b.button(
            text=ok_button_text(m, prefix=f"{mark} "),
            callback_data=clamp_callback(f"okmods:togglei:{i}")
        )

    # нижні кнопки теж в ОДНУ колонку
    b.button(text="Готово", callback_data="okmods:save")
    b.button(text="⬅️ Назад", callback_data="learn:ok")

    b.adjust(1)
    return text, b.as_markup()


def screen_ok_levels(module: str, idx: int, qb: QuestionBank) -> Tuple[str, InlineKeyboardMarkup]:
    levels = sorted(qb.ok_modules.get(module, {}).keys())
    text = (
        f"🧩 <b>{ok_full_label(module)}</b>\n\n"
        f"Оберіть рівень:\n\n"
        f"ℹ️ <i>Під час тестування: при вірній відповіді система автоматично переходить до наступного питання, "
        f"при невірній — відображається екран з поясненням помилки.</i>"
    )

    level_buttons = [(f"Рівень {lvl}", f"learn_start:ok:i:{idx}:{lvl}") for lvl in levels]

    kb_levels = kb_inline(level_buttons, row=1)
    kb_back = kb_inline([("⬅️ Назад", "learn:ok")], row=1)

    kb_levels.inline_keyboard.extend(kb_back.inline_keyboard)
    return text, kb_levels


def screen_test_config(modules: List[str], qb: QuestionBank, temp_levels: Dict[str, int],
                       include_law: bool = True, law_count: int = 50) -> Tuple[str, InlineKeyboardMarkup]:
    lines = [
        "📝 <b>Тестування</b>",
        "",
        "ℹ️ <b>Підказка</b>: розділ «Тестування» — це перевірка знань за законодавством і обраними ОК-модулями.",
        "<i>Під час тестування: при вірній відповіді система автоматично переходить до наступного питання, "
        "при невірній — відображається екран з поясненням помилки.</i>",
        "",
        "Оберіть рівень для кожного модуля ОК (за потреби):",
        "Потім натисніть «Почати тест».",
    ]

    buttons: List[Tuple[str, str]] = []
    law_mark = "✅" if include_law else "❌"
    buttons.append((f"📚 Законодавство • {law_count} питань {law_mark}", "testlaw:toggle"))

    pairs = [(i, m) for i, m in enumerate(modules) if m in qb.ok_modules]
    pairs.sort(key=lambda p: ok_sort_key(p[1]))
    for i, m in pairs:
        levels_map = qb.ok_modules.get(m, {})
        if not levels_map:
            continue
        available = sorted(levels_map.keys())
        lvl = int(temp_levels.get(m, available[0]))
        if lvl not in available:
            lvl = available[0]
        buttons.append((f"🧩 {ok_full_label(m)} • Рівень {lvl}", f"testlvl:modi:{i}"))

    buttons += [("📖 Почати тест", "test:start"), ("⬅️ Меню", "nav:menu")]
    return "\n".join(lines), kb_inline(buttons, row=1)


def screen_test_pick_level(idx: int, module: str, qb: QuestionBank, current: Optional[int]) -> Tuple[str, InlineKeyboardMarkup]:
    levels = sorted(qb.ok_modules.get(module, {}).keys())
    text = f"🧩 <b>{ok_full_label(module)}</b>\n\nОберіть рівень для тесту:"
    buttons: List[Tuple[str, str]] = []
    for lvl in levels:
        mark = "✅ " if current == lvl else ""
        buttons.append((f"{mark}Рівень {lvl}", f"testlvl:seti:{idx}:{lvl}"))
    buttons.append(("⬅️ Назад", "testlvl:back"))
    return text, kb_inline(buttons, row=2)

# -------------------- Session rendering --------------------

def build_question_text(q: Q, header: str, progress: str) -> str:
    question = hescape(q.question or "")
    choices = [hescape(ch or "") for ch in (q.choices or [])]

    lines: List[str] = []
    if header:
        lines.append(header)
    if progress:
        lines.append(progress)

    lines += [
        "❓ <b>Питання</b>",
        question,
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "📝 <b>Варіанти відповіді</b>",
    ]

    for i, ch in enumerate(choices, start=1):
        lines.append(f"  <b>{i})</b> {ch}")

    return "\n".join(lines)


def build_feedback_text(q: Q, header: str, chosen: int) -> str:
    question = hescape(q.question or "")
    choices = [hescape(ch or "") for ch in (q.choices or [])]
    correct_set = set(int(x) for x in (q.correct or []))

    lines: List[str] = []
    if header:
        lines.append(header)

    lines += [
        "❌ <b>Неправильно</b>",
        "",
        "❓ <b>Питання</b>",
        question,
        "",
        "━━━━━━━━━━━━━━━━",
        "📝 <b>Варіанти відповіді</b>",
    ]

    for i, ch in enumerate(choices):
        if (i + 1) in correct_set:
            mark = "✅"
            note = " <i>(правильно)</i>"
        elif i == chosen:
            mark = "❌"
            note = " <i>(ваш вибір)</i>"
        else:
            mark = "▫️"
            note = ""
        lines.append(f"{mark} <b>{i+1})</b> {ch}{note}")

    return "\n".join(lines)



def kb_answers(n: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()

    # Варіанти відповіді — по 4 в рядку
    for i in range(n):
        b.button(text=str(i + 1), callback_data=clamp_callback(f"ans:{i}"))

    # Нижній ряд: Пропустити + Вийти
    b.button(text="⏭ Пропустити", callback_data="skip")
    b.button(text="⏹ Вийти", callback_data="leave:confirm")

    # Розкладка: варіанти по 4 в рядку, потім рядок з двома кнопками
    full_rows = n // 4
    remainder = n % 4
    adjust_list = [4] * full_rows
    if remainder:
        adjust_list.append(remainder)
    adjust_list.append(2)  # для "Пропустити" та "Вийти"

    b.adjust(*adjust_list)

    return b.as_markup()



def kb_feedback() -> InlineKeyboardMarkup:
    return kb_inline([("Зрозуміло / Продовжити", "next")], row=1)


def kb_leave_confirm() -> InlineKeyboardMarkup:
    return kb_inline([
        ("🚪 Вийти в меню", "leave:yes"),
        ("⬅️ Продовжити", "leave:back"),
    ], row=1)



# -------------------- Main app logic --------------------

router = Router()


@router.message(F.text == "/start")
async def cmd_start(message: Message, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    uid = message.from_user.id
    await store.ensure_user(
        uid,
        is_admin=(uid in admin_ids),
        first_name=message.from_user.first_name,
        last_name=message.from_user.last_name
    )
    user = await store.get_user(uid)
    chat_id = message.chat.id

    # try delete /start to avoid chat clutter
    try:
        await message.delete()
    except Exception:
        pass

    ui = await store.get_ui(uid)

    st = ui.get("state", {}) or {}

    # якщо вже є телефон — прибираємо можливе “тимчасове” повідомлення з reply-клавіатурою
    if user.get("phone") and st.get("reg_tmp_msg_id"):
        try:
            await bot.delete_message(chat_id, st["reg_tmp_msg_id"])
        except Exception:
            pass
        st.pop("reg_tmp_msg_id", None)
        await store.set_state(uid, st)

    if not user.get("phone"):
        text, kb = screen_need_registration()
        await render_main(bot, store, uid, chat_id, text, kb)
        await show_contact_request(bot, store, uid, chat_id)
        return

    text, kb = screen_main_menu(user, is_admin=(uid in admin_ids))
    await render_main(bot, store, uid, chat_id, text, kb)

@router.callback_query(F.data == "nav:menu")
async def nav_menu(cb: CallbackQuery, bot: Bot, store: Storage, admin_ids: set[int]):
    uid = cb.from_user.id
    user = await store.get_user(uid)

    # 🔒 якщо не зареєстрований — показуємо тільки реєстрацію
    if not user.get("phone"):
        await store.set_state(uid, {})
        await Storage.show_registration_gate(bot, store, uid, cb.message.chat.id, message=cb.message)
        await cb.answer()
        return

    text, kb = screen_main_menu(user, is_admin=(uid in admin_ids))
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await store.set_state(uid, {})
    await cb.answer()


@router.callback_query(F.data == "nav:help")
async def nav_help(cb: CallbackQuery, bot: Bot, store: Storage, admin_ids: set[int]):
    uid = cb.from_user.id
    user = await store.get_user(uid)

    admin_url = get_admin_contact_url(admin_ids)
    text, kb = screen_help(admin_url, registered=bool(user.get("phone")))

    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()

@router.callback_query(F.data == "nav:reg")
async def nav_reg(cb: CallbackQuery, bot: Bot, store: Storage):
    uid = cb.from_user.id
    chat_id = cb.message.chat.id

    text, kb = screen_need_registration()
    await render_main(bot, store, uid, chat_id, text, kb, message=cb.message)
    await show_contact_request(bot, store, uid, chat_id)

    await cb.answer()


@router.callback_query(F.data == "nav:learn")
async def nav_learn(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    uid = cb.from_user.id
    user = await store.get_user(uid)
    ok_access, _ = access_status(user)
    if not ok_access:
        admin_url = get_admin_contact_url(admin_ids)
        text, kb = screen_no_access(user, admin_url)
        await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
        await cb.answer()
        return

    text, kb = screen_learning_menu(user)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data == "learn:law")
async def learn_law(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    user = await store.get_user(uid)

    text, kb = screen_law_groups(qb, user)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("lawpg:"))
async def law_page(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    user = await store.get_user(uid)

    try:
        page = int(cb.data.split(":", 1)[1])
    except Exception:
        page = 0

    text, kb = screen_law_groups(qb, user, page=page)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("lawgrp:"))
async def law_group(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    group_key = cb.data.split(":", 1)[1]
    text, kb = screen_law_parts(group_key, qb)
    await render_main(bot, store, cb.from_user.id, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data == "learn:ok")
async def learn_ok(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    user = await store.get_user(uid)
    modules = user.get("ok_modules", [])
    text, kb = screen_ok_menu(user, modules, qb)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data == "okmods:pick")
async def okmods_pick(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    user = await store.get_user(uid)
    ui = await store.get_ui(uid)
    state = ui.get("state", {}) or {}

    all_mods = sorted(list(qb.ok_modules.keys()))
    selected = state.get("okmods_temp")
    if selected is None:
        selected = list(user.get("ok_modules", []))

    state["okmods_temp"] = list(selected)
    state["okmods_all"] = all_mods
    await store.set_state(uid, state)

    text, kb = screen_ok_modules_pick(list(selected), all_mods)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()

@router.callback_query(F.data.startswith("okmods:togglei:"))
async def okmods_toggle(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    try:
        idx = int(cb.data.split(":", 2)[2])
    except Exception:
        await cb.answer("Помилка")
        return

    ui = await store.get_ui(uid)
    state = ui.get("state", {}) or {}
    all_mods = state.get("okmods_all") or sorted(list(qb.ok_modules.keys()))
    if idx < 0 or idx >= len(all_mods):
        await cb.answer("Помилка")
        return
    mod = all_mods[idx]

    selected = list(state.get("okmods_temp", []))
    if mod in selected:
        selected.remove(mod)
    else:
        selected.append(mod)
    state["okmods_temp"] = selected
    state["okmods_all"] = all_mods
    await store.set_state(uid, state)

    text, kb = screen_ok_modules_pick(selected, all_mods)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()

@router.callback_query(F.data == "okmods:save")
async def okmods_save(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    ui = await store.get_ui(uid)
    state = ui.get("state", {})
    selected = list(state.get("okmods_temp", []))
    # keep only existing modules
    selected = [m for m in selected if m in qb.ok_modules]
    await store.set_ok_modules(uid, selected)
    state.pop("okmods_temp", None)
    state.pop("okmods_all", None)
    await store.set_state(uid, state)

    user = await store.get_user(uid)
    text, kb = screen_ok_menu(user, user.get("ok_modules", []), qb)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer("Збережено")


@router.callback_query(F.data.startswith("okmodi:"))
async def okmod_levels(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    try:
        idx = int(cb.data.split(":", 1)[1])
    except Exception:
        await cb.answer("Помилка")
        return

    user = await store.get_user(uid)
    modules = list(user.get("ok_modules", []))
    if idx < 0 or idx >= len(modules):
        await cb.answer("Помилка")
        return
    module = modules[idx]

    text, kb = screen_ok_levels(module, idx, qb)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()

@router.callback_query(F.data == "testlaw:toggle")
async def testlaw_toggle(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}

    st["test_include_law"] = not bool(st.get("test_include_law", True))
    await store.set_state(uid, st)

    mod_list = st.get("test_mod_list", []) or []
    temp_levels = st.get("test_levels_temp", {}) or {}
    include_law = bool(st.get("test_include_law", True))

    text, kb = screen_test_config(mod_list, qb, temp_levels, include_law=include_law)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


async def show_contact_request(bot: Bot, store: Storage, uid: int, chat_id: int):
    """Shows ReplyKeyboard with request_contact button (temporary message)."""
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}

    # delete previous temp message (if any) to avoid duplicates
    tmp_id = st.get("reg_tmp_msg_id")
    if tmp_id:
        try:
            await bot.delete_message(chat_id, tmp_id)
        except Exception:
            pass

    kb = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📱 Поділитися номером", request_contact=True)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )
    tmp = await bot.send_message(chat_id, "👇 Поділись номером (кнопка внизу)", reply_markup=kb)

    # remember temp message id so we can delete it after registration
    st["reg_tmp_msg_id"] = tmp.message_id
    st["reg_awaiting"] = True
    await store.set_state(uid, st)

# -------- Registration (contact) --------

@router.callback_query(F.data == "reg:request")
async def reg_request(cb: CallbackQuery, bot: Bot, store: Storage):
    await cb.answer()  # <-- одразу, щоб не було таймауту

    uid = cb.from_user.id
    chat_id = cb.message.chat.id

    # main message stays the same, but we must show a ReplyKeyboard (contact) -> temporary message
    await render_main(
        bot, store, uid, chat_id,
        "📱 <b>Реєстрація</b>\n\nНатисни кнопку внизу, щоб поділитися номером.",
        kb_inline([("⬅️ Назад", "nav:menu")], row=1),
        message=cb.message
    )

    await show_contact_request(bot, store, uid, chat_id)

@router.message(F.contact)
async def on_contact(message: Message, bot: Bot, store: Storage, admin_ids: set[int]):
    uid = message.from_user.id
    chat_id = message.chat.id

    await store.ensure_user(
        uid,
        is_admin=(uid in admin_ids),
        first_name=message.from_user.first_name,
        last_name=message.from_user.last_name,
    )

    c = message.contact
    if not c or not c.phone_number:
        try:
            await message.delete()
        except Exception:
            pass

        await bot.send_message(chat_id, "Не бачу номер. Спробуй ще раз.")
        await show_contact_request(bot, store, uid, chat_id)
        return

    # якщо Telegram передав contact.user_id і це не користувач — відхиляємо
    if c.user_id is not None and c.user_id != uid:
        try:
            await message.delete()
        except Exception:
            pass

        await bot.send_message(chat_id, "Поділись, будь ласка, СВОЇМ номером через кнопку знизу.")
        await show_contact_request(bot, store, uid, chat_id)
        return

    phone = c.phone_number
    first_name = (c.first_name or message.from_user.first_name or "").strip() or None
    last_name = (c.last_name or message.from_user.last_name or "").strip() or None

    await store.set_phone_and_trial(uid, phone, first_name=first_name, last_name=last_name)

    # прибираємо повідомлення користувача з контактом (там видно номер)
    try:
        await message.delete()
    except Exception:
        pass

    # прибираємо тимчасове повідомлення з кнопкою "Поділитися номером"
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    tmp_id = st.pop("reg_tmp_msg_id", None)
    st.pop("reg_awaiting", None)
    old_main_id = ui.get("main_message_id")

    if tmp_id:
        try:
            await bot.delete_message(chat_id, tmp_id)
        except Exception:
            pass

    await store.set_state(uid, st)

    # прибираємо ReplyKeyboard (кнопку знизу)
    cleanup = await bot.send_message(chat_id, "✅", reply_markup=ReplyKeyboardRemove())
    try:
        await bot.delete_message(chat_id, cleanup.message_id)
    except Exception:
        pass

    # (опційно) видаляємо старе "головне" повідомлення (екран реєстрації), щоб не лишалось вище
    if old_main_id:
        try:
            await bot.delete_message(chat_id, old_main_id)
        except Exception:
            pass

    # показуємо меню НОВИМ повідомленням внизу (щоб його точно було видно)
    user = await store.get_user(uid)
    text, kb = screen_main_menu(user, is_admin=(uid in admin_ids))

    sent = await bot.send_message(
        chat_id,
        text,
        reply_markup=kb,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )
    await store.set_ui(uid, chat_id, sent.message_id)



# -------- Learning / Testing sessions --------

async def start_learning_session(
    bot: Bot,
    store: Storage,
    qb: QuestionBank,
    uid: int,
    chat_id: int,
    message: Message,
    qids: List[int],
    header: str,
    save_meta: Dict[str, Any],
):
    if not qids:
        await render_main(
            bot, store, uid, chat_id,
            "Немає доступних питань у цьому наборі (можливо, вони без варіантів відповіді).",
            kb_inline([("⬅️ Назад", "nav:learn")], row=1),
            message=message
        )
        return

    state = {
        "mode": "learn",
        "header": header,
        "pending": qids[:],
        "skipped": [],
        "phase": "pending",
        "feedback": None,
        "current_qid": None,
        "correct_count": 0,
        "total": len(qids),
        "started_at": dt_to_iso(now()),
        "answers": {},
        "meta": save_meta,
    }
    await store.set_state(uid, state)
    await show_next_in_session(bot, store, qb, uid, chat_id, message)


async def show_next_in_session(bot: Bot, store: Storage, qb: QuestionBank, uid: int, chat_id: int, message: Message):
    ui = await store.get_ui(uid)
    st = ui.get("state", {})
    mode = st.get("mode")

    if mode not in ("learn", "test", "mistakes"):
        return

    # feedback stage?
    if st.get("feedback"):
        fb = st["feedback"]
        qid_fb = fb.get("qid")
        chosen = fb.get("chosen")

        q = qb.by_id.get(int(qid_fb)) if qid_fb is not None else None

        if q is not None and chosen is not None:
            text = build_feedback_text(q, st.get("header", ""), int(chosen))
        else:
            text = f"{st.get('header', '')}\n❌ <b>Неправильно</b>"

        await render_main(bot, store, uid, chat_id, text, kb_feedback(), message=message)
        return

    pending = list(st.get("pending", []))
    skipped = list(st.get("skipped", []))
    phase = st.get("phase", "pending")

    if not pending:
        if mode == "learn" and skipped:
            st["pending"] = skipped
            st["skipped"] = []
            st["phase"] = "skipped"
            await store.set_state(uid, st)
            await show_next_in_session(bot, store, qb, uid, chat_id, message)
            return

        # finish
        if mode == "test":
            correct = int(st.get("correct_count", 0))
            total = int(st.get("total", 0))
            percent = (correct / total * 100.0) if total else 0.0
            passed = percent >= 60.0
            started_at = iso_to_dt(st.get("started_at")) or now()
            finished_at = now()
            await store.save_test(uid, started_at, finished_at, total, correct)

            # Обчислення відсотка для відображення
            percent_display = f"{percent:.1f}%"

            text = (
                "📝 <b>Тестування завершено</b>\n\n"
                "📊 <b>Результати:</b>\n\n"
                f"✅ <b>{correct}</b> з <b>{total}</b> питань\n"
                f"📈 <b>{percent_display}</b> правильних відповідей\n"
                f"🎯 Прохідний поріг: <b>60%</b>\n\n"
                f"<b>{'🎉 Вітаємо! Тест складено!' if passed else '❌ Тест не складено. Потрібно ще попрацювати.'}</b>"
            )
            await store.set_state(uid, {})
            await render_main(bot, store, uid, chat_id, text, kb_inline([("⬅️ Головне меню", "nav:menu")], row=1),
                              message=message)
            return

        if mode == "mistakes":
            answers = st.get("answers", {})
            total = len(answers)
            correct_ids = [qid for qid, ok in answers.items() if ok]
            wrong_ids = [qid for qid, ok in answers.items() if not ok]
            # cleanup mistakes: remove correct, keep wrong
            for qid in correct_ids:
                await store.remove_mistake(uid, int(qid))

            percent = (len(correct_ids) / total * 100.0) if total else 0.0
            text = (
                "🧯 <b>Робота над помилками — завершено</b>\n\n"
                f"Правильних: <b>{len(correct_ids)}</b> із <b>{total}</b>\n"
                f"Результат: <b>{percent:.1f}%</b>\n\n"
                f"Залишилось у помилках: <b>{len(wrong_ids)}</b>"
            )
            await store.set_state(uid, {})
            await render_main(bot, store, uid, chat_id, text, kb_inline([("⬅️ Навчання", "nav:learn")], row=1), message=message)
            return

        # learn finish
        correct = int(st.get("correct_count", 0))
        total = int(st.get("total", 0)) or 0
        percent = (correct / total * 100.0) if total else 0.0

        text = (
            "📚 <b>Навчання завершено</b>\n\n"
            "📊 <b>Результат:</b>\n"
            f"✅ <b>{correct}</b> з <b>{total}</b> правильних\n"
            f"📈 <b>{percent:.1f}%</b>"
        )

        # NEW: кнопка "назад до рівнів/ОК" + стара кнопка "Навчання"
        meta = st.get("meta") or {}
        kind = meta.get("kind")

        buttons = []

        # якщо це ОК-рівень — даємо "До рівнів" (повертає на вибір рівня цього модуля)
        if kind == "ok":
            module = meta.get("module")
            idx = None
            if module:
                user = await store.get_user(uid)
                mods = list(user.get("ok_modules", []))
                try:
                    idx = mods.index(module)
                except ValueError:
                    idx = None

            if idx is not None:
                buttons.append(("⬅️ До рівнів", f"okmodi:{idx}"))
            else:
                buttons.append(("⬅️ ОК", "learn:ok"))

        # якщо це законодавство — можна теж дати повернення до частин (не обовʼязково)
        elif kind in ("law", "lawrand"):
            group_key = meta.get("group")
            if group_key:
                buttons.append(("⬅️ До частин", f"lawgrp:{group_key}"))

        # завжди залишаємо вихід у меню навчання
        buttons.append(("⬅️ Навчання", "nav:learn"))

        await store.set_state(uid, {})
        await render_main(
            bot, store, uid, chat_id,
            text,
            kb_inline(buttons, row=1),
            message=message
        )
        return


# -------- Pre-test (question picker) --------

def _qpick_kb(total: int, selected: int, back_cb: Optional[str]) -> InlineKeyboardMarkup:
    """Inline keyboard with 1..total buttons + 'Start test' under them.
    selected: 0-based index of highlighted question.
    """
    total = max(0, int(total))
    selected = max(0, min(int(selected), max(0, total - 1))) if total else 0

    b = InlineKeyboardBuilder()

    # 50 buttons (or less)
    for i in range(total):
        label = str(i + 1)
        if total and i == selected:
            label = "✅" + label
        b.button(text=label, callback_data=clamp_callback(f"qpick:go:{i+1}"))

    # 5 buttons per row (10 rows for 50)
    if total:
        rows = (total + 4) // 5
        b.adjust(*([5] * (rows - 1) + [total - 5 * (rows - 1)]))
    else:
        b.adjust(1)

    # actions under grid
    b.row(InlineKeyboardButton(text="▶️ Розпочати тестування", callback_data="qpick:start"))
    if back_cb:
        b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data=back_cb))
    b.row(InlineKeyboardButton(text="⬅️ Навчання", callback_data="nav:learn"))

    return b.as_markup()


def screen_qpick_grid(header: str, total: int, selected: int = 0, back_cb: Optional[str] = None) -> Tuple[str, InlineKeyboardMarkup]:
    total = max(0, int(total))
    selected = max(0, min(int(selected), max(0, total - 1))) if total else 0

    text = (
        "📝 <b>Підготовка до тесту</b>\n\n"
        f"{header}\n"
        f"Питань у цьому тесті: <b>{total}</b>\n\n"
        "Натисни номер питання, щоб подивитись його, або натисни "
        "<b>«Розпочати тестування»</b>."
    )

    return text, _qpick_kb(total, selected, back_cb)


def screen_qpick_preview(
    header: str,
    q: Q,
    idx_1based: int,
    total: int,
    back_cb: Optional[str] = None,
) -> Tuple[str, InlineKeyboardMarkup]:
    idx_1based = max(1, int(idx_1based))
    total = max(1, int(total))

    # preview only (no answering here)
    opts = "\n".join([f"{i+1}) {hescape(ch)}" for i, ch in enumerate(q.choices or [])])

    text = (
        "📝 <b>Підготовка до тесту</b>\n\n"
        f"{header}\n\n"
        f"<b>Питання {idx_1based}/{total}</b>\n"
        f"{hescape(q.question)}\n\n"
        "📝 <b>Варіанти</b>\n"
        f"{opts}"
    )

    b = InlineKeyboardBuilder()
    b.button(text="⬅️ До списку питань", callback_data="qpick:show")
    b.button(text="▶️ Розпочати тестування", callback_data="qpick:start")
    if back_cb:
        b.button(text="⬅️ Назад", callback_data=back_cb)
    b.button(text="⬅️ Навчання", callback_data="nav:learn")
    b.adjust(2, 2)  # 2 rows

    return text, b.as_markup()


async def start_pretest(
    bot: Bot,
    store: Storage,
    qb: QuestionBank,
    uid: int,
    chat_id: int,
    message: Message,
    qids: List[int],
    header: str,
    meta: Dict[str, Any],
    back_cb: Optional[str] = None,
):
    # limit to 50 questions (as requested)
    qids = list(qids or [])[:50]

    if not qids:
        await render_main(
            bot, store, uid, chat_id,
            "Немає доступних питань у цьому наборі (можливо, вони без варіантів відповіді).",
            kb_inline([("⬅️ Назад", back_cb or "nav:learn")], row=1),
            message=message,
        )
        return

    st = {
        "mode": "pretest",
        "header": header,
        "qids": qids,
        "selected": 0,  # 0-based
        "meta": meta or {},
        "back_cb": back_cb,
    }
    await store.set_state(uid, st)

    text, kb = screen_qpick_grid(header, len(qids), selected=0, back_cb=back_cb)
    await render_main(bot, store, uid, chat_id, text, kb, message=message)


async def start_test_from_pretest(
    bot: Bot,
    store: Storage,
    qb: QuestionBank,
    uid: int,
    chat_id: int,
    message: Message,
    pre: Dict[str, Any],
):
    qids = list(pre.get("qids", []) or [])
    if not qids:
        return

    selected = int(pre.get("selected", 0) or 0)
    selected = max(0, min(selected, len(qids) - 1))

    # start from selected question
    ordered = qids[selected:] + qids[:selected]

    st = {
        "mode": "test",
        "header": pre.get("header", "📝 <b>Тестування</b>"),
        "pending": ordered,
        "skipped": [],
        "phase": "pending",
        "feedback": None,
        "current_qid": None,
        "correct_count": 0,
        "total": len(ordered),
        "started_at": dt_to_iso(now()),
        "answers": {},
        "meta": pre.get("meta", {}) or {},
    }
    await store.set_state(uid, st)
    await show_next_in_session(bot, store, qb, uid, chat_id, message)


@router.callback_query(F.data == "qpick:show")
async def qpick_show(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    if not await guard_access_in_session(cb, bot, store, admin_ids):
        return
    uid = cb.from_user.id
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    if st.get("mode") != "pretest":
        await cb.answer("Немає активної підготовки", show_alert=True)
        return
    header = st.get("header", "")
    back_cb = st.get("back_cb")
    selected = int(st.get("selected", 0) or 0)
    total = len(st.get("qids", []) or [])
    text, kb = screen_qpick_grid(header, total, selected=selected, back_cb=back_cb)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("qpick:go:"))
async def qpick_go(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    if not await guard_access_in_session(cb, bot, store, admin_ids):
        return
    uid = cb.from_user.id
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    if st.get("mode") != "pretest":
        await cb.answer("Немає активної підготовки", show_alert=True)
        return

    try:
        idx_1based = int(cb.data.split(":")[2])
    except Exception:
        await cb.answer("Помилка", show_alert=True)
        return

    qids = list(st.get("qids", []) or [])
    total = len(qids)
    if total <= 0:
        await cb.answer("Немає питань", show_alert=True)
        return

    idx0 = max(0, min(idx_1based - 1, total - 1))
    st["selected"] = idx0
    await store.set_state(uid, st)

    qid = qids[idx0]
    q = qb.by_id.get(int(qid))
    if not q:
        await cb.answer("Питання не знайдено", show_alert=True)
        return

    header = st.get("header", "")
    back_cb = st.get("back_cb")
    text, kb = screen_qpick_preview(header, q, idx_1based=idx0 + 1, total=total, back_cb=back_cb)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data == "qpick:start")
async def qpick_start(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    if not await guard_access_in_session(cb, bot, store, admin_ids):
        return
    uid = cb.from_user.id
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    if st.get("mode") != "pretest":
        await cb.answer("Немає активної підготовки", show_alert=True)
        return

    await cb.answer()
    await start_test_from_pretest(bot, store, qb, uid, cb.message.chat.id, cb.message, st)



@router.callback_query(F.data.startswith("learn_start:"))
async def learn_start(
    cb: CallbackQuery,
    bot: Bot,
    store: Storage,
    qb: QuestionBank,
    admin_ids: set[int],
):
    # 1) швидко відповідаємо на callback (щоб Telegram не таймаутився)
    await cb.answer()

    uid = cb.from_user.id
    user = await store.get_user(uid)

    ok_access, _ = access_status(user)
    if not ok_access:
        # доступ закінчився — зупиняємо будь-яку сесію і показуємо екран без доступу
        await store.set_state(uid, {})
        admin_url = get_admin_contact_url(admin_ids)
        text, kb = screen_no_access(user, admin_url)
        await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
        await cb.answer("Доступ завершився. Потрібна підписка.", show_alert=True)
        return

    parts = (cb.data or "").split(":")
    if len(parts) < 2:
        await cb.answer("Помилка", show_alert=True)
        return

    kind = parts[1]

    # -------- Законодавство --------
    if kind == "law":
        # формат: learn_start:law:<group_key>:<part>
        group_key = parts[2] if len(parts) >= 3 else ""
        try:
            part = int(parts[3]) if len(parts) >= 4 else 1
        except Exception:
            part = 1

        all_qids = qb.law_groups.get(group_key, []) or []
        start = max(0, (part - 1) * 50)
        qids = all_qids[start:start + 50]

        title = qb.law_group_title(group_key)
        header = f"📜 <b>Законодавство</b> • {hescape(title)}"
        if len(all_qids) > 50:
            header += f" • частина {part}"

        await start_pretest(
            bot, store, qb,
            uid, cb.message.chat.id, cb.message,
            qids=qids,
            header=header,
            meta={"kind": "law", "group": group_key, "part": part},
            back_cb=f"lawgrp:{group_key}" if group_key else "nav:learn",
        )
        return

    if kind == "lawrand":
        # формат: learn_start:lawrand:<group_key>
        group_key = parts[2] if len(parts) >= 3 else ""
        all_qids = qb.law_groups.get(group_key, []) or []
        qids = qb.pick_random(all_qids, min(50, len(all_qids)))

        title = qb.law_group_title(group_key)
        header = f"📜 <b>Законодавство</b> • {hescape(title)} • 🎲"

        await start_pretest(
            bot, store, qb,
            uid, cb.message.chat.id, cb.message,
            qids=qids,
            header=header,
            meta={"kind": "lawrand", "group": group_key},
            back_cb=f"lawgrp:{group_key}" if group_key else "nav:learn",
        )
        return

    # -------- ОК --------
    if kind == "ok":
        module: Optional[str] = None
        level = 1
        back_cb: str = "nav:learn"

        # НОВИЙ формат: learn_start:ok:i:<idx>:<level>
        if len(parts) >= 5 and parts[2] == "i":
            try:
                idx = int(parts[3])
                level = int(parts[4])
            except Exception:
                await cb.answer("Помилка", show_alert=True)
                return

            modules = user.get("ok_modules", []) or []
            if idx < 0 or idx >= len(modules):
                await cb.answer("Модуль не знайдено", show_alert=True)
                return
            module = modules[idx]
            back_cb = f"okmodi:{idx}"

        # СТАРИЙ формат: learn_start:ok:<module>:<level?>
        else:
            module = parts[2] if len(parts) >= 3 else None
            if len(parts) >= 4:
                try:
                    level = int(parts[3])
                except Exception:
                    level = 1

            # back to levels screen if possible
            modules = user.get("ok_modules", []) or []
            if module in modules:
                back_cb = f"okmodi:{modules.index(module)}"

        if not module:
            await cb.answer("Модуль не знайдено", show_alert=True)
            return

        qids = (qb.ok_modules.get(module, {}) or {}).get(int(level), []) or []
        qids = list(qids)[:50]  # рівно 50 (або менше, якщо немає)

        # запамʼятовуємо останній рівень (як було)
        await store.set_ok_last_level(uid, module, int(level))

        header = f"🧠 <b>ОК</b> • {hescape(module)} • Рівень {int(level)}"

        await start_pretest(
            bot, store, qb,
            uid, cb.message.chat.id, cb.message,
            qids=qids,
            header=header,
            meta={"kind": "ok", "module": module, "level": int(level)},
            back_cb=back_cb,
        )
        return

    await cb.answer("Невідомий режим", show_alert=True)


async def guard_access_in_session(
    cb: CallbackQuery,
    bot: Bot,
    store: Storage,
    admin_ids: set[int],
) -> Optional[Dict[str, Any]]:
    uid = cb.from_user.id
    user = await store.get_user(uid)
    ok_access, _ = access_status(user)

    if ok_access:
        return user

    # доступ закінчився — зупиняємо сесію і показуємо екран без доступу
    await store.set_state(uid, {})
    admin_url = get_admin_contact_url(admin_ids)
    text, kb = screen_no_access(user, admin_url)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)

    await cb.answer("Доступ завершився. Потрібна підписка.", show_alert=True)
    return None

@router.callback_query(F.data.startswith("ans:"))
async def on_answer(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    if not await guard_access_in_session(cb, bot, store, admin_ids):
        return

    uid = cb.from_user.id
    choice = int(cb.data.split(":")[1])

    ui = await store.get_ui(uid)
    st = ui.get("state", {})
    mode = st.get("mode")
    qid = st.get("current_qid")
    if mode not in ("learn", "test", "mistakes") or not qid:
        await cb.answer("Сесія неактивна")
        return

    q = qb.by_id.get(int(qid))
    if not q or not q.is_valid_mcq:
        await cb.answer("Питання недоступне")
        return

    is_correct = (choice + 1) in set(int(x) for x in q.correct)

    pending = list(st.get("pending", []))
    if pending and int(pending[0]) == int(qid):
        pending = pending[1:]
    st["pending"] = pending

    if mode == "learn":
        if is_correct:
            st["correct_count"] = int(st.get("correct_count", 0)) + 1
            st["feedback"] = None
        else:
            wc, im = await store.bump_wrong(uid, int(qid))
            st["feedback"] = {"qid": int(qid), "chosen": int(choice)}
        await store.set_state(uid, st)
        await cb.answer("✅" if is_correct else "❌")
        await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message)
        return

    if mode == "test":
        st.setdefault("total", st.get("total", 0))
        if is_correct:
            st["correct_count"] = int(st.get("correct_count", 0)) + 1
        await store.set_state(uid, st)
        await cb.answer()
        await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message)
        return

    if mode == "mistakes":
        answers = st.get("answers", {})
        answers[str(qid)] = bool(is_correct)
        st["answers"] = answers
        await store.set_state(uid, st)
        await cb.answer()
        await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message)
        return


@router.callback_query(F.data == "skip")
async def on_skip(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    if not await guard_access_in_session(cb, bot, store, admin_ids):
        return

    uid = cb.from_user.id
    ui = await store.get_ui(uid)
    st = ui.get("state", {})
    if st.get("mode") != "learn":
        await cb.answer("Пропуск доступний лише в навчанні")
        return
    qid = st.get("current_qid")
    if not qid:
        await cb.answer()
        return

    pending = list(st.get("pending", []))
    skipped = list(st.get("skipped", []))
    # current was already popped on render; but to be safe:
    if pending and int(pending[0]) == int(qid):
        pending = pending[1:]
    skipped.append(int(qid))

    st["pending"] = pending
    st["skipped"] = skipped
    st["feedback"] = None
    await store.set_state(uid, st)
    await cb.answer("Пропущено")
    await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message)


@router.callback_query(F.data == "next")
async def on_feedback_next(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    if not await guard_access_in_session(cb, bot, store, admin_ids):
        return

    uid = cb.from_user.id
    ui = await store.get_ui(uid)
    st = ui.get("state", {})
    if st.get("mode") != "learn":
        await cb.answer()
        return
    st["feedback"] = None
    await store.set_state(uid, st)
    await cb.answer()
    await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message)

@router.callback_query(F.data == "leave:confirm")
async def leave_confirm(cb: CallbackQuery, bot: Bot, store: Storage):
    await render_main(
        bot, store, cb.from_user.id, cb.message.chat.id,
        "Вийти з поточної сесії? Прогрес буде втрачено.",
        kb_leave_confirm(),
        message=cb.message
    )
    await cb.answer()


@router.callback_query(F.data == "leave:back")
async def leave_back(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    # just continue session
    await cb.answer()
    await show_next_in_session(bot, store, qb, cb.from_user.id, cb.message.chat.id, cb.message)


@router.callback_query(F.data == "leave:yes")
async def leave_yes(cb: CallbackQuery, bot: Bot, store: Storage, admin_ids: set[int]):
    uid = cb.from_user.id
    await store.set_state(uid, {})
    user = await store.get_user(uid)
    text, kb = screen_main_menu(user, is_admin=(uid in admin_ids))
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


# -------- Mistakes --------

@router.callback_query(F.data == "learn:mistakes")
async def learn_mistakes(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    uid = cb.from_user.id
    user = await store.get_user(uid)
    ok_access, _ = access_status(user)
    if not ok_access:
        admin_url = get_admin_contact_url(admin_ids)
        text, kb = screen_no_access(user, admin_url)
        await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
        await cb.answer()
        return

    qids = await store.list_mistakes(uid)
    if not qids:
        await render_main(
            bot, store, uid, cb.message.chat.id,
            "🧯 <b>Робота над помилками</b>\n\nПоки що немає питань (зʼявляються після 5 неправильних відповідей у навчанні).",
            kb_inline([("⬅️ Назад", "nav:learn")], row=1),
            message=cb.message
        )
        await cb.answer()
        return

    st = {
        "mode": "mistakes",
        "header": "🧯 <b>Робота над помилками</b>",
        "pending": qids[:],
        "skipped": [],
        "phase": "pending",
        "feedback": None,
        "current_qid": None,
        "answers": {},
    }
    await store.set_state(uid, st)
    await cb.answer()
    await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message)


# -------- Testing --------

@router.callback_query(F.data == "nav:test")
async def nav_test(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    uid = cb.from_user.id
    user = await store.get_user(uid)
    ok_access, _ = access_status(user)
    if not ok_access:
        admin_url = get_admin_contact_url(admin_ids)
        text, kb = screen_no_access(user, admin_url)
        await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
        await cb.answer()
        return

    modules = user.get("ok_modules", [])
    last_levels = user.get("ok_last_levels", {}) or {}

    temp_levels: Dict[str, int] = {}
    for m in modules:
        levels_map = qb.ok_modules.get(m, {})
        if not levels_map:
            continue
        available = sorted(levels_map.keys())
        lvl = int(last_levels.get(m, available[0]))
        if lvl not in available:
            lvl = available[0]
        temp_levels[m] = lvl

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    st["test_mod_list"] = list(modules)          # щоб індекси в callback працювали
    st["test_levels_temp"] = dict(temp_levels)
    st["test_include_law"] = True# тимчасовий вибір рівнів
    await store.set_state(uid, st)

    text, kb = screen_test_config(modules, qb, temp_levels)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("testlvl:modi:"))
async def testlvl_pick_module(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    idx = int(cb.data.split(":")[2])

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    mod_list = st.get("test_mod_list", []) or []
    if idx < 0 or idx >= len(mod_list):
        await cb.answer("Модуль не знайдено")
        return

    module = mod_list[idx]
    temp_levels = st.get("test_levels_temp", {}) or {}
    current = temp_levels.get(module)

    text, kb = screen_test_pick_level(idx, module, qb, current)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("testlvl:seti:"))
async def testlvl_set_level(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    parts = cb.data.split(":")
    idx = int(parts[2])
    lvl = int(parts[3])

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    mod_list = st.get("test_mod_list", []) or []
    if idx < 0 or idx >= len(mod_list):
        await cb.answer("Модуль не знайдено")
        return

    module = mod_list[idx]
    temp_levels = dict(st.get("test_levels_temp", {}) or {})
    temp_levels[module] = lvl
    st["test_levels_temp"] = temp_levels
    await store.set_state(uid, st)

    # повертаємось на екран конфігурації
    text, kb = screen_test_config(mod_list, qb, temp_levels)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer("Збережено")


@router.callback_query(F.data == "testlvl:back")
async def testlvl_back(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    mod_list = st.get("test_mod_list", []) or []
    temp_levels = st.get("test_levels_temp", {}) or {}

    text, kb = screen_test_config(mod_list, qb, temp_levels)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()

@router.callback_query(F.data == "test:start")
async def test_start(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    user = await guard_access_in_session(cb, bot, store, admin_ids)
    if not user:
        return

    uid = cb.from_user.id

    ui = await store.get_ui(uid)
    pre = ui.get("state", {}) or {}

    include_law = bool(pre.get("test_include_law", True))
    law_qids = qb.pick_random(qb.law, 50) if include_law else []

    modules = user.get("ok_modules", [])
    last_levels = user.get("ok_last_levels", {}) or {}
    picked_levels = pre.get("test_levels_temp", {}) or {}

    ok_qids: List[int] = []
    for m in modules:
        levels_map = qb.ok_modules.get(m, {})
        if not levels_map:
            continue
        available = sorted(levels_map.keys())

        lvl = int(picked_levels.get(m, last_levels.get(m, available[0])))
        if lvl not in available:
            lvl = available[0]

        qids = levels_map.get(lvl, [])
        ok_qids.extend(qb.pick_random(qids, 20))

    all_qids = law_qids + ok_qids

    if not all_qids:
        await cb.answer("Оберіть хоча б один блок для тесту", show_alert=True)
        return

    random.shuffle(all_qids)

    st = {
        "mode": "test",
        "header": "📝 <b>Тестування</b>",
        "pending": all_qids,
        "skipped": [],
        "phase": "pending",
        "feedback": None,
        "current_qid": None,
        "correct_count": 0,
        "total": len(all_qids),
        "started_at": dt_to_iso(now()),
    }
    await store.set_state(uid, st)

    await cb.answer()
    await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message)

# -------- Statistics --------

@router.callback_query(F.data == "nav:stats")
async def nav_stats(cb: CallbackQuery, bot: Bot, store: Storage):
    uid = cb.from_user.id
    s = await store.stats(uid)

    if s["count"] == 0:
        text = "📊 <b>Статистика</b>\n\nЩе немає завершених тестів."
    else:
        last = s["last"]

        dt = last.get("finished_at")
        if isinstance(dt, str):
            dt = iso_to_dt(dt)  # твоя функція
        # якщо dt вже datetime — залишаємо як є
        finished = dt.strftime("%d.%m.%Y %H:%M") if isinstance(dt, datetime) else "—"

        percent = last.get("percent")
        try:
            percent_f = float(percent)
        except (TypeError, ValueError):
            percent_f = 0.0

        text = (
            "📊 <b>Статистика</b>\n\n"
            f"Тестів (останні 50): <b>{s['count']}</b>\n"
            f"Середній результат: <b>{s['avg']:.1f}%</b>\n\n"
            f"Останній тест:\n"
            f"• {finished}\n"
            f"• {last['correct']}/{last['total']} = {percent_f:.1f}%"
        )

    await render_main(
        bot, store, uid, cb.message.chat.id,
        text,
        kb_inline([("⬅️ Меню", "nav:menu")], row=1),
        message=cb.message
    )
    await cb.answer()


# -------- Admin: users --------

def _admin_user_icon(u: Dict[str, Any]) -> str:
    # 🟢 активна підписка | 🟡 тріал | 🔴 без доступу | ⚪️ без номера
    phone = u.get("phone")
    if not phone:
        return "⚪️"

    t_end = u.get("trial_end")
    s_end = u.get("sub_end")
    inf = bool(u.get("sub_infinite"))
    n = now()

    # ✅ ПІДПИСКА має пріоритет над тріалом
    if inf or (s_end and n <= s_end):
        return "🟢"
    if t_end and n <= t_end:
        return "🟡"
    return "🔴"



def fmt_user_row(u: Dict[str, Any]) -> str:
    phone = u.get("phone") or "без номера"
    fn = (u.get("first_name") or "").strip()
    ln = (u.get("last_name") or "").strip()
    full = " ".join([x for x in [fn, ln] if x]).strip() or "—"
    return f"{_admin_user_icon(u)} {phone} | {full}"

def _is_not_modified_error(e: TelegramBadRequest) -> bool:
    return "message is not modified" in (str(e) or "").lower()


async def render_admin_view(
    bot: Bot,
    store: "Storage",
    uid: int,
    chat_id: int,
    text: str,
    kb: InlineKeyboardMarkup,
    message: Optional[Message] = None,
):
    """
    Редагуємо ОДНЕ адмін-повідомлення.
    Важливо: якщо Telegram каже "message is not modified" — це НЕ помилка,
    нові повідомлення НЕ створюємо.
    """
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}

    async def _try_edit(c_id: int, m_id: int) -> bool:
        try:
            await bot.edit_message_text(
                chat_id=c_id,
                message_id=m_id,
                text=text,
                reply_markup=kb,
                parse_mode="HTML",
            )
            return True
        except TelegramBadRequest as e:
            if _is_not_modified_error(e):
                return True  # нічого не змінилось — але це ок, НЕ шлемо нове
            return False
        except Exception:
            return False

    # 1) Якщо є message з callback — редагуємо його і запамʼятовуємо
    if message:
        st[ADMIN_PANEL_MSG_ID] = message.message_id
        st[ADMIN_PANEL_CHAT_ID] = message.chat.id
        await store.set_state(uid, st)

        if await _try_edit(message.chat.id, message.message_id):
            return

    # 2) Якщо message=None — редагуємо останнє адмін-меню зі state
    msg_id = st.get(ADMIN_PANEL_MSG_ID)
    c_id = st.get(ADMIN_PANEL_CHAT_ID) or chat_id

    if msg_id and await _try_edit(c_id, msg_id):
        return

    # 3) Якщо реально нема що редагувати — тоді створюємо 1 нове
    sent = await bot.send_message(chat_id, text, reply_markup=kb, parse_mode="HTML")
    st[ADMIN_PANEL_MSG_ID] = sent.message_id
    st[ADMIN_PANEL_CHAT_ID] = chat_id
    await store.set_state(uid, st)



# --- renders ---
async def render_admin_users_list(
    bot: Bot,
    store: "Storage",
    admin_uid: int,
    chat_id: int,
    offset: int,
    message: Optional[Message] = None,
):
    ui = await store.get_ui(admin_uid)
    st = ui.get("state", {}) or {}
    query_digits = (st.get(ADMIN_USERS_QUERY) or "").strip()

    limit = 10

    # беремо limit+1 щоб визначити чи є "next"
    if query_digits:
        items = await store.search_users_by_phone(query_digits, offset, limit + 1)
        search_line = f"\n🔎 Пошук: <code>{hescape(query_digits)}</code>\n"
    else:
        items = await store.list_users(offset, limit + 1)
        search_line = ""

    has_next = len(items) > limit
    users = items[:limit]

    c_green = c_yellow = c_red = c_white = 0
    for u in users:
        ic = _admin_user_icon(u)
        if ic == "🟢":
            c_green += 1
        elif ic == "🟡":
            c_yellow += 1
        elif ic == "🔴":
            c_red += 1
        else:
            c_white += 1

    text = (
        "🛠 <b>Користувачі</b>\n"
        "🟢 підписка | 🟡 тріал | 🔴 без доступу | ⚪️ без номера\n"
        f"У цьому списку: 🟢{c_green} 🟡{c_yellow} 🔴{c_red} ⚪️{c_white}"
        f"{search_line}\n"
        "Оберіть користувача:"
    )

    rows: list[list[InlineKeyboardButton]] = []

    # пошук
    top = [
        InlineKeyboardButton(
            text="🔎 Пошук по номеру",
            callback_data=clamp_callback(f"admin:users_search:{offset}"),
        )
    ]
    if query_digits:
        top.append(
            InlineKeyboardButton(
                text="❌ Очистити",
                callback_data=clamp_callback(f"admin:users_clear:{offset}"),
            )
        )
    rows.append(top)

    # користувачі
    if users:
        for u in users:
            rows.append(
                [
                    InlineKeyboardButton(
                        text=fmt_user_row(u),
                        callback_data=clamp_callback(f"admin:user:{u['user_id']}:{offset}"),
                    )
                ]
            )
    else:
        rows.append([InlineKeyboardButton(text="— Нічого не знайдено —", callback_data="noop")])

    # навігація
    nav_row: list[InlineKeyboardButton] = []
    if offset > 0:
        nav_row.append(
            InlineKeyboardButton(
                text="⬅️",
                callback_data=clamp_callback(f"admin:users:{max(0, offset - limit)}"),
            )
        )
    if has_next:
        nav_row.append(
            InlineKeyboardButton(
                text="➡️",
                callback_data=clamp_callback(f"admin:users:{offset + limit}"),
            )
        )
    if nav_row:
        rows.append(nav_row)

    # меню
    rows.append([InlineKeyboardButton(text="⬅️ Меню", callback_data=clamp_callback("nav:menu"))])

    kb = InlineKeyboardMarkup(inline_keyboard=rows)
    await render_admin_view(bot, store, admin_uid, chat_id, text, kb, message=message)


async def render_admin_users_search_prompt(
    bot: Bot,
    store: "Storage",
    uid: int,
    chat_id: int,
    back_offset: int,
    message: Optional[Message] = None,
    error: Optional[str] = None,
):
    err = f"\n\n⚠️ {hescape(error)}" if error else ""
    text = (
        "🔎 <b>Пошук по номеру</b>\n\n"
        "Надішли номер (можна частину).\n"
        "Я шукаю по цифрах, наприклад: <code>38067</code> або <code>067</code>."
        f"{err}"
    )
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⬅️ Назад", callback_data=clamp_callback(f"admin:users:{back_offset}"))]
        ]
    )
    await render_admin_view(bot, store, uid, chat_id, text, kb, message=message)


async def render_admin_user_detail(
    bot: Bot,
    store: "Storage",
    admin_uid: int,
    chat_id: int,
    target_id: int,
    back_offset: int,
    message: Optional[Message] = None,
):
    user = await store.get_user(target_id)

    phone_html = hescape(user.get("phone") or "—")
    first_html = hescape(user.get("first_name") or "—")
    last_html = hescape(user.get("last_name") or "—")

    text = (
        "👤 <b>Користувач</b>\n\n"
        f"ID: <b>{target_id}</b>\n"
        f"Телефон: <b>{phone_html}</b>\n"
        f"Ім'я: <b>{first_html}</b>\n"
        f"Прізвище: <b>{last_html}</b>\n"
        f"{fmt_access_line(user)}"
    )

    ok, stt = access_status(user)
    is_inf = (stt == "sub_infinite")

    b = InlineKeyboardBuilder()

    if is_inf:
        b.button(text="🟢 Доступ безкінечно (активний)", callback_data="noop")
        b.button(
            text="🔴 Завершити доступ",
            callback_data=clamp_callback(f"admin:subcancel:{target_id}:{back_offset}"),
        )
    else:
        b.button(
            text="🔴 Дати доступ (безкінечно)",
            callback_data=clamp_callback(f"admin:subinf:{target_id}:{back_offset}"),
        )
        b.button(
            text="🚫 Забрати доступ",
            callback_data=clamp_callback(f"admin:subcancel:{target_id}:{back_offset}"),
        )

    b.adjust(1)
    b.row(
        InlineKeyboardButton(
            text="⬅️ Назад",
            callback_data=clamp_callback(f"admin:users:{back_offset}"),
        )
    )

    await render_admin_view(bot, store, admin_uid, chat_id, text, b.as_markup(), message=message)


# --- handlers ---
@router.callback_query(F.data == "noop")
async def noop(cb: CallbackQuery):
    await cb.answer()


@router.callback_query(F.data.startswith("admin:users:"))
async def admin_users(cb: CallbackQuery, bot: Bot, store: "Storage", admin_ids: set[int]):
    uid = cb.from_user.id
    if uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    try:
        offset = int(cb.data.split(":")[2])
    except Exception:
        await cb.answer("Помилка")
        return

    await render_admin_users_list(bot, store, uid, cb.message.chat.id, offset, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("admin:users_search:"))
async def admin_users_search_prompt(cb: CallbackQuery, bot: Bot, store: "Storage", admin_ids: set[int]):
    uid = cb.from_user.id
    if uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    try:
        back_offset = int(cb.data.split(":")[2])
    except Exception:
        await cb.answer("Помилка")
        return

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    st[ADMIN_USERS_AWAITING] = "admin_users_phone"
    st[ADMIN_USERS_BACK_OFFSET] = back_offset
    await store.set_state(uid, st)

    await render_admin_users_search_prompt(bot, store, uid, cb.message.chat.id, back_offset, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("admin:users_clear:"))
async def admin_users_clear(cb: CallbackQuery, bot: Bot, store: "Storage", admin_ids: set[int]):
    uid = cb.from_user.id
    if uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    try:
        back_offset = int(cb.data.split(":")[2])
    except Exception:
        await cb.answer("Помилка")
        return

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    st.pop(ADMIN_USERS_QUERY, None)
    st.pop(ADMIN_USERS_AWAITING, None)
    st.pop(ADMIN_USERS_BACK_OFFSET, None)
    await store.set_state(uid, st)

    await render_admin_users_list(bot, store, uid, cb.message.chat.id, back_offset, message=cb.message)
    await cb.answer("Очищено")


@router.message(F.text)
async def admin_users_search_input(
    message: Message,
    bot: Bot,
    store: "Storage",
    admin_ids: set[int],
):
    uid = message.from_user.id
    if uid not in admin_ids:
        raise SkipHandler()

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    if st.get(ADMIN_USERS_AWAITING) != "admin_users_phone":
        raise SkipHandler()

    back_offset = int(st.get(ADMIN_USERS_BACK_OFFSET) or 0)

    digits = "".join(ch for ch in (message.text or "").strip() if ch.isdigit())

    # прибрати повідомлення з номером (щоб не світити)
    try:
        await message.delete()
    except Exception:
        pass

    if not digits:
        await render_admin_users_search_prompt(
            bot, store, uid,
            st.get(ADMIN_PANEL_CHAT_ID) or message.chat.id,
            back_offset,
            message=None,
            error="Введи хоча б одну цифру",
        )
        return

    st[ADMIN_USERS_QUERY] = digits
    st.pop(ADMIN_USERS_AWAITING, None)
    st.pop(ADMIN_USERS_BACK_OFFSET, None)
    await store.set_state(uid, st)

    await render_admin_users_list(
        bot, store, uid,
        st.get(ADMIN_PANEL_CHAT_ID) or message.chat.id,
        0,
        message=None,
    )

@router.callback_query(F.data.startswith("admin:user:"))
async def admin_user_detail(cb: CallbackQuery, bot: Bot, store: "Storage", admin_ids: set[int]):
    admin_uid = cb.from_user.id
    if admin_uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    try:
        _, _, target_id, back_offset = cb.data.split(":")
        target_id = int(target_id)
        back_offset = int(back_offset)
    except Exception:
        await cb.answer("Помилка")
        return

    await render_admin_user_detail(bot, store, admin_uid, cb.message.chat.id, target_id, back_offset, message=cb.message)
    await cb.answer()

@router.callback_query(F.data.startswith("admin:subinf:"))
async def admin_sub_inf(cb: CallbackQuery, bot: Bot, store: "Storage", admin_ids: set[int]):
    admin_uid = cb.from_user.id
    if admin_uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    try:
        _, _, target_id, back_offset = cb.data.split(":")
        target_id = int(target_id)
        back_offset = int(back_offset)
    except Exception:
        await cb.answer("Помилка")
        return

    await store.set_subscription(target_id, None, infinite=True)
    await render_admin_user_detail(bot, store, admin_uid, cb.message.chat.id, target_id, back_offset, message=cb.message)
    await cb.answer()

@router.callback_query(F.data.startswith("admin:subcancel:"))
async def admin_sub_cancel(cb: CallbackQuery, bot: Bot, store: "Storage", admin_ids: set[int]):
    admin_uid = cb.from_user.id
    if admin_uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    try:
        _, _, target_id, back_offset = cb.data.split(":")
        target_id = int(target_id)
        back_offset = int(back_offset)
    except Exception:
        await cb.answer("Помилка")
        return

    # забрати доступ
    await store.set_subscription(target_id, None, infinite=False)

    await render_admin_user_detail(bot, store, admin_uid, cb.message.chat.id, target_id, back_offset, message=cb.message)
    await cb.answer("Ок")

# -------------------- Bootstrap --------------------

async def main():
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Set BOT_TOKEN env var")

    admin_ids_env = os.getenv("ADMIN_IDS", "").strip()
    admin_ids = set()
    if admin_ids_env:
        for x in admin_ids_env.split(","):
            x = x.strip()
            if x.isdigit():
                admin_ids.add(int(x))

    dsn = (
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("POSTGRESQL_URL")
        or os.getenv("PGDATABASE_URL")
    )
    if not dsn:
        raise RuntimeError("Set DATABASE_URL env var (Railway → Variables).")

    store = Storage(dsn)
    await store.init()

    # ---------- Questions bootstrap ----------
    questions_path = os.getenv("QUESTIONS_PATH", "questions_flat.json")

    auto_import = (os.getenv("QUESTIONS_AUTO_IMPORT", "1") or "").strip().lower() in ("1", "true", "yes", "y", "on")
    force_import = (os.getenv("QUESTIONS_FORCE_IMPORT", "0") or "").strip().lower() in ("1", "true", "yes", "y", "on")

    if auto_import:
        cnt = await store.questions_count()
        if force_import or cnt == 0:
            if not os.path.exists(questions_path):
                if cnt == 0:
                    raise RuntimeError(f"Questions table is empty and file not found: {questions_path}")
            else:
                n = await store.import_questions_from_json(
                    questions_path,
                    changed_by="bootstrap",
                    force=force_import,
                )
                print(f"Questions imported/updated: {n}")

    qb = QuestionBank(questions_path)
    await qb.load_from_db(store)

    if not qb.by_id:
        raise RuntimeError("No questions loaded from DB. Fill table 'questions' first.")

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp["store"] = store
    dp["qb"] = qb
    dp["admin_ids"] = admin_ids

    dp.include_router(router)

    async def _inject_middleware(handler, event, data):
        data["store"] = store
        data["qb"] = qb
        data["admin_ids"] = admin_ids
        return await handler(event, data)

    dp.update.outer_middleware(_inject_middleware)

    print(f"Loaded: law={len(qb.law)} | ok_modules={len(qb.ok_modules)} | questions={len(qb.by_id)}")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
