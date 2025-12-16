import os
import json
import asyncio
import random
import asyncpg
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from html import escape as hescape
from aiogram.client.default import DefaultBotProperties
from aiogram.types import InlineKeyboardButton
from aiogram.exceptions import TelegramBadRequest


from aiogram import Bot, Dispatcher, Router, F
from aiogram.enums import ParseMode
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
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
GROUP_URL = normalize_tme_url(os.getenv("GROUP_URL", "t.me/mytnytsia_tests"))

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


def clamp_callback(s: str) -> str:
    # Telegram limit for callback_data ~64 bytes; keep short keys.
    return s[:60]



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

    async def ensure_user(self, user_id: int, is_admin: bool = False):
        await self._exec("""
            INSERT INTO users (user_id, is_admin, created_at)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id) DO UPDATE SET is_admin=EXCLUDED.is_admin
        """, user_id, 1 if is_admin else 0, now())

    async def get_user(self, user_id: int) -> dict:
        r = await self._fetchrow("SELECT * FROM users WHERE user_id=$1", user_id)
        if not r:
            return {}
        d = dict(r)
        d["ok_modules"] = json.loads(d.get("ok_modules_json") or "[]")
        d["ok_last_levels"] = json.loads(d.get("ok_last_levels_json") or "{}")
        return d

    async def set_phone_and_trial(self, user_id: int, phone: str):
        ts = now()
        te = ts + timedelta(days=3)
        await self._exec("""
            UPDATE users SET phone=$1, trial_start=$2, trial_end=$3 WHERE user_id=$4
        """, phone, ts, te, user_id)

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
            UPDATE ui_state SET state_json=$1 WHERE user_id=$2
        """, json.dumps(state, ensure_ascii=False), user_id)

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
            SELECT user_id, phone, trial_end, sub_end, sub_infinite, created_at
            FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2
        """, limit, offset)
        return [dict(r) for r in rows]

    async def search_users_by_phone(self, phone_digits: str, offset: int, limit: int) -> list[dict]:
        q = "".join(ch for ch in (phone_digits or "").strip() if ch.isdigit())
        if not q:
            return []
        rows = await self._fetch("""
            SELECT user_id, phone, trial_end, sub_end, sub_infinite, created_at
            FROM users
            WHERE regexp_replace(COALESCE(phone,''), '\\D', '', 'g') LIKE '%' || $1 || '%'
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
        """, q, limit, offset)
        return [dict(r) for r in rows]



# -------------------- Question bank --------------------

@dataclass
class Q:
    id: int
    section: str
    topic: str
    ok: Optional[str]
    level: Optional[int]
    question: str
    choices: List[str]
    correct: List[int]
    correct_texts: List[str]

    @property
    def is_valid_mcq(self) -> bool:
        return bool(self.choices) and bool(self.correct)


class QuestionBank:
    def __init__(self, path: str):
        self.path = path
        self.by_id: Dict[int, Q] = {}
        self.law: List[int] = []
        self.law_groups: Dict[str, List[int]] = {}   # key -> qids
        self.ok_modules: Dict[str, Dict[int, List[int]]] = {}  # ok -> level -> qids

    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for item in raw:
            q = Q(
                id=int(item.get("id")),
                section=str(item.get("section") or ""),
                topic=str(item.get("topic") or ""),
                ok=item.get("ok"),
                level=item.get("level") if item.get("level") is None else int(item.get("level")),
                question=str(item.get("question") or ""),
                choices=list(item.get("choices") or []),
                correct=list(item.get("correct") or []),
                correct_texts=list(item.get("correct_texts") or []),
            )
            self.by_id[q.id] = q

        # Index: law questions
        for qid, q in self.by_id.items():
            if not q.is_valid_mcq:
                continue
            if "законодавств" in q.section.lower():
                self.law.append(qid)
                key = self._law_group_key(q.topic)
                self.law_groups.setdefault(key, []).append(qid)

        # Index: OK questions
        for qid, q in self.by_id.items():
            if not q.is_valid_mcq:
                continue
            if q.ok:
                self.ok_modules.setdefault(q.ok, {})
                lvl = int(q.level or 1)
                self.ok_modules[q.ok].setdefault(lvl, []).append(qid)

        # stable order
        for k in self.law_groups:
            self.law_groups[k].sort()
        self.law.sort()
        for ok in self.ok_modules:
            for lvl in self.ok_modules[ok]:
                self.ok_modules[ok][lvl].sort()

    def _law_group_key(self, topic: str) -> str:
        topic = (topic or "").strip()
        # typical format: "1. ...", "2. ..."
        if len(topic) >= 2 and topic[0].isdigit() and topic[1] == ".":
            return topic.split(".", 1)[0].strip()  # "1", "2", ...
        # fallback: group by full topic
        return topic[:60] or "Законодавство"

    def law_group_title(self, key: str) -> str:
        # try to find any topic starting with "key."
        if key.isdigit():
            for qid in self.law_groups.get(key, []):
                t = self.by_id[qid].topic.strip()
                if t.startswith(f"{key}."):
                    return t.split(".", 1)[1].strip()
        return key

    def pick_random(self, qids: List[int], n: int) -> List[int]:
        if len(qids) <= n:
            return list(qids)
        return random.sample(qids, n)


# -------------------- Access --------------------

def access_status(user: Dict[str, Any]) -> Tuple[bool, str]:
    # registered?
    if not user or not user.get("phone"):
        return False, "not_registered"

    t_end: Optional[datetime] = user.get("trial_end")
    s_end: Optional[datetime] = user.get("sub_end")
    inf: bool = bool(user.get("sub_infinite"))

    n = now()
    if t_end and n <= t_end:
        return True, "trial"
    if inf:
        return True, "sub_infinite"
    if s_end and n <= s_end:
        return True, "sub_active"
    return False, "expired"


# -------------------- UI helpers --------------------

def kb_inline(buttons: List[Tuple[str, str]], row: int = 2) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for text, data in buttons:
        b.button(text=text, callback_data=clamp_callback(data))
    b.adjust(row)
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
    # Prefer editing existing message (single-message concept)
    if message:
        try:
            await message.edit_text(text, reply_markup=keyboard, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            await store.set_ui(user_id, chat_id, message.message_id)
            return
        except Exception:
            pass

    ui = await store.get_ui(user_id)
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
        except Exception:
            pass

    sent = await bot.send_message(chat_id, text, reply_markup=keyboard, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    await store.set_ui(user_id, chat_id, sent.message_id)


# -------------------- Screens --------------------

def screen_need_registration() -> Tuple[str, InlineKeyboardMarkup]:
    text = (
        "Щоб користуватись ботом, потрібна реєстрація.\n\n"
        "Натисни кнопку нижче."
    )
    kb = kb_inline([
        ("📱 Поділитися номером", "reg:request"),
    ], row=1)
    return text, kb


def screen_main_menu(user: Dict[str, Any], is_admin: bool) -> Tuple[str, InlineKeyboardMarkup]:
    text = (
        "🏠 <b>Головне меню</b>\n"
        f"{fmt_access_line(user)}\n\n"
        "Оберіть розділ:"
    )
    buttons = [
        ("📚 Навчання", "nav:learn"),
        ("📝 Тестування", "nav:test"),
        ("📊 Статистика", "nav:stats"),
        ("❓ Допомога", "nav:help"),
    ]
    if is_admin:
        buttons.append(("🛠 Користувачі", "admin:users:0"))
    kb = kb_inline(buttons, row=2)
    return text, kb


def screen_help(admin_url: str) -> Tuple[str, InlineKeyboardMarkup]:
    text = (
        "❓ <b>Допомога</b>\n\n"
        "Тут ви можете:\n"
        "▪ приєднатися до Telegram-групи\n"
        "▪ звернутися до адміністратора для продовження підписки"
    )

    b = InlineKeyboardBuilder()
    if GROUP_URL:
        b.button(text="🔗 Telegram-група", url=GROUP_URL)
    if admin_url:
        b.button(text="📩 Написати адміну", url=admin_url)
    b.button(text="⬅️ Меню", callback_data="nav:menu")
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


def screen_learning_menu() -> Tuple[str, InlineKeyboardMarkup]:
    text = "📚 <b>Навчання</b>\n\nОберіть напрям:"
    kb = kb_inline([
        ("📜 Законодавство", "learn:law"),
        ("🧩 Операційні компетенції (ОК)", "learn:ok"),
        ("🧯 Робота над помилками", "learn:mistakes"),
        ("⬅️ Меню", "nav:menu"),
    ], row=1)
    return text, kb


def screen_law_groups(qb: QuestionBank) -> Tuple[str, InlineKeyboardMarkup]:
    # show up to 4 groups, but will work with any amount
    keys = list(qb.law_groups.keys())

    def key_sort(k: str):
        return (0, int(k)) if k.isdigit() else (1, k)

    keys.sort(key=key_sort)

    shown = keys[:4]
    buttons = []
    for k in shown:
        title = clean_law_title(qb.law_group_title(k))
        buttons.append((f"{k}. {title}" if k.isdigit() else title, f"lawgrp:{k}"))

    text = "📜 <b>Законодавство</b>\n\nОберіть пункт:"
    buttons.append(("⬅️ Назад", "nav:learn"))
    kb = kb_inline(buttons, row=1)
    return text, kb


def screen_law_parts(group_key: str, qb: QuestionBank) -> Tuple[str, InlineKeyboardMarkup]:
    qids = qb.law_groups.get(group_key, [])
    total = len(qids)

    header = "📜 <b>Законодавство</b>"

    if total <= 50:
        text = f"{header}\n\nПитань: {total}\nПочати?"
        kb = kb_inline([
            ("▶️ Почати", f"learn_start:law:{group_key}:1"),
            ("🎲 Рандомні", f"learn_start:lawrand:{group_key}"),
            ("⬅️ Назад", "learn:law"),
        ], row=1)
        return text, kb

    # make parts: 1-50, 51-100, ...
    part_size = 50
    parts = []
    p = 1
    for i in range(0, total, part_size):
        a = i + 1
        b = min(i + part_size, total)
        parts.append((p, a, b))
        p += 1

    text = f"{header}\n\nОберіть частину:"
    buttons = []

    # нове:
    buttons.append(("🎲 Рандомні 50", f"learn_start:lawrand:{group_key}"))

    for p, a, b in parts:
        buttons.append((f"{a}–{b}", f"learn_start:law:{group_key}:{p}"))

    buttons.append(("⬅️ Назад", "learn:law"))
    kb = kb_inline(buttons, row=2)
    return text, kb


def screen_ok_menu(user_modules: List[str], qb: QuestionBank) -> Tuple[str, InlineKeyboardMarkup]:
    if not user_modules:
        text = "🧩 <b>ОК</b>\n\nСпочатку оберіть модулі (можна кілька)."
        kb = kb_inline([
            ("✅ Обрати модулі", "okmods:pick"),
            ("⬅️ Назад", "nav:learn"),
        ], row=1)
        return text, kb

    text = "🧩 <b>ОК</b>\n\nОберіть модуль:"
    buttons = []
    for m in user_modules:
        if m in qb.ok_modules:
            buttons.append((m, f"okmod:{m}"))
    buttons += [
        ("🔁 Змінити модулі", "okmods:pick"),
        ("⬅️ Назад", "nav:learn"),
    ]
    kb = kb_inline(buttons, row=2)
    return text, kb


def screen_ok_modules_pick(selected: List[str], qb: QuestionBank) -> Tuple[str, InlineKeyboardMarkup]:
    text = "🧩 <b>Оберіть модулі ОК</b>\n\nПозначте потрібні модулі:"
    all_mods = sorted(qb.ok_modules.keys(), key=lambda x: (len(x), x))
    b = InlineKeyboardBuilder()
    for m in all_mods:
        mark = "✅" if m in selected else "⬜️"
        b.button(text=f"{mark} {m}", callback_data=clamp_callback(f"okmods:toggle:{m}"))
    b.adjust(2)
    b.row()
    b.button(text="Готово", callback_data="okmods:save")
    b.button(text="⬅️ Назад", callback_data="learn:ok")
    return text, b.as_markup()


def screen_ok_levels(module: str, qb: QuestionBank) -> Tuple[str, InlineKeyboardMarkup]:
    levels = sorted(qb.ok_modules.get(module, {}).keys())
    text = f"🧩 <b>{module}</b>\n\nОберіть рівень:"
    buttons = [(f"Рівень {lvl}", f"learn_start:ok:{module}:{lvl}") for lvl in levels]
    buttons.append(("⬅️ Назад", "learn:ok"))
    kb = kb_inline(buttons, row=2)
    return text, kb


def screen_test_config(modules: List[str], qb: QuestionBank, temp_levels: Dict[str, int]) -> Tuple[str, InlineKeyboardMarkup]:
    lines = [
        "📝 <b>Тестування</b>",
        "",
        "Оберіть рівень для кожного модуля ОК (за потреби):",
        "Потім натисніть «Почати тест».",
    ]
    if not modules:
        lines += ["", "ℹ️ ОК-модулі не обрані — тест буде лише із законодавства (50 питань)."]

    buttons: List[Tuple[str, str]] = []
    for i, m in enumerate(modules):
        levels_map = qb.ok_modules.get(m, {})
        if not levels_map:
            continue
        available = sorted(levels_map.keys())
        lvl = int(temp_levels.get(m, available[0]))
        if lvl not in available:
            lvl = available[0]
        buttons.append((f"🧩 {m} • Рівень {lvl}", f"testlvl:modi:{i}"))

    buttons += [
        ("▶️ Почати тест", "test:start"),
        ("⬅️ Меню", "nav:menu"),
    ]
    return "\n".join(lines), kb_inline(buttons, row=1)


def screen_test_pick_level(idx: int, module: str, qb: QuestionBank, current: Optional[int]) -> Tuple[str, InlineKeyboardMarkup]:
    levels = sorted(qb.ok_modules.get(module, {}).keys())
    text = f"🧩 <b>{module}</b>\n\nОберіть рівень для тесту:"
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
        "",
        "━━━━━━━━━━━━━━━━",
        "📝 <b>Варіанти відповіді</b>",
    ]

    for i, ch in enumerate(choices):
        lines.append(f"<b>{i+1})</b> {ch}")

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
        ("⬅️ Продовжити", "leave:back"),
        ("✅ Вийти в меню", "leave:yes"),
    ], row=1)


# -------------------- Main app logic --------------------

router = Router()


@router.message(F.text == "/start")
async def cmd_start(message: Message, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    uid = message.from_user.id
    await store.ensure_user(uid, is_admin=(uid in admin_ids))
    user = await store.get_user(uid)

    # try delete /start to avoid chat clutter
    try:
        await message.delete()
    except Exception:
        pass

    ui = await store.get_ui(uid)
    chat_id = message.chat.id

    if not user.get("phone"):
        text, kb = screen_need_registration()
        await render_main(bot, store, uid, chat_id, text, kb)
        return

    text, kb = screen_main_menu(user, is_admin=(uid in admin_ids))
    await render_main(bot, store, uid, chat_id, text, kb)


@router.callback_query(F.data == "nav:menu")
async def nav_menu(cb: CallbackQuery, bot: Bot, store: Storage, admin_ids: set[int]):
    uid = cb.from_user.id
    user = await store.get_user(uid)
    text, kb = screen_main_menu(user, is_admin=(uid in admin_ids))
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await store.set_state(uid, {})
    await cb.answer()


@router.callback_query(F.data == "nav:help")
async def nav_help(cb: CallbackQuery, bot: Bot, store: Storage, admin_ids: set[int]):
    admin_url = get_admin_contact_url(admin_ids)
    text, kb = screen_help(admin_url)
    await render_main(bot, store, cb.from_user.id, cb.message.chat.id, text, kb, message=cb.message)
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

    text, kb = screen_learning_menu()
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data == "learn:law")
async def learn_law(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    text, kb = screen_law_groups(qb)
    await render_main(bot, store, cb.from_user.id, cb.message.chat.id, text, kb, message=cb.message)
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
    text, kb = screen_ok_menu(modules, qb)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data == "okmods:pick")
async def okmods_pick(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    user = await store.get_user(uid)
    ui = await store.get_ui(uid)
    state = ui.get("state", {})
    selected = state.get("okmods_temp")
    if selected is None:
        selected = list(user.get("ok_modules", []))
    state["okmods_temp"] = selected
    await store.set_state(uid, state)

    text, kb = screen_ok_modules_pick(selected, qb)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("okmods:toggle:"))
async def okmods_toggle(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    mod = cb.data.split(":", 2)[2]

    ui = await store.get_ui(uid)
    state = ui.get("state", {})
    selected = list(state.get("okmods_temp", []))
    if mod in selected:
        selected.remove(mod)
    else:
        selected.append(mod)
    state["okmods_temp"] = selected
    await store.set_state(uid, state)

    text, kb = screen_ok_modules_pick(selected, qb)
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
    await store.set_state(uid, state)

    user = await store.get_user(uid)
    text, kb = screen_ok_menu(user.get("ok_modules", []), qb)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer("Збережено")


@router.callback_query(F.data.startswith("okmod:"))
async def okmod_levels(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    module = cb.data.split(":", 1)[1]
    text, kb = screen_ok_levels(module, qb)
    await render_main(bot, store, cb.from_user.id, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


# -------- Registration (contact) --------

@router.callback_query(F.data == "reg:request")
async def reg_request(cb: CallbackQuery, bot: Bot, store: Storage):
    uid = cb.from_user.id
    chat_id = cb.message.chat.id

    # main message stays the same, but we must show a ReplyKeyboard (contact) -> temporary message
    await render_main(
        bot, store, uid, chat_id,
        "📱 <b>Реєстрація</b>\n\nНатисни кнопку нижче, щоб поділитися номером.",
        kb_inline([("⬅️ Назад", "nav:menu")], row=1),
        message=cb.message
    )

    rk = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Поділитися номером", request_contact=True)]],
        resize_keyboard=True,
        one_time_keyboard=True
    )
    tmp = await bot.send_message(chat_id, "👇 Поділись номером (кнопка внизу)", reply_markup=rk)

    ui = await store.get_ui(uid)
    st = ui.get("state", {})
    st["reg_tmp_msg_id"] = tmp.message_id
    await store.set_state(uid, st)

    await cb.answer()


@router.message(F.contact)
async def on_contact(message: Message, bot: Bot, store: Storage, admin_ids: set[int]):
    uid = message.from_user.id
    await store.ensure_user(uid, is_admin=(uid in admin_ids))

    # accept only own contact
    if not message.contact or message.contact.user_id != uid:
        try:
            await message.delete()
        except Exception:
            pass
        return

    phone = message.contact.phone_number
    await store.set_phone_and_trial(uid, phone)

    # cleanup: delete contact and temp message if possible
    try:
        await message.delete()
    except Exception:
        pass

    ui = await store.get_ui(uid)
    st = ui.get("state", {})
    tmp_id = st.get("reg_tmp_msg_id")
    if tmp_id:
        try:
            await bot.delete_message(message.chat.id, tmp_id)
        except Exception:
            pass
        st.pop("reg_tmp_msg_id", None)
        await store.set_state(uid, st)

    # remove reply keyboard (best effort)
    try:
        rm = await bot.send_message(message.chat.id, "✅", reply_markup=ReplyKeyboardRemove())
        try:
            await bot.delete_message(message.chat.id, rm.message_id)
        except Exception:
            pass
    except Exception:
        pass

    user = await store.get_user(uid)
    text, kb = screen_main_menu(user, is_admin=(uid in admin_ids))
    await render_main(bot, store, uid, message.chat.id, text, kb)


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
        await store.set_state(uid, {})
        await render_main(bot, store, uid, chat_id, text, kb_inline([("⬅️ Навчання", "nav:learn")], row=1), message=message)
        return

    qid = int(pending[0])
    q = qb.by_id.get(qid)
    if not q or not q.is_valid_mcq:
        # skip invalid
        st["pending"] = pending[1:]
        await store.set_state(uid, st)
        await show_next_in_session(bot, store, qb, uid, chat_id, message)
        return

    # set current qid
    st["current_qid"] = qid
    await store.set_state(uid, st)

    total = int(st.get("total", 0)) or (len(pending) + len(skipped))
    done = total - len(pending) - len(skipped)
    phase_note = " (пропущені)" if phase == "skipped" else ""
    remaining = len(pending) + len(skipped)
    progress = f"Питань залишилось: <b>{remaining}</b>{phase_note}"

    header = st.get("header", "")
    text = build_question_text(q, header, progress)
    await render_main(bot, store, uid, chat_id, text, kb_answers(len(q.choices)), message=message)


@router.callback_query(F.data.startswith("learn_start:"))
async def learn_start(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    uid = cb.from_user.id
    user = await store.get_user(uid)
    ok_access, _ = access_status(user)
    if not ok_access:
        admin_url = get_admin_contact_url(admin_ids)
        text, kb = screen_no_access(user, admin_url)
        await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
        await cb.answer()
        return

    parts = cb.data.split(":")
    # learn_start:law:<group_key>:<part>
    # learn_start:lawrand:<group_key>
    # learn_start:ok:<module>:<level>
    kind = parts[1]

    if kind == "law":
        group_key = parts[2]
        part = int(parts[3])
        qids = qb.law_groups.get(group_key, [])
        if len(qids) > 50:
            start = (part - 1) * 50
            end = start + 50
            qids = qids[start:end]

        # прибрали показ "Законодавство / Пункт ..."
        header = ""

        await start_learning_session(
            bot, store, qb, uid, cb.message.chat.id, cb.message,
            qids=qids,
            header=header,
            save_meta={"kind": "law", "group": group_key, "part": part},
        )
        await cb.answer()
        return

    if kind == "lawrand":
        group_key = parts[2]
        all_qids = qb.law_groups.get(group_key, [])

        n = min(50, len(all_qids))
        qids = qb.pick_random(all_qids, n)

        # прибрали показ "Законодавство / Пункт ... / Рандомні ..."
        header = ""

        await start_learning_session(
            bot, store, qb, uid, cb.message.chat.id, cb.message,
            qids=qids,
            header=header,
            save_meta={"kind": "lawrand", "group": group_key, "part": 0},
        )
        await cb.answer()
        return

    if kind == "ok":
        module = parts[2]
        level = int(parts[3])
        qids = qb.ok_modules.get(module, {}).get(level, [])
        await store.set_ok_last_level(uid, module, level)
        header = f"🧩 <b>ОК</b>\n{module} • Рівень {level}"

        await start_learning_session(
            bot, store, qb, uid, cb.message.chat.id, cb.message,
            qids=qids,
            header=header,
            save_meta={"kind": "ok", "module": module, "level": level},
        )
        await cb.answer()
        return

    await cb.answer("Невідомий режим")


@router.callback_query(F.data == "ans:0")  # placeholder; real handler below
async def _noop(cb: CallbackQuery):
    await cb.answer()


@router.callback_query(F.data.startswith("ans:"))
async def on_answer(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
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
            st["feedback"] = None
        else:
            # bump wrong counter -> move into mistakes after 5 wrong
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
async def on_skip(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
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
async def on_feedback_next(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
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
    st["test_levels_temp"] = dict(temp_levels)   # тимчасовий вибір рівнів
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
async def test_start(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    user = await store.get_user(uid)

    law_qids = qb.pick_random(qb.law, 50)

    modules = user.get("ok_modules", [])
    last_levels = user.get("ok_last_levels", {}) or {}

    ui = await store.get_ui(uid)
    pre = ui.get("state", {}) or {}
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
        text = (
            "📊 <b>Статистика</b>\n\n"
            f"Тестів (останні 50): <b>{s['count']}</b>\n"
            f"Середній результат: <b>{s['avg']:.1f}%</b>\n\n"
            f"Останній тест:\n"
            f"• {iso_to_dt(last['finished_at']).strftime('%d.%m.%Y %H:%M')}\n"
            f"• {last['correct']}/{last['total']} = {float(last['percent']):.1f}%"
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

    if t_end and n <= t_end:
        return "🟡"
    if inf or (s_end and n <= s_end):
        return "🟢"
    return "🔴"


def fmt_user_row(u: Dict[str, Any]) -> str:
    uid = u["user_id"]
    phone = u.get("phone") or "без номера"
    return f"{_admin_user_icon(u)} {phone} • {uid}"


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
    Гарантія: редагуємо ОДНЕ адмін-повідомлення.
    - якщо є message (callback) -> редагуємо його і запамʼятовуємо id
    - якщо message=None (ввід тексту) -> редагуємо збережений admin_panel_msg_id
    - якщо не вийшло -> створюємо 1 нове і запамʼятовуємо його id
    """
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}

    if message:
        st[ADMIN_PANEL_MSG_ID] = message.message_id
        st[ADMIN_PANEL_CHAT_ID] = message.chat.id
        await store.set_state(uid, st)

        # редагуємо саме це повідомлення
        try:
            await bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=message.message_id,
                text=text,
                reply_markup=kb,
                parse_mode="HTML",
            )
            return
        except TelegramBadRequest:
            # fallback нижче
            pass

    # message=None -> редагуємо останнє адмін меню зі state
    msg_id = st.get(ADMIN_PANEL_MSG_ID)
    chat_id = st.get(ADMIN_PANEL_CHAT_ID) or chat_id

    if msg_id:
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=text,
                reply_markup=kb,
                parse_mode="HTML",
            )
            return
        except TelegramBadRequest:
            pass

    # якщо редагувати нічого — створюємо 1 нове
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

    text = (
        "👤 <b>Користувач</b>\n\n"
        f"ID: <b>{target_id}</b>\n"
        f"Телефон: <b>{phone_html}</b>\n"
        f"{fmt_access_line(user)}"
    )

    b = InlineKeyboardBuilder()

    # одноразова підписка -> тільки "безкінечно"
    b.button(text="✅ Дати доступ (безкінечно)", callback_data=clamp_callback(f"admin:subinf:{target_id}:{back_offset}"))
    b.button(text="🚫 Забрати доступ", callback_data=clamp_callback(f"admin:subcancel:{target_id}:{back_offset}"))
    b.adjust(1)

    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data=clamp_callback(f"admin:users:{back_offset}")))

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
async def admin_users_search_input(message: Message, bot: Bot, store: "Storage", admin_ids: set[int]):
    uid = message.from_user.id
    if uid not in admin_ids:
        return

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    if st.get(ADMIN_USERS_AWAITING) != "admin_users_phone":
        return

    back_offset = int(st.get(ADMIN_USERS_BACK_OFFSET) or 0)

    digits = "".join(ch for ch in (message.text or "").strip() if ch.isdigit())

    # прибрати повідомлення з номером (щоб не світити)
    try:
        await message.delete()
    except Exception:
        pass

    if not digits:
        # не створюємо нових повідомлень — просто міняємо те саме меню з помилкою
        await render_admin_users_search_prompt(
            bot, store, uid, st.get(ADMIN_PANEL_CHAT_ID) or message.chat.id, back_offset,
            message=None, error="Введи хоча б одну цифру"
        )
        return

    st[ADMIN_USERS_QUERY] = digits
    st.pop(ADMIN_USERS_AWAITING, None)
    st.pop(ADMIN_USERS_BACK_OFFSET, None)
    await store.set_state(uid, st)

    # важливо: message=None -> редагуємо збережене адмін-меню
    await render_admin_users_list(
        bot, store, uid,
        st.get(ADMIN_PANEL_CHAT_ID) or message.chat.id,
        0,
        message=None
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

    # одноразова підписка -> доступ безкінечно
    await store.set_subscription(target_id, None, infinite=True)

    await render_admin_user_detail(bot, store, admin_uid, cb.message.chat.id, target_id, back_offset, message=cb.message)
    await cb.answer("Ок")


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

    questions_path = os.getenv("QUESTIONS_PATH", "questions_flat.json")
    if not os.path.exists(questions_path):
        raise RuntimeError(f"Questions file not found: {questions_path}")

    qb = QuestionBank(questions_path)
    qb.load()

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

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp["store"] = store
    dp["qb"] = qb
    dp["admin_ids"] = admin_ids

    # inject deps via handler args (aiogram resolves by name)
    dp.include_router(router)

    async def _inject_middleware(handler, event, data):
        data["store"] = store
        data["qb"] = qb
        data["admin_ids"] = admin_ids
        return await handler(event, data)

    dp.update.outer_middleware(_inject_middleware)

    print(f"Loaded: law={len(qb.law)} | ok_modules={len(qb.ok_modules)}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
