
import asyncio
import json
import logging
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Set, Tuple

import asyncpg
from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder


router = Router()

# -------------------------
# Конфіг
# -------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

ADMIN_IDS: Set[int] = set()
if os.getenv("ADMIN_IDS"):
    for x in os.getenv("ADMIN_IDS", "").split(","):
        x = x.strip()
        if x.isdigit():
            ADMIN_IDS.add(int(x))

TRAIN_QUESTIONS = int(os.getenv("TRAIN_QUESTIONS", "50"))
EXAM_QUESTIONS = int(os.getenv("EXAM_QUESTIONS", "100"))
EXAM_DURATION_MINUTES = int(os.getenv("EXAM_DURATION_MINUTES", "90"))

EXAM_LAW_QUESTIONS = int(os.getenv("EXAM_LAW_QUESTIONS", "50"))
EXAM_PER_TOPIC_QUESTIONS = int(os.getenv("EXAM_PER_TOPIC_QUESTIONS", "20"))


QUESTIONS_FILE = os.getenv("QUESTIONS_FILE", "questions_flat.json")
PROBLEMS_FILE = os.getenv("PROBLEMS_FILE", "problem_questions.json")

KYIV_TZ = ZoneInfo("Europe/Kyiv")
OK_CODE_LAW = "LAW"  # внутрішній код для "законодавства"
LEVEL_ALL = -1  # спеціальне значення: всі рівні для обраного ОК

PENDING_AFTER_OK: dict[int, str] = {}  # tg_id -> "train" | "exam"
REG_PROMPT_MSG_ID: dict[int, int] = {}  # tg_id -> message_id (реєстраційний текст)

POSITION_OK_MAP: Dict[str, Dict[str, int]] = {
    "Начальник відділу": {
        "ОК-4": 2,
        "ОК-10": 3,
        "ОК-14": 2,
        "ОК-15": 2,
    },
    "Головний державний інспектор": {
        "ОК-4": 2,
        "ОК-10": 3,
        "ОК-14": 2,
        "ОК-15": 2,
    },
    "Старший державний інспектор": {
        "ОК-4": 1,
        "ОК-10": 2,
        "ОК-14": 1,
        "ОК-15": 1,
    },
    "Державний інспектор": {
        "ОК-4": 1,
        "ОК-10": 2,
        "ОК-14": 1,
        "ОК-15": 1,
    },
}

POSITIONS: List[str] = list(POSITION_OK_MAP.keys())
POS_ID_BY_NAME: Dict[str, int] = {name: i for i, name in enumerate(POSITIONS)}
POS_NAME_BY_ID: Dict[int, str] = {i: name for name, i in POS_ID_BY_NAME.items()}

def pos_id(name: str) -> int:
    return POS_ID_BY_NAME.get(name, -1)

def pos_name(pid: int) -> str:
    return POS_NAME_BY_ID.get(pid, "")


# -------------------------
# Глобальні кеші (заповнюються на старті)
# -------------------------

DB_POOL: Optional[asyncpg.Pool] = None

QUESTIONS_BY_ID: Dict[int, Dict[str, Any]] = {}
VALID_QIDS: List[int] = []  # валідні (1 правильна відповідь) і не в problem файлі

# scope = (ok_code, level_int)
OK_CODES: List[str] = []
LEVELS_BY_OK: Dict[str, List[int]] = {}
TOPICS_BY_SCOPE: Dict[Tuple[str, int], List[str]] = {}
QIDS_BY_SCOPE: Dict[Tuple[str, int], List[int]] = {}
QIDS_BY_SCOPE_TOPIC: Dict[Tuple[str, int, str], List[int]] = {}

PROBLEM_IDS_FILE: Set[int] = set()
DISABLED_IDS_DB: Set[int] = set()



# -------------------------
# Допоміжні утиліти
# -------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def as_minutes_seconds(seconds: int) -> str:
    seconds = max(0, int(seconds))
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"

def is_question_valid(q: Dict[str, Any]) -> bool:
    """Валідне питання: 1 правильна відповідь і ≥2 варіанти."""
    try:
        choices = q.get("choices") or []
        correct = q.get("correct") or []
        if len(choices) < 2:
            return False
        if len(correct) != 1:
            return False
        ci = int(correct[0])
        if ci < 0 or ci >= len(choices):
            return False
        if not (q.get("question") or "").strip():
            return False
        return True
    except Exception:
        return False

def scope_title(ok_code: str, level: int | None = None) -> str:
    if ok_code == OK_CODE_LAW:
        return "📜 Законодавство"
    if level is None:
        return ok_code
    if int(level) == LEVEL_ALL:
        return f"{ok_code} • всі рівні"
    return f"{ok_code} • рівень {int(level)}"

def truncate_button(text: str, max_len: int = 44) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"

def normalize_ok_code(raw_ok: Any) -> str:
    # у файлі законодавство має ok=None
    return OK_CODE_LAW if raw_ok is None else str(raw_ok)

def normalize_level(raw_level: Any, ok_code: str) -> int:
    if ok_code == OK_CODE_LAW:
        return 0
    if raw_level is None:
        # на випадок неконсистентних даних
        return 1
    return int(raw_level)

def effective_qids(base: List[int]) -> List[int]:
    if not DISABLED_IDS_DB:
        return base
    return [qid for qid in base if qid not in DISABLED_IDS_DB]


def levels_for_ok(ok_code: str) -> List[int]:
    """Повертає доступні рівні для ОК. Для LAW — [0]."""
    if ok_code == OK_CODE_LAW:
        return [0]
    return LEVELS_BY_OK.get(ok_code, [])

def topics_for_scope(ok_code: str, level: int) -> List[str]:
    """Список тем для scope. Якщо level==LEVEL_ALL — об'єднує теми по всіх рівнях."""
    if level == LEVEL_ALL:
        s: Set[str] = set()
        for lvl in levels_for_ok(ok_code):
            s.update(TOPICS_BY_SCOPE.get((ok_code, lvl), []))
        return sorted(s)
    return TOPICS_BY_SCOPE.get((ok_code, level), [])

def base_qids_for_scope(ok_code: str, level: int) -> List[int]:
    """Базові qids для scope без фільтра disabled. Якщо level==LEVEL_ALL — об'єднує по всіх рівнях."""
    if level == LEVEL_ALL:
        out: List[int] = []
        for lvl in levels_for_ok(ok_code):
            out.extend(QIDS_BY_SCOPE.get((ok_code, lvl), []))
        return out
    return QIDS_BY_SCOPE.get((ok_code, level), [])

def base_qids_for_topic(ok_code: str, level: int, topic: str) -> List[int]:
    """Базові qids для теми без фільтра disabled. Якщо level==LEVEL_ALL — об'єднує по всіх рівнях."""
    if level == LEVEL_ALL:
        out: List[int] = []
        for lvl in levels_for_ok(ok_code):
            out.extend(QIDS_BY_SCOPE_TOPIC.get((ok_code, lvl, topic), []))
        return out
    return QIDS_BY_SCOPE_TOPIC.get((ok_code, level, topic), [])

def effective_topics(ok_code: str, level: int) -> List[str]:
    base = topics_for_scope(ok_code, level)
    if not base:
        return []
    out: List[str] = []
    for t in base:
        qids = base_qids_for_topic(ok_code, level, t)
        if any((qid not in DISABLED_IDS_DB) for qid in qids):
            out.append(t)
    return out


# -------------------------
# CallbackData
# -------------------------


MULTI_OK_CODE = "__MULTI_OK__"
MULTI_OK_LEVEL = 0

class MultiOkLevelsCb(CallbackData, prefix="mokl"):
    mode: str

class MultiOkLevelOpenCb(CallbackData, prefix="moko"):
    mode: str
    ok_code: str

class MultiOkLevelPickCb(CallbackData, prefix="mokp"):
    mode: str
    ok_code: str
    level: int

class MultiOkLevelsDoneCb(CallbackData, prefix="mokd"):
    mode: str


class MultiTopicsPageCb(CallbackData, prefix="mtp"):
    mode: str
    page: int

class MultiTopicToggleCb(CallbackData, prefix="mtt"):
    mode: str
    topic_idx: int
    page: int

class MultiTopicDoneCb(CallbackData, prefix="mtd"):
    mode: str

class MultiTopicClearCb(CallbackData, prefix="mtc"):
    mode: str
    page: int

class MultiTopicAllCb(CallbackData, prefix="mta"):
    mode: str


class AnswerCb(CallbackData, prefix="ans"):
    mode: str   # "train" | "exam"
    qid: int
    ci: int     # choice index

class SkipCb(CallbackData, prefix="sk"):
    qid: int

# продовжити після фідбеку (коли показали правильну відповідь)
class NextCb(CallbackData, prefix="nx"):
    mode: str   # "train" | "exam"
    expected_index: int  # який current_index очікуємо у сесії

class AdminToggleQCb(CallbackData, prefix="qt"):
    qid: int
    enable: int  # 1 enable, 0 disable

# вибір scope
class OkPickCb(CallbackData, prefix="ok"):
    ok_code: str

class OkPageCb(CallbackData, prefix="okp"):
    page: int

class OkMultiPageCb(CallbackData, prefix="okmp"):
    mode: str   # train | exam
    page: int

class OkToggleCb(CallbackData, prefix="okt"):
    mode: str
    ok_code: str
    page: int

class OkDoneCb(CallbackData, prefix="okd"):
    mode: str

class OkClearCb(CallbackData, prefix="okc"):
    mode: str
    page: int

class OkAllCb(CallbackData, prefix="oka"):
    mode: str

class StartMultiOkCb(CallbackData, prefix="stmok"):
    mode: str   # train | exam

class LevelPickCb(CallbackData, prefix="lvl"):
    ok_code: str
    level: int

# старт сесій / вибір тем
class StartScopeCb(CallbackData, prefix="st"):
    mode: str        # train/exam
    ok_code: str
    level: int

class TopicPageCb(CallbackData, prefix="tp"):
    mode: str
    ok_code: str
    level: int
    page: int

class TopicPickCb(CallbackData, prefix="tk"):
    mode: str
    ok_code: str
    level: int
    topic_idx: int

# multi-select topics
class TopicToggleCb(CallbackData, prefix="tt"):
    mode: str
    ok_code: str
    level: int
    topic_idx: int
    page: int

class TopicDoneCb(CallbackData, prefix="td"):
    mode: str
    ok_code: str
    level: int

class TopicClearCb(CallbackData, prefix="tc"):
    mode: str
    ok_code: str
    level: int
    page: int

class TopicAllCb(CallbackData, prefix="ta"):
    mode: str
    ok_code: str
    level: int

class TrainModeCb(CallbackData, prefix="tm"):
    mode: str   # train / exam
    kind: str   # position / manual

class PosMenuCb(CallbackData, prefix="pm"):
    mode: str      # 't' або 'e'
    pid: int       # position id
    action: str    # 'r' | 'b' | 'm'

class PosTopicPageCb(CallbackData, prefix="ptp"):
    mode: str
    pid: int
    page: int

class PosTopicToggleCb(CallbackData, prefix="ptt"):
    mode: str
    pid: int
    topic_idx: int
    page: int

class PosTopicDoneCb(CallbackData, prefix="ptd"):
    mode: str
    pid: int

class PosTopicClearCb(CallbackData, prefix="ptc"):
    mode: str
    pid: int
    page: int

class PosTopicAllCb(CallbackData, prefix="pta"):
    mode: str
    pid: int

class TopicBackCb(CallbackData, prefix="tbk"):
    mode: str
    ok_code: str
    level: int

class TrainVariantCb(CallbackData, prefix="tvar"):
    # kind: "scope" | "topics" | "multi"
    kind: str
    ok_code: str
    level: int
    # variant: "all" | "rand"
    variant: str

class TrainVariantBackCb(CallbackData, prefix="tback"):
    kind: str
    ok_code: str
    level: int


# -------------------------
# Клавіатури
# -------------------------

from typing import Optional

def multi_topics_for_ok_set(
    ok_codes: Set[str],
    ok_levels: Optional[Dict[str, int]] = None,
    *,
    include_missing_as_all: bool = False,
) -> List[str]:
    """
    Повертає список "лейблів" тем для multi-OK.

    - Законодавство: по topic (без рівнів), префікс "📜 "
    - Інші ОК: по вибраному рівню, формат: "{OK} • рівень {lvl} • {topic}"

    Якщо для ОК рівень не заданий:
      - за замовчуванням ОК пропускається
      - якщо include_missing_as_all=True -> використовується LEVEL_ALL
    """
    ok_levels = ok_levels or {}
    out: List[str] = []

    ordered = sorted(ok_codes, key=lambda x: (x != OK_CODE_LAW, x))  # LAW першим
    for ok in ordered:
        if ok == OK_CODE_LAW:
            law_topics = effective_topics(OK_CODE_LAW, 0)
            if not law_topics:
                out.append("📜 Законодавство")
            else:
                for t in law_topics:
                    out.append(f"📜 {t}")
            continue

        lvl = ok_levels.get(ok)
        if lvl is None:
            if include_missing_as_all:
                lvl = LEVEL_ALL
            else:
                continue

        for t in effective_topics(ok, int(lvl)):
            out.append(f"{ok} • рівень {int(lvl)} • {t}")

    return out



def qids_for_multi_topic_label(label: str) -> List[int]:
    if label.startswith("📜 "):
        topic = label[2:].strip()
        if topic == "Законодавство":
            return base_qids_for_scope(OK_CODE_LAW, 0)
        return base_qids_for_topic(OK_CODE_LAW, 0, topic)

    parts = label.split(" • ")
    if len(parts) == 3 and parts[1].startswith("рівень "):
        ok_code = parts[0].strip()
        try:
            lvl = int(parts[1].replace("рівень", "").strip())
        except Exception:
            return []
        topic = parts[2].strip()
        return base_qids_for_topic(ok_code, lvl, topic)

    if len(parts) == 2:
        ok_code, topic = parts[0].strip(), parts[1].strip()
        return base_qids_for_topic(ok_code, LEVEL_ALL, topic)

    return []

def _missing_multi_levels(ok_codes: Set[str], ok_levels: Dict[str, int]) -> List[str]:
    return [ok for ok in ok_codes if ok != OK_CODE_LAW and ok_levels.get(ok) is None]

def kb_multi_levels_overview(mode: str, ok_codes: Set[str], ok_levels: Dict[str, int]) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    ordered = sorted(ok_codes, key=lambda x: (x != OK_CODE_LAW, x))

    for ok in ordered:
        if ok == OK_CODE_LAW:
            b.row(InlineKeyboardButton(text="📜 Законодавство (за законами)", callback_data="noop"))
            continue

        lvl = ok_levels.get(ok)
        txt = f"🎚 {ok}: рівень {lvl}" if lvl is not None else f"🎚 {ok}: оберіть рівень"
        b.row(InlineKeyboardButton(text=txt, callback_data=MultiOkLevelOpenCb(mode=mode, ok_code=ok).pack()))

    b.row(
        InlineKeyboardButton(text="🔁 Модулі", callback_data=OkMultiPageCb(mode=mode, page=0).pack()),
        InlineKeyboardButton(text="🏠 Меню", callback_data="menu"),
    )

    if mode == "train":
        b.row(InlineKeyboardButton(text="📚 Далі: теми", callback_data=MultiOkLevelsDoneCb(mode=mode).pack()))
    else:
        b.row(InlineKeyboardButton(text="✅ Почати екзамен", callback_data=StartMultiOkCb(mode=mode).pack()))

    return b.as_markup()


def kb_multi_pick_level(mode: str, ok_code: str, current_level: Optional[int]) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for lvl in levels_for_ok(ok_code):
        mark = "✅" if current_level == lvl else "▫️"
        b.button(
            text=f"{mark} Рівень {lvl}",
            callback_data=MultiOkLevelPickCb(mode=mode, ok_code=ok_code, level=int(lvl)).pack(),
        )
    b.adjust(1)
    b.row(
        InlineKeyboardButton(text="⬅️ Назад", callback_data=MultiOkLevelsCb(mode=mode).pack()),
        InlineKeyboardButton(text="🏠 Меню", callback_data="menu"),
    )
    return b.as_markup()


def kb_multi_topics(
        mode: str,
        ok_codes: Set[str],
        page: int = 0,
        selected: Optional[Set[str]] = None,
        per_page: int = 8,
) -> InlineKeyboardMarkup:
    selected_set: Set[str] = set(selected or [])
    topics = multi_topics_for_ok_set(ok_codes)

    pages: List[List[str]] = [topics[i:i + per_page] for i in range(0, len(topics), per_page)]
    if not pages:
        pages = [[]]
    page = max(0, min(int(page), len(pages) - 1))
    current = pages[page]
    start_idx = page * per_page

    b = InlineKeyboardBuilder()

    for i, label in enumerate(current):
        idx = start_idx + i
        checked = "☑️" if label in selected_set else "⬜️"
        btn_text = truncate_button(f"{checked} {label}", max_len=44)
        b.row(
            InlineKeyboardButton(
                text=btn_text,
                callback_data=MultiTopicToggleCb(mode=mode, topic_idx=idx, page=page).pack(),
            )
        )

    nav: List[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="⬅️", callback_data=MultiTopicsPageCb(mode=mode, page=page - 1).pack()))
    if page < len(pages) - 1:
        nav.append(InlineKeyboardButton(text="➡️", callback_data=MultiTopicsPageCb(mode=mode, page=page + 1).pack()))
    if nav:
        b.row(*nav)

    start_label = f"✅ Почати ({len(selected_set)})" if selected_set else "✅ Почати"

    # Рядок: [ "Змінити модулі ", "Меню" ]
    b.row(
        InlineKeyboardButton(text="🔁 Змінити модулі", callback_data=OkMultiPageCb(mode=mode, page=0).pack()),
        InlineKeyboardButton(text="🏠 Меню", callback_data="menu"),
    )

    # Рядок: [ "Почати" ]
    b.row(
        InlineKeyboardButton(text=start_label, callback_data=MultiTopicDoneCb(mode=mode).pack()),
    )

    return b.as_markup()



@router.callback_query(MultiTopicsPageCb.filter())
async def multi_topics_page(call: CallbackQuery, callback_data: MultiTopicsPageCb) -> None:
    if not DB_POOL:
        return
    tg_id = call.from_user.id
    mode = str(callback_data.mode)
    page = int(callback_data.page)

    ok_codes = await db_get_ok_prefs(DB_POOL, tg_id, mode)
    ok_codes = {c for c in ok_codes if c}
    if not ok_codes:
        await call.answer("Оберіть модулі (ОК) спочатку", show_alert=True)
        return

    available = set(multi_topics_for_ok_set(ok_codes))
    selected = await db_get_topic_prefs(DB_POOL, tg_id, mode, MULTI_OK_CODE, MULTI_OK_LEVEL)
    selected = {t for t in selected if t in available}
    await db_set_topic_prefs(DB_POOL, tg_id, mode, MULTI_OK_CODE, MULTI_OK_LEVEL, selected)

    shown = ", ".join(sorted(ok_codes))
    await safe_edit(
        call,
        f"Обрані модулі: <b>{html_escape(shown)}</b>\n"
        f"Оберіть теми для тренування:\n"
        f"Обрано тем: <b>{len(selected)}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_multi_topics(mode, ok_codes, page=page, selected=selected),
    )
    await call.answer()

@router.callback_query(MultiTopicToggleCb.filter())
async def multi_topic_toggle(call: CallbackQuery, callback_data: MultiTopicToggleCb) -> None:
    if not DB_POOL:
        return
    tg_id = call.from_user.id
    mode = str(callback_data.mode)
    idx = int(callback_data.topic_idx)
    page = int(callback_data.page)

    ok_codes = await db_get_ok_prefs(DB_POOL, tg_id, mode)
    ok_codes = {c for c in ok_codes if c}
    if not ok_codes:
        await call.answer("Оберіть модулі (ОК) спочатку", show_alert=True)
        return

    topics = multi_topics_for_ok_set(ok_codes)
    if idx < 0 or idx >= len(topics):
        await call.answer()
        return

    selected = await db_get_topic_prefs(DB_POOL, tg_id, mode, MULTI_OK_CODE, MULTI_OK_LEVEL)
    t = topics[idx]
    if t in selected:
        selected.remove(t)
    else:
        selected.add(t)

    await db_set_topic_prefs(DB_POOL, tg_id, mode, MULTI_OK_CODE, MULTI_OK_LEVEL, selected)

    shown = ", ".join(sorted(ok_codes))
    await safe_edit(
        call,
        f"Обрані модулі: <b>{html_escape(shown)}</b>\n"
        f"Оберіть теми для тренування:\n"
        f"Обрано тем: <b>{len(selected)}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_multi_topics(mode, ok_codes, page=page, selected=selected),
    )
    await call.answer()

@router.callback_query(MultiTopicClearCb.filter())
async def multi_topic_clear(call: CallbackQuery, callback_data: MultiTopicClearCb) -> None:
    if not DB_POOL:
        return
    tg_id = call.from_user.id
    mode = str(callback_data.mode)
    page = int(callback_data.page)

    ok_codes = await db_get_ok_prefs(DB_POOL, tg_id, mode)
    ok_codes = {c for c in ok_codes if c}
    if not ok_codes:
        await call.answer("Оберіть модулі (ОК) спочатку", show_alert=True)
        return

    await db_clear_topic_prefs(DB_POOL, tg_id, mode, MULTI_OK_CODE, MULTI_OK_LEVEL)

    shown = ", ".join(sorted(ok_codes))
    await safe_edit(
        call,
        f"Обрані модулі: <b>{html_escape(shown)}</b>\n"
        f"Оберіть теми для тренування:\n"
        f"Обрано тем: <b>0</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_multi_topics(mode, ok_codes, page=page, selected=set()),
    )
    await call.answer()

@router.callback_query(MultiTopicAllCb.filter())
async def multi_topic_all(call: CallbackQuery, callback_data: MultiTopicAllCb) -> None:
    if not DB_POOL:
        return
    tg_id = call.from_user.id
    mode = str(callback_data.mode)

    ok_codes = await db_get_ok_prefs(DB_POOL, tg_id, mode)
    ok_codes = {c for c in ok_codes if c}
    if not ok_codes:
        await call.answer("Оберіть модулі (ОК) спочатку", show_alert=True)
        return

    all_topics = set(multi_topics_for_ok_set(ok_codes))
    await db_set_topic_prefs(DB_POOL, tg_id, mode, MULTI_OK_CODE, MULTI_OK_LEVEL, all_topics)

    shown = ", ".join(sorted(ok_codes))
    await safe_edit(
        call,
        f"Обрані модулі: <b>{html_escape(shown)}</b>\n"
        f"Оберіть теми для тренування:\n"
        f"Обрано тем: <b>{len(all_topics)}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_multi_topics(mode, ok_codes, page=0, selected=all_topics),
    )
    await call.answer()

@router.callback_query(MultiTopicDoneCb.filter())
async def multi_topic_done(call: CallbackQuery, callback_data: MultiTopicDoneCb) -> None:
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    mode = str(callback_data.mode)

    user = await db_get_user(DB_POOL, tg_id)
    if not user:
        await call.answer("Немає профілю", show_alert=True)
        return

    if not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    ok_codes = await db_get_ok_prefs(DB_POOL, tg_id, mode)
    ok_codes = {c for c in ok_codes if c}
    if not ok_codes:
        await call.answer("Оберіть модулі (ОК) спочатку", show_alert=True)
        return

    available = set(multi_topics_for_ok_set(ok_codes))
    selected = await db_get_topic_prefs(
        DB_POOL,
        tg_id,
        mode,
        MULTI_OK_CODE,
        MULTI_OK_LEVEL,
    )
    selected = {t for t in selected if t in available}
    if not selected:
        await call.answer("Оберіть хоча б одну тему", show_alert=True)
        return

    pool: list[int] = []
    for label in selected:
        pool.extend(qids_for_multi_topic_label(label))

    pool_qids = effective_qids(list(dict.fromkeys(pool)))
    if not pool_qids:
        await call.answer("У вибраних темах немає питань", show_alert=True)
        return

    await call.answer()

    # прибираємо клавіатуру
    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    await start_session_for_pool(
        call.bot,
        tg_id,
        call.message.chat.id,
        user,
        mode,
        pool_qids,
        edit_message=call.message,
    )

    if mode == "train":
        pool_size = len(pool_qids)
        title = (
            "Навчання • <b>декілька модулів</b>\n"
            f"Обрано тем: <b>{len(selected)}</b>\n"
            "Як сформувати питання?"
        )
        await call.message.edit_text(
            title,
            parse_mode=ParseMode.HTML,
            reply_markup=kb_train_question_set(
                "multi",
                MULTI_OK_CODE,
                MULTI_OK_LEVEL,
                pool_size,
            ),
        )
        return


MAIN_MENU_TEXT = (
    "<b>Оберіть режим</b> 👇\n\n"
    "📚 <b>Навчання</b> — тренування без таймера\n"
    "📝 <b>Екзамен</b> — режим з таймером\n\n"
    "Натисніть потрібну кнопку нижче:"
)

async def show_main_menu(message: Message, *, is_admin: bool) -> None:
    await message.answer(
        MAIN_MENU_TEXT,
        reply_markup=kb_main_menu(is_admin=is_admin),
        parse_mode="HTML",
    )

@router.callback_query(TopicBackCb.filter())
async def topic_back(call: CallbackQuery, callback_data: TopicBackCb) -> None:
    mode = str(callback_data.mode)
    ok_code = str(callback_data.ok_code)
    lvl = int(callback_data.level)

    if mode == "train":
        await safe_edit(
            call,
            f"Навчання для: <b>{html_escape(scope_title(ok_code, lvl))}</b>\nОберіть варіант:",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_train_pick(ok_code, lvl),
        )
    else:
        await safe_edit(
            call,
            f"Екзамен для: <b>{html_escape(scope_title(ok_code, lvl))}</b>\nОберіть варіант:",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_exam_pick(ok_code, lvl),
        )

    await call.answer()


async def safe_edit(
    call,
    text: str,
    *,
    reply_markup=None,
    parse_mode: str | None = None,
) -> None:
    """
    1) пробуємо edit_text
    2) якщо не можна (старе повідомлення/той самий текст) — пробуємо edit_reply_markup
    3) якщо зовсім ніяк — fallback на answer (рідко)
    """
    try:
        await call.message.edit_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
        return
    except Exception:
        pass

    if reply_markup is not None:
        try:
            await call.message.edit_reply_markup(reply_markup=reply_markup)
            return
        except Exception:
            pass

    # останній шанс (небажано, але краще ніж “зависнути”)
    await call.message.answer(text, reply_markup=reply_markup, parse_mode=parse_mode)


def kb_request_contact() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📞 Поділитись номером", request_contact=True)]],
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder="Натисніть кнопку нижче, щоб поділитись номером",
    )

def kb_main_menu(is_admin: bool = False) -> InlineKeyboardMarkup:
    """Головне меню (inline) у форматі 2 кнопки в ряд — як у зразку."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📚 Навчання", callback_data="mm:train"),
                InlineKeyboardButton(text="📝 Екзамен", callback_data="mm:exam"),
            ],
            [
                InlineKeyboardButton(text="📊 Статистика", callback_data="mm:stats"),
                InlineKeyboardButton(text="ℹ️ Доступ", callback_data="mm:access"),
            ],
            [

                InlineKeyboardButton(
                    text=("🛠 Адмін" if is_admin else "🏠 Меню"),
                    callback_data=("mm:admin" if is_admin else "menu"),
                ),
            ],
        ]
    )


    # якщо не адмін — прибираємо заглушку (щоб не було "порожньої" кнопки)
    if not is_admin:
        rows[-1] = [InlineKeyboardButton(text="⚙️ Налаштування", callback_data="mm:settings")]

    return InlineKeyboardMarkup(inline_keyboard=rows)

def kb_admin_panel() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.row(
        InlineKeyboardButton(text="👥 Користувачі", callback_data="ad:users"),
        InlineKeyboardButton(text="⚠️ Проблемні питання", callback_data="ad:problems"),
    )
    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="menu"))
    return b.as_markup()

def kb_question(mode: str, qid: int, choices: List[str], allow_skip: bool) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for i, _ in enumerate(choices):
        label = letters[i] if i < len(letters) else str(i + 1)
        b.button(text=label, callback_data=AnswerCb(mode=mode, qid=qid, ci=i))

    # A B C D в один рядок
    b.adjust(4)

    # нижній ряд: Пропустити + Меню (в одному рядку)
    bottom: List[InlineKeyboardButton] = []
    if allow_skip:
        bottom.append(InlineKeyboardButton(text="⏭ Пропустити", callback_data=SkipCb(qid=qid).pack()))
    bottom.append(InlineKeyboardButton(text="🏠 Меню", callback_data="menu"))

    b.row(*bottom)
    return b.as_markup()


def kb_after_feedback(mode: str, expected_index: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="✅ Зрозуміло / Далі", callback_data=NextCb(mode=mode, expected_index=expected_index))
    b.button(text="🏠 Меню", callback_data="menu")
    b.adjust(1)
    return b.as_markup()

def kb_pick_ok(page: int = 0, per_page: int = 9) -> InlineKeyboardMarkup:
    codes = [OK_CODE_LAW] + [c for c in OK_CODES if c != OK_CODE_LAW]
    pages: List[List[str]] = [codes[i:i+per_page] for i in range(0, len(codes), per_page)]
    if not pages:
        pages = [[]]
    page = max(0, min(page, len(pages) - 1))
    current = pages[page]

    b = InlineKeyboardBuilder()
    for c in current:
        label = "📜 Законодавство" if c == OK_CODE_LAW else c
        b.button(text=label, callback_data=OkPickCb(ok_code=c).pack())
    b.adjust(1)

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="⬅️", callback_data=OkPageCb(page=page-1).pack()))
    if page < len(pages) - 1:
        nav.append(InlineKeyboardButton(text="➡️", callback_data=OkPageCb(page=page+1).pack()))
    if nav:
        b.row(*nav)

    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="menu"))
    return b.as_markup()


def kb_pick_ok_multi(
        mode: str,
        page: int = 0,
        *,
        selected: Optional[Set[str]] = None,
        per_page: int = 18,  # Збільшуємо, щоб показати всі ОК
) -> InlineKeyboardMarkup:
    selected_set: Set[str] = set(selected or [])

    # ✅ СПОЧАТКУ: Законодавство окремо
    # ✅ ПОТІМ: Всі ОК відсортовані від ОК-1 до ОК-17

    # Створюємо список з усіма ОК крім законодавства
    all_codes = []
    for c in OK_CODES:
        if c != OK_CODE_LAW:
            all_codes.append(c)

    # ✅ Сортуємо ОК за номером (від 1 до 17)
    def get_ok_number(code: str) -> int:
        try:
            if code.startswith("ОК-"):
                return int(code.split("-")[1])
            return 999  # якщо не вдалось витягти номер
        except:
            return 999

    # Сортуємо за номером
    all_codes_sorted = sorted(all_codes, key=get_ok_number)

    # Тепер додаємо законодавство першим, потім всі відсортовані ОК
    codes = [OK_CODE_LAW] + all_codes_sorted

    # Розділяємо на дві колонки
    # Перша колонка: законодавство + половина ОК
    # Друга колонка: друга половина ОК

    half_len = (len(all_codes_sorted) + 1) // 2  # +1 для законодавства
    first_column = codes[:half_len]
    second_column = codes[half_len:]

    b = InlineKeyboardBuilder()

    # ✅ Додаємо кнопки в 2 колонки
    max_rows = max(len(first_column), len(second_column))

    for i in range(max_rows):
        row_buttons = []

        # Перша колонка
        if i < len(first_column):
            c = first_column[i]
            if c == OK_CODE_LAW:
                label = "📜 Законодавство"
            else:
                label = c
            mark = "☑️" if c in selected_set else "⬜️"
            row_buttons.append(
                InlineKeyboardButton(
                    text=f"{mark} {label}",
                    callback_data=OkToggleCb(mode=mode, ok_code=c, page=page).pack(),
                )
            )
        else:
            # Пуста кнопка для вирівнювання
            row_buttons.append(
                InlineKeyboardButton(text=" ", callback_data="noop")
            )

        # Друга колонка
        if i < len(second_column):
            c = second_column[i]
            label = c
            mark = "☑️" if c in selected_set else "⬜️"
            row_buttons.append(
                InlineKeyboardButton(
                    text=f"{mark} {label}",
                    callback_data=OkToggleCb(mode=mode, ok_code=c, page=page).pack(),
                )
            )
        else:
            # Пуста кнопка для вирівнювання
            row_buttons.append(
                InlineKeyboardButton(text=" ", callback_data="noop")
            )

        b.row(*row_buttons)

    # ✅ Прибираємо "Всі ОК" і "Очистити", залишаємо тільки "Готово" і "Меню"
    b.row(
        InlineKeyboardButton(text="✅ Готово", callback_data=OkDoneCb(mode=mode).pack()),
        InlineKeyboardButton(text="🏠 Меню", callback_data="menu"),
    )

    return b.as_markup()

@router.callback_query(F.data == "noop")
async def noop_callback(call: CallbackQuery) -> None:
    """Обробник для порожніх кнопок (заглушок)."""
    await call.answer()


def kb_train_pick_multi(mode: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()

    if mode == "train":
        b.button(
            text="📚 Обрати теми",
            callback_data=MultiTopicsPageCb(mode=mode, page=0).pack(),
        )
        b.button(
            text="🔁 Змінити модулі",
            callback_data=OkMultiPageCb(mode=mode, page=0).pack(),
        )
    else:  # exam
        b.button(
            text="✅ Почати екзамен",
            callback_data=StartMultiOkCb(mode=mode).pack(),
        )
        b.button(
            text="🔁 Змінити модулі",
            callback_data=OkMultiPageCb(mode=mode, page=0).pack(),
        )

    b.button(text="🏠 Меню", callback_data="menu")
    b.adjust(1)
    return b.as_markup()


def kb_pick_level(ok_code: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for lvl in sorted(LEVELS_BY_OK.get(ok_code, [1, 2, 3])):
        if ok_code == OK_CODE_LAW:
            # для законодавства рівня нема
            continue
        b.button(text=f"Рівень {lvl}", callback_data=LevelPickCb(ok_code=ok_code, level=lvl).pack())
    b.adjust(1)
    b.row(InlineKeyboardButton(text="🔁 Змінити ОК", callback_data=OkPageCb(page=0).pack()))
    return b.as_markup()

def kb_train_mode(mode: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(
        text="📚 Обрати теми вручну",
        callback_data=TrainModeCb(mode=mode, kind="manual").pack()
    )
    b.button(text="🏠 Меню", callback_data="menu")
    b.adjust(1)
    return b.as_markup()


def kb_train_question_set(kind: str, ok_code: str, level: int, pool_size: int) -> InlineKeyboardMarkup:
    pool_size = int(pool_size or 0)
    rand_n = min(TRAIN_QUESTIONS, pool_size) if pool_size > 0 else 0

    b = InlineKeyboardBuilder()
    b.row(
        InlineKeyboardButton(
            text=f"📋 Всі питання ({pool_size})",
            callback_data=TrainVariantCb(kind=kind, ok_code=ok_code, level=level, variant="all").pack(),
        )
    )
    b.row(
        InlineKeyboardButton(
            text=f"🎲 Рандомні ({rand_n})",
            callback_data=TrainVariantCb(kind=kind, ok_code=ok_code, level=level, variant="rand").pack(),
        )
    )
    b.row(
        InlineKeyboardButton(
            text="⬅️ Назад",
            callback_data=TrainVariantBackCb(kind=kind, ok_code=ok_code, level=level).pack(),
        )
    )
    b.row(InlineKeyboardButton(text="🏠 Меню", callback_data="menu"))
    return b.as_markup()


def kb_train_pick(ok_code: str, level: int | None) -> InlineKeyboardMarkup:
    if level is None:
        level = 0 if ok_code == OK_CODE_LAW else LEVEL_ALL
    else:
        level = int(level)

    b = InlineKeyboardBuilder()
    b.button(
        text="✅ Почати тренування",
        callback_data=StartScopeCb(mode="train", ok_code=ok_code, level=level).pack(),
    )
    b.button(
        text="📚 Тренування по блоку",
        callback_data=TopicPageCb(mode="train", ok_code=ok_code, level=level, page=0).pack(),
    )
    b.button(text="🏠 Меню", callback_data="menu")
    b.adjust(1)
    return b.as_markup()


def kb_exam_pick(ok_code: str, level: int | None) -> InlineKeyboardMarkup:
    if level is None:
        level = 0 if ok_code == OK_CODE_LAW else LEVEL_ALL
    else:
        level = int(level)

    b = InlineKeyboardBuilder()
    b.button(
        text=f"✅ Почати екзамен ({EXAM_QUESTIONS})",
        callback_data=StartScopeCb(mode="exam", ok_code=ok_code, level=level).pack(),
    )
    b.button(
        text="📚 Екзамен по блоку",
        callback_data=TopicPageCb(mode="exam", ok_code=ok_code, level=level, page=0).pack(),
    )
    b.button(text="🏠 Меню", callback_data="menu")
    b.adjust(1)
    return b.as_markup()

def kb_topics(
    mode: str,
    ok_code: str,
    level: int,
    page: int = 0,
    selected: Optional[Set[str]] = None,
    per_page: int = 8,
) -> InlineKeyboardMarkup:
    selected_set: Set[str] = set(selected or [])
    topics = effective_topics(ok_code, level)

    pages: List[List[str]] = [topics[i:i + per_page] for i in range(0, len(topics), per_page)]
    if not pages:
        pages = [[]]
    page = max(0, min(page, len(pages) - 1))
    current = pages[page]
    start_idx = page * per_page

    b = InlineKeyboardBuilder()

    for i, t in enumerate(current):
        idx = start_idx + i
        checked = "☑️" if t in selected_set else "⬜️"
        label = f"{checked} {truncate_button(t, max_len=40)}"
        b.row(
            InlineKeyboardButton(
                text=label,
                callback_data=TopicToggleCb(
                    mode=mode,
                    ok_code=ok_code,
                    level=level,
                    topic_idx=idx,
                    page=page,
                ).pack(),
            )
        )

    nav = []
    if page > 0:
        nav.append(
            InlineKeyboardButton(
                text="⬅️",
                callback_data=TopicPageCb(mode=mode, ok_code=ok_code, level=level, page=page - 1).pack(),
            )
        )
    if page < len(pages) - 1:
        nav.append(
            InlineKeyboardButton(
                text="➡️",
                callback_data=TopicPageCb(mode=mode, ok_code=ok_code, level=level, page=page + 1).pack(),
            )
        )
    if nav:
        b.row(*nav)

    start_label = f"✅ Почати ({len(selected_set)})" if selected_set else "✅ Почати"

    b.row(
        InlineKeyboardButton(
            text="⬅️ Назад",
            callback_data=TopicBackCb(mode=mode, ok_code=ok_code, level=level).pack(),
        ),
        InlineKeyboardButton(
            text="🎯 Всі блоки",
            callback_data=TopicAllCb(mode=mode, ok_code=ok_code, level=level).pack(),
        ),
        InlineKeyboardButton(
            text=start_label,
            callback_data=TopicDoneCb(mode=mode, ok_code=ok_code, level=level).pack(),
        ),
        InlineKeyboardButton(text="🏠 Меню", callback_data="menu"),
    )

    return b.as_markup()





# -------------------------
# База даних
# -------------------------

DDL_CREATE = """
CREATE TABLE IF NOT EXISTS users (
  tg_id BIGINT PRIMARY KEY,
  phone TEXT,
  created_at TIMESTAMPTZ NOT NULL,
  trial_until TIMESTAMPTZ,
  sub_until TIMESTAMPTZ,
  is_admin BOOLEAN NOT NULL DEFAULT FALSE,
  last_seen TIMESTAMPTZ NOT NULL,
  ok_code TEXT,
  ok_level INT,
  train_mode TEXT,
  position TEXT
);

CREATE TABLE IF NOT EXISTS stats (
  tg_id BIGINT NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
  mode TEXT NOT NULL,
  answered INT NOT NULL DEFAULT 0,
  correct INT NOT NULL DEFAULT 0,
  wrong INT NOT NULL DEFAULT 0,
  skipped INT NOT NULL DEFAULT 0,
  PRIMARY KEY (tg_id, mode)
);

CREATE TABLE IF NOT EXISTS sessions (
  session_id UUID PRIMARY KEY,
  tg_id BIGINT NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
  mode TEXT NOT NULL,
  question_ids JSONB NOT NULL,
  current_index INT NOT NULL DEFAULT 0,
  correct_count INT NOT NULL DEFAULT 0,
  wrong_count INT NOT NULL DEFAULT 0,
  skipped_count INT NOT NULL DEFAULT 0,
  started_at TIMESTAMPTZ NOT NULL,
  expires_at TIMESTAMPTZ,
  completed BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS question_flags (
  question_id INT PRIMARY KEY,
  is_disabled BOOLEAN NOT NULL DEFAULT FALSE,
  note TEXT,
  updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS topic_prefs (
  tg_id BIGINT NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
  mode TEXT NOT NULL,
  ok_code TEXT NOT NULL,
  ok_level INT NOT NULL,
  topics JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (tg_id, mode, ok_code, ok_level)
);

CREATE TABLE IF NOT EXISTS ok_prefs (
  tg_id BIGINT NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
  mode TEXT NOT NULL,
  ok_codes JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (tg_id, mode)
);

CREATE TABLE IF NOT EXISTS ok_level_prefs (
  tg_id BIGINT NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
  mode TEXT NOT NULL,
  ok_levels JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (tg_id, mode)
);


"""

DDL_MIGRATIONS = [
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS ok_code TEXT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS ok_level INT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS train_mode TEXT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS position TEXT",
]


async def db_init(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(DDL_CREATE)
        for stmt in DDL_MIGRATIONS:
            await conn.execute(stmt)

async def db_get_user(pool: asyncpg.Pool, tg_id: int) -> Optional[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetchrow("SELECT * FROM users WHERE tg_id=$1", tg_id)

async def db_touch_user(pool: asyncpg.Pool, tg_id: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute("UPDATE users SET last_seen=$2 WHERE tg_id=$1", tg_id, utcnow())

async def db_upsert_user(pool: asyncpg.Pool, tg_id: int, phone: Optional[str], is_admin: bool) -> asyncpg.Record:
    now = utcnow()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT * FROM users WHERE tg_id=$1", tg_id)
        if existing is None:
            trial_until = now + timedelta(days=3)
            await conn.execute(
                """
                INSERT INTO users(tg_id, phone, created_at, trial_until, sub_until, is_admin, last_seen, ok_code, ok_level)
                VALUES($1, $2, $3, $4, NULL, $5, $3, NULL, NULL)
                """,
                tg_id, phone, now, trial_until, is_admin
            )
        else:
            await conn.execute(
                """
                UPDATE users
                SET phone = COALESCE($2, phone),
                    is_admin = (is_admin OR $3),
                    last_seen = $4
                WHERE tg_id=$1
                """,
                tg_id, phone, is_admin, now
            )
        return await conn.fetchrow("SELECT * FROM users WHERE tg_id=$1", tg_id)

async def db_set_position(pool: asyncpg.Pool, tg_id: int, position: Optional[str]) -> asyncpg.Record:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET position=$2 WHERE tg_id=$1",
            tg_id,
            position,
        )
        return await conn.fetchrow(
            "SELECT * FROM users WHERE tg_id=$1",
            tg_id,
        )


async def db_set_scope(
    pool: asyncpg.Pool,
    tg_id: int,
    ok_code: str,
    ok_level: Optional[int] = None
) -> asyncpg.Record:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET ok_code=$2, ok_level=$3 WHERE tg_id=$1",
            tg_id, ok_code, ok_level
        )
        return await conn.fetchrow(
            "SELECT * FROM users WHERE tg_id=$1",
            tg_id
        )

async def db_has_access(user: asyncpg.Record) -> bool:
    now = utcnow()
    if user["is_admin"]:
        return True
    tu = user["trial_until"]
    su = user["sub_until"]
    if tu and tu > now:
        return True
    if su and su > now:
        return True
    return False

async def db_stats_add(pool: asyncpg.Pool, tg_id: int, mode: str, answered=0, correct=0, wrong=0, skipped=0) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO stats(tg_id, mode, answered, correct, wrong, skipped)
            VALUES($1,$2,$3,$4,$5,$6)
            ON CONFLICT (tg_id, mode)
            DO UPDATE SET
              answered = stats.answered + EXCLUDED.answered,
              correct  = stats.correct  + EXCLUDED.correct,
              wrong    = stats.wrong    + EXCLUDED.wrong,
              skipped  = stats.skipped  + EXCLUDED.skipped
            """,
            tg_id, mode, answered, correct, wrong, skipped
        )

async def db_stats_get(pool: asyncpg.Pool, tg_id: int) -> List[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetch("SELECT * FROM stats WHERE tg_id=$1 ORDER BY mode", tg_id)

async def db_create_session(pool: asyncpg.Pool, tg_id: int, mode: str, qids: List[int], expires_at: Optional[datetime]) -> uuid.UUID:
    sid = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute("UPDATE sessions SET completed=TRUE WHERE tg_id=$1 AND mode=$2 AND completed=FALSE", tg_id, mode)
        await conn.execute(
            """
            INSERT INTO sessions(session_id, tg_id, mode, question_ids, current_index, correct_count, wrong_count, skipped_count, started_at, expires_at, completed)
            VALUES($1,$2,$3,$4,0,0,0,0,$5,$6,FALSE)
            """,
            sid, tg_id, mode, json.dumps(qids), utcnow(), expires_at
        )
    return sid

async def db_get_active_session(pool: asyncpg.Pool, tg_id: int, mode: str) -> Optional[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            """
            SELECT * FROM sessions
            WHERE tg_id=$1 AND mode=$2 AND completed=FALSE
            ORDER BY started_at DESC
            LIMIT 1
            """,
            tg_id, mode
        )

async def db_update_session_progress(
    pool: asyncpg.Pool,
    session_id: uuid.UUID,
    current_index: int,
    correct_delta: int = 0,
    wrong_delta: int = 0,
    skipped_delta: int = 0,
    completed: Optional[bool] = None,
) -> None:
    async with pool.acquire() as conn:
        if completed is None:
            await conn.execute(
                """
                UPDATE sessions
                SET current_index=$2,
                    correct_count=correct_count+$3,
                    wrong_count=wrong_count+$4,
                    skipped_count=skipped_count+$5
                WHERE session_id=$1
                """,
                session_id, current_index, correct_delta, wrong_delta, skipped_delta
            )
        else:
            await conn.execute(
                """
                UPDATE sessions
                SET current_index=$2,
                    correct_count=correct_count+$3,
                    wrong_count=wrong_count+$4,
                    skipped_count=skipped_count+$5,
                    completed=$6
                WHERE session_id=$1
                """,
                session_id, current_index, correct_delta, wrong_delta, skipped_delta, completed
            )



async def db_set_session_question_ids(
    pool: asyncpg.Pool,
    session_id: uuid.UUID,
    question_ids: List[int],
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE sessions
            SET question_ids=$2
            WHERE session_id=$1
            """,
            session_id,
            json.dumps(question_ids),
        )

async def db_finish_session(pool: asyncpg.Pool, session_id: uuid.UUID) -> Optional[asyncpg.Record]:
    async with pool.acquire() as conn:
        await conn.execute("UPDATE sessions SET completed=TRUE WHERE session_id=$1", session_id)
        return await conn.fetchrow("SELECT * FROM sessions WHERE session_id=$1", session_id)

async def db_list_users(pool: asyncpg.Pool, limit: int = 30) -> List[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT tg_id, phone, trial_until, sub_until, is_admin, last_seen, ok_code, ok_level
            FROM users
            ORDER BY last_seen DESC
            LIMIT $1
            """,
            limit
        )

async def db_set_sub_days(pool: asyncpg.Pool, tg_id: int, days: int) -> Optional[asyncpg.Record]:
    async with pool.acquire() as conn:
        u = await conn.fetchrow("SELECT * FROM users WHERE tg_id=$1", tg_id)
        if not u:
            return None
        base = u["sub_until"] if u["sub_until"] and u["sub_until"] > utcnow() else utcnow()
        new_until = base + timedelta(days=days)
        await conn.execute("UPDATE users SET sub_until=$2 WHERE tg_id=$1", tg_id, new_until)
        return await conn.fetchrow("SELECT * FROM users WHERE tg_id=$1", tg_id)

async def db_revoke_sub(pool: asyncpg.Pool, tg_id: int) -> Optional[asyncpg.Record]:
    async with pool.acquire() as conn:
        u = await conn.fetchrow("SELECT * FROM users WHERE tg_id=$1", tg_id)
        if not u:
            return None
        await conn.execute("UPDATE users SET sub_until=NULL WHERE tg_id=$1", tg_id)
        return await conn.fetchrow("SELECT * FROM users WHERE tg_id=$1", tg_id)

async def db_seed_problem_flags(pool: asyncpg.Pool, problem_ids: Set[int]) -> None:
    if not problem_ids:
        return
    now = utcnow()
    async with pool.acquire() as conn:
        rows = [(qid, True, "from_problem_questions.json", now) for qid in problem_ids]
        await conn.executemany(
            """
            INSERT INTO question_flags(question_id, is_disabled, note, updated_at)
            VALUES($1,$2,$3,$4)
            ON CONFLICT (question_id) DO UPDATE
            SET is_disabled=EXCLUDED.is_disabled,
                note=EXCLUDED.note,
                updated_at=EXCLUDED.updated_at
            """,
            rows
        )

async def db_get_disabled_ids(pool: asyncpg.Pool) -> Set[int]:
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT question_id FROM question_flags WHERE is_disabled=TRUE")
    return {int(r["question_id"]) for r in rows}

async def db_toggle_question(pool: asyncpg.Pool, qid: int, enable: bool, note: str) -> None:
    now = utcnow()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO question_flags(question_id, is_disabled, note, updated_at)
            VALUES($1,$2,$3,$4)
            ON CONFLICT (question_id) DO UPDATE
            SET is_disabled=EXCLUDED.is_disabled,
                note=EXCLUDED.note,
                updated_at=EXCLUDED.updated_at
            """,
            qid, (not enable), note, now
        )

# -------------------------
# Збереження вибору тем (multi-select)
# -------------------------

async def db_get_topic_prefs(pool: asyncpg.Pool, tg_id: int, mode: str, ok_code: str, ok_level: int) -> Set[str]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT topics FROM topic_prefs WHERE tg_id=$1 AND mode=$2 AND ok_code=$3 AND ok_level=$4",
            tg_id, mode, ok_code, ok_level
        )
    if not row or row["topics"] is None:
        return set()
    topics = row["topics"]
    # asyncpg може віддати list напряму або str
    if isinstance(topics, str):
        try:
            topics = json.loads(topics)
        except Exception:
            topics = []
    if not isinstance(topics, list):
        topics = []
    return {str(t) for t in topics}

async def db_set_topic_prefs(pool: asyncpg.Pool, tg_id: int, mode: str, ok_code: str, ok_level: int, topics: Set[str]) -> None:
    now = utcnow()
    payload = json.dumps(sorted(list(topics)))
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO topic_prefs(tg_id, mode, ok_code, ok_level, topics, updated_at) "
            "VALUES($1,$2,$3,$4,$5,$6) "
            "ON CONFLICT (tg_id, mode, ok_code, ok_level) "
            "DO UPDATE SET topics=EXCLUDED.topics, updated_at=EXCLUDED.updated_at",
            tg_id, mode, ok_code, ok_level, payload, now
        )

async def db_clear_topic_prefs(pool: asyncpg.Pool, tg_id: int, mode: str, ok_code: str, ok_level: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM topic_prefs WHERE tg_id=$1 AND mode=$2 AND ok_code=$3 AND ok_level=$4",
            tg_id, mode, ok_code, ok_level
        )



# -------------------------
# Завантаження питань
# -------------------------

def load_question_bank() -> None:
    global QUESTIONS_BY_ID, VALID_QIDS
    global OK_CODES, LEVELS_BY_OK, TOPICS_BY_SCOPE, QIDS_BY_SCOPE, QIDS_BY_SCOPE_TOPIC
    global PROBLEM_IDS_FILE

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        qlist = json.load(f)

    PROBLEM_IDS_FILE = set()
    if os.path.exists(PROBLEMS_FILE):
        with open(PROBLEMS_FILE, "r", encoding="utf-8") as f:
            prob = json.load(f)
        for item in prob.get("items", []):
            try:
                PROBLEM_IDS_FILE.add(int(item["id"]))
            except Exception:
                pass

    QUESTIONS_BY_ID = {int(q["id"]): q for q in qlist}

    VALID_QIDS = []
    LEVELS_BY_OK = {}
    TOPICS_BY_SCOPE = {}
    QIDS_BY_SCOPE = {}
    QIDS_BY_SCOPE_TOPIC = {}

    for qid, q in QUESTIONS_BY_ID.items():
        if qid in PROBLEM_IDS_FILE:
            continue
        if not is_question_valid(q):
            continue

        ok_code = normalize_ok_code(q.get("ok"))
        lvl = normalize_level(q.get("level"), ok_code)
        topic = str(q.get("topic") or "Без блоку")

        VALID_QIDS.append(qid)

        LEVELS_BY_OK.setdefault(ok_code, set()).add(lvl)
        TOPICS_BY_SCOPE.setdefault((ok_code, lvl), set()).add(topic)

        QIDS_BY_SCOPE.setdefault((ok_code, lvl), []).append(qid)
        QIDS_BY_SCOPE_TOPIC.setdefault((ok_code, lvl, topic), []).append(qid)

    OK_CODES = sorted(LEVELS_BY_OK.keys(), key=lambda x: (x != OK_CODE_LAW, x))

    # set -> list
    LEVELS_BY_OK = {k: sorted(list(v)) for k, v in LEVELS_BY_OK.items()}
    TOPICS_BY_SCOPE = {k: sorted(list(v)) for k, v in TOPICS_BY_SCOPE.items()}


def qids_for_position(position_name: str, include_all_levels: bool = False) -> List[int]:
    """
    Повертає список ID питань для заданої посади.

    Використовує POSITION_OK_MAP та вже заповнені структури QIDS_BY_SCOPE.
    Якщо include_all_levels=True — бере всі рівні <= заданого для кожного ОК.
    """
    ok_levels = POSITION_OK_MAP.get(position_name)
    if not ok_levels:
        return []

    pool: Set[int] = set()

    # основні ОК для посади
    for ok_code, max_level in ok_levels.items():
        if include_all_levels:
            # Беремо всі рівні для цього ОК, які <= max_level
            for lvl in levels_for_ok(ok_code):
                if lvl <= max_level:
                    pool.update(base_qids_for_scope(ok_code, lvl))
        else:
            # Беремо тільки конкретний рівень для цього ОК
            pool.update(base_qids_for_scope(ok_code, max_level))

    # 🔹 ДОДАТКОВО: завжди додаємо загальний блок "Законодавство"
    # (ok=None у файлі -> ok_code == OK_CODE_LAW, рівень 0)
    for lvl in levels_for_ok(OK_CODE_LAW):
        pool.update(base_qids_for_scope(OK_CODE_LAW, lvl))

    # застосовуємо фільтр вимкнених питань
    return effective_qids(sorted(pool))



def get_tasks_for_position(position_name: str, include_all_levels: bool = False) -> List[Dict[str, Any]]:
    """
    Повертає повні записи питань (як у questions_flat.json) для заданої посади.
    Зручно, якщо треба список питань для перегляду/експорту.

    include_all_levels=True — брати всі рівні <= заданого для кожного ОК.
    """
    qids = qids_for_position(position_name, include_all_levels=include_all_levels)
    return [QUESTIONS_BY_ID[qid] for qid in qids if qid in QUESTIONS_BY_ID]

def _pos_pref_ok_code(position: str) -> str:
    # ключ для topic_prefs (можна будь-який рядок)
    return f"POS::{position}"

def _short_mode(mode: str) -> str:
    """
    'train' -> 't', 'exam' -> 'e', інше лишає як є
    """
    mode = str(mode)
    if mode == "train":
        return "t"
    if mode == "exam":
        return "e"
    return mode


def _normalize_mode(raw: str) -> str:
    """
    't' / 'train' -> 'train'
    'e' / 'exam'  -> 'exam'
    інше повертаємо як є (на майбутнє)
    """
    raw = str(raw)
    if raw in ("t", "train"):
        return "train"
    if raw in ("e", "exam"):
        return "exam"
    return raw


def topics_for_position(position_name: str) -> List[str]:
    """
    Повертає список тем (topic) для посади, ВКЛЮЧНО із загальним законодавством.
    """
    qids = qids_for_position(position_name, include_all_levels=False)

    s: Set[str] = set()
    has_law = False

    for qid in qids:
        q = QUESTIONS_BY_ID.get(qid)
        if not q:
            continue

        ok_code = normalize_ok_code(q.get("ok"))
        if ok_code == OK_CODE_LAW:
            has_law = True
            continue

        s.add(str(q.get("topic") or "Без блоку"))

    topics = sorted(s)

    # додаємо "Законодавство" першим, якщо воно є в питаннях
    if has_law:
        topics = ["📜 Законодавство"] + topics

    return topics



def qids_for_position_topic(position_name: str, topic: str) -> List[int]:
    """
    Повертає всі питання по конкретному блоку (topic) для посади.
    Загальне законодавство (LAW) не включається.
    """
    qids = qids_for_position(position_name, include_all_levels=False)
    out: List[int] = []
    for qid in qids:
        q = QUESTIONS_BY_ID.get(qid)
        if not q:
            continue
        ok_code = normalize_ok_code(q.get("ok"))
        if ok_code == OK_CODE_LAW:
            continue
        t = str(q.get("topic") or "Без блоку")
        if t == topic:
            out.append(qid)
    return out

def build_position_exam_qids(position_name: str, topics: Optional[Set[str]] = None) -> List[int]:
    """
    Екзамен за посадою:
    - 50 питань із загального законодавства (LAW)
    - по 20 питань з кожного блоку (topic) по посаді
    """
    # 1) Загальне законодавство
    law_pool: List[int] = []
    for lvl in levels_for_ok(OK_CODE_LAW):
        law_pool.extend(base_qids_for_scope(OK_CODE_LAW, lvl))
    law_pool = effective_qids(sorted(set(law_pool)))
    random.shuffle(law_pool)
    law_qids = law_pool[:EXAM_LAW_QUESTIONS]

    # 2) Блоки (topics) по посаді
    if topics is None:
        topics = set(topics_for_position(position_name))
    else:
        topics = set(topics)

    block_qids: List[int] = []
    used: Set[int] = set(law_qids)

    for topic in sorted(topics):
        topic_qids = qids_for_position_topic(position_name, topic)

        # на всяк випадок ще раз відсіюємо LAW та вимкнені питання
        filtered: List[int] = []
        for qid in topic_qids:
            q = QUESTIONS_BY_ID.get(qid)
            if not q:
                continue
            ok_code = normalize_ok_code(q.get("ok"))
            if ok_code == OK_CODE_LAW:
                continue
            filtered.append(qid)
        filtered = effective_qids(filtered)

        # уникаємо дублів між блоками
        filtered = [qid for qid in filtered if qid not in used]
        if not filtered:
            continue

        random.shuffle(filtered)
        take = filtered[:EXAM_PER_TOPIC_QUESTIONS]
        block_qids.extend(take)
        used.update(take)

    exam_qids = law_qids + block_qids
    random.shuffle(exam_qids)
    return exam_qids
async def db_get_ok_prefs(pool: asyncpg.Pool, tg_id: int, mode: str) -> Set[str]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT ok_codes FROM ok_prefs WHERE tg_id=$1 AND mode=$2",
            tg_id, mode
        )
        if not row:
            return set()
        try:
            payload = row["ok_codes"] or []
            if isinstance(payload, str):
                payload = json.loads(payload)
            return {str(x) for x in (payload or [])}
        except Exception:
            return set()

async def db_set_ok_prefs(pool: asyncpg.Pool, tg_id: int, mode: str, ok_codes: Set[str]) -> None:
    now = utcnow()
    payload = json.dumps(sorted({str(x) for x in ok_codes}))
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO ok_prefs(tg_id, mode, ok_codes, updated_at) VALUES($1, $2, $3::jsonb, $4) "
            "ON CONFLICT (tg_id, mode) DO UPDATE SET ok_codes=EXCLUDED.ok_codes, updated_at=EXCLUDED.updated_at",
            tg_id, mode, payload, now
        )

async def db_clear_ok_prefs(pool: asyncpg.Pool, tg_id: int, mode: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM ok_prefs WHERE tg_id=$1 AND mode=$2",
            tg_id, mode
        )

async def db_get_ok_level_prefs(pool: asyncpg.Pool, tg_id: int, mode: str) -> Dict[str, int]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT ok_levels FROM ok_level_prefs WHERE tg_id=$1 AND mode=$2",
            tg_id, mode
        )
        if not row:
            return {}
        try:
            payload = row["ok_levels"] or {}
            if isinstance(payload, str):
                payload = json.loads(payload)
            out: Dict[str, int] = {}
            for k, v in (payload or {}).items():
                try:
                    out[str(k)] = int(v)
                except Exception:
                    continue
            return out
        except Exception:
            return {}

async def db_set_ok_level_prefs(pool: asyncpg.Pool, tg_id: int, mode: str, ok_levels: Dict[str, int]) -> None:
    now = utcnow()
    payload = json.dumps({str(k): int(v) for k, v in (ok_levels or {}).items()})
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO ok_level_prefs(tg_id, mode, ok_levels, updated_at) VALUES($1, $2, $3::jsonb, $4) "
            "ON CONFLICT (tg_id, mode) DO UPDATE SET ok_levels=EXCLUDED.ok_levels, updated_at=EXCLUDED.updated_at",
            tg_id, mode, payload, now
        )

async def db_clear_ok_level_prefs(pool: asyncpg.Pool, tg_id: int, mode: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM ok_level_prefs WHERE tg_id=$1 AND mode=$2",
            tg_id, mode
        )


async def start_exam_session(
    bot: Bot,
    tg_id: int,
    chat_id: int,
    user: asyncpg.Record,
    qids: List[int],
    edit_message: Optional[Message] = None,  # ✅ Додано параметр
) -> None:
    qids = list(dict.fromkeys(qids))
    if not qids:
        await bot.send_message(chat_id, "Немає доступних питань для екзамену.")
        return

    expires = utcnow() + timedelta(minutes=EXAM_DURATION_MINUTES)
    await db_create_session(DB_POOL, tg_id, "exam", qids, expires_at=expires)

    # ✅ Виклик питання з можливістю редагування повідомлення
    await send_current_question(
        bot, DB_POOL, chat_id, tg_id, "exam", edit_message=edit_message
    )


def kb_position_start(mode: str, position: str, back_to: str = "auto") -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()

    if mode == "train":
        count_label = TRAIN_QUESTIONS
    else:
        num_topics = len(topics_for_position(position))
        count_label = EXAM_LAW_QUESTIONS + num_topics * EXAM_PER_TOPIC_QUESTIONS

    cb_mode = _short_mode(mode)
    pid = pos_id(position)

    b.button(
        text=f"🎲 Випадково ({count_label})",
        callback_data=PosMenuCb(mode=cb_mode, pid=pid, action="r").pack(),
    )
    b.button(
        text="📚 Обрати блоки",
        callback_data=PosMenuCb(mode=cb_mode, pid=pid, action="b").pack(),
    )

    if back_to == "menu":
        back_cb = "menu"
    elif back_to == "mode":
        back_cb = f"backmode:{mode}"
    elif back_to == "positions":
        back_cb = TrainModeCb(mode=mode, kind="position").pack()
    else:
        back_cb = f"backmode:{mode}" if mode == "train" else "menu"

    b.button(text="⬅️ Назад", callback_data=back_cb)

    b.adjust(1)
    return b.as_markup()

def kb_pos_topics(
    mode: str,
    position: str,
    page: int = 0,
    selected: Optional[Set[str]] = None,
    per_page: int = 8,
) -> InlineKeyboardMarkup:
    selected_set: Set[str] = set(selected or [])
    topics = topics_for_position(position)
    pid = pos_id(position)

    pages: List[List[str]] = [topics[i:i + per_page] for i in range(0, len(topics), per_page)]
    if not pages:
        pages = [[]]
    page = max(0, min(page, len(pages) - 1))
    current = pages[page]
    start_idx = page * per_page

    b = InlineKeyboardBuilder()

    for i, t in enumerate(current):
        idx = start_idx + i
        icon = "☑️" if (t in selected_set) else "⬜️"
        b.button(
            text=f"{icon} {t}",
            callback_data=PosTopicToggleCb(mode=mode, pid=pid, topic_idx=idx, page=page).pack(),
        )

    b.adjust(1)

    start_label = f"✅ Почати ({len(selected_set)})" if selected_set else "✅ Почати"

    bottom: List[InlineKeyboardButton] = [
        InlineKeyboardButton(
            text="⬅️ Назад",
            callback_data=PosMenuCb(mode=_short_mode(mode), pid=pid, action="m").pack(),
        )
    ]

    if page > 0:
        bottom.append(
            InlineKeyboardButton(
                text="⬅️",
                callback_data=PosTopicPageCb(mode=mode, pid=pid, page=page - 1).pack(),
            )
        )
    if page < len(pages) - 1:
        bottom.append(
            InlineKeyboardButton(
                text="➡️",
                callback_data=PosTopicPageCb(mode=mode, pid=pid, page=page + 1).pack(),
            )
        )

    bottom += [
        InlineKeyboardButton(text="🎯 Всі блоки", callback_data=PosTopicAllCb(mode=mode, pid=pid).pack()),
        InlineKeyboardButton(text=start_label, callback_data=PosTopicDoneCb(mode=mode, pid=pid).pack()),
    ]

    b.row(*bottom)
    return b.as_markup()


@router.callback_query(PosTopicDoneCb.filter())
async def pos_topic_done(call: CallbackQuery, callback_data: PosTopicDoneCb):
    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user or not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    mode = _normalize_mode(str(callback_data.mode))
    pid = int(callback_data.pid)
    position = pos_name(pid)
    if not position:
        await call.answer("Невірна посада", show_alert=True)
        return

    pref_ok = _pos_pref_ok_code(position)
    selected = await db_get_topic_prefs(DB_POOL, tg_id, mode, pref_ok, 0)
    if not selected:
        await call.answer("Оберіть хоча б 1 блок або натисніть «Всі блоки».", show_alert=True)
        return

    await call.answer()
    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    if mode == "train":
        pool_set: Set[int] = set()

        for t in selected:
            # законодавство у тебе позначене окремою “псевдо-темою”
            if t == "📜 Законодавство":
                for lvl in levels_for_ok(OK_CODE_LAW):
                    pool_set.update(base_qids_for_scope(OK_CODE_LAW, lvl))
                continue

            pool_set.update(qids_for_position_topic(position, t))

        pool_qids = effective_qids(sorted(pool_set))
        if not pool_qids:
            await call.answer("У вибраних блоках немає питань.", show_alert=True)
            return

        await start_session_for_pool(call.bot, tg_id, call.message.chat.id, user, mode, pool_qids)

    else:
        # екзамен: LAW додається всередині build_position_exam_qids, тому цю “тему” краще прибрати
        topics = {t for t in selected if t != "📜 Законодавство"}
        exam_qids = build_position_exam_qids(position, topics=topics)
        await start_exam_session(call.bot, tg_id, call.message.chat.id, user, exam_qids)



# -------------------------
# Логіка доступу/профілю
# -------------------------

def user_has_scope(user: asyncpg.Record) -> bool:
    return bool(user["ok_code"])

def get_user_scope(user: asyncpg.Record) -> tuple[str, int]:
    ok_code = str(user["ok_code"])
    # для LAW рівень завжди 0
    if ok_code == OK_CODE_LAW:
        return ok_code, 0

    lvl = user["ok_level"]
    # якщо рівень не вказаний — трактуємо як «всі рівні»
    if lvl is None:
        lvl = LEVEL_ALL
    return ok_code, int(lvl)

async def ensure_profile(message: Message, user: asyncpg.Record, next_mode: str | None = None) -> bool:
    if user_has_scope(user):
        return True

    if next_mode in ("train", "exam"):
        PENDING_AFTER_OK[int(user["tg_id"])] = next_mode

    await message.answer(
        "⚙️ Потрібно обрати <b>ОК</b>, бо для кожного набір питань різний.\n\n"
        "Оберіть ОК:",
        parse_mode=ParseMode.HTML,
        reply_markup=ReplyKeyboardRemove(),
    )
    await message.answer("ОК:", reply_markup=kb_pick_ok(page=0))
    return False



# -------------------------
# Відправка питань
# -------------------------
def build_question_text(
    q: Dict[str, Any],
    idx: int,
    total: int,
    mode: str,
    remaining_seconds: Optional[int],
) -> str:
    qtext = html_escape(str(q.get("question") or ""))

    remaining_q = max(0, int(total) - int(idx))
    prefix = "📚 <b>Навчання</b>" if mode == "train" else "📝 <b>Екзамен</b>"
    head = f"{prefix} • Питання <b>{idx}/{total}</b> • Залишилось <b>{remaining_q}</b>"
    if mode == "exam" and remaining_seconds is not None:
        head += f" • ⏳ {as_minutes_seconds(remaining_seconds)}"

    sep = "────────────\n"   # ← лінія-розділювач

    body = (
        f"{head}\n\n"
        f"❓ <b>Питання:</b>\n<b>{qtext}</b>\n"
        f"{sep}"
        f"🧾 <b>Варіанти відповіді:</b>\n"
    )

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = q.get("choices") or []
    for i, ch in enumerate(choices):
        label = letters[i] if i < len(letters) else str(i + 1)
        body += f"• <b>{label}</b> — {html_escape(str(ch))}\n"

    return body


async def send_current_question(bot: Bot, pool: asyncpg.Pool, chat_id: int, tg_id: int, mode: str, edit_message: Optional[Message] = None) -> None:
    sess = await db_get_active_session(pool, tg_id, mode)
    if not sess:
        await bot.send_message(chat_id, "Немає активної сесії. Оберіть режим у меню.")
        return

    if mode == "exam" and sess["expires_at"] and sess["expires_at"] <= utcnow():
        await finish_exam_due_to_timeout(bot, pool, tg_id, chat_id, sess)
        return

    qids = json.loads(sess["question_ids"])
    total = len(qids)
    idx0 = int(sess["current_index"])
    if idx0 >= total:
        await complete_session_and_show_summary(bot, pool, tg_id, chat_id, sess, auto=True)
        return

    qid = int(qids[idx0])
    q = QUESTIONS_BY_ID.get(qid)
    if not q:
        await db_update_session_progress(pool, sess["session_id"], idx0 + 1, skipped_delta=1)
        await db_stats_add(pool, tg_id, mode, skipped=1)
        await send_current_question(bot, pool, chat_id, tg_id, mode, edit_message=edit_message)
        return

    remaining = None
    if mode == "exam" and sess["expires_at"]:
        remaining = int((sess["expires_at"] - utcnow()).total_seconds())

    text = build_question_text(q, idx0 + 1, total, mode, remaining)
    allow_skip = (mode == "train")
    markup = kb_question(mode=mode, qid=qid, choices=q.get("choices") or [], allow_skip=allow_skip)
    if edit_message is not None:
        try:
            await edit_message.edit_text(text, reply_markup=markup, parse_mode=ParseMode.HTML)
            return
        except Exception:
            # Якщо не можна редагувати (старе/видалене повідомлення) — шлемо нове
            pass
    await bot.send_message(chat_id, text, reply_markup=markup, parse_mode=ParseMode.HTML)

async def complete_session_and_show_summary(
    bot: Bot,
    pool: asyncpg.Pool,
    tg_id: int,
    chat_id: int,
    sess: asyncpg.Record,
    auto: bool = False,
) -> None:
    finished = await db_finish_session(pool, sess["session_id"])
    if not finished:
        return

    total = len(json.loads(finished["question_ids"]))
    correct = int(finished["correct_count"])
    wrong = int(finished["wrong_count"])
    skipped = int(finished["skipped_count"])
    percent = (correct / total * 100.0) if total else 0.0
    mode = finished["mode"]

    title = "📚 Навчання завершено" if mode == "train" else "📝 Екзамен завершено"
    text = (
        f"<b>{title}</b>\n"
        f"Питань: <b>{total}</b>\n"
        f"🎯 Правильних: <b>{percent:.1f}%</b>\n"
        f"✅ Правильно: <b>{correct}</b>\n"
        f"❌ Невірно: <b>{wrong}</b>\n"
    )
    if mode == "train":
        text += f"⏭ Пропущено: <b>{skipped}</b>\n"
    if auto and mode == "exam":
        text += "\n⏳ Час вийшов — екзамен завершено автоматично."

    u = await db_get_user(pool, tg_id)
    await bot.send_message(
        chat_id,
        text,
        parse_mode=ParseMode.HTML,
        reply_markup=kb_main_menu(is_admin=bool(u and u["is_admin"])),
    )

async def finish_exam_due_to_timeout(bot: Bot, pool: asyncpg.Pool, tg_id: int, chat_id: int, sess: asyncpg.Record) -> None:
    await complete_session_and_show_summary(bot, pool, tg_id, chat_id, sess, auto=True)


# -------------------------
# Router та хендлери
# -------------------------

@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    if not DB_POOL:
        await message.answer("Бот ще ініціалізується. Спробуйте через кілька секунд.")
        return

    tg_id = message.from_user.id
    await db_touch_user(DB_POOL, tg_id)
    user = await db_get_user(DB_POOL, tg_id)

    if not user or not user["phone"]:
        reg_msg = await message.answer(
            "Привіт! Щоб почати, потрібна реєстрація.\n\n"
            "1) Натисніть кнопку <b>«📞 Поділитись номером»</b>\n"
            "2) Ви отримаєте <b>3 дні безкоштовного тестування</b>\n",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_request_contact(),
        )
        REG_PROMPT_MSG_ID[tg_id] = reg_msg.message_id
        return

    tmp = await message.answer("✅", reply_markup=ReplyKeyboardRemove())
    try:
        await tmp.delete()
    except Exception:
        pass

    await show_main_menu(message, is_admin=bool(user["is_admin"]))



@router.message(F.contact)
async def on_contact(message: Message) -> None:
    if not DB_POOL:
        return

    tg_id = message.from_user.id

    # прибираємо попередній реєстраційний текст (якщо був)
    msg_id = REG_PROMPT_MSG_ID.pop(tg_id, None)
    if msg_id:
        try:
            await message.bot.delete_message(chat_id=message.chat.id, message_id=msg_id)
        except Exception:
            pass

    c = message.contact
    if c.user_id and c.user_id != tg_id:
        await message.answer(
            "Будь ласка, надішліть <b>свій</b> номер через кнопку.",
            parse_mode=ParseMode.HTML
        )
        return

    phone = c.phone_number
    is_admin = tg_id in ADMIN_IDS
    user = await db_upsert_user(DB_POOL, tg_id, phone, is_admin)

    # прибираємо reply-клавіатуру (кнопку контакту)
    tmp = await message.answer("✅", reply_markup=ReplyKeyboardRemove())
    try:
        await tmp.delete()
    except Exception:
        pass

    # (опційно) пробуємо прибрати повідомлення з контактом
    try:
        await message.delete()
    except Exception:
        pass

    await show_main_menu(message, is_admin=bool(user["is_admin"]))


@router.callback_query(OkPageCb.filter())
async def ok_page(call: CallbackQuery, callback_data: OkPageCb) -> None:
    await call.message.edit_text("Оберіть ОК:", reply_markup=kb_pick_ok(page=int(callback_data.page)))
    await call.answer()


@router.callback_query(OkMultiPageCb.filter())
async def ok_multi_page(call: CallbackQuery, callback_data: OkMultiPageCb) -> None:
    if not DB_POOL:
        return
    tg_id = call.from_user.id
    mode = str(callback_data.mode)
    selected = await db_get_ok_prefs(DB_POOL, tg_id, mode)

    # Просто показуємо ту ж саму клавіатуру без пагінації
    await safe_edit(
        call,
        f"Оберіть <b>декілька</b> ОК (блоків):\nОбрано: <b>{len(selected)}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_pick_ok_multi(mode, page=0, selected=selected),
    )
    await call.answer()

@router.callback_query(OkToggleCb.filter())
async def ok_multi_toggle(call: CallbackQuery, callback_data: OkToggleCb) -> None:
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    mode = str(callback_data.mode)
    ok_code = str(callback_data.ok_code)
    page = int(callback_data.page)

    selected = await db_get_ok_prefs(DB_POOL, tg_id, mode)
    selected = set(selected or [])

    removed = False
    if ok_code in selected:
        selected.remove(ok_code)
        removed = True
    else:
        selected.add(ok_code)

    await db_set_ok_prefs(DB_POOL, tg_id, mode, selected)

    # ✅ якщо зняли галочку — прибираємо його рівень із мапи
    if removed and ok_code != OK_CODE_LAW:
        ok_levels = await db_get_ok_level_prefs(DB_POOL, tg_id, mode)
        ok_levels = dict(ok_levels or {})
        if ok_code in ok_levels:
            del ok_levels[ok_code]
            await db_set_ok_level_prefs(DB_POOL, tg_id, mode, ok_levels)

    await safe_edit(
        call,
        f"Оберіть <b>декілька</b> ОК (блоків):\nОбрано: <b>{len(selected)}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_pick_ok_multi(mode, page=page, selected=selected),
    )
    await call.answer()


@router.callback_query(OkClearCb.filter())
async def ok_multi_clear(call: CallbackQuery, callback_data: OkClearCb) -> None:
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    mode = str(callback_data.mode)
    page = int(callback_data.page)

    await db_clear_ok_prefs(DB_POOL, tg_id, mode)
    await db_clear_ok_level_prefs(DB_POOL, tg_id, mode)  # ✅ додано

    await safe_edit(
        call,
        "Оберіть <b>декілька</b> ОК (блоків):\nОбрано: <b>0</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_pick_ok_multi(mode, page=page, selected=set()),
    )
    await call.answer()


@router.callback_query(OkAllCb.filter())
async def ok_multi_all(call: CallbackQuery, callback_data: OkAllCb) -> None:
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    mode = str(callback_data.mode)

    codes = {OK_CODE_LAW} | {c for c in OK_CODES if c != OK_CODE_LAW}
    await db_set_ok_prefs(DB_POOL, tg_id, mode, codes)
    await db_clear_ok_level_prefs(DB_POOL, tg_id, mode)  # ✅ додано

    await safe_edit(
        call,
        f"Оберіть <b>декілька</b> ОК (блоків):\nОбрано: <b>{len(codes)}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_pick_ok_multi(mode, page=0, selected=codes),
    )
    await call.answer()



@router.callback_query(OkDoneCb.filter())
async def ok_multi_done(call: CallbackQuery, callback_data: OkDoneCb) -> None:
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    mode = str(callback_data.mode)

    user = await db_get_user(DB_POOL, tg_id)
    if not user:
        await call.answer("Немає профілю", show_alert=True)
        return

    selected = await db_get_ok_prefs(DB_POOL, tg_id, mode)
    selected = {c for c in (selected or set()) if c}
    if not selected:
        await call.answer("Оберіть хоча б один ОК", show_alert=True)
        return

    # Якщо обрано 1 ОК — стара логіка без змін
    if len(selected) == 1:
        ok_code = next(iter(selected))
        lvl_to_store = 0 if ok_code == OK_CODE_LAW else LEVEL_ALL
        await db_set_scope(DB_POOL, tg_id, ok_code, lvl_to_store)

        if mode == "train":
            await safe_edit(
                call,
                f"Навчання для: <b>{html_escape(scope_title(ok_code, lvl_to_store))}</b>\nОберіть варіант:",
                parse_mode=ParseMode.HTML,
                reply_markup=kb_train_pick(ok_code, lvl_to_store),
            )
        else:
            await safe_edit(
                call,
                f"Екзамен для: <b>{html_escape(scope_title(ok_code, lvl_to_store))}</b>\nОберіть варіант:",
                parse_mode=ParseMode.HTML,
                reply_markup=kb_exam_pick(ok_code, lvl_to_store),
            )
        await call.answer()
        return

    # ===== Multi-OK: спочатку вибір рівнів по кожному ОК (крім LAW) =====
    ok_levels = await db_get_ok_level_prefs(DB_POOL, tg_id, mode)
    ok_levels = {k: v for k, v in (ok_levels or {}).items() if k in selected and k != OK_CODE_LAW}
    await db_set_ok_level_prefs(DB_POOL, tg_id, mode, ok_levels)

    missing = sorted([ok for ok in selected if ok != OK_CODE_LAW and ok not in ok_levels])
    shown = ", ".join(sorted(selected))

    if missing:
        await safe_edit(
            call,
            f"Обрані модулі: <b>{html_escape(shown)}</b>\n"
            f"Оберіть рівень для кожного модуля (Законодавство — без рівня):",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_multi_levels_overview(mode, selected, ok_levels),
        )
        await call.answer()
        return

    # ===== Рівні вже вибрані для всіх ОК =====
    if mode == "train":
        available = set(multi_topics_for_ok_set(selected, ok_levels))
        chosen_topics = await db_get_topic_prefs(DB_POOL, tg_id, mode, MULTI_OK_CODE, MULTI_OK_LEVEL)
        chosen_topics = {t for t in (chosen_topics or set()) if t in available}
        await db_set_topic_prefs(DB_POOL, tg_id, mode, MULTI_OK_CODE, MULTI_OK_LEVEL, chosen_topics)

        await safe_edit(
            call,
            f"Обрані модулі: <b>{html_escape(shown)}</b>\n"
            f"Тепер оберіть теми для тренування:\n"
            f"Обрано тем: <b>{len(chosen_topics)}</b>",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_multi_topics(mode, selected, ok_levels, page=0, selected=chosen_topics),
        )
        await call.answer()
        return

    # exam: підтвердження старту (як було), але вже з рівнями
    await safe_edit(
        call,
        f"Обрані модулі: <b>{html_escape(shown)}</b>\nПочати екзамен по всіх обраних модулях?",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_train_pick_multi("exam"),
    )
    await call.answer()


@router.callback_query(StartMultiOkCb.filter())
async def start_multi_ok(call: CallbackQuery, callback_data: StartMultiOkCb) -> None:
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    mode = str(callback_data.mode)

    user = await db_get_user(DB_POOL, tg_id)
    if not user:
        await call.answer("Немає профілю", show_alert=True)
        return
    if not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    selected = await db_get_ok_prefs(DB_POOL, tg_id, mode)
    selected = {c for c in (selected or set()) if c}
    if not selected:
        await call.answer("Оберіть ОК", show_alert=True)
        return

    ok_levels = await db_get_ok_level_prefs(DB_POOL, tg_id, mode)
    ok_levels = {k: v for k, v in (ok_levels or {}).items() if k in selected and k != OK_CODE_LAW}
    await db_set_ok_level_prefs(DB_POOL, tg_id, mode, ok_levels)

    missing = sorted([ok for ok in selected if ok != OK_CODE_LAW and ok not in ok_levels])
    if missing:
        shown = ", ".join(sorted(selected))
        await safe_edit(
            call,
            f"Обрані модулі: <b>{html_escape(shown)}</b>\n"
            f"Спочатку оберіть рівні (Законодавство — без рівня):",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_multi_levels_overview(mode, selected, ok_levels),
        )
        await call.answer()
        return

    pool: List[int] = []
    for ok_code in sorted(selected):
        if ok_code == OK_CODE_LAW:
            pool.extend(base_qids_for_scope(OK_CODE_LAW, 0))
        else:
            pool.extend(base_qids_for_scope(ok_code, int(ok_levels[ok_code])))

    pool_qids = effective_qids(list(dict.fromkeys(pool)))

    await call.answer()
    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    await start_session_for_pool(
        call.bot,
        tg_id,
        call.message.chat.id,
        user,
        mode,
        pool_qids,
        edit_message=call.message,
    )


@router.callback_query(OkPickCb.filter())
async def ok_pick(call: CallbackQuery, callback_data: OkPickCb):
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)

    if not user or not user["phone"]:
        # reply keyboard (контакт) не редагується через edit_text — тут OK робити answer
        await call.message.answer("Спочатку зареєструйтесь.", reply_markup=kb_request_contact())
        await call.answer()
        return

    ok_code = str(callback_data.ok_code)

    # рівень більше не має значення
    lvl_to_store = 0 if ok_code == OK_CODE_LAW else LEVEL_ALL
    user = await db_set_scope(DB_POOL, tg_id, ok_code, lvl_to_store)
    # синхронізуємо manual multi-select (за замовчуванням один блок)
    try:
        await db_set_ok_prefs(DB_POOL, tg_id, "train", {ok_code})
    except Exception:
        pass

    next_mode = PENDING_AFTER_OK.pop(tg_id, None)

    if next_mode == "train":
        await safe_edit(
            call,
            f"Навчання для: <b>{html_escape(scope_title(ok_code, lvl_to_store))}</b>\nОберіть варіант:",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_train_pick(ok_code, lvl_to_store),
        )
        await call.answer()
        return

    if next_mode == "exam":
        await safe_edit(
            call,
            f"Екзамен для: <b>{html_escape(scope_title(ok_code, lvl_to_store))}</b>\nОберіть варіант:",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_exam_pick(ok_code, lvl_to_store),
        )
        await call.answer()
        return

    # дефолт: підтвердження + меню (в тому ж повідомленні)
    await safe_edit(
        call,
        f"✅ Встановлено: <b>{html_escape(scope_title(ok_code, lvl_to_store))}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])),
    )
    await call.answer()



@router.callback_query(LevelPickCb.filter())
async def level_pick(call: CallbackQuery, callback_data: LevelPickCb) -> None:
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)

    if not user or not user["phone"]:
        await call.message.answer("Спочатку зареєструйтесь (поділіться номером).", reply_markup=kb_request_contact())
        await call.answer()
        return

    ok_code = str(callback_data.ok_code)
    lvl = int(callback_data.level)

    user = await db_set_scope(DB_POOL, tg_id, ok_code, lvl)

    await safe_edit(
        call,
        f"✅ Встановлено: <b>{html_escape(scope_title(ok_code, lvl))}</b>\nТепер можете починати навчання/екзамен.",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])),
    )
    await call.answer()


@router.callback_query(F.data == "pickok")
async def pick_ok_from_anywhere(call: CallbackQuery) -> None:
    await safe_edit(call, "Оберіть ОК:", reply_markup=kb_pick_ok(page=0))
    await call.answer()


@router.callback_query(F.data.startswith("mm:"))
async def menu_actions_inline(call: CallbackQuery) -> None:
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    await db_touch_user(DB_POOL, tg_id)
    user = await db_get_user(DB_POOL, tg_id)

    # Не зареєстрований
    if not user or not user["phone"]:
        await call.message.answer(
            "Спочатку зареєструйтесь (поділіться номером).",
            reply_markup=kb_request_contact(),
        )
        await call.answer()
        return

    _, action = call.data.split(":", 1)

    # SETTINGS
    if action == "settings":
        if user_has_scope(user):
            ok_code, lvl = get_user_scope(user)
            out = (
                f"⚙️ Ваш поточний набір: <b>{html_escape(scope_title(ok_code, lvl))}</b>\n"
                "Натисніть нижче, щоб змінити:"
            )
        else:
            out = "⚙️ Потрібно налаштувати ОК:"
        await safe_edit(call, out, parse_mode=ParseMode.HTML, reply_markup=kb_pick_ok(page=0))
        await call.answer()
        return

    # STATS
    if action == "stats":
        rows = await db_stats_get(DB_POOL, tg_id)
        if not rows:
            await safe_edit(
                call,
                "Статистики поки нема.",
                reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])),
            )
            await call.answer()
            return

        out = "<b>📊 Ваша статистика</b>\n\n"
        for r in rows:
            out += (
                f"<b>{'Навчання' if r['mode'] == 'train' else 'Екзамен'}</b>\n"
                f"Відповіли: {r['answered']}\n"
                f"✅ Правильно: {r['correct']}\n"
                f"❌ Невірно: {r['wrong']}\n"
            )
            if r["mode"] == "train":
                out += f"⏭ Пропущено: {r['skipped']}\n"
            out += "\n"

        await safe_edit(
            call,
            out,
            parse_mode=ParseMode.HTML,
            reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])),
        )
        await call.answer()
        return

    # ACCESS
    if action == "access":
        now = utcnow()
        tu = user["trial_until"]
        su = user["sub_until"]
        has = await db_has_access(user)

        out = "<b>ℹ️ Доступ</b>\n\n"
        out += f"Статус: {'✅ активний' if has else '⛔️ неактивний'}\n"
        if tu:
            out += f"Тріал до: <b>{tu.astimezone(KYIV_TZ).strftime('%Y-%m-%d %H:%M Kyiv')}</b>\n"
        if su:
            out += f"Підписка до: <b>{su.astimezone(KYIV_TZ).strftime('%Y-%m-%d %H:%M Kyiv')}</b>\n"
        if user_has_scope(user):
            ok_code, lvl = get_user_scope(user)
            out += f"Набір: <b>{html_escape(scope_title(ok_code, lvl))}</b>\n"
        else:
            out += "Набір: <i>не вибрано</i>\n"
        out += f"Зараз: <code>{now.astimezone(KYIV_TZ).strftime('%Y-%m-%d %H:%M Kyiv')}</code>\n"

        await safe_edit(
            call,
            out,
            parse_mode=ParseMode.HTML,
            reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])),
        )
        await call.answer()
        return

    # ADMIN
    if action == "admin":
        if not user.get("is_admin"):
            await call.answer("Тільки для адміна", show_alert=True)
            return
        await safe_edit(call, "🛠 Адмін-панель", reply_markup=kb_admin_panel())
        await call.answer()
        return

    # TRAIN
    if action == "train":
        if not await db_has_access(user):
            await safe_edit(
                call,
                "⛔️ Доступ завершився.\nНапишіть адміну для доступу.",
                reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])),
            )
            await call.answer()
            return

        selected_ok = await db_get_ok_prefs(DB_POOL, tg_id, "train")
        selected_ok = set(selected_ok or [])

        # fallback: якщо є старий single-scope — підхопимо його
        if not selected_ok and user_has_scope(user):
            ok_code, _lvl = get_user_scope(user)
            selected_ok = {ok_code}
            await db_set_ok_prefs(DB_POOL, tg_id, "train", selected_ok)

        if not selected_ok:
            await safe_edit(
                call,
                "Оберіть <b>модулі</b> (ОК) для навчання:\n"
                f"Обрано: <b>0</b>",
                parse_mode=ParseMode.HTML,
                reply_markup=kb_pick_ok_multi("train", page=0, selected=set()),
            )
        else:
            # Якщо обрано 1 модуль — лишаємо стару логіку
            if len(selected_ok) == 1:
                ok_code = next(iter(selected_ok))
                lvl = 0 if ok_code == OK_CODE_LAW else LEVEL_ALL
                await db_set_scope(DB_POOL, tg_id, ok_code, lvl)
                await safe_edit(
                    call,
                    f"Навчання для: <b>{html_escape(scope_title(ok_code, lvl))}</b>\nОберіть варіант:",
                    parse_mode=ParseMode.HTML,
                    reply_markup=kb_train_pick(ok_code, lvl),
                )
            else:
                shown = ", ".join(sorted(selected_ok))
                available = set(multi_topics_for_ok_set(selected_ok))
                selected = await db_get_topic_prefs(DB_POOL, tg_id, "train", MULTI_OK_CODE, MULTI_OK_LEVEL)
                selected = {t for t in selected if t in available}
                await db_set_topic_prefs(DB_POOL, tg_id, "train", MULTI_OK_CODE, MULTI_OK_LEVEL, selected)
                await safe_edit(
                    call,
                    f"Обрані модулі: <b>{html_escape(shown)}</b>\n"
                    f"Оберіть теми для тренування:\n"
                    f"Обрано тем: <b>{len(selected)}</b>",
                    parse_mode=ParseMode.HTML,
                    reply_markup=kb_multi_topics("train", selected_ok, page=0, selected=selected),
                )

        await call.answer()
        return

    # EXAM
    if action == "exam":
        if not await db_has_access(user):
            await safe_edit(
                call,
                "⛔️ Доступ завершився.\nНапишіть адміну для доступу.",
                reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])),
            )
            await call.answer()
            return

        selected_ok = await db_get_ok_prefs(DB_POOL, tg_id, "exam")
        selected_ok = set(selected_ok or [])

        # fallback: якщо для екзамену немає, беремо з тренування
        if not selected_ok:
            train_ok = await db_get_ok_prefs(DB_POOL, tg_id, "train")
            train_ok = set(train_ok or [])
            if train_ok:
                selected_ok = train_ok
                await db_set_ok_prefs(DB_POOL, tg_id, "exam", selected_ok)

        if not selected_ok:
            await safe_edit(
                call,
                "Оберіть <b>модулі</b> (ОК) для екзамену:\n"
                f"Обрано: <b>0</b>",
                parse_mode=ParseMode.HTML,
                reply_markup=kb_pick_ok_multi("exam", page=0, selected=set()),
            )
        else:
            shown = ", ".join(sorted(selected_ok))
            await safe_edit(
                call,
                "📝 <b>Екзамен</b>\n\n"
                f"Обрані модулі: <b>{html_escape(shown)}</b>\n\n"
                "Оберіть варіант:",
                parse_mode=ParseMode.HTML,
                reply_markup=kb_train_pick_multi("exam"),
            )

        await call.answer()
        return

    await safe_edit(call, "🏠 Меню", reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])))
    await call.answer()



@router.callback_query(TrainModeCb.filter())
async def train_mode_pick(call: CallbackQuery, callback_data: TrainModeCb):
    mode = callback_data.mode      # "train" (для inline-меню)
    kind = callback_data.kind      # "position" | "manual"

    if not DB_POOL:
        return

    # Оновлюємо режим тренування в БД
    result = await DB_POOL.fetchrow(
        "UPDATE users SET train_mode=$2 WHERE tg_id=$1 RETURNING tg_id",
        call.from_user.id,
        kind,
    )

    if result is None:
        await call.answer("Помилка при збереженні режиму. Спробуйте ще раз.", show_alert=True)
        return

    if kind == "manual":
        # показуємо multi-select ОК
        selected = await db_get_ok_prefs(DB_POOL, call.from_user.id, "train")
        # якщо ще нічого не збережено — підхопимо старий single-scope
        if not selected:
            u = await db_get_user(DB_POOL, call.from_user.id)
            if u and u.get("ok_code"):
                selected = {str(u["ok_code"])}
                await db_set_ok_prefs(DB_POOL, call.from_user.id, "train", selected)
        await call.message.edit_text(
            "Оберіть <b>декілька</b> ОК (блоків):\nОбрано: <b>%d</b>" % (len(selected) if selected else 0),
            parse_mode=ParseMode.HTML,
            reply_markup=kb_pick_ok_multi("train", page=0, selected=selected),
        )
        await call.answer()
        return

    if kind == "position":
        await call.message.edit_text(
            "Оберіть посаду:",
            reply_markup=kb_pick_position(mode, back_to="mode"),
        )
        await call.answer()
        return

    await call.answer()



@router.callback_query(F.data.startswith("pos:"))
async def position_pick(call: CallbackQuery):
    _, mode_raw, pid_str = call.data.split(":", 2)
    pid = int(pid_str)
    position = pos_name(pid)
    if not position:
        await call.answer("Невірна посада", show_alert=True)
        return

    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)

    if not user or not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    mode = _normalize_mode(mode_raw)

    pool_qids = qids_for_position(position_name=position, include_all_levels=False)
    if not pool_qids:
        await call.answer("Для цієї посади немає питань", show_alert=True)
        return

    await db_set_position(DB_POOL, tg_id, position)

    pref_ok = _pos_pref_ok_code(position)
    selected = await db_get_topic_prefs(DB_POOL, tg_id, mode, pref_ok, 0)

    title = (
        f"👔 Посада: <b>{html_escape(position)}</b>\n"
        f"Оберіть <b>декілька</b> блоків для "
        f"<b>{'навчання' if mode == 'train' else 'екзамену'}</b>\n"
        f"Обрано блоків: <b>{len(selected)}</b>\n\n"
        "Натискайте блоки (⬜️/☑️), потім — <b>✅ Почати</b> або «🎯 Всі блоки»."
    )

    await call.message.edit_text(
        title,
        parse_mode=ParseMode.HTML,
        reply_markup=kb_pos_topics(mode, position, page=0, selected=selected),
    )
    await call.answer()



@router.callback_query(PosMenuCb.filter())
async def pos_menu(call: CallbackQuery, callback_data: PosMenuCb):
    if not DB_POOL:
        await call.answer()
        return

    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user:
        await call.answer("Немає профілю", show_alert=True)
        return
    if not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    mode = _normalize_mode(str(callback_data.mode))
    position = pos_name(int(callback_data.pid))
    if not position:
        await call.answer("Невірна посада", show_alert=True)
        return

    if user.get("position") != position:
        await db_set_position(DB_POOL, tg_id, position)

    raw_action = str(callback_data.action)
    action_map = {"r": "random", "b": "blocks", "m": "menu"}
    action = action_map.get(raw_action, raw_action)

    pool_qids = qids_for_position(position_name=position, include_all_levels=False)
    if not pool_qids:
        await call.answer("Для цієї посади немає питань", show_alert=True)
        return

    if action == "random":
        # Перед стартом прибираємо клавіатуру/меню, щоб не висіло і не було повторних натискань
        if call.message:
            try:
                await call.message.edit_reply_markup(reply_markup=None)
            except Exception:
                pass

        await call.answer()

        if mode == "train":
            await start_session_for_pool(
                call.bot, tg_id, call.message.chat.id, user, mode, pool_qids
            )
        else:
            exam_qids = build_position_exam_qids(position)
            await start_exam_session(
                call.bot, tg_id, call.message.chat.id, user, exam_qids
            )
        return

    if action == "blocks":
        pref_ok = _pos_pref_ok_code(position)
        selected = await db_get_topic_prefs(DB_POOL, tg_id, mode, pref_ok, 0)

        title = (
            f"👔 Посада: <b>{html_escape(position)}</b>\n"
            f"Оберіть <b>декілька</b> блоків для <b>{'навчання' if mode=='train' else 'екзамену'}</b>\n"
            f"Обрано блоків: <b>{len(selected)}</b>\n\n"
            "Натискайте блоки (⬜️/☑️), потім — <b>✅ Почати</b>."
        )

        await call.message.edit_text(
            title,
            parse_mode=ParseMode.HTML,
            reply_markup=kb_pos_topics(mode, position, page=0, selected=selected),
        )
        await call.answer()
        return

    if action == "menu":
        await call.message.edit_text(
            f"👔 Посада: <b>{html_escape(position)}</b>\nОберіть як почати:",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_position_start(mode, position),
        )
        await call.answer()
        return

    await call.answer()



@router.callback_query(PosTopicPageCb.filter())
async def pos_topic_page(call: CallbackQuery, callback_data: PosTopicPageCb):
    tg_id = call.from_user.id
    mode = _normalize_mode(str(callback_data.mode))
    position = pos_name(int(callback_data.pid))
    if not position:
        await call.answer("Невірна посада", show_alert=True)
        return
    page = int(callback_data.page)

    pref_ok = _pos_pref_ok_code(position)
    selected = await db_get_topic_prefs(DB_POOL, tg_id, mode, pref_ok, 0)

    title = (
        f"👔 Посада: <b>{html_escape(position)}</b>\n"
        f"Оберіть <b>декілька</b> блоків для <b>{'навчання' if mode == 'train' else 'екзамену'}</b>\n"
        f"Обрано блоків: <b>{len(selected)}</b>\n\n"
        "Натискайте блоки (⬜️/☑️), потім — <b>✅ Почати</b>."
    )
    await call.message.edit_text(
        title,
        parse_mode=ParseMode.HTML,
        reply_markup=kb_pos_topics(mode, position, page=page, selected=selected),
    )
    await call.answer()

@router.callback_query(PosTopicToggleCb.filter())
async def pos_topic_toggle(call: CallbackQuery, callback_data: PosTopicToggleCb):
    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user or not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    mode = _normalize_mode(str(callback_data.mode))
    pid = int(callback_data.pid)
    position = pos_name(pid)
    idx = int(callback_data.topic_idx)
    page = int(callback_data.page)

    if not position:
        await call.answer("Невірна посада", show_alert=True)
        return

    topics = topics_for_position(position)
    if idx < 0 or idx >= len(topics):
        await call.answer("Невірний блок", show_alert=True)
        return

    topic = topics[idx]
    pref_ok = _pos_pref_ok_code(position)
    selected = await db_get_topic_prefs(DB_POOL, tg_id, mode, pref_ok, 0)

    if topic in selected:
        selected.remove(topic)
    else:
        selected.add(topic)

    await db_set_topic_prefs(DB_POOL, tg_id, mode, pref_ok, 0, selected)

    title = (
        f"👔 Посада: <b>{html_escape(position)}</b>\n"
        f"Оберіть <b>декілька</b> блоків для <b>{'навчання' if mode == 'train' else 'екзамену'}</b>\n"
        f"Обрано блоків: <b>{len(selected)}</b>\n\n"
        "Натискайте блоки (⬜️/☑️), потім — <b>✅ Почати</b>."
    )
    await call.message.edit_text(
        title,
        parse_mode=ParseMode.HTML,
        reply_markup=kb_pos_topics(mode, position, page=page, selected=selected),
    )
    await call.answer()

@router.callback_query(PosTopicClearCb.filter())
async def pos_topic_clear(call: CallbackQuery, callback_data: PosTopicClearCb):
    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user or not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    mode = _normalize_mode(str(callback_data.mode))
    pid = int(callback_data.pid)
    position = pos_name(pid)
    page = int(callback_data.page)

    if not position:
        await call.answer("Невірна посада", show_alert=True)
        return

    pref_ok = _pos_pref_ok_code(position)
    await db_clear_topic_prefs(DB_POOL, tg_id, mode, pref_ok, 0)

    title = (
        f"👔 Посада: <b>{html_escape(position)}</b>\n"
        f"Оберіть <b>декілька</b> блоків для <b>{'навчання' if mode == 'train' else 'екзамену'}</b>\n"
        "Обрано блоків: <b>0</b>\n\n"
        "Натискайте блоки (⬜️/☑️), потім — <b>✅ Почати</b>."
    )
    await call.message.edit_text(
        title,
        parse_mode=ParseMode.HTML,
        reply_markup=kb_pos_topics(mode, position, page=page, selected=set()),
    )
    await call.answer("Очищено")
@router.callback_query(PosTopicAllCb.filter())
async def pos_topic_all(call: CallbackQuery, callback_data: PosTopicAllCb):
    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user or not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    mode = _normalize_mode(str(callback_data.mode))
    pid = int(callback_data.pid)
    position = pos_name(pid)

    if not position:
        await call.answer("Невірна посада", show_alert=True)
        return

    pool_qids = qids_for_position(position_name=position, include_all_levels=False)
    if not pool_qids:
        await call.answer("Для цієї посади немає питань", show_alert=True)
        return

    await call.answer()

    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    if mode == "train":
        await start_session_for_pool(
            call.bot, tg_id, call.message.chat.id, user, mode, pool_qids,
            edit_message=call.message,
        )
    else:
        exam_qids = build_position_exam_qids(position)
        await start_exam_session(
            call.bot, tg_id, call.message.chat.id, user, exam_qids,
            edit_message=call.message,
        )


@router.callback_query(TopicDoneCb.filter())
async def topic_done(call: CallbackQuery, callback_data: TopicDoneCb) -> None:
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)

    if not user:
        await call.answer("Немає профілю", show_alert=True)
        return

    if not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    ok_code = str(callback_data.ok_code)
    lvl = int(callback_data.level)
    mode = str(callback_data.mode)

    selected = await db_get_topic_prefs(DB_POOL, tg_id, mode, ok_code, lvl)
    if not selected:
        await call.answer(
            "Оберіть хоча б 1 блок або натисніть «Всі блоки».",
            show_alert=True,
        )
        return

    pool_set: set[int] = set()
    for t in selected:
        base = base_qids_for_topic(ok_code, lvl, t)
        pool_set.update(base)

    pool_qids = effective_qids(list(pool_set))
    if not pool_qids:
        await call.answer(
            "У вибраних блоках немає питань.",
            show_alert=True,
        )
        return

    await call.answer()

    # прибираємо клавіатуру під повідомленням
    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    await start_session_for_pool(
        call.bot,
        tg_id,
        call.message.chat.id,
        user,
        mode,
        pool_qids,
        edit_message=call.message,
    )

    if mode == "train":
        pool_size = len(pool_qids)
        title = (
            f"Навчання • <b>{html_escape(scope_title(ok_code, lvl))}</b>\n"
            f"Обрано блоків: <b>{len(selected)}</b>\n"
            "Як сформувати питання?"
        )
        await call.message.edit_text(
            title,
            parse_mode=ParseMode.HTML,
            reply_markup=kb_train_question_set(
                "topics",
                ok_code,
                lvl,
                pool_size,
            ),
        )
        return



def kb_pick_position(mode: str, back_to: str = "auto") -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    m = _short_mode(mode)

    for name in POSITIONS:
        pid = pos_id(name)
        b.row(
            InlineKeyboardButton(
                text=f"👔 {name}",
                callback_data=f"pos:{m}:{pid}",
            )
        )

    if back_to == "menu":
        back_cb = "menu"
    elif back_to == "mode":
        back_cb = f"backmode:{mode}"
    else:
        back_cb = f"backmode:{mode}" if mode == "train" else "menu"

    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data=back_cb))
    return b.as_markup()



@router.callback_query(F.data.startswith("backmode:"))
async def backmode(call: CallbackQuery):
    mode = call.data.split(":", 1)[1]

    text = "Як ви хочете навчатись?" if mode == "train" else "Як ви хочете складати екзамен?"

    await call.message.edit_text(
        text,
        reply_markup=kb_train_mode(mode)
    )
    await call.answer()


@router.message(F.text.in_({"📚 Навчання", "📝 Екзамен", "📊 Статистика", "ℹ️ Доступ", "⚙️ Налаштування"}))
async def menu_actions(message: Message) -> None:
    if not DB_POOL:
        return

    tg_id = message.from_user.id
    await db_touch_user(DB_POOL, tg_id)
    user = await db_get_user(DB_POOL, tg_id)

    if not user or not user["phone"]:
        await message.answer(
            "Спочатку зареєструйтесь (поділіться номером).",
            reply_markup=kb_request_contact(),
        )
        return

    text = (message.text or "").strip()

    if text == "⚙️ Налаштування":
        if user_has_scope(user):
            ok_code, lvl = get_user_scope(user)
            await message.answer(
                f"⚙️ Ваш поточний набір: <b>{html_escape(scope_title(ok_code, lvl))}</b>\n"
                "Натисніть нижче, щоб змінити:",
                parse_mode=ParseMode.HTML,
                reply_markup=ReplyKeyboardRemove(),
            )
        else:
            await message.answer(
                "⚙️ Потрібно налаштувати ОК:",
                reply_markup=ReplyKeyboardRemove(),
            )
        await message.answer("ОК:", reply_markup=kb_pick_ok(page=0))
        return

    # для навчання/екзамену потрібен доступ, а scope потрібен тільки для екзамену
    if text in ("📚 Навчання", "📝 Екзамен"):
        if text == "📝 Екзамен" and not user_has_scope(user):
            await ensure_profile(message, user)
            return

        if not await db_has_access(user):
            await message.answer(
                "⛔️ Доступ завершився.\n"
                "Підписку додамо далі. Напишіть адміну для доступу.",
                reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])),
            )
            return

    # ✅ UPDATED: TRAIN
    if text == "📚 Навчання":
        selected_ok = await db_get_ok_prefs(DB_POOL, tg_id, "train")
        selected_ok = set(selected_ok or [])

        # fallback: якщо є старий single-scope — підхопимо його
        if not selected_ok and user_has_scope(user):
            ok_code, _lvl = get_user_scope(user)
            selected_ok = {ok_code}
            await db_set_ok_prefs(DB_POOL, tg_id, "train", selected_ok)

        # Якщо вже є вибрані ОК - одразу починаємо навчання
        if selected_ok:
            pool: List[int] = []
            for ok_code in sorted(selected_ok):
                lvl = 0 if ok_code == OK_CODE_LAW else LEVEL_ALL
                pool.extend(base_qids_for_scope(ok_code, lvl))

            pool_qids = effective_qids(list(dict.fromkeys(pool)))

            await start_session_for_pool(
                message.bot,
                tg_id,
                message.chat.id,
                user,
                "train",
                pool_qids,
            )
            return

        # Якщо ОК ще не вибрані - показуємо вибір ОК
        await message.answer(
            "Оберіть <b>модулі</b> (ОК):\n"
            f"Обрано: <b>{len(selected_ok)}</b>",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_pick_ok_multi("train", page=0, selected=selected_ok),
        )
        return

    if text == "📝 Екзамен":
        ok_code, lvl = get_user_scope(user)

        # Автоматичний старт екзамену з структурою: 50 з законодавства + 20 з кожного блоку
        law_pool = []
        for law_lvl in levels_for_ok(OK_CODE_LAW):
            law_pool.extend(base_qids_for_scope(OK_CODE_LAW, law_lvl))
        law_pool = effective_qids(sorted(set(law_pool)))
        random.shuffle(law_pool)
        law_qids = law_pool[:EXAM_LAW_QUESTIONS]

        # Блоки (теми) для поточного scope
        topics = effective_topics(ok_code, lvl)
        block_qids = []
        used = set(law_qids)
        for topic in sorted(topics):
            topic_qids = base_qids_for_topic(ok_code, lvl, topic)
            filtered = effective_qids(topic_qids)
            filtered = [qid for qid in filtered if qid not in used]
            if not filtered:
                continue
            random.shuffle(filtered)
            take = filtered[:EXAM_PER_TOPIC_QUESTIONS]
            block_qids.extend(take)
            used.update(take)

        exam_qids = law_qids + block_qids
        random.shuffle(exam_qids)

        if len(exam_qids) < EXAM_LAW_QUESTIONS:
            await message.answer(
                "Недостатньо питань для екзамену. Зверніться до адміністратора.",
                reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])),
            )
            return

        await message.answer(
            f"Екзамен для: <b>{html_escape(scope_title(ok_code, lvl))}</b>\n"
            f"Питань: <b>{len(exam_qids)}</b>, час: <b>{EXAM_DURATION_MINUTES} хв</b>\n"
            "Правильні відповіді не показуються.",
            parse_mode=ParseMode.HTML,
        )
        await start_exam_session(message.bot, tg_id, message.chat.id, user, exam_qids)
        return

    if text == "📊 Статистика":
        rows = await db_stats_get(DB_POOL, tg_id)
        if not rows:
            await message.answer(
                "Поки що статистики немає.",
                reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])),
            )
            return

        out = "<b>📊 Ваша статистика</b>\n\n"
        for r in rows:
            out += (
                f"<b>{'Навчання' if r['mode']=='train' else 'Екзамен'}</b>\n"
                f"Відповіли: {r['answered']}\n"
                f"✅ Правильно: {r['correct']}\n"
                f"❌ Невірно: {r['wrong']}\n"
            )
            if r["mode"] == "train":
                out += f"⏭ Пропущено: {r['skipped']}\n"
            out += "\n"

        await message.answer(
            out,
            parse_mode=ParseMode.HTML,
            reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])),
        )
        return

    if text == "ℹ️ Доступ":
        now = utcnow()
        tu = user["trial_until"]
        su = user["sub_until"]
        has = await db_has_access(user)

        out = "<b>ℹ️ Доступ</b>\n\n"
        out += f"Статус: {'✅ активний' if has else '⛔️ неактивний'}\n"
        if tu:
            out += f"Trial до: <b>{tu.astimezone(KYIV_TZ).strftime('%Y-%m-%d %H:%M Kyiv')}</b>\n"
        if su:
            out += f"Підписка до: <b>{su.astimezone(KYIV_TZ).strftime('%Y-%m-%d %H:%M Kyiv')}</b>\n"
        if user_has_scope(user):
            ok_code, lvl = get_user_scope(user)
            out += f"Набір: <b>{html_escape(scope_title(ok_code, lvl))}</b>\n"
        else:
            out += "Набір: <i>не вибрано</i>\n"
        out += f"Зараз: <code>{now.astimezone(KYIV_TZ).strftime('%Y-%m-%d %H:%M Kyiv')}</code>\n"

        await message.answer(
            out,
            parse_mode=ParseMode.HTML,
            reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])),
        )
        return



# -------------------------
# Старт навчання/екзамену + вибір блоку
# -------------------------
async def start_session_for_pool(
    bot: Bot,
    tg_id: int,
    chat_id: int,
    user: asyncpg.Record,
    mode: str,
    pool_qids: List[int],
    edit_message: Optional[Message] = None,  # ✅ Додано параметр
) -> None:
    if mode == "train":
        if not pool_qids:
            await bot.send_message(chat_id, "Немає доступних питань для навчання.")
            return

        qids = list(dict.fromkeys(pool_qids))
        random.shuffle(qids)

        await db_create_session(DB_POOL, tg_id, "train", qids, expires_at=None)

        # ✅ Додано edit_message
        await send_current_question(
            bot, DB_POOL, chat_id, tg_id, "train", edit_message=edit_message
        )
        return

    if mode == "exam":
        if len(pool_qids) < EXAM_QUESTIONS:
            await bot.send_message(
                chat_id,
                f"Для цього набору доступно лише <b>{len(pool_qids)}</b> питань.\n"
                f"Екзамен потребує <b>{EXAM_QUESTIONS}</b>.\n"
                "Оберіть інший блок/рівень або додайте питання.",
                parse_mode=ParseMode.HTML,
            )
            return

        qids = random.sample(pool_qids, EXAM_QUESTIONS)
        expires = utcnow() + timedelta(minutes=EXAM_DURATION_MINUTES)
        await db_create_session(DB_POOL, tg_id, "exam", qids, expires_at=expires)

        # ✅ Додано edit_message
        await send_current_question(
            bot, DB_POOL, chat_id, tg_id, "exam", edit_message=edit_message
        )
        return

@router.callback_query(TrainVariantCb.filter())
async def train_variant_start(call: CallbackQuery, callback_data: TrainVariantCb) -> None:
    if not DB_POOL:
        await call.answer()
        return

    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user:
        await call.answer("Немає профілю", show_alert=True)
        return
    if not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    kind = str(callback_data.kind)
    ok_code = str(callback_data.ok_code)
    lvl = int(callback_data.level)
    variant = str(callback_data.variant)

    pool_qids: List[int] = []

    if kind == "scope":
        pool_qids = effective_qids(base_qids_for_scope(ok_code, lvl))

    elif kind == "topics":
        selected = await db_get_topic_prefs(DB_POOL, tg_id, "train", ok_code, lvl)
        if not selected:
            await call.answer("Спочатку оберіть хоча б 1 блок.", show_alert=True)
            return
        pool_set: Set[int] = set()
        for t in selected:
            pool_set.update(base_qids_for_topic(ok_code, lvl, t))
        pool_qids = effective_qids(list(pool_set))

    elif kind == "multi":
        ok_codes = await db_get_ok_prefs(DB_POOL, tg_id, "train")
        ok_codes = {c for c in ok_codes if c}
        if not ok_codes:
            await call.answer("Оберіть модулі (ОК) спочатку.", show_alert=True)
            return

        available = set(multi_topics_for_ok_set(ok_codes))
        selected = await db_get_topic_prefs(DB_POOL, tg_id, "train", MULTI_OK_CODE, MULTI_OK_LEVEL)
        selected = {t for t in selected if t in available}
        if not selected:
            await call.answer("Спочатку оберіть хоча б 1 тему.", show_alert=True)
            return

        pool: List[int] = []
        for label in selected:
            pool.extend(qids_for_multi_topic_label(label))
        pool_qids = effective_qids(list(dict.fromkeys(pool)))

    else:
        await call.answer("Невідомий режим.", show_alert=True)
        return

    if not pool_qids:
        await call.answer("Немає доступних питань.", show_alert=True)
        return

    qids = list(dict.fromkeys(pool_qids))

    if variant == "rand":
        k = min(TRAIN_QUESTIONS, len(qids))
        qids = random.sample(qids, k)

    await call.answer()
    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    await start_session_for_pool(
        call.bot, tg_id, call.message.chat.id, user, "train", qids, edit_message=call.message
    )


@router.callback_query(TrainVariantBackCb.filter())
async def train_variant_back(call: CallbackQuery, callback_data: TrainVariantBackCb) -> None:
    if not DB_POOL:
        await call.answer()
        return

    tg_id = call.from_user.id
    kind = str(callback_data.kind)
    ok_code = str(callback_data.ok_code)
    lvl = int(callback_data.level)

    if kind == "scope":
        await call.message.edit_text(
            f"Навчання для: <b>{html_escape(scope_title(ok_code, lvl))}</b>\nОберіть варіант:",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_train_pick(ok_code, lvl),
        )
        await call.answer()
        return

    if kind == "topics":
        selected = await db_get_topic_prefs(DB_POOL, tg_id, "train", ok_code, lvl)
        title = (
            f"Оберіть <b>декілька</b> блоків для <b>навчання</b>\n"
            f"Набір: <b>{html_escape(scope_title(ok_code, lvl))}</b>\n"
            f"Обрано блоків: <b>{len(selected)}</b>\n\n"
            "Натискайте блоки (⬜️/☑️), потім — <b>✅ Почати</b>."
        )
        await call.message.edit_text(
            title,
            parse_mode=ParseMode.HTML,
            reply_markup=kb_topics("train", ok_code, lvl, page=0, selected=selected),
        )
        await call.answer()
        return

    if kind == "multi":
        ok_codes = await db_get_ok_prefs(DB_POOL, tg_id, "train")
        ok_codes = {c for c in ok_codes if c}
        if not ok_codes:
            await call.answer("Оберіть модулі (ОК) спочатку.", show_alert=True)
            return

        available = set(multi_topics_for_ok_set(ok_codes))
        selected = await db_get_topic_prefs(DB_POOL, tg_id, "train", MULTI_OK_CODE, MULTI_OK_LEVEL)
        selected = {t for t in selected if t in available}
        await db_set_topic_prefs(DB_POOL, tg_id, "train", MULTI_OK_CODE, MULTI_OK_LEVEL, selected)

        shown = ", ".join(sorted(ok_codes))
        await call.message.edit_text(
            f"Обрані модулі: <b>{html_escape(shown)}</b>\n"
            f"Оберіть теми для тренування:\n"
            f"Обрано тем: <b>{len(selected)}</b>",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_multi_topics("train", ok_codes, page=0, selected=selected),
        )
        await call.answer()
        return

    await call.answer()



@router.callback_query(TopicPageCb.filter())
async def topic_page(call: CallbackQuery, callback_data: TopicPageCb) -> None:
    if not DB_POOL:
        return
    tg_id = call.from_user.id

    ok_code = str(callback_data.ok_code)
    lvl = int(callback_data.level)
    mode = str(callback_data.mode)
    page = int(callback_data.page)

    selected = await db_get_topic_prefs(DB_POOL, tg_id, mode, ok_code, lvl)

    title = (
        f"Оберіть <b>декілька</b> блоків для <b>{'навчання' if mode=='train' else 'екзамену'}</b>\n"
        f"Набір: <b>{html_escape(scope_title(ok_code, lvl))}</b>\n"
        f"Обрано блоків: <b>{len(selected)}</b>\n\n"
        "Натискайте блоки (⬜️/☑️), потім — <b>✅ Почати</b>."
    )

    await call.message.edit_text(
        title,
        parse_mode=ParseMode.HTML,
        reply_markup=kb_topics(mode, ok_code, lvl, page=page, selected=selected),
    )
    await call.answer()

@router.callback_query(TopicToggleCb.filter())
async def topic_toggle(call: CallbackQuery, callback_data: TopicToggleCb) -> None:
    if not DB_POOL:
        return
    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user:
        await call.answer("Немає профілю", show_alert=True)
        return
    if not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    ok_code = str(callback_data.ok_code)
    lvl = int(callback_data.level)
    mode = str(callback_data.mode)
    idx = int(callback_data.topic_idx)
    page = int(callback_data.page)

    topics = effective_topics(ok_code, lvl)
    if idx < 0 or idx >= len(topics):
        await call.answer("Невірний блок", show_alert=True)
        return

    topic = topics[idx]
    selected = await db_get_topic_prefs(DB_POOL, tg_id, mode, ok_code, lvl)

    if topic in selected:
        selected.remove(topic)
    else:
        selected.add(topic)

    await db_set_topic_prefs(DB_POOL, tg_id, mode, ok_code, lvl, selected)

    title = (
        f"Оберіть <b>декілька</b> блоків для <b>{'навчання' if mode=='train' else 'екзамену'}</b>\n"
        f"Набір: <b>{html_escape(scope_title(ok_code, lvl))}</b>\n"
        f"Обрано блоків: <b>{len(selected)}</b>\n\n"
        "Натискайте блоки (⬜️/☑️), потім — <b>✅ Почати</b>."
    )

    await call.message.edit_text(
        title,
        parse_mode=ParseMode.HTML,
        reply_markup=kb_topics(mode, ok_code, lvl, page=page, selected=selected),
    )
    await call.answer()

@router.callback_query(TopicClearCb.filter())
async def topic_clear(call: CallbackQuery, callback_data: TopicClearCb) -> None:
    if not DB_POOL:
        return
    tg_id = call.from_user.id
    ok_code = str(callback_data.ok_code)
    lvl = int(callback_data.level)
    mode = str(callback_data.mode)
    page = int(callback_data.page)

    await db_clear_topic_prefs(DB_POOL, tg_id, mode, ok_code, lvl)

    title = (
        f"Оберіть <b>декілька</b> блоків для <b>{'навчання' if mode=='train' else 'екзамену'}</b>\n"
        f"Набір: <b>{html_escape(scope_title(ok_code, lvl))}</b>\n"
        "Обрано блоків: <b>0</b>\n\n"
        "Натискайте блоки (⬜️/☑️), потім — <b>✅ Почати</b>."
    )

    await call.message.edit_text(
        title,
        parse_mode=ParseMode.HTML,
        reply_markup=kb_topics(mode, ok_code, lvl, page=page, selected=set()),
    )
    await call.answer("Очищено")
@router.callback_query(TopicDoneCb.filter())
async def topic_done(call: CallbackQuery, callback_data: TopicDoneCb) -> None:
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)

    if not user:
        await call.answer("Немає профілю", show_alert=True)
        return

    if not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    ok_code = str(callback_data.ok_code)
    lvl = int(callback_data.level)
    mode = str(callback_data.mode)

    selected = await db_get_topic_prefs(DB_POOL, tg_id, mode, ok_code, lvl)
    if not selected:
        await call.answer("Оберіть хоча б 1 блок або натисніть «Всі блоки».", show_alert=True)
        return

    pool_set: Set[int] = set()
    for t in selected:
        base = base_qids_for_topic(ok_code, lvl, t)
        pool_set.update(base)

    pool_qids = effective_qids(list(pool_set))
    if not pool_qids:
        await call.answer("У вибраних блоках немає питань.", show_alert=True)
        return

    await call.answer()

    # ✅ прибираємо клавіатуру під повідомленням (без нового тексту)
    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    await start_session_for_pool(
        call.bot,
        tg_id,
        call.message.chat.id,
        user,
        mode,
        pool_qids,
    )


@router.callback_query(TopicAllCb.filter())
async def topic_all(call: CallbackQuery, callback_data: TopicAllCb) -> None:
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user:
        await call.answer("Немає профілю", show_alert=True)
        return

    if not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    ok_code = str(callback_data.ok_code)
    lvl = int(callback_data.level)
    mode = str(callback_data.mode)

    base = base_qids_for_scope(ok_code, lvl)
    pool_qids = effective_qids(base)

    if not pool_qids:
        await call.answer("Для цього ОК немає питань.", show_alert=True)
        return

    await call.answer()

    # ✅ Прибрати старі кнопки (щоб не залишалися після натискання)
    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    if mode == "train":
        await start_session_for_pool(
            call.bot,
            tg_id,
            call.message.chat.id,
            user,
            "train",
            pool_qids,
            edit_message=call.message  # ✅ передано edit_message
        )
    elif mode == "exam":
        if len(pool_qids) < EXAM_QUESTIONS:
            await call.message.answer(
                f"Для цього набору доступно лише <b>{len(pool_qids)}</b> питань.\n"
                f"Екзамен потребує <b>{EXAM_QUESTIONS}</b>.\n"
                "Оберіть інший блок/рівень або додайте питання.",
                parse_mode=ParseMode.HTML
            )
            return

        exam_qids = random.sample(pool_qids, EXAM_QUESTIONS)
        await start_exam_session(
            call.bot,
            tg_id,
            call.message.chat.id,
            user,
            exam_qids,
            edit_message=call.message  # ✅ передано edit_message
        )



# Назад до екрану старту (Навчання/Екзамен) з inline-вибору тем
@router.callback_query(F.data.startswith("back:"))
async def back_to_mode_pick(call: CallbackQuery) -> None:
    if not DB_POOL:
        return
    mode = (call.data or "").split(":", 1)[-1]
    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user or not user_has_scope(user):
        await call.answer("Спочатку налаштуйте ОК", show_alert=True)
        return
    ok_code, lvl = get_user_scope(user)

    if mode == "train":
        await call.message.edit_text(
            f"Навчання для: <b>{html_escape(scope_title(ok_code, lvl))}</b>\nОберіть варіант:",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_train_pick(ok_code, lvl),
        )
    else:
        await call.message.edit_text(
            f"Екзамен для: <b>{html_escape(scope_title(ok_code, lvl))}</b>\n"
            f"Питань: <b>{EXAM_QUESTIONS}</b>, час: <b>{EXAM_DURATION_MINUTES} хв</b>\n"
            "Правильні відповіді не показуються.",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_exam_pick(ok_code, lvl),
        )

    await call.answer()

# В меню з inline-клавіатур
@router.callback_query(F.data == "menu")
async def menu_from_inline(call: CallbackQuery) -> None:
    if not DB_POOL:
        await call.answer()
        return

    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)

    try:
        await call.message.edit_text(
            MAIN_MENU_TEXT,
            reply_markup=kb_main_menu(is_admin=bool(user and user["is_admin"])),
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        # якщо текст такий самий або повідомлення не можна редагувати — просто оновимо клавіатуру
        try:
            await call.message.edit_reply_markup(
                reply_markup=kb_main_menu(is_admin=bool(user and user["is_admin"]))
            )
        except Exception:
            pass

    await call.answer()


# Сумісність: якщо залишились старі кнопки (одиночний вибір блоку)
@router.callback_query(TopicPickCb.filter())
async def topic_pick(call: CallbackQuery, callback_data: TopicPickCb) -> None:
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user:
        await call.answer("Немає профілю", show_alert=True)
        return
    if not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    ok_code = str(callback_data.ok_code)
    lvl = int(callback_data.level)
    mode = str(callback_data.mode)
    idx = int(callback_data.topic_idx)

    topics = effective_topics(ok_code, lvl)
    if idx < 0 or idx >= len(topics):
        await call.answer("Невірний блок", show_alert=True)
        return
    topic = topics[idx]

    # збережемо як "вибрано один блок"
    await db_set_topic_prefs(DB_POOL, tg_id, mode, ok_code, lvl, {topic})

    base = base_qids_for_topic(ok_code, lvl, topic)
    pool_qids = effective_qids(base)

    # ✅ прибрати кнопки вибору (без нового повідомлення)
    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    await call.answer()

    if mode == "train":
        pool_size = len(pool_qids)
        title = (
            f"Навчання • <b>{html_escape(scope_title(ok_code, lvl))}</b>\n"
            f"Блок: <b>{html_escape(topic)}</b>\n"
            "Як сформувати питання?"
        )
        await call.message.edit_text(
            title,
            parse_mode=ParseMode.HTML,
            reply_markup=kb_train_question_set("topics", ok_code, lvl, pool_size),
        )
        return
    else:
        await start_session_for_pool(
            call.bot, tg_id, call.message.chat.id, user, mode, pool_qids,
            edit_message=call.message,
        )



@router.callback_query(StartScopeCb.filter())
async def start_scope(call: CallbackQuery, callback_data: StartScopeCb) -> None:
    if not DB_POOL:
        await call.answer()
        return

    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)

    if not user:
        await call.answer("Немає профілю", show_alert=True)
        return
    if not await db_has_access(user):
        await call.answer("Доступ завершився", show_alert=True)
        return

    ok_code = str(callback_data.ok_code)
    lvl = int(callback_data.level)
    mode = str(callback_data.mode)

    pool_qids = effective_qids(base_qids_for_scope(ok_code, lvl))
    await call.answer()

    if mode == "train":
        pool_size = len(pool_qids)
        title = f"Навчання • <b>{html_escape(scope_title(ok_code, lvl))}</b>\nЯк сформувати питання?"
        await call.message.edit_text(
            title,
            parse_mode=ParseMode.HTML,
            reply_markup=kb_train_question_set("scope", ok_code, lvl, pool_size),
        )
        return

    # exam як було
    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    await start_session_for_pool(call.bot, tg_id, call.message.chat.id, user, mode, pool_qids, edit_message=call.message)



# -------------------------
# Навчання/екзамен: відповіді
# -------------------------

@router.callback_query(NextCb.filter())
async def on_next_after_feedback(call: CallbackQuery, callback_data: NextCb) -> None:
    """Переходимо до наступного питання після того, як показали правильну відповідь."""
    if not DB_POOL:
        return
    tg_id = call.from_user.id
    mode = str(callback_data.mode)

    sess = await db_get_active_session(DB_POOL, tg_id, mode)
    if not sess:
        await call.answer("Немає активної сесії.", show_alert=True)
        return

    # Захист від старих кнопок
    expected = int(callback_data.expected_index)
    if int(sess["current_index"]) != expected:
        await call.answer("Вже відкрито інше питання.", show_alert=False)
    else:
        await call.answer()

    await send_current_question(call.bot, DB_POOL, call.message.chat.id, tg_id, mode, edit_message=call.message)

@router.callback_query(SkipCb.filter())
async def on_skip(call: CallbackQuery, callback_data: SkipCb) -> None:
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    sess = await db_get_active_session(DB_POOL, tg_id, "train")
    if not sess:
        await call.answer("Немає активного навчання.", show_alert=True)
        return

    qids = [int(x) for x in json.loads(sess["question_ids"])]
    idx0 = int(sess["current_index"])
    if idx0 >= len(qids):
        await call.answer()
        return

    expected_qid = int(qids[idx0])
    if int(callback_data.qid) != expected_qid:
        await call.answer("Це старе питання.", show_alert=False)
        return

    # ✅ Пропуск = переносимо поточне питання в кінець черги, щоб повернулось після інших
    cur = qids.pop(idx0)
    qids.append(cur)

    # current_index НЕ збільшуємо: після pop() наступне питання стало на місце idx0
    await db_defer_question_to_end(DB_POOL, sess["session_id"], qids, idx0, skipped_delta=1)

    await db_stats_add(DB_POOL, tg_id, "train", skipped=1)

    await call.answer("⏭ Пропущено (повернеться в кінці)")
    await send_current_question(
        call.bot,
        DB_POOL,
        call.message.chat.id,
        tg_id,
        "train",
        edit_message=call.message,
    )

async def db_defer_question_to_end(
    pool: asyncpg.Pool,
    session_id: uuid.UUID,
    new_qids: List[int],
    current_index: int,
    skipped_delta: int = 1,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE sessions
            SET question_ids=$2,
                current_index=$3,
                skipped_count=skipped_count+$4
            WHERE session_id=$1
            """,
            session_id,
            json.dumps(new_qids),
            current_index,
            skipped_delta,
        )

@router.callback_query(AnswerCb.filter())
async def on_answer(call: CallbackQuery, callback_data: AnswerCb) -> None:
    if not DB_POOL:
        return

    tg_id = call.from_user.id
    mode = str(callback_data.mode)
    if mode not in ("train", "exam"):
        await call.answer()
        return

    sess = await db_get_active_session(DB_POOL, tg_id, mode)
    if not sess:
        await call.answer("Немає активної сесії.", show_alert=True)
        return

    if mode == "exam" and sess["expires_at"] and sess["expires_at"] <= utcnow():
        await finish_exam_due_to_timeout(call.bot, DB_POOL, tg_id, call.message.chat.id, sess)
        await call.answer("Час вийшов", show_alert=True)
        return

    qids = json.loads(sess["question_ids"])
    qids = [int(x) for x in qids]
    idx0 = int(sess["current_index"])
    if idx0 >= len(qids):
        await call.answer()
        return

    expected_qid = int(qids[idx0])
    if int(callback_data.qid) != expected_qid:
        await call.answer("Це старе питання.", show_alert=False)
        return

    q = QUESTIONS_BY_ID.get(expected_qid)
    if not q:
        await call.answer("Питання не знайдено.", show_alert=True)
        await db_update_session_progress(DB_POOL, sess["session_id"], idx0 + 1, skipped_delta=1)
        await db_stats_add(DB_POOL, tg_id, mode, skipped=1)
        await send_current_question(call.bot, DB_POOL, call.message.chat.id, tg_id, mode, edit_message=call.message)
        return

    chosen = int(callback_data.ci)
    correct_idx = int((q.get("correct") or [None])[0]) if is_question_valid(q) else None
    is_correct = (correct_idx is not None and chosen == correct_idx)

    # оновлюємо прогрес (відповідь завжди рахується як крок)
    await db_update_session_progress(
        DB_POOL,
        sess["session_id"],
        idx0 + 1,
        correct_delta=(1 if is_correct else 0),
        wrong_delta=(0 if is_correct else 1),
    )
    await db_stats_add(
        DB_POOL, tg_id, mode,
        answered=1,
        correct=(1 if is_correct else 0),
        wrong=(0 if is_correct else 1),
    )

    # Екзамен: без фідбеку, одразу наступне питання
    if mode == "exam":
        await call.answer("✅ Відповідь зараховано", show_alert=False)
        await send_current_question(call.bot, DB_POOL, call.message.chat.id, tg_id, "exam", edit_message=call.message)
        return

    # Навчання: якщо правильно — одразу наступне питання
    if is_correct:
        await call.answer("✅ Правильно", show_alert=False)
        await send_current_question(call.bot, DB_POOL, call.message.chat.id, tg_id, "train", edit_message=call.message)
        return

    # -------- Навчання: невірно — показуємо ✅/❌ у варіантах + підсумок --------
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = q.get("choices") or []

    # Безпечні лейбли
    chosen_label = letters[chosen] if 0 <= chosen < len(letters) else str(chosen + 1)
    corr_label = (
        letters[correct_idx]
        if (correct_idx is not None and 0 <= correct_idx < len(letters))
        else str((correct_idx or 0) + 1)
    )

    chosen_text = ""
    if 0 <= chosen < len(choices):
        chosen_text = html_escape(str(choices[chosen]))

    corr_text = ""
    if correct_idx is not None and 0 <= correct_idx < len(choices):
        corr_text = html_escape(str(choices[correct_idx]))

    # Текст питання/шапка
    qtext = html_escape(str(q.get("question") or ""))
    remaining_q = max(0, len(qids) - (idx0 + 1))
    head = f"📚 <b>Навчання</b> • Питання <b>{idx0 + 1}/{len(qids)}</b> • Залишилось <b>{remaining_q}</b>"

    # Відображення варіантів з мітками
    options_block = "🧾 <b>Варіанти відповіді:</b>\n"
    for i, ch in enumerate(choices):
        label = letters[i] if i < len(letters) else str(i + 1)
        text = html_escape(str(ch))

        if correct_idx is not None and i == correct_idx:
            mark = "✅"
        elif i == chosen:
            mark = "❌"
        else:
            mark = "▫️"

        options_block += f"{mark} <b>{label}</b> — {text}\n"

    # Підсумок окремим блоком
    result_block = (
        "────────────\n"
        "❌ <b>Неправильно</b>\n"
        "<i>Правильний варіант позначено ✅ вище.</i>"
    )

    qa_sep = "────────────\n"

    shown = (
        f"{head}\n\n"
        f"❓ <b>Питання:</b>\n<b>{qtext}</b>\n"
        f"{qa_sep}"
        f"{options_block}\n"
        f"{result_block}"
    )

    try:
        await call.message.edit_text(
            shown,
            reply_markup=kb_after_feedback(mode="train", expected_index=idx0 + 1),
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        await call.message.answer(shown, parse_mode=ParseMode.HTML)

    await call.answer("❌ Неправильно", show_alert=False)


# -------------------------
# Адмінка
# -------------------------

@router.message(F.text == "🛠 Адмін")
@router.callback_query(F.data.startswith("ad:"))
async def admin_actions_inline(call: CallbackQuery) -> None:
    """Натискання в адмін-панелі (inline)."""
    if not DB_POOL:
        return
    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user or not user["is_admin"]:
        await call.answer("Немає доступу", show_alert=True)
        return

    action = (call.data or "").split(":", 1)[1] if ":" in (call.data or "") else ""

    if action == "users":
        rows = await db_list_users(DB_POOL, limit=30)
        out = "<b>👥 Останні користувачі</b>\n\n"
        for r in rows:
            out += f"<code>{r['tg_id']}</code>  "
            if r["phone"]:
                out += f"{html_escape(r['phone'])}  "
            if r["ok_code"]:
                oc = r["ok_code"]
                ol = r["ok_level"] if r["ok_level"] is not None else "-"
                out += f"[{html_escape(str(oc))}:{ol}]  "
            if r["is_admin"]:
                out += "🛠 "
            active = False
            now = utcnow()
            if r["is_admin"]:
                active = True
            elif r["sub_until"] and r["sub_until"] > now:
                active = True
            elif r["trial_until"] and r["trial_until"] > now:
                active = True
            out += "✅" if active else "⛔️"
            out += "\n"
        out += "\nКоманди:\n"
        out += "<code>/grant TG_ID DAYS</code> — додати підписку (днів)\n"
        out += "<code>/revoke TG_ID</code> — забрати підписку\n"
        out += "<code>/user TG_ID</code> — деталі по користувачу\n"
        out += "<code>/setscope TG_ID OK LEVEL   (LEVEL=-1 озна... — встановити ОК/рівень (OK=ОК-1.., або LAW; LEVEL=0 для LAW)\n"
        await call.message.answer(out, parse_mode=ParseMode.HTML)
        await call.answer()
        return

    if action == "problems":
        problem_ids = sorted(PROBLEM_IDS_FILE)
        out = "<b>⚠️ Проблемні питання</b>\n\n"
        out += f"З файлу: <b>{len(problem_ids)}</b>\n"
        out += f"Вимкнено в БД: <b>{len(DISABLED_IDS_DB)}</b>\n\n"
        out += "Натисніть ID, щоб увімкнути/вимкнути (показуємо перші 15):"
        b = InlineKeyboardBuilder()
        for qid in problem_ids[:15]:
            enabled = (qid not in DISABLED_IDS_DB)
            b.button(
                text=f"{qid} {'✅' if enabled else '⛔️'}",
                callback_data=AdminToggleQCb(qid=qid, enable=(0 if enabled else 1)),
            )
        b.adjust(3)
        await call.message.answer(out, parse_mode=ParseMode.HTML, reply_markup=b.as_markup())
        await call.answer()
        return

    await call.answer()

async def admin_entry(message: Message) -> None:
    if not DB_POOL:
        return
    tg_id = message.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user or not user["is_admin"]:
        await message.answer("⛔️ Немає доступу.")
        return
    await message.answer("Адмін-панель:", reply_markup=kb_admin_panel())

@router.message(F.text == "⬅️ Назад")
async def back_from_admin(message: Message) -> None:
    if not DB_POOL:
        return

    tg_id = message.from_user.id
    user = await db_get_user(DB_POOL, tg_id)

    await show_main_menu(message, is_admin=bool(user and user["is_admin"]))

@router.message(F.text == "👥 Користувачі")
async def admin_users(message: Message) -> None:
    if not DB_POOL:
        return
    tg_id = message.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user or not user["is_admin"]:
        await message.answer("⛔️ Немає доступу.")
        return
    rows = await db_list_users(DB_POOL, limit=30)
    out = "<b>👥 Останні користувачі</b>\n\n"
    for r in rows:
        out += f"<code>{r['tg_id']}</code>  "
        if r["phone"]:
            out += f"{html_escape(r['phone'])}  "
        if r["ok_code"]:
            oc = r["ok_code"]
            ol = r["ok_level"] if r["ok_level"] is not None else "-"
            out += f"[{html_escape(str(oc))}:{ol}]  "
        if r["is_admin"]:
            out += "🛠 "
        active = False
        now = utcnow()
        if r["is_admin"]:
            active = True
        elif r["sub_until"] and r["sub_until"] > now:
            active = True
        elif r["trial_until"] and r["trial_until"] > now:
            active = True
        out += "✅" if active else "⛔️"
        out += "\n"
    out += "\nКоманди:\n"
    out += "<code>/grant TG_ID DAYS</code> — додати підписку (днів)\n"
    out += "<code>/revoke TG_ID</code> — забрати підписку\n"
    out += "<code>/user TG_ID</code> — деталі по користувачу\n"
    out += "<code>/setscope TG_ID OK LEVEL   (LEVEL=-1 означає «всі рівні»)</code> — встановити ОК/рівень (OK=ОК-1.., або LAW; LEVEL=0 для LAW)\n"
    await message.answer(out, parse_mode=ParseMode.HTML)

@router.message(F.text == "⚠️ Проблемні питання")
async def admin_problem_questions(message: Message) -> None:
    if not DB_POOL:
        return
    tg_id = message.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user or not user["is_admin"]:
        await message.answer("⛔️ Немає доступу.")
        return

    problem_ids = sorted(PROBLEM_IDS_FILE)
    out = "<b>⚠️ Проблемні питання</b>\n\n"
    out += f"З файлу: <b>{len(problem_ids)}</b>\n"
    out += f"Вимкнено в БД: <b>{len(DISABLED_IDS_DB)}</b>\n\n"
    out += "Натисніть ID, щоб увімкнути/вимкнути (показуємо перші 15):"
    b = InlineKeyboardBuilder()
    for qid in problem_ids[:15]:
        enabled = (qid not in DISABLED_IDS_DB)
        b.button(
            text=f"{qid} {'✅' if enabled else '⛔️'}",
            callback_data=AdminToggleQCb(qid=qid, enable=(0 if enabled else 1)),
        )
    b.adjust(3)
    await message.answer(out, parse_mode=ParseMode.HTML, reply_markup=b.as_markup())

@router.callback_query(AdminToggleQCb.filter())
async def admin_toggle_question(call: CallbackQuery, callback_data: AdminToggleQCb) -> None:
    if not DB_POOL:
        return
    tg_id = call.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user or not user["is_admin"]:
        await call.answer("Немає доступу", show_alert=True)
        return

    qid = int(callback_data.qid)
    enable = bool(int(callback_data.enable))
    await db_toggle_question(DB_POOL, qid, enable=enable, note=f"admin:{tg_id}")

    global DISABLED_IDS_DB
    DISABLED_IDS_DB = await db_get_disabled_ids(DB_POOL)

    status = "увімкнено" if enable else "вимкнено"
    await call.answer(f"Питання {qid} {status}.")
    q = QUESTIONS_BY_ID.get(qid)
    if q:
        preview = "<b>Питання</b>\n"
        preview += f"ID: <code>{qid}</code>\n"
        preview += f"{html_escape(q.get('question',''))}\n"
        await call.message.answer(preview, parse_mode=ParseMode.HTML)

@router.message(Command("grant"))
async def cmd_grant(message: Message) -> None:
    if not DB_POOL:
        return
    tg_id = message.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user or not user["is_admin"]:
        return
    parts = (message.text or "").split()
    if len(parts) != 3:
        await message.answer("Формат: /grant TG_ID DAYS")
        return
    try:
        uid = int(parts[1]); days = int(parts[2])
    except ValueError:
        await message.answer("Помилка: TG_ID і DAYS мають бути числами.")
        return
    u2 = await db_set_sub_days(DB_POOL, uid, days)
    if not u2:
        await message.answer("Користувача не знайдено.")
        return
    await message.answer(
        f"✅ Ок. Підписка до: <b>{u2['sub_until'].astimezone(KYIV_TZ).strftime('%Y-%m-%d %H:%M Kyiv')}</b>",
        parse_mode=ParseMode.HTML,
    )

@router.message(Command("revoke"))
async def cmd_revoke(message: Message) -> None:
    if not DB_POOL:
        return
    tg_id = message.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user or not user["is_admin"]:
        return
    parts = (message.text or "").split()
    if len(parts) != 2:
        await message.answer("Формат: /revoke TG_ID")
        return
    try:
        uid = int(parts[1])
    except ValueError:
        await message.answer("Помилка: TG_ID має бути числом.")
        return
    u2 = await db_revoke_sub(DB_POOL, uid)
    if not u2:
        await message.answer("Користувача не знайдено.")
        return
    await message.answer("✅ Ок. Підписку знято.")

@router.message(Command("user"))
async def cmd_user(message: Message) -> None:
    if not DB_POOL:
        return
    tg_id = message.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user or not user["is_admin"]:
        return
    parts = (message.text or "").split()
    if len(parts) != 2:
        await message.answer("Формат: /user TG_ID")
        return
    try:
        uid = int(parts[1])
    except ValueError:
        await message.answer("Помилка: TG_ID має бути числом.")
        return
    u2 = await db_get_user(DB_POOL, uid)
    if not u2:
        await message.answer("Користувача не знайдено.")
        return
    out = f"<b>Користувач</b> <code>{uid}</code>\n"
    out += f"Телефон: {html_escape(u2['phone'] or '-')}\n"
    out += f"Trial до: {u2['trial_until'].astimezone(KYIV_TZ).strftime('%Y-%m-%d %H:%M Kyiv') if u2['trial_until'] else '-'}\n"
    out += f"Підписка до: {u2['sub_until'].astimezone(KYIV_TZ).strftime('%Y-%m-%d %H:%M Kyiv') if u2['sub_until'] else '-'}\n"
    if u2["ok_code"]:
        out += f"Набір: {html_escape(scope_title(str(u2['ok_code']), int(u2['ok_level'] or 0)))}\n"
    else:
        out += "Набір: -\n"
    out += f"Адмін: {'так' if u2['is_admin'] else 'ні'}\n"
    out += f"Остання активність: {u2['last_seen'].astimezone(KYIV_TZ).strftime('%Y-%m-%d %H:%M Kyiv')}\n"
    await message.answer(out, parse_mode=ParseMode.HTML)

@router.message(Command("setscope"))
async def cmd_setscope(message: Message) -> None:
    """
    /setscope TG_ID OK LEVEL   (LEVEL=-1 означає «всі рівні»)
    OK: ОК-1..ОК-17 або LAW
    LEVEL: 1..3 (або -1 для всіх рівнів; 0 для LAW)
    """
    if not DB_POOL:
        return
    tg_id = message.from_user.id
    user = await db_get_user(DB_POOL, tg_id)
    if not user or not user["is_admin"]:
        return

    parts = (message.text or "").split()
    if len(parts) != 4:
        await message.answer("Формат: /setscope TG_ID OK LEVEL   (LEVEL=-1 означає «всі рівні»)  (OK=ОК-1.. або LAW)")
        return
    try:
        uid = int(parts[1]); ok_code = parts[2].strip(); lvl = int(parts[3])
    except ValueError:
        await message.answer("Помилка: TG_ID і LEVEL мають бути числами.")
        return

    ok_code = OK_CODE_LAW if ok_code.upper() == "LAW" else ok_code

    if ok_code not in OK_CODES:
        await message.answer("Невідомий OK. Приклад: ОК-3 або LAW")
        return

    if ok_code == OK_CODE_LAW:
        lvl = 0
    else:
        if lvl != LEVEL_ALL and lvl not in LEVELS_BY_OK.get(ok_code, [1, 2, 3]):
            await message.answer("Невірний рівень для цього ОК.")
            return

    u2 = await db_set_scope(DB_POOL, uid, ok_code, lvl)
    await message.answer(
        f"✅ Ок. Встановлено: <b>{html_escape(scope_title(ok_code, lvl))}</b> для <code>{uid}</code>",
        parse_mode=ParseMode.HTML,
    )


# -------------------------
# Фоновий watchdog для таймера екзамену
# -------------------------

async def exam_watchdog(bot: Bot, pool: asyncpg.Pool, interval_sec: int = 30) -> None:
    while True:
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT session_id, tg_id, expires_at
                    FROM sessions
                    WHERE mode='exam' AND completed=FALSE AND expires_at IS NOT NULL AND expires_at <= $1
                    """,
                    utcnow()
                )
            for r in rows:
                tg_id = int(r["tg_id"])
                sess = await db_get_active_session(pool, tg_id, "exam")
                if sess and sess["session_id"] == r["session_id"]:
                    try:
                        # у приватному чаті chat_id == tg_id
                        await finish_exam_due_to_timeout(bot, pool, tg_id, tg_id, sess)
                    except Exception:
                        logging.exception("Failed to finish exam for %s", tg_id)
        except Exception:
            logging.exception("Watchdog error")
        await asyncio.sleep(interval_sec)


# -------------------------
# Startup / main
# -------------------------

async def on_startup(bot: Bot, dp: Dispatcher) -> None:
    global DB_POOL, DISABLED_IDS_DB

    if not BOT_TOKEN or not DATABASE_URL:
        raise RuntimeError("BOT_TOKEN або DATABASE_URL не задані.")

    load_question_bank()

    DB_POOL = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
    await db_init(DB_POOL)

    await db_seed_problem_flags(DB_POOL, PROBLEM_IDS_FILE)
    DISABLED_IDS_DB = await db_get_disabled_ids(DB_POOL)

    dp.workflow_data["exam_watchdog_task"] = asyncio.create_task(exam_watchdog(bot, DB_POOL))

    logging.info(
        "Startup done. Questions total=%d, valid=%d, problems=%d, disabled_db=%d, ok_codes=%d",
        len(QUESTIONS_BY_ID), len(VALID_QIDS), len(PROBLEM_IDS_FILE), len(DISABLED_IDS_DB), len(OK_CODES)
    )

async def on_shutdown(bot: Bot, dp: Dispatcher) -> None:
    task = dp.workflow_data.get("exam_watchdog_task")
    if task:
        task.cancel()
    if DB_POOL:
        await DB_POOL.close()

async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    bot = Bot(
        BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher()
    dp.include_router(router)

    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    await dp.start_polling(bot, dp=dp)

if __name__ == "__main__":
    asyncio.run(main())
