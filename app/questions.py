from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Set, Tuple

from .config import (
    QUESTIONS_FILE,
    PROBLEMS_FILE,
    OK_CODE_LAW,
    LEVEL_ALL,
    POSITION_OK_MAP,
)
from .utils import is_question_valid, normalize_level, normalize_ok_code
from .state import (
    QUESTIONS_BY_ID,
    VALID_QIDS,
    OK_CODES,
    LEVELS_BY_OK,
    TOPICS_BY_SCOPE,
    QIDS_BY_SCOPE,
    QIDS_BY_SCOPE_TOPIC,
    PROBLEM_IDS_FILE,
    DISABLED_IDS_DB,
)


# -------------------------
# Питання / кеші / індекси
# -------------------------
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



def load_question_bank() -> None:

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        qlist = json.load(f)

    PROBLEM_IDS_FILE.clear()
    if os.path.exists(PROBLEMS_FILE):
        with open(PROBLEMS_FILE, "r", encoding="utf-8") as f:
            prob = json.load(f)
        for item in prob.get("items", []):
            try:
                PROBLEM_IDS_FILE.add(int(item["id"]))
            except Exception:
                pass

    QUESTIONS_BY_ID = {int(q["id"]): q for q in qlist}

    VALID_QIDS.clear()
    LEVELS_BY_OK.clear()
    TOPICS_BY_SCOPE.clear()
    QIDS_BY_SCOPE.clear()
    QIDS_BY_SCOPE_TOPIC.clear()

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

    OK_CODES.clear()
    OK_CODES.extend(sorted(LEVELS_BY_OK.keys(), key=lambda x: (x != OK_CODE_LAW, x)))

    # set -> list
    _tmp = {k: sorted(list(v)) for k, v in LEVELS_BY_OK.items()}
    LEVELS_BY_OK.clear()
    LEVELS_BY_OK.update(_tmp)
    _tmp = {k: sorted(list(v)) for k, v in TOPICS_BY_SCOPE.items()}
    TOPICS_BY_SCOPE.clear()
    TOPICS_BY_SCOPE.update(_tmp)


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



