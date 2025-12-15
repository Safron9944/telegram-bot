from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from aiogram.enums import ParseMode
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

from .config import TRAIN_QUESTIONS, EXAM_QUESTIONS, OK_CODE_LAW, LEVEL_ALL, POSITIONS, pos_id
from .state import DB_POOL, OK_CODES, LEVELS_BY_OK
from .callbacks import (
    MULTI_OK_CODE,
    MULTI_OK_LEVEL,
    AnswerCb,
    SkipCb,
    NextCb,
    AdminToggleQCb,
    OkPickCb,
    OkPageCb,
    OkMultiPageCb,
    OkToggleCb,
    OkDoneCb,
    OkClearCb,
    OkAllCb,
    StartMultiOkCb,
    LevelPickCb,
    StartScopeCb,
    TopicPageCb,
    TopicPickCb,
    TopicToggleCb,
    TopicDoneCb,
    TopicClearCb,
    TopicAllCb,
    TopicBackCb,
    MultiTopicsPageCb,
    MultiTopicToggleCb,
    MultiTopicDoneCb,
    MultiTopicClearCb,
    MultiTopicAllCb,
)
from .questions import (
    effective_qids,
    effective_topics,
    levels_for_ok,
    base_qids_for_scope,
    base_qids_for_topic,
    db_get_ok_prefs,
)
from .db import (
    db_get_topic_prefs,
    db_set_topic_prefs,
    db_clear_topic_prefs,
)

def multi_topics_for_ok_set(ok_codes: Set[str]) -> List[str]:
    out: List[str] = []
    ordered = sorted(ok_codes, key=lambda x: (x != OK_CODE_LAW, x))  # LAW першим
    for ok in ordered:
        if ok == OK_CODE_LAW:
            out.append("📜 Законодавство")
            continue
        for t in effective_topics(ok, LEVEL_ALL):
            out.append(f"{ok} • {t}")
    return out

def qids_for_multi_topic_label(label: str) -> List[int]:
    if label == "📜 Законодавство":
        return base_qids_for_scope(OK_CODE_LAW, 0)
    if " • " not in label:
        return []
    ok_code, topic = label.split(" • ", 1)
    return base_qids_for_topic(ok_code, LEVEL_ALL, topic)

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

    b.row(
        InlineKeyboardButton(text="⬅️ Назад", callback_data=OkDoneCb(mode=mode).pack()),
        InlineKeyboardButton(text="🎯 Всі теми", callback_data=MultiTopicAllCb(mode=mode).pack()),
        InlineKeyboardButton(text=start_label, callback_data=MultiTopicDoneCb(mode=mode).pack()),
        InlineKeyboardButton(text="🏠 Меню", callback_data="menu"),
    )
    b.row(
        InlineKeyboardButton(text="🧹 Очистити", callback_data=MultiTopicClearCb(mode=mode, page=page).pack()),
        InlineKeyboardButton(text="🔁 Змінити модулі", callback_data=OkMultiPageCb(mode=mode, page=0).pack()),
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
    selected = await db_get_topic_prefs(DB_POOL, tg_id, mode, MULTI_OK_CODE, MULTI_OK_LEVEL)
    selected = {t for t in selected if t in available}
    if not selected:
        await call.answer("Оберіть хоча б одну тему", show_alert=True)
        return

    pool: List[int] = []
    for label in selected:
        pool.extend(qids_for_multi_topic_label(label))
    pool_qids = effective_qids(list(dict.fromkeys(pool)))

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
                InlineKeyboardButton(text="⚙️ Налаштування", callback_data="mm:settings"),
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
    per_page: int = 9,
) -> InlineKeyboardMarkup:
    selected_set: Set[str] = set(selected or [])
    codes = [OK_CODE_LAW] + [c for c in OK_CODES if c != OK_CODE_LAW]
    pages: List[List[str]] = [codes[i:i + per_page] for i in range(0, len(codes), per_page)]
    if not pages:
        pages = [[]]
    page = max(0, min(page, len(pages) - 1))
    current = pages[page]

    b = InlineKeyboardBuilder()
    for c in current:
        label = "📜 Законодавство" if c == OK_CODE_LAW else c
        mark = "☑️" if c in selected_set else "⬜️"
        b.button(
            text=f"{mark} {label}",
            callback_data=OkToggleCb(mode=mode, ok_code=c, page=page).pack(),
        )
    b.adjust(1)

    nav: List[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="⬅️", callback_data=OkMultiPageCb(mode=mode, page=page - 1).pack()))
    if page < len(pages) - 1:
        nav.append(InlineKeyboardButton(text="➡️", callback_data=OkMultiPageCb(mode=mode, page=page + 1).pack()))
    if nav:
        b.row(*nav)

    b.row(
        InlineKeyboardButton(text="🧹 Очистити", callback_data=OkClearCb(mode=mode, page=page).pack()),
        InlineKeyboardButton(text="🎯 Всі ОК", callback_data=OkAllCb(mode=mode).pack()),
    )
    b.row(
        InlineKeyboardButton(text="✅ Готово", callback_data=OkDoneCb(mode=mode).pack()),
        InlineKeyboardButton(text="🏠 Меню", callback_data="menu"),
    )
    return b.as_markup()

def kb_train_pick_multi(mode: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()

    if mode == "train":
        b.button(
            text="✅ Всі теми (по всіх модулях)",
            callback_data=StartMultiOkCb(mode=mode).pack(),
        )
        b.button(
            text="📚 Обрати теми",
            callback_data=MultiTopicsPageCb(mode=mode, page=0).pack(),
        )
    else:
        # для екзамену лишаємо простий старт по всіх обраних блоках
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
# Позиції
# -------------------------

def _short_mode(mode: str) -> str:
    """'train' -> 't', 'exam' -> 'e', інше лишає як є."""
    mode = str(mode)
    if mode == "train":
        return "t"
    if mode == "exam":
        return "e"
    return mode


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
