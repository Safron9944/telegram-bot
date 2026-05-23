from __future__ import annotations

import re
from difflib import SequenceMatcher
from html import escape as hescape
from typing import Any, Dict, List, Optional, Tuple

from aiogram import Bot, F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from access import access_status
from questions import Q, QuestionBank
from storage import Storage
from ui import fmt_access_line, kb_inline, render_main, screen_main_menu
from utils import (
    ADMIN_PANEL_CHAT_ID,
    ADMIN_PANEL_MSG_ID,
    ADMIN_QEDIT,
    ADMIN_QWORK_AWAITING,
    ADMIN_QWORK_PAGE,
    ADMIN_QWORK_QUERY,
    clamp_callback,
    now,
)
from handlers.session import show_next_in_session

router = Router()


def _admin_user_icon(u: Dict[str, Any]) -> str:
    t_end = u.get("trial_end")
    s_end = u.get("sub_end")
    inf = bool(u.get("sub_infinite"))
    n = now()

    if inf or (s_end and n <= s_end):
        return "🟢"
    if t_end and n <= t_end:
        return "🟡"
    return "🔴"


def fmt_user_row(u: Dict[str, Any]) -> str:
    uid = u.get("user_id")
    fn = (u.get("first_name") or "").strip()
    ln = (u.get("last_name") or "").strip()
    full = " ".join([x for x in [fn, ln] if x]).strip() or "—"
    return f"{_admin_user_icon(u)} {uid} | {full}"


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
                return True
            return False
        except Exception:
            return False

    if message:
        st[ADMIN_PANEL_MSG_ID] = message.message_id
        st[ADMIN_PANEL_CHAT_ID] = message.chat.id
        await store.set_state(uid, st)

        if await _try_edit(message.chat.id, message.message_id):
            return

    msg_id = st.get(ADMIN_PANEL_MSG_ID)
    c_id = st.get(ADMIN_PANEL_CHAT_ID) or chat_id

    if msg_id and await _try_edit(c_id, msg_id):
        return

    sent = await bot.send_message(chat_id, text, reply_markup=kb, parse_mode="HTML")
    st[ADMIN_PANEL_MSG_ID] = sent.message_id
    st[ADMIN_PANEL_CHAT_ID] = chat_id
    await store.set_state(uid, st)


async def render_admin_qedit(
        bot: Bot,
        store: "Storage",
        uid: int,
        fallback_chat_id: int,
        text: str,
        keyboard: InlineKeyboardMarkup,
):
    ui = await store.get_ui(uid) or {}
    st = (ui.get("state", {}) or {})
    qedit = st.get(ADMIN_QEDIT) or {}

    chat_id = int(qedit.get("chat_id") or ui.get("chat_id") or fallback_chat_id)
    msg_id = qedit.get("msg_id")

    if msg_id:
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=int(msg_id),
                text=text,
                reply_markup=keyboard,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            await store.set_ui(uid, chat_id, int(msg_id))
            return
        except TelegramBadRequest:
            pass

    await render_main(bot, store, uid, chat_id, text, keyboard, message=None)


async def render_admin_users_list(
        bot: Bot,
        store: "Storage",
        admin_uid: int,
        chat_id: int,
        offset: int,
        message: Optional[Message] = None,
):
    limit = 10

    items = await store.list_users(offset, limit + 1)

    has_next = len(items) > limit
    users = items[:limit]

    c_green = c_yellow = c_red = 0
    for u in users:
        ic = _admin_user_icon(u)
        if ic == "🟢":
            c_green += 1
        elif ic == "🟡":
            c_yellow += 1
        else:
            c_red += 1

    text = (
        "🛠 <b>Користувачі</b>\n"
        "🟢 підписка | 🟡 тріал | 🔴 без доступу\n"
        f"У цьому списку: 🟢{c_green} 🟡{c_yellow} 🔴{c_red}\n\n"
        "Оберіть користувача:"
    )

    rows: list[list[InlineKeyboardButton]] = []

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

    rows.append([InlineKeyboardButton(text="⬅️ Меню", callback_data=clamp_callback("nav:menu"))])

    kb = InlineKeyboardMarkup(inline_keyboard=rows)
    await render_admin_view(bot, store, admin_uid, chat_id, text, kb, message=message)


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

    first_html = hescape(user.get("first_name") or "—")
    last_html = hescape(user.get("last_name") or "—")

    text = (
        "👤 <b>Користувач</b>\n\n"
        f"ID: <b>{target_id}</b>\n"
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


@router.callback_query(F.data == "noop")
async def noop(cb: CallbackQuery):
    await cb.answer()


# -------------------- Admin: questions editing --------------------

def _q_snip(s: str, max_len: int = 46) -> str:
    s = (s or "").replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 1)] + "…"


def screen_admin_qwork(qb: "QuestionBank", page: int = 0, page_size: int = 10) -> Tuple[str, InlineKeyboardMarkup]:
    ids = sorted(list(qb.by_id.keys()))
    total = len(ids)
    if total == 0:
        text = "✏️ <b>Робота над питаннями</b>\n\nПитань не знайдено."
        kb = kb_inline([("⬅️ Меню", "nav:menu")], row=1)
        return text, kb

    page = max(0, int(page))
    page_size = max(5, int(page_size))
    pages = (total + page_size - 1) // page_size
    if page >= pages:
        page = pages - 1

    start = page * page_size
    end = min(total, start + page_size)
    chunk = ids[start:end]

    text = (
        "✏️ <b>Робота над питаннями</b>\n"
        f"Сторінка <b>{page + 1}</b>/<b>{pages}</b> • Питань: <b>{total}</b>\n\n"
        "Натисни на питання, щоб відкрити редагування, або введи назву/частину питання (кнопка пошуку)."
    )

    b = InlineKeyboardBuilder()
    for qid in chunk:
        q = qb.by_id.get(qid)
        title = f"{qid} • {_q_snip(q.question if q else '')}"
        b.row(
            InlineKeyboardButton(
                text=title,
                callback_data=clamp_callback(f"admin:qedit:{int(qid)}:qwork:{int(page)}"),
            )
        )

    nav_row: list[InlineKeyboardButton] = []
    if page > 0:
        nav_row.append(
            InlineKeyboardButton(text="⬅️ Попер...едня", callback_data=clamp_callback(f"admin:qwork:{page - 1}")))
    if page + 1 < pages:
        nav_row.append(
            InlineKeyboardButton(text="Наступна ➡️", callback_data=clamp_callback(f"admin:qwork:{page + 1}")))
    if nav_row:
        b.row(*nav_row)

    b.row(InlineKeyboardButton(text="🔎 Пошук по назві", callback_data=clamp_callback(f"admin:qwork_find:{int(page)}")))
    b.row(InlineKeyboardButton(text="⬅️ Меню", callback_data="nav:menu"))
    return text, b.as_markup()


def screen_admin_qwork_find(page: int = 0, error: Optional[str] = None) -> Tuple[str, InlineKeyboardMarkup]:
    page = max(0, int(page))
    text = (
        "🔎 <b>Пошук питання за назвою</b>\n\n"
        "Надішли <b>частину тексту питання</b> одним повідомленням.\n"
        "Напр.: <code>строк дії</code> або <code>хто має право</code>"
    )
    if error:
        text += f"\n\n❗️ {hescape(error)}"
    kb = kb_inline([("⬅️ Назад", f"admin:qwork:{page}")], row=1)
    return text, kb


def screen_admin_qwork_results(qb: "QuestionBank", query: str, page: int = 0, limit: int = 12) -> Tuple[
    str, InlineKeyboardMarkup]:
    page = max(0, int(page))
    query = (query or "").strip()
    qids = find_question_ids_by_title(qb, query, limit=limit)

    if not qids:
        return screen_admin_qwork_find(page=page, error="Нічого схожого не знайшов. Спробуй інші слова.")

    text = (
        "🔎 <b>Знайдені схожі питання</b>\n\n"
        f"Запит: <code>{hescape(query)}</code>\n"
        f"Показую: <b>{len(qids)}</b>\n\n"
        "Обери потрібне:"
    )

    b = InlineKeyboardBuilder()
    for qid in qids:
        q = qb.by_id.get(int(qid))
        title = f"{qid} • {_q_snip(q.question if q else '')}"
        b.row(
            InlineKeyboardButton(
                text=title,
                callback_data=clamp_callback(f"admin:qedit:{int(qid)}:qsearch:{int(page)}"),
            )
        )

    b.row(InlineKeyboardButton(text="🔎 Новий пошук", callback_data=clamp_callback(f"admin:qwork_find:{int(page)}")))
    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data=clamp_callback(f"admin:qwork:{int(page)}")))
    return text, b.as_markup()


def _fmt_q_choices(q: Q) -> str:
    lines = []
    for i, c in enumerate(q.choices or []):
        lines.append(f"<b>{i + 1}.</b> {hescape(c)}")
    return "\n".join(lines) if lines else "—"


def screen_admin_qedit(q: Q, note: str = "") -> Tuple[str, InlineKeyboardMarkup]:
    corr = ", ".join(str(int(x)) for x in (q.correct or [])) or "—"
    corr_texts = "; ".join(hescape(x) for x in (q.correct_texts or [])) or "—"

    head = "✏️ <b>Редагування питання</b>"
    if note:
        head += f"\n{note}"

    meta = []
    if q.ok:
        meta.append(f"OK: <b>{hescape(q.ok)}</b>")
    if q.level is not None:
        meta.append(f"Рівень: <b>{int(q.level)}</b>")
    if q.qnum is not None:
        meta.append(f"№: <b>{int(q.qnum)}</b>")
    meta_line = ("\n" + " • ".join(meta)) if meta else ""

    text = (
        f"{head}\n"
        f"ID: <code>{int(q.id)}</code>{meta_line}\n\n"
        f"<b>Питання:</b>\n{hescape(q.question)}\n\n"
        f"<b>Варіанти:</b>\n{_fmt_q_choices(q)}\n\n"
        f"<b>Правильні:</b> <code>{hescape(corr)}</code>\n"
        f"<b>Тексти правильних:</b> {corr_texts}"
    )

    kb = kb_inline(
        [
            ("✏️ Змінити текст", "admin:qedit_set:question"),
            ("🧩 Редагувати варіанти", "admin:qedit_choices"),
            ("✅ Змінити правильні", "admin:qedit_set:correct"),
            ("⬅️ Назад", "admin:qedit_back"),
        ],
        row=1,
    )
    return text, kb


def _norm_qsearch(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.U)
    s = re.sub(r"\s+", " ", s, flags=re.U)
    return s.strip()


def find_question_ids_by_title(qb: "QuestionBank", query: str, limit: int = 12) -> List[int]:
    qn = _norm_qsearch(query)
    if not qn:
        return []

    q_tokens = [t for t in qn.split() if t]
    res: List[Tuple[float, int]] = []

    for qid, q in qb.by_id.items():
        qt = _norm_qsearch(getattr(q, "question", "") or "")
        if not qt:
            continue

        if q_tokens and not any(t in qt for t in q_tokens):
            continue

        ratio = SequenceMatcher(None, qn, qt).ratio()
        if qn in qt:
            ratio = max(ratio, 0.95)

        qtoks = set(qt.split())
        qs = set(q_tokens)
        jacc = (len(qs & qtoks) / max(1, len(qs | qtoks))) if qs else 0.0

        score = 0.75 * ratio + 0.25 * jacc
        res.append((score, int(qid)))

    res.sort(key=lambda x: x[0], reverse=True)
    return [qid for _, qid in res[: max(1, int(limit))]]


def screen_admin_qedit_choices(q: Q, note: str = "", error: Optional[str] = None) -> Tuple[str, InlineKeyboardMarkup]:
    corr_set = set(int(x) for x in (q.correct or []))

    head = "🧩 <b>Редагування варіантів</b>"
    if note:
        head += f"\n{note}"

    lines: list[str] = []
    for i, c in enumerate(q.choices or [], start=1):
        mark = "✅" if i in corr_set else "▫️"
        lines.append(f"{mark} <b>{i}.</b> {hescape(c)}")

    choices_text = "\n".join(lines) if lines else "—"

    text = (
        f"{head}\n"
        f"ID: <code>{int(q.id)}</code>\n\n"
        f"Натисни номер варіанту, який треба відредагувати.\n\n"
        f"<b>Питання:</b>\n{hescape(q.question)}\n\n"
        f"<b>Варіанти:</b>\n{choices_text}"
    )
    if error:
        text += f"\n\n❗️ {hescape(error)}"

    b = InlineKeyboardBuilder()
    n = len(q.choices or [])

    for i in range(n):
        b.button(text=str(i + 1), callback_data=clamp_callback(f"admin:qedit_choice:{i + 1}"))

    controls: list[tuple[str, str]] = [
        ("✅ Змінити правильні", "admin:qedit_set:correct"),
        ("⬅️ Назад", "admin:qedit_cancel"),
    ]
    for t, cb in controls:
        b.button(text=t, callback_data=cb)

    full_rows, remainder = divmod(n, 4)
    adjust_list = [4] * full_rows
    if remainder:
        adjust_list.append(remainder)
    adjust_list.append(len(controls))
    b.adjust(*adjust_list)
    return text, b.as_markup()


def screen_admin_qedit_correct(q: Q, note: str = "", error: Optional[str] = None) -> Tuple[str, InlineKeyboardMarkup]:
    corr_set = set(int(x) for x in (q.correct or []))

    head = "✅ <b>Вибір правильних відповідей</b>"
    if note:
        head += f"\n{note}"

    lines_out: list[str] = []
    for i, c in enumerate(q.choices or [], start=1):
        mark = "✅" if i in corr_set else "▫️"
        lines_out.append(f"{mark} <b>{i}.</b> {hescape(c)}")
    choices_text = "\n".join(lines_out) if lines_out else "—"

    corr = ", ".join(str(x) for x in sorted(corr_set)) or "—"

    text = (
        f"{head}\n"
        f"ID: <code>{int(q.id)}</code>\n\n"
        f"Натискай на кнопки нижче — галочка перемикається, зміни зберігаються одразу.\n\n"
        f"<b>Питання:</b>\n{hescape(q.question)}\n\n"
        f"<b>Варіанти:</b>\n{choices_text}\n\n"
        f"<b>Зараз правильні:</b> <code>{hescape(corr)}</code>"
    )
    if error:
        text += f"\n\n❗️ {hescape(error)}"

    b = InlineKeyboardBuilder()
    n = len(q.choices or [])

    for i in range(1, n + 1):
        mark = "✅" if i in corr_set else "▫️"
        b.button(text=f"{mark} {i}", callback_data=clamp_callback(f"admin:qedit_corr:{i}"))

    b.button(text="⬅️ Назад", callback_data="admin:qedit_cancel")

    full_rows, remainder = divmod(n, 4)
    adjust_list = [4] * full_rows
    if remainder:
        adjust_list.append(remainder)
    adjust_list.append(1)
    b.adjust(*adjust_list)

    return text, b.as_markup()


def screen_admin_qedit_choice_prompt(q: Q, idx: int, error: Optional[str] = None) -> Tuple[str, InlineKeyboardMarkup]:
    idx = int(idx)
    current = "—"
    if 1 <= idx <= len(q.choices or []):
        current = hescape((q.choices or [])[idx - 1])

    text = (
        f"✏️ <b>Редагування варіанту</b> (ID <code>{int(q.id)}</code>)\n\n"
        f"Надішли новий текст для варіанту <b>{idx}</b> одним повідомленням.\n\n"
        f"<b>Поточне:</b>\n{current}"
    )
    if error:
        text += f"\n\n❗️ {hescape(error)}"

    kb = kb_inline(
        [
            ("⬅️ До варіантів", "admin:qedit_choices"),
            ("⬅️ До питання", "admin:qedit_cancel"),
        ],
        row=1,
    )
    return text, kb


def screen_admin_qedit_prompt(q: Q, field: str, error: Optional[str] = None) -> Tuple[str, InlineKeyboardMarkup]:
    field = (field or "").strip().lower()
    if field == "question":
        hint = "Надішли новий <b>текст питання</b> одним повідомленням."
        current = hescape(q.question)
    elif field == "choices":
        hint = (
            "Надішли <b>варіанти</b>, кожен з нового рядка.\n"
            "Опційно останнім рядком можна додати: <code>correct: 2,4</code>"
        )
        current = _fmt_q_choices(q)
    elif field == "correct":
        hint = "Надішли номери правильних варіантів через кому, напр.: <code>2</code> або <code>1,3</code>."
        current = hescape(", ".join(str(x) for x in (q.correct or [])) or "—")
    else:
        hint = "Надішли нове значення."
        current = "—"

    text = f"✏️ <b>Редагування</b> (ID <code>{int(q.id)}</code>)\n\n{hint}\n\n<b>Поточне:</b>\n{current}"
    if error:
        text += f"\n\n❗️ {hescape(error)}"

    kb = kb_inline([("⬅️ Назад", "admin:qedit_cancel")], row=1)
    return text, kb


_CORRECT_LINE_RE = re.compile(r"^(?:correct|правильн\w*|відповід\w*|ans)\s*[:=]\s*(.+)$", re.IGNORECASE)


def _parse_int_from_text(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None


def _parse_correct_list(s: str, n_choices: int) -> Optional[List[int]]:
    nums = [int(x) for x in re.findall(r"\d+", (s or ""))]
    nums = sorted(set(nums))
    nums = [x for x in nums if 1 <= x <= int(n_choices)]
    return nums if nums else None


def _parse_choices_and_optional_correct(text: str) -> Tuple[List[str], Optional[List[int]]]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]

    correct_part: Optional[str] = None
    for i, ln in enumerate(list(lines)):
        m = _CORRECT_LINE_RE.match(ln)
        if m:
            correct_part = m.group(1)
            lines.pop(i)
            break

    choices = [ln for ln in lines if ln]
    correct: Optional[List[int]] = None
    if correct_part is not None:
        correct = _parse_correct_list(correct_part, len(choices))
    return choices, correct


def pretest_mode(st: dict, qb) -> tuple[str, list[int]]:
    header = st.get("header", "")
    qids = list(st.get("qids", []) or [])
    return header, qids


@router.callback_query(F.data.startswith("admin:qwork:"))
async def admin_qwork_list(cb: CallbackQuery, bot: Bot, store: "Storage", qb: QuestionBank, admin_ids: set[int]):
    uid = cb.from_user.id
    if uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    try:
        page = int(cb.data.split(":")[2])
    except Exception:
        page = 0

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    st.pop(ADMIN_QWORK_AWAITING, None)
    await store.set_state(uid, st)

    text, kb = screen_admin_qwork(qb, page=page)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("admin:qwork_find:"))
async def admin_qwork_find_prompt(cb: CallbackQuery, bot: Bot, store: "Storage", qb: QuestionBank, admin_ids: set[int]):
    uid = cb.from_user.id
    if uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    try:
        page = int(cb.data.split(":")[2])
    except Exception:
        page = 0

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    st[ADMIN_QWORK_AWAITING] = "qsearch"
    st[ADMIN_QWORK_PAGE] = page
    st.pop(ADMIN_QWORK_QUERY, None)
    await store.set_state(uid, st)

    text, kb = screen_admin_qwork_find(page=page)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("admin:qedit:"))
async def admin_qedit_open(cb: CallbackQuery, bot: Bot, store: "Storage", qb: QuestionBank, admin_ids: set[int]):
    uid = cb.from_user.id
    if uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    parts = cb.data.split(":")
    if len(parts) < 5:
        await cb.answer("Помилка")
        return

    try:
        qid = int(parts[2])
        ret_kind = (parts[3] or "").strip()
        ret_val = int(parts[4]) if (parts[4] or "0").lstrip("-").isdigit() else 0
    except Exception:
        await cb.answer("Помилка")
        return

    q = qb.by_id.get(qid)
    if not q:
        await cb.answer("Питання не знайдено")
        return

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    st[ADMIN_QEDIT] = {
        "qid": int(qid),
        "ret_kind": ret_kind,
        "ret_val": int(ret_val),
        "await": None,
        "chat_id": cb.message.chat.id,
        "msg_id": cb.message.message_id,
    }
    st.pop(ADMIN_QWORK_AWAITING, None)
    await store.set_state(uid, st)

    text, kb = screen_admin_qedit(q)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("admin:qedit_set:"))
async def admin_qedit_set_field(cb: CallbackQuery, bot: Bot, store: "Storage", qb: QuestionBank, admin_ids: set[int]):
    uid = cb.from_user.id
    if uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    field = (cb.data.split(":")[2] or "").strip().lower()

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    qedit = st.get(ADMIN_QEDIT) or {}
    qid = qedit.get("qid")

    q = qb.by_id.get(int(qid)) if qid is not None else None
    if not q:
        await cb.answer("Питання не знайдено")
        return

    if field == "choices":
        qedit["await"] = None
        st[ADMIN_QEDIT] = qedit
        await store.set_state(uid, st)

        text, kb = screen_admin_qedit_choices(q)
        await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
        await cb.answer()
        return

    if field == "correct":
        qedit["await"] = None
        st[ADMIN_QEDIT] = qedit
        await store.set_state(uid, st)

        text, kb = screen_admin_qedit_correct(q)
        await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
        await cb.answer()
        return

    qedit["await"] = field
    st[ADMIN_QEDIT] = qedit
    await store.set_state(uid, st)

    text, kb = screen_admin_qedit_prompt(q, field)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("admin:qedit_corr:"))
async def admin_qedit_correct_toggle(cb: CallbackQuery, bot: Bot, store: "Storage", qb: QuestionBank,
                                     admin_ids: set[int]):
    uid = cb.from_user.id
    if uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    try:
        idx = int(cb.data.split(":")[2])
    except Exception:
        await cb.answer("Помилка")
        return

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    qedit = st.get(ADMIN_QEDIT) or {}
    qid = qedit.get("qid")

    q = qb.by_id.get(int(qid)) if qid is not None else None
    if not q:
        await cb.answer("Питання не знайдено")
        return

    n = len(q.choices or [])
    if not (1 <= idx <= n):
        text, kb = screen_admin_qedit_correct(q, error="Невірний номер варіанту.")
        await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
        await cb.answer()
        return

    corr = set(int(x) for x in (q.correct or []))

    if idx in corr:
        if len(corr) == 1:
            await cb.answer("Має бути хоча б 1 правильна відповідь")
            return
        corr.remove(idx)
    else:
        corr.add(idx)

    new_corr = sorted(corr)

    after = await store.update_question_content(int(q.id), correct=new_corr, changed_by=f"admin:{uid}")
    if after:
        q.correct = list(after.get("correct") or [])
        q.correct_texts = list(after.get("correct_texts") or [])

    text, kb = screen_admin_qedit_correct(q, note="✅ Збережено")
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data == "admin:qedit_choices")
async def admin_qedit_choices_menu(cb: CallbackQuery, bot: Bot, store: "Storage", qb: QuestionBank,
                                   admin_ids: set[int]):
    uid = cb.from_user.id
    if uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    qedit = st.get(ADMIN_QEDIT) or {}
    qid = qedit.get("qid")

    q = qb.by_id.get(int(qid)) if qid is not None else None
    if not q:
        await cb.answer("Питання не знайдено")
        return

    qedit["await"] = None
    st[ADMIN_QEDIT] = qedit
    await store.set_state(uid, st)

    text, kb = screen_admin_qedit_choices(q)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("admin:qedit_choice:"))
async def admin_qedit_choice_pick(cb: CallbackQuery, bot: Bot, store: "Storage", qb: QuestionBank, admin_ids: set[int]):
    uid = cb.from_user.id
    if uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    try:
        idx = int(cb.data.split(":")[2])
    except Exception:
        await cb.answer("Помилка")
        return

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    qedit = st.get(ADMIN_QEDIT) or {}
    qid = qedit.get("qid")

    q = qb.by_id.get(int(qid)) if qid is not None else None
    if not q:
        await cb.answer("Питання не знайдено")
        return

    if not (1 <= idx <= len(q.choices or [])):
        text, kb = screen_admin_qedit_choices(q, error="Невірний номер варіанту.")
        await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
        await cb.answer()
        return

    qedit["await"] = f"choice:{idx}"
    st[ADMIN_QEDIT] = qedit
    await store.set_state(uid, st)

    text, kb = screen_admin_qedit_choice_prompt(q, idx)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data == "admin:qedit_cancel")
async def admin_qedit_cancel(cb: CallbackQuery, bot: Bot, store: "Storage", qb: QuestionBank, admin_ids: set[int]):
    uid = cb.from_user.id
    if uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    qedit = st.get(ADMIN_QEDIT) or {}
    qedit["await"] = None
    st[ADMIN_QEDIT] = qedit
    await store.set_state(uid, st)

    qid = qedit.get("qid")
    q = qb.by_id.get(int(qid)) if qid is not None else None
    if not q:
        await cb.answer("Питання не знайдено")
        return

    text, kb = screen_admin_qedit(q)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data == "admin:qedit_back")
async def admin_qedit_back(cb: CallbackQuery, bot: Bot, store: "Storage", qb: QuestionBank, admin_ids: set[int]):
    uid = cb.from_user.id
    if uid not in admin_ids:
        await cb.answer("Немає доступу")
        return

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    qedit = st.get(ADMIN_QEDIT) or {}
    ret_kind = (qedit.get("ret_kind") or "").strip().lower()
    ret_val = int(qedit.get("ret_val") or 0)

    st.pop(ADMIN_QEDIT, None)
    st.pop(ADMIN_QWORK_AWAITING, None)
    await store.set_state(uid, st)

    if ret_kind == "qwork":
        text, kb = screen_admin_qwork(qb, page=ret_val)
        await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
        await cb.answer()
        return

    if ret_kind == "qsearch":
        query = (st.get(ADMIN_QWORK_QUERY) or "").strip()
        if not query:
            text, kb = screen_admin_qwork(qb, page=ret_val)
        else:
            text, kb = screen_admin_qwork_results(qb, query=query, page=ret_val, limit=12)
        await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
        await cb.answer()
        return

    if ret_kind == "qpick":
        sel = int(st.get("selected") or 0)
        header, qids = pretest_mode(st, qb)
        total = len(qids)
        from handlers.session import screen_qpick_grid
        text, kb = screen_qpick_grid(header, total, selected=sel)
        await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
        await cb.answer()
        return

    await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message, admin_ids=admin_ids)
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


@router.message(F.text)
async def admin_users_search_input(
        message: Message,
        bot: Bot,
        store: "Storage",
        qb: QuestionBank,
        admin_ids: set[int],
):
    uid = message.from_user.id
    if uid not in admin_ids:
        raise SkipHandler()

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}

    if st.get(ADMIN_QWORK_AWAITING) == "qsearch":
        page = int(st.get(ADMIN_QWORK_PAGE) or 0)
        query = (message.text or "").strip()

        try:
            await message.delete()
        except Exception:
            pass

        if len(_norm_qsearch(query)) < 3:
            text, kb = screen_admin_qwork_find(page=page, error="Введи хоча б 3 символи для пошуку.")
            await render_main(bot, store, uid, ui.get("chat_id") or message.chat.id, text, kb, message=None)
            return

        st[ADMIN_QWORK_QUERY] = query
        await store.set_state(uid, st)

        text, kb = screen_admin_qwork_results(qb, query=query, page=page, limit=12)
        await render_main(bot, store, uid, ui.get("chat_id") or message.chat.id, text, kb, message=None)
        return

    qedit = st.get(ADMIN_QEDIT) or {}
    field = (qedit.get("await") or "").strip().lower()
    qid = qedit.get("qid")

    if qid and field and (field in ("question", "correct") or field == "choices" or field.startswith("choice:")):
        q = qb.by_id.get(int(qid))
        if not q:
            raise SkipHandler()

        try:
            await message.delete()
        except Exception:
            pass

        if field == "choices":
            qedit["await"] = None
            st[ADMIN_QEDIT] = qedit
            await store.set_state(uid, st)

            text, kb = screen_admin_qedit_choices(q, note="Оберіть варіант для редагування")
            await render_admin_qedit(bot, store, uid, ui.get("chat_id") or message.chat.id, text, kb)
            return

        if field == "question":
            new_text = (message.text or "").strip()
            if not new_text:
                text, kb = screen_admin_qedit_prompt(q, field, error="Текст не може бути порожнім.")
                await render_admin_qedit(bot, store, uid, ui.get("chat_id") or message.chat.id, text, kb)
                return

            after = await store.update_question_content(int(qid), question=new_text, changed_by=f"admin:{uid}")
            if after:
                q.question = after["question"]
                q.correct_texts = list(after.get("correct_texts") or q.correct_texts)

            qedit["await"] = None
            st[ADMIN_QEDIT] = qedit
            await store.set_state(uid, st)

            text, kb = screen_admin_qedit(q, note="✅ Збережено")
            await render_admin_qedit(bot, store, uid, ui.get("chat_id") or message.chat.id, text, kb)
            return

        if field.startswith("choice:"):
            try:
                idx = int(field.split(":", 1)[1])
            except Exception:
                idx = 0

            if not (1 <= idx <= len(q.choices or [])):
                qedit["await"] = None
                st[ADMIN_QEDIT] = qedit
                await store.set_state(uid, st)

                text, kb = screen_admin_qedit_choices(q, error="Невірний номер варіанту.")
                await render_admin_qedit(bot, store, uid, ui.get("chat_id") or message.chat.id, text, kb)
                return

            new_text = (message.text or "").strip()
            if not new_text:
                text, kb = screen_admin_qedit_choice_prompt(q, idx, error="Текст не може бути порожнім.")
                await render_admin_qedit(bot, store, uid, ui.get("chat_id") or message.chat.id, text, kb)
                return

            new_choices = list(q.choices or [])
            new_choices[idx - 1] = new_text

            after = await store.update_question_content(
                int(qid),
                choices=new_choices,
                correct=list(q.correct or []),
                changed_by=f"admin:{uid}",
            )
            if after:
                q.choices = list(after.get("choices") or [])
                q.correct = list(after.get("correct") or [])
                q.correct_texts = list(after.get("correct_texts") or [])

            qedit["await"] = None
            st[ADMIN_QEDIT] = qedit
            await store.set_state(uid, st)

            text, kb = screen_admin_qedit_choices(q, note="✅ Збережено")
            await render_admin_qedit(bot, store, uid, ui.get("chat_id") or message.chat.id, text, kb)
            return

        if field == "correct":
            corr = _parse_correct_list(message.text or "", len(q.choices or []))
            if not corr:
                text, kb = screen_admin_qedit_prompt(q, field, error="Не бачу номерів або вони поза діапазоном.")
                await render_admin_qedit(bot, store, uid, ui.get("chat_id") or message.chat.id, text, kb)
                return

            after = await store.update_question_content(int(qid), correct=corr, changed_by=f"admin:{uid}")
            if after:
                q.correct = list(after.get("correct") or [])
                q.correct_texts = list(after.get("correct_texts") or [])

            qedit["await"] = None
            st[ADMIN_QEDIT] = qedit
            await store.set_state(uid, st)

            text, kb = screen_admin_qedit(q, note="✅ Збережено")
            await render_admin_qedit(bot, store, uid, ui.get("chat_id") or message.chat.id, text, kb)
            return

    raise SkipHandler()


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

    await render_admin_user_detail(bot, store, admin_uid, cb.message.chat.id, target_id, back_offset,
                                   message=cb.message)
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
    await render_admin_user_detail(bot, store, admin_uid, cb.message.chat.id, target_id, back_offset,
                                   message=cb.message)
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

    await store.set_subscription(target_id, None, infinite=False)

    await render_admin_user_detail(bot, store, admin_uid, cb.message.chat.id, target_id, back_offset,
                                   message=cb.message)
    await cb.answer("Ок")
