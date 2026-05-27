from __future__ import annotations

import re
from html import escape as hescape
from typing import Any, Dict, List, Optional, Tuple

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from access import access_status, access_tier
from questions import Q, QuestionBank
from storage import Storage
from utils import CASES_PER_PAGE, CASE_QUESTIONS_PER_PAGE, OK_QUESTIONS_PER_PAGE, GROUP_URL, clamp_callback


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
            tail.append(item)
        else:
            main.append(item)

    for text, data in main:
        b.button(text=text, callback_data=clamp_callback(data))
    b.adjust(row)

    for text, data in tail:
        b.button(text=text, callback_data=clamp_callback(data))
        b.adjust(1)

    return b.as_markup()


def fmt_access_line(user: Dict[str, Any]) -> str:
    ok, st = access_status(user)
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
        await store.set_ui(user_id, chat_id, mid)

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

    sent = await bot.send_message(
        chat_id,
        text,
        reply_markup=keyboard,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )
    await save_mid(sent.message_id)


# -------------------- Screens --------------------

def screen_main_menu(user: Dict[str, Any], is_admin: bool) -> Tuple[str, InlineKeyboardMarkup]:
    FILL = "\u2800" * 30

    text = (
        "🏠 <b>Головне меню</b>\n"
        f"{FILL}\n"
        "ℹ️ <b>Підказка</b>:\n"
        "• 📚 <b>Навчання</b> — розділ для вивчення матеріалів (законодавство та ОК-модулі).\n"
        "• 📝 <b>Тестування</b> — розділ для перевірки знань за законодавством і ОК-модулями.\n"
        "• 🗂 <b>Кейси</b> — питання та правильні відповіді, які додає адміністратор.\n"
        "• 🔍 <b>Питання ОК</b> — пошук питань митних компетенцій з правильними відповідями (повний доступ)."
    )

    rows = [
        [InlineKeyboardButton(text="📚 Навчання", callback_data="nav:learn")],
        [InlineKeyboardButton(text="📝 Тестування", callback_data="nav:test")],
        [InlineKeyboardButton(text="🗂 Кейси", callback_data="nav:cases:0")],
        [InlineKeyboardButton(text="🔍 Питання ОК", callback_data="nav:oksearch")],
        [InlineKeyboardButton(text="📊 Статистика", callback_data="nav:stats")],
        [InlineKeyboardButton(text="❓ Допомога", callback_data="nav:help")],
    ]
    if is_admin:
        rows.append([InlineKeyboardButton(text="🛠 Користувачі", callback_data="admin:users:0")])
        rows.append([InlineKeyboardButton(text="✏️ Робота над питаннями", callback_data="admin:qwork:0")])

    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def screen_help(admin_url: str) -> Tuple[str, InlineKeyboardMarkup]:
    text = (
        "❓ <b>Допомога</b>\n\n"
        "Тут ви можете:\n"
        "▪ приєднатися до Telegram-групи\n"
        "▪ звернутися до адміністратора\n"
    )

    b = InlineKeyboardBuilder()
    if GROUP_URL:
        b.button(text="🔗 Telegram-група", url=GROUP_URL)
    if admin_url:
        b.button(text="📩 Написати адміну", url=admin_url)

    b.button(text="⬅️ Меню", callback_data="nav:menu")
    b.adjust(1)
    return text, b.as_markup()


def _clip_for_telegram(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def screen_cases_list(cases: list[dict], page: int = 0) -> Tuple[str, InlineKeyboardMarkup]:
    page = max(0, int(page or 0))
    total = len(cases or [])
    start = page * CASES_PER_PAGE
    visible = (cases or [])[start: start + CASES_PER_PAGE]

    if not total:
        text = (
            "🗂 <b>Кейси</b>\n\n"
            "Кейсів ще немає.\n"
            "Адміністратор може додати Keys.db в адмін-панелі."
        )
        return text, kb_inline([("⬅️ Меню", "nav:menu")], row=1)

    text = (
        "🗂 <b>Кейси</b>\n\n"
        "Оберіть кейс, щоб переглянути питання та правильні відповіді.\n"
        f"Доступно кейсів: <b>{total}</b>"
    )

    rows: list[list[InlineKeyboardButton]] = []
    for item in visible:
        case_id = int(item.get("id") or 0)
        number = _clip_for_telegram(item.get("case_number") or "Без номера", 24)
        count = int(item.get("questions_count") or 0)
        rows.append([
            InlineKeyboardButton(
                text=f"🗂 Кейс {number} · {count}",
                callback_data=clamp_callback(f"case:view:{case_id}:0"),
            )
        ])

    nav_row: list[InlineKeyboardButton] = []
    if page > 0:
        nav_row.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=clamp_callback(f"nav:cases:{page - 1}")))
    if start + CASES_PER_PAGE < total:
        nav_row.append(InlineKeyboardButton(text="Далі ➡️", callback_data=clamp_callback(f"nav:cases:{page + 1}")))
    if nav_row:
        rows.append(nav_row)

    rows.append([InlineKeyboardButton(text="⬅️ Меню", callback_data="nav:menu")])
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def screen_case_detail(case: dict, questions: list[dict], offset: int, has_prev: bool, has_next: bool) -> Tuple[str, InlineKeyboardMarkup]:
    case_id = int(case.get("id") or 0)
    offset = max(0, int(offset or 0))
    number = hescape(str(case.get("case_number") or "Без номера"))
    total = int(case.get("questions_count") or 0)

    parts = [
        f"🗂 <b>Кейс {number}</b>",
        f"Питань: <b>{total}</b>",
    ]

    if not questions:
        parts.append("\nУ цьому кейсі поки немає питань.")

    for item in questions:
        position = hescape(str(item.get("position") or "—"))
        question = hescape(_clip_for_telegram(item.get("question") or "Питання без тексту", 520))
        answer = hescape(_clip_for_telegram(item.get("correct_answer") or "—", 420))
        parts.append(
            f"\n<b>№ {position}</b>\n"
            f"{question}\n"
            f"✅ <b>Правильна відповідь:</b>\n{answer}"
        )

    rows: list[list[InlineKeyboardButton]] = []
    nav_row: list[InlineKeyboardButton] = []
    if has_prev:
        prev_offset = max(0, offset - CASE_QUESTIONS_PER_PAGE)
        nav_row.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=clamp_callback(f"case:view:{case_id}:{prev_offset}")))
    if has_next:
        nav_row.append(InlineKeyboardButton(text="Далі ➡️", callback_data=clamp_callback(f"case:view:{case_id}:{offset + CASE_QUESTIONS_PER_PAGE}")))
    if nav_row:
        rows.append(nav_row)
    rows.append([InlineKeyboardButton(text="🗂 До кейсів", callback_data="nav:cases:0")])
    rows.append([InlineKeyboardButton(text="⬅️ Меню", callback_data="nav:menu")])

    return "\n\n".join([part for part in parts if part]), InlineKeyboardMarkup(inline_keyboard=rows)


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
    full = (full or "").strip()
    m = re.match(r"^\s*(\[[^\]]+\])\s*(.*)$", full)
    if not m:
        return "", full
    return m.group(1).strip(), (m.group(2) or "").strip()


def ok_button_text(module: str, *, prefix: str = "", suffix: str = "", max_len: int = 34) -> str:
    full = ok_full_label(module)
    code, name = _split_ok_label(full)

    if code:
        line1 = (prefix + code).strip()
        line2_raw = name
        if suffix:
            line2_raw = (f"{name} • {suffix}" if name else suffix).strip()
        line2 = _wrap_for_button(line2_raw, max_len=max_len, max_lines=1)
        return (line1 + "\n" + line2).strip()

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
            "Виберіть потрібні модулі для навчання (можна кілька). Натискайте на модулі — з'явиться список обраних. Коли завершите, натисніть «Готово». За потреби ви можете змінити вибір"
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


def screen_test_config(
        modules: List[str],
        qb: QuestionBank,
        temp_levels: Dict[str, Any],
        include_law: bool = True,
        law_count: int = 50
) -> Tuple[str, InlineKeyboardMarkup]:
    def _norm_levels(raw: Any, available: List[int]) -> List[int]:
        if raw is None:
            return [available[0]] if available else []

        if isinstance(raw, int):
            levels = [raw]
        elif isinstance(raw, list):
            levels = []
            for x in raw:
                try:
                    levels.append(int(x))
                except Exception:
                    pass
        else:
            try:
                levels = [int(raw)]
            except Exception:
                levels = []

        levels = [lvl for lvl in levels if lvl in available]
        return sorted(set(levels))

    lines = [
        "📝 <b>Тестування</b>",
        "",
        "ℹ️ <b>Підказка</b>: розділ «Тестування» — це перевірка знань за законодавством і обраними ОК-модулями.",
        "<i>Під час тестування: при вірній відповіді система автоматично переходить до наступного питання, "
        "при невірній — відображається екран з поясненням помилки.</i>",
        "",
        "Оберіть <b>рівні</b> для кожного модуля ОК (можна кілька).",
        "Якщо зняти всі рівні в модулі — він <b>не потрапить</b> у тест.",
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

        selected = _norm_levels(temp_levels.get(m, None), available)

        if not selected:
            lvl_label = "❌ не включати"
        elif len(selected) == 1:
            lvl_label = f"Рівень {selected[0]}"
        else:
            lvl_label = "Рівні " + ", ".join(str(x) for x in selected)

        code = ok_extract_code(m)
        label = f"🧩 [{code}] • {lvl_label}" if code else f"🧩 {ok_full_label(m)} • {lvl_label}"
        buttons.append((label, f"testlvl:modi:{i}"))

    buttons += [("📖 Почати тест", "test:start"), ("⬅️ Меню", "nav:menu")]
    return "\n".join(lines), kb_inline(buttons, row=1)


def screen_test_pick_level(idx: int, module: str, qb: QuestionBank, current: Optional[Any]) -> Tuple[str, InlineKeyboardMarkup]:
    levels = sorted(qb.ok_modules.get(module, {}).keys())

    def _norm_current(raw: Any) -> List[int]:
        if isinstance(raw, int):
            out = [raw]
        elif isinstance(raw, list):
            out = []
            for x in raw:
                try:
                    out.append(int(x))
                except Exception:
                    pass
        else:
            try:
                out = [int(raw)]
            except Exception:
                out = []
        out = [x for x in out if x in levels]
        return sorted(set(out))

    current_levels = set(_norm_current(current))

    text = f"🧩 <b>{ok_full_label(module)}</b>\n\nОберіть рівні для тесту (можна кілька):"
    buttons: List[Tuple[str, str]] = []
    for lvl in levels:
        mark = "✅ " if lvl in current_levels else ""
        buttons.append((f"{mark}Рівень {lvl}", f"testlvl:togglei:{idx}:{lvl}"))

    buttons.append(("⬅️ Назад", "testlvl:back"))
    return text, kb_inline(buttons, row=2)


def screen_ok_search_prompt(error: Optional[str] = None) -> Tuple[str, InlineKeyboardMarkup]:
    text = (
        "🔍 <b>Питання ОК — пошук</b>\n\n"
        "Введіть текст для пошуку серед питань операційних митних компетенцій.\n"
        "Пошук здійснюється по тексту питання та правильних відповідей.\n\n"
        "✏️ <i>Надішліть запит у відповідь на це повідомлення.</i>"
    )
    if error:
        text += f"\n\n❗️ {hescape(error)}"
    kb = kb_inline([("⬅️ Меню", "nav:menu")], row=1)
    return text, kb


def screen_ok_search_results(
        query: str,
        results: List[Dict[str, Any]],
        offset: int,
        has_prev: bool,
        has_next: bool,
) -> Tuple[str, InlineKeyboardMarkup]:
    import json as _json

    def _parse_ct(v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(x) for x in v if x]
        if isinstance(v, str):
            try:
                parsed = _json.loads(v)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if x]
            except Exception:
                pass
        return []

    q_escaped = hescape((query or "").strip())
    parts: List[str] = [f"🔍 <b>Питання ОК</b> · пошук: «{q_escaped}»"]

    if not results:
        parts.append("Нічого не знайдено. Спробуйте інший запит.")
    else:
        for row in results:
            ok_code = str(row.get("ok") or "ОК").strip()
            level = row.get("level")
            lvl_str = f" · Рівень {level}" if level else ""
            header = f"[{ok_code}]{lvl_str}"

            question = hescape(_clip_for_telegram(row.get("question") or "", 500))
            correct_texts = _parse_ct(row.get("correct_texts"))
            answer_raw = "; ".join(correct_texts) if correct_texts else "—"
            answer = hescape(_clip_for_telegram(answer_raw, 400))

            parts.append(
                f"<b>{hescape(header)}</b>\n"
                f"{question}\n"
                f"✅ <b>Правильна відповідь:</b>\n{answer}"
            )

    rows_kb: List[List[InlineKeyboardButton]] = []
    nav_row: List[InlineKeyboardButton] = []
    if has_prev:
        prev_off = max(0, offset - OK_QUESTIONS_PER_PAGE)
        nav_row.append(InlineKeyboardButton(
            text="⬅️ Назад", callback_data=clamp_callback(f"oksearch:pg:{prev_off}")
        ))
    if has_next:
        nav_row.append(InlineKeyboardButton(
            text="Далі ➡️", callback_data=clamp_callback(f"oksearch:pg:{offset + OK_QUESTIONS_PER_PAGE}")
        ))
    if nav_row:
        rows_kb.append(nav_row)
    rows_kb.append([InlineKeyboardButton(text="🔍 Новий пошук", callback_data="nav:oksearch")])
    rows_kb.append([InlineKeyboardButton(text="⬅️ Меню", callback_data="nav:menu")])

    return "\n\n".join(p for p in parts if p), InlineKeyboardMarkup(inline_keyboard=rows_kb)


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
        lines.append(f"{mark} <b>{i + 1})</b> {ch}{note}")

    return "\n".join(lines)


def kb_answers(n: int, allow_skip: bool = True, edit_cb: Optional[str] = None) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()

    n = max(0, int(n))

    for i in range(n):
        b.button(text=str(i + 1), callback_data=clamp_callback(f"ans:{i}"))

    controls: list[tuple[str, str]] = []
    if allow_skip:
        controls.append(("⏭ Пропустити", "skip"))
    if edit_cb:
        controls.append(("✏️ Змінити питання", clamp_callback(edit_cb)))
    controls.append(("⏹ Вийти", "leave:confirm"))

    for text, cb in controls:
        b.button(text=text, callback_data=cb)

    full_rows, remainder = divmod(n, 4)
    adjust_list = [4] * full_rows
    if remainder:
        adjust_list.append(remainder)
    adjust_list.append(len(controls))

    b.adjust(*adjust_list)
    return b.as_markup()


def kb_feedback(edit_cb: Optional[str] = None) -> InlineKeyboardMarkup:
    buttons: List[Tuple[str, str]] = []
    if edit_cb:
        buttons.append(("✏️ Змінити питання", edit_cb))
    buttons.append(("Зрозуміло / Продовжити", "next"))
    return kb_inline(buttons, row=1)


def kb_leave_confirm() -> InlineKeyboardMarkup:
    return kb_inline(
        [
            ("⬅️ Продовжити", "leave:back"),
            ("🚪 Вийти в меню", "leave:yes"),
        ],
        row=1,
        single_row_prefixes=None,
        single_row_exact=None,
    )
