from __future__ import annotations

import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Set, Tuple

from aiogram import Router, F
from aiogram.filters import CommandStart, Command
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

# Імпортуємо спільні частини логіки з модулів (так легше зберегти структуру, як у старому bot.py)
from .config import *  # noqa
from .state import *   # noqa
from .callbacks import *  # noqa
from .utils import *   # noqa
from .questions import *  # noqa
from .db import *      # noqa
from .keyboards import *  # noqa
from .sessions import *   # noqa
from .profile import *    # noqa

router = Router()

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


async def ok_page(call: CallbackQuery, callback_data: OkPageCb) -> None:
    await call.message.edit_text("Оберіть ОК:", reply_markup=kb_pick_ok(page=int(callback_data.page)))
    await call.answer()


async def ok_multi_page(call: CallbackQuery, callback_data: OkMultiPageCb) -> None:
    if not DB_POOL:
        return
    tg_id = call.from_user.id
    mode = str(callback_data.mode)
    selected = await db_get_ok_prefs(DB_POOL, tg_id, mode)
    await safe_edit(
        call,
        f"Оберіть <b>декілька</b> ОК (блоків):\nОбрано: <b>{len(selected)}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_pick_ok_multi(mode, page=int(callback_data.page), selected=selected),
    )
    await call.answer()


async def ok_multi_toggle(call: CallbackQuery, callback_data: OkToggleCb) -> None:
    if not DB_POOL:
        return
    tg_id = call.from_user.id
    mode = str(callback_data.mode)
    ok_code = str(callback_data.ok_code)
    page = int(callback_data.page)

    selected = await db_get_ok_prefs(DB_POOL, tg_id, mode)
    if ok_code in selected:
        selected.remove(ok_code)
    else:
        selected.add(ok_code)

    await db_set_ok_prefs(DB_POOL, tg_id, mode, selected)

    await safe_edit(
        call,
        f"Оберіть <b>декілька</b> ОК (блоків):\nОбрано: <b>{len(selected)}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_pick_ok_multi(mode, page=page, selected=selected),
    )
    await call.answer()


async def ok_multi_clear(call: CallbackQuery, callback_data: OkClearCb) -> None:
    if not DB_POOL:
        return
    tg_id = call.from_user.id
    mode = str(callback_data.mode)
    page = int(callback_data.page)
    await db_clear_ok_prefs(DB_POOL, tg_id, mode)
    await safe_edit(
        call,
        "Оберіть <b>декілька</b> ОК (блоків):\nОбрано: <b>0</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_pick_ok_multi(mode, page=page, selected=set()),
    )
    await call.answer()


async def ok_multi_all(call: CallbackQuery, callback_data: OkAllCb) -> None:
    if not DB_POOL:
        return
    tg_id = call.from_user.id
    mode = str(callback_data.mode)
    codes = {OK_CODE_LAW} | {c for c in OK_CODES if c != OK_CODE_LAW}
    await db_set_ok_prefs(DB_POOL, tg_id, mode, codes)
    await safe_edit(
        call,
        f"Оберіть <b>декілька</b> ОК (блоків):\nОбрано: <b>{len(codes)}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_pick_ok_multi(mode, page=0, selected=codes),
    )
    await call.answer()


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
    selected = {c for c in selected if c}  # sanitize
    if not selected:
        await call.answer("Оберіть хоча б один ОК", show_alert=True)
        return

    # якщо обрано 1 ОК — лишаємо стару логіку (можна ще вибирати теми)
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

    # multi-OK
    shown = ", ".join(sorted(selected))
    if mode == "train":
        await safe_edit(
            call,
            f"Обрані модулі: <b>{html_escape(shown)}</b>\nОберіть як тренуватись:",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_train_pick_multi("train"),
        )
    else:
        await safe_edit(
            call,
            f"Обрані модулі: <b>{html_escape(shown)}</b>\nПочати екзамен по всіх обраних модулях?",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_train_pick_multi("exam"),
        )
    await call.answer()


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
    selected = {c for c in selected if c}
    if not selected:
        await call.answer("Оберіть ОК", show_alert=True)
        return

    pool: List[int] = []
    for ok_code in sorted(selected):
        lvl = 0 if ok_code == OK_CODE_LAW else LEVEL_ALL
        pool.extend(base_qids_for_scope(ok_code, lvl))

    pool_qids = effective_qids(list(dict.fromkeys(pool)))

    # підтверджуємо callback одразу і “замикаємо” клавіатуру
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


async def pick_ok_from_anywhere(call: CallbackQuery) -> None:
    await safe_edit(call, "Оберіть ОК:", reply_markup=kb_pick_ok(page=0))
    await call.answer()


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
            await safe_edit(call, "Статистики поки нема.", reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])))
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

        await safe_edit(call, out, parse_mode=ParseMode.HTML, reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])))
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

        await safe_edit(call, out, parse_mode=ParseMode.HTML, reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])))
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

    # TRAIN / EXAM
    if action in ("train", "exam"):
        if not await db_has_access(user):
            await safe_edit(
                call,
                "⛔️ Доступ завершився.\nНапишіть адміну для доступу.",
                reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])),
            )
            await call.answer()
            return

        # EXAM (як було)
        if action == "exam":
            position = user.get("position")
            if not position:
                await safe_edit(call, "Оберіть посаду для екзамену:", reply_markup=kb_pick_position("exam"))
                await call.answer()
                return

            try:
                await call.message.edit_reply_markup(reply_markup=None)
            except Exception:
                pass

            await call.answer()
            exam_qids = build_position_exam_qids(position)
            await start_exam_session(
                call.bot,
                tg_id,
                call.message.chat.id,
                user,
                exam_qids,
                edit_message=call.message,
            )
            return

        # TRAIN — одразу вибір модулів (ОК)
        mode = "train"

        selected_ok = await db_get_ok_prefs(DB_POOL, tg_id, mode)
        # fallback: якщо є старий single-scope — підхопимо його
        if not selected_ok and user_has_scope(user):
            ok_code, _lvl = get_user_scope(user)
            selected_ok = {ok_code}
            await db_set_ok_prefs(DB_POOL, tg_id, mode, selected_ok)

        await safe_edit(
            call,
            "Оберіть <b>модулі</b> (ОК):\n"
            f"Обрано: <b>{len(selected_ok)}</b>",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_pick_ok_multi(mode, page=0, selected=set(selected_ok)),
        )
        await call.answer()
        return

    await safe_edit(call, "🏠 Меню", reply_markup=kb_main_menu(is_admin=bool(user["is_admin"])))
    await call.answer()


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
        edit_message=call.message,  # ← додано
    )


async def backmode(call: CallbackQuery):
    mode = call.data.split(":", 1)[1]

    text = "Як ви хочете навчатись?" if mode == "train" else "Як ви хочете складати екзамен?"

    await call.message.edit_text(
        text,
        reply_markup=kb_train_mode(mode)
    )
    await call.answer()


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

    if text == "📚 Навчання":
        selected_ok = await db_get_ok_prefs(DB_POOL, tg_id, "train")
        selected_ok = set(selected_ok or [])

        # fallback: якщо є старий single-scope — підхопимо його
        if not selected_ok and user_has_scope(user):
            ok_code, _lvl = get_user_scope(user)
            selected_ok = {ok_code}
            await db_set_ok_prefs(DB_POOL, tg_id, "train", selected_ok)

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

    await call.answer()

    # ✅ прибрати кнопки вибору (без нового повідомлення)
    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    await start_session_for_pool(
        call.bot, tg_id, call.message.chat.id, user, mode, pool_qids,
        edit_message=call.message,
    )


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

    base = base_qids_for_scope(ok_code, lvl)
    pool_qids = effective_qids(base)

    # підтверджуємо callback одразу
    await call.answer()

    # "замикаємо" попередню клавіатуру (щоб не було повторних натискань)
    if call.message:
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


async def back_from_admin(message: Message) -> None:
    if not DB_POOL:
        return

    tg_id = message.from_user.id
    user = await db_get_user(DB_POOL, tg_id)

    await show_main_menu(message, is_admin=bool(user and user["is_admin"]))


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
