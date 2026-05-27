from __future__ import annotations

from aiogram import Bot, F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.types import CallbackQuery, Message

from access import access_status, access_tier
from questions import QuestionBank
from storage import Storage
from ui import (
    kb_inline,
    render_main,
    screen_cases_list,
    screen_case_detail,
    screen_help,
    screen_law_groups,
    screen_law_parts,
    screen_learning_menu,
    screen_main_menu,
    screen_no_access,
    screen_ok_levels,
    screen_ok_menu,
    screen_ok_modules_pick,
    screen_ok_search_prompt,
    screen_ok_search_results,
    screen_test_config,
)
from utils import (
    CASE_QUESTIONS_PER_PAGE,
    OK_QUESTIONS_PER_PAGE,
    OK_SEARCH_AWAITING,
    OK_SEARCH_QUERY,
    clamp_callback,
    get_admin_contact_url,
)

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
    await store.start_trial_if_needed(
        uid,
        first_name=message.from_user.first_name,
        last_name=message.from_user.last_name,
    )

    user = await store.get_user(uid)
    chat_id = message.chat.id

    try:
        await message.delete()
    except Exception:
        pass

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
    uid = cb.from_user.id
    user = await store.get_user(uid)

    admin_url = get_admin_contact_url(admin_ids)
    text, kb = screen_help(admin_url)

    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("nav:cases:"))
async def nav_cases(cb: CallbackQuery, bot: Bot, store: Storage):
    uid = cb.from_user.id
    try:
        page = int(cb.data.rsplit(":", 1)[1])
    except Exception:
        page = 0

    cases = await store.list_case_banks()
    text, kb = screen_cases_list(cases, page=page)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("case:view:"))
async def nav_case_detail(cb: CallbackQuery, bot: Bot, store: Storage):
    uid = cb.from_user.id
    try:
        _, _, case_id_raw, offset_raw = cb.data.split(":", 3)
        case_id = int(case_id_raw)
        offset = max(0, int(offset_raw))
    except Exception:
        await cb.answer("Не вдалося відкрити кейс.")
        return

    case = await store.get_case_bank(case_id)
    if not case:
        await render_main(
            bot, store, uid, cb.message.chat.id,
            "🗂 <b>Кейс не знайдено</b>\n\nМожливо, адміністратор його видалив.",
            kb_inline([("🗂 До кейсів", "nav:cases:0"), ("⬅️ Меню", "nav:menu")], row=1),
            message=cb.message,
        )
        await cb.answer()
        return

    rows = await store.list_case_questions(case_id, offset=offset, limit=CASE_QUESTIONS_PER_PAGE + 1)
    has_next = len(rows) > CASE_QUESTIONS_PER_PAGE
    questions = rows[:CASE_QUESTIONS_PER_PAGE]
    text, kb = screen_case_detail(case, questions, offset, offset > 0, has_next)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
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


@router.callback_query(F.data == "nav:oksearch")
async def nav_ok_search(cb: CallbackQuery, bot: Bot, store: Storage, admin_ids: set[int]):
    from aiogram.utils.keyboard import InlineKeyboardBuilder

    uid = cb.from_user.id
    user = await store.get_user(uid)

    if access_tier(user) != "full":
        admin_url = get_admin_contact_url(admin_ids)
        text = (
            "🔐 <b>Питання ОК</b>\n\n"
            "Цей розділ доступний лише для користувачів з <b>повним оплаченим доступом</b>.\n"
            "Зверніться до адміністратора для отримання доступу."
        )
        b = InlineKeyboardBuilder()
        if admin_url:
            b.button(text="📩 Написати адміну", url=admin_url)
        b.button(text="⬅️ Меню", callback_data="nav:menu")
        b.adjust(1)
        await render_main(bot, store, uid, cb.message.chat.id, text, b.as_markup(), message=cb.message)
        await cb.answer()
        return

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    st[OK_SEARCH_AWAITING] = True
    st.pop(OK_SEARCH_QUERY, None)
    await store.set_state(uid, st)

    text, kb = screen_ok_search_prompt()
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data.startswith("oksearch:pg:"))
async def ok_search_page(cb: CallbackQuery, bot: Bot, store: Storage):
    uid = cb.from_user.id
    try:
        offset = max(0, int(cb.data.split(":", 2)[2]))
    except Exception:
        offset = 0

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    query = (st.get(OK_SEARCH_QUERY) or "").strip()

    if not query:
        text, kb = screen_ok_search_prompt(error="Запит не збережено. Введіть новий.")
        await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
        await cb.answer()
        return

    rows = await store.search_ok_questions(query, limit=OK_QUESTIONS_PER_PAGE + 1, offset=offset)
    has_next = len(rows) > OK_QUESTIONS_PER_PAGE
    results = rows[:OK_QUESTIONS_PER_PAGE]

    text, kb = screen_ok_search_results(query, results, offset, has_prev=(offset > 0), has_next=has_next)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.message(F.text)
async def ok_search_text_input(message: Message, bot: Bot, store: Storage):
    uid = message.from_user.id
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}

    if not st.get(OK_SEARCH_AWAITING):
        raise SkipHandler()

    query = (message.text or "").strip()

    try:
        await message.delete()
    except Exception:
        pass

    if len(query) < 3:
        text, kb = screen_ok_search_prompt(error="Введіть хоча б 3 символи.")
        await render_main(bot, store, uid, ui.get("chat_id") or message.chat.id, text, kb, message=None)
        return

    st[OK_SEARCH_AWAITING] = False
    st[OK_SEARCH_QUERY] = query
    await store.set_state(uid, st)

    rows = await store.search_ok_questions(query, limit=OK_QUESTIONS_PER_PAGE + 1, offset=0)
    has_next = len(rows) > OK_QUESTIONS_PER_PAGE
    results = rows[:OK_QUESTIONS_PER_PAGE]

    text, kb = screen_ok_search_results(query, results, 0, has_prev=False, has_next=has_next)
    await render_main(bot, store, uid, ui.get("chat_id") or message.chat.id, text, kb, message=None)
