from __future__ import annotations

import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from aiogram import Bot, F, Router
from aiogram.types import CallbackQuery

from questions import QuestionBank
from storage import Storage
from ui import (
    build_feedback_text,
    kb_inline,
    render_main,
    screen_test_config,
    screen_test_pick_level,
    ok_sort_key,
)
from utils import dt_to_iso, iso_to_dt, now
from handlers.session import guard_access_in_session, show_next_in_session

router = Router()


@router.callback_query(F.data == "nav:test")
async def nav_test(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    from access import access_status
    from ui import screen_no_access
    from utils import get_admin_contact_url

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

    temp_levels: Dict[str, List[int]] = {}
    for m in modules:
        levels_map = qb.ok_modules.get(m, {})
        if not levels_map:
            continue
        available = sorted(levels_map.keys())
        lvl = int(last_levels.get(m, available[0]))
        if lvl not in available:
            lvl = available[0]
        temp_levels[m] = [lvl]

    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    st["test_mod_list"] = list(modules)
    st["test_levels_temp"] = dict(temp_levels)
    st["test_include_law"] = True
    await store.set_state(uid, st)

    text, kb = screen_test_config(modules, qb, temp_levels, include_law=True)
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


@router.callback_query(F.data.startswith("testlvl:togglei:"))
async def testlvl_toggle_level(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
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
    levels_map = qb.ok_modules.get(module, {}) or {}
    available = sorted(levels_map.keys())
    if lvl not in available:
        await cb.answer("Рівень не знайдено")
        return

    temp_levels = dict(st.get("test_levels_temp", {}) or {})
    raw = temp_levels.get(module, None)

    if raw is None:
        selected = [available[0]] if available else []
    elif isinstance(raw, int):
        selected = [raw]
    elif isinstance(raw, list):
        selected = []
        for x in raw:
            try:
                selected.append(int(x))
            except Exception:
                pass
    else:
        selected = []

    selected = [x for x in selected if x in available]
    selected = sorted(set(selected))

    if lvl in selected:
        selected = [x for x in selected if x != lvl]
    else:
        selected.append(lvl)

    selected = sorted(set(selected))

    temp_levels[module] = selected
    st["test_levels_temp"] = temp_levels
    await store.set_state(uid, st)

    text, kb = screen_test_pick_level(idx, module, qb, selected)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


@router.callback_query(F.data == "testlvl:back")
async def testlvl_back(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank):
    uid = cb.from_user.id
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}
    mod_list = st.get("test_mod_list", []) or []
    temp_levels = st.get("test_levels_temp", {}) or {}
    include_law = bool(st.get("test_include_law", True))

    text, kb = screen_test_config(mod_list, qb, temp_levels, include_law=include_law)
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
    test_blocks: dict[str, list[int]] = {}

    if law_qids:
        test_blocks["Законодавство"] = list(law_qids)

    for m in modules:
        levels_map = qb.ok_modules.get(m, {}) or {}
        if not levels_map:
            continue
        available = sorted(levels_map.keys())

        raw0 = picked_levels.get(m, None)

        if isinstance(raw0, list) and len(raw0) == 0:
            continue

        raw = raw0
        if raw is None:
            raw = last_levels.get(m, available[0])

        if isinstance(raw, int):
            selected = [raw]
        elif isinstance(raw, list):
            selected = []
            for x in raw:
                try:
                    selected.append(int(x))
                except Exception:
                    pass
        else:
            try:
                selected = [int(raw)]
            except Exception:
                selected = []

        selected = [lvl for lvl in selected if lvl in available]
        selected = sorted(set(selected))

        if not selected:
            selected = [available[0]]

        pool: List[int] = []
        for lvl in selected:
            pool.extend(levels_map.get(lvl, []) or [])

        pool = list(dict.fromkeys(pool))

        picked = qb.pick_random(pool, 20)
        if picked:
            ok_qids.extend(picked)
            test_blocks[str(m)] = list(picked)

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
        "answers": {},
        "test_blocks": test_blocks,
    }
    await store.set_state(uid, st)

    await cb.answer()
    await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message, admin_ids=admin_ids)


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
            dt = iso_to_dt(dt)
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


async def _render_test_review(bot: Bot, store: Storage, qb: QuestionBank, uid: int, chat_id: int, message: Any):
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}

    wrong_qids = list(st.get("wrong_qids", []) or [])
    chosen_map = st.get("chosen", {}) or {}
    i = int(st.get("review_index", 0) or 0)

    if not wrong_qids:
        await render_main(
            bot, store, uid, chat_id,
            "✅ У цьому тесті немає помилок.",
            kb_inline([("⬅️ Головне меню", "nav:menu")], row=1),
            message=message
        )
        return

    i = max(0, min(i, len(wrong_qids) - 1))
    qid = int(wrong_qids[i])
    q = qb.by_id.get(qid)

    if not q:
        st["wrong_qids"] = [x for x in wrong_qids if int(x) != qid]
        st["review_index"] = min(i, max(0, len(st["wrong_qids"]) - 1))
        st["mode"] = "test_review"
        await store.set_state(uid, st)
        await _render_test_review(bot, store, qb, uid, chat_id, message)
        return

    chosen = chosen_map.get(str(qid))
    chosen_idx = int(chosen) if chosen is not None else 10**9

    text = (
        "📋 <b>Помилки тесту</b>\n"
        f"<b>Питання {i + 1}/{len(wrong_qids)}</b>\n\n"
        + build_feedback_text(q, "", chosen_idx)
    )
    if chosen is None:
        text += "\n\n<i>Ваш вибір не збережено (старий тест або бот перезапускався).</i>"

    buttons = []
    if i > 0:
        buttons.append(("◀️ Попереднє", "testrev:prev"))
    if i < len(wrong_qids) - 1:
        buttons.append(("▶️ Наступне", "testrev:next"))
    buttons.append(("⬅️ До результату", "testrev:back"))
    buttons.append(("⬅️ Головне меню", "nav:menu"))

    await render_main(bot, store, uid, chat_id, text, kb_inline(buttons, row=2), message=message)


@router.callback_query(F.data == "testrev:start")
async def testrev_start(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    if not await guard_access_in_session(cb, bot, store, admin_ids):
        return
    uid = cb.from_user.id
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}

    if not st.get("wrong_qids"):
        await cb.answer("Немає помилок", show_alert=True)
        return

    st["mode"] = "test_review"
    st["review_index"] = 0
    await store.set_state(uid, st)
    await cb.answer()
    await _render_test_review(bot, store, qb, uid, cb.message.chat.id, cb.message)


@router.callback_query(F.data == "testrev:next")
async def testrev_next(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    if not await guard_access_in_session(cb, bot, store, admin_ids):
        return
    uid = cb.from_user.id
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}

    st["mode"] = "test_review"
    st["review_index"] = int(st.get("review_index", 0) or 0) + 1
    await store.set_state(uid, st)
    await cb.answer()
    await _render_test_review(bot, store, qb, uid, cb.message.chat.id, cb.message)


@router.callback_query(F.data == "testrev:prev")
async def testrev_prev(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    if not await guard_access_in_session(cb, bot, store, admin_ids):
        return
    uid = cb.from_user.id
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}

    st["mode"] = "test_review"
    st["review_index"] = int(st.get("review_index", 0) or 0) - 1
    await store.set_state(uid, st)
    await cb.answer()
    await _render_test_review(bot, store, qb, uid, cb.message.chat.id, cb.message)


@router.callback_query(F.data == "testrev:back")
async def testrev_back(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    if not await guard_access_in_session(cb, bot, store, admin_ids):
        return
    uid = cb.from_user.id
    ui = await store.get_ui(uid)
    st = ui.get("state", {}) or {}

    text = st.get("result_text") or "📝 <b>Результат тесту</b>"
    wrong_qids = list(st.get("wrong_qids", []) or [])

    btns = []
    if wrong_qids:
        btns.append(("📋 Показати помилки", "testrev:start"))
    btns.append(("⬅️ Головне меню", "nav:menu"))

    st["mode"] = "test_result"
    await store.set_state(uid, st)

    await render_main(bot, store, uid, cb.message.chat.id, text, kb_inline(btns, row=1), message=cb.message)
    await cb.answer()
