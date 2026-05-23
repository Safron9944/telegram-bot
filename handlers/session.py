from __future__ import annotations

import re
from html import escape as hescape
from typing import Any, Dict, List, Optional

from aiogram import Bot, F, Router
from aiogram.types import CallbackQuery, Message

from access import access_status
from questions import Q, QuestionBank
from storage import Storage
from ui import (
    build_feedback_text,
    build_question_text,
    kb_answers,
    kb_feedback,
    kb_inline,
    kb_leave_confirm,
    render_main,
    screen_main_menu,
    screen_no_access,
    ok_extract_code,
)
from utils import dt_to_iso, get_admin_contact_url, iso_to_dt, now

router = Router()


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
        admin_ids: Optional[set[int]] = None,
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
    await show_next_in_session(bot, store, qb, uid, chat_id, message, admin_ids=admin_ids)


_BLOCK_NUM_RE = re.compile(r"(\d+)")


def _test_block_sort_key(name: str) -> tuple:
    low = (name or "").lower().strip()
    is_law = low.startswith("закон") or low.startswith("law")
    m = _BLOCK_NUM_RE.search(low)
    num = int(m.group(1)) if m else 10**9
    return (1 if is_law else 0, num, low)


def _derive_test_blocks_from_answers(qb: "QuestionBank", answers: dict) -> dict[str, list[int]]:
    blocks: dict[str, list[int]] = {}
    for qid_s in (answers or {}).keys():
        try:
            qid = int(qid_s)
        except Exception:
            continue
        q = qb.by_id.get(qid)
        name = (q.ok if (q and q.ok) else "Законодавство")
        blocks.setdefault(name, []).append(qid)
    return blocks


def _format_test_blocks(blocks: dict[str, list[int]], answers: dict) -> str:
    if not blocks:
        return ""

    out_lines: list[str] = []
    for name in sorted(blocks.keys(), key=_test_block_sort_key):
        qids = blocks.get(name) or []
        total = len(qids)
        correct = sum(1 for qid in qids if answers.get(str(qid)) is True)
        out_lines.append(f"• <b>{hescape(str(name))}</b>: ✅ <b>{correct}</b> з <b>{total}</b>")
    return "\n".join(out_lines)


async def show_next_in_session(
    bot: Bot,
    store: Storage,
    qb: QuestionBank,
    uid: int,
    chat_id: int,
    message: Message,
    admin_ids: Optional[set[int]] = None
):
    ui = await store.get_ui(uid)
    st = ui.get("state", {})
    mode = st.get("mode")

    admin_ids = admin_ids or set()
    is_admin = uid in admin_ids

    if mode not in ("learn", "test", "mistakes"):
        return

    if st.get("feedback"):
        fb = st["feedback"]
        qid_fb = fb.get("qid")
        chosen = fb.get("chosen")

        q = qb.by_id.get(int(qid_fb)) if qid_fb is not None else None

        if q is not None and chosen is not None:
            text = build_feedback_text(q, st.get("header", ""), int(chosen))
        else:
            text = f"{st.get('header', '')}\n❌ <b>Неправильно</b>"

        edit_cb = (f"admin:qedit:{int(qid_fb)}:sess:0" if (is_admin and qid_fb is not None) else None)
        await render_main(bot, store, uid, chat_id, text, kb_feedback(edit_cb=edit_cb), message=message)
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
            await show_next_in_session(bot, store, qb, uid, chat_id, message, admin_ids=admin_ids)
            return

        if mode == "test":
            correct = int(st.get("correct_count", 0))
            total = int(st.get("total", 0))
            percent = (correct / total * 100.0) if total else 0.0
            passed = percent >= 60.0

            started_at = iso_to_dt(st.get("started_at")) or now()
            finished_at = now()
            await store.save_test(uid, started_at, finished_at, total, correct)

            answers = st.get("answers", {}) or {}
            blocks = st.get("test_blocks", {}) or {}
            if not blocks and answers:
                blocks = _derive_test_blocks_from_answers(qb, answers)

            blocks_text = _format_test_blocks(blocks, answers)
            blocks_section = (f"\n\n📦 <b>По блоках:</b>\n{blocks_text}" if blocks_text else "")

            text = (
                "📝 <b>Тестування завершено</b>\n"
                f"{blocks_section}\n\n"
                "📌 <b>Підсумок:</b>\n"
                f"✅ <b>{correct}</b> з <b>{total}</b> питань\n"
                f"📈 <b>{percent:.1f}%</b>\n"
                f"<b>{'✅ Тест складено' if passed else '❌ Тест не складено'}</b>"
            )

            answers = st.get("answers", {}) or {}
            chosen_map = st.get("chosen", {}) or {}

            wrong_qids = []
            for qid_s, ok in answers.items():
                try:
                    if not bool(ok):
                        wrong_qids.append(int(qid_s))
                except Exception:
                    pass

            btns = []
            if wrong_qids:
                btns.append(("📋 Показати помилки", "testrev:start"))
            btns.append(("⬅️ Головне меню", "nav:menu"))

            result_state = {
                "mode": "test_result",
                "result_text": text,
                "wrong_qids": wrong_qids,
                "chosen": chosen_map,
            }
            await store.set_state(uid, result_state)

            await render_main(
                bot, store, uid, chat_id,
                text,
                kb_inline(btns, row=1),
                message=message
            )
            return

        if mode == "mistakes":
            answers = st.get("answers", {})
            total = len(answers)
            correct_ids = [qid for qid, ok in answers.items() if ok]
            wrong_ids = [qid for qid, ok in answers.items() if not ok]
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
            await render_main(
                bot, store, uid, chat_id, text,
                kb_inline([("⬅️ Навчання", "nav:learn")], row=1),
                message=message
            )
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

        meta = st.get("meta") or {}
        kind = meta.get("kind")

        buttons = []

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

        elif kind in ("law", "lawrand"):
            group_key = meta.get("group")
            if group_key:
                buttons.append(("⬅️ До частин", f"lawgrp:{group_key}"))

        buttons.append(("⬅️ Навчання", "nav:learn"))

        await store.set_state(uid, {})
        await render_main(
            bot, store, uid, chat_id,
            text,
            kb_inline(buttons, row=1),
            message=message
        )
        return

    while pending:
        qid = int(pending[0])
        q = qb.by_id.get(qid)
        if q is not None and (q.choices or []):
            break
        pending = pending[1:]

    if not pending:
        st["pending"] = []
        await store.set_state(uid, st)
        await show_next_in_session(bot, store, qb, uid, chat_id, message, admin_ids=admin_ids)
        return

    qid = int(pending[0])
    q = qb.by_id.get(qid)

    st["pending"] = pending
    st["current_qid"] = qid
    await store.set_state(uid, st)

    total = int(st.get("total", len(pending)) or 0)
    idx = (total - len(pending) + 1) if total else 1
    repeat_note = " • <i>повтор</i>" if (mode == "learn" and phase == "skipped") else ""
    progress = f"<b>Питання {idx}/{total}</b>{repeat_note}"

    text = build_question_text(q, st.get("header", ""), progress)
    edit_cb = (f"admin:qedit:{int(qid)}:sess:0" if is_admin else None)
    kb = kb_answers(len(q.choices or []), allow_skip=(mode == "learn"), edit_cb=edit_cb)
    await render_main(bot, store, uid, chat_id, text, kb, message=message)


# -------- Pre-test (question picker) --------

def _qpick_kb(total: int, selected: int, back_cb: Optional[str]) -> Any:
    from aiogram.utils.keyboard import InlineKeyboardBuilder
    from aiogram.types import InlineKeyboardButton
    from utils import clamp_callback

    total = max(0, int(total))
    selected = max(0, min(int(selected), max(0, total - 1))) if total else 0

    b = InlineKeyboardBuilder()

    for i in range(total):
        label = str(i + 1)
        b.button(text=label, callback_data=clamp_callback(f"qpick:go:{i + 1}"))

    cols = 8
    if total:
        full_rows, remainder = divmod(total, cols)
        sizes = [cols] * full_rows
        if remainder:
            sizes.append(remainder)
        b.adjust(*sizes)
    else:
        b.adjust(1)

    b.row(InlineKeyboardButton(text="▶️ Розпочати тестування", callback_data="qpick:start"))
    if back_cb:
        b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data=back_cb))

    return b.as_markup()


def screen_qpick_grid(header: str, total: int, selected: int = 0, back_cb: Optional[str] = None):
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
        is_admin: bool = False,
):
    from html import escape as hescape

    idx_1based = max(1, int(idx_1based))
    total = max(1, int(total))

    correct_set = set(int(x) for x in (q.correct or []))
    opts_lines: List[str] = []
    for i, ch in enumerate(q.choices or []):
        mark = "✅" if (i + 1) in correct_set else "▫️"
        note = " <i>(правильно)</i>" if (i + 1) in correct_set else ""
        opts_lines.append(f"{mark} <b>{i + 1})</b> {hescape(ch)}{note}")
    opts = "\n".join(opts_lines) if opts_lines else "—"
    corr_line = ", ".join(str(x) for x in sorted(correct_set)) if correct_set else "—"

    text = (
        "📝 <b>Підготовка до тесту</b>\n\n"
        f"{header}\n\n"
        f"<b>Питання {idx_1based}/{total}</b>\n"
        f"{hescape(q.question)}\n\n"
        "📝 <b>Варіанти</b>\n"
        f"{opts}\n\n<b>Правильні:</b> <code>{corr_line}</code>"
    )

    buttons: List[Any] = [
        ("⬅️ До списку питань", "qpick:show"),
        ("▶️ Розпочати тестування", "qpick:start"),
    ]
    if is_admin:
        buttons.insert(0, ("✏️ Змінити це питання", f"admin:qedit:{int(q.id)}:qpick:{int(idx_1based)}"))
    if back_cb:
        buttons.append(("⬅️ Назад", back_cb))

    return text, kb_inline(buttons, row=2)


def pretest_questions_limit(meta: dict) -> int:
    if str(meta.get("kind") or "").lower() == "ok":
        module = str(meta.get("module") or "")
        if ok_extract_code(module) == "ОК-17":
            return 70
    return 50


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
    limit = pretest_questions_limit(meta)
    qids = list(qids or [])[:limit]

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
        "selected": 0,
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
        admin_ids: Optional[set[int]] = None,
):
    qids = list(pre.get("qids", []) or [])
    if not qids:
        return

    selected = int(pre.get("selected", 0) or 0)
    selected = max(0, min(selected, len(qids) - 1))

    ordered = qids[selected:] + qids[:selected]

    header = pre.get("header", "📝 <b>Тестування</b>")
    meta = pre.get("meta", {}) or {}

    await start_learning_session(
        bot, store, qb,
        uid, chat_id, message,
        qids=ordered,
        header=header,
        save_meta=meta,
        admin_ids=admin_ids,
    )


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

    await store.set_state(uid, {})
    admin_url = get_admin_contact_url(admin_ids)
    text, kb = screen_no_access(user, admin_url)
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)

    await cb.answer("Доступ завершився. Потрібна підписка.", show_alert=True)
    return None


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
    text, kb = screen_qpick_preview(header, q, idx_1based=idx0 + 1, total=total, back_cb=back_cb,
                                    is_admin=(uid in admin_ids))
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
    await start_test_from_pretest(bot, store, qb, uid, cb.message.chat.id, cb.message, st, admin_ids=admin_ids)


@router.callback_query(F.data.startswith("learn_start:"))
async def learn_start(
        cb: CallbackQuery,
        bot: Bot,
        store: Storage,
        qb: QuestionBank,
        admin_ids: set[int],
):
    await cb.answer()

    uid = cb.from_user.id
    user = await store.get_user(uid)

    ok_access, _ = access_status(user)
    if not ok_access:
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

    if kind == "law":
        group_key = parts[2] if len(parts) >= 3 else ""
        try:
            part = int(parts[3]) if len(parts) >= 4 else 1
        except Exception:
            part = 1

        all_qids = qb.law_groups.get(group_key, []) or []
        start = max(0, (part - 1) * 50)
        qids = all_qids[start:start + 50]

        title = qb.law_group_title(group_key)
        header = hescape(title)

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
        group_key = parts[2] if len(parts) >= 3 else ""
        all_qids = qb.law_groups.get(group_key, []) or []
        qids = qb.pick_random(all_qids, min(50, len(all_qids)))

        title = qb.law_group_title(group_key)
        header = f"{hescape(title)} • 🎲"

        await start_pretest(
            bot, store, qb,
            uid, cb.message.chat.id, cb.message,
            qids=qids,
            header=header,
            meta={"kind": "lawrand", "group": group_key},
            back_cb=f"lawgrp:{group_key}" if group_key else "nav:learn",
        )
        return

    if kind == "ok":
        module: Optional[str] = None
        level = 1
        back_cb: str = "nav:learn"

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

        else:
            module = parts[2] if len(parts) >= 3 else None
            if len(parts) >= 4:
                try:
                    level = int(parts[3])
                except Exception:
                    level = 1

            modules = user.get("ok_modules", []) or []
            if module in modules:
                back_cb = f"okmodi:{modules.index(module)}"

        if not module:
            await cb.answer("Модуль не знайдено", show_alert=True)
            return

        qids = (qb.ok_modules.get(module, {}) or {}).get(int(level), []) or []
        limit = 70 if ok_extract_code(module) == "ОК-17" else 50
        qids = list(qids)[:limit]

        await store.set_ok_last_level(uid, module, int(level))

        header = f"{hescape(module)} • Рівень {int(level)}"

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
        await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message, admin_ids=admin_ids)
        return

    if mode == "test":
        answers = st.get("answers", {}) or {}
        answers[str(qid)] = bool(is_correct)
        st["answers"] = answers

        chosen_map = st.get("chosen", {}) or {}
        chosen_map[str(qid)] = int(choice)
        st["chosen"] = chosen_map

        if is_correct:
            st["correct_count"] = int(st.get("correct_count", 0)) + 1

        await store.set_state(uid, st)
        await cb.answer()
        await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message, admin_ids=admin_ids)
        return

    if mode == "mistakes":
        answers = st.get("answers", {})
        answers[str(qid)] = bool(is_correct)
        st["answers"] = answers
        await store.set_state(uid, st)
        await cb.answer()
        await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message, admin_ids=admin_ids)
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
    if pending and int(pending[0]) == int(qid):
        pending = pending[1:]
    skipped.append(int(qid))

    st["pending"] = pending
    st["skipped"] = skipped
    st["feedback"] = None
    await store.set_state(uid, st)
    await cb.answer("Пропущено")
    await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message, admin_ids=admin_ids)


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
    await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message, admin_ids=admin_ids)


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
async def leave_back(cb: CallbackQuery, bot: Bot, store: Storage, qb: QuestionBank, admin_ids: set[int]):
    await cb.answer()
    await show_next_in_session(bot, store, qb, cb.from_user.id, cb.message.chat.id, cb.message, admin_ids=admin_ids)


@router.callback_query(F.data == "leave:yes")
async def leave_yes(cb: CallbackQuery, bot: Bot, store: Storage, admin_ids: set[int]):
    uid = cb.from_user.id
    await store.set_state(uid, {})
    user = await store.get_user(uid)
    text, kb = screen_main_menu(user, is_admin=(uid in admin_ids))
    await render_main(bot, store, uid, cb.message.chat.id, text, kb, message=cb.message)
    await cb.answer()


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
    await show_next_in_session(bot, store, qb, uid, cb.message.chat.id, cb.message, admin_ids=admin_ids)
