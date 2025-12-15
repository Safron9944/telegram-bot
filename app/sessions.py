from __future__ import annotations

import random
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import asyncpg
from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.types import Message

from .config import EXAM_DURATION_MINUTES, EXAM_QUESTIONS
from .state import DB_POOL, QUESTIONS_BY_ID
from .utils import utcnow, html_escape, as_minutes_seconds
from .keyboards import kb_main_menu, kb_question
from .db import (
    db_create_session,
    db_finish_session,
    db_get_active_session,
    db_get_user,
    db_set_session_question_ids,
    db_update_session_progress,
)

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


