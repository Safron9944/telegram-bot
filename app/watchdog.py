from __future__ import annotations

import asyncio
import logging

import asyncpg
from aiogram import Bot

from .utils import utcnow
from .db import db_get_active_session
from .sessions import finish_exam_due_to_timeout


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


