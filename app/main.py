from __future__ import annotations

import asyncio
import logging

import asyncpg
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from .config import BOT_TOKEN, DATABASE_URL
from .handlers import router
from .state import DB_POOL, DISABLED_IDS_DB, QUESTIONS_BY_ID, VALID_QIDS, PROBLEM_IDS_FILE, OK_CODES
from .questions import load_question_bank
from .db import db_init, db_seed_problem_flags, db_get_disabled_ids
from .watchdog import exam_watchdog


async def on_startup(bot: Bot, dp: Dispatcher) -> None:
    if not BOT_TOKEN or not DATABASE_URL:
        raise RuntimeError("BOT_TOKEN або DATABASE_URL не задані.")

    # 1) Питання (файл)
    load_question_bank()

    # 2) БД
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
    DB_POOL.set(pool)
    await db_init(DB_POOL)

    # 3) Позначення проблемних і вимкнених питань
    await db_seed_problem_flags(DB_POOL, PROBLEM_IDS_FILE)
    disabled = await db_get_disabled_ids(DB_POOL)
    DISABLED_IDS_DB.clear()
    DISABLED_IDS_DB.update(disabled)

    # 4) Watchdog для екзаменів
    dp.workflow_data["exam_watchdog_task"] = asyncio.create_task(exam_watchdog(bot, DB_POOL))

    logging.info(
        "Startup done. Questions total=%d, valid=%d, problems=%d, disabled_db=%d, ok_codes=%d",
        len(QUESTIONS_BY_ID), len(VALID_QIDS), len(PROBLEM_IDS_FILE), len(DISABLED_IDS_DB), len(OK_CODES),
    )


async def on_shutdown(bot: Bot, dp: Dispatcher) -> None:
    task = dp.workflow_data.get("exam_watchdog_task")
    if task:
        task.cancel()
    if DB_POOL:
        await DB_POOL.close()


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    bot = Bot(
        BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher()
    dp.include_router(router)

    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    await dp.start_polling(bot, dp=dp)
