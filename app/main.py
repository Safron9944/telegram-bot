from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher

from app.state import settings, init_db
from app.health import start_health_server
from app.handlers.common import router as common_router
from app.handlers.admin import router as admin_router
from app.handlers.seed import router as seed_router

logging.basicConfig(level=logging.INFO)


async def main():
    await init_db()

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher()

    dp.include_router(common_router)
    dp.include_router(admin_router)
    dp.include_router(seed_router)

    # Railway healthcheck (не заважає поллінгу)
    await start_health_server(settings.port)

    logging.info("Bot started. Polling...")
    await dp.start_polling(bot)
