from __future__ import annotations

import asyncio
import os

from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

# Re-exports for app.py backward-compatibility
from access import access_status, access_tier, create_stars_invoice_link
from handlers.admin import find_question_ids_by_title
from questions import QuestionBank
from storage import Storage
from ui import clean_law_title, ok_extract_code, ok_full_label, ok_sort_key
from utils import GROUP_URL, dt_to_iso, get_admin_contact_url, iso_to_dt, now

import handlers.nav as _nav
import handlers.session as _session
import handlers.testing as _testing
import handlers.admin as _admin


async def main():
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Set BOT_TOKEN env var")

    admin_ids_env = os.getenv("ADMIN_IDS", "").strip()
    admin_ids: set[int] = set()
    if admin_ids_env:
        for x in admin_ids_env.split(","):
            x = x.strip()
            if x.isdigit():
                admin_ids.add(int(x))

    dsn = (
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("POSTGRESQL_URL")
        or os.getenv("PGDATABASE_URL")
    )
    if not dsn:
        raise RuntimeError("Set DATABASE_URL env var (Railway → Variables).")

    store = Storage(dsn)
    await store.init()

    questions_path = os.getenv("QUESTIONS_PATH", "questions_flat.json")

    auto_import = (os.getenv("QUESTIONS_AUTO_IMPORT", "1") or "").strip().lower() in ("1", "true", "yes", "y", "on")
    force_import = (os.getenv("QUESTIONS_FORCE_IMPORT", "0") or "").strip().lower() in ("1", "true", "yes", "y", "on")

    if auto_import:
        cnt = await store.questions_count()
        if force_import or cnt == 0:
            if not os.path.exists(questions_path):
                if cnt == 0:
                    raise RuntimeError(f"Questions table is empty and file not found: {questions_path}")
            else:
                n = await store.import_questions_from_json(
                    questions_path,
                    changed_by="bootstrap",
                    force=force_import,
                )
                print(f"Questions imported/updated: {n}")

    qb = QuestionBank(questions_path)
    await qb.load_from_db(store)

    if not qb.by_id:
        raise RuntimeError("No questions loaded from DB. Fill table 'questions' first.")

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()

    dp.include_router(_nav.router)
    dp.include_router(_session.router)
    dp.include_router(_testing.router)
    dp.include_router(_admin.router)

    async def _inject_middleware(handler, event, data):
        data["store"] = store
        data["qb"] = qb
        data["admin_ids"] = admin_ids
        return await handler(event, data)

    dp.update.outer_middleware(_inject_middleware)

    print(f"Loaded: law={len(qb.law)} | ok_modules={len(qb.ok_modules)} | questions={len(qb.by_id)}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
