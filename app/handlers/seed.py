from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from app.state import get_session, settings
from app.db.repo import seed_demo

router = Router()


def _is_admin(message: Message) -> bool:
    return message.from_user and message.from_user.id in settings.admin_tg_ids


@router.message(Command("seed"))
async def cmd_seed(message: Message):
    """Імпорт блоків/питань з файлу data/questions_flat.json (адмін)."""
    if not _is_admin(message):
        return await message.answer("Немає доступу.")

    status = await message.answer("⏳ Імпортую питання в базу... (це може зайняти трохи часу)")

    async with get_session() as session:
        msg = await seed_demo(session)

    await status.edit_text(msg)
