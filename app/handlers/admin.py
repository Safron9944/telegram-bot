from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from sqlalchemy import select

from app.state import get_session, settings
from app.db.models import User
from app.db.repo import ensure_trial_subscription
from app.services.subscriptions import grant_days, grant_lifetime, revoke

router = Router()


def _is_admin(message: Message) -> bool:
    return message.from_user and message.from_user.id in settings.admin_tg_ids


@router.message(Command("admin"))
async def admin_help(message: Message):
    if not _is_admin(message):
        return await message.answer("Немає доступу.")

    await message.answer(
        "🛠 Адмін команди:\n"
        "/seed — імпорт блоків/питань з data/questions_flat.json\n"
        "/grant <tg_id> <days> — видати підписку на N днів\n"
        "/grantlife <tg_id> — безстроково\n"
        "/revoke <tg_id> — забрати платну підписку\n"
    )


@router.message(Command("grant"))
async def cmd_grant(message: Message):
    if not _is_admin(message):
        return await message.answer("Немає доступу.")

    parts = (message.text or "").split()
    if len(parts) != 3:
        return await message.answer("Формат: /grant <tg_id> <days>")

    try:
        tg_id = int(parts[1])
        days = int(parts[2])
    except ValueError:
        return await message.answer("tg_id і days мають бути числами.")

    async with get_session() as session:
        res = await session.execute(select(User).where(User.tg_id == tg_id))
        user = res.scalar_one_or_none()
        if not user:
            return await message.answer("Користувача не знайдено. Він має хоча б раз натиснути /start.")
        await ensure_trial_subscription(session, user)
        await grant_days(session, user, days)

    await message.answer(f"Готово ✅ Видано {days} днів для {tg_id}.")


@router.message(Command("grantlife"))
async def cmd_grant_life(message: Message):
    if not _is_admin(message):
        return await message.answer("Немає доступу.")

    parts = (message.text or "").split()
    if len(parts) != 2:
        return await message.answer("Формат: /grantlife <tg_id>")

    try:
        tg_id = int(parts[1])
    except ValueError:
        return await message.answer("tg_id має бути числом.")

    async with get_session() as session:
        res = await session.execute(select(User).where(User.tg_id == tg_id))
        user = res.scalar_one_or_none()
        if not user:
            return await message.answer("Користувача не знайдено. Він має хоча б раз натиснути /start.")
        await ensure_trial_subscription(session, user)
        await grant_lifetime(session, user)

    await message.answer(f"Готово ✅ Безстроковий доступ для {tg_id}.")


@router.message(Command("revoke"))
async def cmd_revoke(message: Message):
    if not _is_admin(message):
        return await message.answer("Немає доступу.")

    parts = (message.text or "").split()
    if len(parts) != 2:
        return await message.answer("Формат: /revoke <tg_id>")

    try:
        tg_id = int(parts[1])
    except ValueError:
        return await message.answer("tg_id має бути числом.")

    async with get_session() as session:
        res = await session.execute(select(User).where(User.tg_id == tg_id))
        user = res.scalar_one_or_none()
        if not user:
            return await message.answer("Користувача не знайдено.")
        await ensure_trial_subscription(session, user)
        await revoke(session, user)

    await message.answer(f"Готово ✅ Підписка забрана для {tg_id}.")
