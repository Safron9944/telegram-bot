from __future__ import annotations

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery

from app.keyboards import kb_request_phone, kb_main, ik_subscribe
from app.db.repo import get_or_create_user, ensure_trial_subscription, set_phone
from app.services.subscriptions import format_status
from app.state import get_session

router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message):
    async with get_session() as session:
        user = await get_or_create_user(session, message.from_user.id, message.from_user.full_name)
        await ensure_trial_subscription(session, user)

        if not user.phone:
            await message.answer(
                "Привіт! Щоб продовжити — поділись номером телефону (кнопка нижче).",
                reply_markup=kb_request_phone(),
            )
            return

        await message.answer("Готово ✅ Обери дію в меню.", reply_markup=kb_main())


@router.message(F.contact)
async def on_contact(message: Message):
    # важливо: приймаємо тільки власний контакт користувача
    if not message.contact or message.contact.user_id != message.from_user.id:
        await message.answer("Будь ласка, надішли *свій* номер через кнопку.", parse_mode="Markdown")
        return

    async with get_session() as session:
        user = await get_or_create_user(session, message.from_user.id, message.from_user.full_name)
        await ensure_trial_subscription(session, user)
        await set_phone(session, user, message.contact.phone_number)

    await message.answer("Дякую! Реєстрація завершена ✅", reply_markup=kb_main())


@router.message(F.text == "👤 Кабінет")
async def cabinet(message: Message):
    async with get_session() as session:
        user = await get_or_create_user(session, message.from_user.id, message.from_user.full_name)
        sub = await ensure_trial_subscription(session, user)

        phone = user.phone or "не вказано"
        status = format_status(sub)

    await message.answer(
        "👤 *Кабінет*\n"
        f"Телефон: `{phone}`\n"
        f"Статус: {status}\n",
        parse_mode="Markdown",
        reply_markup=ik_subscribe(),
    )


@router.callback_query(F.data == "pay_demo")
async def pay_demo(cb: CallbackQuery):
    await cb.answer()
    await cb.message.answer("Оплата поки в демо. Адмін може видати підписку командою /grant.")
