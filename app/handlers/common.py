from __future__ import annotations

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, ReplyKeyboardRemove

from app.keyboards import kb_request_phone, ik_main_menu, ik_subscribe
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

    # меню “на екрані”
    await message.answer("Головне меню", reply_markup=ik_main_menu())


@router.message(Command("menu"))
async def cmd_menu(message: Message):
    await message.answer("Головне меню", reply_markup=ik_main_menu())


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

    # ховаємо reply-клавіатуру після реєстрації
    await message.answer("Дякую! Реєстрація завершена ✅", reply_markup=ReplyKeyboardRemove())
    await message.answer("Головне меню", reply_markup=ik_main_menu())


# ---------- INLINE МЕНЮ (кнопки "на екрані") ----------

@router.callback_query(F.data == "menu:home")
async def menu_home(cb: CallbackQuery):
    await cb.answer()
    await cb.message.edit_text("Головне меню", reply_markup=ik_main_menu())


@router.callback_query(F.data == "menu:cabinet")
async def menu_cabinet(cb: CallbackQuery):
    async with get_session() as session:
        user = await get_or_create_user(session, cb.from_user.id, cb.from_user.full_name)
        sub = await ensure_trial_subscription(session, user)

        phone = user.phone or "не вказано"
        status = format_status(sub)

    await cb.answer()
    await cb.message.edit_text(
        "👤 *Кабінет*\n"
        f"Телефон: `{phone}`\n"
        f"Статус: {status}\n",
        parse_mode="Markdown",
        reply_markup=ik_subscribe(),
    )


@router.callback_query(F.data == "menu:tests")
async def menu_tests(cb: CallbackQuery):
    await cb.answer()
    await cb.message.edit_text(
        "🧪 Тести\n(далі зробимо: Навчання / Екзамен / вибір блоків)",
        reply_markup=None,  # прибирає inline-кнопки
    )


@router.callback_query(F.data == "menu:mistakes")
async def menu_mistakes(cb: CallbackQuery):
    await cb.answer()
    await cb.message.edit_text("❗ Помилки (поки демо)", reply_markup=ik_main_menu())


@router.callback_query(F.data == "menu:help")
async def menu_help(cb: CallbackQuery):
    await cb.answer()
    await cb.message.edit_text(
        "ℹ️ Допомога\n\n"
        "• /start — реєстрація/старт\n"
        "• /menu — головне меню\n",
        reply_markup=ik_main_menu(),
    )



