from aiogram.types import (
    ReplyKeyboardMarkup, KeyboardButton,
    InlineKeyboardMarkup, InlineKeyboardButton
)

def kb_request_phone() -> ReplyKeyboardMarkup:
    # ЄДИНЕ місце, де використовується Reply keyboard (для контакту)
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📞 Поділитися номером", request_contact=True)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )

def ik_main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🧩 Тести", callback_data="menu:tests")],
        [InlineKeyboardButton(text="👤 Кабінет", callback_data="menu:cabinet")],
        [InlineKeyboardButton(text="❗ Помилки", callback_data="menu:mistakes")],
        [InlineKeyboardButton(text="ℹ️ Допомога", callback_data="menu:help")],
    ])

def ik_subscribe() -> InlineKeyboardMarkup:
    # Кнопки “на екрані” у кабінеті
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💳 Оплатити (поки демо)", callback_data="pay_demo")],
        [InlineKeyboardButton(text="⬅️ Головне меню", callback_data="menu:home")],
    ])
