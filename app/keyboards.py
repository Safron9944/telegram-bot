from __future__ import annotations

from aiogram.types import (
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)

def kb_request_phone() -> ReplyKeyboardMarkup:
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
