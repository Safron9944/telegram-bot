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

def kb_main() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="👤 Кабінет"), KeyboardButton(text="🧩 Тести")],
            [KeyboardButton(text="❗ Помилки"), KeyboardButton(text="ℹ️ Допомога")],
        ],
        resize_keyboard=True,
    )

def kb_admin() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🛠 Адмін: користувач"), KeyboardButton(text="🧾 Адмін: підписка")],
        ],
        resize_keyboard=True,
    )

def ik_subscribe() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="💳 Оплатити (поки демо)", callback_data="pay_demo")],
        ]
    )
