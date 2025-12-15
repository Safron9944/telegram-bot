from __future__ import annotations

import asyncpg

from .config import OK_CODE_LAW, LEVEL_ALL
from .utils import scope_title


# -------------------------
# Логіка доступу/профілю
# -------------------------
def user_has_scope(user: asyncpg.Record) -> bool:
    return bool(user["ok_code"])

def get_user_scope(user: asyncpg.Record) -> tuple[str, int]:
    ok_code = str(user["ok_code"])
    # для LAW рівень завжди 0
    if ok_code == OK_CODE_LAW:
        return ok_code, 0

    lvl = user["ok_level"]
    # якщо рівень не вказаний — трактуємо як «всі рівні»
    if lvl is None:
        lvl = LEVEL_ALL
    return ok_code, int(lvl)

async def ensure_profile(message: Message, user: asyncpg.Record, next_mode: str | None = None) -> bool:
    if user_has_scope(user):
        return True

    if next_mode in ("train", "exam"):
        PENDING_AFTER_OK[int(user["tg_id"])] = next_mode

    await message.answer(
        "⚙️ Потрібно обрати <b>ОК</b>, бо для кожного набір питань різний.\n\n"
        "Оберіть ОК:",
        parse_mode=ParseMode.HTML,
        reply_markup=ReplyKeyboardRemove(),
    )
    await message.answer("ОК:", reply_markup=kb_pick_ok(page=0))
    return False



