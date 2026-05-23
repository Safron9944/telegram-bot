from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from aiogram import Bot
from aiogram.types import LabeledPrice

from utils import now


def access_tier(user: Dict[str, Any]) -> str:
    """Return the effective access tier: 'none' | 'trial_full' | 'cases' | 'full'."""
    if not user:
        return "none"
    inf: bool = bool(user.get("sub_infinite"))
    tier: Optional[str] = user.get("sub_tier")
    s_end: Optional[datetime] = user.get("sub_end")
    t_end: Optional[datetime] = user.get("trial_end")
    n = now()
    if inf:
        return "full"
    if tier in ("cases", "full") and s_end and n <= s_end:
        return tier
    if t_end and n <= t_end:
        return "trial_full"  # навчання+тест, але НЕ кейси
    return "none"


def access_status(user: Dict[str, Any]) -> Tuple[bool, str]:
    tier = access_tier(user)
    if tier == "full":
        if bool(user.get("sub_infinite")):
            return True, "sub_infinite"
        return True, "sub_full"
    if tier == "cases":
        return True, "sub_cases"
    if tier == "trial_full":
        return True, "trial"
    if not user:
        return False, "not_registered"
    return False, "expired"


async def create_stars_invoice_link(bot: "Bot", tier: str, amount: int) -> str:
    """Create a Telegram Stars invoice link for the given tier."""
    if tier == "cases":
        title = "Доступ до кейсів"
        description = "30 днів доступу до розділу Кейси"
    else:
        title = "Повний доступ"
        description = "30 днів повного доступу (навчання, тести, кейси)"
    link = await bot.create_invoice_link(
        title=title,
        description=description,
        payload=tier,
        currency="XTR",
        prices=[LabeledPrice(label=title, amount=amount)],
    )
    return link
