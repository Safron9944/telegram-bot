from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from aiogram import Bot

from utils import now


def access_tier(user: Dict[str, Any]) -> str:
    """Return the effective paid access tier: 'none' | 'cases' | 'full'."""
    if not user:
        return "none"
    inf: bool = bool(user.get("sub_infinite"))
    tier: Optional[str] = user.get("sub_tier")
    s_end: Optional[datetime] = user.get("sub_end")
    n = now()
    if inf:
        return tier if tier in ("cases", "full") else "full"
    if tier in ("cases", "full") and s_end and n <= s_end:
        return tier
    return "none"


def access_status(user: Dict[str, Any]) -> Tuple[bool, str]:
    tier = access_tier(user)
    if tier == "full":
        if bool(user.get("sub_infinite")):
            return True, "sub_infinite"
        return True, "sub_full"
    if tier == "cases":
        return True, "sub_cases"
    if not user:
        return False, "not_registered"
    return False, "expired"


def has_attestation_access(user: Dict[str, Any]) -> bool:
    """Stage 1 attestation is included in the 100-star and full tiers."""
    return access_tier(user) in ("cases", "full")


async def create_stars_invoice_link(bot: "Bot", tier: str, amount: int) -> str:
    """Create a Telegram Stars invoice link for the given tier."""
    from aiogram.types import LabeledPrice

    if tier == "cases":
        title = "Атестація"
        description = "Безлімітний доступ до першого етапу атестації"
    else:
        title = "Повний доступ"
        description = "Безлімітний доступ: навчання, тести та атестація"
    link = await bot.create_invoice_link(
        title=title,
        description=description,
        payload=tier,
        currency="XTR",
        prices=[LabeledPrice(label=title, amount=amount)],
    )
    return link


async def create_section_invoice_link(bot: "Bot", section_key: str, title: str, amount: int) -> str:
    """Create a permanent-access invoice for one configurable Mini App section."""
    from aiogram.types import LabeledPrice

    label = f"Доступ: {title}"[:32]
    return await bot.create_invoice_link(
        title=label,
        description=f"Безлімітний доступ до розділу «{title}»"[:255],
        payload=f"section:{section_key}",
        currency="XTR",
        prices=[LabeledPrice(label=label, amount=int(amount))],
    )
