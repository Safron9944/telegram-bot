from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _parse_admin_ids(raw: str | None) -> List[int]:
    if not raw:
        return []
    ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            pass
    return ids


@dataclass(frozen=True)
class Settings:
    bot_token: str
    database_url: str
    admin_tg_ids: List[int]
    port: int

    @staticmethod
    def load() -> "Settings":
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url:
            raise RuntimeError("DATABASE_URL is not set")

        admin_tg_ids = _parse_admin_ids(os.getenv("ADMIN_TG_IDS"))

        port = int(os.getenv("PORT", "8080"))
        return Settings(
            bot_token=bot_token,
            database_url=database_url,
            admin_tg_ids=admin_tg_ids,
            port=port,
        )
