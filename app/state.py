from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import asyncpg


@dataclass
class PoolRef:
    """Посилання на asyncpg.Pool, яке можна 'встановити' на старті.

    Потрібно, щоб у різних модулях можна було робити `from app.state import DB_POOL`
    і при цьому після старту бот бачив реальний пул (без переписування всієї логіки).
    """
    pool: Optional[asyncpg.Pool] = None

    def set(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    def __bool__(self) -> bool:  # дозволяє `if not DB_POOL:`
        return self.pool is not None

    def __getattr__(self, name: str):
        if self.pool is None:
            raise AttributeError(f"DB_POOL не ініціалізовано, немає атрибуту '{name}'")
        return getattr(self.pool, name)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None


# Глобальний пул БД (встановлюється на startup)
DB_POOL = PoolRef()

# Глобальні кеші (заповнюються на старті)
QUESTIONS_BY_ID: Dict[int, Dict[str, Any]] = {}
VALID_QIDS: List[int] = []  # валідні (1 правильна відповідь) і не в problem файлі

# scope = (ok_code, level_int)
OK_CODES: List[str] = []
LEVELS_BY_OK: Dict[str, List[int]] = {}
TOPICS_BY_SCOPE: Dict[Tuple[str, int], List[str]] = {}
QIDS_BY_SCOPE: Dict[Tuple[str, int], List[int]] = {}
QIDS_BY_SCOPE_TOPIC: Dict[Tuple[str, int, str], List[int]] = {}

PROBLEM_IDS_FILE: Set[int] = set()
DISABLED_IDS_DB: Set[int] = set()
