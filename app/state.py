from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.config import Settings
from app.db.engine import make_engine, make_session_factory
from app.db.base import Base

settings = Settings.load()
engine = make_engine(settings.database_url)
SessionFactory = make_session_factory(engine)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionFactory() as session:
        yield session


async def _migrate(conn) -> None:
    """Прості, ідемпотентні міграції для MVP (щоб не піднімати Alembic)."""

    # columns for questions
    await conn.execute(text("ALTER TABLE questions ADD COLUMN IF NOT EXISTS topic VARCHAR(220)"))
    await conn.execute(text("ALTER TABLE questions ADD COLUMN IF NOT EXISTS page_start INTEGER"))
    await conn.execute(text("ALTER TABLE questions ADD COLUMN IF NOT EXISTS q_number INTEGER"))
    await conn.execute(text("ALTER TABLE questions ADD COLUMN IF NOT EXISTS correct_json TEXT"))

    # default for correct_json (не критично, але зручно)
    try:
        await conn.execute(text("ALTER TABLE questions ALTER COLUMN correct_json SET DEFAULT '[]'"))
    except Exception:
        # якщо БД/права не дозволяють — ігноруємо
        pass


async def init_db() -> None:
    # створюємо таблиці (для MVP без міграцій)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate(conn)
