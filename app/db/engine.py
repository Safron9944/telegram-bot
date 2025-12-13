from __future__ import annotations

import re
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession


def to_asyncpg_url(database_url: str) -> str:
    # Railway часто дає postgresql:// або postgres://
    url = database_url.strip()

    # Переводимо postgres:// -> postgresql://
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]

    # SQLAlchemy async driver
    if url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://") :]

    return url


def make_engine(database_url: str):
    async_url = to_asyncpg_url(database_url)
    return create_async_engine(async_url, pool_pre_ping=True)


def make_session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
