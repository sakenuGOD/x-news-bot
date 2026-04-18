"""Async engine + session factory. Инициализация схемы при старте бота."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings
from db.models import Base

log = logging.getLogger(__name__)

engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Лёгкие in-place миграции — добавляем недостающие колонки на существующих БД.
        # SQLAlchemy create_all не трогает существующие таблицы, поэтому ALTER-им вручную.
        for ddl in (
            "ALTER TABLE tweets ADD COLUMN media_type VARCHAR(16)",
            "ALTER TABLE users ADD COLUMN topic_dislikes TEXT DEFAULT '{}'",
            "ALTER TABLE tweets ADD COLUMN quote_image_url VARCHAR(512)",
            "ALTER TABLE tweets ADD COLUMN quote_media_type VARCHAR(16)",
            "ALTER TABLE tweets ADD COLUMN quote_author VARCHAR(64)",
            "ALTER TABLE tweets ADD COLUMN quote_text VARCHAR(1024)",
            "ALTER TABLE users ADD COLUMN blocked_authors TEXT DEFAULT '[]'",
            "ALTER TABLE sent_news ADD COLUMN quote_telegram_message_id INTEGER",
            "ALTER TABLE tweets ADD COLUMN linked_tweet_id VARCHAR(32)",
        ):
            try:
                await conn.exec_driver_sql(ddl)
                log.info("migration applied: %s", ddl)
            except Exception:
                pass  # колонка уже есть — это нормально
    log.info("database initialized at %s", settings.database_url)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Использовать как: `async with session_scope() as s: ...`"""
    session = SessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
