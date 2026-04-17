"""Точка входа: инициализация БД, регистрация хэндлеров, запуск polling + APScheduler."""

from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.handlers import discussion, feed, onboarding, preferences, report
from config import settings
from db.database import init_db
from scheduler import setup_scheduler

log = logging.getLogger(__name__)


def _setup_logging() -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Чистим старые хэндлеры чтобы dev-reload не плодил дубли.
    for h in list(root.handlers):
        root.removeHandler(h)

    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(logging.Formatter(fmt))
    root.addHandler(stream_h)

    # Файловый хэндлер — чтобы логи пережили завершение процесса.
    file_h = logging.FileHandler("bot.log", mode="a", encoding="utf-8")
    file_h.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_h)

    # Понижаем шум от либ.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)


async def main() -> None:
    _setup_logging()

    await init_db()

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    # Порядок важен: onboarding идёт ПОСЛЕДНИМ — у него catch-all F.text, иначе
    # он перехватит и reply-обсуждения, и клики по кнопкам.
    # discussion.router раньше onboarding — чтобы reply-вопросы маршрутизировались туда.
    dp.include_router(feed.router)
    dp.include_router(report.router)
    dp.include_router(preferences.router)
    dp.include_router(discussion.router)
    dp.include_router(onboarding.router)

    scheduler = setup_scheduler(bot)
    scheduler.start()

    log.info("bot starting...")
    # Drop накопившиеся callback-апдейты — иначе после рестарта валимся на
    # «query is too old» для кнопок нажатых ДО рестарта.
    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception as e:
        log.warning("delete_webhook: %s", e)
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        scheduler.shutdown(wait=False)
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("shutdown")
