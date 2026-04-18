"""Хендлеры флоу отчёта и главного меню.

Callback data формат:
  rep:new              — сгенерить свежий отчёт
  rep:topic:<cid>      — показать тему cid (раскрыть summary + посты)
  rep:back             — вернуться к overview-отчёту
  rep:feed             — просто лента (без группировки)
  menu:main            — показать главное меню
  menu:interval        — сменить интервал
  menu:reset           — /reset через кнопку
"""

from __future__ import annotations

import html
import logging

from aiogram import F, Router
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError
from aiogram.types import CallbackQuery, Message
from sqlalchemy import select

from bot.delivery import cleanup_post_by_message, send_one_tweet
from bot.keyboards import (
    interval_kb,
    main_menu_kb,
    report_overview_kb,
    super_topics_kb,
    super_topic_details_kb,
    topic_paginator_kb,
)
from core import ai_client
from core.report import (
    Report,
    build_report,
    get_report,
    save_report,
)
from db.database import session_scope
from db.models import Feedback, Tweet, User

log = logging.getLogger(__name__)

router = Router(name="report")


# ============================= helpers =============================


def _format_overview_text(report: Report, digest: bool = False) -> str:
    """Компактный обзор. И в digest, и в обычном режиме — только заголовок +
    подсказка. Саммари темы показывается при клике на кнопку (в _render_topic_open),
    чтобы overview не дублировал summary текстом И кнопками одновременно.
    """
    when = report.generated_at.strftime("%H:%M")
    lines = [
        f"<b>📊 Дайджест · {when}</b>" if digest else f"<b>📊 Отчёт · {when}</b>",
        f"<i>за последние {report.window_hours:g} ч из твоей ленты</i>",
    ]
    if not report.clusters:
        lines.append("")
        lines.append("<i>Тем не набралось — либо лента спокойная, либо всё "
                     "отфильтровалось. Попробуй через час.</i>")
    else:
        lines.append("")
        lines.append("<i>Жми на тему — покажу выжимку и посты.</i>")
    return "\n".join(lines)


def _overview_kb_meta(report: Report) -> list[tuple[int, str, str, int]]:
    return [(c.id, c.emoji, c.name, len(c.tweet_ids)) for c in report.clusters]


async def _render_report(message: Message, report: Report, *, digest: bool = False,
                         force_flat: bool = False) -> None:
    """Отправляет готовый отчёт — всегда плоский список тем.

    Super-topics отключены: Claude склеивал «Bitcoin + XRP + Jensen Huang» в
    «IT/Технологии» и юзер видел абстрактные ярлыки вместо реальных тем.
    Плоский список показывает honestly что в ленте.
    """
    text = _format_overview_text(report, digest=digest)
    kb = report_overview_kb(_overview_kb_meta(report))
    await message.answer(
        text, parse_mode=ParseMode.HTML, reply_markup=kb, disable_web_page_preview=True,
    )


# ============================= callbacks =============================


@router.callback_query(F.data == "rep:new")
async def cb_new_report(cb: CallbackQuery) -> None:
    """«Что обсуждают» — For You за 12ч, кластеризация по темам.

    Свежий снапшот того что сейчас обсуждают в X. Окно короткое — приоритет
    сейчасному. Upfront-выжимка Claude по топ-7 темам сразу в overview.
    """
    await _run_report_flow(cb, source="for_you", window_hours=20.0, limit_raw=400,
                           auto_summarize_top=7)


async def _run_report_flow(cb: CallbackQuery, *, source: str,
                           window_hours: float, limit_raw: int,
                           digest: bool | None = None,
                           auto_summarize_top: int = 7) -> None:
    # Если digest не указан явно — включаем когда есть upfront summaries.
    if digest is None:
        digest = auto_summarize_top > 0
    user_id = cb.from_user.id
    await cb.answer()

    try:
        await cb.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    status = await cb.message.answer("⏳ Поднимаю данные…")

    async def _progress(msg: str) -> None:
        try:
            await status.edit_text(msg)
        except TelegramAPIError:
            pass

    async with session_scope() as s:
        user = await s.get(User, user_id)
        if not user:
            await status.edit_text("Сначала /start.")
            return
        try:
            report = await build_report(
                s, user,
                window_hours=window_hours,
                limit_raw=limit_raw,
                progress=_progress,
                source=source,
                auto_summarize_top=auto_summarize_top,
            )
        except Exception as e:
            log.exception("build_report failed for %s", user_id)
            await status.edit_text(f"❌ Сбой при сборке: <code>{type(e).__name__}</code>",
                                   parse_mode=ParseMode.HTML)
            return

    save_report(report)

    try:
        await status.delete()
    except Exception:
        pass

    await _render_report(cb.message, report, digest=digest)


@router.callback_query(F.data == "rep:back")
async def cb_back_to_report(cb: CallbackQuery) -> None:
    """Возврат к overview из раскрытой темы/пагинатора.

    Удаляем основное сообщение, quote-сообщение (dual-media) и bubble-комменты —
    чтобы в чате не оставалось «висящего» хвоста (см. фото 4 из жалобы юзера).
    """
    report = get_report(cb.from_user.id)
    if not report:
        await cb.answer("Отчёт устарел, собираю заново…")
        await cb_new_report(cb)
        return
    await cb.answer()
    from bot.handlers.feed import cleanup_bubbles as _cb_cleanup
    from bot.handlers.feed import cleanup_topic_header as _cb_header
    await _cb_cleanup(cb.bot, cb.from_user.id)
    await _cb_header(cb.bot, cb.from_user.id)  # чистим header темы
    await cleanup_post_by_message(cb.bot, cb.from_user.id, cb.message.message_id)
    # Если есть summary хотя бы у одного кластера — это был digest, рисуем так же.
    is_digest = any(c.summary for c in report.clusters)
    await _render_report(cb.message, report, digest=is_digest)


@router.callback_query(F.data.startswith("rep:super:"))
async def cb_show_super(cb: CallbackQuery) -> None:
    """Клик по супер-категории — показать её под-темы кнопками."""
    try:
        sidx = int(cb.data.split(":")[2])
    except (ValueError, IndexError):
        await cb.answer()
        return
    report = get_report(cb.from_user.id)
    if not report or sidx >= len(report.super_topics):
        await cb.answer("Категория устарела, пересобери отчёт")
        return
    await cb.answer()
    st = report.super_topics[sidx]
    by_id = {c.id: c for c in report.clusters}
    sub_meta = [
        (cid, by_id[cid].emoji, by_id[cid].name, len(by_id[cid].tweet_ids))
        for cid in st.sub_ids if cid in by_id
    ]
    total = sum(m[3] for m in sub_meta)
    text = (
        f"<b>{st.emoji} {html.escape(st.name)}</b> · {total} постов\n"
        f"<i>Выбери под-тему:</i>"
    )
    try:
        await cb.message.delete()
    except Exception:
        pass
    await cb.message.answer(text, parse_mode=ParseMode.HTML,
                            reply_markup=super_topic_details_kb(sub_meta))


@router.callback_query(F.data == "rep:supers")
async def cb_back_to_supers(cb: CallbackQuery) -> None:
    """Возврат из под-темы к списку супер-категорий."""
    report = get_report(cb.from_user.id)
    if not report:
        await cb.answer("Отчёт устарел")
        await cb_new_report(cb)
        return
    await cb.answer()
    # Чистим всё что было в теме: комменты, quote-раскрытия, header, пост.
    from bot.handlers.feed import cleanup_bubbles as _cb_cleanup
    from bot.handlers.feed import cleanup_topic_header as _cb_header
    await _cb_cleanup(cb.bot, cb.from_user.id)
    await _cb_header(cb.bot, cb.from_user.id)
    await cleanup_post_by_message(cb.bot, cb.from_user.id, cb.message.message_id)
    try:
        await cb.message.delete()
    except Exception:
        pass
    await _render_report(cb.message, report)


@router.callback_query(F.data == "rep:flat")
async def cb_flat_list(cb: CallbackQuery) -> None:
    """Показать плоский список всех тем (без группировки)."""
    report = get_report(cb.from_user.id)
    if not report:
        await cb.answer("Отчёт устарел")
        await cb_new_report(cb)
        return
    await cb.answer()
    try:
        await cb.message.delete()
    except Exception:
        pass
    await _render_report(cb.message, report, force_flat=True)


@router.callback_query(F.data.startswith("rep:topic:"))
async def cb_show_topic(cb: CallbackQuery) -> None:
    """Открыть тему — показать дайджест + первый пост с пагинатором."""
    await _render_topic_open(cb)


@router.callback_query(F.data.startswith("tpc:"))
async def cb_paginate_topic(cb: CallbackQuery) -> None:
    """Листание постов внутри темы: tpc:<cluster_id>:<pos>."""
    try:
        _, cid_s, pos_s = cb.data.split(":", 2)
        cid = int(cid_s)
        pos = int(pos_s)
    except Exception:
        await cb.answer()
        return
    await _render_topic_post(cb, cid, pos)


@router.callback_query(F.data == "noop")
async def cb_noop(cb: CallbackQuery) -> None:
    await cb.answer()


async def _render_topic_open(cb: CallbackQuery) -> None:
    """Первое открытие темы — чистим список под-тем, шлём заголовок+дайджест, потом первый пост."""
    user_id = cb.from_user.id
    try:
        cid = int(cb.data.split(":")[2])
    except (ValueError, IndexError):
        await cb.answer()
        return

    report = get_report(user_id)
    if not report:
        await cb.answer("Отчёт устарел, собираю заново…")
        await cb_new_report(cb)
        return
    cluster = next((c for c in report.clusters if c.id == cid), None)
    if not cluster:
        await cb.answer("Тема не найдена")
        return

    await cb.answer()

    # Уборка перед входом в тему: удаляем список под-тем (cb.message —
    # это то сообщение на кнопку в котором юзер кликнул). Иначе он останется
    # висеть сверху и при листании поста экран захламляется «навигационными»
    # bubble'ами из предыдущих уровней (см. жалобу юзера про «всё запутано»).
    try:
        await cb.message.delete()
    except TelegramAPIError:
        pass

    # Дайджест (лениво — если upfront-проход в build_report не справился).
    # Не шлём статус-сообщение (бывший «📝 Формирую дайджест темы…»): оно
    # быстро удаляется и только мелькает, а если Claude не ответит — юзер
    # увидит мелькнувший bubble впустую.
    if not cluster.summary:
        from core.report import _accept_summary, _sample_diverse_ids
        sample_ids = _sample_diverse_ids(cluster.tweet_ids, n=10)
        async with session_scope() as s:
            rows = (await s.execute(
                select(Tweet).where(Tweet.tweet_id.in_(sample_ids))
            )).all()
            sample_texts = [r[0].text for r in rows]
        try:
            raw = await ai_client.summarize_discussion(sample_texts, cluster.name)
            cluster.summary = _accept_summary(raw)
        except Exception as e:
            log.debug("lazy summary failed for %s: %s", cluster.name, e)

    # Шапка темы. Если дайджест не получился — просто заголовок + число постов,
    # без раздражающего «(дайджест не получился — листай посты ниже)»: юзер и так
    # видит пост ниже, disclaimer только шум.
    if cluster.summary:
        header = (
            f"<b>{cluster.emoji} {cluster.name}</b> · {len(cluster.tweet_ids)} постов\n\n"
            f"{cluster.summary}"
        )
    else:
        header = f"<b>{cluster.emoji} {cluster.name}</b> · {len(cluster.tweet_ids)} постов"
    header_msg = await cb.bot.send_message(
        chat_id=user_id,
        text=header,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )
    # Трекаем header отдельно от extra-bubble'ов (комментов): он должен
    # пережить листание ⬅/➡ и быть удалён ТОЛЬКО при выходе из темы
    # (back / supers / menu). cleanup_bubbles() его не трогает, только
    # cleanup_topic_header() — вызывается в rep:back / rep:supers / menu:main.
    from bot.handlers.feed import _track_topic_header
    _track_topic_header(user_id, header_msg.message_id)

    # Первый пост темы с пагинатором.
    await _render_topic_post(cb, cid, 0, send_new=True)


async def _render_topic_post(
    cb: CallbackQuery,
    cluster_id: int,
    pos: int,
    *,
    send_new: bool = False,
) -> None:
    """Показать один пост темы с навигацией «пред/след».

    Вся отправка идёт через общий `send_one_tweet` — в т.ч. dual-media
    (пост автора + отдельно цитата с медиа). Оба message_id сохраняются в
    SentNews, чтобы при листании/back удалилось и то и другое.
    """
    user_id = cb.from_user.id
    report = get_report(user_id)
    if not report:
        await cb.answer("Отчёт устарел")
        return
    cluster = next((c for c in report.clusters if c.id == cluster_id), None)
    if not cluster or not cluster.tweet_ids:
        await cb.answer("Тема пуста")
        return

    total = len(cluster.tweet_ids)
    pos = max(0, min(total - 1, pos))
    tweet_id = cluster.tweet_ids[pos]

    # При листании: удаляем предыдущее основное + quote (dual-media) + bubble-комменты.
    if not send_new:
        from bot.handlers.feed import cleanup_bubbles as _cb_cleanup
        await _cb_cleanup(cb.bot, user_id)
        await cleanup_post_by_message(cb.bot, user_id, cb.message.message_id)

    async with session_scope() as s:
        tweet = await s.get(Tweet, tweet_id)
        if not tweet:
            await cb.answer("Пост пропал из кэша")
            return
        liked = await _get_feedback(s, user_id, tweet_id)
        quote_author = tweet.quote_author if (tweet.quote_author and tweet.quote_text) else None
        kb = topic_paginator_kb(
            cluster_id, pos, total,
            tweet_id=tweet_id, liked=liked, translated=False,
            quote_author=quote_author,
        )
        ok = await send_one_tweet(
            cb.bot, s, user_id, tweet,
            liked=liked, reply_markup=kb,
        )
        if not ok:
            await cb.answer("Пост не получилось отправить")
            return

    try:
        await cb.answer()
    except Exception:
        pass


async def _get_feedback(session, user_id: int, tweet_id: str) -> bool | None:
    fb = (
        await session.execute(
            select(Feedback).where(
                Feedback.user_id == user_id,
                Feedback.tweet_id == tweet_id,
            )
        )
    ).scalar_one_or_none()
    return fb.liked if fb else None


@router.callback_query(F.data == "rep:feed")
async def cb_plain_feed(cb: CallbackQuery) -> None:
    """«Моя лента» — Following за 24ч, сгруппированная по темам.

    Источник — Following (хронология подписок), отличается от «📊 что
    обсуждают» (For You). До этого фикса обе кнопки слали одно и то же
    через for_you — юзер прямо жаловался «делают то же самое».

    Чтобы Following не топился доминантными аккаунтами (Reuters × 47/80
    в живом замере), build_report делает per-author cap=3 для source='following'
    (см. core/report.py). Это даёт реальную «лента подписок без спама от
    одного издания».

    Окно — 24ч: подписочная лента читается «за вечер/ночь/утро», в отличие
    от For You где X сам поднимает горячее за сутки.
    """
    await _run_report_flow(cb, source="following", window_hours=24.0, limit_raw=400,
                           auto_summarize_top=7)


# ============================= main menu =============================


@router.callback_query(F.data == "menu:main")
async def cb_main_menu(cb: CallbackQuery) -> None:
    """Выход в главное меню из любого места (тема, под-тема, пост).

    Чистим все контекстные bubble'ы — комменты, раскрытые цитаты, header темы,
    сам пост — и шлём меню отдельным сообщением. Раньше пробовали edit на
    cb.message, но если юзер кликнул с пост-пагинатора, cleanup_post_by_message
    уже удалил это сообщение и edit падал `message to edit not found`.
    """
    user_id = cb.from_user.id
    from bot.handlers.feed import cleanup_bubbles as _cb_cleanup
    from bot.handlers.feed import cleanup_topic_header as _cb_header
    await _cb_cleanup(cb.bot, user_id)
    await _cb_header(cb.bot, user_id)
    try:
        await cleanup_post_by_message(cb.bot, user_id, cb.message.message_id)
    except Exception:
        pass

    async with session_scope() as s:
        user = await s.get(User, user_id)
        paused = bool(user and user.paused)

    try:
        await cb.bot.send_message(
            chat_id=user_id, text="Меню:", reply_markup=main_menu_kb(paused=paused),
        )
    except TelegramAPIError as e:
        log.warning("send menu failed: %s", e)
    await cb.answer()


@router.callback_query(F.data == "menu:interval")
async def cb_menu_interval(cb: CallbackQuery) -> None:
    await cb.message.answer("Выбери интервал доставки:", reply_markup=interval_kb())
    await cb.answer()


@router.callback_query(F.data == "rep:trending")
async def cb_trending(cb: CallbackQuery) -> None:
    """«Что обсуждают» — отчёт с широким окном (20 часов) для discovery-режима.

    Та же pipeline: timeline → trash+hype фильтр → кластеризация → именование.
    Больше времени → больше разных тем, видно что в целом обсуждается в сети
    подписок.
    """
    user_id = cb.from_user.id
    await cb.answer()
    try:
        await cb.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    status = await cb.message.answer("⏳ Смотрю что обсуждают в последние 20 часов…")

    async def _progress(msg: str) -> None:
        try:
            await status.edit_text(msg)
        except TelegramAPIError:
            pass

    async with session_scope() as s:
        user = await s.get(User, user_id)
        if not user:
            await status.edit_text("Сначала /start.")
            return
        try:
            from core.report import build_report as _build
            report = await _build(s, user, window_hours=20.0, progress=_progress, limit_raw=200)
        except Exception as e:
            log.exception("trending report failed for %s", user_id)
            await status.edit_text(f"❌ Сбой: <code>{type(e).__name__}</code>",
                                   parse_mode=ParseMode.HTML)
            return

    save_report(report)
    try:
        await status.delete()
    except Exception:
        pass
    await _render_report(cb.message, report)
