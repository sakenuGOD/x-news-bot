"""Форматирование и отправка подборки новостей пользователю.

Дефолт — оригинал (английский), кнопка 🇷🇺 переводит через Haiku и кэширует
перевод в Tweet.summary_ru. Повторное нажатие возвращает оригинал без AI-вызова.
"""

from __future__ import annotations

import asyncio
import html
import logging
from datetime import datetime, timezone
from typing import Optional

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BufferedInputFile, InlineKeyboardMarkup
from sqlalchemy import select

from bot.keyboards import feedback_kb
from config import settings
from core.recommender import pick_top_for_user
from db.database import session_scope
from db.models import SentNews, Tweet, User

log = logging.getLogger(__name__)


# Маппинг кластер → эмодзи для "шапки" поста.
_TOPIC_EMOJI = {
    "tech": "🖥",
    "ai": "🤖",
    "crypto": "🪙",
    "science": "🔬",
    "politics": "🏛",
    "business": "💼",
    "memes": "😂",
    "culture": "🎨",
    "sports": "⚽",
    "gaming": "🎮",
    "lifestyle": "🧘",
    "news": "📰",
}

# Telegram лимиты.
PHOTO_CAPTION_LIMIT = 1024
TEXT_MESSAGE_LIMIT = 4096


def _hours_ago(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    diff = datetime.now(timezone.utc) - dt
    hours = int(diff.total_seconds() // 3600)
    if hours < 1:
        mins = max(1, int(diff.total_seconds() // 60))
        return f"{mins} мин назад"
    if hours < 24:
        return f"{hours} ч назад"
    return f"{hours // 24} дн назад"


def _format_engagement(tweet: Tweet) -> str:
    """'❤️ 48K · 🔁 1.2K' — или пусто если engagement никакой."""
    parts: list[str] = []
    if tweet.likes_count and tweet.likes_count >= 10:
        parts.append(f"❤️ {_short_num(tweet.likes_count)}")
    if tweet.retweets_count and tweet.retweets_count >= 5:
        parts.append(f"🔁 {_short_num(tweet.retweets_count)}")
    if tweet.replies_count and tweet.replies_count >= 5:
        parts.append(f"💬 {_short_num(tweet.replies_count)}")
    return " · ".join(parts)


def _short_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{n / 1_000:.1f}K".replace(".0K", "K")
    return str(n)


def _caption_body(text: str, limit: int) -> str:
    """Обрезает и HTML-escape'ит тело твита, чтобы помещалось в limit с шапкой/футером."""
    safe = html.escape(text or "")
    if len(safe) > limit:
        safe = safe[: limit - 1].rstrip() + "…"
    return safe


def format_caption(tweet: Tweet, russian: bool = False) -> str:
    """Красиво оформленная подпись к твиту.

    Layout (фиксированный порядок в одном Telegram-сообщении):
        [эмодзи темы] <b>Display</b> · <i>@handle</i>

        [текст автора — без quote-хвоста]

        <blockquote>↪ @quoted: текст цитаты</blockquote>  — если есть quote

        <i>🇬🇧 1 ч · ❤️ 678 · 🔁 45 · 💬 20</i>

    Почему blockquote: юзер жаловался «не понятно где реплай, где авторский
    текст». Telegram рендерит <blockquote> с серой полосой слева и отступом —
    моментально отделяет цитируемый пост от основного. Раньше quote вставлялся
    в тело как обычный текст с префиксом «↪ цитирует» и сливался визуально.

    russian=True → используем tweet.summary_ru (перевод). Если его нет — оригинал.
    Quote блок рендерим из отдельных полей БД (tweet.quote_author/quote_text),
    чтобы форматирование не зависело от парсинга текста.
    """
    emoji = _TOPIC_EMOJI.get(tweet.topic or "news", "📰")
    display = tweet.author_display_name or tweet.author_username
    display_safe = html.escape(display)
    author_safe = html.escape(tweet.author_username)
    when = _hours_ago(tweet.created_at)
    engagement = _format_engagement(tweet)
    lang_marker = "🇷🇺" if russian else "🇬🇧"

    header = f"{emoji} <b>{display_safe}</b> · <i>@{author_safe}</i>"

    body_raw = tweet.summary_ru if (russian and tweet.summary_ru) else tweet.text

    # Quote уходит ОТДЕЛЬНЫМ сообщением (_send_quote_message) — вырезаем её хвост
    # из тела caption, чтобы текст цитаты не дублировался. В БД у нас quote-часть
    # приклеена к tweet.text как «↪ цитирует @user: …» (или «▌ @» для старых).
    if tweet.quote_author and tweet.quote_text:
        for marker in ("↪ цитирует ", "Цитирует @", "▌ @"):
            idx = body_raw.find(marker)
            if idx > 0:
                body_raw = body_raw[:idx].rstrip()
                break

    # Резерв на шапку + футер.
    body_limit = max(200, PHOTO_CAPTION_LIMIT - 220)
    body = _caption_body(body_raw, body_limit)

    footer_parts = [f"{lang_marker} {when}"]
    if engagement:
        footer_parts.append(engagement)
    footer = " · ".join(footer_parts)

    return f"{header}\n\n{body}\n\n<i>{footer}</i>"


async def deliver_news_to_user(bot: Bot, user_id: int) -> int:
    """Подбирает топ-N и отправляет. Возвращает число отправленных сообщений.

    Не шлёт трейлер «подсказка»/«батч-контролы» — кнопки главного меню
    остаются в отдельном сообщении после /start, этого достаточно.
    """
    async with session_scope() as session:
        user = await session.get(User, user_id)
        if not user or not user.onboarding_done or user.paused:
            return 0

        scored = await pick_top_for_user(session, user)
        if not scored:
            try:
                await bot.send_message(
                    user_id,
                    "Пока не нашёл свежих постов под твой профиль. Попробую снова через интервал.",
                )
            except TelegramAPIError as e:
                log.warning("send_message empty_batch failed for %s: %s", user_id, e)
            return 0

        sent_count = 0
        for s in scored:
            ok = await send_one_tweet(bot, session, user_id, s.tweet)
            if ok:
                sent_count += 1

        if sent_count:
            user.last_delivered_at = datetime.now(timezone.utc).replace(tzinfo=None)
            await session.flush()

        return sent_count


async def _download_bytes(url: str, *, max_bytes: int = 45 * 1024 * 1024,
                          timeout: float = 25.0) -> Optional[bytes]:
    """Качаем медиа-файл с ограничением. Нужно для видео/gif: Telegram при
    `send_video(url=...)` сам идёт на video.twimg.com, но там бывает 403 или
    очень медленный отклик → Telegram возвращает failed. Качаем сами и шлём
    как BufferedInputFile — это на порядок надёжнее. 45MB — потолок Telegram
    для bot API на видео.
    """
    try:
        import httpx  # aiogram тянет httpx-совместимый клиент
    except ImportError:
        return None

    headers = {
        # Без referer video.twimg.com иногда 403'ит.
        "Referer": "https://twitter.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        ),
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
            # HEAD чтобы узнать размер и не качать слишком большое.
            try:
                h = await c.head(url, headers=headers)
                if h.status_code == 200:
                    cl = h.headers.get("content-length")
                    if cl and int(cl) > max_bytes:
                        log.info("media too large for tg (%s bytes): %s", cl, url[:80])
                        return None
            except Exception:
                pass
            r = await c.get(url, headers=headers)
            if r.status_code != 200:
                log.debug("download %s: status %s", url[:80], r.status_code)
                return None
            data = r.content
            if len(data) > max_bytes:
                log.info("media oversized after download (%d bytes): %s",
                         len(data), url[:80])
                return None
            return data
    except Exception as e:
        log.debug("_download_bytes(%s) failed: %s", url[:80], e)
        return None


def _detect_media_type(tweet: Tweet) -> Optional[str]:
    if tweet.media_type:
        return tweet.media_type
    if not tweet.image_url:
        return None
    url_l = tweet.image_url.lower()
    # Видео.
    if "video.twimg.com" in url_l or ".mp4" in url_l:
        return "video"
    # GIF-анимация — Twitter отдаёт mp4, но через ext_tw_video / amp/tweet/videos.
    if "tweet_video" in url_l or "ext_tw_video" in url_l:
        return "animation"
    # Фото: prefix pbs.twimg.com (официальный image CDN) — photo даже без
    # расширения (раньше такие посты фолбечились в текст и фото исчезали).
    if "pbs.twimg.com" in url_l:
        return "photo"
    if url_l.endswith((".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif")):
        return "photo"
    # Card-thumbnail: Twitter иногда отдаёт с query-string (…&name=medium).
    # Если URL содержит /media/ и is_image CDN — считаем photo.
    if "/media/" in url_l and "twimg" in url_l:
        return "photo"
    return None


async def _send_media_with_fallback(
    bot: Bot,
    user_id: int,
    *,
    media_url: str,
    media_type: str,
    caption: str,
    reply_markup: InlineKeyboardMarkup | None,
    tweet_id_hint: str = "",
) -> Optional[object]:
    """Шлём photo/video/animation с каскадом fallback'ов: url-mode → bytes-mode.

    Возвращает Message или None если всё упало. Вызывающий решает что делать
    (обычно — отправить plain text).
    """
    sender_map = {
        "photo": bot.send_photo,
        "video": bot.send_video,
        "animation": bot.send_animation,
    }
    kwargs_key_map = {
        "photo": "photo",
        "video": "video",
        "animation": "animation",
    }
    extra_kwargs: dict = {"show_caption_above_media": True}
    if media_type == "video":
        extra_kwargs["supports_streaming"] = True

    sender = sender_map.get(media_type)
    if not sender:
        return None
    field = kwargs_key_map[media_type]

    # 1) fast path — url-mode.
    try:
        return await sender(
            chat_id=user_id,
            **{field: media_url},
            caption=caption[:PHOTO_CAPTION_LIMIT],
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
            **extra_kwargs,
        )
    except TelegramAPIError as e:
        log.warning("send_%s url-mode failed for %s: %s", media_type, tweet_id_hint, e)

    # 2) bytes-mode: качаем сами, шлём как BufferedInputFile.
    max_bytes = 10 * 1024 * 1024 if media_type == "photo" else 45 * 1024 * 1024
    if media_type == "animation":
        max_bytes = 20 * 1024 * 1024
    data = await _download_bytes(media_url, max_bytes=max_bytes)
    if not data:
        return None
    ext = "jpg" if media_type == "photo" else "mp4"
    filename = f"{tweet_id_hint or 'media'}.{ext}"
    try:
        return await sender(
            chat_id=user_id,
            **{field: BufferedInputFile(data, filename=filename)},
            caption=caption[:PHOTO_CAPTION_LIMIT],
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
            **extra_kwargs,
        )
    except TelegramAPIError as e2:
        log.warning("send_%s bytes-mode failed for %s: %s", media_type, tweet_id_hint, e2)
        return None


async def _send_quote_message(
    bot: Bot,
    user_id: int,
    tweet: Tweet,
    *,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> Optional[object]:
    """Второе сообщение — сама цитата (если у поста есть цитата).

    Кнопки (like/translate/comments/paginator) теперь живут ЗДЕСЬ, а не на
    первом сообщении — юзер попросил «чтобы кнопки были у второго поста».
    Это удобно: второй bubble последний в чате, пагинатор/действия под рукой.

    Режимы:
      1. Есть quote_image_url — медиа цитаты + caption «↪ цитирует @X: …».
      2. Нет quote_image_url, но есть quote_text — text-only сообщение.
      3. Нет quote_author/quote_text — не шлём ничего (обычный пост без цитаты).
    """
    if not (tweet.quote_author and tweet.quote_text):
        return None

    q_author = html.escape(tweet.quote_author)
    q_body = html.escape((tweet.quote_text or "").strip())

    has_media = bool(tweet.quote_image_url)
    text_limit = (PHOTO_CAPTION_LIMIT - 40) if has_media else (TEXT_MESSAGE_LIMIT - 40)
    if len(q_body) > text_limit:
        q_body = q_body[: text_limit - 1].rstrip() + "…"
    caption = f"↪ цитирует <b>@{q_author}</b>\n\n{q_body}"

    if has_media:
        qmt = (tweet.quote_media_type or "photo").lower()
        if qmt not in ("photo", "video", "animation"):
            qmt = "photo"
        try:
            sent = await _send_media_with_fallback(
                bot, user_id,
                media_url=tweet.quote_image_url,
                media_type=qmt,
                caption=caption,
                reply_markup=reply_markup,
                tweet_id_hint=f"q_{tweet.tweet_id}",
            )
            if sent is not None:
                return sent
        except Exception as e:
            log.debug("dual-bubble quote media send failed for %s: %s", tweet.tweet_id, e)

    # Text-only bubble (или медиа упало).
    try:
        return await bot.send_message(
            chat_id=user_id,
            text=caption,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
    except TelegramAPIError as e:
        log.debug("dual-bubble quote text send failed for %s: %s", tweet.tweet_id, e)
        return None


async def _upsert_sent_news(
    session,
    user_id: int,
    tweet_id: str,
    main_message_id: int,
    quote_message_id: Optional[int],
) -> None:
    """Пишем оба message_id в SentNews. Если запись уже есть (переoткрытие поста
    в другом листании) — ОБНОВЛЯЕМ поля, не пытаемся вставить дубль.
    """
    existing = (await session.execute(
        select(SentNews).where(
            SentNews.user_id == user_id,
            SentNews.tweet_id == tweet_id,
        )
    )).scalar_one_or_none()
    if existing:
        existing.telegram_message_id = main_message_id
        existing.quote_telegram_message_id = quote_message_id
        return
    try:
        async with session.begin_nested():
            session.add(SentNews(
                user_id=user_id,
                tweet_id=tweet_id,
                telegram_message_id=main_message_id,
                quote_telegram_message_id=quote_message_id,
            ))
    except Exception as e:
        log.debug("sent_news savepoint: %s", e)


async def cleanup_post_by_message(bot: Bot, user_id: int, message_id: int) -> None:
    """Удаляет основное сообщение и (если было) quote-сообщение dual-bubble.

    Важно: юзер теперь жмёт кнопки на ВТОРОМ (quote) bubble — значит message_id
    из callback может принадлежать ЛЮБОМУ из двух. Ищем запись в обоих полях
    (telegram_message_id ИЛИ quote_telegram_message_id) и гасим обе bubble.
    """
    main_mid: Optional[int] = message_id
    quote_mid: Optional[int] = None
    try:
        async with session_scope() as s:
            row = (await s.execute(
                select(SentNews).where(
                    SentNews.user_id == user_id,
                    (SentNews.telegram_message_id == message_id)
                    | (SentNews.quote_telegram_message_id == message_id),
                )
            )).scalar_one_or_none()
            if row:
                main_mid = row.telegram_message_id
                quote_mid = row.quote_telegram_message_id
    except Exception as e:
        log.debug("cleanup_post lookup failed: %s", e)

    for mid in (main_mid, quote_mid):
        if not mid:
            continue
        try:
            await bot.delete_message(user_id, mid)
        except TelegramAPIError:
            pass


async def send_one_tweet(
    bot: Bot,
    session,
    user_id: int,
    tweet: Tweet,
    *,
    liked: bool | None = None,
    reply_markup: InlineKeyboardMarkup | None = None,
    russian: bool = False,
    record_sent: bool = True,
) -> bool:
    """Унифицированная отправка одного твита. Используется и регулярной
    доставкой, и хендлером отчёта (темы, лента), и фидбэк-кнопками.

    Порядок:
      1. photo/video/animation → `_send_media_with_fallback` (url → bytes).
      2. Если медиа не ушло — plain text с теми же кнопками.
      3. Если квота с отдельным медиа — второе сообщение `_send_quote_message`.
      4. SentNews upsert с обоими message_id — для последующего cleanup.

    reply_markup: если None — используется дефолт feedback_kb. Хендлеры
    темы/ленты передают свой пагинатор.
    """
    caption = format_caption(tweet, russian=russian)
    # Если есть цитата — НЕ шлём её сразу вторым сообщением. Вместо этого
    # добавляем в клавиатуру кнопку «↪ Цитата: @X» — клик раскроет её
    # отдельным bubble по запросу. Юзер просил: «не сразу бросаем второй
    # пост, а делаем инлайн-кнопку типо раскрыть».
    quote_author = tweet.quote_author if (tweet.quote_author and tweet.quote_text) else None
    if reply_markup is None:
        kb = feedback_kb(
            tweet.tweet_id, liked=liked, translated=russian, quote_author=quote_author,
        )
    else:
        kb = reply_markup  # хендлеры темы передают свой topic_paginator_kb

    media_type = _detect_media_type(tweet)
    if tweet.image_url and not media_type:
        # Тип неизвестен но URL есть — пробуем photo как безопасный дефолт.
        media_type = "photo"

    msg = None
    if tweet.image_url and media_type in ("photo", "video", "animation"):
        msg = await _send_media_with_fallback(
            bot, user_id,
            media_url=tweet.image_url,
            media_type=media_type,
            caption=caption,
            reply_markup=kb,
            tweet_id_hint=tweet.tweet_id,
        )

    if msg is None:
        # Финальный fallback — текст вместо медиа.
        try:
            msg = await bot.send_message(
                chat_id=user_id,
                text=caption[:TEXT_MESSAGE_LIMIT],
                reply_markup=kb,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except TelegramAPIError as e:
            log.error("send fallback failed for %s: %s", tweet.tweet_id, e)
            return False

    if record_sent:
        # Второе сообщение-цитата больше не отправляется автоматом —
        # quote_message_id всегда None в новой схеме. Поле оставляем
        # для обратной совместимости cleanup_post_by_message.
        await _upsert_sent_news(
            session, user_id, tweet.tweet_id,
            main_message_id=msg.message_id,
            quote_message_id=None,
        )

    return True
