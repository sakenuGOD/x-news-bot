"""Обработка 👍/👎/🌐 под твитом и кнопок «Пауза / Продолжить»."""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError
from aiogram.types import CallbackQuery

from bot.delivery import format_caption
from bot.keyboards import feedback_kb, resume_kb, topic_paginator_kb
from config import settings
from core import ai_client
from core import embeddings as emb
from db.database import session_scope
from db.models import Feedback, Tweet, User
from sqlalchemy import select, update

log = logging.getLogger(__name__)

router = Router(name="feed")


async def _current_feedback(session, user_id: int, tweet_id: str) -> bool | None:
    fb = (
        await session.execute(
            select(Feedback).where(
                Feedback.user_id == user_id,
                Feedback.tweet_id == tweet_id,
            )
        )
    ).scalar_one_or_none()
    return fb.liked if fb else None


def _is_translated_now(cb: CallbackQuery) -> bool:
    """Смотрим текущую клавиатуру, чтобы понять — показан ли уже русский перевод."""
    kb = cb.message.reply_markup if cb.message else None
    if not kb or not kb.inline_keyboard:
        return False
    for row in kb.inline_keyboard:
        for btn in row:
            if btn.callback_data and btn.callback_data.startswith("tr:en:"):
                return True
    return False


def _detect_paginator_context(cb: CallbackQuery) -> tuple[int, int, int] | None:
    """Если текущая клавиатура — пагинатор темы, вытащим (cluster_id, pos, total).

    Нужно чтобы при нажатии 👍/👎/🌐 внутри пагинатора сохранить навигацию
    «⬅ Пред. · N/total · След. ➡», а не схлопнуть её в базовую feedback_kb.
    """
    kb = cb.message.reply_markup if cb.message else None
    if not kb:
        return None
    cid: int | None = None
    positions: list[int] = []
    total: int | None = None
    for row in kb.inline_keyboard:
        for btn in row:
            cd = btn.callback_data or ""
            if cd.startswith("tpc:"):
                parts = cd.split(":")
                if len(parts) >= 3:
                    try:
                        cid = int(parts[1])
                        positions.append(int(parts[2]))
                    except ValueError:
                        pass
            elif "/" in (btn.text or "") and btn.callback_data == "noop":
                # Кнопка вида "N / total"
                try:
                    _, tot = btn.text.split("/")
                    total = int(tot.strip())
                    # pos отобразим из текста — но лучше из neighbor callbacks
                except Exception:
                    pass
    if cid is None or total is None:
        return None
    # pos — это (min of tpc: positions) + 1 … no actually neighbor positions are pos-1 and pos+1.
    # Берём центрированное значение: (min+max)/2 даст текущий pos.
    if not positions:
        return None
    cur_pos = (min(positions) + max(positions)) // 2 if len(positions) == 2 else positions[0]
    return cid, cur_pos, total


# ----------------------------- 👍 / 👎 -----------------------------


@router.callback_query(F.data.startswith("fb:"))
async def handle_feedback(cb: CallbackQuery) -> None:
    try:
        _, action, tweet_id = cb.data.split(":", 2)
    except ValueError:
        await cb.answer()
        return
    liked = action == "like"
    translated_now = _is_translated_now(cb)

    async with session_scope() as s:
        user = await s.get(User, cb.from_user.id)
        tweet = await s.get(Tweet, tweet_id)
        if not user or not tweet:
            await cb.answer("Новость устарела", show_alert=False)
            return

        # Upsert Feedback — разрешаем поменять решение.
        existing = (
            await s.execute(
                select(Feedback).where(
                    Feedback.user_id == user.telegram_id,
                    Feedback.tweet_id == tweet_id,
                )
            )
        ).scalar_one_or_none()

        prev_liked = existing.liked if existing else None
        if existing:
            existing.liked = liked
        else:
            s.add(Feedback(user_id=user.telegram_id, tweet_id=tweet_id, liked=liked))

        # Обновляем preference_vector.
        if tweet.embedding:
            vec = list(user.preference_vector) if user.preference_vector else None
            if vec is None and liked:
                # У нового юзера ещё не было preference — сид от этого твита.
                vec = emb.normalize(tweet.embedding)

            if vec is not None:
                # Если меняем решение — откатываем старый эффект (приблизительно).
                if prev_liked is True and not liked:
                    vec = emb.update_on_dislike(vec, tweet.embedding, settings.preference_learn_rate_like)
                elif prev_liked is False and liked:
                    vec = emb.update_on_like(vec, tweet.embedding, settings.preference_learn_rate_dislike)

                if liked:
                    vec = emb.update_on_like(vec, tweet.embedding, settings.preference_learn_rate_like)
                else:
                    vec = emb.update_on_dislike(vec, tweet.embedding, settings.preference_learn_rate_dislike)

                user.preference_vector = vec

        # Бонус автору — копим полезный сигнал.
        if liked:
            await s.execute(
                update(Tweet)
                .where(Tweet.tweet_id == tweet_id)
                .values(likes_count=Tweet.likes_count + 1)
            )
            # Auto-add автора в FollowedAuthor: если юзер лайкнул пост, этот
            # автор ему зашёл — нужно мониторить его и в будущих отчётах. Если
            # автор уже есть — увеличиваем weight (чтобы pick_top_for_user
            # ранжировал его посты выше).
            from db.models import FollowedAuthor as _FA
            existing_fa = (await s.execute(
                select(_FA).where(
                    _FA.user_id == user.telegram_id,
                    _FA.author_username == tweet.author_username,
                )
            )).scalar_one_or_none()
            if existing_fa:
                existing_fa.weight = min(3.0, (existing_fa.weight or 1.0) + 0.25)
            else:
                try:
                    async with s.begin_nested():
                        s.add(_FA(
                            user_id=user.telegram_id,
                            author_username=tweet.author_username,
                            weight=1.25,
                        ))
                    log.info("auto-add on like: user=%s +@%s",
                             user.telegram_id, tweet.author_username)
                except Exception as e:
                    log.debug("auto-add on like failed: %s", e)

        # Dislike-dampening: если юзер жмёт 👎, запомним тему которая ему не зашла.
        # Дважды:
        #  (a) specific topic name из текущего отчёта — подавит в ближайших отчётах
        #     эту же Claude-сформулированную тему;
        #  (b) стабильный кластер (tech/ai/crypto/...) через tweet.topic — понизит
        #     cluster_weight, эффект переживёт перегенерацию отчёта и работает
        #     на уровне ранжирования pick_top_for_user.
        if not liked:
            from core.report import get_report
            rep = get_report(user.telegram_id)
            topic_name = None
            if rep:
                for cl in rep.clusters:
                    if tweet_id in cl.tweet_ids:
                        topic_name = cl.name.lower().strip()
                        break
            from datetime import datetime, timezone
            if topic_name:
                d = dict(user.topic_dislikes or {})
                entry = dict(d.get(topic_name, {}))
                entry["dislikes"] = int(entry.get("dislikes", 0)) + 1
                entry["last_ts"] = datetime.now(timezone.utc).isoformat()
                d[topic_name] = entry
                user.topic_dislikes = d
            # (b) Понижаем cluster_weight (тематический кластер tweet.topic).
            stable_cluster = (tweet.topic or "").strip()
            if stable_cluster:
                cw = dict(user.cluster_weights or {})
                cur = float(cw.get(stable_cluster, 0.3))
                # Шаг -0.08 за дизлайк, не ниже 0.
                cw[stable_cluster] = max(0.0, cur - 0.08)
                user.cluster_weights = cw

    # Обновляем кнопки — подсвечиваем выбор, сохраняем текущий язык и
    # quote-кнопку (если у поста есть цитата, она должна остаться после
    # like/dislike, иначе юзер её потеряет после первого же клика 👍/👎).
    quote_author = tweet.quote_author if (tweet.quote_author and tweet.quote_text) else None
    paginator_ctx = _detect_paginator_context(cb)
    if paginator_ctx:
        cid, pos, total = paginator_ctx
        new_kb = topic_paginator_kb(cid, pos, total, tweet_id=tweet_id,
                                    liked=liked, translated=translated_now,
                                    quote_author=quote_author)
    else:
        new_kb = feedback_kb(tweet_id, liked=liked, translated=translated_now,
                             quote_author=quote_author)
    try:
        await cb.message.edit_reply_markup(reply_markup=new_kb)
    except TelegramAPIError as e:
        log.debug("edit_reply_markup: %s", e)

    await cb.answer("Учёл! 👍" if liked else "Понял, таких меньше 👌")


# ----------------------------- 🇷🇺 / 🇬🇧 -----------------------------


@router.callback_query(F.data.startswith("tr:"))
async def handle_translate(cb: CallbackQuery) -> None:
    try:
        _, direction, tweet_id = cb.data.split(":", 2)
    except ValueError:
        await cb.answer()
        return

    want_russian = direction == "ru"
    user_id = cb.from_user.id

    async with session_scope() as s:
        tweet = await s.get(Tweet, tweet_id)
        if not tweet:
            await cb.answer("Пост устарел — не могу его найти", show_alert=False)
            return

        # Кэшируем перевод, чтобы повторный клик не платил токены.
        if want_russian and not tweet.summary_ru:
            await cb.answer("Перевожу…")
            tweet.summary_ru = await ai_client.translate_to_ru(tweet.text)
            await s.flush()
        else:
            await cb.answer()

        liked = await _current_feedback(s, user_id, tweet_id)
        new_caption = format_caption(tweet, russian=want_russian)

    # При переводе тоже сохраняем quote-кнопку.
    quote_author = tweet.quote_author if (tweet.quote_author and tweet.quote_text) else None
    paginator_ctx = _detect_paginator_context(cb)
    if paginator_ctx:
        cid, pos, total = paginator_ctx
        new_kb = topic_paginator_kb(cid, pos, total, tweet_id=tweet_id,
                                    liked=liked, translated=want_russian,
                                    quote_author=quote_author)
    else:
        new_kb = feedback_kb(tweet_id, liked=liked, translated=want_russian,
                             quote_author=quote_author)

    # Медиа-сообщения (photo/video/animation/document) редактируются через
    # edit_caption; текстовые — через edit_text. Раньше проверяли только .photo
    # и edit_text на видео/gif падал с "there is no text in the message to edit",
    # из-за чего пропадала кнопка перевода после клика на пост с видео.
    has_media = bool(
        cb.message.photo
        or cb.message.video
        or cb.message.animation
        or cb.message.document
    )
    try:
        if has_media:
            # show_caption_above_media=True сохраняем, чтобы при переводе
            # подпись оставалась над медиа (без этого на edit_caption видео
            # «улетает» наверх — Telegram пересчитывает layout при редакте).
            await cb.message.edit_caption(
                caption=new_caption[:1024],
                reply_markup=new_kb,
                parse_mode=ParseMode.HTML,
                show_caption_above_media=True,
            )
        else:
            await cb.message.edit_text(
                text=new_caption[:4096],
                reply_markup=new_kb,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
    except TelegramAPIError as e:
        # "message is not modified" — ок, молча. Но если это другой сбой — хотя бы
        # обновим клавиатуру, чтобы кнопки не пропали визуально.
        log.debug("translate edit skipped: %s", e)
        try:
            await cb.message.edit_reply_markup(reply_markup=new_kb)
        except TelegramAPIError:
            pass


# ----------------------------- 💬 Comments -----------------------------


# Дополнительные сообщения, которые бот отправил как reply на пост-пагинатор
# (комменты, quote-раскрытия и т.п.). Ключ — user_id, значение — список message_id.
# При навигации на следующий/предыдущий пост эти сообщения УДАЛЯЮТСЯ вместе с
# основным, чтобы не оставалось висящих «осиротевших» bubble'ов (юзер показывал
# фото где после клика на «След» старый «Нет реплаев» остался в чате).
_EXTRA_BUBBLES: dict[int, list[int]] = {}

# Header-bubble темы (эмодзи + имя + дайджест). Трекается ОТДЕЛЬНО от
# `_EXTRA_BUBBLES` потому что нам НЕ нужно его удалять при каждом листании
# ⬅/➡ внутри темы — он даёт контекст «где я нахожусь». Удаляется только при
# выходе из темы (back / supers / menu) — тогда вызывается cleanup_topic_header.
_TOPIC_HEADER_BUBBLE: dict[int, int] = {}


def _track_bubble(user_id: int, message_id: int) -> None:
    _EXTRA_BUBBLES.setdefault(user_id, []).append(message_id)


def _track_topic_header(user_id: int, message_id: int) -> None:
    _TOPIC_HEADER_BUBBLE[user_id] = message_id


async def cleanup_bubbles(bot, user_id: int) -> None:
    """Удаляет extra-bubble'ы (комменты, quote-раскрытия) от предыдущего поста.

    Вызывается из пагинаторов ДО показа нового поста. Header темы НЕ трогает.
    """
    mids = _EXTRA_BUBBLES.pop(user_id, [])
    for mid in mids:
        try:
            await bot.delete_message(user_id, mid)
        except TelegramAPIError:
            pass
    # Сбрасываем «показан ли quote» флаг для этого юзера — при возврате к
    # посту кнопка «↪ Цитата» снова будет работать, а не считать что уже
    # отправляли.
    for k in [k for k in _QUOTE_SHOWN if k[0] == user_id]:
        _QUOTE_SHOWN.pop(k, None)


async def cleanup_topic_header(bot, user_id: int) -> None:
    """Удаляет header темы. Вызывается при выходе из темы (back / supers / menu)."""
    mid = _TOPIC_HEADER_BUBBLE.pop(user_id, None)
    if mid:
        try:
            await bot.delete_message(user_id, mid)
        except TelegramAPIError:
            pass


# ----------------------------- ↪ Раскрыть цитату -----------------------------


@router.callback_query(F.data.startswith("qt:"))
async def handle_show_quote(cb: CallbackQuery) -> None:
    """Клик по «↪ Цитата: @user» — шлём второе сообщение с текстом/медиа цитаты.

    Это замена автоматического dual-bubble. Раньше бот без спроса отправлял
    цитату вторым сообщением под каждым постом, и юзер жаловался что не
    понимает «где пост автора, а где реплай, и когда он закончится». Теперь
    цитата раскрывается только по явному клику, а её message_id трекается
    в `_EXTRA_BUBBLES` — при листании ⬅/➡ она уйдёт вместе с основным постом.
    """
    try:
        _, tweet_id = cb.data.split(":", 1)
    except ValueError:
        await cb.answer()
        return

    async with session_scope() as s:
        tweet = await s.get(Tweet, tweet_id)
        if not tweet or not (tweet.quote_author and tweet.quote_text):
            await cb.answer("Цитата недоступна", show_alert=False)
            return

    await cb.answer()

    # Защита от повторного клика: если юзер уже раскрыл цитату по этому посту
    # — второй клик не должен продублировать bubble. Простой трекер по
    # (user_id, tweet_id) внутри процесса.
    key = (cb.from_user.id, tweet_id)
    if _QUOTE_SHOWN.get(key):
        # Уже отправляли — не дублируем.
        return
    _QUOTE_SHOWN[key] = True

    from bot.delivery import _send_quote_message
    msg = await _send_quote_message(cb.bot, cb.from_user.id, tweet, reply_markup=None)
    if msg is not None:
        _track_bubble(cb.from_user.id, msg.message_id)


# (user_id, tweet_id) → True, чтобы повторный клик «Цитата» не плодил bubble'ы.
# Чистится автоматически: при листании пост выгружается из чата, а
# `_QUOTE_SHOWN` очищается по маске tweet_id через cleanup_bubbles(); в худшем
# случае энтри остаётся в dict, но это O(постов-показанных-в-сессии) — dev-грязь,
# не memleak.
_QUOTE_SHOWN: dict[tuple[int, str], bool] = {}


@router.callback_query(F.data.startswith("cm:"))
async def handle_comments(cb: CallbackQuery) -> None:
    """Тянем топ-7 залайканных реплаев под постом.

    Если реплаев нет (включая 404 от X SearchTimeline) — показываем alert,
    НЕ оставляя сообщение в чате: юзер иначе получает «Нет реплаев» как
    отдельный bubble, и после листания он висит как orphan.
    """
    try:
        _, tweet_id = cb.data.split(":", 1)
    except ValueError:
        await cb.answer()
        return

    # Передаём author_username — нужно для conversation-search fallback.
    from core.x_parser import parser as _parser
    author = None
    tweet_url = None
    try:
        async with session_scope() as s:
            tw = await s.get(Tweet, tweet_id)
            if tw:
                author = tw.author_username
                tweet_url = tw.url
    except Exception:
        pass

    err_msg = None
    try:
        replies = await _parser.get_top_replies(tweet_id, limit=7, author_username=author)
    except Exception as e:
        log.warning("get_top_replies failed for %s: %s", tweet_id, e)
        err_msg = str(e)[:120]
        replies = []

    log.info("comments: tweet=%s author=%s got=%d err=%s",
             tweet_id, author, len(replies or []), bool(err_msg))

    if not replies:
        # Нет ответов — alert вместо сообщения в чате. Телеграм покажет всплывашку
        # и в чате не останется мусора, который потом придётся удалять при листании.
        if err_msg:
            await cb.answer(
                f"X не отдаёт реплаи (404). Открой пост в X вручную.",
                show_alert=True,
            )
        else:
            await cb.answer(
                "Под этим постом пока нет реплаев.",
                show_alert=True,
            )
        return

    # Есть комменты — отвечаем нормальным bubble и ТРЕКАЕМ его message_id.
    await cb.answer("Достаю комменты…")

    import html as _html
    lines = ["<b>💬 Топ комментариев</b>", ""]
    for i, r in enumerate(replies, 1):
        txt = (r.text or "").strip()
        if len(txt) > 300:
            txt = txt[:299].rstrip() + "…"
        r_author = _html.escape(r.author_username or "—")
        body = _html.escape(txt)
        stats = []
        if r.likes_count >= 5:
            stats.append(f"❤️ {r.likes_count}")
        if r.retweets_count >= 3:
            stats.append(f"🔁 {r.retweets_count}")
        stat = " · ".join(stats)
        lines.append(f"<b>{i}. @{r_author}</b>" + (f"  <i>{stat}</i>" if stat else ""))
        lines.append(body)
        lines.append("")
    text = "\n".join(lines).strip()
    if len(text) > 4000:
        text = text[:3990].rstrip() + "…"
    try:
        sent = await cb.message.reply(text, parse_mode=ParseMode.HTML,
                                      disable_web_page_preview=True)
        _track_bubble(cb.from_user.id, sent.message_id)
    except TelegramAPIError as e:
        log.debug("comments reply failed: %s", e)


# ----------------------------- Pause / Resume -----------------------------


@router.callback_query(F.data == "ctl:pause")
async def handle_pause(cb: CallbackQuery) -> None:
    async with session_scope() as s:
        user = await s.get(User, cb.from_user.id)
        if user:
            user.paused = True
    await cb.message.answer(
        "⏸ Пауза. Когда захочешь продолжить — жми кнопку ниже.",
        reply_markup=resume_kb(),
    )
    await cb.answer()


@router.callback_query(F.data == "ctl:resume")
async def handle_resume(cb: CallbackQuery) -> None:
    async with session_scope() as s:
        user = await s.get(User, cb.from_user.id)
        if user:
            user.paused = False
    await cb.message.edit_reply_markup(reply_markup=None)
    await cb.message.answer("▶️ Возобновил. Следующая подборка придёт по расписанию.")
    await cb.answer()
