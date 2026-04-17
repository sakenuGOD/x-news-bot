"""Обсуждение поста.

Два режима:
  1. **Reply-triggered** — пользователь нативно reply'нул на сообщение с твитом
     и задал вопрос. Начинаем диалог: запоминаем «активный» twit_id в
     in-memory state, отвечаем reply'ом.
  2. **Continuation** — после reply-ответа пользователь пишет просто следующее
     сообщение (без reply). Считаем это продолжением обсуждения того же поста,
     если последнее сообщение было недавно.

Активное обсуждение живёт в памяти процесса (dict user_id → context), с TTL
10 минут — не пережило ресайз, и это ок.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message
from sqlalchemy import select

from core import ai_client
from db.database import session_scope
from db.models import SentNews, Tweet, User

log = logging.getLogger(__name__)

router = Router(name="discussion")

_MIN_QUESTION_LEN = 3
_CONTINUATION_TTL_SEC = 600  # 10 минут — если пауза дольше, следующее сообщение не считаем продолжением


@dataclass
class DiscussionContext:
    tweet_id: str
    tweet_text: str
    author: str
    history: list[tuple[str, str]] = field(default_factory=list)  # [(role, text), ...]
    last_ts: float = field(default_factory=time.time)
    reply_to_message_id: int | None = None  # для reply на оригинальный пост


# user_id -> context
_ACTIVE: dict[int, DiscussionContext] = {}


def _set_active(user_id: int, ctx: DiscussionContext) -> None:
    ctx.last_ts = time.time()
    _ACTIVE[user_id] = ctx


def _get_active(user_id: int) -> DiscussionContext | None:
    ctx = _ACTIVE.get(user_id)
    if ctx is None:
        return None
    if time.time() - ctx.last_ts > _CONTINUATION_TTL_SEC:
        # Слишком давно — обнуляем.
        _ACTIVE.pop(user_id, None)
        return None
    return ctx


def has_active_discussion(user_id: int) -> bool:
    return _get_active(user_id) is not None


# ============================= reply-triggered =============================


@router.message(F.reply_to_message & F.text)
async def handle_reply_to_tweet(message: Message) -> None:
    """Срабатывает когда пользователь reply'нул на сообщение с постом."""
    replied = message.reply_to_message
    if not replied or not message.text:
        return
    question = message.text.strip()
    if len(question) < _MIN_QUESTION_LEN:
        return

    user_id = message.from_user.id
    replied_mid = replied.message_id

    async with session_scope() as s:
        user = await s.get(User, user_id)
        if not user:
            return
        sent = (
            await s.execute(
                select(SentNews).where(
                    SentNews.user_id == user_id,
                    SentNews.telegram_message_id == replied_mid,
                )
            )
        ).scalar_one_or_none()
        if not sent:
            # Это reply на другое сообщение (не твит) — пропускаем.
            return
        tweet = await s.get(Tweet, sent.tweet_id)
        if not tweet:
            return
        post_text = tweet.text
        author = tweet.author_username
        tweet_id = tweet.tweet_id

    typing_task = asyncio.create_task(_keep_typing(message))
    try:
        answer = await ai_client.discuss_post(post_text, author, question)
    finally:
        typing_task.cancel()

    # Ответ reply на оригинальный пост-сообщение.
    try:
        sent_msg = await message.bot.send_message(
            chat_id=user_id,
            text=answer,
            reply_to_message_id=replied_mid,
            disable_web_page_preview=True,
        )
    except TelegramAPIError as e:
        log.warning("discussion reply failed: %s", e)
        try:
            sent_msg = await message.reply(answer, disable_web_page_preview=True)
        except TelegramAPIError as e2:
            log.error("discussion fallback failed: %s", e2)
            return

    # Сохраняем активный контекст — следующие сообщения без reply будут
    # считаться продолжением этого обсуждения.
    ctx = DiscussionContext(
        tweet_id=tweet_id,
        tweet_text=post_text,
        author=author,
        history=[("user", question), ("assistant", answer)],
        reply_to_message_id=replied_mid,
    )
    _set_active(user_id, ctx)


# ============================= continuation (без reply) =============================


async def handle_continuation(message: Message) -> bool:
    """Продолжение активного обсуждения. Вызывается из onboarding.handle_onboarding_text
    когда у пользователя нет других state'ов но есть активный discussion-контекст.

    Возвращает True если обработали (следующий handler не должен ничего делать).
    """
    user_id = message.from_user.id
    ctx = _get_active(user_id)
    if ctx is None:
        return False

    text = (message.text or "").strip()
    if not text:
        return False

    # Добавляем реплику пользователя в историю.
    ctx.history.append(("user", text))

    typing_task = asyncio.create_task(_keep_typing(message))
    try:
        answer = await _continue_conversation(ctx)
    finally:
        typing_task.cancel()

    ctx.history.append(("assistant", answer))
    _set_active(user_id, ctx)  # обновляем last_ts

    # В режиме продолжения reply'им на исходный пост если он ещё в чате.
    try:
        if ctx.reply_to_message_id:
            await message.bot.send_message(
                chat_id=user_id,
                text=answer,
                reply_to_message_id=ctx.reply_to_message_id,
                disable_web_page_preview=True,
            )
        else:
            await message.reply(answer, disable_web_page_preview=True)
    except TelegramAPIError:
        # Пост-сообщение могло удалиться — отвечаем reply'ом на последнее сообщение юзера.
        try:
            await message.reply(answer, disable_web_page_preview=True)
        except TelegramAPIError as e:
            log.error("continuation send failed: %s", e)

    return True


async def _continue_conversation(ctx: DiscussionContext) -> str:
    """Шлём Claude пост + полную историю обсуждения, получаем следующий ответ."""
    system = (
        "Ты обсуждаешь с пользователем пост из X, это уже идущий диалог. "
        "Отвечай коротко (2-5 предложений) на русском, по существу — "
        "без воды, без «давайте», без переиначивания вопроса. "
        "Если в разговоре всплывают факты которых у тебя нет — честно говори."
    )
    # Формируем конверсацию: (1) пост как контекст системы, (2) история.
    msgs = [
        {
            "role": "user",
            "content": f"Пост от @{ctx.author}:\n«{ctx.tweet_text}»\n\n(Далее диалог про этот пост.)",
        },
        {"role": "assistant", "content": "Понял, давай обсудим."},
    ]
    for role, text in ctx.history:
        msgs.append({"role": role, "content": text})

    try:
        from anthropic import AsyncAnthropic
        from config import settings as _s
        # Используем тот же клиент что в ai_client
        from core.ai_client import _get_client
        client = _get_client()
        resp = await client.messages.create(
            model=_s.model_haiku,
            max_tokens=600,
            temperature=0.55,
            system=system,
            messages=msgs,
        )
        parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return ("".join(parts)).strip() or "Хм, не получилось ответить — попробуй переформулировать."
    except Exception as e:
        log.warning("continue_conversation failed: %s", e)
        return "Что-то пошло не так — попробуй ещё раз."


# ============================= utils =============================


async def _keep_typing(message: Message) -> None:
    try:
        while True:
            try:
                await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
            except TelegramAPIError:
                pass
            await asyncio.sleep(4.0)
    except asyncio.CancelledError:
        pass
