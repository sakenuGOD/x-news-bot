"""Онбординг: /start → X username → анализ → вопросы → интервал → первая доставка.

Два режима анализа:
  1. Если X-креды бота (X_USERNAME/X_PASSWORD в .env) валидны — скрапим подписки
     пользователя через tweety-ns и анализируем их твиты.
  2. Иначе — пользователь описывает интересы своими словами (state=awaiting_interests_text),
     Claude Sonnet извлекает темы и задаёт уточнения.

В обоих вариантах контент-пул для доставки — demo-арсенал (seed_demo_tweets),
потому что без X-доступа мы не можем собрать реальные свежие посты подписок.

Состояния онбординга (хранятся в User.onboarding_state):
  'awaiting_username'        — ждём @username из X
  'awaiting_interests_text'  — X недоступен, ждём описание интересов
  'awaiting_answer_0'        — ждём ответ на первый уточняющий вопрос
  'awaiting_answer_1'        — ждём ответ на второй
  'awaiting_interval'        — ждём клик по кнопке интервала
  'awaiting_more_text'       — обработка «хочу больше…» (preferences.py)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message
from sqlalchemy import func, select

from bot.delivery import deliver_news_to_user
from bot.keyboards import (
    interval_kb,
    main_menu_kb,
    setup_connect_kb,
    setup_ct0_kb,
    setup_retry_kb,
    setup_verified_kb,
    setup_welcome_kb,
)
from config import settings
from core import ai_client
from core import embeddings as emb
from core.demo_data import seed_demo_tweets
from core.filters import is_trash
from core.x_parser import RawTweet, XAuthError, parser
from db.database import session_scope
from db.models import FollowedAuthor, Tweet, User

log = logging.getLogger(__name__)

router = Router(name="onboarding")

_USERNAME_RE = re.compile(r"^@?([A-Za-z0-9_]{1,15})$")


def _x_auth_available() -> bool:
    """Есть ли хоть один источник X-авторизации."""
    from pathlib import Path
    if settings.x_auth_token:
        return True
    if settings.x_username and settings.x_password:
        return True
    p = Path(settings.x_cookies_path)
    if p.exists() and p.stat().st_size > 0:
        return True
    return False


# ============================= commands =============================


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    async with session_scope() as s:
        user = await s.get(User, message.from_user.id)
        if user and user.onboarding_done:
            await message.answer(
                "С возвращением. Что делаем?",
                reply_markup=main_menu_kb(paused=bool(user.paused)),
            )
            return
        if not user:
            user = User(telegram_id=message.from_user.id)
            s.add(user)
        # Сбрасываем любое старое состояние.
        user.onboarding_state = None
        user.onboarding_payload = {}

    await message.answer(
        "<b>Привет!</b> Я соберу тебе персональную ленту из X (Twitter).\n\n"
        "Под каждым постом будут 👍/👎 — лента подстроится. "
        "Reply с вопросом под любой пост — обсудим его вместе.\n\n"
        "С чего начнём?",
        parse_mode="HTML",
        reply_markup=setup_welcome_kb(),
    )


# ============================= setup callbacks =============================


@router.callback_query(F.data == "setup:info")
async def cb_setup_info(cb: CallbackQuery) -> None:
    """Показываем объяснение — редактируем текущее сообщение, не плодим копии."""
    info_text = (
        "<b>Как это работает</b>\n\n"
        "1. Ты даёшь мне <b>auth_token</b> своего X-аккаунта — это cookie из браузера. "
        "По нему я смогу смотреть на кого ты подписан.\n"
        "2. Я читаю недавние посты твоих подписок, считаю эмбеддинги, анализирую интересы.\n"
        "3. Задаю пару уточняющих вопросов.\n"
        "4. Дальше — отчёт по твоей ленте раз в выбранный интервал.\n\n"
        "<i>Если не хочешь давать X-токен — жми «Демо», покажу как это выглядит.</i>"
    )
    try:
        await cb.message.edit_text(info_text, parse_mode="HTML", reply_markup=setup_welcome_kb())
    except TelegramAPIError:
        # Если исходное сообщение нельзя редактировать (например, к нему уже цеплялось
        # медиа или оно слишком старое) — просто тихо отдаём answer без нового сообщения.
        pass
    await cb.answer()


@router.callback_query(F.data == "setup:back")
async def cb_setup_back(cb: CallbackQuery) -> None:
    async with session_scope() as s:
        user = await s.get(User, cb.from_user.id)
        if user:
            user.onboarding_state = None
    try:
        await cb.message.edit_reply_markup(reply_markup=setup_welcome_kb())
    except Exception:
        await cb.message.answer("Что делаем?", reply_markup=setup_welcome_kb())
    await cb.answer()


@router.callback_query(F.data == "setup:demo")
async def cb_setup_demo(cb: CallbackQuery) -> None:
    """Тот же путь что /demo, но из кнопки."""
    await cb.answer("Запускаю демо…")
    try:
        await cb.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass
    # Переиспользуем логику cmd_demo — делегируем через фейковое сообщение.
    # Проще: продублировать минимальную логику тут.
    await _run_demo(cb.message, cb.from_user.id)


@router.callback_query(F.data == "setup:connect")
async def cb_setup_connect(cb: CallbackQuery) -> None:
    await cb.message.answer(
        "<b>Подключение X-аккаунта</b>\n\n"
        "Мне нужен cookie <code>auth_token</code> из твоего браузера — это "
        "обычный способ авторизации (так же делают расширения вроде Cookie-Editor).\n\n"
        "<b>Где взять:</b>\n"
        "• <b>ПК:</b> x.com в браузере → F12 (DevTools) → вкладка <i>Application</i> → "
        "слева <i>Cookies</i> → <code>https://x.com</code> → строка <code>auth_token</code> → "
        "копируй значение.\n"
        "• <b>Android:</b> браузер Kiwi или Yandex + расширение Cookie-Editor — "
        "удобно прямо с телефона.\n"
        "• <b>iPhone:</b> проще сделать это один раз на ПК — токен живёт месяцами.\n\n"
        "⚠️ Используй <b>бёрнер-аккаунт</b> X (отдельный), а не основной — "
        "X может ограничить API-активность.\n\n"
        "Когда скопировал — жми <b>📥 Ввести auth_token</b>.",
        parse_mode="HTML",
        reply_markup=setup_connect_kb(),
    )
    await cb.answer()


@router.callback_query(F.data == "setup:enter_token")
async def cb_setup_enter_token(cb: CallbackQuery) -> None:
    async with session_scope() as s:
        user = await s.get(User, cb.from_user.id)
        if not user:
            user = User(telegram_id=cb.from_user.id)
            s.add(user)
        user.onboarding_state = "awaiting_auth_token"

    await cb.message.answer(
        "📥 Пришли одним сообщением значение <code>auth_token</code> "
        "(длинная hex-строка 40+ символов).\n\n"
        "Я сразу удалю твоё сообщение из чата — чтобы токен не лежал в истории.",
        parse_mode="HTML",
    )
    await cb.answer()


@router.callback_query(F.data == "setup:enter_ct0")
async def cb_setup_enter_ct0(cb: CallbackQuery) -> None:
    async with session_scope() as s:
        user = await s.get(User, cb.from_user.id)
        if not user:
            return
        user.onboarding_state = "awaiting_ct0"
    await cb.message.answer(
        "📥 Пришли значение <code>ct0</code> одним сообщением.",
        parse_mode="HTML",
    )
    await cb.answer()


@router.callback_query(F.data == "setup:skip_ct0")
async def cb_setup_skip_ct0(cb: CallbackQuery) -> None:
    await cb.answer("Проверяю X-auth…")
    await _verify_and_prompt_username(cb.message, cb.from_user.id)


@router.callback_query(F.data == "setup:start_username")
async def cb_setup_start_username(cb: CallbackQuery) -> None:
    async with session_scope() as s:
        user = await s.get(User, cb.from_user.id)
        if not user:
            return
        user.onboarding_state = "awaiting_username"
    try:
        await cb.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass
    await cb.message.answer(
        "Напиши свой X username (например <code>@elonmusk</code>). "
        "Я посмотрю на кого ты подписан и соберу профиль интересов.",
        parse_mode="HTML",
    )
    await cb.answer()


@router.message(Command("demo"))
async def cmd_demo(message: Message) -> None:
    """Командная альтернатива кнопке «Свежий отчёт»."""
    await _run_demo(message, message.from_user.id)


async def _run_demo(message: Message, user_id: int) -> None:
    """Демо-режим: минимальный onboarding + запуск отчёта.

    Если X timeline пустой (нет auth), build_report сам упадёт в demo-пул.
    """
    async with session_scope() as s:
        user = await s.get(User, user_id)
        if not user:
            user = User(telegram_id=user_id)
            s.add(user)

        user.onboarding_done = True
        user.onboarding_state = None
        user.paused = False
        if not user.x_username:
            user.x_username = "demo_user"
        if not user.cluster_weights:
            user.cluster_weights = {
                "tech": 0.5, "ai": 0.5, "science": 0.3, "business": 0.25, "news": 0.3,
            }
        if not user.delivery_interval_hours:
            user.delivery_interval_hours = 3

        # Сеем демо-пул если его нет — нужен как fallback-контент.
        pool_count = await s.scalar(
            select(func.count(Tweet.tweet_id)).where(Tweet.tweet_id.like("demo_%"))
        )
        if not pool_count:
            await seed_demo_tweets(s)

    # Запускаем report flow — тот же что по кнопке «Свежий отчёт».
    from bot.handlers.report import cb_new_report as _trigger_report  # lazy to avoid cycle
    # Пересылаем управление в overview-генератор — он сам покажет прогресс.
    # Для этого эмулируем CallbackQuery не надо — просто зовём build_report напрямую.
    from core.report import build_report, save_report
    from bot.handlers.report import _render_report

    status = await message.answer("⏳ Поднимаю данные…")

    async def _progress(msg: str) -> None:
        try:
            await status.edit_text(msg)
        except Exception:
            pass

    async with session_scope() as s:
        user = await s.get(User, user_id)
        if not user:
            return
        try:
            report = await build_report(s, user, window_hours=1.5, progress=_progress)
        except Exception as e:
            log.exception("demo report failed")
            await status.edit_text(f"❌ Сбой при сборке: <code>{type(e).__name__}</code>",
                                   parse_mode="HTML")
            return

    save_report(report)
    try:
        await status.delete()
    except Exception:
        pass
    await _render_report(message, report)


@router.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    async with session_scope() as s:
        user = await s.get(User, message.from_user.id)
        if user:
            user.onboarding_done = False
            user.onboarding_state = None
            user.onboarding_payload = {}
    await message.answer(
        "Ок, обнулил профиль. С чего начнём?",
        reply_markup=setup_welcome_kb(),
    )


@router.message(Command("interval"))
async def cmd_interval(message: Message) -> None:
    await message.answer("Выбери интервал доставки:", reply_markup=interval_kb())


@router.message(Command("pause"))
async def cmd_pause(message: Message) -> None:
    async with session_scope() as s:
        user = await s.get(User, message.from_user.id)
        if user:
            user.paused = True
    await message.answer("⏸ Поставил на паузу. /resume — возобновить.")


@router.message(Command("resume"))
async def cmd_resume(message: Message) -> None:
    async with session_scope() as s:
        user = await s.get(User, message.from_user.id)
        if user:
            user.paused = False
    await message.answer("▶️ Работаю.")


# ============================= text routing =============================


@router.message(F.text)
async def handle_onboarding_text(message: Message) -> None:
    async with session_scope() as s:
        user = await s.get(User, message.from_user.id)
        if not user:
            await message.answer("Сначала отправь /start.")
            return
        state = user.onboarding_state

    if state == "awaiting_auth_token":
        await _handle_auth_token(message)
    elif state == "awaiting_ct0":
        await _handle_ct0(message)
    elif state == "awaiting_username":
        await _handle_username(message)
    elif state == "awaiting_interests_text":
        await _handle_interests_text(message)
    elif state == "awaiting_answer_0":
        await _handle_answer(message, 0)
    elif state == "awaiting_answer_1":
        await _handle_answer(message, 1)
    elif state == "awaiting_more_text":
        from bot.handlers.preferences import apply_preference_text  # noqa: local import
        await apply_preference_text(message, message.text)
    else:
        # Нет активного state. Но может быть активный discussion-контекст —
        # продолжаем обсуждение поста без reply.
        from bot.handlers.discussion import handle_continuation
        handled = await handle_continuation(message)
        if handled:
            return
        # Совсем без контекста — молча игнорируем, чтобы не шуметь.


# ============================= auth token handlers =============================


_AUTH_TOKEN_RE = re.compile(r"^[A-Za-z0-9]{30,}$")


async def _handle_auth_token(message: Message) -> None:
    """Получили от юзера auth_token — валидируем, сохраняем, спрашиваем ct0."""
    token = (message.text or "").strip()

    # Токен часто hex 40 символов, но X мог поменять формат — принимаем 30+ alnum.
    if not _AUTH_TOKEN_RE.match(token):
        await message.answer(
            "Хмм, не похоже на auth_token — должна быть длинная строка 30+ символов "
            "без пробелов. Попробуй ещё раз или /start заново."
        )
        return

    # Удаляем сообщение пользователя с секретом.
    try:
        await message.delete()
    except Exception:
        pass

    # Сохраняем как cookies.json. ct0 пока пусто — добавим следующим шагом если захочет.
    await parser.save_cookies_from_dict({"auth_token": token})

    await message.answer(
        "✅ auth_token сохранён.\n\n"
        "Опционально — <b>ct0</b> (CSRF-токен, нужен для некоторых операций). "
        "Если пропустишь — чтение подписок и таймлайнов всё равно будет работать.",
        parse_mode="HTML",
        reply_markup=setup_ct0_kb(),
    )


async def _handle_ct0(message: Message) -> None:
    ct0 = (message.text or "").strip()
    if len(ct0) < 10:
        await message.answer("ct0 должен быть длинной строкой. Попробуй ещё раз или «Пропустить».")
        return

    try:
        await message.delete()
    except Exception:
        pass

    # Читаем текущий cookies.json, докидываем ct0, сохраняем.
    from pathlib import Path
    cookies_path = Path(settings.x_cookies_path)
    existing = {}
    if cookies_path.exists() and cookies_path.stat().st_size > 0:
        try:
            existing = json.loads(cookies_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}
    existing["ct0"] = ct0
    await parser.save_cookies_from_dict(existing)

    await _verify_and_prompt_username(message, message.from_user.id)


async def _verify_and_prompt_username(message: Message, user_id: int) -> None:
    """Проверяем X-auth пробным запросом, показываем результат и предлагаем username."""
    await message.answer("⏳ Проверяю подключение к X…")
    try:
        await parser._ensure_client()
        info = await parser.get_author_info("x")
        ok = info is not None
    except XAuthError as e:
        await message.answer(
            f"❌ X не авторизовался: {e}\n\nПопробуй ввести токен ещё раз.",
            reply_markup=setup_retry_kb(),
        )
        return
    except Exception as e:
        log.warning("X verify error: %s", e)
        await message.answer(
            f"❌ Ошибка при тестовом запросе: {type(e).__name__}.\n"
            "Возможно auth_token просрочен — попробуй снова.",
            reply_markup=setup_retry_kb(),
        )
        return

    if not ok:
        await message.answer(
            "⚠️ Подключился, но тестовый запрос вернул пусто. X может быть тормозит — "
            "попробуем дальше.",
            reply_markup=setup_verified_kb(),
        )
        return

    async with session_scope() as s:
        user = await s.get(User, user_id)
        if user:
            user.onboarding_state = None

    await message.answer(
        f"✅ X подключён (проверил: у <code>@x</code> {info.followers_count:,} подписчиков).\n\n"
        "Готов анализировать твой профиль.",
        parse_mode="HTML",
        reply_markup=setup_verified_kb(),
    )


# ============================= X-based path =============================


async def _handle_username(message: Message) -> None:
    raw = (message.text or "").strip()
    m = _USERNAME_RE.match(raw)
    if not m:
        await message.answer(
            "Хмм, не похоже на X username. Формат: буквы, цифры, подчёркивания, до 15 символов."
        )
        return
    username = m.group(1)

    # Без источника X-авторизации скрапить нечем.
    if not _x_auth_available():
        await _switch_to_text_mode(
            message,
            username,
            reason=(
                "⚠️ У бота нет auth_token для X. Добавь в <code>.env</code> переменную "
                "<code>X_AUTH_TOKEN</code> (DevTools → Application → Cookies → x.com → "
                "скопировать auth_token) — и /reset. Пока соберу профиль иначе."
            ),
        )
        return

    status = await message.answer(f"Смотрю подписки @{username}… Это может занять минуту ⏳")

    try:
        following = await parser.get_following(username, limit=settings.max_authors_per_user)
    except XAuthError as e:
        log.warning("X auth error for %s: %s", username, e)
        try:
            await status.delete()
        except Exception:
            pass
        await _switch_to_text_mode(
            message,
            username,
            reason=f"⚠️ X auth сломался: {e}. Соберу профиль иначе.",
        )
        return
    except Exception as e:
        log.warning("get_following error for %s: %s", username, e)
        following = []

    if not following:
        # X auth есть, но по какой-то причине подписки не отдались — уходим в text-mode.
        try:
            await status.delete()
        except Exception:
            pass
        await _switch_to_text_mode(
            message,
            username,
            reason="Не смог достать подписки из X (аккаунт приватный, rate limit или X сбойнул).",
        )
        return

    await status.edit_text(
        f"Нашёл {len(following)} подписок у @{username}. Беру твою ленту для анализа…"
    )

    # ВАЖНО: раньше дёргали per-author fetch по 15-30 авторам — X ловил rate-limit
    # и онбординг зависал. Теперь — ОДИН запрос get_home_timeline, в нём уже
    # свежие посты подписок вперемешку. Для анализа интересов этого хватает
    # с запасом и не тригерит 429.
    sample_tweets: list[RawTweet] = await parser.get_home_timeline(limit=80)
    sample_tweets = [t for t in sample_tweets if not is_trash(t)[0]]

    if not sample_tweets:
        await _switch_to_text_mode(
            message,
            username,
            reason="У подписок не нашлось содержательных свежих постов. Пойдём через текст.",
        )
        return

    # Эмбеддим и анализируем.
    texts = [t.text for t in sample_tweets]
    embs = await emb.embed_batch(texts)
    valid_vectors = [e for e in embs if e]
    pref_vec = emb.mean_vector(valid_vectors) if valid_vectors else None
    if pref_vec:
        pref_vec = emb.normalize(pref_vec)

    analysis = await ai_client.analyze_onboarding(sample_tweets[:40])

    async with session_scope() as s:
        user = await s.get(User, message.from_user.id)
        if not user:
            return
        user.x_username = username
        user.preference_vector = pref_vec
        user.cluster_weights = analysis.cluster_weights or {"tech": 0.5, "news": 0.5}
        user.onboarding_state = "awaiting_answer_0"
        user.onboarding_payload = {
            "hypothesis": analysis.hypothesis,
            "questions": analysis.questions[:2],
            "answers": [],
        }
        for author in following:
            s.add(FollowedAuthor(user_id=user.telegram_id, author_username=author))
        await _save_tweets_to_cache(s, sample_tweets, embs)
        # Сеем demo-пул тоже — чтобы доставке было из чего брать, пока scheduler
        # не собрал свежие посты подписок.
        await seed_demo_tweets(s)

    q0 = (analysis.questions or ["Что бы ты хотел видеть чаще?"])[0]
    await message.answer(
        f"Кажется, тебе интересно: <i>{analysis.hypothesis}</i>\n\n"
        f"Чтобы настроить точнее — ответь на пару вопросов.\n\n"
        f"❓ {q0}",
        parse_mode="HTML",
    )


# ============================= text-based path =============================


async def _switch_to_text_mode(message: Message, username: str, reason: str) -> None:
    """X-путь недоступен — просим описать интересы своими словами."""
    async with session_scope() as s:
        user = await s.get(User, message.from_user.id)
        if not user:
            return
        user.x_username = username
        user.onboarding_state = "awaiting_interests_text"
        user.onboarding_payload = {}

    await message.answer(
        f"{reason}\n\n"
        "Расскажи своими словами — <b>что тебе интересно</b>? "
        "Темы, любимые авторы, сферы, от чего горят глаза. "
        "2-4 предложения достаточно — я разберу и задам пару уточнений.\n\n"
        "<i>Пример:</i> <i>«Интересуюсь AI и LLM, читаю про стартапы и инди-хакинг, "
        "слежу за крипто-индустрией на уровне новостей, политика не интересна».</i>",
        parse_mode="HTML",
    )


async def _handle_interests_text(message: Message) -> None:
    text = (message.text or "").strip()
    if len(text) < 10:
        await message.answer(
            "Напиши хотя бы пару предложений — мне нужно больше контекста, чтобы собрать ленту."
        )
        return

    status = await message.answer("⏳ Разбираю описание, подбираю темы…")

    analysis = await ai_client.analyze_interests_text(text)

    # preference_vector — из эмбеддинга самого текста. Это честный «центр» его интересов.
    pref_vec_raw = await emb.embed_text(text)
    pref_vec = emb.normalize(pref_vec_raw) if pref_vec_raw else None

    async with session_scope() as s:
        user = await s.get(User, message.from_user.id)
        if not user:
            return
        user.preference_vector = pref_vec
        user.cluster_weights = analysis.cluster_weights or {"tech": 0.4, "news": 0.3}
        user.onboarding_state = "awaiting_answer_0"
        user.onboarding_payload = {
            "hypothesis": analysis.hypothesis,
            "questions": analysis.questions[:2],
            "answers": [],
            "interests_raw": text,
        }
        # Сеем арсенал чтобы /demo и регулярная доставка имели контент.
        await seed_demo_tweets(s)

    try:
        await status.delete()
    except Exception:
        pass

    q0 = (analysis.questions or ["Что бы ты хотел видеть чаще?"])[0]
    await message.answer(
        f"Понял так: <i>{analysis.hypothesis}</i>\n\n"
        f"Чтобы настроить точнее — два уточнения.\n\n"
        f"❓ {q0}",
        parse_mode="HTML",
    )


# ============================= shared: Q&A + finalize =============================


async def _handle_answer(message: Message, index: int) -> None:
    answer = (message.text or "").strip()
    if not answer:
        await message.answer("Напиши пару слов, чтобы я понял.")
        return

    next_question: str | None = None
    finalize = False

    async with session_scope() as s:
        user = await s.get(User, message.from_user.id)
        if not user:
            return
        payload = dict(user.onboarding_payload or {})
        answers: list[str] = list(payload.get("answers", []))
        answers.append(answer)
        payload["answers"] = answers
        questions: list[str] = list(payload.get("questions", []))

        if index == 0 and len(questions) >= 2:
            user.onboarding_state = "awaiting_answer_1"
            next_question = questions[1]
        else:
            user.onboarding_state = "awaiting_interval"
            finalize = True
        user.onboarding_payload = payload

    if next_question:
        await message.answer(f"❓ {next_question}")
        return

    if finalize:
        await _finalize_onboarding(message)


async def _finalize_onboarding(message: Message) -> None:
    async with session_scope() as s:
        user = await s.get(User, message.from_user.id)
        if not user:
            return
        payload = user.onboarding_payload or {}
        refinement = await ai_client.process_onboarding_answers(
            hypothesis=payload.get("hypothesis", ""),
            questions=payload.get("questions", []),
            answers=payload.get("answers", []),
            initial_weights=user.cluster_weights or {},
        )
        user.cluster_weights = refinement.cluster_weights
        reply = refinement.reply

    await message.answer(
        f"{reply}\n\nКак часто присылать новости?",
        reply_markup=interval_kb(),
    )


@router.callback_query(F.data.startswith("intv:"))
async def interval_chosen(cb: CallbackQuery) -> None:
    hours = int(cb.data.split(":", 1)[1])
    async with session_scope() as s:
        user = await s.get(User, cb.from_user.id)
        if not user:
            await cb.answer("Сначала /start", show_alert=True)
            return
        user.delivery_interval_hours = hours
        user.onboarding_done = True
        user.onboarding_state = None
        user.paused = False

    try:
        await cb.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass
    await cb.message.answer(
        f"Готово. Буду собирать отчёт раз в {hours} ч и обновлять ленту.",
        reply_markup=main_menu_kb(),
    )
    await cb.answer()


# ============================= helpers =============================


async def _save_tweets_to_cache(
    session,
    tweets: list[RawTweet],
    embeddings: list[list[float] | None],
) -> None:
    """Сохраняет собранные при онбординге твиты в Tweet-кэш. Молча игнорирует дубликаты."""
    from core.embeddings import nearest_cluster

    existing_ids: set[str] = set()
    if tweets:
        ids = [t.tweet_id for t in tweets]
        rows = (await session.execute(select(Tweet.tweet_id).where(Tweet.tweet_id.in_(ids)))).all()
        existing_ids = {r[0] for r in rows}

    for t, e in zip(tweets, embeddings):
        if t.tweet_id in existing_ids:
            continue
        topic = None
        if e:
            topic, _ = await nearest_cluster(e)
        session.add(
            Tweet(
                tweet_id=t.tweet_id,
                author_username=t.author_username,
                author_display_name=t.author_display_name,
                text=t.text,
                url=t.url,
                image_url=t.image_url,
                media_type=t.media_type,
                embedding=e,
                created_at=t.created_at.replace(tzinfo=None),
                likes_count=t.likes_count,
                retweets_count=t.retweets_count,
                replies_count=t.replies_count,
                topic=topic,
                source_trust_score=0.5,
            )
        )
