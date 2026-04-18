"""APScheduler задачи:

  1. fetch_tweets_job — каждые FETCH_INTERVAL_MINUTES собирает твиты авторов,
     фильтрует, эмбеддит и складывает в кэш (Tweet + Chroma).

  2. delivery_job — каждый час смотрит, каким пользователям пора присылать
     (last_delivered_at + interval_hours ≤ now) и отправляет подборку.

  3. implicit_decay_job — раз в 6 часов проходит по «показано, не кликнуто»
     и двигает preference_vector через неявный негатив.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable

from aiogram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import and_, distinct, or_, select

from bot.delivery import deliver_news_to_user
from config import settings
from core import embeddings as emb
from core import ai_client
from core.filters import (
    dedupe_by_embedding,
    engagement_rate,
    is_noise_by_embedding,
    is_trash,
    needs_antifake_check,
)
from core.x_parser import RawTweet, compute_trust_score, parser
from db.database import session_scope
from db.models import Feedback, FollowedAuthor, Tweet, User
from db.vector_store import upsert_tweets

log = logging.getLogger(__name__)


# ------------------------- 1. fetch_tweets_job ----------------------------


async def fetch_tweets_job() -> None:
    """Собираем свежие твиты всех авторов, которых читают наши пользователи."""
    log.info("fetch_tweets_job started")

    # Уникальные авторы по всем пользователям.
    async with session_scope() as s:
        rows = (await s.execute(select(distinct(FollowedAuthor.author_username)))).all()
        authors = [r[0] for r in rows]

    if not authors:
        log.info("fetch_tweets_job: no authors yet")
        return

    log.info("fetching from %d authors", len(authors))

    # Собираем твиты батчами — чтобы не держать всё в памяти и не улетать по rate-limit.
    chunk_size = 20
    total_saved = 0
    for start in range(0, len(authors), chunk_size):
        chunk = authors[start : start + chunk_size]
        try:
            raw_tweets = await parser.get_recent_tweets_for_authors(
                chunk, limit_per_author=settings.tweets_per_author_on_fetch
            )
        except Exception as e:
            log.exception("parser failure on chunk: %s", e)
            continue

        saved = await _process_and_save(raw_tweets)
        total_saved += saved

    log.info("fetch_tweets_job done, saved=%d new tweets", total_saved)


async def _process_and_save(raw_tweets: list[RawTweet]) -> int:
    """Фильтрует, эмбеддит, дедуплицирует, сохраняет."""
    if not raw_tweets:
        return 0

    # 1) trash filter (дёшево).
    cleaned: list[RawTweet] = []
    for t in raw_tweets:
        trashy, reason = is_trash(t)
        if trashy:
            log.debug("drop %s: %s", t.tweet_id, reason)
            continue
        cleaned.append(t)
    if not cleaned:
        return 0

    # 2) Проверяем — какие уже в кэше. Эмбеддим только новые, для старых просто обновим метрики.
    async with session_scope() as s:
        existing_ids_rows = (
            await s.execute(select(Tweet.tweet_id).where(Tweet.tweet_id.in_([t.tweet_id for t in cleaned])))
        ).all()
        existing_ids = {r[0] for r in existing_ids_rows}

    new_tweets = [t for t in cleaned if t.tweet_id not in existing_ids]
    if not new_tweets:
        return 0

    # 3) Эмбеддинги батчом — OpenAI берёт до 2048 строк за раз, мы ограничиваем 100.
    texts = [t.text for t in new_tweets]
    embs = await emb.embed_batch(texts)

    # 4) Пара (tweet, emb) только если эмбеддинг получился.
    pairs = [(t, e) for t, e in zip(new_tweets, embs) if e]
    if not pairs:
        return 0

    # 4b) Anchor-noise filter (рекламные посты без явных маркеров).
    # Отсекаем ДО сохранения в БД — иначе они попадают в candidate pool
    # рекомендера и светятся в «моей ленте».
    noise_dropped = 0
    kept_pairs: list[tuple[RawTweet, list[float]]] = []
    for t, e in pairs:
        is_noise, reason = await is_noise_by_embedding(e)
        if is_noise:
            noise_dropped += 1
            log.debug("anchor-noise pre-save drop %s: %s", t.tweet_id, reason)
            continue
        kept_pairs.append((t, e))
    if noise_dropped:
        log.info("anchor-noise pre-save: %d/%d dropped", noise_dropped, len(pairs))
    pairs = kept_pairs
    if not pairs:
        return 0

    # 5) Дедупликация внутри батча.
    pairs = dedupe_by_embedding(pairs)

    # 6) Собираем trust scores (по уникальным авторам — чтобы не дёргать X на каждый твит).
    unique_authors = {p[0].author_username for p in pairs}
    author_trust: dict[str, float] = {}
    author_followers: dict[str, int] = {}
    for uname in unique_authors:
        info = await parser.get_author_info(uname)
        if info:
            er = engagement_rate(
                info.followers_count,
                sum(p[0].likes_count for p in pairs if p[0].author_username == uname),
                sum(p[0].retweets_count for p in pairs if p[0].author_username == uname),
                sum(p[0].replies_count for p in pairs if p[0].author_username == uname),
            )
            info.recent_engagement_rate = er
            author_trust[uname] = compute_trust_score(info)
            author_followers[uname] = info.followers_count
        else:
            author_trust[uname] = 0.4  # неизвестный — осторожно.
            author_followers[uname] = 0

    # 7) Для подозрительных — antifake.
    antifake_results: dict[str, float] = {}
    antifake_tasks = []
    antifake_ids = []
    for tw, _e in pairs:
        trust = author_trust.get(tw.author_username, 0.4)
        if needs_antifake_check(tw, trust):
            antifake_tasks.append(ai_client.antifake_check(tw.text))
            antifake_ids.append(tw.tweet_id)
    if antifake_tasks:
        results = await asyncio.gather(*antifake_tasks, return_exceptions=True)
        for tid, r in zip(antifake_ids, results):
            if isinstance(r, Exception):
                continue
            antifake_results[tid] = float(r.misleading)

    # 8) Определяем topic для каждого твита.
    topics: list[str] = []
    for _tw, e in pairs:
        name, _sim = await emb.nearest_cluster(e)
        topics.append(name)

    # 9) Сохраняем в SQLite + Chroma.
    saved = 0
    chroma_ids: list[str] = []
    chroma_embs: list[list[float]] = []
    chroma_metas: list[dict] = []

    async with session_scope() as s:
        for (tw, e), topic in zip(pairs, topics):
            trust = author_trust.get(tw.author_username, 0.4)
            row = Tweet(
                tweet_id=tw.tweet_id,
                author_username=tw.author_username,
                author_display_name=tw.author_display_name,
                text=tw.text,
                url=tw.url,
                image_url=tw.image_url,
                embedding=e,
                summary_ru=None,  # ленивое саммари при отправке
                source_trust_score=trust,
                misleading_score=antifake_results.get(tw.tweet_id),
                topic=topic,
                likes_count=tw.likes_count,
                retweets_count=tw.retweets_count,
                replies_count=tw.replies_count,
                created_at=tw.created_at.replace(tzinfo=None),
            )
            s.add(row)
            saved += 1

            chroma_ids.append(tw.tweet_id)
            chroma_embs.append(e)
            chroma_metas.append(
                {
                    "author": tw.author_username,
                    "topic": topic,
                    "trust": round(trust, 3),
                    "created_ts": int(tw.created_at.timestamp()),
                }
            )

    # Chroma — вне SQLite-транзакции, чтобы не блокировать писателей.
    try:
        upsert_tweets(chroma_ids, chroma_embs, chroma_metas)
    except Exception as e:
        log.warning("chroma upsert failed (non-fatal): %s", e)

    return saved


# ------------------------- 2. delivery_job ----------------------------


async def delivery_job(bot: Bot) -> None:
    """Для каждого готового к доставке пользователя — вызываем deliver_news_to_user."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    async with session_scope() as s:
        stmt = select(User).where(
            and_(
                User.onboarding_done.is_(True),
                User.paused.is_(False),
                or_(
                    User.last_delivered_at.is_(None),
                    # Условие "прошло ≥ interval часов" считаем на Python,
                    # т.к. smells like SQL dialect ловушка в SQLite.
                ),
            )
        )
        users = list((await s.execute(stmt)).scalars().all())

    due: list[int] = []
    for u in users:
        if u.last_delivered_at is None:
            due.append(u.telegram_id)
            continue
        delta = now - u.last_delivered_at
        if delta >= timedelta(hours=u.delivery_interval_hours):
            due.append(u.telegram_id)

    if not due:
        return

    log.info("delivery_job: %d users due", len(due))
    # Доставка параллельна, но с умеренной конкурентностью — чтобы Telegram нас не ограничил.
    sem = asyncio.Semaphore(5)

    async def _one(uid: int):
        async with sem:
            try:
                await deliver_news_to_user(bot, uid)
            except Exception as e:
                log.exception("delivery for %s failed: %s", uid, e)

    await asyncio.gather(*[_one(uid) for uid in due])


# ------------------------- 3. implicit_decay_job ----------------------------


async def implicit_decay_job() -> None:
    """Обрабатываем implicit skips — пользователю присылали, но он не реагировал.

    Важно: прогоняем ТОЛЬКО активных за 48h юзеров с хотя бы одним реальным
    лайком/дизлайком. Иначе джоба штрафовала спящих юзеров за 20 постов
    присланных в 3 ночи и полностью ломала preference_vector.
    """
    from core.recommender import apply_implicit_skip_decay

    now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    active_cutoff = now_naive - timedelta(hours=48)

    async with session_scope() as s:
        users = list(
            (
                await s.execute(
                    select(User).where(
                        User.onboarding_done.is_(True),
                        User.preference_vector.is_not(None),
                        User.last_delivered_at.is_not(None),
                        User.last_delivered_at >= active_cutoff,
                    )
                )
            ).scalars().all()
        )
        for u in users:
            has_real_fb = (await s.execute(
                select(Feedback.id)
                .where(
                    Feedback.user_id == u.telegram_id,
                    Feedback.liked.is_not(None),
                )
                .limit(1)
            )).scalar_one_or_none()
            if not has_real_fb:
                continue
            try:
                await apply_implicit_skip_decay(s, u, hours_without_reaction=24)
            except Exception as e:
                log.warning("implicit decay for %s failed: %s", u.telegram_id, e)


# ------------------------- 4. refresh_cluster_weights_job ----------------------------


async def refresh_cluster_weights_job() -> None:
    """Переpересчитывает cluster_weights из недавних позитивных реакций.

    cluster_weights ставятся на онбординге и точечно корректируются через
    "хочу больше X". Без явных запросов они протухают — юзер лайкает AI-посты
    месяц, а cluster_weights всё ещё отражают его стартовый список Following.
    Раз в неделю смешиваем 70% свежего сигнала (topics лайкнутых за 14 дней)
    с 30% старых весов.
    """
    from collections import Counter

    cutoff = datetime.now(timezone.utc) - timedelta(days=14)
    async with session_scope() as s:
        users = (await s.execute(
            select(User).where(User.onboarding_done.is_(True))
        )).scalars().all()
        for user in users:
            liked_rows = (await s.execute(
                select(Tweet.topic)
                .join(Feedback, Feedback.tweet_id == Tweet.tweet_id)
                .where(
                    Feedback.user_id == user.telegram_id,
                    Feedback.liked.is_(True),
                    Feedback.created_at >= cutoff.replace(tzinfo=None),
                )
            )).all()
            if not liked_rows:
                continue
            topic_counts = Counter(r[0] for r in liked_rows if r[0])
            if not topic_counts:
                continue
            total = sum(topic_counts.values())
            new_weights = {t: count / total for t, count in topic_counts.items()}
            old_weights = user.cluster_weights or {}
            merged: dict[str, float] = {}
            all_keys = set(new_weights) | set(old_weights)
            for k in all_keys:
                merged[k] = 0.7 * new_weights.get(k, 0.0) + 0.3 * old_weights.get(k, 0.0)
            user.cluster_weights = merged


# ------------------------- scheduler setup ----------------------------


def setup_scheduler(bot: Bot) -> AsyncIOScheduler:
    """Scheduler с двумя джобами.

    Раньше был ещё `fetch_tweets_job` который периодически скрапил твиты
    каждого author'а из FollowedAuthor — 64 автора * 1 запрос = 64 запроса
    каждые 30 мин, X быстро заtоттлил. После перехода на on-demand отчёт
    (`get_home_timeline` = 1 запрос на отчёт), fetch_tweets_job стал лишним
    и опасным (триггерил 429 с рекурсией внутри twikit). Поэтому выключен.
    """
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        delivery_job,
        "interval",
        minutes=20,
        args=[bot],
        id="delivery",
        coalesce=True,
        max_instances=1,
        next_run_time=datetime.now(timezone.utc) + timedelta(minutes=2),
    )
    scheduler.add_job(
        implicit_decay_job,
        "interval",
        hours=6,
        id="implicit_decay",
        coalesce=True,
        max_instances=1,
    )
    scheduler.add_job(
        refresh_cluster_weights_job,
        "interval",
        hours=168,
        id="cluster_refresh",
        coalesce=True,
        max_instances=1,
    )
    return scheduler
