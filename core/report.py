"""Отчёт по ленте пользователя: забираем его timeline → фильтруем → векторизуем
→ группируем в тематические кластеры → даём Claude называть каждый кластер.

Логика продукта:
  1. Что было у меня в ленте за последние N часов?
  2. Что из этого мусор (реклама/RT/спам/дубли/уже виденное)?
  3. Что осталось — разбиваем на темы через cosine-кластеризацию
     (union-find по пороговой близости embedding'ов).
  4. Claude подписывает каждый кластер: эмодзи + короткое имя + summary
     («Grok Stories»-стиль — о чём идёт дискуссия).
  5. Пользователь видит отчёт и инлайн-кнопки по темам; клик по кнопке
     раскрывает summary + топ-N постов этой темы.

Важно: кластеры авто-генерируются под пользователя, а не из хардкода
config.topic_clusters — те нужны только для source_trust_score сборки.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from core import ai_client
from core import embeddings as emb
from core.filters import hype_score, is_low_signal
from core.recommender import freshness_score
from core.x_parser import RawTweet, parser
from db.models import Feedback, SentNews, Tweet, User

log = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[str], Awaitable[None]]]


# ----------------------- data classes -----------------------


@dataclass
class ReportCluster:
    """Одна тема в отчёте."""
    id: int
    emoji: str
    name: str
    tweet_ids: list[str]
    summary: Optional[str] = None  # Grok Stories-style — лениво через Claude


@dataclass
class SuperTopic:
    """Супер-категория — группирует несколько sub-cluster ID."""
    emoji: str
    name: str
    sub_ids: list[int]  # ids of ReportCluster


@dataclass
class Report:
    user_id: int
    generated_at: datetime
    window_hours: float
    fetched: int           # сырых твитов от X
    filtered_trash: int    # выкинули как мусор (реклама/RT/дубли/короткое)
    filtered_hype: int     # выкинули как хайп/вода/реакции
    already_seen: int      # уже присылали
    kept: int              # прошли в кластеризацию
    clusters: list[ReportCluster]
    unclustered_ids: list[str] = field(default_factory=list)
    super_topics: list[SuperTopic] = field(default_factory=list)

    def total_in_topics(self) -> int:
        return sum(len(c.tweet_ids) for c in self.clusters)

    def posts_in_super(self, sid_list: list[int]) -> int:
        by_id = {c.id: c for c in self.clusters}
        return sum(len(by_id[i].tweet_ids) for i in sid_list if i in by_id)


# ----------------------- clustering -----------------------


def _cosine_union_find(
    vectors: list[list[float]],
    threshold: float = 0.55,
) -> list[list[int]]:
    """Union-find по cosine-похожести. Возвращает список кластеров (список индексов)."""
    n = len(vectors)
    if n == 0:
        return []
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    normalized = arr / np.clip(norms, 1e-9, None)
    sim = normalized @ normalized.T

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # i<j чтобы не дублировать.
    for i in range(n):
        # векторизованно: находим соседей i выше порога
        row = sim[i, i + 1 :]
        hits = np.where(row > threshold)[0]
        for offset in hits:
            union(i, i + 1 + int(offset))

    by_root: dict[int, list[int]] = {}
    for i in range(n):
        by_root.setdefault(find(i), []).append(i)
    return sorted(by_root.values(), key=len, reverse=True)


# ----------------------- helpers -----------------------


async def _derive_interest_queries(
    session: AsyncSession, user: User, max_queries: int = 3,
) -> list[str]:
    """Поисковые запросы под этого юзера. Составляем через Claude из сигналов
    (cluster_weights, saved_search_queries, подписки) — не по хардкод-маппингу.
    Это делает запросы индивидуальными: «японский стиль» + подписка на
    voguejapan превращается в «japanese streetwear Tokyo», не в generic
    «fashion trends».
    """
    from db.models import FollowedAuthor

    payload = user.onboarding_payload or {}
    saved = payload.get("saved_search_queries") if isinstance(payload, dict) else None
    saved_list = [q for q in (saved or []) if isinstance(q, str) and q.strip()]

    weights = dict(user.cluster_weights or {})

    # Только реально-активные кластеры (>=0.2). Меньше шума для Claude.
    active = {k: round(v, 2) for k, v in weights.items() if v >= 0.2}

    rows = (await session.execute(
        select(FollowedAuthor.author_username).where(
            FollowedAuthor.user_id == user.telegram_id
        )
    )).all()
    following_names = [r[0] for r in rows]

    # Быстрый путь: если активных кластеров/сигналов вообще нет — возвращаем
    # сохранённые запросы юзера (или пусто, если и их нет). Без лишнего вызова Claude.
    if not active and not saved_list:
        return saved_list[:max_queries]

    # Юзер жаловался: «мода — мусор собрал, Tiffany Blue Book гала вместо стритвира».
    # Причина: Claude обобщал его реплики («хочу больше про streetwear») в
    # generic «fashion week» / «haute couture trends» — X-search 404ит на
    # длинных фразах, authors-fallback тянул @voguemagazine с гала-хрониками.
    #
    # Фикс: БЕРЁМ saved_search_queries буквально. Эти фразы уже короткие
    # (2-3 слова) и конкретные — юзер сам формулировал.
    #
    # Свежесть > разнообразия. Раньше использовали random.shuffle по всей
    # истории (до 10 запросов). Итог: если юзер 2 недели назад писал «graphic
    # design», а сегодня «одежда» — shuffle мог поднять graphic design и
    # лента приходила с UI-компонентами вместо fashion. Берём ПОСЛЕДНИЕ 3
    # по порядку сохранения — они отражают текущий интерес юзера.
    from core import ai_client as _ai

    priority = [_ai._shorten_query(q) for q in saved_list[-3:]]
    priority = [q for q in priority if q]  # выкинем пустые после _shorten_query

    # Claude генерит ещё одну СВЕЖУЮ фразу (чтобы лента не была полностью
    # предсказуемой). Это bonus-query, не замена saved.
    bonus: list[str] = []
    try:
        bonus = await _ai.suggest_interest_queries(
            cluster_weights=active,
            saved_queries=saved_list,
            followed_authors=following_names,
            max_queries=max(1, max_queries - len(priority)),
        )
    except Exception as e:
        log.debug("bonus interest query failed (ok): %s", e)

    # Дедуп и лимит.
    out: list[str] = []
    seen = set()
    for q in priority + bonus:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(q)
        if len(out) >= max_queries:
            break
    log.info("interest queries: priority=%s bonus=%s → %s", priority, bonus, out)
    return out


async def _existing_sent_ids(session: AsyncSession, user_id: int) -> set[str]:
    rows = (
        await session.execute(select(SentNews.tweet_id).where(SentNews.user_id == user_id))
    ).all()
    return {r[0] for r in rows}


async def _upsert_raw_tweets(
    session: AsyncSession,
    pairs: list[tuple[RawTweet, list[float]]],
) -> dict[str, Tweet]:
    """Сохраняет RawTweet+embedding в БД (если ещё нет) и возвращает {id: Tweet}."""
    from core.embeddings import nearest_cluster

    ids = [rt.tweet_id for rt, _ in pairs]
    existing_rows = (
        await session.execute(select(Tweet).where(Tweet.tweet_id.in_(ids)))
    ).all()
    existing: dict[str, Tweet] = {row[0].tweet_id: row[0] for row in existing_rows}

    for rt, e in pairs:
        if rt.tweet_id in existing:
            continue
        topic, _sim = await nearest_cluster(e) if e else (None, 0.0)
        row = Tweet(
            tweet_id=rt.tweet_id,
            author_username=rt.author_username,
            author_display_name=rt.author_display_name,
            text=rt.text,
            url=rt.url,
            image_url=rt.image_url,
            media_type=rt.media_type,
            quote_image_url=rt.quote_image_url,
            quote_media_type=rt.quote_media_type,
            quote_author=rt.quote_author,
            quote_text=(rt.quote_text or "")[:1000] or None,
            linked_tweet_id=rt.linked_tweet_id,
            embedding=e,
            source_trust_score=0.6,
            topic=topic,
            likes_count=rt.likes_count,
            retweets_count=rt.retweets_count,
            replies_count=rt.replies_count,
            created_at=rt.created_at.replace(tzinfo=None),
        )
        session.add(row)
        existing[rt.tweet_id] = row

    await session.flush()
    return existing


def _sample_diverse_ids(ids: list[str], n: int = 10) -> list[str]:
    """Representative sample из кластера: голова (топ по score) + середина + хвост.

    Вход: `ids` уже отсортированы в ReportCluster через _score_for_ordering
    (свежесть + relevance + engagement + медиа). Берём 40% из начала, 30% из
    середины, 30% из конца. Для кластера меньше n — возвращаем всё.
    """
    if not ids:
        return []
    if len(ids) <= n:
        return list(ids)
    head_n = max(1, int(round(n * 0.4)))
    mid_n = max(1, int(round(n * 0.3)))
    tail_n = max(1, n - head_n - mid_n)
    head = ids[:head_n]
    tail = ids[-tail_n:]
    mid_start = len(ids) // 2 - mid_n // 2
    mid = ids[mid_start:mid_start + mid_n]
    # Дедуп на случай перекрытия в маленьких кластерах.
    seen: set[str] = set()
    out: list[str] = []
    for tid in head + mid + tail:
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out[:n]


# Маркеры которые значат «haiku честно написал что посты не по заявленной теме».
# Если такое есть в summary — имя кластера плохое, лучше саммари вообще не показывать
# (пусть юзер увидит только пост-пагинатор, он сам прочтёт).
_SUMMARY_OFF_TOPIC_MARKERS = (
    "посты не про",
    "посты не содержат",
    "не про релиз",
    "не про сам",
    "не про тему",
    "не касаются темы",
)


def _accept_summary(text: str | None) -> str | None:
    """Возвращает text если выжимка валидна, иначе None.

    Критерии:
      - ≥40 символов
      - не начинается с «посты не про …» (это сигнал что имя кластера неверное,
        и саммари честно это отражает — но пользователю такая выжимка бесполезна)
      - не состоит из одной шаблонной фразы
    """
    if not text:
        return None
    t = text.strip()
    if len(t) < 40:
        return None
    low = t.lower()
    # Явный off-topic: haiku сообщает «посты не про релиз» — скрываем summary
    # полностью, т.к. имя темы уже неверно (см. _render_topic_open: пользователь
    # увидит кликабельные посты без misleading саммари).
    for bad in _SUMMARY_OFF_TOPIC_MARKERS:
        if bad in low[:80]:
            return None
    return t


def _score_for_ordering(tweet: Tweet, user_vec: Optional[list[float]]) -> float:
    """Быстрое ранжирование внутри кластера — свежесть + relevance + engagement + медиа."""
    base = freshness_score(tweet.created_at) * 0.4
    if user_vec and tweet.embedding:
        a = np.asarray(user_vec, dtype=np.float32)
        b = np.asarray(tweet.embedding, dtype=np.float32)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na > 1e-9 and nb > 1e-9:
            cos01 = max(0.0, (float(np.dot(a, b) / (na * nb)) + 1) / 2)
            base += cos01 * 0.3
    base += min(0.15, (tweet.likes_count or 0) / 50_000.0)
    # MEDIA BOOST — пост с фото/видео визуально сильнее вовлекает, и юзер
    # прямо просил чтобы «везде были фотки». Поднимаем такие посты в топ кластера.
    if tweet.image_url:
        base += 0.25
        if tweet.media_type in ("video", "animation"):
            base += 0.1  # видео — ещё чуть выше чем фото
    return base


# ----------------------- main entrypoint -----------------------


async def build_report(
    session: AsyncSession,
    user: User,
    *,
    window_hours: float = 1.0,
    limit_raw: int = 120,
    progress: ProgressCb = None,
    demo_pool_fallback: bool = True,
    source: str = "for_you",  # 'for_you' → X algorithmic; 'following' → chronological
    auto_summarize_top: int = 0,  # >0: параллельно саммаризуем топ-N кластеров upfront
) -> Report:
    """Основной flow — от X-timeline до готовых кластеров.

    progress(msg) — опциональный коллбэк для апдейта прогресса в чате
    (обычно edit_text одного статус-сообщения).
    """
    t0 = time.perf_counter()
    user_id = user.telegram_id

    async def _notify(msg: str) -> None:
        if progress:
            try:
                await progress(msg)
            except Exception as e:
                log.debug("progress cb failed: %s", e)

    # ---------- 1. Fetch raw timeline ----------
    await _notify("⏳ Смотрю что обсуждают в X…" if source == "for_you"
                  else "⏳ Забираю твою Following-ленту…")
    raw: list[RawTweet] = []
    try:
        if source == "for_you":
            raw = await parser.get_for_you_timeline(limit=limit_raw)
        else:
            raw = await parser.get_home_timeline(limit=limit_raw)
    except Exception as e:
        log.warning("timeline fetch failed (%s): %s", source, e)

    # ---------- 1a. Bot-tracked authors injection (для ОБОИХ source) ----------
    # «Моя лента» = X-Following хроника, но юзер жалуется: «попросил больше
    # моды, добавил GQ/Esquire — а в ленте их нет». Логично: он на GQ в X сам
    # не подписан, бот добавил их только в свою БД-таблицу FollowedAuthor.
    # Подмешиваем последние посты bot-tracked авторов в обе ветки.
    #
    # Сниженный per-author limit (было 6 → 3). При 10 bot-tracked авторах и
    # 6 постах/автора в raw попадало 55-60 постов, и after per-author-cap=3
    # реальная Following-лента юзера (~79 постов из подписок X) вытеснялась —
    # AI/IT темы из настоящих подписок не образовывали кластеры. 3 поста/автор
    # даёт ≤30 кандидатов bot-tracked, и у Following остаётся место.
    # Bot-tracked injection — ограничиваем топ-5 самых недавно добавленных
    # авторов × 2 поста. Было 10×3=30 — при 97 FollowedAuthor (накопленных
    # через годы «хочу больше X») это заливало 30 постов от истории юзера,
    # вытесняя свежую X-ленту. 5×2=10 — лёгкая подмешка от того что юзер
    # недавно добавил, а не от всего списка.
    from db.models import FollowedAuthor as _FA
    fa_rows = (await session.execute(
        select(_FA.author_username)
        .where(_FA.user_id == user.telegram_id)
        .order_by(desc(_FA.added_at))
        .limit(5)
    )).all()
    bot_tracked = [r[0] for r in fa_rows if r[0]]
    bot_tracked_added = 0
    if bot_tracked:
        await _notify(f"📥 Проверяю {len(bot_tracked)} подобранных каналов…")
        try:
            tracked_tweets = await parser.get_recent_tweets_for_authors(
                bot_tracked, limit_per_author=2,
            )
            already_ids = {t.tweet_id for t in raw}
            for t in tracked_tweets:
                if t.tweet_id not in already_ids:
                    raw.append(t)
                    already_ids.add(t.tweet_id)
                    bot_tracked_added += 1
            log.info("injected %d posts from %d bot-tracked authors (source=%s)",
                     bot_tracked_added, len(bot_tracked), source)
            if bot_tracked_added:
                await _notify(
                    f"📥 Подмешал {bot_tracked_added} постов от {len(bot_tracked)} каналов…"
                )
        except Exception as e:
            log.debug("bot-tracked author injection failed: %s", e)

    # ---------- 1b. Interest-driven search ОТКЛЮЧЁН ----------
    # Раньше здесь X-search по saved_queries юзера подмешивал до 450 постов
    # по «fashion» и затоплял реальную For You ленту. Принципиально удалено:
    # топик-бустинг теперь только через explicit FollowedAuthor (который юзер
    # добавил через «хочу больше X»), никаких X-search инъекций.
    # Сохраняем fallback: когда For You вернула совсем пусто (<20 постов),
    # делаем 1 поиск чтобы отчёт не был вообще пустым.
    if source == "for_you" and len(raw) < 20:
        # Минимальный fallback когда For You вернула пусто: 1 search по самому
        # последнему интересу юзера, чтобы отчёт не был вообще пустым.
        # Не топит ленту, т.к. сюда попадаем только при len(raw)<20.
        interest_queries = await _derive_interest_queries(session, user, max_queries=1)
        if interest_queries:
            q = interest_queries[0]
            await _notify(f"🔎 For You пуста, беру по теме «{q}»…")
            already = {t.tweet_id for t in raw}
            try:
                extra = await parser.search_tweets(q, product="Top", count=25)
                for t in extra:
                    if t.tweet_id not in already:
                        raw.append(t)
                        already.add(t.tweet_id)
            except Exception as e:
                log.debug("fallback search(%s) failed: %s", q, e)

    # Для Following — режем по времени (окно). Для For You — нет: X-алгоритм
    # ранжирует НЕ по времени и часто поднимает посты 1-3 дня давности если
    # по ним идёт активная дискуссия сейчас. Отрезать по времени = потерять
    # самые «обсуждаемые» карточки (и есть риск получить 0 тем, как у юзера).
    if source == "following":
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours * 2)
        raw = [
            t for t in raw
            if (t.created_at if t.created_at.tzinfo else t.created_at.replace(tzinfo=timezone.utc)) >= cutoff
        ]
        # Per-author dedup: в Following-фетче подписочные аккаунты-новостники
        # (Reuters/TheEconomist/AP) генерят десятки постов в окне — без cap
        # они занимают 75%+ ленты и юзеру кажется что моя-лента =
        # «Reuters спам». Замер: Reuters 47/80 постов. С cap=3 — лента
        # становится разнообразной, остальные подписки просвечивают.
        # Cap был 5 — из 400 постов 33 уникальных автора давали только 93 поста
        # в кластеризацию, многие темы разваливались. 8 — Reuters/TheEconomist
        # естественно образуют «News» кластер из 8 постов (легитно если юзер
        # подписан), остальные подписки тоже больше просвечивают.
        MAX_PER_AUTHOR = 8
        seen_per_author: dict[str, int] = {}
        capped: list[RawTweet] = []
        # Сортируем по engagement сначала — топовые посты автора важнее
        # хвоста (5-й/6-й Reuters пост за день — мусор).
        raw_sorted = sorted(raw, key=lambda t: t.likes_count + 2 * t.retweets_count, reverse=True)
        for t in raw_sorted:
            key = (t.author_username or "").lower()
            cnt = seen_per_author.get(key, 0)
            if cnt >= MAX_PER_AUTHOR:
                continue
            seen_per_author[key] = cnt + 1
            capped.append(t)
        log.info("following: per-author cap %d → kept %d of %d (unique authors=%d)",
                 MAX_PER_AUTHOR, len(capped), len(raw), len(seen_per_author))
        raw = capped
    fetched = len(raw)

    # Demo fallback — если X вернул пусто.
    if not raw and demo_pool_fallback:
        await _notify("X пока не отдаёт свежее — беру из демо-арсенала…")
        rows = (
            await session.execute(
                select(Tweet).where(Tweet.tweet_id.like("demo_%")).limit(50)
            )
        ).all()
        demo_tweets = [r[0] for r in rows]
        # Собираем сразу как ORM Tweet → минуем фильтры/embedding шаги.
        clusters_result = await _cluster_and_name(
            session, demo_tweets, user, progress=_notify
        )
        return Report(
            user_id=user_id,
            generated_at=datetime.now(timezone.utc),
            window_hours=window_hours,
            fetched=len(demo_tweets),
            filtered_trash=0,
            filtered_hype=0,
            already_seen=0,
            kept=len(demo_tweets),
            clusters=clusters_result[0],
            unclustered_ids=clusters_result[1],
        )

    # ---------- 2. Trash + low-signal filter + blocked_authors ----------
    await _notify(f"📥 Забрал {fetched} постов. Чищу шум…")
    blocked_lower = {str(b).lower() for b in (user.blocked_authors or [])}
    filtered: list[RawTweet] = []
    trash_n = 0
    hype_n = 0
    for t in raw:
        if blocked_lower and (t.author_username or "").lower() in blocked_lower:
            trash_n += 1
            continue
        low, reason = is_low_signal(t, hype_threshold=0.5)
        if low:
            if reason.startswith("hype:"):
                hype_n += 1
            else:
                trash_n += 1
            continue
        filtered.append(t)
    # Обнуляем trash_n для отчёта на «отсеяно как мусор» — пользователь увидит обе строчки.
    # Оставим total_filtered = trash + hype; при рендере покажем разбивку.

    # ---------- 3. Exclude already sent ----------
    sent_ids = await _existing_sent_ids(session, user_id)
    pre_dedup = len(filtered)
    filtered = [t for t in filtered if t.tweet_id not in sent_ids]
    already_seen = pre_dedup - len(filtered)

    if not filtered:
        return Report(
            user_id=user_id,
            generated_at=datetime.now(timezone.utc),
            window_hours=window_hours,
            fetched=fetched, filtered_trash=trash_n, filtered_hype=hype_n,
            already_seen=already_seen, kept=0, clusters=[],
        )

    # ---------- 4. Embed ----------
    await _notify(f"🧠 Векторизую {len(filtered)}…")
    texts = [t.text for t in filtered]
    embs = await emb.embed_batch(texts)
    pairs = [(rt, e) for rt, e in zip(filtered, embs) if e]

    # ---------- 5. Upsert to DB ----------
    tweet_map = await _upsert_raw_tweets(session, pairs)
    tweets: list[Tweet] = [tweet_map[rt.tweet_id] for rt, _ in pairs if rt.tweet_id in tweet_map]

    # ---------- 5b. Pending boost tweets (от «хочу больше X» immediate-fetch) ----------
    # preferences.py сохраняет tweet_ids подтянутых постов в
    # user.onboarding_payload["pending_boost_ids"]. Подмешиваем ограниченное
    # количество (MAX_PENDING=20) чтобы не затопить реальную X-ленту:
    # 76 fashion-постов vs 100 For You делали fashion доминантой. 20 — это
    # supplement, реальная лента остаётся основой.
    MAX_PENDING = 20
    _payload = dict(user.onboarding_payload or {})
    pending_boost_ids = list(_payload.get("pending_boost_ids") or [])
    if pending_boost_ids:
        existing_tids = {t.tweet_id for t in tweets}
        fresh_ids = [
            tid for tid in pending_boost_ids
            if tid not in existing_tids and tid not in sent_ids
        ]
        boost_added = 0
        if fresh_ids:
            # Берём только ПЕРВЫЕ MAX_PENDING — это самые свежие upsert'ы
            # (preferences.py кладёт новые в начало списка).
            rows = (await session.execute(
                select(Tweet).where(Tweet.tweet_id.in_(fresh_ids[:MAX_PENDING * 2]))
            )).all()
            # Сохраняем порядок по свежести
            by_id = {r[0].tweet_id: r[0] for r in rows}
            for tid in fresh_ids:
                if boost_added >= MAX_PENDING:
                    break
                t = by_id.get(tid)
                if t and t.embedding:
                    tweets.append(t)
                    boost_added += 1
        log.info("pending boost: %d ids → %d added (cap=%d)",
                 len(pending_boost_ids), boost_added, MAX_PENDING)
        # Чистим pending после использования.
        _payload["pending_boost_ids"] = []
        user.onboarding_payload = _payload
        await session.flush()

    # ---------- 6. Cluster ----------
    # Для Following сразу стартуем с min_cluster_size=2: подписочная лента
    # тоньше (100 постов × per-author cap=3), темы там естественно мельче.
    # Раньше два прохода (3 → retry 2) давали те же результаты но в 2x время.
    # Для For You оставляем 3 как первый проход — там контента в 3-5x больше,
    # size=3 даёт более чистые темы, без шума.
    #
    # Следовая «сила boost» тоже разная: Following должен уважать реальные
    # подписки юзера (его AI-подписки не должны задавливаться saved_queries=
    # [fashion,...]). Для for_you boost×4 — агрессивный, для following ×1.5
    # — мягкий, чтобы размер кластера (сколько подписок вообще про тему
    # пишут) доминировал.
    await _notify(f"🗂 Группирую в темы…")
    # Boost_multiplier=0 — отключаем re-ranking по saved_queries. Кластеры
    # выигрывают размером: сколько постов на тему в реальной X-ленте, столько
    # и показываем. Без этого юзер жалуется: «3-постный fashion затоптал
    # 12-постный AI». Работает одинаково для любого юзера — код не знает
    # какие у него интересы, ленте виднее.
    if source == "following":
        clusters, unclustered_ids = await _cluster_and_name(
            session, tweets, user, progress=_notify,
            min_cluster_size=2, boost_multiplier=0.0,
        )
    else:
        clusters, unclustered_ids = await _cluster_and_name(
            session, tweets, user, progress=_notify,
            boost_multiplier=0.0,
        )

    # Adaptive retry: если получилось <5 тем (то что юзер называет «анемичный
    # отчёт — всего 2 темы»), снижаем min_cluster_size до 2 и пересчитываем.
    # Пары связанных постов (2 штуки) образуют темы — отчёт становится
    # пышнее. С cap_per_cluster=8 это безопасно: нельзя получить «тему из 50».
    # Для Following уже стартовали с 2 — retry не нужен.
    MIN_TARGET_TOPICS = 5
    if (source != "following"
            and len(clusters) < MIN_TARGET_TOPICS and len(tweets) >= 6):
        await _notify(f"🔬 Тем мало ({len(clusters)}), дроблю мелкие группы…")
        clusters, unclustered_ids = await _cluster_and_name(
            session, tweets, user, progress=_notify, min_cluster_size=2,
        )
        log.info("adaptive retry with min_cluster_size=2: got %d topics", len(clusters))

    # ---------- 6a. Super-topic grouping ОТКЛЮЧЕНО ----------
    # Claude группировал «Bitcoin + XRP + Jensen Huang» в «IT/Технологии» —
    # юзер видел абстрактные ярлыки и думал что бот пропустил его реальные
    # темы. Показываем плоский список (реальные имена кластеров).
    super_topics: list[SuperTopic] = []

    # ---------- 6b. Upfront summaries (digest mode) ----------
    # Сразу саммаризуем топ-N кластеров параллельно, чтобы overview показывал не
    # только имя темы, но и 2-3 фразы о чём пост. 1 вызов Haiku на кластер →
    # 7 кластеров = ~5-8с параллельно.
    if auto_summarize_top > 0 and clusters:
        await _notify(f"📝 Делаю выжимку по темам…")
        top_for_summary = clusters[:auto_summarize_top]
        # Diversity sampling: для каждой темы берём посты из начала (топ по score),
        # середины (engagement-нейтральные) и конца (свежие/хвост) — это даёт
        # representative sample, а не 10 подряд top-liked с одним ракурсом.
        id_to_tweet: dict[str, Tweet] = {}
        needed_ids: set[str] = set()
        for cl in top_for_summary:
            needed_ids.update(_sample_diverse_ids(cl.tweet_ids, n=10))
        if needed_ids:
            rows = (await session.execute(
                select(Tweet).where(Tweet.tweet_id.in_(list(needed_ids)))
            )).all()
            id_to_tweet = {r[0].tweet_id: r[0] for r in rows}

        async def _one_summary(cl: ReportCluster) -> None:
            sample_ids = _sample_diverse_ids(cl.tweet_ids, n=10)
            sample = [id_to_tweet[i].text for i in sample_ids if i in id_to_tweet]
            if not sample:
                return
            try:
                summary = await ai_client.summarize_discussion(sample, cl.name)
                # Валидация качества: короткие/пустые саммари бесполезны — «⏳ Codex
                # — сроки и ограничения · 2 поста» с шаблонным текстом юзер
                # жаловался именно про это. Лучше показать placeholder чем мусор.
                cl.summary = _accept_summary(summary)
            except Exception as e:
                log.debug("upfront summary for %s failed: %s", cl.name, e)

        await asyncio.gather(*[_one_summary(cl) for cl in top_for_summary])

    elapsed = time.perf_counter() - t0
    log.info(
        "report user=%s done in %.1fs: fetched=%d trash=%d seen=%d kept=%d topics=%d",
        user_id, elapsed, fetched, trash_n, already_seen, len(tweets), len(clusters),
    )

    return Report(
        user_id=user_id,
        generated_at=datetime.now(timezone.utc),
        window_hours=window_hours,
        fetched=fetched,
        filtered_trash=trash_n,
        filtered_hype=hype_n,
        already_seen=already_seen,
        kept=len(tweets),
        clusters=clusters,
        unclustered_ids=unclustered_ids,
        super_topics=super_topics,
    )


async def _cluster_and_name(
    session: AsyncSession,
    tweets: list[Tweet],
    user: User,
    *,
    progress: ProgressCb = None,
    max_topics: int = 12,
    min_cluster_size: int = 3,
    cap_per_cluster: int = 8,
    boost_multiplier: float = 4.0,
) -> tuple[list[ReportCluster], list[str]]:
    """Кластеризация tweets + Claude-именование + merge похожих.

    Двухстадийная группировка:
      1. Cosine union-find с порогом 0.42 — ловим смысловые группы
         даже если формулировки разные. Ниже 0.42 начинает склеивать
         не связанные темы; выше 0.48 — разбивает один сюжет на 3 кластера.
      2. После Claude-именования: эмбеддим сами НАЗВАНИЯ тем, объединяем
         кластеры чьи имена семантически похожи (cosine > 0.62) — более
         агрессивно мёржим, чтобы Claude-вариации одной темы не плодили
         2-3 мелких кластера («Opus 4.7 релиз», «Anthropic выпустил 4.7»,
         «Новый Claude» — всё это одна тема).
    """
    if not tweets:
        return [], []

    vectors = [list(t.embedding) for t in tweets if t.embedding]
    if len(vectors) < 2:
        return [], [t.tweet_id for t in tweets]

    import asyncio

    # --- Stage 1: raw clustering, LOW threshold — широкая группировка ---
    clusters_idx = _cosine_union_find(vectors, threshold=0.42)
    user_vec = user.preference_vector

    large: list[list[int]] = [c for c in clusters_idx if len(c) >= min_cluster_size]
    small: list[list[int]] = [c for c in clusters_idx if len(c) < min_cluster_size]
    unclustered_ids = [tweets[i].tweet_id for c in small for i in c]

    # Не ограничиваем заранее — мерж может сократить число тем.
    groups: list[list[int]] = list(large)

    if not groups:
        return [], unclustered_ids

    # --- Stage 2: Claude дает эмодзи+имя каждой группе ---
    async def _name_one(idx_list: list[int]) -> tuple[str, str]:
        sample = [tweets[i].text for i in idx_list[:6]]
        try:
            return await ai_client.name_topic(sample)
        except Exception as e:
            log.warning("name_topic failed: %s", e)
            return ("📰", "Разное")

    named = await asyncio.gather(*[_name_one(grp) for grp in groups])

    # --- Stage 3: мерж групп с похожими названиями ---
    # Embed самих строк-названий, находим пары похожих, сливаем union-find'ом.
    names_to_embed = [n for (_e, n) in named]
    name_embs = await emb.embed_batch(names_to_embed)
    valid_name_idx = [i for i, e in enumerate(name_embs) if e]
    if len(valid_name_idx) >= 2:
        merge_clusters = _cosine_union_find(
            [name_embs[i] for i in valid_name_idx], threshold=0.62
        )
        # Мерж: для каждого кластера-по-именам объединяем исходные группы
        merged_groups: list[list[int]] = []
        merged_names: list[tuple[str, str]] = []
        assigned = set()
        for name_cluster in merge_clusters:
            orig_idxs = [valid_name_idx[i] for i in name_cluster]
            combined = []
            for oi in orig_idxs:
                combined.extend(groups[oi])
                assigned.add(oi)
            # Имя берём от самой крупной под-группы (больше всего твитов).
            best_oi = max(orig_idxs, key=lambda oi: len(groups[oi]))
            merged_groups.append(combined)
            merged_names.append(named[best_oi])
        # Добавляем группы чьи имена эмбеддинга не получили (fail-safe).
        for i, grp in enumerate(groups):
            if i not in assigned:
                merged_groups.append(grp)
                merged_names.append(named[i])
        groups = merged_groups
        named = merged_names

    # --- Stage 3b: dislike-dampening — убираем темы на которые юзер реагировал 👎 ---
    # Мягче чем раньше: 5+ дизлайков (было 3) в пределах 6 часов (было 12).
    # Юзер жаловался что получает 2-3 темы вместо 10 — не хотим вырезать
    # темы за один случайный 👎, а ждать устойчивого сигнала.
    disliked = user.topic_dislikes or {}
    if disliked:
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        active_dislikes: set[str] = set()
        for name_key, entry in disliked.items():
            try:
                n = int((entry or {}).get("dislikes", 0))
                ts = (entry or {}).get("last_ts")
                if ts:
                    ts_dt = datetime.fromisoformat(ts)
                    if ts_dt.tzinfo is None:
                        ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                    if now - ts_dt < timedelta(hours=6) and n >= 5:
                        active_dislikes.add(name_key.lower().strip())
            except Exception:
                continue
        if active_dislikes:
            surviving_groups, surviving_named = [], []
            for grp, nm in zip(groups, named):
                if nm[1].lower().strip() in active_dislikes:
                    log.info("dampening cluster '%s' — user dislikes recently", nm[1])
                    continue
                surviving_groups.append(grp)
                surviving_named.append(nm)
            groups, named = surviving_groups, surviving_named

    # --- Stage 4: ranking + per-cluster cap, weighted by user interest ---
    #
    # Проблема старого подхода (sorted by len): модный кластер из 5 проигрывал
    # AI-кластеру из 20, даже когда юзер явно сказал «больше моды». Cap=8
    # одинаковый — юзер видел 8 модных постов, хотя бот вытянул 40.
    #
    # Старый topic_clusters-based boost тоже не работал: nearest_cluster()
    # ставил «lifestyle» на «уличную еду» и «culture» на «UI компоненты»
    # потому что embeddings схожи — boost получали не те темы.
    #
    # Теперь:
    #   1. Берём явные интересы юзера (saved_search_queries — что он сам
    #      печатал в «хочу больше …»).
    #   2. Эмбеддим эти запросы.
    #   3. Для каждого кластера считаем cosine sim его НАЗВАНИЯ к queries.
    #      Топ-1 sim = boost. Это бьёт точно: «уличная мода снимки» близко
    #      к «street style», но «уличная еда» — нет.
    #   4. ranking_score = (size + 5) × (1 + boost × 4) — формула размывает
    #      разрыв 2 vs 8 постов и даёт boost реально решать.
    #   5. cap_per_cluster: 8 → 18 для clusters с sim > 0.4.
    payload = user.onboarding_payload or {}
    saved_queries: list[str] = []
    if isinstance(payload, dict):
        sq = payload.get("saved_search_queries") or []
        saved_queries = [q for q in sq if isinstance(q, str) and q.strip()][-10:]

    # Interest-ranking через Claude: вызов только если boost_multiplier>0.
    # Когда мультипликатор = 0, результаты никак не влияют на порядок —
    # экономим Claude-вызов и пол-секунды времени.
    cluster_score: list[float] = [0.0] * len(groups)
    if boost_multiplier > 0 and saved_queries and groups:
        try:
            cluster_score = await ai_client.score_clusters_against_interests(
                [n for (_e, n) in named], saved_queries,
            )
        except Exception as ex:
            log.debug("score_clusters_against_interests failed: %s", ex)

    cluster_meta = []
    for i, grp in enumerate(groups):
        boost = cluster_score[i] if i < len(cluster_score) else 0.0
        # `min(size, 12)` ограничивает влияние гигантских кластеров (Claude
        # Design × 50) — без cap'а размер сжирает любой boost. +5 размывает
        # разрыв 2 vs 8 постов. boost_multiplier: для For You=4.0 (агрессивный
        # boost — saved-queries доминируют), для Following=1.5 (мягкий —
        # размер кластера из подписок важнее заявленных интересов, иначе
        # AI-подписки юзера задавливаются когда он поставил «хочу больше моды»).
        weighted = (min(len(grp), 12) + 5) * (1.0 + boost * boost_multiplier)
        cluster_meta.append((i, boost, weighted))

    cluster_meta.sort(key=lambda x: x[2], reverse=True)
    order = [m[0] for m in cluster_meta]
    boost_by_orig = {m[0]: m[1] for m in cluster_meta}
    if saved_queries and any(b > 0 for _, b, _ in cluster_meta):
        top_dbg = [(named[i][1], round(boost_by_orig[i], 2)) for i in order[:6]]
        log.info("interest-ranked clusters (saved=%s): %s", saved_queries[:3], top_dbg)

    groups = [groups[i] for i in order]
    named = [named[i] for i in order]
    boosts = [boost_by_orig[i] for i in order]

    top_groups = groups[:max_topics]
    top_named = named[:max_topics]
    top_boosts = boosts[:max_topics]
    rest = [i for grp in groups[max_topics:] for i in grp]
    if len(rest) >= min_cluster_size:
        top_groups.append(rest)
        top_named.append(("📰", "Остальное"))
        top_boosts.append(0.0)

    out: list[ReportCluster] = []
    for cid, (grp, (emoji, name), boost) in enumerate(zip(top_groups, top_named, top_boosts)):
        # Boosted-кластер: cap 8→18, юзер видит реально много контента темы
        # которую он сам попросил. Порог 0.5 — Claude должен явно отметить
        # тему как релевантную (не на полпути).
        effective_cap = cap_per_cluster + (10 if boost >= 0.5 else 0)
        sorted_idx = sorted(grp, key=lambda i: _score_for_ordering(tweets[i], user_vec), reverse=True)
        capped = sorted_idx[:effective_cap] if effective_cap > 0 else sorted_idx
        out.append(ReportCluster(
            id=cid,
            emoji=emoji,
            name=name,
            tweet_ids=[tweets[i].tweet_id for i in capped],
        ))

    log.info("clustered %d tweets → %d topics (cap=%d/cluster, after name-merge)",
             len(tweets), len(out), cap_per_cluster)
    return out, unclustered_ids


# ----------------------- report cache -----------------------
#
# Отчёт живёт в памяти процесса. Хранить в БД отдельной таблицей было бы чище,
# но отчёт — эфемерный артефакт: ему и не надо переживать рестарт.

_REPORT_CACHE: dict[int, Report] = {}


def save_report(report: Report) -> None:
    _REPORT_CACHE[report.user_id] = report


def get_report(user_id: int) -> Optional[Report]:
    return _REPORT_CACHE.get(user_id)


def clear_report(user_id: int) -> None:
    _REPORT_CACHE.pop(user_id, None)
