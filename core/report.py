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
from sqlalchemy import select
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
    from sqlalchemy import select

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

    queries = await ai_client.suggest_interest_queries(
        cluster_weights=active,
        saved_queries=saved_list,
        followed_authors=following_names,
        max_queries=max_queries,
    )
    return queries[:max_queries]


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


# Слова которые намекают что Haiku сдался и написал шаблон.
_SUMMARY_BAD_SUBSTRINGS = (
    "посты не содержат",
    "нет конкретн",
    "скоро будет",
    "скоро состо",
    "обсуждают что",
    "обсуждается",
)


def _accept_summary(text: str | None) -> str | None:
    """Возвращает text если выжимка валидна, иначе None.

    Критерии: ≥40 символов, не упирается в «постов мало/пусто», не состоит из
    одной шаблонной фразы. Это не фильтрация содержания — это отсев провалов
    модели, которые хуже чем отсутствие саммари.
    """
    if not text:
        return None
    t = text.strip()
    if len(t) < 40:
        return None
    low = t.lower()
    # Если ВСЯ выжимка — одна шаблонная фраза (нет запятых/точек дальше), отсекаем.
    for bad in _SUMMARY_BAD_SUBSTRINGS:
        if low.startswith(bad) and len(t) < 120:
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

    # ---------- 1b. Interest-driven search — добираем темы по интересам юзера ----------
    # Идея: For You не всегда покрывает ВСЕ темы которые нам интересны. Если юзер
    # подписан на Vogue — есть сигнал «мода», и надо поискать активные посты по
    # этой теме явно. Это делает отчёт ближе к X Stories (где темы явные).
    #
    # ТОЛЬКО для For You («Что обсуждают»). «Моя лента» (Following) — чистый
    # поток подписок без доискивания: юзер жаловался что лента долго считается,
    # пока бот запускает 5 search-запросов и 8 author-fetch'ей. Для Following
    # достаточно хронологического timeline, без модификаций.
    if source == "for_you":
        # Обширный поиск: до 5 запросов (было 3), плюс посты от FollowedAuthor
        # помимо X-search. Юзер жаловался: «он ищет по 1-2 поста, не делает
        # полный чекап». 5 запросов × 90 постов = до 450 кандидатов по теме.
        interest_queries = await _derive_interest_queries(session, user, max_queries=5)
        if interest_queries:
            await _notify(f"🔎 Доискиваю темы по интересам: {', '.join(interest_queries[:3])}…")

            # Готовим extra_authors: последние добавленные FollowedAuthor —
            # это те что Claude подобрал специально под запрос юзера
            # (например @voguejapan, @fashionsnap после «японский стиль»).
            # Их посты — самые релевантные для нишевой темы, лучше чем curated.
            from db.models import FollowedAuthor
            from sqlalchemy import select, desc
            recent_fa_rows = (await session.execute(
                select(FollowedAuthor.author_username)
                .where(FollowedAuthor.user_id == user.telegram_id)
                .order_by(desc(FollowedAuthor.added_at))
                .limit(15)
            )).all()
            recent_authors = [r[0] for r in recent_fa_rows]

            already = {t.tweet_id for t in raw}
            for q in interest_queries:
                extra: list[RawTweet] = []
                try:
                    # Большой count — полноценная подборка темы.
                    # Top-search возвращает до 40 за раз; Latest — ещё 40. После
                    # дедупа — 50-70 уникальных постов на запрос.
                    top = await parser.search_tweets(q, product="Top", count=40)
                    extra.extend(top)
                    latest = await parser.search_tweets(q, product="Latest", count=40)
                    seen_q = {x.tweet_id for x in extra}
                    for t in latest:
                        if t.tweet_id not in seen_q:
                            extra.append(t)
                            seen_q.add(t.tweet_id)
                except Exception as e:
                    log.debug("interest search %s failed: %s", q, e)
                # Если X-search ничего не дал (404 / rate-limit) — идём к авторам:
                # recent_authors (которых Claude подобрал) приоритет, затем curated.
                # Большой пул: 12 авторов × 10 постов = до 120 кандидатов на запрос.
                if not extra:
                    try:
                        extra = await parser.topic_authors_fallback(
                            q, per_author=10, max_authors=12,
                            extra_authors=recent_authors,
                        )
                        if extra:
                            log.info("interest search %r: authors-fallback, got %d posts",
                                     q, len(extra))
                    except Exception as e:
                        log.debug("authors fallback for %r failed: %s", q, e)
                # ДОПОЛНЯЕМ X-search постами FollowedAuthor по теме — даже если
                # X-search сам вернул 40 постов. Добавленные каналы должны
                # реально «весить» в результатах, а не быть декоративными.
                # (Юзер: «подсовываю аккаунты с весом 0.1 вместо 10».)
                if recent_authors and len(extra) < 60:
                    try:
                        author_extra = await parser.get_recent_tweets_for_authors(
                            recent_authors[:6], limit_per_author=5,
                        )
                        extra_ids = {x.tweet_id for x in extra}
                        for t in author_extra:
                            if t.tweet_id not in extra_ids:
                                extra.append(t)
                                extra_ids.add(t.tweet_id)
                    except Exception as e:
                        log.debug("extra author-fetch for %r failed: %s", q, e)
                for t in extra:
                    if t.tweet_id not in already:
                        raw.append(t)
                        already.add(t.tweet_id)
                await asyncio.sleep(0.4)  # щадящий ритм, X любит rate-limit

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

    # ---------- 6. Cluster ----------
    await _notify(f"🗂 Группирую в темы…")
    clusters, unclustered_ids = await _cluster_and_name(session, tweets, user, progress=_notify)

    # ---------- 6a. Super-topic grouping ----------
    # Claude объединяет 8-10 конкретных тем в 3-5 широких категорий (IT, Мода,
    # Спорт, ...). Overview показывает супер-категории, клик по ним — сами темы.
    super_topics: list[SuperTopic] = []
    if len(clusters) >= 3:
        await _notify("🧭 Собираю в широкие категории…")
        pairs = [(c.id, c.emoji, c.name) for c in clusters]
        try:
            grouped = await ai_client.group_super_topics(pairs)
            for g in grouped:
                super_topics.append(SuperTopic(
                    emoji=g["emoji"], name=g["name"], sub_ids=list(g["sub_ids"]),
                ))
        except Exception as e:
            log.debug("super_topic grouping failed: %s", e)

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

    # --- Stage 4: сортируем по размеру, обрезаем до max_topics, хвост → «Разное» ---
    order = sorted(range(len(groups)), key=lambda i: len(groups[i]), reverse=True)
    groups = [groups[i] for i in order]
    named = [named[i] for i in order]

    top_groups = groups[:max_topics]
    top_named = named[:max_topics]
    rest = [i for grp in groups[max_topics:] for i in grp]
    if len(rest) >= min_cluster_size:
        top_groups.append(rest)
        top_named.append(("📰", "Остальное"))

    out: list[ReportCluster] = []
    for cid, (grp, (emoji, name)) in enumerate(zip(top_groups, top_named)):
        sorted_idx = sorted(grp, key=lambda i: _score_for_ordering(tweets[i], user_vec), reverse=True)
        out.append(ReportCluster(
            id=cid,
            emoji=emoji,
            name=name,
            tweet_ids=[tweets[i].tweet_id for i in sorted_idx],
        ))

    log.info("clustered %d tweets → %d topics (after name-merge)", len(tweets), len(out))
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
