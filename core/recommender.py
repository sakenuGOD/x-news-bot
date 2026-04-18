"""Ранжирование и отбор топ-N твитов.

Устройство системы рекомендаций (ключевое):

  score(tweet, user) =
      0.40 * relevance       # cosine(tweet.emb, user.preference_vector)
    + 0.15 * cluster_boost   # взвешенная сумма cosine к якорям кластеров
    + 0.25 * trust           # source_trust_score
    + 0.20 * freshness       # exp(-hours_old / 12)
    - 0.15 * diversity_pen   # похожесть на уже выбранные в этом батче
    + exploration_bonus      # небольшая случайность для "разведки"

Почему это работает когда пользователь редко ставит 👍:

  1. ДИЗЛАЙК — основной сигнал обучения. update_on_dislike двигает
     preference_vector В ОБРАТНУЮ сторону от нерелевантной новости.
     Со временем "плохое" направление схлопывается, остаётся "хорошее".

  2. IMPLICIT SKIP — если мы показали 10 постов темы X и ни одного лайка,
     apply_implicit_skips() дают лёгкий негативный сигнал по этим постам.
     Это позволяет модели учиться даже без кликов.

  3. EXPLORATION — 20% слотов отдаются под твиты из недопредставленных
     кластеров / менее уверенные рекомендации. Без этого модель бы
     заштормила в узкий confirmation-bubble.

  4. CLUSTER WEIGHTS — явное управление через "хочу больше..."
     накладывается поверх embedding-similarity, даёт грубый контроль.

  5. DIVERSITY PENALTY — в одну выдачу не попадёт 3 твита одной темы.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from sqlalchemy import and_, func, not_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from core import embeddings as emb
from db.models import Feedback, SentNews, Tweet, User

log = logging.getLogger(__name__)


@dataclass
class ScoredTweet:
    tweet: Tweet
    score: float
    components: dict[str, float] = field(default_factory=dict)


# ----------------------- scoring ------------------------


def freshness_score(created_at: datetime) -> float:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
    return math.exp(-max(0.0, hours) / 12.0)


async def compute_cluster_boost(
    tweet_embedding: Sequence[float],
    cluster_weights: dict[str, float],
) -> float:
    """Взвешенная сумма cosine-сходств тhweet к каждому из якорей кластеров.

    Кластеры с weight=0 не участвуют. Результат нормализуем в [0, 1].
    """
    if not cluster_weights or not tweet_embedding:
        return 0.0
    anchors = await emb.get_cluster_anchors()
    if not anchors:
        return 0.0
    total_weight = 0.0
    weighted_sum = 0.0
    for name, w in cluster_weights.items():
        if w <= 0:
            continue
        anchor = anchors.get(name)
        if not anchor:
            continue
        sim = emb.cosine_similarity(tweet_embedding, anchor)
        # cosine в [-1, 1] -> в [0, 1]
        sim01 = max(0.0, (sim + 1) / 2)
        weighted_sum += w * sim01
        total_weight += w
    if total_weight <= 0:
        return 0.0
    return weighted_sum / total_weight


def diversity_penalty(
    tweet_embedding: Sequence[float],
    already_selected: Iterable[Sequence[float]],
    threshold: float | None = None,
) -> float:
    """Сколько уже выбранных твитов «похожи» на этот (cosine > threshold).

    Возвращаем штраф 0..1+ — чем больше похожих, тем выше.
    """
    th = threshold if threshold is not None else settings.diversity_threshold
    penalty = 0.0
    for other in already_selected:
        sim = emb.cosine_similarity(tweet_embedding, other)
        if sim > th:
            # Линейный штраф от 0 до 1 при sim от th до 1.
            penalty += (sim - th) / (1.0 - th)
    return penalty


def score_tweet(
    tweet: Tweet,
    tweet_emb: Sequence[float],
    user_vec: Sequence[float] | None,
    cluster_boost_val: float,
    already_selected_embs: Sequence[Sequence[float]],
    author_weight: float = 1.0,
) -> ScoredTweet:
    """Скор твита 0..1+.

    author_weight — FollowedAuthor.weight этого автора для данного юзера.
    1.0 — нейтрально, 1.5 — недавно добавленный по «хочу больше», 2.0+ —
    автор чьи посты юзер многократно лайкал. Линейно умножает итог: для
    добавленных через «больше моды» аккаунтов их посты получают значимый
    подъём, чтобы реально вытеснить старую доминирующую подписку из топа.
    """
    relevance = emb.cosine_similarity(tweet_emb, user_vec) if user_vec else 0.0
    relevance01 = max(0.0, (relevance + 1) / 2)

    trust = float(tweet.source_trust_score or 0.0)
    fresh = freshness_score(tweet.created_at)
    div_pen = diversity_penalty(tweet_emb, already_selected_embs)

    # Штраф за misleading если проверяли.
    mis_pen = float(tweet.misleading_score or 0.0)

    base = (
        0.40 * relevance01
        + 0.15 * cluster_boost_val
        + 0.25 * trust
        + 0.20 * fresh
        - 0.15 * div_pen
        - 0.25 * mis_pen
    )
    # Автор-бонус: FollowedAuthor.weight обычно 1.0-3.0. Нормализуем к
    # (weight-1)*0.25 — при weight=2.0 (новый канал после «хочу больше») это
    # +25% к скору, при 3.0 — +50%. Это достаточно чтобы ВЫТЕСНЯТЬ посты
    # старой инертной подписки в пользу свежедобавленных каналов. Юзер
    # жаловался: «добавил каналы с весом 0.1 вместо 10».
    author_bonus = max(0.0, (float(author_weight) - 1.0)) * 0.25
    s = base + author_bonus

    return ScoredTweet(
        tweet=tweet,
        score=s,
        components={
            "relevance": relevance01,
            "cluster": cluster_boost_val,
            "trust": trust,
            "fresh": fresh,
            "diversity_pen": div_pen,
            "misleading_pen": mis_pen,
            "author_bonus": author_bonus,
        },
    )


# ----------------------- candidate pool ------------------------


async def get_candidate_pool(
    session: AsyncSession,
    user: User,
    max_age_hours: int | None = None,
    limit: int = 300,
) -> list[Tweet]:
    """Твиты, которые можно показать: свежие, с embedding, ещё не отправленные этому юзеру."""
    age_h = max_age_hours or settings.tweet_max_age_hours
    cutoff = datetime.now(timezone.utc) - timedelta(hours=age_h)

    # Подзапрос: tweet_id которые уже отправляли этому юзеру.
    sent_ids_stmt = select(SentNews.tweet_id).where(SentNews.user_id == user.telegram_id)

    stmt = (
        select(Tweet)
        .where(
            and_(
                Tweet.created_at >= cutoff.replace(tzinfo=None),  # sqlite naive
                Tweet.embedding.is_not(None),
                not_(Tweet.tweet_id.in_(sent_ids_stmt)),
            )
        )
        .order_by(Tweet.fetched_at.desc())
        .limit(limit)
    )
    res = await session.execute(stmt)
    results = list(res.scalars().all())
    if results:
        return results

    # Fallback для power-юзеров: расширяем окно 2x И разрешаем повторно
    # показать твиты, которые юзер лайкнул (liked=True — явный сигнал
    # "хочу ещё"). Лучше дать подборку из уже виденного, чем пустое
    # сообщение "нет свежих постов".
    extended_cutoff = datetime.now(timezone.utc) - timedelta(hours=age_h * 2)
    liked_sent_stmt = select(Feedback.tweet_id).where(
        Feedback.user_id == user.telegram_id,
        Feedback.liked.is_(True),
    )
    stmt_extended = (
        select(Tweet)
        .where(
            and_(
                Tweet.created_at >= extended_cutoff.replace(tzinfo=None),
                Tweet.embedding.is_not(None),
                or_(
                    not_(Tweet.tweet_id.in_(sent_ids_stmt)),
                    Tweet.tweet_id.in_(liked_sent_stmt),
                ),
            )
        )
        .order_by(Tweet.fetched_at.desc())
        .limit(limit)
    )
    res2 = await session.execute(stmt_extended)
    return list(res2.scalars().all())


# ----------------------- main pick ------------------------


async def pick_top_for_user(
    session: AsyncSession,
    user: User,
    top_n: int | None = None,
) -> list[ScoredTweet]:
    """Главный метод: возвращает топ-N для отправки пользователю.

    Использует:
      - user.preference_vector  — усреднённый профиль по лайкам/дизлайкам
      - user.cluster_weights    — явные предпочтения из онбординга / "хочу больше"
      - diversity penalty       — не 3 похожих твита подряд
      - exploration slots       — 20% рандом из перспективных, но менее очевидных

    Если preference_vector пуст (новый юзер до первого лайка) — опираемся только
    на cluster_weights + свежесть + trust.
    """
    n = top_n or settings.top_n_per_delivery
    pool = await get_candidate_pool(session, user)
    if not pool:
        log.info("user=%s: empty candidate pool", user.telegram_id)
        return []

    user_vec = user.preference_vector
    cluster_weights = user.cluster_weights or {}

    # Bootstrap: если preference_vector ещё не накопился (новый юзер / нет
    # реакций), cosine(tweet, None) = 0 для всех — ранжирование схлопнется
    # в trust+freshness и выглядит как случайное. Синтезируем временный
    # вектор из якорей кластеров, взвешенных по cluster_weights (которые
    # задаются в онбординге). Даёт осмысленные relevance-скоры с дня 1.
    if not user_vec and cluster_weights:
        anchors = await emb.get_cluster_anchors()
        if anchors:
            total_w = sum(cluster_weights.values())
            if total_w > 0:
                dim = len(next(iter(anchors.values())))
                boot_vec = [0.0] * dim
                for name, w in cluster_weights.items():
                    anchor = anchors.get(name)
                    if anchor:
                        for i, v in enumerate(anchor):
                            boot_vec[i] += (w / total_w) * v
                if any(boot_vec):
                    user_vec = emb.normalize(boot_vec)

    # Предзагружаем карту FollowedAuthor.weight для текущего юзера — одним
    # запросом, чтобы не тянуть на каждый твит. Это ключ Bug 9: новые каналы
    # (weight=1.5) и лайкнутые авторы (weight>=1.25) должны влиять на ранжирование.
    from db.models import FollowedAuthor
    fa_rows = (await session.execute(
        select(FollowedAuthor.author_username, FollowedAuthor.weight)
        .where(FollowedAuthor.user_id == user.telegram_id)
    )).all()
    author_weights: dict[str, float] = {}
    for row in fa_rows:
        uname = (row[0] or "").lower()
        if uname:
            author_weights[uname] = float(row[1] or 1.0)

    # Скорим все кандидаты БЕЗ учёта diversity — получаем базовый ранжированный список.
    base_scored: list[ScoredTweet] = []
    for tw in pool:
        if not tw.embedding:
            continue
        cluster_boost_val = await compute_cluster_boost(tw.embedding, cluster_weights)
        author_w = author_weights.get((tw.author_username or "").lower(), 1.0)
        scored = score_tweet(tw, tw.embedding, user_vec, cluster_boost_val,
                             already_selected_embs=[], author_weight=author_w)
        base_scored.append(scored)

    base_scored.sort(key=lambda s: s.score, reverse=True)

    # Стратегия отбора: жадный с учётом diversity penalty.
    # + резервируем ~20% слотов под exploration.
    exploit_n = max(1, n - max(1, int(round(n * settings.exploration_ratio))))
    explore_n = n - exploit_n

    picked: list[ScoredTweet] = []
    picked_embs: list[list[float]] = []

    # 1) Exploit — жадный выбор из топа с учётом diversity.
    for cand in base_scored:
        if len(picked) >= exploit_n:
            break
        div_pen = diversity_penalty(cand.tweet.embedding, picked_embs)
        # Мягкий cap вместо жёсткого пропуска: для узких горячих тем
        # (e.g. "Claude 4.7 release") все посты cosine > 0.95 → div_pen
        # уходит в космос после первого же выбора, и юзер видит 1 пост
        # по горячей теме вместо 3-4. Штрафуем, но не блокируем.
        capped_pen = min(div_pen, 0.8)
        adjusted = cand.score - 0.15 * capped_pen
        if adjusted < -0.3:
            continue
        cand.score = adjusted
        picked.append(cand)
        picked_embs.append(list(cand.tweet.embedding))

    # 2) Explore — случайный weighted выбор из "хвоста" (исключая уже выбранные).
    if explore_n > 0:
        picked_ids = {s.tweet.tweet_id for s in picked}
        rest = [s for s in base_scored if s.tweet.tweet_id not in picked_ids]
        # Выбираем из верхних 40% остатка — не хлам, но и не самый безопасный топ.
        cutoff = max(1, int(len(rest) * 0.4))
        explore_pool = rest[:cutoff]

        # Понижаем вероятность выбора уже сильно похожего на picked.
        def _explore_weight(s: ScoredTweet) -> float:
            div_pen = diversity_penalty(s.tweet.embedding, picked_embs)
            if div_pen > 1.0:
                return 0.0
            # score может быть отрицательным — сдвигаем.
            return max(0.0, s.score + 1.0) * (1.0 - min(1.0, div_pen))

        for _ in range(explore_n):
            if not explore_pool:
                break
            weights = [_explore_weight(s) for s in explore_pool]
            total = sum(weights)
            if total <= 0:
                break
            chosen = random.choices(explore_pool, weights=weights, k=1)[0]
            picked.append(chosen)
            picked_embs.append(list(chosen.tweet.embedding))
            explore_pool = [s for s in explore_pool if s.tweet.tweet_id != chosen.tweet.tweet_id]

    picked.sort(key=lambda s: s.score, reverse=True)
    log.info(
        "user=%s: picked %d/%d (pool=%d). top score=%.3f",
        user.telegram_id,
        len(picked),
        n,
        len(pool),
        picked[0].score if picked else 0.0,
    )
    return picked


# ----------------------- implicit learning ------------------------


async def apply_implicit_skip_decay(
    session: AsyncSession,
    user: User,
    hours_without_reaction: int = 12,
) -> int:
    """Твиты отправленные давно без реакции → implicit negative.

    Слабый сигнал: двигаем preference_vector в сторону ОТ таких твитов
    с маленьким rate. Чем больше таких твитов, тем сильнее суммарный эффект.

    Возвращает количество обработанных твитов.
    """
    if not user.preference_vector:
        return 0

    # Минимальный порог реального фидбэка — decay должен корректировать
    # уже устоявшийся вектор, а не формировать его. Без этого 1 пропущенный
    # батч у нового юзера полностью ломал preference_vector.
    real_feedback_count = (await session.execute(
        select(func.count(Feedback.id)).where(
            Feedback.user_id == user.telegram_id,
            Feedback.liked.is_not(None),
        )
    )).scalar() or 0
    if real_feedback_count < 3:
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_without_reaction)

    # SentNews для этого юзера, где: (a) прошло > cutoff и (b) нет Feedback.
    stmt = (
        select(SentNews, Tweet)
        .join(Tweet, Tweet.tweet_id == SentNews.tweet_id)
        .where(
            and_(
                SentNews.user_id == user.telegram_id,
                SentNews.sent_at <= cutoff.replace(tzinfo=None),
                # feedback отсутствует
                not_(
                    SentNews.tweet_id.in_(
                        select(Feedback.tweet_id).where(Feedback.user_id == user.telegram_id)
                    )
                ),
            )
        )
        .limit(50)
    )
    rows = (await session.execute(stmt)).all()
    if not rows:
        return 0

    vec = list(user.preference_vector)
    count = 0
    for _sent, tweet in rows:
        if not tweet.embedding:
            continue
        # Пониженный rate (0.005 vs settings.implicit_skip_penalty=0.015):
        # 6-часовая джоба обрабатывала сразу 20+ постов у спящего юзера и
        # схлопывала preference_vector. Оставляем мягкий сигнал, а не бульдозер.
        vec = emb.update_on_dislike(vec, tweet.embedding, 0.005)
        # Записываем Feedback(liked=None) чтобы больше не учитывать эти твиты.
        session.add(
            Feedback(
                user_id=user.telegram_id,
                tweet_id=tweet.tweet_id,
                liked=None,
            )
        )
        count += 1

    user.preference_vector = vec
    await session.flush()
    log.info("user=%s: implicit skip applied to %d tweets", user.telegram_id, count)
    return count


def apply_cluster_weight_update(
    current: dict[str, float],
    boost: list[str],
    suppress: list[str],
    boost_delta: float = 0.25,
    suppress_delta: float = 0.30,
) -> dict[str, float]:
    """Обновляет cluster_weights на основе явного запроса пользователя.

    boost:   w += boost_delta, clamp(0..1)
    suppress: w -= suppress_delta (min 0)
    Затем мягко ренормализуем сумму в ~1, но не насильно — веса могут быть
    и с суммой 0.5 (мало явных предпочтений).
    """
    new = dict(current or {})
    for name in boost:
        new[name] = min(1.0, new.get(name, 0.3) + boost_delta)
    for name in suppress:
        new[name] = max(0.0, new.get(name, 0.3) - suppress_delta)

    total = sum(new.values())
    if total > 1.5:
        # Нормализуем только если совсем разъехалось.
        new = {k: v / total for k, v in new.items()}
    return new
