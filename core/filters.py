"""Дешёвые фильтры ДО векторизации и AI — регулярки и эвристики.

Цель: выкинуть 50-70% мусора бесплатно, чтобы не тратить токены на эмбеддинги
и вызовы Claude для рекламы/RT/хайпа.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Sequence

from config import settings
from core.embeddings import cosine_similarity
from core.x_parser import RawTweet

log = logging.getLogger(__name__)


# Маркеры рекламы/промо. Ищем как whole-word где возможно.
_AD_MARKERS = [
    r"\bsponsored\b",
    r"\bpromo\b",
    r"\bad\b",  # как отдельное слово
    r"\baffiliate\b",
    r"\bdiscount code\b",
    r"\buse my link\b",
    r"\bbuy now\b",
    r"\blimited offer\b",
    r"\bpromocode\b",
    r"\bpromo code\b",
    r"\bcashback\b",
    r"\bgiveaway\b",
    r"\bairdrop\b",
    r"\bnot financial advice\b",
    r"\bdyor\b",  # do your own research — маркер крипто-шиллинга
    r"\bpump\b.*\bmoon\b",
]

_AD_RE = re.compile("|".join(_AD_MARKERS), re.IGNORECASE)

# Русские маркеры — тоже встречаются.
_AD_RE_RU = re.compile(
    r"(реклам\w+|промокод|скидк\w+ по ссылк\w+|покупайте|купи сейчас|акция\s+только)",
    re.IGNORECASE,
)


def is_trash(tweet: RawTweet) -> tuple[bool, str]:
    """Возвращает (is_trash, reason). Дешёвая проверка без AI."""
    text = tweet.text or ""
    text_lower = text.lower()

    # Длина в словах (после удаления ссылок и @mentions).
    cleaned = re.sub(r"https?://\S+|@\w+", "", text).strip()
    words = cleaned.split()
    if len(words) < 5:
        return True, "too_short"

    # Слишком много хештегов.
    if len(tweet.hashtags) > 5:
        return True, "hashtag_spam"

    # Реклама / промо.
    if _AD_RE.search(text_lower):
        return True, "ad_marker_en"
    if _AD_RE_RU.search(text):
        return True, "ad_marker_ru"

    # Возраст.
    if tweet.age_hours > settings.tweet_max_age_hours:
        return True, "too_old"

    # Чистый RT без комментария.
    if tweet.is_retweet_no_comment:
        return True, "pure_retweet"
    if text.strip().startswith("RT @"):
        return True, "pure_retweet"

    # Только ссылка в теле — часто спам/promo.
    if re.fullmatch(r"\s*https?://\S+\s*", text):
        return True, "url_only"

    # Капс-лок более 70% символов и длиннее 40 символов.
    if len(text) > 40:
        letters = [c for c in text if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
            return True, "all_caps"

    return False, ""


def is_duplicate(
    embedding: Sequence[float],
    existing_embeddings: Iterable[Sequence[float]],
    threshold: float | None = None,
) -> bool:
    """cosine_sim > threshold с любым из existing → дубликат."""
    th = threshold if threshold is not None else settings.duplicate_threshold
    for other in existing_embeddings:
        if cosine_similarity(embedding, other) > th:
            return True
    return False


def dedupe_by_embedding(
    tweets_with_embs: list[tuple[RawTweet, list[float]]],
    threshold: float | None = None,
) -> list[tuple[RawTweet, list[float]]]:
    """Жадная дедубликация: идём по убыванию engagement, отбрасываем похожие."""
    th = threshold if threshold is not None else settings.duplicate_threshold
    # Сортируем — при дублях предпочитаем более залайканный.
    items = sorted(
        tweets_with_embs,
        key=lambda x: (x[0].likes_count + x[0].retweets_count * 2, -x[0].age_hours),
        reverse=True,
    )
    kept: list[tuple[RawTweet, list[float]]] = []
    for tw, emb in items:
        if not emb:
            continue
        dup = False
        for _, other_emb in kept:
            if cosine_similarity(emb, other_emb) > th:
                dup = True
                break
        if not dup:
            kept.append((tw, emb))
    return kept


def engagement_rate(author_followers: int, likes: int, rts: int, replies: int) -> float:
    """likes+rts+replies, нормализованные на followers. 0 если followers=0."""
    if author_followers <= 0:
        return 0.0
    return (likes + 2 * rts + replies) / max(100, author_followers)


def hype_score(tweet: RawTweet) -> float:
    """Насколько пост «хайп без смысла» (0..1). Выше → меньше ценности.

    Нужен потому что в X много постов-реакций, CAPS-эмоций, engagement-farming
    и однострочных «based/this/gg». В отчёт такие попадать не должны —
    user хочет только «дельное, без воды».

    Порог 0.5 уже неплохой: выше — точно мусор.
    """
    text = (tweet.text or "").strip()
    if not text:
        return 1.0

    score = 0.0
    words = text.split()
    wc = len(words)

    # 1) Чисто-реактивный пост («wow», «based», «lmao», «lfg», «this», «exactly»).
    if re.fullmatch(
        r"(?i)\s*(wow+|lmao+|lol+|omg+|yes+|no+|based|this|exactly|ok+|oof|gg+|lfg+|rip|damn|fr+|ngl+|🚀+|🔥+|💯+)[\s.!?]*",
        text,
    ):
        return 0.95  # почти гарантированно мусор

    # 2) Плотность восклицательных — хайп.
    exclam = text.count("!")
    if exclam >= 3:
        score += 0.25
    if exclam >= 1 and wc < 10:
        score += 0.15  # короткое + восклицание = крик

    # 3) ALL CAPS фразы длиной 2+ слова («TO THE MOON», «BULLISH AF»).
    if re.search(r"\b[A-Z]{4,}(\s+[A-Z]{4,})+\b", text):
        score += 0.25

    # 4) Эмодзи-спам.
    emoji_n = sum(1 for c in text if ord(c) >= 0x1F000)
    if emoji_n >= 5:
        score += 0.2
    if wc < 15 and emoji_n >= 3:
        score += 0.15

    # 5) Лексика хайпа / инфоцыгантсва.
    hype_re = re.compile(
        r"\b(bullish|moon(ing)?|🚀|to the moon|gem|100x|1000x|pump|fomo|huge|massive|"
        r"insane|crazy|shocking|mind[- ]blown|game[- ]chang|literally|absolutely|"
        r"you won't believe|breaking|just in|urgent|alpha)\b",
        re.IGNORECASE,
    )
    if hype_re.search(text):
        score += 0.2

    # 6) Engagement farming.
    farm_re = re.compile(
        r"\b(rt if|retweet if|like if|follow (me|back|for)|tag a friend|"
        r"agree\?|thoughts\?|who else|reply if|mute me if|drop a|"
        r"comment [\"'].+?[\"'] for)\b",
        re.IGNORECASE,
    )
    if farm_re.search(text):
        score += 0.25

    # 7) Риторический короткий вопрос без фактов.
    if text.endswith("?") and wc < 12:
        score += 0.15

    # 8) Длина — очень короткие без цифр/ссылок/имён обычно пустые.
    has_numeric = bool(re.search(r"\d", text))
    has_url = "http" in text
    has_mention = "@" in text
    if wc < 8 and not (has_numeric or has_url or has_mention):
        score += 0.2

    # 9) Слова-наполнители подряд («so», «just», «really», «like»).
    filler_re = re.compile(r"\b(so|just|really|basically|literally|actually)\b", re.IGNORECASE)
    filler_hits = len(filler_re.findall(text))
    if filler_hits >= 3:
        score += 0.1

    return min(1.0, score)


def info_density_score(tweet: RawTweet) -> float:
    """0..1: насколько в посте плотная конкретика.

    Сигналы «это что-то дельное»:
      - Цифры (проценты, K/M, версии, даты)
      - Имена собственные / брэнды (2+ capital-word последовательностей)
      - URL (ссылка на источник)
      - @-меншн (называет конкретные сущности)
      - Длина ≥ 15 слов

    Значение используется в `is_low_signal`: если density < 0.25 — фильтруем
    как «пост без сути» (user жалуется на «булшит в ленте»).
    """
    text = (tweet.text or "").strip()
    if not text:
        return 0.0
    score = 0.0
    # Длина.
    wc = len(text.split())
    if wc >= 15:
        score += 0.25
    elif wc >= 8:
        score += 0.1
    # Цифры: %, K/M/B, версии, даты.
    import re as _re
    if _re.search(r"\d+[%kKmMbB]|\d+\.\d+|\$\d+|\d{4}", text):
        score += 0.25
    # URL — ссылка на статью/пост/видео.
    if "http" in text:
        score += 0.15
    # 2+ capitalized sequences (имена/продукты/брэнды)
    caps = _re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}(?:\s+[A-Z][a-zA-Z0-9]{2,})+\b", text)
    if caps:
        score += 0.2
    # @-меншн вне начала твита.
    mention_count = text.count("@")
    if mention_count >= 1:
        score += 0.15
    # Наличие медиа — сигнал «полноценный пост».
    if tweet.image_url or tweet.media_type:
        score += 0.15
    return min(1.0, score)


_LIST_DUMP_RE = re.compile(
    # Sprawl-посты типа «⚽ City smell blood… / 🏆 Bayern eye history… / 🏒 Stanley…»
    # Несколько «эмодзи + короткая строка» подряд → дайджест-заголовков-ссылок.
    r"^(?:[\U0001F300-\U0001FAFF⚽️🏆🏒🎾🏀⚾️🏈🎬📺📰🔔📊]\s*[^\n]{4,80}\n){2,}",
    re.MULTILINE,
)


def is_list_dump(tweet: RawTweet) -> bool:
    """Пост-агрегатор заголовков с ссылками («Inside Track», «Top 5 stories»).

    Характеристика: 2+ строк вида «эмодзи + короткая фраза», плюс URL в конце.
    Это не обсуждение, а компоновка — в дайджесте занимает место без пользы.
    """
    text = tweet.text or ""
    if _LIST_DUMP_RE.search(text):
        # Ужесточаем: ещё и URL должен быть в тексте.
        if "http" in text.lower() or tweet.hashtags:
            return True
    # Признак редакционного дайджеста: «Here's your Inside Track», «Today's briefing».
    low = text.lower()
    if any(p in low for p in ("inside track", "today's briefing", "here's your briefing",
                               "your daily briefing", "here's what you need to know today")):
        return True
    return False


def is_low_signal(
    tweet: RawTweet,
    hype_threshold: float = 0.5,
    min_density: float = 0.25,
) -> tuple[bool, str]:
    """Комбинированная проверка: мусор ИЛИ хайп ИЛИ низкая инфо-плотность
    ИЛИ list-dump (Reuters-sprawl headlines)."""
    trashy, reason = is_trash(tweet)
    if trashy:
        return True, reason
    if is_list_dump(tweet):
        return True, "list_dump"
    hs = hype_score(tweet)
    if hs >= hype_threshold:
        return True, f"hype:{hs:.2f}"
    density = info_density_score(tweet)
    if density < min_density:
        return True, f"low_density:{density:.2f}"
    return False, ""


def needs_antifake_check(tweet: RawTweet, author_trust: float) -> bool:
    """Триггер для дорогой AI-проверки. Мы не хотим гонять Haiku на каждый твит."""
    text = tweet.text or ""
    if len(text) < 40:
        return False
    # Малодоверенный источник + громкое заявление.
    if author_trust < 0.35:
        hype_markers = re.search(
            r"\b(breaking|shocking|exclusive|revealed|secret|they don't want you|conspiracy)\b",
            text,
            re.IGNORECASE,
        )
        big_numbers = re.search(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+%\b", text)
        if hype_markers or big_numbers:
            return True
    # Одиночный аккаунт на громкое утверждение без ссылки-источника.
    if not re.search(r"https?://\S+", text) and re.search(
        r"\b(official|confirmed|reported)\b", text, re.IGNORECASE
    ):
        return True
    return False
