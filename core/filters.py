"""Дешёвые фильтры ДО векторизации и AI — регулярки и эвристики.

Цель: выкинуть 50-70% мусора бесплатно, чтобы не тратить токены на эмбеддинги
и вызовы Claude для рекламы/RT/хайпа.
"""

from __future__ import annotations

import asyncio
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

# Обобщённые маркеры cross-platform промо / repost-фарма. Только
# универсально-спамные фразы — ничего tied к конкретной теме (мода/спорт/AI).
_PROMO_RE = re.compile(
    r"(link[s]?\s+in\s+bio"
    r"|join\s+our\s+(telegram|discord|whatsapp)"
    r"|follow\s+(me|us)\s+for\s+more"
    r"|tap\s+the\s+link\s+in\s+bio"
    r"|t\.me/\w+"
    r")",
    re.IGNORECASE,
)

# Обобщённый NSFW / onlyfans-маркеры. Хотим резать очевидно-porn/OF только.
_NSFW_RE = re.compile(
    r"(only\s*fans\b|18\+|\bnsfw\b|\bporn\b|\bxxx\b)",
    re.IGNORECASE,
)


def is_trash(tweet: RawTweet) -> tuple[bool, str]:
    """Возвращает (is_trash, reason). Дешёвая проверка без AI."""
    text = tweet.text or ""
    text_lower = text.lower()

    # Длина в словах. ВАЖНО: если в посте есть quote (↪ цитирует / ▌ @),
    # считаем только АВТОРСКУЮ часть — до quote-маркера. Иначе «Fashion
    # trends evolve rapidly. ↪ цитирует @x: Why all women are wearing this?»
    # проходит (quote даёт +8 слов), хотя авторский вклад = 4 слова.
    author_part = text
    for marker in ("↪ цитирует ", "Цитирует @", "\n▌ @", "▌ @"):
        idx = author_part.find(marker)
        if idx > 0:
            author_part = author_part[:idx]
            break
    cleaned = re.sub(r"https?://\S+|@\w+", "", author_part).strip()
    words = cleaned.split()
    if len(words) < 5:
        return True, "too_short"

    # Reply-fragment: ≥2 @-mentions в начале и короткий остаток. Это
    # классический фрагмент реплая («@a @b @c Love the style of the gauges»),
    # который сам по себе бессмыслен — его читать без тред-контекста нельзя.
    # Одиночный @mention оставляем (может быть legit discussion).
    leading = re.match(r"^(\s*@\w+\s*){2,}", text)
    if leading:
        rest = text[leading.end():].strip()
        rest_clean = re.sub(r"https?://\S+", "", rest)
        if len(rest_clean.split()) < 12:
            return True, "reply_fragment"

    # Слишком много хештегов.
    if len(tweet.hashtags) > 5:
        return True, "hashtag_spam"

    # Реклама / промо.
    if _AD_RE.search(text_lower):
        return True, "ad_marker_en"
    if _AD_RE_RU.search(text):
        return True, "ad_marker_ru"

    # Cross-platform promo (repost-farm аккаунты — 0 лайков, ссылка на свой
    # Telegram/Discord). Проверяем и body и quote — спам бывает в обоих.
    if _PROMO_RE.search(text):
        return True, "cross_platform_promo"
    if tweet.quote_text and _PROMO_RE.search(tweet.quote_text):
        return True, "cross_platform_promo_quote"

    # NSFW в цитате — юзер просил fashion, не «голый парень с Grok-промптом».
    if tweet.quote_text and _NSFW_RE.search(tweet.quote_text):
        return True, "nsfw_quote"
    if _NSFW_RE.search(text):
        return True, "nsfw_body"

    # ОТКЛЮЧЁН dead-post-engagement фильтр — резал легитные посты от мелких
    # аккаунтов в AI/IT нише (юзерские подписки типа @dejavucoder, @shawmakesmagic
    # получают 0 лайков за 2ч, но содержат живой контент). Юзер жаловался:
    # «он забил на мою ленту, новостей мало».

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


# ----------------------- embedding-anchor noise detection -----------------------
#
# Регулярки из `is_low_signal` пропускают рекламу, у которой нет слова
# «sponsored/promo» — типа «70% OFF tuxedo» @MenswearDeals, «Dakota Mini Dress
# $80.50» @qwikad_com, «Big & Tall London is pleased to introduce a refined
# collection of premium jeans» @BigandLondon. Их info_density проходит (есть
# цифра/бренд/%), hype_score низкий, цельный текст — формально «контент».
# Но это всё равно шоппинг-реклама.
#
# Решение без хардкода username'ов: эмбеддим 1 раз набор описаний «типов шума»
# (реклама, однострочный chatty-reply, content-farm), и на этапе векторизации
# постов смотрим cosine к anchor'ам. Порог подобран так, чтобы не резать
# легитные обсуждения моды/крипты/IT (cosine к anchor обычно 0.15-0.30),
# а бить именно по текстам которые семантически близки к рекламе или
# болтовне-реплаю. Anchors описывают ПРИРОДУ шума, не домен, поэтому
# работают одинаково для любого кластера (мода, AI, крипта).

_NOISE_ANCHOR_DESCRIPTIONS: dict[str, str] = {
    "promo_ad": (
        "retail product listing with price tag and shop link, "
        "percent off discount deal shop now call to action, "
        "e-commerce storefront post advertising merchandise with purchase URL, "
        "brand new arrival announcement linking to online catalog, "
        "sale pitch buy today limited offer promotional push"
    ),
    "drive_by_reply": (
        "short conversational reply to another tweet with mentions, "
        "offhand chat remark, casual one-liner replying to a mentioned account, "
        "low-effort quip, small-talk response, passing compliment or joke reply, "
        "fragment of a reply chain without standalone meaning"
    ),
    "content_farm": (
        "generic low-effort social media filler stuffed with sparkle and fire emojis, "
        "dazzling emoji-heavy brand announcement with many decorative symbols, "
        "bot-generated hype chatter without specific claim, "
        "engagement-bait question without context, "
        "cliche lifestyle magazine headline roundup with numbered trends, "
        "lifestyle promotional post full of emoji decoration and hashtag spam"
    ),
}

# «Хорошие» anchor'ы — то, что явно НЕ шум. Используем argmax-подход:
# если лучший noise-anchor ближе чем лучший signal-anchor (с запасом MARGIN),
# пост признаём шумом. Это устойчивее к абсолютным порогам: Viktor Oddy
# «recorded 18-min tutorial» — это announcement, ловится positive anchor
# сильнее, чем promo_ad; @BigandLondon «pleased to introduce new arrival»
# сильнее тянется к promo_ad, чем к announcement.
_SIGNAL_ANCHOR_DESCRIPTIONS: dict[str, str] = {
    "opinion_commentary": (
        "personal opinion or commentary on a topic, subjective observation, "
        "critique or analysis, thoughtful reflection, debate or argument about "
        "a trend or event, first-person perspective on cultural or technical shift"
    ),
    "announcement_creation": (
        "first-person announcement of something the author just built made or "
        "recorded, launching a project video or blog, sharing a how-to tutorial "
        "the speaker authored, release note from a creator about their own work"
    ),
    "news_report": (
        "news reporting of a concrete event, factual update about a specific "
        "incident or development, breaking story with named entities and timeline, "
        "journalistic account of what happened"
    ),
    "technical_claim": (
        "technical claim or benchmark about a product or system, "
        "side-by-side comparison with measurements, specific capability statement "
        "about software model or hardware, engineering detail or methodology"
    ),
}

# Пост признаётся шумом если best_noise_sim - best_signal_sim >= MARGIN.
# 0.07 — калибровано на выборке реальных постов: MenswearDeals/BigandLondon/
# qwikad/Lulu/fngdesign проходят (d≥0.083, d-values из _test_anchors.py
# на момент настройки), tutorial-посты «Opus 4.7 + SeedDance» от
# @viktoroddy сохраняются (d≈0.06 <margin). Снижать не стоит — начнёт
# резать announcement'ы с маркетинговым оттенком.
_NOISE_MARGIN = 0.07
# Абсолютный минимум noise-anchor, ниже — даже при большой разнице не режем
# (защита от случаев где все anchor'ы далёкие).
_NOISE_ABS_MIN = 0.22

_noise_anchors: dict[str, list[float]] | None = None
_signal_anchors: dict[str, list[float]] | None = None
_anchor_lock = asyncio.Lock()


async def _ensure_anchors() -> None:
    global _noise_anchors, _signal_anchors
    if _noise_anchors is not None and _signal_anchors is not None:
        return
    async with _anchor_lock:
        if _noise_anchors is not None and _signal_anchors is not None:
            return
        from core import embeddings as _emb  # локальный импорт во избежание circular
        all_names = list(_NOISE_ANCHOR_DESCRIPTIONS.keys()) + list(_SIGNAL_ANCHOR_DESCRIPTIONS.keys())
        all_descs = (
            list(_NOISE_ANCHOR_DESCRIPTIONS.values())
            + list(_SIGNAL_ANCHOR_DESCRIPTIONS.values())
        )
        embs = await _emb.embed_batch(all_descs)
        noise: dict[str, list[float]] = {}
        signal: dict[str, list[float]] = {}
        for name, e in zip(all_names, embs):
            if not e:
                continue
            if name in _NOISE_ANCHOR_DESCRIPTIONS:
                noise[name] = e
            else:
                signal[name] = e
        _noise_anchors = noise
        _signal_anchors = signal
        log.info("anchors computed: %d noise, %d signal", len(noise), len(signal))


async def get_noise_anchors() -> dict[str, list[float]]:
    """Публичный accessor (используется в тестах/диагностике)."""
    await _ensure_anchors()
    return dict(_noise_anchors or {})


async def get_signal_anchors() -> dict[str, list[float]]:
    await _ensure_anchors()
    return dict(_signal_anchors or {})


async def is_noise_by_embedding(
    tweet_emb: Sequence[float],
) -> tuple[bool, str]:
    """Argmax-классификатор: сравниваем с ближайшим signal- и noise-anchor'ом.

    Возвращает (is_noise, "noise:<type>:<noise_sim>-<signal_sim>") для логов.
    Вызывать ПОСЛЕ embed_batch (raw не имеет embedding).
    """
    if not tweet_emb:
        return False, ""
    await _ensure_anchors()
    if not _noise_anchors or not _signal_anchors:
        return False, ""

    best_noise_name = ""
    best_noise_sim = -1.0
    for name, anchor in _noise_anchors.items():
        sim = cosine_similarity(tweet_emb, anchor)
        if sim > best_noise_sim:
            best_noise_sim = sim
            best_noise_name = name

    best_signal_sim = -1.0
    for anchor in _signal_anchors.values():
        sim = cosine_similarity(tweet_emb, anchor)
        if sim > best_signal_sim:
            best_signal_sim = sim

    if best_noise_sim < _NOISE_ABS_MIN:
        return False, ""
    if best_noise_sim - best_signal_sim >= _NOISE_MARGIN:
        return True, f"noise:{best_noise_name}:{best_noise_sim:.2f}-{best_signal_sim:.2f}"
    return False, ""


def _engagement_vs_reach(tweet: RawTweet, author_followers: int) -> float:
    """Лайки-плюс-реплаи нормализованные на followers.

    Промо-аккаунты с >5k followers редко получают даже 1 лайк на пост —
    engagement ниже 0.00005 (1 на 20k подписчиков) достаточно уверенно
    маркирует «никто это не читает, потому что это реклама/мусор».
    Для мелких аккаунтов (≤1k followers) метрика бесполезна — возвращаем
    нейтральное значение.
    """
    if author_followers < 1000:
        return 1.0
    total = tweet.likes_count + 2 * tweet.retweets_count + tweet.replies_count
    return total / max(1, author_followers)


def is_dead_promo(tweet: RawTweet, author_followers: int) -> bool:
    """Большой аккаунт + мёртвый пост + возраст ≥2ч → почти всегда реклама.

    Не белые/чёрные списки — метрика. 5k+ подписчиков и меньше 3 total
    реакций за 2+ часа = пост не читают даже сами подписчики автора.
    Мелкие авторы (<1k) освобождены, чтобы не резать мирных пользователей
    которых юзер лайкал лично.
    """
    if author_followers < 5000:
        return False
    if tweet.age_hours < 2.0:
        return False
    total = tweet.likes_count + tweet.retweets_count + tweet.replies_count
    return total < 3


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
