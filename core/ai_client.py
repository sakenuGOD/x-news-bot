"""Все вызовы Claude (через ProxyAPI).

Правило: 4 метода = 4 точки вызова. Всё остальное — математика.
Каждый метод обёрнут в try/except и возвращает валидный pydantic объект
либо fallback — чтобы логика выше не падала из-за AI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Sequence

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field, ValidationError

from config import settings
from core.x_parser import RawTweet

log = logging.getLogger(__name__)

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(
            api_key=settings.proxyapi_key,
            base_url=settings.anthropic_base_url,
            timeout=60.0,
            max_retries=2,
        )
    return _client


# --------------------- pydantic схемы ответов ---------------------


class OnboardingAnalysis(BaseModel):
    hypothesis: str = Field(..., description="1-2 предложения о предполагаемых интересах")
    cluster_weights: dict[str, float] = Field(default_factory=dict)
    questions: list[str] = Field(default_factory=list, min_length=1, max_length=3)


class OnboardingRefinement(BaseModel):
    cluster_weights: dict[str, float] = Field(default_factory=dict)
    reply: str = Field(...)


class PreferenceRequestResult(BaseModel):
    boost: list[str] = Field(default_factory=list)
    suppress: list[str] = Field(default_factory=list)
    reply: str = Field(...)
    # Поиск-пакеты от Claude для X-search и авторов. Поля на модели, чтобы
    # pydantic v2 не блокировал setattr (extra='ignore' по умолчанию у BaseModel).
    search_queries: list[str] = Field(default_factory=list)
    suggested_accounts: list[str] = Field(default_factory=list)
    # Авторы которых юзер явно попросил больше не показывать
    # («исключи Reuters», «без Илона Маска»). Юзернеймы без @.
    blocked_accounts: list[str] = Field(default_factory=list)
    # Развёрнутое описание намерения юзера на английском (для embedding-anchor).
    # «хочу больше моды» → «editorial fashion coverage: Met Gala red carpet,
    # runway reviews, street style photography, menswear commentary. Avoid
    # shopping promos, affiliate try-on hauls, meme-joke fashion posts».
    # Используется как positive/negative anchor для similarity-gate при
    # immediate-fetch и для фильтра в report pipeline.
    intent_positive: str = Field(default="")
    intent_negative: str = Field(default="")


class AntifakeResult(BaseModel):
    misleading: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(default="")


# --------------------- helpers ---------------------


def _extract_json(text: str) -> dict | list | None:
    """Вытаскивает первый JSON-объект/массив из текста. Claude иногда оборачивает в ```json."""
    if not text:
        return None
    t = text.strip()
    # Попытка 1: весь текст — JSON.
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Попытка 2: достать из fenced code block.
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Попытка 3: первый { ... } жадно.
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


async def _call_claude(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 800,
    temperature: float = 0.4,
) -> str:
    client = _get_client()
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


# --------------------- 1. analyze_onboarding (Sonnet, 1 раз на юзера) -------


_CLUSTER_NAMES_STR = ", ".join(settings.topic_clusters.keys())


async def analyze_onboarding(tweets: Sequence[RawTweet]) -> OnboardingAnalysis:
    """Смотрит на твиты подписок, предлагает гипотезу и задаёт 2 уточняющих вопроса."""
    fallback = OnboardingAnalysis(
        hypothesis="Похоже, тебе интересны технологии и общий новостной контекст.",
        cluster_weights={"tech": 0.5, "news": 0.5},
        questions=[
            "Чего тебе сейчас хочется больше — свежих технических новостей или более лёгкого контента (мемы, культура)?",
            "Есть темы, которые точно не надо присылать? (политика, крипта, спорт — что-то из этого?)",
        ],
    )
    if not tweets:
        return fallback

    # Сокращаем каждый твит до 500 символов — экономим токены.
    sample = []
    for t in tweets[:50]:
        text = (t.text or "")[:500]
        sample.append(f"@{t.author_username}: {text}")
    tweets_block = "\n---\n".join(sample)

    system = (
        "Ты помогаешь настраивать персональную ленту новостей из X. "
        "Анализируешь твиты аккаунтов, на которые подписан пользователь, и предполагаешь его интересы. "
        "Отвечаешь ТОЛЬКО валидным JSON без преамбулы."
    )
    user = (
        f"Доступные кластеры тем: {_CLUSTER_NAMES_STR}.\n\n"
        f"Твиты подписок пользователя:\n{tweets_block}\n\n"
        "Верни JSON строго по схеме:\n"
        "{\n"
        '  "hypothesis": "краткая гипотеза (1-2 предложения на русском) — что интересно пользователю",\n'
        '  "cluster_weights": {"<cluster>": 0.0..1.0, ...},  // только релевантные кластеры, сумма ≈ 1\n'
        '  "questions": ["вопрос 1", "вопрос 2"]  // 2 коротких вопроса на русском, чтобы уточнить интересы\n'
        "}"
    )

    try:
        text = await _call_claude(settings.model_sonnet, system, user, max_tokens=800, temperature=0.5)
        data = _extract_json(text)
        if not data:
            log.warning("analyze_onboarding: no JSON in reply")
            return fallback
        return OnboardingAnalysis.model_validate(data)
    except (ValidationError, Exception) as e:
        log.exception("analyze_onboarding failed: %s", e)
        return fallback


# --------------------- 2. summarize (Haiku, кэшируется) ---------------------


async def summarize(text: str, author: str | None = None) -> str:
    """Краткое русскоязычное саммари твита (1-2 предложения)."""
    fallback = (text[:180] + "…") if len(text) > 180 else text
    if not text or not text.strip():
        return fallback

    system = (
        "Ты кратко пересказываешь твиты на русском. "
        "1-2 предложения, без воды, без хештегов, без ссылок. "
        "Сохраняешь суть и смысл. Никаких преамбул типа 'в твите говорится'."
    )
    user = f"Твит от @{author or 'автор'}:\n{text}\n\nКраткий пересказ на русском:"
    try:
        reply = await _call_claude(settings.model_haiku, system, user, max_tokens=220, temperature=0.3)
        out = reply.strip().strip('"').strip()
        return out or fallback
    except Exception as e:
        log.warning("summarize failed: %s", e)
        return fallback


# --------------------- 3. antifake_check (Haiku, только для подозрительных) ---


async def antifake_check(text: str) -> AntifakeResult:
    """Возвращает вероятность что текст вводит в заблуждение (0..1)."""
    fallback = AntifakeResult(misleading=0.0, reason="")
    if not text:
        return fallback

    system = (
        "Ты оцениваешь, насколько пост в X вводит в заблуждение. "
        "Учитывай: громкие утверждения без источника, фабрикованная статистика, "
        "сенсационный тон, conspiracy-маркеры. Сатира и мнение ≠ введение в заблуждение. "
        "Отвечаешь ТОЛЬКО JSON."
    )
    user = (
        f"Пост:\n{text}\n\n"
        'Верни JSON: {"misleading": 0.0..1.0, "reason": "краткая причина"}'
    )
    try:
        reply = await _call_claude(settings.model_haiku, system, user, max_tokens=160, temperature=0.2)
        data = _extract_json(reply)
        if not data:
            return fallback
        return AntifakeResult.model_validate(data)
    except Exception as e:
        log.warning("antifake_check failed: %s", e)
        return fallback


# --------------------- 4. process_preference_request (Haiku) ---------------


class PreferenceRequestResultWithQueries(BaseModel):
    boost: list[str] = Field(default_factory=list)
    suppress: list[str] = Field(default_factory=list)
    reply: str = Field(...)
    # Конкретные поисковые запросы на английском — для search_tweet / search_users
    # Например «хочу больше моды что в тренде» → ['fashion week', 'streetwear trends'].
    search_queries: list[str] = Field(default_factory=list)
    # X-хендлы конкретных аккаунтов по теме запроса — без @ (например "voguejapan",
    # "nippontrends"). Их модель подбирает сама из своего знания X-инфлюенсеров и
    # изданий. Мы их проверяем через get_author_info и добавляем в FollowedAuthor.
    suggested_accounts: list[str] = Field(default_factory=list)


async def process_preference_request(
    text: str,
    current_weights: dict[str, float],
) -> PreferenceRequestResult:
    """Пользователь пишет 'хочу больше X, меньше Y' — возвращаем какие кластеры поднять/снизить."""
    fallback = PreferenceRequestResult(
        boost=[], suppress=[],
        reply="Понял, постараюсь учесть. Если захочешь точнее — опиши подробнее.",
    )
    if not text or not text.strip():
        return fallback

    system = (
        "Ты помогаешь настроить персональную новостную ленту пользователя. "
        "Твоя задача — ГЛУБОКО понять что именно юзер имел в виду под своей фразой, "
        "и превратить это в структурированные сигналы для ранжирования.\n\n"
        "Ключевой принцип: интерпретируй намерение, а не буквальные слова. "
        "«хочу больше моды» может значить editorial fashion (Met Gala, runway "
        "reviews, street style), а может — retail shopping — но это РАЗНЫЕ вещи, "
        "и тот же текст поста попадает в разные категории. Пиши intent_positive "
        "и intent_negative так, чтобы они работали как embedding-anchor: "
        "развёрнутое описание 2-4 предложений на английском с примерами того "
        "что включается и что исключается.\n\n"
        "КРИТИЧНО про search_queries:\n"
        "- МАКСИМУМ 3 слова в запросе. X SearchTimeline падает с 404 на длинных queries.\n"
        "- Без года, кавычек, операторов (OR/AND).\n"
        "- Конкретные термы: «japanese streetwear», не «fashion». "
        "«claude code», не «AI tools».\n"
        "- Если намерение многогранное — 3-5 отдельных коротких запросов, "
        "каждый под свой аспект.\n\n"
        "Про suggested_accounts: РЕАЛЬНЫЕ X-аккаунты (без @) которые публикуют "
        "по теме — из своих знаний о профессиональных изданиях, авторах, "
        "комьюнити. Не выдумывай, не путай с Instagram/TikTok.\n\n"
        "Про boost/suppress: это слоты для весов тематики. Выбирай из списка "
        "ниже только если тема РЕАЛЬНО попадает в слот. Если запрос уникален и "
        "ни один слот не подходит идеально — оставь boost пустым и положись на "
        "intent_positive + search_queries + suggested_accounts, этого хватит "
        "для ранжирования. Не натягивай.\n\n"
        "Отвечаешь ТОЛЬКО JSON."
    )
    user = (
        f"Слоты весов (необязательные, только если реально попадает): {_CLUSTER_NAMES_STR}\n\n"
        f"Текущие веса: {json.dumps(current_weights, ensure_ascii=False)}\n\n"
        f"Запрос пользователя: «{text}»\n\n"
        "Верни JSON:\n"
        "{\n"
        '  "intent_positive": "English 2-4 sentences describing EXACTLY what content the user wants. Give concrete examples of subjects, angles, post styles that qualify. Be vivid — this string is used as an embedding anchor for similarity ranking.",\n'
        '  "intent_negative": "English 2-4 sentences describing what content from the same general topic the user does NOT want. Concrete counter-examples. Used as negative anchor to push out off-target posts.",\n'
        '  "boost": ["<slot>", ...],\n'
        '  "suppress": ["<slot>", ...],\n'
        '  "reply": "короткий ответ пользователю на русском — что именно учтено",\n'
        '  "search_queries": ["2-3 слова", ...],\n'
        '  "suggested_accounts": ["user1", "user2", ...],\n'
        '  "blocked_accounts": ["username1", ...]\n'
        "}\n\n"
        "Пример для запроса «хочу больше моды как одеваются обычные парни»:\n"
        '  "intent_positive": "Editorial menswear and everyday-style coverage: '
        'candid street style photography, Met Gala and runway show reviews, '
        'GQ/Esquire-style menswear commentary analysing silhouettes, fit breakdowns '
        'and designer drops from working fashion writers and stylists.",\n'
        '  "intent_negative": "Retail affiliate content like \\"save this for outfit '
        'inspo\\", influencer try-on hauls with shop-now links, meme jokes poking '
        'fun at a passing trend with short video, brand promo posts announcing new '
        'arrivals or percent-off deals."'
    )
    try:
        reply = await _call_claude(settings.model_sonnet, system, user, max_tokens=600, temperature=0.3)
        data = _extract_json(reply)
        if not data:
            return fallback
        result = PreferenceRequestResult.model_validate(data)

        def _clean_handles(raw):
            out = []
            for u in list(raw or []):
                if not isinstance(u, str):
                    continue
                u = u.strip().lstrip("@").strip()
                if u and re.fullmatch(r"[A-Za-z0-9_]{1,15}", u):
                    out.append(u)
            return out

        result.suggested_accounts = _clean_handles(result.suggested_accounts)
        result.blocked_accounts = _clean_handles(result.blocked_accounts)

        # Хард-лимит на queries: даже если Claude проигнорировал промпт —
        # обрезаем до 3 слов локально (см. _shorten_query про X 404 на длинных).
        result.search_queries = [
            _shorten_query(q) for q in (result.search_queries or [])
            if isinstance(q, str) and q.strip()
        ]
        result.search_queries = [q for q in result.search_queries if q]

        valid = set(settings.topic_clusters.keys())
        result.boost = [c for c in result.boost if c in valid]
        result.suppress = [c for c in result.suppress if c in valid]
        return result
    except Exception as e:
        log.warning("process_preference_request failed: %s", e)
        return fallback


# --------------------- 5x. name_topic (Haiku, быстро) ---------------------------


class TopicName(BaseModel):
    emoji: str = Field(..., description="один эмодзи-символ")
    name: str = Field(..., description="короткое имя темы на русском, 1-4 слова")


async def name_topic(sample_texts: list[str]) -> tuple[str, str]:
    """Читаем 3-6 твитов из одного кластера, возвращаем (emoji, короткое имя).

    Эмодзи — контекстный, имя — конкретное (не «tech», а «Opus 4.7 / Sora 2»).
    """
    fallback = ("📰", "Разное")
    if not sample_texts:
        return fallback

    joined = "\n---\n".join(t[:400] for t in sample_texts[:10])
    system = (
        "Ты коротко именуешь тему по примерам постов из X. Даёшь один эмодзи + "
        "конкретное имя на русском 2-5 слов.\n\n"
        "КРИТИЧНО: имя должно описывать О ЧЁМ ПОСТЫ НА САМОМ ДЕЛЕ, а не о чём "
        "они МОГУТ БЫТЬ по названию бренда/продукта:\n"
        "- Если посты про рост подписчиков/метрик аккаунта @ClaudeDevs — имя "
        "«Рост Claude Dev аудитории», НЕ «Claude релиз».\n"
        "- Если посты про апдейт версии (bugfix, patch) — «Claude Code 2.1 bugfix», "
        "НЕ общее «Claude релиз».\n"
        "- Если 7 из 10 постов про запуск Claude Design, а 3 упоминают Opus 4.7 "
        "как движок — имя «Claude Design релиз».\n"
        "- Если посты про реакции/скепсис к уже вышедшему продукту — «Opus 4.7 "
        "первые впечатления», не «Opus 4.7 релиз».\n"
        "- Не обобщай до «AI новости» / «Tech апдейт» — пиши конкретный предмет "
        "обсуждения.\n"
        "- Если посты разнородные и общей нити нет — честно: «Claude — разное».\n"
        "Отвечаешь ТОЛЬКО JSON."
    )
    user = (
        f"Посты по теме:\n{joined}\n\n"
        "Верни JSON:\n"
        '{"emoji": "🤖", "name": "точное имя 2-5 слов"}'
    )
    try:
        reply = await _call_claude(settings.model_haiku, system, user, max_tokens=80, temperature=0.3)
        data = _extract_json(reply)
        if not data:
            return fallback
        t = TopicName.model_validate(data)
        emoji = (t.emoji or "📰")[:2]  # обрезаем если вдруг несколько символов
        name = (t.name or "Разное").strip().strip('"').strip("«»")[:60]
        return emoji, name
    except Exception as e:
        log.warning("name_topic failed: %s", e)
        return fallback


# --------------------- 5y. summarize_discussion (Haiku) ----------------------


async def summarize_discussion(texts: list[str], topic_name: str = "") -> str:
    """«Grok Stories»-style — краткий дайджест темы. 3-5 предложений, строго по фактам.

    Жёсткие инварианты промпта:
      - НИЧЕГО не выдумывать. Только то что буквально есть в постах.
      - НЕ экстраполировать («раз X сказал Y, значит Z» — запрещено).
      - Противоречащие сведения → прямо указать что есть спор.
      - Без преамбул.

    Это исправление после случая когда Haiku написал «Anthropic убрал 5-часовые
    лимиты», а в постах было «hit the limits with just 3 prompts» — прямое
    противоречие, неприемлемо.
    """
    if not texts:
        return ""
    joined = "\n\n---POST---\n".join(t[:600] for t in texts[:10])

    system = (
        "Ты делаешь фактический дайджест постов из X. "
        "СТРОГО следуй правилам:\n"
        "1. Пиши ТОЛЬКО то что буквально сказано в постах. Никаких домыслов, "
        "экстраполяций, «это означает что».\n"
        "2. ДАЙДЖЕСТ ДОЛЖЕН БЫТЬ ПРО УКАЗАННУЮ ТЕМУ. Если тема «Claude Opus 4.7 "
        "релиз», а посты в основном про Claude Design (другой продукт, использующий "
        "Opus 4.7 как движок) — ты работаешь не с теми постами, в этом случае "
        "напиши: «посты не про релиз 4.7 как таковой, обсуждают <что там на самом "
        "деле>». Не притягивай чужую тему к заголовку.\n"
        "3. Если два поста противоречат друг другу — ЧЕСТНО скажи: «в постах спор/"
        "противоречие, одни говорят X, другие Y».\n"
        "4. Если ты не уверен что факт есть в посте — НЕ пиши его.\n"
        "5. 3-5 предложений на русском.\n"
        "6. Без преамбул («в этих постах…», «тема обсуждает…»). Сразу по делу.\n"
        "7. Цифры, имена, продукты — только те что БУКВАЛЬНО встречаются в текстах."
    )
    header = f"Название темы: {topic_name}\n\n" if topic_name else ""
    user = f"{header}Посты:\n---POST---\n{joined}\n\nФактический дайджест про эту тему (только то что в постах):"
    try:
        reply = await _call_claude(settings.model_haiku, system, user, max_tokens=500, temperature=0.15)
        return reply.strip().strip('"').strip()
    except Exception as e:
        log.warning("summarize_discussion failed: %s", e)
        return ""


# --------------------- 5a. analyze_interests_text (Sonnet, 1 раз на юзера) ----


async def analyze_interests_text(text: str) -> OnboardingAnalysis:
    """Анализ интересов из свободного описания (fallback когда X-скрапинг недоступен).

    Пользователь пишет в духе «интересуюсь AI, стартапами, иногда крипта, политика
    не надо» — возвращаем ту же структуру что и analyze_onboarding, чтобы дальше
    идти одним путём.
    """
    fallback = OnboardingAnalysis(
        hypothesis="Отметил твои интересы — постараюсь подобрать ленту под них.",
        cluster_weights={"tech": 0.4, "ai": 0.4, "news": 0.3},
        questions=[
            "Насколько глубокий контент хочешь — длинные анализы или короткие апдейты?",
            "Есть темы, которые точно не присылать?",
        ],
    )
    if not text or not text.strip():
        return fallback

    system = (
        "Ты анализируешь текстовое описание интересов пользователя и настраиваешь "
        "персональную ленту новостей из X. Выделяешь темы, задаёшь 2 уточнения. "
        "Отвечаешь ТОЛЬКО JSON без преамбулы."
    )
    user = (
        f"Доступные кластеры: {_CLUSTER_NAMES_STR}.\n\n"
        f"Пользователь описал свои интересы:\n«{text.strip()}»\n\n"
        "Верни JSON строго по схеме:\n"
        "{\n"
        '  "hypothesis": "1-2 предложения на русском — как ты понял его интересы",\n'
        '  "cluster_weights": {"<cluster>": 0.0..1.0, ...},\n'
        '  "questions": ["уточнение 1", "уточнение 2"]\n'
        "}"
    )
    try:
        reply = await _call_claude(settings.model_sonnet, system, user, max_tokens=700, temperature=0.4)
        data = _extract_json(reply)
        if not data:
            return fallback
        result = OnboardingAnalysis.model_validate(data)
        valid = set(settings.topic_clusters.keys())
        result.cluster_weights = {k: v for k, v in result.cluster_weights.items() if k in valid}
        if not result.cluster_weights:
            result.cluster_weights = {"tech": 0.4, "news": 0.3}
        return result
    except Exception as e:
        log.warning("analyze_interests_text failed: %s", e)
        return fallback


# --------------------- 5. translate_to_ru (Haiku, кэшируется) ---------------


async def translate_batch(texts: Sequence[str], target: str = "ru") -> list[str]:
    """Параллельный перевод списка текстов. 7 реплаев параллельно через asyncio
    — ~5с суммарно, вместо 30с последовательно. Падение отдельного перевода
    возвращает оригинал (не блокирует остальные)."""
    if not texts:
        return []
    coros = [translate_to_ru(t) if target == "ru" else translate_to_en(t) for t in texts]
    results = await asyncio.gather(*coros, return_exceptions=True)
    out: list[str] = []
    for orig, res in zip(texts, results):
        if isinstance(res, Exception) or not res:
            out.append(orig)
        else:
            out.append(res)
    return out


async def translate_to_en(text: str) -> str:
    """Обратный перевод — редко, но надо для симметрии (user clicks 🇬🇧 if he
    изначально видел уже переведенную версию)."""
    if not text or not text.strip():
        return text
    system = (
        "Translate posts from Russian to English. Preserve tone, "
        "emphasis, meaning. Translate naturally, not word-for-word. "
        "NO preamble, NO quotes, NO explanations — clean translation only."
    )
    user = f"Translate to English:\n\n{text}"
    try:
        reply = await _call_claude(settings.model_haiku, system, user, max_tokens=700, temperature=0.3)
        return reply.strip().strip('"').strip("«»").strip() or text
    except Exception as e:
        log.warning("translate_to_en failed: %s", e)
        return text


async def translate_to_ru(text: str) -> str:
    """Перевод поста на русский. Сохраняет тон, без преамбулы."""
    if not text or not text.strip():
        return text

    system = (
        "Переводишь посты из X на русский. "
        "Сохраняешь тон, акценты, смысл. Переводишь живо — не дословно, "
        "а так, как бы это написал русскоязычный автор в том же жанре. "
        "БЕЗ преамбулы, БЕЗ кавычек, БЕЗ пояснений — только чистый перевод."
    )
    user = f"Переведи на русский:\n\n{text}"
    try:
        reply = await _call_claude(settings.model_haiku, system, user, max_tokens=700, temperature=0.3)
        out = reply.strip().strip('"').strip("«»").strip()
        return out or text
    except Exception as e:
        log.warning("translate_to_ru failed: %s", e)
        return text


# --------------------- 6. discuss_post (Haiku) -----------------------------


async def discuss_post(post_text: str, author: str, user_question: str) -> str:
    """Пользователь reply'ит на пост с вопросом — отвечаем по контексту поста.

    Модель видит пост целиком и вопрос, отвечает коротко и по существу на русском.
    """
    if not user_question or not user_question.strip():
        return "Напиши вопрос развёрнутее — что именно обсудим?"

    system = (
        "Ты обсуждаешь с пользователем пост из X. "
        "Отвечаешь на русском, 2-5 предложений, по существу — без воды и преамбул. "
        "Если вопрос просит мнение — даёшь своё прямо. "
        "Если нужны свежие данные, которых у тебя нет (цены, метрики в реальном времени) — "
        "честно говоришь об этом. "
        "Если в посте что-то фактически некорректно — указываешь на это. "
        "Не копируешь вопрос пользователя, сразу отвечаешь."
    )
    user = (
        f"Пост от @{author}:\n«{post_text}»\n\n"
        f"Вопрос пользователя: {user_question.strip()}\n\n"
        f"Твой ответ:"
    )
    try:
        reply = await _call_claude(
            settings.model_haiku, system, user, max_tokens=600, temperature=0.55
        )
        out = reply.strip()
        return out or "Хм, не получилось сформулировать ответ. Попробуй переформулировать вопрос."
    except Exception as e:
        log.warning("discuss_post failed: %s", e)
        return "Что-то пошло не так с разбором поста. Попробуй ещё раз через минуту."


# --------------------- refinement после ответов на вопросы -------------------


async def group_super_topics(
    subtopics: list[tuple[int, str, str, int]],  # [(id, emoji, name, post_count), ...]
) -> list[dict]:
    """Группирует мелкие кластеры в широкие супер-категории.

    Вход: список sub-topics (Claude Design релиз, Building Effective Agents,
    японский стиль, streetwear Tokyo, Reuters sports, FISA расширение).
    Выход: `[{"emoji": "💻", "name": "Технологии", "sub_ids": [0, 2, 5]}, ...]`.

    Названия супер-категорий НЕ захардкожены — Claude придумывает их сам
    исходя из того что в под-темах. Если юзер читает про F1/watches/cooking —
    появятся «Гонки», «Часы», «Кулинария». Список фиксированных ярлыков
    ломает эту гибкость.
    """
    if not subtopics:
        return []
    pairs = "\n".join(
        f"  {sid}. {emoji} {name} · {cnt} постов"
        for sid, emoji, name, cnt in subtopics
    )
    system = (
        "Ты группируешь мелкие темы дня в широкие супер-категории для навигации "
        "по новостному боту. Названия супер-категорий ПРИДУМЫВАЙ САМ из смысла "
        "под-тем — не используй шаблонный список, не натягивай под-темы на "
        "заранее заданные ярлыки.\n\n"
        "Правила группировки:\n"
        "1. Каждая под-тема попадает РОВНО в одну супер-категорию.\n"
        "2. Группируй по аудитории: «что за читатель придёт сюда кликнуть?». "
        "Если одна и та же аудитория не читает две темы одновременно — не "
        "сливай их. Типичные разъезды: торговля финансовыми активами "
        "(крипто, акции, сырьё) ≠ создание технологий (софт, железо, "
        "исследования) ≠ бизнес-сделки и макро ≠ стиль одежды ≠ культура "
        "(музыка/кино) ≠ спорт ≠ политика ≠ наука. Одна категория = одна "
        "аудитория.\n"
        "3. Супер-категория с 2+ под-темами — нормальный кейс. Супер-категория "
        "с 1 под-темой разрешена ТОЛЬКО если в этой под-теме ≥5 постов "
        "(важная самостоятельная тема дня). Мелкие одиночки (<5 постов) "
        "скидывай в «Прочее».\n"
        "4. Название супер-категории — 1-3 слова на русском, эмодзи 1 символ. "
        "Пиши конкретно: «Технологии», «Крипто-трейдинг», «Мода», «Большая "
        "политика», «Мировой спорт» — а не общее «Новости».\n\n"
        "Отвечай ТОЛЬКО JSON."
    )
    user_msg = (
        f"Под-темы:\n{pairs}\n\n"
        "Верни JSON:\n"
        "{\n"
        '  "super_topics": [\n'
        '    {"emoji": "<1 символ>", "name": "<1-3 слова>", "sub_ids": [<int>, ...]},\n'
        "    ...\n"
        "  ]\n"
        "}"
    )
    try:
        reply = await _call_claude(settings.model_haiku, system, user_msg,
                                   max_tokens=500, temperature=0.2)
        data = _extract_json(reply)
        if not data or "super_topics" not in data:
            return []
        st = data["super_topics"]
        if not isinstance(st, list):
            return []
        valid_ids = {sid for sid, _, _, _ in subtopics}
        out = []
        seen_ids: set[int] = set()
        for item in st:
            if not isinstance(item, dict):
                continue
            emoji = (item.get("emoji") or "📰").strip()[:2]
            name = (item.get("name") or "Разное").strip()[:40]
            raw_ids = item.get("sub_ids") or []
            clean_ids = []
            for x in raw_ids:
                try:
                    xi = int(x)
                    if xi in valid_ids and xi not in seen_ids:
                        clean_ids.append(xi)
                        seen_ids.add(xi)
                except (TypeError, ValueError):
                    continue
            if clean_ids:
                out.append({"emoji": emoji, "name": name, "sub_ids": clean_ids})
        # Singleton super-категории разрешены если в единственной под-теме
        # ≥5 постов — это самостоятельная крупная тема дня (напр. «Мода»
        # когда у юзера 8 постов про стиль, а остальные кластеры смешанные).
        # Мелкие одиночки (<5 постов) скидываем в «Прочее».
        subtopic_counts = {sid: cnt for sid, _, _, cnt in subtopics}
        misc_ids: list[int] = []
        filtered_out: list[dict] = []
        for item in out:
            if len(item["sub_ids"]) >= 2:
                filtered_out.append(item)
                continue
            lone_id = item["sub_ids"][0]
            if subtopic_counts.get(lone_id, 0) >= 5:
                filtered_out.append(item)
            else:
                misc_ids.append(lone_id)
        out = filtered_out
        # Неучтённые под-темы — тоже в «Прочее».
        leftover = [sid for sid in valid_ids if sid not in seen_ids]
        misc_ids.extend(leftover)
        if misc_ids:
            # Если «Прочее» уже есть — дополняем, иначе создаём.
            misc_existing = next(
                (it for it in out if it["name"].lower() in ("разное", "прочее")),
                None,
            )
            if misc_existing:
                for mid in misc_ids:
                    if mid not in misc_existing["sub_ids"]:
                        misc_existing["sub_ids"].append(mid)
            else:
                out.append({"emoji": "📌", "name": "Прочее", "sub_ids": misc_ids})
        return out
    except Exception as e:
        log.warning("group_super_topics failed: %s", e)
        return []


async def suggest_interest_queries(
    cluster_weights: dict[str, float],
    saved_queries: list[str],
    followed_authors: list[str],
    max_queries: int = 3,
) -> list[str]:
    """Даём Claude сырые сигналы — веса кластеров, явные saved-queries юзера,
    список кого он фолловит — и просим вернуть КОНКРЕТНЫЕ поисковые запросы
    на английском под то что нужно этому юзеру СЕЙЧАС.

    Замена старого захардкоженного маппинга `tech → "tech startups launch"`.
    Теперь запросы индивидуальные: если юзер просил «японский стиль» и
    фолловит @voguejapan — модель вернёт `japanese streetwear`, а не
    generic `fashion trends`.
    """
    fallback: list[str] = []
    if saved_queries:
        fallback = [_shorten_query(q) for q in saved_queries if q][:max_queries]

    system = (
        "Ты подбираешь X-поисковые запросы для персонального news-дайджеста.\n"
        "На вход — сигналы интересов пользователя (веса тем, прямые фразы, "
        "на кого подписан).\n"
        "КРИТИЧНО про формат запроса:\n"
        "- МАКСИМУМ 3 слова в запросе. Длинные запросы («Paris Fashion Week 2025 "
        "haute couture collections») X отвечает 404 — SearchTimeline не работает "
        "с length > 3-4 слов стабильно.\n"
        "- Конкретные термы: «japanese streetwear», «claude code», «dyson sphere». "
        "НЕ generic: «fashion», «tech» (слишком широко).\n"
        "- На английском.\n"
        "- Без кавычек, хештегов, операторов.\n"
        "Отвечаешь ТОЛЬКО JSON."
    )
    user_msg = (
        f"Веса кластеров: {json.dumps(cluster_weights, ensure_ascii=False)}\n\n"
        f"Прямые запросы юзера (история «хочу больше …»): "
        f"{json.dumps(saved_queries, ensure_ascii=False)}\n\n"
        f"Подписан на (топ): {', '.join(followed_authors[:40])}\n\n"
        f'Верни JSON: {{"queries": ["2-3 слова", "2-3 слова", "2-3 слова"]}}'
    )
    try:
        reply = await _call_claude(settings.model_haiku, system, user_msg,
                                   max_tokens=250, temperature=0.3)
        data = _extract_json(reply)
        if not data:
            return fallback
        qs = data.get("queries") or []
        out = []
        for q in qs:
            if isinstance(q, str) and q.strip():
                out.append(_shorten_query(q))
            if len(out) >= max_queries:
                break
        return out or fallback
    except Exception as e:
        log.warning("suggest_interest_queries failed: %s", e)
        return fallback


_AUTHORS_FOR_QUERY_CACHE: dict[str, tuple[float, list[str]]] = {}
_AUTHORS_FOR_QUERY_TTL = 3600.0  # 1ч — авторы по теме меняются редко


async def score_clusters_against_interests(
    cluster_names: list[str],
    user_interests: list[str],
) -> list[float]:
    """Один вызов Haiku: даём список имён кластеров + явные запросы юзера,
    получаем 0..1 score для каждого кластера. Заменяет embedding-similarity
    которая путает «Claude Design» с «fashion design».

    Возвращает список длины len(cluster_names). При фейле — нули."""
    n = len(cluster_names)
    if n == 0 or not user_interests:
        return [0.0] * n
    enumerated = "\n".join(f"  {i}. {nm}" for i, nm in enumerate(cluster_names))
    interests_str = ", ".join(f"«{q}»" for q in user_interests[:8])
    system = (
        "Ты оцениваешь релевантность тем новостной ленты явным интересам "
        "пользователя. Возвращаешь массив чисел 0..1 для каждой темы:\n"
        "  1.0 — тема прямо про интерес юзера («Streetwear Tokyo» при интересе "
        "«japanese fashion» = 1.0)\n"
        "  0.5 — смежно («Дизайн интерьеров» при интересе «fashion» = 0.5)\n"
        "  0.0 — не связано («Claude Design релиз» — это AI продукт, не fashion = 0.0)\n"
        "Главное правило: НЕ путай омонимы и общие слова («design», «style», "
        "«art», «culture») — если тема явно про другую сферу (AI, политика, "
        "крипта), score = 0 даже если есть лексическое пересечение.\n"
        "Отвечаешь ТОЛЬКО JSON."
    )
    user_msg = (
        f"Интересы пользователя: {interests_str}\n\n"
        f"Темы:\n{enumerated}\n\n"
        f'Верни JSON: {{"scores": [0.0, 1.0, 0.5, ...]}} — РОВНО {n} чисел в порядке тем.'
    )
    try:
        reply = await _call_claude(settings.model_haiku, system, user_msg,
                                   max_tokens=300, temperature=0.1)
        data = _extract_json(reply)
        if not data:
            return [0.0] * n
        raw = data.get("scores") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return [0.0] * n
        out: list[float] = []
        for v in raw[:n]:
            try:
                out.append(max(0.0, min(1.0, float(v))))
            except (TypeError, ValueError):
                out.append(0.0)
        while len(out) < n:
            out.append(0.0)
        return out
    except Exception as e:
        log.warning("score_clusters_against_interests failed: %s", e)
        return [0.0] * n


async def suggest_authors_for_query(query: str, n: int = 6) -> list[str]:
    """Claude подбирает X-аккаунты под произвольный запрос темы.

    Заменяет хардкод-словари вида `{"football": ["espn", ...], ...}` —
    которые не масштабируются на «лоу-фай джаз», «античная философия»,
    «реставрация мебели» и любые нишевые темы юзеров.

    Кэш на 1ч в памяти процесса — для одного и того же запроса не зовём
    Claude повторно. Кэш per-process, рестарт обнуляет.
    """
    import time as _time
    key = (query or "").lower().strip()
    if not key:
        return []
    entry = _AUTHORS_FOR_QUERY_CACHE.get(key)
    if entry and entry[0] > _time.monotonic():
        return list(entry[1])

    system = (
        "Ты подбираешь X (Twitter) аккаунты которые активно публикуют по "
        "указанной теме. Возвращаешь только реальных авторов которые ты "
        "ТОЧНО знаешь существуют в X — издания, журналистов, исследователей, "
        "комьюнити. Не путай с Instagram/TikTok/Substack — нужны именно X "
        "хендлы. Без @, только username. Отвечаешь ТОЛЬКО JSON."
    )
    user_msg = (
        f"Тема/запрос: «{query.strip()}»\n\n"
        f"Верни JSON: {{\"handles\": [\"user1\", \"user2\", ...]}}\n"
        f"Лимит: {n} аккаунтов. Приоритет — крупные/проверенные источники "
        f"и известные эксперты по теме. Если тема нишевая — нишевые издания "
        f"и микро-инфлюенсеры допустимы."
    )
    out: list[str] = []
    try:
        reply = await _call_claude(settings.model_haiku, system, user_msg,
                                   max_tokens=300, temperature=0.3)
        data = _extract_json(reply)
        if data and isinstance(data, dict):
            for h in (data.get("handles") or []):
                if isinstance(h, str):
                    h = h.strip().lstrip("@").strip()
                    if h and re.fullmatch(r"[A-Za-z0-9_]{1,15}", h) and h not in out:
                        out.append(h)
                if len(out) >= n:
                    break
    except Exception as e:
        log.warning("suggest_authors_for_query(%r) failed: %s", query, e)

    _AUTHORS_FOR_QUERY_CACHE[key] = (_time.monotonic() + _AUTHORS_FOR_QUERY_TTL, out)
    return out


def _shorten_query(q: str, max_words: int = 3) -> str:
    """X SearchTimeline стабильно работает на 2-3 словах. 5+ слов → 404.

    Обрезаем по словам, убираем года/кавычки/операторы. Стоп-слова типа «the», «a»
    выкидываем чтобы уложиться в лимит без потери смысла.
    """
    if not q:
        return ""
    # Чистим: года-4-цифры, кавычки, операторы типа OR / AND.
    cleaned = re.sub(r"\b(19|20)\d{2}\b", "", q)
    cleaned = re.sub(r"['\"]", "", cleaned)
    cleaned = re.sub(r"\b(OR|AND|NOT)\b", "", cleaned, flags=re.IGNORECASE)
    words = [w for w in cleaned.split() if w]
    stop = {"the", "a", "an", "of", "in", "on", "and", "or", "to", "for", "with"}
    kept = [w for w in words if w.lower() not in stop]
    if not kept:
        kept = words
    return " ".join(kept[:max_words]).strip()


async def process_onboarding_answers(
    hypothesis: str,
    questions: list[str],
    answers: list[str],
    initial_weights: dict[str, float],
) -> OnboardingRefinement:
    """После того как пользователь ответил на вопросы — уточняем cluster_weights."""
    fallback = OnboardingRefinement(
        cluster_weights=initial_weights or {"tech": 0.5, "news": 0.5},
        reply="Отлично, я настроил твою ленту. Сейчас пришлю первую подборку.",
    )
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions, answers))
    system = (
        "Ты финализируешь настройку персональной ленты. "
        "На основе гипотезы и ответов пользователя корректируешь веса кластеров. "
        "Отвечаешь ТОЛЬКО JSON."
    )
    user = (
        f"Доступные кластеры: {_CLUSTER_NAMES_STR}\n"
        f"Начальная гипотеза: {hypothesis}\n"
        f"Начальные веса: {json.dumps(initial_weights, ensure_ascii=False)}\n\n"
        f"Ответы пользователя:\n{qa}\n\n"
        "Верни JSON:\n"
        "{\n"
        '  "cluster_weights": {"<cluster>": 0.0..1.0, ...},\n'
        '  "reply": "тёплый ответ на русском (1-2 предложения) — что настроено"\n'
        "}"
    )
    try:
        reply = await _call_claude(settings.model_haiku, system, user, max_tokens=400, temperature=0.4)
        data = _extract_json(reply)
        if not data:
            return fallback
        result = OnboardingRefinement.model_validate(data)
        valid = set(settings.topic_clusters.keys())
        result.cluster_weights = {k: v for k, v in result.cluster_weights.items() if k in valid}
        if not result.cluster_weights:
            result.cluster_weights = initial_weights or {"tech": 0.5, "news": 0.5}
        return result
    except Exception as e:
        log.warning("process_onboarding_answers failed: %s", e)
        return fallback
