"""Настройки приложения. Читается из .env через python-dotenv.

ProxyAPI (https://proxyapi.ru/docs/overview) предоставляет единый ключ
для Anthropic и OpenAI endpoint'ов:
  - OpenAI compat:    https://api.proxyapi.ru/openai/v1
  - Anthropic compat: https://api.proxyapi.ru/anthropic/v1
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# override=True — чтобы локальный .env выигрывал у системных переменных окружения
# (на dev-машинах часто торчит ANTHROPIC_BASE_URL=https://api.anthropic.com от тулов
# типа Claude Code, и он бы перебил наш прокси).
load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent


def _get(name: str, default: str | None = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"ENV var {name} is required but not set")
    return val or ""


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    # Telegram
    telegram_bot_token: str = field(default_factory=lambda: _get("TELEGRAM_BOT_TOKEN", required=True))

    # ProxyAPI — один ключ на оба провайдера
    proxyapi_key: str = field(default_factory=lambda: _get("PROXYAPI_KEY", required=True))
    anthropic_base_url: str = field(default_factory=lambda: _get("ANTHROPIC_BASE_URL", "https://api.proxyapi.ru/anthropic"))
    openai_base_url: str = field(default_factory=lambda: _get("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1"))

    # Модели
    model_haiku: str = field(default_factory=lambda: _get("MODEL_HAIKU", "claude-haiku-4-5"))
    model_sonnet: str = field(default_factory=lambda: _get("MODEL_SONNET", "claude-sonnet-4-5"))
    embedding_model: str = field(default_factory=lambda: _get("EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_dim: int = field(default_factory=lambda: _get_int("EMBEDDING_DIM", 1536))

    # X auth — через twikit (cookie-based к внутреннему GraphQL X).
    # Приоритет источников: файл cookies → env (X_AUTH_TOKEN) → login по паролю.
    # auth_token извлекается из браузера: DevTools → Application → Cookies → x.com
    x_cookies_path: str = field(default_factory=lambda: _get("X_COOKIES_PATH", str(BASE_DIR / "x_cookies.json")))
    x_auth_token: str = field(default_factory=lambda: _get("X_AUTH_TOKEN"))
    x_ct0: str = field(default_factory=lambda: _get("X_CT0"))
    x_username: str = field(default_factory=lambda: _get("X_USERNAME"))
    x_password: str = field(default_factory=lambda: _get("X_PASSWORD"))
    x_email: str = field(default_factory=lambda: _get("X_EMAIL"))
    x_totp_secret: str = field(default_factory=lambda: _get("X_TOTP_SECRET"))

    # Хранилища
    database_url: str = field(default_factory=lambda: _get("DATABASE_URL", "sqlite+aiosqlite:///./bot.db"))
    chroma_path: str = field(default_factory=lambda: _get("CHROMA_PATH", str(BASE_DIR / "chroma_db")))

    # Расписание
    default_delivery_interval_hours: int = field(default_factory=lambda: _get_int("DEFAULT_DELIVERY_INTERVAL_HOURS", 3))
    fetch_interval_minutes: int = field(default_factory=lambda: _get_int("FETCH_INTERVAL_MINUTES", 30))

    # Лимиты
    tweets_per_author_on_fetch: int = 15
    max_authors_per_user: int = 80
    top_n_per_delivery: int = 5
    tweet_max_age_hours: int = 48

    # Параметры рекомендера
    preference_learn_rate_like: float = 0.08
    preference_learn_rate_dislike: float = 0.06  # чуть меньше чтобы дизлайк не перекашивал
    implicit_skip_penalty: float = 0.015
    exploration_ratio: float = 0.20  # 1 из 5 слотов под разведку
    diversity_threshold: float = 0.78
    duplicate_threshold: float = 0.92

    # Кластеры тем — используем как якоря в embedding пространстве.
    # Описания специально развёрнутые — чтобы embedding получился содержательным.
    topic_clusters: dict[str, str] = field(default_factory=lambda: {
        "tech": "technology, software, programming, startups, AI, machine learning, open source, developers",
        "ai": "artificial intelligence, LLM, GPT, Claude, neural networks, AI research, AI safety",
        "crypto": "cryptocurrency, bitcoin, ethereum, defi, web3, blockchain, tokens, trading",
        "science": "scientific research, physics, biology, space, astronomy, experiments, papers",
        "politics": "politics, government, elections, policy, geopolitics, international relations",
        "business": "business, finance, stocks, markets, economy, companies, earnings",
        "memes": "memes, humor, funny, jokes, viral, entertainment",
        "culture": "art, music, film, books, culture, design, creativity",
        "sports": "sports, football, basketball, olympics, athletes, matches",
        "gaming": "gaming, video games, esports, game development, indie games",
        "lifestyle": "lifestyle, fitness, health, productivity, self-improvement, habits",
        "news": "breaking news, world events, current affairs, journalism",
    })


settings = Settings()
