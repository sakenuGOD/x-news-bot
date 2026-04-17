"""SQLAlchemy модели. Все JSON-поля валидируются pydantic'ом на уровне сервисов."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    telegram_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    x_username: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # float[1536] — серверный preference vector, усредняется по лайкам/дизлайкам.
    preference_vector: Mapped[Optional[list[float]]] = mapped_column(JSON, nullable=True)

    # {"tech": 0.4, "memes": 0.2, ...} — веса тематических кластеров (0..1).
    cluster_weights: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)

    delivery_interval_hours: Mapped[int] = mapped_column(Integer, default=3)
    onboarding_done: Mapped[bool] = mapped_column(Boolean, default=False)
    paused: Mapped[bool] = mapped_column(Boolean, default=False)

    # FSM-подобное состояние онбординга, чтобы не поднимать отдельный storage.
    onboarding_state: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    onboarding_payload: Mapped[dict] = mapped_column(JSON, default=dict)

    last_delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # {"topic_name": {"dislikes": N, "last_dislike_ts": iso-string}} — anti-boredom.
    topic_dislikes: Mapped[dict] = mapped_column(JSON, default=dict)

    # ["reuters", "cnn", ...] — юзернеймы (без @, lowercase) авторов которых
    # юзер явно попросил не показывать. Применяется фильтром в build_report.
    blocked_authors: Mapped[list] = mapped_column(JSON, default=list)

    feedback: Mapped[list["Feedback"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Tweet(Base):
    """Кэш твитов. Embedding и summary хранятся навсегда — пересчитывать дорого."""

    __tablename__ = "tweets"

    tweet_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    author_username: Mapped[str] = mapped_column(String(64), index=True)
    author_display_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    text: Mapped[str] = mapped_column(Text)
    url: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    image_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    # Тип прикреплённого медиа: photo / video / animation. None = без медиа.
    media_type: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    # Вторая дорожка — медиа квоты когда у автора СВОЙ media плюс он цитирует
    # чужой твит с media. Шлём 2 telegram-сообщения: автор + квота (см. delivery).
    quote_image_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    quote_media_type: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    quote_author: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    quote_text: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    # Embedding 1536-мерный — как JSON, чтобы был backup (Chroma вторична).
    embedding: Mapped[Optional[list[float]]] = mapped_column(JSON, nullable=True)
    summary_ru: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Доверие к источнику (0..1). Кэшируется на уровне твита, т.к. мы уже
    # посчитали его при парсинге автора и не хотим делать это заново.
    source_trust_score: Mapped[float] = mapped_column(Float, default=0.5)

    # Результат antifake_check (если делался). 0..1 — вероятность того что
    # твит вводит в заблуждение. None = не проверяли.
    misleading_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Детектированная тематика (ближайший кластер).
    topic: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)

    # Engagement метрики — пригодятся в ранжировании.
    likes_count: Mapped[int] = mapped_column(Integer, default=0)
    retweets_count: Mapped[int] = mapped_column(Integer, default=0)
    replies_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.telegram_id", ondelete="CASCADE"), index=True)
    tweet_id: Mapped[str] = mapped_column(ForeignKey("tweets.tweet_id", ondelete="CASCADE"), index=True)

    # True  = 👍 (явный позитив)
    # False = 👎 (явный негатив)
    # None  = implicit skip (показали, но не кликнул в течение N часов) — см. recommender
    liked: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="feedback")

    __table_args__ = (UniqueConstraint("user_id", "tweet_id", name="uq_user_tweet_feedback"),)


class SentNews(Base):
    """Чтобы не присылать одно и то же дважды."""

    __tablename__ = "sent_news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.telegram_id", ondelete="CASCADE"), index=True)
    tweet_id: Mapped[str] = mapped_column(ForeignKey("tweets.tweet_id", ondelete="CASCADE"), index=True)

    # message_id в Telegram — нужен чтобы обновлять кнопки после лайка.
    telegram_message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Второй message_id для dual-media: когда у поста есть СВОЁ медиа И цитата
    # с отдельным медиа, бот шлёт 2 сообщения. Без этого поля при back/delete
    # второе оставалось «висеть» в чате.
    quote_telegram_message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    sent_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (UniqueConstraint("user_id", "tweet_id", name="uq_user_tweet_sent"),)


class FollowedAuthor(Base):
    """Каких авторов мониторим для пользователя (результат get_following + расширения)."""

    __tablename__ = "followed_authors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.telegram_id", ondelete="CASCADE"), index=True)
    author_username: Mapped[str] = mapped_column(String(64), index=True)

    # Вес автора для этого пользователя (поднимается при лайках твитов этого автора).
    weight: Mapped[float] = mapped_column(Float, default=1.0)

    added_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (UniqueConstraint("user_id", "author_username", name="uq_user_author"),)
