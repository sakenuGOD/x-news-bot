"""Эмбеддинги через OpenAI text-embedding-3-small (проксируется через ProxyAPI).

Батчи по 100. Все embedding кэшируются в Tweet.embedding (SQLite) навсегда.
На случай сбоя возвращаем None — вызывающий код должен это обработать.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Sequence

import numpy as np
from openai import AsyncOpenAI

from config import settings

log = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.proxyapi_key,
            base_url=settings.openai_base_url,
            timeout=30.0,
            max_retries=2,
        )
    return _client


async def embed_text(text: str) -> list[float] | None:
    if not text or not text.strip():
        return None
    res = await embed_batch([text])
    return res[0] if res else None


async def embed_batch(texts: Sequence[str], batch_size: int = 100) -> list[list[float] | None]:
    """Возвращает список эмбеддингов той же длины что входной список.

    Пустые/невалидные тексты дадут None (сохраняем позицию, чтобы индексы сохли).
    """
    if not texts:
        return []

    results: list[list[float] | None] = [None] * len(texts)
    valid_indices: list[int] = []
    valid_texts: list[str] = []
    for i, t in enumerate(texts):
        if t and t.strip():
            valid_indices.append(i)
            valid_texts.append(t.strip())

    client = _get_client()

    for start in range(0, len(valid_texts), batch_size):
        chunk = valid_texts[start : start + batch_size]
        chunk_indices = valid_indices[start : start + batch_size]
        try:
            resp = await client.embeddings.create(
                model=settings.embedding_model,
                input=chunk,
            )
            for local_i, emb in enumerate(resp.data):
                results[chunk_indices[local_i]] = list(emb.embedding)
        except Exception as e:
            log.error("embedding batch failed (size=%d): %s", len(chunk), e)
            # Оставляем None в этих позициях — выше разберутся.

    return results


# --------------------- векторные утилиты -----------------------


def normalize(vec: Sequence[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n < 1e-9:
        return arr.tolist()
    return (arr / n).tolist()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def mean_vector(vectors: Sequence[Sequence[float]]) -> list[float] | None:
    clean = [v for v in vectors if v]
    if not clean:
        return None
    arr = np.asarray(clean, dtype=np.float32)
    return arr.mean(axis=0).tolist()


def update_on_like(
    preference_vector: Sequence[float],
    tweet_vector: Sequence[float],
    rate: float,
) -> list[float]:
    """preference = normalize((1 - rate) * old + rate * tweet)."""
    old = np.asarray(preference_vector, dtype=np.float32)
    tw = np.asarray(tweet_vector, dtype=np.float32)
    new = (1.0 - rate) * old + rate * tw
    return normalize(new.tolist())


def update_on_dislike(
    preference_vector: Sequence[float],
    tweet_vector: Sequence[float],
    rate: float,
) -> list[float]:
    """preference = normalize(old - rate * tweet). Дизлайки — ключевой сигнал."""
    old = np.asarray(preference_vector, dtype=np.float32)
    tw = np.asarray(tweet_vector, dtype=np.float32)
    new = old - rate * tw
    return normalize(new.tolist())


# ------- якоря тематических кластеров (для cluster_weights) ----

_cluster_anchors: dict[str, list[float]] | None = None
_anchors_lock = asyncio.Lock()


async def get_cluster_anchors() -> dict[str, list[float]]:
    """Вычисляет один раз embedding-ы для каждого кластера из config.topic_clusters."""
    global _cluster_anchors
    if _cluster_anchors is not None:
        return _cluster_anchors
    async with _anchors_lock:
        if _cluster_anchors is not None:
            return _cluster_anchors
        names = list(settings.topic_clusters.keys())
        descs = [settings.topic_clusters[n] for n in names]
        embs = await embed_batch(descs)
        mapping: dict[str, list[float]] = {}
        for name, emb in zip(names, embs):
            if emb:
                mapping[name] = emb
        _cluster_anchors = mapping
        log.info("cluster anchors computed for %d clusters", len(mapping))
        return _cluster_anchors


async def nearest_cluster(embedding: Sequence[float]) -> tuple[str, float]:
    """Ближайший тематический кластер и cosine-similarity к нему."""
    anchors = await get_cluster_anchors()
    if not anchors or not embedding:
        return ("news", 0.0)
    best_name = "news"
    best_sim = -1.0
    for name, anchor in anchors.items():
        sim = cosine_similarity(embedding, anchor)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return (best_name, best_sim)
