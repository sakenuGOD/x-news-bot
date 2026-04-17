"""Опциональная обёртка над chromadb.

Если chromadb недоступен (например, на Windows без MSVC build tools) —
модуль превращается в no-op. Ранжирование/дедубликация при этом работают
через SQLite-эмбеддинги, просто чуть медленнее на больших пулах.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from config import settings

log = logging.getLogger(__name__)

_CHROMA_OK = False
_client = None  # type: ignore[var-annotated]
_collection = None  # type: ignore[var-annotated]

try:
    import chromadb  # type: ignore
    from chromadb.config import Settings as ChromaSettings  # type: ignore

    _CHROMA_OK = True
except Exception as e:  # pragma: no cover
    log.warning("chromadb unavailable (%s) — running without vector store", e)


def _get_client():
    global _client
    if not _CHROMA_OK:
        return None
    if _client is None:
        Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
    return _client


def get_collection():
    global _collection
    if not _CHROMA_OK:
        return None
    if _collection is None:
        client = _get_client()
        if client is None:
            return None
        _collection = client.get_or_create_collection(
            name="tweets",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def upsert_tweets(
    tweet_ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    metadatas: Sequence[dict],
) -> None:
    if not tweet_ids or not _CHROMA_OK:
        return
    coll = get_collection()
    if coll is None:
        return
    try:
        coll.upsert(
            ids=list(tweet_ids),
            embeddings=[list(e) for e in embeddings],
            metadatas=list(metadatas),
        )
    except Exception as e:
        log.warning("chroma upsert failed: %s", e)


def query_similar(
    embedding: Sequence[float],
    n_results: int = 20,
    where: dict | None = None,
) -> list[tuple[str, float, dict]]:
    """Возвращает [(tweet_id, distance, metadata), ...]. Пусто если chroma недоступен."""
    if not _CHROMA_OK:
        return []
    coll = get_collection()
    if coll is None:
        return []
    try:
        res = coll.query(
            query_embeddings=[list(embedding)],
            n_results=n_results,
            where=where,
        )
    except Exception as e:
        log.warning("chroma query failed: %s", e)
        return []

    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    return list(zip(ids, dists, metas))


def delete_tweets(tweet_ids: Sequence[str]) -> None:
    if not tweet_ids or not _CHROMA_OK:
        return
    coll = get_collection()
    if coll is None:
        return
    try:
        coll.delete(ids=list(tweet_ids))
    except Exception as e:
        log.warning("chroma delete failed: %s", e)
