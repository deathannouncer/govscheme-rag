import json

import numpy as np
import redis

from app import config

_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)
CACHE_KEY = "rag:query_cache"  # Redis list of {"vector": [...], "answer": ..., "sources": [...]}
MAX_CACHE_ENTRIES = 500


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def lookup(query_vector: list[float]) -> dict | None:
    """
    Brute-force scan over cached entries - fine at demo scale (a few hundred
    entries). A production version would use RediSearch's vector index
    instead of a Python-side cosine loop.
    """
    raw_entries = _client.lrange(CACHE_KEY, 0, -1)
    q = np.array(query_vector)
    for raw in raw_entries:
        entry = json.loads(raw)
        sim = _cosine(q, np.array(entry["vector"]))
        if sim >= config.CACHE_SIMILARITY_THRESHOLD:
            return entry
    return None


def store(query_vector: list[float], answer: str, sources: list[dict]):
    entry = json.dumps({"vector": query_vector, "answer": answer, "sources": sources})
    _client.lpush(CACHE_KEY, entry)
    _client.ltrim(CACHE_KEY, 0, MAX_CACHE_ENTRIES - 1)

def clear():
    _client.delete(CACHE_KEY)