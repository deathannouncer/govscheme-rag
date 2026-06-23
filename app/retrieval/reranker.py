from sentence_transformers import CrossEncoder

from app import config

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(config.RERANKER_MODEL)
    return _model


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    candidates: list of dicts with at least a "content" key.
    Returns the same dicts, sorted by reranker score, trimmed to top_k.
    """
    model = _get_model()
    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
