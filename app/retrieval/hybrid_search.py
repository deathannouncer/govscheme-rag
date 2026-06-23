from sentence_transformers import SentenceTransformer

from app import config, db, vector_store
from app.retrieval.fusion import reciprocal_rank_fusion

_embedder = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def embed_query(query: str) -> list[float]:
    return _get_embedder().encode(query, normalize_embeddings=True).tolist()


def hybrid_search(query: str, query_vector: list[float] | None = None) -> list[dict]:
    if query_vector is None:
        query_vector = embed_query(query)

    vector_hits = vector_store.vector_search(query_vector, config.TOP_K_VECTOR)
    fulltext_hits = db.fulltext_search(query, config.TOP_K_FULLTEXT)

    fused = reciprocal_rank_fusion([vector_hits, fulltext_hits])[: config.TOP_K_FUSED]
    chunk_ids = [doc_id for doc_id, _ in fused]

    chunk_map = db.fetch_chunks_by_id(chunk_ids)
    return [chunk_map[doc_id] for doc_id, _ in fused if doc_id in chunk_map]

def multi_query_hybrid_search(queries: list[str]) -> list[dict]:
    """
    Runs vector + full-text search for each query variant (e.g. the user's
    original question AND an LLM-rewritten version) and fuses all of them
    together. This guards against a bad rewrite - e.g. the LLM "correcting"
    an unusual scheme name into a more common-sounding one - silently
    overriding the original wording and losing the correct match.
    """
    ranked_lists = []
    for q in queries:
        qv = embed_query(q)
        ranked_lists.append(vector_store.vector_search(qv, config.TOP_K_VECTOR))
        ranked_lists.append(db.fulltext_search(q, config.TOP_K_FULLTEXT))

    fused = reciprocal_rank_fusion(ranked_lists)[: config.TOP_K_FUSED]
    chunk_ids = [doc_id for doc_id, _ in fused]

    chunk_map = db.fetch_chunks_by_id(chunk_ids)
    return [chunk_map[doc_id] for doc_id, _ in fused if doc_id in chunk_map]
