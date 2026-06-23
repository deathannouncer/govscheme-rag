from pymilvus import Collection, connections

from app import config

_connected = False


def _ensure_connected():
    global _connected
    if not _connected:
        connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        _connected = True


def vector_search(query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
    _ensure_connected()
    collection = Collection(config.COLLECTION_NAME)
    collection.load()
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["chunk_id"],
    )
    return [(hit.id, hit.distance) for hit in results[0]]
