from pymilvus import MilvusClient

from app import config

_client = None


def _get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}")
    return _client


def vector_search(query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
    client = _get_client()
    client.load_collection(config.COLLECTION_NAME)
    results = client.search(
        collection_name=config.COLLECTION_NAME,
        data=[query_vector],
        anns_field="embedding",
        limit=top_k,
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    )
    return [(hit["id"], hit["distance"]) for hit in results[0]]