"""
Embeds every chunk in Postgres that doesn't yet have a Milvus id,
pushes the vectors into Milvus, and writes the id back to Postgres.

Usage:
    python scripts/embed_index.py
"""
import os

import psycopg2
from dotenv import load_dotenv
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer

load_dotenv()

POSTGRES_DSN = os.environ["POSTGRES_DSN"]
MILVUS_HOST = os.environ["MILVUS_HOST"]
MILVUS_PORT = os.environ["MILVUS_PORT"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
COLLECTION_NAME = "scheme_chunks"
EMBED_DIM = 768  # EmbeddingGemma-300M output dim - verify on the model card if this fails


def get_or_create_collection() -> Collection:
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
    ]
    schema = CollectionSchema(fields, description="Scheme chunk embeddings")
    collection = Collection(COLLECTION_NAME, schema)
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    return collection


def main():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = get_or_create_collection()
    collection.load()

    model = SentenceTransformer(EMBEDDING_MODEL)

    conn = psycopg2.connect(POSTGRES_DSN)
    cur = conn.cursor()
    cur.execute("SELECT id, content FROM scheme_chunks WHERE milvus_id IS NULL")
    rows = cur.fetchall()
    print(f"{len(rows)} chunks need embedding")

    batch_size = 64
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]

        vectors = model.encode(texts, normalize_embeddings=True).tolist()
        collection.insert([ids, vectors])

        for chunk_id in ids:
            cur.execute(
                "UPDATE scheme_chunks SET milvus_id = %s WHERE id = %s", (chunk_id, chunk_id)
            )
        conn.commit()
        print(f"Embedded {i + len(batch)}/{len(rows)}")

    collection.flush()
    cur.close()
    conn.close()
    print("Done")


if __name__ == "__main__":
    main()
