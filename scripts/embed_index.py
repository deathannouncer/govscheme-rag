"""
Embeds every chunk in Postgres that doesn't yet have a Milvus id,
pushes the vectors into Milvus, and writes the id back to Postgres.

Usage:
    python scripts/embed_index.py
"""
import os

import psycopg2
from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient
from sentence_transformers import SentenceTransformer

load_dotenv()

POSTGRES_DSN = os.environ["POSTGRES_DSN"]
MILVUS_HOST = os.environ["MILVUS_HOST"]
MILVUS_PORT = os.environ["MILVUS_PORT"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
COLLECTION_NAME = "scheme_chunks"
EMBED_DIM = 768  # EmbeddingGemma-300M output dim - verify on the model card if this fails


def get_or_create_collection(client: MilvusClient):
    if client.has_collection(COLLECTION_NAME):
        return

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=EMBED_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )

    client.create_collection(
        collection_name=COLLECTION_NAME, schema=schema, index_params=index_params
    )


def main():
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
    get_or_create_collection(client)
    client.load_collection(COLLECTION_NAME)

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
        data = [{"chunk_id": cid, "embedding": vec} for cid, vec in zip(ids, vectors)]
        client.insert(collection_name=COLLECTION_NAME, data=data)

        for chunk_id in ids:
            cur.execute(
                "UPDATE scheme_chunks SET milvus_id = %s WHERE id = %s", (chunk_id, chunk_id)
            )
        conn.commit()
        print(f"Embedded {i + len(batch)}/{len(rows)}")

    client.flush(COLLECTION_NAME)
    cur.close()
    conn.close()
    print("Done")


if __name__ == "__main__":
    main()