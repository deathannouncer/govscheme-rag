import psycopg2
import psycopg2.extras
import re

from app import config


def get_conn():
    return psycopg2.connect(config.POSTGRES_DSN)


def fetch_chunks_by_id(chunk_ids: list[int]) -> dict[int, dict]:
    if not chunk_ids:
        return {}
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT id, scheme_name, chunk_type, content, official_link "
        "FROM scheme_chunks WHERE id = ANY(%s)",
        (chunk_ids,),
    )
    rows = {r["id"]: dict(r) for r in cur.fetchall()}
    cur.close()
    conn.close()
    return rows


def to_tsquery_safe(query: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", query)
    unique_words = list(dict.fromkeys(words))[:30]
    return " | ".join(unique_words) if unique_words else query


def fulltext_search(query: str, top_k: int) -> list[tuple[int, float]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, ts_rank(search_vector, query) AS score
        FROM scheme_chunks, to_tsquery('english', %s) query
        WHERE search_vector @@ query
        ORDER BY score DESC
        LIMIT %s
        """,
        (to_tsquery_safe(query), top_k),
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results
