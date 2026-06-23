"""
Quick diagnostic: checks whether a given scheme's chunks exist in Postgres
and were embedded, then runs hybrid_search() directly (no LLM, no agent
loop) to see what comes back for a question - isolating retrieval from
generation entirely.

Usage:
    python scripts/debug_retrieval.py "Mukhyamantri Jan-Van Scheme" "What are the core guidelines or criteria for the Mukhyamantri Jan-Van Scheme?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_conn
from app.retrieval.hybrid_search import hybrid_search


def check_chunks(scheme_name: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, chunk_type, milvus_id IS NOT NULL AS embedded, content "
        "FROM scheme_chunks WHERE scheme_name = %s",
        (scheme_name,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    print(f"Chunks in Postgres for '{scheme_name}':")
    for r in rows:
        snippet = r[3][:150].replace("\n", " ")
        print(f"  id={r[0]} chunk_type={r[1]} embedded={r[2]}")
        print(f"    content: {snippet}...")
    if not rows:
        print("  (none found - check the scheme_name spelling matches exactly)")
    return rows


def check_retrieval(question: str):
    print(f"\nhybrid_search() results for: {question!r}")
    results = hybrid_search(question)
    if not results:
        print("  (empty - nothing came back from vector or full-text search)")
    for r in results:
        print(f"  [{r['scheme_name']}] chunk_type={r['chunk_type']}")


if __name__ == "__main__":
    scheme_name = sys.argv[1]
    question = sys.argv[2]
    check_chunks(scheme_name)
    check_retrieval(question)