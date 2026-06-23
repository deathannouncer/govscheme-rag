"""
Mimics the live agent pipeline step by step (rewrite -> hybrid search ->
rerank), without the agent loop or final generation, so we can see exactly
where a chunk drops out between raw retrieval and the final reranked
context that actually goes to the LLM.

Usage:
    python scripts/debug_agent.py "What are the core guidelines or criteria for the Mukhyamantri Jan-Van Scheme?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, llm
from app.agent.loop import REWRITE_PROMPT
from app.retrieval.hybrid_search import multi_query_hybrid_search
from app.retrieval.reranker import rerank


def main(question: str):
    rewritten = llm.chat(REWRITE_PROMPT.format(question=question))
    print(f"Rewritten query: {rewritten!r}\n")

    candidates = multi_query_hybrid_search([question, rewritten])
    print(f"hybrid_search() on the REWRITTEN query -> {len(candidates)} candidates:")
    for c in candidates:
        print(f"  [{c['scheme_name']}] chunk_type={c['chunk_type']}")

    reranked = rerank(question, candidates, config.TOP_K_FINAL)
    print(f"\nAfter rerank, final top {config.TOP_K_FINAL} (this is what the LLM actually sees):")
    for c in reranked:
        print(f"  [{c['scheme_name']}] chunk_type={c['chunk_type']} score={c['rerank_score']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])