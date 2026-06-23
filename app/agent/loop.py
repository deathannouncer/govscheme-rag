from app import config, llm
from app.retrieval.hybrid_search import multi_query_hybrid_search
from app.retrieval.reranker import rerank

REWRITE_PROMPT = """You help rewrite a user's question about Indian government schemes
into a clearer search query. Expand abbreviations and add likely keywords,
but do NOT change, "correct", or rename any scheme name, proper noun, or
specific term that appears in the question - copy those exactly as given,
even if they look unusual or non-standard. Return ONLY the rewritten
query, nothing else.

Question: {question}"""

JUDGE_PROMPT = """You are checking whether the retrieved context is enough to answer
the question. Reply with exactly one word: YES or NO.

Question: {question}

Retrieved context:
{context}"""

ANSWER_PROMPT = """Answer the question using ONLY the context below. Cite the scheme
name for every claim. If the context doesn't contain the answer, say so plainly.

Question: {question}

Context:
{context}"""


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['scheme_name']} - {c['chunk_type']}] {c['content']}" for c in chunks
    )


def answer_question(question: str) -> dict:
    rewritten = llm.chat(REWRITE_PROMPT.format(question=question))

    all_chunks: list[dict] = []
    seen_ids: set[int] = set()
    reranked: list[dict] = []

    for _ in range(config.MAX_AGENT_LOOPS):
        candidates = multi_query_hybrid_search([question, rewritten])
        new_chunks = [c for c in candidates if c["id"] not in seen_ids]
        all_chunks.extend(new_chunks)
        seen_ids.update(c["id"] for c in new_chunks)

        reranked = rerank(question, all_chunks, config.TOP_K_FINAL)
        context = _format_context(reranked)

        verdict = llm.chat(
            JUDGE_PROMPT.format(question=question, context=context)
        ).strip().upper()
        if verdict.startswith("YES"):
            break

    final_context = _format_context(reranked)
    answer = llm.chat(ANSWER_PROMPT.format(question=question, context=final_context))

    sources = [
        {"scheme_name": c["scheme_name"], "official_link": c["official_link"]} for c in reranked
    ]
    return {"answer": answer, "sources": sources}