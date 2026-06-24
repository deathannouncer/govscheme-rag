import time

from fastapi import FastAPI
from pydantic import BaseModel

from app import cache
from app.agent.loop import answer_question
from app.retrieval.hybrid_search import embed_query

app = FastAPI(title="Govt Scheme RAG")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    context: str
    cached: bool
    latency_ms: float


@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"status": "cleared"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    start = time.perf_counter()

    probe_vector = embed_query(req.question)
    cached_entry = cache.lookup(probe_vector)
    if cached_entry:
        latency_ms = (time.perf_counter() - start) * 1000
        return QueryResponse(
            answer=cached_entry["answer"],
            sources=cached_entry["sources"],
            context=cached_entry.get("context", ""),
            cached=True,
            latency_ms=latency_ms,
        )

    result = answer_question(req.question)
    cache.store(probe_vector, result["answer"], result["sources"], result["context"])

    latency_ms = (time.perf_counter() - start) * 1000
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        context=result["context"],
        cached=False,
        latency_ms=latency_ms,
    )