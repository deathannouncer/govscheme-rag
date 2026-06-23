# Government Scheme RAG

Agentic, hybrid-search RAG system over Indian government scheme data:
query rewriting, vector + full-text hybrid retrieval, RRF fusion,
cross-encoder reranking, an agent loop that decides whether to retrieve
again, and a Redis semantic cache.

## Architecture

```
question -> rewrite -> hybrid search (vector + full-text) -> RRF fusion
  -> rerank -> agent judges "enough context?" -> loop or generate -> answer
```

Caching wraps the whole flow: a query that's semantically close to a past
one returns the cached answer without touching retrieval at all.

## Setup

1. **Get the dataset.** Download the myScheme dataset from either:
   - https://huggingface.co/datasets/shrijayan/gov_myscheme
   - https://www.kaggle.com/datasets/jainamgada45/indian-government-schemes

   Save it as `data/schemes.json` or `data/schemes.csv`. Check the column
   headers against `FIELD_MAP` in `scripts/ingest.py` first - the two
   sources use slightly different column names.

2. **Start infra:**
   ```
   docker compose up -d
   ```
   Starts Postgres, Redis, and Milvus (with etcd + minio as
   dependencies). Give Milvus ~30s to finish starting before the next steps.

3. **Pull a local LLM via Ollama** (install separately if needed:
   https://ollama.com):
   ```
   ollama pull qwen2.5:7b-instruct
   ```
   Or change `OLLAMA_MODEL` in `.env` to whatever you've pulled.

4. **Python environment:**
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   ```

5. **Ingest + embed:**
   ```
   python scripts/ingest.py --file data/schemes.json
   python scripts/embed_index.py
   ```

6. **Run the API:**
   ```
   uvicorn app.main:app --reload
   ```

7. **Try it:**
   ```
   curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "Am I eligible for PM Kisan if I own 3 acres?"}'
   ```

8. **Evaluate:**
   ```
   python eval/run_eval.py
   ```
   Update `eval/eval_set.json` with real scheme names once you've looked
   at the actual dataset content.

## Notes / known rough edges

- `EMBED_DIM` in `scripts/embed_index.py` assumes EmbeddingGemma-300M's
  output dimension - verify against the model card if collection creation
  fails.
- Exact HF repo IDs for `EMBEDDING_MODEL` / `RERANKER_MODEL` in `.env` -
  double check these resolve on Hugging Face before running
  `embed_index.py`; naming may have shifted.
- The Redis cache does a brute-force similarity scan - fine at this scale.
  A production version would use RediSearch's vector index instead.
- `app/db.py`'s full-text tokenizer is intentionally naive (ORs the query
  terms). Tune if full-text results look too noisy.
- This hasn't been run end-to-end yet - the pieces are individually
  straightforward, but expect a couple of integration snags (model load
  errors, dimension mismatches) on first run. That's normal for a fresh
  scaffold, not a sign something's fundamentally wrong.
