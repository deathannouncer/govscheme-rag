# Government Scheme RAG

An agentic, hybrid-search RAG system over Indian government scheme data.
Every architectural choice below exists because a simpler version of it
broke during development - this README describes what's actually running,
not an idealized version of it.

---

## Architecture

```
User Question
     │
     ▼
[ Redis Semantic Cache ] ──(hit, cosine sim ≥ 0.92)──> Cached Answer + Sources
     │ (miss)
     ▼
[ LLM Query Rewrite ] ── original question + rewrite, both kept ──┐
     │                                                            │
     ▼                                                            │
[ Hybrid Search per query: Vector (Milvus) + Full-Text (Postgres) ] <─┘
     │
     ▼
[ Reciprocal Rank Fusion across all query × search-type lists ]
     │
     ▼
[ Cross-Encoder Rerank against the ORIGINAL question ]
     │
     ▼
[ Agent: "enough context?" ] ──no, loops remain──> [ LLM writes a genuinely
     │                                               different follow-up query ]
     │                                                       │
     ▼ yes, or loops exhausted                               │
[ Grounded Answer Generation ]                                │
     │                                                        │
     ▼                                                        │
Answer + Sources + Context ──> cached for next time            │
                                                                │
        (new query appended, loop back to Hybrid Search) <────┘
```

Both the original question *and* the LLM's rewrite are searched on every
pass, never just the rewrite alone - an LLM rewrite "correcting" an
unusual scheme name into a more common-sounding one (a real failure mode
hit during testing) would otherwise silently destroy retrieval for it.
Reranking is scored against the original question specifically, not the
rewrite, so a bad rewrite can't bias relevance scoring either.

---

## Key design decisions (and the bugs that motivated them)

- **Scheme name is prefixed into every chunk's content** before it's
  indexed or embedded. Many scheme descriptions never restate their own
  name in the body text - without the prefix, full-text search has no
  way to use the single most identifying word in a user's question, no
  matter how it's phrased.
- **Full-text tokenization splits on punctuation** rather than rejecting
  any token containing it. A naive `isalnum()` filter silently dropped
  hyphenated terms like "Jan-Van" entirely, which made full-text search
  fall back to matching on whatever generic word was left.
- **The cache key is the original question's embedding**, not the
  rewrite's. Caching against the rewrite meant identical repeated
  questions could still miss the cache, since the LLM rewrite isn't
  guaranteed to produce identical text twice.
- **All LLM calls run at temperature 0** with a capped `num_predict`.
  Non-zero temperature made eval results genuinely irreproducible (the
  same question could pass on one run and fail on the next); the token
  cap exists because an uncapped generation that degenerates into a
  repetition loop can cascade into a full-text query with thousands of
  OR'd terms, which is slow for Postgres to parse.
- **The agent retry loop appends a new query rather than re-running the
  same one.** The original version re-issued an identical search on
  retry, which can never surface anything new - it was latency cost with
  no retrieval benefit.
- **Multi-condition eligibility reasoning is the system's main weak point**,
  not retrieval. A 28-question faithfulness eval found the local 7B model
  gets logical inference over retrieved criteria wrong often enough to
  matter - see "Known limitations" below for specifics. Retrieval recall
  stays at 1.00 throughout; the right information is consistently found,
  the model just sometimes draws the wrong conclusion from it.

---

## Tech stack

- **Backend:** FastAPI, Python, Uvicorn
- **LLM:** Ollama, local (`qwen2.5:7b-instruct` by default - swap via `.env`)
- **Vector search:** Milvus standalone, via `MilvusClient`
- **Embeddings:** `google/embeddinggemma-300m`
- **Reranker:** `Qwen/Qwen3-Reranker-0.6B` (cross-encoder)
- **Full-text search:** Postgres `tsvector`/`tsquery`
- **Cache:** Redis (semantic, cosine-similarity lookup)
- **Frontend:** Streamlit

---

## Setup

1. **Get the dataset.** Download the myScheme dataset from
   [Hugging Face](https://huggingface.co/datasets/shrijayan/gov_myscheme) or
   [Kaggle](https://www.kaggle.com/datasets/jainamgada45/indian-government-schemes),
   save as `data/schemes.json` or `data/schemes.csv`. Check the column
   headers against `FIELD_MAP` in `scripts/ingest.py` first - sources use
   different column names.

2. **Start infra:**
   ```powershell
   docker compose up -d
   ```
   Starts Postgres, Redis, and Milvus (with etcd + minio). Give Milvus
   ~30s to finish starting. Containers are configured with
   `restart: unless-stopped`, but Docker Desktop itself still needs to be
   running - if you get a connection-refused error later, this is almost
   always why.

3. **Pull a local LLM** ([Ollama](https://ollama.com)):
   ```powershell
   ollama serve
   ollama pull qwen2.5:7b-instruct
   ```
   `ollama serve` needs to be running in its own terminal whenever you
   use the system - it's a separate background process from Docker.

4. **Python environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

5. **Ingest, then embed** (order matters - `embed_index.py` reads from
   the table `ingest.py` creates):
   ```powershell
   python scripts/ingest.py --file data/schemes.csv
   python scripts/embed_index.py
   ```

6. **Run the API** (own terminal, restrict the reload watcher so editing
   scripts/eval files doesn't trigger pointless restarts):
   ```powershell
   uvicorn app.main:app --reload --reload-dir app
   ```

7. **Run the frontend** (separate terminal):
   ```powershell
   streamlit run streamlit_app.py
   ```

---

## Evaluation

Three scripts, each testing a different failure mode:

| Script | What it checks |
| --- | --- |
| `eval/run_eval.py` | Retrieval recall - does the right scheme end up in the sources? Also measures cold vs. warm cache latency. |
| `eval/run_faithfulness_eval.py` | Is the generated answer actually grounded in the retrieved context, or does it embellish? |
| `eval/run_adversarial_eval.py` | Does it correctly decline on fictional schemes, out-of-domain questions, and eligibility-contradiction traps, instead of hallucinating a confident answer? |

```powershell
python eval/run_eval.py
python eval/run_faithfulness_eval.py
python eval/run_adversarial_eval.py
```

`eval/eval_set.json` is the combined 28-question set (3 hand-written + 25
auto-generated from real scheme data) - this is what `run_eval.py` tests
by default now. The faithfulness and adversarial scripts have so far only
been run against the smaller hand-written set; re-running them against
the full 28 is a reasonable next step before relying on those numbers as
heavily as the recall figure.

Caveat worth keeping in mind: the faithfulness and adversarial judges are
the same local LLM doing the generation, just prompted differently. That
makes them a useful second opinion for catching obvious problems, not an
independent ground truth.

---

## Results observed so far

These are what's actually been measured, across multiple separate runs -
reported as ranges because that's what was observed, not a single
cherry-picked number:

| Metric | Result |
| --- | --- |
| Retrieval recall, merged 28-question set (3 hand-written + 25 auto-generated) | 1.00 |
| Adversarial pass rate (8 questions: fictional schemes, out-of-domain, eligibility traps) | 8/8 |
| Faithfulness, full 28-question set (LLM-judge against actual retrieved context) | 20/28 (0.71) - see "Known limitations" for what the 8 failures actually are |
| Cold-cache latency (full pipeline, local 7B model on CPU) | ~10s to ~95s per query, varies by question and by run |
| Warm-cache latency (Redis hit) | ~100ms to ~1.5s |
| Cache speedup | Observed between ~55x and ~230x across different runs |

The auto-generated questions (25 of the 28) skew easier than real usage,
since they're written from the same eligibility text the retriever
searches over - the 3 hand-written questions (abbreviated names, indirect
phrasing) are the more realistic stress test of the set.

---

## Known limitations

- **Multi-condition eligibility reasoning is the main weak point**,
  found via the 28-question faithfulness eval (8 failures, categorized
  honestly rather than averaged into one number):
  - *Logical inference errors (3 cases)* - the model gets straightforward
    inference wrong despite having the right facts. Examples: concluding
    a student "hasn't passed the previous year's exam" specifically
    *because* they're currently enrolled in the next grade up (backwards -
    being in 7th grade implies having passed 6th); failing to match
    "applicant is a woman" against a context that explicitly lists
    "women" as one of four eligible priority groups; reading a
    benefit-capping clause ("assistance availed only once") as a
    disqualification clause.
  - *One internally incoherent answer* - walked through all four
    eligibility criteria, confirmed each was met, then contradicted
    itself and concluded the person wasn't eligible anyway, based on an
    age-ceiling concern that didn't apply to their stated age.
  - *Under-claiming (2 cases)* - hedging despite the context actually
    answering the question clearly (no acreage cap stated -> "cannot
    conclude eligibility" instead of inferring no cap applies).
  - *Over-claiming on a subjective scheme (1 case)* - confidently called
    someone "likely eligible" for a jury-judged honors scheme whose
    criteria are qualitative ("outstanding," "wider impact"), which isn't
    something a threshold check can actually verify.
  - *One flawed test question* - asked about "this scheme" without
    naming one, an artifact of the auto-generated eval set rather than a
    pipeline bug.

  Retrieval recall stayed at 1.00 throughout all of this - the right
  information was consistently found. The failures are downstream, in
  how the local 7B model reasons over multi-condition criteria once
  retrieved. Documented here rather than mitigated for now; a larger
  model or a chain-of-thought-style answer prompt are the likely next
  things to try if this needs improving.
- `official_link` for the Kaggle dataset variant is reconstructed from a
  `slug` field, since that source has no direct link column - best-effort,
  not guaranteed correct for every scheme.
- Latency is dominated by local LLM inference (2-4 calls per query), not
  retrieval - hardware-dependent, will look very different on a machine
  with a GPU Ollama can actually use versus CPU-only.
- The Redis cache does a brute-force cosine-similarity scan over stored
  entries - fine at this scale, would need RediSearch's vector index to
  scale further.
