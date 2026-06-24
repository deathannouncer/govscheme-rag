"""
Samples N random schemes from Postgres and uses the local LLM to write
one natural-sounding question per scheme, for a larger eval set than a
handful of hand-written examples.

Caveat: questions generated directly from the same eligibility text the
retriever searches over tend to be "easier" than organic user questions -
more lexical overlap with the source chunk than a real citizen's phrasing
would have. Treat the recall number from this set as an optimistic upper
bound, not a final benchmark. Keeping the original hand-written questions
mixed in (which paraphrase more, abbreviate scheme names, etc.) is still
worth doing for a more realistic blend.

Usage:
    python scripts/generate_eval_set.py --n 25 --out eval/eval_set_generated.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm
from app.db import get_conn

QUESTION_PROMPT = """Below is eligibility information for an Indian government scheme.
Write ONE natural-sounding question a citizen might ask a chatbot about
this scheme - not a copy of the text. You may mention the scheme by
name, or describe a situation and ask if they qualify. Return ONLY the
question, nothing else.

Scheme name: {scheme_name}
Eligibility text: {content}"""


def sample_schemes(n: int) -> list[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT scheme_name FROM scheme_chunks WHERE chunk_type = 'eligibility'")
    all_names = [r[0] for r in cur.fetchall()]
    sampled_names = random.sample(all_names, min(n, len(all_names)))

    cur.execute(
        "SELECT scheme_name, content FROM scheme_chunks "
        "WHERE chunk_type = 'eligibility' AND scheme_name = ANY(%s)",
        (sampled_names,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"scheme_name": r[0], "content": r[1]} for r in rows]


def main(n: int, out_path: str):
    schemes = sample_schemes(n)
    print(f"Sampled {len(schemes)} schemes, generating questions...")

    eval_set = []
    for i, s in enumerate(schemes):
        question = llm.chat(
            QUESTION_PROMPT.format(scheme_name=s["scheme_name"], content=s["content"])
        ).strip()
        eval_set.append({"question": question, "expected_scheme": s["scheme_name"]})
        print(f"[{i + 1}/{len(schemes)}] {s['scheme_name']}: {question}")

    with open(out_path, "w") as f:
        json.dump(eval_set, f, indent=2)
    print(f"\nWrote {len(eval_set)} generated questions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--out", default="eval/eval_set_generated.json")
    args = parser.parse_args()
    main(args.n, args.out)