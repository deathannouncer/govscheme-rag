"""
For each question in the eval set, runs the live API to get the answer +
sources, then asks the local LLM to judge whether every claim in the
answer is actually grounded in those sources, rather than fabricated.

This is separate from recall (run_eval.py checks whether the *right
scheme* was retrieved) - this checks whether the *answer text itself*
stays honest given whatever was retrieved.

Usage:
    python eval/run_faithfulness_eval.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from app import llm

API_URL = "http://localhost:8000/query"
CACHE_CLEAR_URL = "http://localhost:8000/cache"

JUDGE_PROMPT = """You are checking whether an AI-generated ANSWER is fully supported
by the CONTEXT it was given, or whether it states anything not actually
present in that context (fabrication).

Reply with exactly one word: FAITHFUL or UNFAITHFUL.

Question: {question}

Context the answer was generated from:
{context}

Answer:
{answer}"""


def main(path: str):
    with open(path) as f:
        eval_set = json.load(f)

    requests.delete(CACHE_CLEAR_URL)

    faithful_count = 0
    for item in eval_set:
        resp = requests.post(API_URL, json={"question": item["question"]}).json()

        verdict = llm.chat(
            JUDGE_PROMPT.format(
                question=item["question"], context=resp["context"], answer=resp["answer"]
            ),
            max_tokens=10,
        ).strip().upper()
        is_faithful = verdict.startswith("FAITHFUL")
        faithful_count += int(is_faithful)

        print(f"Q: {item['question']}")
        print(f"  faithful={is_faithful} ({verdict})")
        if not is_faithful:
            print(f"  --- CONTEXT ---\n{resp['context']}\n")
            print(f"  --- ANSWER ---\n{resp['answer']}\n")

    print(
        f"\nfaithfulness_rate={faithful_count}/{len(eval_set)} "
        f"= {faithful_count / len(eval_set):.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="eval/eval_set.json")
    args = parser.parse_args()
    main(args.eval_set)