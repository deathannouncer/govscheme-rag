"""
Runs the adversarial set against the live API and uses the local LLM to
judge whether each answer handled the tricky question correctly (per its
expected_behavior), rather than fabricating a confident-sounding answer.

This tests hallucination resistance specifically - fictional scheme
names, out-of-domain questions, and eligibility-contradiction traps -
which is a different failure mode than recall (run_eval.py) or
faithfulness (run_faithfulness_eval.py).

Usage:
    python eval/run_adversarial_eval.py
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

JUDGE_PROMPT = """You are checking whether an AI assistant's ANSWER to a tricky
question handled it correctly, without fabricating information.

Question: {question}

What correct handling looks like: {expected_behavior}

Answer given:
{answer}

Does the answer match what correct handling looks like? Reply with
exactly one word: PASS or FAIL."""


def main(path: str):
    with open(path) as f:
        adversarial_set = json.load(f)

    requests.delete(CACHE_CLEAR_URL)

    passed = 0
    for item in adversarial_set:
        resp = requests.post(API_URL, json={"question": item["question"]}).json()
        source_names = ", ".join(s["scheme_name"] for s in resp["sources"]) or "(none)"

        verdict = llm.chat(
            JUDGE_PROMPT.format(
                question=item["question"],
                expected_behavior=item["expected_behavior"],
                answer=resp["answer"],
            ),
            max_tokens=10,
        ).strip().upper()
        is_pass = verdict.startswith("PASS")
        passed += int(is_pass)

        print(f"Q: {item['question']}")
        print(f"  sources_cited={source_names}")
        print(f"  answer: {resp['answer'][:200]}...")
        print(f"  result={'PASS' if is_pass else 'FAIL'}\n")

    print(f"adversarial_pass_rate={passed}/{len(adversarial_set)} = {passed / len(adversarial_set):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="eval/adversarial_set.json")
    args = parser.parse_args()
    main(args.set)
