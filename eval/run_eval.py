"""
Runs the eval set against the running API and reports:
  - whether the expected scheme appears in the returned sources (a proxy
    for retrieval recall, since there's no hand-labeled ground-truth
    chunk set here)
  - latency per query
  - cache hit rate on a second pass (same questions run twice)

Usage:
    python eval/run_eval.py
"""
import argparse
import json
import time

import requests

API_URL = "http://localhost:8000/query"
CACHE_CLEAR_URL = "http://localhost:8000/cache"


def run_pass(eval_set, label):
    hits = 0
    latencies = []
    for item in eval_set:
        resp = requests.post(API_URL, json={"question": item["question"]}).json()
        latencies.append(resp["latency_ms"])

        scheme_hit = any(
            item["expected_scheme"].lower() in s["scheme_name"].lower()
            for s in resp["sources"]
        )
        hits += int(scheme_hit)

        print(f"[{label}] Q: {item['question']}")
        print(
            f"  expected_scheme_found={scheme_hit} cached={resp['cached']} "
            f"latency={resp['latency_ms']:.0f}ms"
        )

    recall = hits / len(eval_set)
    avg_latency = sum(latencies) / len(latencies)
    print(f"\n[{label}] recall_proxy={recall:.2f}  avg_latency_ms={avg_latency:.0f}")
    return recall, avg_latency


def main(eval_set_path: str):
    with open(eval_set_path) as f:
        eval_set = json.load(f)

    requests.delete(CACHE_CLEAR_URL)
    print("=== Pass 1 (cold cache) ===")
    run_pass(eval_set, "cold")

    print("\n=== Pass 2 (warm cache) ===")
    run_pass(eval_set, "warm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="eval/eval_set.json")
    args = parser.parse_args()
    main(args.eval_set)