import argparse
import json
import os
from pathlib import Path

import numpy as np

from data import collect_queries, load_instances, skill_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    import bm25s

    corpus_path = os.path.join(args.repo_root, "data", "bench", "corpus", "corpus.json")
    with open(corpus_path, encoding="utf-8") as fh:
        corpus = json.load(fh)
    texts = [skill_text(skill) for skill in corpus]

    print(f"[bm25] tokenize corpus n={len(texts)}", flush=True)
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=True)
    retriever = bm25s.BM25()
    print("[bm25] build index", flush=True)
    retriever.index(corpus_tokens)

    instances_dir = os.path.join(args.repo_root, "data", "bench", "instances")
    tasks = ["theoremqa", "logicbench", "toolqa", "medcalcbench", "champ", "bigcodebench"]
    per_task = {}
    all_rows = []
    for task in tasks:
        instances = load_instances(Path(instances_dir) / f"{task}.json")
        examples = collect_queries(task, instances)
        query_tokens = bm25s.tokenize(
            [ex.query for ex in examples], stopwords="en", show_progress=False
        )
        results, _ = retriever.retrieve(query_tokens, k=args.top_k, show_progress=False)

        n = len(examples)
        metrics = {str(k): {"recall": 0.0, "hit": 0.0, "mrr": 0.0} for k in (1, 5, 10) if k <= args.top_k}
        for i, ex in enumerate(examples):
            gold = set(ex.gold_ids)
            ids = [str(corpus[j]["skill_id"]) for j in results[i, : args.top_k]]
            positions = [rank + 1 for rank, sid in enumerate(ids) if sid in gold]
            for k in metrics:
                kk = int(k)
                metrics[k]["recall"] += len(set(ids[:kk]) & gold) / max(1, len(gold))
                metrics[k]["hit"] += 1.0 if positions and positions[0] <= kk else 0.0
                metrics[k]["mrr"] += 1.0 / positions[0] if positions and positions[0] <= kk else 0.0
        for k in metrics:
            for key in metrics[k]:
                metrics[k][key] /= max(1, n)
        per_task[task] = {"n": n, "topk": metrics}
        all_rows.append((task, n, metrics))
        print(task, n, metrics["10"], flush=True)

    total = sum(row[1] for row in all_rows)
    overall = {str(k): {"recall": 0.0, "hit": 0.0, "mrr": 0.0} for k in (1, 5, 10) if k <= args.top_k}
    for _, n, metrics in all_rows:
        for k in overall:
            for key in overall[k]:
                overall[k][key] += metrics[k][key] * n / total

    out = {"variant": "bm25", "top_k": args.top_k, "per_task": per_task, "overall_micro": overall}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[bm25] overall", overall, flush=True)
    print(f"[bm25] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
