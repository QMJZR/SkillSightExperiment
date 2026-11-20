import argparse
import json
import os

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from data import skill_text


TASKS = ["theoremqa", "logicbench", "toolqa", "medcalcbench", "champ", "bigcodebench"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--top-npz", required=True)
    ap.add_argument("--top5-jsonl", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    corpus_path = os.path.join(args.repo_root, "data", "bench", "corpus", "corpus.json")
    with open(corpus_path, encoding="utf-8") as fh:
        corpus = json.load(fh)
    skill_ids = [str(s["skill_id"]) for s in corpus]
    texts = [skill_text(s) for s in corpus]

    npz = np.load(args.top_npz)
    indices = npz["indices"]
    query_ids = [str(x) for x in npz["query_ids"]]
    tasks = [str(x) for x in npz["tasks"]]

    meta = {}
    with open(args.top5_jsonl, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            meta[r["query_id"]] = (r["query"], set(r["gold_ids"]), r["task"], r["gold_rank_dense"])

    n = len(query_ids)
    if args.limit:
        n = min(n, args.limit)

    print(f"[rerank] loading {args.model_path}", flush=True)
    model = CrossEncoder(
        args.model_path,
        device=args.device,
        max_length=args.max_length,
        automodel_args={"torch_dtype": torch.float16},
    )

    per_task = {t: {"n": 0, "recall": 0.0, "hit": 0.0, "mrr": 0.0} for t in TASKS}
    for start in range(0, n, 100):
        end = min(start + 100, n)
        pairs = []
        spans = []
        for i in range(start, end):
            cand = indices[i]
            spans.append(len(cand))
            q = meta[query_ids[i]][0]
            for j in cand:
                pairs.append((q, texts[int(j)]))
        scores = model.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        offset = 0
        for i in range(start, end):
            m = spans[i - start]
            cand_scores = scores[offset : offset + m]
            offset += m
            order = np.argsort(cand_scores)[::-1][: args.top_k]
            ranked = [skill_ids[int(indices[i, k])] for k in order]
            gold = meta[query_ids[i]][1]
            positions = [r + 1 for r, sid in enumerate(ranked) if sid in gold]
            row = per_task[tasks[i]]
            row["n"] += 1
            row["recall"] += len(set(ranked) & gold) / max(1, len(gold))
            row["hit"] += 1.0 if positions else 0.0
            row["mrr"] += 1.0 / positions[0] if positions else 0.0
        print(f"[rerank] {end}/{n}", flush=True)

    total = sum(v["n"] for v in per_task.values())
    overall = {"n": total, "recall": 0.0, "hit": 0.0, "mrr": 0.0}
    for task, row in per_task.items():
        if row["n"]:
            for key in ("recall", "hit", "mrr"):
                row[key] /= row["n"]
                overall[key] += row[key] * row["n"] / max(1, total)
    out = {"variant": "dense_rerank", "top_k": args.top_k, "per_task": per_task, "overall_micro": overall}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[rerank] overall", overall, flush=True)
    print(f"[rerank] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
