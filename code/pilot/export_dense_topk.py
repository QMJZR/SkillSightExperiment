import argparse
import json
import os
from pathlib import Path

import numpy as np

from data import collect_queries, load_instances, skill_text
from utils import model_tag


TASKS = ["theoremqa", "logicbench", "toolqa", "medcalcbench", "champ", "bigcodebench"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--snippet-chars", type=int, default=1500)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = Path(args.repo_root) / "data" / "bench" / "corpus" / "corpus.json"
    with open(corpus_path, encoding="utf-8") as fh:
        corpus = json.load(fh)
    skill_ids = [str(s["skill_id"]) for s in corpus]
    skill_texts = [skill_text(s) for s in corpus]

    tag = model_tag(args.model_path)
    doc_cache = Path(args.cache_dir) / f"all_{tag}_{args.max_seq_length}.npy"
    docs = np.load(doc_cache).astype(np.float32)
    if docs.shape[0] != len(corpus):
        raise SystemExit(f"doc cache mismatch: {docs.shape[0]} != {len(corpus)}")

    instances_dir = Path(args.repo_root) / "data" / "bench" / "instances"
    top5_path = out_dir / "dense_top5.jsonl"
    npz_path = out_dir / "dense_top100.npz"

    all_indices = []
    all_scores = []
    all_query_ids = []
    all_tasks = []
    all_gold_ranks = []

    with open(top5_path, "w", encoding="utf-8") as out:
        for task in TASKS:
            instances = load_instances(instances_dir / f"{task}.json")
            examples = collect_queries(task, instances)
            query_cache = Path(args.cache_dir) / f"{task}_query_{tag}_{args.max_seq_length}.npy"
            queries = np.load(query_cache).astype(np.float32)
            if queries.shape[0] != len(examples):
                raise SystemExit(f"query cache mismatch {task}: {queries.shape[0]} != {len(examples)}")

            for start in range(0, len(examples), 256):
                end = min(start + 256, len(examples))
                scores = queries[start:end] @ docs.T
                n = end - start
                k = min(args.top_n, scores.shape[1])
                part = np.argpartition(scores, -k, axis=1)[:, -k:]
                part_scores = np.take_along_axis(scores, part, axis=1)
                order = np.argsort(part_scores, axis=1)[:, ::-1]
                idx = np.take_along_axis(part, order, axis=1)
                val = np.take_along_axis(part_scores, order, axis=1)
                all_indices.append(idx)
                all_scores.append(val.astype(np.float32))

                for row in range(n):
                    ex = examples[start + row]
                    inst = instances[ex.instance_index]
                    qid = str(inst.get("instance_id", f"{task}_{start + row}"))
                    gold = set(ex.gold_ids)
                    ranks = [r + 1 for r, sid in enumerate([skill_ids[j] for j in idx[row]]) if sid in gold]
                    gold_rank = min(ranks) if ranks else -1
                    all_query_ids.append(qid)
                    all_tasks.append(task)
                    all_gold_ranks.append(gold_rank)

                    top = []
                    for rank in range(min(args.top_k, idx.shape[1])):
                        j = int(idx[row, rank])
                        top.append(
                            {
                                "rank": rank + 1,
                                "skill_id": skill_ids[j],
                                "score": float(val[row, rank]),
                                "name": str(corpus[j].get("name") or ""),
                                "description": str(corpus[j].get("description") or ""),
                                "content_snippet": skill_texts[j][: args.snippet_chars],
                            }
                        )
                    record = {
                        "query_id": qid,
                        "task": task,
                        "query": ex.query,
                        "reference_answer": str((inst.get("eval_data") or {}).get("answer", ""))[:1000],
                        "gold_ids": ex.gold_ids,
                        "gold_rank_dense": gold_rank,
                        "hit_at_5": bool(gold_rank != -1 and gold_rank <= args.top_k),
                        "top": top,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")

    np.savez_compressed(
        npz_path,
        indices=np.vstack(all_indices).astype(np.int32),
        scores=np.vstack(all_scores).astype(np.float32),
        query_ids=np.array(all_query_ids),
        tasks=np.array(all_tasks),
        gold_ranks=np.array(all_gold_ranks, dtype=np.int32),
    )
    print(f"wrote {top5_path}")
    print(f"wrote {npz_path}")
    hits = sum(1 for r in all_gold_ranks if 0 < r <= args.top_k)
    print(f"queries={len(all_query_ids)} hit@{args.top_k}={hits / max(1, len(all_query_ids)):.4f}")


if __name__ == "__main__":
    main()
