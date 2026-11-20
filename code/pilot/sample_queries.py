import argparse
import csv
import json
import random
from collections import defaultdict


PILOT_N = {
    "theoremqa": 14,
    "logicbench": 14,
    "medcalcbench": 20,
    "toolqa": 26,
    "bigcodebench": 21,
    "champ": 5,
}
VALID_N = {
    "theoremqa": 28,
    "logicbench": 28,
    "medcalcbench": 41,
    "toolqa": 53,
    "bigcodebench": 42,
    "champ": 8,
}
FAIL_RATIO = {"pilot": 0.70, "validation": 0.80}


def split_task(rows, n, fail_ratio, rng):
    fails = [r for r in rows if not r["hit_at_5"]]
    succ = [r for r in rows if r["hit_at_5"]]
    rng.shuffle(fails)
    rng.shuffle(succ)
    want_fail = int(round(n * fail_ratio))
    want_succ = n - want_fail
    take_fail = fails[:want_fail]
    take_succ = succ[:want_succ]
    if len(take_fail) < want_fail:
        need = want_fail - len(take_fail)
        take_succ = succ[: min(len(succ), want_succ + need)]
    if len(take_succ) < want_succ:
        need = want_succ - len(take_succ)
        take_fail = fails[: min(len(fails), len(take_fail) + need)]
    return take_fail, take_succ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top5", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-key", required=True)
    ap.add_argument("--seed", type=int, default=20261120)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.top5, encoding="utf-8") if line.strip()]
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    rng = random.Random(args.seed)
    sampled = []
    key = {"seed": args.seed, "pilot": {}, "validation": {}}
    for split, sizes in (("pilot", PILOT_N), ("validation", VALID_N)):
        for task, n in sizes.items():
            fails, succ = split_task(by_task[task], n, FAIL_RATIO[split], rng)
            chosen = fails + succ
            rng.shuffle(chosen)
            key[split][task] = {
                "fail": [r["query_id"] for r in fails],
                "success": [r["query_id"] for r in succ],
            }
            for r in chosen:
                sampled.append(
                    {
                        "split": split,
                        "query_id": r["query_id"],
                        "task": task,
                        "stratum": "fail" if not r["hit_at_5"] else "success",
                        "query": r["query"],
                        "gold_ids": ";".join(r["gold_ids"]),
                        "hit_at_5": int(r["hit_at_5"]),
                        "gold_rank_dense": r["gold_rank_dense"],
                    }
                )

    with open(args.out_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "split",
                "query_id",
                "task",
                "stratum",
                "query",
                "gold_ids",
                "hit_at_5",
                "gold_rank_dense",
            ],
        )
        writer.writeheader()
        writer.writerows(sampled)

    with open(args.out_key, "w", encoding="utf-8") as fh:
        json.dump(key, fh, ensure_ascii=False, indent=2)

    from collections import Counter

    print("split counts", Counter(r["split"] for r in sampled))
    print("task x split")
    for split in ("pilot", "validation"):
        print(split, Counter(r["task"] for r in sampled if r["split"] == split))
        print(split, "stratum", Counter(r["stratum"] for r in sampled if r["split"] == split))
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_key}")


if __name__ == "__main__":
    main()
