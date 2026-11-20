import argparse
import csv
import json
import random
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top5", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--split", choices=["pilot", "validation"], required=True)
    ap.add_argument("--out-sheet", required=True)
    ap.add_argument("--out-key", required=True)
    ap.add_argument("--seed", type=int, default=20261120)
    args = ap.parse_args()

    top5 = {}
    with open(args.top5, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                top5[r["query_id"]] = r

    sampled = []
    with open(args.sample, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == args.split:
                sampled.append(row)

    rng = random.Random(args.seed + (0 if args.split == "pilot" else 1))
    rng.shuffle(sampled)

    sheet_rows = []
    key_rows = []
    for qi, row in enumerate(sampled, start=1):
        qid = row["query_id"]
        rec = top5[qid]
        skills = list(rec["top"])
        rng.shuffle(skills)
        qblock = f"Q{qi:03d}"
        gold = set(rec["gold_ids"])
        gold_prefixes = {g.split("_")[0] for g in gold}
        for si, skill in enumerate(skills):
            pair_id = f"{qblock}{chr(ord('A') + si)}"
            same_task = skill["skill_id"].split("_")[0] in gold_prefixes
            sheet_rows.append(
                {
                    "pair_id": pair_id,
                    "q_block": qblock,
                    "query": rec["query"],
                    "reference_answer": rec["reference_answer"],
                    "skill_name": skill["name"],
                    "skill_description": skill["description"],
                    "skill_snippet": skill["content_snippet"],
                    "semantic": "",
                    "coverage": "",
                    "judgment": "",
                    "invalid_reason": "",
                    "evidence": "",
                    "confidence": "",
                    "notes": "",
                    "annotator": "",
                }
            )
            key_rows.append(
                {
                    "pair_id": pair_id,
                    "q_block": qblock,
                    "query_id": qid,
                    "task": rec["task"],
                    "gold_ids": ";".join(rec["gold_ids"]),
                    "skill_id": skill["skill_id"],
                    "rank_dense": skill["rank"],
                    "score_dense": skill["score"],
                    "is_gold": int(skill["skill_id"] in gold),
                    "same_task_family": int(same_task),
                }
            )

    with open(args.out_sheet, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sheet_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sheet_rows)
    with open(args.out_key, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)

    print(f"queries={len(sampled)} pairs={len(sheet_rows)}")
    print(f"wrote {args.out_sheet}")
    print(f"wrote {args.out_key}")


if __name__ == "__main__":
    main()
