import argparse
import json
import os

import numpy as np

from data import skill_text
from utils import hash_strings, l2_normalize, model_tag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--shards", required=True, help="comma-separated .npy paths in corpus order")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    corpus_path = os.path.join(args.repo_root, "data", "bench", "corpus", "corpus.json")
    with open(corpus_path, encoding="utf-8") as fh:
        corpus = json.load(fh)
    texts = [skill_text(skill) for skill in corpus]

    shard_paths = [p.strip() for p in args.shards.split(",") if p.strip()]
    arr = np.concatenate([np.load(p) for p in shard_paths], axis=0)
    if arr.shape[0] != len(texts):
        raise SystemExit(f"row mismatch: {arr.shape[0]} != {len(texts)}")
    arr = l2_normalize(arr).astype(np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = model_tag(args.model_path)
    out_path = os.path.join(args.out_dir, f"all_{tag}_{args.max_seq_length}.npy")
    np.save(out_path, arr)

    meta = {
        "n_texts": len(texts),
        "text_sha1": hash_strings(texts),
        "prompt_name": None,
        "embedding_dim": int(arr.shape[1]),
        "max_seq_length": int(args.max_seq_length),
        "model_path": args.model_path,
        "cache_written_by": "SkillSight",
    }
    meta_path = out_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"merged {len(shard_paths)} shards -> {out_path} {arr.shape}")
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
