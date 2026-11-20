import argparse
import json
import os

import numpy as np

from data import skill_text
from encoder import DenseEncoder
from utils import l2_normalize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    corpus_path = os.path.join(args.repo_root, "data", "bench", "corpus", "corpus.json")
    with open(corpus_path, encoding="utf-8") as fh:
        corpus = json.load(fh)
    texts = [skill_text(skill) for skill in corpus[args.start:args.end]]

    encoder = DenseEncoder(
        args.model_path, device="cuda:0", max_seq_length=args.max_seq_length
    )
    arr = encoder.encode(texts, batch_size=args.batch_size, show_progress=True)
    arr = l2_normalize(arr)
    np.save(args.out, arr.astype(np.float32))
    print(f"saved {args.out} {arr.shape}", flush=True)


if __name__ == "__main__":
    main()
