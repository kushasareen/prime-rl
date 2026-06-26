"""Build a filtered, size-capped local copy of reasoning-core/formal-reasoning-env.

reasoning-core-env only exposes num_examples/seed — not difficulty. The published
dataset carries a per-example `level` (0-4) and `task` (40 types), so we filter here
(drop trivial levels, optionally pick tasks, cap size for the data-efficiency regime)
and write parquet train/test splits. Point the env at the output dir via its
`dataset_name` arg; it loads offline from local files on compute nodes.

Run on the LOGIN node (needs the cached dataset):
    uv run --no-sync python scripts/tamia/prep_reasoning_core.py \
        --out $SCRATCH/datasets/reasoning_core_l2-3_n512
"""

import argparse
import collections
import os

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="reasoning-core/formal-reasoning-env")
    ap.add_argument("--min-level", type=int, default=2)
    ap.add_argument("--max-level", type=int, default=3)
    ap.add_argument("--tasks", nargs="*", default=None, help="optional task-name subset")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = load_dataset(args.src, split="train")
    ds = ds.filter(lambda x: args.min_level <= x["level"] <= args.max_level)
    if args.tasks:
        keep = set(args.tasks)
        ds = ds.filter(lambda x: x["task"] in keep)
    ds = ds.shuffle(seed=args.seed)

    need = args.n_train + args.n_eval
    if len(ds) < need:
        raise ValueError(f"only {len(ds)} rows after filter, need {need}")

    train = ds.select(range(args.n_train))
    test = ds.select(range(args.n_train, need))

    os.makedirs(args.out, exist_ok=True)
    train.to_parquet(os.path.join(args.out, "train.parquet"))
    test.to_parquet(os.path.join(args.out, "test.parquet"))

    print(f"train={len(train)} test={len(test)} -> {args.out}")
    print("levels:", dict(sorted(collections.Counter(train["level"]).items())))
    print("top tasks:", dict(collections.Counter(train["task"]).most_common(10)))


if __name__ == "__main__":
    main()
