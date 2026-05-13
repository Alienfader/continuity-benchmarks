#!/usr/bin/env python3
"""
longmemeval-subsample.py — produce a stratified-by-category subsample of
LongMemEval-S for budget-friendly benchmarking.

Default: 50 questions stratified across the 6 question types in
longmemeval_s_cleaned.json. Deterministic (seed=42) so re-running
produces the same sample.

Usage:
  python3 scripts/longmemeval-subsample.py
  python3 scripts/longmemeval-subsample.py --n 100 --seed 7
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "fixtures" / "longmemeval" / "longmemeval_s_cleaned.json"


def stratified_sample(records, n, seed):
    by_type = defaultdict(list)
    for r in records:
        by_type[r["question_type"]].append(r)

    rng = random.Random(seed)
    types = sorted(by_type.keys())

    per_type_base = n // len(types)
    remainder = n - per_type_base * len(types)
    quotas = {t: per_type_base for t in types}
    types_sorted_desc = sorted(types, key=lambda t: -len(by_type[t]))
    for t in types_sorted_desc[:remainder]:
        quotas[t] += 1

    sampled = []
    for t in types:
        pool = by_type[t]
        take = min(quotas[t], len(pool))
        sampled.extend(rng.sample(pool, take))
    return sampled, quotas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50, help="target sample size (default 50)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source", default=str(SOURCE))
    p.add_argument("--output", default=None,
                   help="output path; default fixtures/longmemeval/sample-<n>.json")
    args = p.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"source not found: {src}\n  download with:\n  curl -sSLo {src} https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json")

    records = json.loads(src.read_text())
    print(f"loaded {len(records)} records from {src.name}")

    sample, quotas = stratified_sample(records, args.n, args.seed)
    print(f"sampled {len(sample)} (seed={args.seed}):")
    from collections import Counter
    got = Counter(r["question_type"] for r in sample)
    for t in sorted(got.keys()):
        print(f"  {t:30s} quota={quotas.get(t, 0)}  got={got[t]}")

    out = Path(args.output) if args.output else REPO / "fixtures" / "longmemeval" / f"sample-{args.n}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sample, indent=2))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
