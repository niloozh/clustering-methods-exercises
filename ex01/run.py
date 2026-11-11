#!/usr/bin/env python3
"""Exercise 1 — dummy clustering tool, distance, SSE, average pairwise distance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clustering.io import load_dataset, save_centroids, save_partition  # noqa: E402
from clustering.metrics import euclidean, mean_pairwise_distance, sse_random_centroid  # noqa: E402


def dummy_clustering(data: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(a) k random data points as centroids; (b) random partition in 1..k."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), size=k, replace=False)
    centroids = data[idx].copy()
    partition = rng.integers(1, k + 1, size=len(data))
    return centroids, partition


def main() -> None:
    p = argparse.ArgumentParser(description="Exercise 1: dummy clustering")
    p.add_argument(
        "--data",
        default=str(ROOT / "data" / "s1.txt"),
        help="Path to SIPU-format dataset",
    )
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "output"))
    p.add_argument(
        "--pairwise-sample",
        type=int,
        default=400,
        help="Subsample size for average pairwise distance (full N^2 is slow)",
    )
    args = p.parse_args()

    data = load_dataset(args.data)
    print(f"Loaded {args.data}: N={len(data)}, D={data.shape[1]}")

    centroids, partition = dummy_clustering(data, args.k, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_centroids(centroids, out / "centroid.txt")
    save_partition(partition, out / "partition.txt")
    print(f"Wrote {out / 'centroid.txt'} and {out / 'partition.txt'}")

    # Task 2: distance + random-centroid SSE + average pairwise distance
    d01 = euclidean(data[0], data[1])
    print(f"Example Euclidean distance(data[0], data[1]) = {d01:.4f}")

    rng = np.random.default_rng(args.seed)
    random_sse = sse_random_centroid(data, centroids, rng)
    print(f"SSE vs randomly chosen centroids = {random_sse:.4g}")

    m = min(args.pairwise_sample, len(data))
    sample = data[rng.choice(len(data), size=m, replace=False)]
    avg_d = mean_pairwise_distance(sample)
    print(f"Average pairwise distance on {m} subsampled points = {avg_d:.4f}")


if __name__ == "__main__":
    main()