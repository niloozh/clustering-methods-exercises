#!/usr/bin/env python3
"""Exercise 5 — evaluation indices + outlier ideas + CI / mean-shift demos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clustering.centroid_index import centroid_index  # noqa: E402
from clustering.io import load_dataset  # noqa: E402
from clustering.kmeans import kmeans  # noqa: E402
from clustering.outlier import kdist_scores, mean_shift_outlier_scores, odin_scores  # noqa: E402


def jaccard(a: int, b: int, shared: int) -> float:
    return shared / (a + b - shared) if (a + b - shared) else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Exercise 5")
    p.add_argument("--data", default=str(ROOT / "data" / "s1.txt"))
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    print("=== Tasks 1–3 (figure-based) ===")
    print("See answers.md for shared-point / Jaccard / CI calculations from ex5.pdf figures.")

    print("\n=== Task 4: outlier methods on a tiny 2-d toy (5 inliers + far points) ===")
    # Construct a clear toy matching the spirit of the figure: compact pair + far points
    toy = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.1, -0.1],
            [5.0, 5.0],
            [6.0, -4.0],
            [-5.0, 6.0],
            [4.0, -6.0],
            [-6.0, -5.0],
        ],
        dtype=np.float64,
    )
    print("KDIST k=2:", np.round(kdist_scores(toy, k=2), 3))
    print("ODIN  k=2:", odin_scores(toy, k=2))
    print("Mean-shift scores:", np.round(mean_shift_outlier_scores(toy, k=2, iterations=5), 3))
    print(
        "Medoid-shift: similar to mean-shift but moves to the medoid of the k-NN set; "
        "far points still receive large drift scores."
    )

    print("\n=== Task 5: choosing an outlier threshold ===")
    print(
        "Idea: sort scores descending; look for the largest gap (elbow) between consecutive "
        "scores; or fit a two-component model (inlier vs outlier) on the 1-d scores; "
        "or use OTST / knee detection. The gap between ~19 and ~59 in the figure is exactly "
        "such an elbow."
    )

    print("\n=== Bonus: CI on S1 k-means vs a second run (proxy GT) ===")
    data = load_dataset(args.data)
    a = kmeans(data, args.k, max_iter=25, seed=args.seed)
    b = kmeans(data, args.k, max_iter=25, seed=args.seed + 99)
    ci = centroid_index(a.centroids, b.centroids)
    print(f"CI between two k-means runs = {ci} (0 means matching centroids)")

    print("\n=== Bonus: mean-shift outlier filtering before centroid update ===")
    scores = mean_shift_outlier_scores(data[::10], k=5, iterations=3)  # subsample for speed
    thr = np.percentile(scores, 95)
    print(f"Subsample N={len(scores)}; 95th percentile score thr={thr:.4f}; "
          f"flagged={(scores >= thr).sum()} points")


if __name__ == "__main__":
    main()