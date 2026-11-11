#!/usr/bin/env python3
"""Exercise 4 — cost relations, text edit distance, medoids."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clustering.edit_distance import (  # noqa: E402
    cluster_medoid_cost,
    edit_distance,
    pairwise_edit_distances,
)
from clustering.io import load_dataset  # noqa: E402
from clustering.kmeans import kmeans  # noqa: E402
from clustering.metrics import (  # noqa: E402
    centroid_distance,
    cluster_sse,
    squared_euclidean,
    sum_pairwise_squared_distances,
)


def demo_sse_vs_pairwise() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 2))
    sse = cluster_sse(X)
    pair = sum_pairwise_squared_distances(X)
    n = len(X)
    # Identity: sum_{i<j} ||xi-xj||^2 = n * SSE
    print(f"Cluster SSE = {sse:.6f}")
    print(f"Sum pairwise squared distances = {pair:.6f}")
    print(f"n * SSE = {n * sse:.6f}  (should match pairwise sum)")


def main() -> None:
    p = argparse.ArgumentParser(description="Exercise 4")
    p.add_argument("--data", default=str(ROOT / "data" / "s2.txt"))
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    print("=== Task 1: SSE vs pairwise ===")
    demo_sse_vs_pairwise()

    A = ["ireadL", "relanE", "rlanZd", "irelLITnd"]
    B = ["fiInVlLand", "filanNM", "finPAlaQd", "finlCnUd"]

    print("\n=== Task 4: pairwise edit distances ===")
    print("Cluster A:")
    DA = pairwise_edit_distances(A)
    for i, a in enumerate(A):
        for j in range(i + 1, len(A)):
            print(f"  d({a},{A[j]}) = {DA[i, j]}")
    print("Cluster B:")
    DB = pairwise_edit_distances(B)
    for i, a in enumerate(B):
        for j in range(i + 1, len(B)):
            print(f"  d({a},{B[j]}) = {DB[i, j]}")

    print("\n=== Task 5: medoids ===")
    for name, items in ("A", A), ("B", B):
        info = cluster_medoid_cost(items)
        print(
            f"Cluster {name}: medoid={info['medoid']} "
            f"(index {info['medoid_index']}), cost={info['cost']}"
        )
        # Could a non-member string be a better prototype? For medoid cost we
        # are restricted to cluster members; a freer prototype (e.g. consensus
        # string) might lower total edit distance.
    print(
        "Better non-medoid representative? A consensus / Steiner string can beat "
        "the medoid, but it may not equal any data object (medoid constraint)."
    )

    print("\n=== Bonus Task 6: CentroidDistance on S2 k-means ===")
    data = load_dataset(args.data)
    km = kmeans(data, args.k, max_iter=30, seed=args.seed)
    # cluster sizes
    sizes = np.bincount(km.partition, minlength=args.k + 1)[1:]
    print(f"k-means SSE={km.sse:.6g}")
    print("Pairwise CentroidDistance (Ward-style n_i n_j/(n_i+n_j) ||ci-cj||^2):")
    for i in range(args.k):
        for j in range(i + 1, min(i + 3, args.k)):  # print a few
            cd = centroid_distance(km.centroids[i], km.centroids[j], int(sizes[i]), int(sizes[j]))
            raw = int(sizes[i]) * int(sizes[j]) * squared_euclidean(km.centroids[i], km.centroids[j])
            print(f"  ({i+1},{j+1}): ward={cd:.4g}, n1*n2*d2={raw:.4g}")


if __name__ == "__main__":
    main()