#!/usr/bin/env python3
"""Exercise 7 — divisive clustering, GA crossover ideas."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clustering.divisive import divisive_clustering  # noqa: E402
from clustering.io import load_dataset  # noqa: E402
from clustering.metrics import squared_euclidean  # noqa: E402
from clustering.partition import nearest_neighbor  # noqa: E402


def toy_divisive_trace() -> None:
    """Task 2: step-by-step divisive on the 10-point sketch (approx. coords)."""
    pts = np.array(
        [
            [0.0, 1.0],  # 1
            [0.0, 3.0],  # 2
            [1.0, 1.0],  # 3
            [1.0, 2.0],  # 4
            [0.5, 2.5],  # 5
            [2.0, 1.5],  # 6
            [3.0, 0.5],  # 7
            [4.0, 0.5],  # 8
            [2.5, 2.5],  # 9
            [3.5, 2.0],  # 10
        ],
        dtype=np.float64,
    )
    print("Start: 1 cluster = all points {1..10}, centroid = mean.")
    c0 = pts.mean(axis=0)
    print(f"  c0 = {c0}")
    print("Split cluster with largest SSE using 2-means; repeat until desired k.")
    print("Illustrative first split often separates left group {1..5} from right {6..10}.")
    left = pts[:5]
    right = pts[5:]
    print(f"  left mean={left.mean(0)}, right mean={right.mean(0)}")
    print("Continue splitting the child with larger SSE, etc.")


def multi_parent_crossover(parents: list[np.ndarray], k: int, data: np.ndarray) -> np.ndarray:
    """Task 3: pool centroids from all parents; greedily pick k diverse ones.

    Simple algorithm:
    1. Concatenate all parent centroids into a candidate pool.
    2. Score each candidate by how many data points would prefer it (Voronoi mass)
       among the pool, or by maximin diversity.
    3. Maximin: pick the globally densest candidate first, then repeatedly add the
       candidate farthest from those already chosen, until k centroids selected.
    """
    pool = np.vstack(parents)
    # Start with candidate closest to data mean (stable).
    mean = data.mean(axis=0)
    first, _ = nearest_neighbor(mean, pool)
    chosen = [pool[first]]
    used = {first}
    while len(chosen) < k:
        best_i, best_score = None, -1.0
        for i, c in enumerate(pool):
            if i in used:
                continue
            # distance to nearest already chosen
            dmin = min(squared_euclidean(c, s) for s in chosen)
            if dmin > best_score:
                best_score = dmin
                best_i = i
        used.add(best_i)
        chosen.append(pool[best_i])
    return np.asarray(chosen, dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description="Exercise 7")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["s1.txt", "s2.txt", "s3.txt", "s4.txt"],
    )
    p.add_argument("--k", type=int, default=15)
    args = p.parse_args()

    print("=== Task 1: divisive clustering on S1–S4 ===")
    for name in args.datasets:
        path = ROOT / "data" / name
        data = load_dataset(path)
        t0 = time.perf_counter()
        res = divisive_clustering(data, args.k, seed=args.seed)
        # res.seconds already measured inside; also wall here
        print(
            f"{name}: SSE={res.sse:.6g}, time={res.seconds:.2f}s "
            f"(wall {time.perf_counter()-t0:.2f}s)"
        )

    print("\n=== Task 2: toy divisive walk-through ===")
    toy_divisive_trace()

    print("\n=== Task 3: multi-parent crossover demo ===")
    data = load_dataset(ROOT / "data" / "s1.txt")
    rng = np.random.default_rng(args.seed)
    # Two parents = random point subsets as fake centroids
    parent_red = data[rng.choice(len(data), size=args.k, replace=False)]
    parent_blue = data[rng.choice(len(data), size=args.k, replace=False)]
    child = multi_parent_crossover([parent_red, parent_blue], args.k, data)
    print(f"Pooled {len(parent_red)+len(parent_blue)} centroids → selected {len(child)}")

    print("\n=== Task 4: unbalance dataset ===")
    print(
        "Unbalance has a few huge dense clusters and several tiny ones. "
        "Maximin crossover from two parents can still under-represent tiny clusters "
        "if neither parent placed a centroid there. It *can* work if the pooled set "
        "covers both large and small modes; otherwise the child inherits the blind spot."
    )

    print("\n=== Task 5: outlier detection inside GA crossover ===")
    print(
        "Yes — treat parent centroids as a point set and score them (e.g. KDIST / mean-shift). "
        "High-score 'outlier centroids' often sit in empty space (useless prototypes) and "
        "can be discarded before maximin selection; low-score centroids sit in dense modes "
        "and are preferred. Caution: a legitimate centroid of a small/rare cluster can look "
        "like an outlier — protect low-SSE / high-mass centroids from removal."
    )


if __name__ == "__main__":
    main()