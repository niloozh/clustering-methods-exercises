#!/usr/bin/env python3
"""Exercise 2 — optimal partition, k-means, activity, MPS demo, RS budget, swap trials."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clustering.io import load_dataset, save_centroids, save_partition  # noqa: E402
from clustering.kmeans import kmeans  # noqa: E402
from clustering.metrics import sse  # noqa: E402
from clustering.partition import nearest_neighbor, optimal_partition  # noqa: E402
from clustering.random_swap import compare_km_vs_rs_budget, random_swap_trial  # noqa: E402


def mps_demo() -> None:
    """Task 4: MPS on x-axis for the 10-point toy figure.

    Point coordinates are read left-to-right / bottom-to-top from the course sketch:
    We assign approximate 2-d positions consistent with the numbered layout
    (1..10). Query = midpoint between points 4 and 10.
    """
    # Approximate layout matching the exercise sketch (unit spacing).
    pts = {
        1: np.array([0.0, 1.0]),
        2: np.array([0.0, 3.0]),
        3: np.array([1.0, 1.0]),
        4: np.array([1.0, 2.0]),
        5: np.array([0.5, 2.5]),
        6: np.array([2.0, 1.5]),
        7: np.array([3.0, 0.5]),
        8: np.array([4.0, 0.5]),
        9: np.array([2.5, 2.5]),
        10: np.array([3.5, 2.0]),
    }
    order = list(range(1, 11))
    query = 0.5 * (pts[4] + pts[10])
    print(f"MPS demo query (midpoint 4 & 10) = {query}")
    print("Projection axis = x. Process points 1..10:")

    # Sort candidates by projected x for illustration of pruning idea.
    projections = sorted(((i, pts[i][0]) for i in order), key=lambda t: t[1])
    print("Projection order by x:", [i for i, _ in projections])

    best_i = None
    best_d2 = float("inf")
    for i in order:
        # Partial / projected lower bound on x only: (dx)^2 ≤ full squared distance.
        dx2 = (pts[i][0] - query[0]) ** 2
        if dx2 >= best_d2:
            print(f"  point {i}: prune (dx^2={dx2:.3f} >= best={best_d2:.3f})")
            continue
        d2 = float(np.sum((pts[i] - query) ** 2))
        print(f"  point {i}: evaluate d^2={d2:.3f} (was best {best_d2:.3f})")
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    print(f"Nearest neighbour = point {best_i}, d^2={best_d2:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Exercise 2")
    p.add_argument("--data", default=str(ROOT / "data" / "a1.txt"))
    p.add_argument("--k", type=int, default=20)  # A1 has 20 clusters
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "output"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.data)
    print(f"Loaded {args.data}: N={len(data)}, D={data.shape[1]}")

    # Task 1: random centroids from ex1 style → optimal partition + SSE
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(data), size=args.k, replace=False)
    random_C = data[idx].copy()
    part = optimal_partition(data, random_C)
    s = sse(data, random_C, part)
    print(f"Task1 optimal partition SSE for random centroids = {s:.6g}")
    i_nn, d_nn = nearest_neighbor(data[0], random_C)
    print(f"Task1 NN of data[0] among centroids → index {i_nn}, d^2={d_nn:.4g}")

    # Task 2–3: k-means + activity
    result = kmeans(data, args.k, max_iter=40, seed=args.seed)
    print(f"Task2 k-means SSE = {result.sse:.6g} after {result.iterations} iters")
    save_centroids(result.centroids, out / "centroid.txt")
    save_partition(result.partition, out / "partition.txt")

    with (out / "activity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iteration", "sse", "active", "active_pct"])
        w.writeheader()
        for row in result.history:
            w.writerow(row)
    print(f"Wrote activity table → {out / 'activity.csv'}")
    if result.history:
        print("Activity (first/last):", result.history[0], "...", result.history[-1])

    # Task 4
    print("\n--- Task 4: MPS demo ---")
    mps_demo()

    # Task 5: theoretical swap budget
    print("\n--- Task 5: RS time budget ---")
    budget = compare_km_vs_rs_budget(data, k=args.k, km_iters=25, alpha=4.0, local_iters=2)
    print(budget)
    print(
        f"With α=4, k={args.k}: about {budget['n_swaps']:.2f} trial swaps "
        f"fit in the time of 25 k-means iterations."
    )
    # Closed form for general k: 25k / (4 + 2k)
    print(f"Formula 25k/(4+2k) with k={args.k} = {25 * args.k / (4 + 2 * args.k):.4f}")
    print(f"For the symbolic answer with general k: 25k/(α+2k); α=4 → 25k/(4+2k)")

    # Bonus: 100 swaps from a short k-means start
    print("\n--- Bonus Task 6: 100 random swaps ---")
    warm = kmeans(data, args.k, max_iter=5, seed=args.seed)
    before = warm.sse
    best_C = warm.centroids
    best = before
    improvements = 0
    for t in range(100):
        C, P, s = random_swap_trial(data, best_C, local_iters=2, seed=args.seed + t + 1)
        if s < best:
            improvements += 1
            best = s
            best_C = C
    print(f"SSE before swaps = {before:.6g}")
    print(f"Best SSE after 100 trials = {best:.6g}")
    print(f"Trials that improved the running best = {improvements}/100")
    print(
        "Early swaps can escape a bad local minimum; later most random swaps "
        "are worse and get rejected."
    )


if __name__ == "__main__":
    main()