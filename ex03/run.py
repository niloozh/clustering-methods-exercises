#!/usr/bin/env python3
"""Exercise 3 — Random Swap empirical timing, MIW, Mumford-Shah, graph notes."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clustering.io import load_dataset  # noqa: E402
from clustering.kmeans import kmeans  # noqa: E402
from clustering.random_swap import random_swap, random_swap_trial  # noqa: E402


def mean_internal_weight(intra_left, intra_right, cut_edges) -> float:
    """MIW = average of internal edge weights (course graph example)."""
    all_internal = list(intra_left) + list(intra_right)
    return float(np.mean(all_internal)) if all_internal else 0.0


def mumford_shah_bw(width: int, height: int, rect: int = 10) -> dict:
    """Mumford-Shah for all-white image with a black rect, perfect 2-segment partition.

    Simplified discrete MS:
      E = λ * length(boundary) + μ * Σ_segments area(s) * variance(s)
    For perfect constant segments, variance = 0, so only boundary length remains
    (plus optional area priors depending on the exact formulation used in the lecture).
    """
    # Boundary of the axis-aligned 10x10 square inside the image: perimeter 4*10 = 40
    # (shared with background). Interior variances are zero for perfect B/W segments.
    perimeter = 4 * rect
    area_fg = rect * rect
    area_bg = width * height - area_fg
    # Lecture-style energy with variance term = 0:
    ms_cost = float(perimeter)  # λ=1; data term 0
    return {
        "width": width,
        "height": height,
        "perimeter": perimeter,
        "area_fg": area_fg,
        "area_bg": area_bg,
        "variance_term": 0.0,
        "MS_cost_lambda1": ms_cost,
        "note": "variance=0 for both segments, so MS cost ≈ λ * perimeter (=40 if λ=1)",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Exercise 3")
    p.add_argument("--data", default=str(ROOT / "data" / "s1.txt"))
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--rs-trials", type=int, default=50)
    args = p.parse_args()

    data = load_dataset(args.data)
    print(f"Loaded {args.data}: N={len(data)}")

    # Task 1: empirical comparison — time for k-means vs how many RS trials fit
    t0 = time.perf_counter()
    km = kmeans(data, args.k, max_iter=25, seed=args.seed)
    t_km = time.perf_counter() - t0
    print(f"k-means 25 iters: SSE={km.sse:.6g}, time={t_km:.3f}s")

    t0 = time.perf_counter()
    # Measure average time of one RS trial from the k-means solution
    _ = random_swap_trial(data, km.centroids, local_iters=2, seed=args.seed)
    t1 = time.perf_counter()
    n_probe = 10
    for i in range(n_probe):
        random_swap_trial(data, km.centroids, local_iters=2, seed=args.seed + i)
    t_trial = (time.perf_counter() - t1) / n_probe
    n_fit = t_km / t_trial if t_trial > 0 else float("inf")
    print(f"Avg RS trial time ≈ {t_trial:.4f}s → ~{n_fit:.1f} trials in one k-means budget")

    rs = random_swap(data, args.k, trials=args.rs_trials, seed=args.seed)
    print(
        f"Random Swap ({args.rs_trials} trials): SSE={rs.sse:.6g}, "
        f"improvements={rs.improvements}"
    )

    # Task 3: Mumford-Shah (assume a common lecture size if unspecified — use 100x100)
    ms = mumford_shah_bw(100, 100, 10)
    print("\nMumford-Shah (100x100 with 10x10 black square):", ms)

    # Task 4: MIW from the figure (intra blue weights; cut red weights)
    # From extracted figure description: left internals {7,8,6}, right {3,5,7}, cut edges {6,6}
    left = [7, 8, 6]
    right = [3, 5, 7]
    cut = [6, 6]
    miw = mean_internal_weight(left, right, cut)
    print(f"\nMIW (mean of internal weights {left+right}) = {miw:.4f}")
    print(f"Cut edges (red) = {cut}. Improving MIW: move a boundary vertex if it increases mean internal weight.")

    print(
        "\nTask 5: standard k-means needs vector features + Euclidean SSE. "
        "On a pure similarity/graph example without coordinates, plain k-means does not apply "
        "unless we embed nodes (e.g. spectral embedding) or switch to a graph objective."
    )


if __name__ == "__main__":
    main()