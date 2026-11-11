#!/usr/bin/env python3
"""Exercise 6 — elbow / derivatives, sub-sampling, grid initialization."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clustering.grid_init import grid_based_init, total_grid_cells  # noqa: E402
from clustering.io import load_dataset  # noqa: E402
from clustering.kmeans import kmeans  # noqa: E402


def first_second_derivatives(sse_vals: list[float]) -> tuple[list[float], list[float]]:
    d1 = [sse_vals[i + 1] - sse_vals[i] for i in range(len(sse_vals) - 1)]
    d2 = [d1[i + 1] - d1[i] for i in range(len(d1) - 1)]
    return d1, d2


def main() -> None:
    p = argparse.ArgumentParser(description="Exercise 6")
    p.add_argument("--data", default=str(ROOT / "data" / "s1.txt"))
    p.add_argument("--kmax", type=int, default=20)
    p.add_argument("--true-k", type=int, default=15)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "output"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.data)
    print(f"Loaded {args.data}: N={len(data)}")

    # Task 1–2: SSE vs k + derivatives
    rows = []
    sse_vals = []
    ks = list(range(1, args.kmax + 1))
    for k in ks:
        # Several restarts; keep best SSE
        best = None
        for r in range(3):
            res = kmeans(data, k, max_iter=25, seed=args.seed + 100 * k + r)
            if best is None or res.sse < best.sse:
                best = res
        sse_vals.append(best.sse)
        rows.append({"k": k, "sse": best.sse})
        print(f"k={k:2d}  SSE={best.sse:.6g}")

    d1, d2 = first_second_derivatives(sse_vals)
    with (out / "sse_vs_k.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "sse", "d1", "d2"])
        w.writeheader()
        for i, k in enumerate(ks):
            w.writerow(
                {
                    "k": k,
                    "sse": sse_vals[i],
                    "d1": d1[i] if i < len(d1) else "",
                    "d2": d2[i] if i < len(d2) else "",
                }
            )
    # Heuristic knee: largest positive jump in second derivative of -SSE shape
    # (look at max of d2 of sse which is usually near the elbow for convex curves)
    if d2:
        knee_idx = int(np.argmax(d2)) + 1  # d2[i] corresponds to k = i+2 roughly
        print(f"Heuristic knee near k≈{ks[knee_idx]} (from argmax second derivative)")
    print(f"True k for S1 = {args.true_k}")

    # Task 3: sub-sampling stability
    print("\n=== Task 3: sub-sampling ===")
    full = kmeans(data, args.true_k, max_iter=25, seed=args.seed)
    print(f"Full data SSE={full.sse:.6g}")
    for frac in (0.2, 0.1, 0.05, 0.02):
        rng = np.random.default_rng(args.seed)
        m = max(args.true_k * 5, int(frac * len(data)))
        sample = data[rng.choice(len(data), size=m, replace=False)]
        res = kmeans(sample, args.true_k, max_iter=25, seed=args.seed)
        res_k1 = kmeans(sample, args.true_k + 1, max_iter=25, seed=args.seed)
        print(
            f"frac≈{frac:.2f} (N={m}): SSE(k)={res.sse:.6g}, SSE(k+1)={res_k1.sse:.6g}"
        )

    # Task 4–5: grid init
    print("\n=== Tasks 4–5: grid init ===")
    d = data.shape[1]
    print(f"Cells per attribute=100 → total cells = 100^{d} = {total_grid_cells(d):.6g}")
    print("For D=10: 100^10 = 1e20 (impossible to materialise). For D=100: hopeless.")
    print("Choose cells_per_dim so that expected occupied cells ≈ O(k)..O(N).")

    C_grid = grid_based_init(data, args.true_k, cells_per_dim=50, seed=args.seed)
    km_grid = kmeans(data, args.true_k, centroids=C_grid, max_iter=25, seed=args.seed)
    km_rand = kmeans(data, args.true_k, max_iter=25, seed=args.seed)
    print(f"k-means random init SSE = {km_rand.sse:.6g}")
    print(f"k-means grid init   SSE = {km_grid.sse:.6g}")


if __name__ == "__main__":
    main()