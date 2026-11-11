"""Simple bisecting / divisive k-means style clustering."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .kmeans import kmeans
from .metrics import cluster_sse, sse
from .partition import optimal_partition


@dataclass
class DivisiveResult:
    centroids: np.ndarray
    partition: np.ndarray
    sse: float
    seconds: float


def divisive_clustering(
    data: np.ndarray,
    k: int,
    *,
    seed: int | None = None,
    local_iters: int = 20,
) -> DivisiveResult:
    """Greedy divisive clustering: repeatedly split the cluster with largest SSE.

    Each split runs 2-means on the chosen cluster, then a short global k-means
    fine-tune on the current centroid set.
    """
    data = np.asarray(data, dtype=np.float64)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    # Start with one cluster = global mean.
    centroids = [data.mean(axis=0)]
    partition = np.ones(len(data), dtype=np.int64)

    while len(centroids) < k:
        # Choose cluster with largest SSE to split.
        best_j = 0
        best_cost = -1.0
        for j in range(len(centroids)):
            pts = data[partition == (j + 1)]
            if len(pts) < 2:
                continue
            cost = cluster_sse(pts)
            if cost > best_cost:
                best_cost = cost
                best_j = j

        pts = data[partition == (best_j + 1)]
        if len(pts) < 2:
            break
        split = kmeans(pts, 2, max_iter=local_iters, seed=int(rng.integers(0, 2**31 - 1)))
        # Replace old centroid with first child; append second child.
        centroids[best_j] = split.centroids[0]
        centroids.append(split.centroids[1])
        C = np.asarray(centroids, dtype=np.float64)
        # Optional local fine-tune with current k'
        tuned = kmeans(data, len(C), centroids=C, max_iter=5, seed=seed)
        centroids = [c for c in tuned.centroids]
        partition = tuned.partition

    C = np.asarray(centroids, dtype=np.float64)
    # Final global fine-tune
    final = kmeans(data, k, centroids=C[:k], max_iter=local_iters, seed=seed)
    elapsed = time.perf_counter() - t0
    return DivisiveResult(
        centroids=final.centroids,
        partition=final.partition,
        sse=final.sse,
        seconds=elapsed,
    )