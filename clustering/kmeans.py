"""Lloyd k-means (with a simple activity log per iteration)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .centroids import reseat_empty_clusters, update_centroids
from .metrics import sse
from .partition import optimal_partition


@dataclass
class KMeansResult:
    centroids: np.ndarray
    partition: np.ndarray  # 1-based labels
    sse: float
    iterations: int
    history: list[dict]


def _init_centroids(
    data: np.ndarray, k: int, init: str, rng: np.random.Generator, centroids=None
) -> np.ndarray:
    if centroids is not None:
        return np.asarray(centroids, dtype=np.float64).copy()
    n = len(data)
    if init in ("random_points", "random"):
        idx = rng.choice(n, size=k, replace=False)
        return data[idx].copy()
    raise ValueError(f"Unknown init: {init}")


def kmeans(
    data: np.ndarray,
    k: int,
    *,
    max_iter: int = 100,
    init: str = "random_points",
    centroids: np.ndarray | None = None,
    seed: int | None = None,
    tol: float = 1e-9,
) -> KMeansResult:
    """Standard k-means / GLA."""
    data = np.asarray(data, dtype=np.float64)
    rng = np.random.default_rng(seed)
    C = _init_centroids(data, k, init, rng, centroids)
    history: list[dict] = []
    partition = optimal_partition(data, C)
    prev_sse = sse(data, C, partition)

    it = 0
    for it in range(1, max_iter + 1):
        C_new = update_centroids(data, partition, k)
        C_new = reseat_empty_clusters(data, C_new, rng)
        # how many centroids actually moved this step
        moved = np.sqrt(((C_new - C) ** 2).sum(axis=1)) > tol
        C = C_new
        partition = optimal_partition(data, C)
        cur_sse = sse(data, C, partition)
        history.append(
            {
                "iteration": it,
                "sse": cur_sse,
                "active": int(moved.sum()),
                "active_pct": 100.0 * float(moved.sum()) / k,
            }
        )
        if abs(prev_sse - cur_sse) <= tol or not moved.any():
            prev_sse = cur_sse
            break
        prev_sse = cur_sse

    return KMeansResult(
        centroids=C, partition=partition, sse=prev_sse, iterations=it, history=history
    )


def kmeans_with_activity(data: np.ndarray, k: int, **kwargs) -> KMeansResult:
    # same as kmeans — history already stores active centroid counts
    return kmeans(data, k, **kwargs)