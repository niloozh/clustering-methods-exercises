"""Nearest-neighbour search and optimal partition."""

from __future__ import annotations

import numpy as np

from .metrics import squared_euclidean


def nearest_neighbor(point: np.ndarray, candidates: np.ndarray) -> tuple[int, float]:
    """Return (index, squared_distance) of the nearest candidate to ``point``.

    ``candidates`` may be the full dataset or a set of centroids.
    """
    point = np.asarray(point, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    best_i = 0
    best_d = squared_euclidean(point, candidates[0])
    for i in range(1, len(candidates)):
        d = squared_euclidean(point, candidates[i])
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d


def optimal_partition(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign every point to its nearest centroid.

    Returns 1-based cluster labels (course / Clusterator style).
    """
    data = np.asarray(data, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    labels = np.empty(len(data), dtype=np.int64)
    for i, x in enumerate(data):
        j, _ = nearest_neighbor(x, centroids)
        labels[i] = j + 1
    return labels