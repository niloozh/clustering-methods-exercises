"""Simple outlier scores: KDIST, ODIN, mean-shift drift."""

from __future__ import annotations

import numpy as np

from .metrics import squared_euclidean


def mean_shift_outlier_scores(
    data: np.ndarray, k: int = 5, iterations: int = 5
) -> np.ndarray:
    """Score = how far a point drifts when repeatedly replaced by its k-NN mean."""
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        pos = data[i].copy()
        for _ in range(iterations):
            d2 = np.sum((data - pos) ** 2, axis=1)
            nn = np.argpartition(d2, min(k, n - 1))[:k]
            pos = data[nn].mean(axis=0)
        scores[i] = float(np.sqrt(squared_euclidean(data[i], pos)))
    return scores


def kdist_scores(data: np.ndarray, k: int = 2) -> np.ndarray:
    """Distance to the k-th nearest neighbour."""
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    scores = np.zeros(n)
    for i in range(n):
        d2 = np.sum((data - data[i]) ** 2, axis=1)
        d2[i] = np.inf
        knn = np.partition(d2, k - 1)[k - 1]
        scores[i] = float(np.sqrt(knn))
    return scores


def odin_scores(data: np.ndarray, k: int = 2) -> np.ndarray:
    """ODIN-style: higher score when reverse k-NN in-degree is low."""
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    indeg = np.zeros(n, dtype=np.int64)
    for i in range(n):
        d2 = np.sum((data - data[i]) ** 2, axis=1)
        d2[i] = np.inf
        nn = np.argpartition(d2, k - 1)[:k]
        for j in nn:
            indeg[j] += 1
    return (k - indeg).astype(np.float64).clip(min=0)