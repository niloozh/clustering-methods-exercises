"""Distances and SSE helpers."""

from __future__ import annotations

import numpy as np


def squared_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.dot(diff, diff))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(squared_euclidean(a, b)))


def sse(data: np.ndarray, centroids: np.ndarray, partition: np.ndarray | None = None) -> float:
    """Sum of squared errors to assigned centroids.

    If partition is missing, assign each point to the nearest centroid.
    Labels can be 0-based or 1-based.
    """
    data = np.asarray(data, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    if partition is None:
        from .partition import optimal_partition

        partition = optimal_partition(data, centroids)
    labels = np.asarray(partition, dtype=np.int64)
    if labels.min() >= 1 and labels.max() <= len(centroids):
        idx = labels - 1
    else:
        idx = labels - labels.min()
    total = 0.0
    for i, x in enumerate(data):
        total += squared_euclidean(x, centroids[idx[i]])
    return float(total)


def sse_random_centroid(
    data: np.ndarray, centroids: np.ndarray, rng: np.random.Generator | None = None
) -> float:
    """Ex1: each point compared to a randomly chosen centroid (not NN)."""
    rng = rng or np.random.default_rng()
    data = np.asarray(data, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    choices = rng.integers(0, len(centroids), size=len(data))
    total = 0.0
    for i, x in enumerate(data):
        total += squared_euclidean(x, centroids[choices[i]])
    return float(total)


def mean_pairwise_distance(data: np.ndarray) -> float:
    """Mean Euclidean distance over pairs i < j (slow for large N)."""
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += euclidean(data[i], data[j])
            count += 1
    return total / count


def cluster_sse(points: np.ndarray) -> float:
    """SSE of one cluster w.r.t. its mean."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return 0.0
    c = points.mean(axis=0)
    return float(np.sum((points - c) ** 2))


def sum_pairwise_squared_distances(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += squared_euclidean(points[i], points[j])
    return float(total)


def centroid_distance(c1: np.ndarray, c2: np.ndarray, n1: int, n2: int) -> float:
    """Ward-style: n1*n2/(n1+n2) * ||c1-c2||^2."""
    d2 = squared_euclidean(c1, c2)
    return float(n1 * n2 / (n1 + n2) * d2) if (n1 + n2) else 0.0