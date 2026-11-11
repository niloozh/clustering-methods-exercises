"""Centroid (mean) updates."""

from __future__ import annotations

import numpy as np


def mean_centroid(points: np.ndarray) -> np.ndarray:
    """Average of a set of points, computed dimension-wise."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("Cannot compute centroid of an empty set")
    return points.mean(axis=0)


def update_centroids(
    data: np.ndarray, partition: np.ndarray, k: int | None = None
) -> np.ndarray:
    """Centroid step: mean of each cluster given a partition.

    Empty clusters keep a NaN placeholder that callers should replace
    (k-means re-seeds empty clusters from a random data point).
    Labels may be 0- or 1-based.
    """
    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(partition, dtype=np.int64)
    if labels.min() >= 1:
        labels = labels - 1
    if k is None:
        k = int(labels.max()) + 1
    dim = data.shape[1]
    centroids = np.full((k, dim), np.nan, dtype=np.float64)
    for j in range(k):
        members = data[labels == j]
        if len(members):
            centroids[j] = members.mean(axis=0)
    return centroids


def reseat_empty_clusters(
    data: np.ndarray, centroids: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Replace NaN / empty centroids with random data points."""
    out = centroids.copy()
    for j in range(len(out)):
        if np.any(np.isnan(out[j])):
            out[j] = data[rng.integers(0, len(data))]
    return out