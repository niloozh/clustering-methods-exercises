"""Centroid Index (CI) as used in the course."""

from __future__ import annotations

import numpy as np

from .partition import nearest_neighbor


def _nearest_map(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Nearest neighbour in B for each centroid in A."""
    mapping = np.empty(len(A), dtype=np.int64)
    for i, a in enumerate(A):
        j, _ = nearest_neighbor(a, B)
        mapping[i] = j
    return mapping


def centroid_index(centroids: np.ndarray, ground_truth: np.ndarray) -> int:
    """CI = max(orphan GT centroids, orphan detected centroids)."""
    C = np.asarray(centroids, dtype=np.float64)
    G = np.asarray(ground_truth, dtype=np.float64)
    map_cg = _nearest_map(C, G)
    map_gc = _nearest_map(G, C)

    orphans_gt = sum(1 for j in range(len(G)) if j not in set(map_cg.tolist()))
    orphans_det = sum(1 for j in range(len(C)) if j not in set(map_gc.tolist()))
    return int(max(orphans_gt, orphans_det))


def ci_from_partitions(
    data: np.ndarray, partition: np.ndarray, gt_partition: np.ndarray
) -> int:
    """CI after converting partitions to centroids (cluster means)."""
    from .centroids import update_centroids

    labels = np.asarray(partition, dtype=np.int64)
    gt = np.asarray(gt_partition, dtype=np.int64)
    k = int(max(labels.max(), gt.max()))
    # Ensure 1-based for update_centroids helper path
    if labels.min() == 0:
        labels = labels + 1
    if gt.min() == 0:
        gt = gt + 1
    C = update_centroids(data, labels, k)
    G = update_centroids(data, gt, k)
    # Drop empty
    C = C[~np.isnan(C).any(axis=1)]
    G = G[~np.isnan(G).any(axis=1)]
    return centroid_index(C, G)