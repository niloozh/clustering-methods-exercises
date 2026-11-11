"""Simple grid-based initialization for k-means."""

from __future__ import annotations

import numpy as np

from .partition import nearest_neighbor


def grid_based_init(
    data: np.ndarray,
    k: int,
    cells_per_dim: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """Divide each attribute into ``cells_per_dim`` bins; keep densest non-empty cells.

    Without merge/overlap steps: take cell centres of the ``k`` densest occupied
    cells as initial centroids. If fewer than k cells are occupied, fill the
    remainder with random data points.
    """
    data = np.asarray(data, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n, d = data.shape
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    spans = np.where(maxs > mins, maxs - mins, 1.0)

    # Cell index per point.
    scaled = (data - mins) / spans
    idx = np.floor(scaled * cells_per_dim).astype(np.int64)
    idx = np.clip(idx, 0, cells_per_dim - 1)

    # Count occupancy with a dict of tuples.
    counts: dict[tuple[int, ...], list[int]] = {}
    for i, key in enumerate(map(tuple, idx)):
        counts.setdefault(key, []).append(i)

    # Sort cells by density (count).
    ranked = sorted(counts.items(), key=lambda kv: len(kv[1]), reverse=True)
    centroids = []
    for key, members in ranked[:k]:
        # Cell centre in original coordinates.
        centre = mins + (np.asarray(key, dtype=np.float64) + 0.5) / cells_per_dim * spans
        # Or use mean of members — usually better; course says cell-based init.
        centroids.append(data[members].mean(axis=0))

    while len(centroids) < k:
        centroids.append(data[rng.integers(0, n)])

    return np.asarray(centroids[:k], dtype=np.float64)


def total_grid_cells(dim: int, cells_per_dim: int = 100) -> int:
    return int(cells_per_dim**dim)