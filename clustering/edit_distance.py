"""Edit (Levenshtein) distance and medoids for string data."""

from __future__ import annotations

import numpy as np


def edit_distance(a: str, b: str) -> int:
    """Classic Levenshtein distance (insert / delete / substitute cost 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def pairwise_edit_distances(items: list[str]) -> np.ndarray:
    """Symmetric matrix of pairwise edit distances."""
    n = len(items)
    D = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            d = edit_distance(items[i], items[j])
            D[i, j] = D[j, i] = d
    return D


def medoid(items: list[str], distances: np.ndarray | None = None) -> tuple[int, str, float]:
    """Medoid = item with minimum total distance to other items in the cluster.

    Returns (index, item, total_cost).
    """
    if distances is None:
        distances = pairwise_edit_distances(items)
    totals = distances.sum(axis=1)
    idx = int(np.argmin(totals))
    return idx, items[idx], float(totals[idx])


def cluster_medoid_cost(items: list[str]) -> dict:
    """Medoid and total cost for a string cluster."""
    D = pairwise_edit_distances(items)
    idx, item, cost = medoid(items, D)
    return {
        "items": items,
        "distances": D,
        "medoid_index": idx,
        "medoid": item,
        "cost": cost,
    }