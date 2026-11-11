"""Random Swap: replace one centroid, then a short local k-means."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kmeans import kmeans


@dataclass
class RandomSwapResult:
    centroids: np.ndarray
    partition: np.ndarray
    sse: float
    trials: int
    improvements: int
    history: list[dict]


def random_swap_trial(
    data: np.ndarray,
    centroids: np.ndarray,
    *,
    local_iters: int = 2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One trial: drop a random centroid, put it on a random point, fine-tune."""
    data = np.asarray(data, dtype=np.float64)
    C = np.asarray(centroids, dtype=np.float64).copy()
    rng = np.random.default_rng(seed)
    k = len(C)
    remove_i = int(rng.integers(0, k))
    insert_i = int(rng.integers(0, len(data)))
    C[remove_i] = data[insert_i]
    local = kmeans(data, k, centroids=C, max_iter=local_iters, seed=seed)
    return local.centroids, local.partition, local.sse


def random_swap(
    data: np.ndarray,
    k: int,
    *,
    trials: int = 500,
    local_iters: int = 2,
    init_iters: int = 5,
    seed: int | None = None,
) -> RandomSwapResult:
    data = np.asarray(data, dtype=np.float64)
    rng = np.random.default_rng(seed)
    base = kmeans(data, k, max_iter=init_iters, seed=seed)
    best_C = base.centroids
    best_P = base.partition
    best_sse = base.sse
    improvements = 0
    history = [{"trial": 0, "sse": best_sse, "accepted": True}]

    for t in range(1, trials + 1):
        trial_seed = int(rng.integers(0, 2**31 - 1))
        C, P, s = random_swap_trial(
            data, best_C, local_iters=local_iters, seed=trial_seed
        )
        accepted = s < best_sse
        if accepted:
            best_C, best_P, best_sse = C, P, s
            improvements += 1
        history.append({"trial": t, "sse": s, "accepted": accepted, "best_sse": best_sse})

    return RandomSwapResult(
        centroids=best_C,
        partition=best_P,
        sse=best_sse,
        trials=trials,
        improvements=improvements,
        history=history,
    )


def compare_km_vs_rs_budget(
    data: np.ndarray,
    k: int,
    *,
    km_iters: int = 25,
    alpha: float = 4.0,
    local_iters: int = 2,
) -> dict:
    """How many RS trials fit in ~one full k-means (course cost model).

    T_km ≈ 25 * O(Nk), one swap ≈ αN + 2 * O(Nk) → n ≈ 25k / (α + 2k).
    """
    t_km = km_iters * k
    t_one_swap = alpha + local_iters * k
    n_swaps = t_km / t_one_swap
    return {
        "k": k,
        "km_iters": km_iters,
        "alpha": alpha,
        "local_iters": local_iters,
        "t_km_units": t_km,
        "t_swap_units": t_one_swap,
        "n_swaps": n_swaps,
    }