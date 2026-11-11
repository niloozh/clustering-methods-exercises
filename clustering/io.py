"""Read/write SIPU-style datasets and Clusterator files."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


def load_dataset(path: PathLike) -> np.ndarray:
    """Load whitespace-separated numbers → (N, D) array. Skips blank/# lines."""
    path = Path(path)
    rows: list[list[float]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            rows.append([float(x) for x in parts])
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    data = np.asarray(rows, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def save_centroids(centroids: np.ndarray, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in np.asarray(centroids, dtype=np.float64):
            f.write(" ".join(f"{v:.10g}" for v in np.atleast_1d(c)) + "\n")


def save_partition(partition: np.ndarray, path: PathLike) -> None:
    """One 1-based cluster id per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(partition, dtype=np.int64)
    with path.open("w") as f:
        for lab in labels:
            f.write(f"{int(lab)}\n")


def load_partition(path: PathLike) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int64)