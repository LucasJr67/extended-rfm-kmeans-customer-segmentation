"""Clustering metrics and reproducible sampling utilities."""

from collections.abc import Sequence

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def metric_sample_indices(
    labels: Sequence[int], max_samples: int, random_state: int
) -> np.ndarray:
    """Select a reproducible, approximately proportional sample retaining each cluster."""
    y = np.asarray(labels)
    n_rows = len(y)
    if max_samples < 2:
        raise ValueError("Metric sample limit must be at least 2.")
    if n_rows <= max_samples:
        return np.arange(n_rows)
    groups, counts = np.unique(y, return_counts=True)
    if max_samples < len(groups):
        raise ValueError("Metric sample limit must be at least the number of clusters.")

    rng = np.random.default_rng(random_state)
    allocations = np.ones(len(groups), dtype=int)
    remaining = max_samples - len(groups)
    if remaining:
        proportions = counts / counts.sum()
        extras = np.floor(proportions * remaining).astype(int)
        allocations += np.minimum(extras, counts - allocations)
    while allocations.sum() < max_samples:
        eligible = np.flatnonzero(allocations < counts)
        if len(eligible) == 0:
            break
        next_group = eligible[np.argmax(counts[eligible] / allocations[eligible])]
        allocations[next_group] += 1

    chosen = []
    for group, count in zip(groups, allocations, strict=True):
        members = np.flatnonzero(y == group)
        chosen.extend(rng.choice(members, size=count, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=int)


def dunn_index(features: np.ndarray, labels: Sequence[int]) -> float | None:
    """Compute Euclidean Dunn index as minimum intercluster distance / max diameter."""
    matrix = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    if matrix.ndim != 2 or len(matrix) != len(y) or len(matrix) < 2:
        raise ValueError("Dunn index requires a 2D feature matrix matching at least two labels.")
    groups = np.unique(y)
    if len(groups) < 2:
        return None

    maximum_diameter = 0.0
    minimum_separation = np.inf
    for group in groups:
        members = matrix[y == group]
        if len(members) > 1:
            distances = cdist(members, members, metric="euclidean")
            maximum_diameter = max(maximum_diameter, float(distances.max()))
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            separation = cdist(matrix[y == left], matrix[y == right], metric="euclidean").min()
            minimum_separation = min(minimum_separation, float(separation))
    if maximum_diameter <= 0 or minimum_separation <= 0:
        return None
    return float(minimum_separation / maximum_diameter)


def evaluate_clustering(features: np.ndarray, labels: Sequence[int]) -> dict[str, float | None]:
    """Calculate four complementary internal metrics on the supplied observations."""
    matrix = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    if matrix.ndim != 2 or len(matrix) != len(y):
        raise ValueError("Features must be a 2D matrix with one label per observation.")
    cluster_count = len(np.unique(y))
    if cluster_count < 2 or cluster_count >= len(y):
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "dunn": None,
        }
    return {
        "silhouette": float(silhouette_score(matrix, y)),
        "davies_bouldin": float(davies_bouldin_score(matrix, y)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, y)),
        "dunn": dunn_index(matrix, y),
    }
