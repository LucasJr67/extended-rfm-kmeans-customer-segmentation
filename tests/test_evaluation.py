import numpy as np
import pytest

from src.evaluation import dunn_index, evaluate_clustering, metric_sample_indices


def test_metrics_on_separated_clusters(clustered_points):
    labels = np.array([0] * 12 + [1] * 12)
    metrics = evaluate_clustering(clustered_points, labels)
    assert metrics["silhouette"] > 0.9
    assert metrics["davies_bouldin"] < 0.2
    assert metrics["calinski_harabasz"] > 1000
    assert metrics["dunn"] > 1


def test_dunn_index_known_geometry():
    points = np.array([[0.0], [1.0], [5.0], [6.0]])
    labels = np.array([0, 0, 1, 1])
    assert dunn_index(points, labels) == pytest.approx(4 / 1)


def test_undefined_metrics_for_single_cluster_or_singletons():
    assert all(value is None for value in evaluate_clustering(np.array([[0], [1]]), [0, 0]).values())
    assert all(value is None for value in evaluate_clustering(np.array([[0], [1]]), [0, 1]).values())


def test_metric_sample_is_reproducible_and_retains_clusters():
    labels = np.array([0] * 80 + [1] * 15 + [2] * 5)
    first = metric_sample_indices(labels, 20, 99)
    second = metric_sample_indices(labels, 20, 99)
    assert np.array_equal(first, second)
    assert set(labels[first]) == {0, 1, 2}
    assert len(first) == 20


def test_metric_sample_must_fit_all_clusters():
    with pytest.raises(ValueError, match="number of clusters"):
        metric_sample_indices([0, 1, 2], 2, 1)
