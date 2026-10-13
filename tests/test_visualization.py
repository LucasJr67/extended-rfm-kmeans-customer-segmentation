import pandas as pd

from src.visualization import (
    cluster_scatter_figure,
    metric_comparison_figure,
    profile_clusters,
)


def test_metric_and_cluster_figures_render_structures():
    results = pd.DataFrame(
        {
            "model": ["RFM", "RFM"],
            "k": [2, 3],
            "silhouette": [0.3, 0.4],
        }
    )
    metric_figure = metric_comparison_figure(results, "silhouette")
    assert len(metric_figure.data) == 1

    customers = pd.DataFrame(
        {"R": [1, 2, 12, 14], "F": [5, 7, 1, 2], "M": [50, 60, 10, 20]},
        index=["a", "b", "c", "d"],
    )
    labels = pd.Series([0, 0, 1, 1], index=customers.index)
    scatter = cluster_scatter_figure(customers, labels, "test chart")
    assert len(scatter.data) == 2
    profiles = profile_clusters(customers, labels, "RFM", 2)
    assert profiles["customer_count"].tolist() == [2, 2]
    assert profiles["rule_based_observation"].str.contains("Descriptive only").all()
