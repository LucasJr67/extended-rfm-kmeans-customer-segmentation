"""Plotly charts for actual experiment results."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def metric_comparison_figure(results: pd.DataFrame, metric: str) -> go.Figure:
    """Build a metric-by-K comparison chart."""
    if metric not in results.columns:
        raise ValueError(f"Unknown metric {metric!r}.")
    figure = px.line(
        results,
        x="k",
        y=metric,
        color="model",
        markers=True,
        title=f"{metric.replace('_', ' ').title()} by K",
        labels={"k": "Number of clusters (K)", metric: metric.replace("_", " ").title()},
    )
    return figure


def cluster_scatter_figure(
    scaled: pd.DataFrame, labels: pd.Series, title: str
) -> go.Figure:
    """Project scaled features to two PCA components for a labeled visualization."""
    if len(scaled) < 2 or scaled.shape[1] < 2:
        raise ValueError("At least two customers and two selected features are needed for PCA.")
    projection = PCA(n_components=2, random_state=42).fit_transform(scaled)
    plot_data = pd.DataFrame(
        {
            "PC1": projection[:, 0],
            "PC2": projection[:, 1],
            "Cluster": labels.astype(str).to_numpy(),
            "Customer": scaled.index.astype(str),
        }
    )
    return px.scatter(
        plot_data,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=["Customer"],
        title=title,
    )


def profile_clusters(
    customer_features: pd.DataFrame, labels: pd.Series, model_name: str, k: int
) -> pd.DataFrame:
    """Summarize raw feature medians by cluster and attach cautious rule-based notes."""
    aligned = customer_features.loc[labels.index].copy()
    aligned["cluster"] = labels
    features = list(customer_features.columns)
    medians = aligned.groupby("cluster")[features].median()
    overall = customer_features[features].median()
    rows = []
    for cluster, values in medians.iterrows():
        elevated = [feature for feature in features if values[feature] > overall[feature]]
        reduced = [feature for feature in features if values[feature] < overall[feature]]
        notes = []
        if "R" in elevated:
            notes.append("Higher recency value (longer time since last purchase)")
        if "R" in reduced:
            notes.append("Lower recency value (more recent purchasing)")
        if "F" in elevated:
            notes.append("More distinct orders than the dataset median")
        if "M" in elevated:
            notes.append("Higher mapped total spend than the dataset median")
        if "D" in elevated:
            notes.append("Broader category diversity than the dataset median")
        if "PS" in elevated:
            notes.append("Higher mapped promotional-spend share than the dataset median")
        if "PR" in elevated:
            notes.append("More variable purchase intervals than the dataset median")
        if not notes:
            notes.append("No above-median pattern on the listed features")
        rows.append(
            {
                "model": model_name,
                "k": k,
                "cluster": int(cluster),
                "customer_count": int((labels == cluster).sum()),
                **{f"median_{feature}": float(values[feature]) for feature in features},
                "rule_based_observation": "; ".join(notes)
                + ". Descriptive only; validate before any business action.",
            }
        )
    return pd.DataFrame(rows)
