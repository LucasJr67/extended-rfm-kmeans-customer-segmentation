"""Plotly charts for actual experiment results."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA


_BEST_K_COLORS: dict[int, str] = {
    0: "#2563EB",   # blue-600
    1: "#DC2626",   # red-600
    2: "#059669",   # emerald-600
    3: "#7C3AED",   # violet-600
}


def metric_comparison_figure(
    results: pd.DataFrame, metric: str, best_k: pd.DataFrame | None = None
) -> go.Figure:
    """Build a metric-by-K chart, optionally emphasizing selected K values.

    When *best_k* is supplied the chart highlights each selected K with:
    * a vertical dashed line spanning the full plot height,
    * a prominent annotation box showing the metric value.
    """
    if metric not in results.columns:
        raise ValueError(f"Unknown metric {metric!r}.")
    metric_label = metric.replace("_", " ").title()
    figure = px.line(
        results,
        x="k",
        y=metric,
        color="model",
        markers=True,
        title=f"{metric_label} by K",
        labels={"k": "Number of clusters (K)", metric: metric_label},
    )

    subtitle_parts: list[str] = []

    if best_k is not None and not best_k.empty:
        selected = best_k.loc[:, ["model", "k"]].merge(
            results.loc[:, ["model", "k", metric]], on=["model", "k"], how="left"
        ).dropna(subset=[metric])
        if not selected.empty:
            for idx, (_, row) in enumerate(selected.iterrows()):
                color = _BEST_K_COLORS.get(idx, "#6B7280")
                k_val = int(row["k"])
                metric_val = float(row[metric])
                model_name = row["model"]

                # Vertical dashed line at best K
                figure.add_vline(
                    x=k_val,
                    line_dash="dash",
                    line_color=color,
                    line_width=2,
                    opacity=0.6,
                )

                # Annotation box with metric value
                figure.add_annotation(
                    x=k_val,
                    y=metric_val,
                    text=f"<b>Best K={k_val}</b><br>{model_name}<br>{metric_label}: {metric_val:.4f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.2,
                    arrowwidth=2,
                    arrowcolor=color,
                    ax=40 if idx % 2 == 0 else -40,
                    ay=-50 if idx % 2 == 0 else -60,
                    bordercolor=color,
                    borderwidth=2,
                    borderpad=6,
                    bgcolor="white",
                    opacity=0.95,
                    font={"size": 12, "color": color},
                )

                subtitle_parts.append(f"{model_name}: Best K = {k_val}")

    subtitle_text = ""
    if subtitle_parts:
        subtitle_text = f"<br><span style='font-size:12px;color:#6B7280'>{' · '.join(subtitle_parts)}</span>"

    figure.update_layout(
        title={"text": f"<b>{metric_label} by K</b>{subtitle_text}"},
        legend_title_text="Model",
        hoverlabel={"font": {"size": 14}},
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
