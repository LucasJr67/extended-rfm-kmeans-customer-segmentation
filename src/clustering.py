"""Reproducible K-Means experiments across configured feature sets."""

from dataclasses import dataclass
from itertools import combinations
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    DEFAULT_K_VALUES,
    DEFAULT_METRIC_SAMPLE_LIMIT,
    DEFAULT_N_INIT,
    DEFAULT_RANDOM_STATE,
)
from src.evaluation import evaluate_clustering, metric_sample_indices


@dataclass
class ExperimentResult:
    """Computed experiment tables and assignments, indexed by customer ID."""

    results: pd.DataFrame
    labels: dict[tuple[str, int], pd.Series]
    scaled_features: dict[str, pd.DataFrame]
    eligible_customer_features: pd.DataFrame
    exclusions: dict[str, int]
    metric_sample_size: int
    metric_sample_limit: int


def select_best_k(results: pd.DataFrame) -> pd.DataFrame:
    """Select maximum Silhouette per feature set; ties resolve to smaller K."""
    if results.empty or "silhouette" not in results:
        return pd.DataFrame(columns=results.columns)
    valid = results.loc[results["silhouette"].notna()].copy()
    if valid.empty:
        return valid
    valid = valid.sort_values(["model", "silhouette", "k"], ascending=[True, False, True])
    return valid.groupby("model", sort=False, as_index=False).head(1).reset_index(drop=True)


def run_experiments(
    customer_features: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
    stability_runs: int = 5,
    metric_sample_limit: int = DEFAULT_METRIC_SAMPLE_LIMIT,
) -> ExperimentResult:
    """Fit comparable K-Means experiments and evaluate reproducible metrics."""
    if not feature_sets:
        raise ValueError("Provide at least one named feature set.")
    if len(k_values) == 0 or any(
        not isinstance(k, (int, np.integer)) or k < 2 for k in k_values
    ):
        raise ValueError("K values must be a non-empty sequence of integers >= 2.")
    if n_init < 1 or stability_runs < 2 or random_state < 0:
        raise ValueError(
            "n_init must be positive, stability_runs must be at least 2, "
            "and random_state must be non-negative."
        )
    if metric_sample_limit < 2:
        raise ValueError("Metric sample limit must be at least 2.")
    normalized_sets: dict[str, tuple[str, ...]] = {}
    missing_columns = set()
    for name, selected in feature_sets.items():
        columns = tuple(selected)
        if not name.strip() or len(columns) < 2 or len(set(columns)) != len(columns):
            raise ValueError("Each feature set needs a name and at least two unique features.")
        missing_columns.update(set(columns) - set(customer_features.columns))
        normalized_sets[name] = columns
    if missing_columns:
        raise ValueError(f"Selected feature columns are unavailable: {sorted(missing_columns)}.")

    union_columns = list(dict.fromkeys(column for columns in normalized_sets.values() for column in columns))
    selected_values = customer_features[union_columns].apply(pd.to_numeric, errors="coerce")
    valid = selected_values.notna().all(axis=1)
    valid &= np.isfinite(selected_values).all(axis=1)
    valid &= selected_values.ge(0).all(axis=1)
    eligible = customer_features.loc[valid].copy()
    exclusions = {
        "input_customers": int(len(customer_features)),
        "missing_or_nonfinite_selected_feature": int(
            (selected_values.isna().any(axis=1) | ~np.isfinite(selected_values).all(axis=1)).sum()
        ),
        "negative_selected_feature": int(
            (selected_values.notna().all(axis=1) & selected_values.lt(0).any(axis=1)).sum()
        ),
        "non_positive_spend": int(customer_features["M"].le(0).sum())
        if "M" in customer_features
        else 0,
        "eligible_customers": int(valid.sum()),
    }
    if len(eligible) < 3:
        raise ValueError(
            "At least three customers must have complete, finite, non-negative values "
            "for every feature in every selected model."
        )
    if int(valid.sum()) > metric_sample_limit and metric_sample_limit < max(k_values):
        raise ValueError("Metric sample limit must be at least the largest requested K.")

    results_rows = []
    labels_by_model_k: dict[tuple[str, int], pd.Series] = {}
    scaled_by_model: dict[str, pd.DataFrame] = {}
    for model_name, columns in normalized_sets.items():
        transformed = np.log1p(eligible.loc[:, columns].astype(float))
        scaled_array = StandardScaler().fit_transform(transformed)
        scaled = pd.DataFrame(scaled_array, index=eligible.index, columns=columns)
        scaled_by_model[model_name] = scaled

        for k in sorted(set(int(value) for value in k_values)):
            if k >= len(scaled):
                results_rows.append(
                    {
                        "model": model_name,
                        "features": ", ".join(columns),
                        "k": k,
                        "n_customers": len(scaled),
                        "random_state": random_state,
                        "n_init": n_init,
                        "stability_runs": stability_runs,
                        "metric_sample_limit": metric_sample_limit,
                        "metric_sampling": "cluster-stratified, seeded" if len(scaled) > metric_sample_limit else "full eligible population",
                        "silhouette": None,
                        "davies_bouldin": None,
                        "calinski_harabasz": None,
                        "dunn": None,
                        "stability_ari_mean": None,
                        "stability_ari_std": None,
                        "metric_sample_size": min(len(scaled), metric_sample_limit),
                        "status": "skipped: K must be less than customer count",
                    }
                )
                continue
            run_labels = []
            for repeat in range(stability_runs):
                estimator = KMeans(
                    n_clusters=k,
                    random_state=random_state + repeat,
                    n_init=n_init,
                )
                run_labels.append(estimator.fit_predict(scaled))
            primary_labels = run_labels[0]
            labels_by_model_k[(model_name, k)] = pd.Series(
                primary_labels, index=scaled.index, name="cluster"
            )

            sample_indices = metric_sample_indices(
                primary_labels, metric_sample_limit, random_state
            )
            sampled_labels = primary_labels[sample_indices]
            if len(np.unique(sampled_labels)) < 2 or len(np.unique(sampled_labels)) >= len(sampled_labels):
                metrics = {
                    "silhouette": None,
                    "davies_bouldin": None,
                    "calinski_harabasz": None,
                    "dunn": None,
                }
                status = "undefined: evaluation sample has fewer than two valid clusters"
            else:
                metrics = evaluate_clustering(scaled_array[sample_indices], sampled_labels)
                status = "ok"
            ari_scores = [
                adjusted_rand_score(left, right)
                for left, right in combinations(run_labels, 2)
            ]
            results_rows.append(
                {
                    "model": model_name,
                    "features": ", ".join(columns),
                    "k": k,
                    "n_customers": len(scaled),
                    "random_state": random_state,
                    "n_init": n_init,
                    "stability_runs": stability_runs,
                    "metric_sample_limit": metric_sample_limit,
                    "metric_sampling": "cluster-stratified, seeded" if len(scaled) > metric_sample_limit else "full eligible population",
                    **metrics,
                    "stability_ari_mean": float(np.mean(ari_scores)),
                    "stability_ari_std": float(np.std(ari_scores, ddof=0)),
                    "metric_sample_size": len(sample_indices),
                    "status": status,
                }
            )

    return ExperimentResult(
        results=pd.DataFrame(results_rows),
        labels=labels_by_model_k,
        scaled_features=scaled_by_model,
        eligible_customer_features=eligible,
        exclusions=exclusions,
        metric_sample_size=min(len(eligible), metric_sample_limit),
        metric_sample_limit=metric_sample_limit,
    )
