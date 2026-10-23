"""Model persistence, pipeline serialization, and batch customer segment inference."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class SegmentationPipeline:
    """Packaged reproducible segmentation pipeline for inference."""

    model_name: str
    features: tuple[str, ...]
    k: int
    scaler: StandardScaler
    kmeans: KMeans
    cluster_profiles: pd.DataFrame
    created_at: str
    run_id: str
    feature_meta: dict[str, Any] | None = None


def save_pipeline(pipeline: SegmentationPipeline, destination: str | Path) -> Path:
    """Persist a segmentation pipeline using joblib."""
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, dest)
    return dest


def load_pipeline(source: str | Path) -> SegmentationPipeline:
    """Load a persisted segmentation pipeline."""
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Pipeline file not found: {src}")
    obj = joblib.load(src)
    if not isinstance(obj, SegmentationPipeline):
        raise TypeError(f"Loaded object is not a SegmentationPipeline, got {type(obj)}.")
    return obj


def list_saved_pipelines(models_dir: str | Path) -> list[dict[str, Any]]:
    """Scan models directory and return metadata of available pipelines."""
    path = Path(models_dir)
    if not path.exists():
        return []
    records = []
    for f in sorted(path.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            pipe = joblib.load(f)
            if isinstance(pipe, SegmentationPipeline):
                records.append(
                    {
                        "filename": f.name,
                        "path": str(f),
                        "model_name": pipe.model_name,
                        "k": pipe.k,
                        "features": ", ".join(pipe.features),
                        "created_at": pipe.created_at,
                        "run_id": pipe.run_id,
                        "size_bytes": f.stat().st_size,
                    }
                )
        except Exception:
            continue
    return records


def predict_customer_segments(
    pipeline: SegmentationPipeline,
    customer_features: pd.DataFrame,
) -> pd.DataFrame:
    """Assign cluster labels to new customer feature observations.

    Validates feature presence, applies log1p, standardizes with the saved scaler,
    and assigns clusters with the saved KMeans model.
    """
    required_cols = list(pipeline.features)
    missing = [c for c in required_cols if c not in customer_features.columns]
    if missing:
        raise ValueError(f"Input data is missing required model features: {missing}")

    subset = customer_features[required_cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = subset.notna().all(axis=1) & np.isfinite(subset).all(axis=1) & subset.ge(0).all(axis=1)

    if not valid_mask.any():
        raise ValueError("No customers have complete, finite, non-negative values for model features.")

    eligible = subset.loc[valid_mask].copy()

    # Preprocessing: log1p -> scaler.transform
    transformed = np.log1p(eligible.astype(float))
    scaled_array = pipeline.scaler.transform(transformed)

    predicted_labels = pipeline.kmeans.predict(scaled_array)

    results = customer_features.copy()
    results["predicted_cluster"] = np.nan
    results.loc[eligible.index, "predicted_cluster"] = predicted_labels
    results["status"] = "excluded: missing/negative feature values"
    results.loc[eligible.index, "status"] = "assigned"

    # Attach descriptive profile text if available
    if not pipeline.cluster_profiles.empty and "rule_based_observation" in pipeline.cluster_profiles.columns:
        profile_map = pipeline.cluster_profiles.set_index("cluster")["rule_based_observation"].to_dict()
        results["segment_description"] = results["predicted_cluster"].map(profile_map).fillna("N/A")
    else:
        results["segment_description"] = results["predicted_cluster"].apply(
            lambda c: f"Cluster {int(c)}" if pd.notna(c) else "N/A"
        )

    return results


def create_pipeline_from_experiment(
    eligible_customer_features: pd.DataFrame,
    features: tuple[str, ...],
    model_name: str,
    k: int,
    random_state: int,
    n_init: int,
    cluster_profiles: pd.DataFrame,
    run_id: str,
) -> SegmentationPipeline:
    """Build and fit a standalone SegmentationPipeline from experiment artifacts."""
    feature_cols = list(features)
    subset = eligible_customer_features[feature_cols].astype(float)
    transformed = np.log1p(subset)
    scaler = StandardScaler().fit(transformed)
    scaled_array = scaler.transform(transformed)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(scaled_array)
    return SegmentationPipeline(
        model_name=model_name,
        features=tuple(feature_cols),
        k=k,
        scaler=scaler,
        kmeans=kmeans,
        cluster_profiles=cluster_profiles,
        created_at=datetime.now().isoformat(),
        run_id=run_id,
    )

