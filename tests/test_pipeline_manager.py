from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.pipeline_manager import (
    SegmentationPipeline,
    save_pipeline,
    load_pipeline,
    list_saved_pipelines,
    predict_customer_segments,
)


@pytest.fixture
def dummy_pipeline():
    features = ("R", "F", "M")
    X = np.log1p(np.array([[10, 1, 100], [50, 2, 200], [100, 5, 500], [200, 10, 1000]], dtype=float))
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_scaled)

    profiles = pd.DataFrame(
        [
            {"cluster": 0, "rule_based_observation": "High Value, Active"},
            {"cluster": 1, "rule_based_observation": "Low Value, Lapsed"},
        ]
    )

    return SegmentationPipeline(
        model_name="Traditional RFM",
        features=features,
        k=2,
        scaler=scaler,
        kmeans=kmeans,
        cluster_profiles=profiles,
        created_at="2026-10-23T12:00:00",
        run_id="test_run_123",
    )


def test_save_load_predict_pipeline(dummy_pipeline, tmp_path):
    target = tmp_path / "model.joblib"
    saved_path = save_pipeline(dummy_pipeline, target)
    assert saved_path.exists()

    loaded = load_pipeline(target)
    assert loaded.model_name == dummy_pipeline.model_name
    assert loaded.k == dummy_pipeline.k
    assert loaded.features == dummy_pipeline.features

    # Test predict on new customer features
    new_customers = pd.DataFrame(
        {
            "R": [15.0, 180.0, -10.0, np.nan],
            "F": [2.0, 1.0, 2.0, 1.0],
            "M": [150.0, 80.0, 50.0, 100.0],
        },
        index=["new_1", "new_2", "invalid_1", "invalid_2"],
    )

    preds = predict_customer_segments(loaded, new_customers)
    assert len(preds) == 4
    assert "predicted_cluster" in preds.columns
    assert "segment_description" in preds.columns

    # Valid customers get predictions and segment descriptions
    assert pd.notna(preds.loc["new_1", "predicted_cluster"])
    assert preds.loc["new_1", "status"] == "assigned"
    assert preds.loc["new_1", "segment_description"] in ["High Value, Active", "Low Value, Lapsed"]

    # Invalid customers are flagged
    assert pd.isna(preds.loc["invalid_1", "predicted_cluster"])
    assert preds.loc["invalid_1", "status"] == "excluded: missing/negative feature values"


def test_list_saved_pipelines(dummy_pipeline, tmp_path):
    save_pipeline(dummy_pipeline, tmp_path / "pipe1.joblib")
    pipelines = list_saved_pipelines(tmp_path)
    assert len(pipelines) == 1
    assert pipelines[0]["model_name"] == "Traditional RFM"
    assert pipelines[0]["k"] == 2
