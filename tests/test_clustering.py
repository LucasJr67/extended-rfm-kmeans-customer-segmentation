import numpy as np
import pandas as pd
import pytest

from src.clustering import run_experiments, select_best_k
from src.features import build_customer_features
from src.preprocessing import clean_transactions


def test_pipeline_runs_end_to_end_and_is_reproducible(transactions, full_mapping):
    cleaned, _ = clean_transactions(transactions, full_mapping)
    features, _ = build_customer_features(cleaned, full_mapping, "2025-03-01")
    models = {
        "Traditional RFM": ("R", "F", "M"),
        "Extended RFM": ("R", "F", "M", "D", "PS", "PR"),
    }
    first = run_experiments(
        features, models, k_values=[2, 3], random_state=13, n_init=5, stability_runs=3
    )
    second = run_experiments(
        features, models, k_values=[2, 3], random_state=13, n_init=5, stability_runs=3
    )
    pd.testing.assert_frame_equal(first.results, second.results)
    assert set(first.results["model"]) == set(models)
    assert first.exclusions["eligible_customers"] == len(features)
    assert len(first.labels) == 4
    assert set(select_best_k(first.results)["model"]) == set(models)


def test_missing_selected_feature_uses_shared_population(transactions, full_mapping):
    cleaned, _ = clean_transactions(transactions, full_mapping)
    features, _ = build_customer_features(cleaned, full_mapping, "2025-03-01")
    features.loc[features.index[0], "PS"] = np.nan
    result = run_experiments(
        features,
        {"RFM": ("R", "F", "M"), "Extended": ("R", "F", "M", "PS")},
        k_values=[2],
        stability_runs=2,
    )
    assert result.exclusions["eligible_customers"] == len(features) - 1
    assert all(len(labels) == len(features) - 1 for labels in result.labels.values())


def test_invalid_features_and_too_small_dataset_raise():
    with pytest.raises(ValueError, match="unavailable"):
        run_experiments(pd.DataFrame({"R": [1, 2, 3]}), {"bad": ("R", "F")})
    with pytest.raises(ValueError, match="At least three"):
        run_experiments(
            pd.DataFrame({"R": [1, 2], "F": [1, 1]}),
            {"RFM": ("R", "F")},
            k_values=[2],
            stability_runs=2,
        )


def test_k_equal_or_above_population_is_reported_as_skipped():
    frame = pd.DataFrame({"R": range(3), "F": [1, 1, 1]})
    result = run_experiments(
        frame, {"small": ("R", "F")}, k_values=[2, 3], stability_runs=2
    )
    skipped = result.results.loc[result.results["k"] == 3].iloc[0]
    assert skipped["status"].startswith("skipped")
    assert select_best_k(result.results)["k"].tolist() == [2]
