import numpy as np
import pandas as pd
import pytest

from src.business_validation import (
    calculate_customer_clv,
    summarize_cluster_clv,
    calculate_retention_metrics,
    simulate_promotional_campaign,
    compare_business_separation,
)


@pytest.fixture
def sample_customer_features():
    return pd.DataFrame(
        {
            "R": [10.0, 30.0, 150.0, 300.0, 5.0, 45.0],
            "F": [1.0, 3.0, 5.0, 1.0, 8.0, 2.0],
            "M": [50.0, 300.0, 500.0, 100.0, 1200.0, 150.0],
            "PS": [0.1, 0.45, 0.0, 0.5, 0.15, 0.2],
        },
        index=["c1", "c2", "c3", "c4", "c5", "c6"],
    )


@pytest.fixture
def sample_labels():
    return pd.Series([0, 0, 1, 1, 0, 1], index=["c1", "c2", "c3", "c4", "c5", "c6"], name="cluster")


def test_calculate_customer_clv(sample_customer_features):
    clv = calculate_customer_clv(sample_customer_features, profit_margin=0.2, time_horizon_years=1.0)
    assert len(clv) == len(sample_customer_features)
    assert (clv >= 0).all()

    # Customer 5 has very low recency (5 days) and high spend (1200), so CLV should be highest
    assert clv.idxmax() == "c5"

    # Test parameter validation
    with pytest.raises(ValueError, match="Profit margin must be between 0 and 1"):
        calculate_customer_clv(sample_customer_features, profit_margin=1.5)

    with pytest.raises(ValueError, match="must include 'R', 'F', and 'M'"):
        calculate_customer_clv(sample_customer_features.drop(columns=["M"]))


def test_summarize_cluster_clv(sample_customer_features, sample_labels):
    summary = summarize_cluster_clv(sample_customer_features, sample_labels, "TestModel", 2)
    assert len(summary) == 2
    assert "mean_clv" in summary.columns
    assert "median_clv" in summary.columns
    assert "total_projected_clv" in summary.columns
    assert summary["customer_count"].sum() == 6


def test_calculate_retention_metrics(sample_customer_features, sample_labels):
    retention = calculate_retention_metrics(sample_customer_features, sample_labels, "TestModel", 2, recency_threshold_days=60)
    assert len(retention) == 2
    assert "repurchase_rate" in retention.columns
    assert "active_rate" in retention.columns
    assert (retention["repurchase_rate"] <= 1.0).all()
    assert (retention["repurchase_rate"] >= 0.0).all()


def test_simulate_promotional_campaign(sample_customer_features, sample_labels):
    # Cluster 1 has customers c3, c4, c6
    sim = simulate_promotional_campaign(
        sample_customer_features,
        sample_labels,
        target_cluster=1,
        discount_rate=0.10,
        expected_response_rate=0.33,
        marketing_cost_per_customer=2.0,
        baseline_margin=0.25,
        lift_in_spend=0.15,
    )
    assert sim["target_cluster"] == 1
    assert sim["target_audience_size"] == 3
    assert sim["expected_responders"] >= 1
    assert "total_campaign_cost" in sim
    assert "incremental_revenue" in sim
    assert "net_profit" in sim
    assert "roi_pct" in sim
    assert "risk_assessment" in sim


def test_compare_business_separation(sample_customer_features):
    labels_a = pd.Series([0, 0, 0, 1, 1, 1], index=sample_customer_features.index)
    labels_b = pd.Series([0, 1, 0, 1, 0, 1], index=sample_customer_features.index)
    comp = compare_business_separation(sample_customer_features, labels_a, labels_b)
    assert len(comp) == 4
    assert "f_stat_traditional" in comp.columns
    assert "f_stat_extended" in comp.columns
    assert "better_separation" in comp.columns
