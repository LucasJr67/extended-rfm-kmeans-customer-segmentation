import numpy as np
import pandas as pd
import pytest

from src.config import ColumnMapping, FEATURE_SETS
from src.features import available_features, build_customer_features
from src.preprocessing import clean_transactions


def test_rfm_and_behavioral_formulas(transactions, full_mapping):
    cleaned, _ = clean_transactions(transactions, full_mapping)
    features, report = build_customer_features(cleaned, full_mapping, "2025-03-01")
    customer = features.loc["C1"]
    source = cleaned.loc[cleaned["customer"] == "C1"]
    assert customer["F"] == source["invoice"].nunique()
    assert customer["M"] == source["amount"].sum()
    assert customer["D"] == source["category"].nunique()
    assert customer["PS"] == pytest.approx(source["promo"].sum() / source["amount"].sum())
    days = sorted(pd.to_datetime(source["date"]).dt.normalize().unique())
    gaps = np.diff(pd.to_datetime(days).values).astype("timedelta64[D]").astype(float)
    assert customer["PR"] == pytest.approx(np.std(gaps, ddof=1))
    assert customer["R"] == (pd.Timestamp("2025-03-01") - pd.to_datetime(source["date"]).max().normalize()).days
    assert report["customer_count"] == 12


def test_frequency_counts_unique_orders_not_rows(full_mapping):
    data = pd.DataFrame(
        {
            "customer": ["a", "a", "a"],
            "invoice": ["one", "one", "two"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-03"]),
            "amount": [5, 7, 2],
            "category": ["x", "y", "y"],
            "promo": [0, 0, 0],
        }
    )
    features, _ = build_customer_features(data, full_mapping, "2025-01-10")
    assert features.loc["a", "F"] == 2
    assert features.loc["a", "M"] == 14
    assert features.loc["a", "D"] == 2
    assert np.isnan(features.loc["a", "PR"])


def test_optional_features_are_not_fabricated(transactions):
    mapping = ColumnMapping("customer", "invoice", "date", "amount")
    features, _ = build_customer_features(transactions, mapping, "2025-03-01")
    assert available_features(mapping) == ("R", "F", "M", "PR")
    assert "D" not in features.columns
    assert "PS" not in features.columns
    assert FEATURE_SETS["extended_rfm"] == ("R", "F", "M", "D", "PS", "PR")


def test_invalid_analysis_date_and_promo_ratio(full_mapping, transactions):
    with pytest.raises(ValueError, match="precede"):
        build_customer_features(transactions, full_mapping, "2020-01-01")
    data = transactions.copy()
    data.loc[data["customer"] == "C1", "amount"] = 0
    data.loc[data["customer"] == "C1", "promo"] = 1
    features, _ = build_customer_features(data, full_mapping, "2025-03-01")
    assert np.isnan(features.loc["C1", "PS"])


def test_incomplete_promotional_values_do_not_produce_partial_share(full_mapping, transactions):
    data = transactions.copy()
    data.loc[data.index[0], "promo"] = np.nan
    features, _ = build_customer_features(data, full_mapping, "2025-03-01")
    assert np.isnan(features.loc[data.loc[data.index[0], "customer"], "PS"])


def test_missing_category_is_unknown_not_zero_diversity(full_mapping, transactions):
    data = transactions.copy()
    data.loc[data["customer"] == "C1", "category"] = None
    features, _ = build_customer_features(data, full_mapping, "2025-03-01")
    assert np.isnan(features.loc["C1", "D"])


def test_bad_customer_id_and_non_numeric_source_are_cleaned(transactions, full_mapping):
    bad = transactions.copy()
    bad["amount"] = bad["amount"].astype(object)
    bad.loc[0, "customer"] = None
    bad.loc[1, "amount"] = "not numeric"
    cleaned, report = clean_transactions(bad, full_mapping)
    assert report["missing_customer_id_rows_removed"] == 1
    assert report["invalid_spend_rows_removed"] == 1
    assert cleaned["customer"].notna().all()
