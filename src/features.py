"""Customer-level feature engineering with transparent exclusions."""

from datetime import date, datetime

import numpy as np
import pandas as pd

from src.config import FEATURE_DEFINITIONS, ColumnMapping
from src.preprocessing import validate_mapping


def available_features(mapping: ColumnMapping) -> tuple[str, ...]:
    """Return features whose source fields are explicitly mapped."""
    features = ["R", "F", "M"]
    if mapping.category is not None:
        features.append("D")
    if mapping.promotional_spend is not None:
        features.append("PS")
    features.append("PR")
    return tuple(features)


def build_customer_features(
    transactions: pd.DataFrame,
    mapping: ColumnMapping,
    analysis_date: date | datetime | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate transactions into customer RFM and available behavioral features."""
    validate_mapping(transactions, mapping)
    if transactions.empty:
        raise ValueError("Cannot engineer customer features from an empty transaction table.")

    purchase_dates = pd.to_datetime(transactions[mapping.purchase_date], errors="coerce")
    if purchase_dates.isna().any():
        raise ValueError("Clean transaction dates before feature engineering.")
    dates_only = purchase_dates.dt.normalize()
    if analysis_date is None:
        chosen_date = dates_only.max() + pd.Timedelta(days=1)
    else:
        chosen_date = pd.Timestamp(analysis_date)
        if chosen_date.tz is not None:
            chosen_date = chosen_date.tz_convert(None)
        chosen_date = chosen_date.normalize()
    if chosen_date < dates_only.max():
        raise ValueError("Analysis date cannot precede the latest purchase date.")

    frame = transactions.copy()
    frame["_purchase_date"] = dates_only
    grouped = frame.groupby(mapping.customer_id, sort=True, dropna=True)
    features = grouped.agg(
        last_purchase_date=("_purchase_date", "max"),
        F=(mapping.order_id, "nunique"),
        M=(mapping.spend, "sum"),
    )
    features["R"] = (chosen_date - features["last_purchase_date"]).dt.days.astype(float)
    features["F"] = features["F"].astype(float)
    features = features[["R", "F", "M"]]

    if mapping.category is not None:
        features["D"] = grouped[mapping.category].apply(
            lambda values: (
                float(values.nunique()) if values.notna().any() else np.nan
            )
        )

    if mapping.promotional_spend is not None:
        promo_total = grouped[mapping.promotional_spend].sum(min_count=1)
        promo_incomplete = grouped[mapping.promotional_spend].apply(
            lambda values: values.isna().any()
        )
        features["PS"] = promo_total / features["M"]
        invalid_ratio = (
            promo_total.isna()
            | promo_incomplete
            | features["M"].le(0)
            | features["PS"].lt(0)
            | features["PS"].gt(1)
        )
        features.loc[invalid_ratio, "PS"] = np.nan

    interval_variability = {}
    for customer_id, group in grouped:
        unique_dates = group["_purchase_date"].drop_duplicates().sort_values()
        gaps = unique_dates.diff().dropna().dt.days.to_numpy(dtype=float)
        interval_variability[customer_id] = (
            float(np.std(gaps, ddof=1)) if len(gaps) >= 2 else np.nan
        )
    features["PR"] = pd.Series(interval_variability, dtype=float)
    features.index.name = mapping.customer_id

    report: dict[str, object] = {
        "analysis_date": chosen_date.date().isoformat(),
        "customer_count": int(len(features)),
        "available_features": list(available_features(mapping)),
        "feature_definitions": {
            name: FEATURE_DEFINITIONS[name] for name in available_features(mapping)
        },
        "missing_feature_values": features.isna().sum().astype(int).to_dict(),
        "non_positive_spend_customers": int(features["M"].le(0).sum()),
        "insufficient_purchase_history_for_PR": int(features["PR"].isna().sum()),
    }
    return features, report
