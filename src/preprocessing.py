"""Input loading, exploration, and auditable transaction cleaning."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ColumnMapping


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load CSV or XLSX data without silently substituting parse failures."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".xlsx":
        return pd.read_excel(source)
    raise ValueError(f"Unsupported dataset format {suffix!r}; use CSV or XLSX.")


def explore_dataset(data: pd.DataFrame) -> dict[str, Any]:
    """Summarize the uploaded data without changing it."""
    if data.empty and len(data.columns) == 0:
        raise ValueError("The dataset has no columns.")
    return {
        "shape": data.shape,
        "columns": list(data.columns),
        "dtypes": data.dtypes.astype(str).to_dict(),
        "missing_values": data.isna().sum().to_dict(),
        "duplicate_rows": int(data.duplicated().sum()),
        "describe": data.describe(include="all").transpose(),
    }


def validate_mapping(data: pd.DataFrame, mapping: ColumnMapping) -> None:
    """Ensure required and selected optional source columns exist."""
    required = {
        "customer_id": mapping.customer_id,
        "order_id": mapping.order_id,
        "purchase_date": mapping.purchase_date,
        "spend": mapping.spend,
    }
    selected = {key: value for key, value in required.items()}
    if mapping.category is not None:
        selected["category"] = mapping.category
    if mapping.promotional_spend is not None:
        selected["promotional_spend"] = mapping.promotional_spend

    blank = [key for key, value in selected.items() if not isinstance(value, str) or not value.strip()]
    if blank:
        raise ValueError(f"Column mappings cannot be blank: {', '.join(blank)}.")
    duplicates = sorted({value for value in selected.values() if list(selected.values()).count(value) > 1})
    if duplicates:
        raise ValueError(f"Each mapped field must have a distinct source column: {duplicates}.")
    missing = [value for value in selected.values() if value not in data.columns]
    if missing:
        raise ValueError(f"Mapped source columns are missing: {missing}.")


def clean_transactions(
    data: pd.DataFrame, mapping: ColumnMapping
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean required fields in a documented order and return sequential counts."""
    if data.empty:
        raise ValueError("The uploaded dataset contains no transaction rows.")
    validate_mapping(data, mapping)

    cleaned = data.copy()
    report = {"input_rows": int(len(cleaned))}
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    report["exact_duplicate_rows_removed"] = int(before - len(cleaned))

    for label, column in (
        ("missing_customer_id_rows_removed", mapping.customer_id),
        ("missing_order_id_rows_removed", mapping.order_id),
    ):
        before = len(cleaned)
        has_value = cleaned[column].notna()
        has_text = cleaned[column].astype("string").str.strip().ne("").fillna(False)
        cleaned = cleaned.loc[has_value & has_text].copy()
        report[label] = int(before - len(cleaned))

    before = len(cleaned)
    cleaned[mapping.purchase_date] = pd.to_datetime(
        cleaned[mapping.purchase_date], errors="coerce", utc=True
    ).dt.tz_convert(None)
    cleaned = cleaned.loc[cleaned[mapping.purchase_date].notna()].copy()
    report["invalid_purchase_date_rows_removed"] = int(before - len(cleaned))

    before = len(cleaned)
    cleaned[mapping.spend] = pd.to_numeric(cleaned[mapping.spend], errors="coerce")
    valid_spend = cleaned[mapping.spend].notna() & np.isfinite(cleaned[mapping.spend])
    cleaned = cleaned.loc[valid_spend].copy()
    report["invalid_spend_rows_removed"] = int(before - len(cleaned))

    if mapping.category is not None:
        cleaned[mapping.category] = cleaned[mapping.category].replace(
            r"^\s*$", pd.NA, regex=True
        )

    if mapping.promotional_spend is not None:
        cleaned[mapping.promotional_spend] = pd.to_numeric(
            cleaned[mapping.promotional_spend], errors="coerce"
        )
        cleaned.loc[
            ~(
                cleaned[mapping.promotional_spend].ge(0)
                & np.isfinite(cleaned[mapping.promotional_spend])
            ),
            mapping.promotional_spend,
        ] = float("nan")
        report["invalid_promotional_spend_rows_retained_as_missing"] = int(
            cleaned[mapping.promotional_spend].isna().sum()
        )

    report["retained_rows"] = int(len(cleaned))
    if cleaned.empty:
        raise ValueError("No rows remain after cleaning the required customer, order, date, and spend fields.")
    return cleaned, report
