import pandas as pd
import pytest

from src.config import ColumnMapping
from src.preprocessing import clean_transactions, explore_dataset, load_dataset, validate_mapping


def test_cleaning_reports_sequential_invalid_rows(transactions, full_mapping):
    raw = pd.concat(
        [
            transactions,
            transactions.iloc[[0]],
            pd.DataFrame(
                [
                    {"customer": None, "invoice": "bad-1", "date": "2025-01-01", "amount": 2},
                    {"customer": "bad", "invoice": None, "date": "2025-01-01", "amount": 2},
                    {"customer": "bad", "invoice": "bad-2", "date": "not-a-date", "amount": 2},
                    {"customer": "bad", "invoice": "bad-3", "date": "2025-01-01", "amount": "invalid"},
                ]
            ),
        ],
        ignore_index=True,
    )
    cleaned, report = clean_transactions(raw, full_mapping)
    assert report["exact_duplicate_rows_removed"] == 1
    assert report["missing_customer_id_rows_removed"] == 1
    assert report["missing_order_id_rows_removed"] == 1
    assert report["invalid_purchase_date_rows_removed"] == 1
    assert report["invalid_spend_rows_removed"] == 1
    assert len(cleaned) == len(transactions)


def test_mapping_missing_column_is_explicit(transactions, full_mapping):
    with pytest.raises(ValueError, match="missing"):
        validate_mapping(transactions.drop(columns="invoice"), full_mapping)


def test_loader_csv_and_unsupported_extension(tmp_path):
    path = tmp_path / "transactions.csv"
    pd.DataFrame({"x": [1]}).to_csv(path, index=False)
    assert load_dataset(path).loc[0, "x"] == 1
    excel_path = tmp_path / "transactions.xlsx"
    pd.DataFrame({"x": [2]}).to_excel(excel_path, index=False)
    assert load_dataset(excel_path).loc[0, "x"] == 2
    with pytest.raises(ValueError, match="Unsupported"):
        load_dataset(tmp_path / "transactions.xls")


def test_exploration_describes_missing_and_duplicate_rows():
    data = pd.DataFrame({"x": [1, 1, None]})
    report = explore_dataset(data)
    assert report["shape"] == (3, 1)
    assert report["duplicate_rows"] == 1
    assert report["missing_values"]["x"] == 1


def test_missing_customer_id_is_not_accepted_by_cleaning(full_mapping):
    mapping = ColumnMapping("customer", "invoice", "date", "amount")
    with pytest.raises(ValueError, match="missing"):
        clean_transactions(pd.DataFrame({"customer": [1]}), mapping)
